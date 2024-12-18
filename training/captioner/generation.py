from torch import nn
import torch
import threading
from typing import Optional, Any
from tqdm.auto import tqdm
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from .utils import SharedDict, BeamStep, ThreadSafeBatch, pad_tokens


class CaptionGenerator(nn.Module):
    beam_width: int = 0
    def __init__(self):
        super().__init__()
        pass

    def generate_caption(self, image):
        # used to generate a caption for 1 image
        if self.device is None:
            self.device = next(self.parameters()).device

        with torch.no_grad():
            image_embedding = self._get_embeddings(**{"images": image.unsqueeze(0).to(self.device)}).cpu().detach().unsqueeze(1)

            init_step = BeamStep(
                id=self.tokenizer.bos_token_id,
                prob=1.0,
                parent=None
            )
            step_list: list[BeamStep] = [init_step]
            sequence_list: list[BeamStep] = []

            best_p = -np.inf

            self._get_batch_containers(batch_size=self.beam_width, batched=True)

            # start the batch supervisor threads
            spv_kill_event = threading.Event()
            batch_supervisor = threading.Thread(target=self._batch_gen_supervisor, args=(1, 5.0, spv_kill_event), daemon=True)
            batch_supervisor.start()
            
            # on each iteration (up to the max allowed sequence length)
            # iterate over the beams in the iteration beam_list
            for len_i in range(self.max_length-1):
                try:
                    self._batch_container_dict['decoder'].capacity = len(step_list)

                    # use ThreadPoolExecutor for concurrent step handling
                    with ThreadPoolExecutor() as pool:
                        candidate_list = list(pool.map(lambda step: self._handle_step(step, image_embedding), step_list))
                except KeyboardInterrupt:
                    print(f"{len(step_list) = }")
                    print(f"{self._batch_container_dict['decoder'].capacity = }")
                    print(f"{len(self._batch_container_dict) = }")

                # flatten the list of candidate lists
                candidate_list = [candidate for sublist in candidate_list for candidate in sublist]
                    
                # prune candidates
                candidate_list = sorted(candidate_list, key=lambda s: s.sequence_norm_logprob, reverse=True)[:self.beam_width]
                candidate_list = [c for c in candidate_list if not c.sequence_norm_logprob < best_p] # exclude candidates that are worse than our best so far

                # clear previous step queue and choose to add current candidates to the
                # completed sequences list or next step queue
                step_list = []
                for candidate_step in candidate_list:
                    if candidate_step.id == self.tokenizer.eos_token_id:
                        sequence_list.append(candidate_step)
                        best_p = max(best_p, candidate_step.sequence_norm_logprob)
                    else:
                        step_list.append(candidate_step)
                
                # if we have no queued steps and at least one completed sequence, break
                if (len(step_list) == 0) and (len(sequence_list) > 0):
                    # print('no items left and >0 solutions found, breaking')
                    break

            # stop the batch processor thread
            spv_kill_event.set()
            batch_supervisor.join()
            
            # if the max length has been reached with no complete sequences, take the 
            # step_list item with the best logprob and choose it as the sequence
            if len(sequence_list) == 0:
                final_candidate = BeamStep(
                    id=self.tokenizer.eos_token_id,
                    prob=1.0,
                    parent=max(candidate_list, key=lambda s: s.sequence_norm_logprob),
                )
                sequence_list.append(final_candidate)
            
            # return generated_tokens
            return sequence_list


    def generate_batch_captions(self, dataset, max_workers: Optional[int] = None, batch_size: int = 256, timeout: float = 1.0):
        # used to generate captions for a batch of images
        # derive max workers if left as None
        if max_workers is None:
            max_workers = batch_size // self.beam_width
        else:
            assert max_workers*self.beam_width <= batch_size, "batch size may be exceeded!"

        if self.device is None:
            self.device = next(self.parameters()).device

        # stores results for worker threads to retrieve
        self._result_dict = SharedDict()

        self._initialize_batch_gen_attrs()
        n_images = len(dataset)
        n_workers = min(max_workers, n_images)
        self._image_list = list(range(n_images))  # stores indexes for images in the dataset

        # get batch containers
        self._get_batch_containers(batch_size=batch_size, batched=True)

        # start the batch supervisor threads
        spv_kill_event = threading.Event()
        batch_supervisor = threading.Thread(target=self._batch_gen_supervisor, args=(n_workers, timeout, spv_kill_event), daemon=True)
        batch_supervisor.start()

        # start worker threads dynamically
        worker_list = []
        try:
            # create the main progress bar
            self._pbar_dict = {"main": tqdm(total=n_images, position=0, desc='main')}

            for worker_id in range(n_workers):  # Start with up to `max_workers`
                worker = threading.Thread(target=self._batch_gen_worker, args=(worker_id, dataset), daemon=True)
                worker_list.append(worker)
                # add worker progress bar to the pbar dict
                self._pbar_dict[str(worker_id)] = tqdm(
                    total=self.max_length,
                    position=worker_id+1,
                    desc=f"[w: {worker_id:>3}] [i: ?]",
                    postfix={"status": "starting up"},
                    bar_format='{desc} | {bar} | {postfix}',
                )

            for worker in worker_list:
                worker.start()

        finally:  # ensure all threads are joined
            for t in worker_list:
                t.join()

            # stop the batch processor thread
            spv_kill_event.set()
            batch_supervisor.join()
            # pbar_supervisor.join()

            # collect the outputs from the output queue
            output_list = self._output_list.copy()

            # finally wipe the shared objects
            # self._initialize_batch_gen_attrs()
            return output_list

    def _batch_gen_supervisor(self, n_workers: int, timeout: float, kill_event: threading.Event):
        print('\n\n\nstarted batch generation supervisor...')
        # get self.batch_container_dict
        # self._get_batch_containers(batch_size=batch_size, batched=batched)

        # loop to check the input queue and prep any ready batches for inference
        while True:
            if kill_event.is_set():
                return
            
            # check if either batch is ready
            for batch_name, batch_container in self._batch_container_dict.items():
                if batch_container.check(n_workers, self.beam_width, self._pbar_dict, timeout):
                    # request the batch, batch lock is held until .register_batch() is called [so be quick!! :) ]
                    batch_dict = batch_container.request_batch()
                    batch_outputs = self._run_batch_inference(batch_name, batch_dict)
                    batch_container.register_batch(batch_outputs)
                    # batch lock released

    def _run_batch_inference(self, batch_type: str, batch_dict: dict):
        # send batch dict tensors to the gpu
        batch_dict = {k:v.to(self.device) for k,v in batch_dict.items()}

        with torch.no_grad():
            if batch_type == "encoder":
                outputs = self._get_embeddings(**batch_dict).cpu().detach().unsqueeze(1) # add sequence dim

            elif batch_type == "decoder":
                outputs = self.decoder(**batch_dict).logits[:, -1, :].cpu().detach()
                outputs = torch.nn.functional.softmax(outputs, dim=-1)
                output_vals, output_idxs = torch.topk(outputs, k=self.beam_width, dim=-1)
                outputs = torch.cat([output_vals.unsqueeze(1), output_idxs.unsqueeze(1)], dim=1)

            return outputs

    def _get_batch_containers(self, batch_size: int, batched: bool):
        key_dict = {
            "decoder": {
                'input_ids': {'shape': (batch_size, self.max_length), 'dtype': torch.int32}, 
                'attention_mask': {'shape':(batch_size, self.max_length), 'dtype': torch.int32}, 
                'encoder_hidden_states': {'shape': (batch_size, 1, self.embed_dim), 'dtype': torch.float32},
            },
        }

        if batched:
            key_dict["encoder"] = {
                'images': {'shape': (batch_size, self.image_channels, self.image_size, self.image_size), 'dtype': torch.float32}
            }
        
        # save the batch container dict
        self._batch_container_dict = {k: ThreadSafeBatch(k, key_dict=v) for k,v in key_dict.items()}

    def _update_pbar(self, identifier: str, update_message: dict): #, pbar_dict: dict):
        for update_type, update_value in update_message.items():
            if update_type == 'image_idx':
                self._pbar_dict[identifier].set_description(f"[w: {identifier}] [i: {update_value}]", refresh=False)

            elif update_type == 'status':
                self._pbar_dict[identifier].set_postfix({'status': update_value}, refresh=False)
                if update_value == 'image processed':
                    # increment main progress bar when an image is processed
                    self._pbar_dict['main'].update(1)

            elif update_type == 'increment':
                self._pbar_dict[identifier].n = update_value
        
        self._pbar_dict[identifier].refresh()

    def _batch_gen_worker(self, worker_id, dataset):
        while True:
            self._update_pbar(str(worker_id), {"image_idx": "?", "status": "requesting idx", "increment": 0})
            # try to get an image index from the queue and break if it's empty
            try:
                img_idx = self._image_list.pop() # type: ignore
            except IndexError:
                self._update_pbar(str(worker_id), {"image_idx": img_idx, "status": "image queue empty"})
                break

            # extract the elements from the dataset
            # {"image": img, "index": idx, "label": label}
            idx_dict = dataset[img_idx]

            self._update_pbar(str(worker_id), {"image_idx": img_idx, "status": "encoding image"})
            # send image to the encoder queue and get the embedding response

            embedding = self._send_to_model_queue(
                destination='encoder',
                inputs={"images": idx_dict["image"].unsqueeze(0)}
            )

            # get the caption sequence list
            self._update_pbar(str(worker_id), {"status": "generating caption"})
            sequence_list = self._get_caption_batched(worker_id, embedding)
            # self._update_worker_status(_worker_bar, worker_id, status='caption generated')

            # select the sequence with the highest probability
            # try:
            best_sequence = max(sequence_list, key=lambda s: s.sequence_norm_logprob)
            best_tokens = self.tokenizer.decode(best_sequence.sequence, skip_special_tokens=True)
            
            # put the output for this image to the output queue then go to the next iteration
            self._output_list.append({"idx": img_idx, "ids": best_sequence.sequence, "tokens": best_tokens}) # type: ignore
            # self._update_worker_status(_worker_bar, worker_id, status='image processed')

    def _get_caption_batched(self, worker_id, image_embedding):
        # create an initial beam step object (with the bos token id)
        # and add it to the initial iteration step
        init_step = BeamStep(
            id=self.tokenizer.bos_token_id,
            prob=1.0,
            parent=None
        )
        step_list: list[BeamStep] = [init_step]
        sequence_list: list[BeamStep] = []

        best_p = -np.inf
        
        # on each iteration (up to the max allowed sequence length)
        # iterate over the beams in the iteration beam_list
        for len_i in range(self.max_length-1):
            self._update_pbar(str(worker_id), {"increment": len_i})            
            # Use ThreadPoolExecutor for concurrent step handling
            with ThreadPoolExecutor() as pool:
                candidate_list = list(pool.map(lambda step: self._handle_step(step, image_embedding), step_list))
            
            # flatten the list of candidate lists
            candidate_list = [candidate for sublist in candidate_list for candidate in sublist]
                
            # prune candidates
            candidate_list = sorted(candidate_list, key=lambda s: s.sequence_norm_logprob, reverse=True)[:self.beam_width]
            candidate_list = [c for c in candidate_list if not c.sequence_norm_logprob < best_p] # exclude candidates that are worse than our best so far

            # clear previous step queue and choose to add current candidates to the
            # completed sequences list or next step queue
            step_list = []
            for candidate_step in candidate_list:
                if candidate_step.id == self.tokenizer.eos_token_id:
                    sequence_list.append(candidate_step)
                    best_p = max(best_p, candidate_step.sequence_norm_logprob)
                else:
                    step_list.append(candidate_step)
            
            # if we have no queued steps and at least one completed sequence, break
            if (len(step_list) == 0) and (len(sequence_list) > 0):
                # print('no items left and >0 solutions found, breaking')
                break
        
        # if the max length has been reached with no complete sequences, take the 
        # step_list item with the best logprob and choose it as the sequence
        if len(sequence_list) == 0:
            final_candidate = BeamStep(
                id=self.tokenizer.eos_token_id,
                prob=1.0,
                parent=max(candidate_list, key=lambda s: s.sequence_norm_logprob),
            )
            sequence_list.append(final_candidate)
        
        # return generated_tokens
        self._update_pbar(str(worker_id), {"status": "image processed"})
        return sequence_list

    def _handle_step(self, step_object: BeamStep, image_embedding: torch.Tensor):
        # pad token id array and get the attention mask
        id_arr, att_mask = pad_tokens(step_object.sequence, self.max_length)

        # send model inputs to queue and get response
        outputs = self._send_to_model_queue(
            destination='decoder',
            inputs={
                "input_ids": torch.tensor(id_arr).unsqueeze(0), 
                "attention_mask": torch.tensor(att_mask).unsqueeze(0), 
                "encoder_hidden_states": image_embedding.unsqueeze(0)
            }
        )
        output_top_probs, output_top_ids = outputs

        # return all possible candidates
        return self._expand_candidates(step_object, output_top_ids.tolist(), output_top_probs.tolist())
    
    def _send_to_model_queue(self, destination: str, inputs: dict):
        # add data to the batch and retrieve the relevant index and an airlock object
        # thread will block here until it can acquire the batch lock
        idx, airlock = self._batch_container_dict[destination].add(inputs)

        # wait at the airlock in_barrier until the main thread has posted the results
        # TODO: add a compatible `with airlock.barrier` form
        airlock.in_barrier.wait()

        # access the output data at the index, then wait at the airlock out_barrier to indicate 
        # the thread can proceed
        try:
            outputs = self._batch_container_dict[destination].outputs[idx]

        except IndexError as e:
            print('caught IndexError:', e)
            print(f"{idx = }")
            print(f"outputs shape = {self._batch_container_dict[destination].outputs.shape}")
            outputs = None
            
        airlock.out_barrier.wait()
        return outputs

    def _expand_candidates(self, parent_step: BeamStep, top_id_list: list, top_prob_list: list):
        candidate_list = []
        # for token_id, token_prob in zip(queue_top_ids.tolist(), queue_top_probs[i, :].tolist()):
        for token_id, token_prob in zip(top_id_list, top_prob_list):
            candidate_list.append(
                BeamStep(
                    id=token_id,
                    prob=token_prob,
                    parent=parent_step,
                )
            )

        return candidate_list

    def _initialize_batch_gen_attrs(self):
        self._output_list: list[Any] = []
        self._image_list: list[int] = []
        # self._input_queue: queue.Queue = queue.Queue()
        self._result_dict: SharedDict = SharedDict()
        self._batch_container_dict: dict[str, ThreadSafeBatch] = {}
        self._pbar_dict: dict[str, Any] = {}