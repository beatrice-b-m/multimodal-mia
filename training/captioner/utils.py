import threading
import torch
import math
import time
from typing import Optional, Self
import numpy as np
    

class ThreadSafeBatch:
    """
    **subthread example:**
    ```
    # add data to the batch and retrieve the relevant index and an airlock object
    idx, airlock = self.batch.add({'images': image_tensor})

    # wait at the airlock in_barrier until the main thread has posted the results
    airlock.in_barrier.wait()

    # access the output data at the index, then wait at the airlock out_barrier to indicate 
    # the thread can proceed
    output = self.batch.output[idx]
    airlock.out_barrier.wait()
    ```
    """
    def __init__(self, name: str, key_dict: dict):
        self.name: str = name # encoder/decoder
        self.key_dict: dict[str, dict] = key_dict
        self._lock: threading.RLock = threading.RLock()
        self.capacity: Optional[int] = None
        self.init_batch()

    def __len__(self):
        return self._fill_idx
    
    def init_batch(self):
        # initialize airlock barrier with init count of 1 (for this object)
        self.batch_airlock: AirlockBarrier = AirlockBarrier(initial_count=1)
        self.outputs = torch.empty(size=(1,))
        self._fill_idx: int = 0
        self._dict: dict[str, torch.Tensor] = {k:torch.empty(size=v['shape'], dtype=v['dtype']) for k,v in self.key_dict.items()}

        # reset the time
        self.reset_time()

    def add(self, inputs: dict[str, torch.Tensor]):
        with self._lock: # threads will block here if trying to add data during model execution
            fill_idx = self.request_fill_index()
            # print(f"{fill_idx = }")
            for tensor_key, input_tensor in inputs.items():
                self._dict[tensor_key][fill_idx] = input_tensor
            # give threads a copy of the fill index and a batch_airlock object which will synchronize exit/entry
            self.batch_airlock.subscribe()
            return fill_idx, self.batch_airlock

    def request_fill_index(self):
        with self._lock:
            # acquire lock, then assign the current fill_idx and increment for the next
            last_idx = self._fill_idx
            self._fill_idx = last_idx + 1
            return last_idx
        
    def request_batch(self):
        # acquire lock until the batch outputs have been registered to prevent
        # threads from trying to write to the batch during model inference
        self._lock.acquire()

        # return all filled elements of the batch
        out_dict = {k:v[:self._fill_idx] for k,v in self._dict.items()}
        return out_dict
    
    def register_batch(self, outputs):
        # register model outputs for last batch (should be detached and transferred to cpu by now)
        self.outputs = outputs
        # wait at the in_barrier to allow subscribed threads to access the batch outputs
        # print(f"{self.outputs.shape = }")
        self.batch_airlock.in_barrier.wait()
        # then wait at the out_barrier to prevent the main thread from discarding output 
        # data before subthreads can access it
        self.batch_airlock.out_barrier.wait()
        # initialize the batch then finally release the lock
        # print('re-init')
        self.init_batch()
        self._lock.release()

    def reset_time(self):
        self._start_time = time.time()

    def _estimate_capacity(self, max_users: int, beam_width: int, pbar_dict: dict, utilization: float = 0.8):
        # check the progress bar dict to try to estimate how many workers might need to 
        # add images to this batch
        if self.name == "encoder":
            search_list = ["encoding image", "requesting idx"]
        elif self.name == "decoder":
            search_list = ["generating_caption"]

        n_users = sum([1 for pbar in list(pbar_dict.values()) if parse_postfix(pbar.postfix).get('status', 'None') in search_list])
        return math.ceil(n_users * beam_width * utilization) if n_users>1 else math.ceil(0.4*max_users*beam_width*utilization)

    def check(self, max_users: int, beam_width: int, pbar_dict: dict, timeout: float):
        if self.capacity is not None:
            batch_capacity = self.capacity
            return True if self._fill_idx >= batch_capacity else False
        
        else:
            batch_capacity = self._estimate_capacity(max_users, beam_width, pbar_dict)

            # if the batch is empty, reset the time
            if self._fill_idx == 0:
                self.reset_time()
            # if time has been exceeded and the batch isn't empty
            elif ((time.time() - self._start_time) > timeout):
                return True
            # if the batch > the estimated batch capacity
            elif (self._fill_idx >= batch_capacity):
                return True
        
            return False


class AirlockBarrier:
    """
    provides a two stage barrier (by wrapping 2 DynamicBarriers
    subscribed threads both reach and exit a point synchronously
    """
    def __init__(self, initial_count: int = 1):
        self.in_barrier = DynamicBarrier(initial_count=initial_count)
        self.out_barrier = DynamicBarrier(initial_count=initial_count)

    def subscribe(self):
        self.in_barrier.subscribe()
        self.out_barrier.subscribe()

    def wait_in(self):
        self.in_barrier.wait()

    def wait_out(self):
        self.out_barrier.wait()


class DynamicBarrier:
    def __init__(self, initial_count: int):
        # only able to be used once (then must be reinstantiated)
        self.active_threads: int = initial_count
        self.waiting_threads: int = 0
        self.condition = threading.Condition()

    def subscribe(self):
        """A thread subscribes to the barrier."""
        with self.condition:
            self.active_threads += 1

    def wait(self):
        """Wait for all active threads to reach the barrier."""
        with self.condition:
            self.waiting_threads += 1
            if self.waiting_threads == self.active_threads:
                # All threads have reached the barrier, release them
                self.waiting_threads = 0
                self.condition.notify_all()
            else:
                # Wait until all threads reach the barrier
                self.condition.wait()


class BeamStep:
    def __init__(self, id: int, prob: float, parent: Optional[Self], alpha: float = 0.2) -> None:
        # probability is stored as log probs so the sum of log sequence probs is equivalent to
        # the log of the product of raw sequence probs
        self.id: int = id
        self.logprob: float = math.log(prob + 1e-9)

        self.parent: Optional[Self] = parent

        if self.parent is None: # if parent does not exist
            self.sequence: list[int] = [self.id]
            self.sequence_logprob: float = self.logprob

        else: # if parent does exist
            self.sequence: list[int] = [*self.parent.sequence, self.id]
            self.sequence_logprob: float = self.parent.sequence_logprob + self.logprob

        self.sequence_norm_logprob: float = self.sequence_logprob / 1.0#(len(self.sequence) ** alpha)

    def __len__(self):
        return len(self.sequence)

    def __repr__(self) -> str:
        return f"BeamStep(sequence={self.sequence}, norm_logprob={self.sequence_logprob:.3f})"


class SharedDict:
    def __init__(self):
        self._dict = {}
        self._lock = threading.RLock()  # Reentrant lock for thread-safe access

    def receive(self, key):
        # Retrieve and remove the value associated with the given key.
        with self._lock:
            item_query = self._dict.pop(key, None)  # Atomically retrieve and remove
            return item_query

    def __getitem__(self, key):
        with self._lock:
            return self._dict.get(key, None)

    def __setitem__(self, key, newvalue):
        with self._lock:
            self._dict[key] = newvalue

    def __repr__(self):
        with self._lock:
            return f"SharedDict({dict(self._dict)})"
        


def parse_postfix(postfix: str):
    if postfix is not None:
        return dict([chunk.strip().split("=") for chunk in postfix.split(",")])
    else:
        return dict()
    

def pad_tokens(id_list: list[int], max_length: int = 20):
    # count tokens in original token id list
    n_tokens = len(id_list)

    if n_tokens > max_length:
        id_list = id_list[:max_length]
        n_tokens = max_length

    # get arrays of zeros to store the new id/attention arrays
    id_arr = np.zeros((max_length,), dtype=np.int32)
    att_mask = np.zeros((max_length,), dtype=np.int32)

    # map id_list onto new id_arr and set attention mask
    id_arr[:n_tokens] = id_list
    att_mask[:n_tokens] = 1
    return id_arr, att_mask