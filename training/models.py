# import torch
import torch
from torch import nn
# from torchvision import models
from training.vae import VariationalAutoEncoder
from training.sepresnet.sepresnet import SepResNet18, SepResNet50
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

def get_srn18_vae():
    return VariationalAutoEncoder(SepResNet18(num_classes=None))


def get_srn50_vae():
    return VariationalAutoEncoder(SepResNet50(num_classes=None))


def get_distilgpt2_srn18_vae(param_dict: dict):
    # load autoencoder and extract encoder segment
    autoencoder = get_srn18_vae()
    ae_weights = param_dict.get('autoencoder_weights', None)
    if ae_weights is not None:
        autoencoder.load_state_dict(torch.load(ae_weights, weights_only=True))
    encoder = autoencoder.encoder

    # load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_bos_token = True

    # load decoder
    decoder = GPT2LMHeadModel(
        GPT2Config(
            name_or_path="distilgpt2",
            torch_dtype=torch.float16 if param_dict['mixed_precision'] else torch.float32,
            add_cross_attention=True,
            cross_attention_hidden_size=None
        )
    )

    # assemble captioner
    model = ImageCaptioner(
        encoder=encoder,
        decoder=decoder,
        tokenizer=tokenizer,
        freeze_encoder=param_dict['freeze_encoder']
    )

    # load pretrained captioner weights if they've been specified
    captioner_weights = param_dict.get('captioner_weights', None)
    if captioner_weights is not None:
        print(f'loading captioner weights from: {captioner_weights}')
        model.load_state_dict(torch.load(captioner_weights, weights_only=True))

    return model


from typing import Optional
from typing import Self # python 3.11+
import math

class BeamStep:
    def __init__(self, id: int, prob: float, parent: Optional[Self]) -> None:
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

    def __len__(self):
        return len(self.sequence)

    def __repr__(self) -> str:
        return f"BeamStep(sequence={self.sequence}, logprob={self.sequence_logprob:.3f})"

class ImageCaptioner(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, tokenizer: nn.Module, freeze_encoder: bool = True):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.tokenizer = tokenizer

        if freeze_encoder:
            self.encoder.requires_grad_ = False # type: ignore

        self.device = None
        
    def tokenize(self, captions, **kwargs):
        return self.tokenizer(captions, return_tensors="pt", padding=True, truncation=False, **kwargs)

    def _get_embeddings(self, images):
        z, _, _ = self.encoder(images)
        return z

    def _get_caption(self, image_embedding, max_length: int = 20, beam_width: int = 2, batch_size: int = 128):
        # create an initial beam step object (with the bos token id)
        # and add it to the initial iteration step
        init_step = BeamStep(
            id=self.tokenizer.bos_token_id,
            prob=1.0,
            parent=None
        )
        step_queue: list[BeamStep] = [init_step]
        sequence_list: list[BeamStep] = []
        
        # on each iteration (up to the max allowed sequence length)
        # iterate over the beams in the iteration beam_list
        for _ in range(max_length):
            # if the current step queue is empty break early
            if not len(step_queue):
                break
            
            # our step_queue contains sequences the list of sequences to evaluate next
            # on our first iteration it just contains our BOS token
            # and is then overwritten by our filled next_step_queue at the end of the iteration
            
            # evaluate queue in batches
            queue_inputs = torch.tensor([s.sequence for s in step_queue])
            queue_inputs = torch.tensor_split(queue_inputs, (len(queue_inputs)//batch_size)+1, dim=0)
    
            # iterate over batches and concatenate outputs in queue_logits
            queue_logits = []
            for queue_slice in queue_inputs:
                queue_logits.append(
                    self.decoder(
                        input_ids=queue_slice.to(self.device),
                        encoder_hidden_states=image_embedding,
                        use_cache=True, # should we?
                    ).logits[:, -1, :].cpu().detach()
                )
            queue_probs = torch.nn.functional.softmax(torch.vstack(queue_logits), dim=-1)
            queue_top_probs, queue_top_ids = torch.topk(queue_probs, k=beam_width, dim=-1)                    

            # add all possible steps to the candidate list
            candidate_list = []
            for i, step_object in enumerate(step_queue):
                for token_id, token_prob in zip(queue_top_ids[i, :].tolist(), queue_top_probs[i, :].tolist()):
                    candidate_list.append(
                        BeamStep(
                            id=token_id,
                            prob=token_prob,
                            parent=step_object,
                        )
                    )

            # prune candidates
            candidate_list = sorted(candidate_list, key=lambda s: s.sequence_logprob, reverse=True)[:beam_width]

            # clear previous step queue and choose to add current candidates to the
            # completed sequences list or next step queue
            step_queue = []
            for candidate_step in candidate_list:
                if candidate_step.id == self.tokenizer.eos_token_id:
                    sequence_list.append(candidate_step)
                else:
                    step_queue.append(candidate_step)
        
        # return generated_tokens
        return sequence_list
    
    def generate_caption(self, image, skip_special_tokens: bool = True, **kwargs):
        if self.device is None:
            self.device = next(self.parameters()).device
        
        # add batch dim
        image = image.unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_embedding = self._get_embeddings(image).unsqueeze(1)
            sequence_list = self._get_caption(image_embedding, **kwargs)

        # select the sequence with the highest probability
        best_sequence = max(sequence_list, key=lambda s: s.sequence_logprob)
        best_tokens = self.tokenizer.decode(best_sequence.sequence, skip_special_tokens=skip_special_tokens)
    
        return {"ids": best_sequence.sequence, "tokens": best_tokens}

    def forward(self, images, tokens):
        embeddings = self._get_embeddings(images)

        outputs = self.decoder(
            input_ids=tokens['input_ids'],
            attention_mask=tokens['attention_mask'],
            labels=tokens['input_ids'],
            encoder_hidden_states=embeddings.unsqueeze(1), # add sequence length dim 1 
        )
        return outputs


def get_model_dict():
    # this is only a function so I can easily import
    # it into the eval script
    return { 
        "srn18_vae": get_srn18_vae,
        "srn50_vae": get_srn50_vae,
        "distilgpt2_srn18_vae": get_distilgpt2_srn18_vae,
    }


if __name__ == "__main__":
    from training.interface import handle_params
    from torchinfo import summary
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # get the model dict
    model_dict = get_model_dict()

    # let user select the model
    param_dict = handle_params(
        {"model": {
            "argname": "m",
            "dtype": "str",
            "choices": list(model_dict.keys()),
            "default": None,
        }}, 
        confirm=False
    )
    
    # get the associated function
    model_func = model_dict[param_dict["model"]]

    model = model_func(weights=False)
    
    print(summary(model))

