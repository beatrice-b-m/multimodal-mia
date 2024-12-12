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


class ImageCaptioner(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, tokenizer: nn.Module, freeze_encoder: bool = True):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.tokenizer = tokenizer

        if freeze_encoder:
            self.encoder.requires_grad_ = False

        self.device = None
        
    def tokenize(self, captions, **kwargs):
        return self.tokenizer(captions, return_tensors="pt", padding=True, truncation=False, **kwargs)

    def get_embeddings(self, images):
        z, _, _ = self.encoder(images)
        return z

    def _get_caption(self, image_embedding, max_length: int = 40):
        # prepare initial input and attention mask for the decoder
        input_ids = torch.tensor([[self.tokenizer.bos_token_id]]).to(self.device)
        
        # custom generation loop so we can use image embeddings
        generated_tokens = input_ids
        for _ in range(max_length - 1):
            outputs = self.decoder(
                input_ids=generated_tokens,
                # attention_mask=tokens['attention_mask'],
                encoder_hidden_states=image_embedding,
                use_cache=True,
            )
            # get output logits then take the most likely
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            generated_tokens = torch.cat((generated_tokens, next_token), dim=1)
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        
        return generated_tokens
    
    def generate_caption(self, image, skip_special_tokens: bool = True, n_beams: int = 1):
        if self.device is None:
            self.device = next(self.parameters()).device
        
        # add batch dim
        image = image.unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_embedding = self.get_embeddings(image).unsqueeze(1)

            generated_tokens = self._get_caption(image_embedding)

            generated_caption = self.tokenizer.decode(
                generated_tokens.detach().cpu()[0],
                skip_special_tokens=skip_special_tokens
            )

        return generated_caption

    def forward(self, images, tokens):
        embeddings = self.get_embeddings(images)

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

