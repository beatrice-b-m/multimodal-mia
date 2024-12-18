# import torch
import torch
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

from .vae import VariationalAutoEncoder
from .sepresnet.sepresnet import SepResNet18, SepResNet50
from .captioner import ImageCaptioner

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

