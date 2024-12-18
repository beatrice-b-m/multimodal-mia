from torch import nn
from .generation import CaptionGenerator

class ImageCaptioner(CaptionGenerator):
    max_length = 30
    embed_dim = 768
    beam_width: int = 2

    image_size = 128
    image_channels = 3
    def __init__(self, encoder: nn.Module, decoder: nn.Module, tokenizer: nn.Module, freeze_encoder: bool = True):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.tokenizer = tokenizer

        if freeze_encoder:
            self.encoder.requires_grad_ = False # type: ignore

        self.device = None
        self.sleep_time: float = 5.0

        # initialize params for batch generations
        self._initialize_batch_gen_attrs()
        
    def tokenize(self, captions, **kwargs):
        return self.tokenizer(captions, return_tensors="pt", padding=True, truncation=False, **kwargs)

    def _get_embeddings(self, images):
        z, _, _ = self.encoder(images)
        return z

    def forward(self, images, tokens):
        embeddings = self._get_embeddings(images)

        outputs = self.decoder(
            input_ids=tokens['input_ids'],
            attention_mask=tokens['attention_mask'],
            labels=tokens['input_ids'],
            encoder_hidden_states=embeddings.unsqueeze(1), # add sequence length dim 1 
        )
        return outputs