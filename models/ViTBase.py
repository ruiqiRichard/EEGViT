import torch
import transformers
from transformers import ViTModel
import torch
from torch import nn
import transformers

class ViTBase(nn.Module):
    def __init__(self):
        super().__init__()
        config = transformers.ViTConfig(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            initializer_range=0.02,
            num_channels=1,
            image_size=(129,500),
            patch_size=(8,35)
        )
        model = ViTModel(config)
        model.embeddings.patch_embeddings.projection = torch.nn.Conv2d(1, 768, kernel_size=(8, 36), stride=(8, 36), padding=(0,2))
        model.pooler.activation = torch.nn.Sequential(torch.nn.Dropout(p=0.1),
                                                    torch.nn.Linear(768,2,bias=True))
    def forward(self,x):
        x=self.model(x).pooler_output
        return x