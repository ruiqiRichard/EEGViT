import torch
import transformers
from transformers import ViTModel
import torch
from torch import nn
import transformers

class EEGViT_raw(nn.Module):
    def __init__(self, ViTLayers):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=256,
            kernel_size=(1, 36),
            stride=(1, 36),
            padding=(0,2),
            bias=False
        )
        self.batchnorm1 = nn.BatchNorm2d(256, False)
        config = transformers.ViTConfig(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            initializer_range=0.02,
            num_channels=256,
            image_size=(129,14),
            patch_size=(8,1)
        )
        model = ViTModel(config)
        model.embeddings.patch_embeddings.projection = torch.nn.Conv2d(256, 768, kernel_size=(8, 1), stride=(8, 1), padding=(0,0), groups=256)
        model.pooler.activation = torch.nn.Sequential(torch.nn.Dropout(p=0.1),
                                                       torch.nn.Linear(768,2,bias=True))
        self.ViT = model
        
    def forward(self,x):
        x=self.conv1(x)
        x=self.batchnorm1(x)
        x=self.ViT(x).pooler_output
        
        return x