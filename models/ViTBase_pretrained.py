import torch
import transformers
from transformers import ViTModel
import torch
from torch import nn
import transformers

class ViTBase_pretrained(nn.Module):
    def __init__(self):
        super().__init__()
        model_name = "google/vit-base-patch16-224"
        config = transformers.ViTConfig.from_pretrained(model_name)
        config.update({'num_channels': 1})
        config.update({'image_size': (129,500)})
        config.update({'patch_size': (8,35)})

        model = transformers.ViTForImageClassification.from_pretrained(model_name, config=config, ignore_mismatched_sizes=True)
        model.vit.embeddings.patch_embeddings.projection = nn.Sequential(torch.nn.Conv2d(1, 768, kernel_size=(8, 36), stride=(8, 36), padding=(0,2)),
                                                                        nn.BatchNorm2d(768))
        model.classifier=torch.nn.Sequential(torch.nn.Linear(768,1000,bias=True),
                                     torch.nn.Dropout(p=0.1),
                                     torch.nn.Linear(1000,2,bias=True))
        self.ViT = model
        
    def forward(self,x):
        x=self.ViT(x).logits
        return x