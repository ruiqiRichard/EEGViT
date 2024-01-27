import torch
from transformers import ViTModel, ViTConfig
from torch import nn
from torch.nn import functional as F

class EEGViT_pretrained(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=256,
            kernel_size=(1, 36),
            stride=(1, 36),
            padding=(0,2),
            bias=False
        )
        self.conv2 = nn.Conv2d(
            in_channels=256,
            out_channels=512,  # Increased number of channels
            kernel_size=(1, 7),  # Smaller kernel size to capture more detailed features
            stride=(1, 1),
            padding=(0,1),
            bias=False
        )
        self.batchnorm1 = nn.BatchNorm2d(256)
        self.batchnorm2 = nn.BatchNorm2d(512)
        self.dropout = nn.Dropout(p=0.1)  # Added dropout for regularization

        # Define model_name and config as before
        model_name = "google/vit-base-patch16-224"
        config = ViTConfig.from_pretrained(model_name)
        config.update({'num_channels': 512})  # Updated to match new conv2 layer
        config.update({'image_size': (129,14)})
        config.update({'patch_size': (8,1)})

        self.model = ViTModel.from_pretrained(model_name, config=config, ignore_mismatched_sizes=True)
        self.model.embeddings.patch_embeddings.projection = nn.Conv2d(
            512, config.hidden_size, kernel_size=(8, 1), stride=(8, 1), padding=(0,0), groups=512
        )
        # Update classifier for regression
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(512, 1)  # Assuming a single continuous output for eye position
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.batchnorm1(x))
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = F.relu(self.batchnorm2(x))
        x = self.dropout(x)
        
        # Flatten and pass through ViT
        x = x.flatten(2)
        x = self.model(x).last_hidden_state
        
        # Classification head
        x
