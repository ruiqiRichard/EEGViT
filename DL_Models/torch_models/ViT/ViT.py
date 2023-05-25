from DL_Models.torch_models.ConvNetTorch import ConvNet
import torch.nn as nn
from DL_Models.torch_models.BaseNetTorch import BaseNet
from DL_Models.torch_models.Modules import Pad_Conv, Pad_Pool
import torch
import transformers
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

class ViT(BaseNet):
    """
    The CNN is one of the simplest classifiers. It implements the class ConvNet, which is made of modules with a specific depth.
    """
    def __init__(self, loss, model_number, batch_size, input_shape, output_shape, epochs=2, verbose=True):
        """
        nb_features: specifies number of channels before the output layer 
        """

        super().__init__(loss=loss, input_shape=input_shape, output_shape=output_shape, epochs=epochs, verbose=verbose,
                            model_number=model_number)
        model_name = "google/vit-base-patch16-224"
        config = transformers.ViTConfig.from_pretrained(model_name)
        config.update({'num_channels': 129})
        self.model = transformers.ViTForImageClassification.from_pretrained(model_name, config=config, ignore_mismatched_sizes=True)
        self.model.vit.embeddings.patch_embeddings.projection = torch.nn.Conv2d(129,768, kernel_size=(16, 16), stride=(16, 16))
        self.model.classifier = torch.nn.Linear(768,2, bias=True)
        print(sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        # for name, param in self.model.named_parameters():
        #     if "vit.embeddings.patch_embeddings" not in name:
        #         param.requires_grad = False
        

    def _module(self):
        return self.model
    
    def get_nb_features_output_layer(self):
        """
        Return number of features passed into the output layer of the network 
        nb.features has to be defined in a model implementing ConvNet
        """
        return 1000
    
    def forward(self, x):
        """
        Implements the forward pass of the network
        Modules defined in a class implementing ConvNet are stacked and shortcut connections are used if specified. 
        """
        # print(x.shape)
        features = self.model(x)
        # print(features.logits)
        # output = self.output_layer(features.logits) # Defined in BaseNet
        return features.logits