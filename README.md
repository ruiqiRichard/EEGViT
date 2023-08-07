## EEG-Vision Transformer (EEGViT)

Accepted KDD 2023: https://arxiv.org/pdf/2308.00454.pdf
## Overview
EEGViT is a hybrid Vision Transformer (ViT) incorporated with Depthwise Convolution in patch embedding layers. This work is based on 
Dosovitskiy, et al.'s ["An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"](https://arxiv.org/abs/2010.11929). After finetuning EEGViT pretrained on ImageNet, it achieves a considerable improvement over the SOTA on the Absolute Position task in EEGEyeNet dataset.

This repository consists of four models: ViT pretrained and non-pretrained; EEGViT pretrained and non-pretrained. The pretrained weights of ViT layers are loaded from [huggingface.co](https://huggingface.co/docs/transformers/model_doc/vit).

## Dataset download
Download data for EEGEyeNet absolute position task
```bash
wget -O "./dataset/Position_task_with_dots_synchronised_min.npz" "https://osf.io/download/ge87t/"
```
For more details about EEGEyeNet dataset, please refer to ["EEGEyeNet: a Simultaneous Electroencephalography and Eye-tracking Dataset and Benchmark for Eye Movement Prediction"](https://arxiv.org/abs/2111.05100) and [OSF repository](https://osf.io/ktv7m/)

## Installation

### Requirements

First install the general_requirements.txt

```bash
pip3 install -r general_requirements.txt 
```

### Pytorch Requirements

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

For other installation details and different cuda versions, visit [pytorch.org](https://pytorch.org/get-started/locally/).
