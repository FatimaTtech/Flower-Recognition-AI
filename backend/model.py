"""
ResNet-18 Model Architecture Definition for BloomAI Flower Classification.
Matches the exact checkpoint trained with 5 classes and custom head.
"""
import torch
import torch.nn as nn
from torchvision import models


def create_flower_resnet18(num_classes: int = 5) -> nn.Module:
    """
    Reconstructs the ResNet-18 model architecture matching the trained checkpoint.

    Architecture:
      - Backbone: Standard ResNet-18 (512 feature output from layer4/avgpool)
      - Head:
          0: Linear(512 -> 256)
          1: ReLU(inplace=True)
          2: Dropout(p=0.5)
          3: Linear(256 -> num_classes)
    """
    # Use weights=None so we don't attempt to download ImageNet weights at runtime
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(256, num_classes)
    )
    return model

