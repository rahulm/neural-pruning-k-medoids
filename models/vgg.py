"""
Wraps the official PyTorch VGG implementation for pruning.

The class is generically named "Model" to make it easier to dynamically import
a model based on the Python module name, without knowing the actual class name.
"""

from typing import Callable, List, Text

import torch
import torchvision
from torch import nn


class Model(nn.Module):
    ARCHITECTURE_NAME: Text = "vgg"

    def __init__(
        self,
        vgg_version: Text,
        num_classes: int,
        pretrained_imagenet: bool = False,
        **kwargs
    ) -> None:
        super().__init__()

        self.vgg_version: Text = vgg_version
        self.num_classes: int = num_classes

        self.pytorch_model: torchvision.models.vgg.VGG = getattr(
            torchvision.models.vgg, self.vgg_version
        )(
            pretrained=pretrained_imagenet,
            num_classes=self.num_classes,
            **kwargs
        )

        self.prunable_parameters_ordered: List[nn.Module] = [
            layer
            for layer in (
                list(self.pytorch_model.features)
                + list(self.pytorch_model.classifier)
            )
            if (
                isinstance(layer, nn.Linear)
                or isinstance(layer, nn.Conv2d)
                or isinstance(layer, nn.BatchNorm2d)
            )
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pytorch_model(x)
