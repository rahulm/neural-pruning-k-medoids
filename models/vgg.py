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

    @property
    def ordered_unpacking(self) -> List[nn.Module]:
        return (
            list(self.pytorch_model.features)
            + [self.pytorch_model.avgpool]
            + list(self.pytorch_model.classifier)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pytorch_model(x)
