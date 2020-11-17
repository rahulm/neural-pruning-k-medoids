"""
Parametrized, generic fully connected model for classification tasks.

The class is generically named "Model" to make it easier to dynamically import
a model based on the Python module name, without knowing the actual class name.
"""

from collections import OrderedDict
from typing import Any, Sequence, Text

import numpy as np
import torch
from torch import nn


class Model(nn.Module):
    def __init__(
        self, input_shape: Sequence, layers: Sequence[int], output_dim: int,
    ) -> None:
        super().__init__()
        self.input_dim: int = np.prod(input_shape)
        prev_layer_dim: int = self.input_dim

        layers_od: Any = OrderedDict()
        layers_od["flatten"] = nn.Flatten()

        for layer_i, layer_dim in enumerate(layers):
            layers_od["fc_{}".format(layer_i)] = nn.Linear(
                in_features=prev_layer_dim, out_features=layer_dim, bias=True
            )
            layers_od["relu_{}".format(layer_i)] = nn.ReLU(inplace=True)
            prev_layer_dim = layer_dim

        layers_od["fc_output"] = nn.Linear(
            in_features=prev_layer_dim, out_features=output_dim, bias=True
        )

        # TODO: Maybe create a base class that requires a list or sequential
        #       module that explicitly orders the layers, such that only that
        #       list or sequential module is run in the forward pass.
        # For now, explicitly use sequential_module in every NN for pruning.
        self.sequential_module = nn.Sequential(layers_od)

    def forward(self, x):
        return self.sequential_module(x)
