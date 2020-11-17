"""
Parametrized model with 2 hidden fully connected layers.
Can be used to recreate the LeNet-300-100 model (approximately).

The class is generically named "Model" to make it easier to dynamically import
a model based on the Python module name, without knowing the actual class name.

TODO: Train this model with mnist.
TODO: Make generic fn to prune (remove) nodes (see notes) + construct a new model.
        Probably need to work with some type of PyTorch "mask" for each FC layer.
        Or, directly modify the weights, then remove all nodes that have only
        out weights of zero (or in weights of zero?).
TODO: Experiment with some possible similarity matrices.

"""

from typing import Sequence, Text

import numpy as np
import torch
from torch import nn


class Model(nn.Module):
    ARCHITECTURE_NAME: Text = "fc_2"

    def __init__(
        self,
        input_shape: Sequence,
        layer_1_dim: int,
        layer_2_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.input_dim: int = np.prod(input_shape)
        self.flatten = nn.Flatten()

        self.fc_1 = nn.Linear(
            in_features=self.input_dim, out_features=layer_1_dim, bias=True
        )
        self.relu_1 = nn.ReLU(inplace=True)

        self.fc_2 = nn.Linear(
            in_features=layer_1_dim, out_features=layer_2_dim, bias=True
        )
        self.relu_2 = nn.ReLU(inplace=True)

        self.fc_output = nn.Linear(
            in_features=layer_2_dim, out_features=output_dim, bias=True
        )

        # TODO: Maybe create a base class that requires a list or sequential
        #       module that explicitly orders the layers, such that only that
        #       list or sequential module is run in the forward pass.
        # For now, explicitly use sequential_module in every NN for pruning.
        self.sequential_module = nn.Sequential(
            self.flatten,
            self.fc_1,
            self.relu_1,
            self.fc_2,
            self.relu_2,
            self.fc_output,
        )

    def forward(self, x):
        return self.sequential_module(x)
