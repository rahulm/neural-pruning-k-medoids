"""Some utils for training, checkpointing, etc. of models."""

import csv
import os
from typing import Callable, Dict, List, Optional, Text

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision

FILE_NAME_CSV: Text = "vals.csv"
FILE_NAME_PLOT: Text = "plot.png"


class StatCounter:
    def __init__(self, default_save_params: Optional[Dict] = None) -> None:
        self._counter: List = []
        self._default_save_params: Optional[Dict] = default_save_params

    def add(self, val) -> None:
        self._counter.append(val)

    def save_default(self, **kwargs) -> None:
        save_params: Dict = {}
        if self._default_save_params:
            save_params.update(self._default_save_params)
        save_params.update(kwargs)
        self.save(**save_params)

    def save(
        self,
        folder_path: Text,
        file_prefix: Text = "",
        xlabel: Text = "iteration",
        ylabel: Text = "loss",
        title_prefix: Text = "",
        index_offset: int = 0,
        index_multiplier: int = 1,
    ) -> None:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Make index (x axis).
        indices: List[int] = [
            (index_multiplier * (i + index_offset))
            for i in range(len(self._counter))
        ]

        file_prefix = "{}-".format(file_prefix) if file_prefix else ""

        # Save CSV.
        with open(
            os.path.join(
                folder_path, "{}{}".format(file_prefix, FILE_NAME_CSV)
            ),
            "w",
            newline="",
        ) as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([xlabel, ylabel])
            for ind, val in zip(indices, self._counter):
                writer.writerow([ind, val])

        # Save plot.
        title: Text = (
            "{}{}".format(
                "" if (not title_prefix) else "{} - ".format(title_prefix),
                "{} per {}".format(ylabel.capitalize(), xlabel.capitalize()),
            )
        )
        plt.figure()
        plt.plot(indices, self._counter)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.savefig(
            os.path.join(
                folder_path, "{}{}".format(file_prefix, FILE_NAME_PLOT)
            )
        )


DATA_FOLDER_PATH: Text = os.path.join("data", "pytorch")
DATASET_FUNCTIONS: Dict[Text, Callable] = {
    "mnist": torchvision.datasets.MNIST,
    "cifar10": torchvision.datasets.CIFAR10,
}
DATASET_TRANSFORMS: Dict = {
    "mnist": torchvision.transforms.Compose(  # From https://github.com/pytorch/examples/blob/master/mnist/main.py
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ]
    ),
    "cifar10": torchvision.transforms.Compose(  # From https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    ),
}
