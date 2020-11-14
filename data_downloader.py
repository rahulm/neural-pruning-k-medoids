"""Downloads data as requested into the data/ folder."""
import argparse
import os
from typing import Callable, Dict, Text

import torch
import torchvision

DATASET_FUNCTIONS: Dict[Text, Callable] = {
    "mnist": torchvision.datasets.MNIST,
    "cifar10": torchvision.datasets.CIFAR10,
}


def get_args():
    parser = argparse.ArgumentParser(
        description="Downloads data as requested into the data/ folder."
    )

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Name of dataset to download",
        choices=DATASET_FUNCTIONS.keys(),
    )

    return parser.parse_args()


def main() -> None:
    args = get_args()
    print(args)

    data_name: Text = args.data
    data_folder: Text = os.path.join("data", "pytorch")
    DATASET_FUNCTIONS[data_name](data_folder, download=True)


if __name__ == "__main__":
    main()
