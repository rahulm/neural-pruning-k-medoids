"""Evaluate model on the desired dataset."""

import os
from typing import Callable, Dict, Text, Tuple

import torch
import torch.nn.functional as F
import torchvision

from utils import model_config_utils, train_utils


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    torchdevice: torch.device,
) -> Tuple[float, float]:
    loss_sum: float = 0.0
    num_correct: int = 0

    model.eval()
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(torchdevice), target.to(torchdevice)
            output = model(data)
            loss_sum += F.cross_entropy(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            num_correct += pred.eq(target.view_as(pred)).sum().item()

    loss_ave: float = loss_sum / len(dataloader.dataset)
    accuracy: float = num_correct / len(dataloader.dataset)

    return accuracy, loss_ave


def evaluate_model_from_config(
    model_config: model_config_utils.ModelConfig,
    dataset_name: Text,
    batch_size: int,
    use_gpu: bool,
) -> Tuple[float, float]:
    torchdevice: torch.device = torch.device("cuda" if use_gpu else "cpu")

    # TODO: Support more models.
    model: torch.nn.Module
    if model_config.model_architecture == "fc_2":
        from models.fc_2 import Model

        # Maybe make the map_location dynamic.
        model = torch.load(model_config.model_path, map_location=torchdevice)
    else:
        print(
            "Architecture unsupported: {}".format(
                model_config.model_architecture
            )
        )

    dataloader = torch.utils.data.DataLoader(
        train_utils.DATASET_FUNCTIONS[dataset_name](
            train_utils.DATA_FOLDER_PATH,
            train=False,
            download=True,
            transform=train_utils.DATASET_TRANSFORMS[dataset_name],
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    return evaluate_model(model, dataloader, torchdevice)


### CLI


def get_args():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate model on dataset.")

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to model config JSON.",
    )

    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        required=True,
        choices=train_utils.DATASET_FUNCTIONS.keys(),
        help="Name of dataset.",
    )

    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        required=False,
        default=1024,
        help="Batch size.",
    )

    parser.add_argument(
        "--no-cuda",
        action="store_true",
        default=False,
        help="Use CPU instead of GPU.",
    )

    return parser.parse_args()


def main() -> None:
    args = get_args()
    model_config: model_config_utils.ModelConfig = model_config_utils.get_config_from_file(
        args.config
    )
    test_acc, test_loss = evaluate_model_from_config(
        model_config=model_config,
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        use_gpu=not args.no_cuda,
    )
    print(
        "{} - Test || Accuracy: {:.4f}% | Average loss: {:.4f}".format(
            args.dataset, test_acc, test_loss
        )
    )


if __name__ == "__main__":
    main()
