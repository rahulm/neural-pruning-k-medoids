"""Evaluate model on the desired dataset."""

import os
from typing import Callable, Dict, Text, Tuple

import torch
import torch.nn.functional as F
import torchvision

from utils import train_utils


def get_number_of_model_parameters(model: torch.nn.Module) -> int:
    return sum(
        dict((p.data_ptr(), p.numel()) for p in model.parameters()).values()
    )


def get_number_of_model_parameters_from_path(model_path: Text) -> int:
    return get_number_of_model_parameters(
        torch.load(model_path, map_location=torch.device("cpu"))
    )


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    torch_device: torch.device,
) -> Tuple[float, float]:
    loss_sum: float = 0.0
    num_correct: int = 0

    model.to(torch_device)
    model.eval()
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(torch_device), target.to(torch_device)
            output = model(data)
            loss_sum += F.cross_entropy(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            num_correct += pred.eq(target.view_as(pred)).sum().item()

    loss_ave: float = loss_sum / len(dataloader.dataset)
    accuracy: float = num_correct / len(dataloader.dataset)

    return accuracy, loss_ave


def evaluate_model_from_checkpoint_file(
    model_path_checkpoint: Text,
    dataset_name: Text,
    split: Text,
    batch_size: int,
    use_gpu: bool,
) -> Tuple[float, float]:
    torch_device: torch.device = torch.device("cuda" if use_gpu else "cpu")

    model = torch.load(model_path_checkpoint, map_location=torch_device)

    dataloader = torch.utils.data.DataLoader(
        train_utils.DATASET_FUNCTIONS[dataset_name](
            train_utils.DATA_FOLDER_PATH,
            train=True if (split == "train") else False,
            download=True,
            transform=train_utils.DATASET_TRANSFORMS[dataset_name],
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    return evaluate_model(model, dataloader, torch_device)


### CLI


def get_args():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate model on dataset.")

    parser.add_argument(
        "-m",
        "--model_checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint file.",
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

    parser.add_argument(
        "-s",
        "--split",
        type=str,
        required=False,
        default="test",
        choices=("test", "train"),
        help="The split to evaluate on.",
    )

    return parser.parse_args()


def main() -> None:
    args = get_args()
    test_acc, test_loss = evaluate_model_from_checkpoint_file(
        model_path_checkpoint=args.model_checkpoint,
        dataset_name=args.dataset,
        split=args.split,
        batch_size=args.batch_size,
        use_gpu=not args.no_cuda,
    )
    print(
        "{} - {} || Accuracy: {:.4f} | Average loss: {:.4f}".format(
            args.dataset, args.split, test_acc, test_loss
        )
    )


if __name__ == "__main__":
    main()
