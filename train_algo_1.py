"""
This is a semi-generic training algo, arbitrarily named "train_algo_1".
It may be good to rename this to something more informative.
"""

import importlib
import os
from datetime import datetime
from typing import Text

import torch
import torch.nn.functional as F
import torchvision

import eval_model
from utils import (
    logging_utils,
    model_config_utils,
    train_config_utils,
    train_utils,
)


def train(
    logger,
    log_interval: int,
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    epoch: int,
    optimizer,
    scheduler,
    torch_device: torch.device,
    train_loss_batches: train_utils.StatCounter,
    train_loss_epochs: train_utils.StatCounter,
    train_acc_batches: train_utils.StatCounter,
    train_acc_epochs: train_utils.StatCounter,
) -> None:
    model.train()
    dataset_len: int = len(train_loader.dataset)
    total_data_count: int = 0
    curr_loss: float = 0.0
    num_correct: int = 0

    for batch_ind, (data, target) in enumerate(train_loader):
        total_data_count += len(data)

        data, target = data.to(torch_device), target.to(torch_device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        curr_loss = loss.item()
        train_loss_batches.add(curr_loss)

        pred = output.argmax(dim=1, keepdim=True)
        num_correct_per_batch: int = pred.eq(target.view_as(pred)).sum().item()
        train_acc_batches.add(num_correct_per_batch / len(data))

        num_correct += num_correct_per_batch

        if batch_ind % log_interval == 0:
            logger.info(
                "Train || Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    total_data_count,
                    dataset_len,
                    100.0 * total_data_count / dataset_len,
                    curr_loss,
                )
            )

    train_loss_epochs.add(curr_loss)
    train_acc_epochs.add(num_correct / total_data_count)


def train_model_with_configs(
    logger,
    model_config: model_config_utils.ModelConfig,
    train_config: train_config_utils.TrainConfig,
    experiment_folder_path: Text,
    use_gpu: bool = True,
) -> None:
    log_interval: int = 100

    torch_device = torch.device("cuda" if use_gpu else "cpu")
    torch.manual_seed(train_config.random_seed)

    # Set up counters.
    train_loss_batches: train_utils.StatCounter = train_utils.StatCounter()
    train_loss_epochs: train_utils.StatCounter = train_utils.StatCounter()
    train_acc_batches: train_utils.StatCounter = train_utils.StatCounter()
    train_acc_epochs: train_utils.StatCounter = train_utils.StatCounter()

    test_loss_epochs: train_utils.StatCounter = train_utils.StatCounter()
    test_acc_epochs: train_utils.StatCounter = train_utils.StatCounter()

    # Get data.
    data_transform = train_utils.DATASET_TRANSFORMS[train_config.dataset_name]
    train_loader = torch.utils.data.DataLoader(
        train_utils.DATASET_FUNCTIONS[train_config.dataset_name](
            train_utils.DATA_FOLDER_PATH,
            train=True,
            download=True,
            transform=data_transform,
        ),
        batch_size=train_config.batch_size_train,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        train_utils.DATASET_FUNCTIONS[train_config.dataset_name](
            train_utils.DATA_FOLDER_PATH,
            train=False,
            download=True,
            transform=data_transform,
        ),
        batch_size=train_config.batch_size_test,
        shuffle=True,
    )

    # Load model.
    model_py_module = importlib.import_module(
        "models.{}".format(model_config.model_architecture)
    )
    Model = model_py_module.Model  # type: ignore
    # model: Model = Model(
    model: torch.nn.Module = Model(**model_config.model_params)
    model.to(device=torch_device)

    # Just using basic Stochastic Gradient Descent.
    # TODO: Add weigh decay? May not be necesssary for this task
    optimizer = torch.optim.SGD(
        params=model.parameters(),
        lr=train_config.learning_rate,
        momentum=train_config.momentum,
        weight_decay=train_config.weight_decay,
    )
    # optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=train_config.lr_step_size, gamma=train_config.gamma
    )

    checkpoints_folder_path: Text = os.path.join(
        experiment_folder_path, "checkpoints"
    )
    if not os.path.exists(checkpoints_folder_path):
        os.makedirs(checkpoints_folder_path)

    try:
        for epoch in range(1, train_config.num_epochs + 1):
            train(
                logger,
                log_interval,
                model,
                train_loader,
                epoch,
                optimizer,
                scheduler,
                torch_device,
                train_loss_batches,
                train_loss_epochs,
                train_acc_batches,
                train_acc_epochs,
            )

            test_acc, test_loss = eval_model.evaluate_model(
                model=model, dataloader=test_loader, torchdevice=torch_device
            )
            test_acc_epochs.add(test_acc)
            test_loss_epochs.add(test_loss)
            scheduler.step()

            # Save model checkpoint.
            torch.save(
                model,
                os.path.join(
                    checkpoints_folder_path,
                    "checkpoint-epoch_{}-model.pth".format(epoch),
                ),
            )
            torch.save(
                model.state_dict(),
                os.path.join(
                    checkpoints_folder_path,
                    "checkpoint-epoch_{}-weight_only.pth".format(epoch),
                ),
            )
    finally:
        # Save losses.
        stats_folder_path: Text = os.path.join(experiment_folder_path, "stats")
        train_loss_batches.save(
            folder_path=stats_folder_path,
            file_prefix="train_loss_batches",
            xlabel="batch",
            ylabel="loss",
            title_prefix="train_loss_batches",
        )
        train_loss_epochs.save(
            folder_path=stats_folder_path,
            file_prefix="train_loss_epochs",
            xlabel="epoch",
            ylabel="loss",
            title_prefix="train_loss_epochs",
        )

        train_acc_batches.save(
            folder_path=stats_folder_path,
            file_prefix="train_accuracy_batches",
            xlabel="batch",
            ylabel="accuracy",
            title_prefix="train_accuracy_batches",
        )
        train_acc_epochs.save(
            folder_path=stats_folder_path,
            file_prefix="train_accuracy_epochs",
            xlabel="epoch",
            ylabel="accuracy",
            title_prefix="train_accuracy_epochs",
        )

        test_loss_epochs.save(
            folder_path=stats_folder_path,
            file_prefix="test_loss_epochs",
            xlabel="epoch",
            ylabel="loss",
            title_prefix="test_loss_epochs",
        )
        test_acc_epochs.save(
            folder_path=stats_folder_path,
            file_prefix="test_accuracy_epochs",
            xlabel="epoch",
            ylabel="accuracy",
            title_prefix="test_accuracy_epochs",
        )


### CLI


def get_args():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate model on dataset.")

    parser.add_argument(
        "-m",
        "--model_config",
        type=str,
        required=True,
        help="Path to model config JSON.",
    )

    parser.add_argument(
        "-t",
        "--train_config",
        type=str,
        required=True,
        help="Path to train config JSON.",
    )

    parser.add_argument(
        "-e",
        "--experiment_id",
        type=str,
        required=False,
        default=datetime.now().strftime("%Y_%m_%d-%H_%M_%S"),
        help="Experiment id (folder name).",
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

    experiment_folder_path: Text = os.path.join(
        "experiments", args.experiment_id
    )
    if not os.path.exists(experiment_folder_path):
        os.makedirs(experiment_folder_path)

    logging_utils.setup_logging(
        os.path.join(
            experiment_folder_path,
            "log-{}.txt".format(datetime.now().strftime("%Y_%m_%d-%H_%M_%S")),
        )
    )
    logger = logging_utils.get_logger(name=__name__)

    logger.info(args)

    model_config: model_config_utils.ModelConfig = model_config_utils.get_config_from_file(
        args.model_config
    )

    train_config: train_config_utils.TrainConfig = train_config_utils.get_config_from_file(
        args.train_config
    )

    train_model_with_configs(
        logger,
        model_config,
        train_config,
        experiment_folder_path,
        use_gpu=not args.no_cuda,
    )


if __name__ == "__main__":
    main()

