"""
This is a semi-generic training algo, arbitrarily named "train_algo_1".
It may be good to rename this to something more informative.
"""

import gc
import importlib
import os
import random
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Text, Union

import numpy as np
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

FILE_NAME_FORMAT_CHECKPOINT_MODEL: Text = "checkpoint-epoch_{}-model.pt"
FILE_NAME_FORMAT_CHECKPOINT_STATE_DICT: Text = "checkpoint-epoch_{}-state_dict.pt"
FOLDER_NAME_CHECKPOINTS: Text = "checkpoints"
BEST_CHECKPOINT_EPOCH_TEXT: Text = "best"


def clear_mem(logger) -> None:
    # NOTE: Not sure if this is needed or works well.
    torch.cuda.empty_cache()
    time_before = time.time()
    num_collected = gc.collect()
    logger.info(
        "Garbage collected {} items in {} seconds".format(
            num_collected, time.time() - time_before
        )
    )


def save_model_and_state_dict_checkpoint(
    model: torch.nn.Module,
    checkpoints_folder_path: Text,
    epoch: Union[int, Text],
    checkpoint_name: Optional[Text] = None,
    model_config: Optional[model_config_utils.ModelConfig] = None,
    optimizer=None,
    scheduler=None,
    **kwargs
) -> None:
    if not checkpoint_name:
        checkpoint_name = str(epoch)

    torch.save(
        model,
        os.path.join(
            checkpoints_folder_path,
            FILE_NAME_FORMAT_CHECKPOINT_MODEL.format(checkpoint_name),
        ),
    )

    # state_dict = {
    #     "model_state_dict": model.state_dict(),
    #     "epoch": epoch,
    #     "checkpoint_name": checkpoint_name,
    # }
    # if model_config:
    #     state_dict["model_config"] = model_config._raw_dict
    # if optimizer:
    #     state_dict["optimizer_state_dict"] = optimizer.state_dict()
    # if scheduler:
    #     state_dict["scheduler_state_dict"] = scheduler.state_dict()
    # state_dict.update(kwargs)

    # torch.save(
    #     state_dict,
    #     os.path.join(
    #         checkpoints_folder_path,
    #         FILE_NAME_FORMAT_CHECKPOINT_STATE_DICT.format(checkpoint_name),
    #     ),
    # )


# def save_model_checkpoint(
#     model: torch.nn.Module,
#     checkpoints_folder_path: Text,
#     epoch: Union[int, Text],
# ) -> None:
#     torch.save(
#         model,
#         os.path.join(
#             checkpoints_folder_path,
#             FILE_NAME_FORMAT_CHECKPOINT_MODEL.format(epoch),
#         ),
#     )


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
    model_config_or_checkpoint: Union[model_config_utils.ModelConfig, Text],
    train_config: train_config_utils.TrainConfig,
    experiment_folder_path: Text,
    resume_training: bool = False,
    save_interval: int = 1,
    save_best_checkpoint: bool = True,
    use_gpu: bool = True,
    # cuda_device_id: int = 0,
) -> Dict[Text, train_utils.StatCounter]:
    logger = logging_utils.get_logger(__name__)
    log_interval: int = 100

    assert save_interval >= 0, "save_interval must be >= 0"
    save_checkpoint_per_epoch: bool = (save_interval != 0)

    torch_device = torch.device("cuda" if use_gpu else "cpu")
    if "random_seed" in train_config._raw_dict:
        random.seed(train_config.random_seed)
        np.random.seed(train_config.random_seed)
        torch.manual_seed(train_config.random_seed)
        torch.cuda.manual_seed(train_config.random_seed)
    # Using this for reproducibility
    torch.backends.cudnn.deterministic = True

    random_info_str: Text = """Random info:
random.setstate({random})
np.random.set_state({nprandom})
torch.manual_seed({torch})
torch.cuda.manual_seed({torchcuda})
torch.backends.cudnn.deterministic = {torchcudnn}
    """.format(
        random=random.getstate(),
        nprandom=np.random.get_state(),
        torch=torch.initial_seed(),
        torchcuda=torch.cuda.initial_seed(),
        torchcudnn=torch.backends.cudnn.deterministic,
    )
    logger.info(random_info_str)

    # Set up some experiment directories.
    checkpoints_folder_path: Text = os.path.join(
        experiment_folder_path, FOLDER_NAME_CHECKPOINTS
    )
    if not os.path.exists(checkpoints_folder_path):
        os.makedirs(checkpoints_folder_path)
    stats_folder_path: Text = os.path.join(experiment_folder_path, "stats")

    # Set up counters.
    train_loss_batches: train_utils.StatCounter = train_utils.StatCounter(
        default_save_params=dict(
            folder_path=stats_folder_path,
            file_prefix="train_loss_batches",
            xlabel="batch",
            ylabel="loss",
            title_prefix="train_loss_batches",
        )
    )
    train_loss_epochs: train_utils.StatCounter = train_utils.StatCounter(
        default_save_params=dict(
            folder_path=stats_folder_path,
            file_prefix="train_loss_epochs",
            xlabel="epoch",
            ylabel="loss",
            title_prefix="train_loss_epochs",
        )
    )
    train_acc_batches: train_utils.StatCounter = train_utils.StatCounter(
        default_save_params=dict(
            folder_path=stats_folder_path,
            file_prefix="train_accuracy_batches",
            xlabel="batch",
            ylabel="accuracy",
            title_prefix="train_accuracy_batches",
        )
    )
    train_acc_epochs: train_utils.StatCounter = train_utils.StatCounter(
        default_save_params=dict(
            folder_path=stats_folder_path,
            file_prefix="train_accuracy_epochs",
            xlabel="epoch",
            ylabel="accuracy",
            title_prefix="train_accuracy_epochs",
        )
    )
    test_loss_epochs: train_utils.StatCounter = train_utils.StatCounter(
        default_save_params=dict(
            folder_path=stats_folder_path,
            file_prefix="test_loss_epochs",
            xlabel="epoch",
            ylabel="loss",
            title_prefix="test_loss_epochs",
        )
    )
    test_acc_epochs: train_utils.StatCounter = train_utils.StatCounter(
        default_save_params=dict(
            folder_path=stats_folder_path,
            file_prefix="test_accuracy_epochs",
            xlabel="epoch",
            ylabel="accuracy",
            title_prefix="test_accuracy_epochs",
        )
    )
    model_size_epochs: train_utils.StatCounter = train_utils.StatCounter(
        default_save_params=dict(
            folder_path=stats_folder_path,
            file_prefix="model_size_epochs",
            xlabel="epoch",
            ylabel="number of model parameters",
            title_prefix="model_size_epochs",
        )
    )
    stat_counters: Dict[Text, train_utils.StatCounter] = {
        "train_loss_batches": train_loss_batches,
        "train_loss_epochs": train_loss_epochs,
        "train_acc_batches": train_acc_batches,
        "train_acc_epochs": train_acc_epochs,
        "test_loss_epochs": test_loss_epochs,
        "test_acc_epochs": test_acc_epochs,
        "model_size_epochs": model_size_epochs,
    }

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
    model_config: Optional[model_config_utils.ModelConfig] = None
    optimizer_state_dict: Optional[Any] = None
    scheduler_state_dict: Optional[Any] = None
    resume_epoch: Optional[int] = None
    model: torch.nn.Module
    if isinstance(model_config_or_checkpoint, model_config_utils.ModelConfig):
        model_config = model_config_or_checkpoint
        model_py_module = importlib.import_module(
            "models.{}".format(model_config.model_architecture)
        )
        Model = model_py_module.Model  # type: ignore
        model = Model(**model_config.model_params)
    elif isinstance(model_config_or_checkpoint, Text):
        model_checkpoint_path: Text = model_config_or_checkpoint
        loaded = torch.load(model_checkpoint_path, map_location=torch_device)
        if isinstance(loaded, torch.nn.Module):
            # Model.
            model = loaded
        else:
            # State dict.
            model_config = model_config_utils.ModelConfig(
                loaded["model_config"]
            )
            model_py_module = importlib.import_module(
                "models.{}".format(model_config.model_architecture)
            )
            Model = model_py_module.Model  # type: ignore
            model = Model(**model_config.model_params)
            model.load_state_dict(loaded["model_state_dict"])
            if resume_training:
                optimizer_state_dict = loaded.get("optimizer_state_dict", None)
                scheduler_state_dict = loaded.get("scheduler_state_dict", None)
                resume_epoch = loaded.get("epoch", None)

    else:
        err_msg: Text = "Model config or path to model checkpoint must be provided."
        logger.error(err_msg)
        raise TypeError(err_msg)
    model = model.to(device=torch_device)

    # Just using basic Stochastic Gradient Descent.
    # TODO: Add weigh decay? May not be necesssary for this task
    optimizer = torch.optim.SGD(
        params=model.parameters(),
        lr=train_config.learning_rate,
        momentum=train_config.momentum,
        weight_decay=train_config.weight_decay,
    )
    if optimizer_state_dict:
        optimizer.load_state_dict(optimizer_state_dict)
    # optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=train_config.lr_step_size, gamma=train_config.gamma
    )
    if scheduler_state_dict:
        scheduler.load_state_dict(scheduler_state_dict)

    # Set up first epoch, if need to resume.
    first_epoch: int = 1
    if resume_epoch:
        first_epoch = resume_epoch

    try:
        # First, get initial train and test scores.
        initial_train_acc, initial_train_loss = eval_model.evaluate_model(
            model=model, dataloader=train_loader, torch_device=torch_device
        )
        train_acc_batches.add(initial_train_acc)
        train_acc_epochs.add(initial_train_acc)
        train_loss_batches.add(initial_train_loss)
        train_loss_epochs.add(initial_train_loss)
        initial_test_acc, initial_test_loss = eval_model.evaluate_model(
            model=model, dataloader=test_loader, torch_device=torch_device
        )
        test_acc_epochs.add(initial_test_acc)
        test_loss_epochs.add(initial_test_loss)
        model_size_epochs.add(
            eval_model.get_number_of_model_parameters(model=model)
        )

        # Save initial model checkpoint.
        if save_checkpoint_per_epoch:
            save_model_and_state_dict_checkpoint(
                model=model,
                checkpoints_folder_path=checkpoints_folder_path,
                epoch=0,
                model_config=model_config,
                optimizer=optimizer,
                scheduler=scheduler,
            )

        clear_mem(logger)

        # Track best test accuracy.
        best_test_acc: float = initial_test_acc

        # Train.
        for epoch in range(first_epoch, train_config.num_epochs + 1):
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
                model=model, dataloader=test_loader, torch_device=torch_device
            )
            test_acc_epochs.add(test_acc)
            test_loss_epochs.add(test_loss)
            model_size_epochs.add(
                eval_model.get_number_of_model_parameters(model=model)
            )

            scheduler.step()

            # Save best model checkpoint, if needed.
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                if save_best_checkpoint:
                    save_model_and_state_dict_checkpoint(
                        model=model,
                        checkpoints_folder_path=checkpoints_folder_path,
                        epoch=epoch,
                        checkpoint_name=BEST_CHECKPOINT_EPOCH_TEXT,
                        model_config=model_config,
                        optimizer=optimizer,
                        scheduler=scheduler,
                    )

            # Save incremental checkpoint, if needed.
            if save_checkpoint_per_epoch and (
                (epoch == 1)
                or (epoch == train_config.num_epochs)
                or ((epoch % save_interval) == 0)
            ):
                save_model_and_state_dict_checkpoint(
                    model=model,
                    checkpoints_folder_path=checkpoints_folder_path,
                    epoch=epoch,
                    model_config=model_config,
                    optimizer=optimizer,
                    scheduler=scheduler,
                )

            # Incrementally save losses per epoch.
            for stat_counter in stat_counters.values():
                stat_counter.save_default()

            clear_mem(logger)

    except Exception as exception:
        logger.error(exception, exc_info=True)
    finally:
        # Save losses.
        for stat_counter in stat_counters.values():
            stat_counter.save_default()

        return stat_counters


### CLI


def get_args():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate model on dataset.")

    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "-m",
        "--model_config",
        type=str,
        default=None,
        help="Path to model config JSON. Need this or --checkpoint.",
    )
    model_group.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint to resume or finetune from. Need this or --model_config.",
    )

    parser.add_argument(
        "-t",
        "--train_config",
        type=str,
        required=True,
        help="Path to train config JSON.",
    )

    parser.add_argument(
        "--out_folder",
        type=str,
        required=False,
        default=os.path.join(
            "experiments",
            datetime.now().strftime("%Y_%m_%d-%H_%M_%S"),
            "training",
        ),
        help=(
            "Folder where training results should be output."
            + " If not provided, a timestamped folder is autogenerated in the 'experiments' folder."
        ),
    )

    parser.add_argument(
        "-i",
        "--save_interval",
        type=int,
        required=False,
        default=1,
        help=(
            "Save interval (by epoch) for checkpoints."
            + " Set to 0 to never save per epoch."
        ),
    )

    parser.add_argument(
        "--save_best_checkpoint",
        action="store_true",
        required=False,
        default=False,
        help="Save best checkpoint based on test accuracy.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        required=False,
        default=False,
        help="If state dict is given, resume training instead of starting finetuning.",
    )

    parser.add_argument(
        "--no-cuda",
        action="store_true",
        default=False,
        help="Use CPU instead of GPU.",
    )

    parser.add_argument(
        "--cuda_device_id",
        required=False,
        type=str,
        default=0,
        help="The id of the cuda device, if used.",
    )

    return parser.parse_args()


def main() -> None:
    args = get_args()

    experiment_folder_path: Text = args.out_folder
    if not os.path.exists(experiment_folder_path):
        os.makedirs(experiment_folder_path)

    logging_utils.setup_logging(
        os.path.join(
            experiment_folder_path,
            "log-{}.txt".format(datetime.now().strftime("%Y_%m_%d-%H_%M_%S")),
        )
    )

    logger = logging_utils.get_logger(__name__)
    logger.info(args)

    model_config_or_checkpoint: Union[model_config_utils.ModelConfig, Text]
    if args.model_config:
        model_config_or_checkpoint = model_config_utils.get_config_from_file(
            args.model_config
        )
    elif args.checkpoint:
        model_config_or_checkpoint = args.checkpoint
    else:
        err_msg = "Either --model_config or --checkpoint must be provided."
        logger.error(err_msg)
        raise ValueError(err_msg)

    train_config: train_config_utils.TrainConfig = train_config_utils.get_config_from_file(
        args.train_config
    )

    with torch.cuda.device(args.cuda_device_id):
        train_model_with_configs(
            model_config_or_checkpoint=model_config_or_checkpoint,
            train_config=train_config,
            experiment_folder_path=experiment_folder_path,
            resume_training=args.resume_training,
            save_interval=args.save_interval,
            save_best_checkpoint=args.save_best_checkpoint,
            use_gpu=not args.no_cuda,
        )


if __name__ == "__main__":
    main()

