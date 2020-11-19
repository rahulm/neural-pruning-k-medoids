"""Run experiments."""

import csv
import os
import shutil
from datetime import datetime
from typing import List, Text, Tuple

import craig_pruner
import eval_model
import train_algo_1
from utils import logging_utils, prune_config_utils, train_config_utils


def evaluate_model(model_path: Text) -> Tuple[int, float, float]:
    model_size: int = os.path.getsize(model_path)
    model_train_acc, _ = eval_model.evaluate_model_from_checkpoint_file(
        model_path_checkpoint=model_path,
        dataset_name="mnist",
        batch_size=2048,
        use_gpu=True,
        split="train",
    )
    model_test_acc, _ = eval_model.evaluate_model_from_checkpoint_file(
        model_path_checkpoint=model_path,
        dataset_name="mnist",
        batch_size=2048,
        use_gpu=True,
        split="test",
    )
    return model_size, model_train_acc, model_test_acc


"""
For each experiment:
- Prune network, save pruned model (craig_pruner).
- Get train and test accuracy for pruned model on MNIST (eval_model).
- Fine tune model.
- Get train and test accuracy for pruned+finetuned model on MNIST (eval_model).

Accumulate size, train accuracy, and test accuracy for each run.
Print these out in a csv (or formatted that way).
"""


# original_model_path: Text = "experiments/lenet_300_100-no_finetuning/training/checkpoints/checkpoint-epoch_40-model.pth"
# prune_out_folder_root_path: Text = "experiments/lenet_300_100-no_finetuning/pruning"

original_model_path: Text = "experiments/lenet_300_100-finetuned/training/checkpoints/checkpoint-epoch_40-model.pth"
original_model_config_path: Text = "experiments/lenet_300_100-finetuned/config-model.json"
original_model_train_config_path: Text = "experiments/lenet_300_100-finetuned/config-train.json"
prune_out_folder_root_path: Text = "experiments/lenet_300_100-finetuned/pruning"

# Logging
datetime_string: Text = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
logging_utils.setup_logging(
    os.path.join(
        prune_out_folder_root_path, "runner-log-{}.txt".format(datetime_string)
    )
)
logger = logging_utils.get_logger(name="runner.py")
PRINT_FORMAT: Text = "{} || Size (bytes): {} | Train acc: {} | Test acc: {}"

# Set up root configs
prune_config_root: prune_config_utils.PruneConfig = prune_config_utils.PruneConfig(
    {"prune_type": "craig", "prune_params": {}}
)
finetuning_train_config: train_config_utils.TrainConfig = train_config_utils.get_config_from_file(
    config_file_loc=original_model_train_config_path
)
finetuning_train_config.num_epochs = 20  # TODO: Play with this value?

# Original model
original_model_name: Text = os.path.basename(original_model_path)
original_size, original_train_acc, original_test_acc = evaluate_model(
    model_path=original_model_path
)
logger.info(
    PRINT_FORMAT.format(
        "original", original_size, original_train_acc, original_test_acc
    )
)

# Experiment parameters
prune_config_root.original_model_path = original_model_path
list_similarity_metric: List[Text] = [
    "euclidean_distance",
    "rbf_kernel",
    "l1_norm",
    "cosine_similarity",
    "weights_covariance",
]
list_percent_per_layer: List[float] = [
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
]

experiment_vals: List[List] = []
for similarity_metric in list_similarity_metric:
    for percent_per_layer in list_percent_per_layer:
        prune_config_root.prune_params["similarity_metric"] = similarity_metric
        prune_config_root.prune_params[
            "prune_percent_per_layer"
        ] = percent_per_layer

        prune_exp_name: Text = "{}-{}".format(
            similarity_metric, percent_per_layer
        )
        prune_exp_folder_path: Text = os.path.join(
            prune_out_folder_root_path, prune_exp_name,
        )
        if not os.path.exists(prune_exp_folder_path):
            os.makedirs(prune_exp_folder_path)

        # Copy original model config
        shutil.copyfile(
            src=original_model_config_path,
            dst=os.path.join(
                prune_exp_folder_path, "config-model-original_model.json"
            ),
        )

        logger.info(prune_exp_folder_path)

        # Prune
        logger.info("pruning...")
        craig_pruner.prune_network_with_craig(
            prune_config_root, prune_exp_folder_path
        )
        pruned_model_path: Text = os.path.join(
            prune_exp_folder_path, craig_pruner.FILE_NAME_MODEL
        )

        # Evaluate
        logger.info("evaluating pre-finetuned...")
        pruned_model_size, pruned_train_acc, pruned_test_acc = evaluate_model(
            model_path=pruned_model_path
        )

        # Finetune
        logger.info("finetuning...")
        finetuning_folder_path: Text = os.path.join(
            prune_exp_folder_path, "finetuning"
        )
        train_algo_1.train_model_with_configs(
            model_config_or_checkpoint=pruned_model_path,
            train_config=finetuning_train_config,
            experiment_folder_path=finetuning_folder_path,
        )
        last_finetuned_checkpoint_path: Text = os.path.join(
            finetuning_folder_path,
            "checkpoints",
            "checkpoint-epoch_{}-model.pth".format(
                finetuning_train_config.num_epochs
            ),
        )
        finetuned_model_path: Text = os.path.join(
            prune_exp_folder_path, "pruned_finetuned_model.pth"
        )
        shutil.copyfile(
            src=last_finetuned_checkpoint_path, dst=finetuned_model_path
        )

        # Evaluate
        logger.info("evaluating finetuned...")
        (
            finetuned_model_size,
            finetuned_train_acc,
            finetuned_test_acc,
        ) = evaluate_model(model_path=finetuned_model_path)

        # Add to list
        # NOTE: This is based of my Google Sheets results format
        experiment_vals.append(
            [
                original_model_name,
                prune_exp_name,
                percent_per_layer,
                "",
                original_size,
                original_train_acc,
                original_test_acc,
                "",
                pruned_model_size,
                pruned_train_acc,
                pruned_test_acc,
                "",
                finetuned_model_size,
                finetuned_train_acc,
                finetuned_test_acc,
            ]
        )

        # Print
        logger.info(
            PRINT_FORMAT.format(
                "{} - {}".format(prune_exp_name, "pruned"),
                pruned_model_size,
                pruned_train_acc,
                pruned_test_acc,
            )
        )
        logger.info(
            PRINT_FORMAT.format(
                "{} - {}".format(prune_exp_name, "finetuned"),
                finetuned_model_size,
                finetuned_train_acc,
                finetuned_test_acc,
            )
        )

# Write results to csv
logger.info("writing results...")
out_csv_path: Text = os.path.join(
    prune_out_folder_root_path, "results-{}.csv".format(datetime_string),
)
with open(out_csv_path, "w", newline="") as out_csv:
    csv_writer = csv.writer(out_csv)
    csv_writer.writerows(experiment_vals)
logger.info("results at: {}".format(out_csv_path))
