"""Run experiments."""

import csv
import os
from datetime import datetime
from typing import List, Text

import craig_pruner
import eval_model
from utils import prune_config_utils

# prune_config_root_path: Text = "experiments/lenet_300_100-1/config-prune-euclidean_distance_0.3.json"
# prune_config_root: prune_config_utils.PruneConfig = prune_config_utils.get_config_from_file(
#     prune_config_root_path
# )


"""
For each experiment:
- Prune network, save pruned model (craig_pruner).
- Get train and test accuracy for pruned model on MNIST (eval_model).

Accumulate size, train accuracy, and test accuracy for each run.
Print these out in a csv (or formatted that way).
"""

PRINT_FORMAT: Text = "{} || Size (bytes): {} | Train acc: {} | Test acc: {}"

prune_config_root: prune_config_utils.PruneConfig = prune_config_utils.PruneConfig(
    {"prune_type": "craig", "prune_params": {}}
)

original_model_path: Text = "experiments/lenet_300_100-1/training/checkpoints/checkpoint-epoch_40-model.pth"
prune_out_folder_root: Text = "experiments/lenet_300_100-1/pruning"

original_model_name: Text = os.path.basename(original_model_path)
original_train_acc, _ = eval_model.evaluate_model_from_checkpoint_file(
    model_path_checkpoint=original_model_path,
    dataset_name="mnist",
    batch_size=2056,
    use_gpu=True,
    split="train",
)
original_test_acc, _ = eval_model.evaluate_model_from_checkpoint_file(
    model_path_checkpoint=original_model_path,
    dataset_name="mnist",
    batch_size=2056,
    use_gpu=True,
    split="test",
)
original_size: int = os.path.getsize(original_model_path)
print(
    PRINT_FORMAT.format(
        "original", original_size, original_train_acc, original_test_acc
    )
)


prune_config_root.original_model_path = original_model_path
list_similarity_metric: List[Text] = [
    "euclidean_distance",
    "rbf_kernel",
    "cosine_similarity",
    "l1_norm",
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
            prune_out_folder_root, prune_exp_name,
        )

        print(prune_exp_folder_path)
        print("pruning...")

        # Prune
        craig_pruner.prune_network_with_craig(
            prune_config_root, prune_exp_folder_path
        )
        model_path_checkpoint: Text = os.path.join(
            prune_exp_folder_path, craig_pruner.FILE_NAME_MODEL
        )
        model_size: int = os.path.getsize(model_path_checkpoint)
        print("evaluating...")

        # Evaluate
        train_acc, _ = eval_model.evaluate_model_from_checkpoint_file(
            model_path_checkpoint=model_path_checkpoint,
            dataset_name="mnist",
            batch_size=2056,
            use_gpu=True,
            split="train",
        )
        test_acc, _ = eval_model.evaluate_model_from_checkpoint_file(
            model_path_checkpoint=model_path_checkpoint,
            dataset_name="mnist",
            batch_size=2056,
            use_gpu=True,
            split="test",
        )

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
                model_size,
                train_acc,
                test_acc,
            ]
        )

        # Print
        print(
            PRINT_FORMAT.format(prune_exp_name, model_size, train_acc, test_acc)
        )

# Write results to csv
print("writing results...")
out_csv_path: Text = os.path.join(
    prune_out_folder_root,
    datetime.now().strftime("results-%Y_%m_%d-%H_%M_%S.csv"),
)
with open(out_csv_path, "w", newline="") as out_csv:
    csv_writer = csv.writer(out_csv)
    csv_writer.writerows(experiment_vals)
print("results at: {}".format(out_csv_path))
