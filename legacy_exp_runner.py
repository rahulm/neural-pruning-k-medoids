"""Run experiments."""

import csv
import os
import shutil
from datetime import datetime
from typing import List, Text, Tuple

import eval_model
import pruner
import train_algo_1
from utils import logging_utils, prune_config_utils, train_config_utils

PRINT_FORMAT: Text = "{} || Size (bytes): {} | Train acc: {} | Test acc: {}"

LOGGER_NAME: Text = "runner.py"


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


def run_single_experiment(
    prune_config: prune_config_utils.PruneConfig,
    prune_out_folder_root_path: Text,
    prune_exp_name: Text,
    finetuning_train_config: train_config_utils.TrainConfig,
    original_model_config_path: Text,
    evaluation_epochs_list: List[int],
) -> List:
    # Logging.
    logger = logging_utils.get_logger(name=LOGGER_NAME)

    # Set up prune folder.
    prune_exp_folder_path: Text = os.path.join(
        prune_out_folder_root_path, prune_exp_name,
    )
    if not os.path.exists(prune_exp_folder_path):
        os.makedirs(prune_exp_folder_path)

    # Copy original model config.
    shutil.copyfile(
        src=original_model_config_path,
        dst=os.path.join(
            prune_exp_folder_path, "config-model-original_model.json"
        ),
    )

    logger.info(prune_exp_folder_path)

    # Prune.
    logger.info("pruning...")
    pruner.prune_network(
        prune_config=prune_config, pruned_output_folder=prune_exp_folder_path
    )
    pruned_model_path: Text = os.path.join(
        prune_exp_folder_path, pruner.FILE_NAME_MODEL
    )

    # # Evaluate.
    # logger.info("evaluating pre-finetuned...")
    # pruned_model_size, pruned_train_acc, pruned_test_acc = evaluate_model(
    #     model_path=pruned_model_path
    # )

    # Finetune.
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

    # Evaluate desired epochs.
    eval_results: List = []
    evaluation_epochs_list = [
        (finetuning_train_config.num_epochs if (epoch == -1) else epoch)
        for epoch in evaluation_epochs_list
    ]
    logger.info("evaluating epochs list: {}".format(evaluation_epochs_list))
    for epoch in evaluation_epochs_list:
        checkpoint_path: Text = os.path.join(
            finetuning_folder_path,
            "checkpoints",
            "checkpoint-epoch_{}-model.pth".format(epoch),
        )
        (
            checkpoint_model_size,
            checkpoint_train_acc,
            checkpoint_test_acc,
        ) = evaluate_model(model_path=checkpoint_path)
        eval_results.extend(
            [
                "",
                epoch,
                checkpoint_model_size,
                checkpoint_train_acc,
                checkpoint_test_acc,
            ]
        )
        logger.info(
            PRINT_FORMAT.format(
                "Epoch {}".format(epoch),
                checkpoint_model_size,
                checkpoint_train_acc,
                checkpoint_test_acc,
            )
        )

    return eval_results


def run_craig_experiments(
    experiment_vals: List,
    prune_out_folder_root_path: Text,
    original_model_name: Text,
    original_model_path: Text,
    original_model_config_path: Text,
    original_model_train_config_path: Text,
    original_model_results: List,
    evaluation_epochs_list: List[int],
) -> None:
    # Logging.
    logger = logging_utils.get_logger(name=LOGGER_NAME)

    # Set up root configs.
    prune_config_root: prune_config_utils.PruneConfig = prune_config_utils.PruneConfig(
        {"prune_type": "craig", "prune_params": {}}
    )
    finetuning_train_config: train_config_utils.TrainConfig = train_config_utils.get_config_from_file(
        config_file_loc=original_model_train_config_path
    )
    finetuning_train_config.num_epochs = 20  # TODO: Play with this value?

    # Experiment parameters.
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

    for similarity_metric in list_similarity_metric:
        for percent_per_layer in list_percent_per_layer:
            prune_config_root.prune_params[
                "similarity_metric"
            ] = similarity_metric
            prune_config_root.prune_params[
                "prune_percent_per_layer"
            ] = percent_per_layer

            prune_exp_name: Text = "{}-{}".format(
                similarity_metric, percent_per_layer
            )

            exp_result: List = [
                original_model_name,
                prune_exp_name,
                percent_per_layer,
                "",
            ]
            exp_result.extend(original_model_results.copy())
            exp_result.extend(
                run_single_experiment(
                    prune_config=prune_config_root,
                    prune_out_folder_root_path=prune_out_folder_root_path,
                    prune_exp_name=prune_exp_name,
                    finetuning_train_config=finetuning_train_config,
                    original_model_config_path=original_model_config_path,
                    evaluation_epochs_list=evaluation_epochs_list,
                )
            )

            experiment_vals.append(exp_result)


def run_mussay_experiments(
    experiment_vals: List,
    prune_out_folder_root_path: Text,
    original_model_name: Text,
    original_model_path: Text,
    original_model_config_path: Text,
    original_model_train_config_path: Text,
    original_model_results: List,
    evaluation_epochs_list: List[int],
) -> None:
    # Logging.
    logger = logging_utils.get_logger(name=LOGGER_NAME)

    # Set up root configs.
    prune_config_root: prune_config_utils.PruneConfig = prune_config_utils.PruneConfig(
        {
            "prune_type": "mussay",
            "prune_params": {"compression_type": "Coreset"},
        }
    )
    finetuning_train_config: train_config_utils.TrainConfig = train_config_utils.get_config_from_file(
        config_file_loc=original_model_train_config_path
    )
    finetuning_train_config.num_epochs = 20  # TODO: Play with this value?

    # Experiment parameters.
    prune_config_root.original_model_path = original_model_path
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
    list_upper_bound: List[
        float
    ] = [  # TODO: Figure out what range is used here for beta.
        0.25,
        0.50,
        0.75,
        1.00,
        2.00,
        10.0,
    ]

    for upper_bound in list_upper_bound:
        for percent_per_layer in list_percent_per_layer:
            prune_config_root.prune_params[
                "prune_percent_per_layer"
            ] = percent_per_layer
            prune_config_root.prune_params["upper_bound"] = upper_bound

            prune_exp_name: Text = "upper_bound_{}-prune_percent_{}".format(
                upper_bound, percent_per_layer
            )

            exp_result: List = [
                original_model_name,
                prune_exp_name,
                upper_bound,
                percent_per_layer,
                "",
            ]
            exp_result.extend(original_model_results.copy())
            exp_result.extend(
                run_single_experiment(
                    prune_config=prune_config_root,
                    prune_out_folder_root_path=prune_out_folder_root_path,
                    prune_exp_name=prune_exp_name,
                    finetuning_train_config=finetuning_train_config,
                    original_model_config_path=original_model_config_path,
                    evaluation_epochs_list=evaluation_epochs_list,
                )
            )

            experiment_vals.append(exp_result)


def run_experiments(
    prune_type: Text,
    prune_out_folder_root_path: Text,
    original_model_path: Text,
    original_model_config_path: Text,
    original_model_train_config_path: Text,
    evaluation_epochs_list: List[int],
    datetime_string: Text,
) -> None:
    """
    For each experiment:
    - Prune network, save pruned model (pruner).
    - Get train and test accuracy for pruned model on MNIST (eval_model).
    - Fine tune model.
    - Get train and test accuracy for pruned+finetuned model on MNIST (eval_model).

    Accumulate size, train accuracy, and test accuracy for each run.
    Print these out in a csv (or formatted that way).
    """

    # Logging.
    logger = logging_utils.get_logger(name=LOGGER_NAME)

    # Original model.
    original_model_name: Text = os.path.basename(original_model_path)
    original_size, original_train_acc, original_test_acc = evaluate_model(
        model_path=original_model_path
    )
    logger.info(
        PRINT_FORMAT.format(
            "original", original_size, original_train_acc, original_test_acc
        )
    )

    original_model_results: List = [
        original_size,
        original_train_acc,
        original_test_acc,
    ]

    # Experiment results container.
    experiment_vals: List[List] = []

    # TODO: Add support for non-CRAIG experiments.
    try:
        if prune_type == "craig":
            run_craig_experiments(
                experiment_vals=experiment_vals,
                prune_out_folder_root_path=prune_out_folder_root_path,
                original_model_name=original_model_name,
                original_model_path=original_model_path,
                original_model_config_path=original_model_config_path,
                original_model_train_config_path=original_model_train_config_path,
                original_model_results=original_model_results,
                evaluation_epochs_list=evaluation_epochs_list,
            )
        elif prune_type == "mussay":
            run_mussay_experiments(
                experiment_vals=experiment_vals,
                prune_out_folder_root_path=prune_out_folder_root_path,
                original_model_name=original_model_name,
                original_model_path=original_model_path,
                original_model_config_path=original_model_config_path,
                original_model_train_config_path=original_model_train_config_path,
                original_model_results=original_model_results,
                evaluation_epochs_list=evaluation_epochs_list,
            )
        else:
            raise ValueError("prune_type not supported: {}".format(prune_type))
    finally:
        # Write results to csv.
        logger.info("writing results...")
        out_csv_path: Text = os.path.join(
            prune_out_folder_root_path,
            "results-{}.csv".format(datetime_string),
        )
        with open(out_csv_path, "w", newline="") as out_csv:
            csv_writer = csv.writer(out_csv)
            csv_writer.writerows(experiment_vals)
        logger.info("results at: {}".format(out_csv_path))


def main() -> None:
    # NOTE: Only change these values.
    # original_model_path: Text = "experiments/lenet_300_100-finetuned/training/checkpoints/checkpoint-epoch_40-model.pth"
    # original_model_config_path: Text = "experiments/lenet_300_100-finetuned/config-model.json"
    # original_model_train_config_path: Text = "experiments/lenet_300_100-finetuned/config-train.json"
    # prune_out_folder_root_path: Text = "experiments/lenet_300_100-finetuned/pruning-craig"
    original_model_path: Text = "experiments/lenet_300_100-finetuned/training/checkpoints/checkpoint-epoch_40-model.pth"
    original_model_config_path: Text = "experiments/lenet_300_100-finetuned/config-model.json"
    original_model_train_config_path: Text = "experiments/lenet_300_100-finetuned/config-train.json"
    prune_out_folder_root_path: Text = "experiments/lenet_300_100-finetuned/pruning-mussay"
    prune_type: Text = "mussay"  # craig
    evaluation_epochs_list: List[int] = [0, 1, 2, 3, -1]

    # Logging.
    datetime_string: Text = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    logging_utils.setup_logging(
        os.path.join(
            prune_out_folder_root_path,
            "runner-log-{}.txt".format(datetime_string),
        )
    )

    # Run.
    run_experiments(
        prune_type=prune_type,
        prune_out_folder_root_path=prune_out_folder_root_path,
        original_model_path=original_model_path,
        original_model_config_path=original_model_config_path,
        original_model_train_config_path=original_model_train_config_path,
        evaluation_epochs_list=evaluation_epochs_list,
        datetime_string=datetime_string,
    )


if __name__ == "__main__":
    main()
