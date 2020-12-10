"""Run experiments."""

import argparse
import csv
import itertools
import os
import random
import shutil
import time
from collections import OrderedDict
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Text, Tuple, Union

import eval_model
import pruner
import train_algo_1
from utils import (
    exp_config_utils,
    general_config_utils,
    logging_utils,
    model_config_utils,
    prune_config_utils,
    train_config_utils,
    train_utils,
)

PRINT_FORMAT: Text = "{} ||\tSize (number of parameters): {}\t|\tTrain acc: {}\t|\tTest acc: {}"

LOGGER_NAME: Text = "exp_runner"


def get_exp_str_from_param(param: Union[int, float, Text, Dict]) -> Text:
    if (
        isinstance(param, int)
        or isinstance(param, float)
        or isinstance(param, str)
    ):
        return str(param)
    if isinstance(param, dict):
        return str(param).replace(" ", "").replace(":", "_")

    raise ValueError("param must be str or dict: {}".format(param))


def evaluate_model(
    model_path: Text,
    dataset_name: Text,
    batch_size: int = 2048,
    model_size_type: Text = "numel",  # Choices: numel or disksize
) -> Tuple[int, float, float]:
    model_size: int
    if model_size_type == "disksize":
        model_size = os.path.getsize(model_path)
    elif model_size_type == "numel":
        model_size = eval_model.get_number_of_model_parameters_from_path(
            model_path
        )
    else:
        raise ValueError(
            "model_size_type not supported: {}".format(model_size_type)
        )

    model_train_acc, _ = eval_model.evaluate_model_from_checkpoint_file(
        model_path_checkpoint=model_path,
        dataset_name=dataset_name,
        batch_size=batch_size,
        use_gpu=True,
        split="train",
    )
    model_test_acc, _ = eval_model.evaluate_model_from_checkpoint_file(
        model_path_checkpoint=model_path,
        dataset_name=dataset_name,
        batch_size=batch_size,
        use_gpu=True,
        split="test",
    )
    return model_size, model_train_acc, model_test_acc


def write_results_to_csv(
    experiment_vals: List, out_folder_path: Text, datetime_string: Text
) -> Text:
    out_csv_path: Text = os.path.join(
        out_folder_path, "results_raw-{}.csv".format(datetime_string),
    )
    with open(out_csv_path, "w", newline="") as out_csv:
        csv_writer = csv.writer(out_csv)
        csv_writer.writerows(experiment_vals)
    return out_csv_path


def run_single_experiment(
    prune_config: prune_config_utils.PruneConfig,
    prune_out_folder_path: Text,
    finetuning_train_config: train_config_utils.TrainConfig,
    original_model_config: Optional[model_config_utils.ModelConfig],
    evaluation_epochs_list: Sequence[Union[Text, int]],
) -> List:
    # Logging.
    logger = logging_utils.get_logger(name=LOGGER_NAME)

    # Set up prune folder.
    if not os.path.exists(prune_out_folder_path):
        os.makedirs(prune_out_folder_path)

    # Copy original model config.
    if original_model_config:
        general_config_utils.write_config_to_file(
            original_model_config,
            os.path.join(
                prune_out_folder_path, "config-model-original_model.json"
            ),
        )

    # Prune.
    logger.info("pruning...")
    pruner.prune_network(
        prune_config=prune_config, pruned_output_folder=prune_out_folder_path
    )
    pruned_model_path: Text = os.path.join(
        prune_out_folder_path, pruner.FILE_NAME_MODEL
    )
    # pruned_model_path: Text = os.path.join(
    #     prune_out_folder_path, pruner.FILE_NAME_STATE_DICT
    # )

    # Finetune.
    logger.info("finetuning...")
    finetuning_folder_path: Text = os.path.join(
        prune_out_folder_path, "finetuning"
    )
    stat_counters: Dict[
        Text, train_utils.StatCounter
    ] = train_algo_1.train_model_with_configs(
        model_config_or_checkpoint=pruned_model_path,
        train_config=finetuning_train_config,
        experiment_folder_path=finetuning_folder_path,
        save_interval=0,  # Set to zero to never save per epoch, to save space.
        save_best_checkpoint=True,
        use_gpu=True,
    )

    # Save results from stat_counters: train/test accuracy, and size.
    eval_results: List = []
    evaluation_epochs_list = [
        (finetuning_train_config.num_epochs if (epoch == -1) else epoch)
        for epoch in evaluation_epochs_list
    ]
    model_size_epochs: train_utils.StatCounter = stat_counters[
        "model_size_epochs"
    ]
    train_acc_epochs: train_utils.StatCounter = stat_counters[
        "train_acc_epochs"
    ]
    test_acc_epochs: train_utils.StatCounter = stat_counters["test_acc_epochs"]
    for epoch in evaluation_epochs_list:
        model_size: int
        train_acc: float
        test_acc: float
        if epoch == "best":
            test_acc_ind = max(
                range(len(test_acc_epochs._counter)),
                key=lambda x: test_acc_epochs._counter[x],
            )
            test_acc = test_acc_epochs._counter[test_acc_ind]
            train_acc = train_acc_epochs._counter[test_acc_ind]
            model_size = model_size_epochs._counter[test_acc_ind]
        elif isinstance(epoch, int):
            test_acc = test_acc_epochs._counter[epoch]
            train_acc = train_acc_epochs._counter[epoch]
            model_size = model_size_epochs._counter[epoch]
        else:
            raise TypeError(
                "Found unsupported type in evaluation_epochs_list: {}".format(
                    epoch
                )
            )

        eval_results.extend(
            ["", epoch, model_size, train_acc, test_acc,]
        )
        logger.info(
            PRINT_FORMAT.format(
                "Epoch {}".format(epoch), model_size, train_acc, test_acc,
            )
        )

    return eval_results


def run_craig_experiments(
    experiment_vals: List,
    exp_config: exp_config_utils.ExpConfig,
    original_model_name: Text,
    original_model_path: Text,
    original_model_config: Optional[model_config_utils.ModelConfig],
    original_model_results: List,
    out_folder_path: Text,
    datetime_string: Text,
) -> None:
    # Logging.
    logger = logging_utils.get_logger(name=LOGGER_NAME)

    # Set up root configs.
    prune_config_root: prune_config_utils.PruneConfig = prune_config_utils.PruneConfig(
        {
            "prune_type": "craig",
            "prune_params": {},
            "original_model_path": original_model_path,
        }
    )
    if "model_input_shape" in exp_config._raw_dict:
        prune_config_root.model_input_shape = exp_config.model_input_shape
    if "data_transform_name" in exp_config._raw_dict:
        prune_config_root.data_transform_name = exp_config.data_transform_name
    finetuning_train_config: train_config_utils.TrainConfig = exp_config.finetuning_train_config

    # Experiment parameters.
    # prune_layer_params: OrderedDict = OrderedDict(
    prune_layer_params: Dict = exp_config.prune_params[
        prune_config_utils.KEY_LAYER_PARAMS
    ]
    prune_param_values: List = []
    layer_name_map: List[Text] = []
    param_name_map: List[Text] = []
    for layer_name, layer_params in prune_layer_params.items():
        for param_name, param_list in layer_params.items():
            prune_param_values.append(param_list)
            layer_name_map.append(layer_name)
            param_name_map.append(param_name)
    exp_value_permutations: List[List] = list(
        itertools.product(*prune_param_values)
    )

    for param_permutation in exp_value_permutations:
        exp_layer_params: Dict = {}
        for exp_param_ind, exp_param in enumerate(param_permutation):
            exp_param_dict = exp_layer_params.setdefault(
                layer_name_map[exp_param_ind], {}
            )
            exp_param_dict[param_name_map[exp_param_ind]] = exp_param
        prune_config_root.prune_params = {
            prune_config_utils.KEY_LAYER_PARAMS: exp_layer_params
        }

        exp_name_temp_list = []
        for e_layer_name, e_layer in exp_layer_params.items():
            e_params = [get_exp_str_from_param(e_p) for e_p in e_layer.values()]
            exp_name_temp_list.append(
                "{}-{}".format(e_layer_name, "_".join(e_params))
            )
        exp_name = "--".join(exp_name_temp_list)

        prune_out_folder_path: Text = os.path.join(out_folder_path, exp_name)

        exp_result: List = [
            original_model_name,
            exp_name,
            "",
        ]
        exp_result.extend(original_model_results.copy())
        exp_result.extend(
            run_single_experiment(
                prune_config=prune_config_root,
                prune_out_folder_path=prune_out_folder_path,
                finetuning_train_config=finetuning_train_config,
                original_model_config=original_model_config,
                evaluation_epochs_list=exp_config.evaluation_epochs,
            )
        )

        experiment_vals.append(exp_result)

        # Incrementally save experiment_vals.
        write_results_to_csv(
            experiment_vals=experiment_vals,
            out_folder_path=out_folder_path,
            datetime_string=datetime_string,
        )


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
    exp_config: exp_config_utils.ExpConfig,
    model_checkpoint_path: Text,
    out_folder_path: Text,
    model_config: Optional[model_config_utils.ModelConfig],
    datetime_string: Text,
) -> None:
    """
    For each experiment:
    - Prune network, save pruned model (pruner).
    - Get train and test accuracy for pruned model on MNIST (eval_model).
    - Fine tune model.
    - Get train and test accuracy for pruned+finetuned model on MNIST (eval_model).

    Accumulate size, train accuracy, and test accuracy for each experiment.
    Print these out in a csv (or formatted that way).
    """

    # Logging.
    logger = logging_utils.get_logger(name=LOGGER_NAME)

    # Original model.
    original_model_name: Text = os.path.basename(model_checkpoint_path)
    original_size, original_train_acc, original_test_acc = evaluate_model(
        model_path=model_checkpoint_path,
        dataset_name=exp_config.evaluation_dataset_name,
        batch_size=exp_config.evaluation_dataset_batch_size,
        model_size_type="numel",
    )
    original_model_results: List = [
        original_size,
        original_train_acc,
        original_test_acc,
    ]
    logger.info(
        PRINT_FORMAT.format(
            "original", original_size, original_train_acc, original_test_acc
        )
    )

    # Experiment results container.
    experiment_vals: List[List] = []

    try:
        prune_type: Text = exp_config.prune_type
        if prune_type == "craig":
            run_craig_experiments(
                experiment_vals=experiment_vals,
                exp_config=exp_config,
                original_model_name=original_model_name,
                original_model_path=model_checkpoint_path,
                original_model_config=model_config,
                original_model_results=original_model_results,
                out_folder_path=out_folder_path,
                datetime_string=datetime_string,
            )
        # TODO: Make Mussay compatible again.
        # elif prune_type == "mussay":
        #     run_mussay_experiments(
        #         experiment_vals=experiment_vals,
        #         prune_out_folder_root_path=prune_out_folder_root_path,
        #         original_model_name=original_model_name,
        #         original_model_path=original_model_path,
        #         original_model_config_path=original_model_config_path,
        #         original_model_train_config_path=original_model_train_config_path,
        #         original_model_results=original_model_results,
        #         evaluation_epochs_list=evaluation_epochs_list,
        #     )
        else:
            raise ValueError("prune_type not supported: {}".format(prune_type))
    finally:
        # Write results to csv.
        logger.info("writing final results...")
        out_csv_path: Text = write_results_to_csv(
            experiment_vals=experiment_vals,
            out_folder_path=out_folder_path,
            datetime_string=datetime_string,
        )
        logger.info("results written to: {}".format(out_csv_path))


### CLI


def get_args():
    parser = argparse.ArgumentParser(
        description="Run pruning experiments based on config."
    )

    parser.add_argument(
        "-e",
        "--exp_config",
        type=str,
        required=True,
        help="Path to the experiment config JSON.",
    )
    parser.add_argument(
        "-m",
        "--model_checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint to experiment with.",
    )
    parser.add_argument(
        "-o",
        "--out_folder",
        type=str,
        required=True,
        help="Path to folder where pruning experiments should be output.",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        required=False,
        default=None,
        help="Path to the model config JSON, not required.",
    )

    return parser.parse_args()


def main() -> None:
    args = get_args()

    # Logging.
    # Assuming that an existing log file means that the corresponding results file will be taken.
    datetime_string: Text
    while True:
        datetime_string = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        try:
            logging_utils.setup_logging(
                os.path.join(
                    args.out_folder,
                    "runner-log-{}.txt".format(datetime_string),
                ),
                file_mode="x",  # Use "x" to make sure this specific task id does not conflict.
            )
        except:
            # Try again
            time.sleep(random.random() * 2)  # Sleep up to 2 seconds.
            continue

        # If creation worked, exit loop.
        logger = logging_utils.get_logger(LOGGER_NAME)
        logger.info(args)
        break

    exp_config: exp_config_utils.ExpConfig = exp_config_utils.get_config_from_file(
        config_file_loc=args.exp_config
    )
    model_config: Optional[model_config_utils.ModelConfig] = None
    if args.model_config:
        model_config = model_config_utils.get_config_from_file(
            config_file_loc=args.model_config
        )

    run_experiments(
        exp_config=exp_config,
        model_checkpoint_path=args.model_checkpoint,
        out_folder_path=args.out_folder,
        model_config=model_config,
        datetime_string=datetime_string,
    )


if __name__ == "__main__":
    main()
