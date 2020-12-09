import argparse
import itertools
from typing import Dict, List, Text

from utils import exp_config_utils, general_config_utils

import os


def create_exp_configs(original_config_path: Text):
    exp_config: exp_config_utils.ExpConfig = exp_config_utils.get_config_from_file(
        original_config_path
    )

    # For now, just explicitly search layers and params.
    to_search: Dict = exp_config.prune_params["layer_params"]
    layer_names: List[Text] = ["all", "linear", "conv2d"]
    param_names: List[Text] = [
        "similarity_metric",
        "prune_percent_per_layer",
        "prune_type",
    ]
    all_param_values: List[List] = []
    param_name_map: List[Text] = []
    layer_name_map: List[Text] = []
    for layer_name in layer_names:
        if layer_name not in to_search:
            continue
        layer_params_to_search = to_search[layer_name]
        for param_name in param_names:
            if param_name not in layer_params_to_search:
                continue
            all_param_values.append(layer_params_to_search[param_name])
            layer_name_map.append(layer_name)
            param_name_map.append(param_name)
    param_permutations = itertools.product(*all_param_values)

    original_file_name, original_file_extension = os.path.splitext(
        original_config_path
    )
    for subexp_i, param_perm in enumerate(param_permutations):
        config_layer_params: Dict = {}
        for param_i, param in enumerate(param_perm):
            config_layer_params.setdefault(layer_name_map[param_i], {})[
                param_name_map[param_i]
            ] = [param]

        exp_config.prune_params["layer_params"] = config_layer_params
        general_config_utils.write_config_to_file(
            exp_config,
            "{}.{}{}".format(
                original_file_name, subexp_i + 1, original_file_extension
            ),
        )


def get_args():
    parser = argparse.ArgumentParser(
        description="Create multiple ExpConfigs from an aggregate config."
    )

    parser.add_argument(
        "-e",
        "--exp_config",
        required=True,
        type=str,
        help="Path to original ExpConfig. Individual configs will be output in the same folder.",
    )

    return parser.parse_args()


def main():
    args = get_args()
    print(args)
    create_exp_configs(args.exp_config)


if __name__ == "__main__":
    main()
