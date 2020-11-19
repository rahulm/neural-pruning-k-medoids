"""Performs CRAIG neural pruning"""

import json
import os
from typing import Any, Dict, List, Set, Text

import numpy as np
import sklearn.metrics
import torch
from torch import nn

import craig
from utils import general_config_utils, model_config_utils, prune_config_utils

FILE_NAME_MODEL: Text = "pruned_model.pth"
FILE_NAME_WEIGHT_ONLY: Text = "pruned_weight_only.pth"
FILE_NAME_MODEL_CONFIG: Text = "config-model-pruned_model.json"
FILE_NAME_PRUNE_CONFIG: Text = "config-prune.json"


class SimilarityMetrics:
    @staticmethod
    def weights_covariance(layer, **kwargs):
        """Calculate covariance. Each row corresponds to a node."""
        layer_weights = layer.weight.cpu().detach().numpy()
        node_covariance = np.cov(layer_weights)
        return node_covariance

    @staticmethod
    def euclidean_distance(layer, **kwargs):
        """
        Calculates euclidean distance of nodes, treats weights as coordinates.
        """
        layer_weights = layer.weight.cpu().detach().numpy()
        dists = sklearn.metrics.pairwise_distances(
            layer_weights, metric="euclidean", n_jobs=-1
        )
        return np.max(dists) - dists

    @staticmethod
    def cosine_similarity(layer, **kwargs):
        """
        Calculates the cosine similarity of the nodes in a layer.
        """
        layer_weights = layer.weight.cpu().detach().numpy()
        dists = sklearn.metrics.pairwise_distances(
            layer_weights, metric="cosine", n_jobs=-1
        )
        return 1 - dists

    @staticmethod
    def l1_norm(layer, **kwargs):
        """
        Calculates the L1 norm distance of the nodes in a layer.
        """
        layer_weights = layer.weight.cpu().detach().numpy()
        dists = sklearn.metrics.pairwise_distances(
            layer_weights, metric="l1", n_jobs=-1
        )
        return np.max(dists) - dists

    @staticmethod
    def rbf_kernel(layer, **kwargs):
        """
        Calculates the similarity based on the Radial Basis Function (RBF).
        """
        layer_weights = layer.weight.cpu().detach().numpy()
        return sklearn.metrics.pairwise.rbf_kernel(layer_weights, layer_weights)


def prune_fc_layer_with_craig(
    curr_layer,
    next_layer,
    prune_percent_per_layer: float,
    similarity_metric: Text,
) -> None:
    assert (0 <= prune_percent_per_layer) and (
        prune_percent_per_layer <= 1
    ), "prune_percent_per_layer ({}) must be within [0,1]".format(
        prune_percent_per_layer
    )

    original_num_nodes: int = curr_layer.out_features
    original_nodes = list(range(original_num_nodes))

    # TODO: Instead of percent of nodes, maybe get all weights from CRAIG and take top percent?
    target_num_nodes: int = int(
        (1 - prune_percent_per_layer) * original_num_nodes
    )

    # Maybe make this generic to support different metrics and args.
    similarity_matrix = getattr(SimilarityMetrics, similarity_metric)(
        curr_layer
    )

    facility_location: craig.FacilityLocation = craig.FacilityLocation(
        D=similarity_matrix, V=original_nodes
    )

    subset_nodes, subset_weights = craig.lazy_greedy_heap(
        F=facility_location, V=original_nodes, B=target_num_nodes
    )
    subset_weights_tensor = torch.tensor(subset_weights)

    # Remove nodes+weights+biases (in both curr_layer and next_layer), and adjust weights.
    num_nodes: int = len(subset_nodes)

    # Prune current layer.
    # Multiply weights (and biases?) by subset_weights.
    curr_layer.weight = nn.Parameter(
        curr_layer.weight[subset_nodes] * subset_weights_tensor[:, None]
    )
    if curr_layer.bias is not None:
        curr_layer.bias = nn.Parameter(
            curr_layer.bias[subset_nodes] * subset_weights_tensor
        )
    curr_layer.out_features = num_nodes

    # Prune removed nodes from the next layer.
    next_layer.weight = nn.Parameter(next_layer.weight[:, subset_nodes])
    next_layer.in_features = num_nodes

    # print(">>>>>")
    # print(curr_layer.weight.shape)
    # print(curr_layer.bias.shape)
    # print(next_layer.weight.shape)
    # print(next_layer.bias.shape)


def prune_network_with_craig(
    prune_config: prune_config_utils.PruneConfig, pruned_output_folder: Text
) -> None:
    """This currently assumes that all fully connected layers are directly in
    one sequence, and that there are no non-FC layers after the last FC layer
    of that sequence."""

    if not os.path.exists(pruned_output_folder):
        os.makedirs(pruned_output_folder)

    # Save original prune config.
    general_config_utils.write_config_to_file(
        prune_config, os.path.join(pruned_output_folder, FILE_NAME_PRUNE_CONFIG)
    )

    model_path: Text = prune_config.original_model_path
    load_location = torch.device("cpu")  # Can make this None, as default
    model = torch.load(model_path, map_location=load_location)

    # Ignore non-FC layers, following assumption in docstring.
    fc_layers: List = [
        layer
        for layer in model.sequential_module
        if isinstance(layer, nn.Linear)
    ]

    # Prune the out_features for each layer, except the output (last) layer.
    for layer_i in range(len(fc_layers) - 1):
        curr_layer = fc_layers[layer_i]
        next_layer = fc_layers[layer_i + 1]
        prune_fc_layer_with_craig(
            curr_layer=curr_layer,
            next_layer=next_layer,
            prune_percent_per_layer=prune_config.prune_params[
                "prune_percent_per_layer"
            ],
            similarity_metric=prune_config.prune_params["similarity_metric"],
        )

    out_model_path: Text = os.path.join(pruned_output_folder, FILE_NAME_MODEL)
    out_weights_path: Text = os.path.join(
        pruned_output_folder, FILE_NAME_WEIGHT_ONLY
    )
    torch.save(model, out_model_path)
    torch.save(model.state_dict(), out_weights_path)
    print(model)

    # Save new model config.
    # TODO: Support more model architectures.
    out_model_config_path: Text = os.path.join(
        pruned_output_folder, FILE_NAME_MODEL_CONFIG
    )
    model_architecture = model.ARCHITECTURE_NAME
    out_model_config: Dict
    if model_architecture == "fc_2":
        out_model_config = {
            "model_architecture": "fc_2",
            "model_params": {
                "input_shape": [28, 28],
                "layer_1_dim": fc_layers[0].out_features,
                "layer_2_dim": fc_layers[1].out_features,
                "output_dim": 10,
            },
        }
    elif model_architecture == "fc_classifier":
        out_model_config = {
            "model_architecture": "fc_classifier",
            "model_params": {
                "input_shape": [28, 28],
                "layers": [l.out_features for l in fc_layers[:-1]],
                "output_dim": 10,
            },
        }
    else:
        # Not supported.
        return
    with open(out_model_config_path, "w") as out_model_config_file:
        json.dump(out_model_config, out_model_config_file)


### CLI


def get_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Prune the given neural network with CRAIG."
    )

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to pruning config JSON file.",
    )

    parser.add_argument(
        "-o",
        "--out_folder",
        type=str,
        required=False,
        default=None,
        help="Folder where output should be written. Overrides path from prune config.",
    )

    return parser.parse_args()


def main() -> None:
    args = get_args()
    print(args)

    config: prune_config_utils.PruneConfig = prune_config_utils.get_config_from_file(
        args.config
    )

    pruned_output_folder: Text = (
        args.out_folder if args.out_folder else config.pruned_model_out_folder
    )

    prune_network_with_craig(config, pruned_output_folder)


if __name__ == "__main__":
    main()

