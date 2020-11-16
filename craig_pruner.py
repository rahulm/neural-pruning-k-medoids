"""Performs CRAIG neural pruning"""

import json
import os
from typing import Any, Dict, List, Set, Text

import numpy as np
import torch
from torch import nn

import craig
from utils import model_config_utils, prune_config_utils

FILE_NAME_MODEL: Text = "model.pth"
FILE_NAME_WEIGHT_ONLY: Text = "weight_only.pth"
FILE_NAME_MODEL_CONFIG: Text = "model_config.json"


class SimilarityMetrics:
    # TODO: Maybe make this a dict from name to method
    SUPPORTED_SIMILARITY_METRICS: Set = set(["weights_covariance"])

    @staticmethod
    def weights_covariance(layer):
        # Calculate covariance. Each row corresponds to a node.
        layer_weights = layer.weight
        node_covariance = np.cov(layer_weights.cpu().detach().numpy())
        return node_covariance


def prune_fc_layer_with_craig(
    curr_layer,
    next_layer,
    prune_percent_per_layer: float,
    similarity_metric="weights_covariance",
) -> None:
    assert (0 <= prune_percent_per_layer) and (
        prune_percent_per_layer <= 1
    ), "prune_percent_per_layer ({}) must be within [0,1]".format(
        prune_percent_per_layer
    )

    assert (
        similarity_metric in SimilarityMetrics.SUPPORTED_SIMILARITY_METRICS
    ), "similarity_metric ({}) must be within: {}".format(
        similarity_metric, SimilarityMetrics.SUPPORTED_SIMILARITY_METRICS
    )

    original_num_nodes: int = curr_layer.out_features
    original_nodes = list(range(original_num_nodes))

    # TODO: Instead of percent of nodes, maybe get all weights from CRAIG and take top percent?
    target_num_nodes: int = int(
        (1 - prune_percent_per_layer) * original_num_nodes
    )

    # TODO: Maybe make this generic to support different metrics and args.
    similarity_matrix = SimilarityMetrics.weights_covariance(layer=curr_layer)

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
    prune_config: prune_config_utils.PruneConfig,
) -> None:
    """This currently assumes that all fully connected layers are directly in
    one sequence, and that there are no non-FC layers after the last FC layer
    of that sequence."""

    load_location = None  # Can make this cpu or cuda
    original_model: Any

    model_config: model_config_utils.ModelConfig = prune_config.model_config

    # Currently, only supporting fc_2
    if model_config.model_architecture == "fc_2":
        from models.fc_2 import Model

        model_path: Text = os.path.join(
            model_config.model_folder, FILE_NAME_MODEL
        )
        original_model = torch.load(model_path, map_location=load_location)
    else:
        print(
            "Model architecture is not supported: {}".format(
                model_config.model_architecture
            )
        )
        return

    # Ignore non-FC layers, following assumption in docstring.
    fc_layers: List = [
        layer
        for layer in original_model.sequential_module
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

    # Save newly pruned model+weights.
    pruned_output_folder: Text = prune_config.pruned_model_out_folder
    if not os.path.exists(pruned_output_folder):
        os.makedirs(pruned_output_folder)

    out_model_path: Text = os.path.join(pruned_output_folder, FILE_NAME_MODEL)
    out_weights_path: Text = os.path.join(
        pruned_output_folder, FILE_NAME_WEIGHT_ONLY
    )
    torch.save(original_model, out_model_path)
    torch.save(original_model.state_dict(), out_weights_path)

    # Save model config.
    # TODO: Support more model architectures.
    out_model_config_path: Text = os.path.join(
        pruned_output_folder, FILE_NAME_MODEL_CONFIG
    )
    if model_config.model_architecture == "fc_2":
        out_model_config: Dict = {
            "model_architecture": "fc_2",
            "model_folder": pruned_output_folder,
            "model_params": {
                "input_shape": [28, 28],
                "layer_1_dim": fc_layers[0].out_features,
                "layer_2_dim": fc_layers[1].out_features,
                "output_dim": 10,
            },
        }
        with open(out_model_config_path, "w") as out_model_config_file:
            json.dump(out_model_config, out_model_config_file)

    print(original_model)

