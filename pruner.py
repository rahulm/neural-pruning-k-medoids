"""Performs neural pruning with CRAIG or Mussay algorithm."""

import json
import os
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Text,
    Tuple,
    Type,
    Union,
)

import numpy as np
import sklearn.metrics
import torch
from torch import nn
import random

import craig
from mussay_neural_pruning import coreset
from utils import general_config_utils, model_config_utils, prune_config_utils

FILE_NAME_MODEL: Text = "pruned_model.pth"
FILE_NAME_WEIGHT_ONLY: Text = "pruned_weight_only.pth"
FILE_NAME_MODEL_CONFIG: Text = "config-model-pruned_model.json"
FILE_NAME_PRUNE_CONFIG: Text = "config-prune.json"


class SimilarityMetrics:
    @staticmethod
    def torch_layer_to_numpy_weights(torch_layer) -> np.ndarray:
        """Helper function, flattens each node weight into an numpy array."""
        layer_weights = torch_layer.weight.cpu().detach().numpy()
        return np.reshape(layer_weights, (layer_weights.shape[0], -1))

    @staticmethod
    def weights_covariance(layer, **kwargs):
        """Calculate covariance. Each row corresponds to a node."""
        layer_weights = SimilarityMetrics.torch_layer_to_numpy_weights(layer)
        node_covariance = np.cov(layer_weights)
        return node_covariance

    @staticmethod
    def euclidean_distance(layer, **kwargs):
        """
        Calculates euclidean distance of nodes, treats weights as coordinates.
        """
        layer_weights = SimilarityMetrics.torch_layer_to_numpy_weights(layer)
        dists = sklearn.metrics.pairwise_distances(
            layer_weights, metric="euclidean", n_jobs=-1
        )
        return np.max(dists) - dists

    @staticmethod
    def cosine_similarity(layer, **kwargs):
        """
        Calculates the cosine similarity of the nodes in a layer.
        """
        layer_weights = SimilarityMetrics.torch_layer_to_numpy_weights(layer)
        dists = sklearn.metrics.pairwise_distances(
            layer_weights, metric="cosine", n_jobs=-1
        )
        return 1 - dists

    @staticmethod
    def l1_norm(layer, **kwargs):
        """
        Calculates the L1 norm distance of the nodes in a layer.
        """
        layer_weights = SimilarityMetrics.torch_layer_to_numpy_weights(layer)
        dists = sklearn.metrics.pairwise_distances(
            layer_weights, metric="l1", n_jobs=-1
        )
        return np.max(dists) - dists

    @staticmethod
    def rbf_kernel(layer, **kwargs):
        """
        Calculates the similarity based on the Radial Basis Function (RBF).
        """
        layer_weights = SimilarityMetrics.torch_layer_to_numpy_weights(layer)
        return sklearn.metrics.pairwise.rbf_kernel(layer_weights, layer_weights)


def prune_fc_layer_with_craig(
    curr_layer: nn.Linear,
    prune_percent_per_layer: float,
    similarity_metric: Text = "",
    prune_type: Text = "craig",
    **kwargs
) -> Tuple[List[int], List[float]]:
    assert (0 <= prune_percent_per_layer) and (
        prune_percent_per_layer <= 1
    ), "prune_percent_per_layer ({}) must be within [0,1]".format(
        prune_percent_per_layer
    )

    assert prune_type in (
        "craig",
        "random",
    ), "prune_type must be 'craig' or 'random'"

    assert (prune_type == "random") or (
        similarity_metric
    ), "similarity_metric must be set for prune_type '{}'".format(prune_type)

    original_num_nodes: int = curr_layer.out_features
    original_nodes = list(range(original_num_nodes))
    target_num_nodes: int = int(
        (1 - prune_percent_per_layer) * original_num_nodes
    )

    subset_nodes: List
    subset_weights: List

    if prune_type == "random":
        subset_nodes = random.sample(original_nodes, target_num_nodes)
        subset_weights = [1 for _ in subset_nodes]
    else:  # Assumes similarity_metric is set correctly.
        similarity_matrix = getattr(SimilarityMetrics, similarity_metric)(
            curr_layer
        )
        facility_location: craig.FacilityLocation = craig.FacilityLocation(
            D=similarity_matrix, V=original_nodes
        )

        subset_nodes, subset_weights = craig.lazy_greedy_heap(
            F=facility_location, V=original_nodes, B=target_num_nodes
        )

    # Remove nodes+weights+biases, and adjust weights.
    num_nodes: int = len(subset_nodes)

    # Prune current layer.
    # Multiply weights (and biases?) by subset_weights.
    subset_weights_tensor = torch.tensor(subset_weights)
    curr_layer.weight = nn.Parameter(
        curr_layer.weight[subset_nodes]
        * subset_weights_tensor.reshape((num_nodes, 1))
    )
    if curr_layer.bias is not None:
        curr_layer.bias = nn.Parameter(
            curr_layer.bias[subset_nodes] * subset_weights_tensor
        )
    curr_layer.out_features = num_nodes

    return subset_nodes, subset_weights


def prune_conv2d_layer_with_craig(
    curr_layer: nn.Conv2d,
    prune_percent_per_layer: float,
    similarity_metric: Text = "",
    prune_type: Text = "craig",
    **kwargs
) -> Tuple[List[int], List[float]]:
    assert (0 <= prune_percent_per_layer) and (
        prune_percent_per_layer <= 1
    ), "prune_percent_per_layer ({}) must be within [0,1]".format(
        prune_percent_per_layer
    )

    assert prune_type in (
        "craig",
        "random",
    ), "prune_type must be 'craig' or 'random'"

    assert (prune_type == "random") or (
        similarity_metric
    ), "similarity_metric must be set for prune_type '{}'".format(prune_type)

    original_num_nodes: int = curr_layer.out_channels
    original_nodes = list(range(original_num_nodes))
    target_num_nodes: int = int(
        (1 - prune_percent_per_layer) * original_num_nodes
    )

    subset_nodes: List
    subset_weights: List

    if prune_type == "random":
        subset_nodes = random.sample(original_nodes, target_num_nodes)
        subset_weights = [1 for _ in subset_nodes]
    else:  # Assumes similarity_metric is set correctly.
        similarity_matrix = getattr(SimilarityMetrics, similarity_metric)(
            curr_layer
        )
        facility_location: craig.FacilityLocation = craig.FacilityLocation(
            D=similarity_matrix, V=original_nodes
        )

        subset_nodes, subset_weights = craig.lazy_greedy_heap(
            F=facility_location, V=original_nodes, B=target_num_nodes
        )

    # Remove nodes+weights+biases, and adjust weights.
    num_nodes: int = len(subset_nodes)

    # Prune current layer.
    # Multiply weights (and biases?) by subset_weights.
    subset_weights_tensor = torch.tensor(subset_weights)
    curr_layer.weight = nn.Parameter(
        curr_layer.weight[subset_nodes]
        * subset_weights_tensor.reshape((num_nodes, 1, 1, 1))
    )
    if curr_layer.bias is not None:
        curr_layer.bias = nn.Parameter(
            curr_layer.bias[subset_nodes] * subset_weights_tensor
        )
    curr_layer.out_channels = num_nodes

    return subset_nodes, subset_weights


LAYER_TYPE_MAP: Dict[Type[nn.Module], Text] = {
    nn.Linear: prune_config_utils.KEY_LAYER_LINEAR,
    nn.Conv2d: prune_config_utils.KEY_LAYER_CONV2D,
    nn.BatchNorm2d: prune_config_utils.KEY_LAYER_BATCHNORM2D,
    nn.ReLU: "relu",
    nn.AdaptiveAvgPool2d: "adaptiveavgpool2d",
    nn.MaxPool2d: "maxpool2d",
    nn.Dropout: "dropout",
}

CRAIG_LAYER_FUNCTION_MAP: Dict[
    Text, Callable[..., Tuple[List[int], List[float]]]
] = {
    prune_config_utils.KEY_LAYER_LINEAR: prune_fc_layer_with_craig,
    prune_config_utils.KEY_LAYER_CONV2D: prune_conv2d_layer_with_craig,
    # prune_config_utils.KEY_LAYER_BATCHNORM2D: None,
}


def prune_network_with_craig(
    model: nn.Module, prune_config: prune_config_utils.PruneConfig, **kwargs
) -> None:
    """This currently assumes that all fully connected layers are directly in
    one sequence, and that there are no non-FC layers after the last FC layer
    of that sequence."""
    """
    NOTE:   Across each pooling layer, the number of channels stays
            constant (I think); this is why conv layers on either side of
            a pooling layer can share the same out_channels and in_channels
            values, respectively, without worrying about the actual height
            and width. (I think this is the same for both MaxPool and AvgPool).
            So, when we reach something like a fully connected layer, which
            requires a fixed input size, we have to use a pooling layer
            that outputs a constant size tensor, like the AdaptiveAvgPool2d.
            Since in the previous conv layer we would have pruned only the
            number of out_channels, then we can assume that the output of
            the AdaptiveAvgPool2d layer has the same number of pruned
            channels.
            So, based on the way the post-pooling pre-fc flatten works, we
            can find the subset of channels that were not pruned, and only
            keep the corresponding weights in the fc network.
    """

    prune_params: Dict = prune_config.prune_params
    layer_params: Dict = prune_params[prune_config_utils.KEY_LAYER_PARAMS]

    model_prunable_parameters: Any  # This should be an iterable
    if hasattr(model, "prunable_parameters_ordered"):
        model_prunable_parameters = model.prunable_parameters_ordered
    else:
        model_prunable_parameters = model.sequential_module

    num_prunable_parameters: int = len(model_prunable_parameters)
    output_layer_index: int = num_prunable_parameters - 1
    curr_layer_i: int = 0

    while curr_layer_i < output_layer_index:
        curr_layer: nn.Module = model_prunable_parameters[curr_layer_i]
        curr_layer_type: Text = LAYER_TYPE_MAP[type(curr_layer)]

        if (curr_layer_type not in CRAIG_LAYER_FUNCTION_MAP) or (
            curr_layer_type not in layer_params
        ):
            # If the curr_layer is not prunable, or not configured, skip.
            curr_layer_i += 1
            continue

        # curr_layer is prunable, prune.
        subset_nodes: List[int]
        subset_weights: List[float]
        subset_nodes, subset_weights = CRAIG_LAYER_FUNCTION_MAP[
            curr_layer_type
        ](curr_layer=curr_layer, **(layer_params[curr_layer_type]))

        # Save current layer output shape (or required values) for future use.
        # NOTE: Maybe add an input_shape config parameter to calculate output size?
        previous_shape: Sequence[int]
        previous_shape_is_constant: bool
        if curr_layer_type == prune_config_utils.KEY_LAYER_LINEAR:
            previous_shape = [curr_layer.out_features]
            previous_shape_is_constant = True
        elif curr_layer_type == prune_config_utils.KEY_LAYER_CONV2D:
            previous_shape = [curr_layer.out_channels]
            previous_shape_is_constant = False

        # Find the next prunable layer to update weights.
        next_layer_i: int = curr_layer_i + 1
        while next_layer_i < num_prunable_parameters:
            next_layer: nn.Module = model_prunable_parameters[next_layer_i]
            next_layer_type: Text = LAYER_TYPE_MAP[type(next_layer)]

            # If next_layer is prunable, then remove unused nodes.
            if next_layer_type in CRAIG_LAYER_FUNCTION_MAP:
                if isinstance(next_layer, nn.Linear):
                    if (not previous_shape_is_constant) or (not previous_shape):
                        raise ValueError(
                            "Shape of inputs must be constant for FC layer: {}".format(
                                next_layer
                            )
                        )

                    # NOTE: Currently, only supporting CNNs with explicitly
                    #       constant size inputs to each FC layer. May need to
                    #       figure out how to support inputs that are
                    #       implicitly constant, based on some model input size.

                    # Assumes input uses torch.flatten conventions.
                    # Keeps channels that were part of the subset_nodes.
                    mapping_multiplier: int = int(
                        np.prod(previous_shape[1:])
                    )  # H*W, or 1
                    mapped_nodes: List[int] = []
                    for node in subset_nodes:
                        map_base: int = node * mapping_multiplier
                        for i in range(mapping_multiplier):
                            mapped_nodes.append(map_base + i)

                    num_nodes = len(mapped_nodes)
                    next_layer.weight = nn.Parameter(
                        next_layer.weight[:, mapped_nodes]
                    )
                    next_layer.in_features = num_nodes
                elif isinstance(next_layer, nn.Conv2d):
                    # NOTE: Currently, only supporting conv layers followed by
                    #       conv layers, since that is the most common. So,
                    #       this assumes that the number of out_channels does
                    #       not change over pooling layers, batchnorm,
                    #       dropout, etc.
                    #       Need to add support for linear (or other) layers
                    #       followed by conv layers, where the number of
                    #       channels changes.
                    num_nodes = len(subset_nodes)
                    next_layer.weight = nn.Parameter(
                        next_layer.weight[:, subset_nodes]
                    )
                    next_layer.in_channels = num_nodes
                    next_layer._in_channels = (
                        num_nodes  # Not sure if this is needed.
                    )
                else:
                    raise TypeError(
                        "Pruning of layer not supported: {}".format(
                            type(next_layer)
                        )
                    )
                break

            # Otherwise, not prunable:
            # - Update subset/previous_shape as needed.
            # - Continue searching.

            # Update subset/previous_shape as needed.
            # NOTE: Should not see any prunable layers here.
            if next_layer_type == "adaptiveavgpool2d":
                if not previous_shape:
                    raise ValueError(
                        "previous_shape must be non-empty before adaptiveavgpool2d, found: {}".format(
                            previous_shape
                        )
                    )
                next_shape: Sequence[int] = [
                    previous_shape[0],
                    next_layer.output_size[0],
                    next_layer.output_size[1],
                ]
                previous_shape = next_shape

                previous_shape_is_constant = True

            # Continue searching.
            # curr_layer_i = next_layer_i
            next_layer_i += 1

        # Update curr_layer_i pointer before continuing.
        curr_layer_i = next_layer_i


def prune_network_with_mussay(
    model: nn.Module,
    prune_config: prune_config_utils.PruneConfig,
    torch_device: torch.device,
    **kwargs
) -> None:
    """This currently assumes that all fully connected layers are directly in
    one sequence, and that there are no non-FC layers after the last FC layer
    of that sequence."""

    # Prune params.
    prune_params: Dict = prune_config.prune_params

    # Ignore non-FC layers, following assumption in docstring.
    fc_layers: List[nn.Linear] = [
        layer
        for layer in model.sequential_module  # type: ignore
        if isinstance(layer, nn.Linear)
    ]

    # Prune the out_features for each layer, except the output (last) layer.
    for layer_i in range(len(fc_layers) - 1):
        curr_layer = fc_layers[layer_i]
        next_layer = fc_layers[layer_i + 1]

        # Get compressed layer weights and biases.
        layer_1, layer_2, compressed_node_set = coreset.compress_fc_layer(
            layer1=(curr_layer.weight, curr_layer.bias),
            layer2=(next_layer.weight, next_layer.bias),
            device=torch_device,
            activation=torch.functional.F.relu,  # TODO: Make this changeable via the config.
            compressed_size=(
                (
                    1
                    - prune_params[
                        prune_config_utils.KEY_PARAM_PRUNE_PERCENT_PER_LAYER
                    ]
                )
                * curr_layer.out_features
            ),
            upper_bound=prune_params["upper_bound"],
            compression_type=prune_params["compression_type"],
        )

        num_compressed_nodes: int = len(compressed_node_set)
        curr_layer.weight = nn.Parameter(layer_1[0])
        curr_layer.bias = nn.Parameter(layer_1[1])
        curr_layer.out_features = num_compressed_nodes

        next_layer.weight = nn.Parameter(layer_2[0])
        next_layer.bias = nn.Parameter(layer_2[1])
        next_layer.in_features = num_compressed_nodes


def prune_network(
    prune_config: prune_config_utils.PruneConfig,
    pruned_output_folder: Text,
    model_checkpoint_path: Optional[Text] = None,
    **kwargs
) -> None:
    """
    Can provide a model_checkpoint_path to override any model checkpoint path
    specified in prune_config. 
    """

    if not os.path.exists(pruned_output_folder):
        os.makedirs(pruned_output_folder)

    # Save original prune config.
    general_config_utils.write_config_to_file(
        prune_config, os.path.join(pruned_output_folder, FILE_NAME_PRUNE_CONFIG)
    )

    # Load model.
    model_path: Text
    if model_checkpoint_path:
        model_path = model_checkpoint_path
        prune_config.original_model_path = model_checkpoint_path
    else:
        model_path = prune_config.original_model_path
    load_location = torch.device("cpu")  # Can make this None, as default
    model = torch.load(model_path, map_location=load_location)

    with torch.no_grad():
        # Perform pruning.
        if prune_config.prune_type == "craig":
            model.to(torch.device("cpu"))
            prune_network_with_craig(
                model=model, prune_config=prune_config, **kwargs
            )
        elif prune_config.prune_type == "mussay":
            torch_device: torch.device = torch.device("cpu")
            prune_network_with_mussay(
                model=model,
                prune_config=prune_config,
                torch_device=torch_device,
                **kwargs
            )
        else:
            raise ValueError(
                "prune_type not supported: {}".format(prune_config.prune_type)
            )

    # Save pruned model.
    out_model_path: Text = os.path.join(pruned_output_folder, FILE_NAME_MODEL)
    out_weights_path: Text = os.path.join(
        pruned_output_folder, FILE_NAME_WEIGHT_ONLY
    )
    torch.save(model, out_model_path)
    # torch.save(model.state_dict(), out_weights_path)
    print(model)

    # Save new model config.
    model_architecture = model.ARCHITECTURE_NAME
    out_model_config: Dict
    if model_architecture == "vgg":
        out_model_config = {
            "model_architecture": "vgg",
            "model_params": {
                "vgg_version": model.vgg_version,
                "num_classes": model.num_classes,
                "pretrained_imagenet": getattr(
                    model, "pretrained_imagenet", False
                ),
            },
        }
    elif model_architecture == "fc_classifier":
        fc_layers = [
            layer
            for layer in model.sequential_module
            if isinstance(layer, nn.Linear)
        ]
        out_model_config = {
            "model_architecture": "fc_classifier",
            "model_params": {
                "input_shape": [28, 28],
                "layers": [l.out_features for l in fc_layers[:-1]],
                "output_dim": 10,
            },
        }
    elif model_architecture == "fc_2":
        fc_layers = [
            layer
            for layer in model.sequential_module
            if isinstance(layer, nn.Linear)
        ]
        out_model_config = {
            "model_architecture": "fc_2",
            "model_params": {
                "input_shape": [28, 28],
                "layer_1_dim": fc_layers[0].out_features,
                "layer_2_dim": fc_layers[1].out_features,
                "output_dim": 10,
            },
        }
    else:
        # Not supported.
        return
    out_model_config_path: Text = os.path.join(
        pruned_output_folder, FILE_NAME_MODEL_CONFIG
    )
    with open(out_model_config_path, "w") as out_model_config_file:
        json.dump(out_model_config, out_model_config_file)


### CLI


def get_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Prune the given neural network with CRAIG or Mussay."
    )

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to pruning config JSON file.",
    )

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=False,
        default=None,
        help="Path to model checkpoint. Overrides path from prune config.",
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

    prune_network(
        prune_config=config,
        pruned_output_folder=pruned_output_folder,
        model_checkpoint_path=args.model,
    )


if __name__ == "__main__":
    main()

