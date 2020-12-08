"""Performs neural pruning with CRAIG or Mussay algorithm."""

import json
import os
import random
from datetime import datetime
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

import craig
from mussay_neural_pruning import coreset
from utils import (
    general_config_utils,
    logging_utils,
    model_config_utils,
    prune_config_utils,
    train_utils,
)

LOGGER_NAME: Text = "pruner"
FILE_NAME_FORMAT_LOG: Text = "log-{}.txt"

FILE_NAME_MODEL: Text = "pruned_model.pt"
FILE_NAME_STATE_DICT: Text = "pruned_state_dict.pt"
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
    def rbf_kernel(layer, gamma=None, **kwargs):
        """
        Calculates the similarity based on the Radial Basis Function (RBF).
        In gamma, "f" stands for "# of features".
        """
        layer_weights = SimilarityMetrics.torch_layer_to_numpy_weights(layer)

        gamma_val: Optional[float] = None
        if gamma is None:
            gamma_val = None
        elif isinstance(gamma, int) or isinstance(gamma, float):
            gamma_val = float(gamma)
        elif isinstance(gamma, str):
            num_features: float = float(layer_weights.shape[1])
            if gamma == "f":
                gamma_val = float(num_features)
            elif gamma == "sqrt(f)":
                gamma_val = float(np.sqrt(num_features))
            elif gamma == "sqrt(f)^-1":
                gamma_val = float(1 / np.sqrt(num_features))
            elif gamma == "sqrt(f^-1)":
                gamma_val = float(np.sqrt(1 / num_features))
            elif gamma == "e^(-f)":
                gamma_val = float(np.exp(-num_features))
            elif gamma == "e^(f^-1)":
                gamma_val = float(np.exp(1 / num_features))
            elif gamma == "e^(-f^-1)":
                gamma_val = float(np.exp(-1 / num_features))
            elif gamma == "f^-2":
                gamma_val = float(1 / (num_features ** 2))
            elif gamma == "f^-1":  # This is the default for gamma=None.
                gamma_val = float(1 / num_features)
            elif gamma.lower() == "none":
                gamma_val = None
            else:
                try:
                    gamma_val = float(gamma)
                except:
                    raise ValueError("gamma is not valid: {}".format(gamma))
        else:
            raise ValueError("gamma is not valid: {}".format(gamma))

        return sklearn.metrics.pairwise.rbf_kernel(
            layer_weights, Y=layer_weights, gamma=gamma_val
        )


def get_layer_craig_subset(
    layer: Union[nn.Linear, nn.Conv2d],
    original_num_nodes: int,
    prune_percent_per_layer: float,
    similarity_metric: Union[Text, Dict] = "",
    prune_type: Text = "craig",
    **kwargs
) -> Tuple[List[int], List[float]]:
    logger = logging_utils.get_logger(LOGGER_NAME)

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

    target_num_nodes: int = int(
        (1 - prune_percent_per_layer) * original_num_nodes
    )

    subset_nodes: List
    subset_weights: List

    if prune_type == "random":
        subset_nodes = random.sample(
            list(range(original_num_nodes)), target_num_nodes
        )
        subset_weights = [1 for _ in subset_nodes]
    else:  # Assumes similarity_metric is set correctly.
        similarity_matrix: Any
        if isinstance(similarity_metric, dict):
            similarity_matrix = getattr(
                SimilarityMetrics, similarity_metric["name"]
            )(layer=layer, **similarity_metric)
        else:
            similarity_matrix = getattr(SimilarityMetrics, similarity_metric)(
                layer=layer
            )

        (
            subset_nodes,
            subset_weights,
            craig_time,
        ) = craig.get_craig_subset_and_weights(
            similarity_matrix=similarity_matrix, target_size=target_num_nodes
        )
        logger.info("craig runtime (s): {}".format(craig_time))

    return subset_nodes, subset_weights


def prune_fc_layer_with_craig(
    layer: nn.Linear,
    prune_percent_per_layer: float,
    similarity_metric: Union[Text, Dict] = "",
    prune_type: Text = "craig",
    **kwargs
) -> Tuple[List[int], List[float]]:

    # Get CRAIG subset.
    subset_nodes: List
    subset_weights: List
    subset_nodes, subset_weights = get_layer_craig_subset(
        layer=layer,
        original_num_nodes=layer.out_features,
        prune_percent_per_layer=prune_percent_per_layer,
        similarity_metric=similarity_metric,
        prune_type=prune_type,
        **kwargs
    )

    # Remove nodes+weights+biases, and adjust weights.
    num_nodes: int = len(subset_nodes)

    # Prune current layer.
    # Multiply weights (and biases?) by subset_weights.
    subset_weights_tensor = torch.tensor(subset_weights)
    layer.weight = nn.Parameter(
        layer.weight[subset_nodes]
        * subset_weights_tensor.reshape((num_nodes, 1))
    )
    if layer.bias is not None:
        layer.bias = nn.Parameter(
            layer.bias[subset_nodes] * subset_weights_tensor
        )
    layer.out_features = num_nodes

    return subset_nodes, subset_weights


def prune_conv2d_layer_with_craig(
    layer: nn.Conv2d,
    prune_percent_per_layer: float,
    similarity_metric: Union[Text, Dict] = "",
    prune_type: Text = "craig",
    **kwargs
) -> Tuple[List[int], List[float]]:

    # Get CRAIG subset.
    subset_nodes: List
    subset_weights: List
    subset_nodes, subset_weights = get_layer_craig_subset(
        layer=layer,
        original_num_nodes=layer.out_channels,
        prune_percent_per_layer=prune_percent_per_layer,
        similarity_metric=similarity_metric,
        prune_type=prune_type,
        **kwargs
    )

    # Remove nodes+weights+biases, and adjust weights.
    num_nodes: int = len(subset_nodes)

    # Prune current layer.
    # Multiply weights (and biases?) by subset_weights.
    subset_weights_tensor = torch.tensor(subset_weights)
    layer.weight = nn.Parameter(
        layer.weight[subset_nodes]
        * subset_weights_tensor.reshape((num_nodes, 1, 1, 1))
    )
    if layer.bias is not None:
        layer.bias = nn.Parameter(
            layer.bias[subset_nodes] * subset_weights_tensor
        )
    layer.out_channels = num_nodes

    return subset_nodes, subset_weights


LAYER_NAME_MAP: Dict[Type[nn.Module], Text] = {
    nn.Linear: prune_config_utils.KEY_LAYER_LINEAR,
    nn.Conv2d: prune_config_utils.KEY_LAYER_CONV2D,
    nn.BatchNorm2d: prune_config_utils.KEY_LAYER_BATCHNORM2D,
    nn.ReLU: "relu",
    nn.AdaptiveAvgPool2d: "adaptiveavgpool2d",
    nn.MaxPool2d: "maxpool2d",
    nn.Dropout: "dropout",
    nn.Flatten: "flatten",
}

OLD_CRAIG_LAYER_FUNCTION_MAP: Dict[
    Text, Callable[..., Tuple[List[int], List[float]]]
] = {
    prune_config_utils.KEY_LAYER_LINEAR: prune_fc_layer_with_craig,
    prune_config_utils.KEY_LAYER_CONV2D: prune_conv2d_layer_with_craig,
}

CRAIG_LAYER_FUNCTION_MAP: Dict[
    Type[nn.Module], Callable[..., Tuple[List[int], List[float]]]
] = {
    nn.Linear: prune_fc_layer_with_craig,
    nn.Conv2d: prune_conv2d_layer_with_craig,
}


def run_single_data_point(
    model: nn.Module, model_input_shape: Sequence, data_transform_name: Text
):
    input_data = (
        train_utils.DATASET_TRANSFORMS[data_transform_name](
            np.ones(model_input_shape)
        )
        .cpu()
        .float()
        .unsqueeze(0)
    )
    with torch.no_grad():
        model.eval()
        model = model.cpu()
        model(input_data)


def prune_network_with_craig(
    model: nn.Module, prune_config: prune_config_utils.PruneConfig, **kwargs
) -> None:
    """This currently assumes that all fully connected layers are directly in
    one sequence, and that there are no non-FC layers after the last FC layer
    of that sequence."""
    logger = logging_utils.get_logger(LOGGER_NAME)

    # Get params for each layer.
    layer_params: Dict = prune_config.prune_params[
        prune_config_utils.KEY_LAYER_PARAMS
    ]

    # Get list of model layers/parameters.
    model_layers: List[nn.Module] = model.ordered_unpacking
    num_layers: int = len(model_layers)
    output_layer_index: int = num_layers - 1
    model_data_shapes: List = [[] for _ in model_layers]

    # Use model input shape to get data output shape for each layer.
    def layer_shape_hook(layer_ind):
        def inner(self, input, output):
            # Discard the batch size.
            model_data_shapes[layer_ind] = output.data.shape[1:]

        return inner

    model_hooks = []
    for layer_ind, layer in enumerate(model_layers):
        model_hooks.append(
            layer.register_forward_hook(layer_shape_hook(layer_ind))
        )
    run_single_data_point(
        model=model,
        model_input_shape=prune_config.model_input_shape,
        data_transform_name=prune_config.data_transform_name,
    )
    for mhook in model_hooks:
        mhook.remove()

    curr_layer_i: int = 0
    while curr_layer_i < output_layer_index:
        # Iterate through layers, prune as necessary.
        curr_layer: nn.Module = model_layers[curr_layer_i]
        curr_layer_type: Type[nn.Module] = type(curr_layer)
        curr_layer_name: Text
        curr_layer_prune_func: Callable[..., Tuple[List[int], List[float]]]

        if (curr_layer_type in CRAIG_LAYER_FUNCTION_MAP) and (
            LAYER_NAME_MAP[curr_layer_type] in layer_params
        ):
            # If the current layer is prunable and is set up in the PruneConfig, then we can prune.
            curr_layer_name = LAYER_NAME_MAP[curr_layer_type]
            curr_layer_prune_func = CRAIG_LAYER_FUNCTION_MAP[curr_layer_type]
        else:
            # Otherwise, skip this layer.
            curr_layer_i += 1
            continue

        # Prune the current layer.
        subset_nodes: List[int]
        subset_weights: List[float]
        subset_nodes, subset_weights = curr_layer_prune_func(
            layer=curr_layer, **(layer_params[curr_layer_name])
        )
        subset_len: int = len(subset_nodes)

        next_layer_i: int = curr_layer_i + 1
        while next_layer_i < num_layers:
            # Find the next prunable layer and update the weights accordingly.
            next_layer: nn.Module = model_layers[next_layer_i]
            next_layer_type: Type[nn.Module] = type(next_layer)

            if next_layer_type not in CRAIG_LAYER_FUNCTION_MAP:
                # If this layer is not prunable, skip.
                next_layer_i += 1
                continue

            if isinstance(next_layer, nn.Conv2d):
                # Change conv in channels to match the pruned subset.
                next_layer.weight = nn.Parameter(
                    next_layer.weight[:, subset_nodes]
                )
                next_layer.in_channels = subset_len
                next_layer._in_channels = (
                    subset_len  # Not sure if this is necessary.
                )
            elif isinstance(next_layer, nn.Linear):
                # Assuming a pre-Linear flatten op, need to find the weights
                # that correspond to the channels that were kept in the pruning
                # of the previous layer.
                num_weights_per_channel: int

                if isinstance(curr_layer, nn.Conv2d):
                    # If the initially pruned layer was a conv, then re-iterate
                    # from curr_layer to next_layer, searching for the last
                    # conv/pooling/relu/etc before a flatten-esque operation.
                    for temp_i in range(curr_layer_i, next_layer_i):
                        if len(model_data_shapes[temp_i]) != 3:
                            break
                        num_weights_per_channel = int(
                            np.prod(model_data_shapes[temp_i][1:])
                        )
                else:
                    # Otherwise, the initially pruned layer must have been a
                    # linear layer. In that case, we are currently assuming
                    # that only other linear/relu/flatten/etc layers lie in
                    # between. So, we can simply use the number of original
                    # channels/features, which should be =1.
                    num_weights_per_channel = int(
                        np.prod(model_data_shapes[curr_layer_i][1:])
                    )

                weights_to_keep: List[int] = []
                for si in subset_nodes:
                    weights_to_keep.extend(
                        list(
                            range(
                                num_weights_per_channel * si,
                                num_weights_per_channel * (si + 1),
                            )
                        )
                    )
                next_layer.weight = nn.Parameter(
                    next_layer.weight[:, weights_to_keep]
                )

                next_layer.in_features = len(weights_to_keep)
            else:
                logger.warn(
                    "No pruning adjustment made to layer {} of type {}".format(
                        next_layer_i, next_layer_type
                    )
                )

            # Adjustments were attempted, now continue to the next layer for
            # pruning.
            break

        # Now that we have found the next prunable layer, we can jump to it.
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
    logger = logging_utils.get_logger(LOGGER_NAME)

    # Create output folder, if it does not exist.
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
    logger.info("Loading model checkpoint from: {}".format(model_path))
    load_location = torch.device("cpu")  # Can make this None, as default
    model = torch.load(model_path, map_location=load_location)

    with torch.no_grad():
        # Perform pruning.
        model.eval()
        logger.info(
            "Starting pruning for prune_type: {}".format(
                prune_config.prune_type
            )
        )
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
    torch.save(model, out_model_path)
    logger.info("Pruning complete")
    logger.info(model)

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
        logger.info(
            "Model architecture config not supported: {}".format(
                model_architecture
            )
        )
        return

    out_model_config_path: Text = os.path.join(
        pruned_output_folder, FILE_NAME_MODEL_CONFIG
    )
    with open(out_model_config_path, "w") as out_model_config_file:
        json.dump(out_model_config, out_model_config_file)
    logger.info("Wrote model config to: {}".format(out_model_config_path))

    out_state_dict_path: Text = os.path.join(
        pruned_output_folder, FILE_NAME_STATE_DICT
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": out_model_config,
        },
        out_state_dict_path,
    )
    logger.info(
        "Wrote model state dict with config to: {}".format(out_state_dict_path)
    )


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

    config: prune_config_utils.PruneConfig = prune_config_utils.get_config_from_file(
        args.config
    )
    pruned_output_folder: Text = (
        args.out_folder if args.out_folder else config.pruned_model_out_folder
    )

    # Logging
    datetime_string: Text = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    logging_utils.setup_logging(
        log_file_loc=os.path.join(
            pruned_output_folder, FILE_NAME_FORMAT_LOG.format(datetime_string),
        )
    )
    logger = logging_utils.get_logger(LOGGER_NAME)
    logger.info(args)

    prune_network(
        prune_config=config,
        pruned_output_folder=pruned_output_folder,
        model_checkpoint_path=args.model,
    )


if __name__ == "__main__":
    main()

