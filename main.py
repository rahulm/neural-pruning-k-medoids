"""Front-end to perform CRAIG neural pruning on the given neural network"""

import argparse

import craig_pruner
from utils import prune_config_utils


def get_args():
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

    return parser.parse_args()


def main() -> None:
    args = get_args()
    print(args)

    config: prune_config_utils.PruneConfig = prune_config_utils.get_config_from_file(
        args.config
    )

    craig_pruner.prune_network_with_craig(config)


if __name__ == "__main__":
    main()
