"""
This is a utils file for reading of training config JSON files.
"""

from typing import Dict, Text, Tuple


class TrainConfig:
    """
    Wrapper for easy access of config fields
    """

    _raw_dict: Dict
    config_id: Text
    algo_name: Text
    dataset_name: Text
    num_epochs: int
    batch_size_train: int
    batch_size_test: int
    learning_rate: float
    momentum: float
    weight_decay: float
    lr_step_size: int
    gamma: float
    random_seed: int

    def __init__(self, config_dict: Dict) -> None:
        self._raw_dict: Dict = config_dict

    def __getattr__(self, name):
        if name in self._raw_dict:
            return self._raw_dict[name]
        return super().__getattribute__(name)

    def __repr__(self) -> Text:
        return str(self._raw_dict)


def get_config_from_file(config_file_loc: Text) -> TrainConfig:
    import json

    config: TrainConfig
    with open(config_file_loc, "r") as config_file:
        config_dict: Dict = json.load(config_file)
        config = TrainConfig(config_dict)
    return config

