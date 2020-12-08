"""
This is a utils file for reading of experiment config JSON files.
"""

from typing import Dict, List, Sequence, Text, Tuple, Union

from . import train_config_utils

# from collections import OrderedDict


class ExpConfig:
    """
    Wrapper for easy access of config fields
    """

    _raw_dict: Dict
    config_id: Text
    prune_type: Text
    model_input_shape: Sequence
    data_transform_name: Text
    prune_params: Dict
    finetuning_train_config: train_config_utils.TrainConfig
    evaluation_dataset_name: Text
    evaluation_dataset_batch_size: int
    evaluation_epochs: List[Union[int, Text]]

    def __init__(self, config_dict: Dict) -> None:
        super().__setattr__("_raw_dict", config_dict)

    def __setattr__(self, name, value):
        self._raw_dict[name] = value

    def __getattr__(self, name):
        if name in self._raw_dict:
            val = self._raw_dict[name]
            if name == "finetuning_train_config":
                return train_config_utils.TrainConfig(val)
            else:
                return val
        return super().__getattribute__(name)

    def __repr__(self) -> Text:
        return str(self._raw_dict)


def get_config_from_file(config_file_loc: Text) -> ExpConfig:
    import json

    config: ExpConfig
    with open(config_file_loc, "r") as config_file:
        # config_dict: Dict = json.load(
        #     config_file, object_pairs_hook=OrderedDict
        # )
        config_dict: Dict = json.load(config_file)
        config = ExpConfig(config_dict)
    return config

