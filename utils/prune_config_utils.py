"""
This is a utils file for reading of pruning config JSON files.
"""

from typing import Dict, Text, Tuple

from . import model_config_utils


class PruneConfig:
    """
    Wrapper for easy access of config fields
    """

    _raw_dict: Dict
    config_id: Text
    model_config: model_config_utils.ModelConfig
    prune_type: Text
    prune_params: Dict
    original_model_path: Text
    pruned_model_out_folder: Text

    def __init__(self, config_dict: Dict) -> None:
        super().__setattr__("_raw_dict", config_dict)

    def __setattr__(self, name, value):
        self._raw_dict[name] = value

    def __getattr__(self, name):
        if name in self._raw_dict:
            val = self._raw_dict[name]
            if name == "model_config":
                return model_config_utils.ModelConfig(val)
            else:
                return val
        return super().__getattribute__(name)

    def __repr__(self) -> Text:
        return str(self._raw_dict)


def get_config_from_file(config_file_loc: Text) -> PruneConfig:
    import json

    config: PruneConfig
    with open(config_file_loc, "r") as config_file:
        config_dict: Dict = json.load(config_file)
        config = PruneConfig(config_dict)
    return config

