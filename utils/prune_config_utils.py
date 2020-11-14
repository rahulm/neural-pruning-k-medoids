"""
This is a utils file for reading of pruning config JSON files.
"""

from typing import Dict, Text, Tuple


class PruneConfig:
    """
    Wrapper for easy access of config fields
    """

    _raw_dict: Dict
    config_id: Text
    model_architecture: Text
    model_params: Dict
    model_path: Text
    prune_type: Text
    prune_params: Dict

    def __init__(self, config_dict: Dict) -> None:
        self._raw_dict: Dict = config_dict

    def __getattr__(self, name):
        if name in self._raw_dict:
            return self._raw_dict[name]
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

