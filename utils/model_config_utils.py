"""
This is a utils file for reading of model config JSON files.
"""

from typing import Dict, Text, Tuple


class ModelConfig:
    """
    Wrapper for easy access of config fields
    """

    _raw_dict: Dict
    config_id: Text
    model_architecture: Text
    model_params: Dict
    # model_folder: Text
    # model_path: Text

    def __init__(self, config_dict: Dict) -> None:
        super().__setattr__("_raw_dict", config_dict)

    def __setattr__(self, name, value):
        self._raw_dict[name] = value

    def __getattr__(self, name):
        if name in self._raw_dict:
            return self._raw_dict[name]
        return super().__getattribute__(name)

    def __repr__(self) -> Text:
        return str(self._raw_dict)


def get_config_from_file(config_file_loc: Text) -> ModelConfig:
    import json

    config: ModelConfig
    with open(config_file_loc, "r") as config_file:
        config_dict: Dict = json.load(config_file)
        config = ModelConfig(config_dict)
    return config

