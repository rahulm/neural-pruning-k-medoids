"""
Some general config utils
TODO: May be able to make a generic config reader.
TODO: Maybe make a base class for the configs?
"""

import json
import os
from typing import Text


def write_config_to_file(config, out_file: Text, makedirs: bool = True) -> None:
    if makedirs:
        out_dir: Text = os.path.dirname(out_file)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    with open(out_file, "w") as config_out_file:
        json.dump(config._raw_dict, config_out_file)

