"""Sets up logging for the project"""
import logging
import os
import random
import time
from datetime import datetime
from typing import Optional, Text


def setup_logging(log_file_loc: Optional[Text] = None, file_mode="a") -> None:
    import logging
    import sys

    handler_stdout = logging.StreamHandler(stream=sys.stdout)
    handler_stdout.setLevel(level="NOTSET")

    handler_stderr = logging.StreamHandler(stream=sys.stderr)
    handler_stderr.setLevel(level="ERROR")

    handler_log_file: logging.Handler
    if log_file_loc:
        log_dir: Text = os.path.dirname(log_file_loc)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        handler_log_file = logging.FileHandler(
            filename=log_file_loc, mode=file_mode
        )
        handler_log_file.setLevel(level="NOTSET")
    else:
        handler_log_file = logging.NullHandler()

    logging.basicConfig(
        handlers=[handler_log_file, handler_stdout, handler_stderr],
        format="{levelname:<8}{asctime}  {name:>30}:{lineno:<4}  {message}",
        datefmt="%m/%d/%Y %I:%M:%S %p",
        style="{",
        level="NOTSET",
    )


def get_logger(name: Optional[Text] = None) -> logging.Logger:
    return logging.getLogger(name=name)


def setup_unique_log_file(
    root_folder_path: Text, file_name_format: Text = "log-{}.txt"
) -> Text:
    datetime_string: Text
    while True:
        datetime_string = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        try:
            setup_logging(
                os.path.join(
                    root_folder_path, file_name_format.format(datetime_string),
                ),
                file_mode="x",  # Use "x" to make sure this specific task id does not conflict.
            )
        except:
            # Try again
            time.sleep(random.random() * 2)  # Sleep up to 2 seconds.
            continue

        # If creation worked, exit loop.
        break
    return datetime_string
