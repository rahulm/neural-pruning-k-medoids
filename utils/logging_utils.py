"""Sets up logging for the project"""
import logging
import os
from typing import Optional, Text


def setup_logging(log_file_loc: Optional[Text] = None) -> None:
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

        handler_log_file = logging.FileHandler(filename=log_file_loc, mode="a")
        handler_log_file.setLevel(level="NOTSET")
    else:
        handler_log_file = logging.NullHandler()

    logging.basicConfig(
        handlers=[handler_log_file, handler_stdout, handler_stderr],
        format="{levelname:<8}{asctime}  {name:>30}:{lineno:<4}  {message}",
        style="{",
        level="NOTSET",
    )


def get_logger(name: Optional[Text] = None) -> logging.Logger:
    return logging.getLogger(name=name)
