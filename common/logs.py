"""This module contains the general logging utility functions."""

import logging
import os
import sys
from enum import IntEnum
from logging import Logger

from dotenv import load_dotenv

load_dotenv()


class LogLevel(IntEnum):
    """Enum representing different log levels, used to convert .env to logging levels."""

    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


def get_logger() -> Logger:
    """Returns a configured logger using log level from .env."""
    log_level = os.environ["LOG_LEVEL"]
    logger = logging.getLogger("bookiewand")
    logger.setLevel(LogLevel[log_level].value)
    logger.propagate = False

    if not logger.handlers:
        formatter = logging.Formatter(
            "[*] %(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(LogLevel[log_level].value)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
