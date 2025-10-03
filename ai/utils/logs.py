"""
This module contains the general logging utility functions.
"""
import sys
import logging
from typing import Any


def get_logger() -> Any:
    """
    Returns a configured logger that writes to console and suppresses noisy third-party loggers.
    """
    logger = logging.getLogger('bookiewand')
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        formatter = logging.Formatter(
            '[*] %(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Silence s3transfer, boto3, botocore, urllib3, openai, httpcore, and httpx loggers unless explicitly needed
    for noisy_logger in [
        's3transfer', 'boto3', 'botocore', 'urllib3',
        'urllib3.connectionpool', 'openai', 'httpcore', 'httpx'
    ]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    return logger
