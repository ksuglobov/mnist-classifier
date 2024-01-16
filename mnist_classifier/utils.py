import os
import sys
from pathlib import Path

from mnist_classifier import logger


def find_or_create_dir(path, terminate=True):
    """
    Checks the existence of an object at the specified path.
    If the object exists and is a file, returns False.
    If the object exists and is a directory, returns True.
    If the object does not exist, creates a directory and returns True.
    """
    try:
        path = Path(path)
        if path.exists():
            if path.is_file():
                logger.debug(f"File already exists: {path}")
                return False
            elif path.is_dir():
                logger.debug(f"Directory already exists: {path}")
                return True
        else:
            os.makedirs(str(path))
            logger.debug(f"Directory was successfully created: {path}")
            return True
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {str(e)}")
        if terminate:
            sys.exit(1)
        return False


def is_valid_file(path):
    path = Path(path)
    return path.is_file() and path.exists()
