__version__ = "1.1.0"

import logging

from mnist_classifier.logger import setup_logger


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger = setup_logger(logger)
