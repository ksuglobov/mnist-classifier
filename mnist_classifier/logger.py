"""
Module logger

Module Description: This module contains functions for configuring the logger.
"""


import logging
import types


def log_newline(self, lines_cnt=1):
    # Switch handler, output a blank line
    self.removeHandler(self.console_handler)
    self.addHandler(self.blank_handler)
    for _ in range(lines_cnt):
        self.info("")

    # Switch back
    self.removeHandler(self.blank_handler)
    self.addHandler(self.console_handler)


def setup_logger(logger):
    # formatters
    console_formatter = logging.Formatter("%(levelname)s: %(message)s")
    blank_formatter = logging.Formatter(fmt="")

    # Create a handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(console_formatter)

    # Create a "blank line" handler
    blank_handler = logging.StreamHandler()
    blank_handler.setLevel(logging.DEBUG)
    blank_handler.setFormatter(blank_formatter)

    # Create a logger, with the previously-defined handler
    logger.addHandler(console_handler)

    # Save some data and add a method to logger object
    logger.console_handler = console_handler
    logger.blank_handler = blank_handler
    logger.newline = types.MethodType(log_newline, logger)

    return logger
