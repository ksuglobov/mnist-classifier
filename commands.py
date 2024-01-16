import argparse
import logging
import sys

from mnist_classifier import logger
from mnist_classifier.infer import infer as inference
from mnist_classifier.train import train as training
from mnist_classifier.utils import find_or_create_dir, is_valid_file


def positive_int(value):
    """
    Check if the value is a positive integer number.
    """
    try:
        value = int(value)
        if value <= 0:
            raise ValueError
        return value
    except ValueError:
        logger.error(f"Invalid positive int value: {value}")


def main():
    # command line arguments parser
    parser = argparse.ArgumentParser(
        description="Simple MNIST image classifier using the LeNet-5 neural network"
    )

    # common args
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "--log_level",
        choices=["debug", "info", "warning"],
        default="info",
        help="Logging level (default: %(default)s)",
    )

    subparsers = parser.add_subparsers(dest="mode", help="Mode of operation")
    subparsers.required = True

    # ========== train mode ==========
    train_parser = subparsers.add_parser(
        "train", help="Run training", parents=[parent_parser]
    )
    train_parser.add_argument(
        "--path2model_dir",
        type=str,
        default="models",
        help="Path to directory for saving model",
    )
    train_parser.add_argument(
        "--n_epochs",
        type=positive_int,
        default=5,
        help="Number of epochs (default: %(default)s)",
    )

    # ========== infer mode ==========
    infer_parser = subparsers.add_parser(
        "infer", help="Run inference", parents=[parent_parser]
    )
    infer_parser.add_argument(
        "--path2model", type=str, default="models/model.pkl", help="Path to saved model"
    )
    infer_parser.add_argument(
        "--path2output_dir",
        type=str,
        default="output",
        help="Path to directory for output file",
    )

    args = parser.parse_args()

    # set log level
    log_level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }
    logger.setLevel(log_level_map[args.log_level])

    # modes
    if args.mode == "train":
        # check paths
        if not find_or_create_dir(args.path2model_dir):
            logger.error(
                f"Failed to find or create the specified directory: {args.path2img_dir}"
            )
            sys.exit(1)

        training(args.n_epochs, args.path2model_dir)
    elif args.mode == "infer":
        # check paths
        if not is_valid_file(args.path2model):
            logger.error(
                f"Not a valid file path or the file does not exist: {args.path2vis}"
            )
            sys.exit(1)
        if not find_or_create_dir(args.path2output_dir):
            logger.error(
                f"Failed to find or create the specified directory: {args.path2img_dir}"
            )
            sys.exit(1)

        inference(args.path2model, args.path2output_dir)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.newline()
        msg = "|CLI stopped due to KeyboardInterrupt|"
        logger.warning(f"=========={msg:=<60}==========")
