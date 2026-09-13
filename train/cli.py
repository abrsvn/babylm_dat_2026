"""CLI entrypoint for Relational BabyLM training."""

import logging

from train.config import parse_train_config
from train.runner import run_training


LOG_FORMAT = "%(message)s"


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
    )
    config = parse_train_config(argv)
    run_training(config)


if __name__ == "__main__":
    main()
