"""YAML-config entrypoint for Relational BabyLM training.

A sibling of cli.py that takes its configuration from two YAML files instead of
command-line flags:

    python -m train.cli_yaml --train-config train_config.yaml --model-config model_config.yaml

train_config.yaml is the source of truth: it is loaded into a TrainConfig and
passed to run_training, exactly like the argparse-based cli.py would. The model
architecture (model_config.yaml) is still derived inside the runner from the
TrainConfig plus the tokenizer, so model_config.yaml is used only to validate
that derivation. If the architecture the runner would build does not match
model_config.yaml field-for-field, training aborts before it starts.

Both files use the same schema the trainer writes into each experiment's
logging/ directory (see train/checkpoints.py).

experiment_name and wandb_experiment_name in train_config.yaml may contain
strftime codes, which are expanded against the launch time. For example
``experiment_name: experiment_%Y%m%d`` becomes ``experiment_20250101``.
Both names are expanded against the same time, so the on-disk experiment folder
and the wandb run name carry an identical date stamp. A value without ``%`` codes
is used verbatim.
"""

import argparse
import logging
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import yaml

from models.factory import build_model_config_from_dict
from train.config import TrainConfig
from train.data import load_tokenizer
from train.paths import materialize_train_asset_config, resolve_train_paths
from train.runner import build_model_config, run_training

LOG_FORMAT = "%(message)s"


def _load_yaml_mapping(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    with path.open("r") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {path}")
    return raw


def load_train_config(path: Path) -> TrainConfig:
    return TrainConfig(**_load_yaml_mapping(path))


def resolve_experiment_names(config: TrainConfig, now: datetime | None = None) -> TrainConfig:
    """Expand strftime codes in experiment_name and wandb_experiment_name.

    Both names are expanded against the same launch time, so the on-disk
    experiment folder (derived from experiment_name in resolve_train_paths) and
    the wandb run name carry an identical date stamp. Lets the YAML embed
    date/time placeholders, e.g. ``experiment_%m_%d`` becomes
    ``experiment_06_22``. A name without ``%`` codes is left unchanged, so
    this is idempotent when re-running from a snapshotted config.
    """
    now = now or datetime.now()
    return replace(
        config,
        experiment_name=now.strftime(config.experiment_name),
        wandb_experiment_name=now.strftime(config.wandb_experiment_name),
    )


def validate_model_config(config: TrainConfig, model_config_path: Path) -> None:
    """Fail-fast if model_config.yaml disagrees with the runner-derived ModelConfig.

    Replicates the model-config derivation run_training performs (asset
    materialization, tokenizer load, build_model_config) and compares the result
    field-for-field with the supplied model_config.yaml.
    """
    resolved_config = materialize_train_asset_config(config)
    paths = resolve_train_paths(resolved_config)
    tokenizer = load_tokenizer(paths)
    derived = asdict(build_model_config(resolved_config, tokenizer))

    raw_expected = _load_yaml_mapping(model_config_path)
    expected = asdict(build_model_config_from_dict(config.model_type, raw_expected))

    mismatches = []
    for key in sorted(set(derived) | set(expected)):
        derived_value = derived.get(key, "<absent>")
        expected_value = expected.get(key, "<absent>")
        if derived_value != expected_value:
            mismatches.append(f"  {key}: model_config.yaml={expected_value!r}, derived={derived_value!r}")

    if mismatches:
        raise ValueError(
            f"model_config.yaml ({model_config_path}) does not match the model the runner "
            "would build from train_config.yaml. Fix the mismatched fields:\n"
            + "\n".join(mismatches)
        )

def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train Relational BabyLM language models from YAML config files."
    )
    parser.add_argument(
        "--train-config",
        type=Path,
        required=True,
        help="Path to train_config.yaml (loaded into TrainConfig; source of truth for the run).",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        required=True,
        help="Path to model_config.yaml (validated against the runner-derived model architecture).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
    )
    args = build_argument_parser().parse_args(argv)
    config = load_train_config(args.train_config)
    config = resolve_experiment_names(config)
    validate_model_config(config, args.model_config)
    run_training(config)


if __name__ == "__main__":
    main()
