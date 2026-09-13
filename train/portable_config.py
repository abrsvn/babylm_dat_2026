"""Portable train_config.yaml helpers for shared experiment config snapshots."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PORTABLE_EXPERIMENT_ROOT = "<experiment-root>"
PORTABLE_CACHE_DIR = "<experiment-root>/cache/train"


def load_train_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return raw


def portable_train_config(raw: dict[str, Any]) -> dict[str, Any]:
    raw["base_folder"]
    raw["cache_dir"]
    raw["experiment_name"]
    raw["symbol_retrieval"]

    portable = dict(raw)
    portable["base_folder"] = PORTABLE_EXPERIMENT_ROOT
    portable["cache_dir"] = PORTABLE_CACHE_DIR
    if "relative_symbols_rope" not in portable:
        portable["relative_symbols_rope"] = False
    if not isinstance(portable["relative_symbols_rope"], bool):
        raise TypeError(
            "relative_symbols_rope must be a boolean when present in train_config.yaml, "
            f"got {type(portable['relative_symbols_rope']).__name__}"
        )
    return portable


def write_train_config(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
