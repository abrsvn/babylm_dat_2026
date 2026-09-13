"""Coverage for the tie_lm_head configuration field and its derivation.

tie_lm_head=None derives from the objective: tied for NTP, untied for
NextLat. An explicit value overrides the derivation, which is how the
untied-NTP comparison runs are expressed. The CLI flag accepts exactly
"true" or "false".
"""

from pathlib import Path

import pytest

from conftest import build_tiny_train_config
from train.config import parse_train_config
from train.data import load_tokenizer
from train.paths import resolve_train_paths
from train.runner import build_model_config


def derived_tie_lm_head(tmp_path: Path, **overrides) -> bool:
    config = build_tiny_train_config(tmp_path, "tie-lm-head-contract", **overrides)
    paths = resolve_train_paths(config)
    tokenizer = load_tokenizer(paths)
    return build_model_config(config, tokenizer).tie_lm_head


def test_default_derives_tied_for_ntp(tmp_path: Path) -> None:
    assert derived_tie_lm_head(tmp_path, nextlat_enabled=False) is True


def test_default_derives_untied_for_nextlat(tmp_path: Path) -> None:
    assert derived_tie_lm_head(tmp_path, nextlat_enabled=True) is False


def test_explicit_false_overrides_ntp_derivation(tmp_path: Path) -> None:
    assert derived_tie_lm_head(tmp_path, nextlat_enabled=False, tie_lm_head=False) is False


def test_cli_flag_parses_true_and_false() -> None:
    common = [
        "--tokenizer_dir", "test/fixtures/tiny_tokenizer",
        "--train_data_dir", "test/fixtures/train_assets/clean_train_tiny",
        "--experiment_name", "tie-lm-head-cli",
    ]
    tied = parse_train_config(["--tie_lm_head", "true", *common])
    untied = parse_train_config(["--tie_lm_head", "false", *common])
    omitted = parse_train_config(common)
    assert tied.tie_lm_head is True
    assert untied.tie_lm_head is False
    assert omitted.tie_lm_head is None


def test_cli_flag_rejects_other_values() -> None:
    with pytest.raises(SystemExit):
        parse_train_config([
            "--tie_lm_head", "maybe",
            "--tokenizer_dir", "test/fixtures/tiny_tokenizer",
            "--train_data_dir", "test/fixtures/train_assets/clean_train_tiny",
            "--experiment_name", "tie-lm-head-cli",
        ])
