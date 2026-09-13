"""Step 2 tests: per-phase dataset cache directories and loading in data.py."""

import logging

from train.data import (
    load_or_create_dataset,
    load_or_create_phase_dataset,
    load_tokenizer,
    read_dataset_cache_metadata,
)
from train.paths import resolve_phase_dataset_cache_dir, resolve_train_paths
from conftest import build_tiny_train_config


def test_base_phase_cache_dir_matches_resolved_dataset_cache_dir(tmp_path):
    config = build_tiny_train_config(tmp_path, "phase-base-dir")
    paths = resolve_train_paths(config)
    resolved = resolve_phase_dataset_cache_dir(paths, config.datapoint_length)
    assert resolved == paths.dataset_cache_dir


def test_doubled_phase_cache_dir_is_distinct_and_length_tagged(tmp_path):
    config = build_tiny_train_config(tmp_path, "phase-doubled-dir")
    paths = resolve_train_paths(config)
    doubled = resolve_phase_dataset_cache_dir(paths, config.datapoint_length * 2)
    assert doubled != paths.dataset_cache_dir
    assert f"len{config.datapoint_length * 2}" in doubled.name


def test_phase_datasets_build_separate_caches_with_correct_widths(tmp_path):
    config = build_tiny_train_config(tmp_path, "phase-widths")
    paths = resolve_train_paths(config)
    tokenizer = load_tokenizer(paths)
    base_len = config.datapoint_length
    doubled_len = base_len * 2

    base_dataset = load_or_create_phase_dataset(paths, tokenizer, base_len)
    doubled_dataset = load_or_create_phase_dataset(paths, tokenizer, doubled_len)

    base_dir = resolve_phase_dataset_cache_dir(paths, base_len)
    doubled_dir = resolve_phase_dataset_cache_dir(paths, doubled_len)
    assert base_dir != doubled_dir
    assert base_dir.exists() and doubled_dir.exists()

    base_metadata = read_dataset_cache_metadata(base_dir)
    doubled_metadata = read_dataset_cache_metadata(doubled_dir)
    assert int(base_metadata["datapoint_length"]) == base_len
    assert int(doubled_metadata["datapoint_length"]) == doubled_len

    assert base_dataset.token_chunks.shape[1] == base_len
    assert doubled_dataset.token_chunks.shape[1] == doubled_len

    # __getitem__ returns (chunk_with_specials, actual_length); the
    # chunk wraps with BOS/EOS, so length <= seq_length + 2.
    base_item, _base_actual_len = base_dataset[0]
    doubled_item, _doubled_actual_len = doubled_dataset[0]
    assert int(base_item[0]) == base_dataset.model_bos
    assert int(base_item[-1]) == base_dataset.model_eos
    assert base_item.numel() <= base_len + 2
    assert doubled_item.numel() <= doubled_len + 2


def test_base_wrapper_and_phase_loader_use_same_cache(tmp_path):
    config = build_tiny_train_config(tmp_path, "phase-base-equivalence")
    paths = resolve_train_paths(config)
    tokenizer = load_tokenizer(paths)

    wrapper_dataset = load_or_create_dataset(config, paths, tokenizer)
    phase_dataset = load_or_create_phase_dataset(paths, tokenizer, config.datapoint_length)

    assert wrapper_dataset.token_chunks.shape == phase_dataset.token_chunks.shape
    assert paths.dataset_cache_dir.exists()


def test_phase_loader_reuses_existing_cache_without_rebuild(tmp_path, caplog):
    config = build_tiny_train_config(tmp_path, "phase-reuse")
    paths = resolve_train_paths(config)
    tokenizer = load_tokenizer(paths)
    doubled_len = config.datapoint_length * 2

    load_or_create_phase_dataset(paths, tokenizer, doubled_len)

    with caplog.at_level(logging.INFO):
        load_or_create_phase_dataset(paths, tokenizer, doubled_len)

    assert any("Loading tokenized dataset cache from" in message for message in caplog.messages)
    assert not any("Building tokenized dataset cache at" in message for message in caplog.messages)
