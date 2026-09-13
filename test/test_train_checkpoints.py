"""Pytest coverage for owned train checkpoint and config bundles."""

from pathlib import Path

import torch

from models.attention.dat.config import DatLMConfig
from models.factory import build_model
from train.checkpoints import (
    ensure_experiment_dirs,
    load_model_checkpoint,
    read_experiment_config,
    read_model_config,
    resolve_checkpoint_paths,
    save_training_checkpoint,
    write_experiment_config,
    write_model_config,
)
from train.config import TrainConfig
from train.data import load_tokenizer
from train.nextlat import NextLatDynamicsModel
from train.paths import resolve_train_paths
from train.runner import build_model_config, maybe_save_progress_checkpoint
from conftest import build_tiny_train_config


def test_train_checkpoint_bundle_round_trip(tmp_path: Path) -> None:
    torch.manual_seed(0)
    config = build_tiny_train_config(tmp_path, "checkpoint-contract")
    paths = resolve_train_paths(config)
    ensure_experiment_dirs(paths)
    write_experiment_config(config, paths)

    tokenizer = load_tokenizer(paths)
    model_config = build_model_config(config, tokenizer)
    write_model_config(model_config, paths)

    model = build_model(config.model_type, model_config)

    input_ids = tokenizer("The boy found a dog", return_tensors="pt")["input_ids"][:, :4]
    logits_before, _ = model(input_ids)

    save_training_checkpoint(model, paths, label="unit")
    checkpoint_paths = resolve_checkpoint_paths(paths.checkpoint_dir, "unit")

    assert checkpoint_paths.directory.exists()
    assert checkpoint_paths.model_state_path.exists()
    assert checkpoint_paths.manifest_path.exists()

    loaded_train_config = read_experiment_config(paths)
    loaded_model_config = read_model_config(paths)

    assert loaded_train_config == config
    assert loaded_model_config == model_config

    reloaded_model = build_model(config.model_type, loaded_model_config)
    load_model_checkpoint(reloaded_model, paths.checkpoint_dir, "unit")
    logits_after, _ = reloaded_model(input_ids)

    assert torch.allclose(logits_before, logits_after)

    manifest_text = checkpoint_paths.manifest_path.read_text()
    assert "train_config: logging/train_config.yaml" in manifest_text
    assert "model_config: logging/model_config.yaml" in manifest_text


def test_dat_checkpoint_bundle_round_trip(tmp_path: Path) -> None:
    torch.manual_seed(0)
    config = build_tiny_train_config(
        tmp_path,
        "dat-checkpoint-contract",
        model_type="dat",
        n_layers=1,
        symbol_retrieval="relative",
        ra_type="disrca",
        ra_rel_activation="sigmoid",
        ffn_hidden_dim_mode="dff_factor",
        ffn_activation="swiglu",
        init_scheme="normal_0_02_scaled_projection",
    )
    paths = resolve_train_paths(config)
    ensure_experiment_dirs(paths)
    write_experiment_config(config, paths)

    tokenizer = load_tokenizer(paths)
    model_config = build_model_config(config, tokenizer)
    write_model_config(model_config, paths)

    model = build_model(config.model_type, model_config)

    input_ids = tokenizer("The boy found a dog", return_tensors="pt")["input_ids"][:, :4]
    logits_before, _ = model(input_ids)

    save_training_checkpoint(model, paths, label="dat-unit")
    loaded_model_config = read_model_config(paths)
    reloaded_model = build_model(config.model_type, loaded_model_config)
    load_model_checkpoint(reloaded_model, paths.checkpoint_dir, "dat-unit")
    logits_after, _ = reloaded_model(input_ids)

    assert isinstance(loaded_model_config, DatLMConfig)
    assert loaded_model_config.symbol_retrieval == "relative"
    assert loaded_model_config.ra_type == "disrca"
    assert loaded_model_config.ffn_hidden_dim_mode == "dff_factor"
    assert loaded_model_config.ffn_activation == "swiglu"
    assert loaded_model_config.init_scheme == "normal_0_02_scaled_projection"
    assert torch.allclose(logits_before, logits_after)


def test_checkpoint_saves_nextlat_dynamics_state_separately(tmp_path: Path) -> None:
    torch.manual_seed(0)
    config = build_tiny_train_config(
        tmp_path,
        "nextlat-checkpoint-contract",
        model_type="dat",
        n_layers=1,
        nextlat_enabled=True,
    )
    paths = resolve_train_paths(config)
    ensure_experiment_dirs(paths)
    write_experiment_config(config, paths)

    tokenizer = load_tokenizer(paths)
    model_config = build_model_config(config, tokenizer)
    write_model_config(model_config, paths)

    model = build_model(config.model_type, model_config)
    dynamics = NextLatDynamicsModel(
        hidden_dim=model.config.hidden_dim,
        dropout=config.dropout,
        proj_factor=config.nextlat_proj_factor,
        use_bias=False,
    )

    input_ids = tokenizer("The boy found a dog", return_tensors="pt")["input_ids"][:, :4]
    logits_before, _ = model(input_ids)

    save_training_checkpoint(
        model,
        paths,
        label="nextlat-unit",
        nextlat_dynamics_model=dynamics,
    )
    checkpoint_paths = resolve_checkpoint_paths(paths.checkpoint_dir, "nextlat-unit")

    assert checkpoint_paths.model_state_path.exists()
    assert checkpoint_paths.nextlat_dynamics_state_path.exists()
    assert "nextlat_dynamics_state: nextlat_dynamics.pt" in checkpoint_paths.manifest_path.read_text()

    loaded_model_config = read_model_config(paths)
    reloaded_model = build_model(config.model_type, loaded_model_config)
    load_model_checkpoint(reloaded_model, paths.checkpoint_dir, "nextlat-unit")
    logits_after, _ = reloaded_model(input_ids)

    assert torch.allclose(logits_before, logits_after)


def test_maybe_save_progress_checkpoint_writes_final_milestone_labels(tmp_path: Path) -> None:
    torch.manual_seed(0)
    strict_small_config = build_tiny_train_config(tmp_path, "checkpoint-contract")
    strict_small_paths = resolve_train_paths(strict_small_config)
    ensure_experiment_dirs(strict_small_paths)
    write_experiment_config(strict_small_config, strict_small_paths)

    tokenizer = load_tokenizer(strict_small_paths)
    model_config = build_model_config(strict_small_config, tokenizer)
    write_model_config(model_config, strict_small_paths)

    model = build_model(strict_small_config.model_type, model_config)

    maybe_save_progress_checkpoint(
        config=strict_small_config,
        completed_steps=100,
        num_steps=100,
        model=model,
        nextlat_dynamics_model=None,
        paths=strict_small_paths,
    )

    assert resolve_checkpoint_paths(strict_small_paths.checkpoint_dir, "10M").directory.exists()

    strict_config = TrainConfig(
        **{
            **strict_small_config.__dict__,
            "corpus_id": "strict",
        }
    )
    strict_paths = resolve_train_paths(strict_config)
    ensure_experiment_dirs(strict_paths)
    write_experiment_config(strict_config, strict_paths)
    write_model_config(model_config, strict_paths)

    maybe_save_progress_checkpoint(
        config=strict_config,
        completed_steps=100,
        num_steps=100,
        model=model,
        nextlat_dynamics_model=None,
        paths=strict_paths,
    )

    assert resolve_checkpoint_paths(strict_paths.checkpoint_dir, "100M").directory.exists()
