"""Runner wiring tests: model-config sizing, step accounting, and train_epoch."""

import math

import torch

from models.factory import build_model
from train.checkpoints import write_model_config
from train.data import build_dataloader_for_dataset, load_tokenizer
from train.optim import (
    build_optimizer_and_scheduler,
    resolve_scheduled_total_training_steps,
    resolve_total_training_steps,
)
from train.paths import resolve_train_paths
from train.runner import (
    build_model_config,
    prepare_training_setup,
    train_epoch,
)
from conftest import build_tiny_train_config


def test_build_model_config_sizes_max_seq_len_to_datapoint_length(tmp_path):
    config = build_tiny_train_config(tmp_path, "runner-maxseq", n_epochs=3, batch_size=4)
    paths = resolve_train_paths(config)
    tokenizer = load_tokenizer(paths)
    model_config = build_model_config(config, tokenizer)
    assert model_config.max_seq_len == config.datapoint_length + 1


def test_build_model_config_with_nextlat_adds_horizon_slot(tmp_path):
    config = build_tiny_train_config(
        tmp_path, "runner-maxseq-nextlat", n_epochs=1, nextlat_enabled=True,
    )
    paths = resolve_train_paths(config)
    tokenizer = load_tokenizer(paths)
    model_config = build_model_config(config, tokenizer)
    assert model_config.max_seq_len == config.datapoint_length + 2
    assert model_config.tie_lm_head is False


def test_scheduled_total_matches_plain_total(tmp_path):
    config = build_tiny_train_config(tmp_path, "runner-noschedule", n_epochs=2)
    setup = prepare_training_setup(config)

    scheduled = resolve_scheduled_total_training_steps(setup.config, len(setup.data_artifacts.dataset))
    num_batches = (
        len(setup.data_artifacts.dataset) + setup.config.batch_size - 1
    ) // setup.config.batch_size
    expected = resolve_total_training_steps(setup.config, num_batches)
    assert scheduled == expected


def test_train_epoch_runs_and_steps_scheduler(tmp_path):
    torch.manual_seed(0)
    config = build_tiny_train_config(
        tmp_path,
        "runner-epoch",
        n_layers=1,
        n_epochs=1,
        batch_size=4,
        gradient_accumulation_steps=1,
    )
    setup = prepare_training_setup(config)
    model_config = build_model_config(setup.config, setup.tokenizer)
    write_model_config(model_config, setup.paths)
    model = build_model(setup.config.model_type, model_config).to(setup.device)

    dataset = setup.data_artifacts.dataset
    dataloader = build_dataloader_for_dataset(
        dataset, setup.tokenizer, setup.config.batch_size
    )
    total_steps = resolve_scheduled_total_training_steps(setup.config, len(dataset))
    optimizer, scheduler = build_optimizer_and_scheduler(setup.config, (model,), total_steps)
    optimizer.zero_grad(set_to_none=True)

    metrics = train_epoch(
        config=setup.config,
        paths=setup.paths,
        model=model,
        nextlat_dynamics_model=None,
        optimizer=optimizer,
        scheduler=scheduler,
        dataloader=dataloader,
        epoch=0,
        start_time=0.0,
        wandb_module=None,
        global_step_offset=0,
        total_training_steps=total_steps,
    )

    assert "loss" in metrics
    assert math.isfinite(metrics["loss"])
    assert metrics["clm_tokens"] > 0
    assert metrics["optimizer_step_count"] == len(dataloader)
    assert scheduler.last_epoch == len(dataloader)
