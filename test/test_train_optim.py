"""Pytest coverage for derived optimizer and scheduler step accounting."""

from pathlib import Path

import pytest
import torch

from models.factory import build_model
from models.attention.transformer.config import SelfAttentionLMConfig
from models.attention.transformer.lm import SelfAttentionDecoderLM
from train.checkpoints import ensure_experiment_dirs, write_experiment_config, write_model_config
from train.data import load_tokenizer, load_train_data
from train.nextlat import NextLatDynamicsModel, NextLatLossConfig, compute_nextlat_loss
from train.optim import build_optimizer_and_scheduler, resolve_num_warmup_steps, resolve_total_training_steps
from train.paths import resolve_train_paths
from train.runner import build_model_config, train_epoch
from conftest import build_tiny_train_config


def build_accumulation_test_model() -> SelfAttentionDecoderLM:
    config = SelfAttentionLMConfig(
        vocab_size=24,
        max_seq_len=5,
        hidden_dim=16,
        n_heads=4,
        n_layers=1,
        dropout=0.0,
        tie_lm_head=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    return SelfAttentionDecoderLM(config)


def clone_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
    }


def collect_named_grads(*modules: torch.nn.Module) -> dict[str, torch.Tensor]:
    grads: dict[str, torch.Tensor] = {}
    for module_index, module in enumerate(modules):
        for name, parameter in module.named_parameters():
            if parameter.grad is not None:
                grads[f"{module_index}.{name}"] = parameter.grad.detach().clone()
    return grads


def assert_matching_grads(
    actual: dict[str, torch.Tensor],
    expected: dict[str, torch.Tensor],
) -> None:
    assert actual.keys() == expected.keys()
    for name, actual_grad in actual.items():
        assert torch.allclose(actual_grad, expected[name], atol=1e-6), name


def test_resolve_total_training_steps_uses_real_dataloader_size(tmp_path: Path) -> None:
    config = build_tiny_train_config(
        tmp_path,
        "optim-contract",
        n_epochs=3,
    )
    paths = resolve_train_paths(config)
    artifacts = load_train_data(config, paths)

    total_steps = resolve_total_training_steps(config, len(artifacts.dataloader))

    assert total_steps == (len(artifacts.dataloader) * config.n_epochs) // config.gradient_accumulation_steps


def test_resolve_total_training_steps_counts_accumulated_optimizer_steps(tmp_path: Path) -> None:
    config = build_tiny_train_config(
        tmp_path,
        "optim-accumulation-contract",
        n_epochs=3,
        gradient_accumulation_steps=2,
    )
    paths = resolve_train_paths(config)
    artifacts = load_train_data(config, paths)

    total_steps = resolve_total_training_steps(config, len(artifacts.dataloader))

    expected_steps_per_epoch = (len(artifacts.dataloader) + 1) // 2
    assert total_steps == expected_steps_per_epoch * config.n_epochs


def test_optimizer_scheduler_initialization_uses_training_config(tmp_path: Path) -> None:
    config = build_tiny_train_config(
        tmp_path,
        "optim-contract",
        n_epochs=3,
        warmup_ratio=0.5,
    )
    paths = resolve_train_paths(config)
    artifacts = load_train_data(config, paths)
    total_steps = resolve_total_training_steps(config, len(artifacts.dataloader))
    tokenizer = load_tokenizer(paths)
    model_config = build_model_config(config, tokenizer)
    model = build_model(config.model_type, model_config)
    optimizer, scheduler = build_optimizer_and_scheduler(config, (model,), total_steps)

    assert resolve_num_warmup_steps(config, total_steps) > 0
    assert len(scheduler.base_lrs) == len(optimizer.param_groups)
    assert all(base_lr == pytest.approx(config.learning_rate) for base_lr in scheduler.base_lrs)
    assert optimizer.defaults["lr"] == pytest.approx(config.learning_rate)
    assert all(group["lr"] == pytest.approx(0.0) for group in optimizer.param_groups)
    assert scheduler.state_dict()["last_epoch"] == 0


def test_train_epoch_steps_scheduler_after_accumulated_minibatches(tmp_path: Path) -> None:
    torch.manual_seed(0)
    config = build_tiny_train_config(
        tmp_path,
        "train-epoch-accumulation",
        n_layers=1,
        batch_size=1,
        gradient_accumulation_steps=2,
    )
    paths = resolve_train_paths(config)
    ensure_experiment_dirs(paths)
    write_experiment_config(config, paths)
    artifacts = load_train_data(config, paths)
    tokenizer = load_tokenizer(paths)
    model_config = build_model_config(config, tokenizer)
    write_model_config(model_config, paths)
    model = build_model(config.model_type, model_config)
    total_steps = resolve_total_training_steps(config, len(artifacts.dataloader))
    optimizer, scheduler = build_optimizer_and_scheduler(config, (model,), total_steps)
    optimizer.zero_grad(set_to_none=True)

    train_epoch(
        config=config,
        paths=paths,
        model=model,
        nextlat_dynamics_model=None,
        optimizer=optimizer,
        scheduler=scheduler,
        dataloader=artifacts.dataloader,
        epoch=0,
        start_time=0.0,
        wandb_module=None,
    )

    expected_optimizer_steps = (len(artifacts.dataloader) + 1) // 2
    assert scheduler.state_dict()["last_epoch"] == expected_optimizer_steps


def test_gradient_accumulation_matches_full_batch_ntp_gradients_for_equal_token_counts() -> None:
    torch.manual_seed(0)
    initial_model = build_accumulation_test_model()
    initial_state = clone_state_dict(initial_model)
    full_batch_model = build_accumulation_test_model()
    accumulated_model = build_accumulation_test_model()
    full_batch_model.load_state_dict(initial_state)
    accumulated_model.load_state_dict(initial_state)
    input_tokens = torch.tensor(
        [
            [1, 3, 4, 5],
            [1, 6, 7, 8],
        ],
        dtype=torch.long,
    )
    target_tokens = torch.tensor(
        [
            [3, 4, 5, 2],
            [6, 7, 8, 2],
        ],
        dtype=torch.long,
    )

    _, full_batch_loss = full_batch_model(input_tokens, targets=target_tokens)
    assert full_batch_loss is not None
    full_batch_loss.backward()

    for row_index in range(input_tokens.shape[0]):
        _, microbatch_loss = accumulated_model(
            input_tokens[row_index : row_index + 1],
            targets=target_tokens[row_index : row_index + 1],
        )
        assert microbatch_loss is not None
        (microbatch_loss / input_tokens.shape[0]).backward()

    assert_matching_grads(
        actual=collect_named_grads(accumulated_model),
        expected=collect_named_grads(full_batch_model),
    )


def test_gradient_accumulation_matches_full_batch_nextlat_gradients_for_equal_token_counts() -> None:
    torch.manual_seed(0)
    initial_model = build_accumulation_test_model()
    initial_dynamics = NextLatDynamicsModel(
        hidden_dim=initial_model.config.hidden_dim,
        dropout=0.0,
        proj_factor=1.0,
        use_bias=True,
    )
    initial_model_state = clone_state_dict(initial_model)
    initial_dynamics_state = clone_state_dict(initial_dynamics)
    full_batch_model = build_accumulation_test_model()
    accumulated_model = build_accumulation_test_model()
    full_batch_dynamics = NextLatDynamicsModel(
        hidden_dim=initial_model.config.hidden_dim,
        dropout=0.0,
        proj_factor=1.0,
        use_bias=True,
    )
    accumulated_dynamics = NextLatDynamicsModel(
        hidden_dim=initial_model.config.hidden_dim,
        dropout=0.0,
        proj_factor=1.0,
        use_bias=True,
    )
    full_batch_model.load_state_dict(initial_model_state)
    accumulated_model.load_state_dict(initial_model_state)
    full_batch_dynamics.load_state_dict(initial_dynamics_state)
    accumulated_dynamics.load_state_dict(initial_dynamics_state)
    input_tokens = torch.tensor(
        [
            [1, 3, 4, 5],
            [1, 6, 7, 8],
        ],
        dtype=torch.long,
    )
    target_tokens = torch.tensor(
        [
            [3, 4, 5, 2],
            [6, 7, 8, 2],
        ],
        dtype=torch.long,
    )
    target_mask = torch.ones_like(target_tokens, dtype=torch.bool)
    loss_config = NextLatLossConfig(
        lambda_mse=1.0,
        lambda_kl=0.5,
        lambda_ce=0.25,
        horizon=1,
        eos_token_id=2,
        sequence_boundary_policy="eos_document",
    )

    full_batch_result = compute_nextlat_loss(
        model=full_batch_model,
        dynamics_model=full_batch_dynamics,
        input_tokens=input_tokens,
        target_tokens=target_tokens,
        target_mask=target_mask,
        config=loss_config,
    )
    full_batch_result.loss.backward()

    for row_index in range(input_tokens.shape[0]):
        microbatch_result = compute_nextlat_loss(
            model=accumulated_model,
            dynamics_model=accumulated_dynamics,
            input_tokens=input_tokens[row_index : row_index + 1],
            target_tokens=target_tokens[row_index : row_index + 1],
            target_mask=target_mask[row_index : row_index + 1],
            config=loss_config,
        )
        (microbatch_result.loss / input_tokens.shape[0]).backward()

    assert_matching_grads(
        actual=collect_named_grads(accumulated_model, accumulated_dynamics),
        expected=collect_named_grads(full_batch_model, full_batch_dynamics),
    )


def test_resolve_num_warmup_steps_uses_training_step_ratio(tmp_path: Path) -> None:
    config = build_tiny_train_config(
        tmp_path,
        "optim-contract",
        warmup_ratio=0.01,
    )

    assert resolve_num_warmup_steps(config, 19980) == 200
    assert resolve_num_warmup_steps(config, 5994) == 60
    assert resolve_num_warmup_steps(config, 1998) == 20


def test_resolve_num_warmup_steps_rejects_invalid_total_steps(tmp_path: Path) -> None:
    config = build_tiny_train_config(
        tmp_path,
        "optim-contract",
        warmup_ratio=0.01,
    )

    with pytest.raises(ValueError, match="total_training_steps must be positive"):
        resolve_num_warmup_steps(config, 0)
