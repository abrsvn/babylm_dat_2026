"""Pytest coverage for EMA of model weights and warm-start checkpoint loading.

Covers the EMA configuration fields, the EmaModel shadow-weight math and
buffer syncing, EMA checkpoint artifacts, warm-start loading of raw vs. EMA
weights, and the end-to-end guarantee that a warm-started run's EMA shadow
starts from the warm-started weights rather than the fresh initialization.
"""

from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import yaml

from conftest import build_tiny_train_config
from models.factory import build_model
from train.checkpoints import (
    EmaModel,
    ensure_experiment_dirs,
    load_warm_start_state_dicts,
    resolve_checkpoint_paths,
    resolve_ema_device,
    save_training_checkpoint,
)
from train.config import TrainConfig
from train.data import load_tokenizer
from train.paths import resolve_train_paths
from train.runner import (
    build_model_config,
    run_training,
)


class _Tiny(nn.Module):
    """Minimal module with a parameter and a registered buffer."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 4, bias=False)
        self.register_buffer("buf", torch.zeros(4))


# ---- configuration ----


def test_ema_config_defaults_disabled():
    config = TrainConfig()
    assert config.ema_decay is None
    assert config.ema_device == "auto"
    assert config.init_from_checkpoint is None
    assert config.init_from_use_ema is False


def test_ema_config_accepts_enabled_values():
    config = replace(
        TrainConfig(),
        ema_decay=0.999,
        ema_device="cpu",
        init_from_checkpoint="/exp/phase_a/checkpoints/checkpoint_1",
        init_from_use_ema=True,
    )
    assert config.ema_decay == 0.999
    assert config.ema_device == "cpu"
    assert config.init_from_use_ema is True


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.5, 1.5])
def test_ema_decay_out_of_range_rejected(bad):
    with pytest.raises(ValueError, match="ema_decay"):
        replace(TrainConfig(), ema_decay=bad)


def test_init_from_use_ema_requires_checkpoint():
    with pytest.raises(ValueError, match="init_from_use_ema"):
        replace(TrainConfig(), init_from_use_ema=True)
    # With a checkpoint set the combination is valid.
    config = replace(
        TrainConfig(),
        init_from_checkpoint="/exp/phase_a/checkpoints/checkpoint_1",
        init_from_use_ema=True,
    )
    assert config.init_from_use_ema is True


# ---- device resolution ----


def test_resolve_ema_device_auto_uses_model_device():
    cpu = torch.device("cpu")
    assert resolve_ema_device("auto", cpu) == cpu


def test_resolve_ema_device_explicit_overrides_model_device():
    assert resolve_ema_device("cpu", torch.device("cuda")) == torch.device("cpu")


# ---- EMA math and buffers ----


def test_ema_update_applies_decay_formula():
    torch.manual_seed(0)
    model = _Tiny()
    decay = 0.9
    ema = EmaModel(model, decay=decay, device=torch.device("cpu"))
    before = model.lin.weight.detach().clone()
    with torch.no_grad():
        model.lin.weight.add_(1.0)  # new = old + 1
    ema.update(model)
    expected = decay * before + (1.0 - decay) * (before + 1.0)  # = old + 0.1
    assert torch.allclose(ema.module.lin.weight, expected, atol=1e-6)


def test_ema_shadow_is_frozen_and_eval():
    model = _Tiny()
    ema = EmaModel(model, decay=0.9, device=torch.device("cpu"))
    assert not any(parameter.requires_grad for parameter in ema.module.parameters())
    assert not ema.module.training


def test_ema_buffers_synced_only_on_demand():
    model = _Tiny()
    ema = EmaModel(model, decay=0.9, device=torch.device("cpu"))
    with torch.no_grad():
        model.buf.fill_(1.0)
    # update() leaves buffers alone.
    ema.update(model)
    assert torch.allclose(ema.module.buf, torch.zeros(4))
    ema.sync_buffers(model)
    assert torch.allclose(ema.module.buf, torch.ones(4))


def test_ema_sync_buffers_handles_shape_change():
    # A lazily-sized or resized buffer (e.g. a rotary/mask cache) may have a
    # different shape than the deepcopy captured at EMA creation.
    model = _Tiny()
    ema = EmaModel(model, decay=0.9, device=torch.device("cpu"))
    replacement = torch.ones(9)
    delattr(model, "buf")
    model.register_buffer("buf", replacement)  # was shape (4,)
    ema.sync_buffers(model)
    assert ema.module.buf.shape == (9,)
    assert torch.allclose(ema.module.buf, torch.ones(9))


# ---- checkpoint artifacts and warm-start loading (real model, real paths) ----


def _tiny_training_model(tmp_path: Path, experiment_name: str):
    config = build_tiny_train_config(tmp_path, experiment_name)
    paths = resolve_train_paths(config)
    ensure_experiment_dirs(paths)
    tokenizer = load_tokenizer(paths)
    model_config = build_model_config(config, tokenizer)
    model = build_model(config.model_type, model_config)
    return paths, model


def test_checkpoint_saves_ema_artifact_and_manifest(tmp_path: Path):
    torch.manual_seed(0)
    paths, model = _tiny_training_model(tmp_path, "ema-artifact")
    ema = EmaModel(model, decay=0.9, device=torch.device("cpu"))
    save_training_checkpoint(model, paths, label=1, ema_model=ema)
    checkpoint_paths = resolve_checkpoint_paths(paths.checkpoint_dir, 1)
    assert checkpoint_paths.model_ema_state_path.is_file()
    manifest = yaml.safe_load(checkpoint_paths.manifest_path.read_text())
    assert manifest["model_ema_state"] == "model_ema.pt"


def test_checkpoint_omits_ema_when_disabled(tmp_path: Path):
    torch.manual_seed(0)
    paths, model = _tiny_training_model(tmp_path, "ema-disabled")
    save_training_checkpoint(model, paths, label=1)
    checkpoint_paths = resolve_checkpoint_paths(paths.checkpoint_dir, 1)
    assert not checkpoint_paths.model_ema_state_path.is_file()
    manifest = yaml.safe_load(checkpoint_paths.manifest_path.read_text())
    assert "model_ema_state" not in manifest


def test_warm_start_loads_ema_and_raw_weights(tmp_path: Path):
    torch.manual_seed(0)
    paths, model = _tiny_training_model(tmp_path, "warm-start-state")
    initial = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    # The shadow is captured before the perturbation, then nudged halfway
    # toward the perturbed weights by a single update at decay 0.5.
    ema = EmaModel(model, decay=0.5, device=torch.device("cpu"))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.05)
    ema.update(model)
    save_training_checkpoint(model, paths, label=1, ema_model=ema)
    checkpoint_paths = resolve_checkpoint_paths(paths.checkpoint_dir, 1)

    tokenizer = load_tokenizer(paths)
    model_config = build_model_config(build_tiny_train_config(tmp_path, "warm-start-state"), tokenizer)

    # use_ema=True loads model_ema.pt.
    fresh_ema = build_model("self_attention", model_config)
    loaded = load_warm_start_state_dicts(fresh_ema, checkpoint_paths.directory, use_ema=True)
    assert loaded == (checkpoint_paths.model_ema_state_path,)
    for name, parameter in fresh_ema.named_parameters():
        expected = 0.5 * initial[name] + 0.5 * (initial[name] + 0.05)
        assert torch.allclose(parameter.detach(), expected, atol=1e-6)

    # use_ema=False loads model.pt (the perturbed raw weights).
    fresh_raw = build_model("self_attention", model_config)
    loaded = load_warm_start_state_dicts(fresh_raw, checkpoint_paths.directory, use_ema=False)
    assert loaded == (checkpoint_paths.model_state_path,)
    for name, parameter in fresh_raw.named_parameters():
        assert torch.allclose(parameter.detach(), initial[name] + 0.05, atol=1e-6)


def test_warm_start_missing_state_fails_fast(tmp_path: Path):
    torch.manual_seed(0)
    paths, model = _tiny_training_model(tmp_path, "warm-start-missing")
    ema = EmaModel(model, decay=0.9, device=torch.device("cpu"))
    save_training_checkpoint(model, paths, label=1, ema_model=ema)
    checkpoint_paths = resolve_checkpoint_paths(paths.checkpoint_dir, 1)
    # Declared in the manifest but absent from disk.
    checkpoint_paths.model_ema_state_path.unlink()
    with pytest.raises(FileNotFoundError, match="model_ema.pt"):
        load_warm_start_state_dicts(model, checkpoint_paths.directory, use_ema=True)


def test_warm_start_rejects_stale_ema_artifact(tmp_path: Path):
    """A stale model_ema.pt on disk must not be loadable.

    save_training_checkpoint never deletes files, so overwriting an EMA
    checkpoint label with a non-EMA run leaves the old model_ema.pt behind
    while the manifest no longer declares it. A use_ema=True warm start must
    reject it via the manifest rather than silently loading another run's
    EMA weights.
    """
    torch.manual_seed(0)
    paths, model = _tiny_training_model(tmp_path, "warm-start-stale")
    ema = EmaModel(model, decay=0.9, device=torch.device("cpu"))
    save_training_checkpoint(model, paths, label=1, ema_model=ema)
    # Overwrite the same label with EMA disabled.
    save_training_checkpoint(model, paths, label=1)
    checkpoint_paths = resolve_checkpoint_paths(paths.checkpoint_dir, 1)
    # The stale artifact is still on disk, but the manifest no longer
    # declares it.
    assert checkpoint_paths.model_ema_state_path.is_file()
    manifest = yaml.safe_load(checkpoint_paths.manifest_path.read_text())
    assert "model_ema_state" not in manifest
    with pytest.raises(ValueError, match="model_ema.pt"):
        load_warm_start_state_dicts(model, checkpoint_paths.directory, use_ema=True)
    # use_ema=False still loads: model.pt is declared and current.
    loaded = load_warm_start_state_dicts(model, checkpoint_paths.directory, use_ema=False)
    assert loaded == (checkpoint_paths.model_state_path,)


def test_warm_start_requires_manifest(tmp_path: Path):
    torch.manual_seed(0)
    paths, model = _tiny_training_model(tmp_path, "warm-start-no-manifest")
    save_training_checkpoint(model, paths, label=1)
    checkpoint_paths = resolve_checkpoint_paths(paths.checkpoint_dir, 1)
    checkpoint_paths.manifest_path.unlink()
    with pytest.raises(FileNotFoundError, match="checkpoint.yaml"):
        load_warm_start_state_dicts(model, checkpoint_paths.directory, use_ema=False)


# ---- end-to-end: warm-start ordering ----


def test_warm_started_run_ema_shadow_starts_from_warm_started_weights(tmp_path: Path):
    """A warm-started run's EMA shadow must copy the warm-started weights.

    Phase A trains one tiny epoch with EMA. Phase B warm-starts from A's EMA
    checkpoint with an EMA decay so close to 1 that per-step updates move the
    shadow negligibly, so B's saved model_ema.pt is effectively the shadow as
    it was at construction. The ordering guarantee says that shadow equals
    the warm-started weights (A's EMA); under the buggy ordering it would be
    B's random initialization and the assertion would fail.
    """
    torch.manual_seed(0)
    config_a = build_tiny_train_config(
        tmp_path / "a",
        "ema-ordering-phase-a",
        n_layers=1,
        n_epochs=1,
        batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=0.01,
        ema_decay=0.9,
    )
    run_training(config_a)
    checkpoints_a = sorted(
        (Path(config_a.base_folder) / config_a.experiment_name / "checkpoints").glob("checkpoint_*")
    )
    assert checkpoints_a
    checkpoint_a = [c for c in checkpoints_a if (c / "model_ema.pt").is_file()][-1]
    state_a_ema = torch.load(checkpoint_a / "model_ema.pt", map_location="cpu", weights_only=True)
    state_a_raw = torch.load(checkpoint_a / "model.pt", map_location="cpu", weights_only=True)
    # Phase A trained, so its EMA differs from its raw weights.
    raw_ema_gap = max(float((state_a_raw[k] - state_a_ema[k]).abs().max()) for k in state_a_ema)
    assert raw_ema_gap > 1e-3

    torch.manual_seed(0)
    config_b = build_tiny_train_config(
        tmp_path / "b",
        "ema-ordering-phase-b",
        n_layers=1,
        n_epochs=1,
        batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=0.01,
        ema_decay=0.999999999,
        init_from_checkpoint=str(checkpoint_a),
        init_from_use_ema=True,
    )
    run_training(config_b)
    checkpoints_b = sorted(
        (Path(config_b.base_folder) / config_b.experiment_name / "checkpoints").glob("checkpoint_*")
    )
    assert checkpoints_b
    checkpoint_b = [c for c in checkpoints_b if (c / "model_ema.pt").is_file()][-1]
    state_b_ema = torch.load(checkpoint_b / "model_ema.pt", map_location="cpu", weights_only=True)
    state_b_raw = torch.load(checkpoint_b / "model.pt", map_location="cpu", weights_only=True)

    # B's EMA shadow started from the warm-started weights (A's EMA).
    max_diff = max(float((state_b_ema[k] - state_a_ema[k]).abs().max()) for k in state_a_ema)
    assert max_diff < 1e-5, f"EMA shadow did not start from warm-started weights (max diff {max_diff})"

    # B's raw weights moved during its own epoch (the run really trained).
    raw_moved = max(float((state_b_raw[k] - state_a_ema[k]).abs().max()) for k in state_a_ema)
    assert raw_moved > 1e-4
