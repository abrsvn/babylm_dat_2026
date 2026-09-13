"""Experiment directory, config snapshot, EMA state, and checkpoint helpers."""

import copy
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import yaml

logger = logging.getLogger(__name__)

# Stable filenames for checkpoint state. These are the single source of truth;
# every path resolution and manifest entry references these constants.
MODEL_STATE_FILENAME = "model.pt"
NEXTLAT_DYNAMICS_STATE_FILENAME = "nextlat_dynamics.pt"
MODEL_EMA_STATE_FILENAME = "model_ema.pt"
MANIFEST_FILENAME = "checkpoint.yaml"

from models.factory import ModelConfig, build_model_config_from_dict
from train.config import TrainConfig, train_config_to_dict
from train.paths import TrainPaths


@dataclass(frozen=True)
class CheckpointPaths:
    directory: Path
    manifest_path: Path
    model_state_path: Path
    model_ema_state_path: Path
    nextlat_dynamics_state_path: Path


def ensure_experiment_dirs(paths: TrainPaths) -> None:
    paths.experiments_root.mkdir(parents=True, exist_ok=True)
    paths.experiment_dir.mkdir(parents=True, exist_ok=True)
    paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    paths.log_dir.mkdir(parents=True, exist_ok=True)
    paths.cache_dir.mkdir(parents=True, exist_ok=True)


def write_experiment_config(config: TrainConfig, paths: TrainPaths) -> Path:
    with paths.train_config_path.open("w") as handle:
        yaml.safe_dump(train_config_to_dict(config), handle, sort_keys=True)
    return paths.train_config_path


def write_model_config(model_config: ModelConfig, paths: TrainPaths) -> Path:
    with paths.model_config_path.open("w") as handle:
        yaml.safe_dump(asdict(model_config), handle, sort_keys=True)
    return paths.model_config_path


def resolve_checkpoint_paths(checkpoint_dir: Path, label: str | int) -> CheckpointPaths:
    directory = checkpoint_dir / f"checkpoint_{label}"
    return CheckpointPaths(
        directory=directory,
        manifest_path=directory / MANIFEST_FILENAME,
        model_state_path=directory / MODEL_STATE_FILENAME,
        model_ema_state_path=directory / MODEL_EMA_STATE_FILENAME,
        nextlat_dynamics_state_path=directory / NEXTLAT_DYNAMICS_STATE_FILENAME,
    )


def resolve_ema_device(ema_device: str, model_device: torch.device) -> torch.device:
    """Resolve the EMA device: 'auto' follows the model device, else the explicit value."""
    if ema_device == "auto":
        return model_device
    return torch.device(ema_device)


def _set_module_buffer(root: torch.nn.Module, dotted_name: str, tensor: torch.Tensor) -> None:
    """Replace a (possibly differently shaped) registered buffer by its dotted name."""
    *parent_path, leaf = dotted_name.split(".")
    module = root
    for attr in parent_path:
        module = getattr(module, attr)
    setattr(module, leaf, tensor)


class EmaModel:
    """Exponential moving average of a model's parameters.

    Holds a shadow copy of the tracked module (on ``device``) that is nudged
    toward the live weights after every optimizer step via
    ``ema <- decay * ema + (1 - decay) * live``. Buffers are not averaged; they
    are synced from the live model on demand (before saving) so caches such as
    rotary tables are not left stale.
    """

    def __init__(self, model: torch.nn.Module, decay: float, device: torch.device):
        self.decay = decay
        self.device = device
        self.module = copy.deepcopy(model).to(device)
        self.module.requires_grad_(False)
        self.module.eval()

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for ema_param, model_param in zip(
            self.module.parameters(), model.parameters(), strict=True
        ):
            ema_param.lerp_(model_param.detach().to(self.device), 1.0 - self.decay)

    @torch.no_grad()
    def sync_buffers(self, model: torch.nn.Module) -> None:
        # Buffers (rotary/mask caches) can be lazily sized or resized, so match
        # by name and reassign when the shape differs rather than requiring an
        # in-place copy into a stale buffer.
        ema_buffers = dict(self.module.named_buffers())
        for name, model_buffer in model.named_buffers():
            source = model_buffer.detach().to(self.device)
            ema_buffer = ema_buffers.get(name)
            if ema_buffer is not None and ema_buffer.shape == source.shape:
                ema_buffer.copy_(source)
            else:
                _set_module_buffer(self.module, name, source.clone())

    def state_dict(self) -> dict:
        return self.module.state_dict()

    @torch.no_grad()
    def load_state_dict(self, state_dict: dict) -> None:
        self.module.load_state_dict(state_dict)


def read_experiment_config(paths: TrainPaths) -> TrainConfig:
    with paths.train_config_path.open("r") as handle:
        raw_config = yaml.safe_load(handle)
    return TrainConfig(**raw_config)


def read_model_config(paths: TrainPaths) -> ModelConfig:
    with paths.model_config_path.open("r") as handle:
        raw_config = yaml.safe_load(handle)
    train_config = read_experiment_config(paths)
    return build_model_config_from_dict(train_config.model_type, raw_config)


def load_model_checkpoint(
    model: torch.nn.Module,
    checkpoint_dir: Path,
    label: str | int,
    map_location: str | torch.device = "cpu",
) -> CheckpointPaths:
    checkpoint_paths = resolve_checkpoint_paths(checkpoint_dir, label)
    state_dict = torch.load(checkpoint_paths.model_state_path, map_location=map_location, weights_only=True)
    model.load_state_dict(state_dict)
    return checkpoint_paths


def load_warm_start_state_dicts(
    model: torch.nn.Module,
    checkpoint_directory: Path,
    use_ema: bool,
    nextlat_dynamics_model: torch.nn.Module | None = None,
    map_location: str | torch.device = "cpu",
) -> tuple[Path, ...]:
    """Load all active model state for a fresh second-stage training run.

    The main language model loads EMA weights when ``use_ema`` and raw weights
    otherwise. Active NextLat dynamics modules load their separately saved raw
    states because EMA tracks only the main language model. Optimizer and
    scheduler state are intentionally not loaded. Every artifact to load must
    be declared in the checkpoint manifest (``checkpoint.yaml``); a file that
    is merely present on disk is rejected as stale.
    """
    main_state_filename = MODEL_EMA_STATE_FILENAME if use_ema else MODEL_STATE_FILENAME
    main_manifest_key = "model_ema_state" if use_ema else "model_state"
    state_modules = [
        (checkpoint_directory / main_state_filename, model),
    ]
    required_manifest_entries = {main_manifest_key: main_state_filename}
    if nextlat_dynamics_model is not None:
        state_modules.append(
            (checkpoint_directory / NEXTLAT_DYNAMICS_STATE_FILENAME, nextlat_dynamics_model)
        )
        required_manifest_entries["nextlat_dynamics_state"] = NEXTLAT_DYNAMICS_STATE_FILENAME

    # The manifest is the source of truth for what a checkpoint contains.
    # save_training_checkpoint never deletes files, so a state file can be
    # left on disk by an earlier run at the same label (e.g. a stale
    # model_ema.pt after the label was overwritten without EMA); loading
    # such a file would silently warm-start from another run's weights.
    manifest_path = checkpoint_directory / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Warm-start checkpoint manifest not found: {manifest_path}")
    with manifest_path.open("r") as handle:
        manifest = yaml.safe_load(handle)
    if not isinstance(manifest, dict):
        raise ValueError(f"Checkpoint manifest is not a mapping: {manifest_path}")
    undeclared = [
        filename
        for key, filename in required_manifest_entries.items()
        if manifest.get(key) != filename
    ]
    if undeclared:
        formatted = ", ".join(undeclared)
        raise ValueError(
            "Warm-start state files not declared in the checkpoint manifest "
            f"({manifest_path}); the checkpoint is stale or was written by a "
            f"run without this state. Undeclared: {formatted}"
        )

    missing_paths = [
        state_path for state_path, _ in state_modules if not state_path.is_file()
    ]
    if missing_paths:
        formatted_paths = "\n".join(f"  {path}" for path in missing_paths)
        raise FileNotFoundError(f"Warm-start state files not found:\n{formatted_paths}")

    loaded_paths = []
    for state_path, module in state_modules:
        state_dict = torch.load(state_path, map_location=map_location, weights_only=True)
        module.load_state_dict(state_dict)
        loaded_paths.append(state_path)
    return tuple(loaded_paths)


def save_training_checkpoint(
    model: torch.nn.Module,
    paths: TrainPaths,
    label: str | int,
    nextlat_dynamics_model: torch.nn.Module | None = None,
    ema_model: EmaModel | None = None,
) -> Path:
    """Save a model checkpoint for evaluation and export.

    Checkpoints contain only inference-relevant state: the main language
    model, the EMA weights when EMA is enabled, the NextLat dynamics module
    when active, and a manifest. Optimizer and scheduler state are not
    persisted.
    """
    checkpoint_paths = resolve_checkpoint_paths(paths.checkpoint_dir, label)
    checkpoint_paths.directory.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), checkpoint_paths.model_state_path)
    if ema_model is not None:
        ema_model.sync_buffers(model)
        torch.save(ema_model.state_dict(), checkpoint_paths.model_ema_state_path)
    if nextlat_dynamics_model is not None:
        torch.save(nextlat_dynamics_model.state_dict(), checkpoint_paths.nextlat_dynamics_state_path)

    manifest = {
        "label": str(label),
        "model_state": checkpoint_paths.model_state_path.name,
        "train_config": str(paths.train_config_path.relative_to(paths.experiment_dir)),
        "model_config": str(paths.model_config_path.relative_to(paths.experiment_dir)),
    }
    if ema_model is not None:
        manifest["model_ema_state"] = checkpoint_paths.model_ema_state_path.name
    if nextlat_dynamics_model is not None:
        manifest["nextlat_dynamics_state"] = checkpoint_paths.nextlat_dynamics_state_path.name
    with checkpoint_paths.manifest_path.open("w") as handle:
        yaml.safe_dump(manifest, handle, sort_keys=True)

    return checkpoint_paths.directory
