"""Training setup and loop for Relational BabyLM language model training."""

from __future__ import annotations

import logging
import random
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import time

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerBase

from models.attention.dat.config import DatLMConfig
from models.attention.transformer.config import SelfAttentionLMConfig
from models.factory import ModelConfig, build_model
from train.checkpoints import (
    EmaModel,
    ensure_experiment_dirs,
    load_warm_start_state_dicts,
    resolve_ema_device,
    save_training_checkpoint,
    write_experiment_config,
    write_model_config,
)

from train.config import (
    SMALL_CORPUS_IDS,
    TrainConfig,
)
from train.data import (
    FullBabyLMDataset,
    TrainDataArtifacts,
    build_dataloader_for_dataset,
    load_train_data,
)
from train.nextlat import (
    NextLatDynamicsModel,
    NextLatLossConfig,
    NextLatLossResult,
    compute_nextlat_loss,
)
from train.optim import (
    build_optimizer_and_scheduler,
    resolve_num_warmup_steps,
    resolve_scheduled_total_training_steps,
)
from train.paths import TrainPaths, materialize_train_asset_config, resolve_train_paths

logger = logging.getLogger(__name__)


def resolve_training_device(config: TrainConfig) -> torch.device:
    """Resolve the torch device for training: CUDA if available, else CPU."""
    return get_default_device()


PROMPTS = (
    "The boy found",
    "Once upon a time there was a",
    "She looked at him and said",
)
TRAIN_LOG_CHECKPOINTS_PER_EPOCH = 10


def resolve_epoch_num_steps(epoch_size: int, max_train_steps: int | None) -> int:
    """Resolve actual batches run in an epoch after the optional step cap."""
    return min(epoch_size, max_train_steps) if max_train_steps is not None else epoch_size


def should_stop_epoch_after_step(completed_steps: int, max_train_steps: int | None) -> bool:
    """Return True when the per-epoch step cap has been reached."""
    return max_train_steps is not None and completed_steps >= max_train_steps


def finalize_epoch(
    config: TrainConfig,
    paths: TrainPaths,
    model: torch.nn.Module,
    nextlat_dynamics_model: NextLatDynamicsModel | None,
    ema_model: EmaModel | None,
    wandb_module,
    epoch: int,
    train_metrics: dict,
    global_step_offset: int,
    tokenizer,
) -> None:
    """Rank-0 epoch finalization: metrics, checkpoint, samples.

    Runs once per epoch. Saves the epoch metrics, an epoch checkpoint,
    and generates text samples.
    """
    torch.save(train_metrics, paths.log_dir / f"epoch_{epoch}_metrics.pth")
    save_training_checkpoint(
        model=model,
        paths=paths,
        label=epoch,
        nextlat_dynamics_model=nextlat_dynamics_model,
        ema_model=ema_model,
    )
    generate_samples(
        model=model,
        tokenizer=tokenizer,
        epoch=epoch,
        step=global_step_offset,
        wandb_module=wandb_module,
    )


def build_model_parameter_summary(model: torch.nn.Module) -> str:
    total_parameters = 0
    trainable_parameters = 0
    total_by_group: dict[str, int] = {}
    trainable_by_group: dict[str, int] = {}

    for name, parameter in model.named_parameters():
        parameter_count = parameter.numel()
        group_name = name.split(".", maxsplit=1)[0]
        total_parameters += parameter_count
        total_by_group[group_name] = total_by_group.get(group_name, 0) + parameter_count
        if parameter.requires_grad:
            trainable_parameters += parameter_count
            trainable_by_group[group_name] = trainable_by_group.get(group_name, 0) + parameter_count

    lines = [
        f"total parameters: {total_parameters:,}",
        f"trainable parameters: {trainable_parameters:,}",
        f"non-trainable parameters: {total_parameters - trainable_parameters:,}",
    ]
    if total_by_group:
        lines.append("top-level parameter counts:")
        for group_name, group_total in total_by_group.items():
            group_trainable = trainable_by_group.get(group_name, 0)
            lines.append(f"  {group_name}: total={group_total:,}, trainable={group_trainable:,}")
    return "\n".join(lines)


def build_model_runtime_summary(model: torch.nn.Module) -> str:
    config = model.config
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    lines = [
        f"model: {type(model).__name__}",
        f"vocab_size: {config.vocab_size:,}",
        f"max_seq_len: {config.max_seq_len:,}",
        f"hidden_dim: {config.hidden_dim:,}",
        f"layers: {config.n_layers:,}",
    ]

    if isinstance(config, DatLMConfig):
        lines.extend(
            [
                f"heads: {config.n_heads_sa} self-attention + {config.n_heads_ra} relational-attention "
                f"(total {config.total_n_heads})",
                f"head_dim: {config.head_dim:,}",
                f"positional_encoding: {config.pe_type}",
                f"sequence_boundary_policy: {config.sequence_boundary_policy}",
                f"norm_type: {config.norm_type}",
                f"symbol_retrieval: {config.symbol_retrieval}",
                f"positional_symbols_sinusoidal: {config.positional_symbols_sinusoidal}",
                f"relative_symbols_rope: {config.relative_symbols_rope}",
                f"relsymbolic_neighborhood_size: {config.relsymbolic_neighborhood_size}",
                f"symbolic_attn_n_heads: {config.resolved_symbolic_attn_n_heads}",
                f"ra_type: {config.ra_type}",
                f"ra_rel_activation: {config.ra_rel_activation}",
                f"biases: qkv={config.use_bias_qkv} out={config.use_bias_out} ffn={config.use_bias_ffn}",
                f"ffn_activation: {config.ffn_activation}",
                f"ffn_hidden_dim: {config.resolved_ffn_hidden_dim:,}",
                f"ffn_hidden_dim_mode: {config.ffn_hidden_dim_mode}",
                f"init_scheme: {config.init_scheme}",
            ]
        )
    elif isinstance(config, SelfAttentionLMConfig):
        lines.extend(
            [
                f"heads: {config.n_heads} self-attention",
                f"head_dim: {config.hidden_dim // config.n_heads:,}",
                f"positional_encoding: {config.pe_type}",
                f"sequence_boundary_policy: {config.sequence_boundary_policy}",
            ]
        )

    if hasattr(model, "token_embeddings") and hasattr(model, "lm_head"):
        lines.append(f"tied_lm_head: {model.lm_head.weight is model.token_embeddings.weight}")
    lines.extend(
        [
            f"parameters: {parameter_count:,}",
            f"parameter storage: {parameter_bytes / (1024 * 1024):.2f} MiB",
        ]
    )
    return "\n".join(lines)


def log_model_structure(model: torch.nn.Module) -> None:
    logger.info(f"Model summary:\n{build_model_runtime_summary(model)}")
    logger.info(f"Model structure:\n{model}")
    logger.info(f"Model parameters:\n{build_model_parameter_summary(model)}")


@dataclass(frozen=True)
class TrainingSetup:
    config: TrainConfig
    paths: TrainPaths
    tokenizer: PreTrainedTokenizerBase
    data_artifacts: TrainDataArtifacts
    device: torch.device


@dataclass(frozen=True)
class ClmForwardLossResult:
    loss: torch.Tensor
    loss_metrics: dict[str, torch.Tensor]


@dataclass(frozen=True)
class BatchLossResult:
    """Result of one training batch's forward/loss computation."""
    loss: torch.Tensor
    loss_metrics: dict[str, torch.Tensor]


def get_default_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def resolve_runtime_config(config: TrainConfig) -> TrainConfig:
    resolved_config = materialize_train_asset_config(config)
    if resolved_config.seed != -1:
        return resolved_config
    generated_seed = random.randint(0, 1000000)
    return replace(resolved_config, seed=generated_seed)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_training_layout(config: TrainConfig) -> TrainPaths:
    paths = resolve_train_paths(config)
    ensure_experiment_dirs(paths)
    write_experiment_config(config, paths)
    return paths


def prepare_training_setup(config: TrainConfig) -> TrainingSetup:
    resolved_config = resolve_runtime_config(config)
    seed_everything(resolved_config.seed)
    paths = prepare_training_layout(resolved_config)
    data_artifacts = load_train_data(resolved_config, paths)
    return TrainingSetup(
        config=resolved_config,
        paths=paths,
        tokenizer=data_artifacts.tokenizer,
        data_artifacts=data_artifacts,
        device=resolve_training_device(resolved_config),
    )


def build_model_config(config: TrainConfig, tokenizer: PreTrainedTokenizerBase) -> ModelConfig:
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must define eos_token_id if pad_token_id is missing.")
        pad_token_id = tokenizer.eos_token_id

    if tokenizer.bos_token_id is None:
        raise ValueError("Tokenizer must define bos_token_id.")
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id.")

    # The model must accommodate the full sequence length, since positional
    # buffers are sized once at build time.
    max_seq_len = config.datapoint_length + 2 if config.nextlat_enabled else config.datapoint_length + 1

    common_kwargs = {
        "vocab_size": len(tokenizer),
        "max_seq_len": max_seq_len,
        "pe_type": config.pe_type,
        "hidden_dim": config.hidden_dim,
        "n_layers": config.n_layers,
        "dropout": config.dropout,
        "norm_type": config.norm_type,
        "use_bias_qkv": config.use_bias_qkv,
        "use_bias_out": config.use_bias_out,
        "use_bias_ffn": config.use_bias_ffn,
        "rope_theta": config.rope_theta,
        "max_rel_pos": config.max_rel_pos,
        "sequence_boundary_policy": config.sequence_boundary_policy,
        "tie_lm_head": (
            not config.nextlat_enabled
            if config.tie_lm_head is None
            else config.tie_lm_head
        ),
        "pad_token_id": pad_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if config.model_type == "self_attention":
        return SelfAttentionLMConfig(
            **common_kwargs,
            n_heads=config.n_heads,
        )
    if config.model_type == "dat":
        return DatLMConfig(
            **common_kwargs,
            n_heads_sa=config.n_heads_sa,
            n_heads_ra=config.n_heads_ra,
            ffn_hidden_dim_mode=config.ffn_hidden_dim_mode,
            ffn_activation=config.ffn_activation,
            symbol_dim=config.symbol_dim,
            n_symbols=config.n_symbols,
            symbolic_attn_n_heads=config.symbolic_attn_n_heads,
            symbol_retrieval=config.symbol_retrieval,
            symbolic_use_bias=config.symbolic_use_bias,
            positional_symbols_sinusoidal=config.positional_symbols_sinusoidal,
            relative_symbols_rope=config.relative_symbols_rope,
            relsymbolic_rel_n_heads=config.relsymbolic_rel_n_heads,
            relsymbolic_symbolic_attn_n_heads=config.relsymbolic_symbolic_attn_n_heads,
            relsymbolic_neighborhood_size=config.relsymbolic_neighborhood_size,
            relsymbolic_include_self=config.relsymbolic_include_self,
            relsymbolic_normalize_rels=config.relsymbolic_normalize_rels,
            relsymbolic_trainable_symbols=config.relsymbolic_trainable_symbols,
            relsymbolic_dropout=config.relsymbolic_dropout,
            relsymbolic_rel_scale=config.relsymbolic_rel_scale,
            relsymbolic_symbolic_attn_scale=config.relsymbolic_symbolic_attn_scale,
            relsymbolic_use_bias=config.relsymbolic_use_bias,
            ra_type=config.ra_type,
            ra_n_relations=config.ra_n_relations,
            ra_rel_activation=config.ra_rel_activation,
            ra_symmetric_rels=config.ra_symmetric_rels,
            init_scheme=config.init_scheme,
        )
    raise ValueError(f"Unsupported model_type: {config.model_type}")


def setup_wandb(config: TrainConfig):
    if not config.use_wandb:
        return None
    import wandb

    wandb.init(
        name=config.wandb_experiment_name,
        project=config.wandb_project_name,
        config=asdict(config),
    )
    return wandb


def build_nextlat_loss_config(
    config: TrainConfig,
    model: torch.nn.Module,
) -> NextLatLossConfig:
    raw_model = model
    return NextLatLossConfig(
        lambda_mse=config.nextlat_lambda_mse,
        lambda_kl=config.nextlat_lambda_kl,
        lambda_ce=config.nextlat_lambda_ce,
        horizon=config.nextlat_horizon,
        eos_token_id=raw_model.config.eos_token_id,
        sequence_boundary_policy=config.sequence_boundary_policy,
    )


def build_nextlat_dynamics_model(
    config: TrainConfig,
    model: torch.nn.Module,
) -> NextLatDynamicsModel | None:
    if not config.nextlat_enabled:
        return None
    raw_model = model
    return NextLatDynamicsModel(
        hidden_dim=raw_model.config.hidden_dim,
        dropout=config.dropout,
        proj_factor=config.nextlat_proj_factor,
        use_bias=False,
    ).to(next(raw_model.parameters()).device)


def build_trainable_modules(
    model: torch.nn.Module,
    nextlat_dynamics_model: NextLatDynamicsModel | None,
) -> tuple[torch.nn.Module, ...]:
    modules = [model]
    if nextlat_dynamics_model is not None:
        modules.append(nextlat_dynamics_model)
    return tuple(modules)


def get_autocast_context(device: torch.device, cuda_autocast: bool):
    if not cuda_autocast or device.type != "cuda":
        return nullcontext()
    autocast_dtype = resolve_cuda_autocast_dtype(cuda_autocast)
    if autocast_dtype is None:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=autocast_dtype)


def resolve_cuda_autocast_dtype(cuda_autocast: bool) -> torch.dtype | None:
    return select_cuda_autocast_dtype(
        cuda_autocast=cuda_autocast,
        cuda_available=torch.cuda.is_available(),
        bf16_supported=torch.cuda.is_bf16_supported(),
    )


def select_cuda_autocast_dtype(
    cuda_autocast: bool,
    cuda_available: bool,
    bf16_supported: bool,
) -> torch.dtype | None:
    if not cuda_autocast:
        return None
    if not cuda_available:
        return None
    if bf16_supported:
        return torch.bfloat16
    return None


def unpack_batch(
    minibatch: tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    ],
    device: torch.device,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
]:
    input_tokens, target_tokens, target_mask, attention_mask = minibatch
    return (
        input_tokens.to(device),
        target_tokens.to(device),
        target_mask.to(device),
        attention_mask.to(device),
    )


def log_train_metrics(
    wandb_module,
    loss: float,
    step: int,
    start_time: float,
) -> None:
    if wandb_module is None:
        return
    time_elapsed = (time() - start_time) / 60.0
    wandb_module.log(
        {
            "train_metrics/time_elapsed": time_elapsed,
            "train_metrics/batch_train_loss": loss,
        },
        step=step,
    )


def format_duration_seconds(duration_seconds: float) -> str:
    if duration_seconds < 0.0:
        raise ValueError(f"duration_seconds must be non-negative, got {duration_seconds}")
    total_seconds = int(round(duration_seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def estimate_remaining_seconds(
    completed_steps: int,
    total_steps: int,
    elapsed_seconds: float,
) -> float:
    if completed_steps <= 0:
        raise ValueError(f"completed_steps must be positive, got {completed_steps}")
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    if completed_steps > total_steps:
        raise ValueError(
            f"completed_steps must not exceed total_steps, got {completed_steps} > {total_steps}"
        )
    if elapsed_seconds <= 0.0:
        raise ValueError(f"elapsed_seconds must be positive, got {elapsed_seconds}")
    return (elapsed_seconds / completed_steps) * (total_steps - completed_steps)


def resolve_train_log_interval(total_steps: int) -> int:
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    return max(1, total_steps // TRAIN_LOG_CHECKPOINTS_PER_EPOCH)


def build_train_progress_postfix(
    total_loss: float,
    total_tokens: int,
    elapsed_seconds: float,
    current_lr: float,
    grad_norm: float | None = None,
    remaining_seconds: float | None = None,
) -> dict[str, str]:
    if total_tokens <= 0:
        raise ValueError(f"total_tokens must be positive, got {total_tokens}")
    if elapsed_seconds <= 0.0:
        raise ValueError(f"elapsed_seconds must be positive, got {elapsed_seconds}")
    if current_lr < 0.0:
        raise ValueError(f"current_lr must be non-negative, got {current_lr}")
    postfix = {
        "loss": f"{total_loss / total_tokens:.4f}",
        "tok/s": f"{total_tokens / elapsed_seconds:.1f}",
        "lr": f"{current_lr:.2e}",
    }
    if grad_norm is not None:
        if grad_norm < 0.0:
            raise ValueError(f"grad_norm must be non-negative, got {grad_norm}")
        postfix["gnorm"] = f"{grad_norm:.2f}"
    if remaining_seconds is not None:
        if remaining_seconds < 0.0:
            raise ValueError(f"remaining_seconds must be non-negative, got {remaining_seconds}")
        postfix["eta"] = format_duration_seconds(remaining_seconds)
    return postfix


@torch.no_grad()
def generate_samples(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    epoch: int,
    step: int,
    wandb_module,
) -> None:
    model.eval()
    rows: list[list[str]] = []

    logger.info(f"Generated samples after epoch {epoch}")
    for prompt in PROMPTS:
        encoding = tokenizer(prompt, return_tensors="pt")
        input_ids = encoding["input_ids"].to(model.device)
        attention_mask = encoding["attention_mask"].to(model.device)
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.9,
            eos_token_id=tokenizer.eos_token_id,
        )
        generated = tokenizer.decode(output[0], skip_special_tokens=True)
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Generated: {generated}")
        rows.append([prompt, generated])

    if wandb_module is not None:
        table = wandb_module.Table(columns=["prompt", "generated"], data=rows)
        wandb_module.log({"samples": table}, step=step)

    model.train()


def maybe_save_progress_checkpoint(
    config: TrainConfig,
    completed_steps: int,
    num_steps: int,
    model: torch.nn.Module,
    nextlat_dynamics_model: NextLatDynamicsModel | None,
    paths: TrainPaths,
    ema_model: EmaModel | None = None,
) -> None:
    label = resolve_progress_checkpoint_label(
        config.corpus_id,
        completed_steps,
        num_steps,
    )
    if label is None:
        return
    save_training_checkpoint(
        model,
        paths,
        label,
        nextlat_dynamics_model=nextlat_dynamics_model,
        ema_model=ema_model,
    )


def resolve_progress_checkpoint_label(
    corpus_id: str,
    completed_steps: int,
    total_steps: int,
) -> str | None:
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    if completed_steps <= 0:
        return None
    if completed_steps > total_steps:
        raise ValueError(
            f"completed_steps must not exceed total_steps, got {completed_steps} > {total_steps}"
        )
    # strict_small checkpoints are labeled on a 10M-word budget; strict on 100M.
    if corpus_id in SMALL_CORPUS_IDS:
        if completed_steps == total_steps:
            return "10M"
        interval = total_steps // 10
        label_index = completed_steps // interval if interval > 0 else 0
        if interval > 0 and completed_steps % interval == 0 and label_index < 10:
            return f"{label_index}M"
        return None

    if completed_steps == total_steps:
        return "100M"

    interval_1m = total_steps // 100
    if interval_1m > 0 and completed_steps % interval_1m == 0 and completed_steps // interval_1m < 10:
        return f"{completed_steps // interval_1m}M"

    interval_10m = total_steps // 10
    label_index = completed_steps // interval_10m if interval_10m > 0 else 0
    if interval_10m > 0 and completed_steps % interval_10m == 0 and label_index < 10:
        return f"{10 * label_index}M"

    return None


def compute_gradient_norm(parameters) -> float:
    gradient_norms = []
    for parameter in parameters:
        if parameter.grad is None:
            continue
        gradient_norms.append(parameter.grad.detach().norm(2))
    if not gradient_norms:
        raise ValueError(
            "compute_gradient_norm found no parameters with gradients; "
            "backward() was not called or zero_grad(set_to_none=True) ran first."
        )
    return float(torch.stack(gradient_norms).norm(2).item())


def collect_trainable_parameters(trainable_modules: tuple[torch.nn.Module, ...]) -> list[torch.nn.Parameter]:
    return [
        parameter
        for module in trainable_modules
        for parameter in module.parameters()
        if parameter.requires_grad
    ]


def nextlat_result_to_metrics(result: NextLatLossResult) -> dict[str, torch.Tensor]:
    return {
        "loss": result.loss,
        "next_token_loss": result.next_token_loss,
        "nextlat_hidden_loss": result.hidden_loss,
        "nextlat_kl_loss": result.kl_loss,
        "nextlat_auxiliary_token_loss": result.auxiliary_token_loss,
    }


def _clm_forward_loss(
    model: nn.Module,
    input_tokens: torch.Tensor,
    targets_for_loss: torch.Tensor,
    target_tokens: torch.Tensor,
    target_mask: torch.Tensor,
    attention_mask: torch.Tensor,
    nextlat_dynamics_model: NextLatDynamicsModel | None,
    nextlat_loss_config: NextLatLossConfig | None,
) -> ClmForwardLossResult:
    """Run the CLM forward/loss path.

    Returns the loss used for backward in ``loss`` and per-metric values
    in ``loss_metrics``.
    """
    if nextlat_dynamics_model is None:
        _, clm_loss = model(
            input_tokens,
            targets=targets_for_loss,
            attention_mask=attention_mask,
        )
        if clm_loss is None:
            raise ValueError("Model returned no loss during training.")
        loss_metrics = {"loss": clm_loss, "clm_loss": clm_loss}
    else:
        if nextlat_loss_config is None:
            raise ValueError("NextLat loss config is missing while NextLat dynamics is enabled.")
        nextlat_result = compute_nextlat_loss(
            model=model,
            dynamics_model=nextlat_dynamics_model,
            input_tokens=input_tokens,
            target_tokens=target_tokens,
            target_mask=target_mask,
            config=nextlat_loss_config,
        )
        clm_loss = nextlat_result.loss
        loss_metrics = nextlat_result_to_metrics(nextlat_result)
        # loss_metrics["loss"] is NTP-CE-only (next_token_loss), not the
        # total loss, so the progress bar and W&B show base-CE comparable
        # across NextLat and non-NextLat runs.  The returned ``clm_loss``
        # (used for backward) is still the total.
        loss_metrics["loss"] = nextlat_result.next_token_loss
        loss_metrics["clm_loss"] = nextlat_result.next_token_loss
    return ClmForwardLossResult(loss=clm_loss, loss_metrics=loss_metrics)


def _compute_batch_loss(
    model: nn.Module,
    input_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    target_mask: torch.Tensor,
    attention_mask: torch.Tensor,
    nextlat_dynamics_model: NextLatDynamicsModel | None,
    nextlat_loss_config: NextLatLossConfig | None,
) -> BatchLossResult:
    """Compute the forward/loss for one CLM training batch.

    Handles pure CLM with or without causal NextLat.
    """
    clm_result = _clm_forward_loss(
        model=model,
        input_tokens=input_tokens,
        targets_for_loss=target_tokens.masked_fill(~target_mask, -1),
        target_tokens=target_tokens,
        target_mask=target_mask,
        attention_mask=attention_mask,
        nextlat_dynamics_model=nextlat_dynamics_model,
        nextlat_loss_config=nextlat_loss_config,
    )
    loss_metrics = dict(clm_result.loss_metrics)
    loss = clm_result.loss
    return BatchLossResult(loss=loss, loss_metrics=loss_metrics)


def _step_optimizer(
    optimizer: torch.optim.Optimizer,
    scheduler,
    trainable_parameters: list[torch.nn.Parameter],
    gradient_clip_norm: float,
    optimizer_step_count: int,
    model: torch.nn.Module,
    ema_model: EmaModel | None,
) -> tuple[float | None, int]:
    """Clip gradients, step the optimizer and scheduler, and update the EMA.

    Returns ``(grad_norm, new_optimizer_step_count)``; the caller owns the
    loop-local counter. When EMA is enabled, the shadow weights are nudged
    toward the live model after every optimizer step.
    """
    if gradient_clip_norm != -1:
        grad_norm = float(nn.utils.clip_grad_norm_(trainable_parameters, gradient_clip_norm))
    else:
        grad_norm = compute_gradient_norm(trainable_parameters)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    if ema_model is not None:
        ema_model.update(model)
    return grad_norm, optimizer_step_count + 1


def _build_train_postfix(
    metric_totals: dict[str, float],
    total_tokens: int,
    grad_norm: float | None,
    completed_steps: int,
    num_steps: int,
    epoch_start_time: float,
    optimizer: torch.optim.Optimizer,
) -> dict[str, str]:
    """Build the tqdm postfix dict."""
    elapsed_seconds = max(time() - epoch_start_time, 1e-12)
    remaining_seconds = estimate_remaining_seconds(
        completed_steps=min(completed_steps, num_steps),
        total_steps=num_steps,
        elapsed_seconds=elapsed_seconds,
    )
    current_lr = float(optimizer.param_groups[0]["lr"])
    postfix = build_train_progress_postfix(
        total_loss=metric_totals["loss"],
        total_tokens=total_tokens,
        elapsed_seconds=elapsed_seconds,
        current_lr=current_lr,
        grad_norm=grad_norm,
        remaining_seconds=remaining_seconds,
    )
    if "clm_loss" in metric_totals:
        postfix["clm"] = f"{metric_totals['clm_loss'] / total_tokens:.4f}"
    return postfix


def _log_wandb_step(
    wandb_module,
    train_step: int,
    temp_loss: float,
    temp_tokens: float,
    global_step_offset: int,
    start_time: float,
    metric_totals: dict[str, float],
    total_tokens: int,
) -> tuple[float, float]:
    """Log training metrics to W&B and reset the per-log-interval accumulators.

    Returns ``(new_temp_loss, new_temp_tokens)`` so the caller updates its
    loop-local accumulators.
    """
    if wandb_module is None or train_step % 10 != 0 or train_step == 0 or temp_tokens <= 0:
        return temp_loss, temp_tokens
    steps = global_step_offset + train_step
    log_train_metrics(wandb_module, temp_loss / temp_tokens, steps, start_time)
    clm_metrics = {}
    if "clm_loss" in metric_totals:
        clm_metrics["train_metrics/clm_loss"] = metric_totals["clm_loss"] / total_tokens
    if total_tokens > 0:
        clm_metrics["train_metrics/clm_tokens"] = total_tokens
    if clm_metrics:
        wandb_module.log(clm_metrics, step=steps)
    return 0.0, 0.0


def _log_train_step_line(
    epoch: int,
    completed_steps: int,
    num_steps: int,
    log_interval: int,
    postfix: dict[str, str],
) -> None:
    """Emit the per-log-interval logger.info line."""
    if completed_steps % log_interval != 0 and completed_steps != num_steps:
        return
    grad_norm_text = postfix.get("gnorm", "n/a")
    clm_loss_text = postfix.get("clm", "")
    logger.info(
        f"epoch {epoch} step {completed_steps}/{num_steps} "
        f"loss {postfix['loss']} tok/s {postfix['tok/s']} "
        f"lr {postfix['lr']} gnorm {grad_norm_text} eta {postfix['eta']}"
        + (f" clm {clm_loss_text}" if clm_loss_text else "")
    )


def train_epoch(
    config: TrainConfig,
    paths: TrainPaths,
    model: torch.nn.Module,
    nextlat_dynamics_model: NextLatDynamicsModel | None,
    optimizer: torch.optim.Optimizer,
    scheduler,
    dataloader,
    epoch: int,
    start_time: float,
    wandb_module,
    ema_model: EmaModel | None = None,
    global_step_offset: int = 0,
    total_training_steps: int = 0,
    optimizer_step_count: int = 0,
    global_batch_step: int = 0,
) -> dict[str, float]:
    device = next(model.parameters()).device
    model.train()
    if nextlat_dynamics_model is not None:
        nextlat_dynamics_model.train()
    nextlat_loss_config = (
        build_nextlat_loss_config(config, model)
        if nextlat_dynamics_model is not None
        else None
    )
    trainable_modules = build_trainable_modules(model, nextlat_dynamics_model)
    trainable_parameters = collect_trainable_parameters(trainable_modules)
    metric_totals: dict[str, float] = {"loss": 0.0}
    total_tokens = 0
    temp_loss = 0.0
    temp_tokens = 0
    num_steps = resolve_epoch_num_steps(len(dataloader), config.max_train_steps)
    gradient_accumulation_steps = config.gradient_accumulation_steps
    log_interval = resolve_train_log_interval(num_steps)
    epoch_start_time = time()
    # Throttle progress output so a redirected log (SBATCH --output) gets ~10 lines
    # per epoch instead of one per step. mininterval=0 makes miniters govern the
    # cadence; set_postfix(refresh=False) below stops it from forcing a per-step write.
    progress_log_interval = max(1, num_steps // 10)
    progress = tqdm(
        dataloader,
        total=num_steps,
        desc=f"epoch {epoch}",
        dynamic_ncols=True,
        miniters=progress_log_interval,
        mininterval=0,
    )

    for train_step, minibatch in enumerate(progress):
        input_tokens, target_tokens, target_mask, attention_mask = unpack_batch(minibatch, device)
        num_tokens = int(target_mask.sum().item())

        # Compute accumulation parameters BEFORE the forward pass so the
        # loss can be scaled by the microbatch count within the group.
        accumulation_group_start = train_step - (train_step % gradient_accumulation_steps)
        accumulation_count = min(gradient_accumulation_steps, num_steps - accumulation_group_start)
        completed_steps = train_step + 1
        should_step_optimizer = (
            completed_steps % gradient_accumulation_steps == 0
            or completed_steps == num_steps
        )
        with get_autocast_context(device, cuda_autocast=config.cuda_autocast):
            batch_result = _compute_batch_loss(
                model=model,
                input_tokens=input_tokens,
                target_tokens=target_tokens,
                target_mask=target_mask,
                attention_mask=attention_mask,
                nextlat_dynamics_model=nextlat_dynamics_model,
                nextlat_loss_config=nextlat_loss_config,
            )
            loss = batch_result.loss
            loss_metrics = batch_result.loss_metrics

        (loss / accumulation_count).backward()
        global_batch_step += 1
        grad_norm = None
        if should_step_optimizer:
            grad_norm, optimizer_step_count = _step_optimizer(
                optimizer=optimizer,
                scheduler=scheduler,
                trainable_parameters=trainable_parameters,
                gradient_clip_norm=config.gradient_clip_norm,
                optimizer_step_count=optimizer_step_count,
                model=model,
                ema_model=ema_model,
            )

        for metric_name, metric_value in loss_metrics.items():
            metric_totals.setdefault(metric_name, 0.0)
            metric_totals[metric_name] += float(metric_value.item()) * num_tokens
        total_tokens += num_tokens
        temp_loss += float(loss.item()) * num_tokens
        temp_tokens += num_tokens
        postfix = _build_train_postfix(
            metric_totals=metric_totals,
            total_tokens=total_tokens,
            grad_norm=grad_norm,
            completed_steps=completed_steps,
            num_steps=num_steps,
            epoch_start_time=epoch_start_time,
            optimizer=optimizer,
        )
        progress.set_postfix(postfix, refresh=False)
        temp_loss, temp_tokens = _log_wandb_step(
            wandb_module=wandb_module,
            train_step=train_step,
            temp_loss=temp_loss,
            temp_tokens=temp_tokens,
            global_step_offset=global_step_offset,
            start_time=start_time,
            metric_totals=metric_totals,
            total_tokens=total_tokens,
        )
        _log_train_step_line(
            epoch=epoch,
            completed_steps=completed_steps,
            num_steps=num_steps,
            log_interval=log_interval,
            postfix=postfix,
        )

        # Only save progress checkpoints on optimizer step boundaries,
        # not every microbatch. With gradient accumulation, firing on
        # every microbatch would rewrite the same checkpoint up to
        # ``gradient_accumulation_steps`` times.
        if total_training_steps > 0 and should_step_optimizer:
            maybe_save_progress_checkpoint(
                config=config,
                completed_steps=optimizer_step_count,
                num_steps=total_training_steps,
                model=model,
                nextlat_dynamics_model=nextlat_dynamics_model,
                paths=paths,
                ema_model=ema_model,
            )

        # Break immediately after the optimizer step that hits the cap.
        # This must happen after metric accounting (so the limiting
        # minibatch is included) and after the progress checkpoint
        # (so the final milestone checkpoint is saved).
        if (
            should_step_optimizer
            and config.max_optimizer_steps is not None
            and optimizer_step_count >= config.max_optimizer_steps
        ):
            logger.info(
                f"Reached max_optimizer_steps={config.max_optimizer_steps}, "
                f"stopping after optimizer_step={optimizer_step_count}"
            )
            break

        if should_stop_epoch_after_step(completed_steps, config.max_train_steps):
            break

    progress.close()

    if total_tokens == 0:
        raise ValueError("No train tokens were processed.")

    # Divide each metric by the total token count: every logged metric is a
    # per-token average over valid CLM positions.
    return {
        metric_name: metric_total / total_tokens
        for metric_name, metric_total in metric_totals.items()
    } | {
        "clm_tokens": total_tokens,
        "optimizer_step_count": optimizer_step_count,
        "global_batch_step": global_batch_step,
        "epoch_step": train_step + 1,
    }


def full_train_loop(
    setup: TrainingSetup,
    model: torch.nn.Module,
    nextlat_dynamics_model: NextLatDynamicsModel | None,
    optimizer: torch.optim.Optimizer,
    scheduler,
    wandb_module,
    dataset: FullBabyLMDataset,
    ema_model: EmaModel | None = None,
    total_training_steps: int = 0,
) -> None:
    start_time = time()
    global_step_offset = 0
    optimizer_step_count = 0
    global_batch_step = 0
    config = setup.config
    total_clm_tokens = 0

    for epoch in range(config.n_epochs):
        # Stop before starting an epoch at or beyond the optimizer-step limit.
        if (
            config.max_optimizer_steps is not None
            and optimizer_step_count >= config.max_optimizer_steps
        ):
            logger.info(
                f"Reached max_optimizer_steps={config.max_optimizer_steps}, "
                f"stopping before epoch {epoch} at optimizer_step={optimizer_step_count}"
            )
            break

        if setup.device.type == "cuda":
            torch.cuda.empty_cache()

        if config.batch_size <= 0:
            raise ValueError(
                f"batch_size must be positive, got {config.batch_size}"
            )
        epoch_batch_size = config.batch_size
        dataloader = build_dataloader_for_dataset(
            dataset, setup.tokenizer, epoch_batch_size,
            seed=setup.config.seed, epoch=epoch,
        )
        if hasattr(dataloader.sampler, "set_epoch"):
            dataloader.sampler.set_epoch(epoch)
        epoch_size = len(dataloader)
        logger.info(
            f"Epoch {epoch} size: {epoch_size} "
            f"(seq_length={config.datapoint_length}, batch_size={epoch_batch_size})"
        )

        train_metrics = train_epoch(
            config=config,
            paths=setup.paths,
            model=model,
            nextlat_dynamics_model=nextlat_dynamics_model,
            optimizer=optimizer,
            scheduler=scheduler,
            dataloader=dataloader,
            epoch=epoch,
            start_time=start_time,
            wandb_module=wandb_module,
            ema_model=ema_model,
            global_step_offset=global_step_offset,
            total_training_steps=total_training_steps,
            optimizer_step_count=optimizer_step_count,
            global_batch_step=global_batch_step,
        )
        # Advance by the full epoch length so the next epoch starts at
        # the correct batch offset.
        full_epoch_steps = resolve_epoch_num_steps(epoch_size, config.max_train_steps)
        global_step_offset += full_epoch_steps
        optimizer_step_count = train_metrics["optimizer_step_count"]
        global_batch_step = train_metrics["global_batch_step"]
        logger.info(f"Epoch {epoch} train loss: {train_metrics['loss']:.6f}")
        total_clm_tokens += train_metrics["clm_tokens"]
        train_metrics["total_clm_tokens"] = total_clm_tokens
        # Stop the outer loop if the optimizer-step limit was reached
        # during this epoch.
        if (
            config.max_optimizer_steps is not None
            and optimizer_step_count >= config.max_optimizer_steps
        ):
            logger.info(
                f"Reached max_optimizer_steps={config.max_optimizer_steps}, "
                f"stopping after epoch {epoch} at optimizer_step={optimizer_step_count}"
            )
            finalize_epoch(
                config=config,
                paths=setup.paths,
                model=model,
                nextlat_dynamics_model=nextlat_dynamics_model,
                ema_model=ema_model,
                wandb_module=wandb_module,
                epoch=epoch,
                train_metrics=train_metrics,
                global_step_offset=global_step_offset,
                tokenizer=setup.tokenizer,
            )
            break

        finalize_epoch(
            config=config,
            paths=setup.paths,
            model=model,
            nextlat_dynamics_model=nextlat_dynamics_model,
            ema_model=ema_model,
            wandb_module=wandb_module,
            epoch=epoch,
            train_metrics=train_metrics,
            global_step_offset=global_step_offset,
            tokenizer=setup.tokenizer,
        )


def run_training(config: TrainConfig) -> None:
    setup = prepare_training_setup(config)
    model_config = build_model_config(setup.config, setup.tokenizer)
    model = build_model(setup.config.model_type, model_config).to(setup.device)
    nextlat_dynamics_model = build_nextlat_dynamics_model(setup.config, model)
    write_model_config(model_config, setup.paths)
    trainable_modules = build_trainable_modules(model, nextlat_dynamics_model)
    dataset = setup.data_artifacts.dataset
    total_training_steps = resolve_scheduled_total_training_steps(setup.config, len(dataset))
    num_warmup_steps = resolve_num_warmup_steps(setup.config, total_training_steps)
    optimizer, scheduler = build_optimizer_and_scheduler(setup.config, trainable_modules, total_training_steps)
    optimizer.zero_grad(set_to_none=True)

    if setup.config.init_from_checkpoint is not None:
        loaded_paths = load_warm_start_state_dicts(
            model,
            Path(setup.config.init_from_checkpoint),
            use_ema=setup.config.init_from_use_ema,
            nextlat_dynamics_model=nextlat_dynamics_model,
            map_location=setup.device,
        )
        logger.info(
            "Warm-started training module state from: "
            + ", ".join(str(path) for path in loaded_paths)
        )

    # Constructed after any warm-start so the EMA shadow copies the
    # warm-started weights rather than the fresh initialization.
    ema_model = None
    if setup.config.ema_decay is not None:
        ema_device = resolve_ema_device(setup.config.ema_device, setup.device)
        ema_model = EmaModel(model, setup.config.ema_decay, ema_device)

    wandb_module = setup_wandb(setup.config)
    logger.info(f"Loaded {setup.config.model_type} backend on {setup.device}")
    logger.info(f"Optimizer: {setup.config.optimizer}")
    autocast_dtype = resolve_cuda_autocast_dtype(setup.config.cuda_autocast)
    if autocast_dtype is None:
        logger.info("CUDA autocast: disabled")
    else:
        logger.info(f"CUDA autocast: {autocast_dtype}")
    if nextlat_dynamics_model is not None:
        logger.info(
            "Enabled NextLat objective: "
            f"horizon={setup.config.nextlat_horizon} "
            f"lambda_mse={setup.config.nextlat_lambda_mse} "
            f"lambda_kl={setup.config.nextlat_lambda_kl} "
            f"lambda_ce={setup.config.nextlat_lambda_ce}"
        )
    if setup.config.gradient_accumulation_steps > 1:
        effective_batch_size = setup.config.batch_size * setup.config.gradient_accumulation_steps
        logger.info(
            "Enabled gradient accumulation: "
            f"micro_batch_size={setup.config.batch_size} "
            f"accumulation_steps={setup.config.gradient_accumulation_steps} "
            f"effective_batch_size={effective_batch_size}"
        )
    if ema_model is not None:
        logger.info(
            "Enabled EMA of model weights: "
            f"decay={setup.config.ema_decay} device={ema_model.device}"
        )
    log_model_structure(model)
    logger.info(f"Derived total training steps: {total_training_steps}")
    logger.info(f"Derived warmup steps: {num_warmup_steps} ({setup.config.warmup_ratio:.2%})")

    full_train_loop(
        setup=setup,
        model=model,
        nextlat_dynamics_model=nextlat_dynamics_model,
        optimizer=optimizer,
        scheduler=scheduler,
        wandb_module=wandb_module,
        dataset=dataset,
        ema_model=ema_model,
        total_training_steps=total_training_steps,
    )
