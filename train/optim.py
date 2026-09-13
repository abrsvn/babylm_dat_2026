"""Training optimizer and scheduler helpers."""

import math

import torch
from transformers import optimization as hf_optimization
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

from models.attention.transformer.components import RMSNorm
from train.config import TrainConfig


def _adam_moment_update(
    grad: torch.Tensor,
    state: dict,
    betas: tuple[float, float],
    eps: float,
) -> torch.Tensor:
    """Compute the bias-corrected Adam update.

    Updates ``state`` in-place with ``step``, ``exp_avg``, ``exp_avg_sq``.
    Returns the Adam update tensor ``m_hat / (sqrt(v_hat) + eps)``.
    """
    if len(state) == 0:
        state["step"] = 0
        state["exp_avg"] = torch.zeros_like(grad)
        state["exp_avg_sq"] = torch.zeros_like(grad)
    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
    beta1, beta2 = betas
    state["step"] += 1
    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
    bias_correction1 = 1 - beta1 ** state["step"]
    bias_correction2 = 1 - beta2 ** state["step"]
    m_t = exp_avg / bias_correction1
    v_t = exp_avg_sq / bias_correction2
    torch.sqrt_(v_t)
    return m_t / (v_t + eps)


def _lambw_trust_ratio(
    p: torch.Tensor,
    adam_update: torch.Tensor,
) -> torch.Tensor:
    """Compute the LAMB trust ratio: ``||w|| / ||adam_update||``.

    Returns 1.0 where either norm is 0. Tensor-valued to avoid device
    synchronizations from ``.item()`` on every parameter.
    """
    w_norm = torch.norm(p.flatten())
    g_norm = torch.norm(adam_update.flatten())
    valid_norms = (w_norm > 0.0) & (g_norm > 0.0)
    safe_g_norm = torch.where(valid_norms, g_norm, torch.ones_like(g_norm))
    return torch.where(valid_norms, w_norm / safe_g_norm, torch.ones_like(w_norm))


def _newtonschulz_compute_dtype(grad: torch.Tensor) -> torch.dtype:
    """BF16 for supported CUDA devices, FP32 for CPU or unsupported accelerators."""
    if grad.is_cuda and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def _zeropower_via_newtonschulz5(
    G: torch.Tensor,
    steps: int,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    """Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.

    Adapted from Keller Jordan's Muon implementation
    (https://github.com/KellerJordan/Muon), MIT licensed::

        Copyright (c) 2024 Keller Jordan

        Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
        in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be included in all
        copies or substantial portions of the Software.

    Uses a quintic iteration whose coefficients are selected to maximize the
    slope at zero. This does not produce exact ``UV^T`` but rather something
    like ``US'V^T`` where ``S'`` is diagonal with ``S_{ii}' ~ Uniform(0.5, 1.5)``.
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(compute_dtype)
    if G.size(-2) > G.size(-1):
        X = X.mT
    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X.to(G.dtype)


def _muon_update(
    grad: torch.Tensor,
    momentum_buffer: torch.Tensor,
    beta: float,
    ns_steps: int,
) -> torch.Tensor:
    """Compute Muon update via Nesterov momentum + Newton-Schulz orthogonalization.

    Updates ``momentum_buffer`` in-place. Does not mutate ``grad``.
    """
    # Update momentum buffer in-place (Nesterov momentum)
    momentum_buffer.lerp_(grad, 1 - beta)
    # Out-of-place interpolation to avoid mutating grad
    update = grad.lerp(momentum_buffer, beta)
    # Collapse conv filters to 2D if needed
    if update.ndim == 4:
        update = update.view(len(update), -1)
    compute_dtype = _newtonschulz_compute_dtype(grad)
    update = _zeropower_via_newtonschulz5(update, ns_steps, compute_dtype)
    # Aspect-ratio scaling: scale up tall matrices
    update *= max(1, update.size(-2) / update.size(-1)) ** 0.5
    return update


class LambW(torch.optim.Optimizer):
    """Layer-wise adaptive moments optimizer with decoupled weight decay.

    Standard Adam first/second moments with bias correction. The Adam update
    is normalized by the parameter's own norm (the trust ratio), and weight
    decay is applied decoupled (AdamW-style)::

        adam_update = m_hat / (sqrt(v_hat) + eps)
        ratio = ||w|| / ||adam_update||   (1.0 if either norm is 0)
        w *= (1 - lr * weight_decay)
        w -= lr * ratio * adam_update

    The trust ratio is always computed, so LambW with ``weight_decay=0`` still
    applies layer-wise adaptation. Canonical LAMB couples weight decay into
    the update before computing the norm; this implementation decouples them
    so weight decay and the trust ratio are independently controllable.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError(
                        "LambW does not support sparse gradients, "
                        "consider SparseAdam instead."
                    )

                state = self.state[p]
                adam_update = _adam_moment_update(
                    grad, state, group["betas"], group["eps"]
                )
                ratio = _lambw_trust_ratio(p, adam_update)

                # Decoupled weight decay (AdamW-style), applied before the
                # trust-ratio-scaled Adam update.
                if group["weight_decay"] > 0:
                    p.mul_(1 - group["lr"] * group["weight_decay"])

                p.add_(adam_update * ratio, alpha=-group["lr"])

        return loss


class MuonWithAux(torch.optim.Optimizer):
    """Muon optimizer with auxiliary AdamW/LambW for non-hidden parameters.

    Muon applies Newton-Schulz orthogonalization to hidden weight matrices.
    Other parameters (embeddings, heads, norms, biases) use AdamW or LambW.

    Each parameter group must specify ``update_type``:
    - ``"muon"``: Newton-Schulz orthogonalization + decoupled weight decay
    - ``"adamw"``: standard AdamW + decoupled weight decay
    - ``"lambw"``: AdamW + LAMB trust ratio + decoupled weight decay
    """

    def __init__(self, param_groups):
        for group in param_groups:
            self._validate_group(group)
        super().__init__(param_groups, {})

    @staticmethod
    def _validate_group(group: dict) -> None:
        if "update_type" not in group:
            raise ValueError("Each param group must specify update_type")
        ut = group["update_type"]
        if ut not in {"muon", "lambw"}:
            raise ValueError(f"Invalid update_type: {ut}")

        lr = group["lr"]
        if not isinstance(lr, (int, float)) or not math.isfinite(lr) or lr <= 0:
            raise ValueError(f"lr must be positive and finite, got {lr}")

        wd = group["weight_decay"]
        if not isinstance(wd, (int, float)) or not math.isfinite(wd) or wd < 0:
            raise ValueError(f"weight_decay must be non-negative and finite, got {wd}")

        params = group["params"]
        if not isinstance(params, (list, tuple)) or len(params) == 0:
            raise ValueError("params must be a non-empty list or tuple")

        if ut == "muon":
            mom = group["momentum"]
            if not isinstance(mom, (int, float)) or not 0.0 < mom < 1.0:
                raise ValueError(f"momentum must be in (0.0, 1.0), got {mom}")
            ns = group["n_schulz_steps"]
            if not isinstance(ns, int) or isinstance(ns, bool) or ns <= 0:
                raise ValueError(f"n_schulz_steps must be a positive integer, got {ns}")
            for p in params:
                if p.ndim < 2:
                    raise ValueError(
                        f"Muon parameters must have ndim >= 2, got shape {tuple(p.shape)}"
                    )
        else:
            betas = group["betas"]
            if not (isinstance(betas, tuple) and len(betas) == 2):
                raise ValueError(f"betas must be a 2-tuple, got {betas}")
            b1, b2 = betas
            if not isinstance(b1, (int, float)) or not 0.0 < b1 < 1.0:
                raise ValueError(f"beta1 must be in (0.0, 1.0), got {b1}")
            if not isinstance(b2, (int, float)) or not 0.0 < b2 < 1.0:
                raise ValueError(f"beta2 must be in (0.0, 1.0), got {b2}")
            eps = group["eps"]
            if not isinstance(eps, (int, float)) or not math.isfinite(eps) or eps <= 0:
                raise ValueError(f"eps must be positive and finite, got {eps}")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            update_type = group["update_type"]
            if update_type == "muon":
                self._step_muon(group)
            elif update_type == "lambw":
                self._step_lambw(group)
            else:
                raise ValueError(f"Invalid update_type: {update_type}")

        return loss

    def _step_muon(self, group):
        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if len(state) == 0:
                state["momentum_buffer"] = torch.zeros_like(p)
            update = _muon_update(
                grad,
                state["momentum_buffer"],
                group["momentum"],
                group["n_schulz_steps"],
            )
            if group["weight_decay"] > 0:
                p.mul_(1 - group["lr"] * group["weight_decay"])
            p.add_(update.reshape(p.shape), alpha=-group["lr"])

    def _step_lambw(self, group):
        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            adam_update = _adam_moment_update(
                grad, state, group["betas"], group["eps"]
            )
            ratio = _lambw_trust_ratio(p, adam_update)
            if group["weight_decay"] > 0:
                p.mul_(1 - group["lr"] * group["weight_decay"])
            p.add_(adam_update * ratio, alpha=-group["lr"])


def get_parameter_names(model: torch.nn.Module, forbidden_layer_types) -> list[str]:
    result: list[str] = []
    for name, child in model.named_children():
        result += [
            f"{name}.{child_name}"
            for child_name in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    result += list(model._parameters.keys())
    return result


def _is_muon_hidden_weight(param_name: str, param: torch.nn.Parameter) -> bool:
    """Check if a parameter is a Muon-eligible hidden weight matrix.

    Muon optimizes hidden 2D+ weight matrices via Newton-Schulz
    orthogonalization. Eligible parameters:
    - ``layers.*`` weights with ``ndim >= 2`` (includes batched 3D ``wr_proj``)
    - ``symbol_retriever.*`` projection weights (``q_proj``, ``k_proj``, ``rel_to_hidden``)
    - NextLat dynamics MLP linear weights (``.mlp.*.weight``)

    Not eligible (routed to AdamW/LambW auxiliary groups):
    - ``token_embeddings`` (input embedding)
    - ``position_encoder`` weights (positional encoding)
    - ``template_features``, ``symbol_library`` (symbol tables, embedding-like)
    - ``lm_head`` (output layer)
    - norms, biases (1D)
    """
    if param.ndim < 2:
        return False
    # layers.* hidden weights (includes 3D wr_proj)
    if param_name.startswith("layers."):
        return True
    # symbol_retriever.* projection weights only
    if param_name.startswith("symbol_retriever."):
        # Projection weights: q_proj.weight, k_proj.weight, rel_to_hidden.weight
        # Exclude: template_features, symbol_library, position_encoder.*
        if any(
            proj in param_name
            for proj in ["q_proj.weight", "k_proj.weight", "rel_to_hidden.weight"]
        ):
            return True
        return False
    # NextLat dynamics MLP weights
    if param_name.startswith("mlp.") and param_name.endswith(".weight"):
        return True
    return False


def _classify_muon_parameters(
    trainable_modules: tuple[torch.nn.Module, ...],
    forbidden_layer_types: tuple[type, ...],
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """Partition trainable parameters into Muon, aux-decay, aux-no-decay groups.

    Returns ``(muon_params, aux_decay_params, aux_no_decay_params)``.
    Every trainable parameter appears in exactly one list.
    """
    muon_params: list[torch.nn.Parameter] = []
    aux_decay_params: list[torch.nn.Parameter] = []
    aux_no_decay_params: list[torch.nn.Parameter] = []

    for module in trainable_modules:
        decay_param_names = get_parameter_names(module, forbidden_layer_types)
        decay_param_names = {
            name for name in decay_param_names if "bias" not in name
        }
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue
            if _is_muon_hidden_weight(name, param):
                muon_params.append(param)
            elif name in decay_param_names:
                aux_decay_params.append(param)
            else:
                aux_no_decay_params.append(param)

    return muon_params, aux_decay_params, aux_no_decay_params


def _build_muon_optimizer(
    config: TrainConfig,
    trainable_modules: tuple[torch.nn.Module, ...],
    forbidden_layer_types: tuple[type, ...],
) -> MuonWithAux:
    muon_params, aux_decay_params, aux_no_decay_params = _classify_muon_parameters(
        trainable_modules, forbidden_layer_types
    )

    if not muon_params:
        raise ValueError(
            "Muon optimizer found no hidden weight parameters. "
            "Ensure the model has layers.* or symbol_retriever.* "
            "projection weights."
        )

    aux_update_type = config.muon_aux_optimizer
    aux_betas = (config.optimizer_beta1, config.optimizer_beta2)
    aux_eps = config.optimizer_eps

    param_groups = [
        {
            "params": muon_params,
            "lr": config.muon_lr,
            "weight_decay": config.weight_decay,
            "update_type": "muon",
            "momentum": config.muon_momentum,
            "n_schulz_steps": config.muon_n_schulz_steps,
        },
        {
            "params": aux_decay_params,
            "lr": config.learning_rate,
            "weight_decay": config.weight_decay,
            "update_type": aux_update_type,
            "betas": aux_betas,
            "eps": aux_eps,
        },
        {
            "params": aux_no_decay_params,
            "lr": config.learning_rate,
            "weight_decay": 0.0,
            "update_type": aux_update_type,
            "betas": aux_betas,
            "eps": aux_eps,
        },
    ]

    return MuonWithAux(param_groups)


def build_optimizer(
    config: TrainConfig,
    trainable_modules: tuple[torch.nn.Module, ...],
) -> torch.optim.Optimizer:
    forbidden_layer_types = tuple(ALL_LAYERNORM_LAYERS) + (RMSNorm,)

    if config.optimizer == "muon":
        return _build_muon_optimizer(
            config, trainable_modules, forbidden_layer_types
        )

    decay_parameter_ids = set()
    trainable_parameters: list[torch.nn.Parameter] = []
    for module in trainable_modules:
        decay_parameters = get_parameter_names(module, forbidden_layer_types)
        decay_parameters = {name for name in decay_parameters if "bias" not in name}
        for name, parameter in module.named_parameters():
            if not parameter.requires_grad:
                continue
            trainable_parameters.append(parameter)
            if name in decay_parameters:
                decay_parameter_ids.add(id(parameter))

    optimizer_grouped_parameters = [
        {
            "params": [
                parameter
                for parameter in trainable_parameters
                if id(parameter) in decay_parameter_ids
            ],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [
                parameter
                for parameter in trainable_parameters
                if id(parameter) not in decay_parameter_ids
            ],
            "weight_decay": 0.0,
        },
    ]

    return torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=config.learning_rate,
        eps=config.optimizer_eps,
        betas=(config.optimizer_beta1, config.optimizer_beta2),
    )


def resolve_total_training_steps(config: TrainConfig, num_batches_per_epoch: int) -> int:
    if num_batches_per_epoch <= 0:
        raise ValueError(f"num_batches_per_epoch must be positive, got {num_batches_per_epoch}")
    optimizer_steps_per_epoch = (
        num_batches_per_epoch + config.gradient_accumulation_steps - 1
    ) // config.gradient_accumulation_steps
    return optimizer_steps_per_epoch * config.n_epochs


def resolve_scheduled_total_training_steps(
    config: TrainConfig,
    num_chunks: int,
) -> int:
    """Total optimizer steps across all epochs.

    Each epoch contributes ceil(num_batches / gradient_accumulation_steps)
    optimizer steps, where num_batches = ceil(num_chunks / batch_size)
    and the dataloader keeps its final partial batch (drop_last=False).
    """
    if config.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {config.batch_size}")
    if num_chunks <= 0:
        raise ValueError(f"num_chunks must be positive, got {num_chunks}")
    num_batches = (num_chunks + config.batch_size - 1) // config.batch_size
    # max_train_steps is a per-epoch cap on batches (matching the runner's
    # per-epoch clamp). Clamp before dividing by gradient_accumulation_steps
    # so the scheduler's optimizer-step count matches the runner's batch count.
    if config.max_train_steps is not None:
        num_batches = min(num_batches, config.max_train_steps)
    optimizer_steps = (
        num_batches + config.gradient_accumulation_steps - 1
    ) // config.gradient_accumulation_steps
    total_optimizer_steps = optimizer_steps * config.n_epochs
    if config.max_optimizer_steps is not None:
        total_optimizer_steps = min(total_optimizer_steps, config.max_optimizer_steps)
    return total_optimizer_steps


def resolve_num_warmup_steps(config: TrainConfig, total_training_steps: int) -> int:
    if total_training_steps <= 0:
        raise ValueError(f"total_training_steps must be positive, got {total_training_steps}")
    return int(total_training_steps * config.warmup_ratio + 0.5)


def build_scheduler(
    config: TrainConfig,
    optimizer: torch.optim.Optimizer,
    total_training_steps: int,
):
    num_warmup_steps = resolve_num_warmup_steps(config, total_training_steps)
    scheduler_type = config.lr_scheduler_type
    if scheduler_type == "cosine_min_lr":
        return hf_optimization.get_cosine_with_min_lr_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_training_steps,
            min_lr_rate=config.min_lr_rate,
        )
    raise ValueError(f"Unsupported lr_scheduler_type: {scheduler_type}")


def build_optimizer_and_scheduler(
    config: TrainConfig,
    trainable_modules: tuple[torch.nn.Module, ...],
    total_training_steps: int,
) -> tuple[torch.optim.Optimizer, object]:
    optimizer = build_optimizer(config, trainable_modules)
    scheduler = build_scheduler(config, optimizer, total_training_steps)
    return optimizer, scheduler
