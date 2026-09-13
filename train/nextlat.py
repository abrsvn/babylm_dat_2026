"""Next-latent training objective for Relational BabyLM decoder LMs."""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class NextLatLossConfig:
    lambda_mse: float
    lambda_kl: float
    lambda_ce: float
    horizon: int
    eos_token_id: int
    sequence_boundary_policy: str


@dataclass(frozen=True)
class NextLatLossResult:
    loss: torch.Tensor
    logits: torch.Tensor
    next_token_loss: torch.Tensor
    hidden_loss: torch.Tensor
    kl_loss: torch.Tensor
    auxiliary_token_loss: torch.Tensor


class NextLatDynamicsModel(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        dropout: float,
        proj_factor: float,
        use_bias: bool,
    ):
        super().__init__()
        input_dim = hidden_dim * 2
        hidden_mlp_dim = resolve_dynamics_hidden_dim(input_dim, proj_factor)
        self.hidden_state_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.norm = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_mlp_dim, bias=use_bias),
            nn.GELU(),
            nn.Linear(hidden_mlp_dim, hidden_mlp_dim, bias=use_bias),
            nn.GELU(),
            nn.Linear(hidden_mlp_dim, hidden_dim, bias=use_bias),
        )
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        current_states: torch.Tensor,
        next_token_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.hidden_state_dropout(current_states)
        features = torch.cat([next_token_embeddings, hidden_states], dim=-1)
        delta = self.mlp(self.norm(features))
        return delta + current_states


def resolve_dynamics_hidden_dim(input_dim: int, proj_factor: float) -> int:
    if input_dim <= 0:
        raise ValueError(f"input_dim must be positive, got {input_dim}")
    if proj_factor <= 0.0:
        raise ValueError(f"proj_factor must be positive, got {proj_factor}")
    return 128 * max(1, int((input_dim * proj_factor / 128) + 0.5))


def compute_nextlat_loss(
    model: nn.Module,
    dynamics_model: NextLatDynamicsModel,
    input_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    target_mask: torch.Tensor,
    config: NextLatLossConfig,
) -> NextLatLossResult:
    """Compute causal left-to-right NextLat loss for CLM batches.

    WARNING: tie_lm_head must be False when NextLat is enabled.
    The NextLat objective requires an UNTIED output head:
    token_embedding and lm_head are separate tensors, initialized
    independently.  If the head is tied
    (lm_head.weight = token_embeddings.weight), the NTP loss
    (CE(lm_head(hidden_states), targets)) backprops DIRECTLY into
    the input word embeddings via the output projection path,
    coupling input and output representations and interfering with
    the NextLat belief-state shaping objective.  The NextLat KL loss
    uses lm_head.weight.detach() which blocks the NextLat gradient
    from reaching the shared weight, but the NTP loss does NOT
    detach, so tying still causes direct word-embedding updates from
    the NTP loss.  build_model_config derives tie_lm_head=False
    whenever nextlat_enabled=True (see train/runner.py).
    """
    if input_tokens.shape != target_tokens.shape:
        raise ValueError(
            "input_tokens and target_tokens must have the same shape, "
            f"got {tuple(input_tokens.shape)} and {tuple(target_tokens.shape)}"
        )
    if target_mask.shape != target_tokens.shape:
        raise ValueError(
            "target_mask and target_tokens must have the same shape, "
            f"got {tuple(target_mask.shape)} and {tuple(target_tokens.shape)}"
        )

    full_tokens = torch.cat([input_tokens, target_tokens[:, -1:]], dim=1)
    # target_mask covers x_1..x_T; its first column is forced valid and supplies the BOS mask slot.
    full_valid_mask = torch.cat([target_mask[:, :1], target_mask], dim=1)
    # The lm_head is exercised through the encode forward (return_logits=True)
    # so its gradients flow also when tie_lm_head=False and lm_head is a
    # distinct parameter.
    input_token_embeddings, hidden_states, logits = model(
        full_tokens, attention_mask=full_valid_mask, mode="encode", return_logits=True
    )
    logits = logits[:, :-1]
    target_tokens_for_loss = target_tokens.masked_fill(~target_mask, -1)
    next_token_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        target_tokens_for_loss.reshape(-1),
        ignore_index=-1,
    )

    total_hidden_loss = hidden_states.new_zeros(())
    total_kl_loss = hidden_states.new_zeros(())
    total_auxiliary_token_loss = hidden_states.new_zeros(())
    pred_states = hidden_states

    for horizon_index in range(config.horizon):
        shift = horizon_index + 1
        if shift >= target_tokens.shape[1]:
            raise ValueError(
                f"nextlat_horizon={config.horizon} is too large for sequence length "
                f"{target_tokens.shape[1]}"
            )

        pred_states = dynamics_model(
            pred_states[:, :-1],
            input_token_embeddings[:, shift:],
        )
        target_states = hidden_states[:, shift:]
        hidden_prediction_mask = build_nextlat_prediction_mask(
            input_tokens=full_tokens,
            valid_token_mask=full_valid_mask,
            eos_token_id=config.eos_token_id,
            shift=shift,
            sequence_boundary_policy=config.sequence_boundary_policy,
        )
        token_prediction_mask = hidden_prediction_mask[:, :-1] & target_mask[:, shift:]
        shifted_targets = target_tokens[:, shift:]

        # NextLat KL/CE losses use a DETACHED lm_head so the NextLat losses
        # do not update the output head (matching the original NextLat repo:
        # ``lm_head_weight = self.model.lm_head.weight.detach()``).
        #
        # IMPORTANT: This detach does NOT solve the tied-head problem.
        # When tie_lm_head=True, lm_head.weight IS token_embeddings.weight,
        # so the NTP loss (CE(lm_head(hidden_states), targets)) backprops
        # DIRECTLY into the input word embeddings via the output projection.
        # The original NextLat paper/repo uses an UNTIED head for this reason.
        # build_model_config derives tie_lm_head=False whenever
        # nextlat_enabled=True (see train/runner.py).
        total_hidden_loss = total_hidden_loss + smooth_l1_with_mask(
            pred_states,
            target_states.detach(),
            hidden_prediction_mask,
        )
        student_logits = F.linear(pred_states[:, :-1], model.lm_head.weight.detach())
        teacher_logits = F.linear(target_states[:, :-1].detach(), model.lm_head.weight.detach())
        total_kl_loss = total_kl_loss + categorical_kl_with_mask(
            teacher_logits=teacher_logits,
            student_logits=student_logits,
            mask=token_prediction_mask,
        )
        total_auxiliary_token_loss = total_auxiliary_token_loss + cross_entropy_with_mask(
            student_logits,
            shifted_targets,
            token_prediction_mask,
        )
    hidden_loss = total_hidden_loss / config.horizon
    kl_loss = total_kl_loss / config.horizon
    auxiliary_token_loss = total_auxiliary_token_loss / config.horizon
    nextlat_loss = (
        config.lambda_mse * hidden_loss
        + config.lambda_kl * kl_loss
        + config.lambda_ce * auxiliary_token_loss
    )
    return NextLatLossResult(
        loss=next_token_loss + nextlat_loss,
        logits=logits,
        next_token_loss=next_token_loss,
        hidden_loss=hidden_loss,
        kl_loss=kl_loss,
        auxiliary_token_loss=auxiliary_token_loss,
    )


def build_nextlat_prediction_mask(
    input_tokens: torch.Tensor,
    valid_token_mask: torch.Tensor,
    eos_token_id: int,
    shift: int,
    sequence_boundary_policy: str,
) -> torch.Tensor:
    if shift <= 0:
        raise ValueError(f"shift must be positive, got {shift}")
    if input_tokens.shape != valid_token_mask.shape:
        raise ValueError(
            "input_tokens and valid_token_mask must have the same shape, "
            f"got {tuple(input_tokens.shape)} and {tuple(valid_token_mask.shape)}"
        )
    if shift >= input_tokens.shape[1]:
        raise ValueError(
            f"shift={shift} is too large for sequence length {input_tokens.shape[1]}"
        )

    mask_width = input_tokens.shape[1] - shift
    prediction_mask = valid_token_mask[:, :mask_width].clone()
    for offset in range(1, shift + 1):
        offset_mask = valid_token_mask[:, offset : offset + mask_width]
        prediction_mask = prediction_mask & offset_mask
        if sequence_boundary_policy in {"eos_document", "document_boundary"}:
            prediction_mask = prediction_mask & (input_tokens[:, offset : offset + mask_width] != eos_token_id)
        else:
            raise ValueError(f"Unsupported sequence_boundary_policy: {sequence_boundary_policy}")
    return prediction_mask


def smooth_l1_with_mask(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    element_loss = F.smooth_l1_loss(prediction, target, reduction="none")
    weights = mask.to(dtype=element_loss.dtype).unsqueeze(-1)
    denominator = weights.expand_as(element_loss).sum().clamp_min(1.0)
    return (element_loss * weights).sum() / denominator


def categorical_kl_with_mask(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if teacher_logits.shape != student_logits.shape:
        raise ValueError(
            "teacher_logits and student_logits must have the same shape, "
            f"got {tuple(teacher_logits.shape)} and {tuple(student_logits.shape)}"
        )
    if mask.shape != teacher_logits.shape[:2]:
        raise ValueError(
            "mask must match the first two logit dimensions, "
            f"got {tuple(mask.shape)} and {tuple(teacher_logits.shape[:2])}"
        )

    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    token_kl = F.kl_div(
        student_log_probs,
        teacher_log_probs,
        log_target=True,
        reduction="none",
    ).sum(dim=-1)
    weights = mask.to(dtype=token_kl.dtype)
    return (token_kl * weights).sum() / weights.sum().clamp_min(1.0)


def cross_entropy_with_mask(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if targets.shape != logits.shape[:2]:
        raise ValueError(
            "targets must match the first two logit dimensions, "
            f"got {tuple(targets.shape)} and {tuple(logits.shape[:2])}"
        )
    if mask.shape != targets.shape:
        raise ValueError(
            "mask and targets must have the same shape, "
            f"got {tuple(mask.shape)} and {tuple(targets.shape)}"
        )
    if not mask.any():
        return logits.sum() * 0.0

    masked_targets = targets.masked_fill(~mask, -1)
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        masked_targets.reshape(-1),
        ignore_index=-1,
    )
