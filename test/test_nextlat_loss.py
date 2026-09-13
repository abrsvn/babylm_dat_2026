"""Pytest coverage for the NextLat training objective."""

import pytest
import torch
import torch.nn.functional as F

from models.attention.dat.config import DatLMConfig
from models.attention.dat.lm import DatDecoderLM
from models.attention.transformer.config import SelfAttentionLMConfig
from models.attention.transformer.lm import SelfAttentionDecoderLM
from models.factory import build_model
from train.checkpoints import ensure_experiment_dirs, write_experiment_config, write_model_config
from train.config import TrainConfig, parse_train_config, train_config_to_dict
from train.data import load_train_data
from train.nextlat import (
    NextLatDynamicsModel,
    NextLatLossConfig,
    build_nextlat_prediction_mask,
    categorical_kl_with_mask,
    compute_nextlat_loss,
    cross_entropy_with_mask,
    resolve_dynamics_hidden_dim,
    smooth_l1_with_mask,
)
from train.optim import build_optimizer_and_scheduler, resolve_total_training_steps
from train.paths import resolve_train_paths
from train.runner import build_model_config, build_nextlat_dynamics_model, train_epoch
from conftest import build_tiny_train_config


def build_tiny_self_attention_lm() -> SelfAttentionDecoderLM:
    config = SelfAttentionLMConfig(
        vocab_size=32,
        max_seq_len=8,
        hidden_dim=32,
        n_heads=4,
        n_layers=1,
        dropout=0.0,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    return SelfAttentionDecoderLM(config)


def build_tiny_dat_lm() -> DatDecoderLM:
    config = DatLMConfig(
        vocab_size=32,
        max_seq_len=8,
        hidden_dim=32,
        n_heads_sa=2,
        n_heads_ra=2,
        n_layers=1,
        dropout=0.0,
        symbol_retrieval="relative",
        ra_type="rca",
        ra_rel_activation="sigmoid",
        ffn_activation="swiglu",
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    return DatDecoderLM(config)


class RecordingSelfAttentionDecoderLM(SelfAttentionDecoderLM):
    def __init__(self, config: SelfAttentionLMConfig):
        super().__init__(config)
        self.encoded_input_shapes: list[tuple[int, ...]] = []
        self.encoded_attention_mask_shapes: list[tuple[int, ...] | None] = []

    def encode_for_objective(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        bidirectional: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.encoded_input_shapes.append(tuple(input_ids.shape))
        if attention_mask is None:
            self.encoded_attention_mask_shapes.append(None)
        else:
            self.encoded_attention_mask_shapes.append(tuple(attention_mask.shape))
        return super().encode_for_objective(
            input_ids, attention_mask=attention_mask, bidirectional=bidirectional
        )


def build_nextlat_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_tokens = torch.tensor(
        [
            [1, 3, 4, 5, 2, 0],
            [1, 6, 7, 2, 0, 0],
        ],
        dtype=torch.long,
    )
    target_tokens = torch.tensor(
        [
            [3, 4, 5, 2, 0, 0],
            [6, 7, 2, 0, 0, 0],
        ],
        dtype=torch.long,
    )
    target_mask = target_tokens != 0
    target_mask[:, 0] = True
    return input_tokens, target_tokens, target_mask


def test_train_config_defaults_keep_nextlat_disabled() -> None:
    config = TrainConfig()

    assert not config.nextlat_enabled
    assert config.nextlat_lambda_mse == pytest.approx(1.0)
    assert config.nextlat_lambda_kl == pytest.approx(1.0)
    assert config.nextlat_lambda_ce == pytest.approx(0.0)
    assert config.nextlat_horizon == 1
    assert config.nextlat_proj_factor == pytest.approx(1.0)
    assert config.sequence_boundary_policy == "eos_document"


def test_train_config_validates_nextlat_settings() -> None:
    config = TrainConfig(nextlat_enabled=True, nextlat_lambda_kl=0.5)

    assert config.nextlat_enabled
    assert config.nextlat_lambda_kl == pytest.approx(0.5)

    with pytest.raises(ValueError, match="nextlat_lambda_mse must be non-negative"):
        TrainConfig(nextlat_lambda_mse=-0.1)

    with pytest.raises(ValueError, match="nextlat_horizon must be positive"):
        TrainConfig(nextlat_horizon=0)

    with pytest.raises(ValueError, match="nextlat_horizon must be positive"):
        TrainConfig(nextlat_horizon=-1)

    with pytest.raises(ValueError, match="smaller than the model training sequence length"):
        TrainConfig(nextlat_horizon=1281, datapoint_length=1280)

    with pytest.raises(ValueError, match="nextlat_proj_factor must be positive"):
        TrainConfig(nextlat_proj_factor=0.0)

    with pytest.raises(ValueError, match="At least one NextLat auxiliary coefficient"):
        TrainConfig(
            nextlat_enabled=True,
            nextlat_lambda_mse=0.0,
            nextlat_lambda_kl=0.0,
            nextlat_lambda_ce=0.0,
        )


def test_parse_train_config_accepts_nextlat_flags() -> None:
    config = parse_train_config(
        [
            "--tokenizer_dir", "test/fixtures/tiny_tokenizer",
            "--train_data_dir", "test/fixtures/train_assets/clean_train_tiny",
            "--nextlat_enabled",
            "--nextlat_lambda_mse",
            "0.75",
            "--nextlat_lambda_kl",
            "0.5",
            "--nextlat_lambda_ce",
            "0.25",
            "--nextlat_horizon",
            "2",
            "--nextlat_proj_factor",
            "1.25",
        ]
    )

    assert config.nextlat_enabled
    assert config.nextlat_lambda_mse == pytest.approx(0.75)
    assert config.nextlat_lambda_kl == pytest.approx(0.5)
    assert config.nextlat_lambda_ce == pytest.approx(0.25)
    assert config.nextlat_horizon == 2
    assert config.nextlat_proj_factor == pytest.approx(1.25)


def test_parse_train_config_rejects_unsupported_nextlat_low_memory_flag() -> None:
    with pytest.raises(SystemExit):
        parse_train_config([
            "--tokenizer_dir", "test/fixtures/tiny_tokenizer",
            "--train_data_dir", "test/fixtures/train_assets/clean_train_tiny",
            "--nextlat_low_memory_loss"])


def test_train_config_serialization_omits_unsupported_nextlat_low_memory_flag() -> None:
    serialized = train_config_to_dict(TrainConfig(nextlat_enabled=True, nextlat_lambda_kl=0.5))

    assert "nextlat_low_memory_loss" not in serialized


def test_build_model_config_extends_context_for_nextlat_full_block(tmp_path) -> None:
    ntp_config = build_tiny_train_config(
        tmp_path,
        "ntp-context-length",
        model_type="self_attention",
        datapoint_length=8,
    )
    paths = resolve_train_paths(ntp_config)
    data_artifacts = load_train_data(ntp_config, paths)

    ntp_model_config = build_model_config(ntp_config, data_artifacts.tokenizer)

    assert ntp_model_config.max_seq_len == ntp_config.datapoint_length + 1

    nextlat_config = build_tiny_train_config(
        tmp_path,
        "nextlat-context-length",
        model_type="self_attention",
        datapoint_length=8,
        nextlat_enabled=True,
    )

    nextlat_model_config = build_model_config(nextlat_config, data_artifacts.tokenizer)

    assert nextlat_model_config.max_seq_len == nextlat_config.datapoint_length + 2


def test_nextlat_dynamics_model_returns_residual_state_shape() -> None:
    torch.manual_seed(0)
    dynamics = NextLatDynamicsModel(
        hidden_dim=32,
        dropout=0.0,
        proj_factor=1.0,
        use_bias=False,
    )
    current_states = torch.randn(2, 4, 32)
    next_token_embeddings = torch.randn(2, 4, 32)

    next_states = dynamics(current_states, next_token_embeddings)

    assert resolve_dynamics_hidden_dim(input_dim=64, proj_factor=1.0) == 128
    assert next_states.shape == current_states.shape
    assert not torch.equal(next_states, current_states)


def test_nextlat_categorical_kl_matches_full_torch_reference() -> None:
    torch.manual_seed(0)
    teacher_logits = torch.randn(2, 73, 11)
    student_logits = torch.randn(2, 73, 11)
    mask = torch.zeros(2, 73, dtype=torch.bool)
    mask[0, :70] = True
    mask[1, 3:67] = True

    actual = categorical_kl_with_mask(
        teacher_logits=teacher_logits,
        student_logits=student_logits,
        mask=mask,
    )
    token_kl = F.kl_div(
        F.log_softmax(student_logits, dim=-1),
        F.log_softmax(teacher_logits, dim=-1),
        log_target=True,
        reduction="none",
    ).sum(dim=-1)
    expected = (token_kl * mask.to(dtype=token_kl.dtype)).sum() / mask.sum()

    assert torch.allclose(actual, expected)


def test_nextlat_categorical_kl_rejects_logit_shape_mismatch() -> None:
    teacher_logits = torch.randn(2, 3, 5)
    student_logits = torch.randn(2, 4, 5)
    mask = torch.ones(2, 3, dtype=torch.bool)

    with pytest.raises(
        ValueError,
        match="teacher_logits and student_logits must have the same shape",
    ):
        categorical_kl_with_mask(
            teacher_logits=teacher_logits,
            student_logits=student_logits,
            mask=mask,
        )


def test_nextlat_categorical_kl_rejects_mask_shape_mismatch() -> None:
    teacher_logits = torch.randn(2, 3, 5)
    student_logits = torch.randn(2, 3, 5)
    mask = torch.ones(2, 4, dtype=torch.bool)

    with pytest.raises(
        ValueError,
        match="mask must match the first two logit dimensions",
    ):
        categorical_kl_with_mask(
            teacher_logits=teacher_logits,
            student_logits=student_logits,
            mask=mask,
        )


def test_nextlat_auxiliary_ce_matches_full_torch_reference() -> None:
    torch.manual_seed(0)
    logits = torch.randn(2, 73, 11)
    targets = torch.randint(low=0, high=11, size=(2, 73))
    mask = torch.zeros(2, 73, dtype=torch.bool)
    mask[0, :70] = True
    mask[1, 3:67] = True

    actual = cross_entropy_with_mask(logits, targets, mask)
    masked_targets = targets.masked_fill(~mask, -1)
    expected = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        masked_targets.reshape(-1),
        ignore_index=-1,
    )

    assert torch.allclose(actual, expected)


def test_nextlat_auxiliary_ce_rejects_target_shape_mismatch() -> None:
    logits = torch.randn(2, 3, 5)
    targets = torch.randint(low=0, high=5, size=(2, 4))
    mask = torch.ones(2, 4, dtype=torch.bool)

    with pytest.raises(
        ValueError,
        match="targets must match the first two logit dimensions",
    ):
        cross_entropy_with_mask(logits, targets, mask)


def test_nextlat_auxiliary_ce_rejects_mask_shape_mismatch() -> None:
    logits = torch.randn(2, 3, 5)
    targets = torch.randint(low=0, high=5, size=(2, 3))
    mask = torch.ones(2, 4, dtype=torch.bool)

    with pytest.raises(
        ValueError,
        match="mask and targets must have the same shape",
    ):
        cross_entropy_with_mask(logits, targets, mask)


@pytest.mark.parametrize("model_builder", (build_tiny_self_attention_lm, build_tiny_dat_lm))
def test_nextlat_loss_is_finite_and_backpropagates_through_repo_local_lms(model_builder) -> None:
    torch.manual_seed(0)
    model = model_builder()
    dynamics = NextLatDynamicsModel(
        hidden_dim=model.config.hidden_dim,
        dropout=0.0,
        proj_factor=1.0,
        use_bias=False,
    )
    input_tokens, target_tokens, target_mask = build_nextlat_batch()
    loss_config = NextLatLossConfig(
        lambda_mse=1.0,
        lambda_kl=0.5,
        lambda_ce=0.0,
        horizon=1,
        eos_token_id=2,
        sequence_boundary_policy="eos_document",
    )

    result = compute_nextlat_loss(
        model=model,
        dynamics_model=dynamics,
        input_tokens=input_tokens,
        target_tokens=target_tokens,
        target_mask=target_mask,
        config=loss_config,
    )
    result.loss.backward()

    model_grad_norm = sum(
        parameter.grad.detach().abs().sum().item()
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    dynamics_grad_norm = sum(
        parameter.grad.detach().abs().sum().item()
        for parameter in dynamics.parameters()
        if parameter.grad is not None
    )

    assert torch.isfinite(result.loss)
    assert torch.isfinite(result.next_token_loss)
    assert torch.isfinite(result.hidden_loss)
    assert torch.isfinite(result.kl_loss)
    assert torch.isfinite(result.auxiliary_token_loss)
    assert model_grad_norm > 0.0
    assert dynamics_grad_norm > 0.0


def test_nextlat_loss_encodes_full_shifted_block() -> None:
    torch.manual_seed(0)
    config = SelfAttentionLMConfig(
        vocab_size=32,
        max_seq_len=8,
        hidden_dim=32,
        n_heads=4,
        n_layers=1,
        dropout=0.0,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = RecordingSelfAttentionDecoderLM(config)
    dynamics = NextLatDynamicsModel(
        hidden_dim=model.config.hidden_dim,
        dropout=0.0,
        proj_factor=1.0,
        use_bias=False,
    )
    input_tokens, target_tokens, target_mask = build_nextlat_batch()

    compute_nextlat_loss(
        model=model,
        dynamics_model=dynamics,
        input_tokens=input_tokens,
        target_tokens=target_tokens,
        target_mask=target_mask,
        config=NextLatLossConfig(
            lambda_mse=1.0,
            lambda_kl=0.5,
            lambda_ce=0.0,
            horizon=1,
            eos_token_id=2,
            sequence_boundary_policy="eos_document",
        ),
    )

    assert model.encoded_input_shapes == [(2, input_tokens.shape[1] + 1)]
    assert model.encoded_attention_mask_shapes == [(2, input_tokens.shape[1] + 1)]


def test_nextlat_next_token_loss_matches_shifted_ntp_loss() -> None:
    torch.manual_seed(0)
    model = build_tiny_self_attention_lm()
    dynamics = NextLatDynamicsModel(
        hidden_dim=model.config.hidden_dim,
        dropout=0.0,
        proj_factor=1.0,
        use_bias=False,
    )
    input_tokens, target_tokens, target_mask = build_nextlat_batch()
    target_tokens_for_loss = target_tokens.masked_fill(~target_mask, -1)
    _, ntp_loss = model(input_tokens, targets=target_tokens_for_loss)

    result = compute_nextlat_loss(
        model=model,
        dynamics_model=dynamics,
        input_tokens=input_tokens,
        target_tokens=target_tokens,
        target_mask=target_mask,
        config=NextLatLossConfig(
            lambda_mse=1.0,
            lambda_kl=0.5,
            lambda_ce=0.0,
            horizon=1,
            eos_token_id=2,
            sequence_boundary_policy="eos_document",
        ),
    )

    assert ntp_loss is not None
    assert torch.allclose(result.next_token_loss, ntp_loss)


def test_nextlat_full_logit_path_keeps_auxiliary_ce_diagnostic_when_weight_is_zero() -> None:
    torch.manual_seed(0)
    model = build_tiny_self_attention_lm()
    dynamics = NextLatDynamicsModel(
        hidden_dim=model.config.hidden_dim,
        dropout=0.0,
        proj_factor=1.0,
        use_bias=False,
    )
    input_tokens, target_tokens, target_mask = build_nextlat_batch()

    result = compute_nextlat_loss(
        model=model,
        dynamics_model=dynamics,
        input_tokens=input_tokens,
        target_tokens=target_tokens,
        target_mask=target_mask,
        config=NextLatLossConfig(
            lambda_mse=1.0,
            lambda_kl=0.5,
            lambda_ce=0.0,
            horizon=1,
            eos_token_id=2,
            sequence_boundary_policy="eos_document",
        ),
    )

    assert result.auxiliary_token_loss.item() > 0.0


def test_nextlat_loss_matches_manual_full_logit_objective() -> None:
    torch.manual_seed(0)
    model = build_tiny_self_attention_lm()
    dynamics = NextLatDynamicsModel(
        hidden_dim=model.config.hidden_dim,
        dropout=0.0,
        proj_factor=1.0,
        use_bias=False,
    )
    input_tokens, target_tokens, target_mask = build_nextlat_batch()
    loss_config = NextLatLossConfig(
        lambda_mse=1.0,
        lambda_kl=0.5,
        lambda_ce=0.25,
        horizon=1,
        eos_token_id=2,
        sequence_boundary_policy="eos_document",
    )

    result = compute_nextlat_loss(
        model=model,
        dynamics_model=dynamics,
        input_tokens=input_tokens,
        target_tokens=target_tokens,
        target_mask=target_mask,
        config=loss_config,
    )
    full_tokens = torch.cat([input_tokens, target_tokens[:, -1:]], dim=1)
    full_valid_mask = torch.cat([target_mask[:, :1], target_mask], dim=1)
    input_token_embeddings, hidden_states = model.encode_for_objective(
        full_tokens,
        attention_mask=full_valid_mask,
    )
    next_token_loss = F.cross_entropy(
        model.lm_head(hidden_states[:, :-1]).reshape(-1, model.config.vocab_size),
        target_tokens.masked_fill(~target_mask, -1).reshape(-1),
        ignore_index=-1,
    )
    pred_states = dynamics(hidden_states[:, :-1], input_token_embeddings[:, 1:])
    target_states = hidden_states[:, 1:]
    hidden_mask = build_nextlat_prediction_mask(
        input_tokens=full_tokens,
        valid_token_mask=full_valid_mask,
        eos_token_id=2,
        shift=1,
        sequence_boundary_policy="eos_document",
    )
    token_mask = hidden_mask[:, :-1] & target_mask[:, 1:]
    student_logits = F.linear(pred_states[:, :-1], model.lm_head.weight.detach())
    teacher_logits = F.linear(target_states[:, :-1].detach(), model.lm_head.weight.detach())
    hidden_loss = smooth_l1_with_mask(pred_states, target_states.detach(), hidden_mask)
    kl_loss = categorical_kl_with_mask(
        teacher_logits=teacher_logits,
        student_logits=student_logits,
        mask=token_mask,
    )
    auxiliary_token_loss = cross_entropy_with_mask(
        student_logits,
        target_tokens[:, 1:],
        token_mask,
    )
    expected_loss = next_token_loss + hidden_loss + 0.5 * kl_loss + 0.25 * auxiliary_token_loss

    assert torch.allclose(result.loss, expected_loss, atol=1e-6)
    assert torch.allclose(result.next_token_loss, next_token_loss, atol=1e-6)
    assert torch.allclose(result.hidden_loss, hidden_loss, atol=1e-6)
    assert torch.allclose(result.kl_loss, kl_loss, atol=1e-6)
    assert torch.allclose(result.auxiliary_token_loss, auxiliary_token_loss, atol=1e-6)


def test_nextlat_hidden_loss_uses_final_target_token_embedding() -> None:
    torch.manual_seed(0)
    config = SelfAttentionLMConfig(
        vocab_size=16,
        max_seq_len=4,
        hidden_dim=16,
        n_heads=4,
        n_layers=1,
        dropout=0.0,
        tie_lm_head=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = SelfAttentionDecoderLM(config)
    dynamics = NextLatDynamicsModel(
        hidden_dim=model.config.hidden_dim,
        dropout=0.0,
        proj_factor=1.0,
        use_bias=False,
    )
    input_tokens = torch.tensor([[1, 3, 4]], dtype=torch.long)
    target_tokens = torch.tensor([[3, 4, 5]], dtype=torch.long)
    target_mask = torch.ones_like(target_tokens, dtype=torch.bool)

    result = compute_nextlat_loss(
        model=model,
        dynamics_model=dynamics,
        input_tokens=input_tokens,
        target_tokens=target_tokens,
        target_mask=target_mask,
        config=NextLatLossConfig(
            lambda_mse=1.0,
            lambda_kl=0.0,
            lambda_ce=0.0,
            horizon=1,
            eos_token_id=2,
            sequence_boundary_policy="eos_document",
        ),
    )
    result.hidden_loss.backward()

    final_token_grad = model.token_embeddings.weight.grad[5].detach().abs().sum().item()
    assert final_token_grad > 0.0


def test_nextlat_prediction_mask_excludes_eos_target_states_under_eos_document_policy() -> None:
    input_tokens = torch.tensor([[1, 4, 2, 5, 6, 2, 7]], dtype=torch.long)
    valid_token_mask = torch.ones_like(input_tokens, dtype=torch.bool)

    eos_mask = build_nextlat_prediction_mask(
        input_tokens=input_tokens,
        valid_token_mask=valid_token_mask,
        eos_token_id=2,
        shift=1,
        sequence_boundary_policy="eos_document",
    )

    assert eos_mask.tolist() == [[True, False, True, True, False, True]]


def test_optimizer_can_include_nextlat_dynamics_parameters() -> None:
    model = build_tiny_self_attention_lm()
    dynamics = NextLatDynamicsModel(
        hidden_dim=model.config.hidden_dim,
        dropout=0.0,
        proj_factor=1.0,
        use_bias=False,
    )
    config = TrainConfig(
        model_type="self_attention",
        hidden_dim=32,
        n_heads=4,
        nextlat_enabled=True,
    )

    optimizer, _ = build_optimizer_and_scheduler(config, (model, dynamics), total_training_steps=10)

    optimized_parameter_count = sum(
        parameter.numel()
        for group in optimizer.param_groups
        for parameter in group["params"]
    )
    expected_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    expected_parameter_count += sum(parameter.numel() for parameter in dynamics.parameters())
    assert optimized_parameter_count == expected_parameter_count


def test_train_epoch_uses_nextlat_objective_with_repo_local_data(tmp_path) -> None:
    torch.manual_seed(0)
    config = build_tiny_train_config(
        tmp_path,
        "nextlat-train-epoch",
        model_type="self_attention",
        n_layers=1,
        nextlat_enabled=True,
        nextlat_lambda_kl=0.5,
    )
    paths = resolve_train_paths(config)
    ensure_experiment_dirs(paths)
    write_experiment_config(config, paths)
    data_artifacts = load_train_data(config, paths)
    model_config = build_model_config(config, data_artifacts.tokenizer)
    write_model_config(model_config, paths)
    model = build_model(config.model_type, model_config)
    dynamics = build_nextlat_dynamics_model(config, model)
    assert dynamics is not None
    total_steps = resolve_total_training_steps(config, len(data_artifacts.dataloader))
    optimizer, scheduler = build_optimizer_and_scheduler(config, (model, dynamics), total_steps)
    optimizer.zero_grad(set_to_none=True)

    metrics = train_epoch(
        config=config,
        paths=paths,
        model=model,
        nextlat_dynamics_model=dynamics,
        optimizer=optimizer,
        scheduler=scheduler,
        dataloader=data_artifacts.dataloader,
        epoch=0,
        start_time=0.0,
        wandb_module=None,
    )

    assert metrics["loss"] > 0.0
    assert metrics["next_token_loss"] > 0.0
    assert metrics["nextlat_hidden_loss"] >= 0.0
    assert metrics["nextlat_kl_loss"] >= 0.0
    assert metrics["nextlat_auxiliary_token_loss"] >= 0.0
