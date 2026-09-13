"""Pytest coverage for the local self-attention LM positional encodings."""

import pytest
import torch

from models.attention.transformer.config import SelfAttentionLMConfig
from models.attention.transformer.core import CrossAttention, MultiHeadAttentionBase
from models.attention.transformer.lm import SelfAttentionDecoderLM


def test_self_attention_lm_config_defaults_to_current_transformer_width() -> None:
    config = SelfAttentionLMConfig(vocab_size=16, max_seq_len=8)

    assert config.hidden_dim == 192
    assert config.n_heads == 6
    assert config.hidden_dim // config.n_heads == 32
    assert config.n_layers == 4
    assert config.tie_lm_head is True


def test_self_attention_decoder_lm_can_tie_or_untie_lm_head() -> None:
    tied_config = SelfAttentionLMConfig(
        vocab_size=16,
        max_seq_len=8,
        hidden_dim=32,
        n_heads=4,
        n_layers=1,
    )
    tied_model = SelfAttentionDecoderLM(tied_config)

    assert tied_model.lm_head.weight is tied_model.token_embeddings.weight

    untied_config = SelfAttentionLMConfig(
        vocab_size=16,
        max_seq_len=8,
        hidden_dim=32,
        n_heads=4,
        n_layers=1,
        tie_lm_head=False,
    )
    untied_model = SelfAttentionDecoderLM(untied_config)

    assert untied_model.lm_head.weight is not untied_model.token_embeddings.weight
    assert untied_model.lm_head.weight.shape == untied_model.token_embeddings.weight.shape


def build_controlled_generation_model(token_embeddings: torch.Tensor) -> SelfAttentionDecoderLM:
    config = SelfAttentionLMConfig(
        vocab_size=token_embeddings.shape[0],
        max_seq_len=16,
        hidden_dim=token_embeddings.shape[1],
        n_heads=4,
        n_layers=1,
        dropout=0.0,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = SelfAttentionDecoderLM(config)

    with torch.no_grad():
        model.token_embeddings.weight.copy_(token_embeddings)
        for name, parameter in model.named_parameters():
            if name == "token_embeddings.weight":
                continue
            parameter.zero_()
        model.final_norm.weight.fill_(1.0)

    return model


def test_attention_mask_fill_is_safe_for_float16_scores() -> None:
    attention = MultiHeadAttentionBase(
        query_dim=4,
        output_dim=4,
        n_heads=1,
        hidden_dim=4,
        dropout=0.0,
    )
    scores = torch.tensor(
        [[[[0.0, 1.0], [2.0, 3.0]]]],
        dtype=torch.float16,
    )
    mask = torch.tensor([[[True, False], [True, True]]])

    weights = attention._apply_mask(scores, mask)

    assert weights.dtype == torch.float16
    assert torch.isfinite(weights).all()
    assert weights[0, 0, 0, 0].item() == pytest.approx(1.0)
    assert weights[0, 0, 0, 1].item() == pytest.approx(0.0)


def test_cross_attention_supports_explicit_context_dimension_and_masks() -> None:
    torch.manual_seed(0)
    attention = CrossAttention(
        input_dim=4,
        context_dim=6,
        output_dim=8,
        n_heads=2,
        hidden_dim=8,
        dropout=0.0,
    )
    inputs = torch.randn(2, 3, 4)
    context = torch.randn(2, 3, 6)
    mask = torch.tensor(
        [
            [True, False, False],
            [True, True, False],
            [False, False, False],
        ],
        dtype=torch.bool,
    )

    output, weights = attention(inputs, context, mask=mask)

    assert attention.q_proj.in_features == 4
    assert attention.k_proj.in_features == 6
    assert attention.v_proj.in_features == 6
    assert output.shape == (2, 3, 8)
    assert weights.shape == (2, 2, 3, 3)
    assert torch.equal(
        weights.masked_select(~mask.unsqueeze(0).unsqueeze(0)),
        torch.zeros_like(weights.masked_select(~mask.unsqueeze(0).unsqueeze(0))),
    )
    assert torch.equal(weights[:, :, 2, :], torch.zeros_like(weights[:, :, 2, :]))


def test_cross_attention_rejects_low_level_2d_padding_masks() -> None:
    attention = CrossAttention(
        input_dim=4,
        context_dim=6,
        output_dim=8,
        n_heads=2,
        hidden_dim=8,
        dropout=0.0,
    )
    inputs = torch.randn(2, 3, 4)
    context = torch.randn(2, 3, 6)
    padding_mask = torch.ones(2, 3, dtype=torch.bool)

    with pytest.raises(ValueError, match="2D decoder attention masks must be square"):
        attention(inputs, context, mask=padding_mask)


def test_self_attention_decoder_lm_attention_mask_resets_after_eos_by_default() -> None:
    model = SelfAttentionDecoderLM(SelfAttentionLMConfig(vocab_size=16, max_seq_len=8))
    input_ids = torch.tensor([[1, 5, 2, 6, 0]], dtype=torch.long)

    mask = model._build_attention_mask(input_ids)

    assert mask[0].tolist() == [
        [True, False, False, False, False],
        [True, True, False, False, False],
        [False, False, True, False, False],
        [False, False, True, True, False],
        [False, False, False, False, False],
    ]


def test_self_attention_decoder_lm_forward_and_generate_with_rope() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    config = SelfAttentionLMConfig(
        vocab_size=128,
        max_seq_len=16,
        hidden_dim=32,
        n_heads=4,
        n_layers=2,
        dropout=0.0,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = SelfAttentionDecoderLM(config).to(device)

    input_ids = torch.tensor(
        [
            [1, 7, 8, 9, 2, 0],
            [1, 3, 4, 5, 6, 2],
        ],
        dtype=torch.long,
        device=device,
    )
    targets = torch.tensor(
        [
            [7, 8, 9, 2, -1, -1],
            [3, 4, 5, 6, 2, -1],
        ],
        dtype=torch.long,
        device=device,
    )
    attention_mask = input_ids != 0

    logits, loss = model(input_ids, targets=targets, attention_mask=attention_mask)

    assert logits.shape == (2, 6, 128)
    assert loss is not None
    assert torch.isfinite(loss)

    generated = model.generate(
        input_ids[:, :3],
        attention_mask=attention_mask[:, :3],
        max_new_tokens=2,
        do_sample=False,
    )

    assert generated.shape == (2, 5)


def test_self_attention_decoder_lm_rejects_mismatched_targets_shape() -> None:
    config = SelfAttentionLMConfig(
        vocab_size=64,
        max_seq_len=8,
        pe_type="rope",
        hidden_dim=32,
        n_heads=4,
        n_layers=1,
        dropout=0.0,
    )
    model = SelfAttentionDecoderLM(config)
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    targets = torch.tensor([[1, 2, 3]], dtype=torch.long)

    with pytest.raises(ValueError, match="targets shape must match input_ids shape"):
        model(input_ids, targets=targets)


def test_self_attention_decoder_lm_exposes_hidden_states_for_training_objectives() -> None:
    torch.manual_seed(0)
    config = SelfAttentionLMConfig(
        vocab_size=64,
        max_seq_len=8,
        hidden_dim=32,
        n_heads=4,
        n_layers=1,
        dropout=0.0,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = SelfAttentionDecoderLM(config)
    input_ids = torch.tensor(
        [
            [1, 5, 6, 2, 0],
            [1, 7, 8, 9, 2],
        ],
        dtype=torch.long,
    )
    attention_mask = input_ids != config.pad_token_id

    token_embeddings, hidden_states = model.encode_for_objective(
        input_ids,
        attention_mask=attention_mask,
    )
    logits, loss = model(input_ids, attention_mask=attention_mask)

    assert token_embeddings.shape == (2, 5, config.hidden_dim)
    assert hidden_states.shape == (2, 5, config.hidden_dim)
    assert torch.equal(token_embeddings, model.token_embeddings(input_ids))
    assert torch.allclose(logits, model.lm_head(hidden_states), atol=1e-6)
    assert loss is None


def test_self_attention_decoder_lm_generate_stops_on_eos() -> None:
    config = SelfAttentionLMConfig(
        vocab_size=16,
        max_seq_len=8,
        pe_type="rope",
        hidden_dim=8,
        n_heads=4,
        n_layers=1,
        dropout=0.0,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = SelfAttentionDecoderLM(config)

    with torch.no_grad():
        model.token_embeddings.weight.zero_()
        model.token_embeddings.weight[config.bos_token_id, 0] = 1.0
        model.token_embeddings.weight[config.eos_token_id, 0] = 2.0
        for name, parameter in model.named_parameters():
            if name == "token_embeddings.weight":
                continue
            parameter.zero_()
        model.final_norm.weight.fill_(1.0)

    generated = model.generate(
        torch.tensor([[config.bos_token_id]], dtype=torch.long),
        max_new_tokens=5,
        do_sample=False,
    )

    assert generated.tolist() == [[config.bos_token_id, config.eos_token_id]]


def test_self_attention_decoder_lm_generate_uses_last_unmasked_token_in_batched_prompts() -> None:
    token_embeddings = torch.zeros(8, 8)
    token_embeddings[0, 0] = 1.0
    token_embeddings[1, 1] = 1.0
    token_embeddings[4, 4] = 1.0
    token_embeddings[5, 5] = 1.0
    token_embeddings[6, 6] = 1.0
    model = build_controlled_generation_model(token_embeddings)

    input_ids = torch.tensor(
        [
            [1, 4, 0],
            [1, 5, 6],
        ],
        dtype=torch.long,
    )
    attention_mask = input_ids != 0

    generated = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=1,
        do_sample=False,
    )

    assert generated[0, :3].tolist() == [1, 4, 4]
    assert generated[0, 3].item() == 0
    assert generated[1].tolist() == [1, 5, 6, 6]


def test_self_attention_decoder_lm_generate_freezes_finished_rows_in_batched_generation() -> None:
    token_embeddings = torch.zeros(8, 8)
    token_embeddings[0, 0] = 1.0
    token_embeddings[1, 1] = 1.0
    token_embeddings[4, 5] = 0.1
    token_embeddings[5, 4] = 0.1
    token_embeddings[6, 4] = 1.0
    token_embeddings[6, 5] = 1.0
    token_embeddings[2, 5] = 3.0
    model = build_controlled_generation_model(token_embeddings)

    input_ids = torch.tensor(
        [
            [1, 4, 0],
            [1, 5, 0],
        ],
        dtype=torch.long,
    )
    attention_mask = input_ids != 0

    generated = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=2,
        do_sample=False,
    )

    assert generated[0].tolist() == [1, 4, 2, 0]
    assert generated[1].tolist() == [1, 5, 6, 2]
