"""Pytest coverage for the owned DAT decoder LM backend."""

import math

import pytest
import torch

from models.attention.dat.config import DatLMConfig
from models.attention.dat.core import (
    DisentangledRelationalCrossAttention,
    RelationalAttention,
    RelationalAttentionBase,
    RelationalCrossAttention,
)
from models.attention.dat.lm import DatDecoderBlock, DatDecoderLM
from models.attention.dat.symbols import (
    PositionalSymbolRetriever,
    RelationalSymbolicAttentionRetriever,
    RelativePositionalSymbolRetriever,
    SymbolicAttentionRetriever,
)
from models.attention.transformer.components import PositionalEncoding
from models.attention.transformer.core import MultiHeadAttentionBase, SelfAttention
from models.factory import build_model
from train.config import parse_train_config

DSSL_SYMBOL_RETRIEVAL_TYPES = ("symbolic", "positional", "relative", "relsymbolic")
DSSL_RA_TYPES = ("ra", "rca", "disrca")
DSSL_RA_ACTIVATIONS = ("softmax", "identity", "tanh", "sigmoid")
DSSL_FFN_ACTIVATIONS = ("gelu", "swiglu")

class RecordingSymbolRetriever(torch.nn.Module):
    def __init__(self, symbol_dim: int):
        super().__init__()
        self.symbol_dim = symbol_dim
        self.last_input = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.last_input = x.detach().clone()
        return x.new_zeros(x.shape[0], x.shape[1], self.symbol_dim)


def build_tiny_dat_config(**overrides) -> DatLMConfig:
    values = {
        "vocab_size": 32,
        "max_seq_len": 12,
        "pe_type": "rope",
        "hidden_dim": 32,
        "n_heads_sa": 2,
        "n_heads_ra": 2,
        "n_layers": 2,
        "dropout": 0.0,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
    }
    values.update(overrides)
    return DatLMConfig(**values)


def build_controlled_dat_generation_model(token_embeddings: torch.Tensor) -> DatDecoderLM:
    config = build_tiny_dat_config(
        vocab_size=token_embeddings.shape[0],
        hidden_dim=token_embeddings.shape[1],
        n_layers=1,
    )
    model = DatDecoderLM(config)

    with torch.no_grad():
        model.token_embeddings.weight.copy_(token_embeddings)
        for name, parameter in model.named_parameters():
            if name == "token_embeddings.weight":
                continue
            parameter.zero_()
        for module in model.modules():
            if hasattr(module, "weight") and module.__class__.__name__ in {"RMSNorm", "LayerNorm"}:
                module.weight.fill_(1.0)

    return model


def test_dat_config_validates_head_layout() -> None:
    config = build_tiny_dat_config()

    assert config.total_n_heads == 4
    assert config.head_dim == 8
    assert config.resolved_symbol_dim == 32
    assert config.resolved_symbolic_attn_n_heads == 4
    assert config.resolved_ffn_hidden_dim == 32 * config.dff_factor
    assert config.n_symbols is None
    assert config.resolved_n_symbols == config.max_seq_len
    assert config.resolved_ra_n_relations == 2
    assert config.symbol_retrieval == "symbolic"
    assert config.init_scheme == "xavier_uniform"
    assert config.tie_lm_head is True

    with pytest.raises(ValueError, match="n_heads_ra must be positive"):
        build_tiny_dat_config(n_heads_ra=0)

    with pytest.raises(ValueError, match="hidden_dim .* must be divisible"):
        build_tiny_dat_config(hidden_dim=30)

    with pytest.raises(ValueError, match="Unsupported symbol_retrieval"):
        build_tiny_dat_config(symbol_retrieval="symbolic_attention")

    with pytest.raises(ValueError, match="symbolic_attn_n_heads"):
        build_tiny_dat_config(symbolic_attn_n_heads=0)

    with pytest.raises(ValueError, match="must be divisible by symbolic_attn_n_heads"):
        build_tiny_dat_config(symbol_retrieval="symbolic", symbolic_attn_n_heads=3)


def test_dat_config_validates_relative_rope_symbol_source() -> None:
    config = build_tiny_dat_config(symbol_retrieval="relative", relative_symbols_rope=True)

    assert config.symbol_retrieval == "relative"
    assert config.relative_symbols_rope is True

    with pytest.raises(ValueError, match="requires symbol_retrieval='relative'"):
        build_tiny_dat_config(symbol_retrieval="positional", relative_symbols_rope=True)

    with pytest.raises(ValueError, match="RoPE relative symbols require even symbol_dim"):
        build_tiny_dat_config(
            symbol_retrieval="relative",
            symbol_dim=31,
            relative_symbols_rope=True,
        )


def test_dat_config_accepts_dssl_symbol_retrieval_and_ra_types() -> None:
    for symbol_retrieval in DSSL_SYMBOL_RETRIEVAL_TYPES:
        config = build_tiny_dat_config(symbol_retrieval=symbol_retrieval)
        assert config.symbol_retrieval == symbol_retrieval

    for ra_type in DSSL_RA_TYPES:
        config = build_tiny_dat_config(ra_type=ra_type)
        assert config.ra_type == ra_type

    for ffn_activation in DSSL_FFN_ACTIVATIONS:
        config = build_tiny_dat_config(ffn_activation=ffn_activation)
        assert config.ffn_activation == ffn_activation


def test_dat_config_rejects_invalid_ra_type_and_activation() -> None:
    with pytest.raises(ValueError, match="Unsupported ra_type"):
        build_tiny_dat_config(ra_type="cross")

    with pytest.raises(ValueError, match="Unsupported ra_rel_activation"):
        build_tiny_dat_config(ra_rel_activation="swish")

    with pytest.raises(ValueError, match="Unsupported ffn_activation"):
        build_tiny_dat_config(ffn_activation="swish")

    with pytest.raises(ValueError, match="Unsupported ffn_hidden_dim_mode"):
        build_tiny_dat_config(ffn_hidden_dim_mode="unsupported_mode")

    with pytest.raises(ValueError, match="Unsupported init_scheme"):
        build_tiny_dat_config(init_scheme="unsupported_scheme")

    with pytest.raises(ValueError, match="applies only to ra_type='ra'"):
        build_tiny_dat_config(ra_type="rca", ra_n_relations=2)


def test_factory_builds_dat_decoder_lm() -> None:
    config = build_tiny_dat_config()
    model = build_model("dat", config)

    assert isinstance(model, DatDecoderLM)

    expanded_config = build_tiny_dat_config(symbol_retrieval="relsymbolic", ra_type="disrca")
    expanded_model = build_model("dat", expanded_config)

    assert isinstance(expanded_model, DatDecoderLM)


def test_dat_decoder_lm_exposes_hidden_states_for_training_objectives() -> None:
    torch.manual_seed(0)
    config = build_tiny_dat_config(n_layers=1, dropout=0.0)
    model = DatDecoderLM(config)
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


def test_parse_train_config_accepts_dat_backend() -> None:
    config = parse_train_config(
        [
            "--tokenizer_dir", "test/fixtures/tiny_tokenizer",
            "--train_data_dir", "test/fixtures/train_assets/clean_train_tiny",
            "--model_type",
            "dat",
            "--hidden_dim",
            "32",
            "--n_heads_sa",
            "2",
            "--n_heads_ra",
            "2",
            "--n_symbols",
            "8",
            "--symbolic_attn_n_heads",
            "2",
            "--symbol_retrieval",
            "relsymbolic",
            "--norm_type",
            "layernorm",
            "--ra_type",
            "disrca",
            "--ra_rel_activation",
            "sigmoid",
            "--use_bias_qkv",
            "--no_use_bias_out",
            "--no_use_bias_ffn",
            "--ffn_hidden_dim_mode",
            "dff_factor",
            "--ffn_activation",
            "swiglu",
            "--symbolic_use_bias",
            "--relsymbolic_neighborhood_size",
            "1",
            "--relsymbolic_include_self",
            "--no_relsymbolic_normalize_rels",
            "--relsymbolic_rel_scale",
            "0.25",
            "--relsymbolic_symbolic_attn_scale",
            "0.5",
            "--relsymbolic_use_bias",
            "--warmup_ratio",
            "0.05",
            "--init_scheme",
            "normal_0_02_scaled_projection",
            "--sequence_boundary_policy",
            "document_boundary",
        ]
    )

    assert config.model_type == "dat"
    assert config.n_heads_sa == 2
    assert config.n_heads_ra == 2
    assert config.n_symbols == 8
    assert config.symbolic_attn_n_heads == 2
    assert config.symbol_retrieval == "relsymbolic"
    assert config.norm_type == "layernorm"
    assert config.ra_type == "disrca"
    assert config.ra_rel_activation == "sigmoid"
    assert config.use_bias_qkv
    assert not config.use_bias_out
    assert not config.use_bias_ffn
    assert config.ffn_hidden_dim_mode == "dff_factor"
    assert config.ffn_activation == "swiglu"
    assert config.symbolic_use_bias
    assert config.relsymbolic_neighborhood_size == 1
    assert config.relsymbolic_include_self
    assert not config.relsymbolic_normalize_rels
    assert config.relsymbolic_rel_scale == pytest.approx(0.25)
    assert config.relsymbolic_symbolic_attn_scale == pytest.approx(0.5)
    assert config.relsymbolic_use_bias
    assert config.warmup_ratio == pytest.approx(0.05)
    assert config.init_scheme == "normal_0_02_scaled_projection"
    assert config.sequence_boundary_policy == "document_boundary"


def test_parse_train_config_accepts_relative_rope_symbols() -> None:
    config = parse_train_config(
        [
            "--tokenizer_dir", "test/fixtures/tiny_tokenizer",
            "--train_data_dir", "test/fixtures/train_assets/clean_train_tiny",
            "--model_type",
            "dat",
            "--symbol_retrieval",
            "relative",
            "--relative_symbols_rope",
        ]
    )

    assert config.symbol_retrieval == "relative"
    assert config.relative_symbols_rope is True

    with pytest.raises(ValueError, match="even symbol_dim"):
        build_tiny_dat_config(
            symbol_retrieval="relative",
            relative_symbols_rope=True,
            symbol_dim=31,
        )


def test_parse_train_config_rejects_babylm_symbolic_attention_spelling() -> None:
    with pytest.raises(SystemExit):
        parse_train_config([
            "--tokenizer_dir", "test/fixtures/tiny_tokenizer",
            "--train_data_dir", "test/fixtures/train_assets/clean_train_tiny",
            "--model_type", "dat", "--symbol_retrieval", "symbolic_attention"])


def test_dat_symbol_retrievers_return_dssl_shapes() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 5, 32)

    symbolic = SymbolicAttentionRetriever(
        hidden_dim=32,
        symbol_dim=32,
        n_symbols=8,
        n_heads=4,
    )
    positional = PositionalSymbolRetriever(symbol_dim=32, max_len=8)
    positional_sinusoidal = PositionalSymbolRetriever(symbol_dim=32, max_len=8, sinusoidal=True)
    relative = RelativePositionalSymbolRetriever(symbol_dim=32, max_rel_distance=4)
    relative_rope = RelativePositionalSymbolRetriever(symbol_dim=32, max_rel_distance=4, rope=True)
    relsymbolic = RelationalSymbolicAttentionRetriever(
        hidden_dim=32,
        symbol_dim=32,
        rel_n_heads=4,
        symbolic_attn_n_heads=4,
        n_symbols=8,
        neighborhood_size=2,
    )

    assert symbolic(x).shape == (2, 5, 32)
    assert positional(x).shape == (2, 5, 32)
    assert positional_sinusoidal(x).shape == (2, 5, 32)
    assert relative(x).shape == (5, 5, 32)
    assert relative_rope(x).shape == (5, 5, 32)
    assert relsymbolic(x).shape == (2, 5, 32)


def test_dat_model_uses_symbolic_head_override_and_bias() -> None:
    config = build_tiny_dat_config(
        symbol_retrieval="symbolic",
        symbolic_attn_n_heads=2,
        symbolic_use_bias=True,
        n_layers=1,
    )
    model = DatDecoderLM(config)

    retriever = model.symbol_retriever

    assert isinstance(retriever, SymbolicAttentionRetriever)
    assert retriever.n_heads == 2
    assert retriever.q_proj.bias is not None


def test_dat_uses_shared_dssl_attention_hierarchy_for_sensory_and_rca() -> None:
    config = build_tiny_dat_config(ra_type="rca", n_layers=1)
    model = DatDecoderLM(config)
    layer = model.layers[0]

    assert isinstance(layer.sensory_attention, SelfAttention)
    assert isinstance(layer.sensory_attention, MultiHeadAttentionBase)
    assert isinstance(layer.relational_attention, RelationalCrossAttention)
    assert isinstance(layer.relational_attention, MultiHeadAttentionBase)
    assert layer.sensory_attention.output_dim == config.n_heads_sa * config.head_dim
    assert layer.relational_attention.output_dim == config.n_heads_ra * config.head_dim


@pytest.mark.parametrize(
    ("ra_type", "expected_class"),
    (("ra", RelationalAttention), ("disrca", DisentangledRelationalCrossAttention)),
)
def test_dat_uses_dssl_relational_attention_base_for_ra_and_disrca(
    ra_type: str,
    expected_class: type[torch.nn.Module],
) -> None:
    config = build_tiny_dat_config(ra_type=ra_type, n_layers=1)
    model = DatDecoderLM(config)
    relational_attention = model.layers[0].relational_attention

    assert isinstance(relational_attention, expected_class)
    assert isinstance(relational_attention, RelationalAttentionBase)
    assert isinstance(relational_attention, MultiHeadAttentionBase)
    if ra_type == "disrca":
        assert not isinstance(relational_attention, RelationalAttention)
    assert relational_attention.output_dim == config.n_heads_ra * config.head_dim


@pytest.mark.parametrize("activation", DSSL_RA_ACTIVATIONS)
def test_owned_rca_zeroes_masked_attention_weights_for_decoder_lm(activation: str) -> None:
    torch.manual_seed(0)
    module = RelationalCrossAttention(
        hidden_dim=8,
        symbol_dim=8,
        n_heads=2,
        total_n_heads=4,
        dropout=0.0,
        activation=activation,
    )
    x = torch.randn(1, 3, 8)
    symbols = torch.randn(1, 3, 8)
    mask = torch.tensor(
        [
            [True, False, False],
            [True, True, False],
            [False, False, False],
        ],
        dtype=torch.bool,
    )

    _, info = module(x, symbols, mask=mask, pos_info=None)

    assert torch.equal(
        info["attention"].masked_select(~mask.unsqueeze(0).unsqueeze(0)),
        torch.zeros_like(info["attention"].masked_select(~mask.unsqueeze(0).unsqueeze(0))),
    )
    assert torch.equal(
        info["attention"][:, :, 2, :],
        torch.zeros_like(info["attention"][:, :, 2, :]),
    )


@pytest.mark.parametrize("activation", DSSL_RA_ACTIVATIONS)
@pytest.mark.parametrize("module_class", (RelationalAttention, DisentangledRelationalCrossAttention))
def test_owned_ra_and_disrca_zero_masked_relation_diagnostics_for_decoder_lm(
    activation: str,
    module_class: type[torch.nn.Module],
) -> None:
    torch.manual_seed(0)
    kwargs = {
        "hidden_dim": 8,
        "symbol_dim": 8,
        "n_heads": 2,
        "total_n_heads": 4,
        "dropout": 0.0,
        "rel_activation": activation,
    }
    if module_class is RelationalAttention:
        kwargs["n_relations"] = 2
    module = module_class(**kwargs)
    x = torch.randn(1, 3, 8)
    symbols = torch.randn(1, 3, 8)
    mask = torch.tensor(
        [
            [True, False, False],
            [True, True, False],
            [False, False, False],
        ],
        dtype=torch.bool,
    )

    _, info = module(x, symbols, mask=mask, pos_info=None)

    if module_class is RelationalAttention:
        relation_mask = mask.unsqueeze(0).unsqueeze(-1)
        all_masked_query = info["relations"][:, 2, :, :]
    else:
        relation_mask = mask.unsqueeze(0).unsqueeze(0)
        all_masked_query = info["relations"][:, :, 2, :]
    assert torch.equal(
        info["relations"].masked_select(~relation_mask),
        torch.zeros_like(info["relations"].masked_select(~relation_mask)),
    )
    assert torch.equal(all_masked_query, torch.zeros_like(all_masked_query))


@pytest.mark.parametrize("activation", DSSL_RA_ACTIVATIONS)
def test_shared_attention_base_zeroes_masked_weights_for_decoder_lm(activation: str) -> None:
    attention = MultiHeadAttentionBase(
        query_dim=4,
        output_dim=4,
        n_heads=1,
        hidden_dim=4,
        dropout=0.0,
        activation=activation,
    )
    scores = torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]])
    mask = torch.tensor([[True, False], [False, False]])

    weights = attention._apply_activation_and_mask(scores, mask)

    assert torch.equal(
        weights.masked_select(~mask.unsqueeze(0).unsqueeze(0)),
        torch.zeros_like(weights.masked_select(~mask.unsqueeze(0).unsqueeze(0))),
    )
    assert torch.equal(weights[:, :, 1, :], torch.zeros_like(weights[:, :, 1, :]))



def test_positional_symbol_retriever_returns_position_embeddings_by_index() -> None:
    retriever = PositionalSymbolRetriever(symbol_dim=4, max_len=6)
    expected_table = torch.arange(24, dtype=torch.float32).view(6, 4)
    with torch.no_grad():
        retriever.position_encoder.position_embeddings.weight.copy_(expected_table)
    x = torch.zeros(2, 3, 5)

    symbols = retriever(x)

    assert torch.equal(symbols, expected_table[:3].unsqueeze(0).expand(2, -1, -1))


def test_sinusoidal_positional_encoding_constructs_under_meta_default_device() -> None:
    with torch.device("meta"):
        encoder = PositionalEncoding(embedding_dim=8, pe_type="sinusoidal", max_len=6)

    assert encoder.sinusoidal_embeddings.device.type == "meta"
    embeddings = encoder.get_positional_info(seq_len=4, device=torch.device("cpu")).embeddings
    expected = PositionalEncoding(embedding_dim=8, pe_type="sinusoidal", max_len=6).get_positional_info(
        seq_len=4,
        device=torch.device("cpu"),
    ).embeddings

    assert embeddings is not None
    assert expected is not None
    assert embeddings.device.type == "cpu"
    assert torch.equal(embeddings, expected)


def test_relative_positional_symbol_retriever_uses_key_minus_query_orientation_and_clipping() -> None:
    retriever = RelativePositionalSymbolRetriever(symbol_dim=1, max_rel_distance=2)
    with torch.no_grad():
        retriever.position_encoder.rel_pos_embeddings_table.weight.copy_(
            torch.arange(5, dtype=torch.float32).unsqueeze(1)
        )
    x = torch.zeros(1, 5, 3)
    positions = torch.arange(5)
    expected_indices = (positions[None, :] - positions[:, None]).clamp(-2, 2) + 2

    symbols = retriever(x).squeeze(-1)

    assert torch.equal(symbols, expected_indices.float())


@pytest.mark.parametrize(
    ("include_self", "expected"),
    (
        (False, torch.tensor([[0, 0], [0, 0], [1, 0], [2, 1], [3, 2]])),
        (True, torch.tensor([[0, 0, 0], [1, 0, 0], [2, 1, 0], [3, 2, 1], [4, 3, 2]])),
    ),
)
def test_relsymbolic_neighborhood_indices_are_causal_decoder_lm_adaptation(
    include_self: bool,
    expected: torch.Tensor,
) -> None:
    retriever = RelationalSymbolicAttentionRetriever(
        hidden_dim=8,
        symbol_dim=8,
        rel_n_heads=2,
        symbolic_attn_n_heads=2,
        n_symbols=4,
        neighborhood_size=2,
        include_self=include_self,
    )

    indices = retriever._compute_neighborhood_indices(seq_len=5, device=torch.device("cpu"))

    assert torch.equal(indices, expected)
    assert torch.all(indices <= torch.arange(5).unsqueeze(1))
    assert retriever.neighborhood_dim == (3 if include_self else 2)
    assert retriever.rel_feature_dim == retriever.rel_n_heads * retriever.neighborhood_dim


def test_relsymbolic_supports_scale_overrides_and_bias_while_staying_causal() -> None:
    retriever = RelationalSymbolicAttentionRetriever(
        hidden_dim=8,
        symbol_dim=8,
        rel_n_heads=2,
        symbolic_attn_n_heads=2,
        n_symbols=4,
        neighborhood_size=2,
        include_self=False,
        rel_scale=0.25,
        symbolic_attn_scale=0.5,
        use_bias=True,
    )

    indices = retriever._compute_neighborhood_indices(seq_len=5, device=torch.device("cpu"))

    assert torch.all(indices <= torch.arange(5).unsqueeze(1))
    assert retriever.rel_scale == pytest.approx(0.25)
    assert retriever.symbolic_attention.scale == pytest.approx(0.5)
    assert retriever.q_proj.bias is not None
    assert retriever.k_proj.bias is not None
    assert retriever.symbolic_attention.q_proj.bias is not None



def test_dat_decoder_lm_swiglu_uses_dff_factor_projection_shapes_by_default() -> None:
    config = build_tiny_dat_config(ffn_activation="swiglu", dff_factor=3, n_layers=1)
    model = DatDecoderLM(config)
    feed_forward = model.layers[0].feed_forward

    assert config.resolved_ffn_hidden_dim == config.hidden_dim * config.dff_factor
    assert feed_forward.activation == "swiglu"
    assert feed_forward.w_gate.in_features == config.hidden_dim
    assert feed_forward.w_gate.out_features == config.resolved_ffn_hidden_dim
    assert feed_forward.w_up.in_features == config.hidden_dim
    assert feed_forward.w_up.out_features == config.resolved_ffn_hidden_dim
    assert feed_forward.w_down.in_features == config.resolved_ffn_hidden_dim
    assert feed_forward.w_down.out_features == config.hidden_dim


def test_dat_decoder_lm_can_use_normal_0_02_scaled_projection_initialization() -> None:
    torch.manual_seed(0)
    config = build_tiny_dat_config(
        hidden_dim=96,
        n_heads_sa=4,
        n_heads_ra=2,
        n_layers=2,
        ffn_activation="swiglu",
        init_scheme="normal_0_02_scaled_projection",
    )
    model = DatDecoderLM(config)
    layer = model.layers[0]
    base_std = 0.02
    # GPT-2-style base std, with projection weights depth-scaled by layer count.
    scaled_std = base_std / math.sqrt(2 * config.n_layers)

    assert model.lm_head.weight is model.token_embeddings.weight
    assert layer.sensory_attention.q_proj.weight.std().item() == pytest.approx(base_std, rel=0.15)
    # Projection weights such as o_proj and w_up use the depth-scaled std.
    assert layer.sensory_attention.o_proj.weight.std().item() == pytest.approx(scaled_std, rel=0.25)
    assert layer.feed_forward.w_up.weight.std().item() == pytest.approx(scaled_std, rel=0.25)
    # Non-projection weights keep the GPT-2-style base std.
    assert layer.feed_forward.w_down.weight.std().item() == pytest.approx(base_std, rel=0.15)


def test_dat_decoder_lm_can_use_xavier_uniform_initialization() -> None:
    torch.manual_seed(0)
    config = build_tiny_dat_config(init_scheme="xavier_uniform")
    model = DatDecoderLM(config)
    weights = model.token_embeddings.weight.detach()
    fan_in = config.hidden_dim
    fan_out = config.vocab_size
    expected_bound = math.sqrt(6.0 / (fan_in + fan_out))
    expected_std = expected_bound / math.sqrt(3.0)

    assert model.lm_head.weight is model.token_embeddings.weight
    assert weights.mean().item() == pytest.approx(0.0, abs=0.02)
    assert weights.std().item() == pytest.approx(expected_std, rel=0.12)
    assert weights.min().item() >= -expected_bound - 1e-6
    assert weights.max().item() <= expected_bound + 1e-6


@pytest.mark.parametrize("init_scheme", ["xavier_uniform", "normal_0_02_scaled_projection"])
def test_dat_decoder_lm_can_use_untied_lm_head(init_scheme: str) -> None:
    torch.manual_seed(0)
    config = build_tiny_dat_config(init_scheme=init_scheme, tie_lm_head=False)
    model = DatDecoderLM(config)

    assert model.lm_head.weight is not model.token_embeddings.weight
    assert model.lm_head.weight.shape == model.token_embeddings.weight.shape


def test_dat_decoder_block_retrieves_symbols_from_normed_states_when_pre_norm() -> None:
    torch.manual_seed(0)
    config = build_tiny_dat_config(norm_first=True, n_layers=1, dropout=0.0)
    block = DatDecoderBlock(config)
    retriever = RecordingSymbolRetriever(config.resolved_symbol_dim)
    x = torch.randn(2, 4, config.hidden_dim)
    mask = torch.tril(torch.ones(4, 4, dtype=torch.bool)).unsqueeze(0).expand(2, -1, -1)

    output = block(x, symbol_retriever=retriever, mask=mask, pos_info=None)

    assert output.shape == x.shape
    assert retriever.last_input is not None
    assert torch.allclose(retriever.last_input, block.norm1(x), atol=1e-6)
    assert not torch.allclose(retriever.last_input, x, atol=1e-6)


def test_dat_decoder_block_retrieves_symbols_from_raw_states_when_post_norm() -> None:
    torch.manual_seed(0)
    config = build_tiny_dat_config(norm_first=False, n_layers=1, dropout=0.0)
    block = DatDecoderBlock(config)
    retriever = RecordingSymbolRetriever(config.resolved_symbol_dim)
    x = torch.randn(2, 4, config.hidden_dim)
    mask = torch.tril(torch.ones(4, 4, dtype=torch.bool)).unsqueeze(0).expand(2, -1, -1)

    output = block(x, symbol_retriever=retriever, mask=mask, pos_info=None)

    assert output.shape == x.shape
    assert retriever.last_input is not None
    assert torch.allclose(retriever.last_input, x, atol=1e-6)


def test_dat_decoder_lm_forward_shape_and_loss() -> None:
    torch.manual_seed(0)
    model = DatDecoderLM(build_tiny_dat_config())
    input_ids = torch.tensor(
        [
            [1, 5, 6, 7, 2, 0],
            [1, 8, 9, 10, 11, 2],
        ],
        dtype=torch.long,
    )
    targets = torch.tensor(
        [
            [5, 6, 7, 2, -1, -1],
            [8, 9, 10, 11, 2, -1],
        ],
        dtype=torch.long,
    )
    attention_mask = input_ids != 0

    logits, loss = model(input_ids, targets=targets, attention_mask=attention_mask)

    assert logits.shape == (2, 6, 32)
    assert loss is not None
    assert torch.isfinite(loss)


def test_relative_symbol_retriever_supports_rope_relative_phases() -> None:
    x = torch.zeros(1, 4, 8)
    retriever = RelativePositionalSymbolRetriever(
        symbol_dim=8,
        max_rel_distance=1,
        rope=True,
    )

    symbols = retriever(x)

    assert symbols.shape == (4, 4, 8)
    assert list(retriever.parameters()) == []
    assert torch.allclose(symbols[0, 1], symbols[1, 2])
    assert torch.allclose(symbols[0, 0, 0::2], torch.ones(4))
    assert torch.allclose(symbols[0, 0, 1::2], torch.zeros(4))
    assert torch.allclose(symbols[0, 1], symbols[0, 2])
    assert not torch.allclose(symbols[0, 1], symbols[1, 0])
    assert not torch.allclose(symbols[0, 1, 0::2], torch.ones(4))
    assert not torch.allclose(symbols[0, 1, 1::2], torch.zeros(4))


def test_relative_symbol_retriever_caches_rope_relative_symbols() -> None:
    retriever = RelativePositionalSymbolRetriever(
        symbol_dim=8,
        max_rel_distance=1,
        rope=True,
    )
    x = torch.zeros(1, 4, 8)

    first_symbols = retriever(x)
    second_symbols = retriever(x)
    shorter_symbols = retriever(torch.zeros(1, 3, 8))
    shorter_symbols_again = retriever(torch.zeros(1, 3, 8))
    longer_symbols = retriever(torch.zeros(1, 6, 8))

    assert first_symbols.data_ptr() == second_symbols.data_ptr()
    assert shorter_symbols.data_ptr() == shorter_symbols_again.data_ptr()
    # Slicing the cache means the data_ptr (start of memory) is identical
    assert shorter_symbols.data_ptr() == first_symbols.data_ptr()
    # Growing past the cache causes reallocation
    assert longer_symbols.data_ptr() != first_symbols.data_ptr()
    assert "_rope_relative_cache" not in retriever.state_dict()


def test_dat_decoder_lm_forward_supports_rope_relative_symbols() -> None:
    torch.manual_seed(0)
    model = DatDecoderLM(
        build_tiny_dat_config(
            symbol_retrieval="relative",
            relative_symbols_rope=True,
            n_layers=1,
        )
    )
    input_ids = torch.tensor([[1, 5, 6, 7, 2]], dtype=torch.long)
    targets = torch.tensor([[5, 6, 7, 2, -1]], dtype=torch.long)

    logits, loss = model(input_ids, targets=targets)

    assert logits.shape == (1, 5, 32)
    assert loss is not None
    assert torch.isfinite(loss)


@pytest.mark.parametrize("symbol_retrieval", DSSL_SYMBOL_RETRIEVAL_TYPES)
@pytest.mark.parametrize("ra_type", DSSL_RA_TYPES)
def test_dat_decoder_lm_forward_shape_and_loss_for_dssl_variants(
    symbol_retrieval: str,
    ra_type: str,
) -> None:
    torch.manual_seed(0)
    model = DatDecoderLM(
        build_tiny_dat_config(
            symbol_retrieval=symbol_retrieval,
            ra_type=ra_type,
            n_layers=1,
        )
    )
    input_ids = torch.tensor([[1, 5, 6, 7, 2]], dtype=torch.long)
    targets = torch.tensor([[5, 6, 7, 2, -1]], dtype=torch.long)

    logits, loss = model(input_ids, targets=targets)

    assert logits.shape == (1, 5, 32)
    assert loss is not None
    assert torch.isfinite(loss)


@pytest.mark.parametrize("ra_rel_activation", DSSL_RA_ACTIVATIONS)
def test_dat_decoder_lm_forward_supports_dssl_relation_activations(
    ra_rel_activation: str,
) -> None:
    torch.manual_seed(0)
    model = DatDecoderLM(
        build_tiny_dat_config(
            ra_type="rca",
            ra_rel_activation=ra_rel_activation,
            n_layers=1,
        )
    )
    input_ids = torch.tensor([[1, 5, 6, 7, 2]], dtype=torch.long)

    logits, loss = model(input_ids, targets=input_ids)

    assert logits.shape == (1, 5, 32)
    assert loss is not None
    assert torch.isfinite(loss)


@pytest.mark.parametrize("ffn_activation", DSSL_FFN_ACTIVATIONS)
def test_dat_decoder_lm_forward_supports_dssl_ffn_activations(
    ffn_activation: str,
) -> None:
    torch.manual_seed(0)
    model = DatDecoderLM(
        build_tiny_dat_config(
            ffn_activation=ffn_activation,
            n_layers=1,
        )
    )
    input_ids = torch.tensor([[1, 5, 6, 7, 2]], dtype=torch.long)

    logits, loss = model(input_ids, targets=input_ids)

    assert logits.shape == (1, 5, 32)
    assert loss is not None
    assert torch.isfinite(loss)


def test_dat_decoder_lm_attention_mask_is_causal_and_padding_aware() -> None:
    model = DatDecoderLM(build_tiny_dat_config())
    input_ids = torch.tensor([[1, 5, 0]], dtype=torch.long)

    mask = model._build_attention_mask(input_ids)

    assert mask.shape == (1, 3, 3)
    assert mask[0].tolist() == [
        [True, False, False],
        [True, True, False],
        [False, False, False],
    ]


def test_dat_decoder_lm_attention_mask_resets_after_eos_by_default() -> None:
    model = DatDecoderLM(build_tiny_dat_config())
    input_ids = torch.tensor([[1, 5, 2, 6, 0]], dtype=torch.long)

    mask = model._build_attention_mask(input_ids)

    assert mask[0].tolist() == [
        [True, False, False, False, False],
        [True, True, False, False, False],
        [False, False, True, False, False],
        [False, False, True, True, False],
        [False, False, False, False, False],
    ]


def test_dat_decoder_lm_causal_mask_prevents_future_leakage() -> None:
    torch.manual_seed(0)
    model = DatDecoderLM(build_tiny_dat_config(dropout=0.0))
    model.eval()
    input_a = torch.tensor([[1, 5, 6, 7, 8, 9]], dtype=torch.long)
    input_b = torch.tensor([[1, 5, 6, 7, 10, 11]], dtype=torch.long)

    logits_a, _ = model(input_a)
    logits_b, _ = model(input_b)

    assert torch.allclose(logits_a[:, :4], logits_b[:, :4], atol=1e-6)


@pytest.mark.parametrize("symbol_retrieval", DSSL_SYMBOL_RETRIEVAL_TYPES)
@pytest.mark.parametrize("ra_type", DSSL_RA_TYPES)
def test_dat_decoder_lm_causal_mask_prevents_future_leakage_for_dssl_variants(
    symbol_retrieval: str,
    ra_type: str,
) -> None:
    torch.manual_seed(0)
    model = DatDecoderLM(
        build_tiny_dat_config(
            dropout=0.0,
            symbol_retrieval=symbol_retrieval,
            ra_type=ra_type,
            n_layers=1,
        )
    )
    model.eval()
    input_a = torch.tensor([[1, 5, 6, 7, 8, 9]], dtype=torch.long)
    input_b = torch.tensor([[1, 5, 6, 7, 10, 11]], dtype=torch.long)

    logits_a, _ = model(input_a)
    logits_b, _ = model(input_b)

    assert torch.allclose(logits_a[:, :4], logits_b[:, :4], atol=1e-6)


@pytest.mark.parametrize("ra_type", DSSL_RA_TYPES)
def test_dat_decoder_lm_softmax_relation_activation_stays_causal(ra_type: str) -> None:
    torch.manual_seed(0)
    model = DatDecoderLM(
        build_tiny_dat_config(
            dropout=0.0,
            symbol_retrieval="symbolic",
            ra_type=ra_type,
            ra_rel_activation="softmax",
            n_layers=1,
        )
    )
    model.eval()
    input_a = torch.tensor([[1, 5, 6, 7, 8, 9]], dtype=torch.long)
    input_b = torch.tensor([[1, 5, 6, 7, 10, 11]], dtype=torch.long)

    logits_a, _ = model(input_a)
    logits_b, _ = model(input_b)

    assert torch.allclose(logits_a[:, :4], logits_b[:, :4], atol=1e-6)


def test_dat_decoder_lm_generate_stops_on_eos() -> None:
    token_embeddings = torch.zeros(8, 8)
    token_embeddings[1, 0] = 1.0
    token_embeddings[2, 0] = 2.0
    model = build_controlled_dat_generation_model(token_embeddings)

    generated = model.generate(
        torch.tensor([[1]], dtype=torch.long),
        max_new_tokens=5,
        do_sample=False,
    )

    assert generated.tolist() == [[1, 2]]
