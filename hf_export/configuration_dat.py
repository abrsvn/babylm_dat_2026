"""HuggingFace PretrainedConfig for the dual-attention (DAT) decoder LM.

This is the source-of-truth copy. scripts/convert_dat_to_hf.py copies it into a
generated HF repository (alongside modeling_dat.py and the flattened model
source) so the model can be loaded with
``AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)``.

The config simply carries every field of models.attention.dat.config.DatLMConfig
so that modeling_dat.py can rebuild the exact DatLMConfig at load time.
"""

from transformers import PretrainedConfig


# Every field needed to rebuild DatLMConfig. New exported fields are appended
# after existing constructor parameters to preserve positional compatibility.
# modeling_dat.py rebuilds DatLMConfig as
# DatLMConfig(**{f: getattr(config, f) for f in DAT_LM_FIELDS}).
DAT_LM_FIELDS = (
    "vocab_size",
    "max_seq_len",
    "pe_type",
    "hidden_dim",
    "n_heads_sa",
    "n_heads_ra",
    "n_layers",
    "dropout",
    "dff_factor",
    "ffn_hidden_dim_mode",
    "ffn_activation",
    "rope_theta",
    "max_rel_pos",
    "init_range",
    "init_scheme",
    "norm_type",
    "norm_first",
    "use_bias_qkv",
    "use_bias_out",
    "use_bias_ffn",
    "tie_lm_head",
    "symbol_dim",
    "n_symbols",
    "symbolic_attn_n_heads",
    "symbol_retrieval",
    "symbolic_use_bias",
    "positional_symbols_sinusoidal",
    "relative_symbols_rope",
    "relsymbolic_rel_n_heads",
    "relsymbolic_symbolic_attn_n_heads",
    "relsymbolic_neighborhood_size",
    "relsymbolic_include_self",
    "relsymbolic_normalize_rels",
    "relsymbolic_trainable_symbols",
    "relsymbolic_dropout",
    "relsymbolic_rel_scale",
    "relsymbolic_symbolic_attn_scale",
    "relsymbolic_use_bias",
    "ra_type",
    "ra_n_relations",
    "ra_rel_activation",
    "ra_symmetric_rels",
    "sequence_boundary_policy",
    "pad_token_id",
    "bos_token_id",
    "eos_token_id",
)


class DatConfig(PretrainedConfig):
    model_type = "dat"
    # Expose the model width under HF's conventional name so generic
    # evaluation harnesses that read `config.hidden_size` can size their
    # classification head. The model's own
    # field is hidden_dim; this maps the alias onto it for both get and set.
    attribute_map = {"hidden_size": "hidden_dim"}

    def __init__(
        self,
        vocab_size: int = 16384,
        max_seq_len: int = 513,
        pe_type: str = "rope",
        hidden_dim: int = 256,
        n_heads_sa: int = 2,
        n_heads_ra: int = 2,
        n_layers: int = 4,
        dropout: float = 0.0,
        dff_factor: int = 4,
        ffn_hidden_dim_mode: str = "dff_factor",
        ffn_activation: str = "gelu",
        rope_theta: float = 10000.0,
        max_rel_pos: int | None = None,
        init_range: float = 0.15,
        init_scheme: str = "xavier_uniform",
        norm_type: str = "rmsnorm",
        norm_first: bool = True,
        use_bias_qkv: bool = False,
        use_bias_out: bool = True,
        use_bias_ffn: bool = True,
        tie_lm_head: bool = True,
        symbol_dim: int | None = None,
        n_symbols: int | None = None,
        symbolic_attn_n_heads: int | None = None,
        symbol_retrieval: str = "symbolic",
        symbolic_use_bias: bool = False,
        positional_symbols_sinusoidal: bool = False,
        relative_symbols_rope: bool = False,
        relsymbolic_rel_n_heads: int = 4,
        relsymbolic_symbolic_attn_n_heads: int = 4,
        relsymbolic_neighborhood_size: int = 2,
        relsymbolic_include_self: bool = False,
        relsymbolic_normalize_rels: bool = True,
        relsymbolic_trainable_symbols: bool = True,
        relsymbolic_dropout: float = 0.0,
        relsymbolic_rel_scale: float | None = None,
        relsymbolic_symbolic_attn_scale: float | None = None,
        relsymbolic_use_bias: bool = False,
        ra_type: str = "ra",
        ra_n_relations: int | None = None,
        ra_rel_activation: str = "identity",
        ra_symmetric_rels: bool = False,
        sequence_boundary_policy: str = "eos_document",
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        **kwargs,
    ) -> None:
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.pe_type = pe_type
        self.hidden_dim = hidden_dim
        self.n_heads_sa = n_heads_sa
        self.n_heads_ra = n_heads_ra
        self.n_layers = n_layers
        self.dropout = dropout
        self.dff_factor = dff_factor
        self.ffn_hidden_dim_mode = ffn_hidden_dim_mode
        self.ffn_activation = ffn_activation
        self.rope_theta = rope_theta
        self.max_rel_pos = max_rel_pos
        self.init_range = init_range
        self.init_scheme = init_scheme
        self.norm_type = norm_type
        self.norm_first = norm_first
        self.use_bias_qkv = use_bias_qkv
        self.use_bias_out = use_bias_out
        self.use_bias_ffn = use_bias_ffn
        self.tie_lm_head = tie_lm_head
        self.symbol_dim = symbol_dim
        self.n_symbols = n_symbols
        self.symbolic_attn_n_heads = symbolic_attn_n_heads
        self.symbol_retrieval = symbol_retrieval
        self.symbolic_use_bias = symbolic_use_bias
        self.positional_symbols_sinusoidal = positional_symbols_sinusoidal
        self.relative_symbols_rope = relative_symbols_rope
        self.relsymbolic_rel_n_heads = relsymbolic_rel_n_heads
        self.relsymbolic_symbolic_attn_n_heads = relsymbolic_symbolic_attn_n_heads
        self.relsymbolic_neighborhood_size = relsymbolic_neighborhood_size
        self.relsymbolic_include_self = relsymbolic_include_self
        self.relsymbolic_normalize_rels = relsymbolic_normalize_rels
        self.relsymbolic_trainable_symbols = relsymbolic_trainable_symbols
        self.relsymbolic_dropout = relsymbolic_dropout
        self.relsymbolic_rel_scale = relsymbolic_rel_scale
        self.relsymbolic_symbolic_attn_scale = relsymbolic_symbolic_attn_scale
        self.relsymbolic_use_bias = relsymbolic_use_bias
        self.ra_type = ra_type
        self.ra_n_relations = ra_n_relations
        self.ra_rel_activation = ra_rel_activation
        self.ra_symmetric_rels = ra_symmetric_rels
        self.sequence_boundary_policy = sequence_boundary_policy
        # HF base also exposes max_position_embeddings for generation utilities.
        self.max_position_embeddings = max_seq_len
        # tie_lm_head is the source of truth for weight tying; keep HF's
        # tie_word_embeddings in lockstep so from_pretrained never re-ties an
        # untied head (or fails to tie a tied one). Drop any value coming in via
        # kwargs (e.g. a serialized config.json) so the two cannot disagree.
        kwargs.pop("tie_word_embeddings", None)
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_lm_head,
            **kwargs,
        )
