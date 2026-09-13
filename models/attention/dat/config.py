"""Configuration for the dual-attention decoder LM."""

from dataclasses import dataclass

from models.attention.config import SUPPORTED_SEQUENCE_BOUNDARY_POLICIES


SUPPORTED_PE_TYPES = {"rope"}
SUPPORTED_SYMBOL_RETRIEVAL = {"symbolic", "positional", "relative", "relsymbolic"}
SUPPORTED_RA_TYPES = {"ra", "rca", "disrca"}
SUPPORTED_RA_ACTIVATIONS = {"softmax", "identity", "tanh", "sigmoid"}
SUPPORTED_FFN_ACTIVATIONS = {"gelu", "swiglu"}
SUPPORTED_FFN_HIDDEN_DIM_MODES = {"dff_factor"}
SUPPORTED_INIT_SCHEMES = {"xavier_uniform", "normal_0_02_scaled_projection"}


@dataclass(frozen=True)
class DatLMConfig:
    vocab_size: int
    max_seq_len: int
    pe_type: str = "rope"
    hidden_dim: int = 256
    n_heads_sa: int = 2
    n_heads_ra: int = 2
    n_layers: int = 4
    dropout: float = 0.0
    dff_factor: int = 4
    ffn_hidden_dim_mode: str = "dff_factor"
    ffn_activation: str = "gelu"
    rope_theta: float = 10000.0
    max_rel_pos: int | None = None
    sequence_boundary_policy: str = "eos_document"
    init_range: float = 0.15
    init_scheme: str = "xavier_uniform"
    norm_type: str = "rmsnorm"
    norm_first: bool = True
    use_bias_qkv: bool = False
    use_bias_out: bool = True
    use_bias_ffn: bool = True
    tie_lm_head: bool = True
    symbol_dim: int | None = None
    n_symbols: int | None = None
    symbolic_attn_n_heads: int | None = None
    symbol_retrieval: str = "symbolic"
    symbolic_use_bias: bool = False
    positional_symbols_sinusoidal: bool = False
    relative_symbols_rope: bool = False
    relsymbolic_rel_n_heads: int = 4
    relsymbolic_symbolic_attn_n_heads: int = 4
    relsymbolic_neighborhood_size: int = 2
    relsymbolic_include_self: bool = False
    relsymbolic_normalize_rels: bool = True
    relsymbolic_trainable_symbols: bool = True
    relsymbolic_dropout: float = 0.0
    relsymbolic_rel_scale: float | None = None
    relsymbolic_symbolic_attn_scale: float | None = None
    relsymbolic_use_bias: bool = False
    ra_type: str = "ra"
    ra_n_relations: int | None = None
    ra_rel_activation: str = "identity"
    ra_symmetric_rels: bool = False
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {self.max_seq_len}")
        if self.pe_type not in SUPPORTED_PE_TYPES:
            raise ValueError(f"Unsupported pe_type: {self.pe_type}")
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}")
        if self.n_heads_sa <= 0:
            raise ValueError(f"n_heads_sa must be positive for DAT, got {self.n_heads_sa}")
        if self.n_heads_ra <= 0:
            raise ValueError(f"n_heads_ra must be positive for DAT, got {self.n_heads_ra}")
        total_heads = self.total_n_heads
        if self.hidden_dim % total_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by total DAT heads "
                f"({total_heads} = {self.n_heads_sa} SA + {self.n_heads_ra} RA)"
            )
        if self.n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {self.n_layers}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0.0, 1.0), got {self.dropout}")
        if self.dff_factor <= 0:
            raise ValueError(f"dff_factor must be positive, got {self.dff_factor}")
        if self.ffn_hidden_dim_mode not in SUPPORTED_FFN_HIDDEN_DIM_MODES:
            raise ValueError(f"Unsupported ffn_hidden_dim_mode for DAT: {self.ffn_hidden_dim_mode}")
        if self.ffn_activation not in SUPPORTED_FFN_ACTIVATIONS:
            raise ValueError(f"Unsupported ffn_activation for DAT: {self.ffn_activation}")
        if self.rope_theta <= 0.0:
            raise ValueError(f"rope_theta must be positive, got {self.rope_theta}")
        if self.max_rel_pos is not None and self.max_rel_pos <= 0:
            raise ValueError(f"max_rel_pos must be positive when provided, got {self.max_rel_pos}")
        if self.sequence_boundary_policy not in SUPPORTED_SEQUENCE_BOUNDARY_POLICIES:
            raise ValueError(
                "Unsupported sequence_boundary_policy for DAT: "
                f"{self.sequence_boundary_policy}"
            )
        if self.init_range <= 0.0:
            raise ValueError(f"init_range must be positive, got {self.init_range}")
        if self.init_scheme not in SUPPORTED_INIT_SCHEMES:
            raise ValueError(f"Unsupported init_scheme for DAT: {self.init_scheme}")
        if self.norm_type not in {"layernorm", "rmsnorm"}:
            raise ValueError(
                f"norm_type must be 'layernorm' or 'rmsnorm', got {self.norm_type}"
            )
        if self.head_dim % 2 != 0:
            raise ValueError(
                "RoPE requires even DAT head_dim, got "
                f"{self.head_dim} from hidden_dim={self.hidden_dim}, total_heads={total_heads}"
            )
        if self.symbol_dim is not None and self.symbol_dim <= 0:
            raise ValueError(f"symbol_dim must be positive when provided, got {self.symbol_dim}")
        if self.symbolic_attn_n_heads is not None and self.symbolic_attn_n_heads <= 0:
            raise ValueError(
                "symbolic_attn_n_heads must be positive when provided, "
                f"got {self.symbolic_attn_n_heads}"
            )
        if self.symbol_retrieval not in SUPPORTED_SYMBOL_RETRIEVAL:
            raise ValueError(f"Unsupported symbol_retrieval for DAT: {self.symbol_retrieval}")
        if self.symbol_retrieval == "symbolic":
            symbolic_heads = self.resolved_symbolic_attn_n_heads
            if self.hidden_dim % symbolic_heads != 0:
                raise ValueError(
                    f"hidden_dim ({self.hidden_dim}) must be divisible by symbolic_attn_n_heads "
                    f"({symbolic_heads}) for symbolic retrieval"
                )
            if self.resolved_symbol_dim % symbolic_heads != 0:
                raise ValueError(
                    f"symbol_dim ({self.resolved_symbol_dim}) must be divisible by symbolic_attn_n_heads "
                    f"({symbolic_heads}) for symbolic retrieval"
                )
        if (
            self.symbol_retrieval == "positional"
            and self.positional_symbols_sinusoidal
            and self.resolved_symbol_dim % 2 != 0
        ):
            raise ValueError(
                "Sinusoidal positional symbols require even symbol_dim, "
                f"got {self.resolved_symbol_dim}"
            )
        if self.relative_symbols_rope:
            if self.symbol_retrieval != "relative":
                raise ValueError(
                    "relative_symbols_rope=True requires symbol_retrieval='relative', "
                    f"got {self.symbol_retrieval!r}"
                )
            if self.resolved_symbol_dim % 2 != 0:
                raise ValueError(
                    "RoPE relative symbols require even symbol_dim, "
                    f"got {self.resolved_symbol_dim}"
                )
        if self.positional_symbols_sinusoidal and self.symbol_retrieval != "positional":
            raise ValueError(
                "positional_symbols_sinusoidal=True requires symbol_retrieval='positional', "
                f"got {self.symbol_retrieval!r}"
            )
        if self.resolved_n_symbols <= 0:
            raise ValueError(f"resolved_n_symbols must be positive, got {self.resolved_n_symbols}")
        if self.symbol_retrieval == "relsymbolic":
            if self.relsymbolic_rel_n_heads <= 0:
                raise ValueError(
                    f"relsymbolic_rel_n_heads must be positive, got {self.relsymbolic_rel_n_heads}"
                )
            if self.hidden_dim % self.relsymbolic_rel_n_heads != 0:
                raise ValueError(
                    f"hidden_dim ({self.hidden_dim}) must be divisible by "
                    f"relsymbolic_rel_n_heads ({self.relsymbolic_rel_n_heads})"
                )
            if self.relsymbolic_symbolic_attn_n_heads <= 0:
                raise ValueError(
                    "relsymbolic_symbolic_attn_n_heads must be positive, "
                    f"got {self.relsymbolic_symbolic_attn_n_heads}"
                )
            if self.hidden_dim % self.relsymbolic_symbolic_attn_n_heads != 0:
                raise ValueError(
                    f"hidden_dim ({self.hidden_dim}) must be divisible by "
                    f"relsymbolic_symbolic_attn_n_heads ({self.relsymbolic_symbolic_attn_n_heads})"
                )
            if self.resolved_symbol_dim % self.relsymbolic_symbolic_attn_n_heads != 0:
                raise ValueError(
                    f"symbol_dim ({self.resolved_symbol_dim}) must be divisible by "
                    f"relsymbolic_symbolic_attn_n_heads ({self.relsymbolic_symbolic_attn_n_heads})"
                )
            if self.relsymbolic_neighborhood_size <= 0:
                raise ValueError(
                    "relsymbolic_neighborhood_size must be positive, "
                    f"got {self.relsymbolic_neighborhood_size}"
                )
            if not 0.0 <= self.relsymbolic_dropout < 1.0:
                raise ValueError(
                    f"relsymbolic_dropout must be in [0.0, 1.0), got {self.relsymbolic_dropout}"
                )
            if self.relsymbolic_rel_scale is not None and self.relsymbolic_rel_scale <= 0.0:
                raise ValueError(
                    "relsymbolic_rel_scale must be positive when provided, "
                    f"got {self.relsymbolic_rel_scale}"
                )
            if (
                self.relsymbolic_symbolic_attn_scale is not None
                and self.relsymbolic_symbolic_attn_scale <= 0.0
            ):
                raise ValueError(
                    "relsymbolic_symbolic_attn_scale must be positive when provided, "
                    f"got {self.relsymbolic_symbolic_attn_scale}"
                )
        if self.ra_type not in SUPPORTED_RA_TYPES:
            raise ValueError(f"Unsupported ra_type for DAT: {self.ra_type}")
        if self.ra_n_relations is not None and self.ra_n_relations <= 0:
            raise ValueError(
                f"ra_n_relations must be positive when provided, got {self.ra_n_relations}"
            )
        if self.ra_type != "ra" and self.ra_n_relations is not None:
            raise ValueError(f"ra_n_relations applies only to ra_type='ra', got {self.ra_type}")
        if self.ra_type != "ra" and self.ra_symmetric_rels:
            raise ValueError(f"ra_symmetric_rels applies only to ra_type='ra', got {self.ra_type}")
        n_relations = self.resolved_ra_n_relations
        if self.ra_type == "ra" and (self.head_dim * self.n_heads_ra) % n_relations != 0:
            raise ValueError(
                f"head_dim * n_heads_ra ({self.head_dim * self.n_heads_ra}) must be "
                f"divisible by ra_n_relations ({n_relations})"
            )
        if self.ra_rel_activation not in SUPPORTED_RA_ACTIVATIONS:
            raise ValueError(
                f"Unsupported ra_rel_activation for DAT: {self.ra_rel_activation}"
            )

    @property
    def total_n_heads(self) -> int:
        return self.n_heads_sa + self.n_heads_ra

    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.total_n_heads

    @property
    def resolved_symbol_dim(self) -> int:
        return self.hidden_dim if self.symbol_dim is None else self.symbol_dim

    @property
    def resolved_symbolic_attn_n_heads(self) -> int:
        return self.total_n_heads if self.symbolic_attn_n_heads is None else self.symbolic_attn_n_heads

    @property
    def resolved_ffn_hidden_dim(self) -> int:
        return self.hidden_dim * self.dff_factor

    @property
    def resolved_n_symbols(self) -> int:
        return self.max_seq_len if self.n_symbols is None else self.n_symbols

    @property
    def resolved_ra_n_relations(self) -> int:
        return self.n_heads_ra if self.ra_n_relations is None else self.ra_n_relations

    @property
    def effective_depth(self) -> int:
        """Total effective block depth."""
        return self.n_layers
