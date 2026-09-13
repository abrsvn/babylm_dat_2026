"""Configuration for the Relational BabyLM self-attention decoder LM."""

from dataclasses import dataclass

from models.attention.config import SUPPORTED_SEQUENCE_BOUNDARY_POLICIES


@dataclass(frozen=True)
class SelfAttentionLMConfig:
    vocab_size: int
    max_seq_len: int
    pe_type: str = "rope"
    hidden_dim: int = 192
    n_heads: int = 6
    n_layers: int = 4
    dropout: float = 0.0
    dff_factor: int = 4
    rope_theta: float = 10000.0
    max_rel_pos: int | None = None
    sequence_boundary_policy: str = "eos_document"
    init_range: float = 0.15
    norm_type: str = "rmsnorm"
    norm_first: bool = True
    use_bias_qkv: bool = False
    use_bias_out: bool = True
    use_bias_ffn: bool = True
    tie_lm_head: bool = True
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {self.max_seq_len}")
        if self.pe_type not in {"rope"}:
            raise ValueError(f"Unsupported pe_type: {self.pe_type}")
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}")
        if self.n_heads <= 0:
            raise ValueError(f"n_heads must be positive, got {self.n_heads}")
        if self.hidden_dim % self.n_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {self.n_layers}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0.0, 1.0), got {self.dropout}")
        if self.dff_factor <= 0:
            raise ValueError(f"dff_factor must be positive, got {self.dff_factor}")
        if self.rope_theta <= 0.0:
            raise ValueError(f"rope_theta must be positive, got {self.rope_theta}")
        if self.max_rel_pos is not None and self.max_rel_pos <= 0:
            raise ValueError(f"max_rel_pos must be positive when provided, got {self.max_rel_pos}")
        if self.sequence_boundary_policy not in SUPPORTED_SEQUENCE_BOUNDARY_POLICIES:
            raise ValueError(
                "Unsupported sequence_boundary_policy: "
                f"{self.sequence_boundary_policy}"
            )
        if self.init_range <= 0.0:
            raise ValueError(f"init_range must be positive, got {self.init_range}")
        if self.norm_type not in {"layernorm", "rmsnorm"}:
            raise ValueError(
                f"norm_type must be 'layernorm' or 'rmsnorm', got {self.norm_type}"
            )
        if (self.hidden_dim // self.n_heads) % 2 != 0:
            raise ValueError(
                "RoPE requires even head_dim, got "
                f"{self.hidden_dim // self.n_heads} from hidden_dim={self.hidden_dim}, n_heads={self.n_heads}"
            )
