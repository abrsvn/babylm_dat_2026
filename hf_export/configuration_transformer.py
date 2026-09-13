"""HuggingFace PretrainedConfig for the self-attention decoder LM."""

from transformers import PretrainedConfig

TRANSFORMER_LM_FIELDS = (
    "vocab_size",
    "max_seq_len",
    "pe_type",
    "hidden_dim",
    "n_heads",
    "n_layers",
    "dropout",
    "dff_factor",
    "rope_theta",
    "max_rel_pos",
    "sequence_boundary_policy",
    "init_range",
    "norm_type",
    "norm_first",
    "use_bias_qkv",
    "use_bias_out",
    "use_bias_ffn",
    "tie_lm_head",
    "pad_token_id",
    "bos_token_id",
    "eos_token_id",
)

class TransformerConfig(PretrainedConfig):
    model_type = "transformer"
    attribute_map = {"hidden_size": "hidden_dim"}

    def __init__(
        self,
        vocab_size: int = 16384,
        max_seq_len: int = 514,
        pe_type: str = "rope",
        hidden_dim: int = 192,
        n_heads: int = 6,
        n_layers: int = 4,
        dropout: float = 0.0,
        dff_factor: int = 4,
        rope_theta: float = 10000.0,
        max_rel_pos: int | None = None,
        sequence_boundary_policy: str = "eos_document",
        init_range: float = 0.15,
        norm_type: str = "rmsnorm",
        norm_first: bool = True,
        use_bias_qkv: bool = False,
        use_bias_out: bool = True,
        use_bias_ffn: bool = True,
        tie_lm_head: bool = True,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        **kwargs,
    ) -> None:
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.pe_type = pe_type
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.dff_factor = dff_factor
        self.rope_theta = rope_theta
        self.max_rel_pos = max_rel_pos
        self.sequence_boundary_policy = sequence_boundary_policy
        self.init_range = init_range
        self.norm_type = norm_type
        self.norm_first = norm_first
        self.use_bias_qkv = use_bias_qkv
        self.use_bias_out = use_bias_out
        self.use_bias_ffn = use_bias_ffn
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
        self.tie_lm_head = tie_lm_head
