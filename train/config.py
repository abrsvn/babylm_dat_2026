"""Training runtime config and CLI parsing."""

import argparse
import math
from dataclasses import asdict, dataclass


AUTO_ASSET_PATH = "<auto>"
MAX_NUMPY_SEED = (2**32) - 1

# BabyLM training tracks represented by this release.
SMALL_CORPUS_IDS = frozenset({"strict_small"})
SUPPORTED_CORPUS_IDS = frozenset({"strict", "strict_small"})


@dataclass(frozen=True)
class TrainConfig:
    model_type: str = "dat"
    pe_type: str = "rope"
    hidden_dim: int = 192
    n_heads: int = 6
    n_heads_sa: int = 4
    n_heads_ra: int = 2
    n_layers: int = 4
    dropout: float = 0.0
    norm_type: str = "rmsnorm"
    use_bias_qkv: bool = False
    use_bias_out: bool = True
    use_bias_ffn: bool = True
    ffn_hidden_dim_mode: str = "dff_factor"
    ffn_activation: str = "swiglu"
    rope_theta: float = 10000.0
    max_rel_pos: int | None = None
    sequence_boundary_policy: str = "eos_document"
    symbol_dim: int | None = None
    n_symbols: int | None = None
    symbolic_attn_n_heads: int | None = None
    symbol_retrieval: str = "relative"
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
    ra_type: str = "rca"
    ra_n_relations: int | None = None
    ra_rel_activation: str = "identity"
    ra_symmetric_rels: bool = False
    datapoint_length: int = 1280
    corpus_id: str = "strict_small"
    n_epochs: int = 1
    batch_size: int = 16
    # Per-epoch cap on training steps (batches). None runs the full epoch.
    # Used for smoke tests and debugging; not set in any committed experiment
    # script. The scheduler's per-epoch optimizer-step count is clamped to
    # max_train_steps (divided by gradient_accumulation_steps), so the LR
    # schedule matches the actual training length within each epoch.
    max_train_steps: int | None = None
    # Global optimizer-step limit. None runs all epochs. When set, all ranks
    # stop at this optimizer step and the scheduler/preflight derive from it.
    # The progress checkpoint schedule is uniform across the run.
    max_optimizer_steps: int | None = None
    learning_rate: float = 0.001
    optimizer: str = "adamw"
    weight_decay: float = 0.0
    optimizer_beta1: float = 0.9
    optimizer_beta2: float = 0.999
    optimizer_eps: float = 1e-8
    muon_lr: float = 0.02
    muon_momentum: float = 0.95
    muon_n_schulz_steps: int = 5
    muon_aux_optimizer: str = "lambw"
    warmup_ratio: float = 0.01
    lr_scheduler_type: str = "cosine_min_lr"
    min_lr_rate: float = 0.1
    nextlat_enabled: bool = False
    nextlat_lambda_mse: float = 1.0
    nextlat_lambda_kl: float = 1.0
    nextlat_lambda_ce: float = 0.0
    nextlat_horizon: int = 1
    nextlat_proj_factor: float = 1.0
    # Tie the LM head weights to the token embedding table. None derives the
    # value from the objective: tied for NTP, untied for NextLat. An explicit
    # value overrides the derivation (untied-NTP runs set False).
    tie_lm_head: bool | None = None
    init_scheme: str = "xavier_uniform"
    gradient_clip_norm: float = 1.0
    gradient_accumulation_steps: int = 2
    ema_decay: float | None = None
    ema_device: str = "auto"
    # Warm-start all active model state from an existing checkpoint directory
    # for a second-stage training run (e.g. a curriculum phase). The main model
    # may use EMA weights; NextLat dynamics use their separately saved raw state.
    # Optimizer and scheduler are not restored. None means a cold start.
    init_from_checkpoint: str | None = None
    init_from_use_ema: bool = False
    cuda_autocast: bool = True
    seed: int = 0
    tokenizer_dir: str = AUTO_ASSET_PATH
    train_data_dir: str = AUTO_ASSET_PATH
    cache_dir: str = "cache/train"
    base_folder: str = "experiments"
    experiment_name: str = "testing"
    use_wandb: bool = False
    wandb_project_name: str = "babylm"
    wandb_experiment_name: str = "relational-babylm"

    def __post_init__(self) -> None:
        if self.model_type not in {"self_attention", "dat"}:
            raise ValueError(f"Unsupported model_type: {self.model_type}")
        # Model-architecture fields (pe_type, norm_type, hidden_dim, heads,
        # layers, dropout, symbol/relational/FFN settings, and the sequence
        # boundary policy) are validated at the model-config boundary
        # (DatLMConfig / SelfAttentionLMConfig), which every construction
        # path funnels through; CLI choices still guard the argparse boundary.
        if self.corpus_id not in SUPPORTED_CORPUS_IDS:
            raise ValueError(f"Unsupported corpus_id: {self.corpus_id}")
        if self.datapoint_length <= 0:
            raise ValueError(f"datapoint_length must be positive, got {self.datapoint_length}")
        if self.n_epochs <= 0:
            raise ValueError(f"n_epochs must be positive, got {self.n_epochs}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.max_train_steps is not None and self.max_train_steps <= 0:
            raise ValueError(
                f"max_train_steps must be positive when provided, got {self.max_train_steps}"
            )
        if self.max_optimizer_steps is not None and self.max_optimizer_steps <= 0:
            raise ValueError(
                f"max_optimizer_steps must be positive when provided, got {self.max_optimizer_steps}"
            )
        if self.learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.weight_decay < 0.0:
            raise ValueError(f"weight_decay must be non-negative, got {self.weight_decay}")
        if self.optimizer not in {"adamw", "muon"}:
            raise ValueError(f"Unsupported optimizer: {self.optimizer}")
        if self.muon_aux_optimizer not in {"lambw"}:
            raise ValueError(f"Unsupported muon_aux_optimizer: {self.muon_aux_optimizer}")
        if not math.isfinite(self.muon_lr) or self.muon_lr <= 0.0:
            raise ValueError(f"muon_lr must be a positive finite value, got {self.muon_lr}")
        if not 0.0 < self.muon_momentum < 1.0:
            raise ValueError(f"muon_momentum must be in (0.0, 1.0), got {self.muon_momentum}")
        if not isinstance(self.muon_n_schulz_steps, int) or isinstance(self.muon_n_schulz_steps, bool) or self.muon_n_schulz_steps <= 0:
            raise ValueError(f"muon_n_schulz_steps must be a positive integer, got {self.muon_n_schulz_steps}")
        if not 0.0 < self.optimizer_beta1 < 1.0:
            raise ValueError(f"optimizer_beta1 must be in (0.0, 1.0), got {self.optimizer_beta1}")
        if not 0.0 < self.optimizer_beta2 < 1.0:
            raise ValueError(f"optimizer_beta2 must be in (0.0, 1.0), got {self.optimizer_beta2}")
        if not math.isfinite(self.optimizer_eps) or self.optimizer_eps <= 0.0:
            raise ValueError(f"optimizer_eps must be a positive finite value, got {self.optimizer_eps}")
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError(f"warmup_ratio must be in [0.0, 1.0), got {self.warmup_ratio}")
        if self.lr_scheduler_type not in {"cosine_min_lr"}:
            raise ValueError(f"Unsupported lr_scheduler_type: {self.lr_scheduler_type}")
        if not 0.0 <= self.min_lr_rate < 1.0:
            raise ValueError(f"min_lr_rate must be in [0.0, 1.0), got {self.min_lr_rate}")
        if self.min_lr_rate <= 0.0:
            raise ValueError(
                "lr_scheduler_type='cosine_min_lr' requires min_lr_rate > 0.0"
            )
        if self.nextlat_lambda_mse < 0.0:
            raise ValueError(f"nextlat_lambda_mse must be non-negative, got {self.nextlat_lambda_mse}")
        if self.nextlat_lambda_kl < 0.0:
            raise ValueError(f"nextlat_lambda_kl must be non-negative, got {self.nextlat_lambda_kl}")
        if self.nextlat_lambda_ce < 0.0:
            raise ValueError(f"nextlat_lambda_ce must be non-negative, got {self.nextlat_lambda_ce}")
        if self.nextlat_horizon <= 0:
            raise ValueError(f"nextlat_horizon must be positive, got {self.nextlat_horizon}")
        if self.nextlat_horizon >= self.datapoint_length + 1:
            raise ValueError(
                "nextlat_horizon must be smaller than the model training sequence length, "
                f"got horizon={self.nextlat_horizon}, datapoint_length={self.datapoint_length}"
            )
        if self.nextlat_proj_factor <= 0.0:
            raise ValueError(f"nextlat_proj_factor must be positive, got {self.nextlat_proj_factor}")
        if self.nextlat_enabled:
            if self.nextlat_lambda_mse == 0.0 and self.nextlat_lambda_kl == 0.0 and self.nextlat_lambda_ce == 0.0:
                raise ValueError(
                    "At least one NextLat auxiliary coefficient must be positive when nextlat_enabled=True."
                )
        if self.init_scheme not in {"xavier_uniform", "normal_0_02_scaled_projection"}:
            raise ValueError(f"Unsupported init_scheme: {self.init_scheme}")
        if self.gradient_clip_norm != -1.0 and self.gradient_clip_norm <= 0.0:
            raise ValueError(
                "gradient_clip_norm must be -1.0 or positive, "
                f"got {self.gradient_clip_norm}"
            )
        if self.gradient_accumulation_steps <= 0:
            raise ValueError(
                "gradient_accumulation_steps must be positive, "
                f"got {self.gradient_accumulation_steps}"
            )
        if self.ema_decay is not None and not 0.0 < self.ema_decay < 1.0:
            raise ValueError(
                f"ema_decay must be None or in the open interval (0, 1), got {self.ema_decay}"
            )
        if self.init_from_use_ema and self.init_from_checkpoint is None:
            raise ValueError(
                "init_from_use_ema=True requires init_from_checkpoint to be set "
                "(the checkpoint directory to warm-start EMA weights from)."
            )
        if self.seed != -1 and not 0 <= self.seed <= MAX_NUMPY_SEED:
            raise ValueError(
                f"seed must be -1 or an integer in [0, {MAX_NUMPY_SEED}], got {self.seed}"
            )
        if not self.tokenizer_dir:
            raise ValueError("tokenizer_dir must be non-empty")
        if not self.train_data_dir:
            raise ValueError("train_data_dir must be non-empty")
        if not self.cache_dir:
            raise ValueError("cache_dir must be non-empty")
        if not self.experiment_name:
            raise ValueError("experiment_name must be non-empty")
        if not self.base_folder:
            raise ValueError("base_folder must be non-empty")


def _parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    raise argparse.ArgumentTypeError(f"expected 'true' or 'false', got {value!r}")


def build_argument_parser() -> argparse.ArgumentParser:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description="Train Relational BabyLM language models.")

    parser.add_argument(
        "--model_type",
        type=str,
        choices=["self_attention", "dat"],
        default=defaults.model_type,
        help="Model backend to train.",
    )
    parser.add_argument(
        "--pe_type",
        type=str,
        choices=["rope"],
        default=defaults.pe_type,
        help="Positional encoding type; RoPE rotates attention queries and keys.",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=defaults.hidden_dim,
        help="Transformer hidden dimension.",
    )
    parser.add_argument(
        "--n_heads",
        type=int,
        default=defaults.n_heads,
        help="Number of self-attention heads for model_type=self_attention.",
    )
    parser.add_argument(
        "--n_heads_sa",
        type=int,
        default=defaults.n_heads_sa,
        help="Number of sensory attention heads for model_type=dat.",
    )
    parser.add_argument(
        "--n_heads_ra",
        type=int,
        default=defaults.n_heads_ra,
        help="Number of relational attention heads for model_type=dat.",
    )
    parser.add_argument(
        "--n_layers",
        type=int,
        default=defaults.n_layers,
        help="Number of decoder blocks.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=defaults.dropout,
        help="Model dropout probability.",
    )
    parser.add_argument(
        "--norm_type",
        type=str,
        choices=["layernorm", "rmsnorm"],
        default=defaults.norm_type,
        help="Model normalization type.",
    )
    parser.add_argument(
        "--use_bias_qkv",
        action="store_true",
        default=defaults.use_bias_qkv,
        help="Use bias in Q/K/V projections.",
    )
    parser.add_argument(
        "--no_use_bias_out",
        dest="use_bias_out",
        action="store_false",
        default=defaults.use_bias_out,
        help="Disable bias in attention output projections.",
    )
    parser.add_argument(
        "--no_use_bias_ffn",
        dest="use_bias_ffn",
        action="store_false",
        default=defaults.use_bias_ffn,
        help="Disable bias in feed-forward linear layers.",
    )
    parser.add_argument(
        "--ffn_hidden_dim_mode",
        type=str,
        choices=["dff_factor"],
        default=defaults.ffn_hidden_dim_mode,
        help="DAT FFN hidden-size rule: hidden_dim*dff_factor.",
    )
    parser.add_argument(
        "--ffn_activation",
        type=str,
        choices=["gelu", "swiglu"],
        default=defaults.ffn_activation,
        help="DAT feed-forward activation.",
    )
    parser.add_argument(
        "--rope_theta",
        type=float,
        default=defaults.rope_theta,
        help="Base frequency for RoPE when pe_type=rope.",
    )
    parser.add_argument(
        "--max_rel_pos",
        type=int,
        default=defaults.max_rel_pos,
        help="Maximum relative-position distance; bounds the DAT relative symbol retriever. Defaults to max_seq_len//2 inside the model.",
    )
    parser.add_argument(
        "--sequence_boundary_policy",
        type=str,
        choices=["eos_document", "document_boundary"],
        default=defaults.sequence_boundary_policy,
        help="Sequence-boundary masking policy. eos_document adds a per-line EOS and resets causal attention and NextLat transitions at every EOS; document_boundary adds EOS only at '= = =' document markers (none in marker-less corpora) so attention spans whole documents and resets only between them.",
    )
    parser.add_argument(
        "--symbol_dim",
        type=int,
        default=defaults.symbol_dim,
        help="DAT symbol dimension. Defaults to hidden_dim.",
    )
    parser.add_argument(
        "--n_symbols",
        type=int,
        default=defaults.n_symbols,
        help="Number of learned symbols for model_type=dat. Defaults to model max_seq_len.",
    )
    parser.add_argument(
        "--symbolic_attn_n_heads",
        type=int,
        default=defaults.symbolic_attn_n_heads,
        help="Number of heads inside symbolic retrieval when symbol_retrieval=symbolic. Defaults to total DAT heads.",
    )
    parser.add_argument(
        "--symbol_retrieval",
        type=str,
        choices=["symbolic", "positional", "relative", "relsymbolic"],
        default=defaults.symbol_retrieval,
        help="DAT symbol retrieval method.",
    )
    parser.add_argument(
        "--symbolic_use_bias",
        action="store_true",
        default=defaults.symbolic_use_bias,
        help="Use bias in the symbolic retrieval query projection.",
    )
    parser.add_argument(
        "--positional_symbols_sinusoidal",
        action="store_true",
        default=defaults.positional_symbols_sinusoidal,
        help="Use sinusoidal instead of learned symbols when symbol_retrieval=positional.",
    )
    parser.add_argument(
        "--relative_symbols_rope",
        action="store_true",
        default=defaults.relative_symbols_rope,
        help="Use deterministic RoPE phase features instead of learned clipped relative symbols when symbol_retrieval=relative.",
    )
    parser.add_argument(
        "--relsymbolic_rel_n_heads",
        type=int,
        default=defaults.relsymbolic_rel_n_heads,
        help="Relational head count inside the relsymbolic symbol retriever.",
    )
    parser.add_argument(
        "--relsymbolic_symbolic_attn_n_heads",
        type=int,
        default=defaults.relsymbolic_symbolic_attn_n_heads,
        help="Symbolic-attention head count inside the relsymbolic symbol retriever.",
    )
    parser.add_argument(
        "--relsymbolic_neighborhood_size",
        type=int,
        default=defaults.relsymbolic_neighborhood_size,
        help="Past-only local neighborhood size for relsymbolic retrieval.",
    )
    parser.add_argument(
        "--relsymbolic_include_self",
        action="store_true",
        default=defaults.relsymbolic_include_self,
        help="Include the current token in relsymbolic local neighborhoods.",
    )
    parser.add_argument(
        "--no_relsymbolic_normalize_rels",
        dest="relsymbolic_normalize_rels",
        action="store_false",
        help="Disable softmax normalization over relsymbolic local-neighborhood relations.",
    )
    parser.add_argument(
        "--no_relsymbolic_trainable_symbols",
        dest="relsymbolic_trainable_symbols",
        action="store_false",
        help="Freeze the learned symbol library inside relsymbolic retrieval.",
    )
    parser.add_argument(
        "--relsymbolic_dropout",
        type=float,
        default=defaults.relsymbolic_dropout,
        help="Dropout inside relsymbolic symbol retrieval.",
    )
    parser.add_argument(
        "--relsymbolic_rel_scale",
        type=float,
        default=defaults.relsymbolic_rel_scale,
        help="Optional scale for relsymbolic local relation scores. Defaults to head_dim**-0.5.",
    )
    parser.add_argument(
        "--relsymbolic_symbolic_attn_scale",
        type=float,
        default=defaults.relsymbolic_symbolic_attn_scale,
        help="Optional scale for relsymbolic symbolic-attention retrieval. Defaults to head_dim**-0.5.",
    )
    parser.add_argument(
        "--relsymbolic_use_bias",
        action="store_true",
        default=defaults.relsymbolic_use_bias,
        help="Use bias in relsymbolic Q/K projections and nested symbolic-attention query projection.",
    )
    parser.add_argument(
        "--ra_type",
        type=str,
        choices=["ra", "rca", "disrca"],
        default=defaults.ra_type,
        help="DAT relational attention variant.",
    )
    parser.add_argument(
        "--ra_n_relations",
        type=int,
        default=defaults.ra_n_relations,
        help="Number of relation types for ra_type=ra. Defaults to n_heads_ra.",
    )
    parser.add_argument(
        "--ra_rel_activation",
        type=str,
        choices=["softmax", "identity", "tanh", "sigmoid"],
        default=defaults.ra_rel_activation,
        help="Activation for DAT relation scores.",
    )
    parser.add_argument(
        "--ra_symmetric_rels",
        action="store_true",
        default=defaults.ra_symmetric_rels,
        help="Share relation query/key projection for ra_type=ra.",
    )
    parser.add_argument(
        "--datapoint_length",
        type=int,
        default=defaults.datapoint_length,
        help="The token chunk length before dataset-level BOS/final EOS are added. Corpus policy may insert line-boundary EOS inside chunks.",
    )
    parser.add_argument(
        "--corpus_id",
        type=str,
        choices=sorted(SUPPORTED_CORPUS_IDS),
        default=defaults.corpus_id,
        help="BabyLM training track. Both tracks expect the six official training corpus files.",
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=defaults.n_epochs,
        help="Maximum number of training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=defaults.batch_size,
        help="Training batch size.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=defaults.max_train_steps,
        help=(
            "Per-epoch cap on training steps (batches). None runs the full epoch. "
            "Used for smoke tests and debugging; not set in committed experiment scripts."
        ),
    )
    parser.add_argument(
        "--max_optimizer_steps",
        type=int,
        default=defaults.max_optimizer_steps,
        help=(
            "Global optimizer-step limit. None runs all epochs. "
            "When set, all ranks stop at this optimizer step."
        ),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=defaults.learning_rate,
        help="Optimizer learning rate.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["adamw", "muon"],
        default=defaults.optimizer,
        help="Optimizer. adamw (default), or muon (Newton-Schulz for hidden weights + lambw auxiliary).",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=defaults.weight_decay,
        help="Optimizer weight decay.",
    )
    parser.add_argument(
        "--optimizer_beta1",
        type=float,
        default=defaults.optimizer_beta1,
        help="First moment decay (beta1) for AdamW and LambW.",
    )
    parser.add_argument(
        "--optimizer_beta2",
        type=float,
        default=defaults.optimizer_beta2,
        help="Second moment decay (beta2) for AdamW and LambW.",
    )
    parser.add_argument(
        "--optimizer_eps",
        type=float,
        default=defaults.optimizer_eps,
        help="Epsilon for the AdamW/LambW denominator.",
    )
    parser.add_argument(
        "--muon_lr",
        type=float,
        default=defaults.muon_lr,
        help="Learning rate for Muon hidden-weight updates when optimizer=muon.",
    )
    parser.add_argument(
        "--muon_momentum",
        type=float,
        default=defaults.muon_momentum,
        help="Momentum for Muon Newton-Schulz updates when optimizer=muon.",
    )
    parser.add_argument(
        "--muon_n_schulz_steps",
        type=int,
        default=defaults.muon_n_schulz_steps,
        help="Number of Newton-Schulz orthogonalization iterations for Muon.",
    )
    parser.add_argument(
        "--muon_aux_optimizer",
        type=str,
        choices=["lambw"],
        default=defaults.muon_aux_optimizer,
        help="Auxiliary optimizer for non-Muon parameters when optimizer=muon.",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=defaults.warmup_ratio,
        help="Fraction of derived optimizer steps used for scheduler warmup.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        choices=["cosine_min_lr"],
        default=defaults.lr_scheduler_type,
        help="Learning-rate schedule after warmup: cosine decay floored at min_lr_rate*peak.",
    )
    parser.add_argument(
        "--min_lr_rate",
        type=float,
        default=defaults.min_lr_rate,
        help="Floor learning rate as a fraction of peak. Must be > 0.",
    )
    parser.add_argument(
        "--nextlat_enabled",
        action="store_true",
        default=defaults.nextlat_enabled,
        help="Enable the NextLat auxiliary training objective.",
    )
    parser.add_argument(
        "--nextlat_lambda_mse",
        type=float,
        default=defaults.nextlat_lambda_mse,
        help="Weight for the NextLat hidden-state Smooth L1 loss.",
    )
    parser.add_argument(
        "--nextlat_lambda_kl",
        type=float,
        default=defaults.nextlat_lambda_kl,
        help="Weight for the NextLat teacher-student KL loss.",
    )
    parser.add_argument(
        "--nextlat_lambda_ce",
        type=float,
        default=defaults.nextlat_lambda_ce,
        help="Weight for the optional NextLat auxiliary token CE loss.",
    )
    parser.add_argument(
        "--nextlat_horizon",
        type=int,
        default=defaults.nextlat_horizon,
        help="Teacher-forced latent rollout horizon for the NextLat objective.",
    )
    parser.add_argument(
        "--nextlat_proj_factor",
        type=float,
        default=defaults.nextlat_proj_factor,
        help="Projection factor for the NextLat dynamics MLP hidden width.",
    )
    parser.add_argument(
        "--tie_lm_head",
        type=_parse_bool,
        default=defaults.tie_lm_head,
        help=(
            "Tie the LM head weights to the token embeddings. Accepts true or "
            "false; omit to derive from the objective (tied for NTP, untied for NextLat)."
        ),
    )
    parser.add_argument(
        "--gradient_clip_norm",
        type=float,
        default=defaults.gradient_clip_norm,
        help="Gradient norm clip value. Use -1 to disable clipping.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=defaults.gradient_accumulation_steps,
        help="Number of minibatches to accumulate before each optimizer and scheduler step.",
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=defaults.ema_decay,
        help="Decay for the exponential moving average of model weights. "
        "Omit to disable EMA; typical value when enabled is 0.999.",
    )
    parser.add_argument(
        "--ema_device",
        type=str,
        default=defaults.ema_device,
        help="Device holding the EMA weights. 'auto' uses the model device; "
        "'cpu' offloads the EMA copy to save GPU memory.",
    )
    parser.add_argument(
        "--init_from_checkpoint",
        type=str,
        default=defaults.init_from_checkpoint,
        help="Warm-start all active model state from this checkpoint directory (a "
        "checkpoint_<label> dir). This includes separately saved NextLat dynamics "
        "when enabled. Optimizer and scheduler are not restored, so the run gets "
        "a fresh schedule. Use for a second-stage/curriculum training phase.",
    )
    parser.add_argument(
        "--init_from_use_ema",
        action="store_true",
        default=defaults.init_from_use_ema,
        help="When warm-starting via --init_from_checkpoint, load model_ema.pt "
        "instead of model.pt for the main language model. Auxiliary NextLat "
        "dynamics always load their separately saved raw state.",
    )
    parser.add_argument(
        "--no_cuda_autocast",
        dest="cuda_autocast",
        action="store_false",
        default=defaults.cuda_autocast,
        help="Disable CUDA autocast and run training forward/loss computations in fp32.",
    )
    parser.add_argument(
        "--init_scheme",
        type=str,
        choices=["xavier_uniform", "normal_0_02_scaled_projection"],
        default=defaults.init_scheme,
        help="DAT parameter initialization scheme.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=defaults.seed,
        help="Random seed. Use -1 to request a generated seed later.",
    )
    parser.add_argument(
        "--tokenizer_dir",
        type=str,
        required=True,
        help="Tokenizer directory. This source release does not include tokenizer assets.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        required=True,
        help="Directory containing the six official BabyLM training corpus files.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=defaults.cache_dir,
        help="Directory for tokenized dataset caches. Repo-local cache paths must be under cache/.",
    )
    parser.add_argument(
        "--base_folder",
        type=str,
        default=defaults.base_folder,
        help="Experiment output root. Relative paths resolve from repo root.",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=defaults.experiment_name,
        help="Experiment directory name under the output root.",
    )
    parser.add_argument(
        "--wandb_project_name",
        type=str,
        default=defaults.wandb_project_name,
        help="Weights & Biases project name when --use_wandb is enabled.",
    )
    parser.add_argument(
        "--wandb_experiment_name",
        type=str,
        default=defaults.wandb_experiment_name,
        help="Weights & Biases run name when --use_wandb is enabled.",
    )
    parser.add_argument(
        "--use_wandb",
        dest="use_wandb",
        action="store_true",
        help="Enable opt-in Weights & Biases logging instead of the default local-only run.",
    )
    parser.set_defaults(
        relsymbolic_normalize_rels=defaults.relsymbolic_normalize_rels,
        relsymbolic_trainable_symbols=defaults.relsymbolic_trainable_symbols,
        relsymbolic_use_bias=defaults.relsymbolic_use_bias,
        use_wandb=defaults.use_wandb,
    )

    return parser


def parse_train_config(argv: list[str] | None = None) -> TrainConfig:
    parser = build_argument_parser()
    namespace = parser.parse_args(argv)

    return TrainConfig(**vars(namespace))


def train_config_to_dict(config: TrainConfig) -> dict[str, object]:
    return asdict(config)
