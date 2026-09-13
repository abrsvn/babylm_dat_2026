"""Symbol retrieval modules for the owned DAT backend."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.attention.transformer.components import PositionalEncoding


class SymbolicAttentionRetriever(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        symbol_dim: int,
        n_symbols: int,
        n_heads: int,
        dropout: float = 0.0,
        trainable_symbols: bool = True,
        scale: float | None = None,
        use_bias: bool = False,
    ):
        super().__init__()
        if hidden_dim % n_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by n_heads ({n_heads})")
        if symbol_dim % n_heads != 0:
            raise ValueError(f"symbol_dim ({symbol_dim}) must be divisible by n_heads ({n_heads})")
        if scale is not None and scale <= 0.0:
            raise ValueError(f"scale must be positive when provided, got {scale}")

        self.hidden_dim = hidden_dim
        self.symbol_dim = symbol_dim
        self.n_symbols = n_symbols
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.symbol_head_dim = symbol_dim // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim) if scale is None else scale

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=use_bias)
        self.template_features = nn.Parameter(torch.empty(n_symbols, hidden_dim))
        self.symbol_library = nn.Parameter(
            torch.empty(n_symbols, symbol_dim),
            requires_grad=trainable_symbols,
        )
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.template_features, mean=0.0, std=0.9)
        nn.init.normal_(self.symbol_library, mean=0.0, std=0.9)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        queries = self.q_proj(x)
        queries = queries.view(batch_size, seq_len, self.n_heads, self.head_dim)
        queries = queries.transpose(1, 2)

        keys = self.template_features.view(self.n_symbols, self.n_heads, self.head_dim)
        keys = keys.transpose(0, 1).unsqueeze(0).expand(batch_size, -1, -1, -1)

        values = self.symbol_library.view(self.n_symbols, self.n_heads, self.symbol_head_dim)
        values = values.transpose(0, 1).unsqueeze(0).expand(batch_size, -1, -1, -1)

        scores = torch.matmul(queries, keys.transpose(-2, -1)) * self.scale
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        retrieved = torch.matmul(weights, values)
        retrieved = retrieved.transpose(1, 2).contiguous()
        return retrieved.view(batch_size, seq_len, self.symbol_dim)


class PositionalSymbolRetriever(nn.Module):
    def __init__(
        self,
        symbol_dim: int,
        max_len: int,
        sinusoidal: bool = False,
    ):
        super().__init__()
        self.symbol_dim = symbol_dim
        self.max_len = max_len
        self.sinusoidal = sinusoidal
        pe_type = "sinusoidal" if sinusoidal else "learned"
        self.position_encoder = PositionalEncoding(
            embedding_dim=symbol_dim,
            pe_type=pe_type,
            max_len=max_len,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        pos_info = self.position_encoder.get_positional_info(seq_len, x.device)
        if pos_info.embeddings is None:
            raise ValueError(f"Positional symbol retrieval produced no embeddings for seq_len={seq_len}")
        return pos_info.embeddings.unsqueeze(0).expand(batch_size, -1, -1)


class RelativePositionalSymbolRetriever(nn.Module):
    def __init__(
        self,
        symbol_dim: int,
        max_rel_distance: int,
        rope: bool = False,
        theta: float = 10000.0,
    ):
        super().__init__()
        if rope and symbol_dim % 2 != 0:
            raise ValueError(f"RoPE relative symbols require even symbol_dim, got {symbol_dim}")
        if theta <= 0.0:
            raise ValueError(f"theta must be positive, got {theta}")
        self.symbol_dim = symbol_dim
        self.max_rel_distance = max_rel_distance
        self.rope = rope
        self.theta = theta
        if rope:
            self.register_buffer("_rope_relative_cache", torch.empty(0), persistent=False)
        else:
            self.position_encoder = PositionalEncoding(
                embedding_dim=symbol_dim,
                pe_type="relative",
                max_len=max_rel_distance * 2,
                max_rel_pos=max_rel_distance,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        if self.rope:
            return self._rope_relative_symbols(seq_len, x.device, x.dtype)

        pos_info = self.position_encoder.get_positional_info(seq_len, x.device)
        if pos_info.rel_embeddings is None:
            raise ValueError(f"Relative symbol retrieval produced no embeddings for seq_len={seq_len}")
        return pos_info.rel_embeddings

    def _rope_relative_symbols(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        cached = self._rope_relative_cache
        if (
            cached.shape[0] >= seq_len
            and cached.device == device
            and cached.dtype == dtype
        ):
            return cached[:seq_len, :seq_len, :]

        positions = torch.arange(seq_len, device=device)
        distances = positions[None, :] - positions[:, None]
        if self.max_rel_distance is not None:
            distances = torch.clamp(distances, -self.max_rel_distance, self.max_rel_distance)
        inv_freq = PositionalEncoding._rope_inv_freq(self.symbol_dim, self.theta, device=device)
        phases = distances.to(torch.float32).unsqueeze(-1) * inv_freq
        symbols = torch.empty(seq_len, seq_len, self.symbol_dim, device=device)
        symbols[..., 0::2] = torch.cos(phases)
        symbols[..., 1::2] = torch.sin(phases)
        self._rope_relative_cache = symbols.to(dtype=dtype)
        return self._rope_relative_cache


class RelationalSymbolicAttentionRetriever(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        symbol_dim: int,
        rel_n_heads: int,
        symbolic_attn_n_heads: int,
        n_symbols: int,
        neighborhood_size: int = 2,
        include_self: bool = False,
        normalize_rels: bool = True,
        dropout: float = 0.0,
        trainable_symbols: bool = True,
        rel_scale: float | None = None,
        symbolic_attn_scale: float | None = None,
        use_bias: bool = False,
    ):
        super().__init__()
        if hidden_dim % rel_n_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by rel_n_heads ({rel_n_heads})"
            )
        if rel_scale is not None and rel_scale <= 0.0:
            raise ValueError(f"rel_scale must be positive when provided, got {rel_scale}")
        if symbolic_attn_scale is not None and symbolic_attn_scale <= 0.0:
            raise ValueError(
                "symbolic_attn_scale must be positive when provided, "
                f"got {symbolic_attn_scale}"
            )

        self.hidden_dim = hidden_dim
        self.symbol_dim = symbol_dim
        self.rel_n_heads = rel_n_heads
        self.neighborhood_size = neighborhood_size
        self.include_self = include_self
        self.normalize_rels = normalize_rels
        self.neighborhood_dim = neighborhood_size + (1 if include_self else 0)
        self.rel_feature_dim = rel_n_heads * self.neighborhood_dim
        rel_head_dim = hidden_dim // rel_n_heads
        self.rel_scale = 1.0 / math.sqrt(rel_head_dim) if rel_scale is None else rel_scale

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=use_bias)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=use_bias)
        self.rel_to_hidden = nn.Linear(self.rel_feature_dim, hidden_dim, bias=True)
        self.symbolic_attention = SymbolicAttentionRetriever(
            hidden_dim=hidden_dim,
            symbol_dim=symbol_dim,
            n_symbols=n_symbols,
            n_heads=symbolic_attn_n_heads,
            dropout=dropout,
            trainable_symbols=trainable_symbols,
            scale=symbolic_attn_scale,
            use_bias=use_bias,
        )
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        if self.q_proj.bias is not None:
            nn.init.zeros_(self.q_proj.bias)
        if self.k_proj.bias is not None:
            nn.init.zeros_(self.k_proj.bias)
        nn.init.xavier_uniform_(self.rel_to_hidden.weight)
        nn.init.zeros_(self.rel_to_hidden.bias)

    def _compute_neighborhood_indices(self, seq_len: int, device: torch.device) -> torch.Tensor:
        positions = torch.arange(seq_len, device=device).unsqueeze(1)
        # Decoder-LM adaptation: DSSL uses bidirectional neighborhoods, but this
        # Decoder-only retrieval remains causal: current/immediate-past
        # positions precede older positions in each local neighborhood.
        if self.include_self:
            offsets = torch.arange(0, self.neighborhood_size + 1, device=device).unsqueeze(0)
        else:
            offsets = torch.arange(1, self.neighborhood_size + 1, device=device).unsqueeze(0)
        return (positions - offsets).clamp(0, seq_len - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        head_dim = self.hidden_dim // self.rel_n_heads

        queries = self.q_proj(x).view(batch_size, seq_len, self.rel_n_heads, head_dim)
        keys = self.k_proj(x).view(batch_size, seq_len, self.rel_n_heads, head_dim)
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)

        neighbor_indices = self._compute_neighborhood_indices(seq_len, x.device)
        neighborhood_keys = keys[:, :, neighbor_indices]
        neighborhood_relations = torch.einsum("bhid,bhijd->bhij", queries, neighborhood_keys)
        if self.normalize_rels:
            neighborhood_relations = F.softmax(neighborhood_relations * self.rel_scale, dim=-1)

        neighborhood_relations = neighborhood_relations.permute(0, 2, 3, 1)
        neighborhood_relations = neighborhood_relations.contiguous().view(
            batch_size,
            seq_len,
            self.rel_feature_dim,
        )
        relational_features = self.rel_to_hidden(neighborhood_relations)
        return self.symbolic_attention(relational_features)
