"""Minimal transformer components for decoder LM training."""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


@dataclass(frozen=True)
class PositionalInfo:
    pe_type: str
    embeddings: Optional[Tensor] = None
    rope_freqs: Optional[Tuple[Tensor, Tensor]] = None
    rel_embeddings: Optional[Tensor] = None


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        if x.dtype in (torch.float16, torch.bfloat16):
            output = self._norm(x.float()).type_as(x)
        else:
            output = self._norm(x)
        return output * self.weight


class FeedForward(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        activation: str = "gelu",
        use_bias: bool = True,
    ):
        super().__init__()
        self.activation = activation
        self.use_swiglu = activation == "swiglu"
        self.dropout = nn.Dropout(dropout)
        if self.use_swiglu:
            self.w_gate = nn.Linear(input_dim, hidden_dim, bias=use_bias)
            self.w_up = nn.Linear(input_dim, hidden_dim, bias=use_bias)
            self.w_down = nn.Linear(hidden_dim, input_dim, bias=use_bias)
            self.activation_fn = nn.SiLU()
        else:
            self.linear1 = nn.Linear(input_dim, hidden_dim, bias=use_bias)
            self.linear2 = nn.Linear(hidden_dim, input_dim, bias=use_bias)
            if activation == "gelu":
                self.activation_fn = nn.GELU()
            else:
                raise ValueError(f"Unsupported ffn activation: {activation}")
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:
        if self.use_swiglu:
            gate = self.activation_fn(self.w_gate(x))
            up = self.w_up(x)
            x = gate * up
            x = self.dropout(x)
            return self.w_down(x)

        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        pe_type: str,
        max_len: int,
        theta: float = 10000.0,
        max_rel_pos: Optional[int] = None,
        init_range: float = 0.15,
    ):
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")
        if max_len <= 0:
            raise ValueError(f"max_len must be positive, got {max_len}")
        if pe_type not in {"sinusoidal", "learned", "relative", "rope"}:
            raise ValueError(f"Unsupported pe_type: {pe_type}")
        if theta <= 0.0:
            raise ValueError(f"theta must be positive, got {theta}")
        if init_range <= 0.0:
            raise ValueError(f"init_range must be positive, got {init_range}")

        self.embedding_dim = embedding_dim
        self.pe_type = pe_type
        self.max_len = max_len

        if pe_type == "sinusoidal":
            if embedding_dim % 2 != 0:
                raise ValueError(f"Sinusoidal encoding requires even embedding_dim, got {embedding_dim}")
            pe = self._precompute_sinusoidal_embeddings(
                embedding_dim,
                max_len,
                device=torch.empty(0).device,
            )
            self.register_buffer("sinusoidal_embeddings", pe, persistent=False)
            self._sinusoidal_uninitialized = True
        elif pe_type == "learned":
            self.position_embeddings = nn.Embedding(max_len, embedding_dim)
            nn.init.uniform_(self.position_embeddings.weight, -init_range, init_range)
        elif pe_type == "relative":
            self.max_relative_position = max_rel_pos if max_rel_pos is not None else max_len // 2
            if self.max_relative_position <= 0:
                raise ValueError(
                    f"max_relative_position must be positive, got {self.max_relative_position}"
                )
            num_embeddings = 2 * self.max_relative_position + 1
            self.rel_pos_embeddings_table = nn.Embedding(num_embeddings, embedding_dim)
            nn.init.uniform_(self.rel_pos_embeddings_table.weight, -init_range, init_range)
        elif pe_type == "rope":
            if embedding_dim % 2 != 0:
                raise ValueError(f"RoPE requires even embedding_dim, got {embedding_dim}")
            self.theta = theta
            cos, sin = self._precompute_rope_freqs(embedding_dim, max_len, theta)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
            self._rope_uninitialized = True

    def get_positional_info(self, seq_len: int, device: torch.device) -> PositionalInfo:
        if self.pe_type == "sinusoidal":
            if seq_len > self.sinusoidal_embeddings.size(0):
                raise ValueError(
                    f"seq_len {seq_len} exceeds precomputed sinusoidal length {self.sinusoidal_embeddings.size(0)}"
                )
            if (
                getattr(self, "_sinusoidal_uninitialized", False)
                or self.sinusoidal_embeddings.device.type == "meta"
            ):
                embeddings = self._precompute_sinusoidal_embeddings(
                    self.embedding_dim,
                    self.max_len,
                    device=device,
                )
                self.register_buffer("sinusoidal_embeddings", embeddings, persistent=False)
                self._sinusoidal_uninitialized = False
            return PositionalInfo(
                pe_type="sinusoidal",
                embeddings=self.sinusoidal_embeddings[:seq_len].to(device),
            )
        if self.pe_type == "learned":
            if seq_len > self.max_len:
                raise ValueError(
                    f"seq_len {seq_len} exceeds maximum learned position length {self.max_len}"
                )
            positions = torch.arange(seq_len, dtype=torch.long, device=device)
            return PositionalInfo(
                pe_type="learned",
                embeddings=self.position_embeddings(positions),
            )
        if self.pe_type == "relative":
            return PositionalInfo(
                pe_type="relative",
                rel_embeddings=self._generate_relative_embeddings(seq_len, device),
            )
        if seq_len > self.rope_cos.size(0):
            # Dynamically extend RoPE buffer (standard practice for extrapolation)
            cos, sin = self._precompute_rope_freqs(self.embedding_dim, seq_len, self.theta)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
            self._rope_uninitialized = False
        if getattr(self, "_rope_uninitialized", True) or torch.isnan(self.rope_cos[0, 0]):
            cos, sin = self._precompute_rope_freqs(self.embedding_dim, self.max_len, self.theta)
            self.rope_cos.copy_(cos)
            self.rope_sin.copy_(sin)
            self._rope_uninitialized = False
        return PositionalInfo(
            pe_type="rope",
            rope_freqs=(
                self.rope_cos[:seq_len].to(device),
                self.rope_sin[:seq_len].to(device),
            ),
        )

    @staticmethod
    def _rope_inv_freq(dim: int, theta: float, device: torch.device | str) -> Tensor:
        return 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))

    @staticmethod
    def _precompute_sinusoidal_embeddings(
        dim: int,
        end: int,
        device: torch.device | str,
    ) -> Tensor:
        pe = torch.zeros(end, dim, device=device)
        position = torch.arange(0, end, dtype=torch.float, device=device).unsqueeze(1)
        div_term = PositionalEncoding._rope_inv_freq(dim, 10000.0, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    @staticmethod
    def _precompute_rope_freqs(dim: int, end: int, theta: float) -> Tuple[Tensor, Tensor]:
        freqs = PositionalEncoding._rope_inv_freq(dim, theta, device="cpu")
        positions = torch.arange(end, device="cpu")
        freqs = torch.outer(positions, freqs).float()
        return torch.cos(freqs), torch.sin(freqs)

    def _generate_relative_embeddings(self, seq_len: int, device: torch.device) -> Tensor:
        range_q = torch.arange(seq_len, device=device)
        range_k = torch.arange(seq_len, device=device)
        distance_mat = range_k[None, :] - range_q[:, None]
        distance_mat_clipped = torch.clamp(
            distance_mat,
            -self.max_relative_position,
            self.max_relative_position,
        )
        final_indices = distance_mat_clipped + self.max_relative_position
        return self.rel_pos_embeddings_table(final_indices.long())


def reshape_for_broadcast(freqs_cis: Tensor, x: Tensor) -> Tensor:
    ndim = x.ndim
    if ndim < 2:
        raise ValueError(f"Input tensor x must have at least 2 dimensions, got {ndim}")
    if x.shape[-2] != freqs_cis.shape[0] or x.shape[-1] != freqs_cis.shape[-1]:
        raise ValueError(
            f"Shape mismatch for RoPE broadcasting: freqs_cis {freqs_cis.shape}, x {x.shape}"
        )
    shape = [1] * ndim
    shape[-2] = x.shape[-2]
    shape[-1] = x.shape[-1]
    return freqs_cis.view(shape)


def apply_rotary_emb(xq: Tensor, xk: Tensor, freqs_cos: Tensor, freqs_sin: Tensor) -> Tuple[Tensor, Tensor]:
    if xq.shape[-1] % 2 != 0:
        raise ValueError(f"Query feature dimension must be even for RoPE, got {xq.shape[-1]}")
    if xk.shape[-1] % 2 != 0:
        raise ValueError(f"Key feature dimension must be even for RoPE, got {xk.shape[-1]}")

    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    freqs_cos_q = reshape_for_broadcast(freqs_cos[: xq.shape[-2]], xq_r)
    freqs_sin_q = reshape_for_broadcast(freqs_sin[: xq.shape[-2]], xq_r)
    freqs_cos_k = reshape_for_broadcast(freqs_cos[: xk.shape[-2]], xk_r)
    freqs_sin_k = reshape_for_broadcast(freqs_sin[: xk.shape[-2]], xk_r)

    xq_out_r = xq_r * freqs_cos_q - xq_i * freqs_sin_q
    xq_out_i = xq_r * freqs_sin_q + xq_i * freqs_cos_q
    xk_out_r = xk_r * freqs_cos_k - xk_i * freqs_sin_k
    xk_out_i = xk_r * freqs_sin_k + xk_i * freqs_cos_k

    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(-2)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(-2)

    return xq_out.type_as(xq), xk_out.type_as(xk)
