"""Minimal self-attention core for decoder language models."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from models.attention.transformer.components import PositionalInfo, apply_rotary_emb


def _activate_scores(scores: torch.Tensor, activation: str) -> torch.Tensor:
    if activation == "softmax":
        return torch.softmax(scores, dim=-1)
    if activation == "identity":
        return scores
    if activation == "tanh":
        return torch.tanh(scores)
    if activation == "sigmoid":
        return torch.sigmoid(scores)
    raise ValueError(f"Unsupported attention activation: {activation}")


class MultiHeadAttentionBase(nn.Module):
    def __init__(
        self,
        query_dim: int,
        output_dim: int,
        key_dim: Optional[int] = None,
        value_dim: Optional[int] = None,
        n_heads: int = 4,
        hidden_dim: int = 64,
        dropout: float = 0.0,
        total_n_heads: Optional[int] = None,
        activation: str = "softmax",
        use_bias_qkv: bool = False,
        use_bias_out: bool = True,
    ):
        super().__init__()
        self.query_dim = query_dim
        self.key_dim = query_dim if key_dim is None else key_dim
        self.value_dim = self.key_dim if value_dim is None else value_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.total_n_heads = n_heads if total_n_heads is None else total_n_heads
        self.activation = activation
        self.use_bias_qkv = use_bias_qkv
        self.use_bias_out = use_bias_out

        if self.total_n_heads <= 0:
            raise ValueError(f"total_n_heads must be positive, got {self.total_n_heads}")
        if hidden_dim % self.total_n_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by total_n_heads ({self.total_n_heads})"
            )

        self.head_dim = hidden_dim // self.total_n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        projected_dim = n_heads * self.head_dim
        self.q_proj = nn.Linear(self.query_dim, projected_dim, bias=use_bias_qkv)
        self.k_proj = nn.Linear(self.key_dim, projected_dim, bias=use_bias_qkv)
        self.v_proj = nn.Linear(self.value_dim, projected_dim, bias=use_bias_qkv)
        self.o_proj = nn.Linear(projected_dim, output_dim, bias=use_bias_out)
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        gain = 1.0 / math.sqrt(2.0)
        for projection in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
            nn.init.xavier_uniform_(projection.weight, gain=gain)
            if projection.bias is not None:
                nn.init.zeros_(projection.bias)

    def _reshape_for_multihead(self, x: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
        x = x.view(batch_size, seq_len, self.n_heads, self.head_dim)
        return x.transpose(1, 2)

    def _process_mask(self, mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if mask is None:
            return None
        if mask.dim() == 2:
            if mask.shape[0] != mask.shape[1]:
                raise ValueError(
                    "2D decoder attention masks must be square [seq, seq]; "
                    f"got {tuple(mask.shape)}. Expand padding masks at the LM boundary."
                )
            return mask.bool().unsqueeze(0).unsqueeze(0)
        if mask.dim() == 3:
            return mask.bool().unsqueeze(1)
        raise ValueError(f"Mask must be 2D or 3D, got {mask.dim()}D")

    def _compute_attn_scores(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        pos_info: Optional[PositionalInfo] = None,
    ) -> torch.Tensor:
        if pos_info is not None and pos_info.rope_freqs is not None:
            freqs_cos, freqs_sin = pos_info.rope_freqs
            q, k = apply_rotary_emb(q, k, freqs_cos, freqs_sin)
        return torch.matmul(q, k.transpose(-2, -1)) * self.scale

    def _apply_activation_and_mask(self, scores: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        processed_mask = self._process_mask(mask)
        if processed_mask is not None and self.activation in {"softmax", "sigmoid", "tanh"}:
            scores.masked_fill_(~processed_mask, torch.finfo(scores.dtype).min)
        weights = _activate_scores(scores, self.activation)
        if processed_mask is not None:
            weights = weights.masked_fill(~processed_mask, 0.0)
        return weights

    def _apply_mask(self, scores: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        return self._apply_activation_and_mask(scores, mask)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        pos_info: Optional[PositionalInfo] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = query.shape
        key = query if key is None else key
        value = key if value is None else value
        key_len = key.shape[1]

        if key.shape[0] != batch_size or value.shape[0] != batch_size:
            raise ValueError(
                f"Batch size mismatch: query={batch_size}, key={key.shape[0]}, value={value.shape[0]}"
            )
        if key.shape[1] != value.shape[1]:
            raise ValueError(
                f"Key and value sequence length mismatch: {key.shape[1]} vs {value.shape[1]}"
            )

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = self._reshape_for_multihead(q, batch_size, seq_len)
        k = self._reshape_for_multihead(k, batch_size, key_len)
        v = self._reshape_for_multihead(v, batch_size, key_len)

        attn_scores = self._compute_attn_scores(q, k, pos_info)
        attn_weights = self._apply_activation_and_mask(attn_scores, mask)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.n_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)
        attn_output = self.dropout(attn_output)
        return attn_output, attn_weights


class SelfAttention(MultiHeadAttentionBase):
    def __init__(
        self,
        input_dim: int,
        n_heads: int = 4,
        hidden_dim: int = 64,
        dropout: float = 0.0,
        total_n_heads: Optional[int] = None,
        use_bias_qkv: bool = False,
        use_bias_out: bool = True,
    ):
        head_dim = hidden_dim // (n_heads if total_n_heads is None else total_n_heads)
        super().__init__(
            query_dim=input_dim,
            output_dim=n_heads * head_dim,
            key_dim=input_dim,
            value_dim=input_dim,
            n_heads=n_heads,
            hidden_dim=hidden_dim,
            dropout=dropout,
            total_n_heads=total_n_heads,
            activation="softmax",
            use_bias_qkv=use_bias_qkv,
            use_bias_out=use_bias_out,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        pos_info: Optional[PositionalInfo] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return super().forward(query=x, mask=mask, pos_info=pos_info)


class CrossAttention(MultiHeadAttentionBase):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        context_dim: Optional[int] = None,
        n_heads: int = 8,
        hidden_dim: int = 64,
        dropout: float = 0.0,
        use_bias_qkv: bool = False,
        use_bias_out: bool = True,
    ):
        resolved_context_dim = input_dim if context_dim is None else context_dim
        super().__init__(
            query_dim=input_dim,
            key_dim=resolved_context_dim,
            value_dim=resolved_context_dim,
            output_dim=output_dim,
            n_heads=n_heads,
            hidden_dim=hidden_dim,
            dropout=dropout,
            activation="softmax",
            use_bias_qkv=use_bias_qkv,
            use_bias_out=use_bias_out,
        )

    def forward(
        self,
        inputs: torch.Tensor,
        context: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        pos_info: Optional[PositionalInfo] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return super().forward(
            query=inputs,
            key=context,
            value=context,
            mask=mask,
            pos_info=pos_info,
        )
