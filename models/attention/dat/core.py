"""Dual-attention primitives for decoder language models."""

import math
from typing import Optional

import torch
import torch.nn as nn

from models.attention.transformer.components import PositionalInfo
from models.attention.transformer.core import MultiHeadAttentionBase


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


class RelationalAttentionBase(MultiHeadAttentionBase):
    def __init__(
        self,
        hidden_dim: int,
        symbol_dim: int,
        n_heads: int,
        total_n_heads: int,
        dropout: float = 0.0,
        n_relations: Optional[int] = None,
        rel_activation: str = "identity",
        symmetric_rels: bool = False,
        use_relative_positional_symbols: bool = False,
        use_bias_qkv: bool = False,
        use_bias_out: bool = True,
    ):
        head_dim = hidden_dim // total_n_heads
        output_dim = n_heads * head_dim
        super().__init__(
            query_dim=hidden_dim,
            output_dim=output_dim,
            key_dim=hidden_dim,
            value_dim=symbol_dim,
            n_heads=n_heads,
            hidden_dim=hidden_dim,
            dropout=dropout,
            total_n_heads=total_n_heads,
            activation="softmax",
            use_bias_qkv=use_bias_qkv,
            use_bias_out=use_bias_out,
        )
        self.symbol_dim = symbol_dim
        self.rel_activation = rel_activation
        self.symmetric_rels = symmetric_rels
        self.use_relative_positional_symbols = use_relative_positional_symbols
        self.n_relations = n_heads if n_relations is None else n_relations
        total_rel_dim = self.head_dim * n_heads
        if total_rel_dim % self.n_relations != 0:
            raise ValueError(
                f"head_dim * n_heads ({total_rel_dim}) must be divisible by n_relations "
                f"({self.n_relations})"
            )
        self.rel_proj_dim = total_rel_dim // self.n_relations
        self.rel_scale = 1.0 / math.sqrt(self.rel_proj_dim)
        rel_total_dim = self.n_relations * self.rel_proj_dim
        self.wq_rel = nn.Linear(hidden_dim, rel_total_dim, bias=False)
        self.wk_rel = self.wq_rel if symmetric_rels else nn.Linear(
            hidden_dim,
            rel_total_dim,
            bias=False,
        )
        nn.init.xavier_uniform_(self.wq_rel.weight)
        if self.wk_rel is not self.wq_rel:
            nn.init.xavier_uniform_(self.wk_rel.weight)

    def _compute_base_attention(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        pos_info: Optional[PositionalInfo],
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        q = self._reshape_for_multihead(self.q_proj(x), batch_size, seq_len)
        k = self._reshape_for_multihead(self.k_proj(x), batch_size, seq_len)
        scores = self._compute_attn_scores(q, k, pos_info)
        return self._apply_activation_and_mask(scores, mask)

    def compute_relational_scores(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        return_as: str,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        q_rel = self.wq_rel(x).view(batch_size, seq_len, self.n_relations, self.rel_proj_dim)
        k_rel = self.wk_rel(x).view(batch_size, seq_len, self.n_relations, self.rel_proj_dim)
        q_rel = q_rel.transpose(1, 2)
        k_rel = k_rel.transpose(1, 2)
        relations = torch.matmul(q_rel, k_rel.transpose(-2, -1)) * self.rel_scale
        processed_mask = self._process_mask(mask)
        if self.rel_activation == "softmax" and processed_mask is not None:
            relations.masked_fill_(~processed_mask, torch.finfo(relations.dtype).min)
        relations = _activate_scores(relations, self.rel_activation)
        if processed_mask is not None:
            relations = relations.masked_fill(~processed_mask, 0.0)
        if return_as == "vectors":
            return relations.permute(0, 2, 3, 1)
        if return_as == "scores":
            return relations
        raise ValueError(f"return_as must be 'vectors' or 'scores', got {return_as}")

    def combine_attention_and_relations(
        self,
        attn_weights: torch.Tensor,
        relation_scores: torch.Tensor,
    ) -> torch.Tensor:
        return attn_weights * relation_scores

    def _process_symbols_with_attention(
        self,
        symbols: torch.Tensor,
        attn_weights: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, _, seq_len_q, seq_len_k = attn_weights.shape
        values = self.v_proj(symbols)
        if self.use_relative_positional_symbols:
            values = values.view(seq_len_q, seq_len_k, self.n_heads, self.head_dim)
            return torch.einsum("bhij,ijhd->bihd", attn_weights, values)

        values = values.view(batch_size, seq_len_k, self.n_heads, self.head_dim)
        values = values.transpose(1, 2)
        output = torch.matmul(attn_weights, values)
        return output.transpose(1, 2)

    def _apply_output_projection(self, output: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = output.shape[:2]
        output = output.contiguous().view(batch_size, seq_len, self.output_dim)
        output = self.o_proj(output)
        output = self.dropout(output)
        return output

    def _validate_symbols(self, x: torch.Tensor, symbols: torch.Tensor) -> None:
        if self.use_relative_positional_symbols:
            seq_len = x.shape[1]
            expected_shape = (seq_len, seq_len, self.symbol_dim)
            if tuple(symbols.shape) != expected_shape:
                raise ValueError(
                    f"Relative symbols must have shape {expected_shape}, got {tuple(symbols.shape)}"
                )


class RelationalAttention(RelationalAttentionBase):
    def __init__(
        self,
        hidden_dim: int,
        symbol_dim: int,
        n_heads: int,
        total_n_heads: int,
        n_relations: int,
        dropout: float = 0.0,
        rel_activation: str = "identity",
        symmetric_rels: bool = False,
        use_relative_positional_symbols: bool = False,
        use_bias_qkv: bool = False,
        use_bias_out: bool = True,
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            symbol_dim=symbol_dim,
            n_heads=n_heads,
            total_n_heads=total_n_heads,
            dropout=dropout,
            n_relations=n_relations,
            rel_activation=rel_activation,
            symmetric_rels=symmetric_rels,
            use_relative_positional_symbols=use_relative_positional_symbols,
            use_bias_qkv=use_bias_qkv,
            use_bias_out=use_bias_out,
        )
        self.wr_proj = nn.Parameter(torch.empty(n_heads, self.head_dim, n_relations))
        nn.init.xavier_uniform_(self.wr_proj)

    def forward(
        self,
        x: torch.Tensor,
        symbols: torch.Tensor,
        mask: Optional[torch.Tensor],
        pos_info: Optional[PositionalInfo],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self._validate_symbols(x, symbols)
        attn_weights = self._compute_base_attention(x, mask, pos_info)
        attn_weights = self.attn_dropout(attn_weights)
        relation_vectors = self.compute_relational_scores(x, mask, return_as="vectors")
        attended_symbols = self._process_symbols_with_attention(symbols, attn_weights)
        projected_relations = torch.einsum(
            "bhij,bijr,hdr->bihd",
            attn_weights,
            relation_vectors,
            self.wr_proj,
        )
        output = self._apply_output_projection(attended_symbols + projected_relations)
        return output, {"attention": attn_weights, "relations": relation_vectors}


class RelationalCrossAttention(MultiHeadAttentionBase):
    def __init__(
        self,
        hidden_dim: int,
        symbol_dim: int,
        n_heads: int,
        total_n_heads: int,
        dropout: float = 0.0,
        activation: str = "identity",
        use_relative_positional_symbols: bool = False,
        use_bias_qkv: bool = False,
        use_bias_out: bool = True,
    ):
        head_dim = hidden_dim // total_n_heads
        super().__init__(
            query_dim=hidden_dim,
            output_dim=n_heads * head_dim,
            key_dim=hidden_dim,
            value_dim=symbol_dim,
            n_heads=n_heads,
            hidden_dim=hidden_dim,
            dropout=dropout,
            total_n_heads=total_n_heads,
            activation=activation,
            use_bias_qkv=use_bias_qkv,
            use_bias_out=use_bias_out,
        )
        self.symbol_dim = symbol_dim
        self.use_relative_positional_symbols = use_relative_positional_symbols

    def forward(
        self,
        x: torch.Tensor,
        symbols: torch.Tensor,
        mask: Optional[torch.Tensor],
        pos_info: Optional[PositionalInfo],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.use_relative_positional_symbols:
            batch_size, seq_len, _ = x.shape
            expected_shape = (seq_len, seq_len, self.symbol_dim)
            if tuple(symbols.shape) != expected_shape:
                raise ValueError(
                    f"Relative symbols must have shape {expected_shape}, got {tuple(symbols.shape)}"
                )

            q = self._reshape_for_multihead(self.q_proj(x), batch_size, seq_len)
            k = self._reshape_for_multihead(self.k_proj(x), batch_size, seq_len)
            values = self.v_proj(symbols).view(seq_len, seq_len, self.n_heads, self.head_dim)
            scores = self._compute_attn_scores(q, k, pos_info)
            weights = self._apply_activation_and_mask(scores, mask)
            weights = self.attn_dropout(weights)
            output = torch.einsum("bhij,ijhd->bihd", weights, values)
            output = output.contiguous().view(batch_size, seq_len, self.output_dim)
            output = self.o_proj(output)
            output = self.dropout(output)
            return output, {"attention": weights}

        output, weights = super().forward(
            query=x,
            key=x,
            value=symbols,
            mask=mask,
            pos_info=pos_info,
        )
        return output, {"attention": weights}


class DisentangledRelationalCrossAttention(RelationalAttentionBase):
    def __init__(
        self,
        hidden_dim: int,
        symbol_dim: int,
        n_heads: int,
        total_n_heads: int,
        dropout: float = 0.0,
        rel_activation: str = "identity",
        use_relative_positional_symbols: bool = False,
        use_bias_qkv: bool = False,
        use_bias_out: bool = True,
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            symbol_dim=symbol_dim,
            n_heads=n_heads,
            total_n_heads=total_n_heads,
            dropout=dropout,
            n_relations=None,
            rel_activation=rel_activation,
            symmetric_rels=False,
            use_relative_positional_symbols=use_relative_positional_symbols,
            use_bias_qkv=use_bias_qkv,
            use_bias_out=use_bias_out,
        )

    def forward(
        self,
        x: torch.Tensor,
        symbols: torch.Tensor,
        mask: Optional[torch.Tensor],
        pos_info: Optional[PositionalInfo],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self._validate_symbols(x, symbols)
        attn_weights = self._compute_base_attention(x, mask, pos_info)
        relation_scores = self.compute_relational_scores(x, mask, return_as="scores")
        combined_weights = self.attn_dropout(
            self.combine_attention_and_relations(attn_weights, relation_scores)
        )
        output = self._process_symbols_with_attention(symbols, combined_weights)
        output = self._apply_output_projection(output)
        return output, {
            "attention": attn_weights,
            "relations": relation_scores,
            "combined": combined_weights,
        }
