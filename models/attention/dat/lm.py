"""Decoder-only LM built from dual-attention blocks."""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from models.attention.dat.config import DatLMConfig
from models.attention.dat.core import (
    DisentangledRelationalCrossAttention,
    RelationalAttention,
    RelationalCrossAttention,
)
from models.attention.transformer.core import SelfAttention
from models.attention.dat.symbols import (
    PositionalSymbolRetriever,
    RelationalSymbolicAttentionRetriever,
    RelativePositionalSymbolRetriever,
    SymbolicAttentionRetriever,
)
from models.attention.masks import build_decoder_attention_mask
from models.attention.transformer.components import FeedForward, PositionalEncoding, PositionalInfo, RMSNorm


class DatDecoderBlock(nn.Module):
    def __init__(self, config: DatLMConfig):
        super().__init__()
        self.norm_first = config.norm_first
        # input_dim is the block state width; hidden_dim is the total DAT width
        # used with total_n_heads to derive the shared SA/RA head_dim.
        self.sensory_attention = SelfAttention(
            input_dim=config.hidden_dim,
            n_heads=config.n_heads_sa,
            hidden_dim=config.hidden_dim,
            total_n_heads=config.total_n_heads,
            dropout=config.dropout,
            use_bias_qkv=config.use_bias_qkv,
            use_bias_out=config.use_bias_out,
        )
        self.relational_attention = _build_relational_attention(config)
        self.feed_forward = FeedForward(
            input_dim=config.hidden_dim,
            hidden_dim=config.resolved_ffn_hidden_dim,
            dropout=config.dropout,
            activation=config.ffn_activation,
            use_bias=config.use_bias_ffn,
        )
        self.norm1 = _create_norm(config.norm_type, config.hidden_dim)
        self.norm2 = _create_norm(config.norm_type, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        symbol_retriever: nn.Module,
        mask: torch.Tensor,
        pos_info: Optional[PositionalInfo],
    ) -> torch.Tensor:
        if self.norm_first:
            normed_x = self.norm1(x)
            # Retrieve symbols at the block boundary so pre-norm attention and
            # input-dependent symbol retrieval use the same representation.
            symbols = symbol_retriever(normed_x)
            sensory_output, _ = self.sensory_attention(normed_x, mask=mask, pos_info=pos_info)
            relational_output, _ = self.relational_attention(
                normed_x,
                symbols,
                mask=mask,
                pos_info=pos_info,
            )
            x = x + self.dropout(torch.cat((sensory_output, relational_output), dim=-1))
            x = x + self.dropout(self.feed_forward(self.norm2(x)))
            return x

        symbols = symbol_retriever(x)
        sensory_output, _ = self.sensory_attention(x, mask=mask, pos_info=pos_info)
        relational_output, _ = self.relational_attention(
            x,
            symbols,
            mask=mask,
            pos_info=pos_info,
        )
        x = self.norm1(x + self.dropout(torch.cat((sensory_output, relational_output), dim=-1)))
        x = self.norm2(x + self.dropout(self.feed_forward(x)))
        return x


class DatDecoderLM(nn.Module):
    def __init__(self, config: DatLMConfig):
        super().__init__()
        self.config = config
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.embedding_dropout = nn.Dropout(config.dropout)
        position_dim = config.head_dim
        self.position_encoder = PositionalEncoding(
            embedding_dim=position_dim,
            pe_type=config.pe_type,
            max_len=config.max_seq_len,
            theta=config.rope_theta,
            max_rel_pos=config.max_rel_pos,
            init_range=config.init_range,
        )
        self.symbol_retriever = _build_symbol_retriever(config)
        self.layers = nn.ModuleList(DatDecoderBlock(config) for _ in range(config.n_layers))
        self.final_norm = _create_norm(config.norm_type, config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        self._init_weights()

    @property
    def device(self) -> torch.device:
        return self.token_embeddings.weight.device

    def _init_weights(self) -> None:
        if self.config.init_scheme == "xavier_uniform":
            nn.init.xavier_uniform_(
                self.token_embeddings.weight,
                gain=nn.init.calculate_gain("linear"),
            )
            if self.config.tie_lm_head:
                self.lm_head.weight = self.token_embeddings.weight
            else:
                nn.init.xavier_uniform_(
                    self.lm_head.weight,
                    gain=nn.init.calculate_gain("linear"),
                )
            return

        if self.config.init_scheme == "normal_0_02_scaled_projection":
            self._init_normal_0_02_scaled_projection()
            if self.config.tie_lm_head:
                self.lm_head.weight = self.token_embeddings.weight
            return

        raise ValueError(f"Unsupported init_scheme: {self.config.init_scheme}")

    def _init_normal_0_02_scaled_projection(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

        for module in self.modules():
            if isinstance(module, RelationalAttention):
                nn.init.normal_(module.wr_proj, mean=0.0, std=0.02)
            elif isinstance(module, SymbolicAttentionRetriever):
                nn.init.normal_(module.template_features, mean=0.0, std=1.0)
                nn.init.normal_(module.symbol_library, mean=0.0, std=1.0)
            elif isinstance(module, RelativePositionalSymbolRetriever) and not module.rope:
                nn.init.xavier_uniform_(module.position_encoder.rel_pos_embeddings_table.weight)

        scaled_std = 0.02 / math.sqrt(2 * self.config.effective_depth)
        ffn_scaled_suffix = (
            "feed_forward.w_up.weight"
            if self.config.ffn_activation == "swiglu"
            else "feed_forward.linear2.weight"
        )
        for name, parameter in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith(ffn_scaled_suffix):
                nn.init.normal_(parameter, mean=0.0, std=scaled_std)

    def _apply_layers(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor,
        pos_info: Optional[PositionalInfo],
    ) -> torch.Tensor:
        # A single retriever module is shared by every decoder layer; each
        # layer receives the same instance so no duplicate state_dict paths
        # are created.
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                symbol_retriever=self.symbol_retriever,
                mask=mask,
                pos_info=pos_info,
            )
        return hidden_states

    def _build_attention_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        bidirectional: bool = False,
    ) -> torch.Tensor:
        return build_decoder_attention_mask(
            input_ids=input_ids,
            pad_token_id=self.config.pad_token_id,
            eos_token_id=self.config.eos_token_id,
            sequence_boundary_policy=self.config.sequence_boundary_policy,
            attention_mask=attention_mask,
            bidirectional=bidirectional,
        )

    def encode_for_objective(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        bidirectional: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be rank-2 [batch, seq], got shape {tuple(input_ids.shape)}")
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {input_ids.shape[1]} exceeds max_seq_len {self.config.max_seq_len}"
            )


        mask = self._build_attention_mask(input_ids, attention_mask, bidirectional=bidirectional)
        token_embeddings = self.token_embeddings(input_ids)
        hidden_states = token_embeddings
        pos_info = self.position_encoder.get_positional_info(input_ids.shape[1], input_ids.device)
        hidden_states = self.embedding_dropout(hidden_states)

        hidden_states = self._apply_layers(hidden_states, mask, pos_info)

        hidden_states = self.final_norm(hidden_states)
        return token_embeddings, hidden_states

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        mode: str = "clm",
        bidirectional: bool = False,
        return_logits: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if mode == "clm":
            _, hidden_states = self.encode_for_objective(
                input_ids, attention_mask=attention_mask
            )
            logits = self.lm_head(hidden_states)

            loss = None
            if targets is not None:
                if targets.shape != input_ids.shape:
                    raise ValueError(
                        f"targets shape must match input_ids shape, got {tuple(targets.shape)} "
                        f"vs {tuple(input_ids.shape)}"
                    )
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    ignore_index=-1,
                )

            return logits, loss
        if mode == "encode":
            token_embeddings, hidden_states = self.encode_for_objective(
                input_ids, attention_mask=attention_mask, bidirectional=bidirectional
            )
            if return_logits:
                logits = self.lm_head(hidden_states)
                return token_embeddings, hidden_states, logits
            return token_embeddings, hidden_states
        raise ValueError(f"Unsupported forward mode: {mode}")

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        if temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        if max_new_tokens < 0:
            raise ValueError(f"max_new_tokens must be non-negative, got {max_new_tokens}")
        if top_k is not None and top_k <= 0:
            raise ValueError(f"top_k must be positive when provided, got {top_k}")
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id

        if attention_mask is None:
            generated_sequences = [row.clone() for row in input_ids]
        else:
            current_attention_mask = attention_mask.bool()
            generated_sequences = []
            for row, row_mask in zip(input_ids, current_attention_mask):
                generated_sequence = row[row_mask]
                if generated_sequence.numel() == 0:
                    raise ValueError("Each input row must contain at least one unmasked token for generation.")
                generated_sequences.append(generated_sequence)

        finished = torch.tensor(
            [sequence[-1].item() == eos_token_id for sequence in generated_sequences],
            dtype=torch.bool,
            device=input_ids.device,
        )

        for _ in range(max_new_tokens):
            context_sequences = [sequence[-self.config.max_seq_len :] for sequence in generated_sequences]
            context_ids = pad_sequence(
                context_sequences,
                batch_first=True,
                padding_value=self.config.pad_token_id,
            )
            context_mask = pad_sequence(
                [
                    torch.ones(sequence.shape[0], dtype=torch.bool, device=input_ids.device)
                    for sequence in context_sequences
                ],
                batch_first=True,
                padding_value=False,
            )
            logits, _ = self(context_ids, attention_mask=context_mask)
            last_positions = context_mask.long().sum(dim=1) - 1
            batch_indices = torch.arange(logits.shape[0], device=logits.device)
            next_token_logits = logits[batch_indices, last_positions, :] / temperature

            if top_k is not None:
                k = min(top_k, next_token_logits.size(-1))
                top_values, _ = torch.topk(next_token_logits, k=k)
                cutoff = top_values[:, -1].unsqueeze(-1)
                next_token_logits = next_token_logits.masked_fill(next_token_logits < cutoff, float("-inf"))

            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            for row_index in range(next_token.shape[0]):
                if finished[row_index]:
                    continue
                generated_sequences[row_index] = torch.cat(
                    [generated_sequences[row_index], next_token[row_index]],
                )
                if next_token[row_index, 0].item() == eos_token_id:
                    finished[row_index] = True

            if torch.all(finished):
                break

        return pad_sequence(
            generated_sequences,
            batch_first=True,
            padding_value=self.config.pad_token_id,
        )


def _build_symbol_retriever(config: DatLMConfig) -> nn.Module:
    if config.symbol_retrieval == "symbolic":
        return SymbolicAttentionRetriever(
            hidden_dim=config.hidden_dim,
            symbol_dim=config.resolved_symbol_dim,
            n_symbols=config.resolved_n_symbols,
            n_heads=config.resolved_symbolic_attn_n_heads,
            dropout=config.dropout,
            use_bias=config.symbolic_use_bias,
        )
    if config.symbol_retrieval == "positional":
        return PositionalSymbolRetriever(
            symbol_dim=config.resolved_symbol_dim,
            max_len=config.max_seq_len,
            sinusoidal=config.positional_symbols_sinusoidal,
        )
    if config.symbol_retrieval == "relative":
        max_rel_distance = config.max_rel_pos if config.max_rel_pos is not None else max(1, config.max_seq_len // 2)
        return RelativePositionalSymbolRetriever(
            symbol_dim=config.resolved_symbol_dim,
            max_rel_distance=max_rel_distance,
            rope=config.relative_symbols_rope,
            theta=config.rope_theta,
        )
    if config.symbol_retrieval == "relsymbolic":
        return RelationalSymbolicAttentionRetriever(
            hidden_dim=config.hidden_dim,
            symbol_dim=config.resolved_symbol_dim,
            rel_n_heads=config.relsymbolic_rel_n_heads,
            symbolic_attn_n_heads=config.relsymbolic_symbolic_attn_n_heads,
            n_symbols=config.resolved_n_symbols,
            neighborhood_size=config.relsymbolic_neighborhood_size,
            include_self=config.relsymbolic_include_self,
            normalize_rels=config.relsymbolic_normalize_rels,
            dropout=config.relsymbolic_dropout,
            trainable_symbols=config.relsymbolic_trainable_symbols,
            rel_scale=config.relsymbolic_rel_scale,
            symbolic_attn_scale=config.relsymbolic_symbolic_attn_scale,
            use_bias=config.relsymbolic_use_bias,
        )
    raise ValueError(f"Unsupported symbol_retrieval: {config.symbol_retrieval}")


def _build_relational_attention(config: DatLMConfig) -> nn.Module:
    use_relative_symbols = config.symbol_retrieval == "relative"
    if config.ra_type == "ra":
        return RelationalAttention(
            hidden_dim=config.hidden_dim,
            symbol_dim=config.resolved_symbol_dim,
            n_heads=config.n_heads_ra,
            total_n_heads=config.total_n_heads,
            n_relations=config.resolved_ra_n_relations,
            dropout=config.dropout,
            rel_activation=config.ra_rel_activation,
            symmetric_rels=config.ra_symmetric_rels,
            use_relative_positional_symbols=use_relative_symbols,
            use_bias_qkv=config.use_bias_qkv,
            use_bias_out=config.use_bias_out,
        )
    if config.ra_type == "rca":
        return RelationalCrossAttention(
            hidden_dim=config.hidden_dim,
            symbol_dim=config.resolved_symbol_dim,
            n_heads=config.n_heads_ra,
            total_n_heads=config.total_n_heads,
            dropout=config.dropout,
            activation=config.ra_rel_activation,
            use_relative_positional_symbols=use_relative_symbols,
            use_bias_qkv=config.use_bias_qkv,
            use_bias_out=config.use_bias_out,
        )
    if config.ra_type == "disrca":
        return DisentangledRelationalCrossAttention(
            hidden_dim=config.hidden_dim,
            symbol_dim=config.resolved_symbol_dim,
            n_heads=config.n_heads_ra,
            total_n_heads=config.total_n_heads,
            dropout=config.dropout,
            rel_activation=config.ra_rel_activation,
            use_relative_positional_symbols=use_relative_symbols,
            use_bias_qkv=config.use_bias_qkv,
            use_bias_out=config.use_bias_out,
        )
    raise ValueError(f"Unsupported ra_type: {config.ra_type}")


def _create_norm(norm_type: str, hidden_dim: int) -> nn.Module:
    if norm_type == "layernorm":
        return nn.LayerNorm(hidden_dim)
    if norm_type == "rmsnorm":
        return RMSNorm(hidden_dim)
    raise ValueError(f"Unsupported norm_type: {norm_type}")
