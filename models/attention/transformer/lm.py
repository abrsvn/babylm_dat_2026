"""Decoder-only LM built from the local self-attention stack."""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from models.attention.masks import build_decoder_attention_mask
from models.attention.transformer.components import FeedForward, PositionalEncoding, PositionalInfo, RMSNorm
from models.attention.transformer.config import SelfAttentionLMConfig
from models.attention.transformer.core import SelfAttention


class DecoderBlock(nn.Module):
    def __init__(self, config: SelfAttentionLMConfig):
        super().__init__()
        self.norm_first = config.norm_first
        self.self_attn = SelfAttention(
            input_dim=config.hidden_dim,
            n_heads=config.n_heads,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
            use_bias_qkv=config.use_bias_qkv,
            use_bias_out=config.use_bias_out,
        )
        self.feed_forward = FeedForward(
            input_dim=config.hidden_dim,
            hidden_dim=config.hidden_dim * config.dff_factor,
            dropout=config.dropout,
            use_bias=config.use_bias_ffn,
        )
        self.norm1 = _create_norm(config.norm_type, config.hidden_dim)
        self.norm2 = _create_norm(config.norm_type, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        pos_info: Optional[PositionalInfo],
    ) -> torch.Tensor:
        if self.norm_first:
            normed_x = self.norm1(x)
            attn_output, _ = self.self_attn(normed_x, mask=mask, pos_info=pos_info)
            x = x + self.dropout(attn_output)
            x = x + self.dropout(self.feed_forward(self.norm2(x)))
            return x

        attn_output, _ = self.self_attn(x, mask=mask, pos_info=pos_info)
        x = self.norm1(x + self.dropout(attn_output))
        x = self.norm2(x + self.dropout(self.feed_forward(x)))
        return x


class SelfAttentionDecoderLM(nn.Module):
    def __init__(self, config: SelfAttentionLMConfig):
        super().__init__()
        self.config = config
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.embedding_dropout = nn.Dropout(config.dropout)
        position_dim = config.hidden_dim // config.n_heads
        self.position_encoder = PositionalEncoding(
            embedding_dim=position_dim,
            pe_type=config.pe_type,
            max_len=config.max_seq_len,
            theta=config.rope_theta,
            max_rel_pos=config.max_rel_pos,
            init_range=config.init_range,
        )
        self.layers = nn.ModuleList(DecoderBlock(config) for _ in range(config.n_layers))
        self.final_norm = _create_norm(config.norm_type, config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        self._init_weights()

    @property
    def device(self) -> torch.device:
        return self.token_embeddings.weight.device

    def _init_weights(self) -> None:
        nn.init.normal_(self.token_embeddings.weight, mean=0.0, std=0.02)
        if self.config.tie_lm_head:
            self.lm_head.weight = self.token_embeddings.weight
        else:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

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

        for layer in self.layers:
            hidden_states = layer(hidden_states, mask=mask, pos_info=pos_info)

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


def _create_norm(norm_type: str, hidden_dim: int) -> nn.Module:
    if norm_type == "layernorm":
        return nn.LayerNorm(hidden_dim)
    if norm_type == "rmsnorm":
        return RMSNorm(hidden_dim)
    raise ValueError(f"Unsupported norm_type: {norm_type}")
