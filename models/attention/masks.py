"""Attention-mask helpers shared by Relational BabyLM decoder LMs."""

from typing import Optional

import torch


def build_decoder_attention_mask(
    input_ids: torch.Tensor,
    pad_token_id: int,
    eos_token_id: int,
    sequence_boundary_policy: str,
    attention_mask: Optional[torch.Tensor] = None,
    bidirectional: bool = False,
) -> torch.Tensor:
    if input_ids.ndim != 2:
        raise ValueError(
            "input_ids must be rank-2 [batch, seq], "
            f"got shape {tuple(input_ids.shape)}"
        )
    _, seq_len = input_ids.shape
    if attention_mask is None:
        valid_tokens = input_ids != pad_token_id
    else:
        valid_tokens = attention_mask.bool()

    query_mask = valid_tokens.unsqueeze(2)
    key_mask = valid_tokens.unsqueeze(1)
    if bidirectional:
        mask = query_mask & key_mask
    else:
        directionality = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_ids.device)
        ).unsqueeze(0)
        mask = directionality & query_mask & key_mask

    if sequence_boundary_policy not in {"eos_document", "document_boundary"}:
        raise ValueError(f"Unsupported sequence_boundary_policy: {sequence_boundary_policy}")

    # Next-token training predicts EOS from the preceding document token.
    # document_boundary shares this reset: EOS is placed only at real document
    # markers, so cumsum-on-EOS resets attention only between documents.
    # Once EOS is present as an input token, it starts the next segment so the
    # following document is not predicted with prior-document context.
    document_ids = torch.cumsum(input_ids == eos_token_id, dim=1)
    same_document = document_ids.unsqueeze(1) == document_ids.unsqueeze(2)
    return mask & same_document
