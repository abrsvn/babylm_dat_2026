import sys

import torch
import pytest

from scripts.convert_dat_to_hf import (
    REPO_ROOT,
    _import_export_package,
    copy_templates,
    flatten_model_source,
)
from models.attention.transformer.components import PositionalEncoding


def test_rope_initialization_on_missing_buffers():
    """Verify that RoPE dynamically reconstructs its buffers if they are missing or uninitialized.

    This is required for huggingface accelerate compatibility, which allocates uninitialized
    tensors for persistent=False buffers that are missing from the state dictionary.
    """
    embedding_dim = 16
    max_len = 100
    theta = 10000.0

    # 1. Normal initialization
    pe = PositionalEncoding(embedding_dim, "rope", max_len, theta=theta)
    assert hasattr(pe, "rope_cos")
    assert hasattr(pe, "rope_sin")

    # 2. Simulate accelerate loading uninitialized memory (NaNs)
    # Accelerate allocates uninitialized tensors when loading the model onto a device
    pe.rope_cos.fill_(float('nan'))
    pe.rope_sin.fill_(float('nan'))

    # Verify they are broken
    assert torch.isnan(pe.rope_cos).all()

    # 3. Request positional info; this should trigger the lazy re-initialization
    pos_info = pe.get_positional_info(seq_len=10, device=torch.device("cpu"))

    # 4. Verify they are fixed
    assert not torch.isnan(pe.rope_cos).any()
    assert not torch.isnan(pe.rope_sin).any()
    assert pos_info.rope_freqs[0].shape == (10, embedding_dim // 2)
    assert not torch.isnan(pos_info.rope_freqs[0]).any()

def test_rope_initialization_on_missing_attribute():
    """Verify that if _rope_uninitialized is explicitly True, it recalculates."""
    pe = PositionalEncoding(16, "rope", 100)

    # Simulate missing/empty buffers but valid numbers (e.g. zeros)
    pe.rope_cos.fill_(0.0)
    pe.rope_sin.fill_(0.0)
    pe._rope_uninitialized = True

    # Request positional info
    pe.get_positional_info(seq_len=10, device=torch.device("cpu"))

    # Should not be zero anymore
    assert not torch.allclose(pe.rope_cos, torch.zeros_like(pe.rope_cos))


def test_sequence_classification_rejects_all_padding_inputs(tmp_path):
    output_dir = tmp_path / "hf_export"
    output_dir.mkdir()
    copy_templates(REPO_ROOT / "hf_export", output_dir)
    flatten_model_source(output_dir)

    modeling, configuration, created_init, init_path = _import_export_package(output_dir)
    try:
        config = configuration.DatConfig(
            vocab_size=32,
            max_seq_len=8,
            hidden_dim=16,
            n_heads_sa=1,
            n_heads_ra=1,
            n_layers=1,
            dff_factor=2,
            pad_token_id=0,
            num_labels=3,
        )
        model = modeling.DatForSequenceClassification(config)
        input_ids = torch.zeros((2, 4), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)

        with pytest.raises(ValueError, match="all-padding"):
            model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        if created_init:
            init_path.unlink()
        sys.modules.pop("_dat_hf_export", None)
