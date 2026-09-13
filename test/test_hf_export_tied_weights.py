"""Regression tests for HF export/load with tied and untied LM heads.

NextLat models train with tie_lm_head=False, so the export path must keep
lm_head.weight separate from token_embeddings.weight through
save_pretrained/from_pretrained. These tests verify that:

1. With tie_lm_head=False, lm_head.weight is NOT tied to
   token_embeddings.weight and is fully present in the checkpoint.
2. With tie_lm_head=True, lm_head.weight is tied to token_embeddings.weight.
3. resize_token_embeddings keeps the untied head in sync with the new size.
"""

import sys

from scripts.convert_dat_to_hf import (
    REPO_ROOT,
    _import_export_package,
    build_and_save,
    copy_templates,
    flatten_model_source,
)


def _purge_export_modules():
    for key in list(sys.modules.keys()):
        if key == "_dat_hf_export" or key.startswith("_dat_hf_export."):
            sys.modules.pop(key, None)


def _tiny_dat_config_dict():
    """A full DAT_LM_FIELDS config dict with small dimensions."""
    return {
        "vocab_size": 32,
        "max_seq_len": 8,
        "pe_type": "rope",
        "hidden_dim": 16,
        "n_heads_sa": 2,
        "n_heads_ra": 2,
        "n_layers": 1,
        "dropout": 0.0,
        "dff_factor": 2,
        "ffn_hidden_dim_mode": "dff_factor",
        "ffn_activation": "gelu",
        "rope_theta": 10000.0,
        "max_rel_pos": None,
        "init_range": 0.15,
        "init_scheme": "xavier_uniform",
        "norm_type": "rmsnorm",
        "norm_first": True,
        "use_bias_qkv": False,
        "use_bias_out": True,
        "use_bias_ffn": True,
        "tie_lm_head": False,
        "symbol_dim": None,
        "n_symbols": None,
        "symbolic_attn_n_heads": None,
        "symbol_retrieval": "symbolic",
        "symbolic_use_bias": False,
        "positional_symbols_sinusoidal": False,
        "relative_symbols_rope": False,
        "relsymbolic_rel_n_heads": 4,
        "relsymbolic_symbolic_attn_n_heads": 4,
        "relsymbolic_neighborhood_size": 2,
        "relsymbolic_include_self": False,
        "relsymbolic_normalize_rels": True,
        "relsymbolic_trainable_symbols": True,
        "relsymbolic_dropout": 0.0,
        "relsymbolic_rel_scale": None,
        "relsymbolic_symbolic_attn_scale": None,
        "relsymbolic_use_bias": False,
        "ra_type": "ra",
        "ra_n_relations": None,
        "ra_rel_activation": "identity",
        "ra_symmetric_rels": False,
        "sequence_boundary_policy": "eos_document",
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
    }


def _build_and_export(tmp_path, config_dict):
    """Export a DAT model to HF format. Returns the output directory."""
    output_dir = tmp_path / "hf_export"
    output_dir.mkdir()
    copy_templates(REPO_ROOT / "hf_export", output_dir)
    flatten_model_source(output_dir)

    modeling, configuration, created_init, init_path = _import_export_package(output_dir)
    try:
        config = configuration.DatConfig(
            **{field: config_dict[field] for field in configuration.DAT_LM_FIELDS}
        )
        config.auto_map = {
            "AutoConfig": "configuration_dat.DatConfig",
            "AutoModel": "modeling_dat.DatModel",
            "AutoModelForCausalLM": "modeling_dat.DatForCausalLM",
        }
        config.architectures = ["DatModel", "DatForCausalLM"]

        model = modeling.DatForCausalLM(config)
        model.eval()
        model.save_pretrained(output_dir, safe_serialization=True)
    finally:
        if created_init:
            init_path.unlink()
        sys.modules.pop("_dat_hf_export", None)

    return output_dir


def _load_from_pretrained(output_dir):
    """Load the exported HF model via from_pretrained.

    Returns (model, missing_keys) where missing_keys is the set of
    parameters that from_pretrained reported as missing from the checkpoint.
    """
    _purge_export_modules()
    modeling, configuration, created_init, init_path = _import_export_package(output_dir)
    try:
        model, loading_info = modeling.DatForCausalLM.from_pretrained(
            output_dir,
            output_loading_info=True,
        )
        return model, set(loading_info["missing_keys"])
    finally:
        if created_init:
            init_path.unlink()
        sys.modules.pop("_dat_hf_export", None)


def test_hf_export_load_untied_lm_head(tmp_path):
    """With tie_lm_head=False (NextLat), lm_head is fully present in the
    checkpoint and NOT tied to token_embeddings after from_pretrained.
    """
    config_dict = _tiny_dat_config_dict()
    output_dir = _build_and_export(tmp_path, config_dict)
    model, missing_keys = _load_from_pretrained(output_dir)

    lm_head = model.model.lm_head.weight
    token_emb = model.model.token_embeddings.weight

    assert not missing_keys
    assert lm_head.data_ptr() != token_emb.data_ptr(), (
        "lm_head.weight should NOT be tied to token_embeddings.weight "
        "when tie_lm_head=False"
    )


def test_hf_export_load_tied_lm_head(tmp_path):
    """With tie_lm_head=True, lm_head is tied to token_embeddings."""
    config_dict = _tiny_dat_config_dict()
    config_dict["tie_lm_head"] = True
    output_dir = _build_and_export(tmp_path, config_dict)
    model, missing_keys = _load_from_pretrained(output_dir)

    lm_head = model.model.lm_head.weight
    token_emb = model.model.token_embeddings.weight

    assert not missing_keys
    assert lm_head.data_ptr() == token_emb.data_ptr()


def test_hf_export_resize_untied_lm_head(tmp_path):
    """resize_token_embeddings updates both lm_head and token_embeddings when
    tie_lm_head=False, and the head stays untied.
    """
    from transformers import AutoModelForCausalLM

    config_dict = _tiny_dat_config_dict()
    output_dir = _build_and_export(tmp_path, config_dict)

    model = AutoModelForCausalLM.from_pretrained(output_dir, trust_remote_code=True)
    original_vocab = config_dict["vocab_size"]
    new_vocab = original_vocab + 10

    model.resize_token_embeddings(new_vocab)

    lm_head = model.model.lm_head.weight
    token_emb = model.model.token_embeddings.weight

    assert lm_head.shape[0] == new_vocab
    assert token_emb.shape[0] == new_vocab
    assert lm_head.data_ptr() != token_emb.data_ptr(), (
        "lm_head.weight should stay untied from token_embeddings.weight "
        "after resize_token_embeddings"
    )


def test_build_and_save_removes_import_pycache(tmp_path):
    """A converter-created HF export must not contain Python bytecode."""
    import torch

    config_dict = _tiny_dat_config_dict()
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    copy_templates(REPO_ROOT / "hf_export", source_dir)
    flatten_model_source(source_dir)

    modeling, configuration, created_init, init_path = _import_export_package(source_dir)
    try:
        config = configuration.DatConfig(
            **{field: config_dict[field] for field in configuration.DAT_LM_FIELDS}
        )
        model = modeling.DatForCausalLM(config)
        torch.save(model.model.state_dict(), checkpoint_dir / "model.pt")
    finally:
        if created_init:
            init_path.unlink()
        _purge_export_modules()

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    copy_templates(REPO_ROOT / "hf_export", output_dir)
    flatten_model_source(output_dir)
    build_and_save(checkpoint_dir, config_dict, output_dir)

    assert not (output_dir / "__pycache__").exists()
