"""Convert a DAT checkpoint into a self-contained Hugging Face repository.

The output loads through ``AutoModelForCausalLM.from_pretrained`` with
``trust_remote_code=True`` and can be evaluated with the official BabyLM 2026
pipeline: https://github.com/babylm-org/babylm-eval

Example:
    python scripts/convert_dat_to_hf.py \
        --experiment-dir /path/to/experiment \
        --checkpoint-label 9 \
        --tokenizer-dir /path/to/tokenizer
"""

import argparse
import pathlib
import shutil
import sys

import yaml


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# Source module -> flat module name in the generated HF repo.
SOURCE_TO_FLAT = (
    ("models/attention/config.py", "attention_config.py"),
    ("models/attention/dat/config.py", "dat_config.py"),
    ("models/attention/dat/core.py", "dat_core.py"),
    ("models/attention/dat/symbols.py", "dat_symbols.py"),
    ("models/attention/dat/lm.py", "dat_lm.py"),
    ("models/attention/transformer/core.py", "transformer_core.py"),
    ("models/attention/transformer/components.py", "transformer_components.py"),
    ("models/attention/masks.py", "masks.py"),
)

# Absolute dotted module path -> relative sibling import used after flattening.
DOTTED_TO_RELATIVE = {
    "models.attention.config": ".attention_config",
    "models.attention.dat.config": ".dat_config",
    "models.attention.dat.core": ".dat_core",
    "models.attention.dat.symbols": ".dat_symbols",
    "models.attention.dat.lm": ".dat_lm",
    "models.attention.transformer.core": ".transformer_core",
    "models.attention.transformer.components": ".transformer_components",
    "models.attention.masks": ".masks",
}

TEMPLATE_FILES = ("configuration_dat.py", "modeling_dat.py")
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a DAT checkpoint to a HuggingFace repo.")
    parser.add_argument(
        "--experiment-dir",
        required=True,
        type=pathlib.Path,
        help="Experiment directory containing logging/ and checkpoints/.",
    )
    parser.add_argument(
        "--checkpoint-label",
        default="9",
        help="Checkpoint label, i.e. the suffix in checkpoints/checkpoint_<label>.",
    )
    parser.add_argument(
        "--tokenizer-dir",
        required=True,
        type=pathlib.Path,
        help="Directory containing tokenizer.json and its Hugging Face metadata files.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        type=pathlib.Path,
        help="Where to write the HF repo (default: <experiment-dir>/<experiment-name>_ep<label>_hf).",
    )
    parser.add_argument(
        "--templates-dir",
        default=REPO_ROOT / "hf_export",
        type=pathlib.Path,
        help="Directory holding configuration_dat.py and modeling_dat.py.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the output directory first if it already exists.",
    )
    return parser.parse_args()


def rewrite_imports(source: str) -> str:
    rewritten = source
    for dotted, relative in DOTTED_TO_RELATIVE.items():
        rewritten = rewritten.replace(f"from {dotted} import", f"from {relative} import")
    return rewritten


def flatten_model_source(output_dir: pathlib.Path) -> None:
    for source_rel, flat_name in SOURCE_TO_FLAT:
        source_path = REPO_ROOT / source_rel
        if not source_path.exists():
            raise FileNotFoundError(f"Expected model source not found: {source_path}")
        text = source_path.read_text(encoding="utf-8")
        (output_dir / flat_name).write_text(rewrite_imports(text), encoding="utf-8")


def copy_templates(templates_dir: pathlib.Path, output_dir: pathlib.Path) -> None:
    for name in TEMPLATE_FILES:
        source = templates_dir / name
        if not source.exists():
            raise FileNotFoundError(f"Template file not found: {source}")
        shutil.copy(source, output_dir / name)


def copy_tokenizer(tokenizer_dir: pathlib.Path, output_dir: pathlib.Path) -> None:
    for name in TOKENIZER_FILES:
        source = tokenizer_dir / name
        if not source.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {source}")
        shutil.copy(source, output_dir / name)


def _import_export_package(output_dir: pathlib.Path):
    """Import the just-written HF repo as a package so its relative imports resolve.

    modeling_dat.py uses relative imports (`from .dat_core import ...`), so it can
    only be imported as part of a package. We give the output directory a synthetic
    package name with submodule_search_locations pointing at it. A temporary
    __init__.py is created if missing and removed by the caller.
    """
    import importlib.util

    package_name = "_dat_hf_export"
    init_path = output_dir / "__init__.py"
    created_init = not init_path.exists()
    if created_init:
        init_path.write_text("", encoding="utf-8")

    spec = importlib.util.spec_from_file_location(
        package_name,
        init_path,
        submodule_search_locations=[str(output_dir)],
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = package
    spec.loader.exec_module(package)

    configuration = importlib.import_module(f"{package_name}.configuration_dat")
    modeling = importlib.import_module(f"{package_name}.modeling_dat")
    return modeling, configuration, created_init, init_path


def build_and_save(
    checkpoint_dir: pathlib.Path,
    model_config_dict: dict,
    output_dir: pathlib.Path,
) -> None:
    """Build the exported HF model, load the checkpoint into it, and save_pretrained."""
    import torch

    modeling, configuration, created_init, init_path = _import_export_package(output_dir)
    try:
        config = configuration.DatConfig(
            **{field: model_config_dict[field] for field in configuration.DAT_LM_FIELDS}
        )
        config.auto_map = {
            "AutoConfig": "configuration_dat.DatConfig",
            "AutoModel": "modeling_dat.DatModel",
            "AutoModelForCausalLM": "modeling_dat.DatForCausalLM",
            "AutoModelForSequenceClassification": "modeling_dat.DatForSequenceClassification",
        }
        config.architectures = ["DatModel", "DatForCausalLM", "DatForSequenceClassification"]

        model = modeling.DatForCausalLM(config)
        state_path = checkpoint_dir / "model.pt"
        if not state_path.exists():
            raise FileNotFoundError(f"Checkpoint weights not found: {state_path}")
        state_dict = torch.load(state_path, map_location="cpu", weights_only=True)
        # Load into the inner DatDecoderLM (its keys are unprefixed); strict load
        # surfaces any key/shape mismatch between checkpoint and config.
        model.model.load_state_dict(state_dict)
        model.eval()

        # save_pretrained writes config.json (with auto_map) + model.safetensors,
        # deduplicating tied/shared tensors so from_pretrained round-trips them.
        model.save_pretrained(output_dir, safe_serialization=True)
    finally:
        if created_init:
            init_path.unlink()
        shutil.rmtree(output_dir / "__pycache__", ignore_errors=True)
        sys.modules.pop("_dat_hf_export", None)


def main() -> None:
    args = _parse_arguments()
    sys.path.insert(0, str(REPO_ROOT))

    experiment_dir = args.experiment_dir.resolve()
    model_config_path = experiment_dir / "logging" / "model_config.yaml"
    if not model_config_path.exists():
        raise FileNotFoundError(f"model_config.yaml not found: {model_config_path}")
    model_config_dict = yaml.safe_load(model_config_path.read_text(encoding="utf-8"))
    tokenizer_dir = args.tokenizer_dir.resolve()

    checkpoint_dir = experiment_dir / "checkpoints" / f"checkpoint_{args.checkpoint_label}"
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    default_name = f"{experiment_dir.name}_ep{args.checkpoint_label}_hf"
    output_dir = args.output_dir.resolve() if args.output_dir else experiment_dir / default_name
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output directory already exists (use --overwrite): {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    # Write the custom code first so the output dir is importable as a package,
    # then build the model from it and let save_pretrained write the weights.
    flatten_model_source(output_dir)
    copy_templates(args.templates_dir.resolve(), output_dir)
    build_and_save(checkpoint_dir, model_config_dict, output_dir)
    copy_tokenizer(tokenizer_dir, output_dir)

    print(f"Wrote HuggingFace repo to: {output_dir}")
    print("Files:")
    for path in sorted(output_dir.iterdir()):
        print(f"  {path.name}")
    print("\nSanity-check load with:")
    print(f"  python3 scripts/verify_hf_dat.py --hf-dir {output_dir} \\")
    print(
        f"      --experiment-dir {experiment_dir} "
        f"--checkpoint-label {args.checkpoint_label}"
    )


if __name__ == "__main__":
    main()
