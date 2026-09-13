#!/usr/bin/env python3
import argparse
import csv
import re
import sys
from pathlib import Path

import yaml


ANALYSIS_FAMILY_NAMES = (
    "architecture_comparison",
    "rca_symbol_retrieval_comparison",
    "rca_sa_ra_head_ratio_comparison",
)


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def config_paths(experiment_dir: Path) -> tuple[Path, Path]:
    logging_train_config = experiment_dir / "logging" / "train_config.yaml"
    logging_model_config = experiment_dir / "logging" / "model_config.yaml"
    if logging_train_config.is_file() and logging_model_config.is_file():
        return logging_train_config, logging_model_config

    flat_train_config = experiment_dir / "train_config.yaml"
    flat_model_config = experiment_dir / "model_config.yaml"
    if flat_train_config.is_file() and flat_model_config.is_file():
        return flat_train_config, flat_model_config

    raise FileNotFoundError(f"Missing train/model config pair in {experiment_dir}")


def has_config_pair(path: Path) -> bool:
    try:
        config_paths(path)
    except FileNotFoundError:
        return False
    return True


def is_nested_logging_config_dir(path: Path) -> bool:
    return path.name == "logging" and has_config_pair(path.parent)


def analysis_family_for_path(path: Path) -> str:
    for part in path.parts:
        if part in ANALYSIS_FAMILY_NAMES:
            return part
    return path.parent.name


def is_supported_experiment_dir(path: Path, checkpoint_label: str) -> bool:
    if analysis_family_for_path(path) == "rca_sa_ra_head_ratio_comparison":
        return path.name.endswith(f"_ep{checkpoint_label}_hf")
    return True


def discover_experiment_dirs(root: Path, checkpoint_label: str) -> list[Path]:
    experiment_dirs = []
    if has_config_pair(root) and is_supported_experiment_dir(root, checkpoint_label):
        experiment_dirs.append(root)
    experiment_dirs.extend(
        path
        for path in root.rglob("*")
        if path.is_dir()
        and not is_nested_logging_config_dir(path)
        and has_config_pair(path)
        and is_supported_experiment_dir(path, checkpoint_label)
    )
    return sorted(set(experiment_dirs))


def experiment_name(experiment_dir: Path) -> str:
    train_config_path, _ = config_paths(experiment_dir)
    return str(load_yaml(train_config_path)["experiment_name"])


def causal_dirs(experiment_dir: Path, checkpoint_label: str) -> list[Path]:
    candidates = [
        experiment_dir / "official_reports",
        experiment_dir / "main" / "zero_shot" / "causal",
    ]
    results_dir = experiment_dir / "2026_eval" / "results"
    if results_dir.is_dir():
        candidates.extend(
            sorted(
                path / "main" / "zero_shot" / "causal"
                for path in results_dir.iterdir()
                if path.is_dir() and path.name.endswith(f"_ep{checkpoint_label}_hf")
            )
        )
    return [path for path in candidates if path.is_dir()]


def is_numeric_key(k: str) -> bool:
    try:
        float(k)
        return True
    except ValueError:
        return False

def parse_report(path: Path) -> dict:
    if not path.exists():
        return {}
    content = path.read_text(encoding="utf-8")
    scores = {}
    lines = content.strip().split("\n")
    for line in lines:
        if ":" in line and not line.startswith("###"):
            key, val = line.split(":", 1)
            try:
                scores[key.strip()] = float(val.strip())
            except ValueError:
                pass
    return scores

def parse_args():
    parser = argparse.ArgumentParser(description="Summarize BLiMP subtask outputs into a CSV.")
    parser.add_argument(
        "--experiment_root",
        action="append",
        type=Path,
        required=True,
        help="Root directory containing experiment folders. Can be passed more than once.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        required=True,
        help="Path to write the resulting CSV file.",
    )
    parser.add_argument(
        "--checkpoint_label",
        default="9",
        help="Checkpoint label to summarize.",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    out_path = args.output_csv

    rows = []

    for root in args.experiment_root:
        if not root.is_dir():
            print(f"Error: Experiment root {root} does not exist or is not a directory.", file=sys.stderr)
            return 1

        for exp_dir in discover_experiment_dirs(root, args.checkpoint_label):
            for causal_dir in causal_dirs(exp_dir, args.checkpoint_label):
                blimp_path = causal_dir / "blimp" / "blimp_filtered" / "best_temperature_report.txt"
                if not blimp_path.exists():
                    continue

                scores = parse_report(blimp_path)
                exp_name = experiment_name(exp_dir)
                group_name = re.sub(r"_seed\d+$", "", exp_name)

                row = {
                    "experiment_name": exp_name,
                    "experiment_group": group_name,
                    "analysis_family": analysis_family_for_path(exp_dir),
                }
                scores.pop("TEMPERATURE", None)

                # Remove numeric or junk keys that sometimes leak into reports.
                keys_to_pop = [k for k in scores if is_numeric_key(k)]
                for k in keys_to_pop:
                    scores.pop(k, None)

                row.update(scores)
                rows.append(row)

    if not rows:
        print("No subtask results found.", file=sys.stderr)
        return 1
        
    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())
        
    csv_keys = ["experiment_name", "experiment_group", "analysis_family"] + sorted(
        [
            k
            for k in all_keys
            if k not in ("experiment_name", "experiment_group", "analysis_family", "TEMPERATURE")
        ]
    )
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_keys)
        writer.writeheader()
        writer.writerows(rows)
        
    print(f"Wrote {len(rows)} subtask rows to {out_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
