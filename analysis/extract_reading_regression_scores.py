#!/usr/bin/env python3
"""Extract per-experiment reading regression scores.

The reading eval already computes per-experiment OLS regressions (Stage 1):
  for each reading DV, fit baseline (no pred) vs full (with pred),
  store R2, delta-R2, coefficient, t, p in predictive_power.jsonl.

This script reads those already-computed results across all experiments and
writes a compact per-experiment reading scores CSV.

Output columns:
  experiment_name, experiment_group, analysis_family, seed_name,
  model_variant, dv_name,
  r2_full, delta_r2, pred_coef, pred_t, pred_p,
  aic, delta_aic
"""

import argparse
import csv
import gzip
import json
import re
import sys
from pathlib import Path

import yaml

# Expected per-experiment coverage: 11 DVs x 2 variants = 22 rows.
EXPECTED_DV_COUNT = 11
EXPECTED_VARIANT_COUNT = 2
EXPECTED_ROWS_PER_EXPERIMENT = EXPECTED_DV_COUNT * EXPECTED_VARIANT_COUNT


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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


def discover_experiment_dirs(root: Path) -> list[Path]:
    experiment_dirs = []
    if has_config_pair(root):
        experiment_dirs.append(root)
    experiment_dirs.extend(
        path
        for path in root.rglob("*")
        if path.is_dir()
        and not is_nested_logging_config_dir(path)
        and has_config_pair(path)
    )
    return sorted(set(experiment_dirs))


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_power_jsonl(path: Path, model_variant: str) -> list[dict]:
    """Read predictive_power.jsonl and extract rows.

    Uses direct key access for required schema fields. Fails fast if the
    reading eval output schema has changed.
    """
    rows = []
    for record in load_jsonl(path):
        row = {
            "model_variant": model_variant,
            "dv_name": record["Predicted variable"],
            "r2_full": record["R2"],
            "delta_r2": record["Change in R2 from baseline"],
            "pred_coef": record["Coefficient"],
            "pred_t": record["Number of standard deviations"],
            "pred_p": record["P-value"],
            "aic": record["AIC"],
            "delta_aic": record["Change in AIC from baseline"],
        }
        rows.append(row)
    return rows


def find_reading_dir(experiment_dir: Path) -> Path | None:
    """Locate the reading eval directory under an experiment dir.

    Searches for `reading/predictive_power.jsonl` under standard eval paths,
    including the compact bundle layout (official_reports/reading/).
    """
    candidates = [
        experiment_dir / "main" / "zero_shot" / "causal" / "reading",
        experiment_dir / "2026_eval" / "results",
        experiment_dir / "main" / "zero_shot" / "causal",
        experiment_dir / "official_reports" / "reading",
        experiment_dir / "official_reports",
    ]
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        matches = sorted(candidate.rglob("reading/predictive_power.jsonl"))
        if not matches:
            matches = sorted(candidate.rglob("predictive_power.jsonl"))
        if len(matches) >= 1:
            return matches[0].parent
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract per-experiment reading regression scores."
    )
    parser.add_argument(
        "--experiment_root",
        action="append",
        type=Path,
        required=True,
        help="Root containing experiment dirs with reading eval outputs.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        required=True,
        help="Output CSV path. Forced to end in .gz.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    out_path = args.output_csv
    if not str(out_path).endswith(".gz"):
        out_path = out_path.with_name(out_path.name + ".gz")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "experiment_name",
        "experiment_group",
        "seed_name",
        "model_variant",
        "dv_name",
        "r2_full",
        "delta_r2",
        "pred_coef",
        "pred_t",
        "pred_p",
        "aic",
        "delta_aic",
    ]

    total_rows = 0

    with gzip.open(out_path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()

        for root in args.experiment_root:
            if not root.is_dir():
                print(f"Error: {root} is not a directory", file=sys.stderr)
                return 1

            for exp_dir in discover_experiment_dirs(root):
                try:
                    train_config_path, _ = config_paths(exp_dir)
                except FileNotFoundError:
                    continue
                train_config = load_yaml(train_config_path)
                experiment_name = str(train_config["experiment_name"])
                experiment_group = re.sub(r"_seed\d+$", "", experiment_name)
                seed = train_config.get("seed", "unknown")
                seed_name = f"seed{seed}"

                reading_dir = find_reading_dir(exp_dir)
                if reading_dir is None:
                    print(
                        f"Error: no reading eval directory found for {experiment_name}",
                        file=sys.stderr,
                    )
                    return 1

                exp_rows: list[dict] = []

                # Read predictive_power.jsonl (base model: no spillover).
                base_path = reading_dir / "predictive_power.jsonl"
                if base_path.is_file():
                    exp_rows.extend(parse_power_jsonl(base_path, "base"))

                # Read predictive_power_spillover.jsonl (spillover model: with prev_pred).
                spillover_path = reading_dir / "predictive_power_spillover.jsonl"
                if spillover_path.is_file():
                    exp_rows.extend(parse_power_jsonl(spillover_path, "spillover"))

                # Validate per-experiment coverage: 11 DVs x 2 variants = 22 rows.
                if len(exp_rows) != EXPECTED_ROWS_PER_EXPERIMENT:
                    print(
                        f"Error: {experiment_name} has {len(exp_rows)} reading rows, "
                        f"expected {EXPECTED_ROWS_PER_EXPERIMENT} (11 DVs x 2 variants)",
                        file=sys.stderr,
                    )
                    return 1

                for row in exp_rows:
                    writer.writerow({
                        "experiment_name": experiment_name,
                        "experiment_group": experiment_group,
                        "seed_name": seed_name,
                        "model_variant": row["model_variant"],
                        "dv_name": row["dv_name"],
                        "r2_full": row["r2_full"],
                        "delta_r2": row["delta_r2"],
                        "pred_coef": row["pred_coef"],
                        "pred_t": row["pred_t"],
                        "pred_p": row["pred_p"],
                        "aic": row["aic"],
                        "delta_aic": row["delta_aic"],
                    })
                    total_rows += 1

    if total_rows == 0:
        print("No reading regression scores extracted.", file=sys.stderr)
        return 1

    print(f"Wrote {total_rows} reading regression score rows to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
