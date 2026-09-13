#!/usr/bin/env python3
"""Summarize official 2026 zero-shot submeasures into a long CSV."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf
import yaml


ANALYSIS_FAMILY_NAMES = (
    "architecture_comparison",
    "rca_symbol_retrieval_comparison",
    "rca_sa_ra_head_ratio_comparison",
)
REPORT_SPECS = (
    ("blimp", ("blimp", "blimp_filtered", "best_temperature_report.txt")),
    ("blimp_supplement", ("blimp", "supplement_filtered", "best_temperature_report.txt")),
    ("ewok", ("ewok", "ewok_filtered", "best_temperature_report.txt")),
    ("comps", ("comps", "comps", "best_temperature_report.txt")),
    (
        "entity_tracking",
        ("entity_tracking", "entity_tracking", "best_temperature_report.txt"),
    ),
)
READING_EYE_TRACKING_DVS = ("RTfirstfix", "RTfirstpass", "RTgopast", "RTrightbound")
CSV_FIELDS = (
    "experiment_name",
    "experiment_group",
    "task",
    "submeasure_type",
    "submeasure",
    "score",
    "n_items",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write long-format official 2026 zero-shot submeasure scores."
    )
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
        help="Path to write the submeasure CSV.",
    )
    parser.add_argument(
        "--eval_data_root",
        type=Path,
        required=True,
        help="Official full_eval data directory used for reading submeasures "
        "(required; the evaluation data is not part of this release).",
    )
    parser.add_argument(
        "--checkpoint_label",
        default="9",
        help="Checkpoint label to summarize, matching the matrix checkpoint label.",
    )
    return parser.parse_args()


def experiment_group_from_name(experiment_name: str) -> str:
    return re.sub(r"_seed\d+$", "", experiment_name)


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


def experiment_name_for_dir(experiment_dir: Path) -> str:
    train_config_path, _ = config_paths(experiment_dir)
    return str(load_yaml(train_config_path)["experiment_name"])


def causal_dir_for_checkpoint(
    experiment_dir: Path,
    experiment_name: str,
    checkpoint_label: str,
) -> Path | None:
    candidates = [
        experiment_dir / "official_reports",
        experiment_dir / "main" / "zero_shot" / "causal",
        experiment_dir
        / "2026_eval"
        / "results"
        / f"{experiment_dir.name}_ep{checkpoint_label}_hf"
        / "main"
        / "zero_shot"
        / "causal",
        experiment_dir
        / "2026_eval"
        / "results"
        / f"{experiment_name}_ep{checkpoint_label}_hf"
        / "main"
        / "zero_shot"
        / "causal",
    ]
    for causal_dir in candidates:
        if causal_dir.is_dir() and has_submeasure_inputs(causal_dir):
            return causal_dir

    results_dir = experiment_dir / "2026_eval" / "results"
    if results_dir.is_dir():
        discovered = sorted(
            path / "main" / "zero_shot" / "causal"
            for path in results_dir.iterdir()
            if path.is_dir() and path.name.endswith(f"_ep{checkpoint_label}_hf")
        )
        discovered = [path for path in discovered if path.is_dir() and has_submeasure_inputs(path)]
        if len(discovered) == 1:
            return discovered[0]
        if len(discovered) > 1:
            raise ValueError(
                f"Multiple checkpoint {checkpoint_label} causal eval roots found in {results_dir}"
            )
    return None


def has_submeasure_inputs(causal_dir: Path) -> bool:
    return (
        any(causal_dir.joinpath(*report_parts).is_file() for _task, report_parts in REPORT_SPECS)
        or (causal_dir / "reading" / "predictions.json").is_file()
        or (causal_dir / "reading" / "report.txt").is_file()
    )


def parse_report_sections(report_path: Path) -> list[tuple[str, str, float]]:
    if not report_path.is_file():
        return []

    rows: list[tuple[str, str, float]] = []
    current_section = ""
    lines = report_path.read_text(encoding="utf-8").splitlines()
    line_index = 0
    while line_index < len(lines):
        line = lines[line_index].strip()
        if line == "":
            line_index += 1
            continue
        if line.startswith("### "):
            current_section = normalize_section_name(line[4:])
            if current_section == "average_accuracy":
                if line_index + 1 >= len(lines):
                    raise ValueError(f"Missing average accuracy value in {report_path}")
                rows.append(("average", "average", float(lines[line_index + 1].strip())))
                line_index += 2
                continue
            line_index += 1
            continue
        if ":" in line:
            key, value = line.split(":", 1)
            if key.strip() != "TEMPERATURE":
                rows.append((current_section, key.strip(), float(value.strip())))
        line_index += 1
    return rows


def normalize_section_name(section: str) -> str:
    return section.strip().lower().replace(" ", "_")


def format_score(score: float) -> str:
    return f"{score:.6f}"


def make_row(
    experiment_name: str,
    task: str,
    submeasure_type: str,
    submeasure: str,
    score: float,
    n_items: int | None = None,
) -> dict[str, str]:
    return {
        "experiment_name": experiment_name,
        "experiment_group": experiment_group_from_name(experiment_name),
        "task": task,
        "submeasure_type": submeasure_type,
        "submeasure": submeasure,
        "score": format_score(score),
        "n_items": "" if n_items is None else str(n_items),
    }


def report_rows(experiment_name: str, causal_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for task, report_parts in REPORT_SPECS:
        report_path = causal_dir.joinpath(*report_parts)
        for submeasure_type, submeasure, score in parse_report_sections(report_path):
            rows.append(
                make_row(
                    experiment_name,
                    task,
                    submeasure_type,
                    submeasure,
                    score,
                )
            )
    return rows


def load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def reading_rows(
    experiment_name: str,
    causal_dir: Path,
    eval_data_root: Path,
) -> list[dict[str, str]]:
    predictions_path = causal_dir / "reading" / "predictions.json"
    if not predictions_path.is_file():
        if (causal_dir / "reading" / "report.txt").is_file():
            raise FileNotFoundError(f"Missing reading predictions: {predictions_path}")
        return []

    predictions = load_json(predictions_path)["reading"]["predictions"]
    data = pd.read_csv(eval_data_root / "reading" / "reading_data.csv", dtype={"item": str})
    if len(predictions) != len(data):
        raise ValueError(
            f"Reading prediction count {len(predictions)} does not match data rows {len(data)}"
        )

    data = data.copy()
    data["pred"] = [item["pred"] for item in predictions]
    data["prev_pred"] = [item["prev_pred"] for item in predictions]

    rows = []
    eye_scores = []
    for dv in READING_EYE_TRACKING_DVS:
        score, n_items = eye_tracking_score(data, dv)
        eye_scores.append(score)
        rows.append(make_row(experiment_name, "reading", "eye_tracking_dv", dv, score, n_items))

    rows.append(
        make_row(
            experiment_name,
            "reading",
            "aggregate",
            "eye_tracking_mean",
            sum(eye_scores) / len(eye_scores),
        )
    )

    score, n_items = self_paced_score(data)
    rows.append(
        make_row(
            experiment_name,
            "reading",
            "self_paced",
            "self_paced_reading_time",
            score,
            n_items,
        )
    )
    return rows


def eye_tracking_score(data: pd.DataFrame, dv: str) -> tuple[float, int]:
    baseline_columns = [dv, "Subtlex_log10", "length", "context_length"]
    model_columns = baseline_columns + ["pred"]
    baseline_data = data[baseline_columns].dropna()
    model_data = data[model_columns].dropna()
    baseline = smf.ols(
        formula=(
            f"{dv} ~ Subtlex_log10 + length + context_length + "
            "Subtlex_log10:length + Subtlex_log10:context_length + length:context_length"
        ),
        data=baseline_data,
    ).fit()
    model = smf.ols(
        formula=(
            f"{dv} ~ Subtlex_log10 + length + context_length + "
            "Subtlex_log10:length + Subtlex_log10:context_length + length:context_length + pred"
        ),
        data=model_data,
    ).fit()
    return incremental_r2_percent(float(baseline.rsquared), float(model.rsquared)), len(model_data)


def self_paced_score(data: pd.DataFrame) -> tuple[float, int]:
    baseline_columns = [
        "self_paced_reading_time",
        "Subtlex_log10",
        "length",
        "context_length",
        "prev_length",
        "prev_pred",
    ]
    model_columns = baseline_columns + ["pred"]
    baseline_data = data[baseline_columns].dropna()
    model_data = data[model_columns].dropna()
    baseline_terms = (
        "Subtlex_log10 + length + context_length + prev_length + prev_pred + "
        "Subtlex_log10:length + Subtlex_log10:context_length + Subtlex_log10:prev_length + "
        "Subtlex_log10:prev_pred + length:context_length + length:prev_length + "
        "length:prev_pred + context_length:prev_length + context_length:prev_pred + "
        "prev_length:prev_pred"
    )
    baseline = smf.ols(
        formula=f"self_paced_reading_time ~ {baseline_terms}",
        data=baseline_data,
    ).fit()
    model = smf.ols(
        formula=f"self_paced_reading_time ~ {baseline_terms} + pred",
        data=model_data,
    ).fit()
    return incremental_r2_percent(float(baseline.rsquared), float(model.rsquared)), len(model_data)


def incremental_r2_percent(baseline_r2: float, model_r2: float) -> float:
    if not math.isfinite(baseline_r2) or not math.isfinite(model_r2):
        return float("nan")
    if baseline_r2 >= 1.0:
        return float("nan")
    return 100.0 * ((model_r2 - baseline_r2) / (1.0 - baseline_r2))


def summarize_experiment(
    experiment_dir: Path,
    eval_data_root: Path,
    checkpoint_label: str,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    experiment_name = experiment_name_for_dir(experiment_dir)
    causal_dir = causal_dir_for_checkpoint(experiment_dir, experiment_name, checkpoint_label)
    if causal_dir is None:
        return rows
    rows.extend(report_rows(experiment_name, causal_dir))
    rows.extend(reading_rows(experiment_name, causal_dir, eval_data_root))
    return rows


def main() -> None:
    args = parse_args()
    if not args.eval_data_root.is_dir():
        raise ValueError(f"Eval data root does not exist: {args.eval_data_root}")

    rows = []
    for experiment_root in args.experiment_root:
        if not experiment_root.is_dir():
            raise ValueError(f"Experiment root does not exist: {experiment_root}")
        for experiment_dir in discover_experiment_dirs(experiment_root, args.checkpoint_label):
            rows.extend(
                summarize_experiment(
                    experiment_dir,
                    args.eval_data_root,
                    args.checkpoint_label,
                )
            )
    if not rows:
        raise ValueError("No zero-shot submeasures discovered in any experiment.")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} zero-shot submeasure rows to {args.output_csv}")


if __name__ == "__main__":
    main()
