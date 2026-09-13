"""Export compact RCA symbol-comparison results without checkpoint weights."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train.portable_config import load_train_config, portable_train_config, write_train_config


DEFAULT_SYMBOL_MODES = [
    "relative",
    "relative_rope",
    "positional",
    "positional_sinusoidal",
    "symbolic",
    "relsymbolic",
    "relsymbolic_n4",
]
REPORT_FILES = [
    Path("blimp") / "blimp_filtered" / "best_temperature_report.txt",
    Path("blimp") / "supplement_filtered" / "best_temperature_report.txt",
    Path("ewok") / "ewok_filtered" / "best_temperature_report.txt",
    Path("entity_tracking") / "entity_tracking" / "best_temperature_report.txt",
    Path("comps") / "comps" / "best_temperature_report.txt",
    Path("reading") / "report.txt",
]
PREDICTION_FILES = [
    Path("blimp") / "blimp_filtered" / "predictions.json",
    Path("blimp") / "supplement_filtered" / "predictions.json",
    Path("ewok") / "ewok_filtered" / "predictions.json",
    Path("entity_tracking") / "entity_tracking" / "predictions.json",
    Path("comps") / "comps" / "predictions.json",
    Path("reading") / "predictions.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export shareable RCA symbol-comparison configs, scores, and report files."
    )
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--checkpoint-label", default="9")
    parser.add_argument("--expected-seeds", type=int, default=5)
    parser.add_argument("--seed-names", nargs="+", default=None)
    parser.add_argument("--symbol-modes", nargs="+", default=DEFAULT_SYMBOL_MODES)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def seed_values(args: argparse.Namespace) -> list[str]:
    values = args.seed_names
    if values is None:
        values = [str(seed) for seed in range(args.expected_seeds)]
    if not values:
        raise ValueError("At least one seed is required")
    duplicate_values = sorted({seed for seed in values if values.count(seed) > 1})
    if duplicate_values:
        raise ValueError(f"Duplicate seed values: {', '.join(duplicate_values)}")
    for value in values:
        if not value.isdigit():
            raise ValueError(f"Seed values must be non-negative integers, got {value!r}")
    return values


def symbol_groups(run_tag: str, symbol_modes: list[str]) -> set[str]:
    if not symbol_modes:
        raise ValueError("At least one symbol mode is required")
    duplicate_modes = sorted({mode for mode in symbol_modes if symbol_modes.count(mode) > 1})
    if duplicate_modes:
        raise ValueError(f"Duplicate symbol modes: {', '.join(duplicate_modes)}")
    return {f"{run_tag}_{mode}" for mode in symbol_modes}


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {output_dir}")
        if not output_dir.is_dir():
            raise NotADirectoryError(f"Output path exists and is not a directory: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)


def copy_required_file(source: Path, target: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Required file not found: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def filter_csv_by_groups(
    input_path: Path,
    output_path: Path,
    selected_groups: set[str],
    required_experiments: set[str],
) -> None:
    if not input_path.is_file():
        raise FileNotFoundError(f"Required CSV not found: {input_path}")
    with input_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {input_path}")
        rows = [
            row
            for row in reader
            if row["experiment_group"] in selected_groups
        ]
    if not rows:
        raise ValueError(f"No RCA rows found in {input_path}")
    observed_experiments = {row["experiment_name"] for row in rows}
    missing = sorted(required_experiments - observed_experiments)
    if missing:
        raise ValueError(f"{input_path} is missing RCA experiments: {', '.join(missing)}")
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_summary_artifacts(analysis_dir: Path, output_dir: Path) -> None:
    comparison_dir = analysis_dir / "rca_symbol_retrieval_comparison"
    for name in [
        "group_summary.csv",
        "paired_seed_deltas.csv",
        "paired_delta_summary.csv",
    ]:
        copy_required_file(comparison_dir / name, output_dir / name)

    metric_plot_dir = comparison_dir / "model_outputs"
    plot_paths = sorted(metric_plot_dir.glob("*.png"))
    if not plot_paths:
        raise FileNotFoundError(f"No RCA symbol plot PNGs found in {metric_plot_dir}")
    for plot_path in plot_paths:
        copy_required_file(plot_path, output_dir / plot_path.name)


def official_eval_root(experiment_dir: Path, experiment_name: str, checkpoint_label: str) -> Path:
    hf_name = f"{experiment_name}_ep{checkpoint_label}_hf"
    return (
        experiment_dir
        / "2026_eval"
        / "results"
        / hf_name
        / "main"
        / "zero_shot"
        / "causal"
    )


def export_experiment(
    experiment_root: Path,
    output_dir: Path,
    run_tag: str,
    symbol_mode: str,
    seed: str,
    checkpoint_label: str,
) -> dict[str, str]:
    experiment_name = f"{run_tag}_{symbol_mode}_seed{seed}"
    experiment_dir = experiment_root / experiment_name
    logging_dir = experiment_dir / "logging"
    target_root = output_dir / experiment_name
    target_logging = target_root / "logging"

    if not logging_dir.is_dir():
        raise FileNotFoundError(f"Missing experiment logging directory: {logging_dir}")
    copy_required_file(logging_dir / "model_config.yaml", target_logging / "model_config.yaml")
    write_train_config(
        target_logging / "train_config.yaml",
        portable_train_config(load_train_config(logging_dir / "train_config.yaml")),
    )

    source_summary = experiment_dir / "2026_eval" / "summarize_2026_eval.csv"
    target_summary = target_root / "2026_eval" / "summarize_2026_eval.csv"
    copy_required_file(source_summary, target_summary)

    eval_root = official_eval_root(experiment_dir, experiment_name, checkpoint_label)
    target_reports = target_root / "official_reports"
    for eval_file in [*REPORT_FILES, *PREDICTION_FILES]:
        copy_required_file(eval_root / eval_file, target_reports / eval_file)

    return {
        "experiment_name": experiment_name,
        "symbol_mode": symbol_mode,
        "seed": seed,
        "checkpoint_label": checkpoint_label,
        "train_config": str((target_logging / "train_config.yaml").relative_to(output_dir)),
        "model_config": str((target_logging / "model_config.yaml").relative_to(output_dir)),
        "summary_csv": str(target_summary.relative_to(output_dir)),
        "eval_reports_dir": str(target_reports.relative_to(output_dir)),
    }


def write_manifest(output_dir: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "experiment_name",
        "symbol_mode",
        "seed",
        "checkpoint_label",
        "train_config",
        "model_config",
        "summary_csv",
        "eval_reports_dir",
    ]
    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    seeds = seed_values(args)
    selected_groups = symbol_groups(args.run_tag, args.symbol_modes)
    expected_experiments = {
        f"{args.run_tag}_{symbol_mode}_seed{seed}"
        for symbol_mode in args.symbol_modes
        for seed in seeds
    }

    prepare_output_dir(args.output_dir, args.overwrite)
    export_summary_artifacts(args.analysis_dir, args.output_dir)
    filter_csv_by_groups(
        args.analysis_dir / "long_scores.csv",
        args.output_dir / "long_scores.csv",
        selected_groups,
        expected_experiments,
    )
    filter_csv_by_groups(
        args.analysis_dir / "zero_shot_submeasures.csv",
        args.output_dir / "zero_shot_submeasures.csv",
        selected_groups,
        expected_experiments,
    )

    rows = [
        export_experiment(
            experiment_root=args.experiment_root,
            output_dir=args.output_dir,
            run_tag=args.run_tag,
            symbol_mode=symbol_mode,
            seed=seed,
            checkpoint_label=args.checkpoint_label,
        )
        for symbol_mode in args.symbol_modes
        for seed in seeds
    ]
    write_manifest(args.output_dir, rows)
    print(f"Exported {len(rows)} RCA symbol-comparison runs to {args.output_dir}")


if __name__ == "__main__":
    main()
