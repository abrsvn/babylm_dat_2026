#!/usr/bin/env python3
"""Analyze the multi-seed BabyLM DAT experiment matrix."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


DEFAULT_EXPECTED_GROUPS = 47
DEFAULT_EXPECTED_SEEDS = 5

EVAL_METRICS = [
    "blimp_full",
    "entity_tracking_full",
    "comps_full",
    "reading_full",
    "eye_tracking_full",
    "self_paced_full",
]
SUMMARY_METRICS = ["train_value", *EVAL_METRICS]
DERIVED_DESCRIPTOR_FIELDS = ["architecture", "training_condition"]
LONG_SCORE_FIELDS = [
    "experiment_name",
    "experiment_group",
    "seed_name",
    "analysis_family",
    "seed",
    "metric",
    "score",
    "score_z",
    "metric_mean",
    "metric_std",
    *DERIVED_DESCRIPTOR_FIELDS,
    "model_type",
    "objective",
    "tie_lm_head",
    "hidden_dim",
    "n_layers",
    "n_heads_total",
    "n_heads_sa",
    "n_heads_ra",
    "batch_size",
    "corpus_id",
    "datapoint_length",
    "sequence_boundary_policy",
    "symbol_retrieval",
    "positional_symbols_sinusoidal",
    "relative_symbols_rope",
    "symbolic_attn_n_heads",
    "symbolic_use_bias",
    "relsymbolic_neighborhood_size",
    "ra_type",
    "ra_rel_activation",
    "ffn_activation",
    "init_scheme",
]
DESCRIPTOR_FIELDS = [
    "analysis_family",
    *DERIVED_DESCRIPTOR_FIELDS,
    "model_type",
    "objective",
    "tie_lm_head",
    "hidden_dim",
    "n_layers",
    "n_heads_total",
    "n_heads_sa",
    "n_heads_ra",
    "batch_size",
    "corpus_id",
    "datapoint_length",
    "sequence_boundary_policy",
    "symbol_retrieval",
    "positional_symbols_sinusoidal",
    "relative_symbols_rope",
    "symbolic_attn_n_heads",
    "symbolic_use_bias",
    "relsymbolic_neighborhood_size",
    "ra_type",
    "ra_rel_activation",
    "ffn_activation",
    "init_scheme",
    "nextlat_horizon",
    "nextlat_lambda_mse",
    "nextlat_lambda_kl",
    "nextlat_lambda_ce",
    "nextlat_proj_factor",
    "train_value_type",
]
OBJECTIVE_MATCH_FIELDS = [
    "analysis_family",
    "model_type",
    "tie_lm_head",
    "hidden_dim",
    "n_layers",
    "n_heads_total",
    "n_heads_sa",
    "n_heads_ra",
    "batch_size",
    "corpus_id",
    "datapoint_length",
    "sequence_boundary_policy",
    "symbol_retrieval",
    "positional_symbols_sinusoidal",
    "relative_symbols_rope",
    "symbolic_attn_n_heads",
    "symbolic_use_bias",
    "ra_type",
    "ra_rel_activation",
    "ffn_activation",
    "init_scheme",
]
TIE_MATCH_FIELDS = [
    field for field in OBJECTIVE_MATCH_FIELDS if field != "tie_lm_head"
]
REQUIRED_FIELDS = [
    "experiment_name",
    "experiment_group",
    "seed_name",
    "analysis_family",
    "seed",
    *(field for field in DESCRIPTOR_FIELDS if field not in DERIVED_DESCRIPTOR_FIELDS),
    *SUMMARY_METRICS,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write completeness, aggregate, and matched-delta analysis files."
    )
    parser.add_argument(
        "--input_csv",
        type=Path,
        required=True,
        help="Experiment matrix CSV produced by summarize_experiment_matrix.py.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory where analysis CSV/Markdown/TeX files should be written.",
    )
    parser.add_argument(
        "--expected_groups",
        type=int,
        default=DEFAULT_EXPECTED_GROUPS,
        help="Expected number of experiment groups.",
    )
    parser.add_argument(
        "--expected_seeds",
        type=int,
        default=DEFAULT_EXPECTED_SEEDS,
        help="Expected number of seeds per experiment group.",
    )
    return parser.parse_args()


def read_rows(input_csv: Path) -> list[dict[str, str]]:
    with input_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing_fields = sorted(set(REQUIRED_FIELDS) - set(reader.fieldnames or []))
        if missing_fields:
            raise ValueError(
                f"{input_csv} is missing required columns: {', '.join(missing_fields)}"
            )
        return list(reader)


def parse_float(value: str) -> float | None:
    if value == "":
        return None
    parsed = float(value)
    if math.isnan(parsed):
        return None
    return parsed


def format_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return mean(values)


def stdev_or_none(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    return stdev(values)


def group_rows(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["experiment_group"]].append(row)
    return dict(sorted(grouped.items()))


def numeric_values(rows: list[dict[str, str]], metric: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = parse_float(row[metric])
        if value is not None:
            values.append(value)
    return values


def architecture_label(row: dict[str, str]) -> str:
    if row["model_type"] == "self_attention":
        return "lm"
    head_split = f"{row['n_heads_sa']}sa_{row['n_heads_ra']}ra"
    return (
        f"dat_{head_split}_{row['symbol_retrieval']}_"
        f"{row['ffn_activation']}_{row['ra_type']}"
    )


def training_condition_label(row: dict[str, str]) -> str:
    if row["objective"] == "NTP" and row["tie_lm_head"] == "true":
        return "tied_ntp"
    if row["objective"] == "NTP" and row["tie_lm_head"] == "false":
        return "untied_ntp"
    if row["objective"] == "NextLat" and row["tie_lm_head"] == "false":
        return "untied_nextlat"
    return f"{row['tie_lm_head']}_{row['objective']}".lower()


def add_derived_fields(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    enriched_rows: list[dict[str, str]] = []
    for row in rows:
        enriched = dict(row)
        enriched["architecture"] = architecture_label(row)
        enriched["training_condition"] = training_condition_label(row)
        enriched_rows.append(enriched)
    return enriched_rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def seed_sort_key(seed_name: str) -> tuple[int, str]:
    if seed_name.startswith("seed") and seed_name[4:].isdigit():
        return int(seed_name[4:]), seed_name
    return 10**9, seed_name


def write_completeness(
    grouped: dict[str, list[dict[str, str]]],
    output_dir: Path,
    expected_groups: int,
    expected_seeds: int,
) -> None:
    expected_seed_names = [f"seed{seed}" for seed in range(expected_seeds)]
    rows: list[dict[str, str]] = []
    for experiment_group, group_rows_ in grouped.items():
        seed_names = sorted(
            {row["seed_name"] for row in group_rows_},
            key=seed_sort_key,
        )
        missing = [seed for seed in expected_seed_names if seed not in seed_names]
        duplicate_count = len(group_rows_) - len(seed_names)
        rows.append(
            {
                "experiment_group": experiment_group,
                "row_count": str(len(group_rows_)),
                "seed_count": str(len(seed_names)),
                "seeds": " ".join(seed_names),
                "missing_seeds": " ".join(missing),
                "duplicate_seed_rows": str(duplicate_count),
                "complete": (
                    "true"
                    if len(seed_names) == expected_seeds and not missing and duplicate_count == 0
                    else "false"
                ),
            }
        )

    write_csv(
        output_dir / "completeness.csv",
        [
            "experiment_group",
            "row_count",
            "seed_count",
            "seeds",
            "missing_seeds",
            "duplicate_seed_rows",
            "complete",
        ],
        rows,
    )

    complete_groups = sum(row["complete"] == "true" for row in rows)
    summary_lines = [
        f"input_groups: {len(grouped)}",
        f"expected_groups: {expected_groups}",
        f"complete_groups: {complete_groups}",
        f"expected_seeds_per_group: {expected_seeds}",
        f"total_rows: {sum(len(group_rows_) for group_rows_ in grouped.values())}",
        f"expected_total_rows: {expected_groups * expected_seeds}",
    ]
    (output_dir / "completeness_summary.txt").write_text("\n".join(summary_lines) + "\n")


def write_long_scores(rows: list[dict[str, str]], output_dir: Path) -> None:
    metric_values = {
        metric: [
            parsed
            for row in rows
            if (parsed := parse_float(row[metric])) is not None
        ]
        for metric in EVAL_METRICS
    }
    metric_means = {
        metric: mean_or_none(values) for metric, values in metric_values.items()
    }
    metric_stds = {
        metric: stdev_or_none(values) for metric, values in metric_values.items()
    }

    long_rows: list[dict[str, str]] = []
    for row in rows:
        for metric in EVAL_METRICS:
            score = parse_float(row[metric])
            if score is None:
                continue

            metric_mean = metric_means[metric]
            metric_std = metric_stds[metric]
            score_z = None
            if metric_mean is not None and metric_std is not None and metric_std > 0.0:
                score_z = (score - metric_mean) / metric_std

            long_row = {
                "experiment_name": row["experiment_name"],
                "experiment_group": row["experiment_group"],
                "seed_name": row["seed_name"],
                "seed": row["seed"],
                "metric": metric,
                "score": format_float(score),
                "score_z": format_float(score_z),
                "metric_mean": format_float(metric_mean),
                "metric_std": format_float(metric_std),
            }
            for field in LONG_SCORE_FIELDS:
                if field not in long_row and field in row:
                    long_row[field] = row[field]
            long_rows.append(long_row)

    write_csv(output_dir / "long_scores.csv", LONG_SCORE_FIELDS, long_rows)


def build_group_summaries(
    grouped: dict[str, list[dict[str, str]]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for experiment_group, group_rows_ in grouped.items():
        first = group_rows_[0]
        summary = {
            "experiment_group": experiment_group,
            "n_rows": str(len(group_rows_)),
            "seed_names": " ".join(
                sorted({row["seed_name"] for row in group_rows_}, key=seed_sort_key)
            ),
        }
        for field in DESCRIPTOR_FIELDS:
            values = sorted({row[field] for row in group_rows_})
            summary[field] = values[0] if len(values) == 1 else "|".join(values)
        summary["example_experiment_name"] = first["experiment_name"]

        for metric in SUMMARY_METRICS:
            values = numeric_values(group_rows_, metric)
            summary[f"{metric}_mean"] = format_float(mean_or_none(values))
            summary[f"{metric}_std"] = format_float(stdev_or_none(values))
            summary[f"{metric}_min"] = format_float(min(values) if values else None)
            summary[f"{metric}_max"] = format_float(max(values) if values else None)

        rows.append(summary)
    return rows


def group_summary_fields() -> list[str]:
    fields = [
        "experiment_group",
        "n_rows",
        "seed_names",
        *DESCRIPTOR_FIELDS,
        "example_experiment_name",
    ]
    for metric in SUMMARY_METRICS:
        fields.extend(
            [
                f"{metric}_mean",
                f"{metric}_std",
                f"{metric}_min",
                f"{metric}_max",
            ]
        )
    return fields


def write_group_summary(
    grouped: dict[str, list[dict[str, str]]],
    output_dir: Path,
) -> list[dict[str, str]]:
    rows = build_group_summaries(grouped)
    write_csv(output_dir / "group_summary.csv", group_summary_fields(), rows)
    return rows


def summary_signature(row: dict[str, str], fields: list[str]) -> tuple[str, ...]:
    return tuple(row[field] for field in fields)


def summary_metric_mean(row: dict[str, str], metric: str) -> float | None:
    return parse_float(row[f"{metric}_mean"])


def build_best_by_metric(summary_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for metric in EVAL_METRICS:
        metric_rows = [
            row for row in summary_rows if summary_metric_mean(row, metric) is not None
        ]
        sorted_rows = sorted(
            metric_rows,
            key=lambda row: summary_metric_mean(row, metric) or float("-inf"),
            reverse=True,
        )
        for rank, row in enumerate(sorted_rows, start=1):
            rows.append(
                {
                    "metric": metric,
                    "rank": str(rank),
                    "experiment_group": row["experiment_group"],
                    "analysis_family": row["analysis_family"],
                    "mean": row[f"{metric}_mean"],
                    "std": row[f"{metric}_std"],
                    "n_rows": row["n_rows"],
                    "model_type": row["model_type"],
                    "objective": row["objective"],
                    "tie_lm_head": row["tie_lm_head"],
                    "n_heads_sa": row["n_heads_sa"],
                    "n_heads_ra": row["n_heads_ra"],
                    "symbol_retrieval": row["symbol_retrieval"],
                    "positional_symbols_sinusoidal": row["positional_symbols_sinusoidal"],
                    "relative_symbols_rope": row["relative_symbols_rope"],
                    "ra_type": row["ra_type"],
                    "ffn_activation": row["ffn_activation"],
                }
            )
    return rows


def write_best_by_metric(summary_rows: list[dict[str, str]], output_dir: Path) -> None:
    fieldnames = [
        "metric",
        "rank",
        "experiment_group",
        "analysis_family",
        "mean",
        "std",
        "n_rows",
        "model_type",
        "objective",
        "tie_lm_head",
        "n_heads_sa",
        "n_heads_ra",
        "symbol_retrieval",
        "positional_symbols_sinusoidal",
        "relative_symbols_rope",
        "ra_type",
        "ffn_activation",
    ]
    write_csv(output_dir / "best_by_metric.csv", fieldnames, build_best_by_metric(summary_rows))


def build_matched_deltas(summary_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    ntp_by_objective_signature = {
        summary_signature(row, OBJECTIVE_MATCH_FIELDS): row
        for row in summary_rows
        if row["objective"] == "NTP"
    }
    for row in summary_rows:
        if row["objective"] != "NextLat":
            continue
        baseline = ntp_by_objective_signature.get(
            summary_signature(row, OBJECTIVE_MATCH_FIELDS)
        )
        if baseline is not None:
            rows.extend(delta_rows("nextlat_minus_ntp", row, baseline))

    tied_ntp_by_signature = {
        summary_signature(row, TIE_MATCH_FIELDS): row
        for row in summary_rows
        if row["objective"] == "NTP" and row["tie_lm_head"] == "true"
    }
    for row in summary_rows:
        if row["objective"] != "NTP" or row["tie_lm_head"] != "false":
            continue
        baseline = tied_ntp_by_signature.get(summary_signature(row, TIE_MATCH_FIELDS))
        if baseline is not None:
            rows.extend(delta_rows("untied_minus_tied_ntp", row, baseline))

    return rows


def delta_rows(
    comparison: str,
    experiment: dict[str, str],
    baseline: dict[str, str],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for metric in EVAL_METRICS:
        experiment_mean = summary_metric_mean(experiment, metric)
        baseline_mean = summary_metric_mean(baseline, metric)
        if experiment_mean is None or baseline_mean is None:
            continue
        rows.append(
            {
                "comparison": comparison,
                "metric": metric,
                "experiment_group": experiment["experiment_group"],
                "baseline_group": baseline["experiment_group"],
                "analysis_family": experiment["analysis_family"],
                "experiment_mean": format_float(experiment_mean),
                "baseline_mean": format_float(baseline_mean),
                "delta_mean": format_float(experiment_mean - baseline_mean),
                "experiment_n": experiment["n_rows"],
                "baseline_n": baseline["n_rows"],
                "model_type": experiment["model_type"],
                "n_heads_sa": experiment["n_heads_sa"],
                "n_heads_ra": experiment["n_heads_ra"],
                "symbol_retrieval": experiment["symbol_retrieval"],
                "ra_type": experiment["ra_type"],
                "ffn_activation": experiment["ffn_activation"],
            }
        )
    return rows


def write_matched_deltas(summary_rows: list[dict[str, str]], output_dir: Path) -> None:
    fieldnames = [
        "comparison",
        "metric",
        "experiment_group",
        "baseline_group",
        "analysis_family",
        "experiment_mean",
        "baseline_mean",
        "delta_mean",
        "experiment_n",
        "baseline_n",
        "model_type",
        "n_heads_sa",
        "n_heads_ra",
        "symbol_retrieval",
        "ra_type",
        "ffn_activation",
    ]
    write_csv(output_dir / "matched_deltas.csv", fieldnames, build_matched_deltas(summary_rows))


def markdown_escape(text: str) -> str:
    return text.replace("|", "\\|")


def write_markdown_summary(summary_rows: list[dict[str, str]], output_dir: Path) -> None:
    rows = sorted(
        summary_rows,
        key=lambda row: parse_float(row["blimp_full_mean"]) or float("-inf"),
        reverse=True,
    )
    headers = [
        "Experiment group",
        "N",
        "BLiMP",
        "Entity",
        "COMPS",
        "Reading",
        "WUG adj",
        "WUG past",
    ]
    lines = [
        "# Experiment Matrix Summary",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    markdown_escape(row["experiment_group"]),
                    row["n_rows"],
                    row["blimp_full_mean"],
                    row["entity_tracking_full_mean"],
                    row["comps_full_mean"],
                    row["reading_full_mean"],
                ]
            )
            + " |"
        )
    (output_dir / "summary_table.md").write_text("\n".join(lines) + "\n")


def latex_escape(text: str) -> str:
    replacements = {
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
    }
    return "".join(replacements.get(char, char) for char in text)


def write_latex_summary(summary_rows: list[dict[str, str]], output_dir: Path) -> None:
    rows = sorted(
        summary_rows,
        key=lambda row: parse_float(row["blimp_full_mean"]) or float("-inf"),
        reverse=True,
    )
    lines = [
        "\\begin{table*}[t]",
        "\\centering\\scriptsize",
        "\\begin{tabular}{@{}lrrrrrrr@{}}",
        "\\toprule",
        "Model & N & BLiMP & Entity & COMPS & Reading & WUG Adj & WUG Past \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    latex_escape(row["experiment_group"]),
                    row["n_rows"],
                    row["blimp_full_mean"],
                    row["entity_tracking_full_mean"],
                    row["comps_full_mean"],
                    row["reading_full_mean"],
                ]
            )
            + " \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Full-evaluation means by experiment group.}",
            "\\label{tab:babylm_dat_experiment_matrix}",
            "\\end{table*}",
        ]
    )
    (output_dir / "summary_table.tex").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    rows = add_derived_fields(read_rows(args.input_csv))
    grouped = group_rows(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    write_completeness(
        grouped,
        args.output_dir,
        args.expected_groups,
        args.expected_seeds,
    )
    write_long_scores(rows, args.output_dir)
    summary_rows = write_group_summary(grouped, args.output_dir)
    write_best_by_metric(summary_rows, args.output_dir)
    write_matched_deltas(summary_rows, args.output_dir)
    write_markdown_summary(summary_rows, args.output_dir)
    write_latex_summary(summary_rows, args.output_dir)

    print(f"Wrote analysis for {len(rows)} rows and {len(grouped)} groups to {args.output_dir}")


if __name__ == "__main__":
    main()
