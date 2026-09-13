#!/usr/bin/env python3
"""Analyze the RCA SwiGLU symbol-retrieval comparison."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean, stdev


DEFAULT_SYMBOL_MODES = (
    "relative",
    "relative_rope",
    "positional",
    "positional_sinusoidal",
    "symbolic",
    "relsymbolic",
    "relsymbolic_n4",
)
BASELINE_MODE = "relative"
EXPECTED_SEEDS = 5
METRICS = (
    "train_value",
    "blimp_full",
    "entity_tracking_full",
    "comps_full",
    "reading_full",
    "eye_tracking_full",
    "self_paced_full",
)
SUMMARY_FIELDS = [
    "symbol_retrieval",
    "experiment_group",
    "n_rows",
    "seed_names",
    "missing_seeds",
    "complete",
    "hidden_dim",
    "n_layers",
    "n_heads_sa",
    "n_heads_ra",
    "datapoint_length",
    "batch_size",
    "gradient_accumulation_steps",
    "ra_type",
    "ra_rel_activation",
    "ffn_activation",
    "positional_symbols_sinusoidal",
    "relative_symbols_rope",
    "relsymbolic_neighborhood_size",
]
DELTA_FIELDS = [
    "symbol_retrieval",
    "baseline_symbol_retrieval",
    "metric",
    "n_pairs",
    "seed_names",
    "delta_mean",
    "delta_std",
    "delta_min",
    "delta_max",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write RCA symbol-retrieval comparison summaries from experiment_matrix.csv."
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
        help="Directory where RCA symbol-comparison outputs should be written.",
    )
    parser.add_argument(
        "--run_tag",
        required=True,
        help="Run tag prefix shared by the symbol-retrieval runs to compare.",
    )
    parser.add_argument(
        "--expected_seeds",
        type=int,
        default=EXPECTED_SEEDS,
        help="Expected number of seeds per symbol-retrieval group.",
    )
    parser.add_argument(
        "--expected_seed_names",
        nargs="+",
        default=None,
        help="Explicit expected seeds, such as 0 2 4 or seed0 seed2 seed4.",
    )
    parser.add_argument(
        "--symbol_modes",
        nargs="+",
        default=DEFAULT_SYMBOL_MODES,
        help="Symbol-retrieval mode labels to include in the RCA comparison.",
    )
    return parser.parse_args()


def read_rows(input_csv: Path) -> list[dict[str, str]]:
    with input_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {input_csv}")
    return rows


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


def stdev_or_none(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    return stdev(values)


def seed_sort_key(seed_name: str) -> tuple[int, str]:
    if seed_name.startswith("seed") and seed_name[4:].isdigit():
        return int(seed_name[4:]), seed_name
    return 10**9, seed_name


def normalize_seed_name(seed_name: str) -> str:
    if seed_name.isdigit():
        return f"seed{seed_name}"
    if seed_name.startswith("seed") and seed_name[4:].isdigit():
        return seed_name
    raise ValueError(f"Expected seed names must be non-negative integers or seedN labels, got {seed_name!r}")


def expected_seed_names_from_args(
    expected_seeds: int,
    expected_seed_names: list[str] | None,
) -> tuple[str, ...]:
    if expected_seed_names is None:
        return tuple(f"seed{seed}" for seed in range(expected_seeds))
    normalized = tuple(normalize_seed_name(seed_name) for seed_name in expected_seed_names)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"expected_seed_names contains duplicates: {' '.join(expected_seed_names)}")
    if len(normalized) != expected_seeds:
        raise ValueError(
            "expected_seed_names length must match expected_seeds: "
            f"{len(normalized)} names for expected_seeds={expected_seeds}"
        )
    return tuple(sorted(normalized, key=seed_sort_key))


def expected_group(run_tag: str, symbol_mode: str) -> str:
    return f"{run_tag}_{symbol_mode}"


def filter_symbol_rows(
    rows: list[dict[str, str]],
    run_tag: str,
    symbol_modes: tuple[str, ...],
) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {mode: [] for mode in symbol_modes}
    expected_groups = {expected_group(run_tag, mode): mode for mode in symbol_modes}
    for row in rows:
        mode = expected_groups.get(row["experiment_group"])
        if mode is not None:
            grouped[mode].append(row)

    missing_modes = [mode for mode, mode_rows in grouped.items() if not mode_rows]
    if missing_modes:
        raise ValueError(
            "Missing RCA symbol-retrieval groups in matrix: "
            + ", ".join(expected_group(run_tag, mode) for mode in missing_modes)
        )
    return grouped


def metric_stats(rows: list[dict[str, str]], metric: str) -> dict[str, str]:
    values = [value for row in rows if (value := parse_float(row[metric])) is not None]
    return {
        f"{metric}_mean": format_float(mean(values) if values else None),
        f"{metric}_std": format_float(stdev_or_none(values)),
        f"{metric}_min": format_float(min(values) if values else None),
        f"{metric}_max": format_float(max(values) if values else None),
    }


def descriptor_value(rows: list[dict[str, str]], field: str) -> str:
    values = sorted({row.get(field, "") for row in rows if row.get(field, "") != ""})
    return "|".join(values)


def summary_rows(
    grouped: dict[str, list[dict[str, str]]],
    symbol_modes: tuple[str, ...],
    expected_seed_names: tuple[str, ...],
) -> list[dict[str, str]]:
    expected_seed_names = tuple(normalize_seed_name(s) for s in expected_seed_names)
    expected_seed_name_set = set(expected_seed_names)
    rows: list[dict[str, str]] = []
    for mode in symbol_modes:
        mode_rows = sorted(grouped[mode], key=lambda row: seed_sort_key(normalize_seed_name(row["seed_name"])))
        seed_names = {normalize_seed_name(row["seed_name"]) for row in mode_rows}
        missing_seed_names = sorted(expected_seed_name_set - seed_names, key=seed_sort_key)
        row = {
            "symbol_retrieval": mode,
            "experiment_group": expected_group_from_rows(mode_rows),
            "n_rows": str(len(mode_rows)),
            "seed_names": " ".join(sorted(seed_names, key=seed_sort_key)),
            "missing_seeds": " ".join(missing_seed_names),
            "complete": "true" if not missing_seed_names else "false",
            "hidden_dim": descriptor_value(mode_rows, "hidden_dim"),
            "n_layers": descriptor_value(mode_rows, "n_layers"),
            "n_heads_sa": descriptor_value(mode_rows, "n_heads_sa"),
            "n_heads_ra": descriptor_value(mode_rows, "n_heads_ra"),
            "datapoint_length": descriptor_value(mode_rows, "datapoint_length"),
            "batch_size": descriptor_value(mode_rows, "batch_size"),
            "gradient_accumulation_steps": descriptor_value(
                mode_rows,
                "gradient_accumulation_steps",
            ),
            "ra_type": descriptor_value(mode_rows, "ra_type"),
            "ra_rel_activation": descriptor_value(mode_rows, "ra_rel_activation"),
            "ffn_activation": descriptor_value(mode_rows, "ffn_activation"),
            "positional_symbols_sinusoidal": descriptor_value(
                mode_rows,
                "positional_symbols_sinusoidal",
            ),
            "relative_symbols_rope": descriptor_value(mode_rows, "relative_symbols_rope"),
            "relsymbolic_neighborhood_size": descriptor_value(
                mode_rows,
                "relsymbolic_neighborhood_size",
            ),
        }
        for metric in METRICS:
            row.update(metric_stats(mode_rows, metric))
        rows.append(row)
    return rows


def expected_group_from_rows(rows: list[dict[str, str]]) -> str:
    values = sorted({row["experiment_group"] for row in rows})
    if len(values) != 1:
        raise ValueError(f"Expected exactly one experiment_group, got {values}")
    return values[0]


def rows_by_seed(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    by_seed: dict[str, dict[str, str]] = {}
    for row in rows:
        seed_name = normalize_seed_name(row["seed_name"])
        if seed_name in by_seed:
            raise ValueError(f"Duplicate row for seed {seed_name}: {row['experiment_group']}")
        by_seed[seed_name] = row
    return by_seed


def paired_delta_rows(
    grouped: dict[str, list[dict[str, str]]],
    symbol_modes: tuple[str, ...],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    baseline_by_seed = rows_by_seed(grouped[BASELINE_MODE])
    paired_rows: list[dict[str, str]] = []
    summary: list[dict[str, str]] = []

    for mode in symbol_modes:
        if mode == BASELINE_MODE:
            continue
        mode_by_seed = rows_by_seed(grouped[mode])
        shared_seed_names = sorted(
            set(mode_by_seed) & set(baseline_by_seed),
            key=seed_sort_key,
        )
        for metric in METRICS:
            deltas: list[float] = []
            delta_seed_names: list[str] = []
            for seed_name in shared_seed_names:
                mode_value = parse_float(mode_by_seed[seed_name][metric])
                baseline_value = parse_float(baseline_by_seed[seed_name][metric])
                if mode_value is None or baseline_value is None:
                    continue
                delta = mode_value - baseline_value
                deltas.append(delta)
                delta_seed_names.append(seed_name)
                paired_rows.append(
                    {
                        "symbol_retrieval": mode,
                        "baseline_symbol_retrieval": BASELINE_MODE,
                        "seed_name": seed_name,
                        "metric": metric,
                        "delta": format_float(delta),
                        "score": format_float(mode_value),
                        "baseline_score": format_float(baseline_value),
                    }
                )
            summary.append(
                {
                    "symbol_retrieval": mode,
                    "baseline_symbol_retrieval": BASELINE_MODE,
                    "metric": metric,
                    "n_pairs": str(len(deltas)),
                    "seed_names": " ".join(delta_seed_names),
                    "delta_mean": format_float(mean(deltas) if deltas else None),
                    "delta_std": format_float(stdev_or_none(deltas)),
                    "delta_min": format_float(min(deltas) if deltas else None),
                    "delta_max": format_float(max(deltas) if deltas else None),
                }
            )
    return paired_rows, summary


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def latex_escape(value: str) -> str:
    return value.replace("_", r"\_")


def metric_label(metric: str) -> str:
    return {
        "train_value": "Train loss",
        "blimp_full": "BLiMP",
        "entity_tracking_full": "Entity",
        "comps_full": "COMPS",
        "reading_full": "Reading",
        "eye_tracking_full": "Eye",
        "self_paced_full": "SPR",
    }[metric]


def write_tex_tables(output_dir: Path, summaries: list[dict[str, str]], deltas: list[dict[str, str]]) -> None:
    summary_lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Symbols & $n$ & BLiMP & Entity & COMPS & Eye & SPR \\",
        r"\midrule",
    ]
    for row in summaries:
        summary_lines.append(
            " & ".join(
                [
                    latex_escape(row["symbol_retrieval"]),
                    row["n_rows"],
                    row["blimp_full_mean"] or "--",
                    row["entity_tracking_full_mean"] or "--",
                    row["comps_full_mean"] or "--",
                    row["eye_tracking_full_mean"] or "--",
                    row["self_paced_full_mean"] or "--",
                ]
            )
            + r" \\"
        )
    summary_lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    (output_dir / "summary_table.tex").write_text("\n".join(summary_lines), encoding="utf-8")

    delta_metrics = {
        "blimp_full",
        "entity_tracking_full",
        "comps_full",
        "eye_tracking_full",
        "self_paced_full",
    }
    delta_rows = [row for row in deltas if row["metric"] in delta_metrics]
    delta_lines = [
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Symbols & Metric & $n$ & $\Delta$ vs. relative \\",
        r"\midrule",
    ]
    for row in delta_rows:
        delta_lines.append(
            " & ".join(
                [
                    latex_escape(row["symbol_retrieval"]),
                    metric_label(row["metric"]),
                    row["n_pairs"],
                    row["delta_mean"] or "--",
                ]
            )
            + r" \\"
        )
    delta_lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    (output_dir / "paired_deltas.tex").write_text("\n".join(delta_lines), encoding="utf-8")


def write_report_section(
    output_dir: Path,
    run_tag: str,
    symbol_modes: tuple[str, ...],
    expected_seed_names: tuple[str, ...],
) -> None:
    escaped_run_tag = latex_escape(run_tag)
    mode_text = ", ".join(latex_escape(mode) for mode in symbol_modes)
    seed_text = ", ".join(latex_escape(seed_name) for seed_name in expected_seed_names)
    asset_prefix = output_dir.name
    model_asset_prefix = f"{asset_prefix}/model_outputs"
    section = rf"""
\subsection{{RCA Symbol Retrieval at 264 Tokens}}

The RCA/SwiGLU comparison fixes the 768-dimensional, 12-layer, 9SA/3RA architecture and varies the symbol source across: {mode_text}. The run tag is \texttt{{{escaped_run_tag}}}. The analysis expects seed-indexed experiment folders for: {seed_text}.

The comparative outcomes for these symbol-retrieval mechanisms are summarized in the tables and figures below. At the aggregate level, \texttt{{relative\_symbols\_rope}} and learned relative symbols are close on BLiMP and COMPS, whereas \texttt{{relsymbolic\_n4}} is lower on BLiMP and COMPS. Non-syntactic submeasure effects should be interpreted cautiously because several models are singular.

\begin{{table}}[htpb]
    \centering
    \input{{{asset_prefix}/summary_table.tex}}
    \caption{{RCA/SwiGLU symbol-retrieval comparison. Means are computed across the selected seeds.}}
    \label{{tab:rca_symbol_summary}}
\end{{table}}

\begin{{table}}[htpb]
    \centering
    \input{{{asset_prefix}/paired_deltas.tex}}
    \caption{{Paired seed deltas against the relative-symbol baseline.}}
    \label{{tab:rca_symbol_deltas}}
\end{{table}}

\IfFileExists{{{model_asset_prefix}/blimp_subtest_model_summary.tex}}{{%
\begin{{table}}[htpb]
    \centering
    \input{{{model_asset_prefix}/blimp_subtest_model_summary.tex}}
    \caption{{BLiMP subtest mixed-effects model for the RCA/SwiGLU symbol-retrieval comparison. The model includes random intercepts for seed and BLiMP subtest.}}
    \label{{tab:rca_symbol_blimp_lmm}}
\end{{table}}
}}{{}}

\IfFileExists{{{model_asset_prefix}/submeasure_model_summary.tex}}{{%
\begin{{table}}[htpb]
    \centering
    \input{{{model_asset_prefix}/submeasure_model_summary.tex}}
    \caption{{Zero-shot submeasure mixed-effects models by task family for the RCA/SwiGLU symbol-retrieval comparison. Each model includes random intercepts for seed and submeasure.}}
    \label{{tab:rca_symbol_submeasure_lmms}}
\end{{table}}
}}{{}}

\IfFileExists{{{model_asset_prefix}/metric_model_summary.tex}}{{%
\begin{{table}}[htpb]
    \centering
    \input{{{model_asset_prefix}/metric_model_summary.tex}}
    \caption{{Metric-wise mixed-effects models for the RCA/SwiGLU symbol-retrieval comparison. Each model includes a random intercept for seed.}}
    \label{{tab:rca_symbol_metric_lmms}}
\end{{table}}
}}{{}}

\IfFileExists{{{model_asset_prefix}/metric_vs_relative.tex}}{{%
\begin{{table}}[htpb]
    \centering
    \input{{{model_asset_prefix}/metric_vs_relative.tex}}
    \caption{{Estimated contrasts against the relative-symbol baseline for complete aggregate metrics.}}
    \label{{tab:rca_symbol_metric_relative_contrasts}}
\end{{table}}
}}{{}}

\IfFileExists{{{model_asset_prefix}/rca_symbol_metric_means.png}}{{%
\begin{{figure}}[htpb]
    \centering
    \includegraphics[width=\textwidth]{{{model_asset_prefix}/rca_symbol_metric_means.png}}
    \caption{{Aggregate 2026 zero-shot scores by symbol-retrieval mechanism for the RCA/SwiGLU 264-token comparison. Error bars show one standard error across seeds.}}
    \label{{fig:rca_symbol_metric_means}}
\end{{figure}}
}}{{}}

\IfFileExists{{{model_asset_prefix}/rca_symbol_submeasure_task_means.png}}{{%
\begin{{figure}}[htpb]
    \centering
    \includegraphics[width=\textwidth]{{{model_asset_prefix}/rca_symbol_submeasure_task_means.png}}
    \caption{{Zero-shot submeasure scores by task family and symbol-retrieval mechanism for the RCA/SwiGLU 264-token comparison. Error bars show one standard error across submeasure rows.}}
    \label{{fig:rca_symbol_submeasure_task_means}}
\end{{figure}}
}}{{}}

\IfFileExists{{{model_asset_prefix}/rca_symbol_blimp_ling_terms.png}}{{%
\begin{{figure}}[htpb]
    \centering
    \includegraphics[width=\textwidth]{{{model_asset_prefix}/rca_symbol_blimp_ling_terms.png}}
    \caption{{BLiMP linguistic-domain accuracies by symbol-retrieval mechanism for the RCA/SwiGLU 264-token comparison. Error bars show one standard error across seeds.}}
    \label{{fig:rca_symbol_blimp_ling_terms}}
\end{{figure}}
}}{{}}
"""
    for task, display_name in [
        ("blimp", "BLiMP"),
        ("blimp_supplement", "BLiMP Supplement"),
        ("comps", "COMPS"),
        ("entity_tracking", "Entity Tracking"),
        ("ewok", "EWoK"),
        ("reading", "Reading"),
    ]:
        section += rf"""
\IfFileExists{{{model_asset_prefix}/rca_symbol_{task}_submeasures.png}}{{%
\begin{{figure}}[htpb]
    \centering
    \includegraphics[width=\textwidth,height=0.85\textheight,keepaspectratio]{{{model_asset_prefix}/rca_symbol_{task}_submeasures.png}}
    \caption{{{display_name} individual submeasure scores by symbol-retrieval mechanism for the RCA/SwiGLU 264-token comparison. Error bars show one standard error across seeds.}}
    \label{{fig:rca_symbol_{task}_submeasures}}
\end{{figure}}
}}{{}}
"""
    (output_dir / "report_section.tex").write_text(section.strip() + "\n", encoding="utf-8")


def normalize_symbol_modes(symbol_modes: list[str]) -> tuple[str, ...]:
    normalized = tuple(symbol_modes)
    if BASELINE_MODE not in normalized:
        raise ValueError(f"symbol_modes must include baseline mode {BASELINE_MODE!r}")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"symbol_modes contains duplicates: {' '.join(symbol_modes)}")
    return normalized


def analyze(
    input_csv: Path,
    output_dir: Path,
    run_tag: str,
    expected_seed_names: tuple[str, ...],
    symbol_modes: tuple[str, ...],
) -> None:
    rows = read_rows(input_csv)
    grouped = filter_symbol_rows(rows, run_tag, symbol_modes)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = summary_rows(grouped, symbol_modes, expected_seed_names)
    paired_rows, deltas = paired_delta_rows(grouped, symbol_modes)

    summary_fields = [
        *SUMMARY_FIELDS,
        *(f"{metric}_{suffix}" for metric in METRICS for suffix in ("mean", "std", "min", "max")),
    ]
    write_csv(output_dir / "group_summary.csv", summary_fields, summaries)
    write_csv(
        output_dir / "paired_seed_deltas.csv",
        [
            "symbol_retrieval",
            "baseline_symbol_retrieval",
            "seed_name",
            "metric",
            "delta",
            "score",
            "baseline_score",
        ],
        paired_rows,
    )
    write_csv(output_dir / "paired_delta_summary.csv", DELTA_FIELDS, deltas)
    write_tex_tables(output_dir, summaries, deltas)
    write_report_section(output_dir, run_tag, symbol_modes, expected_seed_names)


def main() -> None:
    args = parse_args()
    symbol_modes = normalize_symbol_modes(args.symbol_modes)
    expected_seed_names = expected_seed_names_from_args(args.expected_seeds, args.expected_seed_names)
    analyze(args.input_csv, args.output_dir, args.run_tag, expected_seed_names, symbol_modes)
    print(f"Wrote RCA symbol comparison analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
