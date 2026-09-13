"""Summarize committed RCA self-attention/relational-attention ratio results."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any

import yaml

from summarize_zero_shot_submeasures import (
    CSV_FIELDS as SUBMEASURE_CSV_FIELDS,
    reading_rows,
    report_rows,
)


REPORT_PATHS = {
    "blimp_full": Path("blimp") / "blimp_filtered" / "best_temperature_report.txt",
    "blimp_supplement": Path("blimp") / "supplement_filtered" / "best_temperature_report.txt",
    "ewok_full": Path("ewok") / "ewok_filtered" / "best_temperature_report.txt",
    "entity_tracking_full": Path("entity_tracking") / "entity_tracking" / "best_temperature_report.txt",
    "comps_full": Path("comps") / "comps" / "best_temperature_report.txt",
}
METRIC_LABELS = {
    "blimp_full": "BLiMP",
    "blimp_supplement": "BLiMP supplement",
    "ewok_full": "EWoK",
    "comps_full": "COMPS",
    "entity_tracking_full": "Entity tracking",
    "reading_mean": "Reading",
}
CSV_FIELDS = [
    "context_length",
    "experiment_name",
    "n_heads_sa",
    "n_heads_ra",
    "ra_head_fraction",
    "datapoint_length",
    "max_seq_len",
    "blimp_full",
    "blimp_supplement",
    "ewok_full",
    "entity_tracking_full",
    "comps_full",
    "eye_tracking",
    "self_paced_reading",
    "reading_mean",
    "source_dir",
]
SCORES_CSV_NAME = "ra_sa_ratio_scores.csv"
SUBMEASURES_CSV_NAME = "ra_sa_ratio_submeasures.csv"
SUBMEASURE_FIELDS = [
    "context_length",
    "n_heads_sa",
    "n_heads_ra",
    "ra_head_fraction",
    "datapoint_length",
    "max_seq_len",
    "source_dir",
    *SUBMEASURE_CSV_FIELDS,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize committed RCA SA/RA head-ratio evaluation results."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Root of the SA/RA head-ratio eval-output tree to summarize.",
    )
    parser.add_argument(
        "--eval-data-root",
        type=Path,
        required=True,
        help="Official full_eval data directory used for reading submeasures "
        "(required; the evaluation data is not part of this release).",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_yaml_mapping(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return raw


def read_average_accuracy(path: Path) -> float:
    if not path.is_file():
        raise FileNotFoundError(f"Missing report: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    headers = [index for index, line in enumerate(lines) if line == "### AVERAGE ACCURACY"]
    if len(headers) != 1:
        raise ValueError(f"Expected one AVERAGE ACCURACY header in {path}, got {len(headers)}")
    value_index = headers[0] + 1
    if value_index >= len(lines):
        raise ValueError(f"Missing AVERAGE ACCURACY value in {path}")
    return float(lines[value_index].strip())


def read_reading_scores(path: Path) -> tuple[float, float]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing reading report: {path}")
    scores: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("EYE TRACKING SCORE:"):
            scores["eye_tracking"] = float(line.split(":", maxsplit=1)[1].strip())
        if line.startswith("SELF-PACED READING SCORE:"):
            scores["self_paced_reading"] = float(line.split(":", maxsplit=1)[1].strip())
    missing = {"eye_tracking", "self_paced_reading"} - set(scores)
    if missing:
        raise ValueError(f"Missing reading scores in {path}: {', '.join(sorted(missing))}")
    return scores["eye_tracking"], scores["self_paced_reading"]


def compact_reading_rows(experiment_name: str, causal_dir: Path) -> list[dict[str, str]]:
    eye_tracking, self_paced = read_reading_scores(causal_dir / "reading" / "report.txt")
    return [
        {
            "experiment_name": experiment_name,
            "experiment_group": experiment_name,
            "task": "reading",
            "submeasure_type": "aggregate",
            "submeasure": "eye_tracking_mean",
            "score": f"{eye_tracking:.6f}",
            "n_items": "",
        },
        {
            "experiment_name": experiment_name,
            "experiment_group": experiment_name,
            "task": "reading",
            "submeasure_type": "self_paced",
            "submeasure": "self_paced_reading_time",
            "score": f"{self_paced:.6f}",
            "n_items": "",
        },
        {
            "experiment_name": experiment_name,
            "experiment_group": experiment_name,
            "task": "reading",
            "submeasure_type": "aggregate",
            "submeasure": "reading_mean",
            "score": f"{((eye_tracking + self_paced) / 2.0):.6f}",
            "n_items": "",
        },
    ]


def discover_context_dirs(input_root: Path) -> list[tuple[int, Path]]:
    if not input_root.is_dir():
        raise NotADirectoryError(f"RA/SA ratio input root is not a directory: {input_root}")
    context_dirs: list[tuple[int, Path]] = []
    for child in input_root.iterdir():
        if not child.is_dir():
            continue
        match = re.fullmatch(r"evaluate_(\d+)", child.name)
        if match:
            context_dirs.append((int(match.group(1)), child))
    if not context_dirs:
        raise ValueError(f"No evaluate_<context> directories found in {input_root}")
    return sorted(context_dirs)


def discover_model_dirs(context_dir: Path) -> list[Path]:
    model_dirs = sorted(
        child
        for child in context_dir.iterdir()
        if child.is_dir() and child.name.endswith("_ep9_hf")
    )
    if not model_dirs:
        raise ValueError(f"No direct *_ep9_hf model directories found in {context_dir}")
    return model_dirs


def eval_root_for_model(model_dir: Path) -> Path:
    eval_root = model_dir / "main" / "zero_shot" / "causal"
    if not eval_root.is_dir():
        raise FileNotFoundError(f"Missing official eval root: {eval_root}")
    return eval_root


def model_metadata(context_length: int, model_dir: Path, input_root: Path) -> dict[str, str]:
    logging_dir = model_dir / "logging"
    train_config = read_yaml_mapping(logging_dir / "train_config.yaml")
    model_config = read_yaml_mapping(logging_dir / "model_config.yaml")

    n_heads_sa = int(model_config["n_heads_sa"])
    n_heads_ra = int(model_config["n_heads_ra"])
    total_heads = n_heads_sa + n_heads_ra
    return {
        "context_length": str(context_length),
        "n_heads_sa": str(n_heads_sa),
        "n_heads_ra": str(n_heads_ra),
        "ra_head_fraction": f"{(n_heads_ra / total_heads):.6f}",
        "datapoint_length": str(train_config["datapoint_length"]),
        "max_seq_len": str(model_config["max_seq_len"]),
        "source_dir": str(model_dir.relative_to(input_root)),
    }


def summarize_model(context_length: int, model_dir: Path, input_root: Path) -> dict[str, str]:
    metadata = model_metadata(context_length, model_dir, input_root)
    eval_root = eval_root_for_model(model_dir)
    eye_tracking, self_paced = read_reading_scores(eval_root / "reading" / "report.txt")
    scores: dict[str, float | None] = {}
    for metric, report_path in REPORT_PATHS.items():
        full_path = eval_root / report_path
        if full_path.is_file():
            scores[metric] = read_average_accuracy(full_path)
        else:
            scores[metric] = None
    result: dict[str, str] = {
        **metadata,
        "experiment_name": model_dir.name,
        "eye_tracking": f"{eye_tracking:.6f}",
        "self_paced_reading": f"{self_paced:.6f}",
        "reading_mean": f"{((eye_tracking + self_paced) / 2.0):.6f}",
    }
    for metric in ("blimp_full", "blimp_supplement", "ewok_full", "entity_tracking_full", "comps_full"):
        val = scores.get(metric)
        result[metric] = f"{val:.6f}" if val is not None else ""
    return result


def summarize_results(input_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for context_length, context_dir in discover_context_dirs(input_root):
        for model_dir in discover_model_dirs(context_dir):
            rows.append(summarize_model(context_length, model_dir, input_root))
    return sorted(rows, key=lambda row: (int(row["context_length"]), int(row["n_heads_sa"])))


def summarize_submeasures(input_root: Path, eval_data_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for context_length, context_dir in discover_context_dirs(input_root):
        for model_dir in discover_model_dirs(context_dir):
            eval_root = eval_root_for_model(model_dir)
            metadata = model_metadata(context_length, model_dir, input_root)
            model_rows = report_rows(model_dir.name, eval_root)
            if (eval_root / "reading" / "predictions.json").is_file():
                if not eval_data_root.is_dir():
                    raise NotADirectoryError(f"Eval data root is not a directory: {eval_data_root}")
                reading_submeasure_rows = reading_rows(model_dir.name, eval_root, eval_data_root)
            else:
                reading_submeasure_rows = compact_reading_rows(model_dir.name, eval_root)
            if not reading_submeasure_rows:
                raise ValueError(f"No reading submeasure rows discovered for {model_dir}")
            model_rows.extend(reading_submeasure_rows)
            rows.extend({**metadata, **row} for row in model_rows)
    if not rows:
        raise ValueError(f"No RA/SA ratio submeasure rows discovered in {input_root}")
    return rows


def write_csv(output_path: Path, rows: list[dict[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def fmt_or_dash(val: float | None) -> str:
    if val is None:
        return "—"
    return f"{val:.2f}"


def score(row: dict[str, str], key: str) -> float:
    return float(row[key])


def score_or_none(row: dict[str, str], key: str) -> float | None:
    val = row.get(key, "")
    if val == "" or val is None:
        return None
    return float(val)


def latex_filename(value: str) -> str:
    return value.replace("_", "\\_")


def write_summary_table(output_path: Path, rows: list[dict[str, str]]) -> None:
    lines = [
        "\\begin{tabular}{rrrrrrrrr}",
        "\\toprule",
        "Context & SA & RA & BLiMP & Supp. & EWoK & COMPS & Entity & Reading \\\\",
        "\\midrule",
    ]
    for row in rows:
        blimp = score_or_none(row, 'blimp_full')
        supp = score_or_none(row, 'blimp_supplement')
        ewok = score_or_none(row, 'ewok_full')
        comps = score_or_none(row, 'comps_full')
        ent = score_or_none(row, 'entity_tracking_full')
        read = score_or_none(row, 'reading_mean')
        lines.append(
            f"{row['context_length']} & {row['n_heads_sa']} & {row['n_heads_ra']} & "
            f"{fmt_or_dash(blimp)} & {fmt_or_dash(supp)} & "
            f"{fmt_or_dash(ewok)} & {fmt_or_dash(comps)} & "
            f"{fmt_or_dash(ent)} & {fmt_or_dash(read)} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def best_by_context(rows: list[dict[str, str]], metric: str) -> list[dict[str, str]]:
    best_rows: list[dict[str, str]] = []
    for context_length in sorted({row["context_length"] for row in rows}, key=int):
        context_rows = [row for row in rows if row["context_length"] == context_length and score_or_none(row, metric) is not None]
        if context_rows:
            best_rows.append(max(context_rows, key=lambda row: score_or_none(row, metric) or float('-inf')))
    return best_rows


def write_best_metric_table(output_path: Path, rows: list[dict[str, str]]) -> None:
    lines = [
        "\\begin{tabular}{llrrr}",
        "\\toprule",
        "Context & Metric & SA & RA & Score \\\\",
        "\\midrule",
    ]
    for context_length in sorted({row["context_length"] for row in rows}, key=int):
        context_rows = [row for row in rows if row["context_length"] == context_length]
        for metric, label in METRIC_LABELS.items():
            valid_rows = [row for row in context_rows if score_or_none(row, metric) is not None]
            if not valid_rows:
                continue
            best = max(valid_rows, key=lambda row: score_or_none(row, metric) or float('-inf'))
            lines.append(
                f"{context_length} & {label} & {best['n_heads_sa']} & {best['n_heads_ra']} & "
                f"{score(best, metric):.2f} \\\\"
            )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_report_section(output_path: Path, rows: list[dict[str, str]]) -> None:
    best_blimp = best_by_context(rows, "blimp_full")
    best_supplement = best_by_context(rows, "blimp_supplement")
    best_comps = best_by_context(rows, "comps_full")
    best_reading = best_by_context(rows, "reading_mean")
    blimp_text = "; ".join(
        f"{row['context_length']}: {row['n_heads_sa']}SA/{row['n_heads_ra']}RA "
        f"({score(row, 'blimp_full'):.2f}\\%)"
        for row in best_blimp
    )
    supplement_text = "; ".join(
        f"{row['context_length']}: {row['n_heads_sa']}SA/{row['n_heads_ra']}RA "
        f"({score(row, 'blimp_supplement'):.2f}\\%)"
        for row in best_supplement
    )
    comps_text = "; ".join(
        f"{row['context_length']}: {row['n_heads_sa']}SA/{row['n_heads_ra']}RA "
        f"({score(row, 'comps_full'):.2f}\\%)"
        for row in best_comps
    )
    reading_text = "; ".join(
        f"{row['context_length']}: {row['n_heads_sa']}SA/{row['n_heads_ra']}RA "
        f"({score(row, 'reading_mean'):.2f})"
        for row in best_reading
    )
    context_lengths = sorted({str(row["context_length"]) for row in rows}, key=int)
    context_tokens = [f"{length}-token" for length in context_lengths]
    if len(context_tokens) > 1:
        contexts_str = f"{', '.join(context_tokens[:-1])} and {context_tokens[-1]}"
    elif context_tokens:
        contexts_str = context_tokens[0]
    else:
        contexts_str = "unknown-token"
    asset_prefix = output_path.parent.name
    escaped_prefix = latex_filename(asset_prefix)
    scores_path_tex = (
        f"\\texttt{{{escaped_prefix}/}}\\allowbreak\\texttt{{{latex_filename(SCORES_CSV_NAME)}}}"
    )
    submeasures_path_tex = (
        f"\\texttt{{{escaped_prefix}/}}\\allowbreak\\texttt{{{latex_filename(SUBMEASURES_CSV_NAME)}}}"
    )

    lines = [
        "\\subsection{RCA SA/RA Head-Ratio Sweep}",
        "",
        "The RCA head-ratio sweep evaluates relative-symbol SwiGLU RCA DAT runs at "
        f"{contexts_str} contexts while varying the split between self-attention and "
        "relational-attention heads. These rows are treated descriptively because the "
        "sweep is not a repeated-seed factorial design.",
        "",
        "The analysis follows the exploratory head-ratio summaries by comparing BLiMP "
        "average accuracy, BLiMP linguistic-category accuracy, supplement-filtered "
        "accuracy, COMPS, EWoK, entity-tracking, and reading scores across SA/RA "
        "splits using the same full zero-shot evaluations.",
        "",
        f"The best BLiMP score by context is {blimp_text}. The best BLiMP supplement score "
        f"by context is {supplement_text}. The best COMPS score by context is {comps_text}. "
        f"The best reading score by context is {reading_text}. Full task-family scores are written to "
        f"{scores_path_tex}, and fine-grained "
        "zero-shot submeasure rows are written to "
        f"{submeasures_path_tex}.",
        "",
        "\\begin{table}[htpb]",
        "    \\centering",
        "    \\small",
        f"    \\input{{{asset_prefix}/summary_table.tex}}",
        "    \\caption{Descriptive RCA head-ratio sweep across context lengths. Reading is the "
        "mean of the eye-tracking and self-paced reading scores reported by the official "
        "reading evaluation.}",
        "    \\label{tab:ra_sa_ratio_sweep}",
        "\\end{table}",
        "",
        "\\begin{table}[htpb]",
        "    \\centering",
        "    \\small",
        f"    \\input{{{asset_prefix}/best_metric_table.tex}}",
        "    \\caption{Best single-seed RCA head split within each context length and "
        "task family. Higher values indicate better zero-shot performance.}",
        "    \\label{tab:ra_sa_ratio_best_by_metric}",
        "\\end{table}",
        "",
        "\\begin{figure}[htpb]",
        "    \\centering",
        f"    \\includegraphics[width=0.48\\textwidth]{{{asset_prefix}/blimp_average.png}}",
        f"    \\includegraphics[width=0.48\\textwidth]{{{asset_prefix}/supplement_average.png}}",
        "    \\caption{Average accuracy for BLiMP (left) and BLiMP supplement (right) across RCA SA/RA splits and context lengths.}",
        "    \\label{fig:ra_sa_ratio_average}",
        "\\end{figure}",
        "",
        "\\begin{figure}[htpb]",
        "    \\centering",
        f"    \\includegraphics[width=\\textwidth]{{{asset_prefix}/blimp_category.png}}",
        "    \\caption{BLiMP accuracy by linguistics category across RCA SA/RA splits and context lengths.}",
        "    \\label{fig:ra_sa_ratio_category}",
        "\\end{figure}",
        "",
        "\\begin{figure}[htpb]",
        "    \\centering",
        f"    \\includegraphics[width=\\textwidth]{{{asset_prefix}/other_zero_shot_average.png}}",
        "    \\caption{COMPS, EWoK, and entity-tracking aggregate accuracy across RCA SA/RA splits and context lengths.}",
        "    \\label{fig:ra_sa_ratio_other_average}",
        "\\end{figure}",
        "",
        "\\begin{figure}[htpb]",
        "    \\centering",
        f"    \\includegraphics[width=\\textwidth,height=0.85\\textheight,keepaspectratio]{{{asset_prefix}/other_zero_shot_submeasures.png}}",
        "    \\caption{COMPS, EWoK, and entity-tracking submeasure accuracy across RCA SA/RA splits and context lengths.}",
        "    \\label{fig:ra_sa_ratio_other_submeasures}",
        "\\end{figure}",
        "",
        "\\begin{figure}[htpb]",
        "    \\centering",
        f"    \\includegraphics[width=0.6\\textwidth]{{{asset_prefix}/reading_average.png}}",
        "    \\caption{Mean reading scores (eye tracking + self-paced) across RCA SA/RA splits and context lengths.}",
        "    \\label{fig:ra_sa_ratio_reading}",
        "\\end{figure}",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = summarize_results(args.input_root)
    submeasure_rows = summarize_submeasures(args.input_root, args.eval_data_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / SCORES_CSV_NAME, rows)
    with (args.output_dir / SUBMEASURES_CSV_NAME).open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=SUBMEASURE_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(submeasure_rows)
    write_summary_table(args.output_dir / "summary_table.tex", rows)
    write_best_metric_table(args.output_dir / "best_metric_table.tex", rows)
    write_report_section(args.output_dir / "report_section.tex", rows)
    print(
        f"Wrote {len(rows)} RA/SA ratio rows and {len(submeasure_rows)} submeasure rows "
        f"to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
