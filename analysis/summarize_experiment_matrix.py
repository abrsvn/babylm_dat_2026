"""Write a CSV summary from completed training/eval experiment directories."""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import torch
import yaml

ANALYSIS_FAMILY_NAMES = (
    "architecture_comparison",
    "rca_symbol_retrieval_comparison",
    "rca_sa_ra_head_ratio_comparison",
)

SENTENCE_TASK_PATHS = {
    "blimp": ("blimp", "blimp_filtered"),
    "supplement": ("blimp", "supplement_filtered"),
    "ewok": ("ewok", "ewok_filtered"),
    "entity_tracking": ("entity_tracking", "entity_tracking"),
    "comps": ("comps", "comps"),
}

READING_PATTERNS = {
    "reading_full": r"^READING SCORE: (?P<value>-?\d+(?:\.\d+)?)$",
    "eye_tracking_full": r"^EYE TRACKING SCORE: (?P<value>-?\d+(?:\.\d+)?)$",
    "self_paced_full": r"^SELF-PACED READING SCORE: (?P<value>-?\d+(?:\.\d+)?)$",
}

CSV_FIELDS = [
    "experiment_name",
    "experiment_group",
    "seed_name",
    "analysis_family",
    "experiment_dir",
    "checkpoint_label",
    "model_type",
    "objective",
    "tie_lm_head",
    "n_epochs",
    "batch_size",
    "seed",
    "corpus_id",
    "hidden_dim",
    "n_layers",
    "n_heads_total",
    "n_heads_sa",
    "n_heads_ra",
    "head_dim",
    "dropout",
    "pe_type",
    "norm_type",
    "datapoint_length",
    "max_seq_len",
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
    "train_value",
    "train_value_type",
    "blimp_full",
    "blimp_supplement_full",
    "ewok_full",
    "entity_tracking_full",
    "comps_full",
    "reading_full",
    "eye_tracking_full",
    "self_paced_full",
    "eval_root",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize completed experiment configs and full eval outputs into a CSV."
    )
    parser.add_argument(
        "--experiment_root",
        action="append",
        type=Path,
        required=True,
        help="Experiment root or individual experiment directory. Can be passed more than once.",
    )
    parser.add_argument(
        "--checkpoint_label",
        default="9",
        help="Checkpoint label whose eval tree should be summarized.",
    )
    parser.add_argument(
        "--eval_output_root",
        action="append",
        type=Path,
        default=None,
        help=(
            "Eval output root containing <experiment>/<checkpoint>/zero_shot/causal. "
            "Can be passed more than once. Defaults to repo-local results/."
        ),
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        required=True,
        help="CSV file to write.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text()) or {}


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


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


def discover_experiments(paths: list[Path], checkpoint_label: str) -> list[Path]:
    experiment_dirs: list[Path] = []
    for path in paths:
        if not path.is_dir():
            raise FileNotFoundError(f"Experiment root does not exist: {path}")
        if has_config_pair(path) and is_supported_experiment_dir(path, checkpoint_label):
            experiment_dirs.append(path)
            continue

        for child in sorted(child for child in path.rglob("*") if child.is_dir()):
            if is_nested_logging_config_dir(child):
                continue
            if has_config_pair(child) and is_supported_experiment_dir(child, checkpoint_label):
                experiment_dirs.append(child)

    return sorted(set(experiment_dirs))


def load_train_value(experiment_dir: Path, checkpoint_label: str) -> str:
    metrics_path = experiment_dir / "logging" / f"epoch_{checkpoint_label}_metrics.pth"
    if not metrics_path.is_file():
        return ""
    metrics = torch.load(metrics_path, map_location="cpu", weights_only=True)
    return format_value(metrics["loss"])


def read_sentence_score(eval_root: Path, task: str) -> tuple[str, str]:
    task_dir, report_dir = SENTENCE_TASK_PATHS[task]
    official_report_path = eval_root / task_dir / report_dir / "best_temperature_report.txt"
    if official_report_path.is_file():
        report_lines = official_report_path.read_text(encoding="utf-8").splitlines()
        for line_index, line in enumerate(report_lines):
            if line == "### AVERAGE ACCURACY" and line_index + 1 < len(report_lines):
                return format_value(float(report_lines[line_index + 1].strip())), ""
        raise ValueError(f"Could not find AVERAGE ACCURACY in {official_report_path}")

    summary_path = eval_root / task_dir / report_dir / "temperature_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        score_data = summary["1"]
        return format_value(score_data["overall_accuracy"]), ""

    return "", ""


def read_reading_scores(eval_root: Path) -> dict[str, str]:
    report_path = eval_root / "reading" / "report.txt"
    scores = {key: "" for key in READING_PATTERNS}
    if not report_path.is_file():
        return scores

    report_lines = report_path.read_text().splitlines()
    for key, pattern in READING_PATTERNS.items():
        regex = re.compile(pattern)
        for line in report_lines:
            match = regex.match(line)
            if match is not None:
                scores[key] = format_value(float(match.group("value")))
                break

    return scores


def complete_reading_scores(eval_root: Path) -> dict[str, str]:
    scores = read_reading_scores(eval_root)
    if scores["reading_full"] == "" and scores["eye_tracking_full"] and scores["self_paced_full"]:
        reading_mean = (
            float(scores["eye_tracking_full"]) + float(scores["self_paced_full"])
        ) / 2.0
        scores["reading_full"] = format_value(reading_mean)
    return scores


def has_eval_outputs(eval_root: Path) -> bool:
    sentence_outputs = []
    for task_dir, report_dir in SENTENCE_TASK_PATHS.values():
        sentence_outputs.append(
            (
                eval_root / task_dir / report_dir / "temperature_summary.json"
            ).is_file()
            or (
                eval_root / task_dir / report_dir / "best_temperature_report.txt"
            ).is_file()
        )
    reading_report = eval_root / "reading" / "report.txt"
    return all(sentence_outputs) and reading_report.is_file()


def resolve_eval_root(
    experiment_dir: Path,
    checkpoint_label: str,
    eval_output_roots: list[Path],
) -> Path:
    compact_reports_root = experiment_dir / "official_reports"
    if has_eval_outputs(compact_reports_root):
        return compact_reports_root

    direct_official_eval_root = experiment_dir / "main" / "zero_shot" / "causal"
    if has_eval_outputs(direct_official_eval_root):
        return direct_official_eval_root

    official_2026_eval_root = experiment_dir / "2026_eval" / "results" / f"{experiment_dir.name}_ep{checkpoint_label}_hf" / "main" / "zero_shot" / "causal"
    if has_eval_outputs(official_2026_eval_root):
        return official_2026_eval_root

    local_eval_root = experiment_dir / checkpoint_label / "zero_shot" / "causal"
    if has_eval_outputs(local_eval_root):
        return local_eval_root

    separate_eval_roots = []
    for output_root in eval_output_roots:
        eval_root = output_root / experiment_dir.name / checkpoint_label / "zero_shot" / "causal"
        separate_eval_roots.append(eval_root)
        if has_eval_outputs(eval_root):
            return eval_root

    candidate_roots = [
        compact_reports_root,
        direct_official_eval_root,
        official_2026_eval_root,
        local_eval_root,
        *separate_eval_roots,
    ]
    raise FileNotFoundError(
        "Could not find complete eval outputs for "
        f"{experiment_dir}. Checked: {', '.join(str(path) for path in candidate_roots)}"
    )


def get_n_heads_total(model_type: str, model_config: dict[str, Any]) -> int:
    if model_type == "dat":
        return int(model_config["n_heads_sa"]) + int(model_config["n_heads_ra"])
    return int(model_config["n_heads"])


def get_ffn_activation(model_type: str, model_config: dict[str, Any]) -> str:
    if "ffn_activation" in model_config:
        return str(model_config["ffn_activation"])
    if model_type == "self_attention":
        return "gelu"
    return ""


def get_seed_name(seed: Any) -> str:
    return f"seed{seed}"


def get_experiment_group(experiment_name: str) -> str:
    return re.sub(r"_seed\d+$", "", experiment_name)


def row_for_experiment(
    experiment_dir: Path,
    checkpoint_label: str,
    eval_output_roots: list[Path],
) -> dict[str, str]:
    train_config_path, model_config_path = config_paths(experiment_dir)
    train_config = load_yaml(train_config_path)
    model_config = load_yaml(model_config_path)
    model_type = str(train_config["model_type"])
    n_heads_total = get_n_heads_total(model_type, model_config)
    hidden_dim = int(model_config["hidden_dim"])
    eval_root = resolve_eval_root(experiment_dir, checkpoint_label, eval_output_roots)
    objective = "NextLat" if bool(train_config["nextlat_enabled"]) else "NTP"
    experiment_name = str(train_config["experiment_name"])

    row = {
        "experiment_name": experiment_name,
        "experiment_group": get_experiment_group(experiment_name),
        "seed_name": get_seed_name(train_config["seed"]),
        "analysis_family": analysis_family_for_path(experiment_dir),
        "experiment_dir": str(experiment_dir),
        "checkpoint_label": checkpoint_label,
        "model_type": model_type,
        "objective": objective,
        "tie_lm_head": format_value(model_config["tie_lm_head"]),
        "n_epochs": format_value(train_config["n_epochs"]),
        "batch_size": format_value(train_config["batch_size"]),
        "seed": format_value(train_config["seed"]),
        "corpus_id": format_value(train_config["corpus_id"]),
        "hidden_dim": format_value(hidden_dim),
        "n_layers": format_value(model_config["n_layers"]),
        "n_heads_total": format_value(n_heads_total),
        "n_heads_sa": format_value(model_config.get("n_heads_sa")),
        "n_heads_ra": format_value(model_config.get("n_heads_ra")),
        "head_dim": format_value(hidden_dim // n_heads_total),
        "dropout": format_value(model_config["dropout"]),
        "pe_type": format_value(model_config["pe_type"]),
        "norm_type": format_value(model_config["norm_type"]),
        "datapoint_length": format_value(train_config["datapoint_length"]),
        "max_seq_len": format_value(model_config["max_seq_len"]),
        "sequence_boundary_policy": format_value(model_config["sequence_boundary_policy"]),
        "symbol_retrieval": format_value(model_config.get("symbol_retrieval")),
        "positional_symbols_sinusoidal": format_value(
            model_config.get("positional_symbols_sinusoidal")
        ),
        "relative_symbols_rope": format_value(model_config.get("relative_symbols_rope")),
        "symbolic_attn_n_heads": format_value(model_config.get("symbolic_attn_n_heads")),
        "symbolic_use_bias": format_value(model_config.get("symbolic_use_bias")),
        "relsymbolic_neighborhood_size": format_value(
            model_config.get("relsymbolic_neighborhood_size")
        ),
        "ra_type": format_value(model_config.get("ra_type")),
        "ra_rel_activation": format_value(model_config.get("ra_rel_activation")),
        "ffn_activation": get_ffn_activation(model_type, model_config),
        "init_scheme": format_value(model_config.get("init_scheme")),
        "nextlat_horizon": format_value(train_config["nextlat_horizon"]),
        "nextlat_lambda_mse": format_value(train_config["nextlat_lambda_mse"]),
        "nextlat_lambda_kl": format_value(train_config["nextlat_lambda_kl"]),
        "nextlat_lambda_ce": format_value(train_config["nextlat_lambda_ce"]),
        "nextlat_proj_factor": format_value(train_config["nextlat_proj_factor"]),
        "train_value": load_train_value(experiment_dir, checkpoint_label),
        "train_value_type": "total_objective" if objective == "NextLat" else "ce",
        "eval_root": str(eval_root),
    }

    blimp_score, _ = read_sentence_score(eval_root, "blimp")
    blimp_supplement_score, _ = read_sentence_score(eval_root, "supplement")
    ewok_score, _ = read_sentence_score(eval_root, "ewok")
    entity_score, _ = read_sentence_score(eval_root, "entity_tracking")
    comps_score, _ = read_sentence_score(eval_root, "comps")
    reading_scores = complete_reading_scores(eval_root)

    row.update(
        {
            "blimp_full": blimp_score,
            "blimp_supplement_full": blimp_supplement_score,
            "ewok_full": ewok_score,
            "entity_tracking_full": entity_score,
            "comps_full": comps_score,
            **reading_scores,
        }
    )
    return row


def write_csv(rows: list[dict[str, str]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    experiment_roots = args.experiment_root
    eval_output_roots = args.eval_output_root or [Path("results")]
    experiment_dirs = discover_experiments(experiment_roots, args.checkpoint_label)
    rows = [
        row_for_experiment(experiment_dir, args.checkpoint_label, eval_output_roots)
        for experiment_dir in experiment_dirs
    ]
    write_csv(rows, args.output_csv)
    print(f"Wrote {len(rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
