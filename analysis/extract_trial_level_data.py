#!/usr/bin/env python3
"""Extract sentence-classification scores into pseudo-items for GLMM analysis."""

import argparse
import csv
import gzip
import json
import sys
import re
from pathlib import Path
import yaml

ANALYSIS_FAMILY_NAMES = (
    "architecture_comparison",
    "rca_symbol_retrieval_comparison",
    "rca_sa_ra_head_ratio_comparison",
)

COMPS_SUBTASK_TO_FILE = {
    "base": "comps_base",
    "wugs": "comps_wugs",
    "wugs_dist_before": "comps_wugs_dist-before",
    "wugs_dist_in_between": "comps_wugs_dist-in-between",
}


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_experiment_group(experiment_name: str) -> str:
    return re.sub(r"_seed\d+$", "", experiment_name)

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

def has_prediction_layout(causal_root: Path, prediction_paths: list[Path]) -> bool:
    return any((causal_root / prediction_path).is_file() for prediction_path in prediction_paths)

def causal_dir_for_checkpoint(
    experiment_dir: Path,
    experiment_name: str,
    checkpoint_label: str,
    prediction_paths: list[Path],
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
    for candidate in candidates:
        if candidate.is_dir() and has_prediction_layout(candidate, prediction_paths):
            return candidate

    results_dir = experiment_dir / "2026_eval" / "results"
    if results_dir.is_dir():
        discovered = sorted(
            path / "main" / "zero_shot" / "causal"
            for path in results_dir.iterdir()
            if path.is_dir() and path.name.endswith(f"_ep{checkpoint_label}_hf")
        )
        discovered = [path for path in discovered if path.is_dir() and has_prediction_layout(path, prediction_paths)]
        if len(discovered) == 1:
            return discovered[0]
        if len(discovered) > 1:
            raise ValueError(
                f"Multiple checkpoint {checkpoint_label} causal eval roots found in {results_dir}"
            )
    return None

def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("pseudo_item_count must be positive")
    return parsed


def load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_metadata_map(path: Path) -> dict[tuple[str, str], str]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"experiment_name", "experiment_group", "analysis_family"}
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError(f"{path} is missing required metadata columns")
        return {
            (row["experiment_name"], row["experiment_group"]): row["analysis_family"]
            for row in reader
        }


def jsonl_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def load_blimp_truth(eval_data_root: Path, data_dir_name: str) -> dict[str, dict]:
    data_dir = eval_data_root / data_dir_name
    if not data_dir.is_dir():
        raise FileNotFoundError(f"{data_dir} not found")
    truth = {}
    for jsonl_file in sorted(data_dir.glob("*.jsonl")):
        subtest = jsonl_file.stem
        answers = []
        ling_term = None
        for row in jsonl_rows(jsonl_file):
            answers.append(str(row["sentence_good"]).strip())
            if ling_term is None and "linguistics_term" in row:
                ling_term = str(row["linguistics_term"])
        truth[subtest] = {
            "answers": answers,
            "submeasure_type": "subtest",
            "ling_term": ling_term or "unknown",
        }
    return truth


def load_comps_truth(eval_data_root: Path) -> dict[str, dict]:
    data_dir = eval_data_root / "comps"
    if not data_dir.is_dir():
        raise FileNotFoundError(f"{data_dir} not found")
    truth = {}
    for submeasure, file_stem in COMPS_SUBTASK_TO_FILE.items():
        jsonl_file = data_dir / f"{file_stem}.jsonl"
        answers = [
            " ".join([str(row["prefix_acceptable"]), str(row["property_phrase"])]).strip()
            for row in jsonl_rows(jsonl_file)
        ]
        truth[submeasure] = {
            "answers": answers,
            "submeasure_type": "subtask",
            "ling_term": "unknown",
        }
    return truth


def load_ewok_truth(eval_data_root: Path) -> dict[str, dict]:
    data_dir = eval_data_root / "ewok_filtered"
    if not data_dir.is_dir():
        raise FileNotFoundError(f"{data_dir} not found")
    truth = {}
    for jsonl_file in sorted(data_dir.glob("*.jsonl")):
        domain = jsonl_file.stem
        answers = [
            " ".join([str(row["Context1"]), str(row["Target1"])]).strip()
            for row in jsonl_rows(jsonl_file)
        ]
        truth[domain] = {
            "answers": answers,
            "submeasure_type": "domain",
            "ling_term": "unknown",
        }
    return truth


def load_entity_tracking_truth(eval_data_root: Path) -> dict[str, dict]:
    data_dir = eval_data_root / "entity_tracking"
    if not data_dir.is_dir():
        raise FileNotFoundError(f"{data_dir} not found")
    truth: dict[str, dict] = {}
    for jsonl_file in sorted(data_dir.glob("*.jsonl")):
        split = jsonl_file.stem
        for row in jsonl_rows(jsonl_file):
            submeasure = f"{split}_{row['numops']}_ops"
            if submeasure not in truth:
                truth[submeasure] = {
                    "answers": [],
                    "submeasure_type": "uid_accuracy",
                    "ling_term": "unknown",
                }
            truth[submeasure]["answers"].append(str(row["options"][0]).strip())
    return truth


def pseudo_item_count(args: argparse.Namespace, task: str) -> int:
    if task == "blimp":
        return args.pseudo_item_count
    if task == "blimp_supplement":
        return args.supplement_pseudo_item_count
    if task == "comps":
        return args.comps_pseudo_item_count
    if task == "ewok":
        return args.ewok_pseudo_item_count
    if task == "entity_tracking":
        return args.entity_tracking_pseudo_item_count
    raise ValueError(f"Unknown classification task: {task}")

def parse_args():
    parser = argparse.ArgumentParser(description="Extract sentence-classification scores into pseudo-items.")
    parser.add_argument("--experiment_root", action="append", type=Path, required=True)
    parser.add_argument("--eval_data_root", type=Path, required=True, help="Path to babylm-eval/strict/evaluation_data/full_eval")
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--checkpoint_label", type=str, default="9")
    parser.add_argument(
        "--metadata_csv",
        type=Path,
        default=None,
        help="Optional matrix/long-scores CSV used to assign analysis_family and filter raw runs.",
    )
    parser.add_argument("--pseudo_item_count", type=positive_int, default=100)
    parser.add_argument("--supplement_pseudo_item_count", type=positive_int, default=10)
    parser.add_argument("--comps_pseudo_item_count", type=positive_int, default=100)
    parser.add_argument("--ewok_pseudo_item_count", type=positive_int, default=10)
    parser.add_argument("--entity_tracking_pseudo_item_count", type=positive_int, default=100)
    return parser.parse_args()

def main():
    args = parse_args()
    out_path = args.output_csv
    eval_data_root = args.eval_data_root
    try:
        metadata_map = load_metadata_map(args.metadata_csv) if args.metadata_csv is not None else None
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    try:
        task_specs = [
            {
                "task": "blimp",
                "split": "blimp_filtered",
                "prediction_path": Path("blimp") / "blimp_filtered" / "predictions.json",
                "truth": load_blimp_truth(eval_data_root, "blimp_filtered"),
            },
            {
                "task": "blimp_supplement",
                "split": "supplement_filtered",
                "prediction_path": Path("blimp") / "supplement_filtered" / "predictions.json",
                "truth": load_blimp_truth(eval_data_root, "supplement_filtered"),
            },
            {
                "task": "comps",
                "split": "comps",
                "prediction_path": Path("comps") / "comps" / "predictions.json",
                "truth": load_comps_truth(eval_data_root),
            },
            {
                "task": "ewok",
                "split": "ewok_filtered",
                "prediction_path": Path("ewok") / "ewok_filtered" / "predictions.json",
                "truth": load_ewok_truth(eval_data_root),
            },
            {
                "task": "entity_tracking",
                "split": "entity_tracking",
                "prediction_path": Path("entity_tracking") / "entity_tracking" / "predictions.json",
                "truth": load_entity_tracking_truth(eval_data_root),
            },
        ]
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    rows = []

    for root in args.experiment_root:
        if not root.is_dir():
            print(f"Error: {root} is not a directory", file=sys.stderr)
            return 1

        prediction_paths = [spec["prediction_path"] for spec in task_specs]
        for exp_dir in discover_experiment_dirs(root, args.checkpoint_label):
            train_config_path, _ = config_paths(exp_dir)
            train_config = load_yaml(train_config_path)
            seed = train_config["seed"]
            seed_name = f"seed{seed}"
            experiment_name = str(train_config["experiment_name"])
            experiment_group = get_experiment_group(experiment_name)
            experiment_key = (experiment_name, experiment_group)
            if metadata_map is not None and experiment_key not in metadata_map:
                continue
            causal_dir = causal_dir_for_checkpoint(
                exp_dir,
                experiment_name,
                args.checkpoint_label,
                prediction_paths,
            )
            if causal_dir is None:
                continue

            analysis_family = (
                metadata_map[experiment_key]
                if metadata_map is not None
                else analysis_family_for_path(exp_dir)
            )
            for spec in task_specs:
                task = str(spec["task"])
                split = str(spec["split"])
                pred_path = causal_dir / spec["prediction_path"]
                if not pred_path.exists():
                    continue

                pred_data = load_json(pred_path)
                ground_truth = spec["truth"]
                prediction_submeasures = set(pred_data)
                expected_submeasures = set(ground_truth)
                unexpected_submeasures = sorted(prediction_submeasures - expected_submeasures)
                missing_submeasures = sorted(expected_submeasures - prediction_submeasures)
                if unexpected_submeasures:
                    print(
                        (
                            f"Error: unexpected submeasure(s) in {experiment_name} "
                            f"for {task}: {', '.join(unexpected_submeasures)}"
                        ),
                        file=sys.stderr,
                    )
                    return 1
                if missing_submeasures:
                    print(
                        (
                            f"Error: missing submeasure(s) in {experiment_name} "
                            f"for {task}: {', '.join(missing_submeasures)}"
                        ),
                        file=sys.stderr,
                    )
                    return 1

                for submeasure, submeasure_data in pred_data.items():
                    predictions = submeasure_data.get("predictions", [])
                    answers = ground_truth[submeasure]["answers"]
                    ling_term = ground_truth[submeasure]["ling_term"]
                    submeasure_type = ground_truth[submeasure]["submeasure_type"]

                    if len(predictions) != len(answers):
                        print(f"Error: length mismatch in {experiment_name} for {task}/{submeasure} (predictions: {len(predictions)}, ground truth: {len(answers)})", file=sys.stderr)
                        return 1

                    # Aggregate into pseudo-item bins deterministically.
                    count = pseudo_item_count(args, task)
                    bins = {i: [] for i in range(count)}

                    for idx, (pred_item, answer) in enumerate(zip(predictions, answers)):
                        if "pred" not in pred_item:
                            print(f"Error: missing 'pred' key in prediction for {experiment_name} {task}/{submeasure} item {idx}", file=sys.stderr)
                            return 1
                        pred_str = pred_item["pred"].strip()
                        score = 1 if pred_str == answer else 0
                        bins[idx % count].append(score)

                    for bin_id, scores in bins.items():
                        if not scores: continue
                        successes = sum(scores)
                        n_items = len(scores)
                        accuracy = successes / n_items

                        rows.append({
                            "experiment_name": experiment_name,
                            "experiment_group": experiment_group,
                            "analysis_family": analysis_family,
                            "seed_name": seed_name,
                            "task": task,
                            "split": split,
                            "submeasure_type": submeasure_type,
                            "submeasure": submeasure,
                            "subtest": submeasure,
                            "ling_term": ling_term,
                            "pseudo_item_id": f"{task}::{submeasure}::bin{bin_id}",
                            "successes": successes,
                            "n_items": n_items,
                            "accuracy": accuracy
                        })

    if not rows:
        print("No trial-level results found.", file=sys.stderr)
        return 1

    # Always emit gzip-compressed CSV. The output path is forced to end in
    # .gz so the canonical committed artifact is classification_pseudo_items.csv.gz.
    # R's read.csv() transparently decompresses .gz, so the downstream metric
    # model scripts need no changes.
    out_path = args.output_csv
    if not str(out_path).endswith(".gz"):
        out_path = out_path.with_name(out_path.name + ".gz")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment_name", "experiment_group", "analysis_family", "seed_name",
        "task", "split", "submeasure_type", "submeasure", "subtest", "ling_term", "pseudo_item_id",
        "successes", "n_items", "accuracy"
    ]
    with gzip.open(out_path, "wt", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} pseudo-item rows to {out_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
