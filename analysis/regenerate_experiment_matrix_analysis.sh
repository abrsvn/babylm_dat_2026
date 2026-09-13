#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

if [[ -n "${EXPERIMENT_ROOTS:-}" ]]; then
    EXPERIMENT_ROOTS_TEXT="${EXPERIMENT_ROOTS}"
    DEFAULT_ANALYSIS_DIR=""
    DEFAULT_READING_EXPERIMENT_ROOTS="${EXPERIMENT_ROOTS_TEXT}"
elif [[ -n "${EXPERIMENT_ROOT:-}" ]]; then
    EXPERIMENT_ROOTS_TEXT="${EXPERIMENT_ROOT}"
    DEFAULT_ANALYSIS_DIR="${EXPERIMENT_ROOT}/analysis"
    DEFAULT_READING_EXPERIMENT_ROOTS="${EXPERIMENT_ROOTS_TEXT}"
else
    echo "Set EXPERIMENT_ROOTS (colon-separated) or EXPERIMENT_ROOT to the experiment tree to analyze" >&2
    exit 1
fi
IFS=: read -r -a EXPERIMENT_ROOT_VALUES <<< "${EXPERIMENT_ROOTS_TEXT}"
if [[ "${#EXPERIMENT_ROOT_VALUES[@]}" -eq 0 ]]; then
    echo "EXPERIMENT_ROOTS must contain at least one root" >&2
    exit 1
fi
EXPERIMENT_ROOT_ARGS=()
for experiment_root_value in "${EXPERIMENT_ROOT_VALUES[@]}"; do
    EXPERIMENT_ROOT_ARGS+=(--experiment_root "${experiment_root_value}")
done
CLASSIFICATION_PSEUDO_ITEM_ROOTS="${CLASSIFICATION_PSEUDO_ITEM_ROOTS:-}"
CLASSIFICATION_PSEUDO_ITEM_ROOT_ARGS=()
if [[ -n "${CLASSIFICATION_PSEUDO_ITEM_ROOTS}" ]]; then
    IFS=: read -r -a CLASSIFICATION_PSEUDO_ITEM_ROOT_VALUES <<< "${CLASSIFICATION_PSEUDO_ITEM_ROOTS}"
    if [[ "${#CLASSIFICATION_PSEUDO_ITEM_ROOT_VALUES[@]}" -eq 0 ]]; then
        echo "CLASSIFICATION_PSEUDO_ITEM_ROOTS must contain at least one root when provided" >&2
        exit 1
    fi
    for classification_root_value in "${CLASSIFICATION_PSEUDO_ITEM_ROOT_VALUES[@]}"; do
        CLASSIFICATION_PSEUDO_ITEM_ROOT_ARGS+=(--experiment_root "${classification_root_value}")
    done
fi
CLASSIFICATION_PSEUDO_ITEM_FAMILIES="${CLASSIFICATION_PSEUDO_ITEM_FAMILIES:-architecture_comparison rca_symbol_retrieval_comparison}"
read -r -a CLASSIFICATION_PSEUDO_ITEM_FAMILY_VALUES <<< "${CLASSIFICATION_PSEUDO_ITEM_FAMILIES}"
if [[ "${#CLASSIFICATION_PSEUDO_ITEM_FAMILY_VALUES[@]}" -eq 0 ]]; then
    echo "CLASSIFICATION_PSEUDO_ITEM_FAMILIES must contain at least one family" >&2
    exit 1
fi
CLASSIFICATION_PSEUDO_ITEM_TASKS="${CLASSIFICATION_PSEUDO_ITEM_TASKS:-blimp blimp_supplement comps ewok entity_tracking}"
read -r -a CLASSIFICATION_PSEUDO_ITEM_TASK_VALUES <<< "${CLASSIFICATION_PSEUDO_ITEM_TASKS}"
if [[ "${#CLASSIFICATION_PSEUDO_ITEM_TASK_VALUES[@]}" -eq 0 ]]; then
    echo "CLASSIFICATION_PSEUDO_ITEM_TASKS must contain at least one task" >&2
    exit 1
fi
ANALYSIS_DIR="${ANALYSIS_DIR:-${DEFAULT_ANALYSIS_DIR}}"
if [[ -z "${ANALYSIS_DIR}" ]]; then
    echo "Set ANALYSIS_DIR to the directory for analysis outputs" >&2
    exit 1
fi
MATRIX_CSV="${MATRIX_CSV:-${ANALYSIS_DIR}/experiment_matrix.csv}"
CHECKPOINT_LABEL="${CHECKPOINT_LABEL:-9}"
EXPECTED_GROUPS="${EXPECTED_GROUPS:-47}"
EXPECTED_SEEDS="${EXPECTED_SEEDS:-5}"
RUN_METRIC_MODELS="${RUN_METRIC_MODELS:-0}"
FORCE_REBUILD="${FORCE_REBUILD:-0}"
REBUILD_CLASSIFICATION_PSEUDO_ITEMS="${REBUILD_CLASSIFICATION_PSEUDO_ITEMS:-0}"
FAST_MIXED_MODELS="${FAST_MIXED_MODELS:-0}"
ULTRA_FAST_MIXED_MODELS="${ULTRA_FAST_MIXED_MODELS:-0}"
RUN_RCA_SYMBOL_ANALYSIS="${RUN_RCA_SYMBOL_ANALYSIS:-0}"
RUN_RCA_SYMBOL_MODELS="${RUN_RCA_SYMBOL_MODELS:-0}"
RUN_RA_SA_RATIO_ANALYSIS="${RUN_RA_SA_RATIO_ANALYSIS:-1}"
RUN_READING_MODELS="${RUN_READING_MODELS:-${RUN_METRIC_MODELS}}"
READING_EXPERIMENT_ROOTS="${READING_EXPERIMENT_ROOTS:-${DEFAULT_READING_EXPERIMENT_ROOTS}}"
RA_SA_RATIO_INPUT_ROOT="${RA_SA_RATIO_INPUT_ROOT:-}"
if [[ "${RUN_RA_SA_RATIO_ANALYSIS}" == "1" && -z "${RA_SA_RATIO_INPUT_ROOT}" ]]; then
    echo "Set RA_SA_RATIO_INPUT_ROOT to the SA/RA head-ratio eval-output tree" >&2
    exit 1
fi
EVAL_DATA_ROOT="${EVAL_DATA_ROOT:-}"
if [[ ( "${RUN_RA_SA_RATIO_ANALYSIS}" == "1" || "${RUN_RCA_SYMBOL_MODELS}" == "1" || "${REBUILD_CLASSIFICATION_PSEUDO_ITEMS}" == "1" ) && -z "${EVAL_DATA_ROOT}" ]]; then
    echo "Set EVAL_DATA_ROOT to the official BabyLM evaluation data directory" >&2
    exit 1
fi
RCA_SYMBOL_RUN_TAG="${RCA_SYMBOL_RUN_TAG:-}"
if [[ ( "${RUN_RCA_SYMBOL_ANALYSIS}" == "1" || "${RUN_RCA_SYMBOL_MODELS}" == "1" ) && -z "${RCA_SYMBOL_RUN_TAG}" ]]; then
    echo "Set RCA_SYMBOL_RUN_TAG to the run-tag prefix shared by the symbol-retrieval runs" >&2
    exit 1
fi
RCA_SYMBOL_MODES="${RCA_SYMBOL_MODES:-relative relative_rope positional positional_sinusoidal symbolic relsymbolic relsymbolic_n4}"
RCA_SYMBOL_SEEDS="${RCA_SYMBOL_SEEDS:-}"
RCA_SYMBOL_EXPORT_SOURCE_ROOT="${RCA_SYMBOL_EXPORT_SOURCE_ROOT:-}"
read -r -a RCA_SYMBOL_MODE_VALUES <<< "${RCA_SYMBOL_MODES}"
if [[ "${#RCA_SYMBOL_MODE_VALUES[@]}" -eq 0 ]]; then
    echo "RCA_SYMBOL_MODES must contain at least one mode" >&2
    exit 1
fi

if [[ $(printf "%s\n" "${RCA_SYMBOL_MODE_VALUES[@]}" | sort | uniq -d | wc -l) -ne 0 ]]; then
    echo "RCA_SYMBOL_MODES contains duplicate values" >&2
    exit 1
fi
RCA_SYMBOL_MODES_CSV="$(IFS=,; printf '%s' "${RCA_SYMBOL_MODE_VALUES[*]}")"
RCA_SYMBOL_SEED_ARGS=()
RCA_SYMBOL_EXPORT_SEED_ARGS=()
if [[ -n "${RCA_SYMBOL_SEEDS}" ]]; then
    read -r -a RCA_SYMBOL_SEED_VALUES <<< "${RCA_SYMBOL_SEEDS}"
    if [[ "${#RCA_SYMBOL_SEED_VALUES[@]}" -eq 0 ]]; then
        echo "RCA_SYMBOL_SEEDS must contain at least one seed when provided" >&2
        exit 1
    fi
    if [[ $(printf "%s\n" "${RCA_SYMBOL_SEED_VALUES[@]}" | sort | uniq -d | wc -l) -ne 0 ]]; then
        echo "RCA_SYMBOL_SEEDS contains duplicate values" >&2
        exit 1
    fi
    RCA_SYMBOL_SEED_ARGS=(--expected_seed_names "${RCA_SYMBOL_SEED_VALUES[@]}")
    RCA_SYMBOL_EXPORT_SEED_ARGS=(--seed-names "${RCA_SYMBOL_SEED_VALUES[@]}")
fi

if [[ "${RUN_METRIC_MODELS}" != "0" && "${RUN_METRIC_MODELS}" != "1" ]]; then
    echo "RUN_METRIC_MODELS must be 0 or 1, got ${RUN_METRIC_MODELS}" >&2
    exit 1
fi
if [[ "${FORCE_REBUILD}" != "0" && "${FORCE_REBUILD}" != "1" ]]; then
    echo "FORCE_REBUILD must be 0 or 1, got ${FORCE_REBUILD}" >&2
    exit 1
fi
if [[ "${REBUILD_CLASSIFICATION_PSEUDO_ITEMS}" != "0" && "${REBUILD_CLASSIFICATION_PSEUDO_ITEMS}" != "1" ]]; then
    echo "REBUILD_CLASSIFICATION_PSEUDO_ITEMS must be 0 or 1, got ${REBUILD_CLASSIFICATION_PSEUDO_ITEMS}" >&2
    exit 1
fi
if [[ "${FAST_MIXED_MODELS}" != "0" && "${FAST_MIXED_MODELS}" != "1" ]]; then
    echo "FAST_MIXED_MODELS must be 0 or 1, got ${FAST_MIXED_MODELS}" >&2
    exit 1
fi
if [[ "${ULTRA_FAST_MIXED_MODELS}" != "0" && "${ULTRA_FAST_MIXED_MODELS}" != "1" ]]; then
    echo "ULTRA_FAST_MIXED_MODELS must be 0 or 1, got ${ULTRA_FAST_MIXED_MODELS}" >&2
    exit 1
fi
if [[ "${FAST_MIXED_MODELS}" == "1" && "${ULTRA_FAST_MIXED_MODELS}" == "1" ]]; then
    echo "FAST_MIXED_MODELS and ULTRA_FAST_MIXED_MODELS are mutually exclusive" >&2
    exit 1
fi
if [[ "${RUN_RCA_SYMBOL_ANALYSIS}" != "0" && "${RUN_RCA_SYMBOL_ANALYSIS}" != "1" ]]; then
    echo "RUN_RCA_SYMBOL_ANALYSIS must be 0 or 1, got ${RUN_RCA_SYMBOL_ANALYSIS}" >&2
    exit 1
fi
if [[ "${RUN_RCA_SYMBOL_MODELS}" != "0" && "${RUN_RCA_SYMBOL_MODELS}" != "1" ]]; then
    echo "RUN_RCA_SYMBOL_MODELS must be 0 or 1, got ${RUN_RCA_SYMBOL_MODELS}" >&2
    exit 1
fi
if [[ "${RUN_RA_SA_RATIO_ANALYSIS}" != "0" && "${RUN_RA_SA_RATIO_ANALYSIS}" != "1" ]]; then
    echo "RUN_RA_SA_RATIO_ANALYSIS must be 0 or 1, got ${RUN_RA_SA_RATIO_ANALYSIS}" >&2
    exit 1
fi
if [[ "${RUN_READING_MODELS}" != "0" && "${RUN_READING_MODELS}" != "1" ]]; then
    echo "RUN_READING_MODELS must be 0 or 1, got ${RUN_READING_MODELS}" >&2
    exit 1
fi

export FAST_MIXED_MODELS
export ULTRA_FAST_MIXED_MODELS
if [[ "${ULTRA_FAST_MIXED_MODELS}" == "1" ]]; then
    MIXED_MODEL_MODE_SIGNATURE="MIXED_MODEL_MODE=ultra_fast"
elif [[ "${FAST_MIXED_MODELS}" == "1" ]]; then
    MIXED_MODEL_MODE_SIGNATURE="MIXED_MODEL_MODE=fast"
else
    MIXED_MODEL_MODE_SIGNATURE="MIXED_MODEL_MODE=full"
fi
if [[ "${ULTRA_FAST_MIXED_MODELS}" == "1" ]]; then
    echo "ULTRA_FAST_MIXED_MODELS=1: fitting one intercept-only random-effects structure with full fixed effects and using *_ultrafast.rds caches."
elif [[ "${FAST_MIXED_MODELS}" == "1" ]]; then
    echo "FAST_MIXED_MODELS=1: selecting among random-intercept effects only and using *_fast.rds caches."
fi

ARCHITECTURE_OUTPUT_DIR="${ANALYSIS_DIR}/architecture_comparison"
RCA_SYMBOL_OUTPUT_DIR="${ANALYSIS_DIR}/rca_symbol_retrieval_comparison"
RCA_SYMBOL_MODEL_OUTPUT_DIR="${RCA_SYMBOL_OUTPUT_DIR}/model_outputs"
RA_SA_RATIO_OUTPUT_DIR="${ANALYSIS_DIR}/rca_sa_ra_head_ratio_comparison"
READING_OUTPUT_DIR="${ANALYSIS_DIR}/reading_models"

MATRIX_REBUILT=0
LONG_SCORES_REBUILT=0
BLIMP_SUBTASKS_REBUILT=0
ZERO_SHOT_SUBMEASURES_REBUILT=0
CLASSIFICATION_PSEUDO_ITEMS_REBUILT=0

mkdir -p "${ARCHITECTURE_OUTPUT_DIR}"

all_files_exist() {
    local path
    for path in "$@"; do
        if [[ ! -f "${path}" ]]; then
            return 1
        fi
    done
    return 0
}

csv_header_has_columns() {
    local path="$1"
    shift
    if [[ ! -f "${path}" ]]; then
        return 1
    fi
    python - "$path" "$@" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
required_columns = set(sys.argv[2:])
with path.open(newline="") as handle:
    reader = csv.reader(handle)
    header = next(reader, None)
if header is None:
    raise SystemExit(1)
if not required_columns.issubset(set(header)):
    raise SystemExit(1)
PY
}

csv_experiment_pairs_match() {
    local candidate_csv="$1"
    local reference_csv="$2"
    if [[ ! -f "${candidate_csv}" || ! -f "${reference_csv}" ]]; then
        return 1
    fi
    python - "$candidate_csv" "$reference_csv" <<'PY'
import csv
import sys
from pathlib import Path

candidate_path = Path(sys.argv[1])
reference_path = Path(sys.argv[2])


def experiment_pairs(path: Path) -> set[tuple[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"experiment_name", "experiment_group"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise SystemExit(1)
        return {
            (row["experiment_name"], row["experiment_group"])
            for row in reader
            if row["experiment_name"] and row["experiment_group"]
        }


if experiment_pairs(candidate_path) != experiment_pairs(reference_path):
    raise SystemExit(1)
PY
}

classification_pseudo_items_cover_required_inputs() {
    local candidate_csv="$1"
    local reference_csv="$2"
    if [[ ! -f "${candidate_csv}" || ! -f "${reference_csv}" ]]; then
        return 1
    fi
    python - "$candidate_csv" "$reference_csv" "${CLASSIFICATION_PSEUDO_ITEM_FAMILY_VALUES[@]}" -- "${CLASSIFICATION_PSEUDO_ITEM_TASK_VALUES[@]}" <<'PY'
import csv
import gzip
import sys
from pathlib import Path

candidate_path = Path(sys.argv[1])
reference_path = Path(sys.argv[2])
separator_index = sys.argv.index("--")
families = set(sys.argv[3:separator_index])
tasks = set(sys.argv[separator_index + 1 :])

def open_text(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", newline="", encoding="utf-8")
    return path.open(newline="", encoding="utf-8")

with open_text(reference_path) as handle:
    reader = csv.DictReader(handle)
    required_columns = {"experiment_name", "experiment_group", "analysis_family"}
    if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
        raise SystemExit("long_scores.csv is missing required experiment metadata columns")
    required_pairs = {
        (row["experiment_name"], row["experiment_group"])
        for row in reader
        if row["analysis_family"] in families
    }

if not required_pairs:
    raise SystemExit(
        "long_scores.csv has no rows for classification pseudo-item families: "
        + ", ".join(sorted(families))
    )

with open_text(candidate_path) as handle:
    reader = csv.DictReader(handle)
    required_columns = {"experiment_name", "experiment_group", "task"}
    if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
        raise SystemExit("classification_pseudo_items.csv is missing required columns")
    observed = {
        (row["experiment_name"], row["experiment_group"], row["task"])
        for row in reader
        if row["task"] in tasks
    }

missing = [
    (experiment_name, experiment_group, task)
    for experiment_name, experiment_group in sorted(required_pairs)
    for task in sorted(tasks)
    if (experiment_name, experiment_group, task) not in observed
]
if missing:
    examples = ", ".join(
        f"{experiment_name} / {experiment_group} / {task}"
        for experiment_name, experiment_group, task in missing[:20]
    )
    suffix = "" if len(missing) <= 20 else f" ... ({len(missing)} missing cells total)"
    raise SystemExit(
        "classification_pseudo_items.csv is missing required experiment/task rows: "
        + examples
        + suffix
    )
PY
}

experiment_matrix_matches_roots() {
    local matrix_path="$1"
    if [[ ! -f "${matrix_path}" ]]; then
        return 1
    fi
    python - "$matrix_path" "${EXPERIMENT_ROOT_VALUES[@]}" <<'PY'
import csv
import sys
from pathlib import Path

matrix_path = Path(sys.argv[1])
roots = [Path(value).resolve() for value in sys.argv[2:]]
covered_roots: set[Path] = set()


def root_for(value: str) -> Path | None:
    if value == "":
        return None
    path = Path(value).resolve()
    for root in roots:
        if path == root or root in path.parents:
            return root
    raise SystemExit(1)


with matrix_path.open(newline="") as handle:
    reader = csv.DictReader(handle)
    if reader.fieldnames is None or not {"experiment_dir", "eval_root"}.issubset(reader.fieldnames):
        raise SystemExit(1)
    for row in reader:
        experiment_root = root_for(row["experiment_dir"])
        if experiment_root is None:
            raise SystemExit(1)
        covered_roots.add(experiment_root)
        eval_root = root_for(row["eval_root"])
        if eval_root is not None:
            covered_roots.add(eval_root)

if set(roots) != covered_roots:
    raise SystemExit(1)
PY
}

model_mode_marker() {
    local output_dir="$1"
    local producer="$2"
    printf "%s/.%s_mode" "${output_dir}" "${producer}"
}

model_outputs_current() {
    local marker="$1"
    shift
    if [[ "${FORCE_REBUILD}" == "1" || ! -f "${marker}" ]]; then
        return 1
    fi
    if [[ "$(cat "${marker}")" != "${MIXED_MODEL_MODE_SIGNATURE}" ]]; then
        return 1
    fi
    all_files_exist "$@"
}

write_model_mode_marker() {
    local marker="$1"
    mkdir -p "$(dirname "${marker}")"
    printf "%s\n" "${MIXED_MODEL_MODE_SIGNATURE}" > "${marker}"
}

echo "=== Summarizing experiment outputs ==="
if [[ "${FORCE_REBUILD}" == "1" || ! -f "${MATRIX_CSV}" ]]; then
    python "${SCRIPT_DIR}/summarize_experiment_matrix.py" \
        "${EXPERIMENT_ROOT_ARGS[@]}" \
        --checkpoint_label "${CHECKPOINT_LABEL}" \
        --output_csv "${MATRIX_CSV}"
    MATRIX_REBUILT=1
elif ! experiment_matrix_matches_roots "${MATRIX_CSV}"; then
    echo "  Existing ${MATRIX_CSV} does not match configured experiment roots; rebuilding."
    python "${SCRIPT_DIR}/summarize_experiment_matrix.py" \
        "${EXPERIMENT_ROOT_ARGS[@]}" \
        --checkpoint_label "${CHECKPOINT_LABEL}" \
        --output_csv "${MATRIX_CSV}"
    MATRIX_REBUILT=1
else
    echo "  Using existing ${MATRIX_CSV}"
fi

echo "=== Analyzing experiment matrix ==="
if [[ "${FORCE_REBUILD}" == "1" || "${MATRIX_REBUILT}" == "1" || ! -f "${ANALYSIS_DIR}/long_scores.csv" ]]; then
    python "${SCRIPT_DIR}/analyze_experiment_matrix.py" \
        --input_csv "${MATRIX_CSV}" \
        --output_dir "${ANALYSIS_DIR}" \
        --expected_groups "${EXPECTED_GROUPS}" \
        --expected_seeds "${EXPECTED_SEEDS}"
    LONG_SCORES_REBUILT=1
else
    echo "  Using existing ${ANALYSIS_DIR}/long_scores.csv"
fi

if [[ "${RUN_READING_MODELS}" == "1" ]]; then
    echo "=== Extracting per-experiment reading regression scores ==="
    if [[ -z "${READING_EXPERIMENT_ROOTS}" ]]; then
        echo "RUN_READING_MODELS=1 requires READING_EXPERIMENT_ROOTS to point at experiment dirs with reading eval outputs." >&2
        exit 1
    fi
    IFS=: read -r -a READING_ROOT_VALUES <<< "${READING_EXPERIMENT_ROOTS}"
    if [[ "${#READING_ROOT_VALUES[@]}" -eq 0 ]]; then
        echo "READING_EXPERIMENT_ROOTS must contain at least one root when provided" >&2
        exit 1
    fi
    READING_ROOT_ARGS=()
    for reading_root in "${READING_ROOT_VALUES[@]}"; do
        READING_ROOT_ARGS+=(--experiment_root "${reading_root}")
    done
    mkdir -p "${READING_OUTPUT_DIR}"
    python "${SCRIPT_DIR}/extract_reading_regression_scores.py" \
        "${READING_ROOT_ARGS[@]}" \
        --output_csv "${READING_OUTPUT_DIR}/reading_scores.csv.gz"

    echo "=== Fitting cross-experiment reading models ==="
    Rscript "${SCRIPT_DIR}/metric_models_reading.R" \
        "${READING_OUTPUT_DIR}/reading_scores.csv.gz" \
        "${ANALYSIS_DIR}/experiment_matrix.csv" \
        "${READING_OUTPUT_DIR}"
fi

if [[ "${RUN_RCA_SYMBOL_ANALYSIS}" == "1" || "${RUN_RCA_SYMBOL_MODELS}" == "1" ]]; then
    echo "=== Analyzing RCA symbol-retrieval comparison ==="
    if [[ "${FORCE_REBUILD}" == "1" || "${MATRIX_REBUILT}" == "1" || ! -f "${RCA_SYMBOL_OUTPUT_DIR}/group_summary.csv" ]]; then
        python "${SCRIPT_DIR}/analyze_rca_symbol_comparison.py" \
            --input_csv "${MATRIX_CSV}" \
            --output_dir "${RCA_SYMBOL_OUTPUT_DIR}" \
            --run_tag "${RCA_SYMBOL_RUN_TAG}" \
            --expected_seeds "${EXPECTED_SEEDS}" \
            "${RCA_SYMBOL_SEED_ARGS[@]}" \
            --symbol_modes "${RCA_SYMBOL_MODE_VALUES[@]}"
    else
        echo "  Using existing RCA analysis outputs"
    fi
fi

if [[ "${RUN_METRIC_MODELS}" == "1" || "${RUN_RCA_SYMBOL_MODELS}" == "1" || "${REBUILD_CLASSIFICATION_PSEUDO_ITEMS}" == "1" ]]; then
    echo "=== Summarizing BLiMP subtasks ==="
    if [[ "${FORCE_REBUILD}" == "1" || "${MATRIX_REBUILT}" == "1" || ! -f "${ANALYSIS_DIR}/blimp_subtasks.csv" ]] ||
        ! csv_experiment_pairs_match "${ANALYSIS_DIR}/blimp_subtasks.csv" "${ANALYSIS_DIR}/long_scores.csv"; then
        python "${SCRIPT_DIR}/summarize_subtasks.py" \
            "${EXPERIMENT_ROOT_ARGS[@]}" \
            --output_csv "${ANALYSIS_DIR}/blimp_subtasks.csv" \
            --checkpoint_label "${CHECKPOINT_LABEL}"
        BLIMP_SUBTASKS_REBUILT=1
    else
        echo "  Using existing ${ANALYSIS_DIR}/blimp_subtasks.csv"
    fi
fi

if [[ "${RUN_RCA_SYMBOL_MODELS}" == "1" ]]; then
    echo "=== Summarizing zero-shot submeasures ==="
    if [[ "${FORCE_REBUILD}" == "1" || "${MATRIX_REBUILT}" == "1" || ! -f "${ANALYSIS_DIR}/zero_shot_submeasures.csv" ]] ||
        ! csv_experiment_pairs_match "${ANALYSIS_DIR}/zero_shot_submeasures.csv" "${ANALYSIS_DIR}/long_scores.csv"; then
        python "${SCRIPT_DIR}/summarize_zero_shot_submeasures.py" \
            "${EXPERIMENT_ROOT_ARGS[@]}" \
            --output_csv "${ANALYSIS_DIR}/zero_shot_submeasures.csv" \
            --eval_data_root "${EVAL_DATA_ROOT}" \
            --checkpoint_label "${CHECKPOINT_LABEL}"
        ZERO_SHOT_SUBMEASURES_REBUILT=1
    else
        echo "  Using existing ${ANALYSIS_DIR}/zero_shot_submeasures.csv"
    fi
fi

if [[ "${RUN_METRIC_MODELS}" == "1" || "${RUN_RCA_SYMBOL_MODELS}" == "1" || "${REBUILD_CLASSIFICATION_PSEUDO_ITEMS}" == "1" ]]; then
    echo "=== Extracting classification pseudo-items ==="
    if [[ "${REBUILD_CLASSIFICATION_PSEUDO_ITEMS}" == "1" ]]; then
        if [[ "${#CLASSIFICATION_PSEUDO_ITEM_ROOT_ARGS[@]}" -eq 0 ]]; then
            echo "REBUILD_CLASSIFICATION_PSEUDO_ITEMS=1 requires CLASSIFICATION_PSEUDO_ITEM_ROOTS to point at raw prediction artifact roots." >&2
            exit 1
        fi
        python "${SCRIPT_DIR}/extract_trial_level_data.py" \
            "${CLASSIFICATION_PSEUDO_ITEM_ROOT_ARGS[@]}" \
            --eval_data_root "${EVAL_DATA_ROOT}" \
            --output_csv "${ANALYSIS_DIR}/classification_pseudo_items.csv.gz" \
            --checkpoint_label "${CHECKPOINT_LABEL}" \
            --metadata_csv "${ANALYSIS_DIR}/long_scores.csv"
        CLASSIFICATION_PSEUDO_ITEMS_REBUILT=1
    elif [[ ! -f "${ANALYSIS_DIR}/classification_pseudo_items.csv.gz" \
         && ! -f "${ANALYSIS_DIR}/classification_pseudo_items.csv" ]]; then
        echo "Missing ${ANALYSIS_DIR}/classification_pseudo_items.csv.gz. Build it from raw prediction artifacts with REBUILD_CLASSIFICATION_PSEUDO_ITEMS=1 and CLASSIFICATION_PSEUDO_ITEM_ROOTS=/path/to/raw/root before running compact repo-only analysis." >&2
        exit 1
    fi

    # Resolve the canonical pseudo-item CSV path. Prefer the committed
    # .csv.gz artifact; fall back to a legacy uncompressed .csv if present.
    if [[ -f "${ANALYSIS_DIR}/classification_pseudo_items.csv.gz" ]]; then
        CLASSIFICATION_PSEUDO_ITEMS_CSV="${ANALYSIS_DIR}/classification_pseudo_items.csv.gz"
    else
        CLASSIFICATION_PSEUDO_ITEMS_CSV="${ANALYSIS_DIR}/classification_pseudo_items.csv"
    fi

    if ! classification_pseudo_items_cover_required_inputs "${CLASSIFICATION_PSEUDO_ITEMS_CSV}" "${ANALYSIS_DIR}/long_scores.csv"; then
        echo "${CLASSIFICATION_PSEUDO_ITEMS_CSV} does not cover the required classification pseudo-item families/tasks. Rebuild it from raw prediction artifacts with REBUILD_CLASSIFICATION_PSEUDO_ITEMS=1 and CLASSIFICATION_PSEUDO_ITEM_ROOTS=/path/to/raw/root." >&2
        exit 1
    fi
    if [[ "${CLASSIFICATION_PSEUDO_ITEMS_REBUILT}" != "1" ]]; then
        echo "  Using existing ${CLASSIFICATION_PSEUDO_ITEMS_CSV}"
    fi
fi

if [[ "${RUN_METRIC_MODELS}" == "1" ]]; then
    echo "=== Running metric-specific models ==="
    systematic_marker="$(model_mode_marker "${ARCHITECTURE_OUTPUT_DIR}" "systematic")"
    if [[ "${LONG_SCORES_REBUILT}" == "1" || "${CLASSIFICATION_PSEUDO_ITEMS_REBUILT}" == "1" ]] ||
        ! model_outputs_current "${systematic_marker}" \
        "${ARCHITECTURE_OUTPUT_DIR}/systematic_baseline_vs_dat_anova.csv" \
        "${ARCHITECTURE_OUTPUT_DIR}/systematic_baseline_vs_dat_contrasts.csv" \
        "${ARCHITECTURE_OUTPUT_DIR}/systematic_dat_internal_anova.csv" \
        "${ARCHITECTURE_OUTPUT_DIR}/systematic_dat_internal_contrasts.csv" ||
        ! csv_header_has_columns "${ARCHITECTURE_OUTPUT_DIR}/systematic_baseline_vs_dat_anova.csv" "Chisq" "Pr(>Chisq)" ||
        ! csv_header_has_columns "${ARCHITECTURE_OUTPUT_DIR}/systematic_dat_internal_anova.csv" "Chisq" "Pr(>Chisq)"; then
        Rscript "${SCRIPT_DIR}/metric_models_systematic.R" \
            "${CLASSIFICATION_PSEUDO_ITEMS_CSV}" \
            "${ANALYSIS_DIR}/long_scores.csv" \
            "${ARCHITECTURE_OUTPUT_DIR}"
        write_model_mode_marker "${systematic_marker}"
    else
        echo "  Using existing systematic DAT metric model outputs"
    fi

    interactions_marker="$(model_mode_marker "${ARCHITECTURE_OUTPUT_DIR}" "interactions")"
    if [[ "${LONG_SCORES_REBUILT}" == "1" || "${CLASSIFICATION_PSEUDO_ITEMS_REBUILT}" == "1" ]] ||
        ! model_outputs_current "${interactions_marker}" \
        "${ARCHITECTURE_OUTPUT_DIR}/interaction_model_ling_anova.csv" \
        "${ARCHITECTURE_OUTPUT_DIR}/interaction_model_ling_contrasts.csv" ||
        ! csv_header_has_columns "${ARCHITECTURE_OUTPUT_DIR}/interaction_model_ling_anova.csv" "Chisq" "Pr(>Chisq)"; then
        Rscript "${SCRIPT_DIR}/metric_models_interactions.R" \
            "${CLASSIFICATION_PSEUDO_ITEMS_CSV}" \
            "${ANALYSIS_DIR}/long_scores.csv" \
            "${ARCHITECTURE_OUTPUT_DIR}"
        write_model_mode_marker "${interactions_marker}"
    else
        echo "  Using existing interaction_model_ling_anova.csv"
    fi

    echo "=== Plotting BLiMP summary scores ==="
    if [[ "${FORCE_REBUILD}" == "1" || "${LONG_SCORES_REBUILT}" == "1" || ! -f "${ANALYSIS_DIR}/blimp_means.png" ]]; then
        Rscript "${SCRIPT_DIR}/plot_experiment_results.R" \
            "${ANALYSIS_DIR}/long_scores.csv" \
            "${ANALYSIS_DIR}"
    else
        echo "  Using existing blimp_means.png"
    fi

    echo "=== Plotting subtasks ==="
    if [[ "${FORCE_REBUILD}" == "1" || "${BLIMP_SUBTASKS_REBUILT}" == "1" || "${LONG_SCORES_REBUILT}" == "1" || ! -f "${ANALYSIS_DIR}/blimp_ling_terms.png" ]]; then
        Rscript "${SCRIPT_DIR}/plot_subtasks.R" \
            "${ANALYSIS_DIR}/blimp_subtasks.csv" \
            "${ANALYSIS_DIR}/long_scores.csv" \
            "${ANALYSIS_DIR}"
    else
        echo "  Using existing blimp_ling_terms.png"
    fi
fi

if [[ "${RUN_RCA_SYMBOL_MODELS}" == "1" ]]; then
    echo "=== Running RCA symbol-retrieval models ==="
    rca_symbol_producer="rca_symbol"
    if [[ "${ULTRA_FAST_MIXED_MODELS}" == "1" ]]; then
        rca_symbol_producer="rca_symbol_ultrafast_ling_glm"
    fi
    rca_symbol_marker="$(model_mode_marker "${RCA_SYMBOL_MODEL_OUTPUT_DIR}" "${rca_symbol_producer}")"
    if [[ "${LONG_SCORES_REBUILT}" == "1" || "${CLASSIFICATION_PSEUDO_ITEMS_REBUILT}" == "1" || "${ZERO_SHOT_SUBMEASURES_REBUILT}" == "1" ]] ||
        ! model_outputs_current "${rca_symbol_marker}" \
        "${RCA_SYMBOL_MODEL_OUTPUT_DIR}/rca_subtest_anova.csv" \
        "${RCA_SYMBOL_MODEL_OUTPUT_DIR}/rca_subtest_contrasts.csv" \
        "${RCA_SYMBOL_MODEL_OUTPUT_DIR}/rca_ling_anova.csv" \
        "${RCA_SYMBOL_MODEL_OUTPUT_DIR}/rca_ling_contrasts.csv" \
        "${RCA_SYMBOL_MODEL_OUTPUT_DIR}/rca_symbol_metric_anova.csv" \
        "${RCA_SYMBOL_MODEL_OUTPUT_DIR}/rca_symbol_metric_vs_relative.csv" \
        "${RCA_SYMBOL_MODEL_OUTPUT_DIR}/rca_symbol_submeasure_anova.csv" \
        "${RCA_SYMBOL_MODEL_OUTPUT_DIR}/rca_symbol_submeasure_vs_relative.csv" \
        "${RCA_SYMBOL_MODEL_OUTPUT_DIR}/blimp_subtest_model_summary.tex" \
        "${RCA_SYMBOL_MODEL_OUTPUT_DIR}/submeasure_model_summary.tex" \
        "${RCA_SYMBOL_MODEL_OUTPUT_DIR}/metric_model_summary.tex" ||
        ! csv_header_has_columns "${RCA_SYMBOL_MODEL_OUTPUT_DIR}/rca_subtest_anova.csv" "Chisq" "Pr(>Chisq)" ||
        ! csv_header_has_columns "${RCA_SYMBOL_MODEL_OUTPUT_DIR}/rca_ling_anova.csv" "Chisq" "Pr(>Chisq)" ||
        ! csv_header_has_columns "${RCA_SYMBOL_MODEL_OUTPUT_DIR}/rca_symbol_metric_anova.csv" "Chisq" "Pr(>Chisq)" ||
        ! csv_header_has_columns "${RCA_SYMBOL_MODEL_OUTPUT_DIR}/rca_symbol_submeasure_anova.csv" "Chisq" "Pr(>Chisq)"; then
        Rscript "${SCRIPT_DIR}/metric_models_rca_symbol_comparison.R" \
            "${ANALYSIS_DIR}/long_scores.csv" \
            "${CLASSIFICATION_PSEUDO_ITEMS_CSV}" \
            "${ANALYSIS_DIR}/zero_shot_submeasures.csv" \
            "${RCA_SYMBOL_MODEL_OUTPUT_DIR}" \
            "${RCA_SYMBOL_RUN_TAG}" \
            "${EXPECTED_SEEDS}" \
            "${RCA_SYMBOL_MODES_CSV}"
        write_model_mode_marker "${rca_symbol_marker}"
    else
        echo "  Using existing RCA symbol-retrieval model outputs"
    fi
fi

if [[ "${RUN_RA_SA_RATIO_ANALYSIS}" == "1" ]]; then
    echo "=== Analyzing RCA SA/RA head-ratio sweep ==="
    python "${SCRIPT_DIR}/analyze_ra_sa_ratio_results.py" \
        --input-root "${RA_SA_RATIO_INPUT_ROOT}" \
        --eval-data-root "${EVAL_DATA_ROOT}" \
        --output-dir "${RA_SA_RATIO_OUTPUT_DIR}"

    echo "=== Plotting RCA SA/RA head-ratio sweep ==="
    Rscript "${SCRIPT_DIR}/plot_ra_sa_ratio.R" "${RA_SA_RATIO_OUTPUT_DIR}"

    echo "=== Fitting RCA SA/RA mixed-effects models ==="
    Rscript "${SCRIPT_DIR}/metric_models_ra_sa_ratio_lmer.R" "${RA_SA_RATIO_OUTPUT_DIR}"
fi

echo "Matrix CSV: ${MATRIX_CSV}"
echo "Analysis directory: ${ANALYSIS_DIR}"

if [[ "${RUN_RCA_SYMBOL_MODELS}" == "1" ]]; then
    EXPORT_DIR="${RCA_SYMBOL_EXPORT_DIR:-}"
    if [[ -z "${EXPORT_DIR}" ]]; then
        echo "Set RCA_SYMBOL_EXPORT_DIR to the directory for exported RCA symbol results" >&2
        exit 1
    fi
    if [[ -n "${RCA_SYMBOL_EXPORT_SOURCE_ROOT}" ]]; then
        echo "=== Exporting results to ${EXPORT_DIR} ==="
        python "${SCRIPT_DIR}/export_rca_symbol_results.py" \
            --experiment-root "${RCA_SYMBOL_EXPORT_SOURCE_ROOT}" \
            --analysis-dir "${ANALYSIS_DIR}" \
            --output-dir "${EXPORT_DIR}" \
            --run-tag "${RCA_SYMBOL_RUN_TAG}" \
            --checkpoint-label "${CHECKPOINT_LABEL}" \
            --expected-seeds "${EXPECTED_SEEDS}" \
            "${RCA_SYMBOL_EXPORT_SEED_ARGS[@]}" \
            --symbol-modes "${RCA_SYMBOL_MODE_VALUES[@]}" \
            --overwrite
    else
        echo "Skipping RCA bundle export; set RCA_SYMBOL_EXPORT_SOURCE_ROOT to refresh ${EXPORT_DIR}."
    fi
fi
