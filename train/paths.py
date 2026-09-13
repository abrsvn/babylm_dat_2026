"""Repo-root-aware training path helpers for training assets and outputs."""

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from train.config import AUTO_ASSET_PATH, TrainConfig
from train.corpus_policy import (
    CORPUS_SERIALIZATION_POLICY,
    DOCUMENT_BOUNDARY_CORPUS_SERIALIZATION_POLICY,
    TOKENIZER_LINE_BATCH_SIZE,
)


BASE_DATASET_NAMES = (
    "bnc_spoken",
    "childes",
    "gutenberg",
    "open_subtitles",
    "simple_wiki",
    "switchboard",
)

def get_dataset_names() -> tuple[str, ...]:
    """Return the ordered corpus names required by both BabyLM tracks."""
    return BASE_DATASET_NAMES


REPO_ROOT = Path(__file__).resolve().parent.parent
REPO_CACHE_ROOT = REPO_ROOT / "cache"
TRAIN_DATASET_CACHE_ROOT = REPO_CACHE_ROOT / "train"
DATASET_CACHE_FORMAT = "token_chunk_array"
DATASET_CACHE_METADATA_FILE = "metadata.json"
DATASET_CACHE_RUNTIME_METADATA_KEYS = ("num_chunks", "bos_token_id", "eos_token_id")


@dataclass(frozen=True)
class TrainPaths:
    repo_root: Path
    corpus_id: str
    tokenizer_dir: Path
    train_data_dir: Path
    cache_dir: Path
    corpus_serialization_policy: str
    dataset_cache_dir: Path
    experiments_root: Path
    experiment_dir: Path
    checkpoint_dir: Path
    log_dir: Path
    train_config_path: Path
    model_config_path: Path


def resolve_repo_path(path_value: str | Path) -> Path:
    base_path = Path(path_value).expanduser()
    if base_path.is_absolute():
        return base_path.resolve()
    return (REPO_ROOT / base_path).resolve()


def resolve_train_asset_path(path_value: str) -> Path:
    if path_value == AUTO_ASSET_PATH:
        raise ValueError(
            "tokenizer_dir must be supplied explicitly because this source release "
            "does not include tokenizer assets."
        )
    return resolve_repo_path(path_value)


def resolve_train_data_path(path_value: str) -> Path:
    if path_value == AUTO_ASSET_PATH:
        raise ValueError(
            "train_data_dir must be supplied explicitly because this source release "
            "does not include training corpora."
        )
    return resolve_repo_path(path_value)


def materialize_train_asset_config(config: TrainConfig) -> TrainConfig:
    """Validate the explicit asset paths required by this source release."""
    resolve_train_asset_path(config.tokenizer_dir)
    resolve_train_data_path(config.train_data_dir)
    return config


def build_asset_cache_key(tokenizer_dir: Path, train_data_dir: Path) -> str:
    """Build a stable short cache key from the resolved tokenizer and corpus roots."""
    digest = sha256(f"{tokenizer_dir}|{train_data_dir}".encode("utf-8")).hexdigest()[:12]
    return f"{tokenizer_dir.name}__{train_data_dir.name}__{digest}"


def build_cache_dir_name(
    datapoint_length: int,
    corpus_serialization_policy: str,
    tokenizer_dir: Path,
    train_data_dir: Path,
) -> str:
    """Build the dataset-cache directory name from assets, chunking, and cache semantics."""
    return (
        f"train_{build_asset_cache_key(tokenizer_dir, train_data_dir)}_"
        f"len{datapoint_length}_batch{TOKENIZER_LINE_BATCH_SIZE}_{DATASET_CACHE_FORMAT}_"
        f"{corpus_serialization_policy}"
    )


def resolve_dataset_cache_base_dir(requested_cache_dir: Path) -> Path:
    """Use one shared train-cache root for repo-local cache paths."""
    requested_cache_dir = requested_cache_dir.expanduser().resolve()
    if requested_cache_dir.is_relative_to(REPO_ROOT) and not requested_cache_dir.is_relative_to(REPO_CACHE_ROOT):
        raise ValueError(
            "Repo-local cache_dir must be under cache/. "
            f"Use cache/train instead of {requested_cache_dir.relative_to(REPO_ROOT)}."
        )
    if requested_cache_dir == REPO_CACHE_ROOT or requested_cache_dir.is_relative_to(REPO_CACHE_ROOT):
        return TRAIN_DATASET_CACHE_ROOT
    return requested_cache_dir


def build_dataset_cache_metadata_contract(
    datapoint_length: int,
    corpus_serialization_policy: str,
    tokenizer_dir: Path,
    train_data_dir: Path,
    tokenization_batch_size: int = TOKENIZER_LINE_BATCH_SIZE,
) -> dict[str, object]:
    """Return the model-independent metadata fields that define a dataset cache."""
    return {
        "cache_format": DATASET_CACHE_FORMAT,
        "tokenizer_dir": str(tokenizer_dir),
        "train_data_dir": str(train_data_dir),
        "corpus_serialization_policy": corpus_serialization_policy,
        "dataset_names": list(BASE_DATASET_NAMES),
        "tokenization_batch_size": tokenization_batch_size,
        "datapoint_length": datapoint_length,
        "token_id_dtype": "int32",
    }


def read_cache_metadata_for_resolution(metadata_path: Path) -> dict[str, object]:
    with metadata_path.open("r") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise ValueError(f"Dataset cache metadata must be a JSON object: {metadata_path}")
    return metadata


def cache_metadata_matches_contract(
    metadata: dict[str, object],
    expected_metadata: dict[str, object],
) -> bool:
    if not all(key in metadata for key in expected_metadata):
        return False
    if not all(key in metadata for key in DATASET_CACHE_RUNTIME_METADATA_KEYS):
        return False
    actual_metadata = {key: metadata[key] for key in expected_metadata}
    return actual_metadata == expected_metadata


def find_dataset_caches_by_metadata(
    search_root: Path,
    expected_metadata: dict[str, object],
) -> list[Path]:
    """Find existing dataset caches with the same tokenizer/corpus/chunking contract."""
    if not search_root.exists():
        return []
    matches = []
    for metadata_path in search_root.rglob(DATASET_CACHE_METADATA_FILE):
        metadata = read_cache_metadata_for_resolution(metadata_path)
        if cache_metadata_matches_contract(metadata, expected_metadata):
            matches.append(metadata_path.parent)
    return sorted(set(matches), key=lambda path: (path.stat().st_mtime_ns, str(path)))


def resolve_dataset_cache_dir(
    requested_cache_dir: Path,
    datapoint_length: int,
    corpus_serialization_policy: str,
    tokenizer_dir: Path,
    train_data_dir: Path,
) -> Path:
    cache_dir = resolve_dataset_cache_base_dir(requested_cache_dir)
    cache_dir_name = build_cache_dir_name(
        datapoint_length,
        corpus_serialization_policy,
        tokenizer_dir,
        train_data_dir,
    )
    canonical_cache_dir = cache_dir / cache_dir_name
    if canonical_cache_dir.exists():
        return canonical_cache_dir

    expected_metadata = build_dataset_cache_metadata_contract(
        datapoint_length,
        corpus_serialization_policy,
        tokenizer_dir,
        train_data_dir,
    )
    existing_cache_dirs = find_dataset_caches_by_metadata(cache_dir.parent, expected_metadata)
    if existing_cache_dirs:
        return existing_cache_dirs[0]
    return canonical_cache_dir


def resolve_phase_dataset_cache_dir(paths: TrainPaths, seq_length: int) -> Path:
    """Resolve the dataset cache directory for one sequence-length schedule phase.

    Passing seq_length == config.datapoint_length reproduces paths.dataset_cache_dir;
    other lengths get their own directory and metadata contract.
    """
    return resolve_dataset_cache_dir(
        paths.cache_dir,
        seq_length,
        paths.corpus_serialization_policy,
        paths.tokenizer_dir,
        paths.train_data_dir,
    )


def resolve_train_paths(config: TrainConfig) -> TrainPaths:
    tokenizer_dir = resolve_train_asset_path(config.tokenizer_dir)
    train_data_dir = resolve_train_data_path(config.train_data_dir)
    experiments_root = resolve_repo_path(config.base_folder)
    experiment_dir = experiments_root / config.experiment_name
    requested_cache_dir = resolve_repo_path(config.cache_dir)
    # Under the document-boundary policy EOS is inserted only at document
    # markers, so no per-line EOS. Distinct policy string => its own dataset
    # cache.
    if config.sequence_boundary_policy == "document_boundary":
        corpus_serialization_policy = DOCUMENT_BOUNDARY_CORPUS_SERIALIZATION_POLICY
    else:
        corpus_serialization_policy = CORPUS_SERIALIZATION_POLICY
    dataset_cache_dir = resolve_dataset_cache_dir(
        requested_cache_dir,
        config.datapoint_length,
        corpus_serialization_policy,
        tokenizer_dir,
        train_data_dir,
    )
    cache_dir = dataset_cache_dir.parent

    return TrainPaths(
        repo_root=REPO_ROOT,
        corpus_id=config.corpus_id,
        tokenizer_dir=tokenizer_dir,
        train_data_dir=train_data_dir,
        cache_dir=cache_dir,
        corpus_serialization_policy=corpus_serialization_policy,
        dataset_cache_dir=dataset_cache_dir,
        experiments_root=experiments_root,
        experiment_dir=experiment_dir,
        checkpoint_dir=experiment_dir / "checkpoints",
        log_dir=experiment_dir / "logging",
        train_config_path=experiment_dir / "logging" / "train_config.yaml",
        model_config_path=experiment_dir / "logging" / "model_config.yaml",
    )
