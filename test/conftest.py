"""Pytest bootstrap for repo-local imports and tiny fixtures."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train.config import TrainConfig

TINY_TOKENIZER_DIR = "test/fixtures/tiny_tokenizer"
TINY_CORPUS_DIR = "test/fixtures/train_assets/clean_train_tiny"


def build_tiny_train_config(
    tmp_path: Path,
    experiment_name: str = "tiny-train-contract",
    *,
    model_type: str = "self_attention",
    n_epochs: int = 1,
    warmup_ratio: float = 0.01,
    **overrides,
) -> TrainConfig:
    values = {
        "model_type": model_type,
        "pe_type": "rope",
        "hidden_dim": 32,
        "n_heads": 4,
        "n_heads_sa": 2,
        "n_heads_ra": 2,
        "n_layers": 2,
        "dropout": 0.0,
        "datapoint_length": 8,
        "corpus_id": "strict_small",
        "n_epochs": n_epochs,
        "batch_size": 2,
        "learning_rate": 0.0001,
        "weight_decay": 0.0,
        "warmup_ratio": warmup_ratio,
        "gradient_clip_norm": 1.0,
        "seed": 0,
        "tokenizer_dir": TINY_TOKENIZER_DIR,
        "train_data_dir": TINY_CORPUS_DIR,
        "cache_dir": str(tmp_path / "cache"),
        "base_folder": str(tmp_path / "experiments"),
        "experiment_name": experiment_name,
        "use_wandb": False,
    }
    values.update(overrides)
    return TrainConfig(**values)
