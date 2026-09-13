"""Training dataset and tokenizer loading."""

from collections.abc import Iterator
from dataclasses import dataclass
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from train.config import TrainConfig
from train.corpus_policy import (
    DOCUMENT_BOUNDARY_CORPUS_SERIALIZATION_POLICY,
    TOKENIZER_LINE_BATCH_SIZE,
    corpus_drops_marker_text,
    extract_document_marker_title,
    get_corpus_name,
    is_document_marker_line,
    iter_corpus_text_batches,
)
from train.paths import (
    TrainPaths,
    build_dataset_cache_metadata_contract,
    get_dataset_names,
    resolve_phase_dataset_cache_dir,
)


logger = logging.getLogger(__name__)
CACHE_METADATA_FILE = "metadata.json"
TOKEN_CHUNKS_FILE = "token_chunks.int32.bin"
TOKEN_CHUNK_LENGTHS_FILE = "token_chunk_lengths.int32.bin"
CACHE_GENERATIONS_DIR = "generations"
CACHE_GENERATION_FIELD = "generation"
TOKEN_ID_DTYPE = np.int32


class FullBabyLMDataset(Dataset):
    def __init__(
        self,
        token_chunks: np.ndarray,
        token_chunk_lengths: np.ndarray,
        bos_token_id: int,
        eos_token_id: int,
    ):
        self.token_chunks = token_chunks
        self.token_chunk_lengths = token_chunk_lengths
        self.model_bos = bos_token_id
        self.model_eos = eos_token_id

    def __len__(self) -> int:
        return int(self.token_chunk_lengths.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        chunk_length = int(self.token_chunk_lengths[idx])
        chunk = torch.tensor(self.token_chunks[idx, :chunk_length], dtype=torch.long)
        chunk_with_specials = torch.cat(
            [
                torch.tensor([self.model_bos], dtype=torch.long),
                chunk,
                torch.tensor([self.model_eos], dtype=torch.long),
            ]
        )
        # actual_length is the real token count including BOS and EOS.
        # collate_fn builds validity masks from this so real EOS tokens
        # (which equal pad_token_id for GPT-2) are preserved as valid
        # prediction targets rather than masked as padding.
        actual_length = chunk_length + 2
        return chunk_with_specials, actual_length


@dataclass(frozen=True)
class TrainDataArtifacts:
    tokenizer: PreTrainedTokenizerBase
    dataset: FullBabyLMDataset
    dataloader: DataLoader


def load_tokenizer(paths: TrainPaths) -> PreTrainedTokenizerBase:
    if not paths.tokenizer_dir.exists():
        raise FileNotFoundError(f"Tokenizer directory does not exist: {paths.tokenizer_dir}")
    tokenizer = AutoTokenizer.from_pretrained(str(paths.tokenizer_dir))
    if tokenizer.bos_token_id is None:
        raise ValueError(f"Tokenizer is missing bos_token_id: {paths.tokenizer_dir}")
    if tokenizer.eos_token_id is None:
        raise ValueError(f"Tokenizer is missing eos_token_id: {paths.tokenizer_dir}")
    return tokenizer


def validate_corpus_paths(paths: TrainPaths) -> list[Path]:
    dataset_names = get_dataset_names()
    corpus_paths = [paths.train_data_dir / f"{dataset_name}.train.txt" for dataset_name in dataset_names]
    missing_paths = [path for path in corpus_paths if not path.exists()]
    if missing_paths:
        missing_text = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Missing training corpus files: {missing_text}")
    return corpus_paths


def build_corpus_token_chunks(
    corpus_path: Path,
    tokenizer: PreTrainedTokenizerBase,
    chunk_size: int,
    text_batch_size: int = TOKENIZER_LINE_BATCH_SIZE,
    use_document_boundary: bool = False,
) -> Iterator[list[int]]:
    # Corpus serialization is decided before tokenization so corpus-specific
    # boundary policy stays local to data loading rather than tokenizer assets.
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if text_batch_size <= 0:
        raise ValueError(f"text_batch_size must be positive, got {text_batch_size}")
    corpus_name = get_corpus_name(corpus_path)
    if tokenizer.eos_token_id is None:
        raise ValueError(f"Tokenizer is missing eos_token_id: {tokenizer}")
    eos_token_id = int(tokenizer.eos_token_id)
    buffered_token_ids: list[int] = []
    # Document-boundary state, tracked across text batches within this corpus file.
    # emitted_document_content stays False until the first content token is buffered
    # so the first document never gets a stray leading EOS. Corpora whose markers are
    # metadata (childes paths, gutenberg ids) drop the marker line's text.
    emitted_document_content = False
    drop_marker_text = corpus_drops_marker_text(corpus_name) if use_document_boundary else False
    for text_batch in iter_corpus_text_batches(corpus_path, batch_size=text_batch_size):
        if use_document_boundary:
            # No per-line EOS. EOS is inserted only at document markers, and only
            # once content precedes the marker, so it acts as a separator between
            # documents. Newlines are kept as the intra-document line boundary.
            tokenization_inputs: list[str] = []
            for line in text_batch:
                if is_document_marker_line(line):
                    tokenization_inputs.append(
                        "" if drop_marker_text else extract_document_marker_title(line) + "\n"
                    )
                else:
                    tokenization_inputs.append(line)
            tokenized_lines = tokenizer(tokenization_inputs, add_special_tokens=False, verbose=False)["input_ids"]
            for line_text, line_token_ids in zip(text_batch, tokenized_lines):
                if is_document_marker_line(line_text):
                    if emitted_document_content:
                        buffered_token_ids.append(eos_token_id)
                    if drop_marker_text:
                        continue
                buffered_token_ids.extend(line_token_ids)
                if line_token_ids:
                    emitted_document_content = True
        else:
            tokenized_lines = tokenizer(text_batch, add_special_tokens=False, verbose=False)["input_ids"]
            for line_token_ids in tokenized_lines:
                buffered_token_ids.extend(line_token_ids)
                buffered_token_ids.append(eos_token_id)
        while len(buffered_token_ids) >= chunk_size:
            yield buffered_token_ids[:chunk_size]
            buffered_token_ids = buffered_token_ids[chunk_size:]
    if buffered_token_ids:
        yield buffered_token_ids


def build_dataset_cache_at(
    paths: TrainPaths,
    tokenizer: PreTrainedTokenizerBase,
    seq_length: int,
    cache_dir: Path,
) -> dict[str, object]:
    corpus_paths = validate_corpus_paths(paths)
    chunk_size = seq_length
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Building tokenized dataset cache at {cache_dir}")
    token_chunks_path = cache_dir / TOKEN_CHUNKS_FILE
    token_chunk_lengths_path = cache_dir / TOKEN_CHUNK_LENGTHS_FILE

    use_document_boundary = (
        paths.corpus_serialization_policy == DOCUMENT_BOUNDARY_CORPUS_SERIALIZATION_POLICY
    )

    num_chunks = 0
    with (
        token_chunks_path.open("wb") as token_chunks_handle,
        token_chunk_lengths_path.open("wb") as lengths_handle,
    ):
        for corpus_path in corpus_paths:
            logger.info(f"Tokenizing training corpus for cache: {corpus_path.name}")
            for token_chunk in build_corpus_token_chunks(
                corpus_path,
                tokenizer,
                chunk_size,
                use_document_boundary=use_document_boundary,
            ):
                token_row = np.zeros(chunk_size, dtype=TOKEN_ID_DTYPE)
                token_row[: len(token_chunk)] = np.asarray(token_chunk, dtype=TOKEN_ID_DTYPE)
                token_row.tofile(token_chunks_handle)
                np.asarray([len(token_chunk)], dtype=TOKEN_ID_DTYPE).tofile(lengths_handle)
                num_chunks += 1

    if num_chunks == 0:
        raise ValueError(f"No training token chunks were produced from {paths.train_data_dir}")

    metadata = build_dataset_cache_metadata(paths, tokenizer, seq_length, num_chunks)
    write_dataset_cache_metadata(cache_dir, metadata)
    logger.info(f"Built tokenized dataset cache with {num_chunks} chunks")
    return metadata


def build_dataset_cache_metadata(
    paths: TrainPaths,
    tokenizer: PreTrainedTokenizerBase,
    seq_length: int,
    num_chunks: int,
) -> dict[str, object]:
    metadata = build_dataset_cache_metadata_contract(
        seq_length,
        paths.corpus_serialization_policy,
        paths.tokenizer_dir,
        paths.train_data_dir,
    )
    metadata.update(
        {
            "num_chunks": num_chunks,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
    )
    return metadata


def write_dataset_cache_metadata(cache_dir: Path, metadata: dict[str, object]) -> None:
    with (cache_dir / CACHE_METADATA_FILE).open("w") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")


def read_dataset_cache_metadata(cache_dir: Path) -> dict[str, object]:
    with (cache_dir / CACHE_METADATA_FILE).open("r") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise ValueError(f"Dataset cache metadata must be a JSON object: {cache_dir / CACHE_METADATA_FILE}")
    return metadata


def resolve_dataset_cache_paths(
    cache_dir: Path,
    metadata: dict[str, object],
) -> tuple[Path, Path]:
    """Resolve both binary files from one metadata snapshot."""
    generation_name = metadata.get(CACHE_GENERATION_FIELD)
    data_dir = cache_dir
    if generation_name is not None:
        if (
            not isinstance(generation_name, str)
            or not generation_name
            or generation_name in {".", ".."}
            or Path(generation_name).name != generation_name
        ):
            raise ValueError(
                f"Dataset cache generation must be one directory name, got {generation_name!r}"
            )
        data_dir = cache_dir / CACHE_GENERATIONS_DIR / generation_name
    return data_dir / TOKEN_CHUNKS_FILE, data_dir / TOKEN_CHUNK_LENGTHS_FILE


def _read_complete_dataset_cache(
    cache_dir: Path,
) -> tuple[dict[str, object], Path, Path] | None:
    if not (cache_dir / CACHE_METADATA_FILE).exists():
        return None
    metadata = read_dataset_cache_metadata(cache_dir)
    chunks_path, lengths_path = resolve_dataset_cache_paths(cache_dir, metadata)
    if not chunks_path.exists() or not lengths_path.exists():
        return None
    return metadata, chunks_path, lengths_path


def validate_cached_dataset_metadata(
    metadata: dict[str, object],
    paths: TrainPaths,
    tokenizer: PreTrainedTokenizerBase,
    seq_length: int,
) -> None:
    expected_metadata = build_dataset_cache_metadata_contract(
        seq_length,
        paths.corpus_serialization_policy,
        paths.tokenizer_dir,
        paths.train_data_dir,
    )
    expected_metadata.update(
        {
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
    )
    # tokenization_batch_size is provenance, not contract: it describes how
    # the cache was built and should not be re-derived from the current
    # training config. Read it from stored metadata instead.
    expected_metadata.pop("tokenization_batch_size", None)
    # A key absent from a stale/older cache is a mismatch, not a crash: use .get
    # so a missing field (e.g. dataset_names in a pre-variant cache) surfaces as
    # the descriptive ValueError below rather than a KeyError.
    actual_metadata = {key: metadata.get(key) for key in expected_metadata}
    if actual_metadata != expected_metadata:
        raise ValueError(
            "Cached dataset metadata does not match the current train assets: "
            f"expected {expected_metadata}, got {actual_metadata}"
        )
    if not isinstance(metadata["num_chunks"], int) or metadata["num_chunks"] <= 0:
        raise ValueError(f"Cached dataset num_chunks must be positive, got {metadata['num_chunks']}")
    stored_batch_size = metadata.get("tokenization_batch_size")
    if not isinstance(stored_batch_size, int) or stored_batch_size <= 0:
        raise ValueError(
            "Cached dataset tokenization_batch_size must be a positive integer, "
            f"got {stored_batch_size!r}"
        )


def load_or_create_dataset(
    config: TrainConfig,
    paths: TrainPaths,
    tokenizer: PreTrainedTokenizerBase,
) -> FullBabyLMDataset:
    return load_or_create_dataset_at(
        paths, tokenizer, config.datapoint_length, paths.dataset_cache_dir,
    )


def load_or_create_phase_dataset(
    paths: TrainPaths,
    tokenizer: PreTrainedTokenizerBase,
    seq_length: int,
) -> FullBabyLMDataset:
    cache_dir = resolve_phase_dataset_cache_dir(paths, seq_length)
    return load_or_create_dataset_at(
        paths, tokenizer, seq_length, cache_dir,
    )


def load_or_create_dataset_at(
    paths: TrainPaths,
    tokenizer: PreTrainedTokenizerBase,
    seq_length: int,
    cache_dir: Path,
) -> FullBabyLMDataset:
    cache_state = _read_complete_dataset_cache(cache_dir)
    if cache_state is None:
        build_dataset_cache_at(paths, tokenizer, seq_length, cache_dir)
        cache_state = _read_complete_dataset_cache(cache_dir)
        if cache_state is None:
            raise RuntimeError(
                f"Expected cache at {cache_dir} after build, "
                "but cache files are missing."
            )
    else:
        logger.info(f"Loading tokenized dataset cache from {cache_dir}")

    metadata, chunks_path, lengths_path = cache_state
    validate_cached_dataset_metadata(metadata, paths, tokenizer, seq_length)
    num_chunks = int(metadata["num_chunks"])
    datapoint_length = int(metadata["datapoint_length"])
    token_chunks = np.memmap(
        chunks_path,
        dtype=TOKEN_ID_DTYPE,
        mode="r",
        shape=(num_chunks, datapoint_length),
    )
    token_chunk_lengths = np.memmap(
        lengths_path,
        dtype=TOKEN_ID_DTYPE,
        mode="r",
        shape=(num_chunks,),
    )

    return FullBabyLMDataset(
        token_chunks=token_chunks,
        token_chunk_lengths=token_chunk_lengths,
        bos_token_id=int(metadata["bos_token_id"]),
        eos_token_id=int(metadata["eos_token_id"]),
    )


def get_collate_fn(pad_token_id: int):
    """Build a collate function that produces validity masks from lengths.

    Returns input_tokens, target_tokens, target_mask, attention_mask. The
    validity masks are built from actual sequence lengths (the second element
    of each __getitem__ tuple), not by testing for equality with pad_token_id.
    This preserves real EOS tokens (which equal pad_token_id for GPT-2) as
    valid prediction targets rather than masking them as padding.
    """

    def collate_fn(
        batch: list[tuple[torch.Tensor, int]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sequences = [item[0] for item in batch]
        actual_lengths = torch.tensor([item[1] for item in batch], dtype=torch.long)
        tokens = pad_sequence(
            sequences, padding_value=pad_token_id, batch_first=True
        )
        input_tokens = tokens[:, :-1]
        target_tokens = tokens[:, 1:]
        seq_len = tokens.shape[1] - 1
        # Build validity masks from actual sequence lengths, not == pad_token_id.
        # target_mask: True for positions 0..actual_length-2 (real target tokens
        # including EOS), False for padding positions.
        # attention_mask: True for positions 0..actual_length-1 (real input tokens
        # including BOS and EOS), False for padding positions.
        positions = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        target_mask = positions < (actual_lengths - 1).unsqueeze(1)
        attention_mask = positions < actual_lengths.unsqueeze(1)
        return input_tokens, target_tokens, target_mask, attention_mask

    return collate_fn


def resolve_pad_token_id(tokenizer: PreTrainedTokenizerBase) -> int:
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer must define pad_token_id or eos_token_id for padding.")
    return int(pad_token_id)


def build_dataloader_for_dataset(
    dataset: FullBabyLMDataset,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    seed: int = 0,
    epoch: int = 0,
) -> DataLoader:
    # A RandomSampler backed by an isolated generator so the permutation is
    # deterministic (seed/epoch) and does not consume global torch RNG before
    # dropout.
    generator = torch.Generator()
    generator.manual_seed(seed + epoch)
    sampler = RandomSampler(dataset, generator=generator)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=get_collate_fn(resolve_pad_token_id(tokenizer)),
    )



def load_train_data(config: TrainConfig, paths: TrainPaths) -> TrainDataArtifacts:
    tokenizer = load_tokenizer(paths)
    dataset = load_or_create_dataset(config, paths, tokenizer)
    dataloader = build_dataloader_for_dataset(
        dataset, tokenizer, config.batch_size,
        seed=config.seed,
    )
    return TrainDataArtifacts(tokenizer=tokenizer, dataset=dataset, dataloader=dataloader)
