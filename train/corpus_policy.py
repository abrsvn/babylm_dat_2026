"""Corpus serialization policies for BabyLM training data."""

from collections.abc import Iterator
from itertools import islice
from pathlib import Path


TOKENIZER_LINE_BATCH_SIZE = 1024
LINE_EOS_CORPUS_SERIALIZATION_POLICY = "line_eos_boundary"
CHILDES_LINE_EOS_CORPUS_SERIALIZATION_POLICY = "childes_line_eos_boundary"
CORPUS_SERIALIZATION_POLICY = "default_line_eos_boundary__childes_line_eos_boundary"
# Document-boundary regime: no per-line EOS. EOS is inserted only at `= = =`
# document markers (and never in marker-less corpora, which each become a single
# document). Newlines are kept as the intra-document line separator. Distinct
# string => distinct cache key, so this never collides with the line-EOS caches.
DOCUMENT_BOUNDARY_CORPUS_SERIALIZATION_POLICY = "document_boundary"
TRAIN_FILE_SUFFIX = ".train.txt"


def get_corpus_name(corpus_path: Path) -> str:
    """Return the canonical corpus name from a BabyLM training filename."""
    corpus_file_name = corpus_path.name
    if not corpus_file_name.endswith(TRAIN_FILE_SUFFIX):
        raise ValueError(
            f"Training corpus file must end with {TRAIN_FILE_SUFFIX}, got {corpus_file_name}"
        )
    return corpus_file_name[: -len(TRAIN_FILE_SUFFIX)]


def get_corpus_serialization_policy(corpus_name: str) -> str:
    """Map each corpus to the serialization policy that should feed tokenization."""
    if corpus_name == "childes":
        return CHILDES_LINE_EOS_CORPUS_SERIALIZATION_POLICY
    return LINE_EOS_CORPUS_SERIALIZATION_POLICY


def iter_raw_text_batches(
    corpus_path: Path,
    batch_size: int = TOKENIZER_LINE_BATCH_SIZE,
) -> Iterator[list[str]]:
    """Yield raw file text batches without reconstructing or normalizing line boundaries."""
    with corpus_path.open("r") as handle:
        while True:
            line_batch = list(islice(handle, batch_size))
            if not line_batch:
                return
            yield line_batch


def iter_childes_text_batches(
    corpus_path: Path,
    batch_size: int = TOKENIZER_LINE_BATCH_SIZE,
) -> Iterator[list[str]]:
    """Yield CHILDES text batches through a local policy hook.

    This first-pass implementation preserves raw file contents exactly. It is
    deliberately not trying to recreate the old newline-token patch. Any later
    CHILDES-specific boundary strengthening should happen here as an explicit
    corpus-policy experiment rather than as tokenizer mutation.
    """
    yield from iter_raw_text_batches(corpus_path, batch_size=batch_size)


def iter_corpus_text_batches(
    corpus_path: Path,
    batch_size: int = TOKENIZER_LINE_BATCH_SIZE,
) -> Iterator[list[str]]:
    """Yield corpus text batches according to the corpus-specific serialization policy."""
    corpus_name = get_corpus_name(corpus_path)
    serialization_policy = get_corpus_serialization_policy(corpus_name)

    if serialization_policy == CHILDES_LINE_EOS_CORPUS_SERIALIZATION_POLICY:
        yield from iter_childes_text_batches(corpus_path, batch_size=batch_size)
        return

    yield from iter_raw_text_batches(corpus_path, batch_size=batch_size)


DOCUMENT_MARKER_PREFIX = "= = = "


def is_document_marker_line(line: str) -> bool:
    """Return True for `= = = ... = = =` document boundary lines."""
    return line.lstrip().startswith(DOCUMENT_MARKER_PREFIX)


DOCUMENT_MARKER_SUFFIX = " = = ="
# Corpora whose `= = =` marker lines carry metadata (transcript paths, book ids)
# rather than content. Under the document-boundary policy these marker lines are
# dropped entirely; other corpora keep the inner title as content.
MARKER_METADATA_CORPORA = frozenset({"childes", "gutenberg"})


def corpus_drops_marker_text(corpus_name: str) -> bool:
    """True if this corpus's document-marker lines are metadata to drop, not content."""
    return corpus_name in MARKER_METADATA_CORPORA


def extract_document_marker_title(line: str) -> str:
    """Return the inner title of a `= = = Title = = =` marker line, decoration removed."""
    stripped = line.strip()
    if not stripped.startswith(DOCUMENT_MARKER_PREFIX):
        raise ValueError(f"Not a document marker line: {line!r}")
    inner = stripped[len(DOCUMENT_MARKER_PREFIX) :]
    if inner.endswith(DOCUMENT_MARKER_SUFFIX):
        inner = inner[: -len(DOCUMENT_MARKER_SUFFIX)]
    return inner
