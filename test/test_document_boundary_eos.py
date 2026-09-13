"""EOS placement under the document_boundary serialization policy.

These tests exercise the real build_corpus_token_chunks with the repo tokenizer.
The document_boundary regime inserts no per-line EOS: EOS marks only real document
boundaries (`= = =` marker lines), and only once content precedes the marker.
simple_wiki keeps the marker's inner title (decoration stripped); childes/gutenberg
drop the marker text; marker-less corpora get no EOS at all.
"""

from pathlib import Path

from conftest import build_tiny_train_config
from train.data import build_corpus_token_chunks, load_tokenizer
from train.paths import resolve_train_paths


def _tok(tokenizer, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False, verbose=False)["input_ids"]


def _document_boundary_stream(tokenizer, corpus_path: Path) -> list[int]:
    # text_batch_size=2 forces markers into a later batch than the content that
    # precedes them, exercising the cross-batch "content emitted" boundary state.
    chunks = build_corpus_token_chunks(
        corpus_path,
        tokenizer,
        chunk_size=1024,
        text_batch_size=2,
        use_document_boundary=True,
    )
    return [token_id for chunk in chunks for token_id in chunk]


def _load_tokenizer(tmp_path: Path):
    config = build_tiny_train_config(tmp_path, "doc-boundary-eos")
    paths = resolve_train_paths(config)
    return load_tokenizer(paths)


def _contains_subsequence(haystack: list[int], needle: list[int]) -> bool:
    if not needle:
        return False
    return any(haystack[i : i + len(needle)] == needle for i in range(len(haystack) - len(needle) + 1))


def test_document_boundary_wiki_strips_title_and_eos_only_between_docs(tmp_path: Path) -> None:
    tokenizer = _load_tokenizer(tmp_path)
    eos = tokenizer.eos_token_id
    corpus_path = tmp_path / "simple_wiki.train.txt"
    corpus_path.write_text(
        "= = = Bullet the Blue Sky = = =\n"
        "Body one.\n"
        "= = = Second Title = = =\n"
        "Body two.\n"
    )

    stream = _document_boundary_stream(tokenizer, corpus_path)

    expected = (
        # First document: title decoration stripped, no leading EOS (nothing precedes it).
        _tok(tokenizer, "Bullet the Blue Sky\n")
        + _tok(tokenizer, "Body one.\n")
        # Single boundary EOS closes document one.
        + [eos]
        # Second document: stripped title kept as content, then body.
        + _tok(tokenizer, "Second Title\n")
        + _tok(tokenizer, "Body two.\n")
    )
    assert stream == expected
    # Exactly one boundary between two documents; no per-line EOS.
    assert stream.count(eos) == 1
    # The `= = =` decoration must not survive anywhere in the stream.
    assert not _contains_subsequence(stream, _tok(tokenizer, "= = ="))


def test_document_boundary_childes_drops_marker_text_and_eos_after_preamble(tmp_path: Path) -> None:
    tokenizer = _load_tokenizer(tmp_path)
    eos = tokenizer.eos_token_id
    marker_path = "childes/CHILDES_NA/Brown/Eve/020100b.cha"
    corpus_path = tmp_path / "childes.train.txt"
    corpus_path.write_text(
        # Content before the first marker: this is document zero (untitled).
        "hi there.\n"
        "more talk.\n"
        f"= = = {marker_path} = = =\n"
        "next utterance.\n"
    )

    stream = _document_boundary_stream(tokenizer, corpus_path)

    expected = (
        # Document zero: leading pre-marker content, no leading EOS.
        _tok(tokenizer, "hi there.\n")
        + _tok(tokenizer, "more talk.\n")
        # Boundary EOS closes document zero (content precedes the marker).
        + [eos]
        # Document one: marker metadata dropped entirely, only the utterance remains.
        + _tok(tokenizer, "next utterance.\n")
    )
    assert stream == expected
    assert stream.count(eos) == 1
    # The transcript path tokens must not appear anywhere in the stream.
    path_tokens = _tok(tokenizer, marker_path)
    assert not any(
        stream[i : i + len(path_tokens)] == path_tokens
        for i in range(len(stream) - len(path_tokens) + 1)
    )


def test_document_boundary_markerless_corpus_has_no_eos(tmp_path: Path) -> None:
    tokenizer = _load_tokenizer(tmp_path)
    eos = tokenizer.eos_token_id
    corpus_path = tmp_path / "bnc_spoken.train.txt"
    corpus_path.write_text("line one.\nline two.\nline three.\n")

    stream = _document_boundary_stream(tokenizer, corpus_path)

    expected = (
        _tok(tokenizer, "line one.\n")
        + _tok(tokenizer, "line two.\n")
        + _tok(tokenizer, "line three.\n")
    )
    assert stream == expected
    # A corpus with no `= = =` markers is a single document: zero interior EOS.
    assert stream.count(eos) == 0
