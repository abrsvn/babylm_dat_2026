"""Build a BPE tokenizer for BabyLM training corpora.

The tokenizer uses BPE with 16 special tokens, NFKC normalization, a GPT-style
regex pre-tokenizer, ByteLevel encoding, and a template post-processor that
prepends ``<s>``. Training data tokenization disables that post-processor by
passing ``add_special_tokens=False``.

Example:
    python scripts/create_tokenizer.py \
        --corpus_dir /path/to/BabyLM-2026-Strict-Small \
        --output_dir /path/to/output-tokenizer \
        --vocab_size 16384 \
        --no_space_before_number
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from tqdm import tqdm
from tokenizers.models import BPE
from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers, processors, Regex
from tokenizers.trainers import BpeTrainer


# ---------------------------------------------------------------------------
# Tokenizer initialization
# ---------------------------------------------------------------------------

def initialize_tokenizer(
    vocab_size: int,
    min_frequency: int,
    space_before_number: bool = True,
) -> tuple[Tokenizer, BpeTrainer]:
    """Initialize the BPE tokenizer and trainer.

    Args:
        vocab_size: Target vocabulary size.
        min_frequency: Minimal number of occurrences of every candidate subword.
        space_before_number: If True (default), the GPT-regex includes an
            optional space before the ``\\p{N}`` alternative (`` ?\\p{N}``).
            If False, the regex uses bare ``\\p{N}``.

    Returns:
        (tokenizer, trainer) pair.
    """
    start_of_text_symbol = "<s>"
    end_of_text_symbol = "</s>"
    unk_symbol = "<unk>"
    mask_symbol = "<mask>"
    pad_symbol = "<pad>"

    special_tokens = [unk_symbol, start_of_text_symbol, end_of_text_symbol, pad_symbol, mask_symbol]
    special_tokens += [f"<special_{i}>" for i in range(11)]

    tokenizer = Tokenizer(BPE(
        unk_token=unk_symbol,
        byte_fallback=False,
        fuse_unk=False,
        ignore_merges=True,
    ))

    tokenizer.normalizer = normalizers.Sequence([
        normalizers.Prepend(" "),
        normalizers.NFKC(),
        normalizers.Replace(Regex("\n"), '\n '),
        normalizers.Replace(Regex(" *\n"), '\n'),
    ])

    number_alt = r" ?\p{N}" if space_before_number else r"\p{N}"
    gpt_regex = (
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+"
        r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*"
        r"|" + number_alt +
        r"| ?[^\s\p{L}\p{N}]+[\r\n/]*"
        r"|\s*[\r\n]+"
        r"|\s+(?!\S)"
        r"|\s+"
    )

    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(
            Regex(gpt_regex),
            behavior="isolated",
            invert=False,
        ),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False, trim_offsets=True),
        pre_tokenizers.Split(Regex(".{1,24}"), behavior="isolated", invert=False),
    ])

    tokenizer.decoder = decoders.Sequence([
        decoders.ByteLevel(add_prefix_space=False, use_regex=False),
        decoders.Strip(' ', 1, 0),
        decoders.Replace("\n ", "\n"),
    ])

    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"{start_of_text_symbol} $A",
        pair=f"{start_of_text_symbol} $A {start_of_text_symbol} $B",
        special_tokens=[(start_of_text_symbol, 1)],
    )

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    return tokenizer, trainer


# ---------------------------------------------------------------------------
# Corpus iteration
# ---------------------------------------------------------------------------

def corpus_line_iterator(corpus_dir: Path, is_json: bool):
    """Yield each line from all .train.txt files in the corpus directory.

    Files are processed in sorted order for reproducibility.
    Empty lines are skipped.

    Args:
        corpus_dir: Directory containing .train.txt corpus files.
        is_json: If True, parse each line as JSON and yield the parsed string.
    """
    corpus_files = sorted(corpus_dir.glob("*.train.txt"))
    if not corpus_files:
        raise FileNotFoundError(f"No .train.txt files found in {corpus_dir}")

    for corpus_file in corpus_files:
        print(f"  Processing {corpus_file.name}...", flush=True)
        with open(corpus_file, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc=corpus_file.name):
                if not line.strip():
                    continue
                if is_json:
                    yield json.loads(line).strip()
                else:
                    yield line


# ---------------------------------------------------------------------------
# Statistics computation
# ---------------------------------------------------------------------------

def calculate_stats(
    tokenizer: Tokenizer,
    corpus_dir: Path,
    is_json: bool,
    vocab_size: int,
    output_path: Path,
) -> None:
    """Compute and save tokenization statistics over the corpus.

    Stats written to output_path:
    - Vocabulary size
    - Average splits per word
    - F_{95%} frequency
    - Sorted subwords by frequency
    """
    counter: Counter = Counter()
    n_words = 0

    for i, text in enumerate(corpus_line_iterator(corpus_dir, is_json)):
        if len(text) == 0:
            continue
        n_words += len(text.split())
        encoding = tokenizer.encode(text)
        counter.update(encoding.tokens)

        if i == 0:
            print("Example of tokenization:", flush=True)
            print(text, flush=True)
            print(tokenizer.decode(encoding.ids), flush=True)
            for j in encoding.ids:
                print(j, tokenizer.id_to_token(j), flush=True)

    sorted_subwords = counter.most_common()
    n_subwords = sum(freq for _, freq in sorted_subwords)

    if n_words == 0 or n_subwords == 0:
        raise ValueError(
            f"No usable text found in corpus at {corpus_dir}. "
            "Check that files contain non-empty lines."
        )

    print(f"Average splits per word: {n_subwords / n_words:.3f}", flush=True)

    f_95 = sorted_subwords[len(sorted_subwords) * 95 // 100][1]
    print(f"F_{{95%}} is {f_95}\n", flush=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"Vocabulary size: {vocab_size}\n")
        f.write(f"Average splits per word: {n_subwords / n_words:.3f}\n")
        f.write(f"F_{{95%}} is {f_95}\n")
        sorted_subwords_str = '\n\t'.join(f"{freq}: {subword}" for subword, freq in sorted_subwords)
        f.write(f"Sorted subwords:\n\t{sorted_subwords_str}\n")


# ---------------------------------------------------------------------------
# Sample tokenization (ported from the test() function)
# ---------------------------------------------------------------------------

SAMPLE_TEXTS = [
    """One of the most impressive long term hobby projects is Robert's Rocket Project. He started building a 100 lbf liquid engine in 2001, fired a regeneratively cooled version in 2007,
started building a regen 250 lbf in 2008.""",
    """what are examples of interfaces that allow you to manage sets of queries (SQL, splunk, lucene/elastic, xpath, whatever other language)?""",
    """### Increasingly seeing a big schism between what I think my research is & what others think it is. I don't do qualitative work and I'm not trained in anthro or theories of race or
gender. I can't supervise students with these interests! I'm a sociophonetician who works on prosody!""",
    """The Northern Lights season is here... Taking these pictures is an art itself and requires preparation, so The Local spoke to an expert to find out how to take awe-inspiring snaps of
the Northern Lights.""",
    """Some people have SOTA facial recognition abilities: "At the very upper end of the performance scale, a cohort of just 1-2% of the population are 'super-recognisers'-people who can
memorise and recall unfamiliar faces, even after the briefest glimpse.\"""",
]


def print_sample_tokenizations(tokenizer: Tokenizer) -> None:
    """Print sample tokenizations to verify the tokenizer works."""
    print("Samples from the tokenizer:", flush=True)

    def test(tokenizer: Tokenizer, text: str) -> str:
        subwords = tokenizer.encode(text).tokens
        return ' '.join(subwords)

    for text in SAMPLE_TEXTS:
        print(f"INPUT:  {text}\nTOKENS: {test(tokenizer, text)}\n", flush=True)


# ---------------------------------------------------------------------------
# Tokenizer config writing
# ---------------------------------------------------------------------------

SPECIAL_TOKENS_MAP = {
    "bos_token": "<s>",
    "eos_token": "</s>",
    "unk_token": "<unk>",
    "sep_token": "</s>",
    "pad_token": "<pad>",
    "cls_token": "<s>",
    "mask_token": "<mask>",
}


def write_tokenizer_config(output_dir: Path) -> None:
    """Write tokenizer_config.json matching the existing asset format."""
    config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        **SPECIAL_TOKENS_MAP,
    }
    config_path = output_dir / "tokenizer_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
    print(f"Saved tokenizer config to {config_path}", flush=True)


def write_special_tokens_map(output_dir: Path) -> None:
    """Write special_tokens_map.json matching the existing asset format."""
    map_path = output_dir / "special_tokens_map.json"
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(SPECIAL_TOKENS_MAP, f, ensure_ascii=False, indent=4)
    print(f"Saved special tokens map to {map_path}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build a BPE tokenizer for BabyLM corpora.")
    parser.add_argument(
        "--corpus_dir", type=Path, required=True,
        help="Directory containing .train.txt corpus files.",
    )
    parser.add_argument(
        "--output_dir", type=Path, required=True,
        help="Directory to write tokenizer.json, tokenizer_config.json, special_tokens_map.json, and stats.",
    )
    parser.add_argument(
        "--vocab_size", type=int, default=2**14,
        help="Target vocabulary size. Default: 16384.",
    )
    parser.add_argument(
        "--min_frequency", type=int, default=0,
        help="Minimal number of occurrences of every candidate subword. Default: 0 (matches original BpeTrainer default).",
    )
    parser.add_argument(
        "--no_space_before_number", action="store_true",
        help="Use bare \\p{N} in the GPT-regex instead of ' ?\\p{N}'.",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Parse each corpus line as JSON (for JSONL corpora). Default: False (raw text lines).",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    space_before_number = not args.no_space_before_number
    print(f"Initializing a BPE tokenizer (vocab_size={args.vocab_size}, min_frequency={args.min_frequency}, space_before_number={space_before_number})", flush=True)
    tokenizer, trainer = initialize_tokenizer(args.vocab_size, args.min_frequency, space_before_number)

    print(f"Training the tokenizer on {args.corpus_dir}", flush=True)
    tokenizer.train_from_iterator(
        corpus_line_iterator(args.corpus_dir, args.json),
        trainer,
    )

    # Save the tokenizer JSON.
    vocab_path = args.output_dir / "tokenizer.json"
    tokenizer.save(str(vocab_path))
    print(f"Saved tokenizer to {vocab_path}", flush=True)

    # Strip the last 256 added tokens from the JSON (matches gpt-bert convention).
    # The tokenizers library adds 256 byte-level tokens to added_tokens when
    # initial_alphabet=pre_tokenizers.ByteLevel.alphabet() is used.
    # Defensive: verify the trailing 256 entries match the exact ByteLevel
    # alphabet before removing them. Fail loudly if the ordering changed.
    byte_alphabet = set(pre_tokenizers.ByteLevel.alphabet())
    with open(vocab_path, "r", encoding="utf-8") as f:
        tokenizer_json = json.load(f)
    if "added_tokens" in tokenizer_json:
        added = tokenizer_json["added_tokens"]
        if len(added) > 256:
            trailing = added[-256:]
            trailing_contents = {t.get("content", "") for t in trailing}
            if trailing_contents != byte_alphabet:
                raise ValueError(
                    "Expected the last 256 added_tokens to be exactly the "
                    "ByteLevel alphabet, but they are not. Refusing to strip."
                )
            tokenizer_json["added_tokens"] = added[:-256]
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(tokenizer_json, f, ensure_ascii=False, indent=4)

    # Verify the tokenizer loads.
    print("TEST", flush=True)
    print("Trying to load the tokenizer...", flush=True)
    tokenizer = Tokenizer.from_file(str(vocab_path))
    print("Success!", flush=True)

    # Write the tokenizer config and special tokens map.
    write_tokenizer_config(args.output_dir)
    write_special_tokens_map(args.output_dir)

    # Compute and save stats.
    print(f"\nComputing tokenization stats over {args.corpus_dir}...", flush=True)
    stats_path = args.output_dir / "tokenizer_stats.txt"
    calculate_stats(tokenizer, args.corpus_dir, args.json, args.vocab_size, stats_path)

    # Print sample tokenizations.
    print_sample_tokenizations(tokenizer)


if __name__ == "__main__":
    main()
