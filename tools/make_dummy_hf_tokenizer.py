#!/usr/bin/env python3
"""make_dummy_hf_tokenizer.py - trains a tiny byte-level BPE tokenizer and writes it in Hugging Face format.

Output directory: tokenizer.json (+ vocab.json, merges.txt, tokenizer_config.json). The layout follows what the
SmolLM2 family uses: special tokens first (ids 0..), then the 256 byte-level characters, then the merges.

Variants (--variant):
  sequence   pre_tokenizer = Sequence[Digits(individual_digits), ByteLevel]       (SmolLM2)
  bytelevel  pre_tokenizer = ByteLevel                                            (plain GPT-2 style)
  noregex    pre_tokenizer = ByteLevel with use_regex = false                     (the whole text is one piece)
  --ignore-merges additionally turns on the ignore_merges option of the BPE model

Usage:
  python tools/make_dummy_hf_tokenizer.py --out dummy_tok --merges 300 --variant sequence
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hf_tok_ref import BYTE_TO_CHAR, gpt2_pieces, split_digits, to_byte_level  # noqa: E402
from tok_corpus import SPECIALS  # noqa: E402

TRAINING_TEXT = """
Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine.
One day, she saw a big, red ball in the sky. It was the sun! She thought it was so pretty.
Lily wanted to play with the ball, but it was too high up in the sky. She tried to jump and reach it, but she couldn't.
"I can't do it," she said. Her mom said, "Don't worry, we'll find another way." They've been friends since 2019.
def add(a, b):
    return a + b

for (int i = 0; i < 100; ++i) { total += values[i] * 2; }
The quick brown fox jumps over the lazy dog. 1234567890 items cost $45.99, or 3.14159 times more.
caf\u00e9 na\u00efve \u00fcber \u65e5\u672c\u8a9e \u4e2d\u6587 \U0001F642 \u0416\u0443\u043a \u03b1\u03b2\u03b3
"""


def train(words: Counter, n_merges: int) -> list[tuple[str, str]]:
    corpus = {tuple(w): f for w, f in words.items()}
    merges: list[tuple[str, str]] = []
    for _ in range(n_merges):
        counts: Counter = Counter()
        for w, f in corpus.items():
            for pair in zip(w, w[1:]):
                counts[pair] += f
        if not counts:
            break
        best = min(counts, key=lambda p: (-counts[p], p))     # most frequent, ties: smallest pair
        merges.append(best)
        merged = best[0] + best[1]
        new_corpus = {}
        for w, f in corpus.items():
            out, i = [], 0
            while i < len(w):
                if i < len(w) - 1 and (w[i], w[i + 1]) == best:
                    out.append(merged)
                    i += 2
                else:
                    out.append(w[i])
                    i += 1
            new_corpus[tuple(out)] = new_corpus.get(tuple(out), 0) + f
        corpus = new_corpus
    return merges


def build_tokenizer(out_dir, n_merges: int = 300, variant: str = "sequence", ignore_merges: bool = False) -> int:
    """Trains and writes the tokenizer files into out_dir; returns the vocabulary size."""
    args = SimpleNamespace(out=out_dir, merges=n_merges, variant=variant, ignore_merges=ignore_merges)
    pieces: list[str] = []
    for line in TRAINING_TEXT.split("\n"):
        text = line + "\n"
        stage = split_digits(text, True) if args.variant == "sequence" else [text]
        for p in stage:
            pieces.extend([p] if args.variant == "noregex" else gpt2_pieces(p))
    words = Counter(to_byte_level(p) for p in pieces)
    merges = train(words, args.merges)

    base = [BYTE_TO_CHAR[b] for b in range(256)]
    vocab: dict[str, int] = {}
    # With ignore_merges a pre-token that is itself in the vocabulary is taken whole. To make that visible,
    # add a few words as tokens although no merge sequence leads to them.
    whole_words = [to_byte_level(w) for w in (" hello", "hello", " world", "Hello", "xyz")] if args.ignore_merges else []
    for tok in SPECIALS + base + [a + b for a, b in merges] + whole_words:
        vocab.setdefault(tok, len(vocab))

    byte_level = {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": args.variant != "noregex"}
    if args.variant == "sequence":
        pre = {"type": "Sequence", "pretokenizers": [{"type": "Digits", "individual_digits": True}, byte_level]}
    else:
        pre = byte_level
    tokenizer = {
        "version": "1.0", "truncation": None, "padding": None,
        "added_tokens": [{"id": vocab[t], "content": t, "single_word": False, "lstrip": False, "rstrip": False,
                          "normalized": False, "special": True} for t in SPECIALS],
        "normalizer": None, "pre_tokenizer": pre, "post_processor": None,
        "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True},
        "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": None,
                  "end_of_word_suffix": None, "fuse_unk": False, "byte_fallback": False, "ignore_merges": args.ignore_merges,
                  "vocab": vocab, "merges": [f"{a} {b}" for a, b in merges]},
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tokenizer.json").write_text(json.dumps(tokenizer, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    (out / "merges.txt").write_text("#version: 0.2\n" + "\n".join(f"{a} {b}" for a, b in merges) + "\n", encoding="utf-8")
    (out / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "GPT2Tokenizer", "bos_token": SPECIALS[0],
                                                           "eos_token": SPECIALS[0], "unk_token": SPECIALS[0]}), encoding="utf-8")
    print(f"wrote {out}/tokenizer.json: {len(vocab)} tokens ({len(SPECIALS)} special + 256 bytes + {len(merges)} merges), variant {args.variant}")
    return len(vocab)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--merges", type=int, default=300)
    ap.add_argument("--variant", choices=["sequence", "bytelevel", "noregex"], default="sequence")
    ap.add_argument("--ignore-merges", action="store_true",
                    help="set ignore_merges and add whole-word tokens that no merge produces")
    a = ap.parse_args()
    build_tokenizer(a.out, a.merges, a.variant, a.ignore_merges)


if __name__ == "__main__":
    main()