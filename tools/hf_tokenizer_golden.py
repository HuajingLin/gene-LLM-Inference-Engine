#!/usr/bin/env python3
"""hf_tokenizer_golden.py - what the Hugging Face `tokenizers` library produces for the tokenizer test corpus.

For every text of tools/tok_corpus.py it records the token ids, the decoded text and (for texts without special
tokens) the pieces of the pre-tokenizer, plus the decoding of random id sequences. tests/verify_hf_tokenizer.py
compares the C++ tokenizer, and the Python reference, with this file.

The reference is the `tokenizers` library applied to tokenizer.json itself (Tokenizer.from_file): that is the
definition of what the file means. The ids that transformers' AutoTokenizer produces are recorded too ("auto_ids"),
because some transformers versions rebuild the pipeline instead of following the file; differences are reported by
tests/verify_hf_tokenizer.py but are not what the C++ code is judged against.

Needs:  pip install transformers   (it brings the `tokenizers` library)

Usage:
  python tools/hf_tokenizer_golden.py --model models/SmolLM2-135M --out tok_golden.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tok_corpus import all_cases  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Hugging Face model directory (with tokenizer.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fuzz", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    import tokenizers
    import transformers
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer

    backend = Tokenizer.from_file(str(Path(args.model) / "tokenizer.json"))   # follows the file exactly
    auto = AutoTokenizer.from_pretrained(args.model)                           # what transformers makes of it
    added = [t.content for t in backend.get_added_tokens_decoder().values()]

    cases = []
    for text in all_cases(args.fuzz, args.seed):
        ids = backend.encode(text, add_special_tokens=False).ids
        case = {"text": text, "ids": ids,
                "decoded": backend.decode(ids, skip_special_tokens=False),
                "auto_ids": auto(text, add_special_tokens=False)["input_ids"],
                "pieces": None}
        if not any(a in text for a in added):
            normalized = backend.normalizer.normalize_str(text) if backend.normalizer is not None else text
            case["pieces"] = [piece for piece, _ in backend.pre_tokenizer.pre_tokenize_str(normalized)]
        cases.append(case)

    rng = random.Random(11)
    vocab_size = backend.get_vocab_size(with_added_tokens=True)
    special_ids = sorted(backend.get_added_tokens_decoder().keys())
    sequences = [[rng.randrange(vocab_size) for _ in range(rng.randint(0, 25))] for _ in range(150)]
    sequences += [special_ids, list(range(0, min(vocab_size, 300)))]
    decode_cases = [{"ids": ids, "decoded": backend.decode(ids, skip_special_tokens=False)} for ids in sequences]
    n_auto_differs = sum(1 for c in cases if c["auto_ids"] != c["ids"])
    if n_auto_differs:
        print(f"note: AutoTokenizer (transformers {transformers.__version__}) differs from the tokenizers library "
              f"on {n_auto_differs} of {len(cases)} texts")

    Path(args.out).write_text(json.dumps({
        "model": str(args.model), "tokenizers": tokenizers.__version__, "transformers": transformers.__version__,
        "reference": "tokenizers library applied to tokenizer.json", "vocab_size": vocab_size, "cases": cases, "decode_cases": decode_cases}, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.out}: {len(cases)} texts and {len(decode_cases)} id sequences "
          f"(tokenizers {tokenizers.__version__}, transformers {transformers.__version__})")


if __name__ == "__main__":
    main()
