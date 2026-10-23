#!/usr/bin/env python3
"""verify_tokenizer.py - M2 acceptance: compare the C++ tokenizer with the Python reference.

Checks:
  1. encode: C++ ids == Python ids, for fixed edge cases and seeded random fuzz
  2. decode: C++ text == Python text for the ids produced above
  3. round trip: decode(encode(text)) == text for texts without leading whitespace
  4. robustness: random invalid UTF-8 input never crashes and yields valid ids
  5. error handling: bad token id and wrong vocab_size are rejected
  6. (optional) --golden ref_out/ref.json: prompt ids and generated story match M0 output

Usage:
  python tests/verify_tokenizer.py --exe build/gi_tokenize \
      --tokenizer models/stories15M/tokenizer.bin --vocab-size 32000 \
      --golden ref_out/ref.json

Note: this file is ASCII-only; non-ASCII test characters are written as escapes.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from llama2c_io import Tokenizer  # noqa: E402

FIXED_CASES = [
    "",
    "a",
    " ",
    "Once upon a time",
    "Once upon a time, there was a little girl named Lily.",
    "Hello, world!",
    "  two leading spaces",
    "trailing spaces  ",
    "multiple   spaces   inside",
    "line one\nline two\n\nline four",
    "tab\tseparated\tvalues",
    "UPPER lower MiXeD Case",
    "numbers 12345 and 3.14159",
    "punctuation: ;:!?'\"()[]{}<>/\\|@#$%^&*-+=~`",
    "caf\u00e9 na\u00efve fa\u00e7ade",            # 2-byte UTF-8
    "\u65e5\u672c\u8a9e\u306e\u30c6\u30ad\u30b9\u30c8",  # 3-byte UTF-8
    "emoji \U0001F642 and \U0001F680 test",         # 4-byte UTF-8
    "control \x01\x02\x1f chars",
    "the the the the the the",
    "a" * 200,
    "aaa", "aaaa", "aaaaa", "aaaaaa",     # equal-score ties: leftmost pair must win
    "   ", "    ", "     ",               # runs of spaces (ties in real vocabularies)
    "She said, \"I can't do it!\" But he replied: yes.",
]

FUZZ_ALPHABET = (
    list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    + [" "] * 8 + list(".,;:!?'\"-()\n\t")
    + ["\u00e9", "\u00fc", "\u65e5", "\u672c", "\U0001F642"]
)


def run(exe: str, tok_path: str, vocab: int, mode: str, data: bytes, *flags: str):
    return subprocess.run([exe, tok_path, str(vocab), mode, *flags],
                          input=data, capture_output=True)


def cpp_encode(exe, tok_path, vocab, text: bytes, *flags) -> list[int]:
    p = run(exe, tok_path, vocab, "encode", text, *flags)
    if p.returncode != 0:
        raise RuntimeError(f"encode failed: {p.stderr.decode(errors='replace')}")
    return [int(x) for x in p.stdout.split()]


def cpp_decode(exe, tok_path, vocab, ids: list[int]) -> bytes:
    p = run(exe, tok_path, vocab, "decode", (" ".join(map(str, ids))).encode())
    if p.returncode != 0:
        raise RuntimeError(f"decode failed: {p.stderr.decode(errors='replace')}")
    return p.stdout


def py_decode(tok: Tokenizer, ids: list[int]) -> bytes:
    """Same semantics as the C++ Tokenizer::decode: BOS/EOS print nothing."""
    out, prev = b"", -1
    for t in ids:
        if t not in (Tokenizer.BOS, Tokenizer.EOS):
            out += tok.decode_piece(prev, t)
        prev = t
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True, help="path to gi_tokenize")
    ap.add_argument("--tokenizer", required=True, help="path to tokenizer.bin")
    ap.add_argument("--vocab-size", type=int, default=32000)
    ap.add_argument("--fuzz", type=int, default=150, help="number of random texts")
    ap.add_argument("--golden", help="ref.json produced by tools/ref.py")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    exe, path, vocab = args.exe, args.tokenizer, args.vocab_size
    py = Tokenizer(path, vocab)
    errors: list[str] = []
    n_encode = n_decode = n_round = 0

    rng = random.Random(args.seed)
    cases = list(FIXED_CASES)
    for _ in range(args.fuzz):
        cases.append("".join(rng.choice(FUZZ_ALPHABET) for _ in range(rng.randint(0, 60))))

    # ---- 1-3: encode / decode / round trip -----------------------------------------
    for text in cases:
        want = py.encode(text, bos=True, eos=False)
        got = cpp_encode(exe, path, vocab, text.encode("utf-8"))
        n_encode += 1
        if got != want:
            errors.append(f"encode mismatch for {text!r}\n    C++:    {got}\n    Python: {want}")
            continue

        n_decode += 1
        got_text, want_text = cpp_decode(exe, path, vocab, want), py_decode(py, want)
        if got_text != want_text:
            errors.append(f"decode mismatch for {text!r}\n    C++:    {got_text!r}\n"
                          f"    Python: {want_text!r}")

        if text and not text[0].isspace():
            n_round += 1
            if got_text != text.encode("utf-8"):
                errors.append(f"round trip failed for {text!r}: got {got_text!r}")

    # flags
    for flags, kw in ((("--no-bos",), dict(bos=False)), (("--eos",), dict(eos=True)),
                      (("--no-bos", "--eos"), dict(bos=False, eos=True))):
        text = "Once upon a time"
        want = py.encode(text, **{"bos": True, "eos": False, **kw})
        got = cpp_encode(exe, path, vocab, text.encode(), *flags)
        if got != want:
            errors.append(f"flags {flags}: C++ {got} != Python {want}")

    # ---- 4: robustness on random (mostly invalid UTF-8) bytes ----------------------
    for _ in range(30):
        blob = bytes(rng.randrange(256) for _ in range(rng.randint(1, 40)))
        p = run(exe, path, vocab, "encode", blob)
        if p.returncode != 0:
            errors.append(f"random bytes crashed the tokenizer: {blob!r}")
            continue
        ids = [int(x) for x in p.stdout.split()]
        if not all(0 <= i < vocab for i in ids):
            errors.append(f"random bytes produced out-of-range ids: {blob!r} -> {ids}")

    # ---- 5: error handling ---------------------------------------------------------
    if run(exe, path, vocab, "decode", f"1 {vocab}".encode()).returncode == 0:
        errors.append("decode accepted an out-of-range token id")
    if run(exe, path, vocab, "decode", b"1 abc").returncode == 0:
        errors.append("decode accepted non-numeric input")
    if run(exe, path, vocab + 1, "encode", b"hi").returncode == 0:
        errors.append("vocab_size + 1 was not rejected (file should be truncated)")
    if run(exe, path, vocab - 1, "encode", b"hi").returncode == 0:
        errors.append("vocab_size - 1 was not rejected (file should have trailing data)")

    # ---- 6: golden data from M0 ----------------------------------------------------
    if args.golden:
        ref = json.loads(Path(args.golden).read_text(encoding="utf-8"))
        ref_vocab = ref.get("vocab_size")
        ref_ids = ref["prompt_tokens"]
        got = cpp_encode(exe, path, vocab, ref["prompt"].encode("utf-8"))
        print(f"golden file: {args.golden}")
        print(f"  generated from model: {ref.get('model', '<unknown>')}")
        print(f"  prompt: {ref['prompt']!r}")
        print(f"  C++ ids: {got}")
        print(f"  ref ids: {ref_ids}")

        if ref_vocab is not None and ref_vocab != vocab:
            # The reference belongs to a different model, so comparing ids is meaningless.
            errors.append(f"golden file was generated with vocab_size={ref_vocab} but this run "
                          f"uses --vocab-size {vocab}; regenerate it with tools/ref.py using the "
                          f"same model and tokenizer")
        elif ref_vocab is None and max(ref_ids) >= vocab:
            errors.append("golden file contains ids outside this vocabulary; it was probably "
                          "generated from a different model - regenerate it with tools/ref.py")
        else:
            if got != ref_ids:
                errors.append("golden prompt ids differ (if 'generated from model' above is not "
                              "the model you are testing, regenerate the golden file)")
            full = cpp_decode(exe, path, vocab, ref_ids + ref["generated_tokens"])
            if full.decode("utf-8", errors="replace") != ref["prompt"] + ref["text"]:
                errors.append("golden story differs when decoding prompt + generated tokens")

    if errors:
        print("FAIL")
        for e in errors:
            print("  -", e)
        return 1
    print(f"PASS  (encode: {n_encode} texts, decode: {n_decode}, round trip: {n_round}, "
          f"vocab={vocab})")
    return 0


if __name__ == "__main__":
    sys.exit(main())