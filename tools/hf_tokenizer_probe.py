#!/usr/bin/env python3
"""hf_tokenizer_probe.py - which tokenizer implementation follows tokenizer.json?

Prints the pre-tokenizer that tokenizer.json declares, then, for a few revealing texts, the pieces and token ids from
  (1) the `tokenizers` library applied to tokenizer.json,
  (2) transformers' AutoTokenizer (some versions rebuild the pipeline instead of following the file),
  (3) optionally this project's C++ tokenizer (--exe build/gi_hf_tokenize).
and a verdict on who agrees with whom.

Usage:
  python tools/hf_tokenizer_probe.py --model models/SmolLM2-135M --exe build/gi_hf_tokenize
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

TEXTS = ["123", "a1b2c3", "x=42;y=7", "  12", "   5j", "1,000,000", "\u00bd \u00b2 \u2155 fractions and superscripts",
         "Once upon a time", "Hello world, it is 2024!"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--exe", help="path to gi_hf_tokenize (optional)")
    args = ap.parse_args()

    import tokenizers
    import transformers
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer

    path = Path(args.model) / "tokenizer.json"
    declared = json.loads(path.read_text(encoding="utf-8"))
    print(f"tokenizers {tokenizers.__version__}, transformers {transformers.__version__}")
    print("pre_tokenizer declared in tokenizer.json:")
    print(json.dumps(declared["pre_tokenizer"], indent=1))
    print("normalizer:", declared.get("normalizer"), "| post_processor:", declared.get("post_processor"),
          "| decoder:", (declared.get("decoder") or {}).get("type"))

    lib = Tokenizer.from_file(str(path))
    auto = AutoTokenizer.from_pretrained(args.model)
    print(f"AutoTokenizer class: {type(auto).__name__}")
    try:
        state = json.loads(auto.backend_tokenizer.pre_tokenizer.__getstate__())
        print("pre_tokenizer inside AutoTokenizer:")
        print(json.dumps(state, indent=1))
    except Exception as e:  # noqa: BLE001
        print(f"(could not read the pre_tokenizer of AutoTokenizer: {e})")

    lib_ids = [lib.encode(t, add_special_tokens=False).ids for t in TEXTS]
    auto_ids = [auto(t, add_special_tokens=False)["input_ids"] for t in TEXTS]
    cpp_ids = None
    if args.exe:
        out = subprocess.run([args.exe, args.model, "encode-batch"], input=json.dumps(TEXTS).encode(), capture_output=True)
        cpp_ids = json.loads(out.stdout) if out.returncode == 0 else None
        if cpp_ids is None:
            print("C++ tokenizer failed:", out.stderr.decode(errors="replace").strip())

    print("\n%-40s %s" % ("text", "tokenizers library | AutoTokenizer | C++"))
    for i, text in enumerate(TEXTS):
        pieces = [p for p, _ in lib.pre_tokenizer.pre_tokenize_str(text)]
        flags = ["lib==auto" if lib_ids[i] == auto_ids[i] else "lib!=auto"]
        if cpp_ids is not None:
            flags.append("lib==cpp" if lib_ids[i] == cpp_ids[i] else "lib!=cpp")
        print(f"{text!r:<40} {' '.join(flags)}")
        print(f"    library pieces: {pieces}")
        print(f"    library ids   : {lib_ids[i][:14]}")
        if lib_ids[i] != auto_ids[i]:
            print(f"    auto ids      : {auto_ids[i][:14]}")
        if cpp_ids is not None and lib_ids[i] != cpp_ids[i]:
            print(f"    C++ ids       : {cpp_ids[i][:14]}")

    print()
    lib_auto = all(a == b for a, b in zip(lib_ids, auto_ids))
    print("tokenizers library vs AutoTokenizer:", "identical" if lib_auto else "DIFFERENT (transformers does not follow tokenizer.json here)")
    if cpp_ids is not None:
        print("tokenizers library vs C++ tokenizer:", "identical" if lib_ids == cpp_ids else "DIFFERENT")
        print("AutoTokenizer      vs C++ tokenizer:", "identical" if auto_ids == cpp_ids else "DIFFERENT")


if __name__ == "__main__":
    main()
