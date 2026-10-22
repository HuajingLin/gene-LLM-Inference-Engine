#!/usr/bin/env python3
"""make_dummy.py — Generates a small Llama2.c checkpoint (with random weights) 
and a tokenizer for offline testing.

This allows you to verify the layout expected by the C++ loader and ensure that 
the NumPy reference implementation runs correctly—all without needing to download
 any external files.
(Note: The output generated using random weights will be gibberish; 
this is expected, as the tool is intended only for testing the "pipeline,"
 not for evaluating "model quality.")

Usage:
python tools/make_dummy.py --out dummy              # Shared lm_head
python tools/make_dummy.py --out dummy_u --unshared # Independent lm_head (vocab size in the header is represented as a negative number)
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llama2c_io import Config, tensor_layout  # noqa: E402


def build_vocab() -> tuple[list[bytes], list[float]]:
    vocab: list[bytes] = [b"<unk>", b"\n<s>\n", b"\n</s>\n"]
    scores: list[float] = [0.0, 0.0, 0.0]
    for b in range(256):                      # byte-fallback token：id = byte + 3
        vocab.append(f"<0x{b:02X}>".encode())
        scores.append(0.0)
    singles = [b" "] + [bytes([c]) for c in range(ord("a"), ord("z") + 1)] + [b".", b","]
    merges = [b" t", b"th", b"he", b" the", b"in", b"an", b" a", b"on", b"er", b" s", b"ing"]
    for i, t in enumerate(singles):
        vocab.append(t)
        scores.append(-100.0 - i)             # Single-character scores are very low, so they are prioritized for merging.
    for i, t in enumerate(merges):
        vocab.append(t)
        scores.append(-1.0 - i)               # Mergers that appear earlier take precedence.
    return vocab, scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dummy")
    ap.add_argument("--unshared", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    vocab, scores = build_vocab()
    cfg = Config(dim=16, hidden_dim=40, n_layers=2, n_heads=4, n_kv_heads=2,   # with GQA
                 vocab_size=len(vocab), seq_len=32, shared_classifier=not args.unshared)

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "model.bin", "wb") as f:
        v = cfg.vocab_size if cfg.shared_classifier else -cfg.vocab_size
        f.write(struct.pack("<7i", cfg.dim, cfg.hidden_dim, cfg.n_layers, cfg.n_heads,
                            cfg.n_kv_heads, v, cfg.seq_len))
        for name, shape in tensor_layout(cfg):
            if name.startswith("rms"):
                arr = 1.0 + 0.1 * rng.standard_normal(shape)
            elif name.startswith("freq_cis"):
                ang = np.outer(np.arange(cfg.seq_len),
                               1.0 / 10000.0 ** (np.arange(0, cfg.head_size, 2) / cfg.head_size))
                arr = np.cos(ang) if name.endswith("real") else np.sin(ang)
            else:
                arr = 0.3 * rng.standard_normal(shape)
            f.write(np.ascontiguousarray(arr, dtype="<f4").tobytes())

    with open(out_dir / "tokenizer.bin", "wb") as f:
        f.write(struct.pack("<i", max(len(t) for t in vocab)))
        for t, s in zip(vocab, scores):
            f.write(struct.pack("<f", s))
            f.write(struct.pack("<i", len(t)))
            f.write(t)

    print(f"wrote {out_dir}/model.bin and {out_dir}/tokenizer.bin  (vocab={cfg.vocab_size}, "
          f"shared_classifier={cfg.shared_classifier})")


if __name__ == "__main__":
    main()
