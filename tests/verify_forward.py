#!/usr/bin/env python3
"""verify_forward.py - M4 acceptance: the C++ forward pass must reproduce the NumPy reference.

The same token sequence is fed through both implementations, one position at a time.
Checks:
  1. logits at EVERY position (this exercises the KV cache over the whole context)
  2. layer-by-layer activations at the last position (embedding, each attention/FFN
     block, final norm) - when something is wrong, the first failing line tells you where
  3. with --golden: the last-prompt-position logits equal ref_out/first_logits.f32, and
     argmax(logits) at every generated position equals the token the reference generated,
     i.e. the C++ model re-derives the whole M0 story greedily

Metric: max|C++ - reference| / max(1, max|reference|), required to be < --tol (default 1e-4).

Usage (real model):
  python tests/verify_forward.py --exe build/gi_forward --model models/stories15M/stories15M.bin \
      --golden ref_out/ref.json
Usage (any model, explicit or random tokens):
  python tests/verify_forward.py --exe build/gi_forward --model dummy/model.bin --random-tokens 20
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from llama2c_io import read_checkpoint  # noqa: E402
from ref import Model  # noqa: E402


def err(got: np.ndarray, want: np.ndarray) -> float:
    got = np.asarray(got, dtype=np.float64)
    want = np.asarray(want, dtype=np.float64)
    if got.shape != want.shape or not np.all(np.isfinite(got)):
        return float("inf")
    return float(np.max(np.abs(got - want)) / max(1.0, np.max(np.abs(want))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True, help="path to gi_forward")
    ap.add_argument("--model", required=True)
    ap.add_argument("--golden", help="ref.json produced by tools/ref.py")
    ap.add_argument("--tokens", help='explicit token ids, e.g. "1 9038 2501"')
    ap.add_argument("--random-tokens", type=int, default=0, help="use N random token ids")
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    ckpt = read_checkpoint(args.model)
    cfg = ckpt.config

    # ---- choose the token sequence --------------------------------------------------
    ref = None
    n_prompt = None
    if args.golden:
        ref = json.loads(Path(args.golden).read_text(encoding="utf-8"))
        if ref.get("vocab_size") not in (None, cfg.vocab_size):
            print(f"FAIL: golden file was generated for vocab_size={ref['vocab_size']}, "
                  f"this model has {cfg.vocab_size}; regenerate it with tools/ref.py")
            return 1
        n_prompt = len(ref["prompt_tokens"])
        seq = list(ref["prompt_tokens"]) + list(ref["generated_tokens"])
        while seq and seq[-1] in (1, 2) and len(seq) > n_prompt:  # drop the BOS/EOS terminator
            seq.pop()
    elif args.tokens:
        seq = [int(t) for t in args.tokens.split()]
    elif args.random_tokens > 0:
        rng = np.random.default_rng(args.seed)
        seq = [1] + [int(t) for t in rng.integers(3, cfg.vocab_size, size=args.random_tokens - 1)]
    else:
        print("error: give --golden, --tokens or --random-tokens")
        return 2
    seq = seq[:cfg.seq_len]
    n = len(seq)
    print(f"sequence: {n} tokens  model: dim={cfg.dim} layers={cfg.n_layers} "
          f"heads={cfg.n_heads}/{cfg.n_kv_heads} vocab={cfg.vocab_size} seq_len={cfg.seq_len}")

    # ---- C++ ------------------------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        p = subprocess.run([args.exe, args.model, tmp, *map(str, seq)], capture_output=True, text=True)
        if p.returncode != 0:
            print(f"FAIL: gi_forward exited with {p.returncode}: {p.stderr.strip()}")
            return 1
        print("C++:", p.stdout.strip())
        cpp_logits = np.fromfile(Path(tmp) / "logits.f32", dtype="<f4")
        if cpp_logits.size != n * cfg.vocab_size:
            print(f"FAIL: logits.f32 has {cpp_logits.size} floats, expected {n * cfg.vocab_size}")
            return 1
        cpp_logits = cpp_logits.reshape(n, cfg.vocab_size)
        cpp_trace = {f.stem: np.fromfile(f, dtype="<f4") for f in Path(tmp).glob("*.f32")
                     if f.stem != "logits"}

    # ---- NumPy reference -------------------------------------------------------------
    model = Model(ckpt)
    ref_logits = np.empty((n, cfg.vocab_size), dtype=np.float32)
    ref_trace: dict[str, np.ndarray] = {}
    for pos, tok in enumerate(seq):
        ref_logits[pos] = model.forward(tok, pos, ref_trace if pos == n - 1 else None)

    failures: list[str] = []

    # ---- 2. layer by layer (last position), in execution order ------------------------
    print(f"\nactivations at the last position (pos {n - 1}):")
    for name in ref_trace:
        if name not in cpp_trace:
            failures.append(f"C++ did not dump '{name}'")
            continue
        e = err(cpp_trace[name], ref_trace[name])
        print(f"  {name:<20} {e:9.2e}")
        if not e < args.tol:
            failures.append(f"activation {name}: error {e:.2e} >= {args.tol:.0e}")

    # ---- 1. logits at every position ----------------------------------------------------
    per_pos = [err(cpp_logits[i], ref_logits[i]) for i in range(n)]
    worst = int(np.argmax(per_pos))
    print(f"\nlogits: worst error {per_pos[worst]:.2e} (at position {worst}) over {n} positions")
    for i, e in enumerate(per_pos):
        if not e < args.tol:
            failures.append(f"logits at position {i}: error {e:.2e} >= {args.tol:.0e}")

    # ---- 3. golden checks ----------------------------------------------------------------
    if ref is not None:
        first = np.fromfile(Path(args.golden).parent / ref["first_logits_file"], dtype="<f4")
        e = err(cpp_logits[ref["first_logits_position"]], first)
        print(f"first_logits.f32 (position {ref['first_logits_position']}): {e:.2e}")
        if not e < args.tol:
            failures.append(f"C++ logits differ from first_logits.f32: {e:.2e}")

        mismatches = near_ties = checked = 0
        for pos in range(n_prompt - 1, n - 1):  # position pos predicts seq[pos + 1]
            checked += 1
            top, want = int(np.argmax(cpp_logits[pos])), seq[pos + 1]
            if top != want:
                gap = float(cpp_logits[pos][top] - cpp_logits[pos][want])
                if gap < 1e-3:
                    near_ties += 1  # a float rounding tie, not a bug
                else:
                    mismatches += 1
                    failures.append(f"greedy token at position {pos}: C++ {top}, reference {want} "
                                    f"(logit gap {gap:.3f})")
        print(f"greedy generation: {checked - mismatches - near_ties}/{checked} tokens identical"
              + (f", {near_ties} near-ties" if near_ties else ""))

    if failures:
        print("\nFAIL")
        for f in failures[:15]:
            print("  -", f)
        if len(failures) > 15:
            print(f"  ... and {len(failures) - 15} more")
        return 1
    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
