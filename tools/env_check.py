#!/usr/bin/env python3
"""env_check.py - is this Python environment fit to produce reference results?

The NumPy reference (tools/ref.py) is the answer key for the C++ engine, so it must be computed by a NumPy
whose matrix products are right. This script checks that, and (optionally) whether a golden file looks sane.

  1. prints the Python / NumPy versions and the BLAS library NumPy uses
  2. matrix-vector products in float32 against float64, on random data, on a read-only memory map
     (how ref.py reads weights) and, with --model, on the real weights
  3. with --model and --golden: reruns the first forward pass in THIS environment and compares it with the
     golden file, and flags golden files whose generated tokens are constant (a typical sign of garbage)

Usage:
  python tools/env_check.py
  python tools/env_check.py --model models/stories15M/stories15M.bin --golden ref_out/ref.json
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import platform
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

LIMIT = 1e-4


def rel_error(got: np.ndarray, want: np.ndarray) -> float:
    got, want = np.asarray(got, dtype=np.float64), np.asarray(want, dtype=np.float64)
    if not np.all(np.isfinite(got)):
        return float("inf")
    return float(np.max(np.abs(got - want)) / max(1.0, float(np.max(np.abs(want)))))


def blas_summary() -> list[str]:
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            np.show_config()
    except Exception as e:  # noqa: BLE001
        return [f"(np.show_config failed: {e})"]
    keep = [ln.strip() for ln in buf.getvalue().splitlines()
            if any(k in ln.lower() for k in ("blas", "lapack", "mkl", "openblas", "accelerate"))]
    return keep[:10] or ["(no BLAS information reported)"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="llama2.c .bin file (to test with the real weights)")
    ap.add_argument("--golden", help="ref.json produced by tools/ref.py")
    args = ap.parse_args()
    problems: list[str] = []

    print(f"Python {platform.python_version()} on {platform.platform()}")
    print(f"NumPy  {np.__version__} ({np.__file__})")
    for line in blas_summary():
        print(f"  {line}")

    # ---- matrix-vector products --------------------------------------------------------------
    print(f"\nfloat32 matrix-vector products vs float64 (max relative error, limit {LIMIT:.0e}):")
    rng = np.random.default_rng(0)

    def check(label: str, w: np.ndarray, x: np.ndarray) -> None:
        got = w @ x
        want = np.asarray(w, dtype=np.float64) @ x.astype(np.float64)
        again = w @ x
        e = rel_error(got, want)
        ok = e < LIMIT and np.array_equal(got, again)
        print(f"  {label:<44} {e:9.2e}  {'OK' if ok else 'WRONG'}")
        if not ok:
            problems.append(f"matrix product '{label}' is wrong (error {e:.2e})")

    for rows, cols in ((288, 288), (768, 288), (288, 768), (32000, 288)):
        w = rng.standard_normal((rows, cols)).astype(np.float32)
        check(f"random {rows}x{cols}, in memory", w, rng.standard_normal(cols).astype(np.float32))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "w.bin"
        rng.standard_normal(32000 * 288).astype("<f4").tofile(path)
        mm = np.memmap(path, dtype="<f4", mode="r").reshape(32000, 288)
        check("random 32000x288, read-only memory map", mm, rng.standard_normal(288).astype(np.float32))

    ckpt = None
    if args.model:
        from llama2c_io import read_checkpoint
        ckpt = read_checkpoint(args.model)
        c = ckpt.config
        check(f"real weights: wcls {c.vocab_size}x{c.dim}", ckpt.tensors["wcls"], rng.standard_normal(c.dim).astype(np.float32))
        check(f"real weights: layer 0 w1 {c.hidden_dim}x{c.dim}", ckpt.tensors["w1"][0], rng.standard_normal(c.dim).astype(np.float32))

    # ---- the golden file -----------------------------------------------------------------------
    if args.golden:
        ref = json.loads(Path(args.golden).read_text(encoding="utf-8"))
        gen = ref["generated_tokens"]
        print(f"\ngolden file {args.golden}: model={ref.get('model')!r} prompt={ref.get('prompt')!r} steps={ref.get('steps')}")
        print(f"  prompt tokens {ref['prompt_tokens']}  first generated tokens {gen[:8]}")
        if len(gen) >= 8 and len(set(gen)) == 1:
            problems.append(f"the golden file's generated tokens are all the same ({gen[0]}): it holds garbage")
        first_path = Path(args.golden).parent / ref["first_logits_file"]
        first = np.fromfile(first_path, dtype="<f4")
        if not np.all(np.isfinite(first)):
            problems.append("first_logits.f32 contains NaN or infinity")

        if ckpt is not None:
            from ref import Model
            model = Model(ckpt, max_len=len(ref["prompt_tokens"]))
            logits = None
            for pos, tok in enumerate(ref["prompt_tokens"]):
                logits = model.forward(tok, pos)
            e = rel_error(first, logits) if first.shape == logits.shape else float("inf")
            print(f"  reference forward pass re-run in this environment vs first_logits.f32: {e:.2e}")
            print(f"  this environment predicts token {int(np.argmax(logits))} after the prompt; "
                  f"the file implies {gen[0] if gen else None}")
            if not e < 1e-4:
                problems.append("re-running the reference here does not reproduce first_logits.f32")

    print()
    if problems:
        print("PROBLEMS")
        for p in problems:
            print("  -", p)
        return 1
    print("OK: matrix products are accurate and the golden file (if given) is consistent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
