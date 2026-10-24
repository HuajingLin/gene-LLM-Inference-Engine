#!/usr/bin/env python3
"""verify_kernels.py - M3 acceptance: compare the C++ kernels with NumPy.

Oracles:
  rmsnorm, softmax, rope  -> the very functions in tools/ref.py that produced the M0 story
  matvec, silu, dot, axpy, mul, add -> float64 NumPy (so the check also measures float32 rounding)

Error metric: max|got - want| / max(1, max|want|), required to be < 1e-5.
The sizes deliberately include 1, primes and non-multiples of 8/16/32: later SIMD
versions must still pass the same cases (loop remainders are a classic bug source).

Usage:
  python tests/verify_kernels.py --exe build/gi_kernel_check
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from ref import apply_rope, rmsnorm, softmax  # noqa: E402

TOL = 1e-5
SIZES = [1, 2, 3, 7, 8, 15, 16, 17, 31, 32, 33, 100, 288, 768]


def f32(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype="<f4")


def run(exe: str, args: list, *arrays: np.ndarray) -> np.ndarray:
    data = b"".join(f32(a).tobytes() for a in arrays)
    p = subprocess.run([exe, *map(str, args)], input=data, capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(f"{args}: {p.stderr.decode(errors='replace').strip()}")
    return np.frombuffer(p.stdout, dtype="<f4")


def err(got: np.ndarray, want: np.ndarray) -> float:
    want = np.asarray(want, dtype=np.float64)
    if got.shape != want.shape:
        return float("inf")
    if not np.all(np.isfinite(got)) and np.all(np.isfinite(want)):
        return float("inf")
    return float(np.max(np.abs(got.astype(np.float64) - want)) / max(1.0, np.max(np.abs(want))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True, help="path to gi_kernel_check")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    exe = args.exe
    rng = np.random.default_rng(args.seed)

    results: dict[str, tuple[int, float, str]] = {}  # op -> (cases, worst error, worst case)
    failures: list[str] = []

    def record(op: str, case: str, e: float) -> None:
        n, worst, wcase = results.get(op, (0, 0.0, ""))
        if e > worst:
            worst, wcase = e, case
        results[op] = (n + 1, worst, wcase)
        if not e < TOL:
            failures.append(f"{op} {case}: error {e:.3e} >= {TOL:.0e}")

    # ---- matvec: non-square shapes catch transposed indexing ------------------------
    shapes = [(1, 1), (1, 17), (17, 1), (7, 13), (13, 7), (33, 65), (100, 3), (288, 288),
              (768, 288), (288, 768), (32000, 288)]  # the last three are real model shapes
    for rows, cols in shapes:
        W = rng.standard_normal((rows, cols)).astype(np.float32)
        x = rng.standard_normal(cols).astype(np.float32)
        want = W.astype(np.float64) @ x.astype(np.float64)
        record("matvec", f"{rows}x{cols}", err(run(exe, ["matvec", rows, cols], W, x), want))

    # ---- rmsnorm ---------------------------------------------------------------------
    for n in SIZES:
        x = rng.standard_normal(n).astype(np.float32)
        w = (1.0 + 0.1 * rng.standard_normal(n)).astype(np.float32)
        record("rmsnorm", f"n={n}", err(run(exe, ["rmsnorm", n], x, w), rmsnorm(x, w)))
    z = np.zeros(16, dtype=np.float32)  # all zeros: only eps keeps this finite
    record("rmsnorm", "zeros", err(run(exe, ["rmsnorm", 16], z, np.ones(16, np.float32)), z))
    big = (1000.0 * rng.standard_normal(64)).astype(np.float32)  # large magnitudes
    w = np.ones(64, dtype=np.float32)
    record("rmsnorm", "large values", err(run(exe, ["rmsnorm", 64], big, w), rmsnorm(big, w)))

    # ---- softmax ---------------------------------------------------------------------
    for n in SIZES + [256, 32000]:
        x = (3.0 * rng.standard_normal(n)).astype(np.float32)
        record("softmax", f"n={n}", err(run(exe, ["softmax", n], x), softmax(x)))
    x = (1000.0 + rng.standard_normal(10)).astype(np.float32)  # overflows without max-subtraction
    record("softmax", "values near 1000", err(run(exe, ["softmax", 10], x), softmax(x)))
    x = np.full(9, -500.0, dtype=np.float32)
    record("softmax", "all equal", err(run(exe, ["softmax", 9], x), np.full(9, 1 / 9)))
    x = rng.standard_normal(40).astype(np.float32)
    s = run(exe, ["softmax", 40], x).sum()
    record("softmax", "sums to 1", abs(float(s) - 1.0))

    # ---- silu, mul, add ----------------------------------------------------------------
    with np.errstate(over="ignore"):
        for n in SIZES:
            x = (4.0 * rng.standard_normal(n)).astype(np.float32)
            x64 = x.astype(np.float64)
            record("silu", f"n={n}", err(run(exe, ["silu", n], x), x64 / (1.0 + np.exp(-x64))))
        x = np.array([-100, -30, -1, 0, 1, 30, 100], dtype=np.float32)  # saturation range
        x64 = x.astype(np.float64)
        record("silu", "extreme values", err(run(exe, ["silu", len(x)], x), x64 / (1.0 + np.exp(-x64))))
    for n in SIZES:
        a = rng.standard_normal(n).astype(np.float32)
        b = rng.standard_normal(n).astype(np.float32)
        record("mul", f"n={n}", err(run(exe, ["mul", n], a, b), a.astype(np.float64) * b))
        record("add", f"n={n}", err(run(exe, ["add", n], a, b), a.astype(np.float64) + b))

    # ---- dot, axpy (attention building blocks) ------------------------------------------
    for n in SIZES:
        a = rng.standard_normal(n).astype(np.float32)
        b = rng.standard_normal(n).astype(np.float32)
        record("dot", f"n={n}", err(run(exe, ["dot", n], a, b),
                                    np.array([a.astype(np.float64) @ b.astype(np.float64)])))
        y = rng.standard_normal(n).astype(np.float32)
        alpha = float(np.float32(rng.standard_normal()))  # exactly representable as float32
        record("axpy", f"n={n}", err(run(exe, ["axpy", n, repr(alpha)], y, a),
                                     y.astype(np.float64) + alpha * a.astype(np.float64)))

    # ---- rope: (n, head_size, pos). Real shapes: dim 288 / head_size 48 ------------------
    rope_cases = [(2, 2, 5), (8, 4, 0), (8, 4, 3), (16, 8, 1), (288, 48, 0), (288, 48, 1),
                  (288, 48, 100), (288, 48, 255), (96, 48, 17), (128, 64, 2047),
                  (256, 64, 8191), (64, 64, 31999)]  # large positions test double-precision angles
    for n, hs, pos in rope_cases:
        v = rng.standard_normal(n).astype(np.float32)
        record("rope", f"n={n} head={hs} pos={pos}",
               err(run(exe, ["rope", n, pos, hs], v), apply_rope(v.copy(), pos, hs)))
    v = rng.standard_normal(48).astype(np.float32)  # rotation preserves each pair's length
    out = run(exe, ["rope", 48, 123, 48], v)
    pair_norm = lambda a: np.sqrt(a[0::2].astype(np.float64) ** 2 + a[1::2].astype(np.float64) ** 2)
    record("rope", "preserves norm", err(pair_norm(out), pair_norm(v)))

    # ---- invalid arguments must be rejected, not crash ----------------------------------
    for bad in (["rope", 10, 0, 4], ["rope", 8, 0, 3], ["softmax", 0], ["matvec", 0, 5]):
        p = subprocess.run([exe, *map(str, bad)], input=b"", capture_output=True)
        if p.returncode == 0:
            failures.append(f"invalid arguments {bad} were accepted")

    width = max(len(op) for op in results)
    print(f"{'op':<{width}}  cases  worst error   worst case")
    for op, (n, worst, wcase) in results.items():
        print(f"{op:<{width}}  {n:>5}  {worst:>11.2e}   {wcase}")
    if failures:
        print("FAIL")
        for f in failures:
            print("  -", f)
        return 1
    print(f"PASS  ({sum(n for n, _, _ in results.values())} cases, tolerance {TOL:.0e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
