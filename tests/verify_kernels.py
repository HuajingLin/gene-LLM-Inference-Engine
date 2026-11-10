#!/usr/bin/env python3
"""verify_kernels.py - M3 acceptance: compare the C++ kernels with NumPy.

Oracles:
  rmsnorm, softmax, rope  -> the very functions in tools/ref.py that produced the M0 story
  matvec, silu, dot, axpy, mul, add -> float64 NumPy (so the check also measures float32 rounding)

Error metric: max|got - want| / max(1, max|want|), required to be < 1e-5.
The sizes deliberately include 1, primes and non-multiples of 8/16/32/64: the SIMD versions must pass the
same cases (loop remainders are a classic bug source).

The whole suite runs once per instruction set the CPU supports (GI_ISA=scalar/avx2/avx512; the program is asked
which one is really active, so a silent fallback to scalar code cannot pass for SIMD) and for several
input addresses (GI_OFFSET: arrays that start 0, 1, 3 or 5 floats into their allocation, so unaligned loads
and stores are exercised; the program also checks that no kernel writes outside its output array).

Usage:
  python tests/verify_kernels.py --exe build/gi_kernel_check
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from hf_io import bf16_to_f32, f32_to_bf16  # noqa: E402
from ref import apply_rope, rmsnorm, softmax  # noqa: E402

TOL = 1e-5
SIZES = [1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 47, 48, 63, 64, 65, 100, 127, 129, 288, 576, 768, 1536]
ENV: dict[str, str] = {}      # GI_ISA / GI_OFFSET for the runs of the current suite


def f32(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype="<f4")


def run(exe: str, args: list, *arrays: np.ndarray) -> np.ndarray:
    data = b"".join(f32(a).tobytes() for a in arrays)
    p = subprocess.run([exe, *map(str, args)], input=data, capture_output=True, env={**os.environ, **ENV})
    if p.returncode != 0:
        raise RuntimeError(f"{args}: {p.stderr.decode(errors='replace').strip()}")
    return np.frombuffer(p.stdout, dtype="<f4")


def run_raw(exe: str, args: list, data: bytes) -> np.ndarray:
    p = subprocess.run([exe, *map(str, args)], input=data, capture_output=True, env={**os.environ, **ENV})
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


def suite(exe: str, seed: int):
    rng = np.random.default_rng(seed)

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
              (768, 288), (288, 768), (32000, 288),      # llama2.c model shapes
              (4, 16), (4, 31), (4, 32), (5, 33), (6, 64), (7, 65), (9, 129), (11, 127), (3, 100), (8, 48),
              (576, 576), (192, 576), (1536, 576), (576, 1536), (2048, 1024), (4099, 576)]   # SmolLM2 / Qwen3 shapes
    for rows, cols in shapes:
        W = rng.standard_normal((rows, cols)).astype(np.float32)
        x = rng.standard_normal(cols).astype(np.float32)
        want = W.astype(np.float64) @ x.astype(np.float64)
        record("matvec", f"{rows}x{cols}", err(run(exe, ["matvec", rows, cols], W, x), want))

    # ---- matvec_bf16: bfloat16 weights ---------------------------------------------------------------
    # (1) accuracy against float64 with the widened weights; (2) BIT-IDENTICAL to matvec() on the same widened
    # matrix (same instruction set, same summation order): widening is exact, so keeping the weights in 16 bits
    # must not change a single bit of the result.
    for rows, cols in shapes:
        if rows * cols > 3_000_000:
            continue
        Wf = rng.standard_normal((rows, cols)).astype(np.float32)
        bits = f32_to_bf16(Wf)
        W = bf16_to_f32(bits)                                  # the values the kernel must see
        x = rng.standard_normal(cols).astype(np.float32)
        want = W.astype(np.float64) @ x.astype(np.float64)
        got = run_raw(exe, ["matvec_bf16", rows, cols], bits.tobytes() + f32(x).tobytes())
        record("matvec_bf16", f"{rows}x{cols}", err(got, want))
        same = run(exe, ["matvec", rows, cols], W, x)
        if got.tobytes() != same.tobytes():
            failures.append(f"matvec_bf16 {rows}x{cols}: not bit-identical to matvec on the widened matrix "
                            f"(max diff {float(np.max(np.abs(got - same))):.3e})")
    # special values survive widening: zeros, negative zero, infinities, the largest and smallest bf16
    bits = np.array([0x0000, 0x8000, 0x7F80, 0xFF80, 0x7F7F, 0x0001, 0x3F80, 0xBF80, 0x0080], dtype=np.uint16)
    W = bf16_to_f32(bits)
    x = np.ones(len(bits), dtype=np.float32)
    for sel in (slice(0, 2), slice(4, 9), slice(0, 4)):      # finite sets and one with +-inf (inf - inf = nan)
        b, w, xx = bits[sel], W[sel], x[sel]
        got = run_raw(exe, ["matvec_bf16", 1, len(b)], b.tobytes() + f32(xx).tobytes())
        ref = run(exe, ["matvec", 1, len(b)], w, xx)
        if got.tobytes() != ref.tobytes():
            failures.append(f"matvec_bf16 special values {sel}: {got} != {ref}")
        record("matvec_bf16", f"special values {sel.start}:{sel.stop}", 0.0)

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
    # theta and pairing style: llama2.c uses (10000, interleaved), Hugging Face Llama (any theta, half_split)
    for style in ("interleaved", "half_split"):
        for theta in (10000.0, 100000.0, 500000.0, 1000000.0):
            for n, hs, pos in ((2, 2, 3), (8, 4, 0), (8, 4, 5), (64, 8, 17), (192, 64, 1), (576, 64, 100),
                               (576, 64, 8191), (128, 128, 2047), (96, 48, 31999)):
                v = rng.standard_normal(n).astype(np.float32)
                record("rope", f"{style} theta={theta:g} n={n} head={hs} pos={pos}",
                       err(run(exe, ["rope", n, pos, hs, repr(theta), style], v),
                           apply_rope(v.copy(), pos, hs, theta, style)))
    # The half_split style written exactly the way Hugging Face's Llama code states it (cos/sin duplicated
    # over both halves, q * cos + rotate_half(q) * sin), in float64: an independent formulation of the
    # semantics, not a copy of the implementation under test.
    def hf_style_rope(x, pos, head_dim, theta):
        heads = x.astype(np.float64).reshape(-1, head_dim)
        inv_freq = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
        emb = np.concatenate([pos * inv_freq, pos * inv_freq])           # [head_dim]
        cos, sin = np.cos(emb), np.sin(emb)
        rotate_half = np.concatenate([-heads[:, head_dim // 2:], heads[:, :head_dim // 2]], axis=-1)
        return (heads * cos + rotate_half * sin).reshape(-1)

    for theta in (10000.0, 100000.0):
        for n, hs, pos in ((16, 16, 1), (192, 64, 9), (576, 64, 100), (128, 32, 3)):
            v = rng.standard_normal(n).astype(np.float32)
            record("rope", f"HF formulation theta={theta:g} n={n} head={hs} pos={pos}",
                   err(run(exe, ["rope", n, pos, hs, repr(theta), "half_split"], v),
                       hf_style_rope(v, pos, hs, theta)))

    v = rng.standard_normal(48).astype(np.float32)  # the two styles must really differ
    a = run(exe, ["rope", 48, 7, 48, "10000", "interleaved"], v)
    b = run(exe, ["rope", 48, 7, 48, "10000", "half_split"], v)
    if np.allclose(a, b, atol=1e-3):
        failures.append("interleaved and half_split RoPE gave the same result")
    v = rng.standard_normal(64).astype(np.float32)  # half_split also preserves the length of each pair
    out = run(exe, ["rope", 64, 33, 32, "100000", "half_split"], v)
    pairs = lambda x: np.sqrt(x[:, :16].astype(np.float64) ** 2 + x[:, 16:].astype(np.float64) ** 2)
    record("rope", "half_split preserves norm", err(pairs(out.reshape(-1, 32)), pairs(v.reshape(-1, 32))))
    v = rng.standard_normal(48).astype(np.float32)  # rotation preserves each pair's length
    out = run(exe, ["rope", 48, 123, 48], v)
    pair_norm = lambda a: np.sqrt(a[0::2].astype(np.float64) ** 2 + a[1::2].astype(np.float64) ** 2)
    record("rope", "preserves norm", err(pair_norm(out), pair_norm(v)))

    return results, failures


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True, help="path to gi_kernel_check")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--offsets", default="0,1,3,5", help="GI_OFFSET values to run")
    args = ap.parse_args()

    info = subprocess.run([args.exe, "isa"], capture_output=True, text=True, env={**os.environ}).stdout
    supported = info.split("supported=")[1].split()[0].split(",")
    print(f"instruction sets supported here: {', '.join(supported)}")
    all_failures: list[str] = []
    total = 0
    last_results = {}
    for isa in supported:
        for offset in [int(o) for o in args.offsets.split(",")]:
            ENV.clear()
            ENV.update({"GI_ISA": isa, "GI_OFFSET": str(offset)})
            got = subprocess.run([args.exe, "isa"], capture_output=True, text=True, env={**os.environ, **ENV}).stdout
            if f"active={isa} " not in got:
                all_failures.append(f"[{isa}] the program did not switch to {isa}: {got.strip()!r}")
                continue
            ops = got.split("simd_ops=")[1].strip()
            want_ops = "" if isa == "scalar" else "matvec,matvec_bf16,rmsnorm,dot,axpy,mul,add"
            if ops != want_ops:
                all_failures.append(f"[{isa}] kernels running SIMD code: {ops!r}, expected {want_ops!r} "
                                    f"(a silent fallback to the scalar version would pass the value checks)")
            try:
                results, failures = suite(args.exe, args.seed)
            except RuntimeError as e:      # the program refused or crashed on some case
                all_failures.append(f"[{isa} offset {offset}] {e}")
                print(f"  isa={isa:<7} offset={offset}: aborted ({e})")
                continue
            n = sum(c for c, _, _ in results.values())
            total += n
            worst = max(w for _, w, _ in results.values())
            print(f"  isa={isa:<7} offset={offset}: {n} cases, worst error {worst:.2e}" + ("  FAIL" if failures else ""))
            all_failures += [f"[{isa} offset {offset}] {f}" for f in failures]
            last_results = results
    ENV.clear()

    # ---- invalid arguments must be rejected, not crash ----------------------------------
    for bad in (["rope", 10, 0, 4], ["rope", 8, 0, 3], ["softmax", 0], ["matvec", 0, 5],
                ["rope", 8, 0, 4, "abc"], ["rope", 8, 0, 4, "1"], ["rope", 8, 0, 4, "10000", "diagonal"]):
        p = subprocess.run([args.exe, *map(str, bad)], input=b"", capture_output=True)
        if p.returncode == 0:
            all_failures.append(f"invalid arguments {bad} were accepted")

    results = last_results
    width = max(len(op) for op in results)
    print(f"{'op':<{width}}  cases  worst error   worst case")
    for op, (n, worst, wcase) in results.items():
        print(f"{op:<{width}}  {n:>5}  {worst:>11.2e}   {wcase}")
    if all_failures:
        print("FAIL")
        for f in all_failures[:40]:
            print("  -", f)
        return 1
    print(f"PASS  ({total} cases over {len(supported)} instruction sets, tolerance {TOL:.0e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
