#!/usr/bin/env python3
"""verify_sampler.py - M5 acceptance for the sampler.

Checks:
  1. random generator: the first draws equal an independent pure-Python implementation of
     splitmix64 seeding + xorshift64* (exact equality)
  2. deterministic sampling: for many (logits, temperature, top_k, top_p) combinations and
     many uniform numbers u, the C++ token equals a NumPy float64 oracle that follows the
     semantics documented in src/sampler/sampler.h. Cases that sit within 1e-6 of a decision
     boundary (where float32 and float64 may legitimately disagree) are counted and skipped.
  3. greedy decoding: argmax, lowest index on ties, independent of u
  4. statistics: with the internal generator, empirical frequencies match the filtered
     distribution (5 sigma), and filtered-out tokens never appear
  5. reproducibility: same seed -> same draws, different seed -> different draws
  6. invalid configurations are rejected

Usage:
  python tests/verify_sampler.py --exe build/gi_sampler_check
"""
from __future__ import annotations

import argparse
import subprocess
import sys

import numpy as np

MASK = (1 << 64) - 1


# ----------------------------------------------------------------------------- RNG oracle
def splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & MASK
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK
    return x ^ (x >> 31)


def py_uniforms(seed: int, count: int) -> np.ndarray:
    state = splitmix64(seed) or 1
    out = np.empty(count, dtype=np.float32)
    for i in range(count):
        state ^= state >> 12
        state ^= (state << 25) & MASK
        state ^= state >> 27
        u32 = ((state * 0x2545F4914F6CDD1D) & MASK) >> 32
        out[i] = (u32 >> 8) / 16777216.0
    return out


# ----------------------------------------------------------------------------- sampling oracle
def softmax64(logits: np.ndarray, temperature: float) -> np.ndarray:
    z = (logits.astype(np.float64) - np.max(logits.astype(np.float64))) / temperature
    e = np.exp(z)
    return e / e.sum()


def kept_tokens(p: np.ndarray, top_k: int, top_p: float, tol: float = 1e-6):
    """Returns (token order, probabilities in that order, ambiguous flag)."""
    v = len(p)
    ambiguous = False
    if not (0 < top_k < v) and top_p >= 1.0:
        return np.arange(v), p, False                      # no filtering: index order
    order = np.lexsort((np.arange(v), -p))                 # prob descending, index ascending
    if 0 < top_k < v:
        if abs(p[order[top_k - 1]] - p[order[top_k]]) < 1e-7:
            ambiguous = True                               # which tie member is kept is unclear
        order = order[:top_k]
    kp = p[order]
    if top_p < 1.0:
        total = kp.sum()
        cum = np.cumsum(kp)
        threshold = top_p * total
        if np.min(np.abs(cum - threshold)) < tol:
            ambiguous = True
        reached = cum >= threshold
        n = int(np.argmax(reached)) + 1 if reached.any() else len(kp)
        order, kp = order[:n], kp[:n]
    return order, kp, ambiguous


def oracle_sample(logits, temperature, top_k, top_p, u, tol: float = 1e-6):
    if temperature <= 0:
        return int(np.argmax(logits)), False
    p = softmax64(logits, temperature)
    order, kp, ambiguous = kept_tokens(p, top_k, top_p, tol)
    cum = np.cumsum(kp)
    r = float(u) * cum[-1]
    if np.min(np.abs(cum - r)) < tol:
        ambiguous = True
    i = int(np.searchsorted(cum, r, side="right"))         # first cumulative value > r
    return int(order[min(i, len(order) - 1)]), ambiguous


# ----------------------------------------------------------------------------- process helpers
def run(exe, args, stdin=b""):
    return subprocess.run([exe, *map(str, args)], input=stdin, capture_output=True)


def f32(a) -> bytes:
    return np.ascontiguousarray(a, dtype="<f4").tobytes()


def cpp_uniform(exe, logits, t, k, p, us) -> np.ndarray:
    proc = run(exe, ["uniform", len(logits), repr(t), k, repr(p), len(us)], f32(logits) + f32(us))
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode(errors="replace"))
    return np.frombuffer(proc.stdout, dtype="<i4")


def cpp_rng(exe, logits, t, k, p, seed, count) -> np.ndarray:
    proc = run(exe, ["rng", len(logits), repr(t), k, repr(p), seed, count], f32(logits))
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode(errors="replace"))
    return np.frombuffer(proc.stdout, dtype="<i4")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True, help="path to gi_sampler_check")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    exe = args.exe
    rng = np.random.default_rng(args.seed)
    failures: list[str] = []

    # ---- 1. random generator ---------------------------------------------------------------
    for seed in (0, 1, 42, 2**63 + 12345):
        proc = run(exe, ["rngdump", seed, 2000])
        got = np.frombuffer(proc.stdout, dtype="<f4")
        want = py_uniforms(seed, 2000)
        if proc.returncode != 0 or not np.array_equal(got, want):
            failures.append(f"random generator differs from the Python implementation (seed {seed})")
    u = np.frombuffer(run(exe, ["rngdump", 7, 100000]).stdout, dtype="<f4")
    if not (u.min() >= 0.0 and u.max() < 1.0 and abs(u.mean() - 0.5) < 0.01):
        failures.append(f"uniform draws look wrong: min {u.min()} max {u.max()} mean {u.mean()}")
    rng_info = f"rng: exact match for 4 seeds, mean of 100000 draws = {u.mean():.4f}"

    # ---- 2. deterministic sampling vs oracle ------------------------------------------------
    grid = [(0.3, 0, 1.0), (1.0, 0, 1.0), (2.0, 0, 1.0), (1.0, 0, 0.9), (0.8, 0, 0.5),
            (1.0, 0, 0.05), (1.0, 0, 0.01), (1.0, 3, 1.0), (1.5, 50, 1.0), (1.0, 5, 0.9),
            (0.7, 1, 1.0), (1.0, 1000000, 1.0), (1.0, 4, 0.3), (0.01, 0, 1.0)]
    trials = skipped = 0
    for vocab in (2, 5, 100, 32000):
        logits = (2.0 * rng.standard_normal(vocab)).astype(np.float32)
        us = np.concatenate([rng.random(200, dtype=np.float32),
                             np.array([0.0, 0.5, np.float32(1.0 - 2.0**-24)], dtype=np.float32)])
        for t, k, p in grid:
            got = cpp_uniform(exe, logits, t, k, p, us)
            for g, uu in zip(got, us):
                want, ambiguous = oracle_sample(logits, t, k, p, uu)
                if ambiguous:
                    skipped += 1
                    continue
                trials += 1
                if g != want:
                    failures.append(f"V={vocab} T={t} k={k} p={p} u={float(uu):.6f}: C++ {g}, oracle {want}")
                    break

    # special distributions
    flat = np.zeros(10, dtype=np.float32)                   # all ties -> lowest indices first
    for t, k, p, allowed in ((1.0, 3, 1.0, {0, 1, 2}), (1.0, 0, 0.35, {0, 1, 2, 3}), (1.0, 1, 1.0, {0})):
        got = set(cpp_uniform(exe, flat, t, k, p, rng.random(300, dtype=np.float32)).tolist())
        if not got <= allowed or (len(allowed) > 1 and len(got) < len(allowed)):
            failures.append(f"flat logits, k={k} p={p}: sampled {sorted(got)}, expected {sorted(allowed)}")
    peaked = np.zeros(20, dtype=np.float32)
    peaked[13] = 30.0
    got = set(cpp_uniform(exe, peaked, 1.0, 0, 0.9, rng.random(100, dtype=np.float32)).tolist())
    if got != {13}:
        failures.append(f"peaked distribution with top-p 0.9 should only give token 13, got {sorted(got)}")

    # ---- 3. greedy ----------------------------------------------------------------------------
    for logits in ([1, 5, 5, 2], [3, 3, 3], [-7], [0, -1, 9, 9, 9, 4]):
        arr = np.array(logits, dtype=np.float32)
        got = set(cpp_uniform(exe, arr, 0.0, 0, 1.0, rng.random(20, dtype=np.float32)).tolist())
        if got != {int(np.argmax(arr))}:
            failures.append(f"greedy on {logits}: got {sorted(got)}, expected {{{int(np.argmax(arr))}}}")
    got = cpp_uniform(exe, rng.standard_normal(32000).astype(np.float32), 0.0, 5, 0.5, rng.random(5, dtype=np.float32))
    if len(set(got.tolist())) != 1:
        failures.append("greedy result depends on u / top-k / top-p")

    # ---- 4. statistics with the internal generator ---------------------------------------------
    logits = np.array([2.0, 1.0, 0.5, 0.0, -1.0, -3.0], dtype=np.float32)
    n_draws = 30000
    worst_z = 0.0
    for t, k, p in ((1.0, 0, 1.0), (1.0, 3, 1.0), (1.0, 0, 0.8), (0.7, 4, 0.9)):
        draws = cpp_rng(exe, logits, t, k, p, 99, n_draws)
        order, kp, _ = kept_tokens(softmax64(logits, t), k, p)
        expected = np.zeros(len(logits))
        expected[order] = kp / kp.sum()
        counts = np.bincount(draws, minlength=len(logits))
        for tok in range(len(logits)):
            if expected[tok] == 0.0:
                if counts[tok] != 0:
                    failures.append(f"T={t} k={k} p={p}: filtered-out token {tok} was drawn {counts[tok]} times")
            else:
                z = (counts[tok] - n_draws * expected[tok]) / np.sqrt(n_draws * expected[tok] * (1 - expected[tok]) + 1e-12)
                worst_z = max(worst_z, abs(z))
                if abs(z) > 5.0 and expected[tok] < 1.0:
                    failures.append(f"T={t} k={k} p={p}: token {tok} frequency {counts[tok] / n_draws:.4f} "
                                    f"vs expected {expected[tok]:.4f} (z={z:.1f})")

    # ---- 5. reproducibility -----------------------------------------------------------------------
    a = cpp_rng(exe, logits, 1.0, 0, 1.0, 5, 200)
    b = cpp_rng(exe, logits, 1.0, 0, 1.0, 5, 200)
    c = cpp_rng(exe, logits, 1.0, 0, 1.0, 6, 200)
    if not np.array_equal(a, b):
        failures.append("same seed gave different draws")
    if np.array_equal(a, c):
        failures.append("different seeds gave identical draws")

    # ---- 6. invalid configurations -------------------------------------------------------------------
    one = f32([0.0, 1.0, 2.0]) + f32([0.5])
    for bad in (["uniform", 3, -1.0, 0, 1.0, 1], ["uniform", 3, 1.0, -1, 1.0, 1],
                ["uniform", 3, 1.0, 0, 0.0, 1], ["uniform", 3, 1.0, 0, 1.5, 1], ["uniform", 0, 1.0, 0, 1.0, 1]):
        if run(exe, bad, one).returncode == 0:
            failures.append(f"invalid configuration {bad[1:]} was accepted")

    print(rng_info)
    print(f"sampling vs oracle: {trials} draws compared, {skipped} skipped as boundary-ambiguous")
    print(f"statistics: worst |z| = {worst_z:.2f} (limit 5)")
    if failures:
        print("FAIL")
        for f in failures[:15]:
            print("  -", f)
        if len(failures) > 15:
            print(f"  ... and {len(failures) - 15} more")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
