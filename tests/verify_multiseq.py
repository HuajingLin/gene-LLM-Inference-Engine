#!/usr/bin/env python3
"""verify_multiseq.py - several sequences on one paged KV pool give the same logits as running each alone.

gi_multiseq runs random token streams alone (reference), then interleaves single steps, small batches and random
rewinds of three sequences on one shared pool; every logits row must be bit-identical. The exhaust mode checks that a
full pool refuses new positions with an exception, leaves lengths unchanged and recovers after sequences are destroyed.
Sweeps block sizes, pool sizes and seeds on several dummy models (llama2.c-shaped SmolLM2-like, Qwen3-like).

  python tests/verify_multiseq.py --build build
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True)
    args = ap.parse_args()
    exe = Path(args.build) / "gi_multiseq"
    maker = ROOT / "tools" / "make_dummy_hf.py"
    failures, runs = [], 0
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        models = []
        for name, extra in (("llama", []), ("qwen3", ["--qwen3"])):
            d = t / name
            r = subprocess.run([sys.executable, str(maker), "--out", str(d), *extra], capture_output=True, text=True)
            if r.returncode != 0:
                print("FAIL cannot create dummy model", name, r.stderr[-300:])
                return 1
            models.append(d)
        ctx = 40
        for d in models:
            for bs in (1, 3, 4, 16, 64):
                per = (ctx + bs - 1) // bs
                for seed in (1, 2, 3):
                    cmd = [str(exe), str(d), "--max-seq-len", str(ctx), "--block-size", str(bs),
                           "--kv-blocks", str(3 * per + 3), "--seed", str(seed)]
                    p = subprocess.run(cmd, capture_output=True, text=True)
                    runs += 1
                    if p.returncode != 0 or not p.stdout.rstrip().endswith("PASS"):
                        failures.append(f"{d.name} bs={bs} seed={seed}: {p.stdout.strip()[-200:]} {p.stderr.strip()[-200:]}")
                for seed in range(1, 6):
                    cmd = [str(exe), str(d), "--max-seq-len", str(ctx), "--block-size", str(bs),
                           "--kv-blocks", str(6 * per + 4), "--seed", str(seed), "--mode", "rows"]
                    p = subprocess.run(cmd, capture_output=True, text=True)
                    runs += 1
                    if p.returncode != 0 or not p.stdout.rstrip().endswith("PASS"):
                        failures.append(f"{d.name} rows bs={bs} seed={seed}: {p.stdout.strip()[-200:]} {p.stderr.strip()[-200:]}")
                for seed in range(1, 5):
                    cmd = [str(exe), str(d), "--max-seq-len", str(ctx), "--block-size", str(bs), "--kv-blocks", str(6 * per + 6),
                           "--seed", str(seed), "--mode", "cow"]
                    p = subprocess.run(cmd, capture_output=True, text=True)
                    runs += 1
                    if p.returncode != 0 or not p.stdout.rstrip().endswith("PASS"):
                        failures.append(f"{d.name} cow bs={bs} seed={seed}: {p.stdout.strip()[-200:]} {p.stderr.strip()[-200:]}")
                for seed in (1, 2):
                    for nb in (3, 5, 9):
                        cmd = [str(exe), str(d), "--max-seq-len", str(ctx), "--block-size", str(bs), "--kv-blocks", str(nb),
                               "--mode", "rows_exhaust", "--seed", str(seed)]
                        p = subprocess.run(cmd, capture_output=True, text=True)
                        runs += 1
                        if p.returncode != 0 or not p.stdout.rstrip().endswith("PASS"):
                            failures.append(f"{d.name} rows_exhaust bs={bs} blocks={nb}: {p.stdout.strip()[-200:]} {p.stderr.strip()[-200:]}")
                for nb in (1, 2, 5):
                    cmd = [str(exe), str(d), "--max-seq-len", str(ctx), "--block-size", str(bs), "--kv-blocks", str(nb),
                           "--mode", "exhaust", "--seed", "7"]
                    if nb * bs > ctx:
                        continue   # the pool must not hold more than the context allows to fill
                    p = subprocess.run(cmd, capture_output=True, text=True)
                    runs += 1
                    if p.returncode != 0 or not p.stdout.rstrip().endswith("PASS"):
                        failures.append(f"{d.name} exhaust bs={bs} blocks={nb}: {p.stdout.strip()[-200:]} {p.stderr.strip()[-200:]}")
    if failures:
        print("FAIL")
        for f in failures[:10]:
            print("  -", f)
        return 1
    print(f"PASS  ({runs} runs: interleaved sequences bit-identical to running alone; full pool refused cleanly and recovered)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
