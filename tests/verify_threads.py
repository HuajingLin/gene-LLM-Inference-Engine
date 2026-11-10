#!/usr/bin/env python3
"""verify_threads.py - results must not depend on the number of threads.

1. gi_pool_check: stress test of the thread pool (every index runs once, completion + visibility, sleep/wake).
2. gi_forward: logits and all traced activations are BIT-IDENTICAL for 1..N threads, on float32 and bfloat16
   dummy models whose row counts are not multiples of 4 (so slices end mid-block), with GI_PAR_MIN=1 so that even
   tiny products are split, and with the default threshold (which keeps small products on one thread).
3. gi_generate: the generated text is byte-identical for different thread counts and under every ISA.
4. Bad arguments are rejected.

  python tests/verify_threads.py --build build
"""
import argparse, os, subprocess, sys, tempfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def run(cmd, env=None):
    return subprocess.run(cmd, capture_output=True, env={**os.environ, **(env or {})})


def forward(exe, model, seq, env):
    with tempfile.TemporaryDirectory() as tmp:
        p = run([exe, model, tmp, "--max-seq-len", str(len(seq)), *map(str, seq)], env)
        if p.returncode != 0:
            raise RuntimeError(f"gi_forward failed: {p.stderr.decode(errors="replace").strip()}")
        out = {f.name: f.read_bytes() for f in Path(tmp).glob("*.f32")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default="build")
    ap.add_argument("--max-threads", type=int, default=7)
    args = ap.parse_args()
    b = Path(args.build)
    failures = []

    # 1. pool stress
    for threads, runs, extra in [(1, 1000, []), (2, 50000, []), (3, 20000, []), (4, 50000, []), (7, 20000, []),
                                 (3, 1500, ["50", "20000"]), (8, 600, ["10", "50000"])]:
        p = run([str(b / "gi_pool_check"), str(threads), str(runs), *extra])
        want = f"ok {threads * runs}"
        if p.returncode != 0 or p.stdout.decode(errors="replace").strip() != want:
            failures.append(f"pool_check {threads} {runs} {extra}: rc={p.returncode} out={p.stdout.decode(errors="replace").strip()!r} err={p.stderr.decode(errors="replace").strip()[:200]}")
    print("pool stress done")

    # 2. bit-identity of the forward pass
    models = [  # (name, dtype, args): rows 13, 22, 37 ... are not multiples of 4
        ("f32_gqa", "f32", ["--dim", "36", "--hidden", "70", "--layers", "3", "--heads", "6", "--kv-heads", "3",
                            "--head-dim", "6", "--vocab", "103"]),
        ("bf16_odd", "bf16", ["--dim", "50", "--hidden", "90", "--layers", "2", "--heads", "5", "--kv-heads", "5",
                              "--head-dim", "10", "--vocab", "77", "--untied"]),
        ("bf16_big", "bf16", ["--dim", "96", "--hidden", "203", "--layers", "2", "--heads", "4", "--kv-heads", "2",
                              "--head-dim", "24", "--vocab", "259"]),
    ]
    seq = [1, 5, 9, 2, 40, 7, 3, 11, 0, 6]
    with tempfile.TemporaryDirectory() as tmp:
        for name, dtype, margs in models:
            d = Path(tmp) / name
            p = run([sys.executable, str(ROOT / "tools/make_dummy_hf.py"), "--out", str(d), "--dtype", dtype, *margs])
            if p.returncode != 0:
                failures.append(f"{name}: make_dummy_hf failed: {p.stderr.decode(errors="replace").strip()[:200]}")
                continue
            for isa in ("scalar", "avx2", "avx512"):
                base = None
                for par_min in ("1", None):
                    for t in range(1, args.max_threads + 1):
                        env = {"GI_ISA": isa, "GI_THREADS": str(t)}
                        if par_min: env["GI_PAR_MIN"] = par_min
                        try:
                            out = forward(str(b / "gi_forward"), str(d), seq, env)
                        except RuntimeError as e:
                            if "not supported" in str(e):
                                break
                            failures.append(f"{name} {isa} t={t}: {e}")
                            continue
                        if base is None:
                            base = out
                        elif out != base:
                            bad = [k for k in out if out[k] != base.get(k)]
                            failures.append(f"{name} {isa} threads={t} par_min={par_min}: differs from 1 thread in {bad[:3]}")
            print(f"forward identity: {name} done")

        # 3. generation identical across threads and ISAs (greedy and sampled with a fixed seed)
        d = Path(tmp) / "tok"
        p = run([sys.executable, str(ROOT / "tools/make_dummy_hf.py"), "--out", str(d), "--dtype", "bf16", "--dim", "48",
                 "--hidden", "98", "--layers", "2", "--heads", "4", "--kv-heads", "2", "--with-tokenizer", "60",
                 "--untied"])
        if p.returncode != 0:
            failures.append(f"tokenizer dummy failed: {p.stderr.decode(errors="replace").strip()[:200]}")
        else:
            for mode in (["--temperature", "0"], ["--temperature", "0.9", "--seed", "7", "--top-k", "20"]):
                got = {}
                for isa in ("scalar", "avx512"):
                    for t in (1, 2, 3, 5):
                        q = run([str(b / "gi_generate"), "--model", str(d), "--prompt", "hello world", "--steps", "40",
                                 "--quiet", "--threads", str(t), *mode], {"GI_PAR_MIN": "1", "GI_ISA": isa})
                        got[(isa, t)] = (q.returncode, q.stdout)
                if any(v[0] != 0 for v in got.values()):
                    failures.append(f"gi_generate failed: {[(k, v[0]) for k, v in got.items() if v[0]]}")
                else:
                    for isa in ("scalar", "avx512"):
                        if len({got[(isa, t)] for t in (1, 2, 3, 5)}) != 1:
                            failures.append(f"gi_generate output ({isa}) depends on --threads ({mode})")
            print("generate identity done")

    # 4. argument validation
    p = run([str(b / "gi_bench"), "--model", "/nonexistent", "--threads", "x"])
    if p.returncode == 0: failures.append("--threads x accepted")
    p = run([str(b / "gi_bench"), "--model", "/nonexistent", "--threads", "-1"])
    if p.returncode == 0: failures.append("--threads -1 accepted")

    if failures:
        print("FAIL")
        for f in failures: print("  -", f)
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
