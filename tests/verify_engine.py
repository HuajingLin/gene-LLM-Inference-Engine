#!/usr/bin/env python3
"""verify_engine.py - the request queue / scheduler (Engine) gives every request exactly the output generate() gives alone.

gi_engine_check draws random requests (prompt, position limit, sampler settings), runs them one by one with
generate() on a roomy model (reference), then all together through an Engine on a model with the pool under test.
Tokens, text and stop reason must be identical, and no KV block may leak. Swept: three dummy models (llama-style,
byte-level tokenizer, Qwen3 with QK-norm), block sizes, pool sizes down to ones that force preemption, running-set
sizes, pass sizes, prefill chunk sizes, and requests submitted all at once or while the engine is already running.
Also: a request that can never fit fails cleanly and does not disturb its neighbour.

  python tests/verify_engine.py --build build
"""
from __future__ import annotations

import argparse
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True)
    ap.add_argument("--seeds", type=int, default=8)
    args = ap.parse_args()
    exe = Path(args.build) / "gi_engine_check"
    maker = ROOT / "tools" / "make_dummy_hf.py"
    rng = random.Random(5)
    failures, runs, preempt_runs, reused, evicting = [], 0, 0, 0, 0
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        models = {}
        for name, extra in (("llama", ["--with-tokenizer", "250", "--seed", "3", "--max-pos", "96"]),
                            ("bytelevel-untied", ["--with-tokenizer", "200", "--tokenizer-variant", "bytelevel", "--untied", "--seed", "4", "--max-pos", "96"]),
                            ("qwen3", ["--qwen3", "--with-tokenizer", "220", "--tokenizer-variant", "qwen2", "--seed", "5", "--max-pos", "96", "--dtype", "bf16"])):
            d = t / name
            r = subprocess.run([sys.executable, str(maker), "--out", str(d), *extra], capture_output=True, text=True)
            if r.returncode != 0:
                print("FAIL cannot create dummy model", name, r.stderr[-300:])
                return 1
            models[name] = d
        ctx = 64
        for name, d in models.items():
            for bs in (1, 4, 16):
                per = (ctx + bs - 1) // bs
                for seed in range(1, args.seeds + 1):
                    for pool in ("roomy", "tight", "very tight"):
                        nb = {"roomy": 8 * per, "tight": max(per + 1, 3 * per // 2), "very tight": per}[pool]
                        cmd = [str(exe), str(d), "--max-seq-len", str(ctx), "--block-size", str(bs), "--kv-blocks", str(nb),
                               "--seed", str(seed), "--requests", str(rng.randint(3, 9)), "--max-running", str(rng.randint(1, 6)),
                               "--step-rows", str(rng.choice([0, 1, 3, 8, 64])), "--chunk", str(rng.choice([0, 1, 5, 16]))]
                        if rng.random() < 0.5:
                            cmd.append("--staggered")
                        if rng.random() < 0.5:
                            cmd.append("--wide-eos")
                        if rng.random() < 0.6:
                            cmd += ["--shared-prefix", str(rng.choice([6, 12, 24, 40]))]
                        if rng.random() < 0.15:
                            cmd.append("--no-prefix-cache")
                        p = subprocess.run(cmd, capture_output=True, text=True)
                        runs += 1
                        if p.returncode != 0 or not p.stdout.rstrip().endswith("PASS"):
                            failures.append(f"{name} bs={bs} pool={pool} seed={seed}: {' '.join(cmd[-14:])}\n      {p.stdout.strip()[-250:]} {p.stderr.strip()[-200:]}")
                        else:
                            if " 0 preemptions" not in p.stdout:
                                preempt_runs += 1
                            m = re.search(r"prefix cache: (\d+) positions reused in \d+ admissions, (\d+) blocks evicted", p.stdout)
                            reused += int(m.group(1)); evicting += 1 if int(m.group(2)) > 0 else 0
            for bs in (1, 4):
                cmd = [str(exe), str(d), "--max-seq-len", str(ctx), "--block-size", str(bs), "--kv-blocks", str(12 // bs + 2),
                       "--seed", "2", "--requests", "2", "--oversize"]
                p = subprocess.run(cmd, capture_output=True, text=True)
                runs += 1
                if p.returncode != 0 or not p.stdout.rstrip().endswith("PASS"):
                    failures.append(f"{name} oversize bs={bs}: {p.stdout.strip()[-250:]} {p.stderr.strip()[-200:]}")
    # the benchmark tool: outputs identical at every level, and with a warm cache the shared prefix is served from it
    bench = Path(args.build) / "gi_serve_bench"
    if bench.exists():
      with tempfile.TemporaryDirectory() as t2:
        bench_model = Path(t2) / "m"
        subprocess.run([sys.executable, str(maker), "--out", str(bench_model), "--seed", "3", "--max-pos", "96"], capture_output=True, check=True)
        for extra, want_cached in ((["--shared-prefix", "40", "--warm-cache"], True), (["--shared-prefix", "40", "--warm-cache", "--no-prefix-cache"], False), ([], False)):
            cmd = [str(bench), "--model", str(bench_model), "--concurrency", "1,3", "--prompt-tokens", "48", "--decode-tokens", "6",
                   "--runs", "1", "--max-seq-len", "64", *extra]
            p = subprocess.run(cmd, capture_output=True, text=True)
            runs += 1
            rows = [l.split() for l in p.stdout.splitlines() if l[:1] in "13" and l[1:2] == " "]
            if p.returncode != 0 or len(rows) != 2 or any("identical" not in " ".join(r) for r in rows):
                failures.append(f"gi_serve_bench {' '.join(extra)}: {p.stdout.strip()[-300:]} {p.stderr.strip()[-200:]}")
            elif (int(rows[1][6]) > 0) != want_cached:
                failures.append(f"gi_serve_bench {' '.join(extra)}: cached positions {rows[1][6]}, expected {'some' if want_cached else 'none'}")
    if reused == 0:
        failures.append("the prefix cache never served a position: the shared-prefix runs do not exercise it")
    if evicting == 0:
        failures.append("the prefix cache never had to evict: the tight pools do not exercise eviction")
    if preempt_runs == 0:
        failures.append("no run ever preempted a request: the tight pools do not exercise preemption")
    if failures:
        print("FAIL")
        for f in failures[:10]:
            print("  -", f)
        return 1
    print(f"PASS  ({runs} runs, {preempt_runs} with preemptions, {evicting} with cache evictions, {reused} cached positions reused: "
          f"every request equals generate() alone, no block leaks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
