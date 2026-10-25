#!/usr/bin/env python3
"""verify_bench.py - M6 acceptance: gi_bench reports sane, self-consistent numbers.

A benchmark cannot be "right" in the way a kernel can, so this checks what can be checked:
  1. the JSON report is complete and echoes the model configuration from the file header
  2. the memory numbers equal formulas computed independently in Python
       KV cache  = layers * context * kv_dim * 2 (K and V) * 4 bytes
       scratch   = activation buffers + attention scores + logits
       weights/token = every tensor except the unused RoPE tables
  3. timing statistics are ordered (min <= median <= p95 <= max) and throughputs positive
  4. the workload is clamped to the context length
  5. resident memory: peak >= after-load, and for a model of >= 20 MB the peak must cover the
     weights (they are memory-mapped and every weight is touched while decoding)
  6. bad arguments are rejected

Usage:
  python tests/verify_bench.py --exe build/gi_bench --model models/stories15M/stories15M.bin
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from llama2c_io import read_checkpoint, tensor_layout  # noqa: E402

KEYS = ["label", "model", "compiler", "optimized_build", "dim", "n_layers", "n_heads", "n_kv_heads",
        "vocab_size", "context", "prompt_tokens", "decode_tokens", "runs", "warmup",
        "prefill_tokens_per_s", "decode_tokens_per_s", "decode_ms_median", "decode_ms_p95",
        "decode_ms_min", "decode_ms_max", "weight_bytes_per_token", "effective_gb_per_s",
        "effective_gflops", "file_bytes", "kv_cache_bytes", "scratch_bytes",
        "rss_before_load_bytes", "rss_after_load_bytes", "peak_rss_bytes"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True, help="path to gi_bench")
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-tokens", type=int, default=16)
    ap.add_argument("--decode-tokens", type=int, default=24)
    ap.add_argument("--runs", type=int, default=4)
    args = ap.parse_args()

    ck = read_checkpoint(args.model)
    c = ck.config
    failures: list[str] = []

    def bench(*extra, json_path=None):
        cmd = [args.exe, "--model", args.model, *map(str, extra)]
        if json_path:
            cmd += ["--json", json_path]
        return subprocess.run(cmd, capture_output=True, text=True)

    ctx = c.seq_len
    n_prompt = min(args.prompt_tokens, ctx - 1)
    n_decode = min(args.decode_tokens, ctx - n_prompt)

    with tempfile.TemporaryDirectory() as tmp:
        jp = str(Path(tmp) / "bench.json")
        p = bench("--prompt-tokens", args.prompt_tokens, "--decode-tokens", args.decode_tokens,
                  "--runs", args.runs, "--label", 'quote " test', json_path=jp)
        if p.returncode != 0:
            print(f"FAIL: gi_bench exited with {p.returncode}: {p.stderr.strip()}")
            return 1
        try:
            r = json.loads(Path(jp).read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            print(f"FAIL: the JSON report is not valid JSON: {e}")
            return 1

        # ---- 1. completeness and configuration echo -------------------------------------
        missing = [k for k in KEYS if k not in r]
        if missing:
            failures.append(f"JSON keys missing: {missing}")
        for key, want in (("dim", c.dim), ("n_layers", c.n_layers), ("n_heads", c.n_heads),
                          ("n_kv_heads", c.n_kv_heads), ("vocab_size", c.vocab_size),
                          ("context", ctx), ("runs", args.runs), ("prompt_tokens", n_prompt),
                          ("decode_tokens", n_decode), ("file_bytes", ck.file_size)):
            if r.get(key) != want:
                failures.append(f"{key}: report {r.get(key)}, expected {want}")
        if r.get("label") != 'quote " test':
            failures.append(f"label was not preserved/escaped: {r.get('label')!r}")
        if "| quote \" test |" not in p.stdout:
            failures.append("the markdown table row is missing from the output")

        # ---- 2. memory formulas ------------------------------------------------------------
        kv = c.n_layers * ctx * c.kv_dim * 2 * 4
        scratch = (4 * c.dim + 2 * c.hidden_dim + c.n_heads * ctx + c.vocab_size) * 4
        weights = 0
        for name, shape in tensor_layout(c):
            if name.startswith("freq_cis"):
                continue
            n = 1
            for s in shape:
                n *= s
            weights += (c.dim * 4) if (name == "token_embedding" and not c.shared_classifier) else n * 4
        for key, want in (("kv_cache_bytes", kv), ("scratch_bytes", scratch), ("weight_bytes_per_token", weights)):
            if r.get(key) != want:
                failures.append(f"{key}: report {r.get(key)}, formula says {want}")

        # ---- 3. timing statistics ---------------------------------------------------------------
        if not (0 < r["decode_ms_min"] <= r["decode_ms_median"] <= r["decode_ms_p95"] <= r["decode_ms_max"]):
            failures.append("decode latency statistics are not ordered: "
                            f"{r['decode_ms_min']} {r['decode_ms_median']} {r['decode_ms_p95']} {r['decode_ms_max']}")
        for key in ("prefill_tokens_per_s", "decode_tokens_per_s", "effective_gb_per_s", "effective_gflops"):
            if not r[key] > 0:
                failures.append(f"{key} is not positive: {r[key]}")
        implied = 1000.0 / r["decode_ms_median"]
        # The mean throughput and the median latency describe the same runs, so they agree up to
        # jitter. A factor-of-`runs` mistake (e.g. forgetting to divide by the run count) is caught.
        if not (0.4 * implied < r["decode_tokens_per_s"] < 2.5 * implied):
            failures.append(f"decode tokens/s ({r['decode_tokens_per_s']:.0f}) is inconsistent with the "
                            f"median latency ({r['decode_ms_median']} ms)")

        # ---- 5. resident memory ---------------------------------------------------------------------
        peak, after = r["peak_rss_bytes"], r["rss_after_load_bytes"]
        if peak < after:
            failures.append(f"peak RSS ({peak}) is below RSS after loading ({after})")
        if sys.platform.startswith("linux"):
            if peak <= 0:
                failures.append("peak RSS is 0 on Linux")
            if ck.file_size >= 20 * 2**20 and peak < 0.9 * ck.file_size:
                failures.append(f"peak RSS {peak / 2**20:.1f} MiB does not cover the touched weights "
                                f"({ck.file_size / 2**20:.1f} MiB)")

    # ---- 4. clamping -----------------------------------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        jp = str(Path(tmp) / "clamp.json")
        p = bench("--prompt-tokens", 10**6, "--decode-tokens", 10**6, "--runs", 1, "--warmup", 0, json_path=jp)
        if p.returncode != 0:
            failures.append(f"oversized workload failed instead of being clamped: {p.stderr.strip()}")
        else:
            r2 = json.loads(Path(jp).read_text(encoding="utf-8"))
            if r2["prompt_tokens"] != ctx - 1 or r2["decode_tokens"] != 1:
                failures.append(f"clamping: prompt {r2['prompt_tokens']}, decode {r2['decode_tokens']} "
                                f"(expected {ctx - 1} and 1)")

    # ---- 6. bad arguments ---------------------------------------------------------------------------------
    for label, cmd, want in (
        ("--runs 0", [args.exe, "--model", args.model, "--runs", "0"], 2),
        ("no --model", [args.exe], 2),
        ("unknown option", [args.exe, "--model", args.model, "--bogus"], 2),
        ("non-numeric", [args.exe, "--model", args.model, "--runs", "x"], 2),
        ("missing file", [args.exe, "--model", "/nonexistent.bin"], 1),
    ):
        got = subprocess.run(cmd, capture_output=True).returncode
        if got != want:
            failures.append(f"{label}: exit code {got}, expected {want}")

    if failures:
        print("FAIL")
        for f in failures:
            print("  -", f)
        return 1
    print(f"memory formulas match (KV cache {kv / 2**20:.2f} MiB, scratch {scratch / 2**20:.2f} MiB, "
          f"weights/token {weights / 2**20:.2f} MiB)")
    print(f"decode {r['decode_tokens_per_s']:.1f} tok/s, median {r['decode_ms_median']:.3f} ms/token, "
          f"peak RSS {peak / 2**20:.1f} MiB")
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
