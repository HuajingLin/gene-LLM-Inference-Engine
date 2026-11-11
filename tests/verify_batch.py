#!/usr/bin/env python3
"""verify_batch.py - batched prompt processing (Transformer::forward_batch) must give EXACTLY the results of
feeding the same tokens one by one.

For dummy models (float32 / bfloat16 / float16 files, GQA and not, dimensions that are not multiples of 4 or 64)
and every weight format (native, f32-expanded, int8, int8 with matrices kept in bf16), under every instruction
set and for several thread counts and batch sizes, gi_forward is run twice on the same token sequence:
  reference   one token at a time (default mode)
  batched     --prefill P [--head H] [--max-batch B]
and the logits of the last prefill position and of every later position must be bit-identical. The cases cover
P = 1 and P = whole sequence, batches cut into pieces (B = 1, 3, 64), a batch that starts on top of an existing
KV cache (--head), a sequence longer than the default batch size, and bad arguments.

  python tests/verify_batch.py --build build
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MAKE = str(ROOT / "tools" / "make_dummy_hf.py")

MODELS = {
    "f32 gqa": ["--dim", "36", "--hidden", "70", "--layers", "3", "--heads", "6", "--kv-heads", "3",
                "--head-dim", "6", "--vocab", "103", "--max-pos", "256"],
    "bf16 odd untied": ["--dtype", "bf16", "--untied", "--dim", "50", "--hidden", "90", "--layers", "2", "--heads", "5",
                        "--kv-heads", "5", "--head-dim", "10", "--vocab", "77", "--max-pos", "256"],
    "f16 big groups": ["--dtype", "f16", "--dim", "96", "--hidden", "203", "--layers", "2", "--heads", "4",
                       "--kv-heads", "2", "--head-dim", "24", "--vocab", "259", "--max-pos", "256"],
}
SEQ = [1, 5, 9, 2, 40, 7, 3, 11, 0, 6, 4, 8, 12, 13, 21, 17]


def run(cmd, env):
    full = {**os.environ, **env}
    return subprocess.run(cmd, capture_output=True, env=full)


def logits(exe, model, seq, extra, env):
    with tempfile.TemporaryDirectory() as tmp:
        p = run([str(exe), str(model), tmp, "--max-seq-len", str(len(seq) + 1), *extra, *map(str, seq)], env)
        if p.returncode != 0:
            raise RuntimeError(p.stderr.decode(errors="replace").strip())
        return np.fromfile(Path(tmp) / "logits.f32", dtype="<f4")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default="build")
    args = ap.parse_args()
    exe = Path(args.build) / "gi_forward"
    failures: list[str] = []
    checked = 0

    with tempfile.TemporaryDirectory() as tmpd:
        tmp = Path(tmpd)
        for label, margs in MODELS.items():
            d = tmp / label.replace(" ", "_")
            p = run([sys.executable, MAKE, "--out", str(d), *margs], {})
            if p.returncode != 0:
                failures.append(f"{label}: make_dummy_hf failed")
                continue
            vocab = int(next(margs[i + 1] for i, a in enumerate(margs) if a == "--vocab"))
            seq = [t % vocab for t in SEQ]
            long_seq = [(i * 7 + 3) % vocab for i in range(150)]      # longer than the default batch of 64
            for wmode, skip in (("native", None), ("f32", None), ("int8", ""), ("int8", "w2@0,wo@1,wcls")):
                for isa in ("scalar", "avx2", "avx512"):
                    env = {"GI_ISA": isa, "GI_WEIGHTS": wmode, "GI_Q8_SKIP": skip if skip is not None else ""}
                    if skip is None:
                        env.pop("GI_Q8_SKIP")
                        env["GI_Q8_SKIP"] = ""
                    try:
                        ref = logits(exe, d, seq, [], env)
                        ref_long = logits(exe, d, long_seq, [], env)
                    except RuntimeError as e:
                        if "not supported" in str(e):
                            continue
                        failures.append(f"{label} {wmode} {isa}: reference run failed: {e}")
                        continue
                    n_vocab = ref.size // len(seq)
                    ref = ref.reshape(len(seq), n_vocab)
                    ref_long = ref_long.reshape(len(long_seq), n_vocab)
                    cases = [  # (prefill, head, max_batch, threads)
                        (1, 0, 0, 1), (5, 0, 0, 1), (len(seq), 0, 0, 1), (len(seq), 0, 1, 1), (len(seq), 0, 3, 1),
                        (len(seq), 0, 3, 3), (9, 0, 4, 2), (9, 4, 0, 1), (9, 4, 2, 3), (len(seq), 5, 64, 2),
                    ]
                    for prefill, head, mb, thr in cases:
                        extra = ["--prefill", str(prefill)] + (["--head", str(head)] if head else []) + (["--max-batch", str(mb)] if mb else [])
                        e2 = dict(env, GI_THREADS=str(thr), GI_PAR_MIN="1")
                        try:
                            got = logits(exe, d, seq, extra, e2).reshape(-1, n_vocab)
                        except RuntimeError as e:
                            failures.append(f"{label} {wmode} {isa} {extra} t={thr}: {e}")
                            continue
                        checked += 1
                        if not np.array_equal(got, ref[prefill - 1:]):
                            bad = int(np.argmax(np.any(got != ref[prefill - 1:], axis=1)))
                            failures.append(f"{label} {wmode} skip={skip!r} {isa} {extra} threads={thr}: differs from "
                                            f"token-by-token at position {prefill - 1 + bad}")
                    for extra, thr in ((["--prefill", "9", "--head", "3", "--traced"], 1), (["--prefill", "9", "--traced"], 2),
                                       (["--prefill", "150"], 1), (["--prefill", "150", "--head", "70"], 2),
                                       (["--prefill", "100", "--max-batch", "37"], 3)):
                        e2 = dict(env, GI_THREADS=str(thr), GI_PAR_MIN="1")
                        prefill = int(extra[1])
                        got = logits(exe, d, long_seq, extra, e2).reshape(-1, n_vocab)
                        checked += 1
                        if not np.array_equal(got, ref_long[prefill - 1:]):
                            failures.append(f"{label} {wmode} {isa} long {extra} threads={thr}: differs from token-by-token")
            print(f"batch identity: {label} done")

        # bad arguments
        d = tmp / "f32_gqa"
        for extra, what in ((["--prefill", "99"], "--prefill larger than the sequence"),
                            (["--prefill", "4", "--head", "4"], "--head not smaller than --prefill"),
                            (["--prefill", "2", "--max-batch", "0", "--head", "-1"], "negative --head")):
            p = run([str(exe), str(d), str(tmp / "bad"), *extra, "1", "2", "3", "4"], {})
            if p.returncode == 0:
                failures.append(f"accepted: {what}")
        # the batch must fit into the context exactly, and not one position more
        okp = run([str(exe), str(d), str(tmp / "fit"), "--max-seq-len", "6", "--prefill", "6", "1", "2", "3", "4", "5", "6"], {})
        if okp.returncode != 0:
            failures.append("a batch that exactly fills the context was rejected")
        for shift in (1, 2):
            p = run([str(exe), str(d), str(tmp / "over"), "--max-seq-len", "6", "--prefill", "6", "--pos-shift", str(shift),
                     "1", "2", "3", "4", "5", "6"], {})
            if p.returncode != 1 or b"exceed the context length" not in p.stderr:   # a clean error, not a crash
                failures.append(f"a batch running {shift} position(s) past the context was not rejected cleanly")
        p = run([str(exe), str(d), str(tmp / "neg"), "--prefill", "3", "--pos-shift", "-1", "1", "2", "3"], {})
        if p.returncode != 1 or b"exceed the context length" not in p.stderr:
            failures.append("a negative start position was not rejected cleanly")
        p = run([str(exe), str(d), str(tmp / "bad2"), "--prefill", "3", "1", "2", "9999"], {})
        if p.returncode == 0 or b"out of range" not in p.stderr:
            failures.append("a token id outside the vocabulary was not rejected in a batch")

    if failures:
        print("FAIL")
        for f in failures[:30]:
            print("  -", f)
        return 1
    print(f"PASS  ({checked} batched runs bit-identical to token-by-token)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
