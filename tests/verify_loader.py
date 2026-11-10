#!/usr/bin/env python3
"""verify_loader.py - M1 acceptance: compare the output of the C++ loader (gi_inspect) with NumPy, item by item.

Compared: the config and, for every tensor, byte offset / element count / shape / first 4 values / sum,
plus "with a shared classifier, wcls is the same memory as the embedding".

Usage:
  python tests/verify_loader.py --exe build/gi_inspect --model models/stories15M.bin
"""
from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from llama2c_io import read_checkpoint  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True, help="path to the gi_inspect executable")
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    proc = subprocess.run([args.exe, args.model], capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"FAIL: gi_inspect exited with {proc.returncode}: {proc.stderr.strip()}")
        return 1

    cfg_out: dict[str, str] = {}
    tensors_out: dict[str, list[str]] = {}
    roles_out: dict[tuple[int, str], tuple[str, int]] = {}
    for line in proc.stdout.splitlines():
        parts = line.split("\t")
        if parts[0] == "config":
            cfg_out[parts[1]] = parts[2]
        elif parts[0] == "tensor":
            tensors_out[parts[1]] = parts[2:]  # offset, numel, shape, first4, sum, storage
        elif parts[0] == "role":
            roles_out[(int(parts[1]), parts[2])] = (parts[3], int(parts[4]))  # tensor name, element offset

    ck = read_checkpoint(args.model)
    c = ck.config
    errors: list[str] = []

    def expect(what: str, got, want) -> None:
        if str(got) != str(want):
            errors.append(f"{what}: C++={got}  NumPy={want}")

    for key in ("dim", "hidden_dim", "n_layers", "n_heads", "n_kv_heads", "vocab_size",
                "seq_len", "head_size", "kv_dim"):
        expect(f"config.{key}", cfg_out.get(key), getattr(c, key))
    expect("config.shared_classifier", cfg_out.get("shared_classifier"), int(c.shared_classifier))
    expect("config.file_size", cfg_out.get("file_size"), ck.file_size)
    expect("config.wcls_is_embedding", cfg_out.get("wcls_is_embedding"), int(c.shared_classifier))

    want_names = [n for n in ck.tensors if n != "wcls" or not c.shared_classifier]
    expect("tensor list", list(tensors_out), want_names)

    for name in want_names:
        if name not in tensors_out:
            continue
        offset, numel, shape, first4, total, storage = tensors_out[name]
        expect(f"{name}.storage", storage, "f32")   # llama2.c tensors are float32 and used in place
        arr = ck.tensors[name]
        expect(f"{name}.offset", offset, ck.offsets[name])
        expect(f"{name}.numel", numel, arr.size)
        expect(f"{name}.shape", shape, "[" + ",".join(map(str, arr.shape)) + "]")

        flat = np.asarray(arr).reshape(-1)
        got4 = [np.float32(x) for x in first4.split(",")]  # %.9g lets a float32 round-trip exactly
        want4 = [np.float32(x) for x in flat[:4]]
        if got4 != want4:  # so we require exact equality as float32
            errors.append(f"{name}.first4: C++={got4}  NumPy={want4}")

        want_sum = float(flat.astype(np.float64).sum())
        if not math.isclose(float(total), want_sum, rel_tol=1e-7, abs_tol=1e-6):
            errors.append(f"{name}.sum: C++={total}  NumPy={want_sum!r}")

    # Which file tensor feeds which slot of the weights structure (the layers are stored stacked:
    # layer l starts l * (elements per layer) into the tensor).
    per_layer = {n: int(np.prod(arr.shape[1:])) for n, arr in ck.tensors.items() if n in
                 ("rms_att", "wq", "wk", "wv", "wo", "rms_ffn", "w1", "w2", "w3")}
    want_roles = {(-1, "token_embedding"): ("token_embedding", 0), (-1, "rms_final"): ("rms_final", 0),
                  (-1, "wcls"): ("token_embedding" if c.shared_classifier else "wcls", 0)}
    for layer in range(c.n_layers):
        for member, n_per in per_layer.items():
            want_roles[(layer, member)] = (member, layer * n_per)
    if roles_out != want_roles:
        bad = [k for k in want_roles if roles_out.get(k) != want_roles[k]][:5]
        errors.append(f"role mapping differs, e.g. {[(k, roles_out.get(k), want_roles[k]) for k in bad]}")

    if errors:
        print("FAIL")
        for e in errors:
            print("  -", e)
        return 1

    print(f"PASS  ({len(want_names)} tensors, {ck.file_size / 1e6:.2f} MB, "
          f"dim={c.dim} layers={c.n_layers} heads={c.n_heads}/{c.n_kv_heads} "
          f"vocab={c.vocab_size} shared_cls={c.shared_classifier})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
