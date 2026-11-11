#!/usr/bin/env python3
"""verify_quantize.py - the int8 weight path, checked against independent NumPy code.

1. Quantizer: gi_quant_dump writes the int8 matrices and bfloat16 scales the loader produced; they must equal the
   output of a NumPy implementation of the format (per group of 64 values: scale = smallest bfloat16 >= max|w| / 127,
   q = rint(w / scale)), BIT FOR BIT, on dummy models with every stored dtype, tied and untied classifiers,
   dimensions that are and are not multiples of 64, and edge-case matrices (zero rows and groups, outliers, tiny
   and subnormal values). The reconstruction error of every weight stays within scale / 2.
2. Bad input: a non-finite weight is rejected at load time.
3. Whole model: gi_forward with GI_WEIGHTS=int8 must agree (float32 rounding only) with gi_forward on a float32
   model holding the DEQUANTIZED weights (q * scale from the dump), for 1..4 threads; int8 runs are byte-identical
   across thread counts and instruction sets of the same kind.
4. Memory: owned_bytes of the loader matches the format.

  python tests/verify_quantize.py --build build
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from hf_io import (HF_ROLE, bf16_to_f32, derived_config, read_hf_config, read_safetensors,  # noqa: E402
                   write_safetensors)

GROUP = 64
MAKE = str(ROOT / "tools" / "make_dummy_hf.py")

VARIANTS = {
    "f32 tied, dims not multiples of 64": ["--dim", "36", "--hidden", "70", "--layers", "2", "--heads", "6",
                                           "--kv-heads", "3", "--head-dim", "6", "--vocab", "103"],
    "bf16 untied, 8 heads / 2 kv": ["--dtype", "bf16", "--untied", "--dim", "64", "--hidden", "192", "--layers", "2",
                                    "--heads", "8", "--kv-heads", "2", "--vocab", "131"],
    "f16 untied, group tails": ["--dtype", "f16", "--untied", "--dim", "100", "--hidden", "230", "--layers", "2",
                                "--heads", "5", "--kv-heads", "5", "--head-dim", "20", "--vocab", "77"],
    "bf16 tied, 1 group of 128": ["--dtype", "bf16", "--dim", "128", "--hidden", "256", "--layers", "1",
                                  "--heads", "4", "--kv-heads", "4", "--vocab", "90"],
}


def run(cmd, env=None):
    return subprocess.run(cmd, capture_output=True, env={**os.environ, **(env or {})})


def bf16_round_up(f: np.ndarray) -> np.ndarray:
    """smallest bfloat16 bit pattern whose value is >= f (f >= 0, float32)."""
    u = np.ascontiguousarray(f, dtype=np.float32).view(np.uint32).astype(np.uint64)
    return (((u + 0xFFFF) & 0xFFFF0000) >> 16).astype(np.uint16)


def reference_quantize(w: np.ndarray):
    """w: float32 [rows, cols] -> (q int8 [rows, cols], scale bf16 bits uint16 [rows, groups])"""
    rows, cols = w.shape
    groups = (cols + GROUP - 1) // GROUP
    q = np.zeros((rows, cols), dtype=np.int8)
    sbits = np.zeros((rows, groups), dtype=np.uint16)
    for g in range(groups):
        blk = w[:, g * GROUP:(g + 1) * GROUP]
        amax = np.max(np.abs(blk), axis=1).astype(np.float32)
        bits = bf16_round_up(amax / np.float32(127.0))
        scale = bf16_to_f32(bits)
        ok = (amax > 0) & (scale > 0) & np.isfinite(scale)
        sbits[:, g] = np.where(ok, bits, 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = blk / np.where(ok, scale, np.float32(1.0))[:, None]
        qq = np.clip(np.rint(ratio), -127, 127)
        q[:, g * GROUP:(g + 1) * GROUP] = np.where(ok[:, None], qq, 0).astype(np.int8)
    return q, sbits


def dequantize(q: np.ndarray, sbits: np.ndarray) -> np.ndarray:
    rows, cols = q.shape
    scale = bf16_to_f32(sbits)
    out = np.zeros((rows, cols), dtype=np.float32)
    for g in range(sbits.shape[1]):
        out[:, g * GROUP:(g + 1) * GROUP] = q[:, g * GROUP:(g + 1) * GROUP].astype(np.float32) * scale[:, g:g + 1]
    return out


def matrices(model_dir: Path):
    """(file name used by gi_quant_dump, float32 source matrix, hf tensor name) for every quantized matrix"""
    cfg = read_hf_config(model_dir)
    d = derived_config(cfg)
    st = read_safetensors(model_dir / "model.safetensors")
    out = []
    for l in range(d["n_layers"]):
        for role in ("wq", "wk", "wv", "wo", "w1", "w2", "w3"):
            name = f"model.layers.{l}.{HF_ROLE[role]}.weight"
            out.append((f"layer{l}_{role}", st[name].as_float32(), name))
    cls = "model.embed_tokens.weight" if d["shared_classifier"] else "lm_head.weight"
    out.append(("wcls", st[cls].as_float32(), cls))
    return out


def expected_quantized(skip: str, n_layers: int) -> set[str]:
    """names of the matrices that GI_Q8_SKIP=skip leaves quantized (independent of the C++ parser)"""
    roles = {"wq", "wk", "wv", "wo", "w1", "w2", "w3", "wcls"}
    kept_roles, first, last, kept_at = set(), 0, 0, set()
    for tok in [t for t in skip.split(",") if t]:
        if "@" in tok:
            r, l = tok.split("@")
            kept_at.add(f"layer{int(l)}_{r}")
        elif tok in roles: kept_roles.add(tok)
        elif tok.startswith("first"): first = int(tok[5:])
        elif tok.startswith("last"): last = int(tok[4:])
    out = set()
    for l in range(n_layers):
        if l < first or l >= n_layers - last:
            continue
        for r in ("wq", "wk", "wv", "wo", "w1", "w2", "w3"):
            if r not in kept_roles and f"layer{l}_{r}" not in kept_at:
                out.add(f"layer{l}_{r}")
    if "wcls" not in kept_roles:
        out.add("wcls")
    return out


def dump(exe_dir: Path, model_dir: Path, out: Path, env=None):
    p = run([str(exe_dir / "gi_quant_dump"), str(model_dir), str(out)], env)
    return p


def check_quantizer(exe_dir: Path, model_dir: Path, label: str, failures: list[str]) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        p = dump(exe_dir, model_dir, Path(tmp))
        if p.returncode != 0:
            failures.append(f"{label}: gi_quant_dump failed: {p.stderr.decode(errors='replace').strip()}")
            return {}
        fields = p.stdout.decode().split()      # "owned_bytes native <n> int8 <m>"
        reported = int(fields[4]) - int(fields[2])   # what quantization added
        expect_bytes = 0
        result = {}
        for name, w, _ in matrices(model_dir):
            rows, cols = w.shape
            groups = (cols + GROUP - 1) // GROUP
            q = np.fromfile(Path(tmp) / f"{name}.q", dtype=np.int8).reshape(rows, cols)
            s = np.fromfile(Path(tmp) / f"{name}.s", dtype="<u2").reshape(rows, groups)
            expect_bytes += rows * cols + 2 * rows * groups
            rq, rs = reference_quantize(w)
            if not np.array_equal(s, rs):
                failures.append(f"{label}: {name}: scales differ from the NumPy quantizer in {int(np.sum(s != rs))} places")
            if not np.array_equal(q, rq):
                failures.append(f"{label}: {name}: int8 values differ from the NumPy quantizer in {int(np.sum(q != rq))} places")
            deq = dequantize(q, s)
            bound = np.repeat(bf16_to_f32(s), GROUP, axis=1)[:, :cols] * 0.5 * (1 + 1e-5)
            bad = np.abs(deq - w) > bound + 1e-30
            if np.any(bad & np.repeat(bf16_to_f32(s) > 0, GROUP, axis=1)[:, :cols]):
                failures.append(f"{label}: {name}: reconstruction error above scale / 2")
            result[name] = (q, s)
        if reported != expect_bytes:
            failures.append(f"{label}: owned_bytes {reported}, expected {expect_bytes}")
    return result


def edit_model(src: Path, dst: Path, edit) -> None:
    """copy a dummy model directory and let edit(name, float32 array) -> float32 array change tensors (stored as F32)"""
    shutil.copytree(src, dst)
    st = read_safetensors(src / "model.safetensors")
    tensors = []
    for name, t in st.items():
        arr = edit(name, t.as_float32().copy())
        tensors.append((name, "F32", tuple(arr.shape), np.ascontiguousarray(arr, dtype="<f4").tobytes()))
    write_safetensors(dst / "model.safetensors", tensors)


def forward_logits(exe: Path, model: Path, seq: list[int], env: dict) -> bytes:
    with tempfile.TemporaryDirectory() as tmp:
        p = run([str(exe), str(model), tmp, "--max-seq-len", str(len(seq)), *map(str, seq)], env)
        if p.returncode != 0:
            raise RuntimeError(f"gi_forward failed: {p.stderr.decode(errors='replace').strip()}")
        return (Path(tmp) / "logits.f32").read_bytes()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default="build")
    args = ap.parse_args()
    b = Path(args.build)
    failures: list[str] = []

    with tempfile.TemporaryDirectory() as tmpd:
        tmp = Path(tmpd)
        models = {}
        for label, margs in VARIANTS.items():
            d = tmp / label.replace(" ", "_").replace(",", "").replace("/", "_")
            p = run([sys.executable, MAKE, "--out", str(d), *margs])
            if p.returncode != 0:
                failures.append(f"{label}: make_dummy_hf failed: {p.stderr.decode(errors='replace')[:200]}")
                continue
            models[label] = d

        # 1. quantizer on the plain dummy models
        for label, d in models.items():
            check_quantizer(b, d, label, failures)
            print(f"quantizer: {label} done")

        # 1b. edge cases: zero rows/groups, outliers, tiny and subnormal magnitudes, exact powers of two
        label = "f32 tied, dims not multiples of 64"
        if label in models:
            def edge(name, arr):
                if name.endswith("q_proj.weight") and ".0." in name:
                    arr[0, :] = 0.0                       # zero row
                    arr[1, :GROUP] = 0.0                  # zero first group only
                    arr[2, 3] = 1.0e4                     # one outlier, the others tiny in comparison
                    arr[3, :] = 1.0e-30                   # tiny but normal
                    arr[4, :] = 1.0e-40                   # subnormal float32
                    arr[5, :] = np.where(np.arange(arr.shape[1]) % 2 == 0, 127.0, -127.0)   # exact: scale 1
                    arr[6, :] = -3.5e-3                   # all equal values
                return arr
            ed = tmp / "edge"
            edit_model(models[label], ed, edge)
            res = check_quantizer(b, ed, "edge cases", failures)
            if res:
                q, s = res["layer0_wq"]
                if not (np.all(q[0] == 0) and np.all(s[0] == 0)): failures.append("edge: zero row not zero")
                if not (np.all(q[1, :GROUP] == 0) and s[1, 0] == 0): failures.append("edge: zero group not zero")
                if abs(int(q[5, 0])) != 127 or float(bf16_to_f32(s[5, :1])[0]) != 1.0: failures.append("edge: +-127 row scale")
            print("quantizer: edge cases done")

            # 2. non-finite weights are rejected
            def bad(name, arr):
                if name.endswith("k_proj.weight") and ".1." in name:
                    arr[2, 5] = np.inf
                return arr
            bd = tmp / "inf"
            edit_model(models[label], bd, bad)
            p = dump(b, bd, tmp / "inf_out")
            if p.returncode == 0 or b"non-finite" not in p.stderr:
                failures.append(f"a model with an infinite weight was accepted for int8 (rc={p.returncode})")
            print("non-finite rejection done")

        # 3. whole model against the dequantized float32 model, for several GI_Q8_SKIP settings (matrices that
        #    stay in their original format; those are NOT replaced in the float32 reference)
        seq = [1, 5, 9, 2, 40, 7, 3, 11, 0, 6]
        for label, d in models.items():
            last_layer = derived_config(read_hf_config(d))["n_layers"] - 1
            for skip in ("", "wcls", "first1", "wq,w2,last1", f"w2@{last_layer},wo@0,w1@{last_layer}"):
                genv = {"GI_Q8_SKIP": skip}
                with tempfile.TemporaryDirectory() as dd:
                    p = dump(b, d, Path(dd), genv)
                    if p.returncode != 0:
                        failures.append(f"{label} skip={skip!r}: gi_quant_dump failed: {p.stderr.decode(errors='replace')[:200]}")
                        continue
                    cfg = read_hf_config(d)
                    dq = {}
                    have = {f.stem for f in Path(dd).glob("*.q")}
                    want_set = expected_quantized(skip, derived_config(cfg)["n_layers"])
                    if have != want_set:
                        failures.append(f"{label} skip={skip!r}: quantized matrices differ from the expectation "
                                        f"(extra {sorted(have - want_set)[:3]}, missing {sorted(want_set - have)[:3]})")
                    for name, w, hf_name in matrices(d):
                        rows, cols = w.shape
                        if not (Path(dd) / f"{name}.q").exists():
                            continue                      # kept in its original format
                        q = np.fromfile(Path(dd) / f"{name}.q", dtype=np.int8).reshape(rows, cols)
                        s_ = np.fromfile(Path(dd) / f"{name}.s", dtype="<u2").reshape(rows, -1)
                        dq[name] = dequantize(q, s_)
                    st = read_safetensors(d / "model.safetensors")
                    tensors = []
                    tied = bool(cfg.get("tie_word_embeddings", False))
                    # (a tied classifier was quantized as its own copy: the embedding table itself stays unquantized)
                    names = {hf: key for key, _, hf in matrices(d) if key in dq and not (tied and key == "wcls")}
                    for name, t in st.items():
                        arr = dq[names[name]] if name in names else t.as_float32()
                        tensors.append((name, "F32", tuple(arr.shape), np.ascontiguousarray(arr, dtype="<f4").tobytes()))
                    if tied and "wcls" in dq:
                        tensors.append(("lm_head.weight", "F32", tuple(dq["wcls"].shape),
                                        np.ascontiguousarray(dq["wcls"], dtype="<f4").tobytes()))
                        cfg = dict(cfg, tie_word_embeddings=False)
                    ref_dir = tmp / ("deq_" + d.name + "_" + (skip.replace(",", "_") or "all"))
                    ref_dir.mkdir()
                    write_safetensors(ref_dir / "model.safetensors", tensors)
                    (ref_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")

                    for isa in ("scalar", "avx2", "avx512"):
                        env = {"GI_ISA": isa}
                        try:
                            want = np.frombuffer(forward_logits(b / "gi_forward", ref_dir, seq, env), dtype="<f4")
                        except RuntimeError as e:
                            if "not supported" in str(e):
                                continue
                            failures.append(f"{label} {isa} skip={skip!r}: reference run failed: {e}")
                            continue
                        base = None
                        for t in (1, 2, 3, 4):
                            env_i = {"GI_ISA": isa, "GI_WEIGHTS": "int8", "GI_THREADS": str(t), "GI_PAR_MIN": "1",
                                     "GI_Q8_SKIP": skip}
                            raw = forward_logits(b / "gi_forward", d, seq, env_i)
                            if base is None:
                                base = raw
                                got = np.frombuffer(raw, dtype="<f4")
                                denom = max(1.0, float(np.max(np.abs(want))))
                                e = float(np.max(np.abs(got.astype(np.float64) - want.astype(np.float64)))) / denom
                                if not e < 2e-5:
                                    failures.append(f"{label} {isa} skip={skip!r}: int8 logits differ from the dequantized "
                                                    f"float32 model (relative error {e:.2e})")
                            elif raw != base:
                                failures.append(f"{label} {isa} skip={skip!r}: int8 logits with {t} threads differ from 1 thread")
            print(f"whole model: {label} done")

        # 3b. a malformed skip list is rejected
        if models:
            d0 = next(iter(models.values()))
            n0 = derived_config(read_hf_config(d0))["n_layers"]
            for bad_skip in ("bogus", "first", "wq,,xx", "w2@", "wcls@1", "w2@99", "w2@-1", "w9@1", f"w2@{n0}"):
                p = dump(b, d0, tmp / "bad_skip_out", {"GI_Q8_SKIP": bad_skip})
                if p.returncode == 0 or b"GI_Q8_SKIP" not in p.stderr:
                    failures.append(f"GI_Q8_SKIP={bad_skip!r} was accepted")

    if failures:
        print("FAIL")
        for f in failures:
            print("  -", f)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
