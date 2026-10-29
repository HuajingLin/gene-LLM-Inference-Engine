#!/usr/bin/env python3
"""verify_hf_loader.py - M7 acceptance for loading Hugging Face model directories (gi_inspect).

For a model directory the C++ loader's report is compared, item by item, with an independent
pure-Python reading of config.json and model.safetensors:
  config   dim, hidden_dim, layers, heads, kv heads, vocab, context, head_dim, attn_dim, kv_dim,
           rope_theta, rms_norm_eps, rope style, shared classifier, file size
  tensors  names and order, file offsets, element counts, shapes, first four values and sum of
           the float32 values (so bf16/f16 conversion is checked too)

Without --model, a set of random dummy models is generated and tested (float32 / bf16 / f16, tied and
untied lm_head, grouped-query attention, attn_dim != dim, unrelated extra tensors, default config
values, shuffled tensor order), followed by a set of corrupted directories that must be rejected
with a clear error. With --model, a real model directory is checked.

Usage:
  python tests/verify_hf_loader.py --exe build/gi_inspect                              # dummy models
  python tests/verify_hf_loader.py --exe build/gi_inspect --model models/SmolLM2-135M  # real model
  (optionally: --forward-exe build/gi_forward to check that unsupported settings are refused)
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from hf_io import (derived_config, llama_tensor_layout, read_hf_config, read_safetensors,  # noqa: E402
                   write_safetensors)

MAKE = str(Path(__file__).resolve().parent.parent / "tools" / "make_dummy_hf.py")

VARIANTS = {
    "f32 tied": [],
    "bf16 tied": ["--dtype", "bf16"],
    "f16 untied": ["--dtype", "f16", "--untied"],
    "bf16 untied, 8 query / 2 kv heads": ["--dtype", "bf16", "--untied", "--dim", "64", "--heads", "8", "--kv-heads", "2"],
    "attn_dim != dim (head_dim 16, dim 48)": ["--dim", "48", "--heads", "4", "--kv-heads", "2", "--head-dim", "16", "--hidden", "96"],
    "extra tensors to ignore": ["--extras"],
    "extra tensors, untied": ["--extras", "--untied", "--dtype", "bf16"],
    "default config values": ["--minimal-config"],
    "single layer, multi-query": ["--layers", "1", "--kv-heads", "1"],
}


def run(exe, *args):
    return subprocess.run([exe, *map(str, args)], capture_output=True, text=True)


def parse_report(text: str):
    cfg, tensors, roles = {}, [], {}
    for line in text.splitlines():
        parts = line.split("\t")
        if parts[0] == "config":
            cfg[parts[1]] = parts[2]
        elif parts[0] == "tensor":
            tensors.append(parts[1:])   # name, offset, numel, shape, first4, sum
        elif parts[0] == "role":
            roles[(int(parts[1]), parts[2])] = (parts[3], int(parts[4]))
    return cfg, tensors, roles


def check_model(exe: str, directory: Path) -> list[str]:
    errors: list[str] = []
    p = run(exe, directory)
    if p.returncode != 0:
        return [f"gi_inspect failed: {p.stderr.strip()}"]
    cfg_out, tensors_out, roles_out = parse_report(p.stdout)

    cfg = read_hf_config(directory)
    want = derived_config(cfg)
    for key in ("dim", "hidden_dim", "n_layers", "n_heads", "n_kv_heads", "vocab_size", "seq_len",
                "head_dim", "attn_dim", "kv_dim", "rope_style"):
        if str(cfg_out.get(key)) != str(want[key]):
            errors.append(f"config.{key}: C++ {cfg_out.get(key)}, expected {want[key]}")
    for key in ("rope_theta", "rms_norm_eps"):
        if np.float32(float(cfg_out[key])) != want[key]:
            errors.append(f"config.{key}: C++ {cfg_out[key]}, expected {want[key]}")
    if cfg_out.get("shared_classifier") != str(int(want["shared_classifier"])):
        errors.append(f"config.shared_classifier: C++ {cfg_out.get('shared_classifier')}")
    if cfg_out.get("wcls_is_embedding") != str(int(want["shared_classifier"])):
        errors.append("wcls_is_embedding does not match tie_word_embeddings")
    st_path = directory / "model.safetensors"
    if cfg_out.get("file_size") != str(st_path.stat().st_size):
        errors.append(f"config.file_size: C++ {cfg_out.get('file_size')}, file has {st_path.stat().st_size}")

    # which Hugging Face tensor feeds which slot of the weights structure
    HF_ROLE = {"rms_att": "input_layernorm", "wq": "self_attn.q_proj", "wk": "self_attn.k_proj",
               "wv": "self_attn.v_proj", "wo": "self_attn.o_proj", "rms_ffn": "post_attention_layernorm",
               "w1": "mlp.gate_proj", "w2": "mlp.down_proj", "w3": "mlp.up_proj"}
    want_roles = {(-1, "token_embedding"): ("model.embed_tokens.weight", 0),
                  (-1, "rms_final"): ("model.norm.weight", 0),
                  (-1, "wcls"): ("model.embed_tokens.weight" if want["shared_classifier"] else "lm_head.weight", 0)}
    for layer in range(want["n_layers"]):
        for member, hf_name in HF_ROLE.items():
            want_roles[(layer, member)] = (f"model.layers.{layer}.{hf_name}.weight", 0)
    if roles_out != want_roles:
        bad = [k for k in want_roles if roles_out.get(k) != want_roles[k]][:4]
        errors.append(f"role mapping differs, e.g. {[(k, roles_out.get(k), want_roles[k]) for k in bad]}")

    st = read_safetensors(st_path)
    layout = llama_tensor_layout(cfg)
    if [t[0] for t in tensors_out] != [n for n, _ in layout]:
        errors.append("tensor names/order differ from the expected Llama layout")
        return errors
    for (name, shape), (_, offset, numel, shape_s, first4, total) in zip(layout, tensors_out):
        t = st[name]
        arr = t.as_float32().reshape(-1)
        if offset != str(t.begin):
            errors.append(f"{name}: offset C++ {offset}, file {t.begin}")
        if numel != str(arr.size):
            errors.append(f"{name}: numel C++ {numel}, expected {arr.size}")
        if shape_s != "[" + ",".join(map(str, shape)) + "]":
            errors.append(f"{name}: shape C++ {shape_s}, expected {shape}")
        got4 = [np.float32(x) for x in first4.split(",")]
        want4 = [np.float32(x) for x in arr[:4]]
        if got4 != want4:
            errors.append(f"{name}: first values C++ {got4}, expected {want4}")
        want_sum = float(arr.astype(np.float64).sum())
        if not math.isclose(float(total), want_sum, rel_tol=1e-7, abs_tol=1e-6):
            errors.append(f"{name}: sum C++ {total}, expected {want_sum!r}")
        if len(errors) > 12:
            errors.append("... (stopping after many errors)")
            break
    return errors


# ----------------------------------------------------------------------------- corrupted inputs
def rewrite_without(src: Path, dst: Path, drop: str = "", dtype_override: dict | None = None):
    st = read_safetensors(src / "model.safetensors")
    tensors = []
    for name, t in st.items():
        if name == drop:
            continue
        dtype = (dtype_override or {}).get(name, t.dtype_name)
        raw = bytes(t.raw.tobytes())
        if dtype == "I64":   # keep the file self-consistent: 8 bytes per element
            raw = np.zeros(int(np.prod(t.shape)), dtype="<i8").tobytes()
        tensors.append((name, dtype, t.shape, raw))
    write_safetensors(dst / "model.safetensors", tensors)


def corrupt_cases(good: Path, tmp: Path):
    """Yields (description, directory, expected substring of the error message)."""
    counter = [0]

    def fresh(name):  # neutral directory names: the path appears in error messages
        counter[0] += 1
        d = tmp / f"case_{counter[0]:02d}"
        shutil.copytree(good, d)
        return d

    def edit_config(d, **changes):
        cfg = json.loads((d / "config.json").read_text())
        cfg.update(changes)
        (d / "config.json").write_text(json.dumps(cfg))

    d = fresh("no_config"); (d / "config.json").unlink(); yield "missing config.json", d, "config.json"
    d = fresh("no_weights"); (d / "model.safetensors").unlink(); yield "missing model.safetensors", d, "model.safetensors"
    d = fresh("sharded"); (d / "model.safetensors").unlink(); (d / "model.safetensors.index.json").write_text("{}")
    yield "sharded checkpoint", d, "sharded"
    d = fresh("bad_json"); (d / "config.json").write_text('{"hidden_size": 32,'); yield "invalid config.json", d, "config.json"
    d = fresh("not_object"); (d / "config.json").write_text("[1,2]"); yield "config.json is not an object", d, "object"
    d = fresh("qwen"); edit_config(d, model_type="qwen3"); yield "unsupported model_type", d, "model_type"
    d = fresh("scaling"); edit_config(d, rope_scaling={"type": "llama3"}); yield "rope_scaling", d, "rope_scaling"
    d = fresh("act"); edit_config(d, hidden_act="gelu"); yield "unsupported activation", d, "hidden_act"
    d = fresh("bias"); edit_config(d, attention_bias=True); yield "attention bias", d, "attention_bias"
    d = fresh("nolayers"); cfg = json.loads((d / "config.json").read_text()); del cfg["num_hidden_layers"]
    (d / "config.json").write_text(json.dumps(cfg)); yield "missing required key", d, "num_hidden_layers"
    d = fresh("zero"); edit_config(d, hidden_size=0); yield "zero hidden_size", d, "hidden_size"
    d = fresh("heads"); edit_config(d, num_key_value_heads=3); yield "heads not divisible by kv heads", d, "divisible"
    d = fresh("shape"); edit_config(d, intermediate_size=128); yield "config disagrees with tensor shapes", d, "shape"
    d = fresh("missing"); rewrite_without(good, d, drop="model.layers.1.mlp.up_proj.weight"); yield "missing tensor", d, "model.layers.1.mlp.up_proj.weight"
    d = fresh("i64"); rewrite_without(good, d, dtype_override={"model.norm.weight": "I64"}); yield "unsupported dtype", d, "dtype"

    blob = (good / "model.safetensors").read_bytes()
    d = fresh("trunc"); (d / "model.safetensors").write_bytes(blob[:len(blob) // 2]); yield "truncated weights", d, "safetensors"
    d = fresh("tiny"); (d / "model.safetensors").write_bytes(blob[:5]); yield "file smaller than the length prefix", d, "safetensors"
    d = fresh("hugehdr"); (d / "model.safetensors").write_bytes(struct.pack("<Q", 2**40) + blob[8:]); yield "absurd header length", d, "header"
    (header_len,) = struct.unpack("<Q", blob[:8])
    d = fresh("badhdr"); (d / "model.safetensors").write_bytes(blob[:8] + b"{" * header_len + blob[8 + header_len:])
    yield "garbage header", d, "header"
    header = json.loads(blob[8:8 + header_len])
    first = next(k for k in header if k != "__metadata__")
    header[first]["data_offsets"][1] += 10**9
    text = json.dumps(header, separators=(",", ":")).encode()
    text += b" " * ((8 - len(text) % 8) % 8)
    d = fresh("offsets"); (d / "model.safetensors").write_bytes(struct.pack("<Q", len(text)) + text + blob[8 + header_len:])
    yield "data_offsets beyond the file", d, "data_offsets"
    header = json.loads(blob[8:8 + header_len])
    header[first]["shape"] = [header[first]["shape"][0] + 1] + header[first]["shape"][1:]
    text = json.dumps(header, separators=(",", ":")).encode()
    text += b" " * ((8 - len(text) % 8) % 8)
    d = fresh("bytes"); (d / "model.safetensors").write_bytes(struct.pack("<Q", len(text)) + text + blob[8 + header_len:])
    yield "byte size does not match shape", d, "match shape"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True, help="path to gi_inspect")
    ap.add_argument("--model", help="a real Hugging Face model directory (skips the dummy models)")
    ap.add_argument("--forward-exe", help="path to gi_forward (checks that unsupported settings are refused)")
    args = ap.parse_args()

    if args.model:
        errors = check_model(args.exe, Path(args.model))
        if errors:
            print("FAIL")
            for e in errors:
                print("  -", e)
            return 1
        cfg = derived_config(read_hf_config(args.model))
        print(f"PASS  ({args.model}: dim={cfg['dim']} layers={cfg['n_layers']} heads={cfg['n_heads']}/"
              f"{cfg['n_kv_heads']} vocab={cfg['vocab_size']} head_dim={cfg['head_dim']} "
              f"theta={cfg['rope_theta']})")
        return 0

    failures: list[str] = []
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        for label, flags in VARIANTS.items():
            d = tmp / ("m_" + str(abs(hash(label))))
            subprocess.run([sys.executable, MAKE, "--out", str(d), *flags], check=True, capture_output=True)
            for e in check_model(args.exe, d):
                failures.append(f"[{label}] {e}")

        good = tmp / "good"
        subprocess.run([sys.executable, MAKE, "--out", str(good), "--dtype", "bf16"], check=True, capture_output=True)
        n_bad = 0
        for desc, d, needle in corrupt_cases(good, tmp / "bad"):
            n_bad += 1
            p = run(args.exe, d)
            if p.returncode != 1:
                failures.append(f"[corrupt: {desc}] exit code {p.returncode}, expected 1")
            elif needle.lower() not in p.stderr.lower():
                failures.append(f"[corrupt: {desc}] error message lacks '{needle}': {p.stderr.strip()[:120]}")
        if run(args.exe, tmp / "nonexistent_dir").returncode != 1:
            failures.append("a nonexistent path was not rejected")
        empty = tmp / "empty"; empty.mkdir()
        if run(args.exe, empty).returncode != 1:
            failures.append("an empty directory was not rejected")

        if args.forward_exe:   # the forward pass must refuse HF settings it does not implement yet
            p = run(args.forward_exe, good, tmp / "out", 1, 2)
            if p.returncode != 1 or "not implemented" not in p.stderr:
                failures.append(f"gi_forward did not refuse an unsupported model cleanly: {p.stderr.strip()[:120]}")

    if failures:
        print("FAIL")
        for f in failures[:20]:
            print("  -", f)
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")
        return 1
    print(f"PASS  ({len(VARIANTS)} model variants checked tensor by tensor, {n_bad + 2} corrupted inputs rejected)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
