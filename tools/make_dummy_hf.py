#!/usr/bin/env python3
"""make_dummy_hf.py - writes a small random Hugging Face style Llama model directory.

Produces config.json + model.safetensors (random weights: the output is gibberish, but the
layout, naming and dtypes are those of real checkpoints). Used to test loaders offline.

Examples:
  python tools/make_dummy_hf.py --out dummy_hf                       # float32, tied embeddings
  python tools/make_dummy_hf.py --out dummy_hf --dtype bf16 --untied
  python tools/make_dummy_hf.py --out dummy_hf --dim 48 --heads 4 --head-dim 16   # attn_dim != dim
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hf_io import f32_to_bf16, llama_tensor_layout, write_safetensors  # noqa: E402


def make_config(args) -> dict:
    cfg = {
        "architectures": ["LlamaForCausalLM"], "model_type": "llama", "attention_bias": False,
        "mlp_bias": False, "hidden_act": "silu", "hidden_size": args.dim,
        "intermediate_size": args.hidden, "num_hidden_layers": args.layers,
        "num_attention_heads": args.heads, "num_key_value_heads": args.kv_heads,
        "vocab_size": args.vocab, "max_position_embeddings": args.max_pos,
        "rms_norm_eps": args.eps, "rope_theta": args.rope_theta, "rope_scaling": None,
        "rope_interleaved": False, "tie_word_embeddings": not args.untied,
        "torch_dtype": {"f32": "float32", "bf16": "bfloat16", "f16": "float16"}[args.dtype],
        "bos_token_id": 0, "eos_token_id": 0,
    }
    if args.head_dim:
        cfg["head_dim"] = args.head_dim
    if args.minimal_config:  # only the required keys: the loader must apply its defaults
        keep = ["model_type", "hidden_size", "intermediate_size", "num_hidden_layers",
                "num_attention_heads", "vocab_size", "tie_word_embeddings"]
        cfg = {k: cfg[k] for k in keep}
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", choices=["f32", "bf16", "f16"], default="f32")
    ap.add_argument("--untied", action="store_true", help="separate lm_head.weight")
    ap.add_argument("--extras", action="store_true",
                    help="also store tensors the loader must ignore (rotary inv_freq, a tied lm_head)")
    ap.add_argument("--minimal-config", action="store_true", help="omit all optional config keys")
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--kv-heads", type=int, default=2)
    ap.add_argument("--head-dim", type=int, default=0, help="0 = dim / heads")
    ap.add_argument("--vocab", type=int, default=100)
    ap.add_argument("--max-pos", type=int, default=64)
    ap.add_argument("--rope-theta", type=float, default=100000.0)
    ap.add_argument("--eps", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = make_config(args)
    rng = np.random.default_rng(args.seed)
    tensors = []
    for name, shape in llama_tensor_layout(cfg):
        if name.endswith("layernorm.weight") or name == "model.norm.weight":
            arr = 1.0 + 0.1 * rng.standard_normal(shape)
        else:
            arr = 0.2 * rng.standard_normal(shape)
        arr = arr.astype(np.float32)
        if args.dtype == "f32":
            dtype_name, raw = "F32", arr.astype("<f4").tobytes()
        elif args.dtype == "bf16":
            dtype_name, raw = "BF16", f32_to_bf16(arr).astype("<u2").tobytes()
        else:
            dtype_name, raw = "F16", arr.astype("<f2").tobytes()
        tensors.append((name, dtype_name, shape, raw))
    if args.extras:
        d_head = cfg.get("head_dim") or args.dim // args.heads
        tensors.append(("model.layers.0.self_attn.rotary_emb.inv_freq", "F32", (d_head // 2,),
                        np.ones(d_head // 2, dtype="<f4").tobytes()))
        if not args.untied:
            tensors.append(("lm_head.weight", "F32", (args.vocab, args.dim),
                            (0.2 * rng.standard_normal((args.vocab, args.dim))).astype("<f4").tobytes()))
    order = list(range(len(tensors)))
    rng.shuffle(order)   # real files do not store tensors in layer order; neither do we
    tensors = [tensors[i] for i in order]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    write_safetensors(out / "model.safetensors", tensors, metadata={"format": "pt"})
    total = sum(len(t[3]) for t in tensors)
    print(f"wrote {out}/config.json and model.safetensors ({total / 1e6:.2f} MB of {args.dtype} weights, "
          f"{len(tensors)} tensors)")


if __name__ == "__main__":
    main()
