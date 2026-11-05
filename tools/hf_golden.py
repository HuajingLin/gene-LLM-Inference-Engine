#!/usr/bin/env python3
"""hf_golden.py - "golden" outputs of a Hugging Face Llama model, for tests/verify_forward.py.

Feeds a token sequence (a prompt, optionally extended with greedily generated tokens) through the model and
saves what a correct implementation must reproduce:
  golden.json       token ids, prompt length, text, vocabulary size, who produced it
  logits.f32        float32 [n_tokens, vocab]  logits at every position
  embedding.f32, layerNN_attn_out.f32, layerNN_ffn_out.f32, final_norm.f32
                    float32 activations at the LAST position (same names as the C++ gi_forward dump)

Backends:
  torch  (default)  the real thing: Hugging Face transformers + PyTorch, float32, eager attention.
                    Needs:  pip install torch transformers
  numpy             the NumPy reference of this project (tools/ref.py); no PyTorch needed. It is only
                    as trustworthy as that reference - use the torch backend at least once per model.

Examples:
  python tools/hf_golden.py --model models/SmolLM2-135M --prompt "Once upon a time" --new-tokens 24 --out golden_hf
  python tools/hf_golden.py --model models/SmolLM2-135M --tokens "1 2 3" --backend numpy --out golden_np
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def save_outputs(out: Path, info: dict, logits: np.ndarray, trace: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    np.ascontiguousarray(logits, dtype="<f4").tofile(out / "logits.f32")
    for name, arr in trace.items():
        np.ascontiguousarray(arr, dtype="<f4").tofile(out / f"{name}.f32")
    (out / "golden.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")


def run_numpy(model_dir: str, ids: list[int], new_tokens: int):
    from hf_io import read_hf_model
    from ref import Model

    ckpt = read_hf_model(model_dir)
    n_total = len(ids) + new_tokens
    model = Model(ckpt, max_len=n_total)
    seq, all_logits, trace = list(ids), [], {}
    for pos in range(n_total):
        trace = {} if pos == n_total - 1 else None
        logits = model.forward(seq[pos], pos, trace)
        all_logits.append(logits)
        if pos >= len(ids) - 1 and len(seq) < n_total:
            seq.append(int(np.argmax(logits)))
    return seq, np.stack(all_logits), trace, {"backend": "numpy", "vocab_size": ckpt.config.vocab_size}


def encode_prompt(model_dir: str, auto_tokenizer, prompt: str):
    """Token ids of the prompt. The `tokenizers` library reading tokenizer.json is the reference: some
    transformers versions (5.22.0 with SmolLM2) drop the Digits pre-tokenizer that tokenizer.json declares and
    give other ids for texts with digits. Falls back to AutoTokenizer if `tokenizers` is not importable."""
    try:
        from tokenizers import Tokenizer
        tk = Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))
        return list(tk.encode(prompt, add_special_tokens=False).ids), "tokenizers"
    except ImportError:
        return list(auto_tokenizer(prompt)["input_ids"]), "transformers"


def run_torch(model_dir: str, ids: list[int] | None, prompt: str | None, new_tokens: int):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    try:   # newer transformers call the argument `dtype`; older ones only know `torch_dtype`
        model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32, attn_implementation="eager")
    except (TypeError, ValueError):
        model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32, attn_implementation="eager")
    model = model.float().eval()   # float32 whatever the library's default dtype is
    prompt_ids_from = "given"
    if ids is None:
        ids, prompt_ids_from = encode_prompt(model_dir, tokenizer, prompt)
    seq = list(ids)

    with torch.no_grad():
        for _ in range(new_tokens):   # greedy continuation, recomputed from scratch each step (short)
            next_logits = model(torch.tensor([seq])).logits[0, -1]
            seq.append(int(torch.argmax(next_logits)))

        captured: dict[str, torch.Tensor] = {}
        hooks = []
        inner = model.model
        hooks.append(inner.embed_tokens.register_forward_hook(
            lambda m, i, o: captured.__setitem__("embedding", o[0, -1].clone())))
        for l, layer in enumerate(inner.layers):
            # the input of post_attention_layernorm is the residual stream right after the attention block
            hooks.append(layer.post_attention_layernorm.register_forward_pre_hook(
                lambda m, a, l=l: captured.__setitem__(f"layer{l:02d}_attn_out", a[0][0, -1].clone())))
            hooks.append(layer.register_forward_hook(
                lambda m, i, o, l=l: captured.__setitem__(
                    f"layer{l:02d}_ffn_out", (o[0] if isinstance(o, tuple) else o)[0, -1].clone())))
        hooks.append(inner.norm.register_forward_hook(
            lambda m, i, o: captured.__setitem__("final_norm", o[0, -1].clone())))
        logits = model(torch.tensor([seq])).logits[0]   # [n, vocab]
        for h in hooks:
            h.remove()

    trace = {k: v.float().numpy() for k, v in captured.items()}
    info = {"backend": "torch", "torch": torch.__version__, "transformers": transformers.__version__,
            "vocab_size": int(logits.shape[-1]), "text": tokenizer.decode(seq),
            "generated_text": tokenizer.decode(seq[len(ids):]), "prompt_text": prompt,
            "prompt_ids_from": prompt_ids_from}
    return seq, logits.float().numpy(), trace, info, len(ids)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model", required=True, help="Hugging Face model directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", help="text (torch backend only; needs the model's tokenizer)")
    ap.add_argument("--tokens", help='token ids, e.g. "1 2 3"')
    ap.add_argument("--new-tokens", type=int, default=16, help="greedy tokens appended to the prompt")
    ap.add_argument("--backend", choices=["torch", "numpy"], default="torch")
    args = ap.parse_args()

    ids = [int(t) for t in args.tokens.split()] if args.tokens else None
    if ids is None and not args.prompt:
        ap.error("give --prompt or --tokens")
    if args.backend == "numpy" and ids is None:
        ap.error("the numpy backend has no tokenizer yet: give --tokens")

    if args.backend == "numpy":
        seq, logits, trace, info = run_numpy(args.model, ids, args.new_tokens)
        n_prompt = len(ids)
    else:
        seq, logits, trace, info, n_prompt = run_torch(args.model, ids, args.prompt, args.new_tokens)

    info.update({"model": str(args.model), "tokens": seq, "n_prompt": n_prompt,
                 "greedy_continuation": args.new_tokens > 0})
    save_outputs(Path(args.out), info, logits, trace)
    print(f"[{info['backend']}] {len(seq)} tokens ({n_prompt} prompt + {len(seq) - n_prompt} greedy) -> {args.out}")
    if "text" in info:
        print(info["text"])


if __name__ == "__main__":
    main()
