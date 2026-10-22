#!/usr/bin/env python3
"""ref.py — NumPy Reference implementation, generating a "standard answer"

Do three things:
  1. Implement llama2.c's forward pass with KV Cache using pure NumPy;
  2. Generate greedily (temperature=0), saving text and token ids;
  3. Save the complete logits at the last prompt position for M4 comparison.

Usage:
  python tools/ref.py --model models/stories15M.bin --tokenizer models/tokenizer.bin \
      --prompt "Once upon a time" --steps 128 --out ref_out
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llama2c_io import Tokenizer, read_checkpoint  # noqa: E402

# Normalization 
def rmsnorm(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    ss = np.float32(np.mean(x * x) + np.float32(1e-5))
    return weight * (x * (np.float32(1.0) / np.sqrt(ss)))

# Convert scores into a probability distribution
def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())  # Subtract the maximum value first to prevent overflow in the exponential function (exp).
    return e / e.sum()

# Rotary Position Embedding
def rope_freqs(head_size: int) -> np.ndarray:
    """Rotation frequency for each pair of adjacent dimensions (2i, 2i+1):1 / 10000^(2i/head_size)。"""
    return 1.0 / (10000.0 ** (np.arange(0, head_size, 2, dtype=np.float64) / head_size))

# Apply positional encoding to vectors
def apply_rope(v: np.ndarray, pos: int, head_size: int) -> np.ndarray:
    """Pairing method in llama2.c: adjacent dimensions (0,1),(2,3)...; Within each head, the frequencies start from the beginning."""
    freqs = np.tile(rope_freqs(head_size), v.shape[0] // head_size)
    angle = pos * freqs
    cos, sin = np.cos(angle).astype(np.float32), np.sin(angle).astype(np.float32)
    pairs = v.reshape(-1, 2)
    out = np.empty_like(pairs)
    out[:, 0] = pairs[:, 0] * cos - pairs[:, 1] * sin
    out[:, 1] = pairs[:, 0] * sin + pairs[:, 1] * cos
    return out.reshape(-1)


class Model:
    def __init__(self, ckpt):
        self.cfg = ckpt.config
        self.w = ckpt.tensors
        c = self.cfg
        self.key_cache = np.zeros((c.n_layers, c.seq_len, c.kv_dim), dtype=np.float32)
        self.value_cache = np.zeros_like(self.key_cache)

    def check_rope_tables(self) -> float:
        """Verify our newly calculated cos/sin formulas against the `freq_cis` table stored in the file, and return the maximum absolute error."""
        c, w = self.cfg, self.w
        angle = np.outer(np.arange(c.seq_len), rope_freqs(c.head_size))
        err_r = np.abs(np.cos(angle) - w["freq_cis_real"]).max()
        err_i = np.abs(np.sin(angle) - w["freq_cis_imag"]).max()
        return float(max(err_r, err_i))

    def forward(self, token: int, pos: int) -> np.ndarray:
        """Process a token at position pos, and return the logits for the next token (float32[vocab])."""
        c, w = self.cfg, self.w
        hs = c.head_size
        x = np.array(w["token_embedding"][token], dtype=np.float32)  # Copy to avoid modifying weights

        for l in range(c.n_layers):
            # ---- attention ----
            xb = rmsnorm(x, w["rms_att"][l])
            q = w["wq"][l] @ xb
            k = w["wk"][l] @ xb
            v = w["wv"][l] @ xb
            q = apply_rope(q, pos, hs)
            k = apply_rope(k, pos, hs)

            self.key_cache[l, pos] = k
            self.value_cache[l, pos] = v

            attn_out = np.empty(c.dim, dtype=np.float32)
            for h in range(c.n_heads):
                kvh = h // c.kv_mul  # GQA：Multiple Q heads share the same set of KV heads.
                qh = q[h * hs:(h + 1) * hs]
                K = self.key_cache[l, :pos + 1, kvh * hs:(kvh + 1) * hs]    # [pos+1, hs]
                V = self.value_cache[l, :pos + 1, kvh * hs:(kvh + 1) * hs]
                scores = (K @ qh) / np.sqrt(np.float32(hs))
                attn_out[h * hs:(h + 1) * hs] = softmax(scores) @ V
            x = x + w["wo"][l] @ attn_out  # Residual

            # ---- FFN (SwiGLU) ----
            xb = rmsnorm(x, w["rms_ffn"][l])
            gate = w["w1"][l] @ xb
            up = w["w3"][l] @ xb
            gate = gate * (np.float32(1.0) / (np.float32(1.0) + np.exp(-gate)))  # SiLU
            x = x + w["w2"][l] @ (gate * up)  # Residual

        x = rmsnorm(x, w["rms_final"])
        return (w["wcls"] @ x).astype(np.float32)


def generate_greedy(model: Model, tok: Tokenizer, prompt: str, steps: int):
    cfg = model.cfg
    prompt_tokens = tok.encode(prompt, bos=True)
    steps = min(steps, cfg.seq_len)

    token, pos = prompt_tokens[0], 0
    generated: list[int] = []
    out = bytearray()
    first_logits = None

    while pos < steps:
        logits = model.forward(token, pos)
        if pos == len(prompt_tokens) - 1:
            first_logits = logits.copy()  # Logits at the final prompt position → determine the first generated token

        if pos < len(prompt_tokens) - 1:
            nxt = prompt_tokens[pos + 1]  # Still feeding the prompt, no sampling
        else:
            nxt = int(np.argmax(logits))  # Greedy
            generated.append(nxt)
        pos += 1

        if nxt in (Tokenizer.BOS, Tokenizer.EOS):  # Sequence ended
            break
        if pos > len(prompt_tokens) - 1:
            out += tok.decode_piece(token, nxt)
        token = nxt

    return prompt_tokens, generated, out.decode("utf-8", errors="replace"), first_logits


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--steps", type=int, default=128, help="Total position count limit (including prompt)")
    ap.add_argument("--out", default="ref_out")
    args = ap.parse_args()

    ckpt = read_checkpoint(args.model)
    cfg = ckpt.config
    tok = Tokenizer(args.tokenizer, cfg.vocab_size)
    model = Model(ckpt)

    print(f"[config] {cfg}")
    print(f"[rope]   Currently calculated cos/sin vs. freq_cis table in the file - Maximum error: {model.check_rope_tables():.2e}")

    prompt_tokens, generated, text, first_logits = generate_greedy(model, tok, args.prompt, args.steps)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    first_logits.astype("<f4").tofile(out_dir / "first_logits.f32")
    top5 = np.argsort(-first_logits)[:5]
    record = {
        "model": str(args.model),
        "prompt": args.prompt,
        "steps": args.steps,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated,
        "text": text,
        "first_logits_file": "first_logits.f32",
        "first_logits_position": len(prompt_tokens) - 1,
        "first_logits_top5": [[int(i), float(first_logits[i])] for i in top5],
    }
    (out_dir / "ref.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[prompt tokens] {prompt_tokens}")
    print(f"[generated]     {len(generated)} tokens")
    print("-" * 60)
    print(args.prompt + text)
    print("-" * 60)
    print(f"saved to {out_dir}/ref.json and {out_dir}/first_logits.f32")


if __name__ == "__main__":
    main()