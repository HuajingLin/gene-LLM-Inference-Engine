#!/usr/bin/env python3
"""quant_accuracy.py - how much does int8 weight quantization change the model's output?

For each prompt:
  1. greedy generation with the reference weights (bf16 as stored): token sequence S (prompt + generated);
  2. both weight formats are run over S ("teacher forcing", so the inputs are identical): at every position the
     logits are compared - top-1 agreement, KL(reference || int8) of the next-token distributions, relative RMS
     error of the logits, and the probability the int8 model gives to the reference's top token;
  3. free-running greedy generation with int8 weights: the position where it first differs from S.
Prints one line per prompt and a summary. Needs gi_generate and gi_forward (build directory) and a model that has
a tokenizer (e.g. models/SmolLM2-135M).

  python tests/quant_accuracy.py --build build --model models/SmolLM2-135M [--steps 64] [--weights int8]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

PROMPTS = [
    "Once upon a time, there was a little girl named Lily.",
    "The capital of France is",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "In 1969, Neil Armstrong became the first person to",
    "Photosynthesis is the process by which plants",
    "Q: What is 17 + 25?\nA:",
    "Dear Sir or Madam,\n\nI am writing to",
    "The three laws of thermodynamics state that",
    "Je m'appelle Marie et j'habite",
    "SELECT name, age FROM users WHERE",
]


def run(cmd, env):
    return subprocess.run(cmd, capture_output=True, env={**os.environ, **env})


def generate(build: Path, model: str, prompt: str, steps: int, env: dict):
    with tempfile.TemporaryDirectory() as tmp:
        dump = Path(tmp) / "t.txt"
        p = run([str(build / "gi_generate"), "--model", model, "--prompt", prompt, "--steps", str(steps),
                 "--temperature", "0", "--quiet", "--no-echo", "--dump-tokens", str(dump)], env)
        if p.returncode != 0:
            raise RuntimeError(p.stderr.decode(errors="replace").strip())
        lines = dump.read_text().splitlines()
    prompt_ids = [int(t) for t in lines[0].split()]
    gen_ids = [int(t) for t in lines[1].split()] if len(lines) > 1 else []
    return prompt_ids, gen_ids


def logits_for(build: Path, model: str, ids: list[int], env: dict) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp:
        p = run([str(build / "gi_forward"), model, tmp, "--max-seq-len", str(len(ids)), *map(str, ids)], env)
        if p.returncode != 0:
            raise RuntimeError(p.stderr.decode(errors="replace").strip())
        raw = np.fromfile(Path(tmp) / "logits.f32", dtype="<f4")
    return raw.reshape(len(ids), -1).astype(np.float64)


def log_softmax(x: np.ndarray) -> np.ndarray:
    m = x.max(axis=1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=1, keepdims=True))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default="build")
    ap.add_argument("--model", required=True)
    ap.add_argument("--steps", type=int, default=64, help="positions per prompt, prompt included")
    ap.add_argument("--weights", default="int8", help="weight format under test (GI_WEIGHTS)")
    ap.add_argument("--prompts-file", help="text file with one prompt per line (default: built-in prompts)")
    args = ap.parse_args()
    build = Path(args.build)
    prompts = PROMPTS
    if args.prompts_file:
        prompts = [l.rstrip("\n").replace("\\n", "\n") for l in Path(args.prompts_file).read_text().splitlines() if l.strip()]

    ref_env = {"GI_WEIGHTS": "native"}
    test_env = {"GI_WEIGHTS": args.weights}
    rows = []
    print(f"reference: native weights   under test: GI_WEIGHTS={args.weights}   model: {args.model}\n")
    print(f"{'prompt':<34} {'pos':>4} {'top1':>6} {'meanKL':>9} {'maxKL':>9} {'relRMS':>8} {'p(ref top)':>10} {'free-run match':>15}")
    for prompt in prompts:
        p_ids, g_ids = generate(build, args.model, prompt, args.steps, ref_env)
        ids = p_ids + g_ids
        ref = logits_for(build, args.model, ids, ref_env)
        got = logits_for(build, args.model, ids, test_env)
        lp_ref, lp_got = log_softmax(ref), log_softmax(got)
        kl = (np.exp(lp_ref) * (lp_ref - lp_got)).sum(axis=1)
        top1 = float(np.mean(ref.argmax(axis=1) == got.argmax(axis=1)))
        rel = float(np.sqrt(np.mean((ref - got) ** 2)) / np.sqrt(np.mean(ref ** 2)))
        ptop = float(np.mean(np.exp(lp_got[np.arange(len(ids)), ref.argmax(axis=1)])))
        p2, g2 = generate(build, args.model, prompt, args.steps, test_env)
        n = min(len(g_ids), len(g2))
        first = next((i for i in range(n) if g_ids[i] != g2[i]), n)
        free = f"{first}/{len(g_ids)}"
        name = prompt.replace("\n", " ")[:32]
        print(f"{name:<34} {len(ids):>4} {top1:>6.1%} {kl.mean():>9.2e} {kl.max():>9.2e} {rel:>8.2e} {ptop:>10.3f} {free:>15}")
        rows.append((top1, kl.mean(), kl.max(), rel, first, len(g_ids)))
    a = np.array(rows, dtype=np.float64)
    print(f"\nsummary over {len(rows)} prompts: top-1 agreement {a[:, 0].mean():.1%} (min {a[:, 0].min():.1%}), "
          f"mean KL {a[:, 1].mean():.2e}, worst KL {a[:, 2].max():.2e}, relative RMS logit error {a[:, 3].mean():.2e}, "
          f"free-running greedy output identical up to the first difference at {int(a[:, 4].sum())} of {int(a[:, 5].sum())} generated tokens")
    return 0


if __name__ == "__main__":
    sys.exit(main())
