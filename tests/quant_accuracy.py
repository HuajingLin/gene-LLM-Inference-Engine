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
  python tests/quant_accuracy.py --build build --model models/SmolLM2-135M --sweep

--sweep compares several GI_Q8_SKIP settings (which matrices stay in bf16) on the same prompts and prints one
line per setting: it shows which matrices the model is sensitive to. KL columns exclude position 0 (the first
token has no context and often has an outsized error); "KL@0" is the mean KL at position 0 alone.
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


SWEEP = ["", "wcls", "first1", "last1", "first2,last2", "wq,wk,wv,wo", "w1,w3", "w2", "w1,w2,w3",
         "wcls,first2,last2"]


def sweep(build: Path, args, prompts, ref_env) -> int:
    seqs = []
    for prompt in prompts:
        p_ids, g_ids = generate(build, args.model, prompt, args.steps, ref_env)
        ids = p_ids + g_ids
        seqs.append((ids, logits_for(build, args.model, ids, ref_env)))
    print(f"model: {args.model}; {len(seqs)} prompts, {sum(len(i) for i, _ in seqs)} positions; "
          f"reference = native weights; matrices listed under 'kept' stay in bf16, all others are int8\n")
    print(f"{'kept in bf16':<20} {'MiB/token':>9} {'top1':>7} {'KL rest':>9} {'KL@0':>9} {'relRMS':>8}")
    for skip in SWEEP:
        env = {"GI_WEIGHTS": "int8", "GI_Q8_SKIP": skip}
        tops, kls, kl0s, rels = [], [], [], []
        for ids, ref in seqs:
            got = logits_for(build, args.model, ids, env)
            lp_ref, lp_got = log_softmax(ref), log_softmax(got)
            kl = (np.exp(lp_ref) * (lp_ref - lp_got)).sum(axis=1)
            tops.append(float(np.mean(ref.argmax(axis=1) == got.argmax(axis=1))))
            kls.append(float(kl[1:].mean())); kl0s.append(float(kl[0]))
            rels.append(float(np.sqrt(np.mean((ref - got) ** 2)) / np.sqrt(np.mean(ref ** 2))))
        mib = bytes_per_token(build, args.model, env)
        print(f"{skip or '(none)':<20} {mib:>9} {np.mean(tops):>7.1%} {np.mean(kls):>9.2e} {np.mean(kl0s):>9.2e} {np.mean(rels):>8.2e}")
    return 0


def bytes_per_token(build: Path, model: str, env: dict) -> str:
    p = run([str(build / "gi_bench"), "--model", model, "--runs", "1", "--max-seq-len", "128"], env)
    for line in p.stdout.decode(errors="replace").splitlines():
        if "weights read per token" in line:
            return line.split("per token")[1].split("MiB")[0].strip()
    return "?"


def evaluate(build: Path, seqs, skip: str):
    env = {"GI_WEIGHTS": "int8", "GI_Q8_SKIP": skip, "GI_THREADS": "4"}   # results do not depend on the thread count
    tops, kls, kl0s = [], [], []
    for ids, ref in seqs:
        got = logits_for(build, args_model(), ids, env)
        lp_ref, lp_got = log_softmax(ref), log_softmax(got)
        kl = (np.exp(lp_ref) * (lp_ref - lp_got)).sum(axis=1)
        tops.append(float(np.mean(ref.argmax(axis=1) == got.argmax(axis=1))))
        kls.append(float(kl[1:].mean())); kl0s.append(float(kl[0]))
    return float(np.mean(tops)), float(np.mean(kls)), float(np.mean(kl0s))


_MODEL = [""]


def args_model() -> str:
    return _MODEL[0]


def sweep_layers(build: Path, args, prompts, ref_env) -> int:
    import json
    _MODEL[0] = args.model
    cfg = json.loads((Path(args.model) / "config.json").read_text())
    n_layers = int(cfg["num_hidden_layers"])
    role = args.sweep_layers
    seqs = []
    for prompt in prompts:
        p_ids, g_ids = generate(build, args.model, prompt, args.steps, ref_env)
        ids = p_ids + g_ids
        seqs.append((ids, logits_for(build, args.model, ids, ref_env)))
    print(f"model: {args.model}; {len(seqs)} prompts, {sum(len(i) for i, _ in seqs)} positions; {n_layers} layers; "
          f"keeping {role} in bf16\n")
    base = evaluate(build, seqs, "")
    print(f"all int8 (baseline): top1 {base[0]:.1%}  KL rest {base[1]:.2e}  KL@0 {base[2]:.2e}\n")
    print(f"{'layer':>5} {'top1':>7} {'KL rest':>10} {'KL@0':>10} {'rest gain':>10} {'@0 gain':>9}   ({role} of this layer in bf16)")
    gains = []
    for l in range(n_layers):
        top1, kl, kl0 = evaluate(build, seqs, f"{role}@{l}")
        gain, gain0 = base[1] - kl, base[2] - kl0
        gains.append((gain + gain0 / 65.0, l))
        print(f"{l:>5} {top1:>7.1%} {kl:>10.2e} {kl0:>10.2e} {gain:>10.2e} {gain0:>9.2e}")
    order = [l for _, l in sorted(gains, reverse=True)]
    print(f"\nmost sensitive layers first: {order}\n")
    print(f"{'kept in bf16':<34} {'MiB/token':>9} {'top1':>7} {'KL rest':>10} {'KL@0':>10}")
    for k in sorted({k for k in (1, 2, 3, 4, 6, 8, 12, n_layers) if k <= n_layers}):
        skip = ",".join(f"{role}@{l}" for l in sorted(order[:k]))
        top1, kl, kl0 = evaluate(build, seqs, skip)
        env = {"GI_WEIGHTS": "int8", "GI_Q8_SKIP": skip}
        label = f"{role} of {k} layers"
        print(f"{label:<34} {bytes_per_token(build, args.model, env):>9} {top1:>7.1%} {kl:>10.2e} {kl0:>10.2e}   [{skip if k <= 8 else '...'}]")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default="build")
    ap.add_argument("--model", required=True)
    ap.add_argument("--steps", type=int, default=64, help="positions per prompt, prompt included")
    ap.add_argument("--weights", default="int8", help="weight format under test (GI_WEIGHTS)")
    ap.add_argument("--sweep", action="store_true", help="compare several GI_Q8_SKIP settings")
    ap.add_argument("--sweep-layers", metavar="ROLE", help="keep ROLE (e.g. w2) in bf16 in one layer at a time, then in the "
                    "most sensitive layers together; shows which layers need the extra precision")
    ap.add_argument("--max-prompts", type=int, default=0, help="use only the first N prompts (faster sweeps)")
    ap.add_argument("--prompts-file", help="text file with one prompt per line (default: built-in prompts)")
    args = ap.parse_args()
    build = Path(args.build)
    prompts = PROMPTS
    if args.prompts_file:
        prompts = [l.rstrip("\n").replace("\\n", "\n") for l in Path(args.prompts_file).read_text().splitlines() if l.strip()]

    ref_env = {"GI_WEIGHTS": "native"}
    if args.max_prompts:
        prompts = prompts[:args.max_prompts]
    if args.sweep_layers:
        return sweep_layers(build, args, prompts, ref_env)
    if args.sweep:
        return sweep(build, args, prompts, ref_env)
    test_env = {"GI_WEIGHTS": args.weights}
    rows = []
    print(f"reference: native weights   under test: GI_WEIGHTS={args.weights}   model: {args.model}\n")
    print(f"{'prompt':<34} {'pos':>4} {'top1':>6} {'KL@0':>9} {'KL rest':>9} {'maxKL':>9} {'relRMS':>8} {'p(ref top)':>10} {'free-run match':>15}")
    for prompt in prompts:
        p_ids, g_ids = generate(build, args.model, prompt, args.steps, ref_env)
        ids = p_ids + g_ids
        ref = logits_for(build, args.model, ids, ref_env)
        got = logits_for(build, args.model, ids, test_env)
        lp_ref, lp_got = log_softmax(ref), log_softmax(got)
        kl = (np.exp(lp_ref) * (lp_ref - lp_got)).sum(axis=1)
        kl0, kl_rest = float(kl[0]), float(kl[1:].mean())
        top1 = float(np.mean(ref.argmax(axis=1) == got.argmax(axis=1)))
        rel = float(np.sqrt(np.mean((ref - got) ** 2)) / np.sqrt(np.mean(ref ** 2)))
        ptop = float(np.mean(np.exp(lp_got[np.arange(len(ids)), ref.argmax(axis=1)])))
        p2, g2 = generate(build, args.model, prompt, args.steps, test_env)
        n = min(len(g_ids), len(g2))
        first = next((i for i in range(n) if g_ids[i] != g2[i]), n)
        free = f"{first}/{len(g_ids)}"
        name = prompt.replace("\n", " ")[:32]
        print(f"{name:<34} {len(ids):>4} {top1:>6.1%} {kl0:>9.2e} {kl_rest:>9.2e} {kl[1:].max():>9.2e} {rel:>8.2e} {ptop:>10.3f} {free:>15}")
        rows.append((top1, kl_rest, kl[1:].max(), rel, first, len(g_ids), kl0))
    a = np.array(rows, dtype=np.float64)
    print(f"\nsummary over {len(rows)} prompts: top-1 agreement {a[:, 0].mean():.1%} (min {a[:, 0].min():.1%}), "
          f"mean KL (positions 1..) {a[:, 1].mean():.2e}, KL at position 0 {a[:, 6].mean():.2e}, worst KL {a[:, 2].max():.2e}, relative RMS logit error {a[:, 3].mean():.2e}, "
          f"free-running greedy output identical up to the first difference at {int(a[:, 4].sum())} of {int(a[:, 5].sum())} generated tokens")
    return 0


if __name__ == "__main__":
    sys.exit(main())
