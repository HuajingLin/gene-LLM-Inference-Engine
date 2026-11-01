#!/usr/bin/env python3
"""verify_forward.py - acceptance for the C++ forward pass (gi_forward) against reference implementations.

The same token sequence is fed through the C++ engine and through the NumPy reference (tools/ref.py), one
position at a time. Compared: the logits at EVERY position (this exercises the KV cache over the whole
context) and the layer-by-layer activations at the last position (embedding, each attention/FFN block, final
norm) - when something is wrong, the first failing line tells you where. Metric:
max|a - b| / max(1, max|b|), required to be < --tol (default 1e-4).

The model can be a llama2.c file or a Hugging Face model directory. Token sources:
  --golden ref.json      a llama2.c golden file from tools/ref.py; also checks first_logits.f32 and that the
                         C++ model re-derives the whole greedy story
  --hf-golden DIR        a directory from tools/hf_golden.py (Hugging Face transformers / PyTorch);
                         C++ and NumPy are both compared with it (tolerance --tol-hf), and the greedy
                         continuation must be re-derived
  --tokens "1 2 3"  or  --random-tokens N

  --dummy-suite          needs no model: generates random Hugging Face style models (dtypes, tied/untied, GQA,
                         MQA, attn_dim != dim, theta, large eps, ...) and compares C++ with NumPy on each

Usage:
  python tests/verify_forward.py --exe build/gi_forward --model models/stories15M/stories15M.bin --golden ref_out/ref.json
  python tests/verify_forward.py --exe build/gi_forward --model models/SmolLM2-135M --random-tokens 12
  python tests/verify_forward.py --exe build/gi_forward --model models/SmolLM2-135M --hf-golden golden_hf
  python tests/verify_forward.py --exe build/gi_forward --dummy-suite
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tools"))
from hf_io import read_hf_model  # noqa: E402
from llama2c_io import read_checkpoint  # noqa: E402
from ref import Model  # noqa: E402

MAKE_HF = str(HERE.parent / "tools" / "make_dummy_hf.py")

SUITE = {
    "f32, theta 1e5, 4/2 heads": [],
    "bf16 weights, tied": ["--dtype", "bf16"],
    "f16 weights, untied": ["--dtype", "f16", "--untied"],
    "theta 10000 (default value)": ["--rope-theta", "10000"],
    "theta 500000": ["--rope-theta", "500000"],
    "large eps 0.05": ["--eps", "0.05"],
    "multi-head, 4/4": ["--kv-heads", "4"],
    "multi-query, 4/1": ["--kv-heads", "1"],
    "attn_dim != dim (dim 48, 4 heads x 16)": ["--dim", "48", "--heads", "4", "--kv-heads", "2", "--head-dim", "16", "--hidden", "96"],
    "attn_dim > dim, untied": ["--dim", "32", "--heads", "4", "--kv-heads", "2", "--head-dim", "16", "--untied"],
    "head_dim 64, 9/3 heads (SmolLM2 shape)": ["--dim", "72", "--heads", "9", "--kv-heads", "3", "--head-dim", "8", "--hidden", "96"],
    "minimal config (default eps and theta)": ["--minimal-config"],
    "single layer": ["--layers", "1"],
    "extra tensors in the file": ["--extras", "--untied"],
}


def err(got, want) -> float:
    got = np.asarray(got, dtype=np.float64)
    want = np.asarray(want, dtype=np.float64)
    if got.shape != want.shape or not np.all(np.isfinite(got)):
        return float("inf")
    return float(np.max(np.abs(got - want)) / max(1.0, np.max(np.abs(want))))


def load_reference(model_path: str):
    return read_hf_model(model_path) if Path(model_path).is_dir() else read_checkpoint(model_path)


def run_cpp(exe: str, model_path: str, seq: list[int], vocab: int):
    with tempfile.TemporaryDirectory() as tmp:
        p = subprocess.run([exe, model_path, tmp, "--max-seq-len", str(len(seq)), *map(str, seq)],
                           capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(f"gi_forward exited with {p.returncode}: {p.stderr.strip()}")
        logits = np.fromfile(Path(tmp) / "logits.f32", dtype="<f4")
        if logits.size != len(seq) * vocab:
            raise RuntimeError(f"logits.f32 has {logits.size} floats, expected {len(seq) * vocab}")
        trace = {f.stem: np.fromfile(f, dtype="<f4") for f in Path(tmp).glob("*.f32") if f.stem != "logits"}
    return logits.reshape(len(seq), vocab), trace, p.stdout.strip()


def run_numpy(ckpt, seq: list[int]):
    model = Model(ckpt, max_len=len(seq))
    logits = np.empty((len(seq), ckpt.config.vocab_size), dtype=np.float32)
    trace: dict[str, np.ndarray] = {}
    for pos, tok in enumerate(seq):
        logits[pos] = model.forward(tok, pos, trace if pos == len(seq) - 1 else None)
    return logits, trace


def compare(label: str, got_logits, got_trace, want_logits, want_trace, tol: float, failures: list[str],
            verbose: bool, skip_missing_ok: bool = False) -> float:
    """Compares logits at every position and the traced activations; returns the worst error."""
    worst = 0.0
    if verbose:
        print(f"\n{label}: activations at the last position")
    def order(k: str):   # execution order: embedding, layers, final norm
        return (0, k) if k == "embedding" else (2, k) if k == "final_norm" else (1, k)

    for name in sorted(want_trace, key=order):
        if name not in got_trace:
            if not skip_missing_ok:
                failures.append(f"{label}: '{name}' missing on the left side")
            continue
        e = err(got_trace[name], want_trace[name])
        worst = max(worst, e)
        if verbose:
            print(f"  {name:<20} {e:9.2e}")
        if not e < tol:
            failures.append(f"{label}: activation {name}: error {e:.2e} >= {tol:.0e}")
    per_pos = [err(got_logits[i], want_logits[i]) for i in range(len(want_logits))]
    w = int(np.argmax(per_pos))
    worst = max(worst, per_pos[w])
    if verbose:
        print(f"{label}: logits worst error {per_pos[w]:.2e} (position {w}) over {len(per_pos)} positions")
    for i, e in enumerate(per_pos):
        if not e < tol:
            failures.append(f"{label}: logits at position {i}: error {e:.2e} >= {tol:.0e}")
    return worst


def check_greedy(cpp_logits, seq, n_prompt, failures, label="C++"):
    mismatches = near_ties = checked = 0
    for pos in range(n_prompt - 1, len(seq) - 1):   # position pos predicts seq[pos + 1]
        checked += 1
        top, want = int(np.argmax(cpp_logits[pos])), seq[pos + 1]
        if top != want:
            gap = float(cpp_logits[pos][top] - cpp_logits[pos][want])
            if gap < 1e-3:
                near_ties += 1      # a float rounding tie, not a bug
            else:
                mismatches += 1
                failures.append(f"{label}: greedy token at position {pos}: got {top}, expected {want} (logit gap {gap:.3f})")
    print(f"greedy continuation ({label}): {checked - mismatches - near_ties}/{checked} tokens identical"
          + (f", {near_ties} near-ties" if near_ties else ""))


def run_dummy_suite(args) -> int:
    failures: list[str] = []
    rows = []
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        for i, (label, flags) in enumerate(SUITE.items()):
            d = tmp / f"model_{i:02d}"
            subprocess.run([sys.executable, MAKE_HF, "--out", str(d), "--seed", str(i), *flags],
                           check=True, capture_output=True)
            ckpt = read_hf_model(d)
            n = min(ckpt.config.seq_len, 40)
            rng = np.random.default_rng(args.seed + i)
            seq = [int(t) for t in rng.integers(0, ckpt.config.vocab_size, size=n)]
            local: list[str] = []
            try:
                cpp_logits, cpp_trace, _ = run_cpp(args.exe, str(d), seq, ckpt.config.vocab_size)
            except RuntimeError as e:
                failures.append(f"[{label}] {e}")
                continue
            ref_logits, ref_trace = run_numpy(ckpt, seq)
            worst = compare(label, cpp_logits, cpp_trace, ref_logits, ref_trace, args.tol, local, verbose=False)
            failures += [f"[{label}] {m}" for m in local]
            rows.append((label, n, worst))
    print(f"{'model variant':<46} {'positions':>9} {'worst error':>12}")
    for label, n, worst in rows:
        print(f"{label:<46} {n:>9} {worst:>12.2e}")
    if failures:
        print("\nFAIL")
        for f in failures[:15]:
            print("  -", f)
        return 1
    print(f"\nPASS  ({len(rows)} model variants, C++ matches the NumPy reference at every position and layer)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True, help="path to gi_forward")
    ap.add_argument("--model", help="llama2.c .bin file or Hugging Face model directory")
    ap.add_argument("--golden", help="llama2.c ref.json produced by tools/ref.py")
    ap.add_argument("--hf-golden", help="directory produced by tools/hf_golden.py")
    ap.add_argument("--tokens", help='explicit token ids, e.g. "1 9038 2501"')
    ap.add_argument("--random-tokens", type=int, default=0, help="use N random token ids")
    ap.add_argument("--dummy-suite", action="store_true", help="test generated random Hugging Face models")
    ap.add_argument("--tol", type=float, default=1e-4, help="C++ vs NumPy")
    ap.add_argument("--tol-hf", type=float, default=2e-3, help="C++ / NumPy vs Hugging Face")
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    if args.dummy_suite:
        return run_dummy_suite(args)
    if not args.model:
        print("error: give --model (or --dummy-suite)")
        return 2

    ckpt = load_reference(args.model)
    cfg = ckpt.config

    ref = hf = None
    n_prompt = None
    if args.golden:
        ref = json.loads(Path(args.golden).read_text(encoding="utf-8"))
        if ref.get("vocab_size") not in (None, cfg.vocab_size):
            print(f"FAIL: golden file was generated for vocab_size={ref['vocab_size']}, this model has "
                  f"{cfg.vocab_size}; regenerate it with tools/ref.py")
            return 1
        n_prompt = len(ref["prompt_tokens"])
        seq = list(ref["prompt_tokens"]) + list(ref["generated_tokens"])
        while seq and seq[-1] in (1, 2) and len(seq) > n_prompt:   # drop the BOS/EOS terminator
            seq.pop()
    elif args.hf_golden:
        hf_dir = Path(args.hf_golden)
        hf = json.loads((hf_dir / "golden.json").read_text(encoding="utf-8"))
        if hf["vocab_size"] != cfg.vocab_size:
            print(f"FAIL: golden vocab_size {hf['vocab_size']} != model vocab_size {cfg.vocab_size}")
            return 1
        seq, n_prompt = list(hf["tokens"]), hf["n_prompt"]
    elif args.tokens:
        seq = [int(t) for t in args.tokens.split()]
    elif args.random_tokens > 0:
        rng = np.random.default_rng(args.seed)
        seq = [int(t) for t in rng.integers(0, cfg.vocab_size, size=args.random_tokens)]
    else:
        print("error: give --golden, --hf-golden, --tokens, --random-tokens or --dummy-suite")
        return 2
    seq = seq[:cfg.seq_len]
    n = len(seq)
    print(f"sequence: {n} tokens  model: dim={cfg.dim} layers={cfg.n_layers} heads={cfg.n_heads}/{cfg.n_kv_heads} "
          f"head_dim={cfg.head_size} vocab={cfg.vocab_size} rope={cfg.rope_style} theta={cfg.rope_theta:g}")

    try:
        cpp_logits, cpp_trace, cpp_info = run_cpp(args.exe, args.model, seq, cfg.vocab_size)
    except RuntimeError as e:
        print(f"FAIL: {e}")
        return 1
    print("C++:", cpp_info)
    ref_logits, ref_trace = run_numpy(ckpt, seq)

    failures: list[str] = []
    compare("C++ vs NumPy", cpp_logits, cpp_trace, ref_logits, ref_trace, args.tol, failures, verbose=True)

    if ref is not None:   # llama2.c golden file
        first = np.fromfile(Path(args.golden).parent / ref["first_logits_file"], dtype="<f4")
        e = err(cpp_logits[ref["first_logits_position"]], first)
        print(f"first_logits.f32 (position {ref['first_logits_position']}): {e:.2e}")
        if not e < args.tol:
            failures.append(f"C++ logits differ from first_logits.f32: {e:.2e}")
        check_greedy(cpp_logits, seq, n_prompt, failures)

    if hf is not None:    # Hugging Face golden directory
        hf_dir = Path(args.hf_golden)
        hf_logits = np.fromfile(hf_dir / "logits.f32", dtype="<f4")
        if hf_logits.size != n * cfg.vocab_size:
            print(f"FAIL: golden logits have {hf_logits.size} floats, expected {n * cfg.vocab_size}")
            return 1
        hf_logits = hf_logits.reshape(n, cfg.vocab_size)
        hf_trace = {f.stem: np.fromfile(f, dtype="<f4") for f in hf_dir.glob("*.f32") if f.stem != "logits"}
        who = "Hugging Face" if hf["backend"] == "torch" else "golden (numpy backend)"
        print(f"\nreference: {hf['backend']} ({hf.get('transformers', 'no transformers')})")
        compare(f"C++ vs {who}", cpp_logits, cpp_trace, hf_logits, hf_trace, args.tol_hf, failures,
                verbose=True, skip_missing_ok=True)
        compare(f"NumPy vs {who}", ref_logits, ref_trace, hf_logits, hf_trace, args.tol_hf, failures,
                verbose=False, skip_missing_ok=True)
        if hf.get("greedy_continuation") and n > n_prompt:
            check_greedy(cpp_logits, seq, n_prompt, failures)

    if failures:
        print("\nFAIL")
        for f in failures[:15]:
            print("  -", f)
        if len(failures) > 15:
            print(f"  ... and {len(failures) - 15} more")
        return 1
    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
