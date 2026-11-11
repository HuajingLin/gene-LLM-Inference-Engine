#!/usr/bin/env python3
"""quant_stats.py - data-free statistics of the int8 quantization error of every weight matrix.

For each layer matrix it prints
  snr      signal-to-noise ratio of the dequantized weights in dB (higher = quantization hurts less)
  peak     max|w| / rms(w): how far the largest weight sticks out
  top      where the largest weight sits (row, column) and its value
and ranks the matrices by SNR. Use it together with `tests/quant_accuracy.py --sweep-layers w2` to see whether
the matrices that are sensitive to quantization can be recognized from the weights alone.

  python tools/quant_stats.py --model models/SmolLM2-135M [--role w2] [--worst 15]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "tests"))
from hf_io import HF_ROLE, derived_config, read_hf_config, read_safetensors  # noqa: E402
from verify_quantize import dequantize, reference_quantize  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--role", help="only this role (wq wk wv wo w1 w2 w3)")
    ap.add_argument("--worst", type=int, default=15, help="how many matrices to list in the ranking")
    args = ap.parse_args()
    model = Path(args.model)
    d = derived_config(read_hf_config(model))
    st = read_safetensors(model / "model.safetensors")
    roles = [args.role] if args.role else ["wq", "wk", "wv", "wo", "w1", "w2", "w3"]
    rows = []
    for l in range(d["n_layers"]):
        for role in roles:
            w = st[f"model.layers.{l}.{HF_ROLE[role]}.weight"].as_float32()
            q, s = reference_quantize(w)
            err = w.astype(np.float64) - dequantize(q, s).astype(np.float64)
            snr = 10 * np.log10(np.sum(w.astype(np.float64) ** 2) / max(np.sum(err ** 2), 1e-300))
            rms = float(np.sqrt(np.mean(w.astype(np.float64) ** 2)))
            idx = np.unravel_index(np.argmax(np.abs(w)), w.shape)
            rows.append((l, role, snr, float(np.max(np.abs(w))) / rms, idx, float(w[idx])))
    print(f"{'layer':>5} {'role':<4} {'snr dB':>7} {'peak':>7}  top weight (row, col) = value")
    for l, role, snr, peak, idx, val in rows:
        print(f"{l:>5} {role:<4} {snr:>7.2f} {peak:>7.1f}  ({idx[0]}, {idx[1]}) = {val:+.3f}")
    print(f"\nlowest SNR first (worst {args.worst}):")
    for l, role, snr, peak, idx, val in sorted(rows, key=lambda r: r[2])[:args.worst]:
        print(f"  layer {l:>2} {role:<4} snr {snr:6.2f} dB  peak {peak:7.1f}  top ({idx[0]}, {idx[1]})")
    print(f"\nlargest peak first (worst {args.worst}):")
    for l, role, snr, peak, idx, val in sorted(rows, key=lambda r: -r[3])[:args.worst]:
        print(f"  layer {l:>2} {role:<4} peak {peak:7.1f}  snr {snr:6.2f} dB  top ({idx[0]}, {idx[1]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
