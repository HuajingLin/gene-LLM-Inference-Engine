#!/usr/bin/env python3
"""verify_device.py - acceptance for the device backend (gpu/): the forward pass on a GPU, or on its host emulation.

The device kernels are one source (src/gpu/device_kernels.inc) compiled twice: for CUDA and as a host emulation
("emu") that runs the same kernel bodies in a loop. This test drives gi_forward / gi_generate with --device:

  0. operations   gi_device_check: every device operation alone against double-precision references (shapes around
                  the kernels' 32-column and 8-token blocks, sentinels around the outputs, large attention scores),
                  the GpuTransformer API (gaps, rewinding), and a tied classifier stored once on the device
  1. reference    every model variant of verify_forward's suite: logits at every position and the layer-by-layer
                  activations equal the NumPy reference (tools/ref.py) and the CPU model within --tol
  2. order        (emu) the indices of every kernel launch run backwards and shuffled: bit-identical results.
                  On a GPU all indices run at once, so a kernel whose result depends on their order is wrong.
  3. batching     prefill in batches of many sizes (around the kernel's 8-token chunks and the batch limit), on
                  top of an existing cache or not: bit-identical to feeding the tokens one by one
  4. restart      a new sequence after an old one (stale keys and values in the cache): bit-identical
  5. generation   greedy text from gi_generate equals the CPU's
  6. errors       unknown device, int8 weights, a position gap, a token id out of range
  7. (cuda)       the GPU's logits equal the emulation's within --tol-emu (same kernels, other libm)

Usage:
  python tests/verify_device.py --build build                 # host emulation, no GPU needed
  python tests/verify_device.py --build build --device cuda   # on a machine with a GPU (cmake -DGI_CUDA=ON)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "tools"))
from hf_io import read_hf_model  # noqa: E402
from verify_forward import MAKE_HF, SUITE, compare, err, run_cpp, run_numpy  # noqa: E402

BATCH_MODELS = {
    "f32 gqa": ["--dim", "36", "--hidden", "70", "--layers", "3", "--heads", "6", "--kv-heads", "3",
                "--head-dim", "6", "--vocab", "103", "--max-pos", "256"],
    "bf16 untied, odd sizes": ["--dtype", "bf16", "--untied", "--dim", "50", "--hidden", "90", "--layers", "2", "--heads", "5",
                               "--kv-heads", "5", "--head-dim", "10", "--vocab", "77", "--max-pos", "256"],
    "qwen3 bf16 qk-norm": ["--qwen3", "--dtype", "bf16", "--dim", "48", "--hidden", "90", "--layers", "2", "--heads", "4",
                           "--kv-heads", "2", "--head-dim", "20", "--vocab", "91", "--max-pos", "256"],
    "llama3 scaling, interleaved-free": ["--dim", "64", "--heads", "4", "--kv-heads", "1", "--head-dim", "16", "--vocab", "140",
                                         "--rope-theta", "500000", "--rope-scaling", "4,1,4,64", "--max-pos", "256"],
}


def forward(exe, model, seq, extra, env, max_seq_len=None):
    """Runs gi_forward; returns (logits as a flat array, stderr text, exit code)."""
    with tempfile.TemporaryDirectory() as tmp:
        p = subprocess.run([str(exe), str(model), tmp, "--max-seq-len", str(max_seq_len or len(seq)), *extra, *map(str, seq)],
                           capture_output=True, env={**os.environ, **env})
        out = np.fromfile(Path(tmp) / "logits.f32", dtype="<f4") if p.returncode == 0 else np.empty(0, dtype="<f4")
        return out, p.stderr.decode(errors="replace"), p.returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default="build")
    ap.add_argument("--device", default="emu", choices=["emu", "cuda"])
    ap.add_argument("--tol", type=float, default=1e-4, help="device vs NumPy and vs the CPU model")
    ap.add_argument("--tol-emu", type=float, default=1e-5, help="cuda vs the host emulation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--matmul", default="narrow", choices=["narrow", "wide"], help="the device's matmul kernel (GI_DEV_MATMUL)")
    args = ap.parse_args()
    os.environ["GI_DEV_MATMUL"] = args.matmul
    fwd = Path(args.build) / "gi_forward"
    gen = Path(args.build) / "gi_generate"
    check = Path(args.build) / "gi_device_check"
    dev = {"GI_DEVICE": args.device}
    failures: list[str] = []
    counts = {"operations": 0, "reference": 0, "order": 0, "batching": 0, "restart": 0, "generation": 0, "errors": 0, "cuda-vs-emu": 0}

    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)

        # 1, 2, 7: the forward suite
        worst_ref = worst_cpu = worst_emu = 0.0
        for i, (label, flags) in enumerate(SUITE.items()):
            d = tmp / f"suite_{i:02d}"
            subprocess.run([sys.executable, MAKE_HF, "--out", str(d), "--seed", str(i), *flags], check=True, capture_output=True)
            ckpt = read_hf_model(d)
            vocab = ckpt.config.vocab_size
            n = min(ckpt.config.seq_len, 40)
            rng = np.random.default_rng(args.seed + i)
            seq = [int(t) for t in rng.integers(0, vocab, size=n)]
            try:
                d_logits, d_trace, _ = run_cpp(str(fwd), str(d), seq, vocab, dev)
                c_logits, c_trace, _ = run_cpp(str(fwd), str(d), seq, vocab, {"GI_DEVICE": "cpu"})
            except RuntimeError as e:
                failures.append(f"[{label}] {e}")
                continue
            ref_logits, ref_trace = run_numpy(ckpt, seq)
            local: list[str] = []
            worst_ref = max(worst_ref, compare("device vs NumPy", d_logits, d_trace, ref_logits, ref_trace, args.tol, local, verbose=False))
            worst_cpu = max(worst_cpu, compare("device vs CPU", d_logits, d_trace, c_logits, c_trace, args.tol, local, verbose=False))
            if len(d_trace) != len(ref_trace):
                local.append(f"the device traced {len(d_trace)} activations, the reference {len(ref_trace)}")
            counts["reference"] += 1
            if args.device == "emu":
                for order in ("reverse", "shuffle"):
                    o_logits, o_trace, _ = run_cpp(str(fwd), str(d), seq, vocab, {**dev, "GI_EMU_ORDER": order})
                    same = o_logits.tobytes() == d_logits.tobytes() and all(
                        k in o_trace and o_trace[k].tobytes() == d_trace[k].tobytes() for k in d_trace)
                    if not same:
                        local.append(f"kernel indices in {order} order change the result: a kernel depends on the order of its threads")
                    counts["order"] += 1
            else:
                e_logits, e_trace, _ = run_cpp(str(fwd), str(d), seq, vocab, {"GI_DEVICE": "emu"})
                worst_emu = max(worst_emu, compare("cuda vs emu", d_logits, d_trace, e_logits, e_trace, args.tol_emu, local, verbose=False))
                counts["cuda-vs-emu"] += 1
            failures += [f"[{label}] {m}" for m in local]
        print(f"reference: {counts['reference']} model variants, worst error vs NumPy {worst_ref:.2e}, vs the CPU model {worst_cpu:.2e}"
              + (f", vs the emulation {worst_emu:.2e}" if args.device == "cuda" else ""))

        # 3, 4: batching and restart
        for label, margs in BATCH_MODELS.items():
            d = tmp / ("batch_" + "".join(c if c.isalnum() else "_" for c in label))
            subprocess.run([sys.executable, MAKE_HF, "--out", str(d), "--seed", "3", *margs], check=True, capture_output=True)
            vocab = int(margs[margs.index("--vocab") + 1])
            seq = [(i * 7 + 3) % vocab for i in range(150)]
            base, stderr, code = forward(fwd, d, seq, [], dev)
            if code != 0:
                failures.append(f"[{label}] token by token: {stderr.strip()}")
                continue
            base = base.reshape(len(seq), vocab)
            cases = [(p, b, 0) for p in (1, 2, 7, 8, 9, 15, 16, 17, 63, 64, 65, 149, 150) for b in (0,)]
            cases += [(150, b, 0) for b in (1, 3, 8, 9, 23, 200)]
            cases += [(p, b, h) for (p, b, h) in ((40, 0, 5), (150, 9, 1), (70, 64, 69), (33, 8, 16))]
            for prefill, max_batch, head in cases:
                extra = ["--prefill", str(prefill)] + (["--max-batch", str(max_batch)] if max_batch else []) + (
                    ["--head", str(head)] if head else [])
                got, stderr, code = forward(fwd, d, seq, extra, dev)
                counts["batching"] += 1
                if code != 0:
                    failures.append(f"[{label}] {' '.join(extra)}: {stderr.strip()}")
                elif got.tobytes() != base[prefill - 1:].tobytes():
                    g = got.reshape(-1, vocab) if got.size == base[prefill - 1:].size else None
                    e = err(g, base[prefill - 1:]) if g is not None else float("inf")
                    failures.append(f"[{label}] {' '.join(extra)}: logits differ from token by token (error {e:.2e})")
            for extra in ([], ["--prefill", "150"], ["--prefill", "20", "--max-batch", "6"]):
                got, stderr, code = forward(fwd, d, seq, ["--decoy", *extra], dev)
                counts["restart"] += 1
                first = int(extra[1]) - 1 if extra else 0
                if code != 0:
                    failures.append(f"[{label}] --decoy {' '.join(extra)}: {stderr.strip()}")
                elif got.tobytes() != base[first:].tobytes():
                    failures.append(f"[{label}] --decoy {' '.join(extra)}: a restarted sequence sees the old cache")

        # 0: every operation alone (independent double-precision references inside the tool), the model API, and
        # the bytes put on the device: a tied classifier must share the embedding table's copy.
        def weight_bytes(c, elem):
            mats = c.attn_dim * c.dim * 2 + c.kv_dim * c.dim * 2 + c.hidden_dim * c.dim * 3
            vecs = 2 * c.dim + (2 * c.head_size if c.qk_norm else 0)
            tables = c.vocab_size * c.dim * (1 if c.shared_classifier else 2)
            return (mats * c.n_layers + tables) * elem + (vecs * c.n_layers + c.dim) * 4

        for idx, elem in ((1, 2), (2, 4), (0, 4), (19, 2)):   # bf16 tied, f16 untied (expanded to f32), f32 tied, qwen3 bf16
            d = tmp / f"suite_{idx:02d}"
            p = subprocess.run([str(check), "--device", args.device, "--model", str(d)], capture_output=True, text=True)
            counts["operations"] += 1
            if p.returncode != 0 or "OK" not in p.stdout:
                failures.append(f"[operations] gi_device_check on suite model {idx}: " + " | ".join(
                    l for l in (p.stdout + p.stderr).splitlines() if l.startswith(("FAIL", "error")))[:300])
                continue
            want = weight_bytes(read_hf_model(d).config, elem)
            got = int(p.stdout.split("weight_bytes=")[1].split()[0])
            if got != want:
                failures.append(f"[operations] suite model {idx}: {got} weight bytes on the device, expected {want}")
        if args.device == "emu":
            for order in ("reverse", "shuffle"):
                p = subprocess.run([str(check), "--device", "emu"], capture_output=True, text=True, env={**os.environ, "GI_EMU_ORDER": order})
                counts["operations"] += 1
                if p.returncode != 0:
                    failures.append(f"[operations] gi_device_check with kernel indices in {order} order failed")

        # 6: errors
        d = tmp / "suite_01"   # bf16, tied
        vocab = read_hf_model(d).config.vocab_size

        def expect_error(what, extra, env, needle, seq=(1, 2, 3, 4), max_seq_len=None):
            _, stderr, code = forward(fwd, d, list(seq), extra, env, max_seq_len)
            counts["errors"] += 1
            if code != 1 or needle.lower() not in stderr.lower():
                failures.append(f"[errors] {what}: exit code {code}, stderr {stderr.strip()[:120]!r} (expected '{needle}')")

        expect_error("unknown device", ["--device", "tpu"], {}, "unknown device")
        expect_error("unknown device in GI_DEVICE", [], {"GI_DEVICE": "gpu0"}, "unknown device")
        expect_error("int8 weights", [], {**dev, "GI_WEIGHTS": "int8"}, "int8 weights are not supported")
        expect_error("position gap", ["--prefill", "3", "--pos-shift", "1"], dev, "leaves a gap", max_seq_len=8)
        expect_error("position gap after one token", ["--prefill", "3", "--head", "1", "--pos-shift", "1"], dev, "leaves a gap", max_seq_len=8)
        expect_error("batch beyond the context", ["--prefill", "4", "--head", "1", "--pos-shift", "2"], dev, "exceed the context length")
        expect_error("token id too large", [], dev, "out of range", seq=(1, vocab, 2))
        expect_error("token id too large in a batch", ["--prefill", "3"], dev, "out of range", seq=(1, 2, vocab))
        expect_error("negative token id", [], dev, "out of range", seq=(1, -1, 2))
        expect_error("bad matmul kernel name", [], {**dev, "GI_DEV_MATMUL": "huge"}, "GI_DEV_MATMUL")
        expect_error("bad emulation order", [], {"GI_DEVICE": "emu", "GI_EMU_ORDER": "sideways"}, "GI_EMU_ORDER")
        _, stderr, code = forward(fwd, d, [1, 2, 3], ["--device", "cuda"], {})
        if args.device == "emu" and code != 0:   # a build without CUDA must say so
            counts["errors"] += 1
            if "CUDA" not in stderr:
                failures.append(f"[errors] --device cuda without CUDA support: {stderr.strip()[:120]!r}")
        # --device on the command line wins over GI_DEVICE
        a, _, ca = forward(fwd, d, [1, 2, 3], ["--device", "cpu"], {"GI_DEVICE": "nonsense"})
        b, _, cb = forward(fwd, d, [1, 2, 3], [], {"GI_DEVICE": "cpu"})
        counts["errors"] += 1
        if ca != 0 or cb != 0 or a.tobytes() != b.tobytes():
            failures.append("[errors] --device cpu did not override GI_DEVICE")

        # The two matmul kernels are really two: other bits, same values. And the profile reports every operation.
        d = tmp / "suite_14"   # bf16, dim 64
        vocab = read_hf_model(d).config.vocab_size
        both = {mm: forward(fwd, d, [3, 1, 4, 1, 5, 9, 2, 6], [], {**dev, "GI_DEV_MATMUL": mm})[0] for mm in ("narrow", "wide")}
        counts["errors"] += 1
        if both["narrow"].size == 0 or both["narrow"].size != both["wide"].size:
            failures.append("[matmul] a run with GI_DEV_MATMUL failed")
        elif both["narrow"].tobytes() == both["wide"].tobytes():
            failures.append("[matmul] GI_DEV_MATMUL=wide gives the same bits as narrow: the switch does nothing")
        elif not err(both["wide"], both["narrow"]) < args.tol_emu:
            failures.append(f"[matmul] wide and narrow differ by {err(both['wide'], both['narrow']):.2e}")

        # 5: generation
        for i, flags in enumerate((["--dtype", "bf16"], ["--qwen3", "--tokenizer-variant", "qwen2"],
                                   ["--tokenizer-variant", "llama3", "--rope-theta", "500000", "--rope-scaling", "8,1,4,32",
                                    "--dim", "64", "--heads", "4", "--kv-heads", "2", "--head-dim", "16"])):
            d = tmp / f"gen_{i}"
            subprocess.run([sys.executable, MAKE_HF, "--out", str(d), "--seed", str(20 + i), "--with-tokenizer", "220",
                            "--max-pos", "96", *flags], check=True, capture_output=True)
            for prompt in ("the quick brown fox", "In 2024, 1234567 apples"):
                outs = {}
                for device in ("cpu", args.device):
                    p = subprocess.run([str(gen), "--model", str(d), "--prompt", prompt, "--steps", "60", "--temperature", "0",
                                        "--quiet", "--device", device], capture_output=True)
                    outs[device] = (p.returncode, p.stdout)
                    if p.returncode != 0:
                        failures.append(f"[generation] model {i} on {device}: {p.stderr.decode(errors='replace').strip()[:160]}")
                counts["generation"] += 1
                if outs["cpu"][0] == 0 and outs[args.device][0] == 0 and outs["cpu"][1] != outs[args.device][1]:
                    failures.append(f"[generation] model {i}, prompt {prompt!r}: greedy text differs from the CPU's")

        d = tmp / "gen_0"
        gen_cmd = [str(gen), "--model", str(d), "--prompt", "ab", "--steps", "12", "--temperature", "0", "--quiet"]
        for device, needle in ((args.device, "device profile:"), ("cpu", "no profile")):
            p = subprocess.run([*gen_cmd, "--device", device, "--profile"], capture_output=True)
            report = p.stderr.decode(errors="replace")
            counts["errors"] += 1
            if p.returncode != 0 or needle not in report:
                failures.append(f"[profile] --profile on {device}: {report.strip()[:160]!r}")
            elif device != "cpu":
                missing = [n for n in ("rmsnorm", "q/k/v projections", "attention", "ffn gate+up", "ffn down", "classifier", "download")
                           if n not in report]
                if missing or "12 positions" not in report:
                    failures.append(f"[profile] report lacks {missing} or the position count: {report.strip()[:200]!r}")
                plain = subprocess.run([*gen_cmd, "--device", device], capture_output=True)
                if plain.stdout != p.stdout:
                    failures.append("[profile] profiling changes the generated text")

    if failures:
        print("\nFAIL")
        for f in failures[:20]:
            print("  -", f)
        return 1
    print(f"PASS  (device '{args.device}', {args.matmul} matmul: " + ", ".join(f"{v} {k}" for k, v in counts.items() if v) + " checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
