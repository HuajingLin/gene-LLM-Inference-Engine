#!/usr/bin/env python3
"""verify_generate.py - M5 acceptance for the generation loop (gi_generate).

Checks (using the prompt/steps/tokens/text stored in the golden file from tools/ref.py):
  1. greedy (--temperature 0): prompt ids, generated ids and the streamed text equal the
     NumPy reference exactly - the C++ engine tells the same story as ref.py
  2. top-k 1 and a tiny top-p must also reproduce the greedy tokens
  3. the same seed gives the same text; a different seed gives different text
  3b. sampled generation is re-derived independently: for every generated token, NumPy
      recomputes the logits of the C++ history, a Python xorshift64* supplies the uniform
      number and a NumPy oracle does the temperature/top-p draw; it must pick the same token
  4. --no-echo prints only the generated part; --quiet silences the statistics line
  5. --steps limits the number of positions
  6. error handling: bad files/options, negative temperature, prompt longer than the context

Usage:
  python tests/verify_generate.py --exe build/gi_generate \
      --model models/stories15M/stories15M.bin --tokenizer models/stories15M/tokenizer.bin \
      --golden ref_out/ref.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))                # verify_sampler
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))  # ref, llama2c_io
from llama2c_io import read_checkpoint  # noqa: E402
from ref import Model  # noqa: E402
from verify_sampler import oracle_sample, py_uniforms  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True, help="path to gi_generate")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--golden", required=True, help="ref.json produced by tools/ref.py")
    args = ap.parse_args()

    ref = json.loads(Path(args.golden).read_text(encoding="utf-8"))
    prompt, steps = ref["prompt"], ref["steps"]
    expected_text = (prompt + ref["text"]).encode("utf-8")
    failures: list[str] = []

    def gen(*extra, prompt_text=prompt, with_ids=True):
        """Runs gi_generate; returns (returncode, stdout bytes, stderr text, prompt ids, generated ids)."""
        with tempfile.TemporaryDirectory() as tmp:
            ids_file = Path(tmp) / "ids.txt"
            cmd = [args.exe, "--model", args.model, "--tokenizer", args.tokenizer,
                   "--prompt", prompt_text, *map(str, extra)]
            if with_ids:
                cmd += ["--dump-tokens", str(ids_file)]
            p = subprocess.run(cmd, capture_output=True)
            p_ids, g_ids = [], []
            if with_ids and ids_file.exists():
                lines = ids_file.read_text().split("\n")
                p_ids = [int(x) for x in lines[0].split()]
                g_ids = [int(x) for x in lines[1].split()] if len(lines) > 1 else []
            return p.returncode, p.stdout, p.stderr.decode(errors="replace"), p_ids, g_ids

    # ---- 1. greedy equals the reference ---------------------------------------------------------
    rc, out, err, p_ids, g_ids = gen("--steps", steps, "--temperature", 0)
    if rc != 0:
        print(f"FAIL: gi_generate exited with {rc}: {err.strip()}")
        return 1
    if p_ids != ref["prompt_tokens"]:
        failures.append(f"prompt ids differ: {p_ids} vs {ref['prompt_tokens']}")
    if g_ids != ref["generated_tokens"]:
        first = next((i for i, (a, b) in enumerate(zip(g_ids, ref["generated_tokens"])) if a != b),
                     min(len(g_ids), len(ref["generated_tokens"])))
        failures.append(f"greedy tokens differ from the reference, first difference at generated "
                        f"index {first} (C++ {len(g_ids)} tokens, reference {len(ref['generated_tokens'])})")
    if out.decode("utf-8", errors="replace") != expected_text.decode("utf-8", errors="replace"):
        failures.append("greedy text differs from prompt + reference text")
    greedy_ids = g_ids

    # ---- 2. top-k 1 and tiny top-p are greedy ------------------------------------------------------
    for label, flags in (("top-k 1", ["--top-k", 1, "--top-p", 1.0]),
                         ("top-p 1e-6", ["--top-k", 0, "--top-p", 1e-6])):
        _, _, _, _, ids = gen("--steps", steps, "--temperature", 1.0, "--seed", 3, *flags)
        if ids != greedy_ids:
            failures.append(f"{label} did not reproduce the greedy tokens")

    # ---- 3. seeds ------------------------------------------------------------------------------------
    sample = ["--steps", steps, "--temperature", 1.0, "--top-p", 1.0]
    _, out_a, err_a, _, ids_a = gen(*sample, "--seed", 42)
    _, out_b, _, _, ids_b = gen(*sample, "--seed", 42)
    _, out_c, _, _, ids_c = gen(*sample, "--seed", 43)
    if out_a != out_b or ids_a != ids_b:
        failures.append("the same seed produced different output")
    if len(ids_a) >= 10 and ids_a == ids_c:
        failures.append("different seeds produced identical output")
    if not ids_a or not all(0 <= t < ref.get("vocab_size", 10**9) for t in ids_a):
        failures.append("sampled token ids are empty or out of range")

    # ---- 3b. sampled generation re-derived with NumPy + the Python sampler oracle -------------------------
    temp, top_p, seed = 0.8, 0.9, 42
    _, _, _, p_ids_s, g_ids_s = gen("--steps", steps, "--temperature", temp, "--top-p", top_p, "--seed", seed)
    seq = p_ids_s + g_ids_s
    n_p = len(p_ids_s)
    if len(g_ids_s) == 0:
        failures.append("sampled run generated no tokens")
    else:
        np_model = Model(read_checkpoint(args.model))
        uniforms = py_uniforms(seed, len(g_ids_s))
        mismatches = near_boundary = 0
        for pos, tok in enumerate(seq[:-1]):
            logits = np_model.forward(tok, pos)
            if pos < n_p - 1:
                continue  # prompt position: nothing is sampled here
            k = pos - (n_p - 1)
            want, ambiguous = oracle_sample(logits, temp, 0, top_p, uniforms[k], tol=2e-5)
            if want != g_ids_s[k]:
                if ambiguous:
                    near_boundary += 1  # float32 vs float64 may legitimately disagree here
                else:
                    mismatches += 1
                    failures.append(f"sampled token {k}: C++ {g_ids_s[k]}, oracle {want}")
        print(f"sampled run re-derived independently: {len(g_ids_s) - mismatches - near_boundary}/"
              f"{len(g_ids_s)} tokens identical" + (f", {near_boundary} near a decision boundary" if near_boundary else ""))

    # ---- 4. --no-echo / --quiet ---------------------------------------------------------------------------
    _, out_noecho, _, _, _ = gen(*sample, "--seed", 42, "--no-echo")
    if out_a != prompt.encode("utf-8") + out_noecho:
        failures.append("--no-echo output is not the echoed output minus the prompt")
    if "tok/s" not in err_a or "seed 42" not in err_a:
        failures.append(f"statistics line missing or incomplete: {err_a.strip()!r}")
    _, _, err_q, _, _ = gen(*sample, "--seed", 42, "--quiet")
    if err_q.strip():
        failures.append(f"--quiet still printed to stderr: {err_q.strip()!r}")

    # ---- 5. --steps --------------------------------------------------------------------------------------------
    n_prompt = len(ref["prompt_tokens"])
    limit = n_prompt + 6
    _, _, _, _, ids_lim = gen("--steps", limit, "--temperature", 0)
    if not (len(ids_lim) == limit - n_prompt + 1 or (ids_lim and ids_lim[-1] in (1, 2))):
        failures.append(f"--steps {limit}: expected {limit - n_prompt + 1} generated tokens, got {len(ids_lim)}")
    if ids_lim != greedy_ids[:len(ids_lim)]:
        failures.append("a shorter run is not a prefix of the longer greedy run")

    # ---- 6. error handling ----------------------------------------------------------------------------------------
    def rc_of(*a):
        return subprocess.run([args.exe, *map(str, a)], capture_output=True).returncode

    base = ["--model", args.model, "--tokenizer", args.tokenizer]
    checks = [
        ("missing model file", rc_of("--model", "/nonexistent.bin", "--tokenizer", args.tokenizer), 1),
        ("negative temperature", rc_of(*base, "--temperature", -1), 1),
        ("top-p 0", rc_of(*base, "--top-p", 0), 1),
        ("unknown option", rc_of(*base, "--bogus"), 2),
        ("missing --model", rc_of("--tokenizer", args.tokenizer), 2),
        ("non-numeric --steps", rc_of(*base, "--steps", "abc"), 2),
    ]
    for label, got, want in checks:
        if got != want:
            failures.append(f"{label}: exit code {got}, expected {want}")
    too_long = subprocess.run([args.exe, *base, "--prompt", "a " * 3000, "--quiet"], capture_output=True)
    if too_long.returncode != 1 or b"does not fit" not in too_long.stderr:
        failures.append("a prompt longer than the context was not rejected cleanly")

    if failures:
        print("FAIL")
        for f in failures:
            print("  -", f)
        return 1
    print(f"greedy: {len(greedy_ids)} generated tokens identical to the reference")
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
