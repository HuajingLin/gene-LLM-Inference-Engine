#!/usr/bin/env python3
"""verify_generate_hf.py - M10 acceptance: text generation from a Hugging Face model directory (gi_generate --model DIR).

Part 1 - dummy models (no downloads; random weights, real file formats). For several model/tokenizer variants
  (float32/bf16/f16, tied/untied, three tokenizer variants, eos as an int or a list, a model vocabulary padded
  beyond the tokenizer's) every run of gi_generate is re-derived independently:
    * prompt ids      from tools/hf_tok_ref.py (the independent Python tokenizer)
    * generated ids   from tools/ref.py (NumPy forward pass) with the same stop rules: only SAMPLED tokens end a
                      generation, prompt tokens never do
    * stdout bytes    prompt text + decoded generated tokens (the end token itself prints nothing)
  Also: eos given as a list, an end token inside the prompt, --bos, the empty prompt (BOS fallback), a sampled
  run re-derived token by token with the Python xorshift oracle, --max-seq-len, --tokenizer DIR, and error cases.
  Hugging Face decoding has no leading-space rule (llama2.c drops the space after BOS): a model whose first token
  starts with a space is searched for and its output must keep that space.

Part 2 - a real model (--model-dir DIR --hf-golden GOLDEN_DIR): greedy output of gi_generate must equal the
  continuation that Hugging Face transformers + PyTorch produced (tools/hf_golden.py --prompt ...).

Usage:
  python tests/verify_generate_hf.py --exe build/gi_generate
  python tests/verify_generate_hf.py --exe build/gi_generate --model-dir models/SmolLM2-135M --hf-golden golden_hf
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "tools"))
from hf_io import read_hf_model  # noqa: E402
from hf_tok_ref import RefTokenizer  # noqa: E402
from ref import Model  # noqa: E402
from verify_sampler import oracle_sample, py_uniforms  # noqa: E402

MAKE = HERE.parent / "tools" / "make_dummy_hf.py"
PROMPTS = ["Once upon a time", "hello 12 world", "caf\u00e9 \u65e5\u672c  x", "<|im_start|>user\nHi<|im_end|>", "  "]


def make_model(out: Path, *flags) -> None:
    subprocess.run([sys.executable, str(MAKE), "--out", str(out), *map(str, flags)], check=True, capture_output=True)


def set_config(directory: Path, **changes) -> None:
    path = directory / "config.json"
    cfg = json.loads(path.read_text(encoding="utf-8"))
    for k, v in changes.items():
        if v is None:
            cfg.pop(k, None)
        else:
            cfg[k] = v
    path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def eos_set(directory: Path) -> set[int]:
    e = json.loads((directory / "config.json").read_text(encoding="utf-8")).get("eos_token_id")
    return set() if e is None else ({e} if isinstance(e, int) else set(e))


class Case:
    """One model directory with its independent references."""

    def __init__(self, exe: str, directory: Path):
        self.exe, self.dir = exe, directory
        self.tok = RefTokenizer(directory)
        self.ckpt = read_hf_model(directory)
        self.ctx = self.ckpt.config.seq_len

    # ---- the independent expectation -----------------------------------------------------------
    def prompt_ids(self, prompt: str, add_bos: bool) -> list[int]:
        ids = self.tok.encode(prompt, add_bos)
        return ids if ids else [self.tok.bos]        # empty prompt: the model needs one input token

    def text(self, ids: list[int]) -> bytes:        # ids the tokenizer does not know (padding) print nothing
        return self.tok.decode([i for i in ids if i in self.tok.id_to_token])

    def greedy(self, prompt_ids: list[int], steps: int, ctx_cap: int = 0) -> list[int]:
        ctx = min(self.ctx, ctx_cap) if ctx_cap else self.ctx
        n_steps = min(steps, ctx) if steps > 0 else ctx
        model = Model(self.ckpt, max_len=n_steps)
        stops, gen, token = eos_set(self.dir), [], prompt_ids[0]
        for pos in range(n_steps):
            logits = model.forward(token, pos)
            if pos < len(prompt_ids) - 1:
                token = prompt_ids[pos + 1]          # still feeding the prompt: an end token here is harmless
                continue
            token = int(np.argmax(logits))
            gen.append(token)
            if token in stops:
                break
        return gen

    # ---- running the program -------------------------------------------------------------------
    def run(self, *extra, prompt: str = "", model_dir=None):
        with tempfile.TemporaryDirectory() as tmp:
            ids_file = Path(tmp) / "ids.txt"
            cmd = [self.exe, "--model", str(model_dir or self.dir), "--prompt", prompt, "--dump-tokens", str(ids_file),
                   *map(str, extra)]
            p = subprocess.run(cmd, capture_output=True)
            p_ids, g_ids = [], []
            if ids_file.exists():
                lines = ids_file.read_text().split("\n")
                p_ids = [int(x) for x in lines[0].split()]
                g_ids = [int(x) for x in lines[1].split()] if len(lines) > 1 else []
            return p.returncode, p.stdout, p.stderr.decode(errors="replace"), p_ids, g_ids


def check_greedy(c: Case, label: str, prompt: str, steps: int, failures: list[str], *flags,
                 add_bos: bool = False, ctx_cap: int = 0, expect_stop: bool | None = None) -> list[int]:
    p_exp = c.prompt_ids(prompt, add_bos)
    g_exp = c.greedy(p_exp, steps, ctx_cap)
    rc, out, err, p_ids, g_ids = c.run("--temperature", 0, "--steps", steps, *flags, prompt=prompt)
    if rc != 0:
        failures.append(f"{label}: exit code {rc}: {err.strip()}")
        return g_ids
    if p_ids != p_exp:
        failures.append(f"{label}: prompt ids {p_ids} != {p_exp}")
    if g_ids != g_exp:
        k = next((i for i, (a, b) in enumerate(zip(g_ids, g_exp)) if a != b), min(len(g_ids), len(g_exp)))
        failures.append(f"{label}: generated ids differ at index {k} (C++ {len(g_ids)} tokens, expected {len(g_exp)})")
    stops = eos_set(c.dir)
    ended = bool(g_exp) and g_exp[-1] in stops
    shown = g_exp[:-1] if ended else g_exp
    if out != c.text(p_exp) + c.text(shown):
        failures.append(f"{label}: stdout differs from prompt + decoded generated tokens")
    if ended != ("ended by an end token" in err):
        failures.append(f"{label}: the statistics line {'misses' if ended else 'wrongly claims'} the end-token note")
    if expect_stop is not None and ended != expect_stop:
        failures.append(f"{label}: scenario broken: expected stop={expect_stop}, the reference says {ended}")
    return g_exp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True, help="path to gi_generate")
    ap.add_argument("--model-dir", help="real Hugging Face model directory (part 2)")
    ap.add_argument("--hf-golden", help="golden directory from tools/hf_golden.py --prompt ... (part 2)")
    args = ap.parse_args()
    failures: list[str] = []
    n_greedy = n_sampled = n_errors = 0

    if args.model_dir:
        if not args.hf_golden:
            ap.error("--model-dir needs --hf-golden")
        return real_model(args, failures)

    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        configs = {
            "f32 tied, sequence tokenizer": ["--with-tokenizer", 250, "--seed", 0, "--max-pos", 96],
            "bf16 untied, bytelevel tokenizer, eos list": ["--with-tokenizer", 200, "--tokenizer-variant", "bytelevel",
                                                           "--dtype", "bf16", "--untied", "--eos-id", "2,3", "--seed", 1,
                                                           "--max-pos", 96],
            "f16, noregex tokenizer, head_dim 16": ["--with-tokenizer", 120, "--tokenizer-variant", "noregex",
                                                    "--dtype", "f16", "--dim", 48, "--heads", 4, "--kv-heads", 2,
                                                    "--head-dim", 16, "--seed", 2, "--max-pos", 96],
            "qwen3: QK-norm model, NFC + Qwen2-split tokenizer, bf16 tied": ["--qwen3", "--with-tokenizer", 220, "--tokenizer-variant", "qwen2",
                                                                          "--dtype", "bf16", "--dim", 48, "--heads", 4, "--kv-heads", 2,
                                                                          "--head-dim", 16, "--rope-theta", 1000000, "--eps", "1e-6",
                                                                          "--eos-id", "1,2", "--seed", 4, "--max-pos", 96],
            "padded vocabulary, ids unknown to the tokenizer": ["--with-tokenizer", 100, "--untied", "--vocab-pad", 4,
                                                                "--pad-sinks", "--seed", 3, "--eos-id", 1, "--max-pos", 96],
        }
        cases: dict[str, Case] = {}
        for name, flags in configs.items():
            d = tmp / f"m{len(cases)}"
            make_model(d, *flags)
            cases[name] = Case(args.exe, d)

        # ---- 1. greedy equals the independent re-derivation ---------------------------------------------
        for name, c in cases.items():
            for prompt in PROMPTS + [""]:
                check_greedy(c, f"[{name}] prompt {prompt!r}", prompt, 40, failures)
                n_greedy += 1

        # ---- 2. end tokens ---------------------------------------------------------------------------------
        base = cases["f32 tied, sequence tokenizer"]
        prompt = "Once upon a time"
        p_exp = base.prompt_ids(prompt, False)
        free = base.greedy(p_exp, 40)                 # no end token yet: eos is id 0, rarely sampled; check
        # (a) eos = a token that really is generated at index k: generation must stop right there
        for k in (0, 3, 9):
            d = tmp / f"stop{k}"
            shutil.copytree(base.dir, d)
            target = free[k]
            if target in free[:k]:
                continue                               # would stop earlier; the earlier index covers it
            set_config(d, eos_token_id=target)
            cs = Case(args.exe, d)
            g = check_greedy(cs, f"eos={target} at generated index {k}", prompt, 40, failures, expect_stop=True)
            if len(g) != k + 1:
                failures.append(f"eos at index {k}: expected {k + 1} generated tokens, the reference has {len(g)}")
            n_greedy += 1
        # (b) the stop token inside a list, as first and as second entry
        k = next(i for i in range(2, 30) if free[i] not in free[:i])
        unused = next(t for t in range(base.ckpt.config.vocab_size - 1, 0, -1) if t not in free)
        for label, ids in (("first", [free[k], unused]), ("second", [unused, free[k]])):
            d = tmp / f"stoplist_{label}"
            shutil.copytree(base.dir, d)
            set_config(d, eos_token_id=ids)
            g = check_greedy(Case(args.exe, d), f"eos list, stop token {label}", prompt, 40, failures, expect_stop=True)
            if len(g) != k + 1:
                failures.append(f"eos list ({label}): expected {k + 1} generated tokens, got {len(g)}")
            n_greedy += 1
        # (c) an end token INSIDE the prompt must not stop generation
        d = tmp / "eosprompt"
        shutil.copytree(base.dir, d)
        im_end = base.tok.added["<|im_end|>"]
        set_config(d, eos_token_id=im_end)
        ce = Case(args.exe, d)
        text_prompt = "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
        pe = ce.prompt_ids(text_prompt, False)
        if im_end not in pe:
            failures.append("scenario broken: the prompt has no end token")
        g = check_greedy(ce, "end token inside the prompt", text_prompt, 40, failures)
        if len(g) < 5:
            failures.append(f"end token inside the prompt: only {len(g)} tokens generated (scenario too short to prove anything)")
        n_greedy += 1
        # (d) no end token configured at all: runs to the limit
        d = tmp / "noeos"
        shutil.copytree(base.dir, d)
        set_config(d, eos_token_id=None)
        g = check_greedy(Case(args.exe, d), "no eos_token_id in config.json", prompt, 25, failures, expect_stop=False)
        if len(g) != 25 - len(p_exp) + 1:
            failures.append(f"no eos: expected {25 - len(p_exp) + 1} generated tokens, got {len(g)}")
        n_greedy += 1

        # ---- 3. --bos, empty prompt, no BOS -----------------------------------------------------------------
        check_greedy(base, "--bos", prompt, 30, failures, "--bos", add_bos=True)
        rc, _, _, p_ids, _ = base.run("--temperature", 0, "--steps", 10, "--bos", prompt=prompt)
        if rc != 0 or p_ids[:1] != [base.tok.bos] or len(p_ids) != len(p_exp) + 1:
            failures.append(f"--bos: prompt ids {p_ids}, expected BOS + {p_exp}")
        check_greedy(base, "--bos with an empty prompt", "", 20, failures, "--bos", add_bos=True)
        n_greedy += 3
        d = tmp / "nobos"
        shutil.copytree(base.dir, d)
        set_config(d, bos_token_id=None)
        rc, _, err, _, _ = base.run("--temperature", 0, "--steps", 10, prompt="", model_dir=d)
        if rc != 1 or "BOS" not in err:
            failures.append(f"empty prompt without a BOS token: exit {rc}, stderr {err.strip()!r} (expected a clean error)")
        rc, _, err, _, _ = base.run("--temperature", 0, "--steps", 10, "--bos", prompt="x", model_dir=d)
        if rc != 1:
            failures.append(f"--bos without a BOS token: exit {rc}, expected 1")
        n_errors += 2

        # ---- 4. the leading-space rule of llama2.c must NOT apply to Hugging Face models -----------------------
        # empty prompt = [BOS]; find a model whose first generated token starts with a space byte
        found = None
        for seed in range(60):
            d = tmp / f"space{seed}"
            make_model(d, "--with-tokenizer", 250, "--seed", 100 + seed, "--max-pos", 64)
            cs = Case(args.exe, d)
            first = cs.greedy(cs.prompt_ids("", False), 3)
            if first and cs.text(first[:1]).startswith(b" ") and first[0] not in eos_set(d):
                found = (cs, seed)
                break
        if found is None:
            failures.append("no model with a space-leading first token found in 60 seeds (test cannot prove the rule)")
        else:
            cs, seed = found
            g = check_greedy(cs, f"first token starts with a space (seed {100 + seed})", "", 6, failures)
            rc, out, _, _, _ = cs.run("--temperature", 0, "--steps", 6, "--no-echo", prompt="")
            if rc != 0 or not out.startswith(b" "):
                failures.append(f"HF output dropped the leading space after BOS: {out[:20]!r}")
            n_greedy += 1

        # ---- 5. sampled run re-derived token by token -------------------------------------------------------------
        for name in ("f32 tied, sequence tokenizer", "bf16 untied, bytelevel tokenizer, eos list"):
            c = cases[name]
            temp, top_p, seed = 0.9, 0.9, 7
            for prompt in ("Once upon a time", "hello 12 world"):
                rc, out, err, p_ids, g_ids = c.run("--temperature", temp, "--top-p", top_p, "--seed", seed,
                                                   "--steps", 40, prompt=prompt)
                if rc != 0 or not g_ids:
                    failures.append(f"[{name}] sampled run failed: rc {rc} {err.strip()}")
                    continue
                seq, n_p = p_ids + g_ids, len(p_ids)
                model = Model(c.ckpt, max_len=len(seq))
                uni = py_uniforms(seed, len(g_ids))
                stops, bad, near = eos_set(c.dir), 0, 0
                for pos, tok in enumerate(seq[:-1]):
                    logits = model.forward(tok, pos)
                    if pos < n_p - 1:
                        continue
                    k = pos - (n_p - 1)
                    want, ambiguous = oracle_sample(logits, temp, 0, top_p, uni[k], tol=2e-5)
                    if want != g_ids[k]:
                        if ambiguous:
                            near += 1
                        else:
                            bad += 1
                            failures.append(f"[{name}] sampled token {k}: C++ {g_ids[k]}, oracle {want}")
                    if g_ids[k] in stops and k != len(g_ids) - 1:
                        failures.append(f"[{name}] generation continued after an end token (index {k})")
                shown = g_ids[:-1] if g_ids[-1] in stops else g_ids
                if out != c.text(p_ids) + c.text(shown):
                    failures.append(f"[{name}] sampled stdout differs from the decoded tokens")
                n_sampled += 1

        # ---- 6. --max-seq-len, --tokenizer DIR -----------------------------------------------------------------------
        g = check_greedy(base, "--max-seq-len 12", "Once upon a time", 60, failures, "--max-seq-len", 12, ctx_cap=12)
        if len(g) != 12 - len(p_exp) + 1 and (not g or g[-1] not in eos_set(base.dir)):
            failures.append(f"--max-seq-len 12: expected {12 - len(p_exp) + 1} generated tokens, got {len(g)}")
        n_greedy += 1
        d = tmp / "weights_only"
        make_model(d, "--seed", 0, "--max-pos", 96, "--vocab", base.ckpt.config.vocab_size)
        shutil.copy(base.dir / "config.json", d / "config.json")   # same config (incl. vocab) as the full model
        shutil.copy(base.dir / "model.safetensors", d / "model.safetensors")
        rc, _, err, _, _ = base.run("--temperature", 0, "--steps", 10, prompt="x", model_dir=d)
        if rc != 1 or "tokenizer.json" not in err:
            failures.append(f"a model directory without tokenizer.json: exit {rc}, stderr {err.strip()!r}")
        rc, out, err, p_ids, g_ids = base.run("--temperature", 0, "--steps", 30, "--tokenizer", base.dir, prompt=prompt,
                                              model_dir=d)
        g_exp = base.greedy(base.prompt_ids(prompt, False), 30)
        if rc != 0 or g_ids != g_exp:
            failures.append(f"--tokenizer DIR: exit {rc}, tokens differ from the reference ({err.strip()!r})")
        n_greedy += 1
        n_errors += 1

        # ---- 7. error cases -----------------------------------------------------------------------------------------
        big = tmp / "big_tok"
        make_model(big, "--with-tokenizer", 300, "--seed", 5)
        small = tmp / "small_model"
        make_model(small, "--vocab", 100, "--seed", 5)            # 100-token model, tokenizer of 564 tokens
        rc, _, err, _, _ = base.run("--steps", 5, "--tokenizer", big, prompt="x", model_dir=small)
        if rc != 1 or "do not belong together" not in err:
            failures.append(f"tokenizer larger than the model vocabulary: exit {rc}, stderr {err.strip()!r}")
        rc, _, err, _, _ = base.run("--steps", 5, prompt="a " * 400)
        if rc != 1 or "does not fit" not in err:
            failures.append(f"prompt longer than the context: exit {rc}, stderr {err.strip()!r}")
        n_errors += 2

    print(f"dummy models: {len(cases)} models, {n_greedy} greedy runs, {n_sampled} sampled runs, "
          f"{n_errors} error cases checked against independent references")
    if failures:
        print("FAIL")
        for f in failures:
            print("  -", f)
        return 1
    print("PASS")
    return 0


def real_model(args, failures: list[str]) -> int:
    golden_dir = Path(args.hf_golden)
    g = json.loads((golden_dir / "golden.json").read_text(encoding="utf-8"))
    if g.get("backend") != "torch":
        print(f"note: the golden file was made by the {g.get('backend')} backend, not by PyTorch/transformers")
    prompt, tokens, n_prompt = g["prompt_text"], g["tokens"], g["n_prompt"]
    expected = tokens[n_prompt:]
    steps = n_prompt + len(expected) - 1                  # positions that produce exactly len(expected) tokens
    if g.get("prompt_ids_from") != "tokenizers":
        print("note: the prompt ids of this golden file did not come from the `tokenizers` library "
              "(transformers' AutoTokenizer may differ for texts with digits)")
    c = Case.__new__(Case)
    c.exe, c.dir = args.exe, Path(args.model_dir)
    rc, out, err, p_ids, g_ids = c.run("--temperature", 0, "--steps", steps, prompt=prompt)
    if rc != 0:
        print(f"FAIL: gi_generate exited with {rc}: {err.strip()}")
        return 1
    if p_ids != tokens[:n_prompt]:
        failures.append(f"prompt ids differ: C++ {p_ids} vs transformers {tokens[:n_prompt]}")
    if g_ids != expected[:len(g_ids)]:
        k = next((i for i, (a, b) in enumerate(zip(g_ids, expected)) if a != b), len(g_ids))
        failures.append(f"greedy tokens differ from transformers at generated index {k}: "
                        f"C++ {g_ids[k:k + 5]} vs {expected[k:k + 5]}")
    elif len(g_ids) < len(expected) and "ended by an end token" not in err:
        failures.append(f"only {len(g_ids)} of {len(expected)} tokens generated and no end token was reported")
    if failures:
        print("FAIL")
        for f in failures:
            print("  -", f)
        return 1
    print(f"prompt: {len(p_ids)} tokens identical; greedy: {len(g_ids)} of {len(expected)} generated tokens identical "
          f"to transformers" + (" (then an end token)" if len(g_ids) < len(expected) else ""))
    print(out.decode("utf-8", errors="replace"))
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
