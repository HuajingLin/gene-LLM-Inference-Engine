#!/usr/bin/env python3
"""verify_hf_tokenizer.py - acceptance for the C++ byte-level BPE tokenizer (gi_hf_tokenize).

Checks, for every tokenizer directory tested:
  1. encode: token ids equal those of the independent Python reference (tools/hf_tok_ref.py)
  2. pre-tokenizer: the pieces equal the reference's (a real regex engine, expanded Unicode classes)
  3. round trip: decode(encode(text)) gives back the exact bytes of the text (NFC-normalized, outside added tokens,
     for tokenizers that have the NFC normalizer)
  4. decode: random id sequences (special tokens included, with and without skipping them) decode to the same bytes
  5. add_bos, and error handling (out-of-range ids, unsupported tokenizer.json settings, broken files)
  6. with --golden: ids, pieces and decoded text equal what the Hugging Face `tokenizers` library produced
     (generate the file with tools/hf_tokenizer_golden.py); the Python reference is compared with it too

Without --model-dir, four generated tokenizers are tested (Digits+ByteLevel, ByteLevel, ByteLevel without regex,
and Digits+ByteLevel with ignore_merges).

Usage:
  python tests/verify_hf_tokenizer.py --exe build/gi_hf_tokenize                       # generated tokenizers
  python tests/verify_hf_tokenizer.py --exe build/gi_hf_tokenize --model-dir models/SmolLM2-135M
  python tests/verify_hf_tokenizer.py --exe build/gi_hf_tokenize --model-dir models/SmolLM2-135M --golden tok_golden.json
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tools"))
from hf_tok_ref import QWEN2_PATTERN, RefTokenizer, to_byte_level  # noqa: E402
from tok_corpus import all_cases  # noqa: E402

MAKE_TOKENIZER = str(HERE.parent / "tools" / "make_dummy_hf_tokenizer.py")


def run(exe, directory, mode, payload=None, *flags):
    data = json.dumps(payload).encode("utf-8") if payload is not None else b""
    return subprocess.run([exe, str(directory), mode, *flags], input=data, capture_output=True)


def batch(exe, directory, mode, payload, *flags):
    p = run(exe, directory, mode, payload, *flags)
    if p.returncode != 0:
        raise RuntimeError(f"{mode} failed: {p.stderr.decode(errors='replace').strip()}")
    return json.loads(p.stdout.decode("utf-8"))


def short(text: str, n: int = 50) -> str:
    return repr(text if len(text) <= n else text[:n] + "...")


def first_difference(a, b) -> str:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return f"first difference at index {i}: {x!r} vs {y!r}"
    return f"lengths {len(a)} vs {len(b)}"


def check_directory(exe: str, directory: Path, texts: list[str], label: str, failures: list[str], golden=None) -> dict:
    ref = RefTokenizer(directory)
    added = list(ref.added)
    stats = {}

    # ---- 1. ids ---------------------------------------------------------------------------------------
    cpp_ids = batch(exe, directory, "encode-batch", texts)
    ref_ids = [ref.encode(t) for t in texts]
    bad = [i for i, (a, b) in enumerate(zip(cpp_ids, ref_ids)) if a != b]
    for i in bad[:4]:
        failures.append(f"[{label}] encode {short(texts[i])}: C++ and reference differ ({first_difference(cpp_ids[i], ref_ids[i])})")
    stats["texts"] = len(texts)

    # ---- 2. pre-tokenizer pieces (texts with no added token in them) ----------------------------------------
    plain = [t for t in texts if not any(a in t for a in added)]
    cpp_pieces = batch(exe, directory, "pretokenize-batch", plain)
    ref_pieces = [ref.pre_tokenize(t) for t in plain]
    bad_p = [i for i, (a, b) in enumerate(zip(cpp_pieces, ref_pieces)) if a != b]
    for i in bad_p[:4]:
        failures.append(f"[{label}] pieces of {short(plain[i])}: C++ {cpp_pieces[i][:6]} vs reference {ref_pieces[i][:6]} ...")

    # ---- 3. round trip -------------------------------------------------------------------------------------------
    decoded = [bytes.fromhex(h) for h in batch(exe, directory, "decode-batch", cpp_ids)]
    for i, (t, d) in enumerate(zip(texts, decoded)):
        if d != ref.normalized_text(t).encode("utf-8"):
            failures.append(f"[{label}] round trip failed for {short(t)}: got {d[:40]!r}")
            break

    # ---- 4. decoding arbitrary id sequences ---------------------------------------------------------------------
    rng = random.Random(11)
    vocab_max = max(ref.id_to_token)
    seqs = [[rng.randint(0, vocab_max) for _ in range(rng.randint(0, 25))] for _ in range(150)]
    seqs += [[i for i in sorted(ref.special_ids)], list(range(0, min(vocab_max, 300)))]
    for skip in (False, True):
        got = batch(exe, directory, "decode-batch", seqs, *(["--skip-special"] if skip else []))
        for ids, h in zip(seqs, got):
            if bytes.fromhex(h) != ref.decode(ids, skip_special=skip):
                failures.append(f"[{label}] decode (skip_special={skip}) of {ids[:8]}... differs from the reference")
                break

    # ---- 5. golden file from Hugging Face ---------------------------------------------------------------------------------
    if golden is not None:
        cases = golden["cases"]
        g_texts = [c["text"] for c in cases]
        assert g_texts == texts, "internal error: the texts must come from the golden file"
        for c, ids_cpp, ids_ref in zip(cases, cpp_ids, ref_ids):
            if ids_ref != c["ids"]:
                failures.append(f"[{label}] the PYTHON REFERENCE disagrees with Hugging Face on {short(c['text'])} "
                                f"({first_difference(ids_ref, c['ids'])}) - the reference itself needs fixing")
                break
        n_bad = 0
        for c, ids_cpp in zip(cases, cpp_ids):
            if ids_cpp != c["ids"]:
                n_bad += 1
                if n_bad <= 4:
                    failures.append(f"[{label}] ids of {short(c['text'])} differ from Hugging Face: {first_difference(ids_cpp, c['ids'])}")
        stats["golden_ids_ok"] = len(cases) - n_bad
        if n_bad and ref.split_digits:
            # Do all differences vanish if the Digits stage is ignored? Then the golden file was produced by a
            # pipeline that does not apply the pre-tokenizer declared in tokenizer.json.
            no_digits = RefTokenizer(directory)
            no_digits.split_digits = False
            wrong = [c for c, ids_ref in zip(cases, ref_ids) if ids_ref != c["ids"]]
            explained = sum(1 for c in wrong if no_digits.encode(c["text"]) == c["ids"])
            if wrong and explained == len(wrong):
                failures.insert(0, f"[{label}] all {len(wrong)} differences disappear when the Digits stage is ignored: the golden file "
                                   "behaves like a ByteLevel-only pipeline, although tokenizer.json declares Digits first. The golden "
                                   "file was probably made by a library that does not honour that stage (for example AutoTokenizer in "
                                   "transformers 5.x). Run tools/hf_tokenizer_probe.py to find out which one follows the file.")
        with_pieces = [(c["text"], c["pieces"]) for c in cases if c.get("pieces") is not None]
        if with_pieces:
            got_pieces = batch(exe, directory, "pretokenize-batch", [t for t, _ in with_pieces])
            for (t, want), got in zip(with_pieces, got_pieces):
                if [to_byte_level(p) for p in got] != want:
                    failures.append(f"[{label}] pre-tokenizer pieces of {short(t)} differ from Hugging Face")
                    break
        if golden.get("decode_cases"):
            dc = golden["decode_cases"]
            got = batch(exe, directory, "decode-batch", [d["ids"] for d in dc])
            for d, h in zip(dc, got):
                if bytes.fromhex(h).decode("utf-8", errors="replace") != d["decoded"]:
                    failures.append(f"[{label}] decoding {d['ids'][:8]}... differs from Hugging Face")
                    break
    return stats


def check_errors_and_options(exe: str, good: Path, tmp: Path, failures: list[str]) -> int:
    n = 0

    def expect_failure(label, directory, needle, mode="info"):
        nonlocal n
        n += 1
        p = run(exe, directory, mode, [] if mode.endswith("batch") else None)
        if p.returncode != 1:
            failures.append(f"[{label}] exit code {p.returncode}, expected 1")
        elif needle.lower() not in p.stderr.decode(errors="replace").lower():
            failures.append(f"[{label}] error message lacks '{needle}': {p.stderr.decode(errors='replace').strip()[:100]}")

    def variant(name, mutate):
        d = tmp / f"case_{name}"
        shutil.copytree(good, d)
        tj = json.loads((d / "tokenizer.json").read_text(encoding="utf-8"))
        mutate(tj)
        (d / "tokenizer.json").write_text(json.dumps(tj, ensure_ascii=False), encoding="utf-8")
        return d

    expect_failure("missing directory", tmp / "nope", "tokenizer.json")
    empty = tmp / "empty"
    empty.mkdir()
    expect_failure("no tokenizer.json", empty, "tokenizer.json")
    broken = tmp / "broken"
    shutil.copytree(good, broken)
    (broken / "tokenizer.json").write_text('{"model": ', encoding="utf-8")
    expect_failure("invalid JSON", broken, "tokenizer.json")
    expect_failure("normalizer", variant("norm", lambda t: t.update(normalizer={"type": "Lowercase"})), "normalizer")
    qwen_split = {"type": "Split", "pattern": {"Regex": r"\w+"}, "behavior": "Isolated", "invert": False}
    byte_level = {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": False}
    expect_failure("Split with another regex", variant("split1", lambda t: t.update(
        pre_tokenizer={"type": "Sequence", "pretokenizers": [qwen_split, byte_level]})), "Qwen2")
    expect_failure("Split behavior", variant("split2", lambda t: t.update(
        pre_tokenizer={"type": "Sequence", "pretokenizers": [dict(qwen_split, pattern={"Regex": QWEN2_PATTERN}, behavior="Removed"), byte_level]})), "Isolated")
    expect_failure("Split followed by the GPT-2 regex", variant("split3", lambda t: t.update(
        pre_tokenizer={"type": "Sequence", "pretokenizers": [dict(qwen_split, pattern={"Regex": QWEN2_PATTERN}), dict(byte_level, use_regex=True)]})), "use_regex")
    expect_failure("post-processor", variant("post", lambda t: t.update(post_processor={"type": "TemplateProcessing"})), "post_processor")
    expect_failure("decoder", variant("dec", lambda t: t.update(decoder={"type": "Metaspace"})), "decoder")
    expect_failure("pre-tokenizer Split", variant("pre", lambda t: t.update(pre_tokenizer={"type": "Split"})), "pre_tokenizer")
    expect_failure("pre-tokenizer null", variant("prenull", lambda t: t.update(pre_tokenizer=None)), "pre_tokenizer")
    expect_failure("add_prefix_space", variant("prefix", lambda t: t["pre_tokenizer"].update(add_prefix_space=True)
                   if t["pre_tokenizer"]["type"] == "ByteLevel" else t["pre_tokenizer"]["pretokenizers"][1].update(add_prefix_space=True)), "add_prefix_space")
    expect_failure("model type", variant("unigram", lambda t: t["model"].update(type="Unigram")), "model type")
    expect_failure("byte_fallback", variant("bf", lambda t: t["model"].update(byte_fallback=True)), "byte_fallback")
    expect_failure("subword prefix", variant("csp", lambda t: t["model"].update(continuing_subword_prefix="##")), "continuing_subword_prefix")
    expect_failure("bad merge", variant("merge", lambda t: t["model"]["merges"].append("zzz yyy")), "merge")
    expect_failure("merge without space", variant("merge2", lambda t: t["model"]["merges"].append("nospace")), "merge")
    expect_failure("lstrip", variant("lstrip", lambda t: t["added_tokens"][0].update(lstrip=True)), "lstrip")
    dup = variant("dupid", lambda t: t["model"]["vocab"].update({"zzzdup": 0}))
    expect_failure("duplicate id", dup, "share the id")

    # end-of-sequence ids: config.json's, then the new ones of generation_config.json
    eos_dir = tmp / "case_eos"
    shutil.copytree(good, eos_dir)
    cfg = json.loads((eos_dir / "config.json").read_text()) if (eos_dir / "config.json").exists() else {}
    cfg["eos_token_id"] = 3
    (eos_dir / "config.json").write_text(json.dumps(cfg))
    (eos_dir / "generation_config.json").write_text(json.dumps({"eos_token_id": [5, 3, 7]}))
    n += 1
    info = run(exe, eos_dir, "info").stdout.decode()
    if "eos ids: 3 5 7" not in info:
        failures.append(f"eos ids of config.json and generation_config.json were not combined: {info.strip().splitlines()[-1]!r}")

    # ids out of range, malformed input of the batch modes
    n += 2
    p = run(exe, good, "decode-batch", [[10**7]])
    if p.returncode != 1:
        failures.append("decoding an out-of-range id was not rejected")
    p = run(exe, good, "encode-batch", {"not": "a list"})
    if p.returncode != 1:
        failures.append("encode-batch accepted JSON that is not an array")

    # bos / eos from config.json
    n += 3
    cfg_dir = tmp / "with_config"
    shutil.copytree(good, cfg_dir)
    (cfg_dir / "config.json").write_text(json.dumps({"bos_token_id": 3, "eos_token_id": [0, 2]}), encoding="utf-8")
    info = run(exe, cfg_dir, "info").stdout.decode()
    if "bos id: 3" not in info or "eos ids: 0 2" not in info:
        failures.append(f"bos/eos ids from config.json were not read: {info!r}")
    with_bos = batch(exe, cfg_dir, "encode-batch", ["ab"], "--bos")[0]
    plain = batch(exe, cfg_dir, "encode-batch", ["ab"])[0]
    if with_bos != [3] + plain:
        failures.append("--bos did not prepend the bos token")
    if run(exe, good, "encode-batch", ["ab"], "--bos").returncode != 1 and not (good / "config.json").exists():
        failures.append("--bos without a bos_token_id in the model was not rejected")
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True, help="path to gi_hf_tokenize")
    ap.add_argument("--model-dir", help="a real model directory (otherwise generated tokenizers are tested)")
    ap.add_argument("--golden", help="JSON from tools/hf_tokenizer_golden.py")
    ap.add_argument("--fuzz", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    failures: list[str] = []

    if args.model_dir:
        golden = json.loads(Path(args.golden).read_text(encoding="utf-8")) if args.golden else None
        texts = [c["text"] for c in golden["cases"]] if golden else all_cases(args.fuzz, args.seed)
        stats = check_directory(args.exe, Path(args.model_dir), texts, "model", failures, golden)
        if failures:
            print("FAIL")
            for f in failures[:12]:
                print("  -", f)
            return 1
        extra = f", {stats['golden_ids_ok']}/{stats['texts']} identical to Hugging Face" if golden else ""
        print(f"PASS  ({stats['texts']} texts: ids, pieces, round trip and decoding match the reference{extra})")
        return 0

    texts = all_cases(args.fuzz, args.seed)
    n_dirs = n_err = 0
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        for name, flags in (("sequence", ["--variant", "sequence"]), ("bytelevel", ["--variant", "bytelevel"]),
                            ("noregex", ["--variant", "noregex"]), ("qwen2", ["--variant", "qwen2"]),
                            ("sequence+ignore_merges", ["--variant", "sequence", "--ignore-merges"])):
            d = tmp / f"tok_{name.replace('+', '_')}"
            subprocess.run([sys.executable, MAKE_TOKENIZER, "--out", str(d), "--merges", "300", *flags],
                           check=True, capture_output=True)
            check_directory(args.exe, d, texts, name, failures)
            n_dirs += 1
            if name == "sequence":
                n_err = check_errors_and_options(args.exe, d, tmp, failures)
    if failures:
        print("FAIL")
        for f in failures[:12]:
            print("  -", f)
        if len(failures) > 12:
            print(f"  ... and {len(failures) - 12} more")
        return 1
    print(f"PASS  ({n_dirs} generated tokenizers x {len(texts)} texts, plus {n_err} error and option checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
