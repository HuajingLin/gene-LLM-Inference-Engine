#!/usr/bin/env python3
"""verify_nfc.py - acceptance for the NFC normalizer (gi_unicode_check nfc).

Python's unicodedata.normalize('NFC', ...) is the answer key. Checked:
  1. the table file matches the checksum recorded by tools/gen_norm_tables.py
  2. every single code point (surrogates excluded), in one run
  3. every pair (base, mark) of 400 bases (letters, Hangul jamo and syllables, composition exclusions, ...) with
     every code point that has a non-zero combining class
  4. random sequences made of marks with different combining classes, bases, Hangul jamo and syllables: this exercises
     canonical reordering, blocking and the Hangul arithmetic
  5. text with invalid UTF-8 bytes in it: valid stretches are normalized separately and the bad bytes pass through
If your Python has a newer Unicode version than the tables, samples containing characters that were still unassigned in
the tables' version are skipped (reported); with the same version the comparison is strict.

  python tests/verify_nfc.py --exe build/gi_unicode_check
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import unicodedata
from pathlib import Path


def nfc_cpp(exe: str, texts: list[bytes]) -> list[bytes]:
    assert all(b"\0" not in t for t in texts)
    p = subprocess.run([exe, "nfc"], input=b"\0".join(texts), capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode(errors="replace"))
    out = p.stdout.split(b"\0")
    if len(out) != len(texts):
        raise RuntimeError(f"expected {len(texts)} results, got {len(out)}")
    return out


def reference(raw: bytes) -> bytes:
    """NFC of valid stretches; undecodable bytes (escaped as lone surrogates) stay and separate the stretches."""
    s = raw.decode("utf-8", "surrogateescape")
    out, chunk = [], []
    for ch in s:
        if 0xDC80 <= ord(ch) <= 0xDCFF:
            out.append(unicodedata.normalize("NFC", "".join(chunk)))
            chunk = []
            out.append(ch)
        else:
            chunk.append(ch)
    out.append(unicodedata.normalize("NFC", "".join(chunk)))
    return "".join(out).encode("utf-8", "surrogateescape")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    d = Path(__file__).resolve().parent.parent / "src" / "tokenizer"
    ap.add_argument("--tables", default=str(d / "unicode_norm_tables.inc"))
    ap.add_argument("--meta", default=str(d / "unicode_norm_tables.meta.json"))
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--random-samples", type=int, default=150000)
    args = ap.parse_args()
    failures: list[str] = []
    rng = random.Random(args.seed)

    meta = json.loads(Path(args.meta).read_text())
    if hashlib.sha256(Path(args.tables).read_bytes()).hexdigest() != meta["sha256"]:
        failures.append("the table file does not match its recorded checksum (edited or damaged)")
    same_version = unicodedata.unidata_version == meta["unicode_version"]
    unassigned = [tuple(r) for r in meta["unassigned"]]

    def excused(text: str) -> bool:
        if same_version:
            return False
        return any(lo <= ord(c) <= hi for c in text for lo, hi in unassigned)

    skipped = 0

    def check(label: str, texts: list[str]) -> None:
        nonlocal skipped
        keep = []
        for t in texts:
            if excused(t):
                skipped += 1
            else:
                keep.append(t)
        got = nfc_cpp(args.exe, [t.encode("utf-8") for t in keep])
        bad = [(t, g) for t, g in zip(keep, got) if g != unicodedata.normalize("NFC", t).encode("utf-8")]
        if bad:
            t, g = bad[0]
            failures.append(f"{label}: {len(bad)} of {len(keep)} differ, e.g. {[hex(ord(c)) for c in t]} -> "
                            f"{[hex(ord(c)) for c in g.decode('utf-8', 'replace')]}, expected "
                            f"{[hex(ord(c)) for c in unicodedata.normalize('NFC', t)]}")
        print(f"{label}: {len(keep)} texts checked")

    # 2. every single code point
    check("single code points", [chr(c) for c in range(1, 0x110000) if not 0xD800 <= c <= 0xDFFF])

    marks = [c for c in range(0x110000) if not 0xD800 <= c <= 0xDFFF and unicodedata.combining(chr(c))]
    letters = [c for c in range(0x20, 0x250) if unicodedata.category(chr(c)).startswith("L")]
    hangul_l, hangul_v, hangul_t = list(range(0x1100, 0x1113)), list(range(0x1161, 0x1176)), list(range(0x11A8, 0x11C3))
    syllables = [rng.randrange(0xAC00, 0xD7A4) for _ in range(40)] + [0xAC00, 0xAC01, 0xD7A3]
    exclusions = [0x0958, 0x0959, 0x095A, 0x0DDC, 0x2126, 0x212A, 0x212B, 0x2ADC, 0x1D15E, 0x0F43, 0x0344, 0x0340, 0x0341]
    others = [rng.randrange(0x370, 0x3000) for _ in range(60)]
    bases = (letters[:160] + hangul_l[:4] + hangul_v[:4] + hangul_t[:4] + [0x11A7, 0x1160, 0x1113, 0x1175, 0x1176] + syllables + exclusions + others)[:400]
    bases = [b for b in bases if not 0xD800 <= b <= 0xDFFF]
    # 3. every (base, mark) pair
    check("base + mark pairs", [chr(b) + chr(m) for b in bases for m in marks])

    # 4. random sequences
    jamo = list(range(0x1100, 0x1200))   # the whole block, including the edges 0x1112/0x1113, 0x1160, 0x11A7
    pool = marks + bases + hangul_l + hangul_v + hangul_t + jamo + syllables + [0x20, 0x41, 0x61, 0x1E0C, 0xE9, 0x300, 0x323]
    seqs = []
    for _ in range(args.random_samples):
        n = rng.randint(2, 9)
        seqs.append("".join(chr(rng.choice(pool)) for _ in range(n)))
    check("random sequences", seqs)

    # 5. invalid bytes
    pieces = [b"a", b"e\xcc\x81", "\u00e9".encode(), b"\xcc\x81", b"\xff", b"\xe2\x82", b"\x80", b"\xed\xa0\x80", b"\xf0\x9f\x98",
              "\u1100".encode(), "\u1161".encode(), b"Z", b" ", "\u4e2d".encode(), b"\xc3", b"\xf4\x90\x80\x80"]
    texts = [b"".join(rng.choice(pieces) for _ in range(rng.randint(1, 8))) for _ in range(20000)]
    got = nfc_cpp(args.exe, texts)
    bad = [(t, g) for t, g in zip(texts, got) if g != reference(t)]
    if bad:
        failures.append(f"invalid bytes: {len(bad)} of {len(texts)} differ, e.g. {bad[0][0]!r} -> {bad[0][1]!r}, "
                        f"expected {reference(bad[0][0])!r}")
    print(f"invalid UTF-8: {len(texts)} texts checked")

    # ascii and empty are returned unchanged
    if nfc_cpp(args.exe, [b"", b"plain ascii text 123"]) != [b"", b"plain ascii text 123"]:
        failures.append("ASCII / empty text was changed")

    if failures:
        print("FAIL")
        for f in failures:
            print("  -", f)
        return 1
    note = "" if same_version else f" ({skipped} samples with characters newer than the tables were skipped)"
    print(f"PASS  (NFC matches Python's unicodedata, Unicode {unicodedata.unidata_version}{note})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
