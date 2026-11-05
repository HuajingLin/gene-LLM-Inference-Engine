#!/usr/bin/env python3
"""verify_unicode.py - acceptance for the Unicode helpers (gi_unicode_check).

  1. lookup code: the C++ classification of all 1,114,112 code points (letter, number, space) equals the
     classification read out of the table file src/tokenizer/unicode_tables.inc by an independent Python parser
     (always strict, whatever Python version you have)
  2. table file: its SHA-256 equals the checksum recorded by tools/gen_unicode_tables.py (detects edits/damage)
  3. tables vs Unicode: the tables are compared with Python's unicodedata (letter = category L*, number = N*,
     space = the White_Space property). This is strict when your Python has the Unicode version the tables were
     generated from. If your Python is NEWER, characters assigned after the tables' version legitimately differ and
     are excused; a few already-assigned characters may also have been reclassified by the Unicode Consortium
     (reported, and tolerated up to a small number). To compare strictly, run tools/gen_unicode_tables.py and rebuild.
  4. UTF-8 decoding: every valid code point round-trips; valid text decodes exactly like Python
  5. malformed UTF-8 never crashes, always makes progress, and yields U+FFFD for what it cannot decode
  6. UTF-8 encoding of every code point, with invalid values (surrogates, > U+10FFFF) written as U+FFFD

Usage:
  python tests/verify_unicode.py --exe build/gi_unicode_check
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

import numpy as np

# The White_Space property, written out independently of tools/gen_unicode_tables.py.
WHITE_SPACE = set(range(0x09, 0x0E)) | {0x20, 0x85, 0xA0, 0x1680, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000} | set(range(0x2000, 0x200B))


def run(exe: str, mode: str, data: bytes = b"") -> bytes:
    p = subprocess.run([exe, mode], input=data, capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode(errors="replace"))
    return p.stdout


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True, help="path to gi_unicode_check")
    default_dir = Path(__file__).resolve().parent.parent / "src" / "tokenizer"
    ap.add_argument("--tables", default=str(default_dir / "unicode_tables.inc"), help="the table file compiled into the tool")
    ap.add_argument("--meta", default=str(default_dir / "unicode_tables.meta.json"), help="metadata written with the tables")
    ap.add_argument("--max-reclassified", type=int, default=200,
                    help="with a newer Python: most already-assigned code points allowed to differ")
    ap.add_argument("--seed", type=int, default=2)
    args = ap.parse_args()
    failures: list[str] = []

    # ---- 1. lookup code against the table file --------------------------------------------------
    got = np.frombuffer(run(args.exe, "classes"), dtype=np.uint8)
    if got.size != 0x110000:
        print(f"FAIL: expected 1114112 class bytes, got {got.size}")
        return 1
    table_text = Path(args.tables).read_text(encoding="utf-8")
    from_table = np.zeros(0x110000, dtype=np.uint8)
    for name, bit in (("kLetterRanges", 1), ("kNumberRanges", 2), ("kSpaceRanges", 4)):
        body = re.search(name + r"\[\] = \{(.*?)\n\};", table_text, re.S)
        if not body:
            print(f"FAIL: {name} not found in {args.tables}")
            return 1
        for lo, hi in re.findall(r"\{0x([0-9A-Fa-f]+), 0x([0-9A-Fa-f]+)\}", body.group(1)):
            from_table[int(lo, 16):int(hi, 16) + 1] |= bit
    diff = np.nonzero(got != from_table)[0]
    tool_matches_file = diff.size == 0
    if diff.size:
        header = table_text.splitlines()[0] if table_text else ""
        found = re.search(r"Unicode (\d+(?:\.\d+)*)", header)
        failures.append(
            f"the compiled tool disagrees with the table file at {diff.size} code points, e.g. "
            + ", ".join(f"U+{int(c):04X} (tool {int(got[c])}, file {int(from_table[c])})" for c in diff[:5])
            + f". The file on disk is for Unicode {found.group(1) if found else '?'}. Most likely the tool was built from "
              "different tables (for example before the file was regenerated): rebuild it (cmake --build build) and run again.")

    # ---- 2. table file integrity -----------------------------------------------------------------
    meta = json.loads(Path(args.meta).read_text(encoding="utf-8"))
    if hashlib.sha256(Path(args.tables).read_bytes()).hexdigest() != meta["sha256"]:
        failures.append("the table file does not match the checksum recorded when it was generated "
                        "(edited or damaged? regenerate it with tools/gen_unicode_tables.py)")

    # ---- 3. tables against Python's Unicode database ---------------------------------------------
    # (meaningless while the tool does not even match the table file: that was reported above)
    want = np.zeros(0x110000, dtype=np.uint8)
    for cp in range(0x110000):
        cat = "Cs" if 0xD800 <= cp <= 0xDFFF else unicodedata.category(chr(cp))
        want[cp] = (1 if cat[0] == "L" else 0) | (2 if cat[0] == "N" else 0) | (4 if cp in WHITE_SPACE else 0)
    bad = np.nonzero(got != want)[0] if tool_matches_file else np.zeros(0, dtype=np.int64)
    note = "" if tool_matches_file else "; comparison with unicodedata skipped: the tool does not match the table file"
    if tool_matches_file and unicodedata.unidata_version != meta["unicode_version"] and bad.size:
        unassigned = np.zeros(0x110000, dtype=bool)
        for lo, hi in meta["unassigned"]:
            unassigned[lo:hi + 1] = True
        newer = unassigned[bad] & (got[bad] == 0)        # did not exist in the tables' version
        bad = bad[~newer]
        note = (f"; Python has Unicode {unicodedata.unidata_version}, the tables {meta['unicode_version']}: "
                f"{int(newer.sum())} newly assigned code points excused, {bad.size} reclassified or differing")
        if bad.size <= args.max_reclassified:
            bad = bad[:0]       # tolerated: cannot be told apart from an official reclassification
    if bad.size:
        failures.append(f"{bad.size} code points are classified differently from unicodedata, e.g. "
                        + ", ".join(f"U+{int(c):04X} (C++ {int(got[c])}, Python {int(want[c])})" for c in bad[:5]))
    n_letter, n_number, n_space = int((got & 1).sum()), int(((got >> 1) & 1).sum()), int(((got >> 2) & 1).sum())

    # ---- 4. decoding valid text --------------------------------------------------------------------
    all_cps = [cp for cp in range(0x110000) if not 0xD800 <= cp <= 0xDFFF]
    text = "".join(map(chr, all_cps))
    decoded = np.frombuffer(run(args.exe, "decode", text.encode("utf-8")), dtype="<u4")
    if not np.array_equal(decoded, np.array(all_cps, dtype="<u4")):
        failures.append("decoding every valid code point did not give them back")

    # ---- 5. malformed input ---------------------------------------------------------------------------
    rng = random.Random(args.seed)
    cases = [b"\x80", b"\xbf", b"\xc0\x80", b"\xc1\xbf", b"\xe0\x80\x80", b"\xed\xa0\x80", b"\xf4\x90\x80\x80",
             b"\xf8\x88\x80\x80\x80", b"\xc3", b"\xe2\x82", b"\xf0\x9f\x99", b"a\xffb", b"\xff\xfe", b"\xe2\x82\xac\xe2",
             "caf\u00e9".encode()[:-1]]
    for _ in range(300):
        cases.append(bytes(rng.randrange(256) for _ in range(rng.randint(1, 30))))
    for blob in cases:
        try:
            out = np.frombuffer(run(args.exe, "decode", blob), dtype="<u4")
        except RuntimeError as e:
            failures.append(f"decoding {blob!r} crashed: {e}")
            continue
        if out.size == 0 or out.size > len(blob):
            failures.append(f"decoding {blob!r} produced {out.size} code points")
        if np.any(out > 0x10FFFF) or np.any((out >= 0xD800) & (out <= 0xDFFF)):
            failures.append(f"decoding {blob!r} produced an invalid code point")
        try:  # input that IS valid UTF-8 must decode exactly like Python
            expected = [ord(c) for c in blob.decode("utf-8")]
            if list(out) != expected:
                failures.append(f"valid text {blob!r} decoded differently from Python")
        except UnicodeDecodeError:
            if 0xFFFD not in out:
                failures.append(f"malformed input {blob!r} produced no U+FFFD")
    for blob, expected in ((b"\xc0\x80", [0xFFFD, 0xFFFD]), (b"\xed\xa0\x80", [0xFFFD] * 3), (b"a\xffb", [0x61, 0xFFFD, 0x62]),
                           (b"\xe2\x82", [0xFFFD, 0xFFFD])):
        out = list(np.frombuffer(run(args.exe, "decode", blob), dtype="<u4"))
        if out != expected:
            failures.append(f"decoding {blob!r}: got {[hex(c) for c in out]}, expected {[hex(c) for c in expected]}")

    # ---- 6. encoding ---------------------------------------------------------------------------------------
    enc = run(args.exe, "encode", np.array(all_cps, dtype="<u4").tobytes())
    if enc != text.encode("utf-8"):
        failures.append("encoding every valid code point differs from Python's UTF-8")
    invalid = np.array([0xD800, 0xDFFF, 0x110000, 0xFFFFFFFF, 0x41], dtype="<u4")
    if run(args.exe, "encode", invalid.tobytes()) != b"\xef\xbf\xbd" * 4 + b"A":
        failures.append("invalid code points were not encoded as U+FFFD")

    if failures:
        print("FAIL")
        for f in failures[:10]:
            print("  -", f)
        return 1
    print(f"PASS  (1114112 code points: lookup equals the table file, table checksum ok; {n_letter} letters, "
          f"{n_number} numbers, {n_space} spaces{note}; UTF-8 round trips and {len(cases)} malformed inputs handled)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
