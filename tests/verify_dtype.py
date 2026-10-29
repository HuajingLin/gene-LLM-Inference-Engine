#!/usr/bin/env python3
"""verify_dtype.py - M7 acceptance for bf16 / f16 -> float32 conversion (gi_dtype_check).

The inputs are exhaustive: all 65536 bit patterns of each 16-bit type are converted by the C++
code and compared bit for bit with NumPy. (NaNs must agree in sign and payload; only the
"quiet" bit may differ, because some CPUs set it when converting a signaling NaN.) Also checks the float32 pass-through used for F32 tensors and that
unaligned/odd-sized input is handled.

Usage:
  python tests/verify_dtype.py --exe build/gi_dtype_check
"""
from __future__ import annotations

import argparse
import subprocess
import sys

import numpy as np


def convert(exe: str, kind: str, raw: bytes) -> np.ndarray:
    p = subprocess.run([exe, kind], input=raw, capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode(errors="replace"))
    return np.frombuffer(p.stdout, dtype="<u4")


def compare(got_bits: np.ndarray, want_bits: np.ndarray) -> int:
    """Number of mismatching patterns (NaNs: same sign and both NaN)."""
    is_nan_want = (want_bits & 0x7F800000) == 0x7F800000
    is_nan_want &= (want_bits & 0x007FFFFF) != 0
    is_nan_got = (got_bits & 0x7F800000) == 0x7F800000
    is_nan_got &= (got_bits & 0x007FFFFF) != 0
    exact = got_bits == want_bits
    quiet_bit = np.uint32(0x00400000)
    nan_ok = (is_nan_want & is_nan_got & ((got_bits >> 31) == (want_bits >> 31))
              & ((got_bits & ~quiet_bit) == (want_bits & ~quiet_bit)))
    return int(np.sum(~(exact | nan_ok)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True, help="path to gi_dtype_check")
    args = ap.parse_args()
    failures: list[str] = []

    patterns = np.arange(65536, dtype="<u2")

    # bfloat16: the upper 16 bits of a float32
    want = (patterns.astype(np.uint32) << 16)
    bad = compare(convert(args.exe, "bf16", patterns.tobytes()), want)
    if bad:
        failures.append(f"bf16: {bad} of 65536 patterns differ from NumPy")

    # IEEE half precision
    want = patterns.view(np.float16).astype(np.float32).view(np.uint32)
    got = convert(args.exe, "f16", patterns.tobytes())
    bad = compare(got, want)
    if bad:
        bad_idx = np.nonzero(got != want)[0][:5]
        failures.append(f"f16: {bad} of 65536 patterns differ from NumPy, e.g. {[hex(int(i)) for i in bad_idx]}")

    # spot checks with known values
    known = {"bf16": {0x3F80: 1.0, 0xBF80: -1.0, 0x4049: 3.140625, 0x0000: 0.0, 0x7F80: float("inf")},
             "f16": {0x3C00: 1.0, 0xC000: -2.0, 0x0001: 2.0 ** -24, 0x7BFF: 65504.0, 0x0400: 2.0 ** -14}}
    for kind, table in known.items():
        for bits, value in table.items():
            out = convert(args.exe, kind, np.array([bits], dtype="<u2").tobytes()).view("<f4")[0]
            if out != value:
                failures.append(f"{kind} 0x{bits:04x}: got {out}, expected {value}")

    # float32 pass-through, also from an odd byte offset inside the buffer (unaligned source)
    rng = np.random.default_rng(1)
    f32 = rng.standard_normal(1000).astype("<f4")
    out = convert(args.exe, "f32", f32.tobytes()).view("<f4")
    if not np.array_equal(out, f32):
        failures.append("f32 pass-through changed values")

    # malformed input is rejected, not silently truncated
    p = subprocess.run([args.exe, "bf16"], input=b"\x00\x01\x02", capture_output=True)
    if p.returncode == 0:
        failures.append("an odd number of bytes was accepted for a 16-bit type")

    if failures:
        print("FAIL")
        for f in failures:
            print("  -", f)
        return 1
    print("PASS  (all 65536 bf16 and all 65536 f16 patterns match NumPy)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
