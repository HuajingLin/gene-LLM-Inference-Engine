#!/usr/bin/env python3
"""verify_json.py - M7 acceptance for the C++ JSON parser (gi_json_check).

  1. valid documents: parse + dump with C++, load the result with Python's json module and
     require it to equal Python's own parse of the original text (exact float equality)
  2. seeded random documents (nesting, unicode, escapes, odd numbers, every whitespace style)
  3. a real-world shaped config.json and a safetensors-style header
  4. invalid documents must be rejected (exit code 1), including a depth bomb
  5. leniencies that are intentional: a UTF-8 BOM is skipped, duplicate keys keep the last value

Usage:
  python tests/verify_json.py --exe build/gi_json_check
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys

VALID = [
    "0", "-0", "1", "-1", "123456789012", "1.5", "-2.25", "1e5", "1E5", "1e+5", "1e-5", "2.5e-10",
    "1.7976931348623157e308", "5e-324", "0.1", "0.30000000000000004", "true", "false", "null",
    '""', '"a"', '"\\u00e9"', '"\\ud83d\\ude42"', '"\\"\\\\\\/\\b\\f\\n\\r\\t"', '"\\u0000"',
    "[]", "{}", "[[]]", "[{}]", '{"a":{}}', "[1,2,3]", '{"a":1,"b":2}', ' \t\r\n[ 1 , 2 ] \n',
    '{"k":"v","k":"w"}',                       # duplicate key: last wins
    '{"":1}', '{"a b":[null,true,false]}',
    "[" * 150 + "]" * 150,                       # deep, but under the limit
    '{"unicode":"caf\u00e9 \u65e5\u672c \U0001F642"}',   # raw UTF-8 in the text
]

CONFIG_LIKE = """{
  "architectures": ["LlamaForCausalLM"],
  "attention_bias": false, "bos_token_id": 0, "eos_token_id": 0, "head_dim": 64,
  "hidden_act": "silu", "hidden_size": 576, "initializer_range": 0.041666666666666664,
  "intermediate_size": 1536, "max_position_embeddings": 8192, "model_type": "llama",
  "num_attention_heads": 9, "num_hidden_layers": 30, "num_key_value_heads": 3,
  "rms_norm_eps": 1e-05, "rope_scaling": null, "rope_theta": 100000,
  "tie_word_embeddings": true, "torch_dtype": "bfloat16", "vocab_size": 49152
}"""

HEADER_LIKE = ('{"__metadata__":{"format":"pt"},"model.embed_tokens.weight":{"dtype":"BF16",'
               '"shape":[49152,576],"data_offsets":[0,56623104]},"model.norm.weight":'
               '{"dtype":"BF16","shape":[576],"data_offsets":[56623104,56624256]}}       ')

INVALID = [
    "", "   ", "{", "}", "[", "]", "[1,]", "[,1]", '{"a":1,}', '{"a":}', '{"a" 1}', '{a:1}', "[1 2]",
    "01", "00", "1.", ".5", "+1", "-", "--1", "1e", "1e+", "0x10", "NaN", "Infinity", "-Infinity",
    "tru", "nul", "True", "'a'", '"abc', '"\\x"', '"\\u12"', '"\\u12G4"', '"\\ud800"', '"\\udc00"',
    '"\\ud800\\u0041"', '"a\x01b"', '"a\nb"', '{"a":1} x', "1 2", "[1]]", "1e999", "-1e999",
    "[" * 1000, "[" * 1000 + "]" * 1000, '{"a":' * 300 + "1" + "}" * 300, "\x00", "[1,2",
]


def run(exe: str, data: bytes):
    return subprocess.run([exe], input=data, capture_output=True)


def random_value(rng: random.Random, depth: int):
    kind = rng.randrange(8 if depth < 5 else 6)
    if kind == 0:
        return None
    if kind == 1:
        return rng.random() < 0.5
    if kind == 2:
        return rng.randint(-10**12, 10**12)
    if kind == 3:
        return rng.choice([rng.uniform(-1e6, 1e6), rng.uniform(-1, 1) * 10 ** rng.randint(-30, 30),
                           float(rng.randint(-1000, 1000)), 5e-324, 1.7976931348623157e308])
    if kind in (4, 5):
        alphabet = "abcXYZ019 _-\"\\/\n\t\u00e9\u00fc\u65e5\u672c\U0001F642\x01\x1f"
        return "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 12)))
    if kind == 6:
        return [random_value(rng, depth + 1) for _ in range(rng.randint(0, 5))]
    key_alphabet = "abcd\u00e9_ \""
    return {"".join(rng.choice(key_alphabet) for _ in range(rng.randint(0, 6))): random_value(rng, depth + 1)
            for _ in range(rng.randint(0, 5))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True, help="path to gi_json_check")
    ap.add_argument("--fuzz", type=int, default=300)
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()
    failures: list[str] = []
    n_valid = 0

    def check_valid(text: str, label: str):
        nonlocal n_valid
        n_valid += 1
        p = run(args.exe, text.encode("utf-8"))
        if p.returncode != 0:
            failures.append(f"{label}: valid JSON rejected: {p.stderr.decode(errors='replace').strip()}")
            return
        try:
            got = json.loads(p.stdout.decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            failures.append(f"{label}: C++ output is not valid JSON ({e}): {p.stdout[:80]!r}")
            return
        if got != json.loads(text):
            failures.append(f"{label}: value differs from Python's parse: {text[:60]!r}")

    for text in VALID:
        check_valid(text, f"valid {text[:30]!r}")
    check_valid(CONFIG_LIKE, "config.json-like")
    check_valid(HEADER_LIKE, "safetensors-header-like")

    rng = random.Random(args.seed)
    for i in range(args.fuzz):
        value = random_value(rng, 0)
        style = i % 4
        text = (json.dumps(value, ensure_ascii=bool(style & 1)) if style < 2 else
                json.dumps(value, indent=rng.choice([1, 2, 4]), ensure_ascii=bool(style & 1)))
        check_valid(text, f"fuzz #{i}")

    for text in INVALID:
        if run(args.exe, text.encode("utf-8", errors="surrogatepass")).returncode == 0:
            failures.append(f"invalid JSON accepted: {text[:40]!r}")

    p = run(args.exe, b"\xef\xbb\xbf" + b'{"a":1}')
    if p.returncode != 0 or json.loads(p.stdout) != {"a": 1}:
        failures.append("a UTF-8 byte-order mark was not skipped")

    if failures:
        print("FAIL")
        for f in failures[:15]:
            print("  -", f)
        if len(failures) > 15:
            print(f"  ... and {len(failures) - 15} more")
        return 1
    print(f"PASS  ({n_valid} valid documents round-tripped, {len(INVALID)} invalid ones rejected)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
