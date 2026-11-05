"""tok_corpus.py - test texts for byte-level BPE tokenizers: hand-picked edge cases plus seeded random ones.

The source file is ASCII only; non-ASCII characters are written as escapes.
"""
from __future__ import annotations

import random

# The last three exercise corner cases of added tokens: one is a prefix of the other (the longest match
# must win), and one contains characters outside the byte-level alphabet (it must decode as plain text).
SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<file_sep>", "<jupyter_code>",
            "<|fill|>", "<|fill|>ed", "<|\u65e5\u672c|>"]


def fixed_cases() -> list[str]:
    cases = [
        "", " ", "  ", "   ", "\n", "\n\n", " \n", "\n ", "\t", " \t ", "\r\n", "a", "A", "Hello", "Hello, world!", " Hello",
        "  Hello", "   Hello", "Hello  world", "Hello world  ", "Hello world \n", "Hello   \n\n  world", "line1\nline2\n\nline4\n",
        "hello", " hello", "hello world", "Hello world", " world", "xyz", "xyzw", "say hello, world",
        "Once upon a time", "Once upon a time, there was a little girl named Lily.",
        "I'm sure it's fine; they've said we'll go, he'd like it", "DON'T STOP", "'s 't 're 've 'm 'll 'd", "x's", "it'sx",
        "'", "''", "'''", "'a", "a'", "'S", "can't won't isn't", "rock 'n' roll",
        "123", "4567890", "3.14159", "a1b2c3", "1,000,000", "2024-10-30", "x=42;y=7", "$100", "100%", "0", "00", "007",
        "foo_bar baz-qux", "snake_case camelCase PascalCase", "http://example.com/a?b=c&d=e#f", "user@example.com",
        "def f(x):\n    return x * 2\n", "for (int i = 0; i < n; ++i) {\n  sum += a[i];\n}\n", "if x == 1: print('hi')",
        ".,;:!?", "...", "!!!???", "(a)[b]{c}", "\"quoted\" 'single' `tick`", "a-b-c", "a--b", "-->", "<=>", "&&||", "\\n \\t",
        "<|endoftext|>", "<|im_start|>user\nHi there!<|im_end|>\n<|im_start|>assistant\n", "a<|im_end|>b<|im_end|><|im_end|>",
        "<|fill|>", "<|fill|>ed", "x<|fill|>edge", "<|fill|><|fill|>ed<|fill|>", "<|fill|", "<|fill|>e",
        "<|\u65e5\u672c|>", "a<|\u65e5\u672c|>b", "<|\u65e5|>",
        "<|im_start", "<|im_start|", "|im_start|>", "<|im_start|><|im_start|>", "text <file_sep> more", "<file_sep>x<jupyter_code>y",
        "<|endoftext|><|endoftext|>", " <|im_end|>", "<|im_end|> ", "<|IM_END|>", "<|im_end|>>",
        "caf\u00e9 na\u00efve fa\u00e7ade", "\u00fcber Stra\u00dfe", "\u65e5\u672c\u8a9e\u306e\u30c6\u30ad\u30b9\u30c8", "\u4e2d\u6587 and English mixed",
        "emoji \U0001F642\U0001F680 test", "\u0627\u0644\u0639\u0631\u0628\u064a\u0629", "\u0928\u092e\u0938\u094d\u0924\u0947",
        "e\u0301 (combining acute)", "non\u00a0breaking\u00a0space", "\u3000ideographic space\u3000", "tab\there",
        "carriage\r\nreturn", "\u2028line separator\u2029", "x\u200by (zero width space)", "\u0661\u0662\u0663 arabic-indic digits",
        "\u00bd \u00b2 \u2155 fractions and superscripts", "\u2167 roman numeral", "\U0001F468\u200d\U0001F469\u200d\U0001F467 family",
        "\u00dcn\u00efc\u00f6d\u00e9 \u00dcn\u00efc\u00f6d\u00e9", "\u201csmart quotes\u201d and \u2018single\u2019", "em\u2014dash and en\u2013dash",
        "\u0e2a\u0e27\u0e31\u0e2a\u0e14\u0e35", "\ud55c\uad6d\uc5b4 \ud14d\uc2a4\ud2b8", "\u03b1\u03b2\u03b3 \u0394\u03a3\u03a9", "\u0420\u0443\u0441\u0441\u043a\u0438\u0439 \u0442\u0435\u043a\u0441\u0442",
        "mixed1234abc5678def", "ab12cd", "1a", "a1", "12 34", "12\n34", " 12", "  12",
        "a" * 50, "a" * 1000, "ab" * 300, " " * 50 + "x", "x" + " " * 50 + "y", "\n" * 5 + "z", "z" + "\n" * 5, "." * 100,
        "the " * 200, "The quick brown fox jumps over the lazy dog. " * 5,
    ]
    return cases


def fuzz_cases(n: int, seed: int = 7) -> list[str]:
    rng = random.Random(seed)
    letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    pieces = (
        [letters[i] for i in range(len(letters))] * 3
        + list("0123456789") * 2
        + [" "] * 14 + ["  "] * 3 + ["\n"] * 3 + ["\t", "\r\n", "\n\n"]
        + list(".,;:!?-_=+*/\\|@#$%^&~()[]{}<>\"`'")
        + ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d", "n't", "the", " the", "ing", "tion", "er", "and", " and"]
        + SPECIALS
        + ["\u00e9", "\u00fc", "\u00df", "\u65e5", "\u672c", "\u8a9e", "\U0001F600", "\u00a0", "\u3000", "\u0661", "\u00bd",
           "\u200b", "\u0301", "\u2014", "\u201c", "\u0416", "\u03b1", "\u2028"]
    )
    cases = []
    for _ in range(n):
        length = rng.choice([0, 1, 2, 3, 5, 8, 13, 21, 34, 55])
        cases.append("".join(rng.choice(pieces) for _ in range(length)))
    return cases


def all_cases(n_fuzz: int = 300, seed: int = 7) -> list[str]:
    seen, out = set(), []
    for text in fixed_cases() + fuzz_cases(n_fuzz, seed):
        if text not in seen:
            seen.add(text)
            out.append(text)
    return out
