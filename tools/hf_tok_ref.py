"""hf_tok_ref.py - a pure Python reference for byte-level BPE tokenizers in the Hugging Face tokenizer.json format.

It exists to check the C++ tokenizer, so it is written differently on purpose:
  * pre-tokenization uses Python's real regex engine with the GPT-2 pattern; \\p{L}, \\p{N} and \\s are expanded into
    explicit character classes computed from unicodedata (the C++ code has a hand-written scanner instead)
  * BPE uses the original GPT-2 algorithm (find the best pair, merge all its occurrences) while C++ uses a
    linked list with a priority queue
It supports the same configurations as the C++ code: ByteLevel, or Digits followed by ByteLevel.
"""
from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

WHITE_SPACE = (list(range(0x09, 0x0E)) + [0x20, 0x85, 0xA0, 0x1680, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000]
               + list(range(0x2000, 0x200B)))


def bytes_to_unicode() -> dict[int, str]:
    """GPT-2's mapping of the 256 byte values to printable characters."""
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


BYTE_TO_CHAR = bytes_to_unicode()
CHAR_TO_BYTE = {c: b for b, c in BYTE_TO_CHAR.items()}


def _class_body(pred) -> str:
    """The inside of a [...] character class matching exactly the code points for which pred is true."""
    parts, start = [], None
    for cp in range(0x110001):
        inside = cp < 0x110000 and pred(cp)
        if inside and start is None:
            start = cp
        elif not inside and start is not None:
            parts.append(f"\\U{start:08x}" if start == cp - 1 else f"\\U{start:08x}-\\U{cp - 1:08x}")
            start = None
    return "".join(parts)


@lru_cache(maxsize=1)
def _gpt2_pattern() -> re.Pattern:
    def cat(prefix):
        return lambda cp: not 0xD800 <= cp <= 0xDFFF and unicodedata.category(chr(cp)).startswith(prefix)

    letters, numbers = _class_body(cat("L")), _class_body(cat("N"))
    space = "".join(f"\\U{cp:08x}" for cp in WHITE_SPACE)
    return re.compile(
        rf"'s|'t|'re|'ve|'m|'ll|'d"
        rf"| ?[{letters}]+"
        rf"| ?[{numbers}]+"
        rf"| ?[^{space}{letters}{numbers}]+"
        rf"|[{space}]+(?![^{space}])"
        rf"|[{space}]+"
    )


def gpt2_pieces(text: str) -> list[str]:
    pieces = _gpt2_pattern().findall(text)
    assert "".join(pieces) == text, "the GPT-2 pattern must cover the whole text"
    return pieces


# The Qwen2 / Qwen3 pre-tokenizer pattern exactly as it appears in their tokenizer.json (the C++ code must accept
# this string and no other for a Split stage).
QWEN2_PATTERN = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*"
                 r"|\s*[\r\n]+|\s+(?!\S)|\s+")


# The Llama 3 pattern: the same as Qwen2's except that a run of up to three digits is one piece.
LLAMA3_PATTERN = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*"
                  r"|\s*[\r\n]+|\s+(?!\S)|\s+")
SPLIT_PATTERNS = {QWEN2_PATTERN: 1, LLAMA3_PATTERN: 3}   # pattern string -> longest digit run in one piece


@lru_cache(maxsize=2)
def _split_pattern(max_digits: int) -> re.Pattern:
    """QWEN2_PATTERN with \\p{L}, \\p{N} and \\s written out as explicit character classes, for Python's regex engine
    (backtracking, leftmost alternative first, like the Oniguruma engine of the Hugging Face library)."""
    def cat(prefix):
        return lambda cp: not 0xD800 <= cp <= 0xDFFF and unicodedata.category(chr(cp)).startswith(prefix)

    letters, numbers = _class_body(cat("L")), _class_body(cat("N"))
    space = "".join(f"\\U{cp:08x}" for cp in WHITE_SPACE)
    return re.compile(
        rf"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
        rf"|[^\r\n{letters}{numbers}]?[{letters}]+"
        rf"|[{numbers}]{{1,{max_digits}}}"
        rf"| ?[^{space}{letters}{numbers}]+[\r\n]*"
        rf"|[{space}]*[\r\n]+"
        rf"|[{space}]+(?![^{space}])"
        rf"|[{space}]+"
    )


def split_pieces(text: str, max_digits: int) -> list[str]:
    pieces = _split_pattern(max_digits).findall(text)
    assert "".join(pieces) == text, "the split pattern must cover the whole text"
    return pieces


def qwen2_pieces(text: str) -> list[str]:
    return split_pieces(text, 1)


def llama3_pieces(text: str) -> list[str]:
    return split_pieces(text, 3)


def split_digits(text: str, individual: bool) -> list[str]:
    out: list[str] = []
    run = ""
    run_digits = False
    for ch in text:
        digit = unicodedata.category(ch).startswith("N")
        if digit and individual:
            if run:
                out.append(run)
                run = ""
            out.append(ch)
            continue
        if run and digit != run_digits:
            out.append(run)
            run = ""
        run += ch
        run_digits = digit
    if run:
        out.append(run)
    return out


def to_byte_level(piece: str) -> str:
    return "".join(BYTE_TO_CHAR[b] for b in piece.encode("utf-8"))


class RefTokenizer:
    def __init__(self, directory: str | Path):
        directory = Path(directory)
        tj = json.loads((directory / "tokenizer.json").read_text(encoding="utf-8"))
        model = tj["model"]
        assert model["type"] == "BPE"
        self.vocab: dict[str, int] = dict(model["vocab"])
        self.id_to_token = {i: t for t, i in self.vocab.items()}
        self.special_ids: set[int] = set()
        self.added: dict[str, int] = {}
        for a in tj.get("added_tokens", []):
            self.added[a["content"]] = a["id"]
            self.id_to_token.setdefault(a["id"], a["content"])
            self.vocab.setdefault(a["content"], a["id"])
            if a.get("special"):
                self.special_ids.add(a["id"])
        self.ignore_merges = bool(model.get("ignore_merges"))
        merges = [tuple(m.split(" ")) if isinstance(m, str) else tuple(m) for m in model["merges"]]
        self.ranks: dict[tuple[str, str], int] = {}
        for r, pair in enumerate(merges):
            self.ranks.setdefault(pair, r)          # the first (best) rank of a pair wins

        pre = tj["pre_tokenizer"]
        stages = pre["pretokenizers"] if pre["type"] == "Sequence" else [pre]
        self.split_digits, self.individual = False, False
        self.use_regex = True
        self.split_max_digits = 0      # 0: no Split stage; else the longest digit run of one piece (1 Qwen2, 3 Llama 3)
        norm = tj.get("normalizer")
        self.nfc = bool(norm) and norm["type"] == "NFC"
        for st in stages:
            if st["type"] == "Split":
                assert st["pattern"]["Regex"] in SPLIT_PATTERNS and st["behavior"] == "Isolated"
                self.split_max_digits = SPLIT_PATTERNS[st["pattern"]["Regex"]]
            elif st["type"] == "Digits":
                self.split_digits, self.individual = True, bool(st.get("individual_digits"))
            elif st["type"] == "ByteLevel":
                self.use_regex = st.get("use_regex", True)
        cfg_path = directory / "config.json"
        self.bos = None
        if cfg_path.exists():
            self.bos = json.loads(cfg_path.read_text(encoding="utf-8")).get("bos_token_id")
        self._added_re = None
        if self.added:
            alternatives = "|".join(re.escape(t) for t in sorted(self.added, key=len, reverse=True))
            self._added_re = re.compile(f"({alternatives})")

    # ---- encoding ----------------------------------------------------------------------------------
    def pre_tokenize(self, text: str) -> list[str]:
        """Pieces of the NORMALIZED text (what the encoder sees)."""
        if self.nfc:
            text = unicodedata.normalize("NFC", text)
        if self.split_max_digits:
            return split_pieces(text, self.split_max_digits) if text else []
        stage = split_digits(text, self.individual) if self.split_digits else ([text] if text else [])
        if not self.use_regex:
            return stage
        out: list[str] = []
        for piece in stage:
            out.extend(gpt2_pieces(piece))
        return out

    def normalized_text(self, text: str) -> str:
        """The text as the encoder sees it: the stretches between added tokens are normalized, added tokens are not."""
        if not self.nfc:
            return text
        if self._added_re is None:
            return unicodedata.normalize("NFC", text)
        return "".join(p if i % 2 else unicodedata.normalize("NFC", p) for i, p in enumerate(self._added_re.split(text)))

    def _bpe(self, mapped: str) -> list[str]:
        """The original GPT-2 merge loop: best pair first, all occurrences of it merged left to right."""
        word = tuple(mapped)
        while len(word) > 1:
            pairs = set(zip(word, word[1:]))
            best = min(pairs, key=lambda p: self.ranks.get(p, float("inf")))
            if best not in self.ranks:
                break
            first, second = best
            new_word, i = [], 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
        return list(word)

    def _encode_plain(self, text: str) -> list[int]:
        ids: list[int] = []
        for piece in self.pre_tokenize(text):   # pre_tokenize normalizes first
            mapped = to_byte_level(piece)
            if self.ignore_merges and mapped in self.vocab:   # a whole pre-token that is a token: take it as is
                ids.append(self.vocab[mapped])
            else:
                ids.extend(self.vocab[tok] for tok in self._bpe(mapped))
        return ids

    def encode(self, text: str, add_bos: bool = False) -> list[int]:
        ids = [self.bos] if add_bos else []
        if self._added_re is None:
            return ids + self._encode_plain(text)
        for i, part in enumerate(self._added_re.split(text)):
            if i % 2 == 1:
                ids.append(self.added[part])
            elif part:
                ids.extend(self._encode_plain(part))   # _encode_plain normalizes
        return ids

    # ---- decoding ------------------------------------------------------------------------------------
    def decode(self, ids: list[int], skip_special: bool = False) -> bytes:
        out = bytearray()
        for i in ids:
            if skip_special and i in self.special_ids:
                continue
            token = self.id_to_token[i]
            if all(c in CHAR_TO_BYTE for c in token):
                out.extend(CHAR_TO_BYTE[c] for c in token)
            else:
                out.extend(token.encode("utf-8"))       # an added token written with ordinary characters
        return bytes(out)
