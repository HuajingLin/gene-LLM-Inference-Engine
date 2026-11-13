"""llama2c_io.py - readers for llama2.c checkpoints and tokenizers (NumPy version).

This is the "answer key" for the C++ implementation: layout, BPE encoding and decoding all follow run.c in karpathy/llama2.c.
"""
from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HEADER_FMT = "<7i"
HEADER_BYTES = struct.calcsize(HEADER_FMT)  # 28


# --------------------------------------------------------------------------- config
@dataclass
class Config:
    dim: int
    hidden_dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    vocab_size: int
    seq_len: int
    shared_classifier: bool
    # Settings that llama2.c does not store (its fixed conventions are the defaults):
    head_dim: int = 0                    # 0 = dim // n_heads
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    rope_style: str = "interleaved"      # or "half_split" (Hugging Face)
    rope_scaling: dict | None = None     # config.json "rope_scaling" of type llama3, or None
    qk_norm: bool = False                # per-head RMSNorm of q and k before RoPE (Qwen3)

    @property
    def head_size(self) -> int:
        return self.head_dim or self.dim // self.n_heads

    @property
    def attn_dim(self) -> int:
        return self.n_heads * self.head_size

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_size

    @property
    def kv_mul(self) -> int:
        return self.n_heads // self.n_kv_heads


def tensor_layout(c: Config) -> list[tuple[str, tuple[int, ...]]]:
    """List all tensors in file order with their shapes. Weight matrices are row-major [out, in]."""
    L, D, H, V = c.n_layers, c.dim, c.hidden_dim, c.vocab_size
    KV, HS, S = c.kv_dim, c.head_size, c.seq_len
    layout = [
        ("token_embedding", (V, D)),
        ("rms_att", (L, D)),
        ("wq", (L, D, D)),
        ("wk", (L, KV, D)),
        ("wv", (L, KV, D)),
        ("wo", (L, D, D)),
        ("rms_ffn", (L, D)),
        ("w1", (L, H, D)),
        ("w2", (L, D, H)),
        ("w3", (L, H, D)),
        ("rms_final", (D,)),
        ("freq_cis_real", (S, HS // 2)),
        ("freq_cis_imag", (S, HS // 2)),
    ]
    if not c.shared_classifier:
        layout.append(("wcls", (V, D)))
    return layout


# --------------------------------------------------------------------------- checkpoint
@dataclass
class Checkpoint:
    config: Config
    tensors: dict[str, np.ndarray]   # name -> float32 view (memmap, no extra memory)
    offsets: dict[str, int]          # name -> byte offset within the file
    file_size: int


def read_checkpoint(path: str | Path) -> Checkpoint:
    path = Path(path)
    file_size = path.stat().st_size
    with open(path, "rb") as f:
        raw = f.read(HEADER_BYTES)
    dim, hidden, layers, heads, kv_heads, vocab, seq_len = struct.unpack(HEADER_FMT, raw)
    cfg = Config(dim, hidden, layers, heads, kv_heads, abs(vocab), seq_len, vocab > 0)

    layout = tensor_layout(cfg)
    expected = HEADER_BYTES + 4 * sum(int(np.prod(s)) for _, s in layout)
    if expected != file_size:
        raise ValueError(f"size mismatch: header implies {expected} bytes, file has {file_size}")

    flat = np.memmap(path, dtype="<f4", mode="r", offset=HEADER_BYTES)
    tensors: dict[str, np.ndarray] = {}
    offsets: dict[str, int] = {}
    pos = 0
    for name, shape in layout:
        n = int(np.prod(shape))
        tensors[name] = flat[pos:pos + n].reshape(shape)
        offsets[name] = HEADER_BYTES + 4 * pos
        pos += n
    if cfg.shared_classifier:
        tensors["wcls"] = tensors["token_embedding"]
    return Checkpoint(cfg, tensors, offsets, file_size)


# --------------------------------------------------------------------------- tokenizer
class Tokenizer:
    """The llama2.c tokenizer.bin format:
        int32 max_token_length
        repeated vocab_size times: float32 score, int32 len, then len bytes of token text
    Special ids: 0=<unk>  1=<s>(BOS)  2=</s>(EOS); byte-fallback token id = byte value + 3.
    """

    BOS, EOS = 1, 2
    _BYTE_RE = re.compile(rb"<0x([0-9A-Fa-f]{2})>")

    def __init__(self, path: str | Path, vocab_size: int):
        self.vocab: list[bytes] = []
        self.scores: list[float] = []
        with open(path, "rb") as f:
            (self.max_token_length,) = struct.unpack("<i", f.read(4))
            for _ in range(vocab_size):
                (score,) = struct.unpack("<f", f.read(4))
                (n,) = struct.unpack("<i", f.read(4))
                self.vocab.append(f.read(n))
                self.scores.append(score)
        self.lookup: dict[bytes, int] = {}
        for i, tok in enumerate(self.vocab):
            self.lookup.setdefault(tok, i)

    def encode(self, text: str, bos: bool = True, eos: bool = False) -> list[int]:
        tokens: list[int] = []
        if bos:
            tokens.append(self.BOS)
        if text:
            # sentencepiece prepends a "dummy prefix" space to the text by default
            tokens.append(self.lookup[b" "])

        # 1) split into single Unicode characters; fall back to bytes (id = byte + 3) if the vocab lacks one
        for ch in text:
            b = ch.encode("utf-8")
            if b in self.lookup:
                tokens.append(self.lookup[b])
            else:
                tokens.extend(byte + 3 for byte in b)

        # 2) BPE: repeatedly merge the adjacent pair whose concatenation is in the vocab with the highest score
        while True:
            best_score, best_id, best_idx = -1e10, -1, -1
            for i in range(len(tokens) - 1):
                idx = self.lookup.get(self.vocab[tokens[i]] + self.vocab[tokens[i + 1]])
                if idx is not None and self.scores[idx] > best_score:
                    best_score, best_id, best_idx = self.scores[idx], idx, i
            if best_idx == -1:
                break
            tokens[best_idx] = best_id
            del tokens[best_idx + 1]

        if eos:
            tokens.append(self.EOS)
        return tokens

    def decode_piece(self, prev: int, tok: int) -> bytes:
        piece = self.vocab[tok]
        if prev == self.BOS and piece.startswith(b" "):
            piece = piece[1:]  # drop the leading space of the first token after BOS
        m = self._BYTE_RE.fullmatch(piece)
        if m:  # byte-fallback token, e.g. <0x0A>
            piece = bytes([int(m.group(1), 16)])
        return piece
