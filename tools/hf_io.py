"""hf_io.py - Hugging Face model directories in pure Python/NumPy (no safetensors package needed).

Reading and writing the safetensors format, converting bf16/f16 to float32, and describing which
tensors a Llama model directory has. This is the independent "answer key" for the C++ loader.
"""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DTYPE_NAMES = {"F32": np.dtype("<f4"), "F16": np.dtype("<f2"), "BF16": np.dtype("<u2")}


# --------------------------------------------------------------------------- conversions
def bf16_to_f32(raw: np.ndarray) -> np.ndarray:
    """bfloat16 bit patterns (uint16) -> float32: a 16-bit shift."""
    return (raw.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(x: np.ndarray) -> np.ndarray:
    """float32 -> bfloat16 bit patterns with round-to-nearest-even (NaNs stay NaN)."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    rounded = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    nan = (u & 0x7FFFFFFF) > 0x7F800000
    return np.where(nan, (u >> 16) | 0x40, rounded).astype(np.uint16)


# --------------------------------------------------------------------------- reading
@dataclass
class StTensor:
    name: str
    dtype_name: str
    shape: tuple[int, ...]
    begin: int          # absolute byte offset of the data in the file
    end: int
    raw: np.ndarray     # the stored bytes interpreted with the stored dtype

    def as_float32(self) -> np.ndarray:
        if self.dtype_name == "F32":
            return self.raw.astype(np.float32).reshape(self.shape)
        if self.dtype_name == "F16":
            return self.raw.astype(np.float32).reshape(self.shape)
        if self.dtype_name == "BF16":
            return bf16_to_f32(self.raw).reshape(self.shape)
        raise ValueError(f"unsupported dtype {self.dtype_name}")


def read_safetensors(path: str | Path) -> dict[str, StTensor]:
    path = Path(path)
    blob = np.memmap(path, dtype=np.uint8, mode="r")
    (header_size,) = struct.unpack("<Q", bytes(blob[:8]))
    header = json.loads(bytes(blob[8:8 + header_size]).decode("utf-8"))
    data_start = 8 + header_size
    out: dict[str, StTensor] = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        b, e = info["data_offsets"]
        dt = DTYPE_NAMES.get(info["dtype"])
        raw = np.frombuffer(blob[data_start + b:data_start + e], dtype=dt) if dt is not None else np.zeros(0)
        out[name] = StTensor(name, info["dtype"], tuple(info["shape"]), data_start + b, data_start + e, raw)
    return out


# --------------------------------------------------------------------------- writing
def write_safetensors(path: str | Path, tensors: list[tuple[str, str, tuple[int, ...], bytes]],
                      metadata: dict | None = None, header_extra_pad: int = 0) -> None:
    """tensors: list of (name, dtype_name, shape, raw bytes) in the order they are stored."""
    header: dict = {}
    if metadata is not None:
        header["__metadata__"] = metadata
    offset = 0
    for name, dtype_name, shape, raw in tensors:
        header[name] = {"dtype": dtype_name, "shape": list(shape), "data_offsets": [offset, offset + len(raw)]}
        offset += len(raw)
    text = json.dumps(header, separators=(",", ":")).encode("utf-8")
    text += b" " * ((8 - len(text) % 8) % 8 + header_extra_pad)   # the spec pads the header to 8 bytes
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(text)))
        f.write(text)
        for _, _, _, raw in tensors:
            f.write(raw)


# --------------------------------------------------------------------------- Llama layout
def read_hf_config(directory: str | Path) -> dict:
    return json.loads((Path(directory) / "config.json").read_text(encoding="utf-8"))


def derived_config(cfg: dict) -> dict:
    """What the C++ loader is expected to derive from config.json (defaults included)."""
    dim, heads = cfg["hidden_size"], cfg["num_attention_heads"]
    kv_heads = cfg.get("num_key_value_heads") or heads
    head_dim = cfg.get("head_dim") or dim // heads
    return {
        "dim": dim, "hidden_dim": cfg["intermediate_size"], "n_layers": cfg["num_hidden_layers"],
        "n_heads": heads, "n_kv_heads": kv_heads, "vocab_size": cfg["vocab_size"],
        "seq_len": cfg.get("max_position_embeddings", 2048), "head_dim": head_dim,
        "attn_dim": heads * head_dim, "kv_dim": kv_heads * head_dim,
        "rope_theta": np.float32(cfg.get("rope_theta", 10000.0)),
        "rms_norm_eps": np.float32(cfg.get("rms_norm_eps", 1e-6)),
        "rope_style": "half_split", "shared_classifier": bool(cfg.get("tie_word_embeddings", False)),
    }


def llama_tensor_layout(cfg: dict) -> list[tuple[str, tuple[int, ...]]]:
    """Tensor names and shapes in the order the C++ loader reads them."""
    d = derived_config(cfg)
    D, H, V, A, KV = d["dim"], d["hidden_dim"], d["vocab_size"], d["attn_dim"], d["kv_dim"]
    layout = [("model.embed_tokens.weight", (V, D))]
    for l in range(d["n_layers"]):
        p = f"model.layers.{l}."
        layout += [(p + "input_layernorm.weight", (D,)), (p + "self_attn.q_proj.weight", (A, D)),
                   (p + "self_attn.k_proj.weight", (KV, D)), (p + "self_attn.v_proj.weight", (KV, D)),
                   (p + "self_attn.o_proj.weight", (D, A)), (p + "post_attention_layernorm.weight", (D,)),
                   (p + "mlp.gate_proj.weight", (H, D)), (p + "mlp.up_proj.weight", (H, D)),
                   (p + "mlp.down_proj.weight", (D, H))]
    layout.append(("model.norm.weight", (D,)))
    if not d["shared_classifier"]:
        layout.append(("lm_head.weight", (V, D)))
    return layout
