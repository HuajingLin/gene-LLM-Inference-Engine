// checkpoint.h — Load model weights in llama2.c format (v0, float32)
//
//File layout (little endian):
// header: 7 int32
// dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len
// (vocab_size is a negative number, indicating that lm_head is not shared with embedding, and the actual vocabulary size takes the absolute value)
// body: continuous float32 tensor, fixed order (see load in checkpoint.cpp)
//
// All weight matrices are row-major, of shape [out_features, in_features], i.e. y = W · x.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "mapped_file.h"

namespace geneinfer {

struct Config {
    int dim = 0;          // Hidden dimension (d_model)
    int hidden_dim = 0;   // FFN intermediate dimension
    int n_layers = 0;
    int n_heads = 0;      // query head count
    int n_kv_heads = 0;   // key/value head count (GQA: <= n_heads)
    int vocab_size = 0;
    int seq_len = 0;      // Maximum context length
    bool shared_classifier = true;  // lm_head Whether to reuse token_embedding

    int head_size() const { return dim / n_heads; }
    int kv_dim() const { return n_kv_heads * head_size(); }
    int kv_mul() const { return n_heads / n_kv_heads; }  // Each KV head is shared by how many Q heads
};

// All pointers directly point to the mmap region, read-only, no copy.
// Tensors with [L, ...], the data of layer l = ptr + l * elements per layer.
struct Weights {
    const float* token_embedding = nullptr;  // [vocab, dim]
    const float* rms_att = nullptr;          // [L, dim]
    const float* wq = nullptr;               // [L, dim, dim]
    const float* wk = nullptr;               // [L, kv_dim, dim]
    const float* wv = nullptr;               // [L, kv_dim, dim]
    const float* wo = nullptr;               // [L, dim, dim]
    const float* rms_ffn = nullptr;          // [L, dim]
    const float* w1 = nullptr;               // [L, hidden, dim]   gate projection
    const float* w2 = nullptr;               // [L, dim, hidden]   down projection
    const float* w3 = nullptr;               // [L, hidden, dim]   up projection
    const float* rms_final = nullptr;        // [dim]
    const float* wcls = nullptr;             // [vocab, dim]（shared 时 == token_embedding）
};

// For debugging/verification: list each tensor in the order of the files.
struct TensorView {
    std::string name;
    const float* data = nullptr;
    std::size_t numel = 0;
    std::size_t byte_offset = 0;  // Relative to the start of the file
    std::vector<std::int64_t> shape;
};

class Checkpoint {
public:
    // Failure (file not found / header invalid / size mismatch) throws std::runtime_error.
    static Checkpoint load(const std::string& path);

    const Config& config() const { return cfg_; }
    const Weights& weights() const { return w_; }
    const std::vector<TensorView>& tensors() const { return tensors_; }
    std::size_t file_size() const { return file_.size(); }

private:
    Checkpoint() = default;

    MappedFile file_;  // Must ensure it lives longer than w_ / tensors_
    Config cfg_;
    Weights w_;
    std::vector<TensorView> tensors_;
};

}  // namespace geneinfer
