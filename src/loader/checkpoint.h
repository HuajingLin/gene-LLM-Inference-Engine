// checkpoint.h - model configuration and weights, loaded from disk.
//
// Two formats are supported; Checkpoint::load() picks one from the path:
//   * a file          -> llama2.c float32 checkpoint (v0), e.g. stories15M.bin
//   * a directory     -> Hugging Face model directory with config.json and model.safetensors
//                        (Llama architecture; weights may be F32, F16 or BF16)
//
// All weight matrices are row-major with shape [out_features, in_features], i.e. y = W * x,
// and are exposed as float32 whatever the file stores. Float32 data in a file is used in place
// (memory-mapped, no copy); F16/BF16 tensors are converted into memory owned by the Checkpoint.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "loader/mapped_file.h"

namespace geneinfer {

class SafetensorsFile;

// How RoPE pairs up the elements of a head:
//   Interleaved: (x[0], x[1]), (x[2], x[3]), ...        llama2.c
//   HalfSplit:   (x[i], x[i + head_dim/2])              Hugging Face Llama ("rotate_half")
enum class RopeStyle { Interleaved, HalfSplit };

struct Config {
    int dim = 0;          // hidden size (d_model)
    int hidden_dim = 0;   // FFN intermediate size
    int n_layers = 0;
    int n_heads = 0;      // number of query heads
    int n_kv_heads = 0;   // number of key/value heads (GQA: <= n_heads)
    int vocab_size = 0;
    int seq_len = 0;      // maximum context length
    bool shared_classifier = true;  // whether lm_head reuses token_embedding

    int head_dim = 0;                              // size of one attention head
    float rope_theta = 10000.0f;                   // RoPE base frequency
    float rms_norm_eps = 1e-5f;                    // RMSNorm epsilon
    RopeStyle rope_style = RopeStyle::Interleaved;

    int head_size() const { return head_dim; }
    int attn_dim() const { return n_heads * head_dim; }  // width of the q projection / attention output
    int kv_dim() const { return n_kv_heads * head_dim; }
    int kv_mul() const { return n_heads / n_kv_heads; }  // query heads sharing one KV head
};

// Weights of one transformer layer. All pointers are read-only.
struct LayerWeights {
    const float* rms_att = nullptr;  // [dim]               input_layernorm
    const float* wq = nullptr;       // [attn_dim, dim]
    const float* wk = nullptr;       // [kv_dim, dim]
    const float* wv = nullptr;       // [kv_dim, dim]
    const float* wo = nullptr;       // [dim, attn_dim]
    const float* rms_ffn = nullptr;  // [dim]               post_attention_layernorm
    const float* w1 = nullptr;       // [hidden, dim]       gate projection
    const float* w2 = nullptr;       // [dim, hidden]       down projection
    const float* w3 = nullptr;       // [hidden, dim]       up projection
};

struct Weights {
    const float* token_embedding = nullptr;  // [vocab, dim]
    std::vector<LayerWeights> layers;        // n_layers entries
    const float* rms_final = nullptr;        // [dim]
    const float* wcls = nullptr;             // [vocab, dim] (== token_embedding when shared)
};

// For debugging/verification: every tensor that was read, in loading order.
struct TensorView {
    std::string name;
    const float* data = nullptr;       // float32 values (converted if the file stores F16/BF16)
    std::size_t numel = 0;
    std::size_t byte_offset = 0;       // where the tensor's data starts in the file
    std::vector<std::int64_t> shape;
};

class Checkpoint {
public:
    ~Checkpoint();
    Checkpoint(Checkpoint&&) noexcept;
    Checkpoint& operator=(Checkpoint&&) noexcept;

    // Throws std::runtime_error on failure (missing file, invalid or unsupported contents).
    static Checkpoint load(const std::string& path);

    const Config& config() const { return cfg_; }
    const Weights& weights() const { return w_; }
    const std::vector<TensorView>& tensors() const { return tensors_; }
    std::size_t file_size() const;

private:
    Checkpoint();

    static Checkpoint load_llama2c(const std::string& path);
    static Checkpoint load_hf(const std::string& directory);

    MappedFile file_;                              // llama2.c file mapping
    std::unique_ptr<SafetensorsFile> safetensors_;  // Hugging Face weights mapping
    std::vector<std::vector<float>> converted_;     // float32 copies of F16/BF16 tensors
    Config cfg_;
    Weights w_;
    std::vector<TensorView> tensors_;
};

}  // namespace geneinfer
