// checkpoint.h - model configuration and weights, loaded from disk.
//
// Two formats are supported; Checkpoint::load() picks one from the path:
//   * a file          -> llama2.c float32 checkpoint (v0), e.g. stories15M.bin
//   * a directory     -> Hugging Face model directory with config.json and model.safetensors
//                        (Llama architecture; weights may be F32, F16 or BF16)
//
// All weight matrices are row-major with shape [out_features, in_features], i.e. y = W * x.
// Float32 data in a file is used in place (memory-mapped, no copy). bfloat16 matrices are ALSO used in
// place, as 16-bit values: they are widened to float32 inside the kernels (an exact 16-bit shift), which
// halves the memory traffic of decoding and the RAM the weights need. Set the environment variable
// GI_WEIGHTS=f32 (or LoadOptions::keep_bf16 = false) to expand them to float32 at load time instead; both
// ways give bit-identical results. F16 tensors, and the small norm vectors of any file, are converted to
// float32 into memory owned by the Checkpoint.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "common/rope_style.h"
#include "loader/mapped_file.h"

namespace geneinfer {

class SafetensorsFile;

// How a weight tensor is stored in memory.
enum class Storage { F32, BF16 };

// A read-only view of a weight matrix: float32, bfloat16 (the upper 16 bits of a float32), or int8 with one
// bfloat16 scale per group of 64 values of a row (common/q8.h).
struct Mat {
    const float* f32 = nullptr;
    const std::uint16_t* bf16 = nullptr;
    const std::int8_t* q8 = nullptr;           // int8 values [rows, cols]
    const std::uint16_t* q8_scale = nullptr;   // bf16 scales [rows, ceil(cols / 64)]

    Mat() = default;
    Mat(const float* p) : f32(p) {}              // NOLINT: implicit on purpose, llama2.c data is plain float
    Mat(const std::uint16_t* p) : bf16(p) {}     // NOLINT
    Mat(const std::int8_t* q, const std::uint16_t* scale) : q8(q), q8_scale(scale) {}
    bool is_bf16() const { return bf16 != nullptr; }
    bool is_q8() const { return q8 != nullptr; }
    const void* address() const {
        if (q8) return q8;
        return bf16 ? static_cast<const void*>(bf16) : static_cast<const void*>(f32);
    }
    friend bool operator==(const Mat& a, const Mat& b) {
        return a.address() == b.address() && a.is_bf16() == b.is_bf16() && a.is_q8() == b.is_q8();
    }
    friend bool operator!=(const Mat& a, const Mat& b) { return !(a == b); }
};

struct LoadOptions {
    bool keep_bf16 = true;   // keep bfloat16 matrices as bfloat16 (see above)
    // Quantize the layer matrices and the classifier to int8 (group size 64, bfloat16 scales) at load time. The
    // token embedding table stays as it is: only one row of it is read per token. Costs a few hundred
    // milliseconds at load; the quantized copy is owned by the Checkpoint (see owned_bytes()).
    bool int8 = false;
    // Defaults, with GI_WEIGHTS=f32 in the environment switching keep_bf16 off and GI_WEIGHTS=int8 switching
    // int8 on.
    static LoadOptions from_environment();
};

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
    Mat wq;       // [attn_dim, dim]
    Mat wk;       // [kv_dim, dim]
    Mat wv;       // [kv_dim, dim]
    Mat wo;       // [dim, attn_dim]
    const float* rms_ffn = nullptr;  // [dim]               post_attention_layernorm
    Mat w1;       // [hidden, dim]       gate projection
    Mat w2;       // [dim, hidden]       down projection
    Mat w3;       // [hidden, dim]       up projection
};

struct Weights {
    Mat token_embedding;                     // [vocab, dim]
    std::vector<LayerWeights> layers;        // n_layers entries
    const float* rms_final = nullptr;        // [dim]
    Mat wcls;                                // [vocab, dim] (== token_embedding when shared)
};

// For debugging/verification: every tensor that was read, in loading order.
struct TensorView {
    std::string name;
    Storage storage = Storage::F32;
    const float* data = nullptr;       // float32 values, when storage == F32 (converted if the file stores F16)
    const std::uint16_t* bf16 = nullptr;  // bfloat16 values, when storage == BF16 (in the mapped file)
    std::size_t numel = 0;
    std::size_t element_bytes() const { return storage == Storage::BF16 ? 2 : 4; }
    const void* address() const { return bf16 ? static_cast<const void*>(bf16) : static_cast<const void*>(data); }
    float value(std::size_t i) const;  // element i as float32 (exact for bfloat16)
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
    static Checkpoint load(const std::string& path, const LoadOptions& options);

    const Config& config() const { return cfg_; }
    const Weights& weights() const { return w_; }
    const std::vector<TensorView>& tensors() const { return tensors_; }
    std::size_t file_size() const;
    bool has_bf16_weights() const;   // any matrix kept as bfloat16
    std::size_t owned_bytes() const; // memory allocated for converted or quantized tensors (the rest is the file mapping)
    bool has_int8_weights() const { return int8_; }

private:
    Checkpoint();

    static Checkpoint load_llama2c(const std::string& path);
    static Checkpoint load_hf(const std::string& directory, const LoadOptions& options);

    MappedFile file_;                              // llama2.c file mapping
    std::unique_ptr<SafetensorsFile> safetensors_;  // Hugging Face weights mapping
    std::vector<std::vector<float>> converted_;     // float32 copies of F16/BF16 tensors
    std::vector<std::vector<std::int8_t>> q8_values_;     // int8 matrices (LoadOptions::int8)
    std::vector<std::vector<std::uint16_t>> q8_scales_;   // their bfloat16 group scales
    bool int8_ = false;

    void quantize_matrices();
    Config cfg_;
    Weights w_;
    std::vector<TensorView> tensors_;
};

}  // namespace geneinfer
