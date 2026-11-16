// gpu_transformer.h - the forward pass on a compute device (see gpu/device_ops.h), for ONE sequence.
//
// The weights are copied to device memory once (float32 or bfloat16, as the checkpoint holds them); activations
// and the KV cache live there too. Per call only the token ids and the rotary cos/sin rows go to the device and
// only the logits (vocab_size floats) come back. The KV cache is contiguous: [layer][position][kv_dim].
//
// The math is the same as Transformer's, but sums are formed in a different order (32 partial sums per matrix
// row, fused multiply-adds), so results agree with the CPU to rounding (about 1e-5 relative), not bit for bit.
// forward_batch() and forward() give the same values for a position whatever the batch size.
#pragma once

#include <chrono>
#include <cstddef>
#include <string>
#include <vector>

#include "gpu/device_ops.h"
#include "model/model.h"

namespace geneinfer {

class GpuTransformer : public Model {
public:
    // The Checkpoint is only needed during construction. `max_seq_len` caps the context (0 = the model's);
    // `max_batch` is the most tokens one pass processes (0 = the environment variable GI_BATCH, default 64).
    // Throws std::runtime_error if the checkpoint holds int8 weights or the device has too little memory.
    GpuTransformer(const Checkpoint& checkpoint, const gpu::DeviceOps* device, int max_seq_len = 0, int max_batch = 0);
    ~GpuTransformer() override;
    GpuTransformer(const GpuTransformer&) = delete;
    GpuTransformer& operator=(const GpuTransformer&) = delete;

    const float* forward(int token, int pos, bool compute_logits = true) override;
    const float* forward_batch(const int* tokens, int n, int pos0, bool compute_logits = true) override;

    const Config& config() const override { return cfg_; }
    int max_seq_len() const override { return ctx_len_; }
    void set_trace(TraceFn fn) override { trace_ = std::move(fn); }
    std::string description() const override;

    void set_profiling(bool on) override;
    std::string profile_report() const override;

    int max_batch() const { return max_batch_; }
    std::size_t device_bytes() const { return device_bytes_; }   // everything allocated on the device
    std::size_t weight_bytes() const { return weight_bytes_; }   // of which weights

private:
    struct DMat {
        const void* data = nullptr;
        int bf16 = 0;
    };
    struct DLayer {
        const float* rms_att = nullptr;
        const float* rms_ffn = nullptr;
        const float* q_norm = nullptr;
        const float* k_norm = nullptr;
        DMat wq, wk, wv, wo, w1, w2, w3;
    };

    void* dalloc(std::size_t bytes);
    float* dalloc_floats(std::size_t count) { return static_cast<float*>(dalloc(count * sizeof(float))); }
    DMat upload_matrix(const Mat& m, std::size_t rows, std::size_t cols);
    const float* upload_vector(const float* v, std::size_t n);
    void matmul(float* out, long out_stride, const DMat& w, const float* x, long x_stride, int ntok, int rows, int cols);
    void run(const int* tokens, int m, int pos0, bool want_logits);
    void trace_row(const char* tag, int layer, const float* device_row);

    enum Section { Transfer, Embed, Norm, QkvProj, Rope, Attention, OutProj, FfnGateUp, Activation, FfnDown, Residual,
                   Classifier, Download, kSections };
    void lap(Section s, std::size_t weight_bytes = 0);   // with profiling: wait for the device, book the time since the last lap
    static std::size_t matrix_bytes(const DMat& w, int rows, int cols, int ntok);

    bool profiling_ = false;
    std::chrono::steady_clock::time_point lap_start_;
    double section_seconds_[kSections] = {};
    double section_weight_bytes_[kSections] = {};
    long profiled_positions_ = 0, profiled_passes_ = 0;

    const gpu::DeviceOps* dev_;
    Config cfg_;
    int ctx_len_ = 0;
    int max_batch_ = 64;
    int length_ = 0;   // positions in the KV cache

    std::vector<void*> allocations_;
    std::vector<std::pair<const void*, const void*>> uploaded_;   // host address -> device address (tied weights)
    std::size_t device_bytes_ = 0, weight_bytes_ = 0;

    DMat embedding_, wcls_;
    const float* rms_final_ = nullptr;
    std::vector<DLayer> layers_;

    float* keys_ = nullptr;     // [n_layers, ctx_len, kv_dim]
    float* values_ = nullptr;
    float *x_ = nullptr, *xb_ = nullptr, *xb2_ = nullptr, *hb_ = nullptr, *hb2_ = nullptr, *q_ = nullptr, *ao_ = nullptr;
    float* logits_dev_ = nullptr;
    float* partial_ = nullptr;
    float* scores_ = nullptr;                         // [max_batch, n_heads, ctx_len]
    float *cos_dev_ = nullptr, *sin_dev_ = nullptr;   // [max_batch, head_size / 2]
    int* tokens_dev_ = nullptr;                       // [max_batch]

    std::vector<float> rope_cos_, rope_sin_;   // host table [rows, head_size / 2], filled on demand
    int rope_rows_ = 0;
    std::vector<float> logits_;                // host copy returned by forward()
    std::vector<float> trace_buf_;
    TraceFn trace_;
};

}  // namespace geneinfer
