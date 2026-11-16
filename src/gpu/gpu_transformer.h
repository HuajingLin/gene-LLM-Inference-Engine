// gpu_transformer.h - the forward pass on a compute device (see gpu/device_ops.h).
//
// The weights are copied to device memory once (float32 or bfloat16, as the checkpoint holds them); activations
// and the KV cache live there too. Per pass only the token ids, the rotary cos/sin rows and the block tables go to
// the device and only the logits (vocab_size floats) come back.
//
// The KV cache is the paged pool of model/kv_cache.h: KvPool keeps the free list and the reference counts on the
// host, the block memory is on the device, and the attention kernels find every key and value through the block
// table of the row's sequence (vLLM's PagedAttention). Sequences, prefix sharing (retain + adopt) and
// copy-on-write work as on the CPU.
//
// The math is the same as Transformer's, but sums are formed in a different order (32 partial sums per matrix
// row, fused multiply-adds), so results agree with the CPU to rounding (about 1e-5 relative), not bit for bit.
// The values of a position depend neither on the batch size, nor on the block size, nor on which blocks hold it.
#pragma once

#include <chrono>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "gpu/device_ops.h"
#include "model/model.h"

namespace geneinfer {

class GpuTransformer : public SequenceModel {
public:
    // The Checkpoint is only needed during construction. `max_seq_len` caps the context (0 = the model's);
    // `max_batch` is the most tokens one pass processes (0 = the environment variable GI_BATCH, default 64).
    // `block_size` and `kv_blocks` shape the paged KV cache as for Transformer: positions per block (0 = the
    // environment variable GI_BLOCK_SIZE, default 16) and blocks in the pool (0 = enough for one full sequence).
    // Throws std::runtime_error if the checkpoint holds int8 weights or the device has too little memory.
    GpuTransformer(const Checkpoint& checkpoint, const gpu::DeviceOps* device, int max_seq_len = 0, int max_batch = 0,
                   int block_size = 0, int kv_blocks = 0);
    ~GpuTransformer() override;
    GpuTransformer(const GpuTransformer&) = delete;
    GpuTransformer& operator=(const GpuTransformer&) = delete;

    const float* forward(int token, int pos, bool compute_logits = true) override;
    const float* forward_batch(const int* tokens, int n, int pos0, bool compute_logits = true) override;
    const float* forward(Sequence& seq, int token, int pos, bool compute_logits = true) override;
    const float* forward_batch(Sequence& seq, const int* tokens, int n, int pos0, bool compute_logits = true) override;
    KvPool& kv_pool() override { return *kv_; }

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
    // m rows in one pass: row i is tokens[i] at positions[i] of seqs[i] (blocks reserved and writable). Rows of one
    // sequence come in increasing position order. Logits of the last row if want_logits.
    void run_rows(const int* tokens, Sequence* const* seqs, const int* positions, int m, bool want_logits);
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

    std::vector<void*> allocations_;
    std::vector<std::pair<const void*, const void*>> uploaded_;   // host address -> device address (tied weights)
    std::size_t device_bytes_ = 0, weight_bytes_ = 0;

    DMat embedding_, wcls_;
    const float* rms_final_ = nullptr;
    std::vector<DLayer> layers_;

    float* pool_dev_ = nullptr;                // [kv_blocks][block_floats], the block memory of kv_
    float *kbuf_ = nullptr, *vbuf_ = nullptr;  // [max_batch, kv_dim] new keys / values before they go into their blocks
    int* tables_dev_ = nullptr;                // [max_batch, blocks of a full context]
    int* positions_dev_ = nullptr;             // [max_batch]
    std::vector<int> tables_host_;
    std::vector<Sequence*> row_seq_;
    std::vector<int> row_pos_;
    std::vector<float> row_cos_, row_sin_;     // [max_batch, head_size / 2] rows gathered for one pass
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

    std::unique_ptr<KvPool> kv_;              // bookkeeping of the device blocks
    std::unique_ptr<Sequence> default_seq_;   // declared after kv_: destroyed first, gives its blocks back
};

}  // namespace geneinfer
