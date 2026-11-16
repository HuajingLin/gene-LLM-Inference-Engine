#include "gpu/gpu_transformer.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "kernels/kernels.h"

namespace geneinfer {

namespace {

inline std::size_t sz(int v) { return static_cast<std::size_t>(v); }

int env_batch() {
    const char* v = std::getenv("GI_BATCH");
    if (!v || !*v) return 64;
    char* end = nullptr;
    const long n = std::strtol(v, &end, 10);
    if (*end != '\0' || n < 0) {
        std::fprintf(stderr, "warning: GI_BATCH=%s is not a non-negative integer; using 64\n", v);
        return 64;
    }
    return n == 0 ? 64 : static_cast<int>(std::min(n, 4096L));
}

}  // namespace

GpuTransformer::GpuTransformer(const Checkpoint& checkpoint, const gpu::DeviceOps* device, int max_seq_len, int max_batch)
    : dev_(device), cfg_(checkpoint.config()) {
    if (!dev_) throw std::invalid_argument("GpuTransformer: no device");
    if (max_seq_len < 0 || max_seq_len > cfg_.seq_len) {
        throw std::invalid_argument("max_seq_len must be in [0, " + std::to_string(cfg_.seq_len) + "], got " +
                                    std::to_string(max_seq_len));
    }
    ctx_len_ = max_seq_len == 0 ? cfg_.seq_len : max_seq_len;
    if (max_batch < 0) throw std::invalid_argument("max_batch must be >= 0");
    if (max_batch > 4096) throw std::invalid_argument("max_batch must be at most 4096");
    max_batch_ = max_batch == 0 ? env_batch() : max_batch;
    if (checkpoint.has_int8_weights()) {
        throw std::runtime_error("int8 weights are not supported on a device yet: load the model with --weights native or f32");
    }

    const std::size_t dim = sz(cfg_.dim), hidden = sz(cfg_.hidden_dim), attn = sz(cfg_.attn_dim()), kv = sz(cfg_.kv_dim());
    const std::size_t vocab = sz(cfg_.vocab_size), head = sz(cfg_.head_size()), batch = sz(max_batch_);
    const Weights& w = checkpoint.weights();
    try {
        embedding_ = upload_matrix(w.token_embedding, vocab, dim);
        wcls_ = upload_matrix(w.wcls, vocab, dim);
        rms_final_ = upload_vector(w.rms_final, dim);
        layers_.resize(sz(cfg_.n_layers));
        for (int l = 0; l < cfg_.n_layers; ++l) {
            const LayerWeights& lw = w.layers[sz(l)];
            DLayer& d = layers_[sz(l)];
            d.rms_att = upload_vector(lw.rms_att, dim);
            d.rms_ffn = upload_vector(lw.rms_ffn, dim);
            if (cfg_.qk_norm) {
                d.q_norm = upload_vector(lw.q_norm, head);
                d.k_norm = upload_vector(lw.k_norm, head);
            }
            d.wq = upload_matrix(lw.wq, attn, dim);
            d.wk = upload_matrix(lw.wk, kv, dim);
            d.wv = upload_matrix(lw.wv, kv, dim);
            d.wo = upload_matrix(lw.wo, dim, attn);
            d.w1 = upload_matrix(lw.w1, hidden, dim);
            d.w2 = upload_matrix(lw.w2, dim, hidden);
            d.w3 = upload_matrix(lw.w3, hidden, dim);
        }
        weight_bytes_ = device_bytes_;

        const std::size_t kv_count = sz(cfg_.n_layers) * sz(ctx_len_) * kv;
        try {
            keys_ = dalloc_floats(kv_count);
            values_ = dalloc_floats(kv_count);
        } catch (const std::runtime_error& e) {
            throw std::runtime_error(std::string(e.what()) + " - the KV cache for " + std::to_string(ctx_len_) +
                                     " positions needs " + std::to_string((2 * kv_count * sizeof(float)) >> 20) +
                                     " MiB; use a smaller --max-seq-len");
        }
        x_ = dalloc_floats(batch * dim);
        xb_ = dalloc_floats(batch * dim);
        xb2_ = dalloc_floats(batch * dim);
        hb_ = dalloc_floats(batch * hidden);
        hb2_ = dalloc_floats(batch * hidden);
        q_ = dalloc_floats(batch * attn);
        ao_ = dalloc_floats(batch * attn);
        logits_dev_ = dalloc_floats(vocab);
        // matmul scratch: the widest product of a batch, or the classifier for one row.
        const std::size_t widest = std::max(std::max(hidden, attn), std::max(dim, kv));
        partial_ = dalloc_floats(sz(gpu::kPartials) * std::max(batch * widest, vocab));
        scores_ = dalloc_floats(batch * sz(cfg_.n_heads) * sz(ctx_len_));   // attention scores of one batch
        cos_dev_ = dalloc_floats(batch * (head / 2));
        sin_dev_ = dalloc_floats(batch * (head / 2));
        tokens_dev_ = static_cast<int*>(dalloc(batch * sizeof(int)));
        logits_.resize(vocab);
        trace_buf_.resize(dim);
        dev_->sync();
    } catch (...) {
        for (void* p : allocations_) dev_->release(p);
        throw;
    }
}

GpuTransformer::~GpuTransformer() {
    for (void* p : allocations_) dev_->release(p);
}

void* GpuTransformer::dalloc(std::size_t bytes) {
    void* p = dev_->alloc(bytes);
    allocations_.push_back(p);
    device_bytes_ += bytes;
    return p;
}

GpuTransformer::DMat GpuTransformer::upload_matrix(const Mat& m, std::size_t rows, std::size_t cols) {
    DMat d;
    d.bf16 = m.is_bf16() ? 1 : 0;
    const void* host = m.address();
    for (const auto& u : uploaded_) {
        if (u.first == host) {   // the classifier of a tied model is the embedding table: one copy
            d.data = u.second;
            return d;
        }
    }
    const std::size_t bytes = rows * cols * (m.is_bf16() ? 2 : 4);
    void* p = dalloc(bytes);
    dev_->upload(p, host, bytes);
    uploaded_.emplace_back(host, p);
    d.data = p;
    return d;
}

const float* GpuTransformer::upload_vector(const float* v, std::size_t n) {
    float* p = dalloc_floats(n);
    dev_->upload(p, v, n * sizeof(float));
    return p;
}

std::string GpuTransformer::description() const {
    return std::string(dev_->name) + " device: " + dev_->describe() + " | on the device: weights " +
           std::to_string(weight_bytes_ >> 20) + " MiB, KV cache and buffers " +
           std::to_string((device_bytes_ - weight_bytes_) >> 20) + " MiB";
}

void GpuTransformer::matmul(float* out, long out_stride, const DMat& w, const float* x, long x_stride, int ntok, int rows,
                            int cols) {
    dev_->matmul(out, out_stride, w.data, w.bf16, x, x_stride, ntok, rows, cols, partial_);
}

void GpuTransformer::set_profiling(bool on) {
    profiling_ = on;
    for (int s = 0; s < kSections; ++s) section_seconds_[s] = section_weight_bytes_[s] = 0.0;
    profiled_positions_ = profiled_passes_ = 0;
}

void GpuTransformer::lap(Section s, std::size_t weight_bytes) {
    if (!profiling_) return;
    dev_->sync();
    const auto now = std::chrono::steady_clock::now();
    section_seconds_[s] += std::chrono::duration<double>(now - lap_start_).count();
    section_weight_bytes_[s] += static_cast<double>(weight_bytes);
    lap_start_ = now;
}

// Weight bytes one matmul reads: the matrix once per group of tokens that share a pass over it.
std::size_t GpuTransformer::matrix_bytes(const DMat& w, int rows, int cols, int ntok) {
    const std::size_t passes = (sz(ntok) + sz(gpu::kTokensPerPass) - 1) / sz(gpu::kTokensPerPass);
    return sz(rows) * sz(cols) * (w.bf16 ? 2 : 4) * passes;
}

std::string GpuTransformer::profile_report() const {
    if (profiled_positions_ == 0) return std::string();
    static const char* const names[kSections] = {
        "upload (tokens, rope rows)", "embedding lookup", "rmsnorm", "q/k/v projections", "rope", "attention",
        "output projection", "ffn gate+up", "silu * up", "ffn down", "residual adds", "classifier (logits)", "download (logits)"};
    double total = 0.0;
    for (int s = 0; s < kSections; ++s) total += section_seconds_[s];
    char line[200];
    std::snprintf(line, sizeof line, "device profile: %ld positions in %ld passes, %.3f ms per position (the device waits after every operation)\n",
                  profiled_positions_, profiled_passes_, total * 1e3 / static_cast<double>(profiled_positions_));
    std::string out = line;
    std::snprintf(line, sizeof line, "  %-28s %12s %7s %14s\n", "operation", "ms/position", "share", "weights GB/s");
    out += line;
    for (int s = 0; s < kSections; ++s) {
        const double ms = section_seconds_[s] * 1e3 / static_cast<double>(profiled_positions_);
        char rate[32] = "";
        if (section_weight_bytes_[s] > 0.0 && section_seconds_[s] > 0.0) {
            std::snprintf(rate, sizeof rate, "%.1f", section_weight_bytes_[s] / section_seconds_[s] / 1e9);
        }
        std::snprintf(line, sizeof line, "  %-28s %12.3f %6.1f%% %14s\n", names[s], ms, total > 0.0 ? 100.0 * section_seconds_[s] / total : 0.0, rate);
        out += line;
    }
    return out;
}

void GpuTransformer::trace_row(const char* tag, int layer, const float* device_row) {
    dev_->download(trace_buf_.data(), device_row, sz(cfg_.dim) * sizeof(float));
    trace_(tag, layer, trace_buf_.data(), cfg_.dim);
}

const float* GpuTransformer::forward(int token, int pos, bool compute_logits) {
    if (token < 0 || token >= cfg_.vocab_size) {
        throw std::out_of_range("forward: token id " + std::to_string(token) + " out of range");
    }
    if (pos < 0 || pos >= ctx_len_) {
        throw std::out_of_range("forward: position " + std::to_string(pos) + " exceeds the context length " +
                                std::to_string(ctx_len_));
    }
    if (pos > length_) {
        throw std::out_of_range("forward: position " + std::to_string(pos) + " leaves a gap, the sequence holds " +
                                std::to_string(length_) + " positions");
    }
    run(&token, 1, pos, compute_logits);
    return compute_logits ? logits_.data() : nullptr;
}

const float* GpuTransformer::forward_batch(const int* tokens, int n, int pos0, bool compute_logits) {
    if (n < 1) throw std::invalid_argument("forward_batch: no tokens");
    if (pos0 < 0 || pos0 > ctx_len_ - n) {
        throw std::out_of_range("forward_batch: positions " + std::to_string(pos0) + ".." + std::to_string(pos0 + n - 1) +
                                " exceed the context length " + std::to_string(ctx_len_));
    }
    if (pos0 > length_) {
        throw std::out_of_range("forward_batch: start position " + std::to_string(pos0) + " leaves a gap, the sequence holds " +
                                std::to_string(length_) + " positions");
    }
    for (int i = 0; i < n; ++i) {
        if (tokens[i] < 0 || tokens[i] >= cfg_.vocab_size) {
            throw std::out_of_range("forward_batch: token id " + std::to_string(tokens[i]) + " out of range");
        }
    }
    const int step = trace_ ? 1 : max_batch_;   // the trace wants every token: one by one
    for (int done = 0; done < n; done += step) {
        const int m = std::min(step, n - done);
        run(tokens + done, m, pos0 + done, compute_logits && done + m == n);
    }
    return compute_logits ? logits_.data() : nullptr;
}

// m tokens at the positions pos0 .. pos0 + m - 1. Everything between the upload of the token ids and the download
// of the logits stays on the device.
void GpuTransformer::run(const int* tokens, int m, int pos0, bool want_logits) {
    const int dim = cfg_.dim, hidden = cfg_.hidden_dim, attn_dim = cfg_.attn_dim(), kv_dim = cfg_.kv_dim();
    const int head_size = cfg_.head_size(), n_heads = cfg_.n_heads, kv_mul = cfg_.kv_mul();
    const float inv_scale = 1.0f / std::sqrt(static_cast<float>(head_size));
    const float eps = cfg_.rms_norm_eps;
    const int half_split = cfg_.rope_style == RopeStyle::HalfSplit ? 1 : 0;
    const std::size_t half = sz(head_size / 2);
    const bool traced = static_cast<bool>(trace_) && m == 1;

    if (profiling_) {
        dev_->sync();
        lap_start_ = std::chrono::steady_clock::now();
        profiled_positions_ += m;
        ++profiled_passes_;
    }
    // Rotary cos/sin rows: computed on the host in double precision exactly as for the CPU model, then uploaded.
    while (rope_rows_ < pos0 + m) {
        rope_cos_.resize((sz(rope_rows_) + 1) * half);
        rope_sin_.resize((sz(rope_rows_) + 1) * half);
        kernels::rope_table_row(rope_cos_.data() + sz(rope_rows_) * half, rope_sin_.data() + sz(rope_rows_) * half,
                                rope_rows_, head_size, cfg_.rope_theta, &cfg_.rope_scaling);
        ++rope_rows_;
    }
    dev_->upload(cos_dev_, rope_cos_.data() + sz(pos0) * half, sz(m) * half * sizeof(float));
    dev_->upload(sin_dev_, rope_sin_.data() + sz(pos0) * half, sz(m) * half * sizeof(float));
    dev_->upload(tokens_dev_, tokens, sz(m) * sizeof(int));
    lap(Transfer);

    dev_->embed(x_, embedding_.data, embedding_.bf16, tokens_dev_, m, dim);
    lap(Embed);
    if (traced) trace_row("embedding", -1, x_);

    for (int l = 0; l < cfg_.n_layers; ++l) {
        const DLayer& lw = layers_[sz(l)];
        float* keys_l = keys_ + sz(l) * sz(ctx_len_) * sz(kv_dim);
        float* values_l = values_ + sz(l) * sz(ctx_len_) * sz(kv_dim);
        float* k_new = keys_l + sz(pos0) * sz(kv_dim);     // the new rows are computed straight into the cache
        float* v_new = values_l + sz(pos0) * sz(kv_dim);

        // ---------------- attention block ----------------
        dev_->rmsnorm(xb_, x_, lw.rms_att, m, dim, eps);
        lap(Norm);
        matmul(q_, attn_dim, lw.wq, xb_, dim, m, attn_dim, dim);
        matmul(k_new, kv_dim, lw.wk, xb_, dim, m, kv_dim, dim);
        matmul(v_new, kv_dim, lw.wv, xb_, dim, m, kv_dim, dim);
        lap(QkvProj, matrix_bytes(lw.wq, attn_dim, dim, m) + 2 * matrix_bytes(lw.wk, kv_dim, dim, m));
        if (cfg_.qk_norm) {   // Qwen3: every q and k head is RMS-normalized before RoPE
            dev_->rmsnorm(q_, q_, lw.q_norm, m * n_heads, head_size, eps);
            dev_->rmsnorm(k_new, k_new, lw.k_norm, m * cfg_.n_kv_heads, head_size, eps);
            lap(Norm);
        }
        dev_->rope(q_, attn_dim, m, attn_dim, cos_dev_, sin_dev_, head_size, half_split);
        dev_->rope(k_new, kv_dim, m, kv_dim, cos_dev_, sin_dev_, head_size, half_split);
        lap(Rope);
        dev_->attention(ao_, q_, keys_l, values_l, m, pos0, n_heads, head_size, kv_mul, kv_dim, inv_scale, scores_);
        lap(Attention);
        matmul(xb2_, dim, lw.wo, ao_, attn_dim, m, dim, attn_dim);
        lap(OutProj, matrix_bytes(lw.wo, dim, attn_dim, m));
        dev_->add(x_, xb2_, static_cast<long>(m) * dim);
        lap(Residual);
        if (traced) trace_row("attn_out", l, x_);

        // ---------------- feed-forward block (SwiGLU) ----------------
        dev_->rmsnorm(xb_, x_, lw.rms_ffn, m, dim, eps);
        lap(Norm);
        matmul(hb_, hidden, lw.w1, xb_, dim, m, hidden, dim);
        matmul(hb2_, hidden, lw.w3, xb_, dim, m, hidden, dim);
        lap(FfnGateUp, 2 * matrix_bytes(lw.w1, hidden, dim, m));
        dev_->silu_mul(hb_, hb2_, static_cast<long>(m) * hidden);
        lap(Activation);
        matmul(xb_, dim, lw.w2, hb_, hidden, m, dim, hidden);
        lap(FfnDown, matrix_bytes(lw.w2, dim, hidden, m));
        dev_->add(x_, xb_, static_cast<long>(m) * dim);
        lap(Residual);
        if (traced) trace_row("ffn_out", l, x_);
    }
    length_ = pos0 + m;

    if (!want_logits) {
        dev_->sync();   // so that a caller timing this call measures the work, not just its launch
        return;
    }
    dev_->rmsnorm(xb_, x_ + sz(m - 1) * sz(dim), rms_final_, 1, dim, eps);
    lap(Norm);
    if (traced) trace_row("final_norm", -1, xb_);
    matmul(logits_dev_, cfg_.vocab_size, wcls_, xb_, dim, 1, cfg_.vocab_size, dim);
    lap(Classifier, matrix_bytes(wcls_, cfg_.vocab_size, dim, 1));
    dev_->download(logits_.data(), logits_dev_, sz(cfg_.vocab_size) * sizeof(float));
    lap(Download);
}

}  // namespace geneinfer
