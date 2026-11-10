#include "model/transformer.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#include "kernels/kernels.h"

namespace geneinfer {

namespace {

inline std::size_t sz(int v) { return static_cast<std::size_t>(v); }

// out = W * x for a float32 or bfloat16 weight matrix.
inline void matvec(float* out, const Mat& W, const float* x, int rows, int cols) {
    if (W.is_bf16()) kernels::matvec_bf16(out, W.bf16, x, rows, cols);
    else kernels::matvec(out, W.f32, x, rows, cols);
}

// Rows [begin, end) of the product, written to out[begin..end).
inline void matvec_rows(float* out, const Mat& W, const float* x, int begin, int end, int cols) {
    const std::size_t offset = static_cast<std::size_t>(begin) * static_cast<std::size_t>(cols);
    if (W.is_bf16()) kernels::matvec_bf16(out + begin, W.bf16 + offset, x, end - begin, cols);
    else kernels::matvec(out + begin, W.f32 + offset, x, end - begin, cols);
}

long env_long(const char* name, long fallback) {
    const char* v = std::getenv(name);
    if (!v || !*v) return fallback;
    char* end = nullptr;
    const long n = std::strtol(v, &end, 10);
    if (*end != '\0' || n < 0) {
        std::fprintf(stderr, "warning: %s=%s is not a non-negative integer; using %ld\n", name, v, fallback);
        return fallback;
    }
    return n;
}

// Attributes the time since the previous checkpoint to `section`.
inline void lap(ForwardProfile* p, ForwardProfile::Section section) {
    if (!p) return;
    const auto now = std::chrono::steady_clock::now();
    p->seconds[section] += std::chrono::duration<double>(now - p->last).count();
    p->last = now;
}

}  // namespace

const char* ForwardProfile::name(int s) {
    static const char* const names[kSections] = {
        "embedding lookup", "rmsnorm", "q/k/v projections", "rope", "attention (scores, softmax, mix)",
        "output projection", "ffn gate+up", "silu * up", "ffn down", "residual adds", "classifier (logits)"};
    return names[s];
}

Transformer::Transformer(const Checkpoint& checkpoint, int max_seq_len, int n_threads)
    : cfg_(checkpoint.config()), w_(checkpoint.weights()) {
    if (max_seq_len < 0 || max_seq_len > cfg_.seq_len) {
        throw std::invalid_argument("max_seq_len must be in [0, " + std::to_string(cfg_.seq_len) +
                                    "], got " + std::to_string(max_seq_len));
    }
    ctx_len_ = max_seq_len == 0 ? cfg_.seq_len : max_seq_len;

    if (n_threads < 0) throw std::invalid_argument("n_threads must be >= 0");
    if (n_threads == 0) n_threads = static_cast<int>(env_long("GI_THREADS", 1));
    if (n_threads == 0) n_threads = 1;
    if (n_threads > 256) throw std::invalid_argument("n_threads must be at most 256");
    par_min_weights_ = env_long("GI_PAR_MIN", par_min_weights_);   // tests lower it to force tiny products to split
    if (n_threads > 1) pool_ = std::make_unique<ThreadPool>(n_threads);

    x_.resize(sz(cfg_.dim));
    xb_.resize(sz(cfg_.dim));
    xb2_.resize(sz(cfg_.dim));
    hb_.resize(sz(cfg_.hidden_dim));
    hb2_.resize(sz(cfg_.hidden_dim));
    q_.resize(sz(cfg_.attn_dim()));
    ao_.resize(sz(cfg_.attn_dim()));
    att_.resize(sz(cfg_.n_heads) * sz(ctx_len_));
    logits_.resize(sz(cfg_.vocab_size));

    const std::size_t cache_size = sz(cfg_.n_layers) * sz(ctx_len_) * sz(cfg_.kv_dim());
    key_cache_.assign(cache_size, 0.0f);
    value_cache_.assign(cache_size, 0.0f);
}

Transformer::~Transformer() = default;

int Transformer::threads() const { return pool_ ? pool_->size() : 1; }

// Runs several independent matrix-vector products with one synchronization of the pool. Each product is cut
// into slices of consecutive rows, one per thread; slice starts are multiples of 4, which is the row blocking of
// the kernels, so every row is computed exactly as in a single-threaded run (bit-identical results). A product
// with fewer than par_min_weights_ weights per thread is not split: thread 0 computes it alone.
void Transformer::run_matvecs(const MatvecJob* jobs, int n) {
    if (!pool_) {
        for (int j = 0; j < n; ++j) matvec(jobs[j].out, jobs[j].W, jobs[j].x, jobs[j].rows, jobs[j].cols);
        return;
    }
    struct Ctx {
        const MatvecJob* jobs;
        int n;
        long par_min;
    } ctx{jobs, n, par_min_weights_};
    pool_->run(
        [](void* p, int t, int nt) {
            const Ctx& c = *static_cast<const Ctx*>(p);
            for (int j = 0; j < c.n; ++j) {
                const MatvecJob& job = c.jobs[j];
                const long weights = static_cast<long>(job.rows) * job.cols;
                if (weights < c.par_min * nt) {   // too small to split
                    if (t == 0) matvec(job.out, job.W, job.x, job.rows, job.cols);
                    continue;
                }
                const int per = (job.rows + nt - 1) / nt;
                const int chunk = (per + 3) / 4 * 4;             // multiple of 4: the kernels' row block
                const int begin = std::min(job.rows, t * chunk);
                const int end = std::min(job.rows, begin + chunk);
                if (begin < end) matvec_rows(job.out, job.W, job.x, begin, end, job.cols);
            }
        },
        &ctx);
}

std::size_t Transformer::kv_cache_bytes() const {
    return (key_cache_.size() + value_cache_.size()) * sizeof(float);
}

std::size_t Transformer::scratch_bytes() const {
    return (x_.size() + xb_.size() + xb2_.size() + hb_.size() + hb2_.size() + q_.size() +
            ao_.size() + att_.size() + logits_.size()) * sizeof(float);
}

const float* Transformer::forward(int token, int pos, bool compute_logits) {
    if (token < 0 || token >= cfg_.vocab_size) {
        throw std::out_of_range("forward: token id " + std::to_string(token) + " out of range");
    }
    if (pos < 0 || pos >= ctx_len_) {
        throw std::out_of_range("forward: position " + std::to_string(pos) +
                                " exceeds the context length " + std::to_string(ctx_len_));
    }

    if (profile_) {
        profile_->last = std::chrono::steady_clock::now();
        ++profile_->forward_calls;
    }
    const int dim = cfg_.dim;
    const int hidden = cfg_.hidden_dim;
    const int attn_dim = cfg_.attn_dim();  // n_heads * head_dim (== dim for llama2.c models)
    const int kv_dim = cfg_.kv_dim();
    const int kv_mul = cfg_.kv_mul();
    const int head_size = cfg_.head_size();
    const float inv_scale = 1.0f / std::sqrt(static_cast<float>(head_size));
    const float eps = cfg_.rms_norm_eps;

    float* x = x_.data();
    float* xb = xb_.data();
    float* xb2 = xb2_.data();
    float* hb = hb_.data();
    float* hb2 = hb2_.data();
    float* q = q_.data();
    float* ao = ao_.data();

    // 1. Token embedding: copy one row of the embedding table into the residual stream.
    if (w_.token_embedding.is_bf16()) {
        kernels::widen_bf16(x, w_.token_embedding.bf16 + sz(token) * sz(dim), dim);
    } else {
        std::memcpy(x, w_.token_embedding.f32 + sz(token) * sz(dim), sz(dim) * sizeof(float));
    }
    lap(profile_, ForwardProfile::Embed);
    if (trace_) trace_("embedding", -1, x, dim);

    for (int l = 0; l < cfg_.n_layers; ++l) {
        const std::size_t layer = sz(l);
        const LayerWeights& lw = w_.layers[layer];

        // Start of this layer's slice of the KV cache; positions follow, each kv_dim wide.
        float* layer_keys = key_cache_.data() + layer * sz(ctx_len_) * sz(kv_dim);
        float* layer_values = value_cache_.data() + layer * sz(ctx_len_) * sz(kv_dim);

        // ---------------- attention block ----------------
        kernels::rmsnorm(xb, x, lw.rms_att, dim, eps);
        lap(profile_, ForwardProfile::Norm);

        // K and V of the current position are written directly into the cache.
        float* k = layer_keys + sz(pos) * sz(kv_dim);
        float* v = layer_values + sz(pos) * sz(kv_dim);
        const MatvecJob qkv[3] = {{q, lw.wq, xb, attn_dim, dim}, {k, lw.wk, xb, kv_dim, dim}, {v, lw.wv, xb, kv_dim, dim}};
        run_matvecs(qkv, 3);
        lap(profile_, ForwardProfile::QkvProj);

        // Rotary position embedding on q and k (v is not rotated).
        kernels::rope(q, attn_dim, pos, head_size, cfg_.rope_theta, cfg_.rope_style);
        kernels::rope(k, kv_dim, pos, head_size, cfg_.rope_theta, cfg_.rope_style);
        lap(profile_, ForwardProfile::Rope);

        // Each query head attends over all positions 0..pos of its (shared) KV head.
        for (int h = 0; h < cfg_.n_heads; ++h) {
            const float* qh = q + sz(h) * sz(head_size);
            float* scores = att_.data() + sz(h) * sz(ctx_len_);
            const int kv_head = h / kv_mul;  // GQA: kv_mul query heads share one KV head

            for (int t = 0; t <= pos; ++t) {
                const float* kt = layer_keys + sz(t) * sz(kv_dim) + sz(kv_head) * sz(head_size);
                scores[t] = kernels::dot(qh, kt, head_size) * inv_scale;
            }
            kernels::softmax(scores, pos + 1);

            // Weighted sum of the value vectors; the head's output goes into ao.
            float* out = ao + sz(h) * sz(head_size);
            std::fill(out, out + head_size, 0.0f);
            for (int t = 0; t <= pos; ++t) {
                const float* vt = layer_values + sz(t) * sz(kv_dim) + sz(kv_head) * sz(head_size);
                kernels::axpy(out, scores[t], vt, head_size);
            }
        }

        lap(profile_, ForwardProfile::Attention);

        // Output projection, then add to the residual stream.
        const MatvecJob oproj = {xb2, lw.wo, ao, dim, attn_dim};
        run_matvecs(&oproj, 1);
        lap(profile_, ForwardProfile::OutProj);
        kernels::add(x, xb2, dim);
        lap(profile_, ForwardProfile::Residual);
        if (trace_) trace_("attn_out", l, x, dim);

        // ---------------- feed-forward block (SwiGLU) ----------------
        kernels::rmsnorm(xb, x, lw.rms_ffn, dim, eps);
        lap(profile_, ForwardProfile::Norm);
        const MatvecJob gate_up[2] = {{hb, lw.w1, xb, hidden, dim}, {hb2, lw.w3, xb, hidden, dim}};   // gate, up
        run_matvecs(gate_up, 2);
        lap(profile_, ForwardProfile::FfnGateUp);
        kernels::silu(hb, hidden);
        kernels::mul(hb, hb2, hidden);
        lap(profile_, ForwardProfile::Activation);
        const MatvecJob down = {xb, lw.w2, hb, dim, hidden};
        run_matvecs(&down, 1);
        lap(profile_, ForwardProfile::FfnDown);
        kernels::add(x, xb, dim);
        lap(profile_, ForwardProfile::Residual);
        if (trace_) trace_("ffn_out", l, x, dim);
    }

    // 3. Final norm and classifier (for shared weights, wcls is the embedding table).
    if (!compute_logits) return nullptr;
    kernels::rmsnorm(x, x, w_.rms_final, dim, eps);
    lap(profile_, ForwardProfile::Norm);
    if (trace_) trace_("final_norm", -1, x, dim);
    const MatvecJob head = {logits_.data(), w_.wcls, x, cfg_.vocab_size, dim};
    run_matvecs(&head, 1);
    lap(profile_, ForwardProfile::Classifier);

    return logits_.data();
}

}  // namespace geneinfer
