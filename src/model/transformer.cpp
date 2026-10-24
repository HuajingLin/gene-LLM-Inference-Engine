#include "model/transformer.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>

#include "kernels/kernels.h"

namespace geneinfer {

namespace {

inline std::size_t sz(int v) { return static_cast<std::size_t>(v); }

}  // namespace

Transformer::Transformer(const Checkpoint& checkpoint, int max_seq_len)
    : cfg_(checkpoint.config()), w_(checkpoint.weights()) {
    if (max_seq_len < 0 || max_seq_len > cfg_.seq_len) {
        throw std::invalid_argument("max_seq_len must be in [0, " + std::to_string(cfg_.seq_len) +
                                    "], got " + std::to_string(max_seq_len));
    }
    ctx_len_ = max_seq_len == 0 ? cfg_.seq_len : max_seq_len;

    x_.resize(sz(cfg_.dim));
    xb_.resize(sz(cfg_.dim));
    xb2_.resize(sz(cfg_.dim));
    hb_.resize(sz(cfg_.hidden_dim));
    hb2_.resize(sz(cfg_.hidden_dim));
    q_.resize(sz(cfg_.dim));
    att_.resize(sz(cfg_.n_heads) * sz(ctx_len_));
    logits_.resize(sz(cfg_.vocab_size));

    const std::size_t cache_size = sz(cfg_.n_layers) * sz(ctx_len_) * sz(cfg_.kv_dim());
    key_cache_.assign(cache_size, 0.0f);
    value_cache_.assign(cache_size, 0.0f);
}

const float* Transformer::forward(int token, int pos) {
    if (token < 0 || token >= cfg_.vocab_size) {
        throw std::out_of_range("forward: token id " + std::to_string(token) + " out of range");
    }
    if (pos < 0 || pos >= ctx_len_) {
        throw std::out_of_range("forward: position " + std::to_string(pos) +
                                " exceeds the context length " + std::to_string(ctx_len_));
    }

    const int dim = cfg_.dim;
    const int hidden = cfg_.hidden_dim;
    const int kv_dim = cfg_.kv_dim();
    const int kv_mul = cfg_.kv_mul();
    const int head_size = cfg_.head_size();
    const float inv_scale = 1.0f / std::sqrt(static_cast<float>(head_size));

    float* x = x_.data();
    float* xb = xb_.data();
    float* xb2 = xb2_.data();
    float* hb = hb_.data();
    float* hb2 = hb2_.data();
    float* q = q_.data();

    // 1. Token embedding: copy one row of the embedding table into the residual stream.
    std::memcpy(x, w_.token_embedding + sz(token) * sz(dim), sz(dim) * sizeof(float));
    if (trace_) trace_("embedding", -1, x, dim);

    for (int l = 0; l < cfg_.n_layers; ++l) {
        const std::size_t layer = sz(l);

        // Start of this layer's slice of the KV cache; positions follow, each kv_dim wide.
        float* layer_keys = key_cache_.data() + layer * sz(ctx_len_) * sz(kv_dim);
        float* layer_values = value_cache_.data() + layer * sz(ctx_len_) * sz(kv_dim);

        // ---------------- attention block ----------------
        kernels::rmsnorm(xb, x, w_.rms_att + layer * sz(dim), dim);

        // K and V of the current position are written directly into the cache.
        float* k = layer_keys + sz(pos) * sz(kv_dim);
        float* v = layer_values + sz(pos) * sz(kv_dim);
        kernels::matvec(q, w_.wq + layer * sz(dim) * sz(dim), xb, dim, dim);
        kernels::matvec(k, w_.wk + layer * sz(kv_dim) * sz(dim), xb, kv_dim, dim);
        kernels::matvec(v, w_.wv + layer * sz(kv_dim) * sz(dim), xb, kv_dim, dim);

        // Rotary position embedding on q and k (v is not rotated).
        kernels::rope(q, dim, pos, head_size);
        kernels::rope(k, kv_dim, pos, head_size);

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

            // Weighted sum of the value vectors; the head's output goes into xb.
            float* out = xb + sz(h) * sz(head_size);
            std::fill(out, out + head_size, 0.0f);
            for (int t = 0; t <= pos; ++t) {
                const float* vt = layer_values + sz(t) * sz(kv_dim) + sz(kv_head) * sz(head_size);
                kernels::axpy(out, scores[t], vt, head_size);
            }
        }

        // Output projection, then add to the residual stream.
        kernels::matvec(xb2, w_.wo + layer * sz(dim) * sz(dim), xb, dim, dim);
        kernels::add(x, xb2, dim);
        if (trace_) trace_("attn_out", l, x, dim);

        // ---------------- feed-forward block (SwiGLU) ----------------
        kernels::rmsnorm(xb, x, w_.rms_ffn + layer * sz(dim), dim);
        kernels::matvec(hb, w_.w1 + layer * sz(hidden) * sz(dim), xb, hidden, dim);   // gate
        kernels::matvec(hb2, w_.w3 + layer * sz(hidden) * sz(dim), xb, hidden, dim);  // up
        kernels::silu(hb, hidden);
        kernels::mul(hb, hb2, hidden);
        kernels::matvec(xb, w_.w2 + layer * sz(dim) * sz(hidden), hb, dim, hidden);   // down
        kernels::add(x, xb, dim);
        if (trace_) trace_("ffn_out", l, x, dim);
    }

    // 3. Final norm and classifier (for shared weights, wcls is the embedding table).
    kernels::rmsnorm(x, x, w_.rms_final, dim);
    if (trace_) trace_("final_norm", -1, x, dim);
    kernels::matvec(logits_.data(), w_.wcls, x, cfg_.vocab_size, dim);

    return logits_.data();
}

}  // namespace geneinfer
