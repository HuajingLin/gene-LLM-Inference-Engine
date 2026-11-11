#include "loader/quantize.h"

#include <cmath>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>

#include "common/q8.h"

namespace geneinfer {

namespace {

inline float widen_bf16(std::uint16_t b) {
    const std::uint32_t u = static_cast<std::uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &u, sizeof f);
    return f;
}

inline float element(const Mat& m, std::size_t i) {
    return m.is_bf16() ? widen_bf16(m.bf16[i]) : m.f32[i];
}

// The smallest bfloat16 pattern whose value is >= f (f finite, >= 0).
std::uint16_t bf16_round_up(float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, sizeof u);
    u = (u + 0xFFFFu) & 0xFFFF0000u;   // a carry out of the mantissa correctly bumps the exponent
    return static_cast<std::uint16_t>(u >> 16);
}

}  // namespace

Q8Matrix quantize_q8(const Mat& src, int rows, int cols) {
    if (rows <= 0 || cols <= 0) throw std::invalid_argument("quantize_q8: bad shape");
    if (src.is_q8() || (!src.is_bf16() && !src.f32)) throw std::invalid_argument("quantize_q8: source must be float32 or bfloat16");
    const int groups = q8_groups(cols);
    Q8Matrix m;
    m.q.assign(static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols), 0);
    m.scale.assign(static_cast<std::size_t>(rows) * static_cast<std::size_t>(groups), 0);

    for (int r = 0; r < rows; ++r) {
        const std::size_t row0 = static_cast<std::size_t>(r) * static_cast<std::size_t>(cols);
        for (int g = 0; g < groups; ++g) {
            const int begin = g * kQ8Group;
            const int end = begin + kQ8Group < cols ? begin + kQ8Group : cols;
            float amax = 0.0f;
            for (int c = begin; c < end; ++c) {
                const float v = element(src, row0 + static_cast<std::size_t>(c));
                if (!std::isfinite(v)) {
                    throw std::runtime_error("quantize_q8: non-finite weight in row " + std::to_string(r) +
                                             ", column " + std::to_string(c));
                }
                const float a = std::fabs(v);
                if (a > amax) amax = a;
            }
            if (amax == 0.0f) continue;   // scale 0, q 0
            const std::uint16_t sbits = bf16_round_up(amax / 127.0f);
            const float scale = widen_bf16(sbits);
            if (!(scale > 0.0f) || !std::isfinite(scale)) continue;   // below the float range: treat as zero
            m.scale[static_cast<std::size_t>(r) * static_cast<std::size_t>(groups) + static_cast<std::size_t>(g)] = sbits;
            for (int c = begin; c < end; ++c) {
                float q = std::nearbyint(element(src, row0 + static_cast<std::size_t>(c)) / scale);
                if (q > 127.0f) q = 127.0f;
                if (q < -127.0f) q = -127.0f;
                m.q[row0 + static_cast<std::size_t>(c)] = static_cast<std::int8_t>(q);
            }
        }
    }
    return m;
}

float dequantize_q8(const Q8Matrix& m, int cols, int row, int col) {
    const int groups = q8_groups(cols);
    const std::uint16_t s = m.scale[static_cast<std::size_t>(row) * static_cast<std::size_t>(groups) +
                                    static_cast<std::size_t>(col / kQ8Group)];
    return static_cast<float>(m.q[static_cast<std::size_t>(row) * static_cast<std::size_t>(cols) +
                                  static_cast<std::size_t>(col)]) * widen_bf16(s);
}

// ---- Checkpoint::quantize_matrices (lives here to keep checkpoint.cpp free of quantization details) ----
void Checkpoint::quantize_matrices() {
    const Config& c = cfg_;
    const std::size_t n_matrices = static_cast<std::size_t>(c.n_layers) * 7 + 1;
    q8_values_.reserve(n_matrices);   // the Mat views point into these vectors: no reallocation allowed
    q8_scales_.reserve(n_matrices);

    auto convert = [&](Mat& m, int rows, int cols) {
        Q8Matrix q = quantize_q8(m, rows, cols);
        q8_values_.push_back(std::move(q.q));
        q8_scales_.push_back(std::move(q.scale));
        m = Mat(q8_values_.back().data(), q8_scales_.back().data());
    };
    const int D = c.dim, H = c.hidden_dim, A = c.attn_dim(), KV = c.kv_dim();
    for (LayerWeights& lw : w_.layers) {
        convert(lw.wq, A, D);
        convert(lw.wk, KV, D);
        convert(lw.wv, KV, D);
        convert(lw.wo, D, A);
        convert(lw.w1, H, D);
        convert(lw.w2, D, H);
        convert(lw.w3, H, D);
    }
    convert(w_.wcls, c.vocab_size, D);   // a tied classifier is quantized as its own copy; the embedding stays
    int8_ = true;
}

}  // namespace geneinfer
