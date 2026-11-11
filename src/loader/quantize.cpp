#include "loader/quantize.h"

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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
namespace {

struct SkipSpec {
    bool role[8] = {};   // wq wk wv wo w1 w2 w3 wcls
    int first = 0, last = 0;
    std::vector<std::pair<int, int>> at;   // (role index, layer): one matrix of one layer
};

constexpr const char* kRoles[8] = {"wq", "wk", "wv", "wo", "w1", "w2", "w3", "wcls"};

SkipSpec parse_skip(const std::string& text) {
    SkipSpec spec;
    std::size_t pos = 0;
    while (pos <= text.size()) {
        std::size_t comma = text.find(',', pos);
        if (comma == std::string::npos) comma = text.size();
        std::string tok = text.substr(pos, comma - pos);
        pos = comma + 1;
        const std::size_t a = tok.find_first_not_of(" \t\r\n");   // spaces around an entry are ignored
        tok = a == std::string::npos ? std::string() : tok.substr(a, tok.find_last_not_of(" \t\r\n") - a + 1);
        if (tok.empty()) continue;
        bool known = false;
        const std::size_t at_pos = tok.find('@');
        if (at_pos != std::string::npos) {   // <role>@<layer>
            const std::string role = tok.substr(0, at_pos), num = tok.substr(at_pos + 1);
            char* end = nullptr;
            const long layer = std::strtol(num.c_str(), &end, 10);
            for (int i = 0; i < 7; ++i) {   // wcls has no layer
                if (role == kRoles[i] && !num.empty() && *end == '\0' && layer >= 0 && layer <= 100000) {
                    spec.at.emplace_back(i, static_cast<int>(layer));
                    known = true;
                }
            }
        }
        for (int i = 0; i < 8 && !known; ++i) {
            if (tok == kRoles[i]) { spec.role[i] = true; known = true; }
        }
        if (!known && (tok.rfind("first", 0) == 0 || tok.rfind("last", 0) == 0)) {
            const bool is_first = tok[0] == 'f';
            const std::string num = tok.substr(is_first ? 5 : 4);
            char* end = nullptr;
            const long n = std::strtol(num.c_str(), &end, 10);
            if (!num.empty() && *end == '\0' && n >= 0 && n <= 100000) {
                (is_first ? spec.first : spec.last) = static_cast<int>(n);
                known = true;
            }
        }
        if (!known) {
            throw std::invalid_argument("GI_Q8_SKIP: unknown entry '" + tok +
                                        "' (expected wq wk wv wo w1 w2 w3 wcls, <role>@<layer>, first<N>, last<N>)");
        }
    }
    return spec;
}

}  // namespace

void Checkpoint::quantize_matrices(const std::string& skip_text) {
    const SkipSpec skip = parse_skip(skip_text);
    const Config& c = cfg_;
    const std::size_t n_matrices = static_cast<std::size_t>(c.n_layers) * 7 + 1;
    q8_values_.reserve(n_matrices);   // the Mat views point into these vectors: no reallocation allowed
    q8_scales_.reserve(n_matrices);
    matrix_bytes_ = 0;

    for (const auto& entry : skip.at) {
        if (entry.second >= c.n_layers) {
            throw std::invalid_argument("GI_Q8_SKIP: layer " + std::to_string(entry.second) + " does not exist (the model has " +
                                        std::to_string(c.n_layers) + " layers)");
        }
    }
    auto at_layer = [&](int role, int layer) {
        for (const auto& entry : skip.at) {
            if (entry.first == role && entry.second == layer) return true;
        }
        return false;
    };
    auto convert = [&](Mat& m, int rows, int cols, bool keep) {
        const std::size_t count = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
        if (keep) {
            matrix_bytes_ += count * (m.is_bf16() ? 2 : 4);
            return;
        }
        Q8Matrix q = quantize_q8(m, rows, cols);
        matrix_bytes_ += q.q.size() + q.scale.size() * sizeof(std::uint16_t);
        q8_values_.push_back(std::move(q.q));
        q8_scales_.push_back(std::move(q.scale));
        m = Mat(q8_values_.back().data(), q8_scales_.back().data());
    };
    const int D = c.dim, H = c.hidden_dim, A = c.attn_dim(), KV = c.kv_dim();
    for (int l = 0; l < c.n_layers; ++l) {
        LayerWeights& lw = w_.layers[static_cast<std::size_t>(l)];
        const bool layer_kept = l < skip.first || l >= c.n_layers - skip.last;
        convert(lw.wq, A, D, layer_kept || skip.role[0] || at_layer(0, l));
        convert(lw.wk, KV, D, layer_kept || skip.role[1] || at_layer(1, l));
        convert(lw.wv, KV, D, layer_kept || skip.role[2] || at_layer(2, l));
        convert(lw.wo, D, A, layer_kept || skip.role[3] || at_layer(3, l));
        convert(lw.w1, H, D, layer_kept || skip.role[4] || at_layer(4, l));
        convert(lw.w2, D, H, layer_kept || skip.role[5] || at_layer(5, l));
        convert(lw.w3, H, D, layer_kept || skip.role[6] || at_layer(6, l));
    }
    convert(w_.wcls, c.vocab_size, D, skip.role[7]);   // a tied classifier is quantized as its own copy
    int8_ = true;
}

}  // namespace geneinfer
