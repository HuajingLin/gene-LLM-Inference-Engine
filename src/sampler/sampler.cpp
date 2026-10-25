#include "sampler/sampler.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <string>

namespace geneinfer {

namespace {

// splitmix64: spreads any seed (including 0 and small integers) over all 64 bits, so that
// nearby seeds give unrelated random streams.
std::uint64_t splitmix64(std::uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

}  // namespace

Sampler::Sampler(const SamplerConfig& config, int vocab_size) : cfg_(config), vocab_(vocab_size) {
    if (vocab_size <= 0) throw std::invalid_argument("sampler: vocab_size must be positive");
    if (!(config.temperature >= 0.0f)) {
        throw std::invalid_argument("sampler: temperature must be >= 0");
    }
    if (config.top_k < 0) throw std::invalid_argument("sampler: top_k must be >= 0");
    if (!(config.top_p > 0.0f && config.top_p <= 1.0f)) {
        throw std::invalid_argument("sampler: top_p must be in (0, 1]");
    }
    state_ = splitmix64(config.seed);
    if (state_ == 0) state_ = 1;  // xorshift must never be in the all-zero state

    probs_.resize(static_cast<std::size_t>(vocab_size));
    order_.resize(static_cast<std::size_t>(vocab_size));
}

std::uint32_t Sampler::next_u32() {
    // xorshift64*, the same generator as llama2.c's run.c.
    state_ ^= state_ >> 12;
    state_ ^= state_ << 25;
    state_ ^= state_ >> 27;
    return static_cast<std::uint32_t>((state_ * 0x2545F4914F6CDD1Dull) >> 32);
}

float Sampler::next_uniform() {
    // 24 random bits -> exactly representable float in [0, 1).
    return static_cast<float>(next_u32() >> 8) / 16777216.0f;
}

int Sampler::sample(const float* logits) {
    if (cfg_.temperature <= 0.0f) return sample(logits, 0.0f);  // greedy: consumes no randomness
    return sample(logits, next_uniform());
}

int Sampler::sample(const float* logits, float u) {
    const std::size_t n = static_cast<std::size_t>(vocab_);

    // ---- greedy ---------------------------------------------------------------------
    if (cfg_.temperature <= 0.0f) {
        std::size_t best = 0;
        for (std::size_t i = 1; i < n; ++i) {
            if (logits[i] > logits[best]) best = i;  // strict ">" keeps the lowest index on ties
        }
        return static_cast<int>(best);
    }

    // ---- probabilities: softmax(logits / temperature) --------------------------------
    float max_logit = logits[0];
    for (std::size_t i = 1; i < n; ++i) max_logit = std::max(max_logit, logits[i]);
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        probs_[i] = std::exp((logits[i] - max_logit) / cfg_.temperature);
        sum += probs_[i];
    }
    const float inv_sum = static_cast<float>(1.0 / sum);
    for (std::size_t i = 0; i < n; ++i) probs_[i] *= inv_sum;

    std::iota(order_.begin(), order_.end(), 0);
    std::size_t kept = n;  // number of leading entries of order_ that stay candidates

    // ---- filtering (top-k, then top-p) ---------------------------------------------------
    const bool use_top_k = cfg_.top_k > 0 && static_cast<std::size_t>(cfg_.top_k) < n;
    const bool use_top_p = cfg_.top_p < 1.0f;
    if (use_top_k || use_top_p) {
        const auto by_prob_desc = [this](int a, int b) {
            const float pa = probs_[static_cast<std::size_t>(a)];
            const float pb = probs_[static_cast<std::size_t>(b)];
            return pa > pb || (pa == pb && a < b);
        };
        if (use_top_k) {
            kept = static_cast<std::size_t>(cfg_.top_k);
            std::partial_sort(order_.begin(), order_.begin() + static_cast<std::ptrdiff_t>(kept),
                              order_.end(), by_prob_desc);
        } else {
            std::sort(order_.begin(), order_.end(), by_prob_desc);
        }

        if (use_top_p) {
            double total = 0.0;
            for (std::size_t i = 0; i < kept; ++i) total += probs_[static_cast<std::size_t>(order_[i])];
            const double threshold = static_cast<double>(cfg_.top_p) * total;
            double cum = 0.0;
            std::size_t nucleus = kept;  // if rounding prevents reaching the threshold, keep all
            for (std::size_t i = 0; i < kept; ++i) {
                cum += probs_[static_cast<std::size_t>(order_[i])];
                if (cum >= threshold) {
                    nucleus = i + 1;
                    break;
                }
            }
            kept = nucleus;
        }
    }

    // ---- inverse-CDF draw over the kept tokens -------------------------------------------
    double total = 0.0;
    for (std::size_t i = 0; i < kept; ++i) total += probs_[static_cast<std::size_t>(order_[i])];
    const double r = static_cast<double>(u) * total;
    double cum = 0.0;
    for (std::size_t i = 0; i < kept; ++i) {
        cum += probs_[static_cast<std::size_t>(order_[i])];
        if (r < cum) return order_[i];
    }
    return order_[kept - 1];  // rounding guard: u was (almost) 1
}

}  // namespace geneinfer
