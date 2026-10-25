// sampler.h - turns logits into the next token id.
//
// Semantics (documented precisely because tests/verify_sampler.py checks them exactly):
//   1. temperature <= 0            -> greedy: the largest logit wins (lowest index on ties).
//   2. otherwise probs = softmax(logits / temperature).
//   3. Filtering is applied to the tokens sorted by probability (descending, lower index
//      first on ties):
//        top_k > 0   keep only the k most likely tokens (k >= vocab_size means no limit);
//        top_p < 1   among those, keep the smallest prefix whose probability mass reaches
//                    top_p of the (top_k-renormalized) mass; at least one token is kept.
//   4. A token is drawn from the kept tokens, renormalized, by inverse-CDF sampling with a
//      uniform number u in [0, 1): walk the kept tokens in order and return the first one
//      whose cumulative probability exceeds u * (total kept probability).
//      If no filtering is active, tokens are walked in index order (no sort is needed).
#pragma once

#include <cstdint>
#include <vector>

namespace geneinfer {

struct SamplerConfig {
    float temperature = 0.8f;  // <= 0 means greedy decoding
    int top_k = 0;             // 0 = disabled
    float top_p = 1.0f;        // 1 = disabled; must be in (0, 1]
    std::uint64_t seed = 1;
};

class Sampler {
public:
    // Throws std::invalid_argument for negative temperature/top_k or top_p outside (0, 1].
    Sampler(const SamplerConfig& config, int vocab_size);

    // Draws the uniform random number from the internal generator (xorshift64*).
    int sample(const float* logits);

    // Deterministic variant: `u` in [0, 1) is supplied by the caller. Ignored when greedy.
    int sample(const float* logits, float u);

    // Next uniform float in [0, 1) from the internal generator.
    float next_uniform();

    const SamplerConfig& config() const { return cfg_; }

private:
    std::uint32_t next_u32();

    SamplerConfig cfg_;
    int vocab_;
    std::uint64_t state_;
    std::vector<float> probs_;  // scratch: probabilities of the current call
    std::vector<int> order_;    // scratch: token indices (sorted when filtering)
};

}  // namespace geneinfer
