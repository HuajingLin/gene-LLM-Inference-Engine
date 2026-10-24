// transformer.h - the Llama-style forward pass for a single token, with a KV cache.
//
// Usage:
//   Checkpoint ck = Checkpoint::load("model.bin");
//   Transformer model(ck);
//   for (int pos = 0; pos < n; ++pos) {
//       const float* logits = model.forward(tokens[pos], pos);   // logits for token pos+1
//   }
//
// Tokens must be fed at consecutive positions 0, 1, 2, ...: the key/value vectors of
// every earlier position live in the KV cache and are reused by later positions.
// To start a new sequence, simply start again at position 0 (old cache rows are
// overwritten before they are read).
#pragma once

#include <functional>
#include <vector>

#include "loader/checkpoint.h"

namespace geneinfer {

// Optional callback to inspect intermediate activations (used by tests and for debugging).
// `layer` is -1 for tensors that do not belong to a layer. Tags:
//   "embedding"  (dim)   token embedding, before layer 0
//   "attn_out"   (dim)   residual stream after the attention block of `layer`
//   "ffn_out"    (dim)   residual stream after the feed-forward block of `layer`
//   "final_norm" (dim)   after the final RMSNorm, right before the classifier
using TraceFn = std::function<void(const char* tag, int layer, const float* data, int size)>;

class Transformer {
public:
    // The weights are not copied: they point into the checkpoint's memory mapping, so the
    // Checkpoint object must stay alive as long as this Transformer is used.
    // `max_seq_len` caps the context length (and therefore the KV cache memory);
    // 0 means "use the model's seq_len". Throws std::invalid_argument if it exceeds it.
    explicit Transformer(const Checkpoint& checkpoint, int max_seq_len = 0);

    // Processes `token` at position `pos` and returns the logits (vocab_size floats) that
    // predict the next token. The returned pointer stays valid until the next forward() call.
    // Throws std::out_of_range for an invalid token id or position.
    const float* forward(int token, int pos);

    const Config& config() const { return cfg_; }
    int max_seq_len() const { return ctx_len_; }

    void set_trace(TraceFn fn) { trace_ = std::move(fn); }

private:
    Config cfg_;
    Weights w_;
    int ctx_len_;

    // Activation buffers (allocated once, reused for every token).
    std::vector<float> x_;      // [dim]       residual stream
    std::vector<float> xb_;     // [dim]       scratch (normalized input / attention output)
    std::vector<float> xb2_;    // [dim]       scratch (projection output)
    std::vector<float> hb_;     // [hidden]    FFN gate
    std::vector<float> hb2_;    // [hidden]    FFN up
    std::vector<float> q_;      // [dim]       query
    std::vector<float> att_;    // [n_heads, ctx_len] attention scores of the current token
    std::vector<float> logits_; // [vocab]

    // KV cache, layout [layer][position][kv_dim].
    std::vector<float> key_cache_;
    std::vector<float> value_cache_;

    TraceFn trace_;
};

}  // namespace geneinfer
