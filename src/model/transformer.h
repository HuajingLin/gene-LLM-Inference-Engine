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

#include <chrono>
#include <memory>
#include <functional>
#include <vector>

#include "loader/checkpoint.h"
#include "util/thread_pool.h"

namespace geneinfer {

// Optional callback to inspect intermediate activations (used by tests and for debugging).
// `layer` is -1 for tensors that do not belong to a layer. Tags:
//   "embedding"  (dim)   token embedding, before layer 0
//   "attn_out"   (dim)   residual stream after the attention block of `layer`
//   "ffn_out"    (dim)   residual stream after the feed-forward block of `layer`
//   "final_norm" (dim)   after the final RMSNorm, right before the classifier
using TraceFn = std::function<void(const char* tag, int layer, const float* data, int size)>;

// Optional per-section timing of forward(): attach one with Transformer::set_profile(). Each section is the
// time between two checkpoints of the forward pass, so the sections add up to the whole call. Costs one clock
// read per section (a few microseconds per token); without a profile attached there is no cost but a branch.
struct ForwardProfile {
    enum Section { Embed, Norm, QkvProj, Rope, Attention, OutProj, FfnGateUp, Activation, FfnDown, Residual,
                   Classifier, kSections };
    static const char* name(int s);
    static bool is_matvec(int s) { return s == QkvProj || s == OutProj || s == FfnGateUp || s == FfnDown || s == Classifier; }

    double seconds[kSections] = {};
    long forward_calls = 0;
    std::chrono::steady_clock::time_point last;
};

class Transformer {
public:
    // The weights are not copied: they point into the checkpoint's memory mapping, so the
    // Checkpoint object must stay alive as long as this Transformer is used.
    // `max_seq_len` caps the context length (and therefore the KV cache memory);
    // 0 means "use the model's seq_len". Throws std::invalid_argument if it exceeds it.
    //
    // `n_threads` is the number of threads that share the weight-matrix products (the calling thread counts):
    // 1 = no extra threads, 0 = use the environment variable GI_THREADS (default 1). The results do not depend
    // on the number of threads: every output element is computed by the same instructions in the same order.
    //
    // `max_batch` is the most tokens forward_batch() processes together (longer inputs are cut into pieces of
    // this size): 0 = the environment variable GI_BATCH (default 64). It only sets the size of the activation
    // buffers and of the cache-sized tiles; results do not depend on it.
    explicit Transformer(const Checkpoint& checkpoint, int max_seq_len = 0, int n_threads = 0, int max_batch = 0);
    ~Transformer();
    int threads() const;
    int max_batch() const { return max_batch_; }

    // Processes `token` at position `pos` and returns the logits (vocab_size floats) that
    // predict the next token. The returned pointer stays valid until the next forward() call.
    // Throws std::out_of_range for an invalid token id or position.
    //
    // While feeding a prompt, the logits of all but the last position are never used. Pass
    // compute_logits = false for those: the final norm and the (large) classifier matrix
    // product are skipped and nullptr is returned. The KV cache is updated either way.
    const float* forward(int token, int pos, bool compute_logits = true);

    // Processes n tokens at the consecutive positions pos0 .. pos0+n-1 in one go (prompt processing, "prefill")
    // and returns the logits of the LAST token if compute_logits is true, otherwise nullptr. The KV cache ends up
    // exactly as if forward() had been called for each token, and so does everything computed from it: the
    // result is BIT-IDENTICAL to the token-by-token calls. It is faster because the weights are read from memory
    // once per tile of rows and then reused from cache for all n tokens, instead of once per token; decoding is
    // limited by memory bandwidth, prefill with batching by arithmetic.
    // pos0 may be > 0 when earlier positions are already in the cache. While a trace callback is set, the tokens
    // are processed one by one so that the trace sees every token (same results, no speedup).
    const float* forward_batch(const int* tokens, int n, int pos0, bool compute_logits = true);

    const Config& config() const { return cfg_; }
    int max_seq_len() const { return ctx_len_; }

    // Memory owned by this object (the weights are not included: they live in the mmap).
    std::size_t kv_cache_bytes() const;  // keys + values for all layers and positions
    std::size_t scratch_bytes() const;   // activation buffers, attention scores, logits

    void set_trace(TraceFn fn) { trace_ = std::move(fn); }
    void set_profile(ForwardProfile* profile) { profile_ = profile; }   // nullptr switches it off

private:
    struct MatvecJob {
        float* out;
        Mat W;
        const float* x;
        int rows, cols;
    };
    void run_matvecs(const MatvecJob* jobs, int n);

    // out[t * out_stride ..] = W * x[t * x_stride ..] for t = 0 .. ntok-1 (several tokens at once, see forward_batch).
    struct MatmulJob {
        float* out;
        Mat W;
        const float* x;
        int rows, cols;
        std::size_t out_stride, x_stride;
    };
    void run_matmuls(const MatmulJob* jobs, int n, int ntok);
    void batch_chunk(const int* tokens, int m, int pos0, bool compute_logits);
    void ensure_rope(int pos);   // cos/sin rows for positions 0..pos

    int max_batch_ = 64;
    std::vector<float> bx_, bxb_, bxb2_, bhb_, bhb2_, bq_, bao_;   // [max_batch, ...] activations of forward_batch
    std::vector<float> batt_;                                       // [threads, ctx_len] attention scores (decode and batch)
    std::vector<float> rope_cos_, rope_sin_;                        // [rows, head_size / 2], filled on demand
    int rope_rows_ = 0;

    std::unique_ptr<ThreadPool> pool_;      // null when running on one thread
    long par_min_weights_ = 32768;          // a product with fewer weights per thread is not split
    ForwardProfile* profile_ = nullptr;
    Config cfg_;
    Weights w_;
    int ctx_len_;

    // Activation buffers (allocated once, reused for every token).
    std::vector<float> x_;      // [dim]       residual stream
    std::vector<float> xb_;     // [dim]       scratch (normalized input, FFN output)
    std::vector<float> xb2_;    // [dim]       scratch (projection output)
    std::vector<float> hb_;     // [hidden]    FFN gate
    std::vector<float> hb2_;    // [hidden]    FFN up
    std::vector<float> q_;      // [attn_dim]  query
    std::vector<float> ao_;     // [attn_dim]  attention output (all heads concatenated)
    std::vector<float> logits_; // [vocab]

    // KV cache, layout [layer][position][kv_dim].
    std::vector<float> key_cache_;
    std::vector<float> value_cache_;

    TraceFn trace_;
};

}  // namespace geneinfer
