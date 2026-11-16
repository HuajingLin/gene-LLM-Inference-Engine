// model.h - what the generation loop and the tools need from a model, whatever computes it.
//
// Two implementations: Transformer (CPU, model/transformer.h) and GpuTransformer (a compute device,
// gpu/gpu_transformer.h). make_model() picks one by name.
#pragma once

#include <functional>
#include <memory>
#include <string>

#include "loader/checkpoint.h"

namespace geneinfer {

// Optional callback to inspect intermediate activations (used by tests and for debugging).
// `layer` is -1 for tensors that do not belong to a layer. Tags:
//   "embedding"  (dim)   token embedding, before layer 0
//   "attn_out"   (dim)   residual stream after the attention block of `layer`
//   "ffn_out"    (dim)   residual stream after the feed-forward block of `layer`
//   "final_norm" (dim)   after the final RMSNorm, right before the classifier
using TraceFn = std::function<void(const char* tag, int layer, const float* data, int size)>;

class Model {
public:
    virtual ~Model() = default;

    // Processes `token` at position `pos` and returns the logits (vocab_size floats) that predict the next token,
    // or nullptr if compute_logits is false. The pointer stays valid until the next call. Positions must be fed
    // without gaps; position 0 starts a new sequence, an earlier position rewinds to it.
    virtual const float* forward(int token, int pos, bool compute_logits = true) = 0;

    // n tokens at the positions pos0 .. pos0 + n - 1 in one go; returns the logits of the LAST token (or nullptr).
    virtual const float* forward_batch(const int* tokens, int n, int pos0, bool compute_logits = true) = 0;

    virtual const Config& config() const = 0;
    virtual int max_seq_len() const = 0;
    virtual void set_trace(TraceFn fn) = 0;
    virtual std::string description() const = 0;   // one line: where and how it computes

    // Optional per-operation timing (device models; the CPU model has ForwardProfile and gi_bench --profile).
    // While it is on, a device waits after every operation, which makes the whole pass a little slower.
    virtual void set_profiling(bool) {}
    virtual std::string profile_report() const { return std::string(); }   // empty when nothing was measured
};

// device: "cpu" (or empty), "cuda", "emu" (host emulation of the device kernels, for tests). An empty name means
// the environment variable GI_DEVICE, default "cpu". n_threads only applies to the CPU. Throws std::runtime_error
// for an unknown device, a build without CUDA, or weights the device cannot run (int8).
std::unique_ptr<Model> make_model(const Checkpoint& checkpoint, const std::string& device, int max_seq_len = 0,
                                  int n_threads = 0, int max_batch = 0);

}  // namespace geneinfer
