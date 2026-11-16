// device_ops.h - the operations of the forward pass on a compute device (a GPU).
//
// A device owns its memory: pointers returned by alloc() are DEVICE pointers. The host never reads or writes
// through them; data moves with upload() / download(), and every operation below takes and returns device
// pointers. Two implementations fill this table from the SAME kernel source (gpu/device_kernels.inc):
//   cuda_ops()  CUDA: each kernel body runs as one GPU thread per index (built with -DGI_CUDA=ON)
//   emu_ops()   host emulation: the same bodies run in a plain loop on the CPU, "device memory" is the heap
// The emulation exists to test the kernels where there is no GPU; it is slow and not meant for real use.
//
// All matrices are row-major [rows, cols]; activations are float32. Weight matrices are float32 or bfloat16
// (widened exactly inside the kernel); int8 weights are not supported on a device yet.
#pragma once

#include <cstddef>
#include <string>

namespace geneinfer {
namespace gpu {

struct DeviceOps {
    const char* name;   // "cuda" or "emu"

    void* (*alloc)(std::size_t bytes);    // throws std::runtime_error when the device is out of memory
    void (*release)(void* p);
    void (*upload)(void* dst, const void* src, std::size_t bytes);     // host -> device
    void (*download)(void* dst, const void* src, std::size_t bytes);   // device -> host (waits for the kernels)
    void (*sync)();                                                    // waits until all launched kernels are done
    std::string (*describe)();                                         // device name and memory, for messages

    // x[i, :] = table[tokens[i], :] for i < m (widened when the table is bfloat16).
    void (*embed)(float* x, const void* table, int bf16, const int* tokens, int m, int dim);

    // RMSNorm of nvec consecutive vectors of `size` values: out = weight * x / sqrt(mean(x^2) + eps). The same
    // weight vector is used for every vector. out == x is allowed.
    void (*rmsnorm)(float* out, const float* x, const float* weight, int nvec, int size, float eps);

    // out[t * out_stride + r] = sum_c W[r, c] * x[t * x_stride + c] for t < ntok, r < rows. `partial` is scratch
    // for kPartials * ntok * rows floats. The value of every output element does not depend on ntok.
    // Two kernels compute the partial sums: "narrow" (a thread reads one weight per step) and "wide" (a thread reads
    // 16 bytes of weights per step: 8 bfloat16 or 4 float32 values). The environment variable GI_DEV_MATMUL picks
    // one (default narrow); wide needs cols to be a multiple of 8 (bf16) or 4 (f32) and otherwise falls back. The
    // two add in a different order, so their results differ in the last bits.
    void (*matmul)(float* out, long out_stride, const void* W, int bf16, const float* x, long x_stride, int ntok,
                   int rows, int cols, float* partial);

    // Rotary embedding in place: row i (v + i * stride, n values = whole heads of head_size) is rotated with
    // row i of the cos / sin tables ([m, head_size / 2]).
    void (*rope)(float* v, long stride, int m, int n, const float* cos_rows, const float* sin_rows, int head_size,
                 int half_split);

    // Attention of m rows over one layer of a contiguous KV cache (keys, values: [positions, kv_dim]). Row i is
    // position pos0 + i and attends to positions 0 .. pos0 + i. q and out are [m, n_heads * head_size]; query
    // head h uses KV head h / kv_mul. `scores` is scratch for m * n_heads * (pos0 + m) floats.
    void (*attention)(float* out, const float* q, const float* keys, const float* values, int m, int pos0,
                      int n_heads, int head_size, int kv_mul, int kv_dim, float inv_scale, float* scores);

    void (*add)(float* a, const float* b, long n);        // a += b
    void (*silu_mul)(float* a, const float* b, long n);   // a = silu(a) * b
};

constexpr int kPartials = 32;        // partial sums per output element of matmul (one warp of GPU threads)
constexpr int kTokensPerPass = 8;   // tokens of a matmul that share one pass over the weights

const DeviceOps* emu_ops();    // always available
const DeviceOps* cuda_ops();   // nullptr when the library was built without CUDA

// "cuda", "emu" -> the table; throws std::runtime_error for an unknown name or when CUDA is not compiled in.
const DeviceOps* device_by_name(const std::string& name);

}  // namespace gpu
}  // namespace geneinfer
