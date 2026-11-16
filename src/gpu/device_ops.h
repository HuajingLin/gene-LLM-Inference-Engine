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

// One layer of a paged KV cache on the device (model/kv_cache.h): the pool is [n_blocks][block_floats] floats, and
// inside a block every layer has block_size rows of keys followed by block_size rows of values, kv_dim floats each.
struct PagedKv {
    float* pool;
    const int* tables;      // [rows, table_stride]: the block ids of each row's sequence
    const int* positions;   // [rows]: the position of each row in its sequence
    long table_stride;
    long block_floats;      // floats of one block (all layers, keys and values)
    long layer_offset;      // floats from the start of a block to this layer's keys
    int block_size, kv_dim;
};

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
    // weight vector is used for every vector. out == x is allowed. `partial` is scratch for kPartials * nvec floats.
    void (*rmsnorm)(float* out, const float* x, const float* weight, int nvec, int size, float eps, float* partial);

    // out[t * out_stride + r] = sum_c W[r, c] * x[t * x_stride + c] for t < ntok, r < rows. `partial` is scratch
    // for kPartials * ntok * rows floats. The value of every output element does not depend on ntok.
    // Three kernels compute the partial sums, chosen by the environment variable GI_DEV_MATMUL: narrow, wide or tile
    // (the default); see device_kernels.inc. wide and tile give the
    // same bits; narrow adds in another order, so its results differ in the last bits.
    void (*matmul)(float* out, long out_stride, const void* W, int bf16, const float* x, long x_stride, int ntok,
                   int rows, int cols, float* partial);

    // Rotary embedding in place: row i (v + i * stride, n values = whole heads of head_size) is rotated with
    // row i of the cos / sin tables ([m, head_size / 2]).
    void (*rope)(float* v, long stride, int m, int n, const float* cos_rows, const float* sin_rows, int head_size,
                 int half_split);

    // Stores the new keys and values of m rows (k, v: [m, kv_dim]) into their blocks: row i goes to position
    // kv.positions[i] of the sequence whose block table is row i of kv.tables.
    void (*store_kv)(const PagedKv& kv, const float* k, const float* v, int m);

    // Paged attention of m rows over one layer of the block pool. Row i attends to the positions
    // 0 .. kv.positions[i] of its sequence, found through its block table. q and out are [m, n_heads * head_size];
    // query head h uses KV head h / kv_mul. `scores` is scratch for m * n_heads * max_len floats, where max_len is
    // the largest position + 1. The result of a row depends neither on the block size nor on where its blocks are.
    void (*attention)(float* out, const float* q, const PagedKv& kv, int m, int max_len, int n_heads, int head_size,
                      int kv_mul, float inv_scale, float* scores);

    void (*copy)(void* dst, const void* src, std::size_t bytes);   // device -> device (ordered after the kernels)

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
