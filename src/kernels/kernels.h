// kernels.h - the numeric building blocks of a Llama-style forward pass.
//
// Every kernel has a plain scalar implementation (the reference, written for clarity) and, on x86-64,
// AVX2+FMA and AVX-512 versions. The best one the CPU supports is chosen at startup; the choice can be
// forced with the environment variable GI_ISA=scalar|avx2|avx512 or with set_isa(). The signatures never
// change, so the model code and the tests do not depend on the instruction set. The SIMD versions sum in a
// different order than the scalar loops, so results agree to float32 rounding, not bit for bit.
//
// Conventions:
//   - all data is float32, contiguous
//   - weight matrices are row-major [rows, cols], so out = W * x
//   - "in place" functions overwrite their first argument
//   - no function allocates memory or throws; sizes are the caller's responsibility
#pragma once

#include <cstdint>

#include "common/rope_style.h"

namespace geneinfer {
namespace kernels {

enum class Isa { Scalar, Avx2, Avx512 };

const char* isa_name(Isa isa);       // "scalar", "avx2", "avx512"
bool isa_supported(Isa isa);         // compiled in AND supported by this CPU/OS (Scalar: always)
Isa best_isa();                      // fastest supported
Isa active_isa();                    // what the kernels below use right now
// Switches the kernels; returns false (and changes nothing) if `isa` is not supported. Not thread-safe:
// call it before any other thread uses the kernels.
bool set_isa(Isa isa);

// out[r] = sum_c W[r * cols + c] * x[c],  r = 0..rows-1.
// `out` must not overlap W or x. This is the hot spot of decoding.
void matvec(float* out, const float* W, const float* x, int rows, int cols);

// Same product with a bfloat16 weight matrix: W holds the upper 16 bits of float32 values, row-major
// [rows, cols]. Widening to float32 is exact and the summation order is the same as in matvec(), so the result
// is bit-identical to matvec() on the float32-expanded matrix (for the same instruction set). The weights
// take half the memory bandwidth, which is what limits decoding.
void matvec_bf16(float* out, const std::uint16_t* W, const float* x, int rows, int cols);

// Same product with an int8 weight matrix (see common/q8.h): W is [rows, cols] int8 and `scales` holds the bf16
// scale of every group of 64 values of a row, [rows, ceil(cols / 64)]. Per group the products q * x are summed
// in float32 and the group sum is multiplied by the group scale; the activations stay float32. Every row is
// computed by the same instruction sequence wherever it sits in the matrix, so computing a matrix in pieces
// (slices of rows, for threads) gives bit-identical results. The weights take about 1/4 of the bandwidth of
// float32 and half of bfloat16.
void matvec_q8(float* out, const std::int8_t* W, const std::uint16_t* scales, const float* x, int rows, int cols);

// dst[i] = float32(src[i]) for bfloat16 values (one embedding-table row, say).
void widen_bf16(float* dst, const std::uint16_t* src, int n);

// RMSNorm: out[i] = weight[i] * x[i] / sqrt(mean(x^2) + eps).
// `out` may be the same pointer as `x` (in-place is safe).
void rmsnorm(float* out, const float* x, const float* weight, int n, float eps = 1e-5f);

// Numerically stable softmax, in place: x[i] = exp(x[i] - max) / sum(exp(x - max)).
void softmax(float* x, int n);

// SiLU (a.k.a. swish), in place: x[i] = x[i] * sigmoid(x[i]).
void silu(float* x, int n);

// Returns sum_i a[i] * b[i]  (attention scores: q . k).
float dot(const float* a, const float* b, int n);

// y[i] += alpha * x[i]  (attention output: weighted sum of value vectors).
// `y` must not overlap `x`.
void axpy(float* y, float alpha, const float* x, int n);

// a[i] *= b[i]   (the "gate * up" step of the SwiGLU feed-forward block)
void mul(float* a, const float* b, int n);

// a[i] += b[i]   (residual connections)
void add(float* a, const float* b, int n);

// Rotary position embedding (RoPE), in place. Every head (head_size consecutive floats) is
// treated as head_size/2 two-dimensional vectors; vector k is rotated by the angle
// pos * theta^(-2k / head_size). `style` decides which two elements form a vector:
//   Interleaved: (v[2k], v[2k+1])            llama2.c
//   HalfSplit:   (v[k], v[k + head_size/2])  Hugging Face Llama
// `n` is the full vector length (attn_dim for q, kv_dim for k) and must be a multiple of
// head_size; head_size must be even.
void rope(float* v, int n, int pos, int head_size, float theta = 10000.0f,
          RopeStyle style = RopeStyle::Interleaved);

// rope() in two steps, so that the trigonometry can be done once per position instead of once per head, layer
// and call. rope_table_row() computes cos and sin of the head_size/2 rotation angles of position `pos`;
// rope_apply() rotates v with them. rope_apply(rope_table_row(pos)) gives exactly the bits of rope(pos).
void rope_table_row(float* cos_out, float* sin_out, int pos, int head_size, float theta = 10000.0f);
void rope_apply(float* v, int n, const float* cos_row, const float* sin_row, int head_size,
                RopeStyle style = RopeStyle::Interleaved);

}  // namespace kernels
}  // namespace geneinfer
