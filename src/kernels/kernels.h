// kernels.h - the numeric building blocks of a Llama-style forward pass.
//
// This is the "naive" baseline: plain scalar loops, written for clarity and
// correctness. Later stages (multithreading, SIMD, quantization) replace the
// implementations in kernels.cpp but keep these signatures, so the model code
// and the tests do not change.
//
// Conventions:
//   - all data is float32, contiguous
//   - weight matrices are row-major [rows, cols], so out = W * x
//   - "in place" functions overwrite their first argument
//   - no function allocates memory or throws; sizes are the caller's responsibility
#pragma once

#include "common/rope_style.h"

namespace geneinfer {
namespace kernels {

// out[r] = sum_c W[r * cols + c] * x[c],  r = 0..rows-1.
// `out` must not overlap W or x. This is the hot spot of decoding.
void matvec(float* out, const float* W, const float* x, int rows, int cols);

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

}  // namespace kernels
}  // namespace geneinfer
