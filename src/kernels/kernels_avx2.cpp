// kernels_avx2.cpp - AVX2 + FMA versions of the hot kernels. Compiled with -mavx2 -mfma (see CMakeLists.txt).
//
// Scalar loop tails use __builtin_fmaf explicitly: whether the compiler fuses `s += a * b` on its own is
// its choice, and the float32 and bfloat16 kernels must round identically.
//
// IMPORTANT: this file must not use inline std:: templates or functions (std::min, std::vector, ...):
// the compiler could emit them with AVX2 instructions and the linker might pick that copy for code that also
// runs on CPUs without AVX2. Only intrinsics and plain loops here.
#if defined(GI_HAVE_X86_SIMD)

#include <immintrin.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "kernels/kernel_table.h"

namespace geneinfer {
namespace kernels {
namespace {

inline float hsum(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
    return _mm_cvtss_f32(lo);
}

// Sum of a[i]*b[i] with 4 independent accumulators: one FMA has a latency of ~4 cycles but two can start
// per cycle, so at least 8 independent chains are needed to keep both units busy (4 here, 8 in matvec).
float dot_avx2(const float* a, const float* b, int n) {
    __m256 s0 = _mm256_setzero_ps(), s1 = s0, s2 = s0, s3 = s0;
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), s1);
        s2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), s2);
        s3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), s3);
    }
    for (; i + 8 <= n; i += 8) s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
    float sum = hsum(_mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3)));
    for (; i < n; ++i) sum = __builtin_fmaf(a[i], b[i], sum);
    return sum;
}

// Four rows at a time: x is loaded once and used for four rows, and 4 rows x 2 vectors = 8 independent
// accumulators hide the FMA latency.
void matvec_avx2(float* out, const float* W, const float* x, int rows, int cols) {
    const std::size_t stride = static_cast<std::size_t>(cols);
    int r = 0;
    for (; r + 4 <= rows; r += 4) {
        const float* w0 = W + static_cast<std::size_t>(r) * stride;
        const float* w1 = w0 + stride;
        const float* w2 = w1 + stride;
        const float* w3 = w2 + stride;
        __m256 a00 = _mm256_setzero_ps(), a01 = a00, a10 = a00, a11 = a00;
        __m256 a20 = a00, a21 = a00, a30 = a00, a31 = a00;
        int c = 0;
        for (; c + 16 <= cols; c += 16) {
            const __m256 xa = _mm256_loadu_ps(x + c);
            const __m256 xb = _mm256_loadu_ps(x + c + 8);
            a00 = _mm256_fmadd_ps(_mm256_loadu_ps(w0 + c), xa, a00);
            a01 = _mm256_fmadd_ps(_mm256_loadu_ps(w0 + c + 8), xb, a01);
            a10 = _mm256_fmadd_ps(_mm256_loadu_ps(w1 + c), xa, a10);
            a11 = _mm256_fmadd_ps(_mm256_loadu_ps(w1 + c + 8), xb, a11);
            a20 = _mm256_fmadd_ps(_mm256_loadu_ps(w2 + c), xa, a20);
            a21 = _mm256_fmadd_ps(_mm256_loadu_ps(w2 + c + 8), xb, a21);
            a30 = _mm256_fmadd_ps(_mm256_loadu_ps(w3 + c), xa, a30);
            a31 = _mm256_fmadd_ps(_mm256_loadu_ps(w3 + c + 8), xb, a31);
        }
        for (; c + 8 <= cols; c += 8) {
            const __m256 xa = _mm256_loadu_ps(x + c);
            a00 = _mm256_fmadd_ps(_mm256_loadu_ps(w0 + c), xa, a00);
            a10 = _mm256_fmadd_ps(_mm256_loadu_ps(w1 + c), xa, a10);
            a20 = _mm256_fmadd_ps(_mm256_loadu_ps(w2 + c), xa, a20);
            a30 = _mm256_fmadd_ps(_mm256_loadu_ps(w3 + c), xa, a30);
        }
        float s0 = hsum(_mm256_add_ps(a00, a01)), s1 = hsum(_mm256_add_ps(a10, a11));
        float s2 = hsum(_mm256_add_ps(a20, a21)), s3 = hsum(_mm256_add_ps(a30, a31));
        for (; c < cols; ++c) {
            const float xv = x[c];
            s0 = __builtin_fmaf(w0[c], xv, s0);
            s1 = __builtin_fmaf(w1[c], xv, s1);
            s2 = __builtin_fmaf(w2[c], xv, s2);
            s3 = __builtin_fmaf(w3[c], xv, s3);
        }
        out[r] = s0;
        out[r + 1] = s1;
        out[r + 2] = s2;
        out[r + 3] = s3;
    }
    for (; r < rows; ++r) out[r] = dot_avx2(W + static_cast<std::size_t>(r) * stride, x, cols);
}

// ---- bfloat16 weights: the same kernels, widening W on the fly (16-bit load, zero-extend, shift left 16) ----
// The structure and the order of every addition mirror dot_avx2 / matvec_avx2 exactly, so the results are
// bit-identical to the float32 versions on the expanded matrix.
inline __m256 load_bf16x8(const std::uint16_t* p) {
    return _mm256_castsi256_ps(
        _mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p))), 16));
}

inline float widen1(std::uint16_t b) {
    const std::uint32_t u = static_cast<std::uint32_t>(b) << 16;
    float f;
    __builtin_memcpy(&f, &u, sizeof f);
    return f;
}

float dot_bf16_avx2(const std::uint16_t* a, const float* b, int n) {
    __m256 s0 = _mm256_setzero_ps(), s1 = s0, s2 = s0, s3 = s0;
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        s0 = _mm256_fmadd_ps(load_bf16x8(a + i), _mm256_loadu_ps(b + i), s0);
        s1 = _mm256_fmadd_ps(load_bf16x8(a + i + 8), _mm256_loadu_ps(b + i + 8), s1);
        s2 = _mm256_fmadd_ps(load_bf16x8(a + i + 16), _mm256_loadu_ps(b + i + 16), s2);
        s3 = _mm256_fmadd_ps(load_bf16x8(a + i + 24), _mm256_loadu_ps(b + i + 24), s3);
    }
    for (; i + 8 <= n; i += 8) s0 = _mm256_fmadd_ps(load_bf16x8(a + i), _mm256_loadu_ps(b + i), s0);
    float sum = hsum(_mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3)));
    for (; i < n; ++i) sum = __builtin_fmaf(widen1(a[i]), b[i], sum);
    return sum;
}

void matvec_bf16_avx2(float* out, const std::uint16_t* W, const float* x, int rows, int cols) {
    const std::size_t stride = static_cast<std::size_t>(cols);
    int r = 0;
    for (; r + 4 <= rows; r += 4) {
        const std::uint16_t* w0 = W + static_cast<std::size_t>(r) * stride;
        const std::uint16_t* w1 = w0 + stride;
        const std::uint16_t* w2 = w1 + stride;
        const std::uint16_t* w3 = w2 + stride;
        __m256 a00 = _mm256_setzero_ps(), a01 = a00, a10 = a00, a11 = a00;
        __m256 a20 = a00, a21 = a00, a30 = a00, a31 = a00;
        int c = 0;
        for (; c + 16 <= cols; c += 16) {
            const __m256 xa = _mm256_loadu_ps(x + c);
            const __m256 xb = _mm256_loadu_ps(x + c + 8);
            a00 = _mm256_fmadd_ps(load_bf16x8(w0 + c), xa, a00);
            a01 = _mm256_fmadd_ps(load_bf16x8(w0 + c + 8), xb, a01);
            a10 = _mm256_fmadd_ps(load_bf16x8(w1 + c), xa, a10);
            a11 = _mm256_fmadd_ps(load_bf16x8(w1 + c + 8), xb, a11);
            a20 = _mm256_fmadd_ps(load_bf16x8(w2 + c), xa, a20);
            a21 = _mm256_fmadd_ps(load_bf16x8(w2 + c + 8), xb, a21);
            a30 = _mm256_fmadd_ps(load_bf16x8(w3 + c), xa, a30);
            a31 = _mm256_fmadd_ps(load_bf16x8(w3 + c + 8), xb, a31);
        }
        for (; c + 8 <= cols; c += 8) {
            const __m256 xa = _mm256_loadu_ps(x + c);
            a00 = _mm256_fmadd_ps(load_bf16x8(w0 + c), xa, a00);
            a10 = _mm256_fmadd_ps(load_bf16x8(w1 + c), xa, a10);
            a20 = _mm256_fmadd_ps(load_bf16x8(w2 + c), xa, a20);
            a30 = _mm256_fmadd_ps(load_bf16x8(w3 + c), xa, a30);
        }
        float s0 = hsum(_mm256_add_ps(a00, a01)), s1 = hsum(_mm256_add_ps(a10, a11));
        float s2 = hsum(_mm256_add_ps(a20, a21)), s3 = hsum(_mm256_add_ps(a30, a31));
        for (; c < cols; ++c) {
            const float xv = x[c];
            s0 = __builtin_fmaf(widen1(w0[c]), xv, s0);
            s1 = __builtin_fmaf(widen1(w1[c]), xv, s1);
            s2 = __builtin_fmaf(widen1(w2[c]), xv, s2);
            s3 = __builtin_fmaf(widen1(w3[c]), xv, s3);
        }
        out[r] = s0;
        out[r + 1] = s1;
        out[r + 2] = s2;
        out[r + 3] = s3;
    }
    for (; r < rows; ++r) out[r] = dot_bf16_avx2(W + static_cast<std::size_t>(r) * stride, x, cols);
}

void axpy_avx2(float* y, float alpha, const float* x, int n) {
    const __m256 va = _mm256_set1_ps(alpha);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(y + i, _mm256_fmadd_ps(va, _mm256_loadu_ps(x + i), _mm256_loadu_ps(y + i)));
    }
    for (; i < n; ++i) y[i] = __builtin_fmaf(alpha, x[i], y[i]);
}

void mul_avx2(float* a, const float* b, int n) {
    int i = 0;
    for (; i + 8 <= n; i += 8) _mm256_storeu_ps(a + i, _mm256_mul_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
    for (; i < n; ++i) a[i] *= b[i];
}

void add_avx2(float* a, const float* b, int n) {
    int i = 0;
    for (; i + 8 <= n; i += 8) _mm256_storeu_ps(a + i, _mm256_add_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
    for (; i < n; ++i) a[i] += b[i];
}

void rmsnorm_avx2(float* out, const float* x, const float* weight, int n, float eps) {
    const float sum_sq = dot_avx2(x, x, n);
    const float scale = 1.0f / std::sqrt(sum_sq / static_cast<float>(n) + eps);
    const __m256 vs = _mm256_set1_ps(scale);
    int i = 0;   // each element is read before it is written, so out == x is fine
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_loadu_ps(weight + i), _mm256_mul_ps(vs, _mm256_loadu_ps(x + i))));
    }
    for (; i < n; ++i) out[i] = weight[i] * (scale * x[i]);
}

}  // namespace

void fill_avx2(KernelTable& t) {
    t.matvec = matvec_avx2;
    t.matvec_bf16 = matvec_bf16_avx2;
    t.rmsnorm = rmsnorm_avx2;
    t.dot = dot_avx2;
    t.axpy = axpy_avx2;
    t.mul = mul_avx2;
    t.add = add_avx2;
}

}  // namespace kernels
}  // namespace geneinfer

#endif  // GI_HAVE_X86_SIMD
