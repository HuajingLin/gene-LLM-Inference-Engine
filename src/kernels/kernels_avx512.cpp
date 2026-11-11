// kernels_avx512.cpp - AVX-512F versions of the hot kernels. Compiled with -mavx512f -mavx512vl -mfma.
// Same structure as kernels_avx2.cpp with 16-float vectors; loop remainders use masked loads/stores, so there
// is no scalar tail. As there, no inline std:: templates or functions may be used in this file.
#if defined(GI_HAVE_X86_SIMD)

#include <immintrin.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "kernels/kernel_table.h"

namespace geneinfer {
namespace kernels {
namespace {

inline __mmask16 tail_mask(int remaining) {   // remaining in 1..15
    return static_cast<__mmask16>((1u << remaining) - 1u);
}

float dot_avx512(const float* a, const float* b, int n) {
    __m512 s0 = _mm512_setzero_ps(), s1 = s0, s2 = s0, s3 = s0;
    int i = 0;
    for (; i + 64 <= n; i += 64) {
        s0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), s0);
        s1 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 16), _mm512_loadu_ps(b + i + 16), s1);
        s2 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 32), _mm512_loadu_ps(b + i + 32), s2);
        s3 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 48), _mm512_loadu_ps(b + i + 48), s3);
    }
    for (; i + 16 <= n; i += 16) s0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), s0);
    if (i < n) {
        const __mmask16 m = tail_mask(n - i);
        s1 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(m, a + i), _mm512_maskz_loadu_ps(m, b + i), s1);
    }
    return _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(s0, s1), _mm512_add_ps(s2, s3)));
}

void matvec_avx512(float* out, const float* W, const float* x, int rows, int cols) {
    const std::size_t stride = static_cast<std::size_t>(cols);
    int r = 0;
    for (; r + 4 <= rows; r += 4) {
        const float* w0 = W + static_cast<std::size_t>(r) * stride;
        const float* w1 = w0 + stride;
        const float* w2 = w1 + stride;
        const float* w3 = w2 + stride;
        __m512 a00 = _mm512_setzero_ps(), a01 = a00, a10 = a00, a11 = a00;
        __m512 a20 = a00, a21 = a00, a30 = a00, a31 = a00;
        int c = 0;
        for (; c + 32 <= cols; c += 32) {
            const __m512 xa = _mm512_loadu_ps(x + c);
            const __m512 xb = _mm512_loadu_ps(x + c + 16);
            a00 = _mm512_fmadd_ps(_mm512_loadu_ps(w0 + c), xa, a00);
            a01 = _mm512_fmadd_ps(_mm512_loadu_ps(w0 + c + 16), xb, a01);
            a10 = _mm512_fmadd_ps(_mm512_loadu_ps(w1 + c), xa, a10);
            a11 = _mm512_fmadd_ps(_mm512_loadu_ps(w1 + c + 16), xb, a11);
            a20 = _mm512_fmadd_ps(_mm512_loadu_ps(w2 + c), xa, a20);
            a21 = _mm512_fmadd_ps(_mm512_loadu_ps(w2 + c + 16), xb, a21);
            a30 = _mm512_fmadd_ps(_mm512_loadu_ps(w3 + c), xa, a30);
            a31 = _mm512_fmadd_ps(_mm512_loadu_ps(w3 + c + 16), xb, a31);
        }
        for (; c + 16 <= cols; c += 16) {
            const __m512 xa = _mm512_loadu_ps(x + c);
            a00 = _mm512_fmadd_ps(_mm512_loadu_ps(w0 + c), xa, a00);
            a10 = _mm512_fmadd_ps(_mm512_loadu_ps(w1 + c), xa, a10);
            a20 = _mm512_fmadd_ps(_mm512_loadu_ps(w2 + c), xa, a20);
            a30 = _mm512_fmadd_ps(_mm512_loadu_ps(w3 + c), xa, a30);
        }
        if (c < cols) {
            const __mmask16 m = tail_mask(cols - c);
            const __m512 xa = _mm512_maskz_loadu_ps(m, x + c);
            a01 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(m, w0 + c), xa, a01);
            a11 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(m, w1 + c), xa, a11);
            a21 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(m, w2 + c), xa, a21);
            a31 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(m, w3 + c), xa, a31);
        }
        out[r] = _mm512_reduce_add_ps(_mm512_add_ps(a00, a01));
        out[r + 1] = _mm512_reduce_add_ps(_mm512_add_ps(a10, a11));
        out[r + 2] = _mm512_reduce_add_ps(_mm512_add_ps(a20, a21));
        out[r + 3] = _mm512_reduce_add_ps(_mm512_add_ps(a30, a31));
    }
    for (; r < rows; ++r) out[r] = dot_avx512(W + static_cast<std::size_t>(r) * stride, x, cols);
}

// ---- bfloat16 weights: widen on the fly; same structure and addition order as the float32 kernels ----
inline __m512 load_bf16x16(const std::uint16_t* p) {
    return _mm512_castsi512_ps(
        _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p))), 16));
}

// The last 1..15 values: byte-granular masked loads would need AVX512BW, so copy them into a zero-padded
// buffer first (once per row; negligible next to streaming the row).
inline __m512 load_bf16x16_tail(const std::uint16_t* p, int remaining) {
    alignas(32) std::uint16_t tmp[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (int k = 0; k < remaining; ++k) tmp[k] = p[k];
    return load_bf16x16(tmp);
}

float dot_bf16_avx512(const std::uint16_t* a, const float* b, int n) {
    __m512 s0 = _mm512_setzero_ps(), s1 = s0, s2 = s0, s3 = s0;
    int i = 0;
    for (; i + 64 <= n; i += 64) {
        s0 = _mm512_fmadd_ps(load_bf16x16(a + i), _mm512_loadu_ps(b + i), s0);
        s1 = _mm512_fmadd_ps(load_bf16x16(a + i + 16), _mm512_loadu_ps(b + i + 16), s1);
        s2 = _mm512_fmadd_ps(load_bf16x16(a + i + 32), _mm512_loadu_ps(b + i + 32), s2);
        s3 = _mm512_fmadd_ps(load_bf16x16(a + i + 48), _mm512_loadu_ps(b + i + 48), s3);
    }
    for (; i + 16 <= n; i += 16) s0 = _mm512_fmadd_ps(load_bf16x16(a + i), _mm512_loadu_ps(b + i), s0);
    if (i < n) {
        const __mmask16 m = tail_mask(n - i);
        s1 = _mm512_fmadd_ps(load_bf16x16_tail(a + i, n - i), _mm512_maskz_loadu_ps(m, b + i), s1);
    }
    return _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(s0, s1), _mm512_add_ps(s2, s3)));
}

void matvec_bf16_avx512(float* out, const std::uint16_t* W, const float* x, int rows, int cols) {
    const std::size_t stride = static_cast<std::size_t>(cols);
    int r = 0;
    for (; r + 4 <= rows; r += 4) {
        const std::uint16_t* w0 = W + static_cast<std::size_t>(r) * stride;
        const std::uint16_t* w1 = w0 + stride;
        const std::uint16_t* w2 = w1 + stride;
        const std::uint16_t* w3 = w2 + stride;
        __m512 a00 = _mm512_setzero_ps(), a01 = a00, a10 = a00, a11 = a00;
        __m512 a20 = a00, a21 = a00, a30 = a00, a31 = a00;
        int c = 0;
        for (; c + 32 <= cols; c += 32) {
            const __m512 xa = _mm512_loadu_ps(x + c);
            const __m512 xb = _mm512_loadu_ps(x + c + 16);
            a00 = _mm512_fmadd_ps(load_bf16x16(w0 + c), xa, a00);
            a01 = _mm512_fmadd_ps(load_bf16x16(w0 + c + 16), xb, a01);
            a10 = _mm512_fmadd_ps(load_bf16x16(w1 + c), xa, a10);
            a11 = _mm512_fmadd_ps(load_bf16x16(w1 + c + 16), xb, a11);
            a20 = _mm512_fmadd_ps(load_bf16x16(w2 + c), xa, a20);
            a21 = _mm512_fmadd_ps(load_bf16x16(w2 + c + 16), xb, a21);
            a30 = _mm512_fmadd_ps(load_bf16x16(w3 + c), xa, a30);
            a31 = _mm512_fmadd_ps(load_bf16x16(w3 + c + 16), xb, a31);
        }
        for (; c + 16 <= cols; c += 16) {
            const __m512 xa = _mm512_loadu_ps(x + c);
            a00 = _mm512_fmadd_ps(load_bf16x16(w0 + c), xa, a00);
            a10 = _mm512_fmadd_ps(load_bf16x16(w1 + c), xa, a10);
            a20 = _mm512_fmadd_ps(load_bf16x16(w2 + c), xa, a20);
            a30 = _mm512_fmadd_ps(load_bf16x16(w3 + c), xa, a30);
        }
        if (c < cols) {
            const __mmask16 m = tail_mask(cols - c);
            const __m512 xa = _mm512_maskz_loadu_ps(m, x + c);
            a01 = _mm512_fmadd_ps(load_bf16x16_tail(w0 + c, cols - c), xa, a01);
            a11 = _mm512_fmadd_ps(load_bf16x16_tail(w1 + c, cols - c), xa, a11);
            a21 = _mm512_fmadd_ps(load_bf16x16_tail(w2 + c, cols - c), xa, a21);
            a31 = _mm512_fmadd_ps(load_bf16x16_tail(w3 + c, cols - c), xa, a31);
        }
        out[r] = _mm512_reduce_add_ps(_mm512_add_ps(a00, a01));
        out[r + 1] = _mm512_reduce_add_ps(_mm512_add_ps(a10, a11));
        out[r + 2] = _mm512_reduce_add_ps(_mm512_add_ps(a20, a21));
        out[r + 3] = _mm512_reduce_add_ps(_mm512_add_ps(a30, a31));
    }
    for (; r < rows; ++r) out[r] = dot_bf16_avx512(W + static_cast<std::size_t>(r) * stride, x, cols);
}

void axpy_avx512(float* y, float alpha, const float* x, int n) {
    const __m512 va = _mm512_set1_ps(alpha);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        _mm512_storeu_ps(y + i, _mm512_fmadd_ps(va, _mm512_loadu_ps(x + i), _mm512_loadu_ps(y + i)));
    }
    if (i < n) {
        const __mmask16 m = tail_mask(n - i);
        _mm512_mask_storeu_ps(y + i, m, _mm512_fmadd_ps(va, _mm512_maskz_loadu_ps(m, x + i), _mm512_maskz_loadu_ps(m, y + i)));
    }
}

void mul_avx512(float* a, const float* b, int n) {
    int i = 0;
    for (; i + 16 <= n; i += 16) _mm512_storeu_ps(a + i, _mm512_mul_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i)));
    if (i < n) {
        const __mmask16 m = tail_mask(n - i);
        _mm512_mask_storeu_ps(a + i, m, _mm512_mul_ps(_mm512_maskz_loadu_ps(m, a + i), _mm512_maskz_loadu_ps(m, b + i)));
    }
}

void add_avx512(float* a, const float* b, int n) {
    int i = 0;
    for (; i + 16 <= n; i += 16) _mm512_storeu_ps(a + i, _mm512_add_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i)));
    if (i < n) {
        const __mmask16 m = tail_mask(n - i);
        _mm512_mask_storeu_ps(a + i, m, _mm512_add_ps(_mm512_maskz_loadu_ps(m, a + i), _mm512_maskz_loadu_ps(m, b + i)));
    }
}

// ---- int8 weights, one bf16 scale per group of 64 values (see common/q8.h) ----
// For every group the products q * x are summed (two interleaved FMA chains), then the group sum is multiplied
// by the group scale and added to the row accumulator (again two chains, by group parity). The widening
// int8 -> int32 -> float32 is exact. Every row runs exactly this sequence, so a matrix computed in pieces gives
// the same bits. The activations are not quantized.
inline __m512 load_q8x16(const std::int8_t* p) {
    return _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p))));
}

inline float scale_to_float(std::uint16_t b) {
    const std::uint32_t u = static_cast<std::uint32_t>(b) << 16;
    float f;
    __builtin_memcpy(&f, &u, sizeof f);
    return f;
}

float row_q8_avx512(const std::int8_t* q, const std::uint16_t* sc, const float* x, int cols) {
    __m512 acc0 = _mm512_setzero_ps(), acc1 = acc0;
    const int groups = (cols + 63) / 64;
    for (int g = 0; g < groups; ++g) {
        const int c0 = g * 64;
        int n = cols - c0;
        if (n > 64) n = 64;
        __m512 g0 = _mm512_setzero_ps(), g1 = g0;
        if (n == 64) {
            g0 = _mm512_fmadd_ps(load_q8x16(q + c0), _mm512_loadu_ps(x + c0), g0);
            g1 = _mm512_fmadd_ps(load_q8x16(q + c0 + 16), _mm512_loadu_ps(x + c0 + 16), g1);
            g0 = _mm512_fmadd_ps(load_q8x16(q + c0 + 32), _mm512_loadu_ps(x + c0 + 32), g0);
            g1 = _mm512_fmadd_ps(load_q8x16(q + c0 + 48), _mm512_loadu_ps(x + c0 + 48), g1);
        } else {
            int j = 0, c = c0;
            const int end = c0 + n;
            for (; c + 16 <= end; c += 16, ++j) {
                if (j & 1) g1 = _mm512_fmadd_ps(load_q8x16(q + c), _mm512_loadu_ps(x + c), g1);
                else g0 = _mm512_fmadd_ps(load_q8x16(q + c), _mm512_loadu_ps(x + c), g0);
            }
            if (c < end) {   // 1..15 values left: zero-padded copy of the int8 values (byte masks need AVX512BW)
                const int rem = end - c;
                alignas(16) std::int8_t tmp[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
                for (int k = 0; k < rem; ++k) tmp[k] = q[c + k];
                const __m512 prod = _mm512_mul_ps(load_q8x16(tmp), _mm512_maskz_loadu_ps(tail_mask(rem), x + c));
                if (j & 1) g1 = _mm512_add_ps(g1, prod);
                else g0 = _mm512_add_ps(g0, prod);
            }
        }
        const __m512 gs = _mm512_add_ps(g0, g1);
        const __m512 s = _mm512_set1_ps(scale_to_float(sc[g]));
        if (g & 1) acc1 = _mm512_fmadd_ps(gs, s, acc1);
        else acc0 = _mm512_fmadd_ps(gs, s, acc0);
    }
    return _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
}

void matvec_q8_avx512(float* out, const std::int8_t* W, const std::uint16_t* scales, const float* x, int rows,
                      int cols) {
    const std::size_t stride = static_cast<std::size_t>(cols);
    const std::size_t sstride = static_cast<std::size_t>((cols + 63) / 64);
    for (int r = 0; r < rows; ++r) {
        out[r] = row_q8_avx512(W + static_cast<std::size_t>(r) * stride, scales + static_cast<std::size_t>(r) * sstride, x, cols);
    }
}

void rmsnorm_avx512(float* out, const float* x, const float* weight, int n, float eps) {
    const float sum_sq = dot_avx512(x, x, n);
    const float scale = 1.0f / std::sqrt(sum_sq / static_cast<float>(n) + eps);
    const __m512 vs = _mm512_set1_ps(scale);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        _mm512_storeu_ps(out + i, _mm512_mul_ps(_mm512_loadu_ps(weight + i), _mm512_mul_ps(vs, _mm512_loadu_ps(x + i))));
    }
    if (i < n) {
        const __mmask16 m = tail_mask(n - i);
        _mm512_mask_storeu_ps(out + i, m, _mm512_mul_ps(_mm512_maskz_loadu_ps(m, weight + i),
                                                          _mm512_mul_ps(vs, _mm512_maskz_loadu_ps(m, x + i))));
    }
}

}  // namespace

void fill_avx512(KernelTable& t) {
    t.matvec = matvec_avx512;
    t.matvec_bf16 = matvec_bf16_avx512;
    t.matvec_q8 = matvec_q8_avx512;
    t.rmsnorm = rmsnorm_avx512;
    t.dot = dot_avx512;
    t.axpy = axpy_avx512;
    t.mul = mul_avx512;
    t.add = add_avx512;
}

}  // namespace kernels
}  // namespace geneinfer

#endif  // GI_HAVE_X86_SIMD
