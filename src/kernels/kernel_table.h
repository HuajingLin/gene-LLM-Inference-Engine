// kernel_table.h - internal: the set of kernels that have several implementations (scalar, AVX2, AVX-512).
// kernels.cpp owns the table and the CPU detection; kernels_avx2.cpp / kernels_avx512.cpp only fill it.
#pragma once

#include <cstdint>

namespace geneinfer {
namespace kernels {

struct KernelTable {
    void (*matvec)(float* out, const float* W, const float* x, int rows, int cols);
    void (*matvec_bf16)(float* out, const std::uint16_t* W, const float* x, int rows, int cols);
    void (*matvec_q8)(float* out, const std::int8_t* W, const std::uint16_t* scales, const float* x, int rows, int cols);
    void (*rmsnorm)(float* out, const float* x, const float* weight, int n, float eps);
    float (*dot)(const float* a, const float* b, int n);
    void (*axpy)(float* y, float alpha, const float* x, int n);
    void (*mul)(float* a, const float* b, int n);
    void (*add)(float* a, const float* b, int n);
};

KernelTable scalar_table();    // the reference implementations
KernelTable current_table();   // what the public kernels call right now (for tests)

#if defined(GI_HAVE_X86_SIMD)
// Overwrite the entries of `t` with the AVX2+FMA / AVX-512F version. Only call after checking the CPU.
void fill_avx2(KernelTable& t);
void fill_avx512(KernelTable& t);
#endif

}  // namespace kernels
}  // namespace geneinfer
