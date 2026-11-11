#include "kernels/kernels.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "common/q8.h"
#include "kernels/kernel_table.h"

namespace geneinfer {
namespace kernels {

// ==================================================================================================
//  scalar reference implementations
// ==================================================================================================
namespace {

void matvec_scalar(float* out, const float* W, const float* x, int rows, int cols) {
    assert(rows > 0 && cols > 0);
    for (int r = 0; r < rows; ++r) {
        const float* row = W + static_cast<std::size_t>(r) * static_cast<std::size_t>(cols);
        float acc = 0.0f;
        for (int c = 0; c < cols; ++c) {
            acc += row[c] * x[c];
        }
        out[r] = acc;
    }
}

inline float widen(std::uint16_t b) {
    const std::uint32_t u = static_cast<std::uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &u, sizeof f);
    return f;
}

void matvec_bf16_scalar(float* out, const std::uint16_t* W, const float* x, int rows, int cols) {
    assert(rows > 0 && cols > 0);
    for (int r = 0; r < rows; ++r) {
        const std::uint16_t* row = W + static_cast<std::size_t>(r) * static_cast<std::size_t>(cols);
        float acc = 0.0f;
        for (int c = 0; c < cols; ++c) {
            acc += widen(row[c]) * x[c];
        }
        out[r] = acc;
    }
}

// int8 weights with one bf16 scale per group of kQ8Group values: out = sum over groups of scale * (sum q * x).
void matvec_q8_scalar(float* out, const std::int8_t* W, const std::uint16_t* scales, const float* x, int rows,
                      int cols) {
    assert(rows > 0 && cols > 0);
    const int groups = q8_groups(cols);
    for (int r = 0; r < rows; ++r) {
        const std::int8_t* q = W + static_cast<std::size_t>(r) * static_cast<std::size_t>(cols);
        const std::uint16_t* sc = scales + static_cast<std::size_t>(r) * static_cast<std::size_t>(groups);
        float acc = 0.0f;
        for (int g = 0; g < groups; ++g) {
            const int begin = g * kQ8Group;
            const int end = begin + kQ8Group < cols ? begin + kQ8Group : cols;
            float sum = 0.0f;
            for (int c = begin; c < end; ++c) sum += static_cast<float>(q[c]) * x[c];
            acc += widen(sc[g]) * sum;
        }
        out[r] = acc;
    }
}

void rmsnorm_scalar(float* out, const float* x, const float* weight, int n, float eps) {
    assert(n > 0);
    float sum_sq = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum_sq += x[i] * x[i];
    }
    const float scale = 1.0f / std::sqrt(sum_sq / static_cast<float>(n) + eps);
    // Each element is read before it is written, so out == x is fine.
    for (int i = 0; i < n; ++i) {
        out[i] = weight[i] * (scale * x[i]);
    }
}

}  // namespace

void softmax(float* x, int n) {
    assert(n > 0);
    float max_val = x[0];
    for (int i = 1; i < n; ++i) {
        if (x[i] > max_val) max_val = x[i];
    }
    // Subtracting the max keeps exp() in a safe range (largest argument is 0).
    // The sum is accumulated in double: a float32 running sum over thousands of terms loses several digits.
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        x[i] = std::exp(x[i] - max_val);
        sum += static_cast<double>(x[i]);
    }
    const float fsum = static_cast<float>(sum);
    for (int i = 0; i < n; ++i) {
        x[i] /= fsum;
    }
}

void silu(float* x, int n) {
    for (int i = 0; i < n; ++i) {
        x[i] = x[i] * (1.0f / (1.0f + std::exp(-x[i])));
    }
}

namespace {

float dot_scalar(const float* a, const float* b, int n) {
    float acc = 0.0f;
    for (int i = 0; i < n; ++i) {
        acc += a[i] * b[i];
    }
    return acc;
}

void axpy_scalar(float* y, float alpha, const float* x, int n) {
    for (int i = 0; i < n; ++i) {
        y[i] += alpha * x[i];
    }
}

void mul_scalar(float* a, const float* b, int n) {
    for (int i = 0; i < n; ++i) {
        a[i] *= b[i];
    }
}

void add_scalar(float* a, const float* b, int n) {
    for (int i = 0; i < n; ++i) {
        a[i] += b[i];
    }
}

}  // namespace

void rope(float* v, int n, int pos, int head_size, float theta, RopeStyle style) {
    assert(head_size > 0 && head_size % 2 == 0 && n % head_size == 0);
    const int half = head_size / 2;
    for (int base = 0; base < n; base += head_size) {
        for (int k = 0; k < half; ++k) {
            // Angles are computed in double precision: for large `pos` a float angle
            // would lose several digits. (A precomputed cos/sin table will replace
            // this later; the pow/cos/sin calls per element are deliberately simple.)
            const double freq = 1.0 / std::pow(static_cast<double>(theta),
                                               2.0 * static_cast<double>(k) / head_size);
            const double angle = static_cast<double>(pos) * freq;
            const float c = static_cast<float>(std::cos(angle));
            const float s = static_cast<float>(std::sin(angle));

            float* a = (style == RopeStyle::Interleaved) ? v + base + 2 * k : v + base + k;
            float* b = (style == RopeStyle::Interleaved) ? a + 1 : v + base + k + half;
            const float a0 = *a;
            const float b0 = *b;
            *a = a0 * c - b0 * s;
            *b = a0 * s + b0 * c;
        }
    }
}


// ==================================================================================================
//  instruction-set selection and dispatch
// ==================================================================================================
namespace {

KernelTable make_scalar_table() {
    return KernelTable{matvec_scalar, matvec_bf16_scalar, matvec_q8_scalar, rmsnorm_scalar, dot_scalar, axpy_scalar, mul_scalar, add_scalar};
}

bool cpu_has(Isa isa) {
#if defined(GI_HAVE_X86_SIMD)
    // __builtin_cpu_supports also checks that the operating system saves the wide registers.
    __builtin_cpu_init();
    if (isa == Isa::Avx2) return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    if (isa == Isa::Avx512) {
        return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512vl") &&
               __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    }
#else
    (void)isa;
#endif
    return isa == Isa::Scalar;
}

KernelTable make_table(Isa isa) {
    KernelTable t = make_scalar_table();
#if defined(GI_HAVE_X86_SIMD)
    if (isa == Isa::Avx2) fill_avx2(t);
    if (isa == Isa::Avx512) fill_avx512(t);
#else
    (void)isa;
#endif
    return t;
}

struct State {
    Isa isa;
    KernelTable table;
    State() : isa(Isa::Scalar), table(make_scalar_table()) {
        isa = best_isa();
        if (const char* env = std::getenv("GI_ISA")) {
            Isa want = isa;
            if (std::strcmp(env, "scalar") == 0) want = Isa::Scalar;
            else if (std::strcmp(env, "avx2") == 0) want = Isa::Avx2;
            else if (std::strcmp(env, "avx512") == 0) want = Isa::Avx512;
            else std::fprintf(stderr, "warning: GI_ISA=%s is not scalar, avx2 or avx512; using %s\n", env, isa_name(isa));
            if (isa_supported(want)) isa = want;
            else std::fprintf(stderr, "warning: GI_ISA=%s is not supported on this machine; using %s\n", env, isa_name(isa));
        }
        table = make_table(isa);
    }
};

State& state() {
    static State s;   // first use decides; initialization is thread-safe
    return s;
}

}  // namespace

KernelTable scalar_table() { return make_scalar_table(); }
KernelTable current_table() { return state().table; }

const char* isa_name(Isa isa) {
    switch (isa) {
        case Isa::Scalar: return "scalar";
        case Isa::Avx2: return "avx2";
        case Isa::Avx512: return "avx512";
    }
    return "?";
}

bool isa_supported(Isa isa) {
#if defined(GI_HAVE_X86_SIMD)
    return cpu_has(isa);
#else
    return isa == Isa::Scalar;
#endif
}

Isa best_isa() {
    if (isa_supported(Isa::Avx512)) return Isa::Avx512;
    if (isa_supported(Isa::Avx2)) return Isa::Avx2;
    return Isa::Scalar;
}

Isa active_isa() { return state().isa; }

bool set_isa(Isa isa) {
    if (!isa_supported(isa)) return false;
    State& s = state();
    s.isa = isa;
    s.table = make_table(isa);
    return true;
}

void matvec(float* out, const float* W, const float* x, int rows, int cols) {
    assert(rows > 0 && cols > 0);
    state().table.matvec(out, W, x, rows, cols);
}
void matvec_bf16(float* out, const std::uint16_t* W, const float* x, int rows, int cols) {
    assert(rows > 0 && cols > 0);
    state().table.matvec_bf16(out, W, x, rows, cols);
}
void matvec_q8(float* out, const std::int8_t* W, const std::uint16_t* scales, const float* x, int rows, int cols) {
    assert(rows > 0 && cols > 0);
    state().table.matvec_q8(out, W, scales, x, rows, cols);
}
void widen_bf16(float* dst, const std::uint16_t* src, int n) {
    for (int i = 0; i < n; ++i) dst[i] = widen(src[i]);
}
void rmsnorm(float* out, const float* x, const float* weight, int n, float eps) {
    assert(n > 0);
    state().table.rmsnorm(out, x, weight, n, eps);
}
float dot(const float* a, const float* b, int n) { return state().table.dot(a, b, n); }
void axpy(float* y, float alpha, const float* x, int n) { state().table.axpy(y, alpha, x, n); }
void mul(float* a, const float* b, int n) { state().table.mul(a, b, n); }
void add(float* a, const float* b, int n) { state().table.add(a, b, n); }

}  // namespace kernels
}  // namespace geneinfer
