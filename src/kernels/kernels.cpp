#include "kernels/kernels.h"

#include <cassert>
#include <cmath>
#include <cstddef>

namespace geneinfer {
namespace kernels {

void matvec(float* out, const float* W, const float* x, int rows, int cols) {
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

void rmsnorm(float* out, const float* x, const float* weight, int n, float eps) {
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

void softmax(float* x, int n) {
    assert(n > 0);
    float max_val = x[0];
    for (int i = 1; i < n; ++i) {
        if (x[i] > max_val) max_val = x[i];
    }
    // Subtracting the max keeps exp() in a safe range (largest argument is 0).
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        x[i] = std::exp(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < n; ++i) {
        x[i] /= sum;
    }
}

void silu(float* x, int n) {
    for (int i = 0; i < n; ++i) {
        x[i] = x[i] * (1.0f / (1.0f + std::exp(-x[i])));
    }
}

float dot(const float* a, const float* b, int n) {
    float acc = 0.0f;
    for (int i = 0; i < n; ++i) {
        acc += a[i] * b[i];
    }
    return acc;
}

void axpy(float* y, float alpha, const float* x, int n) {
    for (int i = 0; i < n; ++i) {
        y[i] += alpha * x[i];
    }
}

void mul(float* a, const float* b, int n) {
    for (int i = 0; i < n; ++i) {
        a[i] *= b[i];
    }
}

void add(float* a, const float* b, int n) {
    for (int i = 0; i < n; ++i) {
        a[i] += b[i];
    }
}

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

}  // namespace kernels
}  // namespace geneinfer
