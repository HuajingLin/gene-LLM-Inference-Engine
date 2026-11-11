// quantize.h - float32/bfloat16 -> int8 weight quantization (format: common/q8.h).
//
// Per group of 64 values of a row: scale = the smallest bfloat16 number >= max|w| / 127 (rounding UP, so that
// no value is clipped), q = round-to-nearest-even(w / scale) in -127..127. An all-zero group gets scale 0 and
// q = 0. The dequantized value is q * scale; its error is at most scale / 2 (about max|w| / 254) per weight.
#pragma once

#include <cstdint>
#include <vector>

#include "loader/checkpoint.h"

namespace geneinfer {

struct Q8Matrix {
    std::vector<std::int8_t> q;           // [rows, cols]
    std::vector<std::uint16_t> scale;     // [rows, ceil(cols / 64)] bfloat16 bits
};

// `src` must be a float32 or bfloat16 matrix [rows, cols]. Throws std::runtime_error for NaN or infinite values.
Q8Matrix quantize_q8(const Mat& src, int rows, int cols);

// Dequantized value of one weight (for tests and diagnostics).
float dequantize_q8(const Q8Matrix& m, int cols, int row, int col);

}  // namespace geneinfer
