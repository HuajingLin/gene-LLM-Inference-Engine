// q8.h - layout constants of the int8 weight format.
//
// A weight matrix [rows, cols] is stored as
//   q      int8,  [rows, cols]    quantized values, -127..127
//   scale  bf16,  [rows, groups]  one scale per group of kQ8Group consecutive values of a row
// with groups = ceil(cols / kQ8Group); the last group of a row is shorter when cols is not a multiple of
// kQ8Group. The dequantized weight is  q * scale  (the scale is the 16-bit bfloat16 pattern widened to float32).
#pragma once

namespace geneinfer {

inline constexpr int kQ8Group = 64;

inline constexpr int q8_groups(int cols) { return (cols + kQ8Group - 1) / kQ8Group; }

}  // namespace geneinfer
