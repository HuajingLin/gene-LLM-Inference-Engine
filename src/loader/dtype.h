// dtype.h - tensor element types of safetensors files and their conversion to float32.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace geneinfer {

enum class DType { F32, F16, BF16, Other };

// Maps the safetensors dtype string ("F32", "F16", "BF16", ...) to a DType; anything else is Other.
DType dtype_from_name(const std::string& name);

// Size in bytes of one element, or 0 if the dtype name is not known.
std::size_t dtype_size(const std::string& name);

// bfloat16 is the upper half of a float32: conversion is exact (a 16-bit shift).
float bf16_to_f32(std::uint16_t bits);

// IEEE half precision -> float32, exact for every input (subnormals, infinities and NaNs included).
float f16_to_f32(std::uint16_t bits);

// Converts `count` elements of `dtype` stored (little-endian) at `src` into float32 at `dst`.
// Throws std::invalid_argument for DType::Other.
void convert_to_f32(DType dtype, const unsigned char* src, float* dst, std::size_t count);

}  // namespace geneinfer
