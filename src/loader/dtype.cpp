#include "loader/dtype.h"

#include <cstring>
#include <stdexcept>

namespace geneinfer {

namespace {

float from_bits(std::uint32_t bits) {
    float f;
    std::memcpy(&f, &bits, sizeof f);
    return f;
}

std::uint16_t load_u16(const unsigned char* p) {
    // Assemble the value byte by byte: portable and safe for unaligned data (files are little-endian).
    return static_cast<std::uint16_t>(p[0] | (p[1] << 8));
}

}  // namespace

DType dtype_from_name(const std::string& name) {
    if (name == "F32") return DType::F32;
    if (name == "F16") return DType::F16;
    if (name == "BF16") return DType::BF16;
    return DType::Other;
}

std::size_t dtype_size(const std::string& name) {
    if (name == "F64" || name == "I64" || name == "U64") return 8;
    if (name == "F32" || name == "I32" || name == "U32") return 4;
    if (name == "F16" || name == "BF16" || name == "I16" || name == "U16") return 2;
    if (name == "I8" || name == "U8" || name == "BOOL" || name == "F8_E4M3" || name == "F8_E5M2") return 1;
    return 0;
}

float bf16_to_f32(std::uint16_t bits) { return from_bits(static_cast<std::uint32_t>(bits) << 16); }

float f16_to_f32(std::uint16_t h) {
    const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
    std::uint32_t exponent = (h >> 10) & 0x1Fu;
    std::uint32_t mantissa = h & 0x3FFu;

    if (exponent == 0) {
        if (mantissa == 0) return from_bits(sign);  // +-0
        // Subnormal half: normalize it into a normal float.
        exponent = 127 - 15 + 1;
        while ((mantissa & 0x400u) == 0) {
            mantissa <<= 1;
            --exponent;
        }
        mantissa &= 0x3FFu;
        return from_bits(sign | (exponent << 23) | (mantissa << 13));
    }
    if (exponent == 31) {  // infinity or NaN (payload is preserved)
        return from_bits(sign | 0x7F800000u | (mantissa << 13));
    }
    return from_bits(sign | ((exponent + (127 - 15)) << 23) | (mantissa << 13));
}

void convert_to_f32(DType dtype, const unsigned char* src, float* dst, std::size_t count) {
    switch (dtype) {
        case DType::F32:
            std::memcpy(dst, src, count * sizeof(float));  // memcpy: the source may be unaligned
            return;
        case DType::BF16:
            for (std::size_t i = 0; i < count; ++i) dst[i] = bf16_to_f32(load_u16(src + 2 * i));
            return;
        case DType::F16:
            for (std::size_t i = 0; i < count; ++i) dst[i] = f16_to_f32(load_u16(src + 2 * i));
            return;
        case DType::Other:
            break;
    }
    throw std::invalid_argument("convert_to_f32: unsupported dtype");
}

}  // namespace geneinfer
