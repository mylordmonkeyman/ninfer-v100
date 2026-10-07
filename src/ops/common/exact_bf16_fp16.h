#pragma once

#include <cstdint>

namespace ninfer::ops::detail {

// Convert represented BF16 bits only when the finite value is exactly representable
// in FP16. Signed zero is preserved. No floating-point rounding mode is involved.
inline bool exact_bf16_to_fp16(std::uint16_t bf16, std::uint16_t& fp16) noexcept {
    const unsigned exponent = (bf16 >> 7) & 0xffU;
    const unsigned fraction = bf16 & 0x7fU;
    const unsigned sign = bf16 & 0x8000U;
    if (exponent == 0) {
        if (fraction != 0) { return false; }
        fp16 = static_cast<std::uint16_t>(sign);
        return true;
    }
    if (exponent == 255) { return false; }
    const int e = static_cast<int>(exponent) - 127;
    if (e > 15 || e < -24) { return false; }
    if (e >= -14) {
        fp16 = static_cast<std::uint16_t>(sign | ((e + 15) << 10) | (fraction << 3));
        return true;
    }
    const unsigned significand = 128U | fraction;
    const int shift = -e - 17;
    unsigned subnormal;
    if (shift > 0) {
        if ((significand & ((1U << shift) - 1U)) != 0) { return false; }
        subnormal = significand >> shift;
    } else {
        subnormal = significand << -shift;
    }
    fp16 = static_cast<std::uint16_t>(sign | subnormal);
    return true;
}

} // namespace ninfer::ops::detail
