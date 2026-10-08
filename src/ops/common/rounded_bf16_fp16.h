#pragma once
// SV7 opt-in conversion policy, independent of CUDA and of dispatch.
// A finite BF16 value may round or saturate in FP16; this does not
// establish numerical eligibility of any tensor-core GEMM.
#include "ops/common/exact_bf16_fp16.h"
#include <cstdint>

namespace ninfer::ops::detail {

enum class Bf16Fp16ConversionClass : std::uint8_t {
    ExactFinite, RoundedFinite, SaturatedFinite, NonfiniteRejected
};

struct Bf16Fp16Conversion {
    std::uint16_t bits = 0; // Meaningful only for non-rejected finite values.
    Bf16Fp16ConversionClass category = Bf16Fp16ConversionClass::NonfiniteRejected;
    bool nonzero_to_zero = false;
};

// BF16 has seven explicit fraction bits. For normal BF16 values too small to
// be an FP16 normal, the FP16 subnormal integer is round_even(
//     (128 + fraction) * 2^(exponent_unbiased + 17)).
// This uses integer arithmetic so that ties-to-even and signed zero are
// deterministic on both hosts and GPU-independent toolchains.
inline Bf16Fp16Conversion finite_bf16_to_fp16_rne_saturate(
        std::uint16_t value) noexcept {
    const unsigned exp = (value >> 7) & 0xffU;
    const unsigned fraction = value & 0x7fU;
    const std::uint16_t sign = value & 0x8000U;

    if (exp == 255U) {
        return {0, Bf16Fp16ConversionClass::NonfiniteRejected, false};
    }

    std::uint16_t exact_half = 0;
    if (exact_bf16_to_fp16(value, exact_half)) {
        return {exact_half, Bf16Fp16ConversionClass::ExactFinite, false};
    }

    const int exponent = static_cast<int>(exp) - 127;
    if (exponent > 15) {
        // All finite BF16 values with unbiased exponent >= 16 exceed
        // max finite FP16 (65504). Saturate rather than produce infinity.
        return {static_cast<std::uint16_t>(sign | 0x7bffU),
                Bf16Fp16ConversionClass::SaturatedFinite, false};
    }

    unsigned half_fraction = 0;
    if (exp != 0U && exponent >= -25) {
        const unsigned significand = 128U | fraction;
        const int shift = -exponent - 17;
        // Exact conversion handled exponent >= -17 (and exact subnormals).
        if (shift > 0 && shift < 16) {
            const unsigned divisor = 1U << shift;
            const unsigned quotient = significand / divisor;
            const unsigned remainder = significand % divisor;
            const unsigned halfway = divisor / 2U;
            half_fraction = quotient +
                (remainder > halfway ||
                 (remainder == halfway && (quotient & 1U) != 0U));
        }
    }

    const auto half = static_cast<std::uint16_t>(sign | half_fraction);
    return {half, Bf16Fp16ConversionClass::RoundedFinite,
            (value & 0x7fffU) != 0 && (half & 0x7fffU) == 0};
}

} // namespace ninfer::ops::detail
