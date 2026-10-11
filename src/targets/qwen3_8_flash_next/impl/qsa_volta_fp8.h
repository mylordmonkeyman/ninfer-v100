#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace ninfer::targets::qwen3_8_flash_next::detail {
// Exact raw E4M3FN -> IEEE half bits. No scaling or floating-point conversion.
// This is qualified against the native codec for every finite signed code.
__host__ __device__ constexpr std::uint16_t qsa_volta_fp8_half_bits(std::uint8_t code) {
    const unsigned sign = (unsigned(code) & 0x80U) << 8;
    const unsigned mag = unsigned(code) & 0x7fU;
    if (mag == 0x7fU) return std::uint16_t(sign | 0x7fffU);
    if (mag >= 8U) return std::uint16_t(sign | ((mag << 7) + 0x2000U));
    if (mag == 0U) return std::uint16_t(sign);
    if (mag == 1U) return std::uint16_t(sign | 0x1800U);
    if (mag < 4U) return std::uint16_t(sign | (0x1c00U + (mag-2U)*0x200U));
    return std::uint16_t(sign | (0x2000U + (mag-4U)*0x100U));
}
} // namespace ninfer::targets::qwen3_8_flash_next::detail
