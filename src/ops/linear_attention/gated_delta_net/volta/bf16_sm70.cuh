#pragma once

#if !defined(NINFER_VOLTA_BUILD)
#error "GDN Volta backend is SM70-only"
#endif

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail::gated_delta_net::volta {

struct Bf16x2Bits {
    std::uint16_t x;
    std::uint16_t y;
};

struct Bf16x4Bits {
    Bf16x2Bits lo;
    Bf16x2Bits hi;
};

__device__ __forceinline__ float bf16_bits_to_float(std::uint16_t bits) {
    return __uint_as_float(static_cast<std::uint32_t>(bits) << 16);
}

__device__ __forceinline__ Bf16x2Bits load_bf16x2_bits(const std::uint16_t* ptr) {
    const std::uint32_t packed = *reinterpret_cast<const std::uint32_t*>(ptr);
    return {
        static_cast<std::uint16_t>(packed & 0xffffU),
        static_cast<std::uint16_t>(packed >> 16),
    };
}

__device__ __forceinline__ Bf16x4Bits load_bf16x4_bits(const std::uint16_t* ptr) {
    const uint2 packed = *reinterpret_cast<const uint2*>(ptr);
    return {
        {
            static_cast<std::uint16_t>(packed.x & 0xffffU),
            static_cast<std::uint16_t>(packed.x >> 16),
        },
        {
            static_cast<std::uint16_t>(packed.y & 0xffffU),
            static_cast<std::uint16_t>(packed.y >> 16),
        },
    };
}

__device__ __forceinline__ std::uint16_t float_to_bf16_rn(float value) {
    const std::uint32_t bits = __float_as_uint(value);
    const std::uint32_t abs  = bits & 0x7fffffffU;

    if (abs >= 0x7f800000U) {
        const std::uint16_t upper = static_cast<std::uint16_t>(bits >> 16);
        if (abs == 0x7f800000U) { return upper; }
        return static_cast<std::uint16_t>(upper | 0x0040U);
    }

    const std::uint32_t lsb  = (bits >> 16) & 1U;
    const std::uint32_t bias = 0x7fffU + lsb;
    return static_cast<std::uint16_t>((bits + bias) >> 16);
}

} // namespace ninfer::ops::detail::gated_delta_net::volta
