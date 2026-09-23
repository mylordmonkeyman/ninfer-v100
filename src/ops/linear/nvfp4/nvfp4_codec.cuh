#pragma once

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"

#include <cuda_bf16.h>
#if !defined(NINFER_VOLTA_BUILD)
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#endif

#include <cstdint>

namespace ninfer::ops::detail {

#if defined(NINFER_VOLTA_BUILD)
__device__ __forceinline__ float decode_nvfp4_e2m1_scalar(std::uint8_t code) {
    constexpr float positive[8] = {0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F};
    const float magnitude = positive[code & 7U];
    return (code & 8U) != 0U ? -magnitude : magnitude;
}

__device__ __forceinline__ float2 decode_nvfp4_e2m1x2(std::uint8_t storage) {
    return make_float2(decode_nvfp4_e2m1_scalar(storage & 0x0FU),
                       decode_nvfp4_e2m1_scalar(storage >> 4));
}

__device__ __forceinline__ float decode_nvfp4_e4m3(std::uint8_t storage) {
    const bool negative = (storage & 0x80U) != 0U;
    const int exponent  = (storage >> 3) & 0x0F;
    const int mantissa  = storage & 0x07;
    float value = 0.0F;
    if (exponent == 0) {
        value = mantissa == 0 ? 0.0F : ldexpf(static_cast<float>(mantissa), -9);
    } else if (exponent < 15) {
        value = ldexpf(1.0F + static_cast<float>(mantissa) * 0.125F, exponent - 7);
    } else if (mantissa < 7) {
        value = ldexpf(1.0F + static_cast<float>(mantissa) * 0.125F, 8);
    } else {
        return __int_as_float(0x7FFFFFFF);
    }
    return negative ? -value : value;
}

__device__ __forceinline__ std::uint8_t encode_nvfp4_e4m3_satfinite(float value) {
    const std::uint32_t bits = __float_as_uint(value);
    const bool negative = (bits >> 31) != 0U;
    float magnitude = fabsf(value);
    if (!isfinite(magnitude) || magnitude >= 448.0F) {
        return static_cast<std::uint8_t>((negative ? 0x80U : 0U) | 0x7EU);
    }
    if (magnitude == 0.0F) {
        return static_cast<std::uint8_t>(negative ? 0x80U : 0U);
    }

    int exponent = 0;
    const float fraction = frexpf(magnitude, &exponent);
    // frexpf returns fraction in [0.5,1); the unbiased IEEE-like exponent is exponent-1.
    int unbiased = exponent - 1;
    int exp_field = unbiased + 7;
    int mantissa = 0;
    if (exp_field <= 0) {
        mantissa = __float2int_rn(ldexpf(magnitude, 9));
        if (mantissa <= 0) {
            exp_field = 0;
            mantissa = 0;
        } else if (mantissa >= 8) {
            exp_field = 1;
            mantissa = 0;
        } else {
            exp_field = 0;
        }
    } else {
        const float normalized = ldexpf(fraction, 1);
        mantissa = __float2int_rn((normalized - 1.0F) * 8.0F);
        if (mantissa == 8) {
            mantissa = 0;
            ++exp_field;
        }
        if (exp_field >= 15) {
            if (exp_field > 15 || mantissa > 6) {
                exp_field = 15;
                mantissa = 6;
            }
        }
    }
    const std::uint8_t code = static_cast<std::uint8_t>((exp_field << 3) | mantissa);
    return static_cast<std::uint8_t>((negative ? 0x80U : 0U) | code);
}

__device__ __forceinline__ std::uint8_t encode_nvfp4_e2m1(float value) {
    const std::uint32_t bits = __float_as_uint(value);
    const std::uint8_t sign = (bits >> 31) != 0U ? 8U : 0U;
    const float x = fabsf(value);
    std::uint8_t code = 0;
    if (x <= 0.25F) {
        code = 0;
    } else if (x < 0.75F) {
        code = 1;
    } else if (x <= 1.25F) {
        code = 2;
    } else if (x < 1.75F) {
        code = 3;
    } else if (x <= 2.5F) {
        code = 4;
    } else if (x < 3.5F) {
        code = 5;
    } else if (x <= 5.0F) {
        code = 6;
    } else {
        code = 7;
    }
    return static_cast<std::uint8_t>(sign | code);
}
#else
__device__ __forceinline__ float2 decode_nvfp4_e2m1x2(std::uint8_t storage) {
    __nv_fp4x2_e2m1 value;
    value.__x = storage;
    return static_cast<float2>(value);
}

__device__ __forceinline__ float decode_nvfp4_e4m3(std::uint8_t storage) {
    __nv_fp8x2_e4m3 value;
    value.__x = static_cast<std::uint16_t>(storage) | (static_cast<std::uint16_t>(storage) << 8);
    return static_cast<float2>(value).x;
}

__device__ __forceinline__ std::uint8_t encode_nvfp4_e4m3_satfinite(float value) {
    return __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3);
}
#endif

struct alignas(8) Nvfp4QuantizedK16 {
    std::uint32_t codes_lo;
    std::uint32_t codes_hi;
    std::uint8_t scale;
};

static_assert(alignof(Nvfp4QuantizedK16) == 8);

__device__ __forceinline__ void
pack_nvfp4_e2m1x16(const float2 (&values)[8], std::uint32_t& codes_lo, std::uint32_t& codes_hi) {
#if defined(NINFER_VOLTA_BUILD)
    std::uint8_t packed[8];
#pragma unroll
    for (int pair = 0; pair < 8; ++pair) {
        const std::uint8_t lo = encode_nvfp4_e2m1(values[pair].x);
        const std::uint8_t hi = encode_nvfp4_e2m1(values[pair].y);
        packed[pair] = static_cast<std::uint8_t>(lo | (hi << 4));
    }
    codes_lo = static_cast<std::uint32_t>(packed[0]) |
               (static_cast<std::uint32_t>(packed[1]) << 8) |
               (static_cast<std::uint32_t>(packed[2]) << 16) |
               (static_cast<std::uint32_t>(packed[3]) << 24);
    codes_hi = static_cast<std::uint32_t>(packed[4]) |
               (static_cast<std::uint32_t>(packed[5]) << 8) |
               (static_cast<std::uint32_t>(packed[6]) << 16) |
               (static_cast<std::uint32_t>(packed[7]) << 24);
#else
    asm volatile("{\n"
                 ".reg .b8 b0;\n"
                 ".reg .b8 b1;\n"
                 ".reg .b8 b2;\n"
                 ".reg .b8 b3;\n"
                 ".reg .b8 b4;\n"
                 ".reg .b8 b5;\n"
                 ".reg .b8 b6;\n"
                 ".reg .b8 b7;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b0, %3, %2;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b1, %5, %4;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b2, %7, %6;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b3, %9, %8;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b4, %11, %10;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b5, %13, %12;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b6, %15, %14;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b7, %17, %16;\n"
                 "mov.b32 %0, {b0,b1,b2,b3};\n"
                 "mov.b32 %1, {b4,b5,b6,b7};\n"
                 "}\n"
                 : "=r"(codes_lo), "=r"(codes_hi)
                 : "f"(values[0].x), "f"(values[0].y), "f"(values[1].x), "f"(values[1].y),
                   "f"(values[2].x), "f"(values[2].y), "f"(values[3].x), "f"(values[3].y),
                   "f"(values[4].x), "f"(values[4].y), "f"(values[5].x), "f"(values[5].y),
                   "f"(values[6].x), "f"(values[6].y), "f"(values[7].x), "f"(values[7].y));
#endif
}

__device__ __forceinline__ Nvfp4QuantizedK16 quantize_nvfp4_k16(const __nv_bfloat16* source,
                                                                float input_scale_divisor) {
    const uint4 packed0                = load_vec<uint4>(source);
    const uint4 packed1                = load_vec<uint4>(source + 8);
    const std::uint32_t represented[8] = {
        packed0.x, packed0.y, packed0.z, packed0.w, packed1.x, packed1.y, packed1.z, packed1.w,
    };

    float2 values[8];
    float max_abs = 0.0F;
#pragma unroll
    for (int pair = 0; pair < 8; ++pair) {
        values[pair] = bf16x2_bits_to_float2(represented[pair]);
        max_abs      = fmaxf(max_abs, fabsf(values[pair].x));
        max_abs      = fmaxf(max_abs, fabsf(values[pair].y));
    }

    Nvfp4QuantizedK16 result{};
    const float scale_unencoded = __fdiv_rn(input_scale_divisor * max_abs, 6.0F);
    result.scale                = encode_nvfp4_e4m3_satfinite(scale_unencoded);
    if (result.scale == 0) { return result; }

    const float decoded_scale = decode_nvfp4_e4m3(result.scale);
#pragma unroll
    for (int pair = 0; pair < 8; ++pair) {
        values[pair].x = __fdiv_rn(values[pair].x * input_scale_divisor, decoded_scale);
        values[pair].y = __fdiv_rn(values[pair].y * input_scale_divisor, decoded_scale);
    }
    pack_nvfp4_e2m1x16(values, result.codes_lo, result.codes_hi);
    return result;
}

} // namespace ninfer::ops::detail
