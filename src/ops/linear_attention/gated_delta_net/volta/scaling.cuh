#pragma once

#if !defined(NINFER_VOLTA_BUILD)
#error "GDN Volta backend is SM70-only"
#endif

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace ninfer::ops::detail::gated_delta_net::volta {

struct RangeStats {
    float max_abs = 0.0F;
    float min_nonzero_abs = 0.0F;
};

enum class BridgeMode : std::uint8_t {
    Fp16Mma,
    Fp32Simt,
};

struct BridgePlan {
    BridgeMode mode = BridgeMode::Fp32Simt;
    float mul = 1.0F;
    float inv = 1.0F;
    RangeStats range{};
};

inline constexpr float kFp16Headroom = 32768.0F;
inline constexpr float kFp16MinSafe = 0x1p-14F;

__host__ __device__ inline RangeStats empty_range_stats() noexcept {
    return {};
}

__host__ __device__ inline void range_observe(RangeStats& range, float value) noexcept {
    const float magnitude = fabsf(value);
    if (!isfinite(magnitude)) {
        range.max_abs = CUDART_INF_F;
        range.min_nonzero_abs = 0.0F;
        return;
    }
    if (magnitude == 0.0F) { return; }
    if (magnitude > range.max_abs) { range.max_abs = magnitude; }
    if (range.min_nonzero_abs == 0.0F || magnitude < range.min_nonzero_abs) {
        range.min_nonzero_abs = magnitude;
    }
}

__host__ __device__ inline RangeStats merge_range_stats(RangeStats a, RangeStats b) noexcept {
    if (!isfinite(a.max_abs) || !isfinite(b.max_abs)) {
        return {CUDART_INF_F, 0.0F};
    }
    RangeStats out{};
    out.max_abs = a.max_abs > b.max_abs ? a.max_abs : b.max_abs;
    if (a.min_nonzero_abs == 0.0F) {
        out.min_nonzero_abs = b.min_nonzero_abs;
    } else if (b.min_nonzero_abs == 0.0F) {
        out.min_nonzero_abs = a.min_nonzero_abs;
    } else {
        out.min_nonzero_abs =
            a.min_nonzero_abs < b.min_nonzero_abs ? a.min_nonzero_abs : b.min_nonzero_abs;
    }
    return out;
}

__host__ __device__ inline BridgePlan plan_fp16_bridge(RangeStats range) noexcept {
    BridgePlan plan{};
    plan.range = range;

    if (!isfinite(range.max_abs) || range.max_abs < 0.0F || range.min_nonzero_abs < 0.0F) {
        return plan;
    }
    if (range.max_abs == 0.0F) {
        plan.mode = BridgeMode::Fp16Mma;
        return plan;
    }
    if (range.min_nonzero_abs == 0.0F || !isfinite(range.min_nonzero_abs) ||
        range.min_nonzero_abs > range.max_abs) {
        return plan;
    }

    const float e_min_f = ceilf(-14.0F - log2f(range.min_nonzero_abs));
    const float e_max_f = floorf(15.0F - log2f(range.max_abs));
    if (!isfinite(e_min_f) || !isfinite(e_max_f)) { return plan; }

    int e_min = static_cast<int>(e_min_f);
    int e_max = static_cast<int>(e_max_f);
    if (e_min < -126) { e_min = -126; }
    if (e_max > 127) { e_max = 127; }
    if (e_min > e_max) { return plan; }

    const int exponent = e_max;
    const float mul = ldexpf(1.0F, exponent);
    const float inv = ldexpf(1.0F, -exponent);
    if (!isfinite(mul) || mul == 0.0F || !isfinite(inv) || inv == 0.0F) { return plan; }

    const float scaled_max = range.max_abs * mul;
    const float scaled_min = range.min_nonzero_abs * mul;
    if (!isfinite(scaled_max) || scaled_max > kFp16Headroom || scaled_min < kFp16MinSafe) {
        return plan;
    }

    plan.mode = BridgeMode::Fp16Mma;
    plan.mul = mul;
    plan.inv = inv;
    return plan;
}

__host__ __device__ inline float bridge_dynamic_range_ratio(RangeStats range) noexcept {
    if (range.max_abs == 0.0F) { return 1.0F; }
    if (range.min_nonzero_abs == 0.0F) { return CUDART_INF_F; }
    return range.max_abs / range.min_nonzero_abs;
}

} // namespace ninfer::ops::detail::gated_delta_net::volta
