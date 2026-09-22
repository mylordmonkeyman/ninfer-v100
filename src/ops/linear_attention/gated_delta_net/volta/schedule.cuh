#pragma once

#if !defined(NINFER_VOLTA_BUILD)
#error "GDN Volta backend is SM70-only"
#endif

#include "ops/linear_attention/gated_delta_net/volta/common.cuh"

#include <cstdint>

namespace ninfer::ops::detail::gated_delta_net::volta {

enum class Schedule : std::uint8_t {
    Dv16 = 16,
    Dv32 = 32,
};

template <int DV_TILE>
struct GdnSchedule {
    static_assert(DV_TILE == 16 || DV_TILE == 32);

    static constexpr int kDvTile           = DV_TILE;
    static constexpr int kDvPerWarp        = 4;
    static constexpr int kStateColsPerLane = 4;

    static constexpr int kWarps = kDvTile / kDvPerWarp;
    static constexpr int kThreads = kWarps * 32;
    static constexpr int kDvStrips = kStateDim / kDvTile;
    static constexpr int kStateValuesPerThread = kDvPerWarp * kStateColsPerLane;
    static constexpr int kMinBlocksPerSm = DV_TILE == 16 ? 3 : 2;

    static_assert(kWarps * kDvPerWarp == kDvTile);
    static_assert(kThreads * kStateValuesPerThread == kDvTile * kStateDim);
};

static_assert(GdnSchedule<16>::kWarps == 4);
static_assert(GdnSchedule<16>::kThreads == 128);
static_assert(GdnSchedule<16>::kDvStrips == 8);
static_assert(GdnSchedule<32>::kWarps == 8);
static_assert(GdnSchedule<32>::kThreads == 256);
static_assert(GdnSchedule<32>::kDvStrips == 4);

} // namespace ninfer::ops::detail::gated_delta_net::volta
