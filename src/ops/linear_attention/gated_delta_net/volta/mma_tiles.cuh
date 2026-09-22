#pragma once

#if !defined(NINFER_VOLTA_BUILD)
#error "GDN Volta backend is SM70-only"
#endif

#include "ops/common/volta_mma.cuh"

#include <cuda_fp16.h>

#include <cstddef>

namespace ninfer::ops::detail::gated_delta_net::volta {

struct MmaMacro16Coord {
    int row;
    int col;
};

__host__ __device__ constexpr MmaMacro16Coord mma_macro16_group_coord(int group) noexcept {
    return {
        group >= 2 ? 8 : 0,
        (group & 1) != 0 ? 8 : 0,
    };
}

__device__ __forceinline__ ninfer::ops::VoltaMma884Operand
load_mma884_a_macro16(const __half* tile, int leading_dimension, int k_offset = 0) {
    const int group = ninfer::ops::volta_mma884_group(ninfer::ops::volta_lane_id());
    const MmaMacro16Coord coord = mma_macro16_group_coord(group);
    const __half* group_tile =
        tile + static_cast<std::ptrdiff_t>(coord.row) * leading_dimension + k_offset;
    return ninfer::ops::volta_mma884_load_a_row(group_tile, leading_dimension);
}

__device__ __forceinline__ ninfer::ops::VoltaMma884Operand
load_mma884_b_macro16_col(const __half* tile, int leading_dimension, int k_offset = 0) {
    const int group = ninfer::ops::volta_mma884_group(ninfer::ops::volta_lane_id());
    const MmaMacro16Coord coord = mma_macro16_group_coord(group);
    const __half* group_tile = tile +
        static_cast<std::ptrdiff_t>(coord.col) * leading_dimension + k_offset;
    return ninfer::ops::volta_mma884_load_b_col(group_tile, leading_dimension);
}

static_assert(mma_macro16_group_coord(0).row == 0 && mma_macro16_group_coord(0).col == 0);
static_assert(mma_macro16_group_coord(1).row == 0 && mma_macro16_group_coord(1).col == 8);
static_assert(mma_macro16_group_coord(2).row == 8 && mma_macro16_group_coord(2).col == 0);
static_assert(mma_macro16_group_coord(3).row == 8 && mma_macro16_group_coord(3).col == 8);

} // namespace ninfer::ops::detail::gated_delta_net::volta
