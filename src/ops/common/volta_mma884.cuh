#pragma once

#include "ops/common/memory.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

#if !defined(NINFER_VOLTA_BUILD)
#error "volta_mma.cuh is an SM70-only implementation boundary"
#endif

struct VoltaMma884Operand {
    std::uint32_t x0 = 0;
    std::uint32_t x1 = 0;
};

struct VoltaMma884Accumulator {
    float x[8]{};

    __device__ __forceinline__ void clear() {
#pragma unroll
        for (int i = 0; i < 8; ++i) { x[i] = 0.0F; }
    }
};

struct VoltaMma884Coordinate {
    int row = 0;
    int col = 0;
};

__device__ __forceinline__ unsigned volta_lane_id() {
    unsigned lane = 0;
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane));
    return lane;
}

// An m8n8k4 instruction executed by a warp performs four independent 8x8x4
// MMAs. Lanes 0-3/16-19 own computation 0, 4-7/20-23 computation 1, etc.
__device__ __forceinline__ int volta_mma884_group(unsigned lane) {
    return static_cast<int>((lane & 0x0FU) >> 2);
}

__device__ __forceinline__ int volta_mma884_row_or_col(unsigned lane) {
    return static_cast<int>(lane & 0x03U) + ((lane & 0x10U) != 0U ? 4 : 0);
}

// Load the row-major A fragment required by
// mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32.
// The tile pointer addresses A[0,0], and leading_dimension is measured in half
// elements. K for one instruction is exactly four, so each lane's fragment is
// two packed f16x2 registers from one row.
__device__ __forceinline__ VoltaMma884Operand
volta_mma884_load_a_row(const __half* tile, int leading_dimension) {
    const int row = volta_mma884_row_or_col(volta_lane_id());
    const __half* ptr = tile + static_cast<std::ptrdiff_t>(row) * leading_dimension;
    return {
        load_vec<std::uint32_t>(ptr),
        load_vec<std::uint32_t>(ptr + 2),
    };
}

// B is logically [K,N] and stored column-major so the four K values for one
// output column are contiguous. leading_dimension is the column stride in half
// elements (normally K=4 for one MMA tile).
__device__ __forceinline__ VoltaMma884Operand
volta_mma884_load_b_col(const __half* tile, int leading_dimension) {
    const int col = volta_mma884_row_or_col(volta_lane_id());
    const __half* ptr = tile + static_cast<std::ptrdiff_t>(col) * leading_dimension;
    return {
        load_vec<std::uint32_t>(ptr),
        load_vec<std::uint32_t>(ptr + 2),
    };
}

__device__ __forceinline__ VoltaMma884Coordinate
volta_mma884_accumulator_coordinate(unsigned lane, int index) {
    const int row_low = static_cast<int>(lane & 0x01U) + (index & 0x02);
    return {
        .row = row_low + ((lane & 0x10U) != 0U ? 4 : 0),
        .col = (index & 0x04) + static_cast<int>(lane & 0x02U) + (index & 0x01),
    };
}

__device__ __forceinline__ void
volta_mma884_f16_f32(VoltaMma884Accumulator& c, VoltaMma884Operand a,
                     VoltaMma884Operand b) {
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, "
        "{%8,%9}, "
        "{%10,%11}, "
        "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
        : "+f"(c.x[0]), "+f"(c.x[1]), "+f"(c.x[2]), "+f"(c.x[3]),
          "+f"(c.x[4]), "+f"(c.x[5]), "+f"(c.x[6]), "+f"(c.x[7])
        : "r"(a.x0), "r"(a.x1), "r"(b.x0), "r"(b.x1));
}

// Register-level bridge for streamed software-dequant kernels. It deliberately
// reuses the same typed m8n8k4 primitive so QPN backends do not carry their own
// inline PTX or fragment contract.
__device__ __forceinline__ void
volta_mma884_f16_f32_raw(float (&accum)[8], std::uint32_t a0, std::uint32_t a1,
                         std::uint32_t b0, std::uint32_t b1) {
    VoltaMma884Accumulator c{};
#pragma unroll
    for (int i = 0; i < 8; ++i) { c.x[i] = accum[i]; }
    volta_mma884_f16_f32(c, VoltaMma884Operand{a0, a1}, VoltaMma884Operand{b0, b1});
#pragma unroll
    for (int i = 0; i < 8; ++i) { accum[i] = c.x[i]; }
}

} // namespace ninfer::ops
