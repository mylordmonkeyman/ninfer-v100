#pragma once

#include "ops/common/memory.cuh"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

#if !defined(NINFER_VOLTA_BUILD)
#error "volta_memory.cuh is an SM70-only implementation boundary"
#endif

// Volta has no cp.async. These routines are intentionally synchronous and have
// no commit/wait API. A caller must place volta_stage_barrier() at the point
// where all threads have finished populating shared memory before any consumer
// reads it.
template <int Bytes>
__device__ __forceinline__ void volta_stage_copy(void* shared_dst, const void* global_src) {
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16,
                  "Volta staging supports 4, 8, or 16-byte vector copies");
    if constexpr (Bytes == 4) {
        const std::uint32_t value =
            *reinterpret_cast<const std::uint32_t*>(global_src);
        *reinterpret_cast<std::uint32_t*>(shared_dst) = value;
    } else if constexpr (Bytes == 8) {
        const uint2 value = *reinterpret_cast<const uint2*>(global_src);
        *reinterpret_cast<uint2*>(shared_dst) = value;
    } else {
        const uint4 value = *reinterpret_cast<const uint4*>(global_src);
        *reinterpret_cast<uint4*>(shared_dst) = value;
    }
}

template <int Bytes>
__device__ __forceinline__ void volta_stage_copy_zfill(void* shared_dst, const void* global_src,
                                                       int src_bytes) {
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16,
                  "Volta staging supports 4, 8, or 16-byte vector copies");
    src_bytes = src_bytes < 0 ? 0 : (src_bytes > Bytes ? Bytes : src_bytes);
    if (src_bytes == Bytes) {
        volta_stage_copy<Bytes>(shared_dst, global_src);
        return;
    }

    auto* dst       = static_cast<std::uint8_t*>(shared_dst);
    const auto* src = static_cast<const std::uint8_t*>(global_src);
#pragma unroll
    for (int i = 0; i < Bytes; ++i) {
        dst[i] = i < src_bytes ? src[i] : std::uint8_t{0};
    }
}

__device__ __forceinline__ void volta_stage_barrier() { __syncthreads(); }

__device__ __forceinline__ void volta_warp_barrier(unsigned mask = 0xFFFFFFFFU) {
    __syncwarp(mask);
}

} // namespace ninfer::ops
