#pragma once

#if !defined(NINFER_VOLTA_BUILD)
#error "GDN Volta backend is SM70-only"
#endif

#include "ops/linear_attention/gated_delta_net/volta/common.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail::gated_delta_net::volta {

inline constexpr int kPrepareThreads = 256;
inline constexpr int kPrepareWarps = 8;
inline constexpr int kPrepareHalfLd = 130;
inline constexpr std::size_t kPrepareSharedBytes =
    2U * static_cast<std::size_t>(kChunkSize) * kPrepareHalfLd * sizeof(std::uint16_t);

inline constexpr int kPrepareTileJobs[3][4] = {
    {0, 1, 2, 3},
    {4, 5, 6, 7},
    {8, 9, -1, -1},
};

__host__ __device__ constexpr std::size_t prepare_norm_index(int chunk, int qk_head, int row,
                                                              int qk_heads) noexcept {
    return (static_cast<std::size_t>(chunk) * qk_heads + qk_head) * kChunkSize + row;
}

__host__ __device__ constexpr std::size_t prepare_tile_index(int chunk, int qk_head, int tile,
                                                              int element,
                                                              int qk_heads) noexcept {
    return ((static_cast<std::size_t>(chunk) * qk_heads + qk_head) * kLowerTiles + tile) *
               kTileElements +
           element;
}

cudaError_t launch_prepare_qk_matrices(const std::uint16_t* q_bf16,
                                       const std::uint16_t* k_bf16,
                                       float* q_inv_norm,
                                       float* k_inv_norm,
                                       float* kk_tiles,
                                       float* qk_tiles,
                                       int qk_heads,
                                       int chunks,
                                       cudaStream_t stream);

#if defined(__CUDACC__)
__global__ void prepare_qk_matrices_kernel(const std::uint16_t* q_bf16,
                                           const std::uint16_t* k_bf16,
                                           float* q_inv_norm,
                                           float* k_inv_norm,
                                           float* kk_tiles,
                                           float* qk_tiles,
                                           int qk_heads,
                                           int chunks);
#endif

static_assert(kPrepareWarps * 32 == kPrepareThreads);
static_assert(kPrepareSharedBytes == 16640);

} // namespace ninfer::ops::detail::gated_delta_net::volta
