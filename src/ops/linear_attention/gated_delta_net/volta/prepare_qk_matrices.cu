#include "ops/linear_attention/gated_delta_net/volta/prepare_qk_matrices.cuh"

#include "ops/common/volta_mma.cuh"
#include "ops/linear_attention/gated_delta_net/volta/bf16_sm70.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail::gated_delta_net::volta {
namespace {

struct alignas(16) PrepareShared {
    __half q[kChunkSize * kPrepareHalfLd];
    __half k[kChunkSize * kPrepareHalfLd];
};

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffU, value, offset);
    }
    return value;
}

__device__ __forceinline__ VoltaMma884Operand load_a_or_zero(const __half* matrix,
                                                              int block_row,
                                                              int k_offset,
                                                              bool active) {
    if (!active) { return {}; }
    const __half* tile = matrix + block_row * kTile * kPrepareHalfLd + k_offset;
    return ninfer::ops::volta_mma884_load_a_row(tile, kPrepareHalfLd);
}

__device__ __forceinline__ VoltaMma884Operand load_b_or_zero(const __half* matrix,
                                                              int block_col,
                                                              int k_offset,
                                                              bool active) {
    if (!active) { return {}; }
    const __half* tile = matrix + block_col * kTile * kPrepareHalfLd + k_offset;
    return ninfer::ops::volta_mma884_load_b_col(tile, kPrepareHalfLd);
}

__device__ __forceinline__ void write_tile(const VoltaMma884Accumulator& accumulator,
                                            float* tiles,
                                            int chunk,
                                            int qk_head,
                                            int tile,
                                            int qk_heads,
                                            unsigned lane) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const auto coordinate = ninfer::ops::volta_mma884_accumulator_coordinate(lane, i);
        const int element = coordinate.row * kTile + coordinate.col;
        tiles[prepare_tile_index(chunk, qk_head, tile, element, qk_heads)] = accumulator.x[i];
    }
}

} // namespace

__global__ void prepare_qk_matrices_kernel(const std::uint16_t* q_bf16,
                                           const std::uint16_t* k_bf16,
                                           float* q_inv_norm,
                                           float* k_inv_norm,
                                           float* kk_tiles,
                                           float* qk_tiles,
                                           int qk_heads,
                                           int chunks) {
    const int chunk = static_cast<int>(blockIdx.x);
    const int qk_head = static_cast<int>(blockIdx.y);
    if (chunk >= chunks || qk_head >= qk_heads) { return; }

    __shared__ PrepareShared shared;
    const int lane = static_cast<int>(threadIdx.x & 31U);
    const int warp = static_cast<int>(threadIdx.x >> 5);

    for (int row = warp; row < kChunkSize; row += kPrepareWarps) {
        const int token = chunk * kChunkSize + row;
        const std::size_t row_offset =
            (static_cast<std::size_t>(token) * qk_heads + qk_head) * kStateDim;
        const std::uint16_t* q_row = q_bf16 + row_offset;
        const std::uint16_t* k_row = k_bf16 + row_offset;
        const int d_base = lane * 4;

        float q_value[4];
        float k_value[4];
        float q_sum = 0.0F;
        float k_sum = 0.0F;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            q_value[i] = bf16_bits_to_float(q_row[d_base + i]);
            k_value[i] = bf16_bits_to_float(k_row[d_base + i]);
            q_sum += q_value[i] * q_value[i];
            k_sum += k_value[i] * k_value[i];
        }

        q_sum = warp_sum(q_sum);
        k_sum = warp_sum(k_sum);
        float q_inv = lane == 0 ? rsqrtf(q_sum + 1.0e-6F) : 0.0F;
        float k_inv = lane == 0 ? rsqrtf(k_sum + 1.0e-6F) : 0.0F;
        q_inv = __shfl_sync(0xffffffffU, q_inv, 0);
        k_inv = __shfl_sync(0xffffffffU, k_inv, 0);

        if (lane == 0) {
            q_inv_norm[prepare_norm_index(chunk, qk_head, row, qk_heads)] = q_inv;
            k_inv_norm[prepare_norm_index(chunk, qk_head, row, qk_heads)] = k_inv;
        }

#pragma unroll
        for (int i = 0; i < 4; ++i) {
            shared.q[row * kPrepareHalfLd + d_base + i] = __float2half_rn(q_value[i] * q_inv);
            shared.k[row * kPrepareHalfLd + d_base + i] = __float2half_rn(k_value[i] * k_inv);
        }
        if (lane == 0) {
            shared.q[row * kPrepareHalfLd + 128] = __float2half_rn(0.0F);
            shared.q[row * kPrepareHalfLd + 129] = __float2half_rn(0.0F);
            shared.k[row * kPrepareHalfLd + 128] = __float2half_rn(0.0F);
            shared.k[row * kPrepareHalfLd + 129] = __float2half_rn(0.0F);
        }
    }

    __syncthreads();

    if (warp >= 6) { return; }
    const bool is_qk = warp >= 3;
    const int job_warp = is_qk ? warp - 3 : warp;
    const int group = ninfer::ops::volta_mma884_group(static_cast<unsigned>(lane));
    const int tile = prepare_tile_job(job_warp, group);
    const bool active = tile >= 0;
    const TileCoord coordinate = lower_tile_coord(tile);
    const __half* a_matrix = is_qk ? shared.q : shared.k;
    const __half* b_matrix = shared.k;

    VoltaMma884Accumulator accumulator{};
    accumulator.clear();
#pragma unroll
    for (int k_offset = 0; k_offset < kStateDim; k_offset += 4) {
        const VoltaMma884Operand a =
            load_a_or_zero(a_matrix, coordinate.row, k_offset, active);
        const VoltaMma884Operand b =
            load_b_or_zero(b_matrix, coordinate.col, k_offset, active);
        ninfer::ops::volta_mma884_f16_f32(accumulator, a, b);
    }

    if (active) {
        write_tile(accumulator, is_qk ? qk_tiles : kk_tiles, chunk, qk_head, tile,
                   qk_heads, static_cast<unsigned>(lane));
    }
}

cudaError_t launch_prepare_qk_matrices(const std::uint16_t* q_bf16,
                                       const std::uint16_t* k_bf16,
                                       float* q_inv_norm,
                                       float* k_inv_norm,
                                       float* kk_tiles,
                                       float* qk_tiles,
                                       int qk_heads,
                                       int chunks,
                                       cudaStream_t stream) {
    if (q_bf16 == nullptr || k_bf16 == nullptr || q_inv_norm == nullptr ||
        k_inv_norm == nullptr || kk_tiles == nullptr || qk_tiles == nullptr ||
        qk_heads <= 0 || chunks <= 0) {
        return cudaErrorInvalidValue;
    }

    const dim3 grid(static_cast<unsigned>(chunks), static_cast<unsigned>(qk_heads), 1U);
    prepare_qk_matrices_kernel<<<grid, kPrepareThreads, 0, stream>>>(
        q_bf16, k_bf16, q_inv_norm, k_inv_norm, kk_tiles, qk_tiles, qk_heads, chunks);
    return cudaGetLastError();
}

} // namespace ninfer::ops::detail::gated_delta_net::volta
