#pragma once

#if !defined(NINFER_VOLTA_BUILD)
#error "GDN Volta backend is SM70-only"
#endif

#include "ops/common/volta_mma.cuh"
#include "ops/linear_attention/gated_delta_net/volta/bf16_sm70.cuh"
#include "ops/linear_attention/gated_delta_net/volta/fused_state_output.cuh"
#include "ops/linear_attention/gated_delta_net/volta/mma_tiles.cuh"
#include "ops/linear_attention/gated_delta_net/volta/prepare_qk_matrices.cuh"
#include "ops/linear_attention/gated_delta_net/volta/scaling.cuh"
#include "ops/linear_attention/gated_delta_net/volta/schedule.cuh"
#include "ops/linear_attention/gated_delta_net/volta/shared_tiles.cuh"
#include "ops/linear_attention/gated_delta_net/volta/triangular_solve.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail::gated_delta_net::volta::fused_dv32_detail {

using Dv32Schedule = GdnSchedule<32>;
using Dv32Shared = FusedSharedLayout<32>;

inline constexpr int kDvTile = 32;
inline constexpr int kXyLd = 2 * kMacroFp32Ld;
inline constexpr int kBridgeLd = 34;
inline constexpr float kOutputScale = 0x1.6a09e6p-4F;

struct DecayShared {
    float alpha[kChunkSize];
    float prefix[kChunkSize];
    float suffix[kChunkSize];
    float beta[kChunkSize];
};

static_assert(sizeof(DecayShared) == 512);
static_assert(2 * kChunkSize * kXyLd * sizeof(float) == 8704);
static_assert(kDvTile * kQkHalfLd * sizeof(__half) == 8320);

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffU, value, offset);
    }
    return value;
}

__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(0xffffffffU, value, offset));
    }
    return value;
}

__device__ __forceinline__ float warp_min(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fminf(value, __shfl_down_sync(0xffffffffU, value, offset));
    }
    return value;
}

__device__ __forceinline__ BridgePlan finish_cta_range(RangeStats local, float* scratch) {
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;

    const float local_min =
        local.min_nonzero_abs == 0.0F ? kBridgeInfinity : local.min_nonzero_abs;
    const float max_value = warp_max(local.max_abs);
    const float min_value = warp_min(local_min);

    if (lane == 0) {
        scratch[warp] = max_value;
        scratch[Dv32Schedule::kWarps + warp] =
            min_value == kBridgeInfinity ? 0.0F : min_value;
    }
    __syncthreads();

    if (tid == 0) {
        RangeStats total{};
        for (int w = 0; w < Dv32Schedule::kWarps; ++w) {
            RangeStats part{
                scratch[w],
                scratch[Dv32Schedule::kWarps + w],
            };
            total = merge_range_stats(total, part);
        }
        scratch[0] = total.max_abs;
        scratch[1] = total.min_nonzero_abs;
    }
    __syncthreads();

    return plan_fp16_bridge({scratch[0], scratch[1]});
}

__device__ __forceinline__ BridgePlan plan_state_range(const float (&h)[4][4],
                                                       float* scratch) {
    RangeStats local{};
#pragma unroll
    for (int r = 0; r < 4; ++r) {
#pragma unroll
        for (int c = 0; c < 4; ++c) { range_observe(local, h[r][c]); }
    }
    return finish_cta_range(local, scratch);
}

__device__ __forceinline__ BridgePlan plan_contiguous_range(const float* values, int count,
                                                            float* scratch) {
    RangeStats local{};
    for (int i = static_cast<int>(threadIdx.x); i < count; i += blockDim.x) {
        range_observe(local, values[i]);
    }
    return finish_cta_range(local, scratch);
}

__device__ __forceinline__ BridgePlan plan_matrix_range(const float* values, int rows, int cols,
                                                        int ld, bool strict_lower,
                                                        float* scratch) {
    RangeStats local{};
    const int count = rows * cols;
    for (int i = static_cast<int>(threadIdx.x); i < count; i += blockDim.x) {
        const int row = i / cols;
        const int col = i - row * cols;
        if (!strict_lower || col < row) { range_observe(local, values[row * ld + col]); }
    }
    return finish_cta_range(local, scratch);
}

__device__ __forceinline__ float load_prepared_lower(const float* tiles, int chunk,
                                                     int qk_head, int row, int col,
                                                     int qk_heads) {
    if (col > row) { return 0.0F; }
    const int block_row = row / kTile;
    const int block_col = col / kTile;
    const int tile = lower_tile_index(block_row, block_col);
    const int element = (row % kTile) * kTile + (col % kTile);
    return tiles[prepare_tile_index(chunk, qk_head, tile, element, qk_heads)];
}

__device__ __forceinline__ float pairwise_decay_shared(const DecayShared& decay, int row,
                                                       int col) {
    if (row <= col) { return 1.0F; }
    float value = 1.0F;
    for (int u = col + 1; u <= row; ++u) { value *= decay.alpha[u]; }
    return value;
}

template <int K, bool TransposeOut, bool AddExisting>
__device__ __forceinline__ void mma_macro16(const __half* a, int a_ld, const __half* b,
                                            int b_ld, float rescale, float* out, int out_ld,
                                            const float* add = nullptr) {
    ninfer::ops::VoltaMma884Accumulator accumulator{};
    accumulator.clear();

#pragma unroll
    for (int k = 0; k < K; k += 4) {
        const auto a_fragment = load_mma884_a_macro16(a, a_ld, k);
        const auto b_fragment = load_mma884_b_macro16_col(b, b_ld, k);
        ninfer::ops::volta_mma884_f16_f32(accumulator, a_fragment, b_fragment);
    }

    const unsigned lane = ninfer::ops::volta_lane_id();
    const int group = ninfer::ops::volta_mma884_group(lane);
    const MmaMacro16Coord group_coord = mma_macro16_group_coord(group);

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const auto local = ninfer::ops::volta_mma884_accumulator_coordinate(lane, i);
        const int row = group_coord.row + local.row;
        const int col = group_coord.col + local.col;
        const int index = TransposeOut ? col * out_ld + row : row * out_ld + col;
        float value = accumulator.x[i] * rescale;
        if constexpr (AddExisting) { value += add[index]; }
        out[index] = value;
    }
}

__device__ __forceinline__ void load_state(float (&h)[4][4], const FusedOneWindowArgs& args,
                                           int value_head, int strip) {
    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int dv_base = strip * kDvTile + warp * Dv32Schedule::kDvPerWarp;
    const int k_base = lane * Dv32Schedule::kStateColsPerLane;
    const float* state =
        args.state_read + static_cast<std::size_t>(value_head) * kStateDim * kStateDim;

#pragma unroll
    for (int r = 0; r < 4; ++r) {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            h[r][c] = state[(dv_base + r) * kStateDim + k_base + c];
        }
    }
}

__device__ __forceinline__ void store_state(const float (&h)[4][4],
                                            const FusedOneWindowArgs& args, int value_head,
                                            int strip) {
    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int dv_base = strip * kDvTile + warp * Dv32Schedule::kDvPerWarp;
    const int k_base = lane * Dv32Schedule::kStateColsPerLane;
    float* state =
        args.state_write + static_cast<std::size_t>(value_head) * kStateDim * kStateDim;

#pragma unroll
    for (int r = 0; r < 4; ++r) {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            state[(dv_base + r) * kStateDim + k_base + c] = h[r][c];
        }
    }
}

__device__ __forceinline__ void stage_qk_chunk(const FusedOneWindowArgs& args, int chunk,
                                         int qk_head, __half* q_half, __half* k_half) {
    const int tid = static_cast<int>(threadIdx.x);
    for (int i = tid; i < kChunkSize * kStateDim; i += blockDim.x) {
        const int row = i / kStateDim;
        const int d = i - row * kStateDim;
        const int token = chunk * kChunkSize + row;
        const std::size_t input_index =
            (static_cast<std::size_t>(token) * args.qk_heads + qk_head) * kStateDim + d;
        const std::size_t norm_index = prepare_norm_index(chunk, qk_head, row, args.qk_heads);
        const float q_value = bf16_bits_to_float(args.q_bf16[input_index]) * args.q_inv_norm[norm_index];
        const float k_value = bf16_bits_to_float(args.k_bf16[input_index]) * args.k_inv_norm[norm_index];
        q_half[row * kQkHalfLd + d] = __float2half_rn(q_value);
        k_half[row * kQkHalfLd + d] = __float2half_rn(k_value);
    }

    if (tid < kChunkSize) {
        q_half[tid * kQkHalfLd + 128] = __float2half_rn(0.0F);
        q_half[tid * kQkHalfLd + 129] = __float2half_rn(0.0F);
        k_half[tid * kQkHalfLd + 128] = __float2half_rn(0.0F);
        k_half[tid * kQkHalfLd + 129] = __float2half_rn(0.0F);
    }
}

__device__ __forceinline__ void stage_state_half(const float (&h)[4][4], __half* h_half,
                                                 BridgePlan plan) {
    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int dv_base = warp * 4;
    const int k_base = lane * 4;

#pragma unroll
    for (int r = 0; r < 4; ++r) {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            h_half[(dv_base + r) * kQkHalfLd + k_base + c] =
                __float2half_rn(h[r][c] * plan.mul);
        }
    }
    if (tid < kDvTile) {
        h_half[tid * kQkHalfLd + 128] = __float2half_rn(0.0F);
        h_half[tid * kQkHalfLd + 129] = __float2half_rn(0.0F);
    }
}

__device__ __forceinline__ void build_decay(const FusedOneWindowArgs& args, int chunk,
                                            int value_head, DecayShared& decay) {
    if (threadIdx.x != 0) { return; }

    float running = 1.0F;
    for (int t = 0; t < kChunkSize; ++t) {
        const int token = chunk * kChunkSize + t;
        const std::size_t gate_index =
            static_cast<std::size_t>(token) * args.value_heads + value_head;
        const float alpha = expf(args.g[gate_index]);
        decay.alpha[t] = alpha;
        decay.beta[t] = args.beta[gate_index];
        running *= alpha;
        decay.prefix[t] = running;
    }

    running = 1.0F;
    for (int t = kChunkSize - 1; t >= 0; --t) {
        decay.suffix[t] = running;
        running *= decay.alpha[t];
    }
}

__device__ __forceinline__ void phase_a_simt(const float (&h)[4][4], const __half* q_half,
                                             const __half* k_half, float* x, float* y) {
    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int k_base = lane * 4;

    for (int t = 0; t < kChunkSize; ++t) {
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            float x_partial = 0.0F;
            float y_partial = 0.0F;
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const float hv = h[r][c];
                x_partial += __half2float(k_half[t * kQkHalfLd + k_base + c]) * hv;
                y_partial += __half2float(q_half[t * kQkHalfLd + k_base + c]) * hv;
            }
            x_partial = warp_sum(x_partial);
            y_partial = warp_sum(y_partial);
            if (lane == 0) {
                const int dv = warp * 4 + r;
                x[t * kXyLd + dv] = x_partial;
                y[t * kXyLd + dv] = y_partial;
            }
        }
    }
}

__device__ __forceinline__ void phase_a_mma(const __half* q_half, const __half* k_half,
                                            const __half* h_half, BridgePlan h_plan,
                                            float* x, float* y) {
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    constexpr int kDvBlocks = kDvTile / 16;
    constexpr int kWarpsPerProduct = 2 * kDvBlocks;
    const bool query = warp >= kWarpsPerProduct;
    const int local_warp = query ? warp - kWarpsPerProduct : warp;
    const int token_block = local_warp / kDvBlocks;
    const int dv_block = local_warp - token_block * kDvBlocks;
    const __half* a =
        (query ? q_half : k_half) + token_block * 16 * kQkHalfLd;
    const __half* b = h_half + dv_block * 16 * kQkHalfLd;
    float* out = (query ? y : x) + token_block * 16 * kXyLd + dv_block * 16;
    mma_macro16<128, false, false>(a, kQkHalfLd, b, kQkHalfLd,
                                   h_plan.inv, out, kXyLd);
}

__device__ __forceinline__ void construct_b_and_r(const FusedOneWindowArgs& args, int chunk,
                                                  int qk_head, const DecayShared& decay,
                                                  float* b, float* r, float* solve_scratch) {
    const int tid = static_cast<int>(threadIdx.x);
    for (int i = tid; i < kChunkSize * kChunkSize; i += blockDim.x) {
        const int row = i / kChunkSize;
        const int col = i - row * kChunkSize;
        float value = 0.0F;
        if (row > col) {
            const float g = load_prepared_lower(args.kk_tiles, chunk, qk_head, row, col,
                                                args.qk_heads);
            value = decay.beta[row] * g * pairwise_decay_shared(decay, row, col);
        }
        b[row * kFp32TileLd + col] = value;
    }
    if (tid < kChunkSize) { b[tid * kFp32TileLd + 32] = 0.0F; }
    __syncthreads();

    if (tid == 0) {
        exact_inverse_bt32(b, kFp32TileLd, r, kFp32TileLd, solve_scratch);
    }
}

__device__ __forceinline__ void construct_d(const FusedOneWindowArgs& args, int chunk,
                                            int value_head, int strip,
                                            const DecayShared& decay, const float* x,
                                            float* d) {
    const int tid = static_cast<int>(threadIdx.x);
    for (int i = tid; i < kChunkSize * kDvTile; i += blockDim.x) {
        const int t = i / kDvTile;
        const int dv = i - t * kDvTile;
        const int token = chunk * kChunkSize + t;
        const int global_dv = strip * kDvTile + dv;
        const std::size_t value_index =
            (static_cast<std::size_t>(token) * args.value_heads + value_head) * kStateDim +
            global_dv;
        const float v = bf16_bits_to_float(args.v_bf16[value_index]);
        d[dv * kChunkSize + t] =
            decay.beta[t] * (v - decay.prefix[t] * x[t * kXyLd + dv]);
    }
}

__device__ __forceinline__ void stage_r_d_half(const float* r, const float* d,
                                               BridgePlan r_plan, BridgePlan d_plan,
                                               __half* r_half, __half* d_half) {
    const int tid = static_cast<int>(threadIdx.x);
    for (int i = tid; i < kChunkSize * kBridgeLd; i += blockDim.x) {
        const int row = i / kBridgeLd;
        const int col = i - row * kBridgeLd;
        const float value = col < kChunkSize && col < row
                                ? r[row * kFp32TileLd + col] * r_plan.mul
                                : 0.0F;
        r_half[i] = __float2half_rn(value);
    }

    // DV32's D bridge destination reaches into the first 128 bytes of the
    // FP32 R source range. All R reads must complete before any D-half write.
    __syncthreads();

    for (int i = tid; i < kDvTile * kBridgeLd; i += blockDim.x) {
        const int dv = i / kBridgeLd;
        const int t = i - dv * kBridgeLd;
        const float value =
            t < kChunkSize ? d[dv * kChunkSize + t] * d_plan.mul : 0.0F;
        d_half[i] = __float2half_rn(value);
    }
}

__device__ __forceinline__ void phase_c_simt(const float* r, const float* d, float* vp) {
    const int tid = static_cast<int>(threadIdx.x);
    for (int i = tid; i < kChunkSize * kDvTile; i += blockDim.x) {
        const int t = i / kDvTile;
        const int dv = i - t * kDvTile;
        float correction = 0.0F;
        for (int s = 0; s < t; ++s) {
            correction += r[t * kFp32TileLd + s] * d[dv * kChunkSize + s];
        }
        vp[dv * kChunkSize + t] = d[dv * kChunkSize + t] + correction;
    }
}

__device__ __forceinline__ void phase_c_mma(const __half* r_half, const __half* d_half,
                                            BridgePlan r_plan, BridgePlan d_plan,
                                            const float* d, float* vp) {
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    constexpr int kDvBlocks = kDvTile / 16;
    constexpr int kActiveWarps = 2 * kDvBlocks;
    if (warp < kActiveWarps) {
        const int token_block = warp / kDvBlocks;
        const int dv_block = warp - token_block * kDvBlocks;
        const std::size_t dv_offset =
            static_cast<std::size_t>(dv_block) * 16 * kChunkSize;
        mma_macro16<32, true, true>(
            r_half + token_block * 16 * kBridgeLd, kBridgeLd,
            d_half + dv_block * 16 * kBridgeLd, kBridgeLd,
            r_plan.inv * d_plan.inv,
            vp + dv_offset + token_block * 16, kChunkSize,
            d + dv_offset + token_block * 16);
    }
}

__device__ __forceinline__ void construct_a(const FusedOneWindowArgs& args, int chunk,
                                            int qk_head, const DecayShared& decay,
                                            float* a) {
    const int tid = static_cast<int>(threadIdx.x);
    for (int i = tid; i < kChunkSize * kChunkSize; i += blockDim.x) {
        const int row = i / kChunkSize;
        const int col = i - row * kChunkSize;
        float value = 0.0F;
        if (col <= row) {
            const float p = load_prepared_lower(args.qk_tiles, chunk, qk_head, row, col,
                                                args.qk_heads);
            value = p * pairwise_decay_shared(decay, row, col);
        }
        a[row * kFp32TileLd + col] = value;
    }
    if (tid < kChunkSize) { a[tid * kFp32TileLd + 32] = 0.0F; }
}

__device__ __forceinline__ void stage_a_vp_half(const float* a, const float* vp,
                                                BridgePlan a_plan, BridgePlan vp_plan,
                                                __half* a_half, __half* vp_half) {
    const int tid = static_cast<int>(threadIdx.x);
    for (int i = tid; i < kChunkSize * kBridgeLd; i += blockDim.x) {
        const int row = i / kBridgeLd;
        const int col = i - row * kBridgeLd;
        const float value =
            col < kChunkSize ? a[row * kFp32TileLd + col] * a_plan.mul : 0.0F;
        a_half[i] = __float2half_rn(value);
    }
    for (int i = tid; i < kDvTile * kBridgeLd; i += blockDim.x) {
        const int dv = i / kBridgeLd;
        const int t = i - dv * kBridgeLd;
        const float value =
            t < kChunkSize ? vp[dv * kChunkSize + t] * vp_plan.mul : 0.0F;
        vp_half[i] = __float2half_rn(value);
    }
}

__device__ __forceinline__ void phase_d_simt(const float* a, const float* vp,
                                             float* local_output) {
    const int tid = static_cast<int>(threadIdx.x);
    for (int i = tid; i < kChunkSize * kDvTile; i += blockDim.x) {
        const int t = i / kDvTile;
        const int dv = i - t * kDvTile;
        float value = 0.0F;
        for (int s = 0; s <= t; ++s) {
            value += a[t * kFp32TileLd + s] * vp[dv * kChunkSize + s];
        }
        local_output[i] = value;
    }
}

__device__ __forceinline__ void phase_d_mma(const __half* a_half, const __half* vp_half,
                                            BridgePlan a_plan, BridgePlan vp_plan,
                                            float* local_output) {
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    constexpr int kDvBlocks = kDvTile / 16;
    constexpr int kActiveWarps = 2 * kDvBlocks;
    if (warp < kActiveWarps) {
        const int token_block = warp / kDvBlocks;
        const int dv_block = warp - token_block * kDvBlocks;
        mma_macro16<32, false, false>(
            a_half + token_block * 16 * kBridgeLd, kBridgeLd,
            vp_half + dv_block * 16 * kBridgeLd, kBridgeLd,
            a_plan.inv * vp_plan.inv,
            local_output + token_block * 16 * kDvTile + dv_block * 16,
            kDvTile);
    }
}

__device__ __forceinline__ void publish_output(const FusedOneWindowArgs& args, int chunk,
                                               int value_head, int strip,
                                               const DecayShared& decay, const float* y,
                                               const float* local_output) {
    const int tid = static_cast<int>(threadIdx.x);
    for (int i = tid; i < kChunkSize * kDvTile; i += blockDim.x) {
        const int t = i / kDvTile;
        const int dv = i - t * kDvTile;
        const int token = chunk * kChunkSize + t;
        const int global_dv = strip * kDvTile + dv;
        const float value =
            kOutputScale *
            (decay.prefix[t] * y[t * kXyLd + dv] + local_output[i]);
        const std::size_t output_index =
            (static_cast<std::size_t>(token) * args.value_heads + value_head) * kStateDim +
            global_dv;
        args.output_bf16[output_index] = float_to_bf16_rn(value);
    }
}

__device__ __forceinline__ void construct_z(const DecayShared& decay, const float* vp,
                                            float* z) {
    const int tid = static_cast<int>(threadIdx.x);
    for (int i = tid; i < kChunkSize * kDvTile; i += blockDim.x) {
        const int t = i / kDvTile;
        const int dv = i - t * kDvTile;
        z[dv * kChunkSize + t] = decay.suffix[t] * vp[dv * kChunkSize + t];
    }
}

__device__ __forceinline__ void stage_kt(const __half* k_half, __half* kt_half) {
    const int tid = static_cast<int>(threadIdx.x);
    for (int i = tid; i < kStateDim * kKtHalfLd; i += blockDim.x) {
        const int d = i / kKtHalfLd;
        const int t = i - d * kKtHalfLd;
        kt_half[i] =
            t < kChunkSize ? k_half[t * kQkHalfLd + d] : __float2half_rn(0.0F);
    }
}

__device__ __forceinline__ void stage_z_half(const float* z, BridgePlan plan,
                                             __half* z_half) {
    const int tid = static_cast<int>(threadIdx.x);
    for (int i = tid; i < kDvTile * kKtHalfLd; i += blockDim.x) {
        const int dv = i / kKtHalfLd;
        const int t = i - dv * kKtHalfLd;
        const float value =
            t < kChunkSize ? z[dv * kChunkSize + t] * plan.mul : 0.0F;
        z_half[i] = __float2half_rn(value);
    }
}

__device__ __forceinline__ void phase_e_simt_macro(const __half* kt_half, const float* z,
                                                   int k_block, float* macro) {
    const unsigned lane = ninfer::ops::volta_lane_id();
    const int group = ninfer::ops::volta_mma884_group(lane);
    const MmaMacro16Coord group_coord = mma_macro16_group_coord(group);

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const auto local = ninfer::ops::volta_mma884_accumulator_coordinate(lane, i);
        const int row = group_coord.row + local.row;
        const int dv = group_coord.col + local.col;
        float value = 0.0F;
#pragma unroll
        for (int t = 0; t < kChunkSize; ++t) {
            value += __half2float(
                         kt_half[(k_block * 16 + row) * kKtHalfLd + t]) *
                     z[dv * kChunkSize + t];
        }
        macro[row * kMacroFp32Ld + dv] = value;
    }
}

__device__ __forceinline__ void consume_delta_macro(float (&h)[4][4], const float* macro_base,
                                                    int round) {
    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int k_base = lane * 4;
    const int k_block = k_base / 16;
    if (k_block < round * 4 || k_block >= (round + 1) * 4) { return; }

    constexpr int kDvBlocks = kDvTile / 16;
    const int dv_base = warp * 4;
    const int dv_block = dv_base / 16;
    const int local_dv_base = dv_base - dv_block * 16;
    const int source_warp =
        (k_block - round * 4) * kDvBlocks + dv_block;
    const float* macro =
        macro_base + source_warp * 16 * kMacroFp32Ld;
    const int row_base = k_base - k_block * 16;

#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int local_dv = local_dv_base + r;
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            h[r][c] += macro[(row_base + c) * kMacroFp32Ld + local_dv];
        }
    }
}


__device__ __forceinline__ void process_value_head_chunk_dv32(
    const FusedOneWindowArgs& args, int chunk, int qk_head, int value_head, int strip,
    const __half* q_half, const __half* k_half, unsigned char* region_a,
    unsigned char* region_c, unsigned char* region_d, unsigned char* region_e,
    float (&h)[4][4]) {
    __half* h_half = reinterpret_cast<__half*>(region_c);
    float* x = reinterpret_cast<float*>(region_d);
    float* y = x + kChunkSize * kXyLd;
    auto* decay = reinterpret_cast<DecayShared*>(region_e);

    BridgePlan h_plan =
        plan_state_range(h, reinterpret_cast<float*>(region_e));
    if (h_plan.mode == BridgeMode::Fp16Mma) {
        stage_state_half(h, h_half, h_plan);
    }
    build_decay(args, chunk, value_head, *decay);

    __syncthreads(); // A1

    if (h_plan.mode == BridgeMode::Fp16Mma) {
        phase_a_mma(q_half, k_half, h_half, h_plan, x, y);
    } else {
        phase_a_simt(h, q_half, k_half, x, y);
    }

    __syncthreads(); // A2

    float* b = reinterpret_cast<float*>(region_a);
    float* r = b + kChunkSize * kFp32TileLd;
    float* solve_scratch = reinterpret_cast<float*>(region_c);
    construct_b_and_r(args, chunk, qk_head, *decay, b, r, solve_scratch);

    __syncthreads(); // B1

    float* d = reinterpret_cast<float*>(region_c);
    float* vp = d + kChunkSize * kDvTile;
    construct_d(args, chunk, value_head, strip, *decay, x, d);
    __syncthreads();

    BridgePlan r_plan =
        plan_matrix_range(r, kChunkSize, kChunkSize, kFp32TileLd, true, b);
    BridgePlan d_plan =
        plan_contiguous_range(d, kChunkSize * kDvTile, b);

    if (r_plan.mode == BridgeMode::Fp16Mma &&
        d_plan.mode == BridgeMode::Fp16Mma) {
        __half* r_half = reinterpret_cast<__half*>(b);
        __half* d_half = r_half + kChunkSize * kBridgeLd;
        stage_r_d_half(r, d, r_plan, d_plan, r_half, d_half);
        __syncthreads();
        phase_c_mma(r_half, d_half, r_plan, d_plan, d, vp);
    } else {
        phase_c_simt(r, d, vp);
    }

    __syncthreads(); // C1

    float* a = reinterpret_cast<float*>(region_a);
    construct_a(args, chunk, qk_head, *decay, a);
    __syncthreads();

    BridgePlan a_plan =
        plan_matrix_range(a, kChunkSize, kChunkSize, kFp32TileLd, false, x);
    BridgePlan vp_plan =
        plan_contiguous_range(vp, kChunkSize * kDvTile, x);

    float* local_output = x;
    if (a_plan.mode == BridgeMode::Fp16Mma &&
        vp_plan.mode == BridgeMode::Fp16Mma) {
        __half* a_half = reinterpret_cast<__half*>(region_a + 4352);
        __half* vp_half = a_half + kChunkSize * kBridgeLd;
        stage_a_vp_half(a, vp, a_plan, vp_plan, a_half, vp_half);
        __syncthreads();
        phase_d_mma(a_half, vp_half, a_plan, vp_plan, local_output);
    } else {
        phase_d_simt(a, vp, local_output);
    }
    __syncthreads();

    publish_output(args, chunk, value_head, strip, *decay, y, local_output);

    __syncthreads(); // D1

    const float final_prefix = decay->prefix[kChunkSize - 1];
#pragma unroll
    for (int rr = 0; rr < 4; ++rr) {
#pragma unroll
        for (int cc = 0; cc < 4; ++cc) { h[rr][cc] *= final_prefix; }
    }

    float* z = d;
    construct_z(*decay, vp, z);
    __syncthreads();

    BridgePlan z_plan =
        plan_contiguous_range(z, kChunkSize * kDvTile, x);

    __half* kt_half = reinterpret_cast<__half*>(region_a);
    stage_kt(k_half, kt_half);
    __half* z_half = reinterpret_cast<__half*>(vp);
    if (z_plan.mode == BridgeMode::Fp16Mma) {
        stage_z_half(z, z_plan, z_half);
    }

    __syncthreads(); // E1

    float* delta_macro = reinterpret_cast<float*>(region_d);
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    constexpr int kDvBlocks = kDvTile / 16;
    constexpr int kKBlocksPerRound = Dv32Schedule::kWarps / kDvBlocks;
    static_assert(kKBlocksPerRound == 4);
    for (int round = 0; round < 2; ++round) {
        const int local_k_block = warp / kDvBlocks;
        const int dv_block = warp - local_k_block * kDvBlocks;
        const int k_block = round * kKBlocksPerRound + local_k_block;
        float* macro =
            delta_macro + warp * 16 * kMacroFp32Ld;
        if (z_plan.mode == BridgeMode::Fp16Mma) {
            mma_macro16<32, false, false>(
                kt_half + k_block * 16 * kKtHalfLd, kKtHalfLd,
                z_half + dv_block * 16 * kKtHalfLd, kKtHalfLd,
                z_plan.inv, macro, kMacroFp32Ld);
        } else {
            phase_e_simt_macro(
                kt_half, z + dv_block * 16 * kChunkSize, k_block, macro);
        }

        __syncthreads(); // E2
        consume_delta_macro(h, delta_macro, round);
        __syncthreads(); // E3
    }
}


} // namespace ninfer::ops::detail::gated_delta_net::volta::fused_dv32_detail
