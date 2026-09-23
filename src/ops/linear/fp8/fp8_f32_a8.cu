#include "ops/linear/fp8/fp8_f32_a8_plan.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/warp.cuh"
#include "ops/linear/fp8/fp8_config.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

template <int SEGMENTS_PER_ROW>
__device__ __forceinline__ int fp8_f32_swizzle_byte(int row, int logical_byte) {
    const int logical_segment  = logical_byte >> 4;
    const int byte_in_segment  = logical_byte & 15;
    const int physical_segment = logical_segment ^ (row & (SEGMENTS_PER_ROW - 1));
    return physical_segment * 16 + byte_in_segment;
}

template <int N_DIM, int K_DIM, int BM = 64, int BN = 128, int BK = 128, int STAGES = 2,
          int WARP_TOKENS = 2, int WARP_ROWS = 2, int SPLIT_K = 1>
__global__ __launch_bounds__(WARP_TOKENS * WARP_ROWS * 32, 2)
void fp8_f32_gemm_mma_kernel(
    const std::uint8_t* __restrict__ act_codes,
    const float* __restrict__ act_scales,
    const std::uint8_t* __restrict__ weight_codes,
    const float* __restrict__ weight_scales,
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ split_workspace,
    int tokens) {
    
    constexpr int WARPS       = WARP_TOKENS * WARP_ROWS;
    constexpr int THREADS     = WARPS * 32;
    constexpr int KW_TOKENS   = BM / WARP_TOKENS;
    constexpr int KW_ROWS     = BN / WARP_ROWS;
    constexpr int MMA_TOKENS  = KW_TOKENS / 16;
    constexpr int MMA_ROWS    = KW_ROWS / 8;
    constexpr int MMA_K       = BK / 32;

    constexpr int SEGMENTS_PER_ROW_ACT = BK / 16;
    constexpr int SEGMENTS_PER_ROW_WGT = BK / 16;

    constexpr int K_PER_SPLIT = K_DIM / SPLIT_K;
    constexpr int TILES_K     = K_PER_SPLIT / BK;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    auto* act_smem = reinterpret_cast<std::uint8_t*>(smem_raw);
    auto* wgt_smem = act_smem + STAGES * BM * BK;

    const int tid      = static_cast<int>(threadIdx.x);
    const int warp     = tid >> 5;
    const int lane     = tid & 31;
    const int warp_tok = warp / WARP_ROWS;
    const int warp_row = warp % WARP_ROWS;

    const int row_tile    = static_cast<int>(blockIdx.x);
    const int token_tile  = static_cast<int>(blockIdx.y);
    const int split_idx   = static_cast<int>(blockIdx.z);

    const int row_begin   = row_tile * BN;
    const int token_begin = token_tile * BM;
    const int k_start     = split_idx * K_PER_SPLIT;

    auto stage_inputs = [&](int stage, int k_tile) {
        const int k_base = k_start + k_tile * BK;
        auto* act_stage  = act_smem + stage * BM * BK;
        auto* wgt_stage  = wgt_smem + stage * BN * BK;

        #pragma unroll 1
        for (int task = tid; task < BM * SEGMENTS_PER_ROW_ACT; task += THREADS) {
            const int r         = task / SEGMENTS_PER_ROW_ACT;
            const int seg       = task % SEGMENTS_PER_ROW_ACT;
            const int log_byte  = seg * 16;
            const int phys_byte = fp8_f32_swizzle_byte<SEGMENTS_PER_ROW_ACT>(r, log_byte);
            auto* dst           = act_stage + r * BK + phys_byte;
            const int tok       = token_begin + r;
            const bool valid    = tok < tokens;
            cp_async_zfill<16, Cache::cg>(
                dst,
                act_codes + static_cast<std::int64_t>(valid ? tok : 0) * K_DIM + k_base + log_byte,
                valid ? 16 : 0);
        }

        #pragma unroll 1
        for (int task = tid; task < BN * SEGMENTS_PER_ROW_WGT; task += THREADS) {
            const int r         = task / SEGMENTS_PER_ROW_WGT;
            const int seg       = task % SEGMENTS_PER_ROW_WGT;
            const int log_byte  = seg * 16;
            const int phys_byte = fp8_f32_swizzle_byte<SEGMENTS_PER_ROW_WGT>(r, log_byte);
            const int w_row     = row_begin + r;
            const bool valid    = w_row < N_DIM;
            cp_async_zfill<16, Cache::cg>(
                wgt_stage + r * BK + phys_byte,
                weight_codes + static_cast<std::int64_t>(valid ? w_row : 0) * K_DIM + k_base + log_byte,
                valid ? 16 : 0);
        }
    };

    #pragma unroll
    for (int stage = 0; stage < STAGES; ++stage) {
        stage_inputs(stage, stage);
        cp_commit();
    }

    float accumulators[MMA_TOKENS][MMA_ROWS][4] = {};
    const int a_matrix      = lane >> 3;
    const int a_row_offset  = (lane & 7) + ((a_matrix & 1) << 3);
    const int a_col_byte    = (a_matrix >> 1) * 16;
    const int b_row_offset  = lane & 7;
    const int b_column_byte = ((lane >> 3) & 1) * 16;

    #pragma unroll 1
    for (int k_tile = 0; k_tile < TILES_K; ++k_tile) {
        const int stage = k_tile % STAGES;
        if (k_tile + STAGES <= TILES_K) {
            cp_wait<STAGES - 1>();
        } else {
            cp_wait<0>();
        }
        __syncthreads();

        auto load_fragments = [&](int k_step, unsigned(&a_frags)[MMA_TOKENS][4],
                                  unsigned(&b_frags)[MMA_ROWS][2]) {
            #pragma unroll
            for (int mt = 0; mt < MMA_TOKENS; ++mt) {
                const int r         = warp_tok * KW_TOKENS + mt * 16 + a_row_offset;
                const int log_byte  = k_step * 32 + a_col_byte;
                const int phys_byte = fp8_f32_swizzle_byte<SEGMENTS_PER_ROW_ACT>(r, log_byte);
                ldmatrix_x4(a_frags[mt][0], a_frags[mt][1], a_frags[mt][2], a_frags[mt][3],
                            smem_addr(act_smem + stage * BM * BK + r * BK + phys_byte));
            }
            #pragma unroll
            for (int mr = 0; mr < MMA_ROWS; ++mr) {
                const int r         = warp_row * KW_ROWS + mr * 8 + b_row_offset;
                const int log_byte  = k_step * 32 + b_column_byte;
                const int phys_byte = fp8_f32_swizzle_byte<SEGMENTS_PER_ROW_WGT>(r, log_byte);
                ldmatrix_x2(b_frags[mr][0], b_frags[mr][1],
                            smem_addr(wgt_smem + stage * BN * BK + r * BK + phys_byte));
            }
        };

        unsigned a_fragments[2][MMA_TOKENS][4];
        unsigned b_fragments[2][MMA_ROWS][2];
        load_fragments(0, a_fragments[0], b_fragments[0]);

        #pragma unroll
        for (int k_step = 0; k_step < MMA_K; ++k_step) {
            const int slot = k_step & 1;
            if (k_step + 1 < MMA_K) {
                load_fragments(k_step + 1, a_fragments[slot ^ 1], b_fragments[slot ^ 1]);
            }
            #pragma unroll
            for (int mt = 0; mt < MMA_TOKENS; ++mt) {
                #pragma unroll
                for (int mr = 0; mr < MMA_ROWS; ++mr) {
                    mma_fp8_e4m3(
                        accumulators[mt][mr][0], accumulators[mt][mr][1],
                        accumulators[mt][mr][2], accumulators[mt][mr][3],
                        a_fragments[slot][mt][0], a_fragments[slot][mt][1],
                        a_fragments[slot][mt][2], a_fragments[slot][mt][3],
                        b_fragments[slot][mr][0], b_fragments[slot][mr][1]);
                }
            }
        }

        __syncthreads();
        if (k_tile + STAGES < TILES_K) {
            stage_inputs((k_tile + STAGES) % STAGES, k_tile + STAGES);
            cp_commit();
        }
    }

    const int accumulator_token = lane >> 2;
    const int accumulator_row   = 2 * (lane & 3);
    constexpr int output_stride = BN + 8;
    auto* shared_output         = reinterpret_cast<__nv_bfloat16*>(smem_raw);

    #pragma unroll
    for (int mt = 0; mt < MMA_TOKENS; ++mt) {
        const int token0     = token_begin + warp_tok * KW_TOKENS + mt * 16 + accumulator_token;
        const int token1     = token0 + 8;
        const float a_scale0 = token0 < tokens ? act_scales[token0] : 0.0F;
        const float a_scale1 = token1 < tokens ? act_scales[token1] : 0.0F;

        #pragma unroll
        for (int mr = 0; mr < MMA_ROWS; ++mr) {
            const int local_row0 = warp_row * KW_ROWS + mr * 8 + accumulator_row;
            const int r0         = row_begin + local_row0;
            const int r1         = r0 + 1;

            const float w_scale0 = r0 < N_DIM ? weight_scales[r0] : 0.0F;
            const float w_scale1 = r1 < N_DIM ? weight_scales[r1] : 0.0F;

            const float val00 = accumulators[mt][mr][0] * a_scale0 * w_scale0;
            const float val01 = accumulators[mt][mr][1] * a_scale0 * w_scale1;
            const float val10 = accumulators[mt][mr][2] * a_scale1 * w_scale0;
            const float val11 = accumulators[mt][mr][3] * a_scale1 * w_scale1;

            if constexpr (SPLIT_K == 1) {
                if (token0 < tokens) {
                    auto* dst = reinterpret_cast<__nv_bfloat162*>(
                        shared_output + (token0 - token_begin) * output_stride + local_row0);
                    *dst = __floats2bfloat162_rn(val00, val01);
                }
                if (token1 < tokens) {
                    auto* dst = reinterpret_cast<__nv_bfloat162*>(
                        shared_output + (token1 - token_begin) * output_stride + local_row0);
                    *dst = __floats2bfloat162_rn(val10, val11);
                }
            } else {
                if (token0 < tokens) {
                    if (r0 < N_DIM) split_workspace[(static_cast<std::int64_t>(split_idx) * N_DIM + r0) * tokens + token0] = val00;
                    if (r1 < N_DIM) split_workspace[(static_cast<std::int64_t>(split_idx) * N_DIM + r1) * tokens + token0] = val01;
                }
                if (token1 < tokens) {
                    if (r0 < N_DIM) split_workspace[(static_cast<std::int64_t>(split_idx) * N_DIM + r0) * tokens + token1] = val10;
                    if (r1 < N_DIM) split_workspace[(static_cast<std::int64_t>(split_idx) * N_DIM + r1) * tokens + token1] = val11;
                }
            }
        }
    }

    if constexpr (SPLIT_K == 1) {
        __syncthreads();
        constexpr int vectors_per_token = BN / 8;
        constexpr int output_vectors    = BM * vectors_per_token;
        for (int task = tid; task < output_vectors; task += THREADS) {
            const int token_local = task / vectors_per_token;
            const int row_vector  = task - token_local * vectors_per_token;
            const int token       = token_begin + token_local;
            if (token < tokens) {
                const uint4 values   = load_vec<uint4>(shared_output + token_local * output_stride + row_vector * 8);
                const int global_row = row_begin + row_vector * 8;
                if (global_row + 8 <= N_DIM) {
                    store_vec(output + static_cast<std::int64_t>(token) * N_DIM + global_row, values);
                } else {
                    auto* src_bf = reinterpret_cast<const __nv_bfloat16*>(&values);
                    for (int i = 0; i < 8; ++i) {
                        if (global_row + i < N_DIM) {
                            output[static_cast<std::int64_t>(token) * N_DIM + global_row + i] = src_bf[i];
                        }
                    }
                }
            }
        }
    }
}

template <int SPLIT_K = 4>
__global__ void __launch_bounds__(256)
split_k_reduction_kernel(
    const float* __restrict__ split_workspace,
    __nv_bfloat16* __restrict__ output,
    int N_DIM, int tokens) {
    
    const int tid            = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    const int total_elements = N_DIM * tokens;

    if (tid >= total_elements) { return; }

    const int token = tid / N_DIM;
    const int row   = tid % N_DIM;

    float sum = 0.0F;
    #pragma unroll
    for (int s = 0; s < SPLIT_K; ++s) {
        sum += split_workspace[(static_cast<std::int64_t>(s) * N_DIM + row) * tokens + token];
    }

    output[static_cast<std::int64_t>(token) * N_DIM + row] = __float2bfloat16_rn(sum);
}

int flash_next_residual_split_k(std::int32_t tokens) {
    const char* env = std::getenv("NINFER_FLASH_NEXT_RESIDUAL_SPLITK");
    if (env != nullptr && env[0] != '\0') {
        if (std::strcmp(env, "1") == 0) { return 1; }
        if (std::strcmp(env, "4") == 0) { return 4; }
    }
    // T-wide residual was tile-starved at small T (hence split-K=4). Unsplit when T is large
    // enough that the unsplit CTA grid covers the device with at least two waves, or T>=512.
    if (tokens >= 512) { return 1; }
    constexpr int kBM = 64;
    constexpr int kBN = 128;
    constexpr int kN  = 2560;
    const int row_tiles   = (kN + kBN - 1) / kBN;
    const int token_tiles = (tokens + kBM - 1) / kBM;
    const int ctas        = row_tiles * token_tiles;
    int sms               = 0;
    if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0) != cudaSuccess ||
        sms <= 0) {
        return kFlashNextResidualSplitK;
    }
    if (ctas >= sms * 2) { return 1; }
    return kFlashNextResidualSplitK;
}

} // namespace

void launch_fp8_f32_a8(const Tensor& x, const Weight& weight, Tensor& out,
                       WorkspaceArena& workspace, cudaStream_t stream) {
    const std::int32_t tokens      = x.ne[1];
    const std::int32_t output_rows = weight.n;
    const std::int32_t input_rows  = weight.k;

    const auto scope                  = workspace.scope();
    const Fp8F32A8Workspace ws_ptrs   = allocate_fp8_f32_a8_workspace(workspace, output_rows, input_rows, tokens);

    // 1. Quantize activations using the existing house quantizer
    launch_fp8_a8_quantize(x, weight, {ws_ptrs.codes, ws_ptrs.scales}, stream);

    // 2. Dispatch native FP8 GEMM MMA
    constexpr int BM = 64, BN = 128, BK = 128, STAGES = 2;
    constexpr int SMEM = STAGES * (BM + BN) * BK;

    const int row_tiles   = (output_rows + BN - 1) / BN;
    const int token_tiles = (tokens + BM - 1) / BM;

    const auto* act_codes    = ws_ptrs.codes;
    const auto* act_scales   = ws_ptrs.scales;
    const auto* weight_codes = static_cast<const std::uint8_t*>(weight.qdata);
    const auto* weight_scales= static_cast<const float*>(weight.scales);
    auto* out_data           = static_cast<__nv_bfloat16*>(out.data);

    if (output_rows == 16384 && input_rows == 2560) {
        dim3 grid(row_tiles, token_tiles, 1);
        fp8_f32_gemm_mma_kernel<16384, 2560, BM, BN, BK, STAGES, 2, 2, 1>
            <<<grid, 128, SMEM, stream>>>(
                act_codes, act_scales, weight_codes, weight_scales, out_data, nullptr, tokens);
        CUDA_CHECK(cudaGetLastError());
    } else if (output_rows == 13312 && input_rows == 2560) {
        dim3 grid(row_tiles, token_tiles, 1);
        fp8_f32_gemm_mma_kernel<13312, 2560, BM, BN, BK, STAGES, 2, 2, 1>
            <<<grid, 128, SMEM, stream>>>(
                act_codes, act_scales, weight_codes, weight_scales, out_data, nullptr, tokens);
        CUDA_CHECK(cudaGetLastError());
    } else if (output_rows == 2560 && input_rows == 6144) {
        const int split_k = flash_next_residual_split_k(tokens);
        if (split_k == 1) {
            dim3 grid(row_tiles, token_tiles, 1);
            fp8_f32_gemm_mma_kernel<2560, 6144, BM, BN, BK, STAGES, 2, 2, 1>
                <<<grid, 128, SMEM, stream>>>(act_codes, act_scales, weight_codes, weight_scales,
                                              out_data, nullptr, tokens);
            CUDA_CHECK(cudaGetLastError());
        } else {
            constexpr int kSplit4 = kFlashNextResidualSplitK;
            dim3 grid(row_tiles, token_tiles, kSplit4);
            fp8_f32_gemm_mma_kernel<2560, 6144, BM, BN, BK, STAGES, 2, 2, kSplit4>
                <<<grid, 128, SMEM, stream>>>(act_codes, act_scales, weight_codes, weight_scales,
                                              out_data, ws_ptrs.split_workspace, tokens);
            CUDA_CHECK(cudaGetLastError());
            const int total_elements = output_rows * tokens;
            const int reduce_threads = 256;
            const int reduce_blocks  = (total_elements + reduce_threads - 1) / reduce_threads;
            split_k_reduction_kernel<kSplit4><<<reduce_blocks, reduce_threads, 0, stream>>>(
                ws_ptrs.split_workspace, out_data, output_rows, tokens);
            CUDA_CHECK(cudaGetLastError());
        }
    } else {
        throw std::invalid_argument("launch_fp8_f32_a8: unsupported geometry");
    }
}

} // namespace ninfer::ops::detail
