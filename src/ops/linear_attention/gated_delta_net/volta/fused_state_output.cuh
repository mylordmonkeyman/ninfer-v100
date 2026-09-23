#pragma once

#if !defined(NINFER_VOLTA_BUILD)
#error "GDN Volta backend is SM70-only"
#endif

#include "ops/linear_attention/gated_delta_net/volta/common.cuh"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail::gated_delta_net::volta {

struct FusedOneWindowArgs {
    const std::uint16_t* q_bf16 = nullptr;
    const std::uint16_t* k_bf16 = nullptr;
    const std::uint16_t* v_bf16 = nullptr;
    const float* g = nullptr;
    const float* beta = nullptr;

    const float* state_read = nullptr;
    float* state_write = nullptr;
    std::uint16_t* output_bf16 = nullptr;

    const float* q_inv_norm = nullptr;
    const float* k_inv_norm = nullptr;
    const float* kk_tiles = nullptr;
    const float* qk_tiles = nullptr;

    int qk_heads = 0;
    int value_heads = 0;
    int tokens = 0;
};

cudaError_t launch_fused_state_output_dv16(const FusedOneWindowArgs& args,
                                            cudaStream_t stream);
cudaError_t launch_fused_state_output_dv32(const FusedOneWindowArgs& args,
                                            cudaStream_t stream);

#if defined(__CUDACC__)
__global__ void fused_state_output_dv16_kernel(FusedOneWindowArgs args);
__global__ void fused_state_output_dv32_kernel(FusedOneWindowArgs args);
#endif

} // namespace ninfer::ops::detail::gated_delta_net::volta
