#include "ops/linear_attention/gated_delta_net/volta/fused_grouped_dv16.cuh"

#include "ops/linear_attention/gated_delta_net/volta/fused_state_output_impl.cuh"
#include "ops/linear_attention/gated_delta_net/volta/launch.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace ninfer::ops::detail::gated_delta_net::volta {

using namespace fused_detail;

__global__ __launch_bounds__(Dv16Schedule::kThreads, 2)
void fused_grouped_dv16_kernel(FusedOneWindowArgs args) {
    extern __shared__ unsigned char shared_bytes[];

    const int job = static_cast<int>(blockIdx.x);
    const int qk_head = job / Dv16Schedule::kDvStrips;
    const int strip = job - qk_head * Dv16Schedule::kDvStrips;

    unsigned char* region_a = shared_bytes + Dv16Shared::A;
    unsigned char* region_b = shared_bytes + Dv16Shared::B;
    unsigned char* region_c = shared_bytes + Dv16Shared::C;
    unsigned char* region_d = shared_bytes + Dv16Shared::D;
    unsigned char* region_e = shared_bytes + Dv16Shared::E;

    __half* q_half =
        reinterpret_cast<__half*>(shared_bytes + GroupedDv16SharedLayout::Q);
    __half* k_half = reinterpret_cast<__half*>(region_b);

    float h[kGroupSize][4][4];
#pragma unroll
    for (int group_member = 0; group_member < kGroupSize; ++group_member) {
        const int value_head = qk_head * kGroupSize + group_member;
        load_state(h[group_member], args, value_head, strip);
    }

    const int chunks = args.tokens / kChunkSize;
    for (int chunk = 0; chunk < chunks; ++chunk) {
        stage_qk_chunk(args, chunk, qk_head, q_half, k_half);

#pragma unroll
        for (int group_member = 0; group_member < kGroupSize; ++group_member) {
            const int value_head = qk_head * kGroupSize + group_member;
            process_value_head_chunk_dv16(
                args, chunk, qk_head, value_head, strip, q_half, k_half,
                region_a, region_c, region_d, region_e, h[group_member]);
        }
    }

#pragma unroll
    for (int group_member = 0; group_member < kGroupSize; ++group_member) {
        const int value_head = qk_head * kGroupSize + group_member;
        store_state(h[group_member], args, value_head, strip);
    }
}

cudaError_t launch_fused_grouped_dv16(const FusedOneWindowArgs& args,
                                      cudaStream_t stream) {
    if (args.q_bf16 == nullptr || args.k_bf16 == nullptr || args.v_bf16 == nullptr ||
        args.g == nullptr || args.beta == nullptr || args.state_read == nullptr ||
        args.state_write == nullptr || args.output_bf16 == nullptr ||
        args.q_inv_norm == nullptr || args.k_inv_norm == nullptr ||
        args.kk_tiles == nullptr || args.qk_tiles == nullptr ||
        args.qk_heads <= 0 || args.value_heads <= 0 ||
        args.value_heads != args.qk_heads * kGroupSize ||
        args.tokens < kChunkSize || args.tokens > kWindowTokens ||
        args.tokens % kChunkSize != 0) {
        return cudaErrorInvalidValue;
    }

    cudaError_t status = ensure_runtime_initialized(stream);
    if (status != cudaSuccess) { return status; }

    const int grid = args.qk_heads * Dv16Schedule::kDvStrips;
    fused_grouped_dv16_kernel<<<grid, Dv16Schedule::kThreads,
                                 GroupedDv16SharedLayout::Bytes, stream>>>(args);
    return cudaGetLastError();
}

} // namespace ninfer::ops::detail::gated_delta_net::volta
