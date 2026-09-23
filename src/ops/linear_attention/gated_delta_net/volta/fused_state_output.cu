#include "ops/linear_attention/gated_delta_net/volta/fused_state_output.cuh"

#include "ops/linear_attention/gated_delta_net/volta/fused_state_output_impl.cuh"
#include "ops/linear_attention/gated_delta_net/volta/fused_state_output_dv32_impl.cuh"
#include "ops/linear_attention/gated_delta_net/volta/launch.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace ninfer::ops::detail::gated_delta_net::volta {

using namespace fused_detail;

__global__ __launch_bounds__(Dv16Schedule::kThreads, Dv16Schedule::kMinBlocksPerSm)
void fused_state_output_dv16_kernel(FusedOneWindowArgs args) {
    extern __shared__ unsigned char shared_bytes[];

    const int jobs_per_qk_head = kGroupSize * Dv16Schedule::kDvStrips;
    const int job = static_cast<int>(blockIdx.x);
    const int qk_head = job / jobs_per_qk_head;
    const int within_qk = job - qk_head * jobs_per_qk_head;
    const int group_member = within_qk / Dv16Schedule::kDvStrips;
    const int strip = within_qk - group_member * Dv16Schedule::kDvStrips;
    const int value_head = qk_head * kGroupSize + group_member;

    unsigned char* region_a = shared_bytes + Dv16Shared::A;
    unsigned char* region_b = shared_bytes + Dv16Shared::B;
    unsigned char* region_c = shared_bytes + Dv16Shared::C;
    unsigned char* region_d = shared_bytes + Dv16Shared::D;
    unsigned char* region_e = shared_bytes + Dv16Shared::E;

    __half* q_half = reinterpret_cast<__half*>(region_a);
    __half* k_half = reinterpret_cast<__half*>(region_b);

    float h[4][4];
    load_state(h, args, value_head, strip);

    const int chunks = args.tokens / kChunkSize;
    for (int chunk = 0; chunk < chunks; ++chunk) {
        stage_qk_chunk(args, chunk, qk_head, q_half, k_half);
        process_value_head_chunk_dv16(
            args, chunk, qk_head, value_head, strip, q_half, k_half,
            region_a, region_c, region_d, region_e, h);
    }

    store_state(h, args, value_head, strip);
}

cudaError_t launch_fused_state_output_dv16(const FusedOneWindowArgs& args,
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

    const int grid =
        args.qk_heads * kGroupSize * Dv16Schedule::kDvStrips;
    fused_state_output_dv16_kernel<<<grid, Dv16Schedule::kThreads,
                                      Dv16Shared::Bytes, stream>>>(args);
    return cudaGetLastError();
}


__global__ __launch_bounds__(GdnSchedule<32>::kThreads, GdnSchedule<32>::kMinBlocksPerSm)
void fused_state_output_dv32_kernel(FusedOneWindowArgs args) {
    using namespace fused_dv32_detail;
    extern __shared__ unsigned char shared_bytes[];

    const int jobs_per_qk_head = kGroupSize * Dv32Schedule::kDvStrips;
    const int job = static_cast<int>(blockIdx.x);
    const int qk_head = job / jobs_per_qk_head;
    const int within_qk = job - qk_head * jobs_per_qk_head;
    const int group_member = within_qk / Dv32Schedule::kDvStrips;
    const int strip = within_qk - group_member * Dv32Schedule::kDvStrips;
    const int value_head = qk_head * kGroupSize + group_member;

    unsigned char* region_a = shared_bytes + Dv32Shared::A;
    unsigned char* region_b = shared_bytes + Dv32Shared::B;
    unsigned char* region_c = shared_bytes + Dv32Shared::C;
    unsigned char* region_d = shared_bytes + Dv32Shared::D;
    unsigned char* region_e = shared_bytes + Dv32Shared::E;

    __half* q_half = reinterpret_cast<__half*>(region_a);
    __half* k_half = reinterpret_cast<__half*>(region_b);

    float h[4][4];
    load_state(h, args, value_head, strip);

    const int chunks = args.tokens / kChunkSize;
    for (int chunk = 0; chunk < chunks; ++chunk) {
        stage_qk_chunk(args, chunk, qk_head, q_half, k_half);
        process_value_head_chunk_dv32(
            args, chunk, qk_head, value_head, strip, q_half, k_half,
            region_a, region_c, region_d, region_e, h);
    }

    store_state(h, args, value_head, strip);
}

cudaError_t launch_fused_state_output_dv32(const FusedOneWindowArgs& args,
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

    const int grid =
        args.qk_heads * kGroupSize * GdnSchedule<32>::kDvStrips;
    fused_state_output_dv32_kernel<<<grid, GdnSchedule<32>::kThreads,
                                      FusedSharedLayout<32>::Bytes, stream>>>(args);
    return cudaGetLastError();
}

} // namespace ninfer::ops::detail::gated_delta_net::volta
