#include "targets/qwen3_8_flash_next/impl/text_decode_kernels.h"

#include "core/device.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::targets::qwen3_8_flash_next::detail {

namespace {

__global__ void repeat_embedding_kernel(const __nv_bfloat16* __restrict__ embedding,
                                        __nv_bfloat16* __restrict__ hyper_hidden, int batch) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;
    if (d >= 2'560 || b >= batch) return;
    const auto val                 = embedding[static_cast<std::int64_t>(b) * 2'560 + d];
    const std::int64_t base        = static_cast<std::int64_t>(b) * 10'240 + d;
    hyper_hidden[base + 0 * 2'560] = val;
    hyper_hidden[base + 1 * 2'560] = val;
    hyper_hidden[base + 2 * 2'560] = val;
    hyper_hidden[base + 3 * 2'560] = val;
}

#if defined(NINFER_VOLTA_BUILD)
__global__ void repeat_embedding_fp32_kernel(
    const __nv_bfloat16* __restrict__ embedding,
    float* __restrict__ hyper_hidden, int batch) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;
    if (d >= 2'560 || b >= batch) return;
    const float val = __bfloat162float(
        embedding[static_cast<std::int64_t>(b) * 2'560 + d]);
    const std::int64_t base = static_cast<std::int64_t>(b) * 10'240 + d;
    #pragma unroll
    for (int stream = 0; stream < 4; ++stream) {
        hyper_hidden[base + stream * 2'560] = val;
    }
}

__global__ void hyper_fp32_to_bf16_kernel(
    const float* __restrict__ source,
    __nv_bfloat16* __restrict__ destination, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        destination[index] = __float2bfloat16_rn(source[index]);
    }
}

__global__ void hyper_add_bf16_to_fp32_kernel(
    const __nv_bfloat16* __restrict__ addend,
    float* __restrict__ destination, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        destination[index] += __bfloat162float(addend[index]);
    }
}

__global__ void hyper_inject_fp32_to_fp32_stage_kernel(
    const float* __restrict__ block_output,
    const float* __restrict__ injection,
    const float* __restrict__ hyper_hidden,
    float* __restrict__ output_stage, int tokens) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int token = blockIdx.y;
    if (index >= 10'240 || token >= tokens) return;
    const int stream = index / 2'560;
    const int hidden = index - stream * 2'560;
    const float out =
        block_output[static_cast<std::int64_t>(token) * 2'560 + hidden];
    const float scale =
        injection[static_cast<std::int64_t>(token) * 4 + stream];
    const std::int64_t offset =
        static_cast<std::int64_t>(token) * 10'240 + index;
    output_stage[offset] = fmaf(out, scale, hyper_hidden[offset]);
}

__global__ void hyper_inject_bf16_to_fp32_kernel(
    const __nv_bfloat16* __restrict__ block_output,
    const float* __restrict__ injection,
    float* __restrict__ hyper_hidden, int tokens) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int token = blockIdx.y;
    if (index >= 10'240 || token >= tokens) return;
    const int stream = index / 2'560;
    const int hidden = index - stream * 2'560;
    const float out = __bfloat162float(
        block_output[static_cast<std::int64_t>(token) * 2'560 + hidden]);
    const float scale =
        injection[static_cast<std::int64_t>(token) * 4 + stream];
    const std::int64_t offset =
        static_cast<std::int64_t>(token) * 10'240 + index;
    hyper_hidden[offset] = fmaf(out, scale, hyper_hidden[offset]);
}
#endif

} // namespace

void repeat_embedding_to_hyper_streams(const Tensor& embedding, Tensor& hyper_hidden,
                                        cudaStream_t stream) {
    const int batch = embedding.ne[1];
    dim3 grid((2'560 + 255) / 256, batch);
    repeat_embedding_kernel<<<grid, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(embedding.data),
        static_cast<__nv_bfloat16*>(hyper_hidden.data), batch);
    CUDA_CHECK(cudaGetLastError());
}

#if defined(NINFER_VOLTA_BUILD)
void repeat_embedding_to_hyper_streams_fp32(
    const Tensor& embedding, Tensor& hyper_hidden, cudaStream_t stream) {
    const int batch = embedding.ne[1];
    dim3 grid((2'560 + 255) / 256, batch);
    repeat_embedding_fp32_kernel<<<grid, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(embedding.data),
        static_cast<float*>(hyper_hidden.data), batch);
    CUDA_CHECK(cudaGetLastError());
}

void hyper_fp32_to_bf16(
    const Tensor& source, Tensor& destination, cudaStream_t stream) {
    const int count = static_cast<int>(source.numel());
    hyper_fp32_to_bf16_kernel<<<(count + 255) / 256, 256, 0, stream>>>(
        static_cast<const float*>(source.data),
        static_cast<__nv_bfloat16*>(destination.data), count);
    CUDA_CHECK(cudaGetLastError());
}

void hyper_add_bf16_to_fp32(
    const Tensor& addend, Tensor& destination, cudaStream_t stream) {
    const int count = static_cast<int>(destination.numel());
    hyper_add_bf16_to_fp32_kernel<<<(count + 255) / 256, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(addend.data),
        static_cast<float*>(destination.data), count);
    CUDA_CHECK(cudaGetLastError());
}

void hyper_inject_fp32_to_fp32_stage(
    const Tensor& block_output, const Tensor& injection,
    const Tensor& hyper_hidden, Tensor& output_stage, cudaStream_t stream) {
    const int tokens = static_cast<int>(hyper_hidden.ne[1]);
    dim3 grid((10'240 + 255) / 256, tokens);
    hyper_inject_fp32_to_fp32_stage_kernel<<<grid, 256, 0, stream>>>(
        static_cast<const float*>(block_output.data),
        static_cast<const float*>(injection.data),
        static_cast<const float*>(hyper_hidden.data),
        static_cast<float*>(output_stage.data), tokens);
    CUDA_CHECK(cudaGetLastError());
}

void hyper_inject_bf16_to_fp32(
    const Tensor& block_output, const Tensor& injection,
    Tensor& hyper_hidden, cudaStream_t stream) {
    const int tokens = static_cast<int>(hyper_hidden.ne[1]);
    dim3 grid((10'240 + 255) / 256, tokens);
    hyper_inject_bf16_to_fp32_kernel<<<grid, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(block_output.data),
        static_cast<const float*>(injection.data),
        static_cast<float*>(hyper_hidden.data), tokens);
    CUDA_CHECK(cudaGetLastError());
}
#endif

} // namespace ninfer::targets::qwen3_8_flash_next::detail
