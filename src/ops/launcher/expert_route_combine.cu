#include "ninfer/ops/expert_route_combine.h"
#include "core/device.h"

#include <stdexcept>

namespace ninfer::ops {
namespace {

__global__ void combine_kernel(const float* routes, const float* alpha, float* output,
                               int tokens, std::size_t pitch) {
    const std::size_t index = std::size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (index >= std::size_t(tokens)*2560) return;
    const std::size_t token = index/2560, row = index%2560;
    float sum = 0.0F;
#pragma unroll
    for (int path=0; path<10; ++path)
        sum = __fmaf_rn(alpha[token*10+path], routes[(token*10+path)*2560+row], sum);
    output[token*pitch+row] = sum;
}

} // namespace

void expert_route_combine(const float* routes, const float* alpha, float* output,
                          std::int32_t tokens, std::size_t output_pitch,
                          cudaStream_t stream) {
    if (!routes || !alpha || !output || tokens <= 0 || output_pitch < 2560)
        throw std::invalid_argument("expert route combine: invalid destination or geometry");
    constexpr unsigned threads = 256;
    const auto blocks = static_cast<unsigned>((std::size_t(tokens)*2560+threads-1)/threads);
    combine_kernel<<<blocks,threads,0,stream>>>(routes,alpha,output,tokens,output_pitch);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops
