#include "ops/linear/bf16/bf16_launch.h"

#include "core/device.h"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kThreads = 256;

__global__ void bf16_volta_simt_kernel(const __nv_bfloat16* __restrict__ x,
                                       const __nv_bfloat16* __restrict__ weight,
                                       __nv_bfloat16* __restrict__ out,
                                       int n, int k, int t) {
    __shared__ float partial[kThreads / 32];
    const int row = static_cast<int>(blockIdx.x);
    const int token = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x);
    if (row >= n || token >= t) { return; }

    const auto* xrow = x + static_cast<std::int64_t>(token) * k;
    const auto* wrow = weight + static_cast<std::int64_t>(row) * k;
    float sum = 0.0F;
    for (int col = tid; col < k; col += kThreads) {
        sum = fmaf(__bfloat162float(xrow[col]), __bfloat162float(wrow[col]), sum);
    }

    for (int delta = 16; delta > 0; delta >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFFU, sum, delta);
    }
    const int lane = tid & 31;
    const int warp = tid >> 5;
    if (lane == 0) { partial[warp] = sum; }
    __syncthreads();

    if (warp == 0) {
        float total = lane < (kThreads / 32) ? partial[lane] : 0.0F;
        for (int delta = 16; delta > 0; delta >>= 1) {
            total += __shfl_down_sync(0xFFFFFFFFU, total, delta);
        }
        if (lane == 0) {
            out[static_cast<std::int64_t>(token) * n + row] = __float2bfloat16_rn(total);
        }
    }
}

} // namespace

void launch_bf16_volta_simt(const Tensor& x, const Weight& weight, Tensor& out,
                            cudaStream_t stream) {
#if defined(NINFER_VOLTA_BUILD)
    const dim3 grid(static_cast<unsigned>(weight.n), static_cast<unsigned>(x.ne[1]));
    bf16_volta_simt_kernel<<<grid, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const __nv_bfloat16*>(weight.qdata),
        static_cast<__nv_bfloat16*>(out.data), weight.n, weight.k, x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
#else
    (void)x;
    (void)weight;
    (void)out;
    (void)stream;
#endif
}

} // namespace ninfer::ops::detail
