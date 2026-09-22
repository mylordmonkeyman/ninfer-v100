#pragma once

#if !defined(NINFER_VOLTA_BUILD)
#error "GDN Volta backend is SM70-only"
#endif

#include "ops/linear_attention/gated_delta_net/volta/common.cuh"

#include <cuda_runtime.h>

namespace ninfer::ops::detail::gated_delta_net::volta {

__host__ __device__ inline void neumann2_inverse_bt32(const float* b, int b_leading_dimension,
                                                      float* inverse,
                                                      int inverse_leading_dimension) noexcept {
    for (int row = 0; row < kChunkSize; ++row) {
        for (int col = 0; col < kChunkSize; ++col) {
            inverse[row * inverse_leading_dimension + col] = 0.0F;
        }
        inverse[row * inverse_leading_dimension + row] = 1.0F;

        for (int col = 0; col < row; ++col) {
            float b2 = 0.0F;
            for (int k = col + 1; k < row; ++k) {
                b2 += b[row * b_leading_dimension + k] *
                      b[k * b_leading_dimension + col];
            }
            inverse[row * inverse_leading_dimension + col] =
                -b[row * b_leading_dimension + col] + b2;
        }
    }
}

} // namespace ninfer::ops::detail::gated_delta_net::volta
