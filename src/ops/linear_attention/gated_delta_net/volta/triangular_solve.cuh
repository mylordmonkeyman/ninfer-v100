#pragma once

#if !defined(NINFER_VOLTA_BUILD)
#error "GDN Volta backend is SM70-only"
#endif

#include "ops/linear_attention/gated_delta_net/volta/common.cuh"

#include <cuda_runtime.h>

namespace ninfer::ops::detail::gated_delta_net::volta {

inline constexpr int kSolveBlock = 16;
inline constexpr int kSolveScratchElements = kSolveBlock * kSolveBlock;

__host__ __device__ inline void clear_bt32_matrix(float* matrix, int leading_dimension) noexcept {
    for (int row = 0; row < kChunkSize; ++row) {
        for (int col = 0; col < kChunkSize; ++col) {
            matrix[row * leading_dimension + col] = 0.0F;
        }
    }
}

__host__ __device__ inline void invert_unit_lower_16(const float* b, int b_leading_dimension,
                                                      int offset, float* inverse,
                                                      int inverse_leading_dimension) noexcept {
    for (int row = 0; row < kSolveBlock; ++row) {
        inverse[(offset + row) * inverse_leading_dimension + offset + row] = 1.0F;
        for (int col = 0; col < row; ++col) {
            float sum = 0.0F;
            for (int k = col; k < row; ++k) {
                sum += b[(offset + row) * b_leading_dimension + offset + k] *
                       inverse[(offset + k) * inverse_leading_dimension + offset + col];
            }
            inverse[(offset + row) * inverse_leading_dimension + offset + col] = -sum;
        }
    }
}

__host__ __device__ inline void exact_inverse_bt32(const float* b, int b_leading_dimension,
                                                   float* inverse,
                                                   int inverse_leading_dimension,
                                                   float* scratch) noexcept {
    clear_bt32_matrix(inverse, inverse_leading_dimension);
    invert_unit_lower_16(b, b_leading_dimension, 0, inverse, inverse_leading_dimension);
    invert_unit_lower_16(b, b_leading_dimension, kSolveBlock, inverse,
                         inverse_leading_dimension);

    for (int row = 0; row < kSolveBlock; ++row) {
        for (int col = 0; col < kSolveBlock; ++col) {
            float sum = 0.0F;
            for (int k = 0; k < kSolveBlock; ++k) {
                sum += inverse[(kSolveBlock + row) * inverse_leading_dimension +
                               kSolveBlock + k] *
                       b[(kSolveBlock + k) * b_leading_dimension + col];
            }
            scratch[row * kSolveBlock + col] = sum;
        }
    }

    for (int row = 0; row < kSolveBlock; ++row) {
        for (int col = 0; col < kSolveBlock; ++col) {
            float sum = 0.0F;
            for (int k = 0; k < kSolveBlock; ++k) {
                sum += scratch[row * kSolveBlock + k] *
                       inverse[k * inverse_leading_dimension + col];
            }
            inverse[(kSolveBlock + row) * inverse_leading_dimension + col] = -sum;
        }
    }
}

} // namespace ninfer::ops::detail::gated_delta_net::volta
