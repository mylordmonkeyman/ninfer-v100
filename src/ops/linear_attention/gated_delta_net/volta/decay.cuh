#pragma once

#if !defined(NINFER_VOLTA_BUILD)
#error "GDN Volta backend is SM70-only"
#endif

#include "ops/linear_attention/gated_delta_net/volta/common.cuh"

#include <cuda_runtime.h>

#include <cmath>

namespace ninfer::ops::detail::gated_delta_net::volta {

struct DecayValues {
    float alpha[kChunkSize];
    float prefix[kChunkSize];
    float suffix[kChunkSize];
};

__host__ __device__ inline void compute_decay_values(const float* g, DecayValues& out) noexcept {
    float running = 1.0F;
    for (int t = 0; t < kChunkSize; ++t) {
        const float alpha = expf(g[t]);
        out.alpha[t] = alpha;
        running *= alpha;
        out.prefix[t] = running;
    }

    running = 1.0F;
    for (int t = kChunkSize - 1; t >= 0; --t) {
        out.suffix[t] = running;
        running *= out.alpha[t];
    }
}

__host__ __device__ inline float pairwise_decay(const DecayValues& decay, int row,
                                                 int col) noexcept {
    if (row <= col) { return 1.0F; }
    float value = 1.0F;
    for (int u = col + 1; u <= row; ++u) { value *= decay.alpha[u]; }
    return value;
}

__host__ __device__ inline void materialize_pairwise_decay(const DecayValues& decay,
                                                            float* matrix,
                                                            int leading_dimension) noexcept {
    for (int row = 0; row < kChunkSize; ++row) {
        for (int col = 0; col < kChunkSize; ++col) {
            matrix[row * leading_dimension + col] = 0.0F;
        }
        matrix[row * leading_dimension + row] = 1.0F;
        float running = 1.0F;
        for (int col = row - 1; col >= 0; --col) {
            running *= decay.alpha[col + 1];
            matrix[row * leading_dimension + col] = running;
        }
    }
}

} // namespace ninfer::ops::detail::gated_delta_net::volta
