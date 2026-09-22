#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::targets::qwen3_8_flash_next::detail {

constexpr bool is_qsa_layer(std::size_t layer) noexcept {
    return layer >= 3 && ((layer - 3) % 4) == 0;
}

constexpr std::size_t qsa_ordinal(std::size_t layer) noexcept { return (layer - 3) / 4; }

constexpr std::size_t gdn_ordinal(std::size_t layer) noexcept {
    return layer - (layer >= 3 ? ((layer - 3) / 4 + 1) : 0);
}

void repeat_embedding_to_hyper_streams(const Tensor& embedding, Tensor& hyper_hidden,
                                       cudaStream_t stream);
#if defined(NINFER_VOLTA_BUILD)
void repeat_embedding_to_hyper_streams_fp32(const Tensor& embedding, Tensor& hyper_hidden,
                                             cudaStream_t stream);
void hyper_fp32_to_bf16(const Tensor& source, Tensor& destination, cudaStream_t stream);
void hyper_add_bf16_to_fp32(const Tensor& addend, Tensor& destination, cudaStream_t stream);
void hyper_inject_bf16_to_fp32(const Tensor& block_output, const Tensor& injection,
                               Tensor& hyper_hidden, cudaStream_t stream);
#endif

} // namespace ninfer::targets::qwen3_8_flash_next::detail
