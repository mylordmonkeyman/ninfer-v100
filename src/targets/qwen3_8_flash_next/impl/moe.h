#pragma once

#include "core/arena.h"
#include "targets/qwen3_8_flash_next/impl/model_view.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::targets::qwen3_8_flash_next::detail {

[[nodiscard]] std::size_t flash_next_moe_workspace_capacity_bytes(std::int32_t min_tokens,
                                                                  std::int32_t max_tokens);

void flash_next_moe(const Tensor& input, const MoeWeights& weights, Tensor& output,
                    WorkspaceArena& workspace, cudaStream_t stream,
                    void* expert_staging = nullptr, std::size_t expert_staging_bytes = 0);

void flash_next_moe_bf16(const Tensor& input, const MoeBf16Weights& weights, Tensor& output,
                         WorkspaceArena& workspace, cudaStream_t stream);

} // namespace ninfer::targets::qwen3_8_flash_next::detail
