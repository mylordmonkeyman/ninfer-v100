#pragma once

#include "core/arena.h"
#include "targets/qwen3_8_flash_next/impl/hyper_workspace.h"
#include "targets/qwen3_8_flash_next/impl/model_view.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::targets::qwen3_8_flash_next::detail {

[[nodiscard]] std::size_t flash_next_hyper_workspace_capacity_bytes(std::int32_t min_tokens,
                                                                    std::int32_t max_tokens);

// Normalizes the four streams, evaluates the low-rank input mixer, and retains the four
// block-output injection gates in scratch for the matching inject call.
void flash_next_hyper_prepare(const Tensor& hidden, const HyperConnectionWeights& weights,
                              FlashNextHyperWorkspace& scratch, Tensor& block_input,
                              cudaStream_t stream);
#if defined(NINFER_VOLTA_BUILD)
// Decode-only diagnostic: read the FP32 hyper master through group RMSNorm, then
// intentionally rejoin production at the BF16 normalized boundary.
void flash_next_hyper_prepare_fp32_hidden_stage(
    const Tensor& hidden_fp32, const HyperConnectionWeights& weights,
    FlashNextHyperWorkspace& scratch, Tensor& block_input, cudaStream_t stream);
#endif

void flash_next_hyper_inject(const Tensor& block_output, const Tensor& injection, Tensor& hidden,
                             cudaStream_t stream);

// Final/MTP mixer form: identical normalized low-rank mixer without an injection projection.
void flash_next_hyper_mix(const Tensor& hidden, const HyperMixerWeights& weights,
                          FlashNextHyperWorkspace& scratch, Tensor& block_input,
                          cudaStream_t stream);

} // namespace ninfer::targets::qwen3_8_flash_next::detail
