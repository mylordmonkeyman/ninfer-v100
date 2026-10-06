#pragma once

#include "core/arena.h"
#include "targets/qwen3_8_flash_next/impl/model_view.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string_view>

namespace ninfer::targets::qwen3_8_flash_next::detail {
class FlashNextExpertCache;
class FlashNextExpertStream;


// Phase-11 vertical-slice proof that the correctness-first host path actually ran.
// Counters are process-local diagnostics; they do not participate in scheduling.
struct FlashNextHostExpertExecutionStats {
    std::uint64_t completed_layer_calls = 0;
    std::uint64_t fp32_output_layer_calls = 0;
    std::uint64_t routed_tokens = 0;
    std::uint64_t expert_pairs = 0;
};

void reset_flash_next_host_expert_execution_stats() noexcept;
[[nodiscard]] FlashNextHostExpertExecutionStats
flash_next_host_expert_execution_stats() noexcept;

[[nodiscard]] std::size_t flash_next_moe_workspace_capacity_bytes(std::int32_t min_tokens,
                                                                  std::int32_t max_tokens);

using MoeStageEmitter = std::function<void(std::string_view, const Tensor&)>;

// Exact Qwen4-exp 512-expert/top-10 MoE leaf with top-10 renormalized probabilities
// (norm_topk_prob=true per transformers Qwen4ExpTextTopKRouter); the independent shared expert is sigmoid-gated.
void flash_next_moe(const Tensor& input, const MoeWeights& weights, Tensor& output,
                    WorkspaceArena& workspace, cudaStream_t stream);

// Host-backed main-text MoE: resident routing/shared expert, optional Ready GPU cache hits,
// AVX2 CPU misses, and ordered FP32 merge. Opt-in route readiness separates
// CPU scheduling from shared-expert completion; serial synchronization remains the control.
void flash_next_moe_host_backed(const Tensor& input, const MoeWeights& resident_weights,
                                const HostNvfp4ExpertLayerView& host_experts, Tensor& output,
                                WorkspaceArena& workspace, cudaStream_t stream,
                                const MoeStageEmitter& emit = {},
                                const Tensor* router_input_fp32 = nullptr,
                                const Tensor* routed_expert_input_fp32 = nullptr,
                                const Tensor* shared_expert_input_fp32 = nullptr,
                                Tensor* output_fp32 = nullptr,
                                FlashNextExpertCache* cache = nullptr, unsigned layer = 0,
                                bool prefill = false, FlashNextExpertStream* expert_stream = nullptr);

void flash_next_moe_bf16(const Tensor& input, const MoeBf16Weights& weights, Tensor& output,
                         WorkspaceArena& workspace, cudaStream_t stream);

} // namespace ninfer::targets::qwen3_8_flash_next::detail
