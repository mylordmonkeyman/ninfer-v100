#pragma once

#include "targets/qwen3_8_flash_next/impl/expert_bank.h"

#include <array>
#include <cstdint>
#include <span>

namespace ninfer::targets::qwen3_8_flash_next::detail {

inline constexpr std::size_t kFlashNextExpertHidden = 2'560;
inline constexpr std::size_t kFlashNextExpertIntermediate = 640;
inline constexpr std::size_t kFlashNextCpuExpertGroupMax = 4;

struct CpuNvfp4ExpertReferenceScratch {
    std::array<float, kFlashNextExpertHidden> input{};
    std::array<float, kFlashNextExpertIntermediate> intermediate{};
};

struct CpuNvfp4ExpertGroupScratch {
    std::array<CpuNvfp4ExpertReferenceScratch, kFlashNextCpuExpertGroupMax> tokens{};
};

// Correctness-first Phase-10 CPU path. Persistent weights remain in the canonical
// compact host NVFP4 mapping. The SiLU(gate)*up boundary is rounded to BF16 before
// the down projection to match the production MoE activation contract.
//
// The result is an unweighted routed-expert down vector in FP32. Routing alpha is
// intentionally applied by the later GPU/CPU merge layer, not inside this reference.
void flash_next_cpu_nvfp4_expert_pair_reference(
    const HostNvfp4ExpertPairView& expert,
    std::span<const std::uint16_t> input_bf16,
    std::span<float> output,
    CpuNvfp4ExpertReferenceScratch& scratch);

// Diagnostic only: matches the independent CPU-FP32 oracle's expert arithmetic by
// retaining SiLU(gate)*up in FP32 before the down projection. Production NInfer
// intentionally uses the BF16-rounded reference above.
void flash_next_cpu_nvfp4_expert_pair_reference_fp32_intermediate(
    const HostNvfp4ExpertPairView& expert,
    std::span<const std::uint16_t> input_bf16,
    std::span<float> output,
    CpuNvfp4ExpertReferenceScratch& scratch);

// Diagnostic only: preserve the unrounded FP32 MoE activation at the routed-expert
// input boundary while retaining the production BF16 SiLU(gate)*up boundary.
void flash_next_cpu_nvfp4_expert_pair_reference_fp32_input(
    const HostNvfp4ExpertPairView& expert,
    std::span<const float> input_fp32,
    std::span<float> output,
    CpuNvfp4ExpertReferenceScratch& scratch);

// Production AVX2/FMA implementation of the same compact-NVFP4 contract.
// It preserves the BF16_RNE activation boundary and returns the unweighted FP32
// down vector. Callers must retain deterministic routing-alpha accumulation order.
[[nodiscard]] bool flash_next_cpu_nvfp4_avx2_available() noexcept;
void flash_next_cpu_nvfp4_expert_pair_avx2(
    const HostNvfp4ExpertPairView& expert,
    std::span<const std::uint16_t> input_bf16,
    std::span<float> output,
    CpuNvfp4ExpertReferenceScratch& scratch);

// Row-sharded production entry points. Prepare completes before gate/up shards;
// all gate/up shards complete before down shards. Each row keeps the same FMA
// reduction and BF16 boundary as the complete-pair kernel.
void flash_next_cpu_nvfp4_expert_prepare_avx2(
    const HostNvfp4ExpertPairView& expert, std::span<const std::uint16_t> input,
    CpuNvfp4ExpertReferenceScratch& scratch);
void flash_next_cpu_nvfp4_expert_gate_up_rows_avx2(
    const HostNvfp4ExpertPairView& expert, CpuNvfp4ExpertReferenceScratch& scratch,
    std::size_t begin, std::size_t end);
void flash_next_cpu_nvfp4_expert_down_rows_avx2(
    const HostNvfp4ExpertPairView& expert, const CpuNvfp4ExpertReferenceScratch& scratch,
    std::span<float> output, std::size_t begin, std::size_t end);

// Grouped AVX2/FMA execution for 2--4 activations routed to the same expert.
// Packed weights and scales are decoded once per K16 block, while every token
// retains the single-token kernel's independent accumulator and reduction order.
void flash_next_cpu_nvfp4_expert_group_prepare_avx2(
    const HostNvfp4ExpertPairView& expert,
    std::span<const std::uint16_t* const> inputs,
    CpuNvfp4ExpertGroupScratch& scratch);
void flash_next_cpu_nvfp4_expert_group_gate_up_rows_avx2(
    const HostNvfp4ExpertPairView& expert, CpuNvfp4ExpertGroupScratch& scratch,
    std::size_t token_count, std::size_t begin, std::size_t end);
void flash_next_cpu_nvfp4_expert_group_down_rows_avx2(
    const HostNvfp4ExpertPairView& expert, const CpuNvfp4ExpertGroupScratch& scratch,
    std::span<float* const> outputs, std::size_t begin, std::size_t end);
void flash_next_cpu_nvfp4_expert_group_avx2(
    const HostNvfp4ExpertPairView& expert,
    std::span<const std::uint16_t* const> inputs,
    std::span<float* const> outputs,
    CpuNvfp4ExpertGroupScratch& scratch);

} // namespace ninfer::targets::qwen3_8_flash_next::detail
