#pragma once

#include "targets/qwen3_8_flash_next/impl/expert_bank.h"

#include <array>
#include <cstdint>
#include <span>

namespace ninfer::targets::qwen3_8_flash_next::detail {

inline constexpr std::size_t kFlashNextExpertHidden = 2'560;
inline constexpr std::size_t kFlashNextExpertIntermediate = 640;

struct CpuNvfp4ExpertReferenceScratch {
    std::array<float, kFlashNextExpertHidden> input{};
    std::array<float, kFlashNextExpertIntermediate> intermediate{};
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

// Production-like AVX2/FMA implementation of the same compact-NVFP4 contract.
// It preserves the BF16_RNE activation boundary and returns the unweighted FP32
// down vector. Callers must retain deterministic routing-alpha accumulation order.
[[nodiscard]] bool flash_next_cpu_nvfp4_avx2_available() noexcept;
void flash_next_cpu_nvfp4_expert_pair_avx2(
    const HostNvfp4ExpertPairView& expert,
    std::span<const std::uint16_t> input_bf16,
    std::span<float> output,
    CpuNvfp4ExpertReferenceScratch& scratch);

} // namespace ninfer::targets::qwen3_8_flash_next::detail
