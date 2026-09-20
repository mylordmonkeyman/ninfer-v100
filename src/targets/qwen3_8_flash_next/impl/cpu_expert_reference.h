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

} // namespace ninfer::targets::qwen3_8_flash_next::detail
