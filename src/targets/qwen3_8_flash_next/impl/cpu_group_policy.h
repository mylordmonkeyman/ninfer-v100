#pragma once
#include <stdexcept>
#include <string_view>

namespace ninfer::targets::qwen3_8_flash_next::detail {

inline bool flash_next_cpu_expert_grouping_enabled(const char* raw, bool prefill) {
    if (raw == nullptr || !*raw || std::string_view(raw) == "0") return false;
    if (std::string_view(raw) == "1") return true;
    if (std::string_view(raw) == "prefill") return prefill;
    throw std::invalid_argument(
        "NINFER_V100_CPU_EXPERT_GROUP must be 0, 1 or prefill");
}

} // namespace ninfer::targets::qwen3_8_flash_next::detail
