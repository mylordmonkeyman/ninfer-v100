#pragma once
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string_view>

namespace ninfer::targets::qwen3_8_flash_next::detail {

// Explicit opt-in only. Streaming prefill normally disables background cache
// admissions to avoid competing H2D uploads.
inline bool flash_next_stream_prefill_admit_hot(const char* policy) {
    if (!policy || !*policy || std::string_view(policy) == "off") return false;
    if (std::string_view(policy) == "hot") return true;
    throw std::invalid_argument(
        "NINFER_V100_PREFILL_STREAM_ADMIT must be off or hot");
}

// Select the highest-traffic nonresident expert in a single prefill layer.
// The stable low-ID tie-break makes the result independent of worker timing.
// An empty/nonresident-free layer must not trigger an admission.
inline int flash_next_stream_prefill_admit_candidate(
    std::span<const std::size_t> missing_routes_per_expert) {
    if (missing_routes_per_expert.size() != 512)
        throw std::invalid_argument("prefill admission requires 512 expert counts");
    std::size_t max_routes = 0;
    int selected = -1;
    for (std::size_t id = 0; id < missing_routes_per_expert.size(); ++id) {
        if (missing_routes_per_expert[id] > max_routes) {
            max_routes = missing_routes_per_expert[id];
            selected = static_cast<int>(id);
        }
    }
    return selected;
}
} // namespace ninfer::targets::qwen3_8_flash_next::detail
