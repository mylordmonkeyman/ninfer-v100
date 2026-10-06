#pragma once
#include <stdexcept>
#include <string_view>

namespace ninfer::targets::qwen3_8_flash_next::detail {
// Phase identity, rather than row count, keeps MTP verification/decode on
// the serial control while allowing an explicit prefill-only experiment.
inline bool route_handoff_enabled(std::string_view mode, bool prefill) {
    if (mode.empty() || mode == "0") return false;
    if (mode == "1") return true;
    if (mode == "prefill") return prefill;
    throw std::invalid_argument("NINFER_V100_ROUTE_HANDOFF must be 0, 1 or prefill");
}
} // namespace ninfer::targets::qwen3_8_flash_next::detail
