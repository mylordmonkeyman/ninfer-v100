#pragma once
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace ninfer::targets::qwen3_8_flash_next::detail {

// Shared bounded policy parser for decode and prefill, preserving 1.0 as
// the default (stream every selected nonresident expert on the GPU).
inline double flash_next_parse_expert_stream_fraction(
    const char* raw, const char* environment_name) {
    if (raw == nullptr || !*raw) return 1.0;
    char* end = nullptr;
    const double value = std::strtod(raw, &end);
    if (end == raw || *end != '\0' || !std::isfinite(value) ||
        value <= 0.0 || value > 1.0) {
        throw std::invalid_argument(
            std::string(environment_name) + " must be in (0, 1]");
    }
    return value;
}
// An opt-in alternative to streaming a fixed fraction of nonresident experts.
// Zero (including unset) preserves the original fraction/decode policies.
// Strict ASCII decimal avoids silently accepting signs, whitespace, or suffixes.
inline unsigned flash_next_parse_prefill_stream_min_routes(const char* raw) {
    constexpr const char* name = "NINFER_V100_PREFILL_EXPERT_STREAM_MIN_ROUTES";
    if (!raw || !*raw) return 0;
    unsigned value = 0;
    for (const char* p = raw; *p; ++p) {
        if (*p < '0' || *p > '9')
            throw std::invalid_argument(std::string(name) + " must be an integer in [0, 4096]");
        const unsigned digit = static_cast<unsigned>(*p - '0');
        if (value > (4096U - digit) / 10U)
            throw std::invalid_argument(std::string(name) + " must be an integer in [0, 4096]");
        value = value * 10U + digit;
    }
    return value;
}

} // namespace ninfer::targets::qwen3_8_flash_next::detail
