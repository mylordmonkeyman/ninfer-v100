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
} // namespace ninfer::targets::qwen3_8_flash_next::detail
