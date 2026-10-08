#pragma once
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>

namespace ninfer::targets::qwen3_8_flash_next::detail {

// Deterministic ordering of nonresident expert groups. The default preserves
// expert-ID order. Busy-first can overlap a long GPU expert kernel with the
// subsequent expert H2D transfers, without changing which route owns an output.
struct FlashNextStreamExpertOrder {
    std::array<std::uint16_t, 512> ids{};
    unsigned size = 0;
};

inline FlashNextStreamExpertOrder flash_next_stream_expert_order(
    std::span<const std::size_t> route_counts, bool busy_first) {
    if (route_counts.size() != 512)
        throw std::invalid_argument("expert stream order requires 512 expert counts");
    FlashNextStreamExpertOrder result;
    for (unsigned expert = 0; expert < 512; ++expert)
        if (route_counts[expert])
            result.ids[result.size++] = static_cast<std::uint16_t>(expert);
    if (busy_first) {
        std::sort(result.ids.begin(), result.ids.begin() + result.size,
                  [&](std::uint16_t lhs, std::uint16_t rhs) {
            if (route_counts[lhs] != route_counts[rhs])
                return route_counts[lhs] > route_counts[rhs];
            return lhs < rhs;
        });
    }
    return result;
}
} // namespace ninfer::targets::qwen3_8_flash_next::detail
