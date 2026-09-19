#pragma once

#include "ninfer/types.h"

#include <bit>
#include <span>
#include <vector>

namespace ninfer::runtime {

// Build from the complete prepared prompt, including reused tokens. Media-only
// placeholders outside the semantic token domain do not identify sampled tokens.
inline std::vector<std::int32_t> prompt_token_presence(std::span<const TokenId> tokens,
                                                      std::int32_t token_domain) {
    std::vector<std::int32_t> bits((token_domain + 31) / 32, 0);
    for (const TokenId token : tokens) {
        if (token >= 0 && token < token_domain) {
            auto word = std::bit_cast<std::uint32_t>(bits[token / 32]);
            bits[token / 32] = std::bit_cast<std::int32_t>(word | (1U << (token % 32)));
        }
    }
    return bits;
}

} // namespace ninfer::runtime
