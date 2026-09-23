#pragma once

#include "ninfer/types.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::product {

[[nodiscard]] inline SpeculativeBackend parse_speculative_backend(std::string_view value) {
    if (value == "mtp") { return SpeculativeBackend::Mtp; }
    if (value == "dflash") { return SpeculativeBackend::DFlash; }
    if (value == "dflash2") { return SpeculativeBackend::DFlash2; }
    throw std::invalid_argument("invalid speculative backend: " + std::string(value));
}

[[nodiscard]] inline const char* speculative_backend_name(SpeculativeBackend backend) noexcept {
    switch (backend) {
    case SpeculativeBackend::None:
        return "none";
    case SpeculativeBackend::Mtp:
        return "mtp";
    case SpeculativeBackend::DFlash:
        return "dflash";
    case SpeculativeBackend::DFlash2:
        return "dflash2";
    }
    return "unknown";
}

inline void validate_speculative_cli_options(const SpeculativeOptions& options,
                                             std::uint32_t max_concurrency = 1) {
    switch (options.backend) {
    case SpeculativeBackend::None:
        if (options.draft_tokens != 0 || options.proposal_head != ProposalHead::Full) {
            throw std::invalid_argument(
                "--draft-tokens and --lm-head-draft require --spec mtp|dflash|dflash2");
        }
        return;
    case SpeculativeBackend::Mtp: {
        const std::uint32_t conc = std::max(1u, max_concurrency);
        // State-slot budget ceiling: (draft_tokens + 1) * conc <= 64 slots.
        const std::uint32_t ceiling = (conc <= 64u) ? (64u / conc) - 1u : 0u;
        // The artifact's runtime applies its exact limit (five for dense, four for Flash-Next).
        const std::uint32_t max_mtp = std::min(5u, ceiling);
        if (options.draft_tokens == 0 || options.draft_tokens > max_mtp) {
            throw std::invalid_argument(
                "--spec mtp requires --draft-tokens in [1, " + std::to_string(max_mtp) +
                "] at max-concurrency " + std::to_string(max_concurrency));
        }
        return;
    }
    case SpeculativeBackend::DFlash:
        if (options.draft_tokens == 0 || options.draft_tokens > 15) {
            throw std::invalid_argument("--spec dflash requires --draft-tokens in [1,15]");
        }
        return;
    case SpeculativeBackend::DFlash2:
        if (options.draft_tokens == 0 || options.draft_tokens > 15) {
            throw std::invalid_argument("--spec dflash2 requires --draft-tokens in [1,15]");
        }
        return;
    }
    throw std::invalid_argument("invalid speculative backend");
}

} // namespace ninfer::product
