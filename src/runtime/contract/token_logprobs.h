#pragma once

#include "ninfer/types.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace ninfer::runtime {

// Host-side readout of one sampled column. The logits are the target model's BF16 output before
// any sampling adjustment: no penalty, no temperature, no truncation. `allowed` is the structured
// output bitmask of that position (bit set = token allowed), or empty when unconstrained.
//
// Two normalisations are reported for every token:
//   raw_logprob  log softmax over the whole valid vocabulary
//   logprob      log softmax over the allowed tokens only; equal to raw_logprob when unconstrained,
//                and -inf for a token the mask forbids
// so that under a mask `logprob` is the distribution the sampler could actually draw from.

[[nodiscard]] inline float bf16_to_float(std::uint16_t bits) noexcept {
    // BF16 is the upper half of an IEEE-754 binary32 with the same exponent width.
    return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16U);
}

[[nodiscard]] inline bool token_allowed(std::span<const std::int32_t> allowed,
                                        std::size_t token) noexcept {
    if (allowed.empty()) { return true; }
    const std::size_t word = token / 32U;
    if (word >= allowed.size()) { return false; }
    return ((static_cast<std::uint32_t>(allowed[word]) >> (token % 32U)) & 1U) != 0U;
}

[[nodiscard]] inline TokenLogprobs compute_token_logprobs(std::span<const std::uint16_t> logits,
                                                          std::span<const std::int32_t> allowed,
                                                          TokenId sampled,
                                                          const TokenLogprobOptions& options) {
    if (logits.empty() || sampled < 0 || static_cast<std::size_t>(sampled) >= logits.size()) {
        throw std::logic_error("token logprobs: sampled token is outside the logit column");
    }
    constexpr float kNegativeInfinity = -std::numeric_limits<float>::infinity();

    float raw_maximum     = kNegativeInfinity;
    float allowed_maximum = kNegativeInfinity;
    for (std::size_t token = 0; token < logits.size(); ++token) {
        const float value = bf16_to_float(logits[token]);
        raw_maximum       = std::max(raw_maximum, value);
        if (token_allowed(allowed, token)) { allowed_maximum = std::max(allowed_maximum, value); }
    }
    if (!std::isfinite(raw_maximum) || !std::isfinite(allowed_maximum)) {
        throw std::runtime_error("token logprobs: logit column has no finite allowed entry");
    }

    // Partial selection of the `top` best allowed tokens; ties break toward the lower token id so
    // the order is deterministic.
    struct Entry {
        float value;
        TokenId token;
    };
    const auto better = [](const Entry& a, const Entry& b) noexcept {
        return a.value > b.value || (a.value == b.value && a.token < b.token);
    };
    std::vector<Entry> best;
    best.reserve(static_cast<std::size_t>(options.top) + 1U);

    double raw_sum     = 0.0;
    double allowed_sum = 0.0;
    for (std::size_t token = 0; token < logits.size(); ++token) {
        const float value = bf16_to_float(logits[token]);
        raw_sum += std::exp(static_cast<double>(value) - raw_maximum);
        if (!token_allowed(allowed, token)) { continue; }
        allowed_sum += std::exp(static_cast<double>(value) - allowed_maximum);
        if (options.top == 0) { continue; }
        const Entry entry{value, static_cast<TokenId>(token)};
        if (best.size() == options.top && !better(entry, best.back())) { continue; }
        best.insert(std::upper_bound(best.begin(), best.end(), entry, better), entry);
        if (best.size() > options.top) { best.pop_back(); }
    }
    const double raw_normaliser     = static_cast<double>(raw_maximum) + std::log(raw_sum);
    const double allowed_normaliser = static_cast<double>(allowed_maximum) + std::log(allowed_sum);

    const auto describe = [&](TokenId token) {
        const float value = bf16_to_float(logits[static_cast<std::size_t>(token)]);
        return TokenLogprob{
            .token       = token,
            .logprob     = token_allowed(allowed, static_cast<std::size_t>(token))
                                   ? static_cast<float>(value - allowed_normaliser)
                                   : kNegativeInfinity,
            .raw_logprob = static_cast<float>(value - raw_normaliser),
        };
    };

    TokenLogprobs out;
    out.sampled = describe(sampled);
    out.top.reserve(best.size());
    for (const Entry& entry : best) { out.top.push_back(describe(entry.token)); }

    // Explicit candidates: the distribution renormalised over exactly this list, in list order.
    // A candidate the mask forbids keeps its slot with probability zero.
    if (!options.candidates.empty()) {
        float candidate_maximum = kNegativeInfinity;
        for (const TokenId token : options.candidates) {
            if (token < 0 || static_cast<std::size_t>(token) >= logits.size()) {
                throw std::logic_error("token logprobs: candidate is outside the logit column");
            }
            if (!token_allowed(allowed, static_cast<std::size_t>(token))) { continue; }
            candidate_maximum =
                std::max(candidate_maximum, bf16_to_float(logits[static_cast<std::size_t>(token)]));
        }
        double candidate_sum = 0.0;
        if (std::isfinite(candidate_maximum)) {
            for (const TokenId token : options.candidates) {
                if (!token_allowed(allowed, static_cast<std::size_t>(token))) { continue; }
                candidate_sum +=
                    std::exp(static_cast<double>(
                                 bf16_to_float(logits[static_cast<std::size_t>(token)])) -
                             candidate_maximum);
            }
        }
        const double candidate_normaliser =
            static_cast<double>(candidate_maximum) + std::log(candidate_sum);
        out.candidates.reserve(options.candidates.size());
        for (const TokenId token : options.candidates) {
            TokenLogprob entry = describe(token);
            if (std::isfinite(entry.logprob)) {
                entry.logprob = static_cast<float>(
                    bf16_to_float(logits[static_cast<std::size_t>(token)]) - candidate_normaliser);
            }
            out.candidates.push_back(entry);
        }
    }
    return out;
}

} // namespace ninfer::runtime
