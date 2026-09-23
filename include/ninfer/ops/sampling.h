#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

// Counter-based RNG subkey. Distinct purposes keep draws at the same logical position separate.
enum SamplePurpose : std::int32_t {
    kSamplePurposePrefill               = 0,
    kSamplePurposeDecode                = 1,
    kSamplePurposeSpeculativeAccept     = 2,
    kSamplePurposeSpeculativeCorrection = 3,
    kSamplePurposeSpeculativeBonus      = 4,
    kSamplePurposeDFlash2Proposal       = 5,
};

// Device-resident sampling parameters. token_counts is an optional device I32
// [token_domain] generated-token occurrence-count array used by all penalties.
struct SamplingConfig {
    float temperature          = 0.0f; // <= 0 => greedy argmax over allowed, adjusted logits
    std::int32_t top_k         = 20;   // runtime contract is [1,20]; Op defensively caps otherwise
    float top_p                = 1.0f; // >= 1 => disabled
    float min_p                = 0.0f; // <= 0 => disabled
    float presence_penalty     = 0.0f;
    float frequency_penalty    = 0.0f;
    float repetition_penalty   = 1.0f;
    unsigned long long seed    = 0;
    std::int32_t* token_counts = nullptr; // device [token_domain] i32, or null
    const std::int32_t* allowed_tokens = nullptr; // device bitset [ceil(token_domain/32)], or null
    const std::int32_t* prompt_presence = nullptr; // immutable prompt-membership bitset
    const std::int32_t* history_overlay = nullptr; // read-only provisional prefix, or null
    std::int32_t history_overlay_size = 0;
    bool commit_token_counts = true; // false: caller commits only the licensed output
};

// Caller-owned transient capacity for every parallel sampling-lane count in the inclusive
// interval. For sample(), one lane is one batch row; speculative acceptance uses the same
// workspace substrate for its verification columns. token_domain is the fixed route profile.
// Invalid profiles or intervals throw; a legal single-block route returns zero.
[[nodiscard]] std::size_t sampling_workspace_capacity_bytes(std::int32_t token_domain,
                                                            std::int32_t min_lanes,
                                                            std::int32_t max_lanes);

/**
 * Produces one token id per independent request row. `logits` is contiguous BF16
 * [physical_rows,B], `out` and `logical_positions` are contiguous I32 [B], and only vocabulary
 * rows v in [0,token_domain) participate. `configs` is a device-resident contiguous
 * SamplingConfig[B] array. Greedy and stochastic rows may coexist in one invocation.
 *
 * With either greedy or positive-temperature sampling, let
 * logit_v=float(logits[v,b]) and c_v=configs[b].token_counts[v] (or zero when null)
 * plus the number of occurrences of v in history_overlay[0:history_overlay_size]:
 *
 *   prompt_v = prompt_presence != null && ((uint32(prompt_presence[v/32]) >> (v%32)) & 1).
 *   repeated_v = prompt_v || c_v > 0.
 *   penalized_v = repeated_v ? (logit_v < 0 ? logit_v * repetition_penalty
 *                                          : logit_v / repetition_penalty) : logit_v.
 *   adjusted_v = penalized_v
 *                - configs[b].presence_penalty * (c_v > 0)
 *                - configs[b].frequency_penalty * c_v.
 * If allowed_tokens is non-null, a zero bit v makes adjusted_v negative infinity before
 * ranking and filtering. Bit v is (uint32_t(allowed_tokens[v/32]) >> (v%32)) & 1.
 * The device mask has ceil(token_domain/32) words and is read-only. Each row must permit
 * at least one finite-logit vocabulary token. Padding bits beyond token_domain are ignored.
 *
 * A greedy row selects min argmax_v adjusted_v and skips filters and RNG. Candidates for a
 * positive-temperature row are sorted by adjusted_v descending with lower token id breaking
 * ties. Per-row top_k
 * in [1,19] keeps that many candidates; top_k<=0 or top_k>=20 keeps min(20,token_domain).
 * Candidate weights are exp(adjusted_v/temperature-max). min_p removes the suffix below
 * min_p*max_weight; top_p keeps the shortest remaining prefix whose cumulative weight reaches
 * top_p times the pre-truncation candidate weight. At least the best candidate remains, the
 * support is renormalized, and one id is drawn for that row.
 *
 * Row b uses counter-based RNG key
 * (configs[b].seed,logical_positions[b],purpose), without mutable RNG state or dependence on the
 * compact row index. In every mode the selected token atomically increments
 * configs[b].token_counts when it is non-null and commit_token_counts is true.
 * The overlay contributes to generated counts for all three penalties. Prompt membership affects
 * only repetition penalty. A repetition penalty must be finite and positive; 1 is neutral.
 * Read-only count arrays may alias across verification columns; a mutable array must belong to
 * exactly one row. Overlay and prompt bitsets are read-only and must remain live through execution.
 * `out` must not overlap logits, configs, logical_positions, or any
 * token-count array. The Op writes all of out, uses caller-owned transient storage reported by
 * sampling_workspace_capacity_bytes(), and has no other persistent-state side effect.
 */
void sample(const Tensor& logits, Tensor& out, std::int32_t token_domain,
            const SamplingConfig* configs, const Tensor& logical_positions, std::int32_t purpose,
            WorkspaceArena& workspace, cudaStream_t stream);

// Adds every id in the contiguous non-empty I32 token_ids vector to the contiguous I32
// [token_domain] committed count array. IDs must be in [0,token_domain).
void increment_token_counts(const Tensor& token_ids, Tensor& token_counts, cudaStream_t stream);

} // namespace ninfer::ops
