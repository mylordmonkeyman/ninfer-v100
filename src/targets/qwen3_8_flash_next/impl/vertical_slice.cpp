#include "targets/qwen3_8_flash_next/impl/vertical_slice.h"

#include "core/device.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

namespace ninfer::targets::qwen3_8_flash_next::detail {
namespace {

struct TopK {
    std::array<std::int32_t, 10> ids{};
    std::array<float, 10> values{};
    std::size_t count = 0;
};

bool ranked_better(float left_value, std::int32_t left_id,
                   float right_value, std::int32_t right_id) {
    return left_value > right_value ||
           (left_value == right_value && left_id < right_id);
}

TopK top10(std::span<const float> logits) {
    TopK out{};
    out.values.fill(-std::numeric_limits<float>::infinity());
    out.ids.fill(std::numeric_limits<std::int32_t>::max());
    const std::size_t limit = std::min<std::size_t>(10, logits.size());
    for (std::size_t index = 0; index < logits.size(); ++index) {
        const float value = logits[index];
        const auto id = static_cast<std::int32_t>(index);
        std::size_t position = 0;
        while (position < out.count &&
               !ranked_better(value, id, out.values[position], out.ids[position])) {
            ++position;
        }
        if (position >= limit) { continue; }
        const std::size_t old_count = out.count;
        out.count = std::min(limit, out.count + 1);
        for (std::size_t move = out.count; move-- > position + 1;) {
            out.values[move] = out.values[move - 1];
            out.ids[move] = out.ids[move - 1];
        }
        out.values[position] = value;
        out.ids[position] = id;
        (void)old_count;
    }
    return out;
}

std::uint32_t overlap(const TopK& left, const TopK& right, std::size_t k) {
    const std::size_t left_count = std::min(k, left.count);
    const std::size_t right_count = std::min(k, right.count);
    std::uint32_t count = 0;
    for (std::size_t i = 0; i < left_count; ++i) {
        for (std::size_t j = 0; j < right_count; ++j) {
            if (left.ids[i] == right.ids[j]) {
                ++count;
                break;
            }
        }
    }
    return count;
}

double logsumexp(std::span<const float> logits) {
    const float maximum = *std::max_element(logits.begin(), logits.end());
    long double sum = 0.0L;
    for (const float value : logits) {
        sum += std::exp(static_cast<long double>(value - maximum));
    }
    return static_cast<double>(static_cast<long double>(maximum) + std::log(sum));
}

std::uint64_t consumed_bytes(const Phase11CudaMemorySnapshot& before,
                             const Phase11CudaMemorySnapshot& after) {
    return before.free_bytes >= after.free_bytes
               ? before.free_bytes - after.free_bytes
               : 0ULL;
}

std::int64_t signed_delta(std::uint64_t observed, std::uint64_t planned) {
    if (observed >= planned) {
        const std::uint64_t delta = observed - planned;
        return delta > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())
                   ? std::numeric_limits<std::int64_t>::max()
                   : static_cast<std::int64_t>(delta);
    }
    const std::uint64_t delta = planned - observed;
    return delta > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())
               ? std::numeric_limits<std::int64_t>::min()
               : -static_cast<std::int64_t>(delta);
}

} // namespace

Phase11VerticalSliceContract
make_phase11_vertical_slice_contract(std::uint32_t max_context,
                                     std::uint32_t prefill_chunk) {
    if (max_context < 4'096) {
        throw std::invalid_argument(
            "Phase 11 vertical slice requires max_context >= 4096");
    }
    if (prefill_chunk == 0 || (prefill_chunk % 128U) != 0U ||
        prefill_chunk > max_context) {
        throw std::invalid_argument(
            "Phase 11 prefill_chunk must be a nonzero multiple of 128 <= max_context");
    }

    Phase11VerticalSliceContract out{};
    out.runtime.max_concurrency = 1;
    out.runtime.max_context = max_context;
    out.runtime.state_slot_capacity = 2;
    out.runtime.continuation_capacity = 0;
    out.runtime.prefill_chunk = prefill_chunk;
    out.runtime.speculative_draft_tokens = 0;
    out.runtime.proposal_head = ProposalHead::Full;
    out.runtime.use_cuda_graph = false;
    out.runtime.vision_enabled = false;
    out.runtime.use_qsa_prefill_mma = false;
    out.runtime.kv_cache = KvCacheStorage::BFloat16;
    out.runtime.gdn_state_storage = GdnStateStorage::FP32;

    out.load.vision = false;
    out.load.mtp = false;
    out.load.proposal_head = ProposalHead::Full;
    out.load.quantize_output_head_fp8 = false;
    out.load.quantize_token_embedding_fp8 = false;
    out.load.host_backed_experts = true;
    out.expert_cache_enabled = false;
    out.minimum_teacher_forced_positions = 4'096;
    return out;
}

void validate_phase11_preflight(const FlashNextPreflightReport& report) {
    const auto& plan = report.runtime_plan;
    const auto& ledger = report.vram_ledger;
    if (plan.config.use_cuda_graph || plan.config.speculative_draft_tokens != 0 ||
        plan.config.max_concurrency != 1 || plan.config.vision_enabled) {
        throw std::logic_error(
            "Phase 11 preflight does not satisfy eager non-MTP single-lane contract");
    }
    if (plan.config.kv_cache != KvCacheStorage::BFloat16 ||
        plan.config.gdn_state_storage != GdnStateStorage::FP32) {
        throw std::logic_error(
            "Phase 11 preflight requires BF16 KV and FP32 GDN state");
    }
    if (ledger.cuda_graph_bytes != 0 || ledger.routed_expert_payload_bytes != 0 ||
        ledger.routed_expert_layers != 0) {
        throw std::logic_error(
            "Phase 11 preflight unexpectedly reserves graph or resident routed-expert VRAM");
    }
    if (ledger.host_backed_expert_layers != 48 ||
        ledger.host_backed_expert_payload_bytes == 0 ||
        ledger.host_backed_expert_layer_payload_bytes == 0) {
        throw std::logic_error(
            "Phase 11 preflight requires all 48 routed-expert layers host-backed");
    }
}

void Phase11OracleAccumulator::observe(std::uint32_t position,
                                      std::int32_t target_token,
                                      std::span<const float> candidate_logits,
                                      std::span<const float> oracle_logits) {
    if (candidate_logits.empty() || candidate_logits.size() != oracle_logits.size()) {
        throw std::invalid_argument(
            "Phase 11 oracle requires equal non-empty candidate/oracle logits");
    }

    ++positions_;
    bool finite = true;
    float local_max_error = 0.0F;
    for (std::size_t i = 0; i < candidate_logits.size(); ++i) {
        const float candidate = candidate_logits[i];
        const float oracle = oracle_logits[i];
        finite = finite && std::isfinite(candidate) && std::isfinite(oracle);
        if (std::isfinite(candidate) && std::isfinite(oracle)) {
            local_max_error =
                std::max(local_max_error, std::abs(candidate - oracle));
        }
    }
    maximum_logit_error_ = std::max(maximum_logit_error_, local_max_error);
    if (!finite) {
        ++nonfinite_positions_;
        return;
    }

    const TopK candidate_top = top10(candidate_logits);
    const TopK oracle_top = top10(oracle_logits);
    if (candidate_top.ids[0] == oracle_top.ids[0]) {
        ++top1_matches_;
    }

    const std::size_t top5_den = std::min<std::size_t>(5, oracle_top.count);
    const std::size_t top10_den = std::min<std::size_t>(10, oracle_top.count);
    top5_overlap_sum_ += overlap(candidate_top, oracle_top, 5);
    top10_overlap_sum_ += overlap(candidate_top, oracle_top, 10);
    top5_denominator_sum_ += top5_den;
    top10_denominator_sum_ += top10_den;

    const double candidate_log_z = logsumexp(candidate_logits);
    const double oracle_log_z = logsumexp(oracle_logits);
    long double kl = 0.0L;
    for (std::size_t i = 0; i < oracle_logits.size(); ++i) {
        const long double log_p =
            static_cast<long double>(oracle_logits[i]) - oracle_log_z;
        const long double log_q =
            static_cast<long double>(candidate_logits[i]) - candidate_log_z;
        const long double p = std::exp(log_p);
        kl += p * (log_p - log_q);
    }
    const double kl_value =
        std::max(0.0, static_cast<double>(kl));
    kl_sum_ += kl_value;
    kl_values_.push_back(kl_value);

    if (target_token >= 0 &&
        static_cast<std::size_t>(target_token) < oracle_logits.size()) {
        ++nll_positions_;
        oracle_nll_sum_ +=
            oracle_log_z - static_cast<double>(oracle_logits[target_token]);
        candidate_nll_sum_ +=
            candidate_log_z - static_cast<double>(candidate_logits[target_token]);
    }

    Phase11DivergenceExample example{
        .position = position,
        .target_token = target_token,
        .candidate_top1 = candidate_top.ids[0],
        .oracle_top1 = oracle_top.ids[0],
        .kl_divergence = kl_value,
        .max_logit_error = local_max_error,
    };
    if (candidate_top.ids[0] != oracle_top.ids[0] &&
        !first_top1_divergence_) {
        first_top1_divergence_ = example;
    }
    if (!worst_kl_divergence_ ||
        kl_value > worst_kl_divergence_->kl_divergence) {
        worst_kl_divergence_ = example;
    }
}

Phase11OracleMetrics Phase11OracleAccumulator::finalize() const {
    Phase11OracleMetrics out{};
    out.positions = positions_;
    out.nll_positions = nll_positions_;
    out.nonfinite_positions = nonfinite_positions_;
    if (positions_ != 0) {
        out.top1_agreement =
            static_cast<double>(top1_matches_) / static_cast<double>(positions_);
    }
    const std::uint32_t finite_positions = positions_ - nonfinite_positions_;
    if (finite_positions != 0) {
        out.mean_kl =
            static_cast<double>(kl_sum_ / static_cast<long double>(finite_positions));
    }
    if (!kl_values_.empty()) {
        std::vector<double> sorted = kl_values_;
        std::sort(sorted.begin(), sorted.end());
        const std::size_t index =
            std::min(sorted.size() - 1,
                     static_cast<std::size_t>(
                         std::ceil(0.99 * static_cast<double>(sorted.size()))) - 1);
        out.p99_kl = sorted[index];
    }
    if (nll_positions_ != 0) {
        const long double oracle_mean =
            oracle_nll_sum_ / static_cast<long double>(nll_positions_);
        const long double candidate_mean =
            candidate_nll_sum_ / static_cast<long double>(nll_positions_);
        const long double denominator =
            std::max(std::abs(oracle_mean), 1.0e-12L);
        out.relative_mean_nll_delta =
            static_cast<double>(std::abs(candidate_mean - oracle_mean) /
                                denominator);
    }
    if (top5_denominator_sum_ != 0) {
        out.mean_top5_overlap =
            static_cast<double>(top5_overlap_sum_) /
            static_cast<double>(top5_denominator_sum_);
    }
    if (top10_denominator_sum_ != 0) {
        out.mean_top10_overlap =
            static_cast<double>(top10_overlap_sum_) /
            static_cast<double>(top10_denominator_sum_);
    }
    out.maximum_logit_error = maximum_logit_error_;
    out.first_top1_divergence = first_top1_divergence_;
    out.worst_kl_divergence = worst_kl_divergence_;
    return out;
}

bool phase11_oracle_passes(const Phase11OracleMetrics& metrics,
                           const Phase11OracleThresholds& thresholds) {
    return metrics.positions >= thresholds.minimum_positions &&
           metrics.nonfinite_positions == 0 &&
           metrics.top1_agreement >= thresholds.minimum_top1_agreement &&
           metrics.mean_kl <= thresholds.maximum_mean_kl &&
           metrics.p99_kl <= thresholds.maximum_p99_kl &&
           metrics.nll_positions != 0 &&
           metrics.relative_mean_nll_delta <=
               thresholds.maximum_relative_mean_nll_delta;
}

Phase11CudaMemorySnapshot phase11_cuda_memory_snapshot() {
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    return Phase11CudaMemorySnapshot{
        .free_bytes = static_cast<std::uint64_t>(free_bytes),
        .total_bytes = static_cast<std::uint64_t>(total_bytes),
    };
}

Phase11VramReconciliation
reconcile_phase11_vram(const FlashNextStaticVramLedger& ledger,
                       const Phase11VramObservation& observation) {
    Phase11VramReconciliation out{};
    out.planned_nonexpert_weight_bytes =
        ledger.token_embedding_payload_bytes +
        ledger.output_head_payload_bytes +
        ledger.nonexpert_model_payload_bytes +
        ledger.device_weight_alignment_padding_bytes;
    out.planned_runtime_bytes = ledger.runtime_plan_device_bytes;
    out.planned_total_device_bytes =
        ledger.total_planned_device_bytes;
    out.observed_model_load_bytes =
        consumed_bytes(observation.before_model_load,
                       observation.after_model_load);
    out.observed_runtime_bytes =
        consumed_bytes(observation.after_model_load,
                       observation.after_runtime_allocation);
    out.observed_total_bytes =
        consumed_bytes(observation.before_model_load,
                       observation.after_runtime_allocation);
    out.observed_minus_planned_bytes =
        signed_delta(out.observed_total_bytes,
                     out.planned_total_device_bytes);
    return out;
}

} // namespace ninfer::targets::qwen3_8_flash_next::detail
