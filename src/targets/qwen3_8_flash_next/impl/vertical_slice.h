#pragma once

#include "targets/qwen3_8_flash_next/impl/load/bindings.h"
#include "targets/qwen3_8_flash_next/impl/load/loader.h"
#include "targets/qwen3_8_flash_next/impl/runtime_plan.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::targets::qwen3_8_flash_next::detail {

struct Phase11VerticalSliceContract {
    FlashNextRuntimeConfig runtime{};
    LoadFeatures load{};
    bool expert_cache_enabled = false;
    std::uint32_t minimum_teacher_forced_positions = 4'096;
};

[[nodiscard]] Phase11VerticalSliceContract
make_phase11_vertical_slice_contract(std::uint32_t max_context = 8'192,
                                     std::uint32_t prefill_chunk = 128);

void validate_phase11_preflight(const FlashNextPreflightReport& report);

struct Phase11DivergenceExample {
    std::uint32_t position = 0;
    std::int32_t target_token = -1;
    std::int32_t candidate_top1 = -1;
    std::int32_t oracle_top1 = -1;
    double kl_divergence = 0.0;
    float max_logit_error = 0.0F;
};

struct Phase11OracleThresholds {
    std::uint32_t minimum_positions = 4'096;
    double minimum_top1_agreement = 0.99;
    double maximum_mean_kl = 1.0e-3;
    double maximum_p99_kl = 1.0e-2;
    double maximum_relative_mean_nll_delta = 5.0e-3;
};

struct Phase11OracleMetrics {
    std::uint32_t positions = 0;
    std::uint32_t nll_positions = 0;
    std::uint32_t nonfinite_positions = 0;
    double top1_agreement = 0.0;
    double mean_kl = 0.0;
    double p99_kl = 0.0;
    double relative_mean_nll_delta = 0.0;
    double mean_top5_overlap = 0.0;
    double mean_top10_overlap = 0.0;
    float maximum_logit_error = 0.0F;
    std::optional<Phase11DivergenceExample> first_top1_divergence;
    std::optional<Phase11DivergenceExample> worst_kl_divergence;
};

class Phase11OracleAccumulator {
public:
    void observe(std::uint32_t position, std::int32_t target_token,
                 std::span<const float> candidate_logits,
                 std::span<const float> oracle_logits);

    [[nodiscard]] Phase11OracleMetrics finalize() const;

private:
    std::uint64_t top1_matches_ = 0;
    std::uint64_t top5_overlap_sum_ = 0;
    std::uint64_t top10_overlap_sum_ = 0;
    std::uint64_t top5_denominator_sum_ = 0;
    std::uint64_t top10_denominator_sum_ = 0;
    std::uint32_t positions_ = 0;
    std::uint32_t nll_positions_ = 0;
    std::uint32_t nonfinite_positions_ = 0;
    long double kl_sum_ = 0.0L;
    long double oracle_nll_sum_ = 0.0L;
    long double candidate_nll_sum_ = 0.0L;
    float maximum_logit_error_ = 0.0F;
    std::vector<double> kl_values_;
    std::optional<Phase11DivergenceExample> first_top1_divergence_;
    std::optional<Phase11DivergenceExample> worst_kl_divergence_;
};

[[nodiscard]] bool
phase11_oracle_passes(const Phase11OracleMetrics& metrics,
                      const Phase11OracleThresholds& thresholds = {});

struct Phase11CudaMemorySnapshot {
    std::uint64_t free_bytes = 0;
    std::uint64_t total_bytes = 0;
};

[[nodiscard]] Phase11CudaMemorySnapshot phase11_cuda_memory_snapshot();

struct Phase11VramObservation {
    Phase11CudaMemorySnapshot before_model_load{};
    Phase11CudaMemorySnapshot after_model_load{};
    Phase11CudaMemorySnapshot after_runtime_allocation{};
};

struct Phase11VramReconciliation {
    std::uint64_t planned_nonexpert_weight_bytes = 0;
    std::uint64_t planned_runtime_bytes = 0;
    std::uint64_t planned_total_device_bytes = 0;
    std::uint64_t observed_model_load_bytes = 0;
    std::uint64_t observed_runtime_bytes = 0;
    std::uint64_t observed_total_bytes = 0;
    std::int64_t observed_minus_planned_bytes = 0;
};

[[nodiscard]] Phase11VramReconciliation
reconcile_phase11_vram(const FlashNextStaticVramLedger& ledger,
                       const Phase11VramObservation& observation);

} // namespace ninfer::targets::qwen3_8_flash_next::detail
