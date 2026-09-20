#include "targets/qwen3_8_flash_next/impl/vertical_slice.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace ninfer::targets::qwen3_8_flash_next::detail;

namespace {

int fail(const char* message) {
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

} // namespace

int main() {
    int failures = 0;

    const auto contract = make_phase11_vertical_slice_contract();
    if (contract.runtime.use_cuda_graph ||
        contract.runtime.speculative_draft_tokens != 0 ||
        contract.runtime.max_concurrency != 1 ||
        contract.runtime.vision_enabled ||
        contract.runtime.kv_cache != ninfer::KvCacheStorage::BFloat16 ||
        contract.runtime.gdn_state_storage != ninfer::GdnStateStorage::FP32 ||
        !contract.load.host_backed_experts ||
        contract.load.mtp ||
        contract.expert_cache_enabled ||
        contract.minimum_teacher_forced_positions != 4'096) {
        failures += fail("Phase 11 contract fields are not correctness-first");
    }

    std::vector<float> oracle(32);
    for (std::size_t i = 0; i < oracle.size(); ++i) {
        oracle[i] = static_cast<float>(i) * 0.03125F - 0.5F;
    }

    Phase11OracleAccumulator identical;
    for (std::uint32_t position = 0; position < 32; ++position) {
        identical.observe(position,
                          static_cast<std::int32_t>((position + 1) % oracle.size()),
                          oracle, oracle);
    }
    const auto identical_metrics = identical.finalize();
    Phase11OracleThresholds smoke_thresholds{};
    smoke_thresholds.minimum_positions = 32;
    if (!phase11_oracle_passes(identical_metrics, smoke_thresholds) ||
        identical_metrics.top1_agreement != 1.0 ||
        identical_metrics.mean_kl > 1.0e-12 ||
        identical_metrics.p99_kl > 1.0e-12 ||
        identical_metrics.relative_mean_nll_delta > 1.0e-12 ||
        identical_metrics.mean_top5_overlap != 1.0 ||
        identical_metrics.mean_top10_overlap != 1.0 ||
        identical_metrics.maximum_logit_error != 0.0F ||
        identical_metrics.first_top1_divergence.has_value()) {
        failures += fail("identical teacher-forced logits did not pass exactly");
    }

    std::vector<float> perturbed = oracle;
    perturbed.back() = -100.0F;
    perturbed.front() = 100.0F;
    Phase11OracleAccumulator divergent;
    for (std::uint32_t position = 0; position < 32; ++position) {
        divergent.observe(position,
                          static_cast<std::int32_t>((position + 1) % oracle.size()),
                          perturbed, oracle);
    }
    const auto divergent_metrics = divergent.finalize();
    if (phase11_oracle_passes(divergent_metrics, smoke_thresholds) ||
        divergent_metrics.top1_agreement >= 1.0 ||
        !divergent_metrics.first_top1_divergence.has_value() ||
        !divergent_metrics.worst_kl_divergence.has_value()) {
        failures += fail("divergent teacher-forced logits were not rejected");
    }

    FlashNextStaticVramLedger ledger{};
    ledger.token_embedding_payload_bytes = 100;
    ledger.output_head_payload_bytes = 200;
    ledger.nonexpert_model_payload_bytes = 300;
    ledger.device_weight_alignment_padding_bytes = 40;
    ledger.runtime_plan_device_bytes = 500;
    ledger.total_planned_device_bytes = 1'140;
    const Phase11VramObservation observation{
        .before_model_load = {.free_bytes = 10'000, .total_bytes = 20'000},
        .after_model_load = {.free_bytes = 9'360, .total_bytes = 20'000},
        .after_runtime_allocation = {.free_bytes = 8'860, .total_bytes = 20'000},
    };
    const auto reconciliation = reconcile_phase11_vram(ledger, observation);
    if (reconciliation.planned_nonexpert_weight_bytes != 640 ||
        reconciliation.planned_runtime_bytes != 500 ||
        reconciliation.observed_model_load_bytes != 640 ||
        reconciliation.observed_runtime_bytes != 500 ||
        reconciliation.observed_total_bytes != 1'140 ||
        reconciliation.observed_minus_planned_bytes != 0) {
        failures += fail("VRAM reconciliation arithmetic mismatch");
    }

    if (failures == 0) {
        std::cout << "PASS: Phase 11 vertical-slice contract and oracle metrics\n";
        return 0;
    }
    return 1;
}
