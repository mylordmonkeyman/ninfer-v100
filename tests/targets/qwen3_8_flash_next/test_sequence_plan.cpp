#include <ninfer/targets/qwen3_8_flash_next/package.h>
#include "runtime/engine/kv_capacity.h"

#include <iostream>
#include <stdexcept>

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

int test_sequence_plan_curve_consistency() {
    using Package = ninfer::targets::qwen3_8_flash_next::Package;

    int failures = 0;

    ninfer::EngineOptions options{};
    options.max_concurrency = 4;
    options.max_context     = 8192;

    ninfer::DeviceContext dummy_device{};
    auto planner = Package::make_sequence_planner(
        dummy_device, options, Package::WeightsProfile::MixedNvfp4Fp8PleInt4);

    const auto curve = planner.capacity_curve();
    failures += check(curve.main_page_tokens == 256, "main_page_tokens must be 256");
    failures += check(curve.minimum_main_page_groups == (8192 + 255) / 256,
                      "minimum_main_page_groups mismatch");
    failures += check(curve.bytes_per_additional_main_page_group == 6488064,
                      "bytes_per_additional_main_page_group mismatch");

    // Test multiple main_page_groups values
    for (std::uint32_t groups : {curve.minimum_main_page_groups,
                                 curve.minimum_main_page_groups + 1,
                                 curve.minimum_main_page_groups + 10,
                                 curve.minimum_main_page_groups + 50}) {
        auto p = Package::make_sequence_planner(
            dummy_device, options, Package::WeightsProfile::MixedNvfp4Fp8PleInt4);
        auto plan = std::move(p).finalize(groups);

        const std::size_t expected_reservation =
            curve.minimum_device_reservation_bytes +
            static_cast<std::size_t>(groups - curve.minimum_main_page_groups) *
                curve.bytes_per_additional_main_page_group;
        const std::uint32_t expected_tokens = groups * 256;

        failures += check(plan.device_reservation_bytes() == expected_reservation,
                          "plan.device_reservation_bytes() does not match expected curve reservation");
        failures += check(plan.kv_capacity() == expected_tokens,
                          "plan.kv_capacity() does not match expected resolved tokens");
        failures += check(plan.capacity() == expected_tokens,
                          "plan.capacity() does not match expected resolved tokens");
        failures += check(plan.max_concurrency() == 4,
                          "plan.max_concurrency() mismatch");
    }

    return failures;
}

int test_sequence_plan_fp8_curve_consistency() {
    using Package = ninfer::targets::qwen3_8_flash_next::Package;

    int failures = 0;

    ninfer::EngineOptions options{};
    options.max_concurrency = 4;
    options.max_context     = 8192;
    options.kv_cache        = ninfer::KvCacheStorage::Fp8E4M3Row256;

    ninfer::DeviceContext dummy_device{};
    auto planner = Package::make_sequence_planner(
        dummy_device, options, Package::WeightsProfile::MixedNvfp4Fp8PleInt4);

    const auto curve = planner.capacity_curve();
    failures += check(curve.main_page_tokens == 256, "FP8 main_page_tokens must be 256");
    failures += check(curve.minimum_main_page_groups == (8192 + 255) / 256,
                      "FP8 minimum_main_page_groups mismatch");
    failures += check(curve.bytes_per_additional_main_page_group == 3342336,
                      "FP8 bytes_per_additional_main_page_group mismatch (expected 3,342,336)");

    // Test multiple main_page_groups values
    for (std::uint32_t groups : {curve.minimum_main_page_groups,
                                 curve.minimum_main_page_groups + 1,
                                 curve.minimum_main_page_groups + 10,
                                 curve.minimum_main_page_groups + 50}) {
        auto p = Package::make_sequence_planner(
            dummy_device, options, Package::WeightsProfile::MixedNvfp4Fp8PleInt4);
        auto plan = std::move(p).finalize(groups);

        const std::size_t expected_reservation =
            curve.minimum_device_reservation_bytes +
            static_cast<std::size_t>(groups - curve.minimum_main_page_groups) *
                curve.bytes_per_additional_main_page_group;
        const std::uint32_t expected_tokens = groups * 256;

        failures += check(plan.device_reservation_bytes() == expected_reservation,
                          "FP8 plan.device_reservation_bytes() does not match expected curve reservation");
        failures += check(plan.kv_capacity() == expected_tokens,
                          "FP8 plan.kv_capacity() does not match expected resolved tokens");
        failures += check(plan.capacity() == expected_tokens,
                          "FP8 plan.capacity() does not match expected resolved tokens");
        failures += check(plan.max_concurrency() == 4,
                          "FP8 plan.max_concurrency() mismatch");
    }

    // Claude acceptance scenario test:
    // Concurrency 8, max-context 262144 (1024 min groups), continuation 48.
    // Capacity 524288 (2048 groups).
    {
        ninfer::EngineOptions accept_opts{};
        accept_opts.max_concurrency = 8;
        accept_opts.max_context     = 262144;
        accept_opts.context_cache.enabled = true;
        accept_opts.context_cache.max_private_continuations = 48;
        accept_opts.kv_cache        = ninfer::KvCacheStorage::Fp8E4M3Row256;

        auto accept_planner = Package::make_sequence_planner(
            dummy_device, accept_opts, Package::WeightsProfile::MixedNvfp4Fp8PleInt4);
        const auto accept_curve = accept_planner.capacity_curve();
        failures += check(accept_curve.minimum_main_page_groups == 1024,
                          "262k context requires 1024 min page groups");
        auto accept_plan = std::move(accept_planner).finalize(2048);
        failures += check(accept_plan.kv_capacity() == 524288,
                          "2048 groups must resolve to 524288 tokens");
        failures += check(accept_plan.continuation_capacity() == 48,
                          "continuation capacity must be 48");
    }

    // Unsupported dtype validation test
    {
        ninfer::EngineOptions bad_opts{};
        bad_opts.kv_cache = ninfer::KvCacheStorage::Nvfp4Group16;
        bool threw = false;
        try {
            (void)Package::make_sequence_planner(
                dummy_device, bad_opts, Package::WeightsProfile::MixedNvfp4Fp8PleInt4);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        failures += check(threw, "Package::make_sequence_planner must reject Nvfp4Group16 KV cache");
    }

    return failures;
}

int test_continuation_capacity_clamping() {
    using Package = ninfer::targets::qwen3_8_flash_next::Package;

    int failures = 0;
    ninfer::DeviceContext dummy_device{};

    // 1. Concurrency 8: floor is 16 slots. kMaxStateSlots is 64, so ceiling for continuations is 48.
    // Requesting 128 must clamp to 48 without throwing.
    {
        ninfer::EngineOptions options{};
        options.max_concurrency = 8;
        options.max_context     = 4096;
        options.context_cache.enabled = true;
        options.context_cache.max_private_continuations = 128;

        auto planner = Package::make_sequence_planner(
            dummy_device, options, Package::WeightsProfile::MixedNvfp4Fp8PleInt4);
        const auto curve = planner.capacity_curve();
        auto plan = std::move(planner).finalize(curve.minimum_main_page_groups);

        failures += check(plan.continuation_capacity() == 48,
                          "concurrency 8: continuation capacity must clamp from 128 to 48");
    }

    // 2. Concurrency 4: floor is 8 slots. Ceiling for continuations is 64 - 8 = 56.
    // Requesting 128 must clamp to 56 without throwing.
    {
        ninfer::EngineOptions options{};
        options.max_concurrency = 4;
        options.max_context     = 4096;
        options.context_cache.enabled = true;
        options.context_cache.max_private_continuations = 128;

        auto planner = Package::make_sequence_planner(
            dummy_device, options, Package::WeightsProfile::MixedNvfp4Fp8PleInt4);
        const auto curve = planner.capacity_curve();
        auto plan = std::move(planner).finalize(curve.minimum_main_page_groups);

        failures += check(plan.continuation_capacity() == 56,
                          "concurrency 4: continuation capacity must clamp from 128 to 56");
    }

    // 3. Concurrency 4: requesting 20 (<= 56) must remain 20 without clamping.
    {
        ninfer::EngineOptions options{};
        options.max_concurrency = 4;
        options.max_context     = 4096;
        options.context_cache.enabled = true;
        options.context_cache.max_private_continuations = 20;

        auto planner = Package::make_sequence_planner(
            dummy_device, options, Package::WeightsProfile::MixedNvfp4Fp8PleInt4);
        const auto curve = planner.capacity_curve();
        auto plan = std::move(planner).finalize(curve.minimum_main_page_groups);

        failures += check(plan.continuation_capacity() == 20,
                          "concurrency 4: continuation capacity 20 must remain 20");
    }

    // 4. Context cache disabled: continuation capacity must be 0.
    {
        ninfer::EngineOptions options{};
        options.max_concurrency = 4;
        options.max_context     = 4096;
        options.context_cache.enabled = false;

        auto planner = Package::make_sequence_planner(
            dummy_device, options, Package::WeightsProfile::MixedNvfp4Fp8PleInt4);
        const auto curve = planner.capacity_curve();
        auto plan = std::move(planner).finalize(curve.minimum_main_page_groups);

        failures += check(plan.continuation_capacity() == 0,
                          "disabled cache: continuation capacity must be 0");
    }

    // 5. Concurrency 1: floor is 2 slots. Ceiling for continuations is 64 - 2 = 62.
    // Requesting 100 must clamp to 62.
    {
        ninfer::EngineOptions options{};
        options.max_concurrency = 1;
        options.max_context     = 4096;
        options.context_cache.enabled = true;
        options.context_cache.max_private_continuations = 100;

        auto planner = Package::make_sequence_planner(
            dummy_device, options, Package::WeightsProfile::MixedNvfp4Fp8PleInt4);
        const auto curve = planner.capacity_curve();
        auto plan = std::move(planner).finalize(curve.minimum_main_page_groups);

        failures += check(plan.continuation_capacity() == 62,
                          "concurrency 1: continuation capacity must clamp from 100 to 62");
    }

    return failures;
}

int test_volta_cuda_graph_policy() {
#if defined(NINFER_VOLTA_BUILD)
    using Package = ninfer::targets::qwen3_8_flash_next::Package;
    ninfer::EngineOptions options{};
    options.max_context     = 4096;
    options.max_concurrency = 1;
    options.use_cuda_graph  = true;

    ninfer::DeviceContext dummy_device{};
    auto planner = Package::make_sequence_planner(
        dummy_device, options, Package::WeightsProfile::MixedNvfp4Fp8PleInt4);
    const auto curve = planner.capacity_curve();
    auto plan = std::move(planner).finalize(curve.minimum_main_page_groups);

    // The public option may request graphing, but initial SM70 bring-up is
    // architecture-forced eager and reserves no graph memory.
    ninfer::targets::qwen3_8_flash_next::detail::FlashNextRuntimeConfig direct{};
    direct.use_cuda_graph = true;
    const auto direct_curve =
        ninfer::targets::qwen3_8_flash_next::detail::flash_next_capacity_curve(direct);
    const auto direct_plan =
        ninfer::targets::qwen3_8_flash_next::detail::finalize_flash_next_runtime_plan(
            direct, direct_curve.minimum_main_page_groups);
    if (direct_plan.config.use_cuda_graph || direct_plan.cuda_graph_allowance_bytes != 0) {
        std::cerr << "FAIL: SM70 runtime plan must force eager execution with zero graph allowance\n";
        return 1;
    }
    (void)plan;
#endif
    return 0;
}

int test_make_sequence_planner_draft_tokens() {
    using Package = ninfer::targets::qwen3_8_flash_next::Package;

    int failures = 0;
    ninfer::DeviceContext dummy_device{};

    // 1. draft_tokens 1 at c=8: floor is (1+1)*8 = 16. Limit 64 - 16 = 48.
    // Requested 48 remains 48 without clamping.
    {
        ninfer::EngineOptions options{};
        options.max_concurrency = 8;
        options.max_context     = 4096;
        options.speculative.backend = ninfer::SpeculativeBackend::Mtp;
        options.speculative.draft_tokens = 1;
        options.context_cache.enabled = true;
        options.context_cache.max_private_continuations = 48;

        auto planner = Package::make_sequence_planner(
            dummy_device, options, Package::WeightsProfile::MixedNvfp4Fp8PleInt4);
        const auto curve = planner.capacity_curve();
        auto plan = std::move(planner).finalize(curve.minimum_main_page_groups);

        failures += check(plan.continuation_capacity() == 48,
                          "draft_tokens 1 at c=8: continuation capacity must be 48");
    }

    // 2. draft_tokens 2 at c=8: floor is (2+1)*8 = 24. Limit 64 - 24 = 40.
    // Requested 48 must clamp cleanly to 40.
    {
        ninfer::EngineOptions options{};
        options.max_concurrency = 8;
        options.max_context     = 4096;
        options.speculative.backend = ninfer::SpeculativeBackend::Mtp;
        options.speculative.draft_tokens = 2;
        options.context_cache.enabled = true;
        options.context_cache.max_private_continuations = 48;

        auto planner = Package::make_sequence_planner(
            dummy_device, options, Package::WeightsProfile::MixedNvfp4Fp8PleInt4);
        const auto curve = planner.capacity_curve();
        auto plan = std::move(planner).finalize(curve.minimum_main_page_groups);

        failures += check(plan.continuation_capacity() == 40,
                          "draft_tokens 2 at c=8: continuation capacity must clamp from 48 to 40");
    }

    // 3. draft_tokens 4 at c=8: floor is (4+1)*8 = 40. Limit 64 - 40 = 24.
    // Requested 48 must clamp cleanly to 24.
    {
        ninfer::EngineOptions options{};
        options.max_concurrency = 8;
        options.max_context     = 4096;
        options.speculative.backend = ninfer::SpeculativeBackend::Mtp;
        options.speculative.draft_tokens = 4;
        options.context_cache.enabled = true;
        options.context_cache.max_private_continuations = 48;

        auto planner = Package::make_sequence_planner(
            dummy_device, options, Package::WeightsProfile::MixedNvfp4Fp8PleInt4);
        const auto curve = planner.capacity_curve();
        auto plan = std::move(planner).finalize(curve.minimum_main_page_groups);

        failures += check(plan.continuation_capacity() == 24,
                          "draft_tokens 4 at c=8: continuation capacity must clamp from 48 to 24");
    }

    return failures;
}

} // namespace

int main() {
    try {
        int failures = 0;
        failures += test_sequence_plan_curve_consistency();
        failures += test_sequence_plan_fp8_curve_consistency();
        failures += test_continuation_capacity_clamping();
        failures += test_volta_cuda_graph_policy();
        failures += test_make_sequence_planner_draft_tokens();

        if (failures == 0) {
            std::cout << "All SequencePlan tests passed cleanly.\n";
            return 0;
        }
        std::cerr << failures << " SequencePlan test(s) failed.\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "EXCEPTION: " << e.what() << "\n";
        return 1;
    }
}
