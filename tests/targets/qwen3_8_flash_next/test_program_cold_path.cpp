#include <ninfer/targets/qwen3_8_flash_next/package.h>
#include <ninfer/targets/qwen3_8_flash_next/runtime.h>
#include <ninfer/targets/qwen3_6/frontend.h>
#include <ninfer/targets/qwen3_6/frontend_resources.h>
#include <ninfer/targets/qwen3_6/prepared_prompt.h>

#include "core/arena.h"
#include "core/device.h"
#include "runtime/contract/types.h"
#include "runtime/engine/context_cost.h"
#include "targets/qwen3_8_flash_next/impl/load/materialized.h"
#include "targets/qwen3_8_flash_next/impl/model_view.h"
#include "targets/qwen3_8_flash_next/impl/ple_table.h"
#include "targets/qwen3_8_flash_next/impl/program_impl.h"
#include "targets/qwen3_8_flash_next/impl/runtime_plan.h"

#include <cuda_runtime.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {

bool cuda_unavailable(cudaError_t error) {
    return error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver;
}

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

ninfer::targets::qwen3_6::PreparedPrompt make_dummy_prompt(std::size_t num_tokens) {
    ninfer::targets::qwen3_6::PreparedPromptData data;
    data.token_ids.resize(num_tokens);
    data.token_types.resize(num_tokens, 0);
    data.positions.resize(num_tokens * 3);
    for (std::size_t i = 0; i < num_tokens; ++i) {
        data.token_ids[i]                  = static_cast<ninfer::TokenId>(100 + i);
        data.positions[i]                  = static_cast<std::int32_t>(i);
        data.positions[num_tokens + i]     = static_cast<std::int32_t>(i);
        data.positions[2 * num_tokens + i] = static_cast<std::int32_t>(i);
    }
    data.identity.reusable = false;
    return ninfer::targets::qwen3_6::PreparedPromptAccess::construct(std::move(data));
}

int test_program_lifecycle(ninfer::DeviceContext& device) {
    using namespace ninfer::targets::qwen3_8_flash_next;
    using namespace ninfer::targets::qwen3_8_flash_next::detail;

    int failures = 0;

    // 1. Synthetic PLE table and mock model
    constexpr std::uint64_t rows         = 1;
    constexpr std::uint64_t width        = 160;
    constexpr std::uint64_t scale_offset = 256;
    std::vector<std::byte> encoded(scale_offset + (width / 16) * 2, std::byte{0});
    std::fill_n(encoded.begin(), width / 2, std::byte{0x88});
    for (std::uint8_t index = 0; index < 8; ++index) {
        encoded[index] = static_cast<std::byte>(index * 2 | ((index * 2 + 1) << 4));
    }
    constexpr std::uint16_t half_point_five = 0x3800;
    for (std::size_t offset = scale_offset; offset < encoded.size(); offset += 2) {
        std::memcpy(encoded.data() + offset, &half_point_five, sizeof(half_point_five));
    }
    PleTableView ple_table;
    for (PleShardView& shard : ple_table.shards) {
        shard = make_ple_shard_view(encoded, rows, width);
    }

    TextModelView mock_model{};
    mock_model.ple.table = ple_table;

    // 2. Runtime Plan: max_concurrency = 2, max_context = 512
    FlashNextRuntimeConfig cfg{
        .max_concurrency     = 2,
        .max_context         = 512,
        .state_slot_capacity = 0,
        .prefill_chunk       = 512,
    };
    const auto curve = flash_next_capacity_curve(cfg);
    auto plan        = finalize_flash_next_runtime_plan(cfg, curve.minimum_main_page_groups);

    auto program_impl = std::make_unique<ProgramImpl>(nullptr, plan, device, mock_model);
    Program program(std::move(program_impl));

    // Check initial state
    failures += check(program.resource_revision().value == 1, "Initial resource revision must be 1");
    failures += check(program.physical_usage().device_state_slots == 0,
                      "Initial active state slots must be 0");
    failures += check(!program.has_context_transaction(),
                      "Initial has_context_transaction must be false");

    // 3. Plan Request 1
    const auto prompt1 = make_dummy_prompt(16);
    ninfer::runtime::ResolvedExecutionOptions exec_options{};
    exec_options.requested_output_tokens = 32;
    exec_options.sampling.temperature   = 0.7F;
    exec_options.sampling.top_p         = 0.8F;

    auto base_plan1 = program.plan_request(prompt1, exec_options);
    failures += check(base_plan1.summary().prompt_tokens == 16, "prompt_tokens mismatch");
    failures += check(base_plan1.summary().requested_output_tokens == 32,
                      "requested_output_tokens mismatch");
    failures += check(base_plan1.summary().effective_output_tokens == 32,
                      "effective_output_tokens mismatch");
    failures += check(base_plan1.summary().prefix_reuse_path == ninfer::PrefixReusePath::Root,
                      "prefix_reuse_path must be Root");
    failures += check(program.isolated_request_feasible(base_plan1),
                      "isolated_request_feasible must be true initially");

    // 4. Admission Inspection 1
    ninfer::runtime::ContextMachineCostModel cost_model{};
    auto candidate1 = program.inspect_admission(
        prompt1, base_plan1, ninfer::runtime::LaneId(0), nullptr, nullptr, std::nullopt, false);
    failures += check(candidate1.has_value(), "Admission candidate 1 must be present");
    failures += check(
        candidate1->identity_assessment().physical_status ==
            ninfer::runtime::MaterializationPhysicalStatus::Feasible,
        "Candidate 1 must be Feasible");

    // 5. Seal Identity 1
    auto resource_plan1 = program.seal_identity(*candidate1, prompt1);
    failures += check(resource_plan1.has_value(), "Resource plan 1 must be present");
    failures += check(resource_plan1->resource_revision().value == 1, "Resource plan revision mismatch");

    // 6. Start Resource Transaction 1
    std::atomic<bool> flag1{false};
    ninfer::runtime::CancellationFlagView cancellation1{&flag1};
    auto reserve_status1 = program.start_resource_transaction(
        std::move(*resource_plan1), make_dummy_prompt(16), cancellation1);
    failures += check(
        reserve_status1 == ninfer::runtime::ContextTransactionReserveStatus::Reserved,
        "start_resource_transaction must return Reserved");
    failures += check(program.has_context_transaction(),
                      "has_context_transaction must be true after start");
    failures += check(program.resource_revision().value == 2,
                      "resource_revision must increment to 2 after lane allocation");

    // 7. Progress Context Transaction 1
    auto progress1 = program.progress_context_transaction(cancellation1);
    auto* mat1     = std::get_if<MaterializationResult>(&progress1);
    failures += check(mat1 != nullptr, "Progress must return MaterializationResult");
    failures += check(mat1->status == ninfer::runtime::ContextTransactionStatus::Published,
                      "Materialization status must be Published");
    failures += check(mat1->published.has_value(), "Published StartResult must be present");

    SequenceHandle seq1 = mat1->published->sequence;
    failures += check(seq1.lane().value == 0, "Sequence 1 lane must be 0");
    failures += check(seq1.epoch() == 1, "Sequence 1 epoch must be 1");

    program.finalize_context_transaction();
    failures += check(!program.has_context_transaction(),
                      "has_context_transaction must be false after finalize");

    // 8. Request 2 on second lane
    const auto prompt2 = make_dummy_prompt(20);
    auto base_plan2    = program.plan_request(prompt2, exec_options);
    auto candidate2    = program.inspect_admission(
        prompt2, base_plan2, ninfer::runtime::LaneId(1), nullptr, nullptr, std::nullopt, false);
    failures += check(
        candidate2->identity_assessment().physical_status ==
            ninfer::runtime::MaterializationPhysicalStatus::Feasible,
        "Candidate 2 must be Feasible");

    auto resource_plan2 = program.seal_identity(*candidate2, prompt2);
    auto reserve_status2 = program.start_resource_transaction(
        std::move(*resource_plan2), make_dummy_prompt(20), cancellation1);
    failures += check(
        reserve_status2 == ninfer::runtime::ContextTransactionReserveStatus::Reserved,
        "start_resource_transaction 2 must return Reserved");
    failures += check(program.resource_revision().value == 3,
                      "resource_revision must increment to 3 after 2nd lane allocation");

    auto progress2 = program.progress_context_transaction(cancellation1);
    auto* mat2     = std::get_if<MaterializationResult>(&progress2);
    SequenceHandle seq2 = mat2->published->sequence;
    failures += check(seq2.lane().value == 1, "Sequence 2 lane must be 1");
    program.finalize_context_transaction();

    // 9. With both lanes active (max_concurrency = 2), request 3 must be Infeasible
    failures += check(program.physical_usage().device_state_slots == 2,
                      "Active state slots must be 2");
    const auto prompt3 = make_dummy_prompt(10);
    auto base_plan3    = program.plan_request(prompt3, exec_options);
    auto candidate3    = program.inspect_admission(
        prompt3, base_plan3, ninfer::runtime::LaneId(0), nullptr, nullptr, std::nullopt, false);
    failures += check(
        candidate3->identity_assessment().physical_status ==
            ninfer::runtime::MaterializationPhysicalStatus::Infeasible,
        "Candidate 3 must be Infeasible when all lanes are occupied");
    // Isolation ignores current lane occupancy: the Engine turns a false answer into
    // Readiness::PermanentlyInfeasible (resource_manager.h), while a busy lane set is only
    // TemporarilyBlocked. The candidate above is what reports the momentary infeasibility.
    failures += check(program.isolated_request_feasible(base_plan3),
                      "isolated_request_feasible must stay true when all lanes are occupied");

    // 10. Finish Sequence 1
    auto finish1 = program.finish(seq1);
    failures += check(finish1.status == ninfer::runtime::ConsumeStatus::Consumed,
                      "finish seq1 status must be Consumed");
    failures += check(program.resource_revision().value == 4,
                      "resource_revision must increment to 4 after releasing lane 0");
    failures += check(program.physical_usage().device_state_slots == 1,
                      "Active state slots must be 1 after finishing seq1");
    failures += check(program.isolated_request_feasible(base_plan3),
                      "isolated_request_feasible must be true after releasing lane 0");

    // 11. Finish Sequence 2
    auto finish2 = program.finish(seq2);
    failures += check(finish2.status == ninfer::runtime::ConsumeStatus::Consumed,
                      "finish seq2 status must be Consumed");
    failures += check(program.resource_revision().value == 5,
                      "resource_revision must increment to 5 after releasing lane 1");
    failures += check(program.physical_usage().device_state_slots == 0,
                      "Active state slots must be 0 after finishing seq2");

    return failures;
}

int test_plan_request_validations(ninfer::DeviceContext& device) {
    using namespace ninfer::targets::qwen3_8_flash_next;
    using namespace ninfer::targets::qwen3_8_flash_next::detail;

    int failures = 0;

    FlashNextRuntimeConfig cfg{
        .max_concurrency     = 2,
        .max_context         = 128,
        .state_slot_capacity = 0,
        .prefill_chunk       = 128,
    };
    const auto curve = flash_next_capacity_curve(cfg);
    auto plan        = finalize_flash_next_runtime_plan(cfg, curve.minimum_main_page_groups);

    auto program_impl = std::make_unique<ProgramImpl>(nullptr, plan, device);
    Program program(std::move(program_impl));

    // 1. Empty prompt rejection
    try {
        ninfer::targets::qwen3_6::PreparedPrompt empty_prompt =
            ninfer::targets::qwen3_6::PreparedPromptAccess::construct(
                ninfer::targets::qwen3_6::PreparedPromptData{});
        ninfer::runtime::ResolvedExecutionOptions options{};
        (void)program.plan_request(empty_prompt, options);
        std::cerr << "Expected empty prompt rejection but none thrown\n";
        failures += 1;
    } catch (const std::invalid_argument&) {
        // Expected
    }

    // 2. Prompt exceeding max_context (plan.resolved_tokens)
    try {
        const auto long_prompt = make_dummy_prompt(plan.resolved_tokens + 1);
        ninfer::runtime::ResolvedExecutionOptions options{};
        (void)program.plan_request(long_prompt, options);
        std::cerr << "Expected prompt exceeding context rejection but none thrown\n";
        failures += 1;
    } catch (const std::invalid_argument&) {
        // Expected
    }

    // 3. Out-of-domain token rejection
    try {
        ninfer::targets::qwen3_6::PreparedPromptData bad_data;
        bad_data.token_ids   = {100, 248'077};
        bad_data.token_types = {0, 0};
        bad_data.positions   = {0, 1, 0, 1, 0, 1};
        auto bad_prompt =
            ninfer::targets::qwen3_6::PreparedPromptAccess::construct(std::move(bad_data));
        ninfer::runtime::ResolvedExecutionOptions options{};
        (void)program.plan_request(bad_prompt, options);
        std::cerr << "Expected out-of-domain token rejection but none thrown\n";
        failures += 1;
    } catch (const std::invalid_argument&) {
        // Expected
    }

    // 4. Invalid sampling parameters
    try {
        const auto prompt = make_dummy_prompt(10);
        ninfer::runtime::ResolvedExecutionOptions options{};
        options.sampling.top_p = 1.5F;
        (void)program.plan_request(prompt, options);
        std::cerr << "Expected invalid top_p rejection but none thrown\n";
        failures += 1;
    } catch (const std::invalid_argument&) {
        // Expected
    }

    // 5. Concurrency limit and CUDA graph allowance validations
    try {
        FlashNextRuntimeConfig bad_cfg_9{
            .max_concurrency = 9,
            .max_context     = 128,
            .prefill_chunk   = 128,
        };
        (void)finalize_flash_next_runtime_plan(bad_cfg_9, 1);
        std::cerr << "Expected max_concurrency > 8 rejection but none thrown\n";
        failures += 1;
    } catch (const std::invalid_argument&) {
        // Expected
    }

    try {
        FlashNextRuntimeConfig bad_cfg_0{
            .max_concurrency = 0,
            .max_context     = 128,
            .prefill_chunk   = 128,
        };
        (void)finalize_flash_next_runtime_plan(bad_cfg_0, 1);
        std::cerr << "Expected max_concurrency == 0 rejection but none thrown\n";
        failures += 1;
    } catch (const std::invalid_argument&) {
        // Expected
    }

    FlashNextRuntimeConfig cfg_graph_on{
        .max_concurrency = 4,
        .max_context     = 512,
        .prefill_chunk   = 512,
        .use_cuda_graph  = true,
    };
    const auto curve_on = flash_next_capacity_curve(cfg_graph_on);
    auto plan_on        = finalize_flash_next_runtime_plan(cfg_graph_on, curve_on.minimum_main_page_groups);
    if (plan_on.cuda_graph_allowance_bytes != 4ULL * 24ULL * 1024ULL * 1024ULL) {
        std::cerr << "Expected 96 MiB cuda_graph_allowance_bytes, got "
                  << plan_on.cuda_graph_allowance_bytes << "\n";
        failures += 1;
    }

    FlashNextRuntimeConfig cfg_graph_8192{
        .max_concurrency = 4,
        .max_context     = 8192,
        .prefill_chunk   = 1024,
        .use_cuda_graph  = true,
    };
    const auto curve_8192 = flash_next_capacity_curve(cfg_graph_8192);
    auto plan_8192 =
        finalize_flash_next_runtime_plan(cfg_graph_8192, curve_8192.minimum_main_page_groups);
    const auto buckets_8192 = flash_next_decode_graph_buckets(plan_8192.maximum_blocks);
    if (buckets_8192.count != 2 || buckets_8192.blocks[0] != 512 ||
        buckets_8192.blocks[1] != 2048) {
        std::cerr << "Expected max_context=8192 buckets {512, 2048}, got count="
                  << buckets_8192.count << "\n";
        failures += 1;
    }
    const std::size_t expected_8192 = 4ULL * 2ULL * 24ULL * 1024ULL * 1024ULL;
    if (plan_8192.cuda_graph_allowance_bytes != expected_8192) {
        std::cerr << "Expected 192 MiB cuda_graph_allowance_bytes for 4 * 2 buckets, got "
                  << plan_8192.cuda_graph_allowance_bytes << "\n";
        failures += 1;
    }

    FlashNextRuntimeConfig cfg_graph_off{
        .max_concurrency = 4,
        .max_context     = 512,
        .prefill_chunk   = 512,
        .use_cuda_graph  = false,
    };
    const auto curve_off = flash_next_capacity_curve(cfg_graph_off);
    auto plan_off        = finalize_flash_next_runtime_plan(cfg_graph_off, curve_off.minimum_main_page_groups);
    if (plan_off.cuda_graph_allowance_bytes != 0) {
        std::cerr << "Expected 0 cuda_graph_allowance_bytes when use_cuda_graph is false, got "
                  << plan_off.cuda_graph_allowance_bytes << "\n";
        failures += 1;
    }

    return failures;
}

int test_mtp_checkpoint_state_isolation(ninfer::DeviceContext& device) {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    FlashNextRuntimeConfig cfg{
        .max_concurrency = 2,
        .max_context = 128,
        .prefill_chunk = 128,
    };
    cfg.speculative_draft_tokens = 4;
    cfg.continuation_capacity = 2;
    const auto curve = flash_next_capacity_curve(cfg);
    auto plan = finalize_flash_next_runtime_plan(cfg, curve.minimum_main_page_groups);
    ProgramImpl program(nullptr, plan, device);
    auto& allocation = program.allocation_;
    auto* hidden = static_cast<std::byte*>(allocation.state_view().mtp_backbone_hidden.data);
    constexpr std::size_t hidden_bytes = 10'240 * sizeof(std::uint16_t);
    CUDA_CHECK(cudaMemsetAsync(hidden, 0x21, plan.config.state_slot_capacity * hidden_bytes,
                               device.stream));
    CUDA_CHECK(cudaMemsetAsync(hidden, 0x42, hidden_bytes, device.stream));
    // Publishing both checkpoints from lane 0 must preserve every speculative state
    // in lane 1, including states which are not its current source or destination.
    for (const auto& checkpoint : program.continuation_slots_) {
        program.executor_.copy_state_slot(0, checkpoint.cache_slot);
    }
    std::vector<std::byte> actual(5 * hidden_bytes);
    CUDA_CHECK(cudaMemcpyAsync(actual.data(), hidden + 5 * hidden_bytes, actual.size(),
                               cudaMemcpyDeviceToHost, device.stream));
    device.synchronize();
    return check(std::all_of(actual.begin(), actual.end(),
                            [](std::byte value) { return value == std::byte{0x21}; }),
                 "MTP checkpoint publication must not overwrite another lane's state ring");
}

int test_program_memory_summary_kv_cache(ninfer::DeviceContext& device) {
    using namespace ninfer::targets::qwen3_8_flash_next;
    using namespace ninfer::targets::qwen3_8_flash_next::detail;

    int failures = 0;

    // 1. FP8 KV Cache
    {
        FlashNextRuntimeConfig cfg{
            .max_concurrency = 2,
            .max_context     = 128,
            .prefill_chunk   = 128,
            .kv_cache        = ninfer::KvCacheStorage::Fp8E4M3Row256,
        };
        const auto curve = flash_next_capacity_curve(cfg);
        auto plan        = finalize_flash_next_runtime_plan(cfg, curve.minimum_main_page_groups);

        auto program_impl = std::make_unique<ProgramImpl>(nullptr, plan, device);
        Program program(std::move(program_impl));

        const auto summary = program.memory_summary();
        failures += check(summary.kv_cache == ninfer::KvCacheStorage::Fp8E4M3Row256,
                          "program.memory_summary().kv_cache must be Fp8E4M3Row256");
    }

    // 2. BF16 KV Cache
    {
        FlashNextRuntimeConfig cfg{
            .max_concurrency = 2,
            .max_context     = 128,
            .prefill_chunk   = 128,
            .kv_cache        = ninfer::KvCacheStorage::BFloat16,
        };
        const auto curve = flash_next_capacity_curve(cfg);
        auto plan        = finalize_flash_next_runtime_plan(cfg, curve.minimum_main_page_groups);

        auto program_impl = std::make_unique<ProgramImpl>(nullptr, plan, device);
        Program program(std::move(program_impl));

        const auto summary = program.memory_summary();
        failures += check(summary.kv_cache == ninfer::KvCacheStorage::BFloat16,
                          "program.memory_summary().kv_cache must be BFloat16");
    }

    return failures;
}

} // namespace

int main() {
    int device_count = 0;
    const cudaError_t err = cudaGetDeviceCount(&device_count);
    if (cuda_unavailable(err) || device_count == 0) {
        std::cout << "CUDA device unavailable; skipped GPU Program lifecycle execution.\n";
        return 0;
    }

    int failures = 0;
    ninfer::DeviceContext device(0);
    failures += test_plan_request_validations(device);
    failures += test_program_lifecycle(device);
    failures += test_program_memory_summary_kv_cache(device);
    failures += test_mtp_checkpoint_state_isolation(device);

    if (failures == 0) {
        std::cout << "All Program cold path contract tests passed cleanly.\n";
        return 0;
    }
    std::cerr << failures << " Program cold path contract test(s) failed.\n";
    return 1;
}
