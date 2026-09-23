#include "core/arena.h"
#include "core/device.h"
#include "core/layout.h"
#include "targets/qwen3_8_flash_next/impl/qsa_indexer_kernels.h"
#include "targets/qwen3_8_flash_next/impl/qsa_indexer_workspace.h"
#include "targets/qwen3_8_flash_next/impl/runtime_plan.h"
#include "targets/qwen3_8_flash_next/impl/runtime_state.h"
#include "targets/qwen3_8_flash_next/impl/text_decode.h"
#include "targets/qwen3_8_flash_next/impl/text_decode_state.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

bool cuda_unavailable(cudaError_t error) {
    return error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver;
}

int test_constants_and_math() {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;

    if (kAttentionKvBytesPerGroup != 6'291'456ULL) {
        std::cerr << "kAttentionKvBytesPerGroup mismatch: " << kAttentionKvBytesPerGroup << "\n";
        return 1;
    }
    if (kIndexerBlockKeysBytesPerGroup != 196'608ULL) {
        std::cerr << "kIndexerBlockKeysBytesPerGroup mismatch: " << kIndexerBlockKeysBytesPerGroup
                  << "\n";
        return 1;
    }
    if (kPhysicalStrideBytesPerGroup != 6'488'064ULL) {
        std::cerr << "kPhysicalStrideBytesPerGroup mismatch: " << kPhysicalStrideBytesPerGroup
                  << "\n";
        return 1;
    }

    if (kGdnConvBytesPerSlot != 2'211'840ULL) {
        std::cerr << "kGdnConvBytesPerSlot mismatch: " << kGdnConvBytesPerSlot << "\n";
        return 1;
    }
    if (kGdnSsmBytesPerSlot != 113'246'208ULL) {
        std::cerr << "kGdnSsmBytesPerSlot mismatch: " << kGdnSsmBytesPerSlot << "\n";
        return 1;
    }
    if (flash_next_gdn_ssm_bytes_per_slot(ninfer::GdnStateStorage::FP32) != 113'246'208ULL) {
        std::cerr << "flash_next_gdn_ssm_bytes_per_slot(FP32) mismatch\n";
        return 1;
    }
    if (flash_next_gdn_ssm_bytes_per_slot(ninfer::GdnStateStorage::BF16) != 56'623'104ULL) {
        std::cerr << "flash_next_gdn_ssm_bytes_per_slot(BF16) mismatch\n";
        return 1;
    }
    if (kPleConvBytesPerSlot != 184'320ULL) {
        std::cerr << "kPleConvBytesPerSlot mismatch: " << kPleConvBytesPerSlot << "\n";
        return 1;
    }
    if (kQsaRawKeysBytesPerSlot != 12'288ULL) {
        std::cerr << "kQsaRawKeysBytesPerSlot mismatch: " << kQsaRawKeysBytesPerSlot << "\n";
        return 1;
    }
    if (kQsaRawPositionsBytesPerSlot != 576ULL) {
        std::cerr << "kQsaRawPositionsBytesPerSlot mismatch: " << kQsaRawPositionsBytesPerSlot
                  << "\n";
        return 1;
    }
    if (kRecurrentStateBytesPerSlot != 115'655'232ULL) {
        std::cerr << "kRecurrentStateBytesPerSlot mismatch: " << kRecurrentStateBytesPerSlot
                  << "\n";
        return 1;
    }
    if (flash_next_recurrent_state_bytes_per_slot(ninfer::GdnStateStorage::FP32) != 115'655'232ULL) {
        std::cerr << "flash_next_recurrent_state_bytes_per_slot(FP32) mismatch\n";
        return 1;
    }
    if (flash_next_recurrent_state_bytes_per_slot(ninfer::GdnStateStorage::BF16) != 59'032'128ULL) {
        std::cerr << "flash_next_recurrent_state_bytes_per_slot(BF16) mismatch\n";
        return 1;
    }

    std::cout << "PASS: test_constants_and_math\n";
    return 0;
}

int test_decode_graph_buckets() {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;

    auto expect = [](std::uint32_t maximum_blocks, std::initializer_list<std::uint32_t> want,
                     const char* label) {
        const auto got = flash_next_decode_graph_buckets(maximum_blocks);
        if (got.count != want.size()) {
            std::cerr << "FAIL: " << label << " bucket count=" << got.count
                      << " expected=" << want.size() << "\n";
            return false;
        }
        std::uint32_t i = 0;
        for (std::uint32_t w : want) {
            if (got.blocks[i] != w) {
                std::cerr << "FAIL: " << label << " bucket[" << i << "]=" << got.blocks[i]
                          << " expected=" << w << "\n";
                return false;
            }
            ++i;
        }
        return true;
    };

    if (!expect(128, {128u}, "max_context=512 / 128 blocks")) { return 1; }
    if (!expect(512, {512u}, "max_context=2048 / 512 blocks")) { return 1; }
    if (!expect(1024, {512u, 1024u}, "max_context=4096")) { return 1; }
    if (!expect(2048, {512u, 2048u}, "max_context=8192")) { return 1; }
    if (!expect(8192, {512u, 2048u, 8192u}, "max_context=32768")) { return 1; }
    if (!expect(65536, {512u, 2048u, 8192u, 65536u}, "max_context=262144")) { return 1; }

    const auto b8192 = flash_next_decode_graph_buckets(2048);
    if (flash_next_decode_graph_select_bucket(b8192, 0) != 0 ||
        flash_next_decode_graph_select_bucket(b8192, 512) != 0 ||
        flash_next_decode_graph_select_bucket(b8192, 513) != 1 ||
        flash_next_decode_graph_select_bucket(b8192, 2048) != 1) {
        std::cerr << "FAIL: 8192-token bucket selection at the 512-block boundary\n";
        return 1;
    }
    if (flash_next_decode_graph_topology_class(1, 0) != 1u ||
        flash_next_decode_graph_topology_class(8, 1) != ((1u << 8) | 8u) ||
        flash_next_decode_graph_topology_class(1, 0) ==
            flash_next_decode_graph_topology_class(1, 1)) {
        std::cerr << "FAIL: topology_class must encode bucket_index explicitly\n";
        return 1;
    }

    FlashNextRuntimeConfig cfg8192{
        .max_concurrency = 4,
        .max_context     = 8192,
        .prefill_chunk   = 1024,
        .use_cuda_graph  = true,
    };
    const auto plan8192 = finalize_flash_next_runtime_plan(
        cfg8192, flash_next_capacity_curve(cfg8192).minimum_main_page_groups);
    if (plan8192.cuda_graph_allowance_bytes != 4ULL * 2ULL * 24ULL * 1024ULL * 1024ULL) {
        std::cerr << "FAIL: max_context=8192 allowance expected 192 MiB got "
                  << plan8192.cuda_graph_allowance_bytes << "\n";
        return 1;
    }

    FlashNextRuntimeConfig cfg64k{
        .max_concurrency = 1,
        .max_context     = 65536,
        .prefill_chunk   = 2048,
        .use_cuda_graph  = true,
    };
    const auto curve64k = flash_next_capacity_curve(cfg64k);
    if (curve64k.minimum_main_page_groups != 256 || curve64k.maximum_main_page_groups != 256) {
        std::cerr << "FAIL: 64k B=1 groups min/max expected 256 got " << curve64k.minimum_main_page_groups
                  << "/" << curve64k.maximum_main_page_groups << "\n";
        return 1;
    }
    const auto plan64k = finalize_flash_next_runtime_plan(cfg64k, curve64k.minimum_main_page_groups);
    const auto b64k    = flash_next_decode_graph_buckets(plan64k.maximum_blocks);
    if (plan64k.maximum_blocks != 16384 || plan64k.main_page_groups != 256 || b64k.count != 4 ||
        b64k.blocks[0] != 512 || b64k.blocks[1] != 2048 || b64k.blocks[2] != 8192 ||
        b64k.blocks[3] != 16384) {
        std::cerr << "FAIL: 64k plan blocks/groups/buckets mismatch blocks=" << plan64k.maximum_blocks
                  << " groups=" << plan64k.main_page_groups << " n_buckets=" << b64k.count << "\n";
        return 1;
    }
    if (plan64k.cuda_graph_allowance_bytes != 4ULL * 24ULL * 1024ULL * 1024ULL) {
        std::cerr << "FAIL: 64k graph allowance expected 96 MiB got "
                  << plan64k.cuda_graph_allowance_bytes << "\n";
        return 1;
    }
    if (plan64k.state_slots != 2) {
        std::cerr << "FAIL: 64k state_slots expected 2 got " << plan64k.state_slots << "\n";
        return 1;
    }
    const double kv_mib =
        static_cast<double>(plan64k.attention_kv_bytes + plan64k.indexer_block_keys_bytes) / 1048576.0;
    const double rec_mib  = static_cast<double>(plan64k.recurrent_state_bytes) / 1048576.0;
    const double graph_mib = static_cast<double>(plan64k.cuda_graph_allowance_bytes) / 1048576.0;
    const double ws_mib    = static_cast<double>(plan64k.workspace_bytes) / 1048576.0;
    const double tot_mib   = static_cast<double>(plan64k.total_device_bytes) / 1048576.0;
    std::cout << "G8 PLAN 64k predicted vs actual (B=1, 256 groups):\n";
    std::cout << "  KV        predicted 1584.0 MiB  actual " << kv_mib << " MiB\n";
    std::cout << "  recurrent predicted  220.6 MiB  actual " << rec_mib << " MiB\n";
    std::cout << "  graphs    predicted   96.0 MiB  actual " << graph_mib << " MiB\n";
    std::cout << "  workspace (not in predicted table) actual " << ws_mib << " MiB\n";
    std::cout << "  total_device_bytes actual " << tot_mib << " MiB (" << plan64k.total_device_bytes
              << " bytes)\n";
    std::cout << "  buckets {512, 2048, 8192, 16384} groups=" << plan64k.main_page_groups
              << " maximum_blocks=" << plan64k.maximum_blocks << " state_slots=" << plan64k.state_slots
              << "\n";

    FlashNextRuntimeConfig cfg256k{
        .max_concurrency = 1,
        .max_context     = 262144,
        .prefill_chunk   = 2048,
        .use_cuda_graph  = true,
    };
    const auto curve256k = flash_next_capacity_curve(cfg256k);
    if (curve256k.minimum_main_page_groups != 1024 || curve256k.maximum_main_page_groups != 1024) {
        std::cerr << "FAIL: 256k B=1 groups min/max expected 1024 got "
                  << curve256k.minimum_main_page_groups << "/" << curve256k.maximum_main_page_groups
                  << "\n";
        return 1;
    }
    const auto plan256k =
        finalize_flash_next_runtime_plan(cfg256k, curve256k.minimum_main_page_groups);
    const auto b256k = flash_next_decode_graph_buckets(plan256k.maximum_blocks);
    if (plan256k.maximum_blocks != 65536 || plan256k.main_page_groups != 1024 || b256k.count != 4 ||
        b256k.blocks[0] != 512 || b256k.blocks[1] != 2048 || b256k.blocks[2] != 8192 ||
        b256k.blocks[3] != 65536 || plan256k.attention_logical_pages != 4096 ||
        plan256k.indexer_logical_pages != 1024 || plan256k.resolved_tokens != 262144) {
        std::cerr << "FAIL: 256k plan blocks/groups/buckets mismatch blocks="
                  << plan256k.maximum_blocks << " groups=" << plan256k.main_page_groups
                  << " n_buckets=" << b256k.count << " att_log=" << plan256k.attention_logical_pages
                  << " idx_log=" << plan256k.indexer_logical_pages << "\n";
        return 1;
    }
    if (plan256k.cuda_graph_allowance_bytes != 4ULL * 24ULL * 1024ULL * 1024ULL) {
        std::cerr << "FAIL: 256k graph allowance expected 96 MiB got "
                  << plan256k.cuda_graph_allowance_bytes << "\n";
        return 1;
    }
    if (plan256k.state_slots != 2) {
        std::cerr << "FAIL: 256k state_slots expected 2 got " << plan256k.state_slots << "\n";
        return 1;
    }
    const std::int32_t tile_64k =
        flash_next_qsa_indexer_tile_size(static_cast<std::int32_t>(plan64k.maximum_blocks), 2048);
    const std::int32_t tile_256k =
        flash_next_qsa_indexer_tile_size(static_cast<std::int32_t>(plan256k.maximum_blocks), 2048);
    std::cout << "G14 tile_size(16384, 2048)=" << tile_64k
              << " tiles_per_2048_chunk=" << (2048 + tile_64k - 1) / tile_64k << "\n";
    std::cout << "G14 tile_size(65536, 2048)=" << tile_256k
              << " tiles_per_2048_chunk=" << (2048 + tile_256k - 1) / tile_256k << "\n";
    if (tile_64k != 1024 || tile_256k != 256) {
        std::cerr << "FAIL: expected tile_size 1024 at 64k and 256 at 256k\n";
        return 1;
    }
    const double kv256 =
        static_cast<double>(plan256k.attention_kv_bytes + plan256k.indexer_block_keys_bytes) /
        1048576.0;
    const double rec256  = static_cast<double>(plan256k.recurrent_state_bytes) / 1048576.0;
    const double graph256 = static_cast<double>(plan256k.cuda_graph_allowance_bytes) / 1048576.0;
    const double ws256    = static_cast<double>(plan256k.workspace_bytes) / 1048576.0;
    const double tot256   = static_cast<double>(plan256k.total_device_bytes) / 1048576.0;
    std::cout << "G14 PLAN 256k predicted vs actual (B=1, 1024 groups):\n";
    std::cout << "  KV        predicted 6336.0 MiB  actual " << kv256 << " MiB\n";
    std::cout << "  recurrent predicted  220.6 MiB  actual " << rec256 << " MiB\n";
    std::cout << "  graphs    predicted   96.0 MiB  actual " << graph256 << " MiB\n";
    std::cout << "  workspace (not in predicted table) actual " << ws256 << " MiB ("
              << plan256k.workspace_bytes << " bytes)\n";
    std::cout << "  total_device_bytes actual " << tot256 << " MiB (" << plan256k.total_device_bytes
              << " bytes)\n";
    std::cout << "  buckets {512, 2048, 8192, 65536} groups=" << plan256k.main_page_groups
              << " maximum_blocks=" << plan256k.maximum_blocks
              << " state_slots=" << plan256k.state_slots << "\n";
    std::cout << "  decode bucket at 64k-depth (16384 blocks)="
              << flash_next_decode_graph_select_bucket(b256k, 16384)
              << " envelope=" << b256k.blocks[flash_next_decode_graph_select_bucket(b256k, 16384)]
              << " (no 16384 slot under a 256k plan)\n";

    std::cout << "PASS: test_decode_graph_buckets\n";
    return 0;
}

int test_capacity_curve_and_finalize() {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;

    // 1. Boundary: context 128 -> 1 group per sequence
    FlashNextRuntimeConfig cfg1{
        .max_concurrency     = 1,
        .max_context         = 128,
        .state_slot_capacity = 0,
        .prefill_chunk       = 128,
    };
    auto curve1 = flash_next_capacity_curve(cfg1);
    if (curve1.minimum_main_page_groups != 1 || curve1.maximum_main_page_groups != 1 ||
        curve1.main_page_tokens != 256 ||
        curve1.bytes_per_additional_main_page_group != kPhysicalStrideBytesPerGroup) {
        std::cerr << "curve1 mismatch\n";
        return 1;
    }

    auto plan1 = finalize_flash_next_runtime_plan(cfg1, 1);
    if (plan1.main_page_groups != 1 || plan1.attention_physical_pages != 4 ||
        plan1.indexer_physical_pages != 1 || plan1.attention_logical_pages != 2 ||
        plan1.indexer_logical_pages != 1 || plan1.state_slots != 2 ||
        plan1.resolved_tokens != 256) {
        std::cerr << "plan1 (context 128) mismatch\n";
        return 1;
    }

    // 2. Boundary: context 262144 -> 1024 groups per sequence
    FlashNextRuntimeConfig cfg_max{
        .max_concurrency     = 1,
        .max_context         = 262'144,
        .state_slot_capacity = 0,
        .prefill_chunk       = 1024,
    };
    auto curve_max = flash_next_capacity_curve(cfg_max);
    if (curve_max.minimum_main_page_groups != 1024 || curve_max.maximum_main_page_groups != 1024) {
        std::cerr << "curve_max mismatch\n";
        return 1;
    }

    auto plan_max = finalize_flash_next_runtime_plan(cfg_max, 1024);
    if (plan_max.main_page_groups != 1024 || plan_max.attention_physical_pages != 4096 ||
        plan_max.indexer_physical_pages != 1024 || plan_max.attention_logical_pages != 4096 ||
        plan_max.indexer_logical_pages != 1024 || plan_max.maximum_blocks != 65'536 ||
        plan_max.resolved_tokens != 262'144) {
        std::cerr << "plan_max (context 262144) mismatch\n";
        return 1;
    }

    // 3. Concurrency 8 envelope with capacity resolution
    FlashNextRuntimeConfig cfg8{
        .max_concurrency     = 8,
        .max_context         = 2048,
        .state_slot_capacity = 0,
        .prefill_chunk       = 1024,
    };
    auto curve8 = flash_next_capacity_curve(cfg8);
    if (curve8.minimum_main_page_groups != 8 || curve8.maximum_main_page_groups != 64) {
        std::cerr << "curve8 concurrency mismatch\n";
        return 1;
    }

    auto plan8_max = finalize_flash_next_runtime_plan(cfg8, curve8.maximum_main_page_groups);
    if (plan8_max.main_page_groups != 64 || plan8_max.state_slots != 16 ||
        plan8_max.resolved_tokens != 64 * 256) {
        std::cerr << "plan8_max mismatch\n";
        return 1;
    }

    auto plan8_min = finalize_flash_next_runtime_plan(cfg8, curve8.minimum_main_page_groups);
    if (plan8_min.main_page_groups != 8 || plan8_min.resolved_tokens != 2048) {
        std::cerr << "plan8_min resolution mismatch\n";
        return 1;
    }
    if (plan8_min.total_device_bytes >= plan8_max.total_device_bytes) {
        std::cerr << "Resolution at min groups did not reduce allocation bytes\n";
        return 1;
    }

    // 4. Invalid configuration rejection
    auto reject_curve = [](FlashNextRuntimeConfig c) {
        try {
            (void)flash_next_capacity_curve(c);
            return false;
        } catch (const std::invalid_argument&) { return true; }
    };

    if (!reject_curve({.max_concurrency = 0, .max_context = 256, .state_slot_capacity = 0, .prefill_chunk = 128})) {
        std::cerr << "Failed to reject max_concurrency = 0\n";
        return 1;
    }
    if (!reject_curve({.max_concurrency = 9, .max_context = 256, .state_slot_capacity = 0, .prefill_chunk = 128})) {
        std::cerr << "Failed to reject max_concurrency = 9\n";
        return 1;
    }
    if (!reject_curve({.max_concurrency = 1, .max_context = 0, .state_slot_capacity = 0, .prefill_chunk = 128})) {
        std::cerr << "Failed to reject max_context = 0\n";
        return 1;
    }
    if (!reject_curve({.max_concurrency = 1, .max_context = 262'145, .state_slot_capacity = 0, .prefill_chunk = 128})) {
        std::cerr << "Failed to reject max_context > 262144\n";
        return 1;
    }
    if (!reject_curve({.max_concurrency = 4, .max_context = 256, .state_slot_capacity = 7, .continuation_capacity = 0, .prefill_chunk = 128})) {
        std::cerr << "Failed to reject state_slot_capacity < 2 * max_concurrency\n";
        return 1;
    }
    if (!reject_curve({.max_concurrency = 2, .max_context = 256, .state_slot_capacity = 10, .continuation_capacity = 8, .prefill_chunk = 128})) {
        std::cerr << "Failed to reject state_slot_capacity < 2 * max_concurrency + continuation_capacity\n";
        return 1;
    }
    if (!reject_curve({.max_concurrency = 4, .max_context = 256, .state_slot_capacity = 65, .prefill_chunk = 128})) {
        std::cerr << "Failed to reject state_slot_capacity > 64\n";
        return 1;
    }
    if (!reject_curve({.max_concurrency = 1, .max_context = 256, .state_slot_capacity = 0, .prefill_chunk = 0})) {
        std::cerr << "Failed to reject prefill_chunk = 0\n";
        return 1;
    }
    if (!reject_curve({.max_concurrency = 1, .max_context = 256, .state_slot_capacity = 0, .prefill_chunk = 64})) {
        std::cerr << "Failed to reject prefill_chunk not aligned to 128\n";
        return 1;
    }
    if (!reject_curve({.max_concurrency = 1, .max_context = 256, .state_slot_capacity = 0, .prefill_chunk = 512})) {
        std::cerr << "Failed to reject prefill_chunk > max_context\n";
        return 1;
    }

    // Reject selected_main_page_groups out of range
    try {
        (void)finalize_flash_next_runtime_plan(cfg8, 7); // < min_groups 8
        std::cerr << "Failed to reject selected_groups < min_groups\n";
        return 1;
    } catch (const std::invalid_argument&) {}

    try {
        (void)finalize_flash_next_runtime_plan(cfg8, 65); // > max_groups 64
        std::cerr << "Failed to reject selected_groups > max_groups\n";
        return 1;
    } catch (const std::invalid_argument&) {}

    // 4. Verify 64-slot (C=8 floor 16 + 48 continuations) exact recurrent state savings
    FlashNextRuntimeConfig cfg64_fp32{
        .max_concurrency       = 8,
        .max_context           = 2048,
        .state_slot_capacity   = 64,
        .continuation_capacity = 48,
        .prefill_chunk         = 1024,
        .gdn_state_storage     = ninfer::GdnStateStorage::FP32,
    };
    FlashNextRuntimeConfig cfg64_bf16 = cfg64_fp32;
    cfg64_bf16.gdn_state_storage      = ninfer::GdnStateStorage::BF16;

    const auto plan64_fp32 = finalize_flash_next_runtime_plan(cfg64_fp32, 64);
    const auto plan64_bf16 = finalize_flash_next_runtime_plan(cfg64_bf16, 64);
    if (plan64_fp32.state_slots != 64 || plan64_bf16.state_slots != 64) {
        std::cerr << "Expected 64 state slots\n";
        return 1;
    }
    const std::size_t rec_diff = plan64_fp32.recurrent_state_bytes - plan64_bf16.recurrent_state_bytes;
    if (rec_diff != 3'623'878'656ULL) {
        std::cerr << "Recurrent state savings mismatch: expected 3623878656, got " << rec_diff << "\n";
        return 1;
    }

    std::cout << "PASS: test_capacity_curve_and_finalize\n";
    return 0;
}

int test_runtime_allocation_and_slots(ninfer::DeviceContext& device) {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;

    for (const auto storage : {ninfer::GdnStateStorage::FP32, ninfer::GdnStateStorage::BF16}) {
        FlashNextRuntimeConfig cfg{
            .max_concurrency     = 2,
            .max_context         = 256,
            .state_slot_capacity = 4,
            .prefill_chunk       = 128,
            .gdn_state_storage   = storage,
        };
        const auto curve = flash_next_capacity_curve(cfg);
        auto plan        = finalize_flash_next_runtime_plan(cfg, curve.minimum_main_page_groups);

        // 1. Test plan tamper rejection (byte mismatch and curve mismatch)
        auto forged_plan_bytes = plan;
        forged_plan_bytes.total_device_bytes -= 1024;
        try {
            FlashNextRuntimeAllocation bad_alloc(forged_plan_bytes);
            std::cerr << "Failed to reject tampered bytes in plan\n";
            return 1;
        } catch (const std::invalid_argument&) {}

        auto forged_plan_curve = plan;
        forged_plan_curve.capacity_curve.minimum_main_page_groups += 1;
        try {
            FlashNextRuntimeAllocation bad_alloc(forged_plan_curve);
            std::cerr << "Failed to reject tampered curve in plan\n";
            return 1;
        } catch (const std::invalid_argument&) {}

        FlashNextRuntimeAllocation alloc(plan);

        // 2. Initialize device slots and zero persistent recurrent state
        alloc.initialize(device.stream);
        device.synchronize();

        // Verify representative state values on device are zeroed
        std::vector<std::uint16_t> ple_host(10'240 * 9 * 4, 0x1234);
        CUDA_CHECK(cudaMemcpy(ple_host.data(), alloc.state_view().ple_convolution_states.data,
                              ple_host.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost));
        for (std::size_t i = 0; i < ple_host.size(); ++i) {
            if (ple_host[i] != 0) {
                std::cerr << "PLE state not zeroed on initialization at index " << i << ": "
                          << ple_host[i] << "\n";
                return 1;
            }
        }

        // Verify SSM states have the expected DType
        const ninfer::DType expected_dtype = (storage == ninfer::GdnStateStorage::BF16)
                                                 ? ninfer::DType::BF16
                                                 : ninfer::DType::FP32;
        if (alloc.state_view().gdn_ssm_states[0].dtype != expected_dtype) {
            std::cerr << "SSM state tensor dtype mismatch\n";
            return 1;
        }

        // 3. Verify state view passes validation
        try {
            validate_flash_next_decode_state(alloc.state_view(), plan.state_slots);
        } catch (const std::exception& e) {
            std::cerr << "State view failed validation: " << e.what() << "\n";
            return 1;
        }

        // 4. Verify shared block tables across all 12 layers
        for (std::size_t i = 1; i < kFullAttentionLayers; ++i) {
            if (alloc.state_view().qsa_attention_caches[i].block_tables.data !=
                alloc.state_view().qsa_attention_caches[0].block_tables.data) {
                std::cerr << "Attention block tables not shared across layers\n";
                return 1;
            }
            if (alloc.state_view().qsa_indexer_caches[i].block_tables.data !=
                alloc.state_view().qsa_indexer_caches[0].block_tables.data) {
                std::cerr << "Indexer block tables not shared across layers\n";
                return 1;
            }
        }

        // 5. Verify initial device slots match host (source=0, destination=1 for row 0)
        std::vector<std::int32_t> dev_src(2), dev_dst(2);
        CUDA_CHECK(cudaMemcpy(dev_src.data(), alloc.round_tensors().source_slots.data,
                              2 * sizeof(std::int32_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(dev_dst.data(), alloc.round_tensors().destination_slots.data,
                              2 * sizeof(std::int32_t), cudaMemcpyDeviceToHost));
        if (dev_src[0] != 0 || dev_dst[0] != 1 || dev_src[1] != 2 || dev_dst[1] != 3) {
            std::cerr << "Initial device slot tensors mismatch\n";
            return 1;
        }

        // 6. Test slot transactional commit and address stability
        const void* initial_ple_data    = alloc.state_view().ple_convolution_states.data;
        const void* initial_logits_data = alloc.round_tensors().logits.data;

        alloc.commit_row_slot(0, device.stream);
        device.synchronize();

        const auto new_src = alloc.current_source_slot(0);
        const auto new_dst = alloc.current_destination_slot(0);
        if (new_src != 1 || new_dst != 0) {
            std::cerr << "Commit row slot failed host swap: expected src=1 dst=0, got src=" << new_src
                      << " dst=" << new_dst << "\n";
            return 1;
        }

        CUDA_CHECK(cudaMemcpy(dev_src.data(), alloc.round_tensors().source_slots.data,
                              2 * sizeof(std::int32_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(dev_dst.data(), alloc.round_tensors().destination_slots.data,
                              2 * sizeof(std::int32_t), cudaMemcpyDeviceToHost));
        if (dev_src[0] != 1 || dev_dst[0] != 0) {
            std::cerr << "Device slot tensors not updated after commit\n";
            return 1;
        }

        if (alloc.state_view().ple_convolution_states.data != initial_ple_data ||
            alloc.round_tensors().logits.data != initial_logits_data) {
            std::cerr << "Device tensor pointers changed after slot commit\n";
            return 1;
        }
    }

    std::cout << "PASS: test_runtime_allocation_and_slots\n";
    return 0;
}

int test_64k_allocation(ninfer::DeviceContext& device) {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    FlashNextRuntimeConfig cfg{
        .max_concurrency = 1,
        .max_context     = 65536,
        .prefill_chunk   = 2048,
        .use_cuda_graph  = true,
    };
    const auto curve = flash_next_capacity_curve(cfg);
    auto plan        = finalize_flash_next_runtime_plan(cfg, curve.minimum_main_page_groups);
    FlashNextRuntimeAllocation alloc(plan);
    alloc.initialize(device.stream);
    device.synchronize();
    std::cout << "G8 64k allocation succeeded total_device_bytes=" << plan.total_device_bytes
              << " workspace_bytes=" << plan.workspace_bytes << "\n";
    if (plan.workspace_bytes >= 2ULL * 1024ULL * 1024ULL * 1024ULL) {
        std::cerr << "ANOMALY: workspace_bytes crosses 2 GiB: " << plan.workspace_bytes << "\n";
        return 1;
    }
    std::cout << "PASS: test_64k_allocation\n";
    return 0;
}

bool g14_fits_i32(std::int64_t value, const char* line) {
    const bool ok = value >= static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::min()) &&
                    value <= static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max());
    std::cout << "  " << line << " value=" << value << (ok ? " fits int32\n" : " DOES NOT FIT int32\n");
    return ok;
}

int test_g14_int32_audit() {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    constexpr std::int32_t kN    = 65536;
    constexpr std::int32_t kT    = 2048;
    const std::int32_t tile      = flash_next_qsa_indexer_tile_size(kN, kT);
    const std::int64_t items_tile = static_cast<std::int64_t>(kN) * tile;
    const std::int64_t items_uncapped =
        static_cast<std::int64_t>(kN) * kT; // if tile_size did not shrink
    std::cout << "G14 INT32 AUDIT at maximum_blocks=65536 tile=" << tile << " tokens=" << kT << "\n";
    bool ok = true;
    ok = g14_fits_i32(items_tile,
                      "qsa_indexer_kernels.cu:748 const int items = active_blocks * current_tile") &&
         ok;
    ok = g14_fits_i32(items_uncapped,
                      "uncapped items = 65536 * 2048 (tile_size prevents this launch shape)") &&
         ok;
    ok = g14_fits_i32(static_cast<std::int64_t>(tile) * kN,
                      "qsa_indexer_kernels.cu:156 offsets[index] = index * active_blocks "
                      "(index<=tile)") &&
         ok;
    ok = g14_fits_i32(static_cast<std::int64_t>(kN) * 1,
                      "qsa_indexer_kernels.cu:154 decode items = active_blocks * batch_size B=1") &&
         ok;
    ok = g14_fits_i32(static_cast<std::int64_t>(kN) * 8,
                      "decode items at B=8 envelope") &&
         ok;
    ok = g14_fits_i32(262143, "token_index up to max_context-1") && ok;
    ok = g14_fits_i32(0 * 1024 + 1023,
                      "indexer block_tables[table_row * logical_pages + logical_page] B=1") &&
         ok;
    ok = g14_fits_i32(0 * 4096 + 4095,
                      "attention block_tables[table_row * logical_pages + logical_page] B=1") &&
         ok;
    ok = g14_fits_i32(1023 * 4 + 3, "lane_ledger.cpp log_att_page = log_group * 4 + s") && ok;
    ok = g14_fits_i32(static_cast<std::int64_t>(tile - 1) * kN + (kN - 1),
                      "score write tok * active_blocks + blk (stored as int64 at "
                      "qsa_indexer_kernels.cu:665)") &&
         ok;
    ok = g14_fits_i32(static_cast<std::int64_t>(kN) * tile * 4,
                      "scores FP32 bytes = N * tile * 4 (size_t path, quoted as 32-bit risk)") &&
         ok;
    const std::int64_t packed_bytes = static_cast<std::int64_t>(kN) * tile * 8;
    std::cout << "  packed_keys I64 bytes=" << packed_bytes
              << (packed_bytes >= (1LL << 31) ? " crosses 2 GiB in a 32-bit intermediate\n"
                                              : " under 2 GiB\n");
    if (packed_bytes >= (1LL << 31)) { ok = false; }
    if (tile != 256 || items_tile != 16777216LL || items_uncapped != 134217728LL) {
        std::cerr << "FAIL: G14 int32 fixture tile/items mismatch tile=" << tile
                  << " items_tile=" << items_tile << "\n";
        return 1;
    }
    if (!ok) {
        std::cerr << "FAIL: G14 int32 audit found a value that does not fit int32\n";
        return 1;
    }
    std::cout << "PASS: test_g14_int32_audit\n";
    return 0;
}

void g14_dump_tensor(const char* name, const ninfer::Tensor& tensor) {
    std::cout << "  " << name << " ne=[" << tensor.ne[0] << "," << tensor.ne[1] << "," << tensor.ne[2]
              << "," << tensor.ne[3] << "] bytes=" << tensor.bytes() << "\n";
}

int test_g14_workspace_dump(ninfer::DeviceContext& device) {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    (void)device;
    constexpr std::int32_t kN = 65536;
    constexpr std::int32_t kT = 2048;
    const std::int32_t tile   = flash_next_qsa_indexer_tile_size(kN, kT);
    const std::size_t sort_temp =
        flash_next_qsa_indexer_sort_temp_bytes(kN, tile);
    ninfer::WorkspaceLayoutBuilder layout;
    auto scratch =
        allocate_flash_next_qsa_indexer_workspace(layout, kN, kT, tile, sort_temp);
    std::cout << "G14 INDEXER WORKSPACE N=65536 tokens=2048 tile=" << tile << "\n";
    g14_dump_tensor("projected", scratch.projected);
    g14_dump_tensor("query", scratch.query);
    g14_dump_tensor("scores", scratch.scores);
    g14_dump_tensor("sorted_scores", scratch.sorted_scores);
    g14_dump_tensor("ids", scratch.ids);
    g14_dump_tensor("sorted_ids", scratch.sorted_ids);
    g14_dump_tensor("packed_keys", scratch.packed_keys);
    g14_dump_tensor("packed_selected", scratch.packed_selected);
    g14_dump_tensor("topk_ids", scratch.topk_ids);
    g14_dump_tensor("offsets", scratch.offsets);
    std::cout << "  sort_temp bytes=" << sort_temp << " span=" << scratch.sort_temp.bytes << "\n";
    std::cout << "  indexer_layout_peak=" << layout.peak_bytes(256) << "\n";

    const std::size_t prefill_ws = flash_next_text_prefill_workspace_capacity_bytes(kN, kT);
    const std::size_t decode_ws  = flash_next_text_decode_workspace_capacity_bytes(kN, 1);
    std::cout << "G14 prefill workspace capacity T=2048 N=65536: " << prefill_ws << " bytes ("
              << (static_cast<double>(prefill_ws) / 1048576.0) << " MiB)\n";
    std::cout << "G14 decode workspace capacity B=1 N=65536: " << decode_ws << " bytes ("
              << (static_cast<double>(decode_ws) / 1048576.0) << " MiB)\n";
    if (prefill_ws >= 2ULL * 1024ULL * 1024ULL * 1024ULL ||
        decode_ws >= 2ULL * 1024ULL * 1024ULL * 1024ULL) {
        std::cerr << "ANOMALY: workspace crosses 2 GiB\n";
        return 1;
    }
    const std::int32_t tile64 = flash_next_qsa_indexer_tile_size(16384, kT);
    const std::size_t prefill64 =
        flash_next_text_prefill_workspace_capacity_bytes(16384, kT);
    std::cout << "G14 64k comparison tile=" << tile64 << " prefill_ws=" << prefill64 << " bytes ("
              << (static_cast<double>(prefill64) / 1048576.0) << " MiB)\n";
    std::cout << "PASS: test_g14_workspace_dump\n";
    return 0;
}

int test_mtp_checkpoint_state(ninfer::DeviceContext& device) {
    using namespace ninfer;
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    for (const auto storage : {KvCacheStorage::BFloat16, KvCacheStorage::Fp8E4M3Row256}) {
        FlashNextRuntimeConfig cfg{
            .max_concurrency = 2,
            .max_context = 512,
            .continuation_capacity = 1,
            .prefill_chunk = 128,
            .speculative_draft_tokens = 4,
            .use_cuda_graph = false,
            .kv_cache = storage,
        };
        auto curve = flash_next_capacity_curve(cfg);
        auto plan = finalize_flash_next_runtime_plan(cfg, curve.maximum_main_page_groups);
        const auto expected_stride = flash_next_physical_stride_bytes_per_group(storage) / 12 * 13;
        if (curve.bytes_per_additional_main_page_group != expected_stride ||
            plan.attention_kv_bytes + plan.indexer_block_keys_bytes !=
                expected_stride * plan.main_page_groups) {
            std::cerr << "FAIL: MTP KV is missing from the capacity reservation\n";
            return 1;
        }
        FlashNextRuntimeAllocation alloc(plan);
        alloc.initialize(device.stream);
        auto& state = alloc.state_view();
        auto& indexer = state.qsa_indexer_caches[kFullAttentionLayers];
        auto& attention = state.qsa_attention_caches[kFullAttentionLayers];
        if (indexer.block_tables.data != state.qsa_indexer_caches[0].block_tables.data ||
            attention.block_tables.data != state.qsa_attention_caches[0].block_tables.data) {
            std::cerr << "FAIL: MTP must follow the lane's physical page ownership\n";
            return 1;
        }
        const auto checkpoint = plan.state_slots - 1;
        Tensor keys = indexer.raw_keys.slice(2, 0, 1);
        Tensor positions = indexer.raw_positions.slice(2, 0, 1);
        Tensor hidden = state.mtp_backbone_hidden.slice(1, 0, 1);
        Tensor backbone_positions = state.mtp_backbone_positions.slice(1, 0, 1);
        for (auto tensor : {keys, positions, hidden, backbone_positions}) {
            CUDA_CHECK(cudaMemsetAsync(tensor.data, 0x35, tensor.bytes(), device.stream));
        }
        alloc.copy_state_slot(0, checkpoint, device.stream);
        alloc.zero_slot(0, device.stream);
        device.synchronize();
        for (auto tensor : {indexer.raw_keys.slice(2, checkpoint, 1),
                            indexer.raw_positions.slice(2, checkpoint, 1),
                            state.mtp_backbone_hidden.slice(1, checkpoint, 1),
                            state.mtp_backbone_positions.slice(1, checkpoint, 1)}) {
            std::vector<std::uint8_t> bytes(tensor.bytes());
            CUDA_CHECK(cudaMemcpy(bytes.data(), tensor.data, bytes.size(), cudaMemcpyDeviceToHost));
            if (!std::all_of(bytes.begin(), bytes.end(), [](auto b) { return b == 0x35; })) {
                std::cerr << "FAIL: checkpoint lost MTP raw keys, positions, or backbone hidden\n";
                return 1;
            }
        }
        alloc.copy_state_slot(checkpoint, 0, device.stream);
        alloc.zero_slot(checkpoint, device.stream);
        device.synchronize();
        std::vector<std::uint8_t> restored(hidden.bytes());
        CUDA_CHECK(cudaMemcpy(restored.data(), hidden.data, restored.size(), cudaMemcpyDeviceToHost));
        if (!std::all_of(restored.begin(), restored.end(), [](auto b) { return b == 0x35; })) {
            std::cerr << "FAIL: restored MTP state aliases the released checkpoint\n";
            return 1;
        }
    }
    std::cout << "PASS: test_mtp_checkpoint_state\n";
    return 0;
}

int test_256k_allocation(ninfer::DeviceContext& device) {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    FlashNextRuntimeConfig cfg{
        .max_concurrency = 1,
        .max_context     = 262144,
        .prefill_chunk   = 2048,
        .use_cuda_graph  = true,
    };
    const auto curve = flash_next_capacity_curve(cfg);
    auto plan        = finalize_flash_next_runtime_plan(cfg, curve.minimum_main_page_groups);
    std::cout << "G14 256k allocation begin total_device_bytes=" << plan.total_device_bytes
              << " workspace_bytes=" << plan.workspace_bytes << "\n";
    FlashNextRuntimeAllocation alloc(plan);
    alloc.initialize(device.stream);
    device.synchronize();
    const std::size_t peak = alloc.workspace().peak_used();
    const std::size_t cap  = alloc.workspace().capacity();
    std::cout << "G14 256k allocation succeeded total_device_bytes=" << plan.total_device_bytes
              << " workspace peak_used=" << peak << " / " << cap << "\n";
    if (plan.workspace_bytes >= 2ULL * 1024ULL * 1024ULL * 1024ULL) {
        std::cerr << "ANOMALY: workspace_bytes crosses 2 GiB: " << plan.workspace_bytes << "\n";
        return 1;
    }
    if (cap < plan.workspace_bytes) {
        std::cerr << "FAIL: allocated workspace capacity " << cap << " < plan " << plan.workspace_bytes
                  << "\n";
        return 1;
    }
    std::cout << "PASS: test_256k_allocation\n";
    return 0;
}

} // namespace

int main() {
    // 1. CPU tests run unconditionally without requiring a CUDA device
    if (test_constants_and_math() != 0) return 1;
    if (test_decode_graph_buckets() != 0) return 1;
    if (test_capacity_curve_and_finalize() != 0) return 1;
    if (test_g14_int32_audit() != 0) return 1;

    // 2. CUDA device tests run only when a CUDA device is available
    int device_count              = 0;
    const cudaError_t count_error = cudaGetDeviceCount(&device_count);
    if (cuda_unavailable(count_error) || device_count == 0) {
        std::cout << "SKIP: CUDA device tests (no usable device)\n";
        return 0;
    }
    CUDA_CHECK(count_error);

    ninfer::DeviceContext device(0);
    if (test_mtp_checkpoint_state(device) != 0) return 1;
    if (test_runtime_allocation_and_slots(device) != 0) return 1;
    if (test_64k_allocation(device) != 0) return 1;
    if (test_g14_workspace_dump(device) != 0) return 1;
    if (test_256k_allocation(device) != 0) return 1;

    std::cout << "OK Flash-Next Runtime Foundation\n";
    return 0;
}
