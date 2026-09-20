#include "synthetic_fixture.h"

#include "artifact/reader.h"
#include "targets/qwen3_8_flash_next/impl/load/bindings.h"
#include "targets/qwen3_8_flash_next/impl/load/loader.h"
#include "targets/qwen3_8_flash_next/impl/ple_index.h"
#include "targets/qwen3_8_flash_next/impl/runtime_plan.h"
#include "tools/reference/qwen3_8_flash_next/options.h"

#include <array>
#include <cstdlib>
#include <filesystem>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace ninfer::targets::qwen3_8_flash_next::detail;
using LoadedModel = StandaloneLoadedModel;

int test_identity_validation() {
    using ninfer::artifact::ArtifactError;
    using ninfer::artifact::ArtifactIdentity;

    // 1. Valid exact identity
    ArtifactIdentity valid_id{
        .model_id   = "qwen3.8-flash-next",
        .weights_id = "mixed-nvfp4-fp8-ple-int4",
    };
    try {
        validate_identity(valid_id);
    } catch (const std::exception& ex) {
        std::cerr << "Unexpected failure validating exact identity: " << ex.what() << "\n";
        return 1;
    }

    // 2. Invalid model_id
    ArtifactIdentity bad_model_id{
        .model_id   = "qwen3.6-27b",
        .weights_id = "mixed-nvfp4-fp8-ple-int4",
    };
    bool caught_bad_model = false;
    try {
        validate_identity(bad_model_id);
    } catch (const ArtifactError&) { caught_bad_model = true; }
    if (!caught_bad_model) {
        std::cerr << "Failed to reject invalid model_id\n";
        return 1;
    }

    // 3. Invalid weights_id
    ArtifactIdentity bad_weights_id{
        .model_id   = "qwen3.8-flash-next",
        .weights_id = "groupwise-int",
    };
    bool caught_bad_weights = false;
    try {
        validate_identity(bad_weights_id);
    } catch (const ArtifactError&) { caught_bad_weights = true; }
    if (!caught_bad_weights) {
        std::cerr << "Failed to reject invalid weights_id\n";
        return 1;
    }

    // 4. Empty identity
    ArtifactIdentity empty_id{"", ""};
    bool caught_empty = false;
    try {
        validate_identity(empty_id);
    } catch (const ArtifactError&) { caught_empty = true; }
    if (!caught_empty) {
        std::cerr << "Failed to reject empty identity\n";
        return 1;
    }

    std::cout << "PASS: test_identity_validation\n";
    return 0;
}

int test_ple_metadata_observable_behavior() {
    // Verify observable runtime PLE metadata properties and index computation
    if (kPleIndexMetadata.multipliers.size() != 3 || kPleIndexMetadata.multipliers[0] <= 0 ||
        kPleIndexMetadata.multipliers[1] <= 0 || kPleIndexMetadata.multipliers[2] <= 0) {
        std::cerr << "PLE multipliers metadata invalid\n";
        return 1;
    }

    if (kPleIndexMetadata.head_offsets.size() != 16 ||
        kPleIndexMetadata.head_vocab_sizes.size() != 16) {
        std::cerr << "PLE head metadata size mismatch\n";
        return 1;
    }

    // Monotonicity and positive vocabulary size checks
    for (std::size_t i = 0; i < 16; ++i) {
        if (kPleIndexMetadata.head_vocab_sizes[i] <= 0) {
            std::cerr << "PLE head vocab size non-positive at " << i << "\n";
            return 1;
        }
        if (i > 0 && kPleIndexMetadata.head_offsets[i] <= kPleIndexMetadata.head_offsets[i - 1]) {
            std::cerr << "PLE head offsets not strictly monotonic at " << i << "\n";
            return 1;
        }
    }

    // Verify index generation for token
    PleTokenHistory history;
    const auto indices0 = ple_indices(kPleIndexMetadata, history, 42);
    for (std::size_t i = 0; i < 16; ++i) {
        const auto offset = kPleIndexMetadata.head_offsets[i];
        const auto vocab  = kPleIndexMetadata.head_vocab_sizes[i];
        if (indices0[i] < offset || indices0[i] >= offset + vocab) {
            std::cerr << "PLE index out of head range at " << i << ": " << indices0[i] << "\n";
            return 1;
        }
    }

    std::cout << "PASS: test_ple_metadata_observable_behavior\n";
    return 0;
}

int test_preflight_memory_accounting() {
    FlashNextRuntimeConfig cfg{
        .max_concurrency     = 4,
        .max_context         = 4096,
        .state_slot_capacity = 8,
    };

    const auto curve = flash_next_capacity_curve(cfg);
    if (curve.main_page_tokens != 256) {
        std::cerr << "Main page tokens mismatch: expected 256 got " << curve.main_page_tokens
                  << "\n";
        return 1;
    }

    // 4096 tokens / 256 = 16 groups * 4 concurrency = 64 maximum groups
    if (curve.maximum_main_page_groups != 64) {
        std::cerr << "Maximum main page groups expected 64, got " << curve.maximum_main_page_groups
                  << "\n";
        return 1;
    }

    const auto plan = finalize_flash_next_runtime_plan(cfg, curve.maximum_main_page_groups);
    if (plan.resolved_tokens != 64 * 256) {
        std::cerr << "Resolved tokens mismatch: expected 16384 got " << plan.resolved_tokens
                  << "\n";
        return 1;
    }

    // Check exact affine memory stride calculation
    if (curve.bytes_per_additional_main_page_group != kPhysicalStrideBytesPerGroup) {
        std::cerr << "Bytes per additional group mismatch: expected "
                  << kPhysicalStrideBytesPerGroup << " got "
                  << curve.bytes_per_additional_main_page_group << "\n";
        return 1;
    }
    if (kPhysicalStrideBytesPerGroup != 6'488'064ULL) {
        std::cerr << "kPhysicalStrideBytesPerGroup constant mismatch: expected 6488064 got "
                  << kPhysicalStrideBytesPerGroup << "\n";
        return 1;
    }

    // Out of bounds page group rejection
    try {
        (void)finalize_flash_next_runtime_plan(cfg, curve.minimum_main_page_groups - 1);
        std::cerr << "Failed to reject under-minimum main page groups\n";
        return 1;
    } catch (const std::invalid_argument&) {}

    try {
        (void)finalize_flash_next_runtime_plan(cfg, curve.maximum_main_page_groups + 1);
        std::cerr << "Failed to reject over-maximum main page groups\n";
        return 1;
    } catch (const std::invalid_argument&) {}

    std::cout << "PASS: test_preflight_memory_accounting\n";
    return 0;
}

int test_options_parser_validation() {
    // 1. Valid invocation path without materialization
    const std::vector<std::string_view> valid_args = {
        "--model", "model.ninfer", "--preflight", "--max-context", "2048", "--max-concurrency", "2",
    };
    try {
        const auto opts = parse_reference_tool_options(valid_args);
        if (opts.model_path != "model.ninfer" || opts.mode != "preflight" ||
            opts.max_context != 2048 || opts.max_concurrency != 2 || opts.state_slots != 0) {
            std::cerr << "Parsed options mismatch on valid args\n";
            return 1;
        }
    } catch (const std::exception& ex) {
        std::cerr << "Unexpected failure parsing valid args: " << ex.what() << "\n";
        return 1;
    }

    const std::vector<std::string_view> full_args = {
        "--model", "model.ninfer", "--materialize-full", "--max-context", "4096",
    };
    try {
        const auto opts = parse_reference_tool_options(full_args);
        if (opts.model_path != "model.ninfer" || opts.mode != "materialize-full" ||
            opts.max_context != 4096) {
            std::cerr << "Parsed options mismatch on materialize-full args\n";
            return 1;
        }
    } catch (const std::exception& ex) {
        std::cerr << "Unexpected failure parsing materialize-full args: " << ex.what() << "\n";
        return 1;
    }

    const std::vector<std::string_view> vision_args = {
        "--model", "model.ninfer", "--materialize-vision", "--max-context", "4096",
    };
    try {
        const auto opts = parse_reference_tool_options(vision_args);
        if (opts.model_path != "model.ninfer" || opts.mode != "materialize-vision" ||
            opts.max_context != 4096) {
            std::cerr << "Parsed options mismatch on materialize-vision args\n";
            return 1;
        }
    } catch (const std::exception& ex) {
        std::cerr << "Unexpected failure parsing materialize-vision args: " << ex.what() << "\n";
        return 1;
    }

    // Helper lambda to test that invalid arguments throw std::invalid_argument
    const auto assert_throws = [](std::initializer_list<std::string_view> args,
                                  std::string_view label) -> bool {
        try {
            std::vector<std::string_view> vec(args);
            (void)parse_reference_tool_options(vec);
            std::cerr << "Failed to reject " << label << "\n";
            return false;
        } catch (const std::invalid_argument&) { return true; }
    };

    // 2. Negative unsigned input
    if (!assert_throws({"--model", "m.ninfer", "--max-context", "-1"}, "negative max-context"))
        return 1;
    if (!assert_throws({"--model", "m.ninfer", "--max-concurrency", "-2"},
                       "negative max-concurrency"))
        return 1;
    if (!assert_throws({"--model", "m.ninfer", "--page-groups", "-10"}, "negative page-groups"))
        return 1;
    if (!assert_throws({"--model", "m.ninfer", "--state-slots", "-4"}, "negative state-slots"))
        return 1;

    // 3. Trailing junk
    if (!assert_throws({"--model", "m.ninfer", "--max-context", "4096abc"},
                       "trailing junk in context"))
        return 1;
    if (!assert_throws({"--model", "m.ninfer", "--token-id", "100xyz"},
                       "trailing junk in token-id"))
        return 1;

    // 4. Integer overflow
    if (!assert_throws({"--model", "m.ninfer", "--max-context", "999999999999999999999"},
                       "overflow in context"))
        return 1;

    // 5. Out-of-range concurrency
    if (!assert_throws({"--model", "m.ninfer", "--max-concurrency", "0"}, "concurrency 0"))
        return 1;
    if (!assert_throws({"--model", "m.ninfer", "--max-concurrency", "9"}, "concurrency 9"))
        return 1;

    // 6. Out-of-range token ID [0, 248320)
    if (!assert_throws({"--model", "m.ninfer", "--token-id", "-1"}, "negative token-id")) return 1;
    if (!assert_throws({"--model", "m.ninfer", "--token-id", "248320"}, "token-id == vocab_size"))
        return 1;
    if (!assert_throws({"--model", "m.ninfer", "--token-id", "300000"}, "token-id > vocab_size"))
        return 1;

    // 7. Missing value
    if (!assert_throws({"--model"}, "missing model path value")) return 1;
    if (!assert_throws({"--model", "m.ninfer", "--max-context"}, "missing context value")) return 1;

    // 8. Unknown option
    if (!assert_throws({"--model", "m.ninfer", "--invalid-option"}, "unknown option")) return 1;

    // 9. Missing required model path
    if (!assert_throws({"--preflight"}, "missing model path")) return 1;

    std::cout << "PASS: test_options_parser_validation\n";
    return 0;
}

int test_real_artifact_preflight_if_available() {
    try {
        const char* env_path = std::getenv("NINFER_QWEN3_8_FLASH_NEXT_WEIGHTS");
        std::filesystem::path path =
            env_path != nullptr && *env_path != '\0'
                ? std::filesystem::path(env_path)
                : std::filesystem::path(
                      "" /* real-artifact cases run only when NINFER_WEIGHTS is set explicitly */);

        if (!std::filesystem::is_regular_file(path)) {
            std::cout << "SKIP: Real artifact preflight (artifact not present at " << path << ")\n";
            return 0;
        }

        FlashNextRuntimeConfig cfg{
            .max_concurrency     = 1,
            .max_context         = 4096,
            .state_slot_capacity = 2,
        };

        const auto report = preflight_text_file(path, cfg);

        // Exact delivered contract verification
        if (report.identity.model_id != kExpectedModelId ||
            report.identity.weights_id != kExpectedWeightsId) {
            std::cerr << "Report identity mismatch: " << report.identity.model_id << "/"
                      << report.identity.weights_id << "\n";
            return 1;
        }
        if (report.file_bytes != 113'298'397'952ULL) {
            std::cerr << "Preflight file_bytes mismatch: expected 113298397952 got "
                      << report.file_bytes << "\n";
            return 1;
        }
        if (report.planned_device_weights_bytes != 75'172'951'040ULL) {
            std::cerr << "Preflight planned_device_weights_bytes mismatch: expected 75172951040 got "
                      << report.planned_device_weights_bytes << "\n";
            return 1;
        }
        if (report.planned_device_tensors_count != 1067) {
            std::cerr << "Preflight planned_device_tensors_count mismatch: expected 1067 got "
                      << report.planned_device_tensors_count << "\n";
            return 1;
        }
        if (report.planned_retained_resources_count != 6) {
            std::cerr << "Preflight planned_retained_resources_count mismatch: expected 6 got "
                      << report.planned_retained_resources_count << "\n";
            return 1;
        }
        if (report.planned_mapped_tensors_count != 131) { // 128 shards + 3 embedding metadata tensors
            std::cerr << "Preflight planned_mapped_tensors_count mismatch: expected 131 got "
                      << report.planned_mapped_tensors_count << "\n";
            return 1;
        }
        if (report.runtime_plan.workspace_bytes != 319'055'616ULL ||
            report.runtime_plan.attention_kv_bytes != 100'663'296ULL ||
            report.runtime_plan.indexer_block_keys_bytes != 3'145'728ULL ||
            report.runtime_plan.recurrent_state_bytes != 231'312'384ULL) {
            std::cerr << "Preflight runtime plan sub-allocations mismatch: workspace_bytes="
                      << report.runtime_plan.workspace_bytes << " expected 319055616\n";
            return 1;
        }
        const std::uint64_t expected_graph_bytes =
            flash_next_cuda_graph_enabled(true) ? 50'331'648ULL : 0ULL;
        const std::uint64_t expected_runtime_total =
            654'705'920ULL + expected_graph_bytes;
        if (report.runtime_plan.cuda_graph_allowance_bytes != expected_graph_bytes ||
            report.runtime_plan.total_device_bytes != expected_runtime_total) {
            std::cerr << "Preflight runtime total_device_bytes mismatch: graph="
                      << report.runtime_plan.cuda_graph_allowance_bytes
                      << " total=" << report.runtime_plan.total_device_bytes << "\n";
            return 1;
        }

        const auto& ledger = report.vram_ledger;
        constexpr std::uint64_t kExpertLayerBytes = 1'415'581'696ULL;
        constexpr std::uint64_t kAllTextExpertsBytes = kExpertLayerBytes * 48ULL;
        constexpr std::uint64_t kEmbeddingBytes = 1'271'398'400ULL;
        if (ledger.planned_device_weight_arena_bytes != report.planned_device_weights_bytes ||
            ledger.routed_expert_layers != 48 ||
            ledger.routed_expert_layer_payload_bytes != kExpertLayerBytes ||
            ledger.routed_expert_payload_bytes != kAllTextExpertsBytes ||
            ledger.token_embedding_payload_bytes != kEmbeddingBytes ||
            ledger.output_head_payload_bytes != kEmbeddingBytes ||
            ledger.runtime_plan_device_bytes != report.runtime_plan.total_device_bytes ||
            ledger.cuda_graph_bytes != report.runtime_plan.cuda_graph_allowance_bytes ||
            ledger.total_planned_device_bytes !=
                report.planned_device_weights_bytes + report.runtime_plan.total_device_bytes) {
            std::cerr << "Static VRAM ledger failed to reconcile\n";
            return 1;
        }
        const std::uint64_t runtime_subtotal =
            ledger.full_attention_kv_bytes + ledger.qsa_block_indexer_bytes +
            ledger.block_tables_bytes + ledger.gdn_recurrent_state_bytes +
            ledger.qsa_raw_state_bytes + ledger.ple_state_bytes +
            ledger.mtp_persistent_state_bytes + ledger.round_tensors_bytes +
            ledger.shared_kernel_workspace_bytes + ledger.sampling_runtime_bytes +
            ledger.cuda_graph_bytes;
        if (runtime_subtotal != ledger.runtime_plan_device_bytes ||
            ledger.device_weight_payload_bytes +
                    ledger.device_weight_alignment_padding_bytes !=
                ledger.planned_device_weight_arena_bytes ||
            ledger.temporary_load_device_bytes != 0 ||
            ledger.cuda_runtime_context_measured ||
            ledger.allocator_fragmentation_measured) {
            std::cerr << "Static VRAM ledger component sum mismatch\n";
            return 1;
        }

        std::cout << "PASS: test_real_artifact_preflight\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "FAILED test_real_artifact_preflight: " << ex.what() << "\n";
        return 1;
    }
}

int test_real_artifact_full_binding_if_available() {
    try {
        const char* env_path = std::getenv("NINFER_QWEN3_8_FLASH_NEXT_WEIGHTS");
        const std::filesystem::path path =
            env_path != nullptr && *env_path != '\0'
                ? std::filesystem::path(env_path)
                : std::filesystem::path(
                      "" /* real-artifact cases run only when NINFER_WEIGHTS is set explicitly */);

        if (!std::filesystem::is_regular_file(path)) {
            std::cout << "SKIP: Real artifact full binding (artifact not present at " << path << ")\n";
            return 0;
        }

        const ninfer::artifact::Reader reader(path);
        validate_identity(reader.identity());
        ninfer::artifact::Binder binder(reader);
        const auto full_plan = bind_artifact(binder, LoadFeatures{.vision = true, .mtp = true});

        std::uint64_t tensor_bytes = 0;
        for (const auto& object : full_plan.materialization.device_objects) {
            tensor_bytes += object.bytes;
        }

        const bool is_nvfp4 = full_plan.bindings.mtp.moe.experts_nvfp4;
        const std::uint64_t expected_tensor_bytes = is_nvfp4 ? 77'667'520'224ULL : 76'251'938'528ULL;
        const std::uint64_t expected_dev_cap      = is_nvfp4 ? 77'667'534'336ULL : 76'251'952'640ULL;
        const std::size_t expected_dev_objs       = is_nvfp4 ? 1'429 : 1'427;
        const std::size_t expected_mapped         = is_nvfp4 ? 131 : 133;
        if (tensor_bytes != expected_tensor_bytes ||
            full_plan.materialization.device_capacity_bytes != expected_dev_cap ||
            full_plan.materialization.device_objects.size() != expected_dev_objs ||
            full_plan.materialization.host_objects.size() != 6 ||
            full_plan.materialization.mapped_tensor_objects.size() != expected_mapped) {
            std::cerr << "Full artifact binding inventory mismatch: tensor_bytes=" << tensor_bytes
                      << " dev_cap=" << full_plan.materialization.device_capacity_bytes
                      << " dev_objs=" << full_plan.materialization.device_objects.size()
                      << " host_objs=" << full_plan.materialization.host_objects.size()
                      << " mapped_objs=" << full_plan.materialization.mapped_tensor_objects.size() << "\n";
            return 1;
        }

        std::cout << "PASS: test_real_artifact_full_binding\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "FAILED test_real_artifact_full_binding: " << ex.what() << "\n";
        return 1;
    }
}

int test_real_artifact_text_and_vision_plan_if_available() {
    try {
        const char* env_path = std::getenv("NINFER_QWEN3_8_FLASH_NEXT_WEIGHTS");
        const std::filesystem::path path =
            env_path != nullptr && *env_path != '\0'
                ? std::filesystem::path(env_path)
                : std::filesystem::path(
                      "" /* real-artifact cases run only when NINFER_WEIGHTS is set explicitly */);

        if (!std::filesystem::is_regular_file(path)) {
            std::cout << "SKIP: Real artifact text+vision plan (artifact not present at " << path
                      << ")\n";
            return 0;
        }

        const ninfer::artifact::Reader reader(path);
        validate_identity(reader.identity());
        ninfer::artifact::Binder binder(reader);
        const auto tv_plan = bind_artifact(binder, LoadFeatures{.vision = true, .mtp = false});

        std::uint64_t tensor_bytes = 0;
        for (const auto& object : tv_plan.materialization.device_objects) {
            tensor_bytes += object.bytes;
        }

        if (tv_plan.materialization.device_objects.size() != 1'400) {
            std::cerr << "Text+Vision device_objects count mismatch: expected 1400, got "
                      << tv_plan.materialization.device_objects.size() << "\n";
            return 1;
        }
        if (tv_plan.materialization.host_objects.size() != 6) {
            std::cerr << "Text+Vision host_objects count mismatch: expected 6, got "
                      << tv_plan.materialization.host_objects.size() << "\n";
            return 1;
        }
        if (tv_plan.materialization.mapped_tensor_objects.size() != 131) {
            std::cerr << "Text+Vision mapped_tensor_objects count mismatch: expected 131, got "
                      << tv_plan.materialization.mapped_tensor_objects.size() << "\n";
            return 1;
        }

        constexpr std::uint64_t kExpectedTvTensorBytes = 76'070'801'632ULL;
        constexpr std::uint64_t kExpectedTvArenaBytes  = 76'070'815'744ULL;
        if (tensor_bytes != kExpectedTvTensorBytes ||
            tv_plan.materialization.device_capacity_bytes != kExpectedTvArenaBytes) {
            std::cerr << "Text+Vision tensor bytes mismatch: expected " << kExpectedTvTensorBytes
                      << " bytes (" << kExpectedTvArenaBytes << " arena), got " << tensor_bytes
                      << " bytes (" << tv_plan.materialization.device_capacity_bytes << " arena)\n";
            return 1;
        }

        std::cout << "PASS: test_real_artifact_text_and_vision_plan\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "FAILED test_real_artifact_text_and_vision_plan: " << ex.what() << "\n";
        return 1;
    }
}

int test_synthetic_artifact_plan_mtp_features() {
    try {
        const auto fixture =
            ninfer::test::flash_next_fixture::create_flash_next_synthetic_artifact(
                "loader_synthetic_mtp");
        const ninfer::artifact::Reader reader(fixture.path);

        // 1. MTP disabled (vision=true, mtp=false)
        {
            ninfer::artifact::Binder binder(reader);
            const auto plan = bind_artifact(binder, LoadFeatures{.vision = true, .mtp = false});
            const auto& m   = plan.materialization;

            if (m.object_count != 1'566 || m.device_objects.size() != 1'400 ||
                m.mapped_tensor_objects.size() != 131 || m.host_objects.size() != 6 ||
                m.device_capacity_bytes != 76'070'815'744ULL) {
                std::cerr << "Synthetic vision-only (mtp off) inventory mismatch: dev_objs="
                          << m.device_objects.size() << " host_objs=" << m.host_objects.size()
                          << " mapped=" << m.mapped_tensor_objects.size()
                          << " dev_cap=" << m.device_capacity_bytes << "\n";
                return 1;
            }
            if (!plan.bindings.features.vision || plan.bindings.features.mtp) {
                std::cerr << "Synthetic features mismatch on mtp=false\n";
                return 1;
            }
        }

        // 2. MTP enabled (vision=true, mtp=true)
        {
            ninfer::artifact::Binder binder(reader);
            const auto plan = bind_artifact(binder, LoadFeatures{.vision = true, .mtp = true});
            const auto& m   = plan.materialization;

            if (m.object_count != 1'566 || m.device_objects.size() != 1'429 ||
                m.mapped_tensor_objects.size() != 131 || m.host_objects.size() != 6 ||
                m.device_capacity_bytes != 77'667'534'336ULL) {
                std::cerr << "Synthetic full (mtp on) inventory mismatch: dev_objs="
                          << m.device_objects.size() << " host_objs=" << m.host_objects.size()
                          << " mapped=" << m.mapped_tensor_objects.size()
                          << " dev_cap=" << m.device_capacity_bytes << "\n";
                return 1;
            }
            if (!plan.bindings.features.vision || !plan.bindings.features.mtp) {
                std::cerr << "Synthetic features mismatch on mtp=true\n";
                return 1;
            }
        }

        // 3. Legacy BF16 MTP enabled
        {
            const auto legacy_fixture =
                ninfer::test::flash_next_fixture::create_flash_next_synthetic_artifact(
                    "synthetic_mtp_legacy_bf16", false);
            const ninfer::artifact::Reader legacy_reader(legacy_fixture.path);
            ninfer::artifact::Binder binder(legacy_reader);
            const auto plan = bind_artifact(binder, LoadFeatures{.vision = true, .mtp = true});
            const auto& m   = plan.materialization;

            if (m.object_count != 1'566 || m.device_objects.size() != 1'427 ||
                m.mapped_tensor_objects.size() != 133 || m.host_objects.size() != 6 ||
                m.device_capacity_bytes != 76'251'952'640ULL) {
                std::cerr << "Synthetic legacy full (mtp on) inventory mismatch: dev_objs="
                          << m.device_objects.size() << " host_objs=" << m.host_objects.size()
                          << " mapped=" << m.mapped_tensor_objects.size()
                          << " dev_cap=" << m.device_capacity_bytes << "\n";
                return 1;
            }
            if (!plan.bindings.features.vision || !plan.bindings.features.mtp ||
                plan.bindings.mtp.moe.experts_nvfp4) {
                std::cerr << "Synthetic legacy features mismatch on mtp=true\n";
                return 1;
            }
        }

        std::cout << "PASS: test_synthetic_artifact_plan_mtp_features\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "FAILED test_synthetic_artifact_plan_mtp_features: " << ex.what() << "\n";
        return 1;
    }
}

} // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::setvbuf(stderr, nullptr, _IONBF, 0);
    if (test_identity_validation() != 0) return 1;
    if (test_ple_metadata_observable_behavior() != 0) return 1;
    if (test_preflight_memory_accounting() != 0) return 1;
    if (test_options_parser_validation() != 0) return 1;
    if (test_synthetic_artifact_plan_mtp_features() != 0) return 1;

    try {
        std::cout << std::flush;
        if (test_real_artifact_preflight_if_available() != 0) return 1;
        if (test_real_artifact_text_and_vision_plan_if_available() != 0) return 1;
        if (test_real_artifact_full_binding_if_available() != 0) return 1;
    } catch (const std::exception& e) {
        std::cerr << "FAILED: real-artifact test threw: " << e.what() << "\n";
        return 1;
    }

    std::cout << "OK Flash-Next Native Loader Tests\n";
    return 0;
}
