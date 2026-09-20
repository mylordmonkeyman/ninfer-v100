#pragma once

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "core/device.h"
#include <ninfer/targets/qwen3_6/frontend_resources.h>
#include "targets/qwen3_8_flash_next/impl/load/bindings.h"
#include "targets/qwen3_8_flash_next/impl/load/materialized.h"
#include "targets/qwen3_8_flash_next/impl/model_view.h"
#include "targets/qwen3_8_flash_next/impl/ple_index.h"
#include "targets/qwen3_8_flash_next/impl/runtime_plan.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace ninfer::targets::qwen3_8_flash_next::detail {

inline constexpr std::string_view kExpectedModelId   = "qwen3.8-flash-next";
inline constexpr std::string_view kExpectedWeightsId = "mixed-nvfp4-fp8-ple-int4";
inline constexpr std::string_view kExpectedIdentity = "qwen3.8-flash-next/mixed-nvfp4-fp8-ple-int4";

void validate_identity(const artifact::ArtifactIdentity& identity);

struct LoadQuantization {
    bool output_head_fp8 = false;
};

struct FlashNextStaticVramLedger {
    // Exact current materialization plan. Category payloads exclude alignment gaps;
    // planned_device_weight_arena_bytes includes them.
    std::uint64_t planned_device_weight_arena_bytes = 0;
    std::uint64_t device_weight_payload_bytes       = 0;
    std::uint64_t device_weight_alignment_padding_bytes = 0;
    std::uint64_t token_embedding_payload_bytes     = 0;
    std::uint64_t output_head_payload_bytes         = 0;
    std::uint64_t routed_expert_payload_bytes       = 0;
    std::uint64_t nonexpert_model_payload_bytes     = 0;
    std::uint64_t routed_expert_layer_payload_bytes = 0;
    std::uint32_t routed_expert_layers              = 0;

    // Canonical pageable/mmap expert storage. These bytes are deliberately not
    // included in the device VRAM total.
    std::uint64_t host_backed_expert_payload_bytes       = 0;
    std::uint64_t host_backed_expert_layer_payload_bytes = 0;
    std::uint32_t host_backed_expert_layers              = 0;

    // Exact runtime-plan device allocations.
    std::uint64_t full_attention_kv_bytes        = 0;
    std::uint64_t qsa_block_indexer_bytes        = 0;
    std::uint64_t block_tables_bytes             = 0;
    std::uint64_t gdn_recurrent_state_bytes      = 0;
    std::uint64_t qsa_raw_state_bytes            = 0;
    std::uint64_t ple_state_bytes                = 0;
    std::uint64_t mtp_persistent_state_bytes     = 0;
    std::uint64_t round_tensors_bytes            = 0;
    std::uint64_t mtp_round_tensors_bytes        = 0;
    std::uint64_t shared_kernel_workspace_bytes  = 0;
    std::uint64_t sampling_runtime_bytes         = 0;
    std::uint64_t cuda_graph_bytes               = 0;
    std::uint64_t runtime_plan_device_bytes      = 0;

    // The generic artifact loader stages through pinned host memory, not a
    // temporary device buffer. CUDA context/driver overhead and allocator
    // external fragmentation are runtime measurements and therefore excluded
    // from this exact dry-run total.
    std::uint64_t temporary_load_device_bytes    = 0;
    bool cuda_runtime_context_measured            = false;
    bool allocator_fragmentation_measured         = false;

    std::uint64_t total_planned_device_bytes      = 0;
};

struct FlashNextPreflightReport {
    artifact::ArtifactIdentity identity;
    std::uint64_t file_bytes                     = 0;
    std::uint64_t planned_device_weights_bytes   = 0;
    std::size_t planned_device_tensors_count     = 0;
    std::size_t planned_retained_resources_count = 0;
    std::size_t planned_mapped_tensors_count     = 0;
    FlashNextRuntimePlan runtime_plan;
    FlashNextStaticVramLedger vram_ledger;
};

// Inspects artifact without allocating device memory for weights.
[[nodiscard]] FlashNextPreflightReport preflight_text_artifact(const artifact::Reader& reader,
                                                               const FlashNextRuntimeConfig& config,
                                                               std::uint32_t main_page_groups = 0);

[[nodiscard]] FlashNextPreflightReport preflight_text_file(const std::filesystem::path& path,
                                                           const FlashNextRuntimeConfig& config,
                                                           std::uint32_t main_page_groups = 0);

class StandaloneLoadedModel {
public:
    ~StandaloneLoadedModel() = default;

    StandaloneLoadedModel(const StandaloneLoadedModel&)                = delete;
    StandaloneLoadedModel& operator=(const StandaloneLoadedModel&)     = delete;
    StandaloneLoadedModel(StandaloneLoadedModel&&) noexcept            = default;
    StandaloneLoadedModel& operator=(StandaloneLoadedModel&&) noexcept = default;

    [[nodiscard]] static StandaloneLoadedModel load(const artifact::Reader& reader,
                                                    DeviceContext& device,
                                                    LoadFeatures features                  = {},
                                                    LoadQuantization quantization          = {},
                                                    const StartupObserver& startup_observer = {});

    [[nodiscard]] static StandaloneLoadedModel
    load_from_file(const std::filesystem::path& path, DeviceContext& device,
                   LoadFeatures features                  = {},
                   LoadQuantization quantization          = {},
                   const StartupObserver& startup_observer = {});

    [[nodiscard]] const TextModelView& text_view() const noexcept { return data_->text; }

    [[nodiscard]] bool has_vision() const noexcept { return data_->vision.has_value(); }

    [[nodiscard]] const VisionModelView& vision_view() const {
        if (!data_->vision) {
            throw std::logic_error(
                "StandaloneLoadedModel: vision was not materialized for this instance");
        }
        return *data_->vision;
    }

    [[nodiscard]] const PleIndexMetadata& ple_metadata() const noexcept { return ple_metadata_; }

    [[nodiscard]] const qwen3_6::FrontendResources& frontend_resources() const noexcept {
        return data_->frontend;
    }

    [[nodiscard]] const artifact::MaterializedArtifact& backing() const noexcept {
        return data_->backing;
    }

    [[nodiscard]] const artifact::MaterializationStats& stats() const noexcept {
        return data_->backing.stats();
    }

private:
    StandaloneLoadedModel(BindingPlan plan, artifact::MaterializedArtifact materialized);
    StandaloneLoadedModel(BindingPlan plan, artifact::MaterializedArtifact materialized,
                          LoadQuantization quantization);

    std::unique_ptr<LoadedModelData> data_;
    PleIndexMetadata ple_metadata_{kPleIndexMetadata};
};

} // namespace ninfer::targets::qwen3_8_flash_next::detail
