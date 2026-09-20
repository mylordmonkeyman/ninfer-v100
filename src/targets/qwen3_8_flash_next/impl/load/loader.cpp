#include "targets/qwen3_8_flash_next/impl/load/loader.h"

#include <array>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <variant>
#include <string>
#include <utility>

namespace ninfer::targets::qwen3_8_flash_next::detail {

namespace {

inline constexpr std::size_t kTextLayerCount = 48;

std::uint64_t checked_add_ledger(std::uint64_t a, std::uint64_t b) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) {
        throw std::overflow_error("Flash-Next static VRAM ledger overflow");
    }
    return a + b;
}

bool text_routed_expert_name(std::string_view name) {
    return name.starts_with("text/layers/") &&
           (name.ends_with("/mlp/experts/gate_up") ||
            name.ends_with("/mlp/experts/down"));
}

std::size_t text_layer_index(std::string_view name) {
    constexpr std::string_view prefix = "text/layers/";
    const auto slash = name.find('/', prefix.size());
    if (slash == std::string_view::npos) {
        throw std::logic_error("Flash-Next routed expert tensor has malformed layer name");
    }
    std::size_t value = 0;
    for (char ch : name.substr(prefix.size(), slash - prefix.size())) {
        if (ch < '0' || ch > '9') {
            throw std::logic_error("Flash-Next routed expert tensor has malformed layer index");
        }
        value = value * 10 + static_cast<std::size_t>(ch - '0');
    }
    if (value >= kTextLayerCount) {
        throw std::logic_error("Flash-Next routed expert tensor layer is out of range");
    }
    return value;
}

FlashNextStaticVramLedger make_static_vram_ledger(
    const artifact::Reader& reader, const artifact::MaterializationPlan& materialization,
    const FlashNextRuntimePlan& runtime_plan) {
    FlashNextStaticVramLedger out{};
    out.planned_device_weight_arena_bytes = materialization.device_capacity_bytes;

    std::array<std::uint64_t, kTextLayerCount> expert_layer_bytes{};
    std::uint64_t previous_end = 0;
    for (const artifact::DeviceMaterialization& placement : materialization.device_objects) {
        if (placement.offset < previous_end) {
            throw std::logic_error("Flash-Next materialization plan is not monotonic");
        }
        out.device_weight_alignment_padding_bytes = checked_add_ledger(
            out.device_weight_alignment_padding_bytes, placement.offset - previous_end);
        out.device_weight_payload_bytes =
            checked_add_ledger(out.device_weight_payload_bytes, placement.bytes);
        previous_end = checked_add_ledger(placement.offset, placement.bytes);

        const auto& descriptor = reader.objects().at(placement.object.index);
        const auto* tensor = std::get_if<artifact::TensorDescriptor>(&descriptor);
        if (tensor == nullptr) {
            throw std::logic_error("Flash-Next device placement is not a tensor");
        }
        const std::string_view name = tensor->name;
        if (name == "text/token_embedding") {
            out.token_embedding_payload_bytes =
                checked_add_ledger(out.token_embedding_payload_bytes, placement.bytes);
        } else if (name == "text/output_head") {
            out.output_head_payload_bytes =
                checked_add_ledger(out.output_head_payload_bytes, placement.bytes);
        } else if (text_routed_expert_name(name)) {
            out.routed_expert_payload_bytes =
                checked_add_ledger(out.routed_expert_payload_bytes, placement.bytes);
            expert_layer_bytes[text_layer_index(name)] =
                checked_add_ledger(expert_layer_bytes[text_layer_index(name)], placement.bytes);
        } else {
            out.nonexpert_model_payload_bytes =
                checked_add_ledger(out.nonexpert_model_payload_bytes, placement.bytes);
        }
    }

    if (previous_end != materialization.device_capacity_bytes) {
        throw std::logic_error("Flash-Next device arena ledger does not reconcile");
    }
    const std::uint64_t classified_payload = checked_add_ledger(
        checked_add_ledger(out.token_embedding_payload_bytes, out.output_head_payload_bytes),
        checked_add_ledger(out.routed_expert_payload_bytes, out.nonexpert_model_payload_bytes));
    if (classified_payload != out.device_weight_payload_bytes) {
        throw std::logic_error("Flash-Next device payload ledger does not reconcile");
    }

    for (std::uint64_t bytes : expert_layer_bytes) {
        if (bytes == 0) { continue; }
        if (out.routed_expert_layer_payload_bytes == 0) {
            out.routed_expert_layer_payload_bytes = bytes;
        } else if (bytes != out.routed_expert_layer_payload_bytes) {
            throw std::logic_error("Flash-Next routed expert layers have unequal payload sizes");
        }
        ++out.routed_expert_layers;
    }

    out.full_attention_kv_bytes       = runtime_plan.attention_kv_bytes;
    out.qsa_block_indexer_bytes       = runtime_plan.indexer_block_keys_bytes;
    out.block_tables_bytes            = runtime_plan.block_tables_bytes;
    out.gdn_recurrent_state_bytes     = runtime_plan.gdn_recurrent_state_bytes;
    out.qsa_raw_state_bytes           = runtime_plan.qsa_raw_state_bytes;
    out.ple_state_bytes               = runtime_plan.ple_state_bytes;
    out.mtp_persistent_state_bytes    = runtime_plan.mtp_persistent_state_bytes;
    out.round_tensors_bytes           = runtime_plan.round_tensors_bytes;
    out.mtp_round_tensors_bytes       = runtime_plan.mtp_round_tensors_bytes;
    out.shared_kernel_workspace_bytes = runtime_plan.workspace_bytes;
    out.sampling_runtime_bytes        = runtime_plan.sampling_runtime_bytes;
    out.cuda_graph_bytes              = runtime_plan.cuda_graph_allowance_bytes;
    out.runtime_plan_device_bytes     = runtime_plan.total_device_bytes;
    out.total_planned_device_bytes =
        checked_add_ledger(out.planned_device_weight_arena_bytes,
                           out.runtime_plan_device_bytes);
    return out;
}

} // namespace

void validate_identity(const artifact::ArtifactIdentity& identity) {
    if (identity.model_id != kExpectedModelId || identity.weights_id != kExpectedWeightsId) {
        throw artifact::ArtifactError("FlashNextTextLoader: unsupported model identity: expected " +
                                      std::string(kExpectedIdentity) + ", got " +
                                      identity.model_id + "/" + identity.weights_id);
    }
}

FlashNextPreflightReport preflight_text_artifact(const artifact::Reader& reader,
                                                 const FlashNextRuntimeConfig& config,
                                                 std::uint32_t main_page_groups) {
    validate_identity(reader.identity());

    artifact::Binder binder(reader);
    const auto load_plan = bind_artifact(binder, LoadFeatures{.vision = false, .mtp = false});

    const auto curve = flash_next_capacity_curve(config);
    const std::uint32_t resolved_groups =
        main_page_groups == 0 ? curve.maximum_main_page_groups : main_page_groups;

    if (resolved_groups < curve.minimum_main_page_groups ||
        resolved_groups > curve.maximum_main_page_groups) {
        throw std::invalid_argument("FlashNextTextLoader: main_page_groups " +
                                    std::to_string(resolved_groups) + " is out of valid range [" +
                                    std::to_string(curve.minimum_main_page_groups) + ", " +
                                    std::to_string(curve.maximum_main_page_groups) + "]");
    }

    auto runtime_plan = finalize_flash_next_runtime_plan(config, resolved_groups);
    auto vram_ledger =
        make_static_vram_ledger(reader, load_plan.materialization, runtime_plan);

    return FlashNextPreflightReport{
        .identity                         = reader.identity(),
        .file_bytes                       = reader.file_bytes(),
        .planned_device_weights_bytes     = load_plan.materialization.device_capacity_bytes,
        .planned_device_tensors_count     = load_plan.materialization.device_objects.size(),
        .planned_retained_resources_count = load_plan.materialization.host_objects.size(),
        .planned_mapped_tensors_count     = load_plan.materialization.mapped_tensor_objects.size(),
        .runtime_plan                     = std::move(runtime_plan),
        .vram_ledger                      = std::move(vram_ledger),
    };
}

FlashNextPreflightReport preflight_text_file(const std::filesystem::path& path,
                                             const FlashNextRuntimeConfig& config,
                                             std::uint32_t main_page_groups) {
    const artifact::Reader reader(path);
    return preflight_text_artifact(reader, config, main_page_groups);
}

StandaloneLoadedModel::StandaloneLoadedModel(BindingPlan plan, artifact::MaterializedArtifact materialized)
    : data_(std::make_unique<LoadedModelData>(std::move(plan), std::move(materialized))) {}

StandaloneLoadedModel::StandaloneLoadedModel(BindingPlan plan, artifact::MaterializedArtifact materialized,
                                             LoadQuantization quantization)
    : data_(std::make_unique<LoadedModelData>(std::move(plan), std::move(materialized),
                                              quantization.output_head_fp8)) {}

StandaloneLoadedModel StandaloneLoadedModel::load(const artifact::Reader& reader, DeviceContext& device,
                                                  LoadFeatures features, LoadQuantization quantization,
                                                  const StartupObserver& startup_observer) {
    validate_identity(reader.identity());

    artifact::Binder binder(reader);
    auto load_plan    = bind_artifact(binder, features);
    auto materialized = artifact::materialize(reader, load_plan.materialization, device, &startup_observer);

    return StandaloneLoadedModel(std::move(load_plan.bindings), std::move(materialized),
                                 quantization);
}

StandaloneLoadedModel StandaloneLoadedModel::load_from_file(const std::filesystem::path& path,
                                                            DeviceContext& device,
                                                            LoadFeatures features,
                                                            LoadQuantization quantization,
                                                            const StartupObserver& startup_observer) {
    const artifact::Reader reader(path);
    return load(reader, device, features, quantization, startup_observer);
}

} // namespace ninfer::targets::qwen3_8_flash_next::detail
