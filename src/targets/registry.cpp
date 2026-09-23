#include "targets/registry.h"

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "core/device.h"
#include "core/device_memory.h"
#include "core/startup.h"
#include "runtime/engine/kv_capacity.h"
#include "runtime/engine/context_cost.h"

#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace ninfer::targets {
namespace {

using Clock = std::chrono::steady_clock;

void validate_options(const EngineOptions& options) {
    if (options.artifact_path.empty()) {
        throw std::invalid_argument("Engine artifact_path must not be empty");
    }
    if (options.artifact_path.extension() != ".ninfer") {
        throw std::invalid_argument("NInfer accepts only .ninfer artifacts");
    }
    if (options.max_context == 0) {
        throw std::invalid_argument("Engine max_context must be nonzero");
    }
    switch (options.kv_capacity.mode) {
    case KvCapacityMode::Explicit:
        if (options.kv_capacity.explicit_tokens == 0) {
            throw std::invalid_argument("Engine explicit kv_capacity must be nonzero");
        }
        if (options.kv_capacity.automatic_headroom_bytes != 0) {
            throw std::invalid_argument(
                "Engine explicit kv_capacity must not carry automatic headroom");
        }
        break;
    case KvCapacityMode::Automatic:
        if (options.kv_capacity.explicit_tokens != 0) {
            throw std::invalid_argument(
                "Engine automatic kv_capacity must not carry explicit tokens");
        }
        break;
    default:
        throw std::invalid_argument("Engine kv_capacity mode is invalid");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("Engine max_concurrency must be in [1,8]");
    }
    if (options.max_pending_requests == 0 || options.pending_timeout_ms == 0) {
        throw std::invalid_argument("Engine pending request capacity and timeout must be nonzero");
    }
    if (options.enable_vision && options.media_live_bytes == 0) {
        throw std::invalid_argument(
            "Engine media_live_bytes must be nonzero when Vision is enabled");
    }
    if (options.media_preprocess_threads > 64) {
        throw std::invalid_argument("Engine media_preprocess_threads must be in [0,64]");
    }
}

std::size_t runtime_bytes_after_planned_weights(int device_index,
                                                std::uint64_t weight_bytes,
                                                std::size_t desktop_reserve_bytes) {
    const DeviceMemorySnapshot mem = query_device_memory(device_index);
    const std::size_t initial_required =
        static_cast<std::size_t>(weight_bytes) + desktop_reserve_bytes;
    if (mem.free_bytes < initial_required) {
        throw std::invalid_argument(format_insufficient_memory_error(
            mem, device_index, static_cast<std::size_t>(weight_bytes), desktop_reserve_bytes));
    }
    return mem.free_bytes - initial_required;
}

std::size_t runtime_bytes_after_materialization(int device_index,
                                                std::size_t desktop_reserve_bytes) {
    const DeviceMemorySnapshot mem = query_device_memory(device_index);
    if (mem.free_bytes < desktop_reserve_bytes) {
        throw std::invalid_argument(
            "Insufficient device memory after loading weights: free memory is " +
            format_device_memory_bytes(mem.free_bytes) +
            ", which is less than the required desktop reserve floor of " +
            format_device_memory_bytes(desktop_reserve_bytes) + ".\n" +
            "Another process or OS component holds " + format_device_memory_bytes(mem.used_bytes) +
            " on device " + std::to_string(device_index) +
            (mem.device_name.empty() ? "" : " (" + mem.device_name + ")") + ".\n" +
            "Telemetry source: " + (mem.is_nvml ? "NVML device-wide query" : "cudaMemGetInfo fallback") + ".");
    }
    return mem.free_bytes - desktop_reserve_bytes;
}

std::size_t current_free_device_bytes(int device_index) {
    return query_device_memory(device_index).free_bytes;
}

template <class Target, class Loaded, class Instance>
ConstructedTarget construct_registered(const EngineOptions& options, DeviceContext& device,
                                       artifact::Reader& reader, Clock::time_point load_start,
                                       std::string_view target_key) {
    set_runtime_desktop_reserve_floor(0);
    StartupPhaseScope target_plan_phase(options.startup_observer, StartupPhase::TargetPlan);
    const auto& identity                          = reader.identity();
    const auto weights_profile                    = Target::resolve_weights(identity);
    const ModelSamplingDefaults sampling_defaults = Target::sampling_defaults(identity.model_id);
    const runtime::ContextCostIdentity context_cost_identity{
        .hardware_class = runtime::context_cost_hardware_class(
            device.props.name, device.props.major, device.props.minor),
        .model_id   = identity.model_id,
        .weights_id = identity.weights_id,
    };
    runtime::ResolvedContextMachineCost context_cost = runtime::resolve_context_machine_cost(
        context_cost_identity, options.context_cost.preset_path);

    artifact::Binder binder(reader);
    auto load_plan        = Target::plan_load(binder, options, weights_profile);
    auto sequence_planner = Target::make_sequence_planner(device, options, weights_profile);
    const runtime::SequenceCapacityCurve curve = sequence_planner.capacity_curve();
    const std::size_t preflight_runtime_bytes =
        runtime_bytes_after_planned_weights(device.device,
                                            load_plan.materialization().device_capacity_bytes,
                                            options.desktop_reserve_bytes);
    try {
        (void)runtime::resolve_kv_capacity(options.kv_capacity, curve, preflight_runtime_bytes);
    } catch (const std::invalid_argument& e) {
        const DeviceMemorySnapshot mem = query_device_memory(device.device);
        throw std::invalid_argument(
            std::string(e.what()) + "\n" +
            format_insufficient_memory_error(
                mem, device.device,
                load_plan.materialization().device_capacity_bytes,
                options.desktop_reserve_bytes,
                curve.minimum_device_reservation_bytes));
    }
    target_plan_phase.complete();

    auto materialized = artifact::materialize(reader, load_plan.materialization(), device,
                                              &options.startup_observer);
    const artifact::MaterializationStats stats = materialized.stats();

    StartupPhaseScope target_finalize_phase(options.startup_observer, StartupPhase::TargetFinalize);
    auto model = Target::construct_loaded_model(std::move(load_plan), std::move(materialized));
    device.synchronize();
    const std::size_t post_weights_runtime_bytes =
        runtime_bytes_after_materialization(device.device, options.desktop_reserve_bytes);
    runtime::KvCapacityResolution capacity_resolution =
        runtime::resolve_kv_capacity(options.kv_capacity, curve, post_weights_runtime_bytes);
    capacity_resolution.desktop_reserve_bytes = options.desktop_reserve_bytes;

    const std::uint32_t groups_per_seq =
        (options.max_context + curve.main_page_tokens - 1U) / curve.main_page_tokens;
    const std::uint32_t full_demand_groups = options.max_concurrency * groups_per_seq;
    const double backing_ratio = static_cast<double>(capacity_resolution.main_page_groups) /
                                 static_cast<double>(std::max(1U, groups_per_seq));
    const double coverage_pct =
        (static_cast<double>(capacity_resolution.main_page_groups) /
         static_cast<double>(std::max(1U, full_demand_groups))) * 100.0;

    std::uint32_t effective_concurrency = options.max_concurrency;
    if (options.clamp_concurrency_to_pool &&
        capacity_resolution.main_page_groups < full_demand_groups) {
        effective_concurrency =
            std::max(1U, capacity_resolution.main_page_groups / std::max(1U, groups_per_seq));
    }
    capacity_resolution.requested_concurrency    = options.max_concurrency;
    capacity_resolution.effective_concurrency    = effective_concurrency;
    capacity_resolution.unbacked_concurrency_ratio =
        (coverage_pct < 100.0) ? (static_cast<double>(full_demand_groups) /
                                  static_cast<double>(std::max(1U, capacity_resolution.main_page_groups)))
                               : 1.0;

    std::fprintf(stderr,
                 "[kv-sizer] KV pool: %u page groups (%u tokens, %zu MiB). Demand: %u groups (%u tokens) per full-context sequence.\n"
                 "[kv-sizer] Pool backing: %.1fx full-context sequences (%.1f%% coverage) against requested max_concurrency %u [effective_concurrency=%u].\n",
                 capacity_resolution.main_page_groups, capacity_resolution.resolved_tokens,
                 capacity_resolution.runtime_reservation_bytes / (1024ULL * 1024ULL),
                 groups_per_seq, options.max_context,
                 backing_ratio, coverage_pct, options.max_concurrency, effective_concurrency);

    if (effective_concurrency < options.max_concurrency) {
        std::fprintf(stderr,
                     "[kv-sizer] CONCURRENCY CLAMPED: effective_concurrency reduced from %u to %u lanes to guarantee 100%% full-context backing (--clamp-concurrency-to-pool enabled).\n",
                     options.max_concurrency, effective_concurrency);
    } else if (coverage_pct < 100.0) {
        std::fprintf(stderr,
                     "[kv-sizer] WARNING: KV pool is oversubscribed (%.1f%% full-context backing). Concurrent long requests exceeding pool budget will experience admission queueing.\n",
                     coverage_pct);
    }

    EngineOptions effective_options = options;
    effective_options.max_concurrency = effective_concurrency;

    if (effective_concurrency != options.max_concurrency) {
        sequence_planner = Target::make_sequence_planner(device, effective_options, weights_profile);
    }

    auto sequence_plan = std::move(sequence_planner).finalize(capacity_resolution.main_page_groups);
    if (sequence_plan.device_reservation_bytes() != capacity_resolution.runtime_reservation_bytes ||
        sequence_plan.kv_capacity() != capacity_resolution.resolved_tokens) {
        throw std::logic_error("resolved KV capacity does not match the finalized target plan");
    }
    target_finalize_phase.complete();

    if (effective_options.context_cache.enabled) {
        effective_options.context_cache.max_private_continuations =
            sequence_plan.continuation_capacity();
    }

    StartupPhaseScope frontend_phase(effective_options.startup_observer, StartupPhase::FrontendInitialize);
    auto loaded = std::make_unique<Loaded>(std::move(model), effective_options);
    frontend_phase.complete();

    const std::size_t available_after_weights_bytes = current_free_device_bytes(device.device);

    StartupPhaseScope program_phase(effective_options.startup_observer, StartupPhase::ProgramInitialize);
    auto instance =
        std::make_unique<Instance>(std::move(loaded), capacity_resolution, std::move(sequence_plan),
                                   device, effective_options.startup_observer);
    device.synchronize();
    program_phase.complete();
    instance->kv_capacity_resolution.available_after_startup_bytes =
        current_free_device_bytes(device.device);
    instance->kv_capacity_resolution.desktop_reserve_bytes = options.desktop_reserve_bytes;
    const std::size_t post_startup_free =
        instance->kv_capacity_resolution.available_after_startup_bytes;
    std::fprintf(stderr,
                 "[kv-sizer] Device free after startup: %zu bytes (%zu MiB). Desktop reserve floor: %zu MiB (Slack floor: %zu MiB).\n",
                 post_startup_free, post_startup_free / (1024ULL * 1024ULL),
                 options.desktop_reserve_bytes / (1024ULL * 1024ULL),
                 options.min_slack_floor_bytes / (1024ULL * 1024ULL));

    const std::size_t resident_growth_bytes =
        (available_after_weights_bytes > post_startup_free)
            ? (available_after_weights_bytes - post_startup_free)
            : 0;
    constexpr std::size_t kGrowthToleranceBytes = 32ULL * 1024ULL * 1024ULL;
    if (options.desktop_reserve_bytes > 0 && post_startup_free < options.desktop_reserve_bytes) {
        if (resident_growth_bytes > capacity_resolution.runtime_reservation_bytes + kGrowthToleranceBytes) {
            throw std::runtime_error(
                "[kv-sizer] Engine resident memory growth (" +
                format_device_memory_bytes(resident_growth_bytes) +
                ") exceeded planned runtime reservation (" +
                format_device_memory_bytes(capacity_resolution.runtime_reservation_bytes) +
                ") by " +
                format_device_memory_bytes(
                    resident_growth_bytes - capacity_resolution.runtime_reservation_bytes) +
                ", breaching the desktop reserve floor (" +
                format_device_memory_bytes(options.desktop_reserve_bytes) +
                ", free remaining: " + format_device_memory_bytes(post_startup_free) + ").");
        } else {
            const std::size_t planned_budget = capacity_resolution.runtime_reservation_bytes;
            const std::size_t expected_free =
                (available_after_weights_bytes > planned_budget)
                    ? (available_after_weights_bytes - planned_budget)
                    : 0;
            const std::size_t foreign_usage =
                (expected_free > post_startup_free)
                    ? (expected_free - post_startup_free)
                    : (options.desktop_reserve_bytes - post_startup_free);
            std::fprintf(stderr,
                         "[kv-sizer] CRITICAL WARNING: Device memory free after startup (%zu bytes / %zu MiB) is below the desktop reserve floor (%zu MiB)!\n"
                         "[kv-sizer] Engine resident growth (%zu MiB) respected planned reservation (%zu MiB), but foreign process usage consumed %zu MiB during load.\n"
                         "[kv-sizer] Desktop compositor stalls may occur.\n",
                         post_startup_free, post_startup_free / (1024ULL * 1024ULL),
                         options.desktop_reserve_bytes / (1024ULL * 1024ULL),
                         resident_growth_bytes / (1024ULL * 1024ULL),
                         capacity_resolution.runtime_reservation_bytes / (1024ULL * 1024ULL),
                         foreign_usage / (1024ULL * 1024ULL));
        }
    }
    set_runtime_desktop_reserve_floor(options.desktop_reserve_bytes);

    LoadSummary summary;
    summary.target               = std::string(target_key);
    summary.model_id             = identity.model_id;
    summary.weights_id           = identity.weights_id;
    summary.load_seconds         = std::chrono::duration<double>(Clock::now() - load_start).count();
    summary.upload_seconds       = stats.upload_seconds;
    summary.artifact_bytes_read  = stats.file_bytes;
    summary.host_to_device_bytes = stats.h2d_bytes;
    summary.peak_staging_bytes   = stats.peak_staging_bytes;
    summary.tensor_count         = stats.tensor_count;
    summary.resource_count       = stats.resource_count;
    summary.context_cost         = context_cost.summary;
    return ConstructedTarget{.active            = ActiveTarget(std::move(instance)),
                             .load              = std::move(summary),
                             .sampling_defaults = sampling_defaults,
                             .context_cost      = std::move(context_cost.model),
                             .effective_options = std::move(effective_options)};
}

} // namespace

LoadedQwen3_6_27B::LoadedQwen3_6_27B(std::unique_ptr<Qwen3_6_27B::LoadedModel> stable_model,
                                     const EngineOptions& options)
    : model(std::move(stable_model)), frontend(Qwen3_6_27B::make_frontend(*model, options)) {}

LoadedQwen3_6_27B::~LoadedQwen3_6_27B() = default;

Qwen3_6_27BInstance::Qwen3_6_27BInstance(std::unique_ptr<LoadedQwen3_6_27B> stable_loaded,
                                         runtime::KvCapacityResolution resolution,
                                         Qwen3_6_27B::SequencePlan sequence_plan,
                                         DeviceContext& device,
                                         const StartupObserver& startup_observer)
    : loaded(std::move(stable_loaded)), kv_capacity_resolution(resolution),
      capacity(sequence_plan.capacity()),
      program(Qwen3_6_27B::create_program(*loaded->model, std::move(sequence_plan), device,
                                          startup_observer)) {}

Qwen3_6_27BInstance::~Qwen3_6_27BInstance() {
    set_runtime_desktop_reserve_floor(0);
}

LoadedQwen3_6_35BA3B::LoadedQwen3_6_35BA3B(
    std::unique_ptr<Qwen3_6_35BA3B::LoadedModel> stable_model, const EngineOptions& options)
    : model(std::move(stable_model)), frontend(Qwen3_6_35BA3B::make_frontend(*model, options)) {}

LoadedQwen3_6_35BA3B::~LoadedQwen3_6_35BA3B() = default;

Qwen3_6_35BA3BInstance::Qwen3_6_35BA3BInstance(std::unique_ptr<LoadedQwen3_6_35BA3B> stable_loaded,
                                               runtime::KvCapacityResolution resolution,
                                               Qwen3_6_35BA3B::SequencePlan sequence_plan,
                                               DeviceContext& device,
                                               const StartupObserver& startup_observer)
    : loaded(std::move(stable_loaded)), kv_capacity_resolution(resolution),
      capacity(sequence_plan.capacity()),
      program(Qwen3_6_35BA3B::create_program(*loaded->model, std::move(sequence_plan), device,
                                             startup_observer)) {}

Qwen3_6_35BA3BInstance::~Qwen3_6_35BA3BInstance() {
    set_runtime_desktop_reserve_floor(0);
}

LoadedQwen3_8FlashNext::LoadedQwen3_8FlashNext(
    std::unique_ptr<Qwen3_8FlashNext::LoadedModel> stable_model, const EngineOptions& options)
    : model(std::move(stable_model)), frontend(Qwen3_8FlashNext::make_frontend(*model, options)) {}

LoadedQwen3_8FlashNext::~LoadedQwen3_8FlashNext() = default;

Qwen3_8FlashNextInstance::Qwen3_8FlashNextInstance(
    std::unique_ptr<LoadedQwen3_8FlashNext> stable_loaded, runtime::KvCapacityResolution resolution,
    Qwen3_8FlashNext::SequencePlan sequence_plan, DeviceContext& device,
    const StartupObserver& startup_observer)
    : loaded(std::move(stable_loaded)), kv_capacity_resolution(resolution),
      capacity(sequence_plan.capacity()),
      program(Qwen3_8FlashNext::create_program(*loaded->model, std::move(sequence_plan), device,
                                              startup_observer)) {}

Qwen3_8FlashNextInstance::~Qwen3_8FlashNextInstance() {
    set_runtime_desktop_reserve_floor(0);
}

ConstructedTarget construct_target(const EngineOptions& options, DeviceContext& device) {
    validate_options(options);
    const auto load_start = Clock::now();

    StartupPhaseScope inspect_phase(options.startup_observer, StartupPhase::ArtifactInspect);
    artifact::Reader reader(options.artifact_path);
    inspect_phase.complete();
    const auto& identity = reader.identity();
    if (identity.model_id == Qwen3_6_27B::model_id) {
        return construct_registered<Qwen3_6_27B, LoadedQwen3_6_27B, Qwen3_6_27BInstance>(
            options, device, reader, load_start, Qwen3_6_27B::target_key);
    }
    if (identity.model_id == Qwen3_6_27B::qwen3_8_model_id ||
        identity.model_id == Qwen3_6_27B::orcarouter_model_id) {
        return construct_registered<Qwen3_6_27B, LoadedQwen3_6_27B, Qwen3_6_27BInstance>(
            options, device, reader, load_start, Qwen3_6_27B::qwen3_8_target_key);
    }
    if (identity.model_id == Qwen3_6_35BA3B::model_id) {
        return construct_registered<Qwen3_6_35BA3B, LoadedQwen3_6_35BA3B, Qwen3_6_35BA3BInstance>(
            options, device, reader, load_start, Qwen3_6_35BA3B::target_key);
    }
    if (identity.model_id == Qwen3_8FlashNext::model_id) {
        return construct_registered<Qwen3_8FlashNext, LoadedQwen3_8FlashNext,
                                    Qwen3_8FlashNextInstance>(
            options, device, reader, load_start, Qwen3_8FlashNext::target_key);
    }
    throw std::runtime_error("artifact identity '" + identity.model_id + "/" + identity.weights_id +
                             "' has no registered target for this device");
}

} // namespace ninfer::targets
