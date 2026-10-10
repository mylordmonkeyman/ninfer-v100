#include "targets/qwen3_8_flash_next/impl/moe.h"
#include "targets/qwen3_8_flash_next/impl/expert_cache.h"
#include "targets/qwen3_8_flash_next/impl/expert_gemm.h"
#include "targets/qwen3_8_flash_next/impl/expert_stream.h"
#include "targets/qwen3_8_flash_next/impl/stream_order.h"
#include "targets/qwen3_8_flash_next/impl/stream_fraction.h"
#include "targets/qwen3_8_flash_next/impl/stream_admission.h"
#include "targets/qwen3_8_flash_next/impl/route_handoff.h"
#include "targets/qwen3_8_flash_next/impl/route_handoff_policy.h"
#include "targets/qwen3_8_flash_next/impl/stream_diagnostics.h"

#include "core/layout.h"
#include "targets/qwen3_8_flash_next/impl/cpu_expert_pool.h"
#include "targets/qwen3_8_flash_next/impl/cpu_group_policy.h"
#include "targets/qwen3_8_flash_next/impl/moe_kernels.h"
#include "targets/qwen3_8_flash_next/impl/moe_route.h"
#include "targets/qwen3_8_flash_next/impl/moe_workspace.h"
#include "targets/qwen3_8_flash_next/impl/stage_ledger.h"
#include "targets/qwen3_8_flash_next/impl/perf_telemetry.h"
#include "ninfer/ops/expert_route_combine.h"

#include <algorithm>
#include <atomic>
#include <bitset>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <mutex>
#include <semaphore>
#include <span>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <vector>
#include <condition_variable>
#include <future>
#include <cstring>

#include "core/device.h"

namespace ninfer::targets::qwen3_8_flash_next::detail {
namespace {

std::atomic<std::uint64_t> s_host_expert_layer_calls{0};
std::atomic<std::uint64_t> s_host_expert_fp32_output_calls{0};
std::atomic<std::uint64_t> s_host_expert_routed_tokens{0};
std::atomic<std::uint64_t> s_host_expert_pairs{0};

unsigned resolve_host_expert_worker_count() {
#if defined(NINFER_VOLTA_BUILD)
    constexpr unsigned kDefaultWorkers = 64;
#else
    constexpr unsigned kDefaultWorkers = 32;
#endif
    constexpr unsigned kMaximumWorkers = 256;
    if (const char* env = std::getenv("NINFER_FLASH_NEXT_CPU_EXPERT_WORKERS");
        env != nullptr && env[0] != '\0') {
        char* end = nullptr;
        const unsigned long parsed = std::strtoul(env, &end, 10);
        if (end == env || *end != '\0' || parsed == 0 || parsed > kMaximumWorkers) {
            throw std::invalid_argument(
                "NINFER_FLASH_NEXT_CPU_EXPERT_WORKERS must be in [1, 256]");
        }
        return static_cast<unsigned>(parsed);
    }
    const unsigned hardware = std::thread::hardware_concurrency();
    return std::min(kDefaultWorkers, hardware == 0 ? 1U : hardware);
}

bool resolve_fp32_intermediate_diagnostic() {
    const char* env = std::getenv("NINFER_FLASH_NEXT_CPU_EXPERT_FP32_INTERMEDIATE");
    return env != nullptr && env[0] != '\0' && std::string_view(env) != "0";
}

bool resolve_shared_fp32_intermediate_diagnostic() {
    const char* env = std::getenv("NINFER_FLASH_NEXT_MOE_SHARED_FP32_INTERMEDIATE");
    return env != nullptr && env[0] != '\0' && std::string_view(env) != "0";
}

bool resolve_cpu_expert_grouping(bool prefill) {
    return flash_next_cpu_expert_grouping_enabled(
        std::getenv("NINFER_V100_CPU_EXPERT_GROUP"), prefill);
}

bool resolve_route_handoff(bool prefill) {
    const char* env = std::getenv("NINFER_V100_ROUTE_HANDOFF");
    return route_handoff_enabled(env ? std::string_view(env) : std::string_view{}, prefill);
}

// Experimental schedule only; the default stream submission order is unchanged.
bool resolve_stream_expert_busy_first() {
    const char* policy = std::getenv("NINFER_V100_STREAM_EXPERT_ORDER");
    if (!policy || !*policy || std::string_view(policy) == "id") return false;
    if (std::string_view(policy) == "busy-first") return true;
    throw std::invalid_argument(
        "NINFER_V100_STREAM_EXPERT_ORDER must be id or busy-first");
}

bool resolve_device_route_combine() {
    const char* env = std::getenv("NINFER_V100_DEVICE_ROUTE_COMBINE");
    if (env == nullptr || env[0] == '\0' || std::string_view(env) == "0") return false;
    if (std::string_view(env) == "1") return true;
    throw std::invalid_argument("NINFER_V100_DEVICE_ROUTE_COMBINE must be 0 or 1");
}

double resolve_decode_expert_stream_fraction(bool prefill, bool stream_experts) {
    if (!stream_experts) return 1.0;
    // Prefill stays GPU-only by default. Opt-in hybrid prefill streams the
    // busiest nonresident experts while the CPU computes the remaining groups.
    // The existing decode fraction retains exactly its prior behavior.
    const char* name = prefill
        ? "NINFER_V100_PREFILL_EXPERT_STREAM_FRACTION"
        : "NINFER_V100_DECODE_EXPERT_STREAM_FRACTION";
    return flash_next_parse_expert_stream_fraction(std::getenv(name), name);
}

bool resolve_avx2_backend() {
    const char* env = std::getenv("NINFER_FLASH_NEXT_CPU_EXPERT_BACKEND");
    if (env == nullptr || env[0] == '\0' || std::string_view(env) == "auto") {
        return !resolve_fp32_intermediate_diagnostic() && flash_next_cpu_nvfp4_avx2_available();
    }
    if (std::string_view(env) == "reference") { return false; }
    if (std::string_view(env) == "avx2") {
        if (!flash_next_cpu_nvfp4_avx2_available()) {
            throw std::runtime_error(
                "NINFER_FLASH_NEXT_CPU_EXPERT_BACKEND=avx2 requested without AVX2/FMA");
        }
        return true;
    }
    throw std::invalid_argument(
        "NINFER_FLASH_NEXT_CPU_EXPERT_BACKEND must be auto, reference or avx2");
}

HostExpertWorkerPool& host_expert_worker_pool() {
    static HostExpertWorkerPool pool(resolve_host_expert_worker_count(),
        resolve_avx2_backend(), resolve_fp32_intermediate_diagnostic());
    return pool;
}

struct RouteHandoffDrain {
    FlashNextRouteHandoff* handoff;
    cudaStream_t compute;
    int exceptions = std::uncaught_exceptions();
    ~RouteHandoffDrain() {
        if (handoff) {
            handoff->drain();
            // An early failure can leave shared kernels using the workspace.
            // Drain them before its scope is restored; successful work keeps
            // the normal ordered merge and asynchronous compute behavior.
            if (std::uncaught_exceptions() > exceptions) cudaStreamSynchronize(compute);
        }
    }
};

struct HostMoeTransferTiming {
    cudaEvent_t start = nullptr, stop = nullptr;
    HostMoeTransferTiming() {
        CUDA_CHECK(cudaEventCreate(&start));
        try { CUDA_CHECK(cudaEventCreate(&stop)); }
        catch (...) { cudaEventDestroy(start); throw; }
    }
    ~HostMoeTransferTiming() { cudaEventDestroy(start); cudaEventDestroy(stop); }
};
struct HostMoeCpuBuffers {
    std::unique_ptr<FlashNextRouteHandoff> route_handoff;
    std::unique_ptr<HostMoeTransferTiming> transfer_timing;
    std::vector<std::uint16_t> input;
    std::vector<float> input_fp32;
    std::vector<std::int32_t> ids;
    std::vector<float> alpha;
    std::vector<float> routed_sum;
    std::vector<float> pair_outputs;
    std::unique_ptr<PinnedHostBuffer> pinned_pair_outputs;
    std::size_t pinned_pair_output_words = 0;
    std::vector<std::size_t> miss_routes;
    std::vector<HostExpertTask> tasks;

    float* ensure_pinned_pair_outputs(std::size_t words) {
        if (pinned_pair_output_words < words) {
            pinned_pair_outputs = std::make_unique<PinnedHostBuffer>(words*sizeof(float));
            pinned_pair_output_words = words;
        }
        return static_cast<float*>(pinned_pair_outputs->data());
    }
};

thread_local HostMoeCpuBuffers s_host_moe_buffers;


bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

bool exact_bf16_weight(const Weight& weight, std::int32_t rows, std::int32_t columns) {
    return weight.qtype == QType::BF16_CTRL && weight.layout == QuantLayout::Contiguous &&
           weight.n == rows && weight.k == columns && weight.ndim == 2 && weight.shape[0] == rows &&
           weight.shape[1] == columns && weight.padded_shape[0] == rows &&
           weight.padded_shape[1] == columns && weight.qdata == weight.payload &&
           weight.payload_bytes >= static_cast<std::uint64_t>(rows) * columns * 2 &&
           aligned_to(weight.qdata, 16);
}

bool exact_expert_bank(const Nvfp4ExpertBankView& bank, std::int32_t rows, std::int32_t columns) {
    const std::uint64_t elements = static_cast<std::uint64_t>(rows) * columns;
    return bank.experts == 512 && bank.rows == rows && bank.columns == columns &&
           bank.code_bytes_per_expert == elements / 2 &&
           bank.scale_bytes_per_expert == elements / 16 && aligned_to(bank.codes, 16) &&
           aligned_to(bank.scales, 16) && aligned_to(bank.weight_scale_divisors, 16);
}

bool exact_bf16_expert_bank(const Bf16ExpertBankView& bank, std::int32_t rows, std::int32_t columns) {
    const std::uint64_t elements = static_cast<std::uint64_t>(rows) * columns;
    return bank.experts == 512 && bank.rows == rows && bank.columns == columns &&
           bank.bytes_per_expert == elements * sizeof(std::uint16_t) &&
           aligned_to(bank.data, 16);
}

} // namespace

void reset_flash_next_host_expert_execution_stats() noexcept {
    s_host_expert_layer_calls.store(0, std::memory_order_relaxed);
    s_host_expert_fp32_output_calls.store(0, std::memory_order_relaxed);
    s_host_expert_routed_tokens.store(0, std::memory_order_relaxed);
    s_host_expert_pairs.store(0, std::memory_order_relaxed);
}

FlashNextHostExpertExecutionStats
flash_next_host_expert_execution_stats() noexcept {
    return FlashNextHostExpertExecutionStats{
        .completed_layer_calls =
            s_host_expert_layer_calls.load(std::memory_order_relaxed),
        .fp32_output_layer_calls =
            s_host_expert_fp32_output_calls.load(std::memory_order_relaxed),
        .routed_tokens =
            s_host_expert_routed_tokens.load(std::memory_order_relaxed),
        .expert_pairs =
            s_host_expert_pairs.load(std::memory_order_relaxed),
    };
}

std::size_t flash_next_moe_workspace_capacity_bytes(std::int32_t min_tokens,
                                                    std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("Flash-Next MoE workspace requires positive tokens");
    }
    WorkspaceLayoutBuilder layout;
    (void)allocate_flash_next_moe_workspace(layout, max_tokens);
    return layout.peak_bytes(256);
}

void flash_next_moe(const Tensor& input, const MoeWeights& weights, Tensor& output,
                    WorkspaceArena& workspace, cudaStream_t stream) {
    const std::int32_t tokens = input.ne[1];
    if (input.dtype != DType::BF16 || output.dtype != DType::BF16 || input.ne[0] != 2'560 ||
        output.ne[0] != 2'560 || tokens < 1 || output.ne[1] != tokens ||
        input.ne[2] != 1 || input.ne[3] != 1 || output.ne[2] != 1 || output.ne[3] != 1 ||
        !input.is_contiguous() || !output.is_contiguous() || !aligned_to(input.data, 16) ||
        !aligned_to(output.data, 16) || !exact_bf16_weight(weights.router, 512, 2'560) ||
        !exact_bf16_weight(weights.shared_down, 2'560, 640) ||
        !exact_bf16_weight(weights.shared_gate, 640, 2'560) ||
        !exact_bf16_weight(weights.shared_up, 640, 2'560) ||
        !exact_bf16_weight(weights.shared_gate_weight, 1, 2'560) ||
        !exact_expert_bank(weights.expert_gate_up, 1'280, 2'560) ||
        !exact_expert_bank(weights.expert_down, 2'560, 640) || stream == nullptr) {
        throw std::invalid_argument("Flash-Next MoE received an invalid exact target view");
    }
    const auto scope              = workspace.scope();
    FlashNextMoeWorkspace scratch = allocate_flash_next_moe_workspace(workspace, tokens);
    flash_next_route(input, weights.router, weights.shared_gate_weight, scratch.scores, scratch.ids,
                     scratch.alpha, scratch.shared_scale, stream);
    stage_ledger_record(stream, FlashNextStageId::MoE_Router);
    flash_next_moe_kernels_launch(input, weights, scratch, output, stream);

    // Diagnostic NINFER_FLASH_NEXT_TRACE_ROUTING only; not on the default prefill chunk path.
    static const char* trace_routing_env = std::getenv("NINFER_FLASH_NEXT_TRACE_ROUTING");
    if (trace_routing_env != nullptr && trace_routing_env[0] != '\0' &&
        tokens >= kFlashNextMoeMmaPrefillThreshold) {
        static int s_route_call = 0;
        int h_active = 0;
        cudaMemcpyAsync(&h_active, scratch.active_count.data, sizeof(int), cudaMemcpyDeviceToHost, stream);
        std::vector<int> h_counts(512);
        cudaMemcpyAsync(h_counts.data(), scratch.expert_counts.data, 512 * sizeof(int), cudaMemcpyDeviceToHost, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        int max_grp = 0, min_grp = 999999, sum_grp = 0;
        for (int e = 0; e < 512; ++e) {
            if (h_counts[e] > 0) {
                max_grp = std::max(max_grp, h_counts[e]);
                min_grp = std::min(min_grp, h_counts[e]);
                sum_grp += h_counts[e];
            }
        }
        int layer_idx = (s_route_call++) % 48;
        if (layer_idx == 0) {
            std::fprintf(stderr, "\n--- MoE Routing Trace (T=%d) ---\n", tokens);
            std::fprintf(stderr, "Layer | Active Experts | %% Active | Min Group | Avg Group | Max Group\n");
            std::fprintf(stderr, "------+----------------+----------+-----------+-----------+----------\n");
        }
        std::fprintf(stderr, " L%02d  |   %3d / 512    |  %5.1f%%  |    %3d    |   %5.1f   |    %3d   \n",
                     layer_idx, h_active, h_active * 100.0f / 512.0f, min_grp, (float)sum_grp / h_active, max_grp);
        if (layer_idx == 47) {
            std::fprintf(stderr, "-----------------------------------------------------------------\n\n");
        }
    }
}

void flash_next_moe_host_backed(const Tensor& input, const MoeWeights& resident_weights,
                                const HostNvfp4ExpertLayerView& host_experts, Tensor& output,
                                WorkspaceArena& workspace, cudaStream_t stream,
                                const MoeStageEmitter& emit,
                                const Tensor* router_input_fp32,
                                const Tensor* routed_expert_input_fp32,
                                const Tensor* shared_expert_input_fp32,
                                Tensor* output_fp32, FlashNextExpertCache* cache, unsigned layer,
                                bool prefill, FlashNextExpertStream* expert_stream,
                                unsigned expert_stream_min_routes) {
    v100_compare::HostSpan moe_span(layer, v100_compare::Stage::moe);
    const std::int32_t tokens = input.ne[1];
    if (input.dtype != DType::BF16 || output.dtype != DType::BF16 || input.ne[0] != 2'560 ||
        output.ne[0] != 2'560 || tokens < 1 || output.ne[1] != tokens ||
        input.ne[2] != 1 || input.ne[3] != 1 || output.ne[2] != 1 || output.ne[3] != 1 ||
        !input.is_contiguous() || !output.is_contiguous() || !aligned_to(input.data, 16) ||
        !aligned_to(output.data, 16) ||
        !exact_bf16_weight(resident_weights.router, 512, 2'560) ||
        !exact_bf16_weight(resident_weights.shared_down, 2'560, 640) ||
        !exact_bf16_weight(resident_weights.shared_gate, 640, 2'560) ||
        !exact_bf16_weight(resident_weights.shared_up, 640, 2'560) ||
        !exact_bf16_weight(resident_weights.shared_gate_weight, 1, 2'560) ||
        !exact_expert_bank(host_experts.gate_up, 1'280, 2'560) ||
        !exact_expert_bank(host_experts.down, 2'560, 640) || stream == nullptr) {
        throw std::invalid_argument("Flash-Next host-backed MoE received an invalid exact target view");
    }

    if (output_fp32 != nullptr &&
        (output_fp32->data == nullptr || output_fp32->dtype != DType::FP32 ||
         output_fp32->ne[0] != 2'560 || output_fp32->ne[1] != tokens ||
         output_fp32->ne[2] != 1 || output_fp32->ne[3] != 1 ||
         !output_fp32->is_contiguous() || !aligned_to(output_fp32->data, 16))) {
        throw std::invalid_argument("Flash-Next MoE received an invalid FP32 output");
    }
    const bool use_routed_expert_input_fp32 =
        routed_expert_input_fp32 != nullptr && routed_expert_input_fp32->data != nullptr;
    if (use_routed_expert_input_fp32 &&
        (routed_expert_input_fp32->dtype != DType::FP32 ||
         routed_expert_input_fp32->ne[0] != kFlashNextExpertHidden ||
         routed_expert_input_fp32->ne[1] != tokens ||
         routed_expert_input_fp32->ne[2] != 1 || routed_expert_input_fp32->ne[3] != 1 ||
         !routed_expert_input_fp32->is_contiguous() ||
         !aligned_to(routed_expert_input_fp32->data, 16))) {
        throw std::invalid_argument(
            "Flash-Next host-backed MoE received an invalid FP32 routed-expert input");
    }
    const bool use_shared_expert_input_fp32 =
        shared_expert_input_fp32 != nullptr && shared_expert_input_fp32->data != nullptr;
    if (use_shared_expert_input_fp32 &&
        (shared_expert_input_fp32->dtype != DType::FP32 ||
         shared_expert_input_fp32->ne[0] != kFlashNextExpertHidden ||
         shared_expert_input_fp32->ne[1] != tokens ||
         shared_expert_input_fp32->ne[2] != 1 || shared_expert_input_fp32->ne[3] != 1 ||
         !shared_expert_input_fp32->is_contiguous() ||
         !aligned_to(shared_expert_input_fp32->data, 16))) {
        throw std::invalid_argument(
            "Flash-Next host-backed MoE received an invalid FP32 shared-expert input");
    }
    const bool use_shared_fp32_intermediate =
        resolve_shared_fp32_intermediate_diagnostic();
    if (use_shared_fp32_intermediate && use_shared_expert_input_fp32) {
        throw std::invalid_argument(
            "shared FP32-intermediate diagnostic requires the production BF16 shared input");
    }

    const std::size_t input_words =
        static_cast<std::size_t>(tokens) * kFlashNextExpertHidden;
    const std::size_t routed_paths = static_cast<std::size_t>(tokens) * 10ULL;
#if defined(NINFER_VOLTA_BUILD)
    const bool device_route_combine = resolve_device_route_combine() &&
        !use_routed_expert_input_fp32 && !resolve_fp32_intermediate_diagnostic();
    const bool route_handoff = resolve_route_handoff(prefill);
#else
    const bool device_route_combine = false;
    const bool route_handoff = false;
#endif
    const bool stream_experts = expert_stream != nullptr;
    if (stream_experts && (expert_stream_min_routes < 1 || expert_stream_min_routes > 2)) {
        throw std::invalid_argument("expert stream minimum routes must be 1 or 2");
    }
    const double stream_fraction =
        resolve_decode_expert_stream_fraction(prefill, stream_experts);
    const unsigned prefill_stream_min_routes = stream_experts && prefill ?
        flash_next_parse_prefill_stream_min_routes(
            std::getenv("NINFER_V100_PREFILL_EXPERT_STREAM_MIN_ROUTES")) : 0;
    if (prefill_stream_min_routes && stream_fraction < 1.0)
        throw std::invalid_argument(
            "NINFER_V100_PREFILL_EXPERT_STREAM_MIN_ROUTES conflicts with "
            "NINFER_V100_PREFILL_EXPERT_STREAM_FRACTION below 1");
    const bool stream_cpu_fallback = stream_experts &&
        (expert_stream_min_routes > 1 || stream_fraction < 1.0 ||
         prefill_stream_min_routes > 1);
    const bool streamed_prefill_admit_hot = prefill && stream_experts && cache &&
        flash_next_stream_prefill_admit_hot(
            std::getenv("NINFER_V100_PREFILL_STREAM_ADMIT"));
    if (route_handoff && (!device_route_combine || stream_experts)) {
        throw std::invalid_argument(
            "route handoff requires device combine and BF16 CPU-cache expert execution");
    }
    HostMoeCpuBuffers& cpu = s_host_moe_buffers;
    if (route_handoff) {
        if (!cpu.route_handoff) {
            cudaStreamCaptureStatus capture;
            CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
            if (capture != cudaStreamCaptureStatusNone) {
                throw std::invalid_argument("route handoff requires eager host-backed execution");
            }
            cpu.route_handoff = std::make_unique<FlashNextRouteHandoff>();
        }
        // Growth occurs before router/shared work, never while a copy owns storage.
        cpu.route_handoff->prepare(input_words, routed_paths, stream);
        // cudaHostAlloc may synchronize: reserve the maximum miss-output
        // payload here so growing it cannot reintroduce the shared-work wait.
        cpu.ensure_pinned_pair_outputs(routed_paths * kFlashNextExpertHidden);
    }
    const auto scope = workspace.scope();
    FlashNextMoeWorkspace scratch = allocate_flash_next_moe_workspace(workspace, tokens);
    const RouteHandoffDrain route_drain{route_handoff ? cpu.route_handoff.get() : nullptr, stream};

#if defined(NINFER_VOLTA_BUILD)
    if (router_input_fp32 != nullptr && router_input_fp32->data != nullptr) {
        flash_next_route_fp32_input(
            *router_input_fp32, resident_weights.router,
            resident_weights.shared_gate_weight, scratch.scores, scratch.ids,
            scratch.alpha, scratch.shared_scale, stream);
    } else
#endif
    {
        flash_next_route(input, resident_weights.router, resident_weights.shared_gate_weight,
                         scratch.scores, scratch.ids, scratch.alpha, scratch.shared_scale,
                         stream);
    }
    const std::uint64_t route_ticket = route_handoff ?
        cpu.route_handoff->submit(input.data, scratch.ids.data, stream) : 0;
    stage_ledger_record(stream, FlashNextStageId::MoE_Router);
    if (emit) {
        emit("moe_expert_input", input);
        emit("moe_router_scores", scratch.scores);
        emit("moe_router_ids", scratch.ids);
        emit("moe_router_alpha", scratch.alpha);
        emit("moe_shared_scale", scratch.shared_scale);
    }

    // The shared expert remains resident on device. By default its gate/up activation is
    // materialized as BF16 before the host rendezvous. Diagnostics may change either the
    // gate/up input boundary or, separately, retain SiLU(gate)*up in FP32 for shared_down.
    // The shared down projection is deferred until the routed FP32 sum returns so both
    // branches can be combined before the final BF16 rounding.
    flash_next_moe_host_shared_launch(
        input, resident_weights, scratch, stream,
        use_shared_expert_input_fp32 ? shared_expert_input_fp32 : nullptr,
        use_shared_fp32_intermediate);

    if (stream_experts && !device_route_combine) {
        throw std::invalid_argument(
            "Flash-Next expert streaming requires device route combine");
    }
    if (use_routed_expert_input_fp32) {
        cpu.input_fp32.resize(input_words);
    } else if ((!stream_experts || stream_cpu_fallback) && !route_handoff) {
        cpu.input.resize(input_words);
    }
    cpu.ids.resize(routed_paths);
    if (!device_route_combine) {
        cpu.alpha.resize(routed_paths);
        cpu.routed_sum.resize(
            static_cast<std::size_t>(tokens) * kFlashNextExpertHidden);
        cpu.pair_outputs.resize(routed_paths * kFlashNextExpertHidden);
        std::fill(cpu.routed_sum.begin(), cpu.routed_sum.end(), 0.0F);
    }
    float* device_route_host_outputs = nullptr;
    cpu.tasks.clear();
    cpu.tasks.reserve(routed_paths);
    cpu.miss_routes.clear();
    cpu.miss_routes.reserve(routed_paths);

    const bool telemetry = v100_perf_telemetry_enabled();
    const bool aggregate = v100_compare::active != nullptr;
    std::optional<v100_compare::CacheSnapshot> cache_begin;
    if (aggregate && cache) cache_begin = cache->telemetry_snapshot(layer, v100_compare::level() >= 2);
    std::bitset<512> resident_ids, missed_ids;
    const bool observe_host = telemetry || aggregate;
    const auto rendezvous_started = observe_host ? PerfClock::now() : PerfClock::time_point{};
    const std::uint16_t* host_input = cpu.input.data();
    if (route_handoff) {
        cpu.route_handoff->wait(route_ticket);
        host_input = cpu.route_handoff->input(route_ticket).data();
        const auto ids = cpu.route_handoff->ids(route_ticket);
        std::copy(ids.begin(), ids.end(), cpu.ids.begin());
    } else {
        if (use_routed_expert_input_fp32) {
            CUDA_CHECK(cudaMemcpyAsync(
                cpu.input_fp32.data(), routed_expert_input_fp32->data,
                input_words * sizeof(float), cudaMemcpyDeviceToHost, stream));
        } else if (!stream_experts || stream_cpu_fallback) {
            CUDA_CHECK(cudaMemcpyAsync(cpu.input.data(), input.data,
                                       input_words * sizeof(std::uint16_t),
                                       cudaMemcpyDeviceToHost, stream));
        }
        CUDA_CHECK(cudaMemcpyAsync(cpu.ids.data(), scratch.ids.data,
                                   routed_paths * sizeof(std::int32_t),
                                   cudaMemcpyDeviceToHost, stream));
        if (!device_route_combine) {
            CUDA_CHECK(cudaMemcpyAsync(cpu.alpha.data(), scratch.alpha.data,
                                       routed_paths * sizeof(float),
                                       cudaMemcpyDeviceToHost, stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    if (cache) cache->begin_layer(prefill);
    const double rendezvous_us = observe_host ? perf_elapsed_us(rendezvous_started) : 0;
    const bool measure = observe_host || (cache != nullptr && cache->timing_enabled());
    using Clock = std::chrono::steady_clock;
    const auto branch_started = measure ? Clock::now() : Clock::time_point{};
    std::array<std::vector<FlashNextStreamRoute>, 512> streamed_routes;
    std::array<std::vector<std::size_t>, 512> streamed_route_indices;
    std::uint64_t streamed_route_count = 0, streamed_experts = 0;
    bool stream_hits_submitted = false;
    const bool early_cpu_requested=prefill && flash_next_cpu_stream_overlap_requested();
    const bool early_cpu=early_cpu_requested && stream_experts && device_route_combine &&
        !(cache && cache->serial_schedule());
    bool grouped_cpu_experts = false;
#if defined(NINFER_VOLTA_BUILD)
    grouped_cpu_experts = tokens > 1 && resolve_cpu_expert_grouping(prefill);
#endif
    std::future<HostExpertBatchStats> pending_cpu;
    auto cpu_started=measure ? Clock::now() : Clock::time_point{};
    auto cpu_finished=cpu_started;
    auto stream_submit_started=cpu_started, stream_submit_finished=cpu_started;

    // Independent routed expert pairs are computed concurrently. Each task writes
    // a private FP32 vector. Routing alpha is then accumulated below on this thread
    // in the original token/path order so the reduction contract remains deterministic.
    try {
    for (std::int32_t token = 0; token < tokens; ++token) {
        const std::size_t token_offset =
            static_cast<std::size_t>(token) * kFlashNextExpertHidden;
        const std::uint16_t* token_input =
            use_routed_expert_input_fp32 || (stream_experts && !stream_cpu_fallback) ? nullptr :
                host_input + token_offset;
        const float* token_input_fp32 =
            use_routed_expert_input_fp32 ? cpu.input_fp32.data() + token_offset : nullptr;
        for (std::int32_t path = 0; path < 10; ++path) {
            const std::size_t route_index =
                static_cast<std::size_t>(token) * 10ULL +
                static_cast<std::size_t>(path);
            if (stream_experts) {
                const auto expert_id = cpu.ids[route_index];
                if (expert_id < 0 || expert_id >= 512) {
                    throw std::runtime_error("Flash-Next router returned invalid expert id");
                }
                if (cache && cache->execute_to(layer, expert_id,
                        static_cast<const std::uint16_t*>(input.data) + token_offset,
                        static_cast<float*>(scratch.down_intermediate.data) +
                            route_index * kFlashNextExpertHidden,
                        static_cast<unsigned>(route_index), stream)) {
                    if (aggregate) resident_ids.set(static_cast<std::size_t>(expert_id));
                    continue;
                }
                if (aggregate) missed_ids.set(static_cast<std::size_t>(expert_id));
                streamed_routes[static_cast<std::size_t>(expert_id)].push_back(
                    FlashNextStreamRoute{
                        static_cast<const std::uint16_t*>(input.data) + token_offset,
                        static_cast<float*>(scratch.down_intermediate.data) +
                            route_index * kFlashNextExpertHidden});
                streamed_route_indices[static_cast<std::size_t>(expert_id)].push_back(route_index);
                continue;
            }
            const bool cache_hit = cache != nullptr && !use_routed_expert_input_fp32 &&
                !resolve_fp32_intermediate_diagnostic() &&
                (device_route_combine ? cache->execute_to(layer, cpu.ids[route_index],
                    static_cast<const std::uint16_t*>(input.data)+token_offset,
                    static_cast<float*>(scratch.down_intermediate.data)+
                        route_index*kFlashNextExpertHidden,
                    static_cast<unsigned>(route_index), stream) :
                 cache->execute(layer, cpu.ids[route_index],
                    static_cast<const std::uint16_t*>(input.data)+token_offset,
                    static_cast<unsigned>(route_index), stream));
            if (cache_hit) {
                if (aggregate) resident_ids.set(static_cast<std::size_t>(cpu.ids[route_index]));
                continue;
            }
            if (aggregate) missed_ids.set(static_cast<std::size_t>(cpu.ids[route_index]));
            cpu.miss_routes.push_back(route_index);
            cpu.tasks.push_back(HostExpertTask{
                .expert = host_experts.expert(cpu.ids[route_index]),
                .expert_id = cpu.ids[route_index],
                .route_id = static_cast<std::uint32_t>(route_index),
                .input = token_input,
                .input_fp32 = token_input_fp32,
                .output = device_route_combine ? nullptr :
                    cpu.pair_outputs.data()+route_index*kFlashNextExpertHidden,
            });
        }
    }
    if (stream_experts) {
        // Fractional decode streaming mirrors Strata's hybrid principle: rank the
        // current layer's distinct misses by routed work, stream only the hottest
        // share, and leave the rest on the AVX2 pool. The older hybrid policy
        // remains the min-routes threshold when the fraction is 1.
        std::array<bool, 512> selected{};
        if (stream_fraction < 1.0) {
            std::vector<std::pair<std::size_t, std::size_t>> active;
            active.reserve(512);
            for (std::size_t expert_id = 0; expert_id < streamed_routes.size(); ++expert_id) {
                if (!streamed_routes[expert_id].empty()) {
                    active.emplace_back(streamed_routes[expert_id].size(), expert_id);
                }
            }
            std::sort(active.begin(), active.end(), [](const auto& a, const auto& b) {
                return a.first != b.first ? a.first > b.first : a.second < b.second;
            });
            const std::size_t limit = active.empty() ? 0 :
                std::max<std::size_t>(1, static_cast<std::size_t>(
                    std::ceil(stream_fraction * static_cast<double>(active.size()))));
            for (std::size_t i = 0; i < std::min(limit, active.size()); ++i) {
                selected[active[i].second] = true;
            }
        }

        // Submit grouped resident consumers before streaming selected nonresidents.
        // Their leases remain held until the common compute stream has completed.
        if (cache) cache->begin_device_results(stream);
        stream_hits_submitted = true;
        std::array<std::size_t, 512> stream_route_counts{};
        for (std::size_t expert_id = 0; expert_id < streamed_routes.size(); ++expert_id)
            stream_route_counts[expert_id] = streamed_routes[expert_id].size();
        const auto stream_order = flash_next_stream_expert_order(
            stream_route_counts, resolve_stream_expert_busy_first());
        for (unsigned entry = 0; entry < stream_order.size; ++entry) {
            const std::size_t expert_id = stream_order.ids[entry];
            auto& routes = streamed_routes[expert_id];
            const bool use_gpu = prefill_stream_min_routes ?
                routes.size() >= prefill_stream_min_routes :
                (stream_fraction < 1.0 ? selected[expert_id] :
                 routes.size() >= expert_stream_min_routes);
            if (use_gpu) {
                continue;
            }
            const auto& indices = streamed_route_indices[expert_id];
            for (std::size_t i = 0; i < routes.size(); ++i) {
                const std::size_t route_index = indices[i];
                const std::size_t token_index = route_index / 10ULL;
                cpu.miss_routes.push_back(route_index);
                cpu.tasks.push_back(HostExpertTask{
                    .expert = host_experts.expert(static_cast<std::int32_t>(expert_id)),
                    .expert_id = static_cast<std::int32_t>(expert_id),
                    .route_id = static_cast<std::uint32_t>(route_index),
                    .input = use_routed_expert_input_fp32 ? nullptr :
                        host_input + token_index * kFlashNextExpertHidden,
                    .input_fp32 = use_routed_expert_input_fp32 ?
                        cpu.input_fp32.data() + token_index * kFlashNextExpertHidden : nullptr,
                    .output = nullptr,
                });
            }
        }
        // Fully classify misses and assign immutable output storage before starting
        // the CPU pool. Blocking ring submissions can otherwise defer all CPU work
        // until most streamed GPU work has already finished.
        if(early_cpu && !cpu.tasks.empty()) {
            device_route_host_outputs=cpu.ensure_pinned_pair_outputs(cpu.tasks.size()*kFlashNextExpertHidden);
            for(std::size_t i=0;i<cpu.tasks.size();++i)
                cpu.tasks[i].output=device_route_host_outputs+i*kFlashNextExpertHidden;
            auto* observer=v100_compare::active;
            cpu_started=measure ? Clock::now() : Clock::time_point{};
            pending_cpu=std::async(std::launch::async,[&,observer,grouped_cpu_experts] {
                // Until get(), the inference owner only submits CUDA/cache work;
                // it does not mutate the round. Retain the existing pool ledger.
                auto* previous=v100_compare::active;v100_compare::active=observer;
                try {
                    auto result=host_expert_worker_pool().run(cpu.tasks,grouped_cpu_experts,layer);
                    cpu_finished=measure ? Clock::now() : Clock::time_point{};
                    v100_compare::active=previous;
                    return result;
                } catch(...) {v100_compare::active=previous;throw;}
            });
        }
        stream_submit_started=measure ? Clock::now() : Clock::time_point{};
        for(unsigned entry=0;entry<stream_order.size;++entry) {
            const std::size_t expert_id=stream_order.ids[entry];
            auto& routes=streamed_routes[expert_id];
            const bool use_gpu=prefill_stream_min_routes ? routes.size()>=prefill_stream_min_routes :
                (stream_fraction<1.0 ? selected[expert_id] : routes.size()>=expert_stream_min_routes);
            if(!use_gpu)continue;
            expert_stream->submit(host_experts.expert(static_cast<std::int32_t>(expert_id)),routes,stream);
            streamed_route_count+=routes.size();++streamed_experts;
        }
        stream_submit_finished=measure ? Clock::now() : Clock::time_point{};
    }    } catch (...) {
        // Join before releasing caller-owned host buffers or round observation.
        if(pending_cpu.valid())pending_cpu.wait();
        if (stream_experts) {
            const auto failure = std::current_exception();
            // Host grouping/packing can fail after Ready consumers were leased.
            // Drain both owners before propagating the original failure.
            if (cache) {
                if (!stream_hits_submitted) cache->begin_device_results(stream);
                cache->finish_device_results(stream);
            }
            expert_stream->finish();
            std::rethrow_exception(failure);
        }
        throw;
    }
    if (device_route_combine && !cpu.tasks.empty() && !pending_cpu.valid()) {
        device_route_host_outputs = cpu.ensure_pinned_pair_outputs(
            cpu.tasks.size()*kFlashNextExpertHidden);
        for (std::size_t i=0;i<cpu.tasks.size();++i)
            cpu.tasks[i].output=device_route_host_outputs+i*kFlashNextExpertHidden;
    }

    if (cache != nullptr && !stream_experts) {
        if (device_route_combine) cache->begin_device_results(stream);
        else cache->begin_download(cpu.pair_outputs.size()*sizeof(float), stream);
    }
    double gpu_us = 0, wait_us = 0, result_copy_us = 0;
    const auto finish_hits = [&] {
        if (cache == nullptr) return;
        gpu_us = device_route_combine ? cache->finish_device_results(stream,&wait_us) :
            cache->finish_download(cpu.pair_outputs,stream,&wait_us,
                telemetry?&result_copy_us:nullptr);
    };
    // Serial is a diagnostic control. Production starts misses while hit kernels and
    // the pinned result transfer are already in flight, then joins only at the merge.
    const bool serial = cache != nullptr && cache->serial_schedule();
    if (serial) finish_hits();
    if(!pending_cpu.valid())cpu_started=measure ? Clock::now() : Clock::time_point{};
    HostExpertBatchStats cpu_batch;
    if (!cpu.tasks.empty()) {
        try {
            if(pending_cpu.valid())cpu_batch=pending_cpu.get();
            else {
                cpu_batch=host_expert_worker_pool().run(cpu.tasks,grouped_cpu_experts,layer);
                cpu_finished=measure ? Clock::now() : Clock::time_point{};
            }
            if (observe_host && !grouped_cpu_experts) {
                cpu_batch.weight_read_bytes = cpu.tasks.size() * kExpertPairBytes;
            }
        } catch (...) {
            if (!serial) finish_hits();
            if (stream_experts) expert_stream->finish();
            throw;
        }
    }
    if(cpu.tasks.empty())cpu_finished=measure ? Clock::now() : Clock::time_point{};
    if(observe_host && early_cpu_requested && !cpu.tasks.empty()) {
        const auto overlap_begin=std::max(cpu_started,stream_submit_started);
        const auto overlap_end=std::min(cpu_finished,stream_submit_finished);
        const auto ms=[](auto end,auto begin) {return std::chrono::duration<double,std::milli>(end-begin).count();};
        std::fprintf(stderr,"{\"kind\":\"early_cpu_stream\",\"layer\":%u,\"eligible\":%s,\"cpu_routes\":%zu,\"cpu_ms\":%.6f,\"stream_submit_ms\":%.6f,\"overlap_ms\":%.6f}\n",
            layer,early_cpu?"true":"false",cpu.tasks.size(),ms(cpu_finished,cpu_started),
            ms(stream_submit_finished,stream_submit_started),std::max(0.0,ms(overlap_end,overlap_begin)));
    }
    if (!serial) finish_hits();
    if (stream_experts && tokens >= 1024 && stream_diagnostics_enabled()) {
        compare_streamed_routes(layer, tokens, host_experts, streamed_routes, stream);
    }
    // Telemetry is already an opt-in synchronization path. Drain ephemeral streamed
    // misses here so its branch timing includes transfer and GPU expert execution.
    if (stream_experts && telemetry) expert_stream->finish();
    const double branch_wall_us = measure ? perf_elapsed_us(branch_started) : 0;
    if (measure && cache) cache->record_schedule(
        std::chrono::duration<double, std::micro>(cpu_finished-cpu_started).count(),
        gpu_us, wait_us,
        branch_wall_us);
    // Background admissions cannot affect this layer's already classified
    // routes. Streaming prefill normally disables admissions to avoid extra H2D
    // contention; this explicit opt-in chooses just one of this layer's hottest
    // nonresident experts to improve residency across repeated long prompts.
    if (cache != nullptr && !use_routed_expert_input_fp32 &&
        !resolve_fp32_intermediate_diagnostic()) {
        if (streamed_prefill_admit_hot) {
            std::array<std::size_t, 512> missed_counts{};
            for (std::size_t id = 0; id < streamed_routes.size(); ++id)
                missed_counts[id] = streamed_routes[id].size();
            const int candidate = flash_next_stream_prefill_admit_candidate(missed_counts);
            if (candidate >= 0) {
                const std::int32_t expert_id = candidate;
                cache->admit(layer, std::span(&expert_id, 1));
            }
        } else if (!stream_experts || !prefill) {
            cache->admit(layer, cpu.ids);
        }
    }

    if (!device_route_combine) {
        for (std::int32_t token = 0; token < tokens; ++token) {
            float* token_sum = cpu.routed_sum.data()+
                static_cast<std::size_t>(token)*kFlashNextExpertHidden;
            for (std::int32_t path = 0; path < 10; ++path) {
                const std::size_t route_index=static_cast<std::size_t>(token)*10ULL+path;
                const float* pair_output=cpu.pair_outputs.data()+
                    route_index*kFlashNextExpertHidden;
                const float alpha=cpu.alpha[route_index];
                for (std::size_t row=0;row<kFlashNextExpertHidden;++row)
                    token_sum[row]=std::fma(alpha,pair_output[row],token_sum[row]);
            }
        }
    }

    s_host_expert_layer_calls.fetch_add(1, std::memory_order_relaxed);
    s_host_expert_routed_tokens.fetch_add(
        static_cast<std::uint64_t>(tokens), std::memory_order_relaxed);
    s_host_expert_pairs.fetch_add(
        static_cast<std::uint64_t>(tokens) * 10ULL, std::memory_order_relaxed);

    // Each token owns 640 * 11 BF16 values (14,080 B). Store the 2,560 FP32 routed
    // values in the first 10,240 B of that token's slab and preserve the shared path at
    // BF16 path 10 (offset 12,800 B). A pitched copy keeps token slabs independent.
    constexpr std::size_t kActivationPitchBytes =
        kFlashNextExpertIntermediate * 11ULL * sizeof(std::uint16_t);
    constexpr std::size_t kRoutedBytesPerToken =
        kFlashNextExpertHidden * sizeof(float);
    static_assert(kActivationPitchBytes >= kRoutedBytesPerToken);
    if (telemetry) {
        if (!cpu.transfer_timing) cpu.transfer_timing = std::make_unique<HostMoeTransferTiming>();
        CUDA_CHECK(cudaEventRecord(cpu.transfer_timing->start, stream));
    }
    if (device_route_combine) {
        for (std::size_t begin=0;begin<cpu.miss_routes.size();) {
            std::size_t end=begin+1;
            while(end<cpu.miss_routes.size()&&
                  cpu.miss_routes[end]==cpu.miss_routes[end-1]+1)++end;
            const std::size_t first=cpu.miss_routes[begin];
            const std::size_t words=(end-begin)*kFlashNextExpertHidden;
            CUDA_CHECK(cudaMemcpyAsync(
                static_cast<float*>(scratch.down_intermediate.data)+
                    first*kFlashNextExpertHidden,
                device_route_host_outputs+begin*kFlashNextExpertHidden,
                words*sizeof(float),cudaMemcpyHostToDevice,stream));
            begin=end;
        }
    } else {
        CUDA_CHECK(cudaMemcpy2DAsync(
            scratch.activations.data, kActivationPitchBytes,
            cpu.routed_sum.data(), kRoutedBytesPerToken,
            kRoutedBytesPerToken, static_cast<std::size_t>(tokens),
            cudaMemcpyHostToDevice, stream));
    }
    if (telemetry) CUDA_CHECK(cudaEventRecord(cpu.transfer_timing->stop, stream));
    if (device_route_combine) {
        ninfer::ops::expert_route_combine(
            static_cast<const float*>(scratch.down_intermediate.data),
            static_cast<const float*>(scratch.alpha.data),
            static_cast<float*>(scratch.activations.data),tokens,
            kActivationPitchBytes/sizeof(float),stream);
    }
    flash_next_moe_host_routed_merge_launch(
        resident_weights, scratch, output, tokens, stream,
        use_shared_fp32_intermediate, output_fp32);
    if (output_fp32 != nullptr) {
        s_host_expert_fp32_output_calls.fetch_add(1, std::memory_order_relaxed);
    }
    if (aggregate) {
        using C = v100_compare::Counter;
        auto& round = *v100_compare::active;
        const auto add = [&](C key, std::uint64_t value) { round.counter(layer, key, value); };
        add(C::routed_tokens, tokens);
        add(C::total_routes, routed_paths);
        add(C::resident_routes, routed_paths - streamed_route_count - cpu.tasks.size());
        add(C::cpu_routes, cpu.tasks.size());
        add(C::nonresident_gpu_routes, streamed_route_count);
        add(C::distinct_experts, ExpertRouteHistogram(cpu.ids).distinct);
        add(C::resident_distinct_experts, resident_ids.count());
        add(C::distinct_missed_experts, missed_ids.count());
        add(C::cpu_groups, cpu_batch.groups);
        add(C::cpu_grouped_routes, cpu_batch.grouped_pairs);
        add(C::cpu_weight_read_bytes, cpu_batch.weight_read_bytes);
        add(C::expert_h2d_bytes, stream_experts ? streamed_experts * kExpertSlotBytes : 0);
        add(C::route_d2h_bytes,
            ((stream_experts && !stream_cpu_fallback) ? 0 :
                input_words*(use_routed_expert_input_fp32 ? sizeof(float) : sizeof(std::uint16_t)))
            + routed_paths*(sizeof(std::int32_t)+(device_route_combine?0:sizeof(float))));
        const auto hits = routed_paths - streamed_route_count - cpu.tasks.size();
        add(C::output_d2h_bytes, device_route_combine ? 0 :
            (hits ? cpu.pair_outputs.size()*sizeof(float) : 0));
        add(C::output_h2d_bytes, device_route_combine ?
            cpu.tasks.size()*kFlashNextExpertHidden*sizeof(float) : cpu.routed_sum.size()*sizeof(float));
        if (cache_begin)
            round.cache(layer, std::move(*cache_begin), cache->telemetry_snapshot(layer, v100_compare::level() >= 2));
        round.duration(layer, v100_compare::Stage::host_wait, rendezvous_us);
        round.duration(layer, v100_compare::Stage::cpu_expert,
            std::chrono::duration<double, std::micro>(cpu_finished-cpu_started).count());
    }
    if (telemetry) {
        ExpertLayerMeasurement m;
        m.layer = layer;
        m.prefill = prefill;
        m.tokens = tokens;
        m.gpu_hit_routes = routed_paths - streamed_route_count - cpu.tasks.size();
        m.cpu_miss_routes = cpu.tasks.size();
        m.cpu_groups = cpu_batch.groups;
        m.cpu_grouped_pairs = cpu_batch.grouped_pairs;
        m.cpu_weight_read_bytes = cpu_batch.weight_read_bytes;
        m.stream_routes = streamed_route_count;
        m.stream_experts = streamed_experts;
        m.stream_expert_h2d_bytes = stream_experts ?
            streamed_experts * kExpertSlotBytes : 0;
        m.cache_result_d2h_bytes = device_route_combine ? 0 :
            (m.gpu_hit_routes ? cpu.pair_outputs.size()*sizeof(float) : 0);
        m.routed_sum_h2d_bytes = device_route_combine ? 0 :
            cpu.routed_sum.size()*sizeof(float);
        m.cpu_miss_h2d_bytes = device_route_combine ?
            m.cpu_miss_routes*kFlashNextExpertHidden*sizeof(float) : 0;
        m.route_input_d2h_bytes =
            ((stream_experts && !stream_cpu_fallback) ? 0 :
                input_words*(use_routed_expert_input_fp32 ? sizeof(float) : sizeof(std::uint16_t)))
            + routed_paths*(sizeof(std::int32_t)+(device_route_combine?0:sizeof(float)));
        m.route_handoff = route_handoff;
        m.route_sequence = route_ticket;
        m.router_rendezvous_us = rendezvous_us;
        m.cpu_branch_us = std::chrono::duration<double, std::micro>(cpu_finished-cpu_started).count();
        m.gpu_hit_window_us = gpu_us;
        m.cache_result_d2h_us = result_copy_us;
        CUDA_CHECK(cudaEventSynchronize(cpu.transfer_timing->stop));
        float routed_copy_ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&routed_copy_ms, cpu.transfer_timing->start, cpu.transfer_timing->stop));
        m.routed_sum_timing = true;
        m.routed_sum_h2d_us = device_route_combine ? 0 : double(routed_copy_ms)*1000;
        m.cpu_miss_h2d_us = device_route_combine ? double(routed_copy_ms)*1000 : 0;
        m.merge_wait_us = wait_us;
        m.branch_wall_us = branch_wall_us;
        m.cache_timing = cache && cache->timing_enabled();
        m.overlap_lower_bound_us = std::max(0.0, m.cpu_branch_us + gpu_us - branch_wall_us);
        m.cache_present = cache != nullptr;
        if (cache) {
            const auto snapshot = cache->layer_snapshot(layer);
            m.ready_experts = snapshot.ready;
            m.uploading_experts = snapshot.uploading;
            m.leased_experts = snapshot.leased;
            m.cache_hits_total = snapshot.totals.hits;
            m.cache_misses_total = snapshot.totals.misses;
            m.admissions_total = snapshot.totals.admitted;
            m.fills_total = snapshot.totals.ready;
            m.evictions_total = snapshot.totals.evicted;
            m.expert_staging_bytes_total = snapshot.totals.fill_bytes;
            m.expert_staging_us_total = snapshot.totals.h2d_us;
            m.cache_bytes = cache->budget().cache_bytes;
            m.cache_transfer_budget_bytes = cache->budget().transfer_bytes;
        }
        emit_perf_json(m.json(ExpertRouteHistogram(cpu.ids)));
    }
}

void flash_next_moe_bf16(const Tensor& input, const MoeBf16Weights& weights, Tensor& output,
                         WorkspaceArena& workspace, cudaStream_t stream) {
    const std::int32_t tokens = input.ne[1];
    if (input.dtype != DType::BF16 || output.dtype != DType::BF16 || input.ne[0] != 2'560 ||
        output.ne[0] != 2'560 || tokens < 1 || output.ne[1] != tokens ||
        input.ne[2] != 1 || input.ne[3] != 1 || output.ne[2] != 1 || output.ne[3] != 1 ||
        !input.is_contiguous() || !output.is_contiguous() || !aligned_to(input.data, 16) ||
        !aligned_to(output.data, 16) || !exact_bf16_weight(weights.router, 512, 2'560) ||
        !exact_bf16_weight(weights.shared_down, 2'560, 640) ||
        !exact_bf16_weight(weights.shared_gate, 640, 2'560) ||
        !exact_bf16_weight(weights.shared_up, 640, 2'560) ||
        !exact_bf16_weight(weights.shared_gate_weight, 1, 2'560) ||
        !exact_bf16_expert_bank(weights.expert_gate_up, 1'280, 2'560) ||
        !exact_bf16_expert_bank(weights.expert_down, 2'560, 640) || stream == nullptr) {
        throw std::invalid_argument("Flash-Next BF16 MoE received an invalid exact target view");
    }
    const auto scope              = workspace.scope();
    FlashNextMoeWorkspace scratch = allocate_flash_next_moe_workspace(workspace, tokens);
    flash_next_route(input, weights.router, weights.shared_gate_weight, scratch.scores, scratch.ids,
                     scratch.alpha, scratch.shared_scale, stream);
    flash_next_moe_bf16_kernels_launch(input, weights, scratch, output, stream);
}

} // namespace ninfer::targets::qwen3_8_flash_next::detail
