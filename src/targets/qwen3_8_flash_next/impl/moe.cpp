#include "targets/qwen3_8_flash_next/impl/moe.h"

#include "core/layout.h"
#include "targets/qwen3_8_flash_next/impl/cpu_expert_reference.h"
#include "targets/qwen3_8_flash_next/impl/moe_kernels.h"
#include "targets/qwen3_8_flash_next/impl/moe_route.h"
#include "targets/qwen3_8_flash_next/impl/moe_workspace.h"
#include "targets/qwen3_8_flash_next/impl/stage_ledger.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <semaphore>
#include <span>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <vector>
#include <condition_variable>

#include "core/device.h"

namespace ninfer::targets::qwen3_8_flash_next::detail {
namespace {

std::atomic<std::uint64_t> s_host_expert_layer_calls{0};
std::atomic<std::uint64_t> s_host_expert_routed_tokens{0};
std::atomic<std::uint64_t> s_host_expert_pairs{0};

struct HostExpertTask {
    HostNvfp4ExpertPairView expert{};
    const std::uint16_t* input = nullptr;
    const float* input_fp32 = nullptr;
    float* output = nullptr;
};

unsigned resolve_host_expert_worker_count() {
    constexpr unsigned kDefaultWorkers = 32;
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

bool resolve_avx2_backend() {
    const char* env = std::getenv("NINFER_FLASH_NEXT_CPU_EXPERT_BACKEND");
    if (env == nullptr || env[0] == '\0' || std::string_view(env) == "reference") {
        return false;
    }
    if (std::string_view(env) == "avx2") {
        if (!flash_next_cpu_nvfp4_avx2_available()) {
            throw std::runtime_error(
                "NINFER_FLASH_NEXT_CPU_EXPERT_BACKEND=avx2 requested without AVX2/FMA");
        }
        return true;
    }
    throw std::invalid_argument(
        "NINFER_FLASH_NEXT_CPU_EXPERT_BACKEND must be reference or avx2");
}

class HostExpertWorkerPool {
  public:
    HostExpertWorkerPool()
        : avx2_(resolve_avx2_backend()),
          fp32_intermediate_(resolve_fp32_intermediate_diagnostic()),
          worker_count_(resolve_host_expert_worker_count()) {
        if (avx2_ && fp32_intermediate_) {
            throw std::invalid_argument(
                "FP32 expert-intermediate diagnostic requires reference backend");
        }
        workers_.reserve(worker_count_);
        for (unsigned worker = 0; worker < worker_count_; ++worker) {
            workers_.emplace_back([this] { worker_loop(); });
        }
        std::fprintf(stderr, "flash_next host_expert_backend=%s workers=%u\n",
                     avx2_ ? "avx2_fma_parallel"
                           : (fp32_intermediate_
                                  ? "scalar_reference_fp32_intermediate_parallel"
                                  : "scalar_reference_parallel"),
                     worker_count_);
    }

    HostExpertWorkerPool(const HostExpertWorkerPool&) = delete;
    HostExpertWorkerPool& operator=(const HostExpertWorkerPool&) = delete;

    ~HostExpertWorkerPool() {
        stop_.store(true, std::memory_order_release);
        if (!workers_.empty()) {
            work_.release(static_cast<std::ptrdiff_t>(workers_.size()));
        }
        for (std::thread& worker : workers_) {
            if (worker.joinable()) { worker.join(); }
        }
    }

    void run(std::span<const HostExpertTask> tasks) {
        if (tasks.empty()) { return; }
        if (workers_.empty()) {
            throw std::runtime_error("host expert worker pool has no workers");
        }
        if (tasks.size() > 1'048'576ULL) {
            throw std::invalid_argument("host expert batch exceeds worker semaphore capacity");
        }

        std::unique_lock<std::mutex> submit_lock(submit_mutex_);
        tasks_ = tasks.data();
        task_count_ = tasks.size();
        next_.store(0, std::memory_order_relaxed);
        remaining_.store(tasks.size(), std::memory_order_release);
        {
            std::lock_guard<std::mutex> error_lock(error_mutex_);
            error_ = nullptr;
        }

        work_.release(static_cast<std::ptrdiff_t>(tasks.size()));
        {
            std::unique_lock<std::mutex> done_lock(done_mutex_);
            done_cv_.wait(done_lock, [this] {
                return remaining_.load(std::memory_order_acquire) == 0;
            });
        }

        std::exception_ptr error;
        {
            std::lock_guard<std::mutex> error_lock(error_mutex_);
            error = error_;
        }
        tasks_ = nullptr;
        task_count_ = 0;
        if (error) { std::rethrow_exception(error); }
    }

  private:
    void worker_loop() {
        CpuNvfp4ExpertReferenceScratch scratch{};
        for (;;) {
            work_.acquire();
            if (stop_.load(std::memory_order_acquire)) { return; }

            const std::size_t index =
                next_.fetch_add(1, std::memory_order_relaxed);
            if (index >= task_count_) {
                // One semaphore permit is released per task, so this is a hard
                // invariant unless the pool state was corrupted.
                std::terminate();
            }

            try {
                const HostExpertTask& task = tasks_[index];
                if (task.input_fp32 != nullptr) {
                    flash_next_cpu_nvfp4_expert_pair_reference_fp32_input(
                        task.expert,
                        std::span<const float>(task.input_fp32, kFlashNextExpertHidden),
                        std::span<float>(task.output, kFlashNextExpertHidden), scratch);
                } else if (avx2_) {
                    flash_next_cpu_nvfp4_expert_pair_avx2(
                        task.expert,
                        std::span<const std::uint16_t>(task.input, kFlashNextExpertHidden),
                        std::span<float>(task.output, kFlashNextExpertHidden), scratch);
                } else if (fp32_intermediate_) {
                    flash_next_cpu_nvfp4_expert_pair_reference_fp32_intermediate(
                        task.expert,
                        std::span<const std::uint16_t>(task.input, kFlashNextExpertHidden),
                        std::span<float>(task.output, kFlashNextExpertHidden), scratch);
                } else {
                    flash_next_cpu_nvfp4_expert_pair_reference(
                        task.expert,
                        std::span<const std::uint16_t>(task.input, kFlashNextExpertHidden),
                        std::span<float>(task.output, kFlashNextExpertHidden), scratch);
                }
            } catch (...) {
                std::lock_guard<std::mutex> error_lock(error_mutex_);
                if (!error_) { error_ = std::current_exception(); }
            }

            if (remaining_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                done_cv_.notify_one();
            }
        }
    }

    bool avx2_ = false;
    bool fp32_intermediate_ = false;
    unsigned worker_count_ = 0;
    std::vector<std::thread> workers_;
    std::counting_semaphore<1'048'576> work_{0};
    std::atomic<bool> stop_{false};
    std::atomic<std::size_t> next_{0};
    std::atomic<std::size_t> remaining_{0};
    const HostExpertTask* tasks_ = nullptr;
    std::size_t task_count_ = 0;
    std::mutex submit_mutex_;
    std::mutex done_mutex_;
    std::condition_variable done_cv_;
    std::mutex error_mutex_;
    std::exception_ptr error_;
};

HostExpertWorkerPool& host_expert_worker_pool() {
    static HostExpertWorkerPool pool;
    return pool;
}

struct HostMoeCpuBuffers {
    std::vector<std::uint16_t> input;
    std::vector<float> input_fp32;
    std::vector<std::int32_t> ids;
    std::vector<float> alpha;
    std::vector<float> routed_sum;
    std::vector<float> pair_outputs;
    std::vector<HostExpertTask> tasks;
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
    s_host_expert_routed_tokens.store(0, std::memory_order_relaxed);
    s_host_expert_pairs.store(0, std::memory_order_relaxed);
}

FlashNextHostExpertExecutionStats
flash_next_host_expert_execution_stats() noexcept {
    return FlashNextHostExpertExecutionStats{
        .completed_layer_calls =
            s_host_expert_layer_calls.load(std::memory_order_relaxed),
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
                                const Tensor* routed_expert_input_fp32) {
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

    const auto scope = workspace.scope();
    FlashNextMoeWorkspace scratch = allocate_flash_next_moe_workspace(workspace, tokens);

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
    stage_ledger_record(stream, FlashNextStageId::MoE_Router);
    if (emit) {
        emit("moe_router_scores", scratch.scores);
        emit("moe_router_ids", scratch.ids);
        emit("moe_router_alpha", scratch.alpha);
        emit("moe_shared_scale", scratch.shared_scale);
    }

    // The shared expert remains resident on device. Compute its BF16 activation before the
    // host rendezvous. The shared down projection is deferred until the routed FP32 sum returns
    // so both branches can be combined before the final BF16 rounding.
    flash_next_moe_host_shared_launch(input, resident_weights, scratch, stream);

    const std::size_t input_words =
        static_cast<std::size_t>(tokens) * kFlashNextExpertHidden;
    const std::size_t routed_paths = static_cast<std::size_t>(tokens) * 10ULL;
    HostMoeCpuBuffers& cpu = s_host_moe_buffers;
    if (use_routed_expert_input_fp32) {
        cpu.input_fp32.resize(input_words);
    } else {
        cpu.input.resize(input_words);
    }
    cpu.ids.resize(routed_paths);
    cpu.alpha.resize(routed_paths);
    cpu.routed_sum.resize(
        static_cast<std::size_t>(tokens) * kFlashNextExpertHidden);
    cpu.pair_outputs.resize(routed_paths * kFlashNextExpertHidden);
    cpu.tasks.resize(routed_paths);
    std::fill(cpu.routed_sum.begin(), cpu.routed_sum.end(), 0.0F);

    if (use_routed_expert_input_fp32) {
        CUDA_CHECK(cudaMemcpyAsync(
            cpu.input_fp32.data(), routed_expert_input_fp32->data,
            input_words * sizeof(float), cudaMemcpyDeviceToHost, stream));
    } else {
        CUDA_CHECK(cudaMemcpyAsync(cpu.input.data(), input.data,
                                   input_words * sizeof(std::uint16_t),
                                   cudaMemcpyDeviceToHost, stream));
    }
    CUDA_CHECK(cudaMemcpyAsync(cpu.ids.data(), scratch.ids.data,
                               routed_paths * sizeof(std::int32_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(cpu.alpha.data(), scratch.alpha.data,
                               routed_paths * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Independent routed expert pairs are computed concurrently. Each task writes
    // a private FP32 vector. Routing alpha is then accumulated below on this thread
    // in the original token/path order so the reduction contract remains deterministic.
    for (std::int32_t token = 0; token < tokens; ++token) {
        const std::size_t token_offset =
            static_cast<std::size_t>(token) * kFlashNextExpertHidden;
        const std::uint16_t* token_input =
            use_routed_expert_input_fp32 ? nullptr : cpu.input.data() + token_offset;
        const float* token_input_fp32 =
            use_routed_expert_input_fp32 ? cpu.input_fp32.data() + token_offset : nullptr;
        for (std::int32_t path = 0; path < 10; ++path) {
            const std::size_t route_index =
                static_cast<std::size_t>(token) * 10ULL +
                static_cast<std::size_t>(path);
            cpu.tasks[route_index] = HostExpertTask{
                .expert = host_experts.expert(cpu.ids[route_index]),
                .input = token_input,
                .input_fp32 = token_input_fp32,
                .output = cpu.pair_outputs.data() +
                          route_index * kFlashNextExpertHidden,
            };
        }
    }

    host_expert_worker_pool().run(cpu.tasks);

    for (std::int32_t token = 0; token < tokens; ++token) {
        float* token_sum =
            cpu.routed_sum.data() +
            static_cast<std::size_t>(token) * kFlashNextExpertHidden;
        for (std::int32_t path = 0; path < 10; ++path) {
            const std::size_t route_index =
                static_cast<std::size_t>(token) * 10ULL +
                static_cast<std::size_t>(path);
            const float* pair_output =
                cpu.pair_outputs.data() + route_index * kFlashNextExpertHidden;
            const float alpha = cpu.alpha[route_index];
            for (std::size_t row = 0; row < kFlashNextExpertHidden; ++row) {
                token_sum[row] =
                    std::fma(alpha, pair_output[row], token_sum[row]);
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
    CUDA_CHECK(cudaMemcpy2DAsync(
        scratch.activations.data, kActivationPitchBytes,
        cpu.routed_sum.data(), kRoutedBytesPerToken,
        kRoutedBytesPerToken, static_cast<std::size_t>(tokens),
        cudaMemcpyHostToDevice, stream));
    flash_next_moe_host_routed_merge_launch(
        resident_weights, scratch, output, tokens, stream);
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
