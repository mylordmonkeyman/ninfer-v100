#include "targets/qwen3_8_flash_next/impl/cpu_expert_pool.h"
#include <stdexcept>

namespace ninfer::targets::qwen3_8_flash_next::detail {
HostExpertWorkerPool::HostExpertWorkerPool(unsigned workers, bool avx2, bool fp32_intermediate)
    : avx2_(avx2), fp32_intermediate_(fp32_intermediate) {
    if (workers == 0 || workers > 256) {
        throw std::invalid_argument("CPU expert workers must be in [1, 256]");
    }
    if (avx2 && (!flash_next_cpu_nvfp4_avx2_available() || fp32_intermediate)) {
        throw std::invalid_argument("AVX2/FMA unavailable or incompatible with FP32 diagnostic");
    }
    workers_.reserve(workers);
    try {
        for (unsigned worker = 0; worker < workers; ++worker) {
            workers_.emplace_back([this] { worker_loop(); });
        }
    } catch (...) {
        stop_workers();
        throw;
    }
}
HostExpertWorkerPool::~HostExpertWorkerPool() { stop_workers(); }
void HostExpertWorkerPool::stop_workers() noexcept {
    stop_.store(true, std::memory_order_release);
    work_.release(static_cast<std::ptrdiff_t>(workers_.size()));
    for (auto& worker : workers_) {
        if (worker.joinable()) { worker.join(); }
    }
}
void HostExpertWorkerPool::run(std::span<const HostExpertTask> tasks) {
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

void HostExpertWorkerPool::worker_loop() {
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

            {
                // Pair completion with the wait mutex to prevent lost wakeups.
                std::lock_guard<std::mutex> done_lock(done_mutex_);
                if (remaining_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                    done_cv_.notify_one();
                }
            }
        }
    }


} // namespace ninfer::targets::qwen3_8_flash_next::detail
