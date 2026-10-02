#include "targets/qwen3_8_flash_next/impl/cpu_expert_pool.h"
#include <stdexcept>
#include <algorithm>

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
void HostExpertWorkerPool::execute_jobs(std::size_t count) {
    work_count_ = count;
    next_.store(0, std::memory_order_relaxed);
    remaining_.store(count, std::memory_order_release);
    work_.release(static_cast<std::ptrdiff_t>(count));
    std::unique_lock<std::mutex> done_lock(done_mutex_);
    done_cv_.wait(done_lock, [this] {
        return remaining_.load(std::memory_order_acquire) == 0;
    });
}

void HostExpertWorkerPool::run(std::span<const HostExpertTask> tasks) {
    if (tasks.empty()) { return; }
    if (tasks.size() > 1'048'576ULL) {
        throw std::invalid_argument("host expert batch exceeds worker semaphore capacity");
    }
    std::unique_lock<std::mutex> submit_lock(submit_mutex_);
    tasks_ = tasks.data();
    {
        std::lock_guard<std::mutex> error_lock(error_mutex_);
        error_ = nullptr;
    }
    row_sharded_ = avx2_ && tasks.size() < workers_.size() &&
        std::none_of(tasks.begin(), tasks.end(), [](const auto& task) {
            return task.input_fp32 != nullptr;
        });
    if (row_sharded_) {
        batch_scratch_.resize(tasks.size());
        row_jobs_.clear();
        for (std::size_t i = 0; i < tasks.size(); ++i) {
            flash_next_cpu_nvfp4_expert_prepare_avx2(
                tasks[i].expert, {tasks[i].input, kFlashNextExpertHidden}, batch_scratch_[i]);
            // Fill the available cores without adding a mostly idle second wave.
            // Bound sharding for a lone expert to keep rendezvous work modest.
            const std::size_t shards = std::min<std::size_t>(8,
                workers_.size() / tasks.size() + (i < workers_.size() % tasks.size()));
            for (std::size_t shard = 0; shard < shards; ++shard) {
                row_jobs_.push_back({i, shard, shards});
            }
        }
        down_phase_ = false;
        execute_jobs(row_jobs_.size());
        {
            std::lock_guard<std::mutex> error_lock(error_mutex_);
            if (error_) { std::rethrow_exception(error_); }
        }
        down_phase_ = true;
        execute_jobs(row_jobs_.size());
    } else {
        execute_jobs(tasks.size());
    }
    std::exception_ptr error;
    {
        std::lock_guard<std::mutex> error_lock(error_mutex_);
        error = error_;
    }
    tasks_ = nullptr;
    work_count_ = 0;
    if (error) { std::rethrow_exception(error); }
}

void HostExpertWorkerPool::worker_loop() {
        CpuNvfp4ExpertReferenceScratch scratch{};
        for (;;) {
            work_.acquire();
            if (stop_.load(std::memory_order_acquire)) { return; }

            const std::size_t index =
                next_.fetch_add(1, std::memory_order_relaxed);
            if (index >= work_count_) {
                // One semaphore permit is released per task, so this is a hard
                // invariant unless the pool state was corrupted.
                std::terminate();
            }

            try {
                const HostExpertTask& task =
                    tasks_[row_sharded_ ? row_jobs_[index].task : index];
                if (row_sharded_) {
                    const auto& job = row_jobs_[index];
                    auto& shared = batch_scratch_[job.task];
                    const std::size_t rows = down_phase_ ? kFlashNextExpertHidden : kFlashNextExpertIntermediate;
                    const std::size_t begin = rows * job.shard / job.shards;
                    const std::size_t end = rows * (job.shard + 1) / job.shards;
                    if (down_phase_) {
                        flash_next_cpu_nvfp4_expert_down_rows_avx2(
                            task.expert, shared, {task.output, kFlashNextExpertHidden}, begin, end);
                    } else {
                        flash_next_cpu_nvfp4_expert_gate_up_rows_avx2(task.expert, shared, begin, end);
                    }
                } else if (task.input_fp32 != nullptr) {
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
