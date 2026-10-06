#include "targets/qwen3_8_flash_next/impl/cpu_expert_pool.h"
#include <algorithm>
#include <stdexcept>

namespace ninfer::targets::qwen3_8_flash_next::detail {
namespace {
std::uint64_t compact_expert_bytes(const HostNvfp4ExpertPairView& expert) {
    const auto matrix_bytes = [](const Nvfp4ExpertMatrixView& matrix) {
        const auto elements = static_cast<std::uint64_t>(matrix.rows) * matrix.columns;
        return elements / 2ULL + elements / 16ULL + sizeof(float);
    };
    return matrix_bytes(expert.gate_up) + matrix_bytes(expert.down);
}
} // namespace

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

HostExpertBatchStats HostExpertWorkerPool::run(std::span<const HostExpertTask> tasks,
                                               bool group_same_experts) {
    HostExpertBatchStats stats;
    if (tasks.empty()) { return stats; }
    if (group_same_experts &&
        (!avx2_ || fp32_intermediate_ ||
         std::any_of(tasks.begin(), tasks.end(), [](const auto& task) {
             return task.input_fp32 != nullptr;
         }))) {
        throw std::invalid_argument("grouped CPU experts require the BF16-input AVX2 profile");
    }
    if (group_same_experts && std::any_of(tasks.begin(), tasks.end(), [](const auto& task) {
            return task.expert_id < 0 || task.expert_id >= 512;
        })) {
        throw std::invalid_argument("grouped CPU expert task has invalid expert id");
    }
    if (group_same_experts && std::any_of(tasks.begin(), tasks.end(), [](const auto& task) {
            return task.input == nullptr || task.output == nullptr;
        })) {
        throw std::invalid_argument("grouped CPU expert task has null storage");
    }
    if (tasks.size() > 1'048'576ULL) {
        throw std::invalid_argument("host expert batch exceeds worker semaphore capacity");
    }
    std::unique_lock<std::mutex> submit_lock(submit_mutex_);
    tasks_ = tasks.data();
    {
        std::lock_guard<std::mutex> error_lock(error_mutex_);
        error_ = nullptr;
    }
    grouped_ = group_same_experts;
    if (grouped_) {
        run_grouped(tasks, stats);
    } else {
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
                // Fill available cores without adding a mostly idle second wave.
                // Bound sharding for a lone expert to keep rendezvous work modest.
                const std::size_t shards = std::min<std::size_t>(
                    8, workers_.size() / tasks.size() + (i < workers_.size() % tasks.size()));
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
    }
    std::exception_ptr error;
    {
        std::lock_guard<std::mutex> error_lock(error_mutex_);
        error = error_;
    }
    tasks_ = nullptr;
    work_count_ = 0;
    if (error) { std::rethrow_exception(error); }
    return stats;
}

void HostExpertWorkerPool::run_grouped(std::span<const HostExpertTask> tasks,
                                       HostExpertBatchStats& stats) {
    for (auto& indices : group_indices_) { indices.clear(); }
    for (std::size_t i = 0; i < tasks.size(); ++i) {
        group_indices_[static_cast<std::size_t>(tasks[i].expert_id)].push_back(i);
    }
    groups_.clear();
    for (const auto& indices : group_indices_) {
        for (std::size_t begin = 0; begin < indices.size();
             begin += kFlashNextCpuExpertGroupMax) {
            HostExpertTaskGroup group;
            group.expert = tasks[indices[begin]].expert;
            group.token_count =
                std::min(kFlashNextCpuExpertGroupMax, indices.size() - begin);
            for (std::size_t token = 0; token < group.token_count; ++token) {
                const auto i = indices[begin + token];
                group.inputs[token] = tasks[i].input;
                group.outputs[token] = tasks[i].output;
                group.route_ids[token] = tasks[i].route_id;
            }
            stats.groups += group.token_count > 1;
            if (group.token_count > 1) { stats.grouped_pairs += group.token_count; }
            groups_.push_back(group);
        }
    }
    for (const auto& group : groups_) {
        stats.weight_read_bytes += compact_expert_bytes(group.expert);
    }
    row_sharded_ = groups_.size() < workers_.size();
    if (!row_sharded_) {
        execute_jobs(groups_.size());
        return;
    }
    // A width-four group performs four independent dot products per row.
    // Equal shards per group leave those jobs on the critical path while
    // singleton workers idle, especially in MTP verification batches. Split
    // the available worker budget by routed rows rather than expert count.
    group_shards_.assign(groups_.size(), 1);
    for (std::size_t jobs = groups_.size(); jobs < workers_.size(); ++jobs) {
        std::size_t selected = groups_.size();
        for (std::size_t i = 0; i < groups_.size(); ++i) {
            if (group_shards_[i] == 8) { continue; }
            if (selected == groups_.size() ||
                groups_[i].token_count * group_shards_[selected] >
                    groups_[selected].token_count * group_shards_[i]) {
                selected = i;
            }
        }
        if (selected == groups_.size()) { break; }
        ++group_shards_[selected];
    }
    group_scratch_.resize(groups_.size());
    row_jobs_.clear();
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        const auto& group = groups_[i];
        if (group.token_count == 1) {
            flash_next_cpu_nvfp4_expert_prepare_avx2(
                group.expert, {group.inputs[0], kFlashNextExpertHidden},
                group_scratch_[i].tokens[0]);
        } else {
            flash_next_cpu_nvfp4_expert_group_prepare_avx2(
                group.expert, {group.inputs.data(), group.token_count}, group_scratch_[i]);
        }
        const auto shards = group_shards_[i];
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
}

void HostExpertWorkerPool::worker_loop() {
    CpuNvfp4ExpertReferenceScratch scratch{};
    CpuNvfp4ExpertGroupScratch group_scratch{};
    for (;;) {
        work_.acquire();
        if (stop_.load(std::memory_order_acquire)) { return; }

        const std::size_t index = next_.fetch_add(1, std::memory_order_relaxed);
        if (index >= work_count_) {
            // One semaphore permit is released per task, so this is a hard
            // invariant unless the pool state was corrupted.
            std::terminate();
        }

        try {
            if (grouped_) {
                const auto group_index = row_sharded_ ? row_jobs_[index].task : index;
                const auto& group = groups_[group_index];
                auto& shared = row_sharded_ ? group_scratch_[group_index] : group_scratch;
                if (row_sharded_) {
                    const auto& job = row_jobs_[index];
                    const auto rows =
                        down_phase_ ? kFlashNextExpertHidden : kFlashNextExpertIntermediate;
                    const auto begin = rows * job.shard / job.shards;
                    const auto end = rows * (job.shard + 1) / job.shards;
                    if (group.token_count == 1) {
                        if (down_phase_) {
                            flash_next_cpu_nvfp4_expert_down_rows_avx2(
                                group.expert, shared.tokens[0],
                                {group.outputs[0], kFlashNextExpertHidden}, begin, end);
                        } else {
                            flash_next_cpu_nvfp4_expert_gate_up_rows_avx2(
                                group.expert, shared.tokens[0], begin, end);
                        }
                    } else {
                        if (down_phase_) {
                            flash_next_cpu_nvfp4_expert_group_down_rows_avx2(
                                group.expert, shared,
                                {group.outputs.data(), group.token_count}, begin, end);
                        } else {
                            flash_next_cpu_nvfp4_expert_group_gate_up_rows_avx2(
                                group.expert, shared, group.token_count, begin, end);
                        }
                    }
                } else if (group.token_count == 1) {
                    flash_next_cpu_nvfp4_expert_pair_avx2(
                        group.expert, {group.inputs[0], kFlashNextExpertHidden},
                        {group.outputs[0], kFlashNextExpertHidden}, scratch);
                } else {
                    flash_next_cpu_nvfp4_expert_group_avx2(
                        group.expert, {group.inputs.data(), group.token_count},
                        {group.outputs.data(), group.token_count}, shared);
                }
            } else {
                const HostExpertTask& task =
                    tasks_[row_sharded_ ? row_jobs_[index].task : index];
                if (row_sharded_) {
                    const auto& job = row_jobs_[index];
                    auto& shared = batch_scratch_[job.task];
                    const std::size_t rows =
                        down_phase_ ? kFlashNextExpertHidden : kFlashNextExpertIntermediate;
                    const std::size_t begin = rows * job.shard / job.shards;
                    const std::size_t end = rows * (job.shard + 1) / job.shards;
                    if (down_phase_) {
                        flash_next_cpu_nvfp4_expert_down_rows_avx2(
                            task.expert, shared, {task.output, kFlashNextExpertHidden}, begin, end);
                    } else {
                        flash_next_cpu_nvfp4_expert_gate_up_rows_avx2(
                            task.expert, shared, begin, end);
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
