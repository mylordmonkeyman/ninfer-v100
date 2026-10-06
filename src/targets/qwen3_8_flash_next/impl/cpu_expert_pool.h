#pragma once
#include "targets/qwen3_8_flash_next/impl/cpu_expert_reference.h"
#include <atomic>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <semaphore>
#include <span>
#include <thread>
#include <vector>

namespace ninfer::targets::qwen3_8_flash_next::detail {
struct HostExpertTask {
    HostNvfp4ExpertPairView expert{};
    std::int32_t expert_id = -1;
    std::uint32_t route_id = 0;
    const std::uint16_t* input = nullptr;
    const float* input_fp32 = nullptr;
    float* output = nullptr;
};

// Several routed activations sharing one expert. Singleton remainders dispatch
// the established single-token kernel; only widths 2..4 reuse decoded weights.
struct HostExpertTaskGroup {
    HostNvfp4ExpertPairView expert{};
    std::array<const std::uint16_t*, kFlashNextCpuExpertGroupMax> inputs{};
    std::array<float*, kFlashNextCpuExpertGroupMax> outputs{};
    std::array<std::uint32_t, kFlashNextCpuExpertGroupMax> route_ids{};
    std::size_t token_count = 0;
};

struct HostExpertBatchStats {
    std::uint64_t groups = 0, grouped_pairs = 0, weight_read_bytes = 0;
};

// Persistent expert-level workers shared by serving and routed-miss replay.
// Outputs retain task order; workers never apply routing weights or change storage.
class HostExpertWorkerPool {
  public:
    HostExpertWorkerPool(unsigned workers, bool avx2, bool fp32_intermediate = false);
    ~HostExpertWorkerPool();
    HostExpertWorkerPool(const HostExpertWorkerPool&) = delete;
    HostExpertWorkerPool& operator=(const HostExpertWorkerPool&) = delete;
    HostExpertBatchStats run(std::span<const HostExpertTask> tasks, bool group_same_experts = false);
  private:
    void worker_loop();
    void stop_workers() noexcept;
    void execute_jobs(std::size_t count);
    struct RowJob {
        std::size_t task, shard, shards;
    };
    std::vector<RowJob> row_jobs_;
    std::vector<CpuNvfp4ExpertReferenceScratch> batch_scratch_;
    void run_grouped(std::span<const HostExpertTask> tasks, HostExpertBatchStats& stats);
    std::vector<HostExpertTaskGroup> groups_;
    std::array<std::vector<std::size_t>, 512> group_indices_;
    std::vector<CpuNvfp4ExpertGroupScratch> group_scratch_;
    bool grouped_ = false;
    bool row_sharded_ = false;
    bool down_phase_ = false;
    bool avx2_;
    bool fp32_intermediate_;
    std::vector<std::thread> workers_;
    std::counting_semaphore<1'048'576> work_{0};
    std::atomic<bool> stop_{false};
    std::atomic<std::size_t> next_{0};
    std::atomic<std::size_t> remaining_{0};
    const HostExpertTask* tasks_ = nullptr;
    std::size_t work_count_ = 0;
    std::mutex submit_mutex_;
    std::mutex done_mutex_;
    std::condition_variable done_cv_;
    std::mutex error_mutex_;
    std::exception_ptr error_;
};
} // namespace ninfer::targets::qwen3_8_flash_next::detail
