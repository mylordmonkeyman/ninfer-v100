#pragma once

#include "core/arena.h"
#include "targets/qwen3_8_flash_next/impl/expert_bank.h"
#include <cuda_runtime.h>
#include <array>
#include <condition_variable>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <span>
#include <thread>
#include <vector>

namespace ninfer::targets::qwen3_8_flash_next::detail {

struct FlashNextExpertCacheBudget {
    std::size_t total_bytes = 0, free_before_bytes = 0, transfer_bytes = 0;
    std::size_t reserve_bytes = 0, used_limit_bytes = 0, cache_bytes = 0;
    unsigned slots_per_layer = 0;
};
// Exact canonical payload plus alignment; no expanded FP16 weight residency.
inline constexpr std::size_t kExpertPairBytes = 2'764'808;
inline constexpr std::size_t kExpertSlotBytes = (kExpertPairBytes + 255) & ~std::size_t(255);
[[nodiscard]] FlashNextExpertCacheBudget flash_next_expert_cache_budget(
    std::size_t free_bytes, std::size_t total_bytes, std::size_t transfer_bytes,
    bool mtp, unsigned maximum_slots = 512);

struct FlashNextExpertCacheStats {
    std::uint64_t hits = 0, misses = 0, admitted = 0, ready = 0, evicted = 0;
    double fill_wall_us = 0, maximum_fill_wall_us = 0;
    std::uint64_t schedule_calls = 0;
    std::uint64_t fill_bytes = 0, queue_declined_calls = 0, victim_declined_calls = 0;
    unsigned maximum_outstanding = 0;
    std::array<std::uint64_t, 3> admission_bursts{};
    double admission_wall_us = 0, pack_wall_us = 0, h2d_us = 0;
    double cpu_branch_us = 0, gpu_branch_us = 0, merge_wait_us = 0;
    double branch_wall_us = 0, overlap_lower_bound_us = 0;
    double hit_submission_us = 0;
    std::uint64_t grouped_tasks = 0, grouped_groups = 0;
    std::uint64_t hit_kernel_launches = 0;
};

struct FlashNextCachedExpertTask {
    HostNvfp4ExpertPairView expert;
    const void* input;
    void* activation;
    float* output;
};

struct FlashNextCachedExpertGroup {
    FlashNextCachedExpertTask tasks[4];
    unsigned count;
};

// Program-owned main-text cache. A separate nonblocking stream and a bounded worker queue
// populate canonical slots; only completed fills are eligible for execution. MTP is not cached.
// Single inference owner calls execute/download/admit; the fill worker never evicts a lease.
class FlashNextExpertCache {
public:
    FlashNextExpertCache(const HostNvfp4ExpertTableView& host, unsigned max_tokens,
                         bool mtp, unsigned maximum_slots = 512, unsigned admission_cap = 1);
    ~FlashNextExpertCache();
    FlashNextExpertCache(const FlashNextExpertCache&) = delete;
    FlashNextExpertCache& operator=(const FlashNextExpertCache&) = delete;
    // Layer owner selects batched submission only for explicit prefill execution.
    void begin_layer(bool prefill);
    [[nodiscard]] bool batched_prefill() const { return batched_prefill_; }
    void set_batched_prefill(bool value) { batched_prefill_ = value; }
    // Phase-17 launch-fusion experiment. Decode remains scalar by default.
    [[nodiscard]] bool batched_decode() const { return batched_decode_; }
    void set_batched_decode(bool value) { batched_decode_ = value; }
    [[nodiscard]] bool grouped_prefill() const { return grouped_prefill_; }
    void set_grouped_prefill(bool value) { grouped_prefill_ = value; }
    bool execute(unsigned layer, int expert, const void* device_input,
                 unsigned path, cudaStream_t stream);
    // Completes hit consumers and releases their slot leases before any eviction.
    void download(std::span<float> pair_outputs, cudaStream_t stream);
    // Enqueue pinned result transfer before CPU misses; wait/copy/release only at merge.
    void begin_download(std::size_t output_bytes, cudaStream_t stream);
    double finish_download(std::span<float> pair_outputs, cudaStream_t stream,
                           double* wait_us = nullptr, double* result_copy_us = nullptr);
    void record_schedule(double cpu_us, double gpu_us, double wait_us, double wall_us);
    [[nodiscard]] bool timing_enabled() const { return timing_enabled_; }
    [[nodiscard]] bool serial_schedule() const { return serial_schedule_; }
    [[nodiscard]] bool prefill_enabled() const { return prefill_enabled_; }
    void set_prefill_enabled(bool value) { prefill_enabled_ = value; }
    // Diagnostic controls for paired scheduling comparisons with one fixed Ready set.
    void set_serial_schedule(bool value) { serial_schedule_ = value; }
    void freeze_admissions() { drain(); admissions_enabled_ = false; }
    // Diagnostic replay boundary: no GPU consumers may be outstanding.
    void reset();
    [[nodiscard]] unsigned admission_cap() const { return admission_cap_; }
    // At most admission_cap new experts per layer call; pool pressure declines admission.
    void admit(unsigned layer, std::span<const std::int32_t> ids);
    void drain(); // test/shutdown boundary, never used by the current-token fill path
    struct LayerSnapshot {
        unsigned ready = 0, uploading = 0, leased = 0;
        FlashNextExpertCacheStats totals;
    };
    // Observability only; snapshot does not drain fills or change slot leases.
    [[nodiscard]] LayerSnapshot layer_snapshot(unsigned layer) const;
    [[nodiscard]] FlashNextExpertCacheStats stats() const;
    [[nodiscard]] const FlashNextExpertCacheBudget& budget() const { return budget_; }
    [[nodiscard]] HostNvfp4ExpertPairView ready_view(unsigned layer, int expert);
private:
    enum class State { Empty, Uploading, Canonical, Ready };
    struct Entry { int expert = -1; State state = State::Empty;
                   std::uint64_t epoch = 0; unsigned leases = 0; };
    HostNvfp4ExpertTableView host_;
    FlashNextExpertCacheBudget budget_;
    std::unique_ptr<DeviceBuffer> storage_, activations_, outputs_, batch_tasks_;
    std::unique_ptr<PinnedHostBuffer> fill_buffer_, result_buffer_, batch_descriptors_, group_descriptors_;
    cudaStream_t fill_stream_ = nullptr;
    cudaEvent_t hit_start_ = nullptr, hit_stop_ = nullptr, result_copy_start_ = nullptr;
    cudaEvent_t fill_start_ = nullptr, fill_stop_ = nullptr;
    unsigned admission_cap_ = 1;
    bool prefill_enabled_ = true, batched_prefill_ = false, batched_decode_ = false;
    bool batching_layer_ = false;
    bool grouped_prefill_ = true, grouping_layer_ = false;
    bool timing_enabled_ = false, serial_schedule_ = false, admissions_enabled_ = true;
    int device_ = 0;
    std::vector<Entry> entries_;
    std::vector<std::pair<unsigned, unsigned>> consumers_;
    std::deque<unsigned> queue_;
    mutable std::mutex mutex_;
    std::condition_variable work_, idle_;
    std::thread worker_;
    std::exception_ptr failure_;
    bool stop_ = false, filling_ = false;
    std::uint64_t epoch_ = 0;
    FlashNextExpertCacheStats stats_;
    HostNvfp4ExpertPairView view(unsigned slot) const;
    void fill_loop() noexcept;
    void check_failure() const;
};

void flash_next_cached_expert_launch(const HostNvfp4ExpertPairView& device_expert,
    const void* input_bf16, void* intermediate_bf16, float* output_fp32, cudaStream_t stream);
void flash_next_cached_expert_batch_launch(const FlashNextCachedExpertTask* tasks,
    unsigned count, cudaStream_t stream);
void flash_next_cached_expert_group_launch(const FlashNextCachedExpertGroup* groups,
    unsigned count, cudaStream_t stream);
} // namespace ninfer::targets::qwen3_8_flash_next::detail
