#pragma once

#include "core/arena.h"
#include "targets/qwen3_8_flash_next/impl/expert_bank.h"
#include <cuda_runtime.h>
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
};

// Program-owned main-text cache. A separate nonblocking stream and a bounded worker queue
// populate canonical slots; only completed fills are eligible for execution. MTP is not cached.
// Single inference owner calls execute/download/admit; the fill worker never evicts a lease.
class FlashNextExpertCache {
public:
    FlashNextExpertCache(const HostNvfp4ExpertTableView& host, unsigned max_tokens,
                         bool mtp, unsigned maximum_slots = 512);
    ~FlashNextExpertCache();
    FlashNextExpertCache(const FlashNextExpertCache&) = delete;
    FlashNextExpertCache& operator=(const FlashNextExpertCache&) = delete;
    bool execute(unsigned layer, int expert, const void* device_input,
                 unsigned path, cudaStream_t stream);
    // Completes hit consumers and releases their slot leases before any eviction.
    void download(std::span<float> pair_outputs, cudaStream_t stream);
    // At most one new expert per layer call; a busy fill pool simply declines admission.
    void admit(unsigned layer, std::span<const std::int32_t> ids);
    void drain(); // test/shutdown boundary, never used by the current-token fill path
    [[nodiscard]] FlashNextExpertCacheStats stats() const;
    [[nodiscard]] const FlashNextExpertCacheBudget& budget() const { return budget_; }
    [[nodiscard]] HostNvfp4ExpertPairView ready_view(unsigned layer, int expert);
private:
    enum class State { Empty, Uploading, Canonical, Ready };
    struct Entry { int expert = -1; State state = State::Empty;
                   std::uint64_t epoch = 0; unsigned leases = 0; };
    HostNvfp4ExpertTableView host_;
    FlashNextExpertCacheBudget budget_;
    std::unique_ptr<DeviceBuffer> storage_, activations_, outputs_;
    std::unique_ptr<PinnedHostBuffer> fill_buffer_, result_buffer_;
    cudaStream_t fill_stream_ = nullptr;
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
} // namespace ninfer::targets::qwen3_8_flash_next::detail
