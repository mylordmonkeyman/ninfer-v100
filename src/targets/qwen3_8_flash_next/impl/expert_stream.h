#pragma once
#include "targets/qwen3_8_flash_next/impl/expert_cache.h"
#include "targets/qwen3_8_flash_next/impl/expert_gemm.h"

namespace ninfer::targets::qwen3_8_flash_next::detail {
[[nodiscard]] bool flash_next_expert_stream_requested();
[[nodiscard]] bool flash_next_decode_expert_stream_requested();
[[nodiscard]] unsigned flash_next_decode_expert_stream_min_routes();
[[nodiscard]] std::size_t flash_next_expert_stream_device_bytes(unsigned maximum_routes);
[[nodiscard]] unsigned flash_next_expert_stream_ring_slots();
[[nodiscard]] bool flash_next_expert_stream_pipeline_reuse();

struct FlashNextStreamRoute {
    const void* input_bf16;
    float* output_fp32;
};

// Explicitly owned, bounded scratch for nonresident prefill groups. No persistent
// cache admission/eviction. Caller retains input/output until finish() returns.
class FlashNextExpertStream {
public:
    explicit FlashNextExpertStream(unsigned maximum_routes_per_expert);
    ~FlashNextExpertStream();
    FlashNextExpertStream(const FlashNextExpertStream&) = delete;
    FlashNextExpertStream& operator=(const FlashNextExpertStream&) = delete;
    void submit(const HostNvfp4ExpertPairView& expert,
                std::span<const FlashNextStreamRoute> routes, cudaStream_t compute);
    void finish();
    [[nodiscard]] std::size_t device_bytes() const { return device_bytes_; }
    [[nodiscard]] std::size_t pinned_bytes() const { return pinned_bytes_; }
    [[nodiscard]] std::uint64_t submitted_experts() const { return submitted_; }
    [[nodiscard]] std::uint64_t expert_h2d_bytes() const { return submitted_ * kExpertSlotBytes; }
private:
    struct Slot {
        std::unique_ptr<DeviceBuffer> weights, activations, groups;
        std::unique_ptr<PinnedHostBuffer> host_weights, host_groups;
        cudaEvent_t ready = nullptr, consumed = nullptr;
        bool pending = false;
        cudaEvent_t copy_start = nullptr, wait_start = nullptr, kernel_start = nullptr;
        bool timing_pending = false;
    };
    std::unique_ptr<FlashNextExpertGemm> gemm_;
    std::array<Slot,8> slots_;  // Up to eight bounded in-flight H2D/compute slots.
    cudaStream_t transfer_ = nullptr;
    unsigned maximum_routes_, next_ = 0, ring_slots_ = 4;
    bool pipeline_reuse_ = false, timing_ = false;
    double copy_ms_ = 0, wait_ms_ = 0, kernel_ms_ = 0;
    std::uint64_t timed_experts_ = 0, timed_routes_ = 0;
    void collect_timing(Slot& slot);
    std::size_t device_bytes_ = 0, pinned_bytes_ = 0;
    std::uint64_t submitted_ = 0;
    void cleanup() noexcept;
};
} // namespace ninfer::targets::qwen3_8_flash_next::detail
