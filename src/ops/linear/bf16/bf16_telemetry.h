#pragma once
#include "ops/linear/bf16/bf16_launch.h"
#include <functional>
#include <vector>

namespace ninfer::ops::detail {
// Optional eager-execution collector. Event pairs are reused, and no host wait occurs
// between projection launches. Captured/replayed CUDA graphs are excluded.
class Bf16TimingCollector {
public:
    Bf16TimingCollector() = default;
    ~Bf16TimingCollector();
    Bf16TimingCollector(const Bf16TimingCollector&) = delete;
    Bf16TimingCollector& operator=(const Bf16TimingCollector&) = delete;
    void begin() noexcept { count_ = 0; }
    void launch(Bf16Launch implementation, const Tensor& x, const Weight& weight,
                Tensor& output, cudaStream_t stream);
    void finish(const std::function<void(int, int, int, const char*, double)>& emit);
private:
    struct Entry {
        cudaEvent_t start = nullptr, stop = nullptr;
        int n = 0, k = 0, t = 0;
        const char* implementation = nullptr;
    };
    std::vector<Entry> entries_;
    std::size_t count_ = 0;
};
inline thread_local Bf16TimingCollector* active_bf16_timing = nullptr;
class ScopedBf16Timing {
public:
    explicit ScopedBf16Timing(Bf16TimingCollector& collector)
        : previous_(active_bf16_timing) {
        collector.begin();
        active_bf16_timing = &collector;
    }
    ~ScopedBf16Timing() { active_bf16_timing = previous_; }
    ScopedBf16Timing(const ScopedBf16Timing&) = delete;
    ScopedBf16Timing& operator=(const ScopedBf16Timing&) = delete;
private:
    Bf16TimingCollector* previous_;
};
} // namespace ninfer::ops::detail
