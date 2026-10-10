#pragma once
#include "targets/qwen3_8_flash_next/impl/expert_cache.h"
#include <memory>

namespace ninfer::targets::qwen3_8_flash_next::detail {
struct FlashNextExpandedExpertView {
    const void* gate_up;
    const void* down;
};
// Fixed token scratch on one compute stream. Resident launches own reusable
// expansion; streamed launches can consume caller-owned ring-slot expansion.
class FlashNextExpertGemm {
public:
    explicit FlashNextExpertGemm(bool own_weight_expansion = true);
    ~FlashNextExpertGemm();
    FlashNextExpertGemm(const FlashNextExpertGemm&) = delete;
    FlashNextExpertGemm& operator=(const FlashNextExpertGemm&) = delete;
    static constexpr unsigned tile_routes = 256;
    static constexpr std::size_t device_bytes =
        (1280*2560+2560*640)*2ULL + tile_routes*(2560*2ULL+1280*4ULL+640*2ULL+2560*4ULL+8ULL)
        + 4ULL*1024*1024;
    static constexpr std::size_t expanded_weight_bytes = (1280*2560+2560*640)*2ULL;
    static constexpr std::size_t compute_bytes = device_bytes - expanded_weight_bytes;
    // Caller owns expansion buffers until every prepared consumer completes.
    static void expand_weights(const HostNvfp4ExpertPairView& device_expert,
                               void* gate_up_fp16, void* down_fp16, cudaStream_t stream);
    void launch_prepared(const HostNvfp4ExpertPairView& device_expert,
                         FlashNextExpandedExpertView weights,
                         const FlashNextCachedExpertGroup* device_groups,
                         unsigned routes, cudaStream_t stream);
    void launch(const HostNvfp4ExpertPairView& device_expert,
                const FlashNextCachedExpertGroup* device_groups,
                unsigned routes, cudaStream_t stream);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
[[nodiscard]] bool flash_next_expert_gemm_requested();
// These selectors default on only for the Volta build. Their environment
// variables remain explicit rollback controls (simt/0) and opt-in controls on
// other architectures (fp16/1).
[[nodiscard]] bool flash_next_resident_expert_gemm_requested();
[[nodiscard]] bool flash_next_cpu_stream_overlap_requested();
} // namespace ninfer::targets::qwen3_8_flash_next::detail
