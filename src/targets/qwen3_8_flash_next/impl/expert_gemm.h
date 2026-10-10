#pragma once
#include "targets/qwen3_8_flash_next/impl/expert_cache.h"
#include <memory>

namespace ninfer::targets::qwen3_8_flash_next::detail {
// One reusable expert expansion and a fixed token tile; compact cache storage is
// unchanged. All launches on this instance belong to the same compute stream.
class FlashNextExpertGemm {
public:
    FlashNextExpertGemm();
    ~FlashNextExpertGemm();
    FlashNextExpertGemm(const FlashNextExpertGemm&) = delete;
    FlashNextExpertGemm& operator=(const FlashNextExpertGemm&) = delete;
    static constexpr unsigned tile_routes = 256;
    static constexpr std::size_t device_bytes =
        (1280*2560+2560*640)*2ULL + tile_routes*(2560*2ULL+1280*4ULL+640*2ULL+2560*4ULL+8ULL)
        + 4ULL*1024*1024;
    void launch(const HostNvfp4ExpertPairView& device_expert,
                const FlashNextCachedExpertGroup* device_groups,
                unsigned routes, cudaStream_t stream);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
[[nodiscard]] bool flash_next_expert_gemm_requested();
} // namespace ninfer::targets::qwen3_8_flash_next::detail
