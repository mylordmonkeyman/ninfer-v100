#pragma once

#include "core/device.h"

#include <string_view>

namespace ninfer::ops::detail {

// Issue #22 qualifies the combined raster/scale-fetch change on the desktop 5090.
// The RTX PRO 6000 regresses, despite sharing its compute capability.
constexpr bool nvfp4_tma_use_token_fast(std::string_view name, int major, int minor) {
    return major == 12 && minor == 0 && name == "NVIDIA GeForce RTX 5090";
}

inline bool nvfp4_tma_use_token_fast() {
    // The product uses one fixed GPU per process, as do the TMA descriptor and
    // kernel-attribute caches. Query once before launching/capturing these Ops.
    static const bool enabled = [] {
        int device = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
        return nvfp4_tma_use_token_fast(properties.name, properties.major, properties.minor);
    }();
    return enabled;
}

} // namespace ninfer::ops::detail
