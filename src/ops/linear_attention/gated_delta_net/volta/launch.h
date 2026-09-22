#pragma once

#if !defined(NINFER_VOLTA_BUILD)
#error "GDN Volta backend is SM70-only"
#endif

#include "ops/linear_attention/gated_delta_net/volta/common.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail::gated_delta_net::volta {

enum class Backend : std::uint8_t {
    Auto,
    Recurrent,
    Dv16,
    Dv32,
    GroupedDv16Experimental,
};

enum class SolveMode : std::uint8_t {
    Exact,
    Neumann2Experimental,
};

struct KernelResources {
    int registers_per_thread = 0;
    int active_blocks_per_sm = 0;
    std::size_t dynamic_smem = 0;
};

struct RuntimeResources {
    KernelResources dv16;
    KernelResources dv32;
    bool dv16_qualified = false;
    bool dv32_qualified = false;
};

cudaError_t initialize_runtime();

[[nodiscard]] bool runtime_initialized() noexcept;

const RuntimeResources& runtime_resources() noexcept;

cudaError_t ensure_runtime_initialized(cudaStream_t stream);

[[nodiscard]] constexpr Backend automatic_backend() noexcept {
    return Backend::Recurrent;
}

} // namespace ninfer::ops::detail::gated_delta_net::volta
