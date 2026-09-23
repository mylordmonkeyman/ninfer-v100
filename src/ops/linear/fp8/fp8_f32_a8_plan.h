#pragma once

#include "core/arena.h"
#include "core/layout.h"
#include "core/tensor.h"
#include "ops/linear/fp8/fp8_a8_plan.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::ops::detail {

struct Fp8F32A8Workspace {
    std::uint8_t* codes           = nullptr;
    float* scales                 = nullptr;
    float* split_workspace        = nullptr;
};

inline constexpr int kFlashNextResidualSplitK = 4;

template <class Arena>
Fp8F32A8Workspace allocate_fp8_f32_a8_workspace(Arena& arena, std::int32_t output_rows,
                                                std::int32_t input_rows, std::int32_t tokens) {
    if (input_rows <= 0 || (input_rows % 32) != 0) {
        throw std::invalid_argument("fp8 F32 A8 workspace: invalid K");
    }
    if (output_rows <= 0 || tokens <= 0) {
        throw std::invalid_argument("fp8 F32 A8 workspace: invalid N or T");
    }
    const DeviceSpan codes =
        arena.alloc_bytes(fp8_a8_checked_bytes(tokens, static_cast<std::size_t>(input_rows)), 256);
    const DeviceSpan scales = arena.alloc_bytes(fp8_a8_checked_bytes(tokens, sizeof(float)), 256);

    float* split_ws = nullptr;
    if (output_rows == 2560 && input_rows == 6144) {
        const std::size_t split_bytes = static_cast<std::size_t>(kFlashNextResidualSplitK) *
                                        output_rows * tokens * sizeof(float);
        const DeviceSpan split_span = arena.alloc_bytes(split_bytes, 256);
        split_ws                    = static_cast<float*>(split_span.data);
    }

    return {static_cast<std::uint8_t*>(codes.data), static_cast<float*>(scales.data), split_ws};
}

inline std::size_t fp8_f32_a8_workspace_capacity_bytes(std::int32_t output_rows,
                                                       std::int32_t input_rows,
                                                       std::int32_t tokens) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_fp8_f32_a8_workspace(layout, output_rows, input_rows, tokens);
    return layout.peak_bytes(1);
}

void launch_fp8_f32_a8(const Tensor& x, const Weight& weight, Tensor& out,
                       WorkspaceArena& workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail
