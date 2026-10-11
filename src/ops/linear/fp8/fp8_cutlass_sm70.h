#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

[[nodiscard]] std::size_t fp8_cutlass_sm70_workspace_bytes(std::int32_t n, std::int32_t k,
                                                            std::int32_t cols);

// Row-scaled E4M3 Weight x BF16 activations -> contiguous BF16 [n, cols]. Intended for wide-T
// Volta routes; narrow-T paths should keep the packed QPN/GEMV implementations.
void fp8_cutlass_sm70_launch(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                             cudaStream_t stream);

// Experimental F32-row-scale operator, directly qualified before dispatch integration.
// Inputs must remain finite when converted to FP16; no production selector uses this yet.
[[nodiscard]] std::size_t fp8_f32_cutlass_sm70_workspace_bytes(int n, int k, int t);
void fp8_f32_cutlass_sm70_launch(const Tensor& x, const Weight& w, Tensor& out,
                                WorkspaceArena& ws, cudaStream_t stream);

} // namespace ninfer::ops::detail
