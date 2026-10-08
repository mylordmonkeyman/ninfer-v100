#pragma once

#include "core/tensor.h"
#include "core/arena.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// Keep the candidate domain separate from the measured production crossover so the two kernel
// families remain directly comparable in the benchmark overlap.
inline constexpr std::int32_t kBf16SmallTMinTokens         = 2;
inline constexpr std::int32_t kBf16SmallTMaxTokens         = 32;
inline constexpr std::int32_t kBf16LinearSmallTDispatchEnd = 27;

using Bf16Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

void launch_bf16_decode(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream);
void launch_bf16_small_t(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream);
void launch_bf16_mma(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream);
void launch_bf16_n256_k5120(const Tensor& x, const Weight& weight, Tensor& out,
                            cudaStream_t stream);
#ifdef NINFER_VOLTA_BUILD
void launch_bf16_volta_simt(const Tensor& x, const Weight& weight, Tensor& out,
                            cudaStream_t stream);
bool bf16_volta_fp16_tc_enabled();
bool bf16_volta_fp16_tc_supported(int n, int k, int t) noexcept;
std::size_t bf16_volta_fp16_tc_workspace_bytes(int n, int k, int t);
void launch_bf16_volta_fp16_tc(const Tensor& x, const Weight& weight, Tensor& out,
                               WorkspaceArena& workspace, cudaStream_t stream);
#endif

} // namespace ninfer::ops::detail

namespace ninfer::ops::detail {
void launch_bf16_cutlass_sm70(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
}
