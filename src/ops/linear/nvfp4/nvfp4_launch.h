#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

void launch_nvfp4_decode(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream);
void launch_nvfp4_small_t(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream);

#ifdef NINFER_VOLTA_BUILD
inline constexpr std::int32_t kNvfp4VoltaQpnRowsPerTile = 8;
inline constexpr std::int32_t kNvfp4VoltaQpnMaxTokens   = 32;
void launch_nvfp4_volta_qpn(const Tensor& x, const Weight& weight, Tensor& out,
                            cudaStream_t stream);
[[nodiscard]] bool nvfp4_volta_qpn_supported(std::int32_t n, std::int32_t k,
                                              std::int32_t t) noexcept;
#endif

} // namespace ninfer::ops::detail
