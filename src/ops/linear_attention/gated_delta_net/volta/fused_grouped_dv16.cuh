#pragma once

#if !defined(NINFER_VOLTA_BUILD)
#error "GDN Volta backend is SM70-only"
#endif

#include "ops/linear_attention/gated_delta_net/volta/fused_state_output.cuh"

#include <cuda_runtime.h>

namespace ninfer::ops::detail::gated_delta_net::volta {

cudaError_t launch_fused_grouped_dv16(const FusedOneWindowArgs& args,
                                      cudaStream_t stream);

#if defined(__CUDACC__)
__global__ void fused_grouped_dv16_kernel(FusedOneWindowArgs args);
#endif

} // namespace ninfer::ops::detail::gated_delta_net::volta
