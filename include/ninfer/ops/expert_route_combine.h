#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

// Combine represented FP32 routes [T][10][2560] with FP32 alpha [T][10].
// The mathematical output is sum(path=0..9, alpha[token,path]*route[token,path,row]).
// This implementation accumulates ten round-to-nearest FP32 FMAs in ascending
// path order, starting at +0, matching the host-backed MoE execution profile.
// output_pitch is in float elements and may include the existing shared-expert slab.
// Inputs and destination must be device-resident, nonoverlapping, and live until
// stream completion. No allocation, host rendezvous, or state is retained.
void expert_route_combine(const float* routes, const float* alpha, float* output,
                          std::int32_t tokens, std::size_t output_pitch,
                          cudaStream_t stream);

} // namespace ninfer::ops
