#include "ops/linear_attention/gated_delta_net/volta/launch.h"

#include "ops/linear_attention/gated_delta_net/volta/schedule.cuh"
#include "ops/linear_attention/gated_delta_net/volta/shared_tiles.cuh"

#include <cuda_runtime.h>

#include <mutex>

namespace ninfer::ops::detail::gated_delta_net::volta {
namespace {

RuntimeResources g_resources{};
bool g_initialized = false;
std::mutex g_runtime_mutex;

template <int DV_TILE>
__global__ __launch_bounds__(GdnSchedule<DV_TILE>::kThreads,
                             GdnSchedule<DV_TILE>::kMinBlocksPerSm)
void resource_probe_kernel() {}

template <int DV_TILE>
cudaError_t configure_probe(KernelResources& out) {
    constexpr int kThreads = GdnSchedule<DV_TILE>::kThreads;
    constexpr std::size_t kDynamicSmem = FusedSharedLayout<DV_TILE>::Bytes;

    cudaError_t status = cudaFuncSetAttribute(
        resource_probe_kernel<DV_TILE>, cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxShared);
    if (status != cudaSuccess) { return status; }

    cudaFuncAttributes attrs{};
    status = cudaFuncGetAttributes(&attrs, resource_probe_kernel<DV_TILE>);
    if (status != cudaSuccess) { return status; }

    int active_blocks = 0;
    status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks, resource_probe_kernel<DV_TILE>, kThreads, kDynamicSmem);
    if (status != cudaSuccess) { return status; }

    out.registers_per_thread = attrs.numRegs;
    out.active_blocks_per_sm = active_blocks;
    out.dynamic_smem = kDynamicSmem;

    // This scaffold records the intended launch/resource envelope. Qualification
    // remains false until the corresponding fused kernel exists and its actual
    // attributes, spill behavior, and physical-V100 residency are measured.
    return cudaSuccess;
}

} // namespace

cudaError_t initialize_runtime() {
    std::lock_guard<std::mutex> lock(g_runtime_mutex);
    if (g_initialized) { return cudaSuccess; }

    RuntimeResources resources{};
    cudaError_t status = configure_probe<16>(resources.dv16);
    if (status != cudaSuccess) { return status; }

    status = configure_probe<32>(resources.dv32);
    if (status != cudaSuccess) { return status; }

    resources.dv16_qualified = false;
    resources.dv32_qualified = false;

    g_resources = resources;
    g_initialized = true;
    return cudaSuccess;
}

bool runtime_initialized() noexcept {
    std::lock_guard<std::mutex> lock(g_runtime_mutex);
    return g_initialized;
}

const RuntimeResources& runtime_resources() noexcept { return g_resources; }

cudaError_t ensure_runtime_initialized(cudaStream_t stream) {
    if (runtime_initialized()) { return cudaSuccess; }

    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    cudaError_t status = cudaStreamIsCapturing(stream, &capture_status);
    if (status != cudaSuccess) { return status; }
    if (capture_status != cudaStreamCaptureStatusNone) {
        return cudaErrorStreamCaptureUnsupported;
    }

    return initialize_runtime();
}

} // namespace ninfer::ops::detail::gated_delta_net::volta
