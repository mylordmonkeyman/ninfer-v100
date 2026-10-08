// Experimental SV7 Volta (SM70) FP16 tensor-core GEMM for BF16 control matrices.
// No default dispatch changes. Converted finite operands use RNE and clamp only
// FP16 overflows; if ANY NaN/Inf appears, fall back to original BF16 SIMT.
// This initial correctness-first pilot synchronizes to inspect the nonfinite
// sentinel and repacks each call; measure these costs before considering
// prequalified/persistent FP16 images or production promotion.
#include "ops/linear/bf16/bf16_launch.h"
#include "core/device.h"
#include "core/layout.h"

#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/half.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

using ElementInput = cutlass::half_t;
using ElementOutput = cutlass::bfloat16_t;
using Gemm = cutlass::gemm::device::Gemm<
    ElementInput, cutlass::layout::RowMajor,
    ElementInput, cutlass::layout::ColumnMajor,
    ElementOutput, cutlass::layout::RowMajor,
    float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm70,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<8, 8, 4>,
    cutlass::epilogue::thread::LinearCombination<
        ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
        float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

__global__ void bf16_to_clamped_fp16(
    const __nv_bfloat16* __restrict__ input,
    cutlass::half_t* __restrict__ output, std::int64_t count,
    unsigned* __restrict__ nonfinite) {
    const std::int64_t index =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float v = __bfloat162float(input[index]);
    if (!isfinite(v)) {
        atomicOr(nonfinite, 1U);
        output[index] = cutlass::half_t(0.0f);
        return;
    }
    const float capped = fminf(fmaxf(v, -65504.0f), 65504.0f);
    output[index] = cutlass::half_t(capped);
}

template <typename Allocator>
struct Scratch {
    Tensor weights, inputs;
    DeviceSpan nonfinite, gemm;
};

std::size_t gemm_workspace_size(int n, int k, int t) {
    const cutlass::gemm::GemmCoord shape(t, n, k);
    const typename Gemm::Arguments args{
        shape, {nullptr, k}, {nullptr, k},
        {nullptr, n}, {nullptr, n}, {1.0F, 0.0F}, 1};
    return Gemm::get_workspace_size(args);
}

template <typename Allocator>
Scratch<Allocator> allocate_scratch(Allocator& a, int n, int k, int t) {
    Scratch<Allocator> result;
    result.weights = a.alloc(DType::FP16, {k, n});
    result.inputs = a.alloc(DType::FP16, {k, t});
    result.nonfinite = a.alloc_bytes(sizeof(unsigned));
    const auto bytes = gemm_workspace_size(n, k, t);
    if (bytes) result.gemm = a.alloc_bytes(bytes);
    return result;
}

} // namespace

bool bf16_volta_fp16_tc_enabled() {
    const char* raw = std::getenv("NINFER_V100_SV7_FP16_TC");
    if (!raw || !raw[0] || std::strcmp(raw, "0") == 0) return false;
    if (std::strcmp(raw, "1") == 0) return true;
    throw std::invalid_argument("NINFER_V100_SV7_FP16_TC must be 0 or 1");
}

bool bf16_volta_fp16_tc_supported(int n, int k, int t) noexcept {
    if (t < 128 || t > 2048) return false; // Experimental prefill only.
    return (n == 640 && k == 2560) ||     // QSA indexer
           (n == 10240 && k == 2560) ||   // PLE key
           (n == 2560 && k == 2560) ||    // PLE value
           (n == 2560 && k == 640);      // Shared expert down
}

std::size_t bf16_volta_fp16_tc_workspace_bytes(int n, int k, int t) {
    if (!bf16_volta_fp16_tc_supported(n, k, t)) return 0;
    WorkspaceLayoutBuilder layout;
    (void)allocate_scratch(layout, n, k, t);
    return layout.peak_bytes(1);
}

void launch_bf16_volta_fp16_tc(const Tensor& x, const Weight& w, Tensor& out,
                                WorkspaceArena& workspace, cudaStream_t stream) {
    const int n = w.n, k = w.k, t = x.ne[1];
    if (!bf16_volta_fp16_tc_supported(n, k, t) ||
        x.dtype != DType::BF16 || out.dtype != DType::BF16 ||
        w.qtype != QType::BF16_CTRL) {
        throw std::invalid_argument("SV7 FP16 TC unsupported matrix or format");
    }
    // Capture cannot include a host synchronization: use the baseline.
    cudaStreamCaptureStatus status_capture;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &status_capture));
    if (status_capture != cudaStreamCaptureStatusNone) {
        launch_bf16_volta_simt(x, w, out, stream);
        return;
    }
    auto scope = workspace.scope();
    Scratch<WorkspaceArena> scratch = allocate_scratch(workspace, n, k, t);
    auto* ww = static_cast<cutlass::half_t*>(scratch.weights.data);
    auto* xx = static_cast<cutlass::half_t*>(scratch.inputs.data);
    auto* bad = static_cast<unsigned*>(scratch.nonfinite.data);
    CUDA_CHECK(cudaMemsetAsync(bad, 0, sizeof(unsigned), stream));

    const std::int64_t nw = static_cast<std::int64_t>(n) * k;
    const std::int64_t nx = static_cast<std::int64_t>(t) * k;
    bf16_to_clamped_fp16<<<static_cast<unsigned>((nw + 255) / 256), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(w.qdata), ww, nw, bad);
    CUDA_CHECK(cudaGetLastError());
    bf16_to_clamped_fp16<<<static_cast<unsigned>((nx + 255) / 256), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), xx, nx, bad);
    CUDA_CHECK(cudaGetLastError());

    unsigned host_nonfinite = 0;
    CUDA_CHECK(cudaMemcpyAsync(&host_nonfinite, bad, sizeof(unsigned),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (host_nonfinite) {
        // Do not silently reinterpret NaN/Inf. BF16 source tensors were unchanged.
        launch_bf16_volta_simt(x, w, out, stream);
        return;
    }

    const cutlass::gemm::GemmCoord shape(t, n, k);
    typename Gemm::Arguments args{
        shape, {xx, k}, {ww, k},
        {static_cast<ElementOutput*>(out.data), n},
        {static_cast<ElementOutput*>(out.data), n}, {1.0F, 0.0F}, 1};
    Gemm op;
    auto status = op.can_implement(args);
    if (status != cutlass::Status::kSuccess)
        throw std::runtime_error("SV7 FP16 TC CUTLASS can_implement failed");
    status = op.initialize(args, scratch.gemm.data, stream);
    if (status != cutlass::Status::kSuccess)
        throw std::runtime_error("SV7 FP16 TC CUTLASS initialize failed");
    status = op(stream);
    if (status != cutlass::Status::kSuccess)
        throw std::runtime_error("SV7 FP16 TC CUTLASS GEMM failed");
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
