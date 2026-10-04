#include "ops/linear/bf16/bf16_telemetry.h"
#include "core/device.h"

namespace ninfer::ops::detail {
namespace {
const char* implementation_name(Bf16Launch launch) {
#ifdef NINFER_VOLTA_BUILD
    if (launch == launch_bf16_volta_simt) return "volta_simt";
    if (launch == launch_bf16_cutlass_sm70) return "cutlass_sm70";
#else
    if (launch == launch_bf16_mma) return "mma";
#endif
    if (launch == launch_bf16_decode) return "decode";
    if (launch == launch_bf16_small_t) return "small_t";
    if (launch == launch_bf16_n256_k5120) return "n256_k5120";
    return "unclassified";
}
}
Bf16TimingCollector::~Bf16TimingCollector() {
    for (const auto& entry : entries_) {
        if (entry.start) cudaEventDestroy(entry.start);
        if (entry.stop) cudaEventDestroy(entry.stop);
    }
}
void Bf16TimingCollector::launch(Bf16Launch implementation, const Tensor& x,
                               const Weight& weight, Tensor& output, cudaStream_t stream) {
    cudaStreamCaptureStatus capture;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
    if (capture != cudaStreamCaptureStatusNone) {
        implementation(x, weight, output, stream);
        return;
    }
    if (count_ == entries_.size()) {
        entries_.emplace_back();
        auto& entry = entries_.back();
        CUDA_CHECK(cudaEventCreate(&entry.start));
        CUDA_CHECK(cudaEventCreate(&entry.stop));
    }
    auto& entry = entries_[count_];
    entry.n = weight.n; entry.k = weight.k; entry.t = x.ne[1];
    entry.implementation = implementation_name(implementation);
    CUDA_CHECK(cudaEventRecord(entry.start, stream));
    implementation(x, weight, output, stream);
    CUDA_CHECK(cudaEventRecord(entry.stop, stream));
    ++count_;
}
void Bf16TimingCollector::finish(
    const std::function<void(int, int, int, const char*, double)>& emit) {
    // Called only after all round work has been submitted; events remain alive in
    // the owning executor across rounds. No pointer to a stack object is captured.
    for (std::size_t i = 0; i < count_; ++i) {
        const auto& entry = entries_[i];
        CUDA_CHECK(cudaEventSynchronize(entry.stop));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, entry.start, entry.stop));
        emit(entry.n, entry.k, entry.t, entry.implementation, double(ms)*1000);
    }
    count_ = 0;
}
} // namespace ninfer::ops::detail
