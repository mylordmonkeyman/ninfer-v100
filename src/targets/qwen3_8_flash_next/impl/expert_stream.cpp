#include "targets/qwen3_8_flash_next/impl/expert_stream.h"
#include "core/device.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string_view>

namespace ninfer::targets::qwen3_8_flash_next::detail {
bool flash_next_expert_stream_requested() {
    const char* policy = std::getenv("NINFER_V100_PREFILL_EXPERT_POLICY");
    if (!policy || !*policy || std::string_view(policy) == "cpu-cache") return false;
    if (std::string_view(policy) == "stream" || std::string_view(policy) == "auto") return true;
    throw std::invalid_argument(
        "NINFER_V100_PREFILL_EXPERT_POLICY must be cpu-cache, stream or auto");
}

unsigned flash_next_decode_expert_stream_min_routes() {
    const char* policy = std::getenv("NINFER_V100_DECODE_EXPERT_POLICY");
    if (!policy || !*policy || std::string_view(policy) == "cpu-cache") return 0;
    if (std::string_view(policy) == "stream") return 1;
    if (std::string_view(policy) == "hybrid") return 2;
    throw std::invalid_argument(
        "NINFER_V100_DECODE_EXPERT_POLICY must be cpu-cache, stream or hybrid");
}

bool flash_next_decode_expert_stream_requested() {
    return flash_next_decode_expert_stream_min_routes() != 0;
}

unsigned flash_next_expert_stream_ring_slots() {
    const char* env = std::getenv("NINFER_V100_EXPERT_STREAM_RING_SLOTS");
    if (env == nullptr || !*env || std::string_view(env) == "4") return 4;
    if (std::string_view(env) == "8") return 8;
    throw std::invalid_argument("NINFER_V100_EXPERT_STREAM_RING_SLOTS must be 4 or 8");
}

std::size_t flash_next_expert_stream_device_bytes(unsigned maximum_routes) {
    if (!maximum_routes || maximum_routes > 8192U * 10U)
        throw std::invalid_argument("invalid prefill stream route capacity");
    const std::size_t descriptors =
        ((std::size_t(maximum_routes) + 3) / 4) * sizeof(FlashNextCachedExpertGroup);
    return flash_next_expert_stream_ring_slots() *
        (kExpertSlotBytes + std::size_t(maximum_routes) * 640 * 2 + descriptors);
}

FlashNextExpertStream::FlashNextExpertStream(unsigned maximum_routes)
    : maximum_routes_(maximum_routes), ring_slots_(flash_next_expert_stream_ring_slots()) {
    if (!maximum_routes || maximum_routes > 8192U*10U)
        throw std::invalid_argument("invalid prefill stream route capacity");
    const std::size_t descriptors = ((std::size_t(maximum_routes)+3)/4)*sizeof(FlashNextCachedExpertGroup);
    try {
        CUDA_CHECK(cudaStreamCreateWithFlags(&transfer_,cudaStreamNonBlocking));
        for (unsigned i = 0; i < ring_slots_; ++i) {
            auto& s = slots_[i];
            s.weights=std::make_unique<DeviceBuffer>(kExpertSlotBytes);
            s.activations=std::make_unique<DeviceBuffer>(std::size_t(maximum_routes)*640*2);
            s.groups=std::make_unique<DeviceBuffer>(descriptors);
            s.host_weights=std::make_unique<PinnedHostBuffer>(kExpertSlotBytes);
            s.host_groups=std::make_unique<PinnedHostBuffer>(descriptors);
            CUDA_CHECK(cudaEventCreateWithFlags(&s.ready,cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&s.consumed,cudaEventDisableTiming));
            device_bytes_+=kExpertSlotBytes+s.activations->bytes+descriptors;
            pinned_bytes_+=kExpertSlotBytes+descriptors;
        }
        if (device_bytes_ != flash_next_expert_stream_device_bytes(maximum_routes_))
            throw std::logic_error("prefill stream device plan mismatch");
    } catch (...) { cleanup(); throw; }
}
FlashNextExpertStream::~FlashNextExpertStream() { cleanup(); }
void FlashNextExpertStream::cleanup() noexcept {
    // Even an exception between upload and consumer submission cannot free pinned
    // or device slots while either stream still owns their contents.
    if (transfer_) cudaStreamSynchronize(transfer_);
    for (auto& s : slots_) {
        if (s.pending) cudaEventSynchronize(s.consumed);
        if (s.ready) cudaEventDestroy(s.ready);
        if (s.consumed) cudaEventDestroy(s.consumed);
        s.ready=nullptr; s.consumed=nullptr; s.pending=false;
    }
    if (transfer_) cudaStreamDestroy(transfer_);
    transfer_=nullptr;
}
void FlashNextExpertStream::submit(const HostNvfp4ExpertPairView& expert,
    std::span<const FlashNextStreamRoute> routes, cudaStream_t compute) {
    if (routes.empty() || routes.size()>maximum_routes_)
        throw std::invalid_argument("prefill stream group exceeds planned capacity");
    if (expert.gate_up.rows!=1280 || expert.gate_up.columns!=2560 ||
        expert.down.rows!=2560 || expert.down.columns!=640)
        throw std::invalid_argument("prefill stream requires canonical Flash-Next pair");
    if (!expert.gate_up.codes || !expert.gate_up.scales || !expert.gate_up.weight_scale_divisor ||
        !expert.down.codes || !expert.down.scales || !expert.down.weight_scale_divisor)
        throw std::invalid_argument("null canonical prefill expert plane");
    for (const auto& r:routes)
        if (!r.input_bf16 || !r.output_fp32) throw std::invalid_argument("null prefill stream route");
    auto& s=slots_[next_];
    if(s.pending) { CUDA_CHECK(cudaEventSynchronize(s.consumed)); s.pending=false; }
    auto* host=static_cast<std::byte*>(s.host_weights->data());
    std::memcpy(host,expert.gate_up.codes,1'638'400);
    std::memcpy(host+1'638'400,expert.gate_up.scales,204'800);
    std::memcpy(host+1'843'200,expert.down.codes,819'200);
    std::memcpy(host+2'662'400,expert.down.scales,102'400);
    std::memcpy(host+2'764'800,expert.gate_up.weight_scale_divisor,4);
    std::memcpy(host+2'764'804,expert.down.weight_scale_divisor,4);
    std::memset(host+kExpertPairBytes,0,kExpertSlotBytes-kExpertPairBytes);
    auto* base=static_cast<const std::byte*>(s.weights->p);
    HostNvfp4ExpertPairView view{
        {base,base+1'638'400,reinterpret_cast<const float*>(base+2'764'800),1,1280,2560},
        {base+1'843'200,base+2'662'400,reinterpret_cast<const float*>(base+2'764'804),1,2560,640}};
    auto* groups=static_cast<FlashNextCachedExpertGroup*>(s.host_groups->data());
    const unsigned count=(routes.size()+3)/4;
    for(unsigned g=0;g<count;++g) {
        auto& group=groups[g]; group={};
        group.count=std::min<std::size_t>(4,routes.size()-std::size_t(g)*4);
        for(unsigned t=0;t<group.count;++t) {
            const auto index=std::size_t(g)*4+t;
            group.tasks[t]={view,routes[index].input_bf16,
                static_cast<std::uint16_t*>(s.activations->p)+index*640,routes[index].output_fp32};
        }
    }
    CUDA_CHECK(cudaMemcpyAsync(s.weights->p,host,kExpertSlotBytes,cudaMemcpyHostToDevice,transfer_));
    CUDA_CHECK(cudaMemcpyAsync(s.groups->p,groups,count*sizeof(*groups),cudaMemcpyHostToDevice,transfer_));
    CUDA_CHECK(cudaEventRecord(s.ready,transfer_));
    CUDA_CHECK(cudaStreamWaitEvent(compute,s.ready,0));
    try {
        flash_next_cached_expert_group_launch(static_cast<const FlashNextCachedExpertGroup*>(s.groups->p),count,compute);
        CUDA_CHECK(cudaEventRecord(s.consumed,compute));
        s.pending=true;
    } catch (...) {
        // A launch may already have submitted work before reporting an error.
        cudaStreamSynchronize(compute);
        throw;
    }
    ++submitted_; next_=(next_+1)%ring_slots_;
}
void FlashNextExpertStream::finish() {
    for(auto& s:slots_) if(s.pending) {
        CUDA_CHECK(cudaEventSynchronize(s.consumed)); s.pending=false;
    }
    CUDA_CHECK(cudaStreamSynchronize(transfer_));
}
} // namespace ninfer::targets::qwen3_8_flash_next::detail
