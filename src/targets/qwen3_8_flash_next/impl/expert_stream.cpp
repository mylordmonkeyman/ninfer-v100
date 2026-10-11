#include "targets/qwen3_8_flash_next/impl/expert_stream.h"
#include "core/device.h"
#include "targets/qwen3_8_flash_next/impl/perf_telemetry.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <cstdio>
#include <cstdint>
#include <stdexcept>
#include <string_view>
#include <utility>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

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

bool flash_next_expert_stream_pipeline_reuse() {
    const char* env = std::getenv("NINFER_V100_EXPERT_STREAM_SLOT_REUSE");
    if (!env || !*env || std::string_view(env) == "blocking") return false;
    if (std::string_view(env) == "pipelined") return true;
    throw std::invalid_argument(
        "NINFER_V100_EXPERT_STREAM_SLOT_REUSE must be blocking or pipelined");
}

bool flash_next_expert_source_pin_requested() {
    const char* env = std::getenv("NINFER_V100_EXPERT_SOURCE_PINNED");
    if (!env || !*env || std::string_view(env) == "0") return false;
    if (std::string_view(env) == "1") return true;
    throw std::invalid_argument("NINFER_V100_EXPERT_SOURCE_PINNED must be 0 or 1");
}

namespace {
std::size_t host_page_bytes() {
#if defined(_WIN32)
    SYSTEM_INFO info{};
    GetSystemInfo(&info);
    return info.dwPageSize;
#else
    const long bytes = ::sysconf(_SC_PAGESIZE);
    if (bytes <= 0) throw std::runtime_error("could not determine host page size");
    return static_cast<std::size_t>(bytes);
#endif
}

std::pair<std::uintptr_t, std::uintptr_t>
expert_bank_range(const Nvfp4ExpertBankView& bank, std::size_t page_bytes) {
    if (!bank.codes || !bank.scales || !bank.weight_scale_divisors || bank.experts <= 0)
        throw std::invalid_argument("cannot pin an invalid host expert bank");
    const auto begin = reinterpret_cast<std::uintptr_t>(bank.codes);
    const auto scales = reinterpret_cast<std::uintptr_t>(bank.scales);
    const auto divisors = reinterpret_cast<std::uintptr_t>(bank.weight_scale_divisors);
    const auto end = divisors + std::size_t(bank.experts) * sizeof(float);
    if (scales < begin || divisors < scales || end <= divisors)
        throw std::invalid_argument("host expert bank planes are not contiguous and ordered");
    const auto aligned_begin = begin - begin % page_bytes;
    const auto aligned_end = ((end + page_bytes - 1) / page_bytes) * page_bytes;
    return {aligned_begin, aligned_end};
}
} // namespace

std::size_t flash_next_expert_stream_device_bytes(unsigned maximum_routes) {
    if (!maximum_routes || maximum_routes > 8192U * 10U)
        throw std::invalid_argument("invalid prefill stream route capacity");
    const std::size_t descriptors =
        ((std::size_t(maximum_routes) + 3) / 4) * sizeof(FlashNextCachedExpertGroup);
    return flash_next_expert_stream_ring_slots() *
        (kExpertSlotBytes + std::size_t(maximum_routes) * 640 * 2 + descriptors)
        + (flash_next_expert_gemm_requested() ? FlashNextExpertGemm::device_bytes : 0);
}

FlashNextExpertStream::FlashNextExpertStream(
    unsigned maximum_routes, std::span<const HostNvfp4ExpertLayerView> host_expert_layers)
    : maximum_routes_(maximum_routes), ring_slots_(flash_next_expert_stream_ring_slots()),
      pipeline_reuse_(flash_next_expert_stream_pipeline_reuse()) {
    if (!maximum_routes || maximum_routes > 8192U*10U)
        throw std::invalid_argument("invalid prefill stream route capacity");
    const char* timing = std::getenv("NINFER_V100_EXPERT_STREAM_TIMING");
    timing_ = timing && std::string_view(timing) == "1";
    const std::size_t descriptors = ((std::size_t(maximum_routes)+3)/4)*sizeof(FlashNextCachedExpertGroup);
    try {
        if (flash_next_expert_source_pin_requested()) {
            if (host_expert_layers.empty())
                throw std::invalid_argument(
                    "pinned expert source requires host expert layers");
            const std::size_t page_bytes = host_page_bytes();
            std::vector<std::pair<std::uintptr_t, std::uintptr_t>> ranges;
            ranges.reserve(host_expert_layers.size() * 2);
            for (const auto& layer : host_expert_layers) {
                ranges.push_back(expert_bank_range(layer.gate_up, page_bytes));
                ranges.push_back(expert_bank_range(layer.down, page_bytes));
            }
            std::sort(ranges.begin(), ranges.end());
            std::vector<RegisteredRange> merged;
            for (const auto& range : ranges) {
                if (!merged.empty() && range.first <=
                        reinterpret_cast<std::uintptr_t>(merged.back().base) +
                            merged.back().bytes) {
                    const auto old_begin =
                        reinterpret_cast<std::uintptr_t>(merged.back().base);
                    merged.back().bytes =
                        std::max(old_begin + merged.back().bytes, range.second) - old_begin;
                } else {
                    merged.push_back({reinterpret_cast<void*>(range.first),
                                      range.second - range.first});
                }
            }
            source_ranges_.reserve(merged.size());
            for (const auto& range : merged) {
                // The artifact mapping is intentionally read-only. This flag
                // prevents the driver from requesting writable page pins for a
                // source that is used only by host-to-device DMA.
                CUDA_CHECK(cudaHostRegister(range.base, range.bytes, cudaHostRegisterReadOnly));
                source_ranges_.push_back(range);
                source_registered_bytes_ += range.bytes;
            }
            source_pinned_ = true;
            std::fprintf(stderr,
                "v100.expert_source_pinned=1 ranges=%zu registered_bytes=%zu\n",
                source_ranges_.size(), source_registered_bytes_);
        }
        CUDA_CHECK(cudaStreamCreateWithFlags(&transfer_,cudaStreamNonBlocking));
        for (unsigned i = 0; i < ring_slots_; ++i) {
            auto& s = slots_[i];
            s.weights=std::make_unique<DeviceBuffer>(kExpertSlotBytes);
            s.activations=std::make_unique<DeviceBuffer>(std::size_t(maximum_routes)*640*2);
            s.groups=std::make_unique<DeviceBuffer>(descriptors);
            if (!source_pinned_)
                s.host_weights=std::make_unique<PinnedHostBuffer>(kExpertSlotBytes);
            s.host_groups=std::make_unique<PinnedHostBuffer>(descriptors);
            CUDA_CHECK(cudaEventCreateWithFlags(&s.ready,timing_ ? cudaEventDefault : cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&s.consumed,timing_ ? cudaEventDefault : cudaEventDisableTiming));
            if (timing_) {
                CUDA_CHECK(cudaEventCreate(&s.copy_start));
                CUDA_CHECK(cudaEventCreate(&s.wait_start));
                CUDA_CHECK(cudaEventCreate(&s.kernel_start));
            }
            device_bytes_+=kExpertSlotBytes+s.activations->bytes+descriptors;
            pinned_bytes_+=descriptors+(source_pinned_ ? 0 : kExpertSlotBytes);
        }
        if (flash_next_expert_gemm_requested()) {
            gemm_=std::make_unique<FlashNextExpertGemm>();
            device_bytes_+=FlashNextExpertGemm::device_bytes;
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
        if (s.copy_start) cudaEventDestroy(s.copy_start);
        if (s.wait_start) cudaEventDestroy(s.wait_start);
        if (s.kernel_start) cudaEventDestroy(s.kernel_start);
        s.copy_start=nullptr; s.wait_start=nullptr; s.kernel_start=nullptr;
        s.ready=nullptr; s.consumed=nullptr; s.pending=false;
    }
    if (transfer_) cudaStreamDestroy(transfer_);
    transfer_=nullptr;
    for (auto it = source_ranges_.rbegin(); it != source_ranges_.rend(); ++it)
        if (it->base) (void)cudaHostUnregister(it->base);
    source_ranges_.clear();
    source_registered_bytes_=0; source_pinned_=false;
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
    if (s.pending) {
        if (pipeline_reuse_) {
            // The host pinned buffers can be safely repacked once the previous
            // upload has completed, even while the previous kernel executes.
            // Guard the GPU destination with an event dependency rather than
            // blocking the host until the previous compute also completes.
            CUDA_CHECK(cudaEventSynchronize(s.ready));
            CUDA_CHECK(cudaStreamWaitEvent(transfer_, s.consumed, 0));
        } else {
            CUDA_CHECK(cudaEventSynchronize(s.consumed));
        }
        // Diagnostic collection uses the same completed slot before its events
        // are overwritten. Only diagnostic pipelined reuse adds a host wait.
        if (timing_) {
            CUDA_CHECK(cudaEventSynchronize(s.consumed));
            collect_timing(s);
        }
        s.pending = false;
    }
    auto* host=s.host_weights ? static_cast<std::byte*>(s.host_weights->data()) : nullptr;
    if (host) {
        std::memcpy(host,expert.gate_up.codes,1'638'400);
        std::memcpy(host+1'638'400,expert.gate_up.scales,204'800);
        std::memcpy(host+1'843'200,expert.down.codes,819'200);
        std::memcpy(host+2'662'400,expert.down.scales,102'400);
        std::memcpy(host+2'764'800,expert.gate_up.weight_scale_divisor,4);
        std::memcpy(host+2'764'804,expert.down.weight_scale_divisor,4);
        std::memset(host+kExpertPairBytes,0,kExpertSlotBytes-kExpertPairBytes);
    }
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
    if (timing_) {
        // Capture the owner at submission, before the caller can leave its
        // context. Slot collection and finish may occur in a later phase.
        s.timing = {};
        if (active_perf_context) {
            s.timing.phase = active_perf_context->phase;
            s.timing.executor = active_perf_context->executor;
            s.timing.transaction = active_perf_context->transaction;
        }
        s.timing.experts = 1; s.timing.routes = routes.size();
        const bool uses_gemm = gemm_ && routes.size() >= 32;
        s.timing.gemm_experts = uses_gemm ? 1 : 0;
        s.timing.gemm_routes = uses_gemm ? routes.size() : 0;
        const unsigned bucket = routes.size()<32 ? 0 : routes.size()<64 ? 1 :
            routes.size()<128 ? 2 : routes.size()<256 ? 3 : routes.size()<512 ? 4 : 5;
        s.timing.route_buckets[bucket] = 1;
        CUDA_CHECK(cudaEventRecord(s.copy_start,transfer_));
    }
    if (source_pinned_) {
        auto* destination=static_cast<std::byte*>(s.weights->p);
        CUDA_CHECK(cudaMemcpyAsync(destination,expert.gate_up.codes,1'638'400,
            cudaMemcpyHostToDevice,transfer_));
        CUDA_CHECK(cudaMemcpyAsync(destination+1'638'400,expert.gate_up.scales,204'800,
            cudaMemcpyHostToDevice,transfer_));
        CUDA_CHECK(cudaMemcpyAsync(destination+1'843'200,expert.down.codes,819'200,
            cudaMemcpyHostToDevice,transfer_));
        CUDA_CHECK(cudaMemcpyAsync(destination+2'662'400,expert.down.scales,102'400,
            cudaMemcpyHostToDevice,transfer_));
        CUDA_CHECK(cudaMemcpyAsync(destination+2'764'800,expert.gate_up.weight_scale_divisor,4,
            cudaMemcpyHostToDevice,transfer_));
        CUDA_CHECK(cudaMemcpyAsync(destination+2'764'804,expert.down.weight_scale_divisor,4,
            cudaMemcpyHostToDevice,transfer_));
        CUDA_CHECK(cudaMemsetAsync(destination+kExpertPairBytes,0,
            kExpertSlotBytes-kExpertPairBytes,transfer_));
    } else {
        CUDA_CHECK(cudaMemcpyAsync(s.weights->p,host,kExpertSlotBytes,
            cudaMemcpyHostToDevice,transfer_));
    }
    CUDA_CHECK(cudaMemcpyAsync(s.groups->p,groups,count*sizeof(*groups),cudaMemcpyHostToDevice,transfer_));
    CUDA_CHECK(cudaEventRecord(s.ready,transfer_));
    if (timing_) CUDA_CHECK(cudaEventRecord(s.wait_start,compute));
    CUDA_CHECK(cudaStreamWaitEvent(compute,s.ready,0));
    if (timing_) CUDA_CHECK(cudaEventRecord(s.kernel_start,compute));
    try {
        if (gemm_ && routes.size()>=32)
            gemm_->launch(view,static_cast<const FlashNextCachedExpertGroup*>(s.groups->p),routes.size(),compute);
        else
            flash_next_cached_expert_group_launch(static_cast<const FlashNextCachedExpertGroup*>(s.groups->p),count,compute);
        CUDA_CHECK(cudaEventRecord(s.consumed,compute));
        s.pending=true;
        s.timing_pending=timing_;
    } catch (...) {
        // A launch may already have submitted work before reporting an error.
        cudaStreamSynchronize(compute);
        throw;
    }
    ++submitted_; next_=(next_+1)%ring_slots_;
}
void FlashNextExpertStream::collect_timing(Slot& s) {
    if (!s.timing_pending) return;
    float copy = 0, wait = 0, kernel = 0;
    CUDA_CHECK(cudaEventElapsedTime(&copy,s.copy_start,s.ready));
    CUDA_CHECK(cudaEventElapsedTime(&wait,s.wait_start,s.kernel_start));
    CUDA_CHECK(cudaEventElapsedTime(&kernel,s.kernel_start,s.consumed));
    auto entry=std::find_if(timings_.begin(),timings_.end(),[&](const TimingRecord& t) {
        return t.executor==s.timing.executor && t.transaction==s.timing.transaction && t.phase==s.timing.phase;
    });
    if(entry==timings_.end()) {
        timings_.push_back(s.timing);entry=std::prev(timings_.end());
    } else {
        entry->experts+=s.timing.experts;entry->routes+=s.timing.routes;
        entry->gemm_experts+=s.timing.gemm_experts;entry->gemm_routes+=s.timing.gemm_routes;
        for(unsigned i=0;i<6;++i)entry->route_buckets[i]+=s.timing.route_buckets[i];
    }
    entry->copy_ms+=copy;entry->wait_ms+=wait;entry->kernel_ms+=kernel;
    s.timing_pending=false;
}
void FlashNextExpertStream::finish() {
    for(auto& s:slots_) if(s.pending) {
        CUDA_CHECK(cudaEventSynchronize(s.consumed));
        collect_timing(s); s.pending=false;
    }
    CUDA_CHECK(cudaStreamSynchronize(transfer_));
    for (const auto& t : timings_) {
        // Separate streams overlap: these sums must never be added as a wall time.
        std::ostringstream out;out.imbue(std::locale::classic());out.precision(17);
        out << "{\"kind\":\"expert_stream_timing\",\"experts\":" << t.experts
            << ",\"routes\":" << t.routes << ",\"copy_ms\":" << t.copy_ms
            << ",\"compute_wait_ms\":" << t.wait_ms << ",\"kernel_interval_ms\":" << t.kernel_ms
            << ",\"gemm_experts\":" << t.gemm_experts << ",\"gemm_routes\":" << t.gemm_routes
            << ",\"route_buckets\":[";
        for(unsigned i=0;i<6;++i){if(i)out<<',';out<<t.route_buckets[i];}
        out << "],\"context\":{\"executor\":" << t.executor << ",\"transaction\":" << t.transaction
            << ",\"phase\":\"" << t.phase << "\"}}";
        emit_perf_json(out.str());
    }
    timings_.clear();
}
} // namespace ninfer::targets::qwen3_8_flash_next::detail
