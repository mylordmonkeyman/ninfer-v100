#include "targets/qwen3_8_flash_next/impl/expert_cache.h"
#include "core/device.h"
#include <algorithm>
#include <cstdio>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace ninfer::targets::qwen3_8_flash_next::detail {
namespace { constexpr std::size_t GiB = 1ULL << 30; }
FlashNextExpertCacheBudget flash_next_expert_cache_budget(
    std::size_t free, std::size_t total, std::size_t transfer, bool mtp, unsigned maximum) {
    FlashNextExpertCacheBudget b{.total_bytes=total, .free_before_bytes=free,
        .transfer_bytes=transfer, .reserve_bytes=2*GiB,
        .used_limit_bytes=std::min(total, std::size_t(mtp ? 30 : 28)*GiB)};
    if (free > total || maximum > 512) throw std::invalid_argument("invalid expert cache budget");
    const auto used = total-free;
    const auto available = std::min(free > b.reserve_bytes ? free-b.reserve_bytes : 0,
        b.used_limit_bytes > used ? b.used_limit_bytes-used : 0);
    const auto weights = available > transfer ? available-transfer : 0;
    b.slots_per_layer = std::min<std::size_t>(maximum,
        weights/(kFlashNextRoutedExpertLayers*kExpertSlotBytes));
    b.cache_bytes = b.slots_per_layer*kFlashNextRoutedExpertLayers*kExpertSlotBytes;
    return b;
}

FlashNextExpertCache::FlashNextExpertCache(const HostNvfp4ExpertTableView& host,
    unsigned max_tokens, bool mtp, unsigned maximum, unsigned admission_cap)
    : host_(host), admission_cap_(admission_cap) {
    if (admission_cap < 1 || admission_cap > 2)
        throw std::invalid_argument("expert cache admission cap must be 1 or 2");
    if (!max_tokens) throw std::invalid_argument("zero expert cache token capacity");
    for (const auto& layer : host.layers) {
        if (layer.compact_bytes_per_expert_pair() != kExpertPairBytes ||
            layer.gate_up.experts != 512 || layer.down.experts != 512)
            throw std::invalid_argument("expert cache requires exact Flash-Next banks");
    }
    const auto enabled = [](const char* name) {
        const char* value = std::getenv(name);
        return value != nullptr && std::strcmp(value, "1") == 0;
    };
    timing_enabled_ = enabled("NINFER_FLASH_NEXT_EXPERT_CACHE_TIMING");
    serial_schedule_ = enabled("NINFER_FLASH_NEXT_EXPERT_CACHE_SERIAL");
    CUDA_CHECK(cudaGetDevice(&device_));
    std::size_t free=0,total=0;
    CUDA_CHECK(cudaMemGetInfo(&free,&total));
    const std::size_t paths=std::size_t(max_tokens)*10;
    const auto transfer=paths*(640*sizeof(std::uint16_t)+2560*sizeof(float));
    budget_=flash_next_expert_cache_budget(free,total,transfer,mtp,maximum);
    if (!budget_.slots_per_layer) return;
    storage_=std::make_unique<DeviceBuffer>(budget_.cache_bytes);
    activations_=std::make_unique<DeviceBuffer>(paths*640*sizeof(std::uint16_t));
    outputs_=std::make_unique<DeviceBuffer>(paths*2560*sizeof(float));
    result_buffer_=std::make_unique<PinnedHostBuffer>(outputs_->bytes);
    // One canonical pair pinned at a time; queued jobs retain only keys into the pageable mmap.
    fill_buffer_=std::make_unique<PinnedHostBuffer>(kExpertSlotBytes);
    entries_.resize(std::size_t(budget_.slots_per_layer)*48);
    CUDA_CHECK(cudaStreamCreateWithFlags(&fill_stream_,cudaStreamNonBlocking));
    std::size_t free_after=0,total_after=0;
    CUDA_CHECK(cudaMemGetInfo(&free_after,&total_after));
    std::fprintf(stderr,"phase13.cache.free_after_bytes=%zu\nphase13.cache.observed_device_bytes=%zu\n",
        free_after,free-free_after);
    if (timing_enabled_) {
        CUDA_CHECK(cudaEventCreate(&hit_start_));
        CUDA_CHECK(cudaEventCreate(&hit_stop_));
        CUDA_CHECK(cudaEventCreate(&fill_start_));
        CUDA_CHECK(cudaEventCreate(&fill_stop_));
    }
    worker_=std::thread([this]{fill_loop();});
    std::fprintf(stderr,"phase13.cache.slots_per_layer=%u\nphase13.cache.bytes=%zu\n"
        "phase13.cache.transfer_bytes=%zu\nphase13.cache.free_before_bytes=%zu\n"
        "phase13.cache.total_bytes=%zu\nphase13.cache.reserve_bytes=%zu\n"
        "phase13.cache.used_limit_bytes=%zu\nphase13.cache.pinned_bytes=%zu\n",
        budget_.slots_per_layer,budget_.cache_bytes,transfer,free,total,budget_.reserve_bytes,
        budget_.used_limit_bytes,result_buffer_->size()+fill_buffer_->size());
}
FlashNextExpertCache::~FlashNextExpertCache() {
    { std::lock_guard lock(mutex_); stop_=true; work_.notify_one(); }
    if(worker_.joinable())worker_.join();
    if(fill_stream_)cudaStreamDestroy(fill_stream_);
    if(hit_start_)cudaEventDestroy(hit_start_);
    if(hit_stop_)cudaEventDestroy(hit_stop_);
    if(fill_start_)cudaEventDestroy(fill_start_);
    if(fill_stop_)cudaEventDestroy(fill_stop_);
}
void FlashNextExpertCache::check_failure() const { if(failure_)std::rethrow_exception(failure_); }
HostNvfp4ExpertPairView FlashNextExpertCache::view(unsigned slot) const {
    auto* base=static_cast<const std::byte*>(storage_->p)+std::size_t(slot)*kExpertSlotBytes;
    // Keep every plane aligned, including the down codes after the gate divisor.
    return {{base,base+1'638'400,reinterpret_cast<const float*>(base+2'764'800),
             1,1280,2560},
            {base+1'843'200,base+2'662'400,
             reinterpret_cast<const float*>(base+2'764'804),1,2560,640}};
}
HostNvfp4ExpertPairView FlashNextExpertCache::ready_view(unsigned layer,int expert) {
    std::lock_guard lock(mutex_); check_failure();
    if(layer>=48||expert<0||expert>=512)throw std::invalid_argument("invalid cache key");
    for(unsigned i=layer*budget_.slots_per_layer;i<(layer+1)*budget_.slots_per_layer;++i)
        if(entries_[i].expert==expert&&entries_[i].state==State::Ready)return view(i);
    throw std::runtime_error("expert cache key is not Ready");
}
bool FlashNextExpertCache::execute(unsigned layer,int expert,const void* input,
    unsigned path,cudaStream_t stream) {
    unsigned slot=0;
    {
        std::lock_guard lock(mutex_); check_failure();
        if(layer>=48||expert<0||expert>=512)throw std::invalid_argument("invalid cache key");
        bool found=false;
        for(unsigned i=layer*budget_.slots_per_layer;i<(layer+1)*budget_.slots_per_layer;++i)
            if(entries_[i].expert==expert&&entries_[i].state==State::Ready){slot=i;found=true;break;}
        if(!found){++stats_.misses;return false;}
        if((std::size_t(path)+1)*2560*sizeof(float)>outputs_->bytes)
            throw std::out_of_range("expert cache hit buffer capacity");
        entries_[slot].epoch=++epoch_;++entries_[slot].leases;
        if (consumers_.empty() && timing_enabled_)
            CUDA_CHECK(cudaEventRecord(hit_start_, stream));
        consumers_.emplace_back(slot,path);++stats_.hits;
    }
    flash_next_cached_expert_launch(view(slot),input,
        static_cast<std::uint16_t*>(activations_->p)+std::size_t(path)*640,
        static_cast<float*>(outputs_->p)+std::size_t(path)*2560,stream);
    return true;
}
void FlashNextExpertCache::begin_download(std::size_t bytes, cudaStream_t stream) {
    if (consumers_.empty()) return;
    if (bytes > outputs_->bytes) throw std::out_of_range("expert cache result capacity");
    CUDA_CHECK(cudaMemcpyAsync(result_buffer_->data(), outputs_->p, bytes,
                               cudaMemcpyDeviceToHost, stream));
    if (timing_enabled_) CUDA_CHECK(cudaEventRecord(hit_stop_, stream));
}
double FlashNextExpertCache::finish_download(std::span<float> output, cudaStream_t stream,
    double* wait_us) {
    if (wait_us) *wait_us = 0;
    if (consumers_.empty()) return 0;
    const auto waiting = timing_enabled_ ? std::chrono::steady_clock::now() :
        std::chrono::steady_clock::time_point{};
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (wait_us && timing_enabled_) *wait_us = std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now()-waiting).count();
    float gpu_ms = 0;
    if (timing_enabled_) CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, hit_start_, hit_stop_));
    std::lock_guard lock(mutex_); check_failure();
    for (auto [slot, path] : consumers_) {
        std::memcpy(output.data() + std::size_t(path)*2560,
            static_cast<float*>(result_buffer_->data()) + std::size_t(path)*2560,
            2560*sizeof(float));
        --entries_[slot].leases;
    }
    consumers_.clear();
    return double(gpu_ms)*1000;
}
void FlashNextExpertCache::download(std::span<float> output, cudaStream_t stream) {
    begin_download(output.size_bytes(), stream);
    finish_download(output, stream);
}
void FlashNextExpertCache::record_schedule(double cpu, double gpu, double wait, double wall) {
    if (!timing_enabled_) return;
    std::lock_guard lock(mutex_);
    ++stats_.schedule_calls;
    stats_.cpu_branch_us += cpu;
    stats_.gpu_branch_us += gpu;
    stats_.merge_wait_us += wait;
    stats_.branch_wall_us += wall;
    // Both intervals lie inside the measured host branch span. Their combined length
    // minus that span is a conservative lower bound on the intersection, without
    // assuming synchronized CPU/CUDA clocks or counting dispatch gaps as overlap.
    stats_.overlap_lower_bound_us += std::max(0.0, cpu + gpu - wall);
}
void FlashNextExpertCache::reset() {
    drain();
    std::lock_guard lock(mutex_);
    if (!consumers_.empty()) throw std::logic_error("cache reset with outstanding consumers");
    std::fill(entries_.begin(), entries_.end(), Entry{});
    stats_ = {};
    epoch_ = 0;
    admissions_enabled_ = true;
}
void FlashNextExpertCache::admit(unsigned layer,std::span<const std::int32_t> ids) {
    const auto started = timing_enabled_ ? std::chrono::steady_clock::now() :
        std::chrono::steady_clock::time_point{};
    std::lock_guard lock(mutex_);check_failure();
    if(layer>=48)throw std::invalid_argument("invalid cache layer");
    if(!admissions_enabled_||!budget_.slots_per_layer)return;
    const unsigned begin=layer*budget_.slots_per_layer,end=begin+budget_.slots_per_layer;
    unsigned admitted = 0;
    for(int id:ids){
        if(id<0||id>=512)throw std::invalid_argument("invalid cache expert id");
        bool present=false;
        for(unsigned i=begin;i<end;++i)if(entries_[i].expert==id){present=true;break;}
        if(present)continue;
        if(queue_.size()+unsigned(filling_)>=4){++stats_.queue_declined_calls;break;}
        unsigned victim=end;
        for(unsigned i=begin;i<end;++i){
            const auto& e=entries_[i];
            if(e.leases||e.state==State::Uploading)continue;
            if(victim==end||e.state==State::Empty||e.epoch<entries_[victim].epoch)victim=i;
            if(e.state==State::Empty)break;
        }
        if(victim==end){++stats_.victim_declined_calls;break;}
        if(entries_[victim].state!=State::Empty)++stats_.evicted;
        entries_[victim]={id,State::Uploading,++epoch_,0};
        queue_.push_back(victim);++stats_.admitted;++admitted;
        stats_.maximum_outstanding=std::max(stats_.maximum_outstanding,
            unsigned(queue_.size())+unsigned(filling_));
        work_.notify_one();
        if(admitted==admission_cap_)break;
    }
    ++stats_.admission_bursts[admitted];
    if(timing_enabled_)stats_.admission_wall_us+=std::chrono::duration<double,std::micro>(
        std::chrono::steady_clock::now()-started).count();
}
void FlashNextExpertCache::fill_loop() noexcept {
    try {
        CUDA_CHECK(cudaSetDevice(device_));
        for(;;){
            unsigned slot;int id;
            {
                std::unique_lock lock(mutex_);
                work_.wait(lock,[this]{return stop_||!queue_.empty();});
                if(stop_&&queue_.empty())return;
                slot=queue_.front();queue_.pop_front();id=entries_[slot].expert;filling_=true;
            }
            const auto started=std::chrono::steady_clock::now();
            const auto src=host_.expert(slot/budget_.slots_per_layer,id);
            auto* dst=static_cast<std::byte*>(fill_buffer_->data());
            std::memcpy(dst,src.gate_up.codes,1'638'400);
            std::memcpy(dst+1'638'400,src.gate_up.scales,204'800);
            std::memcpy(dst+1'843'200,src.down.codes,819'200);
            std::memcpy(dst+2'662'400,src.down.scales,102'400);
            std::memcpy(dst+2'764'800,src.gate_up.weight_scale_divisor,4);
            std::memcpy(dst+2'764'804,src.down.weight_scale_divisor,4);
            std::memset(dst+kExpertPairBytes,0,kExpertSlotBytes-kExpertPairBytes);
            const double pack_us = std::chrono::duration<double,std::micro>(
                std::chrono::steady_clock::now()-started).count();
            if(timing_enabled_)CUDA_CHECK(cudaEventRecord(fill_start_,fill_stream_));
            CUDA_CHECK(cudaMemcpyAsync(static_cast<std::byte*>(storage_->p)+
                std::size_t(slot)*kExpertSlotBytes,dst,kExpertSlotBytes,cudaMemcpyHostToDevice,fill_stream_));
            if(timing_enabled_)CUDA_CHECK(cudaEventRecord(fill_stop_,fill_stream_));
            CUDA_CHECK(cudaStreamSynchronize(fill_stream_));
            float h2d_ms = 0;
            if(timing_enabled_)CUDA_CHECK(cudaEventElapsedTime(&h2d_ms,fill_start_,fill_stop_));
            {
                std::lock_guard lock(mutex_);
                const auto elapsed=std::chrono::duration<double,std::micro>(
                    std::chrono::steady_clock::now()-started).count();
                stats_.fill_wall_us+=elapsed;
                stats_.pack_wall_us+=pack_us;stats_.h2d_us+=double(h2d_ms)*1000;
                stats_.fill_bytes+=kExpertSlotBytes;
                stats_.maximum_fill_wall_us=std::max(stats_.maximum_fill_wall_us,elapsed);
                entries_[slot].state=State::Canonical;
                // This leaf reads the canonical software-NVFP4 layout directly. No prepack
                // allocation or Prepacking transition is needed before publishing Ready.
                entries_[slot].state=State::Ready;++stats_.ready;
                filling_=false;idle_.notify_all();
            }
        }
    }catch(...){std::lock_guard lock(mutex_);failure_=std::current_exception();
        filling_=false;queue_.clear();idle_.notify_all();}
}
void FlashNextExpertCache::drain(){std::unique_lock lock(mutex_);
    idle_.wait(lock,[this]{return queue_.empty()&&!filling_;});check_failure();}
FlashNextExpertCacheStats FlashNextExpertCache::stats() const {
    std::lock_guard lock(mutex_);check_failure();return stats_;
}
} // namespace ninfer::targets::qwen3_8_flash_next::detail
