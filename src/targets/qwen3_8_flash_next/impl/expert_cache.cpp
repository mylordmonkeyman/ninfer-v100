#include "targets/qwen3_8_flash_next/impl/expert_cache.h"
#include "core/device.h"
#include "targets/qwen3_8_flash_next/impl/expert_gemm.h"
#include "targets/qwen3_8_flash_next/impl/perf_telemetry.h"
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
    if (const char* policy = std::getenv("NINFER_V100_EXPERT_POLICY"); policy && *policy) {
        if (std::strcmp(policy, "static") == 0) {
            static_profile_=true; admissions_enabled_=false;
        } else if (std::strcmp(policy, "profile") == 0) {
            const char* weight=std::getenv("NINFER_V100_EXPERT_PRIOR_WEIGHT");
            char* end=nullptr;
            if (!weight) throw std::invalid_argument("profile policy requires explicit prior weight");
            prior_weight_=std::strtod(weight,&end);
            if (end==weight || *end || !std::isfinite(prior_weight_) || prior_weight_<=0)
                throw std::invalid_argument("invalid expert profile prior weight");
            heat_=std::make_unique<FlashNextExpertHeat>();
        } else if (std::strcmp(policy, "heat") == 0) {
            heat_ = std::make_unique<FlashNextExpertHeat>();
        } else if (std::strcmp(policy, "decay") == 0) {
            const char* coefficient = std::getenv("NINFER_V100_EXPERT_DECAY");
            const char* interval = std::getenv("NINFER_V100_EXPERT_DECAY_INTERVAL");
            if (!coefficient || !interval)
                throw std::invalid_argument("decay policy requires explicit coefficient and layer-call interval");
            char* end = nullptr;
            const double decay = std::strtod(coefficient, &end);
            if (end == coefficient || *end) throw std::invalid_argument("invalid expert decay");
            const auto calls = std::strtoul(interval, &end, 10);
            if (end == interval || *end || !calls || calls > std::numeric_limits<unsigned>::max())
                throw std::invalid_argument("invalid expert decay interval");
            heat_ = std::make_unique<FlashNextExpertHeat>(decay, static_cast<unsigned>(calls));
        } else if (std::strcmp(policy, "lru") != 0) {
            throw std::invalid_argument("expert policy must be lru, heat, decay, static, or profile");
        }
    }
    adaptive_heat_=bool(heat_);
    if (const char* path=std::getenv("NINFER_V100_EXPERT_PROFILE_SAVE");path && *path) {
        if (host.model_id.empty() || host.weights_id.empty())
            throw std::invalid_argument("learned profile requires actual artifact identity");
        save_profile_path_=path;
        if (!heat_) heat_=std::make_unique<FlashNextExpertHeat>();
    }
    for (const auto& layer : host.layers) {
        if (layer.compact_bytes_per_expert_pair() != kExpertPairBytes ||
            layer.gate_up.experts != 512 || layer.down.experts != 512)
            throw std::invalid_argument("expert cache requires exact Flash-Next banks");
    }
    const auto enabled = [](const char* name) {
        const char* value = std::getenv(name);
        return value != nullptr && std::strcmp(value, "1") == 0;
    };
    timing_enabled_ = enabled("NINFER_FLASH_NEXT_EXPERT_CACHE_TIMING") || v100_perf_telemetry_enabled();
    serial_schedule_ = enabled("NINFER_FLASH_NEXT_EXPERT_CACHE_SERIAL");
    if (const char* value = std::getenv("NINFER_FLASH_NEXT_EXPERT_CACHE_PREFILL"); value && *value) {
        if (std::strcmp(value, "0") && std::strcmp(value, "1"))
            throw std::invalid_argument("expert cache prefill must be 0 or 1");
        prefill_enabled_ = std::strcmp(value, "0") != 0;
    }
    batched_prefill_ = enabled("NINFER_FLASH_NEXT_EXPERT_CACHE_BATCHED_PREFILL");
    // Experimental only: Phase 17 compares launch fusion against the scalar decode baseline.
    batched_decode_ = enabled("NINFER_FLASH_NEXT_EXPERT_CACHE_BATCHED_DECODE");
    // Reuse canonical weights across routed inputs during prefill by default.
    // The scalar and batched paths remain available as qualification controls.
    if (const char* value = std::getenv("NINFER_FLASH_NEXT_EXPERT_CACHE_GROUPED_PREFILL"); value && *value) {
        if (std::strcmp(value, "0") && std::strcmp(value, "1"))
            throw std::invalid_argument("grouped expert cache prefill must be 0 or 1");
        grouped_prefill_ = std::strcmp(value, "0") != 0;
    }
    CUDA_CHECK(cudaGetDevice(&device_));
    std::size_t free=0,total=0;
    CUDA_CHECK(cudaMemGetInfo(&free,&total));
    const std::size_t paths=std::size_t(max_tokens)*10;
    const bool resident_gemm=flash_next_resident_expert_gemm_requested();
    // Include the bounded expansion/workspace in the same operating budget.
    const auto transfer=paths*(640*sizeof(std::uint16_t)+2560*sizeof(float)+sizeof(FlashNextCachedExpertGroup))
        + (resident_gemm ? FlashNextExpertGemm::device_bytes : 0);
    budget_=flash_next_expert_cache_budget(free,total,transfer,mtp,maximum);
    if (!budget_.slots_per_layer) return;
    if (resident_gemm) prefill_gemm_=std::make_unique<FlashNextExpertGemm>();
    storage_=std::make_unique<DeviceBuffer>(budget_.cache_bytes);
    activations_=std::make_unique<DeviceBuffer>(paths*640*sizeof(std::uint16_t));
    outputs_=std::make_unique<DeviceBuffer>(paths*2560*sizeof(float));
    result_buffer_=std::make_unique<PinnedHostBuffer>(outputs_->bytes);
    batch_tasks_=std::make_unique<DeviceBuffer>(paths*sizeof(FlashNextCachedExpertGroup));
    batch_descriptors_=std::make_unique<PinnedHostBuffer>(paths*sizeof(FlashNextCachedExpertTask));
    group_descriptors_=std::make_unique<PinnedHostBuffer>(batch_tasks_->bytes);
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
        CUDA_CHECK(cudaEventCreate(&result_copy_start_));
        CUDA_CHECK(cudaEventCreate(&fill_start_));
        CUDA_CHECK(cudaEventCreate(&fill_stop_));
    }
    worker_=std::thread([this]{fill_loop();});
    std::fprintf(stderr,"phase13.cache.slots_per_layer=%u\nphase13.cache.bytes=%zu\n"
        "phase13.cache.transfer_bytes=%zu\nphase13.cache.free_before_bytes=%zu\n"
        "phase13.cache.total_bytes=%zu\nphase13.cache.reserve_bytes=%zu\n"
        "phase13.cache.used_limit_bytes=%zu\nphase13.cache.pinned_bytes=%zu\n",
        budget_.slots_per_layer,budget_.cache_bytes,transfer,free,total,budget_.reserve_bytes,
        budget_.used_limit_bytes,result_buffer_->size()+fill_buffer_->size()+batch_descriptors_->size()+group_descriptors_->size());
}
FlashNextExpertCache::~FlashNextExpertCache() {
    if (shutdown_profile_save_ && !save_profile_path_.empty()) {
        try { save_profile(save_profile_path_); }
        catch (const std::exception& e) {
            std::fprintf(stderr,"v100.profile.save_failed=%s\n",e.what());
        }
    }
    { std::lock_guard lock(mutex_); stop_=true; work_.notify_one(); }
    if(worker_.joinable())worker_.join();
    if(fill_stream_)cudaStreamDestroy(fill_stream_);
    if(hit_start_)cudaEventDestroy(hit_start_);
    if(hit_stop_)cudaEventDestroy(hit_stop_);
    if(result_copy_start_)cudaEventDestroy(result_copy_start_);
    if(fill_start_)cudaEventDestroy(fill_start_);
    if(fill_stop_)cudaEventDestroy(fill_stop_);
}
void FlashNextExpertCache::seed(const FlashNextExpertProfile& profile) {
    if (host_.model_id.empty() || host_.weights_id.empty() ||
        profile.model_id != host_.model_id || profile.weights_id != host_.weights_id)
        throw std::invalid_argument("expert profile artifact identity mismatch");
    {
        std::lock_guard lock(mutex_); check_failure();
        if (!consumers_.empty() || filling_ || !queue_.empty() ||
            std::any_of(entries_.begin(), entries_.end(), [](const Entry& e) {
                return e.state != State::Empty;
            })) throw std::logic_error("expert profile seeding requires an empty idle cache");
        // Validate before enqueuing any work, including zero-budget profiles.
        for (const auto& row : profile.ranking) {
            std::array<bool,512> seen{};
            for (int id : row) {
                if (id < 0 || id >= 512 || seen[id])
                    throw std::invalid_argument("invalid expert profile permutation");
                seen[id] = true;
            }
        }
    }
    original_profile_=std::make_unique<FlashNextExpertProfile>(profile);
    if (prior_weight_>0) heat_->set_prior(profile.ranking,prior_weight_);
    const auto start = std::chrono::steady_clock::now();
    for (unsigned layer = 0; layer < 48; ++layer) {
        for (unsigned rank = 0; rank < budget_.slots_per_layer;) {
            {
                std::lock_guard lock(mutex_); check_failure();
                // Same bounded worker and exact canonical pack/H2D/publish path as admission.
                while (rank < budget_.slots_per_layer && queue_.size()+unsigned(filling_) < 4) {
                    const unsigned slot = layer*budget_.slots_per_layer+rank;
                    entries_[slot] = {profile.ranking[layer][rank], State::Uploading, ++epoch_, 0};
                    queue_.push_back(slot); ++stats_.admitted; ++layer_totals_[layer].admitted; ++rank;
                    stats_.maximum_outstanding = std::max(stats_.maximum_outstanding,
                        unsigned(queue_.size())+unsigned(filling_));
                }
                work_.notify_one();
            }
            drain();
        }
    }
    std::fprintf(stderr,"v100.profile.seeded=%zu\nv100.profile.startup_ms=%.3f\n",
        entries_.size(), std::chrono::duration<double,std::milli>(
            std::chrono::steady_clock::now()-start).count());
}
void FlashNextExpertCache::save_profile(const std::filesystem::path& path) {
    drain();
    if (!heat_) throw std::logic_error("profile save requires heat collection enabled before inference");
    FlashNextExpertProfile profile;
    FlashNextExpertProfile::Heat measured{};
    FlashNextExpertProfile::Residents residents{};
    {
        std::lock_guard lock(mutex_); check_failure();
        if (!consumers_.empty()) throw std::logic_error("profile save with outstanding consumers");
        if (host_.model_id.empty() || host_.weights_id.empty())
            throw std::invalid_argument("profile save requires actual artifact identity");
        if (original_profile_) profile=*original_profile_;
        else {
            profile.model_id=host_.model_id;profile.weights_id=host_.weights_id;
            for (auto& row:profile.ranking) for (int id=0;id<512;++id) row[id]=id;
        }
        for (unsigned layer=0;layer<48;++layer) {
            if (heat_) for (unsigned id=0;id<512;++id) measured[layer][id]=heat_->measured(layer,id);
            for (unsigned slot=layer*budget_.slots_per_layer;
                    slot<(layer+1)*budget_.slots_per_layer;++slot)
                if (entries_[slot].state==State::Ready) residents[layer][entries_[slot].expert]=true;
        }
    }
    profile.learned(measured,residents).save(path,measured);
    std::fprintf(stderr,"v100.profile.saved=%s\n",path.c_str());
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
void FlashNextExpertCache::begin_layer(bool prefill) {
    if (!consumers_.empty()) throw std::logic_error("cache layer has outstanding consumers");
    grouping_layer_ = prefill && grouped_prefill_;
    batching_layer_ =
        (prefill && (batched_prefill_ || grouped_prefill_)) || (!prefill && batched_decode_);
}
bool FlashNextExpertCache::execute(unsigned layer,int expert,const void* input,
    unsigned path,cudaStream_t stream) {
    if (outputs_ && (std::size_t(path)+1)*2560*sizeof(float)>outputs_->bytes)
        throw std::out_of_range("expert cache hit buffer capacity");
    return execute_impl(layer,expert,input,
        outputs_ ? static_cast<float*>(outputs_->p)+std::size_t(path)*2560 : nullptr,
        path,stream);
}
bool FlashNextExpertCache::execute_to(unsigned layer,int expert,const void* input,
    float* output,unsigned path,cudaStream_t stream) {
    if (output==nullptr) throw std::invalid_argument("expert cache device output is null");
    return execute_impl(layer,expert,input,output,path,stream);
}
bool FlashNextExpertCache::execute_impl(unsigned layer,int expert,const void* input,
    float* output,unsigned path,cudaStream_t stream) {
    unsigned slot=0;
    {
        std::lock_guard lock(mutex_); check_failure();
        if(layer>=48||expert<0||expert>=512)throw std::invalid_argument("invalid cache key");
        bool found=false;
        for(unsigned i=layer*budget_.slots_per_layer;i<(layer+1)*budget_.slots_per_layer;++i)
            if(entries_[i].expert==expert&&entries_[i].state==State::Ready){slot=i;found=true;break;}
        if(!found){++stats_.misses;++layer_totals_[layer].misses;return false;}
        if(output==nullptr)throw std::logic_error("Ready cache entry has no output storage");
        entries_[slot].epoch=++epoch_;++entries_[slot].leases;
        if (consumers_.empty() && timing_enabled_)
            CUDA_CHECK(cudaEventRecord(hit_start_, stream));
        consumers_.emplace_back(slot,path);++stats_.hits;++layer_totals_[layer].hits;
    }
    if (batching_layer_) {
        auto* descriptors=static_cast<FlashNextCachedExpertTask*>(batch_descriptors_->data());
        descriptors[consumers_.size()-1] = {view(slot), input,
            static_cast<std::uint16_t*>(activations_->p)+std::size_t(path)*640,
            output};
        return true;
    }
    const auto submitted = timing_enabled_ ? std::chrono::steady_clock::now() :
        std::chrono::steady_clock::time_point{};
    flash_next_cached_expert_launch(view(slot),input,
        static_cast<std::uint16_t*>(activations_->p)+std::size_t(path)*640,
        output,stream);
    if (timing_enabled_) {
        std::lock_guard lock(mutex_);
        stats_.hit_submission_us += std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now()-submitted).count();
        stats_.hit_kernel_launches += 2;
    }
    return true;
}
void FlashNextExpertCache::submit_consumers(cudaStream_t stream) {
    if (consumers_.empty()) return;
    if (batching_layer_) {
        const auto submitted = timing_enabled_ ? std::chrono::steady_clock::now() :
            std::chrono::steady_clock::time_point{};
        const auto count=static_cast<unsigned>(consumers_.size());
        unsigned groups=0;
        if(grouping_layer_) {
            auto* tasks=static_cast<FlashNextCachedExpertTask*>(batch_descriptors_->data());
            std::sort(tasks,tasks+count,[](const auto& a,const auto& b) {
                return reinterpret_cast<std::uintptr_t>(a.expert.gate_up.codes)<
                    reinterpret_cast<std::uintptr_t>(b.expert.gate_up.codes);
            });
            auto* descriptors=static_cast<FlashNextCachedExpertGroup*>(group_descriptors_->data());
            struct GemmSpan { HostNvfp4ExpertPairView expert; unsigned first_group, routes; };
            std::vector<GemmSpan> gemm_spans;
            unsigned singles=0;
            // GEMM descriptors form a prefix, followed by the fused SIMT fallback.
            // Pack the prefix first, retaining original task order within each expert.
            unsigned simt_count=prefill_gemm_ ? 0 : count;
            for(unsigned i=0;prefill_gemm_ && i<count;) {
                unsigned end=i+1;
                while(end<count&&tasks[end].expert.gate_up.codes==tasks[i].expert.gate_up.codes)++end;
                if(prefill_gemm_ && end-i>=32) {
                    gemm_spans.push_back({tasks[i].expert,groups,end-i});
                    while(i<end) {
                        auto& group=descriptors[groups++];group={};
                        group.count=std::min(4U,end-i);
                        for(unsigned t=0;t<group.count;++t)group.tasks[t]=tasks[i++];
                    }
                } else {
                    while(i<end)tasks[simt_count++]=tasks[i++];
                }
            }
            const unsigned gemm_groups=groups;
            for(unsigned i=0;i<simt_count;) {
                unsigned end=i+1;
                while(end<simt_count&&tasks[end].expert.gate_up.codes==tasks[i].expert.gate_up.codes)++end;
                while(i<end) {
                    const unsigned n=std::min(4U,end-i);
                    if(n==1)tasks[singles++]=tasks[i++];
                    else {
                        auto& group=descriptors[groups++];group={};group.count=n;
                        for(unsigned t=0;t<n;++t)group.tasks[t]=tasks[i++];
                    }
                }
            }
            const auto group_bytes=groups*sizeof(FlashNextCachedExpertGroup);
            if(groups) {
                CUDA_CHECK(cudaMemcpyAsync(batch_tasks_->p,group_descriptors_->data(),
                    group_bytes,cudaMemcpyHostToDevice,stream));
                const auto* device_groups=static_cast<const FlashNextCachedExpertGroup*>(batch_tasks_->p);
                for(const auto& span:gemm_spans)
                    prefill_gemm_->launch(span.expert,device_groups+span.first_group,span.routes,stream);
                if(groups>gemm_groups)
                    flash_next_cached_expert_group_launch(device_groups+gemm_groups,groups-gemm_groups,stream);
            }
            if(singles) {
                auto* singleton_tasks=reinterpret_cast<FlashNextCachedExpertTask*>(
                    static_cast<std::byte*>(batch_tasks_->p)+group_bytes);
                CUDA_CHECK(cudaMemcpyAsync(singleton_tasks,tasks,
                    singles*sizeof(FlashNextCachedExpertTask),cudaMemcpyHostToDevice,stream));
                flash_next_cached_expert_batch_launch(singleton_tasks,singles,stream);
            }
            if(!gemm_spans.empty()) {
                std::lock_guard lock(mutex_);
                stats_.gemm_experts += gemm_spans.size();
                for(const auto& span:gemm_spans)stats_.gemm_routes += span.routes;
            }
            if(v100_perf_telemetry_enabled() && !gemm_spans.empty()) {
                unsigned routes=0;for(const auto& span:gemm_spans)routes+=span.routes;
                std::fprintf(stderr,"{\"kind\":\"expert_resident_gemm\",\"experts\":%zu,\"routes\":%u}\n",
                    gemm_spans.size(),routes);
            }
            if(timing_enabled_) {
                std::lock_guard lock(mutex_);
                stats_.grouped_tasks += count-singles;
                stats_.hit_kernel_launches += 2*unsigned(groups>gemm_groups)+2*unsigned(singles>0);
                // cuBLAS may issue multiple private kernels; count GEMM experts/routes
                // separately rather than invent a kernel launch total.
            }
        } else {
            CUDA_CHECK(cudaMemcpyAsync(batch_tasks_->p, batch_descriptors_->data(),
                count*sizeof(FlashNextCachedExpertTask), cudaMemcpyHostToDevice, stream));
            flash_next_cached_expert_batch_launch(
                static_cast<const FlashNextCachedExpertTask*>(batch_tasks_->p), count, stream);
        }
        if(timing_enabled_) {
            std::lock_guard lock(mutex_);
            stats_.hit_submission_us += std::chrono::duration<double,std::micro>(
                std::chrono::steady_clock::now()-submitted).count();
            if(!grouping_layer_)stats_.hit_kernel_launches += 2;
            stats_.grouped_groups += groups;
        }
    }
}
void FlashNextExpertCache::begin_download(std::size_t bytes, cudaStream_t stream) {
    if (consumers_.empty()) return;
    if (bytes > outputs_->bytes) throw std::out_of_range("expert cache result capacity");
    submit_consumers(stream);
    if (timing_enabled_) CUDA_CHECK(cudaEventRecord(result_copy_start_, stream));
    CUDA_CHECK(cudaMemcpyAsync(result_buffer_->data(), outputs_->p, bytes,
                               cudaMemcpyDeviceToHost, stream));
    if (timing_enabled_) CUDA_CHECK(cudaEventRecord(hit_stop_, stream));
}
void FlashNextExpertCache::begin_device_results(cudaStream_t stream) {
    if (consumers_.empty()) return;
    submit_consumers(stream);
    if (timing_enabled_) CUDA_CHECK(cudaEventRecord(hit_stop_, stream));
}
double FlashNextExpertCache::finish_consumers(cudaStream_t stream,double* wait_us) {
    if (wait_us) *wait_us=0;
    if (consumers_.empty()) return 0;
    const auto waiting=timing_enabled_?std::chrono::steady_clock::now():
        std::chrono::steady_clock::time_point{};
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if(wait_us&&timing_enabled_)*wait_us=std::chrono::duration<double,std::micro>(
        std::chrono::steady_clock::now()-waiting).count();
    float gpu_ms=0;
    if(timing_enabled_)CUDA_CHECK(cudaEventElapsedTime(&gpu_ms,hit_start_,hit_stop_));
    std::lock_guard lock(mutex_);check_failure();
    for(auto [slot,path]:consumers_){(void)path;--entries_[slot].leases;}
    consumers_.clear();
    return double(gpu_ms)*1000;
}
double FlashNextExpertCache::finish_device_results(cudaStream_t stream,double* wait_us) {
    return finish_consumers(stream,wait_us);
}
double FlashNextExpertCache::finish_download(std::span<float> output, cudaStream_t stream,
    double* wait_us, double* result_copy_us) {
    if (result_copy_us) *result_copy_us = 0;
    if (wait_us) *wait_us = 0;
    if (consumers_.empty()) return 0;
    const auto waiting = timing_enabled_ ? std::chrono::steady_clock::now() :
        std::chrono::steady_clock::time_point{};
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (wait_us && timing_enabled_) *wait_us = std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now()-waiting).count();
    float gpu_ms = 0;
    if (timing_enabled_) CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, hit_start_, hit_stop_));
    if (result_copy_us && timing_enabled_) {
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, result_copy_start_, hit_stop_));
        *result_copy_us = double(ms)*1000;
    }
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
    std::unique_ptr<FlashNextExpertProfile> reseed;
    {
        std::lock_guard lock(mutex_);
        if (!consumers_.empty()) throw std::logic_error("cache reset with outstanding consumers");
        std::fill(entries_.begin(), entries_.end(), Entry{});
        stats_ = {};
        layer_totals_ = {};
        ++telemetry_generation_;
        epoch_ = 0;
        if (heat_) heat_->reset();
        admissions_enabled_ = !static_profile_;
        if (static_profile_ && original_profile_)
            reseed=std::make_unique<FlashNextExpertProfile>(*original_profile_);
    }
    if (reseed) seed(*reseed);
}
void FlashNextExpertCache::admit(unsigned layer,std::span<const std::int32_t> ids) {
    const auto started = timing_enabled_ ? std::chrono::steady_clock::now() :
        std::chrono::steady_clock::time_point{};
    std::lock_guard lock(mutex_);check_failure();
    if(layer>=48)throw std::invalid_argument("invalid cache layer");
    if (heat_) heat_->observe(layer, ids);
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
            bool colder = victim == end;
            if (!colder && adaptive_heat_ && e.state != State::Empty) {
                const auto score = heat_->score(layer, e.expert);
                const auto old_score = heat_->score(layer, entries_[victim].expert);
                colder = score < old_score || (score == old_score && e.epoch < entries_[victim].epoch);
            } else if (!colder) {
                colder = e.epoch < entries_[victim].epoch;
            }
            if(colder||e.state==State::Empty)victim=i;
            if(e.state==State::Empty)break;
        }
        if(victim==end){++stats_.victim_declined_calls;break;}
        if (adaptive_heat_ && entries_[victim].state != State::Empty &&
            heat_->score(layer, id) <= heat_->score(layer, entries_[victim].expert))
            continue;
        if(entries_[victim].state!=State::Empty){++stats_.evicted;++layer_totals_[layer].evicted;}
        entries_[victim]={id,State::Uploading,++epoch_,0};
        queue_.push_back(victim);++stats_.admitted;++layer_totals_[layer].admitted;++admitted;
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
                auto& layer_total = layer_totals_[slot/budget_.slots_per_layer];
                layer_total.fill_bytes += kExpertSlotBytes;
                ++layer_total.ready;
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
FlashNextExpertCache::LayerSnapshot FlashNextExpertCache::layer_snapshot(unsigned layer) const {
    if (layer >= 48) throw std::invalid_argument("invalid telemetry cache layer");
    std::lock_guard lock(mutex_);
    check_failure();
    LayerSnapshot snapshot;
    snapshot.totals = stats_;
    for (unsigned i = layer*budget_.slots_per_layer; i < (layer+1)*budget_.slots_per_layer; ++i) {
        snapshot.ready += entries_[i].state == State::Ready;
        snapshot.uploading += entries_[i].state == State::Uploading;
        snapshot.leased += entries_[i].leases != 0;
    }
    return snapshot;
}
v100_compare::CacheSnapshot FlashNextExpertCache::telemetry_snapshot(unsigned layer, bool ids) const {
    if (layer >= 48) throw std::invalid_argument("invalid telemetry cache layer");
    std::lock_guard lock(mutex_);
    check_failure();
    v100_compare::CacheSnapshot snapshot;
    snapshot.generation = telemetry_generation_;
    snapshot.capacity_experts = budget_.slots_per_layer;
    snapshot.capacity_bytes = std::uint64_t(budget_.slots_per_layer)*kExpertSlotBytes;
    const auto& totals = layer_totals_[layer];
    snapshot.hits_total = totals.hits; snapshot.misses_total = totals.misses;
    snapshot.admissions_total = totals.admitted; snapshot.fills_total = totals.ready;
    snapshot.evictions_total = totals.evicted; snapshot.fill_bytes_total = totals.fill_bytes;
    if (ids) snapshot.resident_ids.emplace();
    for (unsigned i = layer*budget_.slots_per_layer; i < (layer+1)*budget_.slots_per_layer; ++i) {
        const auto& entry = entries_[i];
        snapshot.ready += entry.state == State::Ready;
        snapshot.uploading += entry.state == State::Uploading;
        snapshot.leased += entry.leases != 0;
        if (ids && entry.state == State::Ready) snapshot.resident_ids->push_back(entry.expert);
    }
    return snapshot;
}
FlashNextExpertCacheStats FlashNextExpertCache::stats() const {
    std::lock_guard lock(mutex_);check_failure();return stats_;
}
} // namespace ninfer::targets::qwen3_8_flash_next::detail
