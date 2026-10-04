#include "targets/qwen3_8_flash_next/impl/expert_cache.h"
#include "targets/qwen3_8_flash_next/impl/cpu_expert_reference.h"
#include "core/device.h"
#include "targets/qwen3_8_flash_next/impl/cpu_expert_pool.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
using namespace ninfer;
using namespace ninfer::targets::qwen3_8_flash_next::detail;
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
std::uint16_t bf16(float x){auto b=std::bit_cast<std::uint32_t>(x);b+=0x7fff+((b>>16)&1);return b>>16;}
void compare(const std::vector<float>& actual,const std::vector<float>& expected){
    double error=0,norm=0,dot=0,aa=0;
    for(unsigned i=0;i<actual.size();++i){require(std::isfinite(actual[i]),"nonfinite cached output");
        error+=std::pow(double(actual[i])-expected[i],2);norm+=double(expected[i])*expected[i];
        dot+=double(actual[i])*expected[i];aa+=double(actual[i])*actual[i];}
    auto nrmse=std::sqrt(error/std::max(norm,1e-30)),cosine=dot/std::sqrt(aa*norm);
    std::cout<<"cache.oracle.nrmse="<<nrmse<<" cosine="<<cosine<<'\n';
    require(nrmse<=.002&&cosine>=.99999,"cache expert mathematical oracle mismatch");
}
int main(){try{
    int count=0;if(cudaGetDeviceCount(&count)!=cudaSuccess||count==0)return 77;
    DeviceContext device;
    auto b=flash_next_expert_cache_budget(25ULL<<30,32ULL<<30,64ULL<<20,false);
    require(b.slots_per_layer>32,"cache budget must derive capacity rather than assume 32");
    require(b.cache_bytes+b.transfer_bytes+(7ULL<<30)<=(28ULL<<30),"cache operating ceiling");
    require(flash_next_expert_cache_budget(1ULL<<30,32ULL<<30,64ULL<<20,false).slots_per_layer==0,"cache safety reserve");
    require(flash_next_expert_cache_budget(10ULL<<30,32ULL<<30,64ULL<<20,false).slots_per_layer<b.slots_per_layer,"larger state reduces cache capacity");
    std::mt19937 rng(19);
    std::vector<std::byte> gate_codes(3*1'638'400),gate_scales(3*204'800);
    std::vector<std::byte> down_codes(3*819'200),down_scales(3*102'400);
    for(auto& x:gate_codes)x=std::byte(rng()&255);for(auto& x:down_codes)x=std::byte(rng()&255);
    for(auto& x:gate_scales)x=std::byte(0x28+(rng()%16));
    for(auto& x:down_scales)x=std::byte(0x28+(rng()%16));
    float divisors[3]={64,72,80};
    HostNvfp4ExpertLayerView layer{
        {gate_codes.data(),gate_scales.data(),divisors,512,1280,2560,1'638'400,204'800},
        {down_codes.data(),down_scales.data(),divisors,512,2560,640,819'200,102'400}};
    HostNvfp4ExpertTableView host;host.layers.fill(layer);
    FlashNextExpertCache cache(host,128,false,2);
    std::int32_t ids[]={0,1,2};
    cache.admit(0,ids);cache.drain();
    require(cache.stats().admitted==1,"admission cap one");
    cache.admit(0,ids);cache.drain();require(cache.stats().admitted==2,"next missing admission");
    auto v=cache.ready_view(0,0);
    auto check_bytes=[](const std::byte* dev,const std::byte* src,std::size_t bytes){
        std::vector<std::byte> copy(bytes);CUDA_CHECK(cudaMemcpy(copy.data(),dev,bytes,cudaMemcpyDeviceToHost));
        require(std::memcmp(copy.data(),src,bytes)==0,"canonical cache bytes changed");};
    check_bytes(v.gate_up.codes,gate_codes.data(),1'638'400);check_bytes(v.gate_up.scales,gate_scales.data(),204'800);
    check_bytes(v.down.codes,down_codes.data(),819'200);check_bytes(v.down.scales,down_scales.data(),102'400);
    float divisor=0;CUDA_CHECK(cudaMemcpy(&divisor,v.down.weight_scale_divisor,4,cudaMemcpyDeviceToHost));
    require(divisor==divisors[0],"canonical divisor changed");
    std::vector<std::uint16_t> input(128*2560);
    for(auto& x:input)x=bf16((int(rng()%201)-100)*.002F);
    DeviceBuffer d_input(input.size()*2);d_input.copy_from_host(input.data(),input.size()*2);
    CpuNvfp4ExpertReferenceScratch scratch;
    for(unsigned tokens:{1U,2U,3U,4U,6U,8U,128U}){
        cache.begin_layer(true);
        std::vector<float> out(tokens*2560),expected(tokens*2560);
        for(unsigned t=0;t<tokens;++t){
            require(cache.execute(0,t%2,static_cast<std::uint16_t*>(d_input.p)+t*2560,t,device.stream),"Ready hit missing");
            flash_next_cpu_nvfp4_expert_pair_reference(layer.expert(t%2),
                std::span(input.data()+t*2560,2560),std::span(expected.data()+t*2560,2560),scratch);
        }
        cache.download(out,device.stream);compare(out,expected);
    }
    cache.set_grouped_prefill(false);
    for(unsigned tokens:{1U,3U,5U,8U,128U}) {
        std::vector<float> scalar(tokens*2560),batched(tokens*2560);
        for(bool batch:{false,true}) {
            cache.set_batched_prefill(batch);cache.begin_layer(true);
            for(unsigned t=0;t<tokens;++t)
                require(cache.execute(0,t%2,static_cast<std::uint16_t*>(d_input.p)+t*2560,
                    t,device.stream),"batch Ready hit missing");
            cache.download(batch?batched:scalar,device.stream);
        }
        require(scalar==batched,"batched expert outputs changed scalar arithmetic");
        std::cout<<"cache.batch.exact_parity.tokens="<<tokens<<'\n';
    }
    for (unsigned paths : {1U, 5U, 10U}) {
        std::vector<float> scalar(paths*2560), batched(paths*2560), expected(paths*2560);
        for (bool batch : {false,true}) {
            cache.set_batched_decode(batch);
            cache.begin_layer(false);
            for (unsigned path=0; path<paths; ++path) {
                require(cache.execute(0,path%2,
                    static_cast<std::uint16_t*>(d_input.p)+path*2560,
                    path,device.stream), "decode batch Ready hit missing");
                if (batch) flash_next_cpu_nvfp4_expert_pair_reference(layer.expert(path%2),
                    std::span(input.data()+path*2560,2560),
                    std::span(expected.data()+path*2560,2560),scratch);
            }
            cache.download(batch?batched:scalar,device.stream);
        }
        require(scalar==batched,"decode batching changed exact expert outputs");
        compare(batched,expected);
        std::cout<<"cache.decode_batch.exact_parity.paths="<<paths<<'\n';
    }
    cache.set_batched_decode(false);
    for(unsigned tokens:{1U,3U,5U,8U,128U}) {
        std::vector<float> scalar(tokens*2560),grouped(tokens*2560),expected(tokens*2560);
        for(bool group:{false,true}) {
            cache.set_batched_prefill(false);cache.set_grouped_prefill(group);cache.begin_layer(true);
            for(unsigned t=0;t<tokens;++t) {
                require(cache.execute(0,t%2,static_cast<std::uint16_t*>(d_input.p)+t*2560,
                    t,device.stream),"group Ready hit missing");
                if(group)flash_next_cpu_nvfp4_expert_pair_reference(layer.expert(t%2),
                    std::span(input.data()+t*2560,2560),std::span(expected.data()+t*2560,2560),scratch);
            }
            cache.download(group?grouped:scalar,device.stream);
        }
        require(scalar==grouped,"grouped expert outputs changed scalar arithmetic");
        compare(grouped,expected);
        std::cout<<"cache.group.exact_parity.tokens="<<tokens<<'\n';
    }
    cache.set_grouped_prefill(false);
    {
        constexpr unsigned paths=4;
        DeviceBuffer direct_device(paths*2560*sizeof(float));
        std::vector<float> direct(paths*2560),expected(paths*2560);
        cache.set_batched_prefill(true);cache.set_grouped_prefill(true);
        cache.begin_layer(true);
        for(unsigned path=0;path<paths;++path) {
            require(cache.execute_to(0,path%2,
                static_cast<std::uint16_t*>(d_input.p)+path*2560,
                static_cast<float*>(direct_device.p)+path*2560,path,device.stream),
                "direct device Ready hit missing");
            flash_next_cpu_nvfp4_expert_pair_reference(layer.expert(path%2),
                std::span(input.data()+path*2560,2560),
                std::span(expected.data()+path*2560,2560),scratch);
        }
        cache.begin_device_results(device.stream);
        cache.finish_device_results(device.stream);
        direct_device.copy_to_host(direct.data(),direct.size()*sizeof(float));
        compare(direct,expected);
        std::cout<<"cache.direct_device.paths="<<paths<<'\n';
    }
    cache.set_batched_prefill(false);cache.begin_layer(false);
    HostExpertWorkerPool pool(4, false);
    for (unsigned hits : {0U, 1U, 4U}) {
        std::vector<float> serial(4*2560), overlap(4*2560);
        cache.set_batched_prefill(true);cache.set_grouped_prefill(true);
        for (bool concurrent : {false, true}) {
            cache.begin_layer(true);
            auto& output = concurrent ? overlap : serial;
            std::vector<HostExpertTask> tasks;
            for (unsigned path = 0; path < 4; ++path) {
                if (path < hits) {
                    require(cache.execute(0, path%2,
                        static_cast<std::uint16_t*>(d_input.p)+path*2560,
                        path, device.stream), "mixed schedule hit missing");
                } else {
                    tasks.push_back({.expert = layer.expert(path%2),
                        .input = input.data()+path*2560,
                        .output = output.data()+path*2560});
                }
            }
            cache.begin_download(output.size()*sizeof(float), device.stream);
            if (!concurrent) cache.finish_download(output, device.stream);
            pool.run(tasks);
            if (concurrent) cache.finish_download(output, device.stream);
        }
        require(serial == overlap, "CPU/GPU join changed private expert outputs");
        std::cout << "cache.schedule.exact_parity.hits=" << hits << " paths=4\n";
    }
    cache.set_batched_prefill(false);cache.begin_layer(false);
    cache.set_grouped_prefill(false);
    // A leased slot cannot be recycled even when it would otherwise be the LRU victim.
    require(cache.execute(0,0,d_input.p,0,device.stream),"lease hit");
    std::int32_t new_id=2;cache.admit(0,std::span(&new_id,1));cache.drain();
    cache.ready_view(0,0);cache.ready_view(0,2);
    std::vector<float> out(2560);cache.download(out,device.stream);
    require(!cache.execute(1,0,d_input.p,0,device.stream),"layer namespace collision");
    cache.admit(1,std::span(ids,1));cache.drain();cache.ready_view(1,0);
    require(cache.stats().evicted==1,"LRU victim not replaced");
    for (bool batched : {false,true}) {
        FlashNextExpertCache single_slot(host,1,false,1);
        single_slot.set_batched_decode(batched);
        single_slot.admit(0,std::span(ids,1));single_slot.drain();
        single_slot.begin_layer(false);
        require(single_slot.execute(0,0,d_input.p,0,device.stream),"single-slot lease");
        // A batched descriptor holds its lease even before kernels are submitted.
        single_slot.admit(0,std::span(&new_id,1));single_slot.drain();
        require(single_slot.stats().admitted==1,"leased slot was recycled");
        single_slot.download(out,device.stream);
        std::vector<float> expected(2560);
        flash_next_cpu_nvfp4_expert_pair_reference(layer.expert(0),
            std::span(input.data(),2560),expected,scratch);
        compare(out,expected);
        single_slot.admit(0,std::span(&new_id,1));single_slot.drain();
        require(single_slot.stats().admitted==2,"released slot was not recyclable");
        single_slot.ready_view(0,2);
        single_slot.begin_layer(false);
        require(single_slot.execute(0,2,d_input.p,0,device.stream),"replacement decode hit");
        single_slot.download(out,device.stream);
        flash_next_cpu_nvfp4_expert_pair_reference(layer.expert(2),
            std::span(input.data(),2560),expected,scratch);
        compare(out,expected);
        std::cout<<"cache.decode_lease.batched="<<batched<<'\n';
    }
    FlashNextExpertCache cap_two(host,1,false,2,2);
    cap_two.admit(0,ids);cap_two.drain();
    require(cap_two.stats().admitted==2,"admission cap two");
    cap_two.ready_view(0,0);cap_two.ready_view(0,1);
    cap_two.admit(0,ids);cap_two.drain();
    require(cap_two.stats().admitted==3,"duplicate admission consumed cap");
    require(cap_two.stats().maximum_outstanding<=4,"fill queue exceeded bound");
    cap_two.reset();
    require(!cap_two.execute(0,2,d_input.p,0,device.stream),"reset retained Ready entry");
    cap_two.admit(0,ids);cap_two.drain();
    require(cap_two.stats().admitted==2,"reset did not restore admissions");
    for (const char* policy : {"heat", "decay"}) {
    setenv("NINFER_V100_EXPERT_POLICY", policy, 1);
    setenv("NINFER_V100_EXPERT_DECAY", "0.5", 1);
    setenv("NINFER_V100_EXPERT_DECAY_INTERVAL", "1", 1);
    FlashNextExpertCache heat_cache(host,1,false,1);
    unsetenv("NINFER_V100_EXPERT_POLICY");
    unsetenv("NINFER_V100_EXPERT_DECAY");
    unsetenv("NINFER_V100_EXPERT_DECAY_INTERVAL");
    const std::int32_t hot[] = {0,0,0,0};
    heat_cache.admit(0, hot); heat_cache.drain();
    const std::int32_t cold = 1;
    heat_cache.admit(0, std::span(&cold,1)); heat_cache.drain();
    require(heat_cache.stats().admitted == 1,"heat evicted hotter resident");
    heat_cache.begin_layer(false);
    require(heat_cache.execute(0,0,d_input.p,0,device.stream),"heat resident lease");
    const std::int32_t rising[] = {1,1,1,1,1};
    heat_cache.admit(0, rising); heat_cache.drain();
    require(heat_cache.stats().admitted == 1,"heat evicted leased resident");
    heat_cache.download(out,device.stream);
    heat_cache.admit(0, std::span(&cold,1)); heat_cache.drain();
    heat_cache.ready_view(0,1);
    require(heat_cache.stats().admitted == 2,"heat did not replace colder released resident");
    heat_cache.admit(1, std::span(ids,1)); heat_cache.drain();
    heat_cache.ready_view(0,1); heat_cache.ready_view(1,0);
    }
    host.model_id="fixture";host.weights_id="canonical";
    FlashNextExpertProfile profile;
    profile.model_id=host.model_id;profile.weights_id=host.weights_id;
    for (auto& row : profile.ranking) {
        for (int i=0;i<512;++i) row[i]=i;
        std::swap(row[0],row[2]); // Seed expert 2, unlike default ascending/LRU order.
    }
    FlashNextExpertCache seeded(host,1,false,1);
    auto wrong=profile;wrong.weights_id="other";
    bool rejected=false;
    try { seeded.seed(wrong); } catch (const std::invalid_argument&) { rejected=true; }
    require(rejected && seeded.stats().admitted==0,"seed identity rejection mutated cache");
    seeded.seed(profile);
    require(seeded.stats().ready==48 && seeded.stats().admitted==48,
        "startup seed returned before canonical Ready publication");
    require(seeded.stats().maximum_outstanding<=4,"startup seed exceeded queue bound");
    for (unsigned layer_id=0;layer_id<48;++layer_id) seeded.ready_view(layer_id,2);
    seeded.begin_layer(false);
    require(seeded.execute(0,2,d_input.p,0,device.stream),"ranked startup expert missing");
    seeded.download(out,device.stream);
    std::vector<float> expected_seed(2560);
    flash_next_cpu_nvfp4_expert_pair_reference(layer.expert(2),
        std::span(input.data(),2560),expected_seed,scratch);
    compare(out,expected_seed);
    rejected=false;
    try { seeded.seed(profile); } catch (const std::logic_error&) { rejected=true; }
    require(rejected,"seed overwrote a nonempty cache");
    setenv("NINFER_V100_EXPERT_POLICY","static",1);
    FlashNextExpertCache static_cache(host,1,false,1);
    unsetenv("NINFER_V100_EXPERT_POLICY");
    static_cache.seed(profile);
    static_cache.admit(0,ids);static_cache.drain();
    require(static_cache.stats().admitted==48,"static profile admitted dynamic experts");
    static_cache.ready_view(0,2);
    static_cache.reset();static_cache.admit(0,ids);static_cache.drain();
    require(static_cache.stats().admitted==48,"static profile reset changed fixed residency");
    setenv("NINFER_V100_EXPERT_POLICY","profile",1);
    setenv("NINFER_V100_EXPERT_PRIOR_WEIGHT","512",1);
    FlashNextExpertCache prior_cache(host,1,false,1);
    unsetenv("NINFER_V100_EXPERT_POLICY");unsetenv("NINFER_V100_EXPERT_PRIOR_WEIGHT");
    prior_cache.seed(profile);
    const std::int32_t newcomer=1;
    prior_cache.admit(0,std::span(&newcomer,1));prior_cache.drain();
    require(prior_cache.stats().admitted==48,"profile prior lost hotter seeded expert");
    prior_cache.begin_layer(false);
    require(prior_cache.execute(0,2,d_input.p,0,device.stream),"profile prior lease missing");
    const std::int32_t hotter[]={1,1,1,1,1,1};
    prior_cache.admit(0,hotter);prior_cache.drain();
    require(prior_cache.stats().admitted==48,"profile prior evicted leased expert");
    prior_cache.download(out,device.stream);
    prior_cache.admit(0,std::span(&newcomer,1));prior_cache.drain();
    prior_cache.ready_view(0,1);prior_cache.ready_view(1,2);
    require(prior_cache.stats().admitted==49,"measured heat failed to overcome prior");
    const auto save_path=std::filesystem::temp_directory_path()/"ninfer-v100-cache-learned.json";
    prior_cache.save_profile(save_path);
    const auto learned=FlashNextExpertProfile::load(save_path.c_str(),host.model_id,host.weights_id);
    require(learned.ranking[0][0]==1 && learned.ranking[1][0]==2,
        "learned profile failed resident-first layer ordering");
    std::filesystem::remove(save_path);
    // Saving with LRU must observe heat without changing LRU replacement.
    setenv("NINFER_V100_EXPERT_POLICY","lru",1);
    setenv("NINFER_V100_EXPERT_PROFILE_SAVE",save_path.c_str(),1);
    {
        FlashNextExpertCache learning_lru(host,1,false,1);
        unsetenv("NINFER_V100_EXPERT_POLICY");unsetenv("NINFER_V100_EXPERT_PROFILE_SAVE");
        learning_lru.admit(0,std::array<std::int32_t,4>{0,0,0,0});learning_lru.drain();
        learning_lru.admit(0,std::span(&newcomer,1));learning_lru.drain();
        learning_lru.ready_view(0,1);
        learning_lru.enable_shutdown_profile_save();
    }
    const auto saved=FlashNextExpertProfile::load(save_path.c_str(),host.model_id,host.weights_id);
    require(saved.ranking[0][0]==1 && saved.ranking[0][1]==0,
        "shutdown save or observation-only LRU failed");
    std::ifstream saved_input(save_path);
    nlohmann::json saved_json; saved_input>>saved_json;
    require(saved_json["frequency"][0][0]==4 && saved_json["frequency"][0][1]==1,
        "LRU saving did not preserve measured frequencies");
    saved_input.close();
    std::filesystem::remove(save_path);
    std::cout<<"PASS: budget, canonical bytes, oracle, admission, LRU, heat, leases, namespaces\n";
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
