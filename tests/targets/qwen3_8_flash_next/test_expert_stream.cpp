#include "targets/qwen3_8_flash_next/impl/expert_stream.h"
#include "targets/qwen3_8_flash_next/impl/cpu_expert_reference.h"
#include "core/device.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
using namespace ninfer;
using namespace ninfer::targets::qwen3_8_flash_next::detail;
void require(bool x,const char* s) { if(!x) throw std::runtime_error(s); }
int main() { try {
    int devices=0; if(cudaGetDeviceCount(&devices)!=cudaSuccess || !devices) return 77;
    DeviceContext device;
    std::mt19937 rng(19);
    std::vector<std::byte> gc(3*1'638'400),gs(3*204'800),dc(3*819'200),ds(3*102'400);
    for(auto& v:gc)v=std::byte(rng()&255);for(auto& v:dc)v=std::byte(rng()&255);
    for(auto& v:gs)v=std::byte(0x28+rng()%16);for(auto& v:ds)v=std::byte(0x28+rng()%16);
    float divisors[3]={64,72,80};
    HostNvfp4ExpertLayerView layer{
        {gc.data(),gs.data(),divisors,3,1280,2560,1'638'400,204'800},
        {dc.data(),ds.data(),divisors,3,2560,640,819'200,102'400}};
    std::vector<std::uint16_t> input(2560);
    for(auto& v:input) {
        float f=(int(rng()%201)-100)*.002F;
        auto b=std::bit_cast<std::uint32_t>(f);b+=0x7fff+((b>>16)&1);v=b>>16;
    }
    std::array<std::vector<float>,3> reference;
    CpuNvfp4ExpertReferenceScratch scratch;
    for(unsigned i=0;i<3;++i) {
        reference[i].resize(2560);
        flash_next_cpu_nvfp4_expert_pair_reference(layer.expert(i),input,reference[i],scratch);
    }
    DeviceBuffer d_input(input.size()*2); d_input.copy_from_host(input.data(),input.size()*2);
    const std::array<unsigned,11> counts{1,2,3,4,5,8,16,17,128,256,2048};
    std::size_t total=0;for(auto n:counts)total+=n;
    std::vector<float> actual(total*2560+16);
    DeviceBuffer output(actual.size()*4);output.fill(0xA5);
    FlashNextExpertStream ring(2048);
    require(ring.device_bytes()>4*kExpertSlotBytes && ring.pinned_bytes()>4*kExpertSlotBytes,
            "staging memory accounting excludes bounded scratch");
    std::size_t offset=0;
    for(unsigned i=0;i<counts.size();++i) {
        std::vector<FlashNextStreamRoute> routes(counts[i]);
        for(unsigned t=0;t<counts[i];++t)
            routes[t]={d_input.p,static_cast<float*>(output.p)+(offset+t)*2560};
        ring.submit(layer.expert(i%3),routes,device.stream);offset+=counts[i];
    }
    // A rejected CPU-side group must leave all previously submitted consumers valid.
    bool rejected=false;try { ring.submit(layer.expert(0),{},device.stream); }
    catch(const std::invalid_argument&) { rejected=true; }
    require(rejected,"empty stream group accepted");
    ring.finish();
    require(ring.submitted_experts()==counts.size() &&
            ring.expert_h2d_bytes()==counts.size()*kExpertSlotBytes,"stream transfer accounting");
    CUDA_CHECK(cudaMemcpy(actual.data(),output.p,actual.size()*4,cudaMemcpyDeviceToHost));
    offset=0;
    for(unsigned i=0;i<counts.size();++i) {
        double err=0,norm=0,dot=0,aa=0;
        for(unsigned t=0;t<counts[i];++t)for(unsigned j=0;j<2560;++j) {
            double x=actual[(offset+t)*2560+j], y=reference[i%3][j];
            require(std::isfinite(x),"nonfinite streamed expert");
            err+=(x-y)*(x-y);norm+=y*y;dot+=x*y;aa+=x*x;
        }
        const double nrmse=std::sqrt(err/std::max(norm,1e-30)),cosine=dot/std::sqrt(aa*norm);
        std::cout<<"stream.routes="<<counts[i]<<" nrmse="<<nrmse<<" cosine="<<cosine<<'\n';
        require(nrmse<=.002 && cosine>=.99999,"streamed expert mathematical oracle mismatch");
        offset+=counts[i];
    }
    for(unsigned j=0;j<16;++j)require(std::bit_cast<std::uint32_t>(actual[total*2560+j])==0xA5A5A5A5,
                                      "stream wrote past route destinations");
    // Reuse after the explicit completion boundary and destructor completion are both valid.
    FlashNextStreamRoute last{d_input.p,static_cast<float*>(output.p)};
    ring.submit(layer.expert(2),std::span(&last,1),device.stream);ring.finish();
    { FlashNextExpertStream temporary(1);
      temporary.submit(layer.expert(1),std::span(&last,1),device.stream); }
    std::vector<float> tail(2560);
    CUDA_CHECK(cudaMemcpy(tail.data(),output.p,tail.size()*4,cudaMemcpyDeviceToHost));
    double tail_error=0,tail_norm=0;
    for(unsigned j=0;j<2560;++j) {
        const double d=double(tail[j])-reference[1][j];
        require(std::isfinite(tail[j]),"destructor lost pending output");
        tail_error+=d*d;tail_norm+=double(reference[1][j])*reference[1][j];
    }
    require(std::sqrt(tail_error/std::max(tail_norm,1e-30))<=.002,"destructor completion oracle mismatch");
    std::cout<<"SV3 bounded four-slot staging ring passed; runtime integration pending\n";
    return 0;
} catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; } }
