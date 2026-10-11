#include "targets/qwen3_8_flash_next/impl/expert_gemm.h"
#include "targets/qwen3_8_flash_next/impl/expert_stream.h"
#include "targets/qwen3_8_flash_next/impl/cpu_expert_reference.h"
#include "core/device.h"
#include <bit>
#include <algorithm>
#include <array>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
using namespace ninfer;
using namespace ninfer::targets::qwen3_8_flash_next::detail;
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
int main() {try {
    int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess || !devices)return 77;
    DeviceContext device;
    std::mt19937 rng(6307);
    std::vector<std::byte> gc(1'638'400),gs(204'800),dc(819'200),ds(102'400);
    for(auto& v:gc)v=std::byte(rng()&255);for(auto& v:dc)v=std::byte(rng()&255);
    // All finite nonnegative E4M3 scales, zero and subnormal scales included.
    for(auto& v:gs)v=std::byte(rng()%127);for(auto& v:ds)v=std::byte(rng()%127);
    DeviceBuffer weights(kExpertSlotBytes),input(3*2560*2ULL),output((512*2560+16)*4ULL);
    std::vector<std::byte> packed(kExpertSlotBytes);
    std::copy(gc.begin(),gc.end(),packed.begin());
    std::copy(gs.begin(),gs.end(),packed.begin()+1'638'400);
    std::copy(dc.begin(),dc.end(),packed.begin()+1'843'200);
    std::copy(ds.begin(),ds.end(),packed.begin()+2'662'400);
    std::vector<std::uint16_t> inputs(3*2560);
    for(unsigned t=0;t<3;++t)for(unsigned k=0;k<2560;++k) {
        const float amplitude=t==0?.002F:t==1?1e5F:1e-12F;
        const float f=(int(rng()%201)-100)*amplitude;
        auto b=std::bit_cast<std::uint32_t>(f);b+=0x7fff+((b>>16)&1);inputs[t*2560+k]=b>>16;
    }
    input.copy_from_host(inputs.data(),inputs.size()*2);
    const auto* base=static_cast<const std::byte*>(weights.p);
    HostNvfp4ExpertPairView gpu{
        {base,base+1'638'400,reinterpret_cast<const float*>(base+2'764'800),1,1280,2560},
        {base+1'843'200,base+2'662'400,reinterpret_cast<const float*>(base+2'764'804),1,2560,640}};
    std::vector<FlashNextCachedExpertGroup> descriptors(128);
    DeviceBuffer groups(descriptors.size()*sizeof(descriptors[0]));
    DeviceBuffer simt_activations(512*640*2ULL);
    FlashNextExpertGemm gemm;
    CpuNvfp4ExpertReferenceScratch scratch;
    for(float divisor:{3.1415927F,64.F,.001F}) {
        std::memcpy(packed.data()+2'764'800,&divisor,4);
        std::memcpy(packed.data()+2'764'804,&divisor,4);
        weights.copy_from_host(packed.data(),packed.size());
        HostNvfp4ExpertPairView host{{gc.data(),gs.data(),&divisor,1,1280,2560},
                                    {dc.data(),ds.data(),&divisor,1,2560,640}};
        std::array<std::vector<float>,3> oracle;
        for(unsigned t=0;t<3;++t) {
            oracle[t].resize(2560);
            // Independent scalar decoder and sequential FP32 FMA formula, with
            // FP32 SiLU and intermediate. No candidate FP16 casts in the oracle.
            flash_next_cpu_nvfp4_expert_pair_reference_fp32_intermediate(host,
                std::span(inputs.data()+t*2560,2560),oracle[t],scratch);
        }
        for(unsigned count:{1U,7U,20U,31U,32U,33U,128U,256U,257U,512U}) {
            for(unsigned g=0;g<descriptors.size();++g) {
                descriptors[g]={};descriptors[g].count=std::min(4U,count>g*4?count-g*4:0U);
                for(unsigned t=0;t<descriptors[g].count;++t) {
                    const unsigned route=g*4+t;
                    descriptors[g].tasks[t]={gpu,static_cast<const std::uint16_t*>(input.p)+(route%3)*2560,
                        static_cast<std::uint16_t*>(simt_activations.p)+route*640,static_cast<float*>(output.p)+route*2560};
                }
            }
            groups.copy_from_host(descriptors.data(),descriptors.size()*sizeof(descriptors[0]));
            output.fill(0xA5);
            gemm.launch(gpu,static_cast<const FlashNextCachedExpertGroup*>(groups.p),count,device.stream);
            CUDA_CHECK(cudaStreamSynchronize(device.stream));
            std::vector<float> actual(count*2560+16);
            output.copy_to_host(actual.data(),actual.size()*4);
            double worst=0,min_cosine=1;
            for(unsigned t=0;t<count;++t) {
                double err=0,norm=0,dot=0,aa=0;
                for(unsigned k=0;k<2560;++k) {
                    const double a=actual[t*2560+k],b=oracle[t%3][k];
                    require(std::isfinite(a)&&std::isfinite(b),"nonfinite FP16 expert/oracle value");
                    err+=(a-b)*(a-b);norm+=b*b;dot+=a*b;aa+=a*a;
                }
                const double nrmse=std::sqrt(err/std::max(norm,1e-300));
                const double cosine=dot/std::sqrt(aa*norm);
                require(nrmse<=.002 && cosine>=.99999,"FP16 expert failed independent FP32 formula gate");
                worst=std::max(worst,nrmse);min_cosine=std::min(min_cosine,cosine);
            }
            for(unsigned k=0;k<16;++k)require(std::bit_cast<unsigned>(actual[count*2560+k])==0xA5A5A5A5,
                "FP16 expert scatter overwrote guard");
            std::cout<<"gemm.divisor="<<divisor<<" routes="<<count<<" nrmse="<<worst<<" cosine="<<min_cosine<<'\n';
        }
    }
    // Zero input must produce exact zero, including padded GEMM columns.
    CUDA_CHECK(cudaMemset(input.p,0,input.bytes));
    gemm.launch(gpu,static_cast<const FlashNextCachedExpertGroup*>(groups.p),7,device.stream);
    CUDA_CHECK(cudaStreamSynchronize(device.stream));
    std::vector<float> zero(7*2560);output.copy_to_host(zero.data(),zero.size()*4);
    for(float x:zero)require(x==0,"zero expert input did not produce zero output");
    input.copy_from_host(inputs.data(),inputs.size()*2);
    // Same compact expert, inputs and destinations; expansion is included in
    // GEMM's device interval. Copies and CPU staging are excluded from both.
    cudaEvent_t begin=nullptr,end=nullptr;
    CUDA_CHECK(cudaEventCreate(&begin));CUDA_CHECK(cudaEventCreate(&end));
    for(unsigned count:{8U,16U,20U,32U,33U,64U,128U,256U,512U}) {
        for(unsigned g=0;g<descriptors.size();++g) {
            descriptors[g].count=std::min(4U,count>g*4?count-g*4:0U);
            for(auto& task:descriptors[g].tasks)task.input=input.p;
        }
        groups.copy_from_host(descriptors.data(),descriptors.size()*sizeof(descriptors[0]));
        for(bool use_gemm:{false,true}) {
            double total_ms=0;
            for(unsigned repeat=0;repeat<8;++repeat) {
                CUDA_CHECK(cudaEventRecord(begin,device.stream));
                if(use_gemm)gemm.launch(gpu,static_cast<const FlashNextCachedExpertGroup*>(groups.p),count,device.stream);
                else flash_next_cached_expert_group_launch(static_cast<const FlashNextCachedExpertGroup*>(groups.p),(count+3)/4,device.stream);
                CUDA_CHECK(cudaEventRecord(end,device.stream));CUDA_CHECK(cudaEventSynchronize(end));
                float ms=0;CUDA_CHECK(cudaEventElapsedTime(&ms,begin,end));
                if(repeat>=3)total_ms+=ms;
            }
            std::cout<<"expert.cost routes="<<count<<" backend="<<(use_gemm?"fp16":"simt")
                     <<" mean_ms="<<total_ms/5<<'\n';
        }
    }
    CUDA_CHECK(cudaEventDestroy(begin));CUDA_CHECK(cudaEventDestroy(end));
    std::cout<<"Independent represented-NVFP4 FP32 expert GEMM qualification passed\n";
    return 0;
} catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
