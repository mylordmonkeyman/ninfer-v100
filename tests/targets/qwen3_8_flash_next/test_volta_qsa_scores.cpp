#include "core/device.h"
#include "targets/qwen3_8_flash_next/impl/qsa_attention_kernels.h"
#include <cuda_bf16.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

using namespace ninfer;
using namespace ninfer::targets::qwen3_8_flash_next::detail;
static std::uint16_t bf(float f) { const auto b = __float2bfloat16_rn(f); return __bfloat16_as_ushort(b); }
static float value(std::uint16_t b) { return __bfloat162float(__ushort_as_bfloat16(b)); }

int main() {
#if !defined(NINFER_VOLTA_BUILD)
    return 77;
#else
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
        DeviceContext device;
        constexpr int P = 33, H = 24, D = 256, T = 6;
        const std::vector<int> positions{0, 2, 30, 254, 510, 2054};
        const std::vector<int> counts{0, 0, 7, 63, 127, 257};
        std::vector<int> table(P), blocks(T * 512);
        for (int p = 0; p < P; ++p) table[p] = (p * 13) % P;
        for (int t = 0; t < T; ++t)
            for (int b = 0; b < counts[t]; ++b) blocks[t * 512 + b] = t == T-1 ? b * 2 : b;
        std::mt19937 rng(317);
        std::uniform_real_distribution<float> random(-0.6F, 0.6F);
        std::vector<std::uint16_t> q(T * H * D), keys(P * 2 * 64 * D), values(keys.size());
        for (auto& x : q) x = bf(random(rng));
        for (auto& x : keys) x = bf(random(rng));
        for (auto& x : values) x = bf(random(rng));
        DeviceBuffer dq(q.size()*2), di(T*4), dc(T*4), db(blocks.size()*4), dt(P*4), out(q.size()*2);
        di.copy_from_host(positions.data(), T*4); dc.copy_from_host(counts.data(), T*4);
        db.copy_from_host(blocks.data(), blocks.size()*4); dt.copy_from_host(table.data(), P*4);
        Tensor query(dq.p,DType::BF16,{D,H,T}), indices(di.p,DType::I32,{T});
        Tensor selected(db.p,DType::I32,{512,T}), count(dc.p,DType::I32,{T});
        Tensor attended(out.p,DType::BF16,{D,H,T});
        for (bool fp8 : {false,true}) {
            std::vector<std::uint8_t> k8(keys.size()), v8(keys.size());
            const std::uint8_t codes[]{0x20,0x28,0x30,0xa0,0xa8,0xb0};
            const float decoded[]{0.125F,0.25F,0.5F,-0.125F,-0.25F,-0.5F};
            std::vector<float> kf(keys.size()), vf(keys.size());
            for (std::size_t i=0;i<keys.size();++i) {
                const int a=rng()%6,b=rng()%6; k8[i]=codes[a];v8[i]=codes[b];
                kf[i]=fp8?decoded[a]:value(keys[i]);vf[i]=fp8?decoded[b]:value(values[i]);
            }
            DeviceBuffer dk(keys.size()*(fp8?1:2)),dv(values.size()*(fp8?1:2));
            dk.copy_from_host(fp8?static_cast<void*>(k8.data()):static_cast<void*>(keys.data()),dk.bytes);
            dv.copy_from_host(fp8?static_cast<void*>(v8.data()):static_cast<void*>(values.data()),dv.bytes);
            QsaAttentionCacheView cache{};
            cache.key_pages=Tensor(dk.p,fp8?DType::FP8_E4M3FN:DType::BF16,{D,64,2,P});
            cache.value_pages=Tensor(dv.p,fp8?DType::FP8_E4M3FN:DType::BF16,{D,64,2,P});
            cache.block_tables=Tensor(dt.p,DType::I32,{P,1});
            // Also exercise the finite-half range fallback, with a bounded score:
            // a large query component multiplied by a tiny BF16 key component.
            for (bool overflow : {false,true}) {
                auto represented=q;
                if (overflow) for (int t=0;t<T;++t) for(int h=0;h<H;++h) represented[(t*H+h)*D]=bf(65536.0F);
                if (overflow) {
                    auto narrow_keys = keys;
                    for (std::size_t i = 0; i < keys.size(); i += D) {
                        kf[i] = fp8 ? 0.0F : std::ldexp(1.0F, -18);
                        narrow_keys[i] = bf(kf[i]);
                        k8[i] = 0;
                    }
                    dk.copy_from_host(fp8 ? static_cast<void*>(k8.data()) :
                        static_cast<void*>(narrow_keys.data()), dk.bytes);
                }
                dq.copy_from_host(represented.data(),represented.size()*2);
                std::vector<double> reference(q.size());
                for(int t=0;t<T;++t) for(int h=0;h<H;++h) {
                    std::vector<int> candidates;
                    for(int b=0;b<counts[t];++b) for(int j=0;j<4;++j) candidates.push_back(blocks[t*512+b]*4+j);
                    const int complete=(positions[t]+1)/4;
                    for(int j=0;j<((positions[t]+1)&3);++j) candidates.push_back(complete*4+j);
                    std::vector<double> scores(candidates.size());double maximum=-INFINITY;
                    for(std::size_t c=0;c<candidates.size();++c) {
                        const int pos=candidates[c];const auto base=((table[pos/64]*2+h/12)*64+pos%64)*D;
                        double dot=0;for(int j=0;j<D;++j) dot+=double(value(represented[(t*H+h)*D+j]))*kf[base+j];
                        scores[c]=dot/16.;maximum=std::max(maximum,scores[c]);
                    }
                    double sum=0;for(auto& score:scores){score=std::exp(score-maximum);sum+=score;}
                    for(std::size_t c=0;c<candidates.size();++c){const int pos=candidates[c];const auto base=((table[pos/64]*2+h/12)*64+pos%64)*D;for(int j=0;j<D;++j)reference[(t*H+h)*D+j]+=scores[c]/sum*vf[base+j];}
                }
                for(bool mma:{false,true}) {
                    flash_next_qsa_volta_attend_launch(query,indices,0,selected,count,cache,attended,device.stream,mma);
                    device.synchronize();std::vector<std::uint16_t> actual(q.size());out.copy_to_host(actual.data(),out.bytes);
                    double max_reference=0;
                    for (double x : reference) max_reference=std::max(max_reference,std::abs(x));
                    double squared=0,norm=0;for(std::size_t i=0;i<actual.size();++i){const double expected=value(bf(float(reference[i])));const double diff=value(actual[i])-expected;if(!std::isfinite(value(actual[i])))throw std::runtime_error("nonfinite attended value");if(std::abs(diff)>1e-3*max_reference+1e-2*std::abs(expected))
                        throw std::runtime_error("QSA independent pointwise threshold failed");
                    squared+=diff*diff;norm+=expected*expected;}
                    const double relative=std::sqrt(squared/norm);
                    std::cout<<"FP64 oracle fp8="<<fp8<<" overflow="<<overflow<<" mma="<<mma<<" relative_l2="<<relative<<'\n';
                    if(relative>1e-3)throw std::runtime_error("QSA independent oracle threshold failed");
                }
                if (!overflow) {
                    cudaEvent_t a,b;CUDA_CHECK(cudaEventCreate(&a));CUDA_CHECK(cudaEventCreate(&b));
                    for (int batch : {1,2,4,8,32,128,512,1024,4096,8192}) {
                        std::vector<std::uint16_t> batch_q(batch*H*D);
                        std::vector<int> batch_positions(batch),batch_counts(batch),batch_blocks(batch*512);
                        for (int t=0;t<batch;++t) {
                            const int source=t%T;
                            std::copy_n(q.data()+source*H*D,H*D,batch_q.data()+t*H*D);
                            batch_positions[t]=positions[source];batch_counts[t]=counts[source];
                            std::copy_n(blocks.data()+source*512,512,batch_blocks.data()+t*512);
                        }
                        DeviceBuffer bq(batch_q.size()*2),bo(batch_q.size()*2),bi(batch*4),bc(batch*4),bs(batch_blocks.size()*4);
                        bq.copy_from_host(batch_q.data(),bq.bytes);bi.copy_from_host(batch_positions.data(),bi.bytes);
                        bc.copy_from_host(batch_counts.data(),bc.bytes);bs.copy_from_host(batch_blocks.data(),bs.bytes);
                        Tensor tq(bq.p,DType::BF16,{D,H,batch}),to(bo.p,DType::BF16,{D,H,batch});
                        Tensor ti(bi.p,DType::I32,{batch}),tc(bc.p,DType::I32,{batch}),ts(bs.p,DType::I32,{512,batch});
                        const int iterations=batch<1024?20:5;
                        for(bool mma:{false,true}) {
                            for(int i=0;i<3;++i)flash_next_qsa_volta_attend_launch(tq,ti,0,ts,tc,cache,to,device.stream,mma);
                            CUDA_CHECK(cudaEventRecord(a,device.stream));
                            for(int i=0;i<iterations;++i)flash_next_qsa_volta_attend_launch(tq,ti,0,ts,tc,cache,to,device.stream,mma);
                            CUDA_CHECK(cudaEventRecord(b,device.stream));CUDA_CHECK(cudaEventSynchronize(b));
                            float ms;CUDA_CHECK(cudaEventElapsedTime(&ms,a,b));
                            std::cout<<"Score-path timing fp8="<<fp8<<" T="<<batch<<" mma="<<mma<<" us="<<ms*1000/iterations<<'\n';
                        }
                    }
                    CUDA_CHECK(cudaEventDestroy(a));CUDA_CHECK(cudaEventDestroy(b));
                }
            }
        }
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
#endif
}
