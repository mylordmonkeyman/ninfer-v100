#include "core/arena.h"
#include "core/device.h"
#include "ninfer/ops/linear.h"
#include "ops/linear/fp8/fp8_cutlass_sm70.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
using namespace ninfer;
namespace {
void require(bool b,const char* m){if(!b)throw std::runtime_error(m);}
float bf(std::uint16_t b){return std::bit_cast<float>(std::uint32_t(b)<<16);}
std::uint16_t encode_bf(float f){auto b=std::bit_cast<std::uint32_t>(f);return (b+0x7fff+((b>>16)&1))>>16;}
// Independent mathematical E4M3FN decoder, including finite top exponent and subnormals.
float decode(std::uint8_t c){
    const int e=(c>>3)&15,m=c&7; const float sign=(c&128)?-1.F:1.F;
    require(!(e==15&&m==7),"NaN fixture code");
    return sign*(e?std::ldexp(1.F+float(m)/8,e-7):std::ldexp(float(m),-9));
}
void run(DeviceContext& device,int n,int k,int t,bool extreme,bool measure){
    constexpr int R=31,T=17;
    std::mt19937 rng(127+t+k+int(extreme));
    std::vector<std::uint8_t> prototypes(R*k),codes(std::size_t(n)*k);
    for(auto& c:prototypes){const unsigned mag=rng()%127; c=mag|((rng()&1)<<7);}
    if(extreme)for(int r=0;r<R;++r)for(int j=0;j<k;++j)
        prototypes[r*k+j]=std::array<std::uint8_t,8>{0,1,7,8,0x7e,0xfe,0x38,0xb8}[(r+j)%8];
    for(int r=0;r<n;++r)std::copy_n(prototypes.data()+(r%R)*k,k,codes.data()+std::size_t(r)*k);
    std::vector<std::uint16_t> xp(T*k),input(std::size_t(t)*k);
    for(int u=0;u<T;++u)for(int j=0;j<k;++j){
        const float amp=extreme?std::array<float,4>{.0001F,.125F,16.F,60000.F}[u%4]:.125F;
        xp[u*k+j]=encode_bf((int(rng()%201)-100)*amp/100.F);
    }
    for(int u=0;u<t;++u)std::copy_n(xp.data()+(u%T)*k,k,input.data()+std::size_t(u)*k);
    std::vector<float> scales(n);
    for(int r=0;r<n;++r)scales[r]=(r%11==0?0.F:std::ldexp(1.003F+(r%17)*.017F,(r%23)-15))*(r%3==0?-1.F:1.F);
    // Full logical outputs: repeated independent rows/token inputs reduce only
    // oracle cost, not GPU problem size or coverage. No FP16 staging in oracle.
    std::vector<float> dots(R*T);
    for(int u=0;u<T;++u)for(int r=0;r<R;++r){
        float sum=0;
        for(int j=0;j<k;++j)sum=std::fma(decode(prototypes[r*k+j]),bf(xp[u*k+j]),sum);
        dots[u*R+r]=sum;
    }
    const std::size_t scale_offset=(codes.size()+255)&~std::size_t(255);
    DeviceBuffer weights(scale_offset+n*4),xd(input.size()*2),yd((std::size_t(n)*t+128)*2);
    weights.copy_from_host(codes.data(),codes.size());weights.copy_from_host(scales.data(),n*4,scale_offset);
    xd.copy_from_host(input.data(),input.size()*2);
    Weight w{};w.payload=weights.p;w.payload_bytes=weights.bytes;w.qdata=weights.p;
    w.scales=static_cast<std::byte*>(weights.p)+scale_offset;
    w.qtype=QType::FP8_E4M3FN_ROW_F32S;w.layout=QuantLayout::RowScale;w.scale_dtype=DType::FP32;
    w.n=n;w.k=k;w.group=k;w.group_size=k;w.ndim=2;
    w.shape[0]=w.padded_shape[0]=n;w.shape[1]=w.padded_shape[1]=k;
    w.scale_ne[0]=n;w.scale_nb[0]=4;w.scale_nb[1]=w.scale_nb[2]=w.scale_nb[3]=n*4LL;
    Tensor x(xd.p,DType::BF16,{k,t}),y(yd.p,DType::BF16,{n,t});
    const auto capacity=ops::detail::fp8_f32_cutlass_sm70_workspace_bytes(n,k,t);
    WorkspaceArena ws(capacity+256);auto guard=ws.alloc_bytes(256);CUDA_CHECK(cudaMemset(guard.data,0xa5,256));
    auto launch=[&](bool expanded){
        if(expanded)ops::detail::fp8_f32_cutlass_sm70_launch(x,w,y,ws,device.stream);
        else ops::linear(x,w,y,ops::LinearPolicy::A16Only,ws,device.stream);
        require(ws.used()==256,"scratch lifetime leaked");
    };
    if(!measure)for(bool expanded:{false,true}){
        yd.fill(0xa5);launch(expanded);device.synchronize();
        std::vector<std::uint16_t> actual(std::size_t(n)*t+128);yd.copy_to_host(actual.data(),actual.size()*2);
        double err=0,norm=0,dot=0,anorm=0;
        std::vector<std::array<double,4>> per_token(t);
        for(int u=0;u<t;++u)for(int r=0;r<n;++r){
            const double expected=dots[(u%T)*R+r%R]*scales[r];const double a=bf(actual[std::size_t(u)*n+r]);
            require(std::isfinite(a),"nonfinite output");err+=(a-expected)*(a-expected);norm+=expected*expected;
            dot+=a*expected;anorm+=a*a;
            auto& stats=per_token[u];stats[0]+=(a-expected)*(a-expected);
            stats[1]+=expected*expected;stats[2]+=a*expected;stats[3]+=a*a;
            if(scales[r]==0.F)require(a==0.F,"zero row scale failed");
        }
        const double nrmse=std::sqrt(err/norm),cosine=dot/std::sqrt(norm*anorm);
        double worst_nrmse=0,worst_cosine=1;
        for(const auto& stats:per_token){
            worst_nrmse=std::max(worst_nrmse,std::sqrt(stats[0]/stats[1]));
            worst_cosine=std::min(worst_cosine,stats[2]/std::sqrt(stats[1]*stats[3]));
        }
        std::cout<<"fp8_f32.gate n="<<n<<" k="<<k<<" t="<<t<<" extreme="<<extreme<<" expanded="<<expanded
                 <<" nrmse="<<nrmse<<" cosine="<<cosine<<" worst_token_nrmse="<<worst_nrmse
                 <<" worst_token_cosine="<<worst_cosine<<" scratch_bytes="<<capacity<<std::endl;
        require(nrmse<=.002 && cosine>=.99999 && worst_nrmse<=.002 && worst_cosine>=.99999,"independent FP32 oracle failed");
        for(std::size_t i=std::size_t(n)*t;i<actual.size();++i)require(actual[i]==0xa5a5,"output guard damaged");
        std::array<std::uint8_t,256> check{};CUDA_CHECK(cudaMemcpy(check.data(),guard.data,256,cudaMemcpyDeviceToHost));
        for(auto b:check)require(b==0xa5,"workspace guard damaged");
    }
    if(!measure)return;
    cudaEvent_t start,end;CUDA_CHECK(cudaEventCreate(&start));CUDA_CHECK(cudaEventCreate(&end));
    for(int i=0;i<3;++i){launch(false);launch(true);}device.synchronize();
    std::array<std::vector<float>,2> timings;
    for(int i=0;i<7;++i)for(int order=0;order<2;++order){
        const bool expanded=((i%2)?1-order:order)!=0;
        CUDA_CHECK(cudaEventRecord(start,device.stream));launch(expanded);
        CUDA_CHECK(cudaEventRecord(end,device.stream));CUDA_CHECK(cudaEventSynchronize(end));
        float elapsed;CUDA_CHECK(cudaEventElapsedTime(&elapsed,start,end));
        timings[expanded].push_back(elapsed);
    }
    for(bool expanded:{false,true}){
        auto& ms=timings[expanded];std::sort(ms.begin(),ms.end());
        std::cout<<"fp8_f32.operator n="<<n<<" k="<<k<<" t="<<t<<" expanded="<<expanded
                 <<" median_ms="<<ms[3]<<" min_ms="<<ms.front()<<" max_ms="<<ms.back()
                 <<" useful_tflops="<<(2.0*n*k*t/(ms[3]*1e9))<<std::endl;
    }
    CUDA_CHECK(cudaEventDestroy(start));CUDA_CHECK(cudaEventDestroy(end));
}
}
int main(){try{
    int count=0;if(cudaGetDeviceCount(&count)!=cudaSuccess||!count)return 77;
    DeviceContext device(0);
    for(auto shape:{std::array<int,2>{16384,2560},std::array<int,2>{2560,6144}}){
        for(int t:{128,257,2048})run(device,shape[0],shape[1],t,false,false);
        run(device,shape[0],shape[1],129,true,false);
    }
    std::cout<<"fp8_f32.all_numerical_lifetime_gates=pass"<<std::endl;
    for(auto shape:{std::array<int,2>{16384,2560},std::array<int,2>{2560,6144}})
        for(int t:{128,257,2048})run(device,shape[0],shape[1],t,false,true);
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
