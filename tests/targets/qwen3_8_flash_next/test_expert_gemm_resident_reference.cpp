#include "core/device.h"
#include "targets/qwen3_8_flash_next/impl/expert_gemm.h"
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <algorithm>
#include <bit>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace ninfer;
using ninfer::targets::qwen3_8_flash_next::detail::FlashNextExpertGemm;
namespace {
void check(cublasStatus_t s) {
    if (s != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("resident reference cuBLAS failure");
}
struct Blas {
    cublasHandle_t handle=nullptr;
    Blas() { check(cublasCreate(&handle)); }
    ~Blas() { if(handle)cublasDestroy(handle); }
};
struct Events {
    cudaEvent_t begin=nullptr,end=nullptr;
    Events() { CUDA_CHECK(cudaEventCreate(&begin));CUDA_CHECK(cudaEventCreate(&end)); }
    ~Events() { if(begin)cudaEventDestroy(begin);if(end)cudaEventDestroy(end); }
};
// Same m/n/k, layouts, tile cap, accumulation, algorithm and workspace as the
// production expert GEMMs. Resident synthetic FP16 operands isolate GEMM cost;
// this is NOT an expert-pair semantic substitute (no SiLU/divisors/route joins).
struct Matrix {
    int m,k;
    DeviceBuffer w,x,y;
    std::vector<__half> host_w,host_x;
    Matrix(int rows,int columns):m(rows),k(columns),w(m*k*2ULL),
        x(FlashNextExpertGemm::tile_routes*k*2ULL),
        y((FlashNextExpertGemm::tile_routes*m+16)*4ULL),
        host_w(m*k),host_x(FlashNextExpertGemm::tile_routes*k) {
        for(int row=0;row<m;++row)for(int col=0;col<k;++col)
            host_w[row*k+col]=__float2half_rn(float(((row%31)*7+col*3)%23-11)/32);
        for(unsigned t=0;t<FlashNextExpertGemm::tile_routes;++t)for(int col=0;col<k;++col)
            host_x[t*k+col]=__float2half_rn(float((int(t%17)*5+col*7)%29-14)/16);
        w.copy_from_host(host_w.data(),host_w.size()*2);
        x.copy_from_host(host_x.data(),host_x.size()*2);
    }
    void launch(cublasHandle_t handle,unsigned n) {
        const float one=1,zero=0;
        check(cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,m,n,k,&one,
            w.p,CUDA_R_16F,k,x.p,CUDA_R_16F,k,&zero,y.p,CUDA_R_32F,m,
            CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    void qualify(cublasHandle_t handle,cudaStream_t stream,unsigned n) {
        // Sequential FP32 dot from represented FP16 operands. Repeated row and
        // token prototypes reduce oracle cost; every produced output is checked.
        std::vector<float> oracle(31*17);
        for(int r=0;r<31;++r)for(int t=0;t<17;++t) {
            float sum=0;
            for(int j=0;j<k;++j)sum=std::fma(__half2float(host_w[r*k+j]),
                __half2float(host_x[t*k+j]),sum);
            oracle[r*17+t]=sum;
        }
        y.fill(0xA5);launch(handle,n);CUDA_CHECK(cudaStreamSynchronize(stream));
        std::vector<float> actual(n*m+16);y.copy_to_host(actual.data(),actual.size()*4);
        double err=0,norm=0,dot=0,aa=0;
        for(unsigned t=0;t<n;++t)for(int r=0;r<m;++r) {
            double a=actual[t*m+r],b=oracle[(r%31)*17+t%17];
            if(!std::isfinite(a))throw std::runtime_error("nonfinite resident reference");
            err+=(a-b)*(a-b);norm+=b*b;dot+=a*b;aa+=a*a;
        }
        double nrmse=std::sqrt(err/norm),cosine=dot/std::sqrt(aa*norm);
        if(nrmse>.002 || cosine<.99999)throw std::runtime_error("resident FP32 dot oracle failure");
        for(unsigned i=0;i<16;++i)if(std::bit_cast<unsigned>(actual[n*m+i])!=0xA5A5A5A5)
            throw std::runtime_error("resident reference output guard failure");
        std::cout<<"resident.oracle m="<<m<<" n="<<n<<" k="<<k
                 <<" nrmse="<<nrmse<<" cosine="<<cosine<<'\n';
    }
};
}
int main() { try {
    int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess || !devices)return 77;
    DeviceContext device;Blas blas;DeviceBuffer workspace(4ULL*1024*1024);
    check(cublasSetMathMode(blas.handle,static_cast<cublasMath_t>(
        CUBLAS_TENSOR_OP_MATH | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)));
    check(cublasSetStream(blas.handle,device.stream));
    check(cublasSetWorkspace(blas.handle,workspace.p,workspace.bytes));
    Matrix gate_up(1280,2560),down(2560,640);
    for(unsigned n:{32U,40U,64U,128U,256U}) {
        gate_up.qualify(blas.handle,device.stream,n);down.qualify(blas.handle,device.stream,n);
    }
    cudaDeviceProp prop{};CUDA_CHECK(cudaGetDeviceProperties(&prop,0));
    std::cout<<"resident.device name="<<prop.name<<" sm="<<prop.major<<prop.minor
             <<" multiprocessors="<<prop.multiProcessorCount<<" tile="<<FlashNextExpertGemm::tile_routes<<'\n';
    Events events;
    for(unsigned routes:{32U,33U,64U,128U,256U,512U}) {
        for(int phase=0;phase<3;++phase) {
            std::vector<double> times;
            for(unsigned repeat=0;repeat<10;++repeat) {
                CUDA_CHECK(cudaEventRecord(events.begin,device.stream));
                for(unsigned offset=0;offset<routes;offset+=FlashNextExpertGemm::tile_routes) {
                    const unsigned n=(std::min(FlashNextExpertGemm::tile_routes,routes-offset)+7)&~7U;
                    if(phase!=1)gate_up.launch(blas.handle,n);
                    if(phase!=0)down.launch(blas.handle,n);
                }
                CUDA_CHECK(cudaEventRecord(events.end,device.stream));CUDA_CHECK(cudaEventSynchronize(events.end));
                float ms=0;CUDA_CHECK(cudaEventElapsedTime(&ms,events.begin,events.end));
                if(repeat>=3)times.push_back(ms);
            }
            std::sort(times.begin(),times.end());const double ms=times[times.size()/2];
            const double macs=phase==0?1280.*2560:phase==1?2560.*640:1280.*2560+2560.*640;
            unsigned issued_routes=0;
            for(unsigned o=0;o<routes;o+=FlashNextExpertGemm::tile_routes)
                issued_routes+=(std::min(FlashNextExpertGemm::tile_routes,routes-o)+7)&~7U;
            std::cout<<"resident.cost routes="<<routes<<" issued_routes="<<issued_routes
                     <<" phase="<<(phase==0?"gate_up":phase==1?"down":"pair")
                     <<" median_ms="<<ms<<" min_ms="<<times.front()<<" max_ms="<<times.back()
                     <<" useful_tflops="<<2*macs*routes/(ms*1e9)
                     <<" issued_tflops="<<2*macs*issued_routes/(ms*1e9)<<'\n';
        }
    }
    std::cout<<"Resident GEMM reference passed; no expansion, transfers, activation, gather or scatter timed\n";
    return 0;
} catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;} }
