#include "targets/qwen3_8_flash_next/impl/expert_gemm.h"
#include "core/device.h"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <stdexcept>
#include <string_view>
#include <string>

namespace ninfer::targets::qwen3_8_flash_next::detail {
namespace {
void blas_check(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("Flash-Next expert GEMM: cuBLAS status " + std::to_string(int(status)));
}
__global__ void expand(Nvfp4ExpertMatrixView matrix, __half* weights) {
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=unsigned(matrix.rows*matrix.columns)) return;
    const unsigned row=i/matrix.columns, col=i%matrix.columns, group=col/16;
    const auto* codes=reinterpret_cast<const unsigned char*>(matrix.codes);
    const auto* scales=reinterpret_cast<const unsigned char*>(matrix.scales);
    const unsigned offset=((row/128)*(matrix.columns/64)+group/4)*512
        +(row%32)*16+((row%128)/32)*4+group%4;
    const unsigned byte=codes[i/2];
    const auto code=ops::detail::decode_nvfp4_e2m1x2(byte);
    // E2M1 * E4M3 is exactly representable in FP16 for all finite scale codes.
    // Keep the expert divisor in FP32 after each GEMM, avoiding FP16 overflow
    // and needless division rounding at every weight.
    const float raw=((col&1)?code.y:code.x)*ops::detail::decode_nvfp4_e4m3(scales[offset]);
    weights[i]=__float2half_rn(raw);
}
__global__ void expand_pair(HostNvfp4ExpertPairView expert, __half2* gu, __half2* down) {
    unsigned pair=blockIdx.x*blockDim.x+threadIdx.x;
    constexpr unsigned gu_pairs=1280*2560/2,down_pairs=2560*640/2;
    if(pair>=gu_pairs+down_pairs)return;
    const bool is_gu=pair<gu_pairs;
    const auto matrix=is_gu?expert.gate_up:expert.down;
    __half2* output=is_gu?gu:down;
    if(!is_gu)pair-=gu_pairs;
    const unsigned columns=matrix.columns,row=pair/(columns/2),col=(pair%(columns/2))*2;
    const unsigned group=col/16;
    const unsigned offset=((row/128)*(columns/64)+group/4)*512
        +(row%32)*16+((row%128)/32)*4+group%4;
    // Both codes share one scale. Decode once per packed byte and write the
    // same two FP16 values as the scalar expansion, without changing divisors.
    const auto code=ops::detail::decode_nvfp4_e2m1x2(
        reinterpret_cast<const unsigned char*>(matrix.codes)[pair]);
    const float scale=ops::detail::decode_nvfp4_e4m3(
        reinterpret_cast<const unsigned char*>(matrix.scales)[offset]);
    output[pair]=__floats2half2_rn(code.x*scale,code.y*scale);
}
__device__ float block_max(float x) {
    __shared__ float warps[8];
    for (int delta=16;delta;delta/=2) x=fmaxf(x,__shfl_down_sync(0xffffffffU,x,delta));
    if ((threadIdx.x&31)==0) warps[threadIdx.x/32]=x;
    __syncthreads();
    x=threadIdx.x<8?warps[threadIdx.x]:0.0F;
    if (threadIdx.x<32) {
        for (int delta=16;delta;delta/=2) x=fmaxf(x,__shfl_down_sync(0xffffffffU,x,delta));
        if (threadIdx.x==0) warps[0]=x;
    }
    __syncthreads();return warps[0];
}
__device__ float power_scale(float maximum) {
    return maximum>0 ? ldexpf(1.0F,ilogbf(maximum)) : 1.0F;
}
__global__ void gather(const FlashNextCachedExpertGroup* groups, unsigned offset,
    unsigned count, __half* x, float* scales) {
    const unsigned token=blockIdx.x;
    if (token>=count) {
        for(unsigned k=threadIdx.x;k<2560;k+=blockDim.x)x[token*2560+k]=__float2half_rn(0);
        if(threadIdx.x==0)scales[token]=1;return;
    }
    const unsigned route=offset+token;
    const auto* input=static_cast<const unsigned short*>(groups[route/4].tasks[route%4].input);
    float maximum=0;
    for(unsigned k=threadIdx.x;k<2560;k+=blockDim.x)
        maximum=fmaxf(maximum,fabsf(__uint_as_float(unsigned(input[k])<<16)));
    const float scale=power_scale(block_max(maximum));
    if(threadIdx.x==0)scales[token]=scale;
    for(unsigned k=threadIdx.x;k<2560;k+=blockDim.x)
        x[token*2560+k]=__float2half_rn(__uint_as_float(unsigned(input[k])<<16)/scale);
}
__global__ void activate(const float* gu, const float* input_scales, const float* divisor,
    __half* h, float* h_scales) {
    const unsigned token=blockIdx.x;
    float values[3];float maximum=0;
    for(unsigned j=0;j<3;++j) {
        const unsigned row=threadIdx.x+j*256;float value=0;
        if(row<640) {
            const float gate=(gu[token*1280+row]*input_scales[token])/ *divisor;
            const float up=(gu[token*1280+640+row]*input_scales[token])/ *divisor;
            value=(gate/(1.0F+expf(-gate)))*up;
        }
        values[j]=value;maximum=fmaxf(maximum,fabsf(value));
    }
    const float scale=power_scale(block_max(maximum));
    if(threadIdx.x==0)h_scales[token]=scale;
    for(unsigned j=0;j<3;++j) {
        const unsigned row=threadIdx.x+j*256;
        if(row<640)h[token*640+row]=__float2half_rn(values[j]/scale);
    }
}
__global__ void scatter(const FlashNextCachedExpertGroup* groups, unsigned offset,
    unsigned count, const float* y, const float* h_scales, const float* divisor) {
    const unsigned token=blockIdx.x;
    if(token>=count)return;
    const unsigned route=offset+token;
    float* output=groups[route/4].tasks[route%4].output;
    for(unsigned k=threadIdx.x;k<2560;k+=blockDim.x)
        output[k]=(y[token*2560+k]*h_scales[token])/ *divisor;
}
}

namespace {
constexpr bool selected_v100_default() {
#if defined(NINFER_VOLTA_BUILD)
    return true;
#else
    return false;
#endif
}
bool selected_binary_mode(const char* name) {
    const char* value=std::getenv(name);
    if(!value || !*value)return selected_v100_default();
    if(std::string_view(value)=="0")return false;
    if(std::string_view(value)=="1")return true;
    throw std::invalid_argument(std::string(name)+" must be 0 or 1");
}
}

bool flash_next_expert_gemm_requested() {
    const char* p=std::getenv("NINFER_V100_PREFILL_EXPERT_GEMM");
    if(!p || !*p)return selected_v100_default();
    if(std::string_view(p)=="simt")return false;
    if(std::string_view(p)=="fp16")return true;
    throw std::invalid_argument("NINFER_V100_PREFILL_EXPERT_GEMM must be simt or fp16");
}
bool flash_next_expert_pair_expand_requested() {
    const char* value=std::getenv("NINFER_V100_EXPERT_PAIR_EXPAND");
    if(!value || !*value || std::string_view(value)=="0")return false;
    if(std::string_view(value)=="1")return true;
    throw std::invalid_argument("NINFER_V100_EXPERT_PAIR_EXPAND must be 0 or 1");
}
bool flash_next_resident_expert_gemm_requested() {
    return selected_binary_mode("NINFER_V100_PREFILL_RESIDENT_GEMM");
}
bool flash_next_cpu_stream_overlap_requested() {
    return selected_binary_mode("NINFER_V100_PREFILL_CPU_STREAM_OVERLAP");
}
struct FlashNextExpertGemm::Impl {
    bool pair_expand=flash_next_expert_pair_expand_requested();
    DeviceBuffer gu{1280*2560*2ULL}, down{2560*640*2ULL};
    DeviceBuffer x{tile_routes*2560*2ULL}, gate_up{tile_routes*1280*4ULL};
    DeviceBuffer h{tile_routes*640*2ULL}, y{tile_routes*2560*4ULL};
    DeviceBuffer input_scales{tile_routes*4ULL}, h_scales{tile_routes*4ULL};
    DeviceBuffer workspace{4ULL*1024*1024};
    cublasHandle_t handle=nullptr;
    cudaStream_t compute=nullptr; bool bound=false;
    Impl() {
        blas_check(cublasCreate(&handle));
        try {
            blas_check(cublasSetMathMode(handle,static_cast<cublasMath_t>(
                CUBLAS_TENSOR_OP_MATH | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)));
        } catch(...) {cublasDestroy(handle);handle=nullptr;throw;}
        if(pair_expand)std::fprintf(stderr,"v100.expert_pair_expand=1\n");
    }
    ~Impl(){if(handle)cublasDestroy(handle);}
};
FlashNextExpertGemm::FlashNextExpertGemm():impl_(std::make_unique<Impl>()){}
FlashNextExpertGemm::~FlashNextExpertGemm()=default;
void FlashNextExpertGemm::launch(const HostNvfp4ExpertPairView& expert,
    const FlashNextCachedExpertGroup* groups,unsigned routes,cudaStream_t stream) {
    auto& m=*impl_;
    if(!routes || !groups || expert.gate_up.rows!=1280 || expert.gate_up.columns!=2560 ||
       expert.down.rows!=2560 || expert.down.columns!=640)
        throw std::invalid_argument("invalid Flash-Next expert GEMM geometry");
    if(m.bound && m.compute!=stream)throw std::invalid_argument("expert GEMM scratch requires one compute stream");
    m.bound=true;m.compute=stream;
    blas_check(cublasSetStream(m.handle,stream));
    blas_check(cublasSetWorkspace(m.handle,m.workspace.p,m.workspace.bytes));
    if(m.pair_expand) {
        expand_pair<<<((1280*2560+2560*640)/2+255)/256,256,0,stream>>>(expert,
            static_cast<__half2*>(m.gu.p),static_cast<__half2*>(m.down.p));
        CUDA_CHECK(cudaGetLastError());
    } else {
        expand<<<(1280*2560+255)/256,256,0,stream>>>(expert.gate_up,static_cast<__half*>(m.gu.p));
        CUDA_CHECK(cudaGetLastError());
        expand<<<(2560*640+255)/256,256,0,stream>>>(expert.down,static_cast<__half*>(m.down.p));
        CUDA_CHECK(cudaGetLastError());
    }
    const float one=1,zero=0;
    for(unsigned offset=0;offset<routes;offset+=tile_routes) {
        const unsigned count=std::min(tile_routes,routes-offset), padded=(count+7)&~7U;
        gather<<<padded,256,0,stream>>>(groups,offset,count,static_cast<__half*>(m.x.p),static_cast<float*>(m.input_scales.p));
        CUDA_CHECK(cudaGetLastError());
        blas_check(cublasGemmEx(m.handle,CUBLAS_OP_T,CUBLAS_OP_N,1280,padded,2560,
            &one,m.gu.p,CUDA_R_16F,2560,m.x.p,CUDA_R_16F,2560,&zero,m.gate_up.p,CUDA_R_32F,1280,
            CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        activate<<<padded,256,0,stream>>>(static_cast<const float*>(m.gate_up.p),static_cast<const float*>(m.input_scales.p),
            expert.gate_up.weight_scale_divisor,static_cast<__half*>(m.h.p),static_cast<float*>(m.h_scales.p));
        CUDA_CHECK(cudaGetLastError());
        blas_check(cublasGemmEx(m.handle,CUBLAS_OP_T,CUBLAS_OP_N,2560,padded,640,
            &one,m.down.p,CUDA_R_16F,640,m.h.p,CUDA_R_16F,640,&zero,m.y.p,CUDA_R_32F,2560,
            CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        scatter<<<count,256,0,stream>>>(groups,offset,count,static_cast<const float*>(m.y.p),
            static_cast<const float*>(m.h_scales.p),expert.down.weight_scale_divisor);
        CUDA_CHECK(cudaGetLastError());
    }
}
} // namespace ninfer::targets::qwen3_8_flash_next::detail
