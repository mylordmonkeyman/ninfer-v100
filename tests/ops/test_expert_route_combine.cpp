#include "ninfer/ops/expert_route_combine.h"
#include "ops/op_tester.h"
#include "core/device.h"

#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

int run_case(int tokens, unsigned gpu_paths, std::size_t pitch) {
    const std::size_t routes_count = std::size_t(tokens)*10*2560;
    std::vector<float> routes(routes_count), alpha(std::size_t(tokens)*10);
    for (std::size_t i=0;i<routes_count;++i) {
        // Mix signs and magnitudes to exercise cancellation. Values represent
        // both CPU-miss and GPU-hit outputs; their placement shares one layout.
        routes[i] = std::ldexp(float(int((i*17+13)%127)-63), int(i%9)-6);
    }
    for (std::size_t i=0;i<alpha.size();++i) alpha[i] = float((i*7)%31)/31;
    GuardedDeviceBuffer device_routes(routes.size()*sizeof(float));
    GuardedDeviceBuffer device_alpha(alpha.size()*sizeof(float));
    GuardedDeviceBuffer device_output(std::size_t(tokens)*pitch*sizeof(float));
    device_routes.fill(0xcd);
    device_alpha.copy_from_host(alpha.data(),alpha.size()*sizeof(float));
    // Model mixed provenance by separately populating contiguous hit/miss spans.
    // The kernel receives exactly the same destination layout for 0/50/100% hits.
    for (int token=0;token<tokens;++token) {
        const std::size_t base=std::size_t(token)*10*2560;
        for (unsigned span=0;span<2;++span) {
            const unsigned begin = span == 0 ? 0 : gpu_paths;
            const unsigned count = span == 0 ? gpu_paths : 10-gpu_paths;
            if (!count) continue;
            CUDA_CHECK(cudaMemcpyAsync(static_cast<float*>(device_routes.data())+base+begin*2560,
                routes.data()+base+begin*2560, std::size_t(count)*2560*sizeof(float),
                cudaMemcpyHostToDevice,nullptr));
        }
    }
    device_output.fill(0xa5);
    ops::expert_route_combine(static_cast<const float*>(device_routes.data()),
        static_cast<const float*>(device_alpha.data()),static_cast<float*>(device_output.data()),
        tokens,pitch,nullptr);
    cuda_synchronize();
    const auto got=from_device<float>(device_output.data(),std::size_t(tokens)*pitch);
    std::vector<float> host_fma(std::size_t(tokens)*2560), actual(host_fma.size());
    int failures=0;
    // FP64 naive mathematical oracle is independent of the production FMA order.
    // Its error bound is the standard gamma_10 bound for ten FP32 FMAs.
    constexpr double unit=std::numeric_limits<float>::epsilon()/2.0;
    constexpr double gamma=10*unit/(1-10*unit);
    for (int token=0;token<tokens;++token) {
        for (std::size_t row=0;row<2560;++row) {
            double mathematical=0, magnitude=0;
            float paired=0;
            for (unsigned path=0;path<10;++path) {
                const float x=routes[(std::size_t(token)*10+path)*2560+row];
                const float a=alpha[std::size_t(token)*10+path];
                mathematical += double(a)*double(x);
                magnitude += std::abs(double(a)*double(x));
                paired=std::fma(a,x,paired);
            }
            const auto i=std::size_t(token)*2560+row;
            actual[i]=got[std::size_t(token)*pitch+row];host_fma[i]=paired;
            if (!std::isfinite(actual[i]) || std::abs(double(actual[i])-mathematical)>gamma*magnitude)
                ++failures;
        }
        for (std::size_t row=2560;row<pitch;++row) {
            std::uint32_t bits=0;
            std::memcpy(&bits,&got[std::size_t(token)*pitch+row],sizeof(bits));
            if(bits!=0xa5a5a5a5U)++failures;
        }
    }
    const std::string label="route combine T="+std::to_string(tokens)+
        " GPU paths="+std::to_string(gpu_paths);
    std::vector<std::uint32_t> actual_bits(actual.size()), expected_bits(host_fma.size());
    std::memcpy(actual_bits.data(),actual.data(),actual.size()*sizeof(float));
    std::memcpy(expected_bits.data(),host_fma.data(),host_fma.size()*sizeof(float));
    failures+=verify_exact(label.c_str(),actual_bits,expected_bits);
    failures+=device_routes.verify_guards(label);
    failures+=device_alpha.verify_guards(label);
    failures+=device_output.verify_guards(label);
    std::cout<<label<<" failures="<<failures<<'\n';
    return failures;
}
} // namespace

int main() {
    if(cuda_unavailable())return 77;
    int failures=0;
    for (int tokens : {1,2,3,4,5,128,256,4096})
        for (unsigned gpu_paths : {0U,5U,10U})
            failures+=run_case(tokens,gpu_paths,3520);
    failures+=run_case(1,10,2560);
    std::cout<<(failures?"FAIL":"PASS")<<": expert route combine\n";
    return failures?1:0;
}
