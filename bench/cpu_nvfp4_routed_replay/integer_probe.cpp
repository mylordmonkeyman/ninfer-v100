#include "targets/qwen3_8_flash_next/impl/cpu_expert_pool.h"
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

using namespace ninfer::targets::qwen3_8_flash_next::detail;
namespace {
struct Matrix {
    int rows, columns;
    std::vector<std::byte> codes, scales;
    float divisor = 32;
    Matrix(int r, int c) : rows(r), columns(c), codes(std::size_t(r)*c/2),
                          scales(std::size_t(r)*c/16) {
        std::uint32_t random = 173;
        for (auto& code : codes) {
            random ^= random << 13; random ^= random >> 17; random ^= random << 5;
            code = std::byte(random & 255);
        }
        for (std::size_t i=0; i<scales.size(); ++i)
            scales[i] = std::byte(0x10 + (i * 17 % 41));
    }
    Nvfp4ExpertMatrixView view() {
        return {codes.data(), scales.data(), &divisor, 1, rows, columns};
    }
};
std::uint16_t bf16(float x) {
    auto bits = std::bit_cast<std::uint32_t>(x);
    bits += 0x7fff + ((bits >> 16) & 1);
    return bits >> 16;
}
void numerical(std::span<const float> reference, std::span<const float> actual) {
    double rr=0, aa=0, dot=0, error=0;
    for (std::size_t i=0; i<reference.size(); ++i) {
        if (!std::isfinite(reference[i]) || !std::isfinite(actual[i]))
            throw std::runtime_error("nonfinite CPU expert output");
        rr += double(reference[i])*reference[i]; aa += double(actual[i])*actual[i];
        dot += double(reference[i])*actual[i];
        error += std::pow(double(reference[i])-actual[i], 2);
    }
    const double nrmse = std::sqrt(error / std::max(rr, 1e-30));
    const double cosine = rr==0 && aa==0 ? 1 : dot/std::sqrt(std::max(rr*aa,1e-30));
    std::cout << "oracle nrmse=" << nrmse << " cosine=" << cosine << '\n';
    if (nrmse > .002 || cosine < .99999)
        throw std::runtime_error("CPU expert independent scalar gate failed");
}
}
int main() {
    if (!flash_next_cpu_nvfp4_avx2_available()) return 77;
    Matrix gate(1280,2560), down(2560,640);
    HostNvfp4ExpertPairView expert{gate.view(), down.view()};
    std::array<std::array<std::uint16_t,2560>,4> inputs;
    std::array<std::array<float,2560>,4> actual{}, reference{}, grouped{};
    CpuNvfp4ExpertReferenceScratch scratch;
    // Ordinary BF16 blocks, zeros, exponent spans requiring FP fallback, and
    // subnormal/very small values. The independent reference reads only public
    // represented BF16 inputs and the canonical compact weight format.
    for (int scenario=0; scenario<4; ++scenario) {
        gate.divisor = down.divisor = scenario==1 ? 1024 : 32;
        for (std::size_t i=0;i<gate.scales.size();++i)
            gate.scales[i]=std::byte(scenario==1 ? std::array<unsigned,4>{0,1,0x38,0x7e}[i%4] : 0x10+(i*17%41));
        for (std::size_t i=0;i<down.scales.size();++i)
            down.scales[i]=std::byte(scenario==1 ? std::array<unsigned,4>{0x7e,0x38,1,0}[i%4] : 0x10+(i*17%41));
        for (std::size_t t=0; t<4; ++t) {
            for (std::size_t k=0; k<2560; ++k) {
                float x = float(int((k*29+t*47)%257)-128)/128;
                if (scenario==1 && k%16==0) x = 32;
                if (scenario==2) x = k%16==0 ? 1.0F : std::ldexp(x,-30);
                if (scenario==3) x = k%32<16 ? 0 : std::ldexp(x,-120);
                inputs[t][k] = bf16(x);
            }
            flash_next_cpu_nvfp4_expert_pair_reference(expert, inputs[t], reference[t], scratch);
            flash_next_cpu_nvfp4_expert_pair_avx2(expert, inputs[t], actual[t], scratch);
            numerical(reference[t], actual[t]);
        }
        for (std::size_t width : {2U,3U,4U}) {
            std::array<const std::uint16_t*,4> in;
            std::array<float*,4> out;
            for (std::size_t t=0;t<width;++t) {in[t]=inputs[t].data();out[t]=grouped[t].data();}
            CpuNvfp4ExpertGroupScratch group;
            flash_next_cpu_nvfp4_expert_group_avx2(expert,{in.data(),width},{out.data(),width},group);
            for (std::size_t t=0;t<width;++t)
                if (actual[t]!=grouped[t]) throw std::runtime_error("grouping changed CPU output");
            HostExpertWorkerPool pool(4,true);
            std::vector<HostExpertTask> tasks;
            for (std::size_t t=0;t<width;++t)
                tasks.push_back({.expert=expert,.expert_id=0,.route_id=unsigned(t),
                                 .input=inputs[t].data(),.output=grouped[t].data()});
            pool.run(tasks,true,0);
            for (std::size_t t=0;t<width;++t)
                if (actual[t]!=grouped[t]) throw std::runtime_error("row sharding changed CPU output");
        }
    }
    // Rotate 32 canonical expert pairs (about 84 MiB) to exceed the target
    // host's LLC. This is an operator screen, not an end-to-end claim.
    std::vector<Matrix> gates, downs;
    for(int i=0;i<32;++i) { gates.emplace_back(1280,2560); downs.emplace_back(2560,640); }
    for (std::size_t width : {1U,2U,3U,4U}) {
        std::array<const std::uint16_t*,4> in;
        std::array<float*,4> out;
        CpuNvfp4ExpertGroupScratch group;
        for (std::size_t t=0;t<width;++t) {
            for(std::size_t k=0;k<2560;++k) inputs[t][k]=bf16(float(int((k*29+t*47)%257)-128)/128);
            in[t]=inputs[t].data();out[t]=actual[t].data();
        }
        auto run = [&](int id) {
            HostNvfp4ExpertPairView current{gates[id%32].view(),downs[id%32].view()};
            if (width==1) flash_next_cpu_nvfp4_expert_pair_avx2(current,inputs[0],actual[0],scratch);
            else flash_next_cpu_nvfp4_expert_group_avx2(current,{in.data(),width},{out.data(),width},group);
        };
        for(int i=0;i<32;++i) run(i);
        std::vector<double> samples;
        for(int i=0;i<32;++i) {
            const auto start=std::chrono::steady_clock::now(); run(i);
            samples.push_back(std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count());
        }
        std::sort(samples.begin(),samples.end());
        std::cout << "{\"kind\":\"cpu_expert_operator_screen\",\"exact_s16\":"
                  << (std::getenv("NINFER_V100_CPU_EXACT_S16") ? std::getenv("NINFER_V100_CPU_EXACT_S16") : "0")
                  << ",\"width\":" << width << ",\"median_us\":" << samples[16] << "}\n";
    }
}
