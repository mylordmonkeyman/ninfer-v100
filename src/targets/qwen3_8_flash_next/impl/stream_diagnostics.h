#pragma once

#include "targets/qwen3_8_flash_next/impl/expert_stream.h"
#include "targets/qwen3_8_flash_next/impl/cpu_expert_reference.h"
#include "core/device.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string_view>

namespace ninfer::targets::qwen3_8_flash_next::detail {
// Explicit debugging only. Uses represented production inputs and never replaces
// a route output. Timings collected with this enabled are not performance evidence.
inline bool stream_diagnostics_enabled() {
    const char* value = std::getenv("NINFER_V100_STREAM_COMPARE");
    if (!value || !*value || std::string_view(value) == "0") return false;
    if (std::string_view(value) == "1") return true;
    throw std::invalid_argument("NINFER_V100_STREAM_COMPARE must be 0 or 1");
}

inline void compare_streamed_routes(int layer, int tokens,
    const HostNvfp4ExpertLayerView& experts,
    const std::array<std::vector<FlashNextStreamRoute>, 512>& routes,
    cudaStream_t compute) {
    std::vector<unsigned> active;
    for (unsigned id=0; id<routes.size(); ++id)
        if (!routes[id].empty()) active.push_back(id);
    if (active.empty()) return;
    // Evenly spaced expert IDs and route positions give eight bounded samples per
    // layer, including first/last active experts and early/late prompt inputs.
    const unsigned samples=std::min<std::size_t>(8,active.size());
    FlashNextExpertStream isolated(1);
    DeviceBuffer replay_output(kFlashNextExpertHidden*sizeof(float));
    std::array<std::uint16_t,kFlashNextExpertHidden> input{};
    std::array<float,kFlashNextExpertHidden> original{}, replay{}, reference{}, avx2{};
    CpuNvfp4ExpertReferenceScratch reference_scratch, avx2_scratch;
    for (unsigned sample=0; sample<samples; ++sample) {
        const unsigned id=active[samples==1 ? 0 : sample*(active.size()-1)/(samples-1)];
        const auto index=samples==1 ? 0 : sample*(routes[id].size()-1)/(samples-1);
        const auto& route=routes[id][index];
        CUDA_CHECK(cudaMemcpyAsync(input.data(),route.input_bf16,sizeof(input),
                                   cudaMemcpyDeviceToHost,compute));
        CUDA_CHECK(cudaMemcpyAsync(original.data(),route.output_fp32,sizeof(original),
                                   cudaMemcpyDeviceToHost,compute));
        CUDA_CHECK(cudaStreamSynchronize(compute));
        FlashNextStreamRoute isolated_route{route.input_bf16,
            static_cast<float*>(replay_output.p)};
        isolated.submit(experts.expert(id),std::span(&isolated_route,1),compute);
        isolated.finish();
        CUDA_CHECK(cudaMemcpy(replay.data(),replay_output.p,sizeof(replay),cudaMemcpyDeviceToHost));
        flash_next_cpu_nvfp4_expert_pair_reference(experts.expert(id),input,reference,reference_scratch);
        const bool has_avx2=flash_next_cpu_nvfp4_avx2_available();
        if (has_avx2)
            flash_next_cpu_nvfp4_expert_pair_avx2(experts.expert(id),input,avx2,avx2_scratch);
        double error=0,norm=0,actual_norm=0,dot=0,max_error=0,cpu_error=0,gpu_avx2_error=0,avx2_norm=0;
        unsigned nonfinite=0,replay_differences=0;
        for (unsigned row=0; row<original.size(); ++row) {
            const double x=original[row], y=reference[row];
            nonfinite+=!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(replay[row]);
            replay_differences+=std::bit_cast<std::uint32_t>(original[row])!=
                                std::bit_cast<std::uint32_t>(replay[row]);
            error+=(x-y)*(x-y); norm+=y*y; actual_norm+=x*x; dot+=x*y;
            max_error=std::max(max_error,std::abs(x-y));
            if (has_avx2) {
                cpu_error+=(double(avx2[row])-y)*(double(avx2[row])-y);
                gpu_avx2_error+=(x-double(avx2[row]))*(x-double(avx2[row]));
                avx2_norm+=double(avx2[row])*avx2[row];
            }
        }
        // This is CPU precision-profile evidence, supplementary to the existing
        // independent operator and Phase11 mathematical-oracle qualification.
        std::fprintf(stderr,
            "v100.stream_compare={\"layer\":%d,\"tokens\":%d,\"sample\":%u,"
            "\"expert\":%u,\"expert_routes\":%zu,\"route_in_expert\":%zu,"
            "\"nonfinite\":%u,\"isolated_replay_differences\":%u,"
            "\"cpu_reference_nrmse\":%.12g,\"cpu_reference_cosine\":%.12g,"
            "\"maximum_error\":%.12g,\"avx2_available\":%s,\"avx2_reference_nrmse\":%.12g,\"gpu_avx2_nrmse\":%.12g}\n",
            layer,tokens,sample,id,routes[id].size(),index,nonfinite,replay_differences,
            std::sqrt(error/std::max(norm,1e-30)),
            dot/std::sqrt(std::max(actual_norm*norm,1e-60)),max_error,
            has_avx2 ? "true" : "false",std::sqrt(cpu_error/std::max(norm,1e-30)),
            std::sqrt(gpu_avx2_error/std::max(avx2_norm,1e-30)));
        std::fflush(stderr);
    }
}
} // namespace ninfer::targets::qwen3_8_flash_next::detail
