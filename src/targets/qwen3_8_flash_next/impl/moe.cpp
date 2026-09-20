#include "targets/qwen3_8_flash_next/impl/moe.h"

#include "core/layout.h"
#include "targets/qwen3_8_flash_next/impl/cpu_expert_reference.h"
#include "targets/qwen3_8_flash_next/impl/moe_kernels.h"
#include "targets/qwen3_8_flash_next/impl/moe_route.h"
#include "targets/qwen3_8_flash_next/impl/moe_workspace.h"
#include "targets/qwen3_8_flash_next/impl/stage_ledger.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <span>
#include <stdexcept>
#include <vector>

#include "core/device.h"

namespace ninfer::targets::qwen3_8_flash_next::detail {
namespace {

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

bool exact_bf16_weight(const Weight& weight, std::int32_t rows, std::int32_t columns) {
    return weight.qtype == QType::BF16_CTRL && weight.layout == QuantLayout::Contiguous &&
           weight.n == rows && weight.k == columns && weight.ndim == 2 && weight.shape[0] == rows &&
           weight.shape[1] == columns && weight.padded_shape[0] == rows &&
           weight.padded_shape[1] == columns && weight.qdata == weight.payload &&
           weight.payload_bytes >= static_cast<std::uint64_t>(rows) * columns * 2 &&
           aligned_to(weight.qdata, 16);
}

bool exact_expert_bank(const Nvfp4ExpertBankView& bank, std::int32_t rows, std::int32_t columns) {
    const std::uint64_t elements = static_cast<std::uint64_t>(rows) * columns;
    return bank.experts == 512 && bank.rows == rows && bank.columns == columns &&
           bank.code_bytes_per_expert == elements / 2 &&
           bank.scale_bytes_per_expert == elements / 16 && aligned_to(bank.codes, 16) &&
           aligned_to(bank.scales, 16) && aligned_to(bank.weight_scale_divisors, 16);
}

bool exact_bf16_expert_bank(const Bf16ExpertBankView& bank, std::int32_t rows, std::int32_t columns) {
    const std::uint64_t elements = static_cast<std::uint64_t>(rows) * columns;
    return bank.experts == 512 && bank.rows == rows && bank.columns == columns &&
           bank.bytes_per_expert == elements * sizeof(std::uint16_t) &&
           aligned_to(bank.data, 16);
}

} // namespace

std::size_t flash_next_moe_workspace_capacity_bytes(std::int32_t min_tokens,
                                                    std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("Flash-Next MoE workspace requires positive tokens");
    }
    WorkspaceLayoutBuilder layout;
    (void)allocate_flash_next_moe_workspace(layout, max_tokens);
    return layout.peak_bytes(256);
}

void flash_next_moe(const Tensor& input, const MoeWeights& weights, Tensor& output,
                    WorkspaceArena& workspace, cudaStream_t stream) {
    const std::int32_t tokens = input.ne[1];
    if (input.dtype != DType::BF16 || output.dtype != DType::BF16 || input.ne[0] != 2'560 ||
        output.ne[0] != 2'560 || tokens < 1 || output.ne[1] != tokens ||
        input.ne[2] != 1 || input.ne[3] != 1 || output.ne[2] != 1 || output.ne[3] != 1 ||
        !input.is_contiguous() || !output.is_contiguous() || !aligned_to(input.data, 16) ||
        !aligned_to(output.data, 16) || !exact_bf16_weight(weights.router, 512, 2'560) ||
        !exact_bf16_weight(weights.shared_down, 2'560, 640) ||
        !exact_bf16_weight(weights.shared_gate, 640, 2'560) ||
        !exact_bf16_weight(weights.shared_up, 640, 2'560) ||
        !exact_bf16_weight(weights.shared_gate_weight, 1, 2'560) ||
        !exact_expert_bank(weights.expert_gate_up, 1'280, 2'560) ||
        !exact_expert_bank(weights.expert_down, 2'560, 640) || stream == nullptr) {
        throw std::invalid_argument("Flash-Next MoE received an invalid exact target view");
    }
    const auto scope              = workspace.scope();
    FlashNextMoeWorkspace scratch = allocate_flash_next_moe_workspace(workspace, tokens);
    flash_next_route(input, weights.router, weights.shared_gate_weight, scratch.scores, scratch.ids,
                     scratch.alpha, scratch.shared_scale, stream);
    stage_ledger_record(stream, FlashNextStageId::MoE_Router);
    flash_next_moe_kernels_launch(input, weights, scratch, output, stream);

    // Diagnostic NINFER_FLASH_NEXT_TRACE_ROUTING only; not on the default prefill chunk path.
    static const char* trace_routing_env = std::getenv("NINFER_FLASH_NEXT_TRACE_ROUTING");
    if (trace_routing_env != nullptr && trace_routing_env[0] != '\0' &&
        tokens >= kFlashNextMoeMmaPrefillThreshold) {
        static int s_route_call = 0;
        int h_active = 0;
        cudaMemcpyAsync(&h_active, scratch.active_count.data, sizeof(int), cudaMemcpyDeviceToHost, stream);
        std::vector<int> h_counts(512);
        cudaMemcpyAsync(h_counts.data(), scratch.expert_counts.data, 512 * sizeof(int), cudaMemcpyDeviceToHost, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        int max_grp = 0, min_grp = 999999, sum_grp = 0;
        for (int e = 0; e < 512; ++e) {
            if (h_counts[e] > 0) {
                max_grp = std::max(max_grp, h_counts[e]);
                min_grp = std::min(min_grp, h_counts[e]);
                sum_grp += h_counts[e];
            }
        }
        int layer_idx = (s_route_call++) % 48;
        if (layer_idx == 0) {
            std::fprintf(stderr, "\n--- MoE Routing Trace (T=%d) ---\n", tokens);
            std::fprintf(stderr, "Layer | Active Experts | %% Active | Min Group | Avg Group | Max Group\n");
            std::fprintf(stderr, "------+----------------+----------+-----------+-----------+----------\n");
        }
        std::fprintf(stderr, " L%02d  |   %3d / 512    |  %5.1f%%  |    %3d    |   %5.1f   |    %3d   \n",
                     layer_idx, h_active, h_active * 100.0f / 512.0f, min_grp, (float)sum_grp / h_active, max_grp);
        if (layer_idx == 47) {
            std::fprintf(stderr, "-----------------------------------------------------------------\n\n");
        }
    }
}

void flash_next_moe_host_backed(const Tensor& input, const MoeWeights& resident_weights,
                                const HostNvfp4ExpertLayerView& host_experts, Tensor& output,
                                WorkspaceArena& workspace, cudaStream_t stream) {
    const std::int32_t tokens = input.ne[1];
    if (input.dtype != DType::BF16 || output.dtype != DType::BF16 || input.ne[0] != 2'560 ||
        output.ne[0] != 2'560 || tokens < 1 || output.ne[1] != tokens ||
        input.ne[2] != 1 || input.ne[3] != 1 || output.ne[2] != 1 || output.ne[3] != 1 ||
        !input.is_contiguous() || !output.is_contiguous() || !aligned_to(input.data, 16) ||
        !aligned_to(output.data, 16) ||
        !exact_bf16_weight(resident_weights.router, 512, 2'560) ||
        !exact_bf16_weight(resident_weights.shared_down, 2'560, 640) ||
        !exact_bf16_weight(resident_weights.shared_gate, 640, 2'560) ||
        !exact_bf16_weight(resident_weights.shared_up, 640, 2'560) ||
        !exact_bf16_weight(resident_weights.shared_gate_weight, 1, 2'560) ||
        !exact_expert_bank(host_experts.gate_up, 1'280, 2'560) ||
        !exact_expert_bank(host_experts.down, 2'560, 640) || stream == nullptr) {
        throw std::invalid_argument("Flash-Next host-backed MoE received an invalid exact target view");
    }

    const auto scope = workspace.scope();
    FlashNextMoeWorkspace scratch = allocate_flash_next_moe_workspace(workspace, tokens);

    flash_next_route(input, resident_weights.router, resident_weights.shared_gate_weight,
                     scratch.scores, scratch.ids, scratch.alpha, scratch.shared_scale, stream);
    stage_ledger_record(stream, FlashNextStageId::MoE_Router);

    // The shared expert remains resident on device. Compute its BF16 activation before the
    // host rendezvous. The shared down projection is deferred until the routed FP32 sum returns
    // so both branches can be combined before the final BF16 rounding.
    flash_next_moe_host_shared_launch(input, resident_weights, scratch, stream);

    const std::size_t input_words =
        static_cast<std::size_t>(tokens) * kFlashNextExpertHidden;
    const std::size_t routed_paths = static_cast<std::size_t>(tokens) * 10ULL;
    std::vector<std::uint16_t> host_input(input_words);
    std::vector<std::int32_t> host_ids(routed_paths);
    std::vector<float> host_alpha(routed_paths);

    CUDA_CHECK(cudaMemcpyAsync(host_input.data(), input.data,
                               input_words * sizeof(std::uint16_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(host_ids.data(), scratch.ids.data,
                               routed_paths * sizeof(std::int32_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(host_alpha.data(), scratch.alpha.data,
                               routed_paths * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Correctness-first Phase-10 CPU execution. Each expert pair sees the exact compact
    // mapped bytes and rounds SiLU(gate)*up through BF16 before the down projection.
    // Routing alpha is accumulated in the same path order as the GPU reference.
    std::vector<float> routed_sum(
        static_cast<std::size_t>(tokens) * kFlashNextExpertHidden, 0.0F);
    std::vector<float> pair_output(kFlashNextExpertHidden);
    CpuNvfp4ExpertReferenceScratch cpu_scratch{};

    for (std::int32_t token = 0; token < tokens; ++token) {
        const auto input_span = std::span<const std::uint16_t>(
            host_input.data() + static_cast<std::size_t>(token) * kFlashNextExpertHidden,
            kFlashNextExpertHidden);
        float* token_sum =
            routed_sum.data() + static_cast<std::size_t>(token) * kFlashNextExpertHidden;

        for (std::int32_t path = 0; path < 10; ++path) {
            const std::size_t route_index =
                static_cast<std::size_t>(token) * 10ULL + static_cast<std::size_t>(path);
            const HostNvfp4ExpertPairView expert =
                host_experts.expert(host_ids[route_index]);
            flash_next_cpu_nvfp4_expert_pair_reference(
                expert, input_span, std::span<float>(pair_output), cpu_scratch);

            const float alpha = host_alpha[route_index];
            for (std::size_t row = 0; row < kFlashNextExpertHidden; ++row) {
                token_sum[row] = std::fma(alpha, pair_output[row], token_sum[row]);
            }
        }
    }

    // Each token owns 640 * 11 BF16 values (14,080 B). Store the 2,560 FP32 routed
    // values in the first 10,240 B of that token's slab and preserve the shared path at
    // BF16 path 10 (offset 12,800 B). A pitched copy keeps token slabs independent.
    constexpr std::size_t kActivationPitchBytes =
        kFlashNextExpertIntermediate * 11ULL * sizeof(std::uint16_t);
    constexpr std::size_t kRoutedBytesPerToken =
        kFlashNextExpertHidden * sizeof(float);
    static_assert(kActivationPitchBytes >= kRoutedBytesPerToken);
    CUDA_CHECK(cudaMemcpy2DAsync(
        scratch.activations.data, kActivationPitchBytes,
        routed_sum.data(), kRoutedBytesPerToken,
        kRoutedBytesPerToken, static_cast<std::size_t>(tokens),
        cudaMemcpyHostToDevice, stream));
    flash_next_moe_host_routed_merge_launch(
        resident_weights, scratch, output, tokens, stream);
}

void flash_next_moe_bf16(const Tensor& input, const MoeBf16Weights& weights, Tensor& output,
                         WorkspaceArena& workspace, cudaStream_t stream) {
    const std::int32_t tokens = input.ne[1];
    if (input.dtype != DType::BF16 || output.dtype != DType::BF16 || input.ne[0] != 2'560 ||
        output.ne[0] != 2'560 || tokens < 1 || output.ne[1] != tokens ||
        input.ne[2] != 1 || input.ne[3] != 1 || output.ne[2] != 1 || output.ne[3] != 1 ||
        !input.is_contiguous() || !output.is_contiguous() || !aligned_to(input.data, 16) ||
        !aligned_to(output.data, 16) || !exact_bf16_weight(weights.router, 512, 2'560) ||
        !exact_bf16_weight(weights.shared_down, 2'560, 640) ||
        !exact_bf16_weight(weights.shared_gate, 640, 2'560) ||
        !exact_bf16_weight(weights.shared_up, 640, 2'560) ||
        !exact_bf16_weight(weights.shared_gate_weight, 1, 2'560) ||
        !exact_bf16_expert_bank(weights.expert_gate_up, 1'280, 2'560) ||
        !exact_bf16_expert_bank(weights.expert_down, 2'560, 640) || stream == nullptr) {
        throw std::invalid_argument("Flash-Next BF16 MoE received an invalid exact target view");
    }
    const auto scope              = workspace.scope();
    FlashNextMoeWorkspace scratch = allocate_flash_next_moe_workspace(workspace, tokens);
    flash_next_route(input, weights.router, weights.shared_gate_weight, scratch.scores, scratch.ids,
                     scratch.alpha, scratch.shared_scale, stream);
    flash_next_moe_bf16_kernels_launch(input, weights, scratch, output, stream);
}

} // namespace ninfer::targets::qwen3_8_flash_next::detail
