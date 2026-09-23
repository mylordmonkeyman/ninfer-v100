#include "core/arena.h"
#include "core/device.h"
#include "targets/qwen3_8_flash_next/impl/hyper_connection.h"
#include "targets/qwen3_8_flash_next/impl/hyper_connection_kernels.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

bool cuda_unavailable(cudaError_t error) {
    return error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver;
}

// Timing numbers are always reported. The T=1 25 us CUDA-graph gate fails only when
// NINFER_PERF_GATES=1; an idle-GPU correctness run must not false-fail on a busy device.
bool ninfer_perf_gates_armed() {
    const char* env = std::getenv("NINFER_PERF_GATES");
    return env != nullptr && std::strcmp(env, "1") == 0;
}

float bf16_to_float(std::uint16_t value) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(value) << 16U);
}

std::uint16_t float_to_bf16(float value) {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    const std::uint32_t lsb  = (bits >> 16U) & 1U;
    const std::uint32_t bias = 0x7FFFU + lsb;
    return static_cast<std::uint16_t>((bits + bias) >> 16U);
}

ninfer::Weight bf16_weight(void* data, std::int32_t rows, std::int32_t columns) {
    ninfer::Weight out{};
    out.payload         = data;
    out.payload_bytes   = static_cast<std::uint64_t>(rows) * columns * 2;
    out.qdata           = data;
    out.qtype           = ninfer::QType::BF16_CTRL;
    out.layout          = ninfer::QuantLayout::Contiguous;
    out.n               = rows;
    out.k               = columns;
    out.ndim            = 2;
    out.shape[0]        = rows;
    out.shape[1]        = columns;
    out.padded_shape[0] = rows;
    out.padded_shape[1] = columns;
    return out;
}

struct RelL2Stats {
    double rel         = 0.0;
    double base_sq     = 0.0;
    double act_sq      = 0.0;
    double dot         = 0.0;
    double max_abs_act = 0.0;
    double max_abs_ref = 0.0;
    int n              = 0;
    int nfinite        = 0;
    int nnan           = 0;
    int nzero          = 0;
};

RelL2Stats rel_l2_stats(const float* act, const float* ref, std::size_t n) {
    RelL2Stats s;
    s.n = static_cast<int>(n);
    double diff_sq = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double a = static_cast<double>(act[i]);
        const double e = static_cast<double>(ref[i]);
        if (!std::isfinite(a) || !std::isfinite(e)) {
            ++s.nnan;
            continue;
        }
        ++s.nfinite;
        if (a == 0.0) {
            ++s.nzero;
        }
        s.act_sq += a * a;
        s.base_sq += e * e;
        s.dot += a * e;
        const double d = a - e;
        diff_sq += d * d;
        s.max_abs_act = std::max(s.max_abs_act, std::abs(a));
        s.max_abs_ref = std::max(s.max_abs_ref, std::abs(e));
    }
    if (s.base_sq > 0.0) {
        s.rel = std::sqrt(diff_sq / s.base_sq);
    } else if (diff_sq > 0.0) {
        s.rel = std::sqrt(diff_sq);
    }
    return s;
}

void print_rel_l2_stats(const char* label, const RelL2Stats& s) {
    const double corr =
        (s.act_sq > 0.0 && s.base_sq > 0.0) ? (s.dot / std::sqrt(s.act_sq * s.base_sq)) : 0.0;
    std::cout << "    " << label << ": rel-L2=" << s.rel << " base_sq=" << s.base_sq
              << " act_sq=" << s.act_sq << " corr=" << corr << " nfinite=" << s.nfinite << "/" << s.n
              << " nnan=" << s.nnan << " nzero=" << s.nzero << " max_abs_act=" << s.max_abs_act
              << " max_abs_ref=" << s.max_abs_ref << "\n";
}

bool accept_stage(const char* stage, int tokens, const RelL2Stats& s, double* max_rel,
                  double tolerance) {
    if (s.nnan > 0 || s.nfinite != s.n || s.base_sq <= 0.0 || !std::isfinite(s.rel) ||
        !std::isfinite(s.base_sq)) {
        std::cerr << "VACUOUS/NaN: T=" << tokens << " " << stage << " nnan=" << s.nnan
                  << " nfinite=" << s.nfinite << "/" << s.n << " base_sq=" << s.base_sq
                  << " rel=" << s.rel << "\n";
        return false;
    }
    *max_rel = std::max(*max_rel, s.rel);
    if (s.rel > tolerance) {
        std::cerr << "FAIL: T=" << tokens << " " << stage << " rel-L2=" << s.rel << " exceeds "
                  << tolerance << " (base_sq=" << s.base_sq << ")\n";
        return false;
    }
    return true;
}

struct HyperReferenceOutput {
    std::vector<float> normalized;
    std::vector<float> low_rank;
    std::vector<float> injection;
    std::vector<float> block_input;
    std::vector<float> hidden_injected;
};

HyperReferenceOutput evaluate_reference(
    int tokens,
    const std::vector<float>& host_hidden,
    const std::vector<float>& host_norm,
    const std::vector<float>& host_down,
    const std::vector<float>& host_up,
    const std::vector<float>& host_inject,
    const std::vector<float>& host_block_output) {
    HyperReferenceOutput out;
    out.normalized.resize(static_cast<std::size_t>(tokens) * 10'240);
    out.low_rank.resize(static_cast<std::size_t>(tokens) * 320);
    out.injection.resize(static_cast<std::size_t>(tokens) * 4);
    out.block_input.resize(static_cast<std::size_t>(tokens) * 2'560);
    out.hidden_injected.resize(static_cast<std::size_t>(tokens) * 10'240);

    for (int t = 0; t < tokens; ++t) {
        // 1. Group Norm
        for (int s = 0; s < 4; ++s) {
            float sum_sq = 0.0F;
            for (int c = 0; c < 2'560; ++c) {
                float v = host_hidden[static_cast<std::size_t>(t) * 10'240 + s * 2'560 + c];
                sum_sq += v * v;
            }
            float inv_rms = 1.0F / std::sqrt(sum_sq / 2'560.0F + 1.0e-6F);
            for (int c = 0; c < 2'560; ++c) {
                int flat = s * 2'560 + c;
                float v  = host_hidden[static_cast<std::size_t>(t) * 10'240 + flat];
                float n  = host_norm[flat];
                float normed = v * inv_rms * (1.0F + n);
                out.normalized[static_cast<std::size_t>(t) * 10'240 + flat] =
                    bf16_to_float(float_to_bf16(normed));
            }
        }

        // 2. Low Rank Down Projection
        for (int r = 0; r < 320; ++r) {
            float sum = 0.0F;
            for (int c = 0; c < 10'240; ++c) {
                sum += host_down[static_cast<std::size_t>(r) * 10'240 + c] *
                       out.normalized[static_cast<std::size_t>(t) * 10'240 + c];
            }
            float act = (sum * 0.25F) / (1.0F + std::exp(-(sum * 0.25F)));
            out.low_rank[static_cast<std::size_t>(t) * 320 + r] =
                bf16_to_float(float_to_bf16(act));
        }

        // 3. Injection Projection
        for (int s = 0; s < 4; ++s) {
            float sum = 0.0F;
            for (int c = 0; c < 10'240; ++c) {
                sum += host_inject[static_cast<std::size_t>(s) * 10'240 + c] *
                       out.normalized[static_cast<std::size_t>(t) * 10'240 + c];
            }
            float act = 2.0F / (1.0F + std::exp(-(sum * 0.25F)));
            out.injection[static_cast<std::size_t>(t) * 4 + s] = act;
        }

        // 4. Mix Up Projection
        for (int h = 0; h < 2'560; ++h) {
            float mean_contrib = 0.0F;
            for (int s = 0; s < 4; ++s) {
                int row   = s * 2'560 + h;
                float sum = 0.0F;
                for (int c = 0; c < 320; ++c) {
                    sum += host_up[static_cast<std::size_t>(row) * 320 + c] *
                           out.low_rank[static_cast<std::size_t>(t) * 320 + c];
                }
                float mix_gate = 1.0F / (1.0F + std::exp(-sum));
                mean_contrib  += mix_gate * out.normalized[static_cast<std::size_t>(t) * 10'240 + row];
            }
            out.block_input[static_cast<std::size_t>(t) * 2'560 + h] =
                bf16_to_float(float_to_bf16(mean_contrib * 0.25F));
        }

        // 5. Inject
        for (int s = 0; s < 4; ++s) {
            float inj = out.injection[static_cast<std::size_t>(t) * 4 + s];
            for (int h = 0; h < 2'560; ++h) {
                int flat   = s * 2'560 + h;
                float orig = host_hidden[static_cast<std::size_t>(t) * 10'240 + flat];
                float upd  = host_block_output[static_cast<std::size_t>(t) * 2'560 + h] * inj;
                out.hidden_injected[static_cast<std::size_t>(t) * 10'240 + flat] =
                    bf16_to_float(float_to_bf16(orig + upd));
            }
        }
    }
    return out;
}

int test_basic_unit_injection(ninfer::DeviceContext& device) {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    ninfer::DeviceBuffer hidden(10'240 * 2);
    ninfer::DeviceBuffer norm(10'240 * 2);
    ninfer::DeviceBuffer down(320ULL * 10'240 * 2);
    ninfer::DeviceBuffer up(10'240ULL * 320 * 2);
    ninfer::DeviceBuffer inject(4ULL * 10'240 * 2);
    ninfer::DeviceBuffer block_input(2'560 * 2);
    ninfer::DeviceBuffer block_output(2'560 * 2);
    std::array<std::uint16_t, 10'240> hidden_values{};
    hidden_values.fill(0x3F80U); // 1.0
    hidden.copy_from_host(hidden_values.data(), sizeof(hidden_values));
    norm.fill(0);
    down.fill(0);
    up.fill(0);
    inject.fill(0);
    std::array<std::uint16_t, 2'560> output_values{};
    output_values.fill(0x4000U); // 2.0
    block_output.copy_from_host(output_values.data(), sizeof(output_values));

    HyperConnectionWeights weights{
        .block_inject   = bf16_weight(inject.p, 4, 10'240),
        .norm           = ninfer::Tensor(norm.p, ninfer::DType::BF16, {10'240}),
        .input_mix_down = bf16_weight(down.p, 320, 10'240),
        .input_mix_up   = bf16_weight(up.p, 10'240, 320),
    };
    ninfer::WorkspaceArena workspace(flash_next_hyper_workspace_capacity_bytes(1, 1));
    auto scope                      = workspace.scope();
    FlashNextHyperWorkspace scratch = allocate_flash_next_hyper_workspace(workspace, 1);
    ninfer::Tensor hidden_view(hidden.p, ninfer::DType::BF16, {10'240, 1});
    ninfer::Tensor input_view(block_input.p, ninfer::DType::BF16, {2'560, 1});
    ninfer::Tensor output_view(block_output.p, ninfer::DType::BF16, {2'560, 1});
    flash_next_hyper_prepare(hidden_view, weights, scratch, input_view, device.stream);
    flash_next_hyper_inject(output_view, scratch.injection, hidden_view, device.stream);
    device.synchronize();

    std::array<std::uint16_t, 2'560> actual_input{};
    block_input.copy_to_host(actual_input.data(), sizeof(actual_input));
    if (!std::all_of(actual_input.begin(), actual_input.end(),
                     [](std::uint16_t value) { return value == 0x3F00U; })) {
        std::cerr << "Flash-Next zero-mix hyper input was not exact BF16 0.5\n";
        return 1;
    }
    hidden.copy_to_host(hidden_values.data(), sizeof(hidden_values));
    if (!std::all_of(hidden_values.begin(), hidden_values.end(),
                     [](std::uint16_t value) { return value == 0x4040U; })) {
        std::cerr << "Flash-Next unit injection did not produce exact BF16 3.0\n";
        return 1;
    }

    hidden_values.fill(0x3F80U);
    hidden.copy_from_host(hidden_values.data(), sizeof(hidden_values));
    const HyperMixerWeights mixer{
        .norm           = weights.norm,
        .input_mix_down = weights.input_mix_down,
        .input_mix_up   = weights.input_mix_up,
    };
    flash_next_hyper_mix(hidden_view, mixer, scratch, input_view, device.stream);
    device.synchronize();
    block_input.copy_to_host(actual_input.data(), sizeof(actual_input));
    if (!std::all_of(actual_input.begin(), actual_input.end(),
                     [](std::uint16_t value) { return value == 0x3F00U; })) {
        std::cerr << "Flash-Next final hyper mixer was not exact BF16 0.5\n";
        return 1;
    }
    return 0;
}

int test_synthetic_stage_equivalence(ninfer::DeviceContext& device) {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist_weight(-0.05F, 0.05F);
    std::uniform_real_distribution<float> dist_norm(-0.02F, 0.02F);
    std::uniform_real_distribution<float> dist_act(-1.5F, 1.5F);

    // Weights buffers
    std::vector<float> h_norm_f(10'240);
    std::vector<std::uint16_t> h_norm_bf(10'240);
    for (std::size_t i = 0; i < 10'240; ++i) {
        h_norm_f[i]  = dist_norm(rng);
        h_norm_bf[i] = float_to_bf16(h_norm_f[i]);
        h_norm_f[i]  = bf16_to_float(h_norm_bf[i]);
    }

    std::vector<float> h_down_f(320ULL * 10'240);
    std::vector<std::uint16_t> h_down_bf(320ULL * 10'240);
    for (std::size_t i = 0; i < h_down_f.size(); ++i) {
        h_down_f[i]  = dist_weight(rng);
        h_down_bf[i] = float_to_bf16(h_down_f[i]);
        h_down_f[i]  = bf16_to_float(h_down_bf[i]);
    }

    std::vector<float> h_up_f(10'240ULL * 320);
    std::vector<std::uint16_t> h_up_bf(10'240ULL * 320);
    for (std::size_t i = 0; i < h_up_f.size(); ++i) {
        h_up_f[i]  = dist_weight(rng);
        h_up_bf[i] = float_to_bf16(h_up_f[i]);
        h_up_f[i]  = bf16_to_float(h_up_bf[i]);
    }

    std::vector<float> h_inject_f(4ULL * 10'240);
    std::vector<std::uint16_t> h_inject_bf(4ULL * 10'240);
    for (std::size_t i = 0; i < h_inject_f.size(); ++i) {
        h_inject_f[i]  = dist_weight(rng);
        h_inject_bf[i] = float_to_bf16(h_inject_f[i]);
        h_inject_f[i]  = bf16_to_float(h_inject_bf[i]);
    }

    ninfer::DeviceBuffer d_norm(h_norm_bf.size() * 2);
    d_norm.copy_from_host(h_norm_bf.data(), h_norm_bf.size() * 2);

    ninfer::DeviceBuffer d_down(h_down_bf.size() * 2);
    d_down.copy_from_host(h_down_bf.data(), h_down_bf.size() * 2);

    ninfer::DeviceBuffer d_up(h_up_bf.size() * 2);
    d_up.copy_from_host(h_up_bf.data(), h_up_bf.size() * 2);

    ninfer::DeviceBuffer d_inject(h_inject_bf.size() * 2);
    d_inject.copy_from_host(h_inject_bf.data(), h_inject_bf.size() * 2);

    HyperConnectionWeights weights{
        .block_inject   = bf16_weight(d_inject.p, 4, 10'240),
        .norm           = ninfer::Tensor(d_norm.p, ninfer::DType::BF16, {10'240}),
        .input_mix_down = bf16_weight(d_down.p, 320, 10'240),
        .input_mix_up   = bf16_weight(d_up.p, 10'240, 320),
    };

    double max_rel_l2_norm   = 0.0;
    double max_rel_l2_lr     = 0.0;
    double max_rel_l2_inj    = 0.0;
    double max_rel_l2_input  = 0.0;
    double max_rel_l2_injected = 0.0;

    // Decode T=1..8 (fused). Prefill: T=16 one partial 32-col down tile + partial 64-col
    // up tile; T=48 one full down tile + one partial; T=119 production chunk tail
    // (three full 32-col tiles + 23-token partial); T=128 exact down and up tiles.
    std::vector<int> test_tokens = {1, 2, 3, 4, 5, 6, 7, 8, 16, 48, 119, 128};
    constexpr double kMaxRelL2Tolerance = 1.0e-3;

    std::cout << "\n--- Flash-Next Hyper-Connection per-T stage rel-L2 ---\n";
    std::cout << std::scientific << std::setprecision(6);

    for (int tokens : test_tokens) {
        std::vector<float> h_hidden_f(static_cast<std::size_t>(tokens) * 10'240);
        std::vector<std::uint16_t> h_hidden_bf(h_hidden_f.size());
        for (std::size_t i = 0; i < h_hidden_f.size(); ++i) {
            h_hidden_f[i]  = dist_act(rng);
            h_hidden_bf[i] = float_to_bf16(h_hidden_f[i]);
            h_hidden_f[i]  = bf16_to_float(h_hidden_bf[i]);
        }

        std::vector<float> h_output_f(static_cast<std::size_t>(tokens) * 2'560);
        std::vector<std::uint16_t> h_output_bf(h_output_f.size());
        for (std::size_t i = 0; i < h_output_f.size(); ++i) {
            h_output_f[i]  = dist_act(rng);
            h_output_bf[i] = float_to_bf16(h_output_f[i]);
            h_output_f[i]  = bf16_to_float(h_output_bf[i]);
        }

        // Run host reference
        auto ref = evaluate_reference(tokens, h_hidden_f, h_norm_f, h_down_f, h_up_f, h_inject_f, h_output_f);

        // Run GPU fused execution
        ninfer::DeviceBuffer d_hidden(h_hidden_bf.size() * 2);
        d_hidden.copy_from_host(h_hidden_bf.data(), h_hidden_bf.size() * 2);

        ninfer::DeviceBuffer d_block_input(static_cast<std::size_t>(tokens) * 2'560 * 2);
        ninfer::DeviceBuffer d_block_output(h_output_bf.size() * 2);
        d_block_output.copy_from_host(h_output_bf.data(), h_output_bf.size() * 2);

        ninfer::WorkspaceArena workspace(flash_next_hyper_workspace_capacity_bytes(1, tokens));
        auto scope                      = workspace.scope();
        FlashNextHyperWorkspace scratch = allocate_flash_next_hyper_workspace(workspace, tokens);
        CUDA_CHECK(cudaMemsetAsync(scratch.low_rank.data, 0xFF,
                                   static_cast<std::size_t>(tokens) * 320U * 2U, device.stream));
        CUDA_CHECK(cudaMemsetAsync(scratch.up_gemm.data, 0xFF,
                                   static_cast<std::size_t>(tokens) * 10'240U * 2U, device.stream));
        CUDA_CHECK(cudaMemsetAsync(scratch.down_split.data, 0xFF,
                                   static_cast<std::size_t>(tokens) * 320U * 4U * sizeof(float),
                                   device.stream));
        CUDA_CHECK(cudaMemsetAsync(d_block_input.p, 0xFF,
                                   static_cast<std::size_t>(tokens) * 2'560U * 2U, device.stream));

        ninfer::Tensor hidden_view(d_hidden.p, ninfer::DType::BF16, {10'240, tokens});
        ninfer::Tensor input_view(d_block_input.p, ninfer::DType::BF16, {2'560, tokens});
        ninfer::Tensor output_view(d_block_output.p, ninfer::DType::BF16, {2'560, tokens});

        flash_next_hyper_prepare(hidden_view, weights, scratch, input_view, device.stream);
        flash_next_hyper_inject(output_view, scratch.injection, hidden_view, device.stream);
        device.synchronize();

        auto to_float = [](const std::vector<std::uint16_t>& bits) {
            std::vector<float> out(bits.size());
            for (std::size_t i = 0; i < bits.size(); ++i) {
                out[i] = bf16_to_float(bits[i]);
            }
            return out;
        };

        std::vector<std::uint16_t> act_norm_bf(static_cast<std::size_t>(tokens) * 10'240);
        CUDA_CHECK(cudaMemcpy(act_norm_bf.data(), scratch.normalized.data, act_norm_bf.size() * 2,
                              cudaMemcpyDeviceToHost));
        const std::vector<float> act_norm_f = to_float(act_norm_bf);
        const RelL2Stats err_norm =
            rel_l2_stats(act_norm_f.data(), ref.normalized.data(), act_norm_f.size());

        std::vector<std::uint16_t> act_lr_bf(static_cast<std::size_t>(tokens) * 320);
        CUDA_CHECK(cudaMemcpy(act_lr_bf.data(), scratch.low_rank.data, act_lr_bf.size() * 2,
                              cudaMemcpyDeviceToHost));
        const std::vector<float> act_lr_f = to_float(act_lr_bf);
        const RelL2Stats err_lr =
            rel_l2_stats(act_lr_f.data(), ref.low_rank.data(), act_lr_f.size());

        std::vector<float> act_inj_f(static_cast<std::size_t>(tokens) * 4);
        CUDA_CHECK(cudaMemcpy(act_inj_f.data(), scratch.injection.data,
                              act_inj_f.size() * sizeof(float), cudaMemcpyDeviceToHost));
        const RelL2Stats err_inj =
            rel_l2_stats(act_inj_f.data(), ref.injection.data(), act_inj_f.size());

        std::vector<std::uint16_t> act_in_bf(static_cast<std::size_t>(tokens) * 2'560);
        CUDA_CHECK(cudaMemcpy(act_in_bf.data(), input_view.data, act_in_bf.size() * 2,
                              cudaMemcpyDeviceToHost));
        std::vector<float> act_in_f = to_float(act_in_bf);
        const RelL2Stats err_in =
            rel_l2_stats(act_in_f.data(), ref.block_input.data(), act_in_f.size());

        std::vector<std::uint16_t> act_hid_bf(static_cast<std::size_t>(tokens) * 10'240);
        CUDA_CHECK(cudaMemcpy(act_hid_bf.data(), hidden_view.data, act_hid_bf.size() * 2,
                              cudaMemcpyDeviceToHost));
        const std::vector<float> act_hid_f = to_float(act_hid_bf);
        const RelL2Stats err_hid =
            rel_l2_stats(act_hid_f.data(), ref.hidden_injected.data(), act_hid_f.size());

        const HyperMixerWeights mixer{
            .norm           = weights.norm,
            .input_mix_down = weights.input_mix_down,
            .input_mix_up   = weights.input_mix_up,
        };
        flash_next_hyper_mix(hidden_view, mixer, scratch, input_view, device.stream);
        device.synchronize();
        CUDA_CHECK(cudaMemcpy(act_in_bf.data(), input_view.data, act_in_bf.size() * 2,
                              cudaMemcpyDeviceToHost));
        act_in_f = to_float(act_in_bf);
        auto ref_mix = evaluate_reference(tokens, act_hid_f, h_norm_f, h_down_f, h_up_f, h_inject_f,
                                          h_output_f);
        const RelL2Stats err_mix =
            rel_l2_stats(act_in_f.data(), ref_mix.block_input.data(), act_in_f.size());

        std::cout << "  T=" << tokens << " down%32=" << (tokens % 32) << " up%64=" << (tokens % 64)
                  << " norm=" << err_norm.rel << " down=" << err_lr.rel << " inj=" << err_inj.rel
                  << " input=" << err_in.rel << " hid=" << err_hid.rel << " mix=" << err_mix.rel
                  << " down_base_sq=" << err_lr.base_sq << "\n";

        if (!accept_stage("group_norm", tokens, err_norm, &max_rel_l2_norm, kMaxRelL2Tolerance) ||
            !accept_stage("low_rank", tokens, err_lr, &max_rel_l2_lr, kMaxRelL2Tolerance) ||
            !accept_stage("injection", tokens, err_inj, &max_rel_l2_inj, kMaxRelL2Tolerance) ||
            !accept_stage("block_input", tokens, err_in, &max_rel_l2_input, kMaxRelL2Tolerance) ||
            !accept_stage("hidden_injected", tokens, err_hid, &max_rel_l2_injected,
                          kMaxRelL2Tolerance) ||
            !accept_stage("mixer_block_input", tokens, err_mix, &max_rel_l2_input,
                          kMaxRelL2Tolerance)) {
            return 1;
        }

        if (tokens == 48) {
            const RelL2Stats full_tile =
                rel_l2_stats(act_lr_f.data(), ref.low_rank.data(), 32ULL * 320ULL);
            const RelL2Stats partial_tile =
                rel_l2_stats(act_lr_f.data() + 32ULL * 320ULL, ref.low_rank.data() + 32ULL * 320ULL,
                             16ULL * 320ULL);
            print_rel_l2_stats("T=48 low_rank tokens[0,32) full 32-tile", full_tile);
            print_rel_l2_stats("T=48 low_rank tokens[32,48) partial 16-tile", partial_tile);
            if (!accept_stage("low_rank_full_tile", tokens, full_tile, &max_rel_l2_lr,
                              kMaxRelL2Tolerance) ||
                !accept_stage("low_rank_partial_tile", tokens, partial_tile, &max_rel_l2_lr,
                              kMaxRelL2Tolerance)) {
                return 1;
            }
        }
    }

    std::cout << "\n--- Flash-Next Fused Hyper-Connection vs Reference (rel-L2 Maxima) ---\n";
    std::cout << std::scientific << std::setprecision(6);
    std::cout << "  Stage 1 (group_norm -> normalized) : " << max_rel_l2_norm << "\n";
    std::cout << "  Stage 2 (down_proj  -> low_rank)   : " << max_rel_l2_lr << "\n";
    std::cout << "  Stage 3 (inject_gate-> injection)  : " << max_rel_l2_inj << "\n";
    std::cout << "  Stage 4 (up_proj+mix-> block_input): " << max_rel_l2_input << "\n";
    std::cout << "  Stage 5 (inject     -> hidden)     : " << max_rel_l2_injected << "\n";
    std::cout << "----------------------------------------------------------------------\n";

    if (max_rel_l2_norm > kMaxRelL2Tolerance || max_rel_l2_lr > kMaxRelL2Tolerance ||
        max_rel_l2_inj > kMaxRelL2Tolerance || max_rel_l2_input > kMaxRelL2Tolerance ||
        max_rel_l2_injected > kMaxRelL2Tolerance) {
        std::cerr << "FAILED: Relative L2 error exceeded tolerance\n";
        return 1;
    }

    std::cout << "PASS: test_synthetic_stage_equivalence\n";
    return 0;
}

// ---------------------------------------------------------------------------
// Decode-route bit-exact gate.
//
// The fused hyper_norm_low_rank_fused_kernel replaces group_norm_vectorized_kernel +
// low_rank_and_injection_kernel at T <= 8 and must be indistinguishable from them: every
// buffer the stage writes (normalized, low_rank, injection, block_input, and hidden after
// the inject) is compared with memcmp, for both the prepare and the mixer forms, then
// again through CUDA-graph replay where the fused capture must hold one kernel node less.
// ---------------------------------------------------------------------------

// Stage-1 kernel selection under test. Production goes through the validated public
// wrapper and therefore through whatever flash_next_hyper_decode_route() resolved to.
enum class RouteUnderTest { Legacy, Fused, Production };

const char* route_name(RouteUnderTest route) {
    switch (route) {
        case RouteUnderTest::Legacy: return "legacy";
        case RouteUnderTest::Fused: return "fused";
        default: return "production";
    }
}

struct DecodeRouteBits {
    // After prepare + inject.
    std::vector<std::uint16_t> normalized;
    std::vector<std::uint16_t> low_rank;
    std::vector<std::uint32_t> injection;
    std::vector<std::uint16_t> block_input;
    std::vector<std::uint16_t> hidden;
    // After the mixer form on the injected hidden state.
    std::vector<std::uint16_t> mix_normalized;
    std::vector<std::uint16_t> mix_low_rank;
    std::vector<std::uint32_t> mix_injection; // sentinel: the mixer form never writes it
    std::vector<std::uint16_t> mix_block_input;
};

int count_kernel_nodes(cudaGraph_t graph) {
    std::size_t n = 0;
    CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &n));
    std::vector<cudaGraphNode_t> nodes(n);
    if (n > 0) { CUDA_CHECK(cudaGraphGetNodes(graph, nodes.data(), &n)); }
    int kernels = 0;
    for (std::size_t i = 0; i < n; ++i) {
        cudaGraphNodeType type{};
        CUDA_CHECK(cudaGraphNodeGetType(nodes[i], &type));
        if (type == cudaGraphNodeTypeKernel) { ++kernels; }
    }
    return kernels;
}

template <class T>
void download_bits(std::vector<T>& out, const void* device_ptr, std::size_t count) {
    out.resize(count);
    if (count == 0) { return; }
    CUDA_CHECK(cudaMemcpy(out.data(), device_ptr, count * sizeof(T), cudaMemcpyDeviceToHost));
}

// Runs prepare + inject `repeats` times (eagerly, or as one captured graph replayed
// `repeats` times), then the mixer form once on the resulting hidden state. Every output
// buffer starts from an all-ones sentinel so a region one route writes and the other
// skips shows up as a mismatch rather than as matching stale memory.
DecodeRouteBits run_decode_route(
    ninfer::DeviceContext& device, RouteUnderTest route,
    const ninfer::targets::qwen3_8_flash_next::detail::HyperConnectionWeights& weights,
    const std::vector<std::uint16_t>& h_hidden_bf, const std::vector<std::uint16_t>& h_output_bf,
    int tokens, int repeats, bool via_graph, int* kernel_nodes) {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    const std::size_t concat_n = static_cast<std::size_t>(tokens) * 10'240;
    const std::size_t hidden_n = static_cast<std::size_t>(tokens) * 2'560;
    const std::size_t lr_n     = static_cast<std::size_t>(tokens) * 320;
    const std::size_t inj_n    = static_cast<std::size_t>(tokens) * 4;

    ninfer::DeviceBuffer d_hidden(concat_n * 2);
    d_hidden.copy_from_host(h_hidden_bf.data(), concat_n * 2);
    ninfer::DeviceBuffer d_block_output(hidden_n * 2);
    d_block_output.copy_from_host(h_output_bf.data(), hidden_n * 2);
    ninfer::DeviceBuffer d_block_input(hidden_n * 2);
    // The uploads ride the legacy default stream; device.stream is non-blocking.
    CUDA_CHECK(cudaDeviceSynchronize());

    ninfer::WorkspaceArena workspace(flash_next_hyper_workspace_capacity_bytes(1, tokens));
    auto scope                      = workspace.scope();
    FlashNextHyperWorkspace scratch = allocate_flash_next_hyper_workspace(workspace, tokens);

    auto reset_sentinels = [&] {
        CUDA_CHECK(cudaMemsetAsync(scratch.normalized.data, 0xFF, concat_n * 2, device.stream));
        CUDA_CHECK(cudaMemsetAsync(scratch.low_rank.data, 0xFF, lr_n * 2, device.stream));
        CUDA_CHECK(cudaMemsetAsync(scratch.injection.data, 0xFF, inj_n * sizeof(float),
                                   device.stream));
        CUDA_CHECK(cudaMemsetAsync(d_block_input.p, 0xFF, hidden_n * 2, device.stream));
    };

    ninfer::Tensor hidden_view(d_hidden.p, ninfer::DType::BF16, {10'240, tokens});
    ninfer::Tensor input_view(d_block_input.p, ninfer::DType::BF16, {2'560, tokens});
    ninfer::Tensor output_view(d_block_output.p, ninfer::DType::BF16, {2'560, tokens});
    const HyperMixerWeights mixer{
        .norm           = weights.norm,
        .input_mix_down = weights.input_mix_down,
        .input_mix_up   = weights.input_mix_up,
    };

    auto prepare = [&] {
        switch (route) {
            case RouteUnderTest::Legacy:
                flash_next_hyper_prepare_route_launch(hidden_view, weights, scratch, input_view,
                                                      device.stream,
                                                      FlashNextHyperDecodeRoute::Legacy);
                break;
            case RouteUnderTest::Fused:
                flash_next_hyper_prepare_route_launch(hidden_view, weights, scratch, input_view,
                                                      device.stream,
                                                      FlashNextHyperDecodeRoute::Fused);
                break;
            default:
                flash_next_hyper_prepare(hidden_view, weights, scratch, input_view, device.stream);
                break;
        }
    };
    auto inject = [&] {
        flash_next_hyper_inject(output_view, scratch.injection, hidden_view, device.stream);
    };
    auto mix = [&] {
        switch (route) {
            case RouteUnderTest::Legacy:
                flash_next_hyper_mix_route_launch(hidden_view, mixer, scratch, input_view,
                                                  device.stream, FlashNextHyperDecodeRoute::Legacy);
                break;
            case RouteUnderTest::Fused:
                flash_next_hyper_mix_route_launch(hidden_view, mixer, scratch, input_view,
                                                  device.stream, FlashNextHyperDecodeRoute::Fused);
                break;
            default:
                flash_next_hyper_mix(hidden_view, mixer, scratch, input_view, device.stream);
                break;
        }
    };

    reset_sentinels();
    if (via_graph) {
        cudaGraph_t graph          = nullptr;
        cudaGraphExec_t graph_exec = nullptr;
        CUDA_CHECK(cudaStreamBeginCapture(device.stream, cudaStreamCaptureModeGlobal));
        prepare();
        inject();
        CUDA_CHECK(cudaStreamEndCapture(device.stream, &graph));
        if (kernel_nodes != nullptr) { *kernel_nodes = count_kernel_nodes(graph); }
        CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
        for (int i = 0; i < repeats; ++i) {
            CUDA_CHECK(cudaGraphLaunch(graph_exec, device.stream));
        }
        device.synchronize();
        CUDA_CHECK(cudaGraphExecDestroy(graph_exec));
        CUDA_CHECK(cudaGraphDestroy(graph));
    } else {
        for (int i = 0; i < repeats; ++i) {
            prepare();
            inject();
        }
        device.synchronize();
    }

    DecodeRouteBits bits;
    download_bits(bits.normalized, scratch.normalized.data, concat_n);
    download_bits(bits.low_rank, scratch.low_rank.data, lr_n);
    download_bits(bits.injection, scratch.injection.data, inj_n);
    download_bits(bits.block_input, d_block_input.p, hidden_n);
    download_bits(bits.hidden, d_hidden.p, concat_n);

    reset_sentinels();
    mix();
    device.synchronize();
    download_bits(bits.mix_normalized, scratch.normalized.data, concat_n);
    download_bits(bits.mix_low_rank, scratch.low_rank.data, lr_n);
    download_bits(bits.mix_injection, scratch.injection.data, inj_n);
    download_bits(bits.mix_block_input, d_block_input.p, hidden_n);
    return bits;
}

template <class T>
bool compare_bits(const char* buffer, int tokens, const char* lhs_name, const std::vector<T>& lhs,
                  const char* rhs_name, const std::vector<T>& rhs) {
    if (lhs.size() != rhs.size()) {
        std::cerr << "BIT MISMATCH: T=" << tokens << " " << buffer << " size " << lhs.size()
                  << " (" << lhs_name << ") vs " << rhs.size() << " (" << rhs_name << ")\n";
        return false;
    }
    if (lhs.empty() || std::memcmp(lhs.data(), rhs.data(), lhs.size() * sizeof(T)) == 0) {
        return true;
    }
    std::size_t first = lhs.size();
    std::size_t count = 0;
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        if (lhs[i] != rhs[i]) {
            if (first == lhs.size()) { first = i; }
            ++count;
        }
    }
    std::cerr << "BIT MISMATCH: T=" << tokens << " " << buffer << " " << lhs_name << " vs "
              << rhs_name << " differs in " << count << "/" << lhs.size() << " words; first@"
              << first << " 0x" << std::hex << static_cast<std::uint32_t>(lhs[first]) << " vs 0x"
              << static_cast<std::uint32_t>(rhs[first]) << std::dec << "\n";
    return false;
}

// Evaluates every buffer (no short-circuit) so one failure log names all diverging stages.
bool compare_route_bits(int tokens, const char* lhs_name, const DecodeRouteBits& lhs,
                        const char* rhs_name, const DecodeRouteBits& rhs) {
    bool ok = true;
    ok = compare_bits("normalized", tokens, lhs_name, lhs.normalized, rhs_name, rhs.normalized) && ok;
    ok = compare_bits("low_rank", tokens, lhs_name, lhs.low_rank, rhs_name, rhs.low_rank) && ok;
    ok = compare_bits("injection", tokens, lhs_name, lhs.injection, rhs_name, rhs.injection) && ok;
    ok = compare_bits("block_input", tokens, lhs_name, lhs.block_input, rhs_name, rhs.block_input) && ok;
    ok = compare_bits("hidden(injected)", tokens, lhs_name, lhs.hidden, rhs_name, rhs.hidden) && ok;
    ok = compare_bits("mix.normalized", tokens, lhs_name, lhs.mix_normalized, rhs_name,
                      rhs.mix_normalized) && ok;
    ok = compare_bits("mix.low_rank", tokens, lhs_name, lhs.mix_low_rank, rhs_name, rhs.mix_low_rank) && ok;
    ok = compare_bits("mix.injection(sentinel)", tokens, lhs_name, lhs.mix_injection, rhs_name,
                      rhs.mix_injection) && ok;
    ok = compare_bits("mix.block_input", tokens, lhs_name, lhs.mix_block_input, rhs_name,
                      rhs.mix_block_input) && ok;
    return ok;
}

// Guards against a vacuous pass: a route that never wrote a buffer would leave the 0xFF
// sentinel on both sides and compare equal.
bool bits_were_written(int tokens, const char* route, const DecodeRouteBits& bits) {
    auto any_not = [](const auto& v, auto sentinel) {
        return std::any_of(v.begin(), v.end(), [&](auto x) { return x != sentinel; });
    };
    const bool ok = any_not(bits.normalized, std::uint16_t{0xFFFFU}) &&
                    any_not(bits.low_rank, std::uint16_t{0xFFFFU}) &&
                    any_not(bits.injection, std::uint32_t{0xFFFFFFFFU}) &&
                    any_not(bits.block_input, std::uint16_t{0xFFFFU}) &&
                    any_not(bits.mix_normalized, std::uint16_t{0xFFFFU}) &&
                    any_not(bits.mix_low_rank, std::uint16_t{0xFFFFU}) &&
                    any_not(bits.mix_block_input, std::uint16_t{0xFFFFU}) &&
                    !any_not(bits.mix_injection, std::uint32_t{0xFFFFFFFFU});
    if (!ok) {
        std::cerr << "VACUOUS: T=" << tokens << " route " << route
                  << " left a sentinel-only buffer (or the mixer form wrote injection)\n";
    }
    return ok;
}

int test_decode_route_bit_exact(ninfer::DeviceContext& device) {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    const FlashNextHyperDecodeRoute production = flash_next_hyper_decode_route();
    const RouteUnderTest production_twin = production == FlashNextHyperDecodeRoute::Fused
                                               ? RouteUnderTest::Fused
                                               : RouteUnderTest::Legacy;

    std::cout << "\n--- Flash-Next Hyper-Connection decode-route bit-exact gate ---\n";
    std::cout << "  production route: " << route_name(production_twin)
              << (production == FlashNextHyperDecodeRoute::Legacy
                      ? " (NINFER_FLASH_NEXT_HYPER_LEGACY=1)\n"
                      : " (default)\n");

    // Two regimes: the narrow one keeps the row dot products in the linear part of
    // silu/sigmoid; the wide one saturates them and moves the sum of squares through a
    // different exponent range, so both rsqrtf inputs and the BF16 rounding paths differ.
    struct Regime {
        unsigned seed;
        float weight_span;
        float norm_span;
        float act_span;
        const char* name;
    };
    const Regime regimes[] = {
        {4'242U, 0.05F, 0.02F, 1.5F, "narrow"},
        {9'001U, 0.5F, 0.5F, 8.0F, "wide"},
    };

    for (const Regime& regime : regimes) {
        std::mt19937 rng(regime.seed);
        auto fill_bf16 = [&](std::vector<std::uint16_t>& out, float span) {
            std::uniform_real_distribution<float> dist(-span, span);
            for (auto& value : out) { value = float_to_bf16(dist(rng)); }
        };

        std::vector<std::uint16_t> h_norm(10'240);
        std::vector<std::uint16_t> h_down(320ULL * 10'240);
        std::vector<std::uint16_t> h_up(10'240ULL * 320);
        std::vector<std::uint16_t> h_inject(4ULL * 10'240);
        fill_bf16(h_norm, regime.norm_span);
        fill_bf16(h_down, regime.weight_span);
        fill_bf16(h_up, regime.weight_span);
        fill_bf16(h_inject, regime.weight_span);

        ninfer::DeviceBuffer d_norm(h_norm.size() * 2);
        d_norm.copy_from_host(h_norm.data(), h_norm.size() * 2);
        ninfer::DeviceBuffer d_down(h_down.size() * 2);
        d_down.copy_from_host(h_down.data(), h_down.size() * 2);
        ninfer::DeviceBuffer d_up(h_up.size() * 2);
        d_up.copy_from_host(h_up.data(), h_up.size() * 2);
        ninfer::DeviceBuffer d_inject(h_inject.size() * 2);
        d_inject.copy_from_host(h_inject.data(), h_inject.size() * 2);

        const HyperConnectionWeights weights{
            .block_inject   = bf16_weight(d_inject.p, 4, 10'240),
            .norm           = ninfer::Tensor(d_norm.p, ninfer::DType::BF16, {10'240}),
            .input_mix_down = bf16_weight(d_down.p, 320, 10'240),
            .input_mix_up   = bf16_weight(d_up.p, 10'240, 320),
        };

        // Decode shapes: T = batch at one token per sequence, 1..8 covers B in {1,2,4,8}.
        for (int tokens = 1; tokens <= 8; ++tokens) {
            std::vector<std::uint16_t> h_hidden(static_cast<std::size_t>(tokens) * 10'240);
            std::vector<std::uint16_t> h_output(static_cast<std::size_t>(tokens) * 2'560);
            fill_bf16(h_hidden, regime.act_span);
            fill_bf16(h_output, regime.act_span);

            const DecodeRouteBits legacy = run_decode_route(
                device, RouteUnderTest::Legacy, weights, h_hidden, h_output, tokens, 1, false, nullptr);
            const DecodeRouteBits fused = run_decode_route(
                device, RouteUnderTest::Fused, weights, h_hidden, h_output, tokens, 1, false, nullptr);
            const DecodeRouteBits prod = run_decode_route(
                device, RouteUnderTest::Production, weights, h_hidden, h_output, tokens, 1, false,
                nullptr);

            bool ok = bits_were_written(tokens, "legacy", legacy);
            ok      = bits_were_written(tokens, "fused", fused) && ok;
            ok      = compare_route_bits(tokens, "legacy", legacy, "fused", fused) && ok;
            ok      = compare_route_bits(tokens, "production", prod,
                                         route_name(production_twin),
                                         production_twin == RouteUnderTest::Fused ? fused : legacy) &&
                 ok;
            if (!ok) {
                std::cerr << "FAILED: decode-route bit-exact gate (" << regime.name << ", T="
                          << tokens << ")\n";
                return 1;
            }
            std::cout << "  [" << regime.name << "] T=" << tokens
                      << " legacy == fused == production: identical bits in normalized, "
                         "low_rank, injection, block_input, hidden, mixer outputs\n";
        }

        // Graph replay: three replays of the captured fused chain against three eager
        // legacy passes (hidden accumulates across passes, so the iteration count must
        // match), plus the kernel-node count that is the whole point of the fusion.
        for (int tokens : {1, 8}) {
            std::vector<std::uint16_t> h_hidden(static_cast<std::size_t>(tokens) * 10'240);
            std::vector<std::uint16_t> h_output(static_cast<std::size_t>(tokens) * 2'560);
            fill_bf16(h_hidden, regime.act_span);
            fill_bf16(h_output, regime.act_span);

            constexpr int kReplays = 3;
            int fused_nodes        = 0;
            int legacy_nodes       = 0;
            const DecodeRouteBits legacy_eager = run_decode_route(
                device, RouteUnderTest::Legacy, weights, h_hidden, h_output, tokens, kReplays,
                false, nullptr);
            const DecodeRouteBits fused_graph = run_decode_route(
                device, RouteUnderTest::Fused, weights, h_hidden, h_output, tokens, kReplays, true,
                &fused_nodes);
            const DecodeRouteBits legacy_graph = run_decode_route(
                device, RouteUnderTest::Legacy, weights, h_hidden, h_output, tokens, kReplays,
                true, &legacy_nodes);

            bool ok = compare_route_bits(tokens, "legacy-eager", legacy_eager, "fused-graph",
                                         fused_graph);
            ok = compare_route_bits(tokens, "legacy-eager", legacy_eager, "legacy-graph",
                                    legacy_graph) &&
                 ok;
            // prepare + inject: fused = {fused, mix_up, inject}, legacy = {norm, low_rank, mix_up, inject}.
            if (fused_nodes != 3 || legacy_nodes != 4) {
                std::cerr << "FAILED: T=" << tokens << " kernel nodes fused=" << fused_nodes
                          << " (expected 3) legacy=" << legacy_nodes << " (expected 4)\n";
                ok = false;
            }
            if (!ok) {
                std::cerr << "FAILED: decode-route graph replay gate (" << regime.name
                          << ", T=" << tokens << ")\n";
                return 1;
            }
            std::cout << "  [" << regime.name << "] T=" << tokens << " graph replay x" << kReplays
                      << ": fused graph == legacy eager == legacy graph; kernel nodes fused="
                      << fused_nodes << " legacy=" << legacy_nodes << "\n";
        }
    }

    std::cout << "PASS: test_decode_route_bit_exact\n";
    return 0;
}

int test_kernel_timing_benchmark(ninfer::DeviceContext& device) {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;

    ninfer::DeviceBuffer d_norm(10'240 * 2);
    ninfer::DeviceBuffer d_down(320ULL * 10'240 * 2);
    ninfer::DeviceBuffer d_up(10'240ULL * 320 * 2);
    ninfer::DeviceBuffer d_inject(4ULL * 10'240 * 2);
    d_norm.fill(0);
    d_down.fill(0);
    d_up.fill(0);
    d_inject.fill(0);

    HyperConnectionWeights weights{
        .block_inject   = bf16_weight(d_inject.p, 4, 10'240),
        .norm           = ninfer::Tensor(d_norm.p, ninfer::DType::BF16, {10'240}),
        .input_mix_down = bf16_weight(d_down.p, 320, 10'240),
        .input_mix_up   = bf16_weight(d_up.p, 10'240, 320),
    };

    const bool perf_gates = ninfer_perf_gates_armed();
    std::cout << "\n=== Flash-Next Hyper-Connection Kernel Timing Breakdown ===\n";
    std::cout << "NINFER_PERF_GATES="
              << (perf_gates ? "1 (T=1 25us CUDA-graph gate armed)"
                             : "unset (report-only; T=1 25us CUDA-graph gate not armed)")
              << "\n";
    std::cout << std::fixed << std::setprecision(2);

    for (int tokens : {1, 16, 128, 2048}) {
        ninfer::DeviceBuffer d_hidden(static_cast<std::size_t>(tokens) * 10'240 * 2);
        ninfer::DeviceBuffer d_block_input(static_cast<std::size_t>(tokens) * 2'560 * 2);
        ninfer::DeviceBuffer d_block_output(static_cast<std::size_t>(tokens) * 2'560 * 2);
        d_hidden.fill(0);
        d_block_input.fill(0);
        d_block_output.fill(0);

        ninfer::WorkspaceArena workspace(flash_next_hyper_workspace_capacity_bytes(1, tokens));
        auto scope                      = workspace.scope();
        FlashNextHyperWorkspace scratch = allocate_flash_next_hyper_workspace(workspace, tokens);

        ninfer::Tensor hidden_view(d_hidden.p, ninfer::DType::BF16, {10'240, tokens});
        ninfer::Tensor input_view(d_block_input.p, ninfer::DType::BF16, {2'560, tokens});
        ninfer::Tensor output_view(d_block_output.p, ninfer::DType::BF16, {2'560, tokens});

        // Warmup
        for (int i = 0; i < 20; ++i) {
            flash_next_hyper_prepare(hidden_view, weights, scratch, input_view, device.stream);
            flash_next_hyper_inject(output_view, scratch.injection, hidden_view, device.stream);
        }
        device.synchronize();

        cudaEvent_t start_event, stop_event;
        CUDA_CHECK(cudaEventCreate(&start_event));
        CUDA_CHECK(cudaEventCreate(&stop_event));

        constexpr int kIters = 500;

        // Measure prepare stream time
        CUDA_CHECK(cudaEventRecord(start_event, device.stream));
        for (int i = 0; i < kIters; ++i) {
            flash_next_hyper_prepare(hidden_view, weights, scratch, input_view, device.stream);
        }
        CUDA_CHECK(cudaEventRecord(stop_event, device.stream));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        float total_prep_ms = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(&total_prep_ms, start_event, stop_event));

        // Measure inject stream time
        CUDA_CHECK(cudaEventRecord(start_event, device.stream));
        for (int i = 0; i < kIters; ++i) {
            flash_next_hyper_inject(output_view, scratch.injection, hidden_view, device.stream);
        }
        CUDA_CHECK(cudaEventRecord(stop_event, device.stream));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        float total_inj_ms = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(&total_inj_ms, start_event, stop_event));

        // Measure full chain stream time (Eager)
        CUDA_CHECK(cudaEventRecord(start_event, device.stream));
        for (int i = 0; i < kIters; ++i) {
            flash_next_hyper_prepare(hidden_view, weights, scratch, input_view, device.stream);
            flash_next_hyper_inject(output_view, scratch.injection, hidden_view, device.stream);
        }
        CUDA_CHECK(cudaEventRecord(stop_event, device.stream));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        float total_chain_ms = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(&total_chain_ms, start_event, stop_event));

        // Measure full chain stream time (CUDA Graph Replay)
        cudaGraph_t graph;
        cudaGraphExec_t graph_exec;
        CUDA_CHECK(cudaStreamBeginCapture(device.stream, cudaStreamCaptureModeGlobal));
        flash_next_hyper_prepare(hidden_view, weights, scratch, input_view, device.stream);
        flash_next_hyper_inject(output_view, scratch.injection, hidden_view, device.stream);
        CUDA_CHECK(cudaStreamEndCapture(device.stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

        CUDA_CHECK(cudaEventRecord(start_event, device.stream));
        for (int i = 0; i < kIters; ++i) {
            CUDA_CHECK(cudaGraphLaunch(graph_exec, device.stream));
        }
        CUDA_CHECK(cudaEventRecord(stop_event, device.stream));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        float total_graph_ms = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(&total_graph_ms, start_event, stop_event));

        CUDA_CHECK(cudaGraphExecDestroy(graph_exec));
        CUDA_CHECK(cudaGraphDestroy(graph));

        CUDA_CHECK(cudaEventDestroy(start_event));
        CUDA_CHECK(cudaEventDestroy(stop_event));

        const float avg_prep_us   = (total_prep_ms / kIters) * 1000.0F;
        const float avg_inj_us    = (total_inj_ms / kIters) * 1000.0F;
        const float avg_eager_us  = (total_chain_ms / kIters) * 1000.0F;
        const float avg_graph_us  = (total_graph_ms / kIters) * 1000.0F;

        std::cout << "  Tokens T=" << tokens << ":\n";
        std::cout << "    Eager Prepare Launch Stream : " << avg_prep_us << " us\n";
        std::cout << "    Eager Inject Launch Stream  : " << avg_inj_us << " us\n";
        std::cout << "    Eager Chain Stream Total    : " << avg_eager_us << " us\n";
        std::cout << "    CUDA Graph Hardware Replay  : " << avg_graph_us << " us\n";

        // Decode shapes only: the saving of the norm/low-rank fusion is the difference
        // between the two routes' graph-replayed prepare, 97 launches per token.
        if (tokens <= 8) {
            float route_us[2] = {0.0F, 0.0F};
            const FlashNextHyperDecodeRoute routes[2] = {FlashNextHyperDecodeRoute::Legacy,
                                                         FlashNextHyperDecodeRoute::Fused};
            for (int r = 0; r < 2; ++r) {
                cudaEvent_t route_start, route_stop;
                CUDA_CHECK(cudaEventCreate(&route_start));
                CUDA_CHECK(cudaEventCreate(&route_stop));
                cudaGraph_t route_graph          = nullptr;
                cudaGraphExec_t route_graph_exec = nullptr;
                CUDA_CHECK(cudaStreamBeginCapture(device.stream, cudaStreamCaptureModeGlobal));
                flash_next_hyper_prepare_route_launch(hidden_view, weights, scratch, input_view,
                                                      device.stream, routes[r]);
                CUDA_CHECK(cudaStreamEndCapture(device.stream, &route_graph));
                CUDA_CHECK(cudaGraphInstantiate(&route_graph_exec, route_graph, nullptr, nullptr, 0));
                for (int i = 0; i < 20; ++i) {
                    CUDA_CHECK(cudaGraphLaunch(route_graph_exec, device.stream));
                }
                CUDA_CHECK(cudaEventRecord(route_start, device.stream));
                for (int i = 0; i < kIters; ++i) {
                    CUDA_CHECK(cudaGraphLaunch(route_graph_exec, device.stream));
                }
                CUDA_CHECK(cudaEventRecord(route_stop, device.stream));
                CUDA_CHECK(cudaEventSynchronize(route_stop));
                float route_ms = 0.0F;
                CUDA_CHECK(cudaEventElapsedTime(&route_ms, route_start, route_stop));
                route_us[r] = (route_ms / kIters) * 1000.0F;
                CUDA_CHECK(cudaGraphExecDestroy(route_graph_exec));
                CUDA_CHECK(cudaGraphDestroy(route_graph));
                CUDA_CHECK(cudaEventDestroy(route_start));
                CUDA_CHECK(cudaEventDestroy(route_stop));
            }
            std::cout << "    Prepare graph, legacy route : " << route_us[0]
                      << " us (group_norm + low_rank + mix_up)\n";
            std::cout << "    Prepare graph, fused route  : " << route_us[1]
                      << " us (fused norm/low_rank + mix_up)\n";
            std::cout << "    Fusion saving per prepare   : " << (route_us[0] - route_us[1])
                      << " us  (x97 launches/token = "
                      << (route_us[0] - route_us[1]) * 97.0F / 1000.0F << " ms/token)\n";
        }

        if (tokens == 1) {
            std::cout << "    T=1 25us gate             : " << avg_graph_us << " us vs 25.0 us"
                      << (perf_gates ? " (armed)\n" : " (report-only)\n");
            if (perf_gates && avg_graph_us > 25.0F) {
                std::cerr << "FAILED: T=1 graph chain time (" << avg_graph_us
                          << " us) exceeded 25.0 us threshold!\n";
                return 1;
            }
        }
    }
    std::cout << "===========================================================\n";
    return 0;
}

} // namespace

int main() {
    int device_count              = 0;
    const cudaError_t count_error = cudaGetDeviceCount(&device_count);
    if (cuda_unavailable(count_error) || device_count == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    CUDA_CHECK(count_error);

    ninfer::DeviceContext device(0);

    if (test_basic_unit_injection(device) != 0) { return 1; }
    if (test_synthetic_stage_equivalence(device) != 0) { return 1; }
    if (test_decode_route_bit_exact(device) != 0) { return 1; }
    if (test_kernel_timing_benchmark(device) != 0) { return 1; }

    std::cout << "OK Flash-Next Hyper Connection\n";
    return 0;
}
