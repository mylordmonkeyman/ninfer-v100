#include "core/arena.h"
#include "core/device.h"
#include "core/layout.h"
#include "targets/qwen3_8_flash_next/impl/moe.h"
#include "targets/qwen3_8_flash_next/impl/moe_kernels.h"
#include "targets/qwen3_8_flash_next/impl/moe_route.h"
#include "targets/qwen3_8_flash_next/impl/moe_workspace.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace {

bool cuda_unavailable(cudaError_t error) {
    return error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver;
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

int test_basic_unit(ninfer::DeviceContext& device) {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    ninfer::DeviceBuffer input(2'560 * 2);
    ninfer::DeviceBuffer output(2'560 * 2);
    ninfer::DeviceBuffer router(512ULL * 2'560 * 2);
    ninfer::DeviceBuffer shared_down(2'560ULL * 640 * 2);
    ninfer::DeviceBuffer shared_gate(640ULL * 2'560 * 2);
    ninfer::DeviceBuffer shared_up(640ULL * 2'560 * 2);
    ninfer::DeviceBuffer shared_gate_weight(2'560 * 2);
    std::array<std::uint16_t, 2'560> input_bf16{};
    input_bf16.fill(0x3F80U); // 1.0 BF16
    input.copy_from_host(input_bf16.data(), sizeof(input_bf16));
    output.fill(0xFF);
    router.fill(0);
    shared_down.fill(0);
    shared_gate.fill(0);
    shared_up.fill(0);
    shared_gate_weight.fill(0);

    constexpr std::uint64_t gate_code_bytes_per_expert  = 1'280ULL * 2'560 / 2;
    constexpr std::uint64_t gate_scale_bytes_per_expert = 1'280ULL * 2'560 / 16;
    constexpr std::uint64_t down_code_bytes_per_expert  = 2'560ULL * 640 / 2;
    constexpr std::uint64_t down_scale_bytes_per_expert = 2'560ULL * 640 / 16;
    // Zero router ties select experts 0..9, so the numerical smoke only needs those physical
    // expert spans while the view retains the registered 512-expert stride contract.
    ninfer::DeviceBuffer gate_codes(10 * gate_code_bytes_per_expert);
    ninfer::DeviceBuffer gate_scales(10 * gate_scale_bytes_per_expert);
    ninfer::DeviceBuffer down_codes(10 * down_code_bytes_per_expert);
    ninfer::DeviceBuffer down_scales(10 * down_scale_bytes_per_expert);
    ninfer::DeviceBuffer gate_divisors(512 * sizeof(float));
    ninfer::DeviceBuffer down_divisors(512 * sizeof(float));
    std::vector<std::uint8_t> gate_code_values(10 * gate_code_bytes_per_expert, 0);
    std::vector<std::uint8_t> gate_scale_values(10 * gate_scale_bytes_per_expert, 0x38U);
    std::vector<std::uint8_t> down_code_values(10 * down_code_bytes_per_expert, 0);
    std::vector<std::uint8_t> down_scale_values(10 * down_scale_bytes_per_expert, 0x38U);
    for (std::uint64_t expert = 0; expert < 10; ++expert) {
        for (std::uint64_t row = 0; row < 1'280; ++row) {
            std::fill_n(gate_code_values.begin() +
                            static_cast<std::ptrdiff_t>(expert * gate_code_bytes_per_expert +
                                                        row * (2'560 / 2)),
                        8, 0x22U); // first K16 group: sixteen E2M1 values of 1.0
        }
        for (std::uint64_t row = 0; row < 2'560; ++row) {
            std::fill_n(down_code_values.begin() +
                            static_cast<std::ptrdiff_t>(expert * down_code_bytes_per_expert +
                                                        row * (640 / 2)),
                        8, 0x22U);
        }
    }
    gate_codes.copy_from_host(gate_code_values.data(), gate_code_values.size());
    gate_scales.copy_from_host(gate_scale_values.data(), gate_scale_values.size());
    down_codes.copy_from_host(down_code_values.data(), down_code_values.size());
    down_scales.copy_from_host(down_scale_values.data(), down_scale_values.size());
    std::array<float, 512> divisors{};
    divisors.fill(1.0F);
    gate_divisors.copy_from_host(divisors.data(), sizeof(divisors));
    down_divisors.copy_from_host(divisors.data(), sizeof(divisors));

    MoeWeights weights{
        .router             = bf16_weight(router.p, 512, 2'560),
        .shared_down        = bf16_weight(shared_down.p, 2'560, 640),
        .shared_gate        = bf16_weight(shared_gate.p, 640, 2'560),
        .shared_up          = bf16_weight(shared_up.p, 640, 2'560),
        .shared_gate_weight = bf16_weight(shared_gate_weight.p, 1, 2'560),
        .expert_gate_up     = {.codes                  = static_cast<const std::byte*>(gate_codes.p),
                               .scales                 = static_cast<const std::byte*>(gate_scales.p),
                               .weight_scale_divisors  = static_cast<const float*>(gate_divisors.p),
                               .experts                = 512,
                               .rows                   = 1'280,
                               .columns                = 2'560,
                               .code_bytes_per_expert  = gate_code_bytes_per_expert,
                               .scale_bytes_per_expert = gate_scale_bytes_per_expert},
        .expert_down        = {.codes                  = static_cast<const std::byte*>(down_codes.p),
                               .scales                 = static_cast<const std::byte*>(down_scales.p),
                               .weight_scale_divisors  = static_cast<const float*>(down_divisors.p),
                               .experts                = 512,
                               .rows                   = 2'560,
                               .columns                = 640,
                               .code_bytes_per_expert  = down_code_bytes_per_expert,
                               .scale_bytes_per_expert = down_scale_bytes_per_expert},
    };
    ninfer::Tensor input_view(input.p, ninfer::DType::BF16, {2'560, 1});
    ninfer::Tensor output_view(output.p, ninfer::DType::BF16, {2'560, 1});
    ninfer::WorkspaceArena workspace(flash_next_moe_workspace_capacity_bytes(1, 1));
    flash_next_moe(input_view, weights, output_view, workspace, device.stream);
    device.synchronize();

    std::array<std::uint16_t, 2'560> actual{};
    output.copy_to_host(actual.data(), sizeof(actual));
    // Each selected expert produces 4096, and top-10 softmax renormalization gives each one weight
    // 1/10. Ten selected experts therefore produce exactly 4096.0 (0x4580 in BF16); the zero shared expert adds 0.
    if (!std::all_of(actual.begin(), actual.end(),
                     [](std::uint16_t value) { return value == 0x4580U; })) {
        std::cerr << "Flash-Next encoded NVFP4 MoE did not produce exact BF16 4096.0: first=0x"
                  << std::hex << actual.front() << '\n';
        return 1;
    }
    std::cout << "PASS: test_decode_basic_unit\n";
    return 0;
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

double relative_l2_error(const std::vector<float>& act, const std::vector<float>& exp) {
    double diff_sq = 0.0;
    double exp_sq  = 0.0;
    for (std::size_t i = 0; i < act.size(); ++i) {
        double d = static_cast<double>(act[i]) - static_cast<double>(exp[i]);
        diff_sq += d * d;
        exp_sq  += static_cast<double>(exp[i]) * static_cast<double>(exp[i]);
    }
    return std::sqrt(diff_sq) / (std::sqrt(exp_sq) + 1.0e-12);
}

const char*
down_kernel_name(ninfer::targets::qwen3_8_flash_next::detail::FlashNextMoeDownKernel kernel) {
    using ninfer::targets::qwen3_8_flash_next::detail::FlashNextMoeDownKernel;
    return kernel == FlashNextMoeDownKernel::Legacy ? "legacy" : "path-per-warp";
}

// Bitwise gate for the decode down projection (K1): the legacy one-warp-per-row kernel, the
// path-per-warp kernel, and whichever of the two the launcher selects must write byte-identical
// BF16 [2560, T] outputs from identical inputs. The inputs are random over everything the kernel
// reads: every code byte of every row (all sixteen E2M1 nibbles are finite), valid E4M3 scales
// (no sign bit, never the 0x7F NaN), divisors in [0.5, 2), BF16 shared_down, activations, routing
// ids, alphas and shared scales, for tokens in {1, 2, 3, 4, 5, 6, 7, 8} (grid.y of the decode arm).
// Non-vacuous: the other tests use constant codes, unit scales and a zero shared expert, so they
// cannot see a wrong scale tile, a wrong path order, or a wrong shared-path combine. The kernels
// are chosen through flash_next_moe_down_launch rather than the environment because the launcher
// reads NINFER_FLASH_NEXT_MOE_DOWN_LEGACY once per process.
int test_decode_down_bitwise_gate(ninfer::DeviceContext& device) {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    // The kernels only ever form expert * stride: 64 physical expert spans exercise that index
    // math without the 472 MB a full 512-expert down bank would cost this test.
    constexpr int kExperts                              = 64;
    constexpr int kMaxTokens                            = 8;
    constexpr std::uint64_t down_code_bytes_per_expert  = 2'560ULL * 640 / 2;
    constexpr std::uint64_t down_scale_bytes_per_expert = 2'560ULL * 640 / 16;

    std::mt19937 rng(0x5EED2026U);
    std::uniform_int_distribution<int> scale_dist(0x00, 0x7E); // every finite E4M3 magnitude
    std::uniform_real_distribution<float> divisor_dist(0.5F, 2.0F);
    std::uniform_real_distribution<float> weight_dist(-0.1F, 0.1F);
    std::uniform_real_distribution<float> act_dist(-0.5F, 0.5F);
    std::uniform_real_distribution<float> coef_dist(0.01F, 1.0F);
    std::uniform_int_distribution<int> id_dist(0, kExperts - 1);

    std::vector<std::uint8_t> h_codes(kExperts * down_code_bytes_per_expert);
    static_assert((kExperts * down_code_bytes_per_expert) % 4 == 0);
    for (std::size_t i = 0; i < h_codes.size(); i += 4) {
        const std::uint32_t word = rng();
        std::memcpy(&h_codes[i], &word, sizeof(word));
    }
    std::vector<std::uint8_t> h_scales(kExperts * down_scale_bytes_per_expert);
    for (auto& v : h_scales) { v = static_cast<std::uint8_t>(scale_dist(rng)); }
    std::vector<float> h_divisors(kExperts);
    for (auto& v : h_divisors) { v = divisor_dist(rng); }
    std::vector<std::uint16_t> h_shared_down(2'560ULL * 640);
    for (auto& v : h_shared_down) { v = float_to_bf16(weight_dist(rng)); }

    ninfer::DeviceBuffer d_codes(h_codes.size());
    ninfer::DeviceBuffer d_scales(h_scales.size());
    ninfer::DeviceBuffer d_divisors(h_divisors.size() * sizeof(float));
    ninfer::DeviceBuffer d_shared_down(h_shared_down.size() * sizeof(std::uint16_t));
    d_codes.copy_from_host(h_codes.data(), h_codes.size());
    d_scales.copy_from_host(h_scales.data(), h_scales.size());
    d_divisors.copy_from_host(h_divisors.data(), h_divisors.size() * sizeof(float));
    d_shared_down.copy_from_host(h_shared_down.data(),
                                 h_shared_down.size() * sizeof(std::uint16_t));

    // Only shared_down and expert_down are read by the down launch; the rest stays default.
    MoeWeights weights{
        .shared_down = bf16_weight(d_shared_down.p, 2'560, 640),
        .expert_down = {.codes                  = static_cast<const std::byte*>(d_codes.p),
                        .scales                 = static_cast<const std::byte*>(d_scales.p),
                        .weight_scale_divisors  = static_cast<const float*>(d_divisors.p),
                        .experts                = kExperts,
                        .rows                   = 2'560,
                        .columns                = 640,
                        .code_bytes_per_expert  = down_code_bytes_per_expert,
                        .scale_bytes_per_expert = down_scale_bytes_per_expert},
    };

    const FlashNextMoeDownKernel selected = flash_next_moe_down_kernel_selection();
    std::cout << "  launcher default down kernel: " << down_kernel_name(selected) << "\n";
    for (const FlashNextMoeDownKernel kernel :
         {FlashNextMoeDownKernel::Legacy, FlashNextMoeDownKernel::PathWarp}) {
        const FlashNextMoeDownKernelAttributes attributes =
            flash_next_moe_down_kernel_attributes(kernel);
        std::cout << "  " << down_kernel_name(kernel) << " kernel: " << attributes.threads_per_block
                  << " threads, " << attributes.registers_per_thread << " regs/thread, "
                  << attributes.local_bytes << " B local, " << attributes.static_smem_bytes
                  << " B static smem, " << attributes.max_blocks_per_sm << " CTAs/SM\n";
    }
    // Residency is a performance property, not a correctness one: report loudly, do not fail.
    const FlashNextMoeDownKernelAttributes pathwarp =
        flash_next_moe_down_kernel_attributes(FlashNextMoeDownKernel::PathWarp);
    if (pathwarp.local_bytes != 0) {
        std::cerr << "WARN: path-per-warp down kernel spills " << pathwarp.local_bytes
                  << " B/thread to local memory (expected 0)\n";
    }
    if (pathwarp.max_blocks_per_sm < 4) {
        std::cerr << "WARN: path-per-warp down kernel residency is " << pathwarp.max_blocks_per_sm
                  << " CTAs/SM (designed for 4)\n";
    }

    ninfer::WorkspaceArena workspace(flash_next_moe_workspace_capacity_bytes(1, kMaxTokens));
    ninfer::DeviceBuffer out_legacy(2'560ULL * kMaxTokens * sizeof(std::uint16_t));
    ninfer::DeviceBuffer out_pathwarp(2'560ULL * kMaxTokens * sizeof(std::uint16_t));
    ninfer::DeviceBuffer out_default(2'560ULL * kMaxTokens * sizeof(std::uint16_t));

    for (const int tokens : {1, 2, 3, 4, 5, 6, 7, 8}) {
        const auto scope              = workspace.scope();
        FlashNextMoeWorkspace scratch = allocate_flash_next_moe_workspace(workspace, tokens);

        std::vector<std::int32_t> h_ids(static_cast<std::size_t>(10) * tokens);
        for (auto& v : h_ids) { v = id_dist(rng); }
        std::vector<float> h_alpha(h_ids.size());
        for (auto& v : h_alpha) { v = coef_dist(rng); }
        std::vector<float> h_shared_scale(static_cast<std::size_t>(tokens));
        for (auto& v : h_shared_scale) { v = coef_dist(rng); }
        std::vector<std::uint16_t> h_activations(640ULL * 11 * tokens);
        for (auto& v : h_activations) { v = float_to_bf16(act_dist(rng)); }

        CUDA_CHECK(cudaMemcpy(scratch.ids.data, h_ids.data(), h_ids.size() * sizeof(std::int32_t),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(scratch.alpha.data, h_alpha.data(), h_alpha.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(scratch.shared_scale.data, h_shared_scale.data(),
                              h_shared_scale.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(scratch.activations.data, h_activations.data(),
                              h_activations.size() * sizeof(std::uint16_t),
                              cudaMemcpyHostToDevice));
        // Distinct fill patterns so an unwritten output cannot match by accident.
        out_legacy.fill(0x00);
        out_pathwarp.fill(0xFF);
        out_default.fill(0x55);
        // The uploads and memsets above ran on the legacy default stream; the compute stream is
        // non-blocking, so a stream sync would not order them before the launches.
        CUDA_CHECK(cudaDeviceSynchronize());

        ninfer::Tensor legacy_view(out_legacy.p, ninfer::DType::BF16, {2'560, tokens});
        ninfer::Tensor pathwarp_view(out_pathwarp.p, ninfer::DType::BF16, {2'560, tokens});
        ninfer::Tensor default_view(out_default.p, ninfer::DType::BF16, {2'560, tokens});
        flash_next_moe_down_launch(FlashNextMoeDownKernel::Legacy, weights, scratch, tokens,
                                   legacy_view, device.stream);
        flash_next_moe_down_launch(FlashNextMoeDownKernel::PathWarp, weights, scratch, tokens,
                                   pathwarp_view, device.stream);
        flash_next_moe_down_launch(selected, weights, scratch, tokens, default_view,
                                   device.stream);
        device.synchronize();

        const std::size_t values    = 2'560ULL * tokens;
        const std::size_t out_bytes = values * sizeof(std::uint16_t);
        std::vector<std::uint16_t> legacy_bits(values);
        std::vector<std::uint16_t> pathwarp_bits(values);
        std::vector<std::uint16_t> default_bits(values);
        out_legacy.copy_to_host(legacy_bits.data(), out_bytes);
        out_pathwarp.copy_to_host(pathwarp_bits.data(), out_bytes);
        out_default.copy_to_host(default_bits.data(), out_bytes);

        // Non-vacuous: the reference must be finite and overwhelmingly nonzero, otherwise two
        // kernels agreeing on NaN or on an untouched buffer would pass the memcmp below.
        std::size_t non_finite = 0;
        std::size_t zeros      = 0;
        for (const std::uint16_t v : legacy_bits) {
            if ((v & 0x7F80U) == 0x7F80U) { ++non_finite; }
            if ((v & 0x7FFFU) == 0U) { ++zeros; }
        }
        if (non_finite != 0 || zeros > values / 100) {
            std::cerr << "FAILED: down gate reference at tokens=" << tokens << " has "
                      << non_finite << " non-finite and " << zeros << " zero outputs of "
                      << values << "\n";
            return 1;
        }

        const auto report_mismatch = [&](const char* name,
                                         const std::vector<std::uint16_t>& other) {
            std::size_t mismatches = 0;
            std::size_t first      = values;
            for (std::size_t i = 0; i < values; ++i) {
                if (legacy_bits[i] != other[i]) {
                    if (first == values) { first = i; }
                    ++mismatches;
                }
            }
            std::cerr << "FAILED: " << name << " down kernel differs from legacy at tokens="
                      << tokens << ": " << mismatches << " of " << values
                      << " BF16 values; first at token " << first / 2'560 << " row "
                      << first % 2'560 << " legacy=0x" << std::hex << legacy_bits[first]
                      << " other=0x" << other[first] << std::dec << "\n";
        };
        if (std::memcmp(legacy_bits.data(), pathwarp_bits.data(), out_bytes) != 0) {
            report_mismatch("path-per-warp", pathwarp_bits);
            return 1;
        }
        if (std::memcmp(legacy_bits.data(), default_bits.data(), out_bytes) != 0) {
            report_mismatch("launcher-selected", default_bits);
            return 1;
        }
        std::cout << "  tokens=" << tokens << ": legacy == path-per-warp == launcher default ("
                  << values << " BF16 values, " << non_finite << " non-finite, " << zeros
                  << " zero)\n";
    }
    std::cout << "PASS: test_decode_down_bitwise_gate\n";
    return 0;
}

int test_prefill_equivalence_and_benchmark(ninfer::DeviceContext& device) {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    std::mt19937 rng(1337);
    std::uniform_real_distribution<float> dist_router(-0.1F, 0.1F);
#if defined(NINFER_VOLTA_BUILD)
    // Constrained-residency Phase-9 gate: positive activations plus signed router
    // rows guarantee that experts 0..9 are the selected routed set.
    std::uniform_real_distribution<float> dist_act(0.01F, 0.05F);
    constexpr int kResidentExperts = 10;
#else
    std::uniform_real_distribution<float> dist_act(-0.05F, 0.05F);
    constexpr int kResidentExperts = 512;
#endif

    constexpr std::uint64_t gate_code_bytes_per_expert  = 1'280ULL * 2'560 / 2;
    constexpr std::uint64_t gate_scale_bytes_per_expert = 1'280ULL * 2'560 / 16;
    constexpr std::uint64_t down_code_bytes_per_expert  = 2'560ULL * 640 / 2;
    constexpr std::uint64_t down_scale_bytes_per_expert = 2'560ULL * 640 / 16;

    // Phase 9 permits constrained residency. Volta keeps only the ten experts
    // that the test router can select; other builds retain the full-residency
    // characterization workload.
    ninfer::DeviceBuffer gate_codes(
        static_cast<std::uint64_t>(kResidentExperts) * gate_code_bytes_per_expert);
    ninfer::DeviceBuffer gate_scales(
        static_cast<std::uint64_t>(kResidentExperts) * gate_scale_bytes_per_expert);
    ninfer::DeviceBuffer down_codes(
        static_cast<std::uint64_t>(kResidentExperts) * down_code_bytes_per_expert);
    ninfer::DeviceBuffer down_scales(
        static_cast<std::uint64_t>(kResidentExperts) * down_scale_bytes_per_expert);
    ninfer::DeviceBuffer gate_divisors(512 * sizeof(float));
    ninfer::DeviceBuffer down_divisors(512 * sizeof(float));

    std::vector<std::uint8_t> h_gate_codes(
        static_cast<std::uint64_t>(kResidentExperts) * gate_code_bytes_per_expert, 0x22U);
    std::vector<std::uint8_t> h_gate_scales(
        static_cast<std::uint64_t>(kResidentExperts) * gate_scale_bytes_per_expert, 0x38U);
    std::vector<std::uint8_t> h_down_codes(
        static_cast<std::uint64_t>(kResidentExperts) * down_code_bytes_per_expert, 0x22U);
    std::vector<std::uint8_t> h_down_scales(
        static_cast<std::uint64_t>(kResidentExperts) * down_scale_bytes_per_expert, 0x38U);
    std::vector<float> h_divisors(512, 1.0F);

    gate_codes.copy_from_host(h_gate_codes.data(), h_gate_codes.size());
    gate_scales.copy_from_host(h_gate_scales.data(), h_gate_scales.size());
    down_codes.copy_from_host(h_down_codes.data(), h_down_codes.size());
    down_scales.copy_from_host(h_down_scales.data(), h_down_scales.size());
    gate_divisors.copy_from_host(h_divisors.data(), sizeof(float) * 512);
    down_divisors.copy_from_host(h_divisors.data(), sizeof(float) * 512);

    ninfer::DeviceBuffer d_router(512ULL * 2'560 * 2);
    ninfer::DeviceBuffer d_shared_down(2'560ULL * 640 * 2);
    ninfer::DeviceBuffer d_shared_gate(640ULL * 2'560 * 2);
    ninfer::DeviceBuffer d_shared_up(640ULL * 2'560 * 2);
    ninfer::DeviceBuffer d_shared_gate_weight(2'560 * 2);

    std::vector<std::uint16_t> h_router(512ULL * 2'560);
#if defined(NINFER_VOLTA_BUILD)
    const std::uint16_t selected_router_word = float_to_bf16(0.0625F);
    const std::uint16_t excluded_router_word = float_to_bf16(-0.0625F);
    std::fill(h_router.begin(), h_router.end(), excluded_router_word);
    for (int expert = 0; expert < kResidentExperts; ++expert) {
        std::fill_n(h_router.begin() + static_cast<std::ptrdiff_t>(expert) * 2'560,
                    2'560, selected_router_word);
    }
#else
    for (auto& v : h_router) { v = float_to_bf16(dist_router(rng)); }
#endif
    d_router.copy_from_host(h_router.data(), h_router.size() * 2);

    std::vector<std::uint16_t> h_shared_down(2'560ULL * 640);
    for (auto& v : h_shared_down) { v = float_to_bf16(dist_router(rng)); }
    d_shared_down.copy_from_host(h_shared_down.data(), h_shared_down.size() * 2);

    std::vector<std::uint16_t> h_shared_gate(640ULL * 2'560);
    for (auto& v : h_shared_gate) { v = float_to_bf16(dist_router(rng)); }
    d_shared_gate.copy_from_host(h_shared_gate.data(), h_shared_gate.size() * 2);

    std::vector<std::uint16_t> h_shared_up(640ULL * 2'560);
    for (auto& v : h_shared_up) { v = float_to_bf16(dist_router(rng)); }
    d_shared_up.copy_from_host(h_shared_up.data(), h_shared_up.size() * 2);

    std::vector<std::uint16_t> h_shared_gate_weight(2'560);
    for (auto& v : h_shared_gate_weight) { v = float_to_bf16(dist_router(rng)); }
    d_shared_gate_weight.copy_from_host(h_shared_gate_weight.data(), h_shared_gate_weight.size() * 2);

    MoeWeights weights{
        .router             = bf16_weight(d_router.p, 512, 2'560),
        .shared_down        = bf16_weight(d_shared_down.p, 2'560, 640),
        .shared_gate        = bf16_weight(d_shared_gate.p, 640, 2'560),
        .shared_up          = bf16_weight(d_shared_up.p, 640, 2'560),
        .shared_gate_weight = bf16_weight(d_shared_gate_weight.p, 1, 2'560),
        .expert_gate_up     = {.codes                  = static_cast<const std::byte*>(gate_codes.p),
                               .scales                 = static_cast<const std::byte*>(gate_scales.p),
                               .weight_scale_divisors  = static_cast<const float*>(gate_divisors.p),
                               .experts                = 512,
                               .rows                   = 1'280,
                               .columns                = 2'560,
                               .code_bytes_per_expert  = gate_code_bytes_per_expert,
                               .scale_bytes_per_expert = gate_scale_bytes_per_expert},
        .expert_down        = {.codes                  = static_cast<const std::byte*>(down_codes.p),
                               .scales                 = static_cast<const std::byte*>(down_scales.p),
                               .weight_scale_divisors  = static_cast<const float*>(down_divisors.p),
                               .experts                = 512,
                               .rows                   = 2'560,
                               .columns                = 640,
                               .code_bytes_per_expert  = down_code_bytes_per_expert,
                               .scale_bytes_per_expert = down_scale_bytes_per_expert},
    };

    double max_rel_l2 = 0.0;
    float time_128_us = 0.0F;
    float time_512_us = 0.0F;
    float time_2048_us = 0.0F;

    for (int tokens : {128, 319, 512, 2048}) {
        std::vector<std::uint16_t> h_input(static_cast<std::size_t>(tokens) * 2'560);
        for (auto& v : h_input) { v = float_to_bf16(dist_act(rng)); }

        ninfer::DeviceBuffer d_input(h_input.size() * 2);
        d_input.copy_from_host(h_input.data(), h_input.size() * 2);
        ninfer::DeviceBuffer d_out_prefill(h_input.size() * 2);
        ninfer::DeviceBuffer d_out_decode(h_input.size() * 2);

        ninfer::Tensor in_view(d_input.p, ninfer::DType::BF16, {2'560, tokens});
        ninfer::Tensor out_prefill_view(d_out_prefill.p, ninfer::DType::BF16, {2'560, tokens});
        ninfer::Tensor out_decode_view(d_out_decode.p, ninfer::DType::BF16, {2'560, tokens});

        ninfer::WorkspaceArena ws_prefill(flash_next_moe_workspace_capacity_bytes(1, tokens));
        ninfer::WorkspaceArena ws_decode(flash_next_moe_workspace_capacity_bytes(1, 8));

        // 1. Run grouped prefill
        flash_next_moe(in_view, weights, out_prefill_view, ws_prefill, device.stream);
        device.synchronize();

        // 2. Run decode reference in batches of 8 tokens (decode path T <= 8)
        for (int t = 0; t < tokens; t += 8) {
            const int cur_t = std::min(8, tokens - t);
            ninfer::Tensor in_t(static_cast<__nv_bfloat16*>(d_input.p) + static_cast<std::int64_t>(t) * 2'560,
                                ninfer::DType::BF16, {2'560, cur_t});
            ninfer::Tensor out_t(static_cast<__nv_bfloat16*>(d_out_decode.p) + static_cast<std::int64_t>(t) * 2'560,
                                 ninfer::DType::BF16, {2'560, cur_t});
            flash_next_moe(in_t, weights, out_t, ws_decode, device.stream);
        }
        device.synchronize();

        std::vector<std::uint16_t> act_prefill_bf(h_input.size());
        std::vector<std::uint16_t> act_decode_bf(h_input.size());
        d_out_prefill.copy_to_host(act_prefill_bf.data(), act_prefill_bf.size() * 2);
        d_out_decode.copy_to_host(act_decode_bf.data(), act_decode_bf.size() * 2);

        std::vector<float> act_prefill_f(h_input.size());
        std::vector<float> act_decode_f(h_input.size());
        for (std::size_t i = 0; i < h_input.size(); ++i) {
            act_prefill_f[i] = bf16_to_float(act_prefill_bf[i]);
            act_decode_f[i]  = bf16_to_float(act_decode_bf[i]);
        }

        int nan_count = 0;
        for (std::size_t i = 0; i < act_prefill_f.size(); ++i) {
            if (std::isnan(act_prefill_f[i]) || std::isinf(act_prefill_f[i])) {
                nan_count++;
            }
        }
        if (nan_count > 0) {
            std::cerr << "FAILED: Prefill output contains " << nan_count << " non-finite values!\n";
            return 1;
        }

        double err = relative_l2_error(act_prefill_f, act_decode_f);
        max_rel_l2 = std::max(max_rel_l2, err);
        std::cout << "  Tokens T=" << tokens << " Prefill vs Decode Rel-L2 Error: "
                  << std::scientific << std::setprecision(6) << err << "\n" << std::flush;

#if defined(NINFER_VOLTA_BUILD)
        // Volta always takes the software W4A16 prefill route, including large T.
        if (err > 1.0e-3) {
            std::cerr << "FAILED: Volta W4A16 prefill T=" << tokens
                      << " rel-L2 error exceeded tolerance 1e-3: " << err << "\n";
            return 1;
        }
#else
        if (tokens < kFlashNextMoeMmaPrefillThreshold) {
            // The SIMT path should match the decode reference tightly.
            if (err > 1.0e-3) {
                std::cerr << "FAILED: SIMT prefill T=" << tokens
                          << " rel-L2 error exceeded tolerance 1e-3: " << err << "\n";
                return 1;
            }
        } else {
            // The MMA path dynamically quantizes activations to W4A4.
            if (err > 0.20) {
                std::cerr << "FAILED: MMA prefill T=" << tokens
                          << " rel-L2 error exceeded tolerance 0.20: " << err << "\n";
                return 1;
            }
        }
#endif

        // Detailed breakdown
        cudaEvent_t ev_start, ev_route, ev_end;
        cudaEventCreate(&ev_start);
        cudaEventCreate(&ev_route);
        cudaEventCreate(&ev_end);

        const auto scope = ws_prefill.scope();
        FlashNextMoeWorkspace scratch = allocate_flash_next_moe_workspace(ws_prefill, tokens);

        flash_next_route(in_view, weights.router, weights.shared_gate_weight, scratch.scores, scratch.ids,
                         scratch.alpha, scratch.shared_scale, device.stream);
        flash_next_moe_kernels_launch(in_view, weights, scratch, out_prefill_view, device.stream);
        device.synchronize();

#if !defined(NINFER_VOLTA_BUILD)
        if (tokens >= kFlashNextMoeMmaPrefillThreshold) {
            const std::size_t id_bytes =
                static_cast<std::size_t>(tokens) * 10U * sizeof(std::int32_t);
            std::vector<std::int32_t> ids_old(static_cast<std::size_t>(tokens) * 10U);
            std::vector<std::int32_t> ids_new(ids_old.size());
#ifdef _WIN32
            (void)_putenv_s("NINFER_FLASH_NEXT_MOE_SHARED_MMA", "0");
#else
            (void)setenv("NINFER_FLASH_NEXT_MOE_SHARED_MMA", "0", 1);
#endif
            flash_next_route(in_view, weights.router, weights.shared_gate_weight, scratch.scores,
                             scratch.ids, scratch.alpha, scratch.shared_scale, device.stream);
            device.synchronize();
            CUDA_CHECK(cudaMemcpy(ids_old.data(), scratch.ids.data, id_bytes,
                                  cudaMemcpyDeviceToHost));
#ifdef _WIN32
            (void)_putenv_s("NINFER_FLASH_NEXT_MOE_SHARED_MMA", "1");
#else
            (void)setenv("NINFER_FLASH_NEXT_MOE_SHARED_MMA", "1", 1);
#endif
            flash_next_route(in_view, weights.router, weights.shared_gate_weight, scratch.scores,
                             scratch.ids, scratch.alpha, scratch.shared_scale, device.stream);
            device.synchronize();
            CUDA_CHECK(cudaMemcpy(ids_new.data(), scratch.ids.data, id_bytes,
                                  cudaMemcpyDeviceToHost));
            int nonzero = 0;
            int flips   = 0;
            for (std::size_t i = 0; i < ids_old.size(); ++i) {
                if (ids_old[i] != 0) { ++nonzero; }
                if (ids_old[i] != ids_new[i]) { ++flips; }
            }
            if (nonzero == 0) {
                std::cerr << "FAILED: route-id compare vacuous at T=" << tokens << "\n";
                return 1;
            }
            if (flips != 0) {
                std::cerr << "FAILED: MMA router flipped " << flips << " of " << ids_old.size()
                          << " top-10 slots at T=" << tokens << " (nonzero=" << nonzero << ")\n";
                return 1;
            }
            std::cout << "  Tokens T=" << tokens
                      << " router ids bitwise identical vs scalar (slots=" << ids_old.size()
                      << " nonzero=" << nonzero << ")\n";
        }
#endif

        const int kIters = (tokens >= 2048) ? 10 : 50;
        cudaEventRecord(ev_start, device.stream);
        for (int i = 0; i < kIters; ++i) {
            flash_next_route(in_view, weights.router, weights.shared_gate_weight, scratch.scores, scratch.ids,
                             scratch.alpha, scratch.shared_scale, device.stream);
        }
        cudaEventRecord(ev_route, device.stream);
        for (int i = 0; i < kIters; ++i) {
            flash_next_moe_kernels_launch(in_view, weights, scratch, out_prefill_view, device.stream);
        }
        cudaEventRecord(ev_end, device.stream);
        cudaEventSynchronize(ev_end);

        float route_ms = 0.0f, compute_ms = 0.0f;
        cudaEventElapsedTime(&route_ms, ev_start, ev_route);
        cudaEventElapsedTime(&compute_ms, ev_route, ev_end);

        float avg_route_us = (route_ms / kIters) * 1000.0f;
        float avg_comp_us  = (compute_ms / kIters) * 1000.0f;
        if (tokens == 128) time_128_us = avg_comp_us;
        if (tokens == 512) time_512_us = avg_comp_us;
        if (tokens == 2048) time_2048_us = avg_comp_us;

        std::cout << "  Tokens T=" << tokens << " Timing Breakdown:\n"
                  << "    Router Time : " << avg_route_us << " us\n"
                  << "    MoE GEMMs   : " << avg_comp_us  << " us\n"
                  << "    Total Layer : " << avg_route_us + avg_comp_us << " us\n";

        if (tokens == 2048) {
            cudaEvent_t ev0, ev1, ev2, ev3, ev4, ev5, ev6;
            cudaEventCreate(&ev0); cudaEventCreate(&ev1); cudaEventCreate(&ev2);
            cudaEventCreate(&ev3); cudaEventCreate(&ev4); cudaEventCreate(&ev5);
            cudaEventCreate(&ev6);

            cudaEventRecord(ev0, device.stream);
            for (int i = 0; i < kIters; ++i) {
                // 1. Build groups
                flash_next_moe_kernels_launch(in_view, weights, scratch, out_prefill_view, device.stream);
            }
            cudaEventRecord(ev6, device.stream);
            cudaEventSynchronize(ev6);
        }
    }

    const float scaling_factor_512  = time_512_us / (time_128_us + 1e-6F);
    const float scaling_factor_2048 = time_2048_us / (time_128_us + 1e-6F);
    const float total_128_ms   = (time_128_us * 48.0F) / 1000.0F;
    const float total_2048_ms  = (time_2048_us * 48.0F) / 1000.0F;

    std::cout << "\n=== Flash-Next Prefill MoE Summary ===\n";
    std::cout << "  T=128 48-layer MoE time  : " << total_128_ms << " ms\n";
    std::cout << "  T=2048 48-layer MoE time : " << total_2048_ms << " ms\n";
    std::cout << "  T=512 vs T=128 scaling   : " << scaling_factor_512 << "x\n";
    std::cout << "  T=2048 vs T=128 scaling  : " << scaling_factor_2048 << "x\n";
    std::cout << "======================================\n" << std::flush;

    std::cout << "PASS: test_prefill_equivalence_and_benchmark\n";
    return 0;
}

} // namespace

// Sequence 14 envelope guard: the MoE workspace envelope is computed once at the chunk capacity,
// but a tail chunk below the MMA threshold takes the SIMT arm and carves an FP32 [2560, 10, T]
// intermediate instead of the BF16 [2560, 10 * T] staging. Every SIMT tail must fit inside the
// envelope at every capacity.
int test_prefill_workspace_envelope_covers_simt_tail() {
    using ninfer::targets::qwen3_8_flash_next::detail::allocate_flash_next_moe_workspace;
    using ninfer::targets::qwen3_8_flash_next::detail::flash_next_moe_workspace_capacity_bytes;
    using ninfer::targets::qwen3_8_flash_next::detail::kFlashNextMoeMmaPrefillThreshold;
    const std::array<std::int32_t, 5> capacities = {256, 512, 1024, 2048, 4096};
    const std::array<std::int32_t, 4> tails = {
        9, kFlashNextMoeMmaPrefillThreshold - 1, kFlashNextMoeMmaPrefillThreshold,
        kFlashNextMoeMmaPrefillThreshold + 1};
    int checked = 0;
    for (const std::int32_t capacity : capacities) {
        const std::size_t envelope = flash_next_moe_workspace_capacity_bytes(1, capacity);
        for (const std::int32_t tail : tails) {
            const std::int32_t tokens = std::min(tail, capacity);
            ninfer::WorkspaceLayoutBuilder layout;
            (void)allocate_flash_next_moe_workspace(layout, tokens);
            const std::size_t need = layout.peak_bytes(256);
            if (need > envelope) {
                std::cerr << "FAILED: MoE workspace envelope for capacity " << capacity << " ("
                          << envelope << " bytes) does not cover a " << tokens
                          << "-token chunk (" << need << " bytes)\n";
                return 1;
            }
            ++checked;
        }
    }
    if (checked != 20) {
        std::cerr << "FAILED: envelope test checked " << checked << " cases, expected 20\n";
        return 1;
    }
    std::cout << "  envelope covers SIMT tails at capacities 256..4096 (" << checked << " cases)\n";
    return 0;
}

int main() {
    int device_count              = 0;
    const cudaError_t count_error = cudaGetDeviceCount(&device_count);
    if (cuda_unavailable(count_error) || device_count == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    CUDA_CHECK(count_error);

    ninfer::DeviceContext device(0);

    if (test_basic_unit(device) != 0) {
        std::cerr << "FAILED: test_basic_unit\n";
        return 1;
    }
    std::cout << "PASS: test_decode_basic_unit\n";

    if (test_decode_down_bitwise_gate(device) != 0) {
        std::cerr << "FAILED: test_decode_down_bitwise_gate\n";
        return 1;
    }

    if (test_prefill_workspace_envelope_covers_simt_tail() != 0) {
        std::cerr << "FAILED: test_prefill_workspace_envelope_covers_simt_tail\n";
        return 1;
    }
    std::cout << "PASS: test_prefill_workspace_envelope_covers_simt_tail\n";
    if (test_prefill_equivalence_and_benchmark(device) != 0) {
        std::cerr << "FAILED: test_prefill_equivalence_and_benchmark\n";
        return 1;
    }

    std::cout << "OK Flash-Next MoE\n";
    return 0;
}
