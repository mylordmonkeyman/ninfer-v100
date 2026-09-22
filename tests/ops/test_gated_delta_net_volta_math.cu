#include "ops/linear_attention/gated_delta_net/volta/bf16_sm70.cuh"
#include "ops/linear_attention/gated_delta_net/volta/common.cuh"
#include "ops/linear_attention/gated_delta_net/volta/mma_tiles.cuh"
#include "ops/linear_attention/gated_delta_net/volta/decay.cuh"
#include "ops/linear_attention/gated_delta_net/volta/neumann_solve.cuh"
#include "ops/linear_attention/gated_delta_net/volta/scaling.cuh"
#include "ops/linear_attention/gated_delta_net/volta/triangular_solve.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <iostream>

namespace volta = ninfer::ops::detail::gated_delta_net::volta;

namespace {

bool cuda_ok(cudaError_t status, const char* what) {
    if (status == cudaSuccess) { return true; }
    std::cerr << what << ": " << cudaGetErrorString(status) << "\n";
    return false;
}

__global__ void bf16_conversion_kernel(const std::uint16_t* bf16_bits,
                                       std::uint32_t* expanded_float_bits,
                                       int bf16_count,
                                       const std::uint32_t* float_bits,
                                       std::uint16_t* rounded_bf16_bits,
                                       int float_count) {
    const int i = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);
    if (i < bf16_count) {
        const float expanded = volta::bf16_bits_to_float(bf16_bits[i]);
        expanded_float_bits[i] = __float_as_uint(expanded);
    }
    if (i < float_count) {
        rounded_bf16_bits[i] = volta::float_to_bf16_rn(__uint_as_float(float_bits[i]));
    }
}

struct FragmentWords {
    std::uint32_t a0;
    std::uint32_t a1;
    std::uint32_t b0;
    std::uint32_t b1;
};

__global__ void macro16_fragment_kernel(const __half* a, const __half* b,
                                        FragmentWords* words) {
    const unsigned lane = threadIdx.x & 31U;
    const auto af = volta::load_mma884_a_macro16(a, 4);
    const auto bf = volta::load_mma884_b_macro16_col(b, 4);
    words[lane] = {af.x0, af.x1, bf.x0, bf.x1};
}

__global__ void macro16_basis_kernel(const __half* a, const __half* b, float* output) {
    const unsigned lane = threadIdx.x & 31U;
    const int group = ninfer::ops::volta_mma884_group(lane);
    const auto macro = volta::mma_macro16_group_coord(group);

    const auto af = volta::load_mma884_a_macro16(a, 4);
    const auto bf = volta::load_mma884_b_macro16_col(b, 4);
    ninfer::ops::VoltaMma884Accumulator accum{};
    accum.clear();
    ninfer::ops::volta_mma884_f16_f32(accum, af, bf);

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const auto local = ninfer::ops::volta_mma884_accumulator_coordinate(lane, i);
        const int row = macro.row + local.row;
        const int col = macro.col + local.col;
        output[row * 16 + col] = accum.x[i];
    }
}

int test_bridge_planner() {
    int failures = 0;

    volta::RangeStats observed{};
    volta::range_observe(observed, 0.0F);
    volta::range_observe(observed, -0x1p-20F);
    volta::range_observe(observed, 1.0F);
    if (observed.max_abs != 1.0F || observed.min_nonzero_abs != 0x1p-20F) {
        std::cerr << "range observation mismatch\n";
        ++failures;
    }

    const auto zero = volta::plan_fp16_bridge({0.0F, 0.0F});
    if (zero.mode != volta::BridgeMode::Fp16Mma || zero.mul != 1.0F || zero.inv != 1.0F) {
        std::cerr << "all-zero bridge plan mismatch\n";
        ++failures;
    }

    const auto safe = volta::plan_fp16_bridge({1.0F, 0x1p-20F});
    if (safe.mode != volta::BridgeMode::Fp16Mma || safe.mul != 32768.0F ||
        safe.inv != 0x1p-15F || safe.range.max_abs * safe.mul > volta::kFp16Headroom ||
        safe.range.min_nonzero_abs * safe.mul < volta::kFp16MinSafe) {
        std::cerr << "safe bridge plan mismatch\n";
        ++failures;
    }

    const auto impossible = volta::plan_fp16_bridge({32768.0F, 0x1p-30F});
    if (impossible.mode != volta::BridgeMode::Fp32Simt) {
        std::cerr << "unsafe bridge range did not fall back to FP32\n";
        ++failures;
    }

    const auto nonfinite = volta::plan_fp16_bridge({CUDART_INF_F, 1.0F});
    if (nonfinite.mode != volta::BridgeMode::Fp32Simt) {
        std::cerr << "non-finite bridge range did not fall back to FP32\n";
        ++failures;
    }
    return failures;
}

int test_decay_values() {
    std::array<float, volta::kChunkSize> g{};
    volta::DecayValues decay{};
    volta::compute_decay_values(g.data(), decay);

    for (int t = 0; t < volta::kChunkSize; ++t) {
        if (decay.alpha[t] != 1.0F || decay.prefix[t] != 1.0F || decay.suffix[t] != 1.0F) {
            std::cerr << "zero-g decay mismatch at t=" << t << "\n";
            return 1;
        }
    }
    for (int row = 0; row < volta::kChunkSize; ++row) {
        for (int col = 0; col <= row; ++col) {
            if (volta::pairwise_decay(decay, row, col) != 1.0F) {
                std::cerr << "zero-g pairwise decay mismatch\n";
                return 1;
            }
        }
    }
    return 0;
}

int test_solve_primitives() {
    constexpr int kN = volta::kChunkSize;
    std::array<float, kN * kN> b{};
    std::array<float, kN * kN> exact{};
    std::array<float, kN * kN> neumann{};
    std::array<float, volta::kSolveScratchElements> scratch{};

    for (int row = 1; row < kN; ++row) {
        b[static_cast<std::size_t>(row) * kN + row - 1] = 0.25F;
    }

    volta::exact_inverse_bt32(b.data(), kN, exact.data(), kN, scratch.data());
    volta::neumann2_inverse_bt32(b.data(), kN, neumann.data(), kN);

    for (int row = 0; row < kN; ++row) {
        for (int col = 0; col < kN; ++col) {
            float expected_exact = 0.0F;
            if (col <= row) {
                expected_exact = 1.0F;
                for (int i = col; i < row; ++i) { expected_exact *= -0.25F; }
            }
            const float got_exact = exact[static_cast<std::size_t>(row) * kN + col];
            if (std::bit_cast<std::uint32_t>(got_exact) !=
                std::bit_cast<std::uint32_t>(expected_exact)) {
                std::cerr << "exact triangular inverse mismatch at (" << row << ',' << col
                          << ") got=" << got_exact << " expected=" << expected_exact << "\n";
                return 1;
            }

            float expected_neumann = 0.0F;
            const int distance = row - col;
            if (distance == 0) {
                expected_neumann = 1.0F;
            } else if (distance == 1) {
                expected_neumann = -0.25F;
            } else if (distance == 2) {
                expected_neumann = 0.0625F;
            }
            const float got_neumann = neumann[static_cast<std::size_t>(row) * kN + col];
            if (std::bit_cast<std::uint32_t>(got_neumann) !=
                std::bit_cast<std::uint32_t>(expected_neumann)) {
                std::cerr << "Neumann2 mismatch at (" << row << ',' << col << ") got="
                          << got_neumann << " expected=" << expected_neumann << "\n";
                return 1;
            }
        }
    }
    return 0;
}

int test_bf16_conversion() {
    constexpr std::array<std::uint16_t, 8> kBf16Inputs = {
        0x0000U, 0x8000U, 0x0001U, 0x3f80U,
        0xbf80U, 0x7f80U, 0x7fc1U, 0xffffU,
    };
    constexpr std::array<std::uint32_t, kBf16Inputs.size()> kExpandedExpected = {
        0x00000000U, 0x80000000U, 0x00010000U, 0x3f800000U,
        0xbf800000U, 0x7f800000U, 0x7fc10000U, 0xffff0000U,
    };

    constexpr std::array<std::uint32_t, 12> kFloatInputs = {
        0x00000000U, 0x80000000U, 0x3f800000U, 0x3f808000U,
        0x3f818000U, 0xbf808000U, 0x00000001U, 0x00008000U,
        0x00008001U, 0x7f800000U, 0xff800000U, 0x7f812345U,
    };
    constexpr std::array<std::uint16_t, kFloatInputs.size()> kRoundedExpected = {
        0x0000U, 0x8000U, 0x3f80U, 0x3f80U,
        0x3f82U, 0xbf80U, 0x0000U, 0x0000U,
        0x0001U, 0x7f80U, 0xff80U, 0x7fc1U,
    };

    std::uint16_t* d_bf16 = nullptr;
    std::uint32_t* d_expanded = nullptr;
    std::uint32_t* d_float = nullptr;
    std::uint16_t* d_rounded = nullptr;

    if (!cuda_ok(cudaMalloc(&d_bf16, sizeof(kBf16Inputs)), "cudaMalloc bf16") ||
        !cuda_ok(cudaMalloc(&d_expanded, sizeof(kExpandedExpected)), "cudaMalloc expanded") ||
        !cuda_ok(cudaMalloc(&d_float, sizeof(kFloatInputs)), "cudaMalloc float bits") ||
        !cuda_ok(cudaMalloc(&d_rounded, sizeof(kRoundedExpected)), "cudaMalloc rounded")) {
        cudaFree(d_bf16);
        cudaFree(d_expanded);
        cudaFree(d_float);
        cudaFree(d_rounded);
        return 1;
    }

    int failures = 0;
    if (!cuda_ok(cudaMemcpy(d_bf16, kBf16Inputs.data(), sizeof(kBf16Inputs), cudaMemcpyHostToDevice),
                 "copy bf16 inputs") ||
        !cuda_ok(cudaMemcpy(d_float, kFloatInputs.data(), sizeof(kFloatInputs), cudaMemcpyHostToDevice),
                 "copy float inputs")) {
        failures = 1;
    } else {
        constexpr int kBf16Count = static_cast<int>(kBf16Inputs.size());
        constexpr int kFloatCount = static_cast<int>(kFloatInputs.size());
        bf16_conversion_kernel<<<1, 32>>>(d_bf16, d_expanded, kBf16Count, d_float, d_rounded,
                                          kFloatCount);
        if (!cuda_ok(cudaGetLastError(), "launch bf16_conversion_kernel") ||
            !cuda_ok(cudaDeviceSynchronize(), "sync bf16_conversion_kernel")) {
            failures = 1;
        }
    }

    std::array<std::uint32_t, kExpandedExpected.size()> expanded{};
    std::array<std::uint16_t, kRoundedExpected.size()> rounded{};
    if (failures == 0) {
        if (!cuda_ok(cudaMemcpy(expanded.data(), d_expanded, sizeof(expanded), cudaMemcpyDeviceToHost),
                     "copy expanded") ||
            !cuda_ok(cudaMemcpy(rounded.data(), d_rounded, sizeof(rounded), cudaMemcpyDeviceToHost),
                     "copy rounded")) {
            failures = 1;
        }
    }

    cudaFree(d_bf16);
    cudaFree(d_expanded);
    cudaFree(d_float);
    cudaFree(d_rounded);

    if (failures != 0) { return failures; }
    if (expanded != kExpandedExpected) {
        std::cerr << "BF16 raw-bit expansion mismatch\n";
        ++failures;
    }
    if (rounded != kRoundedExpected) {
        std::cerr << "FP32->BF16 RNE mismatch\n";
        ++failures;
    }
    return failures;
}

std::uint32_t pack_halfwords(std::uint16_t lo, std::uint16_t hi) {
    return static_cast<std::uint32_t>(lo) | (static_cast<std::uint32_t>(hi) << 16);
}

int test_macro16_fragment_mapping() {
    std::array<std::uint16_t, 16 * 4> host_a{};
    std::array<std::uint16_t, 16 * 4> host_b{};
    std::array<FragmentWords, 32> actual{};

    for (int row = 0; row < 16; ++row) {
        for (int k = 0; k < 4; ++k) {
            host_a[static_cast<std::size_t>(row) * 4 + k] =
                static_cast<std::uint16_t>(0x1000U + row * 0x10U + k);
        }
    }
    for (int col = 0; col < 16; ++col) {
        for (int k = 0; k < 4; ++k) {
            host_b[static_cast<std::size_t>(col) * 4 + k] =
                static_cast<std::uint16_t>(0x2000U + col * 0x10U + k);
        }
    }

    __half* d_a = nullptr;
    __half* d_b = nullptr;
    FragmentWords* d_words = nullptr;
    if (!cuda_ok(cudaMalloc(&d_a, sizeof(host_a)), "cudaMalloc fragment A") ||
        !cuda_ok(cudaMalloc(&d_b, sizeof(host_b)), "cudaMalloc fragment B") ||
        !cuda_ok(cudaMalloc(&d_words, sizeof(actual)), "cudaMalloc fragment words")) {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_words);
        return 1;
    }

    int failures = 0;
    if (!cuda_ok(cudaMemcpy(d_a, host_a.data(), sizeof(host_a), cudaMemcpyHostToDevice),
                 "copy fragment A") ||
        !cuda_ok(cudaMemcpy(d_b, host_b.data(), sizeof(host_b), cudaMemcpyHostToDevice),
                 "copy fragment B")) {
        failures = 1;
    } else {
        macro16_fragment_kernel<<<1, 32>>>(d_a, d_b, d_words);
        if (!cuda_ok(cudaGetLastError(), "launch macro16_fragment_kernel") ||
            !cuda_ok(cudaDeviceSynchronize(), "sync macro16_fragment_kernel") ||
            !cuda_ok(cudaMemcpy(actual.data(), d_words, sizeof(actual), cudaMemcpyDeviceToHost),
                     "copy fragment words")) {
            failures = 1;
        }
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_words);
    if (failures != 0) { return failures; }

    for (unsigned lane = 0; lane < 32; ++lane) {
        const int group = static_cast<int>((lane & 0x0fU) >> 2);
        const int row_or_col = static_cast<int>(lane & 0x03U) + ((lane & 0x10U) != 0U ? 4 : 0);
        const auto macro = volta::mma_macro16_group_coord(group);
        const int row = macro.row + row_or_col;
        const int col = macro.col + row_or_col;

        const auto a_at = [&](int k) {
            return host_a[static_cast<std::size_t>(row) * 4 + k];
        };
        const auto b_at = [&](int k) {
            return host_b[static_cast<std::size_t>(col) * 4 + k];
        };
        const FragmentWords expected{
            pack_halfwords(a_at(0), a_at(1)),
            pack_halfwords(a_at(2), a_at(3)),
            pack_halfwords(b_at(0), b_at(1)),
            pack_halfwords(b_at(2), b_at(3)),
        };
        const FragmentWords got = actual[lane];
        if (got.a0 != expected.a0 || got.a1 != expected.a1 || got.b0 != expected.b0 ||
            got.b1 != expected.b1) {
            std::cerr << "m8n8k4 fragment mismatch lane=" << lane << " group=" << group
                      << " row=" << row << " col=" << col << "\n";
            return 1;
        }
    }
    return 0;
}

int test_macro16_basis_mapping() {
    std::array<__half, 16 * 4> host_a{};
    std::array<__half, 16 * 4> host_b{};
    std::array<float, 16 * 16> actual{};

    __half* d_a = nullptr;
    __half* d_b = nullptr;
    float* d_output = nullptr;
    if (!cuda_ok(cudaMalloc(&d_a, sizeof(host_a)), "cudaMalloc macro A") ||
        !cuda_ok(cudaMalloc(&d_b, sizeof(host_b)), "cudaMalloc macro B") ||
        !cuda_ok(cudaMalloc(&d_output, sizeof(actual)), "cudaMalloc macro output")) {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_output);
        return 1;
    }

    int failures = 0;
    for (int target_row = 0; target_row < 16 && failures == 0; ++target_row) {
        for (int target_col = 0; target_col < 16 && failures == 0; ++target_col) {
            host_a.fill(__float2half_rn(0.0F));
            host_b.fill(__float2half_rn(0.0F));
            const int k = (target_row + target_col) & 3;
            const float a_value = static_cast<float>(1 << k);
            const float b_value = 0.5F;
            host_a[static_cast<std::size_t>(target_row) * 4 + k] = __float2half_rn(a_value);
            host_b[static_cast<std::size_t>(target_col) * 4 + k] = __float2half_rn(b_value);

            if (!cuda_ok(cudaMemcpy(d_a, host_a.data(), sizeof(host_a), cudaMemcpyHostToDevice),
                         "copy macro A") ||
                !cuda_ok(cudaMemcpy(d_b, host_b.data(), sizeof(host_b), cudaMemcpyHostToDevice),
                         "copy macro B") ||
                !cuda_ok(cudaMemset(d_output, 0, sizeof(actual)), "clear macro output")) {
                failures = 1;
                break;
            }

            macro16_basis_kernel<<<1, 32>>>(d_a, d_b, d_output);
            if (!cuda_ok(cudaGetLastError(), "launch macro16_basis_kernel") ||
                !cuda_ok(cudaDeviceSynchronize(), "sync macro16_basis_kernel") ||
                !cuda_ok(cudaMemcpy(actual.data(), d_output, sizeof(actual), cudaMemcpyDeviceToHost),
                         "copy macro output")) {
                failures = 1;
                break;
            }

            const std::uint32_t expected_nonzero =
                std::bit_cast<std::uint32_t>(a_value * b_value);
            for (int row = 0; row < 16 && failures == 0; ++row) {
                for (int col = 0; col < 16; ++col) {
                    const std::uint32_t got = std::bit_cast<std::uint32_t>(
                        actual[static_cast<std::size_t>(row) * 16 + col]);
                    const std::uint32_t expected =
                        row == target_row && col == target_col ? expected_nonzero : 0U;
                    if (got != expected) {
                        std::cerr << "m8n8k4 macro mapping mismatch target=(" << target_row << ','
                                  << target_col << ") observed=(" << row << ',' << col
                                  << ") got=0x" << std::hex << got << " expected=0x" << expected
                                  << std::dec << "\n";
                        failures = 1;
                        break;
                    }
                }
            }
        }
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_output);
    return failures;
}

} // namespace

int main() {
#if !defined(NINFER_VOLTA_BUILD)
    std::cerr << "GDN Volta math test built outside NINFER_VOLTA_BUILD\n";
    return 1;
#else
    int failures = 0;
    failures += test_bridge_planner();
    failures += test_decay_values();
    failures += test_solve_primitives();
    if (failures != 0) {
        std::cout << "FAIL: GDN Volta host mathematical qualification\n";
        return 1;
    }

    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::cout << "SKIP: host math passed; no CUDA device for SM70 MMA qualification\n";
        return 77;
    }

    cudaDeviceProp properties{};
    if (!cuda_ok(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties")) { return 1; }
    if (properties.major != 7 || properties.minor != 0) {
        std::cout << "SKIP: GDN Volta substrate qualification requires SM70\n";
        return 77;
    }

    failures += test_bf16_conversion();
    failures += test_macro16_fragment_mapping();
    failures += test_macro16_basis_mapping();

    std::cout << (failures == 0 ? "PASS" : "FAIL")
              << ": GDN Volta BF16 + m8n8k4 mapping qualification\n";
    return failures == 0 ? 0 : 1;
#endif
}
