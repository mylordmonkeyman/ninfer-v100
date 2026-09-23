#include "ops/common/volta_memory.cuh"
#include "ops/common/volta_mma.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

constexpr int kGroups = 4;
constexpr int kM = 8;
constexpr int kN = 8;
constexpr int kK = 4;
constexpr int kMatrixInputElements = kGroups * kM * kK;
constexpr int kOutputElements = kGroups * kM * kN;

struct alignas(16) SharedTiles {
    __half a[kMatrixInputElements];
    __half b[kMatrixInputElements]; // column-major [group][N][K]
    std::uint8_t zfill[16];
};

__global__ void volta_substrate_kernel(const __half* a, const __half* b,
                                       const std::uint8_t* zfill_source,
                                       float* output, std::uint8_t* zfill_output) {
    __shared__ SharedTiles shared;
    const unsigned lane = threadIdx.x & 31U;

    if (lane < 16U) {
        auto* dst = reinterpret_cast<std::uint8_t*>(shared.a) + lane * 16U;
        const auto* src = reinterpret_cast<const std::uint8_t*>(a) + lane * 16U;
        ninfer::ops::volta_stage_copy<16>(dst, src);
    } else {
        const unsigned index = lane - 16U;
        auto* dst = reinterpret_cast<std::uint8_t*>(shared.b) + index * 16U;
        const auto* src = reinterpret_cast<const std::uint8_t*>(b) + index * 16U;
        ninfer::ops::volta_stage_copy<16>(dst, src);
    }

    if (lane == 0U) {
        ninfer::ops::volta_stage_copy_zfill<16>(shared.zfill, zfill_source, 5);
    }
    ninfer::ops::volta_stage_barrier();

    if (lane < 16U) { zfill_output[lane] = shared.zfill[lane]; }

    const int group = ninfer::ops::volta_mma884_group(lane);
    const __half* a_tile = shared.a + group * kM * kK;
    const __half* b_tile = shared.b + group * kN * kK;

    const auto a_fragment = ninfer::ops::volta_mma884_load_a_row(a_tile, kK);
    const auto b_fragment = ninfer::ops::volta_mma884_load_b_col(b_tile, kK);
    ninfer::ops::VoltaMma884Accumulator accumulator{};
    accumulator.clear();
    ninfer::ops::volta_mma884_f16_f32(accumulator, a_fragment, b_fragment);

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const auto coordinate =
            ninfer::ops::volta_mma884_accumulator_coordinate(lane, i);
        output[group * kM * kN + coordinate.row * kN + coordinate.col] =
            accumulator.x[i];
    }
}

bool cuda_ok(cudaError_t status, const char* what) {
    if (status == cudaSuccess) { return true; }
    std::cerr << what << ": " << cudaGetErrorString(status) << "\n";
    return false;
}

float a_value(int group, int row, int k) {
    return static_cast<float>((group + 1) * 2 + row - k - 4) * 0.25F;
}

float b_value(int group, int k, int col) {
    return static_cast<float>(group + col - 2 * k - 3) * 0.125F;
}

} // namespace

int main() {
#if !defined(NINFER_VOLTA_BUILD)
    std::cerr << "volta substrate test built outside NINFER_VOLTA_BUILD\n";
    return 1;
#else
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        return 77;
    }

    cudaDeviceProp properties{};
    if (!cuda_ok(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties")) {
        return 1;
    }
    if (properties.major != 7) {
        std::cout << "SKIP: Volta substrate qualification requires a compute-7.x device\n";
        return 77;
    }

    std::array<__half, kMatrixInputElements> host_a{};
    std::array<__half, kMatrixInputElements> host_b{};
    std::array<float, kOutputElements> expected{};
    std::array<std::uint8_t, 16> zfill_source{};
    std::array<std::uint8_t, 16> zfill_expected{};

    for (int group = 0; group < kGroups; ++group) {
        for (int row = 0; row < kM; ++row) {
            for (int k = 0; k < kK; ++k) {
                host_a[group * kM * kK + row * kK + k] =
                    __float2half_rn(a_value(group, row, k));
            }
        }
        for (int col = 0; col < kN; ++col) {
            for (int k = 0; k < kK; ++k) {
                host_b[group * kN * kK + col * kK + k] =
                    __float2half_rn(b_value(group, k, col));
            }
        }
        for (int row = 0; row < kM; ++row) {
            for (int col = 0; col < kN; ++col) {
                float sum = 0.0F;
                for (int k = 0; k < kK; ++k) {
                    const float av = __half2float(host_a[group * kM * kK + row * kK + k]);
                    const float bv = __half2float(host_b[group * kN * kK + col * kK + k]);
                    sum += av * bv;
                }
                expected[group * kM * kN + row * kN + col] = sum;
            }
        }
    }
    for (int i = 0; i < 16; ++i) {
        zfill_source[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(0xA0 + i);
        zfill_expected[static_cast<std::size_t>(i)] =
            i < 5 ? zfill_source[static_cast<std::size_t>(i)] : std::uint8_t{0};
    }

    __half* device_a = nullptr;
    __half* device_b = nullptr;
    float* device_output = nullptr;
    std::uint8_t* device_zfill_source = nullptr;
    std::uint8_t* device_zfill_output = nullptr;

    const auto cleanup = [&] {
        cudaFree(device_a);
        cudaFree(device_b);
        cudaFree(device_output);
        cudaFree(device_zfill_source);
        cudaFree(device_zfill_output);
    };

    if (!cuda_ok(cudaMalloc(&device_a, sizeof(host_a)), "cudaMalloc A") ||
        !cuda_ok(cudaMalloc(&device_b, sizeof(host_b)), "cudaMalloc B") ||
        !cuda_ok(cudaMalloc(&device_output, sizeof(expected)), "cudaMalloc output") ||
        !cuda_ok(cudaMalloc(&device_zfill_source, sizeof(zfill_source)), "cudaMalloc zfill source") ||
        !cuda_ok(cudaMalloc(&device_zfill_output, sizeof(zfill_expected)), "cudaMalloc zfill output")) {
        cleanup();
        return 1;
    }

    if (!cuda_ok(cudaMemcpy(device_a, host_a.data(), sizeof(host_a), cudaMemcpyHostToDevice),
                 "copy A") ||
        !cuda_ok(cudaMemcpy(device_b, host_b.data(), sizeof(host_b), cudaMemcpyHostToDevice),
                 "copy B") ||
        !cuda_ok(cudaMemcpy(device_zfill_source, zfill_source.data(), sizeof(zfill_source),
                            cudaMemcpyHostToDevice),
                 "copy zfill source")) {
        cleanup();
        return 1;
    }

    volta_substrate_kernel<<<1, 32>>>(
        device_a, device_b, device_zfill_source, device_output, device_zfill_output);
    if (!cuda_ok(cudaGetLastError(), "launch volta_substrate_kernel") ||
        !cuda_ok(cudaDeviceSynchronize(), "synchronize volta_substrate_kernel")) {
        cleanup();
        return 1;
    }

    std::array<float, kOutputElements> actual{};
    std::array<std::uint8_t, 16> zfill_actual{};
    if (!cuda_ok(cudaMemcpy(actual.data(), device_output, sizeof(actual), cudaMemcpyDeviceToHost),
                 "copy output") ||
        !cuda_ok(cudaMemcpy(zfill_actual.data(), device_zfill_output, sizeof(zfill_actual),
                            cudaMemcpyDeviceToHost),
                 "copy zfill output")) {
        cleanup();
        return 1;
    }
    cleanup();

    float max_abs = 0.0F;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        max_abs = std::max(max_abs, std::abs(actual[i] - expected[i]));
    }
    if (max_abs > 1.0e-3F) {
        std::cerr << "Volta m8n8k4 mismatch: max_abs=" << max_abs << "\n";
        return 1;
    }
    if (zfill_actual != zfill_expected) {
        std::cerr << "Volta synchronous zfill staging mismatch\n";
        return 1;
    }

    std::cout << "PASS: Volta synchronous staging + m8n8k4 FP32 accumulation"
              << " max_abs=" << max_abs << "\n";
    return 0;
#endif
}
