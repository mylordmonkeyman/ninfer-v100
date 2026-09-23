#include "ops/linear/nvfp4/nvfp4_codec.cuh"

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <iostream>

namespace {

__global__ void qualify_nvfp4_codec_kernel(std::uint8_t* e2_roundtrip,
                                            std::uint8_t* e4_roundtrip,
                                            std::uint8_t* packed16) {
#if defined(NINFER_VOLTA_BUILD)
    const int index = static_cast<int>(threadIdx.x);
    if (index < 16) {
        const auto code = static_cast<std::uint8_t>(index);
        const float value = ninfer::ops::detail::decode_nvfp4_e2m1_scalar(code);
        e2_roundtrip[index] = ninfer::ops::detail::encode_nvfp4_e2m1(value);
    }
    if (index < 256) {
        const auto code = static_cast<std::uint8_t>(index);
        const float value = ninfer::ops::detail::decode_nvfp4_e4m3(code);
        e4_roundtrip[index] = ninfer::ops::detail::encode_nvfp4_e4m3_satfinite(value);
    }
    if (index == 0) {
        float2 values[8];
#pragma unroll
        for (int pair = 0; pair < 8; ++pair) {
            const auto lo = static_cast<std::uint8_t>(2 * pair);
            const auto hi = static_cast<std::uint8_t>(2 * pair + 1);
            values[pair] = make_float2(
                ninfer::ops::detail::decode_nvfp4_e2m1_scalar(lo),
                ninfer::ops::detail::decode_nvfp4_e2m1_scalar(hi));
        }
        std::uint32_t lo = 0;
        std::uint32_t hi = 0;
        ninfer::ops::detail::pack_nvfp4_e2m1x16(values, lo, hi);
        reinterpret_cast<std::uint32_t*>(packed16)[0] = lo;
        reinterpret_cast<std::uint32_t*>(packed16)[1] = hi;
    }
#else
    (void)e2_roundtrip;
    (void)e4_roundtrip;
    (void)packed16;
#endif
}

bool cuda_ok(cudaError_t status, const char* what) {
    if (status == cudaSuccess) { return true; }
    std::cerr << what << ": " << cudaGetErrorString(status) << "\n";
    return false;
}

} // namespace

int main() {
#if !defined(NINFER_VOLTA_BUILD)
    std::cerr << "Volta NVFP4 codec test built outside NINFER_VOLTA_BUILD\n";
    return 1;
#else
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) { return 77; }

    cudaDeviceProp props{};
    if (!cuda_ok(cudaGetDeviceProperties(&props, 0), "cudaGetDeviceProperties")) { return 1; }
    if (props.major != 7) {
        std::cout << "SKIP: Volta NVFP4 codec qualification requires compute 7.x\n";
        return 77;
    }

    std::uint8_t* d_e2 = nullptr;
    std::uint8_t* d_e4 = nullptr;
    std::uint8_t* d_pack = nullptr;
    if (!cuda_ok(cudaMalloc(&d_e2, 16), "cudaMalloc e2") ||
        !cuda_ok(cudaMalloc(&d_e4, 256), "cudaMalloc e4") ||
        !cuda_ok(cudaMalloc(&d_pack, 8), "cudaMalloc pack")) {
        cudaFree(d_e2);
        cudaFree(d_e4);
        cudaFree(d_pack);
        return 1;
    }

    qualify_nvfp4_codec_kernel<<<1, 256>>>(d_e2, d_e4, d_pack);
    if (!cuda_ok(cudaGetLastError(), "launch codec qualification") ||
        !cuda_ok(cudaDeviceSynchronize(), "synchronize codec qualification")) {
        cudaFree(d_e2);
        cudaFree(d_e4);
        cudaFree(d_pack);
        return 1;
    }

    std::array<std::uint8_t, 16> e2{};
    std::array<std::uint8_t, 256> e4{};
    std::array<std::uint8_t, 8> packed{};
    const bool copies =
        cuda_ok(cudaMemcpy(e2.data(), d_e2, e2.size(), cudaMemcpyDeviceToHost), "copy e2") &&
        cuda_ok(cudaMemcpy(e4.data(), d_e4, e4.size(), cudaMemcpyDeviceToHost), "copy e4") &&
        cuda_ok(cudaMemcpy(packed.data(), d_pack, packed.size(), cudaMemcpyDeviceToHost),
                "copy packed");
    cudaFree(d_e2);
    cudaFree(d_e4);
    cudaFree(d_pack);
    if (!copies) { return 1; }

    for (int code = 0; code < 16; ++code) {
        if (e2[static_cast<std::size_t>(code)] != static_cast<std::uint8_t>(code)) {
            std::cerr << "E2M1 roundtrip mismatch code=" << code
                      << " got=" << static_cast<int>(e2[static_cast<std::size_t>(code)]) << "\n";
            return 1;
        }
    }

    for (int code = 0; code < 256; ++code) {
        const int magnitude_code = code & 0x7F;
        if (magnitude_code == 0x7F) { continue; } // E4M3FN NaN encoding.
        if (e4[static_cast<std::size_t>(code)] != static_cast<std::uint8_t>(code)) {
            std::cerr << "E4M3FN roundtrip mismatch code=" << code
                      << " got=" << static_cast<int>(e4[static_cast<std::size_t>(code)]) << "\n";
            return 1;
        }
    }

    for (int byte = 0; byte < 8; ++byte) {
        const std::uint8_t expected =
            static_cast<std::uint8_t>((2 * byte) | ((2 * byte + 1) << 4));
        if (packed[static_cast<std::size_t>(byte)] != expected) {
            std::cerr << "E2M1 pack mismatch byte=" << byte
                      << " got=" << static_cast<int>(packed[static_cast<std::size_t>(byte)])
                      << " expected=" << static_cast<int>(expected) << "\n";
            return 1;
        }
    }

    std::cout << "PASS: Volta software E2M1/E4M3FN codec qualification\n";
    return 0;
#endif
}
