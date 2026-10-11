#include "ops/linear/nvfp4/nvfp4_codec.cuh"

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>
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

// Flash-Next QSA casts represented BF16 directly to unscaled E4M3.
// Qualify every finite BF16 input against an independent nearest-code oracle.
__global__ void qualify_bf16_to_e4m3_kernel(std::uint8_t* codes) {
#if defined(NINFER_VOLTA_BUILD)
    const unsigned bits = blockIdx.x * blockDim.x + threadIdx.x;
    if (bits < 65536) {
        codes[bits] = ninfer::ops::detail::encode_nvfp4_e4m3_satfinite(
            __uint_as_float(bits << 16));
    }
#else
    (void)codes;
#endif
}

bool cuda_ok(cudaError_t status, const char* what) {
    if (status == cudaSuccess) { return true; }
    std::cerr << what << ": " << cudaGetErrorString(status) << "\n";
    return false;
}

bool qualify_qsa_fp8_conversion() {
    std::uint8_t* device_codes = nullptr;
    if (!cuda_ok(cudaMalloc(&device_codes, 65536), "allocate FP8 conversion codes")) return false;
    qualify_bf16_to_e4m3_kernel<<<256,256>>>(device_codes);
    std::vector<std::uint8_t> actual(65536);
    const bool copied = cuda_ok(cudaGetLastError(), "FP8 conversion launch") &&
        cuda_ok(cudaMemcpy(actual.data(), device_codes, actual.size(), cudaMemcpyDeviceToHost),
                "copy FP8 conversion codes");
    cudaFree(device_codes);
    if (!copied) return false;
    std::array<double,127> levels{};
    for (unsigned code=0;code<levels.size();++code) {
        const int exponent = code >> 3, mantissa = code & 7;
        levels[code] = exponent == 0 ? std::ldexp(double(mantissa),-9)
            : std::ldexp(double(8+mantissa),exponent-10);
    }
    unsigned checked=0;
    for (unsigned bits=0;bits<65536;++bits) {
        const std::uint32_t float_bits=bits<<16;
        float value; std::memcpy(&value,&float_bits,sizeof(value));
        if (!std::isfinite(value)) continue;
        const double magnitude=std::abs(double(value));
        unsigned nearest=126;
        if (magnitude<448) {
            double distance=std::numeric_limits<double>::infinity();
            for (unsigned code=0;code<levels.size();++code) {
                const double candidate=std::abs(magnitude-levels[code]);
                if (candidate<distance || (candidate==distance && !(code&1))) {
                    distance=candidate;nearest=code;
                }
            }
        }
        const unsigned expected=nearest | ((bits&0x8000)?128:0);
        if (actual[bits]!=expected) {
            std::cerr<<"QSA BF16-to-E4M3 exact oracle failed bits="<<bits
                     <<" actual="<<unsigned(actual[bits])<<" expected="<<expected<<'\n';
            return false;
        }
        ++checked;
    }
    std::cout<<"PASS: QSA unscaled E4M3 nearest-even/saturation exact oracle, finite BF16 inputs="
             <<checked<<'\n';
    return true;
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

    if (!qualify_qsa_fp8_conversion()) return 1;

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
