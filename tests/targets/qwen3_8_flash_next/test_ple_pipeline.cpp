#include "core/arena.h"
#include "core/device.h"
#include "targets/qwen3_8_flash_next/impl/ple_pipeline.h"

#include <cuda_runtime.h>

#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

namespace {

bool cuda_unavailable(cudaError_t error) {
    return error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver;
}

float bf16_to_float(std::uint16_t value) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(value) << 16U);
}

} // namespace

int main() {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    int device_count              = 0;
    const cudaError_t count_error = cudaGetDeviceCount(&device_count);
    if (cuda_unavailable(count_error) || device_count == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    CUDA_CHECK(count_error);

    ninfer::DeviceContext device(0);

    constexpr std::uint64_t rows         = 64;
    constexpr std::uint64_t width        = 160;
    constexpr std::uint64_t code_bytes   = rows * width / 2;
    constexpr std::uint64_t scale_offset = (code_bytes + 255) / 256 * 256;
    constexpr std::uint64_t scale_bytes  = rows * (width / 16) * 2;
    std::vector<std::byte> encoded(scale_offset + scale_bytes, std::byte{0});

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> code_dist(0, 255);
    std::uniform_int_distribution<int> scale_dist(0x3000, 0x4800); // Positive finite FP16

    for (std::size_t i = 0; i < code_bytes; ++i) {
        encoded[i] = static_cast<std::byte>(code_dist(rng));
    }
    for (std::size_t i = scale_offset; i < scale_offset + scale_bytes; i += 2) {
        std::uint16_t s = static_cast<std::uint16_t>(scale_dist(rng));
        std::memcpy(encoded.data() + i, &s, sizeof(s));
    }

    PleTableView table;
    for (PleShardView& shard : table.shards) { shard = make_ple_shard_view(encoded, rows, width); }

    // 1. Decode pinned gather test (CPU gather_pinned for decode graph support).
    // B is swept across the inline/pool threshold in gather_pinned: B<=2 gathers on the calling
    // thread (served decode is B=1 and cannot afford the two condvar hops in the ~1 ms/token
    // inter-round window), B>2 goes through the HostWorkerPool. Both must produce identical
    // pinned bytes.
    for (const std::size_t B : {std::size_t{1}, std::size_t{2}, std::size_t{3}, std::size_t{8}}) {
        PleGatherPipeline decode_pipeline(table, device, B);
        std::vector<std::array<std::int64_t, 16>> decode_indices(B);
        for (std::size_t b = 0; b < B; ++b) {
            for (std::size_t head = 0; head < 16; ++head) {
                decode_indices[b][head] = static_cast<std::int64_t>((head + b) % rows);
            }
        }
        decode_pipeline.gather_pinned(decode_indices);
        const auto* pinned_out = static_cast<const std::uint16_t*>(decode_pipeline.fixed_host_buffer());
        double decode_base_sq = 0.0;
        int decode_nan = 0;
        for (std::size_t b = 0; b < B; ++b) {
            std::array<std::uint16_t, 2560> cpu_expected{};
            gather_ple_rows_bf16(table, decode_indices[b], cpu_expected);
            for (std::size_t i = 0; i < 2560; ++i) {
                const std::uint16_t actual = pinned_out[b * 2560 + i];
                const float f = bf16_to_float(actual);
                if (!std::isfinite(f)) { decode_nan++; }
                decode_base_sq += static_cast<double>(f) * f;
                if (actual != cpu_expected[i]) {
                    std::cerr << "FAIL: Decode pinned gather mismatch at b=" << b << ", idx=" << i
                              << ": act=0x" << std::hex << actual << " exp=0x" << cpu_expected[i] << std::dec << "\n";
                    return 1;
                }
            }
        }
        if (decode_nan > 0 || decode_base_sq <= 0.0) {
            std::cerr << "FAIL: Decode pinned gather was vacuous or non-finite\n";
            return 1;
        }
        std::cout << "PASS: Decode pinned gather verified bit-exact across B=" << B << "\n";
    }

    // 2. Prefill Pipeline Bit-Exact Equivalence & Non-Vacuity across T in {1, 16, 64, 128, 512, 2048}
    const std::vector<std::size_t> test_tokens = {1, 16, 64, 128, 512, 2048};
    for (std::size_t T : test_tokens) {
        PleGatherPipeline pipeline(table, device, T);
        std::vector<std::array<std::int64_t, 16>> chunk_indices(T);
        for (std::size_t t = 0; t < T; ++t) {
            for (std::size_t head = 0; head < 16; ++head) {
                chunk_indices[t][head] = static_cast<std::int64_t>((t * 16 + head) % rows);
            }
        }

        // Host prepare timing
        auto t0 = std::chrono::high_resolution_clock::now();
        auto ticket = pipeline.prepare(chunk_indices);
        auto t1 = std::chrono::high_resolution_clock::now();
        double host_prep_us = std::chrono::duration<double, std::micro>(t1 - t0).count();

        ninfer::DeviceBuffer dev_output(T * 2560 * sizeof(std::uint16_t));
        ninfer::Tensor out_tensor(dev_output.p, ninfer::DType::BF16, {2560, static_cast<std::int32_t>(T)});

        pipeline.enqueue_copy(std::move(ticket), out_tensor);
        device.synchronize();

        std::vector<std::uint16_t> gpu_actual(T * 2560);
        dev_output.copy_to_host(gpu_actual.data(), gpu_actual.size() * sizeof(std::uint16_t));

        // CPU Oracle
        double base_sq = 0.0;
        int nan_count = 0;
        int bit_mismatches = 0;

        for (std::size_t t = 0; t < T; ++t) {
            std::array<std::uint16_t, 2560> cpu_expected{};
            gather_ple_rows_bf16(table, chunk_indices[t], cpu_expected);
            for (std::size_t i = 0; i < 2560; ++i) {
                const std::size_t idx = t * 2560 + i;
                const std::uint16_t act = gpu_actual[idx];
                const std::uint16_t exp = cpu_expected[i];
                const float f_act = bf16_to_float(act);
                if (!std::isfinite(f_act)) { nan_count++; }
                base_sq += static_cast<double>(f_act) * f_act;
                if (act != exp) {
                    if (bit_mismatches < 5) {
                        std::cerr << "Mismatch at T=" << T << " (t=" << t << ", i=" << i << "): act=0x"
                                  << std::hex << act << " exp=0x" << exp << std::dec << "\n";
                    }
                    bit_mismatches++;
                }
            }
        }

        if (nan_count > 0) {
            std::cerr << "FAIL: T=" << T << " output had " << nan_count << " non-finite elements\n";
            return 1;
        }
        if (base_sq <= 0.0) {
            std::cerr << "FAIL: T=" << T << " comparison was vacuous (base_sq <= 0)\n";
            return 1;
        }
        if (bit_mismatches > 0) {
            std::cerr << "FAIL: T=" << T << " had " << bit_mismatches << " bitwise mismatches against CPU\n";
            return 1;
        }

        std::cout << "PASS T=" << T << ": 100% BIT-EXACT across " << (T * 2560)
                  << " elements (base_sq=" << base_sq << ", host_prep=" << host_prep_us << " us)\n";
    }

    std::cout << "PASS: test_ple_pipeline\n";
    return 0;
}

