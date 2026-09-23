#include "core/arena.h"
#include "core/device.h"
#include "ninfer/ops/linear.h"
#include "targets/qwen3_8_flash_next/impl/load/quantize_output_head.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

namespace {

bool cuda_unavailable(cudaError_t error) {
    return error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver;
}

std::uint16_t float_to_bf16(float value) {
    const __nv_bfloat16 raw = __float2bfloat16_rn(value);
    return *reinterpret_cast<const std::uint16_t*>(&raw);
}

float bf16_to_float(std::uint16_t bits) {
    return __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&bits));
}

ninfer::Weight make_bf16_head(void* data) {
    constexpr std::int32_t rows    = 248'320;
    constexpr std::int32_t columns = 2'560;
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

void fill_patterned_bf16(ninfer::DeviceBuffer& buffer, std::uint32_t seed) {
    constexpr std::int32_t rows    = 248'320;
    constexpr std::int32_t columns = 2'560;
    std::vector<std::uint16_t> host(static_cast<std::size_t>(rows) * columns);
    for (std::size_t i = 0; i < host.size(); ++i) {
        std::uint32_t x = static_cast<std::uint32_t>(i) * 0x9E3779B9u ^ seed;
        x ^= x >> 16;
        x *= 0x7feb352du;
        x ^= x >> 15;
        const float value =
            static_cast<float>(static_cast<std::int32_t>(x)) * (1.0F / 2147483648.0F);
        host[i] = float_to_bf16(value);
    }
    buffer.copy_from_host(host.data(), host.size() * sizeof(std::uint16_t));
}

// Deterministic BF16 pattern with a per-row magnitude sweep: row 0 is all zeros (its scale must
// come out 0), and the other rows span 2^-12..2^6 so rows land in the E4M3 subnormal region as
// well as the normal one. The row amax is what the kernel has to reduce, so the pattern must not
// be row-uniform.
std::vector<std::uint16_t> patterned_rows(std::int32_t rows, std::int32_t cols, std::uint32_t seed) {
    std::vector<std::uint16_t> host(static_cast<std::size_t>(rows) * cols);
    for (std::int32_t row = 0; row < rows; ++row) {
        const float magnitude = row == 0 ? 0.0F : std::ldexp(1.0F, (row % 19) - 12);
        for (std::int32_t col = 0; col < cols; ++col) {
            std::uint32_t x = (static_cast<std::uint32_t>(row) * 0x9E3779B9u +
                               static_cast<std::uint32_t>(col)) ^
                              seed;
            x ^= x >> 16;
            x *= 0x7feb352du;
            x ^= x >> 15;
            const float unit =
                static_cast<float>(static_cast<std::int32_t>(x)) * (1.0F / 2147483648.0F);
            host[static_cast<std::size_t>(row) * cols + col] = float_to_bf16(unit * magnitude);
        }
    }
    return host;
}

// Quantizes [rows, cols] and compares codes and scales bit-for-bit against a host reference.
// `host_source` exercises the staged H2D path used when the BF16 weight stays in the artifact's
// file mapping; rows=14'000 at cols=2'560 crosses the 64 MiB staging chunk, which is what proves
// the per-chunk code/scale pointer offsets.
bool check_row_quantizer(std::int32_t rows, std::int32_t cols, bool host_source,
                         cudaStream_t stream) {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    const std::vector<std::uint16_t> host = patterned_rows(rows, cols, 0xA5A5A5A5u);
    const std::size_t payload_bytes = flash_next_fp8_head_payload_bytes(rows, cols);
    ninfer::DeviceBuffer payload(payload_bytes);
    ninfer::DeviceBuffer bf16_device;
    ninfer::Weight fp8{};
    if (host_source) {
        quantize_bf16_rows_to_fp8_e4m3_row_f32s(host.data(), payload, fp8, rows, cols, stream);
    } else {
        bf16_device = ninfer::DeviceBuffer(host.size() * sizeof(std::uint16_t));
        bf16_device.copy_from_host(host.data(), host.size() * sizeof(std::uint16_t));
        // DeviceBuffer::copy_from_host is a cudaMemcpy on the legacy default stream, but
        // DeviceContext::stream is created with cudaStreamNonBlocking, so the two are NOT ordered
        // against each other. A pageable H2D cudaMemcpy only returns once the source has been
        // staged - the DMA into device memory can still be in flight - so without this sync the
        // kernel reads whatever the fresh allocation held (zeros) instead of the pattern. Under
        // ~64 KiB the driver's staging completes before the launch and the race is invisible;
        // 64x2560 BF16 is 320 KiB and it is not. Same guard the 248'320-row arm below uses.
        CUDA_CHECK(cudaDeviceSynchronize());
        quantize_bf16_rows_to_fp8_e4m3_row_f32s(bf16_device.p, payload, fp8, rows, cols, stream);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    if (fp8.qtype != ninfer::QType::FP8_E4M3FN_ROW_F32S || fp8.n != rows || fp8.k != cols ||
        fp8.scale_dtype != ninfer::DType::FP32) {
        std::cerr << "FAIL: quantized view is not FP8_E4M3FN_ROW_F32S [" << rows << "," << cols
                  << "]\n";
        return false;
    }

    std::vector<std::uint8_t> device_payload(payload_bytes);
    payload.copy_to_host(device_payload.data(), payload_bytes);
    const std::size_t scale_offset =
        (static_cast<std::size_t>(rows) * cols + 255U) & ~std::size_t{255U};
    const auto* device_scales =
        reinterpret_cast<const float*>(device_payload.data() + scale_offset);

    std::size_t code_mismatches  = 0;
    std::size_t scale_mismatches = 0;
    for (std::int32_t row = 0; row < rows; ++row) {
        float amax = 0.0F;
        for (std::int32_t col = 0; col < cols; ++col) {
            amax = std::fmax(amax, std::fabs(bf16_to_float(
                                       host[static_cast<std::size_t>(row) * cols + col])));
        }
        const float scale   = amax > 0.0F ? amax / 448.0F : 0.0F;
        const float inverse = scale > 0.0F ? 1.0F / scale : 0.0F;
        if (device_scales[row] != scale) {
            if (scale_mismatches == 0) {
                std::cerr << "  first scale mismatch row=" << row << " device=" << device_scales[row]
                          << " expected=" << scale << " amax=" << amax << "\n";
            }
            ++scale_mismatches;
        }
        for (std::int32_t col = 0; col < cols; ++col) {
            const float value =
                bf16_to_float(host[static_cast<std::size_t>(row) * cols + col]) * inverse;
            const auto expected = static_cast<std::uint8_t>(
                __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3));
            if (device_payload[static_cast<std::size_t>(row) * cols + col] != expected) {
                ++code_mismatches;
            }
        }
    }
    std::cout << "K-sweep rows=" << rows << " cols=" << cols
              << (host_source ? " source=host" : " source=device")
              << " code_mismatches=" << code_mismatches
              << " scale_mismatches=" << scale_mismatches << "\n";
    if (code_mismatches != 0 || scale_mismatches != 0) {
        std::cerr << "FAIL: quantize_row_kernel disagrees with the host reference at cols=" << cols
                  << "\n";
        return false;
    }
    return true;
}

int argmax_column(const std::vector<std::uint16_t>& logits, std::int32_t rows, std::int32_t token) {
    int best      = 0;
    float best_v  = bf16_to_float(logits[static_cast<std::size_t>(token) * rows]);
    for (std::int32_t row = 1; row < rows; ++row) {
        const float v = bf16_to_float(logits[static_cast<std::size_t>(token) * rows + row]);
        if (v > best_v) {
            best_v = v;
            best   = row;
        }
    }
    return best;
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

    // Step 2 of the FP8-residue plan: the row quantizer must be correct for every width the later
    // steps reuse it at (mix_up K=320, shared expert K=640, head K=2'560, mix_down K=10'240), from
    // a device source and from a host (file-mapped) source.
    struct QuantCase {
        std::int32_t rows;
        std::int32_t cols;
        bool host_source;
    };
    for (const QuantCase& item : {QuantCase{48, 320, false},   QuantCase{48, 320, true},
                                  QuantCase{48, 640, false},   QuantCase{48, 640, true},
                                  QuantCase{64, 2'560, false}, QuantCase{14'000, 2'560, true},
                                  QuantCase{48, 10'240, false}, QuantCase{48, 10'240, true}}) {
        if (!check_row_quantizer(item.rows, item.cols, item.host_source, device.stream)) {
            return 1;
        }
    }

    constexpr std::int32_t rows    = 248'320;
    constexpr std::int32_t columns = 2'560;
    constexpr std::int32_t tokens  = 8;
    ninfer::DeviceBuffer bf16_storage(static_cast<std::size_t>(rows) * columns * 2);
    fill_patterned_bf16(bf16_storage, 20260901U);
    CUDA_CHECK(cudaDeviceSynchronize());
    ninfer::Weight bf16_head = make_bf16_head(bf16_storage.p);

    ninfer::DeviceBuffer fp8_storage(flash_next_fp8_output_head_payload_bytes());
    ninfer::Weight fp8_head{};
    quantize_bf16_output_head_to_fp8_e4m3_row_f32s(bf16_head, fp8_storage, fp8_head, device.stream);
    CUDA_CHECK(cudaDeviceSynchronize());
    if (fp8_head.qtype != ninfer::QType::FP8_E4M3FN_ROW_F32S || fp8_head.n != rows ||
        fp8_head.k != columns || fp8_head.scale_dtype != ninfer::DType::FP32) {
        std::cerr << "FAIL: quantized output head is not FP8_E4M3FN_ROW_F32S [248320,2560]\n";
        return 1;
    }

    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-0.5F, 0.5F);
    constexpr int kBatches          = 512;
    constexpr int kColumns          = kBatches * tokens;
    std::int32_t flips              = 0;
    double logit_diff_sq            = 0.0;
    double logit_base_sq            = 0.0;
    int nonfinite                   = 0;
    ninfer::DeviceBuffer hidden_dev(static_cast<std::size_t>(columns) * tokens * 2);
    ninfer::DeviceBuffer bf16_logits_dev(static_cast<std::size_t>(rows) * tokens * 2);
    ninfer::DeviceBuffer fp8_logits_dev(static_cast<std::size_t>(rows) * tokens * 2);
    ninfer::WorkspaceArena workspace(256);
    std::vector<std::uint16_t> hidden(static_cast<std::size_t>(columns) * tokens);
    std::vector<std::uint16_t> bf16_logits(static_cast<std::size_t>(rows) * tokens);
    std::vector<std::uint16_t> fp8_logits(static_cast<std::size_t>(rows) * tokens);

    for (int batch = 0; batch < kBatches; ++batch) {
        for (auto& v : hidden) { v = float_to_bf16(dist(rng)); }
        hidden_dev.copy_from_host(hidden.data(), hidden.size() * sizeof(std::uint16_t));
        CUDA_CHECK(cudaDeviceSynchronize());
        ninfer::Tensor hidden_view(hidden_dev.p, ninfer::DType::BF16, {columns, tokens});
        ninfer::Tensor bf16_out(bf16_logits_dev.p, ninfer::DType::BF16, {rows, tokens});
        ninfer::Tensor fp8_out(fp8_logits_dev.p, ninfer::DType::BF16, {rows, tokens});
        ninfer::ops::linear(hidden_view, bf16_head, bf16_out, ninfer::ops::LinearPolicy::A16Only,
                            workspace, device.stream);
        ninfer::ops::linear(hidden_view, fp8_head, fp8_out, ninfer::ops::LinearPolicy::A16Only,
                            workspace, device.stream);
        CUDA_CHECK(cudaDeviceSynchronize());
        bf16_logits_dev.copy_to_host(bf16_logits.data(), bf16_logits.size() * sizeof(std::uint16_t));
        fp8_logits_dev.copy_to_host(fp8_logits.data(), fp8_logits.size() * sizeof(std::uint16_t));
        for (std::int32_t t = 0; t < tokens; ++t) {
            if (argmax_column(bf16_logits, rows, t) != argmax_column(fp8_logits, rows, t)) {
                ++flips;
            }
            for (std::int32_t row = 0; row < rows; ++row) {
                const float a =
                    bf16_to_float(bf16_logits[static_cast<std::size_t>(t) * rows + row]);
                const float b = bf16_to_float(fp8_logits[static_cast<std::size_t>(t) * rows + row]);
                if (!std::isfinite(a) || !std::isfinite(b)) { ++nonfinite; }
                logit_base_sq += static_cast<double>(a) * a;
                const double d = static_cast<double>(a) - b;
                logit_diff_sq += d * d;
            }
        }
    }

    if (nonfinite > 0 || logit_base_sq <= 0.0) {
        std::cerr << "FAIL: BF16 vs FP8 comparison was vacuous or non-finite base_sq="
                  << logit_base_sq << " nonfinite=" << nonfinite << "\n";
        return 1;
    }
    const double rel_l2 = std::sqrt(logit_diff_sq / logit_base_sq);
    std::cout << "G2 argmax samples=" << kColumns << " flips=" << flips
              << " flip_rate=" << (static_cast<double>(flips) / kColumns) << " logits_rel_L2=" << rel_l2
              << " base_sq=" << logit_base_sq << "\n";

    constexpr int kWarmup = 5;
    constexpr int kIters  = 20;
    auto time_linear      = [&](ninfer::Weight& weight, std::int32_t t) {
        ninfer::Tensor hidden_view(hidden_dev.p, ninfer::DType::BF16, {columns, t});
        ninfer::Tensor out_view(bf16_logits_dev.p, ninfer::DType::BF16, {rows, t});
        for (int i = 0; i < kWarmup; ++i) {
            ninfer::ops::linear(hidden_view, weight, out_view, ninfer::ops::LinearPolicy::A16Only,
                                workspace, device.stream);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaEvent_t start = nullptr;
        cudaEvent_t stop  = nullptr;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start, device.stream));
        for (int i = 0; i < kIters; ++i) {
            ninfer::ops::linear(hidden_view, weight, out_view, ninfer::ops::LinearPolicy::A16Only,
                                workspace, device.stream);
        }
        CUDA_CHECK(cudaEventRecord(stop, device.stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        return ms * 1000.0F / static_cast<float>(kIters);
    };

    const float us_bf16_c1 = time_linear(bf16_head, 1);
    const float us_fp8_c1  = time_linear(fp8_head, 1);
    const float us_bf16_c8 = time_linear(bf16_head, 8);
    const float us_fp8_c8  = time_linear(fp8_head, 8);
    std::cout << "G2 head kernel c=1 BF16=" << us_bf16_c1 << " us  FP8=" << us_fp8_c1 << " us\n";
    std::cout << "G2 head kernel c=8 BF16=" << us_bf16_c8 << " us  FP8=" << us_fp8_c8 << " us\n";
    std::cout << "PASS: test_output_head_fp8\n";
    return 0;
}
