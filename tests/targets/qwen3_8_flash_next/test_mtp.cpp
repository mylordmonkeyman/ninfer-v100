#include "core/arena.h"
#include "core/device.h"
#include "targets/qwen3_8_flash_next/impl/expert_bank.h"
#include "targets/qwen3_8_flash_next/impl/hyper_connection.h"
#include "targets/qwen3_8_flash_next/impl/load/quantize_nvfp4_expert_bank.h"
#include "targets/qwen3_8_flash_next/impl/model_view.h"
#include "targets/qwen3_8_flash_next/impl/moe.h"
#include "targets/qwen3_8_flash_next/impl/mtp_forward.h"
#include "targets/qwen3_8_flash_next/impl/mtp_forward_kernels.h"
#include "targets/qwen3_8_flash_next/impl/qsa_attention.h"
#include "targets/qwen3_8_flash_next/impl/state_dumper.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/rmsnorm.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <vector>

namespace {

bool cuda_unavailable(cudaError_t error) {
    return error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver;
}

std::uint16_t float_to_bf16(float value) {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    const std::uint32_t lsb  = (bits >> 16U) & 1U;
    const std::uint32_t bias = 0x7FFFU + lsb;
    return static_cast<std::uint16_t>((bits + bias) >> 16U);
}

float bf16_to_float(std::uint16_t val) {
    const std::uint32_t bits = static_cast<std::uint32_t>(val) << 16U;
    float f                  = 0.0f;
    std::memcpy(&f, &bits, sizeof(float));
    return f;
}

constexpr float kE2M1LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

inline float cpu_decode_e2m1(std::uint8_t nibble) {
    return kE2M1LUT[nibble & 0x0F];
}

inline float cpu_decode_e4m3fn(std::uint8_t byte) {
    const float sign = (byte & 0x80) ? -1.0f : 1.0f;
    const int exp    = (byte >> 3) & 0x0F;
    const int frac   = byte & 0x07;
    if (exp == 0) {
        return (frac == 0) ? std::copysign(0.0f, sign) : sign * static_cast<float>(frac) / 512.0f;
    }
    if (exp == 15 && frac == 7) {
        return std::copysign(std::numeric_limits<float>::quiet_NaN(), sign);
    }
    return sign * (8.0f + static_cast<float>(frac)) * std::ldexp(1.0f, exp - 10);
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

int test_mtp_stem_oracle(ninfer::DeviceContext& device) {
    using namespace ninfer;
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    constexpr int batch = 2, dim = 2560, streams = 4;
    DeviceArena arena(64 * 1024 * 1024);
    Tensor embedding = arena.alloc(DType::BF16, {dim, batch}, 256);
    Tensor hidden = arena.alloc(DType::BF16, {dim * streams, batch}, 256);
    Tensor output = arena.alloc(DType::BF16, {dim * streams, batch}, 256);
    Tensor embedding_norm = arena.alloc(DType::BF16, {dim}, 256);
    Tensor hidden_norm = arena.alloc(DType::BF16, {dim * streams}, 256);
    Tensor projection = arena.alloc(DType::BF16, {dim, dim}, 256);
    std::vector<std::uint16_t> emb(dim * batch), hid(dim * streams * batch);
    std::vector<std::uint16_t> en(dim), hn(dim * streams), weights(dim * dim, 0);
    for (int i = 0; i < dim * batch; ++i) {
        emb[i] = float_to_bf16((static_cast<int>(i % 53) - 26) / 32.0F);
    }
    for (int i = 0; i < dim * streams * batch; ++i) {
        hid[i] = float_to_bf16((static_cast<int>(i % 71) - 35) / 32.0F + (i / dim % 4) * 0.125F);
    }
    for (int i = 0; i < dim; ++i) {
        en[i] = float_to_bf16((i % 5) * 0.0625F);
        weights[i * dim + i] = float_to_bf16(0.5F + (i % 7) * 0.125F);
    }
    for (int i = 0; i < dim * streams; ++i) {
        hn[i] = float_to_bf16((i % 9) * 0.03125F);
    }
    auto upload = [&](Tensor t, const std::vector<std::uint16_t>& v) {
        CUDA_CHECK(cudaMemcpyAsync(t.data, v.data(), v.size() * 2, cudaMemcpyHostToDevice, device.stream));
    };
    upload(embedding, emb); upload(hidden, hid); upload(embedding_norm, en);
    upload(hidden_norm, hn); upload(projection, weights);
    MtpModelView mtp{};
    mtp.embedding_norm = embedding_norm; mtp.hidden_norm = hidden_norm;
    mtp.embedding_projection = bf16_weight(projection.data, dim, dim);
    mtp.hidden_projection = mtp.embedding_projection;
    DeviceBuffer backing(8 * 1024 * 1024);
    WorkspaceArena workspace(DeviceSpan{backing.p, backing.bytes});
    flash_next_mtp_stem(mtp, embedding, hidden, workspace, output, device.stream);
    device.synchronize();
    std::vector<std::uint16_t> actual(hid.size());
    CUDA_CHECK(cudaMemcpy(actual.data(), output.data, actual.size() * 2, cudaMemcpyDeviceToHost));
    double error = 0, energy = 0, max_error = 0;
    // FP64 complete formula from represented BF16 inputs and weights, independent
    // of the GPU's intermediate materializations and reduction association.
    for (int b = 0; b < batch; ++b) {
        double emb_ss = 0, hid_ss = 0;
        for (int c = 0; c < dim; ++c) { const double x = bf16_to_float(emb[b * dim + c]); emb_ss += x*x; }
        for (int c = 0; c < dim * streams; ++c) { const double x = bf16_to_float(hid[b * dim * streams + c]); hid_ss += x*x; }
        const double er = 1.0 / std::sqrt(emb_ss / dim + 1e-6);
        const double hr = 1.0 / std::sqrt(hid_ss / (dim * streams) + 1e-6);
        for (int c = 0; c < dim * streams; ++c) {
            const int d = c % dim, idx = b * dim * streams + c;
            const double w = bf16_to_float(weights[d * dim + d]);
            const double expected = w * (bf16_to_float(emb[b * dim + d]) * er * (1 + bf16_to_float(en[d]))
                + bf16_to_float(hid[idx]) * hr * (1 + bf16_to_float(hn[c])));
            const double delta = bf16_to_float(actual[idx]) - expected;
            error += delta * delta; energy += expected * expected;
            max_error = std::max(max_error, std::abs(delta));
        }
    }
    const double relative = std::sqrt(error / energy);
    if (relative > 0.008 || max_error > 0.04) {
        std::cerr << "FAIL: MTP stem FP64 oracle relative=" << relative << " max=" << max_error << '\n';
        return 1;
    }
    std::cout << "PASS: MTP stem FP64 oracle, distinct streams and full-width norm, relative=" << relative << '\n';
    return 0;
}
int test_mtp_target_state_alignment(ninfer::DeviceContext& device) {
    using namespace ninfer;
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    constexpr int width = 10240, tokens = 2, slots = 3;
    DeviceArena arena(2 * 1024 * 1024);
    Tensor hidden = arena.alloc(DType::BF16, {width, tokens}, 256);
    Tensor saved = arena.alloc(DType::BF16, {width, slots}, 256);
    Tensor positions = arena.alloc(DType::I32, {tokens, 3}, 256);
    Tensor saved_positions = arena.alloc(DType::I32, {3, slots}, 256);
    Tensor indices = arena.alloc(DType::I32, {tokens}, 256);
    Tensor sources = arena.alloc(DType::I32, {tokens}, 256);
    Tensor destinations = arena.alloc(DType::I32, {tokens}, 256);
    std::vector<std::uint16_t> hh(width * tokens), hs(width * slots);
    for (int i = 0; i < width * tokens; ++i) { hh[i] = float_to_bf16((i % 97) / 64.0F); }
    for (int i = 0; i < width * slots; ++i) { hs[i] = float_to_bf16(-2.0F - (i % 47) / 32.0F); }
    const std::array<std::int32_t, 6> hp{101, 103, 17, 19, 5, 11};
    const std::array<std::int32_t, 9> sp{41, 2, 3, 53, 7, 13, 97, 11, 17};
    const std::array<std::int32_t, 2> ti{64, 65}, src{2, 0}, dst{0, 1};
    auto copy = [&](Tensor t, const auto& values) {
        CUDA_CHECK(cudaMemcpyAsync(t.data, values.data(), t.bytes(), cudaMemcpyHostToDevice, device.stream));
    };
    copy(hidden, hh); copy(saved, hs); copy(positions, hp); copy(saved_positions, sp);
    copy(indices, ti); copy(sources, src); copy(destinations, dst);
    for (const auto mode : {0, 1, 2}) {
        const bool prefill = mode == 0, chain = mode != 2;
        const int offset = prefill ? 1 : 0, rows = tokens - offset;
        const auto scope = arena.scope();
        Tensor shifted = arena.alloc(DType::BF16, {width, rows}, 256);
        Tensor shifted_indices = arena.alloc(DType::I32, {rows}, 256);
        Tensor shifted_positions = arena.alloc(DType::I32, {rows, 3}, 256);
        flash_next_mtp_shift_inputs_launch(hidden, indices, positions, saved, saved_positions,
            prefill ? Tensor{} : sources, 2, chain, offset, shifted,
            shifted_indices, shifted_positions, device.stream);
        device.synchronize();
        std::vector<std::uint16_t> got(width * rows);
        std::vector<std::int32_t> got_indices(rows), got_positions(rows * 3);
        CUDA_CHECK(cudaMemcpy(got.data(), shifted.data, shifted.bytes(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(got_indices.data(), shifted_indices.data, shifted_indices.bytes(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(got_positions.data(), shifted_positions.data, shifted_positions.bytes(), cudaMemcpyDeviceToHost));
        for (int row = 0; row < rows; ++row) {
            const int current = row + offset, previous = chain ? current - 1 : -1;
            const int slot = prefill ? 2 : src[current];
            if (got_indices[row] != ti[current] - 1) { return 1; }
            for (int axis = 0; axis < 3; ++axis) {
                const auto expected = previous >= 0 ? hp[axis * tokens + previous] : sp[slot * 3 + axis];
                if (got_positions[axis * rows + row] != expected) { return 1; }
            }
            for (int dim = 0; dim < width; ++dim) {
                const auto expected = previous >= 0 ? hh[previous * width + dim] : hs[slot * width + dim];
                if (got[row * width + dim] != expected) { return 1; }
            }
        }
    }
    flash_next_mtp_save_target_launch(hidden, positions, destinations, 0, saved, saved_positions, device.stream);
    device.synchronize();
    std::vector<std::uint16_t> saved_after(hs.size());
    std::array<std::int32_t, 9> positions_after{};
    CUDA_CHECK(cudaMemcpy(saved_after.data(), saved.data, saved.bytes(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(positions_after.data(), saved_positions.data, saved_positions.bytes(), cudaMemcpyDeviceToHost));
    for (int row = 0; row < tokens; ++row) {
        for (int dim = 0; dim < width; ++dim) {
            if (saved_after[dst[row] * width + dim] != hh[row * width + dim]) { return 1; }
        }
        for (int axis = 0; axis < 3; ++axis) {
            if (positions_after[dst[row] * 3 + axis] != hp[axis * tokens + row]) { return 1; }
        }
    }
    if (!std::equal(hs.begin() + 2 * width, hs.end(), saved_after.begin() + 2 * width)) { return 1; }
    std::cout << "PASS: MTP target alignment, stream carry, lane isolation and three-axis positions\n";
    return 0;
}
int test_mtp_nvfp4_quantizer_unit(ninfer::DeviceContext& device) {
    std::cout << "[TEST 2/4] test_mtp_nvfp4_quantizer_unit ...\n" << std::flush;
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    constexpr int experts = 512;
    constexpr int rows    = 128;
    constexpr int cols    = 64;

    const std::size_t total_elements = static_cast<std::size_t>(experts) * rows * cols;
    const std::size_t bf16_bytes     = total_elements * sizeof(std::uint16_t);
    const std::size_t nvfp4_bytes    = flash_next_nvfp4_expert_bank_payload_bytes(experts, rows, cols);

    std::vector<std::uint16_t> host_bf16(total_elements);
    for (std::size_t i = 0; i < total_elements; ++i) {
        float val = 0.5f * std::sin(static_cast<float>(i % 317)) +
                    0.25f * std::cos(static_cast<float>((i / 16) % 19));
        host_bf16[i] = float_to_bf16(val);
    }

    void* d_bf16 = nullptr;
    void* d_out1 = nullptr;
    void* d_out2 = nullptr;
    cudaMalloc(&d_bf16, bf16_bytes);
    cudaMalloc(&d_out1, nvfp4_bytes);
    cudaMalloc(&d_out2, nvfp4_bytes);

    cudaMemcpy(d_bf16, host_bf16.data(), bf16_bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_out1, 0, nvfp4_bytes);
    cudaMemset(d_out2, 0, nvfp4_bytes);

    // Run 1
    quantize_bf16_expert_bank_to_nvfp4(d_bf16, d_out1, experts, rows, cols, device.stream);
    device.synchronize();

    // Run 2
    quantize_bf16_expert_bank_to_nvfp4(d_bf16, d_out2, experts, rows, cols, device.stream);
    device.synchronize();

    std::vector<std::uint8_t> host_out1(nvfp4_bytes);
    std::vector<std::uint8_t> host_out2(nvfp4_bytes);
    cudaMemcpy(host_out1.data(), d_out1, nvfp4_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_out2.data(), d_out2, nvfp4_bytes, cudaMemcpyDeviceToHost);

    // 1. Two-run bitwise determinism check
    if (std::memcmp(host_out1.data(), host_out2.data(), nvfp4_bytes) != 0) {
        std::cerr << "FAIL: NVFP4 quantizer output is not bitwise deterministic across runs\n";
        cudaFree(d_bf16); cudaFree(d_out1); cudaFree(d_out2);
        return 1;
    }

    // 2. Dequantization oracle and representable error verification
    const std::uint64_t code_plane_bytes   = total_elements / 2;
    const std::uint64_t scale_plane_offset = (code_plane_bytes + 255U) & ~std::uint64_t{255U};
    const std::uint64_t scale_plane_bytes  = total_elements / 16;
    const std::uint64_t weight_div_offset  = scale_plane_offset + scale_plane_bytes;
    const auto* codes                      = host_out1.data();
    const auto* scales                     = host_out1.data() + scale_plane_offset;
    const auto* divisors = reinterpret_cast<const float*>(host_out1.data() + weight_div_offset);
    const int groups_per_row               = cols / 16;
    const int k_tiles                      = cols / 64;

    double diff_sq = 0.0;
    double base_sq = 0.0;
    for (int e = 0; e < experts; ++e) {
        const float divisor = divisors[e];
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                const std::size_t flat_idx = (static_cast<std::size_t>(e) * rows + r) * cols + c;
                const float original       = bf16_to_float(host_bf16[flat_idx]);

                const std::size_t code_off =
                    (static_cast<std::size_t>(e) * rows + r) * (cols / 2) + (c / 2);
                const std::uint8_t byte_val = codes[code_off];
                const std::uint8_t nibble   = (c & 1) ? (byte_val >> 4) : (byte_val & 0x0F);

                const int group     = c / 16;
                const int m_tile    = r / 128;
                const int row_inner = r % 128;
                const int row_mod32 = row_inner & 31;
                const int row_quart = row_inner >> 5;
                const int k_tile    = group >> 2;
                const int k_mod4    = group & 3;

                const std::size_t exp_scale_off =
                    static_cast<std::size_t>(m_tile * k_tiles + k_tile) * 512 +
                    row_mod32 * 16 + row_quart * 4 + k_mod4;
                const std::size_t scale_off =
                    static_cast<std::size_t>(e) * (rows * groups_per_row) + exp_scale_off;
                const std::uint8_t scale_val = scales[scale_off];

                const float dequant = (cpu_decode_e2m1(nibble) * cpu_decode_e4m3fn(scale_val)) / divisor;
                const float diff    = original - dequant;
                diff_sq += diff * diff;
                base_sq += original * original;
            }
        }
    }

    const double rel_l2 = (base_sq > 0.0) ? std::sqrt(diff_sq / base_sq) : 0.0;
    if (base_sq <= 0.0 || rel_l2 > 0.18) {
        std::cerr << "FAIL: NVFP4 quantizer dequantization error too high: rel_l2=" << rel_l2
                  << " (base_sq=" << base_sq << ")\n";
        cudaFree(d_bf16); cudaFree(d_out1); cudaFree(d_out2);
        return 1;
    }

    cudaFree(d_bf16);
    cudaFree(d_out1);
    cudaFree(d_out2);

    std::cout << "PASS: test_mtp_nvfp4_quantizer_unit (rel_l2=" << rel_l2
              << ", bitwise determinism verified)\n";
    return 0;
}

int test_mtp_moe_nvfp4_synthetic(ninfer::DeviceContext& device) {
    std::cout << "[TEST 3/4] test_mtp_moe_nvfp4_synthetic ...\n" << std::flush;
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    constexpr std::int32_t batch = 2;
    constexpr std::int32_t dim   = 2'560;

    ninfer::DeviceArena arena(64 * 1024 * 1024);
    ninfer::Tensor input                = arena.alloc(ninfer::DType::BF16, {dim, batch}, 256);
    ninfer::Tensor output               = arena.alloc(ninfer::DType::BF16, {dim, batch}, 256);
    ninfer::Tensor router_t             = arena.alloc(ninfer::DType::BF16, {512, dim}, 256);
    ninfer::Tensor shared_down_t        = arena.alloc(ninfer::DType::BF16, {dim, 640}, 256);
    ninfer::Tensor shared_gate_t        = arena.alloc(ninfer::DType::BF16, {640, dim}, 256);
    ninfer::Tensor shared_up_t          = arena.alloc(ninfer::DType::BF16, {640, dim}, 256);
    ninfer::Tensor shared_gate_weight_t = arena.alloc(ninfer::DType::BF16, {1, dim}, 256);

    const std::size_t gate_up_bytes = flash_next_nvfp4_expert_bank_payload_bytes(512, 1'280, dim);
    const std::size_t down_bytes    = flash_next_nvfp4_expert_bank_payload_bytes(512, dim, 640);
    const std::size_t bf16_gate_up_bytes = 512ULL * 1'280 * dim * 2;
    const std::size_t bf16_down_bytes    = 512ULL * dim * 640 * 2;

    ninfer::DeviceBuffer d_gate_up(gate_up_bytes);
    ninfer::DeviceBuffer d_down(down_bytes);
    ninfer::DeviceBuffer d_bf16_gate_up(bf16_gate_up_bytes);
    ninfer::DeviceBuffer d_bf16_down(bf16_down_bytes);

    std::vector<std::uint16_t> host_input(dim * batch);
    for (std::size_t i = 0; i < host_input.size(); ++i) {
        host_input[i] = float_to_bf16(0.1f * ((i % 17) + 1));
    }
    cudaMemcpy(input.data, host_input.data(), host_input.size() * 2, cudaMemcpyHostToDevice);
    cudaMemset(output.data, 0, output.bytes());
    populate_synthetic_bf16_bank(router_t.data, 512ULL * dim, 0.05f, device.stream);
    populate_synthetic_bf16_bank(shared_down_t.data, dim * 640ULL, 0.1f, device.stream);
    populate_synthetic_bf16_bank(shared_gate_t.data, 640ULL * dim, 0.1f, device.stream);
    populate_synthetic_bf16_bank(shared_up_t.data, 640ULL * dim, 0.1f, device.stream);
    populate_synthetic_bf16_bank(shared_gate_weight_t.data, dim, 0.1f, device.stream);

    populate_synthetic_bf16_bank(d_bf16_gate_up.p, 512ULL * 1'280 * dim, 0.2f, device.stream);
    populate_synthetic_bf16_bank(d_bf16_down.p, 512ULL * dim * 640, 0.1f, device.stream);

    quantize_bf16_expert_bank_to_nvfp4(d_bf16_gate_up.p, d_gate_up.p, 512, 1'280, dim, device.stream);
    quantize_bf16_expert_bank_to_nvfp4(d_bf16_down.p, d_down.p, 512, dim, 640, device.stream);
    device.synchronize();

    // Free temporary BF16 buffers
    d_bf16_gate_up = ninfer::DeviceBuffer();
    d_bf16_down    = ninfer::DeviceBuffer();

    Nvfp4ExpertBankView gate_up_bank = make_nvfp4_expert_bank_view(d_gate_up.p, gate_up_bytes, 512, 1'280, dim);
    Nvfp4ExpertBankView down_bank    = make_nvfp4_expert_bank_view(d_down.p, down_bytes, 512, dim, 640);

    MoeWeights weights{
        .router             = bf16_weight(router_t.data, 512, dim),
        .shared_down        = bf16_weight(shared_down_t.data, dim, 640),
        .shared_gate        = bf16_weight(shared_gate_t.data, 640, dim),
        .shared_up          = bf16_weight(shared_up_t.data, 640, dim),
        .shared_gate_weight = bf16_weight(shared_gate_weight_t.data, 1, dim),
        .expert_gate_up     = gate_up_bank,
        .expert_down        = down_bank,
    };

    flash_next_moe(input, weights, output, arena, device.stream);
    device.synchronize();

    std::vector<std::uint16_t> host_out1(dim * batch);
    cudaMemcpy(host_out1.data(), output.data, host_out1.size() * 2, cudaMemcpyDeviceToHost);

    double base_sq = 0.0;
    for (std::size_t i = 0; i < host_out1.size(); ++i) {
        float val = bf16_to_float(host_out1[i]);
        if (!std::isfinite(val)) {
            std::cerr << "FAIL: non-finite output at index " << i << "\n";
            return 1;
        }
        base_sq += val * val;
    }
    if (base_sq <= 0.0) {
        std::cerr << "FAIL: MoE output is vacuous zero\n";
        return 1;
    }

    // Determinism check
    cudaMemset(output.data, 0, output.bytes());
    flash_next_moe(input, weights, output, arena, device.stream);
    device.synchronize();

    std::vector<std::uint16_t> host_out2(dim * batch);
    cudaMemcpy(host_out2.data(), output.data, host_out2.size() * 2, cudaMemcpyDeviceToHost);

    if (std::memcmp(host_out1.data(), host_out2.data(), host_out1.size() * 2) != 0) {
        std::cerr << "FAIL: NVFP4 MoE execution is non-deterministic between consecutive runs\n";
        return 1;
    }

    std::cout << "PASS: test_mtp_moe_nvfp4_synthetic (base_sq=" << base_sq << ", bit-exact determinism verified)\n";
    return 0;
}

void flash_next_mtp_step_bf16_for_test(
    const ninfer::targets::qwen3_8_flash_next::detail::TextModelView& model,
    const ninfer::targets::qwen3_8_flash_next::detail::MoeBf16Weights& moe_bf16,
    const ninfer::Tensor& input_embedding, const ninfer::Tensor& backbone_hyper_hidden,
    const ninfer::Tensor& token_indices, const ninfer::Tensor& mrope_positions,
    const ninfer::Tensor& table_rows, const ninfer::Tensor& selected_blocks,
    const ninfer::Tensor& selected_counts,
    const ninfer::targets::qwen3_8_flash_next::detail::QsaAttentionCacheView& mtp_cache,
    ninfer::WorkspaceArena& workspace, ninfer::Tensor& draft_logits,
    ninfer::Tensor& draft_tokens, cudaStream_t stream) {

    using namespace ninfer;
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    const auto scope = workspace.scope();
    const std::int32_t batch = input_embedding.ne[1];
    ninfer::Tensor mtp_hyper_hidden = workspace.alloc(ninfer::DType::BF16, {10'240, batch}, 256);
    ninfer::Tensor attn_in          = workspace.alloc(ninfer::DType::BF16, {2'560, batch}, 256);
    ninfer::Tensor attn_out         = workspace.alloc(ninfer::DType::BF16, {2'560, batch}, 256);
    ninfer::Tensor mlp_in           = workspace.alloc(ninfer::DType::BF16, {2'560, batch}, 256);
    ninfer::Tensor mlp_out          = workspace.alloc(ninfer::DType::BF16, {2'560, batch}, 256);
    ninfer::Tensor mtp_final_hidden = workspace.alloc(ninfer::DType::BF16, {2'560, batch}, 256);

    FlashNextHyperWorkspace hyper_scratch = allocate_flash_next_hyper_workspace(workspace, batch);
    const auto& mtp = *model.mtp;

    flash_next_mtp_stem(mtp, input_embedding, backbone_hyper_hidden, workspace, mtp_hyper_hidden, stream);

    flash_next_hyper_prepare(mtp_hyper_hidden, mtp.attention_hyper, hyper_scratch, attn_in, stream);
    flash_next_qsa_attention_decode(attn_in, mtp.attention, token_indices, mrope_positions,
                                    table_rows, selected_blocks, selected_counts, mtp_cache,
                                    workspace, attn_out, stream);
    flash_next_hyper_inject(attn_out, hyper_scratch.injection, mtp_hyper_hidden, stream);

    flash_next_hyper_prepare(mtp_hyper_hidden, mtp.mlp_hyper, hyper_scratch, mlp_in, stream);
    flash_next_moe_bf16(mlp_in, moe_bf16, mlp_out, workspace, stream);
    flash_next_hyper_inject(mlp_out, hyper_scratch.injection, mtp_hyper_hidden, stream);

    flash_next_hyper_mix(mtp_hyper_hidden, mtp.mixer, hyper_scratch, mtp_final_hidden, stream);
    ops::linear(mtp_final_hidden, model.output_head, draft_logits, ops::LinearPolicy::A16Only, workspace, stream);
    ops::argmax(draft_logits, draft_tokens, draft_logits.ne[0], stream);
}

int test_mtp_full_step_synthetic(ninfer::DeviceContext& device, const std::string& dump_dir = "") {
    std::cout << "[TEST 4/4] test_mtp_full_step_synthetic & benchmark ...\n" << std::flush;
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    constexpr std::int32_t batch = 1;
    constexpr std::int32_t dim   = 2'560;
    constexpr std::int32_t vocab = 248'320;

    // Allocate 2GB arena for weights, activations, and workspaces
    ninfer::DeviceArena arena(2ULL * 1024 * 1024 * 1024);

    // Weights
    ninfer::Tensor emb_proj_t      = arena.alloc(ninfer::DType::BF16, {dim, dim}, 256);
    ninfer::Tensor hid_proj_t      = arena.alloc(ninfer::DType::BF16, {dim, dim}, 256);
    ninfer::Tensor emb_norm_t      = arena.alloc(ninfer::DType::BF16, {dim}, 256);
    ninfer::Tensor hid_norm_t      = arena.alloc(ninfer::DType::BF16, {10'240}, 256);
    ninfer::Tensor mix_norm_t      = arena.alloc(ninfer::DType::BF16, {10'240}, 256);
    ninfer::Tensor mix_down_t      = arena.alloc(ninfer::DType::BF16, {320, 10'240}, 256);
    ninfer::Tensor mix_up_t        = arena.alloc(ninfer::DType::BF16, {10'240, 320}, 256);
    ninfer::Tensor attn_hc_inj_t   = arena.alloc(ninfer::DType::BF16, {4, 10'240}, 256);
    ninfer::Tensor attn_hc_norm_t  = arena.alloc(ninfer::DType::BF16, {10'240}, 256);
    ninfer::Tensor attn_hc_down_t  = arena.alloc(ninfer::DType::BF16, {320, 10'240}, 256);
    ninfer::Tensor attn_hc_up_t    = arena.alloc(ninfer::DType::BF16, {10'240, 320}, 256);
    ninfer::Tensor mlp_hc_inj_t    = arena.alloc(ninfer::DType::BF16, {4, 10'240}, 256);
    ninfer::Tensor mlp_hc_norm_t   = arena.alloc(ninfer::DType::BF16, {10'240}, 256);
    ninfer::Tensor mlp_hc_down_t   = arena.alloc(ninfer::DType::BF16, {320, 10'240}, 256);
    ninfer::Tensor mlp_hc_up_t     = arena.alloc(ninfer::DType::BF16, {10'240, 320}, 256);
    ninfer::Tensor qk_idx_t        = arena.alloc(ninfer::DType::BF16, {640, dim}, 256);
    ninfer::Tensor qk_knorm_t      = arena.alloc(ninfer::DType::BF16, {128}, 256);
    ninfer::Tensor qk_qnorm_t      = arena.alloc(ninfer::DType::BF16, {128}, 256);
    ninfer::Tensor q_norm_t        = arena.alloc(ninfer::DType::BF16, {256}, 256);
    ninfer::Tensor k_norm_t        = arena.alloc(ninfer::DType::BF16, {256}, 256);
    ninfer::Tensor qgkv_t          = arena.alloc(ninfer::DType::BF16, {13'312, dim}, 256);
    ninfer::Tensor o_proj_t        = arena.alloc(ninfer::DType::BF16, {dim, 6'144}, 256);
    ninfer::Tensor router_t        = arena.alloc(ninfer::DType::BF16, {512, dim}, 256);
    ninfer::Tensor shared_down_t   = arena.alloc(ninfer::DType::BF16, {dim, 640}, 256);
    ninfer::Tensor shared_gate_t   = arena.alloc(ninfer::DType::BF16, {640, dim}, 256);
    ninfer::Tensor shared_up_t     = arena.alloc(ninfer::DType::BF16, {640, dim}, 256);
    ninfer::Tensor shared_gw_t     = arena.alloc(ninfer::DType::BF16, {1, dim}, 256);

    const std::size_t gate_up_bytes = flash_next_nvfp4_expert_bank_payload_bytes(512, 1'280, dim);
    const std::size_t down_bytes    = flash_next_nvfp4_expert_bank_payload_bytes(512, dim, 640);
    const std::size_t bf16_gate_up_bytes = 512ULL * 1'280 * dim * 2;
    const std::size_t bf16_down_bytes    = 512ULL * dim * 640 * 2;

    ninfer::DeviceBuffer exp_gate_up(gate_up_bytes);
    ninfer::DeviceBuffer exp_down(down_bytes);
    ninfer::DeviceBuffer bf16_gate_up(bf16_gate_up_bytes);
    ninfer::DeviceBuffer bf16_down(bf16_down_bytes);

    ninfer::Tensor head_t = arena.alloc(ninfer::DType::BF16, {vocab, dim}, 256);

    // Activations
    ninfer::Tensor input_emb       = arena.alloc(ninfer::DType::BF16, {dim, batch}, 256);
    ninfer::Tensor backbone_hidden = arena.alloc(ninfer::DType::BF16, {10'240, batch}, 256);
    ninfer::Tensor token_indices   = arena.alloc(ninfer::DType::I32, {batch}, 16);
    ninfer::Tensor mrope_positions = arena.alloc(ninfer::DType::I32, {batch, 3}, 16);
    ninfer::Tensor table_rows      = arena.alloc(ninfer::DType::I32, {batch}, 16);
    ninfer::Tensor selected_blocks = arena.alloc(ninfer::DType::I32, {512, batch}, 16);
    ninfer::Tensor selected_counts = arena.alloc(ninfer::DType::I32, {batch}, 16);
    ninfer::Tensor draft_logits    = arena.alloc(ninfer::DType::BF16, {vocab, batch}, 256);
    ninfer::Tensor draft_tokens    = arena.alloc(ninfer::DType::I32, {batch}, 16);

    // MTP KV Cache
    ninfer::Tensor key_pages    = arena.alloc(ninfer::DType::BF16, {256, 64, 2, 64}, 256);
    ninfer::Tensor value_pages  = arena.alloc(ninfer::DType::BF16, {256, 64, 2, 64}, 256);
    ninfer::Tensor block_tables = arena.alloc(ninfer::DType::I32, {64, 1}, 16);

    auto source_slots = arena.alloc(ninfer::DType::I32, {batch}, 16);
    auto destination_slots = arena.alloc(ninfer::DType::I32, {batch}, 16);
    QsaIndexerCacheView indexer_cache{
        .block_keys = arena.alloc(ninfer::DType::BF16, {128, 64, 16}, 256),
        .block_tables = arena.alloc(ninfer::DType::I32, {16, 1}, 16),
        .raw_keys = arena.alloc(ninfer::DType::BF16, {128, 4, 1}, 256),
        .raw_positions = arena.alloc(ninfer::DType::I32, {3, 4, 1}, 16),
    };
    cudaMemset(arena.base(), 0, arena.capacity());

    std::vector<std::uint16_t> diag(dim * dim, 0);
    for (int i = 0; i < dim; ++i) {
        diag[i * dim + i] = 0x3F80U;
    }
    cudaMemcpy(emb_proj_t.data, diag.data(), diag.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(hid_proj_t.data, diag.data(), diag.size() * 2, cudaMemcpyHostToDevice);

    std::vector<std::uint16_t> head_prefix(100 * dim, 0);
    for (int i = 0; i < 100; ++i) {
        head_prefix[i * dim] = float_to_bf16(static_cast<float>(i + 1));
    }
    cudaMemcpy(head_t.data, head_prefix.data(), head_prefix.size() * 2, cudaMemcpyHostToDevice);

    std::vector<std::uint16_t> host_input_emb(dim * batch, 0x3F80U);
    std::vector<std::uint16_t> host_backbone(10'240 * batch);
    for (std::size_t i = 0; i < host_backbone.size(); ++i) {
        host_backbone[i] = float_to_bf16(static_cast<float>((i % 10'240) / dim + 1));
    }
    cudaMemcpy(input_emb.data, host_input_emb.data(), host_input_emb.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(backbone_hidden.data, host_backbone.data(), host_backbone.size() * 2, cudaMemcpyHostToDevice);

    populate_constant_bf16_bank(emb_norm_t.data, dim, 0.0f, device.stream);
    populate_constant_bf16_bank(hid_norm_t.data, 10'240, 0.0f, device.stream);
    populate_constant_bf16_bank(mix_norm_t.data, 10'240, 0.0f, device.stream);
    populate_constant_bf16_bank(attn_hc_norm_t.data, 10'240, 0.0f, device.stream);
    populate_constant_bf16_bank(mlp_hc_norm_t.data, 10'240, 0.0f, device.stream);
    populate_constant_bf16_bank(qk_knorm_t.data, 128, 0.0f, device.stream);
    populate_constant_bf16_bank(qk_qnorm_t.data, 128, 0.0f, device.stream);
    populate_constant_bf16_bank(q_norm_t.data, 256, 0.0f, device.stream);
    populate_constant_bf16_bank(k_norm_t.data, 256, 0.0f, device.stream);
    populate_synthetic_bf16_bank(router_t.data, 512ULL * dim, 0.05f, device.stream);
    populate_synthetic_bf16_bank(shared_down_t.data, dim * 640ULL, 0.1f, device.stream);
    populate_synthetic_bf16_bank(shared_gate_t.data, 640ULL * dim, 0.1f, device.stream);
    populate_synthetic_bf16_bank(shared_up_t.data, 640ULL * dim, 0.1f, device.stream);
    populate_synthetic_bf16_bank(shared_gw_t.data, dim, 0.1f, device.stream);

    populate_synthetic_bf16_bank(bf16_gate_up.p, 512ULL * 1'280 * dim, 0.2f, device.stream);
    populate_synthetic_bf16_bank(bf16_down.p, 512ULL * dim * 640, 0.1f, device.stream);

    quantize_bf16_expert_bank_to_nvfp4(bf16_gate_up.p, exp_gate_up.p, 512, 1'280, dim, device.stream);
    quantize_bf16_expert_bank_to_nvfp4(bf16_down.p, exp_down.p, 512, dim, 640, device.stream);
    device.synchronize();

    std::vector<std::uint16_t> host_qgkv(13'312 * dim, 0x3800U);
    std::vector<std::uint16_t> host_oproj(dim * 6'144, 0x3800U);
    cudaMemcpy(qgkv_t.data, host_qgkv.data(), host_qgkv.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(o_proj_t.data, host_oproj.data(), host_oproj.size() * 2, cudaMemcpyHostToDevice);

    Nvfp4ExpertBankView gate_up_bank = make_nvfp4_expert_bank_view(exp_gate_up.p, gate_up_bytes, 512, 1'280, dim);
    Nvfp4ExpertBankView down_bank    = make_nvfp4_expert_bank_view(exp_down.p, down_bytes, 512, dim, 640);

    MtpModelView mtp{
        .embedding_projection = bf16_weight(emb_proj_t.data, dim, dim),
        .hidden_projection    = bf16_weight(hid_proj_t.data, dim, dim),
        .mixer = {
            .norm           = mix_norm_t,
            .input_mix_down = bf16_weight(mix_down_t.data, 320, 10'240),
            .input_mix_up   = bf16_weight(mix_up_t.data, 10'240, 320),
        },
        .attention_hyper = {
            .block_inject   = bf16_weight(attn_hc_inj_t.data, 4, 10'240),
            .norm           = attn_hc_norm_t,
            .input_mix_down = bf16_weight(attn_hc_down_t.data, 320, 10'240),
            .input_mix_up   = bf16_weight(attn_hc_up_t.data, 10'240, 320),
        },
        .attention = {
            .indexer_query_key    = bf16_weight(qk_idx_t.data, 640, dim),
            .indexer_key_norm     = qk_knorm_t,
            .indexer_query_norm   = qk_qnorm_t,
            .key_norm             = k_norm_t,
            .query_norm           = q_norm_t,
            .query_gate_key_value = bf16_weight(qgkv_t.data, 13'312, dim),
            .output               = bf16_weight(o_proj_t.data, dim, 6'144),
        },
        .mlp_hyper = {
            .block_inject   = bf16_weight(mlp_hc_inj_t.data, 4, 10'240),
            .norm           = mlp_hc_norm_t,
            .input_mix_down = bf16_weight(mlp_hc_down_t.data, 320, 10'240),
            .input_mix_up   = bf16_weight(mlp_hc_up_t.data, 10'240, 320),
        },
        .moe = {
            .router             = bf16_weight(router_t.data, 512, dim),
            .shared_down        = bf16_weight(shared_down_t.data, dim, 640),
            .shared_gate        = bf16_weight(shared_gate_t.data, 640, dim),
            .shared_up          = bf16_weight(shared_up_t.data, 640, dim),
            .shared_gate_weight = bf16_weight(shared_gw_t.data, 1, dim),
            .expert_gate_up     = gate_up_bank,
            .expert_down        = down_bank,
        },
        .embedding_norm = emb_norm_t,
        .hidden_norm    = hid_norm_t,
    };

    TextModelView model{};
    model.output_head = bf16_weight(head_t.data, vocab, dim);
    model.mtp         = mtp;

    QsaAttentionCacheView mtp_cache{
        .key_pages    = key_pages,
        .value_pages  = value_pages,
        .block_tables = block_tables,
    };

    std::unique_ptr<StateDumper> dumper;
    std::optional<FlashNextDecodeStateSink> sink;
    if (!dump_dir.empty()) {
        dumper = std::make_unique<StateDumper>(dump_dir);
        sink   = dumper->make_sink(0, 0, {0, 0, 0});
    }

    try {
        flash_next_mtp_step(model, input_emb, backbone_hidden, token_indices, mrope_positions,
                            table_rows, source_slots, destination_slots, indexer_cache, mtp_cache, 1024, 0, arena,
                            draft_logits, draft_tokens, device.stream, sink ? &*sink : nullptr);
        device.synchronize();
    } catch (const std::exception& e) {
        std::cerr << "FAIL: flash_next_mtp_step threw exception: " << e.what() << "\n";
        return 1;
    }

    if (dumper) {
        dumper->write_manifest();
    }

    std::vector<std::uint16_t> host_logits1(vocab * batch);
    std::vector<std::int32_t> host_drafts1(batch);
    cudaMemcpy(host_logits1.data(), draft_logits.data, host_logits1.size() * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_drafts1.data(), draft_tokens.data, host_drafts1.size() * sizeof(std::int32_t),
               cudaMemcpyDeviceToHost);

    double base_sq = 0.0;
    for (std::size_t i = 0; i < host_logits1.size(); ++i) {
        float val = bf16_to_float(host_logits1[i]);
        if (!std::isfinite(val)) {
            std::cerr << "FAIL: non-finite draft logit at " << i << "\n";
            return 1;
        }
        base_sq += val * val;
    }
    if (base_sq <= 0.0) {
        std::cerr << "FAIL: draft logits are vacuous zero\n";
        return 1;
    }

    if (host_drafts1[0] < 0 || host_drafts1[0] >= vocab) {
        std::cerr << "FAIL: draft token out of valid range: " << host_drafts1[0] << "\n";
        return 1;
    }

    // Run 2: Check bitwise determinism
    cudaMemset(draft_logits.data, 0, draft_logits.bytes());
    cudaMemset(draft_tokens.data, 0, draft_tokens.bytes());
    flash_next_mtp_step(model, input_emb, backbone_hidden, token_indices, mrope_positions,
                        table_rows, source_slots, destination_slots, indexer_cache, mtp_cache, 1024, 0, arena,
                        draft_logits, draft_tokens, device.stream, nullptr);
    device.synchronize();

    std::vector<std::uint16_t> host_logits2(vocab * batch);
    std::vector<std::int32_t> host_drafts2(batch);
    cudaMemcpy(host_logits2.data(), draft_logits.data, host_logits2.size() * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_drafts2.data(), draft_tokens.data, host_drafts2.size() * sizeof(std::int32_t),
               cudaMemcpyDeviceToHost);

    if (std::memcmp(host_logits1.data(), host_logits2.data(), host_logits1.size() * 2) != 0 ||
        std::memcmp(host_drafts1.data(), host_drafts2.data(), host_drafts1.size() * sizeof(std::int32_t)) != 0) {
        std::cerr << "FAIL: MTP full step is not bit-deterministic across runs\n";
        return 1;
    }

    // Compare with BF16 MTP draft step on same inputs for parity drift & latency A/B
    Bf16ExpertBankView bf16_gate_up_bank{
        .data             = static_cast<const std::byte*>(bf16_gate_up.p),
        .experts          = 512,
        .rows             = 1'280,
        .columns          = dim,
        .bytes_per_expert = 1'280ULL * dim * 2,
    };
    Bf16ExpertBankView bf16_down_bank{
        .data             = static_cast<const std::byte*>(bf16_down.p),
        .experts          = 512,
        .rows             = dim,
        .columns          = 640,
        .bytes_per_expert = dim * 640ULL * 2,
    };

    // Benchmark BF16 MoE and NVFP4 MoE draft steps
    MoeBf16Weights moe_bf16{
        .router             = mtp.moe.router,
        .shared_down        = mtp.moe.shared_down,
        .shared_gate        = mtp.moe.shared_gate,
        .shared_up          = mtp.moe.shared_up,
        .shared_gate_weight = mtp.moe.shared_gate_weight,
        .expert_gate_up     = bf16_gate_up_bank,
        .expert_down        = bf16_down_bank,
    };

    // Warmup
    constexpr int kWarmup = 10;
    constexpr int kIters  = 50;
    for (int i = 0; i < kWarmup; ++i) {
        flash_next_mtp_step(model, input_emb, backbone_hidden, token_indices, mrope_positions,
                            table_rows, source_slots, destination_slots, indexer_cache, mtp_cache, 1024, 0, arena,
                            draft_logits, draft_tokens, device.stream, nullptr);
        flash_next_mtp_step_bf16_for_test(model, moe_bf16, input_emb, backbone_hidden, token_indices,
                                          mrope_positions, table_rows, selected_blocks,
                                          selected_counts, mtp_cache, arena, draft_logits,
                                          draft_tokens, device.stream);
    }
    device.synchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 1. Time NVFP4 draft step
    cudaEventRecord(start, device.stream);
    for (int i = 0; i < kIters; ++i) {
        flash_next_mtp_step(model, input_emb, backbone_hidden, token_indices, mrope_positions,
                            table_rows, source_slots, destination_slots, indexer_cache, mtp_cache, 1024, 0, arena,
                            draft_logits, draft_tokens, device.stream, nullptr);
    }
    cudaEventRecord(stop, device.stream);
    cudaEventSynchronize(stop);
    float elapsed_ms_nvfp4 = 0.0f;
    cudaEventElapsedTime(&elapsed_ms_nvfp4, start, stop);
    const float us_nvfp4 = (elapsed_ms_nvfp4 * 1000.0f) / kIters;

    // 2. Time BF16 draft step
    cudaEventRecord(start, device.stream);
    for (int i = 0; i < kIters; ++i) {
        flash_next_mtp_step_bf16_for_test(model, moe_bf16, input_emb, backbone_hidden, token_indices,
                                          mrope_positions, table_rows, selected_blocks,
                                          selected_counts, mtp_cache, arena, draft_logits,
                                          draft_tokens, device.stream);
    }
    cudaEventRecord(stop, device.stream);
    cudaEventSynchronize(stop);
    float elapsed_ms_bf16 = 0.0f;
    cudaEventElapsedTime(&elapsed_ms_bf16, start, stop);
    const float us_bf16 = (elapsed_ms_bf16 * 1000.0f) / kIters;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Compute top-1 agreement and Rel-L2 over 25 seeded positions
    int top1_matches = 0;
    constexpr int kTestPositions = 25;
    double sum_rel_l2 = 0.0;

    for (int pos = 0; pos < kTestPositions; ++pos) {
        std::vector<std::uint16_t> h_emb(dim);
        for (int c = 0; c < dim; ++c) {
            h_emb[c] = float_to_bf16(0.2f * sinf(static_cast<float>(pos * 31 + c)));
        }
        cudaMemcpy(input_emb.data, h_emb.data(), dim * 2, cudaMemcpyHostToDevice);

        // Run BF16
        flash_next_mtp_step_bf16_for_test(model, moe_bf16, input_emb, backbone_hidden, token_indices,
                                          mrope_positions, table_rows, selected_blocks,
                                          selected_counts, mtp_cache, arena, draft_logits,
                                          draft_tokens, device.stream);
        device.synchronize();
        std::vector<std::int32_t> tok_bf16(1);
        std::vector<std::uint16_t> logits_bf16(vocab);
        cudaMemcpy(tok_bf16.data(), draft_tokens.data, sizeof(std::int32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(logits_bf16.data(), draft_logits.data, vocab * 2, cudaMemcpyDeviceToHost);

        // Run NVFP4
        flash_next_mtp_step(model, input_emb, backbone_hidden, token_indices, mrope_positions,
                            table_rows, source_slots, destination_slots, indexer_cache, mtp_cache, 1024, 0, arena,
                            draft_logits, draft_tokens, device.stream, nullptr);
        device.synchronize();
        std::vector<std::int32_t> tok_nvfp4(1);
        std::vector<std::uint16_t> logits_nvfp4(vocab);
        cudaMemcpy(tok_nvfp4.data(), draft_tokens.data, sizeof(std::int32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(logits_nvfp4.data(), draft_logits.data, vocab * 2, cudaMemcpyDeviceToHost);

        if (tok_bf16[0] == tok_nvfp4[0]) {
            top1_matches++;
        }

        double pos_diff_sq = 0.0;
        double pos_base_sq = 0.0;
        for (std::size_t i = 0; i < vocab; ++i) {
            float b = bf16_to_float(logits_bf16[i]);
            float n = bf16_to_float(logits_nvfp4[i]);
            float diff = b - n;
            pos_diff_sq += diff * diff;
            pos_base_sq += b * b;
        }
        if (pos_base_sq > 0.0) {
            sum_rel_l2 += std::sqrt(pos_diff_sq / pos_base_sq);
        }
    }
    const float top1_agreement = (static_cast<float>(top1_matches) / kTestPositions) * 100.0f;
    const float avg_rel_l2     = static_cast<float>(sum_rel_l2 / kTestPositions);

    std::cout << "PASS: test_mtp_full_step_synthetic\n"
              << "  - Draft Step Latency (NVFP4): " << us_nvfp4 << " us (" << (us_nvfp4 / 1000.0f) << " ms)\n"
              << "  - Draft Step Latency (BF16):  " << us_bf16 << " us (" << (us_bf16 / 1000.0f) << " ms)\n"
              << "  - Speedup:                    " << (us_bf16 / us_nvfp4) << "x\n"
              << "  - Top-1 Draft Agreement Rate: " << top1_agreement << "% (" << top1_matches << "/" << kTestPositions << " positions match)\n"
              << "  - Draft Logits Rel-L2 Drift:  " << avg_rel_l2 << "\n"
              << "  - Two-Run Determinism:        VERIFIED (bit-exact)\n";
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    int device_count = 0;
    const auto err   = cudaGetDeviceCount(&device_count);
    if (cuda_unavailable(err) || device_count == 0) {
        std::cerr << "skip: CUDA device unavailable\n";
        return 77;
    }
    if (err != cudaSuccess) {
        std::cerr << "CUDA initialization error: " << cudaGetErrorString(err) << "\n";
        return 1;
    }

    std::string dump_dir;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--dump-states" && i + 1 < argc) {
            dump_dir = argv[++i];
        }
    }

    ninfer::DeviceContext device(0);

    if (test_mtp_stem_oracle(device) != 0) return 1;
    if (test_mtp_target_state_alignment(device) != 0) return 1;
    if (test_mtp_nvfp4_quantizer_unit(device) != 0) return 1;
    if (test_mtp_moe_nvfp4_synthetic(device) != 0) return 1;
    if (test_mtp_full_step_synthetic(device, dump_dir) != 0) return 1;

    std::cout << "ALL MTP TESTS PASSED\n";
    return 0;
}
