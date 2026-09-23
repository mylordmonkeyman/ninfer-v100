#include "core/arena.h"
#include "core/device.h"
#include "targets/qwen3_8_flash_next/impl/gdn.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
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

std::uint64_t fp8_payload_bytes(std::int32_t rows, std::int32_t columns) {
    const std::uint64_t codes = static_cast<std::uint64_t>(rows) * columns;
    return ((codes + 255U) & ~std::uint64_t{255U}) + static_cast<std::uint64_t>(rows) * 4;
}

float bf16_to_float(std::uint16_t value) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(value) << 16U);
}

ninfer::Weight fp8_f32_weight(ninfer::DeviceBuffer& storage, std::int32_t rows,
                              std::int32_t columns) {
    const std::uint64_t codes        = static_cast<std::uint64_t>(rows) * columns;
    const std::uint64_t scale_offset = (codes + 255U) & ~std::uint64_t{255U};
    storage.fill(0);
    std::vector<float> scales(static_cast<std::size_t>(rows), 1.0F);
    storage.copy_from_host(scales.data(), scales.size() * sizeof(float), scale_offset);

    auto* payload = static_cast<std::byte*>(storage.p);
    ninfer::Weight out{};
    out.payload         = payload;
    out.payload_bytes   = storage.bytes;
    out.qdata           = payload;
    out.scales          = payload + scale_offset;
    out.qtype           = ninfer::QType::FP8_E4M3FN_ROW_F32S;
    out.layout          = ninfer::QuantLayout::RowScale;
    out.scale_dtype     = ninfer::DType::FP32;
    out.group_size      = static_cast<std::uint32_t>(columns);
    out.group           = columns;
    out.n               = rows;
    out.k               = columns;
    out.ndim            = 2;
    out.shape[0]        = rows;
    out.shape[1]        = columns;
    out.padded_shape[0] = rows;
    out.padded_shape[1] = columns;
    out.scale_ne[0]     = rows;
    out.scale_nb[0]     = 4;
    out.scale_nb[1]     = static_cast<std::int64_t>(rows) * 4;
    out.scale_nb[2]     = out.scale_nb[1];
    out.scale_nb[3]     = out.scale_nb[1];
    return out;
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

    try {
        ninfer::DeviceContext device(0);
    ninfer::DeviceBuffer input(2'560 * 2);
    ninfer::DeviceBuffer output(2'560 * 2);
    ninfer::DeviceBuffer a_log(48 * 2);
    ninfer::DeviceBuffer convolution(10'240ULL * 4 * 2);
    ninfer::DeviceBuffer dt_bias(48 * 2);
    ninfer::DeviceBuffer a_b_projection(96ULL * 2'560 * 2);
    ninfer::DeviceBuffer norm(128 * 2);
    ninfer::DeviceBuffer qkvz(fp8_payload_bytes(16'384, 2'560));
    ninfer::DeviceBuffer output_projection(fp8_payload_bytes(2'560, 6'144));
    ninfer::DeviceBuffer source_slots(sizeof(std::int32_t));
    ninfer::DeviceBuffer destination_slots(sizeof(std::int32_t));
    ninfer::DeviceBuffer convolution_states(10'240ULL * 3 * 2 * 2);
    ninfer::DeviceBuffer ssm_states(128ULL * 128 * 48 * 2 * sizeof(float));

    input.fill(0);
    output.fill(0xFF);
    a_log.fill(0);
    convolution.fill(0);
    dt_bias.fill(0);
    a_b_projection.fill(0);
    norm.fill(0);
    ssm_states.fill(0);
    const std::int32_t source      = 0;
    const std::int32_t destination = 1;
    source_slots.copy_from_host(&source, sizeof(source));
    destination_slots.copy_from_host(&destination, sizeof(destination));

    std::vector<std::uint16_t> convolution_state_values(10'240ULL * 3 * 2, 0x7FC1U);
    std::fill_n(convolution_state_values.begin(), 10'240, 0x3F80U);          // source oldest: 1
    std::fill_n(convolution_state_values.begin() + 10'240, 10'240, 0x4000U); // source: 2
    std::fill_n(convolution_state_values.begin() + 20'480, 10'240, 0x4040U); // source: 3
    convolution_states.copy_from_host(convolution_state_values.data(),
                                      convolution_state_values.size() * sizeof(std::uint16_t));

    GdnWeights weights{
        .a_log             = ninfer::Tensor(a_log.p, ninfer::DType::BF16, {48}),
        .convolution       = ninfer::Tensor(convolution.p, ninfer::DType::BF16, {10'240, 4}),
        .dt_bias           = ninfer::Tensor(dt_bias.p, ninfer::DType::BF16, {48}),
        .a_b_projection    = bf16_weight(a_b_projection.p, 96, 2'560),
        .norm              = ninfer::Tensor(norm.p, ninfer::DType::BF16, {128}),
        .query_key_value_z = fp8_f32_weight(qkvz, 16'384, 2'560),
        .output            = fp8_f32_weight(output_projection, 2'560, 6'144),
    };
    ninfer::Tensor input_view(input.p, ninfer::DType::BF16, {2'560, 1});
    ninfer::Tensor output_view(output.p, ninfer::DType::BF16, {2'560, 1});
    ninfer::Tensor source_view(source_slots.p, ninfer::DType::I32, {1});
    ninfer::Tensor destination_view(destination_slots.p, ninfer::DType::I32, {1});
    ninfer::Tensor convolution_state_view(convolution_states.p, ninfer::DType::BF16,
                                          {10'240, 3, 2});
    ninfer::Tensor ssm_state_view(ssm_states.p, ninfer::DType::FP32, {128, 128, 48, 2});
    ninfer::WorkspaceArena workspace(flash_next_gdn_workspace_capacity_bytes(1, 16));
    flash_next_gdn_decode(input_view, weights, source_view, destination_view,
                          convolution_state_view, ssm_state_view, workspace, output_view,
                          device.stream);
    device.synchronize();

    std::array<std::uint16_t, 2'560> actual_output{};
    output.copy_to_host(actual_output.data(), sizeof(actual_output));
    if (!std::all_of(actual_output.begin(), actual_output.end(),
                     [](std::uint16_t value) { return value == 0; })) {
        std::cerr << "Flash-Next zero GDN did not produce exact BF16 zero\n";
        return 1;
    }

    convolution_states.copy_to_host(convolution_state_values.data(),
                                    convolution_state_values.size() * sizeof(std::uint16_t));
    const auto destination_begin = convolution_state_values.begin() + 30'720;
    if (!std::all_of(destination_begin, destination_begin + 10'240,
                     [](std::uint16_t value) { return value == 0x4000U; }) ||
        !std::all_of(destination_begin + 10'240, destination_begin + 20'480,
                     [](std::uint16_t value) { return value == 0x4040U; }) ||
        !std::all_of(destination_begin + 20'480, destination_begin + 30'720,
                     [](std::uint16_t value) { return value == 0; })) {
        std::cerr << "Flash-Next GDN did not publish the four-tap convolution state transition\n";
        return 1;
    }

    // Exercise the complete nonzero boundary. Each projection row reads input column zero with
    // exact FP8 1.0; the causal current tap is BF16 1.0. The resulting represented q/k/v is
    // BF16 SiLU(1), so the independently evaluated first state element is beta*v*k_normalized.
    // The z rows use FP8 2.0 (E4M3 0x40) to strictly discriminate sigmoid(2) ~= 0.8808 from silu(2) ~= 1.7616.
    std::array<std::uint16_t, 2'560> nonzero_input{};
    nonzero_input.front() = 0x3F80U;
    input.copy_from_host(nonzero_input.data(), sizeof(nonzero_input));
    output.fill(0xFF);
    std::array<std::uint16_t, 128> norm_values{};
    norm_values.fill(0x3F80U); // RMSNormGated owns a direct, ones-initialized weight.
    norm.copy_from_host(norm_values.data(), sizeof(norm_values));
    convolution_states.fill(0);
    ssm_states.fill(0);
    std::vector<std::uint16_t> convolution_values(10'240ULL * 4, 0);
    std::fill(convolution_values.begin() + 30'720, convolution_values.end(), 0x3F80U);
    convolution.copy_from_host(convolution_values.data(),
                               convolution_values.size() * sizeof(std::uint16_t));
    std::vector<std::uint8_t> qkvz_codes(16'384ULL * 2'560, 0);
    for (std::int32_t row = 0; row < 10'240; ++row) {
        qkvz_codes[static_cast<std::size_t>(row) * 2'560] = 0x38U; // E4M3 1.0
    }
    for (std::int32_t row = 10'240; row < 16'384; ++row) {
        qkvz_codes[static_cast<std::size_t>(row) * 2'560] = 0x40U; // E4M3 2.0 (z = 2.0)
    }
    qkvz.copy_from_host(qkvz_codes.data(), qkvz_codes.size());
    std::vector<std::uint8_t> output_codes(2'560ULL * 6'144, 0);
    for (std::int32_t row = 0; row < 2'560; ++row) {
        output_codes[static_cast<std::size_t>(row) * 6'144] = 0x38U;
    }
    output_projection.copy_from_host(output_codes.data(), output_codes.size());
    device.synchronize(); // Drain legacy stream before kernel launch on non-blocking device.stream

    flash_next_gdn_decode(input_view, weights, source_view, destination_view,
                          convolution_state_view, ssm_state_view, workspace, output_view,
                          device.stream);
    device.synchronize();

    float actual_state                     = 0.0F;
    constexpr std::size_t state_slot_bytes = 128ULL * 128 * 48 * sizeof(float);
    ssm_states.copy_to_host(&actual_state, sizeof(actual_state), state_slot_bytes);
    const float activated      = 1.0F / (1.0F + std::exp(-1.0F));
    const float normalized_key = activated / std::sqrt(128.0F * activated * activated + 1.0e-6F);
    const float expected_state = 0.5F * activated * normalized_key;
    if (std::abs(actual_state - expected_state) > 2.0e-4F) {
        std::cerr << "Flash-Next GDN recurrent state disagreed with the independent formula: "
                  << actual_state << " vs " << expected_state << '\n';
        return 1;
    }

    output.copy_to_host(actual_output.data(), sizeof(actual_output));
    const float first_output    = bf16_to_float(actual_output.front());
    const float expected_output = 1.0F / (1.0F + std::exp(-2.0F)); // sigmoid(2.0) ~= 0.8808F
    if (std::abs(first_output - expected_output) > 4.0e-3F ||
        !std::all_of(actual_output.begin(), actual_output.end(),
                     [&](std::uint16_t value) { return value == actual_output.front(); })) {
        std::cerr << "Flash-Next nonzero GDN output was outside its represented gated-norm range: "
                  << first_output << " vs expected " << expected_output << '\n';
        return 1;
    }

    // -----------------------------------------------------------------
    // Test 2: GDN T-wide chunk vs sequential equivalence test for T = 4..16
    // -----------------------------------------------------------------
    for (int T = 4; T <= 16; T += 4) {
        ninfer::DeviceBuffer chunk_input(2'560ULL * T * sizeof(std::uint16_t));
        ninfer::DeviceBuffer chunk_output(2'560ULL * T * sizeof(std::uint16_t));
        ninfer::DeviceBuffer chunk_conv_states(10'240ULL * 3 * 2 * sizeof(std::uint16_t));
        ninfer::DeviceBuffer chunk_ssm_states(128ULL * 128 * 48 * 2 * sizeof(float));

        ninfer::DeviceBuffer seq_output(2'560ULL * T * sizeof(std::uint16_t));
        ninfer::DeviceBuffer seq_conv_states(10'240ULL * 3 * 2 * sizeof(std::uint16_t));
        ninfer::DeviceBuffer seq_ssm_states(128ULL * 128 * 48 * 2 * sizeof(float));

        std::vector<std::uint16_t> h_input(2'560ULL * T);
        std::uint32_t rng = 987654321U + static_cast<std::uint32_t>(T);
        for (auto& v : h_input) {
            rng = rng * 1664525U + 1013904223U;
            const float r = static_cast<float>((rng & 0xFFFFU)) / 65536.0F * 0.2F - 0.1F;
            const std::uint32_t bits = std::bit_cast<std::uint32_t>(r);
            v = static_cast<std::uint16_t>((bits + 0x7FFFU + ((bits >> 16U) & 1U)) >> 16U);
        }
        chunk_input.copy_from_host(h_input.data(), h_input.size() * sizeof(std::uint16_t));

        chunk_conv_states.fill(0);
        chunk_ssm_states.fill(0);
        seq_conv_states.fill(0);
        seq_ssm_states.fill(0);
        device.synchronize(); // Ensure legacy stream copies order against device.stream

        ninfer::Tensor chunk_in_t(chunk_input.p, ninfer::DType::BF16, {2'560, T});
        ninfer::Tensor chunk_out_t(chunk_output.p, ninfer::DType::BF16, {2'560, T});
        ninfer::Tensor chunk_conv_t(chunk_conv_states.p, ninfer::DType::BF16, {10'240, 3, 2});
        ninfer::Tensor chunk_ssm_t(chunk_ssm_states.p, ninfer::DType::FP32, {128, 128, 48, 2});

        // 1. Run T-wide prefill chunk
        flash_next_gdn_prefill_chunk(chunk_in_t, weights, 0, 1, chunk_conv_t, chunk_ssm_t,
                                     workspace, chunk_out_t, device.stream);
        device.synchronize();

        // 2. Run sequential decode
        ninfer::Tensor seq_conv_t(seq_conv_states.p, ninfer::DType::BF16, {10'240, 3, 2});
        ninfer::Tensor seq_ssm_t(seq_ssm_states.p, ninfer::DType::FP32, {128, 128, 48, 2});
        std::int32_t cur_src = 0;
        std::int32_t cur_dst = 1;
        ninfer::DeviceBuffer tok_src(sizeof(std::int32_t));
        ninfer::DeviceBuffer tok_dst(sizeof(std::int32_t));
        ninfer::Tensor tok_src_t(tok_src.p, ninfer::DType::I32, {1});
        ninfer::Tensor tok_dst_t(tok_dst.p, ninfer::DType::I32, {1});

        for (int t = 0; t < T; ++t) {
            tok_src.copy_from_host(&cur_src, sizeof(std::int32_t));
            tok_dst.copy_from_host(&cur_dst, sizeof(std::int32_t));
            device.synchronize(); // Ensure legacy stream copies order against device.stream

            ninfer::Tensor tok_in(
                static_cast<std::uint16_t*>(chunk_input.p) + static_cast<std::size_t>(t) * 2'560,
                ninfer::DType::BF16, {2'560, 1});
            ninfer::Tensor tok_out(
                static_cast<std::uint16_t*>(seq_output.p) + static_cast<std::size_t>(t) * 2'560,
                ninfer::DType::BF16, {2'560, 1});

            flash_next_gdn_decode(tok_in, weights, tok_src_t, tok_dst_t, seq_conv_t, seq_ssm_t,
                                  workspace, tok_out, device.stream);
            device.synchronize();

            std::swap(cur_src, cur_dst);
        }

        // 3. Compare outputs
        std::vector<std::uint16_t> h_chunk_out(2'560ULL * T);
        std::vector<std::uint16_t> h_seq_out(2'560ULL * T);
        chunk_output.copy_to_host(h_chunk_out.data(), h_chunk_out.size() * sizeof(std::uint16_t));
        seq_output.copy_to_host(h_seq_out.data(), h_seq_out.size() * sizeof(std::uint16_t));

        double out_diff = 0.0, out_ref = 0.0;
        int out_nan = 0;
        for (std::size_t i = 0; i < h_chunk_out.size(); ++i) {
            const float a = bf16_to_float(h_chunk_out[i]);
            const float b = bf16_to_float(h_seq_out[i]);
            if (!std::isfinite(a) || !std::isfinite(b)) {
                out_nan++;
                continue;
            }
            out_diff += (a - b) * (a - b);
            out_ref += b * b;
        }
        if (out_nan > 0) {
            std::cerr << "FAIL: GDN chunk T=" << T << " output had " << out_nan << " non-finite elements\n";
            return 1;
        }
        if (out_ref <= 0.0) {
            std::cerr << "FAIL: GDN chunk T=" << T << " output reference was vacuous (out_ref <= 0)\n";
            return 1;
        }
        const double tol = T >= 16 ? 5e-2 : 1e-2;
        const double out_rel_l2 = std::sqrt(out_diff / out_ref);
        if (out_rel_l2 > tol) {
            std::cerr << "FAIL: GDN chunk T=" << T << " output rel-L2=" << out_rel_l2 << " > " << tol << "\n";
            return 1;
        }

        // 4. Compare final SSM state (slot 1 for chunk vs cur_src for sequential)
        constexpr std::size_t ssm_slot_floats = 128ULL * 128 * 48;
        std::vector<float> h_chunk_ssm(ssm_slot_floats * 2);
        std::vector<float> h_seq_ssm(ssm_slot_floats * 2);
        chunk_ssm_states.copy_to_host(h_chunk_ssm.data(), h_chunk_ssm.size() * sizeof(float));
        seq_ssm_states.copy_to_host(h_seq_ssm.data(), h_seq_ssm.size() * sizeof(float));

        double ssm_diff = 0.0, ssm_ref = 0.0;
        int ssm_nan = 0;
        for (std::size_t i = 0; i < ssm_slot_floats; ++i) {
            const float a = h_chunk_ssm[1 * ssm_slot_floats + i];
            const float b = h_seq_ssm[static_cast<std::size_t>(cur_src) * ssm_slot_floats + i];
            if (!std::isfinite(a) || !std::isfinite(b)) {
                ssm_nan++;
                continue;
            }
            ssm_diff += (a - b) * (a - b);
            ssm_ref += b * b;
        }
        if (ssm_nan > 0) {
            std::cerr << "FAIL: GDN chunk T=" << T << " SSM state had " << ssm_nan << " non-finite elements\n";
            return 1;
        }
        if (ssm_ref <= 0.0) {
            std::cerr << "FAIL: GDN chunk T=" << T << " SSM reference was vacuous (ssm_ref <= 0)\n";
            return 1;
        }
        const double ssm_rel_l2 = std::sqrt(ssm_diff / ssm_ref);
        if (ssm_rel_l2 > tol) {
            std::cerr << "FAIL: GDN chunk T=" << T << " SSM state rel-L2=" << ssm_rel_l2 << " > " << tol << "\n";
            return 1;
        }

        // 5. Compare final Conv state
        constexpr std::size_t conv_slot_u16 = 10'240ULL * 3;
        std::vector<std::uint16_t> h_chunk_conv(conv_slot_u16 * 2);
        std::vector<std::uint16_t> h_seq_conv(conv_slot_u16 * 2);
        chunk_conv_states.copy_to_host(h_chunk_conv.data(), h_chunk_conv.size() * sizeof(std::uint16_t));
        seq_conv_states.copy_to_host(h_seq_conv.data(), h_seq_conv.size() * sizeof(std::uint16_t));

        double conv_diff = 0.0, conv_ref = 0.0;
        int conv_nan = 0;
        for (std::size_t i = 0; i < conv_slot_u16; ++i) {
            const float a = bf16_to_float(h_chunk_conv[1 * conv_slot_u16 + i]);
            const float b = bf16_to_float(h_seq_conv[static_cast<std::size_t>(cur_src) * conv_slot_u16 + i]);
            if (!std::isfinite(a) || !std::isfinite(b)) {
                conv_nan++;
                continue;
            }
            conv_diff += (a - b) * (a - b);
            conv_ref += b * b;
        }
        if (conv_nan > 0) {
            std::cerr << "FAIL: GDN chunk T=" << T << " Conv state had " << conv_nan << " non-finite elements\n";
            return 1;
        }
        if (conv_ref <= 0.0) {
            std::cerr << "FAIL: GDN chunk T=" << T << " Conv reference was vacuous (conv_ref <= 0)\n";
            return 1;
        }
        const double conv_rel_l2 = std::sqrt(conv_diff / conv_ref);
        if (conv_rel_l2 > tol) {
            std::cerr << "FAIL: GDN chunk T=" << T << " Conv state rel-L2=" << conv_rel_l2 << " > " << tol << "\n";
            return 1;
        }
    }

        std::cout << "PASS: test_gdn\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "test_gdn exception: " << e.what() << "\n";
        return 1;
    }
}
