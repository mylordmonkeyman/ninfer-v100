#include "core/arena.h"
#include "core/device.h"
#include "targets/qwen3_8_flash_next/impl/ple_decode.h"

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
    out.payload_bytes   = static_cast<std::uint64_t>(rows) * columns * sizeof(std::uint16_t);
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

    constexpr std::int32_t batch       = 1;
    constexpr std::int32_t state_slots = 2;
    constexpr std::int32_t total_dims  = 10'240;
    constexpr std::int32_t stream_dims = 2'560;

    ninfer::DeviceBuffer hidden(total_dims * batch * sizeof(std::uint16_t));
    ninfer::DeviceBuffer gathered(stream_dims * batch * sizeof(std::uint16_t));
    ninfer::DeviceBuffer key_proj_storage(total_dims * stream_dims * sizeof(std::uint16_t));
    ninfer::DeviceBuffer val_proj_storage(stream_dims * stream_dims * sizeof(std::uint16_t));
    ninfer::DeviceBuffer query_norm(total_dims * sizeof(std::uint16_t));
    ninfer::DeviceBuffer key_norm(total_dims * sizeof(std::uint16_t));
    ninfer::DeviceBuffer conv_norm(total_dims * sizeof(std::uint16_t));
    ninfer::DeviceBuffer convolution(total_dims * 4 * sizeof(std::uint16_t));
    ninfer::DeviceBuffer source_slots(batch * sizeof(std::int32_t));
    ninfer::DeviceBuffer destination_slots(batch * sizeof(std::int32_t));
    ninfer::DeviceBuffer convolution_states(total_dims * 9 * state_slots * sizeof(std::uint16_t));
    ninfer::DeviceBuffer output(total_dims * batch * sizeof(std::uint16_t));

    // 1. Gathered embedding: e0 = 1.0, rest 0.0
    std::vector<std::uint16_t> host_gathered(stream_dims * batch, 0);
    host_gathered[0] = float_to_bf16(1.0F);
    gathered.copy_from_host(host_gathered.data(), host_gathered.size() * sizeof(std::uint16_t));

    // 2. Hidden: one nonzero coordinate per stream at local dim 1
    std::vector<std::uint16_t> host_hidden(total_dims * batch, 0);
    for (int s = 0; s < 4; ++s) { host_hidden[s * stream_dims + 1] = float_to_bf16(1.0F); }
    hidden.copy_from_host(host_hidden.data(), host_hidden.size() * sizeof(std::uint16_t));

    // 3. Key projection [10240, 2560]: one nonzero coordinate per stream at local dim 0
    std::vector<std::uint16_t> host_key_proj(total_dims * stream_dims, 0);
    for (int s = 0; s < 4; ++s) {
        const int row                        = s * stream_dims + 0;
        host_key_proj[row * stream_dims + 0] = float_to_bf16(1.0F);
    }
    key_proj_storage.copy_from_host(host_key_proj.data(),
                                    host_key_proj.size() * sizeof(std::uint16_t));

    // 4. Value projection [2560, 2560]: maps e0 to every value coordinate
    std::vector<std::uint16_t> host_val_proj(stream_dims * stream_dims, 0);
    for (int r = 0; r < stream_dims; ++r) {
        host_val_proj[r * stream_dims + 0] = float_to_bf16(1.0F);
    }
    val_proj_storage.copy_from_host(host_val_proj.data(),
                                    host_val_proj.size() * sizeof(std::uint16_t));

    // 5. Zero norm deltas
    query_norm.fill(0);
    key_norm.fill(0);
    conv_norm.fill(0);

    // 6. Convolution weights [10240, 4]: exercise all four taps
    std::vector<std::uint16_t> host_conv(total_dims * 4);
    for (int c = 0; c < total_dims; ++c) {
        host_conv[0 * total_dims + c] = float_to_bf16(0.25F); // tap t-9
        host_conv[1 * total_dims + c] = float_to_bf16(0.50F); // tap t-6
        host_conv[2 * total_dims + c] = float_to_bf16(0.75F); // tap t-3
        host_conv[3 * total_dims + c] = float_to_bf16(1.00F); // tap t
    }
    convolution.copy_from_host(host_conv.data(), host_conv.size() * sizeof(std::uint16_t));

    // 7. Slots: source = 0, destination = 1
    std::vector<std::int32_t> host_src_slots = {0};
    std::vector<std::int32_t> host_dst_slots = {1};
    source_slots.copy_from_host(host_src_slots.data(), sizeof(std::int32_t));
    destination_slots.copy_from_host(host_dst_slots.data(), sizeof(std::int32_t));

    // 8. Convolution states [10240, 9, 2]:
    // Fill source slot 0 with distinct represented history values
    std::vector<std::uint16_t> host_states(total_dims * 9 * state_slots, 0);
    for (int tap = 0; tap < 9; ++tap) {
        for (int c = 0; c < total_dims; ++c) {
            const float val =
                0.1F * static_cast<float>(tap + 1) + 0.00001F * static_cast<float>(c % 100);
            host_states[(0 * 9 + tap) * total_dims + c] = float_to_bf16(val);
        }
    }
    convolution_states.copy_from_host(host_states.data(),
                                      host_states.size() * sizeof(std::uint16_t));
    output.fill(0);

    // Build PleWeights
    PleWeights weights{};
    weights.key_projection   = bf16_weight(key_proj_storage.p, total_dims, stream_dims);
    weights.value_projection = bf16_weight(val_proj_storage.p, stream_dims, stream_dims);
    weights.query_norm       = ninfer::Tensor(query_norm.p, ninfer::DType::BF16, {total_dims});
    weights.key_norm         = ninfer::Tensor(key_norm.p, ninfer::DType::BF16, {total_dims});
    weights.conv_norm        = ninfer::Tensor(conv_norm.p, ninfer::DType::BF16, {total_dims});
    weights.convolution      = ninfer::Tensor(convolution.p, ninfer::DType::BF16, {total_dims, 4});

    ninfer::Tensor hidden_tensor(hidden.p, ninfer::DType::BF16, {total_dims, batch});
    ninfer::Tensor gathered_tensor(gathered.p, ninfer::DType::BF16, {stream_dims, batch});
    ninfer::Tensor src_slots_tensor(source_slots.p, ninfer::DType::I32, {batch});
    ninfer::Tensor dst_slots_tensor(destination_slots.p, ninfer::DType::I32, {batch});
    ninfer::Tensor states_tensor(convolution_states.p, ninfer::DType::BF16,
                                 {total_dims, 9, state_slots});
    ninfer::Tensor output_tensor(output.p, ninfer::DType::BF16, {total_dims, batch});

    ninfer::WorkspaceArena workspace(flash_next_ple_workspace_capacity_bytes(128));
    flash_next_ple_decode(hidden_tensor, gathered_tensor, weights, src_slots_tensor,
                          dst_slots_tensor, states_tensor, workspace, output_tensor, device.stream);
    device.synchronize();

    // Independent FP32 Oracle calculation with exact represented BF16 boundaries
    std::vector<float> oracle_output(total_dims * batch, 0.0F);
    std::vector<float> oracle_norm_gated(total_dims * batch, 0.0F);
    {
        // 1. Projections
        std::vector<float> proj_k(total_dims, 0.0F);
        std::vector<float> proj_v(stream_dims, 0.0F);
        for (int r = 0; r < total_dims; ++r) {
            float sum = 0.0F;
            for (int c = 0; c < stream_dims; ++c) {
                sum += bf16_to_float(host_key_proj[r * stream_dims + c]) *
                       bf16_to_float(host_gathered[c]);
            }
            proj_k[r] = bf16_to_float(float_to_bf16(sum));
        }
        for (int r = 0; r < stream_dims; ++r) {
            float sum = 0.0F;
            for (int c = 0; c < stream_dims; ++c) {
                sum += bf16_to_float(host_val_proj[r * stream_dims + c]) *
                       bf16_to_float(host_gathered[c]);
            }
            proj_v[r] = bf16_to_float(float_to_bf16(sum));
        }

        // 2. Gating and grouped norms per stream
        std::vector<float> gated(total_dims, 0.0F);
        for (int s = 0; s < 4; ++s) {
            float sq_q = 0.0F;
            float sq_k = 0.0F;
            for (int d = 0; d < stream_dims; ++d) {
                const float q_val = bf16_to_float(host_hidden[s * stream_dims + d]);
                const float k_val = proj_k[s * stream_dims + d];
                sq_q += q_val * q_val;
                sq_k += k_val * k_val;
            }
            const float r_rms_q = 1.0F / std::sqrt(sq_q / 2560.0F + 1.0e-6F);
            const float r_rms_k = 1.0F / std::sqrt(sq_k / 2560.0F + 1.0e-6F);

            float dot = 0.0F;
            for (int d = 0; d < stream_dims; ++d) {
                const float q_val = bf16_to_float(host_hidden[s * stream_dims + d]);
                const float k_val = proj_k[s * stream_dims + d];
                const float nq    = bf16_to_float(float_to_bf16(q_val * r_rms_q * (1.0F + 0.0F)));
                const float nk    = bf16_to_float(float_to_bf16(k_val * r_rms_k * (1.0F + 0.0F)));
                dot += nq * nk;
            }
            const float raw_gate = dot / std::sqrt(2560.0F);
            const float sign     = (raw_gate > 0.0F) ? 1.0F : ((raw_gate < 0.0F) ? -1.0F : 0.0F);
            const float gate     = sign * std::sqrt(std::max(std::abs(raw_gate), 1.0e-6F));
            const float sigmoid_gate = 1.0F / (1.0F + std::exp(-gate));

            float sq_g = 0.0F;
            for (int d = 0; d < stream_dims; ++d) {
                const float g_val          = bf16_to_float(float_to_bf16(sigmoid_gate * proj_v[d]));
                gated[s * stream_dims + d] = g_val;
                sq_g += g_val * g_val;
            }
            const float r_rms_g = 1.0F / std::sqrt(sq_g / 2560.0F + 1.0e-6F);
            for (int d = 0; d < stream_dims; ++d) {
                const float norm_g_val = bf16_to_float(
                    float_to_bf16(gated[s * stream_dims + d] * r_rms_g * (1.0F + 0.0F)));
                oracle_norm_gated[s * stream_dims + d] = norm_g_val;
            }
        }

        // 3. Convolution and residual addition
        for (int c = 0; c < total_dims; ++c) {
            const float h0 = bf16_to_float(host_states[(0 * 9 + 0) * total_dims + c]);
            const float h1 = bf16_to_float(host_states[(0 * 9 + 3) * total_dims + c]);
            const float h2 = bf16_to_float(host_states[(0 * 9 + 6) * total_dims + c]);
            const float h3 = oracle_norm_gated[c];

            const float w0        = bf16_to_float(host_conv[0 * total_dims + c]);
            const float w1        = bf16_to_float(host_conv[1 * total_dims + c]);
            const float w2        = bf16_to_float(host_conv[2 * total_dims + c]);
            const float w3        = bf16_to_float(host_conv[3 * total_dims + c]);
            const float conv      = h0 * w0 + h1 * w1 + h2 * w2 + h3 * w3;
            const float silu_conv = conv / (1.0F + std::exp(-conv));
            oracle_output[c]      = bf16_to_float(float_to_bf16(gated[c] + silu_conv));
        }
    }

    // Read back actual results
    std::vector<std::uint16_t> actual_output(total_dims * batch);
    output.copy_to_host(actual_output.data(), actual_output.size() * sizeof(std::uint16_t));

    std::vector<std::uint16_t> actual_states(total_dims * 9 * state_slots);
    convolution_states.copy_to_host(actual_states.data(),
                                    actual_states.size() * sizeof(std::uint16_t));

    int failures = 0;

    // Verify output against independent oracle
    for (int i = 0; i < total_dims; ++i) {
        const float act  = bf16_to_float(actual_output[i]);
        const float exp  = oracle_output[i];
        const float diff = std::abs(act - exp);
        const float tol  = std::max(0.01F, 0.01F * std::abs(exp));
        if (diff > tol) {
            std::cerr << "Mismatch at output[" << i << "]: actual=" << act << " expected=" << exp
                      << " (diff=" << diff << " > tol=" << tol << ")\n";
            failures++;
            if (failures > 10) break;
        }
    }

    // Verify destination history in slot 1 equals source slot 0 history [1..8] + current normalized
    // state
    for (int tap = 0; tap < 8; ++tap) {
        for (int c = 0; c < total_dims; ++c) {
            const std::uint16_t act = actual_states[(1 * 9 + tap) * total_dims + c];
            const std::uint16_t exp = host_states[(0 * 9 + (tap + 1)) * total_dims + c];
            if (act != exp) {
                std::cerr << "Destination state mismatch at tap " << tap << ", dim " << c
                          << ": expected 0x" << std::hex << exp << " got 0x" << act << std::dec
                          << "\n";
                failures++;
                if (failures > 20) break;
            }
        }
        if (failures > 20) break;
    }

    // Destination tap 8 must equal current normalized state
    for (int c = 0; c < total_dims; ++c) {
        const float act  = bf16_to_float(actual_states[(1 * 9 + 8) * total_dims + c]);
        const float exp  = oracle_norm_gated[c];
        const float diff = std::abs(act - exp);
        if (diff > 0.01F) {
            std::cerr << "Destination current state mismatch at dim " << c << ": actual=" << act
                      << " expected=" << exp << "\n";
            failures++;
            if (failures > 20) break;
        }
    }

    // Verify source slot 0 is completely unchanged
    for (int tap = 0; tap < 9; ++tap) {
        for (int c = 0; c < total_dims; ++c) {
            const std::uint16_t act = actual_states[(0 * 9 + tap) * total_dims + c];
            const std::uint16_t exp = host_states[(0 * 9 + tap) * total_dims + c];
            if (act != exp) {
                std::cerr << "Source state was modified at tap " << tap << ", dim " << c << "\n";
                failures++;
                if (failures > 20) break;
            }
        }
        if (failures > 20) break;
    }

    if (failures != 0) {
        std::cerr << "FAIL: " << failures << " errors in test_ple_decode token test\n";
        return 1;
    }

    // -------------------------------------------------------------
    // Test 2: ple_conv_inject_chunk against naive CPU stencil for T in {1, 2, 8, 9, 10, 64, 128}
    // -------------------------------------------------------------
    const std::vector<int> test_t_values = {1, 2, 8, 9, 10, 64, 128};
    for (int T : test_t_values) {
        ninfer::DeviceBuffer chunk_hidden(total_dims * T * sizeof(std::uint16_t));
        ninfer::DeviceBuffer chunk_gathered(stream_dims * T * sizeof(std::uint16_t));
        ninfer::DeviceBuffer chunk_states(total_dims * 9 * state_slots * sizeof(std::uint16_t));
        ninfer::DeviceBuffer chunk_output(total_dims * T * sizeof(std::uint16_t));

        std::vector<std::uint16_t> h_chunk_hidden(total_dims * T);
        std::vector<std::uint16_t> h_chunk_gathered(stream_dims * T);
        std::vector<std::uint16_t> h_chunk_states(total_dims * 9 * state_slots);

        // Fill with pseudo-random reproducible data
        std::uint32_t rng = 123456789U + static_cast<std::uint32_t>(T);
        auto next_rand    = [&]() -> float {
            rng = rng * 1664525U + 1013904223U;
            return static_cast<float>((rng & 0xFFFFU)) / 65536.0F * 2.0F - 1.0F;
        };

        for (auto& v : h_chunk_hidden) { v = float_to_bf16(next_rand() * 0.5F); }
        for (auto& v : h_chunk_gathered) { v = float_to_bf16(next_rand() * 0.5F); }
        for (auto& v : h_chunk_states) { v = float_to_bf16(next_rand() * 0.5F); }

        chunk_hidden.copy_from_host(h_chunk_hidden.data(), h_chunk_hidden.size() * sizeof(std::uint16_t));
        chunk_gathered.copy_from_host(h_chunk_gathered.data(),
                                      h_chunk_gathered.size() * sizeof(std::uint16_t));
        chunk_states.copy_from_host(h_chunk_states.data(),
                                    h_chunk_states.size() * sizeof(std::uint16_t));

        ninfer::Tensor hidden_t(chunk_hidden.p, ninfer::DType::BF16, {total_dims, T});
        ninfer::Tensor gathered_t(chunk_gathered.p, ninfer::DType::BF16, {stream_dims, T});
        ninfer::Tensor states_t(chunk_states.p, ninfer::DType::BF16, {total_dims, 9, state_slots});
        ninfer::Tensor output_t(chunk_output.p, ninfer::DType::BF16, {total_dims, T});

        flash_next_ple_prefill_chunk(hidden_t, gathered_t, weights, 0, 1, states_t, workspace,
                                     output_t, device.stream);
        device.synchronize();

        // Also run sequential token-by-token decode using duplicate initial states to verify parity
        ninfer::DeviceBuffer seq_states(total_dims * 9 * state_slots * sizeof(std::uint16_t));
        seq_states.copy_from_host(h_chunk_states.data(),
                                  h_chunk_states.size() * sizeof(std::uint16_t));
        ninfer::Tensor seq_states_t(seq_states.p, ninfer::DType::BF16,
                                    {total_dims, 9, state_slots});
        std::vector<std::uint16_t> seq_output_host(total_dims * T);

        std::int32_t cur_src = 0;
        std::int32_t cur_dst = 1;
        ninfer::DeviceBuffer tok_src(sizeof(std::int32_t));
        ninfer::DeviceBuffer tok_dst(sizeof(std::int32_t));
        ninfer::Tensor tok_src_t(tok_src.p, ninfer::DType::I32, {1});
        ninfer::Tensor tok_dst_t(tok_dst.p, ninfer::DType::I32, {1});

        for (int t = 0; t < T; ++t) {
            tok_src.copy_from_host(&cur_src, sizeof(std::int32_t));
            tok_dst.copy_from_host(&cur_dst, sizeof(std::int32_t));

            ninfer::Tensor tok_hidden(
                static_cast<std::uint16_t*>(chunk_hidden.p) + static_cast<std::size_t>(t) * total_dims,
                ninfer::DType::BF16, {total_dims, 1});
            ninfer::Tensor tok_gathered(static_cast<std::uint16_t*>(chunk_gathered.p) +
                                            static_cast<std::size_t>(t) * stream_dims,
                                        ninfer::DType::BF16, {stream_dims, 1});
            ninfer::Tensor tok_output(static_cast<std::uint16_t*>(output.p), ninfer::DType::BF16,
                                      {total_dims, 1});

            flash_next_ple_decode(tok_hidden, tok_gathered, weights, tok_src_t, tok_dst_t,
                                  seq_states_t, workspace, tok_output, device.stream);
            device.synchronize();

            output.copy_to_host(seq_output_host.data() + static_cast<std::size_t>(t) * total_dims,
                                total_dims * sizeof(std::uint16_t));
            device.synchronize();

            // Step slots
            std::swap(cur_src, cur_dst);
        }

        std::vector<std::uint16_t> chunk_act_out(total_dims * T);
        chunk_output.copy_to_host(chunk_act_out.data(), chunk_act_out.size() * sizeof(std::uint16_t));

        std::vector<std::uint16_t> chunk_act_states(total_dims * 9 * state_slots);
        chunk_states.copy_to_host(chunk_act_states.data(),
                                  chunk_act_states.size() * sizeof(std::uint16_t));

        std::vector<std::uint16_t> seq_act_states(total_dims * 9 * state_slots);
        seq_states.copy_to_host(seq_act_states.data(),
                                seq_act_states.size() * sizeof(std::uint16_t));

        // Check output parity
        double diff_norm = 0.0;
        double ref_norm  = 0.0;
        for (std::size_t i = 0; i < chunk_act_out.size(); ++i) {
            const float act  = bf16_to_float(chunk_act_out[i]);
            const float seq  = bf16_to_float(seq_output_host[i]);
            const float diff = act - seq;
            diff_norm += diff * diff;
            ref_norm += seq * seq;
        }
        const double rel_l2 = std::sqrt(diff_norm) / std::max(1e-6, std::sqrt(ref_norm));
        if (rel_l2 > 1e-3) {
            std::cerr << "FAIL: PLE chunk T=" << T << " output rel-L2=" << rel_l2 << " > 1e-3\n";
            failures++;
        }

        // Check final history state in destination slot (slot 1 for chunked) vs sequential final source
        for (int tap = 0; tap < 9; ++tap) {
            for (int c = 0; c < total_dims; ++c) {
                const std::uint16_t act = chunk_act_states[(1 * 9 + tap) * total_dims + c];
                const std::uint16_t seq = seq_act_states[(cur_src * 9 + tap) * total_dims + c];
                if (act != seq) {
                    std::cerr << "FAIL: PLE chunk T=" << T << " state mismatch at tap=" << tap
                              << " c=" << c << ": chunk=0x" << std::hex << act << " seq=0x" << seq
                              << std::dec << "\n";
                    failures++;
                    break;
                }
            }
            if (failures > 10) break;
        }
    }

    if (failures != 0) {
        std::cerr << "FAIL: " << failures << " errors in test_ple_decode\n";
        return 1;
    }

    std::cout << "PASS: test_ple_decode\n";
    return 0;
}
