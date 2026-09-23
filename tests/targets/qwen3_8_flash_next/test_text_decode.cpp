#include "core/arena.h"
#include "core/device.h"
#include "targets/qwen3_8_flash_next/impl/text_decode.h"
#include "targets/qwen3_8_flash_next/impl/text_decode_kernels.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
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

int test_layer_mapping() {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    std::size_t qsa_count = 0;
    std::size_t gdn_count = 0;

    std::vector<std::size_t> qsa_layers;
    std::vector<std::size_t> gdn_layers;

    for (std::size_t layer = 0; layer < 48; ++layer) {
        if (is_qsa_layer(layer)) {
            const std::size_t ordinal = qsa_ordinal(layer);
            if (ordinal != qsa_count) {
                std::cerr << "QSA ordinal mismatch at layer " << layer << ": expected " << qsa_count
                          << " got " << ordinal << "\n";
                return 1;
            }
            qsa_layers.push_back(layer);
            ++qsa_count;
        } else {
            const std::size_t ordinal = gdn_ordinal(layer);
            if (ordinal != gdn_count) {
                std::cerr << "GDN ordinal mismatch at layer " << layer << ": expected " << gdn_count
                          << " got " << ordinal << "\n";
                return 1;
            }
            gdn_layers.push_back(layer);
            ++gdn_count;
        }
    }

    if (qsa_count != 12) {
        std::cerr << "Expected exactly 12 QSA layers, got " << qsa_count << "\n";
        return 1;
    }
    if (gdn_count != 36) {
        std::cerr << "Expected exactly 36 GDN layers, got " << gdn_count << "\n";
        return 1;
    }

    const std::vector<std::size_t> expected_qsa = {3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47};
    if (qsa_layers != expected_qsa) {
        std::cerr << "QSA layers list mismatch\n";
        return 1;
    }

    std::cout << "PASS: test_layer_mapping\n";
    return 0;
}

int test_embedding_repeat(ninfer::DeviceContext& device) {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    constexpr std::int32_t batch       = 2;
    constexpr std::int32_t stream_dims = 2'560;
    constexpr std::int32_t total_dims  = 10'240;

    ninfer::DeviceBuffer embedding_buf(stream_dims * batch * sizeof(std::uint16_t));
    ninfer::DeviceBuffer hyper_hidden_buf(total_dims * batch * sizeof(std::uint16_t));

    std::vector<std::uint16_t> host_embedding(stream_dims * batch);
    for (std::size_t b = 0; b < batch; ++b) {
        for (std::size_t d = 0; d < stream_dims; ++d) {
            host_embedding[b * stream_dims + d] =
                float_to_bf16(static_cast<float>(b * 1000 + d + 1));
        }
    }
    embedding_buf.copy_from_host(host_embedding.data(),
                                 host_embedding.size() * sizeof(std::uint16_t));
    hyper_hidden_buf.fill(0);

    ninfer::Tensor embedding(embedding_buf.p, ninfer::DType::BF16, {stream_dims, batch});
    ninfer::Tensor hyper_hidden(hyper_hidden_buf.p, ninfer::DType::BF16, {total_dims, batch});

    repeat_embedding_to_hyper_streams(embedding, hyper_hidden, device.stream);
    device.synchronize();

    std::vector<std::uint16_t> host_hyper(total_dims * batch);
    hyper_hidden_buf.copy_to_host(host_hyper.data(), host_hyper.size() * sizeof(std::uint16_t));

    for (std::size_t b = 0; b < batch; ++b) {
        for (std::size_t s = 0; s < 4; ++s) {
            for (std::size_t d = 0; d < stream_dims; ++d) {
                const std::uint16_t act = host_hyper[b * total_dims + s * stream_dims + d];
                const std::uint16_t exp = host_embedding[b * stream_dims + d];
                if (act != exp) {
                    std::cerr << "Embedding repeat mismatch at batch " << b << " stream " << s
                              << " dim " << d << ": expected 0x" << std::hex << exp << " got 0x"
                              << act << std::dec << "\n";
                    return 1;
                }
            }
        }
    }

    std::cout << "PASS: test_embedding_repeat\n";
    return 0;
}

int test_state_validation(ninfer::DeviceContext& device) {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    constexpr std::int32_t state_slots    = 2;
    constexpr std::int32_t physical_pages = 4;
    constexpr std::int32_t logical_pages  = 4;
    constexpr std::int32_t table_rows     = 2;

    ninfer::DeviceBuffer ple_conv_buf(10'240ULL * 9 * state_slots * sizeof(std::uint16_t));
    ninfer::DeviceBuffer gdn_conv_buf(10'240ULL * 3 * state_slots * sizeof(std::uint16_t));
    ninfer::DeviceBuffer gdn_ssm_buf(128ULL * 128 * 48 * state_slots * sizeof(float));
    ninfer::DeviceBuffer indexer_keys_buf(128ULL * 64 * physical_pages * sizeof(std::uint16_t));
    ninfer::DeviceBuffer indexer_tables_buf(logical_pages * table_rows * sizeof(std::int32_t));
    ninfer::DeviceBuffer raw_keys_buf(128ULL * 4 * state_slots * sizeof(std::uint16_t));
    ninfer::DeviceBuffer raw_pos_buf(3ULL * 4 * state_slots * sizeof(std::int32_t));
    ninfer::DeviceBuffer att_k_buf(256ULL * 64 * 2 * physical_pages * sizeof(std::uint16_t));
    ninfer::DeviceBuffer att_v_buf(256ULL * 64 * 2 * physical_pages * sizeof(std::uint16_t));

    FlashNextDecodeStateView state{};
    state.ple_convolution_states =
        ninfer::Tensor(ple_conv_buf.p, ninfer::DType::BF16, {10'240, 9, state_slots});

    for (std::size_t i = 0; i < kGdnLayers; ++i) {
        state.gdn_convolution_states[i] =
            ninfer::Tensor(gdn_conv_buf.p, ninfer::DType::BF16, {10'240, 3, state_slots});
        state.gdn_ssm_states[i] =
            ninfer::Tensor(gdn_ssm_buf.p, ninfer::DType::FP32, {128, 128, 48, state_slots});
    }

    for (std::size_t i = 0; i < kFullAttentionLayers; ++i) {
        state.qsa_indexer_caches[i] = {
            .block_keys =
                ninfer::Tensor(indexer_keys_buf.p, ninfer::DType::BF16, {128, 64, physical_pages}),
            .block_tables = ninfer::Tensor(indexer_tables_buf.p, ninfer::DType::I32,
                                           {logical_pages, table_rows}),
            .raw_keys = ninfer::Tensor(raw_keys_buf.p, ninfer::DType::BF16, {128, 4, state_slots}),
            .raw_positions = ninfer::Tensor(raw_pos_buf.p, ninfer::DType::I32, {3, 4, state_slots}),
        };
        state.qsa_attention_caches[i] = {
            .key_pages =
                ninfer::Tensor(att_k_buf.p, ninfer::DType::BF16, {256, 64, 2, physical_pages}),
            .value_pages =
                ninfer::Tensor(att_v_buf.p, ninfer::DType::BF16, {256, 64, 2, physical_pages}),
            .block_tables = ninfer::Tensor(indexer_tables_buf.p, ninfer::DType::I32,
                                           {logical_pages, table_rows}),
        };
    }

    // 1. Valid state should succeed
    try {
        validate_flash_next_decode_state(state, state_slots);
    } catch (const std::exception& e) {
        std::cerr << "Valid state threw unexpectedly: " << e.what() << "\n";
        return 1;
    }

    // 2. Reject state_slots <= 0
    bool caught = false;
    try {
        validate_flash_next_decode_state(state, 0);
    } catch (const std::invalid_argument&) { caught = true; }
    if (!caught) {
        std::cerr << "Failed to reject state_slots = 0\n";
        return 1;
    }

    // 3. Reject invalid PLE convolution state dtype
    auto bad_ple                         = state;
    bad_ple.ple_convolution_states.dtype = ninfer::DType::FP32;
    caught                               = false;
    try {
        validate_flash_next_decode_state(bad_ple, state_slots);
    } catch (const std::invalid_argument&) { caught = true; }
    if (!caught) {
        std::cerr << "Failed to reject bad PLE dtype\n";
        return 1;
    }

    // 4. Reject invalid GDN state shape
    auto bad_gdn                            = state;
    bad_gdn.gdn_convolution_states[5].ne[0] = 5'120;
    caught                                  = false;
    try {
        validate_flash_next_decode_state(bad_gdn, state_slots);
    } catch (const std::invalid_argument&) { caught = true; }
    if (!caught) {
        std::cerr << "Failed to reject bad GDN shape\n";
        return 1;
    }

    // 5. Reject null GDN SSM pointer
    auto bad_ssm                    = state;
    bad_ssm.gdn_ssm_states[10].data = nullptr;
    caught                          = false;
    try {
        validate_flash_next_decode_state(bad_ssm, state_slots);
    } catch (const std::invalid_argument&) { caught = true; }
    if (!caught) {
        std::cerr << "Failed to reject null GDN SSM data\n";
        return 1;
    }

    // 6. Reject invalid QSA indexer block_keys shape
    auto bad_idx                                   = state;
    bad_idx.qsa_indexer_caches[3].block_keys.ne[0] = 64;
    caught                                         = false;
    try {
        validate_flash_next_decode_state(bad_idx, state_slots);
    } catch (const std::invalid_argument&) { caught = true; }
    if (!caught) {
        std::cerr << "Failed to reject bad QSA indexer block_keys\n";
        return 1;
    }

    // 7. Reject invalid QSA attention key_pages shape
    auto bad_att                                    = state;
    bad_att.qsa_attention_caches[7].key_pages.ne[2] = 1;
    caught                                          = false;
    try {
        validate_flash_next_decode_state(bad_att, state_slots);
    } catch (const std::invalid_argument&) { caught = true; }
    if (!caught) {
        std::cerr << "Failed to reject bad QSA attention key_pages\n";
        return 1;
    }

    std::cout << "PASS: test_state_validation\n";
    return 0;
}

int test_workspace_and_validation() {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    // 1. Capacity checks
    const std::size_t cap1 = flash_next_text_decode_workspace_capacity_bytes(512, 1);
    const std::size_t cap8 = flash_next_text_decode_workspace_capacity_bytes(512, 8);
    if (cap1 == 0 || cap8 == 0 || cap8 <= cap1) {
        std::cerr << "Workspace capacity error: cap1=" << cap1 << ", cap8=" << cap8 << "\n";
        return 1;
    }

    // 2. Envelope boundary failures
    bool caught = false;
    try {
        (void)flash_next_text_decode_workspace_capacity_bytes(0, 1);
    } catch (const std::invalid_argument&) { caught = true; }
    if (!caught) {
        std::cerr << "Failed to reject maximum_blocks = 0\n";
        return 1;
    }

    caught = false;
    try {
        (void)flash_next_text_decode_workspace_capacity_bytes(512, 0);
    } catch (const std::invalid_argument&) { caught = true; }
    if (!caught) {
        std::cerr << "Failed to reject batch = 0\n";
        return 1;
    }

    caught = false;
    try {
        (void)flash_next_text_decode_workspace_capacity_bytes(512, 9);
    } catch (const std::invalid_argument&) { caught = true; }
    if (!caught) {
        std::cerr << "Failed to reject batch = 9 (outside 1..8)\n";
        return 1;
    }

    std::cout << "PASS: test_workspace_and_validation\n";
    return 0;
}

} // namespace

int main() {
    int device_count              = 0;
    const cudaError_t count_error = cudaGetDeviceCount(&device_count);
    if (cuda_unavailable(count_error) || device_count == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    CUDA_CHECK(count_error);

    ninfer::DeviceContext device(0);

    if (test_layer_mapping() != 0) return 1;
    if (test_embedding_repeat(device) != 0) return 1;
    if (test_workspace_and_validation() != 0) return 1;
    if (test_state_validation(device) != 0) return 1;

    std::cout << "OK Flash-Next Text Decode\n";
    return 0;
}
