#include "core/arena.h"
#include "core/device.h"
#include "targets/qwen3_8_flash_next/impl/lane_ledger.h"
#include "targets/qwen3_8_flash_next/impl/model_view.h"
#include "targets/qwen3_8_flash_next/impl/ple_index.h"
#include "targets/qwen3_8_flash_next/impl/ple_table.h"
#include "targets/qwen3_8_flash_next/impl/runtime_plan.h"
#include "targets/qwen3_8_flash_next/impl/runtime_state.h"
#include "targets/qwen3_8_flash_next/impl/qsa_indexer_workspace.h"
#include "targets/qwen3_8_flash_next/impl/stage_ledger.h"
#include "targets/qwen3_8_flash_next/impl/text_decode.h"
#include "targets/qwen3_8_flash_next/impl/text_decode_workspace.h"
#include "targets/qwen3_8_flash_next/impl/text_executor.h"
#include "tools/reference/qwen3_8_flash_next/oracle_chat23.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <random>

namespace {

bool cuda_unavailable(cudaError_t error) {
    return error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver;
}

// Default remains 3. NINFER_PREFILL_BENCH_ITERS=<n> selects the counted samples after
// discarding the first cudaEvent-timed chunk. Invalid/empty env keeps the default.
int prefill_bench_iters() {
    const char* env = std::getenv("NINFER_PREFILL_BENCH_ITERS");
    if (env == nullptr || env[0] == '\0') { return 3; }
    char* end = nullptr;
    const long parsed = std::strtol(env, &end, 10);
    if (end == env || parsed < 1 || parsed > 10'000) { return 3; }
    return static_cast<int>(parsed);
}

int prefill_bench_warmup_s(bool interleaved_ab) {
    const char* env = std::getenv("NINFER_PREFILL_BENCH_WARMUP_S");
    if (env == nullptr || env[0] == '\0') { return interleaved_ab ? 60 : 0; }
    char* end = nullptr;
    const long parsed = std::strtol(env, &end, 10);
    if (end == env || parsed < 0 || parsed > 600) { return interleaved_ab ? 60 : 0; }
    return static_cast<int>(parsed);
}

bool prefill_bench_ab_requested(bool mode_ab) {
    if (mode_ab) { return true; }
    const char* env = std::getenv("NINFER_PREFILL_BENCH_AB");
    return env != nullptr && std::strcmp(env, "1") == 0;
}

void set_residual_splitk_env(int split_k) {
    const char* value = split_k == 1 ? "1" : "4";
#ifdef _WIN32
    (void)_putenv_s("NINFER_FLASH_NEXT_RESIDUAL_SPLITK", value);
#else
    (void)setenv("NINFER_FLASH_NEXT_RESIDUAL_SPLITK", value, 1);
#endif
}

void set_prefill_host_sync_env(bool restore_old) {
    const char* value = restore_old ? "1" : "0";
#ifdef _WIN32
    (void)_putenv_s("NINFER_FLASH_NEXT_PREFILL_HOST_SYNC", value);
#else
    (void)setenv("NINFER_FLASH_NEXT_PREFILL_HOST_SYNC", value, 1);
#endif
}

void set_moe_shared_mma_env(bool enable_mma) {
    const char* value = enable_mma ? "1" : "0";
#ifdef _WIN32
    (void)_putenv_s("NINFER_FLASH_NEXT_MOE_SHARED_MMA", value);
#else
    (void)setenv("NINFER_FLASH_NEXT_MOE_SHARED_MMA", value, 1);
#endif
}

void set_qsa_mma_sched_env(bool use_new) {
    const char* value = use_new ? "new" : "old";
#ifdef _WIN32
    (void)_putenv_s("NINFER_FLASH_NEXT_QSA_MMA_SCHED", value);
#else
    (void)setenv("NINFER_FLASH_NEXT_QSA_MMA_SCHED", value, 1);
#endif
}

void set_qsa_prefill_mma_env(bool enable) {
    const char* value = enable ? "1" : "0";
#ifdef _WIN32
    (void)_putenv_s("NINFER_FLASH_NEXT_QSA_PREFILL_MMA", value);
#else
    (void)setenv("NINFER_FLASH_NEXT_QSA_PREFILL_MMA", value, 1);
#endif
}

void set_moe_staging_env(bool enable_new) {
    const char* value = enable_new ? "new" : "old";
#ifdef _WIN32
    (void)_putenv_s("NINFER_FLASH_NEXT_MOE_STAGING", value);
#else
    (void)setenv("NINFER_FLASH_NEXT_MOE_STAGING", value, 1);
#endif
}

struct PrefillBenchStats {
    float min_ms    = 0.0F;
    float median_ms = 0.0F;
    float max_ms    = 0.0F;
    float mean_ms   = 0.0F;
};

PrefillBenchStats prefill_bench_stats(std::vector<float> samples) {
    PrefillBenchStats out{};
    if (samples.empty()) { return out; }
    std::sort(samples.begin(), samples.end());
    out.min_ms = samples.front();
    out.max_ms = samples.back();
    const int n = static_cast<int>(samples.size());
    if ((n % 2) == 1) {
        out.median_ms = samples[static_cast<std::size_t>(n / 2)];
    } else {
        out.median_ms = 0.5F * (samples[static_cast<std::size_t>(n / 2 - 1)] +
                                samples[static_cast<std::size_t>(n / 2)]);
    }
    double sum = 0.0;
    for (float sample : samples) { sum += static_cast<double>(sample); }
    out.mean_ms = static_cast<float>(sum / static_cast<double>(n));
    return out;
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

struct SyntheticFlashNextModel {
    ninfer::DeviceBuffer big_bf16_buf;
    ninfer::DeviceBuffer norm_bf16_buf;
    ninfer::DeviceBuffer gdn_a_log_buf;
    ninfer::DeviceBuffer gdn_dt_bias_buf;
    ninfer::DeviceBuffer gdn_conv_buf;
    ninfer::DeviceBuffer ple_conv_buf;
    ninfer::DeviceBuffer shared_gate_weight_buf;
    ninfer::DeviceBuffer inject_buf;

    ninfer::DeviceBuffer fp8_qkvz_buf;
    ninfer::DeviceBuffer fp8_qgkv_buf;
    ninfer::DeviceBuffer fp8_out_buf;

    ninfer::DeviceBuffer big_nvfp4_gate_codes_buf;
    ninfer::DeviceBuffer big_nvfp4_gate_scales_buf;
    ninfer::DeviceBuffer big_nvfp4_down_codes_buf;
    ninfer::DeviceBuffer big_nvfp4_down_scales_buf;
    ninfer::DeviceBuffer big_divisors_buf;
    ninfer::DeviceBuffer per_layer_bf16_buf;

    std::vector<std::byte> ple_table_data;
    ninfer::targets::qwen3_8_flash_next::detail::TextModelView view;
};

SyntheticFlashNextModel make_synthetic_model(ninfer::DeviceContext& device) {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    SyntheticFlashNextModel model;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist_embed(-0.05f, 0.05f);
    std::uniform_real_distribution<float> dist_bf16(-0.02f, 0.02f);
    std::uniform_real_distribution<float> dist_router(-0.10f, 0.10f);
    std::uniform_real_distribution<float> dist_inject(-0.05f, 0.05f);
    std::uniform_real_distribution<float> dist_norm(0.99f, 1.01f);

    // 1. Generic BF16 weights (token_embedding, output_head, linear projections)
    constexpr std::uint64_t kOutputHeadBytes = 248'320ULL * 2'560 * 2;
    model.big_bf16_buf = ninfer::DeviceBuffer(kOutputHeadBytes);
    constexpr std::size_t kChunkFloats = 2'560 * 1024;
    std::vector<std::uint16_t> h_bf16(kChunkFloats);
    for (std::size_t off = 0; off < kOutputHeadBytes; off += h_bf16.size() * sizeof(std::uint16_t)) {
        for (auto& v : h_bf16) { v = float_to_bf16(dist_embed(rng)); }
        std::size_t chunk = std::min<std::size_t>(h_bf16.size() * sizeof(std::uint16_t), kOutputHeadBytes - off);
        model.big_bf16_buf.copy_from_host(h_bf16.data(), chunk, off);
    }

    // 1b. Per-layer distinct weights across all 48 layers (hyper projections, router, shared expert)
    constexpr std::uint64_t kAttnDownFloats   = 320ULL * 10'240;
    constexpr std::uint64_t kAttnUpFloats     = 10'240ULL * 320;
    constexpr std::uint64_t kMlpDownFloats    = 320ULL * 10'240;
    constexpr std::uint64_t kMlpUpFloats      = 10'240ULL * 320;
    constexpr std::uint64_t kRouterFloats     = 512ULL * 2'560;
    constexpr std::uint64_t kSharedDownFloats = 2'560ULL * 640;
    constexpr std::uint64_t kSharedGateFloats = 640ULL * 2'560;
    constexpr std::uint64_t kSharedUpFloats   = 640ULL * 2'560;

    constexpr std::uint64_t kLayerTotalFloats =
        kAttnDownFloats + kAttnUpFloats + kMlpDownFloats + kMlpUpFloats +
        kRouterFloats + kSharedDownFloats + kSharedGateFloats + kSharedUpFloats;

    constexpr std::uint64_t kAllLayersFloats = 48ULL * kLayerTotalFloats;
    constexpr std::uint64_t kAllLayersBytes  = kAllLayersFloats * sizeof(std::uint16_t);
    model.per_layer_bf16_buf = ninfer::DeviceBuffer(kAllLayersBytes);
    std::vector<std::uint16_t> h_layer(kLayerTotalFloats);
    std::uniform_real_distribution<float> dist_mix(-0.025f, 0.025f);
    std::uniform_real_distribution<float> dist_moe_router(-0.10f, 0.10f);
    std::uniform_real_distribution<float> dist_shared(-0.025f, 0.025f);

    for (std::size_t l = 0; l < 48; ++l) {
        std::size_t cur = 0;
        for (std::size_t i = 0; i < kAttnDownFloats + kAttnUpFloats + kMlpDownFloats + kMlpUpFloats; ++i) {
            h_layer[cur++] = float_to_bf16(dist_mix(rng));
        }
        for (std::size_t i = 0; i < kRouterFloats; ++i) {
            h_layer[cur++] = float_to_bf16(dist_moe_router(rng));
        }
        for (std::size_t i = 0; i < kSharedDownFloats + kSharedGateFloats + kSharedUpFloats; ++i) {
            h_layer[cur++] = float_to_bf16(dist_shared(rng));
        }
        model.per_layer_bf16_buf.copy_from_host(
            h_layer.data(), kLayerTotalFloats * sizeof(std::uint16_t), l * kLayerTotalFloats * sizeof(std::uint16_t));
    }

    // 2. RMSNorm weights (~1.0)
    std::vector<std::uint16_t> h_norm(10'240);
    for (auto& v : h_norm) { v = float_to_bf16(dist_norm(rng)); }
    model.norm_bf16_buf = ninfer::DeviceBuffer(10'240 * sizeof(std::uint16_t));
    model.norm_bf16_buf.copy_from_host(h_norm.data(), h_norm.size() * sizeof(std::uint16_t));

    // 3. GDN structural parameters
    std::vector<std::uint16_t> h_a_log(48);
    for (auto& v : h_a_log) { v = float_to_bf16(-1.0f); }
    model.gdn_a_log_buf = ninfer::DeviceBuffer(48 * sizeof(std::uint16_t));
    model.gdn_a_log_buf.copy_from_host(h_a_log.data(), h_a_log.size() * sizeof(std::uint16_t));

    std::vector<std::uint16_t> h_dt_bias(48);
    for (auto& v : h_dt_bias) { v = float_to_bf16(0.05f); }
    model.gdn_dt_bias_buf = ninfer::DeviceBuffer(48 * sizeof(std::uint16_t));
    model.gdn_dt_bias_buf.copy_from_host(h_dt_bias.data(), h_dt_bias.size() * sizeof(std::uint16_t));

    std::vector<std::uint16_t> h_gdn_conv(10'240 * 4);
    for (auto& v : h_gdn_conv) { v = float_to_bf16(dist_inject(rng)); }
    model.gdn_conv_buf = ninfer::DeviceBuffer(10'240 * 4 * sizeof(std::uint16_t));
    model.gdn_conv_buf.copy_from_host(h_gdn_conv.data(), h_gdn_conv.size() * sizeof(std::uint16_t));

    // 4. PLE convolution weights (zero-mean to prevent early layer routing collapse)
    std::vector<std::uint16_t> h_ple_conv(10'240 * 4);
    for (auto& v : h_ple_conv) { v = float_to_bf16(dist_inject(rng)); }
    model.ple_conv_buf = ninfer::DeviceBuffer(10'240 * 4 * sizeof(std::uint16_t));
    model.ple_conv_buf.copy_from_host(h_ple_conv.data(), h_ple_conv.size() * sizeof(std::uint16_t));

    // 5. Shared gate weight and hyper injection (zero-mean to prevent DC bias drift)
    std::vector<std::uint16_t> h_sgw(2'560);
    for (auto& v : h_sgw) { v = float_to_bf16(dist_inject(rng)); }
    model.shared_gate_weight_buf = ninfer::DeviceBuffer(2'560 * sizeof(std::uint16_t));
    model.shared_gate_weight_buf.copy_from_host(h_sgw.data(), h_sgw.size() * sizeof(std::uint16_t));

    std::vector<std::uint16_t> h_inject(4 * 10'240);
    for (auto& v : h_inject) { v = float_to_bf16(dist_inject(rng)); }
    model.inject_buf = ninfer::DeviceBuffer(4 * 10'240 * sizeof(std::uint16_t));
    model.inject_buf.copy_from_host(h_inject.data(), h_inject.size() * sizeof(std::uint16_t));

    // 6. FP8 weights with dedicated code areas and FP32 row scales
    auto init_fp8_buf = [&](ninfer::DeviceBuffer& buf, std::int32_t rows, std::int32_t cols, float scale_val) {
        const std::uint64_t codes_bytes = static_cast<std::uint64_t>(rows) * cols;
        const std::uint64_t scale_off   = (codes_bytes + 255U) & ~255ULL;
        const std::uint64_t total_bytes = scale_off + static_cast<std::uint64_t>(rows) * sizeof(float);
        buf = ninfer::DeviceBuffer(total_bytes);

        std::vector<std::uint8_t> h_codes(codes_bytes);
        for (std::size_t i = 0; i < codes_bytes; ++i) {
            const auto s = (rng() % 2) ? 0x80U : 0x00U;
            h_codes[i] = static_cast<std::uint8_t>(s | (0x10 + (rng() % 32)));
        }
        buf.copy_from_host(h_codes.data(), codes_bytes, 0);

        std::vector<float> h_scales(rows, scale_val);
        buf.copy_from_host(h_scales.data(), rows * sizeof(float), scale_off);
    };

    init_fp8_buf(model.fp8_qkvz_buf, 16'384, 2'560, 1.0f / std::sqrt(2'560.0f));
    init_fp8_buf(model.fp8_qgkv_buf, 13'312, 2'560, 1.0f / std::sqrt(2'560.0f));
    init_fp8_buf(model.fp8_out_buf, 2'560, 6'144, 1.0f / std::sqrt(6'144.0f));

    // 7. NVFP4 Expert Banks (zero-mean symmetric signs to prevent directional drift)
    constexpr std::uint64_t gate_code_bytes_per_expert  = 1'280ULL * 2'560 / 2;
    constexpr std::uint64_t gate_scale_bytes_per_expert = 1'280ULL * 2'560 / 16;
    constexpr std::uint64_t down_code_bytes_per_expert  = 2'560ULL * 640 / 2;
    constexpr std::uint64_t down_scale_bytes_per_expert = 2'560ULL * 640 / 16;

    model.big_nvfp4_gate_codes_buf  = ninfer::DeviceBuffer(512 * gate_code_bytes_per_expert);
    model.big_nvfp4_gate_scales_buf = ninfer::DeviceBuffer(512 * gate_scale_bytes_per_expert);
    model.big_nvfp4_down_codes_buf  = ninfer::DeviceBuffer(512 * down_code_bytes_per_expert);
    model.big_nvfp4_down_scales_buf = ninfer::DeviceBuffer(512 * down_scale_bytes_per_expert);
    model.big_divisors_buf          = ninfer::DeviceBuffer(512 * sizeof(float));

    std::vector<std::uint8_t> h_fp4(1024 * 1024);
    for (auto& b : h_fp4) {
        const auto s_low    = (rng() % 2) ? 0x8U : 0x0U;
        const auto s_high   = (rng() % 2) ? 0x8U : 0x0U;
        const auto val_low  = static_cast<std::uint8_t>(1 + (rng() % 3));
        const auto val_high = static_cast<std::uint8_t>(1 + (rng() % 3));
        const auto low      = static_cast<std::uint8_t>(s_low | val_low);
        const auto high     = static_cast<std::uint8_t>(s_high | val_high);
        b = static_cast<std::uint8_t>((high << 4) | low);
    }
    for (std::size_t off = 0; off < model.big_nvfp4_gate_codes_buf.bytes; off += h_fp4.size()) {
        std::size_t chunk = std::min<std::size_t>(h_fp4.size(), model.big_nvfp4_gate_codes_buf.bytes - off);
        model.big_nvfp4_gate_codes_buf.copy_from_host(h_fp4.data(), chunk, off);
    }
    model.big_nvfp4_gate_scales_buf.fill(0x38);
    for (std::size_t off = 0; off < model.big_nvfp4_down_codes_buf.bytes; off += h_fp4.size()) {
        std::size_t chunk = std::min<std::size_t>(h_fp4.size(), model.big_nvfp4_down_codes_buf.bytes - off);
        model.big_nvfp4_down_codes_buf.copy_from_host(h_fp4.data(), chunk, off);
    }
    model.big_nvfp4_down_scales_buf.fill(0x38);

    std::vector<float> divisors(512, 1.0f);
    model.big_divisors_buf.copy_from_host(divisors.data(), divisors.size() * sizeof(float));

    // 8. PLE table
    constexpr std::uint64_t rows         = 1;
    constexpr std::uint64_t width        = 160;
    constexpr std::uint64_t scale_offset = 256;
    model.ple_table_data = std::vector<std::byte>(scale_offset + (width / 16) * 2, std::byte{0});
    for (std::size_t i = 0; i < width / 2; ++i) {
        model.ple_table_data[i] = static_cast<std::byte>(0x22 + (rng() % 16));
    }
    for (std::uint8_t index = 0; index < 8; ++index) {
        model.ple_table_data[index] = static_cast<std::byte>(index * 2 | ((index * 2 + 1) << 4));
    }
    constexpr std::uint16_t half_point_five = 0x3800;
    for (std::size_t offset = scale_offset; offset < model.ple_table_data.size(); offset += 2) {
        std::memcpy(model.ple_table_data.data() + offset, &half_point_five, sizeof(half_point_five));
    }
    for (PleShardView& shard : model.view.ple.table.shards) {
        shard = make_ple_shard_view(model.ple_table_data, rows, width);
    }

    auto make_bf16_weight_from = [](ninfer::DeviceBuffer& buf, std::int32_t rows, std::int32_t cols, std::size_t byte_offset = 0) {
        ninfer::Weight w{};
        w.payload         = static_cast<std::byte*>(buf.p) + byte_offset;
        w.payload_bytes   = static_cast<std::uint64_t>(rows) * cols * 2;
        w.qdata           = static_cast<std::byte*>(buf.p) + byte_offset;
        w.qtype           = ninfer::QType::BF16_CTRL;
        w.layout          = ninfer::QuantLayout::Contiguous;
        w.n               = rows;
        w.k               = cols;
        w.ndim            = 2;
        w.shape[0]        = rows;
        w.shape[1]        = cols;
        w.padded_shape[0] = rows;
        w.padded_shape[1] = cols;
        return w;
    };

    auto make_bf16_weight = [&](std::int32_t rows, std::int32_t cols) {
        return make_bf16_weight_from(model.big_bf16_buf, rows, cols);
    };

    auto make_fp8_weight = [](ninfer::DeviceBuffer& buf, std::int32_t rows, std::int32_t cols) {
        const std::uint64_t codes = static_cast<std::uint64_t>(rows) * cols;
        const std::uint64_t scale_off = (codes + 255U) & ~255ULL;
        ninfer::Weight w{};
        w.payload           = buf.p;
        w.payload_bytes     = buf.bytes;
        w.qdata             = buf.p;
        w.scales            = static_cast<const std::byte*>(buf.p) + scale_off;
        w.qtype             = ninfer::QType::FP8_E4M3FN_ROW_F32S;
        w.layout            = ninfer::QuantLayout::RowScale;
        w.scale_dtype       = ninfer::DType::FP32;
        w.group_size        = cols;
        w.group             = cols;
        w.n                 = rows;
        w.k                 = cols;
        w.ndim              = 2;
        w.shape[0]          = rows;
        w.shape[1]          = cols;
        w.shape[2]          = 1;
        w.shape[3]          = 1;
        w.padded_shape[0]   = rows;
        w.padded_shape[1]   = cols;
        w.padded_shape[2]   = 1;
        w.padded_shape[3]   = 1;
        w.scale_ne[0]       = rows;
        w.scale_ne[1]       = 1;
        w.scale_ne[2]       = 1;
        w.scale_ne[3]       = 1;
        w.scale_nb[0]       = 4;
        w.scale_nb[1]       = static_cast<std::int64_t>(rows) * 4;
        w.scale_nb[2]       = static_cast<std::int64_t>(rows) * 4;
        w.scale_nb[3]       = static_cast<std::int64_t>(rows) * 4;
        return w;
    };

    model.view.token_embedding = make_bf16_weight(248'320, 2'560);
    model.view.output_head     = make_bf16_weight(248'320, 2'560);

    model.view.ple.convolution      = ninfer::Tensor(model.ple_conv_buf.p, ninfer::DType::BF16, {10'240, 4});
    model.view.ple.key_projection   = make_bf16_weight(10'240, 2'560);
    model.view.ple.conv_norm        = ninfer::Tensor(model.norm_bf16_buf.p, ninfer::DType::BF16, {10'240});
    model.view.ple.key_norm         = ninfer::Tensor(model.norm_bf16_buf.p, ninfer::DType::BF16, {10'240});
    model.view.ple.query_norm       = ninfer::Tensor(model.norm_bf16_buf.p, ninfer::DType::BF16, {10'240});
    model.view.ple.value_projection = make_bf16_weight(2'560, 2'560);

    model.view.final_mixer.norm           = ninfer::Tensor(model.norm_bf16_buf.p, ninfer::DType::BF16, {10'240});
    model.view.final_mixer.input_mix_down = make_bf16_weight(320, 10'240);
    model.view.final_mixer.input_mix_up   = make_bf16_weight(10'240, 320);

    for (std::size_t l = 0; l < 48; ++l) {
        auto& layer = model.view.layers[l];
        std::uint64_t off = l * kLayerTotalFloats;

        layer.attention_hyper.block_inject   = make_bf16_weight_from(model.inject_buf, 4, 10'240);
        layer.attention_hyper.norm           = ninfer::Tensor(model.norm_bf16_buf.p, ninfer::DType::BF16, {10'240});
        layer.attention_hyper.input_mix_down = make_bf16_weight_from(model.per_layer_bf16_buf, 320, 10'240, off * sizeof(std::uint16_t));
        off += kAttnDownFloats;
        layer.attention_hyper.input_mix_up   = make_bf16_weight_from(model.per_layer_bf16_buf, 10'240, 320, off * sizeof(std::uint16_t));
        off += kAttnUpFloats;

        layer.mlp_hyper.block_inject   = make_bf16_weight_from(model.inject_buf, 4, 10'240);
        layer.mlp_hyper.norm           = ninfer::Tensor(model.norm_bf16_buf.p, ninfer::DType::BF16, {10'240});
        layer.mlp_hyper.input_mix_down = make_bf16_weight_from(model.per_layer_bf16_buf, 320, 10'240, off * sizeof(std::uint16_t));
        off += kMlpDownFloats;
        layer.mlp_hyper.input_mix_up   = make_bf16_weight_from(model.per_layer_bf16_buf, 10'240, 320, off * sizeof(std::uint16_t));
        off += kMlpUpFloats;

        layer.moe.router             = make_bf16_weight_from(model.per_layer_bf16_buf, 512, 2'560, off * sizeof(std::uint16_t));
        off += kRouterFloats;
        layer.moe.shared_down        = make_bf16_weight_from(model.per_layer_bf16_buf, 2'560, 640, off * sizeof(std::uint16_t));
        off += kSharedDownFloats;
        layer.moe.shared_gate        = make_bf16_weight_from(model.per_layer_bf16_buf, 640, 2'560, off * sizeof(std::uint16_t));
        off += kSharedGateFloats;
        layer.moe.shared_up          = make_bf16_weight_from(model.per_layer_bf16_buf, 640, 2'560, off * sizeof(std::uint16_t));
        off += kSharedUpFloats;

        layer.moe.shared_gate_weight = make_bf16_weight_from(model.shared_gate_weight_buf, 1, 2'560);
        layer.moe.expert_gate_up     = Nvfp4ExpertBankView{
            .codes                  = static_cast<const std::byte*>(model.big_nvfp4_gate_codes_buf.p),
            .scales                 = static_cast<const std::byte*>(model.big_nvfp4_gate_scales_buf.p),
            .weight_scale_divisors  = static_cast<const float*>(model.big_divisors_buf.p),
            .experts                = 512,
            .rows                   = 1'280,
            .columns                = 2'560,
            .code_bytes_per_expert  = gate_code_bytes_per_expert,
            .scale_bytes_per_expert = gate_scale_bytes_per_expert,
        };
        layer.moe.expert_down        = Nvfp4ExpertBankView{
            .codes                  = static_cast<const std::byte*>(model.big_nvfp4_down_codes_buf.p),
            .scales                 = static_cast<const std::byte*>(model.big_nvfp4_down_scales_buf.p),
            .weight_scale_divisors  = static_cast<const float*>(model.big_divisors_buf.p),
            .experts                = 512,
            .rows                   = 2'560,
            .columns                = 640,
            .code_bytes_per_expert  = down_code_bytes_per_expert,
            .scale_bytes_per_expert = down_scale_bytes_per_expert,
        };
    }

    for (std::size_t i = 0; i < kGdnLayers; ++i) {
        auto& gdn = model.view.gdn[i];
        gdn.a_log             = ninfer::Tensor(model.gdn_a_log_buf.p, ninfer::DType::BF16, {48});
        gdn.convolution       = ninfer::Tensor(model.gdn_conv_buf.p, ninfer::DType::BF16, {10'240, 4});
        gdn.dt_bias           = ninfer::Tensor(model.gdn_dt_bias_buf.p, ninfer::DType::BF16, {48});
        gdn.a_b_projection    = make_bf16_weight(96, 2'560);
        gdn.norm              = ninfer::Tensor(model.norm_bf16_buf.p, ninfer::DType::BF16, {128});
        gdn.query_key_value_z = make_fp8_weight(model.fp8_qkvz_buf, 16'384, 2'560);
        gdn.output            = make_fp8_weight(model.fp8_out_buf, 2'560, 6'144);
    }

    for (std::size_t i = 0; i < kFullAttentionLayers; ++i) {
        auto& att = model.view.full_attention[i];
        att.indexer_query_key    = make_bf16_weight(640, 2'560);
        att.indexer_key_norm     = ninfer::Tensor(model.norm_bf16_buf.p, ninfer::DType::BF16, {128});
        att.indexer_query_norm   = ninfer::Tensor(model.norm_bf16_buf.p, ninfer::DType::BF16, {128});
        att.key_norm             = ninfer::Tensor(model.norm_bf16_buf.p, ninfer::DType::BF16, {256});
        att.query_norm           = ninfer::Tensor(model.norm_bf16_buf.p, ninfer::DType::BF16, {256});
        att.query_gate_key_value = make_fp8_weight(model.fp8_qgkv_buf, 13'312, 2'560);
        att.output               = make_fp8_weight(model.fp8_out_buf, 2'560, 6'144);
    }

    device.synchronize();
    return model;
}

int test_ledger_cpu() {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;

    FlashNextRuntimeConfig cfg{
        .max_concurrency     = 2,
        .max_context         = 512,
        .state_slot_capacity = 4,
        .prefill_chunk       = 512,
    };
    const auto curve = flash_next_capacity_curve(cfg);
    auto plan        = finalize_flash_next_runtime_plan(cfg, curve.maximum_main_page_groups);

    FlashNextLaneLedger ledger(plan);

    if (ledger.available_physical_groups() != plan.main_page_groups) {
        std::cerr << "Initial available physical groups mismatch\n";
        return 1;
    }

    // 1. Allocate two lanes
    auto lane0 = ledger.allocate_lane();
    auto lane1 = ledger.allocate_lane();

    if (lane0.lane_index() != 0 || lane1.lane_index() != 1) {
        std::cerr << "Lane indices mismatch\n";
        return 1;
    }
    if (ledger.active_lanes_count() != 2) {
        std::cerr << "Active lanes count mismatch\n";
        return 1;
    }

    // 2. Reject 3rd lane allocation
    try {
        (void)ledger.allocate_lane();
        std::cerr << "Failed to reject 3rd lane allocation\n";
        return 1;
    } catch (const std::runtime_error&) {}

    // 3. Stale handle rejection
    auto stale_handle = lane0;
    ledger.release_lane(lane0);
    try {
        (void)ledger.committed_frontier(stale_handle);
        std::cerr << "Failed to reject released/stale lane handle\n";
        return 1;
    } catch (const std::invalid_argument&) {}
    lane0 = ledger.allocate_lane();

    // 4. CPU owner-isolation: two fresh ledgers with the same lane index and epoch
    FlashNextLaneLedger owner_ledger_a(plan);
    FlashNextLaneLedger owner_ledger_b(plan);
    auto owner_lane_a = owner_ledger_a.allocate_lane();
    auto owner_lane_b = owner_ledger_b.allocate_lane();
    if (owner_lane_a.lane_index() != owner_lane_b.lane_index() ||
        owner_lane_a.epoch() != owner_lane_b.epoch()) {
        std::cerr << "Owner-isolation setup did not produce matching lane/epoch\n";
        return 1;
    }
    try {
        (void)owner_ledger_a.committed_frontier(owner_lane_b);
        std::cerr << "Failed to reject cross-ledger handle with same lane/epoch\n";
        return 1;
    } catch (const std::invalid_argument&) {}
    owner_ledger_a.release_lane(owner_lane_a);
    owner_ledger_b.release_lane(owner_lane_b);

    // 5. Batch validation: empty, duplicate, non-monotonic
    PleIndexMetadata ple_meta{};
    ple_meta.multipliers = {1, 2, 3};
    ple_meta.head_offsets.fill(0);
    ple_meta.head_vocab_sizes.fill(100);

    std::vector<LaneStepRequest> empty_reqs;
    try {
        (void)ledger.begin_round(empty_reqs, ple_meta);
        std::cerr << "Failed to reject empty batch\n";
        return 1;
    } catch (const std::invalid_argument&) {}

    std::vector<LaneStepRequest> dup_reqs = {
        {.handle = lane0, .token_id = 42, .token_index = 0, .mrope_positions = {0, 0, 0}},
        {.handle = lane0, .token_id = 43, .token_index = 0, .mrope_positions = {0, 0, 0}},
    };
    try {
        (void)ledger.begin_round(dup_reqs, ple_meta);
        std::cerr << "Failed to reject duplicate lane\n";
        return 1;
    } catch (const std::invalid_argument&) {}

    std::vector<LaneStepRequest> non_mono_reqs = {
        {.handle = lane0, .token_id = 42, .token_index = 5, .mrope_positions = {5, 5, 5}},
    };
    try {
        (void)ledger.begin_round(non_mono_reqs, ple_meta);
        std::cerr << "Failed to reject non-monotonic token index\n";
        return 1;
    } catch (const std::invalid_argument&) {}

    // 6. Active blocks exact calculation: token_index=0 -> blocks=0
    std::vector<LaneStepRequest> req_token0 = {
        {.handle = lane0, .token_id = 1, .token_index = 0, .mrope_positions = {0, 0, 0}},
    };
    auto prep0 = ledger.begin_round(req_token0, ple_meta);
    if (prep0.max_active_blocks != 0) {
        std::cerr << "Active blocks for token 0 expected 0 got " << prep0.max_active_blocks << "\n";
        return 1;
    }
    ledger.abort_round(prep0.transaction_id);

    // 7. Valid round, exact table indexing, and full rollback verification
    const auto groups_before = ledger.available_physical_groups();
    std::vector<LaneStepRequest> valid_reqs = {
        {.handle = lane0, .token_id = 10, .token_index = 0, .mrope_positions = {0, 0, 0}},
        {.handle = lane1, .token_id = 20, .token_index = 0, .mrope_positions = {0, 0, 0}},
    };
    auto prep = ledger.begin_round(valid_reqs, ple_meta);

    // Check lane 1 exact indexer and attention table entries
    const auto att_tab = ledger.host_attention_table();
    const auto idx_tab = ledger.host_indexer_table();

    const auto l1_idx_group = idx_tab[1ULL * plan.indexer_logical_pages + 0];
    if (l1_idx_group < 0) {
        std::cerr << "Host indexer table for lane 1 not mapped\n";
        return 1;
    }
    for (std::uint32_t s = 0; s < 4; ++s) {
        const auto att_val = att_tab[1ULL * plan.attention_logical_pages + s];
        if (att_val != static_cast<std::int32_t>(l1_idx_group * 4 + s)) {
            std::cerr << "Host attention table for lane 1 mismatch at subpage " << s << "\n";
            return 1;
        }
    }

    // Rollback: verify ALL attention+indexer entries cleared AND groups restored
    ledger.rollback_prepared_round(prep.transaction_id);
    if (ledger.has_pending_transaction()) {
        std::cerr << "Pending transaction still active after rollback\n";
        return 1;
    }
    if (ledger.available_physical_groups() != groups_before) {
        std::cerr << "Physical groups not restored after rollback: "
                  << ledger.available_physical_groups() << " expected " << groups_before << "\n";
        return 1;
    }
    // All attention entries for lane 1 should be -1
    for (std::uint32_t s = 0; s < 4; ++s) {
        if (att_tab[1ULL * plan.attention_logical_pages + s] != -1) {
            std::cerr << "Attention table entry not cleared after rollback at subpage " << s
                      << "\n";
            return 1;
        }
    }
    if (idx_tab[1ULL * plan.indexer_logical_pages + 0] != -1) {
        std::cerr << "Indexer table entry not cleared after rollback\n";
        return 1;
    }

    // 8. Re-run round and test commit accept/reject
    prep = ledger.begin_round(valid_reqs, ple_meta);
    FlashNextRuntimeAllocation dummy_alloc(plan);
    std::vector<LaneCommitDecision> decisions = {
        {.accept = true},
        {.accept = false},
    };
    ledger.commit_round(prep.transaction_id, decisions, dummy_alloc, nullptr);

    if (ledger.committed_frontier(lane0) != 1) {
        std::cerr << "Lane 0 frontier mismatch on accept\n";
        return 1;
    }
    if (ledger.committed_frontier(lane1) != 0) {
        std::cerr << "Lane 1 frontier mismatch on reject\n";
        return 1;
    }

    ledger.release_lane(lane0);
    ledger.release_lane(lane1);

    if (ledger.available_physical_groups() != plan.main_page_groups) {
        std::cerr << "Physical groups not fully reclaimed after release\n";
        return 1;
    }

    std::cout << "PASS: test_ledger_cpu\n";
    return 0;
}

int test_ple_boundary_lifecycle_cpu() {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;

    FlashNextRuntimeConfig cfg{
        .max_concurrency     = 2,
        .max_context         = 512,
        .state_slot_capacity = 4,
        .prefill_chunk       = 512,
    };
    const auto curve = flash_next_capacity_curve(cfg);
    auto plan        = finalize_flash_next_runtime_plan(cfg, curve.maximum_main_page_groups);

    FlashNextLaneLedger ledger(plan);
    PleIndexMetadata ple_meta{};
    ple_meta.multipliers = {1, 2, 3};
    ple_meta.head_offsets.fill(0);
    ple_meta.head_vocab_sizes.fill(100);

    // 1. Initial allocation: exact default history (248044, 248044)
    auto lane = ledger.allocate_lane();
    const auto& hist0 = ledger.lane_history(lane);
    if (hist0.previous_token() != kPleBoundaryTokenId ||
        hist0.second_previous_token() != kPleBoundaryTokenId) {
        std::cerr << "Initial lane history not seeded with kPleBoundaryTokenId (248044)\n";
        return 1;
    }

    // 2. Abort unchanged
    std::vector<LaneStepRequest> req0 = {
        {.handle = lane, .token_id = 1234, .token_index = 0, .mrope_positions = {0, 0, 0}},
    };
    auto prep0 = ledger.begin_round(req0, ple_meta);
    ledger.abort_round(prep0.transaction_id);
    const auto& hist_after_abort = ledger.lane_history(lane);
    if (hist_after_abort.previous_token() != kPleBoundaryTokenId ||
        hist_after_abort.second_previous_token() != kPleBoundaryTokenId) {
        std::cerr << "Lane history changed after abort_round\n";
        return 1;
    }

    // 3. Commit token 500 -> (500, 248044)
    FlashNextRuntimeAllocation dummy_alloc(plan);
    std::vector<LaneCommitDecision> accept = {{.accept = true}};
    prep0 = ledger.begin_round(req0, ple_meta);
    ledger.commit_round(prep0.transaction_id, accept, dummy_alloc, nullptr);
    const auto& hist1 = ledger.lane_history(lane);
    if (hist1.previous_token() != 1234 || hist1.second_previous_token() != kPleBoundaryTokenId) {
        std::cerr << "Lane history mismatch after first commit\n";
        return 1;
    }

    // 4. Token 248046: ordinary advance (not reset) -> (248046, 1234)
    std::vector<LaneStepRequest> req1 = {
        {.handle = lane, .token_id = 248'046, .token_index = 1, .mrope_positions = {1, 1, 1}},
    };
    auto prep1 = ledger.begin_round(req1, ple_meta);
    ledger.commit_round(prep1.transaction_id, accept, dummy_alloc, nullptr);
    const auto& hist2 = ledger.lane_history(lane);
    if (hist2.previous_token() != 248'046 || hist2.second_previous_token() != 1234) {
        std::cerr << "Token 248046 did not advance history normally\n";
        return 1;
    }

    // 5. Token 248044: boundary reset -> (248044, 248044)
    std::vector<LaneStepRequest> req2 = {
        {.handle = lane, .token_id = 248'044, .token_index = 2, .mrope_positions = {2, 2, 2}},
    };
    auto prep2 = ledger.begin_round(req2, ple_meta);
    ledger.commit_round(prep2.transaction_id, accept, dummy_alloc, nullptr);
    const auto& hist3 = ledger.lane_history(lane);
    if (hist3.previous_token() != kPleBoundaryTokenId ||
        hist3.second_previous_token() != kPleBoundaryTokenId) {
        std::cerr << "Token 248044 did not reset boundary history\n";
        return 1;
    }

    // 6. Next token 777 after reset -> (777, 248044)
    std::vector<LaneStepRequest> req3 = {
        {.handle = lane, .token_id = 777, .token_index = 3, .mrope_positions = {3, 3, 3}},
    };
    auto prep3 = ledger.begin_round(req3, ple_meta);
    ledger.commit_round(prep3.transaction_id, accept, dummy_alloc, nullptr);
    const auto& hist4 = ledger.lane_history(lane);
    if (hist4.previous_token() != 777 || hist4.second_previous_token() != kPleBoundaryTokenId) {
        std::cerr << "Token after boundary reset mismatch\n";
        return 1;
    }

    // 7. Release and reallocate reseed -> (248044, 248044)
    ledger.release_lane(lane);
    auto reallocated_lane = ledger.allocate_lane();
    const auto& hist_realloc = ledger.lane_history(reallocated_lane);
    if (hist_realloc.previous_token() != kPleBoundaryTokenId ||
        hist_realloc.second_previous_token() != kPleBoundaryTokenId) {
        std::cerr << "Reallocated lane history not reseeded with kPleBoundaryTokenId\n";
        return 1;
    }
    ledger.release_lane(reallocated_lane);

    std::cout << "PASS: test_ple_boundary_lifecycle_cpu\n";
    return 0;
}

int test_cuda_ledger_and_executor(ninfer::DeviceContext& device) {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;

    PleIndexMetadata ple_meta{};
    ple_meta.multipliers = {1, 2, 3};
    ple_meta.head_offsets.fill(0);
    ple_meta.head_vocab_sizes.fill(1);

    constexpr std::uint64_t rows         = 1;
    constexpr std::uint64_t width        = 160;
    constexpr std::uint64_t scale_offset = 256;
    std::vector<std::byte> encoded(scale_offset + (width / 16) * 2, std::byte{0});
    std::fill_n(encoded.begin(), width / 2, std::byte{0x88});
    for (std::uint8_t index = 0; index < 8; ++index) {
        encoded[index] = static_cast<std::byte>(index * 2 | ((index * 2 + 1) << 4));
    }
    constexpr std::uint16_t half_point_five = 0x3800;
    for (std::size_t offset = scale_offset; offset < encoded.size(); offset += 2) {
        std::memcpy(encoded.data() + offset, &half_point_five, sizeof(half_point_five));
    }
    PleTableView ple_table;
    for (PleShardView& shard : ple_table.shards) {
        shard = make_ple_shard_view(encoded, rows, width);
    }

    TextModelView mock_model{};
    mock_model.ple.table = ple_table;

    FlashNextRuntimeConfig cfg{
        .max_concurrency     = 2,
        .max_context         = 512,
        .state_slot_capacity = 4,
        .prefill_chunk       = 512,
    };
    const auto curve = flash_next_capacity_curve(cfg);
    auto plan        = finalize_flash_next_runtime_plan(cfg, curve.maximum_main_page_groups);

    FlashNextRuntimeAllocation alloc(plan);
    alloc.initialize(device.stream);
    device.synchronize();

    // 1. Bounded CUDA ledger: table upload and slot commit
    FlashNextLaneLedger ledger(plan);
    auto lane0 = ledger.allocate_lane();
    auto lane1 = ledger.allocate_lane();

    std::vector<LaneStepRequest> reqs = {
        {.handle = lane0, .token_id = 0, .token_index = 0, .mrope_positions = {0, 0, 0}},
        {.handle = lane1, .token_id = 0, .token_index = 0, .mrope_positions = {0, 0, 0}},
    };
    auto prep = ledger.begin_round(reqs, ple_meta);
    ledger.sync_tables_if_dirty(alloc, device.stream);
    device.synchronize();

    // Verify transposed device table for lane 1
    std::vector<std::int32_t> dev_att(plan.attention_logical_pages * cfg.max_concurrency);
    CUDA_CHECK(cudaMemcpy(dev_att.data(),
                          alloc.state_view().qsa_attention_caches[0].block_tables.data,
                          dev_att.size() * sizeof(std::int32_t), cudaMemcpyDeviceToHost));

    std::vector<std::int32_t> dev_idx(plan.indexer_logical_pages * cfg.max_concurrency);
    CUDA_CHECK(cudaMemcpy(dev_idx.data(),
                          alloc.state_view().qsa_indexer_caches[0].block_tables.data,
                          dev_idx.size() * sizeof(std::int32_t), cudaMemcpyDeviceToHost));

    const auto l1_phys = dev_idx[1ULL * plan.indexer_logical_pages + 0];
    if (l1_phys < 0) {
        std::cerr << "Device indexer table for lane 1 not mapped\n";
        return 1;
    }
    for (std::uint32_t s = 0; s < 4; ++s) {
        const auto val = dev_att[1ULL * plan.attention_logical_pages + s];
        if (val != static_cast<std::int32_t>(l1_phys * 4 + s)) {
            std::cerr << "Device attention table for lane 1 mismatch\n";
            return 1;
        }
    }

    std::vector<LaneCommitDecision> decisions = {{.accept = true}, {.accept = false}};
    ledger.commit_round(prep.transaction_id, decisions, alloc, device.stream);
    device.synchronize();

    if (alloc.current_source_slot(0) != 1 || alloc.current_source_slot(1) != 2) {
        std::cerr << "Slots after commit mismatch\n";
        return 1;
    }

    ledger.release_lane(lane0);
    ledger.release_lane(lane1);

    // 2. Executor: cross-executor rejection, invalid-model execute_round failure rollback
    FlashNextTextExecutor executor(mock_model, ple_meta, device, alloc);
    auto elane0 = executor.allocate_lane();
    auto elane1 = executor.allocate_lane();

    // Cross-executor rejection
    FlashNextRuntimeAllocation alloc2(plan);
    alloc2.initialize(device.stream);
    FlashNextTextExecutor executor2(mock_model, ple_meta, device, alloc2);
    try {
        (void)executor2.committed_frontier(elane0);
        std::cerr << "Failed to reject cross-executor handle\n";
        return 1;
    } catch (const std::invalid_argument&) {}

    // 3. Invalid-model execute_round must throw (flash_next_text_decode with null weights),
    //    must leave no pending transaction, and must restore physical groups.
    const auto groups_before_exec = executor.available_physical_groups();
    std::vector<LaneStepRequest> exec_reqs = {
        {.handle = elane0, .token_id = 0, .token_index = 0, .mrope_positions = {0, 0, 0}},
    };
    bool decode_threw = false;
    try {
        auto round = executor.execute_round(exec_reqs);
        round.abort();
    } catch (...) { decode_threw = true; }
    if (!decode_threw) {
        std::cerr << "Invalid model unexpectedly bypassed the real decode path\n";
        return 1;
    }
    if (executor.has_pending_round()) {
        std::cerr << "Pending round leaked after execute_round failure\n";
        return 1;
    }
    if (executor.available_physical_groups() != groups_before_exec) {
        std::cerr << "Physical groups leaked after execute_round failure: "
                  << executor.available_physical_groups() << " expected " << groups_before_exec
                  << "\n";
        return 1;
    }

    // 4. Recurrent state slot zeroing on allocation/reuse
    std::vector<std::uint16_t> dirty_ple(10'240 * 9, 0xABCD);
    CUDA_CHECK(cudaMemcpy(alloc.state_view().ple_convolution_states.data, dirty_ple.data(),
                          dirty_ple.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice));

    executor.release_lane(elane0);
    auto reallocated = executor.allocate_lane();
    device.synchronize();

    std::vector<std::uint16_t> clean_ple(10'240 * 9, 0x1234);
    CUDA_CHECK(cudaMemcpy(clean_ple.data(), alloc.state_view().ple_convolution_states.data,
                          clean_ple.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost));
    for (std::size_t i = 0; i < clean_ple.size(); ++i) {
        if (clean_ple[i] != 0) {
            std::cerr << "Recurrent state not zeroed on lane reallocation\n";
            return 1;
        }
    }

    executor.release_lane(reallocated);
    executor.release_lane(elane1);

    // 5. B=2 planar MRoPE positions layout on device
    auto plane0 = executor.allocate_lane();
    auto plane1 = executor.allocate_lane();
    std::vector<LaneStepRequest> b2_reqs = {
        {.handle = plane0, .token_id = 10, .token_index = 0, .mrope_positions = {100, 101, 102}},
        {.handle = plane1, .token_id = 20, .token_index = 0, .mrope_positions = {200, 201, 202}},
    };
    try {
        auto round = executor.execute_round(b2_reqs);
        round.abort();
    } catch (...) {}
    device.synchronize();

    std::vector<std::int32_t> dev_mrope(6);
    CUDA_CHECK(cudaMemcpy(dev_mrope.data(), alloc.round_tensors().mrope_positions.data,
                          6 * sizeof(std::int32_t), cudaMemcpyDeviceToHost));
    const std::vector<std::int32_t> expected_planar_mrope = {100, 200, 101, 201, 102, 202};
    if (dev_mrope != expected_planar_mrope) {
        std::cerr << "Device MRoPE positions did not match expected planar layout for B=2: got [";
        for (auto v : dev_mrope) std::cerr << v << ", ";
        std::cerr << "]\n";
        return 1;
    }
    executor.release_lane(plane0);
    executor.release_lane(plane1);

    std::cout << "PASS: test_cuda_ledger_and_executor\n";
    return 0;
}

int test_ledger_prefill_chunk_cpu() {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;

    FlashNextRuntimeConfig cfg{
        .max_concurrency     = 2,
        .max_context         = 512,
        .state_slot_capacity = 4,
        .prefill_chunk       = 512,
    };
    const auto curve = flash_next_capacity_curve(cfg);
    auto plan        = finalize_flash_next_runtime_plan(cfg, curve.maximum_main_page_groups);

    FlashNextLaneLedger ledger(plan);
    PleIndexMetadata ple_meta{};
    ple_meta.multipliers = {1, 2, 3};
    ple_meta.head_offsets.fill(0);
    ple_meta.head_vocab_sizes.fill(100);

    auto lane0 = ledger.allocate_lane();

    // 1. Validation: empty, invalid frontier, out-of-range
    std::vector<std::int32_t> empty_tokens;
    try {
        (void)ledger.begin_prefill_chunk(lane0, empty_tokens, 0, ple_meta);
        std::cerr << "Failed to reject empty prefill chunk\n";
        return 1;
    } catch (const std::invalid_argument&) {}

    std::vector<std::int32_t> chunk4 = {10, 11, 12, 13};
    try {
        (void)ledger.begin_prefill_chunk(lane0, chunk4, 2, ple_meta); // expected frontier 0
        std::cerr << "Failed to reject frontier mismatch in prefill chunk\n";
        return 1;
    } catch (const std::invalid_argument&) {}

    // 2. Abort transaction: leaves frontier and history unchanged
    auto prep0 = ledger.begin_prefill_chunk(lane0, chunk4, 0, ple_meta);
    if (!ledger.has_pending_transaction()) {
        std::cerr << "Pending transaction not flagged after begin_prefill_chunk\n";
        return 1;
    }
    ledger.abort_round(prep0.transaction_id);
    if (ledger.has_pending_transaction()) {
        std::cerr << "Pending transaction still active after abort_round\n";
        return 1;
    }
    if (ledger.committed_frontier(lane0) != 0) {
        std::cerr << "Frontier modified after aborted prefill chunk\n";
        return 1;
    }
    const auto& hist0 = ledger.lane_history(lane0);
    if (hist0.previous_token() != kPleBoundaryTokenId ||
        hist0.second_previous_token() != kPleBoundaryTokenId) {
        std::cerr << "History modified after aborted prefill chunk\n";
        return 1;
    }

    // 3. Rollback prepared prefill chunk: restores groups and clears tables
    const auto groups_before = ledger.available_physical_groups();
    auto prep1 = ledger.begin_prefill_chunk(lane0, chunk4, 0, ple_meta);
    ledger.rollback_prepared_round(prep1.transaction_id);
    if (ledger.available_physical_groups() != groups_before) {
        std::cerr << "Groups not restored after rollback_prepared_round\n";
        return 1;
    }

    // 4. Commit prefill chunk (T=4): advances frontier by 4, commits all 4 tokens to history
    FlashNextRuntimeAllocation dummy_alloc(plan);
    prep1 = ledger.begin_prefill_chunk(lane0, chunk4, 0, ple_meta);
    std::vector<LaneCommitDecision> accept = {{.accept = true}};
    ledger.commit_round(prep1.transaction_id, accept, dummy_alloc, nullptr);

    if (ledger.committed_frontier(lane0) != 4) {
        std::cerr << "Committed frontier expected 4, got " << ledger.committed_frontier(lane0) << "\n";
        return 1;
    }
    const auto& hist1 = ledger.lane_history(lane0);
    if (hist1.previous_token() != 13 || hist1.second_previous_token() != 12) {
        std::cerr << "History after T=4 prefill chunk commit mismatch: got ("
                  << hist1.previous_token() << ", " << hist1.second_previous_token() << ")\n";
        return 1;
    }

    // 5. Subsequent chunk (T=3): from index 4 -> 7
    std::vector<std::int32_t> chunk3 = {20, 21, 22};
    auto prep2 = ledger.begin_prefill_chunk(lane0, chunk3, 4, ple_meta);
    ledger.commit_round(prep2.transaction_id, accept, dummy_alloc, nullptr);
    if (ledger.committed_frontier(lane0) != 7) {
        std::cerr << "Committed frontier expected 7, got " << ledger.committed_frontier(lane0) << "\n";
        return 1;
    }
    const auto& hist2 = ledger.lane_history(lane0);
    if (hist2.previous_token() != 22 || hist2.second_previous_token() != 21) {
        std::cerr << "History after second chunk commit mismatch\n";
        return 1;
    }

    ledger.release_lane(lane0);
    std::cout << "PASS: test_ledger_prefill_chunk_cpu\n";
    return 0;
}

int test_finite_model_stages(ninfer::DeviceContext& device) {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    try {
        PleIndexMetadata ple_meta{};
        ple_meta.multipliers = {1, 2, 3};
        ple_meta.head_offsets.fill(0);
        ple_meta.head_vocab_sizes.fill(1);

        auto synthetic_model = make_synthetic_model(device);

        FlashNextRuntimeConfig cfg{
            .max_concurrency     = 1,
            .max_context         = 512,
            .state_slot_capacity = 2,
            .prefill_chunk       = 128,
            .use_cuda_graph      = false,
        };
        const auto curve = flash_next_capacity_curve(cfg);
        auto plan        = finalize_flash_next_runtime_plan(cfg, curve.maximum_main_page_groups);

        FlashNextRuntimeAllocation alloc(plan);
        alloc.initialize(device.stream);
        FlashNextTextExecutor exec(synthetic_model.view, ple_meta, device, alloc);

        // 1. Check one eager round with sink
        std::size_t eager_stages = 0;
        std::string first_eager_non_finite;
        FlashNextDecodeStateSink eager_sink;
        eager_sink.on_state = [&](std::string_view name, const ninfer::Tensor& t) {
            ++eager_stages;
            if (!first_eager_non_finite.empty()) { return; }
            std::size_t count = 1;
            for (int d = 0; d < 4; ++d) { if (t.ne[d] > 0) { count *= static_cast<std::size_t>(t.ne[d]); } }
            if (t.dtype == ninfer::DType::BF16) {
                std::vector<std::uint16_t> host(count);
                device.synchronize();
                CUDA_CHECK(cudaMemcpy(host.data(), t.data, count * sizeof(std::uint16_t), cudaMemcpyDeviceToHost));
                for (auto v : host) {
                    const float f = bf16_to_float(v);
                    if (std::isnan(f) || std::isinf(f)) {
                        first_eager_non_finite = std::string(name);
                        break;
                    }
                }
            } else if (t.dtype == ninfer::DType::FP32) {
                std::vector<float> host(count);
                device.synchronize();
                CUDA_CHECK(cudaMemcpy(host.data(), t.data, count * sizeof(float), cudaMemcpyDeviceToHost));
                for (auto f : host) {
                    if (std::isnan(f) || std::isinf(f)) {
                        first_eager_non_finite = std::string(name);
                        break;
                    }
                }
            }
        };

        auto lane = exec.allocate_lane();
        LaneStepRequest req{
            .handle          = lane,
            .token_id        = 100,
            .token_index     = 0,
            .mrope_positions = {0, 0, 0},
            .sampling        = {.temperature = 0.0F, .top_p = 1.0F},
        };
        auto round = exec.execute_round(std::span(&req, 1), &eager_sink);
        std::vector<LaneCommitDecision> dec = {{.accept = true}};
        round.commit(dec);
        device.synchronize();

        if (!first_eager_non_finite.empty()) {
            std::cerr << "FAIL: Eager decode produced non-finite value at stage: "
                      << first_eager_non_finite << "\n";
            return 1;
        }
        if (eager_stages == 0) {
            std::cerr << "FAIL: Eager decode did not emit any stages to sink\n";
            return 1;
        }

        // 2. Check one prefill chunk with sink
        std::size_t prefill_stages = 0;
        std::string first_prefill_non_finite;
        FlashNextDecodeStateSink prefill_sink;
        prefill_sink.on_state = [&](std::string_view name, const ninfer::Tensor& t) {
            ++prefill_stages;
            if (!first_prefill_non_finite.empty()) { return; }
            std::size_t count = 1;
            for (int d = 0; d < 4; ++d) { if (t.ne[d] > 0) { count *= static_cast<std::size_t>(t.ne[d]); } }
            if (t.dtype == ninfer::DType::BF16) {
                std::vector<std::uint16_t> host(count);
                device.synchronize();
                CUDA_CHECK(cudaMemcpy(host.data(), t.data, count * sizeof(std::uint16_t), cudaMemcpyDeviceToHost));
                for (auto v : host) {
                    const float f = bf16_to_float(v);
                    if (std::isnan(f) || std::isinf(f)) {
                        first_prefill_non_finite = std::string(name);
                        break;
                    }
                }
            } else if (t.dtype == ninfer::DType::FP32) {
                std::vector<float> host(count);
                device.synchronize();
                CUDA_CHECK(cudaMemcpy(host.data(), t.data, count * sizeof(float), cudaMemcpyDeviceToHost));
                for (auto f : host) {
                    if (std::isnan(f) || std::isinf(f)) {
                        first_prefill_non_finite = std::string(name);
                        break;
                    }
                }
            }
        };

        exec.release_lane(lane);
        lane = exec.allocate_lane();
        constexpr std::int32_t kChunk = 128;
        std::vector<std::int32_t> prefill_tokens(kChunk);
        std::vector<std::array<std::int32_t, 3>> prefill_pos(kChunk);
        for (std::int32_t t = 0; t < kChunk; ++t) {
            prefill_tokens[t] = 100 + t;
            prefill_pos[t]    = {t, t, t};
        }
        auto pr = exec.execute_prefill_chunk(lane, prefill_tokens, prefill_pos, 0, &prefill_sink);
        pr.commit(dec);
        device.synchronize();

        if (!first_prefill_non_finite.empty()) {
            std::cerr << "FAIL: Prefill chunk produced non-finite value at stage: "
                      << first_prefill_non_finite << "\n";
            return 1;
        }
        if (prefill_stages == 0) {
            std::cerr << "FAIL: Prefill chunk did not emit any stages to sink\n";
            return 1;
        }

        exec.release_lane(lane);
        std::cout << "NAN PROBE: stages=" << eager_stages << " first non-finite stage: none\n";
        std::cout << "PASS: test_finite_model_stages\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "test_finite_model_stages exception: " << e.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "test_finite_model_stages unknown exception\n";
        return 1;
    }
}

int test_prefill_chunk_executor(ninfer::DeviceContext& device) {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    try {
        PleIndexMetadata ple_meta{};
        ple_meta.multipliers = {1, 2, 3};
        ple_meta.head_offsets.fill(0);
        ple_meta.head_vocab_sizes.fill(1);

        auto synthetic_model = make_synthetic_model(device);

        FlashNextRuntimeConfig cfg{
            .max_concurrency     = 2,
            .max_context         = 512,
            .state_slot_capacity = 4,
            .prefill_chunk       = 512,
        };
        const auto curve = flash_next_capacity_curve(cfg);
        auto plan        = finalize_flash_next_runtime_plan(cfg, curve.maximum_main_page_groups);

        // 1. Sequential execution of 4 tokens on allocation 1
        FlashNextRuntimeAllocation alloc_seq(plan);
        alloc_seq.initialize(device.stream);
        FlashNextTextExecutor exec_seq(synthetic_model.view, ple_meta, device, alloc_seq);
        auto lane_seq = exec_seq.allocate_lane();

        const std::vector<std::int32_t> tokens = {100, 101, 102, 103};
        const std::vector<std::array<std::int32_t, 3>> positions = {
            {0, 0, 0}, {1, 1, 1}, {2, 2, 2}, {3, 3, 3}};

        for (std::size_t t = 0; t < 4; ++t) {
            LaneStepRequest req{
                .handle          = lane_seq,
                .token_id        = tokens[t],
                .token_index     = static_cast<std::int32_t>(t),
                .mrope_positions = positions[t],
            };
            auto round = exec_seq.execute_round(std::span(&req, 1));
            std::vector<LaneCommitDecision> decision = {{.accept = true}};
            round.commit(decision);
        }
        device.synchronize();

        const std::size_t persistent_bytes =
            plan.total_device_bytes - plan.workspace_bytes - plan.cuda_graph_allowance_bytes;
        std::vector<std::uint8_t> seq_storage(persistent_bytes);
        CUDA_CHECK(cudaMemcpy(seq_storage.data(), alloc_seq.state_view().qsa_attention_caches[0].key_pages.data,
                              persistent_bytes, cudaMemcpyDeviceToHost));

        std::vector<std::uint16_t> seq_logits(248'320);
        CUDA_CHECK(cudaMemcpy(seq_logits.data(), alloc_seq.round_tensors().logits.data,
                              seq_logits.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost));

        std::vector<std::uint16_t> seq_hidden(2'560);
        CUDA_CHECK(cudaMemcpy(seq_hidden.data(), alloc_seq.round_tensors().final_hidden.data,
                              seq_hidden.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost));

        // 2. Chunked execution of T=4 tokens on allocation 2
        FlashNextRuntimeAllocation alloc_chunk(plan);
        alloc_chunk.initialize(device.stream);
        FlashNextTextExecutor exec_chunk(synthetic_model.view, ple_meta, device, alloc_chunk);
        auto lane_chunk = exec_chunk.allocate_lane();

        auto chunk_round = exec_chunk.execute_prefill_chunk(lane_chunk, tokens, positions, 0);
        std::vector<LaneCommitDecision> decision = {{.accept = true}};
        chunk_round.commit(decision);
        device.synchronize();

        std::vector<std::uint8_t> chunk_storage(persistent_bytes);
        CUDA_CHECK(cudaMemcpy(chunk_storage.data(), alloc_chunk.state_view().qsa_attention_caches[0].key_pages.data,
                              persistent_bytes, cudaMemcpyDeviceToHost));

        std::vector<std::uint16_t> chunk_logits(248'320);
        CUDA_CHECK(cudaMemcpy(chunk_logits.data(), alloc_chunk.round_tensors().logits.data,
                              chunk_logits.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost));

        std::vector<std::uint16_t> chunk_hidden(2'560);
        CUDA_CHECK(cudaMemcpy(chunk_hidden.data(), alloc_chunk.round_tensors().final_hidden.data,
                              chunk_hidden.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost));

        // 3. Compare final hidden and logits equality within numerical tolerances
        auto bf16_to_f = [](std::uint16_t v) {
            return std::bit_cast<float>(static_cast<std::uint32_t>(v) << 16U);
        };
        double hid_diff = 0.0, hid_ref = 0.0;
        for (std::size_t i = 0; i < seq_hidden.size(); ++i) {
            float a = bf16_to_f(seq_hidden[i]);
            float b = bf16_to_f(chunk_hidden[i]);
            if (std::isnan(a) || std::isnan(b) || std::isinf(a) || std::isinf(b)) {
                std::cerr << "Non-finite value in prefill chunk hidden state\n";
                return 1;
            }
            hid_diff += (a - b) * (a - b);
            hid_ref += a * a;
        }
        double hid_rel_l2 = std::sqrt(hid_diff) / std::max(1e-6, std::sqrt(hid_ref));
        if (hid_rel_l2 > 2e-2) {
            std::cerr << "Final hidden rel-L2 mismatch: " << hid_rel_l2 << " > 2e-2\n";
            return 1;
        }

        double log_diff = 0.0, log_ref = 0.0;
        std::size_t seq_argmax = 0, chunk_argmax = 0;
        float seq_max = -1e30F, chunk_max = -1e30F;
        for (std::size_t i = 0; i < seq_logits.size(); ++i) {
            float a = bf16_to_f(seq_logits[i]);
            float b = bf16_to_f(chunk_logits[i]);
            if (std::isnan(a) || std::isnan(b) || std::isinf(a) || std::isinf(b)) {
                std::cerr << "Non-finite value in prefill chunk logits\n";
                return 1;
            }
            log_diff += (a - b) * (a - b);
            log_ref += a * a;
            if (a > seq_max) { seq_max = a; seq_argmax = i; }
            if (b > chunk_max) { chunk_max = b; chunk_argmax = i; }
        }
        double log_rel_l2 = std::sqrt(log_diff) / std::max(1e-6, std::sqrt(log_ref));
        if (log_rel_l2 > 2e-2) {
            std::cerr << "Logits rel-L2 mismatch: " << log_rel_l2 << " > 2e-2\n";
            return 1;
        }
        if (seq_argmax != chunk_argmax) {
            std::cerr << "Argmax mismatch: seq=" << seq_argmax << " chunk=" << chunk_argmax << "\n";
            return 1;
        }

        exec_seq.release_lane(lane_seq);
        exec_chunk.release_lane(lane_chunk);

        std::cout << "PASS: test_prefill_chunk_executor\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "test_prefill_chunk_executor exception: " << e.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "test_prefill_chunk_executor unknown exception\n";
        return 1;
    }
}

int test_prefill_chunk_workspace_envelope(ninfer::DeviceContext& device) {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    try {
        PleIndexMetadata ple_meta{};
        ple_meta.multipliers = {1, 2, 3};
        ple_meta.head_offsets.fill(0);
        ple_meta.head_vocab_sizes.fill(1);

        auto synthetic_model = make_synthetic_model(device);

        constexpr std::int32_t kChunk = 128;
        FlashNextRuntimeConfig cfg{
            .max_concurrency     = 1,
            .max_context         = 512,
            .state_slot_capacity = 2,
            .prefill_chunk       = kChunk,
        };
        const auto curve = flash_next_capacity_curve(cfg);
        auto plan        = finalize_flash_next_runtime_plan(cfg, curve.maximum_main_page_groups);

        FlashNextRuntimeAllocation alloc(plan);
        alloc.initialize(device.stream);
        FlashNextTextExecutor exec(synthetic_model.view, ple_meta, device, alloc);
        auto lane = exec.allocate_lane();

        for (std::int32_t chunk = 0; chunk < 2; ++chunk) {
            std::vector<std::int32_t> tokens(kChunk);
            std::vector<std::array<std::int32_t, 3>> positions(kChunk);
            for (std::int32_t t = 0; t < kChunk; ++t) {
                const std::int32_t index = chunk * kChunk + t;
                tokens[t]                = 100 + index;
                positions[t]             = {index, index, index};
            }
            auto round = exec.execute_prefill_chunk(lane, tokens, positions, chunk * kChunk);
            std::vector<LaneCommitDecision> decision = {{.accept = true}};
            round.commit(decision);
        }
        device.synchronize();

        const std::size_t peak     = alloc.workspace().peak_used();
        const std::size_t capacity = alloc.workspace().capacity();
        std::cout << "prefill chunk T=" << kChunk << " workspace peak " << peak << " / capacity "
                  << capacity << " bytes\n";
        if (peak > capacity) {
            std::cerr << "Workspace peak exceeds capacity\n";
            return 1;
        }

        exec.release_lane(lane);
        std::cout << "PASS: test_prefill_chunk_workspace_envelope\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "test_prefill_chunk_workspace_envelope exception: " << e.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "test_prefill_chunk_workspace_envelope unknown exception\n";
        return 1;
    }
}

int test_cuda_graph_decode_equivalence(ninfer::DeviceContext& device) {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    try {
        PleIndexMetadata ple_meta{};
        ple_meta.multipliers = {1, 2, 3};
        ple_meta.head_offsets.fill(0);
        ple_meta.head_vocab_sizes.fill(1);

        auto synthetic_model = make_synthetic_model(device);

        const std::vector<std::uint32_t> test_batches = {1, 2, 4, 8};
        for (std::uint32_t B : test_batches) {
            FlashNextRuntimeConfig cfg_eager{
                .max_concurrency     = B,
                .max_context         = 512,
                .state_slot_capacity = 2 * B,
                .prefill_chunk       = 512,
                .use_cuda_graph      = false,
            };
            const auto curve_eager = flash_next_capacity_curve(cfg_eager);
            auto plan_eager =
                finalize_flash_next_runtime_plan(cfg_eager, curve_eager.maximum_main_page_groups);

            FlashNextRuntimeAllocation alloc_eager(plan_eager);
            alloc_eager.initialize(device.stream);
            FlashNextTextExecutor exec_eager(synthetic_model.view, ple_meta, device, alloc_eager);

            FlashNextRuntimeConfig cfg_graph{
                .max_concurrency     = B,
                .max_context         = 512,
                .state_slot_capacity = 2 * B,
                .prefill_chunk       = 512,
                .use_cuda_graph      = true,
            };
            const auto curve_graph = flash_next_capacity_curve(cfg_graph);
            auto plan_graph =
                finalize_flash_next_runtime_plan(cfg_graph, curve_graph.maximum_main_page_groups);

            FlashNextRuntimeAllocation alloc_graph(plan_graph);
            alloc_graph.initialize(device.stream);
            FlashNextTextExecutor exec_graph(synthetic_model.view, ple_meta, device, alloc_graph);

            // Captured graphs, then eager body — isolates capture-time state pollution from replay
            FlashNextRuntimeAllocation alloc_mixed(plan_graph);
            alloc_mixed.initialize(device.stream);
            FlashNextTextExecutor exec_mixed(synthetic_model.view, ple_meta, device, alloc_mixed);
            exec_mixed.set_use_cuda_graph(false);

            const std::size_t expected_cap =
                flash_next_text_decode_workspace_capacity_bytes(plan_graph.maximum_blocks, B);
            const std::size_t peak_capture = alloc_graph.workspace().peak_used();
            if (((peak_capture + 255U) & ~255ULL) != expected_cap) {
                std::cerr << "Workspace peak_used after capture mismatch at B=" << B << ": got "
                          << peak_capture << " (aligned " << ((peak_capture + 255U) & ~255ULL)
                          << ") expected " << expected_cap << "\n";
                return 1;
            }

            std::vector<LaneHandle> lanes_eager;
            std::vector<LaneHandle> lanes_graph;
            std::vector<LaneHandle> lanes_mixed;
            for (std::uint32_t b = 0; b < B; ++b) {
                lanes_eager.push_back(exec_eager.allocate_lane());
                lanes_graph.push_back(exec_graph.allocate_lane());
                lanes_mixed.push_back(exec_mixed.allocate_lane());
            }

            constexpr std::size_t kRounds = 8;
            for (std::size_t step = 0; step < kRounds; ++step) {
                std::vector<LaneStepRequest> reqs_eager(B);
                std::vector<LaneStepRequest> reqs_graph(B);
                for (std::uint32_t b = 0; b < B; ++b) {
                    reqs_eager[b] = LaneStepRequest{
                        .handle          = lanes_eager[b],
                        .token_id        = static_cast<std::int32_t>(100 + step * 10 + b),
                        .token_index     = static_cast<std::int32_t>(step),
                        .mrope_positions = {static_cast<std::int32_t>(step),
                                            static_cast<std::int32_t>(step),
                                            static_cast<std::int32_t>(step)},
                        .sampling        = {.temperature = 0.0F, .top_p = 1.0F},
                    };
                    reqs_graph[b] = LaneStepRequest{
                        .handle          = lanes_graph[b],
                        .token_id        = static_cast<std::int32_t>(100 + step * 10 + b),
                        .token_index     = static_cast<std::int32_t>(step),
                        .mrope_positions = {static_cast<std::int32_t>(step),
                                            static_cast<std::int32_t>(step),
                                            static_cast<std::int32_t>(step)},
                        .sampling        = {.temperature = 0.0F, .top_p = 1.0F},
                    };
                }

                std::vector<LaneStepRequest> reqs_mixed = reqs_graph;
                for (std::uint32_t b = 0; b < B; ++b) { reqs_mixed[b].handle = lanes_mixed[b]; }
                auto round_eager = exec_eager.execute_round(reqs_eager);
                auto round_graph = exec_graph.execute_round(reqs_graph);
                auto round_mixed = exec_mixed.execute_round(reqs_mixed);

                // 1. Bit-exact sampled tokens
                auto eager_tokens = round_eager.sampled_tokens();
                auto graph_tokens = round_graph.sampled_tokens();
                auto mixed_tokens = round_mixed.sampled_tokens();
                for (std::uint32_t b = 0; b < B; ++b) {
                    if (eager_tokens[b] != graph_tokens[b]) {
                        std::cerr << "Sampled token mismatch at B=" << B << " step=" << step
                                  << " b=" << b << ": eager=" << eager_tokens[b]
                                  << " graph=" << graph_tokens[b] << "\n";
                        return 1;
                    }
                    if (eager_tokens[b] != mixed_tokens[b]) {
                        std::cerr << "Sampled token mismatch at B=" << B << " step=" << step
                                  << " b=" << b << ": eager=" << eager_tokens[b]
                                  << " mixed=" << mixed_tokens[b] << "\n";
                        return 1;
                    }
                }

                // 2. Bit-exact logits
                std::vector<std::uint16_t> logits_eager(248'320 * B);
                std::vector<std::uint16_t> logits_graph(248'320 * B);
                std::vector<std::uint16_t> logits_mixed(248'320 * B);

                device.synchronize();
                CUDA_CHECK(cudaMemcpy(logits_eager.data(), round_eager.logits().data,
                                      logits_eager.size() * sizeof(std::uint16_t),
                                      cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(logits_graph.data(), round_graph.logits().data,
                                      logits_graph.size() * sizeof(std::uint16_t),
                                      cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(logits_mixed.data(), round_mixed.logits().data,
                                      logits_mixed.size() * sizeof(std::uint16_t),
                                      cudaMemcpyDeviceToHost));

                // Assert finite and exact bit identity
                for (std::size_t i = 0; i < logits_eager.size(); ++i) {
                    const float e = bf16_to_float(logits_eager[i]);
                    if (std::isnan(e) || std::isinf(e)) {
                        std::cerr << "Non-finite eager logit at index " << i << "\n";
                        return 1;
                    }
                    if (logits_eager[i] != logits_graph[i]) {
                        std::cerr << "Logits bit-mismatch (eager vs graph) at B=" << B
                                  << " step=" << step << " idx=" << i << ": eager=0x"
                                  << std::hex << logits_eager[i] << " graph=0x"
                                  << logits_graph[i] << std::dec << "\n";
                        return 1;
                    }
                    if (logits_eager[i] != logits_mixed[i]) {
                        std::cerr << "Logits bit-mismatch (eager vs mixed) at B=" << B
                                  << " step=" << step << " idx=" << i << ": eager=0x"
                                  << std::hex << logits_eager[i] << " mixed=0x"
                                  << logits_mixed[i] << std::dec << "\n";
                        return 1;
                    }
                }

                std::vector<LaneCommitDecision> decisions(B, {.accept = true});
                round_eager.commit(decisions);
                round_graph.commit(decisions);
                round_mixed.commit(decisions);
                device.synchronize();

                if (((alloc_graph.workspace().peak_used() + 255U) & ~255ULL) != expected_cap ||
                    alloc_graph.workspace().peak_used() != peak_capture) {
                    std::cerr << "Workspace peak_used after replay mismatch at B=" << B << ": got "
                              << alloc_graph.workspace().peak_used() << " expected " << peak_capture
                              << "\n";
                    return 1;
                }
            }

            for (std::uint32_t b = 0; b < B; ++b) {
                exec_eager.release_lane(lanes_eager[b]);
                exec_graph.release_lane(lanes_graph[b]);
                exec_mixed.release_lane(lanes_mixed[b]);
            }
        }
        std::cout << "PASS: test_cuda_graph_decode_equivalence\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "test_cuda_graph_decode_equivalence exception: " << e.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "test_cuda_graph_decode_equivalence unknown exception\n";
        return 1;
    }
}

int test_measure_cuda_graph_footprint(ninfer::DeviceContext& device) {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    try {
        PleIndexMetadata ple_meta{};
        ple_meta.multipliers = {1, 2, 3};
        ple_meta.head_offsets.fill(0);
        ple_meta.head_vocab_sizes.fill(1);

        auto synthetic_model = make_synthetic_model(device);

        std::size_t total_footprint = 0;
        std::cout << "--- CUDA Graph Device Footprint (Synthetic Model, B=1..8) ---\n";

        for (std::uint32_t B = 1; B <= 8; ++B) {
            FlashNextRuntimeConfig cfg{
                .max_concurrency     = B,
                .max_context         = 512,
                .state_slot_capacity = 2 * B,
                .prefill_chunk       = 512,
                .use_cuda_graph      = true,
            };
            const auto curve = flash_next_capacity_curve(cfg);
            auto plan = finalize_flash_next_runtime_plan(cfg, curve.maximum_main_page_groups);

            FlashNextRuntimeAllocation alloc(plan);
            alloc.initialize(device.stream);
            device.synchronize();

            std::size_t free_before = 0, total_mem = 0;
            CUDA_CHECK(cudaMemGetInfo(&free_before, &total_mem));

            {
                FlashNextTextExecutor exec(synthetic_model.view, ple_meta, device, alloc);
                device.synchronize();

                std::size_t free_after = 0;
                CUDA_CHECK(cudaMemGetInfo(&free_after, &total_mem));

                std::size_t footprint = (free_before > free_after) ? (free_before - free_after) : 0;
                total_footprint += footprint;

                std::cout << "  B=" << B << " graph footprint: " << (footprint / (1024.0 * 1024.0))
                          << " MiB (" << footprint << " bytes)\n";
            }
            device.synchronize();
        }
        std::cout << "Total measured delta (8 graphs): "
                  << (total_footprint / (1024.0 * 1024.0)) << " MiB (" << total_footprint << " bytes)\n";
        const auto buckets = flash_next_decode_graph_buckets(512 / 4);
        std::cout << "Configured allowance for B=8, n_buckets=" << buckets.count << ": "
                  << (8ULL * buckets.count * 24) << " MiB ("
                  << (8ULL * buckets.count * 24ULL * 1024ULL * 1024ULL) << " bytes)\n";
        std::cout << "Note: cudaMemGetInfo reports OS-level device allocations; CUDA graph exec structures\n"
                  << "are serviced from driver-internal memory pools that do not alter physical OS page\n"
                  << "counters on small synthetic models. The allowance is 24 MiB * max_concurrency * n_buckets.\n";
        std::cout << "------------------------------------------------------------\n";
        std::cout << "PASS: test_measure_cuda_graph_footprint\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "test_measure_cuda_graph_footprint exception: " << e.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "test_measure_cuda_graph_footprint unknown exception\n";
        return 1;
    }
}

int test_cuda_graph_frontier_masking_and_churn(ninfer::DeviceContext& device) {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    try {
        PleIndexMetadata ple_meta{};
        ple_meta.multipliers = {1, 2, 3};
        ple_meta.head_offsets.fill(0);
        ple_meta.head_vocab_sizes.fill(1);

        auto synthetic_model = make_synthetic_model(device);

        FlashNextRuntimeConfig cfg{
            .max_concurrency     = 4,
            .max_context         = 512,
            .state_slot_capacity = 8,
            .prefill_chunk       = 512,
            .use_cuda_graph      = true,
        };
        const auto curve = flash_next_capacity_curve(cfg);
        auto plan = finalize_flash_next_runtime_plan(cfg, curve.maximum_main_page_groups);

        FlashNextRuntimeAllocation alloc(plan);
        alloc.initialize(device.stream);
        FlashNextTextExecutor exec(synthetic_model.view, ple_meta, device, alloc);

        // 1. Allocate 4 lanes
        auto lane0 = exec.allocate_lane();
        auto lane1 = exec.allocate_lane();
        auto lane2 = exec.allocate_lane();
        auto lane3 = exec.allocate_lane();

        // 2. Prefill lane0 up to frontier 250 (nearing group boundary 256)
        std::vector<std::int32_t> prefill_tokens(250, 100);
        std::vector<std::array<std::int32_t, 3>> prefill_pos(250);
        for (std::int32_t i = 0; i < 250; ++i) { prefill_pos[i] = {i, i, i}; }
        auto pr = exec.execute_prefill_chunk(lane0, prefill_tokens, prefill_pos, 0);
        std::vector<LaneCommitDecision> accept1 = {{.accept = true}};
        pr.commit(accept1);
        device.synchronize();

        // 3. Step lane0 across boundary (N=250 -> 260) with batch B=1 using graph replay
        for (std::int32_t step = 250; step < 260; ++step) {
            LaneStepRequest req{
                .handle          = lane0,
                .token_id        = 100 + step,
                .token_index     = step,
                .mrope_positions = {step, step, step},
                .sampling        = {.temperature = 0.0F, .top_p = 1.0F},
            };
            auto round = exec.execute_round(std::span(&req, 1));
            round.commit(accept1);
            device.synchronize();
        }

        if (exec.committed_frontier(lane0) != 260) {
            std::cerr << "Frontier expected 260 after crossing boundary, got "
                      << exec.committed_frontier(lane0) << "\n";
            return 1;
        }

        // 4. Lane churn and slot recycling under graph replay:
        // Release lane 1 and lane 2
        exec.release_lane(lane1);
        exec.release_lane(lane2);

        // Step remaining lanes (lane 0 and lane 3) as B=2 round
        std::vector<LaneStepRequest> reqs_b2 = {
            {.handle          = lane0,
             .token_id        = 400,
             .token_index     = 260,
             .mrope_positions = {260, 260, 260},
             .sampling        = {.temperature = 0.0F, .top_p = 1.0F}},
            {.handle          = lane3,
             .token_id        = 401,
             .token_index     = 0,
             .mrope_positions = {0, 0, 0},
             .sampling        = {.temperature = 0.0F, .top_p = 1.0F}},
        };
        auto round_b2 = exec.execute_round(reqs_b2);
        std::vector<LaneCommitDecision> accept2(2, {.accept = true});
        round_b2.commit(accept2);
        device.synchronize();

        // Reallocate 2 new lanes (will recycle previously freed state slots)
        auto lane4 = exec.allocate_lane();
        auto lane5 = exec.allocate_lane();

        // Step all 4 lanes as B=4 round
        std::vector<LaneStepRequest> reqs_b4 = {
            {.handle          = lane0,
             .token_id        = 500,
             .token_index     = 261,
             .mrope_positions = {261, 261, 261},
             .sampling        = {.temperature = 0.0F, .top_p = 1.0F}},
            {.handle          = lane3,
             .token_id        = 501,
             .token_index     = 1,
             .mrope_positions = {1, 1, 1},
             .sampling        = {.temperature = 0.0F, .top_p = 1.0F}},
            {.handle          = lane4,
             .token_id        = 502,
             .token_index     = 0,
             .mrope_positions = {0, 0, 0},
             .sampling        = {.temperature = 0.0F, .top_p = 1.0F}},
            {.handle          = lane5,
             .token_id        = 503,
             .token_index     = 0,
             .mrope_positions = {0, 0, 0},
             .sampling        = {.temperature = 0.0F, .top_p = 1.0F}},
        };
        auto round_b4 = exec.execute_round(reqs_b4);
        std::vector<LaneCommitDecision> accept4(4, {.accept = true});
        round_b4.commit(accept4);
        device.synchronize();

        exec.release_lane(lane0);
        exec.release_lane(lane3);
        exec.release_lane(lane4);
        exec.release_lane(lane5);

        std::cout << "PASS: test_cuda_graph_frontier_masking_and_churn\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "test_cuda_graph_frontier_masking_and_churn exception: " << e.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "test_cuda_graph_frontier_masking_and_churn unknown exception\n";
        return 1;
    }
}

int test_cuda_graph_timing_benchmark(ninfer::DeviceContext& device) {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    try {
        PleIndexMetadata ple_meta{};
        ple_meta.multipliers = {1, 2, 3};
        ple_meta.head_offsets.fill(0);
        ple_meta.head_vocab_sizes.fill(1);

        auto synthetic_model = make_synthetic_model(device);

        const std::vector<std::uint32_t> benchmark_batches = {1, 8};
        constexpr int kWarmupRounds = 5;
        constexpr int kBenchRounds  = 30;

        std::cout << "--- Decode Performance Benchmark (Synthetic Model) ---\n";
        for (std::uint32_t B : benchmark_batches) {
            // 1. Eager mode
            FlashNextRuntimeConfig cfg_eager{
                .max_concurrency     = B,
                .max_context         = 512,
                .state_slot_capacity = 2 * B,
                .prefill_chunk       = 512,
                .use_cuda_graph      = false,
            };
            const auto curve_eager = flash_next_capacity_curve(cfg_eager);
            auto plan_eager =
                finalize_flash_next_runtime_plan(cfg_eager, curve_eager.maximum_main_page_groups);
            FlashNextRuntimeAllocation alloc_eager(plan_eager);
            alloc_eager.initialize(device.stream);
            FlashNextTextExecutor exec_eager(synthetic_model.view, ple_meta, device, alloc_eager);
            std::vector<LaneHandle> lanes_eager;
            for (std::uint32_t b = 0; b < B; ++b) {
                lanes_eager.push_back(exec_eager.allocate_lane());
            }

            std::vector<LaneStepRequest> reqs_eager(B);
            std::vector<LaneCommitDecision> accept(B, {.accept = true});

            int step_eager = 0;
            for (int r = 0; r < kWarmupRounds; ++r, ++step_eager) {
                for (std::uint32_t b = 0; b < B; ++b) {
                    reqs_eager[b] = {
                        .handle          = lanes_eager[b],
                        .token_id        = static_cast<std::int32_t>(100 + b),
                        .token_index     = step_eager,
                        .mrope_positions = {step_eager, step_eager, step_eager},
                        .sampling        = {.temperature = 0.0F, .top_p = 1.0F},
                    };
                }
                auto rd = exec_eager.execute_round(reqs_eager);
                rd.commit(accept);
                device.synchronize();
            }

            auto start_eager = std::chrono::high_resolution_clock::now();
            for (int r = 0; r < kBenchRounds; ++r, ++step_eager) {
                for (std::uint32_t b = 0; b < B; ++b) {
                    reqs_eager[b] = {
                        .handle          = lanes_eager[b],
                        .token_id        = static_cast<std::int32_t>(100 + b),
                        .token_index     = step_eager,
                        .mrope_positions = {step_eager, step_eager, step_eager},
                        .sampling        = {.temperature = 0.0F, .top_p = 1.0F},
                    };
                }
                auto rd = exec_eager.execute_round(reqs_eager);
                rd.commit(accept);
            }
            device.synchronize();
            auto end_eager = std::chrono::high_resolution_clock::now();
            double eager_us_per_round =
                std::chrono::duration<double, std::micro>(end_eager - start_eager).count() /
                kBenchRounds;
            double eager_tok_per_sec = (B * 1e6) / eager_us_per_round;

            // 2. Graph Replay mode
            FlashNextRuntimeConfig cfg_graph{
                .max_concurrency     = B,
                .max_context         = 512,
                .state_slot_capacity = 2 * B,
                .prefill_chunk       = 512,
                .use_cuda_graph      = true,
            };
            const auto curve_graph = flash_next_capacity_curve(cfg_graph);
            auto plan_graph =
                finalize_flash_next_runtime_plan(cfg_graph, curve_graph.maximum_main_page_groups);
            FlashNextRuntimeAllocation alloc_graph(plan_graph);
            alloc_graph.initialize(device.stream);
            FlashNextTextExecutor exec_graph(synthetic_model.view, ple_meta, device, alloc_graph);
            std::vector<LaneHandle> lanes_graph;
            for (std::uint32_t b = 0; b < B; ++b) {
                lanes_graph.push_back(exec_graph.allocate_lane());
            }

            std::vector<LaneStepRequest> reqs_graph(B);
            int step_graph = 0;
            for (int r = 0; r < kWarmupRounds; ++r, ++step_graph) {
                for (std::uint32_t b = 0; b < B; ++b) {
                    reqs_graph[b] = {
                        .handle          = lanes_graph[b],
                        .token_id        = static_cast<std::int32_t>(100 + b),
                        .token_index     = step_graph,
                        .mrope_positions = {step_graph, step_graph, step_graph},
                        .sampling        = {.temperature = 0.0F, .top_p = 1.0F},
                    };
                }
                auto rd = exec_graph.execute_round(reqs_graph);
                rd.commit(accept);
                device.synchronize();
            }

            auto start_graph = std::chrono::high_resolution_clock::now();
            for (int r = 0; r < kBenchRounds; ++r, ++step_graph) {
                for (std::uint32_t b = 0; b < B; ++b) {
                    reqs_graph[b] = {
                        .handle          = lanes_graph[b],
                        .token_id        = static_cast<std::int32_t>(100 + b),
                        .token_index     = step_graph,
                        .mrope_positions = {step_graph, step_graph, step_graph},
                        .sampling        = {.temperature = 0.0F, .top_p = 1.0F},
                    };
                }
                auto rd = exec_graph.execute_round(reqs_graph);
                rd.commit(accept);
            }
            device.synchronize();
            auto end_graph = std::chrono::high_resolution_clock::now();
            double graph_us_per_round =
                std::chrono::duration<double, std::micro>(end_graph - start_graph).count() /
                kBenchRounds;
            double graph_tok_per_sec = (B * 1e6) / graph_us_per_round;

            std::cout << "B=" << B << " | Eager: " << eager_us_per_round << " us/round ("
                      << eager_tok_per_sec << " tok/s) | Graph Replay: " << graph_us_per_round
                      << " us/round (" << graph_tok_per_sec << " tok/s) | Speedup: "
                      << (eager_us_per_round / graph_us_per_round) << "x\n";
        }
        std::cout << "--------------------------------------------------------\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "test_cuda_graph_timing_benchmark exception: " << e.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "test_cuda_graph_timing_benchmark unknown exception\n";
        return 1;
    }
}

int test_prefill_chunk_timing_benchmark(ninfer::DeviceContext& device, bool mode_ab,
                                        bool host_sync_ab = false, bool moe_shared_ab = false,
                                        bool qsa_sched_ab = false,
                                        bool moe_staging_ab = false) {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    try {
        PleIndexMetadata ple_meta{};
        ple_meta.multipliers = {1, 2, 3};
        ple_meta.head_offsets.fill(0);
        ple_meta.head_vocab_sizes.fill(100);

        auto synthetic_model = make_synthetic_model(device);

        const bool interleaved_ab =
            prefill_bench_ab_requested(mode_ab) || host_sync_ab || moe_shared_ab || qsa_sched_ab ||
            moe_staging_ab;
        const int bench_iters     = prefill_bench_iters();
        const int warmup_s        = prefill_bench_warmup_s(interleaved_ab);
        const char* kind          = moe_staging_ab ? "moe-staging"
                                    : qsa_sched_ab  ? "qsa-mma-sched"
                                    : moe_shared_ab ? "moe-shared-mma"
                                    : host_sync_ab  ? "host-sync"
                                                    : "residual-splitk";
        std::cout << "--- Prefill Chunk Performance & Host CPU Benchmark (Synthetic Model) ---\n";
        std::cout << "NINFER_PREFILL_BENCH_ITERS=" << bench_iters
                  << " WARMUP_S=" << warmup_s
                  << " AB=" << (interleaved_ab ? "1" : "0")
                  << " kind=" << kind
                  << " (one plan/executor; env reread at each launch)\n";

        const std::vector<std::int32_t> chunk_sizes =
            (moe_shared_ab || qsa_sched_ab || moe_staging_ab) ? std::vector<std::int32_t>{2048, 512}
            : interleaved_ab ? std::vector<std::int32_t>{2048, 128}
                             : std::vector<std::int32_t>{128, 2048};
        if (qsa_sched_ab) { set_qsa_prefill_mma_env(true); }
        bool warmed = false;
        for (std::int32_t T : chunk_sizes) {
            FlashNextRuntimeConfig cfg{
                .max_concurrency     = 1,
                .max_context         = 16384,
                .state_slot_capacity = 2,
                .prefill_chunk       = static_cast<std::uint32_t>(T),
                .use_qsa_prefill_mma = qsa_sched_ab,
            };
            const auto curve = flash_next_capacity_curve(cfg);
            auto plan        = finalize_flash_next_runtime_plan(cfg, curve.maximum_main_page_groups);

            FlashNextRuntimeAllocation alloc(plan);
            alloc.initialize(device.stream);
            FlashNextTextExecutor exec(synthetic_model.view, ple_meta, device, alloc);

            std::vector<std::int32_t> tokens(T);
            std::vector<std::array<std::int32_t, 3>> positions(T);
            for (std::int32_t t = 0; t < T; ++t) {
                tokens[t]    = 100 + t;
                positions[t] = {t, t, t};
            }

            auto lane_w = exec.allocate_lane();
            auto w_rd = exec.execute_prefill_chunk(lane_w, tokens, positions, 0);
            std::vector<LaneCommitDecision> accept = {{.accept = true}};
            w_rd.commit(accept);
            exec.release_lane(lane_w);
            device.synchronize();

            cudaEvent_t start_event = nullptr;
            cudaEvent_t stop_event  = nullptr;
            CUDA_CHECK(cudaEventCreate(&start_event));
            CUDA_CHECK(cudaEventCreate(&stop_event));

            auto time_one_chunk_ms = [&]() -> float {
                auto lane = exec.allocate_lane();
                CUDA_CHECK(cudaEventRecord(start_event, device.stream));
                auto rd = exec.execute_prefill_chunk(lane, tokens, positions, 0);
                rd.commit(accept);
                CUDA_CHECK(cudaEventRecord(stop_event, device.stream));
                CUDA_CHECK(cudaEventSynchronize(stop_event));
                exec.release_lane(lane);
                float elapsed_ms = 0.0F;
                CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event));
                return elapsed_ms;
            };

            const int run_warmup = (!warmed && warmup_s > 0) ? warmup_s : 0;
            if (run_warmup > 0) {
                if (moe_staging_ab) {
                    set_moe_staging_env(true);
                } else if (qsa_sched_ab) {
                    set_qsa_mma_sched_env(false);
                } else if (moe_shared_ab) {
                    set_moe_shared_mma_env(true);
                } else if (host_sync_ab) {
                    set_prefill_host_sync_env(false);
                } else {
                    set_residual_splitk_env(4);
                }
                const auto warm_t0 = std::chrono::steady_clock::now();
                int warm_chunks    = 0;
                while (std::chrono::duration<double>(std::chrono::steady_clock::now() - warm_t0)
                           .count() < static_cast<double>(run_warmup)) {
                    (void)time_one_chunk_ms();
                    ++warm_chunks;
                }
                warmed = true;
                std::cout << "  T=" << T << " warmup " << run_warmup << " s (" << warm_chunks
                          << " chunks, "
                          << (moe_staging_ab ? "staging=new"
                              : qsa_sched_ab   ? "qsa_sched=old"
                              : moe_shared_ab  ? "shared_mma=1"
                              : host_sync_ab   ? "host_sync=0"
                                               : "splitk=4")
                          << ")\n";
            }

            if (interleaved_ab && moe_staging_ab) {
                set_moe_staging_env(false);
                (void)time_one_chunk_ms();
                set_moe_staging_env(true);
                (void)time_one_chunk_ms();
                std::vector<float> samples_old(static_cast<std::size_t>(bench_iters));
                std::vector<float> samples_new(static_cast<std::size_t>(bench_iters));
                for (int i = 0; i < bench_iters; ++i) {
                    set_moe_staging_env(false);
                    samples_old[static_cast<std::size_t>(i)] = time_one_chunk_ms();
                    set_moe_staging_env(true);
                    samples_new[static_cast<std::size_t>(i)] = time_one_chunk_ms();
                }
                const PrefillBenchStats s_old = prefill_bench_stats(samples_old);
                const PrefillBenchStats s_new = prefill_bench_stats(samples_new);
                const float paired            = s_old.median_ms - s_new.median_ms;
                std::cout << std::fixed << std::setprecision(3);
                std::cout << "  Prefill T=" << T << " AB n=" << bench_iters
                          << " paired_median_delta=" << paired
                          << " ms (staging=old minus staging=new; + = new staging faster)\n";
                std::cout << "    arm staging=old min=" << s_old.min_ms
                          << " median=" << s_old.median_ms << " max=" << s_old.max_ms
                          << " mean=" << s_old.mean_ms
                          << " span=" << (s_old.max_ms - s_old.min_ms) << " ms\n";
                std::cout << "    arm staging=new min=" << s_new.min_ms
                          << " median=" << s_new.median_ms << " max=" << s_new.max_ms
                          << " mean=" << s_new.mean_ms
                          << " span=" << (s_new.max_ms - s_new.min_ms) << " ms\n";
            } else if (interleaved_ab && qsa_sched_ab) {
                set_qsa_mma_sched_env(false);
                (void)time_one_chunk_ms();
                set_qsa_mma_sched_env(true);
                (void)time_one_chunk_ms();
                std::vector<float> samples_old(static_cast<std::size_t>(bench_iters));
                std::vector<float> samples_new(static_cast<std::size_t>(bench_iters));
                for (int i = 0; i < bench_iters; ++i) {
                    set_qsa_mma_sched_env(false);
                    samples_old[static_cast<std::size_t>(i)] = time_one_chunk_ms();
                    set_qsa_mma_sched_env(true);
                    samples_new[static_cast<std::size_t>(i)] = time_one_chunk_ms();
                }
                const PrefillBenchStats s_old = prefill_bench_stats(samples_old);
                const PrefillBenchStats s_new = prefill_bench_stats(samples_new);
                const float paired            = s_old.median_ms - s_new.median_ms;
                std::cout << std::fixed << std::setprecision(3);
                std::cout << "  Prefill T=" << T << " AB n=" << bench_iters
                          << " paired_median_delta=" << paired
                          << " ms (qsa_sched=old minus new; + = new faster)\n";
                std::cout << "    arm qsa_sched=old min=" << s_old.min_ms
                          << " median=" << s_old.median_ms << " max=" << s_old.max_ms
                          << " mean=" << s_old.mean_ms
                          << " span=" << (s_old.max_ms - s_old.min_ms) << " ms\n";
                std::cout << "    arm qsa_sched=new min=" << s_new.min_ms
                          << " median=" << s_new.median_ms << " max=" << s_new.max_ms
                          << " mean=" << s_new.mean_ms
                          << " span=" << (s_new.max_ms - s_new.min_ms) << " ms\n";
            } else if (interleaved_ab && moe_shared_ab) {
                set_moe_shared_mma_env(false);
                (void)time_one_chunk_ms();
                set_moe_shared_mma_env(true);
                (void)time_one_chunk_ms();
                std::vector<float> samples_old(static_cast<std::size_t>(bench_iters));
                std::vector<float> samples_new(static_cast<std::size_t>(bench_iters));
                for (int i = 0; i < bench_iters; ++i) {
                    set_moe_shared_mma_env(false);
                    samples_old[static_cast<std::size_t>(i)] = time_one_chunk_ms();
                    set_moe_shared_mma_env(true);
                    samples_new[static_cast<std::size_t>(i)] = time_one_chunk_ms();
                }
                const PrefillBenchStats s_old = prefill_bench_stats(samples_old);
                const PrefillBenchStats s_new = prefill_bench_stats(samples_new);
                const float paired            = s_old.median_ms - s_new.median_ms;
                std::cout << std::fixed << std::setprecision(3);
                std::cout << "  Prefill T=" << T << " AB n=" << bench_iters
                          << " paired_median_delta=" << paired
                          << " ms (shared_mma=0 minus shared_mma=1; + = MMA faster)\n";
                std::cout << "    arm shared_mma=0 min=" << s_old.min_ms
                          << " median=" << s_old.median_ms << " max=" << s_old.max_ms
                          << " mean=" << s_old.mean_ms
                          << " span=" << (s_old.max_ms - s_old.min_ms) << " ms\n";
                std::cout << "    arm shared_mma=1 min=" << s_new.min_ms
                          << " median=" << s_new.median_ms << " max=" << s_new.max_ms
                          << " mean=" << s_new.mean_ms
                          << " span=" << (s_new.max_ms - s_new.min_ms) << " ms\n";
            } else if (interleaved_ab && host_sync_ab) {
                set_prefill_host_sync_env(true);
                (void)time_one_chunk_ms();
                set_prefill_host_sync_env(false);
                (void)time_one_chunk_ms();
                std::vector<float> samples_old(static_cast<std::size_t>(bench_iters));
                std::vector<float> samples_new(static_cast<std::size_t>(bench_iters));
                for (int i = 0; i < bench_iters; ++i) {
                    set_prefill_host_sync_env(true);
                    samples_old[static_cast<std::size_t>(i)] = time_one_chunk_ms();
                    set_prefill_host_sync_env(false);
                    samples_new[static_cast<std::size_t>(i)] = time_one_chunk_ms();
                }
                const PrefillBenchStats s_old = prefill_bench_stats(samples_old);
                const PrefillBenchStats s_new = prefill_bench_stats(samples_new);
                const float paired            = s_old.median_ms - s_new.median_ms;
                std::cout << std::fixed << std::setprecision(3);
                std::cout << "  Prefill T=" << T << " AB n=" << bench_iters
                          << " paired_median_delta=" << paired
                          << " ms (host_sync=1 minus host_sync=0; + = new path faster)\n";
                std::cout << "    arm host_sync=1 min=" << s_old.min_ms
                          << " median=" << s_old.median_ms << " max=" << s_old.max_ms
                          << " mean=" << s_old.mean_ms
                          << " span=" << (s_old.max_ms - s_old.min_ms) << " ms\n";
                std::cout << "    arm host_sync=0 min=" << s_new.min_ms
                          << " median=" << s_new.median_ms << " max=" << s_new.max_ms
                          << " mean=" << s_new.mean_ms
                          << " span=" << (s_new.max_ms - s_new.min_ms) << " ms\n";
            } else if (interleaved_ab) {
                set_residual_splitk_env(4);
                (void)time_one_chunk_ms();
                set_residual_splitk_env(1);
                (void)time_one_chunk_ms();
                std::vector<float> samples4(static_cast<std::size_t>(bench_iters));
                std::vector<float> samples1(static_cast<std::size_t>(bench_iters));
                for (int i = 0; i < bench_iters; ++i) {
                    set_residual_splitk_env(4);
                    samples4[static_cast<std::size_t>(i)] = time_one_chunk_ms();
                    set_residual_splitk_env(1);
                    samples1[static_cast<std::size_t>(i)] = time_one_chunk_ms();
                }
                const PrefillBenchStats s4 = prefill_bench_stats(samples4);
                const PrefillBenchStats s1 = prefill_bench_stats(samples1);
                const float paired         = s4.median_ms - s1.median_ms;
                std::cout << std::fixed << std::setprecision(3);
                std::cout << "  Prefill T=" << T << " AB n=" << bench_iters
                          << " paired_median_delta=" << paired << " ms (split4-split1; + = unsplit faster)\n";
                std::cout << "    arm splitk=4 min=" << s4.min_ms << " median=" << s4.median_ms
                          << " max=" << s4.max_ms << " mean=" << s4.mean_ms
                          << " span=" << (s4.max_ms - s4.min_ms) << " ms\n";
                std::cout << "    arm splitk=1 min=" << s1.min_ms << " median=" << s1.median_ms
                          << " max=" << s1.max_ms << " mean=" << s1.mean_ms
                          << " span=" << (s1.max_ms - s1.min_ms) << " ms\n";
            } else {
                (void)time_one_chunk_ms();
                std::vector<float> samples(static_cast<std::size_t>(bench_iters));
                for (int i = 0; i < bench_iters; ++i) {
                    samples[static_cast<std::size_t>(i)] = time_one_chunk_ms();
                }
                const PrefillBenchStats s = prefill_bench_stats(samples);
                const double tok_per_sec =
                    (static_cast<double>(T) * 1.0e3) / static_cast<double>(s.median_ms);
                std::cout << std::fixed << std::setprecision(3);
                std::cout << "  Prefill T=" << T << " : " << s.median_ms << " ms/chunk ("
                          << tok_per_sec << " tok/s)\n";
                std::cout << "    cudaEvent n=" << bench_iters << " discard-first min=" << s.min_ms
                          << " median=" << s.median_ms << " mean=" << s.mean_ms << " max=" << s.max_ms
                          << " ms\n";
            }
            CUDA_CHECK(cudaEventDestroy(start_event));
            CUDA_CHECK(cudaEventDestroy(stop_event));
        }
        std::cout << "------------------------------------------------------------------------\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "test_prefill_chunk_timing_benchmark exception: " << e.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "test_prefill_chunk_timing_benchmark unknown exception\n";
        return 1;
    }
}

bool logits_bit_identical_nonvacuous(ninfer::DeviceContext& device, const ninfer::Tensor& a,
                                     const ninfer::Tensor& b, std::uint32_t batch,
                                     const char* label) {
    const std::size_t n = 248'320ULL * batch;
    std::vector<std::uint16_t> ha(n), hb(n);
    device.synchronize();
    CUDA_CHECK(cudaMemcpy(ha.data(), a.data, n * sizeof(std::uint16_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hb.data(), b.data, n * sizeof(std::uint16_t), cudaMemcpyDeviceToHost));
    double base_sq = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const float fa = bf16_to_float(ha[i]);
        const float fb = bf16_to_float(hb[i]);
        if (!std::isfinite(fa) || !std::isfinite(fb)) {
            std::cerr << "FAIL: " << label << " non-finite logit at " << i << "\n";
            return false;
        }
        if (ha[i] != hb[i]) {
            std::cerr << "FAIL: " << label << " bit-mismatch at " << i << " a=0x" << std::hex
                      << ha[i] << " b=0x" << hb[i] << std::dec << "\n";
            return false;
        }
        base_sq += static_cast<double>(fa) * static_cast<double>(fa);
    }
    if (!(base_sq > 0.0) || !std::isfinite(base_sq)) {
        std::cerr << "FAIL: " << label << " vacuous logits base_sq=" << base_sq << "\n";
        return false;
    }
    return true;
}

int test_decode_graph_bucket_key_layout(ninfer::DeviceContext& device) {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    try {
        PleIndexMetadata ple_meta{};
        ple_meta.multipliers = {1, 2, 3};
        ple_meta.head_offsets.fill(0);
        ple_meta.head_vocab_sizes.fill(1);
        auto synthetic_model = make_synthetic_model(device);

        FlashNextRuntimeConfig cfg{
            .max_concurrency     = 2,
            .max_context         = 8192,
            .state_slot_capacity = 4,
            .prefill_chunk       = 1024,
            .use_cuda_graph      = true,
        };
        const auto curve = flash_next_capacity_curve(cfg);
        auto plan        = finalize_flash_next_runtime_plan(cfg, curve.maximum_main_page_groups);
        FlashNextRuntimeAllocation alloc(plan);
        alloc.initialize(device.stream);
        FlashNextTextExecutor exec(synthetic_model.view, ple_meta, device, alloc);

        const auto& family = exec.decode_graphs();
        if (family.buckets.count != 2 || family.buckets.blocks[0] != 512 ||
            family.buckets.blocks[1] != 2048) {
            std::cerr << "FAIL: 8192-token buckets expected {512,2048} got count="
                      << family.buckets.count << "\n";
            return 1;
        }
        if (family.topologies.size() != 4) {
            std::cerr << "FAIL: startup should capture all buckets for B=1..2 (2 buckets * 2 B = 4), got "
                      << family.topologies.size() << " topologies\n";
            return 1;
        }
        for (const auto& topology : family.topologies) {
            if (topology.bucket_index > 1 || topology.batch_size < 1 || topology.batch_size > 2) {
                std::cerr << "FAIL: startup topology B=" << topology.batch_size
                          << " bucket_index=" << topology.bucket_index << " outside legal range\n";
                return 1;
            }
            const auto want = flash_next_decode_graph_topology_class(topology.batch_size, topology.bucket_index);
            if (topology.topology_class != want) {
                std::cerr << "FAIL: topology_class=" << topology.topology_class
                          << " expected=" << want << "\n";
                return 1;
            }
        }
        for (const auto& profile : family.profiles) {
            if (profile.bucket_index > 1 ||
                profile.bucket_blocks != family.buckets.blocks[profile.bucket_index] ||
                profile.topology_class !=
                    flash_next_decode_graph_topology_class(profile.batch_size, profile.bucket_index)) {
                std::cerr << "FAIL: profile key layout B=" << profile.batch_size
                          << " bucket=" << profile.bucket_index
                          << " blocks=" << profile.bucket_blocks
                          << " class=" << profile.topology_class << "\n";
            }
        }
        // Sequence D23 invariant: for every B in [1, max_concurrency] x bucket,
        // a ready topology exists with profile.bucket_blocks == buckets.blocks[bucket].
        for (std::uint32_t B = 1; B <= 2; ++B) {
            for (std::uint32_t bucket = 0; bucket < family.buckets.count; ++bucket) {
                bool found = false;
                for (const auto& topology : family.topologies) {
                    if (topology.batch_size == B && topology.bucket_index == bucket) {
                        found = true;
                        if (!topology.executable.ready()) {
                            std::cerr << "FAIL: topology B=" << B << " bucket=" << bucket
                                      << " executable is not ready\n";
                            return 1;
                        }
                        if (!topology.installed_profile.has_value() ||
                            topology.installed_profile.value() >= family.profiles.size()) {
                            std::cerr << "FAIL: topology B=" << B << " bucket=" << bucket
                                      << " installed_profile out of range\n";
                            return 1;
                        }
                        const auto& profile = family.profiles[topology.installed_profile.value()];
                        if (profile.bucket_blocks != family.buckets.blocks[bucket]) {
                            std::cerr << "FAIL: profile B=" << B << " bucket=" << bucket
                                      << " blocks=" << profile.bucket_blocks
                                      << " expected=" << family.buckets.blocks[bucket] << "\n";
                            return 1;
                        }
                        break;
                    }
                }
                if (!found) {
                    std::cerr << "FAIL: missing topology for B=" << B << " bucket=" << bucket << "\n";
                    return 1;
                }
            }
        }
        std::cout << "PASS: test_decode_graph_bucket_key_layout topologies="
                  << family.topologies.size()
                  << " key=(batch_size, bucket_index) topology_class=(bucket_index<<8)|batch_size\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "test_decode_graph_bucket_key_layout exception: " << e.what() << "\n";
        return 1;
    }
}

int test_cuda_graph_bucketed_decode(ninfer::DeviceContext& device) {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    try {
        PleIndexMetadata ple_meta{};
        ple_meta.multipliers = {1, 2, 3};
        ple_meta.head_offsets.fill(0);
        ple_meta.head_vocab_sizes.fill(1);
        auto synthetic_model = make_synthetic_model(device);

        auto make_plan = [](bool use_cuda_graph) {
            FlashNextRuntimeConfig cfg{
                .max_concurrency     = 1,
                .max_context         = 8192,
                .state_slot_capacity = 2,
                .prefill_chunk       = 2048,
                .use_cuda_graph      = use_cuda_graph,
            };
            const auto curve = flash_next_capacity_curve(cfg);
            return finalize_flash_next_runtime_plan(cfg, curve.maximum_main_page_groups);
        };
        auto plan_graph  = make_plan(true);
        auto plan_eager  = make_plan(false);
        auto plan_live   = make_plan(false);
        auto plan_graph2 = make_plan(true);

        FlashNextRuntimeAllocation alloc_graph(plan_graph);
        FlashNextRuntimeAllocation alloc_eager(plan_eager);
        FlashNextRuntimeAllocation alloc_live(plan_live);
        FlashNextRuntimeAllocation alloc_graph2(plan_graph2);
        alloc_graph.initialize(device.stream);
        alloc_eager.initialize(device.stream);
        alloc_live.initialize(device.stream);
        alloc_graph2.initialize(device.stream);

        FlashNextTextExecutor exec_graph(synthetic_model.view, ple_meta, device, alloc_graph);
        FlashNextTextExecutor exec_eager(synthetic_model.view, ple_meta, device, alloc_eager);
        FlashNextTextExecutor exec_live(synthetic_model.view, ple_meta, device, alloc_live);
        FlashNextTextExecutor exec_graph2(synthetic_model.view, ple_meta, device, alloc_graph2);
        exec_eager.set_use_cuda_graph(false);
        exec_live.set_use_cuda_graph(false);

        if (exec_graph.decode_graphs().topologies.size() != 2 ||
            exec_graph.decode_graphs().topologies[0].bucket_index != 0 ||
            exec_graph.decode_graphs().topologies[1].bucket_index != 1) {
            std::cerr << "FAIL: graph executor did not start with eager capture of both buckets\n";
            return 1;
        }

        auto lane_g  = exec_graph.allocate_lane();
        auto lane_e  = exec_eager.allocate_lane();
        auto lane_l  = exec_live.allocate_lane();
        auto lane_g2 = exec_graph2.allocate_lane();

        // Short-context identity bucket: graph vs eager(bucket) vs eager(live) plus run-to-run.
        for (std::int32_t step = 0; step < 8; ++step) {
            LaneStepRequest rg{.handle = lane_g,
                               .token_id = 100 + step,
                               .token_index = step,
                               .mrope_positions = {step, step, step},
                               .sampling = {.temperature = 0.0F, .top_p = 1.0F}};
            LaneStepRequest re = rg;
            re.handle          = lane_e;
            LaneStepRequest rl = rg;
            rl.handle          = lane_l;
            LaneStepRequest rg2 = rg;
            rg2.handle          = lane_g2;

            auto round_g  = exec_graph.execute_round(std::span(&rg, 1));
            auto round_e  = exec_eager.execute_round(std::span(&re, 1));
            const std::int32_t live_blocks = (step + 1) / 4;
            auto round_l  = exec_live.execute_round_eager(std::span(&rl, 1), live_blocks);
            auto round_g2 = exec_graph2.execute_round(std::span(&rg2, 1));

            if (round_g.sampled_tokens()[0] != round_e.sampled_tokens()[0] ||
                round_g.sampled_tokens()[0] != round_l.sampled_tokens()[0] ||
                round_g.sampled_tokens()[0] != round_g2.sampled_tokens()[0]) {
                std::cerr << "FAIL: identity-bucket token mismatch at step=" << step
                          << " graph=" << round_g.sampled_tokens()[0]
                          << " eager=" << round_e.sampled_tokens()[0]
                          << " live=" << round_l.sampled_tokens()[0]
                          << " graph2=" << round_g2.sampled_tokens()[0] << "\n";
                return 1;
            }
            if (!logits_bit_identical_nonvacuous(device, round_g.logits(), round_e.logits(), 1,
                                                 "identity graph vs eager") ||
                !logits_bit_identical_nonvacuous(device, round_g.logits(), round_l.logits(), 1,
                                                 "identity graph vs live") ||
                !logits_bit_identical_nonvacuous(device, round_g.logits(), round_g2.logits(), 1,
                                                 "identity run-to-run")) {
                return 1;
            }
            std::vector<LaneCommitDecision> accept = {{.accept = true}};
            round_g.commit(accept);
            round_e.commit(accept);
            round_l.commit(accept);
            round_g2.commit(accept);
            device.synchronize();
        }

        exec_graph.release_lane(lane_g);
        exec_eager.release_lane(lane_e);
        exec_live.release_lane(lane_l);
        exec_graph2.release_lane(lane_g2);

        lane_g = exec_graph.allocate_lane();
        lane_e = exec_eager.allocate_lane();
        lane_l = exec_live.allocate_lane();

        constexpr std::int32_t kPrefill = 2048;
        std::vector<std::int32_t> tokens(kPrefill);
        std::vector<std::array<std::int32_t, 3>> positions(kPrefill);
        for (std::int32_t t = 0; t < kPrefill; ++t) {
            tokens[t]    = 200 + t;
            positions[t] = {t, t, t};
        }
        auto prefill_one = [&](FlashNextTextExecutor& exec, LaneHandle lane) {
            auto pr = exec.execute_prefill_chunk(lane, tokens, positions, 0);
            std::vector<LaneCommitDecision> accept = {{.accept = true}};
            pr.commit(accept);
            device.synchronize();
        };
        prefill_one(exec_graph, lane_g);
        prefill_one(exec_eager, lane_e);
        prefill_one(exec_live, lane_l);

        if (exec_graph.decode_graphs().topologies.size() != 2) {
            std::cerr << "FAIL: prefill must preserve eager decode bucket count, topologies="
                      << exec_graph.decode_graphs().topologies.size() << "\n";
            return 1;
        }

        bool saw_identity = false;
        bool saw_topk     = false;
        for (std::int32_t step = kPrefill; step < kPrefill + 8; ++step) {
            const std::int32_t live_blocks = (step + 1) / 4;
            LaneStepRequest rg{.handle = lane_g,
                               .token_id = 300 + step,
                               .token_index = step,
                               .mrope_positions = {step, step, step},
                               .sampling = {.temperature = 0.0F, .top_p = 1.0F}};
            LaneStepRequest re = rg;
            re.handle          = lane_e;
            LaneStepRequest rl = rg;
            rl.handle          = lane_l;

            auto round_g = exec_graph.execute_round(std::span(&rg, 1));
            auto round_e = exec_eager.execute_round(std::span(&re, 1));
            auto round_l = exec_live.execute_round_eager(std::span(&rl, 1), live_blocks);

            if (live_blocks <= 512) { saw_identity = true; }
            if (live_blocks > 512) { saw_topk = true; }

            if (round_g.sampled_tokens()[0] != round_e.sampled_tokens()[0] ||
                round_g.sampled_tokens()[0] != round_l.sampled_tokens()[0]) {
                std::cerr << "FAIL: three-way token mismatch at token_index=" << step
                          << " live_blocks=" << live_blocks
                          << " graph=" << round_g.sampled_tokens()[0]
                          << " eager=" << round_e.sampled_tokens()[0]
                          << " live=" << round_l.sampled_tokens()[0] << "\n";
                return 1;
            }
            if (!logits_bit_identical_nonvacuous(device, round_g.logits(), round_e.logits(), 1,
                                                 "bucket graph vs eager") ||
                !logits_bit_identical_nonvacuous(device, round_g.logits(), round_l.logits(), 1,
                                                 "bucket graph vs live frontier")) {
                std::cerr << " at token_index=" << step << " live_blocks=" << live_blocks << "\n";
                return 1;
            }
            std::vector<LaneCommitDecision> accept = {{.accept = true}};
            round_g.commit(accept);
            round_e.commit(accept);
            round_l.commit(accept);
            device.synchronize();
        }

        if (!saw_identity || !saw_topk) {
            std::cerr << "FAIL: crossing test did not cover both sides of the 512-block boundary\n";
            return 1;
        }
        if (exec_graph.decode_graphs().topologies.size() != 2) {
            std::cerr << "FAIL: crossing must reuse eager bucket 1 without runtime capture, topologies="
                      << exec_graph.decode_graphs().topologies.size() << "\n";
            return 1;
        }
        bool saw_bucket1 = false;
        for (const auto& topology : exec_graph.decode_graphs().topologies) {
            if (topology.bucket_index == 1 && topology.batch_size == 1 &&
                topology.topology_class == flash_next_decode_graph_topology_class(1, 1)) {
                saw_bucket1 = true;
            }
        }
        if (!saw_bucket1) {
            std::cerr << "FAIL: missing explicit bucket_index=1 topology after crossing\n";
            return 1;
        }
        if (exec_graph.last_lazy_capture_milliseconds().has_value()) {
            std::cerr << "FAIL: lazy capture occurred when eager capture was expected\n";
            return 1;
        }
        if (exec_graph.decode_graph_pinned_eager(1, 1)) {
            std::cerr << "FAIL: bucket 1 was pinned to eager unexpectedly\n";
            return 1;
        }

        std::cout << "PASS: test_cuda_graph_bucketed_decode eager startup graphs\n";
        exec_graph.release_lane(lane_g);
        exec_eager.release_lane(lane_e);
        exec_live.release_lane(lane_l);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "test_cuda_graph_bucketed_decode exception: " << e.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "test_cuda_graph_bucketed_decode unknown exception\n";
        return 1;
    }
}

ninfer::targets::qwen3_8_flash_next::detail::FlashNextRuntimeConfig
g8_runtime_config(bool use_cuda_graph) {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    return FlashNextRuntimeConfig{
        .max_concurrency     = 1,
        .max_context         = 65536,
        .state_slot_capacity = 2,
        .prefill_chunk       = 2048,
        .use_cuda_graph      = use_cuda_graph,
    };
}

void g8_prefill_one(ninfer::targets::qwen3_8_flash_next::detail::FlashNextTextExecutor& exec,
                    ninfer::targets::qwen3_8_flash_next::detail::LaneHandle lane, std::int32_t first,
                    std::int32_t T, ninfer::DeviceContext& device, double* ms_out) {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    std::vector<std::int32_t> tokens(static_cast<std::size_t>(T));
    std::vector<std::array<std::int32_t, 3>> positions(static_cast<std::size_t>(T));
    for (std::int32_t t = 0; t < T; ++t) {
        const std::int32_t idx = first + t;
        tokens[static_cast<std::size_t>(t)]    = 100 + (idx & 1023);
        positions[static_cast<std::size_t>(t)] = {idx, idx, idx};
    }
    const auto t0 = std::chrono::steady_clock::now();
    auto pr       = exec.execute_prefill_chunk(lane, tokens, positions, first);
    std::vector<LaneCommitDecision> accept = {{.accept = true}};
    pr.commit(accept);
    device.synchronize();
    const auto t1 = std::chrono::steady_clock::now();
    if (ms_out != nullptr) {
        *ms_out = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
}

int test_g8_admission_cpu() {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    FlashNextRuntimeConfig cfg{
        .max_concurrency     = 2,
        .max_context         = 65536,
        .state_slot_capacity = 4,
        .prefill_chunk       = 2048,
        .use_cuda_graph      = false,
    };
    const auto curve = flash_next_capacity_curve(cfg);
    std::cout << "G8 admission curve min_groups=" << curve.minimum_main_page_groups
              << " max_groups=" << curve.maximum_main_page_groups << "\n";
    if (curve.minimum_main_page_groups != 256 || curve.maximum_main_page_groups != 512) {
        std::cerr << "FAIL: B=2 64k groups expected min=256 max=512\n";
        return 1;
    }
    auto plan = finalize_flash_next_runtime_plan(cfg, curve.minimum_main_page_groups);
    if (plan.main_page_groups != 256) {
        std::cerr << "FAIL: selected groups expected 256 got " << plan.main_page_groups << "\n";
        return 1;
    }
    FlashNextLaneLedger ledger(plan);
    PleIndexMetadata ple_meta{};
    ple_meta.multipliers = {1, 2, 3};
    ple_meta.head_offsets.fill(0);
    ple_meta.head_vocab_sizes.fill(1);
    auto lane0 = ledger.allocate_lane();
    auto lane1 = ledger.allocate_lane();
    std::vector<std::int32_t> full(65536, 100);
    auto prep0 = ledger.begin_prefill_chunk(lane0, full, 0, ple_meta);
    ledger.abort_prefill_chunk(prep0.transaction_id);
    std::cout << "G8 admission: lane0 64k-span reservation, available="
              << ledger.available_physical_groups() << "\n";
    if (ledger.available_physical_groups() != 0) {
        std::cerr << "ANOMALY: expected 0 free groups after a 64k-span lane0 reservation, got "
                  << ledger.available_physical_groups() << "\n";
        return 1;
    }
    try {
        std::vector<std::int32_t> one{100};
        (void)ledger.begin_prefill_chunk(lane1, one, 0, ple_meta);
        std::cerr << "ANOMALY: lane1 64k-pool begin_prefill at t=0 succeeded with 0 free groups\n";
        return 1;
    } catch (const std::runtime_error& ex) {
        std::cout << "G8 admission: lane1 rejected verbatim: " << ex.what() << "\n";
    }
    std::cout << "G8 admission note: inspect_admission eviction is engine-owned (M1); "
                 "ledger at 256 groups/B=1-equivalent pool is exactly one 64k sequence. "
                 "A second live 64k occupant is infeasible without engine pressure eviction.\n";
    std::cout << "PASS: test_g8_admission_cpu\n";
    return 0;
}

int test_g8_decode_crossing(ninfer::DeviceContext& device, std::int32_t cross_tokens) {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    try {
        PleIndexMetadata ple_meta{};
        ple_meta.multipliers = {1, 2, 3};
        ple_meta.head_offsets.fill(0);
        ple_meta.head_vocab_sizes.fill(1);
        auto synthetic_model = make_synthetic_model(device);
        auto make_plan       = [](bool graphs) {
            const auto cfg   = g8_runtime_config(graphs);
            const auto curve = flash_next_capacity_curve(cfg);
            return finalize_flash_next_runtime_plan(cfg, curve.minimum_main_page_groups);
        };
        auto plan_g = make_plan(true);
        auto plan_e = make_plan(false);
        FlashNextRuntimeAllocation alloc_g(plan_g);
        FlashNextRuntimeAllocation alloc_e(plan_e);
        FlashNextRuntimeAllocation alloc_l(plan_e);
        alloc_g.initialize(device.stream);
        alloc_e.initialize(device.stream);
        alloc_l.initialize(device.stream);
        FlashNextTextExecutor exec_g(synthetic_model.view, ple_meta, device, alloc_g);
        FlashNextTextExecutor exec_e(synthetic_model.view, ple_meta, device, alloc_e);
        FlashNextTextExecutor exec_l(synthetic_model.view, ple_meta, device, alloc_l);
        exec_e.set_use_cuda_graph(false);
        exec_l.set_use_cuda_graph(false);
        auto lane_g = exec_g.allocate_lane();
        auto lane_e = exec_e.allocate_lane();
        auto lane_l = exec_l.allocate_lane();
        constexpr std::int32_t kChunk = 2048;
        for (std::int32_t first = 0; first < cross_tokens; first += kChunk) {
            g8_prefill_one(exec_g, lane_g, first, kChunk, device, nullptr);
            g8_prefill_one(exec_e, lane_e, first, kChunk, device, nullptr);
            g8_prefill_one(exec_l, lane_l, first, kChunk, device, nullptr);
        }
        bool saw_lo = false;
        bool saw_hi = false;
        for (std::int32_t step = cross_tokens; step < cross_tokens + 6; ++step) {
            const std::int32_t live_blocks = (step + 1) / 4;
            LaneStepRequest rg{.handle          = lane_g,
                               .token_id        = 300 + step,
                               .token_index     = step,
                               .mrope_positions = {step, step, step},
                               .sampling        = {.temperature = 0.0F, .top_p = 1.0F}};
            LaneStepRequest re = rg;
            re.handle          = lane_e;
            LaneStepRequest rl = rg;
            rl.handle          = lane_l;
            auto round_g       = exec_g.execute_round(std::span(&rg, 1));
            auto round_e       = exec_e.execute_round(std::span(&re, 1));
            auto round_l       = exec_l.execute_round_eager(std::span(&rl, 1), live_blocks);
            if (live_blocks * 4 <= cross_tokens) { saw_lo = true; }
            if (live_blocks * 4 > cross_tokens) { saw_hi = true; }
            if (round_g.sampled_tokens()[0] != round_e.sampled_tokens()[0] ||
                round_g.sampled_tokens()[0] != round_l.sampled_tokens()[0]) {
                std::cerr << "FAIL: G8 three-way token mismatch at token_index=" << step
                          << " live_blocks=" << live_blocks << "\n";
                return 1;
            }
            if (!logits_bit_identical_nonvacuous(device, round_g.logits(), round_e.logits(), 1,
                                                 "g8 graph vs eager") ||
                !logits_bit_identical_nonvacuous(device, round_g.logits(), round_l.logits(), 1,
                                                 "g8 graph vs live")) {
                std::cerr << " at token_index=" << step << "\n";
                return 1;
            }
            std::vector<LaneCommitDecision> accept = {{.accept = true}};
            round_g.commit(accept);
            round_e.commit(accept);
            round_l.commit(accept);
            device.synchronize();
        }
        if (!saw_lo || !saw_hi) {
            std::cerr << "FAIL: G8 decode did not cover both sides of " << cross_tokens << "\n";
            return 1;
        }
        std::cout << "PASS: test_g8_decode_crossing tokens=" << cross_tokens << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "test_g8_decode_crossing exception: " << e.what() << "\n";
        return 1;
    }
}

int test_g8_prefill_soak(ninfer::DeviceContext& device) {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    try {
        PleIndexMetadata ple_meta{};
        ple_meta.multipliers = {1, 2, 3};
        ple_meta.head_offsets.fill(0);
        ple_meta.head_vocab_sizes.fill(1);
        auto synthetic_model = make_synthetic_model(device);
        const auto cfg       = g8_runtime_config(true);
        const auto curve     = flash_next_capacity_curve(cfg);
        auto plan            = finalize_flash_next_runtime_plan(cfg, curve.minimum_main_page_groups);
        const std::size_t ws_cap = flash_next_text_prefill_workspace_capacity_bytes(
            static_cast<std::int32_t>(plan.maximum_blocks), 2048);
        std::cout << "G8 prefill workspace capacity T=2048 N=16384: " << ws_cap << " bytes ("
                  << (static_cast<double>(ws_cap) / 1048576.0) << " MiB)\n";
        if (ws_cap >= 2ULL * 1024ULL * 1024ULL * 1024ULL) {
            std::cerr << "ANOMALY: prefill workspace crosses 2 GiB\n";
            return 1;
        }
        auto run_once = [&](std::vector<double>& times, std::vector<std::uint16_t>& hidden) {
            FlashNextRuntimeAllocation alloc(plan);
            alloc.initialize(device.stream);
            FlashNextTextExecutor exec(synthetic_model.view, ple_meta, device, alloc);
            auto lane              = exec.allocate_lane();
            constexpr int kChunks  = 32;
            constexpr int kChunk   = 2048;
            times.assign(kChunks, 0);
            bool saw_identity = false;
            bool saw_sort     = false;
            for (int c = 0; c < kChunks; ++c) {
                const std::int32_t first    = c * kChunk;
                const std::int32_t complete = (first + kChunk) / 4;
                const char* arm = complete <= 512 ? "identity" : "sort";
                if (complete <= 512) { saw_identity = true; }
                else { saw_sort = true; }
                g8_prefill_one(exec, lane, first, kChunk, device, &times[static_cast<std::size_t>(c)]);
                const std::size_t peak = alloc.workspace().peak_used();
                const std::size_t cap  = alloc.workspace().capacity();
                std::cout << "  chunk " << c << " first=" << first << " complete_blocks=" << complete
                          << " arm=" << arm << " ms=" << times[static_cast<std::size_t>(c)]
                          << " ws_peak=" << peak << " / " << cap << "\n";
                if (peak > cap) {
                    std::cerr << "FAIL: workspace peak exceeds capacity at chunk " << c << "\n";
                    return 1;
                }
            }
            if (!saw_identity || !saw_sort) {
                std::cerr << "FAIL: soak did not observe identity-to-sort transition\n";
                return 1;
            }
            hidden.assign(2560, 0);
            CUDA_CHECK(cudaMemcpy(hidden.data(), alloc.round_tensors().final_hidden.data,
                                  hidden.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost));
            exec.release_lane(lane);
            return 0;
        };
        std::vector<double> times_a, times_b;
        std::vector<std::uint16_t> hid_a, hid_b;
        if (run_once(times_a, hid_a) != 0) { return 1; }
        if (run_once(times_b, hid_b) != 0) { return 1; }
        if (hid_a.size() != hid_b.size() ||
            std::memcmp(hid_a.data(), hid_b.data(), hid_a.size() * sizeof(std::uint16_t)) != 0) {
            std::cerr << "FAIL: two full 64k prefills were not bitwise identical on final_hidden\n";
            return 1;
        }
        double energy = 0.0;
        for (auto v : hid_a) {
            const float f = bf16_to_float(v);
            if (!std::isfinite(f)) {
                std::cerr << "FAIL: non-finite final_hidden\n";
                return 1;
            }
            energy += static_cast<double>(f) * static_cast<double>(f);
        }
        if (!(energy > 0.0)) {
            std::cerr << "FAIL: vacuous final_hidden energy\n";
            return 1;
        }
        std::cout << "G8 prefill soak two-run hidden energy=" << energy << " bitwise match\n";
        std::cout << "PASS: test_g8_prefill_soak\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "test_g8_prefill_soak exception: " << e.what() << "\n";
        return 1;
    }
}

struct BitCompare {
    std::size_t mismatches = 0;
    std::size_t first      = static_cast<std::size_t>(-1);
    std::uint16_t a        = 0;
    std::uint16_t b        = 0;
    double base_sq         = 0.0;
    int nonfinite          = 0;
};

BitCompare compare_bf16_host(const std::vector<std::uint16_t>& a,
                             const std::vector<std::uint16_t>& b) {
    BitCompare out;
    const std::size_t n = std::min(a.size(), b.size());
    if (a.size() != b.size()) { out.mismatches = static_cast<std::size_t>(-1); }
    for (std::size_t i = 0; i < n; ++i) {
        const float fa = bf16_to_float(a[i]);
        const float fb = bf16_to_float(b[i]);
        if (!std::isfinite(fa) || !std::isfinite(fb)) { ++out.nonfinite; }
        out.base_sq += static_cast<double>(fa) * static_cast<double>(fa);
        if (a[i] != b[i]) {
            if (out.mismatches == 0) {
                out.first = i;
                out.a     = a[i];
                out.b     = b[i];
            }
            ++out.mismatches;
        }
    }
    return out;
}

void copy_bf16(ninfer::DeviceContext& device, const ninfer::Tensor& t, std::size_t n,
               std::vector<std::uint16_t>& host) {
    host.assign(n, 0);
    device.synchronize();
    CUDA_CHECK(cudaMemcpy(host.data(), t.data, n * sizeof(std::uint16_t), cudaMemcpyDeviceToHost));
}

void print_bit_compare(const char* label, const BitCompare& c) {
    std::cout << "  " << label << " mismatches=" << c.mismatches << " nonfinite=" << c.nonfinite
              << " base_sq=" << c.base_sq;
    if (c.mismatches > 0 && c.mismatches != static_cast<std::size_t>(-1)) {
        std::cout << " first[" << c.first << "] a=0x" << std::hex << c.a << " b=0x" << c.b
                  << std::dec;
    }
    std::cout << "\n";
}

const char* graph_node_type_name(cudaGraphNodeType t) {
    switch (t) {
    case cudaGraphNodeTypeKernel:
        return "Kernel";
    case cudaGraphNodeTypeMemcpy:
        return "Memcpy";
    case cudaGraphNodeTypeMemset:
        return "Memset";
    case cudaGraphNodeTypeHost:
        return "Host";
    case cudaGraphNodeTypeGraph:
        return "Graph";
    case cudaGraphNodeTypeEmpty:
        return "Empty";
    case cudaGraphNodeTypeWaitEvent:
        return "WaitEvent";
    case cudaGraphNodeTypeEventRecord:
        return "EventRecord";
    default:
        return "Other";
    }
}

struct KernelSig {
    const void* func = nullptr;
    unsigned gx = 0, gy = 0, gz = 0;
    unsigned bx = 0, by = 0, bz = 0;
    unsigned shared = 0;
};

void dump_cuda_graph(const char* label, cudaGraph_t graph, std::vector<KernelSig>& kernels) {
    kernels.clear();
    if (graph == nullptr) {
        std::cout << "G8 graph " << label << " native=null\n";
        return;
    }
    std::size_t n = 0;
    CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &n));
    std::vector<cudaGraphNode_t> nodes(n);
    if (n > 0) { CUDA_CHECK(cudaGraphGetNodes(graph, nodes.data(), &n)); }
    int type_hist[16]{};
    int memcpy_nodes = 0;
    std::cout << "G8 graph " << label << " nodes=" << n << "\n";
    for (std::size_t i = 0; i < n; ++i) {
        cudaGraphNodeType t{};
        CUDA_CHECK(cudaGraphNodeGetType(nodes[i], &t));
        const int ti = static_cast<int>(t);
        if (ti >= 0 && ti < 16) { ++type_hist[ti]; }
        if (t == cudaGraphNodeTypeKernel) {
            cudaKernelNodeParams p{};
            CUDA_CHECK(cudaGraphKernelNodeGetParams(nodes[i], &p));
            KernelSig s;
            s.func   = p.func;
            s.gx     = p.gridDim.x;
            s.gy     = p.gridDim.y;
            s.gz     = p.gridDim.z;
            s.bx     = p.blockDim.x;
            s.by     = p.blockDim.y;
            s.bz     = p.blockDim.z;
            s.shared = p.sharedMemBytes;
            kernels.push_back(s);
        } else if (t == cudaGraphNodeTypeMemcpy) {
            ++memcpy_nodes;
            cudaMemcpy3DParms mp{};
            CUDA_CHECK(cudaGraphMemcpyNodeGetParams(nodes[i], &mp));
            const std::size_t bytes = static_cast<std::size_t>(mp.extent.width) *
                                      static_cast<std::size_t>(mp.extent.height) *
                                      static_cast<std::size_t>(mp.extent.depth);
            std::cout << "    memcpy[" << memcpy_nodes - 1 << "] bytes=" << bytes
                      << " kind=" << static_cast<int>(mp.kind) << "\n";
        }
    }
    for (int ti = 0; ti < 16; ++ti) {
        if (type_hist[ti] > 0) {
            std::cout << "    type " << graph_node_type_name(static_cast<cudaGraphNodeType>(ti))
                      << " (" << ti << ") count=" << type_hist[ti] << "\n";
        }
    }
    std::cout << "    kernel_count=" << kernels.size() << "\n";
    unsigned max_gx = 0;
    int wide = 0;
    for (std::size_t i = 0; i < kernels.size(); ++i) {
        const auto& k = kernels[i];
        if (k.gx > max_gx) { max_gx = k.gx; }
        if (k.gx >= 512) { ++wide; }
    }
    std::cout << "    kernels_with_grid.x>=512=" << wide << " max_grid.x=" << max_gx
              << " (identity-bucket must not launch 16384-wide score)\n";
}

int diff_kernel_sigs(const char* a_label, const std::vector<KernelSig>& a, const char* b_label,
                     const std::vector<KernelSig>& b) {
    std::cout << "G8 kernel-sig diff " << a_label << " (" << a.size() << ") vs " << b_label << " ("
              << b.size() << ")\n";
    if (a.size() != b.size()) {
        std::cout << "  COUNT MISMATCH\n";
        return 1;
    }
    int diffs = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const bool same = a[i].func == b[i].func && a[i].gx == b[i].gx && a[i].gy == b[i].gy &&
                          a[i].gz == b[i].gz && a[i].bx == b[i].bx && a[i].by == b[i].by &&
                          a[i].bz == b[i].bz && a[i].shared == b[i].shared;
        if (!same) {
            ++diffs;
            std::cout << "  kernel[" << i << "] " << a_label << " func=" << a[i].func
                      << " grid=" << a[i].gx << "," << a[i].gy << "," << a[i].gz
                      << " block=" << a[i].bx << "," << a[i].by << "," << a[i].bz
                      << " shared=" << a[i].shared << "\n";
            std::cout << "  kernel[" << i << "] " << b_label << " func=" << b[i].func
                      << " grid=" << b[i].gx << "," << b[i].gy << "," << b[i].gz
                      << " block=" << b[i].bx << "," << b[i].by << "," << b[i].bz
                      << " shared=" << b[i].shared << "\n";
        }
    }
    std::cout << "  signature_diffs=" << diffs << "\n";
    return diffs;
}

cudaGraph_t bucket0_native(const ninfer::targets::qwen3_8_flash_next::detail::DecodeGraphFamily& family) {
    for (const auto& profile : family.profiles) {
        if (profile.batch_size == 1 && profile.bucket_index == 0 && profile.definition.ready()) {
            return profile.definition.native();
        }
    }
    return nullptr;
}

int test_g8_graph_node_diff(ninfer::DeviceContext& device) {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    try {
        PleIndexMetadata ple_meta{};
        ple_meta.multipliers = {1, 2, 3};
        ple_meta.head_offsets.fill(0);
        ple_meta.head_vocab_sizes.fill(1);
        auto synthetic_model = make_synthetic_model(device);

        auto make_exec = [&](std::int32_t max_context, std::uint32_t concurrency) {
            FlashNextRuntimeConfig cfg{
                .max_concurrency     = concurrency,
                .max_context         = static_cast<std::uint32_t>(max_context),
                .state_slot_capacity = 2 * concurrency,
                .prefill_chunk       = 2048,
                .use_cuda_graph      = true,
            };
            const auto curve = flash_next_capacity_curve(cfg);
            auto plan        = finalize_flash_next_runtime_plan(cfg, curve.minimum_main_page_groups);
            auto alloc       = std::make_unique<FlashNextRuntimeAllocation>(plan);
            alloc->initialize(device.stream);
            auto exec = std::make_unique<FlashNextTextExecutor>(synthetic_model.view, ple_meta,
                                                                device, *alloc);
            return std::make_pair(std::move(alloc), std::move(exec));
        };

        // 1. Verify B=1 graph node signature diffs between 8192 and 65536
        auto [alloc_8k, exec_8k]   = make_exec(8192, 1);
        std::vector<KernelSig> k8k;
        dump_cuda_graph("max_context=8192 bucket0 (B=1)", bucket0_native(exec_8k->decode_graphs()), k8k);
        exec_8k.reset();
        alloc_8k.reset();

        auto [alloc_64k, exec_64k] = make_exec(65536, 1);
        std::vector<KernelSig> k64k;
        dump_cuda_graph("max_context=65536 bucket0 (B=1)", bucket0_native(exec_64k->decode_graphs()), k64k);
        exec_64k.reset();
        alloc_64k.reset();

        const int diffs_b1 = diff_kernel_sigs("8192_B1", k8k, "65536_B1", k64k);
        std::cout << (diffs_b1 == 0 ? "PASS" : "FAIL")
                  << ": test_g8_graph_node_diff B=1 signature_diffs=" << diffs_b1 << "\n";
        if (diffs_b1 != 0) {
            return 1;
        }

        // 2. Verify B=8 graph node signature diffs between 8192 and 65536
        auto [alloc_8k_b8, exec_8k_b8] = make_exec(8192, 8);
        std::vector<KernelSig> k8k_b8;
        dump_cuda_graph("max_context=8192 bucket0 (B=8)", bucket0_native(exec_8k_b8->decode_graphs()), k8k_b8);
        exec_8k_b8.reset();
        alloc_8k_b8.reset();

        auto [alloc_64k_b8, exec_64k_b8] = make_exec(65536, 8);
        std::vector<KernelSig> k64k_b8;
        dump_cuda_graph("max_context=65536 bucket0 (B=8)", bucket0_native(exec_64k_b8->decode_graphs()), k64k_b8);
        exec_64k_b8.reset();
        alloc_64k_b8.reset();

        const int diffs_b8 = diff_kernel_sigs("8192_B8", k8k_b8, "65536_B8", k64k_b8);
        std::cout << (diffs_b8 == 0 ? "PASS" : "FAIL")
                  << ": test_g8_graph_node_diff B=8 signature_diffs=" << diffs_b8 << "\n";
        if (diffs_b8 != 0) {
            return 1;
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "test_g8_graph_node_diff exception: " << e.what() << "\n";
        return 1;
    }
}

int test_g8_decode_characterization(ninfer::DeviceContext& device) {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    try {
        PleIndexMetadata ple_meta{};
        ple_meta.multipliers = {1, 2, 3};
        ple_meta.head_offsets.fill(0);
        ple_meta.head_vocab_sizes.fill(1);
        auto synthetic_model = make_synthetic_model(device);
        auto make_plan       = [](bool graphs) {
            const auto cfg   = g8_runtime_config(graphs);
            const auto curve = flash_next_capacity_curve(cfg);
            return finalize_flash_next_runtime_plan(cfg, curve.minimum_main_page_groups);
        };
        auto plan_g = make_plan(true);
        auto plan_e = make_plan(false);
        FlashNextRuntimeAllocation alloc_g(plan_g);
        FlashNextRuntimeAllocation alloc_e(plan_e);
        FlashNextRuntimeAllocation alloc_l(plan_e);
        alloc_g.initialize(device.stream);
        alloc_e.initialize(device.stream);
        alloc_l.initialize(device.stream);
        FlashNextTextExecutor exec_g(synthetic_model.view, ple_meta, device, alloc_g);
        FlashNextTextExecutor exec_e(synthetic_model.view, ple_meta, device, alloc_e);
        FlashNextTextExecutor exec_l(synthetic_model.view, ple_meta, device, alloc_l);
        exec_e.set_use_cuda_graph(false);
        exec_l.set_use_cuda_graph(false);
        auto lane_g = exec_g.allocate_lane();
        auto lane_e = exec_e.allocate_lane();
        auto lane_l = exec_l.allocate_lane();
        g8_prefill_one(exec_g, lane_g, 0, 2048, device, nullptr);
        g8_prefill_one(exec_e, lane_e, 0, 2048, device, nullptr);
        g8_prefill_one(exec_l, lane_l, 0, 2048, device, nullptr);

        std::vector<std::uint16_t> pre_g, pre_e, pre_l;
        copy_bf16(device, alloc_g.round_tensors().final_hidden, 2560, pre_g);
        copy_bf16(device, alloc_e.round_tensors().final_hidden, 2560, pre_e);
        copy_bf16(device, alloc_l.round_tensors().final_hidden, 2560, pre_l);
        const auto pre_ge = compare_bf16_host(pre_g, pre_e);
        const auto pre_el = compare_bf16_host(pre_e, pre_l);
        print_bit_compare("prefill-only graph vs eager-bucket hidden", pre_ge);
        print_bit_compare("prefill-only eager-bucket vs eager-live hidden", pre_el);
        if (pre_ge.mismatches > 0 || pre_el.mismatches > 0) {
            std::cout << "  diagnosis: independent identity prefills already diverge — decode "
                         "three-way is not comparing equal KV.\n";
        } else {
            std::cout << "  diagnosis: identity prefills bitwise match; decode-path divergence "
                         "is not inherited from prefill KV.\n";
        }

        constexpr int kRepeats            = 5;
        constexpr std::int32_t kStep      = 2048;
        constexpr std::int32_t kLiveBlocks = (kStep + 1) / 4;
        std::cout << "G8 char at token_index=" << kStep << " live_blocks=" << kLiveBlocks
                  << " (identity bucket under max_context=65536)\n";

        auto one_round = [&](FlashNextTextExecutor& exec, LaneHandle lane, bool live,
                             std::vector<std::uint16_t>& logits, std::vector<std::uint16_t>& hidden,
                             std::int32_t& token) {
            LaneStepRequest req{.handle          = lane,
                                .token_id        = 300 + kStep,
                                .token_index     = kStep,
                                .mrope_positions = {kStep, kStep, kStep},
                                .sampling        = {.temperature = 0.0F, .top_p = 1.0F}};
            auto round = live ? exec.execute_round_eager(std::span(&req, 1), kLiveBlocks)
                              : exec.execute_round(std::span(&req, 1));
            token      = round.sampled_tokens()[0];
            copy_bf16(device, round.logits(), 248320, logits);
            copy_bf16(device, round.final_hidden(), 2560, hidden);
            // Abort (PendingRound destructor) rolls back the ledger only. Decode wrote the
            // destination slot; the next repeat still reads the prefill source slot.
        };

        std::vector<std::vector<std::uint16_t>> g_logits(kRepeats), e_logits(kRepeats),
            l_logits(kRepeats);
        std::vector<std::vector<std::uint16_t>> g_hidden(kRepeats), e_hidden(kRepeats),
            l_hidden(kRepeats);
        std::array<std::int32_t, kRepeats> g_tok{}, e_tok{}, l_tok{};
        for (int r = 0; r < kRepeats; ++r) {
            one_round(exec_g, lane_g, false, g_logits[static_cast<std::size_t>(r)],
                      g_hidden[static_cast<std::size_t>(r)], g_tok[static_cast<std::size_t>(r)]);
            one_round(exec_e, lane_e, false, e_logits[static_cast<std::size_t>(r)],
                      e_hidden[static_cast<std::size_t>(r)], e_tok[static_cast<std::size_t>(r)]);
            one_round(exec_l, lane_l, true, l_logits[static_cast<std::size_t>(r)],
                      l_hidden[static_cast<std::size_t>(r)], l_tok[static_cast<std::size_t>(r)]);
        }

        auto stability = [](const std::vector<std::vector<std::uint16_t>>& runs, const char* label) {
            bool ok = true;
            for (int r = 1; r < static_cast<int>(runs.size()); ++r) {
                const auto c = compare_bf16_host(runs[0], runs[static_cast<std::size_t>(r)]);
                std::cout << "  " << label << " run0 vs run" << r;
                print_bit_compare("", c);
                if (c.mismatches != 0 || c.nonfinite != 0 || !(c.base_sq > 0.0)) { ok = false; }
            }
            return ok;
        };

        const bool g_stable = stability(g_logits, "graph logits");
        const bool e_stable = stability(e_logits, "eager-bucket logits");
        const bool l_stable = stability(l_logits, "eager-live logits");
        (void)stability(g_hidden, "graph hidden");
        (void)stability(e_hidden, "eager-bucket hidden");
        (void)stability(l_hidden, "eager-live hidden");

        const auto ge = compare_bf16_host(g_logits[0], e_logits[0]);
        const auto gl = compare_bf16_host(g_logits[0], l_logits[0]);
        const auto el = compare_bf16_host(e_logits[0], l_logits[0]);
        const auto geh = compare_bf16_host(g_hidden[0], e_hidden[0]);
        const auto elh = compare_bf16_host(e_hidden[0], l_hidden[0]);
        print_bit_compare("graph vs eager-bucket logits", ge);
        print_bit_compare("graph vs eager-live logits", gl);
        print_bit_compare("eager-bucket vs eager-live logits", el);
        print_bit_compare("graph vs eager-bucket hidden", geh);
        print_bit_compare("eager-bucket vs eager-live hidden", elh);
        std::cout << "  greedy tokens graph=" << g_tok[0] << " eager=" << e_tok[0]
                  << " live=" << l_tok[0] << "\n";
        bool tok_match = true;
        for (int r = 0; r < kRepeats; ++r) {
            if (g_tok[static_cast<std::size_t>(r)] != g_tok[0] ||
                e_tok[static_cast<std::size_t>(r)] != e_tok[0] ||
                l_tok[static_cast<std::size_t>(r)] != l_tok[0] || g_tok[0] != e_tok[0] ||
                g_tok[0] != l_tok[0]) {
                tok_match = false;
            }
        }
        std::cout << "  graph_logits_stable=" << (g_stable ? "yes" : "NO")
                  << " eager_stable=" << (e_stable ? "yes" : "NO")
                  << " live_stable=" << (l_stable ? "yes" : "NO")
                  << " greedy_match=" << (tok_match ? "yes" : "NO") << "\n";
        if (ge.mismatches > 0 && g_stable && e_stable) {
            std::cout << "  diagnosis: graph leg internally stable, eager leg internally stable, "
                         "same first-mismatch bits across the pair — frozen capture-vs-eager "
                         "ordering, not a race.\n";
        }
        if (el.mismatches == 0 && ge.mismatches > 0) {
            std::cout << "  diagnosis: eager-bucket == eager-live bitwise; graph is the odd leg.\n";
        }
        if (!(ge.base_sq > 0.0) || ge.nonfinite != 0) {
            std::cerr << "FAIL: characterization comparison vacuous or non-finite\n";
            return 1;
        }
        std::cout << "PASS: test_g8_decode_characterization (report; 1-ULP not papered over)\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "test_g8_decode_characterization exception: " << e.what() << "\n";
        return 1;
    }
}

std::uint64_t fnv1a_64(const void* data, std::size_t n) {
    auto h              = 14695981039346656037ULL;
    const auto* bytes   = static_cast<const std::uint8_t*>(data);
    for (std::size_t i = 0; i < n; ++i) {
        h ^= bytes[i];
        h *= 1099511628211ULL;
    }
    return h;
}

struct StageChecksum {
    std::string name;
    std::uint64_t hash = 0;
    std::size_t bytes  = 0;
    double energy      = 0.0;
    int nonzero        = 0;
};

int test_g8_stage_checksum(ninfer::DeviceContext& device) {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    try {
        PleIndexMetadata ple_meta{};
        ple_meta.multipliers = {1, 2, 3};
        ple_meta.head_offsets.fill(0);
        ple_meta.head_vocab_sizes.fill(1);
        auto synthetic_model = make_synthetic_model(device);
        const auto cfg       = g8_runtime_config(false);
        const auto curve     = flash_next_capacity_curve(cfg);
        auto plan            = finalize_flash_next_runtime_plan(cfg, curve.minimum_main_page_groups);

        auto run_one = [&](std::vector<StageChecksum>& stages) {
            stages.clear();
            FlashNextRuntimeAllocation alloc(plan);
            alloc.initialize(device.stream);
            FlashNextTextExecutor exec(synthetic_model.view, ple_meta, device, alloc);
            exec.set_use_cuda_graph(false);
            auto lane = exec.allocate_lane();
            std::vector<std::byte> host(64ULL * 1024ULL * 1024ULL);
            FlashNextDecodeStateSink sink;
            sink.on_state = [&](std::string_view name, const ninfer::Tensor& tensor) {
                const std::int64_t n0 = tensor.ne[0] > 0 ? tensor.ne[0] : 1;
                const std::int64_t n1 = tensor.ne[1] > 0 ? tensor.ne[1] : 1;
                const std::int64_t n2 = tensor.ne[2] > 0 ? tensor.ne[2] : 1;
                const std::int64_t n3 = tensor.ne[3] > 0 ? tensor.ne[3] : 1;
                const std::size_t elems =
                    static_cast<std::size_t>(n0) * static_cast<std::size_t>(n1) *
                    static_cast<std::size_t>(n2) * static_cast<std::size_t>(n3);
                const std::size_t bytes = elems * ninfer::dtype_size(tensor.dtype);
                if (bytes > host.size()) { host.resize(bytes); }
                CUDA_CHECK(cudaMemcpy(host.data(), tensor.data, bytes, cudaMemcpyDeviceToHost));
                StageChecksum s;
                s.name  = std::string(name);
                s.hash  = fnv1a_64(host.data(), bytes);
                s.bytes = bytes;
                if (tensor.dtype == ninfer::DType::BF16) {
                    const auto* v = reinterpret_cast<const std::uint16_t*>(host.data());
                    for (std::size_t i = 0; i < elems; ++i) {
                        const float f = bf16_to_float(v[i]);
                        if (v[i] != 0) { ++s.nonzero; }
                        s.energy += static_cast<double>(f) * static_cast<double>(f);
                    }
                } else {
                    const auto* b = host.data();
                    for (std::size_t i = 0; i < bytes; ++i) {
                        if (static_cast<unsigned char>(b[i]) != 0) { ++s.nonzero; }
                    }
                }
                stages.push_back(std::move(s));
            };
            constexpr std::int32_t T = 2048;
            std::vector<std::int32_t> tokens(static_cast<std::size_t>(T));
            std::vector<std::array<std::int32_t, 3>> positions(static_cast<std::size_t>(T));
            for (std::int32_t t = 0; t < T; ++t) {
                tokens[static_cast<std::size_t>(t)]    = 100 + (t & 1023);
                positions[static_cast<std::size_t>(t)] = {t, t, t};
            }
            auto pr = exec.execute_prefill_chunk(lane, tokens, positions, 0, &sink);
            std::vector<LaneCommitDecision> accept = {{.accept = true}};
            pr.commit(accept);
            device.synchronize();
            exec.release_lane(lane);
        };

        std::vector<StageChecksum> a, b;
        run_one(a);
        run_one(b);
        std::cout << "G8 stage checksum chunk0 T=2048 identity eager-only stages=" << a.size()
                  << " vs " << b.size() << "\n";
        if (a.size() != b.size() || a.empty()) {
            std::cerr << "FAIL: stage count mismatch or empty\n";
            return 1;
        }
        int first = -1;
        int diffs = 0;
        for (std::size_t i = 0; i < a.size(); ++i) {
            const bool same = a[i].name == b[i].name && a[i].hash == b[i].hash &&
                              a[i].bytes == b[i].bytes;
            if (!same) {
                if (first < 0) { first = static_cast<int>(i); }
                ++diffs;
                if (diffs <= 12) {
                    std::cout << "  DIFF[" << i << "] " << a[i].name << " hash_a=0x" << std::hex
                              << a[i].hash << " hash_b=0x" << b[i].hash << std::dec
                              << " bytes=" << a[i].bytes << " energy_a=" << a[i].energy
                              << " energy_b=" << b[i].energy << " nonzero_a=" << a[i].nonzero
                              << " nonzero_b=" << b[i].nonzero << "\n";
                }
            }
        }
        if (first < 0) {
            std::cout << "  all " << a.size() << " stages bitwise identical (eager-only chunk 0)\n";
            if (!(a.back().energy > 0.0) && a.back().nonzero == 0) {
                std::cerr << "FAIL: vacuous stage checksums\n";
                return 1;
            }
        } else {
            if (diffs > 12) {
                std::cout << "  ... " << (diffs - 12) << " further diffs omitted\n";
            }
            std::cout << "  first_diff_index=" << first << " name=" << a[static_cast<std::size_t>(first)].name
                      << " diffs=" << diffs << " / " << a.size() << "\n";
            std::cout << "  preceding match:";
            for (int i = 0; i < first; ++i) {
                std::cout << " " << a[static_cast<std::size_t>(i)].name;
            }
            std::cout << "\n";
        }
        std::cout << "PASS: test_g8_stage_checksum (report)\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "test_g8_stage_checksum exception: " << e.what() << "\n";
        return 1;
    }
}

int test_g9_prefill_determinism(ninfer::DeviceContext& device) {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    try {
        PleIndexMetadata ple_meta{};
        ple_meta.multipliers = {1, 2, 3};
        ple_meta.head_offsets.fill(0);
        ple_meta.head_vocab_sizes.fill(1);
        auto synthetic_model = make_synthetic_model(device);
        FlashNextRuntimeConfig cfg{
            .max_concurrency     = 1,
            .max_context         = 8192,
            .state_slot_capacity = 2,
            .prefill_chunk       = 2048,
            .use_cuda_graph      = true,
        };
        const auto curve = flash_next_capacity_curve(cfg);
        auto plan        = finalize_flash_next_runtime_plan(cfg, curve.minimum_main_page_groups);
        auto run_once    = [&](std::vector<std::uint16_t>& hidden) {
            FlashNextRuntimeAllocation alloc(plan);
            alloc.initialize(device.stream);
            FlashNextTextExecutor exec(synthetic_model.view, ple_meta, device, alloc);
            auto lane             = exec.allocate_lane();
            constexpr int kChunks = 3;
            constexpr int kChunk  = 2048;
            for (int c = 0; c < kChunks; ++c) {
                g8_prefill_one(exec, lane, c * kChunk, kChunk, device, nullptr);
            }
            hidden.assign(2560, 0);
            CUDA_CHECK(cudaMemcpy(hidden.data(), alloc.round_tensors().final_hidden.data,
                                  hidden.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost));
            exec.release_lane(lane);
        };
        std::vector<std::uint16_t> hid_a, hid_b;
        run_once(hid_a);
        run_once(hid_b);
        if (hid_a.size() != hid_b.size() ||
            std::memcmp(hid_a.data(), hid_b.data(), hid_a.size() * sizeof(std::uint16_t)) != 0) {
            std::cerr << "FAIL: G9 two-run 3x2048 prefill was not bitwise identical on final_hidden\n";
            std::size_t first = static_cast<std::size_t>(-1);
            for (std::size_t i = 0; i < hid_a.size(); ++i) {
                if (hid_a[i] != hid_b[i]) {
                    first = i;
                    break;
                }
            }
            std::cerr << "  first mismatch idx=" << first << " a=0x" << std::hex
                      << (first < hid_a.size() ? hid_a[first] : 0) << " b=0x"
                      << (first < hid_b.size() ? hid_b[first] : 0) << std::dec << "\n";
            return 1;
        }
        double energy = 0.0;
        int nonfinite = 0;
        for (auto v : hid_a) {
            const float f = bf16_to_float(v);
            if (!std::isfinite(f)) { ++nonfinite; }
            energy += static_cast<double>(f) * static_cast<double>(f);
        }
        if (nonfinite != 0 || !(energy > 0.0) || !std::isfinite(energy)) {
            std::cerr << "FAIL: G9 two-run gate vacuous or non-finite energy=" << energy
                      << " nonfinite=" << nonfinite << "\n";
            return 1;
        }
        std::cout << "PASS: test_g9_prefill_determinism 3x2048 two-run hidden energy=" << energy
                  << " bitwise match\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "test_g9_prefill_determinism exception: " << e.what() << "\n";
        return 1;
    }
}

ninfer::targets::qwen3_8_flash_next::detail::FlashNextRuntimeConfig
g14_runtime_config(bool use_cuda_graph) {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    return FlashNextRuntimeConfig{
        .max_concurrency     = 1,
        .max_context         = 262144,
        .state_slot_capacity = 2,
        .prefill_chunk       = 2048,
        .use_cuda_graph      = use_cuda_graph,
    };
}

int test_g14_admission_cpu() {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    FlashNextRuntimeConfig cfg{
        .max_concurrency     = 2,
        .max_context         = 262144,
        .state_slot_capacity = 4,
        .prefill_chunk       = 2048,
        .use_cuda_graph      = false,
    };
    const auto curve = flash_next_capacity_curve(cfg);
    std::cout << "G14 admission curve min_groups=" << curve.minimum_main_page_groups
              << " max_groups=" << curve.maximum_main_page_groups << "\n";
    if (curve.minimum_main_page_groups != 1024 || curve.maximum_main_page_groups != 2048) {
        std::cerr << "FAIL: B=2 256k groups expected min=1024 max=2048\n";
        return 1;
    }
    auto plan = finalize_flash_next_runtime_plan(cfg, curve.minimum_main_page_groups);
    if (plan.main_page_groups != 1024) {
        std::cerr << "FAIL: selected groups expected 1024 got " << plan.main_page_groups << "\n";
        return 1;
    }
    FlashNextLaneLedger ledger(plan);
    PleIndexMetadata ple_meta{};
    ple_meta.multipliers = {1, 2, 3};
    ple_meta.head_offsets.fill(0);
    ple_meta.head_vocab_sizes.fill(1);
    auto lane0 = ledger.allocate_lane();
    auto lane1 = ledger.allocate_lane();
    std::vector<std::int32_t> full(262144, 100);
    auto prep0 = ledger.begin_prefill_chunk(lane0, full, 0, ple_meta);
    ledger.abort_prefill_chunk(prep0.transaction_id);
    std::cout << "G14 admission: lane0 256k-span reservation, available="
              << ledger.available_physical_groups() << "\n";
    if (ledger.available_physical_groups() != 0) {
        std::cerr << "ANOMALY: expected 0 free groups after a 256k-span lane0 reservation, got "
                  << ledger.available_physical_groups() << "\n";
        return 1;
    }
    try {
        std::vector<std::int32_t> one{100};
        (void)ledger.begin_prefill_chunk(lane1, one, 0, ple_meta);
        std::cerr << "ANOMALY: lane1 256k-pool begin_prefill at t=0 succeeded with 0 free groups\n";
        return 1;
    } catch (const std::runtime_error& ex) {
        std::cout << "G14 admission: lane1 rejected verbatim: " << ex.what() << "\n";
    }
    std::cout << "G14 admission note: ledger at 1024 groups/B=1-equivalent pool is exactly one "
                 "256k sequence. A second live 256k occupant is infeasible without engine "
                 "pressure eviction.\n";
    std::cout << "PASS: test_g14_admission_cpu\n";
    return 0;
}

void g14_time_decode_rounds(ninfer::targets::qwen3_8_flash_next::detail::FlashNextTextExecutor& exec,
                            ninfer::targets::qwen3_8_flash_next::detail::LaneHandle lane,
                            const ninfer::targets::qwen3_8_flash_next::detail::FlashNextRuntimePlan&
                                plan,
                            std::int32_t token_index, ninfer::DeviceContext& device, const char* label) {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    const std::int32_t live_blocks = (token_index + 1) / 4;
    const auto buckets             = flash_next_decode_graph_buckets(plan.maximum_blocks);
    const auto bucket              = flash_next_decode_graph_select_bucket(buckets, live_blocks);
    const char* arm                = live_blocks <= 512 ? "identity" : "topk";
    auto make_req                  = [&]() {
        return LaneStepRequest{.handle          = lane,
                               .token_id        = 300 + token_index,
                               .token_index     = token_index,
                               .mrope_positions = {token_index, token_index, token_index},
                               .sampling        = {.temperature = 0.0F, .top_p = 1.0F}};
    };
    {
        auto req   = make_req();
        auto round = exec.execute_round(std::span(&req, 1));
        (void)round.sampled_tokens()[0];
        if (!logits_bit_identical_nonvacuous(device, round.logits(), round.logits(), 1,
                                             "g14 decode self-energy")) {
            throw std::runtime_error("g14 decode logits vacuous or non-finite");
        }
    }
    device.synchronize();
    cudaEvent_t ev0 = nullptr;
    cudaEvent_t ev1 = nullptr;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));
    constexpr int kRepeats = 5;
    double sum_wall        = 0.0;
    double sum_gpu         = 0.0;
    double min_gpu         = 1e300;
    for (int i = 0; i < kRepeats; ++i) {
        auto req = make_req();
        const auto t0 = std::chrono::steady_clock::now();
        CUDA_CHECK(cudaEventRecord(ev0, device.stream));
        auto round = exec.execute_round(std::span(&req, 1));
        (void)round.sampled_tokens()[0];
        CUDA_CHECK(cudaEventRecord(ev1, device.stream));
        CUDA_CHECK(cudaEventSynchronize(ev1));
        float gpu_ms = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, ev0, ev1));
        const double wall_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        sum_wall += wall_ms;
        sum_gpu += static_cast<double>(gpu_ms);
        min_gpu = std::min(min_gpu, static_cast<double>(gpu_ms));
        std::cout << "  decode[" << i << "] " << label << " token_index=" << token_index
                  << " live_blocks=" << live_blocks << " bucket=" << bucket
                  << " envelope=" << buckets.blocks[bucket] << " arm=" << arm
                  << " wall_ms=" << wall_ms << " gpu_ms=" << gpu_ms << "\n";
    }
    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));
    std::cout << "G14 DECODE " << label << " mean_wall_ms=" << (sum_wall / kRepeats)
              << " mean_gpu_ms=" << (sum_gpu / kRepeats) << " min_gpu_ms=" << min_gpu
              << " live_blocks=" << live_blocks << " bucket=" << bucket
              << " envelope=" << buckets.blocks[bucket] << " arm=" << arm << "\n";
}

int test_g14_prefill_and_decode(ninfer::DeviceContext& device, int chunks) {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    try {
        if (chunks < 1 || chunks > 128) {
            std::cerr << "FAIL: g14 chunks must be in [1,128]\n";
            return 1;
        }
        PleIndexMetadata ple_meta{};
        ple_meta.multipliers = {1, 2, 3};
        ple_meta.head_offsets.fill(0);
        ple_meta.head_vocab_sizes.fill(1);
        auto synthetic_model = make_synthetic_model(device);
        const auto cfg       = g14_runtime_config(true);
        const auto curve     = flash_next_capacity_curve(cfg);
        auto plan            = finalize_flash_next_runtime_plan(cfg, curve.minimum_main_page_groups);
        const std::int32_t tile =
            flash_next_qsa_indexer_tile_size(static_cast<std::int32_t>(plan.maximum_blocks), 2048);
        const std::size_t ws_cap = flash_next_text_prefill_workspace_capacity_bytes(
            static_cast<std::int32_t>(plan.maximum_blocks), 2048);
        std::cout << "G14 prefill plan groups=" << plan.main_page_groups
                  << " maximum_blocks=" << plan.maximum_blocks << " tile=" << tile
                  << " tiles_per_chunk=" << (2048 + tile - 1) / tile << " ws_cap=" << ws_cap
                  << " (" << (static_cast<double>(ws_cap) / 1048576.0) << " MiB)\n";
        FlashNextRuntimeAllocation alloc(plan);
        alloc.initialize(device.stream);
        FlashNextTextExecutor exec(synthetic_model.view, ple_meta, device, alloc);
        auto lane                 = exec.allocate_lane();
        constexpr int kChunk      = 2048;
        const auto soak_t0        = std::chrono::steady_clock::now();
        auto is_report            = [](int c) {
            return c == 0 || c == 31 || c == 63 || c == 95 || c == 127;
        };
        auto is_decode_depth = [](int c) { return c == 0 || c == 31 || c == 126; };
        for (int c = 0; c < chunks; ++c) {
            const std::int32_t first    = c * kChunk;
            const std::int32_t complete = (first + kChunk) / 4;
            const char* arm             = complete <= 512 ? "identity" : "sort";
            double ms                   = 0.0;
            g8_prefill_one(exec, lane, first, kChunk, device, &ms);
            const std::size_t peak = alloc.workspace().peak_used();
            const std::size_t cap  = alloc.workspace().capacity();
            const double elapsed =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - soak_t0).count();
            if (is_report(c) || c + 1 == chunks) {
                std::cout << "  chunk " << c << " first=" << first << " complete_blocks=" << complete
                          << " arm=" << arm << " ms=" << ms << " ws_peak=" << peak << " / " << cap
                          << " elapsed_s=" << elapsed << "\n";
            }
            if (peak > cap) {
                std::cerr << "FAIL: workspace peak exceeds capacity at chunk " << c << "\n";
                return 1;
            }
            if (is_decode_depth(c) && first + kChunk < static_cast<std::int32_t>(plan.config.max_context)) {
                char label[64];
                std::snprintf(label, sizeof(label), "after-chunk-%d", c);
                g14_time_decode_rounds(exec, lane, plan, first + kChunk, device, label);
            }
            if (elapsed > 180.0) {
                std::cout << "G14 STOP before watchdog: elapsed_s=" << elapsed << " last_chunk=" << c
                          << "\n";
                break;
            }
        }
        std::vector<std::uint16_t> hidden(2560, 0);
        CUDA_CHECK(cudaMemcpy(hidden.data(), alloc.round_tensors().final_hidden.data,
                              hidden.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost));
        double energy = 0.0;
        int nonfinite = 0;
        for (auto v : hidden) {
            const float f = bf16_to_float(v);
            if (!std::isfinite(f)) { ++nonfinite; }
            energy += static_cast<double>(f) * static_cast<double>(f);
        }
        if (nonfinite != 0 || !(energy > 0.0) || !std::isfinite(energy)) {
            std::cerr << "FAIL: g14 prefill hidden vacuous energy=" << energy
                      << " nonfinite=" << nonfinite << "\n";
            return 1;
        }
        std::cout << "G14 prefill hidden energy=" << energy << "\n";
        exec.release_lane(lane);
        std::cout << "PASS: test_g14_prefill_and_decode chunks=" << chunks << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "test_g14_prefill_and_decode exception: " << e.what() << "\n";
        return 1;
    }
}

int test_g14_decode_ladder(ninfer::DeviceContext& device) {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    try {
        PleIndexMetadata ple_meta{};
        ple_meta.multipliers = {1, 2, 3};
        ple_meta.head_offsets.fill(0);
        ple_meta.head_vocab_sizes.fill(1);
        auto synthetic_model = make_synthetic_model(device);

        auto run_at = [&](std::uint32_t max_context, std::int32_t prefill_tokens, const char* label) {
            FlashNextRuntimeConfig cfg{
                .max_concurrency     = 1,
                .max_context         = max_context,
                .state_slot_capacity = 2,
                .prefill_chunk       = 2048,
                .use_cuda_graph      = true,
            };
            const auto curve = flash_next_capacity_curve(cfg);
            auto plan        = finalize_flash_next_runtime_plan(cfg, curve.minimum_main_page_groups);
            const auto buckets = flash_next_decode_graph_buckets(plan.maximum_blocks);
            const std::int32_t live = (prefill_tokens + 1) / 4;
            const auto bucket       = flash_next_decode_graph_select_bucket(buckets, live);
            std::cout << "G14 LADDER " << label << " max_context=" << max_context
                      << " prefill_tokens=" << prefill_tokens << " live_blocks=" << live
                      << " bucket=" << bucket << " envelope=" << buckets.blocks[bucket]
                      << " n_buckets=" << buckets.count << "\n";
            FlashNextRuntimeAllocation alloc(plan);
            alloc.initialize(device.stream);
            FlashNextTextExecutor exec(synthetic_model.view, ple_meta, device, alloc);
            auto lane = exec.allocate_lane();
            constexpr int kChunk = 2048;
            for (std::int32_t first = 0; first < prefill_tokens; first += kChunk) {
                g8_prefill_one(exec, lane, first, kChunk, device, nullptr);
            }
            g14_time_decode_rounds(exec, lane, plan, prefill_tokens, device, label);
            exec.release_lane(lane);
        };

        // Exact 32k tokens = 8192 blocks: both plans should hit the 8192 rung.
        run_at(262144, 32768, "256k-plan-32k-tokens");
        run_at(65536, 32768, "64k-plan-32k-tokens");
        // First token past the 8192-block rung: 17 chunks = 34816 tokens, 8704 blocks.
        run_at(262144, 34816, "256k-plan-34k-tokens-hole");
        run_at(65536, 34816, "64k-plan-34k-tokens");
        std::cout << "PASS: test_g14_decode_ladder\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "test_g14_decode_ladder exception: " << e.what() << "\n";
        return 1;
    }
}

int test_g14_determinism(ninfer::DeviceContext& device) {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    try {
        PleIndexMetadata ple_meta{};
        ple_meta.multipliers = {1, 2, 3};
        ple_meta.head_offsets.fill(0);
        ple_meta.head_vocab_sizes.fill(1);
        auto synthetic_model = make_synthetic_model(device);
        const auto cfg       = g14_runtime_config(true);
        const auto curve     = flash_next_capacity_curve(cfg);
        auto plan            = finalize_flash_next_runtime_plan(cfg, curve.minimum_main_page_groups);
        auto run_once        = [&](std::vector<std::uint16_t>& hidden) {
            FlashNextRuntimeAllocation alloc(plan);
            alloc.initialize(device.stream);
            FlashNextTextExecutor exec(synthetic_model.view, ple_meta, device, alloc);
            auto lane = exec.allocate_lane();
            for (int c = 0; c < 3; ++c) {
                g8_prefill_one(exec, lane, c * 2048, 2048, device, nullptr);
            }
            hidden.assign(2560, 0);
            CUDA_CHECK(cudaMemcpy(hidden.data(), alloc.round_tensors().final_hidden.data,
                                  hidden.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost));
            exec.release_lane(lane);
        };
        std::vector<std::uint16_t> hid_a, hid_b;
        run_once(hid_a);
        run_once(hid_b);
        if (hid_a.size() != hid_b.size() ||
            std::memcmp(hid_a.data(), hid_b.data(), hid_a.size() * sizeof(std::uint16_t)) != 0) {
            std::cerr << "FAIL: G14 two-run 3x2048 on 256k plan was not bitwise identical\n";
            return 1;
        }
        double energy = 0.0;
        int nonfinite = 0;
        for (auto v : hid_a) {
            const float f = bf16_to_float(v);
            if (!std::isfinite(f)) { ++nonfinite; }
            energy += static_cast<double>(f) * static_cast<double>(f);
        }
        if (nonfinite != 0 || !(energy > 0.0) || !std::isfinite(energy)) {
            std::cerr << "FAIL: G14 two-run vacuous energy=" << energy << " nonfinite=" << nonfinite
                      << "\n";
            return 1;
        }
        std::cout << "PASS: test_g14_determinism 3x2048 two-run hidden energy=" << energy
                  << " bitwise match\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "test_g14_determinism exception: " << e.what() << "\n";
        return 1;
    }
}

int test_stage_ledger_smoke(ninfer::DeviceContext& device) {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    try {
        PleIndexMetadata ple_meta{};
        ple_meta.multipliers = {1, 2, 3};
        ple_meta.head_offsets.fill(0);
        ple_meta.head_vocab_sizes.fill(1);
        auto synthetic_model = make_synthetic_model(device);
        FlashNextRuntimeConfig cfg{
            .max_concurrency     = 1,
            .max_context         = 4096,
            .state_slot_capacity = 2,
            .prefill_chunk       = 2048,
            .use_cuda_graph      = false,
        };
        const auto curve = flash_next_capacity_curve(cfg);
        auto plan        = finalize_flash_next_runtime_plan(cfg, curve.minimum_main_page_groups);
        FlashNextRuntimeAllocation alloc(plan);
        alloc.initialize(device.stream);
        FlashNextTextExecutor exec(synthetic_model.view, ple_meta, device, alloc);
        auto lane = exec.allocate_lane();
        g8_prefill_one(exec, lane, 0, 2048, device, nullptr);
        exec.release_lane(lane);

        if (FlashNextStageLedger::is_enabled()) {
            const auto stats = FlashNextStageLedger::instance().last_stats();
            if (stats.total_chunk_ms <= 0.0f || stats.sum_accounted_ms <= 0.0f) {
                std::cerr << "FAIL: stage ledger non-positive timing\n";
                return 1;
            }
            if (stats.accounted_pct < 95.0f || stats.residual_pct >= 5.0f || stats.residual_pct < -5.0f) {
                std::cerr << "FAIL: stage ledger residual out of range: " << stats.residual_pct << "%\n";
                return 1;
            }
        }
        std::cout << "PASS: test_stage_ledger_smoke\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "test_stage_ledger_smoke exception: " << e.what() << "\n";
        return 1;
    }
}

int count_named_bins(const std::filesystem::path& dir, std::string_view prefix) {
    int n = 0;
    if (!std::filesystem::exists(dir)) { return 0; }
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        const auto name = entry.path().filename().string();
        if (name.rfind(prefix, 0) == 0 && name.size() >= 4 &&
            name.compare(name.size() - 4, 4, ".bin") == 0) {
            ++n;
        }
    }
    return n;
}

bool logit_files_equal(const std::filesystem::path& a, const std::filesystem::path& b) {
    const auto sa = std::filesystem::file_size(a);
    const auto sb = std::filesystem::file_size(b);
    if (sa != sb) { return false; }
    std::ifstream fa(a, std::ios::binary);
    std::ifstream fb(b, std::ios::binary);
    std::vector<char> ba(static_cast<std::size_t>(sa));
    std::vector<char> bb(static_cast<std::size_t>(sb));
    fa.read(ba.data(), static_cast<std::streamsize>(sa));
    fb.read(bb.data(), static_cast<std::streamsize>(sb));
    return fa && fb && ba == bb;
}

int test_oracle_chat23_logits_prefill(ninfer::DeviceContext& device, bool trace_mode) {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    try {
        // Reviewer amendment: the delivered patch hardcoded one lane's worktree
        // (E:/NInfer.grok/out), which does not exist in any other checkout.
        // NINFER_TEST_SCRATCH_DIR overrides; otherwise the system temp directory.
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        std::filesystem::path scratch_base;
        if (const char* env = std::getenv("NINFER_TEST_SCRATCH_DIR");
            env != nullptr && env[0] != '\0') {
            scratch_base = std::filesystem::path(env);
        } else {
            std::error_code base_ec;
            scratch_base = std::filesystem::temp_directory_path(base_ec);
            if (base_ec) { scratch_base = std::filesystem::path("."); }
        }
        const std::filesystem::path root = scratch_base / ("g23c-oracle-" + std::to_string(stamp));
        const auto trace   = root / "trace";
        const auto off_dir = root / "logits-off";
        const auto on_dir  = root / "logits-on";
        std::filesystem::create_directories(trace);
        std::filesystem::create_directories(off_dir);
        std::filesystem::create_directories(on_dir);

        // Reviewer amendment. As delivered this traced every stage of all 46
        // prefills: 18,538 files, each preceded by a synchronous device-to-host
        // copy, adding 260 s to the suite. It also could not have worked as
        // written, because text_executor.cpp latches NINFER_FLASH_NEXT_TRACE_STAGES
        // into a function-local static on the first prefill of the process, so
        // tracing can afterwards be neither redirected nor switched off. The proof
        // is therefore split into its own process: trace_mode dumps one position
        // with tracing on and asserts stage files appeared, which a decode round
        // would never write; the comparison run traces nothing.
        if (trace_mode) {
            const std::string trace_env = trace.generic_string();
#ifdef _WIN32
            (void)_putenv_s("NINFER_FLASH_NEXT_TRACE_STAGES", trace_env.c_str());
#else
            (void)setenv("NINFER_FLASH_NEXT_TRACE_STAGES", trace_env.c_str(), 1);
#endif
        }

        PleIndexMetadata ple_meta{};
        ple_meta.multipliers = {1, 2, 3};
        ple_meta.head_offsets.fill(0);
        ple_meta.head_vocab_sizes.fill(1);
        auto synthetic_model = make_synthetic_model(device);

        auto run_arm = [&](bool mma, const std::filesystem::path& out_dir,
                           std::int32_t positions_wanted) -> int {
#ifdef _WIN32
            (void)_putenv_s("NINFER_FLASH_NEXT_QSA_PREFILL_MMA", mma ? "1" : "0");
#else
            (void)setenv("NINFER_FLASH_NEXT_QSA_PREFILL_MMA", mma ? "1" : "0", 1);
#endif
            FlashNextRuntimeConfig cfg{
                .max_concurrency     = 1,
                .max_context         = 512,
                .state_slot_capacity = 2,
                .prefill_chunk       = 128,
                .use_cuda_graph      = false,
                .use_qsa_prefill_mma = mma,
            };
            const auto curve = flash_next_capacity_curve(cfg);
            auto plan        = finalize_flash_next_runtime_plan(cfg, curve.maximum_main_page_groups);
            if (plan.config.use_qsa_prefill_mma != mma) {
                std::cerr << "plan.config.use_qsa_prefill_mma mismatch for mma=" << mma << "\n";
                return 1;
            }
            FlashNextRuntimeAllocation alloc(plan);
            alloc.initialize(device.stream);
            FlashNextTextExecutor exec(synthetic_model.view, ple_meta, device, alloc);
            dump_oracle_chat23_logits(exec, device, plan, out_dir, positions_wanted);
            for (int p = 0; p < positions_wanted; ++p) {
                char name[32];
                std::snprintf(name, sizeof(name), "pos%04d_logits.bin", p);
                if (!std::filesystem::exists(out_dir / name)) {
                    std::cerr << "missing " << (out_dir / name).string() << "\n";
                    return 1;
                }
            }
            return 0;
        };

        if (trace_mode) {
            if (run_arm(false, off_dir, 1) != 0) { return 1; }
            const int chunks = count_named_bins(trace, "chunk");
            if (chunks == 0) {
                std::cerr << "TRACE_STAGES wrote no chunk*.bin: the dump did not reach "
                             "execute_prefill_chunk (a decode round writes nothing)\n";
                return 1;
            }
            std::cout << "oracle-chat23 prefill path proven: " << chunks
                      << " stage files from one dumped position\n";
            std::cout << "PASS: test_oracle_chat23_logits_trace\n" << std::flush;
            std::error_code trace_ec;
            std::filesystem::remove_all(root, trace_ec);
            return 0;
        }

        if (run_arm(false, off_dir, 23) != 0) { return 1; }
        if (run_arm(true, on_dir, 23) != 0) { return 1; }

        int identical = 0;
        int differ    = 0;
        for (int p = 0; p < 23; ++p) {
            char name[32];
            std::snprintf(name, sizeof(name), "pos%04d_logits.bin", p);
            if (logit_files_equal(off_dir / name, on_dir / name)) {
                ++identical;
            } else {
                ++differ;
            }
        }
        std::cout << "oracle-chat23 synthetic bitwise: identical=" << identical << " differ=" << differ
                  << " of 23\n";
        std::cout << "PASS: test_oracle_chat23_logits_prefill\n" << std::flush;
        std::error_code ec;
        std::filesystem::remove_all(root, ec);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "test_oracle_chat23_logits_prefill exception: " << e.what() << "\n";
        return 1;
    }
}

int test_oracle_chat23_sched(ninfer::DeviceContext& device) {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    try {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        std::filesystem::path scratch_base;
        if (const char* env = std::getenv("NINFER_TEST_SCRATCH_DIR");
            env != nullptr && env[0] != '\0') {
            scratch_base = std::filesystem::path(env);
        } else {
            std::error_code base_ec;
            scratch_base = std::filesystem::temp_directory_path(base_ec);
            if (base_ec) { scratch_base = std::filesystem::path("."); }
        }
        const std::filesystem::path root = scratch_base / ("g24-oracle-sched-" + std::to_string(stamp));
        const auto old_dir = root / "logits-old";
        const auto new_dir = root / "logits-new";
        std::filesystem::create_directories(old_dir);
        std::filesystem::create_directories(new_dir);

        PleIndexMetadata ple_meta{};
        ple_meta.multipliers = {1, 2, 3};
        ple_meta.head_offsets.fill(0);
        ple_meta.head_vocab_sizes.fill(1);
        auto synthetic_model = make_synthetic_model(device);
        set_qsa_prefill_mma_env(true);

        auto run_arm = [&](bool use_new, const std::filesystem::path& out_dir) -> int {
            set_qsa_mma_sched_env(use_new);
            FlashNextRuntimeConfig cfg{
                .max_concurrency     = 1,
                .max_context         = 512,
                .state_slot_capacity = 2,
                .prefill_chunk       = 128,
                .use_cuda_graph      = false,
                .use_qsa_prefill_mma = true,
            };
            const auto curve = flash_next_capacity_curve(cfg);
            auto plan        = finalize_flash_next_runtime_plan(cfg, curve.maximum_main_page_groups);
            FlashNextRuntimeAllocation alloc(plan);
            alloc.initialize(device.stream);
            FlashNextTextExecutor exec(synthetic_model.view, ple_meta, device, alloc);
            dump_oracle_chat23_logits(exec, device, plan, out_dir, 23);
            return 0;
        };

        if (run_arm(false, old_dir) != 0) { return 1; }
        if (run_arm(true, new_dir) != 0) { return 1; }
        int identical = 0;
        int differ    = 0;
        for (int p = 0; p < 23; ++p) {
            char name[32];
            std::snprintf(name, sizeof(name), "pos%04d_logits.bin", p);
            if (logit_files_equal(old_dir / name, new_dir / name)) {
                ++identical;
            } else {
                ++differ;
            }
        }
        std::cout << "oracle-chat23 sched synthetic bitwise: identical=" << identical
                  << " differ=" << differ << " of 23\n";
        if (differ != 0) {
            std::cout << "DECLARE: G24 P->BF16 PV rounding (or MMA fragment layout) differs from "
                         "the G17 serial-PV MMA kernel on the synthetic fixture.\n";
        }
        std::cout << "PASS: test_oracle_chat23_sched\n" << std::flush;
        std::error_code ec;
        std::filesystem::remove_all(root, ec);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "test_oracle_chat23_sched exception: " << e.what() << "\n";
        return 1;
    }
}

} // namespace

int main(int argc, char** argv) {
    const std::string mode = argc >= 2 ? argv[1] : std::string{};
    if (mode == "g8-admission") { return test_g8_admission_cpu(); }
    if (mode == "g14-admission") { return test_g14_admission_cpu(); }

    if (test_ledger_cpu() != 0) return 1;
    if (test_ledger_prefill_chunk_cpu() != 0) return 1;
    if (test_ple_boundary_lifecycle_cpu() != 0) return 1;

    int device_count              = 0;
    const cudaError_t count_error = cudaGetDeviceCount(&device_count);
    if (cuda_unavailable(count_error) || device_count == 0) {
        std::cout << "SKIP: CUDA device tests (no usable device)\n";
        return 0;
    }
    CUDA_CHECK(count_error);

    ninfer::DeviceContext device(0);

    if (mode == "g8-decode-2048") { return test_g8_decode_crossing(device, 2048); }
    if (mode == "g8-decode-8192") { return test_g8_decode_crossing(device, 8192); }
    if (mode == "g8-decode-32768") { return test_g8_decode_crossing(device, 32768); }
    if (mode == "g8-prefill") { return test_g8_prefill_soak(device); }
    if (mode == "g8-char") { return test_g8_decode_characterization(device); }
    if (mode == "g8-graph-nodes") { return test_g8_graph_node_diff(device); }
    if (mode == "g8-stage-checksum") { return test_g8_stage_checksum(device); }
    if (mode == "g9-gate") { return test_g9_prefill_determinism(device); }
    if (mode == "oracle-chat23") { return test_oracle_chat23_logits_prefill(device, false); }
    if (mode == "oracle-chat23-trace") { return test_oracle_chat23_logits_prefill(device, true); }
    if (mode == "oracle-chat23-sched") { return test_oracle_chat23_sched(device); }
    if (mode == "stage-ledger-smoke") { return test_stage_ledger_smoke(device); }
    if (mode == "prefill-timing") { return test_prefill_chunk_timing_benchmark(device, false); }
    if (mode == "prefill-timing-ab") { return test_prefill_chunk_timing_benchmark(device, true); }
    if (mode == "prefill-timing-host-sync-ab") {
        return test_prefill_chunk_timing_benchmark(device, true, true);
    }
    if (mode == "prefill-timing-moe-shared-ab") {
        return test_prefill_chunk_timing_benchmark(device, true, false, true);
    }
    if (mode == "prefill-timing-moe-staging-ab") {
        return test_prefill_chunk_timing_benchmark(device, true, false, false, false, true);
    }
    if (mode == "prefill-timing-qsa-sched-ab") {
        return test_prefill_chunk_timing_benchmark(device, true, false, false, true);
    }
    if (mode == "g14-prefill") { return test_g14_prefill_and_decode(device, 128); }
    if (mode == "g14-prefill-32") { return test_g14_prefill_and_decode(device, 32); }
    if (mode == "g14-det") { return test_g14_determinism(device); }
    if (mode == "g14-ladder") { return test_g14_decode_ladder(device); }

    if (test_cuda_ledger_and_executor(device) != 0) return 1;
    if (test_finite_model_stages(device) != 0) return 1;
    if (test_prefill_chunk_executor(device) != 0) return 1;
    if (test_prefill_chunk_workspace_envelope(device) != 0) return 1;
    if (test_cuda_graph_decode_equivalence(device) != 0) return 1;
    if (test_decode_graph_bucket_key_layout(device) != 0) return 1;
    if (test_cuda_graph_bucketed_decode(device) != 0) return 1;
    if (test_cuda_graph_frontier_masking_and_churn(device) != 0) return 1;
    if (test_measure_cuda_graph_footprint(device) != 0) return 1;
    if (test_cuda_graph_timing_benchmark(device) != 0) return 1;
    if (test_prefill_chunk_timing_benchmark(device, false) != 0) return 1;
    if (test_stage_ledger_smoke(device) != 0) return 1;
    if (test_g9_prefill_determinism(device) != 0) return 1;

    std::cout << "OK Flash-Next Text Executor\n";
    return 0;
}
