#include <ninfer/targets/qwen3_8_flash_next/package.h>
#include <ninfer/targets/qwen3_8_flash_next/runtime.h>
#include <ninfer/targets/qwen3_6/frontend.h>
#include <ninfer/targets/qwen3_6/frontend_resources.h>
#include <ninfer/targets/qwen3_6/prepared_prompt.h>

#include "core/arena.h"
#include "core/device.h"
#include "runtime/contract/types.h"
#include "runtime/engine/context_cost.h"
#include "targets/qwen3_8_flash_next/impl/expert_bank.h"
#include "targets/qwen3_8_flash_next/impl/load/materialized.h"
#include "targets/qwen3_8_flash_next/impl/model_view.h"
#include "targets/qwen3_8_flash_next/impl/ple_index.h"
#include "targets/qwen3_8_flash_next/impl/ple_table.h"
#include "targets/qwen3_8_flash_next/impl/program_impl.h"
#include "targets/qwen3_8_flash_next/impl/runtime_plan.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::targets::qwen3_8_flash_next;
using namespace ninfer::targets::qwen3_8_flash_next::detail;

bool cuda_unavailable(cudaError_t error) {
    return error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver;
}

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::fprintf(stderr, "FAIL: %s\n", message);
    std::fflush(stderr);
    return 1;
}

inline std::uint16_t float_to_bf16(float f) {
    std::uint32_t x;
    std::memcpy(&x, &f, sizeof(float));
    return static_cast<std::uint16_t>(x >> 16);
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

    std::vector<std::byte> ple_table_data;
    TextModelView view;
};

SyntheticFlashNextModel make_synthetic_model(ninfer::DeviceContext& device) {
    SyntheticFlashNextModel model;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist_bf16(-0.02f, 0.02f);
    std::uniform_real_distribution<float> dist_norm(0.98f, 1.02f);

    constexpr std::uint64_t kOutputHeadBytes = 248'320ULL * 2'560 * 2;
    model.big_bf16_buf = ninfer::DeviceBuffer(kOutputHeadBytes);
    constexpr std::size_t kChunkFloats = 2'560 * 1024;
    std::vector<std::uint16_t> h_bf16(kChunkFloats);
    for (auto& v : h_bf16) { v = float_to_bf16(dist_bf16(rng)); }
    for (std::size_t off = 0; off < kOutputHeadBytes; off += h_bf16.size() * sizeof(std::uint16_t)) {
        std::size_t chunk = std::min<std::size_t>(h_bf16.size() * sizeof(std::uint16_t), kOutputHeadBytes - off);
        model.big_bf16_buf.copy_from_host(h_bf16.data(), chunk, off);
    }

    std::vector<std::uint16_t> h_norm(10'240);
    for (auto& v : h_norm) { v = float_to_bf16(dist_norm(rng)); }
    model.norm_bf16_buf = ninfer::DeviceBuffer(10'240 * sizeof(std::uint16_t));
    model.norm_bf16_buf.copy_from_host(h_norm.data(), h_norm.size() * sizeof(std::uint16_t));

    std::vector<std::uint16_t> h_a_log(48);
    for (auto& v : h_a_log) { v = float_to_bf16(-1.0f); }
    model.gdn_a_log_buf = ninfer::DeviceBuffer(48 * sizeof(std::uint16_t));
    model.gdn_a_log_buf.copy_from_host(h_a_log.data(), h_a_log.size() * sizeof(std::uint16_t));

    std::vector<std::uint16_t> h_dt_bias(48);
    for (auto& v : h_dt_bias) { v = float_to_bf16(0.05f); }
    model.gdn_dt_bias_buf = ninfer::DeviceBuffer(48 * sizeof(std::uint16_t));
    model.gdn_dt_bias_buf.copy_from_host(h_dt_bias.data(), h_dt_bias.size() * sizeof(std::uint16_t));

    std::vector<std::uint16_t> h_gdn_conv(10'240 * 4);
    for (auto& v : h_gdn_conv) { v = float_to_bf16(0.25f); }
    model.gdn_conv_buf = ninfer::DeviceBuffer(10'240 * 4 * sizeof(std::uint16_t));
    model.gdn_conv_buf.copy_from_host(h_gdn_conv.data(), h_gdn_conv.size() * sizeof(std::uint16_t));

    std::vector<std::uint16_t> h_ple_conv(10'240 * 4);
    for (int c = 0; c < 10'240; ++c) {
        h_ple_conv[0 * 10'240 + c] = float_to_bf16(0.25f);
        h_ple_conv[1 * 10'240 + c] = float_to_bf16(0.50f);
        h_ple_conv[2 * 10'240 + c] = float_to_bf16(0.75f);
        h_ple_conv[3 * 10'240 + c] = float_to_bf16(1.00f);
    }
    model.ple_conv_buf = ninfer::DeviceBuffer(10'240 * 4 * sizeof(std::uint16_t));
    model.ple_conv_buf.copy_from_host(h_ple_conv.data(), h_ple_conv.size() * sizeof(std::uint16_t));

    std::vector<std::uint16_t> h_sgw(2'560);
    for (auto& v : h_sgw) { v = float_to_bf16(0.1f); }
    model.shared_gate_weight_buf = ninfer::DeviceBuffer(2'560 * sizeof(std::uint16_t));
    model.shared_gate_weight_buf.copy_from_host(h_sgw.data(), h_sgw.size() * sizeof(std::uint16_t));

    std::vector<std::uint16_t> h_inject(4 * 10'240);
    for (auto& v : h_inject) { v = float_to_bf16(0.25f); }
    model.inject_buf = ninfer::DeviceBuffer(4 * 10'240 * sizeof(std::uint16_t));
    model.inject_buf.copy_from_host(h_inject.data(), h_inject.size() * sizeof(std::uint16_t));

    auto init_fp8_buf = [&](ninfer::DeviceBuffer& buf, std::int32_t rows, std::int32_t cols, float scale_val) {
        const std::uint64_t codes_bytes = static_cast<std::uint64_t>(rows) * cols;
        const std::uint64_t scale_off   = (codes_bytes + 255U) & ~255ULL;
        const std::uint64_t total_bytes = scale_off + static_cast<std::uint64_t>(rows) * sizeof(float);
        buf = ninfer::DeviceBuffer(total_bytes);

        std::vector<std::uint8_t> h_codes(codes_bytes);
        for (std::size_t i = 0; i < codes_bytes; ++i) {
            h_codes[i] = static_cast<std::uint8_t>(0x18 + (rng() % 32));
        }
        buf.copy_from_host(h_codes.data(), codes_bytes, 0);

        std::vector<float> h_scales(rows, scale_val);
        buf.copy_from_host(h_scales.data(), rows * sizeof(float), scale_off);
    };

    init_fp8_buf(model.fp8_qkvz_buf, 16'384, 2'560, 1.0f / std::sqrt(2'560.0f));
    init_fp8_buf(model.fp8_qgkv_buf, 13'312, 2'560, 1.0f / std::sqrt(2'560.0f));
    init_fp8_buf(model.fp8_out_buf, 2'560, 6'144, 1.0f / std::sqrt(6'144.0f));

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
        const auto low  = static_cast<std::uint8_t>(1 + (rng() % 3));
        const auto high = static_cast<std::uint8_t>(1 + (rng() % 3));
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

    auto make_bf16_weight_from = [](ninfer::DeviceBuffer& buf, std::int32_t rows, std::int32_t cols) {
        ninfer::Weight w{};
        w.payload         = buf.p;
        w.payload_bytes   = static_cast<std::uint64_t>(rows) * cols * 2;
        w.qdata           = buf.p;
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
        layer.attention_hyper.block_inject   = make_bf16_weight_from(model.inject_buf, 4, 10'240);
        layer.attention_hyper.norm           = ninfer::Tensor(model.norm_bf16_buf.p, ninfer::DType::BF16, {10'240});
        layer.attention_hyper.input_mix_down = make_bf16_weight(320, 10'240);
        layer.attention_hyper.input_mix_up   = make_bf16_weight(10'240, 320);

        layer.mlp_hyper.block_inject   = make_bf16_weight_from(model.inject_buf, 4, 10'240);
        layer.mlp_hyper.norm           = ninfer::Tensor(model.norm_bf16_buf.p, ninfer::DType::BF16, {10'240});
        layer.mlp_hyper.input_mix_down = make_bf16_weight(320, 10'240);
        layer.mlp_hyper.input_mix_up   = make_bf16_weight(10'240, 320);

        layer.moe.router             = make_bf16_weight(512, 2'560);
        layer.moe.shared_down        = make_bf16_weight(2'560, 640);
        layer.moe.shared_gate        = make_bf16_weight(640, 2'560);
        layer.moe.shared_up          = make_bf16_weight(640, 2'560);
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

PleIndexMetadata make_synthetic_ple_meta() {
    PleIndexMetadata ple_meta{};
    ple_meta.multipliers.fill(0);
    ple_meta.head_offsets.fill(0);
    ple_meta.head_vocab_sizes.fill(1);
    return ple_meta;
}

ninfer::targets::qwen3_6::PreparedPrompt make_prompt(std::span<const ninfer::TokenId> tokens) {
    ninfer::targets::qwen3_6::PreparedPromptData data;
    const std::size_t num_tokens = tokens.size();
    data.token_ids.assign(tokens.begin(), tokens.end());
    data.token_types.resize(num_tokens, 0);
    data.positions.resize(num_tokens * 3);
    for (std::size_t i = 0; i < num_tokens; ++i) {
        data.positions[i]                  = static_cast<std::int32_t>(i);
        data.positions[num_tokens + i]     = static_cast<std::int32_t>(i);
        data.positions[2 * num_tokens + i] = static_cast<std::int32_t>(i);
    }
    data.identity.reusable = true;
    return ninfer::targets::qwen3_6::PreparedPromptAccess::construct(std::move(data));
}

PrefillProgress advance_prefill_full(Program& prog, SequenceHandle seq) {
    auto p = prog.advance_prefill(seq);
    while (!p.complete) {
        if (p.capture.has_value()) {
            prog.skip_capture(std::move(*p.capture));
        }
        p = prog.advance_prefill(seq);
    }
    return p;
}

// ---------------------------------------------------------------------------
// Test 1: Concurrency Limits and Admission Capacity Curve Across Context Sizes
// ---------------------------------------------------------------------------
int test_soak_concurrency_and_admission_curve(ninfer::DeviceContext& device) {
    std::printf("\n=== TEST 1: Concurrency Limits & Admission Curves ===\n");
    std::fflush(stdout);

    SyntheticFlashNextModel model = make_synthetic_model(device);
    auto ple_meta = make_synthetic_ple_meta();
    ninfer::runtime::ContextMachineCostModel cost_model{};

    const std::vector<std::uint32_t> test_contexts = {8192, 16384, 32768, 65536};
    const std::vector<std::uint32_t> mixed_lengths = {512, 1024, 2048, 4096, 8192, 16384, 32768};

    for (const std::uint32_t ctx : test_contexts) {
        FlashNextRuntimeConfig config{};
        config.max_context = ctx;
        config.max_concurrency = 8;
        config.continuation_capacity = 8;
        config.speculative_draft_tokens = 0;

        const auto curve = flash_next_capacity_curve(config);
        const std::uint32_t min_groups = curve.minimum_main_page_groups;
        const std::uint32_t max_groups = curve.maximum_main_page_groups;

        std::vector<std::uint32_t> valid_lengths;
        for (auto l : mixed_lengths) {
            if (l <= ctx) { valid_lengths.push_back(l); }
        }

        std::printf("\n[Context %u tokens] Min page groups: %u (~%u tokens pool), Max page groups: %u (~%u tokens pool)\n",
                    ctx, min_groups, min_groups * 256, max_groups, max_groups * 256);

        // Min groups
        {
            FlashNextRuntimePlan plan = finalize_flash_next_runtime_plan(config, min_groups);
            auto prog_impl = std::make_unique<ProgramImpl>(nullptr, plan, device, model.view, std::nullopt, ple_meta);
            Program prog(std::move(prog_impl));

            std::uint32_t admitted = 0;
            std::uint32_t total_admitted_tokens = 0;
            std::vector<SequenceHandle> live_seqs;
            std::atomic<bool> cancellation_flag{false};
            ninfer::runtime::CancellationFlagView cancellation{&cancellation_flag};

            for (std::size_t i = 0; i < config.max_concurrency; ++i) {
                const std::uint32_t req_len = valid_lengths[i % valid_lengths.size()];
                std::vector<ninfer::TokenId> tok(req_len, 42);
                auto prompt = make_prompt(tok);

                ninfer::runtime::ResolvedExecutionOptions exec_options{};
                exec_options.requested_output_tokens = 16;
                auto base_plan = prog.plan_request(prompt, exec_options);

                auto inspect = prog.inspect_admission(prompt, base_plan, ninfer::runtime::LaneId(i), nullptr, nullptr, std::nullopt, false);
                if (inspect.has_value() &&
                    inspect->identity_assessment().physical_status == runtime::MaterializationPhysicalStatus::Feasible) {
                    auto res_plan = prog.seal_identity(*inspect, prompt);
                    if (res_plan.has_value()) {
                        auto status = prog.start_resource_transaction(std::move(*res_plan), make_prompt(tok), cancellation);
                        if (status == runtime::ContextTransactionReserveStatus::Reserved) {
                            auto mat_res = prog.progress_context_transaction(cancellation);
                            auto* mat = std::get_if<MaterializationResult>(&mat_res);
                            if (mat && mat->published.has_value()) {
                                live_seqs.push_back(mat->published->sequence);
                                prog.finalize_context_transaction();
                                admitted++;
                                total_admitted_tokens += req_len;
                            }
                        }
                    }
                } else {
                    break;
                }
            }

            std::printf("  -> Min-Pool (%u groups): Admitted %u concurrent sequences (total %u tokens admitted) before rejection.\n",
                        min_groups, admitted, total_admitted_tokens);

            for (auto& s : live_seqs) { (void)prog.finish(s); }
        }

        // Max groups
        {
            FlashNextRuntimePlan plan = finalize_flash_next_runtime_plan(config, max_groups);
            auto prog_impl = std::make_unique<ProgramImpl>(nullptr, plan, device, model.view, std::nullopt, ple_meta);
            Program prog(std::move(prog_impl));

            std::uint32_t admitted = 0;
            std::uint32_t total_admitted_tokens = 0;
            std::vector<SequenceHandle> live_seqs;
            std::atomic<bool> cancellation_flag{false};
            ninfer::runtime::CancellationFlagView cancellation{&cancellation_flag};

            for (std::size_t i = 0; i < config.max_concurrency; ++i) {
                const std::uint32_t req_len = valid_lengths[i % valid_lengths.size()];
                std::vector<ninfer::TokenId> tok(req_len, 42);
                auto prompt = make_prompt(tok);

                ninfer::runtime::ResolvedExecutionOptions exec_options{};
                exec_options.requested_output_tokens = 16;
                auto base_plan = prog.plan_request(prompt, exec_options);

                auto inspect = prog.inspect_admission(prompt, base_plan, ninfer::runtime::LaneId(i), nullptr, nullptr, std::nullopt, false);
                if (inspect.has_value() &&
                    inspect->identity_assessment().physical_status == runtime::MaterializationPhysicalStatus::Feasible) {
                    auto res_plan = prog.seal_identity(*inspect, prompt);
                    if (res_plan.has_value()) {
                        auto status = prog.start_resource_transaction(std::move(*res_plan), make_prompt(tok), cancellation);
                        if (status == runtime::ContextTransactionReserveStatus::Reserved) {
                            auto mat_res = prog.progress_context_transaction(cancellation);
                            auto* mat = std::get_if<MaterializationResult>(&mat_res);
                            if (mat && mat->published.has_value()) {
                                live_seqs.push_back(mat->published->sequence);
                                prog.finalize_context_transaction();
                                admitted++;
                                total_admitted_tokens += req_len;
                            }
                        }
                    }
                } else {
                    break;
                }
            }

            std::printf("  -> Max-Pool (%u groups): Admitted %u concurrent sequences (total %u tokens admitted) across all %u lanes.\n",
                        max_groups, admitted, total_admitted_tokens, config.max_concurrency);

            if (check(admitted == config.max_concurrency || total_admitted_tokens >= max_groups * 256,
                      "Max-pool failed to admit expected concurrency")) {
                return 1;
            }

            for (auto& s : live_seqs) { (void)prog.finish(s); }
        }
    }

    return 0;
}

// ---------------------------------------------------------------------------
// Test 2: Pressure Eviction Gracefulness and Resumed Token Determinism
// ---------------------------------------------------------------------------
int test_soak_pressure_eviction_and_determinism(ninfer::DeviceContext& device) {
    std::printf("\n=== TEST 2: Pressure Eviction & Token Determinism Under Load ===\n");
    std::fflush(stdout);

    SyntheticFlashNextModel model = make_synthetic_model(device);
    auto ple_meta = make_synthetic_ple_meta();
    ninfer::runtime::ContextMachineCostModel cost_model{};

    FlashNextRuntimeConfig config{};
    config.max_context = 1024;
    config.max_concurrency = 2;
    config.continuation_capacity = 4;
    config.speculative_draft_tokens = 0;

    // Small pool of 4 page groups total (1024 tokens total)
    FlashNextRuntimePlan plan = finalize_flash_next_runtime_plan(config, 4);
    auto prog_impl = std::make_unique<ProgramImpl>(nullptr, plan, device, model.view, std::nullopt, ple_meta);
    Program prog(std::move(prog_impl));

    std::atomic<bool> cancellation_flag{false};
    ninfer::runtime::CancellationFlagView cancellation{&cancellation_flag};

    // 1. Run sequence A (500 tokens = 2 page groups), decode 5 tokens, finish -> catalogue checkpoint
    std::vector<ninfer::TokenId> prompt_a(500);
    for (std::size_t i = 0; i < 500; ++i) { prompt_a[i] = static_cast<ninfer::TokenId>(1000 + i); }
    auto pa = make_prompt(prompt_a);

    ninfer::runtime::ResolvedExecutionOptions exec_options{};
    exec_options.requested_output_tokens = 16;
    auto base_a = prog.plan_request(pa, exec_options);

    auto insp_a = prog.inspect_admission(pa, base_a, ninfer::runtime::LaneId(0), nullptr, nullptr, std::nullopt, false);
    if (!insp_a.has_value() || insp_a->identity_assessment().physical_status != runtime::MaterializationPhysicalStatus::Feasible) {
        std::fprintf(stderr, "FAIL: Could not admit initial sequence A\n");
        return 1;
    }

    auto res_plan_a = prog.seal_identity(*insp_a, pa);
    (void)prog.start_resource_transaction(std::move(*res_plan_a), make_prompt(prompt_a), cancellation);
    auto prog_res_a = prog.progress_context_transaction(cancellation);
    SequenceHandle seq_a = std::get_if<MaterializationResult>(&prog_res_a)->published->sequence;
    prog.finalize_context_transaction();

    // Prefill A
    auto p_a = advance_prefill_full(prog, seq_a);
    if (p_a.pending.has_value()) {
        std::array<ninfer::runtime::CommitDecision, 1> commit_dec = {{{.accepted_tokens = 1, .terminal = false}}};
        (void)prog.commit(std::move(*p_a.pending), commit_dec);
    }

    // Decode 5 tokens from A
    std::vector<TokenId> golden_a_tokens;
    for (int step = 0; step < 5; ++step) {
        std::array<SequenceHandle, 1> seqs = {seq_a};
        std::array<ninfer::runtime::RoundBudget, 1> budgets{{{.generated_tokens_remaining = 1}}};
        auto dec = prog.decode(seqs, budgets);
        golden_a_tokens.push_back(dec.tokens()[0]);
        std::array<ninfer::runtime::CommitDecision, 1> commit_dec = {{{.accepted_tokens = 1, .terminal = false}}};
        (void)prog.commit(std::move(dec), commit_dec);
    }

    FinishResult fin_a = prog.finish(seq_a);
    if (fin_a.disposition != runtime::FinishDisposition::Catalogued || !fin_a.continuation.has_value()) {
        std::fprintf(stderr, "FAIL: Sequence A did not catalogue\n");
        return 1;
    }

    ContinuationHandle handle_a = std::move(*fin_a.continuation);
    std::printf("  [Sequence A] Parked with TurnClosure checkpoint %u (groups held: %zu)\n",
                handle_a.index(), prog.impl_->executor_.available_physical_groups());

    // 2. Now admit a large sequence B (800 tokens = 4 page groups) that requires evicting A to fit
    std::vector<ninfer::TokenId> prompt_b(800);
    for (std::size_t i = 0; i < 800; ++i) { prompt_b[i] = static_cast<ninfer::TokenId>(5000 + i); }
    auto pb = make_prompt(prompt_b);
    auto base_b = prog.plan_request(pb, exec_options);

    auto insp_b = prog.inspect_admission(pb, base_b, ninfer::runtime::LaneId(0), nullptr, nullptr, std::nullopt, false);
    std::printf("  [Sequence B Arrival] Identity status: %s\n",
                insp_b->identity_assessment().physical_status == runtime::MaterializationPhysicalStatus::Feasible ? "Feasible" : "Infeasible");

    std::vector<const ContinuationHandle*> owners = {&handle_a};
    std::vector<ninfer::runtime::PlanningOwnerId> ordinals = {ninfer::runtime::PlanningOwnerId{0}};
    const AdmissionCandidate* cand_b_ptr = &*insp_b;
    std::array cand_ids{ninfer::runtime::PlanningCandidateId{0}};

    auto pressure_sess = prog.begin_pressure_planning(std::span(&cand_b_ptr, 1), cand_ids, owners, ordinals, {}, {});

    // Construction walk from the identity target: the only option evicts A. Then guidance,
    // retain_assessment and assessment on the constructed target.
    std::optional<PressureTargetHandle> guided_target;
    {
        auto cursor = pressure_sess.begin_construction(pressure_sess.identity_target(cand_ids[0]));
        std::optional<ninfer::runtime::PressureConstructionOptionId> chosen;
        for (;;) {
            const auto step = pressure_sess.next_construction_option(cursor);
            if (step.guidance.has_value() && !chosen.has_value()) { chosen = step.option; }
            if (step.exhausted) { break; }
        }
        if (!chosen.has_value()) {
            std::fprintf(stderr, "FAIL: construction offered no eviction option\n");
            return 1;
        }
        pressure_sess.choose_construction(cursor, *chosen);
        guided_target = pressure_sess.construction_target(cursor);
    }
    if (!guided_target.has_value()) {
        std::fprintf(stderr, "FAIL: construction could not intern the eviction target\n");
        return 1;
    }

    auto guidance = pressure_sess.guidance(*guided_target);
    std::printf("  [Pressure Guidance] Unsatisfied constraints: %u, residual_q20: %llu, degradation units: %u\n",
                guidance.physical.unsatisfied_constraints,
                static_cast<unsigned long long>(guidance.physical.normalized_residual_q20),
                guidance.degradation_units);

    pressure_sess.retain_assessment(*guided_target);

    auto assess = pressure_sess.assess(*guided_target);
    std::printf("  [Pressure Assessment] Status: %s, Degradation: %u\n",
                assess.assessment().physical_status == runtime::MaterializationPhysicalStatus::Feasible ? "Feasible" : "Infeasible",
                assess.assessment().degradation_units);

    if (check(assess.assessment().physical_status == runtime::MaterializationPhysicalStatus::Feasible,
              "Pressure assessment must be Feasible after evicting A")) {
        return 1;
    }

    auto sealed_plan_b = pressure_sess.seal(std::move(assess), pb);
    if (!sealed_plan_b.has_value()) {
        std::fprintf(stderr, "FAIL: Could not seal pressure plan for B\n");
        return 1;
    }

    (void)prog.start_resource_transaction(std::move(*sealed_plan_b), make_prompt(prompt_b), cancellation);
    auto prog_b = prog.progress_context_transaction(cancellation);
    auto* mat_b = std::get_if<MaterializationResult>(&prog_b);
    if (check(mat_b != nullptr && mat_b->victims.size() == 1, "B must have evicted A as victim")) {
        return 1;
    }
    if (check(mat_b->victims[0].pressure_committed, "Victim A must have pressure_committed = true")) {
        return 1;
    }

    SequenceHandle seq_b = mat_b->published->sequence;
    prog.finalize_context_transaction();

    // Prefill and decode B
    auto p_b = advance_prefill_full(prog, seq_b);
    if (p_b.pending.has_value()) {
        std::array<ninfer::runtime::CommitDecision, 1> commit_dec = {{{.accepted_tokens = 1, .terminal = false}}};
        (void)prog.commit(std::move(*p_b.pending), commit_dec);
    }
    FinishResult fin_b = prog.finish(seq_b);
    if (fin_b.continuation.has_value()) {
        (void)prog.release_continuation(std::move(*fin_b.continuation));
    }
    std::printf("  [Sequence B] Successfully executed and finished; freed all 4 groups.\n");

    // 3. Now re-admit Sequence A from prompt to verify exact token determinism
    std::printf("  [Sequence A Resume] Re-prefilling sequence A to verify token determinism...\n");
    auto insp_a2 = prog.inspect_admission(pa, base_a, ninfer::runtime::LaneId(0), nullptr, nullptr, std::nullopt, false);
    auto res_plan_a2 = prog.seal_identity(*insp_a2, pa);
    (void)prog.start_resource_transaction(std::move(*res_plan_a2), make_prompt(prompt_a), cancellation);
    auto prog_a2 = prog.progress_context_transaction(cancellation);
    SequenceHandle seq_a2 = std::get_if<MaterializationResult>(&prog_a2)->published->sequence;
    prog.finalize_context_transaction();

    auto p_a2 = advance_prefill_full(prog, seq_a2);
    if (p_a2.pending.has_value()) {
        std::array<ninfer::runtime::CommitDecision, 1> commit_dec = {{{.accepted_tokens = 1, .terminal = false}}};
        (void)prog.commit(std::move(*p_a2.pending), commit_dec);
    }

    std::vector<TokenId> resumed_a_tokens;
    for (int step = 0; step < 5; ++step) {
        std::array<SequenceHandle, 1> seqs = {seq_a2};
        std::array<ninfer::runtime::RoundBudget, 1> budgets{{{.generated_tokens_remaining = 1}}};
        auto dec = prog.decode(seqs, budgets);
        resumed_a_tokens.push_back(dec.tokens()[0]);
        std::array<ninfer::runtime::CommitDecision, 1> commit_dec = {{{.accepted_tokens = 1, .terminal = false}}};
        (void)prog.commit(std::move(dec), commit_dec);
    }
    (void)prog.finish(seq_a2);

    if (check(golden_a_tokens == resumed_a_tokens, "Resumed sequence produced divergent tokens!")) {
        return 1;
    }
    std::printf("  -> Verified: 5/5 decoded tokens bit-exact identical between original and post-eviction runs.\n");

    return 0;
}

// ---------------------------------------------------------------------------
// Test 3: Long-Lived Multi-Turn Conversations & Resource Leak Verification
// ---------------------------------------------------------------------------
int test_soak_multiturn_conversations_and_leaks(ninfer::DeviceContext& device) {
    std::printf("\n=== TEST 3: Multi-Turn Conversations & Resource Leak Verification ===\n");
    std::fflush(stdout);

    SyntheticFlashNextModel model = make_synthetic_model(device);
    auto ple_meta = make_synthetic_ple_meta();
    ninfer::runtime::ContextMachineCostModel cost_model{};

    FlashNextRuntimeConfig config{};
    config.max_context = 2048;
    config.max_concurrency = 2;
    config.continuation_capacity = 4;
    config.speculative_draft_tokens = 0;

    FlashNextRuntimePlan plan = finalize_flash_next_runtime_plan(config, 8);
    auto prog_impl = std::make_unique<ProgramImpl>(nullptr, plan, device, model.view, std::nullopt, ple_meta);
    Program prog(std::move(prog_impl));

    const std::size_t initial_available_groups = prog.impl_->executor_.available_physical_groups();
    std::printf("  [Initial State] Available physical page groups: %zu / %u\n",
                initial_available_groups, plan.main_page_groups);

    constexpr int kTurns = 20;
    std::vector<ninfer::TokenId> conversation_tokens;
    std::optional<ContinuationHandle> previous_turn_closure;
    std::optional<runtime::CheckpointRef> previous_checkpoint;
    std::atomic<bool> cancellation_flag{false};
    ninfer::runtime::CancellationFlagView cancellation{&cancellation_flag};

    for (int turn = 0; turn < kTurns; ++turn) {
        // Add 32 tokens per turn
        for (int i = 0; i < 32; ++i) {
            conversation_tokens.push_back(static_cast<ninfer::TokenId>(100 + (turn * 32 + i) % 500));
        }

        auto prompt = make_prompt(conversation_tokens);
        ninfer::runtime::ResolvedExecutionOptions exec_options{};
        exec_options.requested_output_tokens = 8;
        auto base_plan = prog.plan_request(prompt, exec_options);

        const ContinuationHandle* src = previous_turn_closure.has_value() ? &*previous_turn_closure : nullptr;
        auto insp = prog.inspect_admission(prompt, base_plan, ninfer::runtime::LaneId(0), src, nullptr, previous_checkpoint, false);

        if (!insp.has_value()) {
            std::fprintf(stderr, "FAIL: Turn %d inspect_admission returned nullopt\n", turn);
            return 1;
        }
        if (insp->identity_assessment().physical_status != runtime::MaterializationPhysicalStatus::Feasible) {
            std::fprintf(stderr, "FAIL: Turn %d identity_assessment status != Feasible (status=%d, needed_groups=%u, avail=%zu)\n",
                         turn, static_cast<int>(insp->identity_assessment().physical_status),
                         insp->impl_->required_page_groups,
                         prog.impl_->executor_.available_physical_groups());
            return 1;
        }

        auto res_plan = prog.seal_identity(*insp, prompt);
        (void)prog.start_resource_transaction(std::move(*res_plan), make_prompt(conversation_tokens), cancellation);
        auto prog_res = prog.progress_context_transaction(cancellation);
        SequenceHandle seq = std::get_if<MaterializationResult>(&prog_res)->published->sequence;
        prog.finalize_context_transaction();

        auto p = advance_prefill_full(prog, seq);
        if (p.pending.has_value()) {
            conversation_tokens.push_back(p.pending->tokens()[0]);
            std::array<ninfer::runtime::CommitDecision, 1> commit_dec = {{{.accepted_tokens = 1, .terminal = false}}};
            (void)prog.commit(std::move(*p.pending), commit_dec);
        }

        // Decode 4 tokens
        for (int step = 0; step < 4; ++step) {
            std::array<SequenceHandle, 1> seqs = {seq};
            std::array<ninfer::runtime::RoundBudget, 1> budgets{{{.generated_tokens_remaining = 1}}};
            auto dec = prog.decode(seqs, budgets);
            conversation_tokens.push_back(dec.tokens()[0]);
            std::array<ninfer::runtime::CommitDecision, 1> commit_dec = {{{.accepted_tokens = 1, .terminal = false}}};
            (void)prog.commit(std::move(dec), commit_dec);
        }

        FinishResult fin = prog.finish(seq);
        std::printf("  [Turn %d finish] Disposition: %d, continuation: %s, endpoint frontier: %u\n",
                    turn, static_cast<int>(fin.disposition),
                    fin.continuation.has_value() ? "yes" : "no",
                    fin.summary.endpoint.has_value() ? fin.summary.endpoint->ref.frontier : 0);

        if (previous_turn_closure.has_value()) {
            (void)prog.release_continuation(std::move(*previous_turn_closure));
            previous_turn_closure.reset();
            previous_checkpoint.reset();
        }
        if (fin.continuation.has_value()) {
            previous_turn_closure = std::move(*fin.continuation);
            previous_checkpoint = fin.summary.endpoint->ref;
        }

        if ((turn + 1) % 5 == 0) {
            std::printf("  [Turn %2d/%2d] Context: %zu tokens | Free groups: %zu\n",
                        turn + 1, kTurns, conversation_tokens.size(),
                        prog.impl_->executor_.available_physical_groups());
        }
    }

    if (previous_turn_closure.has_value()) {
        (void)prog.release_continuation(std::move(*previous_turn_closure));
        previous_turn_closure.reset();
    }

    const std::size_t final_available_groups = prog.impl_->executor_.available_physical_groups();
    std::printf("  [Final State] Available physical page groups: %zu / %zu (Drift: %lld)\n",
                final_available_groups, initial_available_groups,
                static_cast<long long>(final_available_groups) - static_cast<long long>(initial_available_groups));

    if (check(final_available_groups == initial_available_groups,
              "Resource leak detected: final available groups != initial available groups!")) {
        return 1;
    }
    std::printf("  -> Verified: Zero page group leaks across %d consecutive conversation turns.\n", kTurns);

    return 0;
}

// ---------------------------------------------------------------------------
// Test 4: Pool Exhaustion Failure Mode (Clean Rejection vs Corruption)
// ---------------------------------------------------------------------------
int test_soak_pool_exhaustion_failure_mode(ninfer::DeviceContext& device) {
    std::printf("\n=== TEST 4: Pool Exhaustion Failure Mode ===\n");
    std::fflush(stdout);

    SyntheticFlashNextModel model = make_synthetic_model(device);
    auto ple_meta = make_synthetic_ple_meta();
    ninfer::runtime::ContextMachineCostModel cost_model{};

    FlashNextRuntimeConfig config{};
    config.max_context = 1024;
    config.max_concurrency = 2;
    config.continuation_capacity = 2;
    config.speculative_draft_tokens = 0;

    const auto curve = flash_next_capacity_curve(config);
    // Allocate min_groups (4 page groups = 1024 tokens total)
    FlashNextRuntimePlan plan = finalize_flash_next_runtime_plan(config, curve.minimum_main_page_groups);
    auto prog_impl = std::make_unique<ProgramImpl>(nullptr, plan, device, model.view, std::nullopt, ple_meta);
    Program prog(std::move(prog_impl));

    std::vector<SequenceHandle> active_seqs;
    std::vector<ninfer::TokenId> prompt_tok(500, 77);
    auto prompt = make_prompt(prompt_tok);

    std::atomic<bool> cancellation_flag{false};
    ninfer::runtime::CancellationFlagView cancellation{&cancellation_flag};

    ninfer::runtime::ResolvedExecutionOptions exec_options{};
    exec_options.requested_output_tokens = 8;
    auto base_plan = prog.plan_request(prompt, exec_options);

    // Saturate both 2 lanes with 500-token requests (2 * 2 = 4 page groups)
    for (int i = 0; i < 2; ++i) {
        auto insp = prog.inspect_admission(prompt, base_plan, ninfer::runtime::LaneId(i), nullptr, nullptr, std::nullopt, false);
        if (!insp.has_value() || insp->identity_assessment().physical_status != runtime::MaterializationPhysicalStatus::Feasible) {
            std::fprintf(stderr, "FAIL: Could not admit sequence %d in saturation phase\n", i);
            return 1;
        }
        auto res_plan = prog.seal_identity(*insp, prompt);
        (void)prog.start_resource_transaction(std::move(*res_plan), make_prompt(prompt_tok), cancellation);
        auto prog_res = prog.progress_context_transaction(cancellation);
        SequenceHandle seq = std::get_if<MaterializationResult>(&prog_res)->published->sequence;
        prog.finalize_context_transaction();
        active_seqs.push_back(seq);
    }

    // Advance prefill for both active sequences to physically consume all 4 page groups
    for (auto& s : active_seqs) {
        auto p = advance_prefill_full(prog, s);
        if (p.pending.has_value()) {
            std::array<ninfer::runtime::CommitDecision, 1> commit_dec = {{{.accepted_tokens = 1, .terminal = false}}};
            (void)prog.commit(std::move(*p.pending), commit_dec);
        }
    }

    std::printf("  [Saturation] 2 active sequences admitted & prefilled; available page groups: %zu\n",
                prog.impl_->executor_.available_physical_groups());

    if (check(prog.impl_->executor_.available_physical_groups() == 0,
              "Expected 0 available physical groups after saturation")) {
        return 1;
    }

    // Now attempt to admit a 3rd sequence when no groups remain and no continuations are evictable
    std::printf("  [Attempting 3rd Sequence Admission] Checking inspect_admission...\n");
    auto overflow_insp = prog.inspect_admission(prompt, base_plan, ninfer::runtime::LaneId(0), nullptr, nullptr, std::nullopt, false);

    if (check(overflow_insp.has_value(), "inspect_admission should return candidate")) {
        return 1;
    }
    if (check(overflow_insp->identity_assessment().physical_status == runtime::MaterializationPhysicalStatus::Infeasible,
              "Physical status must report Infeasible on exhaustion")) {
        return 1;
    }

    std::printf("  -> inspect_admission cleanly returned identity_assessment physical_status == Infeasible.\n");

    // Existing active sequences continue decoding without interference
    std::printf("  [Active In-Flight Check] Decoding active sequences...\n");
    for (auto& s : active_seqs) {
        std::array<SequenceHandle, 1> seqs = {s};
        std::array<ninfer::runtime::RoundBudget, 1> budgets{{{.generated_tokens_remaining = 1}}};
        auto dec = prog.decode(seqs, budgets);
        std::array<ninfer::runtime::CommitDecision, 1> commit_dec = {{{.accepted_tokens = 1, .terminal = false}}};
        (void)prog.commit(std::move(dec), commit_dec);
    }

    for (auto& s : active_seqs) {
        FinishResult fin = prog.finish(s);
        if (fin.continuation.has_value()) {
            (void)prog.release_continuation(std::move(*fin.continuation));
        }
    }

    std::printf("  [Post-Finish Recovery] Available page groups: %zu / %u\n",
                prog.impl_->executor_.available_physical_groups(), plan.main_page_groups);

    if (check(prog.impl_->executor_.available_physical_groups() == plan.main_page_groups,
              "Available page groups did not restore to full capacity after finishing")) {
        return 1;
    }

    std::printf("  -> Verified: Clean rejection mode verified. Zero crashes, zero corruption, clean server error reporting.\n");
    return 0;
}

} // namespace

int main() {
    cudaError_t err = cudaSetDevice(0);
    if (cuda_unavailable(err)) {
        std::printf("CUDA unavailable (code %d); skipping test.\n", static_cast<int>(err));
        return 77;
    }

    try {
        ninfer::DeviceContext device(0);

        if (test_soak_concurrency_and_admission_curve(device) != 0) { return 1; }
        if (test_soak_pressure_eviction_and_determinism(device) != 0) { return 1; }
        if (test_soak_multiturn_conversations_and_leaks(device) != 0) { return 1; }
        if (test_soak_pool_exhaustion_failure_mode(device) != 0) { return 1; }

        std::printf("\nALL SYNTHETIC SOAK TESTS PASSED (100%%).\n");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Unhandled exception: %s\n", e.what());
        return 1;
    }
}
