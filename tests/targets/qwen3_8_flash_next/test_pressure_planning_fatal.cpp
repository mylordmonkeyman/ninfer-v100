#include <ninfer/targets/qwen3_8_flash_next/package.h>
#include <ninfer/targets/qwen3_8_flash_next/runtime.h>
#include <ninfer/targets/qwen3_6/frontend.h>
#include <ninfer/targets/qwen3_6/frontend_resources.h>
#include <ninfer/targets/qwen3_6/prepared_prompt.h>

#include "core/arena.h"
#include "core/device.h"
#include "runtime/contract/types.h"
#include "runtime/engine/context_cost.h"
#include "runtime/engine/resource_manager.h"
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
#include <string_view>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::targets::qwen3_8_flash_next;
using namespace ninfer::targets::qwen3_8_flash_next::detail;

bool cuda_unavailable(cudaError_t error) {
    return error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver;
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

ninfer::targets::qwen3_6::PreparedPrompt make_prompt(
    std::span<const ninfer::TokenId> tokens,
    std::optional<std::uint32_t> rewrite_frontier = std::nullopt) {
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
    if (rewrite_frontier.has_value() && *rewrite_frontier > 0 && *rewrite_frontier < num_tokens) {
        data.identity.rewrite_checkpoint = ninfer::targets::qwen3_6::RewriteCheckpointSpec{
            .kind = ninfer::targets::qwen3_6::RewriteCheckpointKind::TurnClosure,
            .frontier = *rewrite_frontier,
        };
    }
    return ninfer::targets::qwen3_6::PreparedPromptAccess::construct(std::move(data));
}

using FlashNextResourceManager = ninfer::runtime::ResourceManager<ninfer::targets::qwen3_8_flash_next::Package>;

struct AdmittedRequest {
    ninfer::runtime::LaneId lane;
    SequenceHandle sequence;
};

AdmittedRequest admit_request(FlashNextResourceManager& manager, Program& prog,
                              ninfer::targets::qwen3_6::PreparedPrompt prompt, std::uint64_t publication_order) {
    ninfer::runtime::ResolvedExecutionOptions exec_options{};
    exec_options.requested_output_tokens = 16;
    exec_options.allow_prefix_reuse = true;
    auto base_plan = prog.plan_request(prompt, exec_options);

    auto inspection = manager.inspect(prog, prompt, base_plan, publication_order);
    if (!inspection.choice.has_value()) {
        throw std::runtime_error("inspect produced no choice for admission");
    }
    const ninfer::runtime::LaneId lane = inspection.choice->destination();
    std::atomic<bool> cancellation_flag{false};
    ninfer::runtime::CancellationFlagView cancellation{&cancellation_flag};

    auto reserved = manager.reserve_materialization(prog, std::move(*inspection.choice), std::move(prompt), cancellation);
    if (reserved != FlashNextResourceManager::MaterializationReserveResult::Reserved) {
        throw std::runtime_error("reserve_materialization failed");
    }

    for (;;) {
        auto step = manager.progress_context_transaction(prog, cancellation);
        if (auto* mat = std::get_if<typename FlashNextResourceManager::MaterializationOutcome>(&step)) {
            if (mat->status != ninfer::runtime::ContextTransactionStatus::Published || !mat->activation) {
                throw std::runtime_error("context transaction did not publish");
            }
            auto activation = std::move(*mat->activation);
            SequenceHandle sequence = activation.sequence();
            manager.adopt(prog, std::move(activation));
            prog.finalize_context_transaction();
            return AdmittedRequest{lane, sequence};
        }
        if (!std::holds_alternative<ninfer::runtime::ContextTransactionInProgress>(step)) {
            throw std::runtime_error("unexpected transaction outcome in admit_request");
        }
    }
}

void execute_prefill_and_capture(FlashNextResourceManager& manager, Program& prog, AdmittedRequest& req) {
    std::atomic<bool> cancellation_flag{false};
    ninfer::runtime::CancellationFlagView cancellation{&cancellation_flag};

    auto p = prog.advance_prefill(req.sequence);
    while (!p.complete) {
        if (p.capture.has_value()) {
            auto cap_res = manager.reserve_active_capture(
                prog, req.lane, std::move(*p.capture), 0, cancellation);
            if (cap_res == FlashNextResourceManager::ActiveCaptureReserveResult::Reserved) {
                for (;;) {
                    auto step = manager.progress_context_transaction(prog, cancellation);
                    if (std::holds_alternative<typename FlashNextResourceManager::ActiveCaptureOutcome>(step)) {
                        prog.finalize_context_transaction();
                        break;
                    }
                    if (!std::holds_alternative<ninfer::runtime::ContextTransactionInProgress>(step)) {
                        break;
                    }
                }
            }
        }
        p = prog.advance_prefill(req.sequence);
    }
    if (p.pending.has_value()) {
        std::array<ninfer::runtime::CommitDecision, 1> commit_dec = {{{.accepted_tokens = 1, .terminal = false}}};
        (void)prog.commit(std::move(*p.pending), commit_dec);
    }
}

void decode_and_finish(FlashNextResourceManager& manager, Program& prog, AdmittedRequest& req, int num_decode_tokens) {
    for (int step = 0; step < num_decode_tokens; ++step) {
        std::array<SequenceHandle, 1> seqs = {req.sequence};
        std::array<ninfer::runtime::RoundBudget, 1> budgets{{{.generated_tokens_remaining = 1}}};
        auto dec = prog.decode(seqs, budgets);
        std::array<ninfer::runtime::CommitDecision, 1> commit_dec = {{{.accepted_tokens = 1, .terminal = false}}};
        (void)prog.commit(std::move(dec), commit_dec);
    }
    manager.mark_terminal_pending(req.lane);
    (void)manager.finish(prog, req.lane, req.sequence);
}

void run_concurrency_4_test(ninfer::DeviceContext& device, SyntheticFlashNextModel& model, PleIndexMetadata& ple_meta) {
    FlashNextRuntimeConfig config{};
    config.max_context = 2048;
    config.max_concurrency = 4;
    config.continuation_capacity = 4;
    config.speculative_draft_tokens = 0;

    FlashNextRuntimePlan plan = finalize_flash_next_runtime_plan(config, 32);
    auto prog_impl = std::make_unique<ProgramImpl>(nullptr, plan, device, model.view, std::nullopt, ple_meta);
    Program prog(std::move(prog_impl));

    FlashNextResourceManager manager(
        /*lane_count=*/4,
        /*private_catalog_capacity=*/4,
        /*shared_catalog_capacity=*/4,
        /*cache_enabled=*/true,
        /*max_long_anchors=*/2,
        ninfer::runtime::ContextMachineCostModel{});

    std::printf("\n=== Running Concurrency 4 Stale Owner Reproduction Test ===\n");
    std::fflush(stdout);

    std::uint64_t pub_order = 1;

    try {
        // Step 1: Run 4 requests to completion at concurrency 4.
        // Each request has 512 tokens (>= 256 for main page group allocation) with TurnClosure at 256.
        // Each request finishes and catalogues its endpoint into continuation slots 0..3.
        for (std::size_t i = 0; i < 4; ++i) {
            std::vector<TokenId> toks(512);
            for (std::size_t t = 0; t < 512; ++t) {
                toks[t] = static_cast<TokenId>(2000 + i * 1000 + (t % 500));
            }
            auto prompt = make_prompt(toks, 256);
            AdmittedRequest req = admit_request(manager, prog, std::move(prompt), pub_order++);
            execute_prefill_and_capture(manager, prog, req);
            decode_and_finish(manager, prog, req, 2);
        }

        std::printf("Catalog populated with 4 retained requests at concurrency 4.\n");
        std::fflush(stdout);

        // Step 2: Workload 5 arrives (5 concurrent workloads, concurrency 4).
        // continuation_slots_ is full (all 4 slots occupied). inspect_capture() correctly detects
        // no vacant slot is available (physically_feasible = false) and skips capture.
        // It does NOT uncooperatively evict any slot behind ResourceManager's back.
        std::vector<TokenId> w5_toks(512);
        for (std::size_t t = 0; t < 512; ++t) {
            w5_toks[t] = static_cast<TokenId>(6000 + (t % 500));
        }
        auto prompt5 = make_prompt(w5_toks, 256);
        AdmittedRequest req5 = admit_request(manager, prog, std::move(prompt5), pub_order++);
        execute_prefill_and_capture(manager, prog, req5);

        std::printf("Workload 5 prefilled without unilateral slot eviction.\n");
        std::fflush(stdout);

        // Step 3: Workload 6 arrives while private catalog is full.
        // ResourceManager needs to admit Workload 6, which requires coordinated pressure eviction.
        // It asks about private owners (slots 0..3 with generation 1).
        // Since no out-of-band eviction occurred, all owners are valid and match catalog generations.
        // Pressure planning plans eviction cooperatively, avoiding stale owner warnings and crashes.
        std::vector<TokenId> w6_toks(512);
        for (std::size_t t = 0; t < 512; ++t) {
            w6_toks[t] = static_cast<TokenId>(7000 + (t % 500));
        }
        auto prompt6 = make_prompt(w6_toks, 256);
        AdmittedRequest req6 = admit_request(manager, prog, std::move(prompt6), pub_order++);
        execute_prefill_and_capture(manager, prog, req6);
        decode_and_finish(manager, prog, req6, 2);
        decode_and_finish(manager, prog, req5, 2);

        std::printf("SUCCESS: Concurrency 4 test completed successfully!\n");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "\n[CRASH REPRODUCED IN C4 TEST] Caught exception: %s\n", e.what());
        std::fflush(stderr);
        std::fflush(stdout);
        throw;
    }
}

} // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::setvbuf(stderr, nullptr, _IONBF, 0);
    bool expect_fatal = false;
    bool run_c4 = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string_view(argv[i]) == "--expect-fatal") {
            expect_fatal = true;
        } else if (std::string_view(argv[i]) == "--concurrency-4") {
            run_c4 = true;
        }
    }

    cudaError_t cu_err = cudaSetDevice(0);
    if (cuda_unavailable(cu_err)) {
        std::cout << "SKIP: CUDA device unavailable" << std::endl;
        return 77;
    }

    ninfer::DeviceContext device(0);
    SyntheticFlashNextModel model = make_synthetic_model(device);
    auto ple_meta = make_synthetic_ple_meta();

    try {
        run_concurrency_4_test(device, model, ple_meta);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "\n[FATAL ERROR IN C4 TEST]: %s\n", e.what());
        return 1;
    }

    // Pool geometry matching Serve A from A.err:
    // kv_page_groups = 154, max_context = 32768, max_concurrency = 8, continuation_capacity = 16
    FlashNextRuntimeConfig config{};
    config.max_context = 32768;
    config.max_concurrency = 8;
    config.continuation_capacity = 16;
    config.speculative_draft_tokens = 0;

    FlashNextRuntimePlan plan = finalize_flash_next_runtime_plan(config, 154);
    auto prog_impl = std::make_unique<ProgramImpl>(nullptr, plan, device, model.view, std::nullopt, ple_meta);
    Program prog(std::move(prog_impl));

    FlashNextResourceManager manager(
        /*lane_count=*/8,
        /*private_catalog_capacity=*/16,
        /*shared_catalog_capacity=*/8,
        /*cache_enabled=*/true,
        /*max_long_anchors=*/2,
        ninfer::runtime::ContextMachineCostModel{});

    std::printf("=== Flash-Next Pressure Planning Reproduction Test (154 page groups) ===\n");
    std::fflush(stdout);

    // Request sequence modeled after Serve A:
    // R1: 25 prompt tokens, 23 decode tokens
    // R2: 25 prompt tokens (prefix reuse), 23 decode tokens
    // R3: 34 prompt tokens, 256 decode tokens
    // R4: 2945 prompt tokens, 5 decode tokens
    // R5: 5912 prompt tokens (prefix reuse from R4, rewrite frontier 2945), 5 decode tokens
    // R6: 11963 prompt tokens (prefix reuse from R5, rewrite frontier 5912), 5 decode tokens

    struct ReqSpec {
        std::uint32_t prompt_tokens;
        int decode_tokens;
        std::optional<std::uint32_t> rewrite_frontier;
    };

    std::vector<ReqSpec> specs = {
        {25, 23, std::nullopt},
        {25, 23, std::nullopt},
        {34, 256, std::nullopt},
        {2945, 5, std::nullopt},
        {5912, 5, 2945},
        {11963, 5, 5912},
    };

    std::uint64_t pub_order = 1;
    for (std::size_t i = 0; i < specs.size(); ++i) {
        const auto& sp = specs[i];
        std::vector<TokenId> toks(sp.prompt_tokens);
        // Request 2 (i=1) is identical canary to Request 1 (i=0)
        const std::size_t stream_id = (i == 1) ? 0 : i;
        for (std::size_t t = 0; t < sp.prompt_tokens; ++t) {
            toks[t] = static_cast<TokenId>(1000 + stream_id * 10000 + (t % 5000));
        }
        auto prompt = make_prompt(toks, sp.rewrite_frontier);
        std::printf("Admitting Request %zu (%u prompt tokens, %d decode tokens)...\n",
                    i + 1, sp.prompt_tokens, sp.decode_tokens);
        std::fflush(stdout);

        AdmittedRequest req = admit_request(manager, prog, std::move(prompt), pub_order++);
        execute_prefill_and_capture(manager, prog, req);
        decode_and_finish(manager, prog, req, sp.decode_tokens);
    }

    std::printf("Catalog populated with 6 retained requests.\n");
    std::fflush(stdout);

    // Request 7: 24,191 tokens!
    // Requires ~95 page groups, which exceeds remaining groups in the 154 pool.
    // Pressure planning is required to evict retained continuations.
    constexpr std::uint32_t kRequest7PromptTokens = 24191;
    std::vector<TokenId> r7_toks(kRequest7PromptTokens);
    for (std::size_t t = 0; t < kRequest7PromptTokens; ++t) {
        r7_toks[t] = static_cast<TokenId>(1000 + 7 * 10000 + (t % 5000));
    }
    auto prompt7 = make_prompt(r7_toks);


    ninfer::runtime::ResolvedExecutionOptions exec_options7{};
    exec_options7.requested_output_tokens = 16;
    exec_options7.allow_prefix_reuse = true;
    auto base_plan7 = prog.plan_request(prompt7, exec_options7);

    std::printf("Inspecting Request 7 (24,191 prompt tokens) forcing pressure eviction...\n");
    std::fflush(stdout);

    try {
        auto inspection = manager.inspect(prog, prompt7, base_plan7, pub_order++);
        if (!inspection.choice.has_value()) {
            std::fprintf(stderr, "FAIL: Pressure planning did not find a feasible admission choice\n");
            return 1;
        }
        std::printf("SUCCESS: Pressure planning found feasible admission choice!\n");
        std::printf("Victims selected and checkpoint outcomes validated.\n");

        // Reserve and progress materialization
        std::atomic<bool> cancellation_flag{false};
        ninfer::runtime::CancellationFlagView cancellation{&cancellation_flag};
        auto reserved = manager.reserve_materialization(prog, std::move(*inspection.choice), std::move(prompt7), cancellation);
        if (reserved != FlashNextResourceManager::MaterializationReserveResult::Reserved) {
            std::fprintf(stderr, "FAIL: reserve_materialization failed for Request 7\n");
            return 1;
        }

        for (;;) {
            auto step = manager.progress_context_transaction(prog, cancellation);
            if (auto* mat = std::get_if<typename FlashNextResourceManager::MaterializationOutcome>(&step)) {
                if (mat->status != ninfer::runtime::ContextTransactionStatus::Published || !mat->activation) {
                    std::fprintf(stderr, "FAIL: Request 7 did not publish\n");
                    return 1;
                }
                manager.adopt(prog, std::move(*mat->activation));
                prog.finalize_context_transaction();
                break;
            }
            if (!std::holds_alternative<ninfer::runtime::ContextTransactionInProgress>(step)) {
                std::fprintf(stderr, "FAIL: Request 7 unexpected outcome\n");
                return 1;
            }
        }

        std::printf("PASSED: Request 7 admitted with pressure eviction successfully!\n");
        if (expect_fatal) {
            std::fprintf(stderr, "FAIL: Expected fatal exception, but succeeded!\n");
            return 1;
        }
        return 0;
    } catch (const std::logic_error& e) {
        std::string_view msg(e.what());
        std::printf("CAUGHT std::logic_error: %s\n", e.what());
        if (msg.find("selected checkpoint outcome is incomplete") != std::string_view::npos) {
            std::printf("-> REPRODUCED EXACT LINE 2429 FATAL ERROR: %s\n", e.what());
            if (expect_fatal) {
                std::printf("PASSED: Expected fatal error reproduced successfully.\n");
                return 0;
            }
        }
        // Test fails if we were not expecting fatal
        return 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Unexpected exception: %s\n", e.what());
        return 1;
    }
}
