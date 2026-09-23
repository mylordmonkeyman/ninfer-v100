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

double execute_prefill_and_capture(FlashNextResourceManager& manager, Program& prog, AdmittedRequest& req,
                                  std::vector<TokenId>& generated) {
    std::atomic<bool> cancellation_flag{false};
    ninfer::runtime::CancellationFlagView cancellation{&cancellation_flag};

    auto p = prog.advance_prefill(req.sequence);
    double prefill_seconds = static_cast<double>(p.timing.elapsed_ns()) * 1.0e-9;
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
        prefill_seconds += static_cast<double>(p.timing.elapsed_ns()) * 1.0e-9;
    }
    if (p.pending.has_value()) {
        generated.push_back(p.pending->tokens()[0]);
        std::array<ninfer::runtime::CommitDecision, 1> commit_dec = {{{.accepted_tokens = 1, .terminal = false}}};
        (void)prog.commit(std::move(*p.pending), commit_dec);
    }
    return prefill_seconds;
}

ninfer::GenerationTimings decode_and_finish(FlashNextResourceManager& manager, Program& prog,
                                           AdmittedRequest& req, int num_decode_tokens,
                                           std::vector<TokenId>& generated) {
    for (int step = 0; step < num_decode_tokens; ++step) {
        std::array<SequenceHandle, 1> seqs = {req.sequence};
        const std::array<ninfer::runtime::RoundBudget, 1> budgets{{
            {.generated_tokens_remaining = static_cast<std::uint32_t>(num_decode_tokens - step)}}};
        auto dec = prog.decode(seqs, budgets);
        generated.push_back(dec.tokens()[0]);
        std::array<ninfer::runtime::CommitDecision, 1> commit_dec = {{{.accepted_tokens = 1, .terminal = false}}};
        (void)prog.commit(std::move(dec), commit_dec);
    }
    manager.mark_terminal_pending(req.lane);
    auto fin = manager.finish(prog, req.lane, req.sequence);
    if (fin.status != ninfer::runtime::ConsumeStatus::Consumed) {
        throw std::runtime_error("finish failed to consume sequence");
    }
    return fin.timings;
}

} // namespace

int main() {
    try {
        ninfer::DeviceContext device(0);
    SyntheticFlashNextModel model = make_synthetic_model(device);
    PleIndexMetadata ple_meta = make_synthetic_ple_meta();

    FlashNextRuntimeConfig config{};
    config.max_context = 2048;
    config.max_concurrency = 4;
    config.prefill_chunk = 256;
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
        ninfer::runtime::generic_context_machine_cost_model());

    std::printf("=== Multi-Turn Prefix Reuse Test (8 Sequential Turns) ===\n");
    std::fflush(stdout);

    std::uint64_t pub_order = 1;

    // Fixed 256-token preamble shared across all turns
    std::vector<TokenId> preamble(256);
    for (std::size_t i = 0; i < 256; ++i) {
        preamble[i] = static_cast<TokenId>(1000 + (i % 200));
    }

    std::vector<TokenId> conversation_history = preamble;
    std::uint32_t expected_reuse = 0;

    for (int turn = 1; turn <= 8; ++turn) {
        const std::uint32_t turn_closure_frontier = static_cast<std::uint32_t>(conversation_history.size());

        // Append user turn: 64 tokens
        for (std::size_t i = 0; i < 64; ++i) {
            conversation_history.push_back(static_cast<TokenId>(2000 + turn * 100 + i));
        }

        auto prompt = make_prompt(conversation_history, turn_closure_frontier);

        ninfer::runtime::ResolvedExecutionOptions exec_options{};
        exec_options.requested_output_tokens = 4;
        exec_options.allow_prefix_reuse = true;
        auto base_plan = prog.plan_request(prompt, exec_options);

        auto inspection = manager.inspect(prog, prompt, base_plan, pub_order);
        if (!inspection.choice.has_value()) {
            std::fprintf(stderr, "FAIL: Turn %d inspect produced no choice\n", turn);
            return 1;
        }

        const auto& sum = inspection.choice->summary();
        const std::uint32_t cached = sum.reusable_prompt_tokens;
        const auto path = sum.prefix_reuse_path;

        std::printf("[Turn %d] prompt_tokens=%zu, cached=%u, expected=%u, path=%d\n",
                    turn, conversation_history.size(), cached, expected_reuse, static_cast<int>(path));
        std::fflush(stdout);

        if (turn == 1) {
            if (cached != 0) {
                std::fprintf(stderr, "FAIL: Turn 1 should have 0 cached tokens\n");
                return 1;
            }
        } else {
            if (cached == 0) {
                std::fprintf(stderr, "FAIL: Turn %d had 0 cached tokens! Cache miss!\n", turn);
                return 2;
            }
            if (cached < expected_reuse) {
                std::fprintf(stderr, "FAIL: Turn %d cached (%u) < expected (%u)\n", turn, cached, expected_reuse);
                return 1;
            }
        }

        AdmittedRequest req = admit_request(manager, prog, std::move(prompt), pub_order++);
        std::vector<TokenId> generated;
        const double prefill_seconds = execute_prefill_and_capture(manager, prog, req, generated);
        const auto timings = decode_and_finish(manager, prog, req, 4, generated);
        // Compare public completed-request timing with the execution-unit observations. This
        // covers chunk/capture boundaries and verifies that each reused lane starts at zero;
        // time spent by ResourceManager between units must not enter the prefill denominator.
        if (prefill_seconds <= 0.0 || timings.vision_seconds != 0.0 ||
            std::abs(timings.prefill_seconds - prefill_seconds) > 1.0e-9) {
            throw std::runtime_error("prefill timing lost chunks, included capture gaps, or leaked across requests");
        }

        // Append actual generated tokens to conversation history for next turn
        conversation_history.insert(conversation_history.end(), generated.begin(), generated.end());
        expected_reuse = turn_closure_frontier;
    }

    // Cancellation after a completed chunk preserves its measured work.
    {
        const std::vector<TokenId> tokens(600, 1700);
        auto req = admit_request(manager, prog, make_prompt(tokens), pub_order++);
        const auto progress = prog.advance_prefill(req.sequence);
        if (progress.complete || progress.processed_prompt_tokens != config.prefill_chunk) {
            throw std::runtime_error("partial-prefill cancellation fixture did not yield after one chunk");
        }
        const auto aborted = manager.abort(prog, req.lane, req.sequence);
        const double elapsed = static_cast<double>(progress.timing.elapsed_ns()) * 1.0e-9;
        if (elapsed <= 0.0 || std::abs(aborted.timings.prefill_seconds - elapsed) > 1.0e-9 ||
            aborted.timings.vision_seconds != 0.0) {
            throw std::runtime_error("aborted request lost its completed prefill timing");
        }
    }
    // A lane cancelled before execution cannot inherit the preceding request's counters.
    {
        const std::vector<TokenId> tokens(16, 1800);
        auto req = admit_request(manager, prog, make_prompt(tokens), pub_order++);
        const auto aborted = manager.abort(prog, req.lane, req.sequence);
        if (aborted.timings.prefill_seconds != 0.0 || aborted.timings.vision_seconds != 0.0) {
            throw std::runtime_error("lane admission did not reset request timing");
        }
    }
    // Engine consumes the CommitRowResult directly when cancellation releases a pending Begin.
    {
        const std::vector<TokenId> tokens(16, 1900);
        auto req = admit_request(manager, prog, make_prompt(tokens), pub_order++);
        auto progress = prog.advance_prefill(req.sequence);
        if (!progress.complete || !progress.pending) {
            throw std::runtime_error("pending-Begin cancellation fixture did not complete prefill");
        }
        const std::array<ninfer::runtime::CommitDecision, 1> decisions{{
            {.accepted_tokens = 0, .terminal = true}}};
        const auto committed = prog.commit(std::move(*progress.pending), decisions);
        const std::array lanes{req.lane};
        manager.apply_commit(lanes, committed);
        const double elapsed = static_cast<double>(progress.timing.elapsed_ns()) * 1.0e-9;
        if (committed.rows[0].disposition != ninfer::runtime::CommitDisposition::CancelledReleased ||
            elapsed <= 0.0 || std::abs(committed.rows[0].timings.prefill_seconds - elapsed) > 1.0e-9) {
            throw std::runtime_error("cancelled commit lost its completed prefill timing");
        }
    }

    std::printf("-> ALL 8 TURNS PASSED: Prefix reuse and request-owned timing remain isolated.\n");
    return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL EXCEPTION: %s\n", e.what());
        std::fflush(stderr);
        return 1;
    }
}
