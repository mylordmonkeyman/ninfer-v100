#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/schedule.h"

#include "ninfer/ops/sampling.h"
#include "ninfer/ops/scatter.h"
#include "ninfer/ops/prepare_ragged_prefix.h"

#include <stdexcept>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {
namespace {

auto ordinary_batch_body(OrdinaryBatchContext& state, std::int32_t batch_size,
                         ops::CausalAttentionExecutionEnvelope envelope) {
    return [&state, batch_size, envelope] {
        if (batch_size <= 0 || batch_size > static_cast<std::int32_t>(kMaximumConcurrency)) {
            throw std::logic_error("ordinary decode batch state is incomplete");
        }

        qwen3_6::OrdinaryDecodeState& ordinary = state.frame;
        CUDA_CHECK(cudaMemcpyAsync(ordinary.ingress.data, &state.host_ingress,
                                   sizeof(qwen3_6::OrdinaryDecodeIngress), cudaMemcpyHostToDevice,
                                   state.execution.device.stream));

        TextContext card(state.execution.device, state.execution.model, state.execution.work, {},
                         state.execution.linear_attention, state.execution.io,
                         state.execution.prefill_hidden, state.execution.prefill_chunk, 0, {},
                         &state.text_cache, state.mtp_cache);

        Tensor tokens             = ordinary.tokens.slice(0, 0, batch_size);
        Tensor cache_positions    = ordinary.cache_positions.slice(0, 0, batch_size);
        Tensor next_positions     = ordinary.next_cache_positions.slice(0, 0, batch_size);
        Tensor rope_positions     = ordinary.rope_positions.slice(0, 0, batch_size);
        Tensor kv_rows            = ordinary.text_kv_table_rows.slice(0, 0, batch_size);
        Tensor backend_rows       = ordinary.backend_kv_table_rows.slice(0, 0, batch_size);
        Tensor active_lanes       = ordinary.active_lanes.slice(0, 0, batch_size);
        Tensor valid_columns      = ordinary.valid_columns.slice(0, 0, batch_size);
        Tensor state_sources      = ordinary.state_source_slots.slice(0, 0, batch_size);
        Tensor state_destinations = ordinary.state_destination_slots.slice(0, 0, batch_size);
        Tensor hidden             = ordinary.hidden.slice(1, 0, batch_size);
        Tensor logits             = ordinary.logits.slice(1, 0, batch_size);
        Tensor sampled            = ordinary.sampled_tokens.slice(0, 0, batch_size);

        if (state.dflash != nullptr) {
            DFlashFeatureSink sink{
                .batch_features      = &state.dflash->pending_features,
                .batch_lanes         = &active_lanes,
                .batch_valid_columns = &valid_columns,
                .batch_width         = 1,
                .batch_size          = batch_size,
                .layers              = std::span<const int>(DFlashConfig::target_feature_layers),
            };
            card.ordinary_decode_batch(tokens, cache_positions, rope_positions, kv_rows,
                                       state_sources, state_destinations, envelope, hidden, logits,
                                       sink);
        } else {
            card.ordinary_decode_batch(tokens, cache_positions, rope_positions, kv_rows,
                                       state_sources, state_destinations, envelope, hidden, logits);
        }
        ops::scatter(hidden, state_destinations, state.continuation_hidden_store,
                     state.execution.device.stream);
        ops::sample(logits, sampled, TextConfig::token_domain, ordinary.sampling, cache_positions,
                    ops::kSamplePurposeDecode, state.execution.work, state.execution.device.stream);
        // Width-one target rounds still maintain the backend's committed prefix. No proposal
        // head, draft chain, sparse acceptance, or padded target verification is needed.
        state.execution.work.reset();
        if (state.mtp_cache != nullptr) {
            Tensor aligned = state.execution.io.mtp_decode->ar_hidden.slice(1, 0, batch_size)
                                 .view({TextConfig::hidden, 1, batch_size});
            card.mtp_forward_decode_batch(sampled.view({1, batch_size}),
                                           hidden.view({TextConfig::hidden, 1, batch_size}),
                                           cache_positions.view({1, batch_size}),
                                           rope_positions.view({1, batch_size}), valid_columns,
                                           backend_rows, envelope, aligned);
        } else if (state.dflash != nullptr) {
            Tensor features = state.execution.work.alloc(
                DType::BF16, {DFlashConfig::feature_rows, 1, batch_size});
            Tensor positions = state.execution.work.alloc(DType::I32, {1, batch_size});
            Tensor counts = state.execution.work.alloc(DType::I32, {batch_size});
            ops::prepare_ragged_prefix(state.dflash->pending_features.slice(1, 0, 1), active_lanes,
                                       cache_positions, next_positions, features, positions, counts,
                                       state.execution.device.stream);
            DFlashAppendContext context{state.execution, *state.dflash};
            dflash_append_context(context, features, positions, counts, state_destinations,
                                   backend_rows, {1, 1});
        }
        state.execution.work.reset();
        CUDA_CHECK(cudaMemcpyAsync(&state.host_egress, ordinary.egress.data,
                                   sizeof(qwen3_6::OrdinaryDecodeEgress), cudaMemcpyDeviceToHost,
                                   state.execution.device.stream));
    };
}

} // namespace

void capture_ordinary_decode_batch(OrdinaryBatchContext& state, std::int32_t batch_size,
                                   ops::CausalAttentionExecutionEnvelope envelope,
                                   DecodeGraphDefinition& definition) {
    auto body = ordinary_batch_body(state, batch_size, envelope);
    capture_graph(state, definition, body);
}

void ordinary_decode_batch(OrdinaryBatchContext& state, std::int32_t batch_size,
                           ops::CausalAttentionExecutionEnvelope envelope,
                           DecodeGraphExecutable* executable) {
    auto body = ordinary_batch_body(state, batch_size, envelope);
    run_prepared(state, executable, body);
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
