#include "ops/linear_attention/gated_delta_net/launch.h"

#include "core/device.h"
#include "ops/linear_attention/gated_delta_net/recurrent.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace ninfer::ops::detail::gated_delta_net {
namespace {

static_assert(sizeof(GdnReplayFoldKernelRow) == 16);
static_assert(alignof(GdnReplayFoldKernelRow) == 16);
static_assert(sizeof(GdnReplayFoldKernelRows) == 128);
static_assert(alignof(GdnReplayFoldKernelRows) == 16);
static_assert(std::is_trivially_copyable_v<GdnReplayFoldKernelRows>);

template <bool NormalizeQK, typename StateReadT = float, typename StateWriteT = StateReadT>
void launch_recurrent_direct_fixed(const Tensor& q, const Tensor& k, const Tensor& v,
                                   const Tensor& g, const Tensor& beta, float scale,
                                   const Tensor& state_read, Tensor& state_write, Tensor& out,
                                   cudaStream_t stream) {
    const auto heads = head_map::of(q.ne[1], v.ne[1]);
    const dim3 grid(static_cast<unsigned>(v.ne[1]), 1, static_cast<unsigned>(kStateDim / kBlockDv));
    const dim3 block(kWarpSize, kNumWarps, 1);
    recurrent_bf16_direct_kernel<NormalizeQK, StateReadT, StateWriteT><<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data), static_cast<const __nv_bfloat16*>(k.data),
        static_cast<const __nv_bfloat16*>(v.data), static_cast<const float*>(g.data),
        static_cast<const float*>(beta.data), static_cast<const StateReadT*>(state_read.data),
        static_cast<StateWriteT*>(state_write.data), static_cast<__nv_bfloat16*>(out.data), q.ne[2],
        heads, scale);
    CUDA_CHECK(cudaGetLastError());
}

template <bool NormalizeInputs, typename InputT = __nv_bfloat16,
          typename StateT = float, typename OutputT = __nv_bfloat16>
void launch_recurrent_batch_update_fixed(const Tensor& q, const Tensor& k, const Tensor& v,
                                         const Tensor& g, const Tensor& beta, float scale,
                                         Tensor& ssm_states, const Tensor& source_state_slots,
                                         const Tensor& destination_state_slots, Tensor& out,
                                         cudaStream_t stream) {
    const auto heads = head_map::of(q.ne[1], v.ne[1]);
    const dim3 grid(static_cast<unsigned>(v.ne[1]), static_cast<unsigned>(q.ne[3]),
                    static_cast<unsigned>(kStateDim / kBlockDv));
    const dim3 block(kWarpSize, kNumWarps, 1);
    const std::int64_t state_slot_stride =
        static_cast<std::int64_t>(kStateDim) * kStateDim * ssm_states.ne[2];
    const BatchUpdateAccess<InputT, StateT, OutputT> access{
        static_cast<const InputT*>(q.data),
        static_cast<const InputT*>(k.data),
        static_cast<const InputT*>(v.data),
        static_cast<const float*>(g.data),
        static_cast<const float*>(beta.data),
        static_cast<const StateT*>(ssm_states.data),
        static_cast<StateT*>(ssm_states.data),
        static_cast<const std::int32_t*>(source_state_slots.data),
        static_cast<const std::int32_t*>(destination_state_slots.data),
        static_cast<OutputT*>(out.data),
        heads,
        state_slot_stride,
        scale,
    };
    recurrent_batch_update_kernel<NormalizeInputs, InputT, StateT, OutputT>
        <<<grid, block, 0, stream>>>(access);
    CUDA_CHECK(cudaGetLastError());
}

template <bool Masked>
void launch_recurrent_record_fixed(const Tensor& q, const Tensor& k, const Tensor& v,
                                   const Tensor& g, const Tensor& beta, float scale,
                                   const Tensor& ssm_states, const Tensor& valid_columns,
                                   const Tensor& initial_state_slots, Tensor& key_record,
                                   Tensor& value_record, Tensor& gate_record, Tensor& out,
                                   cudaStream_t stream) {
    const auto heads = head_map::of(q.ne[1], v.ne[1]);
    const dim3 grid(static_cast<unsigned>(v.ne[1]), static_cast<unsigned>(q.ne[3]),
                    static_cast<unsigned>(kStateDim / kBlockDv));
    const dim3 block(kWarpSize, kNumWarps, 1);
    const std::int64_t state_slot_stride =
        static_cast<std::int64_t>(kStateDim) * kStateDim * ssm_states.ne[2];
    const RecordAccess<Masked> access{
        static_cast<const __nv_bfloat16*>(q.data),
        static_cast<const __nv_bfloat16*>(k.data),
        static_cast<const __nv_bfloat16*>(v.data),
        static_cast<const float*>(g.data),
        static_cast<const float*>(beta.data),
        static_cast<const float*>(ssm_states.data),
        Masked ? static_cast<const std::int32_t*>(valid_columns.data) : nullptr,
        static_cast<const std::int32_t*>(initial_state_slots.data),
        static_cast<__nv_bfloat16*>(key_record.data),
        static_cast<__nv_bfloat16*>(value_record.data),
        reinterpret_cast<uint2*>(gate_record.data),
        static_cast<__nv_bfloat16*>(out.data),
        heads,
        q.ne[2],
        state_slot_stride,
        scale,
    };
    recurrent_record_kernel<Masked><<<grid, block, 0, stream>>>(access);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry>
void launch_replay_fold_fixed(const GdnReplayRecords& records,
                              LinearAttentionStateAllLayersView states,
                              const GdnReplayFoldKernelRows& rows, std::int32_t active_rows,
                              cudaStream_t stream) {
    const FoldAccess<Geometry> access{
        static_cast<const __nv_bfloat16*>(records.key.data),
        static_cast<const __nv_bfloat16*>(records.value.data),
        reinterpret_cast<const uint2*>(records.gate.data),
        static_cast<const __nv_bfloat16*>(records.conv.data),
        static_cast<float*>(states.recurrent_layer0.data),
        static_cast<__nv_bfloat16*>(states.conv_layer0.data),
        states.recurrent_layer_stride_bytes / static_cast<std::int64_t>(sizeof(float)),
        states.conv_layer_stride_bytes / static_cast<std::int64_t>(sizeof(__nv_bfloat16)),
        records.spec.record_capacity,
        records.spec.width,
        rows,
    };
    const dim3 grid(static_cast<unsigned>(Geometry::kValueHeads),
                    static_cast<unsigned>(active_rows),
                    static_cast<unsigned>(Geometry::kLayers * (kStateDim / kBlockDv)));
    const dim3 block(kWarpSize, kNumWarps, 1);
    recurrent_fold_kernel<Geometry><<<grid, block, 0, stream>>>(access);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void launch_recurrent(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                      const Tensor& beta, float scale, bool normalize_qk, Tensor& ssm_state,
                      Tensor& out, cudaStream_t stream) {
    if (ssm_state.dtype == DType::BF16) {
        if (normalize_qk) {
            launch_recurrent_direct_fixed<true, __nv_bfloat16, __nv_bfloat16>(
                q, k, v, g, beta, scale, ssm_state, ssm_state, out, stream);
        } else {
            launch_recurrent_direct_fixed<false, __nv_bfloat16, __nv_bfloat16>(
                q, k, v, g, beta, scale, ssm_state, ssm_state, out, stream);
        }
    } else if (ssm_state.dtype == DType::FP32) {
        if (normalize_qk) {
            launch_recurrent_direct_fixed<true, float, float>(q, k, v, g, beta, scale, ssm_state,
                                                              ssm_state, out, stream);
        } else {
            launch_recurrent_direct_fixed<false, float, float>(q, k, v, g, beta, scale, ssm_state,
                                                               ssm_state, out, stream);
        }
    } else {
        throw std::invalid_argument("GDN recurrent launcher unsupported state dtype");
    }
}

void launch_recurrent_inout(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                            const Tensor& beta, float scale, bool normalize_qk,
                            const Tensor& ssm_state_in, Tensor& ssm_state_out, Tensor& out,
                            cudaStream_t stream) {
    if (ssm_state_in.dtype == DType::BF16 && ssm_state_out.dtype == DType::BF16) {
        if (normalize_qk) {
            launch_recurrent_direct_fixed<true, __nv_bfloat16, __nv_bfloat16>(
                q, k, v, g, beta, scale, ssm_state_in, ssm_state_out, out, stream);
        } else {
            launch_recurrent_direct_fixed<false, __nv_bfloat16, __nv_bfloat16>(
                q, k, v, g, beta, scale, ssm_state_in, ssm_state_out, out, stream);
        }
    } else if (ssm_state_in.dtype == DType::FP32 && ssm_state_out.dtype == DType::FP32) {
        if (normalize_qk) {
            launch_recurrent_direct_fixed<true, float, float>(q, k, v, g, beta, scale, ssm_state_in,
                                                              ssm_state_out, out, stream);
        } else {
            launch_recurrent_direct_fixed<false, float, float>(q, k, v, g, beta, scale, ssm_state_in,
                                                               ssm_state_out, out, stream);
        }
    } else if (ssm_state_in.dtype == DType::FP32 && ssm_state_out.dtype == DType::BF16) {
        if (normalize_qk) {
            launch_recurrent_direct_fixed<true, float, __nv_bfloat16>(
                q, k, v, g, beta, scale, ssm_state_in, ssm_state_out, out, stream);
        } else {
            launch_recurrent_direct_fixed<false, float, __nv_bfloat16>(
                q, k, v, g, beta, scale, ssm_state_in, ssm_state_out, out, stream);
        }
    } else if (ssm_state_in.dtype == DType::BF16 && ssm_state_out.dtype == DType::FP32) {
        if (normalize_qk) {
            launch_recurrent_direct_fixed<true, __nv_bfloat16, float>(
                q, k, v, g, beta, scale, ssm_state_in, ssm_state_out, out, stream);
        } else {
            launch_recurrent_direct_fixed<false, __nv_bfloat16, float>(
                q, k, v, g, beta, scale, ssm_state_in, ssm_state_out, out, stream);
        }
    } else {
        throw std::invalid_argument("GDN recurrent launcher unsupported state dtypes");
    }
}

template <typename InputT, typename StateT>
void launch_recurrent_batch_update_for_state(
    const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
    const Tensor& beta, float scale, bool normalize_qk, Tensor& ssm_states,
    const Tensor& source_state_slots, const Tensor& destination_state_slots,
    Tensor& out, cudaStream_t stream) {
    if (out.dtype == DType::BF16) {
        if (normalize_qk) {
            launch_recurrent_batch_update_fixed<true, InputT, StateT, __nv_bfloat16>(
                q, k, v, g, beta, scale, ssm_states, source_state_slots,
                destination_state_slots, out, stream);
        } else {
            launch_recurrent_batch_update_fixed<false, InputT, StateT, __nv_bfloat16>(
                q, k, v, g, beta, scale, ssm_states, source_state_slots,
                destination_state_slots, out, stream);
        }
    } else if (out.dtype == DType::FP32) {
        if (normalize_qk) {
            launch_recurrent_batch_update_fixed<true, InputT, StateT, float>(
                q, k, v, g, beta, scale, ssm_states, source_state_slots,
                destination_state_slots, out, stream);
        } else {
            launch_recurrent_batch_update_fixed<false, InputT, StateT, float>(
                q, k, v, g, beta, scale, ssm_states, source_state_slots,
                destination_state_slots, out, stream);
        }
    } else {
        throw std::invalid_argument("GDN recurrent batch update unsupported output dtype");
    }
}

template <typename InputT>
void launch_recurrent_batch_update_for_input(
    const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
    const Tensor& beta, float scale, bool normalize_qk, Tensor& ssm_states,
    const Tensor& source_state_slots, const Tensor& destination_state_slots,
    Tensor& out, cudaStream_t stream) {
    if (ssm_states.dtype == DType::BF16) {
        launch_recurrent_batch_update_for_state<InputT, __nv_bfloat16>(
            q, k, v, g, beta, scale, normalize_qk, ssm_states,
            source_state_slots, destination_state_slots, out, stream);
    } else if (ssm_states.dtype == DType::FP32) {
        launch_recurrent_batch_update_for_state<InputT, float>(
            q, k, v, g, beta, scale, normalize_qk, ssm_states,
            source_state_slots, destination_state_slots, out, stream);
    } else {
        throw std::invalid_argument("GDN recurrent batch update unsupported state dtype");
    }
}

void launch_recurrent_batch_update(const Tensor& q, const Tensor& k, const Tensor& v,
                                   const Tensor& g, const Tensor& beta, float scale,
                                   bool normalize_qk, Tensor& ssm_states,
                                   const Tensor& source_state_slots,
                                   const Tensor& destination_state_slots, Tensor& out,
                                   cudaStream_t stream) {
    if (q.dtype == DType::BF16) {
        launch_recurrent_batch_update_for_input<__nv_bfloat16>(
            q, k, v, g, beta, scale, normalize_qk, ssm_states,
            source_state_slots, destination_state_slots, out, stream);
    } else if (q.dtype == DType::FP32) {
        launch_recurrent_batch_update_for_input<float>(
            q, k, v, g, beta, scale, normalize_qk, ssm_states,
            source_state_slots, destination_state_slots, out, stream);
    } else {
        throw std::invalid_argument("GDN recurrent batch update unsupported q/k/v dtype");
    }
}

void launch_recurrent_record(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                             const Tensor& beta, float scale, const Tensor& ssm_states,
                             const Tensor& valid_columns, const Tensor& initial_state_slots,
                             Tensor& key_record, Tensor& value_record, Tensor& gate_record,
                             Tensor& out, cudaStream_t stream) {
    if (valid_columns.data == nullptr) {
        launch_recurrent_record_fixed<false>(q, k, v, g, beta, scale, ssm_states, valid_columns,
                                             initial_state_slots, key_record, value_record,
                                             gate_record, out, stream);
    } else {
        launch_recurrent_record_fixed<true>(q, k, v, g, beta, scale, ssm_states, valid_columns,
                                            initial_state_slots, key_record, value_record,
                                            gate_record, out, stream);
    }
}

void launch_replay_fold(const GdnReplayRecords& records, LinearAttentionStateAllLayersView states,
                        const GdnReplayFoldKernelRows& rows, std::int32_t active_rows,
                        cudaStream_t stream) {
    if (records.spec.layers == FoldGeometry48x48::kLayers &&
        records.spec.qk_heads == FoldGeometry48x48::kQkHeads &&
        records.spec.value_heads == FoldGeometry48x48::kValueHeads &&
        records.spec.conv_channels == FoldGeometry48x48::kConvChannels) {
        launch_replay_fold_fixed<FoldGeometry48x48>(records, states, rows, active_rows, stream);
        return;
    }
    if (records.spec.layers == FoldGeometry30x32::kLayers &&
        records.spec.qk_heads == FoldGeometry30x32::kQkHeads &&
        records.spec.value_heads == FoldGeometry30x32::kValueHeads &&
        records.spec.conv_channels == FoldGeometry30x32::kConvChannels) {
        launch_replay_fold_fixed<FoldGeometry30x32>(records, states, rows, active_rows, stream);
        return;
    }
    throw std::invalid_argument("GDN replay fold launcher received an unregistered geometry");
}

} // namespace ninfer::ops::detail::gated_delta_net
