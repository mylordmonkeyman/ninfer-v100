#include "ops/attn_input_proj/bf16/bf16_attn_input_plan.h"

namespace ninfer::ops::detail {

void bf16_attn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                              Tensor& k, Tensor& v, cudaStream_t stream) {
    if (x.ne[1] == 1) {
        bf16_attn_input_decode_launch(x, weight, q, gate, k, v, stream);
        return;
    }
    if (x.ne[1] <= kBf16AttnInputSmallTDispatchEnd) {
        bf16_attn_input_small_t_launch(x, weight, q, gate, k, v, stream);
        return;
    }
#if defined(NINFER_VOLTA_BUILD)
    // The projection is independent across token columns. Tile arbitrary prefill
    // widths through the SM70-compatible exact-T kernels instead of the Ampere+
    // cp.async/BF16-MMA backend. The small-T implementation supports T<=32 even
    // though the Blackwell production crossover is lower.
    for (std::int32_t begin = 0; begin < x.ne[1];) {
        const std::int32_t remaining = x.ne[1] - begin;
        const std::int32_t count =
            remaining > kBf16AttnInputSmallTMaxTokens
                ? kBf16AttnInputSmallTMaxTokens
                : remaining;
        Tensor x_chunk = x.slice(1, begin, count);
        Tensor q_chunk = q.slice(1, begin, count);
        Tensor gate_chunk = gate.slice(1, begin, count);
        Tensor k_chunk = k.slice(1, begin, count);
        Tensor v_chunk = v.slice(1, begin, count);
        if (count == 1) {
            bf16_attn_input_decode_launch(
                x_chunk, weight, q_chunk, gate_chunk, k_chunk, v_chunk, stream);
        } else {
            bf16_attn_input_small_t_launch(
                x_chunk, weight, q_chunk, gate_chunk, k_chunk, v_chunk, stream);
        }
        begin += count;
    }
#else
    bf16_attn_input_mma_launch(x, weight, q, gate, k, v, stream);
#endif
}

} // namespace ninfer::ops::detail
