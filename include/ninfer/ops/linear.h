#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * @brief Permitted private activation-compute profiles for a linear projection.
 *
 * The policy constrains private route selection; it does not select a kernel or prescribe a
 * particular MMA instruction. The public activation and output tensors remain BF16 for every
 * policy.
 */
enum class LinearPolicy : std::uint8_t {
    A16Only, ///< Admit only A16 compute profiles.
    AllowA8, ///< Admit either A16 or A8 compute profiles.
    AllowA4, ///< Admit either A16 or A4 compute profiles.
};

/**
 * Returns the caller-owned transient capacity required by Linear for every T in the inclusive
 * `[min_tokens,max_tokens]` interval. Invalid registered profiles, policies, or intervals throw;
 * a legal route that requires no transient storage returns zero.
 */
[[nodiscard]] std::size_t linear_workspace_capacity_bytes(QType qtype, std::int32_t output_rows,
                                                          std::int32_t input_rows,
                                                          LinearPolicy policy,
                                                          std::int32_t min_tokens,
                                                          std::int32_t max_tokens);

/**
 * @brief Applies a bias-free matrix projection independently to every input column.
 *
 * @details The ideal mathematical result is
 *
 * @f[
 *   \mathrm{ideal}_{n,t} =
 *   \sum_{k=0}^{K-1}
 *     \mathrm{FP32Dequant}(w)_{n,k}\,\mathrm{FP32}(x_{k,t}).
 * @f]
 *
 * `out` stores a BF16 approximation of this ideal result under the named numerical criterion for
 * the selected private activation-compute path.
 *
 * @par Logical tensors and layout
 * `x` is contiguous, non-null, 16-byte-aligned BF16 `[K,T]`, `w` has logical shape `[N,K]`, and
 * `out` is contiguous, non-null, 16-byte-aligned BF16 `[N,T]`. Every logical extent is positive;
 * in particular, `T=0` is invalid rather than a no-op. Dimension zero is stored fastest. The Op has
 * no bias, activation, residual addition, or transpose mode.
 *
 * @par Supported execution domain
 * Registered execution uses RowSplit Q4G64_F16S, Q5G64_F16S, Q6G64_F16S, or W8G32_F16S weights
 * with FP16 scales, block-scaled NVFP4 weights, row-scaled FP8_E4M3FN_ROW_BF16S or
 *
 * FP8_E4M3FN_ROW_F32S weights, plus registered contiguous BF16_CTRL problems. Each format owns a
 *
 * finite registry of exact physical weight problems and selects its kernel internally; a valid
 *
 * encoding and alignment do not imply arbitrary N/K support. BF16-scale FP8 registers `[N,K]` in
 *
 * `{[14336,5120], [16384,5120], [34816,5120], [248320,5120], [5120,6144], [5120,17408]}`.
 * F32-scale
 * FP8 registers the Flash-Next problems `{[13312,2560], [16384,2560], [2560,6144], [248320,2560]}`.
 * Both accept
 * every positive T. BF16_CTRL additionally registers the Flash-Next indexer
 * `[640,2560]`, PLE key
 * projection `[10240,2560]`, and PLE value projection `[2560,2560]` for
 * T=1..8. It registers the
 * Vision raw-patch problems `{[1152,1536], [3456,1152], [1152,1152],
 * [4304,1152],
 * [1152,4304]}` for P in `{4,8,...,131072}` and the merged-token problems
 * `{[4608,4608],
 * [2560,4608]}` for V in `[1,32768]`. The current NVFP4 problems register the
 * five non-vocabulary
 * BF16-scale FP8 geometries and accept every positive T. Text and MTP
 * packed-weight problems accept
 * every positive column extent T. A matrix column does not
 * inherently represent a text token.
 * W8 also registers `[5120,25600]`, and BF16_CTRL registers `[256,5120]`, at every
 * positive T for DFlash2. BF16_CTRL also registers the OrcaRouter 27B output head
 * `[248320,5120]` at every positive T. FP32_CTRL is unsupported.
 *
 * @par Numerical contract
 * Test fixture code materializes the persistent weight as its logical FP32 dequantized matrix.
 * The one Linear oracle accepts that matrix and the FP32 values represented by the BF16 activation,
 * evaluates every complete dot product with naive FP64 accumulation, and retains the FP64 result.
 * The BF16 output is promoted and compared against that result. Output representation,
 * accumulator precision, activation quantization, staging, reduction order, and kernel schedule
 * are private implementation effects covered by the named tolerance for the selected
 * activation-compute path; none is copied into the oracle. Kernel, schedule, template instance,
 * host launcher, and T region do not create separate criteria inside one path.
 *
 * @par Compute policy
 * `policy` specifies the permitted private activation-compute set. A permission does not require a
 * corresponding low-precision route: the resolved plan may remain A16 when that is the qualified
 * choice. BF16_CTRL admits only LinearPolicy::A16Only. Registered Q4/Q5/Q6/W8 formats admit
 * LinearPolicy::A16Only and LinearPolicy::AllowA8. The five non-vocabulary FP8 problems admit the
 * same two policies at every positive T. AllowA8 resolves `[14336,5120]` to A16 through T=11 and
 * A8 from T=12; `[16384,5120]` to A16 through T=10 and A8 from T=11; `[34816,5120]` to A8 at T=1,
 * A16 at T=2..4, and A8 from T=5; both `[5120,6144]` and `[5120,17408]` resolve T<25 to A16 and
 * T>=25 to A8. FP8 `[248320,5120]` and the three F32-scale Flash-Next problems admit A16Only,
 *
 * AllowA8, and AllowA4; every policy retains A16 compute at every positive T. NVFP4 admits
 *
 * A16Only and AllowA4; AllowA4 permits the private resolver to select either a qualified A16 route

 * * or activation quantization to NVFP4 at every positive T. The selected route depends only on
 * the
 * registered problem and T.
 *
 * @par Workspace
 * `workspace` is caller-owned call-scoped transient storage sized by
 * linear_workspace_capacity_bytes(). It must not overlap x, any weight plane, or out. Linear does
 * not allocate device memory internally.
 *
 * @param[in] x Contiguous, non-null, 16-byte-aligned BF16 input matrix `[K,T]`.
 * @param[in] w Logical weight matrix `[N,K]` in a registered persistent format and layout.
 * @param[out] out Contiguous, non-null, 16-byte-aligned BF16 output matrix `[N,T]`. It must not
 * overlap `x` or any weight plane.
 * @param[in] policy Permitted private activation-compute profiles.
 * @param[in,out] workspace Caller-owned transient arena.
 * @param[in] stream CUDA stream on which execution is enqueued.
 */
void linear(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
            WorkspaceArena& workspace, cudaStream_t stream);

/**
 * @brief Applies the A16-only form of the bias-free matrix projection.
 *
 * @details This overload admits only A16 compute and requires no transient workspace. All tensor,
 * weight, aliasing, and execution-domain requirements of the policy-bearing overload apply.
 *
 * @param[in] x Contiguous BF16 input matrix `[K,T]`.
 * @param[in] w Logical weight matrix `[N,K]` in a registered persistent format and layout.
 * @param[out] out Contiguous BF16 output matrix `[N,T]`.
 * @param[in] stream CUDA stream on which execution is enqueued.
 */
void linear(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops
