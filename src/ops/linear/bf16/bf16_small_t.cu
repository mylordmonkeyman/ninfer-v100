#include "ops/linear/bf16/bf16_launch.h"

#include "core/device.h"
#include "ops/linear/bf16/bf16_config.h"
#include "ops/linear/bf16/bf16_small_t.cuh"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {
namespace {

template <class Geometry, int ActiveTokens>
void launch_exact(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Schedule = typename Bf16LinearSmallTProductionSchedule<Geometry, ActiveTokens>::Type;
    static_assert((Geometry::kOutputRows % Schedule::kRowsPerCta) == 0);

    const Bf16SmallTContiguousOutput output{static_cast<__nv_bfloat16*>(out.data),
                                            Geometry::kOutputRows};
    constexpr int kBlocks = Geometry::kOutputRows / Schedule::kRowsPerCta;
    bf16_small_t_inner_kernel<Geometry, ActiveTokens, Schedule>
        <<<kBlocks, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const __nv_bfloat16*>(weight.qdata), output);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry, std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<Bf16Launch, sizeof...(Offsets)>{
        &launch_exact<Geometry, kBf16SmallTMinTokens + static_cast<int>(Offsets)>...};
}

using ControlGeometry    = Bf16GemvGeometry<14336, 5120>;
using OutputGeometry     = Bf16GemvGeometry<5120, 6144>;
using QsaIndexerGeometry = Bf16GemvGeometry<640, 2560>;
using PleKeyGeometry     = Bf16GemvGeometry<10240, 2560>;
using PleValueGeometry   = Bf16GemvGeometry<2560, 2560>;
using OutputHeadGeometry = Bf16GemvGeometry<248320, 2560>;
using OrcaRouterHeadGeometry = Bf16GemvGeometry<248320, 5120>;
using MtpQgkvGeometry    = Bf16GemvGeometry<13312, 2560>;
using MtpOutputGeometry  = Bf16GemvGeometry<2560, 6144>;

constexpr auto kQsaIndexerLaunchers =
    make_launchers<QsaIndexerGeometry>(std::make_index_sequence<7>{}); // T=2..8
constexpr auto kPleKeyLaunchers =
    make_launchers<PleKeyGeometry>(std::make_index_sequence<7>{}); // T=2..8
constexpr auto kPleValueLaunchers =
    make_launchers<PleValueGeometry>(std::make_index_sequence<7>{}); // T=2..8
constexpr auto kOutputHeadLaunchers =
    make_launchers<OutputHeadGeometry>(std::make_index_sequence<7>{}); // T=2..8
constexpr auto kOrcaRouterHeadLaunchers =
    make_launchers<OrcaRouterHeadGeometry>(std::make_index_sequence<7>{}); // T=2..8
constexpr auto kMtpQgkvLaunchers =
    make_launchers<MtpQgkvGeometry>(std::make_index_sequence<7>{}); // T=2..8
constexpr auto kMtpOutputLaunchers =
    make_launchers<MtpOutputGeometry>(std::make_index_sequence<7>{}); // T=2..8
constexpr auto kControlLaunchers = make_launchers<ControlGeometry>(
    std::make_index_sequence<kBf16SmallTMaxTokens - kBf16SmallTMinTokens + 1>{});
constexpr auto kOutputLaunchers = make_launchers<OutputGeometry>(
    std::make_index_sequence<kBf16SmallTMaxTokens - kBf16SmallTMinTokens + 1>{});

} // namespace

void launch_bf16_small_t(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const std::size_t index = static_cast<std::size_t>(x.ne[1] - kBf16SmallTMinTokens);
    if (weight.n == OrcaRouterHeadGeometry::kOutputRows && weight.k == OrcaRouterHeadGeometry::kInputRows) {
        kOrcaRouterHeadLaunchers[index](x, weight, out, stream);
        return;
    }
    if (weight.n == QsaIndexerGeometry::kOutputRows && weight.k == QsaIndexerGeometry::kInputRows) {
        kQsaIndexerLaunchers[index](x, weight, out, stream);
        return;
    }
    if (weight.n == PleKeyGeometry::kOutputRows && weight.k == PleKeyGeometry::kInputRows) {
        kPleKeyLaunchers[index](x, weight, out, stream);
        return;
    }
    if (weight.n == PleValueGeometry::kOutputRows && weight.k == PleValueGeometry::kInputRows) {
        kPleValueLaunchers[index](x, weight, out, stream);
        return;
    }
    if (weight.n == OutputHeadGeometry::kOutputRows && weight.k == OutputHeadGeometry::kInputRows) {
        kOutputHeadLaunchers[index](x, weight, out, stream);
        return;
    }
    if (weight.n == MtpQgkvGeometry::kOutputRows && weight.k == MtpQgkvGeometry::kInputRows) {
        kMtpQgkvLaunchers[index](x, weight, out, stream);
        return;
    }
    if (weight.n == MtpOutputGeometry::kOutputRows && weight.k == MtpOutputGeometry::kInputRows) {
        kMtpOutputLaunchers[index](x, weight, out, stream);
        return;
    }
    if (weight.n == ControlGeometry::kOutputRows && weight.k == ControlGeometry::kInputRows) {
        kControlLaunchers[index](x, weight, out, stream);
        return;
    }
    if (weight.n == OutputGeometry::kOutputRows && weight.k == OutputGeometry::kInputRows) {
        kOutputLaunchers[index](x, weight, out, stream);
        return;
    }
    throw std::invalid_argument("bf16 linear small-T: unsupported exact problem");
}

} // namespace ninfer::ops::detail
