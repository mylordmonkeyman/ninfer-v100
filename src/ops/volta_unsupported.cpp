#include "ops/candidate_selector/bf16/candidate_selector_path_kernels.h"
#include "ops/context_kv_materialize/launch.h"

#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {

#if !defined(NINFER_VOLTA_BUILD)
#error "volta_unsupported.cpp must only be compiled for the SM70 build"
#endif

namespace {

[[noreturn]] void throw_unavailable(const char* op) {
    throw std::runtime_error(std::string(op) +
                             " is not part of the Phase-10 Volta bring-up path");
}

} // namespace

void candidate_selector_path_launch(
    SelectorRoute, const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&,
    const Tensor&, const Tensor&, const SamplingConfig*, Tensor&, Tensor&,
    const SelectorWorkspace&, cudaStream_t) {
    throw_unavailable("candidate selector");
}

void context_kv_materialize_launch(
    const Tensor&, const Tensor&, const Tensor&, const Tensor&,
    const std::array<ContextKVMaterializeLayerView, kContextKVMaterializeLayers>&,
    ContextKVMaterializeExecutionEnvelope, ContextKVMaterializeRoute, const Tensor&,
    cudaStream_t) {
    throw_unavailable("context KV materialization");
}

} // namespace ninfer::ops::detail
