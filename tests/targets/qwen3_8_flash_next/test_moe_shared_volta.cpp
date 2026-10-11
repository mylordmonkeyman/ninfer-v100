#include "core/arena.h"
#include "core/device.h"
#include "targets/qwen3_8_flash_next/impl/moe.h"
#include "targets/qwen3_8_flash_next/impl/moe_kernels.h"
#include "targets/qwen3_8_flash_next/impl/moe_workspace.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
using namespace ninfer;
using namespace ninfer::targets::qwen3_8_flash_next::detail;
constexpr int H = 2560, I = 640, P = 11;

float decode(std::uint16_t x) { return std::bit_cast<float>(std::uint32_t(x) << 16); }
std::uint16_t encode(float x) {
    const auto bits = std::bit_cast<std::uint32_t>(x);
    return std::uint16_t((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16);
}
Weight bf16_weight(void* p, int rows, int columns) {
    Weight w{};
    w.payload = w.qdata = p;
    w.payload_bytes = std::uint64_t(rows) * columns * 2;
    w.qtype = QType::BF16_CTRL;
    w.layout = QuantLayout::Contiguous;
    w.n = rows; w.k = columns; w.ndim = 2;
    w.shape[0] = w.padded_shape[0] = rows;
    w.shape[1] = w.padded_shape[1] = columns;
    return w;
}
struct Error {
    double diff = 0, norm = 0;
    void add(double actual, double expected) {
        if (!std::isfinite(actual) || !std::isfinite(expected))
            throw std::runtime_error("nonfinite shared expert output");
        diff += (actual - expected) * (actual - expected);
        norm += expected * expected;
    }
    double relative() const {
        if (!(norm > 0)) throw std::runtime_error("vacuous shared oracle");
        return std::sqrt(diff / norm);
    }
    void require(const char* name) const {
        std::cout << name << " relL2=" << relative() << "\n";
        if (relative() > 1e-3) throw std::runtime_error(name);
    }
};

#if defined(NINFER_VOLTA_BUILD)
void qualify(DeviceContext& device, int tokens, bool admission) {
    std::mt19937 rng(1979 + tokens);
    // The odd-sized case also exercises positive/negative saturated SiLU gates
    // and cancellation, while all represented values remain in the FP16 range.
    const float input_scale = tokens == 257 ? 32.0f : 1.0f;
    std::uniform_real_distribution<float> xdist(-0.5f, 0.5f);
    std::uniform_real_distribution<float> wdist(-0.025f, 0.025f);
    std::vector<std::uint16_t> x(std::size_t(H) * tokens), gate(std::size_t(H) * I),
        up(gate.size()), down(gate.size());
    for (auto& v : x) v = encode(xdist(rng) * input_scale);
    for (auto* matrix : {&gate, &up, &down})
        for (auto& v : *matrix) v = encode(wdist(rng));
    DeviceBuffer dx(x.size()*2), dg(gate.size()*2), du(up.size()*2), dd(down.size()*2);
    dx.copy_from_host(x.data(), dx.bytes);
    dg.copy_from_host(gate.data(), dg.bytes);
    du.copy_from_host(up.data(), du.bytes);
    dd.copy_from_host(down.data(), dd.bytes);
    MoeWeights weights{};
    weights.shared_gate = bf16_weight(dg.p, I, H);
    weights.shared_up = bf16_weight(du.p, I, H);
    weights.shared_down = bf16_weight(dd.p, H, I);
    WorkspaceArena arena(flash_next_moe_workspace_capacity_bytes(1, tokens));
    auto scratch = allocate_flash_next_moe_workspace(arena, tokens);
    std::vector<float> scales(tokens), routed(std::size_t(H)*tokens);
    for (auto& v : scales) v = 0.5f + xdist(rng) * 0.8f;
    for (auto& v : routed) v = xdist(rng) * 0.02f;
    CUDA_CHECK(cudaMemcpy(scratch.shared_scale.data, scales.data(),
                          scales.size()*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy2D(scratch.activations.data, P*I*2, routed.data(), H*4,
                            H*4, tokens, cudaMemcpyHostToDevice));
    constexpr std::size_t guard = 256;
    DeviceBuffer out(std::size_t(H)*tokens*2 + 2*guard), full(std::size_t(H)*tokens*4 + 2*guard);
    out.fill(0xa5); full.fill(0xa5);
    Tensor input(dx.p, DType::BF16, {H, tokens});
    Tensor output(static_cast<std::uint8_t*>(out.p)+guard, DType::BF16, {H,tokens});
    Tensor fp32(static_cast<std::uint8_t*>(full.p)+guard, DType::FP32, {H,tokens});
    auto launch = [&](bool cublas, bool emulate_routed_writes = false) {
        flash_next_moe_host_shared_launch(input, weights, scratch, device.stream,
                                          nullptr, false, cublas);
        if (emulate_routed_writes) {
            // The real schedule fills this aliased buffer with routed outputs,
            // then reduces them into activations before shared down. Overwrite
            // the entire borrowed region to expose a stale staging dependency.
            CUDA_CHECK(cudaMemsetAsync(scratch.down_intermediate.data, 0xa5,
                std::size_t(H)*10*tokens*sizeof(float), device.stream));
        }
        flash_next_moe_host_routed_merge_launch(weights, scratch, output, tokens,
                                                device.stream, false, &fp32, cublas);
    };
    launch(true, true);
    device.synchronize();

    // Each mathematical oracle sums every logical product in FP64 from represented
    // public BF16 inputs. Gate/up verifies its exposed BF16 activation. Down verifies
    // from that represented activation, independently of either GPU reduction tree.
    Error act_error, down_error, merge_error, output_error, complete_error;
    for (int token : {0, tokens/2, tokens-1}) {
        std::array<std::uint16_t, I> activation;
        std::array<std::uint16_t, I> oracle_activation;
        std::array<float, H> actual_down, actual_merge;
        std::array<std::uint16_t, H> actual_output;
        CUDA_CHECK(cudaMemcpy(activation.data(),
            static_cast<std::uint16_t*>(scratch.activations.data)+(std::size_t(token)*P+10)*I,
            I*2, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(actual_down.data(),
            static_cast<float*>(scratch.shared_projection_fp32.data)+std::size_t(token)*H,
            H*4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(actual_merge.data(), static_cast<float*>(fp32.data)+std::size_t(token)*H,
            H*4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(actual_output.data(), static_cast<std::uint16_t*>(output.data)+std::size_t(token)*H,
            H*2, cudaMemcpyDeviceToHost));
        for (int row=0; row<I; ++row) {
            double g=0, u=0;
            for (int k=0; k<H; ++k) {
                const double value=decode(x[std::size_t(token)*H+k]);
                g += value*decode(gate[std::size_t(row)*H+k]);
                u += value*decode(up[std::size_t(row)*H+k]);
            }
            const double expected=(g/(1.0+std::exp(-g)))*u;
            oracle_activation[row] = encode(float(expected));
            act_error.add(decode(activation[row]), decode(oracle_activation[row]));
        }
        for (int row=0; row<H; ++row) {
            double expected=0, complete=0;
            for (int k=0; k<I; ++k) {
                expected += double(decode(activation[k]))*decode(down[std::size_t(row)*I+k]);
                complete += double(decode(oracle_activation[k]))*decode(down[std::size_t(row)*I+k]);
            }
            down_error.add(actual_down[row], expected);
            const double merged=double(scales[token])*expected+routed[std::size_t(token)*H+row];
            merge_error.add(actual_merge[row], merged);
            output_error.add(decode(actual_output[row]), decode(encode(float(merged))));
            const double independent_merged = double(scales[token])*complete+routed[std::size_t(token)*H+row];
            complete_error.add(decode(actual_output[row]), decode(encode(float(independent_merged))));
        }
    }
    std::cout << "SHARED_ORACLE tokens=" << tokens << "\n";
    act_error.require("shared gate/up");
    down_error.require("shared down");
    merge_error.require("shared FP32 merge");
    output_error.require("shared BF16 output");
    complete_error.require("complete shared formula BF16 output");
    for (const auto* buffer : {&out, &full}) {
        std::array<std::uint8_t, guard> before{}, after{};
        buffer->copy_to_host(before.data(), guard);
        buffer->copy_to_host(after.data(), guard, buffer->bytes-guard);
        if (!std::all_of(before.begin(),before.end(),[](auto v){return v==0xa5;}) ||
            !std::all_of(after.begin(),after.end(),[](auto v){return v==0xa5;}))
            throw std::runtime_error("shared output guard");
    }
    // Repeated same-input launches must not depend on reused gate/up/down staging.
    std::vector<std::uint16_t> first(std::size_t(H)*tokens), second(first.size());
    CUDA_CHECK(cudaMemcpy(first.data(),output.data,first.size()*2,cudaMemcpyDeviceToHost));
    launch(true); device.synchronize();
    CUDA_CHECK(cudaMemcpy(second.data(),output.data,second.size()*2,cudaMemcpyDeviceToHost));
    if (first != second) throw std::runtime_error("shared route not repeatable");

    if (admission) {
        CudaEventTimer timer(device);
        std::array<float,3> simt{}, cublas{};
        launch(false); launch(true); device.synchronize();
        for (int trial=0; trial<3; ++trial) {
            for (bool candidate : {bool(trial%2), !bool(trial%2)}) {
                timer.start(); launch(candidate);
                (candidate ? cublas : simt)[trial] = timer.stop_ms();
            }
        }
        std::sort(simt.begin(),simt.end()); std::sort(cublas.begin(),cublas.end());
        const double ratio = cublas[1]/simt[1];
        std::cout << "SHARED_PREFILL_ADMISSION {\"tokens\":" << tokens
            << ",\"simt_ms\":" << simt[1] << ",\"cublas_ms\":" << cublas[1]
            << ",\"ratio\":" << ratio << "}\n";
        if (!(ratio <= .8)) throw std::runtime_error("shared operator did not improve at least 20%");
    }
}
#endif
} // namespace

int main() {
#if !defined(NINFER_VOLTA_BUILD)
    return 77;
#else
    int count=0;
    const auto error=cudaGetDeviceCount(&count);
    if (error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver || count==0) return 77;
    try {
        CUDA_CHECK(error);
        DeviceContext device;
        for (int tokens : {256,257,7111}) qualify(device,tokens,tokens==7111);
        std::cout << "OK shared Volta independent oracles and admission\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL shared Volta: " << e.what() << "\n";
        return 1;
    }
#endif
}
