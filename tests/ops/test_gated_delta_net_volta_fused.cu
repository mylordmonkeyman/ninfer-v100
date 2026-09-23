#include "ops/linear_attention/gated_delta_net/volta/fused_grouped_dv16.cuh"
#include "ops/linear_attention/gated_delta_net/volta/fused_state_output.cuh"
#include "ops/linear_attention/gated_delta_net/volta/launch.h"
#include "ops/linear_attention/gated_delta_net/volta/prepare_qk_matrices.cuh"
#include "ops/linear_attention/gated_delta_net/volta/schedule.cuh"
#include "ops/linear_attention/gated_delta_net/volta/shared_tiles.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string_view>
#include <vector>

namespace volta = ninfer::ops::detail::gated_delta_net::volta;

namespace {

constexpr int kQkHeads = 1;
constexpr int kValueHeads = volta::kGroupSize;
constexpr int kDim = volta::kStateDim;
constexpr double kScale = 0.0883883476483184405501055452631; // 1 / sqrt(128)

bool cuda_ok(cudaError_t status, const char* what) {
    if (status == cudaSuccess) { return true; }
    std::cerr << what << ": " << cudaGetErrorString(status) << "\n";
    return false;
}

std::uint16_t float_to_bf16_host(float value) {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    const std::uint32_t abs = bits & 0x7fffffffU;
    if (abs >= 0x7f800000U) {
        const std::uint16_t upper = static_cast<std::uint16_t>(bits >> 16);
        if (abs == 0x7f800000U) { return upper; }
        return static_cast<std::uint16_t>(upper | 0x0040U);
    }
    const std::uint32_t lsb = (bits >> 16) & 1U;
    return static_cast<std::uint16_t>((bits + 0x7fffU + lsb) >> 16);
}

float bf16_to_float_host(std::uint16_t bits) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16);
}

struct HostCase {
    int tokens = 0;
    std::vector<std::uint16_t> q;
    std::vector<std::uint16_t> k;
    std::vector<std::uint16_t> v;
    std::vector<float> g;
    std::vector<float> beta;
    std::vector<float> state;
};

HostCase make_case(int tokens, bool zero_state) {
    HostCase data{};
    data.tokens = tokens;

    const std::size_t qk_elements =
        static_cast<std::size_t>(tokens) * kQkHeads * kDim;
    const std::size_t v_elements =
        static_cast<std::size_t>(tokens) * kValueHeads * kDim;
    const std::size_t gate_elements =
        static_cast<std::size_t>(tokens) * kValueHeads;
    const std::size_t state_elements =
        static_cast<std::size_t>(kValueHeads) * kDim * kDim;

    data.q.assign(qk_elements, 0U);
    data.k.assign(qk_elements, 0U);
    data.v.resize(v_elements);
    data.g.assign(gate_elements, 0.0F);
    data.beta.resize(gate_elements);
    data.state.resize(state_elements);

    for (int token = 0; token < tokens; ++token) {
        const std::size_t qk_base = static_cast<std::size_t>(token) * kDim;

        const int kd0 = token & 15;
        const int kd1 = 16 + (token & 15);
        data.k[qk_base + kd0] = float_to_bf16_host(1.0F);
        data.k[qk_base + kd1] = float_to_bf16_host(0.5F);

        const int qd0 = (token * 7) & 31;
        const int qd1 = 32 + ((token * 5 + 1) & 31);
        data.q[qk_base + qd0] = float_to_bf16_host(1.0F);
        data.q[qk_base + qd1] = float_to_bf16_host(-0.5F);

        for (int head = 0; head < kValueHeads; ++head) {
            const std::size_t gate_index =
                static_cast<std::size_t>(token) * kValueHeads + head;
            data.beta[gate_index] = 0.125F * static_cast<float>(head + 1);

            const std::size_t value_base = gate_index * kDim;
            for (int dv = 0; dv < kDim; ++dv) {
                const int signed_bucket = (token + head * 3 + dv * 5) % 9 - 4;
                const float value = static_cast<float>(signed_bucket) * 0.03125F;
                data.v[value_base + dv] = float_to_bf16_host(value);
            }
        }
    }

    if (zero_state) {
        std::fill(data.state.begin(), data.state.end(), 0.0F);
    } else {
        for (int head = 0; head < kValueHeads; ++head) {
            for (int dv = 0; dv < kDim; ++dv) {
                for (int d = 0; d < kDim; ++d) {
                    const int signed_bucket = (head * 7 + dv * 3 + d * 5) % 9 - 4;
                    const std::size_t index =
                        (static_cast<std::size_t>(head) * kDim + dv) * kDim + d;
                    data.state[index] =
                        static_cast<float>(signed_bucket) * 0.015625F;
                }
            }
        }
    }

    return data;
}

struct ReferenceResult {
    std::vector<double> output;
    std::vector<double> state;
};

ReferenceResult sequential_reference(const HostCase& data) {
    ReferenceResult result{};
    result.output.resize(
        static_cast<std::size_t>(data.tokens) * kValueHeads * kDim);
    result.state.resize(data.state.size());

    for (std::size_t i = 0; i < data.state.size(); ++i) {
        result.state[i] = static_cast<double>(data.state[i]);
    }

    std::array<double, kDim> q_norm{};
    std::array<double, kDim> k_norm{};

    for (int token = 0; token < data.tokens; ++token) {
        const std::size_t qk_base = static_cast<std::size_t>(token) * kDim;

        double q_sum = 0.0;
        double k_sum = 0.0;
        for (int d = 0; d < kDim; ++d) {
            const double q_value =
                static_cast<double>(bf16_to_float_host(data.q[qk_base + d]));
            const double k_value =
                static_cast<double>(bf16_to_float_host(data.k[qk_base + d]));
            q_norm[d] = q_value;
            k_norm[d] = k_value;
            q_sum += q_value * q_value;
            k_sum += k_value * k_value;
        }
        const double q_inv = 1.0 / std::sqrt(q_sum + 1.0e-6);
        const double k_inv = 1.0 / std::sqrt(k_sum + 1.0e-6);
        for (int d = 0; d < kDim; ++d) {
            q_norm[d] *= q_inv;
            k_norm[d] *= k_inv;
        }

        for (int head = 0; head < kValueHeads; ++head) {
            const std::size_t gate_index =
                static_cast<std::size_t>(token) * kValueHeads + head;
            const double alpha = std::exp(static_cast<double>(data.g[gate_index]));
            const double beta = static_cast<double>(data.beta[gate_index]);
            const std::size_t value_base = gate_index * kDim;

            for (int dv = 0; dv < kDim; ++dv) {
                const std::size_t state_base =
                    (static_cast<std::size_t>(head) * kDim + dv) * kDim;
                double projection = 0.0;
                for (int d = 0; d < kDim; ++d) {
                    projection += result.state[state_base + d] * k_norm[d];
                }

                const double value =
                    static_cast<double>(bf16_to_float_host(data.v[value_base + dv]));
                const double delta = beta * (value - alpha * projection);
                for (int d = 0; d < kDim; ++d) {
                    result.state[state_base + d] =
                        alpha * result.state[state_base + d] + delta * k_norm[d];
                }
            }

            const std::size_t output_base =
                (static_cast<std::size_t>(token) * kValueHeads + head) * kDim;
            for (int dv = 0; dv < kDim; ++dv) {
                const std::size_t state_base =
                    (static_cast<std::size_t>(head) * kDim + dv) * kDim;
                double value = 0.0;
                for (int d = 0; d < kDim; ++d) {
                    value += result.state[state_base + d] * q_norm[d];
                }
                result.output[output_base + dv] = value * kScale;
            }
        }
    }

    return result;
}

struct DeviceBuffers {
    std::uint16_t* q = nullptr;
    std::uint16_t* k = nullptr;
    std::uint16_t* v = nullptr;
    float* g = nullptr;
    float* beta = nullptr;
    float* state_input = nullptr;
    float* state_baseline = nullptr;
    float* state_grouped = nullptr;
    std::uint16_t* output_baseline = nullptr;
    std::uint16_t* output_grouped = nullptr;
    float* q_inv = nullptr;
    float* k_inv = nullptr;
    float* kk = nullptr;
    float* qk = nullptr;

    ~DeviceBuffers() {
        cudaFree(q);
        cudaFree(k);
        cudaFree(v);
        cudaFree(g);
        cudaFree(beta);
        cudaFree(state_input);
        cudaFree(state_baseline);
        cudaFree(state_grouped);
        cudaFree(output_baseline);
        cudaFree(output_grouped);
        cudaFree(q_inv);
        cudaFree(k_inv);
        cudaFree(kk);
        cudaFree(qk);
    }
};

bool allocate_buffers(DeviceBuffers& d, const HostCase& data) {
    const int chunks = data.tokens / volta::kChunkSize;
    const std::size_t norm_elements =
        static_cast<std::size_t>(chunks) * kQkHeads * volta::kChunkSize;
    const std::size_t tile_elements =
        static_cast<std::size_t>(chunks) * kQkHeads *
        volta::kLowerTiles * volta::kTileElements;
    const std::size_t output_elements =
        static_cast<std::size_t>(data.tokens) * kValueHeads * kDim;

    return
        cuda_ok(cudaMalloc(&d.q, data.q.size() * sizeof(std::uint16_t)), "cudaMalloc q") &&
        cuda_ok(cudaMalloc(&d.k, data.k.size() * sizeof(std::uint16_t)), "cudaMalloc k") &&
        cuda_ok(cudaMalloc(&d.v, data.v.size() * sizeof(std::uint16_t)), "cudaMalloc v") &&
        cuda_ok(cudaMalloc(&d.g, data.g.size() * sizeof(float)), "cudaMalloc g") &&
        cuda_ok(cudaMalloc(&d.beta, data.beta.size() * sizeof(float)), "cudaMalloc beta") &&
        cuda_ok(cudaMalloc(&d.state_input, data.state.size() * sizeof(float)),
                "cudaMalloc state input") &&
        cuda_ok(cudaMalloc(&d.state_baseline, data.state.size() * sizeof(float)),
                "cudaMalloc baseline state") &&
        cuda_ok(cudaMalloc(&d.state_grouped, data.state.size() * sizeof(float)),
                "cudaMalloc grouped state") &&
        cuda_ok(cudaMalloc(&d.output_baseline, output_elements * sizeof(std::uint16_t)),
                "cudaMalloc baseline output") &&
        cuda_ok(cudaMalloc(&d.output_grouped, output_elements * sizeof(std::uint16_t)),
                "cudaMalloc grouped output") &&
        cuda_ok(cudaMalloc(&d.q_inv, norm_elements * sizeof(float)), "cudaMalloc q inv") &&
        cuda_ok(cudaMalloc(&d.k_inv, norm_elements * sizeof(float)), "cudaMalloc k inv") &&
        cuda_ok(cudaMalloc(&d.kk, tile_elements * sizeof(float)), "cudaMalloc kk") &&
        cuda_ok(cudaMalloc(&d.qk, tile_elements * sizeof(float)), "cudaMalloc qk");
}

bool copy_inputs(DeviceBuffers& d, const HostCase& data) {
    return
        cuda_ok(cudaMemcpy(d.q, data.q.data(), data.q.size() * sizeof(std::uint16_t),
                           cudaMemcpyHostToDevice), "copy q") &&
        cuda_ok(cudaMemcpy(d.k, data.k.data(), data.k.size() * sizeof(std::uint16_t),
                           cudaMemcpyHostToDevice), "copy k") &&
        cuda_ok(cudaMemcpy(d.v, data.v.data(), data.v.size() * sizeof(std::uint16_t),
                           cudaMemcpyHostToDevice), "copy v") &&
        cuda_ok(cudaMemcpy(d.g, data.g.data(), data.g.size() * sizeof(float),
                           cudaMemcpyHostToDevice), "copy g") &&
        cuda_ok(cudaMemcpy(d.beta, data.beta.data(), data.beta.size() * sizeof(float),
                           cudaMemcpyHostToDevice), "copy beta") &&
        cuda_ok(cudaMemcpy(d.state_input, data.state.data(), data.state.size() * sizeof(float),
                           cudaMemcpyHostToDevice), "copy state input");
}

using LaunchFn = cudaError_t (*)(const volta::FusedOneWindowArgs&, cudaStream_t);

bool launch_backend(LaunchFn launch, DeviceBuffers& d, const HostCase& data,
                    float* state_write, std::uint16_t* output) {
    volta::FusedOneWindowArgs args{};
    args.q_bf16 = d.q;
    args.k_bf16 = d.k;
    args.v_bf16 = d.v;
    args.g = d.g;
    args.beta = d.beta;
    args.state_read = d.state_input;
    args.state_write = state_write;
    args.output_bf16 = output;
    args.q_inv_norm = d.q_inv;
    args.k_inv_norm = d.k_inv;
    args.kk_tiles = d.kk;
    args.qk_tiles = d.qk;
    args.qk_heads = kQkHeads;
    args.value_heads = kValueHeads;
    args.tokens = data.tokens;

    return cuda_ok(launch(args, nullptr), "launch fused backend") &&
           cuda_ok(cudaDeviceSynchronize(), "sync fused backend");
}

bool close_enough(double got, double expected, double abs_tol, double rel_tol) {
    return std::abs(got - expected) <= abs_tol + rel_tol * std::abs(expected);
}

int compare_results(const HostCase& data, const ReferenceResult& reference,
                    const std::vector<std::uint16_t>& baseline_output,
                    const std::vector<std::uint16_t>& grouped_output,
                    const std::vector<float>& baseline_state,
                    const std::vector<float>& grouped_state) {
    if (baseline_output != grouped_output) {
        for (std::size_t i = 0; i < baseline_output.size(); ++i) {
            if (baseline_output[i] != grouped_output[i]) {
                std::cerr << "baseline/grouped output mismatch at " << i
                          << " baseline=0x" << std::hex << baseline_output[i]
                          << " grouped=0x" << grouped_output[i] << std::dec << "\n";
                break;
            }
        }
        return 1;
    }

    for (std::size_t i = 0; i < baseline_state.size(); ++i) {
        if (std::bit_cast<std::uint32_t>(baseline_state[i]) !=
            std::bit_cast<std::uint32_t>(grouped_state[i])) {
            std::cerr << "baseline/grouped state mismatch at " << i
                      << " baseline=" << baseline_state[i]
                      << " grouped=" << grouped_state[i] << "\n";
            return 1;
        }
    }

    double worst_output_abs = 0.0;
    double worst_state_abs = 0.0;
    std::size_t worst_output_index = 0;
    std::size_t worst_state_index = 0;

    for (std::size_t i = 0; i < baseline_output.size(); ++i) {
        const double got =
            static_cast<double>(bf16_to_float_host(baseline_output[i]));
        const double expected = reference.output[i];
        const double error = std::abs(got - expected);
        if (error > worst_output_abs) {
            worst_output_abs = error;
            worst_output_index = i;
        }
        if (!close_enough(got, expected, 0.015, 0.04)) {
            std::cerr << "FP64 oracle output mismatch at " << i
                      << " got=" << got << " expected=" << expected
                      << " abs_error=" << error << "\n";
            return 1;
        }
    }

    for (std::size_t i = 0; i < baseline_state.size(); ++i) {
        const double got = static_cast<double>(baseline_state[i]);
        const double expected = reference.state[i];
        const double error = std::abs(got - expected);
        if (error > worst_state_abs) {
            worst_state_abs = error;
            worst_state_index = i;
        }
        if (!close_enough(got, expected, 0.015, 0.04)) {
            std::cerr << "FP64 oracle state mismatch at " << i
                      << " got=" << got << " expected=" << expected
                      << " abs_error=" << error << "\n";
            return 1;
        }
    }

    std::cout << "tokens=" << data.tokens
              << " worst_output_abs=" << worst_output_abs
              << " @" << worst_output_index
              << " worst_state_abs=" << worst_state_abs
              << " @" << worst_state_index << "\n";
    return 0;
}

int run_case(int tokens, bool zero_state) {
    HostCase data = make_case(tokens, zero_state);
    const ReferenceResult reference = sequential_reference(data);

    DeviceBuffers d{};
    if (!allocate_buffers(d, data) || !copy_inputs(d, data)) { return 1; }

    const int chunks = tokens / volta::kChunkSize;
    if (!cuda_ok(volta::launch_prepare_qk_matrices(
                     d.q, d.k, d.q_inv, d.k_inv, d.kk, d.qk,
                     kQkHeads, chunks, nullptr),
                 "launch prepare_qk_matrices") ||
        !cuda_ok(cudaDeviceSynchronize(), "sync prepare_qk_matrices")) {
        return 1;
    }

    if (!launch_backend(volta::launch_fused_state_output_dv16, d, data,
                        d.state_baseline, d.output_baseline)) {
        return 1;
    }
    if (!launch_backend(volta::launch_fused_grouped_dv16, d, data,
                        d.state_grouped, d.output_grouped)) {
        return 1;
    }

    const std::size_t output_elements =
        static_cast<std::size_t>(tokens) * kValueHeads * kDim;
    std::vector<std::uint16_t> baseline_output(output_elements);
    std::vector<std::uint16_t> grouped_output(output_elements);
    std::vector<float> baseline_state(data.state.size());
    std::vector<float> grouped_state(data.state.size());

    if (!cuda_ok(cudaMemcpy(baseline_output.data(), d.output_baseline,
                            baseline_output.size() * sizeof(std::uint16_t),
                            cudaMemcpyDeviceToHost), "copy baseline output") ||
        !cuda_ok(cudaMemcpy(grouped_output.data(), d.output_grouped,
                            grouped_output.size() * sizeof(std::uint16_t),
                            cudaMemcpyDeviceToHost), "copy grouped output") ||
        !cuda_ok(cudaMemcpy(baseline_state.data(), d.state_baseline,
                            baseline_state.size() * sizeof(float),
                            cudaMemcpyDeviceToHost), "copy baseline state") ||
        !cuda_ok(cudaMemcpy(grouped_state.data(), d.state_grouped,
                            grouped_state.size() * sizeof(float),
                            cudaMemcpyDeviceToHost), "copy grouped state")) {
        return 1;
    }

    return compare_results(data, reference, baseline_output, grouped_output,
                           baseline_state, grouped_state);
}

void print_resources() {
    const auto& resources = volta::runtime_resources();
    std::cout << "DV16 resources: regs/thread=" << resources.dv16.registers_per_thread
              << " active_blocks_per_sm=" << resources.dv16.active_blocks_per_sm
              << " dynamic_smem=" << resources.dv16.dynamic_smem << "\n";

    cudaFuncAttributes grouped_attrs{};
    int grouped_blocks = 0;
    if (cudaFuncGetAttributes(&grouped_attrs, volta::fused_grouped_dv16_kernel) == cudaSuccess &&
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &grouped_blocks, volta::fused_grouped_dv16_kernel,
            volta::GdnSchedule<16>::kThreads,
            volta::GroupedDv16SharedLayout::Bytes) == cudaSuccess) {
        std::cout << "grouped DV16 resources: regs/thread=" << grouped_attrs.numRegs
                  << " active_blocks_per_sm=" << grouped_blocks
                  << " dynamic_smem=" << volta::GroupedDv16SharedLayout::Bytes << "\n";
    }
}

} // namespace

int main() {
#if !defined(NINFER_VOLTA_BUILD)
    std::cerr << "GDN Volta fused test built outside NINFER_VOLTA_BUILD\n";
    return 1;
#else
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::cout << "SKIP: no CUDA device for Volta fused qualification\n";
        return 77;
    }

    cudaDeviceProp properties{};
    if (!cuda_ok(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties")) {
        return 1;
    }
    if (properties.major != 7 || properties.minor != 0) {
        std::cout << "SKIP: GDN Volta fused qualification requires SM70\n";
        return 77;
    }

    if (!cuda_ok(volta::initialize_runtime(), "initialize Volta GDN runtime")) {
        return 1;
    }
    print_resources();

    int failures = 0;
    failures += run_case(32, true);
    failures += run_case(volta::kWindowTokens, false);

    std::cout << (failures == 0 ? "PASS" : "FAIL")
              << ": Volta DV16 baseline/grouped FP64 sequential qualification\n";
    return failures == 0 ? 0 : 1;
#endif
}
