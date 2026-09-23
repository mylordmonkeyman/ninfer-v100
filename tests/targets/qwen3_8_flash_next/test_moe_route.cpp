#include "core/arena.h"
#include "core/device.h"
#include "targets/qwen3_8_flash_next/impl/moe_route.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

namespace {

constexpr int kExperts = 512;
constexpr int kRows    = kExperts + 1; // + shared-expert gate
constexpr int kTopK    = 10;
constexpr int kHidden  = 2'560;

bool cuda_unavailable(cudaError_t error) {
    return error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver;
}

std::uint16_t to_bf16(float value) {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    return static_cast<std::uint16_t>((bits + 0x7FFFU + ((bits >> 16U) & 1U)) >> 16U);
}

float bf16_to_float(std::uint16_t bits) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16U);
}

ninfer::Weight bf16_weight(void* data, std::int32_t rows, std::int32_t columns) {
    ninfer::Weight out{};
    out.payload         = data;
    out.payload_bytes   = static_cast<std::uint64_t>(rows) * columns * 2;
    out.qdata           = data;
    out.qtype           = ninfer::QType::BF16_CTRL;
    out.layout          = ninfer::QuantLayout::Contiguous;
    out.n               = rows;
    out.k               = columns;
    out.ndim            = 2;
    out.shape[0]        = rows;
    out.shape[1]        = columns;
    out.padded_shape[0] = rows;
    out.padded_shape[1] = columns;
    return out;
}

// Double-precision oracle for the selection stage: given the FP32 scores the kernel worked from,
// the ids must be the top-10 by score (lower id first on equal scores), the alphas the top-10
// renormalised softmax, and the gate the sigmoid of row 512.
bool check_route_semantics(const std::vector<float>& scores, const std::vector<std::int32_t>& ids,
                           const std::vector<float>& alpha, const std::vector<float>& shared,
                           int tokens, const char* label) {
    for (int t = 0; t < tokens; ++t) {
        std::vector<double> p(kExperts, 0.0);
        double max_score = -1e30;
        for (int i = 0; i < kExperts; ++i) {
            max_score = std::max(max_score, static_cast<double>(scores[t * kRows + i]));
        }
        double sum_exp = 0.0;
        for (int i = 0; i < kExperts; ++i) {
            p[i] = std::exp(static_cast<double>(scores[t * kRows + i]) - max_score);
            sum_exp += p[i];
        }
        for (int i = 0; i < kExperts; ++i) { p[i] /= sum_exp; }

        std::vector<int> expected_ids(kTopK);
        std::vector<bool> used(kExperts, false);
        for (int rank = 0; rank < kTopK; ++rank) {
            int best_id = -1;
            for (int i = 0; i < kExperts; ++i) {
                if (used[i]) continue;
                if (best_id == -1 || p[i] > p[best_id] || (p[i] == p[best_id] && i < best_id)) {
                    best_id = i;
                }
            }
            expected_ids[rank] = best_id;
            used[best_id]      = true;
        }

        double selected_sum = 0.0;
        for (int rank = 0; rank < kTopK; ++rank) { selected_sum += p[expected_ids[rank]]; }

        float sum_alpha = 0.0F;
        for (int rank = 0; rank < kTopK; ++rank) {
            const int id               = ids[t * kTopK + rank];
            const float val            = alpha[t * kTopK + rank];
            sum_alpha += val;
            const float expected_alpha = static_cast<float>(p[expected_ids[rank]] / selected_sum);
            if (id != expected_ids[rank]) {
                std::cerr << "Flash-Next route (" << label << ") selected id " << id
                          << " expected " << expected_ids[rank] << " at token " << t << " rank "
                          << rank << " (tokens=" << tokens << ")\n";
                return false;
            }
            if (std::abs(val - expected_alpha) > 2e-6F) {
                std::cerr << "Flash-Next route (" << label << ") alpha " << val << " expected "
                          << expected_alpha << " at token " << t << " rank " << rank
                          << " (tokens=" << tokens << ")\n";
                return false;
            }
        }
        if (std::abs(sum_alpha - 1.0F) > 1e-5F) {
            std::cerr << "Flash-Next route (" << label << ") top-10 alpha sum " << sum_alpha
                      << " != 1.0 at token " << t << " (tokens=" << tokens << ")\n";
            return false;
        }

        const float expected_shared =
            1.0F / (1.0F + std::exp(-static_cast<float>(scores[t * kRows + kExperts])));
        if (std::abs(shared[t] - expected_shared) > 2e-6F) {
            std::cerr << "Flash-Next route (" << label << ") shared gate mismatch at token " << t
                      << ": got " << shared[t] << " expected " << expected_shared << "\n";
            return false;
        }
    }
    return true;
}

// Host model of route_projection_kernel's arithmetic, used to pin the FP32 score bit pattern the
// selection compares: legacy thread tid accumulates columns tid + 256*j (j = 0..9) with
// fmaf(w, x, acc) from 0; each warp of 32 accumulators is reduced with the shfl_down tree
// (offsets 16, 8, 4, 2, 1; lane l adds lane l+offset, or itself when l+offset >= 32); the eight
// warp sums sit in lanes 0..7 (zeros elsewhere) of a second identical tree.
float host_warp_sum_lane0(std::array<float, 32> lanes) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        std::array<float, 32> next{};
        for (int lane = 0; lane < 32; ++lane) {
            const float other = lane + offset < 32 ? lanes[lane + offset] : lanes[lane];
            next[lane]        = lanes[lane] + other;
        }
        lanes = next;
    }
    return lanes[0];
}

float host_route_score(const std::uint16_t* weight_row, const std::uint16_t* input_row) {
    std::array<float, 256> acc{};
    for (int tid = 0; tid < 256; ++tid) {
        float a = 0.0F;
        for (int j = 0; j < kHidden / 256; ++j) {
            const int column = tid + 256 * j;
            a = std::fmaf(bf16_to_float(weight_row[column]), bf16_to_float(input_row[column]), a);
        }
        acc[tid] = a;
    }
    std::array<float, 32> second{};
    for (int warp = 0; warp < 8; ++warp) {
        std::array<float, 32> lanes{};
        std::copy(acc.begin() + 32 * warp, acc.begin() + 32 * warp + 32, lanes.begin());
        second[warp] = host_warp_sum_lane0(lanes);
    }
    return host_warp_sum_lane0(second);
}

std::vector<float> host_route_scores(const std::vector<std::uint16_t>& input,
                                     const std::vector<std::uint16_t>& router,
                                     const std::vector<std::uint16_t>& gate, int tokens) {
    std::vector<float> scores(static_cast<std::size_t>(kRows) * tokens);
    for (int t = 0; t < tokens; ++t) {
        const std::uint16_t* input_row = input.data() + static_cast<std::size_t>(t) * kHidden;
        for (int row = 0; row < kRows; ++row) {
            const std::uint16_t* weight_row =
                row < kExperts ? router.data() + static_cast<std::size_t>(row) * kHidden
                               : gate.data();
            scores[static_cast<std::size_t>(t) * kRows + row] =
                host_route_score(weight_row, input_row);
        }
    }
    return scores;
}

struct RouteOutputs {
    std::vector<float> scores;
    std::vector<std::int32_t> ids;
    std::vector<float> alpha;
    std::vector<float> shared;
};

enum class RoutePath { Legacy, Fused, Launcher };

const char* route_path_name(RoutePath path) {
    switch (path) {
    case RoutePath::Legacy: return "legacy";
    case RoutePath::Fused: return "fused";
    default: return "launcher";
    }
}

// Runs one decode router path from host BF16 data. Every output buffer is pre-filled with 0xFF
// so that a byte the path fails to write cannot pass as equal to the other path's.
RouteOutputs run_route(ninfer::DeviceContext& device, RoutePath path,
                       const std::vector<std::uint16_t>& input,
                       const std::vector<std::uint16_t>& router,
                       const std::vector<std::uint16_t>& gate, int tokens) {
    ninfer::DeviceBuffer input_device(input.size() * 2);
    ninfer::DeviceBuffer router_device(router.size() * 2);
    ninfer::DeviceBuffer gate_device(gate.size() * 2);
    ninfer::DeviceBuffer score_device(sizeof(float) * kRows * tokens);
    ninfer::DeviceBuffer id_device(sizeof(std::int32_t) * kTopK * tokens);
    ninfer::DeviceBuffer alpha_device(sizeof(float) * kTopK * tokens);
    ninfer::DeviceBuffer shared_device(sizeof(float) * tokens);
    input_device.copy_from_host(input.data(), input.size() * 2);
    router_device.copy_from_host(router.data(), router.size() * 2);
    gate_device.copy_from_host(gate.data(), gate.size() * 2);
    score_device.fill(0xFF);
    id_device.fill(0xFF);
    alpha_device.fill(0xFF);
    shared_device.fill(0xFF);
    CUDA_CHECK(cudaDeviceSynchronize());

    ninfer::Tensor input_view(input_device.p, ninfer::DType::BF16, {kHidden, tokens});
    ninfer::Tensor score_view(score_device.p, ninfer::DType::FP32, {kRows, tokens});
    ninfer::Tensor id_view(id_device.p, ninfer::DType::I32, {kTopK, tokens});
    ninfer::Tensor alpha_view(alpha_device.p, ninfer::DType::FP32, {kTopK, tokens});
    ninfer::Tensor shared_view(shared_device.p, ninfer::DType::FP32, {tokens});
    const ninfer::Weight router_view = bf16_weight(router_device.p, kExperts, kHidden);
    const ninfer::Weight gate_view   = bf16_weight(gate_device.p, 1, kHidden);

    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    switch (path) {
    case RoutePath::Legacy:
        flash_next_route_decode_legacy(input_view, router_view, gate_view, score_view, id_view,
                                       alpha_view, shared_view, device.stream);
        break;
    case RoutePath::Fused:
        flash_next_route_decode_fused(input_view, router_view, gate_view, score_view, id_view,
                                      alpha_view, shared_view, device.stream);
        break;
    default:
        flash_next_route(input_view, router_view, gate_view, score_view, id_view, alpha_view,
                         shared_view, device.stream);
        break;
    }
    device.synchronize();

    RouteOutputs out;
    out.scores.resize(static_cast<std::size_t>(kRows) * tokens);
    out.ids.resize(static_cast<std::size_t>(kTopK) * tokens);
    out.alpha.resize(static_cast<std::size_t>(kTopK) * tokens);
    out.shared.resize(static_cast<std::size_t>(tokens));
    score_device.copy_to_host(out.scores.data(), out.scores.size() * sizeof(float));
    id_device.copy_to_host(out.ids.data(), out.ids.size() * sizeof(std::int32_t));
    alpha_device.copy_to_host(out.alpha.data(), out.alpha.size() * sizeof(float));
    shared_device.copy_to_host(out.shared.data(), out.shared.size() * sizeof(float));
    CUDA_CHECK(cudaDeviceSynchronize());
    return out;
}

template <class T>
bool bit_equal(const std::vector<T>& left, const std::vector<T>& right) {
    return left.size() == right.size() &&
           std::memcmp(left.data(), right.data(), left.size() * sizeof(T)) == 0;
}

bool report_bit_mismatch(const char* what, const char* left, const char* right, int tokens,
                         int seed) {
    std::cerr << "Flash-Next route " << what << " differ byte-for-byte between " << left
              << " and " << right << " (tokens=" << tokens << ", seed=" << seed << ")\n";
    return false;
}

bool bit_identical_outputs(const RouteOutputs& reference, const RouteOutputs& candidate,
                           const char* reference_name, const char* candidate_name, int tokens,
                           int seed) {
    if (!bit_equal(reference.scores, candidate.scores)) {
        return report_bit_mismatch("scores", reference_name, candidate_name, tokens, seed);
    }
    if (!bit_equal(reference.ids, candidate.ids)) {
        return report_bit_mismatch("ids", reference_name, candidate_name, tokens, seed);
    }
    if (!bit_equal(reference.alpha, candidate.alpha)) {
        return report_bit_mismatch("alphas", reference_name, candidate_name, tokens, seed);
    }
    if (!bit_equal(reference.shared, candidate.shared)) {
        return report_bit_mismatch("shared gates", reference_name, candidate_name, tokens, seed);
    }
    return true;
}

struct Lcg {
    std::uint32_t state;
    std::uint32_t next() {
        state = state * 1664525U + 1013904223U;
        return state;
    }
    // Uniform in [-1, 1) from the high 24 bits.
    float uniform() { return static_cast<float>(next() >> 8U) / 8388608.0F - 1.0F; }
};

} // namespace

bool test_route_for_tokens(ninfer::DeviceContext& device, int tokens) {
    std::vector<float> scores(kRows * tokens, 0.0F);
    for (int t = 0; t < tokens; ++t) {
        if (t == 0) {
            for (int expert = 10; expert < 20; ++expert) {
                scores[t * kRows + expert] = static_cast<float>(expert - 9);
            }
            scores[t * kRows + kExperts] = 2.0F;
        } else if (t == 1) {
            // All-zero tie: deterministic order must select lower expert IDs (0..9)
            scores[t * kRows + kExperts] = 0.0F;
        } else {
            std::uint32_t rng = static_cast<std::uint32_t>(1000 + t * 37);
            for (int expert = 0; expert < kExperts; ++expert) {
                rng = rng * 1664525U + 1013904223U;
                scores[t * kRows + expert] =
                    static_cast<float>((rng & 0xFFFFU)) / 65536.0F * 10.0F - 5.0F;
            }
            rng                          = rng * 1664525U + 1013904223U;
            scores[t * kRows + kExperts] = static_cast<float>((rng & 0xFFFFU)) / 65536.0F * 4.0F - 2.0F;
        }
    }

    ninfer::DeviceBuffer score_device(scores.size() * sizeof(float));
    ninfer::DeviceBuffer id_device(sizeof(std::int32_t) * kTopK * tokens);
    ninfer::DeviceBuffer alpha_device(sizeof(float) * kTopK * tokens);
    ninfer::DeviceBuffer shared_device(sizeof(float) * tokens);

    score_device.copy_from_host(scores.data(), scores.size() * sizeof(float));
    cudaDeviceSynchronize();
    ninfer::Tensor score_view(score_device.p, ninfer::DType::FP32, {kRows, tokens});
    ninfer::Tensor id_view(id_device.p, ninfer::DType::I32, {kTopK, tokens});
    ninfer::Tensor alpha_view(alpha_device.p, ninfer::DType::FP32, {kTopK, tokens});
    ninfer::Tensor shared_view(shared_device.p, ninfer::DType::FP32, {tokens});

    ninfer::targets::qwen3_8_flash_next::detail::flash_next_route_scores(
        score_view, id_view, alpha_view, shared_view, device.stream);
    device.synchronize();

    std::vector<std::int32_t> ids(kTopK * tokens);
    std::vector<float> alpha(kTopK * tokens);
    std::vector<float> shared(tokens);
    id_device.copy_to_host(ids.data(), ids.size() * sizeof(std::int32_t));
    alpha_device.copy_to_host(alpha.data(), alpha.size() * sizeof(float));
    shared_device.copy_to_host(shared.data(), shared.size() * sizeof(float));

    if (!check_route_semantics(scores, ids, alpha, shared, tokens, "route_scores")) {
        return false;
    }

    // Also test BF16 projection path: token 0 has x[0] = 1 so experts 10..19 score 1..10 and the
    // gate scores 2; every other token is all-zero, so it is an all-tie (ids 0..9).
    std::vector<std::uint16_t> input(static_cast<std::size_t>(kHidden) * tokens, 0);
    std::vector<std::uint16_t> router(static_cast<std::size_t>(kExperts) * kHidden, 0);
    std::vector<std::uint16_t> shared_gate(kHidden, 0);
    input[0]       = to_bf16(1.0F);
    shared_gate[0] = to_bf16(2.0F);
    for (int expert = 10; expert < 20; ++expert) {
        router[static_cast<std::size_t>(expert) * kHidden] = to_bf16(static_cast<float>(expert - 9));
    }
    ninfer::DeviceBuffer input_device(input.size() * 2);
    ninfer::DeviceBuffer router_device(router.size() * 2);
    ninfer::DeviceBuffer gate_device(shared_gate.size() * 2);
    ninfer::DeviceBuffer score_workspace(scores.size() * sizeof(float));
    input_device.copy_from_host(input.data(), input.size() * 2);
    router_device.copy_from_host(router.data(), router.size() * 2);
    gate_device.copy_from_host(shared_gate.data(), shared_gate.size() * 2);
    cudaDeviceSynchronize();
    ninfer::Tensor input_view(input_device.p, ninfer::DType::BF16, {kHidden, tokens});
    ninfer::Tensor score_workspace_view(score_workspace.p, ninfer::DType::FP32, {kRows, tokens});
    const ninfer::Weight router_view = bf16_weight(router_device.p, kExperts, kHidden);
    const ninfer::Weight gate_view   = bf16_weight(gate_device.p, 1, kHidden);
    ninfer::targets::qwen3_8_flash_next::detail::flash_next_route(
        input_view, router_view, gate_view, score_workspace_view, id_view, alpha_view, shared_view,
        device.stream);
    device.synchronize();

    // Decode shapes must reproduce route_projection_kernel's FP32 scores bit for bit and select
    // from them correctly. Larger T takes the prefill projection (tensor-core or tiled), whose
    // accumulation order is its own; that path is only exercised here, as before.
    if (tokens <= ninfer::targets::qwen3_8_flash_next::detail::kFlashNextRouteDecodeMaxTokens) {
        std::vector<float> projected(scores.size());
        score_workspace.copy_to_host(projected.data(), projected.size() * sizeof(float));
        id_device.copy_to_host(ids.data(), ids.size() * sizeof(std::int32_t));
        alpha_device.copy_to_host(alpha.data(), alpha.size() * sizeof(float));
        shared_device.copy_to_host(shared.data(), shared.size() * sizeof(float));
        CUDA_CHECK(cudaDeviceSynchronize());
        const std::vector<float> expected_projected =
            host_route_scores(input, router, shared_gate, tokens);
        if (!bit_equal(projected, expected_projected)) {
            std::cerr << "Flash-Next route projection differs from the host model (tokens="
                      << tokens << ")\n";
            return false;
        }
        if (!check_route_semantics(projected, ids, alpha, shared, tokens, "projection")) {
            return false;
        }
    }

    return true;
}

// Bit-exact gate for the decode router: legacy two-kernel path vs fused kernel vs whatever the
// launcher selects, on random BF16 activations and router weights. Rows 256..511 duplicate rows
// 0..255 (every score occurs twice, so each token's top-10 is five tie pairs that must come out
// lower id first) and rows 400..409 duplicate row 7 (a 12-way tie with 7 and 263). Per token the
// activation is random, a single non-zero column (scores are then the raw BF16 weight products,
// which collide often) or all zero (a 512-way tie); the seed rotates the roles so every (T, t)
// sees each one.
bool test_decode_bit_exact(ninfer::DeviceContext& device, int tokens, int seed) {
    Lcg rng{static_cast<std::uint32_t>(0x9E37'79B9U + 7'919U * seed + 104'729U * tokens)};
    std::vector<std::uint16_t> router(static_cast<std::size_t>(kExperts) * kHidden);
    std::vector<std::uint16_t> gate(kHidden);
    std::vector<std::uint16_t> input(static_cast<std::size_t>(kHidden) * tokens, 0);
    for (int expert = 0; expert < kExperts / 2; ++expert) {
        for (int column = 0; column < kHidden; ++column) {
            router[static_cast<std::size_t>(expert) * kHidden + column] =
                to_bf16(rng.uniform() * 0.05F);
        }
    }
    auto copy_row = [&](int destination, int source) {
        std::copy(router.begin() + static_cast<std::ptrdiff_t>(source) * kHidden,
                  router.begin() + static_cast<std::ptrdiff_t>(source + 1) * kHidden,
                  router.begin() + static_cast<std::ptrdiff_t>(destination) * kHidden);
    };
    for (int expert = kExperts / 2; expert < kExperts; ++expert) {
        copy_row(expert, expert - kExperts / 2);
    }
    for (int expert = 400; expert < 410; ++expert) { copy_row(expert, 7); }
    std::copy(router.begin() + 3 * kHidden, router.begin() + 4 * kHidden, gate.begin());

    std::vector<int> role(tokens);
    for (int t = 0; t < tokens; ++t) {
        role[t]                   = (t + tokens - 1 + seed) % 3;
        std::uint16_t* activation = input.data() + static_cast<std::size_t>(t) * kHidden;
        if (role[t] == 0) {
            for (int column = 0; column < kHidden; ++column) {
                activation[column] = to_bf16(rng.uniform() * 2.0F);
            }
        } else if (role[t] == 1) {
            activation[(37 * t + 5 + 101 * seed) % kHidden] = to_bf16(1.5F);
        }
    }

    const std::vector<float> expected_scores = host_route_scores(input, router, gate, tokens);
    const RouteOutputs legacy = run_route(device, RoutePath::Legacy, input, router, gate, tokens);
    if (!bit_equal(legacy.scores, expected_scores)) {
        std::cerr << "Flash-Next legacy route scores differ from the host model of "
                     "route_projection_kernel (tokens="
                  << tokens << ", seed=" << seed << ")\n";
        return false;
    }
    if (!check_route_semantics(legacy.scores, legacy.ids, legacy.alpha, legacy.shared, tokens,
                               "legacy")) {
        return false;
    }
    // The tie scenario must actually be present: a random token's top-10 holds equal scores at
    // adjacent ranks (mirrored rows), and an all-zero token selects ids 0..9.
    for (int t = 0; t < tokens; ++t) {
        if (role[t] == 0) {
            bool tie = false;
            for (int rank = 0; rank + 1 < kTopK; ++rank) {
                const float left  = legacy.scores[t * kRows + legacy.ids[t * kTopK + rank]];
                const float right = legacy.scores[t * kRows + legacy.ids[t * kTopK + rank + 1]];
                tie = tie || left == right;
            }
            if (!tie) {
                std::cerr << "Flash-Next route tie scenario did not produce a tie in the top-10 "
                             "(tokens="
                          << tokens << ", token=" << t << ", seed=" << seed << ")\n";
                return false;
            }
        } else if (role[t] == 2) {
            for (int rank = 0; rank < kTopK; ++rank) {
                if (legacy.ids[t * kTopK + rank] != rank) {
                    std::cerr << "Flash-Next route all-zero token selected id "
                              << legacy.ids[t * kTopK + rank] << " at rank " << rank
                              << " (tokens=" << tokens << ", seed=" << seed << ")\n";
                    return false;
                }
            }
        }
    }

    for (const RoutePath path : {RoutePath::Fused, RoutePath::Launcher}) {
        const RouteOutputs candidate = run_route(device, path, input, router, gate, tokens);
        if (!bit_identical_outputs(legacy, candidate, route_path_name(RoutePath::Legacy),
                                   route_path_name(path), tokens, seed)) {
            return false;
        }
    }
    return true;
}

int main() {
    int device_count              = 0;
    const cudaError_t count_error = cudaGetDeviceCount(&device_count);
    if (cuda_unavailable(count_error) || device_count == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    CUDA_CHECK(count_error);

    ninfer::DeviceContext device(0);

    const std::vector<int> test_tokens = {1, 2, 4, 8, 16, 32, 64};
    for (int t_val : test_tokens) {
        if (!test_route_for_tokens(device, t_val)) {
            std::cerr << "FAIL: test_moe_route failed for tokens=" << t_val << "\n";
            return 1;
        }
    }

    using ninfer::targets::qwen3_8_flash_next::detail::flash_next_route_legacy_enabled;
    using ninfer::targets::qwen3_8_flash_next::detail::kFlashNextRouteDecodeMaxTokens;
    std::cout << "launcher decode path: "
              << (flash_next_route_legacy_enabled() ? "legacy (NINFER_FLASH_NEXT_ROUTE_LEGACY=1)"
                                                    : "fused")
              << "\n";
    for (int tokens = 1; tokens <= kFlashNextRouteDecodeMaxTokens; ++tokens) {
        for (int seed = 0; seed < 3; ++seed) {
            if (!test_decode_bit_exact(device, tokens, seed)) {
                std::cerr << "FAIL: test_moe_route bit-exact decode gate failed for tokens="
                          << tokens << " seed=" << seed << "\n";
                return 1;
            }
        }
    }

    std::cout << "PASS: test_moe_route\n";
    return 0;
}
