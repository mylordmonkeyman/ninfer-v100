#include "ninfer/ops/selected_block_attention.h"
#include "core/device.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {
using namespace ninfer;
constexpr int kPages = 512;
constexpr int kLogical = 512;
constexpr int kRows = 8;
constexpr std::size_t kCacheElements = std::size_t{kPages} * 2 * 64 * 256;

float bf16(std::uint16_t bits) { return std::bit_cast<float>(std::uint32_t{bits} << 16); }
std::uint16_t encode_bf16(float value) {
    std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    bits += 0x7fffU + ((bits >> 16) & 1U);
    return static_cast<std::uint16_t>(bits >> 16);
}
// Exact, independent E4M3FN decoding; the fixtures contain finite codes only.
float fp8(std::uint8_t bits) {
    const int exponent = (bits >> 3) & 15;
    const int mantissa = bits & 7;
    const float magnitude = exponent == 0 ? std::ldexp(float(mantissa), -9)
        : std::ldexp(float(8 + mantissa), exponent - 10);
    return (bits & 128) ? -magnitude : magnitude;
}

struct Fixture {
    std::vector<float> keys, values;
    std::vector<int> tables;
    DeviceBuffer device_keys, device_values, device_tables;
    ops::SelectedBlockAttentionCache cache;

    explicit Fixture(bool low_precision)
        : keys(kCacheElements), values(kCacheElements), tables(kLogical * kRows),
          device_keys(kCacheElements * (low_precision ? 1 : 2)),
          device_values(kCacheElements * (low_precision ? 1 : 2)),
          device_tables(tables.size() * sizeof(int)) {
        std::mt19937 random(1701);
        for (int row = 0; row < kRows; ++row) {
            for (int page = 0; page < kLogical; ++page) {
                tables[row * kLogical + page] = (page * 37 + row * 61) % kPages;
            }
        }
        auto populate = [&](std::vector<float>& decoded, DeviceBuffer& device) {
            if (low_precision) {
                std::vector<std::uint8_t> codes(kCacheElements);
                for (std::size_t i = 0; i < codes.size(); ++i) {
                    codes[i] = static_cast<std::uint8_t>((random() % 57) | ((random() & 1) << 7));
                    decoded[i] = fp8(codes[i]);
                }
                device.copy_from_host(codes.data(), codes.size());
            } else {
                std::vector<std::uint16_t> codes(kCacheElements);
                for (std::size_t i = 0; i < codes.size(); ++i) {
                    const float value = (int(random() % 2049) - 1024) / 1024.0F;
                    codes[i] = encode_bf16(value);
                    decoded[i] = bf16(codes[i]);
                }
                device.copy_from_host(codes.data(), codes.size() * 2);
            }
        };
        populate(keys, device_keys);
        populate(values, device_values);
        device_tables.copy_from_host(tables.data(), tables.size() * sizeof(int));
        const auto type = low_precision ? DType::FP8_E4M3FN : DType::BF16;
        cache = {Tensor(device_keys.p, type, {256, 64, 2, kPages}),
                 Tensor(device_values.p, type, {256, 64, 2, kPages}),
                 Tensor(device_tables.p, DType::I32, {kLogical, kRows})};
    }

    // Full FP64 dot, stable softmax, and weighted sum from represented inputs.
    // This never follows the production partitioning, casts, or reduction tree.
    std::vector<double> oracle(const std::vector<std::uint16_t>& query,
        const std::vector<int>& positions, const std::vector<int>& rows,
        const std::vector<int>& selected, const std::vector<int>& counts) const {
        std::vector<double> output(query.size(), 0.0);
        for (std::size_t b = 0; b < positions.size(); ++b) {
            std::vector<int> visible;
            for (int block = 0; block < counts[b]; ++block) {
                for (int offset = 0; offset < 4; ++offset) {
                    visible.push_back(selected[b * 512 + block] * 4 + offset);
                }
            }
            for (int token = (positions[b] + 1) / 4 * 4; token <= positions[b]; ++token) {
                visible.push_back(token);
            }
            if (visible.empty()) { continue; }
            for (int head = 0; head < 24; ++head) {
                std::vector<double> scores(visible.size());
                std::vector<std::size_t> offsets(visible.size());
                for (std::size_t t = 0; t < visible.size(); ++t) {
                    const int token = visible[t];
                    const int page = tables[rows[b] * kLogical + token / 64];
                    offsets[t] = ((std::size_t{static_cast<unsigned>(page)} * 2 + head / 12) * 64 + token % 64) * 256;
                    double dot = 0.0;
                    for (int dim = 0; dim < 256; ++dim) {
                        dot += double(bf16(query[(b * 24 + head) * 256 + dim])) * keys[offsets[t] + dim];
                    }
                    scores[t] = dot / 16.0;
                }
                const double maximum = *std::max_element(scores.begin(), scores.end());
                double denominator = 0.0;
                for (double& score : scores) { score = std::exp(score - maximum); denominator += score; }
                for (int dim = 0; dim < 256; ++dim) {
                    double numerator = 0.0;
                    for (std::size_t t = 0; t < visible.size(); ++t) {
                        numerator += scores[t] * values[offsets[t] + dim];
                    }
                    output[(b * 24 + head) * 256 + dim] = numerator / denominator;
                }
            }
        }
        return output;
    }
};

void run_case(Fixture& fixture, DeviceContext& device, bool low_precision, int batch,
              int position, int selected_count, float query_scale, bool graph, bool prefill = false) {
    std::mt19937 random(2026 + position + batch);
    std::vector<std::uint16_t> query(batch * 24 * 256);
    for (auto& value : query) {
        value = encode_bf16(query_scale * (int(random() % 2049) - 1024) / 1024.0F);
    }
    std::vector<int> positions(batch), rows(batch), counts(batch), selected(batch * 512, -1);
    for (int b = 0; b < batch; ++b) {
        positions[b] = position + b;
        rows[b] = prefill ? 1 : (b * 3 + 1) % kRows;
        const int blocks = (positions[b] + 1) / 4;
        counts[b] = std::min(selected_count, blocks);
        // Heterogeneous speculative rows, including the empty complete-block set.
        if (batch > 1 && b == batch - 1) { counts[b] = 0; }
        std::vector<int> candidates(blocks);
        std::iota(candidates.begin(), candidates.end(), 0);
        std::shuffle(candidates.begin(), candidates.end(), random);
        std::copy_n(candidates.begin(), counts[b], selected.begin() + b * 512);
    }
    DeviceBuffer dq(query.size() * 2), dout(query.size() * 2), dp(batch * 4), dr(batch * 4),
                 ds(selected.size() * 4), dc(batch * 4);
    dq.copy_from_host(query.data(), query.size() * 2);
    dp.copy_from_host(positions.data(), batch * 4);
    dr.copy_from_host(rows.data(), batch * 4);
    ds.copy_from_host(selected.data(), selected.size() * 4);
    dc.copy_from_host(counts.data(), batch * 4);
    device.synchronize();
    Tensor q(dq.p, DType::BF16, {256, 24, batch}), out(dout.p, DType::BF16, {256, 24, batch});
    Tensor p(dp.p, DType::I32, {batch}), r(dr.p, DType::I32, {batch}),
           s(ds.p, DType::I32, {512, batch}), c(dc.p, DType::I32, {batch});
    WorkspaceArena workspace(std::max<std::size_t>(256,
        prefill ? 0 : ops::selected_block_attention_workspace_capacity_bytes(batch)));
    auto invoke = [&] {
#if defined(NINFER_VOLTA_BUILD)
        if (prefill) {
            throw std::runtime_error(
                "selected block attention shared-row prefill overload is unavailable on SM70");
        }
        ops::selected_block_attention(q, p, r, s, c, fixture.cache, workspace, out, device.stream);
#else
        if (prefill) { ops::selected_block_attention(q, p, 1, s, c, fixture.cache, out, device.stream); }
        else { ops::selected_block_attention(q, p, r, s, c, fixture.cache, workspace, out, device.stream); }
#endif
    };
    auto verify = [&] {
        device.synchronize();
        std::vector<std::uint16_t> actual(query.size());
        dout.copy_to_host(actual.data(), actual.size() * 2);
        const auto expected = fixture.oracle(query, positions, rows, selected, counts);
        // BF16-output attention criterion: normwise FP32 accumulation/rounding
        // allowance, with a finite gross pointwise cap independent of cancellation.
        constexpr double kAttentionRelativeL2 = 0.004;
        constexpr double kAttentionAbsoluteRms = 0.00002;
        constexpr double kAttentionGrossRelative = 0.008;
        constexpr double kAttentionGrossAbsolute = 0.0002;
        double maximum = 0, squared_error = 0, squared_reference = 0, reference_maximum = 0;
        for (std::size_t i = 0; i < actual.size(); ++i) {
            const double value = bf16(actual[i]);
            const double error = std::abs(value - expected[i]);
            maximum = std::max(maximum, error);
            squared_error += error * error;
            squared_reference += expected[i] * expected[i];
            reference_maximum = std::max(reference_maximum, std::abs(expected[i]));
            if (!std::isfinite(value) || !std::isfinite(expected[i])) {
                std::fprintf(stderr, "oracle mismatch batch=%d position=%d idx=%zu actual=%g oracle=%g\n",
                             batch, position, i, value, expected[i]);
                throw std::runtime_error("selected block attention oracle mismatch");
            }
            const auto b = i / (24 * 256);
            if (counts[b] == 0 && (positions[b] + 1) % 4 == 0 && actual[i] != 0) {
                throw std::runtime_error("empty attention must produce exact zero");
            }
        }
        const double error_norm = std::sqrt(squared_error);
        const double reference_norm = std::sqrt(squared_reference);
        if (error_norm > kAttentionRelativeL2 * reference_norm +
                             kAttentionAbsoluteRms * std::sqrt(double(actual.size())) ||
            maximum > kAttentionGrossRelative * reference_maximum + kAttentionGrossAbsolute) {
            throw std::runtime_error("selected block attention normwise/gross oracle mismatch");
        }
        std::printf("oracle dtype=%s B=%d position=%d blocks=%d max_abs=%.7g rel_l2=%.7g PASS\n",
                    low_precision ? "fp8" : "bf16", batch, position, selected_count, maximum,
                    reference_norm > 0 ? error_norm / reference_norm : 0);
    };
    invoke();
    verify();
    if (graph) {
        cudaGraph_t definition{};
        cudaGraphExec_t executable{};
        CUDA_CHECK(cudaStreamBeginCapture(device.stream, cudaStreamCaptureModeThreadLocal));
        invoke();
        CUDA_CHECK(cudaStreamEndCapture(device.stream, &definition));
        CUDA_CHECK(cudaGraphInstantiate(&executable, definition, 0));
        // Reuse the captured addresses with different query values and visible sets.
        for (auto& value : query) { value = encode_bf16(-bf16(value)); }
        counts[0] = 0;
        dq.copy_from_host(query.data(), query.size() * 2);
        dc.copy_from_host(counts.data(), batch * 4);
        device.synchronize();
        CUDA_CHECK(cudaGraphLaunch(executable, device.stream));
        verify();
        CUDA_CHECK(cudaGraphExecDestroy(executable));
        CUDA_CHECK(cudaGraphDestroy(definition));
    } else {
        cudaEvent_t begin{}, end{};
        CUDA_CHECK(cudaEventCreate(&begin));
        CUDA_CHECK(cudaEventCreate(&end));
        constexpr int iterations = 30;
        CUDA_CHECK(cudaEventRecord(begin, device.stream));
        for (int i = 0; i < iterations; ++i) { invoke(); }
        CUDA_CHECK(cudaEventRecord(end, device.stream));
        CUDA_CHECK(cudaEventSynchronize(end));
        float elapsed{};
        CUDA_CHECK(cudaEventElapsedTime(&elapsed, begin, end));
        std::printf("timing dtype=%s B=%d position=%d blocks=%d us=%.3f\n",
                    low_precision ? "fp8" : "bf16", batch, position, selected_count, elapsed * 1000 / iterations);
        CUDA_CHECK(cudaEventDestroy(begin));
        CUDA_CHECK(cudaEventDestroy(end));
    }
}
} // namespace

int main(int argc, char** argv) {
    const bool prefill_only = argc > 1 && std::string_view(argv[1]) == "--prefill-only";
#if defined(NINFER_VOLTA_BUILD)
    if (prefill_only) {
        std::puts("SKIP: selected-block shared-row prefill overload is not built for SM70");
        return 77;
    }
#endif
    int devices = 0;
    const auto status = cudaGetDeviceCount(&devices);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver || devices == 0) { return 77; }
    try {
        CUDA_CHECK(status);
        ninfer::DeviceContext device(0);
        for (const bool fp8_storage : {true, false}) {
            Fixture fixture(fp8_storage);
            if (prefill_only) {
                for (const int position : {0, 3, 4, 54, 63, 256, 30293}) {
                    run_case(fixture, device, fp8_storage, 1, position, 512, 1.0F, false, true);
                }
                for (const int tokens : {8, 128}) {
                    run_case(fixture, device, fp8_storage, tokens, 30292, 512, 1.0F, false, true);
                }
                run_case(fixture, device, fp8_storage, 1, 30294, 512, 32.0F, false, true);
                run_case(fixture, device, fp8_storage, 5, 30292, 257, 1.0F, true, true);
                run_case(fixture, device, fp8_storage, 1, 3, 0, 1.0F, true, true);
                continue;
            }
            for (const int position : {0, 3, 4, 54, 63, 256, 30293}) {
                run_case(fixture, device, fp8_storage, 1, position, 512, 1.0F, false);
            }
            for (const int batch : {2, 5, 8}) {
                run_case(fixture, device, fp8_storage, batch, 30292, 512, 1.0F, false);
            }
            run_case(fixture, device, fp8_storage, 1, 30294, 512, 32.0F, false);
            run_case(fixture, device, fp8_storage, 5, 30292, 257, 1.0F, true);
            run_case(fixture, device, fp8_storage, 1, 3, 0, 1.0F, true);
        }
        std::puts("PASS selected_block_attention independent FP64 oracle and graph replay");
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
