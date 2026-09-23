#include "ninfer/ops/linear.h"

#include "core/decode_graph.h"
#include "core/device.h"
#include "ops/direct_bf16_weight.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <span>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::direct_bf16_weight;

constexpr ReductionCriterion kA16Tolerance{1.0 / 256.0, 1.0 / 256.0, 2.0 / 256.0};

std::vector<std::uint16_t> make_activation_bits(std::int32_t hidden, std::int32_t tokens) {
    std::vector<std::uint16_t> result(static_cast<std::size_t>(hidden) * tokens);
    for (std::int32_t token = 0; token < tokens; ++token) {
        for (std::int32_t column = 0; column < hidden; ++column) {
            const int centered = ((column * 29 + token * 71 + 17) & 0xff) - 128;
            result[static_cast<std::size_t>(token) * hidden + column] =
                f32_to_bf16(static_cast<float>(centered) * (1.0F / 512.0F));
        }
    }
    return result;
}

std::vector<float> materialize(std::span<const std::uint16_t> bits) {
    std::vector<float> result(bits.size());
    for (std::size_t index = 0; index < bits.size(); ++index) {
        result[index] = bf16_to_f32(bits[index]);
    }
    return result;
}

std::vector<double> oracle_all_rows(const HostWeight& weight, std::span<const float> activation) {
    std::vector<double> result(static_cast<std::size_t>(weight.n));
    const unsigned available   = std::max(1U, std::thread::hardware_concurrency());
    const std::int32_t threads = std::min(weight.n, static_cast<std::int32_t>(available));
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(threads));
    for (std::int32_t thread = 0; thread < threads; ++thread) {
        const std::int32_t begin =
            static_cast<std::int32_t>((static_cast<std::int64_t>(weight.n) * thread) / threads);
        const std::int32_t end = static_cast<std::int32_t>(
            (static_cast<std::int64_t>(weight.n) * (thread + 1)) / threads);
        workers.emplace_back([&, begin, end] {
            for (std::int32_t row = begin; row < end; ++row) {
                result[static_cast<std::size_t>(row)] = dot_fp64(weight, row, activation);
            }
        });
    }
    for (std::thread& worker : workers) { worker.join(); }
    return result;
}

std::vector<std::int32_t> sampled_rows(std::int32_t rows) {
    std::vector<std::int32_t> result{0, 1, rows / 4, rows / 2, (3 * rows) / 4, rows - 2, rows - 1};
    if (rows == 14336) {
        result.insert(result.end(), {1023, 6143, 6144, 7167, 7168, 13311, 13312});
    }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

std::vector<std::int32_t> sampled_tokens(std::int32_t tokens) {
    if (tokens <= 32) {
        std::vector<std::int32_t> result(static_cast<std::size_t>(tokens));
        for (std::int32_t token = 0; token < tokens; ++token) {
            result[static_cast<std::size_t>(token)] = token;
        }
        return result;
    }
    std::vector<std::int32_t> result{0, 1, tokens / 2, tokens - 2, tokens - 1};
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

int run_bf16_linear_case(DeviceWeight& weight, std::int32_t tokens,
                         bool use_convenience_entry = false) {
    const std::int32_t rows                          = weight.host.n;
    const std::int32_t hidden                        = weight.host.k;
    const std::vector<std::uint16_t> activation_bits = make_activation_bits(hidden, tokens);
    const std::vector<float> activation              = materialize(activation_bits);
    DeviceBuffer device_activation                   = to_device(activation_bits);
    GuardedDeviceBuffer guarded_output(static_cast<std::size_t>(rows) * tokens *
                                       sizeof(std::uint16_t));
    guarded_output.fill(0xff);

    Tensor x(device_activation.p, DType::BF16, {hidden, tokens});
    Tensor output(guarded_output.data(), DType::BF16, {rows, tokens});
    if (use_convenience_entry) {
        ops::linear(x, weight.view(), output, nullptr);
    } else {
        DeviceArena workspace(256);
        ops::linear(x, weight.view(), output, ops::LinearPolicy::A16Only, workspace, nullptr);
    }
    cuda_synchronize();

    const std::string suffix = " T=" + std::to_string(tokens);
    int failures             = guarded_output.verify_guards("BF16_A16 Linear output" + suffix);
    const std::vector<std::uint16_t> output_bits =
        from_device<std::uint16_t>(guarded_output.data(), static_cast<std::size_t>(rows) * tokens);
    for (std::size_t index = 0; index < output_bits.size(); ++index) {
        const std::uint16_t bits = output_bits[index];
        if (!std::isfinite(bf16_to_f32(bits))) {
            std::cerr << "BF16_A16 Linear output" << suffix << " element " << index
                      << " is not finite\n";
            ++failures;
            break;
        }
    }

    std::vector<double> actual;
    std::vector<double> expected;
    if (tokens == 1) {
        const std::vector<double> complete =
            oracle_all_rows(weight.host, std::span<const float>(activation));
        actual.reserve(rows);
        for (const std::uint16_t bits : output_bits) { actual.push_back(bf16_to_f32(bits)); }
        expected = complete;
    } else {
        const std::vector<std::int32_t> sampled       = sampled_rows(rows);
        const std::vector<std::int32_t> token_samples = sampled_tokens(tokens);
        actual.reserve(sampled.size() * token_samples.size());
        expected.reserve(actual.capacity());
        for (const std::int32_t row : sampled) {
            for (const std::int32_t token : token_samples) {
                actual.push_back(
                    bf16_to_f32(output_bits[static_cast<std::size_t>(token) * rows + row]));
                expected.push_back(dot_fp64(
                    weight.host, row,
                    std::span<const float>(
                        activation.data() + static_cast<std::size_t>(token) * hidden, hidden)));
            }
        }
    }
    failures += verify_reduction("BF16_A16 Linear [" + std::to_string(rows) + "," +
                                     std::to_string(hidden) + "]" + suffix,
                                 actual, expected, kA16Tolerance);
    const std::vector<std::uint16_t> activation_after =
        from_device<std::uint16_t>(device_activation, activation_bits.size());
    if (activation_after != activation_bits) {
        std::cerr << "BF16_A16 Linear" << suffix << " modified its activation\n";
        ++failures;
    }
    failures += weight.verify_preserved("BF16_A16 Linear weight" + suffix);
    return failures;
}

int verify_vision_extent_contract(std::int32_t rows, std::int32_t hidden, bool raw_patch) {
    int failures                = 0;
    const auto expect_supported = [&](std::int32_t tokens) {
        try {
            (void)ops::linear_workspace_capacity_bytes(QType::BF16_CTRL, rows, hidden,
                                                       ops::LinearPolicy::A16Only, tokens, tokens);
        } catch (const std::exception& error) {
            std::cerr << "BF16_A16 Vision [" << rows << ',' << hidden << "] T=" << tokens
                      << " should be supported: " << error.what() << '\n';
            ++failures;
        }
    };
    const auto expect_rejected = [&](std::int32_t tokens) {
        try {
            (void)ops::linear_workspace_capacity_bytes(QType::BF16_CTRL, rows, hidden,
                                                       ops::LinearPolicy::A16Only, tokens, tokens);
            std::cerr << "BF16_A16 Vision [" << rows << ',' << hidden << "] T=" << tokens
                      << " should be rejected\n";
            ++failures;
        } catch (const std::invalid_argument&) {}
    };

    if (raw_patch) {
        expect_supported(4);
        expect_supported(131072);
        for (const std::int32_t tokens : {1, 3, 5, 131073}) { expect_rejected(tokens); }
    } else {
        expect_supported(1);
        expect_supported(32768);
        expect_rejected(32769);
    }
    return failures;
}

int run_selector_linear() {
    constexpr int n = 256, k = 5120, max_t = 1024;
    DeviceWeight weight(make_patterned(n, k, 419U));
    const auto bits       = make_activation_bits(k, max_t);
    const auto activation = materialize(bits);
    std::vector<double> oracle(n * max_t);
    std::vector<std::thread> workers;
    const int threads = std::min(32U, std::max(1U, std::thread::hardware_concurrency()));
    for (int worker = 0; worker < threads; ++worker)
        workers.emplace_back([&, worker] {
            for (int index = worker; index < n * max_t; index += threads) {
                const int token = index / n, row = index % n;
                oracle[index] = dot_fp64(weight.host, row,
                                         std::span<const float>(activation.data() + token * k, k));
            }
        });
    for (auto& worker : workers) worker.join();
    DeviceBuffer input = to_device(bits);
    auto negative      = bits;
    for (auto& value : negative) value ^= 0x8000;
    DeviceContext context;
    int failures   = 0;
    const auto run = [&](int tokens, bool replay) {
        const auto capacity = ops::linear_workspace_capacity_bytes(
            QType::BF16_CTRL, n, k, ops::LinearPolicy::A16Only, tokens, tokens);
        DeviceArena scratch(std::max<std::size_t>(capacity, 256));
        GuardedDeviceBuffer output_buffer(static_cast<std::size_t>(n) * tokens * 2);
        Tensor x(input.p, DType::BF16, {k, tokens});
        Tensor output(output_buffer.data(), DType::BF16, {n, tokens});
        const auto launch = [&] {
            ops::linear(x, weight.view(), output, ops::LinearPolicy::A16Only, scratch,
                        context.stream);
        };
        DecodeGraphDefinition definition;
        DecodeGraphExecutable graph;
        cuda_synchronize();
        if (replay) {
            definition.capture(context.stream, launch);
            graph.instantiate(definition);
        }
        const std::string label =
            "BF16 selector T=" + std::to_string(tokens) + (replay ? " graph" : " eager");
        for (int phase = 0; phase < (replay ? 2 : 1); ++phase) {
            const auto& represented = phase == 0 ? bits : negative;
            if (phase) input.copy_from_host(represented.data(), input.bytes);
            output_buffer.fill(0xff);
            cuda_synchronize();
            if (replay)
                graph.launch(context.stream);
            else
                launch();
            cuda_synchronize(context.stream);
            failures += output_buffer.verify_guards(label);
            if (scratch.peak_used() > capacity || scratch.used() != 0) {
                std::cerr << label << ": workspace query/scope mismatch\n";
                ++failures;
            }
            const auto actual_bits = from_device<std::uint16_t>(output_buffer.data(), n * tokens);
            std::vector<double> actual(actual_bits.size()),
                expected(oracle.begin(), oracle.begin() + n * tokens);
            for (std::size_t i = 0; i < actual.size(); ++i) actual[i] = bf16_to_f32(actual_bits[i]);
            if (phase)
                for (auto& value : expected) value = -value;
            failures += verify_reduction(label, actual, expected, kA16Tolerance);
            if (from_device<std::uint16_t>(input, bits.size()) != represented) {
                std::cerr << label << ": modified input\n";
                ++failures;
            }
        }
        if (replay) input.copy_from_host(bits.data(), input.bytes);
    };
    for (int tokens = 1; tokens <= 120; ++tokens) run(tokens, false);
    for (int tokens : {121, 127, 128, 129, 256, 1024}) run(tokens, false);
    for (int tokens : {1, 7, 8, 15, 16, 63, 64, 65, 76, 77, 80, 81, 119, 120, 129, 1024})
        run(tokens, true);
    failures += weight.verify_preserved("BF16 selector weight");
    return failures;
}

int run_bf16_linear() {
    int failures = 0;
    DeviceWeight attention_weight(make_patterned(14336, 5120, 401U));
    for (const std::int32_t tokens : {1, 2, 4, 8, 16, 17, 27, 28, 32, 33, 128, 129, 1024}) {
        failures += run_bf16_linear_case(attention_weight, tokens);
    }
    DeviceWeight output_weight(make_patterned(5120, 6144, 409U));
    for (const std::int32_t tokens : {1, 2, 4, 8, 16, 27, 28, 32, 33, 127, 128, 129, 1024, 1536}) {
        failures += run_bf16_linear_case(output_weight, tokens);
    }
    DeviceWeight ple_key_weight(make_patterned(10240, 2560, 419U));
    for (const std::int32_t tokens : {1, 2, 4, 8, 9, 128, 1024}) {
        failures += run_bf16_linear_case(ple_key_weight, tokens);
    }
    DeviceWeight ple_value_weight(make_patterned(2560, 2560, 421U));
    for (const std::int32_t tokens : {1, 2, 4, 8, 9, 128, 1024}) {
        failures += run_bf16_linear_case(ple_value_weight, tokens);
    }
    DeviceWeight qsa_indexer_weight(make_patterned(640, 2560, 423U));
    for (const std::int32_t tokens : {1, 2, 4, 8, 9, 128, 1024}) {
        failures += run_bf16_linear_case(qsa_indexer_weight, tokens);
    }
    DeviceWeight mtp_qgkv_weight(make_patterned(13312, 2560, 427U));
    for (const std::int32_t tokens : {1, 4, 8, 9, 127, 128, 129, 8191}) {
        failures += run_bf16_linear_case(mtp_qgkv_weight, tokens);
    }
    DeviceWeight shared_down_weight(make_patterned(2560, 640, 424U));
    for (const std::int32_t tokens : {1, 2, 4, 8, 9, 128, 1024}) {
        failures += run_bf16_linear_case(shared_down_weight, tokens);
    }
    DeviceWeight output_head_weight(make_patterned(248320, 2560, 425U));
    for (const std::int32_t tokens : {1, 2, 4, 8}) {
        failures += run_bf16_linear_case(output_head_weight, tokens);
    }
    DeviceWeight vision_patch_weight(make_patterned(1152, 1536, 431U));
    for (const std::int32_t tokens : {4, 16, 64, 128, 256, 576, 1024}) {
        failures += run_bf16_linear_case(vision_patch_weight, tokens, tokens == 4);
    }
    failures += verify_vision_extent_contract(1152, 1536, true);
    DeviceWeight vision_qkv_weight(make_patterned(3456, 1152, 433U));
    for (const std::int32_t tokens : {4, 16, 64, 128, 256, 576, 1024}) {
        failures += run_bf16_linear_case(vision_qkv_weight, tokens, tokens == 4);
    }
    failures += verify_vision_extent_contract(3456, 1152, true);
    DeviceWeight vision_proj_weight(make_patterned(1152, 1152, 435U));
    for (const std::int32_t tokens : {4, 16, 64, 128, 256, 576, 1024}) {
        failures += run_bf16_linear_case(vision_proj_weight, tokens, tokens == 4);
    }
    failures += verify_vision_extent_contract(1152, 1152, true);
    DeviceWeight vision_fc1_weight(make_patterned(4304, 1152, 437U));
    for (const std::int32_t tokens : {4, 16, 64, 128, 256, 576, 1024}) {
        failures += run_bf16_linear_case(vision_fc1_weight, tokens, tokens == 4);
    }
    failures += verify_vision_extent_contract(4304, 1152, true);
    DeviceWeight vision_fc2_weight(make_patterned(1152, 4304, 439U));
    for (const std::int32_t tokens : {4, 16, 64, 128, 256, 576, 1024}) {
        failures += run_bf16_linear_case(vision_fc2_weight, tokens, tokens == 4);
    }
    failures += verify_vision_extent_contract(1152, 4304, true);
    DeviceWeight vision_merger_fc1_weight(make_patterned(4608, 4608, 441U));
    for (const std::int32_t tokens : {1, 4, 16, 64, 128, 256, 576}) {
        failures += run_bf16_linear_case(vision_merger_fc1_weight, tokens, tokens == 1);
    }
    failures += verify_vision_extent_contract(4608, 4608, false);
    DeviceWeight vision_merger_fc2_weight(make_patterned(2560, 4608, 443U));
    for (const std::int32_t tokens : {1, 4, 16, 64, 128, 256, 576}) {
        failures += run_bf16_linear_case(vision_merger_fc2_weight, tokens, tokens == 1);
    }
    failures += verify_vision_extent_contract(2560, 4608, false);
    failures += run_selector_linear();
    return failures;
}

} // namespace

int main(int argc, char** argv) {
    if (ninfer::test::cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    try {
        int failures = 0;
        if (argc == 2 && std::string_view(argv[1]) == "--orcarouter-only") {
            DeviceWeight head(make_patterned(248320, 5120, 449U));
            for (const int tokens : {1, 2, 8, 9, 33, 129}) {
                failures += run_bf16_linear_case(head, tokens);
            }
        } else {
            failures = run_bf16_linear();
        }
        std::cout << (failures == 0 ? "OK" : "FAIL") << " BF16_A16 Linear\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "BF16_A16 Linear: " << error.what() << '\n';
        return 1;
    }
}
