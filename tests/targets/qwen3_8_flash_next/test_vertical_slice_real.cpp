#include "core/device.h"
#include "targets/qwen3_8_flash_next/impl/gdn.h"
#include "targets/qwen3_8_flash_next/impl/gdn_kernels.h"
#include "targets/qwen3_8_flash_next/impl/gdn_workspace.h"
#include "targets/qwen3_8_flash_next/impl/load/loader.h"
#include "targets/qwen3_8_flash_next/impl/moe.h"
#include "targets/qwen3_8_flash_next/impl/runtime_state.h"
#include "targets/qwen3_8_flash_next/impl/text_decode.h"
#include "targets/qwen3_8_flash_next/impl/text_decode_kernels.h"
#include "targets/qwen3_8_flash_next/impl/text_executor.h"
#include "targets/qwen3_8_flash_next/impl/vertical_slice.h"

#include "nlohmann/json.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ninfer::targets::qwen3_8_flash_next::detail;
namespace fs = std::filesystem;
using json = nlohmann::json;

struct OracleRecord {
    std::uint32_t position = 0;
    std::int32_t token_id = 0;
    fs::path logits_path;
    std::size_t logits_count = 0;
};

bool cuda_unavailable(cudaError_t error) {
    return error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver;
}

std::uint32_t round_up_128(std::uint64_t value) {
    const std::uint64_t rounded = (value + 127ULL) & ~127ULL;
    if (rounded > 262'144ULL) {
        throw std::invalid_argument(
            "Phase 11 oracle exceeds Flash-Next maximum context");
    }
    return static_cast<std::uint32_t>(rounded);
}

std::uint32_t required_oracle_positions() {
    constexpr std::uint32_t kPhase11Minimum = 4'096;
    const char* smoke_env =
        std::getenv("NINFER_PHASE11_ORACLE_SMOKE_POSITIONS");
    if (smoke_env == nullptr || smoke_env[0] == '\0') {
        return kPhase11Minimum;
    }

    char* end = nullptr;
    const unsigned long parsed = std::strtoul(smoke_env, &end, 10);
    if (end == smoke_env || *end != '\0' || parsed < 2 ||
        parsed >= kPhase11Minimum) {
        throw std::invalid_argument(
            "NINFER_PHASE11_ORACLE_SMOKE_POSITIONS must be in [2, 4095]");
    }
    return static_cast<std::uint32_t>(parsed);
}

std::vector<OracleRecord> load_manifest(const fs::path& manifest_path) {
    std::ifstream input(manifest_path);
    if (!input) {
        throw std::runtime_error(
            "Phase 11 cannot open oracle manifest: " + manifest_path.string());
    }
    json root;
    input >> root;
    const auto& positions = root.at("positions");
    if (!positions.is_array()) {
        throw std::runtime_error("Phase 11 oracle manifest positions must be an array");
    }

    std::vector<OracleRecord> records;
    records.reserve(positions.size());
    for (std::size_t index = 0; index < positions.size(); ++index) {
        const auto& position = positions.at(index);
        OracleRecord record{};
        record.position = position.at("position").get<std::uint32_t>();
        record.token_id = position.at("token_id").get<std::int32_t>();
        if (record.position != index) {
            throw std::runtime_error(
                "Phase 11 oracle positions must be contiguous from zero");
        }

        bool found = false;
        for (const auto& tensor : position.at("tensors")) {
            if (tensor.at("name").get<std::string>() != "logits") { continue; }
            if (tensor.at("dtype").get<std::string>() != "FP32") {
                throw std::runtime_error("Phase 11 oracle logits must be FP32");
            }
            const auto shape = tensor.at("shape");
            if (!shape.is_array() || shape.size() != 1) {
                throw std::runtime_error(
                    "Phase 11 oracle logits must be a rank-1 vector");
            }
            record.logits_count = shape.at(0).get<std::size_t>();
            record.logits_path =
                manifest_path.parent_path() / tensor.at("file").get<std::string>();
            const std::uint64_t expected_bytes =
                static_cast<std::uint64_t>(record.logits_count) * sizeof(float);
            if (tensor.contains("bytes") &&
                tensor.at("bytes").get<std::uint64_t>() != expected_bytes) {
                throw std::runtime_error(
                    "Phase 11 oracle logits byte count does not match shape");
            }
            found = true;
            break;
        }
        if (!found) {
            throw std::runtime_error(
                "Phase 11 oracle position is missing logits tensor");
        }
        records.push_back(std::move(record));
    }
    return records;
}

std::vector<float> load_fp32_logits(const OracleRecord& record) {
    std::ifstream input(record.logits_path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error(
            "Phase 11 cannot open oracle logits: " + record.logits_path.string());
    }
    const auto end = input.tellg();
    const std::uint64_t expected_bytes =
        static_cast<std::uint64_t>(record.logits_count) * sizeof(float);
    if (end < 0 || static_cast<std::uint64_t>(end) != expected_bytes) {
        throw std::runtime_error(
            "Phase 11 oracle logits file size does not match manifest");
    }
    input.seekg(0);
    std::vector<float> values(record.logits_count);
    input.read(reinterpret_cast<char*>(values.data()),
               static_cast<std::streamsize>(expected_bytes));
    if (!input) {
        throw std::runtime_error("Phase 11 failed to read oracle logits");
    }
    return values;
}

float bf16_to_float(std::uint16_t word) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(word) << 16U);
}

std::int32_t lower_id_argmax(std::span<const float> values) {
    if (values.empty()) {
        throw std::invalid_argument("argmax requires non-empty logits");
    }
    float best_value = values[0];
    std::int32_t best_id = 0;
    for (std::size_t i = 1; i < values.size(); ++i) {
        if (values[i] > best_value) {
            best_value = values[i];
            best_id = static_cast<std::int32_t>(i);
        }
    }
    return best_id;
}

double gib(std::uint64_t bytes) {
    return static_cast<double>(bytes) / static_cast<double>(1ULL << 30U);
}

void print_metrics(const Phase11OracleMetrics& metrics) {
    std::cout << std::fixed << std::setprecision(8)
              << "phase11.positions=" << metrics.positions << '\n'
              << "phase11.nonfinite_positions=" << metrics.nonfinite_positions << '\n'
              << "phase11.top1_agreement=" << metrics.top1_agreement << '\n'
              << "phase11.mean_kl=" << metrics.mean_kl << '\n'
              << "phase11.p99_kl=" << metrics.p99_kl << '\n'
              << "phase11.relative_mean_nll_delta="
              << metrics.relative_mean_nll_delta << '\n'
              << "phase11.mean_top5_overlap=" << metrics.mean_top5_overlap << '\n'
              << "phase11.mean_top10_overlap=" << metrics.mean_top10_overlap << '\n'
              << "phase11.maximum_logit_error="
              << metrics.maximum_logit_error << '\n';
    if (metrics.first_top1_divergence) {
        const auto& d = *metrics.first_top1_divergence;
        std::cout << "phase11.first_divergence.position=" << d.position << '\n'
                  << "phase11.first_divergence.candidate_top1="
                  << d.candidate_top1 << '\n'
                  << "phase11.first_divergence.oracle_top1="
                  << d.oracle_top1 << '\n';
    }
    if (metrics.worst_kl_divergence) {
        const auto& d = *metrics.worst_kl_divergence;
        std::cout << "phase11.worst_kl.position=" << d.position << '\n'
                  << "phase11.worst_kl.value=" << d.kl_divergence << '\n';
    }
}

void print_vram(const FlashNextStaticVramLedger& ledger,
                const Phase11VramReconciliation& actual) {
    std::cout << std::fixed << std::setprecision(3)
              << "phase11.vram.planned_nonexpert_weights_gib="
              << gib(actual.planned_nonexpert_weight_bytes) << '\n'
              << "phase11.vram.planned_runtime_gib="
              << gib(actual.planned_runtime_bytes) << '\n'
              << "phase11.vram.planned_total_gib="
              << gib(actual.planned_total_device_bytes) << '\n'
              << "phase11.vram.observed_model_load_gib="
              << gib(actual.observed_model_load_bytes) << '\n'
              << "phase11.vram.observed_runtime_gib="
              << gib(actual.observed_runtime_bytes) << '\n'
              << "phase11.vram.observed_total_gib="
              << gib(actual.observed_total_bytes) << '\n'
              << "phase11.vram.observed_minus_planned_mib="
              << static_cast<double>(actual.observed_minus_planned_bytes) /
                     static_cast<double>(1ULL << 20U)
              << '\n'
              << "phase11.vram.host_experts_gib="
              << gib(ledger.host_backed_expert_payload_bytes) << '\n'
              << "phase11.vram.host_expert_layers="
              << ledger.host_backed_expert_layers << '\n'
              << "phase11.vram.resident_routed_expert_layers="
              << ledger.routed_expert_layers << '\n';
}

struct GdnConvIsolationMetrics {
    double cosine = 0.0;
    double nrmse = 0.0;
    double max_error = 0.0;
};

std::vector<float> load_stage_fp32_vector(const fs::path& path, std::size_t expected_count) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error(
            "Phase 11 GDN isolation cannot open stage oracle: " + path.string());
    }
    const auto end = input.tellg();
    const std::uint64_t expected_bytes =
        static_cast<std::uint64_t>(expected_count) * sizeof(float);
    if (end < 0 || static_cast<std::uint64_t>(end) != expected_bytes) {
        throw std::runtime_error(
            "Phase 11 GDN isolation stage oracle has unexpected size: " + path.string());
    }
    input.seekg(0);
    std::vector<float> values(expected_count);
    input.read(reinterpret_cast<char*>(values.data()),
               static_cast<std::streamsize>(expected_bytes));
    if (!input) {
        throw std::runtime_error(
            "Phase 11 GDN isolation failed to read stage oracle: " + path.string());
    }
    return values;
}

std::uint16_t float_to_bf16_rn(float value) {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    const std::uint32_t exponent = bits & 0x7F800000U;
    const std::uint32_t mantissa = bits & 0x007FFFFFU;
    if (exponent == 0x7F800000U && mantissa != 0U) {
        return static_cast<std::uint16_t>((bits >> 16U) | 0x0040U);
    }
    const std::uint32_t rounded =
        bits + 0x00007FFFU + ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>(rounded >> 16U);
}

GdnConvIsolationMetrics compare_float_stage(
    std::span<const float> candidate, std::span<const float> expected) {
    if (candidate.size() != expected.size() || candidate.empty()) {
        throw std::runtime_error(
            "Phase 11 GDN isolation host comparison shape mismatch");
    }

    long double dot = 0.0L;
    long double candidate_sq = 0.0L;
    long double expected_sq = 0.0L;
    long double error_sq = 0.0L;
    double max_error = 0.0;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const long double a = candidate[i];
        const long double b = expected[i];
        if (!std::isfinite(static_cast<double>(a)) ||
            !std::isfinite(static_cast<double>(b))) {
            throw std::runtime_error(
                "Phase 11 GDN isolation host comparison encountered a nonfinite value");
        }
        const long double d = a - b;
        dot += a * b;
        candidate_sq += a * a;
        expected_sq += b * b;
        error_sq += d * d;
        max_error = std::max(
            max_error, std::abs(static_cast<double>(d)));
    }

    const long double denom = std::sqrt(candidate_sq * expected_sq);
    const double cosine =
        denom > 0.0L
            ? static_cast<double>(dot / denom)
            : (candidate_sq == expected_sq ? 1.0 : 0.0);
    const long double expected_rms =
        std::sqrt(expected_sq / static_cast<long double>(expected.size()));
    const long double error_rms =
        std::sqrt(error_sq / static_cast<long double>(expected.size()));
    return {
        .cosine = cosine,
        .nrmse = static_cast<double>(
            error_rms / std::max(expected_rms, 1.0e-12L)),
        .max_error = max_error,
    };
}

GdnConvIsolationMetrics compare_bf16_stage(
    const ninfer::Tensor& candidate, std::span<const float> expected) {
    if (candidate.dtype != ninfer::DType::BF16 ||
        candidate.numel() != expected.size()) {
        throw std::runtime_error(
            "Phase 11 GDN isolation candidate shape/dtype mismatch");
    }

    std::vector<std::uint16_t> words(expected.size());
    CUDA_CHECK(cudaMemcpy(
        words.data(), candidate.data, words.size() * sizeof(std::uint16_t),
        cudaMemcpyDeviceToHost));

    long double dot = 0.0L;
    long double candidate_sq = 0.0L;
    long double expected_sq = 0.0L;
    long double error_sq = 0.0L;
    double max_error = 0.0;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const long double a = bf16_to_float(words[i]);
        const long double b = expected[i];
        if (!std::isfinite(static_cast<double>(a)) ||
            !std::isfinite(static_cast<double>(b))) {
            throw std::runtime_error(
                "Phase 11 GDN isolation encountered a nonfinite value");
        }
        const long double d = a - b;
        dot += a * b;
        candidate_sq += a * a;
        expected_sq += b * b;
        error_sq += d * d;
        max_error = std::max(
            max_error, std::abs(static_cast<double>(d)));
    }

    const long double denom = std::sqrt(candidate_sq * expected_sq);
    const double cosine =
        denom > 0.0L
            ? static_cast<double>(dot / denom)
            : (candidate_sq == expected_sq ? 1.0 : 0.0);
    const long double expected_rms =
        std::sqrt(expected_sq / static_cast<long double>(expected.size()));
    const long double error_rms =
        std::sqrt(error_sq / static_cast<long double>(expected.size()));
    const double nrmse = static_cast<double>(
        error_rms / std::max(expected_rms, 1.0e-12L));
    return {
        .cosine = cosine,
        .nrmse = nrmse,
        .max_error = max_error,
    };
}

bool gdn_conv_isolation_passes(const GdnConvIsolationMetrics& metrics) {
    return metrics.cosine >= 0.99999 && metrics.nrmse <= 2.0e-3;
}

void print_gdn_conv_isolation_metrics(
    std::string_view stage, const GdnConvIsolationMetrics& metrics) {
    std::cout << std::fixed << std::setprecision(8)
              << "phase11.gdn_conv_isolation.stage=" << stage
              << " cosine=" << metrics.cosine
              << " nrmse=" << metrics.nrmse
              << " max_error=" << metrics.max_error
              << " pass=" << (gdn_conv_isolation_passes(metrics) ? 1 : 0)
              << '\n';
}

void print_gdn_conv_ab_metrics(
    std::string_view variant, std::string_view stage,
    const GdnConvIsolationMetrics& metrics) {
    std::cout << std::fixed << std::setprecision(8)
              << "phase11.gdn_conv_ab.variant=" << variant
              << " stage=" << stage
              << " cosine=" << metrics.cosine
              << " nrmse=" << metrics.nrmse
              << " max_error=" << metrics.max_error
              << " pass=" << (gdn_conv_isolation_passes(metrics) ? 1 : 0)
              << '\n';
}

int run_layer0_gdn_conv_isolation(
    const TextModelView& text, const fs::path& stage_root,
    ninfer::DeviceContext& device) {
    constexpr std::size_t kProjected = 16'384;
    constexpr std::size_t kQuery = 2'048;
    constexpr std::size_t kKey = 2'048;
    constexpr std::size_t kValue = 6'144;
    constexpr std::size_t kConvChannels = 10'240;

    const fs::path position_root = stage_root / "pos0000";
    const std::vector<float> expected_projected =
        load_stage_fp32_vector(position_root / "L00_gdn_projected.bin", kProjected);
    const std::vector<float> expected_query =
        load_stage_fp32_vector(position_root / "L00_gdn_query.bin", kQuery);
    const std::vector<float> expected_key =
        load_stage_fp32_vector(position_root / "L00_gdn_key.bin", kKey);
    const std::vector<float> expected_value =
        load_stage_fp32_vector(position_root / "L00_gdn_value.bin", kValue);

    std::vector<std::uint16_t> projected_bf16(kProjected);
    for (std::size_t i = 0; i < expected_projected.size(); ++i) {
        projected_bf16[i] = float_to_bf16_rn(expected_projected[i]);
    }

    ninfer::WorkspaceArena workspace(
        flash_next_gdn_workspace_capacity_bytes(1, 1));
    FlashNextGdnWorkspace scratch =
        allocate_flash_next_gdn_workspace(workspace, 1);
    ninfer::DeviceBuffer convolution_states(
        kConvChannels * 3ULL * sizeof(std::uint16_t));
    ninfer::DeviceBuffer source_slot(sizeof(std::int32_t));
    ninfer::DeviceBuffer destination_slot(sizeof(std::int32_t));
    const std::int32_t zero = 0;

    CUDA_CHECK(cudaMemcpyAsync(
        scratch.projected.data, projected_bf16.data(),
        projected_bf16.size() * sizeof(std::uint16_t),
        cudaMemcpyHostToDevice, device.stream));
    CUDA_CHECK(cudaMemsetAsync(
        convolution_states.p, 0, convolution_states.bytes, device.stream));
    CUDA_CHECK(cudaMemcpyAsync(
        source_slot.p, &zero, sizeof(zero), cudaMemcpyHostToDevice,
        device.stream));
    CUDA_CHECK(cudaMemcpyAsync(
        destination_slot.p, &zero, sizeof(zero), cudaMemcpyHostToDevice,
        device.stream));

    ninfer::Tensor source(source_slot.p, ninfer::DType::I32, {1});
    ninfer::Tensor destination(destination_slot.p, ninfer::DType::I32, {1});
    ninfer::Tensor states(
        convolution_states.p, ninfer::DType::BF16,
        {static_cast<std::int32_t>(kConvChannels), 3, 1});

    flash_next_gdn_conv_launch(
        scratch, text.gdn[0].convolution, source, destination, states,
        device.stream);
    device.synchronize();

    const GdnConvIsolationMetrics projected_metrics =
        compare_bf16_stage(scratch.projected, expected_projected);
    const GdnConvIsolationMetrics query_metrics =
        compare_bf16_stage(scratch.query, expected_query);
    const GdnConvIsolationMetrics key_metrics =
        compare_bf16_stage(scratch.key, expected_key);
    const GdnConvIsolationMetrics value_metrics =
        compare_bf16_stage(scratch.value, expected_value);

    // Position 0 starts from an all-zero convolution history, so the reference
    // causal convolution reduces to one current-tap multiply followed by SiLU.
    // This makes it possible to separate the oracle's FP32 arithmetic from the
    // two BF16 materialization boundaries without introducing another device
    // implementation.
    std::vector<std::uint16_t> convolution_bf16(kConvChannels * 4);
    CUDA_CHECK(cudaMemcpy(
        convolution_bf16.data(), text.gdn[0].convolution.data,
        convolution_bf16.size() * sizeof(std::uint16_t),
        cudaMemcpyDeviceToHost));

    std::vector<float> host_fp32(kConvChannels);
    std::vector<float> host_bf16_input_fp32_output(kConvChannels);
    std::vector<float> host_bf16_input_fp16_output(kConvChannels);
    std::vector<float> host_bf16_input_bf16_output(kConvChannels);
    for (std::size_t channel = 0; channel < kConvChannels; ++channel) {
        const float weight = bf16_to_float(
            convolution_bf16[3 * kConvChannels + channel]);
        const float fp32_product = expected_projected[channel] * weight;
        host_fp32[channel] =
            fp32_product / (1.0F + std::exp(-fp32_product));

        const float bf16_input = bf16_to_float(projected_bf16[channel]);
        const float bf16_input_product = bf16_input * weight;
        const float fp32_output =
            bf16_input_product / (1.0F + std::exp(-bf16_input_product));
        host_bf16_input_fp32_output[channel] = fp32_output;
        host_bf16_input_fp16_output[channel] =
            __half2float(__float2half_rn(fp32_output));
        host_bf16_input_bf16_output[channel] =
            bf16_to_float(float_to_bf16_rn(fp32_output));
    }

    const auto report_host_variant =
        [&](std::string_view variant, std::span<const float> candidate) {
            print_gdn_conv_ab_metrics(
                variant, "L00_gdn_query",
                compare_float_stage(
                    candidate.subspan(0, kQuery), expected_query));
            print_gdn_conv_ab_metrics(
                variant, "L00_gdn_key",
                compare_float_stage(
                    candidate.subspan(kQuery, kKey), expected_key));
            print_gdn_conv_ab_metrics(
                variant, "L00_gdn_value",
                compare_float_stage(
                    candidate.subspan(kQuery + kKey, kValue), expected_value));
        };

    report_host_variant("fp32_input_fp32_output", host_fp32);
    report_host_variant(
        "bf16_input_fp32_output", host_bf16_input_fp32_output);
    report_host_variant(
        "bf16_input_fp16_output", host_bf16_input_fp16_output);
    report_host_variant(
        "bf16_input_bf16_output", host_bf16_input_bf16_output);

    print_gdn_conv_ab_metrics(
        "gpu_vs_host_bf16", "L00_gdn_query",
        compare_bf16_stage(
            scratch.query,
            std::span<const float>(host_bf16_input_bf16_output)
                .subspan(0, kQuery)));
    print_gdn_conv_ab_metrics(
        "gpu_vs_host_bf16", "L00_gdn_key",
        compare_bf16_stage(
            scratch.key,
            std::span<const float>(host_bf16_input_bf16_output)
                .subspan(kQuery, kKey)));
    print_gdn_conv_ab_metrics(
        "gpu_vs_host_bf16", "L00_gdn_value",
        compare_bf16_stage(
            scratch.value,
            std::span<const float>(host_bf16_input_bf16_output)
                .subspan(kQuery + kKey, kValue)));

    std::cout
        << "phase11.gdn_conv_isolation.position=0"
        << " source=cpu_stage_gdn_projected"
        << " projected_boundary=bf16_rn"
        << " initial_conv_state=zero"
        << " production_kernel=flash_next_gdn_conv_launch\n";
    print_gdn_conv_isolation_metrics(
        "L00_gdn_projected_injected", projected_metrics);
    print_gdn_conv_isolation_metrics("L00_gdn_query", query_metrics);
    print_gdn_conv_isolation_metrics("L00_gdn_key", key_metrics);
    print_gdn_conv_isolation_metrics("L00_gdn_value", value_metrics);

    const bool pass =
        gdn_conv_isolation_passes(query_metrics) &&
        gdn_conv_isolation_passes(key_metrics) &&
        gdn_conv_isolation_passes(value_metrics);
    std::cout << "phase11.gdn_conv_isolation.overall_pass="
              << (pass ? 1 : 0) << '\n';
    return pass ? 0 : 1;
}

} // namespace

int main() {
#if !defined(NINFER_VOLTA_BUILD)
    std::cout << "SKIP: Phase 11 real vertical slice is an SM70 qualification target\n";
    return 77;
#else
    try {
        const char* weights_env = std::getenv("NINFER_WEIGHTS");
        const char* oracle_env =
            std::getenv("NINFER_FLASH_NEXT_ORACLE_MANIFEST");
        if (weights_env == nullptr || weights_env[0] == '\0' ||
            oracle_env == nullptr || oracle_env[0] == '\0') {
            std::cout
                << "SKIP: set NINFER_WEIGHTS and "
                   "NINFER_FLASH_NEXT_ORACLE_MANIFEST for Phase 11\n";
            return 77;
        }

        int device_count = 0;
        const cudaError_t count_error = cudaGetDeviceCount(&device_count);
        if (cuda_unavailable(count_error) || device_count == 0) {
            std::cout << "SKIP: no usable CUDA device for Phase 11\n";
            return 77;
        }
        CUDA_CHECK(count_error);

        const fs::path weights_path(weights_env);
        const fs::path manifest_path(oracle_env);
        const std::uint32_t required_positions = required_oracle_positions();
        std::vector<OracleRecord> records = load_manifest(manifest_path);
        if (records.size() < required_positions) {
            throw std::runtime_error(
                "Phase 11 oracle has fewer positions than required");
        }
        if (required_positions < 4'096U && records.size() > required_positions) {
            records.resize(required_positions);
        }
        std::cout << "phase11.oracle_mode="
                  << (required_positions < 4'096U ? "smoke" : "qualification")
                  << '\n'
                  << "phase11.required_positions=" << required_positions << '\n';

        const std::uint32_t max_context = round_up_128(
            std::max<std::uint64_t>(8'192ULL, records.size() + 1ULL));
        const auto contract =
            make_phase11_vertical_slice_contract(max_context, 128);

        const auto preflight =
            preflight_text_file(weights_path, contract.runtime, 0);
        validate_phase11_preflight(preflight);

        ninfer::DeviceContext device(0);
        Phase11VramObservation vram{};
        vram.before_model_load = phase11_cuda_memory_snapshot();

        auto model = StandaloneLoadedModel::load_from_file(
            weights_path, device, contract.load);
        device.synchronize();
        vram.after_model_load = phase11_cuda_memory_snapshot();

        if (const char* isolation_root =
                std::getenv("NINFER_PHASE11_GDN_CONV_ISOLATION_ROOT");
            isolation_root != nullptr && isolation_root[0] != '\0') {
            return run_layer0_gdn_conv_isolation(
                model.text_view(), fs::path(isolation_root), device);
        }

        FlashNextRuntimeAllocation allocation(preflight.runtime_plan);
        allocation.initialize(device.stream);
        FlashNextTextExecutor executor(
            model.text_view(), model.ple_metadata(), device, allocation);
        executor.set_use_cuda_graph(false);
        device.synchronize();
        vram.after_runtime_allocation = phase11_cuda_memory_snapshot();

        if (executor.use_cuda_graph() ||
            allocation.plan().config.use_cuda_graph) {
            throw std::runtime_error(
                "Phase 11 unexpectedly enabled CUDA Graph");
        }

        const auto vram_result =
            reconcile_phase11_vram(preflight.vram_ledger, vram);
        print_vram(preflight.vram_ledger, vram_result);

        reset_flash_next_host_expert_execution_stats();

        if (const char* prefill_probe =
                std::getenv("NINFER_PHASE11_PREFILL_PROBE_POSITIONS");
            prefill_probe != nullptr && prefill_probe[0] != '\0') {
            char* end = nullptr;
            const unsigned long parsed = std::strtoul(prefill_probe, &end, 10);
            if (end == prefill_probe || *end != '\0' || parsed == 0 ||
                parsed > records.size() || parsed > contract.runtime.prefill_chunk) {
                throw std::invalid_argument(
                    "NINFER_PHASE11_PREFILL_PROBE_POSITIONS must be in "
                    "[1, min(oracle positions, prefill_chunk)]");
            }
            const std::size_t probe_positions = static_cast<std::size_t>(parsed);
            std::vector<std::int32_t> probe_tokens(probe_positions);
            std::vector<std::array<std::int32_t, 3>> probe_mrope(probe_positions);
            for (std::size_t i = 0; i < probe_positions; ++i) {
                probe_tokens[i] = records[i].token_id;
                const auto position = static_cast<std::int32_t>(records[i].position);
                probe_mrope[i] = {position, position, position};
            }

            auto probe_lane = executor.allocate_lane();
            const auto started = std::chrono::steady_clock::now();
            auto round = executor.execute_prefill_chunk(
                probe_lane, probe_tokens, probe_mrope, 0);
            const ninfer::Tensor logits = round.logits();
            if (logits.dtype != ninfer::DType::BF16 || logits.ne[0] <= 0 ||
                static_cast<std::size_t>(logits.ne[0]) !=
                    records[probe_positions - 1].logits_count) {
                throw std::runtime_error(
                    "Phase 11 prefill probe logits do not match oracle shape");
            }

            std::vector<std::uint16_t> candidate_bf16(
                records[probe_positions - 1].logits_count);
            CUDA_CHECK(cudaMemcpyAsync(
                candidate_bf16.data(), logits.data,
                candidate_bf16.size() * sizeof(std::uint16_t),
                cudaMemcpyDeviceToHost, device.stream));
            device.synchronize();

            std::vector<float> candidate(candidate_bf16.size());
            for (std::size_t i = 0; i < candidate.size(); ++i) {
                candidate[i] = bf16_to_float(candidate_bf16[i]);
            }
            const OracleRecord& record = records[probe_positions - 1];
            const std::vector<float> oracle = load_fp32_logits(record);
            const std::int32_t target_token =
                probe_positions < records.size()
                    ? records[probe_positions].token_id
                    : -1;

            Phase11OracleAccumulator probe_accumulator;
            probe_accumulator.observe(
                record.position, target_token, candidate, oracle);
            const Phase11OracleMetrics probe_metrics =
                probe_accumulator.finalize();
            const auto expert_stats =
                flash_next_host_expert_execution_stats();
            const double elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - started).count();

            std::cout << std::fixed << std::setprecision(8)
                      << "phase11.prefill_probe.positions=" << probe_positions
                      << " final_position=" << record.position
                      << " candidate_top1=" << lower_id_argmax(candidate)
                      << " oracle_top1=" << lower_id_argmax(oracle)
                      << " kl=" << probe_metrics.mean_kl
                      << " relative_nll_delta="
                      << probe_metrics.relative_mean_nll_delta
                      << " max_logit_error="
                      << probe_metrics.maximum_logit_error
                      << " elapsed_s=" << elapsed
                      << " expert_pairs=" << expert_stats.expert_pairs
                      << '\n';

            const std::array<LaneCommitDecision, 1> decisions{{
                LaneCommitDecision{.accept = true},
            }};
            round.commit(decisions);
            if (executor.committed_frontier(probe_lane) !=
                static_cast<std::int32_t>(probe_positions)) {
                throw std::runtime_error(
                    "Phase 11 prefill probe frontier did not advance");
            }
            executor.release_lane(probe_lane);
            device.synchronize();
            return 0;
        }

        Phase11OracleAccumulator metrics_accumulator;
        auto lane = executor.allocate_lane();
        std::uint64_t sampled_tokens = 0;
        const auto qualification_started = std::chrono::steady_clock::now();

        const char* stage_root_env =
            std::getenv("NINFER_PHASE11_STAGE_ORACLE_ROOT");
        const bool stage_trace_enabled =
            stage_root_env != nullptr && stage_root_env[0] != '\0';
        const fs::path stage_root =
            stage_trace_enabled ? fs::path(stage_root_env) : fs::path{};
        const char* candidate_root_env =
            std::getenv("NINFER_PHASE11_CANDIDATE_TRACE_ROOT");
        const bool dump_candidate_trace =
            candidate_root_env != nullptr && candidate_root_env[0] != '\0';
        if (dump_candidate_trace && !stage_trace_enabled) {
            throw std::invalid_argument(
                "Phase 11 candidate trace requires a stage oracle root");
        }
        const fs::path candidate_root = dump_candidate_trace
            ? fs::path(candidate_root_env) : fs::path{};
        std::ofstream candidate_index;
        if (dump_candidate_trace) {
            fs::create_directories(candidate_root);
            candidate_index.open(candidate_root / "stages.jsonl", std::ios::trunc);
            if (!candidate_index) {
                throw std::runtime_error("cannot create Phase 11 candidate trace index");
            }
        }
        const char* conv_history_layer_env =
            std::getenv("NINFER_PHASE11_ORACLE_GDN_CONV_HISTORY_LAYER");
        const bool inject_conv_history =
            conv_history_layer_env != nullptr && conv_history_layer_env[0] != '\0';
        if (inject_conv_history &&
            (!stage_trace_enabled || std::string_view(conv_history_layer_env) != "24")) {
            throw std::invalid_argument(
                "Phase 11 convolution-history diagnostic requires a stage oracle and layer 24");
        }
        std::uint32_t stage_trace_position = 6;
        if (const char* position_env =
                std::getenv("NINFER_PHASE11_STAGE_ORACLE_POSITION");
            position_env != nullptr && position_env[0] != '\0') {
            char* end = nullptr;
            const unsigned long parsed = std::strtoul(position_env, &end, 10);
            if (end == position_env || *end != '\0' ||
                parsed > std::numeric_limits<std::uint32_t>::max()) {
                throw std::invalid_argument(
                    "NINFER_PHASE11_STAGE_ORACLE_POSITION is invalid");
            }
            stage_trace_position = static_cast<std::uint32_t>(parsed);
        }
        const bool stage_trace_all_positions =
            stage_trace_enabled &&
            std::getenv("NINFER_PHASE11_STAGE_ORACLE_ALL_POSITIONS") != nullptr;
        const char* oracle_inject_stage_env =
            std::getenv("NINFER_PHASE11_ORACLE_INJECT_STAGE");
        const std::string oracle_inject_stage =
            oracle_inject_stage_env != nullptr ? oracle_inject_stage_env : "";
        const char* second_inject_env =
            std::getenv("NINFER_PHASE11_ORACLE_INJECT_SECOND_STAGE");
        const std::string second_inject_stage =
            second_inject_env != nullptr ? second_inject_env : "";
        const bool oracle_inject_all_positions =
            std::getenv("NINFER_PHASE11_ORACLE_INJECT_ALL_POSITIONS") != nullptr;
        if (!oracle_inject_stage.empty() && !stage_trace_enabled) {
            throw std::invalid_argument(
                "NINFER_PHASE11_ORACLE_INJECT_STAGE requires a stage oracle root");
        }
        if (oracle_inject_all_positions &&
            (!stage_trace_all_positions || oracle_inject_stage.empty())) {
            throw std::invalid_argument(
                "Phase 11 all-position injection requires all-position stage tracing and a stage name");
        }
        if (!second_inject_stage.empty() &&
            (!stage_trace_enabled || second_inject_stage == oracle_inject_stage)) {
            throw std::invalid_argument(
                "Phase 11 second injection requires a distinct stage and stage oracle");
        }
        std::uint32_t current_stage_trace_position = stage_trace_position;
        std::vector<std::string> first_bad_stage_by_position(records.size());
        std::vector<double> first_bad_cosine_by_position(records.size(), 1.0);
        std::vector<double> first_bad_nrmse_by_position(records.size(), 0.0);
        std::vector<double> first_bad_max_error_by_position(records.size(), 0.0);
        std::vector<float> last_router_scores;
        std::string last_router_score_prefix;
        constexpr std::size_t kRouterHidden = 2'560;
        constexpr std::size_t kRouterExperts = 512;
        std::array<std::vector<std::uint16_t>, 48> router_weight_words;

        const auto router_layer_from_stage = [](std::string_view name) {
            if (name.size() < 4 || name[0] != 'L' ||
                name[1] < '0' || name[1] > '9' ||
                name[2] < '0' || name[2] > '9' || name[3] != '_') {
                return -1;
            }
            return (name[1] - '0') * 10 + (name[2] - '0');
        };
        const auto is_pre_router_autopsy_target =
            [](std::uint32_t position, int layer) {
                return (position == 0 && layer >= 7 && layer <= 10) ||
                       (position == 1 && layer >= 12 && layer <= 15);
            };
        const auto round_fp32_to_bf16 = [](float value) {
            std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
            const std::uint32_t exponent = bits & 0x7F80'0000U;
            const std::uint32_t mantissa = bits & 0x007F'FFFFU;
            if (exponent == 0x7F80'0000U) {
                if (mantissa != 0U) {
                    bits |= 0x0040'0000U;
                }
                return static_cast<std::uint16_t>(bits >> 16U);
            }
            const std::uint32_t lsb = (bits >> 16U) & 1U;
            bits += 0x0000'7FFFU + lsb;
            return static_cast<std::uint16_t>(bits >> 16U);
        };
        const auto reduce_v100_router_lanes =
            [](std::array<float, 256> lanes) {
                std::array<float, 8> partial{};
                for (std::size_t warp = 0; warp < 8; ++warp) {
                    auto* values = lanes.data() + warp * 32;
                    for (int offset : {16, 8, 4, 2, 1}) {
                        for (int lane = 0; lane < 32 - offset; ++lane) {
                            values[lane] =
                                values[lane] + values[lane + offset];
                        }
                    }
                    partial[warp] = values[0];
                }
                std::array<float, 32> final{};
                std::copy(partial.begin(), partial.end(), final.begin());
                for (int offset : {16, 8, 4, 2, 1}) {
                    for (int lane = 0; lane < 32 - offset; ++lane) {
                        final[static_cast<std::size_t>(lane)] =
                            final[static_cast<std::size_t>(lane)] +
                            final[static_cast<std::size_t>(lane + offset)];
                    }
                }
                return final[0];
            };
        const auto router_v100_tree_scores =
            [&](std::uint32_t position, int layer, bool bf16_input) {
                if (layer < 0 || layer >= 48) {
                    throw std::runtime_error(
                        "Phase 11 router tree reference has invalid layer");
                }

                char reference_pos_dir[32];
                std::snprintf(
                    reference_pos_dir, sizeof(reference_pos_dir),
                    "pos%04u", position);
                char layer_prefix[8];
                std::snprintf(
                    layer_prefix, sizeof(layer_prefix), "L%02d_", layer);
                const fs::path router_input_path =
                    stage_root / reference_pos_dir /
                    (std::string(layer_prefix) + "mlp_block_input.bin");
                const std::uint64_t router_input_bytes =
                    kRouterHidden * sizeof(float);
                if (!fs::is_regular_file(router_input_path) ||
                    fs::file_size(router_input_path) != router_input_bytes) {
                    throw std::runtime_error(
                        "Phase 11 router tree reference is missing oracle input " +
                        router_input_path.string());
                }

                std::vector<float> oracle_router_input(kRouterHidden);
                {
                    std::ifstream input(router_input_path, std::ios::binary);
                    input.read(
                        reinterpret_cast<char*>(oracle_router_input.data()),
                        static_cast<std::streamsize>(router_input_bytes));
                    if (!input) {
                        throw std::runtime_error(
                            "Phase 11 failed to read oracle router tree input " +
                            router_input_path.string());
                    }
                }
                if (bf16_input) {
                    for (float& value : oracle_router_input) {
                        value = bf16_to_float(round_fp32_to_bf16(value));
                    }
                }

                auto& weight_words =
                    router_weight_words[static_cast<std::size_t>(layer)];
                if (weight_words.empty()) {
                    const auto& router =
                        model.text_view().layers[
                            static_cast<std::size_t>(layer)].moe.router;
                    if (router.qtype != ninfer::QType::BF16_CTRL ||
                        router.n != static_cast<std::int32_t>(kRouterExperts) ||
                        router.k != static_cast<std::int32_t>(kRouterHidden) ||
                        router.qdata == nullptr) {
                        throw std::runtime_error(
                            "Phase 11 router tree reference expected BF16 router weights");
                    }
                    weight_words.resize(kRouterExperts * kRouterHidden);
                    CUDA_CHECK(cudaMemcpy(
                        weight_words.data(), router.qdata,
                        weight_words.size() * sizeof(std::uint16_t),
                        cudaMemcpyDeviceToHost));
                }

                std::array<float, kRouterExperts> scores{};
                for (std::size_t expert = 0;
                     expert < kRouterExperts; ++expert) {
                    std::array<float, 256> lanes{};
                    const std::size_t row = expert * kRouterHidden;
                    for (std::size_t tid = 0; tid < lanes.size(); ++tid) {
                        float acc = 0.0F;
                        for (std::size_t column = tid;
                             column < kRouterHidden;
                             column += lanes.size()) {
                            acc = std::fma(
                                bf16_to_float(weight_words[row + column]),
                                oracle_router_input[column],
                                acc);
                        }
                        lanes[tid] = acc;
                    }
                    scores[expert] = reduce_v100_router_lanes(lanes);
                }
                return scores;
            };

        const auto router_reference_scores =
            [&](std::uint32_t position, int layer) {
                if (layer < 0 || layer >= 48) {
                    throw std::runtime_error(
                        "Phase 11 router reference has invalid layer");
                }

                char reference_pos_dir[32];
                std::snprintf(
                    reference_pos_dir, sizeof(reference_pos_dir),
                    "pos%04u", position);
                char layer_prefix[8];
                std::snprintf(
                    layer_prefix, sizeof(layer_prefix), "L%02d_", layer);
                const fs::path router_input_path =
                    stage_root / reference_pos_dir /
                    (std::string(layer_prefix) + "mlp_block_input.bin");
                const std::uint64_t router_input_bytes =
                    kRouterHidden * sizeof(float);
                if (!fs::is_regular_file(router_input_path) ||
                    fs::file_size(router_input_path) != router_input_bytes) {
                    throw std::runtime_error(
                        "Phase 11 router reference is missing oracle input " +
                        router_input_path.string());
                }

                std::vector<float> oracle_router_input(kRouterHidden);
                {
                    std::ifstream input(router_input_path, std::ios::binary);
                    input.read(
                        reinterpret_cast<char*>(oracle_router_input.data()),
                        static_cast<std::streamsize>(router_input_bytes));
                    if (!input) {
                        throw std::runtime_error(
                            "Phase 11 failed to read oracle router input " +
                            router_input_path.string());
                    }
                }

                auto& weight_words =
                    router_weight_words[static_cast<std::size_t>(layer)];
                if (weight_words.empty()) {
                    const auto& router =
                        model.text_view().layers[
                            static_cast<std::size_t>(layer)].moe.router;
                    if (router.qtype != ninfer::QType::BF16_CTRL ||
                        router.n != static_cast<std::int32_t>(kRouterExperts) ||
                        router.k != static_cast<std::int32_t>(kRouterHidden) ||
                        router.qdata == nullptr) {
                        throw std::runtime_error(
                            "Phase 11 router reference expected BF16 router weights");
                    }
                    weight_words.resize(kRouterExperts * kRouterHidden);
                    CUDA_CHECK(cudaMemcpy(
                        weight_words.data(), router.qdata,
                        weight_words.size() * sizeof(std::uint16_t),
                        cudaMemcpyDeviceToHost));
                }

                std::array<double, kRouterExperts> reference_scores{};
                for (std::size_t expert = 0;
                     expert < kRouterExperts; ++expert) {
                    long double acc = 0.0L;
                    const std::size_t row = expert * kRouterHidden;
                    for (std::size_t k = 0; k < kRouterHidden; ++k) {
                        acc +=
                            static_cast<long double>(
                                bf16_to_float(weight_words[row + k])) *
                            static_cast<long double>(oracle_router_input[k]);
                    }
                    reference_scores[expert] = static_cast<double>(acc);
                }
                return reference_scores;
            };

        FlashNextDecodeStateSink stage_sink;
        stage_sink.on_state = [&](std::string_view name, const ninfer::Tensor& tensor) {
            if (!stage_trace_enabled) {
                return;
            }

            if (name.size() == 21 &&
                name.substr(4) == "moe_router_scores" &&
                tensor.dtype == ninfer::DType::FP32 &&
                tensor.numel() >= 512) {
                last_router_scores.resize(tensor.numel());
                CUDA_CHECK(cudaMemcpy(
                    last_router_scores.data(), tensor.data,
                    last_router_scores.size() * sizeof(float),
                    cudaMemcpyDeviceToHost));
                last_router_score_prefix = std::string(name.substr(0, 4));

                const int layer = router_layer_from_stage(name);
                if (is_pre_router_autopsy_target(
                        current_stage_trace_position, layer)) {
                    const auto reference_scores =
                        router_reference_scores(
                            current_stage_trace_position, layer);
                    long double dot = 0.0L;
                    long double candidate_sq = 0.0L;
                    long double reference_sq = 0.0L;
                    long double error_sq = 0.0L;
                    double max_error = 0.0;
                    for (std::size_t expert = 0;
                         expert < kRouterExperts; ++expert) {
                        const long double candidate =
                            last_router_scores[expert];
                        const long double reference =
                            reference_scores[expert];
                        const long double error = candidate - reference;
                        dot += candidate * reference;
                        candidate_sq += candidate * candidate;
                        reference_sq += reference * reference;
                        error_sq += error * error;
                        max_error = std::max(
                            max_error,
                            std::abs(static_cast<double>(error)));
                    }
                    const long double denom =
                        std::sqrt(candidate_sq * reference_sq);
                    const double cosine =
                        denom > 0.0L
                            ? static_cast<double>(dot / denom)
                            : (candidate_sq == reference_sq ? 1.0 : 0.0);
                    const long double reference_rms =
                        std::sqrt(
                            reference_sq /
                            static_cast<long double>(kRouterExperts));
                    const long double error_rms =
                        std::sqrt(
                            error_sq /
                            static_cast<long double>(kRouterExperts));
                    const double nrmse = static_cast<double>(
                        error_rms /
                        std::max(reference_rms, 1.0e-12L));
                    const int branch_layer =
                        current_stage_trace_position == 0 ? 10 : 15;
                    std::cout << std::fixed << std::setprecision(8)
                              << "phase11.pre_router_autopsy.position="
                              << current_stage_trace_position
                              << " layer=" << layer
                              << " checkpoint=router_scores"
                              << " branch_state="
                              << (layer == branch_layer ? "decision" : "pre")
                              << " cosine=" << cosine
                              << " nrmse=" << nrmse
                              << " rms_error="
                              << static_cast<double>(error_rms)
                              << " max_error=" << max_error
                              << " pass="
                              << ((cosine >= 0.99999 && nrmse <= 2.0e-3)
                                      ? 1 : 0)
                              << '\n' << std::flush;
                }
            }

            const auto is_router_probe = [&](std::string_view suffix) {
                return
                    (current_stage_trace_position == 2 &&
                     (name == std::string("L20_") + std::string(suffix) ||
                      name == std::string("L38_") + std::string(suffix))) ||
                    (current_stage_trace_position == 4 &&
                     name == std::string("L41_") + std::string(suffix)) ||
                    (current_stage_trace_position == 6 &&
                     name == std::string("L32_") + std::string(suffix));
            };

            if (is_router_probe("moe_router_scores") &&
                tensor.dtype == ninfer::DType::FP32 &&
                tensor.numel() >= 513) {
                std::vector<float> scores(tensor.numel());
                CUDA_CHECK(cudaMemcpy(
                    scores.data(), tensor.data,
                    scores.size() * sizeof(float),
                    cudaMemcpyDeviceToHost));
                std::vector<std::pair<float, std::int32_t>> ranked;
                ranked.reserve(512);
                for (std::int32_t expert = 0; expert < 512; ++expert) {
                    ranked.emplace_back(scores[expert], expert);
                }
                std::stable_sort(
                    ranked.begin(), ranked.end(),
                    [](const auto& lhs, const auto& rhs) {
                        return lhs.first > rhs.first ||
                               (lhs.first == rhs.first && lhs.second < rhs.second);
                    });
                std::cout << std::fixed << std::setprecision(8)
                          << "phase11.router_probe.position="
                          << current_stage_trace_position
                          << " stage=" << name
                          << " top10_margin="
                          << (ranked[9].first - ranked[10].first)
                          << " top10_ids=";
                for (std::size_t i = 0; i < 10; ++i) {
                    if (i != 0) { std::cout << ','; }
                    std::cout << ranked[i].second;
                }
                std::cout << " top12_scores=";
                for (std::size_t i = 0; i < 12; ++i) {
                    if (i != 0) { std::cout << ','; }
                    std::cout << ranked[i].first;
                }
                std::cout << '\n' << std::flush;
            }

            if (is_router_probe("moe_router_ids") &&
                tensor.dtype == ninfer::DType::I32) {
                std::vector<std::int32_t> ids(tensor.numel());
                CUDA_CHECK(cudaMemcpy(
                    ids.data(), tensor.data,
                    ids.size() * sizeof(std::int32_t),
                    cudaMemcpyDeviceToHost));
                std::cout << "phase11.router_probe.position="
                          << current_stage_trace_position
                          << " stage=" << name << " ids=";
                for (std::size_t i = 0; i < ids.size(); ++i) {
                    if (i != 0) { std::cout << ','; }
                    std::cout << ids[i];
                }
                std::cout << '\n' << std::flush;
            }

            if (is_router_probe("moe_router_alpha") &&
                tensor.dtype == ninfer::DType::FP32) {
                std::vector<float> alpha(tensor.numel());
                CUDA_CHECK(cudaMemcpy(
                    alpha.data(), tensor.data,
                    alpha.size() * sizeof(float),
                    cudaMemcpyDeviceToHost));
                std::cout << std::fixed << std::setprecision(8)
                          << "phase11.router_probe.position="
                          << current_stage_trace_position
                          << " stage=" << name << " alpha=";
                for (std::size_t i = 0; i < alpha.size(); ++i) {
                    if (i != 0) { std::cout << ','; }
                    std::cout << alpha[i];
                }
                std::cout << '\n' << std::flush;
            }

            char pos_dir[32];
            std::snprintf(
                pos_dir, sizeof(pos_dir), "pos%04u", current_stage_trace_position);
            std::string oracle_stage_name(name);
            if (name.size() > 4 &&
                name.substr(4) == "mlp_block_input_fp32") {
                oracle_stage_name =
                    std::string(name.substr(0, 4)) + "mlp_block_input";
            }
            const fs::path expected_path =
                stage_root / pos_dir / (oracle_stage_name + ".bin");
            if (!fs::is_regular_file(expected_path)) {
                return;
            }

            const std::size_t count = tensor.numel();
            const bool integer_tensor = tensor.dtype == ninfer::DType::I32;
            // Stage-oracle payloads are normalized to FP32 even when the live
            // tensor is integer-valued (for example MoE router IDs).
            const std::uint64_t expected_bytes =
                static_cast<std::uint64_t>(count) * sizeof(float);
            if (fs::file_size(expected_path) != expected_bytes) {
                throw std::runtime_error(
                    "Phase 11 stage oracle shape mismatch for " +
                    std::string(name));
            }

            std::vector<float> expected(count);
            {
                std::ifstream input(
                    expected_path, std::ios::binary);
                input.read(
                    reinterpret_cast<char*>(expected.data()),
                    static_cast<std::streamsize>(expected_bytes));
                if (!input) {
                    throw std::runtime_error(
                        "Phase 11 failed to read stage oracle " +
                        expected_path.string());
                }
            }

            std::vector<float> candidate(count);
            std::vector<std::int32_t> expected_i32;
            std::vector<std::int32_t> candidate_i32;
            if (integer_tensor) {
                expected_i32.resize(count);
                candidate_i32.resize(count);
                CUDA_CHECK(cudaMemcpy(
                    candidate_i32.data(), tensor.data,
                    count * sizeof(std::int32_t),
                    cudaMemcpyDeviceToHost));
                for (std::size_t i = 0; i < count; ++i) {
                    expected_i32[i] =
                        static_cast<std::int32_t>(expected[i]);
                    candidate[i] =
                        static_cast<float>(candidate_i32[i]);
                }
            } else if (tensor.dtype == ninfer::DType::BF16) {
                std::vector<std::uint16_t> words(count);
                CUDA_CHECK(cudaMemcpy(
                    words.data(), tensor.data,
                    count * sizeof(std::uint16_t),
                    cudaMemcpyDeviceToHost));
                for (std::size_t i = 0; i < count; ++i) {
                    candidate[i] = bf16_to_float(words[i]);
                }
            } else if (tensor.dtype == ninfer::DType::FP32) {
                CUDA_CHECK(cudaMemcpy(
                    candidate.data(), tensor.data,
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost));
            } else {
                return;
            }

            // The CPU precision experiment compares these same stage values
            // against the FP32 oracle and the live V100.  Dump before any
            // diagnostic oracle injection modifies the device tensor.
            const std::string_view suffix = name.size() > 4 ? name.substr(4) : name;
            const bool selected_stage =
                name == "embedding" || name == "ple_injection" ||
                name == "final_hidden" || name == "logits" ||
                suffix == "attn_block_input" ||
                suffix == "mlp_block_input" ||
                suffix == "hyper_after_mlp" ||
                suffix == "moe_router_ids";
            if (dump_candidate_trace && selected_stage) {
                const fs::path folder = candidate_root / pos_dir;
                fs::create_directories(folder);
                const fs::path path = folder / (std::string(name) + ".bin");
                std::ofstream output_file(path, std::ios::binary | std::ios::trunc);
                output_file.write(
                    reinterpret_cast<const char*>(candidate.data()),
                    static_cast<std::streamsize>(count * sizeof(float)));
                if (!output_file) {
                    throw std::runtime_error("failed to write Phase 11 candidate stage");
                }
                candidate_index << json{{"position", current_stage_trace_position},
                                        {"name", std::string(name)},
                                        {"dtype", "FP32"},
                                        {"count", count},
                                        {"file", (fs::path(pos_dir) /
                                                  (std::string(name) + ".bin")).string()}}
                                << '\n' << std::flush;
                // The frozen oracle has router IDs but no full score vector.
                // Scores were captured before top-k and before any diagnostic
                // injection. Export them alongside the ID stage so the CPU
                // storage profile can compare near-tie decisions directly.
                if (suffix == "moe_router_ids" &&
                    last_router_score_prefix == name.substr(0, 4) &&
                    last_router_scores.size() >= kRouterExperts) {
                    const std::string score_name =
                        std::string(name.substr(0, 4)) + "moe_router_scores";
                    const fs::path score_path = folder / (score_name + ".bin");
                    std::ofstream scores_file(
                        score_path, std::ios::binary | std::ios::trunc);
                    scores_file.write(
                        reinterpret_cast<const char*>(last_router_scores.data()),
                        static_cast<std::streamsize>(kRouterExperts * sizeof(float)));
                    if (!scores_file) {
                        throw std::runtime_error("failed to write Phase 11 router scores");
                    }
                    candidate_index << json{{"position", current_stage_trace_position},
                                            {"name", score_name},
                                            {"dtype", "FP32"},
                                            {"count", kRouterExperts},
                                            {"file", (fs::path(pos_dir) /
                                                      (score_name + ".bin")).string()}}
                                    << '\n' << std::flush;
                }
            }

            long double dot = 0.0L;
            long double candidate_sq = 0.0L;
            long double expected_sq = 0.0L;
            long double error_sq = 0.0L;
            long double floor_error_sq = 0.0L;
            long double residual_error_sq = 0.0L;
            double max_error = 0.0;
            const bool bf16_stage = tensor.dtype == ninfer::DType::BF16;
            for (std::size_t i = 0; i < count; ++i) {
                const long double a = candidate[i];
                const long double b = expected[i];
                const long double d = a - b;
                dot += a * b;
                candidate_sq += a * a;
                expected_sq += b * b;
                error_sq += d * d;
                max_error =
                    std::max(max_error, std::abs(
                        static_cast<double>(d)));
                if (bf16_stage) {
                    const float rounded =
                        bf16_to_float(round_fp32_to_bf16(expected[i]));
                    const long double floor_error =
                        static_cast<long double>(rounded) - b;
                    const long double residual_error =
                        a - static_cast<long double>(rounded);
                    floor_error_sq += floor_error * floor_error;
                    residual_error_sq += residual_error * residual_error;
                }
            }
            const long double denom =
                std::sqrt(candidate_sq * expected_sq);
            const double cosine =
                denom > 0.0L
                    ? static_cast<double>(dot / denom)
                    : (candidate_sq == expected_sq ? 1.0 : 0.0);
            const long double expected_rms =
                std::sqrt(expected_sq /
                          static_cast<long double>(count));
            const long double error_rms =
                std::sqrt(error_sq /
                          static_cast<long double>(count));
            const double nrmse =
                static_cast<double>(
                    error_rms /
                    std::max(expected_rms, 1.0e-12L));
            const bool pass =
                integer_tensor ? max_error == 0.0
                               : (cosine >= 0.99999 && nrmse <= 2.0e-3);

            std::cout << std::fixed << std::setprecision(8)
                      << "phase11.stage_trace.position="
                      << current_stage_trace_position
                      << " stage=" << name
                      << " cosine=" << cosine
                      << " nrmse=" << nrmse
                      << " max_error=" << max_error
                      << " pass=" << (pass ? 1 : 0)
                      << '\n' << std::flush;

            // Error decomposition: observed NRMSE against the oracle, the
            // rounding floor (oracle re-rounded to this stage's storage
            // dtype), the residual beyond the floor, and their ratio. FP32
            // and integer stages have a zero floor: any nonzero NRMSE is an
            // implementation deviation, not storage rounding.
            const char* dtype_name =
                integer_tensor ? "i32"
                : bf16_stage ? "bf16" : "fp32";
            const double floor_nrmse =
                bf16_stage
                    ? static_cast<double>(
                          std::sqrt(floor_error_sq /
                                    static_cast<long double>(count)) /
                          std::max(expected_rms, 1.0e-12L))
                    : 0.0;
            const double residual_nrmse =
                bf16_stage
                    ? static_cast<double>(
                          std::sqrt(residual_error_sq /
                                    static_cast<long double>(count)) /
                          std::max(expected_rms, 1.0e-12L))
                    : nrmse;
            const double floor_ratio =
                floor_nrmse > 0.0
                    ? nrmse / floor_nrmse
                    : (nrmse > 0.0
                           ? std::numeric_limits<double>::infinity()
                           : -1.0);
            std::cout << std::fixed << std::setprecision(8)
                      << "phase11.error_decomposition.position="
                      << current_stage_trace_position
                      << " stage=" << name
                      << " dtype=" << dtype_name
                      << " nrmse=" << nrmse
                      << " floor_nrmse=" << floor_nrmse
                      << " residual_nrmse=" << residual_nrmse
                      << " floor_ratio=" << floor_ratio
                      << '\n' << std::flush;

            const int autopsy_layer = router_layer_from_stage(name);
            const std::string_view autopsy_checkpoint =
                autopsy_layer >= 0 ? name.substr(4) : std::string_view{};
            if (is_pre_router_autopsy_target(
                    current_stage_trace_position, autopsy_layer) &&
                (autopsy_checkpoint == "hyper_after_attn" ||
                 autopsy_checkpoint == "mlp_block_input_fp32" ||
                 autopsy_checkpoint == "mlp_block_input")) {
                std::cout << std::fixed << std::setprecision(8)
                          << "phase11.pre_router_autopsy.position="
                          << current_stage_trace_position
                          << " layer=" << autopsy_layer
                          << " checkpoint=" << autopsy_checkpoint
                          << " branch_state=pre"
                          << " cosine=" << cosine
                          << " nrmse=" << nrmse
                          << " rms_error="
                          << static_cast<double>(error_rms)
                          << " max_error=" << max_error
                          << " pass=" << (pass ? 1 : 0)
                          << '\n' << std::flush;
            }

            const bool inject_first =
                !oracle_inject_stage.empty() &&
                (oracle_inject_all_positions ||
                 current_stage_trace_position == stage_trace_position) &&
                name == oracle_inject_stage;
            const bool inject_second =
                !second_inject_stage.empty() &&
                current_stage_trace_position == stage_trace_position &&
                name == second_inject_stage;
            if (inject_first || inject_second) {
                if (integer_tensor) {
                    throw std::invalid_argument(
                        "Phase 11 oracle injection does not support integer stages");
                }
                if (tensor.dtype == ninfer::DType::BF16) {
                    std::vector<std::uint16_t> words(count);
                    for (std::size_t i = 0; i < count; ++i) {
                        words[i] = round_fp32_to_bf16(expected[i]);
                    }
                    CUDA_CHECK(cudaMemcpy(
                        tensor.data, words.data(),
                        count * sizeof(std::uint16_t),
                        cudaMemcpyHostToDevice));
                    std::vector<std::uint16_t> verified(count);
                    CUDA_CHECK(cudaMemcpy(verified.data(), tensor.data,
                                          count * sizeof(std::uint16_t),
                                          cudaMemcpyDeviceToHost));
                    if (verified != words) {
                        throw std::runtime_error("Phase 11 BF16 stage injection readback mismatch");
                    }
                } else if (tensor.dtype == ninfer::DType::FP32) {
                    CUDA_CHECK(cudaMemcpy(
                        tensor.data, expected.data(),
                        count * sizeof(float),
                        cudaMemcpyHostToDevice));
                    std::vector<float> verified(count);
                    CUDA_CHECK(cudaMemcpy(verified.data(), tensor.data,
                                          count * sizeof(float),
                                          cudaMemcpyDeviceToHost));
                    if (!std::equal(verified.begin(), verified.end(), expected.begin(),
                                    [](float a, float b) {
                                        return std::bit_cast<std::uint32_t>(a) ==
                                               std::bit_cast<std::uint32_t>(b);
                                    })) {
                        throw std::runtime_error("Phase 11 FP32 stage injection readback mismatch");
                    }
                } else {
                    throw std::invalid_argument(
                        "Phase 11 oracle injection supports only BF16/FP32 stages");
                }
                std::cout << "phase11.oracle_injection.position="
                          << current_stage_trace_position
                          << " stage=" << name
                          << " count=" << count << '\n'
                          << std::flush;
            }

            if (integer_tensor && !pass) {
                auto expected_ids_sorted = expected_i32;
                auto candidate_ids_sorted = candidate_i32;
                std::sort(expected_ids_sorted.begin(), expected_ids_sorted.end());
                std::sort(candidate_ids_sorted.begin(), candidate_ids_sorted.end());
                const bool same_expert_set =
                    expected_ids_sorted == candidate_ids_sorted;
                std::cout << "phase11.router_ids_mismatch.position="
                          << current_stage_trace_position
                          << " stage=" << name
                          << " same_expert_set=" << (same_expert_set ? 1 : 0)
                          << " expected=";
                for (std::size_t i = 0; i < expected_i32.size(); ++i) {
                    if (i != 0) { std::cout << ','; }
                    std::cout << expected_i32[i];
                }
                std::cout << " candidate=";
                for (std::size_t i = 0; i < candidate_i32.size(); ++i) {
                    if (i != 0) { std::cout << ','; }
                    std::cout << candidate_i32[i];
                }
                std::cout << '\n' << std::flush;

                if (name.size() == 18 &&
                    name.substr(4) == "moe_router_ids" &&
                    last_router_score_prefix == name.substr(0, 4) &&
                    last_router_scores.size() >= 512) {
                    const int layer = router_layer_from_stage(name);
                    if (layer < 0 || layer >= 48) {
                        throw std::runtime_error(
                            "Phase 11 router margin diagnostic has invalid layer name");
                    }
                    const auto reference_scores =
                        router_reference_scores(
                            current_stage_trace_position, layer);

                    std::array<std::int32_t, kRouterExperts> reference_rank{};
                    std::iota(reference_rank.begin(), reference_rank.end(), 0);
                    std::stable_sort(
                        reference_rank.begin(), reference_rank.end(),
                        [&](std::int32_t lhs, std::int32_t rhs) {
                            const double a =
                                reference_scores[static_cast<std::size_t>(lhs)];
                            const double b =
                                reference_scores[static_cast<std::size_t>(rhs)];
                            return a > b || (a == b && lhs < rhs);
                        });

                    std::array<bool, kRouterExperts> oracle_set{};
                    for (const std::int32_t id : expected_i32) {
                        if (id >= 0 &&
                            id < static_cast<std::int32_t>(kRouterExperts)) {
                            oracle_set[static_cast<std::size_t>(id)] = true;
                        }
                    }
                    bool reference_set_matches_oracle = true;
                    for (std::size_t i = 0; i < 10; ++i) {
                        if (!oracle_set[
                                static_cast<std::size_t>(reference_rank[i])]) {
                            reference_set_matches_oracle = false;
                            break;
                        }
                    }

                    const double cutoff_margin =
                        reference_scores[
                            static_cast<std::size_t>(reference_rank[9])] -
                        reference_scores[
                            static_cast<std::size_t>(reference_rank[10])];

                    double min_oracle_top10_gap =
                        std::numeric_limits<double>::infinity();
                    for (std::size_t i = 1;
                         i < std::min<std::size_t>(10, expected_i32.size());
                         ++i) {
                        const auto lhs =
                            static_cast<std::size_t>(expected_i32[i - 1]);
                        const auto rhs =
                            static_cast<std::size_t>(expected_i32[i]);
                        min_oracle_top10_gap = std::min(
                            min_oracle_top10_gap,
                            std::abs(reference_scores[lhs] -
                                     reference_scores[rhs]));
                    }

                    long double score_error_sq = 0.0L;
                    double score_max_error = 0.0;
                    for (std::size_t expert = 0;
                         expert < kRouterExperts; ++expert) {
                        const double error =
                            static_cast<double>(last_router_scores[expert]) -
                            reference_scores[expert];
                        score_error_sq +=
                            static_cast<long double>(error) * error;
                        score_max_error =
                            std::max(score_max_error, std::abs(error));
                    }
                    const double score_rms_error = std::sqrt(
                        static_cast<double>(
                            score_error_sq /
                            static_cast<long double>(kRouterExperts)));
                    const double cutoff_over_max_error =
                        score_max_error > 0.0
                            ? cutoff_margin / score_max_error
                            : std::numeric_limits<double>::infinity();
                    const double cutoff_over_rms_error =
                        score_rms_error > 0.0
                            ? cutoff_margin / score_rms_error
                            : std::numeric_limits<double>::infinity();
                    const double top10_gap_over_max_error =
                        score_max_error > 0.0
                            ? min_oracle_top10_gap / score_max_error
                            : std::numeric_limits<double>::infinity();

                    const auto fp32_tree_scores =
                        router_v100_tree_scores(
                            current_stage_trace_position, layer, false);
                    const auto bf16_tree_scores =
                        router_v100_tree_scores(
                            current_stage_trace_position, layer, true);
                    double association_max_error = 0.0;
                    long double association_error_sq = 0.0L;
                    double materialization_max_error = 0.0;
                    long double materialization_error_sq = 0.0L;
                    std::array<std::int32_t, kRouterExperts> fp32_tree_rank{};
                    std::array<std::int32_t, kRouterExperts> bf16_tree_rank{};
                    std::iota(
                        fp32_tree_rank.begin(), fp32_tree_rank.end(), 0);
                    std::iota(
                        bf16_tree_rank.begin(), bf16_tree_rank.end(), 0);
                    for (std::size_t expert = 0;
                         expert < kRouterExperts; ++expert) {
                        const double association_error =
                            static_cast<double>(fp32_tree_scores[expert]) -
                            reference_scores[expert];
                        association_max_error = std::max(
                            association_max_error,
                            std::abs(association_error));
                        association_error_sq +=
                            static_cast<long double>(association_error) *
                            association_error;
                        const double materialization_error =
                            static_cast<double>(bf16_tree_scores[expert]) -
                            static_cast<double>(fp32_tree_scores[expert]);
                        materialization_max_error = std::max(
                            materialization_max_error,
                            std::abs(materialization_error));
                        materialization_error_sq +=
                            static_cast<long double>(materialization_error) *
                            materialization_error;
                    }
                    const auto tree_better =
                        [](const auto& scores,
                           std::int32_t lhs, std::int32_t rhs) {
                            const float a =
                                scores[static_cast<std::size_t>(lhs)];
                            const float b =
                                scores[static_cast<std::size_t>(rhs)];
                            return a > b || (a == b && lhs < rhs);
                        };
                    std::stable_sort(
                        fp32_tree_rank.begin(), fp32_tree_rank.end(),
                        [&](std::int32_t lhs, std::int32_t rhs) {
                            return tree_better(
                                fp32_tree_scores, lhs, rhs);
                        });
                    std::stable_sort(
                        bf16_tree_rank.begin(), bf16_tree_rank.end(),
                        [&](std::int32_t lhs, std::int32_t rhs) {
                            return tree_better(
                                bf16_tree_scores, lhs, rhs);
                        });
                    const auto top10_matches_oracle =
                        [&](const auto& rank) {
                            for (std::size_t i = 0; i < 10; ++i) {
                                if (!oracle_set[
                                        static_cast<std::size_t>(rank[i])]) {
                                    return false;
                                }
                            }
                            return true;
                        };
                    const bool fp32_tree_set_matches_oracle =
                        top10_matches_oracle(fp32_tree_rank);
                    const bool bf16_tree_set_matches_oracle =
                        top10_matches_oracle(bf16_tree_rank);
                    const double association_rms_error = std::sqrt(
                        static_cast<double>(
                            association_error_sq /
                            static_cast<long double>(kRouterExperts)));
                    const double materialization_rms_error = std::sqrt(
                        static_cast<double>(
                            materialization_error_sq /
                            static_cast<long double>(kRouterExperts)));
                    const double cutoff_over_association_max =
                        association_max_error > 0.0
                            ? cutoff_margin / association_max_error
                            : std::numeric_limits<double>::infinity();
                    const double cutoff_over_materialization_max =
                        materialization_max_error > 0.0
                            ? cutoff_margin / materialization_max_error
                            : std::numeric_limits<double>::infinity();
                    const double top10_gap_over_association_max =
                        association_max_error > 0.0
                            ? min_oracle_top10_gap / association_max_error
                            : std::numeric_limits<double>::infinity();
                    const double top10_gap_over_materialization_max =
                        materialization_max_error > 0.0
                            ? min_oracle_top10_gap / materialization_max_error
                            : std::numeric_limits<double>::infinity();

                    std::cout << std::fixed << std::setprecision(8)
                              << "phase11.router_margin.position="
                              << current_stage_trace_position
                              << " stage=" << name
                              << " same_expert_set="
                              << (same_expert_set ? 1 : 0)
                              << " reference_set_matches_oracle="
                              << (reference_set_matches_oracle ? 1 : 0)
                              << " cutoff_margin=" << cutoff_margin
                              << " min_oracle_top10_gap="
                              << min_oracle_top10_gap
                              << " score_max_error=" << score_max_error
                              << " score_rms_error=" << score_rms_error
                              << " cutoff_over_max_error="
                              << cutoff_over_max_error
                              << " cutoff_over_rms_error="
                              << cutoff_over_rms_error
                              << " top10_gap_over_max_error="
                              << top10_gap_over_max_error
                              << " fp32_tree_set_matches_oracle="
                              << (fp32_tree_set_matches_oracle ? 1 : 0)
                              << " bf16_tree_set_matches_oracle="
                              << (bf16_tree_set_matches_oracle ? 1 : 0)
                              << " association_max_error="
                              << association_max_error
                              << " association_rms_error="
                              << association_rms_error
                              << " materialization_max_error="
                              << materialization_max_error
                              << " materialization_rms_error="
                              << materialization_rms_error
                              << " cutoff_over_association_max="
                              << cutoff_over_association_max
                              << " cutoff_over_materialization_max="
                              << cutoff_over_materialization_max
                              << " top10_gap_over_association_max="
                              << top10_gap_over_association_max
                              << " top10_gap_over_materialization_max="
                              << top10_gap_over_materialization_max
                              << '\n' << std::flush;
                }
            }

            if (!pass &&
                current_stage_trace_position < first_bad_stage_by_position.size() &&
                first_bad_stage_by_position[current_stage_trace_position].empty()) {
                first_bad_stage_by_position[current_stage_trace_position] =
                    std::string(name);
                first_bad_cosine_by_position[current_stage_trace_position] = cosine;
                first_bad_nrmse_by_position[current_stage_trace_position] = nrmse;
                first_bad_max_error_by_position[current_stage_trace_position] = max_error;
            }
        };

        for (std::size_t index = 0; index < records.size(); ++index) {
            const OracleRecord& record = records[index];
            if (inject_conv_history && record.position == 13) {
                constexpr std::size_t kChannels = 10'240;
                constexpr std::size_t kHistory = 3;
                if (executor.committed_frontier(lane) != 13 || is_qsa_layer(24)) {
                    throw std::runtime_error("Phase 11 layer-24 history requires committed position 12");
                }
                std::vector<std::uint16_t> oracle_history(kChannels * kHistory);
                for (std::size_t history = 0; history < kHistory; ++history) {
                    const std::uint32_t prior_position =
                        13U - static_cast<std::uint32_t>(kHistory) +
                        static_cast<std::uint32_t>(history);
                    char directory[32];
                    std::snprintf(directory, sizeof(directory), "pos%04u", prior_position);
                    const fs::path projected_path =
                        stage_root / directory / "L24_gdn_projected.bin";
                    constexpr std::size_t kProjected = 16'384;
                    if (!fs::is_regular_file(projected_path) ||
                        fs::file_size(projected_path) != kProjected * sizeof(float)) {
                        throw std::runtime_error(
                            "Phase 11 missing L24 projected history: " +
                            projected_path.string());
                    }
                    std::ifstream projected_file(projected_path, std::ios::binary);
                    std::vector<float> projected(kProjected);
                    projected_file.read(reinterpret_cast<char*>(projected.data()),
                                        static_cast<std::streamsize>(kProjected * sizeof(float)));
                    if (!projected_file) {
                        throw std::runtime_error("Phase 11 failed reading L24 projected history");
                    }
                    for (std::size_t channel = 0; channel < kChannels; ++channel) {
                        oracle_history[history * kChannels + channel] =
                            round_fp32_to_bf16(projected[channel]);
                    }
                }
                const auto active_slot = allocation.current_source_slot(0);
                if (active_slot < 0) {
                    throw std::runtime_error("Phase 11 has no active state slot");
                }
                auto& conv = allocation.state_view().gdn_convolution_states[gdn_ordinal(24)];
                auto* slot_ptr = static_cast<std::byte*>(conv.data) +
                    static_cast<std::size_t>(active_slot) * oracle_history.size() *
                        sizeof(std::uint16_t);
                std::vector<std::uint16_t> original(oracle_history.size());
                CUDA_CHECK(cudaMemcpyAsync(original.data(), slot_ptr,
                                           original.size() * sizeof(std::uint16_t),
                                           cudaMemcpyDeviceToHost, device.stream));
                device.synchronize();
                std::size_t changed = 0;
                for (std::size_t i = 0; i < original.size(); ++i) {
                    changed += original[i] != oracle_history[i];
                }
                CUDA_CHECK(cudaMemcpyAsync(slot_ptr, oracle_history.data(),
                                           oracle_history.size() * sizeof(std::uint16_t),
                                           cudaMemcpyHostToDevice, device.stream));
                std::vector<std::uint16_t> verified(oracle_history.size());
                CUDA_CHECK(cudaMemcpyAsync(verified.data(), slot_ptr,
                                           verified.size() * sizeof(std::uint16_t),
                                           cudaMemcpyDeviceToHost, device.stream));
                device.synchronize();
                if (verified != oracle_history) {
                    throw std::runtime_error("Phase 11 convolution-history injection did not persist");
                }
                std::cout << "phase11.conv_history_injection.position=13 layer=24"
                          << " source_slot=" << active_slot
                          << " changed_bf16_elements=" << changed
                          << " total_elements=" << oracle_history.size() << '\n'
                          << std::flush;
            }
            LaneStepRequest request{
                .handle = lane,
                .token_id = record.token_id,
                .token_index = static_cast<std::int32_t>(record.position),
                .mrope_positions = {
                    static_cast<std::int32_t>(record.position),
                    static_cast<std::int32_t>(record.position),
                    static_cast<std::int32_t>(record.position),
                },
                .sampling = {},
                .custom_embedding = nullptr,
            };

            current_stage_trace_position = record.position;
            const FlashNextDecodeStateSink* round_sink =
                stage_trace_enabled &&
                        (stage_trace_all_positions ||
                         record.position == stage_trace_position)
                    ? &stage_sink
                    : nullptr;
            auto round = executor.execute_round(
                std::span<const LaneStepRequest>(&request, 1),
                round_sink);
            const ninfer::Tensor logits = round.logits();
            if (logits.dtype != ninfer::DType::BF16 ||
                logits.ne[1] != 1 || logits.ne[2] != 1 ||
                logits.ne[3] != 1 ||
                logits.ne[0] <= 0 ||
                static_cast<std::size_t>(logits.ne[0]) !=
                    record.logits_count) {
                throw std::runtime_error(
                    "Phase 11 candidate logits do not match oracle shape");
            }

            std::vector<std::uint16_t> candidate_bf16(record.logits_count);
            CUDA_CHECK(cudaMemcpyAsync(
                candidate_bf16.data(), logits.data,
                candidate_bf16.size() * sizeof(std::uint16_t),
                cudaMemcpyDeviceToHost, device.stream));
            device.synchronize();

            std::vector<float> candidate(candidate_bf16.size());
            for (std::size_t i = 0; i < candidate.size(); ++i) {
                candidate[i] = bf16_to_float(candidate_bf16[i]);
            }
            const std::vector<float> oracle = load_fp32_logits(record);
            const std::int32_t target_token =
                index + 1 < records.size()
                    ? records[index + 1].token_id
                    : -1;
            metrics_accumulator.observe(
                record.position, target_token, candidate, oracle);

            if (const char* trace = std::getenv("NINFER_PHASE11_TRACE_POSITIONS");
                trace != nullptr && trace[0] != '\0') {
                Phase11OracleAccumulator position_accumulator;
                position_accumulator.observe(
                    record.position, target_token, candidate, oracle);
                const Phase11OracleMetrics position_metrics =
                    position_accumulator.finalize();
                std::cout << std::fixed << std::setprecision(8)
                          << "phase11.trace.position=" << record.position
                          << " candidate_top1=" << lower_id_argmax(candidate)
                          << " oracle_top1=" << lower_id_argmax(oracle)
                          << " kl=" << position_metrics.mean_kl
                          << " relative_nll_delta="
                          << position_metrics.relative_mean_nll_delta
                          << " max_logit_error="
                          << position_metrics.maximum_logit_error << '\n'
                          << std::flush;
            }

            const auto sampled = round.sampled_tokens();
            if (sampled.size() != 1 ||
                sampled[0] != lower_id_argmax(candidate)) {
                throw std::runtime_error(
                    "Phase 11 greedy sampled token does not match candidate argmax");
            }
            ++sampled_tokens;

            const std::array<LaneCommitDecision, 1> decisions{{
                LaneCommitDecision{.accept = true},
            }};
            round.commit(decisions);
            if (executor.committed_frontier(lane) !=
                static_cast<std::int32_t>(record.position + 1U)) {
                throw std::runtime_error(
                    "Phase 11 recurrent/state frontier did not advance");
            }

            const std::size_t completed = index + 1U;
            if ((completed % 64U) == 0U || completed == records.size()) {
                const double elapsed = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - qualification_started).count();
                const auto progress_stats =
                    flash_next_host_expert_execution_stats();
                const double positions_per_second =
                    elapsed > 0.0 ? static_cast<double>(completed) / elapsed : 0.0;
                const double pairs_per_second =
                    elapsed > 0.0
                        ? static_cast<double>(progress_stats.expert_pairs) / elapsed
                        : 0.0;
                const double remaining_seconds =
                    positions_per_second > 0.0
                        ? static_cast<double>(records.size() - completed) /
                              positions_per_second
                        : 0.0;
                std::cout << std::fixed << std::setprecision(3)
                          << "phase11.progress.positions=" << completed
                          << "/" << records.size()
                          << " elapsed_s=" << elapsed
                          << " positions_per_s=" << positions_per_second
                          << " expert_pairs_per_s=" << pairs_per_second
                          << " eta_s=" << remaining_seconds << '\n'
                          << std::flush;
            }
        }

        if (stage_trace_enabled) {
            const std::size_t begin =
                stage_trace_all_positions ? 0U :
                static_cast<std::size_t>(stage_trace_position);
            const std::size_t end =
                stage_trace_all_positions ? records.size() :
                std::min(records.size(), begin + 1U);
            for (std::size_t position = begin; position < end; ++position) {
                if (first_bad_stage_by_position[position].empty()) {
                    std::cout
                        << "phase11.stage_trace.position=" << position
                        << " first_bad_stage=none\n";
                } else {
                    std::cout << std::fixed << std::setprecision(8)
                              << "phase11.stage_trace.position=" << position
                              << " first_bad_stage="
                              << first_bad_stage_by_position[position]
                              << " cosine="
                              << first_bad_cosine_by_position[position]
                              << " nrmse="
                              << first_bad_nrmse_by_position[position]
                              << " max_error="
                              << first_bad_max_error_by_position[position]
                              << '\n';
                }
            }
        }

        executor.release_lane(lane);
        device.synchronize();

        const auto expert_stats =
            flash_next_host_expert_execution_stats();
        const std::uint64_t expected_layer_calls =
            static_cast<std::uint64_t>(records.size()) * 48ULL;
        const std::uint64_t expected_pairs =
            expected_layer_calls * 10ULL;
        std::cout << "phase11.host_expert.completed_layer_calls="
                  << expert_stats.completed_layer_calls << '\n'
                  << "phase11.host_expert.routed_tokens="
                  << expert_stats.routed_tokens << '\n'
                  << "phase11.host_expert.expert_pairs="
                  << expert_stats.expert_pairs << '\n'
                  << "phase11.sampled_tokens=" << sampled_tokens << '\n';

        if (expert_stats.completed_layer_calls != expected_layer_calls ||
            expert_stats.routed_tokens != expected_layer_calls ||
            expert_stats.expert_pairs != expected_pairs) {
            throw std::runtime_error(
                "Phase 11 did not execute every routed layer through the host expert path");
        }

        const Phase11OracleMetrics metrics =
            metrics_accumulator.finalize();
        print_metrics(metrics);
        Phase11OracleThresholds thresholds{};
        thresholds.minimum_positions = required_positions;
        if (!phase11_oracle_passes(metrics, thresholds)) {
            std::cerr << "FAIL: Phase 11 teacher-forced oracle gate\n";
            return 1;
        }

        std::cout << "PASS: Phase 11 whole-model eager host-backed vertical slice\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: Phase 11 vertical slice: "
                  << error.what() << '\n';
        return 1;
    }
#endif
}
