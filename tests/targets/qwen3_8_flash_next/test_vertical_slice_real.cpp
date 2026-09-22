#include "core/device.h"
#include "targets/qwen3_8_flash_next/impl/load/loader.h"
#include "targets/qwen3_8_flash_next/impl/moe.h"
#include "targets/qwen3_8_flash_next/impl/runtime_state.h"
#include "targets/qwen3_8_flash_next/impl/text_decode.h"
#include "targets/qwen3_8_flash_next/impl/text_executor.h"
#include "targets/qwen3_8_flash_next/impl/vertical_slice.h"

#include "nlohmann/json.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
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
        std::string first_bad_stage;
        double first_bad_cosine = 1.0;
        double first_bad_nrmse = 0.0;
        double first_bad_max_error = 0.0;

        FlashNextDecodeStateSink stage_sink;
        stage_sink.on_state = [&](std::string_view name, const ninfer::Tensor& tensor) {
            if (!stage_trace_enabled) {
                return;
            }
            char pos_dir[32];
            std::snprintf(
                pos_dir, sizeof(pos_dir), "pos%04u", stage_trace_position);
            const fs::path expected_path =
                stage_root / pos_dir / (std::string(name) + ".bin");
            if (!fs::is_regular_file(expected_path)) {
                return;
            }

            const std::size_t count = tensor.numel();
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
            if (tensor.dtype == ninfer::DType::BF16) {
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

            long double dot = 0.0L;
            long double candidate_sq = 0.0L;
            long double expected_sq = 0.0L;
            long double error_sq = 0.0L;
            double max_error = 0.0;
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
                cosine >= 0.99999 && nrmse <= 2.0e-3;

            std::cout << std::fixed << std::setprecision(8)
                      << "phase11.stage_trace.position="
                      << stage_trace_position
                      << " stage=" << name
                      << " cosine=" << cosine
                      << " nrmse=" << nrmse
                      << " max_error=" << max_error
                      << " pass=" << (pass ? 1 : 0)
                      << '\n' << std::flush;

            if (!pass && first_bad_stage.empty()) {
                first_bad_stage = std::string(name);
                first_bad_cosine = cosine;
                first_bad_nrmse = nrmse;
                first_bad_max_error = max_error;
            }
        };

        for (std::size_t index = 0; index < records.size(); ++index) {
            const OracleRecord& record = records[index];
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

            const FlashNextDecodeStateSink* round_sink =
                stage_trace_enabled &&
                        record.position == stage_trace_position
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
            if (first_bad_stage.empty()) {
                std::cout
                    << "phase11.stage_trace.first_bad_stage=none\n";
            } else {
                std::cout << std::fixed << std::setprecision(8)
                          << "phase11.stage_trace.first_bad_stage="
                          << first_bad_stage
                          << " cosine=" << first_bad_cosine
                          << " nrmse=" << first_bad_nrmse
                          << " max_error=" << first_bad_max_error
                          << '\n';
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
