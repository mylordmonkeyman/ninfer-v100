#include "artifact/reader.h"
#include "targets/qwen3_8_flash_next/impl/cpu_expert_pool.h"
#include "targets/qwen3_8_flash_next/impl/cpu_expert_reference.h"

#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <optional>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

std::vector<std::byte> make_bank(std::int32_t rows, std::int32_t columns) {
    const std::array<std::uint64_t, 3> shape = {
        1, static_cast<std::uint64_t>(rows), static_cast<std::uint64_t>(columns)};
    const auto geometry = ninfer::artifact::block_scale_bank_geometry(
        ninfer::artifact::NumericFormat::NVFP4, shape);
    std::vector<std::byte> payload(geometry.encoded_bytes, std::byte{0});
    const std::size_t row_bytes = static_cast<std::size_t>(columns) / 2;
    for (int row = 0; row < rows; ++row) {
        for (std::size_t byte = 0; byte < row_bytes; ++byte) {
            const std::uint8_t lo = static_cast<std::uint8_t>(
                1U + ((static_cast<unsigned>(row) * 17U + static_cast<unsigned>(byte) * 5U) % 7U));
            const std::uint8_t hi = static_cast<std::uint8_t>(
                9U + ((static_cast<unsigned>(row) * 11U + static_cast<unsigned>(byte) * 3U) % 7U));
            payload[static_cast<std::size_t>(row) * row_bytes + byte] =
                static_cast<std::byte>(lo | static_cast<std::uint8_t>(hi << 4U));
        }
    }
    for (std::size_t i = 0; i < geometry.scale_plane_bytes; ++i) {
        payload[geometry.scale_plane_offset + i] =
            static_cast<std::byte>((i % 5U) == 0U ? 0x30U : 0x38U);
    }
    const float divisor = 8.0F;
    std::memcpy(payload.data() + geometry.weight_divisor_offset, &divisor, sizeof(divisor));
    return payload;
}

ninfer::targets::qwen3_8_flash_next::detail::Nvfp4ExpertBankView
make_bank_view(const std::vector<std::byte>& payload, std::int32_t rows,
               std::int32_t columns) {
    const std::array<std::uint64_t, 3> shape = {
        1, static_cast<std::uint64_t>(rows), static_cast<std::uint64_t>(columns)};
    const auto geometry = ninfer::artifact::block_scale_bank_geometry(
        ninfer::artifact::NumericFormat::NVFP4, shape);
    return {
        .codes = payload.data(),
        .scales = payload.data() + geometry.scale_plane_offset,
        .weight_scale_divisors = reinterpret_cast<const float*>(
            payload.data() + geometry.weight_divisor_offset),
        .experts = 1,
        .rows = rows,
        .columns = columns,
        .code_bytes_per_expert = geometry.code_plane_bytes,
        .scale_bytes_per_expert = geometry.scale_plane_bytes,
    };
}

ninfer::targets::qwen3_8_flash_next::detail::Nvfp4ExpertMatrixView
first_expert(const ninfer::targets::qwen3_8_flash_next::detail::Nvfp4ExpertBankView& bank) {
    return {
        .codes = bank.codes,
        .scales = bank.scales,
        .weight_scale_divisor = bank.weight_scale_divisors,
        .input_scale_divisor = 1.0F,
        .rows = bank.rows,
        .columns = bank.columns,
    };
}

std::uint16_t float_to_bf16(float value) {
    std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    bits += 0x7FFFU + ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>(bits >> 16U);
}

} // namespace

int main() {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    std::optional<v100_compare::Round> telemetry;
    if (v100_compare::level()) telemetry.emplace("ninfer", "cpu-group-fixture", "verify");
    if (!flash_next_cpu_nvfp4_avx2_available()) {
        std::cout << "SKIP: AVX2/FMA unavailable\n";
        return 77;
    }

    auto gate_payload = make_bank(1'280, 2'560);
    auto down_payload = make_bank(2'560, 640);
    const auto gate_bank = make_bank_view(gate_payload, 1'280, 2'560);
    const auto down_bank = make_bank_view(down_payload, 2'560, 640);
    const HostNvfp4ExpertPairView expert{
        .gate_up = first_expert(gate_bank), .down = first_expert(down_bank)};
    const double expert_bytes = static_cast<double>(gate_payload.size() + down_payload.size());

    std::array<std::array<std::uint16_t, kFlashNextExpertHidden>, kFlashNextCpuExpertGroupMax> inputs{};
    for (std::size_t token = 0; token < inputs.size(); ++token) {
        for (std::size_t column = 0; column < inputs[token].size(); ++column) {
            float value = static_cast<float>(static_cast<int>((column * 29U + token * 47U) % 257U) - 128) / 128.0F;
            if (column == 0) value = token == 0 ? 0.0F : (token == 1 ? 32.0F : -32.0F);
            if (column == 1) value = std::numeric_limits<float>::min();
            inputs[token][column] = float_to_bf16(value);
        }
    }
    HostExpertWorkerPool pool(4, true);

    for (std::size_t width : {2ULL, 3ULL, 4ULL}) {
        std::array<std::array<float, kFlashNextExpertHidden>, kFlashNextCpuExpertGroupMax> single{};
        std::array<std::array<float, kFlashNextExpertHidden>, kFlashNextCpuExpertGroupMax> grouped{};
        std::array<std::array<float, kFlashNextExpertHidden>, kFlashNextCpuExpertGroupMax> pooled{};
        std::array<const std::uint16_t*, kFlashNextCpuExpertGroupMax> input_ptrs{};
        std::array<float*, kFlashNextCpuExpertGroupMax> output_ptrs{};

        const auto single_start = Clock::now();
        for (std::size_t token = 0; token < width; ++token) {
            CpuNvfp4ExpertReferenceScratch scratch{};
            flash_next_cpu_nvfp4_expert_pair_avx2(expert, inputs[token], single[token], scratch);
            input_ptrs[token] = inputs[token].data();
            output_ptrs[token] = grouped[token].data();
        }
        const auto single_end = Clock::now();
        CpuNvfp4ExpertGroupScratch group_scratch{};
        const auto group_start = Clock::now();
        flash_next_cpu_nvfp4_expert_group_avx2(
            expert, {input_ptrs.data(), width}, {output_ptrs.data(), width}, group_scratch);
        const auto group_end = Clock::now();

        std::vector<HostExpertTask> tasks;
        for (std::size_t token = 0; token < width; ++token) {
            tasks.push_back(HostExpertTask{
                .expert = expert,
                .expert_id = 0,
                .route_id = static_cast<std::uint32_t>(17 + token),
                .input = inputs[token].data(),
                .output = pooled[token].data(),
            });
        }
        const auto pool_stats = pool.run(tasks, true, 0);
        if (pool_stats.groups != 1 || pool_stats.grouped_pairs != width ||
            pool_stats.weight_read_bytes != static_cast<std::uint64_t>(expert_bytes)) {
            std::cerr << "group width " << width << " returned invalid pool accounting\n";
            return 1;
        }

        for (std::size_t token = 0; token < width; ++token) {
            for (std::size_t row = 0; row < kFlashNextExpertHidden; ++row) {
                if (!std::isfinite(grouped[token][row]) ||
                    std::bit_cast<std::uint32_t>(grouped[token][row]) !=
                        std::bit_cast<std::uint32_t>(single[token][row]) ||
                    std::bit_cast<std::uint32_t>(pooled[token][row]) !=
                        std::bit_cast<std::uint32_t>(single[token][row])) {
                    std::cerr << "group width " << width << " token " << token
                              << " row " << row << " differs: "
                              << grouped[token][row] << " vs " << single[token][row] << '\n';
                    return 1;
                }
            }
        }
        const auto single_us = std::chrono::duration<double, std::micro>(single_end - single_start).count();
        const auto group_us = std::chrono::duration<double, std::micro>(group_end - group_start).count();
        std::cout << "{\"schema\":1,\"kind\":\"sv4.cpu_group\",\"width\":" << width
                  << ",\"single_us\":" << single_us << ",\"group_us\":" << group_us
                  << ",\"expert_pairs_per_second\":" << (1.0e6 * static_cast<double>(width) / group_us)
                  << ",\"expert_weight_bytes\":" << static_cast<std::uint64_t>(expert_bytes)
                  << ",\"effective_weight_read_gbps\":" << (expert_bytes / group_us / 1.0e3)
                  << ",\"exact\":true}\n";
    }

    // Exercise both whole-group jobs and two-phase row sharding, mixed expert
    // identities, singleton remainders, original output slots and pool reuse.
    float alternate_divisor = 16.0F;
    auto alternate = expert;
    alternate.gate_up.weight_scale_divisor = &alternate_divisor;
    alternate.down.weight_scale_divisor = &alternate_divisor;
    for (unsigned workers : {1U, 4U, 32U}) {
        HostExpertWorkerPool pool(workers, true);
        for (unsigned count : {1U, 2U, 3U, 4U, 5U, 17U}) {
            std::vector<std::array<float, kFlashNextExpertHidden>> expected(count), actual(count);
            std::vector<HostExpertTask> tasks;
            for (unsigned i = 0; i < count; ++i) {
                const auto& view = (i % 5 == 0) ? alternate : expert;
                CpuNvfp4ExpertReferenceScratch scratch;
                flash_next_cpu_nvfp4_expert_pair_avx2(
                    view, inputs[i % 4], expected[i], scratch);
                tasks.push_back(HostExpertTask{
                    .expert = view,
                    .expert_id = i % 5 == 0 ? 1 : 0,
                    .route_id = i,
                    .input = inputs[i % 4].data(),
                    .output = actual[i].data(),
                });
            }
            const auto stats = pool.run(tasks, true, 0);
            for (unsigned i = 0; i < count; ++i) {
                if (std::memcmp(actual[i].data(), expected[i].data(), sizeof(actual[i])) != 0) {
                    throw std::runtime_error("grouped worker pool changed an output slot");
                }
            }
            if (stats.grouped_pairs > count || stats.weight_read_bytes > count * 2'764'808ULL ||
                (count >= 3 && stats.groups == 0)) {
                throw std::runtime_error("grouped worker pool accounting mismatch");
            }
            // Failure drains all submitted workers; the next valid batch must run.
            auto invalid = tasks;
            invalid.back().input = nullptr;
            bool rejected = false;
            try {
                (void)pool.run(invalid, true);
            } catch (const std::exception&) {
                rejected = true;
            }
            if (!rejected) { throw std::runtime_error("null grouped input accepted"); }
            (void)pool.run(tasks, true, 0);
            for (unsigned i = 0; i < count; ++i) {
                if (std::memcmp(actual[i].data(), expected[i].data(), sizeof(actual[i])) != 0) {
                    throw std::runtime_error("grouped worker pool recovery changed output");
                }
            }
        }
    }
    // MTP-like sparse reuse: one width-four expert among independent
    // singletons. Exercise balanced row phases and whole-group dispatch on
    // either side of the 32-worker budget, with stable original output slots.
    for (unsigned count : {28U, 35U, 40U}) {
        HostExpertWorkerPool sparse_pool(32, true);
        std::vector<std::array<float, kFlashNextExpertHidden>> expected(count), actual(count);
        std::vector<HostExpertTask> tasks;
        for (unsigned i = 0; i < count; ++i) {
            const auto& view = i < 4 ? expert : alternate;
            CpuNvfp4ExpertReferenceScratch scratch;
            flash_next_cpu_nvfp4_expert_pair_avx2(view, inputs[i % 4], expected[i], scratch);
            tasks.push_back(HostExpertTask{
                .expert = view, .expert_id = static_cast<std::int32_t>(i < 4 ? 0 : i),
                .route_id = count - i, .input = inputs[i % 4].data(),
                .output = actual[i].data(),
            });
        }
        for (unsigned repeat = 0; repeat < 3; ++repeat) {
            const auto stats = sparse_pool.run(tasks, true, 0);
            if (stats.groups != 1 || stats.grouped_pairs != 4 ||
                stats.weight_read_bytes != (count - 3) * 2'764'808ULL) {
                throw std::runtime_error("sparse grouped worker accounting mismatch");
            }
            for (unsigned i = 0; i < count; ++i) {
                if (std::memcmp(actual[i].data(), expected[i].data(), sizeof(actual[i])) != 0) {
                    throw std::runtime_error("sparse grouped row schedule changed output bits");
                }
            }
        }
    }
    std::cout << "PASS: grouped worker pool mixed identities, row phases and failure recovery\n";
    std::cout << "PASS: grouped AVX2 widths 2/3/4 preserve single-token output bits\n";
    return 0;
}
