#include "targets/qwen3_8_flash_next/impl/cpu_expert_reference.h"

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::targets::qwen3_8_flash_next::detail {
namespace {

float bf16_to_float(std::uint16_t word) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(word) << 16U);
}

float round_to_bf16_rne(float value) {
    std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    if ((bits & 0x7F800000U) != 0x7F800000U) {
        bits += 0x7FFFU + ((bits >> 16U) & 1U);
        bits &= 0xFFFF0000U;
    }
    return std::bit_cast<float>(bits);
}

float decode_e2m1(std::uint8_t code) {
    static constexpr float positive[8] = {
        0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F,
    };
    const float magnitude = positive[code & 7U];
    return (code & 8U) != 0U ? -magnitude : magnitude;
}

float decode_e4m3fn(std::uint8_t code) {
    const bool negative = (code & 0x80U) != 0U;
    const int exponent = (code >> 3U) & 0x0F;
    const int mantissa = code & 0x07U;
    float value = 0.0F;
    if (exponent == 0) {
        value = mantissa == 0 ? 0.0F : std::ldexp(static_cast<float>(mantissa), -9);
    } else if (exponent < 15) {
        value = std::ldexp(1.0F + static_cast<float>(mantissa) / 8.0F, exponent - 7);
    } else if (mantissa < 7) {
        value = std::ldexp(1.0F + static_cast<float>(mantissa) / 8.0F, 8);
    } else {
        return std::numeric_limits<float>::quiet_NaN();
    }
    return negative ? -value : value;
}

std::size_t scale_offset(std::int32_t row, std::int32_t group, std::int32_t columns) {
    const int k_tiles = columns / 64;
    const int row_tile = row / 128;
    const int row_inner = row % 128;
    const int scale_tile = group / 4;
    const int scale_lane = group % 4;
    return static_cast<std::size_t>(row_tile * k_tiles + scale_tile) * 512ULL +
           static_cast<std::size_t>(row_inner % 32) * 16ULL +
           static_cast<std::size_t>(row_inner / 32) * 4ULL +
           static_cast<std::size_t>(scale_lane);
}

void validate_matrix(const Nvfp4ExpertMatrixView& matrix, std::int32_t rows,
                     std::int32_t columns, const char* label) {
    if (matrix.codes == nullptr || matrix.scales == nullptr ||
        matrix.weight_scale_divisor == nullptr || matrix.rows != rows ||
        matrix.columns != columns) {
        throw std::invalid_argument(std::string("CPU NVFP4 reference: invalid ") + label +
                                    " expert matrix");
    }
    const float divisor = *matrix.weight_scale_divisor;
    if (!(divisor > 0.0F) || !std::isfinite(divisor)) {
        throw std::invalid_argument(std::string("CPU NVFP4 reference: invalid ") + label +
                                    " weight divisor");
    }
}

float matrix_row(const Nvfp4ExpertMatrixView& matrix, std::int32_t row,
                 const float* input) {
    const float inverse_divisor = 1.0F / *matrix.weight_scale_divisor;
    const std::size_t code_row =
        static_cast<std::size_t>(row) * static_cast<std::size_t>(matrix.columns / 2);
    float sum = 0.0F;
    for (int group = 0; group < matrix.columns / 16; ++group) {
        const float scale =
            decode_e4m3fn(static_cast<std::uint8_t>(matrix.scales[
                scale_offset(row, group, matrix.columns)])) *
            inverse_divisor;
        const std::byte* packed =
            matrix.codes + code_row + static_cast<std::size_t>(group) * 8ULL;
        for (int lane = 0; lane < 16; ++lane) {
            const std::uint8_t byte =
                static_cast<std::uint8_t>(packed[static_cast<std::size_t>(lane >> 1)]);
            const std::uint8_t code =
                (lane & 1) == 0 ? (byte & 0x0FU) : (byte >> 4U);
            sum = std::fma(decode_e2m1(code) * scale, input[group * 16 + lane], sum);
        }
    }
    return sum;
}

} // namespace

namespace {

void expert_pair_reference_compute(
    const HostNvfp4ExpertPairView& expert,
    const float* input,
    std::span<float> output,
    CpuNvfp4ExpertReferenceScratch& scratch,
    bool round_intermediate_to_bf16) {
    if (output.size() != kFlashNextExpertHidden) {
        throw std::invalid_argument("CPU NVFP4 reference: invalid output length");
    }
    validate_matrix(expert.gate_up, 1'280, 2'560, "gate/up");
    validate_matrix(expert.down, 2'560, 640, "down");

    for (int row = 0; row < static_cast<int>(kFlashNextExpertIntermediate); ++row) {
        const float gate = matrix_row(expert.gate_up, row, input);
        const float up = matrix_row(
            expert.gate_up, row + static_cast<int>(kFlashNextExpertIntermediate), input);
        const float activation = gate / (1.0F + std::exp(-gate)) * up;
        scratch.intermediate[static_cast<std::size_t>(row)] =
            round_intermediate_to_bf16 ? round_to_bf16_rne(activation) : activation;
    }

    for (int row = 0; row < static_cast<int>(kFlashNextExpertHidden); ++row) {
        output[static_cast<std::size_t>(row)] =
            matrix_row(expert.down, row, scratch.intermediate.data());
    }
}

void expert_pair_reference_impl(
    const HostNvfp4ExpertPairView& expert,
    std::span<const std::uint16_t> input_bf16,
    std::span<float> output,
    CpuNvfp4ExpertReferenceScratch& scratch,
    bool round_intermediate_to_bf16) {
    if (input_bf16.size() != kFlashNextExpertHidden) {
        throw std::invalid_argument("CPU NVFP4 reference: invalid activation length");
    }
    for (std::size_t i = 0; i < input_bf16.size(); ++i) {
        scratch.input[i] = bf16_to_float(input_bf16[i]);
    }
    expert_pair_reference_compute(
        expert, scratch.input.data(), output, scratch, round_intermediate_to_bf16);
}

} // namespace

void flash_next_cpu_nvfp4_expert_pair_reference(
    const HostNvfp4ExpertPairView& expert,
    std::span<const std::uint16_t> input_bf16,
    std::span<float> output,
    CpuNvfp4ExpertReferenceScratch& scratch) {
    expert_pair_reference_impl(expert, input_bf16, output, scratch, true);
}

void flash_next_cpu_nvfp4_expert_pair_reference_fp32_intermediate(
    const HostNvfp4ExpertPairView& expert,
    std::span<const std::uint16_t> input_bf16,
    std::span<float> output,
    CpuNvfp4ExpertReferenceScratch& scratch) {
    expert_pair_reference_impl(expert, input_bf16, output, scratch, false);
}

void flash_next_cpu_nvfp4_expert_pair_reference_fp32_input(
    const HostNvfp4ExpertPairView& expert,
    std::span<const float> input_fp32,
    std::span<float> output,
    CpuNvfp4ExpertReferenceScratch& scratch) {
    if (input_fp32.size() != kFlashNextExpertHidden) {
        throw std::invalid_argument("CPU NVFP4 reference: invalid FP32 activation length");
    }
    expert_pair_reference_compute(expert, input_fp32.data(), output, scratch, true);
}

} // namespace ninfer::targets::qwen3_8_flash_next::detail
