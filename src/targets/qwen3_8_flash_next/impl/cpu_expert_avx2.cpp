#include "targets/qwen3_8_flash_next/impl/cpu_expert_reference.h"

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#endif

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

std::size_t scale_row_base(std::int32_t row, std::int32_t columns) {
    const int k_tiles = columns / 64;
    const int row_tile = row / 128;
    const int row_inner = row % 128;
    return static_cast<std::size_t>(row_tile * k_tiles) * 512ULL +
           static_cast<std::size_t>(row_inner & 31) * 16ULL +
           static_cast<std::size_t>(row_inner >> 5) * 4ULL;
}

std::size_t scale_byte_offset_from_base(std::size_t row_base, int group) {
    return row_base + static_cast<std::size_t>(group >> 2) * 512ULL +
           static_cast<std::size_t>(group & 3);
}

void validate_matrix(const Nvfp4ExpertMatrixView& matrix, std::int32_t rows,
                     std::int32_t columns, const char* label) {
    if (matrix.codes == nullptr || matrix.scales == nullptr ||
        matrix.weight_scale_divisor == nullptr || matrix.rows != rows ||
        matrix.columns != columns) {
        throw std::invalid_argument(std::string("CPU NVFP4 AVX2: invalid ") + label +
                                    " expert matrix");
    }
    const float divisor = *matrix.weight_scale_divisor;
    if (!(divisor > 0.0F) || !std::isfinite(divisor)) {
        throw std::invalid_argument(std::string("CPU NVFP4 AVX2: invalid ") + label +
                                    " weight divisor");
    }
}

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))

struct ScaleTable {
    float values[256]{};
    ScaleTable() {
        for (int i = 0; i < 256; ++i) {
            values[i] = decode_e4m3fn(static_cast<std::uint8_t>(i));
        }
    }
};

const ScaleTable kScaleTable;

struct DecodedWeights16 {
    __m256 lo;
    __m256 hi;
};

__attribute__((target("avx2,fma")))
DecodedWeights16 decode_weights16(const std::byte* packed, float coefficient) {
    const __m128i nibble_mask = _mm_set1_epi8(0x0F);
    const __m128i q2_lut =
        _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
    const __m128i bytes =
        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(packed));
    const __m128i lo_nibbles = _mm_and_si128(bytes, nibble_mask);
    const __m128i hi_nibbles =
        _mm_and_si128(_mm_srli_epi16(bytes, 4), nibble_mask);
    const __m128i interleaved = _mm_unpacklo_epi8(lo_nibbles, hi_nibbles);
    const __m128i q2 = _mm_shuffle_epi8(q2_lut, interleaved);
    const __m256 scale = _mm256_set1_ps(coefficient * 0.5F);
    const __m256i ints_lo = _mm256_cvtepi8_epi32(q2);
    const __m256i ints_hi =
        _mm256_cvtepi8_epi32(_mm_srli_si128(q2, 8));
    return {
        .lo = _mm256_mul_ps(_mm256_cvtepi32_ps(ints_lo), scale),
        .hi = _mm256_mul_ps(_mm256_cvtepi32_ps(ints_hi), scale),
    };
}

__attribute__((target("avx2,fma")))
float hsum256(__m256 value) {
    const __m128 lo = _mm256_castps256_ps128(value);
    const __m128 hi = _mm256_extractf128_ps(value, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

__attribute__((target("avx2,fma")))
std::pair<float, float> gate_up_rows_avx2(
    const Nvfp4ExpertMatrixView& gate_up, std::int32_t row, const float* input) {
    const float inverse = 1.0F / *gate_up.weight_scale_divisor;
    const std::int32_t up_row =
        row + static_cast<std::int32_t>(kFlashNextExpertIntermediate);
    const std::size_t gate_scale_base =
        scale_row_base(row, gate_up.columns);
    const std::size_t up_scale_base =
        scale_row_base(up_row, gate_up.columns);
    __m256 gate0 = _mm256_setzero_ps();
    __m256 gate1 = _mm256_setzero_ps();
    __m256 up0 = _mm256_setzero_ps();
    __m256 up1 = _mm256_setzero_ps();
    const int groups = gate_up.columns / 16;

    for (int group = 0; group < groups; ++group) {
        const __m256 x0 = _mm256_loadu_ps(input + group * 16);
        const __m256 x1 = _mm256_loadu_ps(input + group * 16 + 8);

        const float gate_coeff =
            kScaleTable.values[static_cast<std::uint8_t>(
                gate_up.scales[scale_byte_offset_from_base(gate_scale_base, group)])] *
            inverse;
        const std::byte* gate_codes =
            gate_up.codes +
            static_cast<std::size_t>(row) *
                static_cast<std::size_t>(gate_up.columns / 2) +
            static_cast<std::size_t>(group) * 8ULL;
        const DecodedWeights16 gate_w =
            decode_weights16(gate_codes, gate_coeff);
        gate0 = _mm256_fmadd_ps(gate_w.lo, x0, gate0);
        gate1 = _mm256_fmadd_ps(gate_w.hi, x1, gate1);

        const float up_coeff =
            kScaleTable.values[static_cast<std::uint8_t>(
                gate_up.scales[scale_byte_offset_from_base(up_scale_base, group)])] *
            inverse;
        const std::byte* up_codes =
            gate_up.codes +
            static_cast<std::size_t>(up_row) *
                static_cast<std::size_t>(gate_up.columns / 2) +
            static_cast<std::size_t>(group) * 8ULL;
        const DecodedWeights16 up_w =
            decode_weights16(up_codes, up_coeff);
        up0 = _mm256_fmadd_ps(up_w.lo, x0, up0);
        up1 = _mm256_fmadd_ps(up_w.hi, x1, up1);
    }

    return {
        hsum256(_mm256_add_ps(gate0, gate1)),
        hsum256(_mm256_add_ps(up0, up1)),
    };
}

__attribute__((target("avx2,fma")))
float down_row_avx2(const Nvfp4ExpertMatrixView& down, std::int32_t row,
                    const float* activation) {
    const float inverse = 1.0F / *down.weight_scale_divisor;
    const std::size_t scale_base = scale_row_base(row, down.columns);
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    const int groups = down.columns / 16;

    for (int group = 0; group < groups; ++group) {
        const float coeff =
            kScaleTable.values[static_cast<std::uint8_t>(
                down.scales[scale_byte_offset_from_base(scale_base, group)])] *
            inverse;
        const std::byte* packed =
            down.codes +
            static_cast<std::size_t>(row) *
                static_cast<std::size_t>(down.columns / 2) +
            static_cast<std::size_t>(group) * 8ULL;
        const DecodedWeights16 weights = decode_weights16(packed, coeff);
        acc0 = _mm256_fmadd_ps(
            weights.lo, _mm256_loadu_ps(activation + group * 16), acc0);
        acc1 = _mm256_fmadd_ps(
            weights.hi, _mm256_loadu_ps(activation + group * 16 + 8), acc1);
    }

    return hsum256(_mm256_add_ps(acc0, acc1));
}

#endif

} // namespace

bool flash_next_cpu_nvfp4_avx2_available() noexcept {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    static const bool available = [] {
        __builtin_cpu_init();
        return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    }();
    return available;
#else
    return false;
#endif
}

void flash_next_cpu_nvfp4_expert_pair_avx2(
    const HostNvfp4ExpertPairView& expert,
    std::span<const std::uint16_t> input_bf16,
    std::span<float> output,
    CpuNvfp4ExpertReferenceScratch& scratch) {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    if (!flash_next_cpu_nvfp4_avx2_available()) {
        throw std::runtime_error("CPU NVFP4 AVX2/FMA backend is unavailable");
    }
    if (input_bf16.size() != kFlashNextExpertHidden ||
        output.size() != kFlashNextExpertHidden) {
        throw std::invalid_argument("CPU NVFP4 AVX2: invalid activation/output length");
    }
    validate_matrix(expert.gate_up, 1'280, 2'560, "gate/up");
    validate_matrix(expert.down, 2'560, 640, "down");

    for (std::size_t i = 0; i < input_bf16.size(); ++i) {
        scratch.input[i] = bf16_to_float(input_bf16[i]);
    }

    for (std::int32_t row = 0;
         row < static_cast<std::int32_t>(kFlashNextExpertIntermediate); ++row) {
        const auto [gate, up] =
            gate_up_rows_avx2(expert.gate_up, row, scratch.input.data());
        scratch.intermediate[static_cast<std::size_t>(row)] =
            round_to_bf16_rne(gate / (1.0F + std::exp(-gate)) * up);
    }

    for (std::int32_t row = 0;
         row < static_cast<std::int32_t>(kFlashNextExpertHidden); ++row) {
        output[static_cast<std::size_t>(row)] =
            down_row_avx2(expert.down, row, scratch.intermediate.data());
    }
#else
    (void)expert;
    (void)input_bf16;
    (void)output;
    (void)scratch;
    throw std::runtime_error("CPU NVFP4 AVX2/FMA backend is unavailable on this architecture");
#endif
}

} // namespace ninfer::targets::qwen3_8_flash_next::detail
