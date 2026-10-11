#include "targets/qwen3_8_flash_next/impl/cpu_expert_reference.h"

#include <bit>
#include <algorithm>
#include <cstdlib>
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

bool integer_blocks_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_V100_CPU_EXACT_S16");
        if (!value || !*value || std::string(value) == "0") return false;
        if (std::string(value) == "1") return true;
        throw std::invalid_argument("NINFER_V100_CPU_EXACT_S16 must be 0 or 1");
    }();
    return enabled;
}

// A power-of-two block scale represents BF16 exactly when its exponent range
// fits signed 16 bits. Wider blocks retain the original floating-point input.
// This is an execution representation, never an activation quantization.
struct IntegerInput {
    const float* input = nullptr;
    std::array<std::int16_t, kFlashNextExpertHidden> codes{};
    std::array<float, kFlashNextExpertHidden / 16> scales{};
    std::array<bool, kFlashNextExpertHidden / 16> exact{};

    IntegerInput() = default;
    IntegerInput(const float* values, std::size_t columns) : input(values) {
        for (std::size_t group = 0; group < columns / 16; ++group) {
            float maximum = 0;
            bool finite = true;
            for (std::size_t i = 0; i < 16; ++i) {
                const float x = values[group * 16 + i];
                finite &= std::isfinite(x);
                maximum = std::max(maximum, std::abs(x));
            }
            if (!finite) continue;
            if (maximum == 0) { exact[group] = true; continue; }
            int exponent = 0;
            std::frexp(maximum, &exponent);
            const float scale = std::ldexp(1.0F, exponent - 15);
            if (scale < std::numeric_limits<float>::min() || !std::isfinite(scale)) continue;
            bool representable = true;
            for (std::size_t i = 0; i < 16; ++i) {
                const float x = values[group * 16 + i];
                const float integer = x / scale;
                // Bounds make the cast defined even for unsupported inputs.
                if (!(integer >= -32767 && integer <= 32767)) {
                    representable = false; break;
                }
                const auto q = static_cast<std::int16_t>(integer);
                codes[group * 16 + i] = q;
                representable &= static_cast<float>(q) * scale == x;
            }
            scales[group] = scale;
            exact[group] = representable;
        }
    }
};

__attribute__((target("avx2,fma")))
__m256i signed_codes16(const std::byte* packed) {
    const __m128i bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(packed));
    const __m128i mask = _mm_set1_epi8(15);
    const __m128i indices = _mm_unpacklo_epi8(_mm_and_si128(bytes, mask),
        _mm_and_si128(_mm_srli_epi16(bytes, 4), mask));
    const __m128i lut = _mm_setr_epi8(0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12);
    return _mm256_cvtepi8_epi16(_mm_shuffle_epi8(lut, indices));
}

__attribute__((target("avx2,fma")))
__m256 block_product(__m256i weights, const IntegerInput& input, int group,
                     float coefficient) {
    if (input.exact[group]) {
        const __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
            input.codes.data() + group * 16));
        // Each pair sum is bounded by 2*12*32767, well within signed int32.
        const __m256 pairs = _mm256_cvtepi32_ps(_mm256_madd_epi16(weights, x));
        return _mm256_mul_ps(pairs, _mm256_set1_ps(coefficient * .5F * input.scales[group]));
    }
    const __m128i low = _mm256_castsi256_si128(weights);
    const __m128i high = _mm256_extracti128_si256(weights, 1);
    const __m256 scale = _mm256_set1_ps(coefficient * .5F);
    const __m256 w0 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(low)), scale);
    const __m256 w1 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(high)), scale);
    const __m256 p0 = _mm256_mul_ps(w0, _mm256_loadu_ps(input.input + group * 16));
    const __m256 p1 = _mm256_mul_ps(w1, _mm256_loadu_ps(input.input + group * 16 + 8));
    return _mm256_hadd_ps(p0, p1);
}

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
void integer_gate_rows(const Nvfp4ExpertMatrixView& matrix,
    CpuNvfp4ExpertReferenceScratch* scratch, std::size_t count,
    std::size_t begin, std::size_t end) {
    std::array<IntegerInput, kFlashNextCpuExpertGroupMax> inputs;
    for (std::size_t t = 0; t < count; ++t)
        inputs[t] = IntegerInput(scratch[t].input.data(), matrix.columns);
    const float inverse = 1.0F / *matrix.weight_scale_divisor;
    for (std::size_t row = begin; row < end; ++row) {
        __m256 gate[kFlashNextCpuExpertGroupMax], up[kFlashNextCpuExpertGroupMax];
        for (std::size_t t = 0; t < count; ++t) gate[t] = up[t] = _mm256_setzero_ps();
        const std::size_t up_row = row + kFlashNextExpertIntermediate;
        const auto gate_base = scale_row_base(row, matrix.columns);
        const auto up_base = scale_row_base(up_row, matrix.columns);
        for (int group = 0; group < matrix.columns / 16; ++group) {
            const auto gw = signed_codes16(matrix.codes + row * (matrix.columns / 2) + group * 8);
            const auto uw = signed_codes16(matrix.codes + up_row * (matrix.columns / 2) + group * 8);
            const float gs = kScaleTable.values[static_cast<std::uint8_t>(
                matrix.scales[scale_byte_offset_from_base(gate_base, group)])] * inverse;
            const float us = kScaleTable.values[static_cast<std::uint8_t>(
                matrix.scales[scale_byte_offset_from_base(up_base, group)])] * inverse;
            for (std::size_t t = 0; t < count; ++t) {
                gate[t] = _mm256_add_ps(gate[t], block_product(gw, inputs[t], group, gs));
                up[t] = _mm256_add_ps(up[t], block_product(uw, inputs[t], group, us));
            }
        }
        for (std::size_t t = 0; t < count; ++t) {
            const float g = hsum256(gate[t]), u = hsum256(up[t]);
            scratch[t].intermediate[row] = round_to_bf16_rne(g / (1.0F + std::exp(-g)) * u);
        }
    }
}

__attribute__((target("avx2,fma")))
void integer_down_rows(const Nvfp4ExpertMatrixView& matrix,
    const CpuNvfp4ExpertReferenceScratch* scratch, std::span<float* const> outputs,
    std::size_t begin, std::size_t end) {
    std::array<IntegerInput, kFlashNextCpuExpertGroupMax> inputs;
    // All gate rows complete before this phase. Each down shard owns its
    // prepared blocks, so there is no mutation/race on shared intermediate state.
    for (std::size_t t = 0; t < outputs.size(); ++t)
        inputs[t] = IntegerInput(scratch[t].intermediate.data(), matrix.columns);
    const float inverse = 1.0F / *matrix.weight_scale_divisor;
    for (std::size_t row = begin; row < end; ++row) {
        __m256 acc[kFlashNextCpuExpertGroupMax];
        for (std::size_t t = 0; t < outputs.size(); ++t) acc[t] = _mm256_setzero_ps();
        const auto base = scale_row_base(row, matrix.columns);
        for (int group = 0; group < matrix.columns / 16; ++group) {
            const auto w = signed_codes16(matrix.codes + row * (matrix.columns / 2) + group * 8);
            const float s = kScaleTable.values[static_cast<std::uint8_t>(
                matrix.scales[scale_byte_offset_from_base(base, group)])] * inverse;
            for (std::size_t t = 0; t < outputs.size(); ++t)
                acc[t] = _mm256_add_ps(acc[t], block_product(w, inputs[t], group, s));
        }
        for (std::size_t t = 0; t < outputs.size(); ++t) outputs[t][row] = hsum256(acc[t]);
    }
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

__attribute__((target("avx2,fma")))
void gate_up_row_group_avx2(
    const Nvfp4ExpertMatrixView& gate_up, std::int32_t row,
    CpuNvfp4ExpertGroupScratch& scratch, std::size_t token_count) {
    const float inverse = 1.0F / *gate_up.weight_scale_divisor;
    const std::int32_t up_row =
        row + static_cast<std::int32_t>(kFlashNextExpertIntermediate);
    const std::size_t gate_scale_base = scale_row_base(row, gate_up.columns);
    const std::size_t up_scale_base = scale_row_base(up_row, gate_up.columns);
    __m256 gate0[kFlashNextCpuExpertGroupMax];
    __m256 gate1[kFlashNextCpuExpertGroupMax];
    __m256 up0[kFlashNextCpuExpertGroupMax];
    __m256 up1[kFlashNextCpuExpertGroupMax];
    for (std::size_t token = 0; token < token_count; ++token) {
        gate0[token] = gate1[token] = up0[token] = up1[token] = _mm256_setzero_ps();
    }
    const int groups = gate_up.columns / 16;
    for (int group = 0; group < groups; ++group) {
        const float gate_coeff =
            kScaleTable.values[static_cast<std::uint8_t>(
                gate_up.scales[scale_byte_offset_from_base(gate_scale_base, group)])] * inverse;
        const std::byte* gate_codes = gate_up.codes +
            static_cast<std::size_t>(row) * static_cast<std::size_t>(gate_up.columns / 2) +
            static_cast<std::size_t>(group) * 8ULL;
        const DecodedWeights16 gate_w = decode_weights16(gate_codes, gate_coeff);
        const float up_coeff =
            kScaleTable.values[static_cast<std::uint8_t>(
                gate_up.scales[scale_byte_offset_from_base(up_scale_base, group)])] * inverse;
        const std::byte* up_codes = gate_up.codes +
            static_cast<std::size_t>(up_row) * static_cast<std::size_t>(gate_up.columns / 2) +
            static_cast<std::size_t>(group) * 8ULL;
        const DecodedWeights16 up_w = decode_weights16(up_codes, up_coeff);
        for (std::size_t token = 0; token < token_count; ++token) {
            const float* input = scratch.tokens[token].input.data();
            const __m256 x0 = _mm256_loadu_ps(input + group * 16);
            const __m256 x1 = _mm256_loadu_ps(input + group * 16 + 8);
            gate0[token] = _mm256_fmadd_ps(gate_w.lo, x0, gate0[token]);
            gate1[token] = _mm256_fmadd_ps(gate_w.hi, x1, gate1[token]);
            up0[token] = _mm256_fmadd_ps(up_w.lo, x0, up0[token]);
            up1[token] = _mm256_fmadd_ps(up_w.hi, x1, up1[token]);
        }
    }
    for (std::size_t token = 0; token < token_count; ++token) {
        const float gate = hsum256(_mm256_add_ps(gate0[token], gate1[token]));
        const float up = hsum256(_mm256_add_ps(up0[token], up1[token]));
        scratch.tokens[token].intermediate[static_cast<std::size_t>(row)] =
            round_to_bf16_rne(gate / (1.0F + std::exp(-gate)) * up);
    }
}

__attribute__((target("avx2,fma")))
void down_row_group_avx2(
    const Nvfp4ExpertMatrixView& down, std::int32_t row,
    const CpuNvfp4ExpertGroupScratch& scratch, std::span<float* const> outputs) {
    const float inverse = 1.0F / *down.weight_scale_divisor;
    const std::size_t scale_base = scale_row_base(row, down.columns);
    __m256 acc0[kFlashNextCpuExpertGroupMax];
    __m256 acc1[kFlashNextCpuExpertGroupMax];
    for (std::size_t token = 0; token < outputs.size(); ++token) {
        acc0[token] = acc1[token] = _mm256_setzero_ps();
    }
    const int groups = down.columns / 16;
    for (int group = 0; group < groups; ++group) {
        const float coeff =
            kScaleTable.values[static_cast<std::uint8_t>(
                down.scales[scale_byte_offset_from_base(scale_base, group)])] * inverse;
        const std::byte* packed = down.codes +
            static_cast<std::size_t>(row) * static_cast<std::size_t>(down.columns / 2) +
            static_cast<std::size_t>(group) * 8ULL;
        const DecodedWeights16 weights = decode_weights16(packed, coeff);
        for (std::size_t token = 0; token < outputs.size(); ++token) {
            const float* activation = scratch.tokens[token].intermediate.data();
            acc0[token] = _mm256_fmadd_ps(
                weights.lo, _mm256_loadu_ps(activation + group * 16), acc0[token]);
            acc1[token] = _mm256_fmadd_ps(
                weights.hi, _mm256_loadu_ps(activation + group * 16 + 8), acc1[token]);
        }
    }
    for (std::size_t token = 0; token < outputs.size(); ++token) {
        outputs[token][static_cast<std::size_t>(row)] =
            hsum256(_mm256_add_ps(acc0[token], acc1[token]));
    }
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

void flash_next_cpu_nvfp4_expert_prepare_avx2(
    const HostNvfp4ExpertPairView& expert, std::span<const std::uint16_t> input,
    CpuNvfp4ExpertReferenceScratch& scratch) {
    if (!flash_next_cpu_nvfp4_avx2_available()) {
        throw std::runtime_error("CPU NVFP4 AVX2/FMA backend is unavailable");
    }
    if (input.size() != kFlashNextExpertHidden) {
        throw std::invalid_argument("CPU NVFP4 AVX2: invalid input length");
    }
    validate_matrix(expert.gate_up, 1'280, 2'560, "gate/up");
    validate_matrix(expert.down, 2'560, 640, "down");
    for (std::size_t i = 0; i < input.size(); ++i) {
        scratch.input[i] = bf16_to_float(input[i]);
    }
}

void flash_next_cpu_nvfp4_expert_gate_up_rows_avx2(
    const HostNvfp4ExpertPairView& expert, CpuNvfp4ExpertReferenceScratch& scratch,
    std::size_t begin, std::size_t end) {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    if (integer_blocks_enabled()) {
        integer_gate_rows(expert.gate_up, &scratch, 1, begin, end);
        return;
    }
    for (std::size_t row = begin; row < end; ++row) {
        const auto [gate, up] =
            gate_up_rows_avx2(expert.gate_up, static_cast<std::int32_t>(row), scratch.input.data());
        scratch.intermediate[row] =
            round_to_bf16_rne(gate / (1.0F + std::exp(-gate)) * up);
    }
#else
    (void)expert; (void)scratch; (void)begin; (void)end;
    throw std::runtime_error("CPU NVFP4 AVX2/FMA backend is unavailable");
#endif
}

void flash_next_cpu_nvfp4_expert_down_rows_avx2(
    const HostNvfp4ExpertPairView& expert, const CpuNvfp4ExpertReferenceScratch& scratch,
    std::span<float> output, std::size_t begin, std::size_t end) {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    if (integer_blocks_enabled()) {
        float* pointer = output.data();
        integer_down_rows(expert.down, &scratch, {&pointer, 1}, begin, end);
        return;
    }
    for (std::size_t row = begin; row < end; ++row) {
        output[row] = down_row_avx2(expert.down, static_cast<std::int32_t>(row), scratch.intermediate.data());
    }
#else
    (void)expert; (void)scratch; (void)output; (void)begin; (void)end;
    throw std::runtime_error("CPU NVFP4 AVX2/FMA backend is unavailable");
#endif
}

void flash_next_cpu_nvfp4_expert_pair_avx2(
    const HostNvfp4ExpertPairView& expert,
    std::span<const std::uint16_t> input_bf16,
    std::span<float> output,
    CpuNvfp4ExpertReferenceScratch& scratch) {
    if (output.size() != kFlashNextExpertHidden) {
        throw std::invalid_argument("CPU NVFP4 AVX2: invalid output length");
    }
    flash_next_cpu_nvfp4_expert_prepare_avx2(expert, input_bf16, scratch);
    flash_next_cpu_nvfp4_expert_gate_up_rows_avx2(expert, scratch, 0, kFlashNextExpertIntermediate);
    flash_next_cpu_nvfp4_expert_down_rows_avx2(expert, scratch, output, 0, kFlashNextExpertHidden);
}

void flash_next_cpu_nvfp4_expert_group_prepare_avx2(
    const HostNvfp4ExpertPairView& expert,
    std::span<const std::uint16_t* const> inputs,
    CpuNvfp4ExpertGroupScratch& scratch) {
    if (inputs.size() < 2 || inputs.size() > kFlashNextCpuExpertGroupMax) {
        throw std::invalid_argument("CPU NVFP4 AVX2: grouped width must be in [2, 4]");
    }
    for (std::size_t token = 0; token < inputs.size(); ++token) {
        if (inputs[token] == nullptr) {
            throw std::invalid_argument("CPU NVFP4 AVX2: null grouped input");
        }
        flash_next_cpu_nvfp4_expert_prepare_avx2(
            expert, {inputs[token], kFlashNextExpertHidden}, scratch.tokens[token]);
    }
}

void flash_next_cpu_nvfp4_expert_group_gate_up_rows_avx2(
    const HostNvfp4ExpertPairView& expert, CpuNvfp4ExpertGroupScratch& scratch,
    std::size_t token_count, std::size_t begin, std::size_t end) {
    if (token_count < 2 || token_count > kFlashNextCpuExpertGroupMax ||
        begin > end || end > kFlashNextExpertIntermediate) {
        throw std::invalid_argument("CPU NVFP4 AVX2: invalid grouped gate/up range");
    }
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    if (integer_blocks_enabled()) {
        integer_gate_rows(expert.gate_up, scratch.tokens.data(), token_count, begin, end);
        return;
    }
    for (std::size_t row = begin; row < end; ++row) {
        gate_up_row_group_avx2(expert.gate_up, static_cast<std::int32_t>(row), scratch, token_count);
    }
#else
    (void)expert; (void)scratch; (void)begin; (void)end;
    throw std::runtime_error("CPU NVFP4 AVX2/FMA backend is unavailable");
#endif
}

void flash_next_cpu_nvfp4_expert_group_down_rows_avx2(
    const HostNvfp4ExpertPairView& expert, const CpuNvfp4ExpertGroupScratch& scratch,
    std::span<float* const> outputs, std::size_t begin, std::size_t end) {
    if (outputs.size() < 2 || outputs.size() > kFlashNextCpuExpertGroupMax ||
        begin > end || end > kFlashNextExpertHidden) {
        throw std::invalid_argument("CPU NVFP4 AVX2: invalid grouped down range");
    }
    for (float* output : outputs) {
        if (output == nullptr) {
            throw std::invalid_argument("CPU NVFP4 AVX2: null grouped output");
        }
    }
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    if (integer_blocks_enabled()) {
        integer_down_rows(expert.down, scratch.tokens.data(), outputs, begin, end);
        return;
    }
    for (std::size_t row = begin; row < end; ++row) {
        down_row_group_avx2(expert.down, static_cast<std::int32_t>(row), scratch, outputs);
    }
#else
    (void)expert; (void)scratch; (void)begin; (void)end;
    throw std::runtime_error("CPU NVFP4 AVX2/FMA backend is unavailable");
#endif
}

void flash_next_cpu_nvfp4_expert_group_avx2(
    const HostNvfp4ExpertPairView& expert,
    std::span<const std::uint16_t* const> inputs,
    std::span<float* const> outputs,
    CpuNvfp4ExpertGroupScratch& scratch) {
    if (inputs.size() != outputs.size()) {
        throw std::invalid_argument("CPU NVFP4 AVX2: grouped input/output width mismatch");
    }
    flash_next_cpu_nvfp4_expert_group_prepare_avx2(expert, inputs, scratch);
    flash_next_cpu_nvfp4_expert_group_gate_up_rows_avx2(
        expert, scratch, inputs.size(), 0, kFlashNextExpertIntermediate);
    flash_next_cpu_nvfp4_expert_group_down_rows_avx2(
        expert, scratch, outputs, 0, kFlashNextExpertHidden);
}

} // namespace ninfer::targets::qwen3_8_flash_next::detail
