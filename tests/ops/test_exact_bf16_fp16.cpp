#include "ops/common/exact_bf16_fp16.h"
#include "ops/common/rounded_bf16_fp16.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <utility>

int main() {
    // Independent mathematical oracle: enumerate every finite FP16 value from
    // its sign/exponent/significand formula, then compare represented FP32 bits.
    std::unordered_map<std::uint32_t, std::uint16_t> finite;
    for (unsigned bits = 0; bits < 65536; ++bits) {
        const unsigned exponent = (bits >> 10) & 31U;
        if (exponent == 31) { continue; }
        const unsigned fraction = bits & 1023U;
        const double magnitude = exponent == 0
            ? std::ldexp(static_cast<double>(fraction), -24)
            : std::ldexp(static_cast<double>(1024 + fraction), static_cast<int>(exponent) - 25);
        const float value = static_cast<float>((bits & 0x8000U) ? -magnitude : magnitude);
        finite.emplace(std::bit_cast<std::uint32_t>(value), static_cast<std::uint16_t>(bits));
    }
    unsigned eligible = 0;
    for (unsigned bits = 0; bits < 65536; ++bits) {
        const auto expected = finite.find(bits << 16);
        std::uint16_t converted = 0xdead;
        const bool accepted = ninfer::ops::detail::exact_bf16_to_fp16(
            static_cast<std::uint16_t>(bits), converted);
        if (accepted != (expected != finite.end()) ||
            (accepted && converted != expected->second)) {
            std::cerr << "BF16 exact FP16 eligibility mismatch at " << bits << '\n';
            return 1;
        }
        eligible += accepted;
    }
    // Independent FP64 oracle for all 65,536 BF16 patterns. Positive FP16
    // representations are increasing in their integer bit encodings, so we
    // can binary-search the independently constructed finite FP16 table.
    std::vector<std::pair<double, std::uint16_t>> positive;
    for (unsigned bits = 0; bits <= 0x7bffU; ++bits) {
        const unsigned exponent = (bits >> 10) & 31U;
        const unsigned mantissa = bits & 1023U;
        const double value = exponent == 0
            ? std::ldexp(static_cast<double>(mantissa), -24)
            : std::ldexp(static_cast<double>(1024U + mantissa),
                         static_cast<int>(exponent) - 25);
        positive.emplace_back(value, static_cast<std::uint16_t>(bits));
    }
    unsigned rounded = 0, saturated = 0, nonfinite = 0, zeroed = 0;
    for (unsigned bits = 0; bits < 65536; ++bits) {
        const auto source = static_cast<std::uint16_t>(bits);
        const unsigned exp = (bits >> 7) & 255U;
        const unsigned mantissa = bits & 127U;
        const bool negative = (bits & 0x8000U) != 0;
        const auto converted =
            ninfer::ops::detail::finite_bf16_to_fp16_rne_saturate(source);
        using Kind = ninfer::ops::detail::Bf16Fp16ConversionClass;
        if (exp == 255U) {
            if (converted.category != Kind::NonfiniteRejected) {
                std::cerr << "BF16 nonfinite should be rejected at " << bits << '\n';
                return 1;
            }
            ++nonfinite;
            continue;
        }
        const double value = exp == 0
            ? std::ldexp(static_cast<double>(mantissa), -133)
            : std::ldexp(static_cast<double>(128U + mantissa),
                         static_cast<int>(exp) - 134);
        std::uint16_t expected = 0;
        Kind kind = Kind::ExactFinite;
        if (value > 65504.0) {
            expected = 0x7bffU;
            kind = Kind::SaturatedFinite;
            ++saturated;
        } else {
            const auto it = std::lower_bound(
                positive.begin(), positive.end(), value,
                [](const auto& row, double needle) { return row.first < needle; });
            if (it == positive.begin()) {
                expected = it->second;
            } else if (it == positive.end()) {
                expected = positive.back().second;
            } else {
                const auto before = std::prev(it);
                const double lower_error = value - before->first;
                const double upper_error = it->first - value;
                expected = lower_error < upper_error ? before->second
                    : upper_error < lower_error ? it->second
                    : (before->second & 1U) == 0U ? before->second : it->second;
            }
            if (positive[expected].first != value) {
                kind = Kind::RoundedFinite;
                ++rounded;
            }
        }
        if (negative) expected |= 0x8000U;
        const bool expected_zeroed = (source & 0x7fffU) != 0 &&
                                     (expected & 0x7fffU) == 0;
        if (converted.category != kind || converted.bits != expected ||
            converted.nonzero_to_zero != expected_zeroed) {
            std::cerr << "BF16 RNE/saturating FP16 mismatch at " << bits
                      << " expected " << expected << " got " << converted.bits << '\n';
            return 1;
        }
        zeroed += expected_zeroed;
    }
    if (rounded == 0 || saturated == 0 || nonfinite != 256U || zeroed == 0) {
        std::cerr << "BF16 exhaustive conversion class coverage missing\\n";
        return 1;
    }
    std::cout << "BF16 RNE candidate: " << rounded << " rounded, "
              << saturated << " saturated, " << nonfinite
              << " nonfinite rejected, " << zeroed << " nonzero-to-zero\\n";
    std::cout << "Exhaustive 65536 BF16 patterns: " << eligible << " exact finite FP16 values\n";
}
