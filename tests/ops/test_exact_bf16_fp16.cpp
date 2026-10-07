#include "ops/common/exact_bf16_fp16.h"

#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <unordered_map>

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
    std::cout << "Exhaustive 65536 BF16 patterns: " << eligible << " exact finite FP16 values\n";
}
