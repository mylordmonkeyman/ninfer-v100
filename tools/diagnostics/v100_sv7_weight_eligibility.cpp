// Host-only SV7 inventory: deterministic BF16 -> FP16 conversion classes.
// Does not mutate resident weights, enable GEMM, or constitute accuracy qualification.
#include "artifact/reader.h"
#include "ops/common/rounded_bf16_fp16.h"

#include <nlohmann/json.hpp>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

double bf16_value(std::uint16_t bits) {
    return static_cast<double>(std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16));
}

double half_value(std::uint16_t bits) {
    const unsigned fraction = bits & 0x3ffU;
    const unsigned exponent = (bits >> 10) & 31U;
    const double value = exponent == 0U
        ? std::ldexp(static_cast<double>(fraction), -24)
        : std::ldexp(static_cast<double>(1024U + fraction),
                     static_cast<int>(exponent) - 25);
    return (bits & 0x8000U) ? -value : value;
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) { throw std::runtime_error("usage: sv7-weight-eligibility artifact.ninfer"); }
        ninfer::artifact::Reader reader(argv[1]);
        nlohmann::json rows = nlohmann::json::array();
        std::uint64_t total_exact = 0, total_rounded = 0, total_saturated = 0,
                      total_nonfinite = 0, total_zeroed = 0;
        for (const auto& object : reader.objects()) {
            const auto* tensor = std::get_if<ninfer::artifact::TensorDescriptor>(&object);
            if (!tensor || tensor->format != ninfer::artifact::NumericFormat::BF16 ||
                tensor->layout != ninfer::artifact::StorageLayout::ContiguousLeV1 ||
                tensor->shape.size() != 2) { continue; }
            const auto payload = reader.payload(object).data;
            if ((payload.size() % 2U) != 0U) {
                throw std::runtime_error("odd-byte BF16 matrix payload");
            }
            std::uint64_t exact = 0, rounded = 0, saturated = 0, nonfinite = 0, zeroed = 0;
            double max_abs_error = 0.0, max_relative_error = 0.0;
            for (std::size_t i = 0; i < payload.size(); i += 2) {
                const std::uint16_t bits = std::to_integer<unsigned>(payload[i]) |
                    (std::to_integer<unsigned>(payload[i + 1]) << 8);
                const auto converted = ninfer::ops::detail::finite_bf16_to_fp16_rne_saturate(bits);
                using Class = ninfer::ops::detail::Bf16Fp16ConversionClass;
                switch (converted.category) {
                case Class::ExactFinite: ++exact; break;
                case Class::RoundedFinite: ++rounded; break;
                case Class::SaturatedFinite: ++saturated; break;
                case Class::NonfiniteRejected: ++nonfinite; break;
                }
                if (converted.nonzero_to_zero) ++zeroed;
                if (converted.category != Class::NonfiniteRejected) {
                    const double before = bf16_value(bits);
                    const double after = half_value(converted.bits);
                    const double absolute = std::abs(before - after);
                    max_abs_error = std::max(max_abs_error, absolute);
                    if (before != 0.0) {
                        max_relative_error = std::max(max_relative_error, absolute / std::abs(before));
                    }
                }
            }
            total_exact += exact;
            total_rounded += rounded;
            total_saturated += saturated;
            total_nonfinite += nonfinite;
            total_zeroed += zeroed;
            rows.push_back({{"name", tensor->name}, {"shape", tensor->shape},
                {"elements", payload.size() / 2},
                {"exact", exact}, {"nonfinite", nonfinite},
                // Legacy exact-only fields retained for consumer compatibility.
                {"range_or_inexact", rounded + saturated},
                {"eligible", rounded == 0 && saturated == 0 && nonfinite == 0},
                // Revised SV7 candidates: rounded/clamped are opt-in, NOT approved.
                {"rounded_finite", rounded}, {"saturated_finite", saturated},
                {"nonzero_to_zero", zeroed},
                {"max_abs_conversion_error", max_abs_error},
                {"max_relative_conversion_error", max_relative_error},
                {"finite_candidate", nonfinite == 0}});
        }
        if (rows.empty()) { throw std::runtime_error("artifact has no BF16 matrices"); }
        nlohmann::json report{{"milestone", "SV7"}, {"qualified", false},
            {"model_id", reader.identity().model_id}, {"weights_id", reader.identity().weights_id},
            {"scope", "sv7_rounded_saturated_conversion_inventory_only"},
            {"conversion_policy", "finite RNE ties-even; finite overflow saturates to 65504; nonfinite rejected"},
            {"totals", {{"exact", total_exact}, {"rounded_finite", total_rounded},
                        {"saturated_finite", total_saturated},
                        {"nonfinite_rejected", total_nonfinite},
                        {"nonzero_to_zero", total_zeroed}}},
            {"tensors", rows}};
        std::cout << report.dump(2) << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
