// Host-only inventory of exact BF16->FP16 eligibility; no GPU arithmetic is changed.
#include "artifact/reader.h"
#include "ops/common/exact_bf16_fp16.h"

#include <nlohmann/json.hpp>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    try {
        if (argc != 2) { throw std::runtime_error("usage: sv7-weight-eligibility artifact.ninfer"); }
        ninfer::artifact::Reader reader(argv[1]);
        nlohmann::json rows = nlohmann::json::array();
        for (const auto& object : reader.objects()) {
            const auto* tensor = std::get_if<ninfer::artifact::TensorDescriptor>(&object);
            if (!tensor || tensor->format != ninfer::artifact::NumericFormat::BF16 ||
                tensor->layout != ninfer::artifact::StorageLayout::ContiguousLeV1 ||
                tensor->shape.size() != 2) { continue; }
            const auto payload = reader.payload(object).data;
            std::uint64_t nonfinite = 0, inexact = 0, exact = 0;
            for (std::size_t i = 0; i < payload.size(); i += 2) {
                const std::uint16_t bits = std::to_integer<unsigned>(payload[i]) |
                    (std::to_integer<unsigned>(payload[i + 1]) << 8);
                std::uint16_t half;
                if (ninfer::ops::detail::exact_bf16_to_fp16(bits, half)) { ++exact; }
                else if (((bits >> 7) & 255U) == 255U) { ++nonfinite; }
                else { ++inexact; }
            }
            rows.push_back({{"name", tensor->name}, {"shape", tensor->shape},
                {"elements", payload.size() / 2}, {"exact", exact}, {"nonfinite", nonfinite},
                {"range_or_inexact", inexact}, {"eligible", nonfinite == 0 && inexact == 0}});
        }
        if (rows.empty()) { throw std::runtime_error("artifact has no BF16 matrices"); }
        nlohmann::json report{{"milestone", "SV7"}, {"qualified", false},
            {"model_id", reader.identity().model_id}, {"weights_id", reader.identity().weights_id},
            {"scope", "exact_weight_representation_only"}, {"tensors", rows}};
        std::cout << report.dump(2) << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
