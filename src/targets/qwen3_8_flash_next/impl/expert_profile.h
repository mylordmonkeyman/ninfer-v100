#pragma once

#include <nlohmann/json.hpp>
#include <array>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::targets::qwen3_8_flash_next::detail {
// Bound to the actual artifact Reader identity, not an environment-supplied model name.
struct FlashNextExpertProfile {
    std::string model_id, weights_id;
    std::array<std::array<int, 512>, 48> ranking{};

    static FlashNextExpertProfile parse(const nlohmann::json& j,
            std::string_view expected_model, std::string_view expected_weights) {
        if (j.at("magic") != "NINFER_V100_EXPERT_PROFILE" || j.at("version") != 2 ||
            j.at("layers") != 48 || j.at("experts_per_layer") != 512)
            throw std::invalid_argument("invalid expert profile schema");
        FlashNextExpertProfile profile;
        profile.model_id = j.at("model_id").get<std::string>();
        profile.weights_id = j.at("weights_id").get<std::string>();
        if (expected_model.empty() || expected_weights.empty() ||
            profile.model_id != expected_model || profile.weights_id != expected_weights)
            throw std::invalid_argument("expert profile artifact identity mismatch");
        const auto& rows = j.at("ranking");
        if (!rows.is_array() || rows.size() != 48)
            throw std::invalid_argument("expert profile requires 48 layer rankings");
        for (unsigned layer = 0; layer < 48; ++layer) {
            if (!rows[layer].is_array() || rows[layer].size() != 512)
                throw std::invalid_argument("expert profile requires 512 experts per layer");
            std::array<bool, 512> seen{};
            for (unsigned rank = 0; rank < 512; ++rank) {
                const auto& value = rows[layer][rank];
                if (!value.is_number_integer() || value < 0 || value >= 512)
                    throw std::invalid_argument("invalid expert profile expert ID");
                const int id = value.get<int>();
                if (seen[id]) throw std::invalid_argument("duplicate expert profile expert ID");
                seen[id] = true;
                profile.ranking[layer][rank] = id;
            }
        }
        return profile;
    }
    static FlashNextExpertProfile load(const char* path, std::string_view model,
            std::string_view weights) {
        std::ifstream input(path);
        if (!input) throw std::runtime_error("cannot open expert profile");
        nlohmann::json j;
        input >> j;
        return parse(j, model, weights);
    }
};
} // namespace ninfer::targets::qwen3_8_flash_next::detail
