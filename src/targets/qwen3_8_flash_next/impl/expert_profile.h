#pragma once

#include <nlohmann/json.hpp>
#include <array>
#include <algorithm>
#include <cmath>
#include <filesystem>
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
    using Heat = std::array<std::array<double,512>,48>;
    using Residents = std::array<std::array<bool,512>,48>;
    // Residents first; measured heat, original ranking, then expert index resolve ties.
    [[nodiscard]] FlashNextExpertProfile learned(const Heat& heat, const Residents& resident) const {
        auto result = *this;
        for (unsigned layer=0;layer<48;++layer) {
            std::array<unsigned,512> prior{};
            for (unsigned rank=0;rank<512;++rank) prior[ranking[layer][rank]]=rank;
            for (unsigned id=0;id<512;++id) {
                if (!std::isfinite(heat[layer][id]) || heat[layer][id]<0)
                    throw std::invalid_argument("invalid learned expert heat");
                result.ranking[layer][id]=id;
            }
            auto& row=result.ranking[layer];
            std::sort(row.begin(),row.end(),[&](int a,int b) {
                if (resident[layer][a]!=resident[layer][b]) return resident[layer][a];
                if (heat[layer][a]!=heat[layer][b]) return heat[layer][a]>heat[layer][b];
                if (prior[a]!=prior[b]) return prior[a]<prior[b];
                return a<b;
            });
        }
        return result;
    }
    void save(const std::filesystem::path& path, const Heat& heat) const {
        nlohmann::json j={{"magic","NINFER_V100_EXPERT_PROFILE"},{"version",2},
            {"model_id",model_id},{"weights_id",weights_id},{"layers",48},
            {"experts_per_layer",512},{"ranking",ranking},{"frequency",heat},
            {"source","runtime_learned"}};
        auto temporary=path; temporary += ".tmp";
        {
            std::ofstream output(temporary);
            if (!output) throw std::runtime_error("cannot create learned expert profile");
            output << j.dump(2) << '\n';
            output.close();
            if (!output) throw std::runtime_error("cannot write learned expert profile");
        }
        std::filesystem::rename(temporary,path);
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
