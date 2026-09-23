#include "targets/qwen3_6/impl/frontend/tokenizer.h"
#include <ninfer/targets/qwen3_6/frontend_resources.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>

namespace fi = ninfer::targets::qwen3_6::frontend_internal;

std::string read(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot read " + path.string());
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

int main() {
    const char* source = std::getenv("NINFER_ORCAROUTER_MODEL_DIR");
    if (!source || !*source) return 77;
    try {
        const std::filesystem::path dir(source);
        const auto tokenizer_json = read(dir / "tokenizer.json");
        const auto config = read(dir / "tokenizer_config.json");
        const auto generation = read(dir / "generation_config.json");
        const ninfer::targets::qwen3_6::FrontendResources resources{
            .tokenizer_json = tokenizer_json,
            .tokenizer_config_json = config,
            .chat_template_jinja = read(dir / "chat_template.jinja"),
            .generation_config_json = generation,
            .preprocessor_config_json = read(dir / "preprocessor_config.json"),
            .video_preprocessor_config_json = read(dir / "video_preprocessor_config.json"),
        };
        auto frontend = ninfer::targets::qwen3_6::make_frontend(resources, {});
        const fi::Tokenizer tokenizer({tokenizer_json, config, generation});
        const auto fixture = nlohmann::json::parse(read(
            std::filesystem::path(NINFER_SOURCE_DIR) / "tests/fixtures/qwen3_8_27b_orcarouter_tokenizer.json"));
        for (const auto& item : fixture.at("cases")) {
            const auto text = item.at("text").get<std::string>();
            const auto ids = item.at("input_ids").get<std::vector<int>>();
            const fi::EncodeOptions options{.parse_added_tokens = item.at("parse_added_tokens").get<bool>()};
            const std::array<std::size_t, 2> boundaries{0, text.size()};
            const auto result = tokenizer.encode_with_boundaries(text, boundaries, options);
            if (result.input_ids != ids || result.boundaries.back().exact_frontier != ids.size() ||
                tokenizer.decode(ids) != item.at("decoded").get<std::string>() ||
                tokenizer.decode(ids, {.skip_special_tokens = true}) != item.at("decoded_skip_special_tokens").get<std::string>()) {
                std::cerr << "reference mismatch: " << item.at("name") << '\n';
                return 1;
            }
            const auto limited = tokenizer.encode(text, {.parse_added_tokens = options.parse_added_tokens, .max_tokens = 32});
            if (limited != std::vector<int>(ids.begin(), ids.begin() + std::min<std::size_t>(32, ids.size()))) {
                std::cerr << "bounded reference mismatch: " << item.at("name") << '\n';
                return 1;
            }
        }
        std::cout << fixture.at("cases").size() << " independent tokenizer cases passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
