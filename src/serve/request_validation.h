#pragma once

#include "serve/request.h"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>

namespace ninfer::serve {

[[noreturn]] void bad_request(std::string message, std::string param = {}, std::string code = {});

std::optional<int> optional_int(const nlohmann::json& object, const char* key);
std::optional<int> optional_int(const nlohmann::ordered_json& object, const char* key);
std::optional<double> optional_number(const nlohmann::json& object, const char* key);
std::optional<double> optional_number(const nlohmann::ordered_json& object, const char* key);
bool optional_bool(const nlohmann::json& object, const char* key, bool fallback);
bool optional_bool(const nlohmann::ordered_json& object, const char* key, bool fallback);
[[nodiscard]] StructuredOutputOptions parse_structured_output_format(
    const nlohmann::json& format, std::string_view param, bool nested_schema);
[[nodiscard]] StructuredOutputOptions parse_structured_output_format(
    const nlohmann::ordered_json& format, std::string_view param, bool nested_schema);

[[nodiscard]] bool valid_tool_name(std::string_view name, std::size_t maximum_length) noexcept;

} // namespace ninfer::serve
