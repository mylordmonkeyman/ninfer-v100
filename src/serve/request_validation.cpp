#include "serve/request_validation.h"
#include "runtime/contract/structured_output.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>

namespace ninfer::serve {

namespace {

template <typename JsonType>
StructuredOutputOptions parse_structured_output_format_impl(const JsonType& format,
                                                            std::string_view param,
                                                            bool nested_schema) {
    const auto invalid = [&](std::string message) -> void {
        bad_request(std::move(message), std::string(param), "invalid_response_format");
    };
    if (!format.is_object() || !format.contains("type") || !format["type"].is_string()) {
        invalid("output format must contain a string type");
    }
    const auto type = format["type"].template get<std::string>();
    StructuredOutputOptions out;
    if (type == "text" || type == "json_object") {
        if (format.size() != 1) { invalid("text/json_object format accepts only type"); }
        out.kind = type == "text" ? StructuredOutputKind::Text : StructuredOutputKind::JsonObject;
        return out;
    }
    if (type != "json_schema") { invalid("output format type must be text, json_object, or json_schema"); }
    if (nested_schema && (!format.contains("json_schema") || !format["json_schema"].is_object() || format.size() != 2)) {
        invalid("json_schema format requires a json_schema object");
    }
    const auto& descriptor = nested_schema ? format["json_schema"] : format;
    if (!descriptor.contains("schema") || !descriptor["schema"].is_object()) {
        invalid("json_schema format requires an object schema");
    }
    for (auto it = descriptor.begin(); it != descriptor.end(); ++it) {
        if (it.key() != "schema" && it.key() != "name" && it.key() != "strict" && it.key() != "description" &&
            !(it.key() == "type" && !nested_schema)) { invalid("unknown output schema descriptor field: " + it.key()); }
    }
    if (nested_schema && !descriptor.contains("name")) { invalid("json_schema.name is required"); }
    if (descriptor.contains("name")) {
        if (!descriptor["name"].is_string()) { invalid("json_schema.name must be a string"); }
        const auto name = descriptor["name"].template get<std::string>();
        if (name.empty() || name.size() > 64 || !std::all_of(name.begin(), name.end(), [](unsigned char c) {
            return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '-';
        })) { invalid("json_schema.name must contain 1..64 letters, digits, underscores or hyphens"); }
    }
    if (descriptor.contains("strict") && !descriptor["strict"].is_boolean()) { invalid("json_schema.strict must be boolean"); }
    if (descriptor.contains("description") && !descriptor["description"].is_string()) { invalid("json_schema.description must be a string"); }
    out.kind = StructuredOutputKind::JsonSchema;
    out.schema = descriptor["schema"].dump();
    try { runtime::validate_structured_output(out); }
    catch (const std::invalid_argument& error) { bad_request(error.what(), std::string(param), "unsupported_json_schema"); }
    return out;
}

template <typename JsonType>
std::optional<int> optional_int_impl(const JsonType& object, const char* key) {
    if (!object.contains(key) || object.at(key).is_null()) { return std::nullopt; }
    const auto& value = object.at(key);
    if (!value.is_number_integer()) { bad_request(std::string(key) + " must be an integer", key); }
    if (value.is_number_unsigned()) {
        const std::uint64_t converted = value.template get<std::uint64_t>();
        if (converted > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
            bad_request(std::string(key) + " is out of range", key);
        }
        return static_cast<int>(converted);
    }
    const std::int64_t converted = value.template get<std::int64_t>();
    if (converted < std::numeric_limits<int>::min() ||
        converted > std::numeric_limits<int>::max()) {
        bad_request(std::string(key) + " is out of range", key);
    }
    return static_cast<int>(converted);
}

template <typename JsonType>
std::optional<double> optional_number_impl(const JsonType& object, const char* key) {
    if (!object.contains(key) || object.at(key).is_null()) { return std::nullopt; }
    if (!object.at(key).is_number()) { bad_request(std::string(key) + " must be a number", key); }
    const double value = object.at(key).template get<double>();
    if (!std::isfinite(value)) { bad_request(std::string(key) + " must be finite", key); }
    return value;
}

template <typename JsonType>
bool optional_bool_impl(const JsonType& object, const char* key, bool fallback) {
    if (!object.contains(key) || object.at(key).is_null()) { return fallback; }
    if (!object.at(key).is_boolean()) { bad_request(std::string(key) + " must be a boolean", key); }
    return object.at(key).template get<bool>();
}

} // namespace

StructuredOutputOptions parse_structured_output_format(const nlohmann::json& format,
                                                       std::string_view param, bool nested_schema) {
    return parse_structured_output_format_impl(format, param, nested_schema);
}

StructuredOutputOptions parse_structured_output_format(const nlohmann::ordered_json& format,
                                                       std::string_view param, bool nested_schema) {
    return parse_structured_output_format_impl(format, param, nested_schema);
}

[[noreturn]] void bad_request(std::string message, std::string param, std::string code) {
    ApiError error;
    error.status  = 400;
    error.type    = "invalid_request_error";
    error.message = std::move(message);
    error.param   = std::move(param);
    error.code    = std::move(code);
    throw ApiException(std::move(error));
}

std::optional<int> optional_int(const nlohmann::json& object, const char* key) {
    return optional_int_impl(object, key);
}

std::optional<int> optional_int(const nlohmann::ordered_json& object, const char* key) {
    return optional_int_impl(object, key);
}

std::optional<double> optional_number(const nlohmann::json& object, const char* key) {
    return optional_number_impl(object, key);
}

std::optional<double> optional_number(const nlohmann::ordered_json& object, const char* key) {
    return optional_number_impl(object, key);
}

bool optional_bool(const nlohmann::json& object, const char* key, bool fallback) {
    return optional_bool_impl(object, key, fallback);
}

bool optional_bool(const nlohmann::ordered_json& object, const char* key, bool fallback) {
    return optional_bool_impl(object, key, fallback);
}

bool valid_tool_name(std::string_view name, std::size_t maximum_length) noexcept {
    if (name.empty() || name.size() > maximum_length) { return false; }
    for (const unsigned char character : name) {
        if (std::isalnum(character) == 0 && character != '_' && character != '-') { return false; }
    }
    return true;
}

} // namespace ninfer::serve
