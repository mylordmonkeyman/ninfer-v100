#include "targets/qwen3_6/impl/frontend/tool_call_parser.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace ninfer::targets::qwen3_6::frontend_internal {
namespace {

using Json = nlohmann::json;

std::string trim_ascii(std::string_view text) {
    std::size_t begin = 0;
    while (begin < text.size() && std::isspace(static_cast<unsigned char>(text[begin])) != 0) {
        ++begin;
    }
    std::size_t end = text.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1])) != 0) { --end; }
    return std::string(text.substr(begin, end - begin));
}

std::string rtrim_ascii(std::string_view text) {
    std::size_t end = text.size();
    while (end != 0 && std::isspace(static_cast<unsigned char>(text[end - 1])) != 0) { --end; }
    return std::string(text.substr(0, end));
}

void skip_ws(std::string_view text, std::size_t& pos) {
    while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos])) != 0) { ++pos; }
}

bool starts_with_at(std::string_view text, std::size_t pos, std::string_view prefix) {
    return pos <= text.size() && text.substr(pos, prefix.size()) == prefix;
}

bool valid_function_name(std::string_view name, std::size_t max_name_length) {
    if (name.empty() || name.size() > max_name_length) { return false; }
    for (const unsigned char c : name) {
        if (std::isalnum(c) == 0 && c != '_' && c != '-') { return false; }
    }
    return true;
}

bool is_json_schema_type(std::string_view type) {
    return type == "string" || type == "integer" || type == "number" || type == "boolean" ||
           type == "object" || type == "array" || type == "null";
}

bool explicit_parameter_encoding(const Json& property,
                                 ToolArgumentTypeContracts::Encoding& encoding) {
    if (!property.is_object()) { return false; }
    const auto type = property.find("type");
    if (type == property.end()) { return false; }

    if (type->is_string()) {
        const std::string& name = type->get_ref<const std::string&>();
        if (!is_json_schema_type(name)) { return false; }
        encoding = name == "string" ? ToolArgumentTypeContracts::Encoding::String
                                    : ToolArgumentTypeContracts::Encoding::Json;
        return true;
    }
    if (!type->is_array() || type->empty()) { return false; }
    bool admits_string = false;
    for (const Json& member : *type) {
        if (!member.is_string()) { return false; }
        const std::string& name = member.get_ref<const std::string&>();
        if (!is_json_schema_type(name)) { return false; }
        admits_string |= name == "string";
    }
    encoding = admits_string ? ToolArgumentTypeContracts::Encoding::String
                             : ToolArgumentTypeContracts::Encoding::Json;
    return true;
}

ToolArgumentTypeContracts::Tool compile_tool_contract(const Json& definition) {
    ToolArgumentTypeContracts::Tool contract;
    if (!definition.is_object()) { return contract; }
    const auto function = definition.find("function");
    if (function == definition.end() || !function->is_object()) { return contract; }
    const auto name = function->find("name");
    if (name == function->end() || !name->is_string()) { return contract; }
    contract.name = name->get<std::string>();

    const auto schema = function->find("parameters");
    if (schema == function->end() || !schema->is_object()) { return contract; }
    const auto properties = schema->find("properties");
    if (properties == schema->end() || !properties->is_object()) { return contract; }

    for (const auto& [parameter_name, property] : properties->items()) {
        ToolArgumentTypeContracts::Encoding encoding;
        if (explicit_parameter_encoding(property, encoding)) {
            contract.parameters.push_back({parameter_name, encoding});
        }
    }
    return contract;
}

bool same_contract(const ToolArgumentTypeContracts::Tool& lhs,
                   const ToolArgumentTypeContracts::Tool& rhs) {
    if (lhs.parameters.size() != rhs.parameters.size()) { return false; }
    for (std::size_t i = 0; i < lhs.parameters.size(); ++i) {
        if (lhs.parameters[i].name != rhs.parameters[i].name ||
            lhs.parameters[i].encoding != rhs.parameters[i].encoding) {
            return false;
        }
    }
    return true;
}

void append_tool_contract(ToolArgumentTypeContracts& contracts, const Json& definition) {
    ToolArgumentTypeContracts::Tool compiled = compile_tool_contract(definition);
    if (compiled.name.empty()) { return; }
    const auto existing =
        std::find_if(contracts.tools.begin(), contracts.tools.end(),
                     [&](const auto& tool) { return tool.name == compiled.name; });
    if (existing == contracts.tools.end()) {
        contracts.tools.push_back(std::move(compiled));
        return;
    }
    if (existing->unambiguous && !same_contract(*existing, compiled)) {
        existing->parameters.clear();
        existing->unambiguous = false;
    }
}

const ToolArgumentTypeContracts::Parameter*
find_parameter_contract(const ToolArgumentTypeContracts& contracts, std::string_view tool_name,
                        std::string_view parameter_name) {
    const auto tool =
        std::find_if(contracts.tools.begin(), contracts.tools.end(),
                     [&](const auto& candidate) { return candidate.name == tool_name; });
    if (tool == contracts.tools.end() || !tool->unambiguous) { return nullptr; }
    const auto parameter =
        std::find_if(tool->parameters.begin(), tool->parameters.end(),
                     [&](const auto& candidate) { return candidate.name == parameter_name; });
    return parameter == tool->parameters.end() ? nullptr : &*parameter;
}

bool declares_tool(const ToolArgumentTypeContracts& contracts, std::string_view tool_name) {
    if (!contracts.enforce_declared_names) { return true; }
    return std::any_of(contracts.tools.begin(), contracts.tools.end(),
                       [&](const auto& tool) { return tool.name == tool_name; });
}

std::string_view remove_parameter_framing_newlines(std::string_view text) {
    std::size_t begin = 0;
    std::size_t end   = text.size();
    if (text.starts_with("\r\n")) {
        begin = 2;
    } else if (text.starts_with('\n')) {
        begin = 1;
    }
    if (end >= begin + 2 && text.substr(end - 2, 2) == "\r\n") {
        end -= 2;
    } else if (end > begin && text[end - 1] == '\n') {
        --end;
    }
    return text.substr(begin, end - begin);
}

// Degrading is better than discarding, but it is still the model failing to write
// what it declared, so it must be visible. Named like the fallback diagnostic
// below so both are greppable from one place.
void note_degraded_parameter(std::string_view tool, std::string_view parameter,
                             std::size_t bytes) {
    std::fprintf(stderr,
                 "tool_call_parameter_degraded tool=%.*s parameter=%.*s bytes=%zu "
                 "(value did not parse as its declared type; passed through as text)\n",
                 static_cast<int>(tool.size()), tool.data(),
                 static_cast<int>(parameter.size()), parameter.data(), bytes);
}

bool parse_parameter(std::string_view inner, std::size_t& pos, Json& args,
                     std::string_view tool_name, const ToolArgumentTypeContracts& contracts) {
    constexpr std::string_view kParamOpen  = "<parameter=";
    constexpr std::string_view kParamClose = "</parameter>";
    if (!starts_with_at(inner, pos, kParamOpen)) { return false; }
    const std::size_t name_begin = pos + kParamOpen.size();
    const std::size_t name_end   = inner.find('>', name_begin);
    if (name_end == std::string_view::npos || name_end == name_begin) { return false; }
    const std::string key       = std::string(inner.substr(name_begin, name_end - name_begin));
    pos                         = name_end + 1;
    const std::size_t value_end = inner.find(kParamClose, pos);
    if (value_end == std::string_view::npos) { return false; }
    const std::string_view encoded_value = inner.substr(pos, value_end - pos);
    const ToolArgumentTypeContracts::Parameter* contract =
        find_parameter_contract(contracts, tool_name, key);
    if (contract == nullptr) {
        const std::string legacy_value = trim_ascii(encoded_value);
        Json parsed                    = Json::parse(legacy_value, nullptr, false);
        args[key] = parsed.is_discarded() ? Json(legacy_value) : std::move(parsed);
    } else {
        const std::string value(remove_parameter_framing_newlines(encoded_value));
        if (contract->encoding == ToolArgumentTypeContracts::Encoding::String) {
            args[key] = value;
        } else {
            Json parsed = Json::parse(value, nullptr, false);
            // A typed parameter whose value will not parse used to discard the
            // ENTIRE call, and the raw <tool_call> markup was handed back as prose
            // -- which a client renders as chat text, so the user sees XML and the
            // tool never runs. Declaring a schema therefore made the parser more
            // brittle than declaring nothing, since the contract-free branch above
            // already degrades to the raw string.
            //
            // Pass the value through instead and let the client decide. It can
            // reject an argument of the wrong type, retry, or repair it; it can do
            // nothing at all with markup in a chat bubble. The call is
            // structurally complete here -- unterminated and malformed BLOCKS
            // still fall back, because those may be arguments the model never
            // finished writing.
            if (parsed.is_discarded()) {
                note_degraded_parameter(tool_name, key, value.size());
                args[key] = value;
            } else {
                args[key] = std::move(parsed);
            }
        }
    }
    pos = value_end + kParamClose.size();
    return true;
}

bool parse_one_tool_call(std::string_view block, std::size_t max_name_length,
                         const ToolArgumentTypeContracts& contracts, GeneratedToolCall& out) {
    constexpr std::string_view kFunctionOpen  = "<function=";
    constexpr std::string_view kFunctionClose = "</function>";
    std::size_t pos                           = 0;
    skip_ws(block, pos);
    if (!starts_with_at(block, pos, kFunctionOpen)) { return false; }
    const std::size_t name_begin = pos + kFunctionOpen.size();
    const std::size_t name_end   = block.find('>', name_begin);
    if (name_end == std::string_view::npos || name_end == name_begin) { return false; }
    const std::string name = std::string(block.substr(name_begin, name_end - name_begin));
    if (!valid_function_name(name, max_name_length) || !declares_tool(contracts, name)) {
        return false;
    }
    pos = name_end + 1;

    const std::size_t function_end = block.find(kFunctionClose, pos);
    if (function_end == std::string_view::npos) { return false; }
    const std::string_view params = block.substr(pos, function_end - pos);
    Json args                     = Json::object();
    std::size_t param_pos         = 0;
    for (;;) {
        skip_ws(params, param_pos);
        if (param_pos >= params.size()) { break; }
        if (!parse_parameter(params, param_pos, args, name, contracts)) { return false; }
    }

    pos = function_end + kFunctionClose.size();
    skip_ws(block, pos);
    if (pos != block.size()) { return false; }

    out.name           = name;
    out.arguments_json = args.dump();
    return true;
}

ParsedToolCallOutput fallback(const std::string& text) {
    ParsedToolCallOutput out;
    out.content = text;
    return out;
}

// Every fallback path throws away calls the model did emit and hands the raw
// <tool_call> markup back as prose, which an agent client reports as an empty
// response. Naming the path turns that into a diagnosable event instead of a
// silent one.
ParsedToolCallOutput fallback(const std::string& text, const char* reason) {
    std::fprintf(stderr, "tool_call_parse_fallback reason=%s bytes=%zu\n", reason, text.size());
    return fallback(text);
}

// A model that narrates after its call ("...now let me read the file") loses
// every parsed call under the strict rule. Other OpenAI-compatible servers keep
// the calls and treat the surrounding text as content. Dropping was a deliberate
// decision here (tests/test_tool_call_parser.cpp asserts it), so the tolerant
// behaviour is opt-in rather than a silent reversal.
// Read every call rather than latching in a function-local static: this runs at
// most once per generated response, and a latched read cannot be toggled by a
// test (or by anything else) after the first parse in the process.
bool allow_trailing_text_after_tool_calls() {
    const char* env = std::getenv("NINFER_TOOL_CALLS_ALLOW_TRAILING_TEXT");
    return env != nullptr && env[0] == '1' && env[1] == '\0';
}

} // namespace

std::string required_tool_call_grammar(const std::vector<std::string>& names) {
    if (names.empty()) { throw std::invalid_argument("required tool call needs declared names"); }
    std::string grammar =
        "root ::= ws \"<tool_call>\" ws \"<function=\" name \">\" ws "
        "parameter* \"</function>\" ws \"</tool_call>\" ws\n"
        "ws ::= [ \\t\\r\\n]*\n"
        "parameter ::= \"<parameter=\" [^ \\t\\r\\n<>]+ \">\" data-0 \"</parameter>\" ws\n"
        "name ::= ";
    for (std::size_t i = 0; i < names.size(); ++i) {
        if (!valid_function_name(names[i], 128)) {
            throw std::invalid_argument("invalid required tool name: " + names[i]);
        }
        if (i != 0) { grammar += " | "; }
        grammar += Json(names[i]).dump();
    }
    grammar += '\n';

    // Raw Qwen arguments may contain code, HTML, or JSON. Admit all text except the three
    // complete delimiters consumed by the parser. A prefix automaton preserves '<', closing
    // HTML tags, and partial delimiter prefixes without letting them terminate the wrong frame.
    const std::vector<std::string> forbidden{"</parameter>", "</function>", "</tool_call>"};
    std::vector<std::string> prefixes{std::string{}};
    std::string alphabet;
    for (const auto& word : forbidden) {
        for (std::size_t n = 1; n < word.size(); ++n) {
            const auto prefix = word.substr(0, n);
            if (std::find(prefixes.begin(), prefixes.end(), prefix) == prefixes.end()) {
                prefixes.push_back(prefix);
            }
        }
        for (char c : word) {
            if (alphabet.find(c) == std::string::npos) { alphabet += c; }
        }
    }
    for (std::size_t i = 0; i < prefixes.size(); ++i) {
        grammar += "data-" + std::to_string(i) + " ::= \"\" | [^" + alphabet + "] data-0";
        for (char c : alphabet) {
            const auto next = prefixes[i] + c;
            if (std::any_of(forbidden.begin(), forbidden.end(), [&](const auto& word) {
                    return next.ends_with(word);
                })) { continue; }
            std::size_t target = 0;
            for (std::size_t j = 1; j < prefixes.size(); ++j) {
                if (prefixes[j].size() > prefixes[target].size() && next.ends_with(prefixes[j])) {
                    target = j;
                }
            }
            grammar += " | " + Json(std::string(1, c)).dump() + " data-" + std::to_string(target);
        }
        grammar += '\n';
    }
    return grammar;
}

std::shared_ptr<const ToolCallOutputContract>
build_tool_call_output_contract(std::span<const std::string> tool_jsons, bool enabled) {
    if (!enabled) { return {}; }
    auto contract                                   = std::make_shared<ToolCallOutputContract>();
    contract->argument_types.enforce_declared_names = true;
    contract->argument_types.tools.reserve(tool_jsons.size());
    for (const std::string& tool_json : tool_jsons) {
        const Json definition = Json::parse(tool_json, nullptr, false);
        if (!definition.is_discarded()) {
            append_tool_contract(contract->argument_types, definition);
        }
    }
    return contract;
}

ParsedToolCallOutput parse_qwen_tool_call_output(const std::string& text,
                                                 std::size_t max_tool_name_length,
                                                 const ToolArgumentTypeContracts& contracts) {
    constexpr std::string_view kToolOpen  = "<tool_call>";
    constexpr std::string_view kToolClose = "</tool_call>";

    const std::size_t first = text.find(kToolOpen);
    if (first == std::string::npos) { return fallback(text); }

    ParsedToolCallOutput out;
    out.content = rtrim_ascii(std::string_view(text).substr(0, first));

    std::size_t pos = first;
    while (pos < text.size()) {
        skip_ws(text, pos);
        if (pos >= text.size()) { break; }
        if (!starts_with_at(text, pos, kToolOpen)) {
            if (!out.tool_calls.empty() && allow_trailing_text_after_tool_calls()) {
                const std::string_view suffix = std::string_view(text).substr(pos);
                if (!out.content.empty()) { out.content.append("\n"); }
                out.content.append(suffix);
                out.is_tool_call_response = true;
                return out;
            }
            return fallback(text, "trailing_text_after_tool_call");
        }
        const std::size_t inner_begin = pos + kToolOpen.size();
        const std::size_t close       = text.find(kToolClose, inner_begin);
        if (close == std::string::npos) { return fallback(text, "unterminated_tool_call"); }
        GeneratedToolCall call;
        if (!parse_one_tool_call(std::string_view(text).substr(inner_begin, close - inner_begin),
                                 max_tool_name_length, contracts, call)) {
            return fallback(text, "malformed_tool_call_body");
        }
        out.tool_calls.push_back(std::move(call));
        pos = close + kToolClose.size();
    }

    if (out.tool_calls.empty()) { return fallback(text, "no_calls_extracted"); }
    out.is_tool_call_response = true;
    return out;
}

ToolCallOutputDecoder::ToolCallOutputDecoder(std::shared_ptr<const ToolCallOutputContract> contract,
                                             std::size_t max_tool_name_length)
    : contract_(std::move(contract)), max_tool_name_length_(max_tool_name_length) {}

std::string ToolCallOutputDecoder::feed(std::string_view text) {
    if (finished_) { throw std::logic_error("tool-call output decoder is already finished"); }
    if (text.empty()) { return {}; }
    if (!contract_) { return std::string(text); }
    if (saw_tool_marker_) {
        tool_region_.append(text);
        return {};
    }

    constexpr std::string_view kToolOpen = "<tool_call>";
    std::string visible;
    for (std::size_t index = 0; index < text.size(); ++index) {
        const char byte = text[index];
        if (marker_prefix_bytes_ != 0) {
            if (byte == kToolOpen[marker_prefix_bytes_]) {
                ++marker_prefix_bytes_;
                if (marker_prefix_bytes_ == kToolOpen.size()) {
                    tool_region_ = std::move(trailing_whitespace_);
                    trailing_whitespace_.clear();
                    tool_region_.append(kToolOpen);
                    tool_region_.append(text.substr(index + 1));
                    marker_prefix_bytes_ = 0;
                    saw_tool_marker_     = true;
                    break;
                }
                continue;
            }
            visible.append(trailing_whitespace_);
            trailing_whitespace_.clear();
            visible.append(kToolOpen.substr(0, marker_prefix_bytes_));
            marker_prefix_bytes_ = 0;
        }

        if (byte == kToolOpen.front()) {
            marker_prefix_bytes_ = 1;
        } else if (std::isspace(static_cast<unsigned char>(byte)) != 0) {
            trailing_whitespace_.push_back(byte);
        } else {
            visible.append(trailing_whitespace_);
            trailing_whitespace_.clear();
            visible.push_back(byte);
        }
    }
    return visible;
}

ToolCallOutputDecoder::Terminal ToolCallOutputDecoder::finish() {
    if (finished_) { throw std::logic_error("tool-call output decoder is already finished"); }
    finished_ = true;
    if (!contract_) { return {}; }

    ParsedToolCallOutput parsed =
        parse_qwen_tool_call_output(tool_region_, max_tool_name_length_, contract_->argument_types);
    if (saw_tool_marker_ && parsed.is_tool_call_response) {
        trailing_whitespace_.clear();
        tool_region_.clear();
        marker_prefix_bytes_ = 0;
        return Terminal{.content = {}, .tool_calls = std::move(parsed.tool_calls)};
    }

    constexpr std::string_view kToolOpen = "<tool_call>";
    std::string tail                     = std::move(trailing_whitespace_);
    tail.append(kToolOpen.substr(0, marker_prefix_bytes_));
    marker_prefix_bytes_ = 0;
    tail += tool_region_;
    tool_region_.clear();
    return Terminal{.content = std::move(tail), .tool_calls = {}};
}

} // namespace ninfer::targets::qwen3_6::frontend_internal
