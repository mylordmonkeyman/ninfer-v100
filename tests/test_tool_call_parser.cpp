#include "targets/qwen3_6/impl/frontend/tool_call_parser.h"

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <iostream>
#include <memory>
#include <span>
#include <string>

namespace {

using Json   = nlohmann::json;
namespace fi = ninfer::targets::qwen3_6::frontend_internal;

const fi::ToolArgumentTypeContracts kNoTypeContracts;

int fail(const std::string& message) {
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

int check(bool condition, const std::string& message) { return condition ? 0 : fail(message); }

// The tolerant path is process-latched, so the tests drive it through the same
// environment variable the server uses.
void set_allow_trailing_text(bool on) {
#ifdef _WIN32
    (void)_putenv_s("NINFER_TOOL_CALLS_ALLOW_TRAILING_TEXT", on ? "1" : "0");
#else
    (void)setenv("NINFER_TOOL_CALLS_ALLOW_TRAILING_TEXT", on ? "1" : "0", 1);
#endif
}

fi::ToolArgumentTypeContracts contracts_for(const std::string& tool_name, Json properties) {
    const std::string definition =
        Json{{"type", "function"},
             {"function", Json{{"name", tool_name},
                               {"parameters",
                                Json{{"type", "object"}, {"properties", std::move(properties)}}}}}}
            .dump();
    return fi::build_tool_call_output_contract(std::span<const std::string>(&definition, 1), true)
        ->argument_types;
}

int test_single_call() {
    const fi::ParsedToolCallOutput parsed =
        fi::parse_qwen_tool_call_output("Calling weather.\n"
                                        "<tool_call>\n"
                                        "<function=get_weather>\n"
                                        "<parameter=city>\nParis\n</parameter>\n"
                                        "<parameter=days>\n2\n</parameter>\n"
                                        "</function>\n"
                                        "</tool_call>",
                                        64, kNoTypeContracts);

    int failures = 0;
    failures += check(parsed.is_tool_call_response, "single call parsed as tool response");
    failures += check(parsed.content == "Calling weather.", "content prefix trimmed");
    failures += check(parsed.tool_calls.size() == 1, "one parsed call");
    failures += check(parsed.tool_calls[0].name == "get_weather", "function name parsed");
    const Json args = Json::parse(parsed.tool_calls[0].arguments_json);
    failures += check(args.at("city") == "Paris", "string parameter parsed");
    failures += check(args.at("days") == 2, "number parameter parsed");
    return failures;
}

int test_multiple_calls_and_json_values() {
    const fi::ParsedToolCallOutput parsed = fi::parse_qwen_tool_call_output(
        "<tool_call>\n"
        "<function=first>\n"
        "<parameter=payload>\n{\"ok\":true,\"items\":[1,2]}\n</parameter>\n"
        "</function>\n"
        "</tool_call>\n"
        "<tool_call>\n"
        "<function=second>\n"
        "<parameter=value>\nplain text\n</parameter>\n"
        "</function>\n"
        "</tool_call>",
        64, kNoTypeContracts);

    int failures = 0;
    failures += check(parsed.is_tool_call_response, "multiple calls parsed as tool response");
    failures += check(parsed.tool_calls.size() == 2, "two parsed calls");
    failures += check(parsed.tool_calls[0].name == "first", "first call name");
    failures += check(parsed.tool_calls[1].name == "second", "second call name");
    const Json first = Json::parse(parsed.tool_calls[0].arguments_json);
    failures += check(first.at("payload").at("ok") == true, "object parameter bool");
    failures += check(first.at("payload").at("items").at(1) == 2, "object parameter array");
    const Json second = Json::parse(parsed.tool_calls[1].arguments_json);
    failures += check(second.at("value") == "plain text", "plain text parameter string");
    return failures;
}

// A typed parameter the model wrote badly must not cost the whole call. It used
// to: the raw <tool_call> markup was returned as prose and the client rendered
// XML into the chat instead of running anything. Observed on real agent traffic,
// twice in one day, on calls of 722 and 4517 bytes.
int test_unparseable_typed_parameter_degrades_to_text() {
    const fi::ToolArgumentTypeContracts contracts = contracts_for(
        "edit_file", Json{{"explanation", Json{{"type", "string"}}},
                          {"replacements", Json{{"type", "array"}}}});
    // Structurally complete, but the array value is not valid JSON.
    const fi::ParsedToolCallOutput parsed =
        fi::parse_qwen_tool_call_output("<tool_call>\n"
                                        "<function=edit_file>\n"
                                        "<parameter=explanation>\nWhy\n</parameter>\n"
                                        "<parameter=replacements>\n[{\"a\": `oops`,}]\n</parameter>\n"
                                        "</function>\n"
                                        "</tool_call>",
                                        64, contracts);
    int failures = 0;
    failures += check(parsed.is_tool_call_response, "degraded call is still a tool response");
    failures += check(parsed.tool_calls.size() == 1, "degraded call is still delivered");
    failures += check(parsed.content.empty(), "degraded call does not leak markup as prose");
    if (parsed.tool_calls.size() == 1) {
        const Json args = Json::parse(parsed.tool_calls[0].arguments_json);
        failures += check(args.at("explanation") == "Why", "sound parameter still typed");
        failures += check(args.at("replacements").is_string(),
                          "unparseable parameter passed through as text");
    }
    return failures;
}

// The tolerance above must not extend to structure: arguments the model never
// finished writing are not safe to hand to a client as if they were complete.
int test_unterminated_call_still_falls_back_with_contracts() {
    const fi::ToolArgumentTypeContracts contracts =
        contracts_for("edit_file", Json{{"replacements", Json{{"type", "array"}}}});
    const fi::ParsedToolCallOutput parsed =
        fi::parse_qwen_tool_call_output("<tool_call>\n"
                                        "<function=edit_file>\n"
                                        "<parameter=replacements>\n[{\"a\":1}]\n",
                                        64, contracts);
    int failures = 0;
    failures += check(!parsed.is_tool_call_response, "unterminated stays a fallback");
    failures += check(parsed.tool_calls.empty(), "unterminated yields no calls");
    return failures;
}

int test_malformed_falls_back_to_text() {
    const std::string text = "<tool_call>\n<function=get_weather>\n";
    const fi::ParsedToolCallOutput parsed =
        fi::parse_qwen_tool_call_output(text, 64, kNoTypeContracts);
    int failures = 0;
    failures += check(!parsed.is_tool_call_response, "malformed xml is not tool response");
    failures += check(parsed.content == text, "malformed xml preserved as text");
    failures += check(parsed.tool_calls.empty(), "malformed xml has no calls");
    return failures;
}

int test_suffix_after_tool_falls_back_to_text() {
    const std::string text = "<tool_call>\n"
                             "<function=get_weather>\n"
                             "<parameter=city>\nParis\n</parameter>\n"
                             "</function>\n"
                             "</tool_call>\n"
                             "extra answer";
    const fi::ParsedToolCallOutput parsed =
        fi::parse_qwen_tool_call_output(text, 64, kNoTypeContracts);
    int failures = 0;
    failures += check(!parsed.is_tool_call_response, "non-whitespace suffix falls back to text");
    failures += check(parsed.content == text, "suffix fallback preserves text");
    return failures;
}

int test_configured_name_limit() {
    const std::string name(128, 'a');
    const std::string text = "<tool_call>\n<function=" + name + ">\n</function>\n</tool_call>";

    const fi::ParsedToolCallOutput anthropic =
        fi::parse_qwen_tool_call_output(text, 128, kNoTypeContracts);
    const fi::ParsedToolCallOutput openai =
        fi::parse_qwen_tool_call_output(text, 64, kNoTypeContracts);
    const std::string too_long_text =
        "<tool_call>\n<function=" + std::string(129, 'a') + ">\n</function>\n</tool_call>";
    const fi::ParsedToolCallOutput too_long =
        fi::parse_qwen_tool_call_output(too_long_text, 128, kNoTypeContracts);

    int failures = 0;
    failures += check(anthropic.is_tool_call_response && anthropic.tool_calls.size() == 1 &&
                          anthropic.tool_calls[0].name == name,
                      "128-character name accepted with Anthropic limit");
    failures +=
        check(!openai.is_tool_call_response, "128-character name rejected with OpenAI limit");
    failures +=
        check(!too_long.is_tool_call_response, "129-character name rejected with Anthropic limit");
    return failures;
}

int test_declared_strings_are_not_json_sniffed() {
    const auto contracts = contracts_for(
        "TaskUpdate",
        Json{{"taskId", Json{{"type", "string"}}},
             {"content", Json{{"type", "string"}}},
             {"truthy", Json{{"type", "string"}}},
             {"nullish", Json{{"type", "string"}}},
             {"quoted", Json{{"type", "string"}}},
             {"windows", Json{{"type", "string"}}},
             {"string_or_number", Json{{"type", Json::array({"number", "string"})}}}});
    const fi::ParsedToolCallOutput parsed =
        fi::parse_qwen_tool_call_output("<tool_call>\n"
                                        "<function=TaskUpdate>\n"
                                        "<parameter=taskId>\n1\n</parameter>\n"
                                        "<parameter=content>\n  {\"x\":1}\n\n</parameter>\n"
                                        "<parameter=truthy>\ntrue\n</parameter>\n"
                                        "<parameter=nullish>\nnull\n</parameter>\n"
                                        "<parameter=quoted>\n\"literal\"\n</parameter>\n"
                                        "<parameter=windows>\r\n  value  \r\n</parameter>\n"
                                        "<parameter=string_or_number>\n7\n</parameter>\n"
                                        "</function>\n"
                                        "</tool_call>",
                                        128, contracts);

    int failures = 0;
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1,
                      "declared-string tool call was not parsed");
    const Json args = Json::parse(parsed.tool_calls.at(0).arguments_json);
    failures += check(args.at("taskId").is_string() && args.at("taskId") == "1",
                      "numeric-shaped task ID was not preserved as a string");
    failures += check(args.at("content") == "  {\"x\":1}\n",
                      "string content lost meaningful whitespace or was JSON-decoded");
    failures += check(args.at("truthy") == "true" && args.at("nullish") == "null",
                      "boolean/null-shaped strings were promoted");
    failures += check(args.at("quoted") == "\"literal\"",
                      "string payload was reinterpreted as embedded JSON");
    failures += check(args.at("windows") == "  value  ",
                      "CRLF framing or string spaces were not preserved");
    failures += check(args.at("string_or_number") == "7",
                      "string-admitting union destructively promoted raw text");
    return failures;
}

int test_declared_non_string_values_are_json_decoded() {
    const auto contracts = contracts_for(
        "configure", Json{{"count", Json{{"type", "integer"}}},
                          {"total", Json{{"type", "number"}}},
                          {"ratio", Json{{"type", "number"}}},
                          {"enabled", Json{{"type", "boolean"}}},
                          {"payload", Json{{"type", "object"}}},
                          {"items", Json{{"type", "array"}}},
                          {"optional", Json{{"type", Json::array({"integer", "null"})}}},
                          {"flag_or_null", Json{{"type", Json::array({"null", "boolean"})}}}});
    const auto parsed =
        fi::parse_qwen_tool_call_output("<tool_call>\n"
                                        "<function=configure>\n"
                                        "<parameter=count>\n7\n</parameter>\n"
                                        "<parameter=total>\n8\n</parameter>\n"
                                        "<parameter=ratio>\n1.5\n</parameter>\n"
                                        "<parameter=enabled>\ntrue\n</parameter>\n"
                                        "<parameter=payload>\n{\"x\":1}\n</parameter>\n"
                                        "<parameter=items>\n[\"a\",2]\n</parameter>\n"
                                        "<parameter=optional>\nnull\n</parameter>\n"
                                        "<parameter=flag_or_null>\nfalse\n</parameter>\n"
                                        "</function>\n"
                                        "</tool_call>",
                                        64, contracts);

    int failures    = 0;
    const Json args = Json::parse(parsed.tool_calls.at(0).arguments_json);
    failures += check(args.at("count").is_number_integer() && args.at("count") == 7,
                      "integer parameter was not decoded");
    failures += check(args.at("total").is_number_integer() && args.at("total") == 8,
                      "integer JSON value did not satisfy number schema");
    failures += check(args.at("ratio").is_number_float() && args.at("ratio") == 1.5,
                      "number parameter was not decoded");
    failures += check(args.at("enabled").is_boolean() && args.at("enabled") == true,
                      "boolean parameter was not decoded");
    failures += check(args.at("payload").is_object() && args.at("payload").at("x") == 1,
                      "object parameter was not decoded");
    failures += check(args.at("items").is_array() && args.at("items").at(1) == 2,
                      "array parameter was not decoded");
    failures += check(args.at("optional").is_null(), "declared nullable integer rejected null");
    failures += check(args.at("flag_or_null").is_boolean() && args.at("flag_or_null") == false,
                      "type-array order changed boolean interpretation");
    return failures;
}

int test_declared_type_mismatches_are_forwarded_without_coercion() {
    const auto contracts =
        contracts_for("configure", Json{{"object_as_integer", Json{{"type", "integer"}}},
                                        {"one_as_boolean", Json{{"type", "boolean"}}},
                                        {"string_as_boolean", Json{{"type", "boolean"}}},
                                        {"python_boolean", Json{{"type", "boolean"}}},
                                        {"null_as_boolean", Json{{"type", "boolean"}}}});

    const auto parsed =
        fi::parse_qwen_tool_call_output("<tool_call>\n"
                                        "<function=configure>\n"
                                        "<parameter=object_as_integer>\n{}\n</parameter>\n"
                                        "<parameter=one_as_boolean>\n1\n</parameter>\n"
                                        "<parameter=string_as_boolean>\n\"true\"\n</parameter>\n"
                                        "<parameter=null_as_boolean>\nnull\n</parameter>\n"
                                        "</function>\n"
                                        "</tool_call>",
                                        64, contracts);

    int failures = 0;
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1,
                      "valid JSON was rejected because it did not match the declared type");
    const Json args = Json::parse(parsed.tool_calls.at(0).arguments_json);
    failures += check(args.at("object_as_integer").is_object(),
                      "object-shaped JSON was coerced to the declared integer type");
    failures += check(args.at("one_as_boolean").is_number_integer(),
                      "numeric JSON was coerced to the declared boolean type");
    failures += check(args.at("string_as_boolean").is_string(),
                      "string JSON was coerced to the declared boolean type");
    failures += check(args.at("null_as_boolean").is_null(),
                      "null JSON was coerced to the declared boolean type");

    const std::string invalid =
        "<tool_call>\n<function=configure>\n<parameter=python_boolean>\nTrue\n</parameter>\n"
        "</function>\n</tool_call>";
    const auto degraded = fi::parse_qwen_tool_call_output(invalid, 64, contracts);
    failures += check(degraded.is_tool_call_response && degraded.tool_calls.size() == 1,
                      "non-JSON value for a declared non-string parameter discarded the call");
    if (degraded.tool_calls.size() == 1) {
        const Json degraded_args = Json::parse(degraded.tool_calls.at(0).arguments_json);
        failures += check(degraded_args.at("python_boolean").is_string() &&
                              degraded_args.at("python_boolean") == "True",
                          "non-JSON value was not forwarded as text");
    }
    return failures;
}

int test_unknown_schema_keeps_legacy_inference() {
    const auto contracts = contracts_for(
        "legacy", Json{{"missing_type", Json::object()}, {"invalid_type", Json{{"type", "int"}}}});
    const auto parsed =
        fi::parse_qwen_tool_call_output("<tool_call>\n"
                                        "<function=legacy>\n"
                                        "<parameter=missing_type>\n7\n</parameter>\n"
                                        "<parameter=invalid_type>\n8\n</parameter>\n"
                                        "<parameter=undeclared>\n9\n</parameter>\n"
                                        "</function>\n"
                                        "</tool_call>",
                                        64, contracts);

    int failures    = 0;
    const Json args = Json::parse(parsed.tool_calls.at(0).arguments_json);
    failures += check(args.at("missing_type") == 7 && args.at("invalid_type") == 8 &&
                          args.at("undeclared") == 9,
                      "unknown-schema parameter changed legacy inference");
    return failures;
}

int test_parser_enforces_active_tool_set() {
    const auto contracts = contracts_for("declared", Json{{"value", Json{{"type", "string"}}}});
    const auto parsed    = fi::parse_qwen_tool_call_output(
        "<tool_call>\n<function=other>\n<parameter=value>\n1\n</parameter>\n"
           "</function>\n</tool_call>",
        64, contracts);
    int failures = 0;
    failures += check(!parsed.is_tool_call_response && parsed.tool_calls.empty(),
                      "undeclared tool escaped the active tool-name set");
    failures += check(parsed.content.find("<function=other>") != std::string::npos,
                      "undeclared tool was not preserved as ordinary content");
    return failures;
}

int test_incremental_filter_valid_tool() {
    fi::ToolCallOutputDecoder filter(std::make_shared<fi::ToolCallOutputContract>(), 64);
    std::string visible;
    visible += filter.feed("Calling weather.  \n<tool_");
    visible += filter.feed("call>\n<function=get_weather>");
    visible += filter.feed("\n</function>\n</tool_call>");
    auto terminal = filter.finish();
    visible += terminal.content;
    int failures = 0;
    failures += check(visible == "Calling weather.",
                      "valid tool filter did not stream the trimmed content prefix");
    failures +=
        check(terminal.tool_calls.size() == 1 && terminal.tool_calls.front().name == "get_weather",
              "valid tool filter did not retain the structured call");
    return failures;
}

int test_incremental_filter_fallback() {
    const std::string original = "prefix  \n<tool_call>\n<function=broken>";
    fi::ToolCallOutputDecoder malformed(std::make_shared<fi::ToolCallOutputContract>(), 64);
    std::string restored;
    restored += malformed.feed(original.substr(0, 10));
    restored += malformed.feed(original.substr(10));
    restored += malformed.finish().content;

    fi::ToolCallOutputDecoder normal(std::make_shared<fi::ToolCallOutputContract>(), 64);
    std::string ordinary;
    ordinary += normal.feed("ordinary text  ");
    ordinary += normal.finish().content;

    const std::string partial_original = "  <tool_x then <tool_";
    fi::ToolCallOutputDecoder partial(std::make_shared<fi::ToolCallOutputContract>(), 64);
    std::string partial_restored;
    partial_restored += partial.feed("  <too");
    partial_restored += partial.feed("l_x then <tool_");
    partial_restored += partial.finish().content;

    int failures = 0;
    failures += check(restored == original, "malformed tool filter fallback lost raw bytes");
    failures +=
        check(ordinary == "ordinary text  ", "ordinary filtered output lost trailing whitespace");
    failures += check(partial_restored == partial_original,
                      "partial marker mismatch did not preserve raw bytes");
    return failures;
}

} // namespace

int test_trailing_text_opt_in_keeps_calls() {
    // A coding agent narrating after its call is the shape that broke a live
    // VS Code session: the strict rule dropped the call and handed the raw
    // markup back as prose, which the client reported as an empty response.
    const std::string text = "I will check the weather.\n"
                             "<tool_call>\n"
                             "<function=get_weather>\n"
                             "<parameter=city>\nParis\n</parameter>\n"
                             "</function>\n"
                             "</tool_call>\n"
                             "Then I will summarise it for you.";
    int failures = 0;

    const fi::ParsedToolCallOutput strict =
        fi::parse_qwen_tool_call_output(text, 64, kNoTypeContracts);
    failures += check(!strict.is_tool_call_response, "default stays strict for trailing text");
    failures += check(strict.tool_calls.empty(), "default drops the call");

    set_allow_trailing_text(true);
    const fi::ParsedToolCallOutput tolerant =
        fi::parse_qwen_tool_call_output(text, 64, kNoTypeContracts);
    set_allow_trailing_text(false);
    failures += check(tolerant.is_tool_call_response, "opt-in keeps it a tool response");
    failures += check(tolerant.tool_calls.size() == 1, "opt-in keeps the call");
    failures += check(tolerant.tool_calls[0].name == "get_weather", "opt-in parses the name");
    failures += check(tolerant.content.find("I will check the weather.") != std::string::npos,
                      "opt-in keeps the prefix");
    failures += check(tolerant.content.find("Then I will summarise it") != std::string::npos,
                      "opt-in keeps the narration as content");
    failures += check(tolerant.content.find("<tool_call>") == std::string::npos,
                      "opt-in never leaks markup into content");
    return failures;
}

int test_trailing_text_opt_in_does_not_rescue_malformed() {
    // Tolerance must not invent calls: an unterminated block still falls back,
    // because accepting it could execute arguments the model never finished.
    const std::string text = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n";
    set_allow_trailing_text(true);
    const fi::ParsedToolCallOutput parsed =
        fi::parse_qwen_tool_call_output(text, 64, kNoTypeContracts);
    set_allow_trailing_text(false);
    int failures = 0;
    failures += check(!parsed.is_tool_call_response, "unterminated stays a fallback");
    failures += check(parsed.tool_calls.empty(), "unterminated yields no calls");
    return failures;
}

int main() {
    int failures = 0;
    failures += test_single_call();
    failures += test_multiple_calls_and_json_values();
    failures += test_unparseable_typed_parameter_degrades_to_text();
    failures += test_unterminated_call_still_falls_back_with_contracts();
    failures += test_malformed_falls_back_to_text();
    failures += test_suffix_after_tool_falls_back_to_text();
    failures += test_trailing_text_opt_in_keeps_calls();
    failures += test_trailing_text_opt_in_does_not_rescue_malformed();
    failures += test_configured_name_limit();
    failures += test_declared_strings_are_not_json_sniffed();
    failures += test_declared_non_string_values_are_json_decoded();
    failures += test_declared_type_mismatches_are_forwarded_without_coercion();
    failures += test_unknown_schema_keeps_legacy_inference();
    failures += test_parser_enforces_active_tool_set();
    failures += test_incremental_filter_valid_tool();
    failures += test_incremental_filter_fallback();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
