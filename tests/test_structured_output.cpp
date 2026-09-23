#include "runtime/contract/structured_output.h"
#include <nlohmann/json.hpp>
#include <iostream>
#include <stdexcept>

using namespace ninfer;
using namespace ninfer::runtime;
namespace {
void require(bool value, const char* message) { if (!value) { throw std::runtime_error(message); } }
bool allowed(OutputConstraintState& state, int token) {
    const auto mask = state.next_mask();
    return (static_cast<std::uint32_t>(mask[token / 32]) & (1U << (token % 32))) != 0;
}
bool accepts(OutputConstraintState state, std::string_view text) {
    for (unsigned char byte : text) {
        if (!allowed(state, byte) || !state.try_accept(byte)) { return false; }
    }
    return allowed(state, 256) && state.try_accept(256) && state.terminated();
}
}
int main() {
    try {
        std::vector<std::string> vocab(258);
        for (int i = 0; i < 256; ++i) { vocab[i] = std::string(1, static_cast<char>(i)); }
        OutputConstraintCompiler compiler(std::move(vocab), {256}, 257);
        const auto json = compiler.compile({StructuredOutputKind::JsonObject, {}});
        OutputConstraintState initial(json, false);
        require(!allowed(initial, 256), "EOS admitted before a JSON object");
        require(!allowed(initial, '[') && !allowed(initial, 'x'), "non-object start admitted");
        require(accepts(initial.fork(), R"({"text":"a\\b\"c","array":[null,true,1.25,-2],"nested":{}})"), "valid JSON rejected");
        require(!accepts(initial.fork(), R"({"text":"unterminated})"), "invalid string accepted");
        require(!accepts(initial.fork(), R"({"x":01})"), "invalid number accepted");
        const std::string schema = R"({"type":"object","properties":{"status":{"type":"string","enum":["confirmada","contraposta"]},"n":{"type":"integer","minimum":1,"maximum":3}},"required":["status","n"],"additionalProperties":false})";
        auto compiled = compiler.compile({StructuredOutputKind::JsonSchema, schema});
        OutputConstraintState structured(compiled, false);
        const std::string valid = R"({"status":"confirmada","n":2})";
        const auto parsed = nlohmann::json::parse(valid);
        require(parsed["status"] == "confirmada" && parsed["n"] == 2, "independent JSON parse failed");
        require(accepts(structured.fork(), valid), "schema-valid object rejected");
        for (auto invalid : {R"({"status":"inventada","n":2})", R"({"status":"confirmada"})",
                             R"({"status":"confirmada","n":9})", R"({"status":"confirmada","n":2,"extra":0})"}) {
            require(!accepts(structured.fork(), invalid), "schema-invalid object admitted");
        }
        auto speculative = structured.fork();
        const TokenId brace = '{';
        speculative.accept(std::span(&brace, 1));
        require(!allowed(speculative, '{') && allowed(structured, '{'), "speculative state leaked into committed state");
        OutputConstraintState reasoning(compiled, true);
        require(allowed(reasoning, 'x') && !allowed(reasoning, 256), "reasoning mask invalid");
        const TokenId end = 257;
        reasoning.accept(std::span(&end, 1));
        require(!allowed(reasoning, 'x') && allowed(reasoning, '{'), "reasoning boundary did not activate JSON constraint");
        require(accepts(reasoning.fork(), "\n\n" + valid), "forced reasoning close whitespace rejected");
        for (auto unsupported : {R"({"type":"array","uniqueItems":true})", R"({"oneOf":[{"type":"string"},{"type":"number"}]})",
                                 R"({"type":"string","format":"email"})", R"({"type":"integer","const":"wrong"})",
                                 R"({"minimum":3})", R"({"required":["x"]})",
                                 R"({"type":"string","pattern":".*"})", R"({"type":"string","maxLength":3})",
                                 R"({"$defs":{"a/b":{"type":"integer"}},"$ref":"#/$defs/a~1b"})",
                                 R"({"default":{"type":"string","pattern":".*"},"$ref":"#/default"})",
                                 R"({"type":"object","required":["x"]})", R"({"type":[],"minimum":5})"}) {
            bool rejected = false;
            try { (void)compiler.compile({StructuredOutputKind::JsonSchema, unsupported}); }
            catch (const std::invalid_argument&) { rejected = true; }
            require(rejected, "unsupported or contradictory assertion silently accepted");
        }
        auto array = compiler.compile({StructuredOutputKind::JsonSchema,
            R"({"type":"array","items":{"anyOf":[{"type":"string"},{"type":"null"}]},"minItems":1,"maxItems":2})"});
        require(accepts(OutputConstraintState(array, false), R"(["ação",null])"), "UTF-8 nullable array rejected");
        require(!accepts(OutputConstraintState(array, false), "[]"), "minItems ignored");
        require(!accepts(OutputConstraintState(array, false), "[null,null,null]"), "maxItems ignored");
        auto ref = compiler.compile({StructuredOutputKind::JsonSchema,
            R"({"$defs":{"result":{"type":"string","enum":["ok"]}},"$ref":"#/$defs/result"})"});
        require(accepts(OutputConstraintState(ref, false), R"("ok")"), "local reference rejected");
        require(!accepts(OutputConstraintState(ref, false), R"("bad")"), "local reference ignored");
        auto number = compiler.compile({StructuredOutputKind::JsonSchema,
            R"({"type":"number","exclusiveMinimum":1.25,"maximum":3.5})"});
        require(accepts(OutputConstraintState(number, false), "2.5"), "bounded decimal rejected");
        for (auto invalid : {"1.25", "3.51", "-2", "2e9"}) {
            require(!accepts(OutputConstraintState(number, false), invalid), "numeric bound ignored");
        }
        auto tuple = compiler.compile({StructuredOutputKind::JsonSchema,
            R"({"type":"array","prefixItems":[{"type":"integer"},{"type":"boolean"}],"items":false,"minItems":2})"});
        require(accepts(OutputConstraintState(tuple, false), "[1,true]"), "tuple rejected");
        require(!accepts(OutputConstraintState(tuple, false), "[true,1]"), "tuple position schema ignored");
        require(!accepts(OutputConstraintState(tuple, false), "[1,true,null]"), "items:false ignored");
        require(!accepts(initial.fork(), std::string("{\"x\":\"raw") + '\t' + "tab\"}"), "raw JSON control admitted");
        const std::string order_schema = R"({"type":"object","properties":{"zebra":{"type":"string"},"alpha":{"type":"string"}},"required":["zebra","alpha"],"additionalProperties":false})";
        auto order_compiled = compiler.compile({StructuredOutputKind::JsonSchema, order_schema});
        OutputConstraintState order_state(order_compiled, false);
        require(accepts(order_state.fork(), R"({"zebra":"z","alpha":"a"})"), "schema property declaration order rejected");
        require(!accepts(order_state.fork(), R"({"alpha":"a","zebra":"z"})"), "alphabetical property order admitted over schema declaration order");
        std::cout << "structured output grammar, masks, schema, reasoning and fork tests passed\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
