#include "runtime/contract/structured_output.h"
#include "targets/qwen3_6/impl/frontend/tool_call_parser.h"

#include <nlohmann/json.hpp>
#include <iostream>
#include <stdexcept>

using namespace ninfer;
using namespace ninfer::runtime;
namespace fi = ninfer::targets::qwen3_6::frontend_internal;

namespace {
void require(bool value, const char* message) {
    if (!value) { throw std::runtime_error(message); }
}
bool allowed(OutputConstraintState& state, TokenId token) {
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
        std::vector<std::string> vocab(260);
        for (int i = 0; i < 256; ++i) { vocab[i] = std::string(1, static_cast<char>(i)); }
        vocab[258] = "<tool_call>";
        vocab[259] = "</tool_call>";
        OutputConstraintCompiler compiler(std::move(vocab), {256}, 257);
        auto compiled = compiler.compile_grammar(fi::required_tool_call_grammar({"ping", "edit_file"}));
        OutputConstraintState state(compiled, false);
        require(!allowed(state, 256) && !allowed(state, 'I'), "required call allowed EOS or prose");
        require(allowed(state, 258), "atomic tool-open token was masked");
        const std::string empty = "<tool_call>\n<function=ping>\n</function>\n</tool_call>";
        require(accepts(state.fork(), empty), "no-argument call rejected");
        const std::string value = "<section>ação</section>\nif (a < b) { return \"</param\"; }";
        const std::string edit = "<tool_call><function=edit_file><parameter=text>\n" + value +
            "\n</parameter><parameter=count>3</parameter></function></tool_call>";
        require(accepts(state.fork(), edit), "HTML, code, UTF-8 or partial delimiter rejected");
        const std::vector<std::string> declarations{
            R"({"type":"function","function":{"name":"edit_file","parameters":{"type":"object","properties":{"text":{"type":"string"},"count":{"type":"integer"}}}}})"
        };
        const auto contracts = fi::build_tool_call_output_contract(declarations, true);
        const auto parsed = fi::parse_qwen_tool_call_output(edit, 64, contracts->argument_types);
        require(parsed.is_tool_call_response && parsed.tool_calls.size() == 1, "grammar output did not parse as a call");
        const auto args = nlohmann::json::parse(parsed.tool_calls[0].arguments_json);
        require(args["text"] == value && args["count"] == 3, "argument contents changed");
        for (const auto& invalid : {std::string("I cannot do that."), empty + "done", empty + empty,
                std::string("<tool_call><function=undeclared></function></tool_call>"),
                std::string("<tool_call><function=ping>"),
                std::string("<tool_call><function=ping><parameter=x>bad</function>tail</parameter></function></tool_call>")}) {
            require(!accepts(state.fork(), invalid), "malformed or non-call output admitted");
        }
        auto draft = state.fork();
        require(draft.try_accept(258), "draft rejected tool token");
        require(allowed(state, 258) && !allowed(draft, 258), "draft mutated committed mask");
        OutputConstraintState reasoning(compiled, true);
        require(allowed(reasoning, 'I') && !allowed(reasoning, 256), "reasoning allowed premature stop");
        require(reasoning.try_accept(257), "reasoning close rejected");
        require(accepts(reasoning.fork(), "\n\n" + empty), "forced reasoning-close whitespace rejected");
        std::cout << "required call masks, parsing, reasoning, atomic tokens and speculative fork passed\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
