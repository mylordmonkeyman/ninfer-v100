#include "artifact/reader.h"
#include "core/arena.h"
#include "core/device.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/targets/qwen3_6/frontend.h"
#include "ninfer/targets/qwen3_6/frontend_resources.h"
#include "ninfer/targets/qwen3_6/prepared_prompt.h"
#include "targets/qwen3_8_flash_next/impl/frontend.h"
#include "targets/qwen3_8_flash_next/impl/load/loader.h"
#include "targets/qwen3_8_flash_next/impl/runtime_plan.h"
#include "targets/qwen3_8_flash_next/impl/runtime_state.h"
#include "targets/qwen3_8_flash_next/impl/text_executor.h"
#include "tools/reference/qwen3_8_flash_next/options.h"

#include <cuda_runtime.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace ninfer::targets::qwen3_8_flash_next::detail;
using LoadedModel = StandaloneLoadedModel;

int test_chat_diagnostic_options_parsing() {
    // 1. Valid full options
    const std::vector<std::string_view> full_args = {
        "--model",            "model.ninfer",
        "--chat-diagnostic",
        "--prompt",           "Tell me about physics",
        "--system",           "You are a helpful assistant.",
        "--temperature",      "0.7",
        "--top-k",            "10",
        "--top-p",            "0.9",
        "--seed",             "42",
        "--max-tokens",       "128",
        "--reasoning-effort", "high",
    };
    try {
        const auto opts = parse_reference_tool_options(full_args);
        if (opts.model_path != "model.ninfer" || opts.mode != "chat-diagnostic" ||
            opts.prompt != "Tell me about physics" ||
            opts.system_prompt != "You are a helpful assistant." ||
            opts.temperature != 0.7f || opts.top_k != 10 || opts.top_p != 0.9f ||
            opts.seed != 42ULL || opts.max_tokens != 128 ||
            opts.reasoning_effort != "high") {
            std::cerr << "Parsed options mismatch for chat diagnostic full args\n";
            return 1;
        }
    } catch (const std::exception& ex) {
        std::cerr << "Unexpected failure parsing chat diagnostic args: " << ex.what() << "\n";
        return 1;
    }

    // 2. Greedy flag
    const std::vector<std::string_view> greedy_args = {
        "--model", "model.ninfer", "--chat-diagnostic", "--greedy",
    };
    try {
        const auto opts = parse_reference_tool_options(greedy_args);
        if (opts.temperature != 0.0f) {
            std::cerr << "Greedy flag did not set temperature to 0.0f\n";
            return 1;
        }
    } catch (const std::exception& ex) {
        std::cerr << "Unexpected failure parsing greedy args: " << ex.what() << "\n";
        return 1;
    }

    // 3. Helper lambda to test that invalid arguments throw std::invalid_argument
    const auto assert_throws = [](std::initializer_list<std::string_view> args,
                                  std::string_view label) -> bool {
        try {
            std::vector<std::string_view> vec(args);
            (void)parse_reference_tool_options(vec);
            std::cerr << "Failed to reject " << label << "\n";
            return false;
        } catch (const std::invalid_argument&) { return true; }
    };

    // 4. Reject thinking-budget forcing
    if (!assert_throws({"--model", "m.ninfer", "--chat-diagnostic", "--thinking-budget", "100"},
                       "positive thinking-budget")) {
        return 1;
    }

    // 5. Reject invalid temperature, top-p, max-tokens
    if (!assert_throws({"--model", "m.ninfer", "--temperature", "-0.5"}, "negative temperature"))
        return 1;
    if (!assert_throws({"--model", "m.ninfer", "--top-p", "0.0"}, "zero top-p")) return 1;
    if (!assert_throws({"--model", "m.ninfer", "--top-p", "1.5"}, "top-p > 1.0")) return 1;
    if (!assert_throws({"--model", "m.ninfer", "--max-tokens", "0"}, "zero max-tokens"))
        return 1;
    if (!assert_throws({"--model", "m.ninfer", "--mode", "invalid-mode"}, "invalid mode"))
        return 1;

    std::cout << "PASS: test_chat_diagnostic_options_parsing\n";
    return 0;
}

int test_oracle_chat23_logits_options() {
    const std::vector<std::string_view> args = {
        "--model", "model.ninfer", "--oracle-chat23-logits", "E:/tmp/oracle-mma-on",
        "--qsa-prefill-mma",
    };
    try {
        const auto opts = parse_reference_tool_options(args);
        if (opts.mode != "oracle-chat23-logits" ||
            opts.oracle_chat23_logits != "E:/tmp/oracle-mma-on" || !opts.qsa_prefill_mma ||
            opts.model_path != "model.ninfer") {
            std::cerr << "Parsed options mismatch for oracle-chat23-logits + qsa-prefill-mma\n";
            return 1;
        }
    } catch (const std::exception& ex) {
        std::cerr << "Unexpected failure parsing oracle-chat23-logits args: " << ex.what() << "\n";
        return 1;
    }

    const std::vector<std::string_view> off_args = {
        "--model", "model.ninfer", "--oracle-chat23-logits", "E:/tmp/oracle-mma-off",
    };
    try {
        const auto opts = parse_reference_tool_options(off_args);
        if (opts.mode != "oracle-chat23-logits" || opts.qsa_prefill_mma ||
            opts.oracle_chat23_logits != "E:/tmp/oracle-mma-off") {
            std::cerr << "oracle-chat23-logits without --qsa-prefill-mma should leave MMA off\n";
            return 1;
        }
    } catch (const std::exception& ex) {
        std::cerr << "Unexpected failure parsing oracle-chat23-logits off args: " << ex.what()
                  << "\n";
        return 1;
    }

    try {
        const std::vector<std::string_view> missing = {"--model", "m.ninfer",
                                                      "--oracle-chat23-logits"};
        (void)parse_reference_tool_options(missing);
        std::cerr << "Failed to reject missing --oracle-chat23-logits directory\n";
        return 1;
    } catch (const std::invalid_argument&) {}

    std::cout << "PASS: test_oracle_chat23_logits_options\n";
    return 0;
}

int test_sampler_token_domain() {
    ninfer::DeviceContext device(0);

    constexpr std::int32_t kPhysicalRows        = 248'320;
    constexpr std::int32_t kSemanticTokenDomain = 248'077;
    constexpr std::int32_t kTargetPaddedRow     = 248'100;

    // Fill host logits: semantic rows are 0.0f, padded rows are huge (+100.0f)
    std::vector<std::uint16_t> host_logits_bf16(kPhysicalRows, 0);
    // 0.0f in BF16 is 0x0000
    // +100.0f in BF16: 100.0f = 0x42c80000 -> BF16 = 0x42c8
    // +10.0f in BF16:  10.0f  = 0x41200000 -> BF16 = 0x4120
    constexpr std::uint16_t kBf16_Zero = 0x0000;
    constexpr std::uint16_t kBf16_Ten  = 0x4120;
    constexpr std::uint16_t kBf16_Huge = 0x42c8;

    for (std::int32_t i = 0; i < kPhysicalRows; ++i) {
        if (i == 42) {
            host_logits_bf16[i] = kBf16_Ten; // highest in semantic domain
        } else if (i < kSemanticTokenDomain) {
            host_logits_bf16[i] = kBf16_Zero;
        } else {
            host_logits_bf16[i] = kBf16_Huge; // huge in padded domain (>= 248077)
        }
    }

    ninfer::DeviceBuffer device_logits(kPhysicalRows * sizeof(std::uint16_t));
    device_logits.copy_from_host(host_logits_bf16.data(), device_logits.bytes);

    ninfer::ops::SamplingConfig sampling_cfg{
        .temperature = 0.0f, // greedy
        .top_k       = 20,
        .top_p       = 0.95f,
        .seed        = 0,
    };
    ninfer::DeviceBuffer device_configs(sizeof(ninfer::ops::SamplingConfig));
    device_configs.copy_from_host(&sampling_cfg, sizeof(ninfer::ops::SamplingConfig));

    std::int32_t pos = 0;
    ninfer::DeviceBuffer device_pos(sizeof(std::int32_t));
    device_pos.copy_from_host(&pos, sizeof(std::int32_t));

    ninfer::DeviceBuffer device_out(sizeof(std::int32_t));

    ninfer::Tensor logits_tensor(device_logits.p, ninfer::DType::BF16, {kPhysicalRows, 1});
    ninfer::Tensor out_tensor(device_out.p, ninfer::DType::I32, {1});
    ninfer::Tensor positions_tensor(device_pos.p, ninfer::DType::I32, {1});

    const std::size_t ws_bytes =
        ninfer::ops::sampling_workspace_capacity_bytes(kSemanticTokenDomain, 1, 1);
    ninfer::WorkspaceArena workspace(std::max<std::size_t>(256, ws_bytes));

    ninfer::ops::sample(logits_tensor, out_tensor, kSemanticTokenDomain,
                        static_cast<const ninfer::ops::SamplingConfig*>(device_configs.p),
                        positions_tensor, ninfer::ops::kSamplePurposeDecode, workspace,
                        device.stream);

    device.synchronize();
    std::int32_t sampled_token = -1;
    device_out.copy_to_host(&sampled_token, sizeof(std::int32_t));

    if (sampled_token != 42) {
        std::cerr << "Sampler selected token " << sampled_token
                  << " instead of semantic argmax 42; padded rows were not properly ignored\n";
        return 1;
    }

    std::cout << "PASS: test_sampler_token_domain\n";
    return 0;
}

int test_frontend_parity_with_embedded_resources_if_available() {
    const char* env_path = std::getenv("NINFER_WEIGHTS");
    const std::filesystem::path path =
        env_path ? std::filesystem::path(env_path)
                 : std::filesystem::path(
                       "" /* real-artifact cases run only when NINFER_WEIGHTS is set explicitly */);

    if (!std::filesystem::is_regular_file(path)) {
        std::cout << "SKIP: test_frontend_parity (artifact not present at " << path << ")\n";
        return 0;
    }

    ninfer::DeviceContext device(0);
    auto loaded = LoadedModel::load_from_file(
        path.string(), device, LoadFeatures{.vision = false, .mtp = false});

    const auto& fe = loaded.frontend_resources();
    if (fe.tokenizer_json.empty() || fe.tokenizer_config_json.empty() ||
        fe.chat_template_jinja.empty() || fe.generation_config_json.empty() ||
        fe.preprocessor_config_json.empty() || fe.video_preprocessor_config_json.empty()) {
        std::cerr << "Expected non-empty frontend resources\n";
        return 1;
    }

    // Prove exact byte identity of all six frontend resources against the artifact
    ninfer::artifact::Reader reader(path);
    const auto check_resource = [&](std::string_view name, const std::string& loaded_str) -> bool {
        const auto raw = reader.payload(name);
        return std::string_view(reinterpret_cast<const char*>(raw.data.data()), raw.data.size()) ==
               loaded_str;
    };
    if (!check_resource("frontend/tokenizer.json", fe.tokenizer_json) ||
        !check_resource("frontend/tokenizer_config.json", fe.tokenizer_config_json) ||
        !check_resource("frontend/chat_template.jinja", fe.chat_template_jinja) ||
        !check_resource("frontend/generation_config.json", fe.generation_config_json) ||
        !check_resource("frontend/preprocessor_config.json", fe.preprocessor_config_json) ||
        !check_resource("frontend/video_preprocessor_config.json",
                        fe.video_preprocessor_config_json)) {
        std::cerr << "Frontend resources byte identity check against artifact failed\n";
        return 1;
    }

    auto frontend = make_frontend(
        loaded,
        ninfer::targets::qwen3_6::FrontendOptions{
            .vision_enabled = false,
            .max_context    = 4096,
        });

    // 1. Verify canary prompt renders exact 23 tokens
    {
        ninfer::PromptInput canary_input;
        canary_input.options.reasoning_effort = ninfer::ReasoningEffort::Medium;
        ninfer::ChatMessage user_msg;
        user_msg.role = ninfer::ChatRole::User;
        user_msg.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text,
            .text = "Reply in one short sentence: what makes a good inference engine?",
        });
        canary_input.messages.push_back(std::move(user_msg));

        auto prepared = frontend.prepare(std::move(canary_input));
        const auto& prepared_data =
            ninfer::targets::qwen3_6::PreparedPromptAccess::view(prepared);
        const auto& prompt_tokens = prepared_data.token_ids;
        const std::vector<ninfer::TokenId> expected_canary_tokens = {
            248045, 846, 198, 20206, 303, 799, 2716, 11316, 25, 1092, 3520, 264,
            1603, 42903, 4560, 30, 248046, 198, 248045, 74455, 198, 248068, 198
        };
        if (std::vector<ninfer::TokenId>(prompt_tokens.begin(), prompt_tokens.end()) !=
            expected_canary_tokens) {
            std::cerr << "Canary prompt tokens mismatch! Expected 23 tokens, got "
                      << prompt_tokens.size() << "\n";
            return 1;
        }
    }

    ninfer::PromptInput input;
    ninfer::ChatMessage user_msg;
    user_msg.role = ninfer::ChatRole::User;
    user_msg.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text,
        .text = "Hello",
    });
    input.messages.push_back(std::move(user_msg));

    auto prepared = frontend.prepare(std::move(input));
    const auto& prepared_data =
        ninfer::targets::qwen3_6::PreparedPromptAccess::view(prepared);
    const auto& prompt_tokens = prepared_data.token_ids;
    if (prompt_tokens.empty()) {
        std::cerr << "Prepared prompt yielded zero tokens\n";
        return 1;
    }

    ninfer::StopPolicy stop_policy;
    stop_policy.token_ids = {248'046, 248'044};
    auto output_session   = frontend.make_output_session(
        prepared, stop_policy, ninfer::OutputOptions{});

    // Test stop token detection
    const ninfer::TokenId stop_tok = 248'044;
    const auto dec                 = output_session.preview_model(
        std::span<const ninfer::TokenId>(&stop_tok, 1), 10,
        ninfer::FinishReason::OutputLimit);

    if (!dec.finished()) {
        std::cerr << "OutputSession failed to finish on stop token 248044\n";
        return 1;
    }

    std::cout << "PASS: test_frontend_parity_with_embedded_resources (prompt tokens: "
              << prompt_tokens.size() << ")\n";
    return 0;
}

int test_real_artifact_chat_diagnostic_smoke_if_available() {
    const char* env_path = std::getenv("NINFER_WEIGHTS");
    const std::filesystem::path path =
        env_path ? std::filesystem::path(env_path)
                 : std::filesystem::path(
                       "" /* real-artifact cases run only when NINFER_WEIGHTS is set explicitly */);

    if (!std::filesystem::is_regular_file(path)) {
        std::cout << "SKIP: test_real_artifact_chat_diagnostic_smoke (artifact not present at "
                  << path << ")\n";
        return 0;
    }

    // Let's run a short 3-token deterministic chat smoke
    ninfer::DeviceContext device(0);
    auto loaded = LoadedModel::load_from_file(
        path.string(), device, LoadFeatures{.vision = false, .mtp = false});

    auto frontend = make_frontend(
        loaded,
        ninfer::targets::qwen3_6::FrontendOptions{
            .vision_enabled = false,
            .max_context    = 4096,
        });

    ninfer::PromptInput input;
    ninfer::ChatMessage user_msg;
    user_msg.role = ninfer::ChatRole::User;
    user_msg.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text,
        .text = "Hi",
    });
    input.messages.push_back(std::move(user_msg));

    auto prepared = frontend.prepare(std::move(input));
    const auto& prepared_data =
        ninfer::targets::qwen3_6::PreparedPromptAccess::view(prepared);
    const auto& prompt_tokens = prepared_data.token_ids;
    const auto pos0           = prepared_data.position_axis(0);
    const auto pos1           = prepared_data.position_axis(1);
    const auto pos2           = prepared_data.position_axis(2);

    FlashNextRuntimeConfig config{
        .max_concurrency     = 1,
        .max_context         = 4096,
        .state_slot_capacity = 2,
    };
    const auto curve = flash_next_capacity_curve(config);
    const auto plan  = finalize_flash_next_runtime_plan(config, curve.maximum_main_page_groups);
    FlashNextRuntimeAllocation alloc(plan);
    alloc.initialize(device.stream);
    FlashNextTextExecutor executor(loaded.text_view(), kPleIndexMetadata, device, alloc);
    auto lane = executor.allocate_lane();

    constexpr std::int32_t kSemanticTokenDomain = 248'077;
    const std::size_t ws_bytes =
        ninfer::ops::sampling_workspace_capacity_bytes(kSemanticTokenDomain, 1, 1);
    ninfer::WorkspaceArena workspace(std::max<std::size_t>(256, ws_bytes));

    ninfer::ops::SamplingConfig sampling_cfg{
        .temperature = 0.0f, // greedy
        .top_k       = 20,
        .top_p       = 0.95f,
        .seed        = 0,
    };

    ninfer::DeviceBuffer device_configs(sizeof(ninfer::ops::SamplingConfig));
    device_configs.copy_from_host(&sampling_cfg, sizeof(ninfer::ops::SamplingConfig));
    ninfer::DeviceBuffer device_positions(sizeof(std::int32_t));
    ninfer::DeviceBuffer device_out(sizeof(std::int32_t));
    ninfer::Tensor out_tensor(device_out.p, ninfer::DType::I32, {1});
    ninfer::Tensor positions_tensor(device_positions.p, ninfer::DType::I32, {1});

    ninfer::StopPolicy stop_policy;
    stop_policy.token_ids = {248'046, 248'044};
    auto output_session   = frontend.make_output_session(
        prepared, stop_policy, ninfer::OutputOptions{});

    // Prefill
    const std::size_t prompt_len = prompt_tokens.size();
    for (std::size_t i = 0; i + 1 < prompt_len; ++i) {
        LaneStepRequest req{
            .handle          = lane,
            .token_id        = prompt_tokens[i],
            .token_index     = static_cast<std::int32_t>(i),
            .mrope_positions = {pos0[i], pos1[i], pos2[i]},
        };
        auto round = executor.execute_round(std::span<const LaneStepRequest>(&req, 1));
        std::vector<LaneCommitDecision> decision = {{.accept = true}};
        round.commit(decision);
    }
    LaneStepRequest last_prompt_req{
        .handle          = lane,
        .token_id        = prompt_tokens[prompt_len - 1],
        .token_index     = static_cast<std::int32_t>(prompt_len - 1),
        .mrope_positions = {pos0[prompt_len - 1], pos1[prompt_len - 1], pos2[prompt_len - 1]},
    };
    auto round = executor.execute_round(std::span<const LaneStepRequest>(&last_prompt_req, 1));

    // Generate 3 tokens
    std::uint32_t generated_tokens   = 0;
    std::int32_t current_pos         = pos0[prompt_len - 1];
    std::int32_t current_token_index = static_cast<std::int32_t>(prompt_len - 1);
    constexpr std::uint32_t kMaxGen  = 3;

    while (generated_tokens < kMaxGen) {
        device_positions.copy_from_host(&current_pos, sizeof(std::int32_t));
        ninfer::ops::sample(
            round.logits(), out_tensor, kSemanticTokenDomain,
            static_cast<const ninfer::ops::SamplingConfig*>(device_configs.p),
            positions_tensor, ninfer::ops::kSamplePurposeDecode, workspace, device.stream);
        std::int32_t sampled_token = 0;
        device_out.copy_to_host(&sampled_token, sizeof(std::int32_t));
        device.synchronize();

        std::vector<LaneCommitDecision> decision = {{.accept = true}};
        round.commit(decision);

        ++generated_tokens;

        const ninfer::TokenId tok_id = static_cast<ninfer::TokenId>(sampled_token);
        const std::uint32_t remaining_budget =
            kMaxGen >= (generated_tokens - 1) ? kMaxGen - (generated_tokens - 1) : 0;
        const auto dec = output_session.preview_model(
            std::span<const ninfer::TokenId>(&tok_id, 1), remaining_budget,
            ninfer::FinishReason::OutputLimit);
        (void)output_session.commit_preview();

        if (dec.finished()) { break; }

        ++current_pos;
        ++current_token_index;
        LaneStepRequest next_req{
            .handle          = lane,
            .token_id        = sampled_token,
            .token_index     = current_token_index,
            .mrope_positions = {current_pos, current_pos, current_pos},
        };
        round = executor.execute_round(std::span<const LaneStepRequest>(&next_req, 1));
    }

    executor.release_lane(lane);

    if (generated_tokens != kMaxGen) {
        std::cerr << "Expected " << kMaxGen << " generated tokens, got " << generated_tokens
                  << "\n";
        return 1;
    }

    std::cout << "PASS: test_real_artifact_chat_diagnostic_smoke\n";
    return 0;
}

} // namespace

int main() {
    std::cout << "Starting Flash-Next Chat Diagnostic Tests...\n" << std::flush;
    try {
        std::cout << "[1/5] Running test_chat_diagnostic_options_parsing...\n" << std::flush;
        if (test_chat_diagnostic_options_parsing() != 0) return 1;

        std::cout << "[2/5] Running test_oracle_chat23_logits_options...\n" << std::flush;
        if (test_oracle_chat23_logits_options() != 0) return 1;

        std::cout << "[3/5] Running test_sampler_token_domain...\n" << std::flush;
        if (test_sampler_token_domain() != 0) return 1;

        std::cout << "[4/5] Running test_frontend_parity_with_embedded_resources_if_available...\n" << std::flush;
        if (test_frontend_parity_with_embedded_resources_if_available() != 0) return 1;

        std::cout << "[5/5] Running test_real_artifact_chat_diagnostic_smoke_if_available...\n" << std::flush;
        if (test_real_artifact_chat_diagnostic_smoke_if_available() != 0) return 1;
    } catch (const std::exception& ex) {
        std::cerr << "Fatal exception in test_chat_diagnostic: " << ex.what() << "\n" << std::flush;
        return 1;
    }

    std::cout << "OK Flash-Next Chat Diagnostic Tests\n" << std::flush;
    return 0;
}
