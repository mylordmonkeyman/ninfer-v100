#include "ninfer/engine.h"
#include <cstdlib>
#include <iostream>
#include <vector>

int main() {
    try {
        ninfer::EngineOptions options;
        // Real-artifact cases run only when NINFER_WEIGHTS is set explicitly (311cc0f5).
        const char* weights = std::getenv("NINFER_WEIGHTS");
        if (weights == nullptr || weights[0] == 0) {
            std::cout << "SKIP: NINFER_WEIGHTS not set" << std::endl;
            return 77;
        }
        options.artifact_path = weights;
        options.max_concurrency = 1;
        options.max_pending_requests = 1;
        options.max_context = 4096;
        options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(4096);
        options.prefill_chunk = 1024;
        options.context_cache.enabled = true;
        options.context_cache.device_state_slots = 4;
        options.context_cache.max_private_continuations = 4;
        options.context_cache.max_shared_prefixes       = 2;
        options.context_cache.max_long_anchors_per_continuation = 2;

        std::cout << "Creating Engine..." << std::endl;
        ninfer::Engine engine(options);
        std::cout << "Engine created successfully!" << std::endl;

        ninfer::PromptInput input;
        ninfer::ChatMessage msg;
        msg.role = ninfer::ChatRole::User;
        msg.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text,
            .text = "What is 17 + 28? Answer in one number.",
            .media = {}
        });
        input.messages.push_back(std::move(msg));

        std::cout << "Preparing prompt..." << std::endl;
        auto prepared = engine.prepare(std::move(input));
        std::cout << "Prompt prepared (" << prepared.summary().prompt_tokens << " tokens)!" << std::endl;

        ninfer::RequestOptions req_opts;
        req_opts.execution.requested_output_tokens = 32;
        req_opts.execution.sampling.temperature    = 0.0F;

        std::cout << "Generating response..." << std::endl;
        auto result = engine.generate(std::move(prepared), req_opts);
        std::cout << "Generated text: " << result.content << std::endl;
        std::cout << "Tokens generated: " << result.generated_token_ids.size() << std::endl;
        std::cout << "Token IDs: ";
        for (auto tok : result.generated_token_ids) {
            std::cout << tok << " ";
        }
        std::cout << std::endl;
        std::cout << "Turn 1 Finished!" << std::endl;

        // Turn 2: Exact continuation reuse using tokens
        std::vector<ninfer::TokenId> prompt1_tokens(67, 198);
        prompt1_tokens[0] = 151644;
        prompt1_tokens[1] = 872;
        prompt1_tokens[2] = 198;
        
        std::cout << "\nPreparing Turn 2 with exact prefix (67 prompt + 32 generated + 5 delta)..." << std::endl;
        std::vector<ninfer::TokenId> t2_tokens;
        t2_tokens.reserve(67 + 32 + 5);
        // Copy turn 1 prompt tokens from prepared prompt data or rebuild
        // Turn 1 prompt tokens + generated tokens:
        // Let's run a clean 2-turn token test
        auto prep1 = engine.prepare_tokens(prompt1_tokens);
        auto res1 = engine.generate(std::move(prep1), req_opts);
        std::cout << "Token Turn 1 finished, generated " << res1.generated_token_ids.size() << " tokens." << std::endl;

        std::vector<ninfer::TokenId> prompt2_tokens = prompt1_tokens;
        prompt2_tokens.insert(prompt2_tokens.end(), res1.generated_token_ids.begin(), res1.generated_token_ids.end());
        prompt2_tokens.push_back(198);
        prompt2_tokens.push_back(151644);
        prompt2_tokens.push_back(872);
        prompt2_tokens.push_back(198);
        prompt2_tokens.push_back(100);

        auto prep2 = engine.prepare_tokens(prompt2_tokens);
        std::cout << "Turn 2 prompt tokens: " << prep2.summary().prompt_tokens << std::endl;
        auto res2 = engine.generate(std::move(prep2), req_opts);
        std::cout << "Turn 2 Reused prompt tokens: " << res2.reused_prompt_tokens << " (Expected: " << (prompt1_tokens.size() + res1.generated_token_ids.size()) << ")" << std::endl;
        std::cout << "Turn 2 Tokens generated: " << res2.generated_token_ids.size() << std::endl;
        std::cout << "Turn 2 Finished successfully!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "EXCEPTION: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
