#include "ninfer/engine.h"
#include "../qwen3_6/speculative_page_boundary.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

ninfer::RequestOptions request(std::uint32_t outputs, bool reuse = false) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = outputs;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = reuse;
    options.stop.include_model_defaults       = false;
    return options;
}

void valid(const ninfer::GenerationResult& result, std::size_t outputs) {
    require(result.generated_token_ids.size() == outputs &&
                result.finish_reason == ninfer::FinishReason::OutputLimit,
            "DFlash2 did not honor the requested output budget");
    require(
        result.speculative.backend == ninfer::SpeculativeBackend::DFlash2 &&
            (outputs == 1 || result.speculative.rounds + result.speculative.fallback_steps != 0),
        "generation bypassed DFlash2");
}

ninfer::PromptInput media_prompt(ninfer::MediaKind kind) {
    const std::string header = "P6\n64 64\n255\n";
    ninfer::MessagePart media;
    media.kind              = ninfer::MessagePartKind::Media;
    media.media.kind        = kind;
    media.media.media_type  = "image/x-portable-pixmap";
    media.media.source_name = "pattern.ppm";
    media.media.bytes.assign(header.begin(), header.end());
    for (int i = 0; i < 64 * 64; ++i) {
        media.media.bytes.push_back(i & 255);
        media.media.bytes.push_back((i * 3) & 255);
        media.media.bytes.push_back((i * 7) & 255);
    }
    ninfer::ChatMessage user;
    user.role = ninfer::ChatRole::User;
    user.parts.push_back(std::move(media));
    user.parts.push_back({.kind  = ninfer::MessagePartKind::Text,
                          .text  = "Describe the pattern briefly.",
                          .media = {}});
    ninfer::PromptInput input;
    input.messages.push_back(std::move(user));
    input.options.enable_thinking = false;
    return input;
}

void constrained_batch(ninfer::Engine& engine, const std::vector<ninfer::TokenId>& prompt,
                       const std::vector<ninfer::TokenId>& reference, unsigned batch) {
    // Exercise the production chat boundary. Values occur only in each row's schema, and the
    // second pass changes them while reusing the same prompt/lane. Poisoned unmasked argmax
    // behavior is independently covered by the sparse-acceptance Op oracle.
    ninfer::PromptInput structured_prompt;
    structured_prompt.options.enable_thinking = false;
    ninfer::ChatMessage user;
    user.role = ninfer::ChatRole::User;
    user.parts.push_back({
        .kind = ninfer::MessagePartKind::Text,
        .text = "Return exactly one compact JSON object with the fields lane and status, "
                "following the supplied response schema. Do not include explanations, Markdown, "
                "or leading blank lines.",
    });
    structured_prompt.messages.push_back(std::move(user));
    for (unsigned pass = 0; pass < 4; ++pass) {
        std::vector<ninfer::GenerationHandle> handles;
        for (unsigned row = 0; row < batch; ++row) {
            const bool structured = pass >= 2 || row % 2 == 0;
            auto options = request(structured ? 64 : 24, true);
            if (structured) {
                options.execution.sampling.repetition_penalty = 1.1F;
                options.execution.sampling.presence_penalty = 0.25F;
                const auto value = pass * batch + row;
                const nlohmann::json schema = {
                    {"type", "object"},
                    {"properties", {{"lane", {{"type", "integer"}, {"const", value}}},
                                     {"status", {{"type", "string"}, {"const", "verified"}}}}},
                    {"required", {"lane", "status"}},
                    {"additionalProperties", false},
                };
                options.execution.structured_output =
                    {ninfer::StructuredOutputKind::JsonSchema, schema.dump()};
                options.stop.include_model_defaults = true;
            }
            auto prepared = structured ? engine.prepare(structured_prompt)
                                         : engine.prepare_tokens(prompt);
            handles.push_back(engine.submit(std::move(prepared), options));
        }
        for (unsigned row = 0; row < batch; ++row) {
            const auto result = handles[row].wait();
            if (pass >= 2 || row % 2 == 0) {
                const nlohmann::json expected = {
                    {"lane", pass * batch + row}, {"status", "verified"}};
                try {
                    require(nlohmann::json::parse(result.content) == expected,
                            "DFlash2 constrained lane violated its own schema");
                    require(result.finish_reason == ninfer::FinishReason::StopToken &&
                                result.reasoning.empty(),
                            "DFlash2 non-thinking structured response did not finish normally");
                    require(result.speculative.accepted_tokens == 0,
                            "DFlash2 constrained lane speculated past an unadvanced grammar");
                } catch (const std::exception& error) {
                    const nlohmann::json diagnostic = {
                        {"case", "constrained_batch"},
                        {"pass", pass},
                        {"row", row},
                        {"error", error.what()},
                        {"expected", expected},
                        {"finish_reason", static_cast<unsigned>(result.finish_reason)},
                        {"generated_token_ids", result.generated_token_ids},
                        {"content", result.content},
                        {"reasoning", result.reasoning},
                        {"reused_prompt_tokens", result.reused_prompt_tokens},
                        {"speculative_rounds", result.speculative.rounds},
                        {"drafted_tokens", result.speculative.drafted_tokens},
                        {"accepted_tokens", result.speculative.accepted_tokens},
                    };
                    std::cerr << diagnostic.dump() << '\n';
                    throw;
                }
            } else {
                require(result.generated_token_ids == reference,
                        "constrained neighbor changed an unconstrained target result");
            }
        }
    }
    const auto after = engine.generate(engine.prepare_tokens(prompt), request(24));
    require(after.generated_token_ids == reference,
            "DFlash2 request reused a previous request's grammar mask");
}

class CancelAfterDecode final : public ninfer::OutputSink {
public:
    void start(ninfer::GenerationStart) override {}
    void publish(ninfer::OutputDelta delta) override {
        if (!delta.text.empty() && ++publications_ == 2) { cancelled.store(true); }
    }
    std::atomic<bool> cancelled{false};

private:
    unsigned publications_ = 0;
};

void host_restore(ninfer::Engine& engine) {
    // Two retained sessions with rewrite/endpoint checkpoints exceed the two Device slots.
    // KV pressure alone can leave the StateImage resident and does not test its Host codec.
    ninfer::PromptInput input;
    input.options.enable_thinking   = false;
    input.context_cache.session_key = "dflash2-host-restore";
    input.context_cache.retention   = ninfer::CacheRetentionHint::LiveSession;
    std::string text;
    for (unsigned i = 0; i < 1700; ++i) { text += "alpha "; }
    text += "\nCount from one to twenty: one, two, three,";
    ninfer::ChatMessage user;
    user.role = ninfer::ChatRole::User;
    user.parts.push_back({.kind = ninfer::MessagePartKind::Text, .text = std::move(text)});
    input.messages.push_back(std::move(user));
    const auto retained = engine.generate(engine.prepare(input), request(12, true));
    valid(retained, 12);
    require(retained.prompt.prompt_tokens > engine.options().max_context / 2 &&
                retained.prompt.prompt_tokens + 128 < engine.options().max_context,
            "Host restore fixture must pressure the KV pool and leave continuation room");

    const auto before_pressure = engine.runtime_stats();
    auto competitor = input;
    competitor.context_cache.session_key = "dflash2-host-competitor";
    competitor.messages.front().parts.front().text.insert(0, "A separate session: ");
    const auto other = engine.generate(engine.prepare(competitor), request(12, true));
    valid(other, 12);

    ninfer::ChatMessage assistant;
    assistant.role              = ninfer::ChatRole::Assistant;
    assistant.reasoning_content = retained.reasoning;
    assistant.parts.push_back({.kind = ninfer::MessagePartKind::Text, .text = retained.content});
    input.messages.push_back(std::move(assistant));
    ninfer::ChatMessage follow;
    follow.role = ninfer::ChatRole::User;
    follow.parts.push_back({.kind = ninfer::MessagePartKind::Text,
                            .text = "Continue counting from twenty-one to thirty."});
    input.messages.push_back(std::move(follow));

    const auto after_pressure = engine.runtime_stats();
    require(after_pressure.state_d2h_count > before_pressure.state_d2h_count,
            "DFlash2 Host fixture did not demote a StateImage under pressure");
    const auto restored = engine.generate(engine.prepare(input), request(8, true));
    valid(restored, 8);
    const auto after_restore = engine.runtime_stats();
    require(restored.reused_prompt_tokens != 0 &&
                after_restore.state_h2d_count > after_pressure.state_h2d_count,
            "DFlash2 reuse request did not restore its cyclic StateImage from Host");
    // Compute the reference after the restore: a fresh request may publish a third owner and
    // evict the very two-owner-catalog entry whose Host restore this test needs to observe.
    auto fresh_input = input;
    fresh_input.context_cache = {};
    const auto fresh = engine.generate(engine.prepare(fresh_input), request(8, false));
    valid(fresh, 8);
    require(restored.generated_token_ids == fresh.generated_token_ids,
            "DFlash2 Host-restored target/context state differs from fresh execution");
    std::cout << "host_restore reused=" << restored.reused_prompt_tokens
              << " state_h2d=" << after_restore.state_h2d_count - after_pressure.state_h2d_count
              << " main_h2d=" << after_restore.main_kv_h2d_pages - after_pressure.main_kv_h2d_pages
              << '\n';
}
} // namespace

// Optional K, Graph, optimized-head, B and KV codec arguments select representative integration
// routes without multiplying test binaries. The artifact explicitly selects the storage profile.
int main(int argc, char** argv) {
    const char* artifact = std::getenv("NINFER_QWEN3_8_27B_DFLASH2_WEIGHTS");
    if (!artifact || !*artifact) {
        std::cout << "skip: NINFER_QWEN3_8_27B_DFLASH2_WEIGHTS is not set\n";
        return 77;
    }
    try {
        const auto k         = argc > 1 ? static_cast<unsigned>(std::stoul(argv[1])) : 15U;
        const bool graph     = argc > 2 ? std::stoi(argv[2]) != 0 : true;
        const bool optimized = argc > 3 ? std::stoi(argv[3]) != 0 : true;
        const auto batch     = argc > 4 ? static_cast<unsigned>(std::stoul(argv[4])) : 8U;
        ninfer::EngineOptions options;
        options.artifact_path   = artifact;
        options.max_context     = 2304;
        options.kv_capacity     = ninfer::KvCapacityPolicy::explicit_capacity(2304 * batch);
        options.prefill_chunk   = 2304;
        options.max_concurrency = batch;
        options.context_cache.device_state_slots = argc > 7 ? std::stoul(argv[7]) : 3U;
        options.use_cuda_graph                   = graph;
        options.enable_vision                    = argc > 6 && std::stoi(argv[6]) != 0;
        const bool check_host_restore = *options.context_cache.device_state_slots == 1;
        if (check_host_restore) {
            require(batch == 1, "Host restore qualification requires B=1 and H=1");
            options.context_cache.host_state_slots                  = 2;
            options.context_cache.host_kv_capacity_bytes             = 256ULL << 20;
            options.context_cache.max_private_continuations          = 2;
            options.context_cache.max_shared_prefixes                = 0;
            options.context_cache.max_long_anchors_per_continuation = 0;
        }
        const std::string kv_codec = argc > 5 ? argv[5] : "bf16";
        if (kv_codec == "bf16") {
            options.kv_cache = ninfer::KvCacheStorage::BFloat16;
        } else if (kv_codec == "int8") {
            options.kv_cache = ninfer::KvCacheStorage::Int8Group64;
        } else if (kv_codec == "fp8") {
            options.kv_cache = ninfer::KvCacheStorage::Fp8E4M3Row256;
        } else {
            throw std::invalid_argument("KV codec must be bf16, int8, or fp8");
        }
        std::vector<ninfer::TokenId> prompt, reference, penalty_reference, repetition_reference;
        auto penalty                                 = request(24);
        penalty.execution.sampling.presence_penalty  = 0.5F;
        penalty.execution.sampling.frequency_penalty = 0.25F;
        // The bounded counting segment has stable margins across single-token and
        // speculative target kernels; later free-form prose can cross close logit margins.
        auto repetition = penalty;
        repetition.execution.requested_output_tokens = 16;
        repetition.execution.sampling.repetition_penalty = 1.1F;
        {
            auto ordinary_options            = options;
            ordinary_options.max_concurrency = 1;
            ordinary_options.kv_capacity     = ninfer::KvCapacityPolicy::explicit_capacity(2304);
            ordinary_options.use_cuda_graph  = false;
            ordinary_options.enable_vision   = false;
            ninfer::Engine ordinary(ordinary_options);
            prompt = ordinary.tokenize_text("Count from one to twenty: one, two, three,");
            reference =
                ordinary.generate(ordinary.prepare_tokens(prompt), request(24)).generated_token_ids;
            penalty_reference =
                ordinary.generate(ordinary.prepare_tokens(prompt), penalty).generated_token_ids;
            repetition_reference =
                ordinary.generate(ordinary.prepare_tokens(prompt), repetition).generated_token_ids;
        }
        options.speculative.backend      = ninfer::SpeculativeBackend::DFlash2;
        options.speculative.draft_tokens = k;
        options.speculative.proposal_head =
            optimized ? ninfer::ProposalHead::Optimized : ninfer::ProposalHead::Full;
        ninfer::Engine engine(options);
        if (check_host_restore) { host_restore(engine); }
        const auto first = engine.generate(engine.prepare_tokens(prompt), request(24));
        valid(first, 24);
        require(first.generated_token_ids == reference,
                "DFlash2 greedy target result differs from ordinary decoding");
        require(first.speculative.accepted_tokens != 0, "real draft fixture accepted no proposal");
        const auto penalized = engine.generate(engine.prepare_tokens(prompt), penalty);
        valid(penalized, 24);
        require(penalized.generated_token_ids == penalty_reference,
                "DFlash2 committed token counts differ from ordinary decoding");

        const auto repetition_result = engine.generate(engine.prepare_tokens(prompt), repetition);
        valid(repetition_result, 16);
        require(repetition_result.generated_token_ids == repetition_reference,
                "DFlash2 repetition history differs from ordinary decoding");
        require(!std::equal(repetition_reference.begin(), repetition_reference.end(), reference.begin()),
                "penalized counting fixture did not exercise adjusted token selection");

        // All rows share a known target prefix, while their budgets force P=0, partial and full W.
        std::vector<ninfer::GenerationHandle> handles;
        for (unsigned row = 0; row < batch; ++row) {
            handles.push_back(engine.submit(engine.prepare_tokens(prompt), request(2 + row * 3)));
        }
        for (unsigned row = 0; row < batch; ++row) {
            const auto result = handles[row].wait();
            valid(result, 2 + row * 3);
            require(std::equal(result.generated_token_ids.begin(), result.generated_token_ids.end(),
                               reference.begin()),
                    "compact DFlash2 batch changed a row's target result");
        }
        // Rejection RNG uses the same logical positions and seed for repeat requests.
        auto sampled                                 = request(16);
        sampled.execution.sampling.temperature       = 0.8F;
        sampled.execution.sampling.top_p             = 0.9F;
        sampled.execution.sampling.top_k             = 20;
        sampled.execution.sampling.presence_penalty  = 0.3F;
        sampled.execution.sampling.frequency_penalty = 0.2F;
        sampled.execution.sampling.repetition_penalty = 1.1F;
        sampled.execution.sampling.seed              = 42;
        const auto sample1 = engine.generate(engine.prepare_tokens(prompt), sampled);
        const auto sample2 = engine.generate(engine.prepare_tokens(prompt), sampled);
        valid(sample1, 16);
        require(sample1.generated_token_ids == sample2.generated_token_ids,
                "same DFlash2 seed and inputs did not reproduce the conditional path");

        constrained_batch(engine, prompt, reference, batch);
        CancelAfterDecode sink;
        auto cancel_request = request(128, true);
        const auto cancelled = engine.generate(
            engine.prepare_tokens(prompt), cancel_request, &sink,
            ninfer::CancellationView([&] { return sink.cancelled.load(); }));
        require(sink.cancelled.load() &&
                    cancelled.finish_reason == ninfer::FinishReason::Cancelled,
                "DFlash2 decode cancellation fixture did not cancel after output");
        const auto after_cancel = engine.generate(engine.prepare_tokens(prompt), request(24));
        require(after_cancel.generated_token_ids == reference,
                "cancelled DFlash2 request polluted the next request's target state");

        // Terminal flush and fork: reuse the consumed prefix, then compare with a fresh prefill.
        const auto retained = engine.generate(engine.prepare_tokens(prompt), request(12, true));
        auto continuation   = prompt;
        continuation.insert(continuation.end(), retained.generated_token_ids.begin(),
                            retained.generated_token_ids.end());
        continuation.push_back(198);
        auto reuse_penalty = request(8, true);
        reuse_penalty.execution.sampling.repetition_penalty = 1.1F;
        const auto reused = engine.generate(engine.prepare_tokens(continuation), reuse_penalty);
        reuse_penalty.execution.allow_prefix_reuse = false;
        const auto fresh = engine.generate(engine.prepare_tokens(continuation), reuse_penalty);
        require(reused.reused_prompt_tokens != 0 &&
                    reused.generated_token_ids == fresh.generated_token_ids,
                "DFlash2 prefix restore did not preserve target and local context state");

        if (k >= 7) {
            bool checked_partial = false;
            for (std::size_t i = 1; i < std::min<std::size_t>(k, reference.size()); ++i) {
                if (std::find(reference.begin(), reference.begin() + i, reference[i]) !=
                    reference.begin() + i) {
                    continue;
                }
                auto stopped_options = request(24, true);
                stopped_options.stop.token_ids.push_back(reference[i]);
                const auto stopped =
                    engine.generate(engine.prepare_tokens(prompt), stopped_options);
                const auto licensed = 1 + stopped.speculative.rounds +
                                      stopped.speculative.accepted_tokens +
                                      stopped.speculative.fallback_steps;
                if (stopped.generated_token_ids.size() >= licensed) { continue; }
                require(stopped.finish_reason == ninfer::FinishReason::StopToken,
                        "partial terminal did not stop at its token");
                auto follow = prompt;
                follow.insert(follow.end(), stopped.generated_token_ids.begin(),
                              stopped.generated_token_ids.end());
                follow.push_back(198);
                const auto reused_stop =
                    engine.generate(engine.prepare_tokens(follow), request(8, true));
                const auto fresh_stop =
                    engine.generate(engine.prepare_tokens(follow), request(8, false));
                require(reused_stop.reused_prompt_tokens != 0 &&
                            reused_stop.generated_token_ids == fresh_stop.generated_token_ids,
                        "partial terminal folded the wrong GDN/hidden/context prefix");
                checked_partial = true;
                break;
            }
            require(checked_partial, "fixture did not exercise a stop within a licensed block");
        }
        if (options.enable_vision) {
            for (const auto kind : {ninfer::MediaKind::Image, ninfer::MediaKind::Video}) {
                const auto image =
                    engine.generate(engine.prepare(media_prompt(kind)), request(8, true));
                const auto reused_image =
                    engine.generate(engine.prepare(media_prompt(kind)), request(8, true));
                valid(image, 8);
                require(image.prompt.has_media && image.timings.vision_seconds > 0 &&
                            reused_image.reused_prompt_tokens != 0 &&
                            image.generated_token_ids == reused_image.generated_token_ids,
                        "Vision DFlash2 capture/restore changed the result");
            }
        }
        if (k >= 7) {
            require(prompt.size() < 2044, "ring fixture suffix is too long");
            // E starts at 2044; 24 outputs commit through E=2067, forcing decode context
            // appends across the physical 2048-slot boundary. Oversized prefill replacement is
            // a separate route and therefore uses a second, 2100-token prompt.
            for (const unsigned length : {2044U, 2100U}) {
                auto long_prompt = std::vector<ninfer::TokenId>(length - prompt.size(), 198);
                long_prompt.insert(long_prompt.end(), prompt.begin(), prompt.end());
                const auto long_run =
                    engine.generate(engine.prepare_tokens(long_prompt), request(24, true));
                valid(long_run, 24);
                require(long_run.prompt.prompt_tokens == length &&
                            long_run.prompt.prompt_tokens + long_run.generated_token_ids.size() - 1 > 2048,
                        "DFlash2 ring fixture did not commit beyond the physical boundary");
                long_prompt.insert(long_prompt.end(), long_run.generated_token_ids.begin(),
                                   long_run.generated_token_ids.end());
                const auto long_reuse =
                    engine.generate(engine.prepare_tokens(long_prompt), request(6, true));
                const auto long_fresh =
                    engine.generate(engine.prepare_tokens(long_prompt), request(6, false));
                require(long_reuse.reused_prompt_tokens > 2048 &&
                            long_reuse.generated_token_ids == long_fresh.generated_token_ids,
                        "DFlash2 wrapped/replaced cyclic state changed the retained-prefix result");
            }
        }
        if (k >= 7) {
            auto tail_prompt   = std::vector<ninfer::TokenId>(options.max_context - 4, 198);
            tail_prompt.back() = prompt.back();
            const auto tail    = engine.generate(engine.prepare_tokens(tail_prompt), request(9));
            require(tail.finish_reason == ninfer::FinishReason::ContextCapacity &&
                        tail.generated_token_ids.size() == 5,
                    "full proposal window escaped the target context capacity tail");
        }
        ninfer::test::speculative_page_boundary(engine);
        const auto stats = engine.runtime_stats();
        if (batch > 1) {
            require(stats.state_forks != 0,
                    "concurrent prefix fixture did not exercise StateImage fork admission");
        }
        require(stats.device_backend_kv_occupied_pages == 0 && stats.backend_kv_d2h_bytes == 0 &&
                    stats.backend_kv_h2d_bytes == 0,
                "DFlash2 allocated or transferred a full backend KV pool");
        std::cout << "ok K=" << k << " B=" << batch << " graph=" << graph
                  << " optimized=" << optimized << " accepted=" << first.speculative.accepted_tokens
                  << "/" << first.speculative.drafted_tokens
                  << " state_d2h=" << stats.state_d2h_count
                  << " state_h2d=" << stats.state_h2d_count << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
