#pragma once

#include "ninfer/types.h"

#include <chrono>
#include <functional>
#include <optional>
#include <memory>
#include <string_view>
#include <vector>

namespace ninfer {

class PreparedPrompt {
public:
    PreparedPrompt() noexcept;
    ~PreparedPrompt();

    PreparedPrompt(PreparedPrompt&&) noexcept;
    PreparedPrompt& operator=(PreparedPrompt&&) noexcept;

    PreparedPrompt(const PreparedPrompt&)            = delete;
    PreparedPrompt& operator=(const PreparedPrompt&) = delete;

    [[nodiscard]] const PromptSummary& summary() const noexcept;
    [[nodiscard]] const PromptPreparationStats& preparation_stats() const noexcept;
    [[nodiscard]] explicit operator bool() const noexcept;

private:
    class Impl;
    explicit PreparedPrompt(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;

    friend class Engine;
};

class GenerationHandle {
public:
    GenerationHandle() noexcept;
    ~GenerationHandle();

    GenerationHandle(GenerationHandle&&) noexcept;
    GenerationHandle& operator=(GenerationHandle&&) noexcept;

    GenerationHandle(const GenerationHandle&)            = delete;
    GenerationHandle& operator=(const GenerationHandle&) = delete;

    [[nodiscard]] explicit operator bool() const noexcept;
    [[nodiscard]] const ResolvedSamplingParameters& resolved_sampling() const noexcept;

    GenerationResult wait(OutputSink* sink = nullptr, const CancellationView& cancellation = {});

private:
    class Impl;
    explicit GenerationHandle(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;

    friend class Engine;
};

class Engine {
public:
    explicit Engine(EngineOptions options);
    ~Engine();

    Engine(Engine&&) noexcept;
    Engine& operator=(Engine&&) noexcept;

    Engine(const Engine&)            = delete;
    Engine& operator=(const Engine&) = delete;

    [[nodiscard]] PreparedPrompt prepare(PromptInput input,
                                         const PreparationControl& control = {}) const;

    // Raw token input is retained for repeatable correctness and performance measurement.
    [[nodiscard]] PreparedPrompt prepare_tokens(std::vector<TokenId> token_ids,
                                                bool allow_prefix_identity = true) const;

    // Artifact-tokenizer raw-text encoding. No chat template or implicit special token is added.
    [[nodiscard]] std::vector<TokenId> tokenize_text(std::string_view text) const;
    // Exact bytes one token id decodes to, for reporting TokenLogprobs entries.
    [[nodiscard]] std::string token_bytes(TokenId token) const;

    // Returns log p(tokens[i] | tokens[0..i)) for i in [first_target,tokens.size()).
    [[nodiscard]] std::vector<float> score_tokens(std::vector<TokenId> tokens,
                                                  std::uint32_t first_target);

    [[nodiscard]] std::uint32_t count_tokens(PromptInput input,
                                             const PreparationControl& control = {}) const;
    [[nodiscard]] PromptCapabilities prompt_capabilities() const;
    [[nodiscard]] ModelSamplingDefaults sampling_defaults() const;

    // Establishes queue membership synchronously with a fixed output consumer mode. Destroying an
    // unconsumed handle cancels its request; wait() owns result consumption and may run
    // independently from GPU execution. Streaming mode requires a non-null sink in wait() and
    // publishes one exact GenerationStart before output deltas; Aggregate mode requires a null
    // sink.
    [[nodiscard]] GenerationHandle
    submit(PreparedPrompt prompt, RequestOptions options,
           OutputConsumerMode consumer_mode                       = OutputConsumerMode::Aggregate,
           std::chrono::steady_clock::time_point pending_deadline = {});

    GenerationResult generate(PreparedPrompt prompt, RequestOptions options,
                              OutputSink* sink                     = nullptr,
                              const CancellationView& cancellation = {});

    [[nodiscard]] const EngineOptions& options() const;
    [[nodiscard]] LoadSummary load_summary() const;
    [[nodiscard]] MemorySummary memory_summary() const;
    // Non-blocking variant for telemetry: returns nothing while the engine is
    // mid-execution instead of waiting for it. Anything polled on a timer must
    // use this -- the blocking form queues behind execution and will exhaust a
    // server's thread pool under load.
    [[nodiscard]] std::optional<MemorySummary> try_memory_summary() const;
    [[nodiscard]] RuntimeStats runtime_stats() const;
    [[nodiscard]] MediaCacheSummary media_cache_summary() const;
    void reset_memory_peaks() noexcept;

    // Result of holding the engine still for a moment. Reported rather than
    // returned as a bare duration because the interesting question when the
    // engine gives memory back is not "did it work" but "what did it cost the
    // requests that were waiting".
    struct QuiescenceReport {
        std::chrono::nanoseconds drain;   // request accepted -> engine held still
        std::chrono::nanoseconds work;    // how long the held-still work took
        std::size_t requests_held = 0;    // waiting requests carried across, none dropped
    };

    // Runs `work` at a boundary where no execution unit is in flight, no lane is
    // active and no context transaction is open. Admission is refused from the
    // call until the work has run, so active lanes drain; requests arriving in
    // that window WAIT and are admitted afterwards with their deadlines extended
    // by the time the hold took, so the engine pausing is never itself the reason
    // a request times out.
    //
    // This is the safety fence for changing physical KV residency. It blocks the
    // caller until the work has run. An exception from `work` propagates here and
    // does not disturb the engine: a capacity change that cannot proceed is a
    // refusal, not a failed engine.
    QuiescenceReport run_at_quiescence(std::function<void()> work);

private:
    class Impl;
    std::shared_ptr<Impl> impl_;
};

} // namespace ninfer
