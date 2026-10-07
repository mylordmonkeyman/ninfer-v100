#pragma once
#include "compare_telemetry.h"

namespace v100_compare {
// Generated trace labels are process-local correlation, never external request IDs.
inline std::atomic<std::uint64_t> caller_sequence{0};
inline thread_local std::string_view linked_round_id;
class RoundLink {
public:
    explicit RoundLink(const char* engine) : previous_(linked_round_id) {
        if (level()) {
            id_ = std::string(engine) + ":caller:" + std::to_string(caller_sequence.fetch_add(1));
            linked_round_id = id_;
            linked_ = true;
        }
    }
    RoundLink(const RoundLink&) = delete;
    RoundLink& operator=(const RoundLink&) = delete;
    ~RoundLink() { finish(); }
    void finish() noexcept { if (linked_) { linked_round_id = previous_; linked_ = false; } }
    const std::string& id() const noexcept { return id_; }
private:
    std::string id_;
    std::string_view previous_;
    bool linked_ = false;
};
inline std::string sequence_trace(const char* engine) {
    return std::string(engine) + ":sequence:" + std::to_string(caller_sequence.fetch_add(1));
}
// One actual draft call, independent of the future verifier's selected proposals.
// Host duration includes existing native waits, never introduces CUDA work.
class DraftCall {
public:
    DraftCall(const char* engine, std::string sequence, std::uint64_t anchor,
              std::string after_round = {})
        : engine_(engine), sequence_(std::move(sequence)),
          after_round_(std::move(after_round)), anchor_(anchor), enabled_(level() != 0) {
        if (enabled_) {
            id_ = std::string(engine) + ":draft:" + std::to_string(caller_sequence.fetch_add(1));
            if (level() >= 2) start_ = Clock::now();
        }
    }
    DraftCall(const DraftCall&) = delete;
    DraftCall& operator=(const DraftCall&) = delete;
    template<class Token> void complete(std::span<const Token> tokens) {
        if (!enabled_) return;
        if (level() >= 2) duration_us_ = elapsed_us(start_);
        produced_ = tokens.size();
        if (level() >= 2) ids_.emplace(tokens.begin(), tokens.end());
        ok_ = true;
    }
    ~DraftCall() noexcept {
        if (!enabled_) return;
        try {
            if (level() >= 2 && !duration_us_) duration_us_ = elapsed_us(start_);
            const char* run = std::getenv("V100_COMPARE_RUN_ID");
            std::ostringstream out;
            out << "{\"schema\":\"ninfer-strata-v100-draft-v1\",\"kind\":\"draft\",\"engine\":"
                << quote(engine_) << ",\"run_id\":" << quote(run ? run : "unassigned")
                << ",\"level\":" << level() << ",\"draft_id\":" << quote(id_)
                << ",\"sequence_trace_id\":" << quote(sequence_)
                << ",\"first_token_index\":" << anchor_ << ",\"proposal_source\":\"mtp\""
                << ",\"status\":" << quote(ok_ ? "ok" : "failed");
            if (!after_round_.empty()) out << ",\"after_round_id\":" << quote(after_round_);
            if (produced_) out << ",\"produced_drafts\":" << *produced_;
            if (duration_us_) out << ",\"host_call_us\":" << *duration_us_;
            if (ids_) {
                out << ",\"token_ids\":["; bool sep = false;
                for (auto id : *ids_) { if (sep) out << ','; sep = true; out << id; }
                out << ']';
            }
            out << "}\n"; const auto line = out.str();
            std::fwrite(line.data(), 1, line.size(), stderr);
        } catch (...) { std::fputs("v100 draft serialization failed\n", stderr); }
    }
private:
    const char* engine_;
    std::string sequence_, after_round_, id_;
    std::uint64_t anchor_;
    bool enabled_, ok_ = false;
    Clock::time_point start_{};
    std::optional<double> duration_us_;
    std::optional<std::uint64_t> produced_;
    std::optional<std::vector<std::int64_t>> ids_;
};
struct LifecycleEvent {
    const char* engine;
    std::string round_id, sequence_trace_id;
    const char* event; // verified, committed, emitted, aborted
    std::uint64_t first_token_index;
    std::optional<unsigned> lane{};
    std::optional<std::uint64_t> epoch{};
    std::optional<std::uint64_t> proposed_drafts{}, accepted_drafts{}, verified_tokens{}, token_count{};
    const char* token_semantics = "unknown";
    const char* proposal_source = "unknown";
    std::optional<std::vector<std::int64_t>> token_ids{};
    template<class Token> void tokens(std::span<const Token> values) {
        token_count = values.size();
        if (level() >= 2) token_ids.emplace(values.begin(), values.end());
    }
    std::string json() const {
        const char* run = std::getenv("V100_COMPARE_RUN_ID");
        std::ostringstream out;
        out << "{\"schema\":\"ninfer-strata-v100-lifecycle-v1\",\"kind\":\"lifecycle\",\"engine\":"
            << quote(engine) << ",\"run_id\":" << quote(run ? run : "unassigned")
            << ",\"round_id\":" << quote(round_id) << ",\"sequence_trace_id\":" << quote(sequence_trace_id)
            << ",\"event\":" << quote(event) << ",\"level\":" << level()
            << ",\"first_token_index\":" << first_token_index
            << ",\"token_semantics\":" << quote(token_semantics)
            << ",\"proposal_source\":" << quote(proposal_source);
        const auto field = [&](const char* key, const auto& value) {
            if (value) out << ',' << quote(key) << ':' << *value;
        };
        field("lane", lane); field("epoch", epoch);
        field("proposed_drafts", proposed_drafts); field("accepted_drafts", accepted_drafts);
        field("verified_tokens", verified_tokens); field("token_count", token_count);
        if (token_ids) {
            out << ",\"token_ids\":["; bool sep = false;
            for (auto id : *token_ids) { if (sep) out << ','; sep = true; out << id; }
            out << ']';
        }
        out << '}'; return out.str();
    }
    void emit() const noexcept {
        try { const auto line = json() + '\n'; std::fwrite(line.data(), 1, line.size(), stderr); }
        catch (...) { std::fputs("v100 lifecycle serialization failed\n", stderr); }
    }
};
} // namespace v100_compare
