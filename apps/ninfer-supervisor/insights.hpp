#pragma once

#include "logic.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ninfer::supervisor {

inline nlohmann::json insight_unavailable(std::string id, std::string title, std::string statement,
                                          nlohmann::json evidence = nlohmann::json::object(),
                                          nlohmann::json measured_over = {{"requests", 0}}) {
    return {{"id", std::move(id)},
            {"severity", "notice"},
            {"title", std::move(title)},
            {"statement", std::move(statement)},
            {"evidence", std::move(evidence)},
            {"confidence", "measured"},
            {"measured_over", std::move(measured_over)},
            {"availability", "unavailable"}};
}

inline nlohmann::json insight_available(std::string id, std::string severity, std::string title,
                                        std::string statement, nlohmann::json evidence,
                                        std::string recommendation, std::string confidence,
                                        nlohmann::json measured_over) {
    nlohmann::json out = {{"id", std::move(id)},
                          {"severity", std::move(severity)},
                          {"title", std::move(title)},
                          {"statement", std::move(statement)},
                          {"evidence", std::move(evidence)},
                          {"confidence", std::move(confidence)},
                          {"measured_over", std::move(measured_over)},
                          {"availability", "available"}};
    if (!recommendation.empty()) { out["recommendation"] = std::move(recommendation); }
    return out;
}

inline std::int64_t json_i64(const nlohmann::json& j, const char* key, std::int64_t fallback = 0) {
    if (!j.contains(key)) { return fallback; }
    const auto& v = j.at(key);
    if (v.is_number_integer()) { return v.get<std::int64_t>(); }
    if (v.is_number()) { return static_cast<std::int64_t>(v.get<double>()); }
    return fallback;
}

inline double json_f64(const nlohmann::json& j, const char* key, double fallback = 0) {
    if (!j.contains(key)) { return fallback; }
    const auto& v = j.at(key);
    if (v.is_number()) { return v.get<double>(); }
    return fallback;
}

// Analyze a JSONL blob. Does not invent content/reasoning_content — those keys are
// not written by the engine request log (schema_version 10).
inline nlohmann::json analyze_request_log_jsonl(std::string_view jsonl, std::string_view path) {
    nlohmann::json report = {
        {"source",
         {{"request_log", jsonl.empty() ? "empty" : "ok"}, {"path", std::string(path)}}},
        {"insights", nlohmann::json::array()},
    };

    std::unordered_map<std::string, nlohmann::json> starts;
    std::unordered_map<std::string, std::uint32_t> host_state_capacities;
    std::string latest_server_instance;
    std::int64_t latest_server_start = 0;
    nlohmann::json latest_engine     = nlohmann::json::object();
    std::vector<nlohmann::json> dones;
    std::vector<nlohmann::json> throughputs;
    std::vector<nlohmann::json> errors;
    std::int64_t tmin = 0;
    std::int64_t tmax = 0;
    int parsed        = 0;

    std::string line;
    std::istringstream in{std::string(jsonl)};
    while (std::getline(in, line)) {
        if (line.empty()) { continue; }
        nlohmann::json j;
        try {
            j = nlohmann::json::parse(line);
        } catch (...) { continue; }
        const std::string event = j.value("event", "");
        if (event.empty()) { continue; }
        ++parsed;
        const auto ts = json_i64(j, "timestamp_unix_ms");
        if (tmin == 0 || ts < tmin) { tmin = ts; }
        if (ts > tmax) { tmax = ts; }
        if (event == "server_start" && j.contains("engine")) {
            const std::string instance = j.value("server_instance_id", "");
            const auto& engine         = j.at("engine");
            if (!instance.empty() && ts >= latest_server_start) {
                latest_server_instance = instance;
                latest_server_start    = ts;
                latest_engine          = engine;
            }
            if (!instance.empty() && engine.contains("context_cache")) {
                const auto capacity = json_i64(engine.at("context_cache"), "host_state_slots");
                if (capacity > 0 &&
                    capacity <= static_cast<std::int64_t>(
                                    std::numeric_limits<std::uint32_t>::max())) {
                    host_state_capacities[instance] = static_cast<std::uint32_t>(capacity);
                }
            }
        } else if (event == "request_start" && j.contains("request")) {
            const auto rid = json_i64(j.at("request"), "request_id");
            starts[j.value("server_instance_id", "") + ":" + std::to_string(rid)] = j;
        } else if (event == "request_done") {
            dones.push_back(std::move(j));
        } else if (event == "throughput") {
            throughputs.push_back(std::move(j));
        } else if (event == "request_error") {
            errors.push_back(std::move(j));
        }
    }

    auto& insights = report["insights"];
    const double window_s =
        (tmax > tmin) ? static_cast<double>(tmax - tmin) / 1000.0 : 0.0;

    if (parsed == 0) {
        insights.push_back(insight_unavailable(
            "source.request_log", "Request log has no usable records",
            "no request_done records in window",
            {{"path", std::string(path)}, {"parsed_events", 0}}));
        report["generated_note"] = "unavailable is not a clean zero";
        return report;
    }

    if (dones.empty()) {
        insights.push_back(insight_unavailable(
            "source.request_done", "No completed requests in window",
            "no request_done records in window",
            {{"parsed_events", parsed}, {"request_start", starts.size()}},
            {{"requests", 0}, {"parsed_events", parsed}}));
        return report;
    }

    struct Bucket {
        int queued = 0;
        int prefill = 0;
        int decode  = 0;
        int mixed   = 0;
        int unpaired = 0;
        double queue_wait_sum = 0;
        double prepare_sum    = 0;
        double prefill_sum    = 0;
        double decode_sum     = 0;
        double vision_sum     = 0;
        double ttft_sum       = 0;
        double total_sum      = 0;
        std::vector<std::int64_t> queued_ids;
        std::vector<std::int64_t> prefill_ids;
        std::vector<std::int64_t> decode_ids;
    } b;

    int output_limit_thinking = 0;
    int output_limit_hit_cap  = 0;
    int thinking_requests     = 0;
    int tools_declared        = 0;
    std::vector<std::int64_t> output_limit_ids;
    std::vector<int> output_limit_caps;
    int reuse_reset_single = 0;
    int reuse_reset_multi  = 0;
    int reuse_restore      = 0;
    int reuse_seed         = 0;
    int reuse_append       = 0;
    int reuse_other        = 0;
    std::uint64_t multi_prompt_tokens = 0;
    std::uint64_t multi_hit_tokens    = 0;
    std::vector<nlohmann::json> reset_multi_samples;

    // Speculative decoding, as the engine writes it per request_done: backend, draft_window,
    // rounds, drafted/accepted tokens, fallback_steps and accepted_per_position, where position
    // p counts the draft rounds in which the p-th drafted token was accepted. Flash-Next counts a
    // fallback step as a round too and the 27B path does not, and a round near the output limit
    // drafts fewer than draft_window tokens, so neither "rounds" nor drafted/draft_window is the
    // number of rounds that drafted. max(rounds - fallback, ceil(drafted / window)) is exact on
    // Flash-Next and a lower bound on the 27B path; per-position rates are clamped to 1.
    // The backend and window are launch configuration, so only the latest server instance is
    // measured: a whole-file sum would blend windows 4, 5 and 7 from different launches into one
    // curve that belongs to none of them.
    struct SpecPosition {
        std::uint64_t accepted = 0;
        std::uint64_t rounds   = 0;
    };
    struct Spec {
        std::string backend;
        int draft_window                = 0;
        int requests_with_telemetry     = 0;
        int requests_with_drafts        = 0;
        int requests_backend_none       = 0;
        int requests_other_instance     = 0;
        std::uint64_t rounds_reported   = 0;
        std::uint64_t draft_rounds      = 0;
        std::uint64_t drafted           = 0;
        std::uint64_t accepted          = 0;
        std::uint64_t fallback          = 0;
        std::uint64_t first_half_steps  = 0;
        std::uint64_t first_half_fallback  = 0;
        std::uint64_t second_half_steps = 0;
        std::uint64_t second_half_fallback = 0;
        std::vector<SpecPosition> positions;
        std::vector<std::int64_t> sample_ids;
        std::vector<nlohmann::json> low_acceptance_samples;
        std::vector<nlohmann::json> high_fallback_samples;
        std::map<int, int> windows_seen;
        std::int64_t tmin = 0;
        std::int64_t tmax = 0;
    } spec;
    std::size_t spec_scope_dones = 0;
    for (const auto& done : dones) {
        if (latest_server_instance.empty() ||
            done.value("server_instance_id", "") == latest_server_instance) {
            ++spec_scope_dones;
        }
    }
    std::size_t spec_scope_index = 0;

    for (const auto& done : dones) {
        const auto& req = done.contains("request") ? done.at("request") : nlohmann::json::object();
        const auto id   = json_i64(req, "request_id");
        if (req.value("enable_thinking", false)) { ++thinking_requests; }
        if (json_i64(req, "tool_count") > 0) { ++tools_declared; }
        const auto& result = done.contains("result") ? done.at("result") : nlohmann::json::object();
        const std::string finish = result.value("finish_reason", "");
        const int cap            = static_cast<int>(json_i64(req, "requested_output_tokens"));
        const int completion     = static_cast<int>(json_i64(result, "completion_tokens"));
        if (finish == "output_limit" && req.value("enable_thinking", false)) {
            ++output_limit_thinking;
            output_limit_ids.push_back(id);
            output_limit_caps.push_back(cap);
            if (cap > 0 && completion >= cap) { ++output_limit_hit_cap; }
        }
        const int messages        = static_cast<int>(json_i64(req, "message_count"));
        const std::string reuse   = result.value("prefix_reuse_path", "");
        const auto prompt_tokens  = static_cast<std::uint64_t>(json_i64(result, "prompt_tokens"));
        const auto hit_tokens     = static_cast<std::uint64_t>(json_i64(result, "prefix_cache_hit_tokens"));
        const bool multiturn      = messages >= 2;
        if (multiturn) {
            multi_prompt_tokens += prompt_tokens;
            multi_hit_tokens += hit_tokens;
        }
        if (reuse == "full_reset") {
            if (multiturn) {
                ++reuse_reset_multi;
                if (reset_multi_samples.size() < 8) {
                    reset_multi_samples.push_back({{"request_id", id},
                                                   {"message_count", messages},
                                                   {"prompt_tokens", prompt_tokens},
                                                   {"prefix_cache_hit_tokens", hit_tokens}});
                }
            } else {
                ++reuse_reset_single;
            }
        } else if (reuse.find("restore") != std::string::npos) {
            ++reuse_restore;
        } else if (reuse.find("seed") != std::string::npos) {
            ++reuse_seed;
        } else if (reuse.find("append") != std::string::npos) {
            ++reuse_append;
        } else if (!reuse.empty()) {
            ++reuse_other;
        }

        if (!latest_server_instance.empty() &&
            done.value("server_instance_id", "") != latest_server_instance) {
            ++spec.requests_other_instance;
        } else if (done.contains("speculative") && done.at("speculative").is_object()) {
            const std::size_t scope_index = spec_scope_index++;
            const auto& sp             = done.at("speculative");
            const auto done_ts         = json_i64(done, "timestamp_unix_ms");
            if (spec.tmin == 0 || done_ts < spec.tmin) { spec.tmin = done_ts; }
            if (done_ts > spec.tmax) { spec.tmax = done_ts; }
            const std::string backend  = sp.value("backend", "");
            const int window           = static_cast<int>(json_i64(sp, "draft_window"));
            if (backend.empty() || backend == "none" || window <= 0) {
                ++spec.requests_backend_none;
            } else {
                ++spec.requests_with_telemetry;
                spec.backend      = backend;
                spec.draft_window = std::max(spec.draft_window, window);
                ++spec.windows_seen[window];
                auto count = [&](const char* key) {
                    return static_cast<std::uint64_t>(std::max<std::int64_t>(0, json_i64(sp, key)));
                };
                const auto drafted       = count("drafted_tokens");
                const auto accepted      = count("accepted_tokens");
                const auto fallback      = count("fallback_steps");
                const auto rounds        = count("rounds");
                const auto full_windows  = (drafted + static_cast<std::uint64_t>(window) - 1) /
                                          static_cast<std::uint64_t>(window);
                const auto draft_rounds  = std::max(rounds > fallback ? rounds - fallback : std::uint64_t{0}, full_windows);
                spec.rounds_reported += rounds;
                spec.draft_rounds += draft_rounds;
                spec.drafted += drafted;
                spec.accepted += accepted;
                spec.fallback += fallback;
                const bool second_half = scope_index * 2 >= spec_scope_dones;
                (second_half ? spec.second_half_steps : spec.first_half_steps) += draft_rounds + fallback;
                (second_half ? spec.second_half_fallback : spec.first_half_fallback) += fallback;
                if (sp.contains("accepted_per_position") && sp.at("accepted_per_position").is_array()) {
                    const auto& pos = sp.at("accepted_per_position");
                    if (spec.positions.size() < pos.size()) { spec.positions.resize(pos.size()); }
                    for (std::size_t p = 0; p < pos.size(); ++p) {
                        if (!pos.at(p).is_number()) { continue; }
                        spec.positions[p].accepted += pos.at(p).get<std::uint64_t>();
                        if (static_cast<int>(p) < window) { spec.positions[p].rounds += draft_rounds; }
                    }
                }
                if (fallback >= 8 && fallback * 2 >= draft_rounds + fallback &&
                    spec.high_fallback_samples.size() < 8) {
                    spec.high_fallback_samples.push_back(
                        {{"request_id", id},
                         {"fallback_steps", fallback},
                         {"draft_rounds", draft_rounds},
                         {"completion_tokens", completion},
                         {"message_count", messages},
                         {"media_item_count", json_i64(req, "media_item_count")},
                         {"enable_thinking", req.value("enable_thinking", false)}});
                }
                if (drafted > 0) {
                    ++spec.requests_with_drafts;
                    if (spec.sample_ids.size() < 8) { spec.sample_ids.push_back(id); }
                    const double ratio = static_cast<double>(accepted) / static_cast<double>(drafted);
                    if (draft_rounds >= 8 && ratio < 0.25 && spec.low_acceptance_samples.size() < 8) {
                        spec.low_acceptance_samples.push_back({{"request_id", id},
                                                               {"drafted_tokens", drafted},
                                                               {"accepted_tokens", accepted},
                                                               {"fallback_steps", fallback},
                                                               {"completion_tokens", completion}});
                    }
                }
            }
        }

        const auto& timings =
            done.contains("timings_seconds") ? done.at("timings_seconds") : nlohmann::json::object();
        const double total   = json_f64(timings, "total");
        const double prepare = json_f64(timings, "prepare");
        const double prefill = json_f64(timings, "prefill");
        const double decode  = json_f64(timings, "decode");
        const double vision  = json_f64(timings, "vision");
        const double ttft    = json_f64(timings, "ttft");
        b.prepare_sum += prepare;
        b.prefill_sum += prefill;
        b.decode_sum += decode;
        b.vision_sum += vision;
        b.ttft_sum += ttft;
        b.total_sum += total;

        const std::string join =
            done.value("server_instance_id", "") + ":" + std::to_string(id);
        auto it = starts.find(join);
        if (it == starts.end()) {
            ++b.unpaired;
            continue;
        }
        const double wall_s =
            static_cast<double>(json_i64(done, "timestamp_unix_ms") -
                                json_i64(it->second, "timestamp_unix_ms")) /
            1000.0;
        double queue_wait = wall_s - total;
        if (queue_wait < 0.0) { queue_wait = 0.0; }
        b.queue_wait_sum += queue_wait;
        const bool queued = queue_wait >= 0.020 && wall_s > 0.0 && queue_wait >= 0.25 * wall_s;
        if (queued) {
            ++b.queued;
            if (b.queued_ids.size() < 8) { b.queued_ids.push_back(id); }
        } else if (total > 0.0 && prefill >= decode && prefill >= 0.4 * total) {
            ++b.prefill;
            if (b.prefill_ids.size() < 8) { b.prefill_ids.push_back(id); }
        } else if (total > 0.0 && decode >= 0.4 * total) {
            ++b.decode;
            if (b.decode_ids.size() < 8) { b.decode_ids.push_back(id); }
        } else {
            ++b.mixed;
        }
    }

    const int paired = static_cast<int>(dones.size()) - b.unpaired;
    int max_waiting  = 0;
    int max_running  = 0;
    int max_prefill  = 0;
    for (const auto& tp : throughputs) {
        if (!tp.contains("scheduler")) { continue; }
        const auto& sch = tp.at("scheduler");
        max_waiting     = std::max(max_waiting, static_cast<int>(json_i64(sch, "waiting")));
        max_running     = std::max(max_running, static_cast<int>(json_i64(sch, "running")));
        max_prefill     = std::max(max_prefill, static_cast<int>(json_i64(sch, "prefilling")));
    }

    const auto over = nlohmann::json{{"requests", dones.size()},
                                     {"paired_requests", paired},
                                     {"unpaired_done", b.unpaired},
                                     {"window_s", window_s},
                                     {"throughput_events", throughputs.size()}};

    // A full Host State pool is not itself a fault: pressure can legitimately fill it briefly.
    // The production failure in #15 is the conjunction of a pool that remains full and later
    // multi-turn requests that all fall to Root with no cache hit. Keep the two signals joined so
    // ordinary first-turn Root selections never become a false alarm.
    for (const auto& [instance, capacity] : host_state_capacities) {
        if (instance != latest_server_instance) { continue; }
        std::size_t saturated_samples = 0;
        std::int64_t saturated_since  = 0;
        std::int64_t saturated_until  = 0;
        for (auto it = throughputs.rbegin(); it != throughputs.rend(); ++it) {
            if (it->value("server_instance_id", "") != instance) { continue; }
            const auto& cache = it->contains("context_cache") ? it->at("context_cache")
                                                              : nlohmann::json::object();
            const auto& occupancy = cache.contains("occupancy") ? cache.at("occupancy")
                                                                 : nlohmann::json::object();
            if (json_i64(occupancy, "host_state_slots", -1) != capacity) { break; }
            const std::int64_t timestamp = json_i64(*it, "timestamp_unix_ms");
            if (saturated_until == 0) { saturated_until = timestamp; }
            saturated_since = timestamp;
            ++saturated_samples;
        }
        if (saturated_samples < 3 || saturated_until - saturated_since < 60'000) { continue; }

        std::size_t root_streak = 0;
        std::uint64_t recomputed_tokens = 0;
        std::vector<std::int64_t> sample_ids;
        for (auto it = dones.rbegin(); it != dones.rend(); ++it) {
            if (it->value("server_instance_id", "") != instance ||
                json_i64(*it, "timestamp_unix_ms") < saturated_since) {
                continue;
            }
            const auto& request = it->contains("request") ? it->at("request")
                                                           : nlohmann::json::object();
            if (json_i64(request, "message_count") < 2) { continue; }
            const auto& result = it->contains("result") ? it->at("result")
                                                         : nlohmann::json::object();
            if (result.value("prefix_reuse_path", "") != "root" ||
                json_i64(result, "prefix_cache_hit_tokens", -1) != 0) {
                break;
            }
            ++root_streak;
            recomputed_tokens += static_cast<std::uint64_t>(
                std::max<std::int64_t>(0, json_i64(result, "computed_prefill_tokens")));
            if (sample_ids.size() < 8) {
                sample_ids.push_back(json_i64(request, "request_id"));
            }
        }
        if (root_streak < 3) { continue; }

        std::ostringstream statement;
        statement << "Host State occupancy remained at " << capacity << "/" << capacity
                  << " for " << saturated_samples << " consecutive throughput samples over "
                  << static_cast<double>(saturated_until - saturated_since) / 1000.0
                  << " s, while the latest " << root_streak
                  << " multi-turn requests all selected root with zero prefix-cache hits.";
        insights.push_back(insight_available(
            "prefix.reuse_collapsed", "warning", "Prefix reuse has collapsed", statement.str(),
            {{"server_instance_id", instance},
             {"host_state_slots", capacity},
             {"saturated_samples", saturated_samples},
             {"saturated_since_unix_ms", saturated_since},
             {"saturated_until_unix_ms", saturated_until},
             {"consecutive_multiturn_root_misses", root_streak},
             {"recomputed_prefill_tokens", recomputed_tokens},
             {"sample_request_ids", sample_ids}},
            "Restart restores reuse temporarily. Preserve the log and investigate Host State "
            "checkpoint ownership before the pool saturates again.",
            "measured",
            {{"requests", root_streak},
             {"throughput_events", saturated_samples},
             {"window_s", static_cast<double>(saturated_until - saturated_since) / 1000.0}}));
    }

    const double mean_queue = paired > 0 ? b.queue_wait_sum / paired : 0.0;
    const double mean_prefill =
        dones.empty() ? 0.0 : b.prefill_sum / static_cast<double>(dones.size());
    const double mean_decode =
        dones.empty() ? 0.0 : b.decode_sum / static_cast<double>(dones.size());
    const int cause_max = std::max({b.queued, b.prefill, b.decode, b.mixed});
    std::string cause   = "mixed";
    std::string cause_id = "latency.mixed";
    std::vector<std::int64_t> cause_ids;
    if (cause_max == b.queued && b.queued > 0) {
        cause    = "queued behind concurrency";
        cause_id = "latency.queued_behind_concurrency";
        cause_ids = b.queued_ids;
    } else if (cause_max == b.prefill && b.prefill > 0) {
        cause    = "long prefill";
        cause_id = "latency.prefill_dominated";
        cause_ids = b.prefill_ids;
    } else if (cause_max == b.decode && b.decode > 0) {
        cause    = "decode-dominated";
        cause_id = "latency.decode_dominated";
        cause_ids = b.decode_ids;
    }

    std::ostringstream sat;
    sat << paired << " paired of " << dones.size() << " request_done: " << b.queued
        << " queued, " << b.prefill << " prefill-dominated, " << b.decode
        << " decode-dominated, " << b.mixed << " mixed. Dominant cause: " << cause
        << ". Mean queue wait " << (mean_queue * 1000.0) << " ms, mean prefill "
        << (mean_prefill * 1000.0) << " ms, mean decode " << (mean_decode * 1000.0)
        << " ms. Scheduler peak waiting=" << max_waiting << " running=" << max_running
        << " prefilling=" << max_prefill << ".";

    const bool pressure = b.queued > 0 && (b.queued * 3 >= paired || max_waiting > 0);
    insights.push_back(insight_available(
        cause_id, pressure ? "warning" : "info", "Saturation vs latency", sat.str(),
        {{"queued", b.queued},
         {"prefill_dominated", b.prefill},
         {"decode_dominated", b.decode},
         {"mixed", b.mixed},
         {"mean_queue_wait_s", mean_queue},
         {"mean_prefill_s", mean_prefill},
         {"mean_decode_s", mean_decode},
         {"scheduler_peak",
          {{"waiting", max_waiting}, {"running", max_running}, {"prefilling", max_prefill}}},
         {"sample_request_ids", cause_ids}},
        pressure ? "Queued wait is a concurrency/backlog problem, not a slow kernel. "
                   "Raise --max-concurrency only if KV/headroom allows; otherwise the "
                   "engine is saturated."
                 : "",
        "measured", over));

    const double n_done = static_cast<double>(dones.size());
    const double mean_prepare = n_done > 0 ? b.prepare_sum / n_done : 0.0;
    const double mean_vision  = n_done > 0 ? b.vision_sum / n_done : 0.0;
    const double mean_ttft    = n_done > 0 ? b.ttft_sum / n_done : 0.0;
    const double ttft_body    = mean_prepare + mean_prefill + mean_vision;
    std::string ttft_cause    = "mixed";
    std::string ttft_id       = "latency.ttft_mixed";
    std::string ttft_rec;
    if (mean_prefill >= mean_prepare && mean_prefill >= mean_vision && mean_prefill >= 0.4 * std::max(ttft_body, mean_ttft)) {
        ttft_cause = "prefill";
        ttft_id    = "latency.ttft_prefill_dominated";
        ttft_rec   = "TTFT is prefill-dominated. --prefill-chunk is the lever, not decode kernels.";
    } else if (mean_prepare >= mean_prefill && mean_prepare >= 0.4 * std::max(ttft_body, mean_ttft)) {
        ttft_cause = "prepare";
        ttft_id    = "latency.ttft_prepare_dominated";
        ttft_rec   = "TTFT is prepare-dominated (tokenize/media), not GPU decode.";
    } else if (mean_vision >= 0.4 * std::max(ttft_body, mean_ttft) && mean_vision > 0.0) {
        ttft_cause = "vision";
        ttft_id    = "latency.ttft_vision_dominated";
        ttft_rec   = "TTFT is vision-preprocess dominated.";
    }
    std::ostringstream ttft_stmt;
    ttft_stmt << "Mean TTFT " << (mean_ttft * 1000.0) << " ms over " << dones.size()
              << " request_done: prepare " << (mean_prepare * 1000.0) << " ms, prefill "
              << (mean_prefill * 1000.0) << " ms, vision " << (mean_vision * 1000.0)
              << " ms (decode " << (mean_decode * 1000.0)
              << " ms is after first token). Dominant TTFT component: " << ttft_cause << ".";
    insights.push_back(insight_available(
        ttft_id, "info", "TTFT decomposition", ttft_stmt.str(),
        {{"mean_ttft_s", mean_ttft},
         {"mean_prepare_s", mean_prepare},
         {"mean_prefill_s", mean_prefill},
         {"mean_vision_s", mean_vision},
         {"mean_decode_s", mean_decode},
         {"dominant", ttft_cause}},
        ttft_rec, "measured", over));

    const double multi_hit_ratio =
        multi_prompt_tokens == 0
            ? 0.0
            : static_cast<double>(multi_hit_tokens) / static_cast<double>(multi_prompt_tokens);
    std::ostringstream reuse_stmt;
    reuse_stmt << "Reuse mix over " << dones.size() << " request_done: full_reset single-turn "
               << reuse_reset_single << " (expected), full_reset multi-turn " << reuse_reset_multi
               << ", restore " << reuse_restore << ", seed " << reuse_seed << ", append "
               << reuse_append << ", other " << reuse_other << ". Multi-turn prefix-hit ratio "
               << (multi_hit_ratio * 100.0) << "% (" << multi_hit_tokens << "/"
               << multi_prompt_tokens << " tokens).";
    insights.push_back(insight_available(
        "prefix.reuse_mix", reuse_reset_multi > 0 ? "notice" : "info", "Prefix-cache reuse mix",
        reuse_stmt.str(),
        {{"full_reset_single_turn", reuse_reset_single},
         {"full_reset_multi_turn", reuse_reset_multi},
         {"restore", reuse_restore},
         {"seed", reuse_seed},
         {"append", reuse_append},
         {"other", reuse_other},
         {"multi_turn_prompt_tokens", multi_prompt_tokens},
         {"multi_turn_hit_tokens", multi_hit_tokens},
         {"multi_turn_hit_ratio", multi_hit_ratio}},
        "", "measured", over));
    if (reuse_reset_multi > 0) {
        std::ostringstream miss;
        miss << reuse_reset_multi << " of " << dones.size()
             << " request_done were multi-turn (message_count>=2) on full_reset with "
             << "prefix_cache_hit_tokens often 0. A single-message full_reset is expected; "
             << "a multi-turn full_reset is a miss that should have been restore or seed.";
        insights.push_back(insight_available(
            "prefix.multiturn_full_reset", "warning", "Multi-turn conversations resetting the prefix",
            miss.str(),
            {{"full_reset_multi_turn", reuse_reset_multi},
             {"full_reset_single_turn", reuse_reset_single},
             {"samples", reset_multi_samples}},
            "Check seed store / turn checkpoints. restore_turn_checkpoint or seed_prefix "
            "should fire when message_count>=2.",
            "measured", over));
    }

    // Speculative decoding: the counters are measured; any draft-window advice is inferred from
    // the position curve and says so. No telemetry or no drafts is unavailable, never 0%.
    {
        constexpr double kMinUsefulPositionRate = 0.20;
        constexpr std::uint64_t kMinDraftRoundsForAdvice = 50;
        auto pct = [](double ratio) {
            std::ostringstream o;
            o << std::fixed << std::setprecision(1) << (ratio * 100.0) << "%";
            return o.str();
        };
        const auto spec_over = nlohmann::json{{"requests", spec.requests_with_drafts},
                                              {"requests_with_telemetry", spec.requests_with_telemetry},
                                              {"examined_requests", spec_scope_dones},
                                              {"other_instance_requests", spec.requests_other_instance},
                                              {"server_instance_id", latest_server_instance},
                                              {"draft_rounds", spec.draft_rounds},
                                              {"window_s", spec.tmax > spec.tmin
                                                               ? static_cast<double>(spec.tmax - spec.tmin) / 1000.0
                                                               : 0.0}};
        if (spec.requests_with_telemetry == 0) {
            std::ostringstream stmt;
            stmt << "no request_done of the latest server instance carries speculative telemetry "
                 << "with a configured backend; " << spec.requests_backend_none << " of "
                 << spec_scope_dones << " report backend none and the rest have no speculative object";
            insights.push_back(insight_unavailable(
                "speculative.draft_acceptance", "Speculative decoding telemetry is not in the window",
                stmt.str(),
                {{"requests_backend_none", spec.requests_backend_none},
                 {"requests", spec_scope_dones},
                 {"server_instance_id", latest_server_instance}},
                {{"requests", 0},
                 {"examined_requests", spec_scope_dones},
                 {"other_instance_requests", spec.requests_other_instance}}));
        } else if (spec.drafted == 0) {
            std::ostringstream stmt;
            stmt << "backend " << spec.backend << " with draft window " << spec.draft_window
                 << " is configured, but the " << spec.requests_with_telemetry
                 << " request_done with telemetry recorded 0 drafted tokens ("
                 << spec.fallback << " fallback steps); acceptance cannot be measured from zero drafts";
            insights.push_back(insight_unavailable(
                "speculative.draft_acceptance", "Speculative decoding has not drafted yet", stmt.str(),
                {{"backend", spec.backend},
                 {"draft_window", spec.draft_window},
                 {"drafted_tokens", 0},
                 {"fallback_steps", spec.fallback},
                 {"requests_with_telemetry", spec.requests_with_telemetry},
                 {"server_instance_id", latest_server_instance}},
                spec_over));
        } else {
            nlohmann::json per_position_accepted = nlohmann::json::array();
            nlohmann::json per_position_rate     = nlohmann::json::array();
            int effective_window                 = 0;
            bool prefix_useful                   = true;
            std::ostringstream curve;
            for (std::size_t p = 0; p < spec.positions.size(); ++p) {
                const auto& pos   = spec.positions[p];
                const double rate = pos.rounds > 0
                                        ? std::min(1.0, static_cast<double>(pos.accepted) /
                                                            static_cast<double>(pos.rounds))
                                        : 0.0;
                per_position_accepted.push_back(pos.accepted);
                per_position_rate.push_back(rate);
                if (prefix_useful && rate >= kMinUsefulPositionRate) {
                    ++effective_window;
                } else {
                    prefix_useful = false;
                }
                curve << (p == 0 ? "" : ", ") << "P" << (p + 1) << " " << pct(rate);
            }
            const double acceptance =
                static_cast<double>(spec.accepted) / static_cast<double>(spec.drafted);
            const std::uint64_t steps = spec.draft_rounds + spec.fallback;
            auto share = [](std::uint64_t part, std::uint64_t whole) {
                return whole > 0 ? static_cast<double>(part) / static_cast<double>(whole) : 0.0;
            };
            const double fallback_share = share(spec.fallback, steps);
            const double first_share    = share(spec.first_half_fallback, spec.first_half_steps);
            const double second_share   = share(spec.second_half_fallback, spec.second_half_steps);
            const bool fallback_high    = spec.fallback >= 10 && fallback_share >= 0.20;
            const bool fallback_rising  = spec.second_half_fallback >= 10 &&
                                         second_share >= first_share + 0.10;
            const bool advise_window = spec.draft_rounds >= kMinDraftRoundsForAdvice &&
                                       effective_window >= 1 &&
                                       effective_window < spec.draft_window;

            nlohmann::json windows_seen = nlohmann::json::object();
            for (const auto& [w, n] : spec.windows_seen) { windows_seen[std::to_string(w)] = n; }

            std::ostringstream stmt;
            stmt << "Backend " << spec.backend << ", draft window " << spec.draft_window << ": "
                 << spec.accepted << " of " << spec.drafted << " drafted tokens accepted ("
                 << pct(acceptance) << ") over " << spec.draft_rounds << " draft rounds in "
                 << spec.requests_with_drafts << " requests with drafts ("
                 << spec.requests_with_telemetry << " with telemetry, " << spec_scope_dones
                 << " examined on the latest server instance). Acceptance by draft position: "
                 << curve.str() << ". Fallback steps " << spec.fallback << " = "
                 << pct(fallback_share) << " of " << steps << " decode steps (first half "
                 << pct(first_share) << ", second half " << pct(second_share) << ").";

            std::ostringstream rec;
            if (fallback_high || fallback_rising) {
                rec << "Measured: the engine decoded without a draft on " << pct(fallback_share)
                    << " of steps";
                if (fallback_rising) {
                    rec << ", rising from " << pct(first_share) << " to " << pct(second_share)
                        << " across the window";
                }
                rec << ". The log records that those steps ran on the ordinary decode path, not why. "
                       "Inferred: compare the high_fallback_samples (media, thinking, message "
                       "count) with requests that drafted normally, and check the engine log "
                       "around them, before changing the draft window. ";
            }
            if (advise_window) {
                rec << "Inferred, not measured: draft positions " << (effective_window + 1) << ".."
                    << spec.draft_window << " are accepted in under "
                    << pct(kMinUsefulPositionRate) << " of draft rounds, so a draft window of "
                    << effective_window << " would drop mostly rejected work. Verify with "
                    << "--draft-tokens " << effective_window
                    << " and compare decode tok/s before keeping it.";
            }
            std::string recommendation = rec.str();
            while (!recommendation.empty() && recommendation.back() == ' ') { recommendation.pop_back(); }
            insights.push_back(insight_available(
                "speculative.draft_acceptance", (fallback_high || fallback_rising) ? "warning" : "info",
                "Speculative draft acceptance", stmt.str(),
                {{"backend", spec.backend},
                 {"server_instance_id", latest_server_instance},
                 {"draft_window", spec.draft_window},
                 {"draft_windows_seen", windows_seen},
                 {"draft_rounds", spec.draft_rounds},
                 {"rounds_reported", spec.rounds_reported},
                 {"drafted_tokens", spec.drafted},
                 {"accepted_tokens", spec.accepted},
                 {"acceptance_ratio", acceptance},
                 {"accepted_per_position", per_position_accepted},
                 {"per_position_rate", per_position_rate},
                 {"min_useful_position_rate", kMinUsefulPositionRate},
                 {"effective_draft_window", effective_window},
                 {"inferred_draft_window",
                  advise_window ? nlohmann::json(effective_window) : nlohmann::json(nullptr)},
                 {"fallback_steps", spec.fallback},
                 {"fallback_share", fallback_share},
                 {"fallback_share_first_half", first_share},
                 {"fallback_share_second_half", second_share},
                 {"requests_with_drafts", spec.requests_with_drafts},
                 {"requests_with_telemetry", spec.requests_with_telemetry},
                 {"sample_request_ids", spec.sample_ids},
                 {"low_acceptance_samples", spec.low_acceptance_samples},
                 {"high_fallback_samples", spec.high_fallback_samples}},
                recommendation, "measured", spec_over));
        }
    }

    // Context and KV capacity pressure on the latest server instance: the configured limits come
    // from its server_start, traffic from its request_done records, occupancy and pressure from
    // its throughput samples. Occupancy counts retained checkpoints while prefix reuse is on, so a
    // full pool is only pressure when the engine also evicted, spilled or queued.
    {
        constexpr double kNearContextRatio = 0.90;
        constexpr double kHighKvRatio      = 0.90;
        auto in_scope = [&](const nlohmann::json& j) {
            return latest_server_instance.empty() ||
                   j.value("server_instance_id", "") == latest_server_instance;
        };
        auto pct = [](double ratio) {
            std::ostringstream o;
            o << std::fixed << std::setprecision(1) << (ratio * 100.0) << "%";
            return o.str();
        };

        const auto max_context     = json_i64(latest_engine, "max_context");
        const auto max_concurrency = json_i64(latest_engine, "max_concurrency");
        const auto kv_tokens       = json_i64(latest_engine, "kv_capacity");
        const auto kv_page_groups  = json_i64(latest_engine, "kv_capacity_page_groups");
        const bool prefix_reuse    = latest_engine.value("prefix_reuse", false);
        const bool have_config     = max_context > 0 && kv_page_groups > 0;

        struct Largest {
            std::int64_t id = 0;
            std::int64_t prompt = 0;
            std::int64_t completion = 0;
            std::int64_t total = 0;
            std::string finish;
        };
        std::vector<std::int64_t> prompts;
        std::vector<Largest> largest;
        for (const auto& done : dones) {
            if (!in_scope(done)) { continue; }
            const auto& req    = done.contains("request") ? done.at("request") : nlohmann::json::object();
            const auto& result = done.contains("result") ? done.at("result") : nlohmann::json::object();
            Largest l;
            l.id         = json_i64(req, "request_id");
            l.prompt     = std::max<std::int64_t>(0, json_i64(result, "prompt_tokens"));
            l.completion = std::max<std::int64_t>(0, json_i64(result, "completion_tokens"));
            l.total      = l.prompt + l.completion;
            l.finish     = result.value("finish_reason", "");
            prompts.push_back(l.prompt);
            largest.push_back(std::move(l));
        }
        std::sort(prompts.begin(), prompts.end());
        std::sort(largest.begin(), largest.end(),
                  [](const Largest& a, const Largest& b) { return a.total > b.total; });
        const std::size_t n_req = prompts.size();
        const std::int64_t prompt_median = n_req ? prompts[(n_req - 1) / 2] : 0;
        const std::int64_t prompt_p90 = n_req ? prompts[std::min(n_req - 1, (n_req * 9) / 10)] : 0;
        const std::int64_t prompt_max = n_req ? prompts.back() : 0;
        const std::int64_t largest_total = largest.empty() ? 0 : largest.front().total;
        const double largest_ratio =
            max_context > 0 ? static_cast<double>(largest_total) / static_cast<double>(max_context) : 0.0;
        int near_limit = 0;
        nlohmann::json largest_samples = nlohmann::json::array();
        for (const auto& l : largest) {
            const double ratio =
                max_context > 0 ? static_cast<double>(l.total) / static_cast<double>(max_context) : 0.0;
            if (ratio >= kNearContextRatio) { ++near_limit; }
            if (largest_samples.size() < 8) {
                largest_samples.push_back({{"request_id", l.id},
                                           {"prompt_tokens", l.prompt},
                                           {"completion_tokens", l.completion},
                                           {"total_tokens", l.total},
                                           {"context_ratio", ratio},
                                           {"finish_reason", l.finish}});
            }
        }

        std::size_t occupancy_samples = 0;
        std::int64_t kv_now = 0;
        std::int64_t kv_high = 0;
        std::int64_t peak_running = 0;
        std::int64_t peak_waiting = 0;
        std::int64_t checkpoints_dropped = 0;
        std::int64_t private_evicted = 0;
        std::int64_t shared_evicted = 0;
        std::int64_t spill_pages = 0;
        std::int64_t search_exhaustions = 0;
        for (const auto& tp : throughputs) {
            if (!in_scope(tp)) { continue; }
            if (tp.contains("scheduler") && tp.at("scheduler").is_object()) {
                const auto& sch = tp.at("scheduler");
                peak_running    = std::max(peak_running, json_i64(sch, "running"));
                peak_waiting    = std::max(peak_waiting, json_i64(sch, "waiting"));
            }
            if (!tp.contains("context_cache") || !tp.at("context_cache").is_object()) { continue; }
            const auto& cache = tp.at("context_cache");
            if (cache.contains("occupancy") && cache.at("occupancy").is_object() &&
                cache.at("occupancy").contains("device_main_kv_pages")) {
                const auto pages = json_i64(cache.at("occupancy"), "device_main_kv_pages", -1);
                if (pages >= 0) {
                    ++occupancy_samples;
                    kv_now  = pages;
                    kv_high = std::max(kv_high, pages);
                }
            }
            if (cache.contains("pressure") && cache.at("pressure").is_object()) {
                // Throughput records carry per-interval deltas, so sums are window totals.
                const auto& pr = cache.at("pressure");
                checkpoints_dropped += json_i64(pr, "checkpoints_dropped");
                private_evicted += json_i64(pr, "private_owners_evicted");
                shared_evicted += json_i64(pr, "shared_owners_evicted");
                spill_pages += json_i64(pr, "spill_pages");
                search_exhaustions += json_i64(pr, "search_budget_exhaustions");
            }
        }
        int admission_expired = 0;
        nlohmann::json admission_expired_ids = nlohmann::json::array();
        for (const auto& err : errors) {
            if (!in_scope(err)) { continue; }
            const auto& e = err.contains("error") ? err.at("error") : nlohmann::json::object();
            if (e.value("message", "").find("waiting for admission") == std::string::npos) { continue; }
            ++admission_expired;
            if (admission_expired_ids.size() < 8) {
                const auto& req = err.contains("request") ? err.at("request") : nlohmann::json::object();
                admission_expired_ids.push_back(json_i64(req, "request_id"));
            }
        }
        const double kv_now_ratio =
            kv_page_groups > 0 ? static_cast<double>(kv_now) / static_cast<double>(kv_page_groups) : 0.0;
        const double kv_high_ratio =
            kv_page_groups > 0 ? static_cast<double>(kv_high) / static_cast<double>(kv_page_groups) : 0.0;
        const std::int64_t pressure_events =
            private_evicted + shared_evicted + spill_pages + search_exhaustions;

        nlohmann::json config = {{"max_context", max_context},
                                 {"max_concurrency", max_concurrency},
                                 {"kv_capacity_tokens", kv_tokens},
                                 {"kv_page_groups", kv_page_groups},
                                 {"tokens_per_page_group",
                                  kv_page_groups > 0 ? kv_tokens / kv_page_groups : 0},
                                 {"prefix_reuse", prefix_reuse}};
        nlohmann::json requests_evidence = {
            {"count", n_req},
            {"prompt_tokens", {{"median", prompt_median}, {"p90", prompt_p90}, {"max", prompt_max}}},
            {"largest_total_tokens", largest_total},
            {"largest_context_ratio", largest_ratio},
            {"near_limit_count", near_limit},
            {"near_limit_ratio", kNearContextRatio}};
        nlohmann::json kv_evidence = {{"pages_now", kv_now},
                                      {"pages_high_water", kv_high},
                                      {"pages_capacity", kv_page_groups},
                                      {"utilization_now", kv_now_ratio},
                                      {"utilization_high_water", kv_high_ratio},
                                      {"samples", occupancy_samples},
                                      {"includes_retained_checkpoints", prefix_reuse}};
        nlohmann::json scheduler_evidence = {{"peak_running", peak_running},
                                             {"peak_waiting", peak_waiting},
                                             {"admission_expired", admission_expired},
                                             {"admission_expired_request_ids", admission_expired_ids}};
        nlohmann::json pressure_evidence = {{"checkpoints_dropped", checkpoints_dropped},
                                            {"private_owners_evicted", private_evicted},
                                            {"shared_owners_evicted", shared_evicted},
                                            {"spill_pages", spill_pages},
                                            {"search_budget_exhaustions", search_exhaustions}};
        const auto cap_over = nlohmann::json{{"requests", n_req},
                                             {"throughput_events", occupancy_samples},
                                             {"server_instance_id", latest_server_instance}};

        if (!have_config) {
            insights.push_back(insight_unavailable(
                "capacity.context_kv_pressure", "Capacity limits are not in the log window",
                "the latest server instance has no server_start with max_context and "
                "kv_capacity_page_groups, so traffic cannot be compared to a limit; prompt "
                "sizes are reported without a pressure verdict",
                {{"server_instance_id", latest_server_instance},
                 {"requests", requests_evidence},
                 {"largest_requests", largest_samples}},
                cap_over));
        } else if (occupancy_samples == 0) {
            insights.push_back(insight_unavailable(
                "capacity.context_kv_pressure", "KV occupancy has not been sampled yet",
                "the latest server instance has capacity configuration but no throughput record "
                "with context_cache.occupancy.device_main_kv_pages; KV utilization is unknown, "
                "not zero",
                {{"server_instance_id", latest_server_instance},
                 {"config", config},
                 {"requests", requests_evidence},
                 {"largest_requests", largest_samples}},
                cap_over));
        } else {
            std::string kind = "comfortable";
            std::string severity = "info";
            std::ostringstream rec;
            if (near_limit > 0) {
                kind     = "large_prompt";
                severity = "warning";
                rec << "Measured: " << near_limit << " request(s) used at least "
                    << pct(kNearContextRatio) << " of max_context " << max_context
                    << "; the largest reached " << largest_total << " tokens. Inferred: raise "
                       "--max-context (and --kv-capacity to hold it) if VRAM allows, or shorten "
                       "those conversations; a request over the limit is rejected at admission.";
            } else if ((peak_waiting > 0 || admission_expired > 0) &&
                       max_concurrency > 0 && peak_running >= max_concurrency) {
                // Every lane was busy while requests waited: the concurrency cap, not the pool.
                kind     = "lanes_full";
                severity = admission_expired > 0 ? "warning" : "notice";
                rec << "Measured: requests waited (peak " << peak_waiting << " waiting, "
                    << admission_expired << " expired before admission) while all "
                    << max_concurrency << " lanes were running; KV occupancy peaked at "
                    << pct(kv_high_ratio) << ". Inferred: the concurrency limit is what queued "
                       "them; raise --max-concurrency only if KV capacity and VRAM leave room "
                       "for another request's context.";
            } else if ((peak_waiting > 0 || admission_expired > 0) && kv_high_ratio >= kHighKvRatio) {
                // Lanes were free but requests still waited: the pool could not admit them.
                kind     = "kv_admission";
                severity = "warning";
                rec << "Measured: requests waited (peak " << peak_waiting << " waiting, "
                    << admission_expired << " expired before admission) with only "
                    << peak_running << " of " << max_concurrency << " lanes running and KV "
                       "occupancy at " << pct(kv_high_ratio) << ", while the largest single "
                       "request used " << pct(largest_ratio) << " of max_context. Inferred: "
                       "concurrent requests exhausted the KV pool; raise --kv-capacity if VRAM "
                       "allows, otherwise lower --max-context so each request reserves less.";
            } else if (kv_high_ratio >= kHighKvRatio && pressure_events > 0) {
                // No queueing: the engine made room by evicting retained state. Follow-ups on
                // the evicted conversations re-prefill; nothing failed.
                kind     = "cache_churn";
                severity = "notice";
                rec << "Measured: KV occupancy peaked at " << pct(kv_high_ratio) << " and the "
                       "engine evicted " << (private_evicted + shared_evicted)
                    << " retained conversation states (" << checkpoints_dropped
                    << " checkpoints dropped) with no request waiting. Follow-ups on evicted "
                       "conversations re-prefill instead of reusing. Inferred: raise "
                       "--kv-capacity if VRAM allows; lowering --max-concurrency does not "
                       "help this.";
            } else if (kv_high_ratio >= kHighKvRatio) {
                kind = "retained_cache";
                rec << "Measured: KV occupancy peaked at " << pct(kv_high_ratio)
                    << " without evictions, spills or queueing. With prefix reuse "
                    << (prefix_reuse ? "on" : "off")
                    << " the pool holds retained checkpoints, so a full pool alone is not "
                       "pressure. Nothing to change from this window.";
            }

            std::ostringstream stmt;
            stmt << "Configured max_context " << max_context << " tokens, KV capacity " << kv_tokens
                 << " tokens (" << kv_page_groups << " page groups), max concurrency "
                 << max_concurrency << ", prefix reuse " << (prefix_reuse ? "on" : "off") << ". Over "
                 << n_req << " request_done on the latest server instance: prompt tokens median "
                 << prompt_median << ", p90 " << prompt_p90 << ", max " << prompt_max
                 << "; largest request " << largest_total << " tokens = " << pct(largest_ratio)
                 << " of max_context, " << near_limit << " within " << pct(kNearContextRatio)
                 << " of it. KV pages " << kv_now << " of " << kv_page_groups << " now ("
                 << pct(kv_now_ratio) << "), high-water " << kv_high << " (" << pct(kv_high_ratio)
                 << ") over " << occupancy_samples << " samples"
                 << (prefix_reuse ? ", including retained checkpoints" : "")
                 << ". Scheduler peak running " << peak_running << " of " << max_concurrency
                 << ", waiting " << peak_waiting << ", " << admission_expired
                 << " expired before admission. Pressure over the window: checkpoints dropped "
                 << checkpoints_dropped << ", owners evicted " << private_evicted << " private / "
                 << shared_evicted << " shared, spill pages " << spill_pages
                 << ", search budget exhaustions " << search_exhaustions << ". Verdict: " << kind
                 << ".";

            insights.push_back(insight_available(
                "capacity.context_kv_pressure", severity, "Context and KV capacity pressure",
                stmt.str(),
                {{"server_instance_id", latest_server_instance},
                 {"kind", kind},
                 {"config", config},
                 {"requests", requests_evidence},
                 {"largest_requests", largest_samples},
                 {"kv", kv_evidence},
                 {"scheduler", scheduler_evidence},
                 {"pressure", pressure_evidence},
                 {"high_kv_ratio", kHighKvRatio}},
                rec.str(), "measured", cap_over));
        }
    }

    // Content/reasoning_content are not in schema_version 10 request logs.
    insights.push_back(insight_unavailable(
        "client.content_fields", "Visitor content is not in the request log",
        "result.content and reasoning_content are not written to request_done; "
        "empty-reply-vs-reasoning cannot be confirmed from this source",
        {{"schema_version", 10},
         {"looked_for", nlohmann::json::array({"content", "reasoning_content"})},
         {"requests", dones.size()}},
        over));

    if (output_limit_thinking > 0) {
        std::ostringstream stmt;
        stmt << output_limit_thinking << " of " << dones.size()
             << " request_done finished on output_limit with enable_thinking=true"
             << " (" << output_limit_hit_cap << " also hit requested_output_tokens). "
             << thinking_requests << " of " << dones.size() << " had thinking enabled.";
        int cap_sum = 0;
        for (int c : output_limit_caps) { cap_sum += c; }
        const double cap_mean =
            output_limit_caps.empty()
                ? 0.0
                : static_cast<double>(cap_sum) / static_cast<double>(output_limit_caps.size());
        insights.push_back(insight_available(
            "client.output_limit_while_thinking",
            output_limit_thinking * 5 >= static_cast<int>(dones.size()) ? "warning" : "notice",
            "Thinking requests hitting output_limit", stmt.str(),
            {{"output_limit_thinking", output_limit_thinking},
             {"hit_requested_cap", output_limit_hit_cap},
             {"thinking_requests", thinking_requests},
             {"mean_requested_output_tokens", cap_mean},
             {"sample_request_ids", output_limit_ids}},
            "Inferred: a thinking model with a small max_tokens can spend the budget on "
            "reasoning and return an empty visitor reply. Content fields are not in this log, "
            "so raise requested_output_tokens and compare finish_reason.",
            "measured", over));
    }

    insights.push_back(insight_unavailable(
        "client.narrated_tool_intent", "Narrated tool intent cannot be scored from JSONL",
        "detecting narrated-intent-with-no-tools needs visitor-facing text; the request log "
        "does not store content. tool_count is measurable and is reported in evidence.",
        {{"requests_with_tools", tools_declared},
         {"requests", dones.size()},
         {"requests_without_tools", static_cast<int>(dones.size()) - tools_declared}},
        over));

    return report;
}

inline void append_admin_vram_insights(nlohmann::json& report, const nlohmann::json& admin,
                                       const std::string& note) {
    if (!report.contains("insights") || !report["insights"].is_array()) {
        report["insights"] = nlohmann::json::array();
    }
    if (!admin.is_object()) {
        report["insights"].push_back(insight_unavailable(
            "vram.admin", "Admin VRAM is not available",
            note.empty() ? "admin/vram was not readable; cannot tell if any tier is releasable"
                         : note,
            {{"note", note}}));
        return;
    }
    nlohmann::json pinned = nlohmann::json::array();
    nlohmann::json released = nlohmann::json::array();
    if (admin.contains("tiers") && admin.at("tiers").is_array()) {
        for (const auto& tier : admin.at("tiers")) {
            const auto min_b = json_i64(tier, "min_bytes");
            const auto max_b = json_i64(tier, "max_bytes");
            const bool rel   = tier.value("released", false);
            if (rel) { released.push_back(tier.value("name", "?")); }
            if (min_b > 0 && min_b == max_b) {
                pinned.push_back({{"name", tier.value("name", "")},
                                  {"min_bytes", min_b},
                                  {"max_bytes", max_b},
                                  {"reclaimable_bytes", json_i64(tier, "reclaimable_bytes")},
                                  {"released", rel}});
            }
        }
    }
    const auto over = nlohmann::json{{"requests", 0},
                                     {"admin_tiers", pinned.size() + released.size()},
                                     {"last_transition", admin.value("last_transition", "")},
                                     {"last_reason", admin.value("last_reason", "")}};
    if (!pinned.empty()) {
        report["insights"].push_back(insight_available(
            "vram.tier_pinned_unreleasable", "warning",
            "Admin VRAM is enabled but a tier cannot be released",
            "A tier has min_bytes == max_bytes while --admin-vram is on. "
            "--prefix-cache-mib N pins seed min=max=N, so reclaimable_bytes stays 0 "
            "and the admin surface looks healthy while nothing can be released.",
            {{"pinned_tiers", pinned},
             {"last_transition", admin.value("last_transition", "")},
             {"last_reason", admin.value("last_reason", "")}},
            "Omit --prefix-cache-mib or set a max above min if you want idle release.",
            "measured", over));
    }
    if (!released.empty()) {
        report["insights"].push_back(insight_available(
            "vram.tier_currently_released", "notice",
            "A VRAM tier is currently released",
            "Released tiers: " + released.dump() +
                ". The engine is serving degraded (no cross-request prefix seeding) until reclaim. "
                "Release is ~120x cheaper than reclaim on this hardware.",
            {{"released", released},
             {"last_transition", admin.value("last_transition", "")},
             {"last_reason", admin.value("last_reason", "")}},
            "Reclaim before a traffic burst; a released seed store will full_reset more often.",
            "measured", over));
    }
}

inline nlohmann::json insights_from_request_log_path(const std::string& path) {
    nlohmann::json report;
    report["insights"] = nlohmann::json::array();
    if (path.empty()) {
        report["source"] = {{"request_log", "unconfigured"}, {"path", ""}};
        report["insights"].push_back(insight_unavailable(
            "source.request_log", "Request log is not configured",
            "no request_done records in window", {{"path", ""}}));
        return report;
    }
    std::ifstream in(path);
    if (!in) {
        report["source"] = {{"request_log", "missing"}, {"path", path}};
        report["insights"].push_back(insight_unavailable(
            "source.request_log", "Request log is not present",
            "no request_done records in window", {{"path", path}}));
        return report;
    }
    std::ostringstream body;
    body << in.rdbuf();
    return analyze_request_log_jsonl(body.str(), path);
}

} // namespace ninfer::supervisor
