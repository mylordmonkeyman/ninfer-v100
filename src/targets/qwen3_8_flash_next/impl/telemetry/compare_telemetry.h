#pragma once
// Shared with the Strata fork. Host observations only; no CUDA work or waiting.
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace v100_compare {
inline int level() {
    static const int value = [] {
        const char* s = std::getenv("V100_COMPARE_TELEMETRY_LEVEL");
        if (!s) return 0;
        if (std::strlen(s) != 1 || *s < '0' || *s > '3')
            throw std::invalid_argument("V100_COMPARE_TELEMETRY_LEVEL must be 0..3");
        return *s - '0';
    }();
    return value;
}
using Clock = std::chrono::steady_clock;
inline double elapsed_us(Clock::time_point start) {
    return std::chrono::duration<double, std::micro>(Clock::now()-start).count();
}
inline std::string quote(std::string_view text) {
    std::string out = "\"";
    constexpr char hex[] = "0123456789abcdef";
    for (unsigned char c : text) {
        if (c == '"' || c == '\\') { out += '\\'; out += char(c); }
        else if (c < 32) { out += "\\u00"; out += hex[c >> 4]; out += hex[c & 15]; }
        else out += char(c);
    }
    return out + '"';
}
enum class Counter : unsigned {
    routed_tokens, total_routes, resident_routes, cpu_routes, nonresident_gpu_routes,
    distinct_experts, resident_distinct_experts, distinct_missed_experts,
    cpu_groups, cpu_grouped_routes, cpu_weight_read_bytes, expert_h2d_bytes,
    route_d2h_bytes, output_d2h_bytes, output_h2d_bytes, count
};
inline constexpr std::array counter_names{
    "routed_tokens", "total_routes", "resident_routes", "cpu_routes", "nonresident_gpu_routes",
    "distinct_experts", "resident_distinct_experts", "distinct_missed_experts",
    "cpu_groups", "cpu_grouped_routes", "cpu_weight_read_bytes", "expert_h2d_bytes",
    "route_d2h_bytes", "output_d2h_bytes", "output_h2d_bytes"};
enum class Stage : unsigned { layer, qsa, gdn, ple, moe, cpu_expert, host_wait, head, count };
inline constexpr std::array stage_names{"layer", "qsa", "gdn", "ple", "moe", "cpu_expert", "host_wait", "head"};
struct Layer {
    bool observed = false;
    std::array<std::uint64_t, unsigned(Counter::count)> counters{};
    std::array<bool, unsigned(Counter::count)> counter_observed{};
    std::array<double, unsigned(Stage::count)> host_us{};
    std::array<bool, unsigned(Stage::count)> stage_observed{};
};
class Round;
inline thread_local Round* active = nullptr;
class Round {
public:
    Round(const char* engine, std::string id, const char* phase)
        : engine_(engine), id_(std::move(id)), phase_(phase), previous_(active),
          started_(Clock::now()), exceptions_(std::uncaught_exceptions()) { active = this; }
    Round(const Round&) = delete;
    Round& operator=(const Round&) = delete;
    ~Round() noexcept {
        active = previous_;
        try {
            const auto line = json(elapsed_us(started_), std::uncaught_exceptions() > exceptions_) + '\n';
            std::fwrite(line.data(), 1, line.size(), stderr);
        } catch (...) { std::fputs("v100 compare telemetry serialization failed\n", stderr); }
    }
    void counter(unsigned layer, Counter key, std::uint64_t value) noexcept {
        if (layer >= layers_.size()) return;
        auto& l = layers_[layer]; l.observed = true;
        l.counter_observed[unsigned(key)] = true; l.counters[unsigned(key)] += value;
    }
    void duration(unsigned layer, Stage key, double us) noexcept {
        if (layer >= layers_.size()) return;
        auto& l = layers_[layer]; l.observed = true;
        l.stage_observed[unsigned(key)] = true; l.host_us[unsigned(key)] += us;
    }
    std::string json(double wall, bool failed = false) const {
        const char* run = std::getenv("V100_COMPARE_RUN_ID");
        std::ostringstream out; out.imbue(std::locale::classic()); out.precision(17);
        out << "{\"schema\":\"ninfer-strata-v100-telemetry-v1\",\"kind\":\"round\",\"engine\":"
            << quote(engine_) << ",\"run_id\":" << quote(run ? run : "unassigned")
            << ",\"round_id\":" << quote(id_) << ",\"phase\":" << quote(phase_)
            << ",\"level\":" << level() << ",\"status\":\"" << (failed ? "failed" : "ok")
            << "\",\"timing_semantics\":\"host_observed_not_gpu_execution\",\"host_wall_us\":"
            << wall << ",\"layers\":[";
        bool comma = false;
        for (unsigned i = 0; i < layers_.size(); ++i) {
            const auto& l = layers_[i]; if (!l.observed) continue;
            if (comma) out << ',';
            comma = true;
            out << "{\"layer\":" << i << ",\"counters\":{";
            bool sep = false;
            for (unsigned j=0; j < l.counters.size(); ++j) {
                if (!l.counter_observed[j]) continue;
                if (sep) out << ',';
                sep = true;
                out << quote(counter_names[j]) << ':' << l.counters[j];
            }
            out << "},\"host_us\":{"; sep = false;
            for (unsigned j=0; j < l.host_us.size(); ++j) {
                if (!l.stage_observed[j]) continue;
                if (sep) out << ',';
                sep = true;
                out << quote(stage_names[j]) << ':' << l.host_us[j];
            }
            out << "}}";
        }
        out << "]}"; return out.str();
    }
private:
    const char* engine_; std::string id_; const char* phase_;
    Round* previous_; Clock::time_point started_; int exceptions_;
    std::array<Layer, 48> layers_{};
};
class HostSpan {
public:
    HostSpan(unsigned layer, Stage stage) : owner_(active), layer_(layer), stage_(stage),
        started_(owner_ ? Clock::now() : Clock::time_point{}) {}
    ~HostSpan() { if (owner_) owner_->duration(layer_, stage_, elapsed_us(started_)); }
    HostSpan(const HostSpan&) = delete;
    HostSpan& operator=(const HostSpan&) = delete;
private:
    Round* owner_; unsigned layer_; Stage stage_; Clock::time_point started_;
};
} // namespace v100_compare
