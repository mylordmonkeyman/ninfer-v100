#pragma once

// SV0 host-only diagnostics. No CUDA allocation, synchronization, or policy selection.
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <locale>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>

namespace ninfer::targets::qwen3_8_flash_next::detail {

inline bool v100_perf_telemetry_enabled() noexcept {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_V100_TELEMETRY");
        return value && std::strcmp(value, "1") == 0;
    }();
    return enabled;
}

using PerfClock = std::chrono::steady_clock;
inline double perf_elapsed_us(PerfClock::time_point started) {
    return std::chrono::duration<double, std::micro>(PerfClock::now()-started).count();
}

struct PerfLaneSpan {
    unsigned first_column = 0, columns = 0, lane = 0;
    std::uint64_t epoch = 0;
    int first_token_index = 0;
};
struct PerfContext {
    std::uint64_t executor = 0, transaction = 0;
    const char* phase = "unscoped";
    std::array<PerfLaneSpan, 8> spans{};
    unsigned span_count = 0;
};
inline thread_local const PerfContext* active_perf_context = nullptr;
inline std::uint64_t next_perf_executor_id() {
    static std::atomic<std::uint64_t> next{1};
    return next.fetch_add(1, std::memory_order_relaxed);
}
class PerfContextScope {
public:
    explicit PerfContextScope(const PerfContext& context)
        : context_(context), previous_(active_perf_context) { active_perf_context = &context_; }
    ~PerfContextScope() { active_perf_context = previous_; }
    PerfContextScope(const PerfContextScope&) = delete;
    PerfContextScope& operator=(const PerfContextScope&) = delete;
private:
    PerfContext context_;
    const PerfContext* previous_;
};
inline void append_perf_context(std::ostream& out) {
    out << ",\"context\":";
    if (!active_perf_context) { out << "null"; return; }
    const auto& c = *active_perf_context;
    out << "{\"executor\":" << c.executor << ",\"transaction\":" << c.transaction
        << ",\"phase\":\"" << c.phase << "\",\"lanes\":[";
    for (unsigned i = 0; i < c.span_count; ++i) {
        if (i) out << ',';
        const auto& s = c.spans[i];
        out << "{\"first_column\":" << s.first_column << ",\"columns\":" << s.columns
            << ",\"lane\":" << s.lane << ",\"epoch\":" << s.epoch
            << ",\"first_token_index\":" << s.first_token_index << '}';
    }
    out << "]}";
}

struct ExpertRouteHistogram {
    std::array<std::uint64_t, 512> frequency{};
    std::uint64_t routes = 0;
    unsigned distinct = 0;
    explicit ExpertRouteHistogram(std::span<const std::int32_t> ids) {
        for (int id : ids) {
            if (id < 0 || id >= 512) throw std::invalid_argument("invalid telemetry expert id");
            if (frequency[id]++ == 0) ++distinct;
            ++routes;
        }
    }
};

struct ExpertLayerMeasurement {
    unsigned layer = 0;
    bool prefill = false, cache_present = false, cache_timing = false, routed_sum_timing = false;
    std::uint64_t tokens = 0, gpu_hit_routes = 0, cpu_miss_routes = 0;
    std::uint64_t cpu_groups = 0, cpu_grouped_pairs = 0, cpu_weight_read_bytes = 0;
    std::uint64_t stream_routes = 0, stream_experts = 0, stream_expert_h2d_bytes = 0;
    // Actual payload submitted: the baseline downloads the WHOLE route buffer if ANY hit exists.
    std::uint64_t cache_result_d2h_bytes = 0, routed_sum_h2d_bytes = 0;
    std::uint64_t route_input_d2h_bytes = 0, cpu_miss_h2d_bytes = 0;
    bool route_handoff = false;
    std::uint64_t route_sequence = 0;
    double router_rendezvous_us = 0, cpu_branch_us = 0, gpu_hit_window_us = 0;
    double cache_result_d2h_us = 0, routed_sum_h2d_us = 0, cpu_miss_h2d_us = 0;
    double merge_wait_us = 0, branch_wall_us = 0, overlap_lower_bound_us = 0;
    unsigned ready_experts = 0, uploading_experts = 0, leased_experts = 0;
    std::uint64_t cache_hits_total = 0, cache_misses_total = 0;
    std::uint64_t admissions_total = 0, fills_total = 0, evictions_total = 0;
    std::uint64_t expert_staging_bytes_total = 0, cache_bytes = 0, cache_transfer_budget_bytes = 0;
    double expert_staging_us_total = 0;

    [[nodiscard]] std::string json(const ExpertRouteHistogram& histogram) const {
        std::ostringstream out;
        out.imbue(std::locale::classic());
        out.precision(17);
        out << "{\"sv\":0,\"schema\":1,\"kind\":\"expert_layer\",\"layer\":" << layer
            << ",\"prefill\":" << (prefill ? "true" : "false") << ",\"tokens\":" << tokens
            << ",\"routes\":" << histogram.routes << ",\"distinct_experts\":" << histogram.distinct
            << ",\"routes_per_distinct_expert\":" << (histogram.distinct ? double(histogram.routes)/histogram.distinct : 0)
            << ",\"gpu_hit_routes\":" << gpu_hit_routes << ",\"cpu_miss_routes\":" << cpu_miss_routes
            << ",\"cpu_groups\":" << cpu_groups
            << ",\"cpu_grouped_pairs\":" << cpu_grouped_pairs
            << ",\"cpu_weight_read_bytes\":" << cpu_weight_read_bytes
            << ",\"stream_routes\":" << stream_routes
            << ",\"stream_experts\":" << stream_experts
            << ",\"stream_expert_h2d_bytes\":" << stream_expert_h2d_bytes
            << ",\"hit_fraction\":" << (histogram.routes ? double(gpu_hit_routes)/histogram.routes : 0)
            << ",\"cache_result_d2h_bytes\":" << cache_result_d2h_bytes
            << ",\"routed_sum_h2d_bytes\":" << routed_sum_h2d_bytes
            << ",\"route_input_d2h_bytes\":" << route_input_d2h_bytes
            << ",\"cpu_miss_h2d_bytes\":" << cpu_miss_h2d_bytes
            << ",\"route_handoff\":" << (route_handoff ? "true" : "false")
            << ",\"route_sequence\":" << route_sequence
            << ",\"router_rendezvous_us\":" << router_rendezvous_us
            << ",\"cpu_branch_us\":" << cpu_branch_us
            << ",\"gpu_hit_window_us\":";
        if (cache_timing) out << gpu_hit_window_us; else out << "null";
        out << ",\"cache_result_d2h_us\":";
        if (cache_timing) out << cache_result_d2h_us; else out << "null";
        out << ",\"routed_sum_h2d_us\":";
        if (routed_sum_timing) out << routed_sum_h2d_us; else out << "null";
        out << ",\"cpu_miss_h2d_us\":";
        if (routed_sum_timing) out << cpu_miss_h2d_us; else out << "null";
        out << ",\"merge_wait_us\":";
        if (cache_timing) out << merge_wait_us; else out << "null";
        out << ",\"branch_wall_us\":" << branch_wall_us << ",\"overlap_lower_bound_us\":";
        if (cache_timing) out << overlap_lower_bound_us; else out << "null";
        out << ",\"cache\":";
        if (!cache_present) out << "null";
        else out << "{\"ready_experts\":" << ready_experts << ",\"uploading_experts\":" << uploading_experts
                 << ",\"leased_experts\":" << leased_experts << ",\"hits_total\":" << cache_hits_total
                 << ",\"misses_total\":" << cache_misses_total << ",\"admissions_total\":" << admissions_total
                 << ",\"fills_total\":" << fills_total << ",\"evictions_total\":" << evictions_total
                 << ",\"expert_staging_bytes_total\":" << expert_staging_bytes_total
                 << ",\"expert_staging_us_total\":" << expert_staging_us_total
                 << ",\"cache_bytes\":" << cache_bytes << ",\"cache_transfer_budget_bytes\":" << cache_transfer_budget_bytes << '}';
        out << ",\"frequency\":[";
        for (unsigned i = 0; i < histogram.frequency.size(); ++i) {
            if (i) out << ',';
            out << histogram.frequency[i];
        }
        out << "]";
        append_perf_context(out);
        out << '}';
        return out.str();
    }
};

inline void emit_perf_json(const std::string& record) {
    const std::string line = record + '\n';
    // Single stdio call prevents interleaved partial records across inference owners.
    std::fwrite(line.data(), 1, line.size(), stderr);
}

inline void emit_ple_perf(std::size_t tokens, std::size_t bytes, double gather_us, bool compressed) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out.precision(17);
    out << "{\"sv\":0,\"schema\":1,\"kind\":\"ple_gather\",\"tokens\":" << tokens
        << ",\"payload_bytes\":" << bytes << ",\"gather_us\":" << gather_us
        << ",\"compressed\":" << (compressed ? "true" : "false")
        << ",\"page_read_us\":null";
    append_perf_context(out);
    out << '}';
    emit_perf_json(out.str());
}
} // namespace ninfer::targets::qwen3_8_flash_next::detail
