#include "targets/qwen3_8_flash_next/impl/perf_telemetry.h"
#include "targets/qwen3_8_flash_next/impl/route_handoff_policy.h"
#include <iostream>

using namespace ninfer::targets::qwen3_8_flash_next::detail;
int main() {
    for (bool prefill : {false, true}) {
        if (route_handoff_enabled("", prefill) || route_handoff_enabled("0", prefill) ||
            !route_handoff_enabled("1", prefill) ||
            route_handoff_enabled("prefill", prefill) != prefill) return 1;
    }
    try { (void)route_handoff_enabled("invalid", false); return 1; }
    catch (const std::invalid_argument&) {}

    const std::array<std::int32_t, 10> ids{0, 0, 1, 2, 3, 511, 511, 511, 4, 4};
    ExpertRouteHistogram histogram(ids);
    if (histogram.distinct != 6 || histogram.routes != 10 || histogram.frequency[511] != 3)
        return 1;
    try {
        const std::array<std::int32_t, 1> bad{512};
        ExpertRouteHistogram invalid(bad);
        return 1;
    } catch (const std::invalid_argument&) {}
    ExpertLayerMeasurement mixed;
    mixed.layer = 47;
    mixed.tokens = 1;
    mixed.gpu_hit_routes = 5;
    mixed.cpu_miss_routes = 5;
    mixed.cache_result_d2h_bytes = 10*2560*4;
    mixed.routed_sum_h2d_bytes = 2560*4;
    mixed.cpu_miss_h2d_bytes = 5*2560*4;
    PerfContext context{.executor=1, .transaction=9, .phase="verify"};
    context.span_count = 1;
    context.spans[0] = {0, 1, 0, 2, 42};
    context.input_token_ids = {101};
    {
        PerfContextScope scope(context);
        const auto record = mixed.json(histogram);
        if (record.find("\"transaction\":9") == std::string::npos ||
            record.find("\"phase\":\"verify\"") == std::string::npos ||
            record.find("\"cpu_miss_h2d_bytes\":51200") == std::string::npos) return 1;
        try { PerfContextScope nested(context); throw std::runtime_error("test unwind"); }
        catch (const std::runtime_error&) {}
        if (!active_perf_context || active_perf_context->transaction != 9) return 1;
    }
    if (active_perf_context != nullptr) return 1;
    std::cout << mixed.json(histogram) << '\n';
    return 0;
}
