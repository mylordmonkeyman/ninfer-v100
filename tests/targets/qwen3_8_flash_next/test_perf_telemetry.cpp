#include "targets/qwen3_8_flash_next/impl/perf_telemetry.h"
#include <iostream>

using namespace ninfer::targets::qwen3_8_flash_next::detail;
int main() {
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
    std::cout << mixed.json(histogram) << '\n';
    return 0;
}
