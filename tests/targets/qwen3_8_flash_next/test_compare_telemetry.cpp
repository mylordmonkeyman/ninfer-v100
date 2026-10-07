#include "targets/qwen3_8_flash_next/impl/telemetry/compare_telemetry.h"
#include <iostream>
using namespace v100_compare;
int main() {
    if (!level()) return 1;
    {
        Round round("ninfer", "owner:transaction", "decode");
        round.counter(0, Counter::total_routes, 10);
        round.counter(0, Counter::resident_routes, 4);
        round.counter(0, Counter::cpu_routes, 6);
        round.counter(0, Counter::nonresident_gpu_routes, 0);
        round.duration(0, Stage::cpu_expert, 25);
        { Round nested("ninfer", "owner:nested", "verify"); }
        if (active != &round) return 1;
        std::cout << round.json(30) << '\n';
    }
    return active ? 1 : 0;
}
