#include "targets/qwen3_8_flash_next/impl/telemetry/compare_telemetry.h"
#include <iostream>
using namespace v100_compare;
int main() {
    if (!level()) return 1;
    {
        Round round("ninfer", "owner:transaction", "decode");
        InputContext context;
        context.input_columns = 2;
        context.executor = 7; context.transaction = 19;
        context.spans.push_back({0, 1, 42, 0, 3});
        context.spans.push_back({1, 1, 9, 1, 8});
        if (level() >= 2) context.input_token_ids = std::vector<std::int64_t>{101, 202};
        round.context(std::move(context));
        round.execution_mode("cuda_graph");
        const std::array<std::int32_t, 2> candidates{303, 404};
        round.sampled_tokens(std::span<const std::int32_t>(candidates));
        CacheSnapshot before, after;
        before.capacity_experts = after.capacity_experts = 4;
        before.capacity_bytes = after.capacity_bytes = 400;
        before.ready = 1; after.ready = 2;
        before.admissions_total = 3; after.admissions_total = 4;
        before.fills_total = 2; after.fills_total = 3;
        before.hits_total = 8; after.hits_total = 12;
        if (level() >= 2) { before.resident_ids = std::vector<int>{7}; after.resident_ids = std::vector<int>{7, 9}; }
        round.cache(0, before, after);
        round.counter(0, Counter::total_routes, 10);
        round.counter(0, Counter::resident_routes, 4);
        round.counter(0, Counter::cpu_routes, 6);
        round.counter(0, Counter::nonresident_gpu_routes, 0);
        round.duration(0, Stage::cpu_expert, 25);
        WorkerObservation worker;
        worker.configured_workers = 4;
        worker.job(1, 0, false);
        round.worker(0, worker);
        round.worker(0, worker);
        { Round nested("ninfer", "owner:nested", "verify"); }
        if (active != &round) return 1;
        const auto serialized = round.json(30);
        if (level() == 1 && serialized.find("\"workers_compact\"") == std::string::npos) return 1;
        if (level() >= 2 && serialized.find("\"workers\"") == std::string::npos) return 1;
        if (level() >= 2 && serialized.find("\"workers_compact\"") != std::string::npos) return 1;
        std::cout << serialized << '\n';
    }
    return active ? 1 : 0;
}
