#include "runtime/engine/kv_capacity.h"

#include <iostream>
#include <stdexcept>

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

} // namespace

int main() {
    int failures = 0;
    const ninfer::runtime::SequenceCapacityCurve curve{
        .main_page_tokens                     = 64,
        .minimum_main_page_groups             = 2,
        .maximum_main_page_groups             = 6,
        .minimum_device_reservation_bytes     = 1000,
        .bytes_per_additional_main_page_group = 128,
    };

    const auto automatic =
        ninfer::runtime::resolve_kv_capacity(ninfer::KvCapacityPolicy::automatic(50), curve, 1360);
    failures +=
        check(automatic.main_page_groups == 4 && automatic.resolved_tokens == 256 &&
                  automatic.runtime_reservation_bytes == 1256 &&
                  automatic.automatic_headroom_bytes == 50 && automatic.planned_slack_bytes == 104,
              "automatic KV capacity did not select the largest fitting page count");

    const auto capped =
        ninfer::runtime::resolve_kv_capacity(ninfer::KvCapacityPolicy::automatic(50), curve, 10000);
    failures += check(capped.main_page_groups == 6 && capped.resolved_tokens == 384,
                      "automatic KV capacity exceeded or missed the target maximum");

    const auto explicit_capacity = ninfer::runtime::resolve_kv_capacity(
        ninfer::KvCapacityPolicy::explicit_capacity(129), curve, 1200);
    failures +=
        check(explicit_capacity.main_page_groups == 3 && explicit_capacity.resolved_tokens == 192 &&
                  explicit_capacity.runtime_reservation_bytes == 1128,
              "explicit KV capacity did not use page-aligned token semantics");

    bool insufficient_rejected = false;
    try {
        (void)ninfer::runtime::resolve_kv_capacity(ninfer::KvCapacityPolicy::automatic(50), curve,
                                                   1049);
    } catch (const std::invalid_argument&) { insufficient_rejected = true; }
    failures += check(insufficient_rejected,
                      "automatic KV capacity accepted less than the minimum reservation");

    // Test: Automatic mode enforces slack_floor when larger than headroom
    const auto slack_floored = ninfer::runtime::resolve_kv_capacity(
        ninfer::KvCapacityPolicy::automatic(50, 200), curve, 1360);
    failures += check(slack_floored.main_page_groups == 3 && slack_floored.resolved_tokens == 192 &&
                          slack_floored.runtime_reservation_bytes == 1128 &&
                          slack_floored.slack_floor_bytes == 200 &&
                          slack_floored.planned_slack_bytes >= 200,
                      "automatic KV capacity did not respect the larger slack floor");

    // Test: Explicit mode rejects configurations violating requested slack floor
    bool explicit_starved_rejected = false;
    try {
        (void)ninfer::runtime::resolve_kv_capacity(
            ninfer::KvCapacityPolicy::explicit_capacity(129, 100), curve, 1200);
    } catch (const std::invalid_argument&) { explicit_starved_rejected = true; }
    failures += check(explicit_starved_rejected,
                      "explicit KV capacity allowed memory starvation violating slack floor");

    // Test: Explicit mode accepts when slack floor is satisfied
    const auto explicit_with_floor = ninfer::runtime::resolve_kv_capacity(
        ninfer::KvCapacityPolicy::explicit_capacity(129, 50), curve, 1200);
    failures += check(explicit_with_floor.main_page_groups == 3 &&
                          explicit_with_floor.planned_slack_bytes >= 50,
                      "explicit KV capacity rejected valid configuration with sufficient slack");

    // Test: Production 98k context with 816 page groups geometry
    const ninfer::runtime::SequenceCapacityCurve flash_next_curve{
        .main_page_tokens                     = 256,
        .minimum_main_page_groups             = 384,   // 98304 tokens
        .maximum_main_page_groups             = 1536,  // 4 * 98304 tokens
        .minimum_device_reservation_bytes     = 3'000'000'000ULL,
        .bytes_per_additional_main_page_group = 6'488'064ULL,
    };

    // Subtest: Starved explicit reservation violating slack floor is rejected
    const std::size_t reservation_816 = flash_next_curve.reservation_bytes(816);
    bool prod_starved_rejected = false;
    try {
        (void)ninfer::runtime::resolve_kv_capacity(
            ninfer::KvCapacityPolicy::explicit_capacity(208896, 512ULL * 1024ULL * 1024ULL),
            flash_next_curve,
            reservation_816 + 100ULL * 1024ULL * 1024ULL); // only 100 MiB free, floor is 512 MiB
    } catch (const std::invalid_argument&) {
        prod_starved_rejected = true;
    }
    failures += check(prod_starved_rejected,
                      "production geometry allowed starved explicit capacity violating slack floor");

    // Subtest: Automatic sizing in production always reserves at least the slack floor
    const std::size_t kOneGib = 1024ULL * 1024ULL * 1024ULL;
    const auto auto_prod = ninfer::runtime::resolve_kv_capacity(
        ninfer::KvCapacityPolicy::automatic(kOneGib, kOneGib),
        flash_next_curve,
        8'639'635'456ULL); // Igor's available_after_weights_bytes
    failures += check(auto_prod.planned_slack_bytes >= kOneGib,
                      "automatic production sizing failed to reserve the required 1 GiB slack floor");
    failures += check(auto_prod.main_page_groups >= 384,
                      "automatic production sizing resolved fewer than minimum required sequence groups");

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
