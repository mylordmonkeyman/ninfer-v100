#include "serve/device_snapshot_cache.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <thread>

using namespace std::chrono_literals;

void require(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        std::fflush(stderr);
        std::_Exit(1); // A deadlocked worker must not turn a failed check into a hung test.
    }
}

int main() {
    ninfer::serve::DeviceSnapshotCache cache;
    std::atomic<int> queries{0};
    std::promise<void> entered;
    std::promise<void> release;
    auto released = release.get_future().share();
    auto first = std::async(std::launch::async, [&] {
        return cache.read([&] {
            ++queries;
            entered.set_value();
            released.wait();
            ninfer::DeviceMemorySnapshot snapshot;
            snapshot.free_bytes = 123;
            return snapshot;
        });
    });
    entered.get_future().wait();
    auto second = std::async(std::launch::async, [&] {
        return cache.read([&] {
            ++queries;
            return ninfer::DeviceMemorySnapshot{};
        });
    });
    const bool returned = second.wait_for(250ms) == std::future_status::ready;
    release.set_value();
    require(first.wait_for(1s) == std::future_status::ready,
            "cold telemetry readers deadlock the refresher while holding the value lock");
    require(returned, "cold telemetry reader blocks behind an in-progress driver query");
    require(!second.get(), "cold concurrent reader must report unavailable");
    require(first.get()->snapshot.free_bytes == 123 && queries == 1,
            "one driver query must publish the initial snapshot");
    auto fresh = cache.read([&] { ++queries; return ninfer::DeviceMemorySnapshot{}; });
    require(fresh && fresh->snapshot.free_bytes == 123 && queries == 1,
            "fresh telemetry should reuse the published reading");

    std::this_thread::sleep_for(2100ms);
    std::promise<void> stale_entered;
    std::promise<void> stale_release;
    auto stale_released = stale_release.get_future().share();
    auto refresher = std::async(std::launch::async, [&] {
        return cache.read([&] {
            ++queries;
            stale_entered.set_value();
            stale_released.wait();
            ninfer::DeviceMemorySnapshot snapshot;
            snapshot.free_bytes = 456;
            return snapshot;
        });
    });
    stale_entered.get_future().wait();
    auto stale = cache.read([&] { ++queries; return ninfer::DeviceMemorySnapshot{}; });
    require(stale && stale->snapshot.free_bytes == 123 && stale->age_ms >= 2000 && queries == 2,
            "concurrent refresh must serve the stale reading without another driver query");
    stale_release.set_value();
    require(refresher.get()->snapshot.free_bytes == 456,
            "completed refresh must publish the new device reading");
    std::puts("Device snapshot cache concurrency tests passed");
}
