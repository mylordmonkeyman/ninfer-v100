#pragma once

#include "core/device_memory.h"

#include <chrono>
#include <cstdint>
#include <mutex>
#include <optional>
#include <utility>

namespace ninfer::serve {

struct DeviceSnapshotReading {
    DeviceMemorySnapshot snapshot;
    std::int64_t age_ms = 0;
};

// A telemetry reader receives the last reading or an unavailable result while a
// refresh is in progress. The driver query runs outside the value lock.
class DeviceSnapshotCache {
public:
    template <typename Query>
    std::optional<DeviceSnapshotReading> read(Query&& query) {
        using Clock = std::chrono::steady_clock;
        constexpr auto fresh_for = std::chrono::seconds(2);
        {
            std::lock_guard lock(value_mu_);
            if (value_ && Clock::now() - taken_ < fresh_for) { return reading(); }
        }

        std::unique_lock refresh(refresh_mu_, std::try_to_lock);
        if (!refresh.owns_lock()) {
            std::lock_guard lock(value_mu_);
            if (value_) { return reading(); }
            return std::nullopt;
        }
        {
            std::lock_guard lock(value_mu_);
            if (value_ && Clock::now() - taken_ < fresh_for) { return reading(); }
        }
        DeviceMemorySnapshot fresh = std::forward<Query>(query)();
        {
            std::lock_guard lock(value_mu_);
            value_ = std::move(fresh);
            taken_ = Clock::now();
            return reading();
        }
    }

private:
    DeviceSnapshotReading reading() const {
        return {*value_, std::chrono::duration_cast<std::chrono::milliseconds>(
                             std::chrono::steady_clock::now() - taken_).count()};
    }

    std::mutex value_mu_;
    std::mutex refresh_mu_;
    std::optional<DeviceMemorySnapshot> value_;
    std::chrono::steady_clock::time_point taken_{};
};

} // namespace ninfer::serve
