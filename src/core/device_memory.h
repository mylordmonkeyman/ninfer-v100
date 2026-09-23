#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer {

struct ProcessMemoryInfo {
    std::uint32_t pid        = 0;
    std::uint64_t used_bytes = 0; // 0 if per-process memory is unavailable (e.g. under WDDM)
};

struct DeviceMemorySnapshot {
    std::size_t total_bytes = 0;
    std::size_t free_bytes  = 0;
    std::size_t used_bytes  = 0;
    bool is_nvml            = false;
    std::string device_name;
    std::string pci_bus_id;
    std::string warning;
    std::vector<ProcessMemoryInfo> compute_processes;
};

// Queries device memory. Prefers NVML (device-wide truth across all processes).
// Falls back to cudaMemGetInfo if NVML is unavailable or fails, logging a warning that
// memory sizing is single-process-only.
DeviceMemorySnapshot query_device_memory(int device_index = 0);

// Formats a byte quantity into a human-readable string (e.g. "71.25 GiB", "512.00 MiB").
std::string format_device_memory_bytes(std::size_t bytes);

// Constructs an actionable error string when device memory is insufficient for planned weights and desktop reserve.
std::string format_insufficient_memory_error(
    const DeviceMemorySnapshot& mem,
    int device_index,
    std::size_t weight_bytes,
    std::size_t desktop_reserve_bytes,
    std::size_t additional_required_bytes = 0);

// Thrown when an allocation would breach the active desktop reserve floor.
// The runtime desktop reserve floor serves as a backstop against unbounded allocations.
struct DeviceReserveBreach : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Runtime desktop reserve floor management (Sequence D23).
// The desktop reserve floor is an active safety backstop ensuring runtime device allocations
// never exhaust VRAM below the operating system / desktop compositor reserve floor.
void set_runtime_desktop_reserve_floor(std::size_t floor_bytes) noexcept;
std::size_t runtime_desktop_reserve_floor() noexcept;

// Checks whether an allocation of additional_bytes on device_index would violate
// the active desktop reserve floor. Returns true if safe, false if it would breach.
bool check_runtime_desktop_reserve(int device_index, std::size_t additional_bytes,
                                   std::string* error_detail = nullptr);

} // namespace ninfer
