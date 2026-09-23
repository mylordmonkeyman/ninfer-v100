#include "core/device_memory.h"

#include <cassert>
#include <iostream>
#include <string>

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

} // namespace

int main() {
    int failures = 0;

    std::cout << "[test_device_memory] Running byte formatting tests...\n";
    failures += check(ninfer::format_device_memory_bytes(0) == "0 B", "format 0 B");
    failures += check(ninfer::format_device_memory_bytes(1024) == "1.00 KiB", "format 1 KiB");
    failures += check(ninfer::format_device_memory_bytes(1048576) == "1.00 MiB", "format 1 MiB");
    failures += check(ninfer::format_device_memory_bytes(8ULL * 1024ULL * 1024ULL * 1024ULL) == "8.00 GiB",
                      "format 8 GiB");

    std::cout << "[test_device_memory] Running insufficient memory error message formatting test...\n";
    ninfer::DeviceMemorySnapshot mock_snap{
        .total_bytes      = 96ULL * 1024ULL * 1024ULL * 1024ULL,
        .free_bytes       = 24ULL * 1024ULL * 1024ULL * 1024ULL,
        .used_bytes       = 72ULL * 1024ULL * 1024ULL * 1024ULL,
        .is_nvml          = true,
        .device_name      = "NVIDIA RTX PRO 6000 Blackwell",
        .pci_bus_id       = "00000000:01:00.0",
        .warning          = "",
        .compute_processes = {
            ninfer::ProcessMemoryInfo{.pid = 14788, .used_bytes = 70ULL * 1024ULL * 1024ULL * 1024ULL},
        },
    };

    const std::string err_msg = ninfer::format_insufficient_memory_error(
        mock_snap, 0, 71ULL * 1024ULL * 1024ULL * 1024ULL, 8ULL * 1024ULL * 1024ULL * 1024ULL);
    std::cout << "Generated Error Message Preview:\n" << err_msg << "\n\n";

    failures += check(err_msg.find("Another process or OS component holds 72.00 GiB") != std::string::npos,
                      "error message reports other process usage");
    failures += check(err_msg.find("requires at least 79.00 GiB (71.00 GiB model weights + 8.00 GiB desktop reserve)") !=
                          std::string::npos,
                      "error message reports required weights + reserve");
    failures += check(err_msg.find("only 24.00 GiB of 96.00 GiB total device memory is available") != std::string::npos,
                      "error message reports free of total memory");
    failures += check(err_msg.find("deficit: 55.00 GiB") != std::string::npos,
                      "error message reports exact deficit");
    failures += check(err_msg.find("PID 14788") != std::string::npos,
                      "error message reports competing PID");
    failures += check(err_msg.find("NVML device-wide query") != std::string::npos,
                      "error message reports NVML telemetry source");

    std::cout << "[test_device_memory] Querying live device memory on device 0...\n";
    ninfer::DeviceMemorySnapshot live_snap = ninfer::query_device_memory(0);
    std::cout << "  Device name:  " << live_snap.device_name << '\n';
    std::cout << "  PCI bus ID:   " << live_snap.pci_bus_id << '\n';
    std::cout << "  Is NVML:      " << (live_snap.is_nvml ? "YES (NVML device-wide truth)" : "NO (cudaMemGetInfo fallback)") << '\n';
    std::cout << "  Total VRAM:   " << ninfer::format_device_memory_bytes(live_snap.total_bytes) << '\n';
    std::cout << "  Free VRAM:    " << ninfer::format_device_memory_bytes(live_snap.free_bytes) << '\n';
    std::cout << "  Used VRAM:    " << ninfer::format_device_memory_bytes(live_snap.used_bytes) << '\n';
    std::cout << "  Active procs: " << live_snap.compute_processes.size() << '\n';
    for (const auto& proc : live_snap.compute_processes) {
        std::cout << "    PID " << proc.pid << " (used: " << ninfer::format_device_memory_bytes(proc.used_bytes) << ")\n";
    }

    failures += check(live_snap.total_bytes > 0, "total VRAM > 0");
    failures += check(live_snap.is_nvml, "live query resolved via NVML");

    if (failures == 0) {
        std::cout << "\nALL TESTS PASSED.\n";
        return 0;
    }
    std::cerr << "\nTESTS FAILED with " << failures << " errors.\n";
    return 1;
}
