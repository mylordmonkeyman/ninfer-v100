#include "core/device_memory.h"

#include <cuda_runtime.h>
#include <nvml.h>

#include <atomic>
#include <cstdio>
#include <iomanip>
#include <mutex>
#include <sstream>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace ninfer {
namespace {

using FnNvmlInit = nvmlReturn_t (*)();
using FnNvmlShutdown = nvmlReturn_t (*)();
using FnNvmlErrorString = const char* (*)(nvmlReturn_t);
using FnNvmlDeviceGetHandleByIndex = nvmlReturn_t (*)(unsigned int, nvmlDevice_t*);
using FnNvmlDeviceGetHandleByPciBusId = nvmlReturn_t (*)(const char*, nvmlDevice_t*);
using FnNvmlDeviceGetMemoryInfo = nvmlReturn_t (*)(nvmlDevice_t, nvmlMemory_t*);
using FnNvmlDeviceGetName = nvmlReturn_t (*)(nvmlDevice_t, char*, unsigned int);
using FnNvmlDeviceGetComputeRunningProcesses =
    nvmlReturn_t (*)(nvmlDevice_t, unsigned int*, nvmlProcessInfo_t*);

class NvmlDriverLoader {
public:
    static NvmlDriverLoader& instance() {
        static NvmlDriverLoader loader;
        return loader;
    }

    bool is_available() const noexcept { return available_; }

    const std::string& init_warning() const noexcept { return init_warning_; }

    bool query_device(int device_index, DeviceMemorySnapshot& snapshot) {
        if (!available_) { return false; }

        char pci_bus[64] = {0};
        cudaError_t cuda_rc =
            cudaDeviceGetPCIBusId(pci_bus, static_cast<int>(sizeof(pci_bus)), device_index);
        nvmlDevice_t handle = nullptr;
        nvmlReturn_t dev_rc = NVML_ERROR_UNKNOWN;

        if (cuda_rc == cudaSuccess && pci_bus[0] != '\0' && fn_get_by_pci_) {
            dev_rc = fn_get_by_pci_(pci_bus, &handle);
        }
        if (dev_rc != NVML_SUCCESS && fn_get_by_index_) {
            dev_rc = fn_get_by_index_(static_cast<unsigned int>(device_index), &handle);
        }
        if (dev_rc != NVML_SUCCESS) {
            std::call_once(device_warn_flag_, [&]() {
                std::fprintf(stderr,
                             "[device-memory] WARNING: NVML handle lookup failed for device %d (%s); "
                             "falling back to cudaMemGetInfo.\n",
                             device_index,
                             fn_error_string_ ? fn_error_string_(dev_rc) : "unknown error");
            });
            return false;
        }

        nvmlMemory_t mem{};
        nvmlReturn_t mem_rc = fn_get_mem_(handle, &mem);
        if (mem_rc != NVML_SUCCESS) {
            std::call_once(device_warn_flag_, [&]() {
                std::fprintf(stderr,
                             "[device-memory] WARNING: nvmlDeviceGetMemoryInfo failed for device %d (%s); "
                             "falling back to cudaMemGetInfo.\n",
                             device_index,
                             fn_error_string_ ? fn_error_string_(mem_rc) : "unknown error");
            });
            return false;
        }

        snapshot.total_bytes = static_cast<std::size_t>(mem.total);
        snapshot.free_bytes  = static_cast<std::size_t>(mem.free);
        snapshot.used_bytes  = static_cast<std::size_t>(mem.used);
        snapshot.is_nvml     = true;
        snapshot.pci_bus_id  = pci_bus[0] != '\0' ? std::string(pci_bus) : "";

        char name_buf[96] = {0};
        if (fn_get_name_ && fn_get_name_(handle, name_buf, sizeof(name_buf)) == NVML_SUCCESS) {
            snapshot.device_name = name_buf;
        }

        if (fn_get_compute_procs_) {
            unsigned int proc_count = 32;
            std::vector<nvmlProcessInfo_t> procs(proc_count);
            if (fn_get_compute_procs_(handle, &proc_count, procs.data()) == NVML_SUCCESS) {
                snapshot.compute_processes.reserve(proc_count);
                for (unsigned int i = 0; i < proc_count; ++i) {
                    const std::uint64_t vram =
                        procs[i].usedGpuMemory == 0xFFFFFFFFFFFFFFFFULL ? 0ULL : procs[i].usedGpuMemory;
                    snapshot.compute_processes.push_back(ProcessMemoryInfo{
                        .pid        = procs[i].pid,
                        .used_bytes = vram,
                    });
                }
            }
        }

        return true;
    }

private:
    NvmlDriverLoader() {
        init_library();
    }

    ~NvmlDriverLoader() {
        if (available_ && fn_shutdown_) {
            fn_shutdown_();
        }
#if defined(_WIN32)
        if (dll_module_) { FreeLibrary(dll_module_); }
#else
        if (so_handle_) { dlclose(so_handle_); }
#endif
    }

    void init_library() {
#if defined(_WIN32)
        dll_module_ = LoadLibraryA("nvml.dll");
        if (!dll_module_) {
            dll_module_ = LoadLibraryA("C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvml.dll");
        }
        if (!dll_module_) {
            init_warning_ = "NVML runtime library (nvml.dll) not found in system path or NVSMI directory";
            warn_fallback();
            return;
        }

        auto resolve = [this](const char* sym) {
            return reinterpret_cast<void*>(GetProcAddress(dll_module_, sym));
        };
#else
        so_handle_ = dlopen("libnvidia-ml.so.1", RTLD_NOW);
        if (!so_handle_) {
            so_handle_ = dlopen("libnvidia-ml.so", RTLD_NOW);
        }
        if (!so_handle_) {
            init_warning_ = "NVML runtime library (libnvidia-ml.so.1) not found in loader path";
            warn_fallback();
            return;
        }

        auto resolve = [this](const char* sym) {
            return dlsym(so_handle_, sym);
        };
#endif

        fn_init_ = reinterpret_cast<FnNvmlInit>(resolve("nvmlInit_v2"));
        if (!fn_init_) {
            fn_init_ = reinterpret_cast<FnNvmlInit>(resolve("nvmlInit"));
        }
        fn_shutdown_ = reinterpret_cast<FnNvmlShutdown>(resolve("nvmlShutdown"));
        fn_error_string_ = reinterpret_cast<FnNvmlErrorString>(resolve("nvmlErrorString"));
        fn_get_by_pci_ = reinterpret_cast<FnNvmlDeviceGetHandleByPciBusId>(
            resolve("nvmlDeviceGetHandleByPciBusId_v2"));
        if (!fn_get_by_pci_) {
            fn_get_by_pci_ = reinterpret_cast<FnNvmlDeviceGetHandleByPciBusId>(
                resolve("nvmlDeviceGetHandleByPciBusId"));
        }
        fn_get_by_index_ = reinterpret_cast<FnNvmlDeviceGetHandleByIndex>(
            resolve("nvmlDeviceGetHandleByIndex_v2"));
        if (!fn_get_by_index_) {
            fn_get_by_index_ = reinterpret_cast<FnNvmlDeviceGetHandleByIndex>(
                resolve("nvmlDeviceGetHandleByIndex"));
        }
        fn_get_mem_ = reinterpret_cast<FnNvmlDeviceGetMemoryInfo>(
            resolve("nvmlDeviceGetMemoryInfo"));
        fn_get_name_ = reinterpret_cast<FnNvmlDeviceGetName>(
            resolve("nvmlDeviceGetName"));
        fn_get_compute_procs_ = reinterpret_cast<FnNvmlDeviceGetComputeRunningProcesses>(
            resolve("nvmlDeviceGetComputeRunningProcesses_v3"));
        if (!fn_get_compute_procs_) {
            fn_get_compute_procs_ = reinterpret_cast<FnNvmlDeviceGetComputeRunningProcesses>(
                resolve("nvmlDeviceGetComputeRunningProcesses"));
        }

        if (!fn_init_ || !fn_get_mem_ || (!fn_get_by_pci_ && !fn_get_by_index_)) {
            init_warning_ = "NVML library loaded but required API entry points were missing";
            warn_fallback();
            return;
        }

        const nvmlReturn_t rc = fn_init_();
        if (rc != NVML_SUCCESS) {
            const char* err_str = fn_error_string_ ? fn_error_string_(rc) : "unknown error";
            init_warning_ = std::string("nvmlInit failed: ") + err_str;
            warn_fallback();
            return;
        }

        available_ = true;
    }

    void warn_fallback() const {
        std::fprintf(stderr,
                     "[device-memory] WARNING: %s; falling back to cudaMemGetInfo. "
                     "Memory sizing is single-process-only and blind to other GPU allocations.\n",
                     init_warning_.c_str());
    }

#if defined(_WIN32)
    HMODULE dll_module_ = nullptr;
#else
    void* so_handle_ = nullptr;
#endif
    bool available_ = false;
    std::string init_warning_;
    std::once_flag device_warn_flag_;

    FnNvmlInit fn_init_                                        = nullptr;
    FnNvmlShutdown fn_shutdown_                                = nullptr;
    FnNvmlErrorString fn_error_string_                          = nullptr;
    FnNvmlDeviceGetHandleByPciBusId fn_get_by_pci_              = nullptr;
    FnNvmlDeviceGetHandleByIndex fn_get_by_index_              = nullptr;
    FnNvmlDeviceGetMemoryInfo fn_get_mem_                      = nullptr;
    FnNvmlDeviceGetName fn_get_name_                          = nullptr;
    FnNvmlDeviceGetComputeRunningProcesses fn_get_compute_procs_ = nullptr;
};

} // namespace

DeviceMemorySnapshot query_device_memory(int device_index) {
    DeviceMemorySnapshot snapshot;

    if (NvmlDriverLoader::instance().query_device(device_index, snapshot)) {
        return snapshot;
    }

    // Fallback: cudaMemGetInfo
    std::size_t free_bytes  = 0;
    std::size_t total_bytes = 0;
    const cudaError_t err   = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (err != cudaSuccess) {
        std::fprintf(stderr,
                     "[device-memory] ERROR: cudaMemGetInfo failed for device %d: %s\n",
                     device_index, cudaGetErrorString(err));
        return snapshot;
    }

    snapshot.total_bytes = total_bytes;
    snapshot.free_bytes  = free_bytes;
    snapshot.used_bytes  = total_bytes > free_bytes ? (total_bytes - free_bytes) : 0;
    snapshot.is_nvml     = false;
    snapshot.warning     = NvmlDriverLoader::instance().init_warning();

    cudaDeviceProp props{};
    if (cudaGetDeviceProperties(&props, device_index) == cudaSuccess) {
        snapshot.device_name = props.name;
    }
    char pci_bus[64] = {0};
    if (cudaDeviceGetPCIBusId(pci_bus, static_cast<int>(sizeof(pci_bus)), device_index) ==
        cudaSuccess) {
        snapshot.pci_bus_id = pci_bus;
    }

    return snapshot;
}

std::string format_device_memory_bytes(std::size_t bytes) {
    constexpr double kKiB = 1024.0;
    constexpr double kMiB = 1024.0 * kKiB;
    constexpr double kGiB = 1024.0 * kMiB;
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);
    if (bytes >= static_cast<std::size_t>(kGiB)) {
        oss << static_cast<double>(bytes) / kGiB << " GiB";
    } else if (bytes >= static_cast<std::size_t>(kMiB)) {
        oss << static_cast<double>(bytes) / kMiB << " MiB";
    } else if (bytes >= static_cast<std::size_t>(kKiB)) {
        oss << static_cast<double>(bytes) / kKiB << " KiB";
    } else {
        oss << bytes << " B";
    }
    return oss.str();
}

std::string format_insufficient_memory_error(
    const DeviceMemorySnapshot& mem,
    int device_index,
    std::size_t weight_bytes,
    std::size_t desktop_reserve_bytes,
    std::size_t additional_required_bytes) {
    const std::size_t total_required =
        weight_bytes + desktop_reserve_bytes + additional_required_bytes;
    const std::size_t deficit =
        total_required > mem.free_bytes ? (total_required - mem.free_bytes) : 0;

    std::ostringstream oss;
    oss << "Insufficient device memory: Another process or OS component holds "
        << format_device_memory_bytes(mem.used_bytes)
        << " on device " << device_index;
    if (!mem.device_name.empty()) {
        oss << " (" << mem.device_name << ")";
    }
    oss << ".\nThis configuration requires at least "
        << format_device_memory_bytes(total_required) << " ("
        << format_device_memory_bytes(weight_bytes) << " model weights";
    if (desktop_reserve_bytes > 0) {
        oss << " + " << format_device_memory_bytes(desktop_reserve_bytes) << " desktop reserve";
    }
    if (additional_required_bytes > 0) {
        oss << " + " << format_device_memory_bytes(additional_required_bytes)
            << " minimum runtime capacity";
    }
    oss << "), but only " << format_device_memory_bytes(mem.free_bytes)
        << " of " << format_device_memory_bytes(mem.total_bytes)
        << " total device memory is available (deficit: "
        << format_device_memory_bytes(deficit) << ").\n";

    if (!mem.compute_processes.empty()) {
        oss << "Active compute processes detected on device: ";
        for (std::size_t i = 0; i < mem.compute_processes.size(); ++i) {
            if (i > 0) { oss << ", "; }
            oss << "PID " << mem.compute_processes[i].pid;
            if (mem.compute_processes[i].used_bytes > 0) {
                oss << " (" << format_device_memory_bytes(mem.compute_processes[i].used_bytes) << ")";
            }
        }
        oss << ".\n";
    }

    oss << "Telemetry source: "
        << (mem.is_nvml ? "NVML device-wide query (global physical memory)"
                        : "cudaMemGetInfo (single-process fallback; blind to other GPU allocations)")
        << ".";
    return oss.str();
}

namespace {
std::atomic<std::size_t> g_runtime_desktop_reserve_floor{0};
} // namespace

void set_runtime_desktop_reserve_floor(std::size_t floor_bytes) noexcept {
    g_runtime_desktop_reserve_floor.store(floor_bytes, std::memory_order_relaxed);
}

std::size_t runtime_desktop_reserve_floor() noexcept {
    return g_runtime_desktop_reserve_floor.load(std::memory_order_relaxed);
}

bool check_runtime_desktop_reserve(int device_index, std::size_t additional_bytes,
                                   std::string* error_detail) {
    const std::size_t floor = runtime_desktop_reserve_floor();
    if (floor == 0) {
        return true;
    }
    const DeviceMemorySnapshot mem = query_device_memory(device_index);
    if (mem.free_bytes < floor || mem.free_bytes - floor < additional_bytes) {
        if (error_detail != nullptr) {
            std::ostringstream oss;
            oss << "Device memory allocation of " << format_device_memory_bytes(additional_bytes)
                << " would breach runtime desktop reserve floor of "
                << format_device_memory_bytes(floor)
                << " (available: " << format_device_memory_bytes(mem.free_bytes)
                << ", remaining after allocation: "
                << (mem.free_bytes > additional_bytes
                        ? format_device_memory_bytes(mem.free_bytes - additional_bytes)
                        : "0 B")
                << ").";
            *error_detail = oss.str();
        }
        return false;
    }
    return true;
}

} // namespace ninfer
