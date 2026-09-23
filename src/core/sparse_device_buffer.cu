#include "core/sparse_device_buffer.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace ninfer {
namespace {

static_assert(sizeof(CUmemGenericAllocationHandle) == sizeof(std::uint64_t),
              "SparseDeviceBuffer stores allocation handles as uint64 to keep the CUDA driver "
              "API out of its header");

std::string driver_error(const char* prefix, CUresult result) {
    const char* name = nullptr;
    cuGetErrorName(result, &name);
    const char* text = nullptr;
    cuGetErrorString(result, &text);
    return std::string(prefix) + ": " + (name != nullptr ? name : "?") + ": " +
           (text != nullptr ? text : "");
}

void log_cleanup_failure(const char* op, CUresult result) noexcept {
    if (result == CUDA_SUCCESS) { return; }
    const char* name = nullptr;
    cuGetErrorName(result, &name);
    std::fprintf(stderr, "CUDA cleanup failed during %s: %s\n", op, name != nullptr ? name : "?");
}

CUmemAllocationProp allocation_property(int device) noexcept {
    CUmemAllocationProp prop{};
    prop.type          = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id   = device;
    // NONE, not WIN32: an exportable handle needs a security descriptor in
    // win32HandleMetaData and fails with INVALID_VALUE without one. This memory
    // never leaves the process.
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;
    return prop;
}

std::size_t round_up(std::size_t value, std::size_t unit) noexcept {
    if (unit == 0) { return value; }
    return ((value + unit - 1) / unit) * unit;
}

} // namespace

bool SparseDeviceBuffer::supported(int device) {
    if (cuInit(0) != CUDA_SUCCESS) { return false; }
    CUdevice handle{};
    if (cuDeviceGet(&handle, device) != CUDA_SUCCESS) { return false; }
    int value = 0;
    if (cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
                             handle) != CUDA_SUCCESS) {
        return false;
    }
    return value != 0;
}

std::size_t SparseDeviceBuffer::granularity(int device) {
    const CUmemAllocationProp prop = allocation_property(device);
    std::size_t value              = 0;
    if (cuMemGetAllocationGranularity(&value, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED) !=
        CUDA_SUCCESS) {
        return 0;
    }
    return value;
}

SparseDeviceBuffer::SparseDeviceBuffer(std::size_t max_bytes, std::size_t initial_bytes) {
    if (max_bytes == 0) { return; }
    if (cuInit(0) != CUDA_SUCCESS) {
        throw std::runtime_error("SparseDeviceBuffer: CUDA driver API is unavailable");
    }
    cudaGetDevice(&device_);
    if (!supported(device_)) {
        throw std::runtime_error(
            "SparseDeviceBuffer: device does not support virtual memory management");
    }
    granularity_ = granularity(device_);
    if (granularity_ == 0) {
        throw std::runtime_error("SparseDeviceBuffer: allocation granularity is unavailable");
    }
    reserved_bytes_ = round_up(max_bytes, granularity_);

    CUdeviceptr base = 0;
    const CUresult reserved = cuMemAddressReserve(&base, reserved_bytes_, 0, 0, 0);
    if (reserved != CUDA_SUCCESS) {
        reserved_bytes_ = 0;
        throw std::runtime_error(driver_error("cuMemAddressReserve failed", reserved));
    }
    base_ = reinterpret_cast<void*>(base);

    try {
        grow_to(initial_bytes);
    } catch (...) {
        release_all();
        throw;
    }
}

SparseDeviceBuffer::~SparseDeviceBuffer() { release_all(); }

SparseDeviceBuffer::SparseDeviceBuffer(SparseDeviceBuffer&& other) noexcept
    : base_(other.base_), reserved_bytes_(other.reserved_bytes_),
      backed_bytes_(other.backed_bytes_), granularity_(other.granularity_), device_(other.device_),
      handles_(std::move(other.handles_)) {
    other.base_           = nullptr;
    other.reserved_bytes_ = 0;
    other.backed_bytes_   = 0;
    other.handles_.clear();
}

SparseDeviceBuffer& SparseDeviceBuffer::operator=(SparseDeviceBuffer&& other) noexcept {
    if (this == &other) { return *this; }
    release_all();
    base_           = other.base_;
    reserved_bytes_ = other.reserved_bytes_;
    backed_bytes_   = other.backed_bytes_;
    granularity_    = other.granularity_;
    device_         = other.device_;
    handles_        = std::move(other.handles_);
    other.base_           = nullptr;
    other.reserved_bytes_ = 0;
    other.backed_bytes_   = 0;
    other.handles_.clear();
    return *this;
}

std::size_t SparseDeviceBuffer::grow_to(std::size_t bytes) {
    if (base_ == nullptr) {
        if (bytes == 0) { return 0; }
        throw std::logic_error("SparseDeviceBuffer: grow on an unreserved buffer");
    }
    const std::size_t target = round_up(bytes, granularity_);
    if (target > reserved_bytes_) {
        throw std::invalid_argument("SparseDeviceBuffer: grow beyond the reserved range");
    }
    if (target <= backed_bytes_) { return backed_bytes_; }

    const CUmemAllocationProp prop = allocation_property(device_);
    CUmemAccessDesc access{};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id   = device_;
    access.flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    const std::size_t first_new = backed_bytes_;
    std::size_t offset          = backed_bytes_;
    // Every chunk added by this call, so a failure part-way can be undone. Growing
    // half-way and reporting success would advertise capacity the caller would
    // then hand out to kernels.
    const std::size_t rollback_mark = handles_.size();
    try {
        while (offset < target) {
            CUmemGenericAllocationHandle handle{};
            const CUresult created = cuMemCreate(&handle, granularity_, &prop, 0);
            if (created != CUDA_SUCCESS) {
                throw std::runtime_error(driver_error("cuMemCreate failed", created));
            }
            const CUresult mapped = cuMemMap(
                reinterpret_cast<CUdeviceptr>(base_) + offset, granularity_, 0, handle, 0);
            if (mapped != CUDA_SUCCESS) {
                cuMemRelease(handle);
                throw std::runtime_error(driver_error("cuMemMap failed", mapped));
            }
            handles_.push_back(static_cast<std::uint64_t>(handle));
            offset += granularity_;
        }
        const CUresult granted = cuMemSetAccess(
            reinterpret_cast<CUdeviceptr>(base_) + first_new, target - first_new, &access, 1);
        if (granted != CUDA_SUCCESS) {
            throw std::runtime_error(driver_error("cuMemSetAccess failed", granted));
        }
    } catch (...) {
        for (std::size_t index = handles_.size(); index > rollback_mark; --index) {
            const std::size_t chunk_offset = first_new + (index - 1 - rollback_mark) * granularity_;
            log_cleanup_failure("cuMemUnmap",
                                cuMemUnmap(reinterpret_cast<CUdeviceptr>(base_) + chunk_offset,
                                           granularity_));
            log_cleanup_failure(
                "cuMemRelease",
                cuMemRelease(static_cast<CUmemGenericAllocationHandle>(handles_[index - 1])));
        }
        handles_.resize(rollback_mark);
        throw;
    }

    // Fresh physical handles carry no promise about their contents, and the
    // caller cannot distinguish reused backing from new. Zeroing here means a
    // grown region is always defined, which is what makes a reclaim safe to
    // publish without the caller having to remember to initialize it.
    const cudaError_t cleared =
        cudaMemset(static_cast<char*>(base_) + first_new, 0, target - first_new);
    if (cleared != cudaSuccess) {
        throw std::runtime_error(std::string("SparseDeviceBuffer: zeroing new backing failed: ") +
                                 cudaGetErrorString(cleared));
    }
    backed_bytes_ = target;
    return backed_bytes_;
}

std::size_t SparseDeviceBuffer::shrink_to(std::size_t bytes) {
    if (base_ == nullptr) { return 0; }
    // Rounded UP: a chunk that is only partly beyond `bytes` still holds live
    // data at its start and cannot be split. Callers that report physical
    // savings must report this number, not the requested one.
    const std::size_t target = round_up(bytes, granularity_);
    if (target >= backed_bytes_) { return backed_bytes_; }

    while (backed_bytes_ > target) {
        const std::size_t chunk_offset = backed_bytes_ - granularity_;
        const CUresult unmapped =
            cuMemUnmap(reinterpret_cast<CUdeviceptr>(base_) + chunk_offset, granularity_);
        if (unmapped != CUDA_SUCCESS) {
            // Stop at the first refusal and report the backing that is really
            // still held. Continuing would leave the address range in a state
            // neither the caller nor this object could describe.
            log_cleanup_failure("cuMemUnmap", unmapped);
            return backed_bytes_;
        }
        log_cleanup_failure(
            "cuMemRelease",
            cuMemRelease(static_cast<CUmemGenericAllocationHandle>(handles_.back())));
        handles_.pop_back();
        backed_bytes_ = chunk_offset;
    }
    return backed_bytes_;
}

void SparseDeviceBuffer::release_all() noexcept {
    if (base_ == nullptr) {
        handles_.clear();
        backed_bytes_ = 0;
        return;
    }
    std::size_t offset = 0;
    for (const std::uint64_t handle : handles_) {
        log_cleanup_failure("cuMemUnmap",
                            cuMemUnmap(reinterpret_cast<CUdeviceptr>(base_) + offset,
                                       granularity_));
        log_cleanup_failure("cuMemRelease",
                            cuMemRelease(static_cast<CUmemGenericAllocationHandle>(handle)));
        offset += granularity_;
    }
    handles_.clear();
    log_cleanup_failure("cuMemAddressFree",
                        cuMemAddressFree(reinterpret_cast<CUdeviceptr>(base_), reserved_bytes_));
    base_           = nullptr;
    reserved_bytes_ = 0;
    backed_bytes_   = 0;
}

} // namespace ninfer
