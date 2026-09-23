#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ninfer {

// A device allocation whose ADDRESS is fixed at its maximum extent but whose
// physical backing is only a prefix, so the backing can shrink and grow while
// every pointer, stride and tensor shape derived from it stays put.
//
// This exists so the engine can hand GPU memory back to the rest of the machine
// -- a game, a second model, a compositor -- without tearing down and rebuilding
// its runtime state. A plain reallocation cannot do that: the flash-next
// persistent block is one bump-allocated buffer in which the stride between the
// 24 attention planes is a function of the page count, so shrinking it moves
// every base address after the first, invalidating every tensor view and every
// address baked into the 32 captured decode graphs.
//
// Reserving the maximum virtual range up front and varying only the backing
// avoids all of that: cuMemAddressReserve costs no physical memory, so the
// unbacked tail is free until it is mapped. Measured on an RTX PRO 6000
// Blackwell, a captured CUDA graph replays correctly across unmap and remap of
// memory it addresses, including reaching REPLACED backing through a device-side
// table, with a faulting negative control to show the test could fail.
//
// The caller is responsible for the only invariant this class cannot enforce:
// nothing may address bytes beyond backed_bytes(). Reads and writes there fault.
class SparseDeviceBuffer {
public:
    SparseDeviceBuffer() noexcept = default;

    // Reserves `max_bytes` of address space and backs the first `initial_bytes`.
    // Both are rounded up to the allocation granularity.
    SparseDeviceBuffer(std::size_t max_bytes, std::size_t initial_bytes);
    ~SparseDeviceBuffer();

    SparseDeviceBuffer(const SparseDeviceBuffer&)            = delete;
    SparseDeviceBuffer& operator=(const SparseDeviceBuffer&) = delete;
    SparseDeviceBuffer(SparseDeviceBuffer&& other) noexcept;
    SparseDeviceBuffer& operator=(SparseDeviceBuffer&& other) noexcept;

    // Backs at least `bytes` from the base, mapping whole chunks as needed, and
    // returns the resulting backed size. New chunks are zeroed before they are
    // reachable, because a fresh physical handle carries no promise about its
    // contents and the caller cannot tell reused backing from new.
    //
    // Strong guarantee: on failure any chunk mapped by this call is released and
    // the buffer keeps the backing it had. A half-grown buffer would advertise
    // capacity the ledger would then hand out.
    std::size_t grow_to(std::size_t bytes);

    // Releases whole chunks beyond `bytes` and returns the resulting backed size,
    // which is `bytes` rounded UP -- a partially used boundary chunk stays mapped
    // because the part of it still in use cannot be separated from the rest.
    // That rounding is the feature's physical waste and is reported, not hidden.
    std::size_t shrink_to(std::size_t bytes);

    [[nodiscard]] void* data() const noexcept { return base_; }
    [[nodiscard]] std::size_t reserved_bytes() const noexcept { return reserved_bytes_; }
    [[nodiscard]] std::size_t backed_bytes() const noexcept { return backed_bytes_; }

    // Whether this device supports the virtual memory management API at all. When
    // false the buffer cannot be constructed and the caller must fall back to a
    // fixed allocation; residency changes are simply unavailable.
    [[nodiscard]] static bool supported(int device = 0);

    // Physical mapping unit. Every grow and shrink moves a multiple of this, so
    // it is also the resolution at which memory can be returned.
    [[nodiscard]] static std::size_t granularity(int device = 0);

private:
    void release_all() noexcept;

    void* base_                 = nullptr;
    std::size_t reserved_bytes_ = 0;
    std::size_t backed_bytes_   = 0;
    std::size_t granularity_    = 0;
    int device_                 = 0;
    // One handle per mapped chunk, in address order. Held because releasing
    // physical memory needs the handle, not just the address. Stored as a plain
    // integer so this header does not drag in the CUDA driver API; the
    // translation unit static_asserts that the widths match.
    std::vector<std::uint64_t> handles_;
};

} // namespace ninfer
