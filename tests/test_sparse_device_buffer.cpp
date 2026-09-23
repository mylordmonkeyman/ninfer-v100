// SparseDeviceBuffer is the primitive cooperative residency rests on: a fixed
// address range whose physical backing can shrink and grow. The properties that
// matter are not "it allocates" but:
//
//   - reserving address space costs no physical memory
//   - the base pointer NEVER moves, because tensor views and captured CUDA graph
//     nodes hold it
//   - shrinking actually returns bytes to the driver, measured rather than assumed
//   - shrinking rounds UP, keeping a partly used boundary chunk, and reports the
//     number it really kept rather than the one that was asked for
//   - data below the new size survives, which is what makes a retained floor a
//     floor and not a promise
//   - grown backing is zeroed, since a fresh physical handle promises nothing
//
// Free-memory checks read cuMemGetInfo in BYTES. Reporting it in GiB to two
// decimals once hid an 8 MiB change behind the format string and produced a wrong
// conclusion about the API's resolution.
#include "core/sparse_device_buffer.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* what) {
    if (!condition) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

std::size_t free_bytes() {
    std::size_t free_now = 0;
    std::size_t total    = 0;
    if (cuMemGetInfo(&free_now, &total) != CUDA_SUCCESS) { return 0; }
    return free_now;
}

} // namespace

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::cout << "no CUDA device; skipping\n";
        return 0;
    }
    cudaFree(nullptr); // establish the primary context before touching the driver API
    if (cuInit(0) != CUDA_SUCCESS) {
        std::cout << "CUDA driver API unavailable; skipping\n";
        return 0;
    }
    if (!ninfer::SparseDeviceBuffer::supported()) {
        std::cout << "device does not support virtual memory management; skipping\n";
        return 0;
    }

    const std::size_t unit = ninfer::SparseDeviceBuffer::granularity();
    check(unit > 0, "granularity is reported");
    if (unit == 0) { return 1; }

    // Reserving address space must not consume physical memory. Reserve a range
    // far larger than the backing to make the difference unmistakable.
    const std::size_t reserve = unit * 64;
    const std::size_t initial = unit * 8;
    const std::size_t before_construct = free_bytes();
    {
        ninfer::SparseDeviceBuffer buffer(reserve, initial);
        const std::size_t after_construct = free_bytes();
        check(buffer.reserved_bytes() == reserve, "the full range is reserved");
        check(buffer.backed_bytes() == initial, "only the requested prefix is backed");
        const std::size_t consumed = before_construct - after_construct;
        // Physical cost tracks the BACKING, not the reservation. Allow slack for
        // context bookkeeping but not for 56 unbacked chunks.
        check(consumed < initial + unit * 4,
              "reserving address space does not cost the reserved size in physical memory");

        void* const base = buffer.data();
        check(base != nullptr, "a base address is produced");

        // Fill the whole backed prefix with a recognisable pattern.
        std::vector<std::uint8_t> pattern(initial, 0xAB);
        check(cudaMemcpy(base, pattern.data(), initial, cudaMemcpyHostToDevice) == cudaSuccess,
              "the backed prefix is writable");

        // Shrink to a size that lands INSIDE a chunk. The implementation must keep
        // that chunk whole and report the rounded-up size, not the requested one.
        const std::size_t ragged   = unit * 4 + unit / 2;
        const std::size_t before   = free_bytes();
        const std::size_t after_shrink = buffer.shrink_to(ragged);
        const std::size_t reclaimed    = free_bytes() - before;
        check(after_shrink == unit * 5,
              "shrink rounds up and keeps the partly used boundary chunk");
        check(buffer.backed_bytes() == after_shrink, "the reported backing matches what was kept");
        check(buffer.data() == base, "the base address does not move when backing shrinks");
        check(reclaimed >= (initial - after_shrink),
              "shrinking actually returns bytes to the driver");

        // The retained prefix is the floor: it must still hold what was written.
        std::vector<std::uint8_t> read_back(after_shrink, 0);
        check(cudaMemcpy(read_back.data(), base, after_shrink, cudaMemcpyDeviceToHost) ==
                  cudaSuccess,
              "the retained prefix is still readable");
        bool intact = true;
        for (const std::uint8_t value : read_back) { intact = intact && value == 0xAB; }
        check(intact, "data below the new size survives the shrink");

        // Growing again must zero the new region, because a fresh handle carries
        // no promise about its contents and the caller cannot tell it from reuse.
        const std::size_t grown = buffer.grow_to(unit * 10);
        check(grown == unit * 10, "grow backs the requested size");
        check(buffer.data() == base, "the base address does not move when backing grows");
        std::vector<std::uint8_t> tail(unit, 0xFF);
        check(cudaMemcpy(tail.data(), static_cast<char*>(base) + unit * 9, unit,
                         cudaMemcpyDeviceToHost) == cudaSuccess,
              "the grown region is readable");
        bool zeroed = true;
        for (const std::uint8_t value : tail) { zeroed = zeroed && value == 0; }
        check(zeroed, "newly backed memory is zeroed");

        // The floor is still intact after a shrink and a grow.
        std::vector<std::uint8_t> floor_again(unit * 4, 0);
        check(cudaMemcpy(floor_again.data(), base, unit * 4, cudaMemcpyDeviceToHost) ==
                  cudaSuccess,
              "the floor is readable after a grow");
        bool floor_intact = true;
        for (const std::uint8_t value : floor_again) { floor_intact = floor_intact && value == 0xAB; }
        check(floor_intact, "the floor survives a shrink followed by a grow");

        // Growing past the reservation is a programming error, not a silent clamp:
        // a clamp would advertise capacity that has no address space behind it.
        bool threw = false;
        try {
            (void)buffer.grow_to(reserve + unit);
        } catch (const std::exception&) { threw = true; }
        check(threw, "growing beyond the reserved range is refused");
    }

    // Destruction returns everything, including the reservation.
    const std::size_t after_destruct = free_bytes();
    check(after_destruct + unit * 4 >= before_construct,
          "destruction returns the backing to the driver");

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
