#include "core/arena.h"
#include "core/device.h"
#include "runtime/host_kv_extent_store.h"
#include "runtime/logical_kv_store.h"

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <exception>
#include <iostream>
#include <string_view>
#include <vector>

namespace {

namespace rt = ninfer::runtime;

int failures = 0;

void expect(bool condition, std::string_view message) {
    if (condition) {
        return;
    }
    ++failures;
    std::cerr << "FAIL: " << message << std::endl;
}

bool cuda_unavailable(cudaError_t error) {
    return error == cudaErrorNoDevice ||
           error == cudaErrorInsufficientDriver ||
           error == cudaErrorInitializationError;
}

void test_logical_page_lifecycle_and_residency(const ninfer::DeviceContext& device) {
    ninfer::LayoutBuilder builder;
    ninfer::DeviceKVPagePoolSpec page_spec{
        .page_group_count = 8,
        .geometry =
            {
                .page_tokens        = static_cast<std::uint32_t>(ninfer::kPagedKVPageSize),
                .device_plane_order = ninfer::PagedKVPlaneOrder::PageMajor,
                .planes = {{.dtype = ninfer::DType::BF16, .leading_extent = 4, .head_extent = 2}},
            },
    };
    const ninfer::DeviceKVPagePoolLayout page_layout =
        ninfer::plan_device_kv_page_pool(builder, page_spec);
    const ninfer::KVExecutionTableLayout table_layout =
        ninfer::plan_kv_execution_tables(builder, {.logical_page_capacity = 8, .table_rows = 2});

    const std::size_t backing_bytes = builder.finish(256);
    ninfer::DeviceArena arena(backing_bytes);
    const ninfer::DeviceSpan backing{arena.base(), arena.capacity()};

    ninfer::DeviceKVPagePool physical_pages(backing, page_layout);
    ninfer::KVExecutionTablePool physical_tables(backing, table_layout, physical_pages);

    const ninfer::HostKVPageLayout host_layout =
        ninfer::plan_host_kv_page_layout(physical_pages.geometry());
    const std::array host_layouts{host_layout};
    ninfer::HostKVArena host_arena(host_layout.page_stride * 8, host_layouts);

    // Instantiate runtime stores under namespace ninfer::runtime
    rt::LogicalKVPageStore pages(physical_pages, physical_pages.capacity_pages() + 8U);
    rt::HostKVExtentStore extents(host_arena, 8);
    rt::KVAddressSpaceStore addresses(pages, physical_tables, 4, 8);

    expect(pages.capacity() == physical_pages.capacity_pages() + 8U, "LogicalKVPageStore capacity");
    expect(pages.occupied() == 0, "Initial LogicalKVPageStore occupied is zero");
    expect(extents.capacity() == 8, "HostKVExtentStore capacity");
    expect(extents.occupied() == 0, "Initial HostKVExtentStore occupied is zero");

    // Test 1: Single and batch materialization
    auto reservation = physical_pages.reserve(4);
    expect(reservation.has_value(), "Physical reservation for 4 pages");

    const rt::LogicalKVPageHandle p0 = pages.materialize(*reservation);
    expect(p0.valid(), "p0 is valid");
    expect(pages.occupied() == 1, "pages occupied is 1");
    expect(pages.device_resident(p0), "p0 is device resident");
    expect(!pages.host_resident(p0), "p0 is not host resident yet");
    expect(pages.writer_references(p0) == 1, "p0 writer references is 1");

    std::array<rt::LogicalKVPageHandle, 3> batch{};
    pages.materialize(*reservation, batch);
    expect(batch[0].valid() && batch[1].valid() && batch[2].valid(), "Batch materialization valid");
    expect(pages.occupied() == 4, "pages occupied is 4 after batch");

    // Test 2: Reference counting and epochs
    pages.set_writer(p0, false);
    expect(pages.writer_references(p0) == 0, "p0 writer cleared");
    expect(pages.can_pin_source(p0), "p0 can be pinned as source");

    pages.pin_source(p0);
    expect(pages.source_pins(p0) == 1, "p0 source_pins is 1");
    pages.unpin_source(p0);
    expect(pages.source_pins(p0) == 0, "p0 source_pins is 0");

    // Test 3: Host extent preparation and publication
    const std::array membership_pages{p0, batch[0]};
    pages.set_writer(batch[0], false);

    auto host_prep = extents.prepare(pages, membership_pages);
    expect(host_prep.has_value(), "HostKVExtentStore prepare successful");
    expect(extents.occupied() == 1, "extents occupied is 1 during reservation");

    const std::vector<ninfer::DeviceKVPageHandle> dev_sources = extents.device_sources(*host_prep);
    expect(dev_sources.size() == 2, "dev_sources has 2 pages");
    const ninfer::HostKVAllocationView writable_host = extents.writable_view(*host_prep);
    expect(writable_host.valid() && writable_host.page_count() == 2,
           "writable host view is valid for 2 pages");

    // Publish host extent
    const rt::HostKVExtentCapability host_cap = extents.publish(std::move(*host_prep));
    expect(host_cap.valid(), "HostKVExtentCapability is valid");
    expect(pages.host_resident(p0), "p0 is now host resident");
    expect(pages.host_resident(batch[0]), "batch[0] is now host resident");
    expect(pages.host_replica_current(p0), "p0 host replica is current");

    // Test 4: Device replica dropping under memory pressure (VRAM reclamation)
    expect(pages.can_drop_device_replica(p0), "p0 can drop device replica");
    const bool dropped = pages.drop_device_replica(p0);
    expect(dropped, "drop_device_replica succeeded");
    expect(!pages.device_resident(p0), "p0 is no longer device resident");
    expect(pages.host_resident(p0), "p0 remains host resident");

    // Test 5: Device replica restoration from host
    auto restore_res = physical_pages.reserve(1);
    expect(restore_res.has_value(), "Reservation for restore");
    const ninfer::DeviceKVPageHandle restored_handle =
        pages.reserve_device_replica(p0, *restore_res);
    expect(restored_handle.valid(), "Restored device handle is valid");
    pages.publish_device_replica(p0);
    expect(pages.device_resident(p0), "p0 is device resident once again");

    // Test 6: Host extent partitioning upon partial page release
    const std::array release_targets{p0};
    const bool released_p0 = extents.release_page_replicas(pages, release_targets);
    expect(released_p0, "extents release_page_replicas for p0");
    expect(!pages.host_resident(p0), "p0 host replica released");
    expect(pages.host_resident(batch[0]), "batch[0] host replica retained in partitioned extent");

    const std::array release_batch0{batch[0]};
    const bool released_b0 = extents.release_page_replicas(pages, release_batch0);
    expect(released_b0, "extents release_page_replicas for batch[0]");
    expect(!pages.host_resident(batch[0]), "batch[0] host replica released");
    expect(host_arena.occupied_bytes() == 0, "host_arena has 0 occupied bytes");

    // Cleanup pages
    expect(pages.release(p0), "p0 released");
    expect(pages.release(batch[0]), "batch[0] released");
    expect(pages.release(batch[1]), "batch[1] released");
    expect(pages.release(batch[2]), "batch[2] released");
    expect(pages.occupied() == 0, "All pages released, occupied is 0");
}

void test_address_space_store(const ninfer::DeviceContext& device) {
    ninfer::LayoutBuilder builder;
    ninfer::DeviceKVPagePoolSpec page_spec{
        .page_group_count = 4,
        .geometry =
            {
                .page_tokens        = static_cast<std::uint32_t>(ninfer::kPagedKVPageSize),
                .device_plane_order = ninfer::PagedKVPlaneOrder::PageMajor,
                .planes = {{.dtype = ninfer::DType::BF16, .leading_extent = 4, .head_extent = 2}},
            },
    };
    const ninfer::DeviceKVPagePoolLayout page_layout =
        ninfer::plan_device_kv_page_pool(builder, page_spec);
    const ninfer::KVExecutionTableLayout table_layout =
        ninfer::plan_kv_execution_tables(builder, {.logical_page_capacity = 4, .table_rows = 2});

    const std::size_t backing_bytes = builder.finish(256);
    ninfer::DeviceArena arena(backing_bytes);
    const ninfer::DeviceSpan backing{arena.base(), arena.capacity()};

    ninfer::DeviceKVPagePool physical_pages(backing, page_layout);
    ninfer::KVExecutionTablePool physical_tables(backing, table_layout, physical_pages);

    rt::LogicalKVPageStore pages(physical_pages, 8);
    rt::KVAddressSpaceStore addresses(pages, physical_tables, 4, 4);

    const auto addr = addresses.create_active(3, 0);
    expect(addr.has_value(), "Active address created");
    expect(addresses.entitlement(*addr) == 3, "Entitlement is 3");
    expect(addresses.bound_row(*addr) == 0, "Bound row is 0");

    // Materialize 65 tokens (spans across 2 64-token pages) with real stream
    addresses.ensure_mapped_to_tokens(*addr, 65, device.stream);
    expect(addresses.mapped_pages(*addr) == 2, "Mapped pages is 2");

    addresses.commit_frontier(*addr, 65);
    expect(addresses.committed_frontier(*addr) == 65, "Committed frontier is 65");

    // Truncate back to 32 tokens
    addresses.destructive_truncate(*addr, 32);
    expect(addresses.committed_frontier(*addr) == 32, "Committed frontier is 32");

    // Deactivate address space
    addresses.deactivate(*addr);
    expect(addresses.bound_row(*addr) == -1, "Deactivated address has no bound row");
    expect(addresses.mapped_pages(*addr) == 1, "Deactivated address retained mapped prefix page");

    expect(addresses.release(*addr), "Address released");
    expect(pages.occupied() == 0, "Pages clean after address release");
}

} // namespace

int main() {
    int count                   = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (cuda_unavailable(count_err) || count == 0) {
        std::cout << "SKIP: no usable CUDA device" << std::endl;
        return 77;
    }
    if (count_err != cudaSuccess) {
        std::cerr << "CUDA error during device probe: " << cudaGetErrorString(count_err) << std::endl;
        return 1;
    }

    try {
        ninfer::DeviceContext device(0);
        std::cout << "Running ninfer_logical_kv_store_test..." << std::endl;
        test_logical_page_lifecycle_and_residency(device);
        test_address_space_store(device);
        device.synchronize();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: unexpected exception: " << error.what() << std::endl;
        return 1;
    }

    if (failures == 0) {
        std::cout << "ALL TESTS PASSED" << std::endl;
        return 0;
    }
    std::cerr << failures << " TEST FAILURES" << std::endl;
    return 1;
}
