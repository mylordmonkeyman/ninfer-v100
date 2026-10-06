#pragma once

#include "core/host_worker_pool.h"
#include "targets/qwen3_8_flash_next/impl/ple_read_batch.h"
#include "targets/qwen3_8_flash_next/impl/ple_table.h"

#include <string_view>

namespace ninfer::targets::qwen3_8_flash_next::detail {

enum class PleIoMode : std::uint8_t { Mmap, Auto, Direct };
struct PleIoPolicy {
    PleIoMode mode = PleIoMode::Mmap;
    bool strict_direct = false;
};
PleIoPolicy parse_ple_io_policy(std::string_view mode, std::string_view strict);
PleIoPolicy ple_io_policy_from_environment();

struct PleIoResult {
    bool direct = false;
    bool fallback = false;
    std::size_t pages = 0;
    double read_us = 0.0;
};

// Exact compressed gather. Direct/auto recover through mmap after draining page writers unless
// strict direct was explicitly selected. Invalid row addresses never enter either storage path.
PleIoResult gather_ple_rows_storage(
    const PleTableView& table, std::span<const std::array<std::int64_t, 16>> rows,
    std::span<std::byte> codes, std::span<std::byte> scales,
    PlePageBuffer& pages, HostWorkerPool* workers, std::size_t queue_depth, PleIoPolicy policy);

} // namespace ninfer::targets::qwen3_8_flash_next::detail
