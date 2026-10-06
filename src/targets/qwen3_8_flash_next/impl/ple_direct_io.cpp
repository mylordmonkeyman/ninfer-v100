#include "targets/qwen3_8_flash_next/impl/ple_direct_io.h"
#include "targets/qwen3_8_flash_next/impl/perf_telemetry.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>

namespace ninfer::targets::qwen3_8_flash_next::detail {
namespace {
constexpr std::size_t kPleCodesPerRow = 80;
constexpr std::size_t kPleScalesPerRow = 20;
struct PageCopy {
    std::size_t page_offset = 0;
    std::size_t destination = 0;
    std::size_t bytes       = 0;
};

void add_piece(std::map<std::uint64_t, std::vector<PageCopy>>& pages,
               std::uint64_t absolute, std::size_t bytes, std::size_t destination) {
    constexpr std::size_t page = artifact::Reader::direct_io_alignment;
    while (bytes != 0) {
        const std::uint64_t page_begin = absolute / page * page;
        const std::size_t in_page      = static_cast<std::size_t>(absolute - page_begin);
        const std::size_t amount       = std::min(bytes, page - in_page);
        pages[page_begin].push_back(PageCopy{in_page, destination, amount});
        absolute += amount;
        destination += amount;
        bytes -= amount;
    }
}

struct DirectGatherResult {
    std::size_t pages = 0;
    double read_us    = 0.0;
};

DirectGatherResult gather_direct(const PleTableView& table,
                   std::span<const std::array<std::int64_t, 16>> rows,
                   std::span<std::byte> codes, std::span<std::byte> scales,
                   PlePageBuffer& page_storage, HostWorkerPool& workers,
                   std::size_t queue_depth) {
    if (!table.direct_reader || !table.direct_reader.supported()) {
        throw std::runtime_error("PLE artifact direct I/O is unavailable");
    }
    constexpr std::size_t page = artifact::Reader::direct_io_alignment;
    if (reinterpret_cast<std::uintptr_t>(page_storage.data()) % page != 0 ||
        page_storage.size() < queue_depth * page) {
        throw std::runtime_error("PLE direct staging is not page aligned");
    }
    std::map<std::uint64_t, std::vector<PageCopy>> pages;
    for (std::size_t token = 0; token < rows.size(); ++token) {
        for (std::size_t head = 0; head < 16; ++head) {
            const std::int64_t global = rows[token][head];
            if (global < 0) { throw std::out_of_range("PLE row must be non-negative"); }
            const PleRowAddress address = locate_ple_row(static_cast<std::uint64_t>(global));
            const PleShardView& shard = table.shards[address.shard];
            if (address.row >= shard.rows || shard.width != kPleRowWidth ||
                shard.groups_per_row != kPleRowWidth / 16) {
                throw std::out_of_range("PLE direct row is outside the shard geometry");
            }
            const std::size_t index = token * 16 + head;
            add_piece(pages, shard.code_absolute_offset + address.row * kPleCodesPerRow,
                      kPleCodesPerRow, index * kPleCodesPerRow);
            add_piece(pages, shard.scale_absolute_offset + address.row * kPleScalesPerRow,
                      kPleScalesPerRow, codes.size() + index * kPleScalesPerRow);
        }
    }
    std::vector<std::pair<std::uint64_t, std::vector<PageCopy>>> ordered(pages.begin(), pages.end());
    const auto read_started = PerfClock::now();
    auto* compact = codes.data();
    for (std::size_t begin = 0; begin < ordered.size(); begin += queue_depth) {
        const std::size_t count = std::min(queue_depth, ordered.size() - begin);
        PleReadBatch pending(count);
        for (std::size_t i = 0; i < count; ++i) {
            auto* destination = static_cast<std::byte*>(page_storage.data()) + i * page;
            const std::uint64_t offset = ordered[begin + i].first;
            pending.add(workers.submit([reader = table.direct_reader, offset, destination] {
                return reader.read(offset, std::span<std::byte>(destination, page));
            }));
        }
        for (std::size_t i = 0; i < count; ++i) {
            const std::size_t read = pending.get(i);
            auto* source = static_cast<const std::byte*>(page_storage.data()) + i * page;
            for (const PageCopy& copy : ordered[begin + i].second) {
                if (copy.page_offset + copy.bytes > read) {
                    throw std::runtime_error("PLE direct read ended before a requested row");
                }
                std::byte* destination = copy.destination < codes.size()
                    ? compact + copy.destination
                    : scales.data() + (copy.destination - codes.size());
                std::memcpy(destination, source + copy.page_offset, copy.bytes);
            }
        }
    }
    return DirectGatherResult{ordered.size(), perf_elapsed_us(read_started)};
}

} // namespace

PleIoPolicy parse_ple_io_policy(std::string_view mode, std::string_view strict) {
    PleIoPolicy policy;
    if (mode.empty() || mode == "mmap") { policy.mode = PleIoMode::Mmap; }
    else if (mode == "direct") { policy.mode = PleIoMode::Direct; }
    else if (mode == "auto") { policy.mode = PleIoMode::Auto; }
    else { throw std::invalid_argument("NINFER_V100_PLE_IO must be mmap, auto, or direct"); }
    if (strict == "1") { policy.strict_direct = true; }
    else if (!strict.empty() && strict != "0") {
        throw std::invalid_argument("NINFER_V100_PLE_STRICT_DIRECT must be 0 or 1");
    }
    if (policy.strict_direct && policy.mode != PleIoMode::Direct) {
        throw std::invalid_argument("strict PLE reads require NINFER_V100_PLE_IO=direct");
    }
    return policy;
}

PleIoPolicy ple_io_policy_from_environment() {
    const char* mode = std::getenv("NINFER_V100_PLE_IO");
    const char* strict = std::getenv("NINFER_V100_PLE_STRICT_DIRECT");
    return parse_ple_io_policy(mode ? mode : "", strict ? strict : "");
}

PleIoResult gather_ple_rows_storage(
    const PleTableView& table, std::span<const std::array<std::int64_t, 16>> rows,
    std::span<std::byte> codes, std::span<std::byte> scales,
    PlePageBuffer& pages, HostWorkerPool* workers, std::size_t queue_depth, PleIoPolicy policy) {
    if (queue_depth == 0 || queue_depth > 256 || codes.size() != rows.size() * 16 * 80 ||
        scales.size() != rows.size() * 16 * 20) {
        throw std::invalid_argument("invalid PLE storage batch capacity");
    }
    for (const auto& token : rows) {
        for (const auto global : token) {
            if (global < 0) { throw std::out_of_range("PLE row must be non-negative"); }
            const auto address = locate_ple_row(static_cast<std::uint64_t>(global));
            const auto& shard = table.shards[address.shard];
            if (address.row >= shard.rows || shard.width != kPleRowWidth ||
                shard.groups_per_row != 10 || shard.codes.size() < shard.rows * 80 ||
                shard.scales.size() < shard.rows * 20) {
                throw std::out_of_range("PLE row is outside valid shard storage");
            }
        }
    }
    bool fallback = false;
    if (policy.mode != PleIoMode::Mmap) {
        try {
            if (workers == nullptr) { throw std::runtime_error("PLE direct workers unavailable"); }
            const auto result = gather_direct(table, rows, codes, scales, pages, *workers, queue_depth);
            return {true, false, result.pages, result.read_us};
        } catch (...) {
            if (policy.strict_direct) { throw; }
            fallback = true;
        }
    }
    gather_ple_rows_compressed(table, rows, codes, scales);
    return {false, fallback, 0, 0.0};
}

} // namespace ninfer::targets::qwen3_8_flash_next::detail
