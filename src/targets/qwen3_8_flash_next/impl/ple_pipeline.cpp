#include "targets/qwen3_8_flash_next/impl/ple_pipeline.h"

#include "core/arena.h"
#include "targets/qwen3_8_flash_next/impl/perf_telemetry.h"
#include "targets/qwen3_8_flash_next/impl/ple_decode_kernels.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <future>
#include <map>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace ninfer::targets::qwen3_8_flash_next::detail {
namespace {

constexpr std::size_t kPleOutputWidth          = 16 * kPleRowWidth;
constexpr std::size_t kPleTokenBytes           = kPleOutputWidth * sizeof(std::uint16_t);
constexpr std::size_t kPleCodesPerRow          = 80;
constexpr std::size_t kPleScalesPerRow         = 20;
constexpr std::size_t kPleCompressedRowBytes   = kPleCodesPerRow + kPleScalesPerRow; // 100
constexpr std::size_t kPleCompressedTokenBytes = 16 * kPleCompressedRowBytes;       // 1600

enum class SlotState : std::uint8_t { Idle, Gathering, Copying };

} // namespace

struct PleGatherPipeline::Slot {
    Slot(DeviceContext& device, std::size_t host_bytes, std::size_t dev_bytes,
         std::size_t direct_bytes)
        : buffer(host_bytes), direct_pages(direct_bytes), device_compressed(dev_bytes),
          completion(device), ordering(device) {}

    PinnedHostBuffer buffer;
    PinnedHostBuffer direct_pages;
    DeviceBuffer device_compressed;
    CudaCompletionEvent completion;
    // Orders the H2D copy after everything already enqueued on the compute stream: the startup
    // zero-fill of the round tensors and the previous round's consumers of `output`. Without it
    // the transfer-stream copy can land before an in-flight memset (WAW) or before the previous
    // PLE layer has read the buffer (WAR).
    CudaCompletionEvent ordering;
    std::vector<std::future<void>> work;
    std::uint64_t generation = 0;
    SlotState state          = SlotState::Idle;
    bool telemetry = false;
    PerfClock::time_point gather_started{};
    std::string_view storage_backend = "mmap";
    std::optional<double> page_read_us;
    std::size_t coalesced_pages = 0;
    bool storage_fallback = false;
};

PleGatherPipeline::Ticket::Ticket(PleGatherPipeline* owner, std::size_t slot,
                                  std::uint64_t generation, std::size_t tokens) noexcept
    : owner_(owner), slot_(slot), generation_(generation), tokens_(tokens) {}

PleGatherPipeline::Ticket::~Ticket() {
    if (owner_ != nullptr) { owner_->abandon(*this); }
}

PleGatherPipeline::Ticket::Ticket(Ticket&& other) noexcept
    : owner_(std::exchange(other.owner_, nullptr)), slot_(other.slot_),
      generation_(other.generation_), tokens_(other.tokens_) {}

PleGatherPipeline::Ticket& PleGatherPipeline::Ticket::operator=(Ticket&& other) noexcept {
    if (this == &other) { return *this; }
    if (owner_ != nullptr) { owner_->abandon(*this); }
    owner_      = std::exchange(other.owner_, nullptr);
    slot_       = other.slot_;
    generation_ = other.generation_;
    tokens_     = other.tokens_;
    return *this;
}

namespace {

std::size_t ple_direct_queue_depth() {
    const char* raw = std::getenv("NINFER_V100_PLE_QUEUE_DEPTH");
    if (raw == nullptr || raw[0] == '\0') { return 64; }
    char* end = nullptr;
    const unsigned long value = std::strtoul(raw, &end, 10);
    if (end == raw || *end != '\0' || value == 0 || value > 256) {
        throw std::invalid_argument("NINFER_V100_PLE_QUEUE_DEPTH must be in [1,256]");
    }
    return static_cast<std::size_t>(value);
}

PleGatherPipeline::StorageMode ple_storage_mode() {
    const char* raw = std::getenv("NINFER_V100_PLE_STORAGE");
    const std::string_view mode = raw == nullptr ? std::string_view{} : std::string_view(raw);
    if (mode.empty() || mode == "mmap") { return PleGatherPipeline::StorageMode::Mmap; }
    if (mode == "auto") { return PleGatherPipeline::StorageMode::Auto; }
    if (mode == "direct") { return PleGatherPipeline::StorageMode::Direct; }
    throw std::invalid_argument("NINFER_V100_PLE_STORAGE must be mmap, auto, or direct");
}

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
                   PinnedHostBuffer& page_storage, HostWorkerPool& workers,
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
        std::vector<std::future<std::size_t>> pending;
        pending.reserve(count);
        for (std::size_t i = 0; i < count; ++i) {
            auto* destination = static_cast<std::byte*>(page_storage.data()) + i * page;
            const std::uint64_t offset = ordered[begin + i].first;
            pending.push_back(workers.submit([reader = table.direct_reader, offset, destination] {
                return reader.read(offset, std::span<std::byte>(destination, page));
            }));
        }
        for (std::size_t i = 0; i < count; ++i) {
            const std::size_t read = pending[i].get();
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

PleGatherPipeline::PleGatherPipeline(PleTableView table, DeviceContext& device,
                                     std::size_t max_tokens, std::size_t slot_count,
                                     std::uint32_t worker_threads)
    : table_(std::move(table)), device_(device), max_tokens_(max_tokens),
      workers_(worker_threads, slot_count * max_tokens),
      direct_queue_depth_(ple_direct_queue_depth()), storage_mode_(ple_storage_mode()),
      fixed_host_buffer_(max_tokens * kPleTokenBytes) {
    if (max_tokens == 0 || slot_count == 0) {
        throw std::invalid_argument("PLE gather pipeline capacity must be nonzero");
    }
    if (storage_mode_ != StorageMode::Mmap) {
        direct_workers_ = std::make_unique<HostWorkerPool>(
            std::min<std::uint32_t>(8, static_cast<std::uint32_t>(direct_queue_depth_)),
            direct_queue_depth_);
    }
    slots_.reserve(slot_count);
    for (std::size_t slot = 0; slot < slot_count; ++slot) {
        slots_.push_back(std::make_unique<Slot>(
            device_, max_tokens_ * kPleCompressedTokenBytes,
            max_tokens_ * kPleCompressedTokenBytes,
            direct_queue_depth_ * artifact::Reader::direct_io_alignment));
    }
}

PleGatherPipeline::~PleGatherPipeline() {
    for (const std::unique_ptr<Slot>& slot : slots_) {
        for (std::future<void>& work : slot->work) {
            if (work.valid()) {
                try {
                    work.get();
                } catch (...) {}
            }
        }
        if (slot->state == SlotState::Copying) {
            try {
                slot->completion.synchronize();
            } catch (...) {}
        }
    }
}

std::size_t PleGatherPipeline::acquire_slot() {
    for (std::size_t offset = 0; offset < slots_.size(); ++offset) {
        const std::size_t index = (next_slot_ + offset) % slots_.size();
        Slot& slot              = *slots_[index];
        if (slot.state == SlotState::Copying && slot.completion.ready()) {
            slot.state = SlotState::Idle;
        }
        if (slot.state == SlotState::Idle) {
            next_slot_ = (index + 1) % slots_.size();
            return index;
        }
    }
    throw std::runtime_error("PLE gather pipeline has no reusable slot");
}

PleGatherPipeline::Ticket
PleGatherPipeline::prepare(std::span<const std::array<std::int64_t, 16>> global_rows) {
    Ticket ticket = prepare_async(global_rows);
    Slot& slot = *slots_[ticket.slot_];
    for (std::future<void>& work : slot.work) { work.get(); }
    slot.work.clear();
    return ticket;
}

PleGatherPipeline::Ticket
PleGatherPipeline::prepare_async(std::span<const std::array<std::int64_t, 16>> global_rows) {
    if (global_rows.empty() || global_rows.size() > max_tokens_) {
        throw std::invalid_argument("PLE gather batch is outside the startup-fixed capacity");
    }
    const std::size_t slot_index = acquire_slot();
    Slot& slot                   = *slots_[slot_index];
    slot.state                   = SlotState::Gathering;
    ++slot.generation;
    slot.work.clear();
    slot.telemetry = v100_perf_telemetry_enabled();
    slot.gather_started = slot.telemetry ? PerfClock::now() : PerfClock::time_point{};
    slot.storage_backend = "mmap";
    slot.page_read_us.reset();
    slot.coalesced_pages = 0;
    slot.storage_fallback = false;

    const std::size_t tokens       = global_rows.size();
    auto* base                     = static_cast<std::byte*>(slot.buffer.data());
    const std::size_t codes_bytes  = tokens * 16 * kPleCodesPerRow;
    const std::size_t scales_bytes = tokens * 16 * kPleScalesPerRow;

    std::vector<std::array<std::int64_t, 16>> owned(global_rows.begin(), global_rows.end());
    slot.work.push_back(workers_.submit(
        [this, &slot, owned = std::move(owned), base, codes_bytes, scales_bytes, tokens] {
            auto codes = std::span<std::byte>(base, codes_bytes);
            auto scales = std::span<std::byte>(base + codes_bytes, scales_bytes);
            bool direct = storage_mode_ != StorageMode::Mmap;
            if (direct) {
                try {
                    const DirectGatherResult result = gather_direct(
                        table_, owned, codes, scales, slot.direct_pages,
                        *direct_workers_, direct_queue_depth_);
                    slot.storage_backend = "direct";
                    slot.page_read_us = result.read_us;
                    slot.coalesced_pages = result.pages;
                } catch (...) {
                    if (storage_mode_ == StorageMode::Direct) { throw; }
                    direct = false;
                    slot.storage_fallback = true;
                }
            }
            if (!direct) { gather_ple_rows_compressed(table_, owned, codes, scales); }
        }));
    return Ticket(this, slot_index, slot.generation, tokens);
}

void PleGatherPipeline::abandon(Ticket& ticket) noexcept {
    if (ticket.owner_ != this || ticket.slot_ >= slots_.size()) {
        ticket.owner_ = nullptr;
        return;
    }
    Slot& slot = *slots_[ticket.slot_];
    if (slot.generation == ticket.generation_) {
        for (std::future<void>& work : slot.work) {
            if (work.valid()) {
                try { work.get(); } catch (...) {}
            }
        }
        slot.work.clear();
        if (slot.state == SlotState::Copying) {
            try { slot.completion.synchronize(); } catch (...) {}
        }
        slot.state = SlotState::Idle;
    }
    ticket.owner_ = nullptr;
}

void PleGatherPipeline::enqueue_copy(Ticket&& ticket, Tensor& output) {
    if (ticket.owner_ != this || ticket.slot_ >= slots_.size()) {
        throw std::invalid_argument("PLE gather ticket belongs to another pipeline");
    }
    Slot& slot = *slots_[ticket.slot_];
    if (slot.state != SlotState::Gathering || slot.generation != ticket.generation_) {
        throw std::invalid_argument("PLE gather ticket is stale");
    }
    if (output.dtype != DType::BF16 || output.ne[0] != static_cast<std::int32_t>(kPleOutputWidth) ||
        output.ne[1] != static_cast<std::int32_t>(ticket.tokens_) || output.ne[2] != 1 ||
        output.ne[3] != 1 || !output.is_contiguous() || output.data == nullptr) {
        throw std::invalid_argument("PLE gather output must be contiguous BF16 [2560,tokens]");
    }

    try {
        for (std::future<void>& work : slot.work) { work.get(); }
    } catch (...) {
        slot.work.clear();
        slot.state    = SlotState::Idle;
        ticket.owner_ = nullptr;
        throw;
    }
    slot.work.clear();
    if (slot.telemetry) {
        emit_ple_perf(ticket.tokens_, ticket.tokens_ * kPleCompressedTokenBytes,
                      perf_elapsed_us(slot.gather_started), true, slot.storage_backend,
                      slot.page_read_us, slot.coalesced_pages, slot.storage_fallback);
    }
    const std::size_t bytes = ticket.tokens_ * kPleCompressedTokenBytes;
    // Payload H2D of gathered PLE rows; not a host control-flow round-trip.
    slot.ordering.record(device_.stream);
    slot.ordering.wait(device_.transfer_stream);
    CUDA_CHECK(cudaMemcpyAsync(slot.device_compressed.p, slot.buffer.data(), bytes,
                               cudaMemcpyHostToDevice, device_.transfer_stream));
    slot.completion.record(device_.transfer_stream);
    slot.completion.wait(device_.stream);
    flash_next_ple_dequant_launch(slot.device_compressed.p, output,
                                  static_cast<int>(ticket.tokens_), device_.stream);
    slot.state    = SlotState::Copying;
    ticket.owner_ = nullptr;
}

void PleGatherPipeline::gather_pinned(
    std::span<const std::array<std::int64_t, 16>> global_rows) {
    if (global_rows.empty() || global_rows.size() > max_tokens_) {
        throw std::invalid_argument("PLE gather batch is outside the startup-fixed capacity");
    }
    auto* output = static_cast<std::uint16_t*>(fixed_host_buffer_.data());
    const bool telemetry = v100_perf_telemetry_enabled();
    const auto started = telemetry ? PerfClock::now() : PerfClock::time_point{};

    // Served decode runs at B=1, and this gather sits in the inter-round window that nsys
    // measured at 0.99 ms/token of GPU idle (~1.2 host gaps >100 us per token). At that batch the
    // HostWorkerPool round-trip is two Windows condvar wake-ups (submit notify, then the future
    // wait) to buy ~2560 scalar dequants of parallelism that has nowhere to go. Below the
    // threshold the calling thread does the work itself.
    // Bitwise identity: same gather_ple_rows_bf16 over the same rows into the same
    // token * kPleOutputWidth offsets, so the pinned bytes are unchanged either way. Only the
    // thread that produces them differs.
    constexpr std::size_t kInlineGatherTokens = 2;
    if (global_rows.size() <= kInlineGatherTokens) {
        for (std::size_t token = 0; token < global_rows.size(); ++token) {
            gather_ple_rows_bf16(
                table_, global_rows[token],
                std::span<std::uint16_t>(output + token * kPleOutputWidth, kPleOutputWidth));
        }
        if (telemetry) emit_ple_perf(global_rows.size(), global_rows.size()*kPleTokenBytes, perf_elapsed_us(started), false);
        return;
    }

    std::vector<std::future<void>> work;
    work.reserve(global_rows.size());
    for (std::size_t token = 0; token < global_rows.size(); ++token) {
        const std::array<std::int64_t, 16> rows = global_rows[token];
        work.push_back(workers_.submit([this, output, rows, token] {
            gather_ple_rows_bf16(
                table_, rows,
                std::span<std::uint16_t>(output + token * kPleOutputWidth, kPleOutputWidth));
        }));
    }
    for (std::future<void>& w : work) {
        w.get();
    }
    if (telemetry) emit_ple_perf(global_rows.size(), global_rows.size()*kPleTokenBytes, perf_elapsed_us(started), false);
}

} // namespace ninfer::targets::qwen3_8_flash_next::detail
