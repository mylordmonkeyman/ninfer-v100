#pragma once

#include <cstddef>
#include <future>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen3_8_flash_next::detail {

// Disk pages are scattered into the pinned compact payload; these pages never participate in
// H2D themselves. cudaHostAlloc does not guarantee the alignment required by O_DIRECT.
class PlePageBuffer {
public:
    static constexpr std::size_t alignment = 4096;
    explicit PlePageBuffer(std::size_t bytes)
        : storage_(bytes + alignment - 1), bytes_(bytes) {
        data_ = storage_.data();
        std::size_t available = storage_.size();
        if (std::align(alignment, bytes, data_, available) == nullptr) {
            throw std::runtime_error("unable to align PLE page storage");
        }
    }
    PlePageBuffer(const PlePageBuffer&) = delete;
    PlePageBuffer& operator=(const PlePageBuffer&) = delete;
    void* data() noexcept { return data_; }
    std::size_t size() const noexcept { return bytes_; }

private:
    std::vector<std::byte> storage_;
    void* data_ = nullptr;
    std::size_t bytes_;
};

// Packaged-task futures do not wait on destruction. Keep every page writer alive until it has
// completed, including when submission, a read, or scattering an earlier page fails.
class PleReadBatch {
public:
    explicit PleReadBatch(std::size_t count) { reads_.reserve(count); }
    ~PleReadBatch() {
        for (auto& read : reads_) {
            if (read.valid()) { read.wait(); }
        }
    }
    PleReadBatch(const PleReadBatch&) = delete;
    PleReadBatch& operator=(const PleReadBatch&) = delete;

    void add(std::future<std::size_t> read) {
        try {
            reads_.push_back(std::move(read));
        } catch (...) {
            if (read.valid()) { read.wait(); }
            throw;
        }
    }
    std::size_t get(std::size_t index) { return reads_.at(index).get(); }

private:
    std::vector<std::future<std::size_t>> reads_;
};

} // namespace ninfer::targets::qwen3_8_flash_next::detail
