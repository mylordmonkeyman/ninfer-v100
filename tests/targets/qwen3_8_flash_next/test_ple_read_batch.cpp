#include "targets/qwen3_8_flash_next/impl/ple_read_batch.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <future>
#include <iostream>
#include <stdexcept>
#include <thread>

int main() {
    using ninfer::targets::qwen3_8_flash_next::detail::PleReadBatch;
    using namespace std::chrono_literals;

    for (std::size_t pages : {std::size_t{1}, std::size_t{64}, std::size_t{256}}) {
        ninfer::targets::qwen3_8_flash_next::detail::PlePageBuffer buffer(pages * 4096);
        if (reinterpret_cast<std::uintptr_t>(buffer.data()) % 4096 != 0 ||
            buffer.size() != pages * 4096) {
            std::cerr << "FAIL: PLE disk staging does not satisfy direct-read alignment\n";
            return 1;
        }
    }

    // The first page fails while another packaged task still owns the staging buffer. Recovery
    // must retain that buffer until the second writer completes and preserve the original error.
    std::promise<void> release;
    auto gate = release.get_future();
    std::atomic<int> staging{0};
    std::packaged_task<std::size_t()> writer([&] {
        gate.wait();
        staging.store(42);
        return std::size_t{4096};
    });
    auto pending = writer.get_future();
    std::thread io(std::move(writer));
    std::promise<std::size_t> failed;
    failed.set_exception(std::make_exception_ptr(std::runtime_error("page read failed")));
    auto first = failed.get_future();
    std::promise<void> entered;
    auto ready = entered.get_future();
    std::packaged_task<bool()> consumer([&] {
        try {
            PleReadBatch batch(2);
            batch.add(std::move(first));
            batch.add(std::move(pending));
            entered.set_value();
            (void)batch.get(0);
            return false;
        } catch (const std::runtime_error& error) {
            return std::string(error.what()) == "page read failed" && staging.load() == 42;
        }
    });
    auto recovered = consumer.get_future();
    std::thread owner(std::move(consumer));
    ready.wait();
    const bool drained_before_recovery = recovered.wait_for(50ms) == std::future_status::timeout;
    release.set_value();
    const bool original_error = recovered.get();
    owner.join();
    io.join();
    if (!drained_before_recovery || !original_error) {
        std::cerr << "FAIL: pending page writer survived recovery or its error was lost\n";
        return 1;
    }

    std::cout << "PASS: failed page batch drains staging writers before recovery\n";
}
