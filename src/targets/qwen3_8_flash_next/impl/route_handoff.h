#pragma once

#include "core/arena.h"
#include "core/device.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>

namespace ninfer::targets::qwen3_8_flash_next::detail {

// Persistent route/control transfer resources for eager host-backed MoE.
// A blocking completion event is the ready indicator: it yields the host
// thread and cannot expose a previous layer's completion as a new ticket.
// Host expert execution is eager; graph capture is explicitly rejected.
class FlashNextRouteHandoff {
public:
    FlashNextRouteHandoff() {
        CUDA_CHECK(cudaStreamCreateWithFlags(&transfer_, cudaStreamNonBlocking));
        try {
            CUDA_CHECK(cudaEventCreateWithFlags(&router_, cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(
                &ready_, cudaEventDisableTiming | cudaEventBlockingSync));
        } catch (...) {
            if (router_) cudaEventDestroy(router_);
            cudaStreamDestroy(transfer_);
            throw;
        }
    }
    ~FlashNextRouteHandoff() {
        // Buffers and events must outlive even a partially submitted transfer.
        cudaStreamSynchronize(transfer_);
        cudaEventDestroy(ready_);
        cudaEventDestroy(router_);
        cudaStreamDestroy(transfer_);
    }
    FlashNextRouteHandoff(const FlashNextRouteHandoff&) = delete;
    FlashNextRouteHandoff& operator=(const FlashNextRouteHandoff&) = delete;

    void prepare(std::size_t input_words, std::size_t routes, cudaStream_t compute) {
        cudaStreamCaptureStatus capture;
        CUDA_CHECK(cudaStreamIsCapturing(compute, &capture));
        if (capture != cudaStreamCaptureStatusNone) {
            throw std::invalid_argument("route handoff requires eager host-backed execution");
        }
        if (pending_ || input_words == 0 || routes == 0 ||
            input_words > std::numeric_limits<std::size_t>::max() / sizeof(std::uint16_t) ||
            routes > std::numeric_limits<std::size_t>::max() / sizeof(std::int32_t)) {
            throw std::invalid_argument("invalid or outstanding route handoff shape");
        }
        completed_ = 0;
        grow(input_, input_words * sizeof(std::uint16_t));
        grow(ids_, routes * sizeof(std::int32_t));
        input_words_ = input_words;
        routes_ = routes;
    }
    std::uint64_t submit(const void* input, const void* ids, cudaStream_t compute) {
        if (pending_ || !input_ || input == nullptr || ids == nullptr ||
            sequence_ == std::numeric_limits<std::uint64_t>::max()) {
            throw std::invalid_argument("invalid route handoff submission");
        }
        cudaStreamCaptureStatus capture;
        CUDA_CHECK(cudaStreamIsCapturing(compute, &capture));
        if (capture != cudaStreamCaptureStatusNone) {
            throw std::invalid_argument("route handoff requires eager host-backed execution");
        }
        pending_ = true;
        ++sequence_;
        try {
            // Record immediately after routing, before shared-expert kernels.
            CUDA_CHECK(cudaEventRecord(router_, compute));
            CUDA_CHECK(cudaStreamWaitEvent(transfer_, router_, 0));
            CUDA_CHECK(cudaMemcpyAsync(input_->data(), input,
                input_words_ * sizeof(std::uint16_t), cudaMemcpyDeviceToHost, transfer_));
            CUDA_CHECK(cudaMemcpyAsync(ids_->data(), ids,
                routes_ * sizeof(std::int32_t), cudaMemcpyDeviceToHost, transfer_));
            CUDA_CHECK(cudaEventRecord(ready_, transfer_));
        } catch (...) {
            cudaStreamSynchronize(transfer_);
            pending_ = false;
            throw;
        }
        return sequence_;
    }
    void drain() noexcept {
        if (pending_) {
            const auto status = cudaEventSynchronize(ready_);
            completed_ = status == cudaSuccess ? sequence_ : 0;
            pending_ = false;
        }
    }
    void wait(std::uint64_t ticket) {
        if (!pending_ || ticket != sequence_) {
            throw std::invalid_argument("stale route handoff ticket");
        }
        CUDA_CHECK(cudaEventSynchronize(ready_));
        completed_ = ticket;
        pending_ = false;
    }
    std::span<const std::uint16_t> input(std::uint64_t ticket) const {
        check_ready(ticket);
        return {static_cast<const std::uint16_t*>(input_->data()), input_words_};
    }
    std::span<const std::int32_t> ids(std::uint64_t ticket) const {
        check_ready(ticket);
        return {static_cast<const std::int32_t*>(ids_->data()), routes_};
    }
private:
    static void grow(std::unique_ptr<PinnedHostBuffer>& buffer, std::size_t bytes) {
        if (!buffer || buffer->size() < bytes) {
            buffer = std::make_unique<PinnedHostBuffer>(bytes);
        }
    }
    void check_ready(std::uint64_t ticket) const {
        if (pending_ || ticket == 0 || ticket != sequence_ || completed_ != ticket) {
            throw std::invalid_argument("route data is not ready for this ticket");
        }
    }
    cudaStream_t transfer_ = nullptr;
    cudaEvent_t router_ = nullptr, ready_ = nullptr;
    std::unique_ptr<PinnedHostBuffer> input_, ids_;
    std::size_t input_words_ = 0, routes_ = 0;
    std::uint64_t sequence_ = 0, completed_ = 0;
    bool pending_ = false;
};
} // namespace ninfer::targets::qwen3_8_flash_next::detail
