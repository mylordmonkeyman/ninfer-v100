#include "targets/qwen3_8_flash_next/impl/route_handoff.h"

#include <algorithm>
#include <iostream>
#include <vector>

using namespace ninfer;
using namespace ninfer::targets::qwen3_8_flash_next::detail;

__global__ void unrelated_work(unsigned long long cycles) {
    const auto started = clock64();
    while (clock64() - started < cycles) { }
}

template<class F> void rejected(F&& f) {
    bool threw = false;
    try { f(); } catch (const std::invalid_argument&) { threw = true; }
    if (!threw) throw std::runtime_error("unsafe route handoff state accepted");
}

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return 77;
    DeviceContext device(0);
    FlashNextRouteHandoff handoff;
    cudaEvent_t shared_done;
    CUDA_CHECK(cudaEventCreateWithFlags(&shared_done, cudaEventDisableTiming));
    std::uint64_t previous = 0;
    for (unsigned tokens : {1U, 2U, 4U, 512U, 2U, 2048U, 1U}) {
        std::vector<std::uint16_t> input(tokens * 2560);
        std::vector<std::int32_t> ids(tokens * 10);
        for (unsigned i = 0; i < input.size(); ++i) input[i] = (i + tokens * 17) % 65536;
        for (unsigned i = 0; i < ids.size(); ++i) ids[i] = (i * 13 + tokens) % 512;
        DeviceBuffer d_input(input.size() * sizeof(input[0]));
        DeviceBuffer d_ids(ids.size() * sizeof(ids[0]));
        handoff.prepare(input.size(), ids.size(), device.stream);
        CUDA_CHECK(cudaMemcpyAsync(d_input.p, input.data(), d_input.bytes,
                                   cudaMemcpyHostToDevice, device.stream));
        CUDA_CHECK(cudaMemcpyAsync(d_ids.p, ids.data(), d_ids.bytes,
                                   cudaMemcpyHostToDevice, device.stream));
        const auto ticket = handoff.submit(d_input.p, d_ids.p, device.stream);
        if (ticket <= previous) throw std::runtime_error("route ticket did not advance");
        rejected([&] { (void)handoff.input(ticket); });
        rejected([&] { handoff.wait(previous); });
        rejected([&] { handoff.prepare(1, 1, device.stream); });
        if (previous == 0) {
            // One SM stays busy after routing. Route data must reach the host
            // before this unrelated compute work completes on the same stream.
            unrelated_work<<<1, 1, 0, device.stream>>>(
                static_cast<unsigned long long>(device.props.clockRate) * 500ULL);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(shared_done, device.stream));
        }
        handoff.wait(ticket);
        if (previous == 0 && cudaEventQuery(shared_done) != cudaErrorNotReady) {
            throw std::runtime_error("route readiness waited for unrelated compute work");
        }
        if (!std::equal(input.begin(), input.end(), handoff.input(ticket).begin()) ||
            !std::equal(ids.begin(), ids.end(), handoff.ids(ticket).begin())) {
            throw std::runtime_error("route handoff changed represented input or ids");
        }
        rejected([&] { handoff.wait(ticket); });
        CUDA_CHECK(cudaStreamSynchronize(device.stream));
        previous = ticket;
    }
    // Discarded/error-path transfers are drained before storage can grow/reuse.
    DeviceBuffer data(64);
    CUDA_CHECK(cudaMemsetAsync(data.p, 0, data.bytes, device.stream));
    handoff.prepare(2, 2, device.stream);
    (void)handoff.submit(data.p, data.p, device.stream);
    handoff.drain();
    handoff.prepare(4, 4, device.stream);
    rejected([&] { (void)handoff.ids(previous); });
    const auto recovered = handoff.submit(data.p, data.p, device.stream);
    handoff.wait(recovered);
    if (handoff.ids(recovered)[0] != 0) throw std::runtime_error("route drain recovery failed");
    CUDA_CHECK(cudaStreamBeginCapture(device.stream, cudaStreamCaptureModeThreadLocal));
    rejected([&] { handoff.prepare(1, 1, device.stream); });
    cudaGraph_t graph = nullptr;
    CUDA_CHECK(cudaStreamEndCapture(device.stream, &graph));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaEventDestroy(shared_done));
    std::cout << "PASS: route data exact, early readiness, growth/reuse, stale tickets, drain and capture rejection\n";
}
