#include "targets/qwen3_8_flash_next/impl/cpu_expert_pool.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>
using namespace ninfer::targets::qwen3_8_flash_next::detail;
struct Matrix {
    std::vector<std::byte> codes, scales;
    float divisor = 1;
    int rows, columns;
    Matrix(int r, int c) : codes(std::size_t(r)*c/2), scales(std::size_t(r)*c/16, std::byte{0x38}),
                          rows(r), columns(c) {
        for (int row = 0; row < r; ++row) {
            std::fill_n(codes.data() + std::size_t(row)*c/2, 8, std::byte{0x22});
        }
    }
    Nvfp4ExpertMatrixView view() {
        return {codes.data(), scales.data(), &divisor, 1, rows, columns};
    }
};
int main() {
    try {
        Matrix gate(1280,2560), down(2560,640);
        HostNvfp4ExpertPairView expert{gate.view(), down.view()};
        std::array<std::uint16_t,2560> input;
        input.fill(0x3f80);
        const bool avx2 = flash_next_cpu_nvfp4_avx2_available();
        HostExpertWorkerPool pool(4, avx2);
        auto exercise = [&] {
            std::array<float,2560*3> output{};
            std::array<HostExpertTask,3> tasks;
            for (int i=0; i<3; ++i) { tasks[i]={expert,input.data(),nullptr,output.data()+2560*i}; }
            for (int round=0; round<8; ++round) {
                pool.run(tasks);
                for (float value : output) {
                    if (value != 4096) { throw std::runtime_error("Pool changed expert outputs/order"); }
                }
            }
        };
        exercise();
        // A worker failure must be delivered to the caller, and the pool must
        // still be usable for a later real batch.
        std::array<float,2560> result{};
        HostExpertTask bad{expert,input.data(),nullptr,result.data()};
        bad.expert.gate_up.codes=nullptr;
        bool rejected=false;
        try { pool.run(std::span(&bad,1)); } catch(const std::invalid_argument&) { rejected=true; }
        if (!rejected) { throw std::runtime_error("Worker exception was lost"); }
        pool.run({});
        std::exception_ptr errors[2];
        std::thread a([&]{try { exercise(); } catch(...) { errors[0]=std::current_exception(); }});
        std::thread b([&]{try { exercise(); } catch(...) { errors[1]=std::current_exception(); }});
        a.join(); b.join();
        for (auto& error : errors) { if (error) { std::rethrow_exception(error); } }
        std::cout << "PASS: expert values, repeated rendezvous, concurrent callers and error recovery\n";
    } catch(const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
