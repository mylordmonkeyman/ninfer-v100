#include "targets/qwen3_8_flash_next/impl/cpu_expert_pool.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <list>
#include <set>
#include <stdexcept>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace ninfer::targets::qwen3_8_flash_next::detail;
using Clock = std::chrono::steady_clock;
struct Mapping {
    int fd;
    std::size_t size;
    const std::byte* data;
    explicit Mapping(const char* path) {
        fd = open(path, O_RDONLY | O_CLOEXEC);
        if (fd < 0) { throw std::runtime_error("Cannot open model"); }
        struct stat st{};
        if (fstat(fd, &st)) { close(fd); throw std::runtime_error("Cannot stat model"); }
        size = st.st_size;
        auto ptr = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (ptr == MAP_FAILED) { close(fd); throw std::runtime_error("Cannot mmap model"); }
        data = static_cast<const std::byte*>(ptr);
        madvise(ptr, size, MADV_RANDOM);
    }
    ~Mapping() { munmap(const_cast<std::byte*>(data), size); close(fd); }
};
struct Bank {
    const std::byte* data{};
    int rows{}, columns{};
    Nvfp4ExpertMatrixView expert(int id) const {
        const std::size_t elements = std::size_t(rows) * columns;
        const std::size_t codes = 512 * elements / 2;
        const std::size_t scales = 512 * elements / 16;
        return {.codes=data + id * elements / 2,
                .scales=data + codes + id * elements / 16,
                .weight_scale_divisor=reinterpret_cast<const float*>(data + codes + scales) + id,
                .input_scale_divisor=1.0F, .rows=rows, .columns=columns};
    }
};
struct Record {
    std::uint32_t position, layer;
    std::array<std::uint16_t, 2560> input;
    std::array<std::int32_t, 10> ids;
    std::array<float, 10> alpha;
};
struct Batch {
    const Record* record;
    std::vector<int> misses;
};
double percentile(std::vector<double> values, double q) {
    if (values.empty()) { return 0; }
    std::sort(values.begin(), values.end());
    return values[std::size_t(std::ceil(q * values.size())) - 1];
}
int main(int argc, char** argv) {
    try {
        if (argc != 9) {
            throw std::invalid_argument("Usage: replay MODEL OFFSETS TRACE WORKERS SLOTS HOST_LAYERS TARGET_TOK_S REPEATS");
        }
        const unsigned workers = std::stoul(argv[4]);
        const int slots = std::stoi(argv[5]), layers = std::stoi(argv[6]);
        const double target = std::stod(argv[7]);
        const int repeats = std::stoi(argv[8]);
        if (slots < 0 || slots > 512 || layers < 1 || layers > 48 ||
            target <= 0 || !std::isfinite(target) || repeats < 1) {
            throw std::invalid_argument("Invalid gate scenario");
        }
        Mapping model(argv[1]);
        std::ifstream offsets(argv[2]);
        std::array<std::array<Bank, 2>, 48> banks{};
        for (auto& layer : banks) {
            for (int part = 0; part < 2; ++part) {
                std::uint64_t offset, bytes;
                const int rows = part == 0 ? 1280 : 2560, columns = part == 0 ? 2560 : 640;
                const std::uint64_t expected = std::uint64_t(512) * rows * columns * 9 / 16 + 2048;
                if (!(offsets >> offset >> bytes) || bytes != expected ||
                    offset > model.size || bytes > model.size - offset) {
                    throw std::runtime_error("Invalid model bank offsets");
                }
                layer[part] = {model.data + offset, rows, columns};
            }
        }
        std::ifstream trace(argv[3], std::ios::binary);
        char magic[8]{};
        trace.read(magic, 8);
        if (std::memcmp(magic, "FNCPU12\0", 8)) { throw std::runtime_error("Invalid trace header"); }
        std::vector<Record> records;
        for (;;) {
            Record r{};
            trace.read(reinterpret_cast<char*>(&r.position), 4);
            if (trace.eof() && trace.gcount() == 0) { break; }
            trace.read(reinterpret_cast<char*>(&r.layer), 4);
            trace.read(reinterpret_cast<char*>(r.input.data()), sizeof(r.input));
            trace.read(reinterpret_cast<char*>(r.ids.data()), sizeof(r.ids));
            trace.read(reinterpret_cast<char*>(r.alpha.data()), sizeof(r.alpha));
            if (!trace || r.layer != records.size() % 48 || r.position != records.size() / 48) {
                throw std::runtime_error("Truncated, noncontiguous or incomplete trace");
            }
            std::set<int> ids;
            for (int i = 0; i < 10; ++i) {
                if (r.ids[i] < 0 || r.ids[i] >= 512 || !ids.insert(r.ids[i]).second ||
                    !std::isfinite(r.alpha[i])) { throw std::runtime_error("Invalid routed expert"); }
            }
            records.push_back(r);
        }
        if (records.size() < 48 * 128 || records.size() % 48) {
            throw std::runtime_error("Replay requires >=128 complete natural-routing positions");
        }
        const auto pair = [&](int layer, int id) {
            return HostNvfp4ExpertPairView{banks[layer][0].expert(id), banks[layer][1].expert(id)};
        };
        // Independent scalar mathematical evaluation at represented BF16 boundaries.
        // Spread checks across the complete routed layer range and trace prefix.
        CpuNvfp4ExpertReferenceScratch scratch;
        std::array<float, 2560> reference{}, actual{};
        double max_nrmse = 0, min_cosine = 1;
        for (std::size_t i = 0; i < 48; ++i) {
            const auto& r = records[(i * (records.size() / 48) / 48) * 48 + i];
            const auto expert = pair(r.layer, r.ids[i % 10]);
            flash_next_cpu_nvfp4_expert_pair_reference(expert, r.input, reference, scratch);
            flash_next_cpu_nvfp4_expert_pair_avx2(expert, r.input, actual, scratch);
            double error = 0, rr = 0, aa = 0, dot = 0;
            for (int j = 0; j < 2560; ++j) {
                if (!std::isfinite(reference[j]) || !std::isfinite(actual[j])) {
                    throw std::runtime_error("Nonfinite expert output");
                }
                error += std::pow(double(actual[j]) - reference[j], 2);
                rr += double(reference[j]) * reference[j];
                aa += double(actual[j]) * actual[j];
                dot += double(reference[j]) * actual[j];
            }
            const double nrmse = std::sqrt(error / std::max(rr, 1e-30));
            const double cosine = rr == 0 && aa == 0 ? 1 : dot / std::sqrt(std::max(rr * aa, 1e-30));
            max_nrmse = std::max(max_nrmse, nrmse); min_cosine = std::min(min_cosine, cosine);
            if (nrmse > .002 || cosine < .99999) { throw std::runtime_error("AVX2 expert numerical gate failed"); }
        }
        // Provisional uniform LRU, admission cap one per layer/token. All misses
        // are determined before admission; no simulated hit executes on the CPU.
        std::array<std::list<int>, 48> cache;
        std::vector<Batch> batches;
        std::set<std::pair<int,int>> working_set;
        std::size_t misses = 0, hits = 0;
        for (const auto& r : records) {
            if (int(r.layer) >= layers) { continue; } // Explicit first-L scenario.
            Batch b{&r, {}};
            auto& lru = cache[r.layer];
            for (int id : r.ids) {
                auto found = std::find(lru.begin(), lru.end(), id);
                if (found == lru.end()) {
                    b.misses.push_back(id); working_set.emplace(r.layer, id); ++misses;
                } else {
                    lru.splice(lru.begin(), lru, found); ++hits;
                }
            }
            if (slots && !b.misses.empty()) {
                lru.push_front(b.misses.front());
                if (int(lru.size()) > slots) { lru.pop_back(); }
            }
            batches.push_back(std::move(b));
        }
        if (working_set.size() * 2764808ULL < 256ULL * 1024 * 1024) {
            throw std::runtime_error("Routed miss working set must exceed 256 MiB");
        }
        HostExpertWorkerPool pool(workers, true);
        std::array<float, 2560 * 10> output{};
        std::array<float, 2560> merged{};
        std::vector<HostExpertTask> tasks;
        const auto execute = [&](const Batch& b) {
            tasks.clear();
            for (std::size_t i = 0; i < b.misses.size(); ++i) {
                tasks.push_back({pair(b.record->layer, b.misses[i]), b.record->input.data(),
                    nullptr, output.data() + i * 2560});
            }
            pool.run(tasks);
            merged.fill(0);
            for (std::size_t i = 0; i < tasks.size(); ++i) {
                const auto at = std::find(b.record->ids.begin(), b.record->ids.end(), b.misses[i]);
                const float alpha = b.record->alpha[at - b.record->ids.begin()];
                for (int j = 0; j < 2560; ++j) { merged[j] += alpha * output[i * 2560 + j]; }
            }
        };
        // Warm actual miss bytes and workers, never warm a single repeated expert.
        for (const auto& b : batches) { execute(b); }
        // File-backed page placement may survive an earlier process. Record
        // actual residency; process binding alone does not prove NUMA locality.
        std::ifstream numa("/proc/self/numa_maps");
        std::string numa_line;
        while (std::getline(numa, numa_line)) {
            if (numa_line.find("file=") != std::string::npos &&
                numa_line.find(argv[1]) != std::string::npos) {
                std::cerr << "model_numa_residency " << numa_line << '\n';
            }
        }
        std::vector<double> latency;
        latency.reserve(batches.size() * repeats);
        double checksum = 0;
        const auto started = Clock::now();
        for (int repeat = 0; repeat < repeats; ++repeat) {
            for (const auto& b : batches) {
                const auto batch_start = Clock::now();
                execute(b);
                latency.push_back(std::chrono::duration<double, std::micro>(Clock::now() - batch_start).count());
                checksum += merged[0];
            }
        }
        const double seconds = std::chrono::duration<double>(Clock::now() - started).count();
        const double pairs_s = misses * repeats / seconds;
        const double misses_token = double(misses) / (records.size() / 48);
        const double required = target * misses_token;
        std::cout.precision(10);
        std::cout << "{\"workers\":" << workers << ",\"slots_per_layer\":" << slots
          << ",\"host_layers\":" << layers << ",\"host_layer_selection\":\"first_L\""
          << ",\"admission_cap\":1,\"positions\":" << records.size()/48
          << ",\"miss_pairs\":" << misses << ",\"hit_rate\":" << double(hits)/(hits+misses)
          << ",\"working_set_bytes\":" << working_set.size()*2764808ULL
          << ",\"repeats\":" << repeats << ",\"seconds\":" << seconds
          << ",\"pairs_per_s\":" << pairs_s
          << ",\"compact_GiB_per_s\":" << pairs_s*2764808/std::pow(2.,30)
          << ",\"useful_GFLOP_per_s\":" << pairs_s*9830400/1e9
          << ",\"us_per_pair\":" << 1e6/pairs_s
          << ",\"batch_p50_us\":" << percentile(latency,.50)
          << ",\"batch_p95_us\":" << percentile(latency,.95)
          << ",\"batch_p99_us\":" << percentile(latency,.99)
          << ",\"misses_per_token\":" << misses_token
          << ",\"target_tokens_per_s\":" << target
          << ",\"required_pairs_per_s\":" << required
          << ",\"headroom_ratio\":" << pairs_s/required
          << ",\"minimum_pass\":" << (pairs_s>=required?"true":"false")
          << ",\"preferred_pass\":" << (pairs_s>=1.3*required?"true":"false")
          << ",\"expert_max_nrmse\":" << max_nrmse << ",\"expert_min_cosine\":" << min_cosine
          << ",\"checksum\":" << checksum << "}\n";
        // A failed architecture scenario is evidence, not a benchmark malfunction.
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
