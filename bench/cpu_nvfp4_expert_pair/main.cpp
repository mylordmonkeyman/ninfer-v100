#include <immintrin.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

constexpr int kHidden = 2560;
constexpr int kIntermediate = 640;
constexpr int kGateUpRows = 1280;
constexpr int kDownRows = 2560;
constexpr int kExpertsDefault = 512;
constexpr std::uint64_t kUsefulFlopsPerPair = 9'830'400ULL;
constexpr double kDefaultTargetTokS = 16.49;
constexpr int kDefaultHostLayers = 36;
constexpr double kDefaultHitRate = 0.527;
constexpr double kDefaultHeadroom = 1.30;

[[nodiscard]] std::size_t align_up(std::size_t value, std::size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

[[noreturn]] void throw_errno(const std::string& what) {
    throw std::runtime_error(what + ": " + std::strerror(errno));
}

class MappedFile {
  public:
    MappedFile(const std::string& path, bool drop_file_cache) : path_(path) {
        fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd_ < 0) { throw_errno("open " + path); }
        struct stat st {};
        if (::fstat(fd_, &st) != 0) { throw_errno("fstat " + path); }
        if (st.st_size <= 0) { throw std::runtime_error(path + " is empty"); }
        size_ = static_cast<std::size_t>(st.st_size);
#ifdef POSIX_FADV_DONTNEED
        if (drop_file_cache) {
            const int rc = ::posix_fadvise(fd_, 0, 0, POSIX_FADV_DONTNEED);
            if (rc != 0) {
                std::cerr << "warning: posix_fadvise(DONTNEED) failed for " << path << ": "
                          << std::strerror(rc) << '\n';
            }
        }
#else
        (void)drop_file_cache;
#endif
        void* mapping = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (mapping == MAP_FAILED) { throw_errno("mmap " + path); }
        data_ = static_cast<const std::uint8_t*>(mapping);
#ifdef MADV_RANDOM
        (void)::madvise(const_cast<std::uint8_t*>(data_), size_, MADV_RANDOM);
#endif
    }

    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    MappedFile(MappedFile&& other) noexcept { *this = std::move(other); }
    MappedFile& operator=(MappedFile&& other) noexcept {
        if (this == &other) { return *this; }
        reset();
        fd_ = std::exchange(other.fd_, -1);
        size_ = std::exchange(other.size_, 0);
        data_ = std::exchange(other.data_, nullptr);
        path_ = std::move(other.path_);
        return *this;
    }

    ~MappedFile() { reset(); }

    [[nodiscard]] const std::uint8_t* data() const { return data_; }
    [[nodiscard]] std::size_t size() const { return size_; }

    std::uint64_t prefault() const {
        constexpr std::size_t kPage = 4096;
        std::uint64_t sum = 0;
        for (std::size_t i = 0; i < size_; i += kPage) { sum += data_[i]; }
        sum += data_[size_ - 1];
        return sum;
    }

  private:
    void reset() noexcept {
        if (data_ != nullptr) { ::munmap(const_cast<std::uint8_t*>(data_), size_); }
        if (fd_ >= 0) { ::close(fd_); }
        fd_ = -1;
        size_ = 0;
        data_ = nullptr;
    }

    int fd_ = -1;
    std::size_t size_ = 0;
    const std::uint8_t* data_ = nullptr;
    std::string path_;
};

struct Nvfp4BankView {
    const std::uint8_t* payload = nullptr;
    std::size_t payload_bytes = 0;
    int experts = 0;
    int rows = 0;
    int columns = 0;
    std::size_t code_plane_bytes = 0;
    std::size_t scale_plane_offset = 0;
    std::size_t scale_plane_bytes = 0;
    std::size_t divisor_offset = 0;
    std::size_t code_bytes_per_expert = 0;
    std::size_t scale_bytes_per_expert = 0;

    Nvfp4BankView() = default;

    Nvfp4BankView(const std::uint8_t* data, std::size_t bytes, int expert_count, int row_count,
                  int column_count)
        : payload(data), payload_bytes(bytes), experts(expert_count), rows(row_count),
          columns(column_count) {
        if (payload == nullptr || experts <= 0 || rows <= 0 || columns <= 0 || (rows % 128) != 0 ||
            (columns % 64) != 0) {
            throw std::invalid_argument("invalid NVFP4 bank geometry");
        }
        const std::size_t elements = static_cast<std::size_t>(experts) * rows * columns;
        code_plane_bytes = elements / 2;
        scale_plane_offset = align_up(code_plane_bytes, 256);
        scale_plane_bytes = elements / 16;
        divisor_offset = scale_plane_offset + scale_plane_bytes;
        const std::size_t expected = divisor_offset + static_cast<std::size_t>(experts) * sizeof(float);
        if (payload_bytes != expected) {
            throw std::invalid_argument("NVFP4 bank byte size does not match expert-blockscale-k16-m128x4-v1");
        }
        code_bytes_per_expert = static_cast<std::size_t>(rows) * columns / 2;
        scale_bytes_per_expert = static_cast<std::size_t>(rows) * columns / 16;
    }

    [[nodiscard]] const std::uint8_t* codes(int expert) const {
        return payload + static_cast<std::size_t>(expert) * code_bytes_per_expert;
    }

    [[nodiscard]] const std::uint8_t* scales(int expert) const {
        return payload + scale_plane_offset + static_cast<std::size_t>(expert) * scale_bytes_per_expert;
    }

    [[nodiscard]] float divisor(int expert) const {
        float value = 0.0F;
        std::memcpy(&value, payload + divisor_offset + static_cast<std::size_t>(expert) * sizeof(float),
                    sizeof(value));
        return value;
    }

    [[nodiscard]] std::size_t compact_bytes_per_expert() const {
        return code_bytes_per_expert + scale_bytes_per_expert + sizeof(float);
    }
};

[[nodiscard]] std::size_t scale_row_base(int row, int columns) {
    const int k_tiles = columns / 64;
    const int row_tile = row / 128;
    const int row_inner = row % 128;
    return static_cast<std::size_t>(row_tile * k_tiles) * 512 +
           static_cast<std::size_t>(row_inner & 31) * 16 +
           static_cast<std::size_t>(row_inner >> 5) * 4;
}

[[nodiscard]] std::size_t scale_byte_offset_from_base(std::size_t row_base, int group) {
    return row_base + static_cast<std::size_t>(group >> 2) * 512 +
           static_cast<std::size_t>(group & 3);
}

[[nodiscard]] float decode_e2m1(std::uint8_t code) {
    static constexpr float kPositive[8] = {0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F};
    const float magnitude = kPositive[code & 7U];
    return (code & 8U) != 0 ? -magnitude : magnitude;
}

[[nodiscard]] float decode_e4m3fn(std::uint8_t code) {
    const bool negative = (code & 0x80U) != 0;
    const int exponent = (code >> 3) & 0x0F;
    const int mantissa = code & 0x07;
    float value = 0.0F;
    if (exponent == 0) {
        value = mantissa == 0 ? 0.0F : std::ldexp(static_cast<float>(mantissa), -9);
    } else if (exponent < 15) {
        value = std::ldexp(1.0F + static_cast<float>(mantissa) / 8.0F, exponent - 7);
    } else if (mantissa < 7) {
        value = std::ldexp(1.0F + static_cast<float>(mantissa) / 8.0F, 8);
    } else {
        return std::numeric_limits<float>::quiet_NaN();
    }
    return negative ? -value : value;
}

struct ScaleTable {
    std::array<float, 256> value{};
    ScaleTable() {
        for (int i = 0; i < 256; ++i) { value[static_cast<std::size_t>(i)] = decode_e4m3fn(static_cast<std::uint8_t>(i)); }
    }
};

const ScaleTable kScaleTable;

struct DecodedWeights16 {
    __m256 lo;
    __m256 hi;
};

[[nodiscard]] DecodedWeights16 decode_weights16(const std::uint8_t* packed, float coefficient) {
    static const __m128i kNibbleMask = _mm_set1_epi8(0x0F);
    static const __m128i kQ2Lut =
        _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
    const __m128i bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(packed));
    const __m128i lo_nibbles = _mm_and_si128(bytes, kNibbleMask);
    const __m128i hi_nibbles = _mm_and_si128(_mm_srli_epi16(bytes, 4), kNibbleMask);
    const __m128i interleaved = _mm_unpacklo_epi8(lo_nibbles, hi_nibbles);
    const __m128i q2 = _mm_shuffle_epi8(kQ2Lut, interleaved);
    const __m256 scale = _mm256_set1_ps(coefficient * 0.5F);
    const __m256i ints_lo = _mm256_cvtepi8_epi32(q2);
    const __m256i ints_hi = _mm256_cvtepi8_epi32(_mm_srli_si128(q2, 8));
    return {
        .lo = _mm256_mul_ps(_mm256_cvtepi32_ps(ints_lo), scale),
        .hi = _mm256_mul_ps(_mm256_cvtepi32_ps(ints_hi), scale),
    };
}

[[nodiscard]] float hsum256(__m256 value) {
    const __m128 lo = _mm256_castps256_ps128(value);
    const __m128 hi = _mm256_extractf128_ps(value, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

[[nodiscard]] std::pair<float, float> gate_up_rows_avx2(
    const Nvfp4BankView& gate_up, int expert, int row, const float* input) {
    const auto* codes = gate_up.codes(expert);
    const auto* scales = gate_up.scales(expert);
    const float divisor = gate_up.divisor(expert);
    if (!(divisor > 0.0F) || !std::isfinite(divisor)) {
        throw std::runtime_error("invalid NVFP4 gate/up divisor");
    }
    const float inverse = 1.0F / divisor;
    const int up_row = row + kIntermediate;
    const std::size_t gate_scale_base = scale_row_base(row, gate_up.columns);
    const std::size_t up_scale_base = scale_row_base(up_row, gate_up.columns);
    __m256 gate0 = _mm256_setzero_ps();
    __m256 gate1 = _mm256_setzero_ps();
    __m256 up0 = _mm256_setzero_ps();
    __m256 up1 = _mm256_setzero_ps();
    const int groups = gate_up.columns / 16;
    for (int group = 0; group < groups; ++group) {
        const __m256 x0 = _mm256_loadu_ps(input + group * 16);
        const __m256 x1 = _mm256_loadu_ps(input + group * 16 + 8);

        const std::size_t gate_scale_off = scale_byte_offset_from_base(gate_scale_base, group);
        const float gate_coeff = kScaleTable.value[scales[gate_scale_off]] * inverse;
        const auto* gate_codes =
            codes + static_cast<std::size_t>(row) * (gate_up.columns / 2) + group * 8;
        const DecodedWeights16 gate_w = decode_weights16(gate_codes, gate_coeff);
        gate0 = _mm256_fmadd_ps(gate_w.lo, x0, gate0);
        gate1 = _mm256_fmadd_ps(gate_w.hi, x1, gate1);

        const std::size_t up_scale_off = scale_byte_offset_from_base(up_scale_base, group);
        const float up_coeff = kScaleTable.value[scales[up_scale_off]] * inverse;
        const auto* up_codes =
            codes + static_cast<std::size_t>(up_row) * (gate_up.columns / 2) + group * 8;
        const DecodedWeights16 up_w = decode_weights16(up_codes, up_coeff);
        up0 = _mm256_fmadd_ps(up_w.lo, x0, up0);
        up1 = _mm256_fmadd_ps(up_w.hi, x1, up1);
    }
    return {hsum256(_mm256_add_ps(gate0, gate1)), hsum256(_mm256_add_ps(up0, up1))};
}

[[nodiscard]] float down_row_avx2(const Nvfp4BankView& down, int expert, int row,
                                  const float* activation) {
    const auto* codes = down.codes(expert);
    const auto* scales = down.scales(expert);
    const float divisor = down.divisor(expert);
    if (!(divisor > 0.0F) || !std::isfinite(divisor)) {
        throw std::runtime_error("invalid NVFP4 down divisor");
    }
    const float inverse = 1.0F / divisor;
    const std::size_t scale_base = scale_row_base(row, down.columns);
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    const int groups = down.columns / 16;
    for (int group = 0; group < groups; ++group) {
        const float coeff =
            kScaleTable.value[scales[scale_byte_offset_from_base(scale_base, group)]] * inverse;
        const auto* packed = codes + static_cast<std::size_t>(row) * (down.columns / 2) + group * 8;
        const DecodedWeights16 weights = decode_weights16(packed, coeff);
        acc0 = _mm256_fmadd_ps(weights.lo, _mm256_loadu_ps(activation + group * 16), acc0);
        acc1 = _mm256_fmadd_ps(weights.hi, _mm256_loadu_ps(activation + group * 16 + 8), acc1);
    }
    return hsum256(_mm256_add_ps(acc0, acc1));
}

struct Scratch {
    alignas(64) std::array<float, kIntermediate> activation{};
    alignas(64) std::array<float, kHidden> output{};
};

[[nodiscard]] double compute_pair_avx2(const Nvfp4BankView& gate_up, const Nvfp4BankView& down,
                                       int expert, const float* input, Scratch& scratch) {
    for (int row = 0; row < kIntermediate; ++row) {
        const auto [gate, up] = gate_up_rows_avx2(gate_up, expert, row, input);
        scratch.activation[static_cast<std::size_t>(row)] = gate / (1.0F + std::exp(-gate)) * up;
    }
    double checksum = 0.0;
    for (int row = 0; row < kHidden; ++row) {
        const float value = down_row_avx2(down, expert, row, scratch.activation.data());
        scratch.output[static_cast<std::size_t>(row)] = value;
        checksum += static_cast<double>(value);
    }
    return checksum;
}

[[nodiscard]] float scalar_row(const Nvfp4BankView& bank, int expert, int row, const float* input) {
    const auto* codes = bank.codes(expert);
    const auto* scales = bank.scales(expert);
    const float divisor = bank.divisor(expert);
    if (!(divisor > 0.0F) || !std::isfinite(divisor)) {
        throw std::runtime_error("invalid NVFP4 divisor in scalar reference");
    }
    const float inverse = 1.0F / divisor;
    const std::size_t scale_base = scale_row_base(row, bank.columns);
    float sum = 0.0F;
    for (int group = 0; group < bank.columns / 16; ++group) {
        const float coeff =
            kScaleTable.value[scales[scale_byte_offset_from_base(scale_base, group)]] * inverse;
        const auto* packed = codes + static_cast<std::size_t>(row) * (bank.columns / 2) + group * 8;
        for (int lane = 0; lane < 16; ++lane) {
            const std::uint8_t byte = packed[lane >> 1];
            const std::uint8_t code = (lane & 1) == 0 ? (byte & 0x0FU) : (byte >> 4);
            sum = std::fma(decode_e2m1(code) * coeff, input[group * 16 + lane], sum);
        }
    }
    return sum;
}

void compute_pair_scalar(const Nvfp4BankView& gate_up, const Nvfp4BankView& down, int expert,
                         const float* input, Scratch& scratch) {
    for (int row = 0; row < kIntermediate; ++row) {
        const float gate = scalar_row(gate_up, expert, row, input);
        const float up = scalar_row(gate_up, expert, row + kIntermediate, input);
        scratch.activation[static_cast<std::size_t>(row)] = gate / (1.0F + std::exp(-gate)) * up;
    }
    for (int row = 0; row < kHidden; ++row) {
        scratch.output[static_cast<std::size_t>(row)] =
            scalar_row(down, expert, row, scratch.activation.data());
    }
}

struct Comparison {
    double cosine = 0.0;
    double nrmse = 0.0;
    double max_abs = 0.0;
};

[[nodiscard]] Comparison compare_vectors(std::span<const float> reference,
                                         std::span<const float> candidate) {
    if (reference.size() != candidate.size() || reference.empty()) {
        throw std::invalid_argument("invalid vector comparison");
    }
    long double dot = 0.0L;
    long double ref_sq = 0.0L;
    long double cand_sq = 0.0L;
    long double err_sq = 0.0L;
    double max_abs = 0.0;
    for (std::size_t i = 0; i < reference.size(); ++i) {
        const long double r = reference[i];
        const long double c = candidate[i];
        const long double e = c - r;
        dot += r * c;
        ref_sq += r * r;
        cand_sq += c * c;
        err_sq += e * e;
        max_abs = std::max(max_abs, std::abs(static_cast<double>(e)));
    }
    const long double denom = std::sqrt(ref_sq * cand_sq);
    const double cosine = denom == 0.0L ? 1.0 : static_cast<double>(dot / denom);
    const long double ref_rms = std::sqrt(ref_sq / reference.size());
    const long double err_rms = std::sqrt(err_sq / reference.size());
    const double nrmse = ref_rms == 0.0L ? static_cast<double>(err_rms)
                                          : static_cast<double>(err_rms / ref_rms);
    return {.cosine = cosine, .nrmse = nrmse, .max_abs = max_abs};
}

void verify_avx2(const Nvfp4BankView& gate_up, const Nvfp4BankView& down, int expert,
                 const std::array<float, kHidden>& input) {
    Scratch scalar{};
    Scratch vectorized{};
    compute_pair_scalar(gate_up, down, expert, input.data(), scalar);
    (void)compute_pair_avx2(gate_up, down, expert, input.data(), vectorized);
    const Comparison result = compare_vectors(scalar.output, vectorized.output);
    std::cout << std::setprecision(9) << "verify.cosine=" << result.cosine
              << " verify.nrmse=" << result.nrmse << " verify.max_abs=" << result.max_abs << '\n';
    if (!(result.cosine >= 0.99999) || !(result.nrmse <= 2.0e-3)) {
        throw std::runtime_error("AVX2 expert-pair result failed the Phase-0 numerical gate");
    }
}

struct Options {
    std::string gate_up_path;
    std::string down_path;
    std::string ids_path;
    int experts = kExpertsDefault;
    unsigned threads = std::max(1U, std::thread::hardware_concurrency());
    std::size_t pairs_per_batch = 256;
    int warmup_batches = 2;
    int batches = 20;
    std::uint64_t seed = 0x5A17'70ULL;
    bool prefault = true;
    bool drop_file_cache = false;
    bool verify = true;
    bool self_test = false;
    double target_tok_s = kDefaultTargetTokS;
    int host_layers = kDefaultHostLayers;
    double hit_rate = kDefaultHitRate;
    double headroom = kDefaultHeadroom;
};

[[nodiscard]] std::string require_value(int& i, int argc, char** argv, std::string_view flag) {
    if (++i >= argc) { throw std::invalid_argument(std::string(flag) + " requires a value"); }
    return argv[i];
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (arg == "--gate-up") {
            options.gate_up_path = require_value(i, argc, argv, arg);
        } else if (arg == "--down") {
            options.down_path = require_value(i, argc, argv, arg);
        } else if (arg == "--ids") {
            options.ids_path = require_value(i, argc, argv, arg);
        } else if (arg == "--experts") {
            options.experts = std::stoi(require_value(i, argc, argv, arg));
        } else if (arg == "--threads") {
            options.threads = static_cast<unsigned>(std::stoul(require_value(i, argc, argv, arg)));
        } else if (arg == "--pairs-per-batch") {
            options.pairs_per_batch = std::stoull(require_value(i, argc, argv, arg));
        } else if (arg == "--warmup-batches") {
            options.warmup_batches = std::stoi(require_value(i, argc, argv, arg));
        } else if (arg == "--batches") {
            options.batches = std::stoi(require_value(i, argc, argv, arg));
        } else if (arg == "--seed") {
            options.seed = std::stoull(require_value(i, argc, argv, arg), nullptr, 0);
        } else if (arg == "--target-tok-s") {
            options.target_tok_s = std::stod(require_value(i, argc, argv, arg));
        } else if (arg == "--host-layers") {
            options.host_layers = std::stoi(require_value(i, argc, argv, arg));
        } else if (arg == "--hit-rate") {
            options.hit_rate = std::stod(require_value(i, argc, argv, arg));
        } else if (arg == "--headroom") {
            options.headroom = std::stod(require_value(i, argc, argv, arg));
        } else if (arg == "--no-prefault") {
            options.prefault = false;
        } else if (arg == "--drop-file-cache") {
            options.drop_file_cache = true;
        } else if (arg == "--no-verify") {
            options.verify = false;
        } else if (arg == "--self-test") {
            options.self_test = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage:\n"
                << "  ninfer_cpu_nvfp4_expert_pair --gate-up FILE --down FILE [options]\n"
                << "  ninfer_cpu_nvfp4_expert_pair --self-test\n\n"
                << "Options:\n"
                << "  --threads N              worker threads\n"
                << "  --pairs-per-batch N      expert pairs per timed batch (default 256)\n"
                << "  --warmup-batches N       untimed batches (default 2)\n"
                << "  --batches N              timed batches (default 20)\n"
                << "  --ids FILE               recorded expert IDs, one integer per line\n"
                << "  --seed N                 random sequence seed\n"
                << "  --no-prefault            do not touch every mapped page before timing\n"
                << "  --drop-file-cache        request POSIX_FADV_DONTNEED before mapping\n"
                << "  --no-verify              skip scalar-vs-AVX2 one-pair numerical check\n"
                << "  --target-tok-s R         architecture-gate decode rate (default 16.49)\n"
                << "  --host-layers N          architecture-gate L_host (default 36)\n"
                << "  --hit-rate H             architecture-gate cache hit rate (default 0.527)\n"
                << "  --headroom X             preferred throughput multiplier (default 1.3)\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown option: " + std::string(arg));
        }
    }
    if (options.threads == 0 || options.pairs_per_batch == 0 || options.batches <= 0 ||
        options.warmup_batches < 0 || options.experts <= 0 || options.target_tok_s <= 0.0 ||
        options.host_layers <= 0 || options.hit_rate < 0.0 || options.hit_rate > 1.0 ||
        options.headroom < 1.0) {
        throw std::invalid_argument("invalid non-positive or out-of-range option");
    }
    if (!options.self_test && (options.gate_up_path.empty() || options.down_path.empty())) {
        throw std::invalid_argument("--gate-up and --down are required unless --self-test is used");
    }
    return options;
}

std::vector<int> load_ids(const std::string& path, int experts) {
    std::ifstream input(path);
    if (!input) { throw std::runtime_error("cannot open ID sequence: " + path); }
    std::vector<int> ids;
    int id = 0;
    while (input >> id) {
        if (id < 0 || id >= experts) { throw std::runtime_error("expert ID is outside bank range"); }
        ids.push_back(id);
    }
    if (ids.empty()) { throw std::runtime_error("ID sequence is empty"); }
    return ids;
}

std::vector<int> make_sequence(const Options& options, std::size_t count) {
    std::vector<int> sequence;
    sequence.reserve(count);
    if (!options.ids_path.empty()) {
        const std::vector<int> recorded = load_ids(options.ids_path, options.experts);
        for (std::size_t i = 0; i < count; ++i) { sequence.push_back(recorded[i % recorded.size()]); }
        return sequence;
    }
    std::mt19937_64 rng(options.seed);
    std::uniform_int_distribution<int> distribution(0, options.experts - 1);
    for (std::size_t i = 0; i < count; ++i) { sequence.push_back(distribution(rng)); }
    return sequence;
}

std::array<float, kHidden> make_input(std::uint64_t seed) {
    std::array<float, kHidden> input{};
    std::mt19937_64 rng(seed ^ 0xA511'CE55ULL);
    std::uniform_real_distribution<float> distribution(-0.25F, 0.25F);
    for (float& value : input) { value = distribution(rng); }
    return input;
}

[[nodiscard]] double percentile(std::vector<double> values, double q) {
    if (values.empty()) { return 0.0; }
    std::sort(values.begin(), values.end());
    const double position = q * static_cast<double>(values.size() - 1);
    const std::size_t lo = static_cast<std::size_t>(std::floor(position));
    const std::size_t hi = static_cast<std::size_t>(std::ceil(position));
    const double fraction = position - static_cast<double>(lo);
    return values[lo] * (1.0 - fraction) + values[hi] * fraction;
}

struct WorkerSlot {
    Scratch scratch{};
    double checksum = 0.0;
};

struct BenchResult {
    double elapsed_s = 0.0;
    double pairs_per_s = 0.0;
    double compact_gib_s = 0.0;
    double useful_gflops = 0.0;
    double us_per_pair = 0.0;
    double p50_batch_us = 0.0;
    double p95_batch_us = 0.0;
    double p99_batch_us = 0.0;
    double checksum = 0.0;
};

BenchResult run_benchmark(const Nvfp4BankView& gate_up, const Nvfp4BankView& down,
                          const Options& options, const std::array<float, kHidden>& input,
                          const std::vector<int>& sequence) {
    const int total_batches = options.warmup_batches + options.batches;
    const std::size_t expected_sequence = static_cast<std::size_t>(total_batches) * options.pairs_per_batch;
    if (sequence.size() < expected_sequence) { throw std::logic_error("expert sequence too short"); }

    std::atomic<std::size_t> next{0};
    std::atomic<std::size_t> batch_base{0};
    std::atomic<bool> stop{false};
    std::barrier start_barrier(static_cast<std::ptrdiff_t>(options.threads + 1));
    std::barrier done_barrier(static_cast<std::ptrdiff_t>(options.threads + 1));
    std::vector<WorkerSlot> slots(options.threads);
    std::vector<std::thread> workers;
    workers.reserve(options.threads);

    for (unsigned worker = 0; worker < options.threads; ++worker) {
        workers.emplace_back([&, worker] {
            for (;;) {
                start_barrier.arrive_and_wait();
                if (stop.load(std::memory_order_acquire)) { break; }
                const std::size_t base = batch_base.load(std::memory_order_relaxed);
                double checksum = 0.0;
                for (;;) {
                    const std::size_t local = next.fetch_add(1, std::memory_order_relaxed);
                    if (local >= options.pairs_per_batch) { break; }
                    const int expert = sequence[base + local];
                    checksum += compute_pair_avx2(gate_up, down, expert, input.data(), slots[worker].scratch);
                }
                slots[worker].checksum = checksum;
                done_barrier.arrive_and_wait();
            }
        });
    }

    std::vector<double> batch_us;
    batch_us.reserve(static_cast<std::size_t>(options.batches));
    double timed_checksum = 0.0;
    double timed_seconds = 0.0;

    for (int batch = 0; batch < total_batches; ++batch) {
        const std::size_t base = static_cast<std::size_t>(batch) * options.pairs_per_batch;
        batch_base.store(base, std::memory_order_relaxed);
        next.store(0, std::memory_order_relaxed);
        const auto t0 = std::chrono::steady_clock::now();
        start_barrier.arrive_and_wait();
        done_barrier.arrive_and_wait();
        const auto t1 = std::chrono::steady_clock::now();
        double checksum = 0.0;
        for (const WorkerSlot& slot : slots) { checksum += slot.checksum; }
        if (batch >= options.warmup_batches) {
            const double seconds = std::chrono::duration<double>(t1 - t0).count();
            timed_seconds += seconds;
            batch_us.push_back(seconds * 1.0e6);
            timed_checksum += checksum;
        }
    }

    stop.store(true, std::memory_order_release);
    start_barrier.arrive_and_wait();
    for (std::thread& worker : workers) { worker.join(); }

    const double pairs = static_cast<double>(options.batches) * options.pairs_per_batch;
    const double pairs_per_s = pairs / timed_seconds;
    const std::size_t compact_bytes = gate_up.compact_bytes_per_expert() + down.compact_bytes_per_expert();
    return {
        .elapsed_s = timed_seconds,
        .pairs_per_s = pairs_per_s,
        .compact_gib_s = pairs_per_s * static_cast<double>(compact_bytes) / static_cast<double>(1ULL << 30),
        .useful_gflops = pairs_per_s * static_cast<double>(kUsefulFlopsPerPair) / 1.0e9,
        .us_per_pair = 1.0e6 / pairs_per_s,
        .p50_batch_us = percentile(batch_us, 0.50),
        .p95_batch_us = percentile(batch_us, 0.95),
        .p99_batch_us = percentile(batch_us, 0.99),
        .checksum = timed_checksum,
    };
}

void print_result(const BenchResult& result, const Options& options, std::size_t pair_bytes) {
    const double required_pairs_s =
        options.target_tok_s * 10.0 * static_cast<double>(options.host_layers) * (1.0 - options.hit_rate);
    const double preferred_pairs_s = options.headroom * required_pairs_s;
    std::cout << std::fixed << std::setprecision(3)
              << "workers=" << options.threads << '\n'
              << "pair_bytes=" << pair_bytes << '\n'
              << "timed_seconds=" << result.elapsed_s << '\n'
              << "pairs_per_s=" << result.pairs_per_s << '\n'
              << "us_per_pair=" << result.us_per_pair << '\n'
              << "compact_GiB_per_s=" << result.compact_gib_s << '\n'
              << "useful_GFLOP_per_s=" << result.useful_gflops << '\n'
              << "batch_p50_us=" << result.p50_batch_us << '\n'
              << "batch_p95_us=" << result.p95_batch_us << '\n'
              << "batch_p99_us=" << result.p99_batch_us << '\n'
              << "checksum=" << result.checksum << '\n'
              << "gate_target_tok_s=" << options.target_tok_s << '\n'
              << "gate_host_layers=" << options.host_layers << '\n'
              << "gate_hit_rate=" << options.hit_rate << '\n'
              << "gate_required_pairs_per_s=" << required_pairs_s << '\n'
              << "gate_preferred_pairs_per_s=" << preferred_pairs_s << '\n'
              << "gate_minimum=" << (result.pairs_per_s >= required_pairs_s ? "PASS" : "FAIL") << '\n'
              << "gate_preferred=" << (result.pairs_per_s >= preferred_pairs_s ? "PASS" : "FAIL") << '\n';
}

struct OwnedBank {
    std::vector<std::uint8_t> bytes;
    Nvfp4BankView view;
};

OwnedBank make_synthetic_bank(int rows, int columns, std::uint64_t seed) {
    const int experts = 1;
    const std::size_t elements = static_cast<std::size_t>(experts) * rows * columns;
    const std::size_t code_bytes = elements / 2;
    const std::size_t scale_offset = align_up(code_bytes, 256);
    const std::size_t scale_bytes = elements / 16;
    const std::size_t divisor_offset = scale_offset + scale_bytes;
    OwnedBank out;
    out.bytes.resize(divisor_offset + sizeof(float), 0);
    std::mt19937_64 rng(seed);
    for (std::size_t i = 0; i < code_bytes; ++i) { out.bytes[i] = static_cast<std::uint8_t>(rng()); }
    for (std::size_t i = 0; i < scale_bytes; ++i) {
        out.bytes[scale_offset + i] = static_cast<std::uint8_t>(rng() % 0x7FU);
    }
    const float divisor = 7.25F;
    std::memcpy(out.bytes.data() + divisor_offset, &divisor, sizeof(divisor));
    out.view = Nvfp4BankView(out.bytes.data(), out.bytes.size(), experts, rows, columns);
    return out;
}

int run_self_test() {
    OwnedBank gate = make_synthetic_bank(kGateUpRows, kHidden, 0x1234ULL);
    OwnedBank down = make_synthetic_bank(kDownRows, kIntermediate, 0x5678ULL);
    const auto input = make_input(0x9ABCULL);
    verify_avx2(gate.view, down.view, 0, input);
    const std::size_t pair_bytes = gate.view.compact_bytes_per_expert() + down.view.compact_bytes_per_expert();
    if (pair_bytes != 2'764'808ULL) {
        throw std::runtime_error("self-test expert-pair byte count does not match Flash-Next contract");
    }
    std::cout << "self_test=PASS pair_bytes=" << pair_bytes << '\n';
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        if (options.self_test) { return run_self_test(); }

        MappedFile gate_file(options.gate_up_path, options.drop_file_cache);
        MappedFile down_file(options.down_path, options.drop_file_cache);
        const Nvfp4BankView gate_up(gate_file.data(), gate_file.size(), options.experts, kGateUpRows,
                                    kHidden);
        const Nvfp4BankView down(down_file.data(), down_file.size(), options.experts, kDownRows,
                                 kIntermediate);
        const std::size_t pair_bytes = gate_up.compact_bytes_per_expert() + down.compact_bytes_per_expert();
        if (pair_bytes != 2'764'808ULL) {
            throw std::runtime_error("expert-pair byte count does not match the Flash-Next contract");
        }

        if (options.prefault) {
            const auto t0 = std::chrono::steady_clock::now();
            const std::uint64_t prefault_checksum = gate_file.prefault() + down_file.prefault();
            const auto t1 = std::chrono::steady_clock::now();
            std::cout << "prefault_seconds=" << std::chrono::duration<double>(t1 - t0).count()
                      << " prefault_checksum=" << prefault_checksum << '\n';
        }

        const auto input = make_input(options.seed);
        const std::size_t sequence_count =
            static_cast<std::size_t>(options.warmup_batches + options.batches) * options.pairs_per_batch;
        const std::vector<int> sequence = make_sequence(options, sequence_count);

        if (options.verify) { verify_avx2(gate_up, down, sequence.front(), input); }

        const BenchResult result = run_benchmark(gate_up, down, options, input, sequence);
        print_result(result, options, pair_bytes);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
