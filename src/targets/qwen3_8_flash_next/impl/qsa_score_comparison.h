#pragma once

// Host-only, explicitly requested attribution. All temporary allocations and
// synchronizations are confined to this diagnostic; no production policy changes.
#include "core/device.h"
#include "targets/qwen3_8_flash_next/impl/qsa_attention_kernels.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <stdexcept>
#include <vector>

namespace ninfer::targets::qwen3_8_flash_next::detail {

inline float qsa_reference_bf16(std::uint16_t bits) {
    return std::bit_cast<float>(std::uint32_t(bits) << 16);
}
// Independent mathematical E4M3FN decoding, rather than the CUDA kernel codec.
inline double qsa_reference_fp8(std::uint8_t bits) {
    const int exponent = (bits >> 3) & 15, mantissa = bits & 7;
    if (exponent == 15 && mantissa == 7)
        throw std::runtime_error("nonfinite real QSA reference input");
    const double magnitude = exponent == 0 ? std::ldexp(double(mantissa), -9) :
        std::ldexp(1.0 + double(mantissa) / 8.0, exponent - 7);
    return bits & 128 ? -magnitude : magnitude;
}
inline double qsa_reference_rounded(double value) {
    const auto f = std::bit_cast<std::uint32_t>(float(value));
    return qsa_reference_bf16(std::uint16_t((f + 0x7fffU + ((f >> 16) & 1U)) >> 16));
}

inline void compare_real_qsa_scores(const Tensor& query, const Tensor& indices,
    int table_row, const Tensor& selected, const Tensor& counts,
    QsaAttentionCacheView cache, const Tensor& actual, cudaStream_t stream,
    bool actual_mma) {
    constexpr int heads = 24, dim = 256, kv_heads = 2, page_tokens = 64;
    const int tokens = indices.ne[0];
    if (cache.key_pages.dtype != DType::FP8_E4M3FN || tokens < 512)
        throw std::invalid_argument("real QSA comparison requires eligible large FP8 inputs");
    const std::size_t elements = std::size_t(tokens) * heads * dim;
    DeviceBuffer alternate(elements * 2);
    Tensor other(alternate.p, DType::BF16, {dim, heads, tokens});
    flash_next_qsa_volta_attend_launch(query, indices, table_row, selected, counts,
                                      cache, other, stream, !actual_mma);
    std::vector<std::uint16_t> q(elements), a(elements), b(elements);
    std::vector<int> positions(tokens), selected_counts(tokens), blocks(tokens * 512);
    std::vector<int> table(cache.block_tables.ne[0]);
    struct DrainCopies {
        cudaStream_t stream;
        ~DrainCopies() { (void)cudaStreamSynchronize(stream); }
    } initial_drain{stream};
    const auto copy = [&](void* to, const void* from, std::size_t bytes) {
        CUDA_CHECK(cudaMemcpyAsync(to, from, bytes, cudaMemcpyDeviceToHost, stream));
    };
    copy(q.data(), query.data, elements * 2);
    copy(a.data(), actual.data, elements * 2); copy(b.data(), other.data, elements * 2);
    copy(positions.data(), indices.data, tokens * 4);
    copy(selected_counts.data(), counts.data, tokens * 4);
    copy(blocks.data(), selected.data, blocks.size() * 4);
    copy(table.data(), static_cast<const int*>(cache.block_tables.data) +
         table_row * table.size(), table.size() * 4);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto& simt = actual_mma ? b : a;
    const auto& mma = actual_mma ? a : b;
    double error = 0, norm = 0, max_abs = 0;
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < elements; ++i) {
        const double x = qsa_reference_bf16(simt[i]), y = qsa_reference_bf16(mma[i]);
        if (!std::isfinite(x) || !std::isfinite(y))
            throw std::runtime_error("nonfinite real QSA comparison output");
        mismatches += simt[i] != mma[i];
        error += (x-y)*(x-y); norm += x*x; max_abs = std::max(max_abs, std::abs(x-y));
    }
    // Three real query positions and ALL heads. Read only pages those queries
    // actually select; never copy unused or uninitialized physical KV pages.
    const std::vector<int> sample_tokens{0, tokens / 2, tokens - 1};
    std::vector<std::vector<int>> candidates;
    std::map<int, std::pair<std::vector<std::uint8_t>, std::vector<std::uint8_t>>> pages;
    DrainCopies page_drain{stream};
    constexpr std::size_t page_bytes = kv_heads * page_tokens * dim;
    for (int t : sample_tokens) {
        std::vector<int> current;
        const int complete = (positions[t] + 1) / 4;
        for (int block = 0; block < selected_counts[t]; ++block)
            for (int j = 0; j < 4; ++j) current.push_back(blocks[t*512+block]*4+j);
        for (int j = 0; j < ((positions[t]+1)&3); ++j) current.push_back(complete*4+j);
        for (int position : current) {
            if (position < 0 || position / page_tokens >= int(table.size()))
                throw std::runtime_error("real QSA sample selected invalid logical page");
            const int physical = table[position/page_tokens];
            if (physical < 0 || physical >= cache.key_pages.ne[3])
                throw std::runtime_error("real QSA sample selected invalid physical page");
            if (pages.contains(physical)) continue;
            auto& page = pages[physical]; page.first.resize(page_bytes); page.second.resize(page_bytes);
            copy(page.first.data(), static_cast<const std::uint8_t*>(cache.key_pages.data)+physical*page_bytes, page_bytes);
            copy(page.second.data(), static_cast<const std::uint8_t*>(cache.value_pages.data)+physical*page_bytes, page_bytes);
        }
        candidates.push_back(std::move(current));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<double> reference, simt_sample, mma_sample;
    for (std::size_t sample = 0; sample < sample_tokens.size(); ++sample) {
        const int t = sample_tokens[sample]; const auto& current = candidates[sample];
        for (int h = 0; h < heads; ++h) {
            std::vector<double> scores(current.size()); double maximum = -INFINITY;
            for (std::size_t c = 0; c < current.size(); ++c) {
                const int position = current[c]; const auto& key = pages.at(table[position/page_tokens]).first;
                const auto offset = ((h/12)*page_tokens+position%page_tokens)*dim;
                double dot = 0;
                for (int d = 0; d < dim; ++d)
                    dot += double(qsa_reference_bf16(q[(std::size_t(t)*heads+h)*dim+d])) *
                           qsa_reference_fp8(key[offset+d]);
                scores[c] = dot / 16.; maximum = std::max(maximum, scores[c]);
            }
            double sum = 0; for (double& score : scores) { score = std::exp(score-maximum); sum += score; }
            double result[dim]{};
            for (std::size_t c = 0; c < current.size(); ++c) {
                const int position = current[c]; const auto& value = pages.at(table[position/page_tokens]).second;
                const auto offset = ((h/12)*page_tokens+position%page_tokens)*dim;
                for (int d = 0; d < dim; ++d) result[d] += scores[c]/sum*qsa_reference_fp8(value[offset+d]);
            }
            for (int d = 0; d < dim; ++d) {
                reference.push_back(qsa_reference_rounded(result[d]));
                simt_sample.push_back(qsa_reference_bf16(simt[(std::size_t(t)*heads+h)*dim+d]));
                mma_sample.push_back(qsa_reference_bf16(mma[(std::size_t(t)*heads+h)*dim+d]));
            }
        }
    }
    double reference_max = 0, reference_norm = 0;
    for (double x : reference) { reference_max = std::max(reference_max, std::abs(x)); reference_norm += x*x; }
    double squared[2]{}, pointwise[2]{};
    for (int path = 0; path < 2; ++path) {
        const auto& values = path ? mma_sample : simt_sample;
        for (std::size_t i = 0; i < reference.size(); ++i) {
            const double diff = std::abs(values[i]-reference[i]); squared[path] += diff*diff;
            const double allowance = 1e-3*reference_max+1e-2*std::abs(reference[i]);
            pointwise[path] = std::max(pointwise[path], allowance > 0 ? diff/allowance : (diff == 0 ? 0 : INFINITY));
        }
    }
    const double relative[2]{std::sqrt(squared[0]/std::max(reference_norm, 1e-300)),
                            std::sqrt(squared[1]/std::max(reference_norm, 1e-300))};
    const bool passes = relative[0] <= 1e-3 && relative[1] <= 1e-3 &&
                        pointwise[0] <= 1 && pointwise[1] <= 1;
    std::fprintf(stderr,
        "{\"kind\":\"qsa_real_comparison\",\"tokens\":%d,\"sample_queries\":3,\"sample_heads\":24,"
        "\"elements\":%zu,\"bit_mismatches\":%zu,\"max_absolute_difference\":%.9g,"
        "\"relative_l2_difference\":%.9g,\"simt_fp64_relative_l2\":%.9g,\"mma_fp64_relative_l2\":%.9g,"
        "\"simt_pointwise_ratio\":%.9g,\"mma_pointwise_ratio\":%.9g,\"sample_oracle_pass\":%s}\n",
        tokens, elements, mismatches, max_abs, std::sqrt(error/std::max(norm, 1e-300)),
        relative[0], relative[1], pointwise[0], pointwise[1], passes ? "true" : "false");
    if (!passes) throw std::runtime_error("real QSA sampled independent FP64 threshold failed");
}

} // namespace ninfer::targets::qwen3_8_flash_next::detail
