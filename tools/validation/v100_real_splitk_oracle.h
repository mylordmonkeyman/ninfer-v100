#pragma once

// Diagnostic-only instrumentation of real model contractions. The workflow inserts this
// capture around actual Q4/Q5 launches; it is never linked into the delivered Engine.
#include "core/device.h"
#include "core/tensor.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <iostream>
#include <set>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace ninfer::validation {
inline double bf16_value(std::uint16_t bits) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16);
}
inline double f16_value(std::uint16_t bits) {
    const int exponent = (bits >> 10) & 31;
    const int fraction = bits & 1023;
    const double magnitude = exponent == 0 ? std::ldexp(static_cast<double>(fraction), -24)
        : exponent == 31 ? (fraction ? NAN : INFINITY)
        : std::ldexp(1.0 + fraction / 1024.0, exponent - 15);
    return bits & 0x8000 ? -magnitude : magnitude;
}

class RealSplitKOracle {
    Tensor output_;
    cudaStream_t stream_;
    bool active_ = false, q5_ = false, add_ = false;
    int n_ = 0, t_ = 0, k_ = 0;
    std::vector<int> rows_;
    std::vector<double> reference_;
    std::vector<std::uint16_t> result_;
public:
    RealSplitKOracle(const Tensor& x, const Weight& w, const Tensor& out, bool q5,
                     bool add_residual, int weight_row_offset, cudaStream_t stream)
        : output_(out), stream_(stream), q5_(q5), add_(add_residual), n_(out.ne[0]), t_(x.ne[1]), k_(x.ne[0]) {
        if (t_ != 31) return;
        static std::set<std::tuple<bool, int, int, bool>> checked;
        if (!checked.emplace(q5, n_, k_, add_residual).second) return;
        active_ = true;
        const int k = x.ne[0], groups = w.padded_shape[1] / 64;
        std::vector<std::uint16_t> activation(static_cast<std::size_t>(k) * t_);
        result_.resize(static_cast<std::size_t>(n_) * t_);
        CUDA_CHECK(cudaMemcpyAsync(activation.data(), x.data, activation.size() * 2,
                                   cudaMemcpyDeviceToHost, stream));
        if (add_residual) {
            CUDA_CHECK(cudaMemcpy2DAsync(result_.data(), n_ * 2, out.data, out.nb[1],
                                         n_ * 2, t_, cudaMemcpyDeviceToHost, stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (int sample = 0; sample < 32; ++sample) {
            const int row = sample < 16 ? sample : (sample - 16) * (n_ - 1) / 15;
            rows_.push_back(row);
            const auto parent_row = static_cast<std::int64_t>(weight_row_offset + row);
            std::vector<std::uint8_t> codes(groups * 32), high(q5 ? groups * 8 : 0);
            std::vector<std::uint16_t> scales(groups);
            CUDA_CHECK(cudaMemcpyAsync(codes.data(), static_cast<const std::uint8_t*>(w.qdata) +
                parent_row * groups * 32, codes.size(), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(scales.data(), static_cast<const std::uint8_t*>(w.scales) +
                parent_row * groups * 2, scales.size() * 2, cudaMemcpyDeviceToHost, stream));
            if (q5) CUDA_CHECK(cudaMemcpyAsync(high.data(), static_cast<const std::uint8_t*>(w.qhigh) +
                parent_row * groups * 8, high.size(), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            for (int token = 0; token < t_; ++token) {
                double sum = add_residual ? bf16_value(result_[static_cast<std::size_t>(token) * n_ + row]) : 0;
                for (int column = 0; column < k; ++column) {
                    int code = (codes[column / 2] >> (4 * (column & 1))) & 15;
                    if (q5) code |= ((high[column / 8] >> (column & 7)) & 1) << 4;
                    const int sign = q5 ? 16 : 8;
                    if (code & sign) code -= sign * 2;
                    const double weight = code * f16_value(scales[column / 64]);
                    sum += weight * bf16_value(activation[static_cast<std::size_t>(token) * k + column]);
                }
                reference_.push_back(sum);
            }
        }
    }
    void finish() {
        if (!active_) return;
        CUDA_CHECK(cudaMemcpy2DAsync(result_.data(), n_ * 2, output_.data, output_.nb[1],
                                     n_ * 2, t_, cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        double error2 = 0, reference2 = 0, max_error = 0, max_reference = 0;
        for (std::size_t sample = 0; sample < rows_.size(); ++sample) {
            for (int token = 0; token < t_; ++token) {
                const double expected = reference_[sample * t_ + token];
                const double actual = bf16_value(result_[static_cast<std::size_t>(token) * n_ + rows_[sample]]);
                if (!std::isfinite(expected) || !std::isfinite(actual)) throw std::runtime_error("nonfinite real split-K result");
                const double error = actual - expected;
                error2 += error * error;
                reference2 += expected * expected;
                max_error = std::max(max_error, std::abs(error));
                max_reference = std::max(max_reference, std::abs(expected));
            }
        }
        const double relative_l2 = std::sqrt(error2 / std::max(reference2, 1e-30));
        // Existing A16 Linear criterion: one BF16 unit roundoff in relative L2,
        // plus a gross bound of 1/256 + (2/256)*maximum reference magnitude.
        const bool pass = relative_l2 <= 1.0 / 256 && max_error <= (1 + 2 * max_reference) / 256;
        std::cout << "REAL_FP64_ORACLE Q" << (q5_ ? 5 : 4) << " N=" << n_
                  << " K=" << k_ << " T=" << t_ << " residual=" << add_ << " samples=" << reference_.size()
                  << " relative_l2=" << relative_l2 << " max_error=" << max_error
                  << " pass=" << pass << std::endl;
        if (!pass) throw std::runtime_error("real split-K FP64 oracle failed");
    }
};
} // namespace ninfer::validation
