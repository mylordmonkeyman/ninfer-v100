#pragma once
// Volta QK + PV attention adapted from Strata's qsa_prompt_attn.cu at ad5206f.
// NInfer supplies BF16 queries, raw E4M3 KV, its selected blocks and BF16 output.
// MIT License
// Copyright (c) 2026 Niko1221 and the Strata contributors
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Included inside the target-private anonymous namespace after selected_token.
struct QsaVoltaFullSmem {
    __half qh[16][264], ql[16][264];
    __half k[32][264], v[32][264];
    float scores[16][33], qmax[4], alpha[16], lsum[16], mrow[16];
    std::int64_t rows[32];
};

__device__ __forceinline__ std::uint32_t qsa_pack_half(float x, float y) {
    const __half2 h = __floats2half2_rn(x, y);
    return *reinterpret_cast<const std::uint32_t*>(&h);
}

__global__ void __launch_bounds__(128) qsa_prefill_volta_full_mma_kernel(
    const __nv_bfloat16* __restrict__ query, const std::int32_t* __restrict__ positions,
    int table_row, const std::int32_t* __restrict__ blocks,
    const std::int32_t* __restrict__ counts, const std::int32_t* __restrict__ tables,
    int logical_pages, const __nv_fp8_e4m3* __restrict__ keys,
    const __nv_fp8_e4m3* __restrict__ values, __nv_bfloat16* __restrict__ output) {
    extern __shared__ __align__(16) unsigned char raw[];
    auto& s = *reinterpret_cast<QsaVoltaFullSmem*>(raw);
    const int token = blockIdx.y, kv = blockIdx.x;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int comp = (lane & 15) >> 2;
    const int row = 8 * (comp & 1) + (lane & 3) + ((lane & 16) ? 4 : 0);
    const int col = (lane & 3) + ((lane & 16) ? 4 : 0);
    query += (std::int64_t(token) * kQueryHeads + kv * 12) * kHeadDim;
    output += (std::int64_t(token) * kQueryHeads + kv * 12) * kHeadDim;
    const int complete = (positions[token] + 1) / 4;
    const int selected_count = counts[token];
    const bool dense = complete <= 512 && selected_count == complete;
    const int n = dense ? positions[token] + 1 : selected_count * 4 + ((positions[token] + 1) & 3);
    const auto* selected = blocks + std::int64_t(token) * 512;
    const auto* table = tables + std::int64_t(table_row) * logical_pages;
    float qm = 0;
    for (int i = t; i < 12 * 256; i += 128) qm = fmaxf(qm, fabsf(__bfloat162float(query[i])));
    for (int d = 16; d; d >>= 1) qm = fmaxf(qm, __shfl_xor_sync(0xffffffffU, qm, d));
    if (!lane) s.qmax[warp] = qm;
    __syncthreads();
    qm = fmaxf(fmaxf(s.qmax[0], s.qmax[1]), fmaxf(s.qmax[2], s.qmax[3]));
    int exponent = 0;
    if (qm > 0) frexpf(qm, &exponent);
    // ldexp on the value avoids an overflowing scale for tiny BF16 inputs.
    const int shift = max(-120, min(120, 14 - exponent));
    const float score_scale = ldexpf(kScale * 1.4426950408889634F, -shift);
    for (int i = t; i < 16 * 256; i += 128) {
        const int h = i / 256, d = i % 256;
        const float x = h < 12 ? ldexpf(__bfloat162float(query[i]), shift) : 0;
        const __half hi = __float2half_rn(x);
        s.qh[h][d] = hi;
        s.ql[h][d] = __float2half_rn(x - __half2float(hi));
    }
    if (t < 16) { s.mrow[t] = -INFINITY; s.lsum[t] = 0; }
    float acc[8][8]{};
    for (int start = 0; start < n; start += 32) {
        const int count = min(32, n - start);
        if (t < 32) {
            std::int64_t r = -1;
            if (t < count) {
                const int pos = dense ? start + t : selected_token(start + t, selected_count, complete, selected);
                r = (std::int64_t(table[pos / kPageTokens]) * kKvHeads + kv) * kPageTokens + pos % kPageTokens;
            }
            s.rows[t] = r;
        }
        __syncthreads(); // Previous chunk has finished reading all shared buffers.
        for (int i = t; i < 32 * 256; i += 128) {
            const int c = i / 256, d = i % 256;
            const auto r = s.rows[c];
            const float k = r < 0 ? 0 : ops::detail::decode_nvfp4_e4m3(keys[r * 256 + d].__x);
            const float v = r < 0 ? 0 : ops::detail::decode_nvfp4_e4m3(values[r * 256 + d].__x);
            // Every finite raw E4M3 value is exactly representable in FP16.
            s.k[c][d] = __float2half_rn(k); s.v[c][d] = __float2half_rn(v);
        }
        __syncthreads();
        ops::VoltaMma884Accumulator score{};
        for (int k = 0; k < 256; k += 4) {
            const auto b = ops::VoltaMma884Operand{
                *reinterpret_cast<const std::uint32_t*>(&s.k[warp * 8 + col][k]),
                *reinterpret_cast<const std::uint32_t*>(&s.k[warp * 8 + col][k + 2])};
            ops::volta_mma884_f16_f32(score, {
                *reinterpret_cast<const std::uint32_t*>(&s.qh[row][k]),
                *reinterpret_cast<const std::uint32_t*>(&s.qh[row][k + 2])}, b);
            ops::volta_mma884_f16_f32(score, {
                *reinterpret_cast<const std::uint32_t*>(&s.ql[row][k]),
                *reinterpret_cast<const std::uint32_t*>(&s.ql[row][k + 2])}, b);
        }
        if (comp < 2) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const auto at = ops::volta_mma884_accumulator_coordinate(lane, i);
                const int h = 8 * comp + at.row, c = warp * 8 + at.col;
                s.scores[h][c] = h < 12 && c < count ? score.x[i] * score_scale : -INFINITY;
            }
        }
        __syncthreads();
        const int h = t >> 3, sub = t & 7;
        float maximum = -INFINITY;
#pragma unroll
        for (int c = sub; c < 32; c += 8) maximum = fmaxf(maximum, s.scores[h][c]);
        for (int d = 1; d < 8; d <<= 1) maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffU, maximum, d, 8));
        maximum = fmaxf(maximum, s.mrow[h]);
        const float alpha = isfinite(s.mrow[h]) ? exp2f(s.mrow[h] - maximum) : 0;
        float sum = 0;
#pragma unroll
        for (int c = sub; c < 32; c += 8) {
            const float p = h < 12 && c < count ? exp2f(s.scores[h][c] - maximum) : 0;
            s.scores[h][c] = p; sum += p;
        }
        for (int d = 1; d < 8; d <<= 1) sum += __shfl_xor_sync(0xffffffffU, sum, d, 8);
        if (!sub) { s.alpha[h] = alpha; s.lsum[h] = s.lsum[h] * alpha + sum; s.mrow[h] = maximum; }
        __syncthreads();
        // Warp owns 64 output dimensions. Split scaled probabilities into FP16
        // hi+lo; FP32 MMA accumulation and online renormalization preserve range.
        float tmp[8][8]{};
        for (int c = 0; c < 32; c += 4) {
            const float p0 = s.scores[row][c] * 16384, p1 = s.scores[row][c + 1] * 16384;
            const float p2 = s.scores[row][c + 2] * 16384, p3 = s.scores[row][c + 3] * 16384;
            const auto ah0 = qsa_pack_half(p0, p1), ah1 = qsa_pack_half(p2, p3);
            const auto hh0 = __half22float2(*reinterpret_cast<const __half2*>(&ah0));
            const auto hh1 = __half22float2(*reinterpret_cast<const __half2*>(&ah1));
            const auto al0 = qsa_pack_half(p0 - hh0.x, p1 - hh0.y);
            const auto al1 = qsa_pack_half(p2 - hh1.x, p3 - hh1.y);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int d = warp * 64 + j * 8 + col;
                const auto b0 = qsa_pack_half(__half2float(s.v[c][d]), __half2float(s.v[c + 1][d]));
                const auto b1 = qsa_pack_half(__half2float(s.v[c + 2][d]), __half2float(s.v[c + 3][d]));
                ops::volta_mma884_f16_f32_raw(tmp[j], ah0, ah1, b0, b1);
                ops::volta_mma884_f16_f32_raw(tmp[j], al0, al1, b0, b1);
            }
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const auto at = ops::volta_mma884_accumulator_coordinate(lane, i);
                // Duplicate computations 2/3 still execute MMA: keep their shared
                // reads inside rows 0..15, and discard them at output.
                acc[j][i] = fmaf(acc[j][i], s.alpha[8 * (comp & 1) + at.row], tmp[j][i] / 16384);
            }
        }
    }
    __syncthreads();
    if (comp < 2) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const auto at = ops::volta_mma884_accumulator_coordinate(lane, i);
                const int h = 8 * comp + at.row, d = warp * 64 + j * 8 + at.col;
                if (h < 12) output[h * 256 + d] = __float2bfloat16_rn(s.lsum[h] > 0 ? acc[j][i] / s.lsum[h] : 0);
            }
        }
    }
}
