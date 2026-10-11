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
// All four independent Volta quad-pair products carry useful cells/dimensions.
// Adapted from Strata prompt_attn_v70_kernel, preserving its MIT notice above.
struct QsaVoltaQuadSmem {
    __half qh[16][264], ql[16][264];
    __half k[32][264], v[32][264];
    float part[4][16][36], probability[16][36];
    float qmax[4], alpha[16], lsum[16], mrow[16];
    std::int64_t rows[32];
};

__global__ void __launch_bounds__(128) qsa_prefill_volta_quad_mma_kernel(
    const __nv_bfloat16* __restrict__ query, const std::int32_t* __restrict__ positions,
    int table_row, const std::int32_t* __restrict__ blocks,
    const std::int32_t* __restrict__ counts, const std::int32_t* __restrict__ tables,
    int logical_pages, const __nv_fp8_e4m3* __restrict__ keys,
    const __nv_fp8_e4m3* __restrict__ values, __nv_bfloat16* __restrict__ output) {
    extern __shared__ __align__(16) unsigned char raw[];
    auto& s = *reinterpret_cast<QsaVoltaQuadSmem*>(raw);
    const int token = blockIdx.y, kv = blockIdx.x;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int qp = (lane >> 2) & 3, jj = (lane & 3) + 4 * (lane >> 4);
    const int crow = (lane & 1) + 4 * (lane >> 4), ccol = lane & 2;
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
#pragma unroll
    for (int d = 16; d; d >>= 1) qm = fmaxf(qm, __shfl_xor_sync(0xffffffffU, qm, d));
    if (!lane) s.qmax[warp] = qm;
    __syncthreads();
    qm = fmaxf(fmaxf(s.qmax[0], s.qmax[1]), fmaxf(s.qmax[2], s.qmax[3]));
    int exponent = 0;
    if (qm > 0) frexpf(qm, &exponent);
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
    float acc[2][2][8]{};
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
        __syncthreads();
        // Four adjacent raw bytes per load become four exact half values; this
        // removes the scalar FP8->FP32->FP16 conversion and preserves every code.
        for (int i = t; i < 32 * 64; i += 128) {
            const int c = i / 64, d = (i % 64) * 4;
            const auto r = s.rows[c];
            const auto* kp = reinterpret_cast<const std::uint8_t*>(keys);
            const auto* vp = reinterpret_cast<const std::uint8_t*>(values);
            const std::uint32_t k = r < 0 ? 0 : *reinterpret_cast<const std::uint32_t*>(kp + r*256+d);
            const std::uint32_t v = r < 0 ? 0 : *reinterpret_cast<const std::uint32_t*>(vp + r*256+d);
            const uint2 kh = make_uint2(
                unsigned(qsa_volta_fp8_half_bits(std::uint8_t(k))) |
                    (unsigned(qsa_volta_fp8_half_bits(std::uint8_t(k >> 8))) << 16),
                unsigned(qsa_volta_fp8_half_bits(std::uint8_t(k >> 16))) |
                    (unsigned(qsa_volta_fp8_half_bits(std::uint8_t(k >> 24))) << 16));
            const uint2 vh = make_uint2(
                unsigned(qsa_volta_fp8_half_bits(std::uint8_t(v))) |
                    (unsigned(qsa_volta_fp8_half_bits(std::uint8_t(v >> 8))) << 16),
                unsigned(qsa_volta_fp8_half_bits(std::uint8_t(v >> 16))) |
                    (unsigned(qsa_volta_fp8_half_bits(std::uint8_t(v >> 24))) << 16));
            *reinterpret_cast<uint2*>(&s.k[c][d]) = kh;
            *reinterpret_cast<uint2*>(&s.v[c][d]) = vh;
        }
        __syncthreads();
        float scores[2][8]{};
        const int cell = qp * 8 + jj;
#pragma unroll
        for (int step = 0; step < 16; ++step) {
            const int d = warp * 64 + step * 4;
            const uint2 b = *reinterpret_cast<const uint2*>(&s.k[cell][d]);
#pragma unroll
            for (int rb = 0; rb < 2; ++rb) {
                const uint2 ah = *reinterpret_cast<const uint2*>(&s.qh[rb * 8 + jj][d]);
                const uint2 al = *reinterpret_cast<const uint2*>(&s.ql[rb * 8 + jj][d]);
                ops::volta_mma884_f16_f32_raw(scores[rb], ah.x, ah.y, b.x, b.y);
                ops::volta_mma884_f16_f32_raw(scores[rb], al.x, al.y, b.x, b.y);
            }
        }
#pragma unroll
        for (int rb = 0; rb < 2; ++rb) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int h = rb * 8 + crow + 2 * ((i >> 1) & 1);
                const int c = qp * 8 + ccol + 4 * (i >> 2) + (i & 1);
                s.part[warp][h][c] = scores[rb][i];
            }
        }
        __syncthreads();
        const int h = t >> 3, sub = t & 7;
        float x[4], maximum = -INFINITY;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int c = sub * 4 + j;
            const float sum4 = ((s.part[0][h][c] + s.part[1][h][c]) + s.part[2][h][c]) + s.part[3][h][c];
            x[j] = h < 12 && c < count ? sum4 * score_scale : -INFINITY;
            maximum = fmaxf(maximum, x[j]);
        }
#pragma unroll
        for (int d = 1; d < 8; d <<= 1)
            maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffU, maximum, d, 8));
        const float old_maximum = s.mrow[h];
        maximum = fmaxf(maximum, old_maximum);
        float sum = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float p = x[j] == -INFINITY ? 0 : exp2f(x[j] - maximum);
            s.probability[h][sub * 4 + j] = p;
            sum += p;
        }
#pragma unroll
        for (int d = 1; d < 8; d <<= 1) sum += __shfl_xor_sync(0xffffffffU, sum, d, 8);
        if (!sub) {
            const float a = isfinite(old_maximum) ? exp2f(old_maximum - maximum) : 0;
            s.alpha[h] = a; s.lsum[h] = fmaf(s.lsum[h], a, sum); s.mrow[h] = maximum;
        }
        __syncthreads();
        float tmp[2][2][8]{};
#pragma unroll
        for (int step = 0; step < 8; ++step) {
            const int c = step * 4;
            uint2 ah[2], al[2];
#pragma unroll
            for (int rb = 0; rb < 2; ++rb) {
                const float4 p = *reinterpret_cast<const float4*>(&s.probability[rb * 8 + jj][c]);
                const float p0 = p.x * 16384, p1 = p.y * 16384;
                const float p2 = p.z * 16384, p3 = p.w * 16384;
                ah[rb] = make_uint2(qsa_pack_half(p0, p1), qsa_pack_half(p2, p3));
                const auto h0 = __half22float2(*reinterpret_cast<const __half2*>(&ah[rb].x));
                const auto h1 = __half22float2(*reinterpret_cast<const __half2*>(&ah[rb].y));
                al[rb] = make_uint2(qsa_pack_half(p0-h0.x, p1-h0.y), qsa_pack_half(p2-h1.x, p3-h1.y));
            }
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                const int d = warp * 64 + qp * 16 + nt * 8 + jj;
                const std::uint32_t b0 = qsa_pack_half(__half2float(s.v[c][d]), __half2float(s.v[c+1][d]));
                const std::uint32_t b1 = qsa_pack_half(__half2float(s.v[c+2][d]), __half2float(s.v[c+3][d]));
#pragma unroll
                for (int rb = 0; rb < 2; ++rb) {
                    ops::volta_mma884_f16_f32_raw(tmp[rb][nt], ah[rb].x, ah[rb].y, b0, b1);
                    ops::volta_mma884_f16_f32_raw(tmp[rb][nt], al[rb].x, al[rb].y, b0, b1);
                }
            }
        }
#pragma unroll
        for (int rb = 0; rb < 2; ++rb) {
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int h = rb * 8 + crow + 2 * ((i >> 1) & 1);
                    acc[rb][nt][i] = fmaf(acc[rb][nt][i], s.alpha[h], tmp[rb][nt][i] / 16384);
                }
            }
        }
    }
    __syncthreads();
#pragma unroll
    for (int rb = 0; rb < 2; ++rb) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int h = rb * 8 + crow + 2 * ((i >> 1) & 1);
            if (h < 12) {
#pragma unroll
                for (int nt = 0; nt < 2; ++nt) {
                    const int d = warp * 64 + qp * 16 + nt * 8 + ccol + 4 * (i >> 2) + (i & 1);
                    output[h * 256 + d] = __float2bfloat16_rn(
                        s.lsum[h] > 0 ? acc[rb][nt][i] / s.lsum[h] : 0);
                }
            }
        }
    }
}
