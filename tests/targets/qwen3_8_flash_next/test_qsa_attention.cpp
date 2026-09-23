#include "core/arena.h"
#include "core/device.h"
#include "ninfer/ops/linear.h"
#include "targets/qwen3_8_flash_next/impl/qsa_attention.h"
#include "targets/qwen3_8_flash_next/impl/qsa_attention_kernels.h"
#include "targets/qwen3_8_flash_next/impl/qsa_attention_workspace.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

namespace {

bool cuda_unavailable(cudaError_t error) {
    return error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver;
}

inline std::uint16_t float_to_bf16(float value) {
    const __nv_bfloat16 raw = __float2bfloat16_rn(value);
    return *reinterpret_cast<const std::uint16_t*>(&raw);
}

inline float bf16_to_float(std::uint16_t value) {
    return __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&value));
}

ninfer::Weight fp8_f32_weight(ninfer::DeviceBuffer& storage, std::int32_t rows,
                              std::int32_t columns) {
    const std::uint64_t codes        = static_cast<std::uint64_t>(rows) * columns;
    const std::uint64_t scale_offset = (codes + 255U) & ~std::uint64_t{255U};
    auto* payload                    = static_cast<std::byte*>(storage.p);
    ninfer::Weight out{};
    out.payload         = payload;
    out.payload_bytes   = storage.bytes;
    out.qdata           = payload;
    out.scales          = payload + scale_offset;
    out.qtype           = ninfer::QType::FP8_E4M3FN_ROW_F32S;
    out.layout          = ninfer::QuantLayout::RowScale;
    out.scale_dtype     = ninfer::DType::FP32;
    out.group_size      = static_cast<std::uint32_t>(columns);
    out.group           = columns;
    out.n               = rows;
    out.k               = columns;
    out.ndim            = 2;
    out.shape[0]        = rows;
    out.shape[1]        = columns;
    out.shape[2]        = 1;
    out.shape[3]        = 1;
    out.padded_shape[0] = rows;
    out.padded_shape[1] = columns;
    out.padded_shape[2] = 1;
    out.padded_shape[3] = 1;
    out.scale_ne[0]     = rows;
    out.scale_ne[1]     = 1;
    out.scale_ne[2]     = 1;
    out.scale_ne[3]     = 1;
    out.scale_nb[0]     = 4;
    out.scale_nb[1]     = static_cast<std::int64_t>(rows) * 4;
    out.scale_nb[2]     = out.scale_nb[1];
    out.scale_nb[3]     = out.scale_nb[1];
    return out;
}

std::uint64_t fp8_payload_bytes(std::int32_t rows, std::int32_t columns) {
    const std::uint64_t codes = static_cast<std::uint64_t>(rows) * columns;
    return ((codes + 255U) & ~std::uint64_t{255U}) + static_cast<std::uint64_t>(rows) * 4;
}

bool test_prefill_vs_decode_equivalence(ninfer::DeviceContext& device) {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    constexpr std::int32_t physical_pages = 64;
    constexpr std::int32_t logical_pages  = 64;
    constexpr std::int32_t input_dim      = 2'560;
    constexpr std::int32_t proj_rows      = 13'312;
    constexpr std::int32_t out_rows       = 2'560;
    constexpr std::int32_t out_cols       = 6'144;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    const std::uint64_t qgkv_codes        = static_cast<std::uint64_t>(proj_rows) * input_dim;
    const std::uint64_t qgkv_scale_offset = (qgkv_codes + 255U) & ~std::uint64_t{255U};
    std::vector<std::uint8_t> host_qgkv_codes(qgkv_codes);
    for (auto& v : host_qgkv_codes) { v = static_cast<std::uint8_t>((rng() % 120) + 1); }
    std::vector<float> host_qgkv_scales(proj_rows, 0.05f);

    ninfer::DeviceBuffer qgkv_storage(fp8_payload_bytes(proj_rows, input_dim));
    qgkv_storage.copy_from_host(host_qgkv_codes.data(), host_qgkv_codes.size());
    qgkv_storage.copy_from_host(host_qgkv_scales.data(), host_qgkv_scales.size() * sizeof(float),
                                qgkv_scale_offset);

    const std::uint64_t out_codes        = static_cast<std::uint64_t>(out_rows) * out_cols;
    const std::uint64_t out_scale_offset = (out_codes + 255U) & ~std::uint64_t{255U};
    std::vector<std::uint8_t> host_out_codes(out_codes);
    for (auto& v : host_out_codes) { v = static_cast<std::uint8_t>((rng() % 120) + 1); }
    std::vector<float> host_out_scales(out_rows, 0.05f);

    ninfer::DeviceBuffer out_storage(fp8_payload_bytes(out_rows, out_cols));
    out_storage.copy_from_host(host_out_codes.data(), host_out_codes.size());
    out_storage.copy_from_host(host_out_scales.data(), host_out_scales.size() * sizeof(float),
                               out_scale_offset);

    ninfer::DeviceBuffer query_norm(256 * sizeof(std::uint16_t));
    ninfer::DeviceBuffer key_norm(256 * sizeof(std::uint16_t));
    std::vector<std::uint16_t> host_qnorm(256), host_knorm(256);
    for (auto& v : host_qnorm) { v = float_to_bf16(dist(rng) * 0.1f); }
    for (auto& v : host_knorm) { v = float_to_bf16(dist(rng) * 0.1f); }
    query_norm.copy_from_host(host_qnorm.data(), host_qnorm.size() * 2);
    key_norm.copy_from_host(host_knorm.data(), host_knorm.size() * 2);
    device.synchronize(); // Ensure legacy stream copies order against device.stream

    AttentionWeights weights{};
    weights.query_norm           = ninfer::Tensor(query_norm.p, ninfer::DType::BF16, {256});
    weights.key_norm             = ninfer::Tensor(key_norm.p, ninfer::DType::BF16, {256});
    weights.query_gate_key_value = fp8_f32_weight(qgkv_storage, proj_rows, input_dim);
    weights.output               = fp8_f32_weight(out_storage, out_rows, out_cols);

    const std::vector<std::int32_t> test_t = {1, 2, 4, 8, 16, 64, 128, 512};
    for (std::int32_t T : test_t) {
        const std::int32_t first_token_index = 64;

        std::vector<std::uint16_t> host_input(static_cast<std::size_t>(input_dim) * T);
        for (auto& v : host_input) { v = float_to_bf16(dist(rng)); }

        std::vector<std::int32_t> host_pos(3 * T);
        for (std::int32_t t = 0; t < T; ++t) {
            host_pos[0 * T + t] = first_token_index + t;
            host_pos[1 * T + t] = first_token_index + t;
            host_pos[2 * T + t] = first_token_index + t;
        }

        std::vector<std::int32_t> host_selected_blocks(512 * T, -1);
        std::vector<std::int32_t> host_selected_counts(T);
        for (std::int32_t t = 0; t < T; ++t) {
            const int complete_blocks = (first_token_index + t + 1) / 4;
            const int count           = std::min(complete_blocks, 512);
            host_selected_counts[t]   = count;
            for (int k = 0; k < count; ++k) { host_selected_blocks[t * 512 + k] = k; }
        }

        ninfer::DeviceBuffer key_pages_a(256ULL * 64 * 2 * physical_pages * 2);
        ninfer::DeviceBuffer value_pages_a(256ULL * 64 * 2 * physical_pages * 2);
        ninfer::DeviceBuffer block_tables(logical_pages * sizeof(std::int32_t));

        // Pre-populate page 0 (tokens 0..63) with real non-zero K/V so selected_blocks (0..15)
        // produce realistic non-zero attended outputs.
        std::vector<std::uint16_t> init_kv(256ULL * 64 * 2 * physical_pages, 0);
        for (std::size_t i = 0; i < 256ULL * 64 * 2; ++i) {
            init_kv[i] = float_to_bf16(dist(rng));
        }
        key_pages_a.copy_from_host(init_kv.data(), init_kv.size() * sizeof(std::uint16_t));
        value_pages_a.copy_from_host(init_kv.data(), init_kv.size() * sizeof(std::uint16_t));

        std::array<std::int32_t, logical_pages> page_ids{};
        for (std::int32_t p = 0; p < logical_pages; ++p) { page_ids[p] = p; }
        block_tables.copy_from_host(page_ids.data(), sizeof(page_ids));

        QsaAttentionCacheView cache_a{
            .key_pages =
                ninfer::Tensor(key_pages_a.p, ninfer::DType::BF16, {256, 64, 2, physical_pages}),
            .value_pages =
                ninfer::Tensor(value_pages_a.p, ninfer::DType::BF16, {256, 64, 2, physical_pages}),
            .block_tables = ninfer::Tensor(block_tables.p, ninfer::DType::I32, {logical_pages, 1}),
        };

        ninfer::DeviceBuffer key_pages_b(256ULL * 64 * 2 * physical_pages * 2);
        ninfer::DeviceBuffer value_pages_b(256ULL * 64 * 2 * physical_pages * 2);
        key_pages_b.copy_from_host(init_kv.data(), init_kv.size() * sizeof(std::uint16_t));
        value_pages_b.copy_from_host(init_kv.data(), init_kv.size() * sizeof(std::uint16_t));

        QsaAttentionCacheView cache_b{
            .key_pages =
                ninfer::Tensor(key_pages_b.p, ninfer::DType::BF16, {256, 64, 2, physical_pages}),
            .value_pages =
                ninfer::Tensor(value_pages_b.p, ninfer::DType::BF16, {256, 64, 2, physical_pages}),
            .block_tables = ninfer::Tensor(block_tables.p, ninfer::DType::I32, {logical_pages, 1}),
        };

        ninfer::DeviceBuffer input_dev(static_cast<std::size_t>(input_dim) * T * 2);
        input_dev.copy_from_host(host_input.data(), host_input.size() * 2);
        device.synchronize(); // Ensure legacy stream copies order against device.stream

        // Path A: Sequential decode
        std::vector<std::uint16_t> seq_output(static_cast<std::size_t>(input_dim) * T);
        ninfer::DeviceBuffer token_buf(sizeof(std::int32_t));
        ninfer::DeviceBuffer pos_buf(3 * sizeof(std::int32_t));
        ninfer::DeviceBuffer row_buf(sizeof(std::int32_t));
        ninfer::DeviceBuffer sel_buf(512 * sizeof(std::int32_t));
        ninfer::DeviceBuffer cnt_buf(sizeof(std::int32_t));
        ninfer::DeviceBuffer out_slice(input_dim * sizeof(std::uint16_t));
        std::int32_t zero = 0;
        row_buf.copy_from_host(&zero, sizeof(zero));

        ninfer::WorkspaceArena ws_decode(flash_next_qsa_attention_workspace_capacity_bytes(1));

        for (std::int32_t t = 0; t < T; ++t) {
            std::int32_t tok_idx = first_token_index + t;
            std::array<std::int32_t, 3> pos_t = {host_pos[0 * T + t], host_pos[1 * T + t],
                                                 host_pos[2 * T + t]};
            token_buf.copy_from_host(&tok_idx, sizeof(tok_idx));
            pos_buf.copy_from_host(pos_t.data(), sizeof(pos_t));
            sel_buf.copy_from_host(&host_selected_blocks[t * 512], 512 * sizeof(std::int32_t));
            cnt_buf.copy_from_host(&host_selected_counts[t], sizeof(std::int32_t));
            device.synchronize(); // Ensure legacy stream copies order against device.stream

            ninfer::Tensor in_t(
                static_cast<std::uint16_t*>(input_dev.p) + static_cast<std::size_t>(t) * input_dim,
                ninfer::DType::BF16, {input_dim, 1});
            ninfer::Tensor out_t(out_slice.p, ninfer::DType::BF16, {input_dim, 1});
            ninfer::Tensor tok_t(token_buf.p, ninfer::DType::I32, {1});
            ninfer::Tensor pos_t_view(pos_buf.p, ninfer::DType::I32, {1, 3});
            ninfer::Tensor row_view(row_buf.p, ninfer::DType::I32, {1});
            ninfer::Tensor sel_view(sel_buf.p, ninfer::DType::I32, {512, 1});
            ninfer::Tensor cnt_view(cnt_buf.p, ninfer::DType::I32, {1});

            flash_next_qsa_attention_decode(in_t, weights, tok_t, pos_t_view, row_view, sel_view,
                                            cnt_view, cache_a, ws_decode, out_t, device.stream);
            device.synchronize();
            out_slice.copy_to_host(&seq_output[static_cast<std::size_t>(t) * input_dim],
                                   input_dim * 2);
        }

        // Path B: Chunked prefill
        ninfer::DeviceBuffer dev_indices(T * sizeof(std::int32_t));
        std::vector<std::int32_t> host_indices(T);
        for (std::int32_t t = 0; t < T; ++t) { host_indices[t] = first_token_index + t; }
        dev_indices.copy_from_host(host_indices.data(), T * sizeof(std::int32_t));

        ninfer::DeviceBuffer dev_mrope_pos(3 * T * sizeof(std::int32_t));
        dev_mrope_pos.copy_from_host(host_pos.data(), host_pos.size() * sizeof(std::int32_t));

        ninfer::DeviceBuffer dev_sel(512 * T * sizeof(std::int32_t));
        ninfer::DeviceBuffer dev_cnt(T * sizeof(std::int32_t));
        dev_sel.copy_from_host(host_selected_blocks.data(),
                               host_selected_blocks.size() * sizeof(std::int32_t));
        dev_cnt.copy_from_host(host_selected_counts.data(),
                               host_selected_counts.size() * sizeof(std::int32_t));
        device.synchronize(); // Ensure legacy stream copies order against device.stream

        ninfer::DeviceBuffer chunk_out(static_cast<std::size_t>(input_dim) * T * 2);
        ninfer::Tensor chunk_in(input_dev.p, ninfer::DType::BF16, {input_dim, T});
        ninfer::Tensor chunk_out_view(chunk_out.p, ninfer::DType::BF16, {input_dim, T});
        ninfer::Tensor chunk_idx(dev_indices.p, ninfer::DType::I32, {T});
        ninfer::Tensor chunk_pos(dev_mrope_pos.p, ninfer::DType::I32, {T, 3});
        ninfer::Tensor chunk_sel(dev_sel.p, ninfer::DType::I32, {512, T});
        ninfer::Tensor chunk_cnt(dev_cnt.p, ninfer::DType::I32, {T});

        ninfer::WorkspaceArena ws_prefill(flash_next_qsa_attention_workspace_capacity_bytes(T));
        flash_next_qsa_attention_prefill_chunk(chunk_in, weights, chunk_idx, chunk_pos, 0,
                                               chunk_sel, chunk_cnt, cache_b, ws_prefill,
                                               chunk_out_view, device.stream);
        device.synchronize();

        std::vector<std::uint16_t> chunk_output(static_cast<std::size_t>(input_dim) * T);
        chunk_out.copy_to_host(chunk_output.data(), chunk_output.size() * 2);

        // Compare KV cache entries
        std::vector<std::uint16_t> kp_a(256ULL * 64 * 2 * physical_pages);
        std::vector<std::uint16_t> kp_b(256ULL * 64 * 2 * physical_pages);
        key_pages_a.copy_to_host(kp_a.data(), kp_a.size() * 2);
        key_pages_b.copy_to_host(kp_b.data(), kp_b.size() * 2);
        double key_diff_sq = 0.0, key_base_sq = 0.0, key_max_diff = 0.0;
        int key_nan = 0;
        for (std::size_t i = 0; i < kp_a.size(); ++i) {
            float f_a = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&kp_a[i]));
            float f_b = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&kp_b[i]));
            if (!std::isfinite(f_a) || !std::isfinite(f_b)) {
                key_nan++;
                continue;
            }
            if (T < 16) {
                if (kp_a[i] != kp_b[i]) {
                    std::cerr << "Key cache bitwise mismatch at T=" << T << ", idx=" << i << ": seq=0x"
                              << std::hex << kp_a[i] << ", chunk=0x" << kp_b[i] << std::dec << "\n";
                    return false;
                }
            }
            double d = f_a - f_b;
            key_diff_sq += d * d;
            key_base_sq += static_cast<double>(f_a) * f_a;
            key_max_diff = std::max(key_max_diff, std::abs(d));
        }
        if (key_nan > 0) {
            std::cerr << "FAIL: Key cache contained " << key_nan << " non-finite elements at T=" << T << "\n";
            return false;
        }
        if (key_base_sq <= 0.0) {
            std::cerr << "FAIL: Key cache was vacuous at T=" << T << "\n";
            return false;
        }
        const double key_rel_l2 = std::sqrt(key_diff_sq / key_base_sq);

        std::vector<std::uint16_t> vp_a(256ULL * 64 * 2 * physical_pages);
        std::vector<std::uint16_t> vp_b(256ULL * 64 * 2 * physical_pages);
        value_pages_a.copy_to_host(vp_a.data(), vp_a.size() * 2);
        value_pages_b.copy_to_host(vp_b.data(), vp_b.size() * 2);
        double val_diff_sq = 0.0, val_base_sq = 0.0, val_max_diff = 0.0;
        int val_nan = 0;
        for (std::size_t i = 0; i < vp_a.size(); ++i) {
            float f_a = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&vp_a[i]));
            float f_b = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&vp_b[i]));
            if (!std::isfinite(f_a) || !std::isfinite(f_b)) {
                val_nan++;
                continue;
            }
            if (T < 16) {
                if (vp_a[i] != vp_b[i]) {
                    std::cerr << "Value cache bitwise mismatch at T=" << T << ", idx=" << i << ": seq=0x"
                              << std::hex << vp_a[i] << ", chunk=0x" << vp_b[i] << std::dec << "\n";
                    return false;
                }
            }
            double d = f_a - f_b;
            val_diff_sq += d * d;
            val_base_sq += static_cast<double>(f_a) * f_a;
            val_max_diff = std::max(val_max_diff, std::abs(d));
        }
        if (val_nan > 0) {
            std::cerr << "FAIL: Value cache contained " << val_nan << " non-finite elements at T=" << T << "\n";
            return false;
        }
        if (val_base_sq <= 0.0) {
            std::cerr << "FAIL: Value cache was vacuous at T=" << T << "\n";
            return false;
        }
        const double val_rel_l2 = std::sqrt(val_diff_sq / val_base_sq);

        // Compare output
        double diff_sq = 0.0, base_sq = 0.0, max_diff = 0.0;
        int nan_count = 0;
        for (std::size_t i = 0; i < seq_output.size(); ++i) {
            float f_seq =
                __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&seq_output[i]));
            float f_chk =
                __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&chunk_output[i]));
            if (!std::isfinite(f_seq) || !std::isfinite(f_chk)) {
                nan_count++;
                continue;
            }
            double d = f_seq - f_chk;
            diff_sq += d * d;
            base_sq += static_cast<double>(f_seq) * f_seq;
            max_diff = std::max(max_diff, std::abs(d));
        }
        double rel_l2 = (diff_sq == 0.0) ? 0.0 : (base_sq > 0.0 ? std::sqrt(diff_sq / base_sq) : std::sqrt(diff_sq));
        std::printf("  [QSA Equivalence] T=%3d: out_rel-L2 = %.6e (max-d=%.4e), key_rel-L2 = %.6e, val_rel-L2 = %.6e, base_sq=%.3e\n",
                    T, rel_l2, max_diff, key_rel_l2, val_rel_l2, base_sq);
        if (nan_count > 0) {
            std::cerr << "FAIL: T=" << T << " output contained " << nan_count << " non-finite elements\n";
            return false;
        }
        if (base_sq <= 0.0) {
            std::cerr << "FAIL: T=" << T << " output was vacuous (base_sq <= 0)\n";
            return false;
        }
        const double out_tol = T >= 16 ? 8e-2 : 1e-3;
        const double kv_tol  = T >= 16 ? 4e-2 : 1e-3;
        if (rel_l2 > out_tol) {
            std::cerr << "FAIL: T=" << T << " out rel-L2 " << rel_l2 << " > tol " << out_tol << "\n";
            return false;
        }
        if (key_rel_l2 > kv_tol) {
            std::cerr << "FAIL: T=" << T << " key rel-L2 " << key_rel_l2 << " > tol " << kv_tol << "\n";
            return false;
        }
        if (val_rel_l2 > kv_tol) {
            std::cerr << "FAIL: T=" << T << " val rel-L2 " << val_rel_l2 << " > tol " << kv_tol << "\n";
            return false;
        }
    }
    return true;
}

int test_g17_prefill_mma(ninfer::DeviceContext& device,
                         ninfer::targets::qwen3_8_flash_next::detail::AttentionWeights& weights) {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    constexpr std::int32_t input_dim = 2'560;
    std::uniform_real_distribution<float> dist(-0.4f, 0.4f);

    auto run_t = [&](std::int32_t T, bool use_mma, std::vector<std::uint16_t>& host_out,
                     int timed_iters, float& ms_out) -> int {
        std::mt19937 rng(0xC17u + static_cast<unsigned>(T));
        const std::int32_t first           = 0;
        const std::int32_t physical_pages  = std::max(8, (T + 63) / 64 + 2);
        const std::int32_t logical_pages   = physical_pages;
        std::vector<std::uint16_t> host_in(static_cast<std::size_t>(input_dim) * T);
        for (auto& v : host_in) { v = float_to_bf16(dist(rng)); }
        std::vector<std::int32_t> host_pos(3 * T);
        std::vector<std::int32_t> host_tok(T);
        std::vector<std::int32_t> host_sel(512 * T, -1);
        std::vector<std::int32_t> host_cnt(T);
        for (std::int32_t t = 0; t < T; ++t) {
            host_tok[t]             = first + t;
            host_pos[0 * T + t]     = first + t;
            host_pos[1 * T + t]     = first + t;
            host_pos[2 * T + t]     = first + t;
            const int complete      = (first + t + 1) / 4;
            const int count         = std::min(complete, 512);
            host_cnt[t]             = count;
            for (int k = 0; k < count; ++k) { host_sel[t * 512 + k] = k; }
        }
        ninfer::DeviceBuffer in_b(host_in.size() * 2);
        ninfer::DeviceBuffer out_b(host_in.size() * 2);
        ninfer::DeviceBuffer tok_b(T * 4);
        ninfer::DeviceBuffer pos_b(3 * T * 4);
        ninfer::DeviceBuffer sel_b(512 * T * 4);
        ninfer::DeviceBuffer cnt_b(T * 4);
        ninfer::DeviceBuffer key_pages(256ULL * 64 * 2 * physical_pages * 2);
        ninfer::DeviceBuffer value_pages(256ULL * 64 * 2 * physical_pages * 2);
        ninfer::DeviceBuffer block_tables(logical_pages * 4);
        std::vector<std::uint16_t> init_kv(256ULL * 64 * 2 * physical_pages);
        for (auto& v : init_kv) { v = float_to_bf16(dist(rng)); }
        std::vector<std::int32_t> pages(logical_pages);
        for (std::int32_t p = 0; p < logical_pages; ++p) { pages[p] = p; }
        in_b.copy_from_host(host_in.data(), host_in.size() * 2);
        tok_b.copy_from_host(host_tok.data(), host_tok.size() * 4);
        pos_b.copy_from_host(host_pos.data(), host_pos.size() * 4);
        sel_b.copy_from_host(host_sel.data(), host_sel.size() * 4);
        cnt_b.copy_from_host(host_cnt.data(), host_cnt.size() * 4);
        key_pages.copy_from_host(init_kv.data(), init_kv.size() * 2);
        value_pages.copy_from_host(init_kv.data(), init_kv.size() * 2);
        block_tables.copy_from_host(pages.data(), pages.size() * 4);
        device.synchronize();
        QsaAttentionCacheView cache{
            .key_pages    = ninfer::Tensor(key_pages.p, ninfer::DType::BF16,
                                           {256, 64, 2, physical_pages}),
            .value_pages  = ninfer::Tensor(value_pages.p, ninfer::DType::BF16,
                                           {256, 64, 2, physical_pages}),
            .block_tables = ninfer::Tensor(block_tables.p, ninfer::DType::I32, {logical_pages, 1}),
        };
        ninfer::Tensor tin(in_b.p, ninfer::DType::BF16, {input_dim, T});
        ninfer::Tensor tout(out_b.p, ninfer::DType::BF16, {input_dim, T});
        ninfer::Tensor ttok(tok_b.p, ninfer::DType::I32, {T});
        ninfer::Tensor tpos(pos_b.p, ninfer::DType::I32, {T, 3});
        ninfer::Tensor tsel(sel_b.p, ninfer::DType::I32, {512, T});
        ninfer::Tensor tcnt(cnt_b.p, ninfer::DType::I32, {T});
        ninfer::WorkspaceArena ws(flash_next_qsa_attention_workspace_capacity_bytes(T));
        flash_next_qsa_attention_prefill_chunk(tin, weights, ttok, tpos, 0, tsel, tcnt, cache, ws,
                                               tout, device.stream, {}, use_mma);
        device.synchronize();
        host_out.resize(host_in.size());
        out_b.copy_to_host(host_out.data(), host_out.size() * 2);
        if (timed_iters > 0) {
            for (int i = 0; i < 2; ++i) {
                flash_next_qsa_attention_prefill_chunk(tin, weights, ttok, tpos, 0, tsel, tcnt,
                                                       cache, ws, tout, device.stream, {}, use_mma);
            }
            device.synchronize();
            cudaEvent_t start, stop;
            CUDA_CHECK(cudaEventCreate(&start));
            CUDA_CHECK(cudaEventCreate(&stop));
            CUDA_CHECK(cudaEventRecord(start, device.stream));
            for (int i = 0; i < timed_iters; ++i) {
                flash_next_qsa_attention_prefill_chunk(tin, weights, ttok, tpos, 0, tsel, tcnt,
                                                       cache, ws, tout, device.stream, {}, use_mma);
            }
            CUDA_CHECK(cudaEventRecord(stop, device.stream));
            CUDA_CHECK(cudaEventSynchronize(stop));
            CUDA_CHECK(cudaEventElapsedTime(&ms_out, start, stop));
            ms_out /= static_cast<float>(timed_iters);
            CUDA_CHECK(cudaEventDestroy(start));
            CUDA_CHECK(cudaEventDestroy(stop));
        }
        return 0;
    };

    for (std::int32_t T : {128, 512, 2048}) {
        std::vector<std::uint16_t> old_out, mma_a, mma_b;
        float unused = 0.0f;
        if (run_t(T, false, old_out, 0, unused) != 0) { return 1; }
        if (run_t(T, true, mma_a, 0, unused) != 0) { return 1; }
        if (run_t(T, true, mma_b, 0, unused) != 0) { return 1; }
        if (mma_a != mma_b) {
            std::cerr << "G17 two-run bitwise mismatch at T=" << T << "\n";
            return 1;
        }
        double diff_sq = 0.0, base_sq = 0.0, max_abs = 0.0;
        int nan = 0, nonzero = 0;
        for (std::size_t i = 0; i < old_out.size(); ++i) {
            const float a = bf16_to_float(old_out[i]);
            const float b = bf16_to_float(mma_a[i]);
            if (!std::isfinite(a) || !std::isfinite(b)) {
                ++nan;
                continue;
            }
            if ((old_out[i] & 0x7FFFU) != 0) { ++nonzero; }
            const double d = static_cast<double>(a) - static_cast<double>(b);
            diff_sq += d * d;
            base_sq += static_cast<double>(a) * a;
            max_abs = std::max(max_abs, std::abs(d));
        }
        if (nan != 0 || base_sq <= 0.0 || nonzero == 0) {
            std::cerr << "G17 vacuous/nonfinite T=" << T << " nan=" << nan << " base_sq=" << base_sq
                      << " nonzero=" << nonzero << "\n";
            return 1;
        }
        const double rel = std::sqrt(diff_sq / base_sq);
        std::cout << "G17 T=" << T << " rel-L2=" << rel << " max_abs=" << max_abs
                  << " two-run bitwise OK nonzero=" << nonzero << "\n";
        // Current MMA kernel is bitwise with the scalar path (rel-L2=0). A schedule
        // that is only rounding P to BF16 should sit near 1e-3; 0.5 is a defect.
        constexpr double kMmaScalarRelL2Gate = 1.0e-3;
        if (rel > kMmaScalarRelL2Gate) {
            std::cerr << "FAIL: G17 T=" << T << " MMA vs scalar rel-L2=" << rel
                      << " exceeds " << kMmaScalarRelL2Gate << "\n";
            return 1;
        }
    }

    std::cout << "G17 prefill_chunk wall (includes QGKV+out linear):\n";
    float t_old[4] = {};
    float t_mma[4] = {};
    const std::int32_t sweep[] = {512, 1024, 2048, 4096};
    for (int i = 0; i < 4; ++i) {
        std::vector<std::uint16_t> dummy;
        const int iters = sweep[i] >= 2048 ? 5 : 10;
        if (run_t(sweep[i], false, dummy, iters, t_old[i]) != 0) { return 1; }
        if (run_t(sweep[i], true, dummy, iters, t_mma[i]) != 0) { return 1; }
        std::cout << "  T=" << sweep[i] << " old=" << t_old[i] << " ms mma=" << t_mma[i] << " ms\n";
    }
    auto exponent = [](float a, float b, float t0, float t1) {
        return std::log(b / a) / std::log(t1 / t0);
    };
    std::cout << "G17 exponent T=1024/512 old=" << exponent(t_old[0], t_old[1], 512, 1024)
              << " mma=" << exponent(t_mma[0], t_mma[1], 512, 1024) << "\n";
    std::cout << "G17 exponent T=2048/1024 old=" << exponent(t_old[1], t_old[2], 1024, 2048)
              << " mma=" << exponent(t_mma[1], t_mma[2], 1024, 2048) << "\n";
    std::cout << "G17 exponent T=4096/2048 old=" << exponent(t_old[2], t_old[3], 2048, 4096)
              << " mma=" << exponent(t_mma[2], t_mma[3], 2048, 4096) << "\n";
    std::cout << "PASS: test_g17_prefill_mma\n";
    return 0;
}

// The MTP draft head runs decode attention with selected_count == 0 (it never runs an indexer), so
// it only sees the tail of the current 4-token block. At token_index == 7 the tail is empty
// ((7 + 1) % 4 == 0) and the kernel's softmax loop never executes: running_sum stays 0. Before the
// guard that was 0/0 -> NaN, which propagated to the draft logits and made argmax return token 0 on
// one draft round in four. Both KV storage instantiations of sparse_attention_kernel are covered.
int test_empty_selection_decode_is_finite(
    ninfer::DeviceContext& device,
    const ninfer::targets::qwen3_8_flash_next::detail::AttentionWeights& weights) {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    constexpr std::int32_t batch          = 1;
    constexpr std::int32_t physical_pages = 2;
    constexpr std::int32_t logical_pages  = 2;
    constexpr std::int32_t input_dim      = 2'560;
    constexpr std::int32_t token_index    = 7; // (token_index + 1) % 4 == 0 -> tail_count == 0

    ninfer::DeviceBuffer input(input_dim * batch * sizeof(std::uint16_t));
    ninfer::DeviceBuffer output(input_dim * batch * sizeof(std::uint16_t));
    ninfer::DeviceBuffer token_indices(batch * sizeof(std::int32_t));
    ninfer::DeviceBuffer mrope_positions(batch * 3 * sizeof(std::int32_t));
    ninfer::DeviceBuffer table_rows(batch * sizeof(std::int32_t));
    ninfer::DeviceBuffer selected_blocks(512 * batch * sizeof(std::int32_t));
    ninfer::DeviceBuffer selected_counts(batch * sizeof(std::int32_t));
    ninfer::DeviceBuffer block_tables(logical_pages * sizeof(std::int32_t));

    std::vector<std::uint16_t> host_input(input_dim * batch, 0);
    host_input[0] = 0x3F80U; // 1.0 -> non-zero query, so a NaN cannot be masked by a zero query
    input.copy_from_host(host_input.data(), host_input.size() * sizeof(std::uint16_t));
    mrope_positions.fill(0);
    table_rows.fill(0);
    selected_blocks.fill(0);
    selected_counts.fill(0); // the MTP head's selected_counts are memset to 0 and never written
    std::array<std::int32_t, logical_pages> host_block_tables = {0, 1};
    block_tables.copy_from_host(host_block_tables.data(), sizeof(host_block_tables));
    const std::int32_t host_token_index = token_index;
    token_indices.copy_from_host(&host_token_index, sizeof(host_token_index));
    device.synchronize(); // Ensure legacy stream copies order against device.stream

    ninfer::Tensor input_tensor(input.p, ninfer::DType::BF16, {input_dim, batch});
    ninfer::Tensor output_tensor(output.p, ninfer::DType::BF16, {input_dim, batch});
    ninfer::Tensor token_indices_tensor(token_indices.p, ninfer::DType::I32, {batch});
    ninfer::Tensor mrope_positions_tensor(mrope_positions.p, ninfer::DType::I32, {batch, 3});
    ninfer::Tensor table_rows_tensor(table_rows.p, ninfer::DType::I32, {batch});
    ninfer::Tensor selected_blocks_tensor(selected_blocks.p, ninfer::DType::I32, {512, batch});
    ninfer::Tensor selected_counts_tensor(selected_counts.p, ninfer::DType::I32, {batch});

    for (int fp8_kv = 0; fp8_kv < 2; ++fp8_kv) {
        const std::size_t element_bytes = fp8_kv != 0 ? 1U : sizeof(std::uint16_t);
        ninfer::DeviceBuffer key_pages(256ULL * 64 * 2 * physical_pages * element_bytes);
        ninfer::DeviceBuffer value_pages(256ULL * 64 * 2 * physical_pages * element_bytes);
        key_pages.fill(0);
        value_pages.fill(0);
        output.fill(0xFF); // poison: a kernel that never writes would read as non-finite here

        const ninfer::DType kv_dtype =
            fp8_kv != 0 ? ninfer::DType::FP8_E4M3FN : ninfer::DType::BF16;
        QsaAttentionCacheView cache{
            .key_pages   = ninfer::Tensor(key_pages.p, kv_dtype, {256, 64, 2, physical_pages}),
            .value_pages = ninfer::Tensor(value_pages.p, kv_dtype, {256, 64, 2, physical_pages}),
            .block_tables = ninfer::Tensor(block_tables.p, ninfer::DType::I32, {logical_pages, 1}),
        };

        ninfer::WorkspaceArena workspace(flash_next_qsa_attention_workspace_capacity_bytes(batch));
        flash_next_qsa_attention_decode(input_tensor, weights, token_indices_tensor,
                                        mrope_positions_tensor, table_rows_tensor,
                                        selected_blocks_tensor, selected_counts_tensor, cache,
                                        workspace, output_tensor, device.stream);
        device.synchronize();

        std::vector<std::uint16_t> host_output(input_dim * batch);
        output.copy_to_host(host_output.data(), host_output.size() * sizeof(std::uint16_t));
        for (std::size_t i = 0; i < host_output.size(); ++i) {
            // Attended == 0 gates to 0 and the A16Only output linear carries it through exactly, so
            // the whole row must be BF16 +0. Anything else (0x7FC0/0xFFFF...) is the NaN or the
            // untouched poison.
            if (host_output[i] != 0x0000U) {
                std::cerr << "FAIL: empty-selection decode (" << (fp8_kv != 0 ? "FP8" : "BF16")
                          << " KV) produced " << bf16_to_float(host_output[i]) << " at idx=" << i
                          << " (raw 0x" << std::hex << host_output[i] << std::dec
                          << "), expected finite 0\n";
                return 1;
            }
        }
    }
    std::cout << "PASS: empty-selection decode (selected_count=0, token_index=7) is finite\n";
    return 0;
}

} // namespace

int main() {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    int device_count              = 0;
    const cudaError_t count_error = cudaGetDeviceCount(&device_count);
    if (cuda_unavailable(count_error) || device_count == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    CUDA_CHECK(count_error);

    ninfer::DeviceContext device(0);
    constexpr std::int32_t batch          = 1;
    constexpr std::int32_t physical_pages = 2;
    constexpr std::int32_t logical_pages  = 2;
    constexpr std::int32_t input_dim      = 2'560;
    constexpr std::int32_t proj_rows      = 13'312;
    constexpr std::int32_t out_rows       = 2'560;
    constexpr std::int32_t out_cols       = 6'144;

    ninfer::DeviceBuffer input(input_dim * batch * sizeof(std::uint16_t));
    ninfer::DeviceBuffer output(input_dim * batch * sizeof(std::uint16_t));
    ninfer::DeviceBuffer query_norm(256 * sizeof(std::uint16_t));
    ninfer::DeviceBuffer key_norm(256 * sizeof(std::uint16_t));
    ninfer::DeviceBuffer qgkv_storage(fp8_payload_bytes(proj_rows, input_dim));
    ninfer::DeviceBuffer out_storage(fp8_payload_bytes(out_rows, out_cols));
    ninfer::DeviceBuffer token_indices(batch * sizeof(std::int32_t));
    ninfer::DeviceBuffer mrope_positions(batch * 3 * sizeof(std::int32_t));
    ninfer::DeviceBuffer table_rows(batch * sizeof(std::int32_t));
    ninfer::DeviceBuffer selected_blocks(512 * batch * sizeof(std::int32_t));
    ninfer::DeviceBuffer selected_counts(batch * sizeof(std::int32_t));
    ninfer::DeviceBuffer key_pages(256ULL * 64 * 2 * physical_pages * sizeof(std::uint16_t));
    ninfer::DeviceBuffer value_pages(256ULL * 64 * 2 * physical_pages * sizeof(std::uint16_t));
    ninfer::DeviceBuffer block_tables(logical_pages * sizeof(std::int32_t));

    // 1. Input: first BF16 value 1.0 (0x3F80), remaining 0
    std::vector<std::uint16_t> host_input(input_dim * batch, 0);
    host_input[0] = 0x3F80U;
    input.copy_from_host(host_input.data(), host_input.size() * sizeof(std::uint16_t));
    output.fill(0);

    // 2. Query & Key norm: zero norm deltas
    query_norm.fill(0);
    key_norm.fill(0);

    // 3. QGKV FP8 weight:
    // Rows: per query head [q256, gate256] for 24 heads (12288 rows),
    // then key [2, 256] (rows 12288..12799), then value [2, 256] (rows 12800..13311).
    // q, k, v rows have W[row, 0] = 0x38 (1.0 in FP8 E4M3); gate rows have W[row, 0] = 0x00 (0.0).
    // All other column weights are 0x00. All row scales are 1.0F.
    const std::uint64_t qgkv_codes        = static_cast<std::uint64_t>(proj_rows) * input_dim;
    const std::uint64_t qgkv_scale_offset = (qgkv_codes + 255U) & ~std::uint64_t{255U};
    std::vector<std::uint8_t> host_qgkv_codes(qgkv_codes, 0);
    for (std::int32_t head = 0; head < 24; ++head) {
        for (std::int32_t dim = 0; dim < 256; ++dim) {
            const std::int32_t q_row                                     = head * 512 + dim;
            host_qgkv_codes[static_cast<std::size_t>(q_row) * input_dim] = 0x38U;
            // gate row is head * 512 + 256 + dim, stays 0x00
        }
    }
    for (std::int32_t r = 12'288; r < 13'312; ++r) {
        host_qgkv_codes[static_cast<std::size_t>(r) * input_dim] = 0x38U;
    }
    std::vector<float> host_qgkv_scales(proj_rows, 1.0F);
    qgkv_storage.fill(0);
    qgkv_storage.copy_from_host(host_qgkv_codes.data(), host_qgkv_codes.size());
    qgkv_storage.copy_from_host(host_qgkv_scales.data(), host_qgkv_scales.size() * sizeof(float),
                                qgkv_scale_offset);

    // 4. Output FP8 weight [2560, 6144]:
    // Map the first gated attended channel (channel 0) to every output row:
    // W[row, 0] = 0x38 (1.0 in FP8 E4M3), W[row, col > 0] = 0. Scale = 1.0F.
    const std::uint64_t out_codes        = static_cast<std::uint64_t>(out_rows) * out_cols;
    const std::uint64_t out_scale_offset = (out_codes + 255U) & ~std::uint64_t{255U};
    std::vector<std::uint8_t> host_out_codes(out_codes, 0);
    for (std::int32_t r = 0; r < out_rows; ++r) {
        host_out_codes[static_cast<std::size_t>(r) * out_cols] = 0x38U;
    }
    std::vector<float> host_out_scales(out_rows, 1.0F);
    out_storage.fill(0);
    out_storage.copy_from_host(host_out_codes.data(), host_out_codes.size());
    out_storage.copy_from_host(host_out_scales.data(), host_out_scales.size() * sizeof(float),
                               out_scale_offset);

    // 5. Positions, token index, table rows, selected blocks
    token_indices.fill(0); // token 0
    mrope_positions.fill(0);
    table_rows.fill(0);
    selected_blocks.fill(0);
    selected_counts.fill(0);

    // 6. Cache pages & block tables
    key_pages.fill(0);
    value_pages.fill(0);
    std::array<std::int32_t, logical_pages> host_block_tables = {0, 1};
    block_tables.copy_from_host(host_block_tables.data(), sizeof(host_block_tables));

    // Construct AttentionWeights & CacheView
    AttentionWeights weights{};
    weights.query_norm           = ninfer::Tensor(query_norm.p, ninfer::DType::BF16, {256});
    weights.key_norm             = ninfer::Tensor(key_norm.p, ninfer::DType::BF16, {256});
    weights.query_gate_key_value = fp8_f32_weight(qgkv_storage, proj_rows, input_dim);
    weights.output               = fp8_f32_weight(out_storage, out_rows, out_cols);

    QsaAttentionCacheView cache{
        .key_pages = ninfer::Tensor(key_pages.p, ninfer::DType::BF16, {256, 64, 2, physical_pages}),
        .value_pages =
            ninfer::Tensor(value_pages.p, ninfer::DType::BF16, {256, 64, 2, physical_pages}),
        .block_tables = ninfer::Tensor(block_tables.p, ninfer::DType::I32, {logical_pages, 1}),
    };

    ninfer::Tensor input_tensor(input.p, ninfer::DType::BF16, {input_dim, batch});
    ninfer::Tensor output_tensor(output.p, ninfer::DType::BF16, {input_dim, batch});
    ninfer::Tensor token_indices_tensor(token_indices.p, ninfer::DType::I32, {batch});
    ninfer::Tensor mrope_positions_tensor(mrope_positions.p, ninfer::DType::I32, {batch, 3});
    ninfer::Tensor table_rows_tensor(table_rows.p, ninfer::DType::I32, {batch});
    ninfer::Tensor selected_blocks_tensor(selected_blocks.p, ninfer::DType::I32, {512, batch});
    ninfer::Tensor selected_counts_tensor(selected_counts.p, ninfer::DType::I32, {batch});

    // Execute QSA attention decode
    ninfer::WorkspaceArena workspace(flash_next_qsa_attention_workspace_capacity_bytes(batch));
    flash_next_qsa_attention_decode(input_tensor, weights, token_indices_tensor,
                                    mrope_positions_tensor, table_rows_tensor,
                                    selected_blocks_tensor, selected_counts_tensor, cache,
                                    workspace, output_tensor, device.stream);
    device.synchronize();

    // Verify output: every output BF16 must be 0.5 (0x3F00)
    std::vector<std::uint16_t> host_output(input_dim * batch);
    output.copy_to_host(host_output.data(), host_output.size() * sizeof(std::uint16_t));
    int failures = 0;
    for (std::size_t i = 0; i < host_output.size(); ++i) {
        if (host_output[i] != 0x3F00U) {
            std::cerr << "Mismatch at output[" << i << "]: expected 0x3F00 (0.5), got 0x"
                      << std::hex << host_output[i] << std::dec << "\n";
            failures++;
            if (failures > 10) break;
        }
    }

    // Verify appended key and value cache:
    // Physical page 0, token 0: both KV heads, 256 dims must be 0x3F80 (1.0)
    std::vector<std::uint16_t> host_key_pages(256ULL * 64 * 2 * physical_pages);
    std::vector<std::uint16_t> host_value_pages(256ULL * 64 * 2 * physical_pages);
    key_pages.copy_to_host(host_key_pages.data(), host_key_pages.size() * sizeof(std::uint16_t));
    value_pages.copy_to_host(host_value_pages.data(),
                             host_value_pages.size() * sizeof(std::uint16_t));

    for (int kv_head = 0; kv_head < 2; ++kv_head) {
        for (int dim = 0; dim < 256; ++dim) {
            // Layout: [256, 64, 2, physical_pages]
            // index: ((page * 2 + head) * 64 + token) * 256 + dim
            const std::size_t idx = ((0 * 2 + kv_head) * 64 + 0) * 256 + dim;
            if (host_key_pages[idx] != 0x3F80U) {
                std::cerr << "Key cache mismatch at head " << kv_head << ", dim " << dim
                          << ": expected 0x3F80 (1.0), got 0x" << std::hex << host_key_pages[idx]
                          << std::dec << "\n";
                failures++;
                if (failures > 20) break;
            }
            if (host_value_pages[idx] != 0x3F80U) {
                std::cerr << "Value cache mismatch at head " << kv_head << ", dim " << dim
                          << ": expected 0x3F80 (1.0), got 0x" << std::hex << host_value_pages[idx]
                          << std::dec << "\n";
                failures++;
                if (failures > 20) break;
            }
        }
    }

    // Verify untouched cache entries remain 0
    for (std::size_t i = 0; i < host_key_pages.size(); ++i) {
        // page 0 token 0 is indices in [0, 256) for head 0 and [64*256, 64*256+256) for head 1
        const bool is_head0_tok0 = (i < 256);
        const bool is_head1_tok0 = (i >= 64 * 256 && i < 64 * 256 + 256);
        if (!is_head0_tok0 && !is_head1_tok0) {
            if (host_key_pages[i] != 0) {
                std::cerr << "Untouched key cache dirty at " << i << ": 0x" << std::hex
                          << host_key_pages[i] << std::dec << "\n";
                failures++;
                break;
            }
            if (host_value_pages[i] != 0) {
                std::cerr << "Untouched value cache dirty at " << i << ": 0x" << std::hex
                          << host_value_pages[i] << std::dec << "\n";
                failures++;
                break;
            }
        }
    }

    if (failures != 0) {
        std::cerr << "FAIL: " << failures << " errors in test_qsa_attention\n";
        return 1;
    }

    if (test_empty_selection_decode_is_finite(device, weights) != 0) {
        std::cerr << "FAIL: test_empty_selection_decode_is_finite\n";
        return 1;
    }

    if (!test_prefill_vs_decode_equivalence(device)) {
        std::cerr << "FAIL: test_prefill_vs_decode_equivalence failed\n";
        return 1;
    }

    if (test_g17_prefill_mma(device, weights) != 0) {
        std::cerr << "FAIL: test_g17_prefill_mma\n";
        return 1;
    }

    // Decode batch test B=1..8
    for (std::int32_t b = 1; b <= 8; ++b) {
        ninfer::DeviceBuffer in_b(input_dim * b * sizeof(std::uint16_t));
        ninfer::DeviceBuffer out_b(input_dim * b * sizeof(std::uint16_t));
        ninfer::DeviceBuffer tok_b(b * sizeof(std::int32_t));
        ninfer::DeviceBuffer pos_b(b * 3 * sizeof(std::int32_t));
        ninfer::DeviceBuffer row_b(b * sizeof(std::int32_t));
        ninfer::DeviceBuffer sel_b(512 * b * sizeof(std::int32_t));
        ninfer::DeviceBuffer cnt_b(b * sizeof(std::int32_t));

        std::vector<std::uint16_t> h_in(input_dim * b, 0);
        for (int i = 0; i < b; ++i) { h_in[i * input_dim] = 0x3F80U; }
        in_b.copy_from_host(h_in.data(), h_in.size() * sizeof(std::uint16_t));
        out_b.fill(0);
        tok_b.fill(0);
        pos_b.fill(0);
        row_b.fill(0);
        sel_b.fill(0);
        cnt_b.fill(0);

        ninfer::Tensor in_t(in_b.p, ninfer::DType::BF16, {input_dim, b});
        ninfer::Tensor out_t(out_b.p, ninfer::DType::BF16, {input_dim, b});
        ninfer::Tensor tok_t(tok_b.p, ninfer::DType::I32, {b});
        ninfer::Tensor pos_t(pos_b.p, ninfer::DType::I32, {b, 3});
        ninfer::Tensor row_t(row_b.p, ninfer::DType::I32, {b});
        ninfer::Tensor sel_t(sel_b.p, ninfer::DType::I32, {512, b});
        ninfer::Tensor cnt_t(cnt_b.p, ninfer::DType::I32, {b});

        ninfer::WorkspaceArena ws(flash_next_qsa_attention_workspace_capacity_bytes(b));
        flash_next_qsa_attention_decode(in_t, weights, tok_t, pos_t, row_t, sel_t, cnt_t, cache, ws,
                                        out_t, device.stream);
        device.synchronize();

        std::vector<std::uint16_t> h_out(input_dim * b);
        out_b.copy_to_host(h_out.data(), h_out.size() * sizeof(std::uint16_t));
        for (std::size_t i = 0; i < h_out.size(); ++i) {
            if (h_out[i] != 0x3F00U) {
                std::cerr << "Mismatch at decode B=" << b << ", idx=" << i << ": 0x"
                          << std::hex << h_out[i] << std::dec << "\n";
                return 1;
            }
        }
    }
    std::cout << "PASS: Decode B=1..8 bit-exact verification passed\n";

    // FP8 KV Cache Verification
    {
        ninfer::DeviceBuffer fp8_key_pages(256ULL * 64 * 2 * physical_pages);
        ninfer::DeviceBuffer fp8_value_pages(256ULL * 64 * 2 * physical_pages);
        fp8_key_pages.fill(0);
        fp8_value_pages.fill(0);

        QsaAttentionCacheView fp8_cache{
            .key_pages = ninfer::Tensor(fp8_key_pages.p, ninfer::DType::FP8_E4M3FN,
                                        {256, 64, 2, physical_pages}),
            .value_pages = ninfer::Tensor(fp8_value_pages.p, ninfer::DType::FP8_E4M3FN,
                                          {256, 64, 2, physical_pages}),
            .block_tables = ninfer::Tensor(block_tables.p, ninfer::DType::I32, {logical_pages, 1}),
        };

        // Decode B=1..8 with FP8 cache
        for (std::int32_t b = 1; b <= 8; ++b) {
            ninfer::DeviceBuffer in_b(input_dim * b * sizeof(std::uint16_t));
            ninfer::DeviceBuffer out_b(input_dim * b * sizeof(std::uint16_t));
            ninfer::DeviceBuffer tok_b(b * sizeof(std::int32_t));
            ninfer::DeviceBuffer pos_b(b * 3 * sizeof(std::int32_t));
            ninfer::DeviceBuffer row_b(b * sizeof(std::int32_t));
            ninfer::DeviceBuffer sel_b(512 * b * sizeof(std::int32_t));
            ninfer::DeviceBuffer cnt_b(b * sizeof(std::int32_t));

            std::vector<std::uint16_t> h_in(input_dim * b, 0);
            for (int i = 0; i < b; ++i) { h_in[i * input_dim] = 0x3F80U; }
            in_b.copy_from_host(h_in.data(), h_in.size() * sizeof(std::uint16_t));
            out_b.fill(0);
            tok_b.fill(0);
            pos_b.fill(0);
            row_b.fill(0);
            sel_b.fill(0);
            cnt_b.fill(0);

            ninfer::Tensor in_t(in_b.p, ninfer::DType::BF16, {input_dim, b});
            ninfer::Tensor out_t(out_b.p, ninfer::DType::BF16, {input_dim, b});
            ninfer::Tensor tok_t(tok_b.p, ninfer::DType::I32, {b});
            ninfer::Tensor pos_t(pos_b.p, ninfer::DType::I32, {b, 3});
            ninfer::Tensor row_t(row_b.p, ninfer::DType::I32, {b});
            ninfer::Tensor sel_t(sel_b.p, ninfer::DType::I32, {512, b});
            ninfer::Tensor cnt_t(cnt_b.p, ninfer::DType::I32, {b});

            ninfer::WorkspaceArena ws(flash_next_qsa_attention_workspace_capacity_bytes(b));
            flash_next_qsa_attention_decode(in_t, weights, tok_t, pos_t, row_t, sel_t, cnt_t,
                                            fp8_cache, ws, out_t, device.stream);
            device.synchronize();

            std::vector<std::uint16_t> h_out(input_dim * b);
            out_b.copy_to_host(h_out.data(), h_out.size() * sizeof(std::uint16_t));
            for (std::size_t i = 0; i < h_out.size(); ++i) {
                if (h_out[i] != 0x3F00U) {
                    std::cerr << "Mismatch at FP8 decode B=" << b << ", idx=" << i << ": 0x"
                              << std::hex << h_out[i] << std::dec << "\n";
                    return 1;
                }
            }
        }

        // Verify appended FP8 key and value: 0x38 is 1.0 in FP8 E4M3
        std::vector<std::uint8_t> h_fp8_keys(256ULL * 64 * 2 * physical_pages);
        std::vector<std::uint8_t> h_fp8_vals(256ULL * 64 * 2 * physical_pages);
        fp8_key_pages.copy_to_host(h_fp8_keys.data(), h_fp8_keys.size());
        fp8_value_pages.copy_to_host(h_fp8_vals.data(), h_fp8_vals.size());
        for (int kv_head = 0; kv_head < 2; ++kv_head) {
            for (int dim = 0; dim < 256; ++dim) {
                const std::size_t idx = ((0 * 2 + kv_head) * 64 + 0) * 256 + dim;
                if (h_fp8_keys[idx] != 0x38U) {
                    std::cerr << "FP8 Key cache mismatch at head " << kv_head << ", dim " << dim
                              << ": expected 0x38 (1.0), got 0x" << std::hex
                              << static_cast<int>(h_fp8_keys[idx]) << std::dec << "\n";
                    return 1;
                }
                if (h_fp8_vals[idx] != 0x38U) {
                    std::cerr << "FP8 Value cache mismatch at head " << kv_head << ", dim " << dim
                              << ": expected 0x38 (1.0), got 0x" << std::hex
                              << static_cast<int>(h_fp8_vals[idx]) << std::dec << "\n";
                    return 1;
                }
            }
        }
        std::cout << "PASS: FP8 Decode B=1..8 and cache append verification passed\n";

        // FP8 Prefill Chunk test: MMA vs non-MMA
        for (std::int32_t T : {16, 64, 128}) {
            ninfer::DeviceBuffer in_t(input_dim * T * sizeof(std::uint16_t));
            ninfer::DeviceBuffer out_mma(input_dim * T * sizeof(std::uint16_t));
            ninfer::DeviceBuffer out_non_mma(input_dim * T * sizeof(std::uint16_t));
            ninfer::DeviceBuffer tok_t(T * sizeof(std::int32_t));
            ninfer::DeviceBuffer pos_t(T * 3 * sizeof(std::int32_t));
            ninfer::DeviceBuffer sel_t(512 * T * sizeof(std::int32_t));
            ninfer::DeviceBuffer cnt_t(T * sizeof(std::int32_t));
            in_t.fill(0);
            out_mma.fill(0);
            out_non_mma.fill(0);
            tok_t.fill(0);
            pos_t.fill(0);
            sel_t.fill(0);
            cnt_t.fill(0);

            std::vector<std::uint16_t> h_in(input_dim * T, 0);
            for (int i = 0; i < T; ++i) { h_in[i * input_dim] = 0x3F80U; }
            in_t.copy_from_host(h_in.data(), h_in.size() * sizeof(std::uint16_t));

            ninfer::Tensor tin(in_t.p, ninfer::DType::BF16, {input_dim, T});
            ninfer::Tensor tout_mma(out_mma.p, ninfer::DType::BF16, {input_dim, T});
            ninfer::Tensor tout_non(out_non_mma.p, ninfer::DType::BF16, {input_dim, T});
            ninfer::Tensor ttok(tok_t.p, ninfer::DType::I32, {T});
            ninfer::Tensor tpos(pos_t.p, ninfer::DType::I32, {T, 3});
            ninfer::Tensor tsel(sel_t.p, ninfer::DType::I32, {512, T});
            ninfer::Tensor tcnt(cnt_t.p, ninfer::DType::I32, {T});

            ninfer::WorkspaceArena ws_mma(flash_next_qsa_attention_workspace_capacity_bytes(T));
            ninfer::WorkspaceArena ws_non(flash_next_qsa_attention_workspace_capacity_bytes(T));

            flash_next_qsa_attention_prefill_chunk(tin, weights, ttok, tpos, 0, tsel, tcnt,
                                                   fp8_cache, ws_mma, tout_mma, device.stream,
                                                   nullptr, true);
            flash_next_qsa_attention_prefill_chunk(tin, weights, ttok, tpos, 0, tsel, tcnt,
                                                   fp8_cache, ws_non, tout_non, device.stream,
                                                   nullptr, false);
            device.synchronize();

            std::vector<std::uint16_t> h_mma(input_dim * T);
            std::vector<std::uint16_t> h_non(input_dim * T);
            out_mma.copy_to_host(h_mma.data(), h_mma.size() * sizeof(std::uint16_t));
            out_non_mma.copy_to_host(h_non.data(), h_non.size() * sizeof(std::uint16_t));

            for (std::size_t i = 0; i < h_mma.size(); ++i) {
                float diff = std::abs(bf16_to_float(h_mma[i]) - bf16_to_float(h_non[i]));
                if (diff > 0.05f) {
                    std::cerr << "FP8 Prefill mismatch T=" << T << " idx=" << i
                              << " mma=" << bf16_to_float(h_mma[i])
                              << " non=" << bf16_to_float(h_non[i]) << "\n";
                    return 1;
                }
            }
        }
        std::cout << "PASS: FP8 Prefill Chunk MMA vs Non-MMA verification passed\n";
    }

    // Timing measurements
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::printf("\n=== Flash-Next QSA Attention Prefill & Decode Timing ===\n");
    for (std::int32_t T : {128, 512}) {
        ninfer::DeviceBuffer in_t(input_dim * T * sizeof(std::uint16_t));
        ninfer::DeviceBuffer out_t(input_dim * T * sizeof(std::uint16_t));
        ninfer::DeviceBuffer tok_t(T * sizeof(std::int32_t));
        ninfer::DeviceBuffer pos_t(T * 3 * sizeof(std::int32_t));
        ninfer::DeviceBuffer sel_t(512 * T * sizeof(std::int32_t));
        ninfer::DeviceBuffer cnt_t(T * sizeof(std::int32_t));
        in_t.fill(0);
        out_t.fill(0);
        tok_t.fill(0);
        pos_t.fill(0);
        sel_t.fill(0);
        cnt_t.fill(0);

        ninfer::Tensor tin(in_t.p, ninfer::DType::BF16, {input_dim, T});
        ninfer::Tensor tout(out_t.p, ninfer::DType::BF16, {input_dim, T});
        ninfer::Tensor ttok(tok_t.p, ninfer::DType::I32, {T});
        ninfer::Tensor tpos(pos_t.p, ninfer::DType::I32, {T, 3});
        ninfer::Tensor tsel(sel_t.p, ninfer::DType::I32, {512, T});
        ninfer::Tensor tcnt(cnt_t.p, ninfer::DType::I32, {T});

        ninfer::WorkspaceArena ws(flash_next_qsa_attention_workspace_capacity_bytes(T));

        // Warmup
        flash_next_qsa_attention_prefill_chunk(tin, weights, ttok, tpos, 0, tsel, tcnt, cache, ws,
                                               tout, device.stream);
        device.synchronize();

        constexpr int kIters = 50;
        CUDA_CHECK(cudaEventRecord(start, device.stream));
        for (int i = 0; i < kIters; ++i) {
            flash_next_qsa_attention_prefill_chunk(tin, weights, ttok, tpos, 0, tsel, tcnt, cache, ws,
                                                   tout, device.stream);
        }
        CUDA_CHECK(cudaEventRecord(stop, device.stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        float us = (ms / kIters) * 1000.0f;
        std::printf("  Prefill T=%3d: %.2f us / chunk-layer (%.3f ms across 12 layers)\n",
                    T, us, (us * 12.0f) / 1000.0f);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    std::cout << "PASS: test_qsa_attention\n";
    return 0;
}
