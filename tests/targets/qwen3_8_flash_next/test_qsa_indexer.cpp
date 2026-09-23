#include "core/arena.h"
#include "core/device.h"
#include "core/layout.h"
#include "ninfer/ops/linear.h"
#include "targets/qwen3_8_flash_next/impl/qsa_indexer.h"
#include "targets/qwen3_8_flash_next/impl/qsa_indexer_kernels.h"
#include "targets/qwen3_8_flash_next/impl/qsa_indexer_workspace.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

using ninfer::targets::qwen3_8_flash_next::detail::AttentionWeights;
using ninfer::targets::qwen3_8_flash_next::detail::QsaIndexerCacheView;
using ninfer::targets::qwen3_8_flash_next::detail::flash_next_qsa_indexer_decode;
using ninfer::targets::qwen3_8_flash_next::detail::flash_next_qsa_indexer_prefill_chunk;
using ninfer::targets::qwen3_8_flash_next::detail::flash_next_qsa_indexer_workspace_capacity_bytes;
using ninfer::targets::qwen3_8_flash_next::detail::flash_next_qsa_indexer_prefill_launch;
using ninfer::targets::qwen3_8_flash_next::detail::flash_next_qsa_indexer_sort_temp_bytes;
using ninfer::targets::qwen3_8_flash_next::detail::flash_next_qsa_indexer_tile_size;
using ninfer::targets::qwen3_8_flash_next::detail::allocate_flash_next_qsa_indexer_workspace;

float bf16_bits_to_float(std::uint16_t bits) {
    __nv_bfloat16 raw{};
    std::memcpy(&raw, &bits, sizeof(bits));
    return __bfloat162float(raw);
}

std::uint16_t float_to_bf16(float value) {
    const __nv_bfloat16 raw = __float2bfloat16_rn(value);
    return *reinterpret_cast<const std::uint16_t*>(&raw);
}

void drain_all_streams() { CUDA_CHECK(cudaDeviceSynchronize()); }

bool cuda_unavailable(cudaError_t error) {
    return error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver;
}

ninfer::Weight bf16_weight(void* data, std::int32_t rows, std::int32_t columns) {
    ninfer::Weight out{};
    out.payload         = data;
    out.payload_bytes   = static_cast<std::uint64_t>(rows) * columns * 2;
    out.qdata           = data;
    out.qtype           = ninfer::QType::BF16_CTRL;
    out.layout          = ninfer::QuantLayout::Contiguous;
    out.n               = rows;
    out.k               = columns;
    out.ndim            = 2;
    out.shape[0]        = rows;
    out.shape[1]        = columns;
    out.padded_shape[0] = rows;
    out.padded_shape[1] = columns;
    return out;
}

bool test_block_publish_equivalence(ninfer::DeviceContext& device) {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    constexpr std::int32_t maximum_blocks = 256;
    constexpr std::int32_t logical_pages  = 4;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<std::uint16_t> host_proj(640ULL * 2560);
    for (auto& v : host_proj) { v = float_to_bf16(dist(rng)); }
    std::vector<std::uint16_t> host_qnorm(128), host_knorm(128);
    for (auto& v : host_qnorm) v = float_to_bf16(dist(rng) * 0.1f);
    for (auto& v : host_knorm) v = float_to_bf16(dist(rng) * 0.1f);

    ninfer::DeviceBuffer proj_buf(640ULL * 2560 * 2);
    ninfer::DeviceBuffer qnorm_buf(128 * 2);
    ninfer::DeviceBuffer knorm_buf(128 * 2);
    proj_buf.copy_from_host(host_proj.data(), host_proj.size() * 2);
    qnorm_buf.copy_from_host(host_qnorm.data(), host_qnorm.size() * 2);
    knorm_buf.copy_from_host(host_knorm.data(), host_knorm.size() * 2);

    AttentionWeights weights{};
    weights.indexer_query_key  = bf16_weight(proj_buf.p, 640, 2560);
    weights.indexer_query_norm = ninfer::Tensor(qnorm_buf.p, ninfer::DType::BF16, {128});
    weights.indexer_key_norm   = ninfer::Tensor(knorm_buf.p, ninfer::DType::BF16, {128});

    const std::vector<std::int32_t> test_t = {1, 2, 3, 4, 5, 7, 8, 16, 32, 64};
    for (std::int32_t T : test_t) {
        for (std::int32_t leftover_in = 0; leftover_in < 4; ++leftover_in) {
            const std::int32_t first_token_index = 100 + leftover_in;

            std::vector<std::uint16_t> host_input(static_cast<std::size_t>(2560) * T);
            for (auto& v : host_input) v = float_to_bf16(dist(rng));

            std::vector<std::int32_t> host_pos(3 * T);
            for (std::int32_t t = 0; t < T; ++t) {
                host_pos[0 * T + t] = first_token_index + t;
                host_pos[1 * T + t] = first_token_index + t;
                host_pos[2 * T + t] = first_token_index + t;
            }

            std::vector<std::uint16_t> init_raw_keys(128ULL * 4 * 2, 0);
            std::vector<std::int32_t> init_raw_pos(3ULL * 4 * 2, 0);
            for (std::int32_t s = 0; s < leftover_in; ++s) {
                for (int d = 0; d < 128; ++d) {
                    init_raw_keys[s * 128 + d] = float_to_bf16(dist(rng));
                }
                for (int d = 0; d < 3; ++d) {
                    init_raw_pos[s * 3 + d] = first_token_index - leftover_in + s;
                }
            }

            ninfer::DeviceBuffer block_keys_a(128ULL * 64 * logical_pages * 2);
            ninfer::DeviceBuffer block_tables_a(logical_pages * sizeof(std::int32_t));
            ninfer::DeviceBuffer raw_keys_a(128ULL * 4 * 2 * 2);
            ninfer::DeviceBuffer raw_positions_a(3ULL * 4 * 2 * sizeof(std::int32_t));
            block_keys_a.fill(0);
            raw_keys_a.copy_from_host(init_raw_keys.data(), init_raw_keys.size() * 2);
            raw_positions_a.copy_from_host(init_raw_pos.data(), init_raw_pos.size() * sizeof(std::int32_t));
            std::array<std::int32_t, logical_pages> page_ids{};
            for (std::int32_t p = 0; p < logical_pages; ++p) page_ids[p] = p;
            block_tables_a.copy_from_host(page_ids.data(), sizeof(page_ids));
            drain_all_streams();

            QsaIndexerCacheView cache_a{
                .block_keys   = ninfer::Tensor(block_keys_a.p, ninfer::DType::BF16, {128, 64, logical_pages}),
                .block_tables = ninfer::Tensor(block_tables_a.p, ninfer::DType::I32, {logical_pages, 1}),
                .raw_keys     = ninfer::Tensor(raw_keys_a.p, ninfer::DType::BF16, {128, 4, 2}),
                .raw_positions = ninfer::Tensor(raw_positions_a.p, ninfer::DType::I32, {3, 4, 2}),
            };

            ninfer::DeviceBuffer input_dev(static_cast<std::size_t>(2560) * T * 2);
            input_dev.copy_from_host(host_input.data(), host_input.size() * 2);

            ninfer::DeviceBuffer token_buf(sizeof(std::int32_t));
            ninfer::DeviceBuffer pos_buf(3 * sizeof(std::int32_t));
            ninfer::DeviceBuffer row_buf(sizeof(std::int32_t));
            ninfer::DeviceBuffer src_buf(sizeof(std::int32_t));
            ninfer::DeviceBuffer dst_buf(sizeof(std::int32_t));
            ninfer::DeviceBuffer sel_buf(512 * sizeof(std::int32_t));
            ninfer::DeviceBuffer cnt_buf(sizeof(std::int32_t));
            std::int32_t zero = 0;
            row_buf.copy_from_host(&zero, sizeof(zero));

            ninfer::WorkspaceArena ws_decode(flash_next_qsa_indexer_workspace_capacity_bytes(maximum_blocks, 1));

            for (std::int32_t t = 0; t < T; ++t) {
                // DeviceBuffer::copy_from_host is a legacy-stream cudaMemcpy: it does not order against
                // the non-blocking device.stream, so drain the previous iteration first.
                drain_all_streams();
                std::int32_t tok_idx = first_token_index + t;
                std::array<std::int32_t, 3> pos_t = {host_pos[0 * T + t], host_pos[1 * T + t], host_pos[2 * T + t]};
                token_buf.copy_from_host(&tok_idx, sizeof(tok_idx));
                pos_buf.copy_from_host(pos_t.data(), sizeof(pos_t));
                std::int32_t src = t % 2;
                std::int32_t dst = 1 - src;
                src_buf.copy_from_host(&src, sizeof(src));
                dst_buf.copy_from_host(&dst, sizeof(dst));
                drain_all_streams();

                ninfer::Tensor in_slice(static_cast<std::uint16_t*>(input_dev.p) + static_cast<std::size_t>(t) * 2560,
                                       ninfer::DType::BF16, {2560, 1});
                ninfer::Tensor tok_t(token_buf.p, ninfer::DType::I32, {1});
                ninfer::Tensor pos_t_view(pos_buf.p, ninfer::DType::I32, {1, 3});
                ninfer::Tensor row_view(row_buf.p, ninfer::DType::I32, {1});
                ninfer::Tensor src_view(src_buf.p, ninfer::DType::I32, {1});
                ninfer::Tensor dst_view(dst_buf.p, ninfer::DType::I32, {1});
                ninfer::Tensor sel_view(sel_buf.p, ninfer::DType::I32, {512, 1});
                ninfer::Tensor cnt_view(cnt_buf.p, ninfer::DType::I32, {1});

                flash_next_qsa_indexer_decode(in_slice, weights, tok_t, pos_t_view, row_view,
                                              src_view, dst_view, cache_a, maximum_blocks, maximum_blocks,
                                              ws_decode, sel_view, cnt_view, device.stream);
            }
            drain_all_streams();

            ninfer::DeviceBuffer block_keys_b(128ULL * 64 * logical_pages * 2);
            ninfer::DeviceBuffer block_tables_b(logical_pages * sizeof(std::int32_t));
            ninfer::DeviceBuffer raw_keys_b(128ULL * 4 * 2 * 2);
            ninfer::DeviceBuffer raw_positions_b(3ULL * 4 * 2 * sizeof(std::int32_t));
            block_keys_b.fill(0);
            raw_keys_b.copy_from_host(init_raw_keys.data(), init_raw_keys.size() * 2);
            raw_positions_b.copy_from_host(init_raw_pos.data(), init_raw_pos.size() * sizeof(std::int32_t));
            block_tables_b.copy_from_host(page_ids.data(), sizeof(page_ids));
            drain_all_streams();

            QsaIndexerCacheView cache_b{
                .block_keys   = ninfer::Tensor(block_keys_b.p, ninfer::DType::BF16, {128, 64, logical_pages}),
                .block_tables = ninfer::Tensor(block_tables_b.p, ninfer::DType::I32, {logical_pages, 1}),
                .raw_keys     = ninfer::Tensor(raw_keys_b.p, ninfer::DType::BF16, {128, 4, 2}),
                .raw_positions = ninfer::Tensor(raw_positions_b.p, ninfer::DType::I32, {3, 4, 2}),
            };

            ninfer::DeviceBuffer dev_indices(T * sizeof(std::int32_t));
            std::vector<std::int32_t> host_indices(T);
            for (std::int32_t t = 0; t < T; ++t) host_indices[t] = first_token_index + t;
            dev_indices.copy_from_host(host_indices.data(), T * sizeof(std::int32_t));

            ninfer::DeviceBuffer dev_mrope_pos(3 * T * sizeof(std::int32_t));
            dev_mrope_pos.copy_from_host(host_pos.data(), host_pos.size() * sizeof(std::int32_t));

            ninfer::DeviceBuffer chunk_sel_buf(512 * T * sizeof(std::int32_t));
            ninfer::DeviceBuffer chunk_cnt_buf(T * sizeof(std::int32_t));

            ninfer::Tensor chunk_in(input_dev.p, ninfer::DType::BF16, {2560, T});
            ninfer::Tensor chunk_idx(dev_indices.p, ninfer::DType::I32, {T});
            ninfer::Tensor chunk_pos(dev_mrope_pos.p, ninfer::DType::I32, {T, 3});
            ninfer::Tensor chunk_sel(chunk_sel_buf.p, ninfer::DType::I32, {512, T});
            ninfer::Tensor chunk_cnt(chunk_cnt_buf.p, ninfer::DType::I32, {T});

            ninfer::WorkspaceArena ws_prefill(100ULL * 1024 * 1024);
            drain_all_streams();

            flash_next_qsa_indexer_prefill_chunk(
                chunk_in, weights, chunk_idx, chunk_pos, 0, 0, 1, cache_b, maximum_blocks,
                first_token_index, ws_prefill, chunk_sel, chunk_cnt, device.stream);
            drain_all_streams();

            std::vector<std::uint16_t> keys_a(128ULL * 64 * logical_pages);
            std::vector<std::uint16_t> keys_b(128ULL * 64 * logical_pages);
            block_keys_a.copy_to_host(keys_a.data(), keys_a.size() * 2);
            block_keys_b.copy_to_host(keys_b.data(), keys_b.size() * 2);

            const std::int32_t complete_out = (leftover_in + T) / 4;
            double key_diff_sq              = 0.0;
            double key_base_sq              = 0.0;
            int key_nonfinite               = 0;
            for (std::size_t i = 0; i < keys_a.size(); ++i) {
                float fa = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&keys_a[i]));
                float fb = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&keys_b[i]));
                if (!std::isfinite(fa) || !std::isfinite(fb)) { ++key_nonfinite; }
                key_base_sq += static_cast<double>(fa) * static_cast<double>(fa);
                const double delta = static_cast<double>(fa) - static_cast<double>(fb);
                key_diff_sq += delta * delta;
                const float key_tol = 1e-2f * std::max(1.0f, std::max(std::abs(fa), std::abs(fb)));
                if (std::abs(fa - fb) > key_tol) {
                    std::cerr << "Block keys mismatch at T=" << T << ", leftover_in=" << leftover_in
                              << ", idx " << i << ": seq=" << fa << " chunk=" << fb << "\n";
                    return false;
                }
            }
            if (complete_out > 0 && (key_nonfinite > 0 || key_base_sq <= 0.0)) {
                const std::int32_t first_block = (first_token_index - leftover_in) / 4;
                const std::size_t block_off =
                    static_cast<std::size_t>(first_block % 64) * 128ULL +
                    static_cast<std::size_t>(first_block / 64) * 64ULL * 128ULL;
                std::cerr << "FAIL: block-key comparison vacuous or non-finite at T=" << T
                          << " leftover_in=" << leftover_in << " base_sq=" << key_base_sq
                          << " nonfinite=" << key_nonfinite << " first_block=" << first_block
                          << " bits:";
                for (int i = 0; i < 8; ++i) {
                    std::cerr << " 0x" << std::hex << keys_a[block_off + static_cast<std::size_t>(i)]
                              << std::dec;
                }
                std::cerr << "\n";
                return false;
            }

            const std::int32_t final_slot_a = (T % 2 == 1) ? 1 : 0;
            const std::int32_t leftover_out = (leftover_in + T) & 3;
            std::vector<std::uint16_t> rk_a(128ULL * 4 * 2);
            std::vector<std::uint16_t> rk_b(128ULL * 4 * 2);
            raw_keys_a.copy_to_host(rk_a.data(), rk_a.size() * 2);
            raw_keys_b.copy_to_host(rk_b.data(), rk_b.size() * 2);
            double raw_diff_sq = 0.0;
            double raw_base_sq = 0.0;
            int raw_nonfinite  = 0;
            for (std::int32_t s = 0; s < leftover_out; ++s) {
                for (int d = 0; d < 128; ++d) {
                    const std::size_t idx_a = final_slot_a * 128 * 4 + s * 128 + d;
                    const std::size_t idx_b = 1 * 128 * 4 + s * 128 + d;
                    const float fa =
                        __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&rk_a[idx_a]));
                    const float fb =
                        __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&rk_b[idx_b]));
                    if (!std::isfinite(fa) || !std::isfinite(fb)) { ++raw_nonfinite; }
                    raw_base_sq += static_cast<double>(fa) * static_cast<double>(fa);
                    const double delta = static_cast<double>(fa) - static_cast<double>(fb);
                    raw_diff_sq += delta * delta;
                    const float raw_tol = 1e-2f * std::max(1.0f, std::max(std::abs(fa), std::abs(fb)));
                    if (std::abs(fa - fb) > raw_tol) {
                        std::cerr << "Raw keys mismatch at T=" << T
                                  << ", leftover_in=" << leftover_in << ", slot=" << s
                                  << ", d=" << d << ": seq=" << fa << " chunk=" << fb << "\n";
                        return false;
                    }
                }
            }
            if (leftover_out > 0 && (raw_nonfinite > 0 || raw_base_sq <= 0.0)) {
                std::cerr << "FAIL: leftover raw-key comparison vacuous or non-finite at T=" << T
                          << " leftover_in=" << leftover_in << " base_sq=" << raw_base_sq
                          << " nonfinite=" << raw_nonfinite << "\n";
                return false;
            }

            std::vector<std::int32_t> rp_a(3ULL * 4 * 2);
            std::vector<std::int32_t> rp_b(3ULL * 4 * 2);
            raw_positions_a.copy_to_host(rp_a.data(), rp_a.size() * sizeof(std::int32_t));
            raw_positions_b.copy_to_host(rp_b.data(), rp_b.size() * sizeof(std::int32_t));
            for (std::int32_t s = 0; s < leftover_out; ++s) {
                for (int d = 0; d < 3; ++d) {
                    const std::size_t idx_a = final_slot_a * 3 * 4 + s * 3 + d;
                    const std::size_t idx_b = 1 * 3 * 4 + s * 3 + d;
                    if (rp_a[idx_a] != rp_b[idx_b]) {
                        std::cerr << "Raw positions mismatch at T=" << T
                                  << ", leftover_in=" << leftover_in << ", slot=" << s
                                  << ", d=" << d << ": seq=" << rp_a[idx_a] << " chunk=" << rp_b[idx_b]
                                  << "\n";
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

bool test_selection_equivalence(ninfer::DeviceContext& device) {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    constexpr std::int32_t maximum_blocks = 256;
    constexpr std::int32_t logical_pages  = 4;

    std::mt19937 rng(1337);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<std::uint16_t> host_proj(640ULL * 2560);
    for (auto& v : host_proj) v = float_to_bf16(dist(rng));
    std::vector<std::uint16_t> host_qnorm(128), host_knorm(128);
    for (auto& v : host_qnorm) v = float_to_bf16(dist(rng) * 0.1f);
    for (auto& v : host_knorm) v = float_to_bf16(dist(rng) * 0.1f);

    ninfer::DeviceBuffer proj_buf(640ULL * 2560 * 2);
    ninfer::DeviceBuffer qnorm_buf(128 * 2);
    ninfer::DeviceBuffer knorm_buf(128 * 2);
    proj_buf.copy_from_host(host_proj.data(), host_proj.size() * 2);
    qnorm_buf.copy_from_host(host_qnorm.data(), host_qnorm.size() * 2);
    knorm_buf.copy_from_host(host_knorm.data(), host_knorm.size() * 2);

    AttentionWeights weights{};
    weights.indexer_query_key  = bf16_weight(proj_buf.p, 640, 2560);
    weights.indexer_query_norm = ninfer::Tensor(qnorm_buf.p, ninfer::DType::BF16, {128});
    weights.indexer_key_norm   = ninfer::Tensor(knorm_buf.p, ninfer::DType::BF16, {128});

    const std::vector<std::int32_t> test_t = {1, 4, 8, 32, 128, 256};
    for (std::int32_t T : test_t) {
        const std::int32_t first_token_index = 40;

        std::vector<std::uint16_t> host_input(static_cast<std::size_t>(2560) * T);
        for (auto& v : host_input) v = float_to_bf16(dist(rng));

        std::vector<std::int32_t> host_pos(3 * T);
        for (std::int32_t t = 0; t < T; ++t) {
            host_pos[0 * T + t] = first_token_index + t;
            host_pos[1 * T + t] = first_token_index + t;
            host_pos[2 * T + t] = first_token_index + t;
        }

        std::vector<std::uint16_t> host_bkeys(128ULL * 64 * logical_pages);
        for (auto& v : host_bkeys) v = float_to_bf16(dist(rng));

        ninfer::DeviceBuffer block_keys_a(128ULL * 64 * logical_pages * 2);
        ninfer::DeviceBuffer block_keys_b(128ULL * 64 * logical_pages * 2);
        block_keys_a.copy_from_host(host_bkeys.data(), host_bkeys.size() * 2);
        block_keys_b.copy_from_host(host_bkeys.data(), host_bkeys.size() * 2);

        ninfer::DeviceBuffer block_tables(logical_pages * sizeof(std::int32_t));
        std::array<std::int32_t, logical_pages> page_ids{};
        for (std::int32_t p = 0; p < logical_pages; ++p) page_ids[p] = p;
        block_tables.copy_from_host(page_ids.data(), sizeof(page_ids));

        ninfer::DeviceBuffer raw_keys_a(128ULL * 4 * 2 * 2), raw_keys_b(128ULL * 4 * 2 * 2);
        ninfer::DeviceBuffer raw_pos_a(3ULL * 4 * 2 * sizeof(std::int32_t)), raw_pos_b(3ULL * 4 * 2 * sizeof(std::int32_t));
        raw_keys_a.fill(0); raw_keys_b.fill(0);
        raw_pos_a.fill(0); raw_pos_b.fill(0);

        QsaIndexerCacheView cache_a{
            .block_keys   = ninfer::Tensor(block_keys_a.p, ninfer::DType::BF16, {128, 64, logical_pages}),
            .block_tables = ninfer::Tensor(block_tables.p, ninfer::DType::I32, {logical_pages, 1}),
            .raw_keys     = ninfer::Tensor(raw_keys_a.p, ninfer::DType::BF16, {128, 4, 2}),
            .raw_positions = ninfer::Tensor(raw_pos_a.p, ninfer::DType::I32, {3, 4, 2}),
        };
        QsaIndexerCacheView cache_b{
            .block_keys   = ninfer::Tensor(block_keys_b.p, ninfer::DType::BF16, {128, 64, logical_pages}),
            .block_tables = ninfer::Tensor(block_tables.p, ninfer::DType::I32, {logical_pages, 1}),
            .raw_keys     = ninfer::Tensor(raw_keys_b.p, ninfer::DType::BF16, {128, 4, 2}),
            .raw_positions = ninfer::Tensor(raw_pos_b.p, ninfer::DType::I32, {3, 4, 2}),
        };

        ninfer::DeviceBuffer input_dev(static_cast<std::size_t>(2560) * T * 2);
        input_dev.copy_from_host(host_input.data(), host_input.size() * 2);

        std::vector<std::int32_t> seq_counts(T);
        std::vector<std::int32_t> seq_blocks(512 * T);

        ninfer::DeviceBuffer token_buf(sizeof(std::int32_t));
        ninfer::DeviceBuffer pos_buf(3 * sizeof(std::int32_t));
        ninfer::DeviceBuffer row_buf(sizeof(std::int32_t));
        ninfer::DeviceBuffer src_buf(sizeof(std::int32_t));
        ninfer::DeviceBuffer dst_buf(sizeof(std::int32_t));
        ninfer::DeviceBuffer sel_buf(512 * sizeof(std::int32_t));
        ninfer::DeviceBuffer cnt_buf(sizeof(std::int32_t));
        std::int32_t zero = 0;
        row_buf.copy_from_host(&zero, sizeof(zero));

        ninfer::WorkspaceArena ws_decode(flash_next_qsa_indexer_workspace_capacity_bytes(maximum_blocks, 1));

        for (std::int32_t t = 0; t < T; ++t) {
            std::int32_t tok_idx = first_token_index + t;
            std::array<std::int32_t, 3> pos_t = {host_pos[0 * T + t], host_pos[1 * T + t], host_pos[2 * T + t]};
            drain_all_streams();
            token_buf.copy_from_host(&tok_idx, sizeof(tok_idx));
            pos_buf.copy_from_host(pos_t.data(), sizeof(pos_t));
            std::int32_t src = t % 2;
            std::int32_t dst = 1 - src;
            src_buf.copy_from_host(&src, sizeof(src));
            dst_buf.copy_from_host(&dst, sizeof(dst));
            drain_all_streams();

            ninfer::Tensor in_slice(static_cast<std::uint16_t*>(input_dev.p) + static_cast<std::size_t>(t) * 2560,
                                   ninfer::DType::BF16, {2560, 1});
            ninfer::Tensor tok_t(token_buf.p, ninfer::DType::I32, {1});
            ninfer::Tensor pos_t_view(pos_buf.p, ninfer::DType::I32, {1, 3});
            ninfer::Tensor row_view(row_buf.p, ninfer::DType::I32, {1});
            ninfer::Tensor src_view(src_buf.p, ninfer::DType::I32, {1});
            ninfer::Tensor dst_view(dst_buf.p, ninfer::DType::I32, {1});
            ninfer::Tensor sel_view(sel_buf.p, ninfer::DType::I32, {512, 1});
            ninfer::Tensor cnt_view(cnt_buf.p, ninfer::DType::I32, {1});

            flash_next_qsa_indexer_decode(in_slice, weights, tok_t, pos_t_view, row_view,
                                          src_view, dst_view, cache_a, maximum_blocks, maximum_blocks,
                                          ws_decode, sel_view, cnt_view, device.stream);
            drain_all_streams();
            cnt_buf.copy_to_host(&seq_counts[t], sizeof(std::int32_t));
            sel_buf.copy_to_host(&seq_blocks[t * 512], 512 * sizeof(std::int32_t));
        }

        ninfer::DeviceBuffer dev_indices(T * sizeof(std::int32_t));
        std::vector<std::int32_t> host_indices(T);
        for (std::int32_t t = 0; t < T; ++t) host_indices[t] = first_token_index + t;
        dev_indices.copy_from_host(host_indices.data(), T * sizeof(std::int32_t));

        ninfer::DeviceBuffer dev_mrope_pos(3 * T * sizeof(std::int32_t));
        dev_mrope_pos.copy_from_host(host_pos.data(), host_pos.size() * sizeof(std::int32_t));

        ninfer::DeviceBuffer chunk_sel_buf(512 * T * sizeof(std::int32_t));
        ninfer::DeviceBuffer chunk_cnt_buf(T * sizeof(std::int32_t));

        ninfer::Tensor chunk_in(input_dev.p, ninfer::DType::BF16, {2560, T});
        ninfer::Tensor chunk_idx(dev_indices.p, ninfer::DType::I32, {T});
        ninfer::Tensor chunk_pos(dev_mrope_pos.p, ninfer::DType::I32, {T, 3});
        ninfer::Tensor chunk_sel(chunk_sel_buf.p, ninfer::DType::I32, {512, T});
        ninfer::Tensor chunk_cnt(chunk_cnt_buf.p, ninfer::DType::I32, {T});

        ninfer::WorkspaceArena ws_prefill(100ULL * 1024 * 1024);
        drain_all_streams();
        flash_next_qsa_indexer_prefill_chunk(
            chunk_in, weights, chunk_idx, chunk_pos, 0, 0, 1, cache_b, maximum_blocks,
            first_token_index, ws_prefill, chunk_sel, chunk_cnt, device.stream);
        drain_all_streams();

        std::vector<std::int32_t> chunk_counts(T);
        std::vector<std::int32_t> chunk_blocks(512 * T);
        chunk_cnt_buf.copy_to_host(chunk_counts.data(), T * sizeof(std::int32_t));
        chunk_sel_buf.copy_to_host(chunk_blocks.data(), 512 * T * sizeof(std::int32_t));

        for (std::int32_t t = 0; t < T; ++t) {
            if (seq_counts[t] != chunk_counts[t]) {
                std::cerr << "Selection count mismatch at T=" << T << ", t=" << t
                          << ": seq=" << seq_counts[t] << ", chunk=" << chunk_counts[t] << "\n";
                return false;
            }
            const std::int32_t count = seq_counts[t];
            const std::int32_t complete_blocks = (first_token_index + t + 1) / 4;
            if (count <= 0 || count != std::min(complete_blocks, 512)) {
                std::cerr << "FAIL: selection count vacuous at T=" << T << ", t=" << t
                          << " count=" << count << " complete_blocks=" << complete_blocks << "\n";
                return false;
            }
            std::vector<std::int32_t> seq_ids(static_cast<std::size_t>(count));
            for (std::int32_t k = 0; k < count; ++k) {
                if (seq_blocks[t * 512 + k] != chunk_blocks[t * 512 + k]) {
                    std::cerr << "Selected block mismatch at T=" << T << ", t=" << t << ", rank=" << k
                              << ": seq=" << seq_blocks[t * 512 + k] << ", chunk=" << chunk_blocks[t * 512 + k] << "\n";
                    return false;
                }
                seq_ids[static_cast<std::size_t>(k)] = seq_blocks[t * 512 + k];
                if (seq_ids[static_cast<std::size_t>(k)] < 0 ||
                    seq_ids[static_cast<std::size_t>(k)] >= complete_blocks) {
                    std::cerr << "FAIL: selected id out of range at T=" << T << " t=" << t
                              << " id=" << seq_ids[static_cast<std::size_t>(k)] << "\n";
                    return false;
                }
            }
            std::sort(seq_ids.begin(), seq_ids.end());
            if (std::adjacent_find(seq_ids.begin(), seq_ids.end()) != seq_ids.end()) {
                std::cerr << "FAIL: duplicate selected ids at T=" << T << " t=" << t << "\n";
                return false;
            }
            for (std::int32_t k = count; k < 512; ++k) {
                if (seq_blocks[t * 512 + k] != -1 || chunk_blocks[t * 512 + k] != -1) {
                    std::cerr << "FAIL: padding was not -1 at T=" << T << " t=" << t << " k=" << k
                              << "\n";
                    return false;
                }
            }
        }
    }
    return true;
}

bool selection_is_identity(const std::int32_t* ids, std::int32_t count) {
    if (count <= 0) { return false; }
    for (std::int32_t i = 0; i < count; ++i) {
        if (ids[i] != i) { return false; }
    }
    for (std::int32_t i = count; i < 512; ++i) {
        if (ids[i] != -1) { return false; }
    }
    return true;
}

bool selection_sets_equal(const std::int32_t* a, const std::int32_t* b, std::int32_t count,
                          std::int32_t complete_blocks) {
    if (count <= 0 || count > 512) { return false; }
    std::vector<std::int32_t> sa(a, a + count);
    std::vector<std::int32_t> sb(b, b + count);
    std::sort(sa.begin(), sa.end());
    std::sort(sb.begin(), sb.end());
    if (sa != sb) { return false; }
    if (std::adjacent_find(sa.begin(), sa.end()) != sa.end()) { return false; }
    for (std::int32_t id : sa) {
        if (id < 0 || id >= complete_blocks) { return false; }
    }
    for (std::int32_t i = count; i < 512; ++i) {
        if (a[i] != -1 || b[i] != -1) { return false; }
    }
    return true;
}

void fill_equal_block_keys(std::vector<std::uint16_t>& keys, std::int32_t complete_blocks) {
    std::fill(keys.begin(), keys.end(), 0);
    const std::uint16_t packed = float_to_bf16(1.0F);
    for (std::int32_t block = 0; block < complete_blocks; ++block) {
        const std::int32_t page   = block / 64;
        const std::int32_t offset = block % 64;
        const std::size_t index =
            static_cast<std::size_t>(page) * 64ULL * 128ULL + static_cast<std::size_t>(offset) * 128ULL;
        for (int dim = 0; dim < 128; ++dim) { keys[index + static_cast<std::size_t>(dim)] = packed; }
    }
}

void fill_tie_band_block_keys(std::vector<std::uint16_t>& keys, std::int32_t complete_blocks,
                              std::int32_t unique_high) {
    std::fill(keys.begin(), keys.end(), 0);
    const std::uint16_t medium = float_to_bf16(0.0F);
    for (std::int32_t block = 0; block < complete_blocks; ++block) {
        const std::int32_t page   = block / 64;
        const std::int32_t offset = block % 64;
        const std::size_t index =
            static_cast<std::size_t>(page) * 64ULL * 128ULL + static_cast<std::size_t>(offset) * 128ULL;
        const std::uint16_t packed =
            block < unique_high ? float_to_bf16(static_cast<float>(block + 1)) : medium;
        for (int dim = 0; dim < 128; ++dim) { keys[index + static_cast<std::size_t>(dim)] = packed; }
    }
}

void expected_tie_band_order(std::array<std::int32_t, 512>& expected, std::int32_t unique_high) {
    std::int32_t out = 0;
    for (std::int32_t block = unique_high - 1; block >= 0; --block) {
        expected[static_cast<std::size_t>(out++)] = block;
    }
    for (std::int32_t block = unique_high; out < 512; ++block) {
        expected[static_cast<std::size_t>(out++)] = block;
    }
}

bool selection_matches(const std::int32_t* actual, const std::int32_t* expected, std::int32_t count,
                       const char* label) {
    if (count != 512) {
        std::cerr << "FAIL: " << label << " count=" << count << " (vacuous or wrong)\n";
        return false;
    }
    for (std::int32_t i = 0; i < 512; ++i) {
        if (actual[i] != expected[i]) {
            std::cerr << "FAIL: " << label << " id[" << i << "]=" << actual[i]
                      << " expected=" << expected[i] << "\n";
            return false;
        }
    }
    return true;
}

void fill_monotonic_block_keys(std::vector<std::uint16_t>& keys, std::int32_t complete_blocks) {
    std::fill(keys.begin(), keys.end(), 0);
    for (std::int32_t block = 0; block < complete_blocks; ++block) {
        const std::int32_t page   = block / 64;
        const std::int32_t offset = block % 64;
        const std::size_t index =
            static_cast<std::size_t>(page) * 64ULL * 128ULL + static_cast<std::size_t>(offset) * 128ULL;
        const std::uint16_t packed = float_to_bf16(static_cast<float>(block + 1));
        for (int dim = 0; dim < 128; ++dim) { keys[index + static_cast<std::size_t>(dim)] = packed; }
    }
}

struct IndexerProbe {
    ninfer::DeviceBuffer input;
    ninfer::DeviceBuffer projection;
    ninfer::DeviceBuffer query_norm;
    ninfer::DeviceBuffer key_norm;
    ninfer::DeviceBuffer block_keys;
    ninfer::DeviceBuffer block_tables;
    ninfer::DeviceBuffer raw_keys;
    ninfer::DeviceBuffer raw_positions;
    ninfer::DeviceBuffer token_index;
    ninfer::DeviceBuffer mrope_positions;
    ninfer::DeviceBuffer table_row;
    ninfer::DeviceBuffer state_slot;
    ninfer::DeviceBuffer selected_blocks;
    ninfer::DeviceBuffer selected_count;
    ninfer::WorkspaceArena workspace;
    AttentionWeights weights{};
    QsaIndexerCacheView cache{};
    ninfer::Tensor input_view;
    ninfer::Tensor token_view;
    ninfer::Tensor position_view;
    ninfer::Tensor table_row_view;
    ninfer::Tensor state_slot_view;
    ninfer::Tensor selected_view;
    ninfer::Tensor count_view;

    IndexerProbe(std::int32_t maximum_blocks, std::int32_t logical_pages)
        : input(2'560ULL * 2),
          projection(640ULL * 2'560 * 2),
          query_norm(128 * 2),
          key_norm(128 * 2),
          block_keys(128ULL * 64 * static_cast<std::size_t>(logical_pages) * 2),
          block_tables(static_cast<std::size_t>(logical_pages) * sizeof(std::int32_t)),
          raw_keys(128ULL * 4 * 2),
          raw_positions(3ULL * 4 * sizeof(std::int32_t)),
          token_index(sizeof(std::int32_t)),
          mrope_positions(3 * sizeof(std::int32_t)),
          table_row(sizeof(std::int32_t)),
          state_slot(sizeof(std::int32_t)),
          selected_blocks(512 * sizeof(std::int32_t)),
          selected_count(sizeof(std::int32_t)),
          workspace(flash_next_qsa_indexer_workspace_capacity_bytes(maximum_blocks, 1)),
          input_view(input.p, ninfer::DType::BF16, {2'560, 1}),
          token_view(token_index.p, ninfer::DType::I32, {1}),
          position_view(mrope_positions.p, ninfer::DType::I32, {1, 3}),
          table_row_view(table_row.p, ninfer::DType::I32, {1}),
          state_slot_view(state_slot.p, ninfer::DType::I32, {1}),
          selected_view(selected_blocks.p, ninfer::DType::I32, {512, 1}),
          count_view(selected_count.p, ninfer::DType::I32, {1}) {
        std::vector<std::uint16_t> input_values(2'560, 0);
        input_values[0] = 0x3F80U;
        input.copy_from_host(input_values.data(), input_values.size() * sizeof(std::uint16_t));
        std::vector<std::uint16_t> projection_values(640ULL * 2'560, 0);
        for (std::int32_t row = 0; row < 640; ++row) {
            projection_values[static_cast<std::size_t>(row) * 2'560] = 0x3F80U;
        }
        projection.copy_from_host(projection_values.data(),
                                  projection_values.size() * sizeof(std::uint16_t));
        query_norm.fill(0);
        key_norm.fill(0);
        raw_keys.fill(0);
        raw_positions.fill(0);
        std::vector<std::int32_t> page_ids(static_cast<std::size_t>(logical_pages));
        for (std::int32_t page = 0; page < logical_pages; ++page) { page_ids[static_cast<std::size_t>(page)] = page; }
        block_tables.copy_from_host(page_ids.data(), page_ids.size() * sizeof(std::int32_t));
        constexpr std::array<std::int32_t, 3> zero_positions{};
        mrope_positions.copy_from_host(zero_positions.data(), sizeof(zero_positions));
        constexpr std::int32_t zero = 0;
        table_row.copy_from_host(&zero, sizeof(zero));
        state_slot.copy_from_host(&zero, sizeof(zero));
        weights.indexer_query_key  = bf16_weight(projection.p, 640, 2'560);
        weights.indexer_query_norm = ninfer::Tensor(query_norm.p, ninfer::DType::BF16, {128});
        weights.indexer_key_norm   = ninfer::Tensor(key_norm.p, ninfer::DType::BF16, {128});
        cache = {
            .block_keys    = ninfer::Tensor(block_keys.p, ninfer::DType::BF16, {128, 64, logical_pages}),
            .block_tables  = ninfer::Tensor(block_tables.p, ninfer::DType::I32, {logical_pages, 1}),
            .raw_keys      = ninfer::Tensor(raw_keys.p, ninfer::DType::BF16, {128, 4, 1}),
            .raw_positions = ninfer::Tensor(raw_positions.p, ninfer::DType::I32, {3, 4, 1}),
        };
        drain_all_streams();
    }

    void load_keys(const std::vector<std::uint16_t>& keys, ninfer::DeviceContext& device) {
        (void)device;
        block_keys.copy_from_host(keys.data(), keys.size() * 2);
        raw_keys.fill(0);
        raw_positions.fill(0);
        drain_all_streams();
    }

    void set_token(std::int32_t token) { token_index.copy_from_host(&token, sizeof(token)); }

    void launch(std::int32_t maximum_blocks, std::int32_t active_blocks, cudaStream_t stream) {
        flash_next_qsa_indexer_decode(input_view, weights, token_view, position_view, table_row_view,
                                      state_slot_view, state_slot_view, cache, maximum_blocks,
                                      active_blocks, workspace, selected_view, count_view, stream);
    }

    void run(ninfer::DeviceContext& device, std::int32_t token, std::int32_t maximum_blocks,
             std::int32_t active_blocks, std::int32_t& count, std::array<std::int32_t, 512>& ids) {
        drain_all_streams();
        set_token(token);
        drain_all_streams();
        launch(maximum_blocks, active_blocks, device.stream);
        drain_all_streams();
        selected_count.copy_to_host(&count, sizeof(count));
        selected_blocks.copy_to_host(ids.data(), sizeof(ids));
    }
};

bool test_fully_selected_identity_bypass(ninfer::DeviceContext& device, float& us_bypass,
                                         float& us_sort) {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    constexpr std::int32_t maximum_blocks = 513;
    constexpr std::int32_t logical_pages  = 9;
    IndexerProbe probe(maximum_blocks, logical_pages);
    IndexerProbe sort_probe(maximum_blocks, logical_pages);

    std::vector<std::uint16_t> host_keys(128ULL * 64 * logical_pages, 0);
    fill_monotonic_block_keys(host_keys, 512);

    // Exactly 512 complete blocks: token 2048, tail=0 so update_key does not rewrite a block key.
    constexpr std::int32_t token_512 = 2'048;
    constexpr std::int32_t complete_512 = (token_512 + 1) / 4;
    if (complete_512 != 512) {
        std::cerr << "FAIL: token 2048 must yield 512 complete blocks, got " << complete_512 << "\n";
        return false;
    }

    probe.load_keys(host_keys, device);
    sort_probe.load_keys(host_keys, device);
    std::int32_t bypass_count = -1;
    std::int32_t sort_count   = -1;
    std::array<std::int32_t, 512> bypass_ids{};
    std::array<std::int32_t, 512> sort_ids{};
    probe.run(device, token_512, maximum_blocks, 512, bypass_count, bypass_ids);
    sort_probe.run(device, token_512, maximum_blocks, 513, sort_count, sort_ids);

    if (bypass_count != 512 || !selection_is_identity(bypass_ids.data(), bypass_count)) {
        std::cerr << "FAIL: active_blocks=512 did not publish the identity selection (count="
                  << bypass_count << " front=" << bypass_ids.front() << ")\n";
        return false;
    }
    if (sort_count != 512) {
        std::cerr << "FAIL: sorted path at 512 complete blocks returned count=" << sort_count << "\n";
        return false;
    }
    if (!selection_sets_equal(bypass_ids.data(), sort_ids.data(), 512, 512)) {
        std::cerr << "FAIL: bypass vs sorted selection SET mismatch at exactly 512 complete blocks\n";
        return false;
    }
    bool sort_is_identity = selection_is_identity(sort_ids.data(), sort_count);
    if (sort_is_identity) {
        std::cerr << "FAIL: sorted path at 512 was identity; monotonic keys should permute rank order\n";
        return false;
    }

    // Exactly 511 complete blocks: still fully selected, identity bypass.
    constexpr std::int32_t token_511 = 2'043;
    constexpr std::int32_t complete_511 = (token_511 + 1) / 4;
    if (complete_511 != 511) {
        std::cerr << "FAIL: token 2043 must yield 511 complete blocks, got " << complete_511 << "\n";
        return false;
    }
    probe.load_keys(host_keys, device);
    std::int32_t count_511 = -1;
    std::array<std::int32_t, 512> ids_511{};
    probe.run(device, token_511, maximum_blocks, 511, count_511, ids_511);
    if (count_511 != 511 || !selection_is_identity(ids_511.data(), count_511)) {
        std::cerr << "FAIL: active_blocks=511 did not publish identity 0..510 (count=" << count_511
                  << ")\n";
        return false;
    }

    // Exactly 513 complete blocks: sort path, not identity.
    std::vector<std::uint16_t> keys_513 = host_keys;
    fill_monotonic_block_keys(keys_513, 513);
    sort_probe.load_keys(keys_513, device);
    constexpr std::int32_t token_513 = 2'052; // tail=0, (2052+1)/4 = 513
    constexpr std::int32_t complete_513 = (token_513 + 1) / 4;
    if (complete_513 != 513) {
        std::cerr << "FAIL: token 2052 must yield 513 complete blocks, got " << complete_513 << "\n";
        return false;
    }
    std::int32_t count_513 = -1;
    std::array<std::int32_t, 512> ids_513{};
    sort_probe.run(device, token_513, maximum_blocks, 513, count_513, ids_513);
    if (count_513 != 512) {
        std::cerr << "FAIL: 513 complete blocks should select 512, got " << count_513 << "\n";
        return false;
    }
    if (selection_is_identity(ids_513.data(), count_513)) {
        std::cerr << "FAIL: 513-block path published identity; top-k must drop one complete block\n";
        return false;
    }
    std::vector<std::int32_t> kept(ids_513.begin(), ids_513.begin() + 512);
    std::sort(kept.begin(), kept.end());
    if (std::adjacent_find(kept.begin(), kept.end()) != kept.end()) {
        std::cerr << "FAIL: 513-block top-k contained duplicates\n";
        return false;
    }
    for (std::int32_t id : kept) {
        if (id < 0 || id >= 513) {
            std::cerr << "FAIL: 513-block top-k id " << id << " out of range\n";
            return false;
        }
    }
    if (kept.size() != 512) {
        std::cerr << "FAIL: 513-block top-k size " << kept.size() << "\n";
        return false;
    }

    constexpr int kWarmup = 5;
    constexpr int kIters  = 50;
    probe.load_keys(host_keys, device);
    sort_probe.load_keys(host_keys, device);
    probe.set_token(token_512);
    sort_probe.set_token(token_512);
    drain_all_streams();
    for (int i = 0; i < kWarmup; ++i) {
        probe.launch(maximum_blocks, 512, device.stream);
        sort_probe.launch(maximum_blocks, 513, device.stream);
    }
    device.synchronize();

    cudaEvent_t start = nullptr;
    cudaEvent_t stop  = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, device.stream));
    for (int i = 0; i < kIters; ++i) {
        probe.launch(maximum_blocks, 512, device.stream);
    }
    CUDA_CHECK(cudaEventRecord(stop, device.stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float bypass_ms = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&bypass_ms, start, stop));

    CUDA_CHECK(cudaEventRecord(start, device.stream));
    for (int i = 0; i < kIters; ++i) {
        sort_probe.launch(maximum_blocks, 513, device.stream);
    }
    CUDA_CHECK(cudaEventRecord(stop, device.stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float sort_ms = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&sort_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    us_bypass = bypass_ms * 1000.0F / static_cast<float>(kIters);
    us_sort   = sort_ms * 1000.0F / static_cast<float>(kIters);
    return true;
}

bool test_long_context_topk(ninfer::DeviceContext& device) {
    constexpr std::array<std::int32_t, 5> ns{513, 1024, 4096, 8192, 16384};
    constexpr std::int32_t unique_high = 400;
    constexpr int kIters               = 30;
    std::cout << "\n=== G5 long-context indexer top-512 ===\n";
    for (std::int32_t n : ns) {
        const std::int32_t logical_pages = (n + 63) / 64;
        const std::int32_t token         = 4 * n; // tail=0, complete_blocks = n
        IndexerProbe probe(n, logical_pages);
        std::vector<std::uint16_t> keys(128ULL * 64 * static_cast<std::size_t>(logical_pages), 0);

        fill_equal_block_keys(keys, n);
        probe.load_keys(keys, device);
        std::int32_t count = -1;
        std::array<std::int32_t, 512> ids{};
        probe.run(device, token, n, n, count, ids);
        std::array<std::int32_t, 512> equal_expected{};
        for (std::int32_t i = 0; i < 512; ++i) { equal_expected[static_cast<std::size_t>(i)] = i; }
        if (!selection_matches(ids.data(), equal_expected.data(), count, "equal-keys")) {
            std::cerr << " at N=" << n << "\n";
            return false;
        }
        std::array<std::int32_t, 512> ids_again{};
        std::int32_t count_again = -1;
        probe.run(device, token, n, n, count_again, ids_again);
        if (count_again != count || std::memcmp(ids.data(), ids_again.data(), sizeof(ids)) != 0) {
            std::cerr << "FAIL: equal-keys path was not run-to-run identical at N=" << n << "\n";
            return false;
        }

        fill_tie_band_block_keys(keys, n, unique_high);
        probe.load_keys(keys, device);
        probe.run(device, token, n, n, count, ids);
        if (count != 512) {
            std::cerr << "FAIL: tie-band count=" << count << " at N=" << n << "\n";
            return false;
        }
        std::vector<std::int32_t> unique_prefix(ids.begin(), ids.begin() + unique_high);
        std::sort(unique_prefix.begin(), unique_prefix.end());
        for (std::int32_t i = 0; i < unique_high; ++i) {
            if (unique_prefix[static_cast<std::size_t>(i)] != i) {
                std::cerr << "FAIL: tie-band unique-high SET at N=" << n << " missing " << i << "\n";
                return false;
            }
        }
        for (std::int32_t i = 0; i < 512 - unique_high; ++i) {
            const std::int32_t got = ids[static_cast<std::size_t>(unique_high + i)];
            const std::int32_t exp = unique_high + i;
            if (got != exp) {
                std::cerr << "FAIL: tie-band tail id[" << (unique_high + i) << "]=" << got
                          << " expected=" << exp << " at N=" << n << "\n";
                return false;
            }
        }
        probe.run(device, token, n, n, count_again, ids_again);
        if (count_again != 512 || std::memcmp(ids.data(), ids_again.data(), sizeof(ids)) != 0) {
            std::cerr << "FAIL: tie-band path was not run-to-run identical at N=" << n << "\n";
            return false;
        }
        if (ids[0] == 0 && ids[1] == 0) {
            std::cerr << "FAIL: vacuous tie-band selection at N=" << n << "\n";
            return false;
        }

        drain_all_streams();
        for (int i = 0; i < 5; ++i) { probe.launch(n, n, device.stream); }
        device.synchronize();
        cudaEvent_t start = nullptr;
        cudaEvent_t stop  = nullptr;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start, device.stream));
        for (int i = 0; i < kIters; ++i) { probe.launch(n, n, device.stream); }
        CUDA_CHECK(cudaEventRecord(stop, device.stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        const float us_topk = ms * 1000.0F / static_cast<float>(kIters);
        std::cout << "  N=" << n << " indexer(topk)=" << us_topk
                  << " us  12-layer round=" << (12.0F * us_topk) << " us\n";
    }
    std::cout << "PASS: test_long_context_topk\n";
    return true;
}

bool test_padded_score_nan_sentinel(ninfer::DeviceContext& device) {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    constexpr std::int32_t maximum_blocks  = 1024;
    constexpr std::int32_t logical_pages   = 16;
    constexpr std::int32_t complete_blocks = 600;
    constexpr std::int32_t token           = complete_blocks * 4 - 1;
    IndexerProbe poisoned(maximum_blocks, logical_pages);
    IndexerProbe clean(maximum_blocks, logical_pages);

    std::vector<std::uint16_t> host_keys(128ULL * 64 * logical_pages, 0);
    fill_monotonic_block_keys(host_keys, complete_blocks);
    poisoned.load_keys(host_keys, device);
    clean.load_keys(host_keys, device);

    std::int32_t count_clean = -1;
    std::int32_t count_poisoned = -1;
    std::array<std::int32_t, 512> ids_clean{};
    std::array<std::int32_t, 512> ids_poisoned{};

    clean.workspace.reset();
    CUDA_CHECK(cudaMemsetAsync(clean.workspace.base(), 0, clean.workspace.capacity(),
                               device.stream));
    clean.run(device, token, maximum_blocks, maximum_blocks, count_clean, ids_clean);

    poisoned.workspace.reset();
    CUDA_CHECK(cudaMemsetAsync(poisoned.workspace.base(), 0xFF, poisoned.workspace.capacity(),
                               device.stream));
    poisoned.set_token(token);
    poisoned.launch(maximum_blocks, maximum_blocks, device.stream);
    drain_all_streams();
    poisoned.selected_count.copy_to_host(&count_poisoned, sizeof(count_poisoned));
    poisoned.selected_blocks.copy_to_host(ids_poisoned.data(), sizeof(ids_poisoned));

    if (count_clean != 512 || count_poisoned != 512) {
        std::cerr << "FAIL: NaN sentinel count clean=" << count_clean
                  << " poisoned=" << count_poisoned << "\n";
        return false;
    }
    double id_energy = 0.0;
    for (std::int32_t i = 0; i < 512; ++i) {
        if (ids_poisoned[static_cast<std::size_t>(i)] >= complete_blocks) {
            std::cerr << "FAIL: NaN sentinel selected padded id[" << i
                      << "]=" << ids_poisoned[static_cast<std::size_t>(i)]
                      << " >= complete_blocks=" << complete_blocks << "\n";
            return false;
        }
        if (ids_poisoned[static_cast<std::size_t>(i)] != ids_clean[static_cast<std::size_t>(i)]) {
            std::cerr << "FAIL: NaN sentinel selection mismatch at " << i
                      << " poisoned=" << ids_poisoned[static_cast<std::size_t>(i)]
                      << " clean=" << ids_clean[static_cast<std::size_t>(i)] << "\n";
            return false;
        }
        id_energy += static_cast<double>(ids_clean[static_cast<std::size_t>(i)]) *
                     static_cast<double>(ids_clean[static_cast<std::size_t>(i)]);
    }
    if (!(id_energy > 0.0) || !std::isfinite(id_energy) || ids_clean[0] == ids_clean[1]) {
        std::cerr << "FAIL: NaN sentinel comparison was vacuous (energy=" << id_energy
                  << " id0=" << ids_clean[0] << " id1=" << ids_clean[1] << ")\n";
        return false;
    }
    std::cout << "PASS: test_padded_score_nan_sentinel complete_blocks=" << complete_blocks
              << " id0=" << ids_clean[0] << " energy=" << id_energy << "\n";
    return true;
}

// G8 STEER 2 CHECK 3: the G6 NaN sentinel applied to the PREFILL select arm at a 64k
// envelope (active_blocks = maximum_blocks = 16384, complete_blocks = 600 so the sort
// path is taken). Poison the whole workspace with 0xFF before the public prefill_chunk.
bool test_prefill_padded_score_nan_sentinel(ninfer::DeviceContext& device) {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    constexpr std::int32_t maximum_blocks  = 16384;
    constexpr std::int32_t logical_pages   = 256;
    constexpr std::int32_t complete_blocks = 600;
    constexpr std::int32_t token           = complete_blocks * 4 - 1;
    IndexerProbe poisoned(maximum_blocks, logical_pages);
    IndexerProbe clean(maximum_blocks, logical_pages);

    std::vector<std::uint16_t> host_keys(128ULL * 64 * logical_pages, 0);
    fill_monotonic_block_keys(host_keys, complete_blocks);
    poisoned.load_keys(host_keys, device);
    clean.load_keys(host_keys, device);

    auto run_prefill = [&](IndexerProbe& probe, std::int32_t& count,
                           std::array<std::int32_t, 512>& ids) {
        probe.set_token(token);
        drain_all_streams();
        flash_next_qsa_indexer_prefill_chunk(
            probe.input_view, probe.weights, probe.token_view, probe.position_view, 0, 0, 0,
            probe.cache, maximum_blocks, token, probe.workspace, probe.selected_view,
            probe.count_view, device.stream);
        drain_all_streams();
        probe.selected_count.copy_to_host(&count, sizeof(count));
        probe.selected_blocks.copy_to_host(ids.data(), sizeof(ids));
    };

    std::int32_t count_clean = -1;
    std::int32_t count_poisoned = -1;
    std::array<std::int32_t, 512> ids_clean{};
    std::array<std::int32_t, 512> ids_poisoned{};

    clean.workspace.reset();
    CUDA_CHECK(cudaMemsetAsync(clean.workspace.base(), 0, clean.workspace.capacity(),
                               device.stream));
    run_prefill(clean, count_clean, ids_clean);

    poisoned.workspace.reset();
    CUDA_CHECK(cudaMemsetAsync(poisoned.workspace.base(), 0xFF, poisoned.workspace.capacity(),
                               device.stream));
    run_prefill(poisoned, count_poisoned, ids_poisoned);

    if (count_clean != 512 || count_poisoned != 512) {
        std::cerr << "FAIL: prefill NaN sentinel count clean=" << count_clean
                  << " poisoned=" << count_poisoned << "\n";
        return false;
    }
    double id_energy = 0.0;
    std::int32_t padded_hits = 0;
    for (std::int32_t i = 0; i < 512; ++i) {
        if (ids_poisoned[static_cast<std::size_t>(i)] >= complete_blocks) {
            ++padded_hits;
            std::cerr << "FAIL: prefill NaN sentinel selected padded id[" << i
                      << "]=" << ids_poisoned[static_cast<std::size_t>(i)]
                      << " >= complete_blocks=" << complete_blocks << "\n";
            return false;
        }
        if (ids_poisoned[static_cast<std::size_t>(i)] != ids_clean[static_cast<std::size_t>(i)]) {
            std::cerr << "FAIL: prefill NaN sentinel selection mismatch at " << i
                      << " poisoned=" << ids_poisoned[static_cast<std::size_t>(i)]
                      << " clean=" << ids_clean[static_cast<std::size_t>(i)] << "\n";
            return false;
        }
        id_energy += static_cast<double>(ids_clean[static_cast<std::size_t>(i)]) *
                     static_cast<double>(ids_clean[static_cast<std::size_t>(i)]);
    }
    if (!(id_energy > 0.0) || !std::isfinite(id_energy) || ids_clean[0] == ids_clean[1] ||
        padded_hits != 0) {
        std::cerr << "FAIL: prefill NaN sentinel comparison was vacuous (energy=" << id_energy
                  << " id0=" << ids_clean[0] << " id1=" << ids_clean[1]
                  << " padded_hits=" << padded_hits << ")\n";
        return false;
    }

    std::int32_t count_again = -1;
    std::array<std::int32_t, 512> ids_again{};
    clean.workspace.reset();
    CUDA_CHECK(cudaMemsetAsync(clean.workspace.base(), 0, clean.workspace.capacity(),
                               device.stream));
    run_prefill(clean, count_again, ids_again);
    if (count_again != 512 || std::memcmp(ids_clean.data(), ids_again.data(), sizeof(ids_clean)) != 0) {
        std::cerr << "FAIL: prefill select arm was not run-to-run identical at envelope=16384\n";
        return false;
    }

    std::cout << "PASS: test_prefill_padded_score_nan_sentinel envelope=" << maximum_blocks
              << " complete_blocks=" << complete_blocks << " id0=" << ids_clean[0]
              << " energy=" << id_energy << " padded_hits=0 run-to-run identical\n";
    return true;
}

bool test_prefill_decode_select_equivalence(ninfer::DeviceContext& device) {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    const std::int32_t ns[] = {513, 1024, 4096, 16384};
    constexpr std::int32_t unique_high = 400;
    auto compare_one = [&](IndexerProbe& decode_probe, IndexerProbe& prefill_probe,
                           std::int32_t n, const char* fixture) {
        const std::int32_t token = n * 4 - 1;
        std::int32_t count_decode = -1;
        std::int32_t count_prefill = -1;
        std::array<std::int32_t, 512> ids_decode{};
        std::array<std::int32_t, 512> ids_prefill{};
        decode_probe.run(device, token, n, n, count_decode, ids_decode);
        prefill_probe.set_token(token);
        drain_all_streams();
        flash_next_qsa_indexer_prefill_chunk(
            prefill_probe.input_view, prefill_probe.weights, prefill_probe.token_view,
            prefill_probe.position_view, 0, 0, 0, prefill_probe.cache, n, token,
            prefill_probe.workspace, prefill_probe.selected_view, prefill_probe.count_view,
            device.stream);
        drain_all_streams();
        prefill_probe.selected_count.copy_to_host(&count_prefill, sizeof(count_prefill));
        prefill_probe.selected_blocks.copy_to_host(ids_prefill.data(), sizeof(ids_prefill));
        if (count_decode != 512 || count_prefill != 512) {
            std::cerr << "FAIL: prefill-vs-decode count decode=" << count_decode
                      << " prefill=" << count_prefill << " at N=" << n << " " << fixture << "\n";
            return false;
        }
        std::array<std::int32_t, 512> set_decode = ids_decode;
        std::array<std::int32_t, 512> set_prefill = ids_prefill;
        std::sort(set_decode.begin(), set_decode.end());
        std::sort(set_prefill.begin(), set_prefill.end());
        if (std::memcmp(set_decode.data(), set_prefill.data(), sizeof(set_decode)) != 0) {
            std::cerr << "FAIL: prefill-vs-decode SET mismatch at N=" << n << " " << fixture << "\n";
            return false;
        }
        int order_mismatches = 0;
        double id_l2 = 0.0;
        double id_max_abs = 0.0;
        double energy = 0.0;
        for (std::int32_t i = 0; i < 512; ++i) {
            const double a = static_cast<double>(ids_decode[static_cast<std::size_t>(i)]);
            const double b = static_cast<double>(ids_prefill[static_cast<std::size_t>(i)]);
            const double d = a - b;
            id_l2 += d * d;
            id_max_abs = std::max(id_max_abs, std::abs(d));
            energy += a * a;
            if (ids_decode[static_cast<std::size_t>(i)] != ids_prefill[static_cast<std::size_t>(i)]) {
                ++order_mismatches;
            }
        }
        const double base_sq = energy;
        const double rel_l2 = (base_sq > 0.0) ? std::sqrt(id_l2 / base_sq) : 0.0;
        if (!(energy > 0.0) || !std::isfinite(energy) || !std::isfinite(rel_l2) ||
            ids_decode[0] == ids_decode[1]) {
            std::cerr << "FAIL: prefill-vs-decode comparison vacuous at N=" << n << " " << fixture
                      << " energy=" << energy << "\n";
            return false;
        }
        const bool want_order = std::strcmp(fixture, "monotonic") != 0;
        if (want_order && order_mismatches != 0) {
            std::cerr << "FAIL: prefill-vs-decode ORDER mismatch at N=" << n << " " << fixture
                      << " mismatches=" << order_mismatches << "\n";
            return false;
        }
        std::cout << "  N=" << n << " " << fixture << " SET exact order_mismatches="
                  << order_mismatches << " rel-L2=" << rel_l2 << " max_abs=" << id_max_abs
                  << " id0=" << ids_decode[0] << " energy=" << energy << "\n";
        return true;
    };
    for (std::int32_t n : ns) {
        const std::int32_t logical_pages = (n + 63) / 64;
        IndexerProbe decode_probe(n, logical_pages);
        IndexerProbe prefill_probe(n, logical_pages);
        std::vector<std::uint16_t> host_keys(128ULL * 64 * logical_pages, 0);

        fill_monotonic_block_keys(host_keys, n);
        decode_probe.load_keys(host_keys, device);
        prefill_probe.load_keys(host_keys, device);
        if (!compare_one(decode_probe, prefill_probe, n, "monotonic")) { return false; }

        fill_equal_block_keys(host_keys, n);
        decode_probe.load_keys(host_keys, device);
        prefill_probe.load_keys(host_keys, device);
        if (!compare_one(decode_probe, prefill_probe, n, "equal-keys")) { return false; }

        fill_tie_band_block_keys(host_keys, n, unique_high);
        decode_probe.load_keys(host_keys, device);
        prefill_probe.load_keys(host_keys, device);
        if (!compare_one(decode_probe, prefill_probe, n, "tie-band")) { return false; }
    }
    std::cout << "PASS: test_prefill_decode_select_equivalence\n";
    return true;
}

// G12: complete_blocks must be applied per token column inside one BN=64 GEMM tile.
// token_index 2044..2107 => complete_blocks 511..527. A per-CTA (tile-max) mask would
// let early tokens select later blocks that are not yet complete for them.
bool test_prefill_per_token_complete_threshold(ninfer::DeviceContext& device) {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    constexpr std::int32_t tokens          = 64;
    constexpr std::int32_t first_token     = 2044;
    constexpr std::int32_t last_token      = first_token + tokens - 1;
    constexpr std::int32_t first_complete  = (first_token + 1) / 4;
    constexpr std::int32_t last_complete   = (last_token + 1) / 4;
    constexpr std::int32_t maximum_blocks  = 1024;
    constexpr std::int32_t logical_pages   = (maximum_blocks + 63) / 64;
    static_assert(first_complete == 511 && last_complete == 527);
    static_assert(last_token == 2107);

    ninfer::DeviceBuffer input(2'560ULL * tokens * 2);
    ninfer::DeviceBuffer projection(640ULL * 2'560 * 2);
    ninfer::DeviceBuffer query_norm(128 * 2);
    ninfer::DeviceBuffer key_norm(128 * 2);
    ninfer::DeviceBuffer block_keys(128ULL * 64 * logical_pages * 2);
    ninfer::DeviceBuffer block_tables(static_cast<std::size_t>(logical_pages) * sizeof(std::int32_t));
    ninfer::DeviceBuffer raw_keys(128ULL * 4 * 2);
    ninfer::DeviceBuffer raw_positions(3ULL * 4 * sizeof(std::int32_t));
    ninfer::DeviceBuffer token_index(static_cast<std::size_t>(tokens) * sizeof(std::int32_t));
    ninfer::DeviceBuffer mrope_positions(static_cast<std::size_t>(tokens) * 3 * sizeof(std::int32_t));
    ninfer::DeviceBuffer selected_blocks(512ULL * tokens * sizeof(std::int32_t));
    ninfer::DeviceBuffer selected_count(static_cast<std::size_t>(tokens) * sizeof(std::int32_t));

    std::vector<std::uint16_t> input_values(2'560ULL * tokens, 0);
    for (std::int32_t t = 0; t < tokens; ++t) {
        input_values[static_cast<std::size_t>(t) * 2'560] = 0x3F80U;
    }
    input.copy_from_host(input_values.data(), input_values.size() * 2);
    std::vector<std::uint16_t> projection_values(640ULL * 2'560, 0);
    for (std::int32_t row = 0; row < 640; ++row) {
        projection_values[static_cast<std::size_t>(row) * 2'560] = 0x3F80U;
    }
    projection.copy_from_host(projection_values.data(), projection_values.size() * 2);
    query_norm.fill(0);
    key_norm.fill(0);
    raw_keys.fill(0);
    raw_positions.fill(0);
    std::vector<std::int32_t> page_ids(static_cast<std::size_t>(logical_pages));
    for (std::int32_t page = 0; page < logical_pages; ++page) { page_ids[static_cast<std::size_t>(page)] = page; }
    block_tables.copy_from_host(page_ids.data(), page_ids.size() * sizeof(std::int32_t));
    std::vector<std::int32_t> token_values(static_cast<std::size_t>(tokens));
    std::vector<std::int32_t> pos_values(static_cast<std::size_t>(tokens) * 3);
    for (std::int32_t t = 0; t < tokens; ++t) {
        token_values[static_cast<std::size_t>(t)] = first_token + t;
        pos_values[static_cast<std::size_t>(t)] = first_token + t;
        pos_values[static_cast<std::size_t>(tokens + t)] = first_token + t;
        pos_values[static_cast<std::size_t>(2 * tokens + t)] = first_token + t;
    }
    token_index.copy_from_host(token_values.data(), token_values.size() * sizeof(std::int32_t));
    mrope_positions.copy_from_host(pos_values.data(), pos_values.size() * sizeof(std::int32_t));

    std::vector<std::uint16_t> host_keys(128ULL * 64 * logical_pages, 0);
    fill_monotonic_block_keys(host_keys, last_complete);
    block_keys.copy_from_host(host_keys.data(), host_keys.size() * 2);

    AttentionWeights weights{};
    weights.indexer_query_key  = bf16_weight(projection.p, 640, 2'560);
    weights.indexer_query_norm = ninfer::Tensor(query_norm.p, ninfer::DType::BF16, {128});
    weights.indexer_key_norm   = ninfer::Tensor(key_norm.p, ninfer::DType::BF16, {128});
    QsaIndexerCacheView cache{
        .block_keys    = ninfer::Tensor(block_keys.p, ninfer::DType::BF16, {128, 64, logical_pages}),
        .block_tables  = ninfer::Tensor(block_tables.p, ninfer::DType::I32, {logical_pages, 1}),
        .raw_keys      = ninfer::Tensor(raw_keys.p, ninfer::DType::BF16, {128, 4, 1}),
        .raw_positions = ninfer::Tensor(raw_positions.p, ninfer::DType::I32, {3, 4, 1}),
    };
    ninfer::Tensor input_view(input.p, ninfer::DType::BF16, {2'560, tokens});
    ninfer::Tensor token_view(token_index.p, ninfer::DType::I32, {tokens});
    ninfer::Tensor position_view(mrope_positions.p, ninfer::DType::I32, {tokens, 3});
    ninfer::Tensor selected_view(selected_blocks.p, ninfer::DType::I32, {512, tokens});
    ninfer::Tensor count_view(selected_count.p, ninfer::DType::I32, {tokens});
    ninfer::WorkspaceArena workspace(256ULL * 1024 * 1024);
    CUDA_CHECK(cudaMemsetAsync(workspace.base(), 0xFF, workspace.capacity(), device.stream));
    drain_all_streams();
    flash_next_qsa_indexer_prefill_chunk(input_view, weights, token_view, position_view, 0, 0, 0,
                                         cache, maximum_blocks, first_token, workspace,
                                         selected_view, count_view, device.stream);
    drain_all_streams();

    std::vector<std::int32_t> counts(static_cast<std::size_t>(tokens), -1);
    std::vector<std::int32_t> ids(512ULL * tokens, -2);
    selected_count.copy_to_host(counts.data(), counts.size() * sizeof(std::int32_t));
    selected_blocks.copy_to_host(ids.data(), ids.size() * sizeof(std::int32_t));

    double energy = 0.0;
    int oob = 0;
    int saw_low = 0;
    int saw_high = 0;
    for (std::int32_t t = 0; t < tokens; ++t) {
        const std::int32_t complete = (first_token + t + 1) / 4;
        const std::int32_t want     = std::min(complete, 512);
        if (complete == first_complete) { ++saw_low; }
        if (complete == last_complete) { ++saw_high; }
        if (counts[static_cast<std::size_t>(t)] != want) {
            std::cerr << "FAIL: per-token sentinel count[" << t << "]=" << counts[static_cast<std::size_t>(t)]
                      << " want=" << want << " complete=" << complete << "\n";
            return false;
        }
        for (std::int32_t i = 0; i < 512; ++i) {
            const std::int32_t id = ids[static_cast<std::size_t>(t) * 512 + static_cast<std::size_t>(i)];
            if (i < want) {
                if (id < 0 || id >= complete) {
                    ++oob;
                    std::cerr << "FAIL: per-token sentinel token=" << (first_token + t)
                              << " complete=" << complete << " id[" << i << "]=" << id << "\n";
                    return false;
                }
                energy += static_cast<double>(id) * static_cast<double>(id);
            } else if (id != -1) {
                std::cerr << "FAIL: per-token sentinel pad token=" << (first_token + t)
                          << " id[" << i << "]=" << id << "\n";
                return false;
            }
        }
    }
    if (saw_low == 0 || saw_high == 0 || !(energy > 0.0) || !std::isfinite(energy) || oob != 0) {
        std::cerr << "FAIL: per-token sentinel vacuous energy=" << energy << " saw_low=" << saw_low
                  << " saw_high=" << saw_high << "\n";
        return false;
    }
    std::cout << "PASS: test_prefill_per_token_complete_threshold tokens=" << first_token << ".."
              << last_token << " complete=" << first_complete << ".." << last_complete
              << " energy=" << energy << " oob=0\n";
    return true;
}

std::uint32_t host_float_ascending_bits(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t mask =
        (static_cast<std::int32_t>(bits) < 0) ? 0xFFFFFFFFu : 0x80000000u;
    return bits ^ mask;
}

bool test_prefill_score_gemm_vs_host_oracle(ninfer::DeviceContext& device, std::int32_t n,
                                            bool random_inputs) {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    const std::int32_t tokens         = 1;
    const std::int32_t logical_pages  = (n + 63) / 64;
    const std::int32_t token          = n * 4 - 1;
    ninfer::DeviceBuffer input(2'560ULL * 2);
    ninfer::DeviceBuffer projection(640ULL * 2'560 * 2);
    ninfer::DeviceBuffer query_norm(128 * 2);
    ninfer::DeviceBuffer key_norm(128 * 2);
    ninfer::DeviceBuffer block_keys(128ULL * 64 * static_cast<std::size_t>(logical_pages) * 2);
    ninfer::DeviceBuffer block_tables(static_cast<std::size_t>(logical_pages) * sizeof(std::int32_t));
    ninfer::DeviceBuffer raw_keys(128ULL * 4 * 2);
    ninfer::DeviceBuffer raw_positions(3ULL * 4 * sizeof(std::int32_t));
    ninfer::DeviceBuffer token_index(sizeof(std::int32_t));
    ninfer::DeviceBuffer mrope_positions(3 * sizeof(std::int32_t));
    ninfer::DeviceBuffer selected_blocks(512 * sizeof(std::int32_t));
    ninfer::DeviceBuffer selected_count(sizeof(std::int32_t));

    std::vector<std::uint16_t> input_values(2'560, 0);
    std::vector<std::uint16_t> projection_values(640ULL * 2'560, 0);
    std::vector<std::uint16_t> qnorm_values(128, 0);
    std::vector<std::uint16_t> knorm_values(128, 0);
    if (random_inputs) {
        std::mt19937 rng(20260901u + static_cast<unsigned>(n));
        std::uniform_real_distribution<float> dist(-1.25F, 1.25F);
        for (auto& v : input_values) { v = float_to_bf16(dist(rng)); }
        for (auto& v : projection_values) { v = float_to_bf16(dist(rng)); }
        for (auto& v : qnorm_values) { v = float_to_bf16(dist(rng) * 0.1F); }
        for (auto& v : knorm_values) { v = float_to_bf16(dist(rng) * 0.1F); }
    } else {
        input_values[0] = 0x3F80U;
        for (std::int32_t row = 0; row < 640; ++row) {
            projection_values[static_cast<std::size_t>(row) * 2'560] = 0x3F80U;
        }
    }
    input.copy_from_host(input_values.data(), input_values.size() * 2);
    projection.copy_from_host(projection_values.data(), projection_values.size() * 2);
    query_norm.copy_from_host(qnorm_values.data(), qnorm_values.size() * 2);
    key_norm.copy_from_host(knorm_values.data(), knorm_values.size() * 2);
    raw_keys.fill(0);
    raw_positions.fill(0);
    std::vector<std::int32_t> page_ids(static_cast<std::size_t>(logical_pages));
    for (std::int32_t page = 0; page < logical_pages; ++page) {
        page_ids[static_cast<std::size_t>(page)] = page;
    }
    block_tables.copy_from_host(page_ids.data(), page_ids.size() * sizeof(std::int32_t));
    token_index.copy_from_host(&token, sizeof(token));
    constexpr std::array<std::int32_t, 3> zero_pos{};
    mrope_positions.copy_from_host(zero_pos.data(), sizeof(zero_pos));
    std::vector<std::uint16_t> host_keys(128ULL * 64 * static_cast<std::size_t>(logical_pages), 0);
    if (random_inputs) {
        std::mt19937 rng(20260901u + 17u + static_cast<unsigned>(n));
        std::uniform_real_distribution<float> dist(-1.25F, 1.25F);
        for (std::int32_t block = 0; block < n; ++block) {
            const std::int32_t page   = block / 64;
            const std::int32_t offset = block % 64;
            const std::size_t index =
                static_cast<std::size_t>(page) * 64ULL * 128ULL +
                static_cast<std::size_t>(offset) * 128ULL;
            for (int dim = 0; dim < 128; ++dim) {
                host_keys[index + static_cast<std::size_t>(dim)] = float_to_bf16(dist(rng));
            }
        }
    } else {
        fill_monotonic_block_keys(host_keys, n);
    }
    block_keys.copy_from_host(host_keys.data(), host_keys.size() * 2);

    AttentionWeights weights{};
    weights.indexer_query_key  = bf16_weight(projection.p, 640, 2'560);
    weights.indexer_query_norm = ninfer::Tensor(query_norm.p, ninfer::DType::BF16, {128});
    weights.indexer_key_norm   = ninfer::Tensor(key_norm.p, ninfer::DType::BF16, {128});
    QsaIndexerCacheView cache{
        .block_keys    = ninfer::Tensor(block_keys.p, ninfer::DType::BF16, {128, 64, logical_pages}),
        .block_tables  = ninfer::Tensor(block_tables.p, ninfer::DType::I32, {logical_pages, 1}),
        .raw_keys      = ninfer::Tensor(raw_keys.p, ninfer::DType::BF16, {128, 4, 1}),
        .raw_positions = ninfer::Tensor(raw_positions.p, ninfer::DType::I32, {3, 4, 1}),
    };
    ninfer::Tensor input_view(input.p, ninfer::DType::BF16, {2'560, 1});
    ninfer::Tensor token_view(token_index.p, ninfer::DType::I32, {1});
    ninfer::Tensor position_view(mrope_positions.p, ninfer::DType::I32, {1, 3});
    ninfer::Tensor selected_view(selected_blocks.p, ninfer::DType::I32, {512, 1});
    ninfer::Tensor count_view(selected_count.p, ninfer::DType::I32, {1});
    ninfer::WorkspaceArena workspace(flash_next_qsa_indexer_workspace_capacity_bytes(n, 1) +
                                     32ULL * 1024 * 1024);
    const std::int32_t tile_size = flash_next_qsa_indexer_tile_size(n, tokens);
    const std::size_t sort_temp  = flash_next_qsa_indexer_sort_temp_bytes(n, tile_size);
    auto scratch = allocate_flash_next_qsa_indexer_workspace(workspace, n, tokens, tile_size,
                                                             sort_temp);
    ninfer::ops::linear(input_view, weights.indexer_query_key, scratch.projected, device.stream);
    flash_next_qsa_indexer_prefill_launch(token_view, position_view, 0, 0, 0,
                                          weights.indexer_query_norm, weights.indexer_key_norm,
                                          cache, scratch, n, token, selected_view, count_view,
                                          device.stream);
    drain_all_streams();

    std::vector<float> gemm_scores(static_cast<std::size_t>(n));
    std::vector<std::uint16_t> host_query(128ULL * 4);
    CUDA_CHECK(cudaMemcpy(gemm_scores.data(), scratch.scores.data,
                          static_cast<std::size_t>(n) * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_query.data(), scratch.query.data, host_query.size() * 2,
                          cudaMemcpyDeviceToHost));
    block_keys.copy_to_host(host_keys.data(), host_keys.size() * 2);

    constexpr float kIndexerScaling = 0.08838834764831845F;
    double num = 0.0;
    double den = 0.0;
    double max_abs = 0.0;
    int compared = 0;
    for (std::int32_t block = 0; block < n; ++block) {
        const std::int32_t page   = block / 64;
        const std::int32_t offset = block % 64;
        const std::size_t key_i =
            static_cast<std::size_t>(page) * 64ULL * 128ULL + static_cast<std::size_t>(offset) * 128ULL;
        float score = 0.0F;
        for (int head = 0; head < 4; ++head) {
            float dot = 0.0F;
            for (int dim = 0; dim < 128; ++dim) {
                const float q = bf16_bits_to_float(
                    host_query[static_cast<std::size_t>(head) * 128 + static_cast<std::size_t>(dim)]);
                const float k = bf16_bits_to_float(host_keys[key_i + static_cast<std::size_t>(dim)]);
                dot += q * k;
            }
            score += std::max(dot, 0.0F);
        }
        score *= kIndexerScaling;
        const float got = gemm_scores[static_cast<std::size_t>(block)];
        if (!std::isfinite(got) || !std::isfinite(score)) {
            std::cerr << "FAIL: non-finite score N=" << n << " block=" << block << " gemm=" << got
                      << " host=" << score << "\n";
            return false;
        }
        const double d = static_cast<double>(got) - static_cast<double>(score);
        num += d * d;
        den += static_cast<double>(score) * static_cast<double>(score);
        max_abs = std::max(max_abs, std::abs(d));
        ++compared;
    }
    if (compared != n || !(den > 0.0) || !std::isfinite(den) || !std::isfinite(num)) {
        std::cerr << "FAIL: score oracle vacuous N=" << n << " den=" << den << "\n";
        return false;
    }
    const double rel_l2 = std::sqrt(num / den);
    std::vector<std::pair<std::uint64_t, std::int32_t>> ranked(static_cast<std::size_t>(n));
    for (std::int32_t block = 0; block < n; ++block) {
        float score = 0.0F;
        const std::int32_t page   = block / 64;
        const std::int32_t offset = block % 64;
        const std::size_t key_i =
            static_cast<std::size_t>(page) * 64ULL * 128ULL + static_cast<std::size_t>(offset) * 128ULL;
        for (int head = 0; head < 4; ++head) {
            float dot = 0.0F;
            for (int dim = 0; dim < 128; ++dim) {
                const float q = bf16_bits_to_float(
                    host_query[static_cast<std::size_t>(head) * 128 + static_cast<std::size_t>(dim)]);
                const float k = bf16_bits_to_float(host_keys[key_i + static_cast<std::size_t>(dim)]);
                dot += q * k;
            }
            score += std::max(dot, 0.0F);
        }
        score *= kIndexerScaling;
        const std::uint32_t rank = host_float_ascending_bits(score);
        const std::uint32_t tie  = ~static_cast<std::uint32_t>(block);
        ranked[static_cast<std::size_t>(block)] = {
            (static_cast<std::uint64_t>(rank) << 32) | tie, block};
    }
    std::sort(ranked.begin(), ranked.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    std::array<std::int32_t, 512> host_top{};
    std::array<std::int32_t, 512> gemm_top{};
    std::int32_t gemm_count = -1;
    selected_count.copy_to_host(&gemm_count, sizeof(gemm_count));
    selected_blocks.copy_to_host(gemm_top.data(), sizeof(gemm_top));
    if (gemm_count != 512) {
        std::cerr << "FAIL: GEMM select count=" << gemm_count << " at N=" << n << "\n";
        return false;
    }
    for (std::int32_t i = 0; i < 512; ++i) {
        host_top[static_cast<std::size_t>(i)] = ranked[static_cast<std::size_t>(i)].second;
    }
    std::array<std::int32_t, 512> host_set = host_top;
    std::array<std::int32_t, 512> gemm_set = gemm_top;
    std::sort(host_set.begin(), host_set.end());
    std::sort(gemm_set.begin(), gemm_set.end());
    const bool set_match =
        std::memcmp(host_set.data(), gemm_set.data(), sizeof(host_set)) == 0;
    int order_mismatches = 0;
    for (std::int32_t i = 0; i < 512; ++i) {
        if (host_top[static_cast<std::size_t>(i)] != gemm_top[static_cast<std::size_t>(i)]) {
            ++order_mismatches;
        }
    }
    const char* label = random_inputs ? "random-bf16" : "monotonic";
    std::cout << "  N=" << n << " " << label << " GEMM vs host-sequential-FP32 rel-L2=" << rel_l2
              << " max_abs=" << max_abs << " compared=" << compared << " den=" << den
              << " SET=" << (set_match ? "exact" : "DIFF") << " order_mismatches=" << order_mismatches
              << "\n";
    return true;
}

bool test_g14_prefill_indexer_cost(ninfer::DeviceContext& device, std::int32_t n,
                                   std::int32_t tokens, std::int32_t first_token) {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    const std::int32_t logical_pages = (n + 63) / 64;
    const std::int32_t tile          = flash_next_qsa_indexer_tile_size(n, tokens);
    const int n_tiles                = (tokens + tile - 1) / tile;
    ninfer::DeviceBuffer input(2'560ULL * static_cast<std::size_t>(tokens) * 2);
    ninfer::DeviceBuffer projection(640ULL * 2'560 * 2);
    ninfer::DeviceBuffer query_norm(128 * 2);
    ninfer::DeviceBuffer key_norm(128 * 2);
    ninfer::DeviceBuffer block_keys(128ULL * 64 * static_cast<std::size_t>(logical_pages) * 2);
    ninfer::DeviceBuffer block_tables(static_cast<std::size_t>(logical_pages) * sizeof(std::int32_t));
    ninfer::DeviceBuffer raw_keys(128ULL * 4 * 2);
    ninfer::DeviceBuffer raw_positions(3ULL * 4 * sizeof(std::int32_t));
    ninfer::DeviceBuffer token_index(static_cast<std::size_t>(tokens) * sizeof(std::int32_t));
    ninfer::DeviceBuffer mrope_positions(static_cast<std::size_t>(tokens) * 3 * sizeof(std::int32_t));
    ninfer::DeviceBuffer selected_blocks(512ULL * static_cast<std::size_t>(tokens) *
                                         sizeof(std::int32_t));
    ninfer::DeviceBuffer selected_count(static_cast<std::size_t>(tokens) * sizeof(std::int32_t));

    std::vector<std::uint16_t> input_values(2'560ULL * static_cast<std::size_t>(tokens), 0);
    std::vector<std::uint16_t> projection_values(640ULL * 2'560, 0);
    input_values[0] = 0x3F80U;
    for (std::int32_t row = 0; row < 640; ++row) {
        projection_values[static_cast<std::size_t>(row) * 2'560] = 0x3F80U;
    }
    input.copy_from_host(input_values.data(), input_values.size() * 2);
    projection.copy_from_host(projection_values.data(), projection_values.size() * 2);
    query_norm.fill(0);
    key_norm.fill(0);
    raw_keys.fill(0);
    raw_positions.fill(0);
    std::vector<std::int32_t> page_ids(static_cast<std::size_t>(logical_pages));
    for (std::int32_t page = 0; page < logical_pages; ++page) {
        page_ids[static_cast<std::size_t>(page)] = page;
    }
    block_tables.copy_from_host(page_ids.data(), page_ids.size() * sizeof(std::int32_t));
    std::vector<std::int32_t> tokens_host(static_cast<std::size_t>(tokens));
    std::vector<std::int32_t> pos_host(static_cast<std::size_t>(tokens) * 3);
    for (std::int32_t t = 0; t < tokens; ++t) {
        tokens_host[static_cast<std::size_t>(t)] = first_token + t;
        pos_host[static_cast<std::size_t>(t)] = first_token + t;
        pos_host[static_cast<std::size_t>(tokens + t)] = first_token + t;
        pos_host[static_cast<std::size_t>(2 * tokens + t)] = first_token + t;
    }
    token_index.copy_from_host(tokens_host.data(), tokens_host.size() * sizeof(std::int32_t));
    mrope_positions.copy_from_host(pos_host.data(), pos_host.size() * sizeof(std::int32_t));
    std::vector<std::uint16_t> host_keys(128ULL * 64 * static_cast<std::size_t>(logical_pages), 0);
    fill_monotonic_block_keys(host_keys, n);
    block_keys.copy_from_host(host_keys.data(), host_keys.size() * 2);

    AttentionWeights weights{};
    weights.indexer_query_key  = bf16_weight(projection.p, 640, 2'560);
    weights.indexer_query_norm = ninfer::Tensor(query_norm.p, ninfer::DType::BF16, {128});
    weights.indexer_key_norm   = ninfer::Tensor(key_norm.p, ninfer::DType::BF16, {128});
    QsaIndexerCacheView cache{
        .block_keys    = ninfer::Tensor(block_keys.p, ninfer::DType::BF16, {128, 64, logical_pages}),
        .block_tables  = ninfer::Tensor(block_tables.p, ninfer::DType::I32, {logical_pages, 1}),
        .raw_keys      = ninfer::Tensor(raw_keys.p, ninfer::DType::BF16, {128, 4, 1}),
        .raw_positions = ninfer::Tensor(raw_positions.p, ninfer::DType::I32, {3, 4, 1}),
    };
    ninfer::Tensor input_view(input.p, ninfer::DType::BF16, {2'560, tokens});
    ninfer::Tensor token_view(token_index.p, ninfer::DType::I32, {tokens});
    ninfer::Tensor position_view(mrope_positions.p, ninfer::DType::I32, {tokens, 3});
    ninfer::Tensor selected_view(selected_blocks.p, ninfer::DType::I32, {512, tokens});
    ninfer::Tensor count_view(selected_count.p, ninfer::DType::I32, {tokens});
    const std::size_t sort_temp = flash_next_qsa_indexer_sort_temp_bytes(n, tile);
    ninfer::WorkspaceLayoutBuilder layout;
    (void)allocate_flash_next_qsa_indexer_workspace(layout, n, tokens, tile, sort_temp);
    ninfer::WorkspaceArena workspace(layout.peak_bytes(256) + 32ULL * 1024 * 1024);
    auto scratch =
        allocate_flash_next_qsa_indexer_workspace(workspace, n, tokens, tile, sort_temp);
    ninfer::ops::linear(input_view, weights.indexer_query_key, scratch.projected, device.stream);
    drain_all_streams();

    auto launch_once = [&]() {
        flash_next_qsa_indexer_prefill_launch(token_view, position_view, 0, 0, 0,
                                              weights.indexer_query_norm, weights.indexer_key_norm,
                                              cache, scratch, n, first_token, selected_view,
                                              count_view, device.stream);
    };
    launch_once();
    drain_all_streams();
    constexpr int kIters = 5;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop  = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, device.stream));
    for (int i = 0; i < kIters; ++i) { launch_once(); }
    CUDA_CHECK(cudaEventRecord(stop, device.stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    const float layer_ms = ms / static_cast<float>(kIters);
    std::cout << "G14 INDEXER prefill_launch n=" << n << " tokens=" << tokens
              << " first=" << first_token << " tile=" << tile << " n_tiles=" << n_tiles
              << " items_per_tile=" << (static_cast<long long>(n) * tile)
              << " layer_ms=" << layer_ms << " 12-layer_ms=" << (12.0F * layer_ms) << "\n";
    return true;
}

} // namespace

int main(int argc, char** argv) {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    int device_count              = 0;
    const cudaError_t count_error = cudaGetDeviceCount(&device_count);
    if (cuda_unavailable(count_error) || device_count == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    CUDA_CHECK(count_error);

    ninfer::DeviceContext device(0);
    if (argc >= 2 && std::string(argv[1]) == "g14-sort") {
        const bool only64  = argc >= 3 && std::string(argv[2]) == "16384";
        const bool only256 = argc >= 3 && std::string(argv[2]) == "65536";
        if (!only256 && !test_g14_prefill_indexer_cost(device, 16384, 2048, 63488)) { return 1; }
        if (!only64 && !test_g14_prefill_indexer_cost(device, 65536, 2048, 260096)) { return 1; }
        std::cout << "PASS: test_g14_prefill_indexer_cost\n";
        return 0;
    }

    constexpr std::int32_t maximum_blocks = 513;
    constexpr std::int32_t logical_pages  = 9;
    ninfer::DeviceBuffer input(2'560ULL * 8 * 2);
    ninfer::DeviceBuffer small_t_output(640ULL * 8 * 2);
    ninfer::DeviceBuffer projection(640ULL * 2'560 * 2);
    ninfer::DeviceBuffer query_norm(128 * 2);
    ninfer::DeviceBuffer key_norm(128 * 2);
    ninfer::DeviceBuffer block_keys(128ULL * 64 * logical_pages * 2);
    ninfer::DeviceBuffer block_tables(logical_pages * sizeof(std::int32_t));
    ninfer::DeviceBuffer raw_keys(128ULL * 4 * 2);
    ninfer::DeviceBuffer raw_positions(3ULL * 4 * sizeof(std::int32_t));
    ninfer::DeviceBuffer token_index(sizeof(std::int32_t));
    ninfer::DeviceBuffer mrope_positions(3 * sizeof(std::int32_t));
    ninfer::DeviceBuffer table_row(sizeof(std::int32_t));
    ninfer::DeviceBuffer state_slot(sizeof(std::int32_t));
    ninfer::DeviceBuffer selected_blocks(512 * sizeof(std::int32_t));
    ninfer::DeviceBuffer selected_count(sizeof(std::int32_t));

    std::vector<std::uint16_t> input_values(2'560ULL * 8, 0);
    for (std::int32_t token = 0; token < 8; ++token) {
        input_values[static_cast<std::size_t>(token) * 2'560] = 0x3F80U;
    }
    input.copy_from_host(input_values.data(), input_values.size() * sizeof(std::uint16_t));
    std::vector<std::uint16_t> projection_values(640ULL * 2'560, 0);
    for (std::int32_t row = 0; row < 640; ++row) {
        projection_values[static_cast<std::size_t>(row) * 2'560] = 0x3F80U;
    }
    projection.copy_from_host(projection_values.data(),
                              projection_values.size() * sizeof(std::uint16_t));
    query_norm.fill(0);
    key_norm.fill(0);
    block_keys.fill(0);
    raw_keys.fill(0);
    raw_positions.fill(0);
    std::array<std::int32_t, logical_pages> page_ids{};
    for (std::int32_t page = 0; page < logical_pages; ++page) { page_ids[page] = page; }
    block_tables.copy_from_host(page_ids.data(), sizeof(page_ids));
    constexpr std::array<std::int32_t, 3> zero_positions{};
    mrope_positions.copy_from_host(zero_positions.data(), sizeof(zero_positions));
    constexpr std::int32_t zero = 0;
    table_row.copy_from_host(&zero, sizeof(zero));
    state_slot.copy_from_host(&zero, sizeof(zero));

    AttentionWeights weights{};
    weights.indexer_query_key  = bf16_weight(projection.p, 640, 2'560);
    weights.indexer_query_norm = ninfer::Tensor(query_norm.p, ninfer::DType::BF16, {128});
    weights.indexer_key_norm   = ninfer::Tensor(key_norm.p, ninfer::DType::BF16, {128});
    QsaIndexerCacheView cache{
        .block_keys   = ninfer::Tensor(block_keys.p, ninfer::DType::BF16, {128, 64, logical_pages}),
        .block_tables = ninfer::Tensor(block_tables.p, ninfer::DType::I32, {logical_pages, 1}),
        .raw_keys     = ninfer::Tensor(raw_keys.p, ninfer::DType::BF16, {128, 4, 1}),
        .raw_positions = ninfer::Tensor(raw_positions.p, ninfer::DType::I32, {3, 4, 1}),
    };
    ninfer::Tensor input_view(input.p, ninfer::DType::BF16, {2'560, 1});
    ninfer::Tensor token_view(token_index.p, ninfer::DType::I32, {1});
    ninfer::Tensor position_view(mrope_positions.p, ninfer::DType::I32, {1, 3});
    ninfer::Tensor table_row_view(table_row.p, ninfer::DType::I32, {1});
    ninfer::Tensor state_slot_view(state_slot.p, ninfer::DType::I32, {1});
    ninfer::Tensor selected_view(selected_blocks.p, ninfer::DType::I32, {512, 1});
    ninfer::Tensor count_view(selected_count.p, ninfer::DType::I32, {1});
    ninfer::WorkspaceArena workspace(
        flash_next_qsa_indexer_workspace_capacity_bytes(maximum_blocks, 1));

    token_index.copy_from_host(&zero, sizeof(zero));
    drain_all_streams();
    flash_next_qsa_indexer_decode(input_view, weights, token_view, position_view, table_row_view,
                                  state_slot_view, state_slot_view, cache, maximum_blocks, 0,
                                  workspace, selected_view, count_view, device.stream);
    drain_all_streams();
    std::int32_t actual_count = -1;
    selected_count.copy_to_host(&actual_count, sizeof(actual_count));
    if (actual_count != 0) {
        std::cerr << "Flash-Next QSA indexer selected a complete block for active_blocks=0\n";
        return 1;
    }

    constexpr std::int32_t final_token = 2'051; // completes block 512, yielding 513 blocks
    token_index.copy_from_host(&final_token, sizeof(final_token));
    drain_all_streams();
    flash_next_qsa_indexer_decode(input_view, weights, token_view, position_view, table_row_view,
                                  state_slot_view, state_slot_view, cache, maximum_blocks,
                                  maximum_blocks, workspace, selected_view, count_view,
                                  device.stream);
    drain_all_streams();
    selected_count.copy_to_host(&actual_count, sizeof(actual_count));
    std::array<std::int32_t, 512> actual_ids{};
    selected_blocks.copy_to_host(actual_ids.data(), sizeof(actual_ids));
    if (actual_count != 512 || actual_ids.front() != 512) {
        std::cerr << "Flash-Next QSA indexer did not prioritize the positive 513th block\n";
        return 1;
    }
    for (std::int32_t index = 1; index < 512; ++index) {
        if (actual_ids[static_cast<std::size_t>(index)] != index - 1) {
            std::cerr << "Flash-Next QSA stable top-k tie order changed at " << index << '\n';
            return 1;
        }
    }
    std::array<std::uint16_t, 128> compressed_key{};
    constexpr std::size_t block_512_byte_offset = 8ULL * 128 * 64 * 2;
    block_keys.copy_to_host(compressed_key.data(), sizeof(compressed_key), block_512_byte_offset);
    if (!std::all_of(compressed_key.begin(), compressed_key.end(),
                     [](std::uint16_t value) { return value == 0x3F80U; })) {
        std::cerr << "Flash-Next QSA four-token compressed key was not exact BF16 one\n";
        return 1;
    }

    ninfer::Tensor small_t_input(input.p, ninfer::DType::BF16, {2'560, 8});
    ninfer::Tensor small_t_output_view(small_t_output.p, ninfer::DType::BF16, {640, 8});
    ninfer::WorkspaceArena empty_workspace(1);
    if (ninfer::ops::linear_workspace_capacity_bytes(
            ninfer::QType::BF16_CTRL, 640, 2'560, ninfer::ops::LinearPolicy::A16Only, 1, 8) != 0) {
        std::cerr << "Flash-Next QSA BF16 indexer unexpectedly required workspace\n";
        return 1;
    }
    ninfer::ops::linear(small_t_input, weights.indexer_query_key, small_t_output_view,
                        ninfer::ops::LinearPolicy::A16Only, empty_workspace, device.stream);
    drain_all_streams();
    std::vector<std::uint16_t> actual_small_t(640ULL * 8);
    small_t_output.copy_to_host(actual_small_t.data(),
                                actual_small_t.size() * sizeof(std::uint16_t));
    if (!std::all_of(actual_small_t.begin(), actual_small_t.end(),
                     [](std::uint16_t value) { return value == 0x3F80U; })) {
        std::cerr << "Flash-Next QSA BF16 T=8 projection was not exact BF16 one\n";
        return 1;
    }

    if (!test_block_publish_equivalence(device)) {
        std::cerr << "FAIL: test_block_publish_equivalence failed\n";
        return 1;
    }
    if (!test_selection_equivalence(device)) {
        std::cerr << "FAIL: test_selection_equivalence failed\n";
        return 1;
    }

    float us_bypass = 0.0F;
    float us_sort   = 0.0F;
    if (!test_fully_selected_identity_bypass(device, us_bypass, us_sort)) {
        std::cerr << "FAIL: test_fully_selected_identity_bypass failed\n";
        return 1;
    }
    const float us_round_bypass = 12.0F * us_bypass;
    const float us_round_sort   = 12.0F * us_sort;
    std::cout << "G1 indexer decode active_blocks=512 (identity bypass): " << us_bypass
              << " us/iter\n";
    std::cout << "G1 indexer decode active_blocks=513 (long-context select): " << us_sort
              << " us/iter\n";
    std::cout << "G1 decode-round estimate (12 QSA layers): bypass=" << us_round_bypass
              << " us  sort=" << us_round_sort << " us  saving=" << (us_round_sort - us_round_bypass)
              << " us\n";

    if (!test_long_context_topk(device)) {
        std::cerr << "FAIL: test_long_context_topk failed\n";
        return 1;
    }
    if (!test_padded_score_nan_sentinel(device)) {
        std::cerr << "FAIL: test_padded_score_nan_sentinel failed\n";
        return 1;
    }
    if (!test_prefill_padded_score_nan_sentinel(device)) {
        std::cerr << "FAIL: test_prefill_padded_score_nan_sentinel failed\n";
        return 1;
    }
    if (!test_prefill_decode_select_equivalence(device)) {
        std::cerr << "FAIL: test_prefill_decode_select_equivalence failed\n";
        return 1;
    }
    if (!test_prefill_per_token_complete_threshold(device)) {
        std::cerr << "FAIL: test_prefill_per_token_complete_threshold failed\n";
        return 1;
    }
    std::cout << "\n=== G12 score GEMM vs host sequential FP32 ===\n";
    if (!test_prefill_score_gemm_vs_host_oracle(device, 1024, false) ||
        !test_prefill_score_gemm_vs_host_oracle(device, 16384, false) ||
        !test_prefill_score_gemm_vs_host_oracle(device, 1024, true) ||
        !test_prefill_score_gemm_vs_host_oracle(device, 16384, true)) {
        std::cerr << "FAIL: test_prefill_score_gemm_vs_host_oracle failed\n";
        return 1;
    }
    std::cout << "PASS: test_prefill_score_gemm_vs_host_oracle\n";

    std::cout << "PASS: test_qsa_indexer\n";
    return 0;
}
