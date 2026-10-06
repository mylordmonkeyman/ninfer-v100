#pragma once

#include "artifact/reader.h"

#include <cstddef>
#include <cstdint>
#include <array>
#include <span>

namespace ninfer::targets::qwen3_8_flash_next::detail {

inline constexpr std::uint64_t kPleRowsPerShard = 2'500'012;
inline constexpr std::uint64_t kPleShardCount   = 128;
inline constexpr std::uint64_t kPleTableRows    = kPleRowsPerShard * kPleShardCount;
inline constexpr std::uint64_t kPleRowWidth     = 160;

struct PleRowAddress {
    std::uint32_t shard;
    std::uint32_t row;
};

[[nodiscard]] PleRowAddress locate_ple_row(std::uint64_t global_row);

struct PleShardView {
    std::span<const std::byte> codes;
    std::span<const std::byte> scales;
    std::uint64_t rows;
    std::uint64_t width;
    std::uint64_t groups_per_row;
    std::uint64_t code_absolute_offset = 0;
    std::uint64_t scale_absolute_offset = 0;
};

struct PleTableView {
    std::array<PleShardView, kPleShardCount> shards;
    artifact::DirectReader direct_reader;
};

[[nodiscard]] PleShardView make_ple_shard_view(std::span<const std::byte> encoded,
                                               std::uint64_t rows  = kPleRowsPerShard,
                                               std::uint64_t width = kPleRowWidth,
                                               std::uint64_t absolute_offset = 0);

// Independent scalar oracle for U4Z8 group-16 with FP16 multipliers.
void dequantize_ple_row(const PleShardView& shard, std::uint64_t row, std::span<float> output);

void gather_ple_rows(const PleTableView& table, std::span<const std::int64_t, 16> global_rows,
                     std::span<float> output);

// Runtime gather contract: FP32 dequantization followed by round-to-nearest-even BF16 storage.
void gather_ple_rows_bf16(const PleTableView& table, std::span<const std::int64_t, 16> global_rows,
                          std::span<std::uint16_t> output);

// Runtime compressed gather: gathers raw U4 codes and FP16 scales into pinned staging buffers.
void gather_ple_rows_compressed(const PleTableView& table,
                                std::span<const std::array<std::int64_t, 16>> global_rows,
                                std::span<std::byte> out_codes,
                                std::span<std::byte> out_scales);

} // namespace ninfer::targets::qwen3_8_flash_next::detail
