#include "targets/qwen3_8_flash_next/impl/ple_table.h"

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

int main() {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    constexpr std::uint64_t rows         = 2;
    constexpr std::uint64_t width        = 160;
    constexpr std::uint64_t scale_offset = 256;
    std::vector<std::byte> encoded(scale_offset + rows * (width / 16) * 2, std::byte{0});

    // Row zero is all affine zero codes except its first group, which contains
    // codes 0..15. Packed U4 is low nibble first.
    std::fill_n(encoded.begin(), rows * width / 2, std::byte{0x88});
    for (std::uint8_t index = 0; index < 8; ++index) {
        encoded[index] = static_cast<std::byte>(index * 2 | ((index * 2 + 1) << 4));
    }
    constexpr std::uint16_t half_point_five = 0x3800;
    for (std::size_t offset = scale_offset; offset < encoded.size(); offset += 2) {
        std::memcpy(encoded.data() + offset, &half_point_five, sizeof(half_point_five));
    }

    const PleShardView shard = make_ple_shard_view(encoded, rows, width);
    std::vector<float> row(width);
    dequantize_ple_row(shard, 0, row);
    for (std::size_t index = 0; index < 16; ++index) {
        const float expected = (static_cast<float>(index) - 8.0F) * 0.5F;
        if (std::abs(row[index] - expected) > 1e-7F) {
            std::cerr << "PLE U4 dequant mismatch at column " << index << '\n';
            return 1;
        }
    }
    for (std::size_t index = 16; index < row.size(); ++index) {
        if (row[index] != 0.0F) {
            std::cerr << "PLE affine zero code did not decode to zero\n";
            return 1;
        }
    }

    const PleRowAddress final = locate_ple_row(320'001'535);
    if (final.shard != 127 || final.row != 2'500'011) {
        std::cerr << "PLE final row address is wrong\n";
        return 1;
    }

    PleTableView table;
    for (PleShardView& view : table.shards) { view = shard; }
    std::array<std::int64_t, 16> indices{};
    for (std::size_t head = 0; head < indices.size(); ++head) {
        indices[head] = static_cast<std::int64_t>(head * kPleRowsPerShard);
    }
    std::array<float, 16 * width> gathered{};
    gather_ple_rows(table, indices, gathered);
    for (std::size_t head = 0; head < indices.size(); ++head) {
        for (std::size_t column = 0; column < width; ++column) {
            if (gathered[head * width + column] != row[column]) {
                std::cerr << "PLE sparse gather changed head/row ordering\n";
                return 1;
            }
        }
    }

    std::array<std::uint16_t, 16 * width> gathered_bf16{};
    gather_ple_rows_bf16(table, indices, gathered_bf16);
    for (std::size_t index = 0; index < gathered_bf16.size(); ++index) {
        const std::uint32_t expected_bits = std::bit_cast<std::uint32_t>(gathered[index]);
        if (gathered_bf16[index] != static_cast<std::uint16_t>(expected_bits >> 16U)) {
            std::cerr << "PLE BF16 gather changed represented row values\n";
            return 1;
        }
    }

    std::array<std::array<std::int64_t, 16>, 1> chunk_indices = {indices};
    std::vector<std::byte> comp_codes(16 * 80);
    std::vector<std::byte> comp_scales(16 * 20);
    gather_ple_rows_compressed(table, chunk_indices, comp_codes, comp_scales);
    for (std::size_t head = 0; head < 16; ++head) {
        if (std::memcmp(comp_codes.data() + head * 80, shard.codes.data(), 80) != 0) {
            std::cerr << "PLE compressed gather code mismatch at head " << head << '\n';
            return 1;
        }
        if (std::memcmp(comp_scales.data() + head * 20, shard.scales.data(), 20) != 0) {
            std::cerr << "PLE compressed gather scale mismatch at head " << head << '\n';
            return 1;
        }
    }

    return 0;
}
