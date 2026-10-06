#include "targets/qwen3_8_flash_next/impl/ple_direct_io.h"
#include "artifact_fixture.h"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace ninfer::targets::qwen3_8_flash_next::detail;

int main() {
    try {
        constexpr std::size_t rows = 512, encoded_bytes = rows * 100;
        using ninfer::test::artifact_fixture::Json;
        const Json directory = {
            {"identity", {{"model_id", "ple-fixture"}, {"weights_id", "exact-bytes"}}},
            {"objects", Json::array({{{"name", "ple"}, {"kind", "resource"},
                {"encoding", "raw-bytes-v1"}, {"offset", 256}, {"bytes", encoded_bytes}}})},
        };
        auto fixture = ninfer::test::artifact_fixture::write_fixture(directory, "ple_direct_io");
        const auto payload_offset = ninfer::test::artifact_fixture::align_up(
            16 + directory.dump().size(), 4096);
        std::vector<std::byte> encoded(encoded_bytes);
        for (std::size_t i = 0; i < encoded.size(); ++i) {
            encoded[i] = std::byte((i * 17 + i / 80) % 251);
        }
        {
            std::fstream file(fixture.path, std::ios::in | std::ios::out | std::ios::binary);
            file.seekp(static_cast<std::streamoff>(payload_offset + 256));
            file.write(reinterpret_cast<const char*>(encoded.data()), encoded.size());
            if (!file) { throw std::runtime_error("fixture write failed"); }
        }
        ninfer::artifact::Reader reader(fixture.path);
        const auto payload = reader.payload("ple");
        PleTableView table;
        table.direct_reader = reader.direct_reader();
        for (auto& shard : table.shards) {
            shard = make_ple_shard_view(payload.data, rows, 160, payload.absolute_offset);
        }
        // Visit every row, including code and scale pieces crossing page boundaries and the
        // partial final page. Alternate shard identities sharing this fixture's encoded region.
        std::vector<std::array<std::int64_t, 16>> indices(rows / 16);
        for (std::size_t row = 0; row < rows; ++row) {
            indices[row / 16][row % 16] = static_cast<std::int64_t>(
                (row % 2) * kPleRowsPerShard + row);
        }
        std::vector<std::byte> expected_codes(rows * 80), expected_scales(rows * 20);
        gather_ple_rows_compressed(table, indices, expected_codes, expected_scales);
        std::vector<std::byte> codes(expected_codes.size()), scales(expected_scales.size());
        const auto check = [&] {
            if (codes != expected_codes || scales != expected_scales) {
                throw std::runtime_error("direct/fallback compressed bytes differ from mmap");
            }
        };
        for (const std::size_t depth : {std::size_t{1}, std::size_t{3}, std::size_t{64}, std::size_t{256}}) {
            PlePageBuffer pages(depth * 4096);
            ninfer::HostWorkerPool workers(2, depth);
            if (table.direct_reader.supported()) {
                const auto result = gather_ple_rows_storage(table, indices, codes, scales,
                    pages, &workers, depth, parse_ple_io_policy("direct", "1"));
                check();
                const auto unique_pages = (payload.absolute_offset + encoded_bytes - 1) / 4096 -
                                          payload.absolute_offset / 4096 + 1;
                if (!result.direct || result.fallback || result.pages != unique_pages) {
                    throw std::runtime_error("direct pages were not deduplicated exactly");
                }
            }
            auto unavailable = table;
            unavailable.direct_reader = {};
            for (const auto mode : {"auto", "direct"}) {
                const auto result = gather_ple_rows_storage(unavailable, indices, codes, scales,
                    pages, &workers, depth, parse_ple_io_policy(mode, "0"));
                check();
                if (result.direct || !result.fallback) {
                    throw std::runtime_error("unavailable direct reads did not report mmap fallback");
                }
            }
            bool strict_failed = false;
            try {
                (void)gather_ple_rows_storage(unavailable, indices, codes, scales,
                    pages, &workers, depth, parse_ple_io_policy("direct", "1"));
            } catch (const std::runtime_error&) { strict_failed = true; }
            if (!strict_failed) { throw std::runtime_error("strict direct unexpectedly fell back"); }

            if (table.direct_reader.supported()) {
                auto truncated = table;
                for (auto& shard : truncated.shards) {
                    shard.code_absolute_offset = (reader.file_bytes() / 4096 + 2) * 4096;
                }
                const auto result = gather_ple_rows_storage(truncated, indices, codes, scales,
                    pages, &workers, depth, parse_ple_io_policy("direct", "0"));
                check();
                if (!result.fallback) { throw std::runtime_error("short direct read did not fall back"); }
                // Immediately reuse the page storage after the failed batch.
                (void)gather_ple_rows_storage(table, indices, codes, scales,
                    pages, &workers, depth, parse_ple_io_policy("direct", "1"));
                check();
            }
            auto invalid = indices;
            invalid[0][0] = rows;
            bool rejected = false;
            try {
                (void)gather_ple_rows_storage(table, invalid, codes, scales,
                    pages, &workers, depth, parse_ple_io_policy("auto", "0"));
            } catch (const std::out_of_range&) { rejected = true; }
            if (!rejected) { throw std::runtime_error("invalid shard row entered fallback"); }
        }
        if (parse_ple_io_policy("", "").mode != PleIoMode::Mmap) {
            throw std::runtime_error("PLE default changed");
        }
        bool rejected_policy = false;
        try { (void)parse_ple_io_policy("auto", "1"); }
        catch (const std::invalid_argument&) { rejected_policy = true; }
        if (!rejected_policy) { throw std::runtime_error("ambiguous strict policy accepted"); }
        std::cout << "PASS: exact PLE compressed pages, crossings, deduplication, fallback and reuse; "
                  << "direct_supported=" << table.direct_reader.supported() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
