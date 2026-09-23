#pragma once

#include "artifact/reader.h"
#include "artifact_fixture.h"
#include "targets/qwen3_8_flash_next/impl/ple_index.h"

#if defined(_WIN32)
#include <windows.h>
#include <winioctl.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::test::flash_next_fixture {

inline artifact_fixture::TemporaryArtifact
create_flash_next_synthetic_artifact(std::string_view suffix = "synthetic_flash_next",
                                     bool mtp_nvfp4 = true) {
    using Json = artifact_fixture::Json;
    using artifact::NumericFormat;
    using artifact::ResourceEncoding;
    using artifact::StorageLayout;

    Json directory;
    directory["identity"] = {
        {"model_id", "qwen3.8-flash-next"},
        {"weights_id", "mixed-nvfp4-fp8-ple-int4"},
    };

    Json objects         = Json::array();
    std::uint64_t offset = 0;

    struct Patch {
        std::uint64_t offset;
        std::vector<std::byte> data;
    };
    std::vector<Patch> patches;

    auto add_resource = [&](std::string_view name) {
        constexpr std::uint64_t bytes = 32;
        objects.push_back({
            {"name", std::string(name)},
            {"kind", "resource"},
            {"encoding", "raw-bytes-v1"},
            {"offset", offset},
            {"bytes", bytes},
        });
        offset = artifact_fixture::align_up(offset + bytes, 256);
    };

    auto add_tensor = [&](std::string_view name, NumericFormat fmt, StorageLayout layout,
                          std::initializer_list<std::uint64_t> shape) {
        const std::vector<std::uint64_t> shape_vec(shape);
        const std::uint64_t bytes = artifact::tensor_encoded_size(layout, fmt, shape_vec);
        objects.push_back({
            {"name", std::string(name)},
            {"kind", "tensor"},
            {"shape", Json(shape_vec)},
            {"format", std::string(artifact::format_name(fmt))},
            {"layout", std::string(artifact::layout_name(layout))},
            {"offset", offset},
            {"bytes", bytes},
        });
        const auto cur_offset = offset;
        offset                = artifact_fixture::align_up(offset + bytes, 256);
        return cur_offset;
    };

    // 1. Frontend resources (6)
    add_resource("frontend/tokenizer.json");
    add_resource("frontend/tokenizer_config.json");
    add_resource("frontend/chat_template.jinja");
    add_resource("frontend/generation_config.json");
    add_resource("frontend/preprocessor_config.json");
    add_resource("frontend/video_preprocessor_config.json");

    constexpr auto kBf16Layout   = StorageLayout::ContiguousLeV1;
    constexpr auto kFp8Layout    = StorageLayout::RowScaleF32V1;
    constexpr auto kExpertLayout = StorageLayout::ExpertBlockScaleK16M128x4V1;
    constexpr auto kPleLayout    = StorageLayout::PackedU4G16V1;

    // 2. Base text endpoints and hyper connection (5)
    add_tensor("text/token_embedding", NumericFormat::BF16, kBf16Layout, {248'320, 2'560});
    add_tensor("text/output_head", NumericFormat::BF16, kBf16Layout, {248'320, 2'560});
    add_tensor("text/hyper_connection/norm", NumericFormat::BF16, kBf16Layout, {10'240});
    add_tensor("text/hyper_connection/input_mix/down", NumericFormat::BF16, kBf16Layout,
               {320, 10'240});
    add_tensor("text/hyper_connection/input_mix/up", NumericFormat::BF16, kBf16Layout,
               {10'240, 320});

    // 3. PLE tensors (137)
    add_tensor("text/layers/1/ple/convolution", NumericFormat::BF16, kBf16Layout, {4, 10'240});
    add_tensor("text/layers/1/ple/key_projection", NumericFormat::BF16, kBf16Layout, {10'240, 2'560});
    add_tensor("text/layers/1/ple/conv_norm", NumericFormat::BF16, kBf16Layout, {10'240});
    add_tensor("text/layers/1/ple/key_norm", NumericFormat::BF16, kBf16Layout, {10'240});
    add_tensor("text/layers/1/ple/query_norm", NumericFormat::BF16, kBf16Layout, {10'240});
    add_tensor("text/layers/1/ple/value_projection", NumericFormat::BF16, kBf16Layout,
               {2'560, 2'560});

    {
        const auto off = add_tensor("text/layers/1/ple/embedding/layer_multipliers",
                                    NumericFormat::I64, kBf16Layout, {3});
        std::vector<std::byte> d(3 * sizeof(std::int64_t));
        std::memcpy(
            d.data(),
            targets::qwen3_8_flash_next::detail::kPleLayerMultipliers.data(),
            d.size());
        patches.push_back({off, std::move(d)});
    }
    {
        const auto off = add_tensor("text/layers/1/ple/embedding/ngram_head_offsets",
                                    NumericFormat::I64, kBf16Layout, {16});
        std::vector<std::byte> d(16 * sizeof(std::int64_t));
        std::memcpy(
            d.data(),
            targets::qwen3_8_flash_next::detail::kPleIndexMetadata.head_offsets.data(),
            d.size());
        patches.push_back({off, std::move(d)});
    }
    {
        const auto off = add_tensor("text/layers/1/ple/embedding/ngram_head_vocab_sizes",
                                    NumericFormat::I64, kBf16Layout, {16});
        std::vector<std::byte> d(16 * sizeof(std::int64_t));
        std::memcpy(
            d.data(),
            targets::qwen3_8_flash_next::detail::kPleIndexMetadata.head_vocab_sizes.data(),
            d.size());
        patches.push_back({off, std::move(d)});
    }

    for (std::size_t s = 0; s < 128; ++s) {
        add_tensor("text/layers/1/ple/embedding/shards/" + std::to_string(s),
                   NumericFormat::U4Z8G16_F16S, kPleLayout, {2'500'012, 160});
    }

    // 4. 48 text layers (48 * 22 = 1056)
    for (std::size_t layer = 0; layer < 48; ++layer) {
        const std::string p = "text/layers/" + std::to_string(layer) + "/";
        add_tensor(p + "attention/hyper_connection/norm", NumericFormat::BF16, kBf16Layout,
                   {10'240});
        add_tensor(p + "attention/hyper_connection/input_mix/down", NumericFormat::BF16, kBf16Layout,
                   {320, 10'240});
        add_tensor(p + "attention/hyper_connection/input_mix/up", NumericFormat::BF16, kBf16Layout,
                   {10'240, 320});
        add_tensor(p + "attention/hyper_connection/block_inject", NumericFormat::BF16, kBf16Layout,
                   {4, 10'240});

        add_tensor(p + "mlp/hyper_connection/norm", NumericFormat::BF16, kBf16Layout, {10'240});
        add_tensor(p + "mlp/hyper_connection/input_mix/down", NumericFormat::BF16, kBf16Layout,
                   {320, 10'240});
        add_tensor(p + "mlp/hyper_connection/input_mix/up", NumericFormat::BF16, kBf16Layout,
                   {10'240, 320});
        add_tensor(p + "mlp/hyper_connection/block_inject", NumericFormat::BF16, kBf16Layout,
                   {4, 10'240});

        add_tensor(p + "mlp/router", NumericFormat::BF16, kBf16Layout, {512, 2'560});
        add_tensor(p + "mlp/shared_expert/down", NumericFormat::BF16, kBf16Layout, {2'560, 640});
        add_tensor(p + "mlp/shared_expert/gate", NumericFormat::BF16, kBf16Layout, {640, 2'560});
        add_tensor(p + "mlp/shared_expert/up", NumericFormat::BF16, kBf16Layout, {640, 2'560});
        add_tensor(p + "mlp/shared_expert_gate", NumericFormat::BF16, kBf16Layout, {1, 2'560});
        add_tensor(p + "mlp/experts/gate_up", NumericFormat::NVFP4, kExpertLayout,
                   {512, 1280, 2560});
        add_tensor(p + "mlp/experts/down", NumericFormat::NVFP4, kExpertLayout,
                   {512, 2560, 640});

        const bool is_full = layer >= 3 && (layer - 3) % 4 == 0;
        if (is_full) {
            add_tensor(p + "attention/indexer/query_key", NumericFormat::BF16, kBf16Layout,
                       {640, 2'560});
            add_tensor(p + "attention/indexer/key_norm", NumericFormat::BF16, kBf16Layout, {128});
            add_tensor(p + "attention/indexer/query_norm", NumericFormat::BF16, kBf16Layout, {128});
            add_tensor(p + "attention/key_norm", NumericFormat::BF16, kBf16Layout, {256});
            add_tensor(p + "attention/query_norm", NumericFormat::BF16, kBf16Layout, {256});
            add_tensor(p + "attention/query_gate_key_value", NumericFormat::FP8_E4M3FN_ROW_F32S,
                       kFp8Layout, {13'312, 2'560});
            add_tensor(p + "attention/output", NumericFormat::FP8_E4M3FN_ROW_F32S, kFp8Layout,
                       {2'560, 6'144});
        } else {
            add_tensor(p + "gdn/a_log", NumericFormat::BF16, kBf16Layout, {48});
            add_tensor(p + "gdn/convolution", NumericFormat::BF16, kBf16Layout, {4, 10'240});
            add_tensor(p + "gdn/dt_bias", NumericFormat::BF16, kBf16Layout, {48});
            add_tensor(p + "gdn/a_b_projection", NumericFormat::BF16, kBf16Layout, {96, 2'560});
            add_tensor(p + "gdn/norm", NumericFormat::BF16, kBf16Layout, {128});
            add_tensor(p + "gdn/query_key_value_z", NumericFormat::FP8_E4M3FN_ROW_F32S,
                       kFp8Layout, {16'384, 2'560});
            add_tensor(p + "gdn/output", NumericFormat::FP8_E4M3FN_ROW_F32S, kFp8Layout,
                       {2'560, 6'144});
        }
    }

    // 5. MTP objects (29)
    add_tensor("mtp/embedding_projection", NumericFormat::BF16, kBf16Layout, {2'560, 2'560});
    add_tensor("mtp/hidden_projection", NumericFormat::BF16, kBf16Layout, {2'560, 2'560});
    add_tensor("mtp/hyper_connection/norm", NumericFormat::BF16, kBf16Layout, {10'240});
    add_tensor("mtp/hyper_connection/input_mix/down", NumericFormat::BF16, kBf16Layout,
               {320, 10'240});
    add_tensor("mtp/hyper_connection/input_mix/up", NumericFormat::BF16, kBf16Layout,
               {10'240, 320});
    add_tensor("mtp/layer/attention/hyper_connection/norm", NumericFormat::BF16, kBf16Layout,
               {10'240});
    add_tensor("mtp/layer/attention/hyper_connection/input_mix/down", NumericFormat::BF16,
               kBf16Layout, {320, 10'240});
    add_tensor("mtp/layer/attention/hyper_connection/input_mix/up", NumericFormat::BF16,
               kBf16Layout, {10'240, 320});
    add_tensor("mtp/layer/attention/hyper_connection/block_inject", NumericFormat::BF16,
               kBf16Layout, {4, 10'240});
    add_tensor("mtp/layer/mlp/hyper_connection/norm", NumericFormat::BF16, kBf16Layout, {10'240});
    add_tensor("mtp/layer/mlp/hyper_connection/input_mix/down", NumericFormat::BF16, kBf16Layout,
               {320, 10'240});
    add_tensor("mtp/layer/mlp/hyper_connection/input_mix/up", NumericFormat::BF16, kBf16Layout,
               {10'240, 320});
    add_tensor("mtp/layer/mlp/hyper_connection/block_inject", NumericFormat::BF16, kBf16Layout,
               {4, 10'240});
    add_tensor("mtp/layer/mlp/router", NumericFormat::BF16, kBf16Layout, {512, 2'560});
    add_tensor("mtp/layer/mlp/shared_expert/down", NumericFormat::BF16, kBf16Layout, {2'560, 640});
    add_tensor("mtp/layer/mlp/shared_expert/gate", NumericFormat::BF16, kBf16Layout, {640, 2'560});
    add_tensor("mtp/layer/mlp/shared_expert/up", NumericFormat::BF16, kBf16Layout, {640, 2'560});
    add_tensor("mtp/layer/mlp/shared_expert_gate", NumericFormat::BF16, kBf16Layout, {1, 2'560});
    if (mtp_nvfp4) {
        add_tensor("mtp/layer/mlp/experts/gate_up", NumericFormat::NVFP4, kExpertLayout,
                   {512, 1280, 2560});
        add_tensor("mtp/layer/mlp/experts/down", NumericFormat::NVFP4, kExpertLayout,
                   {512, 2560, 640});
    } else {
        add_tensor("mtp/layer/mlp/experts/gate_up", NumericFormat::BF16, kBf16Layout,
                   {512, 1280, 2560});
        add_tensor("mtp/layer/mlp/experts/down", NumericFormat::BF16, kBf16Layout,
                   {512, 2560, 640});
    }
    add_tensor("mtp/layer/attention/indexer/query_key", NumericFormat::BF16, kBf16Layout,
               {640, 2'560});
    add_tensor("mtp/layer/attention/indexer/key_norm", NumericFormat::BF16, kBf16Layout, {128});
    add_tensor("mtp/layer/attention/indexer/query_norm", NumericFormat::BF16, kBf16Layout, {128});
    add_tensor("mtp/layer/attention/key_norm", NumericFormat::BF16, kBf16Layout, {256});
    add_tensor("mtp/layer/attention/query_norm", NumericFormat::BF16, kBf16Layout, {256});
    add_tensor("mtp/layer/attention/query_gate_key_value", NumericFormat::BF16, kBf16Layout,
               {13'312, 2'560});
    add_tensor("mtp/layer/attention/output", NumericFormat::BF16, kBf16Layout, {2'560, 6'144});
    add_tensor("mtp/embedding_norm", NumericFormat::BF16, kBf16Layout, {2'560});
    add_tensor("mtp/hidden_norm", NumericFormat::BF16, kBf16Layout, {10'240});

    // 6. Vision objects (333)
    add_tensor("vision/patch_embedding", NumericFormat::BF16, kBf16Layout, {1152, 1536});
    add_tensor("vision/patch_embedding_bias", NumericFormat::BF16, kBf16Layout, {1152});
    add_tensor("vision/position_embedding", NumericFormat::BF16, kBf16Layout, {2304, 1152});
    add_tensor("vision/merger/fc1", NumericFormat::BF16, kBf16Layout, {4608, 4608});
    add_tensor("vision/merger/fc1_bias", NumericFormat::BF16, kBf16Layout, {4608});
    add_tensor("vision/merger/fc2", NumericFormat::BF16, kBf16Layout, {2560, 4608});
    add_tensor("vision/merger/fc2_bias", NumericFormat::BF16, kBf16Layout, {2560});
    add_tensor("vision/merger/norm/weight", NumericFormat::BF16, kBf16Layout, {1152});
    add_tensor("vision/merger/norm/bias", NumericFormat::BF16, kBf16Layout, {1152});
    for (std::size_t layer = 0; layer < 27; ++layer) {
        const std::string p = "vision/layers/" + std::to_string(layer) + "/";
        add_tensor(p + "attention/qkv", NumericFormat::BF16, kBf16Layout, {3456, 1152});
        add_tensor(p + "attention/qkv_bias", NumericFormat::BF16, kBf16Layout, {3456});
        add_tensor(p + "attention/output", NumericFormat::BF16, kBf16Layout, {1152, 1152});
        add_tensor(p + "attention/output_bias", NumericFormat::BF16, kBf16Layout, {1152});
        add_tensor(p + "mlp/fc1", NumericFormat::BF16, kBf16Layout, {4304, 1152});
        add_tensor(p + "mlp/fc1_bias", NumericFormat::BF16, kBf16Layout, {4304});
        add_tensor(p + "mlp/fc2", NumericFormat::BF16, kBf16Layout, {1152, 4304});
        add_tensor(p + "mlp/fc2_bias", NumericFormat::BF16, kBf16Layout, {1152});
        add_tensor(p + "norm1/weight", NumericFormat::BF16, kBf16Layout, {1152});
        add_tensor(p + "norm1/bias", NumericFormat::BF16, kBf16Layout, {1152});
        add_tensor(p + "norm2/weight", NumericFormat::BF16, kBf16Layout, {1152});
        add_tensor(p + "norm2/bias", NumericFormat::BF16, kBf16Layout, {1152});
    }

    directory["objects"] = std::move(objects);

    const std::string json     = directory.dump();
    const auto payload_offset  = artifact_fixture::align_up(16 + json.size(), 4096);
    const auto total_file_size = payload_offset + offset;

    auto path = std::filesystem::temp_directory_path() /
                ("ninfer_artifact_" + std::string(suffix) + ".ninfer");

#if defined(_WIN32)
    HANDLE h = ::CreateFileW(path.wstring().c_str(), GENERIC_READ | GENERIC_WRITE,
                             FILE_SHARE_READ, nullptr, CREATE_ALWAYS,
                             FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        throw std::runtime_error("failed to create sparse fixture file: " + path.string());
    }

    DWORD bytes_returned = 0;
    ::DeviceIoControl(h, FSCTL_SET_SPARSE, nullptr, 0, nullptr, 0, &bytes_returned, nullptr);

    DWORD written = 0;
    ::WriteFile(h, artifact_fixture::kMagic.data(),
                static_cast<DWORD>(artifact_fixture::kMagic.size()), &written, nullptr);
    std::uint64_t json_size = json.size();
    ::WriteFile(h, &json_size, sizeof(json_size), &written, nullptr);
    ::WriteFile(h, json.data(), static_cast<DWORD>(json.size()), &written, nullptr);

    LARGE_INTEGER li;
    li.QuadPart = static_cast<LONGLONG>(total_file_size);
    ::SetFilePointerEx(h, li, nullptr, FILE_BEGIN);
    ::SetEndOfFile(h);

    for (const auto& patch : patches) {
        LARGE_INTEGER patch_offset;
        patch_offset.QuadPart = static_cast<LONGLONG>(payload_offset + patch.offset);
        ::SetFilePointerEx(h, patch_offset, nullptr, FILE_BEGIN);
        ::WriteFile(h, patch.data.data(), static_cast<DWORD>(patch.data.size()), &written,
                    nullptr);
    }
    ::CloseHandle(h);
#else
    int fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        throw std::runtime_error("failed to create fixture file: " + path.string());
    }
    ::write(fd, artifact_fixture::kMagic.data(), artifact_fixture::kMagic.size());
    std::uint64_t json_size = json.size();
    ::write(fd, &json_size, sizeof(json_size));
    ::write(fd, json.data(), json.size());
    ::ftruncate(fd, total_file_size);

    for (const auto& patch : patches) {
        ::lseek(fd, payload_offset + patch.offset, SEEK_SET);
        ::write(fd, patch.data.data(), patch.data.size());
    }
    ::close(fd);
#endif

    return {std::move(path)};
}

} // namespace ninfer::test::flash_next_fixture
