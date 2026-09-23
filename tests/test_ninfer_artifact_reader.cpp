#include "artifact/reader.h"
#include "artifact_fixture.h"

#include <nlohmann/json.hpp>

#include <array>
#include <cstdint>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using ninfer::artifact::NumericFormat;
using ninfer::artifact::ObjectDescriptor;
using ninfer::artifact::Reader;
using ninfer::artifact::ResourceDescriptor;
using ninfer::artifact::StorageLayout;
using ninfer::artifact::TensorDescriptor;
using Json = nlohmann::json;
using ninfer::test::artifact_fixture::write_fixture;

Json normative_directory() {
    return {
        {"identity", {{"model_id", "fixture-model"}, {"weights_id", "fixture-weights"}}},
        {"objects", Json::array({
                        {{"name", "resource"},
                         {"kind", "resource"},
                         {"encoding", "raw-bytes-v1"},
                         {"offset", 0},
                         {"bytes", 3}},
                        {{"name", "bf16"},
                         {"kind", "tensor"},
                         {"shape", {2, 3}},
                         {"format", "BF16"},
                         {"layout", "contiguous-le-v1"},
                         {"offset", 256},
                         {"bytes", 12}},
                        {{"name", "fp32_scalar"},
                         {"kind", "tensor"},
                         {"shape", Json::array()},
                         {"format", "FP32"},
                         {"layout", "contiguous-le-v1"},
                         {"offset", 512},
                         {"bytes", 4}},
                        {{"name", "i32"},
                         {"kind", "tensor"},
                         {"shape", {2}},
                         {"format", "I32"},
                         {"layout", "contiguous-le-v1"},
                         {"offset", 768},
                         {"bytes", 8}},
                        {{"name", "q4"},
                         {"kind", "tensor"},
                         {"shape", {1, 1}},
                         {"format", "Q4G64_F16S"},
                         {"layout", "row-split-k128-v1"},
                         {"offset", 1024},
                         {"bytes", 260}},
                        {{"name", "q5"},
                         {"kind", "tensor"},
                         {"shape", {2, 130}},
                         {"format", "Q5G64_F16S"},
                         {"layout", "row-split-k128-v1"},
                         {"offset", 1536},
                         {"bytes", 528}},
                        {{"name", "q6"},
                         {"kind", "tensor"},
                         {"shape", {1, 64}},
                         {"format", "Q6G64_F16S"},
                         {"layout", "row-split-k128-v1"},
                         {"offset", 2304},
                         {"bytes", 516}},
                        {{"name", "w8"},
                         {"kind", "tensor"},
                         {"shape", {1, 33}},
                         {"format", "W8G32_F16S"},
                         {"layout", "row-split-k128-v1"},
                         {"offset", 3072},
                         {"bytes", 264}},
                        {{"name", "fp8_row"},
                         {"kind", "tensor"},
                         {"shape", {2, 4}},
                         {"format", "FP8_E4M3FN_ROW_BF16S"},
                         {"layout", "row-scale-v1"},
                         {"offset", 3584},
                         {"bytes", 260}},
                        {{"name", "fp8_row_f32"},
                         {"kind", "tensor"},
                         {"shape", {2, 4}},
                         {"format", "FP8_E4M3FN_ROW_F32S"},
                         {"layout", "row-scale-f32-v1"},
                         {"offset", 4096},
                         {"bytes", 264}},
                        {{"name", "ple_u4"},
                         {"kind", "tensor"},
                         {"shape", {2, 32}},
                         {"format", "U4Z8G16_F16S"},
                         {"layout", "packed-u4-g16-v1"},
                         {"offset", 4608},
                         {"bytes", 264}},
                        {{"name", "nvfp4_bank"},
                         {"kind", "tensor"},
                         {"shape", {2, 128, 64}},
                         {"format", "NVFP4"},
                         {"layout", "expert-blockscale-k16-m128x4-v1"},
                         {"offset", 5120},
                         {"bytes", 9224}},
                        {{"name", "i64"},
                         {"kind", "tensor"},
                         {"shape", {2}},
                         {"format", "I64"},
                         {"layout", "contiguous-le-v1"},
                         {"offset", 14592},
                         {"bytes", 16}},
                    })},
    };
}

template <typename Function>
void expect_artifact_error(Function&& function, std::string_view label) {
    try {
        function();
    } catch (const ninfer::artifact::ArtifactError&) { return; }
    throw std::runtime_error(std::string(label) + " was accepted");
}

void test_registered_sizes() {
    using ninfer::artifact::tensor_encoded_size;
    constexpr StorageLayout direct       = StorageLayout::ContiguousLeV1;
    constexpr StorageLayout rows         = StorageLayout::RowSplitK128V1;
    constexpr StorageLayout fp8_rows     = StorageLayout::RowScaleV1;
    constexpr StorageLayout fp8_rows_f32 = StorageLayout::RowScaleF32V1;
    constexpr StorageLayout packed_u4    = StorageLayout::PackedU4G16V1;
    constexpr StorageLayout nvfp4_bank   = StorageLayout::ExpertBlockScaleK16M128x4V1;

    const std::array<std::uint64_t, 2> shape_2x3  = {2, 3};
    const std::array<std::uint64_t, 1> shape_2    = {2};
    const std::array<std::uint64_t, 2> q4_shape   = {1, 1};
    const std::array<std::uint64_t, 2> q5_shape   = {2, 130};
    const std::array<std::uint64_t, 2> q6_shape   = {1, 64};
    const std::array<std::uint64_t, 2> w8_shape   = {1, 33};
    const std::array<std::uint64_t, 2> fp8_shape  = {2, 4};
    const std::array<std::uint64_t, 2> ple_shape  = {2, 32};
    const std::array<std::uint64_t, 3> bank_shape = {2, 128, 64};

    if (tensor_encoded_size(direct, NumericFormat::BF16, shape_2x3) != 12 ||
        tensor_encoded_size(direct, NumericFormat::FP32, {}) != 4 ||
        tensor_encoded_size(direct, NumericFormat::I32, shape_2) != 8 ||
        tensor_encoded_size(direct, NumericFormat::I64, shape_2) != 16 ||
        tensor_encoded_size(rows, NumericFormat::Q4G64_F16S, q4_shape) != 260 ||
        tensor_encoded_size(rows, NumericFormat::Q5G64_F16S, q5_shape) != 528 ||
        tensor_encoded_size(rows, NumericFormat::Q6G64_F16S, q6_shape) != 516 ||
        tensor_encoded_size(rows, NumericFormat::W8G32_F16S, w8_shape) != 264 ||
        tensor_encoded_size(fp8_rows, NumericFormat::FP8_E4M3FN_ROW_BF16S, fp8_shape) != 260 ||
        tensor_encoded_size(fp8_rows_f32, NumericFormat::FP8_E4M3FN_ROW_F32S, fp8_shape) != 264 ||
        tensor_encoded_size(packed_u4, NumericFormat::U4Z8G16_F16S, ple_shape) != 264 ||
        tensor_encoded_size(nvfp4_bank, NumericFormat::NVFP4, bank_shape) != 9224) {
        throw std::runtime_error("registered encoded-size calculation is wrong");
    }
    expect_artifact_error([&] { tensor_encoded_size(fp8_rows, NumericFormat::NVFP4, fp8_shape); },
                          "row-scale format mismatch");
    expect_artifact_error(
        [&] { tensor_encoded_size(fp8_rows, NumericFormat::FP8_E4M3FN_ROW_BF16S, shape_2); },
        "row-scale rank mismatch");
    expect_artifact_error(
        [&] { tensor_encoded_size(fp8_rows_f32, NumericFormat::FP8_E4M3FN_ROW_BF16S, fp8_shape); },
        "row-scale-f32 format mismatch");
    expect_artifact_error(
        [&] { tensor_encoded_size(packed_u4, NumericFormat::U4Z8G16_F16S, fp8_shape); },
        "packed-u4 group mismatch");
}

void test_normative_fixture() {
    auto fixture = write_fixture(normative_directory(), "valid");
    Reader reader(fixture.path);
    if (reader.identity().model_id != "fixture-model" ||
        reader.identity().weights_id != "fixture-weights" || reader.objects().size() != 13 ||
        reader.payload_offset() != 4096) {
        throw std::runtime_error("fixture root descriptor mismatch");
    }

    const std::array<std::string_view, 13> expected_names = {
        "resource", "bf16", "fp32_scalar", "i32",         "q4",     "q5",
        "q6",       "w8",   "fp8_row",     "fp8_row_f32", "ple_u4", "nvfp4_bank", "i64",
    };
    for (std::size_t i = 0; i < expected_names.size(); ++i) {
        const auto& object = reader.objects()[i];
        if (ninfer::artifact::object_name(object) != expected_names[i] ||
            reader.find(expected_names[i]) != &object) {
            throw std::runtime_error("fixture name index mismatch");
        }
        const auto payload = reader.payload(object);
        if (payload.absolute_offset !=
                reader.payload_offset() + ninfer::artifact::object_offset(object) ||
            payload.data.size() != ninfer::artifact::object_bytes(object) ||
            payload.data.front() != std::byte(i + 1) || payload.data.back() != std::byte(i + 1)) {
            throw std::runtime_error("fixture payload span mismatch");
        }
    }
    if (reader.find("missing") != nullptr) {
        throw std::runtime_error("missing object unexpectedly resolved");
    }

    const auto* resource   = std::get_if<ResourceDescriptor>(&reader.objects().front());
    const auto* q5         = std::get_if<TensorDescriptor>(reader.find("q5"));
    const auto* fp8        = std::get_if<TensorDescriptor>(reader.find("fp8_row"));
    const auto* fp8_f32    = std::get_if<TensorDescriptor>(reader.find("fp8_row_f32"));
    const auto* ple_u4     = std::get_if<TensorDescriptor>(reader.find("ple_u4"));
    const auto* nvfp4_bank = std::get_if<TensorDescriptor>(reader.find("nvfp4_bank"));
    if (resource == nullptr || q5 == nullptr || q5->shape != std::vector<std::uint64_t>({2, 130}) ||
        q5->format != NumericFormat::Q5G64_F16S || q5->layout != StorageLayout::RowSplitK128V1 ||
        fp8 == nullptr || fp8->shape != std::vector<std::uint64_t>({2, 4}) ||
        fp8->format != NumericFormat::FP8_E4M3FN_ROW_BF16S ||
        fp8->layout != StorageLayout::RowScaleV1 || fp8_f32 == nullptr ||
        fp8_f32->format != NumericFormat::FP8_E4M3FN_ROW_F32S ||
        fp8_f32->layout != StorageLayout::RowScaleF32V1 || ple_u4 == nullptr ||
        ple_u4->format != NumericFormat::U4Z8G16_F16S ||
        ple_u4->layout != StorageLayout::PackedU4G16V1 || nvfp4_bank == nullptr ||
        nvfp4_bank->shape != std::vector<std::uint64_t>({2, 128, 64}) ||
        nvfp4_bank->format != NumericFormat::NVFP4 ||
        nvfp4_bank->layout != StorageLayout::ExpertBlockScaleK16M128x4V1) {
        throw std::runtime_error("fixture object signature mismatch");
    }
}

void test_common_validation() {
    {
        auto directory                   = normative_directory();
        directory["objects"][5]["bytes"] = 527;
        auto fixture                     = write_fixture(directory, "wrong_encoded_size");
        expect_artifact_error([&] { Reader reader(fixture.path); }, "wrong encoded size");
    }
    {
        auto directory                    = normative_directory();
        directory["objects"][1]["offset"] = 257;
        auto fixture                      = write_fixture(directory, "misaligned_offset");
        expect_artifact_error([&] { Reader reader(fixture.path); }, "misaligned offset");
    }
    {
        constexpr std::array<std::uint8_t, 8> invalid_magic = {
            'I', 'N', 'V', 'A', 'L', 'I', 'D', '!',
        };
        auto fixture = write_fixture(normative_directory(), "invalid_magic", invalid_magic);
        expect_artifact_error([&] { Reader reader(fixture.path); }, "invalid magic");
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc > 2) { throw std::runtime_error("usage: ninfer_artifact_reader_test [artifact]"); }
        test_registered_sizes();
        test_normative_fixture();
        test_common_validation();
        if (argc == 2) {
            Reader reader(argv[1]);
            if (reader.objects().empty()) {
                throw std::runtime_error("external artifact has an empty object directory");
            }
            for (const auto& object : reader.objects()) {
                if (reader.payload(object).data.size() != ninfer::artifact::object_bytes(object)) {
                    throw std::runtime_error("external artifact payload span mismatch");
                }
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
