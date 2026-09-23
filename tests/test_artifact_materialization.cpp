#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "artifact/typed_binding.h"
#include "artifact_fixture.h"
#include "core/device.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace {

constexpr std::array<std::byte, 3> kResource = {
    std::byte{1},
    std::byte{1},
    std::byte{1},
};
constexpr std::array<std::byte, 4> kTensor = {
    std::byte{2},
    std::byte{2},
    std::byte{2},
    std::byte{2},
};
constexpr std::array<std::byte, 8> kSecondTensor = {
    std::byte{3}, std::byte{3}, std::byte{3}, std::byte{3},
    std::byte{3}, std::byte{3}, std::byte{3}, std::byte{3},
};
constexpr std::array<std::byte, 4> kMappedTensor = {
    std::byte{5},
    std::byte{5},
    std::byte{5},
    std::byte{5},
};
constexpr std::size_t kFp8TensorBytes    = 260;
constexpr std::size_t kFp8F32TensorBytes = 264;
constexpr std::size_t kTailReadBytes     = 1288;

ninfer::test::artifact_fixture::TemporaryArtifact write_fixture() {
    using Json = ninfer::test::artifact_fixture::Json;
    return ninfer::test::artifact_fixture::write_fixture(
        {
            {"identity", {{"model_id", "fixture-model"}, {"weights_id", "fixture-weights"}}},
            {"objects", Json::array({
                            {{"name", "frontend/test.json"},
                             {"kind", "resource"},
                             {"encoding", "raw-bytes-v1"},
                             {"offset", 0},
                             {"bytes", 3}},
                            {{"name", "weights/test"},
                             {"kind", "tensor"},
                             {"shape", {2}},
                             {"format", "BF16"},
                             {"layout", "contiguous-le-v1"},
                             {"offset", 256},
                             {"bytes", 4}},
                            {{"name", "weights/second"},
                             {"kind", "tensor"},
                             {"shape", {4}},
                             {"format", "BF16"},
                             {"layout", "contiguous-le-v1"},
                             {"offset", 8192},
                             {"bytes", 8}},
                            {{"name", "weights/fp8"},
                             {"kind", "tensor"},
                             {"shape", {2, 4}},
                             {"format", "FP8_E4M3FN_ROW_BF16S"},
                             {"layout", "row-scale-v1"},
                             {"offset", 8448},
                             {"bytes", kFp8TensorBytes}},
                            {{"name", "weights/mapped"},
                             {"kind", "tensor"},
                             {"shape", {2}},
                             {"format", "BF16"},
                             {"layout", "contiguous-le-v1"},
                             {"offset", 8960},
                             {"bytes", kMappedTensor.size()}},
                            {{"name", "weights/fp8_f32"},
                             {"kind", "tensor"},
                             {"shape", {2, 4}},
                             {"format", "FP8_E4M3FN_ROW_F32S"},
                             {"layout", "row-scale-f32-v1"},
                             {"offset", 9216},
                             {"bytes", kFp8F32TensorBytes}},
                        })},
        },
        "materialization");
}

bool cuda_unavailable(cudaError_t error) {
    return error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver;
}

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

struct MappedResult {
    ninfer::artifact::MaterializedArtifact artifact;
    ninfer::artifact::ObjectHandle handle;
};

MappedResult materialize_mapped_tensor(const std::filesystem::path& path,
                                       ninfer::DeviceContext& device) {
    ninfer::artifact::Reader reader(path);
    ninfer::artifact::Binder binder(reader);
    const auto resource = binder.require_resource("frontend/test.json",
                                                  ninfer::artifact::ResourceEncoding::RawBytesV1);
    binder.retain_on_host(resource);
    constexpr std::array<std::uint64_t, 1> tensor_shape = {2};
    const auto tensor =
        binder.require_tensor("weights/test", ninfer::artifact::NumericFormat::BF16,
                              ninfer::artifact::StorageLayout::ContiguousLeV1, tensor_shape);
    binder.materialize_on_device(tensor);
    constexpr std::array<std::uint64_t, 1> second_shape = {4};
    const auto second =
        binder.require_tensor("weights/second", ninfer::artifact::NumericFormat::BF16,
                              ninfer::artifact::StorageLayout::ContiguousLeV1, second_shape);
    binder.validate_only(second);
    constexpr std::array<std::uint64_t, 2> fp8_shape = {2, 4};
    const auto fp8 =
        binder.require_tensor("weights/fp8", ninfer::artifact::NumericFormat::FP8_E4M3FN_ROW_BF16S,
                              ninfer::artifact::StorageLayout::RowScaleV1, fp8_shape);
    binder.validate_only(fp8);
    const auto mapped =
        binder.require_tensor("weights/mapped", ninfer::artifact::NumericFormat::BF16,
                              ninfer::artifact::StorageLayout::ContiguousLeV1, tensor_shape);
    binder.retain_mapped_tensor(mapped);
    const auto fp8_f32 = binder.require_tensor(
        "weights/fp8_f32", ninfer::artifact::NumericFormat::FP8_E4M3FN_ROW_F32S,
        ninfer::artifact::StorageLayout::RowScaleF32V1, fp8_shape);
    binder.validate_only(fp8_f32);
    return {ninfer::artifact::materialize(reader, binder.finish(), device), mapped};
}

} // namespace

int main() {
    try {
        auto fixture = write_fixture();
        ninfer::artifact::Reader reader(fixture.path);
        ninfer::artifact::Binder validation_binder(reader);
        const auto validated_resource = validation_binder.require_resource(
            "frontend/test.json", ninfer::artifact::ResourceEncoding::RawBytesV1);
        validation_binder.retain_on_host(validated_resource);
        constexpr std::array<std::uint64_t, 1> validated_shape = {2};
        const auto validated_only                              = validation_binder.require_tensor(
            "weights/test", ninfer::artifact::NumericFormat::BF16,
            ninfer::artifact::StorageLayout::ContiguousLeV1, validated_shape);
        validation_binder.validate_only(validated_only);
        constexpr std::array<std::uint64_t, 1> retained_shape = {4};
        const auto retained_tensor                            = validation_binder.require_tensor(
            "weights/second", ninfer::artifact::NumericFormat::BF16,
            ninfer::artifact::StorageLayout::ContiguousLeV1, retained_shape);
        validation_binder.materialize_on_device(retained_tensor);
        constexpr std::array<std::uint64_t, 2> fp8_shape = {2, 4};
        const auto validated_fp8                         = validation_binder.require_tensor(
            "weights/fp8", ninfer::artifact::NumericFormat::FP8_E4M3FN_ROW_BF16S,
            ninfer::artifact::StorageLayout::RowScaleV1, fp8_shape);
        validation_binder.validate_only(validated_fp8);
        const auto validated_mapped = validation_binder.require_tensor(
            "weights/mapped", ninfer::artifact::NumericFormat::BF16,
            ninfer::artifact::StorageLayout::ContiguousLeV1, validated_shape);
        validation_binder.retain_mapped_tensor(validated_mapped);
        const auto validated_fp8_f32 = validation_binder.require_tensor(
            "weights/fp8_f32", ninfer::artifact::NumericFormat::FP8_E4M3FN_ROW_F32S,
            ninfer::artifact::StorageLayout::RowScaleF32V1, fp8_shape);
        validation_binder.validate_only(validated_fp8_f32);
        const auto validation_plan = validation_binder.finish();
        require(validation_plan.object_count == 6 && validation_plan.host_objects.size() == 1 &&
                    validation_plan.mapped_tensor_objects.size() == 1 &&
                    validation_plan.device_objects.size() == 1 &&
                    validation_plan.device_capacity_bytes == kSecondTensor.size(),
                "validate-only tensor was included in the materialization plan");

        int device_count              = 0;
        const cudaError_t count_error = cudaGetDeviceCount(&device_count);
        if (cuda_unavailable(count_error)) {
            std::cout << "SKIP: no usable CUDA device\n";
            return 77;
        }
        CUDA_CHECK(count_error);
        if (device_count == 0) {
            std::cout << "SKIP: no CUDA devices\n";
            return 77;
        }

        ninfer::artifact::Binder binder(reader);

        const auto resource = binder.require_resource(
            "frontend/test.json", ninfer::artifact::ResourceEncoding::RawBytesV1);
        binder.retain_on_host(resource);
        constexpr std::array<std::uint64_t, 1> second_shape = {4};
        const auto second =
            binder.require_tensor("weights/second", ninfer::artifact::NumericFormat::BF16,
                                  ninfer::artifact::StorageLayout::ContiguousLeV1, second_shape);
        binder.materialize_on_device(second);

        // Bind in the opposite order from the artifact. Device placement order and file read order
        // are intentionally independent, exercising the direct-I/O scatter path.
        constexpr std::array<std::uint64_t, 1> tensor_shape = {2};
        const auto tensor =
            binder.require_tensor("weights/test", ninfer::artifact::NumericFormat::BF16,
                                  ninfer::artifact::StorageLayout::ContiguousLeV1, tensor_shape);
        binder.materialize_on_device(tensor);

        const auto fp8 = binder.require_tensor(
            "weights/fp8", ninfer::artifact::NumericFormat::FP8_E4M3FN_ROW_BF16S,
            ninfer::artifact::StorageLayout::RowScaleV1, fp8_shape);
        binder.materialize_on_device(fp8);
        const auto mapped =
            binder.require_tensor("weights/mapped", ninfer::artifact::NumericFormat::BF16,
                                  ninfer::artifact::StorageLayout::ContiguousLeV1, tensor_shape);
        binder.retain_mapped_tensor(mapped);
        const auto fp8_f32 = binder.require_tensor(
            "weights/fp8_f32", ninfer::artifact::NumericFormat::FP8_E4M3FN_ROW_F32S,
            ninfer::artifact::StorageLayout::RowScaleF32V1, fp8_shape);
        binder.materialize_on_device(fp8_f32);

        const ninfer::artifact::MaterializationPlan plan = binder.finish();
        require(plan.object_count == 6 && plan.host_objects.size() == 1 &&
                    plan.mapped_tensor_objects.size() == 1 && plan.device_objects.size() == 4 &&
                    plan.device_capacity_bytes == 1288,
                "binder produced the wrong materialization plan");

        ninfer::DeviceContext device(0);
        auto materialized = ninfer::artifact::materialize(reader, plan, device);

        std::array<std::byte, kTensor.size()> copied{};
        CUDA_CHECK(cudaMemcpy(copied.data(), materialized.device_data(tensor), copied.size(),
                              cudaMemcpyDeviceToHost));
        require(copied == kTensor, "device tensor payload differs from the artifact");
        std::array<std::byte, kSecondTensor.size()> second_copied{};
        CUDA_CHECK(cudaMemcpy(second_copied.data(), materialized.device_data(second),
                              second_copied.size(), cudaMemcpyDeviceToHost));
        require(second_copied == kSecondTensor,
                "second device tensor payload differs from the artifact");
        std::array<std::byte, kFp8TensorBytes> fp8_copied{};
        CUDA_CHECK(cudaMemcpy(fp8_copied.data(), materialized.device_data(fp8), fp8_copied.size(),
                              cudaMemcpyDeviceToHost));
        require(std::all_of(fp8_copied.begin(), fp8_copied.end(),
                            [](std::byte value) { return value == std::byte{4}; }),
                "FP8 device tensor payload differs from the artifact");

        const ninfer::Weight fp8_weight = ninfer::artifact::materialized_weight(
            materialized, fp8, ninfer::artifact::NumericFormat::FP8_E4M3FN_ROW_BF16S, 2, 4);
        require(fp8_weight.qtype == ninfer::QType::FP8_E4M3FN_ROW_BF16S &&
                    fp8_weight.layout == ninfer::QuantLayout::RowScale &&
                    fp8_weight.scale_dtype == ninfer::DType::BF16 && fp8_weight.n == 2 &&
                    fp8_weight.k == 4 && fp8_weight.group == 4 && fp8_weight.group_size == 4 &&
                    fp8_weight.qdata == fp8_weight.payload && fp8_weight.qhigh == nullptr &&
                    fp8_weight.scales == static_cast<const std::byte*>(fp8_weight.payload) + 256 &&
                    fp8_weight.payload_bytes == kFp8TensorBytes,
                "materialized FP8 Weight metadata is incomplete");
        const ninfer::Weight fp8_f32_weight = ninfer::artifact::materialized_weight(
            materialized, fp8_f32, ninfer::artifact::NumericFormat::FP8_E4M3FN_ROW_F32S, 2, 4);
        require(fp8_f32_weight.qtype == ninfer::QType::FP8_E4M3FN_ROW_F32S &&
                    fp8_f32_weight.layout == ninfer::QuantLayout::RowScale &&
                    fp8_f32_weight.scale_dtype == ninfer::DType::FP32 &&
                    fp8_f32_weight.scales ==
                        static_cast<const std::byte*>(fp8_f32_weight.payload) + 256 &&
                    fp8_f32_weight.scale_nb[0] == 4 && fp8_f32_weight.scale_nb[1] == 8 &&
                    fp8_f32_weight.payload_bytes == kFp8F32TensorBytes,
                "materialized FP8/FP32 Weight metadata is incomplete");

        const auto retained = materialized.resource_bytes(resource);
        require(std::equal(retained.begin(), retained.end(), kResource.begin(), kResource.end()),
                "retained resource payload differs from the artifact");

        const auto mapped_bytes = materialized.mapped_tensor_bytes(mapped);
        require(std::equal(mapped_bytes.begin(), mapped_bytes.end(), kMappedTensor.begin(),
                           kMappedTensor.end()),
                "mapped tensor payload differs from the artifact");

        // The returned materialization must retain the file mapping independently of Reader.
        MappedResult lifetime     = materialize_mapped_tensor(fixture.path, device);
        const auto lifetime_bytes = lifetime.artifact.mapped_tensor_bytes(lifetime.handle);
        require(std::equal(lifetime_bytes.begin(), lifetime_bytes.end(), kMappedTensor.begin(),
                           kMappedTensor.end()),
                "mapped tensor lifetime ended with the source Reader");

        const auto& stats = materialized.stats();
        require(stats.tensor_count == 5 && stats.mapped_tensor_count == 1 &&
                    stats.mapped_tensor_bytes == kMappedTensor.size() &&
                    stats.resource_count == 1 &&
                    stats.h2d_bytes == kTensor.size() + kSecondTensor.size() + kFp8TensorBytes +
                                           kFp8F32TensorBytes &&
                    stats.retained_resource_bytes == kResource.size() &&
                    stats.file_bytes == kResource.size() +
                                            ninfer::artifact::Reader::direct_io_alignment +
                                            kTailReadBytes,
                "materialization statistics are incomplete");
        require(materialized.device_arena().capacity() == plan.device_capacity_bytes &&
                    materialized.device_arena().used() == plan.device_capacity_bytes,
                "materialized tensor does not own the planned device backing");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
