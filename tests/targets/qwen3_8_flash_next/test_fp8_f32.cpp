#include "core/arena.h"
#include "core/device.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

bool cuda_unavailable(cudaError_t error) {
    return error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver;
}

int run_shape(ninfer::DeviceContext& device, std::int32_t rows, std::int32_t columns,
              std::int32_t tokens) {
    const std::size_t code_bytes    = static_cast<std::size_t>(rows) * columns;
    const std::size_t scale_offset  = (code_bytes + 255) & ~std::size_t{255};
    const std::size_t payload_bytes = scale_offset + static_cast<std::size_t>(rows) * sizeof(float);

    ninfer::DeviceBuffer weight_device(payload_bytes);
    weight_device.fill(0x38); // E4M3 1.0 in every code position.
    std::vector<float> scales(static_cast<std::size_t>(rows), 1.0F);
    weight_device.copy_from_host(scales.data(), scales.size() * sizeof(float), scale_offset);

    std::vector<std::uint16_t> input(static_cast<std::size_t>(columns) * tokens, 0);
    for (std::int32_t token = 0; token < tokens; ++token) {
        input[static_cast<std::size_t>(token) * columns] =
            token == 0 ? 0x3F80U : 0x4000U; // BF16 1.0, then 2.0
    }
    ninfer::DeviceBuffer input_device(input.size() * sizeof(std::uint16_t));
    ninfer::DeviceBuffer output_device(static_cast<std::size_t>(rows) * tokens * 2);
    input_device.copy_from_host(input.data(), input.size() * sizeof(std::uint16_t));
    output_device.fill(0xFF);

    ninfer::Weight weight{};
    weight.payload         = weight_device.p;
    weight.payload_bytes   = payload_bytes;
    weight.qdata           = weight_device.p;
    weight.scales          = static_cast<std::byte*>(weight_device.p) + scale_offset;
    weight.qtype           = ninfer::QType::FP8_E4M3FN_ROW_F32S;
    weight.layout          = ninfer::QuantLayout::RowScale;
    weight.scale_dtype     = ninfer::DType::FP32;
    weight.n               = rows;
    weight.k               = columns;
    weight.group           = columns;
    weight.group_size      = static_cast<std::uint32_t>(columns);
    weight.ndim            = 2;
    weight.shape[0]        = rows;
    weight.shape[1]        = columns;
    weight.padded_shape[0] = rows;
    weight.padded_shape[1] = columns;
    weight.scale_ne[0]     = rows;
    weight.scale_nb[0]     = 4;
    weight.scale_nb[1]     = static_cast<std::int64_t>(rows) * 4;
    weight.scale_nb[2]     = weight.scale_nb[1];
    weight.scale_nb[3]     = weight.scale_nb[1];

    ninfer::Tensor input_view(input_device.p, ninfer::DType::BF16, {columns, tokens});
    ninfer::Tensor output_view(output_device.p, ninfer::DType::BF16, {rows, tokens});
    ninfer::WorkspaceArena workspace(256);
    ninfer::ops::linear(input_view, weight, output_view, ninfer::ops::LinearPolicy::A16Only,
                        workspace, device.stream);
    device.synchronize();

    std::vector<std::uint16_t> actual(static_cast<std::size_t>(rows) * tokens);
    output_device.copy_to_host(actual.data(), actual.size() * sizeof(std::uint16_t));
    for (std::int32_t token = 0; token < tokens; ++token) {
        const std::uint16_t expected = token == 0 ? 0x3F80U : 0x4000U;
        for (std::int32_t row = 0; row < rows; ++row) {
            if (actual[static_cast<std::size_t>(token) * rows + row] != expected) {
                std::cerr << "Flash-Next FP8/F32 projection mismatch for [" << rows << ','
                          << columns << "] T=" << tokens << " at row " << row << '\n';
                return 1;
            }
        }
    }
    return 0;
}

} // namespace

int main() {
    int device_count              = 0;
    const cudaError_t count_error = cudaGetDeviceCount(&device_count);
    if (cuda_unavailable(count_error) || device_count == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    CUDA_CHECK(count_error);

    ninfer::DeviceContext device(0);
    int failures = 0;
    failures += run_shape(device, 2'560, 6'144, 1); // decode mainloop
    failures += run_shape(device, 13'312, 2'560, 2);
    failures += run_shape(device, 16'384, 2'560, 2);
    failures += run_shape(device, 2'560, 6'144, 2); // small-T mainloop
    return failures == 0 ? 0 : 1;
}
