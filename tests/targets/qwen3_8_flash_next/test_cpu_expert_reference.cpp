#include "artifact/reader.h"
#include "targets/qwen3_8_flash_next/impl/cpu_expert_reference.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <iostream>
#include <vector>

namespace {

std::vector<std::byte> make_unit_bank(std::int32_t rows, std::int32_t columns) {
    const std::array<std::uint64_t, 3> shape = {
        1, static_cast<std::uint64_t>(rows), static_cast<std::uint64_t>(columns)};
    const auto geometry = ninfer::artifact::block_scale_bank_geometry(
        ninfer::artifact::NumericFormat::NVFP4, shape);
    std::vector<std::byte> payload(geometry.encoded_bytes, std::byte{0});

    // Every row has sixteen unit E2M1 values in the first K16 group and zeros
    // elsewhere. E4M3FN 0x38 is exactly 1.0 and the divisor is exactly 1.0.
    const std::size_t row_code_bytes = static_cast<std::size_t>(columns) / 2;
    for (int row = 0; row < rows; ++row) {
        std::fill_n(payload.begin() + static_cast<std::ptrdiff_t>(
                        static_cast<std::size_t>(row) * row_code_bytes),
                    8, std::byte{0x22});
    }
    std::fill_n(payload.begin() + static_cast<std::ptrdiff_t>(geometry.scale_plane_offset),
                static_cast<std::ptrdiff_t>(geometry.scale_plane_bytes), std::byte{0x38});
    const float divisor = 1.0F;
    std::memcpy(payload.data() + geometry.weight_divisor_offset, &divisor, sizeof(divisor));
    return payload;
}

} // namespace

int main() {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;

    std::vector<std::byte> gate_payload = make_unit_bank(1'280, 2'560);
    std::vector<std::byte> down_payload = make_unit_bank(2'560, 640);
    const Nvfp4ExpertBankView gate_bank =
        make_nvfp4_expert_bank_view(gate_payload.data(), gate_payload.size(), 1, 1'280, 2'560);
    const Nvfp4ExpertBankView down_bank =
        make_nvfp4_expert_bank_view(down_payload.data(), down_payload.size(), 1, 2'560, 640);
    const HostNvfp4ExpertPairView expert{
        .gate_up = gate_bank.expert(0),
        .down = down_bank.expert(0),
    };

    std::array<std::uint16_t, kFlashNextExpertHidden> input{};
    input.fill(0x3F80U); // BF16 1.0
    std::array<float, kFlashNextExpertHidden> output{};
    CpuNvfp4ExpertReferenceScratch scratch{};
    flash_next_cpu_nvfp4_expert_pair_reference(expert, input, output, scratch);

    if (flash_next_cpu_nvfp4_avx2_available()) {
        std::array<float, kFlashNextExpertHidden> avx2_output{};
        CpuNvfp4ExpertReferenceScratch avx2_scratch{};
        flash_next_cpu_nvfp4_expert_pair_avx2(
            expert, input, avx2_output, avx2_scratch);
        for (std::size_t row = 0; row < output.size(); ++row) {
            const float tolerance =
                1.0e-5F * std::max(1.0F, std::abs(output[row]));
            if (std::abs(avx2_output[row] - output[row]) > tolerance) {
                std::cerr << "CPU NVFP4 AVX2 row " << row
                          << " differs from scalar reference: "
                          << avx2_output[row] << " vs " << output[row] << '\n';
                return 1;
            }
        }
    }

    for (std::size_t row = 0; row < output.size(); ++row) {
        if (output[row] != 4096.0F) {
            std::cerr << "CPU NVFP4 reference row " << row << " produced "
                      << output[row] << ", expected 4096\n";
            return 1;
        }
    }
    std::cout << "PASS: CPU NVFP4 expert-pair scalar/AVX2 parity\n";
    return 0;
}
