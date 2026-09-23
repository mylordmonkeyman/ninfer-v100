#include "synthetic_fixture.h"

#include "artifact/binder.h"
#include "artifact/reader.h"
#include "targets/qwen3_8_flash_next/impl/load/bindings.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>

namespace {

using namespace ninfer::targets::qwen3_8_flash_next::detail;

std::filesystem::path real_artifact_path() {
    if (const char* value = std::getenv("NINFER_QWEN3_8_FLASH_NEXT_WEIGHTS");
        value != nullptr && *value != '\0') {
        return value;
    }
    return "" /* real-artifact cases run only when NINFER_WEIGHTS is set explicitly */;
}

int test_synthetic_fixture_load_plans() {
    try {
        const auto fixture =
            ninfer::test::flash_next_fixture::create_flash_next_synthetic_artifact(
                "load_plan_synthetic");
        const ninfer::artifact::Reader reader(fixture.path);

        // Case 1: Full features (vision=true, mtp=true)
        {
            ninfer::artifact::Binder binder(reader);
            const auto plan = bind_artifact(binder, {.vision = true, .mtp = true});
            const auto& m   = plan.materialization;

            if (m.object_count != 1'566 || m.device_objects.size() != 1'429 ||
                m.mapped_tensor_objects.size() != 131 || m.host_objects.size() != 6 ||
                m.device_capacity_bytes != 77'667'534'336ULL) {
                std::cerr << "Synthetic full binding mismatch: objects=" << m.object_count
                          << " device=" << m.device_objects.size()
                          << " mapped=" << m.mapped_tensor_objects.size()
                          << " host=" << m.host_objects.size()
                          << " device_bytes=" << m.device_capacity_bytes << "\n";
                return 1;
            }
            if (!plan.bindings.features.vision || !plan.bindings.features.mtp) {
                std::cerr << "Synthetic full features mismatch\n";
                return 1;
            }
        }

        // Case 2: Vision enabled, MTP disabled (vision=true, mtp=false)
        {
            ninfer::artifact::Binder binder(reader);
            const auto plan = bind_artifact(binder, {.vision = true, .mtp = false});
            const auto& m   = plan.materialization;

            if (m.object_count != 1'566 || m.device_objects.size() != 1'400 ||
                m.mapped_tensor_objects.size() != 131 || m.host_objects.size() != 6 ||
                m.device_capacity_bytes != 76'070'815'744ULL) {
                std::cerr << "Synthetic vision-only binding mismatch: objects=" << m.object_count
                          << " device=" << m.device_objects.size()
                          << " mapped=" << m.mapped_tensor_objects.size()
                          << " host=" << m.host_objects.size()
                          << " device_bytes=" << m.device_capacity_bytes << "\n";
                return 1;
            }
            if (!plan.bindings.features.vision || plan.bindings.features.mtp) {
                std::cerr << "Synthetic vision-only features mismatch\n";
                return 1;
            }
        }

        // Case 3: Text only, MTP and Vision disabled (vision=false, mtp=false)
        {
            ninfer::artifact::Binder binder(reader);
            const auto plan = bind_artifact(binder, {.vision = false, .mtp = false});
            const auto& m   = plan.materialization;

            if (m.object_count != 1'566 || m.device_objects.size() != 1'067 ||
                m.mapped_tensor_objects.size() != 131 || m.host_objects.size() != 6) {
                std::cerr << "Synthetic text-only binding mismatch: objects=" << m.object_count
                          << " device=" << m.device_objects.size()
                          << " mapped=" << m.mapped_tensor_objects.size()
                          << " host=" << m.host_objects.size() << "\n";
                return 1;
            }
            if (plan.bindings.features.vision || plan.bindings.features.mtp) {
                std::cerr << "Synthetic text-only features mismatch\n";
                return 1;
            }
        }

        // Case 4: Text + MTP, Vision disabled (vision=false, mtp=true)
        {
            ninfer::artifact::Binder binder(reader);
            const auto plan = bind_artifact(binder, {.vision = false, .mtp = true});
            const auto& m   = plan.materialization;

            if (m.object_count != 1'566 || m.device_objects.size() != 1'096 ||
                m.mapped_tensor_objects.size() != 131 || m.host_objects.size() != 6) {
                std::cerr << "Synthetic text+mtp binding mismatch: objects=" << m.object_count
                          << " device=" << m.device_objects.size()
                          << " mapped=" << m.mapped_tensor_objects.size()
                          << " host=" << m.host_objects.size() << "\n";
                return 1;
            }
            if (plan.bindings.features.vision || !plan.bindings.features.mtp) {
                std::cerr << "Synthetic text+mtp features mismatch\n";
                return 1;
            }
        }

        // Case 5: Legacy BF16 MTP fixture with full features
        {
            const auto legacy_fixture =
                ninfer::test::flash_next_fixture::create_flash_next_synthetic_artifact(
                    "load_plan_legacy_bf16", false);
            const ninfer::artifact::Reader legacy_reader(legacy_fixture.path);
            ninfer::artifact::Binder binder(legacy_reader);
            const auto plan = bind_artifact(binder, {.vision = true, .mtp = true});
            const auto& m   = plan.materialization;

            if (m.object_count != 1'566 || m.device_objects.size() != 1'427 ||
                m.mapped_tensor_objects.size() != 133 || m.host_objects.size() != 6 ||
                m.device_capacity_bytes != 76'251'952'640ULL) {
                std::cerr << "Synthetic legacy BF16 binding mismatch: objects=" << m.object_count
                          << " device=" << m.device_objects.size()
                          << " mapped=" << m.mapped_tensor_objects.size()
                          << " host=" << m.host_objects.size()
                          << " device_bytes=" << m.device_capacity_bytes << "\n";
                return 1;
            }
            if (!plan.bindings.features.vision || !plan.bindings.features.mtp ||
                plan.bindings.mtp.moe.experts_nvfp4) {
                std::cerr << "Synthetic legacy features mismatch\n";
                return 1;
            }
        }

        // Case 6: Phase-10 host-backed main experts preserve exact compact payloads
        // in the artifact mapping instead of the device arena.
        {
            ninfer::artifact::Binder resident_binder(reader);
            const auto resident =
                bind_artifact(resident_binder, {.vision = false, .mtp = false,
                                                .host_backed_experts = false});
            ninfer::artifact::Binder host_binder(reader);
            const auto host =
                bind_artifact(host_binder, {.vision = false, .mtp = false,
                                            .host_backed_experts = true});

            constexpr std::size_t kMainExpertObjects = 48ULL * 2ULL;
            constexpr std::uint64_t kMainExpertBytes =
                48ULL * 1'415'581'696ULL;
            if (resident.materialization.device_objects.size() !=
                    host.materialization.device_objects.size() + kMainExpertObjects ||
                resident.materialization.mapped_tensor_objects.size() + kMainExpertObjects !=
                    host.materialization.mapped_tensor_objects.size() ||
                resident.materialization.device_capacity_bytes <
                    host.materialization.device_capacity_bytes ||
                resident.materialization.device_capacity_bytes -
                        host.materialization.device_capacity_bytes !=
                    kMainExpertBytes) {
                std::cerr << "Synthetic host-backed expert placement mismatch: resident_device="
                          << resident.materialization.device_objects.size()
                          << " host_device=" << host.materialization.device_objects.size()
                          << " resident_mapped="
                          << resident.materialization.mapped_tensor_objects.size()
                          << " host_mapped="
                          << host.materialization.mapped_tensor_objects.size()
                          << " saved_bytes="
                          << (resident.materialization.device_capacity_bytes -
                              host.materialization.device_capacity_bytes)
                          << "\n";
                return 1;
            }
            for (const auto& layer : host.bindings.text_layers) {
                if (!layer.moe.experts_nvfp4 || !layer.moe.experts_host_mapped) {
                    std::cerr << "Synthetic host-backed MoE binding lost NVFP4/mapped contract\n";
                    return 1;
                }
            }
        }

        std::cout << "PASS: test_synthetic_fixture_load_plans\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR in test_synthetic_fixture_load_plans: " << e.what() << "\n";
        return 1;
    }
}

int test_real_artifact_if_available() {
    try {
        const std::filesystem::path path = real_artifact_path();
        std::error_code ec;
        if (!std::filesystem::is_regular_file(path, ec)) {
            std::cout << "SKIP: Real Flash-Next artifact test (not present at " << path.string()
                      << ")\n";
            return 0;
        }

        ninfer::artifact::Reader reader(path);
        ninfer::artifact::Binder binder(reader);
        const auto plan = bind_artifact(binder, {.vision = true, .mtp = true});

        const auto& m = plan.materialization;
        const bool is_nvfp4 = plan.bindings.mtp.moe.experts_nvfp4;
        const std::size_t expected_dev_objs = is_nvfp4 ? 1'429 : 1'427;
        const std::size_t expected_mapped   = is_nvfp4 ? 131 : 133;
        const std::uint64_t expected_bytes  = is_nvfp4 ? 77'667'534'336ULL : 76'251'952'640ULL;
        if (m.object_count != 1'566 || m.device_objects.size() != expected_dev_objs ||
            m.mapped_tensor_objects.size() != expected_mapped || m.host_objects.size() != 6 ||
            m.device_capacity_bytes != expected_bytes) {
            std::cerr << "unexpected Flash-Next load plan: objects=" << m.object_count
                      << " device=" << m.device_objects.size()
                      << " mapped=" << m.mapped_tensor_objects.size()
                      << " host=" << m.host_objects.size()
                      << " device_bytes=" << m.device_capacity_bytes << '\n';
            return 1;
        }

        const auto& bindings = plan.bindings;
        if (bindings.ple.shards.size() != 128 || bindings.text_layers.size() != 48 ||
            bindings.vision.layers.size() != 27 || !bindings.features.vision ||
            !bindings.features.mtp) {
            std::cerr << "Flash-Next semantic binding topology is incomplete\n";
            return 1;
        }

        std::cout << "PASS: test_real_artifact_load_plan\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR in test_real_artifact_if_available: " << e.what() << '\n';
        return 1;
    }
}

} // namespace

int main() {
    if (test_synthetic_fixture_load_plans() != 0) return 1;
    if (test_real_artifact_if_available() != 0) return 1;

    std::cout << "PASS: test_load_plan\n";
    return 0;
}
