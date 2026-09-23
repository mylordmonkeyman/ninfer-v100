#include <ninfer/targets/qwen3_8_flash_next/package.h>
#include "artifact/binder.h"
#include "artifact/reader.h"
#include "artifact_fixture.h"
#include "targets/registry.h"

#include <iostream>
#include <stdexcept>

namespace {

using Json = nlohmann::json;
using ninfer::test::artifact_fixture::write_fixture;
using Package = ninfer::targets::qwen3_8_flash_next::Package;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

int test_package_identity() {
    int failures = 0;

    failures += check(Package::model_id == "qwen3.8-flash-next",
                      "Package::model_id must be 'qwen3.8-flash-next'");
    failures += check(Package::target_key == "qwen3_8_flash_next",
                      "Package::target_key must be 'qwen3_8_flash_next'");

    ninfer::artifact::ArtifactIdentity valid_id{
        .model_id   = "qwen3.8-flash-next",
        .weights_id = "mixed-nvfp4-fp8-ple-int4",
    };
    try {
        auto profile = Package::resolve_weights(valid_id);
        failures += check(profile == Package::WeightsProfile::MixedNvfp4Fp8PleInt4,
                          "resolve_weights returned unexpected profile");
    } catch (const std::exception& ex) {
        std::cerr << "Unexpected failure resolving weights: " << ex.what() << '\n';
        failures += 1;
    }

    ninfer::artifact::ArtifactIdentity invalid_model{
        .model_id   = "qwen3.6-27b",
        .weights_id = "mixed-nvfp4-fp8-ple-int4",
    };
    try {
        (void)Package::resolve_weights(invalid_model);
        std::cerr << "Expected exception on invalid model_id but none thrown\n";
        failures += 1;
    } catch (const std::exception&) {
        // Expected
    }

    ninfer::artifact::ArtifactIdentity invalid_weights{
        .model_id   = "qwen3.8-flash-next",
        .weights_id = "groupwise-int",
    };
    try {
        (void)Package::resolve_weights(invalid_weights);
        std::cerr << "Expected exception on invalid weights_id but none thrown\n";
        failures += 1;
    } catch (const std::exception&) {
        // Expected
    }

    return failures;
}

int test_sampling_defaults() {
    int failures = 0;

    try {
        const auto defaults = Package::sampling_defaults(Package::model_id);
        failures += check(defaults.thinking.temperature == 1.0F, "thinking temperature mismatch");
        failures += check(defaults.thinking.top_p == 0.95F, "thinking top_p mismatch");
        failures += check(defaults.non_thinking.temperature == 0.7F, "non-thinking temperature mismatch");
        failures += check(defaults.non_thinking.top_p == 0.80F, "non-thinking top_p mismatch");
        failures += check(defaults.non_thinking.presence_penalty == 0.0F, "non-thinking presence_penalty mismatch");
    } catch (const std::exception& ex) {
        std::cerr << "Unexpected failure querying sampling_defaults: " << ex.what() << '\n';
        failures += 1;
    }

    try {
        (void)Package::sampling_defaults("unknown-model");
        std::cerr << "Expected exception for unknown model sampling defaults\n";
        failures += 1;
    } catch (const std::exception&) {
        // Expected
    }

    return failures;
}

int test_unregistered_identity_rejection() {
    int failures = 0;

    Json dir = {
        {"identity", {{"model_id", "unregistered-model"}, {"weights_id", "unregistered-weights"}}},
        {"objects", Json::array({
                        {{"name", "dummy"},
                         {"kind", "resource"},
                         {"encoding", "raw-bytes-v1"},
                         {"offset", 0},
                         {"bytes", 4}},
                    })},
    };
    auto fixture = write_fixture(dir, "unregistered_test.ninfer");

    ninfer::EngineOptions options{};
    options.artifact_path = fixture.path;
    options.max_context   = 512;
    options.kv_capacity   = ninfer::KvCapacityPolicy::explicit_capacity(512);

    ninfer::DeviceContext dummy_device{};
    try {
        (void)ninfer::targets::construct_target(options, dummy_device);
        std::cerr << "Expected construct_target to reject unregistered identity\n";
        failures += 1;
    } catch (const std::exception& ex) {
        const std::string msg = ex.what();
        failures += check(msg.find("has no registered target for this device") != std::string::npos,
                          msg.c_str());
    }

    return failures;
}

int test_unsupported_speculation_rejection() {
    Json dir = {
        {"identity", {{"model_id", "qwen3.8-flash-next"},
                      {"weights_id", "mixed-nvfp4-fp8-ple-int4"}}},
        {"objects", Json::array({{{"name", "dummy"}, {"kind", "resource"},
                                  {"encoding", "raw-bytes-v1"}, {"offset", 0}, {"bytes", 4}}})},
    };
    const auto fixture = write_fixture(dir, "unsupported_flash_speculation.ninfer");
    const ninfer::artifact::Reader reader(fixture.path);
    int failures = 0;
    for (auto backend : {ninfer::SpeculativeBackend::DFlash, ninfer::SpeculativeBackend::DFlash2}) {
        ninfer::artifact::Binder binder(reader);
        ninfer::EngineOptions options;
        options.speculative = {.backend = backend, .draft_tokens = 7};
        try {
            (void)Package::plan_load(binder, options, Package::WeightsProfile::MixedNvfp4Fp8PleInt4);
            failures += check(false, "Flash-Next silently accepted an unsupported speculative backend");
        } catch (const std::invalid_argument& error) {
            failures += check(std::string(error.what()).find("ordinary decoding and MTP") != std::string::npos,
                              error.what());
        }
    }
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_package_identity();
    failures += test_sampling_defaults();
    failures += test_unregistered_identity_rejection();
    failures += test_unsupported_speculation_rejection();

    if (failures == 0) {
        std::cout << "All registry tests passed cleanly.\n";
        return 0;
    }
    std::cerr << failures << " registry test(s) failed.\n";
    return 1;
}
