#include "ninfer/engine.h"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

template<class T> void neutral_reserve(T& options) {
    if constexpr (requires { options.desktop_reserve_bytes; }) options.desktop_reserve_bytes = 0;
}
template<class T> void neutral_repetition(T& sampling) {
    if constexpr (requires { sampling.repetition_penalty; }) sampling.repetition_penalty = 1.0F;
}

// Same public Engine workload in both forks; isolate the failing 31-token prompt.
int main(int argc, char** argv) {
    try {
        if (argc != 5) throw std::runtime_error("usage: short-probe ARTIFACT CORPUS KV OUTPUT_TOKENS");
        std::ifstream corpus(argv[2]);
        std::vector<ninfer::TokenId> ids;
        ninfer::TokenId token;
        while (ids.size() < 31 && corpus >> token) ids.push_back(token);
        if (ids.size() != 31) throw std::runtime_error("short corpus");
        ninfer::EngineOptions options;
        options.artifact_path = argv[1];
        options.max_context = 8192;
        options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(8192);
        neutral_reserve(options);
        options.context_cache.enabled = false;
        options.prefill_chunk = 1024;
        options.use_cuda_graph = false;
        const std::string kv = argv[3];
        if (kv == "int8") options.kv_cache = ninfer::KvCacheStorage::Int8Group64;
        else if (kv == "fp8") options.kv_cache = ninfer::KvCacheStorage::Fp8E4M3Row256;
        else if (kv == "bf16") options.kv_cache = ninfer::KvCacheStorage::BFloat16;
        else throw std::runtime_error("unknown KV storage");
        ninfer::Engine engine(options);
        for (int round = 0; round < 3; ++round) {
            ninfer::RequestOptions request;
            request.execution.requested_output_tokens = std::stoul(argv[4]);
            request.execution.allow_prefix_reuse = false;
            request.execution.sampling.temperature = 0;
            request.execution.sampling.presence_penalty = 0;
            request.execution.sampling.frequency_penalty = 0;
            neutral_repetition(request.execution.sampling);
            request.stop.include_model_defaults = false;
            auto result = engine.generate(engine.prepare_tokens(ids, false), request);
            std::cout << "ROUND " << round;
            for (auto generated : result.generated_token_ids) std::cout << ' ' << generated;
            std::cout << std::endl;
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
