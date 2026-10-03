#include "ninfer/engine.h"
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

template<class T> void neutral_reserve(T& options) {
    if constexpr (requires { options.desktop_reserve_bytes; }) options.desktop_reserve_bytes = 0;
}
template<class T> void neutral_repetition(T& sampling) {
    if constexpr (requires { sampling.repetition_penalty; }) sampling.repetition_penalty = 1.0F;
}
template<class T> void enable_logprobs(T& execution) {
    if constexpr (requires { execution.logprobs.enabled; execution.logprobs.top; }) {
        execution.logprobs.enabled = true;
        execution.logprobs.top = 2;
    }
}
template<class T> void print_logprobs(const T& result) {
    if constexpr (requires { result.token_logprobs; }) {
        for (std::size_t position = 0; position < result.token_logprobs.size(); ++position) {
            const auto& row = result.token_logprobs[position];
            std::cout << "CHOICE " << position << " forced=" << row.forced
                      << " sampled=" << row.sampled.token << " logprob=" << row.sampled.raw_logprob;
            for (const auto& best : row.top)
                std::cout << " top=" << best.token << ':' << best.raw_logprob;
            std::cout << std::endl;
        }
    }
}

// Same public Engine workload in both forks; isolate the failing 31-token prompt.
int main(int argc, char** argv) {
    try {
        if (argc != 5 && argc != 6) throw std::runtime_error("usage: short-probe ARTIFACT CORPUS KV OUTPUT_TOKENS [fresh-engine|PREFIX_TOKENS]");
        const bool fresh_engine = argc == 6 && std::string(argv[5]) == "fresh-engine";
        const std::size_t prefix_tokens = argc == 6 && !fresh_engine ? std::stoul(argv[5]) : 31;
        std::ifstream corpus(argv[2]);
        std::vector<ninfer::TokenId> ids;
        ninfer::TokenId token;
        while (ids.size() < prefix_tokens && corpus >> token) ids.push_back(token);
        if (prefix_tokens == 0 || ids.size() != prefix_tokens) throw std::runtime_error("short corpus");
        ninfer::EngineOptions options;
        options.artifact_path = argv[1];
        options.max_context = 8192;
        options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(8192);
        neutral_reserve(options);
        options.context_cache.enabled = false;
        options.prefill_chunk = 1024;
        options.use_cuda_graph = false;
        const std::string kv = argv[3];
        if (kv == "int8" || kv == "int8-logprobs") options.kv_cache = ninfer::KvCacheStorage::Int8Group64;
        else if (kv == "fp8") options.kv_cache = ninfer::KvCacheStorage::Fp8E4M3Row256;
        else if (kv == "bf16") options.kv_cache = ninfer::KvCacheStorage::BFloat16;
        else throw std::runtime_error("unknown KV storage");
        auto engine = std::make_unique<ninfer::Engine>(options);
        for (int round = 0; round < 3; ++round) {
            if (round > 0 && fresh_engine) {
                engine.reset();
                engine = std::make_unique<ninfer::Engine>(options);
            }
            ninfer::RequestOptions request;
            request.execution.requested_output_tokens = std::stoul(argv[4]);
            request.execution.allow_prefix_reuse = false;
            request.execution.sampling.temperature = 0;
            request.execution.sampling.presence_penalty = 0;
            request.execution.sampling.frequency_penalty = 0;
            neutral_repetition(request.execution.sampling);
            if (kv == "int8-logprobs") enable_logprobs(request.execution);
            request.stop.include_model_defaults = false;
            auto result = engine->generate(engine->prepare_tokens(ids, false), request);
            std::cout << "ROUND " << round;
            for (auto generated : result.generated_token_ids) std::cout << ' ' << generated;
            std::cout << std::endl;
            print_logprobs(result);
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
