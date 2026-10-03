#include "ninfer/engine.h"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// Public options added after the baseline fork; set their neutral values when present.
template<class T> void neutral_reserve(T& options) {
    if constexpr (requires { options.desktop_reserve_bytes; }) options.desktop_reserve_bytes = 0;
}
template<class T> void neutral_repetition(T& sampling) {
    if constexpr (requires { sampling.repetition_penalty; }) sampling.repetition_penalty = 1.0F;
}
// Built unchanged against both public Engine APIs. No model math is replaced.
int main(int argc, char** argv) {
    try {
        if (argc != 5) throw std::runtime_error("usage: probe ARTIFACT CORPUS MODE OUTPUT");
        std::ifstream corpus(argv[2]);
        std::vector<ninfer::TokenId> ids;
        ninfer::TokenId token;
        while (corpus >> token) ids.push_back(token);
        if (ids.size() < 4097) throw std::runtime_error("corpus requires 4097 tokens");
        ids.resize(4097);
        std::string mode = argv[3];
        ninfer::EngineOptions options;
        options.artifact_path = argv[1];
        options.max_context = 8192;
        options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(8192);
        neutral_reserve(options);
        options.context_cache.enabled = false;
        options.prefill_chunk = 1024;
        options.use_cuda_graph = mode != "eager";
        options.kv_cache = ninfer::KvCacheStorage::Int8Group64;
        if (mode == "score") {
            options.purpose = ninfer::EnginePurpose::CausalScoring;
            options.kv_cache = ninfer::KvCacheStorage::Fp8E4M3Row256;
        } else if (mode == "mtp" || mode == "mtp7") {
            options.speculative.backend = ninfer::SpeculativeBackend::Mtp;
            options.speculative.draft_tokens = mode == "mtp7" ? 7 : 3;
            options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
        } else if (mode == "dflash" || mode == "dflash2") {
            options.speculative.backend = mode == "dflash" ? ninfer::SpeculativeBackend::DFlash
                                                            : ninfer::SpeculativeBackend::DFlash2;
            options.speculative.draft_tokens = 3;
            options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
        } else if (mode != "graph" && mode != "eager") throw std::runtime_error("unknown mode");
        ninfer::Engine engine(options);
        std::ofstream out(argv[4]);
        if (!out) throw std::runtime_error("cannot open output");
        out << std::setprecision(9);
        for (size_t i = 0; i < ids.size(); ++i) out << "input " << i << ' ' << ids[i] << '\n';
        if (mode == "score") {
            auto scores = engine.score_tokens(ids, 1);
            if (scores.size() != 4096) throw std::runtime_error("invalid score count");
            for (size_t i = 0; i < scores.size(); ++i) {
                if (!std::isfinite(scores[i])) throw std::runtime_error("nonfinite score");
                out << "score " << i+1 << ' ' << scores[i] << '\n';
            }
        } else {
            std::vector<std::vector<ninfer::TokenId>> prompts;
            for (size_t length : {size_t(31), size_t(1024), size_t(4096)})
                prompts.emplace_back(ids.begin(), ids.begin()+length);
            std::string repetition;
            for (int i = 0; i < 32; ++i) repetition += " alpha beta gamma delta epsilon zeta eta theta";
            prompts.push_back(engine.tokenize_text(repetition));
            for (size_t i = 0; i < prompts.back().size(); ++i)
                out << "repeat_input " << i << ' ' << prompts.back()[i] << '\n';
            for (size_t scenario = 0; scenario < prompts.size(); ++scenario) {
                ninfer::RequestOptions request;
                request.execution.requested_output_tokens = 64;
                request.execution.allow_prefix_reuse = false;
                request.execution.sampling.temperature = 0.0F;
                request.execution.sampling.presence_penalty = 0.0F;
                request.execution.sampling.frequency_penalty = 0.0F;
                neutral_repetition(request.execution.sampling);
                request.stop.include_model_defaults = false;
                auto result = engine.generate(engine.prepare_tokens(prompts[scenario]), request);
                if (result.generated_token_ids.size() != 64) throw std::runtime_error("short generation");
                for (size_t i = 0; i < result.generated_token_ids.size(); ++i)
                    out << "generation " << scenario << ':' << i << ' ' << result.generated_token_ids[i] << '\n';
                std::uint64_t lookup_accepted = 0;
                for (size_t i = 3; i < result.speculative.accepted_per_position.size(); ++i)
                    lookup_accepted += result.speculative.accepted_per_position[i];
                std::cout << "LOOKUP_ACCEPTED scenario=" << scenario << " count=" << lookup_accepted << '\n';
            }
        }
        if (!out) throw std::runtime_error("output write failed");
        std::cout << "PASS exported " << mode << '\n';
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
