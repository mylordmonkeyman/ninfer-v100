#include "serve/http_server.h"

#include "serve/http_transport.h"
#include "serve/openai_chat.h"
#include "serve/openai_common.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

// POST /v1/score: closed-set scoring of many isolated questions against one shared prefix.
//
// Every question is an ordinary one-token chat request made of the shared messages plus one user
// message, so questions never see each other. The first question carries an explicit shared-prefix
// boundary at the end of the shared messages and runs alone; the remaining questions then run
// concurrently against the published prefix, read-only in the context cache so that a large call
// does not displace other conversations' cached state.

namespace ninfer::serve {
namespace {

using Json = nlohmann::ordered_json;

constexpr std::size_t kMaximumScoreQuestions = 256;
constexpr double kLogprobFloor               = -9999.0;

struct ScoreQuestion {
    std::string id;
    Json body; // complete Chat Completions body of this question
};

struct ScoreResult {
    std::optional<GenerationOutcome> outcome;
    std::optional<ApiError> error;
};

[[noreturn]] void score_bad_request(std::string message, std::string param,
                                    std::string code = {}) {
    ApiError error;
    error.message = std::move(message);
    error.param   = std::move(param);
    error.code    = std::move(code);
    throw ApiException(std::move(error));
}

// Marks the end of the shared messages as an explicit shared-prefix boundary.
Json with_prefix_boundary(Json messages) {
    Json& last = messages.back();
    if (last.contains("content") && last["content"].is_string()) {
        last["content"] = Json::array({Json{{"type", "text"}, {"text", last["content"]}}});
    }
    if (last.contains("content") && last["content"].is_array() && !last["content"].empty() &&
        last["content"].back().is_object()) {
        last["content"].back()["prompt_cache_breakpoint"] = Json{{"mode", "explicit"}};
    }
    return messages;
}

double finite_logprob(float value) {
    return std::isfinite(value) ? static_cast<double>(value) : kLogprobFloor;
}

Json entry_json(const TokenLogprobEntry& entry) {
    Json bytes = Json::array();
    for (const char byte : entry.bytes) {
        bytes.push_back(static_cast<int>(static_cast<unsigned char>(byte)));
    }
    const std::string token =
        Json::parse(Json(entry.bytes).dump(-1, ' ', false, Json::error_handler_t::replace))
            .get<std::string>();
    return Json{{"token", token},
                {"token_id", entry.token_id},
                {"logprob", finite_logprob(entry.logprob)},
                {"raw_logprob", finite_logprob(entry.raw_logprob)},
                {"bytes", std::move(bytes)}};
}

Json error_json(const ApiError& error) {
    return Json{{"message", error.message},
                {"type", error.type},
                {"param", error.param.empty() ? Json(nullptr) : Json(error.param)},
                {"code", error.code.empty() ? Json(nullptr) : Json(error.code)}};
}

} // namespace

void HttpServer::handle_score(const httplib::Request& req, httplib::Response& res) {
    std::vector<ScoreQuestion> questions;
    std::vector<OpenAIChatRequest> requests;
    std::string model;
    try {
        const Json body = parse_json_body(req);
        if (!body.is_object()) { score_bad_request("request body must be an object", ""); }
        if (!body.contains("model") || !body.at("model").is_string()) {
            score_bad_request("model must be a string", "model");
        }
        model = body.at("model").get<std::string>();
        validate_openai_model(model, public_model_id_);
        if (!body.contains("messages") || !body.at("messages").is_array() ||
            body.at("messages").empty()) {
            score_bad_request("messages must be a nonempty array holding the shared prefix",
                              "messages");
        }
        if (!body.contains("questions") || !body.at("questions").is_array() ||
            body.at("questions").empty() || body.at("questions").size() > kMaximumScoreQuestions) {
            score_bad_request("questions must be an array of 1 to 256 entries", "questions");
        }
        if (body.contains("stream") && body.at("stream").is_boolean() &&
            body.at("stream").get<bool>()) {
            score_bad_request("score responses are not streamed", "stream");
        }
        const Json default_candidates =
            body.contains("candidates") ? body.at("candidates") : Json(nullptr);

        RequestLimits limits;
        limits.default_max_tokens = options_.default_max_tokens;
        std::size_t index         = 0;
        for (const Json& question : body.at("questions")) {
            const std::string param = "questions[" + std::to_string(index) + "]";
            if (!question.is_object() || !question.contains("content") ||
                !question.at("content").is_string() ||
                question.at("content").get_ref<const std::string&>().empty()) {
                score_bad_request("each question needs a nonempty string content", param);
            }
            ScoreQuestion entry;
            entry.id = question.contains("id") && question.at("id").is_string()
                           ? question.at("id").get<std::string>()
                           : std::to_string(index);
            const Json& candidates =
                question.contains("candidates") ? question.at("candidates") : default_candidates;
            if (!candidates.is_array() || candidates.empty()) {
                score_bad_request("question \"" + entry.id + "\" has no candidates", param);
            }
            // One token per option: the readout is a single position.
            for (const Json& candidate : candidates) {
                if (!candidate.is_string()) { continue; }
                const std::string text = candidate.get<std::string>();
                const std::size_t tokens = text.empty() ? 0 : service_->tokenize_text(text).size();
                if (tokens != 1) {
                    score_bad_request("question \"" + entry.id + "\": candidate \"" + text +
                                          "\" encodes to " + std::to_string(tokens) +
                                          " tokens; each candidate must be one token",
                                      param + ".candidates", "invalid_logprob_candidate");
                }
            }

            Json messages = index == 0 ? with_prefix_boundary(body.at("messages"))
                                       : body.at("messages");
            messages.push_back(Json{{"role", "user"}, {"content", question.at("content")}});
            entry.body = Json{{"model", model},
                              {"messages", std::move(messages)},
                              {"max_tokens", 1},
                              {"temperature", 0},
                              {"logprobs", true},
                              {"top_logprobs", body.value("top_logprobs", 0)},
                              {"logprob_candidates", candidates},
                              // Only the first question writes: it publishes the shared prefix.
                              {"prompt_cache_read_only", index != 0}};
            if (body.contains("chat_template_kwargs")) {
                entry.body["chat_template_kwargs"] = body.at("chat_template_kwargs");
            }
            if (question.contains("response_format")) {
                entry.body["response_format"] = question.at("response_format");
            }
            try {
                requests.push_back(parse_chat_completion_request(entry.body, limits));
            } catch (const ApiException& exception) {
                ApiError error = exception.error();
                error.message  = "question \"" + entry.id + "\": " + error.message;
                throw ApiException(std::move(error));
            }
            questions.push_back(std::move(entry));
            ++index;
        }
    } catch (const ApiException& exception) {
        write_openai_error(res, exception.error());
        return;
    }

    std::vector<ScoreResult> results(questions.size());
    std::shared_ptr<RequestLifetime> lifetime;
    std::mutex lifetime_mutex;
    const auto run_question = [&](std::size_t index) {
        const OpenAIChatRequest& request = requests[index];
        const std::uint64_t req_id       = ++request_seq_;
        const RequestLogMetadata metadata{.model                  = request.model,
                                          .stream                 = false,
                                          .output_tokens_explicit = true};
        PreparedRequest prepared;
        try {
            prepared = service_->prepare(request.generation, GenerationConsumerMode::Aggregate,
                                         [&req] { return client_disconnected(req); });
        } catch (const ApiException& exception) {
            record_request_rejected(make_request_rejection_log_context(
                req_id, "ninfer_score", request.generation, metadata, exception.error()));
            results[index].error = exception.error();
            return;
        } catch (const std::exception& exception) {
            ApiError error;
            error.status         = 500;
            error.type           = "internal_error";
            error.message        = exception.what();
            results[index].error = std::move(error);
            return;
        }
        auto lifecycle = begin_request(make_request_log_context(
            req_id, "ninfer_score", request.generation, metadata, prepared, client_label(req)));
        try {
            results[index].outcome =
                service_->run(prepared, nullptr, [&req] { return client_disconnected(req); });
            lifecycle->done(*results[index].outcome);
        } catch (const ApiException& exception) {
            lifecycle->failure(make_generation_request_failure(exception.error()));
            results[index].error = exception.error();
        } catch (const std::exception& exception) {
            lifecycle->failure(
                make_internal_request_failure(RequestFailurePhase::Generation, exception.what()));
            ApiError error;
            error.status         = 500;
            error.type           = "internal_error";
            error.message        = exception.what();
            results[index].error = std::move(error);
        }
        std::lock_guard lock(lifetime_mutex);
        if (!lifetime) { lifetime = prepared.lifetime; }
    };

    // The first question publishes the prefix; the others must not race it to a cold prefill.
    run_question(0);
    if (questions.size() > 1) {
        std::atomic<std::size_t> next{1};
        const std::size_t workers = std::min<std::size_t>(
            questions.size() - 1, std::max<std::uint32_t>(1U, options_.max_concurrency));
        std::vector<std::thread> pool;
        pool.reserve(workers);
        for (std::size_t worker = 0; worker < workers; ++worker) {
            pool.emplace_back([&] {
                for (std::size_t index = next.fetch_add(1); index < questions.size();
                     index             = next.fetch_add(1)) {
                    run_question(index);
                }
            });
        }
        for (std::thread& thread : pool) { thread.join(); }
    }

    try {
        Json rendered         = Json::array();
        std::int64_t prompt   = 0;
        std::int64_t cached   = 0;
        std::int64_t produced = 0;
        for (std::size_t index = 0; index < questions.size(); ++index) {
            Json entry{{"id", questions[index].id}, {"index", index}};
            const ScoreResult& result = results[index];
            if (result.error || !result.outcome || result.outcome->token_logprobs.empty()) {
                ApiError missing;
                missing.status  = 500;
                missing.type    = "internal_error";
                missing.message = "the question produced no scored position";
                entry["error"]  = error_json(result.error ? *result.error : missing);
                rendered.push_back(std::move(entry));
                continue;
            }
            const GenerationOutcome& outcome    = *result.outcome;
            const TokenLogprobPosition& position = outcome.token_logprobs.front();
            const Json sampled                   = entry_json(position.sampled);
            entry["token"]                       = sampled.at("token");
            entry["token_id"]                    = sampled.at("token_id");
            Json candidates                      = Json::array();
            double inside                        = 0.0;
            for (const TokenLogprobEntry& value : position.candidates) {
                candidates.push_back(entry_json(value));
                if (std::isfinite(value.raw_logprob)) { inside += std::exp(value.raw_logprob); }
            }
            entry["candidate_logprobs"] = std::move(candidates);
            Json top                    = Json::array();
            for (const TokenLogprobEntry& value : position.top) { top.push_back(entry_json(value)); }
            entry["top_logprobs"] = std::move(top);
            // Vocabulary-wide probability the model put outside the candidate list.
            entry["outside_mass"]  = std::clamp(1.0 - inside, 0.0, 1.0);
            entry["cached_tokens"] = outcome.metrics.prefix_cache_hit_tokens;
            prompt += outcome.prompt_tokens;
            cached += outcome.metrics.prefix_cache_hit_tokens;
            produced += outcome.completion_tokens;
            rendered.push_back(std::move(entry));
        }
        Json payload{{"object", "score"},
                     {"model", model},
                     {"results", std::move(rendered)},
                     {"usage", Json{{"prompt_tokens", prompt},
                                    {"cached_tokens", cached},
                                    {"completion_tokens", produced}}}};
        if (lifetime) {
            set_owned_json_content(res, payload.dump(), lifetime);
        } else {
            res.set_content(payload.dump(), "application/json");
        }
    } catch (const std::exception& exception) {
        ApiError error;
        error.status  = 500;
        error.type    = "internal_error";
        error.message = exception.what();
        write_openai_error(res, error);
    }
}

} // namespace ninfer::serve
