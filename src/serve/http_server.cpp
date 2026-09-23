#include "serve/http_server.h"

#include "serve/anthropic_messages.h"
#include "serve/device_snapshot_cache.h"
#include "serve/http_transport.h"
#include "serve/openai_common.h"
#include "serve/request_log.h"

#include "core/device_memory.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

namespace ninfer::serve {
namespace {

void write_exception(httplib::Response& res, const std::exception& ex) {
    ApiError error;
    error.status  = 500;
    error.type    = "internal_error";
    error.message = ex.what();
    write_openai_error(res, error);
}

bool is_anthropic_path(std::string_view path) { return path.starts_with("/v1/messages"); }

bool is_openai_path(std::string_view path) {
    return path.starts_with("/v1/") && !is_anthropic_path(path);
}

void ensure_openai_request_id(const httplib::Request& request, httplib::Response& response) {
    if (is_openai_path(request.path) && !response.has_header("x-request-id")) {
        response.set_header("x-request-id", new_openai_request_id());
    }
}

ThroughputReport make_throughput_report(const ninfer::RuntimeStats& previous,
                                        const ninfer::RuntimeStats& current,
                                        double interval_seconds) {
    return ThroughputReport{
        .interval_seconds = interval_seconds,
        .computed_prefill_tokens =
            current.computed_prefill_tokens - previous.computed_prefill_tokens,
        .committed_decode_tokens =
            current.committed_decode_tokens - previous.committed_decode_tokens,
        .decode_rounds     = current.decode_rounds - previous.decode_rounds,
        .decode_row_rounds = current.decode_row_rounds - previous.decode_row_rounds,
        .previous          = previous,
        .current           = current,
    };
}

bool report_has_activity(const ThroughputReport& report) {
    return report.computed_prefill_tokens != 0 || report.committed_decode_tokens != 0 ||
           report.decode_rounds != 0 || report.current.running_requests != 0 ||
           report.current.waiting_requests != 0 || report.current.materializing_requests != 0 ||
           report.current.capture_pending_requests != 0 ||
           report.current.terminal_pending_requests != 0 ||
           report.current.active_captures_completed != report.previous.active_captures_completed ||
           report.current.active_captures_aborted != report.previous.active_captures_aborted ||
           report.current.root_selections != report.previous.root_selections ||
           report.current.private_endpoint_selections !=
               report.previous.private_endpoint_selections ||
           report.current.private_turn_closure_selections !=
               report.previous.private_turn_closure_selections ||
           report.current.private_response_replay_selections !=
               report.previous.private_response_replay_selections ||
           report.current.private_long_anchor_selections !=
               report.previous.private_long_anchor_selections ||
           report.current.shared_stable_prefix_selections !=
               report.previous.shared_stable_prefix_selections ||
           report.current.state_moves != report.previous.state_moves ||
           report.current.state_forks != report.previous.state_forks ||
           report.current.state_restores != report.previous.state_restores ||
           report.current.state_d2h_count != report.previous.state_d2h_count ||
           report.current.state_h2d_count != report.previous.state_h2d_count ||
           report.current.state_d2d_count != report.previous.state_d2d_count ||
           report.current.main_kv_d2h_pages != report.previous.main_kv_d2h_pages ||
           report.current.main_kv_h2d_pages != report.previous.main_kv_h2d_pages ||
           report.current.main_kv_d2d_pages != report.previous.main_kv_d2d_pages ||
           report.current.backend_kv_d2h_pages != report.previous.backend_kv_d2h_pages ||
           report.current.backend_kv_h2d_pages != report.previous.backend_kv_h2d_pages ||
           report.current.backend_kv_d2d_pages != report.previous.backend_kv_d2d_pages ||
           report.current.pressure_spill_pages != report.previous.pressure_spill_pages ||
           report.current.partial_tail_cow_pages != report.previous.partial_tail_cow_pages ||
           report.current.pressure_private_owners_degraded !=
               report.previous.pressure_private_owners_degraded ||
           report.current.pressure_private_owners_evicted !=
               report.previous.pressure_private_owners_evicted ||
           report.current.pressure_shared_owners_degraded !=
               report.previous.pressure_shared_owners_degraded ||
           report.current.pressure_shared_owners_evicted !=
               report.previous.pressure_shared_owners_evicted ||
           report.current.pressure_checkpoints_dropped !=
               report.previous.pressure_checkpoints_dropped ||
           report.current.pressure_searches != report.previous.pressure_searches ||
           report.current.pressure_search_budget_exhaustions !=
               report.previous.pressure_search_budget_exhaustions ||
           report.current.pressure_maximal_fallback_selections !=
               report.previous.pressure_maximal_fallback_selections ||
           report.current.historical_fork_hits != report.previous.historical_fork_hits ||
           report.current.device_state_occupied_slots !=
               report.previous.device_state_occupied_slots ||
           report.current.host_state_occupied_slots != report.previous.host_state_occupied_slots ||
           report.current.device_main_kv_occupied_pages !=
               report.previous.device_main_kv_occupied_pages ||
           report.current.device_backend_kv_occupied_pages !=
               report.previous.device_backend_kv_occupied_pages ||
           report.current.host_kv_occupied_bytes != report.previous.host_kv_occupied_bytes ||
           report.current.shared_active_references != report.previous.shared_active_references ||
           report.current.host_work.engine_boundary_ns !=
               report.previous.host_work.engine_boundary_ns ||
           report.current.host_work.program_submit_ns !=
               report.previous.host_work.program_submit_ns ||
           report.current.host_work.program_post_ns != report.previous.host_work.program_post_ns ||
           report.current.host_work.engine_commit_output_ns !=
               report.previous.host_work.engine_commit_output_ns ||
           report.current.host_work.engine_maintenance_ns !=
               report.previous.host_work.engine_maintenance_ns ||
           report.current.host_work.device_wait_ns != report.previous.host_work.device_wait_ns;
}

const char* endpoint_name(std::string_view path) noexcept {
    if (path == "/v1/chat/completions") { return "openai_chat_completions"; }
    if (path == "/v1/score") { return "ninfer_score"; }
    if (path == "/v1/responses") { return "openai_responses"; }
    if (path == "/v1/responses/input_tokens") { return "openai_responses_input_tokens"; }
    if (path == "/v1/messages") { return "anthropic_messages"; }
    if (path == "/v1/messages/count_tokens") { return "anthropic_count_tokens"; }
    return "http_route";
}

std::string response_request_id(const httplib::Response& response) {
    if (response.has_header("x-request-id")) { return response.get_header_value("x-request-id"); }
    if (response.has_header("request-id")) { return response.get_header_value("request-id"); }
    return {};
}

} // namespace

void write_openai_error(httplib::Response& response, const ApiError& error) {
    response.status = error.status;
    response.set_content(make_error_body(error), "application/json");
}

void write_anthropic_error(httplib::Response& response, const ApiError& api_error,
                           const std::string& request_id) {
    const ApiError error = normalize_anthropic_error(api_error);
    response.status      = error.status;
    response.headers.erase("request-id");
    response.set_header("request-id", request_id);
    response.set_content(make_anthropic_error_body(error, request_id), "application/json");
}

httplib::Server::HandlerResponse handle_unrendered_http_error(const ServeOptions& options,
                                                              const httplib::Request& request,
                                                              httplib::Response& response) {
    ensure_openai_request_id(request, response);
    if (!response.body.empty()) { return httplib::Server::HandlerResponse::Unhandled; }

    ApiError error;
    if (response.status == 413) {
        error.status  = 413;
        error.type    = "invalid_request_error";
        error.code    = "request_too_large";
        error.message = "request body exceeds the configured payload limit of " +
                        std::to_string(options.max_request_bytes) + " bytes";
    } else if (response.status == 404 && request.path.rfind("/v1/messages", 0) == 0) {
        error.status  = 404;
        error.code    = "not_found";
        error.message = "requested Anthropic resource was not found";
    } else {
        return httplib::Server::HandlerResponse::Unhandled;
    }
    if (request.path.rfind("/v1/messages", 0) == 0) {
        write_anthropic_error(response, error, new_anthropic_request_id());
    } else {
        write_openai_error(response, error);
    }
    return httplib::Server::HandlerResponse::Handled;
}

bool matches_bearer_credential(std::string_view authorization, std::string_view api_key) noexcept {
    if (api_key.empty()) { return false; }
    const auto is_whitespace = [](char value) { return value == ' ' || value == '\t'; };
    const auto ascii_equal   = [](char lhs, char rhs) {
        if (lhs >= 'A' && lhs <= 'Z') { lhs = static_cast<char>(lhs - 'A' + 'a'); }
        if (rhs >= 'A' && rhs <= 'Z') { rhs = static_cast<char>(rhs - 'A' + 'a'); }
        return lhs == rhs;
    };

    std::size_t position = 0;
    while (position < authorization.size() && is_whitespace(authorization[position])) {
        ++position;
    }
    constexpr std::string_view scheme = "Bearer";
    if (authorization.size() - position < scheme.size()) { return false; }
    for (std::size_t index = 0; index < scheme.size(); ++index) {
        if (!ascii_equal(authorization[position + index], scheme[index])) { return false; }
    }
    position += scheme.size();
    if (position == authorization.size() || !is_whitespace(authorization[position])) {
        return false;
    }
    while (position < authorization.size() && is_whitespace(authorization[position])) {
        ++position;
    }
    std::size_t end = authorization.size();
    while (end > position && is_whitespace(authorization[end - 1])) { --end; }
    return authorization.substr(position, end - position) == api_key;
}

HttpServer::HttpServer(ServeOptions options, std::shared_ptr<spdlog::logger> logger)
    : options_(std::move(options)), openai_responses_store_(options_.response_store_max_records,
                                                            options_.response_store_max_bytes),
      operational_log_(logger),
      request_jsonl_(options_.request_log_jsonl, options_.artifact_path, std::move(logger)) {
    const std::size_t queued_requests =
        static_cast<std::size_t>(options_.max_concurrency) + options_.max_pending_requests;
    const std::size_t worker_count = queued_requests + 1;
    server_.new_task_queue         = [queued_requests, worker_count] {
        return new httplib::ThreadPool(worker_count, queued_requests);
    };
    server_.set_socket_options(configure_http_server_socket);
    server_.set_payload_max_length(options_.max_request_bytes);
    register_routes();
}

HttpServer::RequestLifecycle::RequestLifecycle(HttpServer& owner, RequestLogContext context)
    : owner_(&owner), context_(std::move(context)) {
    owner_->record_request_start(context_);
}

bool HttpServer::RequestLifecycle::claim(State terminal) noexcept {
    State expected = State::Pending;
    return state_.compare_exchange_strong(expected, terminal, std::memory_order_acq_rel);
}

void HttpServer::RequestLifecycle::done(const GenerationOutcome& outcome) {
    if (claim(State::Done)) { owner_->record_request_done(context_, outcome); }
}

void HttpServer::RequestLifecycle::failure(const RequestFailure& failure) {
    if (claim(State::Error)) { owner_->record_request_failure(context_, failure); }
}

void HttpServer::RequestLifecycle::response_failure(const RequestFailure& failure) {
    owner_->record_response_failure(context_.id, failure);
}

std::shared_ptr<HttpServer::RequestLifecycle> HttpServer::begin_request(RequestLogContext context) {
    return std::make_shared<RequestLifecycle>(*this, std::move(context));
}

void HttpServer::record_request_start(const RequestLogContext& context) {
    request_jsonl_.write_request_start(context);
    operational_log_.request_start(context);
}

void HttpServer::record_request_rejected(const RequestRejectionLogContext& context) {
    request_jsonl_.write_request_rejected(context);
    operational_log_.request_rejected(context);
}

void HttpServer::record_request_done(const RequestLogContext& context,
                                     const GenerationOutcome& outcome) {
    request_jsonl_.write_request_done(context, outcome);
    operational_log_.request_done(context, outcome);
}

void HttpServer::record_request_failure(const RequestLogContext& context,
                                        const RequestFailure& failure) {
    request_jsonl_.write_request_error(context, failure.machine_message);
    operational_log_.request_failure(context, failure);
}

void HttpServer::record_response_failure(std::uint64_t request_id, const RequestFailure& failure) {
    operational_log_.response_failure(request_id, failure);
}

void HttpServer::record_throughput(const ThroughputReport& report) {
    request_jsonl_.write_throughput(report);
    operational_log_.throughput(report);
}

void HttpServer::run_stats_reporter() {
    using Clock                     = std::chrono::steady_clock;
    ninfer::RuntimeStats previous   = service_->runtime_stats();
    Clock::time_point previous_time = Clock::now();
    const auto interval             = std::chrono::milliseconds(options_.log_stats_interval_ms);

    for (;;) {
        {
            std::unique_lock lock(stats_mutex_);
            if (stats_cv_.wait_for(lock, interval, [this] { return stats_stopping_; })) { break; }
        }

        const ninfer::RuntimeStats current = service_->runtime_stats();
        const Clock::time_point now        = Clock::now();
        const ThroughputReport report      = make_throughput_report(
            previous, current, std::chrono::duration<double>(now - previous_time).count());
        if (report_has_activity(report)) { record_throughput(report); }
        previous      = current;
        previous_time = now;
    }

    const ninfer::RuntimeStats current = service_->runtime_stats();
    const Clock::time_point now        = Clock::now();
    const ThroughputReport tail        = make_throughput_report(
        previous, current, std::chrono::duration<double>(now - previous_time).count());
    if (report_has_activity(tail)) { record_throughput(tail); }
}

void HttpServer::stop_stats_reporter() {
    if (!stats_thread_.joinable()) { return; }
    {
        std::lock_guard lock(stats_mutex_);
        stats_stopping_ = true;
    }
    stats_cv_.notify_one();
    stats_thread_.join();
}

void HttpServer::register_routes() {
    server_.set_error_handler([this](const httplib::Request& request, httplib::Response& response) {
        return handle_unrendered_http_error(options_, request, response);
    });
    if (options_.enable_cors) {
        server_.set_default_headers(
            {{"Access-Control-Allow-Origin", "*"},
             {"Access-Control-Expose-Headers", "x-request-id, request-id"},
             {"Access-Control-Allow-Headers",
              "Authorization, Content-Type, X-API-Key, anthropic-version, anthropic-beta, "
              "anthropic-user-profile-id"},
             {"Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS"}});
        // CORS preflight: browsers send OPTIONS with no credentials before the real
        // request; answer it without auth so the actual GET/POST can carry the key.
        server_.Options(R"(.*)",
                        [](const httplib::Request&, httplib::Response& res) { res.status = 204; });
    }

    server_.set_pre_routing_handler([this](const httplib::Request& req, httplib::Response& res) {
        ensure_openai_request_id(req, res);
        if (options_.api_key.empty() || req.path == "/health" || req.method == "OPTIONS") {
            return httplib::Server::HandlerResponse::Unhandled;
        }
        // Accept both the OpenAI-style bearer token and the Anthropic-style
        // x-api-key header so OpenAI clients and Claude Code (ANTHROPIC_API_KEY
        // -> x-api-key, ANTHROPIC_AUTH_TOKEN -> Authorization: Bearer) both work.
        const bool bearer_ok =
            matches_bearer_credential(req.get_header_value("Authorization"), options_.api_key);
        const bool x_api_key_ok = req.get_header_value("x-api-key") == options_.api_key;
        if (!bearer_ok && !x_api_key_ok) {
            ApiError error;
            error.status  = 401;
            error.type    = "invalid_request_error";
            error.code    = "invalid_api_key";
            error.message = "missing or invalid API key";
            // Render the 401 in the shape the target endpoint speaks.
            if (req.path.rfind("/v1/messages", 0) == 0) {
                write_anthropic_error(res, error, new_anthropic_request_id());
            } else {
                write_openai_error(res, error);
            }
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });

    server_.set_exception_handler(
        [this](const httplib::Request& req, httplib::Response& res, std::exception_ptr ep) {
            ensure_openai_request_id(req, res);
            try {
                std::rethrow_exception(ep);
            } catch (const ApiException& e) {
                if (e.error().status >= 500) {
                    operational_log_.http_failure(
                        endpoint_name(req.path),
                        make_request_failure(RequestFailurePhase::Http, e.error()),
                        response_request_id(res));
                }
                if (req.path.rfind("/v1/messages", 0) == 0) {
                    write_anthropic_error(res, e.error(), new_anthropic_request_id());
                } else {
                    write_openai_error(res, e.error());
                }
            } catch (const std::exception& e) {
                operational_log_.http_failure(
                    endpoint_name(req.path),
                    make_internal_request_failure(RequestFailurePhase::Http, e.what()),
                    response_request_id(res));
                if (req.path.rfind("/v1/messages", 0) == 0) {
                    ApiError error;
                    error.status  = 500;
                    error.message = e.what();
                    write_anthropic_error(res, error, new_anthropic_request_id());
                } else {
                    write_exception(res, e);
                }
            } catch (...) {
                operational_log_.http_failure(
                    endpoint_name(req.path),
                    make_internal_request_failure(RequestFailurePhase::Http, "unknown error"),
                    response_request_id(res));
                ApiError error;
                error.status  = 500;
                error.type    = "internal_error";
                error.message = "unknown error";
                if (req.path.rfind("/v1/messages", 0) == 0) {
                    write_anthropic_error(res, error, new_anthropic_request_id());
                } else {
                    write_openai_error(res, error);
                }
            }
        });

    server_.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(nlohmann::json{{"status", "ok"}}.dump(), "application/json");
    });
    server_.Get("/v1/models", [this](const httplib::Request& req, httplib::Response& res) {
        handle_models(req, res);
    });
    server_.Get(R"(/v1/models/(.+))", [this](const httplib::Request& req, httplib::Response& res) {
        handle_model(req, res);
    });
    server_.Post("/v1/chat/completions",
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_chat_completions(req, res);
                 });
    server_.Post("/v1/score", [this](const httplib::Request& req, httplib::Response& res) {
        handle_score(req, res);
    });
    server_.Post("/v1/responses", [this](const httplib::Request& req, httplib::Response& res) {
        handle_responses(req, res);
    });
    server_.Post("/v1/responses/input_tokens",
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_response_input_tokens(req, res);
                 });
    server_.Post("/v1/responses/compact",
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_response_compact(req, res);
                 });
    server_.Post(R"(/v1/responses/([^/]+)/cancel)",
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_response_cancel(req, res);
                 });
    server_.Get(R"(/v1/responses/([^/]+)/input_items)",
                [this](const httplib::Request& req, httplib::Response& res) {
                    handle_response_input_items(req, res);
                });
    server_.Get(R"(/v1/responses/([^/]+))",
                [this](const httplib::Request& req, httplib::Response& res) {
                    handle_response_get(req, res);
                });
    server_.Delete(R"(/v1/responses/([^/]+))",
                   [this](const httplib::Request& req, httplib::Response& res) {
                       handle_response_delete(req, res);
                   });
    server_.Post("/v1/messages/count_tokens",
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_count_tokens(req, res);
                 });
    server_.Post("/v1/messages", [this](const httplib::Request& req, httplib::Response& res) {
        handle_messages(req, res);
    });
    server_.Get("/admin/vram", [this](const httplib::Request& req, httplib::Response& res) {
        handle_admin_vram(req, res);
    });
    server_.Post("/admin/quiesce", [this](const httplib::Request& req, httplib::Response& res) {
        handle_admin_quiesce(req, res);
    });
}

namespace {

const char* kv_cache_storage_name(KvCacheStorage storage) noexcept {
    switch (storage) {
        case KvCacheStorage::BFloat16: return "bf16";
        case KvCacheStorage::Int8Group64: return "int8";
        case KvCacheStorage::Fp8E4M3Row256: return "fp8";
        case KvCacheStorage::Nvfp4Group16: return "nvfp4";
        case KvCacheStorage::Fp8KeyNvfp4Value: return "k8v4";
    }
    return "unknown";
}

nlohmann::json arena_json(const ArenaMemorySummary& arena) {
    return {{"capacity_bytes", arena.capacity_bytes},
            {"used_bytes", arena.used_bytes},
            {"peak_used_bytes", arena.peak_used_bytes}};
}

// NVML process enumeration can take seconds on Windows. Only the refreshing
// request waits for the driver; concurrent readers receive the previous snapshot
// with its age, or unavailable until the first snapshot has been published.
std::optional<DeviceSnapshotReading> device_snapshot(int device) {
    static DeviceSnapshotCache cache;
    return cache.read([device] { return query_device_memory(device); });
}

} // namespace

// The memory report an operator or supervisor needs, from the process that actually owns the
// memory. Two rules it exists to enforce:
//
//   1. Device-wide free comes from NVML, never cudaMemGetInfo and never the DXGI budget. Both
//      of those report an empty card while another process holds 70 GiB under WDDM, which is
//      how this workstation's desktop got starved twice; `device.source` says which one
//      answered so a reader can distrust the number when NVML is missing.
//   2. The plan and the live device numbers are reported side by side. The gap between
//      `plan.runtime_reservation_bytes` and what is actually resident is the engine's own
//      overshoot; free memory falling without that gap moving is somebody else's doing, and
//      the two must never be conflated when deciding whether to shed load.
void HttpServer::handle_admin_vram(const httplib::Request&, httplib::Response& res) const {
    if (service_ == nullptr) {
        ApiError error;
        error.status  = 503;
        error.type    = "service_unavailable";
        error.message = "engine is not attached";
        write_openai_error(res, error);
        return;
    }

    // Never the blocking memory_summary() here. It takes the engine's execution
    // mutex, which the worker holds for a whole execution unit, so a once-a-second
    // poller would take one HTTP worker per poll and block it; under sustained
    // load every worker is consumed and the server accepts connections only to
    // close them. Serve the last good reading with its age instead -- a couple of
    // seconds of staleness in a diagnostics payload costs nothing, and taking the
    // engine offline to avoid it costs everything.
    static std::mutex memory_cache_mu;
    static MemorySummary memory_cached;
    static std::chrono::steady_clock::time_point memory_taken{};
    static bool memory_primed = false;

    MemorySummary memory;
    std::int64_t memory_age_ms = 0;
    if (std::optional<MemorySummary> fresh = service_->try_memory_summary()) {
        std::lock_guard lock(memory_cache_mu);
        memory_cached = *fresh;
        memory_taken  = std::chrono::steady_clock::now();
        memory_primed = true;
        memory        = *fresh;
    } else {
        std::lock_guard lock(memory_cache_mu);
        if (!memory_primed) {
            ApiError error;
            error.status  = 503;
            error.type    = "service_unavailable";
            error.message = "engine memory summary is not available yet";
            write_openai_error(res, error);
            return;
        }
        memory        = memory_cached;
        memory_age_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - memory_taken)
                            .count();
    }
    const EngineOptions& engine  = service_->engine_options();
    const auto reading = device_snapshot(engine.device);
    if (!reading) {
        ApiError error;
        error.status = 503;
        error.type = "service_unavailable";
        error.message = "device memory snapshot is not available yet";
        write_openai_error(res, error);
        return;
    }
    const DeviceMemorySnapshot& device = reading->snapshot;
    const std::int64_t device_age_ms = reading->age_ms;

    nlohmann::json processes = nlohmann::json::array();
    for (const ProcessMemoryInfo& process : device.compute_processes) {
        processes.push_back({{"pid", process.pid}, {"used_bytes", process.used_bytes}});
    }

    // `runtime_floor_bytes` is the value an allocation is actually checked against at run time
    // (Sequence D23); `configured_bytes` is what was asked for. They differ when the floor has
    // not been armed, which is worth seeing rather than assuming.
    const std::size_t floor = runtime_desktop_reserve_floor();
    nlohmann::json reserve  = {
        {"configured_bytes", engine.desktop_reserve_bytes},
        {"runtime_floor_bytes", floor},
        {"free_bytes", device.free_bytes},
        {"holding", floor == 0 || device.free_bytes >= floor}};

    nlohmann::json plan = {
        {"runtime_reservation_bytes", memory.runtime_reservation_bytes},
        {"minimum_runtime_reservation_bytes", memory.minimum_runtime_reservation_bytes},
        {"available_after_weights_bytes", memory.available_after_weights_bytes},
        {"available_after_startup_bytes", memory.available_after_startup_bytes},
        {"planned_slack_bytes", memory.planned_slack_bytes},
        {"workspace_logical_peak_bytes", memory.workspace_logical_peak_bytes},
        // Inside the reservation but never allocated: headroom for CUDA graphs captured later.
        // A check of the form "reservation should equal resident bytes" is wrong by this much.
        {"cuda_graph_allowance_bytes", memory.cuda_graph_allowance_bytes},
        {"kv_capacity_headroom_bytes", memory.kv_capacity_headroom_bytes}};

    nlohmann::json arenas = {{"weights", arena_json(memory.weights)},
                             {"sequence", arena_json(memory.sequence)},
                             {"workspace", arena_json(memory.workspace)}};
    if (memory.vision_workspace.has_value()) {
        // Logical regions inside the one physical workspace allocation; they describe layout,
        // and must not be added to workspace.capacity_bytes.
        arenas["vision_workspace"] = {
            {"general_capacity_bytes", memory.vision_workspace->general_capacity_bytes},
            {"encode_peak_bytes", memory.vision_workspace->encode_peak_bytes},
            {"handoff_capacity_bytes", memory.vision_workspace->handoff_capacity_bytes},
            {"aggregate_prompt_tokens", memory.vision_workspace->aggregate_prompt_tokens},
            {"max_item_tokens", memory.vision_workspace->max_item_tokens}};
    } else {
        arenas["vision_workspace"] = nullptr;
    }

    nlohmann::json kv = {{"capacity_tokens", memory.kv_capacity},
                         {"page_groups", memory.kv_capacity_page_groups},
                         {"max_page_groups", memory.kv_capacity_max_page_groups},
                         {"payload_bytes", memory.kv_payload_bytes},
                         {"storage", kv_cache_storage_name(memory.kv_cache)},
                         {"host_state_capacity_slots", memory.host_state_capacity_slots},
                         {"host_state_occupied_slots", memory.host_state_occupied_slots},
                         {"host_kv_capacity_bytes", memory.host_kv_capacity_bytes},
                         {"host_kv_occupied_bytes", memory.host_kv_occupied_bytes}};

    nlohmann::json body = {
        {"schema_version", 1},
        {"model_id", public_model_id_},
        // Non-zero when the engine was busy and this payload reports the previous
        // reading rather than blocking behind execution to take a fresh one.
        {"memory_age_ms", memory_age_ms},
        {"device",
         {{"index", engine.device},
          {"name", device.device_name},
          {"pci_bus_id", device.pci_bus_id},
          {"total_bytes", device.total_bytes},
          {"free_bytes", device.free_bytes},
          {"used_bytes", device.used_bytes},
          {"source", device.is_nvml ? "nvml" : "cudaMemGetInfo"},
          // How old this reading is. Non-zero means it was served from cache
          // while a refresh was in flight or still fresh; a caller deciding
          // anything on free memory should know it is not reading the driver.
          {"age_ms", device_age_ms},
          {"warning", device.warning},
          {"compute_processes", processes}}},
        {"desktop_reserve", reserve},
        {"plan", plan},
        {"arenas", arenas},
        {"kv", kv},
        {"options",
         {{"max_concurrency", engine.max_concurrency},
          {"vision", engine.enable_vision},
          {"output_head_fp8", engine.quantize_output_head_fp8},
          {"token_embedding_fp8", engine.quantize_token_embedding_fp8},
          {"gdn_state_storage",
           engine.gdn_state_storage == GdnStateStorage::FP32 ? "fp32" : "bf16"}}},
        // Stated rather than implied: nothing here frees memory. A caller that wants device
        // memory back has one option today, which is to stop the process.
        {"release",
         {{"supported", false},
          {"reason", "this build allocates device memory once at startup and holds it; no "
                     "residency path exists to release and rebuild it"}}}};

    res.set_content(body.dump(), "application/json");
}

void HttpServer::handle_admin_quiesce(const httplib::Request& req, httplib::Response& res) const {
    if (service_ == nullptr) {
        ApiError error;
        error.status  = 503;
        error.type    = "service_unavailable";
        error.message = "engine is not attached";
        write_openai_error(res, error);
        return;
    }

    // An optional hold, so the request-holding behaviour can be observed rather
    // than inferred from a fence that returns instantly. Bounded because this
    // stops the engine for everyone: a caller cannot use it to take the server
    // down. Real residency work will replace it and is not expected to need it.
    std::int64_t hold_ms = 0;
    if (!req.body.empty()) {
        try {
            const auto parsed = nlohmann::json::parse(req.body);
            hold_ms           = parsed.value("hold_ms", static_cast<std::int64_t>(0));
        } catch (const std::exception&) {
            ApiError error;
            error.status  = 400;
            error.type    = "invalid_request_error";
            error.message = "body must be JSON with an optional integer hold_ms";
            write_openai_error(res, error);
            return;
        }
    }
    if (hold_ms < 0 || hold_ms > 10'000) {
        ApiError error;
        error.status  = 400;
        error.type    = "invalid_request_error";
        error.message = "hold_ms must be between 0 and 10000";
        write_openai_error(res, error);
        return;
    }

    try {
        const ninfer::Engine::QuiescenceReport report =
            service_->run_at_quiescence([hold_ms] {
                if (hold_ms > 0) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(hold_ms));
                }
            });
        const auto ms = [](std::chrono::nanoseconds value) {
            return std::chrono::duration<double, std::milli>(value).count();
        };
        nlohmann::json body = {
            {"schema_version", 1},
            // How long the in-flight work took to finish once admission stopped.
            // This is the part a residency change cannot avoid paying.
            {"drain_ms", ms(report.drain)},
            {"work_ms", ms(report.work)},
            // Requests that waited through the hold and were carried across with
            // their admission deadlines extended. None of them were dropped;
            // that is the property this endpoint exists to demonstrate.
            {"requests_held", report.requests_held},
            {"capacity_change", nullptr}};
        res.set_content(body.dump(), "application/json");
    } catch (const std::exception& ex) {
        // A fence that cannot run is a refusal, not a broken engine. The engine
        // keeps serving either way.
        ApiError error;
        error.status  = 409;
        error.type    = "conflict";
        error.message = std::string("engine could not be quiesced: ") + ex.what();
        write_openai_error(res, error);
    }
}

void HttpServer::handle_models(const httplib::Request&, httplib::Response& res) const {
    res.set_content(make_models_list(public_model_id_, unix_time_now(), options_.max_context),
                    "application/json");
}

void HttpServer::handle_model(const httplib::Request& req, httplib::Response& res) const {
    const std::string id = req.matches.size() > 1 ? req.matches[1].str() : std::string();
    if (id != public_model_id_) {
        ApiError error;
        error.status  = 404;
        error.type    = "invalid_request_error";
        error.code    = "model_not_found";
        error.message = "model '" + id + "' not found";
        write_openai_error(res, error);
        return;
    }
    res.set_content(make_model_object(public_model_id_, unix_time_now(), options_.max_context),
                    "application/json");
}

bool HttpServer::bind() { return server_.bind_to_port(options_.host, options_.port); }

void HttpServer::attach(GenerationService& service) {
    if (service_ != nullptr) {
        throw std::logic_error("HTTP generation service is already attached");
    }
    const ninfer::LoadSummary load = service.load_summary();
    public_model_id_               = resolve_public_model_id(options_, load.model_id);
    service_                       = &service;
    request_jsonl_.write_server_start(options_, service.engine_options(),
                                      service.sampling_defaults(), public_model_id_, load,
                                      service.memory_summary());
}

bool HttpServer::listen() {
    if (service_ == nullptr) { throw std::logic_error("HTTP generation service is not attached"); }
    if (public_model_id_.empty()) {
        throw std::logic_error("HTTP public model id is not resolved");
    }
    if (options_.log_stats_interval_ms != 0) {
        stats_stopping_ = false;
        stats_thread_   = std::thread([this] { run_stats_reporter(); });
    }
    try {
        const bool result = server_.listen_after_bind();
        stop_stats_reporter();
        return result;
    } catch (...) {
        stop_stats_reporter();
        throw;
    }
}

void HttpServer::stop() { server_.stop(); }

} // namespace ninfer::serve
