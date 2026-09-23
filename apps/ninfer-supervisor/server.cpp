#include "server.hpp"
#include "insights.hpp"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#endif
#define CPPHTTPLIB_NO_EXCEPTIONS
#include <httplib.h>

#include <chrono>
#include <cstdio>
#include <iostream>
#include <thread>

namespace ninfer::supervisor {
namespace {

const char* state_name(EngineState s) {
    switch (s) {
    case EngineState::Stopped: return "Stopped";
    case EngineState::Starting: return "Starting";
    case EngineState::Running: return "Running";
    case EngineState::Stopping: return "Stopping";
    case EngineState::BackingOff: return "BackingOff";
    case EngineState::Halted: return "Halted";
    }
    return "?";
}

std::int64_t now_unix_s() {
    using namespace std::chrono;
    return duration_cast<seconds>(system_clock::now().time_since_epoch()).count();
}

} // namespace

DashboardServer::DashboardServer(SupervisorConfig cfg, EngineChild& child, Collector& collector)
    : cfg_(std::move(cfg)), child_(child), collector_(collector) {}

DashboardServer::~DashboardServer() { stop(); }

bool DashboardServer::control_allowed(const std::string& remote) const {
    return is_loopback_peer(remote);
}

namespace {

const char* param_kind_name(ParamKind kind) noexcept {
    switch (kind) {
    case ParamKind::Int: return "int";
    case ParamKind::Text: return "text";
    case ParamKind::Enum: return "enum";
    case ParamKind::Flag: return "flag";
    case ParamKind::IntOrAuto: return "int_or_auto";
    }
    return "text";
}

} // namespace

nlohmann::json DashboardServer::config_json() const {
    const ParsedEngineArgs parsed = parse_engine_args(cfg_.engine.args);
    const std::string identity = artifact_model_identity(parsed.artifact);

    // The form's field list comes from the engine parameter table, so a knob added
    // in C++ appears in the page with its own label, help text and bounds, and no
    // second description can go stale.
    nlohmann::json schema = nlohmann::json::array();
    for (const auto& spec : engine_param_specs()) {
        nlohmann::json choices = nlohmann::json::array();
        for (const auto& choice : spec.choices) {
            if (choice.empty()) { continue; }
            if (spec.key == "spec" && !identity.empty() &&
                ((choice == "dflash2" && identity != "qwen3.8-27b" &&
                  identity != "qwen3.8-27b-orcarouter") ||
                 (choice == "dflash" && identity != "qwen3.6-35b-a3b"))) { continue; }
            choices.push_back(std::string(choice));
        }
        schema.push_back({{"key", std::string(spec.key)},
                          {"flag", std::string(spec.flag)},
                          {"kind", param_kind_name(spec.kind)},
                          {"group", std::string(spec.group)},
                          {"label", std::string(spec.label)},
                          {"help", std::string(spec.help)},
                          {"min", spec.min_value},
                          {"max", spec.max_value},
                          {"choices", choices}});
    }

    nlohmann::json values = nlohmann::json::object();
    for (const auto& param : parsed.params) { values[param.key] = param.value; }

    return {
        {"config_path", cfg_.source_path},
        {"model_identity", identity},
        {"writable", !cfg_.source_path.empty()},
        {"manages_engine", manages_engine_process(cfg_)},
        {"engine",
         {// Shown, never editable here: changing which binary runs is a file-only
          // decision. See the note on the parameter table.
          {"executable", cfg_.engine.executable},
          {"workdir", cfg_.engine.workdir},
          {"artifact", parsed.artifact},
          {"engine_host", cfg_.engine.engine_host},
          {"engine_port", cfg_.engine.engine_port},
          {"device", cfg_.engine.device},
          {"request_log", cfg_.engine.request_log},
          {"unmanaged", cfg_.engine.unmanaged},
          // The path, and whether it resolves to something. Never the key itself.
          {"api_key_file", cfg_.engine.api_key_file},
          {"api_key_present", !read_api_key_quiet(cfg_.engine.api_key_file).empty()},
          {"args", redact_engine_args(cfg_.engine.args)},
          {"passthrough", parsed.passthrough}}},
        {"supervisor",
         {{"host", cfg_.host},
          {"port", cfg_.port},
          {"bind_any", cfg_.bind_any},
          {"monitor_only", cfg_.monitor_only},
          {"logs_dir", cfg_.logs_dir},
          {"restart",
           {{"max_backoff_s", cfg_.restart.max_backoff_s},
            {"crash_loop_window_s", cfg_.restart.crash_loop_window_s},
            {"crash_loop_max", cfg_.restart.crash_loop_max},
            {"health_fail_threshold", cfg_.restart.health_fail_threshold}}}}},
        {"params", values},
        {"schema", schema}};
}

DashboardServer::ConfigResult DashboardServer::apply_config(const std::string& request_body) {
    if (cfg_.source_path.empty()) {
        return {409, {{"error", "this supervisor has no config file to write to"}}};
    }
    nlohmann::json body;
    try {
        body = nlohmann::json::parse(request_body);
    } catch (const std::exception& ex) {
        return {400, {{"error", std::string("request is not valid JSON: ") + ex.what()}}};
    }

    SupervisorConfig next = cfg_;
    std::vector<std::string> errors;
    bool engine_restart_required     = false;
    bool supervisor_restart_required = false;

    if (body.contains("params") && body.at("params").is_object()) {
        ParsedEngineArgs parsed = parse_engine_args(cfg_.engine.args);
        std::vector<EngineParam> edited;
        for (const auto& [key, value] : body.at("params").items()) {
            const EngineParamSpec* spec = find_engine_param(key);
            if (spec == nullptr) {
                errors.push_back(key + " is not an editable parameter");
                continue;
            }
            // Everything is compared and stored as text: it is what the argument
            // vector holds, and it keeps "8192" from a number field and "8192"
            // from a text field indistinguishable, as they should be.
            std::string text;
            if (value.is_string()) {
                text = value.get<std::string>();
            } else if (value.is_boolean()) {
                text = value.get<bool>() ? "true" : "false";
            } else if (value.is_number_integer()) {
                text = std::to_string(value.get<long long>());
            } else if (value.is_null()) {
                text = ""; // explicit removal
            } else {
                errors.push_back(std::string(spec->label) + " has an unsupported value type");
                continue;
            }
            edited.push_back({key, text});
        }
        // An empty value means "unset this parameter", which is not the same as a
        // bad value: validating "" as an integer would reject every removal.
        std::vector<EngineParam> to_check;
        for (const auto& param : edited) {
            if (!param.value.empty()) { to_check.push_back(param); }
        }
        for (const auto& message : validate_engine_params(to_check)) { errors.push_back(message); }

        // Apply onto a copy so a later cross-field failure leaves nothing behind.
        std::vector<EngineParam> merged = parsed.params;
        for (const auto& param : edited) {
            const bool remove = param.value.empty() ||
                                (find_engine_param(param.key)->kind == ParamKind::Flag &&
                                 param.value == "false");
            std::erase_if(merged, [&](const EngineParam& p) { return p.key == param.key; });
            if (!remove) { merged.push_back(param); }
        }
        for (const auto& message : validate_engine_param_combination(
                 merged, artifact_model_identity(parsed.artifact))) {
            errors.push_back(message);
        }
        parsed.params            = std::move(merged);
        next.engine.args         = render_engine_args(parsed);
        engine_restart_required  = next.engine.args != cfg_.engine.args;
    }

    if (body.contains("engine") && body.at("engine").is_object()) {
        const auto& e = body.at("engine");
        if (e.contains("engine_host")) {
            next.engine.engine_host = e.value("engine_host", cfg_.engine.engine_host);
        }
        if (e.contains("engine_port")) {
            const int port = e.value("engine_port", cfg_.engine.engine_port);
            if (port < 1 || port > 65535) {
                errors.emplace_back("Engine port must be between 1 and 65535");
            } else {
                next.engine.engine_port = port;
            }
        }
        if (e.contains("device")) {
            const int device = e.value("device", cfg_.engine.device);
            if (device < 0 || device > 15) {
                errors.emplace_back("Device index must be between 0 and 15");
            } else {
                next.engine.device = device;
            }
        }
        if (e.contains("request_log")) {
            next.engine.request_log = e.value("request_log", cfg_.engine.request_log);
        }
        if (e.contains("api_key_file")) {
            next.engine.api_key_file = e.value("api_key_file", cfg_.engine.api_key_file);
        }
        // Where the engine listens and where the supervisor polls have to be the same
        // place. Rewriting --host/--port here is what keeps a port edit from producing
        // an engine that starts and is then permanently unreachable.
        if (next.engine.engine_host != cfg_.engine.engine_host ||
            next.engine.engine_port != cfg_.engine.engine_port) {
            next.engine.args = with_engine_endpoint(std::move(next.engine.args),
                                                    next.engine.engine_host,
                                                    next.engine.engine_port);
        }
        // The engine's own listen address and its launch-time flags both only take
        // effect when the process is started again.
        engine_restart_required =
            engine_restart_required || next.engine.engine_host != cfg_.engine.engine_host ||
            next.engine.engine_port != cfg_.engine.engine_port ||
            next.engine.device != cfg_.engine.device ||
            next.engine.request_log != cfg_.engine.request_log ||
            next.engine.api_key_file != cfg_.engine.api_key_file;
    }

    if (body.contains("supervisor") && body.at("supervisor").is_object()) {
        const auto& s = body.at("supervisor");
        if (s.contains("port")) {
            const int port = s.value("port", cfg_.port);
            if (port < 1 || port > 65535) {
                errors.emplace_back("Dashboard port must be between 1 and 65535");
            } else {
                next.port = port;
            }
        }
        if (s.contains("host")) { next.host = s.value("host", cfg_.host); }
        if (s.contains("bind_any")) { next.bind_any = s.value("bind_any", cfg_.bind_any); }
        // Refusing this in the UI rather than at startup: a non-loopback host with
        // bind_any off is a config the supervisor will not start with, and finding
        // that out at the next login is worse than being told now.
        if (!next.bind_any && !is_loopback_host(next.host)) {
            errors.emplace_back("Dashboard host must be a loopback address unless 'bind beyond "
                                "loopback' is enabled");
        }
        if (s.contains("restart") && s.at("restart").is_object()) {
            const auto& r                     = s.at("restart");
            next.restart.max_backoff_s        = r.value("max_backoff_s", cfg_.restart.max_backoff_s);
            next.restart.crash_loop_window_s  = r.value("crash_loop_window_s",
                                                        cfg_.restart.crash_loop_window_s);
            next.restart.crash_loop_max       = r.value("crash_loop_max", cfg_.restart.crash_loop_max);
            next.restart.health_fail_threshold =
                r.value("health_fail_threshold", cfg_.restart.health_fail_threshold);
        }
        supervisor_restart_required = next.host != cfg_.host || next.port != cfg_.port ||
                                      next.bind_any != cfg_.bind_any;
    }

    if (!errors.empty()) { return {400, {{"error", "invalid configuration"}, {"details", errors}}}; }

    // Keep the selected preset in sync, so selecting it again preserves edits.
    for (auto& model : next.models) {
        if (model.id == cfg_.active_model && model_engine_args(model) == cfg_.engine.args) {
            model.args.assign(next.engine.args.begin() + 1, next.engine.args.end());
            break;
        }
    }

    const bool dry_run = body.value("dry_run", false);
    if (!dry_run) {
        try {
            save_config_json(cfg_.source_path, next, cfg_.source_stamp);
        } catch (const std::exception& ex) {
            return {500, {{"error", ex.what()}}};
        }
        // Re-read the stamp we just created. `next` still carries the stamp from
        // BEFORE this write, and storing that makes the supervisor refuse its own
        // next edit as an external change -- which it did: saving anything here
        // then blocked every later model switch with "config file changed on disk"
        // until the supervisor was restarted.
        next.source_stamp = config_file_stamp(cfg_.source_path);
        cfg_              = next;
        // The child owns the copy that spawn() reads; without this the file would
        // change and the next restart would still launch the old parameters.
        child_.update_config(next);
    }

    nlohmann::json out          = config_json();
    out["saved"]                = !dry_run;
    out["engine_restart_required"]     = engine_restart_required;
    out["supervisor_restart_required"] = supervisor_restart_required;
    return {200, out};
}

// Blocks until the engine answers /health, or the breaker halts, or time runs out.
//
// Health, not the process existing: the failure mode being waited on is a process
// that starts and then exits during target planning, so "is it serving" is the
// only question worth asking. A Halted breaker short-circuits, since nothing
// changes by waiting once it has given up.
bool DashboardServer::wait_until_serving(std::chrono::seconds limit, int replaced_pid) {
    const auto deadline = std::chrono::steady_clock::now() + limit;
    while (std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        const EngineStatus st = child_.status();
        if (st.state == EngineState::Halted) { return false; }
        // Two guards, and both are needed.
        //
        // The pid must have changed, or a health reading could describe the
        // engine being replaced rather than the one being waited for. Spawning
        // is fast -- it is the model load that takes 15 to 60 seconds -- so the
        // pid turns over almost immediately and this alone is not enough.
        //
        // And the probe must be fresh. Using the collector's snapshot here made a
        // failed switch answer "serving: true" in one second, because that
        // snapshot is sampled on its own schedule and still held the old
        // engine's 200 well after the process it described was gone. Asking the
        // engine directly is the only reading that cannot be stale.
        if (st.pid == 0 || st.pid == replaced_pid) { continue; }
        httplib::Client probe(engine_connect_host(cfg_.engine), cfg_.engine.engine_port);
        probe.set_connection_timeout(1, 0);
        probe.set_read_timeout(2, 0);
        // /health is exempt from the engine's API key, so no credential is needed.
        if (auto res = probe.Get("/health"); res && res->status == 200) { return true; }
    }
    return false;
}

nlohmann::json DashboardServer::models_json() const {
    nlohmann::json models = nlohmann::json::array();
    const std::string active_label = model_label_from_args(cfg_.engine.args);
    for (const auto& model : cfg_.models) {
        const ModelAvailability status = check_model_available(model);
        models.push_back({
            {"id", model.id},
            {"name", model.name.empty() ? model.id : model.name},
            {"description", model.description},
            {"artifact", model.artifact},
            {"available", status.available},
            {"reason", status.reason},
            {"size_bytes", status.size_bytes},
            // Active is decided by what the engine is actually configured to
            // launch, not by the active_model field, so a hand-edited args array
            // cannot make the dashboard claim a model that is not being served.
            {"active", model_engine_args(model) == cfg_.engine.args},
        });
    }
    return {{"models", models},
            {"active_model", cfg_.active_model},
            {"active_artifact_label", active_label}};
}

DashboardServer::ConfigResult DashboardServer::select_model(const std::string& request_body) {
    if (cfg_.source_path.empty()) {
        return {409, {{"error", "this supervisor has no config file to write to"}}};
    }
    if (cfg_.models.empty()) {
        return {409, {{"error", "no models are configured; add a \"models\" array to the config"}}};
    }
    if (!manages_engine_process(cfg_)) {
        return {409, {{"error", "this supervisor does not manage the engine process"}}};
    }
    nlohmann::json body;
    try {
        body = nlohmann::json::parse(request_body);
    } catch (const std::exception& ex) {
        return {400, {{"error", std::string("request is not valid JSON: ") + ex.what()}}};
    }
    const std::string id = body.value("id", "");
    if (id.empty()) { return {400, {{"error", "id is required"}}}; }
    const ModelEntry* model = find_model(cfg_, id);
    if (model == nullptr) { return {404, {{"error", "no configured model with id " + id}}}; }

    // Checked here and not only in the listing: the file can move between the
    // dashboard rendering a picker and someone clicking it, and a switch that
    // fails leaves the engine down and crash-looping on a missing artifact.
    const ModelAvailability status = check_model_available(*model);
    if (!status.available) {
        return {409, {{"error", "model " + id + " is unavailable: " + status.reason}}};
    }
    const auto parsed = parse_engine_args(model_engine_args(*model));
    auto errors = validate_engine_params(parsed.params);
    const auto combinations = validate_engine_param_combination(
        parsed.params, artifact_model_identity(model->artifact));
    errors.insert(errors.end(), combinations.begin(), combinations.end());
    if (!errors.empty()) {
        return {400, {{"error", "invalid launch preset"}, {"details", errors}}};
    }

    SupervisorConfig next = cfg_;
    next.engine.args      = model_engine_args(*model);
    next.active_model     = model->id;

    try {
        save_config_json(cfg_.source_path, next, cfg_.source_stamp);
    } catch (const std::exception& ex) {
        // 409, not 500: the usual cause is that the file was edited while the
        // supervisor held it, which is the operator's to resolve, not a fault.
        return {409, {{"error", ex.what()}}};
    }
    next.source_stamp = config_file_stamp(cfg_.source_path);
    const SupervisorConfig previous = cfg_;
    // Captured before the restart: the wait needs to know which process it is
    // replacing, or a stale health sample from the old one reads as success.
    const int replaced_pid = child_.status().pid;
    cfg_ = next;
    child_.update_config(next);
    // Restarting is the switch. Loading a different model means tearing down the
    // device state that belongs to the old one, so there is no in-place path, and
    // saying so beats a dashboard that looks changed while the old model serves.
    child_.restart();

    // Wait for the new model to actually serve, and put the old one back if it
    // does not.
    //
    // A valid artifact is not a valid configuration: flags are per-model, and a
    // catalog entry carrying another model's tuning fails in target planning
    // seconds after launch. Without this the switch leaves the engine crash-
    // looping until the breaker halts it, and the only way back is another API
    // call that the operator has to know to make. Observed exactly that twice
    // while building this, on a live engine.
    const bool healthy = wait_until_serving(std::chrono::seconds(180), replaced_pid);
    if (!healthy) {
        cfg_ = previous;
        try {
            save_config_json(cfg_.source_path, previous);
            // Same reason as the config editor above: the rollback wrote the file,
            // so the cached stamp has to describe what is on disk now.
            cfg_.source_stamp = config_file_stamp(cfg_.source_path);
        } catch (const std::exception&) { /* reported below either way */ }
        child_.update_config(previous);
        const int failed_pid = child_.status().pid;
        child_.restart();
        // Wait for the previous model too, and say which of the two things
        // actually happened.
        //
        // Returning as soon as the rollback was REQUESTED made "rolled back to Y"
        // a claim about an intention, not an observation: the caller was told it
        // was serving Y while Y was still loading, or had itself failed to come
        // back. Those are very different situations for whoever is reading it --
        // one is a bad model entry, the other is an engine that is now down.
        const bool restored = wait_until_serving(std::chrono::seconds(180), failed_pid);
        nlohmann::json out   = models_json();
        out["rolled_back_to"] = previous.active_model;
        out["serving"]        = restored;
        if (restored) {
            // Do not name a cause. This used to say the flags were probably not
            // valid, which was wrong often enough to matter: the same entry that
            // failed here has succeeded on a later attempt with nothing changed
            // but how much device memory was free at the time. The dashboard
            // shows this string verbatim, so guessing here becomes a guess the
            // operator reads as a finding.
            out["error"] = "model " + model->id + " did not start; rolled back to " +
                           previous.active_model +
                           ", which is serving again. The engine log records the phase it "
                           "failed in -- commonly this entry's launch settings, or too "
                           "little free device memory at that moment, in which case "
                           "retrying can succeed unchanged.";
        } else {
            out["error"] = "model " + model->id + " did not start, and " +
                           previous.active_model +
                           " has not come back either. The engine is down; check the engine "
                           "log and the crash-loop state.";
        }
        return {409, out};
    }

    nlohmann::json out = models_json();
    out["switched_to"] = model->id;
    out["serving"]     = true;
    return {200, out};
}

nlohmann::json DashboardServer::state_json() {
    const EngineStatus st = child_.status();
    Collected snap        = collector_.snapshot();
    nlohmann::json engine = {
        {"state", state_name(st.state)},
        {"pid", st.pid},
        {"restart_count", st.restart_count},
        {"last_exit_code", st.last_exit_code},
        {"last_event", st.last_event},
        {"crash_loop_halted", st.crash_loop_halted},
        {"kv_capacity_configured", st.kv_capacity_configured},
        {"kv_capacity_effective", st.kv_capacity_effective},
        {"kv_capacity_note", st.kv_capacity_note},
        {"launch_args", st.launch_args},
        {"uptime_s", st.started_unix_ms == 0
                         ? 0
                         : now_unix_s() - st.started_unix_ms / 1000},
    };
    nlohmann::json dxgi = {
        {"ok", snap.dxgi.ok},
        {"error", snap.dxgi.error},
        {"adapter_name", snap.dxgi.adapter_name},
        {"budget_bytes", snap.dxgi.budget_bytes},
        {"supervisor_usage_bytes", snap.dxgi.current_usage_bytes},
        {"supervisor_usage_note", "DXGI CurrentUsage of the supervisor process, not the engine"},
    };
    nlohmann::json nvidia = {{"ok", snap.nvidia.ok},
                             {"error", snap.nvidia.error},
                             {"index", snap.nvidia.index},
                             {"used_bytes", mib_to_bytes(snap.nvidia.used_mib)},
                             {"total_bytes", mib_to_bytes(snap.nvidia.total_mib)}};
    nlohmann::json req  = {{"done", snap.requests.done},
                           {"ttft_ms_mean", snap.requests.ttft_ms_mean},
                           {"decode_tok_s_mean", snap.requests.decode_tok_s_mean},
                           {"decode_tok_s_total", snap.requests.decode_tok_s_total},
                           {"prefill_tok_s_total", snap.requests.prefill_tok_s_total},
                           {"running_requests", snap.requests.running_requests},
                           {"clients_window_minutes", snap.requests.clients_window_minutes},
                           {"reuse_full_reset", snap.requests.reuse_full_reset},
                           {"reuse_append", snap.requests.reuse_append},
                           {"reuse_seed", snap.requests.reuse_seed},
                           {"last_reuse", snap.requests.last_reuse},
                           {"log_available", snap.requests.log_available},
                           {"log_error", snap.requests.log_error},
                           {"mtp_backend", snap.requests.mtp_backend},
                           {"mtp_draft_window", snap.requests.mtp_draft_window},
                           {"mtp_drafted", snap.requests.mtp_drafted},
                           {"mtp_accepted", snap.requests.mtp_accepted},
                           {"mtp_fallback_steps", snap.requests.mtp_fallback_steps},
                           {"mtp_rounds", snap.requests.mtp_rounds},
                           {"mtp_accepted_per_position", snap.requests.mtp_accepted_per_position},
                           {"mtp_last_accept_rate", snap.requests.mtp_last_accept_rate}};
    nlohmann::json clients = nlohmann::json::array();
    for (const auto& c : snap.requests.clients) {
        clients.push_back({{"name", c.name},
                           {"requests", c.requests},
                           {"from_root", c.from_root},
                           {"prompt_tokens", c.prompt_tokens},
                           {"refill_tokens", c.refill_tokens},
                           {"reuse_percent", c.reuse_percent()},
                           {"ttft_ms_mean", c.ttft_ms_mean()},
                           {"tool_count", c.tool_count},
                           {"last_seen_ms", c.last_seen_ms}});
    }
    req["clients"] = std::move(clients);
    nlohmann::json health = {{"status", snap.health_status}, {"body", snap.health_body}};
    // Health observation and engine-state transitions are driven by the
    // Collector's 1 Hz observe_loop, not from here. Doing it here as well made
    // note_health_fail() count twice per second whenever a dashboard was open,
    // which halves the effective health-fail threshold.
    const std::string log = child_.log_tail(16 * 1024);
    std::string cap       = extract_kv_capacity_line(log);
    if (cap.empty()) { cap = snap.engine_capacity_line; }
    nlohmann::json insights = collector_.insights_report();
    return {{"monitor_only", !manages_engine_process(cfg_)},
            {"engine", std::move(engine)},
            {"dxgi", std::move(dxgi)},
            {"nvidia_smi", std::move(nvidia)},
            {"gpu_processes", snap.gpu_processes},
            {"reserve_budget", child_.reserve_budget()},
            {"desktop_reserve", {{"next_gib", desktop_reserve_launch_mib(child_.config().engine.args, st.desktop_reserve_gib) / 1024.0},
                                  {"pinned", desktop_reserve_pinned(child_.config().engine.args)}}},
            {"engine_capacity_line", cap},
            {"admin_vram", snap.admin_vram},
            {"admin_vram_note", snap.admin_vram_note},
            {"vram_control", collector_.vram_control_json()},
            {"requests", std::move(req)},
            {"insights", insights},
            {"series", collector_.series_json()},
            {"throughput", collector_.throughput_series_json()},
            {"health", std::move(health)},
            {"log_tail", log}};
}

void DashboardServer::stop() {
    stop_ = true;
    if (server_ != nullptr) { static_cast<httplib::Server*>(server_)->stop(); }
}

void DashboardServer::run() {
    httplib::Server svr;
    server_ = &svr;
    // No Access-Control-Allow-Origin. The custom mutating header is a CSRF
    // brake only because cross-origin preflight then fails closed.
    svr.set_pre_routing_handler([this](const httplib::Request& req, httplib::Response& res) {
        if (!host_header_allowed(req.get_header_value("Host"), cfg_.port, cfg_.host, cfg_.bind_any)) {
            res.status = 403;
            res.set_content(nlohmann::json{{"error", "host not allowed"}}.dump(), "application/json");
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });
    svr.Get("/", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(std::string(kDashboardHtml), "text/html; charset=utf-8");
    });
    svr.Get("/api/state", [this](const httplib::Request&, httplib::Response& res) {
        res.set_content(state_json().dump(), "application/json");
    });
    svr.Get("/api/insights", [this](const httplib::Request&, httplib::Response& res) {
        (void)collector_.snapshot();
        res.set_content(collector_.insights_report().dump(), "application/json");
    });
    svr.Get("/api/events", [this](const httplib::Request&, httplib::Response& res) {
        res.set_header("Cache-Control", "no-cache");
        res.set_header("Connection", "keep-alive");
        res.set_chunked_content_provider("text/event-stream", [this](std::size_t, httplib::DataSink& sink) {
            if (stop_.load()) {
                sink.done();
                return false;
            }
            const std::string payload = "data: " + state_json().dump() + "\n\n";
            sink.write(payload.data(), payload.size());
            for (int i = 0; i < 10 && !stop_.load(); ++i) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            return !stop_.load();
        });
    });
    auto control = [this](const httplib::Request& req, httplib::Response& res, auto fn) {
        if (!control_allowed(req.remote_addr)) {
            res.status = 403;
            res.set_content(nlohmann::json{{"error", "control is loopback-only"}}.dump(),
                            "application/json");
            return;
        }
        if (!supervisor_control_header_ok(req.get_header_value(std::string(kSupervisorControlHeader)))) {
            res.status = 403;
            res.set_content(nlohmann::json{{"error", "missing X-NInfer-Supervisor header"}}.dump(),
                            "application/json");
            return;
        }
        if (!manages_engine_process(cfg_)) {
            res.status = 409;
            res.set_content(nlohmann::json{{"error", "engine is unmanaged"}}.dump(),
                            "application/json");
            return;
        }
        fn();
        res.set_content(state_json().dump(), "application/json");
    };
    svr.Post("/api/start", [this, control](const httplib::Request& req, httplib::Response& res) {
        control(req, res, [this] { child_.start(); });
    });
    svr.Post("/api/stop", [this, control](const httplib::Request& req, httplib::Response& res) {
        control(req, res, [this] { child_.stop(); });
    });
    svr.Post("/api/restart", [this, control](const httplib::Request& req, httplib::Response& res) {
        control(req, res, [this] { child_.restart(); });
    });

    svr.Post("/api/desktop-reserve", [this](const httplib::Request& req, httplib::Response& res) {
        auto error = [&res](int status, const char* message) {
            res.status = status;
            res.set_content(nlohmann::json{{"error", message}}.dump(), "application/json");
        };
        if (!control_allowed(req.remote_addr) || !supervisor_control_header_ok(req.get_header_value(std::string(kSupervisorControlHeader)))) {
            error(403, "Reserve changes require a local supervisor request."); return;
        }
        if (!manages_engine_process(cfg_)) { error(409, "This supervisor does not manage the engine."); return; }
        const auto body = nlohmann::json::parse(req.body, nullptr, false);
        if (!body.is_object() || !body.contains("gib") || !body["gib"].is_number_integer() || body["gib"] < 0 || body["gib"] > 64) {
            error(400, "Choose engine default (0), or a whole number from 1 to 64 GiB."); return;
        }
        if (desktop_reserve_pinned(child_.config().engine.args)) { error(409, "Reserve is pinned by --desktop-reserve-mib in the launch configuration."); return; }
        const auto budget = child_.reserve_budget();
        if (!budget.value("ok", false)) { error(409, "Wait for a ready engine with the saved model settings before changing the reserve."); return; }
        const int gib = body["gib"].get<int>();
        if ((gib == 0 ? 8 : gib) > budget.value("max_gib", 0)) { error(409, "That reserve would leave too little memory for this model and its configured capacity. Choose a value within the current slider limit."); return; }
        if (!child_.save_desktop_reserve_gib(gib)) { error(409, "Could not save: the available model budget changed or the preferences folder is not writable. Refresh and try again."); return; }
        res.set_content(state_json().dump(), "application/json");
    });

    // Reading the configuration is gated exactly like changing it. The payload
    // describes how to reach the engine and what it was launched with, which is
    // not something to hand to any page that can reach the port.
    svr.Get("/api/config", [this](const httplib::Request& req, httplib::Response& res) {
        if (!control_allowed(req.remote_addr)) {
            res.status = 403;
            res.set_content(nlohmann::json{{"error", "config is loopback-only"}}.dump(),
                            "application/json");
            return;
        }
        res.set_content(config_json().dump(), "application/json");
    });

    svr.Post("/api/config", [this](const httplib::Request& req, httplib::Response& res) {
        if (!control_allowed(req.remote_addr)) {
            res.status = 403;
            res.set_content(nlohmann::json{{"error", "config is loopback-only"}}.dump(),
                            "application/json");
            return;
        }
        if (!supervisor_control_header_ok(
                req.get_header_value(std::string(kSupervisorControlHeader)))) {
            res.status = 403;
            res.set_content(nlohmann::json{{"error", "missing X-NInfer-Supervisor header"}}.dump(),
                            "application/json");
            return;
        }
        const ConfigResult result = apply_config(req.body);
        res.status                = result.status;
        res.set_content(result.body.dump(), "application/json");
    });

    // The catalog, with each artifact checked as the request is served rather
    // than at load. A model whose file has been moved or is still copying should
    // read as unavailable now, not at the next restart when the engine fails to
    // start and trips the crash-loop breaker.
    svr.Get("/api/models", [this](const httplib::Request& req, httplib::Response& res) {
        if (!control_allowed(req.remote_addr)) {
            res.status = 403;
            res.set_content(nlohmann::json{{"error", "models are loopback-only"}}.dump(),
                            "application/json");
            return;
        }
        res.set_content(models_json().dump(), "application/json");
    });

    svr.Post("/api/model", [this](const httplib::Request& req, httplib::Response& res) {
        if (!control_allowed(req.remote_addr)) {
            res.status = 403;
            res.set_content(nlohmann::json{{"error", "models are loopback-only"}}.dump(),
                            "application/json");
            return;
        }
        if (!supervisor_control_header_ok(
                req.get_header_value(std::string(kSupervisorControlHeader)))) {
            res.status = 403;
            res.set_content(nlohmann::json{{"error", "missing X-NInfer-Supervisor header"}}.dump(),
                            "application/json");
            return;
        }
        const ConfigResult result = select_model(req.body);
        res.status                = result.status;
        res.set_content(result.body.dump(), "application/json");
    });

    const std::string host = cfg_.bind_any ? "0.0.0.0" : cfg_.host;
    if (cfg_.bind_any || !is_loopback_host(host)) {
        std::cerr << "WARNING: ninfer-supervisor is binding " << host
                  << " — control endpoints start/stop the engine. Loopback is the default.\n";
    }
    // listen() returns false immediately when the bind fails, and true only after
    // stop(). Ignoring it is how a second supervisor ended up with a silently
    // dead dashboard while still fighting for the engine port.
    if (!svr.listen(host, cfg_.port) && !stop_.load()) { listen_failed_ = true; }
    server_ = nullptr;
}

} // namespace ninfer::supervisor
