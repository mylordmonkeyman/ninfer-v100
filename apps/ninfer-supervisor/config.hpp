#pragma once

#include "logic.hpp"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <filesystem>
#include <system_error>
#include <string>
#include <vector>

namespace ninfer::supervisor {

struct EngineSpec {
    std::string executable;
    std::vector<std::string> args;
    std::string workdir;
    std::string api_key_file;
    std::string engine_host = "127.0.0.1";
    int engine_port         = 8010;
    std::string request_log;
    int device      = 0;
    bool unmanaged  = false; // observe an engine this process did not spawn
};

// A wildcard is a listening address, not a destination for the supervisor's
// health and memory requests. Keep the configured engine binding unchanged.
inline std::string engine_connect_host(const EngineSpec& spec) {
    if (spec.engine_host == "0.0.0.0") { return "127.0.0.1"; }
    if (spec.engine_host == "::") { return "::1"; }
    return spec.engine_host;
}

// One servable model. `args` is everything the engine needs BESIDES the artifact
// path, because switching models is not a path swap: the 27B wants
// --kv-dtype int8 --spec mtp --draft-tokens 4, Flash-Next wants
// --kv-dtype fp8 --gdn-state-dtype bf16 and a different --kv-capacity. A catalog
// that carried only paths would produce a configuration that does not start.
struct ModelEntry {
    std::string id;
    std::string artifact;
    std::vector<std::string> args;
    std::string name;
    std::string description;
};

// Inspect only the small directory, never model payloads. The engine remains
// responsible for complete artifact binding and numerical compatibility.
inline std::string artifact_model_identity(const std::string& path) {
    try {
        std::ifstream in(std::filesystem::path(path), std::ios::binary);
        unsigned char prefix[16]{};
        in.read(reinterpret_cast<char*>(prefix), sizeof(prefix));
        if (!in || std::memcmp(prefix, "NINFER\0\2", 8) != 0) { return {}; }
        std::uint64_t size = 0;
        for (unsigned i = 0; i < 8; ++i) { size |= std::uint64_t(prefix[8 + i]) << (8 * i); }
        if (size == 0 || size > 16 * 1024 * 1024) { return {}; }
        std::string directory(static_cast<std::size_t>(size), '\0');
        in.read(directory.data(), static_cast<std::streamsize>(size));
        if (!in) { return {}; }
        return nlohmann::json::parse(directory).at("identity").at("model_id").get<std::string>();
    } catch (const std::exception&) { return {}; }
}

// What a model's artifact looks like on disk right now. Availability is checked
// rather than assumed: a catalog entry pointing at a moved or half-copied file
// should say so in the dashboard, not at the next restart when the engine fails
// to come up.
struct ModelAvailability {
    bool available = false;
    std::string reason;       // empty when available
    std::uint64_t size_bytes = 0;
};

// Reads the artifact header. `NINFER\0\2` is the v2 artifact magic (see
// src/artifact/reader.cpp), so a file that exists but is a partial copy, the
// wrong format, or a stray rename is caught here instead of costing a failed
// engine start and a crash-loop backoff.
inline ModelAvailability check_model_available(const ModelEntry& model) {
    ModelAvailability out;
    if (model.artifact.empty()) {
        out.reason = "no artifact path configured";
        return out;
    }
    std::error_code ec;
    const std::filesystem::path path(model.artifact);
    if (!std::filesystem::exists(path, ec) || ec) {
        out.reason = "artifact not found";
        return out;
    }
    if (!std::filesystem::is_regular_file(path, ec) || ec) {
        out.reason = "artifact path is not a file";
        return out;
    }
    out.size_bytes = static_cast<std::uint64_t>(std::filesystem::file_size(path, ec));
    if (ec) {
        out.size_bytes = 0;
        out.reason     = "artifact size is unreadable";
        return out;
    }
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        out.reason = "artifact cannot be opened for reading";
        return out;
    }
    char magic[8] = {};
    in.read(magic, sizeof(magic));
    if (in.gcount() != static_cast<std::streamsize>(sizeof(magic))) {
        out.reason = "artifact is truncated";
        return out;
    }
    static constexpr char kNinferV2[8] = {'N', 'I', 'N', 'F', 'E', 'R', '\0', '\2'};
    if (std::memcmp(magic, kNinferV2, sizeof(magic)) != 0) {
        out.reason = "not an NInfer v2 artifact";
        return out;
    }
    out.available = true;
    return out;
}

struct SupervisorConfig {
    EngineSpec engine;
    // Optional catalog. Empty means the engine's own args are the only
    // configuration, which is how every config before this worked and still does.
    std::vector<ModelEntry> models;
    std::string active_model;
    // Where this config was loaded from. The dashboard writes edits back here, so
    // it must be the resolved path rather than whatever relative string the CLI
    // was given -- the supervisor's working directory is not the user's.
    std::string source_path;
    // The file's write stamp as of the last load or save by this process. A
    // mismatch at save time means someone edited the file while we held it, and
    // writing our copy would silently revert their change.
    std::int64_t source_stamp = 0;
    std::string host = "127.0.0.1";
    int port         = 8099;
    bool bind_any    = false;
    bool monitor_only = false; // never spawn/stop/restart; HTTP observe only
    std::string logs_dir;
    bool run_at_login = false;
    RestartPolicy restart;
};

inline bool manages_engine_process(const SupervisorConfig& cfg) noexcept {
    return !cfg.monitor_only && !cfg.engine.unmanaged;
}

inline std::string read_file_text(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { throw std::runtime_error("cannot read " + path); }
    std::ostringstream out;
    out << in.rdbuf();
    return out.str();
}

inline std::string read_api_key(const std::string& path) {
    if (path.empty()) { return {}; }
    std::string raw = read_file_text(path);
    while (!raw.empty() && (raw.back() == '\n' || raw.back() == '\r' || raw.back() == ' ' ||
                            raw.back() == '\t')) {
        raw.pop_back();
    }
    return raw;
}

// Serializes a config back to the on-disk shape. Round-trips the fields the loader
// reads and nothing else, so a hand-written file keeps its meaning after an edit
// made through the dashboard.
//
// The API key is not here and never will be: it lives in the file named by
// api_key_file, and the supervisor only ever reads that file to talk to the engine.
inline nlohmann::json config_to_json(const SupervisorConfig& cfg) {
    nlohmann::json engine = {
        {"executable", cfg.engine.executable},
        {"args", cfg.engine.args},
        {"workdir", cfg.engine.workdir},
        {"api_key_file", cfg.engine.api_key_file},
        {"engine_host", cfg.engine.engine_host},
        {"engine_port", cfg.engine.engine_port},
        {"request_log", cfg.engine.request_log},
        {"device", cfg.engine.device},
        {"unmanaged", cfg.engine.unmanaged},
    };
    nlohmann::json supervisor = {
        {"host", cfg.host},
        {"port", cfg.port},
        {"bind_any", cfg.bind_any},
        {"monitor_only", cfg.monitor_only},
        {"logs_dir", cfg.logs_dir},
        {"run_at_login", cfg.run_at_login},
        {"restart",
         {{"max_backoff_s", cfg.restart.max_backoff_s},
          {"crash_loop_window_s", cfg.restart.crash_loop_window_s},
          {"crash_loop_max", cfg.restart.crash_loop_max},
          {"health_fail_threshold", cfg.restart.health_fail_threshold}}},
    };
    nlohmann::json out = {{"engine", engine}, {"supervisor", supervisor}};
    // Only written when present, so a config that never used a catalog is not
    // rewritten with empty keys the operator did not ask for.
    if (!cfg.models.empty()) {
        nlohmann::json models = nlohmann::json::array();
        for (const auto& m : cfg.models) {
            models.push_back({{"id", m.id}, {"artifact", m.artifact}, {"args", m.args},
                              {"name", m.name}, {"description", m.description}});
        }
        out["models"] = models;
        if (!cfg.active_model.empty()) { out["active_model"] = cfg.active_model; }
    }
    return out;
}

// The full command line for a catalog entry: the artifact first, because the
// engine takes it positionally, then that model's own flags.
inline std::vector<std::string> model_engine_args(const ModelEntry& model) {
    std::vector<std::string> args;
    args.reserve(model.args.size() + 1);
    args.push_back(model.artifact);
    args.insert(args.end(), model.args.begin(), model.args.end());
    return args;
}

inline const ModelEntry* find_model(const SupervisorConfig& cfg, const std::string& id) {
    for (const auto& m : cfg.models) {
        if (m.id == id) { return &m; }
    }
    return nullptr;
}

// When the config file was last written, as far as this process knows. Used to
// notice that somebody edited it underneath us.
inline std::int64_t config_file_stamp(const std::string& path) {
    if (path.empty()) { return 0; }
    std::error_code ec;
    const auto t = std::filesystem::last_write_time(path, ec);
    if (ec) { return 0; }
    return static_cast<std::int64_t>(t.time_since_epoch().count());
}

// Write to a sibling temp file, then rename over the original. A half-written
// config is a supervisor that will not start at the next login, and the edit that
// produces it is made from a browser where a refresh mid-write is normal.
//
// `expected_stamp` guards against clobbering an external edit. The supervisor
// holds the config in memory from startup and writes that copy back on every
// save, so a file edited while it runs is silently reverted at the next save --
// which is exactly what happened while building the model catalog: a corrected
// model entry was overwritten by the running supervisor's stale copy, and the
// resulting failures looked like bad flags rather than a lost edit. Pass 0 to
// skip the check.
inline void save_config_json(const std::string& path, const SupervisorConfig& cfg,
                             std::int64_t expected_stamp = 0) {
    if (path.empty()) { throw std::runtime_error("no config path to write to"); }
    if (expected_stamp != 0) {
        const std::int64_t actual = config_file_stamp(path);
        if (actual != 0 && actual != expected_stamp) {
            throw std::runtime_error(
                "config file changed on disk since it was loaded; refusing to overwrite it. "
                "Restart the supervisor to pick the edit up, or undo the external change.");
        }
    }
    const std::string tmp = path + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) { throw std::runtime_error("cannot write " + tmp); }
        out << config_to_json(cfg).dump(2) << "\n";
        if (!out) { throw std::runtime_error("cannot write " + tmp); }
    }
    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    if (ec) {
        std::filesystem::remove(tmp, ec);
        throw std::runtime_error("cannot replace " + path);
    }
}

// Same as read_api_key but never throws: used on presentation paths where a
// missing or unreadable key file is a fact to report, not an error to raise.
inline std::string read_api_key_quiet(const std::string& path) noexcept {
    try {
        return read_api_key(path);
    } catch (...) {
        return {};
    }
}

inline SupervisorConfig load_config_json(const std::string& json_text,
                                         bool monitor_only_cli = false) {
    const auto body = nlohmann::json::parse(json_text);
    SupervisorConfig cfg;
    if (body.contains("engine") && body.at("engine").is_object()) {
        const auto& e = body.at("engine");
        cfg.engine.executable   = e.value("executable", "");
        cfg.engine.workdir      = e.value("workdir", "");
        cfg.engine.api_key_file = e.value("api_key_file", "");
        cfg.engine.engine_host  = e.value("engine_host", "127.0.0.1");
        cfg.engine.engine_port  = e.value("engine_port", 8010);
        cfg.engine.request_log  = e.value("request_log", "");
        cfg.engine.device       = e.value("device", 0);
        cfg.engine.unmanaged    = e.value("unmanaged", false);
        if (e.contains("args") && e.at("args").is_array()) {
            for (const auto& a : e.at("args")) {
                if (a.is_string()) { cfg.engine.args.push_back(a.get<std::string>()); }
            }
        }
    }
    if (body.contains("models") && body.at("models").is_array()) {
        for (const auto& m : body.at("models")) {
            if (!m.is_object()) { continue; }
            ModelEntry entry;
            entry.id       = m.value("id", "");
            entry.artifact = m.value("artifact", "");
            entry.name = m.value("name", "");
            entry.description = m.value("description", "");
            if (m.contains("args") && m.at("args").is_array()) {
                for (const auto& a : m.at("args")) {
                    if (a.is_string()) { entry.args.push_back(a.get<std::string>()); }
                }
            }
            // An entry with no id cannot be selected, and one with no artifact
            // cannot be launched. Dropping them keeps the catalog honest rather
            // than surfacing entries that can never work.
            if (!entry.id.empty() && !entry.artifact.empty()) {
                cfg.models.push_back(std::move(entry));
            }
        }
    }
    cfg.active_model = body.value("active_model", "");
    if (body.contains("supervisor") && body.at("supervisor").is_object()) {
        const auto& s = body.at("supervisor");
        cfg.host         = s.value("host", "127.0.0.1");
        cfg.port         = s.value("port", 8099);
        cfg.bind_any      = s.value("bind_any", false);
        cfg.monitor_only  = s.value("monitor_only", false) || monitor_only_cli;
        cfg.logs_dir      = s.value("logs_dir", "");
        cfg.run_at_login = s.value("run_at_login", false);
        if (s.contains("restart") && s.at("restart").is_object()) {
            const auto& r = s.at("restart");
            cfg.restart.max_backoff_s         = r.value("max_backoff_s", 60);
            cfg.restart.crash_loop_window_s   = r.value("crash_loop_window_s", 60);
            cfg.restart.crash_loop_max        = r.value("crash_loop_max", 5);
            cfg.restart.health_fail_threshold = r.value("health_fail_threshold", 3);
        }
    }
    if (monitor_only_cli) { cfg.monitor_only = true; }
    if (manages_engine_process(cfg) && cfg.engine.executable.empty()) {
        throw std::invalid_argument(
            "engine.executable is required unless monitor_only or engine.unmanaged");
    }
    if (!cfg.bind_any && !is_loopback_host(cfg.host)) {
        throw std::invalid_argument(
            "supervisor host must be loopback unless bind_any is true");
    }
    return cfg;
}

} // namespace ninfer::supervisor
