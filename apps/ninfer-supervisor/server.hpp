#pragma once

#include "collector.hpp"
#include "dashboard.hpp"
#include "engine_child.hpp"

#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <thread>

namespace ninfer::supervisor {

class DashboardServer {
public:
    DashboardServer(SupervisorConfig cfg, EngineChild& child, Collector& collector);
    ~DashboardServer();

    void run();
    void stop();

    // True once listen() has come back having failed to bind. A second
    // supervisor on the same port otherwise runs with no dashboard at all and
    // says nothing about it; the tray shows this as a disabled menu line.
    [[nodiscard]] bool listen_failed() const noexcept { return listen_failed_.load(); }

private:
    nlohmann::json state_json();
    // The editable surface, with the parameter table's own descriptions so the
    // page draws the form from the engine's vocabulary rather than a second copy
    // of it in JavaScript that drifts.
    nlohmann::json config_json() const;
    // Validates, writes the config file, and reports what still has to happen for
    // the change to take effect. Never applies a partially valid edit. Returns the
    // HTTP status and body rather than taking a response, so this header stays free
    // of httplib -- the same reason server_ is a void*.
    struct ConfigResult {
        int status = 200;
        nlohmann::json body;
    };
    ConfigResult apply_config(const std::string& request_body);
    // The configured model catalog with each artifact's availability checked now.
    [[nodiscard]] nlohmann::json models_json() const;
    // Switches the served model: validates the artifact, rewrites the engine
    // args, saves, and restarts. Loading a different model requires tearing down
    // the old one's device state, so a restart is the mechanism, not a side
    // effect.
    ConfigResult select_model(const std::string& request_body);
    // Blocks until the engine answers /health, the crash-loop breaker halts, or
    // the limit passes. Used on both the switch and the rollback so a reported
    // "serving" is observed rather than merely requested.
    [[nodiscard]] bool wait_until_serving(std::chrono::seconds limit, int replaced_pid);
    bool control_allowed(const std::string& remote) const;

    SupervisorConfig cfg_;
    EngineChild& child_;
    Collector& collector_;
    std::atomic<bool> stop_{false};
    std::atomic<bool> listen_failed_{false};
    void* server_ = nullptr; // httplib::Server*
};

} // namespace ninfer::supervisor
