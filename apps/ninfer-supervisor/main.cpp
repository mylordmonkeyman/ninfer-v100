#include "collector.hpp"
#include "config.hpp"
#include "engine_child.hpp"
#include "logic.hpp"
#include "run_at_login.hpp"
#include "server.hpp"
#include "tray.hpp"
#include "tray_prefs.hpp"
#include "reserve_budget.hpp"
#include "desktop_launch.hpp"

#include <windows.h>
#include <shellapi.h>

#include <cstdio>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

namespace {

std::string state_label(ninfer::supervisor::EngineState s) {
    using ninfer::supervisor::EngineState;
    switch (s) {
    case EngineState::Stopped: return "Stopped";
    case EngineState::Starting: return "Starting";
    case EngineState::Running: return "Running";
    case EngineState::Stopping: return "Stopping";
    case EngineState::BackingOff: return "BackingOff";
    case EngineState::Halted: return "Halted";
    }
    return "Stopped";
}

void usage() {
    std::cout
        << "usage: ninfer-supervisor --config FILE [--host 127.0.0.1] [--port 8099] [--bind-any]\n"
           "                         [--monitor-only] [--install-login] [--uninstall-login]\n"
           "                         [--background]\n"
           "  --background asks Windows Explorer to launch independently of this terminal.\n"
           "  Dashboard binds loopback by default. --bind-any is required for 0.0.0.0 and prints\n"
           "  a warning. Control POST /api/start|stop|restart is loopback-peer only, requires\n"
           "  header X-NInfer-Supervisor: 1, and returns 409 in --monitor-only / unmanaged mode.\n"
           "  Host is allowlisted on every route. Do not send CORS headers.\n";
}

} // namespace

// A tray application has no business conjuring a console window. The target is
// linked for the Windows subsystem so nothing appears when the supervisor starts
// at login, which is how it usually starts. But this is also a small CLI --
// --install-login, --uninstall-login and usage all print -- so when it IS run
// from a terminal, adopt that terminal and send stdout/stderr back to it.
// Without this the Windows-subsystem build would silently swallow those.
void attach_parent_console_if_any() {
    // Leave an already-connected stdout alone. A Windows-subsystem process gets
    // null standard handles when launched from Explorer or the Run key, but real
    // ones when the caller redirected to a file or a pipe -- and reopening
    // CONOUT$ in that case would write past the redirection to the console,
    // quietly breaking `ninfer-supervisor --help > file`.
    const HANDLE existing = GetStdHandle(STD_OUTPUT_HANDLE);
    if (existing != nullptr && existing != INVALID_HANDLE_VALUE) { return; }
    if (AttachConsole(ATTACH_PARENT_PROCESS) == 0) { return; }
    FILE* out = nullptr;
    (void)freopen_s(&out, "CONOUT$", "w", stdout);
    FILE* err = nullptr;
    (void)freopen_s(&err, "CONOUT$", "w", stderr);
    std::ios::sync_with_stdio(true);
}

int main(int argc, char** argv) {
    attach_parent_console_if_any();
    try {
        std::string config_path;
        std::string host_override;
        int port_override = -1;
        bool bind_any     = false;
        bool monitor_only = false;
        bool install      = false;
        bool uninstall    = false;
        bool background   = false;
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            auto need           = [&](const char* name) -> const char* {
                if (i + 1 >= argc) { throw std::invalid_argument(std::string("missing ") + name); }
                return argv[++i];
            };
            if (a == "--help" || a == "-h") {
                usage();
                return 0;
            } else if (a == "--config") {
                config_path = need("--config");
            } else if (a == "--host") {
                host_override = need("--host");
            } else if (a == "--port") {
                port_override = std::stoi(need("--port"));
            } else if (a == "--bind-any") {
                bind_any = true;
            } else if (a == "--monitor-only") {
                monitor_only = true;
            } else if (a == "--install-login") {
                install = true;
            } else if (a == "--uninstall-login") {
                uninstall = true;
            } else if (a == "--background") {
                background = true;
            } else {
                throw std::invalid_argument("unknown argument: " + a);
            }
        }
        if (uninstall) {
            ninfer::supervisor::uninstall_run_at_login();
            std::cout << "removed HKCU Run\\NInferSupervisor\n";
            return 0;
        }
        if (config_path.empty()) {
            usage();
            return 2;
        }
        // The Run key and the single-instance mutex both key on the config path,
        // so it has to be the same string however the supervisor was launched --
        // a relative path in the Run entry resolves against Explorer's working
        // directory, which is not the one the user typed it in.
        std::error_code path_ec;
        auto canonical = std::filesystem::weakly_canonical(config_path, path_ec);
        const std::string config_abs =
            path_ec ? config_path : canonical.generic_string();
        auto cfg = ninfer::supervisor::load_config_json(
            ninfer::supervisor::read_file_text(config_path), monitor_only);
        // Resolved, because the dashboard writes edits back to this path and the
        // supervisor's working directory is not the one the CLI was invoked from.
        {
            std::error_code ec;
            const auto resolved = std::filesystem::weakly_canonical(config_path, ec);
            cfg.source_path     = ec ? config_path : resolved.string();
            // Remembered so a later save can tell that somebody edited the
            // file while the supervisor held it in memory.
            cfg.source_stamp =
                ninfer::supervisor::config_file_stamp(cfg.source_path);
        }
        if (!host_override.empty()) { cfg.host = host_override; }
        if (port_override > 0) { cfg.port = port_override; }
        if (bind_any) { cfg.bind_any = true; }
        if (monitor_only) { cfg.monitor_only = true; }
        if (cfg.bind_any) {
            std::cerr << "WARNING: binding beyond loopback; engine start/stop is exposed on "
                      << (cfg.host.empty() ? "0.0.0.0" : cfg.host) << ":" << cfg.port << "\n";
        } else if (!ninfer::supervisor::is_loopback_host(cfg.host)) {
            throw std::invalid_argument("--host must be loopback without --bind-any");
        }
        if (background) {
            std::string args = "--config \"" + config_abs + "\"";
            if (!host_override.empty()) { args += " --host " + host_override; }
            if (port_override > 0) { args += " --port " + std::to_string(port_override); }
            if (bind_any) { args += " --bind-any"; }
            if (monitor_only) { args += " --monitor-only"; }
            if (install) { args += " --install-login"; }
            ninfer::supervisor::launch_from_desktop(
                ninfer::supervisor::widen_utf8(ninfer::supervisor::module_path_utf8()),
                ninfer::supervisor::widen_utf8(args),
                std::filesystem::path(config_abs).parent_path().wstring());
            return 0;
        }
        const std::string login_cmd = ninfer::supervisor::run_at_login_command(
            ninfer::supervisor::module_path_utf8(), config_abs);
        bool announced_login_install = false;
        if (install) {
            ninfer::supervisor::install_run_at_login(login_cmd);
            std::cout << "installed HKCU Run\\NInferSupervisor\n";
        } else if (cfg.run_at_login && !ninfer::supervisor::run_at_login_installed()) {
            // The config field is a bootstrap hint, not a live setting: it
            // installs the Run entry once, and the tray's checkbox owns it after
            // that. Parsing it and never acting on it was the previous state.
            //
            // Only when NO entry exists at all -- not merely when this config's
            // is missing -- so a second configuration cannot silently take over
            // another one's autostart. And it is announced: this is a change to
            // the machine, the binary is Windows-subsystem, and a user who
            // started it from Explorer would otherwise get no indication that
            // launching it once made it start at every login. The console line
            // covers a terminal launch, the tray balloon covers every other.
            ninfer::supervisor::install_run_at_login(login_cmd);
            announced_login_install = true;
            std::cout << "ninfer-supervisor: run_at_login is set in " << config_abs
                      << ", so HKCU Run\\NInferSupervisor now starts it at login. "
                         "Remove it with --uninstall-login or the tray's \"Start at login\".\n";
        }

        const std::string url =
            "http://" + (cfg.bind_any ? std::string("127.0.0.1") : cfg.host) + ":" +
            std::to_string(cfg.port) + "/";

        // One supervisor per config. Two of them race for the same engine port:
        // the loser's engine cannot bind, crash-loops, and trips the breaker,
        // which looks exactly like an engine bug. Held for the process lifetime.
        const std::wstring mutex_name =
            ninfer::supervisor::widen_utf8(ninfer::supervisor::single_instance_name(config_abs));
        HANDLE instance_mutex = CreateMutexW(nullptr, TRUE, mutex_name.c_str());
        if (instance_mutex != nullptr && GetLastError() == ERROR_ALREADY_EXISTS) {
            std::cerr << "ninfer-supervisor: already running for " << config_abs << "\n";
            ShellExecuteA(nullptr, "open", url.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
            MessageBoxW(nullptr,
                        L"NInfer supervisor is already running for this configuration.\n"
                        L"Its dashboard has been opened instead.",
                        L"NInfer", MB_OK | MB_ICONINFORMATION);
            CloseHandle(instance_mutex);
            return 3;
        }

        ninfer::supervisor::EngineChild child(cfg);
        // Before run_loop's first spawn, not in the TrayIcon constructor: the
        // engine thread starts below and would otherwise launch once with the
        // config's reserve and only pick up the stored choice on a later restart.
        const std::string prefs_path = ninfer::supervisor::tray_prefs_path(config_abs);
        child.set_desktop_reserve_gib(
            ninfer::supervisor::load_tray_prefs(prefs_path).desktop_reserve_gib);
        ninfer::supervisor::Collector collector(cfg.engine, cfg.logs_dir);
        child.set_reserve_budget_provider([&collector, &child, probe = cfg.engine] {
            const auto snapshot = collector.snapshot();
            const auto current = child.config().engine;
            if (!child.reserve_plan_matches_config() || snapshot.health_status != 200 ||
                probe.device != current.device || probe.engine_host != current.engine_host || probe.engine_port != current.engine_port) {
                return nlohmann::json{{"ok", false}, {"reason", "A ready engine with the saved model settings is needed to calculate the reserve limit."}};
            }
            return ninfer::supervisor::model_reserve_budget(snapshot.admin_vram, current.args);
        });
        // Wired BEFORE start_series: the 1 Hz observe thread reads them, and
        // this is what makes health and engine-state observation independent of
        // whether anybody has the dashboard open.
        collector.set_health_observer([&child](int http_status) { child.observe_health(http_status); });
        collector.set_engine_state_provider([&child] {
            const auto st = child.status();
            return std::pair<std::string, std::string>(state_label(st.state), st.last_event);
        });
        collector.start_series();
        ninfer::supervisor::DashboardServer server(cfg, child, collector);
        std::thread engine_thread([&] { child.run_loop(); });
        std::thread http_thread([&] { server.run(); });
        std::cout << "ninfer-supervisor dashboard " << url << "\n";
        ninfer::supervisor::TrayIcon tray(child, collector, url,
                                          ninfer::supervisor::manages_engine_process(cfg),
                                          prefs_path, config_abs);
        tray.set_dashboard_listen_failed([&server] { return server.listen_failed(); });
        if (announced_login_install) { tray.note_login_installed(); }
        tray.run();
        server.stop();
        collector.stop_series();
        child.request_quit();
        child.stop();
        if (http_thread.joinable()) { http_thread.join(); }
        if (engine_thread.joinable()) { engine_thread.join(); }
        if (instance_mutex != nullptr) {
            ReleaseMutex(instance_mutex);
            CloseHandle(instance_mutex);
        }
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "ninfer-supervisor: " << ex.what() << "\n";
        return 1;
    }
}
