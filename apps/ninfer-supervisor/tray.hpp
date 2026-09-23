#pragma once

#include "collector.hpp"
#include "config.hpp"
#include "engine_child.hpp"
#include "logic.hpp"
#include "tray_prefs.hpp"

#include <chrono>
#include <cstdint>
#include <functional>
#include <string>

namespace ninfer::supervisor {

// What the tray icon says at a glance. In monitor-only mode this still reflects
// the OBSERVED health of the engine rather than a neutral "not my process":
// EngineChild keeps st_.health current for unmanaged engines too, so the colour
// is measured, not invented. The tooltip carries the managed/observing
// distinction instead, because that belongs in words rather than in a hue.
enum class TrayStatus : std::uint8_t {
    Idle,     // grey  — nothing running, or nothing known yet
    Working,  // green — engine running and answering /health
    Pending,  // amber — starting, stopping, backing off, or unreachable
    Failed,   // red   — halted, crash-looped, or reporting unhealthy
};

class TrayIcon {
public:
    TrayIcon(EngineChild& child, Collector& collector, std::string dashboard_url,
             bool manages_engine, std::string prefs_path, std::string config_abs_path);
    ~TrayIcon();
    void run();
    void request_quit();
    void open_dashboard() const;
    void open_logs_folder() const;
    void copy_dashboard_url() const;
    EngineChild& child();

    // Optional: lets the tray say "dashboard port N is in use" instead of the
    // user finding a dead link. A predicate rather than a DashboardServer& so
    // this header does not have to pull in the 50 KB dashboard HTML blob.
    void set_dashboard_listen_failed(std::function<bool()> pred) {
        dashboard_listen_failed_ = std::move(pred);
    }

    // Repaints the tray icon when the status changes. Called on a timer; cheap
    // because it reads EngineChild's in-memory status and never polls the
    // engine or shells out to nvidia-smi.
    void refresh_icon();

    // Shell_NotifyIcon(NIM_ADD), reported rather than assumed. It fails when
    // this process beats Explorer to the desktop at login, and when Explorer has
    // just restarted; both are re-add situations, not modify situations.
    bool try_add_icon();
    void refresh_icon_add();

    // Idle unload, evaluated on the same timer. Stops the engine after the
    // configured quiet period so a workstation is not holding 70 GiB of weights
    // while nobody is using it. Reload happens on the next explicit start.
    void tick_idle_unload();

    // Read by the menu builder. Small accessors rather than friendship: the menu
    // lives in an anonymous namespace in the .cpp and only needs to read state
    // and hand back the choices a user can make.
    // Derived from the snapshot the menu already took, so the header dot and the
    // header text can never disagree about what the engine is doing.
    [[nodiscard]] TrayStatus menu_status(const EngineStatus& st) const { return status_from(st); }
    [[nodiscard]] bool manages_engine() const noexcept { return manages_engine_; }
    [[nodiscard]] Collector& collector() const noexcept { return collector_; }
    [[nodiscard]] int device_index() const noexcept { return child_.config().engine.device; }

    // The reserve the checkmark should sit on: the user's stored choice when
    // they have made one, otherwise whatever the config file already asks for.
    [[nodiscard]] int reserve_gib() const;
    [[nodiscard]] int idle_minutes() const noexcept { return prefs_.idle_unload_minutes; }
    [[nodiscard]] bool dashboard_unavailable() const;
    [[nodiscard]] std::wstring header_line(const EngineStatus& st) const;
    [[nodiscard]] std::wstring detail_line() const;

    // The config pins --desktop-reserve-mib, which the menu cannot express, so
    // every entry in the Reserve submenu would be a no-op. The menu greys it and
    // says so instead.
    [[nodiscard]] bool reserve_pinned_by_config() const;

    // "Start at login", answered for THIS config rather than for the single
    // shared Run value name.
    [[nodiscard]] bool start_at_login_enabled() const;

    void on_reserve_chosen(int gib);
    void on_idle_chosen(int minutes);
    void on_start_at_login_toggled();

    // One-shot balloon for the Run entry the config asked for at startup. A
    // Windows-subsystem process started from Explorer has nowhere to print, so
    // without this the supervisor silently makes itself start at every login.
    void note_login_installed() { login_install_pending_ = true; }

private:
    TrayStatus current_status() const;
    // The status derived from ONE snapshot. refresh_icon used to take two --
    // child_.status() and then current_status(), which calls it again -- so a
    // halt landing between them announced nothing and never re-announced.
    TrayStatus status_from(const EngineStatus& st) const;
    std::wstring tooltip(TrayStatus status, const EngineStatus& st) const;
    void set_idle_minutes(int minutes);
    // False when the shell did not take the balloon -- at login the icon may not
    // be added yet, and NIM_MODIFY on a missing icon fails.
    bool notify(const wchar_t* title, const std::wstring& body, unsigned niif_flags);
    void announce_transition(TrayStatus prev, TrayStatus next, const EngineStatus& st);

    EngineChild& child_;
    Collector& collector_;
    std::function<bool()> dashboard_listen_failed_;
    std::string dashboard_url_;
    bool manages_engine_ = true;
    std::string prefs_path_;
    std::string config_abs_path_;
    TrayPrefs prefs_;
    void* hwnd_      = nullptr;
    void* hicon_     = nullptr;
    int last_status_ = -1;
    // Last KV plan announced. A reduced plan is a compromise the person should hear about
    // once, from the shell, not discover on the dashboard.
    std::int64_t last_kv_effective_ = 0;
    // refresh_icon() early-outs on an unchanged key. The key has to include the
    // error text, or a tooltip that gained a reason without changing colour
    // would never be written to the shell.
    std::wstring last_tip_;
    // Readiness is announced as well as status, because a halt-to-recovery
    // transition can land entirely inside TrayStatus::Working: the enum turns
    // green when the process exists, a minute before it can answer anything.
    bool last_ready_             = false;
    bool halt_announced_         = false;
    bool listen_announced_       = false;
    bool login_install_pending_  = false;
};

// Device-wide memory via nvidia-smi, run hidden.
//
// Deliberately NOT cudaMemGetInfo and NOT the DXGI budget: both report an empty
// card while another process holds 70 GiB -- verified this session, and the cause
// of two incidents where the engine filled the card and the desktop stopped
// responding. Anything the tray says about memory has to be device-wide truth.
// Returns ok=false when the tool is missing, and the menu then omits the memory
// line rather than showing a wrong one.
//
// Only a fallback now: the menu reads Collector::last_nvidia(), which is the
// same measurement already taken at 1 Hz off the UI thread.
NvidiaSmiMemory query_device_memory_smi(int device);

} // namespace ninfer::supervisor
