#pragma once

#include "config.hpp"

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace ninfer::supervisor {

enum class EngineState : std::uint8_t {
    Stopped,
    Starting,
    Running,
    Stopping,
    BackingOff,
    Halted,
};

struct EngineStatus {
    EngineState state          = EngineState::Stopped;
    std::uint64_t pid          = 0;
    std::int64_t started_unix_ms = 0;
    int restart_count          = 0;
    int last_exit_code         = 0;
    bool crash_loop_halted     = false;
    std::string last_event;
    std::string health; // ok / unhealthy / unreachable
    int health_fails           = 0;

    // Why, not just what. The tray's whole failure mode so far has been an amber
    // dot with no explanation; these come from scanning the engine's own stderr,
    // which is read for engine.log anyway.
    std::string last_error_line;
    std::int64_t last_error_unix_ms    = 0;
    std::int64_t last_activity_unix_ms = 0;
    bool ready                         = false; // engine printed `server status=ready`
    int recent_exits                   = 0;
    int crash_window_s                 = 0;
    int desktop_reserve_gib            = 0; // effective at the last spawn; <0 = config default
    // Requests the engine opened and has not finished, counted from its own
    // stderr. Non-zero means "do not unload", whatever the activity clock says.
    int inflight_requests              = 0;
    // KV plan actually launched. effective is 0 unless the supervisor shrank the configured
    // value to fit the runtime reservation the engine could get; note says why.
    std::int64_t kv_capacity_configured = 0;
    std::int64_t kv_capacity_effective  = 0;
    std::string kv_capacity_note;
    // The argument vector the running process was actually started with, after every
    // supervisor adjustment (reserve, request log, key file, KV capacity). This, not the
    // config file, is what the engine is running.
    std::vector<std::string> launch_args;
};

class EngineChild {
public:
    explicit EngineChild(SupervisorConfig cfg);
    ~EngineChild();

    EngineChild(const EngineChild&)            = delete;
    EngineChild& operator=(const EngineChild&) = delete;

    void start();
    void stop();
    void restart();
    void request_quit();
    void observe_health(int http_status);

    // The reserve the tray last chose, applied to the argument vector at the
    // NEXT spawn. This lives here rather than in the tray because EngineChild
    // owns a *copy* of the config: mutating main's copy from the tray, which is
    // what the first version did, could never reach the command line.
    // kReserveUnset leaves the config's own arguments untouched.
    void set_desktop_reserve_gib(int gib);
    bool save_desktop_reserve_gib(int gib);
    void set_reserve_budget_provider(std::function<nlohmann::json()> provider) { reserve_budget_provider_ = std::move(provider); }
    nlohmann::json reserve_budget() const { return reserve_budget_provider_ ? reserve_budget_provider_() : nlohmann::json{{"ok", false}}; }
    bool reserve_plan_matches_config() const;

    // Replaces the configuration the NEXT spawn will use. The dashboard's config
    // editor writes the file and calls this; without it the file would change and
    // a restart would still launch the previous parameters, because this object
    // owns a copy. Takes effect on the next start, never on the running process.
    void update_config(SupervisorConfig cfg);

    [[nodiscard]] EngineStatus status() const;
    [[nodiscard]] std::string log_tail(std::size_t max_bytes) const;
    // By value and under the lock: update_config can swap this out while the tray
    // is drawing a menu from it, and a reference into a member that another thread
    // is assigning is a race whatever the members happen to be.
    [[nodiscard]] SupervisorConfig config() const;
    [[nodiscard]] std::string logs_dir() const;

    // Last moment the engine proved it was doing something, from its own stderr.
    // Atomic because the reader thread writes it and the UI timer reads it.
    [[nodiscard]] std::int64_t last_activity_unix_ms() const noexcept {
        return last_activity_ms_.load(std::memory_order_relaxed);
    }

    void run_loop();

private:
    void spawn();
    void capture_wait();
    void append_log(const char* data, std::size_t n);
    // Records a restart's cause where it outlives st_.last_event, which is
    // overwritten within seconds of the restart it describes. Caller holds mu_.
    void note_restart_reason(const std::string& reason);
    void rotate_logs_if_needed();
    void note_engine_output(const char* data, std::size_t n);

    // The output reader owns nothing: it locks mu_ and mutates st_/line_buf_, so
    // it must never outlive them. Joining is the whole guarantee -- the thread
    // ends on its own when the child's last write handle to the pipe closes, so
    // this only makes the ordering explicit. Never call it while holding mu_.
    void join_reader();

    SupervisorConfig cfg_;
    RestartGate gate_;
    mutable std::mutex mu_;
    EngineStatus st_;
    std::atomic<bool> stop_child_{false};
    std::atomic<bool> quit_{false};
    std::atomic<bool> auto_restart_{true};
    std::atomic<std::int64_t> last_activity_ms_{0};
    int desktop_reserve_gib_ = kReserveUnset; // under mu_
    EngineSpec launched_spec_;
    std::function<nlohmann::json()> reserve_budget_provider_;
    std::string line_buf_;                    // under mu_; partial stderr line
    // Under mu_. The last reservation failure the engine reported, consumed by the next
    // spawn; and the shrunken KV plan that spawn chose, kept across automatic restarts and
    // retired when the configuration changes or the engine is started by hand.
    RuntimeReservationShortfall pending_shortfall_;
    std::int64_t effective_kv_tokens_ = 0;
    std::set<std::uint64_t> inflight_;        // under mu_; open request ids
    void* process_handle_ = nullptr;          // HANDLE
    void* job_handle_     = nullptr;
    std::string log_path_;
    // Joinable, not detached: see join_reader(). Only the engine thread creates
    // it, and every join happens after the process handle is gone.
    std::thread reader_;
};

} // namespace ninfer::supervisor
