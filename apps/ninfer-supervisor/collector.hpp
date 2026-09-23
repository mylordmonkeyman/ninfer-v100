#pragma once

#include "config.hpp"
#include "dxgi_query.hpp"

#include <nlohmann/json.hpp>

#include <atomic>
#include <cstdint>
#include <deque>
#include <fstream>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace ninfer::supervisor {

struct RequestMix {
    std::uint64_t done                 = 0;
    double ttft_ms_mean                = 0;
    // Per response: how fast one answer feels. The mean of each request's own rate.
    double decode_tok_s_mean           = 0;
    // Across everything at once: what the card is actually producing. Concurrent
    // requests add up here and do not in the mean above.
    double decode_tok_s_total          = 0;
    double prefill_tok_s_total         = 0;
    int running_requests               = 0;
    std::uint64_t reuse_full_reset     = 0;
    std::uint64_t reuse_append         = 0;
    std::uint64_t reuse_seed           = 0;
    std::uint64_t reuse_other          = 0;
    std::string last_reuse;
    bool log_available                 = false;
    std::string log_error;
    std::string mtp_backend;
    int mtp_draft_window               = 0;
    std::uint64_t mtp_drafted          = 0;
    std::uint64_t mtp_accepted         = 0;
    std::uint64_t mtp_fallback_steps   = 0;
    std::uint64_t mtp_rounds           = 0;
    std::vector<std::uint64_t> mtp_accepted_per_position;
    double mtp_last_accept_rate        = 0;
    // Who is actually on the engine, busiest first, over a rolling window.
    std::vector<ClientActivity> clients;
    int clients_window_minutes         = 0;
};

struct Collected {
    DxgiSnapshot dxgi;
    NvidiaSmiMemory nvidia;
    nlohmann::json gpu_processes = {{"ok", false}, {"apps", nlohmann::json::array()}};
    nlohmann::json admin_vram = nullptr;
    std::string admin_vram_note;
    RequestMix requests;
    std::string health_body;
    std::string engine_capacity_line;
    int health_status = 0;
};

class Collector {
public:
    explicit Collector(EngineSpec spec, std::string logs_dir = {})
        : spec_(std::move(spec)), logs_dir_(std::move(logs_dir)), series_(6000) {}
    ~Collector() { stop_series(); }

    Collector(const Collector&)            = delete;
    Collector& operator=(const Collector&) = delete;

    void start_series();
    void stop_series();
    Collected snapshot();
    nlohmann::json series_json();
    nlohmann::json throughput_series_json();
    nlohmann::json vram_control_json();
    nlohmann::json insights_report();
    void note_engine_state(const std::string& state, const std::string& last_event);

    // Observation must not depend on somebody having a browser open. Health and
    // engine-state were previously fed from DashboardServer::state_json(), so
    // with no dashboard the tray never turned green, health-driven restarts
    // never fired, and only one of two crash-loop halts reached series.jsonl.
    // Both must be set BEFORE start_series(); they are read by the 1 Hz thread.
    using HealthObserver      = std::function<void(int http_status)>;
    using EngineStateProvider = std::function<std::pair<std::string, std::string>()>;
    void set_health_observer(HealthObserver obs) { health_observer_ = std::move(obs); }
    void set_engine_state_provider(EngineStateProvider p) { engine_state_provider_ = std::move(p); }

    // The 1 Hz cached nvidia-smi reading. The tray menu uses this instead of
    // spawning nvidia-smi on the UI thread: that spawn costs ~51 ms measured, and
    // its read loop had no timeout, so a wedged nvidia-smi froze the menu.
    [[nodiscard]] NvidiaSmiMemory last_nvidia();

    // mtime of the engine's request log, or 0 when none is configured. A second,
    // coarser activity source than the stderr scan: it survives a supervisor
    // restart, where the stderr clock does not.
    [[nodiscard]] std::int64_t request_log_last_write_unix_ms();

private:
    void poll_health(Collected& out);
    void poll_admin(Collected& out);
    void poll_nvidia_smi(Collected& out);
    void poll_request_log(Collected& out);
    void series_loop();
    // Reads only what has been appended since the last call and folds any engine
    // throughput reports into the ring. Caller holds mu_.
    void tail_throughput_locked();
    void observe_loop();
    [[nodiscard]] std::int64_t poll_request_log_mtime() const;
    void record_transitions(const Collected& snap);
    void persist_sample(const VramSample& s);
    void persist_event(const VramSeriesEvent& e);
    void load_persisted_series();
    void rotate_series_file_locked();
    static std::int64_t now_ms();

    EngineSpec spec_;
    std::string logs_dir_;
    std::string series_path_;
    std::ofstream series_file_;
    std::int64_t series_bytes_since_check_ = 0;
    std::mutex mu_;
    VramSeriesRing series_;
    // Throughput is derived by tailing the request log incrementally from
    // throughput_offset_, because the log reaches tens of megabytes and re-reading
    // it once a second to compute a rate would cost more than the engine it watches.
    ThroughputRing throughput_;
    // request_start carries the client, request_done carries the result; they are
    // joined on request_id, so a started-but-unfinished request waits here.
    std::unordered_map<std::uint64_t, std::pair<std::string, int>> pending_clients_;
    std::deque<ClientRequest> client_window_;
    std::uintmax_t throughput_offset_ = 0;
    std::atomic<bool> series_run_{false};
    std::thread series_thread_;
    std::thread observe_thread_;
    std::int64_t detector_last_ran_ms_ = 0;
    int last_health_status_          = -1;
    std::string last_health_body_;
    AdminVramCursor admin_cursor_;
    std::int64_t last_release_ms_    = 0;
    std::string last_engine_state_;
    nlohmann::json last_admin_vram_  = nullptr;
    std::string last_admin_note_;
    DxgiSnapshot last_dxgi_;
    NvidiaSmiMemory last_nvidia_;
    nlohmann::json last_gpu_processes_ = {{"ok", false}, {"apps", nlohmann::json::array()}};
    HealthObserver health_observer_;
    EngineStateProvider engine_state_provider_;
    std::int64_t request_log_mtime_ms_ = 0;
};

} // namespace ninfer::supervisor
