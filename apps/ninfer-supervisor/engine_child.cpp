#include "engine_child.hpp"
#include "tray_prefs.hpp"
#include "reserve_budget.hpp"

#include <windows.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>

namespace ninfer::supervisor {
namespace {

std::int64_t unix_ms() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

std::wstring utf8_to_wide(const std::string& s) {
    if (s.empty()) { return {}; }
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
    std::wstring out(static_cast<std::size_t>(n), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, out.data(), n);
    if (!out.empty() && out.back() == L'\0') { out.pop_back(); }
    return out;
}

std::wstring quote_arg(const std::string& a) {
    std::wstring w = utf8_to_wide(a);
    if (w.find_first_of(L" \t\"") == std::wstring::npos) { return w; }
    std::wstring q = L"\"";
    for (wchar_t c : w) {
        if (c == L'"') { q += L"\\\""; } else { q += c; }
    }
    q += L'"';
    return q;
}

void close_handle(void*& h) {
    if (h != nullptr) {
        CloseHandle(static_cast<HANDLE>(h));
        h = nullptr;
    }
}

} // namespace

EngineChild::EngineChild(SupervisorConfig cfg) : cfg_(std::move(cfg)), gate_(cfg_.restart) {
    if (cfg_.logs_dir.empty()) { cfg_.logs_dir = "ninfer-supervisor-logs"; }
    std::filesystem::create_directories(cfg_.logs_dir);
    log_path_ = (std::filesystem::path(cfg_.logs_dir) / "engine.log").string();
    if (!manages_engine_process(cfg_)) {
        auto_restart_  = false;
        st_.last_event = "monitor-only: not managing engine process";
    }
}

EngineChild::~EngineChild() {
    quit_ = true;
    stop();
    // Close the job BEFORE joining: JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE is the
    // backstop that guarantees the child is gone even if TerminateProcess did
    // not take, and only a dead child closes the pipe's last write handle and
    // lets the reader's ReadFile return. Joining after that -- and before the
    // members it locks and mutates are destroyed -- is the point: TerminateProcess
    // is asynchronous, so without this there is a window on every shutdown where
    // the reader sits inside note_engine_output holding a mutex being destroyed.
    close_handle(job_handle_);
    join_reader();
}

void EngineChild::join_reader() {
    if (reader_.joinable()) { reader_.join(); }
}

void EngineChild::request_quit() { quit_ = true; }

EngineStatus EngineChild::status() const {
    std::lock_guard lock(mu_);
    EngineStatus s          = st_;
    s.crash_loop_halted     = gate_.halted();
    s.health_fails          = gate_.health_fails();
    s.recent_exits          = gate_.recent_exits();
    s.crash_window_s        = gate_.policy().crash_loop_window_s;
    s.desktop_reserve_gib   = desktop_reserve_gib_;
    s.last_activity_unix_ms = last_activity_ms_.load(std::memory_order_relaxed);
    s.inflight_requests     = static_cast<int>(inflight_.size());
    return s;
}

void EngineChild::set_desktop_reserve_gib(int gib) {
    std::lock_guard lock(mu_);
    desktop_reserve_gib_ = gib;
}

bool EngineChild::save_desktop_reserve_gib(int gib) {
    const auto budget = reserve_budget();
    if (!budget.value("ok", false) || (gib == 0 ? 8 : gib) > budget.value("max_gib", 0)) return false;
    std::lock_guard lock(mu_);
    if (gib < 0 || gib > 64 || desktop_reserve_pinned(cfg_.engine.args) || cfg_.source_path.empty()) return false;
    if (!update_tray_preference(tray_prefs_path(cfg_.source_path), true, gib)) return false;
    desktop_reserve_gib_ = gib;
    return true;
}

bool EngineChild::reserve_plan_matches_config() const {
    std::lock_guard lock(mu_);
    return st_.state == EngineState::Running && launched_spec_.executable == cfg_.engine.executable &&
        launched_spec_.device == cfg_.engine.device && launched_spec_.workdir == cfg_.engine.workdir &&
        without_reserve_args(launched_spec_.args) == without_reserve_args(cfg_.engine.args);
}

void EngineChild::update_config(SupervisorConfig cfg) {
    {
        // A changed configuration is the user's word on the plan; try it as written.
        std::lock_guard lock(mu_);
        effective_kv_tokens_ = 0;
        pending_shortfall_   = {};
    }
    std::lock_guard lock(mu_);
    cfg_ = std::move(cfg);
}

SupervisorConfig EngineChild::config() const {
    std::lock_guard lock(mu_);
    return cfg_;
}

std::string EngineChild::logs_dir() const {
    std::lock_guard lock(mu_);
    return cfg_.logs_dir;
}

// Line-buffers the engine's stderr and asks logic.hpp what each complete line
// means. This is the only activity signal available for a config with no request
// log, and it is free: the pipe is already being read to write engine.log.
void EngineChild::note_engine_output(const char* data, std::size_t n) {
    std::lock_guard lock(mu_);
    for (std::size_t i = 0; i < n; ++i) {
        const char c = data[i];
        if (c == '\r') { continue; }
        if (c != '\n') {
            // A pathological producer (a progress bar with no newline) must not
            // grow this without bound; 64 KiB is far past any real log line.
            if (line_buf_.size() < 64u * 1024u) { line_buf_.push_back(c); }
            continue;
        }
        const EngineLineSignal sig = classify_engine_line(line_buf_);
        if (sig.activity || sig.ready) {
            last_activity_ms_.store(unix_ms(), std::memory_order_relaxed);
        }
        if (sig.ready) {
            st_.ready = true;
            // A successful start retires the previous failure's text; keeping it
            // would leave the tooltip explaining a crash that has been fixed.
            st_.last_error_line.clear();
            st_.last_error_unix_ms = 0;
        }
        // Only a process-scoped error becomes the engine's "why". A per-request
        // failure (`[error] ... request id=812 status=failed`) is about one
        // request; keeping it would put a stale, unrelated line in the tooltip
        // and blame it for a later crash-loop halt.
        if (is_process_scoped_error(sig)) {
            st_.last_error_line =
                sig.message.empty() ? line_buf_.substr(0, 200) : sig.message;
            st_.last_error_unix_ms = unix_ms();
        }
        // A startup that died for want of runtime memory is not a crash to repeat: the
        // next spawn shrinks the KV plan by the shortfall it names.
        const RuntimeReservationShortfall shortfall =
            parse_runtime_reservation_failure(line_buf_);
        if (shortfall.matched) { pending_shortfall_ = shortfall; }
        // In-flight accounting: the one activity signal no log-cadence flag can
        // switch off, and the reason an idle unload cannot land mid-generation.
        const RequestLifecycle rq = classify_request_line(line_buf_);
        if (rq.matched) {
            // Bounded: a producer that opens without ever closing must not grow
            // this without limit. 4096 is far past any real concurrency here,
            // and stopping at the cap only ever errs towards "busy".
            if (rq.opens && inflight_.size() < 4096u) { inflight_.insert(rq.id); }
            if (rq.closes) { inflight_.erase(rq.id); }
        }
        line_buf_.clear();
    }
}

std::string EngineChild::log_tail(std::size_t max_bytes) const {
    std::ifstream in(log_path_, std::ios::binary);
    if (!in) { return {}; }
    in.seekg(0, std::ios::end);
    const auto size = static_cast<std::size_t>(in.tellg());
    const std::size_t off = size > max_bytes ? size - max_bytes : 0;
    in.seekg(static_cast<std::streamoff>(off));
    std::ostringstream out;
    out << in.rdbuf();
    return out.str();
}

void EngineChild::append_log(const char* data, std::size_t n) {
    rotate_logs_if_needed();
    std::ofstream out(log_path_, std::ios::binary | std::ios::app);
    if (out) { out.write(data, static_cast<std::streamsize>(n)); }
}

void EngineChild::rotate_logs_if_needed() {
    std::error_code ec;
    const auto sz = std::filesystem::file_size(log_path_, ec);
    if (ec || sz < (8ULL << 20)) { return; }
    const auto rotated = log_path_ + ".1";
    std::filesystem::remove(rotated, ec);
    std::filesystem::rename(log_path_, rotated, ec);
}

void EngineChild::start() {
    {
        // Started by hand: memory conditions may have changed, try the configured plan.
        std::lock_guard lock(mu_);
        effective_kv_tokens_ = 0;
        pending_shortfall_   = {};
    }
    if (!manages_engine_process(cfg_)) { return; }
    auto_restart_ = true;
    // RestartGate holds a deque; every other toucher takes mu_, so this one must
    // too or a concurrent note_exit() is a data race on its nodes.
    std::lock_guard lock(mu_);
    gate_.reset_halt();
    if (st_.state == EngineState::Running || st_.state == EngineState::Starting) { return; }
    st_.last_event = "start requested";
}

void EngineChild::stop() {
    if (!manages_engine_process(cfg_)) { return; }
    auto_restart_ = false;
    stop_child_   = true;
    HANDLE proc   = nullptr;
    {
        std::lock_guard lock(mu_);
        proc = static_cast<HANDLE>(process_handle_);
        // Only a live process has an exit to wait for. Setting Stopping with no
        // process left the tray amber forever, because capture_wait -- the only
        // path back to Stopped -- returns immediately when there is nothing to
        // wait on. Stop from Halted or BackingOff hit this every time.
        st_.state      = proc != nullptr ? EngineState::Stopping : EngineState::Stopped;
        st_.last_event = "stop requested";
        st_.ready      = false;
    }
    if (proc != nullptr) { TerminateProcess(proc, 1); }
}

void EngineChild::observe_health(int http_status) {
    if (!manages_engine_process(cfg_)) {
        std::lock_guard lock(mu_);
        st_.ready = http_status == 200;
        if (http_status == 200) {
            st_.health     = "ok";
            st_.state      = EngineState::Running;
            st_.last_event = "unmanaged engine reachable";
        } else if (http_status == 503) {
            st_.health     = "unhealthy";
            st_.state      = EngineState::Running;
            st_.last_event = "unmanaged engine unhealthy";
        } else {
            st_.health     = "unreachable";
            st_.state      = EngineState::Stopped;
            st_.last_event = "unmanaged engine unreachable";
        }
        return;
    }
    bool restart_now = false;
    {
        std::lock_guard lock(mu_);
        if (http_status == 200) {
            gate_.note_healthy();
            st_.health = "ok";
            // Answering /health proves the model finished loading. It is NOT
            // activity: the supervisor polls it once a second, so counting it
            // would mean the idle timer never fires.
            st_.ready = true;
            return;
        }
        if (http_status == 503) {
            st_.health = "unhealthy";
            if (gate_.note_health_fail() && auto_restart_.load()) {
                st_.last_event = "health restart threshold";
                restart_now    = true;
            }
        } else {
            st_.health = "unreachable";
            // An engine that answered /health and then stopped answering is the
            // worst failure there is: the process is alive, so the crash-loop
            // breaker never sees an exit, it still owns the port and the GPU
            // memory, and it serves nothing. Only 503 used to advance the
            // counter, so this -- the case the supervisor exists for -- was the
            // one case it never recovered from. Observed: health unreachable for
            // minutes while state stayed Running and no restart was attempted.
            //
            // Guarded on st_.ready, which spawn() clears and only a 200 sets, so
            // a model that is still loading is legitimately unreachable for its
            // first seconds and must not be restarted before it ever binds.
            if (st_.ready && gate_.note_health_fail() && auto_restart_.load()) {
                st_.last_event = "health restart threshold (unreachable)";
                restart_now    = true;
            }
        }
    }
    if (restart_now) { restart(); }
}

// Writes why the engine is being restarted into the engine log, where it
// survives.
//
// st_.last_event is the only record of a restart's cause, and it is overwritten
// within seconds -- "health restart threshold" becomes "restart requested"
// becomes "engine started" -- so by the time anyone looks at the dashboard the
// reason is gone. An unexplained restart then cannot be told apart from a crash:
// the engine exits with code 1 either way, because that is what TerminateProcess
// is given here. Diagnosing one after the fact meant ruling out possibilities
// rather than reading the answer.
void EngineChild::note_restart_reason(const std::string& reason) {
    const std::string line = "[supervisor] restarting engine: " + reason + "\n";
    append_log(line.data(), line.size());
}

void EngineChild::restart() {
    if (!manages_engine_process(cfg_)) { return; }
    {
        std::lock_guard lock(mu_);
        note_restart_reason(st_.last_event.empty() ? "no reason recorded" : st_.last_event);
    }
    auto_restart_ = true;
    stop_child_   = true;
    HANDLE proc   = nullptr;
    {
        std::lock_guard lock(mu_);
        gate_.reset_halt(); // see start(): the gate's deque is mu_-protected state
        proc           = static_cast<HANDLE>(process_handle_);
        st_.last_event = "restart requested";
    }
    if (proc != nullptr) { TerminateProcess(proc, 1); }
}

void EngineChild::spawn() {
    stop_child_ = false;
    // The previous spawn's reader may still be draining its pipe. Two readers
    // append to one line_buf_, which interleaves two engines' stderr into one
    // half-parsed line; joining first makes a spawn a clean boundary.
    join_reader();
    // The reserve is applied HERE, to the copy of the config this object owns.
    // The tray's choice reaches the engine only through set_desktop_reserve_gib;
    // rewriting main's config object cannot, because that copy is never read
    // again after construction.
    std::vector<std::string> args;
    EngineSpec launch_spec;
    int reserve = kReserveUnset;
    {
        std::lock_guard lock(mu_);
        args    = cfg_.engine.args;
        launch_spec = cfg_.engine;
        reserve = desktop_reserve_gib_;
    }
    if (reserve >= 0) { args = with_desktop_reserve(std::move(args), reserve); }
    // The config's request_log was parsed and then only ever READ: the collector tailed a
    // file the engine had never been told to write, so every request panel on the dashboard
    // (TTFT, decode rate, prefix reuse, speculative acceptance) was empty unless someone
    // also hand-wrote --request-log-jsonl into args. Pass it through.
    args = with_request_log(std::move(args), cfg_.engine.request_log);
    // Same omission, with teeth: api_key_file was read only by the collector, so the
    // engine ran unauthenticated however it was bound. launch_spec, not cfg_, because
    // that copy was taken under the lock.
    args = with_api_key_file(std::move(args), launch_spec.api_key_file);

    // KV capacity follows what the engine could actually reserve last time. The configured
    // value is the ceiling; a shortfall reported by the previous attempt shrinks the plan for
    // this one, and the shrunken plan persists across automatic restarts so a crash loop
    // never relaunches a plan already known not to fit. update_config() and start() retire
    // it, so editing the capacity or starting the engine by hand tries the ceiling again.
    std::string kv_note_line;
    {
        std::lock_guard lock(mu_);
        const std::optional<std::int64_t> configured = explicit_kv_capacity(args);
        if (configured) {
            const std::int64_t current =
                effective_kv_tokens_ > 0 ? effective_kv_tokens_ : *configured;
            if (pending_shortfall_.matched) {
                const std::int64_t next = reduced_kv_capacity(
                    current, pending_shortfall_.required_bytes,
                    pending_shortfall_.available_bytes, kv_capacity_floor_tokens(args));
                if (next < current) {
                    effective_kv_tokens_ = next;
                    const std::int64_t short_mib =
                        (pending_shortfall_.required_bytes - pending_shortfall_.available_bytes) /
                        (1024 * 1024);
                    kv_note_line = "[supervisor] KV capacity reduced to " + std::to_string(next) +
                                   " tokens (configured " + std::to_string(*configured) +
                                   "): the engine's runtime reservation was " +
                                   std::to_string(short_mib) +
                                   " MiB short of free GPU memory. Edit the capacity or stop "
                                   "and start the engine to try the configured value again.";
                }
                pending_shortfall_ = {};
            }
            st_.kv_capacity_configured = *configured;
            if (effective_kv_tokens_ > 0 && effective_kv_tokens_ < *configured) {
                args                      = with_kv_capacity(std::move(args), effective_kv_tokens_);
                st_.kv_capacity_effective = effective_kv_tokens_;
                st_.kv_capacity_note =
                    "Running at " + std::to_string(effective_kv_tokens_) + " of the configured " +
                    std::to_string(*configured) +
                    " KV tokens: other apps hold GPU memory the full plan needed. Edit the "
                    "capacity or stop and start the engine to try the configured value again.";
            } else {
                st_.kv_capacity_effective = 0;
                st_.kv_capacity_note.clear();
            }
        } else {
            st_.kv_capacity_configured = 0;
            st_.kv_capacity_effective  = 0;
            st_.kv_capacity_note.clear();
        }
    }
    if (!kv_note_line.empty()) {
        kv_note_line.push_back('\n');
        append_log(kv_note_line.data(), kv_note_line.size());
    }

    std::wstring cmd = quote_arg(cfg_.engine.executable);
    for (const auto& a : args) {
        cmd += L' ';
        cmd += quote_arg(a);
    }
    std::vector<wchar_t> cmd_buf(cmd.begin(), cmd.end());
    cmd_buf.push_back(L'\0');

    SECURITY_ATTRIBUTES sa{};
    sa.nLength        = sizeof(sa);
    sa.bInheritHandle = TRUE;

    HANDLE out_r = nullptr;
    HANDLE out_w = nullptr;
    if (!CreatePipe(&out_r, &out_w, &sa, 0)) {
        throw std::runtime_error("CreatePipe failed");
    }
    SetHandleInformation(out_r, HANDLE_FLAG_INHERIT, 0);

    STARTUPINFOW si{};
    si.cb          = sizeof(si);
    si.dwFlags     = STARTF_USESTDHANDLES;
    si.hStdOutput  = out_w;
    si.hStdError   = out_w;
    si.hStdInput   = GetStdHandle(STD_INPUT_HANDLE);

    PROCESS_INFORMATION pi{};
    const std::wstring cwd = utf8_to_wide(cfg_.engine.workdir);
    const wchar_t* cwd_ptr = cwd.empty() ? nullptr : cwd.c_str();
    if (!CreateProcessW(nullptr, cmd_buf.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW,
                        nullptr, cwd_ptr, &si, &pi)) {
        CloseHandle(out_r);
        CloseHandle(out_w);
        throw std::runtime_error("CreateProcessW failed");
    }
    CloseHandle(out_w);
    CloseHandle(pi.hThread);

    if (job_handle_ == nullptr) {
        HANDLE job = CreateJobObjectW(nullptr, nullptr);
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION lim{};
        lim.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        SetInformationJobObject(job, JobObjectExtendedLimitInformation, &lim, sizeof(lim));
        job_handle_ = job;
    }
    AssignProcessToJobObject(static_cast<HANDLE>(job_handle_), pi.hProcess);

    // A spawn is activity: the model load that follows takes about a minute, and
    // an idle-unload timer that started counting before it finished would kill
    // the engine during its own startup.
    last_activity_ms_.store(unix_ms(), std::memory_order_relaxed);
    {
        std::lock_guard lock(mu_);
        process_handle_      = pi.hProcess;
        st_.pid              = static_cast<std::uint64_t>(pi.dwProcessId);
        st_.launch_args      = args;
        launched_spec_       = std::move(launch_spec);
        st_.state            = EngineState::Running;
        st_.started_unix_ms  = unix_ms();
        st_.ready            = false; // Running means "process exists", not "serving"
        line_buf_.clear();
        inflight_.clear(); // a new process has nothing in flight

        ++st_.restart_count;
        st_.last_event = "engine started";
    }

    reader_ = std::thread([this, out_r] {
        char buf[4096];
        DWORD n = 0;
        while (ReadFile(out_r, buf, sizeof(buf), &n, nullptr) && n > 0) {
            append_log(buf, n);
            note_engine_output(buf, n);
        }
        CloseHandle(out_r);
    });
}

void EngineChild::capture_wait() {
    HANDLE proc = nullptr;
    {
        std::lock_guard lock(mu_);
        proc = static_cast<HANDLE>(process_handle_);
    }
    if (proc == nullptr) { return; }
    WaitForSingleObject(proc, INFINITE);
    DWORD code = 0;
    GetExitCodeProcess(proc, &code);
    {
        std::lock_guard lock(mu_);
        st_.last_exit_code = static_cast<int>(code);
        st_.pid            = 0;
        process_handle_    = nullptr;
        st_.state          = EngineState::Stopped;
        st_.ready          = false;
        st_.last_event     = "engine exited";
        inflight_.clear(); // an exited process is not serving anything
    }
    CloseHandle(proc);
    // The process is gone, so its end of the pipe is closed and the reader is
    // finishing the last buffered bytes. Joining here does three things: the
    // final stderr line (usually the reason it died) is parsed before run_loop
    // decides what to announce, the next spawn cannot interleave with it, and
    // nothing is left to touch mu_ if the object is destroyed next.
    join_reader();
}

void EngineChild::run_loop() {
    while (!quit_.load()) {
        const bool running = [&] {
            std::lock_guard lock(mu_);
            return process_handle_ != nullptr;
        }();
        if (running) {
            capture_wait();
            const bool intentional = stop_child_.exchange(false);
            if (quit_.load() || !auto_restart_.load() || intentional) { continue; }
            bool allow = false;
            {
                std::lock_guard lock(mu_);
                allow = gate_.note_exit(std::chrono::steady_clock::now());
            }
            if (!allow) {
                std::lock_guard lock(mu_);
                st_.state      = EngineState::Halted;
                st_.last_event = "crash-loop breaker: too many exits";
                auto_restart_  = false;
                continue;
            }
            int wait_s = 0;
            {
                std::lock_guard lock(mu_);
                gate_.advance_backoff();
                wait_s = gate_.backoff_seconds();
            }
            {
                std::lock_guard lock(mu_);
                st_.state      = EngineState::BackingOff;
                st_.last_event = "backing off";
            }
            for (int i = 0; i < wait_s * 10 && !quit_.load() && auto_restart_.load(); ++i) {
                Sleep(100);
            }
            continue;
        }
        const bool halted = [&] {
            std::lock_guard lock(mu_);
            return gate_.halted();
        }();
        if (manages_engine_process(cfg_) && auto_restart_.load() && !halted && !quit_.load()) {
            try {
                {
                    std::lock_guard lock(mu_);
                    st_.state = EngineState::Starting;
                }
                spawn();
            } catch (const std::exception& ex) {
                std::lock_guard lock(mu_);
                st_.state      = EngineState::Stopped;
                st_.last_event = std::string("spawn failed: ") + ex.what();
                auto_restart_  = false;
            }
            continue;
        }
        Sleep(200);
    }
    stop();
}

} // namespace ninfer::supervisor
