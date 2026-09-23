#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::supervisor {

inline bool is_loopback_host(std::string_view host) {
    return host == "127.0.0.1" || host == "::1" || host == "localhost" || host == "localhost.";
}

inline bool is_loopback_peer(std::string_view addr) {
    if (addr.empty()) { return false; }
    if (is_loopback_host(addr)) { return true; }
    // httplib may report IPv4-mapped IPv6.
    return addr == "::ffff:127.0.0.1";
}

inline constexpr std::string_view kSupervisorControlHeader      = "X-NInfer-Supervisor";
inline constexpr std::string_view kSupervisorControlHeaderValue = "1";

inline std::string_view trim_sv(std::string_view s) {
    while (!s.empty() && (s.front() == ' ' || s.front() == '\t' || s.front() == '\r' ||
                          s.front() == '\n')) {
        s.remove_prefix(1);
    }
    while (!s.empty() && (s.back() == ' ' || s.back() == '\t' || s.back() == '\r' ||
                          s.back() == '\n')) {
        s.remove_suffix(1);
    }
    return s;
}

// Split Host into name and optional port. IPv6 literals must be bracketed when a
// port is present (`[::1]:8099`).
inline bool split_host_header(std::string_view host, std::string& name, int& port, bool& has_port) {
    host     = trim_sv(host);
    name.clear();
    port     = 0;
    has_port = false;
    if (host.empty()) { return false; }
    if (host.front() == '[') {
        const auto rb = host.find(']');
        if (rb == std::string_view::npos) { return false; }
        name = std::string(host.substr(0, rb + 1));
        if (rb + 1 == host.size()) { return true; }
        if (host[rb + 1] != ':') { return false; }
        const auto p = host.substr(rb + 2);
        if (p.empty()) { return false; }
        int value = 0;
        for (char c : p) {
            if (c < '0' || c > '9') { return false; }
            value = value * 10 + (c - '0');
            if (value > 65535) { return false; }
        }
        port     = value;
        has_port = true;
        return true;
    }
    const auto colon = host.rfind(':');
    if (colon != std::string_view::npos && host.find(':') == colon) {
        name = std::string(host.substr(0, colon));
        const auto p = host.substr(colon + 1);
        if (p.empty() || name.empty()) { return false; }
        int value = 0;
        for (char c : p) {
            if (c < '0' || c > '9') { return false; }
            value = value * 10 + (c - '0');
            if (value > 65535) { return false; }
        }
        port     = value;
        has_port = true;
        return true;
    }
    name = std::string(host);
    return !name.empty();
}

inline bool is_loopback_host_name(std::string_view name) {
    return is_loopback_host(name) || name == "[::1]";
}

// DNS-rebinding defense: only the listen port's loopback names, plus the
// configured bind host when --bind-any names a specific interface. Binding
// 0.0.0.0 does not open the Host allowlist.
inline bool host_header_allowed(std::string_view host_header, int listen_port,
                                std::string_view bind_host, bool bind_any) {
    std::string name;
    int port         = 0;
    bool has_port    = false;
    if (!split_host_header(host_header, name, port, has_port)) { return false; }
    if (has_port && port != listen_port) { return false; }
    if (is_loopback_host_name(name)) { return true; }
    if (!bind_any) { return false; }
    if (bind_host.empty() || bind_host == "0.0.0.0" || bind_host == "::" || bind_host == "[::]") {
        return false;
    }
    return name == bind_host;
}

inline bool supervisor_control_header_ok(std::string_view value) {
    return trim_sv(value) == kSupervisorControlHeaderValue;
}

struct NvidiaSmiMemory {
    bool ok                 = false;
    int index               = -1;
    std::uint64_t used_mib  = 0;
    std::uint64_t total_mib = 0;
    std::string error;
};

// Parses `nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits`.
// Values are mebibytes. Picks the row whose index equals `device`.
inline NvidiaSmiMemory parse_nvidia_smi_memory_csv(std::string_view csv, int device) {
    NvidiaSmiMemory out;
    std::string_view rest = csv;
    bool saw_row          = false;
    while (!rest.empty()) {
        auto nl    = rest.find_first_of("\n\r");
        auto line  = trim_sv(nl == std::string_view::npos ? rest : rest.substr(0, nl));
        rest       = nl == std::string_view::npos ? std::string_view{}
                                                  : rest.substr(nl + 1);
        if (line.empty()) { continue; }
        saw_row = true;
        const auto c1 = line.find(',');
        if (c1 == std::string_view::npos) { continue; }
        const auto c2 = line.find(',', c1 + 1);
        if (c2 == std::string_view::npos) { continue; }
        const auto idx_s  = trim_sv(line.substr(0, c1));
        const auto used_s = trim_sv(line.substr(c1 + 1, c2 - c1 - 1));
        const auto tot_s  = trim_sv(line.substr(c2 + 1));
        int idx           = 0;
        std::uint64_t used = 0;
        std::uint64_t tot  = 0;
        try {
            idx  = std::stoi(std::string(idx_s));
            used = std::stoull(std::string(used_s));
            tot  = std::stoull(std::string(tot_s));
        } catch (...) { continue; }
        if (idx != device) { continue; }
        out.ok       = true;
        out.index    = idx;
        out.used_mib = used;
        out.total_mib = tot;
        return out;
    }
    out.error = saw_row ? "nvidia-smi csv has no row for the configured device"
                        : "nvidia-smi csv is empty";
    return out;
}

inline std::uint64_t mib_to_bytes(std::uint64_t mib) { return mib * 1024ull * 1024ull; }

// Pre-filter that agrees with the engine JSONL schema: the field is "event",
// not "type". A substring on the value without the key name is how a panel
// can look populated while every record is then discarded.
inline bool jsonl_event_is(std::string_view line, std::string_view event) {
    const std::string compact = std::string("\"event\":\"") + std::string(event) + "\"";
    const std::string spaced  = std::string("\"event\": \"") + std::string(event) + "\"";
    return line.find(compact) != std::string_view::npos ||
           line.find(spaced) != std::string_view::npos;
}

struct VramSample {
    std::int64_t t_ms                = 0;
    std::uint64_t budget_bytes       = 0;
    std::uint64_t nvidia_used_bytes  = 0;
};

struct VramSeriesEvent {
    std::int64_t t_ms = 0;
    std::string kind;
    std::string label;
};

// One engine throughput report: tokens per WALL second across everything running
// at once, which is a different number from the average speed of one response.
// Taken from the engine's own `throughput` log event rather than derived here.
//
// Deriving it from completed requests was the obvious approach and it is wrong:
// a request only reaches the log when it finishes, so a window over completions
// reads ZERO for the whole of a long generation -- precisely when the card is
// busiest. Measured that happening: 207.9 tok/s, then 0.0 with four generations
// still streaming. The engine emits this every ~5 s while the work is happening.
struct ThroughputSample {
    std::int64_t t_ms     = 0;
    double decode_tok_s   = 0;
    double prefill_tok_s  = 0;
    int running           = 0;
};

// What one app has been doing, rolled up from its completed requests. Keyed by
// the User-Agent the engine now records, so the dashboard can show WHICH client
// is re-prefilling its whole context every turn rather than only that somebody
// is. Clients differ enormously on the same engine: one held 99% prefix reuse
// all day beside another that held 3%, and the aggregates hid both.
struct ClientActivity {
    std::string name;                 // empty means the client sent no User-Agent
    std::uint64_t requests    = 0;
    std::uint64_t from_root   = 0;    // conversations that had to start over
    std::uint64_t prompt_tokens = 0;
    std::uint64_t refill_tokens = 0;  // of those, the ones actually recomputed
    std::int64_t last_seen_ms = 0;
    int tool_count            = 0;
    double ttft_ms_sum        = 0;

    [[nodiscard]] double reuse_percent() const {
        if (prompt_tokens == 0) { return 0.0; }
        const double reused = static_cast<double>(prompt_tokens - refill_tokens);
        return 100.0 * reused / static_cast<double>(prompt_tokens);
    }
    [[nodiscard]] double ttft_ms_mean() const {
        return requests == 0 ? 0.0 : ttft_ms_sum / static_cast<double>(requests);
    }
};

// One completed request, kept only long enough to age out of the window.
struct ClientRequest {
    std::int64_t t_ms = 0;
    std::string client;
    std::uint64_t prompt_tokens = 0;
    std::uint64_t refill_tokens = 0;
    bool from_root              = false;
    int tool_count              = 0;
    double ttft_ms              = 0;
};

// Rolls the window up per client, busiest first. A rolling window rather than
// totals since startup: the question this answers is "who is on the engine now",
// and a client that stopped an hour ago should fall off rather than keep its
// place in the list.
[[nodiscard]] inline std::vector<ClientActivity>
summarize_clients(const std::deque<ClientRequest>& window) {
    std::vector<ClientActivity> out;
    for (const ClientRequest& r : window) {
        auto it = std::find_if(out.begin(), out.end(),
                               [&](const ClientActivity& c) { return c.name == r.client; });
        if (it == out.end()) {
            out.push_back(ClientActivity{.name = r.client});
            it = out.end() - 1;
        }
        ++it->requests;
        it->from_root += r.from_root ? 1U : 0U;
        it->prompt_tokens += r.prompt_tokens;
        it->refill_tokens += r.refill_tokens;
        it->ttft_ms_sum += r.ttft_ms;
        it->last_seen_ms = std::max(it->last_seen_ms, r.t_ms);
        it->tool_count   = std::max(it->tool_count, r.tool_count);
    }
    std::sort(out.begin(), out.end(), [](const ClientActivity& a, const ClientActivity& b) {
        if (a.requests != b.requests) { return a.requests > b.requests; }
        return a.last_seen_ms > b.last_seen_ms;
    });
    return out;
}

// Ring of engine throughput reports (~0.2 Hz). Separate from the 10 Hz VRAM
// series: this arrives on the engine's own cadence, and folding it into the fast
// series would decuple series.jsonl for no extra information.
struct ThroughputRing {
    explicit ThroughputRing(std::size_t cap = 900) : cap_(cap), buf_(cap) {}

    void push(ThroughputSample s) {
        if (cap_ == 0) { return; }
        buf_[head_] = s;
        head_       = (head_ + 1) % cap_;
        if (size_ < cap_) { ++size_; }
    }

    [[nodiscard]] std::vector<ThroughputSample> samples() const {
        std::vector<ThroughputSample> out;
        out.reserve(size_);
        const std::size_t start = size_ < cap_ ? 0 : head_;
        for (std::size_t i = 0; i < size_; ++i) { out.push_back(buf_[(start + i) % cap_]); }
        return out;
    }

    [[nodiscard]] std::size_t size() const noexcept { return size_; }

private:
    std::size_t cap_  = 0;
    std::size_t head_ = 0;
    std::size_t size_ = 0;
    std::vector<ThroughputSample> buf_;
};

// Diff last_transition/last_reason, not held_bytes. A 42 ms release finishes
// between 1 Hz polls; the persisted last_reason is what remains observable.
struct AdminVramCursor {
    bool seen                = false;
    std::string last_transition;
    std::string last_reason;

    // Returns true if this observation is an event after the baseline poll.
    bool observe(std::string_view trans, std::string_view reason, std::string& kind) {
        if (!seen) {
            seen             = true;
            last_transition  = std::string(trans);
            last_reason      = std::string(reason);
            return false;
        }
        if (trans == last_transition && reason == last_reason) { return false; }
        last_transition = std::string(trans);
        last_reason     = std::string(reason);
        if (trans == "release") {
            kind = "vram_release";
        } else if (trans == "reclaim" || trans == "reclaim-failed") {
            kind = "vram_reclaim";
        } else {
            kind = "admin_vram";
        }
        return true;
    }
};

inline bool parse_series_event_line(std::string_view line, VramSeriesEvent& ev);
inline bool parse_series_sample_line(std::string_view line, VramSample& s);

// Ring of raw 10 Hz samples. No averaging. Oldest is dropped on overflow.
struct VramSeriesRing {
    explicit VramSeriesRing(std::size_t cap = 6000) : cap_(cap), buf_(cap) {}

    void push(VramSample s) {
        if (cap_ == 0) { return; }
        buf_[head_] = s;
        head_       = (head_ + 1) % cap_;
        if (size_ < cap_) { ++size_; }
    }

    void push_event(VramSeriesEvent e, std::size_t event_cap = 128) {
        events_.push_back(std::move(e));
        while (events_.size() > event_cap) { events_.pop_front(); }
    }

    [[nodiscard]] std::vector<VramSample> samples() const {
        std::vector<VramSample> out;
        out.reserve(size_);
        const std::size_t start = size_ < cap_ ? 0 : head_;
        for (std::size_t i = 0; i < size_; ++i) { out.push_back(buf_[(start + i) % cap_]); }
        return out;
    }

    [[nodiscard]] std::vector<VramSeriesEvent> events() const {
        return {events_.begin(), events_.end()};
    }

    [[nodiscard]] std::size_t size() const noexcept { return size_; }

    void load_jsonl(std::string_view jsonl) {
        std::string_view rest = jsonl;
        while (!rest.empty()) {
            auto nl   = rest.find('\n');
            auto line = trim_sv(nl == std::string_view::npos ? rest : rest.substr(0, nl));
            rest      = nl == std::string_view::npos ? std::string_view{} : rest.substr(nl + 1);
            if (line.empty()) { continue; }
            VramSeriesEvent ev;
            VramSample samp;
            if (parse_series_event_line(line, ev)) {
                push_event(std::move(ev));
            } else if (parse_series_sample_line(line, samp)) {
                push(samp);
            }
        }
    }

private:
    std::size_t cap_  = 0;
    std::size_t head_ = 0;
    std::size_t size_ = 0;
    std::vector<VramSample> buf_;
    std::deque<VramSeriesEvent> events_;
};

inline bool extract_json_i64(std::string_view line, std::string_view key, std::int64_t& out) {
    const std::string pat = "\"" + std::string(key) + "\":";
    auto pos              = line.find(pat);
    if (pos == std::string_view::npos) { return false; }
    pos += pat.size();
    while (pos < line.size() && (line[pos] == ' ')) { ++pos; }
    bool neg = false;
    if (pos < line.size() && line[pos] == '-') {
        neg = true;
        ++pos;
    }
    if (pos >= line.size() || line[pos] < '0' || line[pos] > '9') { return false; }
    std::int64_t v = 0;
    while (pos < line.size() && line[pos] >= '0' && line[pos] <= '9') {
        v = v * 10 + (line[pos] - '0');
        ++pos;
    }
    out = neg ? -v : v;
    return true;
}

inline bool extract_json_str(std::string_view line, std::string_view key, std::string& out) {
    const std::string pat = "\"" + std::string(key) + "\":\"";
    auto pos              = line.find(pat);
    if (pos == std::string_view::npos) { return false; }
    pos += pat.size();
    std::string s;
    while (pos < line.size() && line[pos] != '"') {
        s.push_back(line[pos]);
        ++pos;
    }
    out = std::move(s);
    return true;
}

inline bool parse_series_event_line(std::string_view line, VramSeriesEvent& ev) {
    if (line.find("\"kind\"") == std::string_view::npos) { return false; }
    std::int64_t t = 0;
    if (!extract_json_i64(line, "t_ms", t)) { return false; }
    ev.t_ms = t;
    extract_json_str(line, "kind", ev.kind);
    extract_json_str(line, "label", ev.label);
    return !ev.kind.empty();
}

inline bool parse_series_sample_line(std::string_view line, VramSample& s) {
    if (line.find("\"kind\"") != std::string_view::npos) { return false; }
    std::int64_t t = 0, b = 0, n = 0;
    if (!extract_json_i64(line, "t_ms", t)) { return false; }
    extract_json_i64(line, "budget_bytes", b);
    extract_json_i64(line, "nvidia_used_bytes", n);
    s.t_ms               = t;
    s.budget_bytes       = static_cast<std::uint64_t>(b);
    s.nvidia_used_bytes  = static_cast<std::uint64_t>(n);
    return true;
}

inline std::string format_series_sample_line(const VramSample& s) {
    return std::string("{\"t_ms\":") + std::to_string(s.t_ms) +
           ",\"budget_bytes\":" + std::to_string(s.budget_bytes) +
           ",\"nvidia_used_bytes\":" + std::to_string(s.nvidia_used_bytes) + "}";
}

inline std::string format_series_event_line(const VramSeriesEvent& e) {
    return std::string("{\"t_ms\":") + std::to_string(e.t_ms) + ",\"kind\":\"" + e.kind +
           "\",\"label\":\"" + e.label + "\"}";
}

inline std::string extract_kv_capacity_line(std::string_view log) {
    const auto key = std::string_view("KV capacity ");
    const auto pos = log.rfind(key);
    if (pos == std::string_view::npos) { return {}; }
    auto start = log.find_last_of("\n", pos);
    start      = start == std::string_view::npos ? 0 : start + 1;
    auto end   = log.find('\n', pos);
    auto line  = log.substr(start, end == std::string_view::npos ? log.size() - start : end - start);
    if (!line.empty() && line.back() == '\r') { line.remove_suffix(1); }
    return std::string(trim_sv(line));
}

struct RestartPolicy {
    int initial_backoff_s     = 1;
    int max_backoff_s         = 60;
    int crash_loop_max        = 5;
    int crash_loop_window_s   = 60;
    int health_fail_threshold = 3;
};

class RestartGate {
public:
    explicit RestartGate(RestartPolicy policy = {}) : policy_(policy), backoff_s_(policy.initial_backoff_s) {}

    // Record an engine exit. Returns false if auto-restart is halted (crash loop).
    bool note_exit(std::chrono::steady_clock::time_point now) {
        if (halted_) { return false; }
        const auto window = std::chrono::seconds(policy_.crash_loop_window_s);
        while (!exits_.empty() && now - exits_.front() > window) { exits_.pop_front(); }
        exits_.push_back(now);
        if (static_cast<int>(exits_.size()) >= policy_.crash_loop_max) {
            halted_ = true;
            return false;
        }
        return true;
    }

    [[nodiscard]] int backoff_seconds() const noexcept { return backoff_s_; }

    void advance_backoff() {
        if (backoff_s_ < policy_.max_backoff_s) {
            const int next = backoff_s_ * 2;
            backoff_s_     = next > policy_.max_backoff_s ? policy_.max_backoff_s : next;
        }
    }

    void note_healthy() {
        backoff_s_ = policy_.initial_backoff_s;
        health_fails_ = 0;
    }

    bool note_health_fail() {
        ++health_fails_;
        return health_fails_ >= policy_.health_fail_threshold;
    }

    void clear_health_fails() { health_fails_ = 0; }

    void reset_halt() {
        halted_ = false;
        exits_.clear();
        backoff_s_    = policy_.initial_backoff_s;
        health_fails_ = 0;
    }

    [[nodiscard]] bool halted() const noexcept { return halted_; }
    [[nodiscard]] int recent_exits() const noexcept { return static_cast<int>(exits_.size()); }
    [[nodiscard]] int health_fails() const noexcept { return health_fails_; }
    [[nodiscard]] const RestartPolicy& policy() const noexcept { return policy_; }

private:
    RestartPolicy policy_;
    std::deque<std::chrono::steady_clock::time_point> exits_;
    int backoff_s_    = 1;
    int health_fails_ = 0;
    bool halted_      = false;
};

// ---------------------------------------------------------------------------
// Tray logic. Everything below is pure so it can be unit-tested off a GPU and
// off Win32: the tray's bugs so far have all been in arithmetic (menu id
// collisions, argument rewriting), not in the Win32 calls around it.
// ---------------------------------------------------------------------------

// The reserve choices. 0 is "engine default" and means the flag is dropped so
// the engine's own kDefaultDesktopReserveBytes (8 GiB) applies -- the tray must
// not silently lower a reserve nobody asked it to change. These are shortcuts;
// the dashboard accepts custom whole-GiB targets. None is a universal minimum
// or recommendation: app demand depends on the user's workload.
inline constexpr std::array<int, 7> kReserveChoicesGib{0, 2, 3, 4, 6, 8, 12};

// A duration, not a toggle. Unloading after five minutes would punish anyone who
// steps away mid-task; an hour reliably catches "finished for the day".
inline constexpr std::array<int, 4> kIdleChoicesMinutes{0, 15, 60, 240};

// Sentinel for TrayPrefs::desktop_reserve_gib: the user has never picked one, so
// whatever the config file says stays untouched. Distinct from 0 ("I chose the
// engine default"), which does strip the flag.
inline constexpr int kReserveUnset = -1;

// Rewrites --desktop-reserve-gib in an engine argument vector.
//  - an existing value is replaced in place
//  - a trailing "--desktop-reserve-gib" with no value gets one appended
//  - --desktop-reserve-mib is more specific than anything the menu can express,
//    so its presence makes this a no-op rather than a silent coarsening
//  - gib <= 0 removes the pair, leaving the engine's own default in force
//
// A config that pins --desktop-reserve-mib owns the reserve outright: the menu
// cannot express MiB, so every choice in it would be a silent coarsening. The
// UI needs to know this too -- an editable submenu whose every entry is a no-op
// is worse than a greyed one that says why -- so the test is its own function
// rather than a loop buried in the rewrite.
inline bool desktop_reserve_pinned(const std::vector<std::string>& args) {
    for (const auto& a : args) {
        if (a == "--desktop-reserve-mib") { return true; }
    }
    return false;
}

// ---------------------------------------------------------------------------
// Engine launch parameters as data.
//
// One table drives four things that used to drift apart: parsing the configured
// argument vector, validating an edit, rendering the vector back, and describing
// the form the dashboard draws. Adding a knob means adding a row.
//
// What is deliberately NOT here: the executable path and the working directory.
// Those decide which binary runs, and the dashboard must not be able to change
// them -- the control gate is a loopback peer check plus a header, which is a
// CSRF brake, not an authorization system. Parameters are editable; what gets
// executed is a file-only decision.
// ---------------------------------------------------------------------------

enum class ParamKind : std::uint8_t {
    Int,       // "--flag N"
    Text,      // "--flag STRING"
    Enum,      // "--flag CHOICE"
    Flag,      // "--flag" with no value; present or absent
    IntOrAuto, // "--flag N" or "--flag auto"
};

struct EngineParamSpec {
    std::string_view flag;
    std::string_view key;
    ParamKind kind{};
    std::string_view group;
    std::string_view label;
    std::string_view help;
    long long min_value = 0;
    long long max_value = 0;
    std::array<std::string_view, 6> choices{};
};

// Ranges are the engine's own limits where one exists (max_concurrency throws
// above 8 in the engine, so the form must not offer 16), and otherwise a bound
// wide enough not to be a policy while still catching a typo'd extra digit.
inline const std::vector<EngineParamSpec>& engine_param_specs() {
    static const std::vector<EngineParamSpec> specs = {
        {"--max-context", "max_context", ParamKind::Int, "capacity", "Max context",
         "Longest prompt plus generation the engine will admit, in tokens.", 256, 262144, {}},
        {"--kv-capacity", "kv_capacity", ParamKind::IntOrAuto, "capacity", "KV capacity",
         "Shared KV pool for every active request. 'auto' sizes it from free memory.", 1024, 4194304, {}},
        {"--max-concurrency", "max_concurrency", ParamKind::Int, "capacity", "Max concurrency",
         "How many requests may be active at once. 8 is the Engine's compile-time lane maximum, not a GPU-memory guess.", 1, 8, {}},
        {"--max-pending-requests", "max_pending_requests", ParamKind::Int, "capacity",
         "Max pending requests", "Queue depth before new requests are rejected.", 1, 4096, {}},
        {"--pending-timeout-ms", "pending_timeout_ms", ParamKind::Int, "capacity",
         "Pending timeout (ms)", "How long a request may wait for a lane before it fails.", 0,
         3600000, {}},
        {"--prefill-chunk", "prefill_chunk", ParamKind::Int, "capacity", "Prefill chunk",
         "Tokens per prefill step. Larger is faster and needs more workspace.", 128, 65536, {}},

        {"--desktop-reserve-gib", "desktop_reserve_gib", ParamKind::Int, "memory",
         "Desktop reserve (GiB)",
         "Free GPU memory target at startup; leaves headroom beyond existing app allocations.", 0, 64,
         {}},
        {"--kv-dtype", "kv_dtype", ParamKind::Enum, "memory", "KV cache dtype",
         "FP8 halves KV bytes per token. NVFP4 is not usable on this card.", 0, 0,
         {"bf16", "int8", "fp8", "nvfp4", "k8v4"}},
        {"--gdn-state-dtype", "gdn_state_dtype", ParamKind::Enum, "memory", "Recurrent state dtype",
         "BF16 halves the linear-attention state slots. Error accumulates along a sequence.", 0, 0,
         {"fp32", "bf16"}},
        {"--output-head-dtype", "output_head_dtype", ParamKind::Enum, "memory", "Output head dtype",
         "FP8 saves about 605 MiB and changes greedy output after ~95 tokens.", 0, 0,
         {"bf16", "fp8"}},
        {"--token-embedding-dtype", "token_embedding_dtype", ParamKind::Enum, "memory",
         "Token embedding dtype", "FP8 saves about 605 MiB. Diverges sooner than the head.", 0, 0,
         {"bf16", "fp8"}},
        {"--device-state-slots", "device_state_slots", ParamKind::Int, "memory",
         "Device state slots", "Extra checkpoint capacity on the device. Costs VRAM.", 0, 64, {}},
        {"--host-state-slots", "host_state_slots", ParamKind::Int, "memory", "Host state slots",
         "Checkpoints kept in host RAM. Costs no VRAM.", 0, 256, {}},
        {"--host-kv-mib", "host_kv_mib", ParamKind::Int, "memory", "Host KV (MiB)",
         "KV offloaded to host RAM. Costs no VRAM.", 0, 262144, {}},
        {"--max-private-continuations", "max_private_continuations", ParamKind::Int, "memory",
         "Private continuations", "Per-session checkpoints. Raise for many long agent sessions.", 0,
         512, {}},
        {"--max-shared-prefixes", "max_shared_prefixes", ParamKind::Int, "memory",
         "Shared prefixes", "Cached prefixes shared across sessions.", 0, 512, {}},

        {"--vision", "vision", ParamKind::Flag, "features", "Vision",
         "Loads the vision encoder. Costs about 1.1 GiB.", 0, 0, {}},
        {"--spec", "spec", ParamKind::Enum, "features", "Speculative backend",
         "MTP suits concurrent work; DFlash2 accelerates interactive 27B generation and requires a companion artifact.", 0, 0,
         {"mtp", "dflash", "dflash2"}},
        {"--draft-tokens", "draft_tokens", ParamKind::Int, "features", "Draft tokens",
         "Tokens proposed per step. Start with 5 for 27B MTP or 7 for DFlash2.", 1, 15, {}},
        {"--lm-head-draft", "lm_head_draft", ParamKind::Flag, "features", "Optimized draft head",
         "Uses the optimized proposal head for faster speculation. Recommended for the 27B presets.", 0, 0, {}},
        {"--no-prefix-reuse", "no_prefix_reuse", ParamKind::Flag, "features", "Disable prefix reuse",
         "Turns off cross-request prefix caching. Multi-turn TTFT gets much worse.", 0, 0, {}},
        {"--no-cuda-graph", "no_cuda_graph", ParamKind::Flag, "features", "Disable CUDA graphs",
         "Runs decode eagerly. Slower; use to isolate a graph-related defect.", 0, 0, {}},
        {"--no-thinking", "no_thinking", ParamKind::Flag, "features", "Disable thinking",
         "Server-wide default. Reasoning is normally a per-request choice.", 0, 0, {}},

        {"--model-id", "model_id", ParamKind::Text, "api", "Model id",
         "The name clients send and see. Overrides the artifact's own id.", 0, 0, {}},
        {"--cors", "cors", ParamKind::Flag, "api", "Send CORS headers",
         "Allows browser pages from other origins to call the engine.", 0, 0, {}},
        {"--default-max-tokens", "default_max_tokens", ParamKind::Int, "api", "Default max tokens",
         "Applied when a request omits max_tokens.", 1, 262144, {}},
        {"--default-thinking-budget", "default_thinking_budget", ParamKind::Int, "api",
         "Default thinking budget", "Caps model-origin reasoning tokens per request.", 0, 262144,
         {}},
        {"--log-stats-interval-ms", "log_stats_interval_ms", ParamKind::Int, "api",
         "Throughput log interval (ms)", "0 disables the periodic throughput lines.", 0, 3600000,
         {}},
        {"--max-request-mib", "max_request_mib", ParamKind::Int, "api", "Max request (MiB)",
         "Largest accepted request body.", 1, 4096, {}},
    };
    return specs;
}

inline const EngineParamSpec* find_engine_param(std::string_view key) {
    for (const auto& spec : engine_param_specs()) {
        if (spec.key == key) { return &spec; }
    }
    return nullptr;
}

// Flags whose VALUE is a secret. The value is replaced before the argument vector
// is shown anywhere: supervisor.prod.json embeds a real key, and the dashboard is
// a web page. A path is not a secret; the bytes at that path are.
inline bool engine_flag_value_is_secret(std::string_view flag) {
    return flag == "--api-key";
}

inline std::vector<std::string> redact_engine_args(std::vector<std::string> args) {
    for (std::size_t i = 0; i + 1 < args.size(); ++i) {
        if (engine_flag_value_is_secret(args[i]) && !args[i + 1].empty()) {
            args[i + 1] = "(redacted)";
        }
    }
    return args;
}

struct EngineParam {
    std::string key;
    std::string value; // Flag kind: "true" when present; absent keys are omitted entirely.
};

// Parsed view of the configured argument vector: recognised parameters, plus every
// token this table does not know, preserved verbatim and in order so that rendering
// after an edit cannot silently drop a flag the operator added by hand.
struct ParsedEngineArgs {
    std::string artifact; // first positional (the .ninfer path), if any
    std::vector<EngineParam> params;
    std::vector<std::string> passthrough;
};

inline ParsedEngineArgs parse_engine_args(const std::vector<std::string>& args) {
    ParsedEngineArgs out;
    bool artifact_taken = false;
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string& token = args[i];
        if (!token.starts_with("--")) {
            if (!artifact_taken) {
                out.artifact  = token;
                artifact_taken = true;
            } else {
                out.passthrough.push_back(token);
            }
            continue;
        }
        const EngineParamSpec* spec = nullptr;
        for (const auto& candidate : engine_param_specs()) {
            if (candidate.flag == token) {
                spec = &candidate;
                break;
            }
        }
        if (spec == nullptr) {
            out.passthrough.push_back(token);
            // An unknown flag's value must travel with it, or rendering would
            // promote the value to a positional argument.
            if (i + 1 < args.size() && !args[i + 1].starts_with("--")) {
                out.passthrough.push_back(args[++i]);
            }
            continue;
        }
        if (spec->kind == ParamKind::Flag) {
            out.params.push_back({std::string(spec->key), "true"});
            continue;
        }
        if (i + 1 < args.size()) {
            out.params.push_back({std::string(spec->key), args[++i]});
        }
    }
    return out;
}

inline const std::string* find_param_value(const std::vector<EngineParam>& params,
                                           std::string_view key) {
    for (const auto& p : params) {
        if (p.key == key) { return &p.value; }
    }
    return nullptr;
}

// Rebuilds the argument vector: artifact first, then parameters in table order so
// the file stays diffable across edits, then everything unrecognised, unchanged.
inline std::vector<std::string> render_engine_args(const ParsedEngineArgs& parsed) {
    std::vector<std::string> args;
    if (!parsed.artifact.empty()) { args.push_back(parsed.artifact); }
    for (const auto& spec : engine_param_specs()) {
        const std::string* value = find_param_value(parsed.params, spec.key);
        if (value == nullptr) { continue; }
        if (spec.kind == ParamKind::Flag) {
            if (*value == "true") { args.emplace_back(spec.flag); }
            continue;
        }
        if (value->empty()) { continue; }
        args.emplace_back(spec.flag);
        args.push_back(*value);
    }
    for (const auto& token : parsed.passthrough) { args.push_back(token); }
    return args;
}

inline bool parse_long_long(std::string_view text, long long& out) {
    if (text.empty()) { return false; }
    long long value = 0;
    std::size_t i   = 0;
    const bool neg  = text[0] == '-';
    if (neg) { i = 1; }
    if (i >= text.size()) { return false; }
    for (; i < text.size(); ++i) {
        if (text[i] < '0' || text[i] > '9') { return false; }
        value = value * 10 + (text[i] - '0');
        if (value > 4294967296LL) { return false; }
    }
    out = neg ? -value : value;
    return true;
}

// Returns one message per rejected field. An empty result means the edit is safe
// to write; the caller never applies a partially valid patch.
inline std::vector<std::string> validate_engine_params(const std::vector<EngineParam>& params) {
    std::vector<std::string> errors;
    for (const auto& p : params) {
        const EngineParamSpec* spec = find_engine_param(p.key);
        if (spec == nullptr) {
            errors.push_back(p.key + " is not an editable parameter");
            continue;
        }
        switch (spec->kind) {
        case ParamKind::Flag:
            if (p.value != "true" && p.value != "false") {
                errors.push_back(std::string(spec->label) + " must be true or false");
            }
            break;
        case ParamKind::Int: {
            long long value = 0;
            if (!parse_long_long(p.value, value)) {
                errors.push_back(std::string(spec->label) + " must be a whole number");
            } else if (value < spec->min_value || value > spec->max_value) {
                errors.push_back(std::string(spec->label) + " must be between " +
                                 std::to_string(spec->min_value) + " and " +
                                 std::to_string(spec->max_value));
            }
            break;
        }
        case ParamKind::IntOrAuto: {
            if (p.value == "auto") { break; }
            long long value = 0;
            if (!parse_long_long(p.value, value)) {
                errors.push_back(std::string(spec->label) + " must be a whole number or 'auto'");
            } else if (value < spec->min_value || value > spec->max_value) {
                errors.push_back(std::string(spec->label) + " must be between " +
                                 std::to_string(spec->min_value) + " and " +
                                 std::to_string(spec->max_value) + ", or 'auto'");
            }
            break;
        }
        case ParamKind::Enum: {
            bool ok = false;
            for (const auto& choice : spec->choices) {
                if (!choice.empty() && p.value == choice) {
                    ok = true;
                    break;
                }
            }
            if (!ok) { errors.push_back(std::string(spec->label) + " has an unsupported value"); }
            break;
        }
        case ParamKind::Text:
            // A value is passed to CreateProcess as one argv element and is quoted
            // there, so spaces are safe; control characters are not, and a newline
            // would corrupt the config file this is written back into.
            for (char c : p.value) {
                if (static_cast<unsigned char>(c) < 0x20) {
                    errors.push_back(std::string(spec->label) + " must not contain control "
                                                                "characters");
                    break;
                }
            }
            break;
        }
    }
    return errors;
}

// Cross-field rules the per-field pass cannot see.
inline std::vector<std::string> validate_engine_param_combination(
    const std::vector<EngineParam>& params, std::string_view model_identity = {}) {
    std::vector<std::string> errors;
    const std::string* kv_capacity  = find_param_value(params, "kv_capacity");
    const std::string* max_context  = find_param_value(params, "max_context");
    const std::string* draft_tokens = find_param_value(params, "draft_tokens");
    const std::string* spec_backend = find_param_value(params, "spec");
    if (kv_capacity != nullptr && max_context != nullptr && *kv_capacity != "auto") {
        long long capacity = 0;
        long long context  = 0;
        if (parse_long_long(*kv_capacity, capacity) && parse_long_long(*max_context, context) &&
            capacity < context) {
            errors.emplace_back("KV capacity must be at least the max context, or the engine "
                                "cannot hold one full-length request");
        } else if (parse_long_long(*kv_capacity, capacity) && parse_long_long(*max_context, context)) {
            // Same page bounds the Engine uses: M in [max(L, C), C*L] with 64-token pages.
            long long lanes = 1;
            const std::string* conc = find_param_value(params, "max_concurrency");
            if (conc != nullptr) { parse_long_long(*conc, lanes); }
            if (lanes < 1) { lanes = 1; }
            const auto pages = [](long long tokens) { return (tokens + 63) / 64; };
            const long long logical = pages(context);
            const long long pool    = pages(capacity);
            const long long minimum = logical > lanes ? logical : lanes;
            const long long maximum = lanes * logical;
            if (pool < minimum || pool > maximum) {
                errors.emplace_back("KV pool must fit between one full-length request and "
                                    "max-concurrency full-length requests");
            }
        }
    }
    if (draft_tokens != nullptr && spec_backend == nullptr) {
        errors.emplace_back("Draft tokens has no effect without a speculative backend");
    }
    const auto* head = find_param_value(params, "lm_head_draft");
    if (head != nullptr && *head == "true" && spec_backend == nullptr) {
        errors.emplace_back("Optimized draft head requires a speculative backend");
    }
    if (spec_backend != nullptr) {
        const bool flash = model_identity == "qwen3.8-flash-next";
        const int limit = *spec_backend == "mtp" ? (flash ? 4 : 5) : 15;
        long long count = 0;
        if (draft_tokens && parse_long_long(*draft_tokens, count) && (count < 1 || count > limit)) {
            errors.emplace_back("Draft tokens must be between 1 and " + std::to_string(limit) +
                                " for this model and backend");
        }
        if (!model_identity.empty() &&
            ((*spec_backend == "dflash2" && model_identity != "qwen3.8-27b" &&
              model_identity != "qwen3.8-27b-orcarouter") ||
             (*spec_backend == "dflash" && model_identity != "qwen3.6-35b-a3b"))) {
            errors.emplace_back("Selected speculative backend is not supported by this model");
        }
    }
    return errors;
}

// Keeps the engine's LISTEN address in step with the address the supervisor polls.
//
// These are two different things in the config -- `--host`/`--port` in the argument
// vector decide where the engine listens, while engine_host/engine_port decide where
// health checks and /admin/vram are fetched from -- and nothing kept them in step.
// Editing the port in the dashboard would otherwise leave the supervisor polling a
// port nothing listens on, which presents as an engine that starts and is then
// permanently unhealthy.
inline std::vector<std::string> with_engine_endpoint(std::vector<std::string> args,
                                                     const std::string& host, int port) {
    auto set_flag = [&args](std::string_view flag, const std::string& value) {
        for (std::size_t i = 0; i < args.size(); ++i) {
            if (args[i] != flag) { continue; }
            if (i + 1 < args.size()) {
                args[i + 1] = value;
            } else {
                args.push_back(value);
            }
            return;
        }
        args.emplace_back(flag);
        args.push_back(value);
    };
    if (!host.empty()) { set_flag("--host", host); }
    if (port > 0) { set_flag("--port", std::to_string(port)); }
    return args;
}

// Adds --request-log-jsonl PATH unless the caller already passed one explicitly, in which
// case the explicit flag wins and the config field is ignored -- an operator who wrote the
// flag by hand meant it. An empty path is a no-op, so a config that does not ask for a
// request log does not get one.
inline std::vector<std::string> with_request_log(std::vector<std::string> args,
                                                 const std::string& path) {
    if (path.empty()) { return args; }
    for (const auto& a : args) {
        if (a == "--request-log-jsonl") { return args; }
    }
    args.emplace_back("--request-log-jsonl");
    args.emplace_back(path);
    return args;
}

// Adds --api-key-file PATH so the engine actually authenticates. Without this the
// config's api_key_file was read only by the supervisor's own collector, and the
// engine itself ran wide open: /v1/models answered 200 with no key and with a wrong
// one, on whatever interface it was bound to.
//
// The PATH is passed, never the key. A child's command line is readable by every
// process running as this user, so --api-key would publish the secret to the
// machine; --api-key-file leaves it in the file it already lives in. An explicit
// --api-key or --api-key-file in args wins, as with the request log.
inline std::vector<std::string> with_api_key_file(std::vector<std::string> args,
                                                  const std::string& path) {
    if (path.empty()) { return args; }
    for (const auto& a : args) {
        if (a == "--api-key" || a == "--api-key-file") { return args; }
    }
    args.emplace_back("--api-key-file");
    args.emplace_back(path);
    return args;
}

inline std::vector<std::string> with_desktop_reserve(std::vector<std::string> args, int gib) {
    if (desktop_reserve_pinned(args)) { return args; }
    for (std::size_t i = 0; i < args.size(); ++i) {
        if (args[i] != "--desktop-reserve-gib") { continue; }
        const bool has_value = i + 1 < args.size();
        if (gib <= 0) {
            args.erase(args.begin() + static_cast<std::ptrdiff_t>(i),
                       args.begin() + static_cast<std::ptrdiff_t>(i + (has_value ? 2 : 1)));
            return args;
        }
        if (has_value) {
            args[i + 1] = std::to_string(gib);
        } else {
            args.emplace_back(std::to_string(gib));
        }
        return args;
    }
    if (gib > 0) {
        args.emplace_back("--desktop-reserve-gib");
        args.emplace_back(std::to_string(gib));
    }
    return args;
}

// The engine refuses to start when its runtime reservation does not fit beside what the
// desktop currently holds, and says so on one line:
//   server status=failed phase=startup detail="requested Engine runtime reservation
//   requires 17572624288 bytes, but only 16943534080 bytes are available for runtime capacity"
// A fixed --kv-capacity demands a fixed reservation whatever else is resident, so the same
// plan fails the same way on every retry; the supervisor looped seven times on it. Parsed
// here so the retry can shrink the plan instead.
struct RuntimeReservationShortfall {
    bool matched                 = false;
    std::int64_t required_bytes  = 0;
    std::int64_t available_bytes = 0;
};

inline RuntimeReservationShortfall parse_runtime_reservation_failure(std::string_view line) {
    RuntimeReservationShortfall out;
    const auto read_number = [&](std::string_view text, std::size_t from, std::int64_t& value) {
        std::size_t i = from;
        while (i < text.size() && text[i] == ' ') { ++i; }
        std::int64_t v = 0;
        bool any      = false;
        while (i < text.size() && text[i] >= '0' && text[i] <= '9') {
            v   = v * 10 + (text[i] - '0');
            any = true;
            ++i;
        }
        value = v;
        return any;
    };
    const std::string_view key = "runtime reservation requires ";
    const auto pos             = line.find(key);
    if (pos == std::string_view::npos) { return out; }
    if (!read_number(line, pos + key.size(), out.required_bytes)) { return out; }
    const std::string_view key2 = "but only ";
    const auto pos2             = line.find(key2, pos);
    if (pos2 == std::string_view::npos) { return out; }
    if (!read_number(line, pos2 + key2.size(), out.available_bytes)) { return out; }
    out.matched = out.required_bytes > 0 && out.available_bytes >= 0;
    return out;
}

// The configured KV capacity when it is an explicit token count. 'auto' or an absent flag
// returns nullopt: the engine sizes those itself and there is nothing to adapt.
inline std::optional<std::int64_t> explicit_kv_capacity(const std::vector<std::string>& args) {
    for (std::size_t i = 0; i + 1 < args.size(); ++i) {
        if (args[i] != "--kv-capacity") { continue; }
        const std::string& v = args[i + 1];
        if (v.empty() || v.find_first_not_of("0123456789") != std::string::npos) { return std::nullopt; }
        return std::stoll(v);
    }
    return std::nullopt;
}

// The engine must still admit one full-context request, so the plan never shrinks below
// --max-context (or a modest default when the flag is absent).
inline std::int64_t kv_capacity_floor_tokens(const std::vector<std::string>& args) {
    for (std::size_t i = 0; i + 1 < args.size(); ++i) {
        if (args[i] != "--max-context") { continue; }
        const std::string& v = args[i + 1];
        if (!v.empty() && v.find_first_not_of("0123456789") == std::string::npos) { return std::stoll(v); }
    }
    return 32768;
}

// Tokens to run at after a shortfall. The supervisor does not know the model's KV bytes per
// token, so it assumes no more than 8 KiB (Flash-Next FP8 is about 12 KiB, BF16 twice that):
// assuming less removes more tokens, which is the safe direction. Adds a 256 MiB margin so
// a second attempt is not itself a few megabytes short, rounds down to a 4096-token boundary,
// and never goes below the floor.
inline std::int64_t reduced_kv_capacity(std::int64_t current_tokens, std::int64_t required_bytes,
                                        std::int64_t available_bytes, std::int64_t floor_tokens) {
    constexpr std::int64_t kKvBytesPerTokenFloor = 8LL * 1024;
    constexpr std::int64_t kMarginBytes          = 256LL * 1024 * 1024;
    if (required_bytes <= available_bytes) { return current_tokens; }
    const std::int64_t shortfall = required_bytes - available_bytes + kMarginBytes;
    const std::int64_t drop      = (shortfall + kKvBytesPerTokenFloor - 1) / kKvBytesPerTokenFloor;
    std::int64_t next            = current_tokens - drop;
    next -= next % 4096;
    if (next < floor_tokens) { next = floor_tokens; }
    return next;
}

inline std::vector<std::string> with_kv_capacity(std::vector<std::string> args, std::int64_t tokens) {
    for (std::size_t i = 0; i < args.size(); ++i) {
        if (args[i] != "--kv-capacity") { continue; }
        if (i + 1 < args.size()) {
            args[i + 1] = std::to_string(tokens);
        } else {
            args.emplace_back(std::to_string(tokens));
        }
        return args;
    }
    args.emplace_back("--kv-capacity");
    args.emplace_back(std::to_string(tokens));
    return args;
}

// The reserve the config file already asks for, so the menu can put a checkmark
// on it before the user has ever chosen one. 0 when the flag is absent.
inline int desktop_reserve_from_args(const std::vector<std::string>& args) {
    for (std::size_t i = 0; i + 1 < args.size(); ++i) {
        if (args[i] != "--desktop-reserve-gib") { continue; }
        try {
            return std::stoi(args[i + 1]);
        } catch (...) { return 0; }
    }
    return 0;
}

inline std::uint64_t desktop_reserve_launch_mib(const std::vector<std::string>& args, int preference) {
    const auto effective = preference >= 0 ? with_desktop_reserve(args, preference) : args;
    std::uint64_t mib = 8192;
    for (std::size_t i = 0; i + 1 < effective.size(); ++i) {
        if (effective[i] == "--desktop-reserve-gib") mib = std::stoull(effective[i + 1]) * 1024;
        if (effective[i] == "--desktop-reserve-mib") mib = std::stoull(effective[i + 1]);
    }
    return mib;
}

enum class TrayAction : std::uint8_t {
    None,
    OpenDashboard,
    Start,
    Stop,
    Restart,
    Quit,
    Reserve,
    Idle,
    StartAtLogin,
    OpenLogs,
    CopyUrl,
};

struct TrayCommand {
    TrayAction action = TrayAction::None;
    int value         = 0;
};

// Menu command ids are INDICES into the choice tables, never the values
// themselves. Encoding minutes directly put "After 4 hours" at 200+240=440,
// past the next base, where the dispatch guard silently discarded it.
inline constexpr unsigned kCmdFixedBase   = 1;
inline constexpr unsigned kCmdReserveBase = 100;
inline constexpr unsigned kCmdIdleBase    = 200;
inline constexpr unsigned kCmdEnd         = 300;

inline constexpr unsigned kCmdOpenDashboard = kCmdFixedBase + 0;
inline constexpr unsigned kCmdStart         = kCmdFixedBase + 1;
inline constexpr unsigned kCmdStop          = kCmdFixedBase + 2;
inline constexpr unsigned kCmdRestart       = kCmdFixedBase + 3;
inline constexpr unsigned kCmdQuit          = kCmdFixedBase + 4;
inline constexpr unsigned kCmdStartAtLogin  = kCmdFixedBase + 6;
inline constexpr unsigned kCmdOpenLogs      = kCmdFixedBase + 7;
inline constexpr unsigned kCmdCopyUrl       = kCmdFixedBase + 8;

static_assert(kCmdCopyUrl < kCmdReserveBase, "fixed ids must not reach the reserve range");
static_assert(kCmdReserveBase + kReserveChoicesGib.size() <= kCmdIdleBase,
              "reserve ids must not reach the idle range");
static_assert(kCmdIdleBase + kIdleChoicesMinutes.size() <= kCmdEnd,
              "idle ids must not leave the dispatch range");

inline unsigned tray_cmd_reserve(std::size_t index) {
    return kCmdReserveBase + static_cast<unsigned>(index);
}
inline unsigned tray_cmd_idle(std::size_t index) {
    return kCmdIdleBase + static_cast<unsigned>(index);
}

inline TrayCommand decode_tray_command(unsigned cmd) {
    if (cmd >= kCmdReserveBase && cmd < kCmdIdleBase) {
        const std::size_t i = cmd - kCmdReserveBase;
        if (i >= kReserveChoicesGib.size()) { return {}; }
        return {TrayAction::Reserve, kReserveChoicesGib[i]};
    }
    if (cmd >= kCmdIdleBase && cmd < kCmdEnd) {
        const std::size_t i = cmd - kCmdIdleBase;
        if (i >= kIdleChoicesMinutes.size()) { return {}; }
        return {TrayAction::Idle, kIdleChoicesMinutes[i]};
    }
    switch (cmd) {
    case kCmdOpenDashboard: return {TrayAction::OpenDashboard, 0};
    case kCmdStart: return {TrayAction::Start, 0};
    case kCmdStop: return {TrayAction::Stop, 0};
    case kCmdRestart: return {TrayAction::Restart, 0};
    case kCmdQuit: return {TrayAction::Quit, 0};
    case kCmdStartAtLogin: return {TrayAction::StartAtLogin, 0};
    case kCmdOpenLogs: return {TrayAction::OpenLogs, 0};
    case kCmdCopyUrl: return {TrayAction::CopyUrl, 0};
    default: break;
    }
    return {};
}

// What one line of engine stderr tells the supervisor. The engine already
// writes everything needed here (spdlog pattern `[ts] [level] [name] msg`), so
// this costs no extra I/O: the pipe is read for engine.log regardless.
struct EngineLineSignal {
    bool activity = false;
    bool ready    = false;
    bool error    = false;
    bool fatal    = false;
    std::string message;
};

inline bool contains_sv(std::string_view hay, std::string_view needle) {
    return hay.find(needle) != std::string_view::npos;
}

inline EngineLineSignal classify_engine_line(std::string_view raw) {
    EngineLineSignal out;
    const std::string_view line = trim_sv(raw);
    if (line.empty()) { return out; }

    // Activity: a request arriving or finishing, or the periodic throughput
    // report -- which the engine only writes while something is running or
    // waiting. Either is proof the engine is not idle.
    out.activity = contains_sv(line, "request id=") || contains_sv(line, "throughput interval_ms=");
    out.ready    = contains_sv(line, "server status=ready");
    out.fatal    = contains_sv(line, "server status=failed");
    out.error    = out.fatal || contains_sv(line, "] [error] ") ||
                contains_sv(line, "] [critical] ") || line.rfind("ninfer-serve: ", 0) == 0;

    // The message is what a person needs in a balloon. Two producers: the
    // logger (`[ninfer-serve] msg`) and the argument parser, which fails before
    // the logger exists and writes `ninfer-serve: msg` straight to stderr.
    constexpr std::string_view kCerr   = "ninfer-serve: ";
    constexpr std::string_view kLogger = "[ninfer-serve] ";
    std::string_view msg;
    if (const auto p = line.find(kCerr); p != std::string_view::npos) {
        msg = line.substr(p + kCerr.size());
    } else if (const auto q = line.find(kLogger); q != std::string_view::npos) {
        msg = line.substr(q + kLogger.size());
    }
    msg = trim_sv(msg);
    if (msg.size() > 200) { msg = msg.substr(0, 200); }
    out.message = std::string(msg);
    return out;
}

// Is this error about the ENGINE, or about one request?
//
// `[error] [ninfer-serve] request id=812 status=failed ...` is an internal
// failure of a single request; the engine is still up and serving everything
// else. Storing it as the engine's "why" put a stale, unrelated request failure
// in the tooltip and in the "Last error:" clause of a later crash-loop balloon,
// attributing the halt to it. A fatal line stays process-scoped even though it
// can mention a request.
inline bool is_process_scoped_error(const EngineLineSignal& sig) {
    return sig.error && (sig.fatal || !sig.activity);
}

// One request's lifecycle, from the same stderr line the activity signal reads.
//
// This is the hard floor under "unload when idle": the log-cadence signals can
// all be silenced (--log-stats-interval-ms 0 removes the throughput line, and a
// config need not have a request log), and a 20-minute generation would then
// look idle and be terminated mid-answer. A request that opened and has not
// closed is proof of work in flight regardless of any cadence.
//
// Formats (src/serve/operational_log.cpp): open is `request id=<n>
// status=submitted ...`; close is status=done / failed / rejected / cancelled.
// `response request_id=` is deliberately not matched -- it is a second record
// about a request that has its own terminal line.
struct RequestLifecycle {
    bool matched         = false; // the line is about a specific request
    std::uint64_t id     = 0;
    bool opens           = false;
    bool closes          = false;
};

inline RequestLifecycle classify_request_line(std::string_view raw) {
    RequestLifecycle out;
    const std::string_view line = trim_sv(raw);
    constexpr std::string_view kReq = "request id=";
    const auto p                    = line.find(kReq);
    if (p == std::string_view::npos) { return out; }
    std::size_t i          = p + kReq.size();
    std::uint64_t id       = 0;
    std::size_t digits     = 0;
    while (i < line.size() && line[i] >= '0' && line[i] <= '9') {
        // Ignore overflow past 19 digits rather than wrapping: an id that long
        // is not something this engine emits, and a wrapped one could collide.
        if (digits < 19) { id = id * 10 + static_cast<std::uint64_t>(line[i] - '0'); }
        ++digits;
        ++i;
    }
    if (digits == 0 || digits > 19) { return out; }
    out.matched = true;
    out.id      = id;
    constexpr std::string_view kStatus = " status=";
    const auto s                       = line.find(kStatus, i);
    if (s == std::string_view::npos) { return out; }
    const std::size_t b = s + kStatus.size();
    std::size_t e       = b;
    while (e < line.size() && line[e] != ' ') { ++e; }
    const std::string_view status = line.substr(b, e - b);
    out.opens                     = status == "submitted";
    out.closes = status == "done" || status == "failed" || status == "rejected" ||
                 status == "cancelled";
    return out;
}

// Inputs to "unload the model because nobody is using it". Kept as a struct so
// the decision is one pure function: killing a live engine mid-request is the
// worst thing this program can do, so the rule must be testable in isolation.
struct IdleInputs {
    std::int64_t now_ms           = 0;
    std::int64_t last_activity_ms = 0;
    int idle_minutes              = 0;
    bool running                  = false;
    bool healthy                  = false;
    // Requests the engine has opened and not closed. The activity clock depends
    // on the engine's log cadence; this does not, and it is what makes "cannot
    // unload mid-request" absolute rather than conditional on a flag.
    int inflight_requests         = 0;
};

inline bool idle_unload_due(const IdleInputs& in) {
    if (in.idle_minutes <= 0 || !in.running || !in.healthy) { return false; }
    if (in.inflight_requests > 0) { return false; }
    // No activity clock yet (engine just adopted, log not yet seen): never
    // unload. An unknown last-use time is not the same as a long-ago one.
    if (in.last_activity_ms <= 0) { return false; }
    const std::int64_t window_ms = static_cast<std::int64_t>(in.idle_minutes) * 60000;
    return in.now_ms - in.last_activity_ms >= window_ms;
}

// Balloon body for a crash-loop halt. NOTIFYICONDATA::szInfo is 256 wchar_t
// including the terminator, so 255 is the hard ceiling and the error text is
// what gets trimmed, never the counts.
inline std::string format_halt_notice(int recent_exits, int window_s, int exit_code,
                                      std::string_view last_error) {
    constexpr std::size_t kMax = 255;
    const std::string tail     = " Open the dashboard for the log.";
    std::string out = "Crashed " + std::to_string(recent_exits) + " times in " +
                      std::to_string(window_s) + " s (exit code " + std::to_string(exit_code) + ").";
    const std::string_view err = trim_sv(last_error);
    const std::size_t fixed    = out.size() + tail.size() + 14; // " Last error: " + "."
    if (!err.empty() && fixed < kMax) {
        std::string e(err);
        const std::size_t room = kMax - fixed;
        if (e.size() > room) { e.resize(room); }
        out += " Last error: " + e + ".";
    }
    if (out.size() + tail.size() <= kMax) { out += tail; }
    if (out.size() > kMax) { out.resize(kMax); }
    return out;
}

// The model a person recognises, for the menu header. The checkpoint path is the
// one argument that is always present and always specific.
inline std::string model_label_from_args(const std::vector<std::string>& args) {
    std::string picked;
    for (const auto& a : args) {
        if (a.size() > 7 && a.compare(a.size() - 7, 7, ".ninfer") == 0) {
            picked = a;
            break;
        }
    }
    if (picked.empty() && !args.empty() && !args.front().empty() && args.front().front() != '-') {
        picked = args.front();
    }
    if (picked.empty()) { return {}; }
    const auto slash = picked.find_last_of("/\\");
    std::string base = slash == std::string::npos ? picked : picked.substr(slash + 1);
    if (base.size() > 7 && base.compare(base.size() - 7, 7, ".ninfer") == 0) {
        base.resize(base.size() - 7);
    }
    return base;
}

// Named-mutex identity for "a supervisor is already running for this config".
// Keyed on the config path rather than on the executable: two configs are two
// engines and may legitimately coexist, while two supervisors on one config race
// for the same engine port and crash-loop each other. Hashed because a kernel
// object name may not contain a backslash after the namespace prefix.
inline std::string single_instance_name(std::string_view canonical_config_path) {
    std::uint64_t h = 1469598103934665603ull; // FNV-1a 64 offset basis
    for (const char c : canonical_config_path) {
        auto u = static_cast<unsigned char>(c);
        if (u == '\\') { u = '/'; }
        if (u >= 'A' && u <= 'Z') { u = static_cast<unsigned char>(u - 'A' + 'a'); }
        h ^= u;
        h *= 1099511628211ull;
    }
    static constexpr char kHex[] = "0123456789abcdef";
    std::string out              = "Local\\NInferSupervisor.";
    for (int i = 15; i >= 0; --i) {
        out.push_back(kHex[(h >> (i * 4)) & 0xFull]);
    }
    return out;
}

inline std::string run_at_login_command(std::string_view module_path,
                                        std::string_view config_abs_path) {
    return "\"" + std::string(module_path) + "\" --config \"" + std::string(config_abs_path) + "\"";
}

// "Start at login" must mean "at login, for THIS config". There is one Run
// value name, so testing that it merely exists put a checkmark on the menu of
// whichever config happened to be running -- including the one the entry does
// not point at, whose toggle would then delete the other's entry. Compare the
// stored command instead. Whitespace-tolerant because the value may have been
// hand-edited in regedit; slash- and case-insensitive because the installer
// and regedit write native "C:\..." paths while the supervisor spells its
// canonical config path with forward slashes. Comparing those literally made
// the installer's own entry show as unchecked, and a click then replaced or
// removed an entry that was already correct.
inline bool run_at_login_command_matches(std::string_view stored, std::string_view expected) {
    const auto fold = [](std::string_view s) {
        std::string out(trim_sv(s));
        for (char& c : out) {
            if (c == '\\') { c = '/'; }
            if (c >= 'A' && c <= 'Z') { c = static_cast<char>(c - 'A' + 'a'); }
        }
        return out;
    };
    const std::string want = fold(expected);
    return !want.empty() && fold(stored) == want;
}

} // namespace ninfer::supervisor
