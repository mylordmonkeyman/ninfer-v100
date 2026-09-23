#include "logic.hpp"
#include "config.hpp"
#include "insights.hpp"
#include "tray_prefs.hpp"
#include "reserve_budget.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>

namespace {

int fail(const std::string& m) {
    std::cerr << "FAIL: " << m << '\n';
    return 1;
}
int check(bool c, const std::string& m) { return c ? 0 : fail(m); }

int test_loopback() {
    using namespace ninfer::supervisor;
    int f = 0;
    f += check(is_loopback_host("127.0.0.1") && is_loopback_host("localhost") &&
                   is_loopback_host("::1"),
               "loopback hosts");
    f += check(!is_loopback_host("0.0.0.0") && !is_loopback_host("192.168.1.2"),
               "non-loopback hosts");
    f += check(is_loopback_peer("127.0.0.1") && is_loopback_peer("::ffff:127.0.0.1"),
               "loopback peers");
    f += check(!is_loopback_peer("10.0.0.8") && !is_loopback_peer(""), "off-box peers");
    return f;
}

int test_model_reserve_budget() {
    using namespace ninfer::supervisor;
    constexpr std::uint64_t gib = 1073741824ULL;
    nlohmann::json report{{"plan", {{"runtime_reservation_bytes", 16 * gib}, {"minimum_runtime_reservation_bytes", 9 * gib}, {"available_after_weights_bytes", 18 * gib}, {"cuda_graph_allowance_bytes", gib}}},
        {"arenas", {{"weights", {{"capacity_bytes", 70 * gib}}}}},
        {"desktop_reserve", {{"configured_bytes", 4 * gib}}}, {"device", {{"free_bytes", 8 * gib}}}};
    int f = 0;
    f += check(model_reserve_budget(report, {"--kv-capacity", "786432"})["max_gib"] == 6,
               "reserve retains explicit runtime capacity rather than just model minimum");
    f += check(model_reserve_budget(report, {"--kv-capacity", "auto"})["max_gib"] == 12,
               "automatic capacity may shrink to its model minimum while retaining slack");
    f += check(model_reserve_budget(report, {"--kv-capacity", "auto", "--kv-slack-floor-mib", "0"})["max_gib"] == 12,
               "automatic headroom remains required even with zero slack floor");
    f += check(model_reserve_budget(report, {"--kv-capacity", "786432", "--kv-slack-floor-mib", "4096"})["max_gib"] == 6,
               "explicit capacity matches serving policy and ignores automatic slack option");
    report["device"]["free_bytes"] = 4 * gib;
    f += check(model_reserve_budget(report, {"--kv-capacity", "auto"})["max_gib"] == 9,
               "live app pressure still leaves automatic sizing headroom");
    f += check(model_reserve_budget(report, {})["max_gib"] == 3,
               "other app pressure and graph headroom reduce limit");
    report["device"]["free_bytes"] = gib / 2;
    f += check(model_reserve_budget(report, {})["max_gib"] == 0, "no negative or underflowed reserve limit");
    f += check(!model_reserve_budget(nullptr, {})["ok"].get<bool>(), "no fabricated limit without model plan");
    f += check(without_reserve_args({"model.ninfer", "--desktop-reserve-gib", "3", "--kv-capacity", "400"}) == std::vector<std::string>({"model.ninfer", "--kv-capacity", "400"}), "model-setting identity ignores only reserve flags");
    return f;
}

int test_engine_probe_destination() {
    using namespace ninfer::supervisor;
    EngineSpec spec;
    int f = 0;
    spec.engine_host = "0.0.0.0";
    f += check(engine_connect_host(spec) == "127.0.0.1" && spec.engine_host == "0.0.0.0",
               "probe wildcard IPv4 engine locally without changing its listener");
    spec.engine_host = "::";
    f += check(engine_connect_host(spec) == "::1" && spec.engine_host == "::",
               "probe wildcard IPv6 engine on IPv6 loopback");
    spec.engine_host = "192.168.1.8";
    f += check(engine_connect_host(spec) == "192.168.1.8",
               "preserve a specifically configured engine destination");
    return f;
}

int test_crash_loop() {
    using clock = std::chrono::steady_clock;
    ninfer::supervisor::RestartPolicy p;
    p.crash_loop_max      = 3;
    p.crash_loop_window_s = 60;
    ninfer::supervisor::RestartGate g(p);
    const auto t0 = clock::now();
    int f         = 0;
    f += check(g.note_exit(t0) && g.note_exit(t0 + std::chrono::seconds(1)), "first exits allowed");
    f += check(!g.note_exit(t0 + std::chrono::seconds(2)) && g.halted(),
               "third exit in window must halt");
    g.reset_halt();
    f += check(!g.halted() && g.note_exit(t0 + std::chrono::seconds(120)),
               "reset allows restart");
    return f;
}

int test_backoff() {
    ninfer::supervisor::RestartGate g;
    int f = 0;
    f += check(g.backoff_seconds() == 1, "initial backoff 1s");
    g.advance_backoff();
    f += check(g.backoff_seconds() == 2, "backoff 2s");
    g.advance_backoff();
    g.advance_backoff();
    g.advance_backoff();
    g.advance_backoff();
    g.advance_backoff();
    f += check(g.backoff_seconds() == 60, "backoff caps at 60s");
    g.note_healthy();
    f += check(g.backoff_seconds() == 1, "healthy resets backoff");
    return f;
}

int test_config_bind() {
    int f = 0;
    {
        auto c = ninfer::supervisor::load_config_json(
            R"({"engine":{"executable":"x"},"models":[{"id":"interactive","name":"27B Interactive","description":"DFlash2, seven drafts","artifact":"model.ninfer","args":["--spec","dflash2","--draft-tokens","7","--lm-head-draft"]}]})");
        const auto restored = ninfer::supervisor::load_config_json(
            ninfer::supervisor::config_to_json(c).dump());
        f += check(restored.models.size() == 1 && restored.models[0].name == "27B Interactive" &&
                   restored.models[0].description == "DFlash2, seven drafts" &&
                   restored.models[0].args == c.models[0].args,
                   "preset names, descriptions and launch flags survive configuration saves");
    }
    const char* ok =
        R"({"engine":{"executable":"C:/ninfer-serve.exe"},"supervisor":{"host":"127.0.0.1"}})";
    try {
        const auto c = ninfer::supervisor::load_config_json(ok);
        f += check(c.host == "127.0.0.1" && !c.bind_any, "loopback config");
    } catch (...) { f += fail("loopback config threw"); }
    bool rejected = false;
    try {
        (void)ninfer::supervisor::load_config_json(
            R"({"engine":{"executable":"x"},"supervisor":{"host":"0.0.0.0"}})");
    } catch (const std::invalid_argument&) { rejected = true; }
    f += check(rejected, "0.0.0.0 without bind_any must be rejected");
    bool any_ok = false;
    try {
        const auto c = ninfer::supervisor::load_config_json(
            R"({"engine":{"executable":"x"},"supervisor":{"host":"0.0.0.0","bind_any":true}})");
        any_ok = c.bind_any;
    } catch (...) {}
    f += check(any_ok, "bind_any allows 0.0.0.0");
    return f;
}

int test_host_header() {
    using namespace ninfer::supervisor;
    int f = 0;
    f += check(host_header_allowed("127.0.0.1:8099", 8099, "127.0.0.1", false),
               "loopback ipv4 host");
    f += check(host_header_allowed("localhost:8099", 8099, "127.0.0.1", false),
               "localhost host");
    f += check(host_header_allowed("[::1]:8099", 8099, "127.0.0.1", false), "ipv6 loopback host");
    f += check(host_header_allowed("127.0.0.1", 8099, "127.0.0.1", false),
               "loopback host without port");
    f += check(!host_header_allowed("attacker.example", 8099, "127.0.0.1", false),
               "rebinding host rejected");
    f += check(!host_header_allowed("attacker.example:8099", 8099, "127.0.0.1", false),
               "rebinding host:port rejected");
    f += check(!host_header_allowed("127.0.0.1:8080", 8099, "127.0.0.1", false),
               "wrong port rejected");
    f += check(!host_header_allowed("", 8099, "127.0.0.1", false), "empty host rejected");
    f += check(!host_header_allowed("192.168.1.5:8099", 8099, "0.0.0.0", true),
               "bind-any 0.0.0.0 does not open Host allowlist");
    f += check(host_header_allowed("192.168.1.5:8099", 8099, "192.168.1.5", true),
               "bind-any named host is allowed");
    f += check(!host_header_allowed("192.168.1.5:8099", 8099, "192.168.1.5", false),
               "named host without bind_any rejected");
    // Suffix/prefix traps. These pass a naive substring or starts_with check and
    // are the classic way a rebinding defense gets reintroduced as a bug: an
    // attacker controls the whole label, so "localhost.evil.com" is evil.com.
    f += check(!host_header_allowed("localhost.evil.com:8099", 8099, "127.0.0.1", false),
               "localhost-prefixed attacker domain rejected");
    f += check(!host_header_allowed("127.0.0.1.evil.com:8099", 8099, "127.0.0.1", false),
               "ip-prefixed attacker domain rejected");
    f += check(!host_header_allowed("evil-localhost:8099", 8099, "127.0.0.1", false),
               "localhost-suffixed attacker domain rejected");
    f += check(!host_header_allowed("192.168.1.5.evil.com:8099", 8099, "192.168.1.5", true),
               "bind-any named host is matched exactly, not as a prefix");
    f += check(supervisor_control_header_ok("1") && !supervisor_control_header_ok("") &&
                   !supervisor_control_header_ok("true"),
               "control header is exactly 1");
    return f;
}

int test_nvidia_csv() {
    using namespace ninfer::supervisor;
    int f     = 0;
    const auto a = parse_nvidia_smi_memory_csv("0, 24576, 32607\n1, 10, 20\n", 0);
    f += check(a.ok && a.used_mib == 24576 && a.total_mib == 32607, "device 0 csv");
    const auto b = parse_nvidia_smi_memory_csv("0, 1, 2\n1, 99, 100\n", 1);
    f += check(b.ok && b.used_mib == 99 && b.total_mib == 100, "device 1 csv");
    const auto c = parse_nvidia_smi_memory_csv("0, 1, 2\n", 3);
    f += check(!c.ok && !c.error.empty(), "missing device");
    const auto d = parse_nvidia_smi_memory_csv("", 0);
    f += check(!d.ok, "empty csv");
    f += check(mib_to_bytes(1) == 1048576, "mib_to_bytes");
    return f;
}

int test_kv_line() {
    using namespace ninfer::supervisor;
    int f = 0;
    const char* log =
        "[info] ninfer-serve: model loaded in 1.2 s\n"
        "[info] ninfer-serve: KV capacity auto resolved=8192 tokens pages=1/2 "
        "runtime=1 prefix-cache=2 free-after-weights=3 free-after-startup=4 "
        "headroom=5 slack=6 graphs=7/8\n"
        "later line\n";
    const auto line = extract_kv_capacity_line(log);
    f += check(line.find("KV capacity auto resolved=8192") != std::string::npos,
               "extracts last KV capacity line");
    f += check(extract_kv_capacity_line("no capacity here").empty(), "missing line");
    return f;
}

int test_monitor_only_config() {
    int f = 0;
    try {
        const auto c = ninfer::supervisor::load_config_json(
            R"({"engine":{"unmanaged":true,"engine_port":8010},"supervisor":{"host":"127.0.0.1"}})");
        f += check(!ninfer::supervisor::manages_engine_process(c) && c.engine.unmanaged,
                   "unmanaged does not require executable");
    } catch (...) { f += fail("unmanaged config threw"); }
    try {
        const auto c = ninfer::supervisor::load_config_json(
            R"({"engine":{"engine_port":8010},"supervisor":{"host":"127.0.0.1"}})", true);
        f += check(c.monitor_only && !ninfer::supervisor::manages_engine_process(c),
                   "CLI monitor_only does not require executable");
    } catch (...) { f += fail("monitor_only cli config threw"); }
    bool rejected = false;
    try {
        (void)ninfer::supervisor::load_config_json(
            R"({"engine":{},"supervisor":{"host":"127.0.0.1"}})");
    } catch (const std::invalid_argument&) { rejected = true; }
    f += check(rejected, "managed config still requires executable");
    return f;
}

int test_insights_honesty() {
    using namespace ninfer::supervisor;
    int f = 0;
    const auto missing = insights_from_request_log_path("");
    f += check(missing.at("source").at("request_log") == "unconfigured", "unconfigured source");
    f += check(missing.at("insights").size() >= 1 &&
                   missing.at("insights").at(0).at("availability") == "unavailable",
               "missing log is unavailable, not a zero");
    f += check(missing.at("insights").at(0).at("statement").get<std::string>().find(
                   "no request_done records") != std::string::npos,
               "unavailable statement");

    const auto typed = analyze_request_log_jsonl(
        R"({"type":"request_done","timestamp_unix_ms":1})"
        "\n",
        "mem");
    f += check(typed.at("insights").at(0).at("availability") == "unavailable",
               "type-key records do not count as request_done");

    const char* jsonl =
        R"({"event":"request_start","server_instance_id":"a","timestamp_unix_ms":1000,"request":{"request_id":1,"enable_thinking":true,"tool_count":0,"requested_output_tokens":8}})"
        "\n"
        R"({"event":"request_done","server_instance_id":"a","timestamp_unix_ms":1600,"request":{"request_id":1,"enable_thinking":true,"tool_count":0,"requested_output_tokens":8},"result":{"finish_reason":"output_limit","completion_tokens":8},"timings_seconds":{"prepare":0.01,"prefill":0.02,"decode":0.02,"ttft":0.03,"total":0.1}})"
        "\n"
        R"({"event":"request_start","server_instance_id":"b","timestamp_unix_ms":2000,"request":{"request_id":1,"enable_thinking":false,"tool_count":0,"requested_output_tokens":16}})"
        "\n"
        R"({"event":"throughput","server_instance_id":"a","timestamp_unix_ms":1601,"scheduler":{"waiting":2,"running":1,"prefilling":0}})"
        "\n";
    const auto r = analyze_request_log_jsonl(jsonl, "mem");
    bool saw_sat = false, saw_limit = false, saw_content = false, saw_ttft = false;
    for (const auto& it : r.at("insights")) {
        const auto id = it.at("id").get<std::string>();
        if (id.find("latency.") == 0 && id.find("ttft") == std::string::npos) {
            saw_sat = true;
            f += check(it.at("availability") == "available" && it.at("confidence") == "measured",
                       "saturation is measured");
            f += check(it.at("measured_over").at("requests") == 1, "measured_over.requests is 1");
            f += check(it.at("evidence").at("queued") == 1, "0.5s queue wait classifies queued");
        }
        if (id.find("ttft") != std::string::npos) {
            saw_ttft = true;
            f += check(it.at("availability") == "available", "ttft split is measured");
            f += check(it.at("evidence").contains("mean_prepare_s") &&
                           it.at("evidence").contains("mean_prefill_s"),
                       "ttft evidence has the split");
        }
        if (id == "client.output_limit_while_thinking") {
            saw_limit = true;
            f += check(it.at("evidence").at("output_limit_thinking") == 1, "output_limit counted");
            f += check(it.at("evidence").at("sample_request_ids").at(0) == 1, "request_id evidence");
        }
        if (id == "client.content_fields") {
            saw_content = true;
            f += check(it.at("availability") == "unavailable",
                       "content fields unavailable, not fabricated");
            f += check(it.at("measured_over").at("requests") == 1,
                       "unavailable measured_over is examined count, not 0");
            f += check(!it.contains("recommendation"), "empty recommendation is omitted");
        }
    }
    f += check(saw_sat && saw_limit && saw_content && saw_ttft, "required insight ids present");
    return f;
}

int test_admin_vram_markers() {
    using namespace ninfer::supervisor;
    AdminVramCursor c;
    std::string kind;
    int f = 0;
    f += check(!c.observe("", "", kind), "first poll is baseline, does not fire");
    f += check(c.observe("release", "seed store released", kind) && kind == "vram_release",
               "empty -> release fires vram_release");
    f += check(c.observe("reclaim", "seed store reclaimed", kind) && kind == "vram_reclaim",
               "release -> reclaim fires vram_reclaim");
    f += check(!c.observe("reclaim", "seed store reclaimed", kind), "unchanged does not fire");
    return f;
}

int test_insights_pinned_tier() {
    using namespace ninfer::supervisor;
    nlohmann::json report = {{"insights", nlohmann::json::array()}};
    nlohmann::json admin  = {
        {"last_transition", ""},
        {"last_reason", ""},
        {"tiers",
         nlohmann::json::array(
             {{{"name", "seed"},
               {"min_bytes", 4294967296ull},
               {"max_bytes", 4294967296ull},
               {"reclaimable_bytes", 0},
               {"released", false}}})}};
    append_admin_vram_insights(report, admin, "");
    int f     = 0;
    bool saw  = false;
    for (const auto& it : report.at("insights")) {
        if (it.at("id") == "vram.tier_pinned_unreleasable") {
            saw = true;
            f += check(it.at("severity") == "warning", "pinned tier is warning");
            f += check(it.at("availability") == "available", "config trap is measured");
        }
    }
    f += check(saw, "pinned min==max insight present");

    nlohmann::json elastic_report = {{"insights", nlohmann::json::array()}};
    nlohmann::json elastic        = {
        {"last_transition", ""},
        {"last_reason", ""},
        {"tiers",
         nlohmann::json::array(
             {{{"name", "seed"},
               {"min_bytes", 0},
               {"max_bytes", 4294967296ull},
               {"reclaimable_bytes", 4294967296ull},
               {"released", false}}})}};
    append_admin_vram_insights(elastic_report, elastic, "");
    bool elastic_fired = false;
    for (const auto& it : elastic_report.at("insights")) {
        if (it.at("id") == "vram.tier_pinned_unreleasable") { elastic_fired = true; }
    }
    f += check(!elastic_fired, "elastic min!=max does not fire pinned insight");
    return f;
}

int test_insights_prefix() {
    using namespace ninfer::supervisor;
    const char* jsonl =
        R"({"event":"request_start","server_instance_id":"a","timestamp_unix_ms":1,"request":{"request_id":1,"message_count":1}})"
        "\n"
        R"({"event":"request_done","server_instance_id":"a","timestamp_unix_ms":2,"request":{"request_id":1,"message_count":1},"result":{"prefix_reuse_path":"full_reset","prompt_tokens":100,"prefix_cache_hit_tokens":0},"timings_seconds":{"total":0.05,"prepare":0.01,"prefill":0.02,"decode":0.02,"ttft":0.03,"vision":0}})"
        "\n"
        R"({"event":"request_start","server_instance_id":"a","timestamp_unix_ms":3,"request":{"request_id":2,"message_count":5}})"
        "\n"
        R"({"event":"request_done","server_instance_id":"a","timestamp_unix_ms":4,"request":{"request_id":2,"message_count":5},"result":{"prefix_reuse_path":"full_reset","prompt_tokens":7749,"prefix_cache_hit_tokens":0},"timings_seconds":{"total":0.05,"prepare":0.01,"prefill":0.02,"decode":0.02,"ttft":0.03,"vision":0}})"
        "\n";
    const auto r = analyze_request_log_jsonl(jsonl, "mem");
    int f = 0;
    bool mix = false, miss = false;
    for (const auto& it : r.at("insights")) {
        const auto id = it.at("id").get<std::string>();
        if (id == "prefix.reuse_mix") {
            mix = true;
            f += check(it.at("evidence").at("full_reset_single_turn") == 1, "single-turn reset");
            f += check(it.at("evidence").at("full_reset_multi_turn") == 1, "multi-turn reset");
            f += check(it.at("availability") == "available", "mix is measured");
        }
        if (id == "prefix.multiturn_full_reset") {
            miss = true;
            f += check(it.at("severity") == "warning", "multi-turn reset is a warning");
            f += check(it.at("evidence").at("samples").at(0).at("prompt_tokens") == 7749,
                       "live-shaped sample");
            f += check(it.at("measured_over").at("requests") == 2, "examined both dones");
        }
    }
    f += check(mix && miss, "prefix insights present");
    return f;
}

int test_insights_speculative() {
    using namespace ninfer::supervisor;
    int f = 0;
    auto find = [](const nlohmann::json& report) -> const nlohmann::json* {
        for (const auto& it : report.at("insights")) {
            if (it.at("id") == "speculative.draft_acceptance") { return &it; }
        }
        return nullptr;
    };
    auto done = [](int id, const char* speculative, int completion = 64) {
        return std::string(R"({"event":"request_done","server_instance_id":"a","timestamp_unix_ms":)") +
               std::to_string(1000 + id) + R"(,"request":{"request_id":)" + std::to_string(id) +
               R"(,"message_count":1},"result":{"finish_reason":"stop","completion_tokens":)" +
               std::to_string(completion) + R"(,"prompt_tokens":100},"timings_seconds":{"total":0.5,"prepare":0.01,"prefill":0.1,"decode":0.39,"ttft":0.11,"vision":0},"speculative":)" +
               speculative + "}\n";
    };

    // Strong early acceptance, weak late positions: the measured curve decays and the window
    // advice is inferred from it. 100 draft rounds of window 5 per request, no fallback.
    const std::string decaying =
        done(1, R"({"backend":"mtp","draft_window":5,"rounds":100,"drafted_tokens":500,"accepted_tokens":210,"fallback_steps":0,"accepted_per_position":[90,70,40,8,2]})") +
        done(2, R"({"backend":"mtp","draft_window":5,"rounds":100,"drafted_tokens":500,"accepted_tokens":210,"fallback_steps":0,"accepted_per_position":[90,70,40,8,2]})");
    const auto decaying_report = analyze_request_log_jsonl(decaying, "mem");
    const auto* decay          = find(decaying_report);
    f += check(decay != nullptr, "speculative insight present when telemetry exists");
    if (decay != nullptr) {
        const auto& e = decay->at("evidence");
        f += check(decay->at("availability") == "available" && decay->at("confidence") == "measured",
                   "acceptance counters are measured");
        f += check(decay->at("severity") == "info", "no fallback pressure is info, not warning");
        f += check(e.at("backend") == "mtp" && e.at("draft_window") == 5, "backend and window carried");
        f += check(e.at("drafted_tokens") == 1000 && e.at("accepted_tokens") == 420 &&
                       e.at("draft_rounds") == 200,
                   "counters summed over the window");
        f += check(std::fabs(e.at("acceptance_ratio").get<double>() - 0.42) < 1e-9, "acceptance ratio");
        f += check(e.at("accepted_per_position") == nlohmann::json({180, 140, 80, 16, 4}),
                   "per-position acceptance summed");
        const auto& rate = e.at("per_position_rate");
        f += check(rate.size() == 5 && std::fabs(rate.at(0).get<double>() - 0.9) < 1e-9 &&
                       std::fabs(rate.at(2).get<double>() - 0.4) < 1e-9 &&
                       std::fabs(rate.at(4).get<double>() - 0.02) < 1e-9,
                   "per-position rate is accepted / draft rounds covering that position");
        f += check(e.at("effective_draft_window") == 3 && e.at("inferred_draft_window") == 3,
                   "positions 4..5 below 20% -> effective window 3");
        f += check(decay->contains("recommendation") &&
                       decay->at("recommendation").get<std::string>().find("Inferred") != std::string::npos &&
                       decay->at("recommendation").get<std::string>().find("--draft-tokens 3") != std::string::npos,
                   "window advice is marked inferred and names the window");
        f += check(e.at("fallback_steps") == 0 && e.at("fallback_share") == 0.0, "no fallback measured");
        f += check(e.at("sample_request_ids") == nlohmann::json({1, 2}), "representative request ids");
        f += check(decay->at("measured_over").at("requests") == 2 &&
                       decay->at("measured_over").at("examined_requests") == 2,
                   "measured_over counts requests with drafts and examined");
    }

    // A healthy curve gives counters but no window advice.
    const std::string healthy =
        done(3, R"({"backend":"mtp","draft_window":4,"rounds":100,"drafted_tokens":400,"accepted_tokens":300,"fallback_steps":0,"accepted_per_position":[95,85,70,50]})");
    const auto healthy_report = analyze_request_log_jsonl(healthy, "mem");
    const auto* fine          = find(healthy_report);
    f += check(fine != nullptr && fine->at("evidence").at("inferred_draft_window").is_null() &&
                   !fine->contains("recommendation"),
               "healthy curve carries no inferred window");

    // Too few draft rounds for advice even when the curve decays.
    const std::string few =
        done(4, R"({"backend":"mtp","draft_window":5,"rounds":10,"drafted_tokens":50,"accepted_tokens":21,"fallback_steps":0,"accepted_per_position":[9,7,4,1,0]})");
    const auto few_report    = analyze_request_log_jsonl(few, "mem");
    const auto* short_window = find(few_report);
    f += check(short_window != nullptr && short_window->at("evidence").at("effective_draft_window") == 3 &&
                   short_window->at("evidence").at("inferred_draft_window").is_null(),
               "under 50 draft rounds measures the curve but does not advise");

    // Fallback pressure: half the decode steps ran without a draft -> warning, measured share.
    const std::string fallback =
        done(5, R"({"backend":"mtp","draft_window":4,"rounds":100,"drafted_tokens":200,"accepted_tokens":150,"fallback_steps":50,"accepted_per_position":[48,42,35,25]})");
    const auto fallback_report = analyze_request_log_jsonl(fallback, "mem");
    const auto* pressure       = find(fallback_report);
    f += check(pressure != nullptr && pressure->at("severity") == "warning", "high fallback share warns");
    if (pressure != nullptr) {
        f += check(std::fabs(pressure->at("evidence").at("fallback_share").get<double>() - 0.5) < 1e-9,
                   "fallback share = fallback / (draft rounds + fallback)");
        f += check(pressure->at("recommendation").get<std::string>().find("Measured") == 0,
                   "fallback statement leads with what was measured");
        const auto& samples = pressure->at("evidence").at("high_fallback_samples");
        f += check(samples.size() == 1 && samples.at(0).at("request_id") == 5 &&
                       samples.at(0).at("fallback_steps") == 50 && samples.at(0).at("draft_rounds") == 50,
                   "fallback-heavy requests are sampled with their request fields");
    }

    // Rising fallback across the window: quiet first half, fallback-heavy second half.
    const std::string rising =
        done(6, R"({"backend":"mtp","draft_window":4,"rounds":100,"drafted_tokens":400,"accepted_tokens":300,"fallback_steps":0,"accepted_per_position":[95,85,70,50]})") +
        done(7, R"({"backend":"mtp","draft_window":4,"rounds":100,"drafted_tokens":400,"accepted_tokens":300,"fallback_steps":0,"accepted_per_position":[95,85,70,50]})") +
        done(8, R"({"backend":"mtp","draft_window":4,"rounds":115,"drafted_tokens":400,"accepted_tokens":300,"fallback_steps":15,"accepted_per_position":[95,85,70,50]})") +
        done(9, R"({"backend":"mtp","draft_window":4,"rounds":115,"drafted_tokens":400,"accepted_tokens":300,"fallback_steps":15,"accepted_per_position":[95,85,70,50]})");
    const auto rising_report = analyze_request_log_jsonl(rising, "mem");
    const auto* rise         = find(rising_report);
    f += check(rise != nullptr && rise->at("severity") == "warning" &&
                   rise->at("evidence").at("fallback_share_first_half") == 0.0 &&
                   std::fabs(rise->at("evidence").at("fallback_share_second_half").get<double>() - 30.0 / 230.0) < 1e-9,
               "fallback rising across the window warns with both halves measured");

    // Missing telemetry: backend none, or no speculative object at all -> unavailable, not 0%.
    const std::string off =
        done(10, R"({"backend":"none","draft_window":0,"rounds":0,"drafted_tokens":0,"accepted_tokens":0,"fallback_steps":0,"accepted_per_position":[]})") +
        R"({"event":"request_done","server_instance_id":"a","timestamp_unix_ms":2011,"request":{"request_id":11,"message_count":1},"result":{"finish_reason":"stop","completion_tokens":3},"timings_seconds":{}})"
        "\n";
    const auto off_report = analyze_request_log_jsonl(off, "mem");
    const auto* none      = find(off_report);
    f += check(none != nullptr && none->at("availability") == "unavailable", "no telemetry is unavailable");
    if (none != nullptr) {
        f += check(none->at("evidence").at("requests_backend_none") == 1 &&
                       none->at("measured_over").at("requests") == 0 &&
                       none->at("measured_over").at("examined_requests") == 2,
                   "unavailable still reports what was examined");
        f += check(!none->at("evidence").contains("acceptance_ratio"), "unavailable carries no ratio");
    }

    // A round near the output limit drafts fewer tokens than the window; rounds still counts it.
    const std::string partial =
        done(13, R"({"backend":"mtp","draft_window":4,"rounds":3,"drafted_tokens":10,"accepted_tokens":6,"fallback_steps":0,"accepted_per_position":[3,2,1,0]})");
    const auto partial_report = analyze_request_log_jsonl(partial, "mem");
    const auto* partial_ins   = find(partial_report);
    f += check(partial_ins != nullptr && partial_ins->at("evidence").at("draft_rounds") == 3 &&
                   std::fabs(partial_ins->at("evidence").at("per_position_rate").at(0).get<double>() - 1.0) < 1e-9,
               "partial last window does not inflate per-position rates");

    // Only the latest server instance is measured: an older launch with window 5 must not blend
    // into the current window-4 curve, and the older requests are reported as out of scope.
    const std::string relaunched =
        std::string(R"({"event":"server_start","server_instance_id":"old","timestamp_unix_ms":10,"engine":{}})") + "\n" +
        R"({"event":"request_done","server_instance_id":"old","timestamp_unix_ms":1001,"request":{"request_id":1,"message_count":1},"result":{"finish_reason":"stop","completion_tokens":64},"timings_seconds":{},"speculative":{"backend":"mtp","draft_window":5,"rounds":100,"drafted_tokens":500,"accepted_tokens":210,"fallback_steps":0,"accepted_per_position":[90,70,40,8,2]}})" "\n" +
        R"({"event":"server_start","server_instance_id":"new","timestamp_unix_ms":5000,"engine":{}})" "\n" +
        R"({"event":"request_done","server_instance_id":"new","timestamp_unix_ms":6001,"request":{"request_id":1,"message_count":1},"result":{"finish_reason":"stop","completion_tokens":64},"timings_seconds":{},"speculative":{"backend":"mtp","draft_window":4,"rounds":100,"drafted_tokens":400,"accepted_tokens":300,"fallback_steps":0,"accepted_per_position":[95,85,70,50]}})" "\n";
    const auto relaunched_report = analyze_request_log_jsonl(relaunched, "mem");
    const auto* scoped           = find(relaunched_report);
    f += check(scoped != nullptr && scoped->at("availability") == "available", "latest instance measured");
    if (scoped != nullptr) {
        const auto& e = scoped->at("evidence");
        f += check(e.at("server_instance_id") == "new" && e.at("draft_window") == 4 &&
                       e.at("draft_windows_seen") == nlohmann::json({{"4", 1}}) &&
                       e.at("drafted_tokens") == 400,
                   "older launch does not blend into the current curve");
        f += check(scoped->at("measured_over").at("requests") == 1 &&
                       scoped->at("measured_over").at("examined_requests") == 1 &&
                       scoped->at("measured_over").at("other_instance_requests") == 1,
                   "out-of-scope requests are counted, not silently dropped");
        f += check(e.at("inferred_draft_window").is_null(), "healthy current curve has no advice");
    }

    // Configured backend that has not drafted: unavailable with the configuration named.
    const std::string idle =
        done(12, R"({"backend":"mtp","draft_window":4,"rounds":0,"drafted_tokens":0,"accepted_tokens":0,"fallback_steps":0,"accepted_per_position":[0,0,0,0]})", 0);
    const auto idle_report = analyze_request_log_jsonl(idle, "mem");
    const auto* no_drafts  = find(idle_report);
    f += check(no_drafts != nullptr && no_drafts->at("availability") == "unavailable" &&
                   no_drafts->at("evidence").at("backend") == "mtp" &&
                   no_drafts->at("evidence").at("drafted_tokens") == 0,
               "zero drafts is unavailable, not a 0% acceptance");
    return f;
}

int test_insights_capacity() {
    using namespace ninfer::supervisor;
    int f = 0;
    auto find = [](const nlohmann::json& report) -> const nlohmann::json* {
        for (const auto& it : report.at("insights")) {
            if (it.at("id") == "capacity.context_kv_pressure") { return &it; }
        }
        return nullptr;
    };
    const std::string start =
        R"({"event":"server_start","server_instance_id":"s","timestamp_unix_ms":10,"engine":{"max_context":8192,"max_concurrency":4,"kv_capacity":16384,"kv_capacity_page_groups":64,"kv_capacity_max_page_groups":128,"prefix_reuse":true}})"
        "\n";
    auto done = [](int id, int prompt, int completion) {
        return std::string(R"({"event":"request_done","server_instance_id":"s","timestamp_unix_ms":)") +
               std::to_string(1000 + id) + R"(,"request":{"request_id":)" + std::to_string(id) +
               R"(,"message_count":1},"result":{"finish_reason":"stop_token","completion_tokens":)" +
               std::to_string(completion) + R"(,"prompt_tokens":)" + std::to_string(prompt) +
               R"(},"timings_seconds":{}})" "\n";
    };
    auto sample = [](int t, int pages, int running, int waiting, int evicted) {
        return std::string(R"({"event":"throughput","server_instance_id":"s","timestamp_unix_ms":)") +
               std::to_string(t) + R"(,"scheduler":{"running":)" + std::to_string(running) +
               R"(,"waiting":)" + std::to_string(waiting) +
               R"(},"context_cache":{"occupancy":{"device_main_kv_pages":)" + std::to_string(pages) +
               R"(},"pressure":{"private_owners_evicted":)" + std::to_string(evicted) +
               R"(,"shared_owners_evicted":0,"spill_pages":0,"search_budget_exhaustions":0,"checkpoints_dropped":0}}})"
               "\n";
    };

    // Comfortable: small prompts, half the pool, one request at a time.
    const auto comfy_report = analyze_request_log_jsonl(
        start + done(1, 900, 100) + done(2, 1500, 200) + sample(2000, 20, 1, 0, 0) + sample(7000, 32, 1, 0, 0),
        "mem");
    const auto* comfy = find(comfy_report);
    f += check(comfy != nullptr && comfy->at("availability") == "available" &&
                   comfy->at("severity") == "info" && comfy->at("evidence").at("kind") == "comfortable",
               "comfortable workload is info");
    if (comfy != nullptr) {
        const auto& e = comfy->at("evidence");
        f += check(e.at("config").at("max_context") == 8192 && e.at("config").at("kv_page_groups") == 64 &&
                       e.at("config").at("tokens_per_page_group") == 256,
                   "configured capacity carried");
        f += check(e.at("requests").at("count") == 2 && e.at("requests").at("prompt_tokens").at("max") == 1500 &&
                       e.at("requests").at("largest_total_tokens") == 1700,
                   "prompt distribution measured");
        f += check(e.at("kv").at("pages_high_water") == 32 && e.at("kv").at("pages_now") == 32 &&
                       std::fabs(e.at("kv").at("utilization_high_water").get<double>() - 0.5) < 1e-9,
                   "KV high-water measured against page-group capacity");
        f += check(!comfy->contains("recommendation"), "comfortable carries no recommendation");
    }

    // Near max context: one request at 7900 of 8192 tokens.
    const auto big_report = analyze_request_log_jsonl(
        start + done(1, 900, 100) + done(2, 7500, 400) + sample(2000, 40, 1, 0, 0), "mem");
    const auto* big = find(big_report);
    f += check(big != nullptr && big->at("severity") == "warning" &&
                   big->at("evidence").at("kind") == "large_prompt",
               "near-max-context request warns as large_prompt");
    if (big != nullptr) {
        const auto& e = big->at("evidence");
        f += check(e.at("requests").at("near_limit_count") == 1 &&
                       e.at("largest_requests").at(0).at("request_id") == 2 &&
                       e.at("largest_requests").at(0).at("total_tokens") == 7900,
                   "largest request is the representative sample");
        f += check(big->at("recommendation").get<std::string>().find("Inferred") != std::string::npos,
                   "capacity advice is marked inferred");
    }

    // Concurrency-driven KV pressure: small prompts, pool at 97% with three of four lanes
    // running, requests waiting and one expiring before admission.
    const std::string expired =
        R"({"event":"request_error","server_instance_id":"s","timestamp_unix_ms":7500,"request":{"request_id":9},"error":{"message":"inference request expired while waiting for admission"}})"
        "\n";
    const auto conc_report = analyze_request_log_jsonl(
        start + done(1, 1200, 300) + done(2, 1100, 300) + done(3, 1300, 300) + done(4, 1000, 300) +
            sample(2000, 30, 2, 0, 0) + sample(7000, 62, 3, 2, 3) + expired + sample(12000, 40, 1, 0, 0),
        "mem");
    const auto* conc = find(conc_report);
    f += check(conc != nullptr && conc->at("severity") == "warning" &&
                   conc->at("evidence").at("kind") == "kv_admission",
               "queueing with free lanes and a full pool warns as kv_admission");
    if (conc != nullptr) {
        const auto& e = conc->at("evidence");
        f += check(e.at("scheduler").at("peak_running") == 3 && e.at("scheduler").at("peak_waiting") == 2 &&
                       e.at("scheduler").at("admission_expired") == 1 &&
                       e.at("scheduler").at("admission_expired_request_ids").at(0) == 9 &&
                       e.at("pressure").at("private_owners_evicted") == 3 && e.at("kv").at("pages_now") == 40,
                   "scheduler peaks, admission expiries, summed pressure deltas and current occupancy measured");
    }

    // Queueing with every lane busy is the concurrency cap, not the pool.
    const auto lanes_report = analyze_request_log_jsonl(
        start + done(1, 1200, 300) + sample(2000, 30, 4, 2, 0), "mem");
    const auto* lanes = find(lanes_report);
    f += check(lanes != nullptr && lanes->at("severity") == "notice" &&
                   lanes->at("evidence").at("kind") == "lanes_full",
               "queueing with all lanes busy is lanes_full");

    // Full pool, evictions, nobody waiting: retained state churned, nothing failed.
    const auto churn_report = analyze_request_log_jsonl(
        start + done(1, 1200, 300) + sample(2000, 62, 2, 0, 5), "mem");
    const auto* churn = find(churn_report);
    f += check(churn != nullptr && churn->at("severity") == "notice" &&
                   churn->at("evidence").at("kind") == "cache_churn" &&
                   churn->at("recommendation").get<std::string>().find("does not") != std::string::npos,
               "evictions without queueing are cache churn, and concurrency advice is withheld");

    // Full pool with no evictions and no queue is retained cache, not pressure.
    const auto cache_report = analyze_request_log_jsonl(
        start + done(1, 1200, 300) + sample(2000, 63, 1, 0, 0), "mem");
    const auto* cache = find(cache_report);
    f += check(cache != nullptr && cache->at("severity") == "info" &&
                   cache->at("evidence").at("kind") == "retained_cache",
               "full pool without pressure events is info");

    // Missing telemetry: no server_start at all, or a start with no occupancy samples yet.
    const auto nostart_report = analyze_request_log_jsonl(done(1, 900, 100) + sample(2000, 20, 1, 0, 0), "mem");
    const auto* nostart = find(nostart_report);
    f += check(nostart != nullptr && nostart->at("availability") == "unavailable" &&
                   nostart->at("evidence").at("requests").at("count") == 1,
               "no capacity configuration is unavailable, with the prompt sizes still reported");
    const auto nosample_report = analyze_request_log_jsonl(start + done(1, 900, 100), "mem");
    const auto* nosample = find(nosample_report);
    f += check(nosample != nullptr && nosample->at("availability") == "unavailable" &&
                   !nosample->at("evidence").contains("kv"),
               "no occupancy sample is unavailable, not zero utilization");

    // An earlier instance's traffic and limits do not leak into the latest one.
    const std::string old_instance =
        R"({"event":"server_start","server_instance_id":"old","timestamp_unix_ms":1,"engine":{"max_context":2048,"max_concurrency":1,"kv_capacity":2048,"kv_capacity_page_groups":8,"prefix_reuse":false}})"
        "\n"
        R"({"event":"request_done","server_instance_id":"old","timestamp_unix_ms":5,"request":{"request_id":7},"result":{"finish_reason":"stop_token","completion_tokens":100,"prompt_tokens":1900},"timings_seconds":{}})"
        "\n"
        R"({"event":"throughput","server_instance_id":"old","timestamp_unix_ms":6,"scheduler":{"running":1,"waiting":0},"context_cache":{"occupancy":{"device_main_kv_pages":8},"pressure":{"private_owners_evicted":9}}})"
        "\n";
    const auto scoped_report = analyze_request_log_jsonl(
        old_instance + start + done(1, 900, 100) + sample(2000, 20, 1, 0, 0), "mem");
    const auto* scoped = find(scoped_report);
    f += check(scoped != nullptr && scoped->at("evidence").at("kind") == "comfortable" &&
                   scoped->at("evidence").at("requests").at("count") == 1 &&
                   scoped->at("evidence").at("pressure").at("private_owners_evicted") == 0,
               "older instance does not leak into the latest verdict");
    return f;
}

int test_insights_prefix_collapse() {
    using namespace ninfer::supervisor;
    std::string jsonl =
        R"({"event":"server_start","server_instance_id":"leaking","timestamp_unix_ms":1,"engine":{"context_cache":{"host_state_slots":2}}})"
        "\n";
    for (int sample = 0; sample < 3; ++sample) {
        jsonl +=
            "{\"event\":\"throughput\",\"server_instance_id\":\"leaking\","
            "\"timestamp_unix_ms\":" + std::to_string(1000 + sample * 35000) +
            ",\"context_cache\":{\"occupancy\":{\"host_state_slots\":2}}}\n";
    }
    for (int request = 0; request < 3; ++request) {
        jsonl +=
            "{\"event\":\"request_done\",\"server_instance_id\":\"leaking\","
            "\"timestamp_unix_ms\":" + std::to_string(72000 + request) +
            ",\"request\":{\"request_id\":" + std::to_string(request + 1) +
            ",\"message_count\":4},\"result\":{\"prefix_reuse_path\":\"root\","
            "\"prefix_cache_hit_tokens\":0,\"computed_prefill_tokens\":1000},"
            "\"timings_seconds\":{}}\n";
    }

    const auto report = analyze_request_log_jsonl(jsonl, "mem");
    int f = 0;
    const nlohmann::json* collapse = nullptr;
    for (const auto& insight : report.at("insights")) {
        if (insight.at("id") == "prefix.reuse_collapsed") { collapse = &insight; }
    }
    f += check(collapse != nullptr, "saturated Host State plus Root streak is detected");
    if (collapse != nullptr) {
        f += check(collapse->at("severity") == "warning", "collapse is a warning");
        f += check(collapse->at("evidence").at("host_state_slots") == 2,
                   "collapse carries measured Host State occupancy");
        f += check(collapse->at("evidence").at("consecutive_multiturn_root_misses") == 3,
                   "collapse carries the Root miss streak");
        f += check(collapse->at("evidence").at("recomputed_prefill_tokens") == 3000,
                   "collapse carries avoidable prefill work");
    }

    const std::string recovered =
        jsonl +
        R"({"event":"request_done","server_instance_id":"leaking","timestamp_unix_ms":73000,"request":{"request_id":4,"message_count":4},"result":{"prefix_reuse_path":"private_turn_closure","prefix_cache_hit_tokens":995,"computed_prefill_tokens":5},"timings_seconds":{}})"
        "\n";
    const auto recovered_report = analyze_request_log_jsonl(recovered, "mem");
    bool false_alarm = false;
    for (const auto& insight : recovered_report.at("insights")) {
        false_alarm = false_alarm || insight.at("id") == "prefix.reuse_collapsed";
    }
    f += check(!false_alarm, "a later cache hit clears the collapse alarm");

    const std::string restarted =
        jsonl +
        R"({"event":"server_start","server_instance_id":"fresh","timestamp_unix_ms":80000,"engine":{"context_cache":{"host_state_slots":2}}})"
        "\n"
        R"({"event":"throughput","server_instance_id":"fresh","timestamp_unix_ms":81000,"context_cache":{"occupancy":{"host_state_slots":0}}})"
        "\n";
    const auto restarted_report = analyze_request_log_jsonl(restarted, "mem");
    false_alarm = false;
    for (const auto& insight : restarted_report.at("insights")) {
        false_alarm = false_alarm || insight.at("id") == "prefix.reuse_collapsed";
    }
    f += check(!false_alarm, "a restarted server does not inherit the previous instance alarm");
    return f;
}

int test_jsonl_event_key() {
    using namespace ninfer::supervisor;
    int f = 0;
    f += check(jsonl_event_is(R"({"event":"request_done","result":{}})", "request_done"),
               "compact event key");
    f += check(jsonl_event_is(R"({"event": "request_done"})", "request_done"), "spaced event key");
    f += check(!jsonl_event_is(R"({"type":"request_done"})", "request_done"),
               "type key is not the event key");
    f += check(!jsonl_event_is(R"({"event":"server_start"})", "request_done"), "other event");
    return f;
}

int test_series_persist() {
    using namespace ninfer::supervisor;
    int f = 0;
    VramSample s{};
    f += check(parse_series_sample_line(R"({"t_ms":10,"budget_bytes":20,"nvidia_used_bytes":30})", s) &&
                   s.t_ms == 10 && s.budget_bytes == 20 && s.nvidia_used_bytes == 30,
               "parse sample");
    VramSeriesEvent e{};
    f += check(parse_series_event_line(R"({"t_ms":11,"kind":"vram_release","label":"seed store released"})", e) &&
                   e.kind == "vram_release",
               "parse event");
    VramSeriesRing r(8);
    r.load_jsonl("{\"t_ms\":1,\"budget_bytes\":2,\"nvidia_used_bytes\":3}\n"
                 "{\"t_ms\":4,\"kind\":\"engine_up\",\"label\":\"health 200\"}\n");
    f += check(r.size() == 1 && r.events().size() == 1 && r.events()[0].kind == "engine_up",
               "load_jsonl restores samples and events");
    return f;
}

int test_series_ring() {
    ninfer::supervisor::VramSeriesRing r(3);
    int f = 0;
    r.push({1, 10, 4});
    r.push({2, 20, 5});
    r.push({3, 30, 6});
    r.push({4, 40, 7});
    const auto s = r.samples();
    f += check(s.size() == 3 && s[0].t_ms == 2 && s[2].t_ms == 4 && s[2].budget_bytes == 40,
               "ring drops oldest, keeps raw values");
    r.push_event({4, "admin_vram", "release"}, 2);
    r.push_event({5, "engine_down", "health 0"}, 2);
    r.push_event({6, "engine_up", "health 200"}, 2);
    const auto e = r.events();
    f += check(e.size() == 2 && e[0].kind == "engine_down" && e[1].kind == "engine_up",
               "event ring cap");
    return f;
}

int test_health_threshold() {
    ninfer::supervisor::RestartPolicy p;
    p.health_fail_threshold = 3;
    ninfer::supervisor::RestartGate g(p);
    int f = 0;
    f += check(!g.note_health_fail() && !g.note_health_fail(), "below threshold");
    f += check(g.note_health_fail(), "threshold trips restart");
    g.note_healthy();
    f += check(!g.note_health_fail(), "healthy clears fail count");
    return f;
}


int test_with_desktop_reserve() {
    using namespace ninfer::supervisor;
    int f = 0;
    const std::vector<std::string> base{"model.ninfer", "--port", "8190",
                                        "--desktop-reserve-gib", "3"};
    const auto rewritten = with_desktop_reserve(base, 6);
    f += check(rewritten.size() == 5 && rewritten[3] == "--desktop-reserve-gib" &&
                   rewritten[4] == "6",
               "existing reserve is rewritten in place");

    const std::vector<std::string> none{"model.ninfer", "--port", "8190"};
    const auto appended = with_desktop_reserve(none, 4);
    f += check(appended.size() == 5 && appended[3] == "--desktop-reserve-gib" &&
                   appended[4] == "4",
               "missing reserve is appended");

    const std::vector<std::string> mib{"model.ninfer", "--desktop-reserve-mib", "1700"};
    f += check(desktop_reserve_launch_mib(mib, 8) == 1700, "MiB pin determines next launch despite standing preference");
    f += check(desktop_reserve_launch_mib({"--desktop-reserve-gib", "0"}, kReserveUnset) == 0, "explicit zero remains zero");
    f += check(desktop_reserve_launch_mib(base, 0) == 8192, "engine default preference removes configured reserve");
    f += check(with_desktop_reserve(mib, 8) == mib, "--desktop-reserve-mib is left alone");

    const std::vector<std::string> trailing{"model.ninfer", "--desktop-reserve-gib"};
    const auto fixed = with_desktop_reserve(trailing, 2);
    f += check(fixed.size() == 3 && fixed[1] == "--desktop-reserve-gib" && fixed[2] == "2",
               "trailing flag with no value gets one, not a second pair");

    const auto removed = with_desktop_reserve(base, 0);
    f += check(removed.size() == 3 && removed[0] == "model.ninfer" && removed[2] == "8190",
               "gib 0 removes the pair so the engine default applies");
    f += check(with_desktop_reserve(none, 0) == none, "gib 0 with no flag changes nothing");
    f += check(with_desktop_reserve(trailing, 0).size() == 1,
               "gib 0 removes a valueless trailing flag");

    f += check(desktop_reserve_from_args(base) == 3 && desktop_reserve_from_args(none) == 0,
               "reserve read back from args");

    // The menu asks the same question before it decides whether to offer the
    // submenu at all: a live submenu over a pinned MiB reserve lets the user
    // pick a number, writes it to prefs, and offers a model reload that changes
    // nothing.
    f += check(desktop_reserve_pinned(mib), "--desktop-reserve-mib is a pin");
    f += check(!desktop_reserve_pinned(base) && !desktop_reserve_pinned(none),
               "a gib reserve, or none, is not a pin");
    return f;
}

int test_tray_command_ids() {
    using namespace ninfer::supervisor;
    int f = 0;
    for (std::size_t i = 0; i < kReserveChoicesGib.size(); ++i) {
        const auto cmd = decode_tray_command(tray_cmd_reserve(i));
        f += check(cmd.action == TrayAction::Reserve && cmd.value == kReserveChoicesGib[i],
                   "reserve index " + std::to_string(i) + " round-trips");
    }
    for (std::size_t i = 0; i < kIdleChoicesMinutes.size(); ++i) {
        const auto cmd = decode_tray_command(tray_cmd_idle(i));
        f += check(cmd.action == TrayAction::Idle && cmd.value == kIdleChoicesMinutes[i],
                   "idle index " + std::to_string(i) + " round-trips");
    }
    // The bug this replaces: "After 4 hours" encoded as 200+240=440, past the
    // next base, and was silently discarded by the dispatch guard.
    const auto four_hours = decode_tray_command(tray_cmd_idle(3));
    f += check(four_hours.action == TrayAction::Idle && four_hours.value == 240,
               "After 4 hours decodes to 240 minutes");
    f += check(decode_tray_command(0).action == TrayAction::None, "0 is not a command");
    f += check(decode_tray_command(kCmdReserveBase + 99).action == TrayAction::None,
               "out-of-range reserve index is not a command");
    f += check(decode_tray_command(kCmdIdleBase + 99).action == TrayAction::None,
               "out-of-range idle index is not a command");
    f += check(decode_tray_command(kCmdQuit).action == TrayAction::Quit, "fixed id decodes");
    f += check(decode_tray_command(kCmdCopyUrl).action == TrayAction::CopyUrl,
               "copy-url id decodes");
    return f;
}

int test_classify_engine_line() {
    using namespace ninfer::supervisor;
    int f = 0;
    // Verbatim from supervisor-logs-demo/engine.log, the crash loop of 2026-09-04.
    const auto logged = classify_engine_line(
        "[2026-09-04 23:52:12.090] [error] ninfer-serve: unknown argument: --gdn-state-dtype");
    f += check(logged.error && !logged.activity && !logged.ready, "logged arg error is an error");
    f += check(logged.message == "unknown argument: --gdn-state-dtype",
               "logged arg error carries the reason");

    // Current HEAD fails argument parsing before the logger exists.
    const auto raw = classify_engine_line("ninfer-serve: unknown argument: --x");
    f += check(raw.error && raw.message == "unknown argument: --x", "bare cerr form is an error");

    const auto tput = classify_engine_line(
        "[2026-09-04 23:52:12.090] [info] [ninfer-serve] throughput interval_ms=5000.000 tok_s=42");
    f += check(tput.activity && !tput.error, "throughput line is activity");

    const auto req = classify_engine_line(
        "[2026-09-04 23:52:12.090] [info] [ninfer-serve] request id=7 status=submitted");
    f += check(req.activity && !req.error, "request line is activity");

    const auto ready = classify_engine_line(
        "[2026-09-04 23:52:12.090] [info] [ninfer-serve] server status=ready host=\"127.0.0.1\" "
        "port=8190");
    f += check(ready.ready && !ready.error, "ready line is readiness");

    const auto fatal = classify_engine_line(
        "[2026-09-04 23:52:12.090] [critical] [ninfer-serve] server status=failed phase=serving "
        "detail=out of memory");
    f += check(fatal.fatal && fatal.error, "failed line is fatal and an error");

    const auto noise = classify_engine_line("usage: ninfer-serve MODEL [options]");
    f += check(!noise.error && !noise.activity && !noise.ready && !noise.fatal,
               "usage text signals nothing");
    f += check(!classify_engine_line("").error, "empty line signals nothing");
    return f;
}

// A per-request failure is not the engine's halt reason. Before this split, one
// internal 500 became the tray's stored "why" until the next `server
// status=ready`, and then showed up as the "Last error:" clause of an unrelated
// crash-loop balloon.
int test_process_scoped_error() {
    using namespace ninfer::supervisor;
    int f = 0;
    const auto req_fail = classify_engine_line(
        "[2026-09-04 23:52:12.090] [error] [ninfer-serve] request id=812 status=failed "
        "protocol=\"openai\" error_type=\"internal\"");
    f += check(req_fail.error && req_fail.activity, "a failed request is an error and activity");
    f += check(!is_process_scoped_error(req_fail),
               "a failed request is not the engine's halt reason");

    const auto arg_error = classify_engine_line(
        "[2026-09-04 23:52:12.090] [error] ninfer-serve: unknown argument: --gdn-state-dtype");
    f += check(is_process_scoped_error(arg_error), "an argument error is process-scoped");

    const auto fatal = classify_engine_line(
        "[2026-09-04 23:52:12.090] [critical] [ninfer-serve] server status=failed phase=serving "
        "detail=request id=9 ran out of memory");
    f += check(is_process_scoped_error(fatal),
               "a fatal line stays process-scoped even when it names a request");

    const auto ok = classify_engine_line(
        "[2026-09-04 23:52:12.090] [info] [ninfer-serve] request id=7 status=done");
    f += check(!is_process_scoped_error(ok), "a successful request is not an error at all");
    return f;
}

// The in-flight window: opened by status=submitted, closed by every terminal
// status. This is what makes "cannot unload mid-request" independent of the
// engine's log cadence.
int test_classify_request_line() {
    using namespace ninfer::supervisor;
    int f = 0;
    const auto open = classify_request_line(
        "[2026-09-04 23:52:12.090] [info] [ninfer-serve] request id=812 status=submitted "
        "protocol=\"openai\" stream=true messages=3");
    f += check(open.matched && open.id == 812 && open.opens && !open.closes,
               "submitted opens request 812");

    const auto done = classify_request_line(
        "[2026-09-04 23:52:12.090] [info] [ninfer-serve] request id=812 status=done "
        "finish_reason=stop prompt_tokens=10");
    f += check(done.matched && done.id == 812 && done.closes && !done.opens, "done closes it");

    for (const char* status : {"failed", "rejected", "cancelled"}) {
        const auto terminal =
            classify_request_line(std::string("request id=5 status=") + status + " protocol=\"x\"");
        f += check(terminal.matched && terminal.closes,
                   std::string("status=") + status + " closes the request");
    }

    // A second record about a request that has its own terminal line; counting
    // it would close an id that is still running, or open one that never closes.
    const auto response = classify_request_line(
        "[2026-09-04 23:52:12.090] [error] [ninfer-serve] response request_id=812 status=failed");
    f += check(!response.matched, "a response record is not a request lifecycle line");

    const auto throughput = classify_request_line(
        "[2026-09-04 23:52:12.090] [info] [ninfer-serve] throughput interval_ms=5000.000 running=1");
    f += check(!throughput.matched, "the throughput line is not a request lifecycle line");

    const auto no_status = classify_request_line("request id=42");
    f += check(no_status.matched && no_status.id == 42 && !no_status.opens && !no_status.closes,
               "a request line with no status neither opens nor closes");
    f += check(!classify_request_line("request id=x status=submitted").matched,
               "a non-numeric id is not a request");
    f += check(!classify_request_line("").matched, "an empty line is not a request");
    return f;
}

int test_idle_unload_due() {
    using namespace ninfer::supervisor;
    int f = 0;
    const std::int64_t now  = 1000000000;
    const std::int64_t hour = 60 * 60 * 1000;
    IdleInputs base{now, now - hour, 60, true, true};
    f += check(idle_unload_due(base), "exactly at the window is due");

    IdleInputs never   = base;
    never.idle_minutes = 0;
    f += check(!idle_unload_due(never), "0 minutes never unloads");

    IdleInputs stopped = base;
    stopped.running    = false;
    f += check(!idle_unload_due(stopped), "a stopped engine is not unloaded");

    IdleInputs sick = base;
    sick.healthy    = false;
    f += check(!idle_unload_due(sick), "an engine not answering /health is not unloaded");

    IdleInputs inside       = base;
    inside.last_activity_ms = now - hour + 1;
    f += check(!idle_unload_due(inside), "one millisecond inside the window is not due");

    IdleInputs past       = base;
    past.last_activity_ms = now - 5 * hour;
    f += check(idle_unload_due(past), "long past the window is due");

    IdleInputs unknown       = base;
    unknown.last_activity_ms = 0;
    f += check(!idle_unload_due(unknown), "an unknown activity clock never unloads");

    // The hard floor. With --log-stats-interval-ms 0 and no request log a long
    // generation emits nothing between submit and done, so the activity clock
    // ages past the window while the engine is mid-answer.
    IdleInputs busy        = past;
    busy.inflight_requests = 1;
    f += check(!idle_unload_due(busy),
               "an open request blocks the unload however old the activity clock is");
    IdleInputs drained        = busy;
    drained.inflight_requests = 0;
    f += check(idle_unload_due(drained), "the unload proceeds once the request closes");
    return f;
}

int test_format_halt_notice() {
    using namespace ninfer::supervisor;
    int f = 0;
    const auto n = format_halt_notice(5, 60, 1, "unknown argument: --gdn-state-dtype");
    f += check(n.find("5 times") != std::string::npos && n.find("60 s") != std::string::npos,
               "notice carries the count and window");
    f += check(n.find("exit code 1") != std::string::npos, "notice carries the exit code");
    f += check(n.find("unknown argument: --gdn-state-dtype") != std::string::npos,
               "notice carries the error");
    f += check(n.size() <= 255, "notice fits szInfo");
    const auto long_err = format_halt_notice(5, 60, 1, std::string(4096, 'x'));
    f += check(long_err.size() <= 255, "a long error is trimmed, not the counts");
    f += check(long_err.find("5 times") != std::string::npos, "counts survive trimming");
    const auto no_err = format_halt_notice(3, 30, 0, "");
    f += check(no_err.find("Last error") == std::string::npos, "no error text, no error clause");
    return f;
}

int test_model_label() {
    using namespace ninfer::supervisor;
    int f = 0;
    const std::vector<std::string> demo{
        "C:/models/Qwen3.8-Flash-Next/qwen3_8_flash_next_mixed.ninfer", "--host", "127.0.0.1",
        "--port", "8190"};
    f += check(model_label_from_args(demo) == "qwen3_8_flash_next_mixed",
               "demo args yield the checkpoint name");
    const std::vector<std::string> flagged{"--model", "P:\\models\\a.b\\thing.ninfer", "--port",
                                           "8190"};
    f += check(model_label_from_args(flagged) == "thing",
               "checkpoint found behind a flag, backslashes and all");
    f += check(model_label_from_args({}).empty(), "no args, no label");
    f += check(model_label_from_args({"--port", "8190"}).empty(), "flags only, no label");
    return f;
}

int test_single_instance_name() {
    using namespace ninfer::supervisor;
    int f = 0;
    const auto a = single_instance_name("C:/NInfer/supervisor.json");
    const auto b = single_instance_name("c:\\ninfer\\SUPERVISOR.json");
    f += check(a == b, "case and separator differences are the same instance");
    f += check(a != single_instance_name("C:/NInfer/other.json"), "different configs differ");
    f += check(a.rfind("Local\\", 0) == 0, "namespaced Local");
    f += check(a.find('\\', 6) == std::string::npos, "no backslash after the namespace prefix");
    f += check(a.size() < 200, "well under MAX_PATH");
    return f;
}

int test_run_at_login_command() {
    using namespace ninfer::supervisor;
    int f = 0;
    const auto cmd = run_at_login_command("C:/Program Files/NInfer/ninfer-supervisor.exe",
                                          "C:/NInfer/supervisor.json");
    f += check(cmd == "\"C:/Program Files/NInfer/ninfer-supervisor.exe\" --config "
                      "\"C:/NInfer/supervisor.json\"",
               "both paths quoted, spaces survive");

    // There is one Run value name for every config, so "Start at login" is
    // answered by comparing the stored command, not by testing that some entry
    // exists -- otherwise the checkmark appears on the menu of a config the
    // entry does not name, and unchecking deletes the other config's autostart.
    const auto other = run_at_login_command("C:/Program Files/NInfer/ninfer-supervisor.exe",
                                            "C:/NInfer/other.json");
    f += check(run_at_login_command_matches(cmd, cmd), "the same command matches");
    f += check(!run_at_login_command_matches(other, cmd), "another config's entry does not match");
    f += check(run_at_login_command_matches(" " + cmd + " ", cmd),
               "a hand-edited entry with stray whitespace still matches");
    f += check(!run_at_login_command_matches("", cmd), "no entry does not match");
    f += check(!run_at_login_command_matches("", ""), "an empty expectation never matches");
    // The installer writes the entry with native backslashes; the supervisor
    // spells the same config path with forward slashes. Windows paths are also
    // case-insensitive. Both must read as the same entry, or the tray shows the
    // installer's entry as unchecked and a click replaces or removes it.
    f += check(run_at_login_command_matches(
                   "\"C:\\Program Files\\NInfer\\ninfer-supervisor.exe\" --config "
                   "\"c:\\ninfer\\Supervisor.json\"",
                   cmd),
               "an installer-written native path matches the supervisor's canonical form");
    return f;
}

int test_tray_prefs() {
    using namespace ninfer::supervisor;
    int f = 0;
    const auto valid =
        parse_tray_prefs_json(R"({"desktop_reserve_gib":4,"idle_unload_minutes":60})");
    f += check(valid.desktop_reserve_gib == 4 && valid.idle_unload_minutes == 60, "valid prefs");

    const auto corrupt = parse_tray_prefs_json("{not json");
    f += check(corrupt.desktop_reserve_gib == kReserveUnset && corrupt.idle_unload_minutes == 0,
               "corrupt JSON falls back to defaults");

    const auto wrong_type =
        parse_tray_prefs_json(R"({"desktop_reserve_gib":"4","idle_unload_minutes":60})");
    f += check(wrong_type.desktop_reserve_gib == kReserveUnset &&
                   wrong_type.idle_unload_minutes == 60,
               "a wrong type defaults that field only");

    // The engine rejects a negative reserve in parse_nonnegative_int, so a
    // hand-edited -5 would have turned into a crash loop.
    const auto negative = parse_tray_prefs_json(R"({"desktop_reserve_gib":-5})");
    f += check(negative.desktop_reserve_gib == kReserveUnset,
               "negative reserve is normalised away");
    const auto odd =
        parse_tray_prefs_json(R"({"desktop_reserve_gib":7,"idle_unload_minutes":37})");
    f += check(odd.desktop_reserve_gib == 7 && odd.idle_unload_minutes == 0,
               "custom reserve is retained while invalid idle value is normalised");
    f += check(parse_tray_prefs_json(R"({"desktop_reserve_gib":65})").desktop_reserve_gib == kReserveUnset,
               "reserve above supported range is rejected");
    f += check(parse_tray_prefs_json(R"({"desktop_reserve_gib":1})").desktop_reserve_gib == 1,
               "one GiB custom reserve is retained");

    TrayPrefs p;
    p.desktop_reserve_gib = 0;
    p.idle_unload_minutes = 240;
    const auto round      = parse_tray_prefs_json(tray_prefs_to_json(p));
    f += check(round.desktop_reserve_gib == 0 && round.idle_unload_minutes == 240,
               "round-trip preserves an explicit engine-default choice");

    f += check(tray_prefs_path("x.json") == "x.tray.json", "simple extension");
    f += check(tray_prefs_path("C:/a.b/cfg") == "C:/a.b/cfg.tray.json",
               "a dot in a directory is not an extension");
    f += check(tray_prefs_path("C:\\d\\cfg.json") == "C:\\d\\cfg.tray.json",
               "backslash paths keep their directory");
    return f;
}

} // namespace


int test_engine_param_round_trip() {
    using namespace ninfer::supervisor;
    int f = 0;
    const std::vector<std::string> args{
        "model.ninfer", "--host", "127.0.0.1", "--port", "8190", "--max-context", "262144",
        "--kv-dtype", "fp8", "--vision", "--some-future-flag", "7"};
    const ParsedEngineArgs parsed = parse_engine_args(args);
    f += check(parsed.artifact == "model.ninfer", "first positional is the artifact");
    const std::string* ctx = find_param_value(parsed.params, "max_context");
    f += check(ctx != nullptr && *ctx == "262144", "int parameter parses");
    const std::string* vis = find_param_value(parsed.params, "vision");
    f += check(vis != nullptr && *vis == "true", "presence flag parses");
    // An unknown flag and its value must survive together, or rendering would turn
    // the value into a positional argument.
    f += check(parsed.passthrough.size() == 6 && parsed.passthrough[4] == "--some-future-flag" &&
                   parsed.passthrough[5] == "7",
               "unknown flags keep their values");

    const auto rendered = render_engine_args(parsed);
    std::vector<std::string> a = args;
    std::vector<std::string> b = rendered;
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    f += check(a == b, "round trip loses nothing");
    return f;
}

int test_engine_param_validation() {
    using namespace ninfer::supervisor;
    int f = 0;
    f += check(validate_engine_param_combination({{"spec", "dflash2"}, {"draft_tokens", "15"},
                {"lm_head_draft", "true"}}, "qwen3.8-27b").empty(), "DFlash2 permits 15 drafts");
    f += check(validate_engine_param_combination({{"spec", "dflash2"}, {"draft_tokens", "7"},
                {"lm_head_draft", "true"}}, "qwen3.8-27b-orcarouter").empty(),
                "OrcaRouter permits DFlash2 with its separate artifact identity");
    f += check(!validate_engine_param_combination({{"spec", "mtp"}, {"draft_tokens", "6"}},
                "qwen3.8-27b-orcarouter").empty(), "OrcaRouter MTP rejects six drafts");
    f += check(!validate_engine_param_combination({{"spec", "mtp"}, {"draft_tokens", "6"}},
                "qwen3.8-27b").empty(), "MTP rejects six drafts");
    f += check(!validate_engine_param_combination({{"spec", "mtp"}, {"draft_tokens", "5"}},
                "qwen3.8-flash-next").empty(), "Flash MTP rejects five drafts");
    f += check(!validate_engine_param_combination({{"spec", "dflash2"}},
                "qwen3.8-flash-next").empty(), "Flash rejects DFlash2");
    f += check(!validate_engine_param_combination({{"lm_head_draft", "true"}}).empty(),
               "draft head requires speculation");
    const auto head = parse_engine_args({"model.ninfer", "--spec", "dflash2", "--lm-head-draft"});
    f += check(head.passthrough.empty() && find_param_value(head.params, "lm_head_draft"),
               "draft head is editable instead of passthrough");
    f += check(validate_engine_params({{"max_concurrency", "8"}}).empty(), "in-range value passes");
    f += check(!validate_engine_params({{"max_concurrency", "16"}}).empty(),
               "the engine's own concurrency limit is enforced");
    f += check(!validate_engine_params({{"max_context", "not-a-number"}}).empty(),
               "non-numeric int is rejected");
    f += check(validate_engine_params({{"kv_capacity", "auto"}}).empty(), "auto is accepted");
    f += check(!validate_engine_params({{"kv_dtype", "fp4"}}).empty(),
               "unsupported enum value is rejected");
    f += check(!validate_engine_params({{"rm_rf", "1"}}).empty(),
               "an unknown key is not silently written through");
    f += check(!validate_engine_params({{"model_id", std::string(1, '\n')}}).empty(),
               "a newline cannot be smuggled into the config file");
    f += check(!validate_engine_param_combination({{"kv_capacity", "4096"},
                                                   {"max_context", "262144"}})
                    .empty(),
               "a KV pool smaller than one full context is refused");
    f += check(!validate_engine_param_combination({{"kv_capacity", "4194304"},
                                                   {"max_context", "32768"},
                                                   {"max_concurrency", "8"}})
                    .empty(),
               "a KV pool larger than concurrency times max context is refused");
    f += check(validate_engine_param_combination({{"kv_capacity", "262144"},
                                                  {"max_context", "32768"},
                                                  {"max_concurrency", "8"}})
                   .empty(),
               "eight 32K lanes may share a 256K pool");
    return f;
}

int test_redact_engine_args() {
    using namespace ninfer::supervisor;
    int f = 0;
    const auto out = redact_engine_args({"model.ninfer", "--api-key", "sk-live-secret", "--port",
                                         "8190"});
    f += check(out[2] == "(redacted)", "an inline api key never leaves the process");
    f += check(out[4] == "8190", "ordinary values are untouched");
    return f;
}

int test_engine_endpoint_sync() {
    using namespace ninfer::supervisor;
    int f = 0;
    const auto rewritten =
        with_engine_endpoint({"model.ninfer", "--host", "127.0.0.1", "--port", "8190"},
                             "127.0.0.1", 8191);
    f += check(rewritten[4] == "8191", "listen port follows the polled port");
    const auto appended = with_engine_endpoint({"model.ninfer"}, "127.0.0.1", 8191);
    f += check(appended.size() == 5, "endpoint flags are added when absent");
    return f;
}

int test_with_request_log() {
    using namespace ninfer::supervisor;
    int f = 0;
    const auto added = with_request_log({"model.ninfer"}, "C:/logs/r.jsonl");
    f += check(added.size() == 3 && added[1] == "--request-log-jsonl", "config path is passed on");
    const auto kept =
        with_request_log({"model.ninfer", "--request-log-jsonl", "explicit.jsonl"}, "other.jsonl");
    f += check(kept[2] == "explicit.jsonl", "an explicit flag wins over the config field");
    f += check(with_request_log({"model.ninfer"}, "").size() == 1, "empty path adds nothing");
    return f;
}

int test_with_api_key_file() {
    using namespace ninfer::supervisor;
    int f = 0;
    const auto added = with_api_key_file({"model.ninfer"}, "P:/models/key.txt");
    f += check(added.size() == 3 && added[1] == "--api-key-file" && added[2] == "P:/models/key.txt",
               "the configured key file reaches the engine");
    // The value must be the path, never the key: a child's command line is readable
    // by any process running as this user.
    const auto kept_inline =
        with_api_key_file({"model.ninfer", "--api-key", "hand-written"}, "P:/models/key.txt");
    f += check(kept_inline.size() == 3 && kept_inline[2] == "hand-written",
               "a hand-written --api-key wins over the config field");
    const auto kept_file =
        with_api_key_file({"model.ninfer", "--api-key-file", "explicit.txt"}, "P:/models/key.txt");
    f += check(kept_file.size() == 3 && kept_file[2] == "explicit.txt",
               "an explicit --api-key-file wins over the config field");
    f += check(with_api_key_file({"model.ninfer"}, "").size() == 1,
               "no configured key file leaves the engine unauthenticated, as before");
    return f;
}

int test_kv_capacity_adaptation() {
    using namespace ninfer::supervisor;
    int f = 0;
    const std::string line =
        "[2026-09-07 18:23:15.282] [critical] [ninfer-serve] server status=failed phase=startup "
        "detail=\"requested Engine runtime reservation requires 17572624288 bytes, but only "
        "16943534080 bytes are available for runtime capacity\"";
    const auto sf = parse_runtime_reservation_failure(line);
    f += check(sf.matched && sf.required_bytes == 17572624288LL && sf.available_bytes == 16943534080LL,
               "runtime reservation shortfall line was not parsed");
    f += check(!parse_runtime_reservation_failure("[error] request id=7 status=failed").matched,
               "unrelated error line parsed as a reservation shortfall");

    const std::vector<std::string> args{"model.ninfer", "--max-context", "262144",
                                        "--kv-capacity", "737280", "--spec", "mtp"};
    f += check(explicit_kv_capacity(args) == std::optional<std::int64_t>(737280),
               "explicit kv capacity was not read from args");
    f += check(!explicit_kv_capacity({"--kv-capacity", "auto"}).has_value(),
               "auto kv capacity must not be adapted");
    f += check(kv_capacity_floor_tokens(args) == 262144, "kv floor did not follow --max-context");

    // The production case: 629 MB short at 737280 tokens. Must land below the value that
    // fit (655360), stay a multiple of 4096, and never below the floor.
    const auto next = reduced_kv_capacity(737280, sf.required_bytes, sf.available_bytes, 262144);
    f += check(next < 655360 && next % 4096 == 0 && next >= 262144,
               "reduced kv capacity is not a safe multiple below the plan that fit");
    f += check(reduced_kv_capacity(737280, 100, 200, 262144) == 737280,
               "a plan that fits must not be reduced");
    f += check(reduced_kv_capacity(300000, 40LL << 30, 1, 262144) == 262144,
               "reduction must clamp at the floor");

    const auto rewritten = with_kv_capacity(args, next);
    f += check(explicit_kv_capacity(rewritten) == std::optional<std::int64_t>(next) &&
                   rewritten.size() == args.size(),
               "with_kv_capacity did not replace the value in place");
    const auto appended = with_kv_capacity({"model.ninfer"}, 4096);
    f += check(appended.size() == 3 && appended[1] == "--kv-capacity" && appended[2] == "4096",
               "with_kv_capacity did not append when absent");
    return f;
}

int main() {
    int failures = 0;
    failures += test_kv_capacity_adaptation();
    failures += test_loopback();
    failures += test_engine_probe_destination();
    failures += test_crash_loop();
    failures += test_backoff();
    failures += test_config_bind();
    failures += test_host_header();
    failures += test_nvidia_csv();
    failures += test_kv_line();
    failures += test_monitor_only_config();
    failures += test_insights_honesty();
    failures += test_insights_prefix();
    failures += test_insights_prefix_collapse();
    failures += test_insights_speculative();
    failures += test_insights_capacity();
    failures += test_admin_vram_markers();
    failures += test_insights_pinned_tier();
    failures += test_jsonl_event_key();
    failures += test_series_persist();
    failures += test_series_ring();
    failures += test_health_threshold();
    failures += test_with_desktop_reserve();
    failures += test_model_reserve_budget();
    failures += test_engine_param_round_trip();
    failures += test_engine_param_validation();
    failures += test_redact_engine_args();
    failures += test_engine_endpoint_sync();
    failures += test_with_request_log();
    failures += test_with_api_key_file();
    failures += test_tray_command_ids();
    failures += test_classify_engine_line();
    failures += test_process_scoped_error();
    failures += test_classify_request_line();
    failures += test_idle_unload_due();
    failures += test_format_halt_notice();
    failures += test_model_label();
    failures += test_single_instance_name();
    failures += test_run_at_login_command();
    failures += test_tray_prefs();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
