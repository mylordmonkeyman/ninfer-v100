#pragma once
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace ninfer::supervisor {
inline std::vector<std::string> without_reserve_args(const std::vector<std::string>& args) {
    std::vector<std::string> out;
    for (std::size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "--desktop-reserve-gib" || args[i] == "--desktop-reserve-mib") { ++i; continue; }
        out.push_back(args[i]);
    }
    return out;
}

// Preserve the loaded model and its capacity policy. A minimum runtime alone
// would silently shrink an explicitly configured KV pool; only auto may shrink. The
// startup budget already excludes foreign allocations and the chosen reserve;
// add that reserve back, then subtract the exact runtime and required slack.
inline nlohmann::json model_reserve_budget(const nlohmann::json& admin,
                                          const std::vector<std::string>& args) {
    nlohmann::json out{{"ok", false}, {"reason", "Start the engine and wait for its memory plan before adjusting the reserve."}};
    try {
        const auto& plan = admin.at("plan");
        const auto& reserve = admin.at("desktop_reserve");
        const double runtime = plan.at("runtime_reservation_bytes").get<double>();
        const double weights = admin.at("arenas").at("weights").at("capacity_bytes").get<double>();
        const double available = plan.at("available_after_weights_bytes").get<double>();
        const double configured = reserve.at("configured_bytes").get<double>();
        const double free = admin.at("device").at("free_bytes").get<double>();
        const double graphs = plan.at("cuda_graph_allowance_bytes").get<double>();
        // Match serve_options: explicit capacity uses zero sizing slack. Auto
        // uses max(1 GiB automatic headroom, the requested/default slack floor).
        bool automatic = false;
        double slack = -1;
        for (std::size_t i = 0; i + 1 < args.size(); ++i) {
            if (args[i] == "--kv-capacity") automatic = args[i + 1] == "auto";
            if (args[i] == "--kv-slack-floor-mib") slack = std::stod(args[i + 1]) * 1048576.0;
        }
        slack = automatic ? std::max(1073741824.0, slack) : 0;
        const double required_runtime = automatic ? plan.at("minimum_runtime_reservation_bytes").get<double>() : runtime;
        if (runtime <= 0 || weights <= 0 || available < 0 || configured < 0 || free < 0 || graphs < 0) return out;
        if (required_runtime <= 0 || required_runtime > runtime) return out;
        const double startup_limit = available + configured - required_runtime - slack;
        const double live_limit = free + (runtime - required_runtime) - graphs - slack;
        const int maximum = static_cast<int>(std::floor(std::max(0.0, std::min({startup_limit, live_limit, 64.0 * 1073741824.0})) / 1073741824.0));
        return {{"ok", true}, {"max_gib", maximum}, {"weights_bytes", weights},
                {"runtime_bytes", required_runtime}, {"automatic_capacity", automatic}, {"model", admin.value("model_id", std::string{})},
                {"reason", "Keeps the loaded model and configured capacity policy, including startup slack and graph headroom. Other apps can reduce this limit."}};
    } catch (...) { return out; }
}
} // namespace ninfer::supervisor
