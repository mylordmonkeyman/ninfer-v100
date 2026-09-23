#pragma once

// Tray preferences: the memory posture the user chose, persisted next to the
// supervisor config so it survives a restart.
//
// These are deliberately few. The tray asks for a posture, not a tuning: a
// desktop reserve the engine must leave alone, and how long an idle engine
// should stay resident. Everything finer belongs in the config file for the
// handful of people who want it.

#include "logic.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <mutex>

namespace ninfer::supervisor {

struct TrayPrefs {
    // GiB of device memory the engine must not allocate into.
    //
    // The default is kReserveUnset, not a number: until the user picks one the
    // config file's own --desktop-reserve-gib is the truth. A tray default of 3
    // would have quietly lowered every 8 GiB config to 3 GiB the first time the
    // supervisor ran, which is a 5 GiB change to the memory envelope that nobody
    // asked for. 0 means the user did choose, and chose the engine's default.
    int desktop_reserve_gib = kReserveUnset;

    // Minutes of inactivity after which the engine is unloaded entirely. 0 = never.
    // Conservative by design -- unloading after five minutes would punish anyone
    // who steps away mid-task, while an hour reliably catches "finished for the
    // day" and little else.
    int idle_unload_minutes = 0;
};

// Replace only a trailing extension, and only one that belongs to the file name.
// find_last_of('.') over the whole path turned "C:/dir.v2/config" into
// "C:/dir.tray.json" -- a preferences file in the wrong directory, silently.
inline std::string tray_prefs_path(const std::string& config_path) {
    const auto slash = config_path.find_last_of("/\\");
    const auto dot   = config_path.find_last_of('.');
    const auto name_start = slash == std::string::npos ? std::size_t{0} : slash + 1;
    // `> name_start`, not `>=`: a leading dot is a hidden file's name, not an
    // extension, so ".ninfer" must not become ".tray.json".
    const bool has_ext = dot != std::string::npos && dot > name_start;
    const std::string base = has_ext ? config_path.substr(0, dot) : config_path;
    return base + ".tray.json";
}

// A hand-edited preferences file must never be able to hurt the engine: a
// negative reserve becomes `--desktop-reserve-gib -5`, which the engine rejects
// in parse_nonnegative_int and turns into a crash loop the tray then reports as
// the engine's fault.
inline TrayPrefs normalize_tray_prefs(TrayPrefs p) {
    const bool reserve_ok =
        p.desktop_reserve_gib == kReserveUnset ||
        (p.desktop_reserve_gib >= 0 && p.desktop_reserve_gib <= 64);
    if (!reserve_ok) { p.desktop_reserve_gib = kReserveUnset; }
    const bool idle_ok = std::find(kIdleChoicesMinutes.begin(), kIdleChoicesMinutes.end(),
                                   p.idle_unload_minutes) != kIdleChoicesMinutes.end();
    if (!idle_ok) { p.idle_unload_minutes = 0; }
    return p;
}

inline TrayPrefs parse_tray_prefs_json(std::string_view text) {
    TrayPrefs p;
    try {
        const auto j = nlohmann::json::parse(text);
        if (!j.is_object()) { return TrayPrefs{}; }
        if (j.contains("desktop_reserve_gib") && j["desktop_reserve_gib"].is_number_integer()) {
            p.desktop_reserve_gib = j["desktop_reserve_gib"].get<int>();
        }
        if (j.contains("idle_unload_minutes") && j["idle_unload_minutes"].is_number_integer()) {
            p.idle_unload_minutes = j["idle_unload_minutes"].get<int>();
        }
    } catch (...) {
        // A corrupt preferences file must never stop the supervisor starting.
        return TrayPrefs{};
    }
    return normalize_tray_prefs(p);
}

inline std::string tray_prefs_to_json(const TrayPrefs& p) {
    const nlohmann::json j{{"desktop_reserve_gib", p.desktop_reserve_gib},
                           {"idle_unload_minutes", p.idle_unload_minutes}};
    return j.dump(2);
}

inline TrayPrefs load_tray_prefs(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { return TrayPrefs{}; }
    std::ostringstream buf;
    buf << in.rdbuf();
    return parse_tray_prefs_json(buf.str());
}

inline void save_tray_prefs(const std::string& path, const TrayPrefs& p) {
    try {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        if (out) { out << tray_prefs_to_json(p) << "\n"; }
    } catch (...) {
        // Best effort. Failing to persist a preference is not worth an error path.
    }
}

// Both UI surfaces update only their field, preserving the other preference.
inline bool update_tray_preference(const std::string& path, bool reserve, int value) {
    static std::mutex mutex;
    std::lock_guard lock(mutex);
    auto prefs = load_tray_prefs(path);
    if (reserve) prefs.desktop_reserve_gib = value;
    else prefs.idle_unload_minutes = value;
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << tray_prefs_to_json(prefs) << '\n';
    out.flush();
    return out.good();
}

} // namespace ninfer::supervisor
