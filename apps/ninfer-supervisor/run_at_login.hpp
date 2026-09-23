#pragma once

// HKCU\...\Run entry for "Start at login". Header-only Win32 so both main() (the
// --install-login CLI) and the tray (the checked menu item) go through exactly
// one implementation -- the previous split had the CLI writing the key and the
// tray unable to tell whether it was set.

#include "logic.hpp" // run_at_login_command_matches

#include <windows.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::supervisor {

inline constexpr wchar_t kRunKeyPath[]  = L"Software\\Microsoft\\Windows\\CurrentVersion\\Run";
inline constexpr wchar_t kRunValueName[] = L"NInferSupervisor";

// UTF-8 -> UTF-16 through the codepage converter. The first version widened
// bytes one by one, which is correct only for ASCII: any accented character in
// the install path produced a Run entry that pointed nowhere.
inline std::wstring widen_utf8(const std::string& s) {
    if (s.empty()) { return {}; }
    const int n =
        MultiByteToWideChar(CP_UTF8, 0, s.c_str(), static_cast<int>(s.size()), nullptr, 0);
    if (n <= 0) { return {}; }
    std::wstring out(static_cast<std::size_t>(n), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), static_cast<int>(s.size()), out.data(), n);
    return out;
}

inline std::string narrow_utf8(const std::wstring& s) {
    if (s.empty()) { return {}; }
    const int n = WideCharToMultiByte(CP_UTF8, 0, s.c_str(), static_cast<int>(s.size()), nullptr, 0,
                                      nullptr, nullptr);
    if (n <= 0) { return {}; }
    std::string out(static_cast<std::size_t>(n), '\0');
    WideCharToMultiByte(CP_UTF8, 0, s.c_str(), static_cast<int>(s.size()), out.data(), n, nullptr,
                        nullptr);
    return out;
}

[[nodiscard]] inline bool run_at_login_installed() {
    HKEY key = nullptr;
    if (RegOpenKeyExW(HKEY_CURRENT_USER, kRunKeyPath, 0, KEY_QUERY_VALUE, &key) != ERROR_SUCCESS) {
        return false;
    }
    DWORD type = 0;
    DWORD size = 0;
    const LONG st = RegQueryValueExW(key, kRunValueName, nullptr, &type, nullptr, &size);
    RegCloseKey(key);
    return st == ERROR_SUCCESS && size > 0;
}

// The stored command, UTF-8, or empty when there is no entry. Read rather than
// merely probed, because "Start at login" is a per-config question and there is
// only one value name for every config.
[[nodiscard]] inline std::string run_at_login_command_stored() {
    HKEY key = nullptr;
    if (RegOpenKeyExW(HKEY_CURRENT_USER, kRunKeyPath, 0, KEY_QUERY_VALUE, &key) != ERROR_SUCCESS) {
        return {};
    }
    DWORD type = 0;
    DWORD size = 0;
    if (RegQueryValueExW(key, kRunValueName, nullptr, &type, nullptr, &size) != ERROR_SUCCESS ||
        (type != REG_SZ && type != REG_EXPAND_SZ) || size < sizeof(wchar_t)) {
        RegCloseKey(key);
        return {};
    }
    std::vector<wchar_t> buf(size / sizeof(wchar_t) + 1, L'\0');
    DWORD got = size;
    const LONG st = RegQueryValueExW(key, kRunValueName, nullptr, &type,
                                     reinterpret_cast<BYTE*>(buf.data()), &got);
    RegCloseKey(key);
    if (st != ERROR_SUCCESS) { return {}; }
    // A REG_SZ is not required to be terminated; trust the returned length.
    std::size_t n = got / sizeof(wchar_t);
    while (n > 0 && buf[n - 1] == L'\0') { --n; }
    return narrow_utf8(std::wstring(buf.data(), n));
}

[[nodiscard]] inline bool run_at_login_installed_for(const std::string& command) {
    return run_at_login_command_matches(run_at_login_command_stored(), command);
}

inline void install_run_at_login(const std::string& command) {
    HKEY key = nullptr;
    if (RegCreateKeyExW(HKEY_CURRENT_USER, kRunKeyPath, 0, nullptr, 0, KEY_SET_VALUE, nullptr, &key,
                        nullptr) != ERROR_SUCCESS) {
        throw std::runtime_error("cannot open Run key");
    }
    const std::wstring w = widen_utf8(command);
    const LONG st        = RegSetValueExW(key, kRunValueName, 0, REG_SZ,
                                          reinterpret_cast<const BYTE*>(w.c_str()),
                                          static_cast<DWORD>((w.size() + 1) * sizeof(wchar_t)));
    RegCloseKey(key);
    if (st != ERROR_SUCCESS) { throw std::runtime_error("cannot write Run key"); }
}

inline void uninstall_run_at_login() {
    HKEY key = nullptr;
    if (RegOpenKeyExW(HKEY_CURRENT_USER, kRunKeyPath, 0, KEY_SET_VALUE, &key) != ERROR_SUCCESS) {
        return;
    }
    RegDeleteValueW(key, kRunValueName);
    RegCloseKey(key);
}

// Full path of this executable, UTF-8. GetModuleFileNameW rather than the ANSI
// form for the same reason as widen_utf8.
[[nodiscard]] inline std::string module_path_utf8() {
    std::vector<wchar_t> buf(MAX_PATH);
    for (;;) {
        const DWORD n = GetModuleFileNameW(nullptr, buf.data(), static_cast<DWORD>(buf.size()));
        if (n == 0) { return {}; }
        if (n < buf.size() - 1) { return narrow_utf8(std::wstring(buf.data(), n)); }
        buf.resize(buf.size() * 2);
    }
}

} // namespace ninfer::supervisor
