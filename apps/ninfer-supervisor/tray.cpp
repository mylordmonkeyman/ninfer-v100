#include "tray.hpp"

#include "run_at_login.hpp"

#include <windows.h>
#include <shellapi.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>

namespace ninfer::supervisor {
namespace {

constexpr UINT kTrayMsg     = WM_APP + 1;
constexpr UINT_PTR kTimer   = 1;
constexpr UINT_PTR kAddRetry = 2;
constexpr UINT kTrayUid     = 1;
constexpr wchar_t kClass[]  = L"NInferSupervisorTray";

// Explorer broadcasts this after it restarts and after an early-login taskbar
// comes up. Without handling it the process is still running but has no icon --
// invisible and unkillable except through Task Manager.
UINT g_taskbar_created = 0;

std::int64_t now_unix_ms() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

COLORREF status_fill(TrayStatus status) {
    switch (status) {
    case TrayStatus::Working: return RGB(0x2F, 0x9E, 0x54);
    case TrayStatus::Pending: return RGB(0xD1, 0x8B, 0x12);
    case TrayStatus::Failed: return RGB(0xC5, 0x3B, 0x33);
    case TrayStatus::Idle: break;
    }
    return RGB(0x6B, 0x72, 0x80);
}

const wchar_t* status_word(TrayStatus status) {
    switch (status) {
    case TrayStatus::Working: return L"running";
    case TrayStatus::Pending: return L"pending";
    case TrayStatus::Failed: return L"failed";
    case TrayStatus::Idle: break;
    }
    return L"idle";
}

std::wstring widen(const std::string& in) {
    if (in.empty()) { return {}; }
    const int n = MultiByteToWideChar(CP_UTF8, 0, in.c_str(), static_cast<int>(in.size()), nullptr, 0);
    if (n <= 0) { return {}; }
    std::wstring out(static_cast<std::size_t>(n), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, in.c_str(), static_cast<int>(in.size()), out.data(), n);
    return out;
}

// A status dot for the header line, matching the tray icon's colour. Same 24bpp
// DIB reasoning as make_tray_icon: a screen-compatible DDB is 32bpp and menus
// read its alpha, which GDI never writes, so the dot would come out invisible.
HBITMAP make_status_dot(TrayStatus status) {
    const int d = GetSystemMetrics(SM_CXMENUCHECK);
    const int size = d > 0 ? d : 16;
    HDC screen = GetDC(nullptr);
    if (screen == nullptr) { return nullptr; }
    BITMAPINFO bi{};
    bi.bmiHeader.biSize        = sizeof(BITMAPINFOHEADER);
    bi.bmiHeader.biWidth       = size;
    bi.bmiHeader.biHeight      = size;
    bi.bmiHeader.biPlanes      = 1;
    bi.bmiHeader.biBitCount    = 24;
    bi.bmiHeader.biCompression = BI_RGB;
    void* bits = nullptr;
    HBITMAP bmp = CreateDIBSection(screen, &bi, DIB_RGB_COLORS, &bits, nullptr, 0);
    if (bmp != nullptr) {
        HDC dc = CreateCompatibleDC(screen);
        if (dc == nullptr) {
            // No DC means no drawing surface; an undrawn DIB would render as a
            // black square, which reads as a fourth status colour.
            DeleteObject(bmp);
            ReleaseDC(nullptr, screen);
            return nullptr;
        }
        HGDIOBJ old = SelectObject(dc, bmp);
        // Menu background varies with theme; COLOR_MENU is the closest system
        // answer and keeps the dot from sitting on a wrong-coloured tile.
        HBRUSH back = GetSysColorBrush(COLOR_MENU);
        RECT all{0, 0, size, size};
        FillRect(dc, &all, back);
        HBRUSH fill = CreateSolidBrush(status_fill(status));
        HGDIOBJ oldb = SelectObject(dc, fill);
        HGDIOBJ oldp = SelectObject(dc, GetStockObject(NULL_PEN));
        const int inset = size / 4;
        Ellipse(dc, inset, inset, size - inset, size - inset);
        SelectObject(dc, oldp);
        SelectObject(dc, oldb);
        DeleteObject(fill);
        SelectObject(dc, old);
        DeleteDC(dc);
    }
    ReleaseDC(nullptr, screen);
    return bmp;
}

// Free memory leads, because that is the number a person can act on; the split is
// secondary. Reads device-wide truth -- see query_device_memory_smi.
std::wstring memory_line(const NvidiaSmiMemory& m) {
    if (!m.ok || m.total_mib == 0) { return L"GPU memory unavailable"; }
    wchar_t buf[128];
    const double freeg = static_cast<double>(m.total_mib - m.used_mib) / 1024.0;
    const double tot   = static_cast<double>(m.total_mib) / 1024.0;
    _snwprintf_s(buf, _TRUNCATE, L"%.1f GB free of %.0f GB", freeg, tot);
    return buf;
}

void append_caption(HMENU menu, const std::wstring& text, HBITMAP dot = nullptr) {
    AppendMenuW(menu, MF_STRING | MF_DISABLED | MF_GRAYED, 0, text.c_str());
    if (dot != nullptr) {
        MENUITEMINFOW mii{};
        mii.cbSize     = sizeof(mii);
        mii.fMask      = MIIM_BITMAP;
        mii.hbmpItem   = dot;
        SetMenuItemInfoW(menu, GetMenuItemCount(menu) - 1, TRUE, &mii);
    }
}

// A greyed item always says WHY it is greyed. Docker greys "Kubernetes Context"
// with its reason in the right-hand column; an item that is simply dead teaches
// the user to stop reading the menu.
void append_disabled(HMENU menu, UINT id, const std::wstring& label, const std::wstring& reason) {
    std::wstring text = label;
    if (!reason.empty()) {
        text += L"\t";
        text += reason;
    }
    AppendMenuW(menu, MF_STRING | MF_DISABLED | MF_GRAYED, id, text.c_str());
}

int small_icon_size() {
    const int size = GetSystemMetrics(SM_CXSMICON);
    return size > 0 ? size : 16;
}

// Drawn separately from the application's launcher icon so the fill colour
// carries status at whatever SM_CXSMICON the display scaling reports.
//
// The colour bitmap is an explicit 24bpp DIB section, NOT CreateCompatibleBitmap.
// A screen-compatible DDB is 32bpp on any modern display, and CreateIconIndirect
// then reads its alpha channel as per-pixel alpha. GDI never writes alpha, so
// every pixel would come out fully transparent and the icon would vanish. At
// 24bpp there is no alpha channel to misread and the 1bpp mask decides shape.
HICON make_tray_icon(TrayStatus status, int size) {
    HDC screen = GetDC(nullptr);
    if (screen == nullptr) { return nullptr; }

    BITMAPINFO bi{};
    bi.bmiHeader.biSize        = sizeof(BITMAPINFOHEADER);
    bi.bmiHeader.biWidth       = size;
    bi.bmiHeader.biHeight      = -size; // top-down
    bi.bmiHeader.biPlanes      = 1;
    bi.bmiHeader.biBitCount    = 24;
    bi.bmiHeader.biCompression = BI_RGB;

    void* bits        = nullptr;
    HBITMAP color_bmp = CreateDIBSection(screen, &bi, DIB_RGB_COLORS, &bits, nullptr, 0);
    HBITMAP mask_bmp  = CreateBitmap(size, size, 1, 1, nullptr);
    HDC color_dc      = CreateCompatibleDC(screen);
    HDC mask_dc       = CreateCompatibleDC(screen);
    if (color_bmp == nullptr || mask_bmp == nullptr || color_dc == nullptr || mask_dc == nullptr) {
        if (color_dc != nullptr) { DeleteDC(color_dc); }
        if (mask_dc != nullptr) { DeleteDC(mask_dc); }
        if (color_bmp != nullptr) { DeleteObject(color_bmp); }
        if (mask_bmp != nullptr) { DeleteObject(mask_bmp); }
        ReleaseDC(nullptr, screen);
        return nullptr;
    }
    auto* old_color = static_cast<HBITMAP>(SelectObject(color_dc, color_bmp));
    auto* old_mask  = static_cast<HBITMAP>(SelectObject(mask_dc, mask_bmp));

    // Full-bleed fill; the rounded corners are cut by the mask, so the glyph
    // antialiases against the fill rather than fringing against a background.
    RECT rc{0, 0, size, size};
    HBRUSH fill = CreateSolidBrush(status_fill(status));
    FillRect(color_dc, &rc, fill);
    DeleteObject(fill);

    LOGFONTW lf{};
    lf.lfHeight  = -(size * 3 / 4);
    lf.lfWeight  = FW_BOLD;
    lf.lfQuality = ANTIALIASED_QUALITY;
    lf.lfCharSet = DEFAULT_CHARSET;
    lstrcpyW(lf.lfFaceName, L"Segoe UI");
    HFONT font = CreateFontIndirectW(&lf);
    if (font != nullptr) {
        auto* old_font = static_cast<HFONT>(SelectObject(color_dc, font));
        SetBkMode(color_dc, TRANSPARENT);
        SetTextColor(color_dc, RGB(0xFF, 0xFF, 0xFF));
        DrawTextW(color_dc, L"N", 1, &rc, DT_CENTER | DT_VCENTER | DT_SINGLELINE | DT_NOCLIP);
        SelectObject(color_dc, old_font);
        DeleteObject(font);
    }

    // 1bpp mask: white (1) transparent, black (0) opaque.
    PatBlt(mask_dc, 0, 0, size, size, WHITENESS);
    HPEN pen         = CreatePen(PS_SOLID, 1, RGB(0, 0, 0));
    auto* old_brush  = static_cast<HBRUSH>(SelectObject(mask_dc, GetStockObject(BLACK_BRUSH)));
    auto* old_pen    = static_cast<HPEN>(SelectObject(mask_dc, pen));
    const int radius = size / 3 < 2 ? 2 : size / 3;
    RoundRect(mask_dc, 0, 0, size, size, radius, radius);
    SelectObject(mask_dc, old_brush);
    SelectObject(mask_dc, old_pen);
    DeleteObject(pen);

    SelectObject(color_dc, old_color);
    SelectObject(mask_dc, old_mask);

    ICONINFO info{};
    info.fIcon    = TRUE;
    info.hbmMask  = mask_bmp;
    info.hbmColor = color_bmp;
    HICON icon    = CreateIconIndirect(&info);

    DeleteObject(color_bmp);
    DeleteObject(mask_bmp);
    DeleteDC(color_dc);
    DeleteDC(mask_dc);
    ReleaseDC(nullptr, screen);
    return icon;
}

std::wstring reserve_label(int gib) {
    wchar_t label[64];
    if (gib <= 0) {
        _snwprintf_s(label, _TRUNCATE, L"Engine default (8 GiB)");
    } else {
        const wchar_t* note = L"";
        _snwprintf_s(label, _TRUNCATE, L"%d GiB%s", gib, note);
    }
    return label;
}

std::wstring idle_label(int mins) {
    wchar_t label[64];
    if (mins == 0) {
        _snwprintf_s(label, _TRUNCATE, L"Never");
    } else if (mins < 60) {
        _snwprintf_s(label, _TRUNCATE, L"After %d minutes", mins);
    } else {
        _snwprintf_s(label, _TRUNCATE, L"After %d hour%s", mins / 60, mins == 60 ? L"" : L"s");
    }
    return label;
}

void show_menu(HWND hwnd, TrayIcon* self) {
    POINT pt{};
    GetCursorPos(&pt);
    HMENU menu = CreatePopupMenu();
    const EngineStatus est   = self->child().status();
    const TrayStatus tstatus = self->menu_status(est);
    const bool running       = est.state == EngineState::Running;
    const bool alive         = running || est.state == EngineState::Starting;

    HBITMAP dot = make_status_dot(tstatus);
    append_caption(menu, self->header_line(est), dot);
    const std::wstring detail = self->detail_line();
    if (!detail.empty()) { append_caption(menu, L"    " + detail); }
    append_caption(menu, L"    " + memory_line(self->collector().last_nvidia()));
    AppendMenuW(menu, MF_SEPARATOR, 0, nullptr);

    if (self->dashboard_unavailable()) {
        append_disabled(menu, kCmdOpenDashboard, L"Go to the dashboard", L"port in use");
    } else {
        AppendMenuW(menu, MF_STRING, kCmdOpenDashboard, L"Go to the dashboard");
    }
    AppendMenuW(menu, MF_STRING, kCmdCopyUrl, L"Copy dashboard URL");
    AppendMenuW(menu, MF_STRING, kCmdOpenLogs, L"Open logs folder");
    AppendMenuW(menu, MF_SEPARATOR, 0, nullptr);

    if (self->reserve_pinned_by_config()) {
        // The config pins --desktop-reserve-mib, which is finer than anything
        // this menu can say, so with_desktop_reserve leaves the arguments alone.
        // A live submenu here would let the user pick a number, write it to
        // prefs, and be offered a full model reload that changes nothing. Id 0
        // because a greyed item is never returned by TrackPopupMenu.
        append_disabled(menu, 0, L"Reserve for the desktop", L"pinned by --desktop-reserve-mib");
    } else {
        HMENU reserve = CreatePopupMenu();
        const int selected = self->reserve_gib();
        const auto budget = self->child().reserve_budget();
        if (std::find(kReserveChoicesGib.begin(), kReserveChoicesGib.end(), selected) == kReserveChoicesGib.end()) {
            append_caption(reserve, (std::to_wstring(selected) + L" GiB custom (dashboard)").c_str());
        }
        for (std::size_t i = 0; i < kReserveChoicesGib.size(); ++i) {
            const int gib = kReserveChoicesGib[i];
            const bool fits = budget.value("ok", false) && (gib == 0 ? 8 : gib) <= budget.value("max_gib", 0);
            AppendMenuW(reserve, MF_STRING | (fits ? 0 : MF_GRAYED) | (gib == self->reserve_gib() ? MF_CHECKED : 0),
                        tray_cmd_reserve(i), reserve_label(gib).c_str());
        }
        if (alive) {
            // The reserve is a sizing input read once at model load; there is no
            // runtime enforcement, so say when the choice takes effect rather
            // than implying a running engine can be resized.
            AppendMenuW(reserve, MF_SEPARATOR, 0, nullptr);
            append_caption(reserve, L"Applies at next start");
        }
        AppendMenuW(menu, MF_POPUP, reinterpret_cast<UINT_PTR>(reserve),
                    L"Reserve for the desktop");
    }

    HMENU idle = CreatePopupMenu();
    for (std::size_t i = 0; i < kIdleChoicesMinutes.size(); ++i) {
        const int mins = kIdleChoicesMinutes[i];
        AppendMenuW(idle, MF_STRING | (mins == self->idle_minutes() ? MF_CHECKED : 0),
                    tray_cmd_idle(i), idle_label(mins).c_str());
    }
    AppendMenuW(menu, MF_POPUP, reinterpret_cast<UINT_PTR>(idle), L"Unload when idle");

    // No "Release cache" item. It sat here disabled, explaining that it needed engine
    // support, which read as "pass a flag and this works" -- and no such flag exists.
    // The engine allocates device memory once at startup and holds it; freeing any of it
    // needs a residency path in the target that can rebuild what it drops. Until a target
    // has one there is nothing to release, so the menu does not offer it. The dashboard
    // reports the same fact from /admin/vram, where it belongs.

    AppendMenuW(menu, MF_STRING | (self->start_at_login_enabled() ? MF_CHECKED : 0),
                kCmdStartAtLogin, L"Start at login");
    AppendMenuW(menu, MF_SEPARATOR, 0, nullptr);

    if (self->manages_engine()) {
        AppendMenuW(menu, MF_STRING, kCmdRestart, L"Restart engine");
        AppendMenuW(menu, MF_STRING, running ? kCmdStop : kCmdStart,
                    running ? L"Unload model\tfrees ~70 GB" : L"Load model");
    } else {
        append_disabled(menu, kCmdRestart, L"Restart engine", L"monitor-only");
    }
    AppendMenuW(menu, MF_STRING, kCmdQuit, L"Quit supervisor");

    SetForegroundWindow(hwnd);
    const int cmd = TrackPopupMenu(menu, TPM_RETURNCMD | TPM_NONOTIFY, pt.x, pt.y, 0, hwnd, nullptr);
    // MS KB135788: without a message posted to the owning window the menu can
    // fail to dismiss when the user clicks outside it.
    PostMessageW(hwnd, WM_NULL, 0, 0);
    DestroyMenu(menu);
    if (dot != nullptr) { DeleteObject(dot); }

    const TrayCommand action = decode_tray_command(static_cast<unsigned>(cmd));
    switch (action.action) {
    case TrayAction::OpenDashboard: self->open_dashboard(); break;
    case TrayAction::CopyUrl: self->copy_dashboard_url(); break;
    case TrayAction::OpenLogs: self->open_logs_folder(); break;
    case TrayAction::Start: self->child().start(); break;
    case TrayAction::Stop: self->child().stop(); break;
    case TrayAction::Restart: self->child().restart(); break;
    case TrayAction::Quit: PostQuitMessage(0); break;
    case TrayAction::Reserve: self->on_reserve_chosen(action.value); break;
    case TrayAction::Idle: self->on_idle_chosen(action.value); break;
    case TrayAction::StartAtLogin: self->on_start_at_login_toggled(); break;
    case TrayAction::None: break;
    }
}

LRESULT CALLBACK tray_wnd(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam) {
    TrayIcon* self = nullptr;
    if (msg == WM_NCCREATE) {
        auto* cs = reinterpret_cast<CREATESTRUCTW*>(lparam);
        self     = static_cast<TrayIcon*>(cs->lpCreateParams);
        SetWindowLongPtrW(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(self));
    } else {
        self = reinterpret_cast<TrayIcon*>(GetWindowLongPtrW(hwnd, GWLP_USERDATA));
    }
    if (self == nullptr) { return DefWindowProcW(hwnd, msg, wparam, lparam); }
    if (g_taskbar_created != 0 && msg == g_taskbar_created) {
        self->refresh_icon_add();
        return 0;
    }
    if (msg == WM_TIMER && wparam == kAddRetry) {
        // NIM_ADD fails when the process wins the race against Explorer at
        // login. Retry until it takes, then drop the timer.
        if (self->try_add_icon()) { KillTimer(hwnd, kAddRetry); }
        return 0;
    }
    if (msg == WM_TIMER && wparam == kTimer) {
        self->refresh_icon();
        self->tick_idle_unload();
        return 0;
    }
    if (msg == kTrayMsg && LOWORD(lparam) == WM_LBUTTONUP) {
        // Docker Desktop opens its dashboard on left-click; the menu is the
        // right-click surface. Matching that is one less thing to learn.
        self->open_dashboard();
        return 0;
    }
    if (msg == kTrayMsg && (LOWORD(lparam) == WM_RBUTTONUP || LOWORD(lparam) == WM_CONTEXTMENU)) {
        show_menu(hwnd, self);
        return 0;
    }
    if (msg == WM_DESTROY) {
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProcW(hwnd, msg, wparam, lparam);
}

} // namespace

TrayIcon::TrayIcon(EngineChild& child, Collector& collector, std::string dashboard_url,
                   bool manages_engine, std::string prefs_path, std::string config_abs_path)
    : child_(child), collector_(collector), dashboard_url_(std::move(dashboard_url)),
      manages_engine_(manages_engine), prefs_path_(std::move(prefs_path)),
      config_abs_path_(std::move(config_abs_path)), prefs_(load_tray_prefs(prefs_path_)) {
    // The stored reserve is the user's standing choice, so the engine must be
    // told it before the first spawn rather than only after they touch the menu.
    // It goes to EngineChild, which owns the config copy that spawn() actually
    // reads -- rewriting main's copy, which is what the first version did, could
    // never reach the command line.
    child_.set_desktop_reserve_gib(prefs_.desktop_reserve_gib);
}

TrayIcon::~TrayIcon() {
    if (hwnd_ != nullptr) { DestroyWindow(static_cast<HWND>(hwnd_)); }
    if (hicon_ != nullptr) { DestroyIcon(static_cast<HICON>(hicon_)); }
}

EngineChild& TrayIcon::child() { return child_; }

int TrayIcon::reserve_gib() const {
    const int saved = child_.status().desktop_reserve_gib;
    if (saved >= 0) { return saved; }
    // No stored choice: the checkmark belongs on whatever the config file asks
    // for, so the menu never claims a number the engine is not running with.
    return desktop_reserve_from_args(child_.config().engine.args);
}

bool TrayIcon::dashboard_unavailable() const {
    return dashboard_listen_failed_ && dashboard_listen_failed_();
}

bool TrayIcon::reserve_pinned_by_config() const {
    return desktop_reserve_pinned(child_.config().engine.args);
}

// Not "does the Run value exist" but "does it point at this config": there is
// one value name, and existence alone put a checkmark on the menu of a config
// the entry does not name.
bool TrayIcon::start_at_login_enabled() const {
    return run_at_login_installed_for(run_at_login_command(module_path_utf8(), config_abs_path_));
}

std::wstring TrayIcon::header_line(const EngineStatus& st) const {
    const TrayStatus status = st.crash_loop_halted || st.state == EngineState::Halted
                                  ? TrayStatus::Failed
                                  : status_from(st);
    std::wstring head = L"NInfer is ";
    head += status_word(status);
    if (!manages_engine_) { head += L" (observing)"; }
    if (status == TrayStatus::Failed && !st.last_error_line.empty()) {
        head += L" - ";
        head += widen(st.last_error_line);
    }
    return head;
}

std::wstring TrayIcon::detail_line() const {
    const std::string model = model_label_from_args(child_.config().engine.args);
    std::wstring out        = model.empty() ? L"engine" : widen(model);
    out += L" on :";
    out += std::to_wstring(child_.config().engine.engine_port);
    return out;
}

void TrayIcon::on_reserve_chosen(int gib) {
    // Against the EFFECTIVE reserve, not the raw preference. The preference is
    // kReserveUnset until the user has ever chosen, so comparing to it meant
    // that on a fresh install clicking the item that already carries the
    // checkmark popped a "restart? the model reloads" dialog for a no-op.
    if (gib == reserve_gib()) { return; }
    // Nothing the menu can say survives a pinned --desktop-reserve-mib; the item
    // is greyed, and this is the guard for the path that greying does not cover.
    if (reserve_pinned_by_config()) { return; }
    if (!child_.save_desktop_reserve_gib(gib)) {
        MessageBoxW(nullptr, L"Could not save the reserve. It must fit the ready model's current memory budget. Check the dashboard for the available limit and retry.", L"NInfer", MB_OK | MB_ICONERROR);
        return;
    }

    // Ask before restarting, and only when there is something to interrupt.
    // Reserving memory is cheap to choose and expensive to apply -- a restart
    // costs a full model load -- so the cost is stated and the decision is the
    // user's. A stopped engine just picks it up on the next start.
    const EngineStatus st = child_.status();
    if (st.state == EngineState::Running) {
        wchar_t msg[256];
        _snwprintf_s(msg, _TRUNCATE,
                     L"Reserve set to %s.\n\nThis applies when the engine next starts. "
                     L"Restart now? The model reloads, which takes about a minute.",
                     reserve_label(gib).c_str());
        if (MessageBoxW(nullptr, msg, L"NInfer", MB_YESNO | MB_ICONQUESTION) == IDYES) {
            child_.restart();
        }
    }
}

void TrayIcon::on_idle_chosen(int minutes) { set_idle_minutes(minutes); }

void TrayIcon::set_idle_minutes(int minutes) {
    prefs_.idle_unload_minutes = minutes;
    update_tray_preference(prefs_path_, false, prefs_.idle_unload_minutes);
}

void TrayIcon::on_start_at_login_toggled() {
    try {
        const std::string cmd = run_at_login_command(module_path_utf8(), config_abs_path_);
        if (run_at_login_installed_for(cmd)) {
            // Only ever removes an entry that names this config, so unchecking
            // here cannot delete another configuration's autostart.
            uninstall_run_at_login();
            return;
        }
        install_run_at_login(cmd);
    } catch (const std::exception& ex) {
        MessageBoxW(nullptr, widen(ex.what()).c_str(), L"NInfer", MB_OK | MB_ICONWARNING);
    }
}

// Unloads an idle engine so a workstation is not holding ~70 GB of weights while
// nobody is using it.
//
// "Idle" is now measured rather than assumed: the engine's own stderr says when
// a request arrives or finishes, and the request log's mtime says the same thing
// across a supervisor restart. The previous version reset its clock only on a
// state change, so an engine that served continuously for longer than the window
// was terminated mid-request.
void TrayIcon::tick_idle_unload() {
    if (prefs_.idle_unload_minutes <= 0 || !manages_engine_) { return; }
    const EngineStatus st = child_.status();
    IdleInputs in;
    in.now_ms           = now_unix_ms();
    in.last_activity_ms = (std::max)(st.last_activity_unix_ms,
                                     collector_.request_log_last_write_unix_ms());
    in.idle_minutes     = prefs_.idle_unload_minutes;
    in.running          = st.state == EngineState::Running;
    in.healthy          = st.health == "ok";
    // The hard floor. Both activity clocks depend on the engine's log cadence --
    // --log-stats-interval-ms 0 removes the throughput line, and a config need
    // not have a request log -- so a 20-minute generation could otherwise emit
    // nothing between submit and done and be terminated in the middle of itself.
    // An open request that has not closed does not depend on any cadence.
    in.inflight_requests = st.inflight_requests;
    if (!idle_unload_due(in)) { return; }
    child_.stop();
    notify(L"NInfer model unloaded",
           L"The engine was idle and its memory has been returned to the desktop. "
           L"Load it again from the tray menu.",
           NIIF_INFO | NIIF_RESPECT_QUIET_TIME);
}

// Device-wide memory. nvidia-smi rather than cudaMemGetInfo or DXGI, both of
// which report an empty card while another process holds 70 GiB.
//
// The menu no longer calls this: a process spawn measured at ~51 ms on the UI
// thread, with a read loop that had no timeout, could freeze the menu for as
// long as nvidia-smi was wedged. Collector::last_nvidia() is the same reading
// taken at 1 Hz on a background thread. Kept for callers with no Collector.
NvidiaSmiMemory query_device_memory_smi(int device) {
    NvidiaSmiMemory out;
    SECURITY_ATTRIBUTES sa{sizeof(sa), nullptr, TRUE};
    HANDLE r = nullptr;
    HANDLE w = nullptr;
    if (!CreatePipe(&r, &w, &sa, 0)) { return out; }
    SetHandleInformation(r, HANDLE_FLAG_INHERIT, 0);

    STARTUPINFOW si{};
    si.cb         = sizeof(si);
    si.dwFlags    = STARTF_USESTDHANDLES;
    si.hStdOutput = w;
    si.hStdError  = w;

    wchar_t cmd[] = L"nvidia-smi --query-gpu=index,memory.used,memory.total "
                    L"--format=csv,noheader,nounits";
    PROCESS_INFORMATION pi{};
    if (!CreateProcessW(nullptr, cmd, nullptr, nullptr, TRUE, CREATE_NO_WINDOW, nullptr, nullptr,
                        &si, &pi)) {
        CloseHandle(r);
        CloseHandle(w);
        return out;
    }
    CloseHandle(w);
    CloseHandle(pi.hThread);

    std::string buf;
    char chunk[512];
    DWORD got = 0;
    while (ReadFile(r, chunk, sizeof(chunk), &got, nullptr) && got > 0) {
        buf.append(chunk, got);
    }
    CloseHandle(r);
    WaitForSingleObject(pi.hProcess, 2000);
    CloseHandle(pi.hProcess);
    return parse_nvidia_smi_memory_csv(buf, device);
}

void TrayIcon::open_dashboard() const {
    ShellExecuteA(nullptr, "open", dashboard_url_.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
}

void TrayIcon::open_logs_folder() const {
    std::error_code ec;
    // Absolute: the logs_dir may be relative to the working directory the
    // supervisor was started in, which Explorer knows nothing about.
    const auto dir = std::filesystem::absolute(child_.logs_dir(), ec);
    if (ec) { return; }
    ShellExecuteW(nullptr, L"open", dir.wstring().c_str(), nullptr, nullptr, SW_SHOWNORMAL);
}

void TrayIcon::copy_dashboard_url() const {
    const std::wstring url = widen(dashboard_url_);
    if (hwnd_ == nullptr || !OpenClipboard(static_cast<HWND>(hwnd_))) { return; }
    EmptyClipboard();
    const std::size_t bytes = (url.size() + 1) * sizeof(wchar_t);
    HGLOBAL mem             = GlobalAlloc(GMEM_MOVEABLE, bytes);
    if (mem != nullptr) {
        auto* dst = static_cast<wchar_t*>(GlobalLock(mem));
        if (dst != nullptr) {
            std::memcpy(dst, url.c_str(), bytes);
            GlobalUnlock(mem);
            // Ownership transfers to the clipboard on success only.
            if (SetClipboardData(CF_UNICODETEXT, mem) == nullptr) { GlobalFree(mem); }
        } else {
            GlobalFree(mem);
        }
    }
    CloseClipboard();
}

void TrayIcon::request_quit() {
    if (hwnd_ != nullptr) { PostMessageW(static_cast<HWND>(hwnd_), WM_CLOSE, 0, 0); }
}

// Everything that needs a status derives it from a snapshot the caller already
// has. Taking a second one inside this function was how a halt could slip
// between the two reads: `status` came back Failed while the `est` handed to
// announce_transition still said crash_loop_halted == false, so the balloon was
// skipped -- and never re-fired, because the enum had already changed.
TrayStatus TrayIcon::current_status() const { return status_from(child_.status()); }

TrayStatus TrayIcon::status_from(const EngineStatus& status) const {
    if (!manages_engine_) {
        // Unmanaged: no process of ours has a state, so observed health is the
        // only thing actually measured. EngineChild keeps it current for
        // unmanaged engines through observe_health().
        if (status.health == "ok") { return TrayStatus::Working; }
        if (status.health == "unhealthy") { return TrayStatus::Failed; }
        if (status.health == "unreachable") { return TrayStatus::Pending; }
        return TrayStatus::Idle;
    }
    if (status.crash_loop_halted) { return TrayStatus::Failed; }
    switch (status.state) {
    case EngineState::Running:
        return status.health == "unhealthy" ? TrayStatus::Failed : TrayStatus::Working;
    case EngineState::Starting:
    case EngineState::Stopping:
    case EngineState::BackingOff: return TrayStatus::Pending;
    case EngineState::Halted: return TrayStatus::Failed;
    case EngineState::Stopped: break;
    }
    return TrayStatus::Idle;
}

// szTip is 128 wchar_t including the terminator, so the reason is what gets
// trimmed. Balloons are suppressed by Focus Assist; the tooltip is the surface
// that always survives, so the "why" has to be here too.
std::wstring TrayIcon::tooltip(TrayStatus status, const EngineStatus& st) const {
    std::wstring tip = L"NInfer - ";
    tip += status_word(status);
    if (!manages_engine_) { tip += L" (monitor-only)"; }
    if (status == TrayStatus::Failed && !st.last_error_line.empty()) {
        tip += L": ";
        tip += widen(st.last_error_line);
    }
    if (tip.size() > 127) { tip.resize(127); }
    return tip;
}

// Reports whether the shell took it. A balloon is a NIM_MODIFY on an icon that
// must already exist, and at login it may not yet -- the one notice that must
// not be lost (the Run entry the config just installed) retries on false.
bool TrayIcon::notify(const wchar_t* title, const std::wstring& body, unsigned niif_flags) {
    if (hwnd_ == nullptr) { return false; }
    NOTIFYICONDATAW nid{};
    nid.cbSize      = sizeof(nid);
    nid.hWnd        = static_cast<HWND>(hwnd_);
    nid.uID         = kTrayUid;
    nid.uFlags      = NIF_INFO;
    nid.dwInfoFlags = niif_flags;
    lstrcpynW(nid.szInfoTitle, title, ARRAYSIZE(nid.szInfoTitle));
    lstrcpynW(nid.szInfo, body.c_str(), ARRAYSIZE(nid.szInfo));
    return Shell_NotifyIconW(NIM_MODIFY, &nid) != FALSE;
}

// One balloon per failure, one per recovery. The halt is the event a person
// needs to be interrupted for -- the engine has given up restarting itself and
// nothing will serve until someone acts.
void TrayIcon::announce_transition(TrayStatus prev, TrayStatus next, const EngineStatus& st) {
    (void)prev;
    if (next == TrayStatus::Failed && st.crash_loop_halted && !halt_announced_) {
        halt_announced_ = true;
        notify(L"NInfer engine stopped",
               widen(format_halt_notice(st.recent_exits, st.crash_window_s, st.last_exit_code,
                                        st.last_error_line)),
               NIIF_ERROR | NIIF_RESPECT_QUIET_TIME);
        return;
    }
    // "Answering requests" has to be true when it is said. Working means the
    // process exists -- spawn() sets Running and ready=false on the next line --
    // so announcing on the enum alone gave an all-clear about a second after the
    // respawn, a minute before the model had loaded, and on a config that was
    // still broken it was a false all-clear followed by another halt balloon.
    // st.ready is the engine's own `server status=ready` or a 200 from /health.
    if (halt_announced_ && next == TrayStatus::Working && st.ready && st.health != "unhealthy") {
        halt_announced_ = false;
        notify(L"NInfer engine is running", L"The engine restarted and is answering requests.",
               NIIF_INFO | NIIF_RESPECT_QUIET_TIME);
    }
}

void TrayIcon::refresh_icon() {
    if (hwnd_ == nullptr) { return; }
    // ONE snapshot, and everything below derives from it. See status_from().
    const EngineStatus est  = child_.status();
    const TrayStatus status = status_from(est);
    const std::wstring tip  = tooltip(status, est);

    // A dashboard that never bound is worth exactly one balloon; after that the
    // greyed menu line carries it.
    if (!listen_announced_ && dashboard_unavailable()) {
        listen_announced_ = true;
        notify(L"NInfer dashboard unavailable",
               L"Another program is already using the dashboard port. The engine is unaffected.",
               NIIF_WARNING | NIIF_RESPECT_QUIET_TIME);
    }
    if (login_install_pending_ &&
        notify(L"NInfer starts at login",
               L"This configuration asked for it (\"run_at_login\"). Turn it off with "
               L"\"Start at login\" in this menu.",
               NIIF_INFO | NIIF_RESPECT_QUIET_TIME)) {
        login_install_pending_ = false;
    }

    // The engine could not reserve the memory its configured plan needed and the
    // supervisor launched a smaller one. Say so, once per plan, with the numbers.
    if (est.kv_capacity_effective > 0 && est.kv_capacity_effective != last_kv_effective_) {
        const std::wstring body =
            L"KV capacity is " + std::to_wstring(est.kv_capacity_effective) + L" of the configured " +
            std::to_wstring(est.kv_capacity_configured) +
            L" tokens: other apps hold GPU memory the full plan needed. The dashboard shows "
            L"the running plan.";
        if (notify(L"NInfer reduced the engine's memory plan", body,
                   NIIF_WARNING | NIIF_RESPECT_QUIET_TIME)) {
            last_kv_effective_ = est.kv_capacity_effective;
        }
    } else if (est.kv_capacity_effective == 0) {
        last_kv_effective_ = 0;
    }

    const bool status_changed = static_cast<int>(status) != last_status_;
    // Readiness is announced too: the recovery balloon waits for it, and it
    // arrives without any change to the status enum.
    const bool ready_changed  = est.ready != last_ready_;
    const auto prev = last_status_ < 0 ? status : static_cast<TrayStatus>(last_status_);
    if (status_changed || ready_changed) { announce_transition(prev, status, est); }
    last_ready_ = est.ready;

    // The key includes the tooltip text, not just the enum: a status that gains
    // a reason without changing colour must still reach the shell.
    if (!status_changed && tip == last_tip_) { return; }
    last_status_    = static_cast<int>(status);
    last_tip_       = tip;

    HICON icon = make_tray_icon(status, small_icon_size());
    NOTIFYICONDATAW nid{};
    nid.cbSize = sizeof(nid);
    nid.hWnd   = static_cast<HWND>(hwnd_);
    nid.uID    = kTrayUid;
    nid.uFlags = NIF_TIP | (icon != nullptr ? NIF_ICON : 0u);
    nid.hIcon  = icon;
    lstrcpynW(nid.szTip, tip.c_str(), ARRAYSIZE(nid.szTip));
    Shell_NotifyIconW(NIM_MODIFY, &nid);

    if (icon != nullptr) {
        if (hicon_ != nullptr) { DestroyIcon(static_cast<HICON>(hicon_)); }
        hicon_ = icon;
    }
}

bool TrayIcon::try_add_icon() {
    if (hwnd_ == nullptr) { return false; }
    const EngineStatus est  = child_.status();
    const TrayStatus status = status_from(est);
    const std::wstring tip  = tooltip(status, est);
    if (hicon_ == nullptr) { hicon_ = make_tray_icon(status, small_icon_size()); }

    NOTIFYICONDATAW nid{};
    nid.cbSize           = sizeof(nid);
    nid.hWnd             = static_cast<HWND>(hwnd_);
    nid.uID              = kTrayUid;
    nid.uFlags           = NIF_MESSAGE | NIF_TIP | NIF_ICON;
    nid.uCallbackMessage = kTrayMsg;
    nid.hIcon = hicon_ != nullptr ? static_cast<HICON>(hicon_)
                                  : LoadIconW(nullptr, MAKEINTRESOURCEW(32512));
    lstrcpynW(nid.szTip, tip.c_str(), ARRAYSIZE(nid.szTip));
    last_status_ = static_cast<int>(status);
    last_tip_    = tip;
    last_ready_  = est.ready;
    return Shell_NotifyIconW(NIM_ADD, &nid) != FALSE;
}

// Explorer restarted: the shell has forgotten every icon, so re-add rather than
// modify. A NIM_MODIFY here would fail silently and the tray would stay empty.
void TrayIcon::refresh_icon_add() {
    if (hwnd_ == nullptr) { return; }
    if (!try_add_icon()) { SetTimer(static_cast<HWND>(hwnd_), kAddRetry, 2000, nullptr); }
}

void TrayIcon::run() {
    if (g_taskbar_created == 0) { g_taskbar_created = RegisterWindowMessageW(L"TaskbarCreated"); }

    WNDCLASSEXW wc{};
    wc.cbSize        = sizeof(wc);
    wc.lpfnWndProc   = tray_wnd;
    wc.hInstance     = GetModuleHandleW(nullptr);
    wc.lpszClassName = kClass;
    RegisterClassExW(&wc);
    // A real top-level window that is simply never shown -- NOT a message-only
    // (HWND_MESSAGE) one. Message-only windows are excluded from broadcasts, and
    // Explorer delivers TaskbarCreated with HWND_BROADCAST, so the handler above
    // could never run: after an Explorer restart the supervisor kept running
    // with no icon, invisible and unkillable except through Task Manager.
    //
    // WS_OVERLAPPED without WS_VISIBLE and no ShowWindow means nothing appears
    // on screen; WS_EX_TOOLWINDOW keeps it out of Alt-Tab and the taskbar.
    // WS_EX_NOACTIVATE is deliberately NOT set: SetForegroundWindow(hwnd) before
    // TrackPopupMenu is what lets the menu dismiss on an outside click.
    HWND hwnd = CreateWindowExW(WS_EX_TOOLWINDOW, kClass, L"NInfer supervisor", WS_OVERLAPPED, 0, 0,
                                0, 0, nullptr, nullptr, wc.hInstance, this);
    hwnd_     = hwnd;

    // NIM_ADD fails when this process beats Explorer to the desktop at login;
    // retry until the shell is ready rather than running with no icon at all.
    if (!try_add_icon()) { SetTimer(hwnd, kAddRetry, 2000, nullptr); }
    SetTimer(hwnd, kTimer, 1000, nullptr);

    MSG msg;
    while (GetMessageW(&msg, nullptr, 0, 0) > 0) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
    KillTimer(hwnd, kTimer);
    KillTimer(hwnd, kAddRetry);
    NOTIFYICONDATAW nid{};
    nid.cbSize = sizeof(nid);
    nid.hWnd   = hwnd;
    nid.uID    = kTrayUid;
    Shell_NotifyIconW(NIM_DELETE, &nid);
}

} // namespace ninfer::supervisor
