#pragma once

#include "dxgi_query.hpp"
#include <pdh.h>
#include <pdhmsg.h>
#include <tlhelp32.h>
#include <nlohmann/json.hpp>
#include <map>
#include <vector>
#include <cstdio>

namespace ninfer::supervisor {

// Windows owns per-process memory accounting under WDDM. Keep a PDH query
// open instead of launching a command for each sample. Match the DXGI adapter
// LUID: a process can allocate on more than one graphics card.
class GpuProcessSource {
public:
    ~GpuProcessSource() { if (query_) PdhCloseQuery(query_); }
    nlohmann::json sample(const DxgiSnapshot& adapter) {
        nlohmann::json out{{"ok", false}, {"apps", nlohmann::json::array()}};
        if (!adapter.ok) { out["error"] = "Graphics card identification unavailable."; return out; }
        if (!query_) {
            if (PdhOpenQueryW(nullptr, 0, &query_) != ERROR_SUCCESS ||
                PdhAddEnglishCounterW(query_, L"\\GPU Process Memory(*)\\Dedicated Usage", 0,
                                      &counter_) != ERROR_SUCCESS) {
                if (query_) PdhCloseQuery(query_);
                query_ = nullptr;
                out["error"] = "Windows GPU memory counters are unavailable.";
                return out;
            }
        }
        if (PdhCollectQueryData(query_) != ERROR_SUCCESS) {
            out["error"] = "Windows GPU memory sample unavailable."; return out;
        }
        DWORD bytes = 0, count = 0;
        const auto size_status = PdhGetFormattedCounterArrayW(counter_, PDH_FMT_LARGE,
                                                              &bytes, &count, nullptr);
        if (size_status != PDH_MORE_DATA) {
            out["error"] = "Waiting for Windows GPU process counters."; return out;
        }
        std::vector<unsigned char> storage(bytes);
        auto* items = reinterpret_cast<PDH_FMT_COUNTERVALUE_ITEM_W*>(storage.data());
        if (PdhGetFormattedCounterArrayW(counter_, PDH_FMT_LARGE, &bytes, &count, items) != ERROR_SUCCESS) {
            out["error"] = "GPU process list changed; retrying."; return out;
        }
        std::map<DWORD, std::uint64_t> processes;
        for (DWORD i = 0; i < count; ++i) {
            unsigned pid = 0, high = 0, low = 0, physical = 0;
            const auto& value = items[i].FmtValue;
            if ((value.CStatus != PDH_CSTATUS_VALID_DATA && value.CStatus != PDH_CSTATUS_NEW_DATA) ||
                value.largeValue <= 0 ||
                swscanf_s(items[i].szName, L"pid_%u_luid_0x%x_0x%x_phys_%u", &pid, &high, &low, &physical) != 4 ||
                high != static_cast<unsigned>(adapter.adapter_luid.HighPart) ||
                low != adapter.adapter_luid.LowPart) continue;
            processes[pid] += static_cast<std::uint64_t>(value.largeValue);
        }
        std::map<DWORD, std::string> names;
        HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
        if (snapshot != INVALID_HANDLE_VALUE) {
            PROCESSENTRY32W entry{}; entry.dwSize = sizeof(entry);
            if (Process32FirstW(snapshot, &entry)) do {
                char utf8[2048]{};
                if (WideCharToMultiByte(CP_UTF8, 0, entry.szExeFile, -1, utf8, sizeof(utf8), nullptr, nullptr))
                    names[entry.th32ProcessID] = utf8;
            } while (Process32NextW(snapshot, &entry));
            CloseHandle(snapshot);
        }
        for (const auto& [pid, usage] : processes) {
            const std::string name = names.count(pid) ? names.at(pid) : "Process " + std::to_string(pid);
            out["apps"].push_back({{"pid", pid}, {"name", name}, {"dedicated_bytes", usage}});
        }
        out["ok"] = true;
        return out;
    }
private:
    PDH_HQUERY query_ = nullptr;
    PDH_HCOUNTER counter_ = nullptr;
};
} // namespace ninfer::supervisor
