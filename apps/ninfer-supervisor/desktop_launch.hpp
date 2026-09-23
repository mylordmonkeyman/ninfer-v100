#pragma once

#include <windows.h>
#include <exdisp.h>
#include <shldisp.h>
#include <shlguid.h>
#include <shlobj.h>
#include <servprov.h>
#include <wrl/client.h>

#include <stdexcept>
#include <string>

namespace ninfer::supervisor {

// Ask the running desktop's IShellDispatch2 to launch us. ShellExecute in this
// process (or Start-Process in an agent terminal) can inherit its lifetime job.
// The desktop broker creates the tray app outside the caller's process tree.
inline void launch_from_desktop(const std::wstring& executable, const std::wstring& arguments,
                                const std::wstring& workdir) {
    const HRESULT initialized = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    if (FAILED(initialized)) { throw std::runtime_error("cannot initialize desktop launcher COM"); }
    const HRESULT result = [&]() -> HRESULT {
        using Microsoft::WRL::ComPtr;
        ComPtr<IShellWindows> windows;
        HRESULT hr = CoCreateInstance(CLSID_ShellWindows, nullptr, CLSCTX_LOCAL_SERVER,
                                      IID_PPV_ARGS(&windows));
        if (FAILED(hr)) return hr;
        VARIANT location{};
        location.vt = VT_I4;
        location.lVal = CSIDL_DESKTOP;
        VARIANT empty{};
        long hwnd = 0;
        ComPtr<IDispatch> desktop;
        hr = windows->FindWindowSW(&location, &empty, SWC_DESKTOP, &hwnd,
                                   SWFO_NEEDDISPATCH, &desktop);
        if (FAILED(hr) || !desktop) return FAILED(hr) ? hr : E_FAIL;
        ComPtr<IServiceProvider> provider;
        if (FAILED(hr = desktop.As(&provider))) return hr;
        ComPtr<IShellBrowser> browser;
        hr = provider->QueryService(SID_STopLevelBrowser, IID_PPV_ARGS(&browser));
        if (FAILED(hr)) return hr;
        ComPtr<IShellView> view;
        if (FAILED(hr = browser->QueryActiveShellView(&view))) return hr;
        ComPtr<IDispatch> background;
        if (FAILED(hr = view->GetItemObject(SVGIO_BACKGROUND, IID_PPV_ARGS(&background)))) return hr;
        ComPtr<IShellFolderViewDual> folder;
        if (FAILED(hr = background.As(&folder))) return hr;
        ComPtr<IDispatch> application;
        if (FAILED(hr = folder->get_Application(&application))) return hr;
        ComPtr<IShellDispatch2> shell;
        if (FAILED(hr = application.As(&shell))) return hr;
        BSTR file = SysAllocString(executable.c_str());
        VARIANT args{}, directory{}, verb{}, show{};
        args.vt = directory.vt = verb.vt = VT_BSTR;
        args.bstrVal = SysAllocString(arguments.c_str());
        directory.bstrVal = SysAllocString(workdir.c_str());
        verb.bstrVal = SysAllocString(L"open");
        show.vt = VT_I4;
        show.lVal = SW_SHOWNORMAL;
        hr = file && args.bstrVal && directory.bstrVal && verb.bstrVal
                 ? shell->ShellExecute(file, args, directory, verb, show) : E_OUTOFMEMORY;
        SysFreeString(file);
        VariantClear(&args);
        VariantClear(&directory);
        VariantClear(&verb);
        return hr;
    }();
    CoUninitialize();
    if (FAILED(result)) {
        throw std::runtime_error("Windows desktop launch failed (HRESULT " +
                                 std::to_string(static_cast<unsigned long>(result)) + ")");
    }
}

} // namespace ninfer::supervisor
