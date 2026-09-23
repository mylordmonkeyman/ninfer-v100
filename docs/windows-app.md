# Windows app

For image/video input, use a media-enabled build. FFmpeg must include its C/C++ headers and
MSVC import libraries, as well as the matching runtime DLLs; an `ffmpeg.exe` command-line
installation alone is insufficient. A shared Windows build linked from
[FFmpeg's download page](https://ffmpeg.org/download.html) supplies those files. Build or install
a shared libcurl with headers and an MSVC import library; Schannel uses Windows' native TLS and
certificate store. Set the dependency locations for your machine:

```powershell
$env:PATH = "C:/deps/ffmpeg/bin;C:/deps/curl/bin;$env:PATH"
cmake -S . -B build-win -DNINFER_BUILD_MEDIA=ON `
  -DFFMPEG_ROOT=C:/deps/ffmpeg -DCURL_ROOT=C:/deps/curl
```

Add `--vision` to the configured model's engine arguments to load its Vision allocations.
The media-enabled binaries still accept text-only profiles without that flag. The installer
packages the discovered media runtime DLLs alongside the applications, so the installed app does
not depend on a developer shell's `PATH`. After updating an existing installation, change the
installed configuration through the dashboard or its JSON file and restart the engine.

Build and install for the current Windows user, without administrator access:

```powershell
cmake --build build-win --config Release --target ninfer ninfer-serve ninfer-supervisor -j
.\scripts\windows\install.ps1 -ConfigPath .\supervisor.local.json
```

The configuration must describe the model artifacts already present on this machine. Relative
artifact and API-key paths are resolved against its original working directory. Models are not
copied or downloaded. The installed engine works from the app's data directory; use absolute
paths for any other auxiliary resources in custom arguments.

| Item | Default location |
|---|---|
| Executables and runtime DLLs | `%LOCALAPPDATA%\Programs\NInfer\bin` |
| Active configuration | `%LOCALAPPDATA%\NInfer\supervisor.json` |
| Tray preferences | `%LOCALAPPDATA%\NInfer\supervisor.tray.json` |
| Engine and request logs | `%LOCALAPPDATA%\NInfer\logs` |
| Application shortcuts | Start menu → NInfer |

The installed configuration is the active authority. Changes to the original source JSON do not
alter the running installation. The dashboard saves model switches and settings to the installed
copy. `-InstallDir` and `-DataDir` select custom app/data directories; keep them separate.

The setup and management scripts detect packaged callers such as Codex and relaunch through the
Windows desktop. This prevents Windows from redirecting app data and registration into the
caller's private MSIX storage. A configuration from a previous redirected installation is used
only when the desktop installation has no configuration yet.

NInfer starts at sign-in and displays a tray icon. Closing a browser, terminal, editor, or coding
agent does not stop it. **Exit** in the tray closes Supervisor and its managed engine. Signing out
also closes the app; running before sign-in would require a separate Windows service design.
Toggle **Start at login** in the tray to change the sign-in preference.

Launch from the Start menu, or use the installed binary from automation:

```powershell
& "$env:LOCALAPPDATA\Programs\NInfer\bin\ninfer-supervisor.exe" `
  --config "$env:LOCALAPPDATA\NInfer\supervisor.json" --background
```

`--background` asks the running Explorer desktop to create the app. Directly spawning a GUI
executable from an agent terminal can inherit that terminal's process job, even with a hidden
window. The desktop launch requires an interactive Windows desktop; it fails explicitly if the
desktop broker is unavailable. See Microsoft's [desktop launch pattern](https://devblogs.microsoft.com/oldnewthing/20131118-00/?p=2643).

For local command-line operations:

```powershell
.\scripts\windows\manage.ps1 status
.\scripts\windows\manage.ps1 restart
.\scripts\windows\manage.ps1 logs
```

`stop` stops the managed engine while leaving the tray/dashboard available. `start` launches the
installed app if needed, or starts its engine. Dashboard controls provide the same engine actions.

## Request capacity

Settings → Request capacity sizes three related Engine startup values:

1. **KV capacity** is the shared token pool for every active request. Engine default omits the
   flag so the pool follows max context. Auto sizes the pool from free GPU memory at startup.
2. **Max context** is the longest one request may be. It cannot exceed an explicit pool; enlarge
   the pool first. Lowering the pool pulls max context down.
3. **Max concurrency** is how many requests may be Active at once. The control’s maximum of 8 is
   the Engine compile-time lane cap (`kMaximumConcurrency`): CUDA Graphs and kernels are built
   for exact batch `1..8`. It is not a GPU-memory estimate. Omitting it uses 1 lane.

An explicit pool must cover one full-length request and cannot exceed
`max concurrency × max context`. The live preview shows how many full-length requests the pool
can hold against the configured lane count. Extra lanes share the pool and must use shorter
requests. Save does not restart the engine; use Save & restart or Overview when the new
capacity should take effect.

To update, rebuild Release and rerun `install.ps1` without `-ConfigPath`. The installer validates
runtime dependencies before stopping the installed app, replaces the binaries, retains one previous
binary directory, and relaunches through Explorer. Configuration, logs and tray preferences remain.
The login preference remains as selected in the tray. `-NoStart` installs without launching.

Uninstall through **Settings → Apps → Installed apps → NInfer**, or run the installed
`uninstall.ps1`. Uninstall stops the installed processes and removes the app, shortcuts and its
login entry. Configuration, logs and model artifacts are retained.
