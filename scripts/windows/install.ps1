[CmdletBinding()]
param(
    [string]$BuildDir = (Join-Path $PSScriptRoot '../../build-win'),
    [string]$ConfigPath,
    [string]$InstallDir = (Join-Path $env:LOCALAPPDATA 'Programs/NInfer'),
    [string]$DataDir = (Join-Path $env:LOCALAPPDATA 'NInfer'),
    [switch]$NoStart
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'desktop-context.ps1')
$desktopParameters = @{
    BuildDir=[IO.Path]::GetFullPath($BuildDir); InstallDir=[IO.Path]::GetFullPath($InstallDir);
    DataDir=[IO.Path]::GetFullPath($DataDir); NoStart=[bool]$NoStart
}
if (Test-NInferRedirectedContext) {
    # Migrate a config from a previous redirected installation if the desktop
    # has no config yet. Existing desktop configuration always takes precedence.
    $sourceConfig = if ($ConfigPath) { $ConfigPath } else { Join-Path $DataDir 'supervisor.json' }
    if (Test-Path -LiteralPath $sourceConfig) {
        $desktopParameters.ConfigPath = [NInfer.DesktopContext]::PhysicalFile($sourceConfig)
    }
    if (Invoke-NInferOutsidePackage -Script $PSCommandPath -Parameters $desktopParameters) { return }
}
$InstallDir = [IO.Path]::GetFullPath($InstallDir)
$DataDir = [IO.Path]::GetFullPath($DataDir)
if ($DataDir.Equals($InstallDir, [StringComparison]::OrdinalIgnoreCase) -or
    $DataDir.StartsWith($InstallDir.TrimEnd('\','/') + [IO.Path]::DirectorySeparatorChar,
                       [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Keep the data directory outside the application directory so uninstall preserves it.'
}
$binDir = Join-Path $InstallDir 'bin'
$configFile = Join-Path $DataDir 'supervisor.json'
$manifestFile = Join-Path $InstallDir 'installation.json'
$runKey = 'HKCU:/Software/Microsoft/Windows/CurrentVersion/Run'
$uninstallKey = 'HKCU:/Software/Microsoft/Windows/CurrentVersion/Uninstall/NInfer'
$menuDir = Join-Path $env:APPDATA 'Microsoft/Windows/Start Menu/Programs/NInfer'
$packager = Join-Path $BuildDir 'windows-package-Release.cmake'
if (!(Test-Path -LiteralPath $packager)) { throw 'Build the Release applications before installing.' }
if ((Test-Path -LiteralPath $binDir) -and !(Test-Path -LiteralPath $manifestFile)) {
    throw "Install directory is not an existing NInfer installation: $InstallDir"
}
$updating = Test-Path -LiteralPath $manifestFile
if ($updating) {
    $previous = Get-Content -LiteralPath $manifestFile -Raw | ConvertFrom-Json
    if ($previous.product -ne 'NInfer' -or $previous.data_dir -ne $DataDir) {
        throw 'Existing installation has a different data directory.'
    }
}
$firstConfiguration = !(Test-Path -LiteralPath $configFile)
if ($firstConfiguration -and !$ConfigPath) {
    throw 'First installation requires -ConfigPath pointing to a configured Supervisor JSON file.'
}

# Stage and validate the complete runtime before stopping an installed app.
$stageDir = Join-Path $InstallDir ('staging-' + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $stageDir -Force | Out-Null
$savedErrorPreference = $ErrorActionPreference
try {
    # Windows PowerShell 5 reports native stderr as errors when output is
    # captured. Collect it and use CMake's exit status, retaining the diagnostic.
    $ErrorActionPreference = 'Continue'
    $packaging = & cmake "-DDESTINATION=$stageDir" -P $packager 2>&1
    $packageExit = $LASTEXITCODE
} finally { $ErrorActionPreference = $savedErrorPreference }
$packaging | ForEach-Object { Write-Output $_.ToString() }
if ($packageExit -ne 0) { throw "Runtime packaging failed; staged files are in $stageDir" }
$savedPath = $env:PATH
try {
    $env:PATH = "$env:SystemRoot/System32;$env:SystemRoot"
    foreach ($name in @('ninfer-supervisor.exe', 'ninfer-serve.exe', 'ninfer.exe')) {
        $process = Start-Process -FilePath (Join-Path $stageDir $name) -ArgumentList '--help' `
            -WorkingDirectory $stageDir -WindowStyle Hidden -PassThru -Wait `
            -RedirectStandardOutput (Join-Path $stageDir 'validation-out.log') `
            -RedirectStandardError (Join-Path $stageDir 'validation-error.log')
        if ($process.ExitCode -ne 0) { throw "$name failed runtime dependency validation: $($process.ExitCode)" }
    }
} finally { $env:PATH = $savedPath }
Remove-Item -LiteralPath (Join-Path $stageDir 'validation-out.log'),(Join-Path $stageDir 'validation-error.log')

New-Item -ItemType Directory -Path $DataDir,(Join-Path $DataDir 'logs'),$menuDir -Force | Out-Null
if (!(Test-Path -LiteralPath $configFile)) {
    $source = (Resolve-Path -LiteralPath $ConfigPath).Path
    $cfg = Get-Content -LiteralPath $source -Raw | ConvertFrom-Json
    $sourceDir = Split-Path -Parent $source
    $sourceWorkdir = if ($cfg.engine.workdir) { $cfg.engine.workdir } else { $sourceDir }
    if (![IO.Path]::IsPathRooted($sourceWorkdir)) { $sourceWorkdir = Join-Path $sourceDir $sourceWorkdir }
    function Absolute-SourcePath([string]$Value) {
        if ([IO.Path]::IsPathRooted($Value)) { return $Value }
        return [IO.Path]::GetFullPath((Join-Path $sourceWorkdir $Value))
    }
    function Installed-Arguments($Values) {
        $result = @($Values)
        for ($i = 0; $i -lt $result.Count - 1; $i++) {
            if ($result[$i] -eq '--request-log-jsonl') { $result[$i + 1] = Join-Path $DataDir 'logs/requests.jsonl' }
            if ($result[$i] -eq '--api-key-file') { $result[$i + 1] = Absolute-SourcePath $result[$i + 1] }
        }
        return ,$result
    }
    $cfg.engine.args = Installed-Arguments $cfg.engine.args
    if ($cfg.engine.args.Count) { $cfg.engine.args[0] = Absolute-SourcePath $cfg.engine.args[0] }
    foreach ($model in $cfg.models) {
        $model.artifact = Absolute-SourcePath $model.artifact
        $model.args = Installed-Arguments $model.args
    }
    if ($cfg.engine.api_key_file) { $cfg.engine.api_key_file = Absolute-SourcePath $cfg.engine.api_key_file }
    $cfg.engine.executable = Join-Path $binDir 'ninfer-serve.exe'
    $cfg.engine.workdir = $DataDir
    $cfg.engine.request_log = Join-Path $DataDir 'logs/requests.jsonl'
    $cfg.supervisor.logs_dir = Join-Path $DataDir 'logs'
    # The installer sets the initial login preference; the tray owns it thereafter.
    $cfg.supervisor.run_at_login = $false
    $cfg | ConvertTo-Json -Depth 40 | Set-Content -LiteralPath $configFile -Encoding UTF8
    $sourcePrefs = [IO.Path]::ChangeExtension($source, 'tray.json')
    if (Test-Path -LiteralPath $sourcePrefs) {
        Copy-Item -LiteralPath $sourcePrefs -Destination (Join-Path $DataDir 'supervisor.tray.json')
    }
}

# Match exact installed paths. Other configurations and test engines are not ours.
$appExe = Join-Path $binDir 'ninfer-supervisor.exe'
$installedProcesses = @(Get-CimInstance Win32_Process -Filter "Name='ninfer-supervisor.exe'" |
    Where-Object { $_.ExecutablePath -eq $appExe })
foreach ($process in $installedProcesses) {
    Stop-Process -Id $process.ProcessId -Force
    Wait-Process -Id $process.ProcessId -Timeout 20 -ErrorAction SilentlyContinue
}
foreach ($process in @(Get-CimInstance Win32_Process -Filter "Name='ninfer-serve.exe'" |
    Where-Object { $_.ExecutablePath -eq (Join-Path $binDir 'ninfer-serve.exe') })) {
    Stop-Process -Id $process.ProcessId -Force -ErrorAction SilentlyContinue
    Wait-Process -Id $process.ProcessId -Timeout 20 -ErrorAction SilentlyContinue
}
$backupDir = Join-Path $InstallDir 'previous-bin'
foreach ($target in @($binDir, $backupDir, $stageDir)) {
    $resolved = [IO.Path]::GetFullPath($target)
    if (!(Split-Path -Parent $resolved).Equals($InstallDir, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Application directory escaped installation root: $resolved"
    }
    if ((Test-Path -LiteralPath $resolved) -and
        ((Get-Item -LiteralPath $resolved).Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        throw "Application directory must not be a link: $resolved"
    }
}
if (Test-Path -LiteralPath $backupDir) { Remove-Item -LiteralPath $backupDir -Recurse -Force }
if (Test-Path -LiteralPath $binDir) { Move-Item -LiteralPath $binDir -Destination $backupDir }
Move-Item -LiteralPath $stageDir -Destination $binDir
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'uninstall.ps1') -Destination (Join-Path $InstallDir 'uninstall.ps1') -Force
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'desktop-context.ps1') -Destination (Join-Path $InstallDir 'desktop-context.ps1') -Force
@{product='NInfer'; data_dir=$DataDir; config=$configFile; menu_dir=$menuDir} |
    ConvertTo-Json | Set-Content -LiteralPath $manifestFile -Encoding UTF8

$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut((Join-Path $menuDir 'NInfer.lnk'))
$shortcut.TargetPath = $appExe
$shortcut.Arguments = '--config "' + $configFile + '" --background'
$shortcut.WorkingDirectory = $DataDir
$shortcut.Description = 'NInfer inference engine and Supervisor dashboard'
$shortcut.IconLocation = "$appExe,0"
$shortcut.Save()
$dashboard = $shell.CreateShortcut((Join-Path $menuDir 'Dashboard.url'))
$installedConfig = Get-Content -LiteralPath $configFile -Raw | ConvertFrom-Json
$dashboard.TargetPath = "http://127.0.0.1:$($installedConfig.supervisor.port)/"
$dashboard.Save()

# The Run key is shared with every other application's autostart entry, and
# New-Item -Force on an existing registry key deletes all of its values: it
# emptied the whole key on every update, taking this product's own entry with it.
if (!(Test-Path -LiteralPath $runKey)) { New-Item -Path $runKey | Out-Null }
$login = '"' + $appExe + '" --config "' + $configFile + '"'
if (!$updating -or $firstConfiguration -or (Get-ItemProperty -Path $runKey -Name NInferSupervisor -ErrorAction SilentlyContinue)) {
    New-ItemProperty -Path $runKey -Name NInferSupervisor -Value $login -PropertyType String -Force | Out-Null
}
New-Item -Path $uninstallKey -Force | Out-Null
$properties = @{
    DisplayName='NInfer'; DisplayVersion=(Get-Date -Format 'yyyy.MM.dd.HHmmss'); Publisher='NInfer';
    InstallLocation=$InstallDir; DisplayIcon="$appExe,0";
    UninstallString='powershell.exe -NoProfile -ExecutionPolicy Bypass -File "' + (Join-Path $InstallDir 'uninstall.ps1') + '"'
}
foreach ($entry in $properties.GetEnumerator()) {
    New-ItemProperty -Path $uninstallKey -Name $entry.Key -Value $entry.Value -PropertyType String -Force | Out-Null
}
foreach ($name in @('NoModify','NoRepair')) {
    New-ItemProperty -Path $uninstallKey -Name $name -Value 1 -PropertyType DWord -Force | Out-Null
}
# Invalidate cached artwork when an update replaces an executable at the same path.
if (!('NInfer.InstallShell' -as [type])) {
    Add-Type -TypeDefinition @'
namespace NInfer {
    public static class InstallShell {
        [System.Runtime.InteropServices.DllImport("shell32.dll")]
        public static extern void SHChangeNotify(int eventId, uint flags, System.IntPtr item1, System.IntPtr item2);
    }
}
'@
}
[NInfer.InstallShell]::SHChangeNotify(0x08000000, 0, [IntPtr]::Zero, [IntPtr]::Zero)
if (!$NoStart) {
    $process = Start-Process -FilePath $appExe -ArgumentList "--config `"$configFile`" --background" `
        -WorkingDirectory $DataDir -WindowStyle Hidden -PassThru -Wait
    if ($process.ExitCode -ne 0) { throw "Windows desktop launch failed: $($process.ExitCode)" }
}
Write-Output "Installed NInfer: $InstallDir"
Write-Output "Configuration: $configFile"
Write-Output "Dashboard: http://127.0.0.1:$($installedConfig.supervisor.port)/"
