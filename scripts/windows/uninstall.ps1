$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'desktop-context.ps1')
if (Invoke-NInferOutsidePackage -Script $PSCommandPath -Parameters @{}) { return }
$installDir = [IO.Path]::GetFullPath($PSScriptRoot)
$manifest = Get-Content -LiteralPath (Join-Path $installDir 'installation.json') -Raw | ConvertFrom-Json
$key = 'HKCU:/Software/Microsoft/Windows/CurrentVersion/Uninstall/NInfer'
$registered = Get-ItemProperty -Path $key
if ($manifest.product -ne 'NInfer' -or $registered.InstallLocation -ne $installDir -or
    ((Get-Item -LiteralPath $installDir).Attributes -band [IO.FileAttributes]::ReparsePoint)) {
    throw 'This directory is not the registered NInfer installation.'
}
$appExe = Join-Path $installDir 'bin/ninfer-supervisor.exe'
foreach ($process in @(Get-CimInstance Win32_Process -Filter "Name='ninfer-supervisor.exe'" |
    Where-Object { $_.ExecutablePath -eq $appExe })) {
    Stop-Process -Id $process.ProcessId -Force
    Wait-Process -Id $process.ProcessId -Timeout 20 -ErrorAction SilentlyContinue
}
$runKey = 'HKCU:/Software/Microsoft/Windows/CurrentVersion/Run'
foreach ($process in @(Get-CimInstance Win32_Process -Filter "Name='ninfer-serve.exe'" |
    Where-Object { $_.ExecutablePath -eq (Join-Path $installDir 'bin/ninfer-serve.exe') })) {
    Stop-Process -Id $process.ProcessId -Force -ErrorAction SilentlyContinue
    Wait-Process -Id $process.ProcessId -Timeout 20 -ErrorAction SilentlyContinue
}
$run = Get-ItemProperty -Path $runKey -Name NInferSupervisor -ErrorAction SilentlyContinue
if ($run -and $run.NInferSupervisor.StartsWith('"' + $appExe + '"', [StringComparison]::OrdinalIgnoreCase)) {
    Remove-ItemProperty -Path $runKey -Name NInferSupervisor
}
$menu = [IO.Path]::GetFullPath((Join-Path $env:APPDATA 'Microsoft/Windows/Start Menu/Programs/NInfer'))
if ($manifest.menu_dir -ne $menu) { throw 'Unexpected Start menu directory.' }
if (Test-Path -LiteralPath $menu) { Remove-Item -LiteralPath $menu -Recurse -Force }
Remove-Item -LiteralPath $key -Recurse
# The verified app root contains only installed files. Keep configuration, logs and models.
Set-Location $env:LOCALAPPDATA
Remove-Item -LiteralPath $installDir -Recurse -Force
Write-Output "NInfer uninstalled. Configuration and logs remain in $($manifest.data_dir)."
