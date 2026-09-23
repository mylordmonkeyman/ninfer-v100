param(
    [ValidateSet('status','start','stop','restart','logs')][string]$Action = 'status',
    [string]$DataDir = (Join-Path $env:LOCALAPPDATA 'NInfer')
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'desktop-context.ps1')
if (Invoke-NInferOutsidePackage -Script $PSCommandPath -Parameters @{
    Action=$Action; DataDir=[IO.Path]::GetFullPath($DataDir)
}) { return }
$config = Join-Path $DataDir 'supervisor.json'
$cfg = Get-Content -LiteralPath $config -Raw | ConvertFrom-Json
$url = "http://127.0.0.1:$($cfg.supervisor.port)"
if ($Action -eq 'logs') {
    Get-Content -LiteralPath (Join-Path $cfg.supervisor.logs_dir 'engine.log') -Tail 30
    return
}
$state = $null
try { $state = Invoke-RestMethod "$url/api/state" -TimeoutSec 5 } catch {}
if ($Action -eq 'status') {
    if (!$state) { Write-Output 'NInfer is stopped.'; return }
    [pscustomobject]@{Engine=$state.engine.state; PID=$state.engine.pid; Health=$state.health.status; Dashboard=$url}
} elseif (!$state -and $Action -in @('start','restart')) {
    $exe = Join-Path (Split-Path -Parent $cfg.engine.executable) 'ninfer-supervisor.exe'
    $process = Start-Process -FilePath $exe -ArgumentList "--config `"$config`" --background" `
        -WorkingDirectory $DataDir -WindowStyle Hidden -PassThru -Wait
    if ($process.ExitCode -ne 0) { throw "Windows launch failed: $($process.ExitCode)" }
} elseif ($state) {
    Invoke-RestMethod "$url/api/$Action" -Method Post -Headers @{'X-NInfer-Supervisor'='1'} -TimeoutSec 30
}
