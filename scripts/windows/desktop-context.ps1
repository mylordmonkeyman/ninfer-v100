# MSIX applications can redirect LocalAppData and registry writes into their
# private storage. Run installation and management through the real desktop so
# their paths mean the same thing as they do when NInfer starts from the menu.
if (!('NInfer.DesktopContext' -as [type])) {
    Add-Type -TypeDefinition @'
using System;
using System.IO;
using System.Text;
using System.ComponentModel;
using System.Runtime.InteropServices;
namespace NInfer {
    public static class DesktopContext {
        [DllImport("kernel32.dll", CharSet=CharSet.Unicode)]
        static extern int GetCurrentPackageFullName(ref uint length, IntPtr name);
        [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true)]
        static extern uint GetFinalPathNameByHandle(IntPtr file, StringBuilder path, uint size, uint flags);
        public static bool IsPackaged() {
            uint length = 0;
            int result = GetCurrentPackageFullName(ref length, IntPtr.Zero);
            if (result == 15700) return false; // APPMODEL_ERROR_NO_PACKAGE
            if (result == 122 || result == 0) return true;
            throw new Win32Exception(result);
        }
        public static string PhysicalFile(string path) {
            using (var file = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete)) {
                var buffer = new StringBuilder(32768);
                uint size = GetFinalPathNameByHandle(file.SafeFileHandle.DangerousGetHandle(), buffer, (uint)buffer.Capacity, 0);
                if (size == 0 || size >= buffer.Capacity) throw new Win32Exception();
                string resolved = buffer.ToString();
                if (resolved.StartsWith(@"\\?\UNC\")) return @"\\" + resolved.Substring(8);
                return resolved.StartsWith(@"\\?\") ? resolved.Substring(4) : resolved;
            }
        }
    }
}
'@
}

function Test-NInferRedirectedContext {
    if ([NInfer.DesktopContext]::IsPackaged()) { return $true }
    # A desktop child can inherit MSIX filesystem redirection without having a
    # package identity of its own. Inspect an actual write, not just identity.
    $probe = Join-Path $env:LOCALAPPDATA ('ninfer-context-' + [Guid]::NewGuid().ToString('N') + '.tmp')
    try {
        [IO.File]::WriteAllText($probe, '')
        $physical = [NInfer.DesktopContext]::PhysicalFile($probe)
        $packages = (Join-Path $env:LOCALAPPDATA 'Packages') + [IO.Path]::DirectorySeparatorChar
        return $physical.StartsWith($packages, [StringComparison]::OrdinalIgnoreCase)
    } finally { if (Test-Path -LiteralPath $probe) { Remove-Item -LiteralPath $probe } }
}

function Invoke-NInferOutsidePackage([string]$Script, [hashtable]$Parameters) {
    if (!(Test-NInferRedirectedContext)) { return $false }
    $id = 'ninfer-desktop-' + [Guid]::NewGuid().ToString('N')
    $scratch = Join-Path ([IO.Path]::GetTempPath()) $id
    New-Item -ItemType Directory -Path $scratch | Out-Null
    $payloadPath = Join-Path $scratch 'payload.json'
    @{script=[NInfer.DesktopContext]::PhysicalFile($Script); parameters=$Parameters} |
        ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $payloadPath -Encoding UTF8
    $physicalRoot = Split-Path -Parent ([NInfer.DesktopContext]::PhysicalFile($payloadPath))
    $runner = Join-Path $physicalRoot 'run.ps1'
    @'
$ErrorActionPreference = 'Stop'
try {
    $payload = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'payload.json') -Raw | ConvertFrom-Json
    $arguments = @{}
    foreach ($entry in $payload.parameters.PSObject.Properties) { $arguments[$entry.Name] = $entry.Value }
    & $payload.script @arguments *>&1 | Out-File -LiteralPath (Join-Path $PSScriptRoot 'output.log') -Encoding UTF8
    if (!$?) { throw 'Desktop operation failed.' }
    @{ok=$true} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'result.json')
} catch {
    @{ok=$false; error=$_.Exception.Message} | ConvertTo-Json |
        Set-Content -LiteralPath (Join-Path $PSScriptRoot 'result.json')
}
'@ | Set-Content -LiteralPath $runner -Encoding UTF8
    $shell = New-Object -ComObject WScript.Shell
    $shortcutPath = Join-Path $physicalRoot 'run.lnk'
    $shortcut = $shell.CreateShortcut($shortcutPath)
    $shortcut.TargetPath = "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe"
    $shortcut.Arguments = '-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File "' + $runner + '"'
    $shortcut.WorkingDirectory = $physicalRoot
    $shortcut.WindowStyle = 7
    $shortcut.Save()
    Write-Host 'Running through the Windows desktop, outside packaged app storage.'
    Start-Process -FilePath "$env:SystemRoot\explorer.exe" -ArgumentList ('"' + $shortcutPath + '"') -WindowStyle Hidden
    $resultPath = Join-Path $physicalRoot 'result.json'
    $deadline = (Get-Date).AddMinutes(3)
    while (!(Test-Path -LiteralPath $resultPath)) {
        if ((Get-Date) -gt $deadline) { throw "Desktop operation timed out; details remain in $physicalRoot" }
        Start-Sleep -Milliseconds 250
    }
    $result = Get-Content -LiteralPath $resultPath -Raw | ConvertFrom-Json
    $output = Join-Path $physicalRoot 'output.log'
    if (Test-Path -LiteralPath $output) { Get-Content -LiteralPath $output | ForEach-Object { Write-Host $_ } }
    if (!$result.ok) { throw "Desktop operation failed: $($result.error). Details: $physicalRoot" }
    if ((Split-Path -Leaf $physicalRoot) -ne $id -or
        ((Get-Item -LiteralPath $physicalRoot).Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        throw "Unexpected setup scratch directory: $physicalRoot"
    }
    Remove-Item -LiteralPath $physicalRoot -Recurse -Force
    return $true
}
