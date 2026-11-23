# Applies a staged sealed_lab update: waits for the app to exit, copies the new files over the install, syncs the environment
# with uv and starts the app again.  Started (detached) by the app's "Update now"; the copy used is the NEW version's script.
# Everything is logged to <app home>\logs\update.log.
param(
  [Parameter(Mandatory = $true)][string]$Source,
  [Parameter(Mandatory = $true)][string]$AppDir,
  [int]$WaitPid = 0,
  [switch]$NoShortcuts,          # (testing) leave the Desktop / Start-menu shortcuts alone
  [switch]$NoStart,              # (testing) do not start the app afterwards
  [switch]$NoDialog              # (testing) no message box on failure (the log has it)
)

$ErrorActionPreference = 'Stop'
$home_ = if ($env:SEALED_LAB_HOME) { $env:SEALED_LAB_HOME } else { Join-Path $env:LOCALAPPDATA 'sealed_lab' }
$logDir = Join-Path $home_ 'logs'
New-Item -ItemType Directory -Force $logDir | Out-Null
$log = Join-Path $logDir 'update.log'

function Log([string]$m) {
  $line = '{0}  {1}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $m
  Add-Content -Path $log -Value $line -Encoding UTF8
  Write-Host $m
}

function Fail([string]$m) {
  Log "ERROR: $m"
  if ($NoDialog) { return }
  try {
    Add-Type -AssemblyName System.Windows.Forms
    [System.Windows.Forms.MessageBox]::Show("The sealed_lab update did not complete:`n$m`n`nDetails: $log", 'sealed_lab update') | Out-Null
  } catch { }
}

function Start-App {
  if ($NoStart) { return }
  $pythonw = Join-Path $AppDir '.venv\Scripts\pythonw.exe'
  if (Test-Path $pythonw) {
    Log 'Starting sealed_lab...'
    Start-Process -FilePath $pythonw -ArgumentList '-m', 'sealed_lab' -WorkingDirectory $AppDir
  } else {
    Log 'No .venv found: run run_windows.bat in the app folder.'
  }
}

$version = Split-Path $Source -Leaf
$init = Join-Path $Source 'sealed_lab\__init__.py'
if (Test-Path $init) {
  $m = Select-String -Path $init -Pattern '^__version__\s*=\s*"([^"]+)"' | Select-Object -First 1
  if ($m) { $version = $m.Matches[0].Groups[1].Value }
}
Log "Updating sealed_lab to $version... (source $Source, app folder $AppDir)"

# a developer's git checkout is updated with git, never overwritten
for ($p = $AppDir; $p; $p = Split-Path $p -Parent) {
  if (Test-Path (Join-Path $p '.git')) { Fail "$p is a git checkout; update it with git instead."; exit 1 }
}

try {
  if ($WaitPid -gt 0) {
    Log "Waiting for sealed_lab (pid $WaitPid) to close..."
    $proc = Get-Process -Id $WaitPid -ErrorAction SilentlyContinue
    if ($proc -and -not $proc.WaitForExit(60000)) { throw 'sealed_lab did not close within 60 seconds.' }
  }

  Log 'Copying files...'
  # native tools write progress to stderr: with 'Stop' PowerShell 5.1 would turn the first such line into a terminating error
  $ErrorActionPreference = 'Continue'
  & robocopy $Source $AppDir /E /XD .venv .git __pycache__ /R:2 /W:1 /NFL /NDL /NJH /NJS /NP | ForEach-Object { if ($_) { Log $_ } }
  if ($LASTEXITCODE -ge 8) { throw "robocopy failed (exit code $LASTEXITCODE)." }

  $uv = Get-Command uv -ErrorAction SilentlyContinue
  $uvPath = if ($uv) { $uv.Source } else { Join-Path $env:USERPROFILE '.local\bin\uv.exe' }
  if (-not (Test-Path $uvPath)) { throw 'uv was not found; run run_windows.bat in the app folder to finish the update.' }
  Log 'Installing dependencies (uv sync)...'
  Push-Location $AppDir
  try {
    & $uvPath sync --frozen --inexact 2>&1 | ForEach-Object { Log "$_" }
    if ($LASTEXITCODE -ne 0) { throw "uv sync failed (exit code $LASTEXITCODE)." }
  } finally { Pop-Location }

  $sc = Join-Path $AppDir 'create_shortcuts.ps1'
  if ((Test-Path $sc) -and -not $NoShortcuts) {
    & powershell -NoProfile -ExecutionPolicy Bypass -File $sc -AppDir $AppDir 2>&1 | ForEach-Object { Log "$_" }
  }
  Log "Update to $version done."
} catch {
  Fail $_.Exception.Message
}

Start-App
