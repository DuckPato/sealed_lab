# Applies a staged (already signature-checked) sealed_lab update: waits for the app to exit, backs the current files up, copies the
# new ones in, syncs the environment with uv, checks that the new version starts (imports, reports its version) and only then keeps it.
# Any failure restores the backup exactly (files and environment) and the previous version is started again, so a failed update never
# leaves a mix of two versions.  Started (detached) by the app's "Update now"; the copy used is the NEW version's script.
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
$backup = Join-Path $home_ 'update\previous'
$Skip = @('.venv', '.uv', '.git', '__pycache__')       # never copied, backed up or removed

function Log([string]$m) {
  $line = '{0}  {1}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $m
  Add-Content -Path $log -Value $line -Encoding UTF8
  Write-Host $m
}

function Show-Error([string]$m) {
  if ($NoDialog) { return }
  try {
    Add-Type -AssemblyName System.Windows.Forms
    [System.Windows.Forms.MessageBox]::Show("$m`n`nDetails: $log", 'sealed_lab update') | Out-Null
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

function Invoke-Robocopy([string[]]$RcArgs) {
  # robocopy: exit codes 0-7 are success (files copied / extra files / ...), 8 and above are failures.  /IS /IT: copy every file even
  # when size and time match (two versions of a file can have both), so the destination is always exactly the source.
  $ErrorActionPreference = 'Continue'
  & robocopy @RcArgs /IS /IT /R:2 /W:1 /NFL /NDL /NJH /NJS /NP | ForEach-Object { if ($_) { Log "  $_" } }
  if ($LASTEXITCODE -ge 8) { throw "robocopy failed (exit code $LASTEXITCODE)." }
}

function Find-Uv {
  foreach ($c in @((Join-Path $AppDir '.uv\uv.exe'), (Get-Command uv -ErrorAction SilentlyContinue | ForEach-Object Source),
                   (Join-Path $env:USERPROFILE '.local\bin\uv.exe'))) {
    if ($c -and (Test-Path $c)) { return $c }
  }
  return $null
}

function Sync-Env([string]$uvPath) {
  Log 'Installing dependencies (uv sync)...'
  $ErrorActionPreference = 'Continue'
  Push-Location $AppDir
  try {
    & $uvPath sync --frozen --inexact 2>&1 | ForEach-Object { Log "  $_" }
    if ($LASTEXITCODE -ne 0) { throw "uv sync failed (exit code $LASTEXITCODE)." }
  } finally { Pop-Location }
}

function Test-NewVersion([string]$want) {
  # the new code must import (GUI and updater included) in the synced environment and report the version it was signed as
  $py = Join-Path $AppDir '.venv\Scripts\python.exe'
  $ErrorActionPreference = 'Continue'
  Push-Location $AppDir
  try {
    $out = & $py -c "import sealed_lab, sealed_lab.updater, sealed_lab.gui.app; print(sealed_lab.__version__)" 2>&1
    $code = $LASTEXITCODE
  } finally { Pop-Location }
  $out | ForEach-Object { Log "  $_" }
  if ($code -ne 0) { throw "the new version does not start (exit code $code)." }
  $got = ($out | Select-Object -Last 1).ToString().Trim()
  if ($got -ne $want) { throw "the new version reports $got instead of $want." }
}

function Copy-Release([string]$from) {
  # top-level files and folders except the package, then the package folder mirrored (stale modules of the old version go)
  Invoke-Robocopy (@($from, $AppDir, '/E', '/XD') + $Skip + @('sealed_lab'))
  Invoke-Robocopy (@((Join-Path $from 'sealed_lab'), (Join-Path $AppDir 'sealed_lab'), '/MIR', '/XD') + $Skip)
}

function Restore-Backup([string[]]$added) {
  Log 'Restoring the previous version...'
  Copy-Release $backup
  foreach ($rel in $added) {                                          # files the update added outside the package folder
    $p = Join-Path $AppDir $rel
    if (Test-Path -LiteralPath $p -PathType Leaf) { Remove-Item -LiteralPath $p -Force }
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
  if (Test-Path (Join-Path $p '.git')) {
    Log "ERROR: $p is a git checkout; update it with git instead."
    Show-Error "The sealed_lab update did not start: $p is a git checkout."
    exit 1
  }
}

$status = 0
$copied = $false
$added = @()
try {
  if ($WaitPid -gt 0) {
    Log "Waiting for sealed_lab (pid $WaitPid) to close..."
    $proc = Get-Process -Id $WaitPid -ErrorAction SilentlyContinue
    if ($proc -and -not $proc.WaitForExit(60000)) { throw 'sealed_lab did not close within 60 seconds.' }
  }
  $uv = Find-Uv
  if (-not $uv) { throw 'uv was not found; run run_windows.bat in the app folder to finish the update.' }

  Log "Backing up the current version to $backup..."
  if (Test-Path $backup) { Remove-Item -LiteralPath $backup -Recurse -Force }
  New-Item -ItemType Directory -Force $backup | Out-Null
  Invoke-Robocopy (@($AppDir, $backup, '/E', '/XD') + $Skip)
  $srcRoot = (Resolve-Path $Source).Path.TrimEnd('\')
  $added = @(Get-ChildItem -LiteralPath $srcRoot -Recurse -File -Force |
             ForEach-Object { $_.FullName.Substring($srcRoot.Length + 1) } |
             Where-Object { -not $_.StartsWith('sealed_lab\') -and -not (Test-Path -LiteralPath (Join-Path $backup $_)) })

  Log 'Copying the new files...'
  $copied = $true
  Copy-Release $Source
  Sync-Env $uv
  Log 'Checking that the new version starts...'
  Test-NewVersion $version

  $sc = Join-Path $AppDir 'create_shortcuts.ps1'
  if ((Test-Path $sc) -and -not $NoShortcuts) {
    $ErrorActionPreference = 'Continue'
    & powershell -NoProfile -ExecutionPolicy Bypass -File $sc -AppDir $AppDir 2>&1 | ForEach-Object { Log "  $_" }
    $ErrorActionPreference = 'Stop'
  }
  Log "Update to $version done."
} catch {
  $status = 1
  $why = $_.Exception.Message
  Log "ERROR: $why"
  if ($copied) {
    try {
      $ErrorActionPreference = 'Stop'
      Restore-Backup $added
      $uv = Find-Uv
      if ($uv) { Sync-Env $uv }
      Log 'The previous version was restored.'
      Show-Error "The sealed_lab update to $version did not complete:`n$why`n`nThe previous version was restored."
    } catch {
      Log "ERROR: the previous version could not be restored either: $($_.Exception.Message)"
      Show-Error "The sealed_lab update failed and the previous version could not be restored:`n$($_.Exception.Message)`n`nDownload sealed_lab again from its release page."
      exit 2
    }
  } else {
    Show-Error "The sealed_lab update to $version did not complete:`n$why`n`nNothing was changed."
  }
}

Start-App
exit $status
