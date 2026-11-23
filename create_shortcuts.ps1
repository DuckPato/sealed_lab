# Creates (or refreshes) "sealed_lab" shortcuts on the Desktop and in the Start menu.
# They start the app with pythonw.exe: no console window. Called by run_windows.bat; safe to run again.
param([string]$AppDir = $PSScriptRoot)

$AppDir = (Resolve-Path $AppDir).Path.TrimEnd('\')
$pythonw = Join-Path $AppDir '.venv\Scripts\pythonw.exe'
$icon = Join-Path $AppDir 'sealed_lab\gui\assets\sealed_lab.ico'
if (-not (Test-Path $pythonw)) { Write-Host "pythonw.exe not found - run run_windows.bat first"; exit 1 }

$shell = New-Object -ComObject WScript.Shell
$targets = @(
  [Environment]::GetFolderPath('Desktop'),
  (Join-Path ([Environment]::GetFolderPath('Programs')) '')
)
foreach ($dir in $targets) {
  if (-not $dir -or -not (Test-Path $dir)) { continue }
  $lnk = $shell.CreateShortcut((Join-Path $dir 'sealed_lab.lnk'))
  $lnk.TargetPath = $pythonw
  $lnk.Arguments = '-m sealed_lab'
  $lnk.WorkingDirectory = $AppDir
  $lnk.IconLocation = "$icon,0"
  $lnk.Description = 'sealed_lab - 17lands Sealed deck builder and scorer'
  $lnk.Save()
}
