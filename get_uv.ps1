# Installs uv (the tool that sets up Python and the app's libraries from uv.lock) into -Dest: a PINNED version, downloaded from uv's
# GitHub releases and checked against the SHA-256 below before anything is unpacked or run.  Called by run_windows.bat when no uv is
# found.  Nothing outside -Dest is changed (no PATH or profile edits).  -Archive: use a local copy instead of downloading (still checked).
param(
  [Parameter(Mandatory = $true)][string]$Dest,
  [string]$Archive = ''
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$Version = '0.13.7'
$Sha256 = @{
  'x86_64'  = '0cdc5b38fe5a8168ed6a25814f5ba43c21d313dd3187314fccb37b7cec46e021'
  'aarch64' = '505d9e65e7692bac3ddf98c66754aa26c5050c409c184af30a881a0a72cc85e3'
}
$arch = if ($env:PROCESSOR_ARCHITECTURE -eq 'ARM64' -or $env:PROCESSOR_ARCHITEW6432 -eq 'ARM64') { 'aarch64' } else { 'x86_64' }
$name = "uv-$arch-pc-windows-msvc.zip"
$url = "https://github.com/astral-sh/uv/releases/download/$Version/$name"
$tmp = Join-Path ([IO.Path]::GetTempPath()) ('sealed_lab_uv_' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory $tmp | Out-Null
try {
  $zip = Join-Path $tmp $name
  if ($Archive) {
    Copy-Item -LiteralPath $Archive -Destination $zip
  } else {
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $zip
  }
  $got = (Get-FileHash -Algorithm SHA256 -LiteralPath $zip).Hash.ToLowerInvariant()
  if ($got -ne $Sha256[$arch]) {
    throw "the uv download does not match its pinned SHA-256 (got $got); nothing was installed."
  }
  Expand-Archive -LiteralPath $zip -DestinationPath (Join-Path $tmp 'x') -Force
  $exe = Get-ChildItem -LiteralPath (Join-Path $tmp 'x') -Recurse -Filter 'uv.exe' | Select-Object -First 1
  if (-not $exe) { throw 'uv.exe is not in the archive.' }
  New-Item -ItemType Directory -Force $Dest | Out-Null
  Copy-Item -LiteralPath $exe.FullName -Destination (Join-Path $Dest 'uv.exe') -Force
  Write-Host "uv $Version installed in $Dest (SHA-256 checked)."
  exit 0
} catch {
  Write-Host "ERROR: $($_.Exception.Message)"
  exit 1
} finally {
  Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
}
