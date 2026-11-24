@echo off
setlocal
cd /d "%~dp0"
if exist "%~dp0.uv\uv.exe" set "PATH=%~dp0.uv;%PATH%"
where uv >nul 2>nul
if errorlevel 1 (
  echo Installing uv, pinned version checked against its SHA-256...
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0get_uv.ps1" -Dest "%~dp0.uv"
  if errorlevel 1 exit /b 1
  set "PATH=%~dp0.uv;%PATH%"
)
rem .venv with the exact versions in uv.lock (--inexact: extra packages, e.g. a developer's pytest, are left alone)
uv sync --frozen --inexact
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -m sealed_lab cli %*
endlocal
