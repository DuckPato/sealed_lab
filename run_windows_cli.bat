@echo off
setlocal
cd /d "%~dp0"
where uv >nul 2>nul
if errorlevel 1 (
  echo Installing uv...
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  set "PATH=%USERPROFILE%\.local\bin;%LOCALAPPDATA%\uv\bin;%PATH%"
)
rem .venv with the exact versions in uv.lock (--inexact: extra packages, e.g. a developer's pytest, are left alone)
uv sync --frozen --inexact
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -m sealed_lab cli %*
endlocal
