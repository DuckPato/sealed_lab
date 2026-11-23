@echo off
setlocal
cd /d "%~dp0"
where uv >nul 2>nul
if errorlevel 1 (
  echo Installing uv...
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  set "PATH=%USERPROFILE%\.local\bin;%LOCALAPPDATA%\uv\bin;%PATH%"
)
where uv >nul 2>nul
if errorlevel 1 (
  echo Could not find uv after installation. Please reopen this window and retry.
  pause
  exit /b 1
)
rem .venv with the exact versions in uv.lock (--inexact: extra packages, e.g. a developer's pytest, are left alone)
uv sync --frozen --inexact
if errorlevel 1 ( pause & exit /b 1 )
rem Desktop + Start-menu shortcuts that open the app without a console (refreshed every run, cheap)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0create_shortcuts.ps1" -AppDir "%~dp0." >nul 2>nul
rem Start the GUI detached with pythonw (no console) and close this window
start "" ".venv\Scripts\pythonw.exe" -m sealed_lab
endlocal
exit /b 0
