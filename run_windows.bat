@echo off
setlocal
cd /d "%~dp0"
rem uv: the app's own pinned copy (.uv), else one already installed, else get the pinned one (SHA-256 checked, see get_uv.ps1)
if exist "%~dp0.uv\uv.exe" set "PATH=%~dp0.uv;%PATH%"
where uv >nul 2>nul
if errorlevel 1 (
  echo Installing uv, pinned version checked against its SHA-256...
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0get_uv.ps1" -Dest "%~dp0.uv"
  if errorlevel 1 ( pause & exit /b 1 )
  set "PATH=%~dp0.uv;%PATH%"
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
