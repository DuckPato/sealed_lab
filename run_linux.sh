#!/usr/bin/env bash
# Installs / starts sealed_lab on Ubuntu (and other Linux): gets uv if needed, builds .venv from uv.lock, adds an app-menu entry
# and starts the app detached.  Safe to run again.   Options: --no-start  --no-shortcuts
set -euo pipefail

start=1
shortcuts=1
for arg in "$@"; do
  case "$arg" in
    --no-start) start=0 ;;
    --no-shortcuts) shortcuts=0 ;;
    -h|--help) echo "usage: bash run_linux.sh [--no-start] [--no-shortcuts]"; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

cd "$(dirname "${BASH_SOURCE[0]}")"
app_dir="$PWD"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv (the tool that sets up Python and the libraries)..."
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- https://astral.sh/uv/install.sh | sh
  else
    echo "Need curl or wget to install uv: sudo apt install curl" >&2
    exit 1
  fi
  export PATH="$HOME/.local/bin:$PATH"
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "Could not find uv after installation. Open a new terminal and run this again." >&2
  exit 1
fi

echo "Setting up sealed_lab (first run takes a few minutes)..."
uv sync --frozen --inexact

if [ "$shortcuts" = 1 ]; then
  echo "Adding sealed_lab to the app menu..."
  bash create_shortcuts.sh "$app_dir" || echo "(could not create the app-menu entry; the app still works)"
fi

ldc="$(command -v ldconfig || echo /sbin/ldconfig)"      # /sbin is not on every user's PATH
if [ -x "$ldc" ] && ! "$ldc" -p 2>/dev/null | grep -q 'libxcb-cursor\.so\.0'; then
  echo "Hint: if the window does not open, Qt needs: sudo apt install libxcb-cursor0"
fi

if [ "$start" = 1 ]; then
  echo "Starting sealed_lab..."
  nohup setsid .venv/bin/python -m sealed_lab >/dev/null 2>&1 &
fi
echo "Done."
