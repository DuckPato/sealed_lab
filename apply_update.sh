#!/usr/bin/env bash
# Applies a staged sealed_lab update (Linux twin of apply_update.ps1): waits for the app to exit, copies the new files over the
# install, syncs the environment with uv and starts the app again.  Started (detached) by the app's "Update now"; the copy used is
# the NEW version's script.  Everything is logged to <app home>/logs/update.log.
#   bash apply_update.sh --source DIR --app-dir DIR [--wait-pid N] [--no-shortcuts] [--no-start] [--no-dialog]
set -uo pipefail

source_dir=""
app_dir=""
wait_pid=0
no_shortcuts=0
no_start=0
no_dialog=0
while [ $# -gt 0 ]; do
  case "$1" in
    --source) source_dir="${2:-}"; shift 2 ;;
    --app-dir) app_dir="${2:-}"; shift 2 ;;
    --wait-pid) wait_pid="${2:-0}"; shift 2 ;;
    --no-shortcuts) no_shortcuts=1; shift ;;
    --no-start) no_start=1; shift ;;
    --no-dialog) no_dialog=1; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
if [ -z "$source_dir" ] || [ -z "$app_dir" ]; then
  echo "usage: apply_update.sh --source DIR --app-dir DIR [--wait-pid N] [--no-shortcuts] [--no-start] [--no-dialog]" >&2
  exit 2
fi

home_dir="${SEALED_LAB_HOME:-$HOME/.sealed_lab}"
mkdir -p "$home_dir/logs"
log="$home_dir/logs/update.log"

log_line() {
  printf '%s  %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$1" >> "$log"
  echo "$1"
}

fail() {
  log_line "ERROR: $1"
  if [ "$no_dialog" = 1 ]; then return 0; fi
  local msg="The sealed_lab update did not complete:
$1

Details: $log"
  if command -v zenity >/dev/null 2>&1; then
    zenity --error --title="sealed_lab update" --text="$msg" >/dev/null 2>&1 || true
  elif command -v kdialog >/dev/null 2>&1; then
    kdialog --error "$msg" --title "sealed_lab update" >/dev/null 2>&1 || true
  elif command -v notify-send >/dev/null 2>&1; then
    notify-send "sealed_lab update" "$msg" >/dev/null 2>&1 || true
  fi
  return 0
}

start_app() {
  if [ "$no_start" = 1 ]; then return 0; fi
  if [ -x "$app_dir/.venv/bin/python" ]; then
    log_line "Starting sealed_lab..."
    (cd "$app_dir" && nohup setsid .venv/bin/python -m sealed_lab >/dev/null 2>&1 &)
  else
    log_line "No .venv found: run run_linux.sh in the app folder."
  fi
}

version="$(basename "$source_dir")"
init="$source_dir/sealed_lab/__init__.py"
if [ -f "$init" ]; then
  v="$(sed -n 's/^__version__[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p' "$init" | head -n 1)"
  if [ -n "$v" ]; then version="$v"; fi
fi
log_line "Updating sealed_lab to $version... (source $source_dir, app folder $app_dir)"

# a developer's git checkout is updated with git, never overwritten
p="$app_dir"
while :; do
  if [ -e "$p/.git" ]; then
    fail "$p is a git checkout; update it with git instead."
    exit 1
  fi
  parent="$(dirname "$p")"
  if [ "$parent" = "$p" ]; then break; fi
  p="$parent"
done

# the steps run in a function so that the first failure ends them and the app is still started afterwards
run_update() {
  if [ "$wait_pid" -gt 0 ] 2>/dev/null; then
    log_line "Waiting for sealed_lab (pid $wait_pid) to close..."
    local i
    for ((i = 0; i < 600; i++)); do
      kill -0 "$wait_pid" 2>/dev/null || break
      sleep 0.1
    done
    if kill -0 "$wait_pid" 2>/dev/null; then
      err="sealed_lab did not close within 60 seconds."; return 1
    fi
  fi

  log_line "Copying files..."
  mkdir -p "$app_dir" || { err="could not create $app_dir."; return 1; }
  if ! (cd "$source_dir" && tar --exclude=.venv --exclude=.git --exclude=__pycache__ -cf - .) | (cd "$app_dir" && tar -xf -); then
    err="copying the files failed."; return 1
  fi

  local uv_path=""
  if command -v uv >/dev/null 2>&1; then uv_path="$(command -v uv)"
  elif [ -x "$HOME/.local/bin/uv" ]; then uv_path="$HOME/.local/bin/uv"
  elif [ -x "$HOME/.cargo/bin/uv" ]; then uv_path="$HOME/.cargo/bin/uv"
  fi
  if [ -z "$uv_path" ]; then
    err="uv was not found; run run_linux.sh in the app folder to finish the update."; return 1
  fi
  log_line "Installing dependencies (uv sync)..."
  local rc
  (cd "$app_dir" && "$uv_path" sync --frozen --inexact 2>&1) | while IFS= read -r line; do log_line "$line"; done
  rc="${PIPESTATUS[0]}"
  if [ "$rc" -ne 0 ]; then
    err="uv sync failed (exit code $rc)."; return 1
  fi

  if [ "$no_shortcuts" = 0 ] && [ -f "$app_dir/create_shortcuts.sh" ]; then
    bash "$app_dir/create_shortcuts.sh" "$app_dir" 2>&1 | while IFS= read -r line; do log_line "$line"; done
  fi
  log_line "Update to $version done."
  return 0
}

err=""
if run_update; then status=0; else status=1; fail "$err"; fi
start_app
exit "$status"
