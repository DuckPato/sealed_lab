#!/usr/bin/env bash
# Applies a staged (already signature-checked) sealed_lab update - the Linux twin of apply_update.ps1: waits for the app to exit, backs
# the current files up, copies the new ones in, syncs the environment with uv, checks that the new version starts (imports, reports its
# version) and only then keeps it.  Any failure restores the backup exactly (files and environment) and the previous version is started
# again, so a failed update never leaves a mix of two versions.  Started (detached) by the app's "Update now"; the copy used is the NEW
# version's script.  Everything is logged to <app home>/logs/update.log.
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
backup="$home_dir/update/previous"
excludes=(--exclude=./.venv --exclude=./.uv --exclude=.git --exclude=__pycache__)

log_line() {
  printf '%s  %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$1" >> "$log"
  echo "$1"
}

show_error() {
  if [ "$no_dialog" = 1 ]; then return 0; fi
  local msg="$1

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

find_uv() {
  local c
  for c in "$app_dir/.uv/uv" "$(command -v uv 2>/dev/null)" "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv"; do
    if [ -n "$c" ] && [ -x "$c" ]; then echo "$c"; return 0; fi
  done
  return 1
}

# copy tree $1 into $2 (overwrite, no deletion) without .venv / .uv / .git / __pycache__ and without the package folder
copy_top() {
  (cd "$1" && tar "${excludes[@]}" --exclude=./sealed_lab -cf - .) | (cd "$2" && tar -xf -)
}

# the package folder of $1 replaces the one of $2 entirely (stale modules of the other version go)
copy_pkg() {
  rm -rf "$2/sealed_lab.new" && mkdir -p "$2/sealed_lab.new" &&
    (cd "$1/sealed_lab" && tar --exclude=__pycache__ -cf - .) | (cd "$2/sealed_lab.new" && tar -xf -) &&
    rm -rf "$2/sealed_lab" && mv "$2/sealed_lab.new" "$2/sealed_lab"
}

sync_env() {
  log_line "Installing dependencies (uv sync)..."
  (cd "$app_dir" && "$1" sync --frozen --inexact 2>&1) | while IFS= read -r line; do log_line "  $line"; done
  return "${PIPESTATUS[0]}"
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
    log_line "ERROR: $p is a git checkout; update it with git instead."
    show_error "The sealed_lab update did not start: $p is a git checkout."
    exit 1
  fi
  parent="$(dirname "$p")"
  if [ "$parent" = "$p" ]; then break; fi
  p="$parent"
done

err=""
copied=0
added=()

run_update() {
  if [ "$wait_pid" -gt 0 ] 2>/dev/null; then
    log_line "Waiting for sealed_lab (pid $wait_pid) to close..."
    local i
    for ((i = 0; i < 600; i++)); do
      kill -0 "$wait_pid" 2>/dev/null || break
      sleep 0.1
    done
    if kill -0 "$wait_pid" 2>/dev/null; then err="sealed_lab did not close within 60 seconds."; return 1; fi
  fi
  uv_path="$(find_uv)" || { err="uv was not found; run run_linux.sh in the app folder to finish the update."; return 1; }

  log_line "Backing up the current version to $backup..."
  rm -rf "$backup" && mkdir -p "$backup" || { err="could not create $backup."; return 1; }
  (cd "$app_dir" && tar "${excludes[@]}" -cf - .) | (cd "$backup" && tar -xf -) || { err="the backup failed."; return 1; }
  local rel
  while IFS= read -r -d '' rel; do
    rel="${rel#./}"
    case "$rel" in sealed_lab/*) continue ;; esac
    [ -e "$backup/$rel" ] || added+=("$rel")
  done < <(cd "$source_dir" && find . -type f -print0)

  log_line "Copying the new files..."
  copied=1
  copy_top "$source_dir" "$app_dir" && copy_pkg "$source_dir" "$app_dir" || { err="copying the files failed."; return 1; }
  sync_env "$uv_path" || { err="uv sync failed."; return 1; }

  log_line "Checking that the new version starts..."
  local out rc
  out="$(cd "$app_dir" && .venv/bin/python -c 'import sealed_lab, sealed_lab.updater, sealed_lab.gui.app; print(sealed_lab.__version__)' 2>&1)"
  rc=$?
  while IFS= read -r line; do log_line "  $line"; done <<< "$out"
  if [ "$rc" -ne 0 ]; then err="the new version does not start (exit code $rc)."; return 1; fi
  if [ "$(printf '%s\n' "$out" | tail -n 1)" != "$version" ]; then err="the new version reports $(printf '%s\n' "$out" | tail -n 1) instead of $version."; return 1; fi

  if [ "$no_shortcuts" = 0 ] && [ -f "$app_dir/create_shortcuts.sh" ]; then
    bash "$app_dir/create_shortcuts.sh" "$app_dir" 2>&1 | while IFS= read -r line; do log_line "  $line"; done
  fi
  log_line "Update to $version done."
  return 0
}

restore() {
  log_line "Restoring the previous version..."
  copy_top "$backup" "$app_dir" && copy_pkg "$backup" "$app_dir" || return 1
  local rel
  for rel in "${added[@]}"; do rm -f -- "$app_dir/$rel"; done
  local u
  u="$(find_uv)" && sync_env "$u"
}

if run_update; then
  status=0
else
  status=1
  log_line "ERROR: $err"
  if [ "$copied" = 1 ]; then
    if restore; then
      log_line "The previous version was restored."
      show_error "The sealed_lab update to $version did not complete:
$err

The previous version was restored."
    else
      log_line "ERROR: the previous version could not be restored either."
      show_error "The sealed_lab update failed and the previous version could not be restored. Download sealed_lab again from its release page."
      exit 2
    fi
  else
    show_error "The sealed_lab update to $version did not complete:
$err

Nothing was changed."
  fi
fi
start_app
exit "$status"
