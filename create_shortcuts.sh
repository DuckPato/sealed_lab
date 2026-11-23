#!/usr/bin/env bash
# Creates (or refreshes) the app-menu entry "sealed_lab" (a .desktop file in ~/.local/share/applications).
# Called by run_linux.sh and apply_update.sh; safe to run again.   Usage: bash create_shortcuts.sh "<app dir>"
set -euo pipefail

app_dir="$(cd "${1:-$(dirname "${BASH_SOURCE[0]}")}" && pwd)"
python="$app_dir/.venv/bin/python"
icon="$app_dir/sealed_lab/gui/assets/sealed_lab.png"
if [ ! -x "$python" ]; then
  echo ".venv/bin/python not found - run run_linux.sh first" >&2
  exit 1
fi

# Desktop Entry spec: Exec arguments with special characters are double-quoted; inside quotes \ " ` $ are backslash-escaped,
# and a literal % is written %%.
q="$python"
q="${q//\\/\\\\}"
q="${q//\"/\\\"}"
q="${q//\`/\\\`}"
q="${q//\$/\\\$}"
q="${q//%/%%}"

dir="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
mkdir -p "$dir"
file="$dir/sealed_lab.desktop"
cat > "$file" <<EOF
[Desktop Entry]
Type=Application
Name=sealed_lab
Comment=17lands Sealed deck builder and scorer
Exec="$q" -m sealed_lab
Path=$app_dir
Icon=$icon
Terminal=false
Categories=Game;CardGame;
StartupWMClass=sealed_lab
EOF
chmod 644 "$file"
if command -v update-desktop-database >/dev/null 2>&1; then
  update-desktop-database "$dir" >/dev/null 2>&1 || true
fi
echo "App-menu entry written: $file"
