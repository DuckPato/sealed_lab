#!/usr/bin/env bash
# Installs uv (the tool that sets up Python and the app's libraries from uv.lock) into DEST: a PINNED version, downloaded from uv's
# GitHub releases and checked against the SHA-256 below before anything is unpacked or run.  Called by run_linux.sh when no uv is
# found.  Nothing outside DEST is changed (no PATH or profile edits).   Usage: bash get_uv.sh DEST [LOCAL_ARCHIVE]
set -euo pipefail

dest="${1:?usage: get_uv.sh DEST [LOCAL_ARCHIVE]}"
archive="${2:-}"
version="0.13.7"
case "$(uname -m)" in
  x86_64|amd64) arch=x86_64; want=c5221ff41b7892bfc223543883600125f6a47fe845093e0f5bdff4f359196d83 ;;
  aarch64|arm64) arch=aarch64; want=71279f8d3c67241e807b9d8db872d1bf541579fb9f08b7d1eb1682196d1e57dc ;;
  *) echo "ERROR: no pinned uv for this CPU ($(uname -m)); install uv yourself: https://docs.astral.sh/uv/" >&2; exit 1 ;;
esac
name="uv-$arch-unknown-linux-gnu.tar.gz"
url="https://github.com/astral-sh/uv/releases/download/$version/$name"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

if [ -n "$archive" ]; then
  cp -- "$archive" "$tmp/$name"
elif command -v curl >/dev/null 2>&1; then
  curl -fsSL --proto '=https' --tlsv1.2 -o "$tmp/$name" "$url"
elif command -v wget >/dev/null 2>&1; then
  wget -q --https-only -O "$tmp/$name" "$url"
else
  echo "ERROR: need curl or wget to download uv: sudo apt install curl" >&2
  exit 1
fi
got="$(sha256sum "$tmp/$name" | cut -d' ' -f1)"
if [ "$got" != "$want" ]; then
  echo "ERROR: the uv download does not match its pinned SHA-256 (got $got); nothing was installed." >&2
  exit 1
fi
mkdir -p "$tmp/x"
tar -xzf "$tmp/$name" -C "$tmp/x" --no-same-owner
bin="$(find "$tmp/x" -type f -name uv | head -n 1)"
if [ -z "$bin" ]; then
  echo "ERROR: uv is not in the archive." >&2
  exit 1
fi
mkdir -p "$dest"
install -m 755 "$bin" "$dest/uv"
echo "uv $version installed in $dest (SHA-256 checked)."
