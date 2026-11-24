"""The pinned uv bootstrap (get_uv.ps1 / get_uv.sh): one pinned version with a SHA-256 per platform, and an archive that does not
match is refused before anything is unpacked or installed.  No network: the tests hand the scripts a local (wrong) archive."""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parent.parent
PS1, SH = APP / "get_uv.ps1", APP / "get_uv.sh"


def test_both_scripts_pin_the_same_version_and_full_hashes():
    ps1, sh = PS1.read_text(encoding="utf-8"), SH.read_text(encoding="utf-8")
    v1 = re.search(r"\$Version = '([\d.]+)'", ps1).group(1)
    v2 = re.search(r'version="([\d.]+)"', sh).group(1)
    assert v1 == v2
    hashes = re.findall(r"\b[0-9a-f]{64}\b", ps1) + re.findall(r"\b[0-9a-f]{64}\b", sh)
    assert len(hashes) == 4 and len(set(hashes)) == 4                  # x86_64 + aarch64 for Windows and for Linux
    for text in (ps1, sh):
        assert "astral.sh/uv/install" not in text and "| iex" not in text and "| sh" not in text
        assert f"https://github.com/astral-sh/uv/releases/download/$" in text


def test_the_launchers_no_longer_pipe_an_installer_into_a_shell():
    for name in ("run_windows.bat", "run_windows_cli.bat", "run_linux.sh"):
        text = (APP / name).read_text(encoding="utf-8")
        assert "astral.sh" not in text and "| iex" not in text and "| sh" not in text, name
        assert "get_uv" in text, name


@pytest.mark.skipif(sys.platform != "win32", reason="Windows PowerShell")
def test_ps1_refuses_an_archive_that_is_not_the_pinned_one(tmp_path):
    fake = tmp_path / "uv.zip"
    shutil.make_archive(str(tmp_path / "uv"), "zip", root_dir=str(tmp_path), base_dir=".")  # any zip: the hash cannot match
    dest = tmp_path / "dest"
    r = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(PS1), "-Dest", str(dest), "-Archive", str(fake)],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 1 and "pinned SHA-256" in r.stdout, r.stdout + r.stderr
    assert not dest.exists()


@pytest.mark.skipif(not sys.platform.startswith("linux") or shutil.which("bash") is None, reason="Linux bash")
def test_sh_refuses_an_archive_that_is_not_the_pinned_one(tmp_path):
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "uv").write_text("#!/bin/sh\necho evil\n", encoding="utf-8")
    subprocess.run(["tar", "-czf", str(tmp_path / "uv.tar.gz"), "-C", str(tmp_path / "x"), "uv"], check=True)
    dest = tmp_path / "dest"
    r = subprocess.run(["bash", str(SH), str(dest), str(tmp_path / "uv.tar.gz")], capture_output=True, text=True, timeout=120)
    assert r.returncode == 1 and "pinned SHA-256" in r.stderr, r.stdout + r.stderr
    assert not dest.exists()
