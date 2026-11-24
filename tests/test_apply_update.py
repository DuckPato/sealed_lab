"""The update scripts for real (apply_update.ps1 on Windows, apply_update.sh on Linux): backup, copy, uv sync (a stub), the start check in
a real virtual environment, and the exact rollback when a step fails.  Skipped where the platform's script cannot run."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

IS_WIN = sys.platform == "win32"
IS_LINUX = sys.platform.startswith("linux") and shutil.which("bash") is not None
pytestmark = pytest.mark.skipif(not (IS_WIN or IS_LINUX), reason="needs Windows PowerShell or Linux bash")

APP = Path(__file__).resolve().parent.parent


def write_pkg(root: Path, version: str, modules: dict[str, str] | None = None, broken: bool = False) -> None:
    """A minimal sealed_lab package: what the start check imports (sealed_lab, .updater, .gui.app) plus `modules`."""
    pkg = root / "sealed_lab"
    (pkg / "gui").mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text(f'__version__ = "{version}"\n', encoding="utf-8")
    (pkg / "updater.py").write_text("", encoding="utf-8")
    (pkg / "gui" / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "gui" / "app.py").write_text("raise ImportError('broken build')\n" if broken else "", encoding="utf-8")
    for name, text in (modules or {}).items():
        (pkg / name).write_text(text, encoding="utf-8")


@pytest.fixture()
def world(tmp_path):
    """An installed 1.0.0 (with a real .venv) and a staged 2.0.0; a stub uv first on PATH that logs its calls and exits with
    the code in tmp/uv_exit (default 0), which goes back to 0 after one call unless tmp/uv_sticky exists."""
    app, src = tmp_path / "app", tmp_path / "staged" / "sealed_lab-2.0.0"
    write_pkg(app, "1.0.0", {"old_only.py": "OLD = 1\n", "shared.py": "V = 1\n"})
    (app / "pyproject.toml").write_text("old\n", encoding="utf-8")
    (app / "stale.txt").write_text("kept\n", encoding="utf-8")
    write_pkg(src, "2.0.0", {"new_mod.py": "NEW = 1\n", "shared.py": "V = 2\n"})
    (src / "pyproject.toml").write_text("new\n", encoding="utf-8")
    (src / "added.txt").write_text("new file\n", encoding="utf-8")
    r = subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(app / ".venv")], capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stderr
    (app / ".venv" / "marker").write_text("untouched\n", encoding="utf-8")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log, code, sticky = tmp_path / "uv_calls.txt", tmp_path / "uv_exit", tmp_path / "uv_sticky"
    code.write_text("0", encoding="utf-8")
    if IS_WIN:
        (bin_dir / "uv.cmd").write_text(f'@echo off\r\necho %*>>"{log}"\r\nset /p RC=<"{code}"\r\n'
                                        f'if not exist "{sticky}" (echo 0)>"{code}"\r\nexit /b %RC%\r\n', encoding="utf-8")
    else:
        (bin_dir / "uv").write_text(f'#!/bin/sh\necho "$@" >> "{log}"\nrc="$(cat "{code}")"\n[ -e "{sticky}" ] || echo 0 > "{code}"\n'
                                    f'exit "$rc"\n', encoding="utf-8")
        (bin_dir / "uv").chmod(0o755)
    home = tmp_path / "home"
    home.mkdir()
    env = dict(os.environ, PATH=f"{bin_dir}{os.pathsep}{os.environ['PATH']}", SEALED_LAB_HOME=str(tmp_path / "labhome"),
               HOME=str(home), USERPROFILE=str(home) if IS_WIN else os.environ.get("USERPROFILE", ""))
    return dict(app=app, src=src, env=env, uv_log=log, uv_exit=code, uv_sticky=sticky, tmp=tmp_path)


def run_apply(w) -> subprocess.CompletedProcess:
    if IS_WIN:
        cmd = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(APP / "apply_update.ps1"), "-Source", str(w["src"]),
               "-AppDir", str(w["app"]), "-NoStart", "-NoShortcuts", "-NoDialog"]
    else:
        cmd = ["bash", str(APP / "apply_update.sh"), "--source", str(w["src"]), "--app-dir", str(w["app"]), "--no-start", "--no-shortcuts",
               "--no-dialog"]
    return subprocess.run(cmd, env=w["env"], capture_output=True, text=True, timeout=300)


def update_log(w) -> str:
    return (w["tmp"] / "labhome" / "logs" / "update.log").read_text(encoding="utf-8-sig")


def version_of(root: Path) -> str:
    return (root / "sealed_lab" / "__init__.py").read_text(encoding="utf-8").split('"')[1]


def assert_old_version_intact(w) -> None:
    app = w["app"]
    assert version_of(app) == "1.0.0"
    assert (app / "sealed_lab" / "old_only.py").is_file() and not (app / "sealed_lab" / "new_mod.py").exists()
    assert (app / "sealed_lab" / "shared.py").read_text(encoding="utf-8") == "V = 1\n"
    assert (app / "pyproject.toml").read_text(encoding="utf-8") == "old\n"
    assert not (app / "added.txt").exists() and (app / "stale.txt").is_file()
    assert (app / ".venv" / "marker").read_text(encoding="utf-8") == "untouched\n"


def test_update_replaces_the_package_and_keeps_a_backup(world):
    r = run_apply(world)
    assert r.returncode == 0, r.stdout + r.stderr + update_log(world)
    app = world["app"]
    assert version_of(app) == "2.0.0" and (app / "sealed_lab" / "new_mod.py").is_file()
    assert not (app / "sealed_lab" / "old_only.py").exists()              # the package is mirrored: no stale module of 1.0.0
    assert (app / "sealed_lab" / "shared.py").read_text(encoding="utf-8") == "V = 2\n"
    assert (app / "added.txt").is_file() and (app / "stale.txt").is_file()
    assert (app / ".venv" / "marker").read_text(encoding="utf-8") == "untouched\n"
    assert "sync --frozen --inexact" in world["uv_log"].read_text(encoding="utf-8")
    log = update_log(world)
    assert "Checking that the new version starts" in log and "Update to 2.0.0 done." in log
    assert version_of(world["tmp"] / "labhome" / "update" / "previous") == "1.0.0"


def test_a_failed_uv_sync_restores_the_previous_version(world):
    world["uv_exit"].write_text("3", encoding="utf-8")
    r = run_apply(world)
    assert r.returncode == 1, r.stdout + r.stderr
    assert_old_version_intact(world)
    log = update_log(world)
    assert "uv sync failed" in log and "The previous version was restored." in log
    assert world["uv_log"].read_text(encoding="utf-8").count("sync --frozen --inexact") == 2       # the attempt + the restore


def test_when_even_the_restore_cannot_sync_it_says_so(world):
    world["uv_exit"].write_text("3", encoding="utf-8")
    world["uv_sticky"].write_text("", encoding="utf-8")
    r = run_apply(world)
    assert r.returncode == 2, r.stdout + r.stderr
    assert version_of(world["app"]) == "1.0.0"                          # the files are back all the same
    assert "could not be restored either" in update_log(world)


def test_a_new_version_that_does_not_start_is_rolled_back(world):
    write_pkg(world["src"], "2.0.0", broken=True)
    r = run_apply(world)
    assert r.returncode == 1, r.stdout + r.stderr
    assert_old_version_intact(world)
    log = update_log(world)
    assert "broken build" in log and "does not start" in log and "The previous version was restored." in log


def test_a_git_checkout_is_never_overwritten(world):
    (world["app"] / ".git").mkdir()
    r = run_apply(world)
    assert r.returncode != 0
    assert_old_version_intact(world)
    assert "git checkout" in update_log(world) and not world["uv_log"].exists()


def test_without_uv_nothing_is_touched(world):
    for f in (world["tmp"] / "bin").iterdir():
        f.unlink()
    env = dict(world["env"])
    env["PATH"] = os.pathsep.join(p for p in env["PATH"].split(os.pathsep) if not (Path(p) / ("uv.exe" if IS_WIN else "uv")).exists())
    world["env"] = env
    r = run_apply(world)
    assert r.returncode == 1
    assert_old_version_intact(world)
    assert "uv was not found" in update_log(world)
