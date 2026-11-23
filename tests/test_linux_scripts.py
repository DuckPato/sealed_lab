"""run_linux.sh, create_shortcuts.sh and apply_update.sh with a stub `uv` (Linux only; skipped elsewhere)."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux") or shutil.which("bash") is None, reason="Linux shell scripts")

APP = Path(__file__).resolve().parent.parent


def make_env(tmp_path: Path) -> tuple[dict[str, str], Path]:
    """A stub `uv` first on PATH that records its arguments; HOME, SEALED_LAB_HOME and XDG_DATA_HOME are scratch folders."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    uv_log = tmp_path / "uv_calls.txt"
    uv = bin_dir / "uv"
    uv.write_text(f'#!/bin/sh\necho "$@" >> "{uv_log}"\nexit 0\n', encoding="utf-8")
    uv.chmod(0o755)
    env = dict(os.environ, PATH=f"{bin_dir}{os.pathsep}{os.environ['PATH']}", HOME=str(tmp_path / "home"),
               SEALED_LAB_HOME=str(tmp_path / "labhome"), XDG_DATA_HOME=str(tmp_path / "xdg"))
    (tmp_path / "home").mkdir(exist_ok=True)
    return env, uv_log


def fake_venv(app: Path) -> Path:
    py = app / ".venv" / "bin" / "python"
    py.parent.mkdir(parents=True, exist_ok=True)
    py.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    py.chmod(0o755)
    return py


def fake_source(src: Path) -> None:
    (src / "sealed_lab").mkdir(parents=True)
    (src / "sealed_lab" / "__init__.py").write_text('__version__ = "9.9.9"\n', encoding="utf-8")
    (src / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (src / "run_linux.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    (src / "create_shortcuts.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    (src / "new_file.txt").write_text("new", encoding="utf-8")
    (src / ".venv").mkdir()
    (src / ".venv" / "ignored.txt").write_text("no", encoding="utf-8")


def run_apply(env, src, app, *extra):
    return subprocess.run(["bash", str(APP / "apply_update.sh"), "--source", str(src), "--app-dir", str(app), "--no-start",
                           "--no-shortcuts", "--no-dialog", *extra], env=env, capture_output=True, text=True, timeout=120)


def test_apply_update_copies_syncs_and_logs(tmp_path):
    env, uv_log = make_env(tmp_path)
    src, app = tmp_path / "src", tmp_path / "app"
    fake_source(src)
    app.mkdir()
    (app / "stale.txt").write_text("old", encoding="utf-8")
    py = fake_venv(app)
    before = py.read_text(encoding="utf-8")
    r = run_apply(env, src, app)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (app / "new_file.txt").read_text(encoding="utf-8") == "new" and (app / "sealed_lab" / "__init__.py").is_file()
    assert (app / "stale.txt").exists()                                  # nothing is deleted
    assert py.read_text(encoding="utf-8") == before and not (app / ".venv" / "ignored.txt").exists()
    assert "sync --frozen --inexact" in uv_log.read_text(encoding="utf-8")
    assert "Update to 9.9.9 done." in (tmp_path / "labhome" / "logs" / "update.log").read_text(encoding="utf-8")


def test_apply_update_refuses_a_git_checkout(tmp_path):
    env, uv_log = make_env(tmp_path)
    src, app = tmp_path / "src", tmp_path / "app"
    fake_source(src)
    (app / ".git").mkdir(parents=True)
    r = run_apply(env, src, app)
    assert r.returncode != 0
    assert "ERROR:" in (tmp_path / "labhome" / "logs" / "update.log").read_text(encoding="utf-8")
    assert not (app / "new_file.txt").exists() and not uv_log.exists()


def test_apply_update_fails_when_uv_sync_fails(tmp_path):
    env, _ = make_env(tmp_path)
    (tmp_path / "bin" / "uv").write_text("#!/bin/sh\necho boom\nexit 3\n", encoding="utf-8")
    src, app = tmp_path / "src", tmp_path / "app"
    fake_source(src)
    app.mkdir()
    r = run_apply(env, src, app)
    assert r.returncode != 0
    log = (tmp_path / "labhome" / "logs" / "update.log").read_text(encoding="utf-8")
    assert "ERROR: uv sync failed" in log and "boom" in log


def test_create_shortcuts_writes_the_desktop_entry(tmp_path):
    env, _ = make_env(tmp_path)
    app = tmp_path / "my app"
    app.mkdir()
    r = subprocess.run(["bash", str(APP / "create_shortcuts.sh"), str(app)], env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 1 and "run_linux.sh" in r.stderr             # no .venv yet
    py = fake_venv(app)
    for _ in range(2):                                                   # safe to run again
        r = subprocess.run(["bash", str(APP / "create_shortcuts.sh"), str(app)], env=env, capture_output=True, text=True, timeout=60)
        assert r.returncode == 0, r.stderr
    f = tmp_path / "xdg" / "applications" / "sealed_lab.desktop"
    lines = f.read_text(encoding="utf-8").splitlines()
    assert f'Exec="{py}" -m sealed_lab' in lines
    assert f"Path={app}" in lines and f"Icon={app}/sealed_lab/gui/assets/sealed_lab.png" in lines
    assert "Terminal=false" in lines and "StartupWMClass=sealed_lab" in lines and "Categories=Game;CardGame;" in lines
    assert (f.stat().st_mode & 0o777) == 0o644
    assert not (tmp_path / "home" / "Desktop").exists()


def test_run_linux_syncs_with_uv(tmp_path):
    env, uv_log = make_env(tmp_path)
    app = tmp_path / "appcopy"
    app.mkdir()
    shutil.copy2(APP / "run_linux.sh", app / "run_linux.sh")
    r = subprocess.run(["bash", str(app / "run_linux.sh"), "--no-start", "--no-shortcuts"], env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "sync --frozen --inexact" in uv_log.read_text(encoding="utf-8")
    assert not (tmp_path / "xdg").exists()
