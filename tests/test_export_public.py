"""tools/export_public.py: the allowlist, the personal-string scan, the sync into the public repo and the release zip."""
from __future__ import annotations

import importlib.util
import re
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("export_public", ROOT / "tools" / "export_public.py")
ex = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ex)


def test_allowlist_keeps_app_and_drops_research():
    files = ["sealed_lab/__init__.py", "sealed_lab/gui/assets/sealed_lab.ico", "tests/test_x.py", "scripts/a.py",
             "docs/builder_scoreboard/scoreboard.json", "docs/screenshots/pool_tab.png", "SPEC.md", "pyproject.toml",
             "uv.lock", "run_windows.bat", "apply_update.ps1", "README.md", ".github/workflows/release.yml", "tools/.export_deny.txt"]
    got = ex.selected(files)
    assert "sealed_lab/__init__.py" in got and "sealed_lab/gui/assets/sealed_lab.ico" in got and "uv.lock" in got
    assert ".github/workflows/release.yml" in got and "apply_update.ps1" in got
    assert not [f for f in got if f.startswith("docs/")]
    assert "SPEC.md" not in got and "tools/.export_deny.txt" not in got


def test_scan_flags_real_user_folders_only(tmp_path):
    bob = "C:" + "\\Users\\" + "Bob"                     # put together here, so this file itself passes the scan
    (tmp_path / "a.py").write_text(f'P = r"{bob}\\Documents\\x"\nQ = "C:/Users/someone/Documents/y"\n', encoding="utf-8")
    (tmp_path / "b.md").write_text("mail me at friend@example.org\n", encoding="utf-8")
    (tmp_path / "c.png").write_bytes(bob.encode())                            # binaries are not scanned
    pats = [re.compile(p, re.I) for p in ex.DENY_BUILTIN] + [re.compile("friend@example", re.I)]
    hits = ex.scan(["a.py", "b.md", "c.png"], root=tmp_path, patterns=pats)
    assert len(hits) == 2
    assert hits[0].startswith("a.py:1:") and hits[1].startswith("b.md:1:")


def test_the_tracked_app_is_clean():
    """What would be published right now has no personal strings (built-in patterns + the local deny file, if any)."""
    if not any((p / ".git").exists() for p in (ROOT, *ROOT.parents)):
        pytest.skip("not a git checkout (e.g. an unzipped release)")
    assert ex.scan(ex.selected()) == []


def test_sync_mirrors_and_keeps_git(tmp_path, monkeypatch):
    src = tmp_path / "src"
    for rel in ("sealed_lab/__init__.py", "tools/public.gitignore", "README.md"):
        (src / rel).parent.mkdir(parents=True, exist_ok=True)
        (src / rel).write_text(rel, encoding="utf-8")
    monkeypatch.setattr(ex, "ROOT", src)
    dest = tmp_path / "dest"
    (dest / ".git").mkdir(parents=True)
    (dest / ".git" / "HEAD").write_text("ref", encoding="utf-8")
    (dest / "old.py").write_text("stale", encoding="utf-8")
    (dest / "gone" / "x.py").parent.mkdir()
    (dest / "gone" / "x.py").write_text("stale", encoding="utf-8")
    (dest / ".venv").mkdir()
    (dest / ".venv" / "python.exe").write_text("untracked", encoding="utf-8")
    copied, removed = ex.sync(dest, ["sealed_lab/__init__.py", "tools/public.gitignore", "README.md"],
                              tracked=["old.py", "gone/x.py", "README.md"])
    assert (copied, removed) == (3, 2)
    assert (dest / ".venv" / "python.exe").exists() and not (dest / "gone").exists()
    assert (dest / ".gitignore").read_text(encoding="utf-8") == "tools/public.gitignore"
    assert (dest / ".git" / "HEAD").exists() and not (dest / "old.py").exists() and not (dest / "tools").exists()


def test_zip_layout_and_tag_check(tmp_path, monkeypatch):
    src = tmp_path / "src"
    for rel, text in (("sealed_lab/__init__.py", '__version__ = "1.2.3"\n'), ("pyproject.toml", "[project]\n"),
                      ("tests/test_a.py", ""), ("README.md", "hi")):
        (src / rel).parent.mkdir(parents=True, exist_ok=True)
        (src / rel).write_text(text, encoding="utf-8")
    monkeypatch.setattr(ex, "ROOT", src)
    rels = ["sealed_lab/__init__.py", "pyproject.toml", "tests/test_a.py", "README.md"]
    path = ex.build_zip(tmp_path / "dist", rels, tag="v1.2.3")
    assert path.name == "sealed_lab-1.2.3.zip"
    names = zipfile.ZipFile(path).namelist()
    assert sorted(names) == ["sealed_lab-1.2.3/README.md", "sealed_lab-1.2.3/pyproject.toml", "sealed_lab-1.2.3/sealed_lab/__init__.py"]
    with pytest.raises(SystemExit):
        ex.build_zip(tmp_path / "dist", rels, tag="v9.9.9")
