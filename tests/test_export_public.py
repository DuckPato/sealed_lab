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


def test_allowlist_has_the_linux_files():
    files = ["run_linux.sh", "create_shortcuts.sh", "apply_update.sh", ".gitattributes", "apply_update.ps1", "other.sh"]
    assert ex.selected(files) == ["run_linux.sh", "create_shortcuts.sh", "apply_update.sh", ".gitattributes", "apply_update.ps1"]


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


def _zip_src(tmp_path, monkeypatch, sh_text="#!/usr/bin/env bash\necho hi\n"):
    src = tmp_path / "src"
    files = {"sealed_lab/__init__.py": '__version__ = "1.2.3"\n', "pyproject.toml": "[project]\n", "README.md": "hi",
             "run_linux.sh": sh_text, "apply_update.sh": sh_text, "create_shortcuts.sh": sh_text, ".gitattributes": "*.sh text eol=lf\n"}
    for rel, text in files.items():
        (src / rel).parent.mkdir(parents=True, exist_ok=True)
        (src / rel).write_bytes(text.encode("utf-8"))
    monkeypatch.setattr(ex, "ROOT", src)
    return list(files)


def test_zip_shell_scripts_are_executable_and_lf(tmp_path, monkeypatch):
    rels = _zip_src(tmp_path, monkeypatch)
    path = ex.build_zip(tmp_path / "dist", rels)
    with zipfile.ZipFile(path) as z:
        assert "sealed_lab-1.2.3/.gitattributes" not in z.namelist()           # public repo root only, not in the zip
        for n in ("run_linux.sh", "apply_update.sh", "create_shortcuts.sh"):
            info = z.getinfo(f"sealed_lab-1.2.3/{n}")
            assert (info.external_attr >> 16) == 0o100755
            assert b"\r" not in z.read(info)
        assert (z.getinfo("sealed_lab-1.2.3/README.md").external_attr >> 16) == 0o100644


def test_zip_refuses_crlf_shell_script(tmp_path, monkeypatch):
    rels = _zip_src(tmp_path, monkeypatch, sh_text="#!/usr/bin/env bash\r\necho hi\r\n")
    with pytest.raises(SystemExit, match="CRLF"):
        ex.build_zip(tmp_path / "dist", rels)


def test_getting_started_images_exist_are_used_and_exported():
    text = (ROOT / "GETTING_STARTED.md").read_text(encoding="utf-8")
    used = set(re.findall(r"\]\((guide_images/[^)]+\.png)\)", text))
    on_disk = {f"guide_images/{p.name}" for p in (ROOT / "guide_images").glob("*.png")}
    assert used and used == on_disk                     # every picture is shown, every link has its picture
    assert set(ex.selected(sorted(on_disk | {"GETTING_STARTED.md"}))) == on_disk | {"GETTING_STARTED.md"}
