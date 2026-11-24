"""tools/import_pr.py: a pull request's commits land, with their author, on a pr-<N> branch in a separate worktree under the app
folder's prefix; nothing is left behind when the patch does not apply; odd or renamed paths are refused."""
from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("import_pr", ROOT / "tools" / "import_pr.py")
ip = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ip)


@pytest.fixture(autouse=True)
def _identity(monkeypatch):
    for k, v in (("GIT_AUTHOR_NAME", "Owner"), ("GIT_AUTHOR_EMAIL", "owner@example.org"), ("GIT_COMMITTER_NAME", "Owner"),
                 ("GIT_COMMITTER_EMAIL", "owner@example.org")):
        monkeypatch.setenv(k, v)


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout


def repo(path: Path, files: dict[str, str]) -> Path:
    path.mkdir(parents=True)
    git(path, "init", "-q", "-b", "main")
    git(path, "config", "core.autocrlf", "false")
    for rel, text in files.items():
        (path / rel).parent.mkdir(parents=True, exist_ok=True)
        (path / rel).write_text(text, encoding="utf-8", newline="\n")
    git(path, "add", "-A")
    git(path, "commit", "-q", "-m", "base")
    return path


def friend_patch(tmp_path: Path, base: dict[str, str], change: dict[str, str]) -> str:
    pub = repo(tmp_path / "pub", base)
    for rel, text in change.items():
        (pub / rel).parent.mkdir(parents=True, exist_ok=True)
        (pub / rel).write_text(text, encoding="utf-8", newline="\n")
    git(pub, "add", "-A")
    git(pub, "-c", "user.name=Friend", "-c", "user.email=friend@example.org", "commit", "-q", "--author",
        "Friend <friend@example.org>", "-m", "Better wording")
    return git(pub, "format-patch", "--stdout", "HEAD~1")


BASE = {"sealed_lab/x.py": "a = 1\n", "README.md": "hello\n"}


def test_the_pull_request_lands_on_its_own_branch_and_worktree(tmp_path):
    patch = friend_patch(tmp_path, BASE, {"sealed_lab/x.py": "a = 2\n"})
    dev = repo(tmp_path / "dev", {f"app/{k}": v for k, v in BASE.items()} | {"notes.md": "private\n"})
    dest = ip.import_patch(patch, 7, repo_root=dev, prefix="app")
    assert dest == dev / ".pr" / "pr-7"
    assert (dest / "app" / "sealed_lab" / "x.py").read_text(encoding="utf-8").replace("\r\n", "\n") == "a = 2\n"
    assert git(dest, "log", "-1", "--format=%an <%ae>|%s").strip() == "Friend <friend@example.org>|Better wording"
    assert (dev / "app" / "sealed_lab" / "x.py").read_text(encoding="utf-8") == "a = 1\n"            # the running checkout is untouched
    assert git(dev, "rev-parse", "--abbrev-ref", "HEAD").strip() == "main"
    with pytest.raises(SystemExit, match="already exists"):
        ip.import_patch(patch, 7, repo_root=dev, prefix="app")


def test_a_patch_that_does_not_apply_leaves_nothing_behind(tmp_path):
    patch = friend_patch(tmp_path, BASE, {"sealed_lab/x.py": "a = 2\n"})
    dev = repo(tmp_path / "dev", {"app/sealed_lab/x.py": "completely = 'different'\n", "app/README.md": "hello\n"})
    with pytest.raises(SystemExit, match="does not apply"):
        ip.import_patch(patch, 8, repo_root=dev, prefix="app")
    assert not (dev / ".pr" / "pr-8").exists()
    assert "pr-8" not in git(dev, "branch", "--list")


def test_odd_or_renamed_paths_are_refused_and_risky_ones_flagged(tmp_path):
    problems, warnings = ip.review_notes(["sealed_lab/x.py", ".gitignore", "../evil.py", ".git/hooks/pre-commit", "SPEC.md",
                                          "sealed_lab/updater.py", "run_windows.bat", ".github/workflows/release.yml"])
    assert any(p.startswith(".gitignore:") for p in problems)
    assert any(p.startswith("../evil.py:") for p in problems) and any(p.startswith(".git/hooks") for p in problems)
    assert not any(p.startswith("sealed_lab/x.py") for p in problems + warnings)
    assert any(w.startswith("SPEC.md:") and "never ship" in w for w in warnings)
    for risky in ("sealed_lab/updater.py", "run_windows.bat", ".github/workflows/release.yml"):
        assert any(w.startswith(risky) and "read every line" in w for w in warnings), risky
    patch = friend_patch(tmp_path, BASE, {".gitignore": "x\n"})
    dev = repo(tmp_path / "dev", {f"app/{k}": v for k, v in BASE.items()})
    with pytest.raises(SystemExit, match="refusing"):
        ip.import_patch(patch, 9, repo_root=dev, prefix="app")
    assert not (dev / ".pr").exists()


def test_a_missing_git_identity_is_reported_before_anything_is_created(tmp_path, monkeypatch):
    patch = friend_patch(tmp_path, BASE, {"sealed_lab/x.py": "a = 2\n"})
    dev = repo(tmp_path / "dev", {f"app/{k}": v for k, v in BASE.items()})
    real = ip._git

    def no_identity(args, cwd, check=True, stdin=None):
        if args[:2] == ["var", "GIT_COMMITTER_IDENT"]:
            return subprocess.CompletedProcess(args, 128, "", "unable to auto-detect email address")
        return real(args, cwd, check=check, stdin=stdin)

    monkeypatch.setattr(ip, "_git", no_identity)
    with pytest.raises(SystemExit, match="git config user.name"):
        ip.import_patch(patch, 10, repo_root=dev, prefix="app")
    assert not (dev / ".pr").exists()


def test_patch_paths_are_read_from_the_diff_headers():
    patch = "From x\n\ndiff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\ndiff --git a/old.md b/new.md\n"
    assert ip.changed_paths(patch) == ["a.py", "old.md", "new.md"]
    with pytest.raises(SystemExit, match="unusual"):
        ip.changed_paths('diff --git "a/sp ace.py" "b/sp ace.py"\n')
    with pytest.raises(SystemExit, match="no files"):
        ip.changed_paths("just text\n")
