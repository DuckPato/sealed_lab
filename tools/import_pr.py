"""Bring a pull request from the public repository into the development repository, for review.

The public repository is an export of this one (tools/export_public.py), so a pull request merged only on GitHub would be overwritten by
the next export. This tool fetches the pull request's commits as patches and applies them, with their authors, on a new branch
`pr-<N>` in a separate worktree (`<repo>/.pr/pr-<N>`), so the working tree the app runs from is not touched. It runs nothing from the
pull request: review the change before running its tests or merging it.

    uv run python tools/import_pr.py 12                          # pull request #12 of DuckPato/sealed_lab (needs the GitHub CLI)
    uv run python tools/import_pr.py 12 --patch-file pr12.patch  # or a patch saved from https://github.com/<repo>/pull/12.patch
"""
from __future__ import annotations

import argparse
import importlib.util
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the app root (holds pyproject.toml)
DEFAULT_REPO = "DuckPato/sealed_lab"

_spec = importlib.util.spec_from_file_location("export_public", ROOT / "tools" / "export_public.py")
export_public = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(export_public)

# Files where a change can affect how the app installs, updates, downloads or what gets released: read every line.
SENSITIVE = [
    "sealed_lab/updater.py", "sealed_lab/release_sig.py", "sealed_lab/release_keys.py", "sealed_lab/gui/update_ui.py",
    "sealed_lab/ingest/datasets.py", "sealed_lab/ingest/scryfall*.py", "*.bat", "*.ps1", "*.sh", ".github/**", "tools/**",
    "pyproject.toml", "uv.lock", ".gitattributes", ".gitignore",
]


class ImportFailed(SystemExit):
    pass


def _die(msg: str) -> None:
    raise ImportFailed(f"import_pr: {msg}")


def _git(args: list[str], cwd: Path, check: bool = True, stdin: str | None = None) -> subprocess.CompletedProcess:
    r = subprocess.run(["git", *args], cwd=cwd, input=stdin, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if check and r.returncode != 0:
        _die(f"git {' '.join(args)} failed:\n{r.stderr.strip() or r.stdout.strip()}")
    return r


def _gh() -> str:
    for cand in (shutil.which("gh"), r"C:\Program Files\GitHub CLI\gh.exe"):
        if cand and Path(cand).exists():
            return cand
    _die("the GitHub CLI (gh) was not found; save the patch from the pull request page and use --patch-file")
    return ""


def fetch_patch(number: int, repo: str) -> tuple[str, str]:
    """(patch text, pull request URL) through the GitHub CLI."""
    gh = _gh()
    view = subprocess.run([gh, "pr", "view", str(number), "-R", repo, "--json", "url,state,title", "-q", ".url + \" \" + .state"],
                          capture_output=True, text=True, encoding="utf-8")
    if view.returncode != 0:
        _die(f"pull request #{number} of {repo} not found: {view.stderr.strip()}")
    url, state = (view.stdout.strip().split(" ", 1) + [""])[:2]
    if state != "OPEN":
        print(f"note: pull request #{number} is {state.lower()}")
    diff = subprocess.run([gh, "pr", "diff", str(number), "-R", repo, "--patch"], capture_output=True, text=True, encoding="utf-8")
    if diff.returncode != 0 or not diff.stdout.strip():
        _die(f"could not download the patch of #{number}: {diff.stderr.strip()}")
    return diff.stdout, url


def changed_paths(patch: str) -> list[str]:
    """Every path (old and new name) the patch touches, as the public repository names them."""
    paths: list[str] = []
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            m = re.fullmatch(r"diff --git a/(\S+) b/(\S+)", line)
            if not m:
                _die(f"unusual file name in the patch (quoted or with spaces): {line[11:]}")
            for p in m.groups():
                if p not in paths:
                    paths.append(p)
    if not paths:
        _die("the patch changes no files (is it a .patch from the pull request?)")
    return paths


def review_notes(paths: list[str]) -> tuple[list[str], list[str]]:
    """(problems that stop the import, warnings to print). Paths are public-repository paths."""
    problems, warnings = [], []
    renamed = {v: k for k, v in export_public.RENAME.items()}
    for p in paths:
        parts = p.split("/")
        if p.startswith("/") or ".." in parts or parts[0] == ".git" or ":" in parts[0]:
            problems.append(f"{p}: not a plain path inside the repository")
        elif p in renamed:
            problems.append(f"{p}: lives here as {renamed[p]}; copy that change by hand")
        elif not export_public._match(p, export_public.ALLOW):
            warnings.append(f"{p}: not part of the public export, so it would never ship")
        if export_public._match(p, SENSITIVE):
            warnings.append(f"{p}: install / update / release file, read every line")
    return problems, warnings


def import_patch(patch: str, number: int, *, repo_root: Path, prefix: str, base: str = "HEAD",
                 dest: Path | None = None) -> Path:
    """Apply `patch` (git format-patch / mbox, paths relative to the public repository) under `prefix` on a new branch `pr-<number>`
    in a new worktree. Returns the worktree. Leaves nothing behind when the patch does not apply."""
    paths = changed_paths(patch)
    problems, _ = review_notes(paths)
    if problems:
        _die("refusing this pull request:\n  " + "\n  ".join(problems))
    if _git(["var", "GIT_COMMITTER_IDENT"], repo_root, check=False).returncode != 0:
        _die("git does not know who you are (the imported commits keep their author, but you are their committer). Set it for this "
             'repository: git config user.name "Your Name" ; git config user.email "you@example.com"')
    branch = f"pr-{number}"
    dest = (repo_root / ".pr" / branch) if dest is None else dest
    if _git(["rev-parse", "--verify", "--quiet", f"refs/heads/{branch}"], repo_root, check=False).returncode == 0:
        _die(f"branch {branch} already exists (remove it with: git worktree remove {dest} ; git branch -D {branch})")
    if dest.exists():
        _die(f"{dest} already exists")
    _git(["worktree", "add", "-b", branch, str(dest), base], repo_root)
    am = ["am", "--3way"] + ([f"--directory={prefix}"] if prefix else [])
    r = _git(am, dest, check=False, stdin=patch)
    if r.returncode != 0:
        _git(["am", "--abort"], dest, check=False)
        _git(["worktree", "remove", "--force", str(dest)], repo_root, check=False)
        _git(["branch", "-D", branch], repo_root, check=False)
        _die("the patch does not apply to this repository (the pull request may be based on an older release; ask for a rebase "
             f"onto the latest main):\n{(r.stderr or r.stdout).strip()[-1500:]}")
    return dest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("number", type=int, help="pull request number")
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--patch-file", type=Path, help="use this saved .patch instead of downloading it")
    ap.add_argument("--base", default="HEAD", help="branch or commit to start from (default: the current one)")
    a = ap.parse_args(argv)

    repo_root = Path(_git(["rev-parse", "--show-toplevel"], ROOT).stdout.strip())
    prefix = ROOT.resolve().relative_to(repo_root.resolve()).as_posix()
    prefix = "" if prefix == "." else prefix
    if a.patch_file:
        patch, url = a.patch_file.read_text(encoding="utf-8"), str(a.patch_file)
    else:
        patch, url = fetch_patch(a.number, a.repo)
    base = _git(["rev-parse", "--verify", f"{a.base}^{{commit}}"], repo_root).stdout.strip()[:12]    # fixed: HEAD differs in the worktree
    dest = import_patch(patch, a.number, repo_root=repo_root, prefix=prefix, base=base)

    branch = f"pr-{a.number}"
    _, warnings = review_notes(changed_paths(patch))
    log = _git(["log", "--format=  %h %an <%ae>: %s", f"{base}..{branch}"], dest).stdout.rstrip()
    stat = _git(["diff", "--stat", f"{base}...{branch}"], dest).stdout.rstrip()
    print(f"Imported {url} into branch {branch}, worktree {dest}\n\nCommits:\n{log}\n\n{stat}")
    if warnings:
        print("\nLook closely at:\n  " + "\n  ".join(warnings))
    app = dest / prefix if prefix else dest
    print(f"""
Next:
  1. Review the change:   git -C "{dest}" log -p {base}..{branch}
  2. Only then run its tests (this runs the contributor's code):
                          cd "{app}" ; uv run --extra dev pytest
  3. Accept it (app closed): git merge --ff-only {branch}   (or rebase it first if main has moved)
     Drop it:             git worktree remove "{dest}" ; git branch -D {branch}
  4. Export and release as usual, then close the pull request with a link to the release.""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
