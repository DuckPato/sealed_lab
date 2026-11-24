"""Copy the shareable part of the app into the public (friends') repository, and build the release zip.

The development repository holds more than the app (research notes, result files, planning documents). The GitHub repository
that friends get holds only an allowlist of tracked files, and every exported text file is scanned for personal strings first.

    python tools/export_public.py --dest PATH_TO_PUBLIC_CLONE        # sync the allowlist into that folder (its .git is kept)
    python tools/export_public.py --zip dist [--tag v0.2.0]          # build dist/sealed_lab-<version>.zip (the release asset)
    python tools/export_public.py --check                            # only list and scan what would be exported

Deny list: built-in patterns (a Windows user folder other than the made-up names the tests use) plus one pattern per line
in `tools/.export_deny.txt` (untracked and never exported: put your own name, email and similar there).
"""
from __future__ import annotations

import argparse
import fnmatch
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the app root (holds pyproject.toml)

# Paths (relative to ROOT, "/" separators) that go to the public repository. Everything else stays private.
ALLOW = [
    "sealed_lab/**",
    "tests/**",
    "scripts/*.py",
    "tools/export_public.py",
    "tools/release_sign.py",
    "tools/import_pr.py",
    "tools/public.gitignore",
    ".github/**",
    "pyproject.toml",
    "uv.lock",
    ".python-version",
    "run_windows.bat",
    "run_windows_cli.bat",
    "create_shortcuts.ps1",
    "apply_update.ps1",
    "run_linux.sh",
    "create_shortcuts.sh",
    "apply_update.sh",
    "get_uv.ps1",
    "get_uv.sh",
    ".gitattributes",
    "README.md",
    "GUIDE.md",
    "DEVELOPING.md",
    "SECURITY.md",
    "CONTRIBUTING.md",
    "LICENSE",
]
# What friends download in the release zip: the app without the developer parts.
ZIP_EXCLUDE = ["tests/**", "tools/**", ".github/**", "scripts/**", ".gitattributes"]
RENAME = {"tools/public.gitignore": ".gitignore"}        # exported under another name

DENY_BUILTIN = [
    r"[A-Za-z]:[\\/]+Users[\\/]+(?!someone\b|someone-with|<|%|\{|\$)[A-Za-z0-9._-]+",     # C:\Users\<real name>
    r"AppData[\\/]+Local[\\/]+Temp[\\/]+claude",                                          # agent scratch folders
]
DENY_FILE = ROOT / "tools" / ".export_deny.txt"
BINARY_EXT = {".png", ".ico", ".jpg", ".jpeg", ".gif", ".zip", ".gz", ".pyc"}


def _match(rel: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(rel, p) or (p.endswith("/**") and rel.startswith(p[:-2])) for p in patterns)


def tracked_files() -> list[str]:
    """Files git tracks under ROOT (untracked scratch never leaves), relative to ROOT with "/" separators."""
    out = subprocess.run(["git", "ls-files", "-z", "--", "."], cwd=ROOT, capture_output=True, check=True).stdout
    rels = [r for r in out.decode("utf-8").split("\0") if r]
    return sorted(r for r in rels if (ROOT / r).is_file())


def selected(files: list[str] | None = None) -> list[str]:
    return [r for r in (tracked_files() if files is None else files) if _match(r, ALLOW)]


def deny_patterns() -> list[re.Pattern]:
    pats = list(DENY_BUILTIN)
    if DENY_FILE.is_file():
        pats += [ln.strip() for ln in DENY_FILE.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]
    return [re.compile(p, re.IGNORECASE) for p in pats]


def scan(rels: list[str], root: Path | None = None, patterns: list[re.Pattern] | None = None) -> list[str]:
    """`path:line: match` for every denied string in the text files (empty: clean)."""
    patterns = deny_patterns() if patterns is None else patterns
    root = ROOT if root is None else root
    hits = []
    for rel in rels:
        if Path(rel).suffix.lower() in BINARY_EXT:
            continue
        text = (root / rel).read_text(encoding="utf-8", errors="replace")
        for i, line in enumerate(text.splitlines(), 1):
            for p in patterns:
                m = p.search(line)
                if m:
                    hits.append(f"{rel}:{i}: {m.group(0)}")
    return hits


def version(root: Path | None = None) -> str:
    root = ROOT if root is None else root
    m = re.search(r'^__version__\s*=\s*"([^"]+)"', (root / "sealed_lab" / "__init__.py").read_text(encoding="utf-8"), re.M)
    if not m:
        raise SystemExit("no __version__ in sealed_lab/__init__.py")
    return m.group(1)


def _dest_tracked(dest: Path) -> list[str]:
    """Files the public clone's git tracks (none when `dest` is not a git repository)."""
    if not (dest / ".git").exists():
        return []
    out = subprocess.run(["git", "ls-files", "-z"], cwd=dest, capture_output=True, check=True).stdout
    return [r for r in out.decode("utf-8").split("\0") if r]


def sync(dest: Path, rels: list[str], tracked: list[str] | None = None) -> tuple[int, int]:
    """Copy the exported files into `dest` and delete the files its git tracks that are no longer exported. Untracked files
    (a .venv, test output) are never touched. Returns (copied, removed)."""
    want = {RENAME.get(r, r) for r in rels}
    removed = 0
    for rel in (_dest_tracked(dest) if tracked is None else tracked):
        p = dest / rel
        if rel not in want and p.is_file():
            p.unlink()
            removed += 1
            parent = p.parent
            while parent != dest and parent.is_dir() and not any(parent.iterdir()):
                parent.rmdir()
                parent = parent.parent
    for r in rels:
        target = dest / RENAME.get(r, r)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / r, target)
    return len(rels), removed


def build_zip(out_dir: Path, rels: list[str], tag: str | None = None) -> Path:
    ver = version()
    if tag is not None and tag.lstrip("v") != ver:
        raise SystemExit(f"tag {tag} does not match sealed_lab.__version__ {ver}")
    shipped = [r for r in rels if not (_match(r, ZIP_EXCLUDE) or "__pycache__" in r)]
    for r in shipped:                                   # a CRLF shell script fails on Linux (bad interpreter)
        if r.endswith(".sh") and b"\r" in (ROOT / r).read_bytes():
            raise SystemExit(f"{r} has CRLF line endings; shell scripts must use LF (see .gitattributes)")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"sealed_lab-{ver}.zip"
    top = f"sealed_lab-{ver}"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for r in shipped:
            name = f"{top}/{RENAME.get(r, r)}"
            info = zipfile.ZipInfo.from_file(ROOT / r, name)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3                          # 'made on Unix': unzip applies the mode bits only then
            info.external_attr = (0o100755 if r.endswith(".sh") else 0o100644) << 16      # unzip keeps the scripts executable
            zf.writestr(info, (ROOT / r).read_bytes())
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dest", type=Path, help="public repository folder to sync into")
    ap.add_argument("--zip", type=Path, help="folder to write the release zip into")
    ap.add_argument("--tag", help="release tag; must match sealed_lab.__version__")
    ap.add_argument("--check", action="store_true", help="list and scan only")
    a = ap.parse_args(argv)
    rels = selected()
    hits = scan(rels)
    if hits:
        print("Personal strings found, nothing exported:\n  " + "\n  ".join(hits), file=sys.stderr)
        return 2
    print(f"{len(rels)} files, scan clean (version {version()})")
    if a.check:
        print("\n".join(rels))
    if a.dest:
        if (a.dest.resolve() == ROOT or ROOT in a.dest.resolve().parents):
            raise SystemExit("--dest must be outside the app folder")
        a.dest.mkdir(parents=True, exist_ok=True)
        copied, removed = sync(a.dest, rels)
        print(f"synced into {a.dest}: {copied} files copied, {removed} removed")
    if a.zip:
        print(f"wrote {build_zip(a.zip, rels, a.tag)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
