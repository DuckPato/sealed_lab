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
import struct
import subprocess
import sys
import zipfile
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the app root (holds pyproject.toml)

# Paths (relative to ROOT, "/" separators) that go to the public repository. Everything else stays private.
ALLOW = [
    "sealed_lab/**",
    "tests/**",
    "scripts/*.py",
    "tools/export_public.py",
    "tools/release_sign.py",
    "tools/bundle_sets.py",
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
    "GETTING_STARTED.md",
    "guide_images/*.png",
    "DEVELOPING.md",
    "SECURITY.md",
    "CONTRIBUTING.md",
    "LICENSE",
]
# What friends download in the release zip: the app without the developer parts.
ZIP_EXCLUDE = ["tests/**", "tools/**", ".github/**", "scripts/**", ".gitattributes"]
RENAME = {"tools/public.gitignore": ".gitignore"}        # exported under another name

_PLACEHOLDER = r"(?!someone\b|someone-with|you\b|<|%|\{|\$)"                  # made-up user names the docs and tests use
DENY_BUILTIN = [
    r"[A-Za-z]:[\\/]+Users[\\/]+" + _PLACEHOLDER + r"[A-Za-z0-9._-]+",     # C:\Users\<real name>; someone / you are placeholders
    r"(?<![A-Za-z0-9._-])/[A-Za-z]/Users/" + _PLACEHOLDER + r"[A-Za-z0-9._-]+",          # /c/Users/<name> (Git Bash, any drive)
    r"(?<![A-Za-z0-9:\\/._-])/Users/" + _PLACEHOLDER + r"[A-Za-z0-9._-]+",               # /Users/<name> (macOS)
    r"(?<![A-Za-z0-9:\\/._-])/home/(?!someone\b|someone-with|you\b|user\b|runner\b|name\b|<|%|\{|\$)[A-Za-z0-9._-]+",   # /home/<name> (CI's runner is no person)
    r"[A-Za-z](?:%3A|:(?=%))(?:%5C|%2F|/|\\)+Users(?:%5C|%2F|/|\\)+" + _PLACEHOLDER + r"[A-Za-z0-9._-]+",   # URL-encoded C%3A%5CUsers%5C<name>
    r"AppData[\\/]+Local[\\/]+Temp[\\/]+claude",                                          # agent scratch folders
]
DENY_FILE = ROOT / "tools" / ".export_deny.txt"
BINARY_EXT = {".png", ".ico", ".jpg", ".jpeg", ".gif", ".zip", ".gz", ".pyc"}
METADATA_EXT = {".png", ".ico", ".jpg", ".jpeg", ".gif"}   # images: their text metadata is scanned too (a screenshot tool can write paths)
PNG_SIG = b"\x89PNG\r\n\x1a\n"


def _match(rel: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(rel, p) or (p.endswith("/**") and rel.startswith(p[:-2])) for p in patterns)


def tracked_files(root: Path | None = None) -> list[str]:
    """Files git tracks under `root` (default ROOT; untracked scratch never leaves), relative to it with "/" separators."""
    root = ROOT if root is None else root
    out = subprocess.run(["git", "ls-files", "-z", "--", "."], cwd=root, capture_output=True, check=True).stdout
    rels = [r for r in out.decode("utf-8").split("\0") if r]
    return sorted(r for r in rels if (root / r).is_file())


# Never exported, whatever the allowlist says: the ready-made sets of a release (the maintainer's calculated data; they go into the signed
# release zip through tools/bundle_sets.py + release_sign.py, not through the repository).
NEVER = ["bundled_sets/**", "*/bundled_sets/**"]


def selected(files: list[str] | None = None, root: Path | None = None) -> list[str]:
    return [r for r in (tracked_files(root) if files is None else files) if _match(r, ALLOW) and not _match(r, NEVER)]


def deny_patterns() -> list[re.Pattern]:
    pats = list(DENY_BUILTIN)
    if DENY_FILE.is_file():
        pats += [ln.strip() for ln in DENY_FILE.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]
    return [re.compile(p, re.IGNORECASE) for p in pats]


def _png_blobs(data: bytes) -> list[tuple[str, bytes]]:
    """(label, bytes) of what a PNG can carry besides pixels: tEXt / zTXt / iTXt text (decompressed), eXIf and any other ancillary
    chunk as raw bytes, and anything after IEND.  A damaged file yields its whole raw bytes as well."""
    if not data.startswith(PNG_SIG):
        return [("raw", data)]
    out: list[tuple[str, bytes]] = []
    pos = len(PNG_SIG)
    try:
        while pos + 8 <= len(data):
            n, kind = struct.unpack(">I4s", data[pos:pos + 8])
            body = data[pos + 8:pos + 8 + n]
            pos += 12 + n
            k = kind.decode("latin-1")
            if k in ("IDAT", "IHDR", "PLTE"):
                continue
            if k == "zTXt":
                kw, _, rest = body.partition(b"\0")
                out.append((k, kw + b"\0" + zlib.decompressobj().decompress(rest[1:], 1 << 20)))
            elif k == "iTXt":
                kw, _, rest = body.partition(b"\0")
                flag, rest = rest[:1], rest[2:]
                _lang, _, rest = rest.partition(b"\0")
                _tkw, _, text = rest.partition(b"\0")
                out.append((k, kw + b"\0" + (zlib.decompressobj().decompress(text, 1 << 20) if flag == b"\x01" else text)))
            else:
                out.append((k, body))                          # tEXt, eXIf, tIME, anything else: as bytes
            if k == "IEND":
                if pos < len(data):
                    out.append(("after IEND", data[pos:]))
                break
    except (struct.error, zlib.error):
        out.append(("raw", data))
    return out


def _blob_hits(rel: str, label: str, blob: bytes, patterns: list[re.Pattern]) -> list[str]:
    hits, seen = [], set()
    texts = [blob.decode("latin-1"), blob.decode("utf-8", errors="replace"),
             blob.decode("utf-16-le", errors="replace"), blob[1:].decode("utf-16-le", errors="replace")]
    for text in texts:
        for line in re.split(r"[\r\n\0]+", text):
            for p in patterns:
                m = p.search(line)
                if m and m.group(0) not in seen:
                    seen.add(m.group(0))
                    hits.append(f"{rel}:{label}: {m.group(0)}")
    return hits


def scan(rels: list[str], root: Path | None = None, patterns: list[re.Pattern] | None = None) -> list[str]:
    """`path:line: match` for every denied string in the text files (empty: clean).  Images are scanned as bytes: a PNG's text, EXIF
    and other ancillary chunks and any trailing data, whole ICO / JPEG / GIF files (`path:chunk: match`)."""
    patterns = deny_patterns() if patterns is None else patterns
    root = ROOT if root is None else root
    hits = []
    for rel in rels:
        ext = Path(rel).suffix.lower()
        if ext in METADATA_EXT:
            data = (root / rel).read_bytes()
            for label, blob in (_png_blobs(data) if ext == ".png" else [("bytes", data)]):
                hits += _blob_hits(rel, label, blob, patterns)
            continue
        if ext in BINARY_EXT:
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


def build_zip(out_dir: Path, rels: list[str], tag: str | None = None, root: Path | None = None) -> Path:
    """The release zip of the files `rels` under `root` (default ROOT); release_sign.py passes the tag's checkout as `root`."""
    root = ROOT if root is None else root
    ver = version(root)
    if tag is not None and tag.lstrip("v") != ver:
        raise SystemExit(f"tag {tag} does not match sealed_lab.__version__ {ver}")
    shipped = [r for r in rels if not (_match(r, ZIP_EXCLUDE) or "__pycache__" in r)]
    for r in shipped:                                   # a CRLF shell script fails on Linux (bad interpreter)
        if r.endswith(".sh") and b"\r" in (root / r).read_bytes():
            raise SystemExit(f"{r} has CRLF line endings; shell scripts must use LF (see .gitattributes)")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"sealed_lab-{ver}.zip"
    top = f"sealed_lab-{ver}"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for r in shipped:
            name = f"{top}/{RENAME.get(r, r)}"
            info = zipfile.ZipInfo.from_file(root / r, name)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3                          # 'made on Unix': unzip applies the mode bits only then
            info.external_attr = (0o100755 if r.endswith(".sh") else 0o100644) << 16      # unzip keeps the scripts executable
            zf.writestr(info, (root / r).read_bytes())
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
