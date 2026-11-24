"""Release signing, run by the maintainer on their own computer (never in CI: the private key must not reach GitHub).

    python tools/release_sign.py init [--key PATH] [--no-passphrase]
        Create the Ed25519 release key (private key: PATH, default ~/.sealed_lab_release/release_ed25519.pem, passphrase-protected
        unless --no-passphrase) and add its public half to sealed_lab/release_keys.py.  Back the private key up somewhere safe: without
        it, installed copies cannot be updated from the app any more (a fresh download still works).

    python tools/release_sign.py sign --tag vX.Y.Z --public PATH_TO_PUBLIC_CLONE [--key PATH] [--no-publish]
        For the DRAFT release that CI built for the tag: download its zip, rebuild the zip from a fresh checkout of the tag (a temporary
        worktree of the public clone) and require the two to be identical (same files, bytes and modes), then sign the manifest, attach it
        and publish the release.

    python tools/release_sign.py verify --tag vX.Y.Z
        Check a published release exactly as the app's updater does (manifest signature, repo / tag / file, size, SHA-256).

Needs the GitHub CLI (`gh`, signed in) for sign; verify is anonymous.
"""
from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import re
import runpy
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sealed_lab import release_sig  # noqa: E402

DEFAULT_KEY = Path.home() / ".sealed_lab_release" / "release_ed25519.pem"
KEYS_FILE = ROOT / "sealed_lab" / "release_keys.py"
DEFAULT_REPO = "DuckPato/sealed_lab"


def _die(msg: str) -> "None":
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(1)


def _gh() -> str:
    for cand in (shutil.which("gh"), r"C:\Program Files\GitHub CLI\gh.exe"):
        if cand and Path(cand).exists():
            return cand
    _die("the GitHub CLI (gh) was not found")
    return ""


def _run(cmd: list[str], cwd: Path | None = None) -> str:
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        _die(f"{' '.join(cmd[:4])} ... failed:\n{r.stderr.strip() or r.stdout.strip()}")
    return r.stdout


# --------------------------------------------------------------------------------------------- keys
def read_trusted(keys_file: Path) -> dict[str, str]:
    return dict(runpy.run_path(str(keys_file)).get("TRUSTED_KEYS") or {})


def write_trusted(keys_file: Path, keys: dict[str, str]) -> None:
    text = keys_file.read_text(encoding="utf-8")
    body = "".join(f'    "{k}": "{v}",\n' for k, v in sorted(keys.items()))
    new = re.sub(r"TRUSTED_KEYS: dict\[str, str\] = \{.*?\}\n", "TRUSTED_KEYS: dict[str, str] = {\n" + body + "}\n", text, flags=re.S)
    if new == text and keys:
        _die(f"could not find TRUSTED_KEYS in {keys_file}")
    keys_file.write_text(new, encoding="utf-8", newline="\n")


def _restrict(path: Path) -> None:
    """Owner-only access to the private key file (best effort)."""
    if os.name == "nt":
        user = os.environ.get("USERNAME")
        if user:
            subprocess.run(["icacls", str(path), "/inheritance:r", "/grant:r", f"{user}:F"], capture_output=True)
    else:
        os.chmod(path, 0o600)


def _public_raw(priv) -> bytes:
    from cryptography.hazmat.primitives import serialization

    return priv.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)


def cmd_init(a) -> int:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    key = Path(a.key)
    if key.exists():
        _die(f"{key} exists already (one key is enough; keep using it)")
    if a.no_passphrase:
        enc = serialization.NoEncryption()
    else:
        p1 = getpass.getpass("Passphrase for the release key (asked at every release): ")
        p2 = getpass.getpass("Same passphrase again: ")
        if p1 != p2:
            _die("the two passphrases differ")
        if len(p1) < 8:
            _die("use a passphrase of at least 8 characters (or --no-passphrase)")
        enc = serialization.BestAvailableEncryption(p1.encode("utf-8"))
    priv = Ed25519PrivateKey.generate()
    pem = priv.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, enc)
    key.parent.mkdir(parents=True, exist_ok=True)
    key.write_bytes(pem)
    _restrict(key)
    raw = _public_raw(priv)
    kid = release_sig.key_id(raw)
    keys = read_trusted(Path(a.keys_file))
    keys[kid] = release_sig._b64(raw)
    write_trusted(Path(a.keys_file), keys)
    print(f"Release key created: {key}")
    print(f"Key id {kid} added to {a.keys_file} (commit that file).")
    print("Back the private key file up (e.g. on a USB stick or in a password manager): it cannot be recreated.")
    return 0


def load_private(path: Path):
    from cryptography.hazmat.primitives import serialization

    data = Path(path).read_bytes()
    try:
        return serialization.load_pem_private_key(data, password=None)
    except TypeError:                                   # encrypted: asked every time, never read from the environment or a file
        pw = getpass.getpass("Release key passphrase: ")
        try:
            return serialization.load_pem_private_key(data, password=pw.encode("utf-8"))
        except ValueError:
            _die("wrong passphrase")


# --------------------------------------------------------------------------------------------- zips
def zip_entries(path: Path) -> dict[str, tuple[int, bytes]]:
    """name -> (unix mode, content) of every file entry."""
    with zipfile.ZipFile(path) as z:
        return {i.filename: ((i.external_attr >> 16) & 0o7777, z.read(i)) for i in z.infolist() if not i.is_dir()}


def compare_zips(built: Path, reference: Path) -> list[str]:
    a, b = zip_entries(built), zip_entries(reference)
    diffs = [f"only in the release zip: {n}" for n in sorted(set(a) - set(b))]
    diffs += [f"missing from the release zip: {n}" for n in sorted(set(b) - set(a))]
    for n in sorted(set(a) & set(b)):
        if a[n][1] != b[n][1]:
            diffs.append(f"content differs: {n}")
        elif a[n][0] != b[n][0]:
            diffs.append(f"mode differs: {n} ({oct(a[n][0])} vs {oct(b[n][0])})")
    return diffs


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------------------------- sign
def cmd_sign(a) -> int:
    gh, repo, tag = _gh(), a.repo, a.tag
    m = re.fullmatch(r"v(\d+(?:\.\d+)*)", tag)
    if not m:
        _die(f"{tag!r} is not a vX.Y.Z tag")
    version = m.group(1)
    public = Path(a.public).resolve()
    if not (public / ".git").exists():
        _die(f"{public} is not a git clone")
    tagged = _run(["git", "rev-parse", f"{tag}^{{commit}}"], public).strip()
    remote = _run(["git", "ls-remote", "origin", f"refs/tags/{tag}"], public).split()
    if not remote or remote[0] != _run(["git", "rev-parse", tag], public).strip():
        _die(f"the local tag {tag} is not the one on GitHub")
    rel = json.loads(_run([gh, "release", "view", tag, "-R", repo, "--json", "isDraft,assets"]))
    names = {x["name"] for x in rel.get("assets") or []}
    zname, mname = release_sig.zip_name(version), release_sig.manifest_name(version)
    if zname not in names:
        _die(f"release {tag} has no {zname}")
    if not rel.get("isDraft") and not a.allow_published:
        _die(f"release {tag} is already published (sign the draft CI creates; --allow-published to override)")
    if mname in names and not a.force:
        _die(f"release {tag} already has {mname} (--force to replace it)")
    priv = load_private(Path(a.key))
    kid = release_sig.key_id(_public_raw(priv))
    with tempfile.TemporaryDirectory(prefix="sealed_lab_sign_") as td:
        td = Path(td)
        src = td / "src"                                # a fresh checkout of the tag: the same bytes CI built from
        _run(["git", "worktree", "add", "--detach", str(src), tagged], public)
        try:
            trusted = read_trusted(src / "sealed_lab" / "release_keys.py")
            if kid not in trusted:
                _die(f"key {kid} is not in {tag}'s sealed_lab/release_keys.py: that version could not check the next update")
            _run([gh, "release", "download", tag, "-R", repo, "-p", zname, "-D", str(td / "gh")])
            built = td / "gh" / zname
            _run([sys.executable, "tools/export_public.py", "--zip", str(td / "ref"), "--tag", tag], src)
        finally:
            subprocess.run(["git", "worktree", "remove", "--force", str(src)], cwd=public, capture_output=True)
            subprocess.run(["git", "worktree", "prune"], cwd=public, capture_output=True)
        diffs = compare_zips(built, td / "ref" / zname)
        if diffs:
            _die("the zip on GitHub is not the one the tagged source builds:\n  " + "\n  ".join(diffs[:40]))
        print(f"{zname}: identical to the zip built from {tag} ({len(zip_entries(built))} files)")
        payload = release_sig.make_payload(version, repo, zname, built.stat().st_size, sha256(built),
                                           datetime.now(timezone.utc).isoformat(timespec="seconds"))
        manifest = release_sig.sign_payload(payload, priv)
        release_sig.check_payload(release_sig.verify_manifest(manifest, trusted), repo=repo, tag=tag, asset=zname)
        mpath = td / mname
        mpath.write_bytes(manifest)
        _run([gh, "release", "upload", tag, str(mpath), "-R", repo] + (["--clobber"] if a.force else []))
        print(f"signed with key {kid}: {mname} attached")
    if a.no_publish:
        print("left as a draft (--no-publish)")
        return 0
    _run([gh, "release", "edit", tag, "-R", repo, "--draft=false", "--latest"])
    print(f"{tag} published")
    return cmd_verify(a)


# --------------------------------------------------------------------------------------------- verify
def cmd_verify(a) -> int:
    from sealed_lab import updater

    info = updater.check(repo=a.repo, current="0")
    if info.error:
        _die(info.error)
    if info.latest != a.tag:
        _die(f"the latest published release is {info.latest}, not {a.tag}")
    trusted = read_trusted(KEYS_FILE)
    with tempfile.TemporaryDirectory(prefix="sealed_lab_verify_") as td:
        path, payload = updater.download(info, Path(td), repo=a.repo, trusted=trusted)
        print(f"OK: {info.latest} {path.name} {payload['size']} bytes, sha256 {payload['sha256'][:16]}..., signature valid")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init")
    p.add_argument("--key", default=str(DEFAULT_KEY))
    p.add_argument("--keys-file", default=str(KEYS_FILE))
    p.add_argument("--no-passphrase", action="store_true")
    p = sub.add_parser("sign")
    p.add_argument("--tag", required=True)
    p.add_argument("--public", required=True)
    p.add_argument("--key", default=str(DEFAULT_KEY))
    p.add_argument("--repo", default=DEFAULT_REPO)
    p.add_argument("--no-publish", action="store_true")
    p.add_argument("--allow-published", action="store_true")
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("verify")
    p.add_argument("--tag", required=True)
    p.add_argument("--repo", default=DEFAULT_REPO)
    a = ap.parse_args(argv)
    return {"init": cmd_init, "sign": cmd_sign, "verify": cmd_verify}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
