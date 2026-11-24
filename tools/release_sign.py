"""Release signing, run by the maintainer on their own computer (never in CI: the private key must not reach GitHub).

    python tools/release_sign.py init [--key PATH] [--no-passphrase]
        Create the Ed25519 release key (private key: PATH, default ~/.sealed_lab_release/release_ed25519.pem, passphrase-protected
        unless --no-passphrase) and add its public half to sealed_lab/release_keys.py.  Back the private key up somewhere safe: without
        it, installed copies cannot be updated from the app any more (a fresh download still works).

    python tools/release_sign.py sign --tag vX.Y.Z --public PATH_TO_PUBLIC_CLONE [--key PATH] [--no-publish] [--bundle DIR]
        For the DRAFT release that CI built for the tag: download its zip, rebuild the zip from a fresh checkout of the tag (a temporary
        worktree of the public clone) and require the two to be identical (same files, bytes and modes), then sign the manifest, attach it
        and publish the release.
        --bundle DIR (made by tools/bundle_sets.py): after that comparison passes, a new zip is written = the CI zip's entries unchanged
        + the ready-made sets as `bundled_sets/...` entries; THAT zip is signed and replaces the CI asset.  The bundle is data from this
        computer: it cannot be rebuilt from source, only the signature covers it.

    python tools/release_sign.py verify --tag vX.Y.Z [--public PATH_TO_PUBLIC_CLONE]
        Check a published release exactly as the app's updater does (manifest signature, repo / tag / file, size, SHA-256); with --public
        also compare every entry except `bundled_sets/` with the tag's rebuild.  Reports the ready-made sets the zip carries.

Needs the GitHub CLI (`gh`, signed in) for sign; verify is anonymous.
"""
from __future__ import annotations

import argparse
import ast
import getpass
import hashlib
import json
import os
import re
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
    """TRUSTED_KEYS of a release_keys.py, read as data (the file is never executed: it may come from a tag checkout)."""
    tree = ast.parse(Path(keys_file).read_text(encoding="utf-8"))
    for node in tree.body:
        target = node.target if isinstance(node, ast.AnnAssign) else (node.targets[0] if isinstance(node, ast.Assign) else None)
        if isinstance(target, ast.Name) and target.id == "TRUSTED_KEYS" and node.value is not None:
            return dict(ast.literal_eval(node.value))
    return {}


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
BUNDLE_DIR = "bundled_sets"
BUNDLE_TIME = (2020, 1, 1, 0, 0, 0)                     # fixed time of the bundle's entries: the same bundle gives the same zip


def is_bundle_entry(name: str) -> bool:
    """`sealed_lab-X.Y.Z/bundled_sets/...` (the ready-made sets: data from the maintainer's computer, not rebuilt from source)."""
    return "/" in name and name.split("/", 1)[1].startswith(BUNDLE_DIR + "/")


def zip_entries(path: Path, skip_bundle: bool = False) -> dict[str, tuple[int, bytes]]:
    """name -> (unix mode, content) of every file entry (`skip_bundle`: without the bundled_sets/ entries, which are not even read)."""
    with zipfile.ZipFile(path) as z:
        return {i.filename: ((i.external_attr >> 16) & 0o7777, z.read(i)) for i in z.infolist()
                if not i.is_dir() and not (skip_bundle and is_bundle_entry(i.filename))}


def zip_dirs(path: Path, skip_bundle: bool = False) -> dict[str, int]:
    """name -> unix mode of every directory entry."""
    with zipfile.ZipFile(path) as z:
        return {i.filename: (i.external_attr >> 16) & 0o7777 for i in z.infolist()
                if i.is_dir() and not (skip_bundle and is_bundle_entry(i.filename))}


def zip_duplicates(path: Path) -> list[str]:
    """Entry names that occur more than once (extractors differ on which one wins, so such a zip is never accepted)."""
    with zipfile.ZipFile(path) as z:
        names = [i.filename for i in z.infolist()]
    return sorted({n for n in names if names.count(n) > 1})


def compare_zips(built: Path, reference: Path, skip_bundle: bool = False) -> list[str]:
    """Differences between a zip and the reference rebuilt from source.  `skip_bundle`: ignore the `bundled_sets/` entries of `built`
    (the reference has none: they cannot be rebuilt)."""
    diffs = [f"duplicate entry in the release zip: {n}" for n in zip_duplicates(built)]
    diffs += [f"duplicate entry in the reference zip: {n}" for n in zip_duplicates(reference)]
    da, db = zip_dirs(built, skip_bundle), zip_dirs(reference)
    diffs += [f"directory only in the release zip: {n}" for n in sorted(set(da) - set(db))]
    diffs += [f"directory missing from the release zip: {n}" for n in sorted(set(db) - set(da))]
    diffs += [f"directory mode differs: {n}" for n in sorted(set(da) & set(db)) if da[n] != db[n]]
    a, b = zip_entries(built, skip_bundle), zip_entries(reference)
    diffs += [f"only in the release zip: {n}" for n in sorted(set(a) - set(b))]
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


def bundle_files(bundle: Path) -> list[Path]:
    """The files of a bundle folder, sorted by relative path (a deterministic zip); refuses anything but plain files and folders."""
    bundle = Path(bundle)
    if not (bundle / "bundle.json").is_file():
        _die(f"{bundle} has no bundle.json (build it with tools/bundle_sets.py)")
    out = []
    for p in sorted(bundle.rglob("*"), key=lambda q: q.relative_to(bundle).as_posix()):
        if p.is_symlink():
            _die(f"{p} is a link; a bundle holds plain files only")
        if p.is_file():
            out.append(p)
    return out


# the limits of the apps that shipped before ready-made sets (0.1.4 and older): a bigger zip is refused by them
OLD_MAX_ZIP, OLD_MAX_UNPACKED, OLD_MAX_MEMBERS = 200 * 1024 * 1024, 400 * 1024 * 1024, 5000


def merge_bundle(ci_zip: Path, bundle: Path, out_zip: Path) -> Path:
    """`out_zip` = the entries of `ci_zip` unchanged (any earlier bundled_sets/ entries dropped) + every file of `bundle` as
    `<top folder>/bundled_sets/<relative path>` (sorted, fixed time, mode 0644).  Streams the bundle: it can be a gigabyte."""
    files = bundle_files(bundle)
    with zipfile.ZipFile(ci_zip) as src:
        infos = [i for i in src.infolist() if not is_bundle_entry(i.filename)]
        tops = {i.filename.split("/", 1)[0] for i in infos}
        if len(tops) != 1:
            _die(f"{Path(ci_zip).name} should have one top folder, found {sorted(tops)}")
        top = tops.pop()
        out_zip.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as dst:
            for i in infos:
                ni = zipfile.ZipInfo(i.filename, i.date_time)
                ni.compress_type, ni.create_system, ni.external_attr = i.compress_type, i.create_system, i.external_attr
                dst.writestr(ni, src.read(i))
            for f in files:
                ni = zipfile.ZipInfo(f"{top}/{BUNDLE_DIR}/{f.relative_to(bundle).as_posix()}", BUNDLE_TIME)
                ni.compress_type, ni.create_system, ni.external_attr = zipfile.ZIP_DEFLATED, 3, 0o100644 << 16
                ni.file_size = f.stat().st_size
                with open(f, "rb") as fh, dst.open(ni, "w", force_zip64=ni.file_size > (1 << 30)) as w:
                    shutil.copyfileobj(fh, w, 1 << 20)
    return out_zip


def bundle_info(zip_path: Path) -> dict | None:
    """bundle.json of a release zip (None without a bundle)."""
    with zipfile.ZipFile(zip_path) as z:
        for i in z.infolist():
            if is_bundle_entry(i.filename) and i.filename.endswith("/bundle.json") and i.filename.count("/") == 2:
                return json.loads(z.read(i).decode("utf-8"))
    return None


def describe_bundle(info: dict | None) -> str:
    if not info:
        return "no ready-made sets in this zip"
    sets = ", ".join(f"{c} (built {str(v.get('built_at', '?'))[:10]})" for c, v in sorted((info.get("sets") or {}).items()))
    return (f"ready-made sets (bundle of sealed_lab {info.get('bundle_version', '?')}): {sets}. They were calculated on the maintainer's "
            "computer: covered by the signature, but data that cannot be rebuilt from source")


def reference_zip(public: Path, tag: str, td: Path, tagged: str | None = None) -> tuple[Path, dict[str, str]]:
    """(the zip the tag's source builds, the tag's trusted keys): built from a fresh worktree of the public clone by THIS repository's
    export_public (nothing from the tag is executed)."""
    tagged = tagged or _run(["git", "rev-parse", f"{tag}^{{commit}}"], public).strip()
    src = td / "src"
    _run(["git", "worktree", "add", "--detach", str(src), tagged], public)
    try:
        trusted = read_trusted(src / "sealed_lab" / "release_keys.py")
        ep = _dev_export_public()
        ep.build_zip(td / "ref", ep.selected(root=src), tag, root=src)
    finally:
        subprocess.run(["git", "worktree", "remove", "--force", str(src)], cwd=public, capture_output=True)
        subprocess.run(["git", "worktree", "prune"], cwd=public, capture_output=True)
    return td / "ref" / release_sig.zip_name(tag.lstrip("vV")), trusted


def _scan_bundle(bundle: Path) -> list[str]:
    """Personal strings in the bundle folder (tools/bundle_sets.py's scan, run again on what is about to be signed)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("_dev_bundle_sets", ROOT / "tools" / "bundle_sets.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.scan_folder(Path(bundle))


# --------------------------------------------------------------------------------------------- sign
def _dev_export_public():
    """tools/export_public.py of THIS (development) repository, imported by path."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("_dev_export_public", ROOT / "tools" / "export_public.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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
    with tempfile.TemporaryDirectory(prefix="sealed_lab_sign_") as td:
        td = Path(td)
        _run([gh, "release", "download", tag, "-R", repo, "-p", zname, "-D", str(td / "gh")])
        built = td / "gh" / zname
        # the reference zip is built by THIS repository's export_public (loaded from ROOT, in this process) from the tag's files:
        # nothing from the tag is executed, and the private key is not loaded until the comparison has passed
        ref, trusted = reference_zip(public, tag, td, tagged)
        if bundle_info(built) is not None and not a.bundle:
            _die("the zip on the draft already carries ready-made sets (an earlier sign --bundle): pass --bundle DIR again")
        diffs = compare_zips(built, ref, skip_bundle=True)             # a bundle from an earlier attempt is ignored; it is replaced below
        if diffs:
            _die("the zip on GitHub is not the one the tagged source builds:\n  " + "\n  ".join(diffs[:40]))
        print(f"{zname}: identical to the zip built from {tag} ({len(zip_entries(ref))} files)")
        if a.bundle:
            bdir = Path(a.bundle).resolve()
            hits = _scan_bundle(bdir)
            if hits:
                _die("personal strings in the bundle:\n  " + "\n  ".join(hits[:30]))
            built = merge_bundle(built, bdir, td / "merged" / zname)
            with zipfile.ZipFile(built) as zf:
                unpacked, members = sum(i.file_size for i in zf.infolist()), len(zf.infolist())
            if (built.stat().st_size > OLD_MAX_ZIP or unpacked > OLD_MAX_UNPACKED or members > OLD_MAX_MEMBERS) and not a.allow_large:
                _die(f"the zip with the bundle ({built.stat().st_size / 1e6:.0f} MB, {unpacked / 1e6:.0f} MB unpacked, {members} files) is over "
                     "what apps up to 0.1.4 accept (200 MB zip, 400 MB unpacked, 5000 files): they would refuse this update.  Release a "
                     "version with the new limits WITHOUT --bundle first; once friends run it, sign with --bundle --allow-large.")
            print(f"{zname}: + {len(bundle_files(bdir))} bundle files, {built.stat().st_size / 1e6:.1f} MB; " + describe_bundle(bundle_info(built)))
        priv = load_private(Path(a.key))                # only now: the key is decrypted after everything from GitHub has been checked
        kid = release_sig.key_id(_public_raw(priv))
        if kid not in trusted:
            _die(f"key {kid} is not in {tag}'s sealed_lab/release_keys.py: that version could not check the next update")
        payload = release_sig.make_payload(version, repo, zname, built.stat().st_size, sha256(built),
                                           datetime.now(timezone.utc).isoformat(timespec="seconds"))
        manifest = release_sig.sign_payload(payload, priv)
        release_sig.check_payload(release_sig.verify_manifest(manifest, trusted), repo=repo, tag=tag, asset=zname)
        mpath = td / mname
        mpath.write_bytes(manifest)
        if a.bundle:                                    # the zip that was signed replaces the CI asset (same name)
            _run([gh, "release", "upload", tag, str(built), "-R", repo, "--clobber"])
            print(f"{zname} with the ready-made sets uploaded in place of the CI zip")
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

    public = Path(a.public).resolve() if getattr(a, "public", None) else None
    info = updater.check(repo=a.repo, current="0")
    if info.error:
        _die(info.error)
    if info.latest != a.tag:
        _die(f"the latest published release is {info.latest}, not {a.tag}")
    trusted = read_trusted(KEYS_FILE)
    with tempfile.TemporaryDirectory(prefix="sealed_lab_verify_") as td:
        path, payload = updater.download(info, Path(td), repo=a.repo, trusted=trusted)
        print(f"OK: {info.latest} {path.name} {payload['size']} bytes, sha256 {payload['sha256'][:16]}..., signature valid")
        print(describe_bundle(bundle_info(path)))
        if public is not None:
            ref, _keys = reference_zip(public, a.tag, Path(td))
            diffs = compare_zips(path, ref, skip_bundle=True)
            if diffs:
                _die("the published zip differs from the tag's rebuild (bundled_sets/ aside):\n  " + "\n  ".join(diffs[:40]))
            print(f"every entry except bundled_sets/ is identical to the zip built from {a.tag}")
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
    p.add_argument("--bundle", help="folder made by tools/bundle_sets.py: ship its ready-made sets inside the signed zip")
    p.add_argument("--allow-large", action="store_true", help="the bundle may exceed the limits of apps up to 0.1.4 (only once a release "
                   "with the larger limits has shipped and friends run it)")
    p = sub.add_parser("verify")
    p.add_argument("--tag", required=True)
    p.add_argument("--public", help="public clone: also compare the zip (bundled_sets/ aside) with the tag's rebuild")
    p.add_argument("--repo", default=DEFAULT_REPO)
    a = ap.parse_args(argv)
    return {"init": cmd_init, "sign": cmd_sign, "verify": cmd_verify}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
