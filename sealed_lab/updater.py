"""In-app update check: find the newest GitHub release, verify its signed manifest, download and check the zip, stage it and hand
over to apply_update.ps1 (Windows) or apply_update.sh (Linux).

Pure Python (no Qt) so it can be tested without a GUI and without a network: `check` and `download` take a `session` (anything with
`.get`, like `requests.Session`) and `launch_apply` takes the `Popen`.  Nothing here raises for network or HTTP problems in `check`: they
come back in `UpdateInfo.error`, because an automatic check must never become a crash or a popup.

Security (see release_sig):
  * Every request is anonymous: the repository is public, so no token is read, stored or sent (older versions looked for one).
  * Nothing from a release is extracted or run unless its `sealed_lab-X.Y.Z.manifest.json` carries a valid Ed25519 signature by a key in
    `release_keys.TRUSTED_KEYS` (the private key never leaves the maintainer's computer), the signed payload names exactly this repo, tag
    and file, and the downloaded zip has the signed size and SHA-256 (and GitHub's own asset digest, when GitHub reports one).
  * Only GitHub API urls of this repository are fetched; downloads are capped at the signed size; extraction rejects unsafe paths,
    links and archive bombs.
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import stat
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from sealed_lab import __version__, release_sig
from sealed_lab.config import app_home, user_agent

DEFAULT_REPO = "DuckPato/sealed_lab"
API = "https://api.github.com"
TIMEOUT = 15
MAX_UNPACKED_BYTES = 400 * 1024 * 1024
MAX_MEMBERS = 5000


def repo_name() -> str:
    return os.environ.get("SEALED_LAB_UPDATE_REPO", "").strip() or DEFAULT_REPO


def parse_version(s: str | None) -> tuple[int, ...] | None:
    """"v0.2.0" / "0.2.0" -> (0, 2, 0); anything that is not dotted integers (after an optional v) -> None.  A suffix such as
    "-rc1" is not a release we offer, so it is junk too."""
    if not s:
        return None
    m = re.fullmatch(r"\s*[vV]?(\d+(?:\.\d+)*)\s*", str(s))
    return tuple(int(p) for p in m.group(1).split(".")) if m else None


def is_newer(remote: str | None, local: str | None) -> bool:
    """True when `remote` is a strictly higher version than `local` (0.2 == 0.2.0); unparseable on either side -> False."""
    r, l = parse_version(remote), parse_version(local)
    if r is None or l is None:
        return False
    n = max(len(r), len(l))
    return r + (0,) * (n - len(r)) > l + (0,) * (n - len(l))


def _plain_version(tag: str) -> str:
    return ".".join(map(str, parse_version(tag) or ()))


@dataclass
class UpdateInfo:
    available: bool = False
    current: str = ""
    latest: str = ""
    notes: str = ""
    html_url: str = ""
    asset_name: str = ""
    asset_url: str = ""
    asset_size: int = 0
    asset_digest: str = ""            # GitHub's "sha256:<hex>" of the asset ("" when GitHub does not report one)
    manifest_url: str = ""            # the signed manifest ("" = the release is not signed: no self-update)
    error: str = ""

    @property
    def signed(self) -> bool:
        return bool(self.manifest_url)


def _headers(accept: str = "application/vnd.github+json") -> dict[str, str]:
    """Anonymous on purpose: no Authorization header, ever."""
    return {"Accept": accept, "X-GitHub-Api-Version": "2022-11-28", "User-Agent": user_agent()}


def _http_error(status: int) -> str:
    if status == 404:
        return "GitHub says the release was not found (the repository has no published release yet)."
    if status in (403, 429):
        return "GitHub refused the request (probably its hourly limit for anonymous requests): try again later."
    return f"GitHub answered with HTTP {status}."


def _session(session: Any) -> Any:
    if session is not None:
        return session
    import requests

    return requests.Session()


def _api_url(url: Any, repo: str) -> str:
    """Only GitHub API urls of this repository are fetched (the release JSON could otherwise point a download anywhere)."""
    url = str(url or "")
    return url if url.startswith(f"{API}/repos/{repo}/") else ""


def release_page(info: UpdateInfo, repo: str | None = None) -> str:
    """The page to open in a browser: GitHub's release page of this repository only, else the repository's releases list."""
    base = f"https://github.com/{repo or repo_name()}/releases"
    url = str(info.html_url or "")
    return url if url.startswith(base + "/") else base


def check(session: Any = None, repo: str | None = None, current: str | None = None) -> UpdateInfo:
    """Ask GitHub (anonymously) for the latest release and compare it with `current` (default: this app's version).  Never raises."""
    current = current or __version__
    repo = repo or repo_name()
    info = UpdateInfo(current=current)
    try:
        r = _session(session).get(f"{API}/repos/{repo}/releases/latest", headers=_headers(), timeout=TIMEOUT)
    except Exception as exc:  # noqa: BLE001 - requests.RequestException and anything a fake session raises
        info.error = f"Could not reach GitHub ({type(exc).__name__}). Check the internet connection."
        return info
    if r.status_code != 200:
        info.error = _http_error(r.status_code)
        return info
    try:
        data = r.json()
    except ValueError:
        data = None
    if not isinstance(data, dict):
        info.error = "GitHub's answer was not understood."
        return info
    info.latest = str(data.get("tag_name") or "").strip()
    info.notes = str(data.get("body") or "")
    info.html_url = str(data.get("html_url") or "")
    if parse_version(info.latest) is None:
        info.error = f"Release tag {info.latest!r} is not a version number."
        return info
    version = _plain_version(info.latest)
    assets = [a for a in data.get("assets") or [] if isinstance(a, dict)]
    asset = next((a for a in assets if a.get("name") == release_sig.zip_name(version)), None)
    if asset is None:
        info.error = f"Release {info.latest} has no {release_sig.zip_name(version)} file attached."
        return info
    info.asset_name, info.asset_url = str(asset["name"]), _api_url(asset.get("url"), repo)
    try:
        info.asset_size = int(asset.get("size") or 0)
    except (TypeError, ValueError):
        info.asset_size = 0
    digest = str(asset.get("digest") or "")
    info.asset_digest = digest.lower() if re.fullmatch(r"sha256:[0-9a-fA-F]{64}", digest) else ""
    man = next((a for a in assets if a.get("name") == release_sig.manifest_name(version)), None)
    info.manifest_url = _api_url(man.get("url"), repo) if man else ""
    if not info.asset_url:
        info.error = f"Release {info.latest} points its download outside the sealed_lab repository."
        return info
    info.available = is_newer(info.latest, current)
    return info


def _validate_zip(path: Path) -> None:
    """A usable release zip has pyproject.toml and sealed_lab/__init__.py one level down (or at the top of a flat zip)."""
    try:
        with zipfile.ZipFile(path) as z:
            names = [n.replace("\\", "/") for n in z.namelist()]
    except zipfile.BadZipFile as exc:
        raise ValueError("The downloaded file is not a zip archive.") from exc
    ok_pkg = any(n == "sealed_lab/__init__.py" or n.endswith("/sealed_lab/__init__.py") for n in names)
    ok_proj = any(n == "pyproject.toml" or n.endswith("/pyproject.toml") for n in names)
    if not (ok_pkg and ok_proj):
        raise ValueError("The downloaded zip does not look like a sealed_lab release (no pyproject.toml / sealed_lab package).")


def _get(session: Any, url: str, accept: str) -> Any:
    try:
        return _session(session).get(url, headers=_headers(accept), stream=True, timeout=TIMEOUT)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Could not reach GitHub ({type(exc).__name__}). Check the internet connection.") from exc


def _close(r: Any) -> None:
    close = getattr(r, "close", None)
    if close:
        close()


def fetch_manifest(info: UpdateInfo, session: Any = None, repo: str | None = None, trusted: dict[str, str] | None = None) -> dict:
    """The verified, signed payload of `info`'s release; raises release_sig.ManifestError / RuntimeError."""
    repo = repo or repo_name()
    if not info.manifest_url:
        raise release_sig.ManifestError(f"Release {info.latest} is not signed, so it cannot be installed from the app.")
    r = _get(session, info.manifest_url, "application/octet-stream")
    try:
        if r.status_code != 200:
            raise RuntimeError(_http_error(r.status_code))
        data = b""
        for chunk in r.iter_content(chunk_size=16 * 1024):
            data += chunk or b""
            if len(data) > release_sig.MAX_MANIFEST_BYTES:
                raise release_sig.ManifestError("The manifest is too large.")
    finally:
        _close(r)
    payload = release_sig.verify_manifest(data, trusted)
    release_sig.check_payload(payload, repo=repo, tag="v" + _plain_version(info.latest), asset=info.asset_name)
    if info.asset_size and info.asset_size != payload["size"]:
        raise release_sig.ManifestError("GitHub reports a different file size than the signed manifest.")
    if info.asset_digest and info.asset_digest != "sha256:" + payload["sha256"]:
        raise release_sig.ManifestError("GitHub reports a different file hash than the signed manifest.")
    return payload


def download(info: UpdateInfo, dest_dir: Path, session: Any = None, progress: Callable[[float, str], None] | None = None,
             cancel: Any = None, repo: str | None = None, trusted: dict[str, str] | None = None) -> tuple[Path, dict]:
    """Verify the release's signed manifest, then stream the zip into `dest_dir` (as `<name>.part`, renamed only when its size and
    SHA-256 match the signed ones and it is a sealed_lab zip).  Returns (zip path, signed payload).

    Raises release_sig.ManifestError (unsigned / bad signature / mismatch), RuntimeError (network, HTTP), ValueError (not a sealed_lab
    zip) and sealed_lab.types.Cancelled when `cancel` (a threading.Event) is set.  A file that fails a check is deleted.
    """
    from sealed_lab.types import Cancelled

    payload = fetch_manifest(info, session, repo, trusted)
    size, want = payload["size"], payload["sha256"]
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    final = dest_dir / payload["asset"]
    part = final.with_name(final.name + ".part")
    r = _get(session, info.asset_url, "application/octet-stream")
    try:
        if r.status_code != 200:
            raise RuntimeError(_http_error(r.status_code))
        h = hashlib.sha256()
        done = 0
        try:
            with open(part, "wb") as fh:
                for chunk in r.iter_content(chunk_size=64 * 1024):
                    if cancel is not None and cancel.is_set():
                        raise Cancelled()
                    if not chunk:
                        continue
                    done += len(chunk)
                    if done > size:
                        raise release_sig.ManifestError("The download is larger than the signed manifest says.")
                    h.update(chunk)
                    fh.write(chunk)
                    if progress:
                        progress(min(done / size, 1.0), f"Downloading {payload['asset']} ({done / 1e6:.1f} MB)")
            if done != size:
                raise release_sig.ManifestError("The download is incomplete or has the wrong size.")
            if h.hexdigest() != want:
                raise release_sig.ManifestError("The downloaded file does not match the signed SHA-256: it was changed or damaged.")
            _validate_zip(part)
        except BaseException:
            part.unlink(missing_ok=True)
            raise
    finally:
        _close(r)
    final.unlink(missing_ok=True)
    part.replace(final)
    return final, payload


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def staging_root() -> Path:
    return app_home() / "update"


def stage(zip_path: Path, version: str, sha256: str | None = None) -> Path:
    """Extract the zip to app_home()/update/staged/<version> (emptied first) and return the folder holding pyproject.toml.
    `sha256`: the signed hash, checked again right before extracting.  Every member is checked first: no absolute / drive / `..` paths,
    no links, at most MAX_MEMBERS entries and MAX_UNPACKED_BYTES in total (an archive bomb)."""
    if sha256 is not None and sha256_file(Path(zip_path)) != sha256.lower():
        raise ValueError("The update file changed after it was checked.")
    dest = staging_root() / "staged" / (re.sub(r"[^0-9A-Za-z._-]", "_", version.lstrip("vV")) or "new")
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    with zipfile.ZipFile(zip_path) as z:
        infos = z.infolist()
        if len(infos) > MAX_MEMBERS or sum(m.file_size for m in infos) > MAX_UNPACKED_BYTES:
            raise ValueError("The update zip is far larger than a sealed_lab release.")
        for member in infos:                              # check every member before extracting any (zip-slip)
            name = member.filename.replace("\\", "/")
            if name.startswith("/") or re.match(r"^[A-Za-z]:", name) or ".." in name.split("/"):
                raise ValueError(f"Unsafe path in the update zip: {member.filename!r}")
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError(f"Link in the update zip: {member.filename!r}")
            target = (root / name).resolve()
            if root != target and root not in target.parents:
                raise ValueError(f"Unsafe path in the update zip: {member.filename!r}")
        z.extractall(root)
    if (root / "pyproject.toml").is_file():
        return root
    for p in sorted(root.iterdir()):
        if p.is_dir() and (p / "pyproject.toml").is_file():
            return p
    raise ValueError("The update zip has no pyproject.toml.")


def app_dir() -> Path:
    """The app root: the folder that holds the sealed_lab package (and pyproject.toml, run_windows.bat / run_linux.sh)."""
    return Path(__file__).resolve().parent.parent


def can_self_update(folder: Path | None = None) -> tuple[bool, str]:
    """(True, "") when this installation may be replaced in place; else (False, why).  A git checkout is a developer's and is
    updated with git; an install without its launcher files or without write access cannot be replaced safely."""
    folder = Path(folder) if folder is not None else app_dir()
    for p in (folder, *folder.parents):
        if (p / ".git").exists():
            return False, f"This copy of sealed_lab is a git checkout ({p}); update it with git instead."
    launcher = "run_windows.bat" if sys.platform == "win32" else "run_linux.sh"
    for name in ("pyproject.toml", launcher):
        if not (folder / name).is_file():
            return False, f"{name} is missing from {folder}, so this does not look like a normal installation."
    if not os.access(folder, os.W_OK):
        return False, f"The folder {folder} is not writable."
    return True, ""


def launch_apply(staged: Path, folder: Path, pid: int, popen: Callable[..., Any] | None = None) -> list[str]:
    """Start the NEW version's apply script (apply_update.ps1 on Windows, apply_update.sh elsewhere) from the staged folder, detached;
    it waits for `pid` to exit, copies the files over `folder`, syncs the environment and relaunches the app.  Returns the command line."""
    staged, folder = Path(staged), Path(folder)
    kwargs: dict[str, Any] = {"close_fds": True}
    if sys.platform == "win32":
        script = staging_root() / "apply_update.ps1"
        script.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(staged / "apply_update.ps1", script)
        cmd = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script),
               "-Source", str(staged), "-AppDir", str(folder), "-WaitPid", str(pid)]
        # its own console window (not DETACHED_PROCESS): the user sees "Updating sealed_lab..." while uv syncs, then it closes
        kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        script = staging_root() / "apply_update.sh"
        script.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(staged / "apply_update.sh", script)
        cmd = ["bash", str(script), "--source", str(staged), "--app-dir", str(folder), "--wait-pid", str(pid)]
        kwargs.update(start_new_session=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    (popen or subprocess.Popen)(cmd, **kwargs)
    return cmd


def due(last_check_iso: str | None, now: datetime, hours: float = 24) -> bool:
    """Once-a-day automatic check: True when there is no (readable) last check or it is at least `hours` old."""
    if not last_check_iso:
        return True
    try:
        last = datetime.fromisoformat(str(last_check_iso))
    except ValueError:
        return True
    if last.tzinfo is not None and now.tzinfo is None:
        last = last.replace(tzinfo=None)
    elif last.tzinfo is None and now.tzinfo is not None:
        last = last.replace(tzinfo=now.tzinfo)
    return now - last >= timedelta(hours=hours)
