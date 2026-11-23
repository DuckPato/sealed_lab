"""In-app update check: find the newest GitHub release, download its zip, stage it and hand over to apply_update.ps1 (Windows) or apply_update.sh (Linux).

Pure Python (no Qt) so it can be tested without a GUI and without a network: `check` and `download` take a `session` (anything with
`.get`, like `requests.Session`), `find_token` and `launch_apply` take the `gh` runner / `Popen` as arguments.  Nothing here raises for
network or HTTP problems in `check`: they come back in `UpdateInfo.error`, because an automatic check must never become a crash or a popup.

The repo is public, so no token is needed; every call can still carry one (env, a stored one, or the one `gh auth login` left behind).  The asset is fetched
through the API url with `Accept: application/octet-stream`; GitHub answers with a redirect to its storage host, and `requests` drops the
Authorization header when a redirect changes host, which is exactly what the storage host needs (it rejects a bearer token).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from sealed_lab import __version__
from sealed_lab.config import app_home

DEFAULT_REPO = "DuckPato/sealed_lab"
API = "https://api.github.com"
TIMEOUT = 15
ASSET_RE = re.compile(r"^sealed_lab-.*\.zip$", re.I)
SIGN_IN_HELP = "Sign in to GitHub with `gh auth login`, or paste a token in Help > GitHub sign-in token."


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


def _gh_auth_token() -> str | None:
    """`gh auth token`: the token of an installed, signed-in GitHub CLI.  Any failure (no gh, not signed in, hang) -> None."""
    try:
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        r = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, timeout=5, creationflags=flags)
    except (OSError, subprocess.SubprocessError):
        return None
    out = (r.stdout or "").strip()
    return out if r.returncode == 0 and out else None


def find_token(stored: str | None = None, gh: Callable[[], str | None] | None = None) -> str | None:
    """First non-empty of: env SEALED_LAB_GITHUB_TOKEN, GH_TOKEN, GITHUB_TOKEN, the `stored` one (QSettings), then `gh auth token`."""
    for name in ("SEALED_LAB_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"):
        v = os.environ.get(name, "").strip()
        if v:
            return v
    if stored and stored.strip():
        return stored.strip()
    return ((gh or _gh_auth_token)() or "").strip() or None


@dataclass
class UpdateInfo:
    available: bool = False
    current: str = ""
    latest: str = ""
    notes: str = ""
    html_url: str = ""
    asset_name: str = ""
    asset_url: str = ""
    error: str = ""


def _headers(token: str | None, accept: str = "application/vnd.github+json") -> dict[str, str]:
    h = {"Accept": accept, "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "sealed_lab-updater"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _http_error(status: int, token: str | None) -> str:
    if status == 404:
        if token:
            return "GitHub says the release was not found: the repository has no release yet, or your token has no access to it."
        return "GitHub says the repository was not found. If it has been made private, you need to sign in. " + SIGN_IN_HELP
    if status in (401, 403):
        return "GitHub refused the request: the token is invalid or expired, or it has no access to the repository (or the rate limit is hit)."
    return f"GitHub answered with HTTP {status}."


def _session(session: Any) -> Any:
    if session is not None:
        return session
    import requests

    return requests.Session()


def check(session: Any = None, token: str | None = None, repo: str | None = None, current: str | None = None) -> UpdateInfo:
    """Ask GitHub for the latest release and compare it with `current` (default: this app's version).  Never raises."""
    current = current or __version__
    info = UpdateInfo(current=current)
    try:
        r = _session(session).get(f"{API}/repos/{repo or repo_name()}/releases/latest", headers=_headers(token), timeout=TIMEOUT)
    except Exception as exc:  # noqa: BLE001 - requests.RequestException and anything a fake session raises
        info.error = f"Could not reach GitHub ({type(exc).__name__}). Check the internet connection."
        return info
    if r.status_code != 200:
        info.error = _http_error(r.status_code, token)
        return info
    try:
        data = r.json()
    except ValueError:
        info.error = "GitHub's answer was not understood."
        return info
    info.latest = str(data.get("tag_name") or "").strip()
    info.notes = str(data.get("body") or "")
    info.html_url = str(data.get("html_url") or "")
    asset = next((a for a in data.get("assets") or [] if ASSET_RE.match(str(a.get("name", "")))), None)
    if asset is None:
        info.error = f"Release {info.latest or '?'} has no sealed_lab-*.zip file attached."
        return info
    info.asset_name, info.asset_url = str(asset["name"]), str(asset.get("url") or "")
    if parse_version(info.latest) is None:
        info.error = f"Release tag {info.latest!r} is not a version number."
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


def download(info: UpdateInfo, dest_dir: Path, session: Any = None, token: str | None = None,
             progress: Callable[[float, str], None] | None = None, cancel: Any = None) -> Path:
    """Stream the release asset into `dest_dir` (as `<name>.part`, renamed when complete and validated).

    Raises RuntimeError with a friendly text on HTTP problems, ValueError when the file is not a sealed_lab zip, and
    sealed_lab.types.Cancelled when `cancel` (a threading.Event) is set.
    """
    from sealed_lab.types import Cancelled

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    final = dest_dir / (info.asset_name or "sealed_lab.zip")
    part = final.with_name(final.name + ".part")
    try:
        r = _session(session).get(info.asset_url, headers=_headers(token, "application/octet-stream"), stream=True, timeout=TIMEOUT)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Could not reach GitHub ({type(exc).__name__}). Check the internet connection.") from exc
    try:
        if r.status_code != 200:
            raise RuntimeError(_http_error(r.status_code, token))
        total = int((getattr(r, "headers", {}) or {}).get("Content-Length") or 0)
        done = 0
        try:
            with open(part, "wb") as fh:
                for chunk in r.iter_content(chunk_size=64 * 1024):
                    if cancel is not None and cancel.is_set():
                        raise Cancelled()
                    if not chunk:
                        continue
                    fh.write(chunk)
                    done += len(chunk)
                    if progress:
                        progress(min(done / total, 1.0) if total else 0.0, f"Downloading {info.asset_name} ({done / 1e6:.1f} MB)")
        except BaseException:
            part.unlink(missing_ok=True)
            raise
    finally:
        close = getattr(r, "close", None)
        if close:
            close()
    try:
        _validate_zip(part)
    except ValueError:
        part.unlink(missing_ok=True)
        raise
    final.unlink(missing_ok=True)
    part.replace(final)
    return final


def staging_root() -> Path:
    return app_home() / "update"


def stage(zip_path: Path, version: str) -> Path:
    """Extract the zip to app_home()/update/staged/<version> (emptied first) and return the folder holding pyproject.toml."""
    dest = staging_root() / "staged" / (re.sub(r"[^0-9A-Za-z._-]", "_", version.lstrip("vV")) or "new")
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    with zipfile.ZipFile(zip_path) as z:
        for member in z.infolist():                       # check every member before extracting any (zip-slip)
            name = member.filename.replace("\\", "/")
            if name.startswith("/") or re.match(r"^[A-Za-z]:", name) or ".." in name.split("/"):
                raise ValueError(f"Unsafe path in the update zip: {member.filename!r}")
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
