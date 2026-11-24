"""Ready-made sets shipped inside the release zip (`<app folder>/bundled_sets/`), installed into the app home at start.

The maintainer's calculated newest sets travel in the release zip (built by `tools/bundle_sets.py`, merged into the signed zip by
`tools/release_sign.py sign --bundle`).  At the next start `install_bundled()` moves them into `<app home>/sets/<SET>/` and deletes the
previous ready-made versions.  A set the user calculated from files of their own is never touched.

Bundle layout (`bundled_sets/`):
    bundle.json                  {"bundle_version", "created_at", "sets": {CODE: {"built_at", "release_date"}}}
    <CODE>/model/*  corpus/*.parquet + manifest.json (paths reduced to file names)  cache/report_*.json ...  written/<CODE>.md|json

An installed ready-made set carries `sets/<CODE>/bundled.json` ({built_at, bundle_version, installed_at, complete, written}) and a
registry with no files (`registry.ensure_tracked`).  The marker goes away when the user calculates the set from their own files.
"""
from __future__ import annotations

import contextlib
import json
import logging
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from sealed_lab import config
from sealed_lab.ingest import registry
from sealed_lab.types import SetBusy

log = logging.getLogger(__name__)

BUNDLE_DIR = "bundled_sets"
BUNDLE_FILE = "bundle.json"
MARKER = "bundled.json"
CACHE_PATTERNS = ("report_*.json", "scryfall.json", "creature_support*.json", "archetype_proj_*.json")   # what the tabs need at once
_ABS_PATH = re.compile(r"(?:[A-Za-z]:[\\/]|\\\\|/(?:home|Users|mnt|tmp|var|media|opt|root)/)[^\s\"',;)\]]*")


def app_folder() -> Path:
    """The folder that holds the sealed_lab package (and, in a fresh release, `bundled_sets/`)."""
    return Path(__file__).resolve().parent.parent


def bundle_root(app_dir: str | Path | None = None) -> Path:
    return (Path(app_dir) if app_dir is not None else app_folder()) / BUNDLE_DIR


def _read_json(p: Path) -> dict | None:
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) else None


# ---------------------------------------------------------------------------------------------
# the marker of an installed ready-made set
# ---------------------------------------------------------------------------------------------

def marker_path(set_code: str) -> Path:
    return config.set_dir(set_code) / MARKER


def read_marker(set_code: str) -> dict | None:
    """The marker of a ready-made set, None for a set that is not one (or whose installation did not finish)."""
    try:
        d = _read_json(marker_path(set_code))
    except ValueError:
        return None
    return d if d is not None and d.get("complete", True) else None


def clear_marker(set_code: str) -> None:
    """The set is the user's own from now on (called after a successful calculation)."""
    with contextlib.suppress(OSError, ValueError):
        marker_path(set_code).unlink(missing_ok=True)


def set_note(set_code: str) -> str:
    """"ready-made model from sealed_lab 0.4.0 (built 2026-11-10)" for a ready-made set, else ""."""
    mk = read_marker(set_code)
    if mk is None:
        return ""
    built = str(mk.get("built_at") or "")[:10]
    return f"ready-made model from sealed_lab {mk.get('bundle_version', '?')}" + (f" (built {built})" if built else "")


# ---------------------------------------------------------------------------------------------
# sanitising (used when the bundle is built; shared so the rule lives in one place)
# ---------------------------------------------------------------------------------------------

def _scrub(text: str) -> str:
    return _ABS_PATH.sub(lambda m: re.split(r"[\\/]", m.group(0).rstrip("\\/"))[-1], text)


def sanitize_manifest(man):
    """A copy of a corpus manifest with every path reduced to its file name: the value of any key whose name is or ends in `path` /
    `dir` / `folder`, and any absolute path inside a string (warnings, notes).  Fingerprints, kinds, events and times stay."""
    if isinstance(man, dict):
        out = {}
        for k, v in man.items():
            lk = str(k).lower()
            if isinstance(v, str) and (lk in ("path", "dir", "folder") or lk.endswith(("_path", "_dir", "_folder"))):
                out[k] = re.split(r"[\\/]", v.rstrip("\\/"))[-1]
            else:
                out[k] = sanitize_manifest(v)
        return out
    if isinstance(man, list):
        return [sanitize_manifest(v) for v in man]
    if isinstance(man, str):
        return _scrub(man)
    return man


# ---------------------------------------------------------------------------------------------
# installation
# ---------------------------------------------------------------------------------------------

def _has_own_files(code: str) -> bool:
    return bool(registry.files(code, include_shared=False))


def _is_own(code: str) -> bool:
    """A set the user made: files registered, or a model / corpus there that is not a ready-made one."""
    sd = config.set_dir(code)
    if _has_own_files(code):
        return True
    if (sd / MARKER).exists():
        return False
    return (sd / "model" / "meta.json").exists() or (sd / "corpus" / "manifest.json").exists()


def _wipe_set(code: str) -> None:
    """Delete everything of the set's folder except the calculation lock this process holds."""
    sd = config.set_dir(code)
    if not sd.exists():
        return
    for p in sd.iterdir():
        if p.name == "calc.lock":
            continue
        if p.is_dir() and not p.is_symlink():
            shutil.rmtree(p)
        else:
            p.unlink()


def _written_dir() -> Path:
    from sealed_lab import written_report

    return written_report.folder()


def _delete_written(code: str) -> None:
    for ext in ("md", "json"):
        with contextlib.suppress(OSError):
            (_written_dir() / f"{code}.{ext}").unlink(missing_ok=True)


def _install_one(code: str, src: Path, info: dict, version: str) -> str:
    """Install one bundled set; returns "installed", "current" or "own" (skipped)."""
    if _is_own(code):
        log.info("ready-made set %s skipped: the set was calculated from your own files", code)
        return "own"
    old = _read_json(marker_path(code)) if marker_path(code).exists() else None
    if old and old.get("complete", True) and old.get("built_at") == info.get("built_at"):
        return "current"
    if not (src / "model" / "meta.json").is_file() or not (src / "corpus" / "manifest.json").is_file():
        raise FileNotFoundError(f"the bundled set {code} is incomplete")
    had_written = bool(old and old.get("written"))
    own_written = (_written_dir() / f"{code}.md").exists() and not had_written
    with registry.calc_lock(code):
        _wipe_set(code)
        sd = config.set_dir(code)
        sd.mkdir(parents=True, exist_ok=True)
        base = dict(built_at=info.get("built_at"), bundle_version=version, installed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        registry.write_text_atomic(sd / MARKER, json.dumps(dict(base, complete=False)))       # an interrupted install is never taken for the user's own
        for sub in ("model", "corpus", "cache"):
            if (src / sub).is_dir():
                shutil.move(str(src / sub), str(sd / sub))
            else:
                (sd / sub).mkdir(exist_ok=True)
        registry.ensure_tracked(code)
        written = False
        w_md, w_js = src / "written" / f"{code}.md", src / "written" / f"{code}.json"
        if had_written:
            _delete_written(code)                                              # the previous ready-made text
        if w_md.is_file() and w_js.is_file() and not own_written:
            wd = _written_dir()
            wd.mkdir(parents=True, exist_ok=True)
            shutil.move(str(w_md), str(wd / f"{code}.md"))
            shutil.move(str(w_js), str(wd / f"{code}.json"))
            written = True
        registry.write_text_atomic(sd / MARKER, json.dumps(dict(base, complete=True, written=written)))
    return "installed"


def _drop_stale(bundle_codes: set[str]) -> list[str]:
    """Delete ready-made sets that the new bundle no longer carries (a set the user has registered files for only loses its marker)."""
    dropped: list[str] = []
    root = config.sets_dir()
    if not root.is_dir():
        return dropped
    for p in sorted(root.iterdir()):
        code = p.name
        if not p.is_dir() or code in bundle_codes or config.valid_set_code(code) != code or not (p / MARKER).exists():
            continue
        mk = _read_json(p / MARKER) or {}
        try:
            with registry.calc_lock(code):
                if _has_own_files(code):
                    (p / MARKER).unlink(missing_ok=True)
                    log.info("ready-made set %s kept: you registered files for it", code)
                    continue
                if mk.get("written"):
                    _delete_written(code)
                shutil.rmtree(p)
                dropped.append(code)
                log.info("ready-made set %s deleted (not in this release's bundle)", code)
        except SetBusy:
            log.info("ready-made set %s is being calculated; it is dealt with at the next start", code)
    return dropped


def install_bundled(app_dir: str | Path | None = None) -> dict:
    """Install the sets of `<app folder>/bundled_sets/` into the app home, delete the previous ready-made sets and, when every set
    went in, the bundle folder.  Returns {"installed", "current", "own", "dropped", "errors", "busy"} (lists of set codes / messages).
    Does nothing without a bundle; a failure of one set is logged and kept in "errors" (the bundle then stays for the next start)."""
    root = bundle_root(app_dir)
    res: dict = dict(installed=[], current=[], own=[], dropped=[], errors=[], busy=[])
    bj = _read_json(root / BUNDLE_FILE)
    if bj is None:
        if root.exists():
            log.warning("%s has no readable %s; ignored", root, BUNDLE_FILE)
        return res
    sets = bj.get("sets") if isinstance(bj.get("sets"), dict) else {}
    version = str(bj.get("bundle_version", "?"))
    codes = {c for c in (config.valid_set_code(k) for k in sets) if c}
    for raw, info in sets.items():
        code = config.valid_set_code(raw)
        if code is None or not isinstance(info, dict) or not (root / raw).is_dir():
            if code is not None and not (root / raw).is_dir():
                res["errors"].append(f"{raw}: folder missing in the bundle")
            continue
        try:
            res[_install_one(code, root / raw, info, version)].append(code)
        except SetBusy:
            res["busy"].append(code)
            log.info("ready-made set %s is being calculated; installed at the next start", code)
        except Exception as e:  # noqa: BLE001 - one set must not stop the others
            res["errors"].append(f"{code}: {e}")
            log.exception("could not install the ready-made set %s", code)
    try:
        res["dropped"] = _drop_stale(codes)
    except Exception as e:  # noqa: BLE001
        res["errors"].append(f"dropping old ready-made sets: {e}")
        log.exception("could not delete old ready-made sets")
    if not res["errors"] and not res["busy"]:
        try:
            shutil.rmtree(root)
        except OSError as e:
            log.warning("could not delete %s (%s); it is deleted at the next start", root, e)
    if res["installed"] or res["dropped"] or res["errors"]:
        log.info("ready-made sets: %s", res)
    return res
