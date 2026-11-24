"""Tracked 17lands data folders: the app follows ONE data folder (default: where the owner's 17lands downloader saves, one
sub-folder per set code) and registers the 17lands files it finds there by itself.

* a set may point at its own folder (`set_folders.json` in the app home) instead of `<data folder>/<SET>`;
* `scan` registers new / changed files of the sets the app already knows (plus sets the user asks to track), skipping files whose
  size and mtime were already looked at (`folder_scan.json`), and reports registered files that disappeared (they stay in the
  registry, flagged missing: a model already built from them is not dropped);
* `untracked_sets` lists the sets that exist on disk (a sub-folder with dataset files) or on 17lands' list (the stored listing of
  `datasets_state.json`, no network) but are not tracked yet.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from sealed_lab import config
from sealed_lab.config import FILE_NAME_RE
from sealed_lab.ingest import registry
from sealed_lab.ingest.detect import detect_file
from sealed_lab.types import Cancelled, FileInfo

log = logging.getLogger(__name__)

_LOCK = threading.Lock()
_CARDS_RE = re.compile(r"^cards.*\.(csv|csv\.gz)$", re.IGNORECASE)
DATA_EXTS = (".csv", ".csv.gz")
# The scan registers every file of the events the model always uses (datasets.RELEVANT_EVENTS) and the GAME files of the extra draft
# events (PickTwoDraft, QuickDraft ...): the pipeline keeps those only if they pass their hold-out test (component "extra_draft"), so
# they cannot silently change a model's draft prior.  Their replay and draft-pick files feed statistics that only know the four core
# events, so they are registered only when the user drops them.
SCAN_EVENTS = ("Sealed", "TradSealed", "PremierDraft", "TradDraft")
SCAN_GAME_EVENTS = SCAN_EVENTS + config.EXTRA_DRAFT_EVENTS


# ---------------------------------------------------------------------------------------------
# per-set folders and the scan cache (small JSON files in the app home)
# ---------------------------------------------------------------------------------------------

def _folders_path() -> Path:
    return config.app_home() / "set_folders.json"


def _cache_path() -> Path:
    return config.app_home() / "folder_scan.json"


def _read_json(p: Path) -> dict:
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except OSError:
        return {}
    except ValueError:
        registry.quarantine(p)
        return {}


def _write_json(p: Path, d: dict) -> None:
    try:
        registry.write_text_atomic(p, json.dumps(d, indent=1))
    except OSError as e:  # pragma: no cover - a cache
        log.warning("could not write %s: %s", p, e)


def set_folders() -> dict[str, str]:
    """{SET: folder} of the sets that have their own folder instead of `<data folder>/<SET>`."""
    return {k.upper(): str(v) for k, v in (_read_json(_folders_path()).get("sets") or {}).items() if v}


def set_set_folder(set_code: str, path: str | None) -> None:
    """Give a set its own folder (None / "" removes the override).  The scan cache is cleared so the folder is looked at again."""
    with _LOCK:
        d = _read_json(_folders_path())
        sets = dict(d.get("sets") or {})
        if path:
            sets[set_code.upper()] = str(path)
        else:
            sets.pop(set_code.upper(), None)
        _write_json(_folders_path(), {"sets": sets})
        _write_json(_cache_path(), {"files": {}})


def set_folder(set_code: str, data_dir: str | None) -> Path | None:
    """The folder a set is read from: its own folder, else the data folder's sub-folder of that name (any letter case), else None."""
    own = set_folders().get(set_code.upper())
    if own:
        return Path(own)
    if data_dir:
        root = Path(data_dir)
        direct = root / set_code.upper()
        if direct.is_dir():
            return direct
        try:
            for p in root.iterdir():
                if p.is_dir() and p.name.upper() == set_code.upper():
                    return p
        except OSError:
            pass
    return None


# ---------------------------------------------------------------------------------------------
# scanning
# ---------------------------------------------------------------------------------------------

@dataclass
class ScanReport:
    data_dir: str = ""
    scanned: int = 0                                          # files looked at (detected)
    skipped: int = 0                                          # files unchanged since the last scan
    added: list[FileInfo] = field(default_factory=list)       # files newly registered (or replacing an older version)
    changed_sets: list[str] = field(default_factory=list)     # sets that got a new file
    missing: dict[str, list[str]] = field(default_factory=dict)   # {set: registered paths that are not on disk any more}
    deferred: list[str] = field(default_factory=list)         # sets not scanned because they were being calculated
    notes: list[str] = field(default_factory=list)

    @property
    def n_added(self) -> int:
        return len(self.added)


def _scanned(kind: str, event: str) -> bool:
    """Whether the scan registers a file of this kind (game / draft / replay) and event."""
    return event in SCAN_EVENTS or (kind.lower() == "game" and event in SCAN_GAME_EVENTS)


def _is_dataset_name(name: str) -> bool:
    m = FILE_NAME_RE.search(name)
    if m is not None:
        return _scanned(m.group(1), m.group(3))
    return _CARDS_RE.match(name) is not None


def _walk(folder: Path, all_events: bool = False) -> list[Path]:
    out: list[Path] = []
    try:
        for root, _dirs, files in os.walk(folder):
            for fn in sorted(files):
                if not fn.lower().endswith(DATA_EXTS):
                    continue
                m = FILE_NAME_RE.search(fn)
                if (m is not None and (all_events or _scanned(m.group(1), m.group(3)))) or (m is None and _CARDS_RE.match(fn) is not None):
                    out.append(Path(root) / fn)
    except OSError:
        pass
    return out


def _key(p: Path) -> str:
    return os.path.normcase(os.path.abspath(p))


def scan(data_dir: str | None, track: list[str] | tuple[str, ...] = (), skip_sets: list[str] | tuple[str, ...] = (),
         progress: Callable[[float, str], None] | None = None, cancel: threading.Event | None = None) -> ScanReport:
    """Register the new / changed dataset files of the tracked sets; never raises for unreadable files.

    Tracked sets = the sets with a registry + `track` (sets the user just asked to track).  `skip_sets` (sets being calculated) are
    left alone and reported in `deferred`: scan again when they are done.  Files are looked at once per (size, mtime)."""
    rep = ScanReport(data_dir=str(data_dir or ""))
    known = set(registry.known_sets()) | {s.upper() for s in track}
    skip = {s.upper() for s in skip_sets}
    root = Path(data_dir) if data_dir else None
    cands: dict[str, Path] = {}
    for s in sorted(known):
        if s in skip:
            rep.deferred.append(s)
            continue
        f = set_folder(s, data_dir)
        if f is not None and f.is_dir():
            for p in _walk(f):
                cands.setdefault(_key(p), p)
    if root is not None and root.is_dir():                    # files lying directly in the data folder
        try:
            for p in sorted(root.iterdir()):
                if p.is_file() and p.name.lower().endswith(DATA_EXTS) and _is_dataset_name(p.name):
                    cands.setdefault(_key(p), p)
        except OSError:
            pass
    with _LOCK:
        cache = dict((_read_json(_cache_path()).get("files")) or {})
    changed: set[str] = set()
    n = len(cands)
    for i, (k, p) in enumerate(cands.items()):
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        if progress is not None and (i % 5 == 0 or i == n - 1):
            progress(i / max(n, 1), f"Looking at {p.name}")
        try:
            st = p.stat()
        except OSError:
            continue
        sig = [st.st_size, st.st_mtime_ns]
        if cache.get(k) == sig:
            rep.skipped += 1
            continue
        m = FILE_NAME_RE.search(p.name)
        if m and m.group(2).upper() in skip:
            rep.deferred.append(m.group(2).upper())
            continue
        rep.scanned += 1
        info = detect_file(p)
        if info.kind == "unknown":
            cache[k] = sig
            continue
        code = (info.set_code or "").upper()
        if info.kind in registry.SHARED_KINDS and not code:
            pass                                               # a card list of several sets: shared
        elif code not in known or code in skip:
            continue                                           # a set the app does not track (yet): look again when it is
        cache[k] = sig
        res = registry.register(info)
        if res.registered:
            if res.note and res.note not in (info.note or ""):
                info.note = (info.note + "; " if info.note else "") + res.note
            rep.added.append(info)
            if code:
                changed.add(code)
        elif res.note and not res.note.startswith("already"):
            rep.notes.append(f"{p.name}: {res.note}")
    with _LOCK:
        _write_json(_cache_path(), {"files": cache})
    rep.changed_sets = sorted(changed)
    rep.deferred = sorted(set(rep.deferred))
    for s in sorted(known):
        gone = missing_files(s)
        if gone:
            rep.missing[s] = gone
    if progress is not None:
        progress(1.0, "Folder scan finished")
    return rep


def missing_files(set_code: str) -> list[str]:
    """Registered data files of the set that are no longer on disk (the registry keeps them: a model built from them stays valid)."""
    try:
        ents = registry.entries(set_code)
    except Exception:  # noqa: BLE001
        return []
    return [d["path"] for d in ents if d.get("path") and not os.path.exists(d["path"])]


# ---------------------------------------------------------------------------------------------
# sets that exist but are not tracked
# ---------------------------------------------------------------------------------------------

def set_codes_in(folder: Path) -> dict[str, int]:
    """{SET code: number of dataset files} of the files in a folder (from the file names; every event counts)."""
    out: dict[str, int] = {}
    for p in _walk(folder, all_events=True):
        m = FILE_NAME_RE.search(p.name)
        if m:
            out[m.group(2).upper()] = out.get(m.group(2).upper(), 0) + 1
    return out


def untracked_sets(data_dir: str | None) -> list[dict]:
    """Sets that are not tracked: [{set_code, folder (existing folder with dataset files or ""), n_local, n_remote, events}] sorted by
    code.  From the data folder's sub-folders and from 17lands' stored listing (no network)."""
    known = set(registry.known_sets())
    out: dict[str, dict] = {}
    root = Path(data_dir) if data_dir else None
    if root is not None and root.is_dir():
        try:
            subs = sorted(p for p in root.iterdir() if p.is_dir())
        except OSError:
            subs = []
        for sub in subs:
            for code, n in set_codes_in(sub).items():
                if code not in known:
                    e = out.setdefault(code, dict(set_code=code, folder=str(sub), n_local=0, n_remote=0, events=[]))
                    e["n_local"] += n
    for code, folder in set_folders().items():
        if code not in known and Path(folder).is_dir():
            n = set_codes_in(Path(folder)).get(code, 0)
            e = out.setdefault(code, dict(set_code=code, folder=folder, n_local=0, n_remote=0, events=[]))
            e["folder"], e["n_local"] = folder, max(e["n_local"], n)
    try:
        from sealed_lab.ingest import datasets

        remote = datasets.load_state().get("remote") or []
    except Exception:  # noqa: BLE001 - the listing is optional
        remote = []
    for d in remote:
        code = str(d.get("set_code", "")).upper()
        if not code or code in known:
            continue
        e = out.setdefault(code, dict(set_code=code, folder="", n_local=0, n_remote=0, events=[]))
        e["n_remote"] += 1
        ev = d.get("event")
        if ev and ev not in e["events"]:
            e["events"].append(ev)
    return [out[k] for k in sorted(out)]
