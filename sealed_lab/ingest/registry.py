"""Per-set file registry (`sets/<SET>/registry.json`) plus a global registry for set-less shared files.

Replacement rule: for the same (kind, event) within a set, a newer file (by mtime) replaces the older one.
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from sealed_lab.config import app_home, ensure_set_dirs, set_dir, sets_dir
from sealed_lab.types import FileInfo

SHARED_KINDS = ("cards", "scryfall")
_LOCK = threading.RLock()      # registry files are read-modify-written from several GUI worker threads (scan, download, drop)


@dataclass
class RegisterResult:
    registered: bool
    info: FileInfo
    replaced: list[FileInfo] = field(default_factory=list)
    note: str = ""


def _reg_path(set_code: str) -> Path:
    return set_dir(set_code) / "registry.json"


def _global_path() -> Path:
    return app_home() / "global_registry.json"


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("files", [])
    except (OSError, ValueError):
        return []


def _write(path: Path, files: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"version": 1, "files": files}, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _to_info(d: dict) -> FileInfo:
    return FileInfo(d["path"], d["kind"], d.get("set_code"), d.get("event"), d.get("fmt"), d.get("size", 0),
                    d.get("fingerprint", ""), d.get("note", ""))


def _to_dict(info: FileInfo, mtime: float) -> dict:
    d = asdict(info)
    d["mtime"] = mtime
    d["added_at"] = time.time()
    return d


def uncompressed_size(path: str) -> int | None:
    """Size of the data a file holds: the file size, or for a .gz the gzip trailer's ISIZE (size mod 2**32). None if unreadable."""
    try:
        if str(path).lower().endswith(".gz"):
            with open(path, "rb") as f:
                f.seek(-4, os.SEEK_END)
                return int.from_bytes(f.read(4), "little")
        return os.stat(path).st_size
    except OSError:
        return None


def same_content(csv_path: str, gz_path: str) -> bool | None:
    """Whether an uncompressed file and a .gz hold the same data, judged by size (the gzip trailer records the uncompressed size
    modulo 2**32): True / False, None if either cannot be read."""
    a, b = uncompressed_size(csv_path), uncompressed_size(gz_path)
    if a is None or b is None:
        return None
    return (a % (1 << 32)) == b


def _mtime(path: str) -> float:
    try:
        return os.stat(path).st_mtime
    except OSError:
        return 0.0


def known_sets() -> list[str]:
    root = sets_dir()
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir() and (p / "registry.json").exists())


def entries(set_code: str) -> list[dict]:
    p = _reg_path(set_code)
    ents = _read(p)
    if any(_mislabeled_replay(d) for d in ents):
        with _LOCK:
            ents = _migrate_replays(p, _read(p))
    return ents


REPLAY_NOTE = "replay data (turn by turn): registered, not used by the model"


def _mislabeled_replay(d: dict) -> bool:
    """Entries written before replay detection was fixed: replay files registered as game data."""
    return d.get("kind") == "game" and Path(d.get("path", "")).name.lower().startswith("replay_data_public.")


def _migrate_replays(path: Path, ents: list[dict]) -> list[dict]:
    """Relabel mislabeled replay entries and register the matching game_data file from the same folder, if any."""
    from sealed_lab.ingest.detect import detect_file

    siblings = []
    for d in ents:
        if _mislabeled_replay(d):
            d["kind"] = "replay"
            d["note"] = REPLAY_NOTE
            src = Path(d["path"])
            rest = src.name[len("replay_data_public."):]
            base = rest[:-3] if rest.lower().endswith(".gz") else rest
            siblings += [src.with_name("game_data_public." + r) for r in (base, base + ".gz")]
    _write(path, ents)
    have_game = {d.get("event") for d in ents if d.get("kind") == "game"}
    for sib in siblings:
        if not sib.exists():
            continue
        try:
            info = detect_file(sib)
        except OSError:
            continue
        if info.kind == "game" and info.event not in have_game:
            _register_in(path, info, _mtime(str(sib)))
            have_game.add(info.event)
    return _read(path)


def files(set_code: str, include_shared: bool = True) -> list[FileInfo]:
    """Files registered for a set (plus shared cards/scryfall files when `include_shared`)."""
    out = [_to_info(d) for d in entries(set_code)]
    if include_shared:
        have = {f.fingerprint for f in out}
        for d in _read(_global_path()):
            if d["fingerprint"] not in have:
                out.append(_to_info(d))
    return out


def shared_files() -> list[FileInfo]:
    return [_to_info(d) for d in _read(_global_path())]


def _register_in(path: Path, info: FileInfo, mtime: float) -> RegisterResult:
    with _LOCK:
        return _register_locked(path, info, mtime)


def _register_locked(path: Path, info: FileInfo, mtime: float) -> RegisterResult:
    ents = _read(path)
    for d in ents:
        if d["fingerprint"] == info.fingerprint and d["kind"] == info.kind:
            return RegisterResult(False, info, [], "already registered")
    same = [d for d in ents if d["kind"] == info.kind and d.get("event") == info.event]
    replaced: list[FileInfo] = []
    for d in same:
        new_gz, old_gz = info.path.lower().endswith(".gz"), d["path"].lower().endswith(".gz")
        if d["path"] != info.path and new_gz != old_gz and os.path.exists(d["path"]):
            # the same dataset as .csv and as .csv.gz: the server publishes the .gz, so when the two differ the .gz is the one to
            # use whatever the file times say (a decompressed older download can be the newer file)
            csv_p, gz_p = (d["path"], info.path) if new_gz else (info.path, d["path"])
            if same_content(csv_p, gz_p) is True:
                if mtime >= d.get("mtime", 0.0):
                    replaced.append(_to_info(d))
                    continue
                return RegisterResult(False, info, [], f"same data as the registered {Path(d['path']).name}; ignored")
            if new_gz:
                replaced.append(_to_info(d))
                continue
            return RegisterResult(False, info, [], f"differs from the registered {Path(d['path']).name}, which is preferred "
                                                   "(compressed copies match the 17lands download); ignored")
        if d["path"] == info.path or mtime >= d.get("mtime", 0.0):
            replaced.append(_to_info(d))
        else:
            return RegisterResult(False, info, [], f"older than registered file {Path(d['path']).name}; ignored")
    keep = [d for d in ents if d not in same]
    keep.append(_to_dict(info, mtime))
    _write(path, keep)
    return RegisterResult(True, info, replaced, "replaces older file" if replaced else "")


def register(info: FileInfo) -> RegisterResult:
    """Register a detected file. Files without a set code are only accepted for shared kinds."""
    mtime = _mtime(info.path)
    if info.kind in ("unknown",):
        return RegisterResult(False, info, [], "unknown file type")
    if info.set_code is None:
        if info.kind in SHARED_KINDS:
            return _register_in(_global_path(), info, mtime)
        return RegisterResult(False, info, [], "set code unknown")
    set_code = info.set_code.upper()
    ensure_set_dirs(set_code)
    return _register_in(_reg_path(set_code), info, mtime)


def remove(set_code: str, fingerprint: str) -> bool:
    """Remove a file (by fingerprint) from the set registry, or from the shared registry if only there."""
    with _LOCK:
        p = _reg_path(set_code)
        ents = _read(p)
        keep = [d for d in ents if d["fingerprint"] != fingerprint]
        if len(keep) != len(ents):
            _write(p, keep)
            return True
        g = _read(_global_path())
        keepg = [d for d in g if d["fingerprint"] != fingerprint]
        if len(keepg) != len(g):
            _write(_global_path(), keepg)
            return True
        return False
