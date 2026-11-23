"""File detection: classify 17lands / Scryfall files by name, then by header sniffing."""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import logging
import os
import tarfile
from pathlib import Path

from sealed_lab.config import FILE_NAME_RE, event_format
from sealed_lab.types import FileInfo

log = logging.getLogger(__name__)

_MIB = 1 << 20


def is_gzip(path: str | Path) -> bool:
    """True if the file starts with the gzip magic bytes (regardless of its extension)."""
    with open(path, "rb") as f:
        return f.read(2) == b"\x1f\x8b"


# old header names -> current ones (older 17lands files, e.g. AFR, call the skill bucket "user_win_rate_bucket")
HEADER_ALIASES = {"user_win_rate_bucket": "user_game_win_rate_bucket"}


def normalize_header(header: list[str]) -> list[str]:
    """Rename old column names to the current ones (only when the current name is absent)."""
    hset = set(header)
    return [HEADER_ALIASES[h] if h in HEADER_ALIASES and HEADER_ALIASES[h] not in hset else h for h in header]


def csv_stream(raw):
    """The CSV bytes of an open binary file: the file itself, its gzip stream, or the first regular file of a tar archive
    (plain or gzipped; some older 17lands downloads, e.g. AFR, are a .tar inside the .csv.gz). Decided by magic bytes.
    `raw` stays the underlying file, so raw.tell() still measures progress through the file on disk."""
    head = raw.read(2)
    raw.seek(0)
    stream = gzip.GzipFile(fileobj=raw, mode="rb") if head == bytes((0x1F, 0x8B)) else raw
    block = stream.read(512)
    stream.seek(0)
    if block[257:262] != b"ustar":
        return stream
    tf = tarfile.open(fileobj=stream, mode="r:")       # random access (seekable member), but only the first header is read
    member = tf.next()
    while member is not None and not member.isfile():
        member = tf.next()
    if member is None:
        raise ValueError("tar archive without a data file")
    return tf.extractfile(member)


class _Owned:
    """A stream that also closes the file it reads from."""

    def __init__(self, stream, raw):
        self._stream, self._raw = stream, raw

    def __getattr__(self, name):
        return getattr(self._stream, name)

    def close(self):
        try:
            self._stream.close()
        finally:
            self._raw.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def open_binary(path: str | Path):
    """Open a possibly-gzipped (or tar-wrapped) file for binary reading of its data (decided by magic bytes, not extension)."""
    raw = open(str(path), "rb")
    try:
        stream = csv_stream(raw)
    except BaseException:
        raw.close()
        raise
    return raw if stream is raw else _Owned(stream, raw)


def fingerprint(path: str | Path) -> str:
    """size + mtime + sha1 of first and last 1 MiB, as a 20-hex-char id."""
    p = Path(path)
    st = p.stat()
    h = hashlib.sha1()
    h.update(f"{st.st_size}|{st.st_mtime_ns}".encode())
    with open(p, "rb") as f:
        h.update(f.read(_MIB))
        if st.st_size > _MIB:
            f.seek(max(_MIB, st.st_size - _MIB))
            h.update(f.read(_MIB))
    return h.hexdigest()[:20]


def read_head_rows(path: str | Path, n_rows: int = 2) -> list[list[str]]:
    """Parse the first `n_rows` CSV rows (header first) with a real CSV parser."""
    with open_binary(path) as raw:
        text = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
        reader = csv.reader(text)
        out = []
        for _ in range(n_rows):
            try:
                out.append(next(reader))
            except StopIteration:
                break
        return out


def _sniff_json(path: Path) -> dict | None:
    """Return {'set': code|None} if the file looks like a Scryfall card list / bulk file, else None."""
    try:
        with open(path, "rb") as f:
            start = f.read(4096).lstrip()
        if not start or start[:1] not in (b"[", b"{"):
            return None
        from sealed_lab.ingest.scryfall import iter_json_cards  # lazy

        sets: set[str] = set()
        n = 0
        for obj in iter_json_cards(path):
            if not isinstance(obj, dict) or "name" not in obj or ("type_line" not in obj and "card_faces" not in obj):
                return None
            sets.add(str(obj.get("set", "")).upper())
            n += 1
            if n >= 200:
                break
        if n == 0:
            return None
        sets.discard("")
        return {"set": next(iter(sets)) if len(sets) == 1 else None}
    except Exception:  # noqa: BLE001 - detection must never raise
        return None


def detect_file(path: str | Path) -> FileInfo:
    """Classify a file. Never raises for unreadable/unknown files (returns kind='unknown' with a note)."""
    p = Path(path)
    try:
        size = p.stat().st_size
        fp = fingerprint(p)
    except OSError as e:
        return FileInfo(str(p), "unknown", None, None, None, 0, "", note=f"unreadable: {e}")
    name = p.name
    m = FILE_NAME_RE.search(name)
    name_kind = name_set = name_event = None
    if m:
        name_kind, name_set, name_event = m.group(1).lower(), m.group(2).upper(), m.group(3)
    if size == 0:
        return FileInfo(str(p), "unknown", name_set, name_event, None, 0, fp, note="empty file")

    # JSON (Scryfall) files
    lower = name.lower()
    if lower.endswith(".json") or lower.endswith(".json.gz"):
        js = _sniff_json(p)
        if js is not None:
            return FileInfo(str(p), "scryfall", js["set"], None, None, size, fp,
                            note="Scryfall card data" + ("" if js["set"] else " (multi-set / bulk)"))
        return FileInfo(str(p), "unknown", None, None, None, size, fp, note="JSON file that is not Scryfall card data")

    try:
        rows = read_head_rows(p, 2)
    except Exception as e:  # noqa: BLE001
        return FileInfo(str(p), "unknown", name_set, name_event, None, size, fp, note=f"cannot read header: {e}")
    if not rows:
        return FileInfo(str(p), "unknown", name_set, name_event, None, size, fp, note="no header")
    header = rows[0]
    first = dict(zip(header, rows[1])) if len(rows) > 1 else {}
    hs = set(header)
    has = lambda pre: any(c.startswith(pre) for c in header)  # noqa: E731

    # replay files also carry `won` and `deck_*` columns, so their turn-by-turn markers are checked first;
    # game files have per-card opening_hand_* / drawn_* columns that replay files lack
    if has("user_turn_") or has("oppo_turn_") or "history_id" in hs:
        kind = "replay"
    elif "won" in hs and has("deck_"):
        kind = "game"
    elif "pick" in hs and has("pack_card_"):
        kind = "draft"
    elif "name" in hs and "rarity" in hs:
        kind = "cards"
    else:
        kind = "unknown"

    notes = []
    if name_kind and kind != "unknown" and name_kind != kind:
        notes.append(f"file name says '{name_kind}' but header looks like '{kind}'; trusting header")
    if kind == "unknown" and name_kind:
        notes.append(f"file name says '{name_kind}' but header not recognised")
        kind = "unknown"

    set_code = event = None
    if kind in ("game", "draft", "replay"):
        set_code = name_set if (m and name_kind == kind) else None
        event = name_event if (m and name_kind == kind) else None
        if set_code is None:
            exp = (first.get("expansion") or "").strip()
            if exp:
                set_code = exp.upper()
                notes.append("set taken from first data row")
        if event is None:
            ev = (first.get("event_type") or "").strip()
            if ev:
                event = ev
                notes.append("event taken from first data row")
        if set_code is None:
            notes.append("could not determine set code")
        if event is None:
            notes.append("could not determine event type")
    elif kind == "cards":
        set_code = _cards_single_set(p)
        if set_code is None:
            notes.append("card list covers several sets (or none): shared by all sets")
    fmt = event_format(event) if kind in ("game", "draft", "replay") and event else None
    return FileInfo(str(p), kind, set_code, event, fmt, size, fp, note="; ".join(notes))


def _cards_single_set(path: Path) -> str | None:
    """A cards.csv with exactly one distinct `expansion` value belongs to that set."""
    try:
        with open_binary(path) as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
            reader = csv.DictReader(text)
            if "expansion" not in (reader.fieldnames or []):
                return None
            seen = set()
            for i, row in enumerate(reader):
                v = (row.get("expansion") or "").strip().upper()
                if v:
                    seen.add(v)
                if len(seen) > 1 or i > 200000:
                    break
            return next(iter(seen)) if len(seen) == 1 else None
    except Exception:  # noqa: BLE001
        return None


def expand_paths(paths: list[str | Path]) -> list[Path]:
    """Expand folders recursively (only plausible data files); explicit files are always kept."""
    out: list[Path] = []
    exts = (".csv", ".gz", ".json")
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            for root, _dirs, files in os.walk(p):
                for fn in sorted(files):
                    if fn.lower().endswith(exts):
                        out.append(Path(root) / fn)
        elif p.exists():
            out.append(p)
        else:
            log.warning("path does not exist: %s", p)
    seen, uniq = set(), []
    for p in out:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq


__all__ = ["detect_file", "fingerprint", "expand_paths", "open_binary", "csv_stream", "normalize_header", "is_gzip",
           "read_head_rows"]
