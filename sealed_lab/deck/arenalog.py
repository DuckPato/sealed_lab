"""Import the sealed pool of the current Arena event from Arena's own log.

With "Detailed Logs (Plugin Support)" enabled (Arena: Options > Account), `Player.log` records the JSON replies of the
game server.  A sealed event's course looks like

    {"CourseId": "...", "InternalEventName": "Sealed_MSH_20261001", "CardPool": [grpId, ...],
     "CourseDeck": {"MainDeck": [{"cardId": 95947, "quantity": 1}, ...], "Sideboard": [...]}, ...}

(either alone, as `{"Course": {...}}` or inside `{"Courses": [{...}, ...]}`).  `CardPool` has one grpId per card.  The
latest course with a non-empty pool whose event name contains "Sealed" is returned as Arena-export text.

Draft events (`PremierDraft_MSH_20261001`, `TradDraft_...`, `QuickDraft_...`) carry the same `CardPool` (the drafted cards; a real
Premier Draft course of 2026 has 42 pool entries and a 40-card `CourseDeck`).  `import_pool(include_draft=True)` also considers them:
the most recent course of either kind wins, and the result says which draft format it belongs to (`ArenaPool.draft_fmt`).  Without the
flag (the default, also the log watcher) nothing about the Sealed behaviour changes.

Arena grpIds are translated with Scryfall's `arena_id`: one `e:<set>` search per set (cached in
`APP_HOME/images/<SET>/arena_ids.json`), then `/cards/arena/{id}` for the ids still unknown (<= 2 requests/s).  All
network access is optional: without it the ids already cached are used and the others are reported.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterator

from sealed_lab import config
from sealed_lab.deck.arena import arena_name

log = logging.getLogger(__name__)

SEALED_RE = re.compile(r"sealed", re.IGNORECASE)
DRAFT_RE = re.compile(r"draft", re.IGNORECASE)
MIN_DRAFT_POOL = 40             # a smaller draft pool is probably an unfinished draft
LOG_NAMES = ("Player.log", "Player-prev.log")
MAX_JSON_LINES = 400            # a pretty-printed JSON document is joined from at most this many log lines
MAX_LOG_READ_BYTES = 64 << 20   # cap Player.log reads (64 MiB tail + 20 KiB header) so runaway Unity logs cannot OOM the app
LOG_HEAD_BYTES = 20_000
FALLBACK_INTERVAL = 0.55        # seconds between card-endpoint requests (search pages, /cards/arena/{id}): <= 2 per second
MAX_FALLBACK = 60               # at most this many single-card lookups per import


class ArenaLogError(Exception):
    """Base class: the message is meant for the user."""


class LogNotFound(ArenaLogError):
    pass


class DetailedLogsOff(ArenaLogError):
    pass


class NoSealedPool(ArenaLogError):
    pass


class UnknownCards(ArenaLogError):
    pass


@dataclass
class ArenaPool:
    event: str
    timestamp: str                       # ISO local time of the log entry, "" when unknown
    text: str                            # Arena export text (every pool card, "N Name" lines under "Deck")
    n_cards: int
    set_code: str | None = None
    source: str = ""                     # log file the pool came from
    unknown_ids: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    kind: str = "sealed"                 # "sealed" | "draft"
    draft_fmt: str = ""                  # for a draft: the model format of the queue, "dbo1" (Premier / Quick) or "dbo3" (Traditional)


def draft_format_of(event: str) -> str:
    """Model format of a draft event name: Traditional ("Trad" in the name) is "dbo3", every other draft queue (Premier, Quick, Pick Two) "dbo1";
    "" when the event is no draft."""
    if not DRAFT_RE.search(event or ""):
        return ""
    return "dbo3" if re.search(r"trad", event, re.IGNORECASE) else "dbo1"


@dataclass
class CourseEntry:
    event: str
    pool: list[int]
    main: dict[int, int]
    side: dict[int, int]
    timestamp: str
    source: str
    order: int                           # position in the log (later = more recent)
    last_played: str = ""                # ISO time from the course's own "LastPlayed" attribute ("" if absent)
    completed: bool = False              # the event is over (module Complete / ClaimPrize / ...)


# =============================================================================================
# finding and reading the log
# =============================================================================================

_MTGA_REL = Path("AppData") / "LocalLow" / "Wizards Of The Coast" / "MTGA"
_STEAM_ROOTS = (".steam/steam", ".steam/root", ".local/share/Steam", ".var/app/com.valvesoftware.Steam/.local/share/Steam",
                "snap/steam/common/.local/share/Steam")
_WINE_GLOBS = (".wine/drive_c/users/*", "Games/*/drive_c/users/*", ".local/share/bottles/bottles/*/drive_c/users/*",
               ".var/app/com.usebottles.bottles/data/bottles/bottles/*/drive_c/users/*")
ARENA_APP_ID = "2141910"                  # Steam's id of MTG Arena: only used to list its prefix first, never required


def _glob(base: Path, pattern: str) -> list[Path]:
    try:
        return sorted(base.glob(pattern))
    except (OSError, ValueError):
        return []


def linux_log_dirs(home: Path) -> list[Path]:
    """Candidate MTGA log folders on Linux: Arena under Steam (Proton prefixes, every Steam library) or under Wine / Lutris /
    Bottles.  Existing folders only, except that the default Steam one is listed first even when absent.  Never raises."""
    home = Path(home)
    libs: list[Path] = []
    for rel in _STEAM_ROOTS:
        root = home / rel
        libs.append(root)
        try:
            vdf = (root / "steamapps" / "libraryfolders.vdf").read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        libs.extend(Path(m.replace("\\\\", "\\")) for m in re.findall(r'"path"\s+"([^"]+)"', vdf))
    first = home / _STEAM_ROOTS[0] / "steamapps" / "compatdata" / ARENA_APP_ID / "pfx" / "drive_c" / "users" / "steamuser" / _MTGA_REL
    found: list[Path] = [first]
    for lib in libs:
        found += _glob(lib, f"steamapps/compatdata/{ARENA_APP_ID}/pfx/drive_c/users/*/{_MTGA_REL.as_posix()}")
        found += _glob(lib, f"steamapps/compatdata/*/pfx/drive_c/users/*/{_MTGA_REL.as_posix()}")
    for g in _WINE_GLOBS:
        found += _glob(home, f"{g}/{_MTGA_REL.as_posix()}")
    out: list[Path] = []
    seen: set[str] = set()
    for d in found:
        try:
            key = str(d.resolve())
        except (OSError, RuntimeError):
            key = str(d)
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


_LINUX_CACHE: dict = {}                   # the candidate folders, listed again at most once a minute
_LINUX_CACHE_S = 60.0


def _newest_log_mtime(d: Path) -> float | None:
    best = None
    for n in LOG_NAMES:
        try:
            t = (d / n).stat().st_mtime
        except OSError:
            continue
        best = t if best is None or t > best else best
    return best


def default_log_dir() -> Path:
    """The folder holding Arena's Player.log: env SEALED_LAB_ARENA_LOG_DIR; on Windows the usual AppData folder; on Linux the Steam /
    Proton or Wine prefix whose log was written last (the first candidate when none has a log yet)."""
    base = os.environ.get("SEALED_LAB_ARENA_LOG_DIR")
    if base:
        return Path(base)
    if sys.platform == "win32":
        home = os.environ.get("USERPROFILE") or str(Path.home())
        return Path(home) / _MTGA_REL
    now = time.monotonic()
    key = str(Path.home())
    if _LINUX_CACHE.get("key") != key or now - _LINUX_CACHE.get("at", -1e9) > _LINUX_CACHE_S:   # the watcher polls every few s
        _LINUX_CACHE.update(key=key, at=now, dirs=linux_log_dirs(Path.home()))
    cands = _LINUX_CACHE["dirs"]
    best, best_t = cands[0], None
    for d in cands:
        t = _newest_log_mtime(d)
        if t is not None and (best_t is None or t > best_t):
            best, best_t = d, t
    return best


def find_logs(log_dir: str | Path | None = None) -> list[Path]:
    """Existing log files, newest data first (Player.log, then Player-prev.log)."""
    d = Path(log_dir) if log_dir else default_log_dir()
    return [d / n for n in LOG_NAMES if (d / n).is_file()]


def detailed_logs_state(text_head: str) -> bool | None:
    """True / False when the log header says whether detailed logs are on; None when it does not say."""
    m = re.search(r"DETAILED LOGS:\s*(ENABLED|DISABLED)", text_head)
    return None if m is None else m.group(1) == "ENABLED"


_TS_RES = [
    re.compile(r"(\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2}:\d{2}(?: ?[AaPp][Mm])?)"),
    re.compile(r"(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})"),
]
_KEYS = ('"InternalEventName"', '"CardPool"', '"Courses"')


def _parse_ts(s: str, ref: datetime | None = None) -> str:
    """ISO local time of a log timestamp.  Arena writes the machine's locale (24/10/2026 9:03:22 or 10/24/2026 9:03:22
    PM); when day and month are both <= 12 the reading closest to `ref` (the log file's mtime) and not after it wins."""
    s = s.strip()
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2}):(\d{2})$", s)
    if m:
        return datetime(*map(int, m.groups())).isoformat(timespec="seconds")
    m = re.match(r"(\d{1,2})/(\d{1,2})/(\d{4}) (\d{1,2}):(\d{2}):(\d{2})(?: ?([AaPp])[Mm])?$", s)
    if not m:
        return ""
    a, b, y, hh, mi, ss, ap = m.groups()
    a, b, y, hh, mi, ss = int(a), int(b), int(y), int(hh), int(mi), int(ss)
    if ap:
        hh = hh % 12 + (12 if ap.lower() == "p" else 0)
    cands = []
    for day, mon in ((a, b), (b, a)):
        try:
            cands.append(datetime(y, mon, day, hh, mi, ss))
        except ValueError:
            continue
    if not cands:
        return ""
    if len(cands) > 1 and cands[0] != cands[1]:
        if ref is not None:
            ok = [c for c in cands if c <= ref.replace(tzinfo=None) + timedelta(days=1)] or cands
            cands = sorted(ok, key=lambda c: abs((ref.replace(tzinfo=None) - c).total_seconds()))
    return cands[0].isoformat(timespec="seconds")


def _line_ts(line: str) -> str:
    for rx in _TS_RES:
        m = rx.search(line)
        if m:
            return m.group(1)
    return ""


def iter_json_docs(lines: list[str]) -> Iterator[tuple[int, dict, str]]:
    """Yield (line index, JSON object, timestamp text of the closest preceding stamped line) for every JSON document that
    can hold a course: starting a log line or following a prefix such as `<== EventJoin(...)`, on one line or pretty-printed
    over several."""
    dec = json.JSONDecoder()
    last_ts = ""
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        j = line.find("{")
        ts = _line_ts(line if j < 0 else line[:j])
        if ts:
            last_ts = ts
        if j < 0:
            i += 1
            continue
        text = line[j:]
        used = 1
        obj = None
        if any(k in text for k in _KEYS):
            try:
                obj, _ = dec.raw_decode(text)
            except ValueError:
                obj = None
        if obj is None and text.rstrip().endswith(("{", "[", ",")) and                 any(k in lines[m] for m in range(i, min(n, i + 8)) for k in _KEYS):
            buf = [text]
            depth = text.count("{") - text.count("}")
            k = 1
            while depth > 0 and k < MAX_JSON_LINES and i + k < n:
                buf.append(lines[i + k])
                depth += lines[i + k].count("{") - lines[i + k].count("}")
                k += 1
            try:
                obj, _ = dec.raw_decode("\n".join(buf))
                used = k
            except ValueError:
                obj = None
        if isinstance(obj, dict):
            yield i, obj, last_ts
        i += used


def _counts(entries) -> dict[int, int]:
    out: dict[int, int] = {}
    for e in entries or []:
        try:
            out[int(e["cardId"])] = out.get(int(e["cardId"]), 0) + int(e.get("quantity", 1))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _pool_ids(course: dict) -> list[int]:
    ids: list[int] = []
    for x in course.get("CardPool") or []:
        try:
            ids.append(int(x))
        except (TypeError, ValueError):
            continue
    if not ids:                                           # newer logs: CardPoolByCollation [{collationId, cardPool}]
        for grp in course.get("CardPoolByCollation") or []:
            if isinstance(grp, dict):
                for x in grp.get("cardPool") or grp.get("CardPool") or []:
                    try:
                        ids.append(int(x))
                    except (TypeError, ValueError):
                        continue
    return ids


def _courses_in(doc: dict) -> Iterator[dict]:
    if isinstance(doc.get("Courses"), list):
        for c in doc["Courses"]:
            if isinstance(c, dict):
                yield c
    if isinstance(doc.get("Course"), dict):
        yield doc["Course"]
    if "InternalEventName" in doc:
        yield doc


def _course_time(c: dict, fallback: str, ref: datetime | None) -> str:
    t = _parse_ts(fallback, ref) if fallback else ""
    if t:
        return t
    for a in (c.get("CourseDeckSummary") or {}).get("Attributes") or []:
        if isinstance(a, dict) and a.get("name") == "LastPlayed":
            t = _parse_ts(_line_ts(str(a.get("value", ""))), ref)
            if t:
                return t
    return ""


_DONE_MODULES = ("complete", "completed", "claimprize", "claimreward", "eventended", "finished")


def _last_played(c: dict, ref: datetime | None) -> str:
    for a in (c.get("CourseDeckSummary") or {}).get("Attributes") or []:
        if isinstance(a, dict) and a.get("name") == "LastPlayed":
            v = str(a.get("value", "")).strip()
            try:                              # ISO 8601 ("2026-10-24T20:01:02Z"): compared in local time like the log's own stamps
                d = datetime.fromisoformat(v.replace("Z", "+00:00"))
                if d.tzinfo is not None:
                    d = d.astimezone().replace(tzinfo=None)
                return d.isoformat(timespec="seconds")
            except ValueError:
                return _parse_ts(_line_ts(v) or v, ref)
    return ""


def _is_completed(c: dict) -> bool:
    mod = str(c.get("CurrentModule") or "").replace("_", "").replace(" ", "").casefold()
    return bool(mod) and any(mod.startswith(m) for m in _DONE_MODULES)


def pick_sealed(found: list[CourseEntry]) -> CourseEntry | None:
    """The sealed course the player is most likely working on: not completed first, then the most recent LastPlayed (else the log
    time), then the later position in the log.  The last course in log order alone can be an old event that the client re-sent."""
    sealed = [c for c in found if SEALED_RE.search(c.event)]
    if not sealed:
        return None
    live = [c for c in sealed if not c.completed]
    pool = live or sealed
    return max(pool, key=lambda c: (c.last_played or c.timestamp or "", c.order))


def pick_course(found: list[CourseEntry], include_draft: bool = False) -> CourseEntry | None:
    """`pick_sealed`, or with `include_draft` the same choice among the Sealed AND the draft courses (a draft course whose pool is the
    most recent one wins).  A Sealed course is picked exactly as `pick_sealed` does when no draft course is involved."""
    if not include_draft:
        return pick_sealed(found)
    cands = [c for c in found if SEALED_RE.search(c.event) or DRAFT_RE.search(c.event)]
    if not cands:
        return None
    live = [c for c in cands if not c.completed]
    return max(live or cands, key=lambda c: (c.last_played or c.timestamp or "", c.order))


def scan_courses(text: str, source: str = "", ref: datetime | None = None) -> list[CourseEntry]:
    """Every course with a non-empty card pool in the log text, in log order (`ref`: file mtime, for date parsing)."""
    lines = text.splitlines()
    out: list[CourseEntry] = []
    for idx, doc, ts in iter_json_docs(lines):
        for c in _courses_in(doc):
            pool = _pool_ids(c)
            if not pool:
                continue
            deck = c.get("CourseDeck") or {}
            out.append(CourseEntry(event=str(c.get("InternalEventName", "")), pool=pool, main=_counts(deck.get("MainDeck")),
                                   side=_counts(deck.get("Sideboard")), timestamp=_course_time(c, ts, ref), source=source,
                                   order=idx, last_played=_last_played(c, ref), completed=_is_completed(c)))
    return out


def latest_sealed_pool(texts: list[tuple[str, str]], mtimes: list[datetime | None] | None = None, include_draft: bool = False,
                       course_key: tuple | None = None) -> tuple[CourseEntry | None, int, bool | None]:
    """From [(source name, log text)], newest file first: (latest sealed course with a pool, number of courses with a pool
    seen, detailed-logs flag of the newest file).  `include_draft`: draft courses compete too (see `pick_course`).  `course_key`
    ((event, sorted pool), as the Arena watcher reports it) picks exactly that course when it is in the logs, else the usual choice."""
    seen = 0
    detailed: bool | None = None
    for k, (name, text) in enumerate(texts):
        if k == 0:
            detailed = detailed_logs_state(text[:20000])
        found = scan_courses(text, name, (mtimes or [None] * len(texts))[k])
        seen += len(found)
        best = None
        if course_key is not None:
            best = next((c for c in reversed(found) if (SEALED_RE.search(c.event) or (include_draft and DRAFT_RE.search(c.event)))
                         and (c.event, tuple(sorted(c.pool))) == tuple(course_key)), None)
        best = best or pick_course(found, include_draft)
        if best is not None:
            return best, seen, detailed
    return None, seen, detailed


# =============================================================================================
# grpId -> card name
# =============================================================================================

def ids_file(set_code: str) -> Path:
    code = config.valid_set_code(set_code)
    if code is None:
        raise ValueError(f"invalid set code: {set_code!r}")
    return config.app_home() / "images" / code / "arena_ids.json"


def _http_get_scryfall(http, url: str, *, params: dict | None = None, headers: dict | None = None, timeout=(6, 30)):
    """GET a Scryfall URL with host allowlist and post-redirect host validation."""
    from sealed_lab.ingest.scryfall import allowed_url

    if not allowed_url(url):
        raise ValueError(f"refusing to fetch {url!r}: only https://api.scryfall.com and scryfall.io are allowed")
    try:
        r = http.get(url, params=params, headers=headers, timeout=timeout, stream=True)
    except TypeError:
        r = http.get(url, params=params, headers=headers, timeout=timeout)
    final = getattr(r, "url", None)
    if isinstance(final, str) and not allowed_url(final):
        close = getattr(r, "close", None)
        if close:
            close()
        raise ValueError(f"refusing a response redirected to {final!r}")
    return r


def _read_log_bounded(path: Path, max_bytes: int = MAX_LOG_READ_BYTES) -> str:
    """Read a log file safely: full text when <= `max_bytes`, else the first `LOG_HEAD_BYTES` (for the Detailed Logs header)
    plus the newline-aligned last `max_bytes` tail so multi-GB Unity log loops cannot exhaust memory."""
    size = path.stat().st_size
    if size <= max_bytes:
        return path.read_text(encoding="utf-8", errors="replace")
    with open(path, "rb") as fh:
        head = fh.read(LOG_HEAD_BYTES)
        fh.seek(max(size - max_bytes, LOG_HEAD_BYTES))
        tail = fh.read()
    nl = tail.find(b"\n")
    if nl >= 0:
        tail = tail[nl + 1:]
    return head.decode("utf-8", errors="replace") + "\n" + tail.decode("utf-8", errors="replace")


def load_ids(set_code: str) -> dict[int, str]:
    try:
        d = json.loads(ids_file(set_code).read_text(encoding="utf-8"))
        return {int(k): str(v) for k, v in (d.get("ids") or {}).items()}
    except (OSError, ValueError):
        return {}


def save_ids(set_code: str, ids: dict[int, str], searched: bool) -> None:
    p = ids_file(set_code)
    try:
        old = {}
        try:
            old = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(dict(ids={str(k): v for k, v in sorted(ids.items())},
                                       searched=bool(searched or old.get("searched")),
                                       updated=datetime.now(timezone.utc).isoformat(timespec="seconds")),
                                  ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, p)
    except OSError:
        log.info("could not save the Arena id cache")


def _searched(set_code: str) -> bool:
    try:
        return bool(json.loads(ids_file(set_code).read_text(encoding="utf-8")).get("searched"))
    except (OSError, ValueError):
        return False


def _card_names(card: dict) -> str:
    return str(card.get("name") or "")


def fetch_set_ids(set_code: str, http=None, sleep: Callable[[float], None] = time.sleep) -> dict[int, str]:
    """{arena_id: name} of every printing of the set that has an arena_id (one paginated `e:<set>` search)."""
    from sealed_lab.ingest.scryfall import API, HEADERS, allowed_url, json_capped

    code = config.valid_set_code(set_code)
    if code is None:
        raise ValueError(f"invalid set code: {set_code!r}")
    if http is None:
        import requests

        http = requests.Session()
    out: dict[int, str] = {}
    url: str | None = f"{API}/cards/search"
    params: dict | None = {"q": f"e:{code.lower()}", "unique": "prints", "order": "set"}
    pages = 0
    while url and pages < 20:
        r = _http_get_scryfall(http, url, params=params, headers=HEADERS, timeout=(6, 30))
        try:
            if r.status_code != 200:
                log.info("Scryfall set search for %s: HTTP %s", code, r.status_code)
                break
            j = json_capped(r)
        finally:
            close = getattr(r, "close", None)
            if close:
                close()
        for c in j.get("data", []):
            aid = c.get("arena_id")
            if aid is not None:
                out[int(aid)] = _card_names(c)
        nxt = j.get("next_page") if j.get("has_more") else None
        if nxt and not allowed_url(nxt):
            log.warning("Scryfall next_page is not an allowed URL: %r", nxt)
            break
        url = nxt
        params = None
        pages += 1
        if url:
            sleep(FALLBACK_INTERVAL)
    return out


def fetch_arena_card(aid: int, http=None) -> tuple[str | None, bool]:
    """(name, rate_limited) of one grpId through `/cards/arena/{id}`."""
    from sealed_lab.ingest.scryfall import API, HEADERS, json_capped

    if http is None:
        import requests

        http = requests.Session()
    r = _http_get_scryfall(http, f"{API}/cards/arena/{int(aid)}", headers=HEADERS, timeout=(6, 20))
    try:
        if r.status_code == 429:
            return None, True
        if r.status_code != 200:
            return None, False
        return _card_names(json_capped(r)) or None, False
    finally:
        close = getattr(r, "close", None)
        if close:
            close()


def resolve_ids(grp_ids: list[int], set_code: str | None, offline: bool = False, http=None,
                sleep: Callable[[float], None] = time.sleep) -> tuple[dict[int, str], list[int], list[str]]:
    """Names for `grp_ids`: (id -> name, ids still unknown, notes).  Uses the per-set cache, then a set search, then single
    lookups; every network error just leaves ids unresolved."""
    notes: list[str] = []
    want = set(grp_ids)
    sc = (set_code or "").upper()
    known = load_ids(sc) if sc else {}
    names = {i: known[i] for i in want if i in known}
    if sc and not offline and (want - set(names)) and not _searched(sc):
        try:
            got = fetch_set_ids(sc, http=http, sleep=sleep)
            known.update(got)
            save_ids(sc, known, searched=bool(got))
            names = {i: known[i] for i in want if i in known}
        except Exception as e:  # noqa: BLE001 - offline-safe
            notes.append(f"Scryfall set search failed ({type(e).__name__}); using cached ids only")
            log.info("arena id search failed: %s", e)
    missing = sorted(want - set(names))
    if missing and not offline and len(missing) <= MAX_FALLBACK:
        added = False
        try:
            for k, aid in enumerate(missing):
                if k:
                    sleep(FALLBACK_INTERVAL)
                name, limited = fetch_arena_card(aid, http=http)
                if limited:
                    notes.append("Scryfall rate limit reached; some cards could not be looked up")
                    break
                if name:
                    names[aid] = name
                    known[aid] = name
                    added = True
        except Exception as e:  # noqa: BLE001
            notes.append(f"Scryfall lookup failed ({type(e).__name__})")
            log.info("arena id lookup failed: %s", e)
        if added and sc:
            save_ids(sc, known, searched=False)
    elif missing and not offline:
        notes.append(f"{len(missing)} unknown card ids: too many for single lookups")
    return names, sorted(want - set(names)), notes


_NOT_SETS = {"play", "trad", "bo3", "bo1", "comp", "ladder", "cube", "mwm", "draft", "quick", "event", "open", "prem"}
_NOT_SET_RE = re.compile(r"day\d+|y\d{2}|\d+", re.IGNORECASE)      # DAY1, Y25 (year events), plain numbers (dates)


def known_set_codes() -> set[str]:
    """Set codes this app knows: its own sets plus Scryfall's set list when it is cached (`scryfall_sets.json`); offline-safe."""
    out: set[str] = set()
    try:
        from sealed_lab.ingest import registry

        out.update(c.upper() for c in registry.known_sets())
    except Exception:  # noqa: BLE001
        pass
    try:
        d = json.loads((config.app_home() / "scryfall_sets.json").read_text(encoding="utf-8"))
        out.update(str(c).upper() for c in (d.get("sets") or {}))
    except (OSError, ValueError, AttributeError):
        pass
    return out


def event_set(event: str, known: set[str] | None = None) -> str | None:
    """Set code inside an event name such as `Sealed_MSH_20261001` / `TradSealed_MSH_...`; None when it has none.  With a list of known
    set codes (default: `known_set_codes()`; when it is empty only the shape is checked) the token must be one of them, so
    `Sealed_DAY1_...`, `Cube_...` or `..._Y25_...` give None."""
    known = known_set_codes() if known is None else {k.upper() for k in known}
    for part in re.split(r"[_\-\s]+", event):
        if (re.fullmatch(r"[A-Za-z0-9]{3,4}", part) and not SEALED_RE.search(part) and part.lower() not in _NOT_SETS
                and not _NOT_SET_RE.fullmatch(part)):
            if known and part.upper() not in known:
                continue
            return part.upper()
    return None


# =============================================================================================
# entry point
# =============================================================================================

def pool_to_text(names: dict[str, int]) -> str:
    lines = ["Deck"] + [f"{n} {arena_name(k)}" for k, n in sorted(names.items())]
    return "\n".join(lines) + "\n"


def import_pool(set_code: str | None = None, log_dir: str | Path | None = None, offline: bool = False, http=None,
                sleep: Callable[[float], None] = time.sleep, include_draft: bool = False, course_key: tuple | None = None) -> ArenaPool:
    """The pool of the most recent sealed event in Arena's logs, as Arena-export text.  `include_draft`: a draft event (Premier / Trad /
    Quick Draft) whose pool is more recent than every Sealed one is imported instead; `ArenaPool.kind` / `draft_fmt` say so.
    `course_key` (the watcher's offer) imports exactly that event.

    Raises an `ArenaLogError` subclass with a user-facing message when there is nothing to import."""
    paths = find_logs(log_dir)
    if not paths:
        d = Path(log_dir) if log_dir else default_log_dir()
        raise LogNotFound(f"Arena log not found in {d}. Start Arena once with 'Detailed Logs (Plugin Support)' enabled "
                          "(Options > Account), or paste the pool by hand.")
    texts: list[tuple[str, str]] = []
    mtimes: list[datetime | None] = []
    for p in paths:
        try:
            texts.append((p.name, _read_log_bounded(p)))
            mtimes.append(datetime.fromtimestamp(p.stat().st_mtime))
        except OSError as e:
            log.info("cannot read %s: %s", p, e)
    if not texts:
        raise LogNotFound(f"Arena log files could not be read ({', '.join(p.name for p in paths)}).")
    course, seen, detailed = latest_sealed_pool(texts, mtimes, include_draft, course_key)
    if course is None:
        if detailed is False or (detailed is None and not any('"InternalEventName"' in t for _, t in texts)):
            raise DetailedLogsOff("Arena is not writing detailed logs. Enable Options > Account > 'Detailed Logs (Plugin "
                                  "Support)' in Arena, restart it, open your sealed event, then try again.")
        what = "sealed or draft" if include_draft else "sealed"
        raise NoSealedPool(f"No {what} pool found in the Arena log"
                           + (f" ({seen} course(s) with a pool, none from a {'Sealed or draft' if include_draft else 'Sealed'} event)" if seen else
                              " (no sealed event with a card pool: join the event in Arena first, then reopen it)")
                           + ("." if detailed else ". Check that 'Detailed Logs (Plugin Support)' is enabled in Arena "
                              "(Options > Account)."))
    sc = (event_set(course.event) or set_code or "").upper() or None      # the event's own set wins over the one selected in the GUI
    names, unknown, notes = resolve_ids(course.pool, sc, offline=offline, http=http, sleep=sleep)
    if not names:
        raise UnknownCards(f"Could not translate the {len(course.pool)} card ids of {course.event} to names"
                           + ("" if offline else " (Scryfall is unreachable or does not know this set yet)")
                           + f"; first ids: {', '.join(str(i) for i in sorted(set(course.pool))[:5])}.")
    counts: dict[str, int] = {}
    for gid in course.pool:
        nm = names.get(gid)
        if nm:
            counts[nm] = counts.get(nm, 0) + 1
    warns = list(notes)
    if unknown:
        warns.append(f"{len(unknown)} card id(s) could not be translated and are missing from the pool: "
                     + ", ".join(str(i) for i in unknown[:8]) + ("..." if len(unknown) > 8 else ""))
    dfmt = "" if SEALED_RE.search(course.event) else draft_format_of(course.event)
    if dfmt and sum(counts.values()) < MIN_DRAFT_POOL:
        warns.append(f"the draft pool has only {sum(counts.values())} cards: the draft may not be finished")
    return ArenaPool(event=course.event, timestamp=course.timestamp, text=pool_to_text(counts),
                     n_cards=sum(counts.values()), set_code=sc, source=course.source, unknown_ids=unknown, warnings=warns,
                     kind="draft" if dfmt else "sealed", draft_fmt=dfmt)
