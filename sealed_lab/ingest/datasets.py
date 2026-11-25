"""Remote 17lands public datasets: what exists on the server, what is new or changed compared with the local data folder, download.

The public datasets page (https://www.17lands.com/public_datasets) is fed by a small JSON API, one request for the complete
list of file URLs (plus the date 17lands says each set/event was last updated):

    GET https://www.17lands.com/api/public_datasets   -> [{expansion, event_type, last_updated, urls: {game, draft, replay}}]

If that fails, the older source is used as a fallback: the page used to be a Prismic CMS document (two requests):

    GET https://17lands.cdn.prismic.io/api/v2                      -> refs[] (the master ref)
    GET .../documents/search?ref=<ref>&q=[[at(document.type,"public-data")]]  -> results[0].data.datasets[]

The files live on data.17landscapes.com (they used to be on an S3 bucket, 17lands-public.s3.amazonaws.com, which now answers
403 for everything; old-host URLs found in a stored listing, the state file or the Prismic page are rewritten to the new host,
`canonical_url`).  The file host cannot be listed but HEAD on a file URL returns ETag, Last-Modified and Content-Length; that is
how a local copy is judged:

* NEW       no local file with the same stem (``game_data_public.X.Y``; ``.csv.gz`` and uncompressed ``.csv`` count as the same);
* UPDATED   the local ``.csv.gz`` has another size than the server file (or another ETag than the one stored when this app
            downloaded it), or the local copy is an uncompressed ``.csv`` older than the server file's Last-Modified;
* UNREGISTERED  the local file is current, but the app still uses another (older) file for that set/kind/event: only
            registering is needed, nothing to download.

Only what matters to the app is reported: game_data and replay_data of Sealed / TradSealed / PremierDraft / TradDraft (any other
event only when a local copy exists), draft_data as optional.  To stay quick only the files of "priority" sets are HEAD-ed (sets
known to the app plus the newest few sets); other sets of the data folder only when 17lands' own "last updated" date is newer
than the local files.  Nothing here ever raises on network problems: offline simply yields an empty check with `error` set.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Callable

from sealed_lab import config
from sealed_lab.types import Cancelled

log = logging.getLogger(__name__)

LISTING_API = "https://www.17lands.com/api/public_datasets"
PRISMIC_API = "https://17lands.cdn.prismic.io/api/v2"
PRISMIC_QUERY = '[[at(document.type,"public-data")]]'
USER_AGENT = config.user_agent()
# The folder an earlier downloader setup used (kept so an existing setup keeps finding its files without a setting).
LEGACY_DATA_DIR = str(Path.home() / "Documents" / "Claude" / "17LandsData" / "17LandsData")

# Events offered for download.  The extra draft events (PickTwo, QuickDraft: config.EXTRA_DRAFT_EVENTS) are never offered: they are
# optional, the model keeps them only if they pass a hold-out test.  Game files of them that are already on disk are still checked
# for updates (a local file is always wanted, see `_wanted`) and the folder scan registers them.
RELEVANT_EVENTS = ("Sealed", "TradSealed", "PremierDraft", "TradDraft")
KIND_ORDER = {"game": 0, "replay": 1, "draft": 2}
OPTIONAL_KINDS = ("draft",)
NEWEST_SETS = 3                 # newest sets (by 17lands' date) that are always checked
HEAD_TIMEOUT = (4.0, 6.0)       # connect, read
HEAD_BUDGET_S = 8.0             # total wall-clock budget for all HEAD requests
HEAD_WORKERS = 8
LISTING_TTL_S = 15 * 60         # a check younger than this reuses the stored listing (no network) unless forced
HEAD_TTL_S = 15 * 60
MTIME_SLACK_S = 120.0
MAX_LISTING_BYTES = 20_000_000   # listing JSON (API or Prismic CMS) is well under 1 MB
MAX_DATASET_BYTES = 4 << 30      # 4 GiB safety ceiling when a server omits Content-Length (chunked downloads)

# set and event: letters, digits, _ and - only (the name becomes <data folder>/<SET>/<name>: no separators, drive letters or "..")
_URL_FILE_RE = re.compile(r"^(game|draft|replay)_data_public\.([A-Za-z0-9_-]+)\.([A-Za-z0-9_-]+)\.csv(\.gz)?$", re.IGNORECASE)
DOWNLOAD_HOST = "https://data.17landscapes.com/"                 # the only place dataset files are downloaded from
OLD_DOWNLOAD_HOST = "https://17lands-public.s3.amazonaws.com/"   # dead since 2026 (403 for every file); only ever rewritten
DOWNLOAD_HOSTS = (DOWNLOAD_HOST, OLD_DOWNLOAD_HOST)               # hosts whose links are recognised (and canonicalised)
_URL_RE = re.compile(r"https://(?:data\.17landscapes\.com|17lands-public\.s3\.amazonaws\.com)/[^\s\"'<>\\]+")
_DATE_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


def canonical_url(url):
    """`url` on the current file host: an old-host URL (exact `https://17lands-public.s3.amazonaws.com/` prefix) gets the same path
    on `DOWNLOAD_HOST`; anything else (other hosts, lookalikes, non-strings) is returned unchanged."""
    if isinstance(url, str) and url.startswith(OLD_DOWNLOAD_HOST):
        return DOWNLOAD_HOST + url[len(OLD_DOWNLOAD_HOST):]
    return url


# ---------------------------------------------------------------------------------------------
# data classes
# ---------------------------------------------------------------------------------------------

@dataclass
class RemoteFile:
    url: str
    name: str
    kind: str             # game | replay | draft
    set_code: str
    event: str
    row_date: str = ""    # 17lands' "last updated" for the set/event row ("" when unknown)


@dataclass
class RemoteListing:
    files: list[RemoteFile]
    published: str = ""   # Prismic last_publication_date of the page


@dataclass
class DatasetItem:
    url: str
    name: str
    kind: str
    set_code: str
    event: str
    status: str                       # new | updated | unregistered
    size: int | None = None           # server size in bytes (None if the HEAD failed)
    last_modified: str = ""           # server Last-Modified, ISO 8601 UTC
    etag: str = ""
    local_path: str = ""              # existing local file (updated / unregistered)
    local_size: int | None = None
    optional: bool = False            # not pre-selected (draft data)
    set_has_model: bool = False       # the app has a model for this set
    set_in_app: bool = False          # the app knows this set
    reason: str = ""
    row_date: str = ""                # 17lands' "last updated" date of the set/event

    @property
    def signature(self) -> str:
        """Identifies this version of the remote file (what "don't ask again" remembers)."""
        return f"{self.etag or ''}|{self.size if self.size is not None else ''}|{self.last_modified}"

    @property
    def needs_download(self) -> bool:
        return self.status != "unregistered"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DatasetCheck:
    items: list[DatasetItem] = field(default_factory=list)
    data_dir: str = ""
    checked_at: str = ""
    error: str = ""                   # non-empty when the check could not run (offline ...)
    from_cache: bool = False          # listing came from the stored one (checked a few minutes ago)
    elapsed: float = 0.0
    n_remote: int = 0                 # files in the 17lands listing
    n_head: int = 0                   # HEAD requests made
    n_head_failed: int = 0
    n_dismissed: int = 0              # hidden because the user said "don't ask again" for this version
    published: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def sets(self) -> list[str]:
        return sorted({i.set_code for i in self.items})

    def by_set(self) -> dict[str, list[DatasetItem]]:
        out: dict[str, list[DatasetItem]] = {}
        for it in self.items:
            out.setdefault(it.set_code, []).append(it)
        return out


@dataclass
class DownloadReport:
    downloaded: list[str] = field(default_factory=list)       # paths written
    registered: list = field(default_factory=list)            # FileInfo of the files added to the app
    to_register: list[str] = field(default_factory=list)      # paths to register (downloaded + current-but-unregistered)
    failed: list[tuple[str, str]] = field(default_factory=list)   # (file name, error)
    cancelled: bool = False
    stale_sets: list[str] = field(default_factory=list)       # sets that now need "Calculate"
    nbytes: int = 0
    elapsed: float = 0.0

    @property
    def message(self) -> str:
        parts = []
        if self.downloaded:
            parts.append(f"Downloaded {len(self.downloaded)} file(s) ({self.nbytes / 1e6:,.1f} MB)")
        n_reg_only = len(self.registered) - len(self.downloaded)
        if n_reg_only > 0:
            parts.append(f"registered {n_reg_only} file(s) already on disk")
        if self.failed:
            parts.append(f"{len(self.failed)} failed")
        if self.cancelled:
            parts.append("cancelled")
        msg = "; ".join(parts) or "Nothing downloaded"
        if self.stale_sets:
            msg += f". Press Calculate to refit: {', '.join(self.stale_sets)}"
        return msg


# ---------------------------------------------------------------------------------------------
# state file (APP_HOME/datasets_state.json)
# ---------------------------------------------------------------------------------------------

_state_lock = threading.Lock()


def state_path() -> Path:
    return config.app_home() / "datasets_state.json"


def _newer_entry(a: dict, b: dict) -> dict:
    """Of two stored entries for the same file the one downloaded / seen last; on a tie `b`."""
    def rank(e):
        try:
            return (float(e.get("downloaded_at") or 0), float(e.get("seen_at") or 0))
        except (TypeError, ValueError):
            return (0.0, 0.0)
    return a if rank(a) > rank(b) else b


def _normalise_state(d: dict) -> dict:
    """Move everything keyed by (or holding) an old-host URL to the canonical host: the listing (`remote`), the per-file cache
    (`files`: dl_etag / dl_size / ... survive, so earlier downloads are not reported as new or updated) and `dismissed`.  If both
    the old and the canonical key exist, the entry with the later downloaded_at / seen_at wins (the canonical one on a tie)."""
    rem = d.get("remote")
    if isinstance(rem, list):
        seen, out = set(), []
        for r in rem:
            if isinstance(r, dict) and isinstance(r.get("url"), str):
                r = {**r, "url": canonical_url(r["url"])}
                if r["url"] in seen:
                    continue
                seen.add(r["url"])
            out.append(r)
        d["remote"] = out
    files = d.get("files")
    if isinstance(files, dict):
        new: dict = {}
        for k, v in files.items():
            ck = canonical_url(k)
            if ck in new and isinstance(v, dict) and isinstance(new[ck], dict):
                # the canonical key wins ties whichever order the two came in
                v = _newer_entry(new[ck], v) if k == ck else _newer_entry(v, new[ck])
            new[ck] = v
        d["files"] = new
    dis = d.get("dismissed")
    if isinstance(dis, dict):
        new_d: dict = {}
        for k, v in dis.items():
            ck = canonical_url(k)
            if ck == k or ck not in dis:        # a canonical key already there is not overwritten by its old-host twin
                new_d[ck] = v
        d["dismissed"] = new_d
    return d


def load_state() -> dict:
    try:
        d = json.loads(state_path().read_text(encoding="utf-8"))
        return _normalise_state(d) if isinstance(d, dict) else {}
    except OSError:
        return {}
    except ValueError:
        from sealed_lab.ingest import registry

        registry.quarantine(state_path())
        return {}


def save_state(state: dict) -> None:
    try:
        from sealed_lab.ingest import registry

        registry.write_text_atomic(state_path(), json.dumps(state, indent=1))
    except OSError as e:  # pragma: no cover - cosmetic cache
        log.warning("could not write %s: %s", state_path(), e)


def default_data_dir() -> str:
    """Folder holding the 17lands downloads: env SEALED_LAB_DATA_DIR, else the legacy downloader folder if it exists, else a folder
    under the app home."""
    env = os.environ.get("SEALED_LAB_DATA_DIR")
    if env:
        return env
    if Path(LEGACY_DATA_DIR).is_dir():
        return LEGACY_DATA_DIR
    return str(config.app_home() / "downloads")


# ---------------------------------------------------------------------------------------------
# remote listing (17lands JSON API, Prismic as a fallback)
# ---------------------------------------------------------------------------------------------

def _text(field_) -> str:
    if not isinstance(field_, list):
        return ""
    return " ".join(str(b.get("text", "")) for b in field_ if isinstance(b, dict)).strip()


def _links(field_) -> list[str]:
    out: list[str] = []
    if isinstance(field_, list):
        for b in field_:
            if isinstance(b, dict):
                for s in b.get("spans") or []:
                    if isinstance(s, dict) and s.get("type") == "hyperlink":
                        u = ((s.get("data") or {}).get("url") or "").strip()
                        if u:
                            out.append(u)
    return out


def _remote_file(url: str, row_date: str = "") -> RemoteFile | None:
    """A dataset file of the listing, or None for anything else: a link to another host, or a name that is not a plain
    `<kind>_data_public.<SET>.<EVENT>.csv[.gz]` (the listing comes from a CMS: its links are not trusted as paths)."""
    url = canonical_url(url.strip())
    if not url.startswith(DOWNLOAD_HOST) or any(c in url for c in "?#\\"):
        return None
    name = url.rstrip("/").rsplit("/", 1)[-1]
    m = _URL_FILE_RE.match(name)
    if m is None:
        return None
    return RemoteFile(url, name, m.group(1).lower(), m.group(2).upper(), m.group(3), row_date)


def parse_listing(doc: dict) -> RemoteListing:
    """Dataset files of a Prismic `public-data` search response. Structured rows first (they carry the "last updated" date),
    then a regex over the whole JSON so a changed page layout still yields every URL."""
    results = doc.get("results") or []
    page = results[0] if results else {}
    data = page.get("data") or {}
    seen: dict[str, RemoteFile] = {}
    for row in data.get("datasets") or []:
        if not isinstance(row, dict):
            continue
        row_date = ""
        m = _DATE_RE.search(_text(row.get("last_updated")))
        if m:
            row_date = m.group(0)
        for key in ("game_data", "draft_data", "replay_data"):
            for u in _links(row.get(key)):
                rf = _remote_file(u, row_date)
                if rf is not None and rf.url not in seen:
                    seen[rf.url] = rf
    for u in _URL_RE.findall(json.dumps(doc)):
        u = canonical_url(u.strip())
        if u not in seen:
            rf = _remote_file(u)
            if rf is not None:
                seen[rf.url] = rf
    return RemoteListing(sorted(seen.values(), key=lambda r: (r.set_code, r.event, KIND_ORDER.get(r.kind, 9))),
                         str(page.get("last_publication_date") or ""))


def _session():
    import requests

    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    ad = requests.adapters.HTTPAdapter(pool_connections=HEAD_WORKERS, pool_maxsize=HEAD_WORKERS)
    s.mount("https://", ad)
    return s


def parse_api_listing(rows) -> RemoteListing:
    """Dataset files of the 17lands `public_datasets` API response (a list of {expansion, event_type, last_updated, urls}).  Rows
    that are not objects, rows without `urls`, unknown keys under `urls`, non-string links, links to other hosts and file names that
    are not plain dataset names are skipped; each link goes through `_remote_file` like the Prismic ones."""
    if not isinstance(rows, list):
        raise ValueError("unexpected 17lands API response")
    seen: dict[str, RemoteFile] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("urls"), dict):
            continue
        m = _DATE_RE.search(str(row.get("last_updated") or ""))
        row_date = m.group(0) if m else ""
        for u in row["urls"].values():
            if isinstance(u, str):
                rf = _remote_file(u, row_date)
                if rf is not None and rf.url not in seen:
                    seen[rf.url] = rf
    return RemoteListing(sorted(seen.values(), key=lambda r: (r.set_code, r.event, KIND_ORDER.get(r.kind, 9))), "")


def _json_capped(r, cap: int = MAX_LISTING_BYTES):
    """Parse JSON from `r` while refusing bodies larger than `cap` bytes."""
    from sealed_lab.ingest.scryfall import read_capped

    if getattr(r, "iter_content", None) is None:
        return r.json()
    raw = read_capped(r, cap)
    return json.loads(raw.decode("utf-8")) if raw else r.json()


def _check_download_redirect(r) -> None:
    """Reject a response whose final URL (after redirects) left DOWNLOAD_HOST."""
    final = getattr(r, "url", None)
    if isinstance(final, str) and final and not canonical_url(final).startswith(DOWNLOAD_HOST):
        raise ValueError(f"Refusing redirect outside 17lands' file host: {final}")


def list_remote(session=None, timeout: float = 10.0) -> RemoteListing:
    """The 17lands file list: the JSON API (1 request), else the Prismic page (2 requests; also used when the API yields no valid
    file).  Raises requests exceptions / ValueError only when both fail."""
    s = session or _session()
    try:
        r = s.get(LISTING_API, timeout=timeout, stream=True)
        try:
            r.raise_for_status()
            listing = parse_api_listing(_json_capped(r))
        finally:
            close = getattr(r, "close", None)
            if close:
                close()
        if listing.files:
            return listing
        log.warning("17lands API listed no valid dataset file; trying Prismic")
    except Exception as e:  # noqa: BLE001 - any failure of the API means: use the older source
        log.warning("17lands API listing failed (%s: %s); trying Prismic", type(e).__name__, e)
    return list_remote_prismic(s, timeout)


def list_remote_prismic(session=None, timeout: float = 10.0) -> RemoteListing:
    """The file list from the Prismic page (2 requests). Raises requests exceptions / ValueError on failure."""
    s = session or _session()
    refs = s.get(PRISMIC_API, timeout=timeout, stream=True)
    try:
        refs.raise_for_status()
        refs_json = _json_capped(refs)
    finally:
        close = getattr(refs, "close", None)
        if close:
            close()
    master = next((r["ref"] for r in refs_json.get("refs", []) if r.get("isMasterRef")), None)
    if not master:
        raise ValueError("Prismic master ref not found")
    doc = s.get(PRISMIC_API + "/documents/search", params={"ref": master, "q": PRISMIC_QUERY}, timeout=timeout, stream=True)
    try:
        doc.raise_for_status()
        doc_json = _json_capped(doc)
    finally:
        close = getattr(doc, "close", None)
        if close:
            close()
    return parse_listing(doc_json)


def head(url: str, session=None) -> dict | None:
    """{size, last_modified (ISO), etag} of a remote file, or None (missing, 403, timeout)."""
    s = session or _session()
    try:
        r = s.head(url, timeout=HEAD_TIMEOUT, allow_redirects=True)
        _check_download_redirect(r)
    except Exception as e:  # noqa: BLE001 - network problems are expected
        log.debug("HEAD %s failed: %s", url, e)
        return None
    if r.status_code != 200:
        return None
    lm = ""
    try:
        lm = parsedate_to_datetime(r.headers["Last-Modified"]).astimezone(timezone.utc).isoformat(timespec="seconds")
    except (KeyError, TypeError, ValueError):
        pass
    try:
        size = int(r.headers["Content-Length"])
    except (KeyError, ValueError):
        size = None
    return {"size": size, "last_modified": lm, "etag": (r.headers.get("ETag") or "").strip('"')}


# ---------------------------------------------------------------------------------------------
# local side
# ---------------------------------------------------------------------------------------------

@dataclass
class LocalFile:
    path: str
    size: int
    mtime: float

    @property
    def compressed(self) -> bool:
        return self.path.lower().endswith(".gz")


def _stem(name: str) -> str | None:
    """'game_data_public.MSH.TradSealed.csv.gz' -> 'game_data_public.MSH.TradSealed' (None if not a dataset file)."""
    m = _URL_FILE_RE.match(name)
    return name[:m.end(3)].casefold() if m else None


def _scan_local(data_dir: str, extra_paths: list[str]) -> dict[str, list[LocalFile]]:
    out: dict[str, list[LocalFile]] = {}
    seen: set[str] = set()

    def add(path: str) -> None:
        key = os.path.normcase(os.path.abspath(path))
        if key in seen:
            return
        stem = _stem(os.path.basename(path))
        if stem is None:
            return
        try:
            st = os.stat(path)
        except OSError:
            return
        seen.add(key)
        out.setdefault(stem, []).append(LocalFile(path, st.st_size, st.st_mtime))

    if data_dir and os.path.isdir(data_dir):
        for root, _dirs, files in os.walk(data_dir):
            for fn in files:
                add(os.path.join(root, fn))
    for p in extra_paths:
        add(p)
    return out


def _best_local(files: list[LocalFile]) -> LocalFile:
    """The compressed copy if there is one, else the newest."""
    gz = [f for f in files if f.compressed]
    return max(gz or files, key=lambda f: f.mtime)


def _registry_view() -> tuple[set[str], dict[str, dict[tuple[str, str], dict]]]:
    """(known set codes, {set: {(kind, event): registry entry}})."""
    from sealed_lab.ingest import registry

    sets = set(registry.known_sets())
    reg: dict[str, dict[tuple[str, str], dict]] = {}
    for s in sets:
        try:
            reg[s] = {(d.get("kind"), d.get("event")): d for d in registry.entries(s)}
        except Exception:  # noqa: BLE001
            reg[s] = {}
    return sets, reg


def _row_date(s: str) -> date | None:
    m = _DATE_RE.search(s or "")
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


def _iso_to_ts(s: str) -> float | None:
    try:
        return datetime.fromisoformat(s).timestamp()
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------------------------
# the check
# ---------------------------------------------------------------------------------------------

def _wanted(rf: RemoteFile, local: bool) -> bool:
    if local:
        return True
    return rf.event in RELEVANT_EVENTS and rf.kind in KIND_ORDER


def _priority_sets(files: list[RemoteFile], app_sets: set[str]) -> set[str]:
    newest: dict[str, date] = {}
    for rf in files:
        d = _row_date(rf.row_date)
        if d is not None and (rf.set_code not in newest or d > newest[rf.set_code]):
            newest[rf.set_code] = d
    ranked = sorted(newest, key=lambda s: (newest[s], s), reverse=True)[:NEWEST_SETS]
    return set(ranked) | set(app_sets)


def _candidates(files: list[RemoteFile], local: dict[str, list[LocalFile]], priority: set[str]) -> list[RemoteFile]:
    """Files worth a HEAD request."""
    local_by_set: dict[str, float] = {}
    for rf in files:
        for lf in local.get(_stem(rf.name) or "", []):
            local_by_set[rf.set_code] = max(local_by_set.get(rf.set_code, 0.0), lf.mtime)
    out = []
    for rf in files:
        has_local = (_stem(rf.name) or "") in local
        if rf.set_code in priority:
            if _wanted(rf, has_local):
                out.append(rf)
        elif has_local:
            rd = _row_date(rf.row_date)
            newest = local_by_set.get(rf.set_code, 0.0)
            # 17lands says the set changed after our newest local file of that set was written
            if rd is not None and newest and rd > date.fromtimestamp(newest):
                out.append(rf)
    return out


def _fmt_mb(n: int | None) -> str:
    return "?" if n is None else (f"{n / 1e6:,.1f} MB" if n >= 1e5 else f"{n:,} B")


def _stale_reason(h: dict, lf: LocalFile, stored: dict | None = None) -> str:
    """Why a local copy is older than the server file ("" when it is current)."""
    if lf.compressed:
        if h["size"] is not None and h["size"] != lf.size:
            return f"size differs: yours {_fmt_mb(lf.size)}, server {_fmt_mb(h['size'])}"
        # the ETag / size of the copy THIS app downloaded are kept apart from the latest HEAD (which overwrites "etag" / "size"
        # before the judgement), otherwise the two could never differ
        if (stored and stored.get("dl_etag") and h["etag"] and stored["dl_etag"] != h["etag"]
                and stored.get("dl_size") == lf.size):
            return "server file changed since you downloaded it"
        return ""
    lm = _iso_to_ts(h["last_modified"])
    if lm is not None and lm > lf.mtime + MTIME_SLACK_S:
        return (f"server file modified {h['last_modified'][:10]}, your uncompressed copy is from "
                f"{datetime.fromtimestamp(lf.mtime).date()}")
    return ""


def _unregistered_reason(h: dict, reg_path: str, lf: LocalFile) -> str:
    """Why the file the app uses (`reg_path`) is not the current copy `lf` on disk ("" when it is as good as `lf`).

    A compressed `lf` that matches the server size IS the server content; the registered file is judged against it: a registered
    .csv is only as good as `lf` if it is the same data (the .gz records its uncompressed size), whatever the file times say."""
    from sealed_lab.ingest import registry

    base = os.path.basename(reg_path)
    try:
        st = os.stat(reg_path)
    except OSError:
        return "the file the app uses is missing"
    reg_lf = LocalFile(reg_path, st.st_size, st.st_mtime)
    if lf.compressed and (h["size"] is None or h["size"] == lf.size):
        if reg_lf.compressed:
            why = _stale_reason(h, reg_lf)
            return f"the app still uses {base}, which is outdated ({why})" if why else ""
        same = registry.same_content(reg_path, lf.path)
        if same is True:
            return ""
        return (f"the app uses the uncompressed {base}, which {'differs from' if same is False else 'cannot be compared with'} the "
                f"{os.path.basename(lf.path)} that matches the server file")
    why = _stale_reason(h, reg_lf)
    return f"the app still uses {base}, which is outdated ({why})" if why else ""


def _judge(rf: RemoteFile, h: dict, locals_: list[LocalFile] | None, stored: dict | None) -> tuple[str, str, LocalFile | None] | None:
    """(status, reason, local file) for a HEAD-ed remote file, or None when the local copy is current."""
    if not locals_:
        return "new", "not in your data folder", None
    lf = _best_local(locals_)
    why = _stale_reason(h, lf, stored)
    return ("updated", why, lf) if why else None


def _norm(p: str) -> str:
    return os.path.normcase(os.path.abspath(p))


def check(data_dir: str | None = None, force: bool = False, session=None, budget: float = HEAD_BUDGET_S) -> DatasetCheck:
    """New / updated / unregistered 17lands files compared with the local data folder.  Never raises."""
    t0 = time.monotonic()
    data_dir = str(data_dir or default_data_dir())
    res = DatasetCheck(data_dir=data_dir, checked_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    try:
        _check(res, data_dir, force, session, budget)
    except Exception as e:  # noqa: BLE001 - startup must never fail because of this
        log.warning("dataset check failed: %s", e)
        res.error = f"{type(e).__name__}: {e}"
        res.items = []
    res.elapsed = time.monotonic() - t0
    return res


def _check(res: DatasetCheck, data_dir: str, force: bool, session, budget: float) -> None:
    with _state_lock:
        state = load_state()
    now = time.time()
    cached = state.get("remote") or []
    fresh = bool(not force and cached and now - float(state.get("listing_at", 0)) < LISTING_TTL_S)
    sess = session or _session()
    if fresh:
        files = [RemoteFile(**d) for d in cached]
        res.published = state.get("published", "")
        res.from_cache = True
    else:
        try:
            listing = list_remote(sess)
        except Exception as e:  # noqa: BLE001
            res.error = f"offline or 17lands unreachable ({type(e).__name__})"
            res.notes.append(res.error)
            return
        files, res.published = listing.files, listing.published
        state["remote"] = [asdict(f) for f in files]
        state["listing_at"] = now
        state["published"] = res.published
    res.n_remote = len(files)

    app_sets, reg = _registry_view()
    reg_paths = [d.get("path", "") for ents in reg.values() for d in ents.values()]
    local = _scan_local(data_dir, [p for p in reg_paths if p])
    priority = _priority_sets(files, app_sets)
    cands = _candidates(files, local, priority)

    stored: dict[str, dict] = state.setdefault("files", {})
    heads: dict[str, dict | None] = {}
    todo = []
    for rf in cands:
        s = stored.get(rf.url)
        if not force and s and now - float(s.get("seen_at", 0)) < HEAD_TTL_S and s.get("size") is not None:
            heads[rf.url] = {"size": s.get("size"), "last_modified": s.get("last_modified", ""), "etag": s.get("etag", "")}
        else:
            todo.append(rf)
    if todo:
        ex = ThreadPoolExecutor(max_workers=HEAD_WORKERS, thread_name_prefix="head")
        futs = {ex.submit(head, rf.url, sess): rf for rf in todo}
        done, pending = wait(list(futs), timeout=budget)
        for f in done:
            heads[futs[f].url] = f.result()
        for f in pending:
            f.cancel()
            heads[futs[f].url] = None
            res.notes.append(f"timed out: {futs[f].name}")
        ex.shutdown(wait=False, cancel_futures=True)
        res.n_head = len(todo)
        res.n_head_failed = sum(1 for rf in todo if heads.get(rf.url) is None)
        for rf in todo:
            h = heads.get(rf.url)
            if h is not None:
                prev = stored.get(rf.url, {})
                stored[rf.url] = {**prev, **h, "seen_at": now}

    items: list[DatasetItem] = []
    for rf in cands:
        h = heads.get(rf.url)
        if h is None:
            continue
        stem = _stem(rf.name) or ""
        locals_ = local.get(stem)
        verdict = _judge(rf, h, locals_, stored.get(rf.url))
        has_model = bool(config.valid_set_code(rf.set_code) and (config.model_dir(rf.set_code) / "meta.json").exists())
        base = dict(url=rf.url, name=rf.name, kind=rf.kind, set_code=rf.set_code, event=rf.event, size=h["size"],
                    last_modified=h["last_modified"], etag=h["etag"], optional=rf.kind in OPTIONAL_KINDS,
                    set_has_model=has_model, set_in_app=rf.set_code in app_sets, row_date=rf.row_date)
        if verdict is not None:
            status, reason, lf = verdict
            items.append(DatasetItem(status=status, reason=reason, local_path=lf.path if lf else "",
                                     local_size=lf.size if lf else None, **base))
            continue
        # the local file is current: does the app still use an older one for this set / kind / event?
        if locals_ and rf.set_code in app_sets:
            lf = _best_local(locals_)
            ent = reg.get(rf.set_code, {}).get((rf.kind, rf.event))
            if ent is not None and _norm(ent.get("path", "")) != _norm(lf.path):
                why = _unregistered_reason(h, ent["path"], lf)
                if why:
                    items.append(DatasetItem(status="unregistered", reason=f"current copy on disk, not loaded: {why}",
                                             local_path=lf.path, local_size=lf.size, **base))

    dismissed: dict[str, str] = state.get("dismissed") or {}
    if not force:
        kept = []
        for it in items:
            if dismissed.get(canonical_url(it.url)) == it.signature:
                res.n_dismissed += 1
            else:
                kept.append(it)
        items = kept
    items.sort(key=lambda i: (i.set_code, KIND_ORDER.get(i.kind, 9), i.event))
    res.items = items

    # prune the per-URL cache to the current listing and save
    live = {f.url for f in files}
    state["files"] = {u: v for u, v in stored.items() if u in live}
    state["last_check"] = res.checked_at
    with _state_lock:
        old = load_state()
        old.update({k: state[k] for k in ("remote", "listing_at", "published", "files", "last_check") if k in state})
        save_state(old)


def registry_sets() -> set[str]:
    from sealed_lab.ingest import registry

    return set(registry.known_sets())


def items_for_set(set_code: str, data_dir: str | None = None) -> list[DatasetItem]:
    """The files 17lands lists for a set (stored listing, no network) that are not in the data folder yet, as `DatasetItem`s ready for
    `download`: game / replay data of Sealed, TradSealed, PremierDraft, TradDraft (draft data optional, not pre-selected).  Empty when
    the set is not in the stored listing."""
    sc = set_code.upper()
    state = load_state()
    remote = [RemoteFile(**d) for d in (state.get("remote") or []) if str(d.get("set_code", "")).upper() == sc]
    local = _scan_local(str(data_dir or default_data_dir()), [])
    stored = state.get("files") or {}
    has_model = bool(config.valid_set_code(sc) and (config.model_dir(sc) / "meta.json").exists())
    in_app = sc in registry_sets()
    out: list[DatasetItem] = []
    for rf in remote:
        if not _wanted(rf, False) or local.get(_stem(rf.name) or ""):
            continue
        h = stored.get(rf.url) or {}
        out.append(DatasetItem(url=rf.url, name=rf.name, kind=rf.kind, set_code=sc, event=rf.event, status="new", size=h.get("size"),
                               last_modified=h.get("last_modified", ""), etag=h.get("etag", ""), optional=rf.kind in OPTIONAL_KINDS,
                               set_has_model=has_model, set_in_app=in_app, reason="not in your data folder", row_date=rf.row_date))
    out.sort(key=lambda i: (KIND_ORDER.get(i.kind, 9), i.event))
    return out


def dismiss(items: list[DatasetItem]) -> None:
    """Remember "don't ask again for these": the current version of each file stays hidden until it changes again."""
    with _state_lock:
        st = load_state()
        d = st.setdefault("dismissed", {})
        for it in items:
            d[canonical_url(it.url)] = it.signature
        save_state(st)


def clear_dismissed() -> None:
    with _state_lock:
        st = load_state()
        st["dismissed"] = {}
        save_state(st)


# ---------------------------------------------------------------------------------------------
# download
# ---------------------------------------------------------------------------------------------

def destination(item: DatasetItem, data_dir: str) -> Path:
    """Where an item is written: next to the existing compressed copy inside `data_dir` (updates), else <data_dir>/<SET>/<name>."""
    if item.local_path and item.local_path.lower().endswith(".gz") and item.name.lower().endswith(".gz"):
        try:
            Path(item.local_path).resolve().relative_to(Path(data_dir).resolve())
            return Path(item.local_path)
        except (ValueError, OSError):
            pass
    if not (_URL_FILE_RE.match(item.name) and re.fullmatch(r"[A-Za-z0-9_-]+", item.set_code)):
        raise ValueError(f"Unsafe dataset file name: {item.set_code!r} / {item.name!r}")
    return Path(data_dir) / item.set_code.upper() / item.name


def download_one(item: DatasetItem, dest: Path, on_bytes: Callable[[int], None] | None = None,
                 cancel: threading.Event | None = None, session=None, max_bytes: int = MAX_DATASET_BYTES) -> int:
    """Stream `item.url` to `<dest>.part`, check the size, rename.  Returns bytes written; raises Cancelled / OSError / requests errors."""
    url = canonical_url(item.url)
    if not url.startswith(DOWNLOAD_HOST):
        raise ValueError(f"Refusing to download from outside 17lands' file host: {item.url}")
    s = session or _session()
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    n = 0
    try:
        with s.get(url, stream=True, timeout=(10, 60)) as r:
            _check_download_redirect(r)
            r.raise_for_status()
            expect = int(r.headers.get("Content-Length") or 0)
            if expect > max_bytes:
                raise ValueError(f"Dataset file exceeds the {max_bytes:,} byte limit ({expect:,} bytes)")
            with open(part, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    if cancel is not None and cancel.is_set():
                        raise Cancelled()
                    if not chunk:
                        continue
                    n += len(chunk)
                    if n > max_bytes:
                        raise ValueError(f"Dataset download exceeds the {max_bytes:,} byte limit")
                    f.write(chunk)
                    if on_bytes is not None:
                        on_bytes(len(chunk))
        if n == 0:
            raise OSError("empty download")
        if expect and n != expect:
            raise OSError(f"incomplete download ({n:,} of {expect:,} bytes)")
        os.replace(part, dest)
    finally:
        try:
            if part.exists():
                part.unlink()
        except OSError:
            pass
    return n


def download(items: list[DatasetItem], data_dir: str, progress: Callable[[float, str], None] | None = None,
             cancel: threading.Event | None = None, session=None) -> DownloadReport:
    """Download the items that need it (sequentially) into `data_dir`.  Returns paths of all files to register
    (downloaded + 'unregistered' ones already on disk) in `report.to_register`.  Never raises Cancelled: a cancel stops
    after the current chunk and is reported."""
    t0 = time.monotonic()
    rep = DownloadReport()
    todo = [i for i in items if i.needs_download]
    total = sum(i.size or 0 for i in todo) or 1
    done_bytes = [0]
    sess = session or _session()
    paths_ok: list[str] = []
    for k, it in enumerate(todo):
        if cancel is not None and cancel.is_set():
            rep.cancelled = True
            break
        dest = destination(it, data_dir)

        def on_bytes(nb: int, it=it, k=k) -> None:
            done_bytes[0] += nb
            if progress is not None:
                progress(min(0.99, done_bytes[0] / total),
                         f"Downloading {it.name} ({k + 1}/{len(todo)}): {done_bytes[0] / 1e6:,.0f} of {total / 1e6:,.0f} MB")
        try:
            if progress is not None:
                progress(min(0.99, done_bytes[0] / total), f"Downloading {it.name} ({k + 1}/{len(todo)})")
            rep.nbytes += download_one(it, dest, on_bytes, cancel, sess)
        except Cancelled:
            rep.cancelled = True
            break
        except Exception as e:  # noqa: BLE001
            log.warning("download of %s failed: %s", it.url, e)
            rep.failed.append((it.name, f"{type(e).__name__}: {e}"))
            continue
        paths_ok.append(str(dest))
        rep.downloaded.append(str(dest))
        with _state_lock:
            st = load_state()
            st.setdefault("files", {})[canonical_url(it.url)] = {"etag": it.etag, "size": it.size, "last_modified": it.last_modified,
                                                  "seen_at": time.time(), "downloaded_at": time.time(),
                                                  "dl_etag": it.etag, "dl_size": it.size}
            st.get("dismissed", {}).pop(canonical_url(it.url), None)
            save_state(st)
    rep.elapsed = time.monotonic() - t0
    rep.to_register = paths_ok + [i.local_path for i in items if i.status == "unregistered" and i.local_path]
    if progress is not None:
        progress(1.0, "Download finished" if not rep.cancelled else "Download cancelled")
    return rep
