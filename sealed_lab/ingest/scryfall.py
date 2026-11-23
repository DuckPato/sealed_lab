"""Scryfall data: fetch + cache (rate limited, offline-safe) and local json/bulk file import."""
from __future__ import annotations

import gzip
import json
import logging
import os
import re
import threading
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator

from sealed_lab.config import BASICS, cache_dir
from sealed_lab.types import Cancelled

log = logging.getLogger(__name__)

API = "https://api.scryfall.com"
HEADERS = {"User-Agent": "sealed_lab/0.1", "Accept": "application/json"}
CACHE_NAME = "scryfall.json"
MIN_INTERVAL = 0.125  # <= 8 requests/second
NAMED_FACTOR = 4.4    # /cards/named: <= 2 requests/second
KEEP = ("name", "mana_cost", "cmc", "type_line", "colors", "color_identity", "rarity", "produced_mana",
        "oracle_text", "layout", "set", "power", "toughness", "keywords", "printed_name")
FACE_KEEP = ("name", "mana_cost", "type_line", "oracle_text", "colors", "produced_mana", "power", "toughness", "printed_name")
# version of the trimmed fields stored in the cache; caches written with fewer fields (no power / toughness / keywords)
# are refreshed once from Scryfall when the network is allowed. 3: printed_name (OM1's Omenpaths names are printed names of
# Spider-Man cards; 17lands uses the printed name)
FIELDS_VERSION = 3


def trim_card(o: dict) -> dict:
    out = {k: o[k] for k in KEEP if k in o}
    if o.get("card_faces"):
        out["card_faces"] = [{k: f[k] for k in FACE_KEEP if k in f} for f in o["card_faces"]]
    return out


def fold(s: str) -> str:
    """Casefold, strip accents and punctuation: the loose key used for name matching."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return "".join(ch for ch in s.casefold() if ch.isalnum() or ch == " " or ch == "/").strip()


def front(name: str) -> str:
    return name.split(" // ")[0].strip()


def iter_json_cards(path: str | Path) -> Iterator[dict]:
    """Stream the objects of a (possibly huge, possibly gzipped) JSON array, or of a `{"data": [...]}` wrapper."""
    p = str(path)
    with open(p, "rb") as f:
        gz = f.read(2) == b"\x1f\x8b"
    opener = gzip.open if gz else open
    dec = json.JSONDecoder()
    with opener(p, "rt", encoding="utf-8-sig") as f:
        buf = ""
        pos = 0
        started = False
        eof = False
        need_more = False
        while True:
            if not eof and (need_more or len(buf) - pos < (1 << 20)):
                need_more = False
                chunk = f.read(8 << 20)
                if chunk:
                    buf = buf[pos:] + chunk
                    pos = 0
                else:
                    eof = True
            if not started:
                while pos < len(buf) and buf[pos].isspace():
                    pos += 1
                if pos >= len(buf):
                    return
                if buf[pos] == "[":
                    pos += 1
                    started = True
                elif buf[pos] == "{":
                    # wrapper object: find '"data"' array start
                    idx = buf.find('"data"', pos)
                    if idx < 0:
                        if eof:
                            return
                        need_more = True
                        continue
                    br = buf.find("[", idx)
                    if br < 0:
                        if eof:
                            return
                        need_more = True
                        continue
                    pos = br + 1
                    started = True
                else:
                    return
            while True:
                while pos < len(buf) and (buf[pos].isspace() or buf[pos] == ","):
                    pos += 1
                if pos >= len(buf):
                    break
                if buf[pos] == "]":
                    return
                try:
                    obj, end = dec.raw_decode(buf, pos)
                except ValueError:
                    break  # incomplete object; need more data
                yield obj
                pos = end
            if eof:
                return


def cache_path(set_code: str) -> Path:
    return cache_dir(set_code) / CACHE_NAME


def load_cache(set_code: str) -> dict:
    p = cache_path(set_code)
    if p.exists():
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(d, dict) and "cards" in d:
                return d
        except (OSError, ValueError):
            log.warning("corrupt scryfall cache %s ignored", p)
    return {"version": 1, "set": set_code, "cards": [], "sources": {}, "searched": False, "not_found": [],
            "fields": FIELDS_VERSION}


def save_cache(set_code: str, cache: dict) -> None:
    p = cache_path(set_code)
    p.parent.mkdir(parents=True, exist_ok=True)
    cache["updated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, p)


def _names_of(o: dict) -> list[str]:
    names = [o.get("name", "")]
    for f in o.get("card_faces") or []:
        if f.get("name"):
            names.append(f["name"])
    return names


def _printed_names_of(o: dict) -> list[str]:
    names = [o["printed_name"]] if o.get("printed_name") else []
    return names + [f["printed_name"] for f in o.get("card_faces") or [] if f.get("printed_name")]


def build_index(cards: list[dict]) -> dict[str, dict]:
    """Lookup keys (exact, face names, folded, folded front) -> card. First card wins."""
    idx: dict[str, dict] = {}
    for o in cards:
        for n in _names_of(o):
            idx.setdefault(n, o)
            idx.setdefault("~" + fold(n), o)
        idx.setdefault("~" + fold(front(o.get("name", ""))), o)
    # printed names (a reprint under another name, e.g. Through the Omenpaths) after every real name, so they never shadow one
    for o in cards:
        for n in _printed_names_of(o):
            idx.setdefault(n, o)
            idx.setdefault("~" + fold(n), o)
        if o.get("printed_name"):
            idx.setdefault("~" + fold(front(o["printed_name"])), o)
    return idx


def lookup(idx: dict[str, dict], name: str) -> dict | None:
    for key in (name, front(name)):
        if key in idx:
            return idx[key]
    for key in (name, front(name)):
        k = "~" + fold(key)
        if k in idx:
            return idx[k]
    if "?" in name:
        return _lookup_lost_chars(idx, name)
    return None


def _lookup_lost_chars(idx: dict[str, dict], name: str) -> dict | None:
    """17lands files sometimes carry a non-ASCII letter as '?' ("Bespoke B?" for "Bespoke Bō"): each '?' stands for one
    character. Only a single matching card counts, so an ambiguous name stays unmatched."""
    rx = re.compile("".join("." if ch == "?" else re.escape(ch) for ch in name), re.IGNORECASE)
    hits = {id(o): o for k, o in idx.items() if not k.startswith("~") and rx.fullmatch(k)}
    return next(iter(hits.values())) if len(hits) == 1 else None


def _merge(cache: dict, new_cards: list[dict], replace: bool = False) -> int:
    """Add trimmed cards by name. With `replace`, an existing card of the same name is overwritten (field refresh)."""
    pos = {c["name"]: i for i, c in enumerate(cache["cards"])}
    added = 0
    seen: set[str] = set()
    for o in new_cards:
        t = trim_card(o)
        nm = t.get("name")
        if not nm:
            continue
        if nm not in pos:
            cache["cards"].append(t)
            pos[nm] = len(cache["cards"]) - 1
            added += 1
        elif replace and nm not in seen:       # first print of a name wins, like in the search order
            cache["cards"][pos[nm]] = t
        seen.add(nm)
    return added


def import_local_file(path: str | Path, set_code: str, wanted: set[str] | None) -> list[dict]:
    """Read a user-supplied Scryfall json/bulk file; keep cards of this set or whose name is wanted."""
    code = set_code.lower()
    wanted_f = {fold(front(n)) for n in wanted} if wanted else set()
    out = []
    for o in iter_json_cards(path):
        if not isinstance(o, dict) or "name" not in o:
            continue
        if str(o.get("set", "")).lower() == code or (wanted_f and fold(front(o["name"])) in wanted_f):
            out.append(trim_card(o))
    return out


class _Http:
    """Rate-limited GET wrapper around a requests-like session."""

    def __init__(self, session, min_interval: float, cancel: threading.Event | None):
        self.session = session
        self.min_interval = min_interval
        self.last = 0.0
        self.cancel = cancel

    def get(self, url: str, params: dict | None = None):
        if self.cancel is not None and self.cancel.is_set():
            raise Cancelled()
        wait = self.min_interval - (time.monotonic() - self.last)
        if wait > 0:
            time.sleep(wait)
        self.last = time.monotonic()
        r = self.session.get(url, params=params, headers=HEADERS, timeout=(6, 30))
        if getattr(r, "status_code", 200) == 429:
            time.sleep(max(1.0, 8 * self.min_interval))
            self.last = time.monotonic()
            r = self.session.get(url, params=params, headers=HEADERS, timeout=(6, 30))
        return r


def fetch_set_search(set_code: str, http: _Http, progress: Callable[[str], None] | None = None) -> list[dict] | None:
    """Paginated `/cards/search?q=e:{set}&unique=prints`. None on network/HTTP failure; [] if set has no cards."""
    url, params = f"{API}/cards/search", {"q": f"e:{set_code.lower()}", "unique": "prints", "order": "set"}
    out: list[dict] = []
    page = 0
    while url:
        r = http.get(url, params=params)
        params = None
        if r.status_code == 404:
            return out
        if r.status_code != 200:
            log.warning("Scryfall search failed: HTTP %s", r.status_code)
            return None
        j = r.json()
        out.extend(j.get("data", []))
        page += 1
        if progress:
            progress(f"Scryfall page {page}: {len(out)} cards")
        url = j.get("next_page") if j.get("has_more") else None
    return out


def fetch_named(name: str, http: _Http) -> tuple[dict | None, bool]:
    """Returns (card or None, hard_error). 404 is not a hard error."""
    for nm in dict.fromkeys([name, front(name)]):
        r = http.get(f"{API}/cards/named", params={"exact": nm})
        if r.status_code == 200:
            return r.json(), False
        if r.status_code != 404:
            return None, True
    return None, False


def get_scryfall_cards(
    set_code: str,
    names: list[str],
    local_files: list[tuple[str, str]] | None = None,   # (path, fingerprint)
    fetch: bool = True,
    session=None,
    min_interval: float = MIN_INTERVAL,
    progress: Callable[[float, str], None] | None = None,
    cancel: threading.Event | None = None,
) -> tuple[list[dict], list[str]]:
    """Cards (trimmed Scryfall objects) for the set, from cache + local files + (optionally) the network.

    Never raises on network problems; returns (cards, warnings).
    """
    warnings: list[str] = []
    cache = load_cache(set_code)
    dirty = False
    wanted = {n for n in names if n not in BASICS}
    # local files (only re-read when their fingerprint is new)
    for path, fp in local_files or []:
        if fp in cache["sources"]:
            continue
        try:
            got = import_local_file(path, set_code, wanted)
            _merge(cache, got)
            cache["sources"][fp] = {"path": str(path), "cards": len(got)}
            dirty = True
        except Exception as e:  # noqa: BLE001
            warnings.append(f"could not read Scryfall file {Path(path).name}: {e}")
    idx = build_index(cache["cards"])

    def missing() -> list[str]:
        return sorted(n for n in wanted if lookup(idx, n) is None and n not in cache["not_found"])

    stale_fields = bool(cache["searched"]) and int(cache.get("fields", 1)) < FIELDS_VERSION
    if fetch and (missing() or not cache["searched"] or stale_fields):
        if session is None:
            import requests

            session = requests.Session()
        http = _Http(session, min_interval, cancel)
        try:
            if not cache["searched"] or stale_fields:
                if progress:
                    progress(0.0, "Fetching card data from Scryfall...")
                res = fetch_set_search(set_code, http, (lambda m: progress(0.0, m)) if progress else None)
                if res is None:
                    warnings.append("Scryfall unavailable (HTTP error); using fallback card data")
                else:
                    _merge(cache, res, replace=stale_fields)
                    cache["searched"] = True
                    dirty = True
                    idx = build_index(cache["cards"])
                    cache["not_found"] = [n for n in cache["not_found"] if lookup(idx, n) is None]
                    if stale_fields:
                        # cards that came from /cards/named (not in the set search) lack the new fields too: refetch them
                        fresh = {c.get("name") for c in res}
                        redo = [c["name"] for c in cache["cards"] if c["name"] not in fresh]
                        http_r = _Http(session, min_interval * NAMED_FACTOR, cancel)
                        ok = True
                        for i, nm in enumerate(redo):
                            if progress:
                                progress(i / max(len(redo), 1), f"Scryfall refresh {i + 1}/{len(redo)}: {nm}")
                            card, hard = fetch_named(nm, http_r)
                            if hard:
                                ok = False
                                break
                            if card is not None:
                                _merge(cache, [card], replace=True)
                        if ok:
                            cache["fields"] = FIELDS_VERSION
                        else:
                            warnings.append("Scryfall refresh of power / toughness / keywords incomplete; oracle-text fallback used")
                        idx = build_index(cache["cards"])
            miss = missing() if cache["searched"] else []
            fails = 0
            # /cards/named allows ~2 requests/s (429 beyond that): a slower lane than the search endpoint
            http_named = _Http(session, min_interval * NAMED_FACTOR, cancel)
            for i, nm in enumerate(miss):
                if progress:
                    progress(i / max(len(miss), 1), f"Scryfall lookup {i + 1}/{len(miss)}: {nm}")
                card, hard = fetch_named(nm, http_named)
                if hard:
                    fails += 1
                    if fails >= 3:
                        warnings.append("Scryfall lookups failing; stopped")
                        break
                    continue
                fails = 0
                if card is None:
                    cache["not_found"].append(nm)
                else:
                    _merge(cache, [card])
                dirty = True
        except Cancelled:
            if dirty:
                save_cache(set_code, cache)
            raise
        except Exception as e:  # noqa: BLE001 - offline-safe: requests exceptions and anything else
            warnings.append(f"Scryfall unavailable ({type(e).__name__}); using fallback card data")
            log.warning("Scryfall fetch failed: %s", e)
        idx = build_index(cache["cards"])
    if dirty:
        try:
            save_cache(set_code, cache)
        except OSError as e:
            warnings.append(f"could not write Scryfall cache: {e}")
    left = [n for n in wanted if lookup(idx, n) is None]
    if cache["cards"] and left:
        warnings.append(f"{len(left)} card(s) not found in Scryfall data (fallback metadata used)")
    return cache["cards"], warnings
