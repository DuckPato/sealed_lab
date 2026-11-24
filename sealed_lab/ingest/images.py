"""Card images from Scryfall, cached on disk under `APP_HOME/images/<SET>/` (shared by all views, offline-safe).

Image URLs are resolved in batches of 75 names with `POST /cards/collection` (the set's own printing first, then any
printing: bonus-sheet / special-guest cards live in other set codes) and remembered in `images/<SET>/urls.json`.
The files themselves come from Scryfall's image CDN, which is not rate limited the way the API is (the API allows
about 2 requests/s on the card endpoints and answers 429 beyond that).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from pathlib import Path

from sealed_lab.config import BASICS, app_home, valid_set_code
from sealed_lab.ingest.scryfall import API, HEADERS, allowed_url, front, json_capped, read_capped

log = logging.getLogger(__name__)

VERSION = "normal"           # 488 x 680 jpg, ~80 kB
API_INTERVAL = 0.55          # seconds between api.scryfall.com requests
CDN_INTERVAL = 0.05
BATCH = 75
MAX_IMAGE_BYTES = 5_000_000   # a 488 x 680 jpg is ~80 kB
COOLDOWN = 65.0              # after a 429
_lock = threading.RLock()
_last = {"api": 0.0, "cdn": 0.0}
_cool_until = [0.0]
_failed: set[tuple[str, str]] = set()
_urls: dict[str, dict[str, str]] = {}
_session = None


def images_dir(set_code: str) -> Path:
    code = valid_set_code(set_code)
    if code is None:
        raise ValueError(f"invalid set code: {set_code!r}")
    return app_home() / "images" / code


def image_file(set_code: str, name: str) -> Path:
    stem = re.sub(r"[^A-Za-z0-9]+", "_", front(name)).strip("_")[:60] or "card"
    h = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
    return images_dir(set_code) / f"{stem}_{h}.jpg"


def cached_image(set_code: str, name: str) -> Path | None:
    p = image_file(set_code, name)
    return p if p.exists() and p.stat().st_size > 0 else None


def _disabled() -> bool:
    return os.environ.get("SEALED_LAB_NO_IMAGES", "").strip().lower() in ("1", "true", "yes")


def _http():
    global _session
    if _session is None:
        import requests

        _session = requests.Session()
    return _session


def _wait(kind: str) -> None:
    gap = API_INTERVAL if kind == "api" else CDN_INTERVAL
    with _lock:
        delay = gap - (time.monotonic() - _last[kind])
        if delay > 0:
            time.sleep(delay)
        _last[kind] = time.monotonic()


def _url_of(card: dict) -> str | None:
    iu = card.get("image_uris") or ((card.get("card_faces") or [{}])[0].get("image_uris"))
    return (iu or {}).get(VERSION)


def _url_map(set_code: str) -> dict[str, str]:
    sc = set_code.upper()
    m = _urls.get(sc)
    if m is None:
        p = images_dir(sc) / "urls.json"
        try:
            m = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        except (OSError, ValueError):
            m = {}
        _urls[sc] = m
    return m


def _save_urls(set_code: str) -> None:
    p = images_dir(set_code) / "urls.json"
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(_url_map(set_code), ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, p)
    except OSError:
        log.info("could not save image url cache")


def _collection(identifiers: list[dict]) -> tuple[list[dict], bool]:
    """POST /cards/collection. Returns (cards, rate_limited)."""
    _wait("api")
    r = _http().post(f"{API}/cards/collection", json={"identifiers": identifiers}, headers=HEADERS, timeout=(6, 30))
    if r.status_code == 429:
        return [], True
    if r.status_code != 200:
        log.info("Scryfall collection: HTTP %s", r.status_code)
        return [], False
    return list(json_capped(r).get("data", [])), False


def resolve_urls(set_code: str, names: list[str]) -> int:
    """Look up image URLs for the names not resolved yet (batched). Returns how many were added. Never raises."""
    if _disabled() or time.monotonic() < _cool_until[0]:
        return 0
    with _lock:
        m = _url_map(set_code)
        todo = [n for n in dict.fromkeys(names) if n not in m and (set_code.upper(), n) not in _failed]
        if not todo:
            return 0
        added = 0
        try:
            for with_set in (True, False):
                left = [n for n in todo if n not in m]
                for i in range(0, len(left), BATCH):
                    chunk = left[i:i + BATCH]
                    ids = [dict(name=front(n), set=set_code.lower()) if with_set and n not in BASICS else dict(name=front(n))
                           for n in chunk]
                    got, limited = _collection(ids)
                    if limited:
                        _cool_until[0] = time.monotonic() + COOLDOWN
                        return added
                    by_front = {}
                    for c in got:
                        for nm in [c.get("name", "")] + [f.get("name", "") for f in c.get("card_faces") or []]:
                            by_front.setdefault(front(nm).casefold(), c)
                    for n in chunk:
                        c = by_front.get(front(n).casefold())
                        url = _url_of(c) if c else None
                        if url:
                            m[n] = url
                            added += 1
            for n in todo:
                if n not in m:
                    _failed.add((set_code.upper(), n))
        except Exception as e:  # noqa: BLE001 - offline-safe
            log.info("image url lookup failed: %s", e)
        finally:
            if added:
                _save_urls(set_code)
        return added


def fetch_image(set_code: str, name: str, fetch: bool = True) -> Path | None:
    """Local path of the card image, downloading it if needed. Never raises; None when unavailable."""
    if valid_set_code(set_code) is None:
        return None
    hit = cached_image(set_code, name)
    if hit is not None or not fetch or _disabled():
        return hit
    if (set_code.upper(), name) in _failed:
        return None
    url = _url_map(set_code).get(name)
    if url is None:
        resolve_urls(set_code, [name])
        url = _url_map(set_code).get(name)
        if url is None:
            return None
    target = image_file(set_code, name)
    try:
        if not allowed_url(url):
            log.info("image for %s: refusing URL %r (not https on scryfall.io)", name, url)
            _failed.add((set_code.upper(), name))
            return None
        _wait("cdn")
        r = _http().get(url, headers={"User-Agent": HEADERS["User-Agent"]}, timeout=(6, 30), stream=True)
        try:
            final = getattr(r, "url", None)
            if isinstance(final, str) and not allowed_url(final):
                raise ValueError(f"redirected to {final!r}")
            body = read_capped(r, MAX_IMAGE_BYTES) if r.status_code == 200 else b""
        finally:
            close = getattr(r, "close", None)
            if close:
                close()
        if r.status_code == 200 and body:
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(".part")
            tmp.write_bytes(body)
            os.replace(tmp, target)
            return target
        log.info("image for %s: HTTP %s", name, r.status_code)
    except Exception as e:  # noqa: BLE001 - offline-safe
        log.info("image for %s unavailable: %s", name, e)
        return None
    _failed.add((set_code.upper(), name))
    return None
