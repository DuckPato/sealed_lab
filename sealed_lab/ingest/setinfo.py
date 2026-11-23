"""Set release dates (for ordering sets), from Scryfall's set list, cached in APP_HOME/scryfall_sets.json.

Offline-safe: without a network or a cached entry a set has no date and sorts after the dated ones.
"""
from __future__ import annotations

import json
import logging
import time

from sealed_lab import config

log = logging.getLogger(__name__)

URL = "https://api.scryfall.com/sets"
REFETCH_AFTER = 24 * 3600      # an unknown set triggers at most one fetch per day


def _cache_path():
    return config.app_home() / "scryfall_sets.json"


def _load() -> dict:
    try:
        return json.loads(_cache_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _fetch() -> dict[str, str] | None:
    try:
        import requests

        from sealed_lab.config import user_agent

        r = requests.get(URL, timeout=10, headers={"User-Agent": user_agent(), "Accept": "application/json"})
        r.raise_for_status()
        return {s["code"].upper(): s["released_at"] for s in r.json().get("data", []) if s.get("code") and s.get("released_at")}
    except Exception as e:  # noqa: BLE001 - offline-safe
        log.info("Scryfall set list not fetched: %s", e)
        return None


def release_dates(codes, fetch: bool = True) -> dict[str, str]:
    """{set code: 'YYYY-MM-DD'} for the codes with a known release date. With `fetch`, a code missing from the cache refreshes
    the cache from Scryfall (at most once per REFETCH_AFTER)."""
    codes = {c.upper() for c in codes if c}
    cache = _load()
    dates = cache.get("sets") or {}
    if fetch and codes - set(dates) and time.time() - cache.get("fetched_at", 0) > REFETCH_AFTER:
        new = _fetch()
        cache = dict(fetched_at=time.time(), sets=new if new is not None else dates)
        try:
            _cache_path().parent.mkdir(parents=True, exist_ok=True)
            _cache_path().write_text(json.dumps(cache), encoding="utf-8")
        except OSError:
            pass
        dates = cache["sets"]
    return {c: dates[c] for c in codes if c in dates}


def order_sets(codes, dates: dict[str, str]) -> list[str]:
    """Newest release first; sets without a date last, alphabetical."""
    codes = list(dict.fromkeys(codes))
    dated = sorted((c for c in codes if c.upper() in dates), key=lambda c: (dates[c.upper()], c), reverse=True)
    return dated + sorted(c for c in codes if c.upper() not in dates)
