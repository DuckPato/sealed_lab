"""Purely descriptive opening-hand keep / mulligan summary per set and Sealed format.

Reads `candidate_hand_1` (draws `kind == 3`) and `num_mulligans` from the set's replay shards (`sealed_lab.ingest.replay`).
For each nonland, nonbasic card present in at least `MIN_HANDS` (200) initial 7-card hands (`candidate_hand_1`), reports:
  - how often players kept vs mulliganed initial hands containing the card (`keep_rate`, `mull_rate`),
  - the number of distinct opening hands (`n`),
  - the 95% Wilson confidence interval (`keep_lo`, `keep_hi`, `mull_lo`, `mull_hi`).

Purely descriptive: never computes or shows win rates after keeping/mulliganing, and uses no causal or editorial labels.
Cached per set & format in `sets/<SET>/cache/mulligan_summary_<fmt>.json` (portable across ready-made set bundles).
"""
from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path

import polars as pl

from sealed_lab import config
from sealed_lab.ingest import replay

log = logging.getLogger(__name__)

SUMMARY_VERSION = 1
MIN_HANDS = 200
TOP_N = 10
FORMATS = ("bo1", "bo3")
EVENTS_FOR_FMT = {"bo1": ("Sealed",), "bo3": ("TradSealed",)}
ALL_SEALED_EVENTS = ("Sealed", "TradSealed")


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion k / n."""
    if n <= 0:
        return (0.0, 0.0)
    p = float(k) / float(n)
    z2 = z * z
    denom = 1.0 + z2 / float(n)
    center = (p + z2 / (2.0 * float(n))) / denom
    half = z * math.sqrt(p * (1.0 - p) / float(n) + z2 / (4.0 * float(n) * float(n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def compute_from_tables(
    cards: pl.DataFrame,
    games: pl.DataFrame,
    draws: pl.DataFrame | pl.LazyFrame,
    idmap: dict[int, int],
    *,
    min_hands: int = MIN_HANDS,
    top_n: int = TOP_N,
) -> dict:
    """Compute opening-hand keep/mulligan rates from raw replay tables and card metadata.

    - `games`: columns `gid`, `num_mulligans`
    - `draws`: columns `gid`, `kind`, `id` (filtered to `kind == 3`, i.e., `candidate_hand_1`)
    - `cards`: columns `card_id`, `name`, `is_land`, `is_basic` (optional `colors`, `mana_value`, `rarity`)
    - `idmap`: `{arena_grp_id: corpus_card_id}`
    """
    g = games.select(
        pl.col("gid").cast(pl.Int64),
        (pl.col("num_mulligans").fill_null(0) == 0).alias("kept"),
    ).unique(subset=["gid"])
    d_lf = draws.lazy() if isinstance(draws, pl.DataFrame) else draws
    c1 = (
        d_lf.filter(pl.col("kind") == 3)
        .select(pl.col("gid").cast(pl.Int64), pl.col("id").cast(pl.Int32))
        .collect()
    )
    if g.height == 0 or c1.height == 0:
        return dict(
            n_hands=0,
            n_kept=0,
            n_mull=0,
            base_keep_rate=None,
            base_keep_lo=None,
            base_keep_hi=None,
            base_mull_rate=None,
            base_mull_lo=None,
            base_mull_hi=None,
            min_hands=int(min_hands),
            n_cards_qualifying=0,
            most_kept=[],
            most_mulliganed=[],
        )

    c1_gids = c1.select("gid").unique()
    g_with = g.join(c1_gids, on="gid", how="inner")
    n_hands = int(g_with.height)
    n_kept_tot = int(g_with["kept"].sum()) if n_hands else 0
    n_mull_tot = n_hands - n_kept_tot
    bk_lo, bk_hi = wilson_interval(n_kept_tot, n_hands) if n_hands else (None, None)
    bm_lo, bm_hi = wilson_interval(n_mull_tot, n_hands) if n_hands else (None, None)

    if not idmap or cards.height == 0 or n_hands == 0:
        return dict(
            n_hands=n_hands,
            n_kept=n_kept_tot,
            n_mull=n_mull_tot,
            base_keep_rate=(float(n_kept_tot) / n_hands) if n_hands else None,
            base_keep_lo=bk_lo,
            base_keep_hi=bk_hi,
            base_mull_rate=(float(n_mull_tot) / n_hands) if n_hands else None,
            base_mull_lo=bm_lo,
            base_mull_hi=bm_hi,
            min_hands=int(min_hands),
            n_cards_qualifying=0,
            most_kept=[],
            most_mulliganed=[],
        )

    ii = pl.DataFrame(
        {"id": [int(k) for k in idmap], "card_id": [int(v) for v in idmap.values()]},
        schema={"id": pl.Int32, "card_id": pl.Int64},
    )
    nonland = cards.filter(~pl.col("is_land").fill_null(False) & ~pl.col("is_basic").fill_null(False))
    keep_cols = ["card_id", "name"] + [c for c in ("colors", "mana_value", "rarity") if c in nonland.columns]
    meta_df = nonland.select([pl.col("card_id").cast(pl.Int64)] + [pl.col(c) for c in keep_cols[1:]])

    # Deduplicate (gid, card_id) so multiple copies in the same opening hand count as 1 hand
    per_hand = (
        c1.join(ii, on="id", how="inner")
        .select("gid", "card_id")
        .unique()
        .join(g_with, on="gid", how="inner")
    )
    agg = (
        per_hand.group_by("card_id")
        .agg(pl.len().cast(pl.Int64).alias("n"), pl.col("kept").sum().cast(pl.Int64).alias("n_kept"))
        .filter(pl.col("n") >= int(min_hands))
        .join(meta_df, on="card_id", how="inner")
    )

    rows: list[dict] = []
    for r in agg.iter_rows(named=True):
        n = int(r["n"])
        nk = int(r["n_kept"])
        nm = n - nk
        k_lo, k_hi = wilson_interval(nk, n)
        m_lo, m_hi = wilson_interval(nm, n)
        rows.append(
            dict(
                card_id=int(r["card_id"]),
                name=str(r["name"]),
                colors=str(r.get("colors") or ""),
                mana_value=float(r["mana_value"]) if r.get("mana_value") is not None else None,
                rarity=str(r.get("rarity") or ""),
                n=n,
                n_kept=nk,
                n_mull=nm,
                keep_rate=float(nk) / float(n),
                keep_lo=k_lo,
                keep_hi=k_hi,
                mull_rate=float(nm) / float(n),
                mull_lo=m_lo,
                mull_hi=m_hi,
            )
        )

    most_kept = sorted(rows, key=lambda r: (-r["keep_rate"], -r["n"], r["name"]))[: int(top_n)]
    most_mull = sorted(rows, key=lambda r: (-r["mull_rate"], -r["n"], r["name"]))[: int(top_n)]
    return dict(
        n_hands=n_hands,
        n_kept=n_kept_tot,
        n_mull=n_mull_tot,
        base_keep_rate=float(n_kept_tot) / float(n_hands),
        base_keep_lo=bk_lo,
        base_keep_hi=bk_hi,
        base_mull_rate=float(n_mull_tot) / float(n_hands),
        base_mull_lo=bm_lo,
        base_mull_hi=bm_hi,
        min_hands=int(min_hands),
        n_cards_qualifying=len(rows),
        most_kept=most_kept,
        most_mulliganed=most_mull,
    )


def _find_shards(set_code: str, events: tuple[str, ...] | None) -> list[dict]:
    """Return replay shard metadata dicts for `set_code` and `events`.

    Uses `replay._shards` first (registered replay files), and falls back to scanning valid shard folders inside
    `cache/replay/` so pre-ingested shards still work if the raw CSVs were moved."""
    ms = replay._shards(set_code, list(events) if events is not None else None)
    if ms:
        return ms
    rdir = replay.replay_dir(set_code)
    if not rdir.is_dir():
        return []
    out: list[dict] = []
    for p in sorted(rdir.iterdir()):
        if not p.is_dir():
            continue
        m = replay.shard_meta(set_code, p.name)
        if not m:
            continue
        if events is not None and m.get("event") not in events:
            continue
        if (p / "games.parquet").exists() and (p / "draws.parquet").exists():
            out.append(m)
    return out


def _signature(set_code: str, fmt: str, shards: list[dict]) -> str:
    try:
        man = json.loads((config.corpus_dir(set_code) / "manifest.json").read_text(encoding="utf-8"))
        csig = str(man.get("built_at") or "")
    except (OSError, ValueError):
        csig = ""
    return json.dumps([SUMMARY_VERSION, replay.REPLAY_VERSION, fmt, sorted(m["fingerprint"] for m in shards), csig])


def cache_path(set_code: str, fmt: str) -> Path:
    return config.cache_dir(set_code.upper()) / f"mulligan_summary_{fmt}.json"


def _read_cache(path: Path, expected_sig: str | None) -> dict | None:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(d, dict) or d.get("version") != SUMMARY_VERSION:
        return None
    if expected_sig is not None and d.get("signature") != expected_sig:
        return None
    return d


def _write_cache(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _load_idmap(set_code: str, *, read_only: bool) -> dict[int, int]:
    cpath = replay.replay_dir(set_code) / "idmap.json"
    if cpath.exists():
        try:
            raw = json.loads(cpath.read_text(encoding="utf-8"))
            mp = raw.get("map", raw) if isinstance(raw, dict) else {}
            if isinstance(mp, dict) and mp:
                return {int(k): int(v) for k, v in mp.items()}
        except (OSError, ValueError, TypeError):
            pass
    if read_only:
        return {}
    try:
        idm = replay.resolve_idmap(set_code)
        return {int(k): int(v) for k, v in (idm.get("map") or {}).items()}
    except Exception:  # noqa: BLE001
        log.debug("resolve_idmap failed for %s", set_code, exc_info=True)
        return {}


def _load_cards(set_code: str) -> pl.DataFrame | None:
    for p in (config.corpus_dir(set_code) / "cards.parquet", config.model_dir(set_code) / "cards.parquet"):
        if p.exists():
            try:
                return pl.read_parquet(p)
            except Exception:  # noqa: BLE001
                continue
    return None


def mulligan_summary(
    set_code: str,
    fmt: str = "bo3",
    *,
    refresh: bool = False,
    min_hands: int = MIN_HANDS,
    top_n: int = TOP_N,
    write_dir: Path | None = None,
    read_only: bool = False,
) -> dict:
    """Return the descriptive opening-hand keep/mulligan summary for `(set_code, fmt)`.

    Never raises for missing replay data: returns `{"available": False, "reason": ...}` when the set or format has no
    replay shards (e.g. ECL, FRA)."""
    sc = set_code.upper()
    if fmt not in FORMATS:
        raise ValueError(f"format must be one of {FORMATS}")
    events = EVENTS_FOR_FMT[fmt]
    shards = _find_shards(sc, events)
    cpath = (Path(write_dir) / f"mulligan_summary_{fmt}.json") if write_dir is not None else cache_path(sc, fmt)

    # If no local replay shards exist (e.g. ready-made bundled set), serve a bundled precomputed cache if present
    if not shards:
        if not refresh and write_dir is None:
            bundled_hit = _read_cache(cpath, expected_sig=None)
            if bundled_hit is not None and bundled_hit.get("set_code") == sc and bundled_hit.get("fmt") == fmt:
                return bundled_hit
        any_shards = _find_shards(sc, None) or replay.replay_files(sc)
        fmt_label = "Best of 1" if fmt == "bo1" else "Best of 3"
        reason = (
            f"No replay data is available for {sc} in {fmt_label}."
            if any_shards
            else f"No replay data is available for {sc}."
        )
        return dict(
            version=SUMMARY_VERSION,
            available=False,
            set_code=sc,
            fmt=fmt,
            min_hands=int(min_hands),
            reason=reason,
            n_hands=0,
            n_cards_qualifying=0,
            most_kept=[],
            most_mulliganed=[],
        )

    sig = _signature(sc, fmt, shards)
    if not refresh and write_dir is None and min_hands == MIN_HANDS and top_n == TOP_N:
        hit = _read_cache(cpath, expected_sig=sig)
        if hit is not None:
            return hit

    cards = _load_cards(sc)
    idmap = _load_idmap(sc, read_only=read_only)
    if cards is None or not idmap:
        return dict(
            version=SUMMARY_VERSION,
            available=False,
            set_code=sc,
            fmt=fmt,
            min_hands=int(min_hands),
            reason=f"Replay card mapping is not available for {sc} yet.",
            n_hands=0,
            n_cards_qualifying=0,
            most_kept=[],
            most_mulliganed=[],
        )

    games_frames = [
        pl.scan_parquet(replay.shard_path(sc, m["fingerprint"]) / "games.parquet").select("gid", "num_mulligans")
        for m in shards
    ]
    draws_frames = [
        pl.scan_parquet(replay.shard_path(sc, m["fingerprint"]) / "draws.parquet")
        .filter(pl.col("kind") == 3)
        .select("gid", "kind", "id")
        for m in shards
    ]
    games_df = pl.concat(games_frames).collect()
    draws_lf = pl.concat(draws_frames)

    stats = compute_from_tables(cards, games_df, draws_lf, idmap, min_hands=min_hands, top_n=top_n)
    out = dict(
        version=SUMMARY_VERSION,
        signature=sig,
        available=stats["n_hands"] > 0,
        set_code=sc,
        fmt=fmt,
        reason="" if stats["n_hands"] > 0 else f"No opening-hand replay data recorded for {sc} ({fmt.upper()}).",
        **stats,
    )
    if not read_only and min_hands == MIN_HANDS and top_n == TOP_N:
        try:
            _write_cache(cpath, out)
        except OSError:
            log.debug("could not write mulligan summary cache for %s %s", sc, fmt, exc_info=True)
    return out
