"""Per-set Sealed in-color play rates for the Deck builder's "rarely played" note.

Counts follow the exact definition in `scripts/play_prior_check.py` (`sealed_runs`, `sealed_long`, `play_counts`):
- Data: Sealed runs (`fmt` in `bo1`, `bo3`) from the set's corpus (`pools.parquet`, `games.parquet`, `deck_cards.parquet`, `cards.parquet`).
  - Bo1: the run's deck is the `deck_key` with the most games (ties: lowest `build_index`, then `deck_key`).
  - Bo3: the run's deck is the registered game-1 deck (each match's lowest `game_number` game, taking the `deck_key` of the most
    matches; ties: earliest `match_number`, then `deck_key`).
  - `main_colors`: most common non-null `main_colors` over that deck's games (ties: alphabetical). Runs without `main_colors`
    are dropped.
- Eligible run for a nonland, nonbasic card: the card is in the run's pool (`pool_n > 0`) AND its colors are a subset of the
  deck's `main_colors` (colorless cards are eligible in every run). Splashed copies count in neither numerator nor denominator.
- Played: `deck_n > 0` in that run's deck.

Cached once per set next to the model (`sets/<SET>/model/play_rates.parquet` with `card_id`, `eligible`, `played`), keyed by a
portable corpus stamp so ready-made set bundles keep valid cached rates across machines.
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

import polars as pl

from sealed_lab import config

log = logging.getLogger(__name__)

MIN_ELIGIBLE = 50
MAX_PLAY_RATE = 0.15       # strict "< 0.15" (15% and above are not flagged)
MARKER_TEXT = "rarely played"

_PR_VERSION = 1
_PR_FILE = "play_rates.parquet"
_PR_STAMP = "play_rates.json"

_LOCK = threading.RLock()
_MEM_CACHE: dict[str, tuple[tuple | None, pl.DataFrame]] = {}


def _empty_df() -> pl.DataFrame:
    return pl.DataFrame(schema={"card_id": pl.Int64, "eligible": pl.Int64, "played": pl.Int64})


def forget(set_code: str | None = None) -> None:
    with _LOCK:
        if set_code is None:
            _MEM_CACHE.clear()
        else:
            _MEM_CACHE.pop(set_code.upper(), None)


def is_rarely_played(eligible: int, played: int) -> bool:
    """True when a nonland card has at least 50 eligible Sealed runs and is played in strictly under 15% of them."""
    return int(eligible) >= MIN_ELIGIBLE and (float(played) / float(eligible)) < MAX_PLAY_RATE


def format_tooltip(played: int, eligible: int) -> str:
    """Factual tooltip text for a rarely played card (no editorial labels)."""
    pct = 100.0 * float(played) / float(eligible) if eligible > 0 else 0.0
    return f"Played in {pct:.0f}% of the {int(eligible)} Sealed pools that opened it in its colors."


def _sealed_runs(games: pl.DataFrame) -> pl.DataFrame:
    """One row per Sealed run (`bo1` and `bo3`): `draft_id`, `deck_key`, `main_colors`."""
    if games.height == 0 or "fmt" not in games.columns:
        return pl.DataFrame(schema={"draft_id": pl.String, "deck_key": pl.Int64, "main_colors": pl.String})
    g = games.filter(pl.col("fmt").is_in(["bo1", "bo3"]))
    if g.height == 0:
        return pl.DataFrame(schema={"draft_id": pl.String, "deck_key": pl.Int64, "main_colors": pl.String})
    parts: list[pl.DataFrame] = []
    g1 = g.filter(pl.col("fmt") == "bo1")
    if g1.height:
        b = g1.group_by(["draft_id", "deck_key"]).agg(
            pl.len().alias("n"),
            pl.col("build_index").min().alias("o") if "build_index" in g1.columns else pl.lit(0).alias("o"),
        )
        parts.append(
            b.sort(["draft_id", "n", "o", "deck_key"], descending=[False, True, False, False])
            .group_by("draft_id", maintain_order=True)
            .first()
            .select("draft_id", "deck_key")
        )
    g3 = g.filter(pl.col("fmt") == "bo3")
    if g3.height:
        if "match_number" in g3.columns and "game_number" in g3.columns:
            m = (
                g3.sort(["draft_id", "match_number", "game_number"], nulls_last=True)
                .group_by(["draft_id", "match_number"], maintain_order=True)
                .agg(pl.col("deck_key").first())
            )
            b = m.group_by(["draft_id", "deck_key"]).agg(pl.len().alias("n"), pl.col("match_number").min().alias("o"))
        else:
            b = g3.group_by(["draft_id", "deck_key"]).agg(pl.len().alias("n"), pl.lit(0).alias("o"))
        parts.append(
            b.sort(["draft_id", "n", "o", "deck_key"], descending=[False, True, False, False])
            .group_by("draft_id", maintain_order=True)
            .first()
            .select("draft_id", "deck_key")
        )
    if not parts:
        return pl.DataFrame(schema={"draft_id": pl.String, "deck_key": pl.Int64, "main_colors": pl.String})
    pick = pl.concat(parts)
    mc = (
        g.join(pick, on=["draft_id", "deck_key"], how="inner")
        .filter(pl.col("main_colors").is_not_null())
        .group_by(["draft_id", "main_colors"])
        .len()
        .sort(["draft_id", "len", "main_colors"], descending=[False, True, False])
        .group_by("draft_id", maintain_order=True)
        .first()
        .select("draft_id", "main_colors")
    )
    return pick.join(mc, on="draft_id", how="inner").select(
        pl.col("draft_id").cast(pl.String),
        pl.col("deck_key").cast(pl.Int64),
        pl.col("main_colors").cast(pl.String),
    )


def compute_from_tables(cards: pl.DataFrame, games: pl.DataFrame, pools: pl.DataFrame | pl.LazyFrame,
                        deck_cards: pl.DataFrame) -> pl.DataFrame:
    """Compute `(card_id, eligible, played)` for every nonland, nonbasic card from raw corpus tables."""
    runs = _sealed_runs(games)
    if runs.height == 0 or cards.height == 0:
        return _empty_df()
    nonland = cards.filter(~pl.col("is_land").fill_null(False) & ~pl.col("is_basic").fill_null(False)).select(
        pl.col("card_id").cast(pl.Int64),
        pl.col("colors").fill_null("").cast(pl.String),
    )
    if nonland.height == 0:
        return _empty_df()
    p_lf = pools.lazy() if isinstance(pools, pl.DataFrame) else pools
    p = (
        p_lf.filter(pl.col("n") > 0)
        .select(pl.col("draft_id").cast(pl.String), pl.col("card_id").cast(pl.Int64), pl.col("n").cast(pl.Int64).alias("pool_n"))
        .join(runs.lazy(), on="draft_id", how="inner")
        .collect()
    )
    if p.height == 0:
        return _empty_df()
    dc = deck_cards.select(
        pl.col("deck_key").cast(pl.Int64),
        pl.col("card_id").cast(pl.Int64),
        pl.col("n").cast(pl.Int64).alias("deck_n"),
    )
    d = (
        p.join(dc, on=["deck_key", "card_id"], how="left")
        .with_columns(pl.col("deck_n").fill_null(0))
        .join(nonland, on="card_id", how="inner")
        .group_by(["draft_id", "card_id"])
        .agg(
            pl.col("pool_n").sum(),
            pl.col("deck_n").sum(),
            pl.col("main_colors").first(),
            pl.col("colors").first(),
        )
    )
    if d.height == 0:
        return _empty_df()
    pairs = d.select("colors", "main_colors").unique()
    ok = [set(c or "") <= set(m or "") for c, m in zip(pairs["colors"].to_list(), pairs["main_colors"].to_list())]
    pairs = pairs.with_columns(pl.Series("_elig", ok, dtype=pl.Boolean))
    d = d.join(pairs, on=["colors", "main_colors"], how="left").filter(pl.col("_elig"))
    if d.height == 0:
        return _empty_df()
    return (
        d.group_by("card_id")
        .agg(
            pl.len().cast(pl.Int64).alias("eligible"),
            (pl.col("deck_n") > 0).sum().cast(pl.Int64).alias("played"),
        )
        .sort("card_id")
    )


def _corpus_stamp(set_code: str) -> tuple | None:
    """Portable stamp of the corpus tables `play_rates.parquet` depends on (independent of file mtimes)."""
    cdir = config.corpus_dir(set_code)
    req = [cdir / name for name in ("cards.parquet", "games.parquet", "pools.parquet", "deck_cards.parquet")]
    if not all(p.exists() for p in req):
        return None
    man_path = cdir / "manifest.json"
    c_stamp = ""
    if man_path.exists():
        try:
            man = json.loads(man_path.read_text(encoding="utf-8"))
            built = str(man.get("built_at") or "")
            tabs = man.get("tables") or {}
            if built:
                c_stamp = f"{built}:{tabs.get('pools', '')}:{tabs.get('games', '')}:{tabs.get('deck_cards', '')}"
        except (OSError, ValueError):
            c_stamp = ""
    if not c_stamp:
        try:
            c_stamp = "sizes:" + ",".join(str(p.stat().st_size) for p in req)
        except OSError:
            return None
    return (_PR_VERSION, c_stamp)


def _write_cache(dest: Path, stamp: tuple | None, df: pl.DataFrame) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    tmp = dest / (_PR_FILE + ".tmp")
    df.write_parquet(tmp, compression="zstd")
    tmp.replace(dest / _PR_FILE)
    if stamp is not None:
        tmp_j = dest / (_PR_STAMP + ".tmp")
        tmp_j.write_text(json.dumps({"stamp": list(stamp)}), encoding="utf-8")
        tmp_j.replace(dest / _PR_STAMP)


def _read_disk_cache(mdir: Path, stamp: tuple | None) -> pl.DataFrame | None:
    pq = mdir / _PR_FILE
    if not pq.exists():
        return None
    sp = mdir / _PR_STAMP
    if stamp is not None and sp.exists():
        try:
            saved = json.loads(sp.read_text(encoding="utf-8"))
            if tuple(saved.get("stamp") or ()) != stamp:
                return None
        except (OSError, ValueError):
            return None
    elif stamp is not None and not sp.exists():
        # A cache file without a stamp sidecar is accepted only if there is no corpus to verify against
        pass
    try:
        df = pl.read_parquet(pq)
        if not {"card_id", "eligible", "played"} <= set(df.columns):
            return None
        return df.select(
            pl.col("card_id").cast(pl.Int64),
            pl.col("eligible").cast(pl.Int64),
            pl.col("played").cast(pl.Int64),
        )
    except Exception:  # noqa: BLE001 - recompute on unreadable cache
        return None


def _compute_for_set(set_code: str) -> pl.DataFrame:
    cdir = config.corpus_dir(set_code)
    req = [cdir / name for name in ("cards.parquet", "games.parquet", "pools.parquet", "deck_cards.parquet")]
    if not all(p.exists() for p in req):
        return _empty_df()
    dc_schema = pl.read_parquet_schema(cdir / "deck_cards.parquet")
    pool_schema = pl.read_parquet_schema(cdir / "pools.parquet")
    games_schema = pl.read_parquet_schema(cdir / "games.parquet")
    cards_schema = pl.read_parquet_schema(cdir / "cards.parquet")
    if (not {"deck_key", "card_id", "n"} <= set(dc_schema)
            or not {"draft_id", "card_id", "n"} <= set(pool_schema)
            or not {"draft_id", "fmt", "deck_key", "main_colors"} <= set(games_schema)
            or not {"card_id", "colors", "is_land"} <= set(cards_schema)):
        return _empty_df()
    cards = pl.read_parquet(cdir / "cards.parquet")
    games_cols = [c for c in ("draft_id", "fmt", "deck_key", "build_index", "match_number", "game_number", "main_colors")
                  if c in games_schema]
    games = pl.read_parquet(cdir / "games.parquet", columns=games_cols)
    pools = pl.scan_parquet(cdir / "pools.parquet")
    deck_cards = pl.read_parquet(cdir / "deck_cards.parquet", columns=["deck_key", "card_id", "n"])
    df = compute_from_tables(cards, games, pools, deck_cards)
    # Align card_ids with the model's cards.parquet if one exists and differs from the corpus card_id mapping
    mp = config.model_dir(set_code) / "cards.parquet"
    if df.height > 0 and mp.exists():
        try:
            mcards = pl.read_parquet(mp, columns=["card_id", "name"])
            ccards = cards.select(pl.col("card_id").cast(pl.Int64).alias("_cid"), "name")
            m_map = mcards.select(pl.col("card_id").cast(pl.Int64).alias("m_cid"), "name")
            if m_map.height != ccards.height or m_map.sort("m_cid")["name"].to_list() != ccards.sort("_cid")["name"].to_list():
                df = (
                    df.join(ccards, left_on="card_id", right_on="_cid", how="inner")
                    .join(m_map, on="name", how="inner")
                    .select(pl.col("m_cid").alias("card_id"), "eligible", "played")
                    .sort("card_id")
                )
        except Exception:  # noqa: BLE001 - keep corpus card_ids if model table is minimal
            pass
    return df


def play_rates_df(set_code: str, *, refresh: bool = False, write_dir: Path | None = None) -> pl.DataFrame:
    """Return `(card_id, eligible, played)` for `set_code`, computing and caching next to the model on first use.

    When `write_dir` is given, computes from the corpus and writes directly into `write_dir` without touching `APP_HOME`
    (used by `tools/bundle_sets.py`)."""
    sc = set_code.upper()
    stamp = _corpus_stamp(sc)
    if write_dir is not None:
        df = _compute_for_set(sc)
        _write_cache(Path(write_dir), stamp, df)
        return df

    mdir = config.model_dir(sc)
    with _LOCK:
        if not refresh:
            hit = _MEM_CACHE.get(sc)
            if hit is not None and hit[0] == stamp:
                return hit[1]
            disk = _read_disk_cache(mdir, stamp)
            if disk is not None:
                _MEM_CACHE[sc] = (stamp, disk)
                return disk
        df = _compute_for_set(sc)
        if mdir.exists() and stamp is not None:
            try:
                _write_cache(mdir, stamp, df)
            except OSError:
                log.debug("could not write play_rates cache for %s", sc, exc_info=True)
        _MEM_CACHE[sc] = (stamp, df)
        return df


def rarely_played_notes(set_code: str, *, refresh: bool = False) -> dict[int, dict]:
    """`{card_id: {eligible, played, rate, marker, tip}}` for nonland cards with >= 50 eligible Sealed runs and < 15% play rate."""
    df = play_rates_df(set_code, refresh=refresh)
    if df.height == 0:
        return {}
    out: dict[int, dict] = {}
    for cid, elig, played in df.iter_rows():
        e, k = int(elig), int(played)
        if is_rarely_played(e, k):
            out[int(cid)] = dict(
                eligible=e,
                played=k,
                rate=float(k) / float(e),
                marker=MARKER_TEXT,
                tip=format_tooltip(k, e),
            )
    return out
