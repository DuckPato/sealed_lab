"""Full corpus explorer: query real 17Lands runs, analytics, similar Sealed pools, mid-run rebuilds, Bo3 sideboarding,
and opening-hand / replay card profiles.

Read-only over `sets/<SET>/corpus/*.parquet` (games, builds, decks, deck_cards, cards, pools, pool_audit, hands)
and optionally existing cached replay files in `sets/<SET>/cache/replay/`.

One deck search result is one *run* (a 17Lands `draft_id`: one draft or Sealed event of one player), not one game: the record of
a deck is the record of its run.  A run "contains" a card when any of its builds (17Lands `build_index`: the deck changes between
games after a rebuild or Bo3 sideboarding) has it in the main deck; the decklist shown is the matching build that played the most
games, and the result also says in how many of the run's games the card was in the deck.

What the corpus stores:
* main decks of every event (Sealed and draft), from 17Lands' `deck_*` columns;
* the Sealed pool (deck + sideboard, non-basic cards) per run, so a Sealed deck's sideboard is pool - main deck;
* NOT the sideboard of draft runs (ingest keeps 17Lands' `sideboard_*` columns only for Sealed), so draft results have no sideboard.
"""
from __future__ import annotations

import json
import logging
import math
import threading
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl

from sealed_lab.config import cache_dir, corpus_dir, event_format, model_dir

log = logging.getLogger(__name__)

# Labels for the 17lands event types; unknown events get their raw name plus (Bo1)/(Bo3).
EVENT_LABELS = {
    "PremierDraft": "Premier Draft (Bo1)",
    "TradDraft": "Traditional Draft (Bo3)",
    "QuickDraft": "Quick Draft (Bo1)",
    "PickTwoDraft": "Pick-Two Draft (Bo1)",
    "PickTwoTradDraft": "Pick-Two Traditional Draft (Bo3)",
    "Sealed": "Sealed (Bo1)",
    "TradSealed": "Traditional Sealed (Bo3)",
}
EVENT_ORDER = ["PremierDraft", "TradDraft", "QuickDraft", "PickTwoDraft", "PickTwoTradDraft", "Sealed", "TradSealed"]
SORTS = ("result", "win_rate", "newest", "oldest", "model")
COLOR_MATCH = ("exact", "include")   # ticked colors: exactly the main colors, or all of them among the main colors
COLOR_PAIRS = ("WU", "UB", "BR", "RG", "WG", "WB", "UR", "BG", "WR", "UG")
RANK_TIERS = {"bronze": 1, "silver": 2, "gold": 3, "platinum": 4, "diamond": 5, "mythic": 6}
DEFAULT_MIN_GAMES = 3
DEFAULT_MIN_REBUILD_CHANGES = 5
DEFAULT_LIMIT = 200
DEFAULT_CARD_SHARE = 0.5
CAP_SHARE = 0.005          # an event's "max wins" is the highest win count that at least 0.5% of its runs reach

_TABLES = ("games", "builds", "decks", "deck_cards", "cards", "pools", "pool_audit", "hands")
_LOCK = threading.RLock()
_CACHE: dict[str, tuple[tuple, "_SetIndex"]] = {}
_CACHE_KEEP = 3
_MS_CACHE: dict[str, tuple[tuple, pl.DataFrame]] = {}
_MS_FILE, _MS_STAMP = "deck_scores.parquet", "deck_scores.json"
_MS_VERSION = 2


def _wilson_ci(wins: int, games: int, z: float = 1.96) -> tuple[float, float] | None:
    """95% Wilson score interval for a binomial proportion (wins / games)."""
    if games <= 0:
        return None
    p = wins / games
    z2 = z * z
    denom = 1.0 + z2 / games
    center = (p + z2 / (2.0 * games)) / denom
    half = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * games)) / games) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def is_bo3_event(event: str) -> bool:
    """Bo3 events count matches (17lands Traditional events)."""
    return "trad" in (event or "").lower()


def event_label(event: str) -> str:
    if event in EVENT_LABELS:
        return EVENT_LABELS[event]
    return f"{event} ({'Bo3' if is_bo3_event(event) else 'Bo1'})"


def _event_rank(event: str) -> tuple[int, str]:
    return (EVENT_ORDER.index(event) if event in EVENT_ORDER else len(EVENT_ORDER), event)


def _canon_colors(s: str | None) -> str:
    if not s:
        return ""
    up = str(s).upper()
    return "".join(c for c in "WUBRG" if c in up)


def _rank_tier(rank: str | None) -> int:
    if not rank:
        return 0
    head = str(rank).split("-", 1)[0].strip().lower()
    return RANK_TIERS.get(head, 0)


@dataclass
class _SetIndex:
    set_code: str
    cdir: Path
    cards: pl.DataFrame          # card_id, name, front_name, is_basic, is_land, is_creature, mana_value, colors, types, rarity
    runs: pl.DataFrame           # one row per (event, draft_id): see _build_runs
    builds: pl.DataFrame         # event, draft_id, build_index, deck_key, games, wins, main_colors, splash_colors, n_lands, n_cards
    events: list[dict]
    by_name: dict[str, int]      # lower-cased name / front name -> card_id
    names: list[str]             # non-basic card names, sorted (autocomplete)
    has_pools: bool
    has_hands: bool
    card_meta: dict[int, dict] = field(default_factory=dict)
    _bo3_cache: dict | None = None
    _card_weights: dict[int, float] | None = None
    _replay_hist_cache: dict[int, dict | None] = field(default_factory=dict)
    _replay_mull_cache: dict | None = None
    _cast_state: dict | None = None      # replay cast lens: stamp, grpIds per card, replay game -> run link (see _cast_prepare)
    _bo3_reg: pl.DataFrame | None = None
    provisional: bool = False    # a pick-file set: the tables are the draft decks rebuilt from 17Lands pick data (provisional.py)
    dc_path: Path | None = None        # deck_cards.parquet (deck_key, card_id, n)
    pools_path: Path | None = None     # pools.parquet (draft_id, card_id, n): the Sealed pool, or a pick-file set's drafted pool

    def __post_init__(self) -> None:
        if self.dc_path is None:
            self.dc_path = self.cdir / "deck_cards.parquet"
        if self.pools_path is None:
            self.pools_path = self.cdir / "pools.parquet"


def _stamp(cdir: Path) -> tuple:
    out = []
    for t in _TABLES:
        p = cdir / f"{t}.parquet"
        try:
            st = p.stat()
            out.append((t, st.st_size, st.st_mtime_ns))
        except OSError:
            out.append((t, None, None))
    return tuple(out)


def _build_runs(games: pl.DataFrame, builds: pl.DataFrame, decks: pl.DataFrame | None = None) -> tuple[pl.DataFrame, pl.DataFrame, list[dict]]:
    """Per run: games, wins, losses, matches (Bo3, separating incomplete 1-1 ties), date, rank, skill, colors;
    per build: colors from that build's own games + n_lands/n_cards; and the per-event summary."""
    g = games.with_columns(pl.col("won").fill_null(False))
    run_g = (g.group_by(["event", "draft_id"])
             .agg(pl.len().cast(pl.Int32).alias("games"), pl.col("won").sum().cast(pl.Int32).alias("wins"),
                  pl.col("game_time").min().alias("first_game"), pl.col("game_time").max().alias("last_game"),
                  pl.col("draft_time").min().alias("draft_time"),
                  pl.col("rank").drop_nulls().first().alias("rank"),
                  pl.col("skill").drop_nulls().first().alias("skill"),
                  pl.col("main_colors").drop_nulls().first().alias("run_main_colors"),
                  pl.col("splash_colors").drop_nulls().first().alias("run_splash_colors"),
                  pl.col("fmt").first().alias("fmt")))
    run_g = run_g.with_columns((pl.col("games") - pl.col("wins")).alias("losses"))
    # matches: a Bo3 match is won with more game wins than losses, lost with fewer, and a 1-1 tie is flagged incomplete (not counted as a loss)
    m = (g.group_by(["event", "draft_id", "match_number"])
         .agg(pl.col("won").sum().alias("gw"), (~pl.col("won")).sum().alias("gl")))
    run_m = m.group_by(["event", "draft_id"]).agg(
        (pl.col("gw") > pl.col("gl")).sum().cast(pl.Int32).alias("match_wins"),
        (pl.col("gw") < pl.col("gl")).sum().cast(pl.Int32).alias("match_losses"),
        (pl.col("gw") == pl.col("gl")).sum().cast(pl.Int32).alias("incomplete_matches"))
    nb = builds.group_by(["event", "draft_id"]).agg(pl.len().cast(pl.Int16).alias("n_builds"))
    # Per-build colors from that build's own games (so mid-run color pivots display the shown build's colors, not a run-wide mode)
    b_cols = (g.group_by(["event", "draft_id", "build_index"])
              .agg(pl.col("main_colors").drop_nulls().first().alias("main_colors"),
                   pl.col("splash_colors").drop_nulls().first().alias("splash_colors")))
    builds_enriched = builds.join(b_cols, on=["event", "draft_id", "build_index"], how="left")
    if decks is not None and decks.height:
        builds_enriched = builds_enriched.join(decks.select("deck_key", "n_lands", "n_cards"), on="deck_key", how="left")
    else:
        builds_enriched = builds_enriched.with_columns(
            pl.lit(None, dtype=pl.Int16).alias("n_lands"),
            pl.lit(None, dtype=pl.Int16).alias("n_cards"),
        )
    runs = run_g.join(run_m, on=["event", "draft_id"], how="left").join(nb, on=["event", "draft_id"], how="left")
    return _finish_runs(runs, builds_enriched)


def _finish_runs(runs: pl.DataFrame, builds_enriched: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, list[dict]]:
    """The derived run columns (bo3, results, date, rank tier, event caps, score, trophy) and the per-event summary."""
    runs = runs.with_columns(pl.col("event").fill_null("").str.to_lowercase().str.contains("trad", literal=True).alias("bo3"))  # = is_bo3_event
    # Precompute rank_tier for fast rank filtering
    r_head = pl.col("rank").fill_null("").str.split("-").list.first().str.strip_chars().str.to_lowercase()
    runs = runs.with_columns(
        pl.when(pl.col("bo3")).then(pl.col("match_wins")).otherwise(pl.col("wins")).alias("res_wins"),
        pl.when(pl.col("bo3")).then(pl.col("match_losses")).otherwise(pl.col("losses")).alias("res_losses"),
        pl.coalesce(pl.col("first_game"), pl.col("draft_time")).alias("date"),
        (pl.col("wins") / pl.col("games")).alias("win_rate"),
        pl.when(r_head == "mythic").then(6)
        .when(r_head == "diamond").then(5)
        .when(r_head == "platinum").then(4)
        .when(r_head == "gold").then(3)
        .when(r_head == "silver").then(2)
        .when(r_head == "bronze").then(1)
        .otherwise(0).cast(pl.Int8).alias("rank_tier"),
    )
    events = []
    caps = {}
    for ev in sorted(runs["event"].unique().to_list(), key=_event_rank):
        r = runs.filter(pl.col("event") == ev)
        vc = r["res_wins"].value_counts()
        need = max(1, int(CAP_SHARE * r.height))
        ok = [int(w) for w, n in vc.iter_rows() if n >= need]
        cap = max(ok) if ok else int(r["res_wins"].max() or 0)
        caps[ev] = max(cap, 1)
        events.append(dict(event=ev, label=event_label(ev), fmt=event_format(ev) or "", bo3=is_bo3_event(ev), max_wins=caps[ev],
                           n_runs=int(r.height), n_games=int(r["games"].sum())))
    cap_df = pl.DataFrame({"event": list(caps), "max_wins": list(caps.values())}, schema={"event": pl.String, "max_wins": pl.Int32})
    runs = runs.join(cap_df, on="event", how="left").with_columns(
        (pl.col("res_wins") / pl.col("max_wins")).clip(upper_bound=1.0).alias("score"),
        (pl.col("res_wins") >= pl.col("max_wins")).alias("trophy"))
    return runs, builds_enriched, events


def _is_pick_set(sc: str) -> bool:
    from sealed_lab import provisional

    return provisional.is_provisional(sc)


def _load_pick(sc: str) -> _SetIndex:
    """The index of a pick-file set (provisional model): one run per rebuilt draft, one deck per run, the drafted pool as its "pool"."""
    from sealed_lab import provisional

    with _LOCK:
        if not provisional.ensure_pick_decks(sc):
            raise FileNotFoundError(provisional.DECKS_NO_TABLES)
        mdir = model_dir(sc)
        stamp = ("pick", provisional.pick_decks_stamp(sc))
        hit = _CACHE.get(sc)
        if hit is not None and hit[0] == stamp:
            return hit[1]
        cards = pl.read_parquet(mdir / "cards.parquet", columns=["card_id", "name", "front_name", "is_basic", "is_land", "is_creature",
                                                                 "mana_value", "colors", "types", "rarity"])
        pr = pl.read_parquet(mdir / provisional.PICK_DECKS)
        event = provisional.EVENT
        g = (pr.with_columns((pl.col("wins") + pl.col("losses")).cast(pl.Int32).alias("games"))
             .select(pl.lit(event).alias("event"), "draft_id", "games", "wins", "losses",
                     pl.col("draft_time").alias("first_game"), pl.col("draft_time").alias("last_game"), "draft_time", "rank", "skill",
                     pl.col("main_colors").alias("run_main_colors"), pl.col("splash_colors").alias("run_splash_colors"),
                     pl.lit("draft").alias("fmt"),
                     pl.lit(0, dtype=pl.Int32).alias("match_wins"), pl.lit(0, dtype=pl.Int32).alias("match_losses"),
                     pl.lit(0, dtype=pl.Int32).alias("incomplete_matches"), pl.lit(1, dtype=pl.Int16).alias("n_builds")))
        builds = pr.select(pl.lit(event).alias("event"), "draft_id", pl.lit(0, dtype=pl.Int16).alias("build_index"), "deck_key",
                           (pl.col("wins") + pl.col("losses")).cast(pl.Int32).alias("games"), "wins", "main_colors", "splash_colors",
                           "n_lands", "n_cards")
        runs, builds, events = _finish_runs(g, builds)
        idx = _index_of(sc, mdir, cards, runs, builds, events, True, False, provisional=True,
                        dc_path=mdir / provisional.PICK_DECK_CARDS, pools_path=mdir / provisional.PICK_POOLS)
        _CACHE[sc] = (stamp, idx)
        while len(_CACHE) > _CACHE_KEEP:
            _CACHE.pop(next(iter(_CACHE)))
        return idx


def _index_of(sc: str, cdir: Path, cards: pl.DataFrame, runs: pl.DataFrame, builds: pl.DataFrame, events: list[dict],
              has_pools: bool, has_hands: bool, **kw) -> _SetIndex:
    by_name: dict[str, int] = {}
    names = []
    card_meta: dict[int, dict] = {}
    for row in cards.iter_rows(named=True):
        cid = int(row["card_id"])
        nm = row["name"]
        card_meta[cid] = dict(
            card_id=cid, name=nm or "", front_name=row["front_name"] or "",
            is_basic=bool(row["is_basic"]), is_land=bool(row["is_land"]), is_creature=bool(row["is_creature"]),
            mana_value=float(row["mana_value"] or 0.0), colors=_canon_colors(row["colors"]),
            types=row["types"] or "", rarity=(row["rarity"] or "").lower(),
        )
        if not nm:
            continue
        for k in (nm, row["front_name"], nm.split(" // ")[0].strip()):
            if k:
                by_name.setdefault(k.strip().lower(), cid)
        if not row["is_basic"]:
            names.append(nm)
    return _SetIndex(sc, cdir, cards, runs, builds, events, by_name, sorted(set(names), key=str.lower),
                     has_pools, has_hands, card_meta, **kw)


def _need_games(idx: _SetIndex) -> None:
    """What reads game-by-game data (rebuilds, sideboarding, similar Sealed pools) has none for a pick-file set."""
    if idx.provisional:
        from sealed_lab import provisional

        raise ValueError(provisional.UNAVAILABLE)


def _load(set_code: str) -> _SetIndex:
    sc = set_code.upper()
    if _is_pick_set(sc):
        return _load_pick(sc)
    cdir = corpus_dir(sc)
    stamp = _stamp(cdir)
    with _LOCK:
        hit = _CACHE.get(sc)
        if hit is not None and hit[0] == stamp:
            return hit[1]
        for t in ("games", "builds", "deck_cards", "cards"):
            if not (cdir / f"{t}.parquet").exists():
                raise FileNotFoundError(f"{sc} has no corpus yet: calculate it on the Sets tab")
        cards = pl.read_parquet(cdir / "cards.parquet", columns=["card_id", "name", "front_name", "is_basic", "is_land", "is_creature",
                                                                 "mana_value", "colors", "types", "rarity"])
        games = pl.read_parquet(cdir / "games.parquet", columns=["event", "fmt", "draft_id", "build_index", "match_number", "game_time",
                                                                 "draft_time", "rank", "skill", "main_colors", "splash_colors", "won"])
        builds = pl.read_parquet(cdir / "builds.parquet", columns=["event", "draft_id", "build_index", "deck_key", "games", "wins"])
        decks = None
        dp = cdir / "decks.parquet"
        if dp.exists():
            try:
                decks = pl.read_parquet(dp, columns=["deck_key", "n_lands", "n_cards"])
            except Exception:  # noqa: BLE001
                decks = None
        runs, builds, events = _build_runs(games, builds, decks)
        has_pools = (cdir / "pools.parquet").exists()
        has_hands = (cdir / "hands.parquet").exists()
        idx = _index_of(sc, cdir, cards, runs, builds, events, has_pools, has_hands)
        _CACHE[sc] = (stamp, idx)
        while len(_CACHE) > _CACHE_KEEP:
            _CACHE.pop(next(iter(_CACHE)))
        return idx


def clear_cache() -> None:
    with _LOCK:
        _CACHE.clear()
        _MS_CACHE.clear()


# ---------------------------------------------------------------------------------------------
# public lookups
# ---------------------------------------------------------------------------------------------

def card_names(set_code: str) -> list[str]:
    """The set's non-basic card names (corpus card table), sorted: the autocomplete list."""
    return list(_load(set_code).names)


def events(set_code: str) -> list[dict]:
    """[{event, label, fmt, bo3, max_wins, n_runs, n_games}] for the events in the set's corpus (Premier Draft first)."""
    return [dict(e) for e in _load(set_code).events]


def resolve_card(set_code: str, name: str) -> int | None:
    idx = _load(set_code)
    return _resolve_in(idx, name)


def _resolve_in(idx: _SetIndex, card: str | int | None) -> int | None:
    if card is None:
        return None
    if isinstance(card, int) and not isinstance(card, bool):
        return card if card in idx.card_meta else None
    key = str(card).strip().lower()
    if not key:
        return None
    if key in idx.by_name:
        return idx.by_name[key]
    hits = [n for n in idx.names if n.lower().startswith(key)]
    if len(hits) == 1:
        return idx.by_name[hits[0].lower()]
    return None


def _require_card(idx: _SetIndex, card: str | int) -> int:
    cid = _resolve_in(idx, card)
    if cid is None:
        raise ValueError(f"no card named {card!r} in {idx.set_code}")
    return cid


def _sort_exprs(sort: str):
    """(columns, descending) for a sort; `draft_id` last so that ties keep one order."""
    if sort == "newest":
        return ["date", "score", "res_losses", "draft_id"], [True, True, False, False]
    if sort == "oldest":
        return ["date", "score", "res_losses", "draft_id"], [False, True, False, False]
    if sort == "win_rate":
        return ["win_rate", "games", "date", "draft_id"], [True, True, True, False]
    if sort == "model":
        return ["model_score", "model_q", "score", "date", "draft_id"], [True, True, True, True, False]
    # best result: wins relative to the event's max (7-x Premier = 3-x Traditional Draft), fewer losses, game win rate, newest
    return ["score", "res_losses", "win_rate", "games", "date", "draft_id"], [True, False, True, True, True, False]


# ---------------------------------------------------------------------------------------------
# Phase 3A: multi-constraint deck search & instant analytics
# ---------------------------------------------------------------------------------------------

def _apply_build_filters(b: pl.DataFrame, colors: str | None, splash: str | None,
                         min_lands: int | None, max_lands: int | None, color_match: str = "exact") -> pl.DataFrame:
    if colors:
        c = colors.strip().lower()
        mc = pl.col("main_colors").fill_null("")
        if c == "mono":
            b = b.filter(mc.str.len_chars() == 1)
        elif c == "2c":
            b = b.filter(mc.str.len_chars() == 2)
        elif c in ("3c+", "3+"):
            b = b.filter(mc.str.len_chars() >= 3)
        else:
            want = _canon_colors(colors)
            if want:
                # canonicalize via set of colors (17Lands stores WUBRG order, fall back to char length + contains)
                cond = (mc.str.len_chars() == len(want)) if color_match != "include" else pl.lit(True)
                for ch in want:
                    cond = cond & mc.str.contains(ch, literal=True)
                b = b.filter(cond)
    if splash:
        sp = splash.strip().lower()
        scol = pl.col("splash_colors").fill_null("")
        if sp == "none":
            b = b.filter(scol == "")
        elif sp == "any":
            b = b.filter(scol != "")
        elif sp.upper() in "WUBRG" and len(sp) == 1:
            b = b.filter(scol.str.contains(sp.upper(), literal=True))
    if min_lands is not None and min_lands > 0:
        b = b.filter(pl.col("n_lands").is_not_null() & (pl.col("n_lands") >= int(min_lands)))
    if max_lands is not None and max_lands > 0:
        b = b.filter(pl.col("n_lands").is_not_null() & (pl.col("n_lands") <= int(max_lands)))
    return b


def _date_cutoff(idx: _SetIndex, events: list[str] | None, days: int | None):
    """Return (cutoff_datetime, iso_date_str) relative to the latest game date of the set (or chosen events)."""
    if days is None or int(days) <= 0:
        return None, None
    base = idx.runs.filter(pl.col("event").is_in(list(events))) if events else idx.runs
    mx = base["date"].max() if base.height else None
    if mx is None:
        mx = idx.runs["date"].max() if idx.runs.height else None
    if mx is None:
        return None, None
    cut = mx - timedelta(days=int(days))
    return cut, cut.date().isoformat()


def search(set_code: str, card: str | int | None = None, events: list[str] | None = None, sort: str = "result",
           min_games: int = DEFAULT_MIN_GAMES, limit: int = DEFAULT_LIMIT, offset: int = 0,
           min_card_share: float = DEFAULT_CARD_SHARE, *,
           include_cards: list[tuple[str | int, int] | str | int] | None = None,
           exclude_cards: list[str | int] | None = None,
           in_pool_only: list[str | int] | None = None,
           colors: str | None = None,
           color_match: str = "exact",
           splash: str | None = None,
           min_skill: float | None = None,
           min_rank: str | None = None,
           ranks: list[str] | None = None,
           trophy_only: bool = False,
           min_lands: int | None = None,
           max_lands: int | None = None,
           days: int | None = None,
           with_analytics: bool = True) -> dict:
    """Real runs of the corpus matching card and deck constraints, sorted, with corpus-wide analytics.

    `card`: primary card name or corpus card_id (optional — None/"" browses all decks matching the filters).
    `include_cards`: extra required cards [(name_or_id, min_copies), ...] (or plain names/ids for >=1 copy).
    `exclude_cards`: cards that must NOT appear in the main deck (0 copies).
    `in_pool_only`: Sealed only — cards that must be in the run's pool (`pools.parquet`) and NOT in the main deck.
    `colors`: "mono", "2c", "3c+", or a color string like "WU".
    `color_match`: for a color string, "exact" (main colors are exactly those) or "include" (main colors contain all of them).
    `splash`: "none", "any", or single color letter "W"/"U"/"B"/"R"/"G".
    `min_skill`: minimum pilot `skill` (17Lands user_game_win_rate_bucket, e.g. 0.55 or 0.60).
    `min_rank`: minimum Arena rank tier ("Platinum", "Diamond", "Mythic", ...).
    `ranks`: keep only these Arena rank tiers (["Diamond", "Mythic"], ...); runs without a rank are left out when it is given.
    `sort="model"`: by the set model's score of the deck (the deck scorer's percentile among real builds), Sealed runs only; the
    first such search of a set scores every Sealed build once (`model_scores`), later ones read the cached scores.
    `trophy_only`: keep only trophy runs (reached the event's max wins).
    `min_lands` / `max_lands`: filter by build land count (`decks.parquet`).
    `days`: keep only runs within the last `days` days relative to the latest recorded game date in the set/events.
    """
    if sort not in SORTS:
        raise ValueError(f"sort must be one of {SORTS}")
    if color_match not in COLOR_MATCH:
        raise ValueError(f"color_match must be one of {COLOR_MATCH}")
    with _LOCK:              # one search at a time (the GUI's worker threads share the cache; polars frames are built here)
        return _search(
            set_code, card, events, sort, min_games, limit, offset, min_card_share,
            include_cards=include_cards, exclude_cards=exclude_cards, in_pool_only=in_pool_only,
            colors=colors, color_match=color_match, splash=splash, min_skill=min_skill, min_rank=min_rank, ranks=ranks,
            trophy_only=trophy_only, min_lands=min_lands, max_lands=max_lands, days=days,
            with_analytics=with_analytics,
        )


# ---------------------------------------------------------------------------------------------
# Model score of real decks (the deck scorer's number, for the "Model score" sort)
# ---------------------------------------------------------------------------------------------

def _model_stamp(set_code: str) -> tuple | None:
    """Portable stamp of what cached deck scores depend on: the model's own `built_at` and the corpus's `built_at` (with
    `deck_cards.parquet` byte size as fallback if a minimal corpus has no manifest).  Independent of file mtimes so ready-made
    set bundles keep valid cached scores after unpacking on another machine.  A pick-file set (provisional model) is stamped by its
    model and its pick-deck tables instead of a corpus."""
    if _is_pick_set(set_code.upper()):
        from sealed_lab import provisional

        ps = provisional.pick_decks_stamp(set_code)
        return None if ps is None else (_MS_VERSION, "pick", ps)
    try:
        meta = json.loads((model_dir(set_code) / "meta.json").read_text(encoding="utf-8"))
        m_stamp = str(meta.get("built_at") or "")
        if not m_stamp:
            return None
        cdir = corpus_dir(set_code)
        dc_path = cdir / "deck_cards.parquet"
        if not dc_path.exists():
            return None
        c_stamp = ""
        man_path = cdir / "manifest.json"
        if man_path.exists():
            try:
                man = json.loads(man_path.read_text(encoding="utf-8"))
                c_stamp = str(man.get("built_at") or "")
                dc_rows = (man.get("tables") or {}).get("deck_cards")
                if c_stamp and dc_rows is not None:
                    c_stamp = f"{c_stamp}:{int(dc_rows)}"
            except (OSError, ValueError):
                c_stamp = ""
        if not c_stamp:
            c_stamp = f"size:{dc_path.stat().st_size}"
        return (_MS_VERSION, m_stamp, c_stamp)
    except (OSError, ValueError):
        return None


def _write_model_scores(dest: Path, stamp: tuple, df: pl.DataFrame) -> None:
    tmp = dest / (_MS_FILE + ".tmp")
    df.write_parquet(tmp)
    tmp.replace(dest / _MS_FILE)
    tmp_j = dest / (_MS_STAMP + ".tmp")
    tmp_j.write_text(json.dumps(dict(stamp=list(stamp))), encoding="utf-8")
    tmp_j.replace(dest / _MS_STAMP)


def cached_model_scores(set_code: str) -> pl.DataFrame | None:
    """The model scores of the set's Sealed decks if they were already computed for this model and corpus (memory, then the file
    next to the model), else None.  Never computes: cheap enough to call on every search."""
    sc = set_code.upper()
    stamp = _model_stamp(sc)
    if stamp is None:
        return None
    hit = _MS_CACHE.get(sc)
    if hit is not None and hit[0] == stamp:
        return hit[1]
    d = model_dir(sc)
    try:
        saved = json.loads((d / _MS_STAMP).read_text(encoding="utf-8"))
        if tuple(saved.get("stamp") or ()) != stamp:
            return None
        df = pl.read_parquet(d / _MS_FILE)
    except (OSError, ValueError, pl.exceptions.PolarsError):
        return None
    _MS_CACHE[sc] = (stamp, df)
    return df


def model_scores(set_code: str, progress=None, cancel: threading.Event | None = None,
                 write_dir: Path | None = None) -> pl.DataFrame:
    """Score every Sealed build of the corpus with the set's model, as the deck scorer does: `model_q` = the model's game win rate
    of the deck (reference draw), `model_score` = its percentile (0-100) among the real builds of the format.  One row per
    (deck_key, fmt); Bo1 runs are scored in bo1, Bo3 runs in bo3, draft runs not at all (the model is a Sealed model).

    A few ms per deck (about a minute or two for a big set), so the result is kept in memory and written next to the model; it is
    computed again only when the model or the corpus changes.  When `write_dir` is given (used by `tools/bundle_sets.py`), the
    cache files are written there instead of into `model_dir(set_code)` so the app home stays read-only.  Raises ValueError when
    the set has no model."""
    from sealed_lab.types import Cancelled

    sc = set_code.upper()
    with _LOCK:                                # the search lock (re-entrant): searches already hold it when they call this
        stamp = _model_stamp(sc)
        hit = cached_model_scores(sc)
        if hit is not None:
            if write_dir is not None and stamp is not None:
                _write_model_scores(Path(write_dir), stamp, hit)
            return hit
        if stamp is None:
            raise ValueError(f"{sc} has no model yet: calculate the set to sort decks by model score")
        from sealed_lab.deck import builder

        model = builder.load_model(sc)
        idx = _load(sc)
        fmts = [f for f in ("bo1", "bo3") if f in model.formats]
        if idx.provisional:
            # a pick-file set: its Premier Draft runs are Bo1 games, scored by the provisional model's bo1 format (rows keep the runs' own "draft" fmt)
            if "bo1" not in model.formats:
                raise ValueError(f"{sc} has no Bo1 model")
            runs = idx.runs.select("event", "draft_id", "fmt")
            score_fmt = {"draft": "bo1"}
        else:
            runs = idx.runs.filter(pl.col("fmt").is_in(fmts)).select("event", "draft_id", "fmt")
            score_fmt = {}
        b = idx.builds.join(runs, on=["event", "draft_id"], how="inner").select("deck_key", "fmt").unique()
        keys = b["deck_key"].unique().to_list()
        dc = (pl.scan_parquet(idx.dc_path).filter(pl.col("deck_key").is_in(keys))
              .select("deck_key", "card_id", "n").collect())
        decks: dict[int, dict[int, int]] = {}
        for k, c, n in dc.iter_rows():
            if n and n > 0:
                decks.setdefault(int(k), {})[int(c)] = int(n)
        rows_k, rows_f, rows_q, rows_s = [], [], [], []
        todo = b.rows()
        n_all = max(len(todo), 1)
        for i, (k, f) in enumerate(todo):
            if i % 250 == 0:
                if cancel is not None and cancel.is_set():
                    raise Cancelled()
                if progress is not None:
                    progress(i / n_all, f"Scoring real decks with the model: {i:,} of {len(todo):,}")
            deck = decks.get(int(k))
            if not deck:
                continue
            try:
                mf = score_fmt.get(f, f)
                q = float(model.q(deck, mf))
                pct = float(model.percentile(q, mf))
            except Exception:  # noqa: BLE001 - one odd deck must not stop the others
                log.debug("model score of deck %s (%s) failed", k, f, exc_info=True)
                continue
            rows_k.append(int(k))
            rows_f.append(f)
            rows_q.append(q)
            rows_s.append(pct)
        df = pl.DataFrame(dict(deck_key=pl.Series(rows_k, dtype=pl.Int64), fmt=pl.Series(rows_f, dtype=pl.String),
                               model_q=pl.Series(rows_q, dtype=pl.Float32), model_score=pl.Series(rows_s, dtype=pl.Float32)))
        if progress is not None:
            progress(1.0, f"Scored {df.height:,} real decks")
        _MS_CACHE[sc] = (stamp, df)
        dest = Path(write_dir) if write_dir is not None else model_dir(sc)
        try:                                   # the file only saves time next run: failing to write it is not an error
            _write_model_scores(dest, stamp, df)
        except OSError:
            log.warning("could not save the model scores of %s", sc, exc_info=True)
        return df


def _parse_includes(idx: _SetIndex, card: str | int | None,
                    include_cards: list | None) -> tuple[int | None, str | None, list[tuple[int, int]]]:
    primary_cid: int | None = None
    primary_name: str | None = None
    reqs: dict[int, int] = {}
    if card is not None and (not isinstance(card, str) or card.strip()):
        primary_cid = _require_card(idx, card)
        primary_name = idx.card_meta[primary_cid]["name"]
        reqs[primary_cid] = 1
    for item in (include_cards or []):
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            c_ref, min_n = item[0], max(1, int(item[1]))
        else:
            c_ref, min_n = item, 1
        cid = _require_card(idx, c_ref)
        reqs[cid] = max(reqs.get(cid, 0), min_n)
        if primary_cid is None:
            primary_cid = cid
            primary_name = idx.card_meta[cid]["name"]
    return primary_cid, primary_name, list(reqs.items())


def _search(set_code, card, events, sort, min_games, limit, offset, min_card_share, *,
            include_cards=None, exclude_cards=None, in_pool_only=None, colors=None, color_match="exact", splash=None,
            min_skill=None, min_rank=None, ranks=None, trophy_only=False, min_lands=None, max_lands=None,
            days=None, with_analytics=True) -> dict:
    idx = _load(set_code)
    primary_cid, primary_name, reqs = _parse_includes(idx, card, include_cards)
    excl_ids = [_require_card(idx, c) for c in (exclude_cards or [])]
    pool_only_ids = [_require_card(idx, c) for c in (in_pool_only or [])]
    all_excl = sorted(set(excl_ids + pool_only_ids))

    b = idx.builds
    dc_scan = pl.scan_parquet(idx.dc_path)

    # 1. Include card constraints (each required card must be in the same build with n >= min_n)
    if reqs:
        # Primary card's copies column
        dk_p = (dc_scan.filter((pl.col("card_id") == primary_cid) & (pl.col("n") >= dict(reqs)[primary_cid]))
                .select("deck_key", pl.col("n").alias("copies")).collect())
        b = b.join(dk_p, on="deck_key", how="inner")
        for cid, min_n in reqs:
            if cid == primary_cid:
                continue
            dk_x = (dc_scan.filter((pl.col("card_id") == cid) & (pl.col("n") >= min_n))
                    .select("deck_key").unique().collect())
            b = b.join(dk_x, on="deck_key", how="inner")
    else:
        b = b.with_columns(pl.lit(0, dtype=pl.Int8).alias("copies"))

    # 2. Exclude card constraints (must not appear in the build)
    if all_excl and b.height:
        bad_dk = (dc_scan.filter(pl.col("card_id").is_in(all_excl))
                  .select("deck_key").unique().collect())
        if bad_dk.height:
            b = b.join(bad_dk, on="deck_key", how="anti")

    # 3. In-pool-only constraints (Sealed: must be in pools.parquet with n >= 1, and already excluded from main deck above)
    if pool_only_ids and b.height:
        if not idx.has_pools:
            b = b.head(0)
        else:
            p_scan = pl.scan_parquet(idx.pools_path)
            for cid in pool_only_ids:
                pids = p_scan.filter(pl.col("card_id") == cid).select("draft_id").unique().collect()
                b = b.join(pids, on="draft_id", how="inner")

    # 4. Build-level filters (colors, splash, lands)
    if b.height and (colors or splash or min_lands or max_lands):
        b = _apply_build_filters(b, colors, splash, min_lands, max_lands, color_match=color_match)

    # Collapse builds -> per-run representative build (most games played, then earliest build_index)
    if b.height:
        b = b.sort(["event", "draft_id", "games", "build_index"], descending=[False, False, True, False])
        per_run = b.group_by(["event", "draft_id"], maintain_order=True).agg(
            pl.col("deck_key").first(), pl.col("build_index").first(), pl.col("copies").first(),
            pl.col("games").first().alias("build_games"), pl.col("wins").first().alias("build_wins"),
            pl.col("main_colors").first().alias("main_colors"), pl.col("splash_colors").first().alias("splash_colors"),
            pl.col("games").sum().cast(pl.Int32).alias("games_with_card"),
            pl.col("wins").sum().cast(pl.Int32).alias("wins_with_card"),
            pl.col("copies").max().alias("max_copies"),
        )
        r = idx.runs.join(per_run, on=["event", "draft_id"], how="inner")
    else:
        r = idx.runs.head(0).with_columns(
            pl.lit(0, dtype=pl.Int64).alias("deck_key"), pl.lit(0, dtype=pl.Int16).alias("build_index"),
            pl.lit(0, dtype=pl.Int8).alias("copies"), pl.lit(0, dtype=pl.Int32).alias("build_games"),
            pl.lit(0, dtype=pl.Int32).alias("build_wins"), pl.lit("", dtype=pl.String).alias("main_colors"),
            pl.lit("", dtype=pl.String).alias("splash_colors"), pl.lit(0, dtype=pl.Int32).alias("games_with_card"),
            pl.lit(0, dtype=pl.Int32).alias("wins_with_card"), pl.lit(0, dtype=pl.Int8).alias("max_copies"),
        )

    by_event_all = {e: int(n) for e, n in r.group_by("event").len().iter_rows()} if r.height else {}
    total_all = r.height

    # 5. Run-level filters
    if events:
        r = r.filter(pl.col("event").is_in(list(events)))
    if min_games and min_games > 1:
        r = r.filter(pl.col("games") >= int(min_games))
    if primary_cid is not None and min_card_share and min_card_share > 0:
        r = r.filter(pl.col("games_with_card") >= float(min_card_share) * pl.col("games"))
    if min_skill is not None and min_skill > 0:
        r = r.filter(pl.col("skill").is_not_null() & (pl.col("skill") >= float(min_skill)))
    if min_rank:
        tier = _rank_tier(min_rank)
        if tier > 0:
            r = r.filter(pl.col("rank_tier") >= tier)
    if ranks:
        tiers = sorted({t for t in (_rank_tier(x) for x in ranks) if t > 0})
        if tiers:
            r = r.filter(pl.col("rank_tier").is_in(tiers))
    if trophy_only:
        r = r.filter(pl.col("trophy"))
    cut_dt, cut_str = _date_cutoff(idx, events, days)
    if cut_dt is not None:
        r = r.filter(pl.col("date").is_not_null() & (pl.col("date") >= cut_dt))

    cast_all, cast_summary, cast_note = (None, None, None)
    if primary_cid is not None:
        cast_all, cast_summary, cast_note = _cast_lens(idx, r, primary_cid, summary=with_analytics)

    ms = model_scores(idx.set_code) if sort == "model" else cached_model_scores(idx.set_code)
    if ms is not None:
        r = r.join(ms, on=["deck_key", "fmt"], how="left")
    else:
        r = r.with_columns(pl.lit(None, dtype=pl.Float32).alias("model_q"), pl.lit(None, dtype=pl.Float32).alias("model_score"))
    by_event = {e: int(n) for e, n in r.group_by("event").len().iter_rows()} if r.height else {}
    analytics = _compute_analytics(idx, r, primary_cid, [c for c, _ in reqs], events, colors, cut_dt=cut_dt) if with_analytics else None

    cols, desc = _sort_exprs(sort)
    r = r.sort(cols, descending=desc, nulls_last=True)
    page = r.slice(int(offset), int(limit)) if limit else r.slice(int(offset))
    return dict(
        set_code=idx.set_code, card=primary_name, card_id=primary_cid,
        total=r.height, total_all=total_all, shown=page.height, offset=int(offset),
        by_event=by_event, by_event_all=by_event_all, sort=sort, min_games=int(min_games),
        min_card_share=float(min_card_share or 0.0), days=days, date_cutoff=cut_str,
        analytics=analytics,
        cast_summary=cast_summary, cast_note=cast_note,
        results=_attach_cast(_result_rows(idx, page), cast_all, primary_cid is not None),
    )


# ---------------------------------------------------------------------------------------------
# How often the searched card was cast in each deck's games (17Lands replay data)
# ---------------------------------------------------------------------------------------------

CAST_NOTE = "Cast counts come from 17Lands replay data and cover each player's first 12 turns."
CAST_NOTE_NO_REPLAY = "No replay data for this set"
CAST_NOTE_PICK_ONLY = "This set has only pick data: its decks were never played in recorded games."
CAST_NOTE_NO_ID = "No replay data for this card"
CAST_NOTE_NO_GAMES = "No replay games for these decks"


def _cast_prepare(idx: _SetIndex) -> dict | str:
    """Per-set cache of the cast lens: grpIds per corpus card and the link replay gid -> (event, draft_id, deck_key).
    Returns the state dict, or the note explaining why it is unavailable.  Rebuilt when the replay folder changes."""
    if idx.provisional:
        return CAST_NOTE_PICK_ONLY
    rdir = cache_dir(idx.set_code) / "replay"
    idmap_path = rdir / "idmap.json"
    if not (rdir.is_dir() and idmap_path.exists()):
        return CAST_NOTE_NO_REPLAY
    shards = sorted(p for p in rdir.iterdir() if p.is_dir() and (p / "casts.parquet").exists() and (p / "games.parquet").exists())
    if not shards:
        return CAST_NOTE_NO_REPLAY
    stamp = (str(rdir), idmap_path.stat().st_mtime_ns, tuple((p.name, (p / "casts.parquet").stat().st_mtime_ns) for p in shards))
    st = idx._cast_state
    if st is not None and st["stamp"] == stamp:
        return st
    raw = json.loads(idmap_path.read_text(encoding="utf-8"))
    mapping = raw.get("map", raw) if isinstance(raw, dict) else {}
    grp: dict[int, list[int]] = {}
    for gid_s, target in (mapping.items() if isinstance(mapping, dict) else ()):
        if isinstance(target, dict):
            target = target.get("card_id")
        try:
            grp.setdefault(int(target), []).append(int(gid_s))
        except (TypeError, ValueError):
            continue
    k3 = ["draft_id", "match_number", "game_number"]
    rg = pl.concat([pl.scan_parquet(p / "games.parquet").select("gid", *k3) for p in shards]).unique(subset=k3).collect()
    cg = (pl.scan_parquet(idx.cdir / "games.parquet").select("event", "deck_key", *k3).unique(subset=k3)
          .join(rg.lazy(), on=k3, how="inner").select("event", "draft_id", "deck_key", "gid").collect())
    st = dict(stamp=stamp, grp=grp, links=cg, casts=[p / "casts.parquet" for p in shards])
    idx._cast_state = st
    return st


def _cast_lens(idx: _SetIndex, r: pl.DataFrame, cid: int, summary: bool = True):
    """(per-run frame, summary dict, note) of the card's casts over the runs of `r` (one lazy scan of the replay casts).
    The per-run frame is keyed by (event, draft_id, deck_key); every part is None when unavailable.  Never raises."""
    try:
        st = _cast_prepare(idx)
        if isinstance(st, str):
            return None, None, st
        ids = st["grp"].get(cid)
        if not ids:
            return None, None, CAST_NOTE_NO_ID
        if r.height == 0:
            return None, None, CAST_NOTE
        keys = ["event", "draft_id", "deck_key"]
        links = st["links"].join(r.select(keys).unique(), on=keys, how="semi")
        if links.height == 0:
            return None, None, CAST_NOTE_NO_GAMES
        per_gid = (pl.scan_parquet(st["casts"]).filter(pl.col("id").is_in(ids) & pl.col("gid").is_in(links["gid"].implode()))
                   .group_by("gid").agg(pl.col("n").sum().cast(pl.Int64).alias("times"), pl.col("turn").min().cast(pl.Float64).alias("ft"))
                   .collect())
        per_run = (links.join(per_gid, on="gid", how="left")
                   .group_by(keys).agg(pl.len().cast(pl.Int64).alias("games"), pl.col("times").fill_null(0).sum().alias("times"),
                                       pl.col("ft").is_not_null().sum().cast(pl.Int64).alias("games_cast"),
                                       pl.col("ft").sum().alias("ft_sum")))
        summ = None
        if summary:
            tot = per_run.select(pl.len().alias("runs"), pl.col("games").sum().alias("games"), pl.col("times").sum().alias("times"),
                                 pl.col("games_cast").sum().alias("games_cast"), pl.col("ft_sum").sum().alias("ft")).row(0, named=True)
            summ = dict(runs=int(tot["runs"]), games=int(tot["games"]), times=int(tot["times"]), games_cast=int(tot["games_cast"]),
                        first_turn=float(tot["ft"] / tot["games_cast"]) if tot["games_cast"] else None,
                        per_game=float(tot["times"] / tot["games"]))
        return per_run, summ, CAST_NOTE
    except Exception:  # noqa: BLE001 - the cast lens is optional and must never break the search
        log.debug("cast lens skipped for %s/%s", idx.set_code, cid, exc_info=True)
        return None, None, CAST_NOTE_NO_REPLAY


def _attach_cast(rows: list[dict], per_run: pl.DataFrame | None, card_search: bool) -> list[dict]:
    """Give every row of a card search its `cast` dict (None where the run has no replay game)."""
    if not card_search:
        return rows
    look: dict[tuple, dict] = {}
    if per_run is not None:
        for e, d, k, g, t, gc, fs in per_run.select("event", "draft_id", "deck_key", "games", "times", "games_cast", "ft_sum").iter_rows():
            look[(e, d, int(k))] = dict(games=int(g), times=int(t), games_cast=int(gc), first_turn=float(fs / gc) if gc else None)
    for x in rows:
        x["cast"] = look.get((x["event"], x["draft_id"], x["deck_key"]))
    return rows


def _compute_analytics(idx: _SetIndex, r: pl.DataFrame, primary_cid: int | None,
                       req_ids: list[int], events: list[str] | None, colors: str | None,
                       cut_dt=None) -> dict:
    """Instant corpus-wide analytics computed over ALL filtered runs `r` (not just the paged slice)."""
    if r.height == 0:
        return dict(n_runs=0, n_games=0, avg_wr=None, trophy_rate=None, by_color=[],
                    main_vs_splash=None, by_copies=[], companions=[], inclusion_gap=None, play_profile=None)

    n_runs = int(r.height)
    tot_games = int(r["games"].sum())
    tot_wins = int(r["wins"].sum())
    avg_wr = float(tot_wins / tot_games) if tot_games else None
    trophy_rate = float(r["trophy"].mean()) if n_runs else None

    # Effective main_colors per run
    r_c = r.with_columns(pl.coalesce(pl.col("main_colors"), pl.col("run_main_colors")).fill_null("").alias("_mc"))

    # No skill "adjustment": the 17lands win-rate bucket includes these very games and its effect is not one-for-one, so each color
    # pair shows its raw win rate next to the games-weighted average skill bucket of its pilots (who played it), not a corrected rate.
    r_c = r_c.with_columns((pl.col("skill") * pl.col("games")).alias("_sk_g"),
                           pl.when(pl.col("skill").is_not_null()).then(pl.col("games")).otherwise(0).alias("_sk_n"))

    # 1. Color pair breakdown (top 10 by runs)
    bc_df = (r_c.filter(pl.col("_mc") != "")
             .group_by("_mc")
             .agg(pl.len().alias("runs"), pl.col("games").sum().alias("games"),
                  pl.col("wins").sum().alias("wins"), pl.col("_sk_g").sum().alias("sk_g"), pl.col("_sk_n").sum().alias("sk_n"),
                  *((pl.col("model_score").median().alias("ms_med"), pl.col("model_score").max().alias("ms_max"),
                     pl.col("model_score").count().alias("ms_n")) if "model_score" in r_c.columns else ()))
             .sort(["runs", "games"], descending=[True, True])
             .head(10))
    by_color = []
    for row in bc_df.iter_rows(named=True):
        g_ = int(row["games"])
        by_color.append(dict(
            colors=row["_mc"], runs=int(row["runs"]), share=float(row["runs"] / n_runs),
            games=g_, raw_wr=float(row["wins"] / g_) if g_ else None,
            avg_skill=float(row["sk_g"] / row["sk_n"]) if row["sk_n"] else None,
            model_median=float(row["ms_med"]) if row.get("ms_med") is not None else None,
            model_best=float(row["ms_max"]) if row.get("ms_max") is not None else None,
            model_n=int(row.get("ms_n") or 0),
        ))

    # 2. Main vs Splash (when primary card has color(s))
    main_vs_splash = None
    if primary_cid is not None:
        ccols = idx.card_meta.get(primary_cid, {}).get("colors", "")
        if ccols:
            is_main = pl.lit(True)
            for ch in ccols:
                is_main = is_main & pl.col("_mc").str.contains(ch, literal=True)
            ms = r_c.filter(pl.col("_mc") != "").with_columns(is_main.alias("_is_main"))
            m_df = ms.filter(pl.col("_is_main"))
            s_df = ms.filter(~pl.col("_is_main"))
            def _pack(df: pl.DataFrame) -> dict:
                gg = int(df["games"].sum()) if df.height else 0
                ww = int(df["wins"].sum()) if df.height else 0
                return dict(runs=int(df.height), games=gg, wr=float(ww / gg) if gg else None)
            if m_df.height or s_df.height:
                main_vs_splash = dict(main=_pack(m_df), splash=_pack(s_df))

    # 3. Copy count breakdown (1 / 2 / 3+)
    by_copies = []
    if primary_cid is not None:
        cp_df = (r_c.with_columns(
            pl.when(pl.col("copies") >= 3).then(pl.lit("3+")).otherwise(pl.col("copies").cast(pl.String)).alias("_cp")
        ).group_by("_cp").agg(pl.len().alias("runs"), pl.col("games").sum().alias("games"), pl.col("wins").sum().alias("wins")))
        order = {"1": 0, "2": 1, "3+": 2}
        for row in sorted(cp_df.iter_rows(named=True), key=lambda x: order.get(x["_cp"], 9)):
            gg = int(row["games"])
            by_copies.append(dict(copies=row["_cp"], runs=int(row["runs"]), games=gg,
                                  wr=float(row["wins"] / gg) if gg else None))

    # 4. Companion cards (top 8 non-land co-stars by lift in top filtered builds vs baseline)
    companions = _compute_companions(idx, r_c, set(req_ids), events, colors, cut_dt=cut_dt)

    # 5. Sealed inclusion gap (all pilots vs top >=60% pilots among on-color pools containing primary_cid)
    inclusion_gap = _compute_inclusion_gap(idx, primary_cid, events, cut_dt=cut_dt) if primary_cid is not None else None

    # 6. Phase 3D: Opening-hand & drawn-later WR + optional replay cast-turn profile
    play_profile = _compute_play_profile(idx, r_c, primary_cid) if primary_cid is not None else None

    return dict(
        n_runs=n_runs, n_games=tot_games, avg_wr=avg_wr, trophy_rate=trophy_rate,
        by_color=by_color, main_vs_splash=main_vs_splash, by_copies=by_copies,
        companions=companions, inclusion_gap=inclusion_gap, play_profile=play_profile,
        pool_kind="drafted" if idx.provisional else "sealed",
    )


def _compute_companions(idx: _SetIndex, r: pl.DataFrame, skip_ids: set[int],
                        events: list[str] | None, colors: str | None, cut_dt=None) -> list[dict]:
    """Cards that strong pilots' matching decks play more often than decks of the SAME color pairs in the same events and dates.

    "Strong" = skill bucket >= 0.60 only: trophy decks are chosen by their result, so they are not used.  The baseline is color-
    standardised (the strong decks' own mix of color pairs, each pair's inclusion rate among all builds of the events / dates), so a
    lift is not just "this card is in the colors the search asks for".  Descriptive: what strong pilots play, not what wins."""
    top = r.filter(pl.col("skill").is_not_null() & (pl.col("skill") >= 0.60))
    if top.height < 10:
        return []                                           # too few strong pilots: no list rather than a silent fallback to everyone
    use = top.with_columns(pl.col("_mc").fill_null("")) if "_mc" in top.columns else top.with_columns(pl.lit("").alias("_mc"))
    n_use = use.height
    keys = use["deck_key"].unique().to_list()
    nonland_ids = [cid for cid, m in idx.card_meta.items() if not m["is_land"] and not m["is_basic"] and cid not in skip_ids]
    if not nonland_ids or not keys:
        return []
    dc = (pl.scan_parquet(idx.dc_path)
          .filter(pl.col("deck_key").is_in(keys) & pl.col("card_id").is_in(nonland_ids))
          .select("deck_key", "card_id").unique().collect())
    if dc.height == 0:
        return []
    j = use.select("deck_key", "games", "wins").join(dc, on="deck_key", how="inner")
    agg = j.group_by("card_id").agg(pl.len().alias("n_runs"), pl.col("games").sum().alias("games"), pl.col("wins").sum().alias("wins"))
    agg = agg.filter(pl.col("n_runs") >= max(3, int(0.08 * n_use)))
    if agg.height == 0:
        return []

    # baseline: builds of the same events (and dates), per color pair
    base_b = idx.builds
    if events:
        base_b = base_b.filter(pl.col("event").is_in(list(events)))
    if cut_dt is not None:
        dated = idx.runs.filter(pl.col("date").is_not_null() & (pl.col("date") >= cut_dt)).select("event", "draft_id")
        base_b = base_b.join(dated, on=["event", "draft_id"], how="inner")
    base_b = base_b.with_columns(pl.col("main_colors").fill_null("").alias("_mc")).select("deck_key", "_mc").unique()
    mix = use.group_by("_mc").agg((pl.len() / n_use).alias("w"))                          # the strong decks' color-pair mix
    base_n = base_b.group_by("_mc").agg(pl.len().alias("n_pair"))
    cand_ids = agg["card_id"].to_list()
    base_dc = (pl.scan_parquet(idx.dc_path).filter(pl.col("card_id").is_in(cand_ids))
               .select("deck_key", "card_id").unique().collect()
               .join(base_b, on="deck_key", how="inner")
               .group_by("card_id", "_mc").len().rename({"len": "n_with"})
               .join(base_n, on="_mc").with_columns((pl.col("n_with") / pl.col("n_pair")).alias("rate"))
               .join(mix, on="_mc", how="inner")
               .group_by("card_id").agg((pl.col("rate") * pl.col("w")).sum().alias("base")))
    base_map = {int(c): float(v) for c, v in base_dc.iter_rows()}

    rows = []
    for row in agg.iter_rows(named=True):
        cid = int(row["card_id"])
        pct = float(row["n_runs"] / n_use)
        bp = base_map.get(cid, 0.0)
        lift = float(pct / max(bp, 0.005))
        gg = int(row["games"])
        rows.append(dict(card_id=cid, name=idx.card_meta[cid]["name"], pct_in_matches=pct, base_pct=bp, lift=lift,
                         wr=float(row["wins"] / gg) if gg else None, n_runs=int(row["n_runs"]), _score=lift * math.sqrt(pct)))
    rows.sort(key=lambda x: (-x["_score"], -x["pct_in_matches"], x["name"]))
    for x in rows:
        x.pop("_score", None)
    return rows[:8]


def _compute_inclusion_gap(idx: _SetIndex, cid: int, events: list[str] | None, cut_dt=None) -> dict | None:
    if not idx.has_pools:
        return None
    sealed_evs = [e["event"] for e in idx.events if e["fmt"] in ("bo1", "bo3") or idx.provisional]   # Sealed pools; a pick-file set: drafted pools
    if events:
        sealed_evs = [e for e in sealed_evs if e in events]
    if not sealed_evs:
        return None
    pids = (pl.scan_parquet(idx.pools_path)
            .filter(pl.col("card_id") == cid)
            .select("draft_id").unique().collect())
    if pids.height == 0:
        return None
    s_runs = idx.runs.filter(pl.col("event").is_in(sealed_evs)).join(pids, on="draft_id", how="inner")
    if cut_dt is not None:
        s_runs = s_runs.filter(pl.col("date").is_not_null() & (pl.col("date") >= cut_dt))
    if s_runs.height < 5:
        return None
    ccols = idx.card_meta.get(cid, {}).get("colors", "")
    if ccols:
        mc = pl.col("run_main_colors").fill_null("")
        cond = pl.lit(True)
        for ch in ccols:
            cond = cond & mc.str.contains(ch, literal=True)
        s_runs = s_runs.filter(cond)
    if s_runs.height < 5:
        return None
    played_dk = (pl.scan_parquet(idx.dc_path)
                 .filter(pl.col("card_id") == cid).select("deck_key").unique().collect())
    played_runs = idx.builds.join(played_dk, on="deck_key", how="inner").select("event", "draft_id").unique().with_columns(pl.lit(True).alias("_played"))
    j = s_runs.select("event", "draft_id", "skill").join(played_runs, on=["event", "draft_id"], how="left").with_columns(
        pl.col("_played").fill_null(False)
    )
    all_n = int(j.height)
    all_rate = float(j["_played"].mean())
    top = j.filter(pl.col("skill").is_not_null() & (pl.col("skill") >= 0.60))
    top_n = int(top.height)
    top_rate = float(top["_played"].mean()) if top_n >= 5 else None
    return dict(all_rate=all_rate, all_n=all_n, top_rate=top_rate, top_n=top_n)


# ---------------------------------------------------------------------------------------------
# Phase 3D: Opening-Hand & Turn-by-Turn Replay Lens
# ---------------------------------------------------------------------------------------------

def _get_replay_mull_info(idx: _SetIndex, shards: list[Path]) -> dict | None:
    """Lazy per-set cache of baseline 1st-hand keep rate and mulliganed gids from replay shards' games.parquet."""
    if idx._replay_mull_cache is not None:
        return idx._replay_mull_cache or None
    gfiles = [p / "games.parquet" for p in shards if (p / "games.parquet").exists()]
    if not gfiles:
        idx._replay_mull_cache = {}
        return None
    try:
        gdf = pl.concat([pl.scan_parquet(p).select("gid", pl.col("num_mulligans").fill_null(0).alias("m")) for p in gfiles]).collect()
        if gdf.height == 0:
            idx._replay_mull_cache = {}
            return None
        mull1 = gdf.filter(pl.col("m") >= 1).select("gid")
        mull2 = gdf.filter(pl.col("m") >= 2).select("gid")
        base_keep = float(1.0 - mull1.height / gdf.height)
        idx._replay_mull_cache = dict(n_games=int(gdf.height), base_keep_rate=base_keep, mull1=mull1, mull2=mull2)
        return idx._replay_mull_cache
    except Exception:  # noqa: BLE001
        idx._replay_mull_cache = {}
        return None


def _replay_card_info(idx: _SetIndex, cid: int) -> dict | None:
    """Fast read-only replay lens for `cid`: reads cast-turn distribution, seen/cast rate, and pre-mulligan candidate-hand keep rate."""
    if cid in idx._replay_hist_cache:
        return idx._replay_hist_cache[cid]
    out: dict | None = None
    try:
        rdir = cache_dir(idx.set_code) / "replay"
        idmap_path = rdir / "idmap.json"
        if rdir.is_dir() and idmap_path.exists():
            raw_map = json.loads(idmap_path.read_text(encoding="utf-8"))
            grp_ids: list[int] = []
            cname = idx.card_meta.get(cid, {}).get("name", "")
            mapping = raw_map.get("map", raw_map) if isinstance(raw_map, dict) else {}
            if isinstance(mapping, dict):
                for gid_s, target in mapping.items():
                    if target == cid or target == cname or (isinstance(target, dict) and (target.get("card_id") == cid or target.get("name") == cname)):
                        try:
                            grp_ids.append(int(gid_s))
                        except (TypeError, ValueError):
                            pass
            if grp_ids:
                shards = [p for p in rdir.iterdir() if p.is_dir() and ((p / "casts.parquet").exists() or (p / "draws.parquet").exists())]
                if shards:
                    cast_files = [p / "casts.parquet" for p in shards if (p / "casts.parquet").exists()]
                    fc = (pl.concat([pl.scan_parquet(p).filter(pl.col("id").is_in(grp_ids)).select("gid", "turn") for p in cast_files])
                          .group_by("gid").agg(pl.col("turn").clip(1, 10).min().alias("t")).collect()) if cast_files else pl.DataFrame(schema={"gid": pl.Int64, "t": pl.Int8})
                    draw_files = [p / "draws.parquet" for p in shards if (p / "draws.parquet").exists()]
                    n_seen = 0
                    n_cand1, cand1_keep_rate, base_keep_rate, keep_delta = 0, None, None, None
                    n_cand2, cand2_keep_rate = 0, None
                    if draw_files:
                        ddf = (pl.concat([pl.scan_parquet(p).filter(pl.col("id").is_in(grp_ids) & pl.col("kind").is_in([0, 1, 2, 3, 4]))
                                          .select("gid", "kind") for p in draw_files]).collect())
                        if ddf.height:
                            n_seen = int(ddf.filter(pl.col("kind") <= 2)["gid"].n_unique())
                            c1 = ddf.filter(pl.col("kind") == 3).select("gid").unique()
                            n_cand1 = int(c1.height)
                            c2 = ddf.filter(pl.col("kind") == 4).select("gid").unique()
                            n_cand2 = int(c2.height)
                            minfo = _get_replay_mull_info(idx, shards)
                            if minfo is not None and n_cand1 >= 20:
                                n_m1 = int(c1.join(minfo["mull1"], on="gid", how="inner").height)
                                cand1_keep_rate = float(1.0 - n_m1 / n_cand1)
                                base_keep_rate = float(minfo["base_keep_rate"])
                                keep_delta = float(cand1_keep_rate - base_keep_rate)
                                if n_cand2 >= 15:
                                    n_m2 = int(c2.join(minfo["mull2"], on="gid", how="inner").height)
                                    cand2_keep_rate = float(1.0 - n_m2 / n_cand2)
                    tot = int(fc.height)
                    if tot >= 10 or cand1_keep_rate is not None:
                        hist: list[dict] = []
                        mean_turn = None
                        cast_rate = None
                        if tot >= 10:
                            vc = fc.with_columns(pl.when(pl.col("t") >= 8).then(8).otherwise(pl.col("t")).alias("tb")).group_by("tb").len().sort("tb")
                            n_seen = max(n_seen, tot)
                            hist = [dict(turn=("8+" if int(t) == 8 else str(int(t))), count=int(n), share=float(n / tot))
                                    for t, n in vc.iter_rows()]
                            mean_turn = float(fc["t"].mean())
                            cast_rate = float(tot / n_seen) if n_seen >= tot and n_seen > 0 else None
                        out = dict(
                            n_seen=n_seen, n_cast=tot,
                            cast_rate=cast_rate,
                            mean_cast_turn=mean_turn,
                            open_cast_rate=None, open_cast_turn=None,
                            turn_hist=hist,
                            n_cand1=n_cand1, cand1_keep_rate=cand1_keep_rate,
                            base_keep_rate=base_keep_rate, keep_delta=keep_delta,
                            n_cand2=n_cand2, cand2_keep_rate=cand2_keep_rate,
                        )
    except Exception:  # noqa: BLE001 - optional replay cache must never fail deck search
        log.debug("replay card lens skipped for %s/%s", idx.set_code, cid, exc_info=True)
    idx._replay_hist_cache[cid] = out
    return out


def _compute_play_profile(idx: _SetIndex, r: pl.DataFrame, cid: int) -> dict | None:
    """Within-deck Opening-Hand & Drawn-Later WR delta from hands.parquet + optional replay cast-turn lens."""
    if not idx.has_hands or r.height == 0:
        return None
    try:
        dks = r["deck_key"].unique().to_list()
        if not dks:
            return None
        # Filter games to these exact filtered runs & matching decks containing `cid`
        run_keys = r.select("event", "draft_id", "deck_key").unique()
        g = (pl.scan_parquet(idx.cdir / "games.parquet")
             .filter(pl.col("deck_key").is_in(dks))
             .select("game_key", "event", "draft_id", "deck_key", "won")
             .collect()
             .join(run_keys, on=["event", "draft_id", "deck_key"], how="inner"))
        if g.height < 10:
            return None
        g = g.with_columns(pl.col("won").fill_null(False))
        h = (pl.scan_parquet(idx.cdir / "hands.parquet")
             .filter(pl.col("card_id") == cid)
             .select("game_key", "opening", "drawn")
             .collect())
        j = g.join(h, on="game_key", how="left").with_columns(
            pl.col("opening").fill_null(0).alias("op"),
            pl.col("drawn").fill_null(0).alias("dr"),
        )
        oh = j.filter(pl.col("op") > 0)
        not_oh = j.filter(pl.col("op") == 0)
        drawn = j.filter((pl.col("op") == 0) & (pl.col("dr") > 0))
        unseen = j.filter((pl.col("op") == 0) & (pl.col("dr") == 0))
        n_tot = int(j.height)
        oh_n, not_oh_n = int(oh.height), int(not_oh.height)
        dr_n, un_n = int(drawn.height), int(unseen.height)
        oh_wr = float(oh["won"].mean()) if oh_n else None
        not_oh_wr = float(not_oh["won"].mean()) if not_oh_n else None
        dr_wr = float(drawn["won"].mean()) if dr_n else None
        un_wr = float(unseen["won"].mean()) if un_n else None
        oh_delta = (oh_wr - not_oh_wr) if (oh_wr is not None and not_oh_wr is not None and oh_n >= 5 and not_oh_n >= 5) else None
        dr_delta = (dr_wr - un_wr) if (dr_wr is not None and un_wr is not None and dr_n >= 5 and un_n >= 5) else None
        return dict(
            n_games=n_tot,
            oh_games=oh_n, oh_wr=oh_wr, oh_delta=oh_delta,
            drawn_games=dr_n, drawn_wr=dr_wr, drawn_delta=dr_delta,
            unseen_games=un_n, unseen_wr=un_wr,
            replay=_replay_card_info(idx, cid),
        )
    except Exception:  # noqa: BLE001
        log.debug("play profile failed for %s/%s", idx.set_code, cid, exc_info=True)
        return None


# ---------------------------------------------------------------------------------------------
# Phase 3B: Similar real Sealed pools
# ---------------------------------------------------------------------------------------------

def _pool_card_weights(idx: _SetIndex) -> dict[int, float]:
    """Per-card importance weights for pool similarity (model |beta| when available, else rarity weights)."""
    if idx._card_weights is not None:
        return idx._card_weights
    rar_w = {"mythic": 3.5, "rare": 3.0, "uncommon": 1.6, "common": 1.0}
    w = {cid: rar_w.get(m["rarity"], 1.0) for cid, m in idx.card_meta.items() if not m["is_basic"]}
    # Enhance with model |beta| if a model exists for this set (read-only)
    try:
        mp = model_dir(idx.set_code)
        dp, cp = mp / "draws.npz", mp / "cards.parquet"
        if dp.exists() and cp.exists():
            mc = pl.read_parquet(cp, columns=["card_id", "name"])
            z = np.load(dp, allow_pickle=False)
            b_arr = None
            for k in ("bo3__beta", "bo1__beta"):                # the artifact's keys (model/artifact.save_artifact): <fmt>__beta
                if k in z.files:
                    arr = np.asarray(z[k], dtype=float)
                    b_arr = arr.mean(axis=0) if arr.ndim == 2 else arr  # posterior mean over the draws
                    break
            if b_arr is not None:
                for m_cid, nm in mc.iter_rows():
                    i = int(m_cid)
                    if 0 <= i < len(b_arr):
                        corp_cid = idx.by_name.get(str(nm).strip().lower())
                        if corp_cid is not None and corp_cid in w:
                            w[corp_cid] = float(1.0 + 20.0 * abs(float(b_arr[i])))
    except Exception:  # noqa: BLE001 - rarity weights are already a good fallback
        pass
    idx._card_weights = w
    return w


def similar_pools(set_code: str, pool_counts: dict[int | str, int], events: list[str] | None = None,
                  min_skill: float | None = None, trophy_only: bool = False,
                  limit: int = 50, min_games: int = DEFAULT_MIN_GAMES, days: int | None = None) -> dict:
    """Find the closest real 17Lands Sealed pools to `pool_counts` ({card_name_or_id: copies}).

    Uses weighted cosine similarity (emphasising rares/mythics or high-|beta| model cards) and returns
    the closest matching runs plus a color-pair breakdown of how pilots built similar pools.
    """
    with _LOCK:
        return _similar_pools(set_code, pool_counts, events, min_skill, trophy_only, limit, min_games, days)


def _similar_pools(set_code, pool_counts, events, min_skill, trophy_only, limit, min_games, days=None) -> dict:
    idx = _load(set_code)
    _need_games(idx)
    if not idx.has_pools:
        return dict(set_code=idx.set_code, total=0, shown=0, n_pool_cards=0, days=days, date_cutoff=None, color_breakdown=[], results=[])

    # Resolve user pool to corpus card_ids (non-basic only)
    user_pool: dict[int, int] = {}
    for k, n in (pool_counts or {}).items():
        nn = int(n or 0)
        if nn <= 0:
            continue
        cid = _resolve_in(idx, k)
        if cid is not None and not idx.card_meta[cid]["is_basic"]:
            user_pool[cid] = user_pool.get(cid, 0) + nn
    if not user_pool:
        return dict(set_code=idx.set_code, total=0, shown=0, n_pool_cards=0, days=days, date_cutoff=None, color_breakdown=[], results=[])

    # Filter candidate Sealed runs
    sealed_evs = [e["event"] for e in idx.events if e["fmt"] in ("bo1", "bo3")]
    if events:
        sealed_evs = [e for e in sealed_evs if e in events]
    if not sealed_evs:
        return dict(set_code=idx.set_code, total=0, shown=0, n_pool_cards=sum(user_pool.values()), days=days, date_cutoff=None, color_breakdown=[], results=[])

    cand = idx.runs.filter(pl.col("event").is_in(sealed_evs))
    if min_games and min_games > 1:
        cand = cand.filter(pl.col("games") >= int(min_games))
    if min_skill is not None and min_skill > 0:
        cand = cand.filter(pl.col("skill").is_not_null() & (pl.col("skill") >= float(min_skill)))
    if trophy_only:
        cand = cand.filter(pl.col("trophy"))
    cut_dt, cut_str = _date_cutoff(idx, sealed_evs, days)
    if cut_dt is not None:
        cand = cand.filter(pl.col("date").is_not_null() & (pl.col("date") >= cut_dt))
    if cand.height == 0:
        return dict(set_code=idx.set_code, total=0, shown=0, n_pool_cards=sum(user_pool.values()), days=days, date_cutoff=cut_str, color_breakdown=[], results=[])

    weights = _pool_card_weights(idx)
    u_df = pl.DataFrame({
        "card_id": list(user_pool.keys()),
        "u_n": [float(v) for v in user_pool.values()],
        "w": [float(weights.get(c, 1.0)) for c in user_pool.keys()],
    }, schema={"card_id": pl.Int32, "u_n": pl.Float64, "w": pl.Float64})
    user_w_sum = sum(float(weights.get(c, 1.0)) * float(n) for c, n in user_pool.items())

    w_df = pl.DataFrame({
        "card_id": list(weights.keys()),
        "w": list(weights.values()),
    }, schema={"card_id": pl.Int32, "w": pl.Float64})

    cand_ids = cand.select("draft_id").unique()
    pools = pl.scan_parquet(idx.pools_path).join(cand_ids.lazy(), on="draft_id", how="inner")

    # Symmetric weighted multiset Jaccard (Ruzicka similarity): sum(w * min(n, u)) / sum(w * max(n, u))
    dot_df = (pools.join(u_df.lazy(), on="card_id", how="inner")
              .with_columns((pl.col("w") * pl.min_horizontal(pl.col("n").cast(pl.Float64), pl.col("u_n"))).alias("_ov_w"),
                            pl.min_horizontal(pl.col("n").cast(pl.Int32), pl.col("u_n").cast(pl.Int32)).alias("_ov"))
              .group_by("draft_id")
              .agg(pl.col("_ov_w").sum().alias("ov_w"), pl.col("_ov").sum().alias("shared_n"))
              .collect())
    if dot_df.height == 0:
        return dict(set_code=idx.set_code, total=0, shown=0, n_pool_cards=sum(user_pool.values()), days=days, date_cutoff=cut_str, color_breakdown=[], results=[])

    # Pool weighted totals only for overlapping draft_ids
    ov_ids = dot_df.select("draft_id")
    norm_df = (pools.join(ov_ids.lazy(), on="draft_id", how="inner")
               .join(w_df.lazy(), on="card_id", how="left")
               .with_columns((pl.col("w").fill_null(1.0) * pl.col("n").cast(pl.Float64)).alias("_pw"))
               .group_by("draft_id").agg(pl.col("_pw").sum().alias("p_w_sum"))
               .collect())

    sim = (dot_df.join(norm_df, on="draft_id", how="inner")
           .with_columns((pl.col("ov_w") / (pl.col("p_w_sum") + float(user_w_sum) - pl.col("ov_w")).clip(lower_bound=1e-9))
                         .clip(0.0, 1.0).alias("similarity")))

    # Representative build per run (most games played)
    b_rep = (idx.builds.filter(pl.col("event").is_in(sealed_evs))
             .sort(["event", "draft_id", "games", "build_index"], descending=[False, False, True, False])
             .group_by(["event", "draft_id"], maintain_order=True)
             .agg(pl.col("deck_key").first(), pl.col("build_index").first(),
                  pl.col("games").first().alias("build_games"), pl.col("wins").first().alias("build_wins"),
                  pl.col("main_colors").first().alias("main_colors"), pl.col("splash_colors").first().alias("splash_colors")))

    r = (cand.join(sim.select("draft_id", "similarity", "shared_n"), on="draft_id", how="inner")
         .join(b_rep, on=["event", "draft_id"], how="inner")
         .with_columns(pl.lit(0, dtype=pl.Int8).alias("copies"),
                       pl.col("games").alias("games_with_card"), pl.col("wins").alias("wins_with_card")))
    ms = cached_model_scores(idx.set_code)
    if ms is not None:
        r = r.join(ms, on=["deck_key", "fmt"], how="left")
    else:
        r = r.with_columns(pl.lit(None, dtype=pl.Float32).alias("model_q"), pl.lit(None, dtype=pl.Float32).alias("model_score"))
    r = r.sort(["similarity", "score", "win_rate", "draft_id"], descending=[True, True, True, False])

    # Color breakdown over top 200 closest pools ("How pilots built pools like yours")
    top_n = r.head(min(200, r.height))
    cb_df = (top_n.with_columns(pl.coalesce(pl.col("main_colors"), pl.col("run_main_colors")).fill_null("").alias("_mc"))
             .filter(pl.col("_mc") != "")
             .group_by("_mc")
             .agg(pl.len().alias("runs"), pl.col("games").sum().alias("games"), pl.col("wins").sum().alias("wins"),
                  pl.col("model_score").median().alias("ms_med"), pl.col("model_score").max().alias("ms_max"),
                  pl.col("model_score").count().alias("ms_n"))
             .sort(["runs", "games"], descending=[True, True]).head(8))
    color_breakdown = [
        dict(colors=row["_mc"], runs=int(row["runs"]), share=float(row["runs"] / max(top_n.height, 1)),
             games=int(row["games"]), avg_wr=float(row["wins"] / row["games"]) if row["games"] else None,
             model_median=float(row["ms_med"]) if row.get("ms_med") is not None else None,
             model_best=float(row["ms_max"]) if row.get("ms_max") is not None else None,
             model_n=int(row.get("ms_n") or 0))
        for row in cb_df.iter_rows(named=True)
    ]

    page = r.head(int(limit)) if limit else r
    rows = _result_rows(idx, page)
    # Attach similarity & top shared notable cards
    sim_map = {row["draft_id"]: (float(row["similarity"]), int(row["shared_n"])) for row in page.select("draft_id", "similarity", "shared_n").iter_rows(named=True)}
    if rows:
        page_ids = [x["draft_id"] for x in rows]
        ov_cards = (pl.scan_parquet(idx.pools_path)
                    .filter(pl.col("draft_id").is_in(page_ids) & pl.col("card_id").is_in(list(user_pool.keys())))
                    .select("draft_id", "card_id").collect())
        shared_by: dict[str, list[int]] = {}
        for did, cid in ov_cards.iter_rows():
            shared_by.setdefault(did, []).append(int(cid))
        for x in rows:
            sc_v, sh_n = sim_map.get(x["draft_id"], (0.0, 0))
            x["similarity"] = sc_v
            x["shared_count"] = sh_n
            cids = sorted(set(shared_by.get(x["draft_id"], [])), key=lambda c: (-weights.get(c, 1.0), idx.card_meta[c]["name"]))
            x["shared_cards"] = [idx.card_meta[c]["name"] for c in cids[:8]]

    return dict(
        set_code=idx.set_code, total=r.height, shown=len(rows),
        n_pool_cards=sum(user_pool.values()), days=days, date_cutoff=cut_str,
        color_breakdown=color_breakdown, results=rows,
    )


# ---------------------------------------------------------------------------------------------
# Phase 3C-1: Mid-run rebuilds & 0-2 rescue pivots
# ---------------------------------------------------------------------------------------------

REBUILD_SORTS = ("default", "record", "before_after", "model_delta", "format", "colors", "changed", "date", "skill")


def _rebuild_sort_exprs(sort_by: str, sort_desc: bool) -> tuple[list[str], list[bool]]:
    d = bool(sort_desc)
    if sort_by == "record":
        return ["score", "res_losses", "win_rate", "games", "draft_id"], [d, not d, d, d, False]
    if sort_by == "before_after":
        return ["is_rescue", "wr_gain", "after_wins", "score", "draft_id"], [d, d, d, d, False]
    if sort_by == "model_delta":
        return ["model_score_delta", "b1_model_score", "score", "draft_id"], [d, d, True, False]
    if sort_by == "format":
        return ["event", "score", "date", "draft_id"], [d, True, True, False]
    if sort_by == "colors":
        return ["b0_colors", "b1_colors", "score", "draft_id"], [d, d, True, False]
    if sort_by == "changed":
        return ["cards_changed", "model_score_delta", "score", "draft_id"], [d, True, True, False]
    if sort_by == "date":
        return ["date", "score", "draft_id"], [d, True, False]
    if sort_by == "skill":
        return ["skill", "score", "date", "draft_id"], [d, True, True, False]
    return ["is_rescue", "model_score_delta", "cards_changed", "score", "draft_id"], [True, True, True, True, False]


def search_rebuilds(set_code: str, card: str | int | None = None, events: list[str] | None = None,
                    min_skill: float | None = None, rescue_only: bool = False, color_pivot_only: bool = False,
                    limit: int = DEFAULT_LIMIT, offset: int = 0, days: int | None = None,
                    min_changes: int = DEFAULT_MIN_REBUILD_CHANGES,
                    sort_by: str = "default", sort_desc: bool = True) -> dict:
    """Runs where the player meaningfully rebuilt their deck mid-run (`>= min_changes` non-basic cards changed; default 5),
    with full per-build diffs and unbiased Before → After model scores when cached."""
    if sort_by not in REBUILD_SORTS:
        raise ValueError(f"sort_by must be one of {REBUILD_SORTS}")
    with _LOCK:
        return _search_rebuilds(set_code, card, events, min_skill, rescue_only, color_pivot_only, limit, offset, days,
                                min_changes=min_changes, sort_by=sort_by, sort_desc=sort_desc)


_REBUILD_COLS = ["event", "draft_id", "build_index", "deck_key", "games", "wins", "main_colors", "splash_colors"]


def _nonbasic_diff_count(d1: dict[int, int], d2: dict[int, int]) -> int:
    """Number of non-basic card slots changed between two decks (max of cards added and cards cut)."""
    added = 0
    cut = 0
    for c in set(d1) | set(d2):
        diff = d2.get(c, 0) - d1.get(c, 0)
        if diff > 0:
            added += diff
        elif diff < 0:
            cut -= diff
    return max(added, cut)


def _bo3_registered_builds(idx: _SetIndex) -> pl.DataFrame:
    """Bo3 runs as registered decks: each match's game-1 deck; consecutive matches starting with the same deck are one build, and
    a build's games / wins are all games of its matches (sideboarded games 2-3 included).  Cached on the index."""
    if getattr(idx, "_bo3_reg", None) is not None:
        return idx._bo3_reg
    bo3_evs = [e["event"] for e in idx.events if e["bo3"]]
    empty = pl.DataFrame(schema={"event": pl.String, "draft_id": pl.String, "build_index": pl.Int16, "deck_key": pl.Int64,
                                 "games": pl.Int32, "wins": pl.Int32, "main_colors": pl.String, "splash_colors": pl.String})
    if not bo3_evs:
        idx._bo3_reg = empty
        return empty
    g = (pl.scan_parquet(idx.cdir / "games.parquet").filter(pl.col("event").is_in(bo3_evs))
         .select("event", "draft_id", "match_number", "game_number", "deck_key", "won", "main_colors", "splash_colors").collect()
         .with_columns(pl.col("won").fill_null(False)))
    per_match = (g.sort(["event", "draft_id", "match_number", "game_number"])
                 .group_by(["event", "draft_id", "match_number"], maintain_order=True)
                 .agg(pl.col("deck_key").filter(pl.col("game_number") == 1).first().alias("g1_dk"),
                      pl.col("deck_key").first().alias("any_dk"),
                      pl.col("main_colors").drop_nulls().first().alias("main_colors"),
                      pl.col("splash_colors").drop_nulls().first().alias("splash_colors"),
                      pl.len().cast(pl.Int32).alias("games"), pl.col("won").sum().cast(pl.Int32).alias("wins")))
    # If a match missed logging Game 1 (starts at Game 2 sideboarded deck), carry the run's registered Game-1 deck rather than inventing a rebuild
    per_match = per_match.sort(["event", "draft_id", "match_number"]).with_columns(
        pl.coalesce(
            pl.col("g1_dk").forward_fill().over(["event", "draft_id"]),
            pl.col("g1_dk").backward_fill().over(["event", "draft_id"]),
            pl.col("any_dk"),
        ).alias("deck_key")
    ).with_columns(
        (pl.col("deck_key") != pl.col("deck_key").shift(1).over(["event", "draft_id"])).fill_null(True)
        .cast(pl.Int16).cum_sum().over(["event", "draft_id"]).alias("build_index"))
    out = (per_match.group_by(["event", "draft_id", "build_index"], maintain_order=True)
           .agg(pl.col("deck_key").first(), pl.col("games").sum(), pl.col("wins").sum(),
                pl.col("main_colors").first(), pl.col("splash_colors").first())
           .select(_REBUILD_COLS).cast({"build_index": pl.Int16, "games": pl.Int32, "wins": pl.Int32}))
    idx._bo3_reg = out
    return out


def _search_rebuilds(set_code, card, events, min_skill, rescue_only, color_pivot_only, limit, offset, days=None,
                     min_changes: int = DEFAULT_MIN_REBUILD_CHANGES,
                     sort_by: str = "default", sort_desc: bool = True) -> dict:
    idx = _load(set_code)
    _need_games(idx)
    cid = _require_card(idx, card) if (card is not None and (not isinstance(card, str) or card.strip())) else None
    cname = idx.card_meta[cid]["name"] if cid is not None else None
    min_ch = max(1, int(min_changes)) if min_changes is not None else 1
    ms = cached_model_scores(idx.set_code)

    def _empty_analytics():
        return dict(n_multi=0, min_changes=min_ch, pct_color_pivot=0.0, n_rescues=0,
                    has_model_scores=bool(ms is not None), n_scored=0,
                    median_before_score=None, median_after_score=None, median_score_delta=None, pct_improved=None)

    # Bo3: 17lands starts a new build for every sideboarded game, so a "build" there is the deck a match STARTS with (game 1);
    # games 2-3 are sideboarding (see search_sideboarding), not a rebuild.  Bo1 builds are used as they are.
    bo3_evs = {e["event"] for e in idx.events if e["bo3"]}
    b = pl.concat([idx.builds.filter(~pl.col("event").is_in(list(bo3_evs))).select(_REBUILD_COLS),
                   _bo3_registered_builds(idx)], how="vertical_relaxed")
    if events:
        b = b.filter(pl.col("event").is_in(list(events)))
    # Keep only runs with >1 distinct deck_key
    multi = b.group_by(["event", "draft_id"]).agg(pl.col("deck_key").n_unique().alias("n_dk")).filter(pl.col("n_dk") > 1)
    if multi.height == 0:
        return dict(set_code=idx.set_code, card=cname, card_id=cid, total=0, shown=0, offset=int(offset),
                    min_changes=min_ch, analytics=_empty_analytics(), results=[])

    mb = b.join(multi.select("event", "draft_id"), on=["event", "draft_id"], how="inner")
    if cid is not None:
        dk_c = pl.scan_parquet(idx.dc_path).filter(pl.col("card_id") == cid).select("deck_key").unique().collect()
        has_c = mb.join(dk_c, on="deck_key", how="inner").select("event", "draft_id").unique()
        mb = mb.join(has_c, on=["event", "draft_id"], how="inner")

    mb = mb.sort(["event", "draft_id", "build_index"])
    # Collapse per-run: first build vs subsequent builds combined
    first_b = mb.group_by(["event", "draft_id"], maintain_order=True).agg(
        pl.col("build_index").first().alias("b0_index"),
        pl.col("deck_key").first().alias("b0_dk"),
        pl.col("games").first().alias("b0_games"),
        pl.col("wins").first().alias("b0_wins"),
        pl.col("main_colors").first().alias("b0_colors"),
        pl.col("build_index").last().alias("b1_index"),
        pl.col("deck_key").last().alias("b1_dk"),
        pl.col("main_colors").last().alias("b1_colors"),
        (pl.col("games").sum() - pl.col("games").first()).cast(pl.Int32).alias("after_games"),
        (pl.col("wins").sum() - pl.col("wins").first()).cast(pl.Int32).alias("after_wins"),
        pl.len().cast(pl.Int16).alias("n_distinct_builds"),
    ).with_columns(
        (pl.col("b0_games") - pl.col("b0_wins")).alias("b0_losses"),
        (pl.col("after_games") - pl.col("after_wins")).alias("after_losses"),
        (pl.col("b0_colors").fill_null("") != pl.col("b1_colors").fill_null("")).alias("color_pivot"),
    )

    r = idx.runs.join(first_b, on=["event", "draft_id"], how="inner")
    if min_skill is not None and min_skill > 0:
        r = r.filter(pl.col("skill").is_not_null() & (pl.col("skill") >= float(min_skill)))
    cut_dt, cut_str = _date_cutoff(idx, events, days)
    if cut_dt is not None:
        r = r.filter(pl.col("date").is_not_null() & (pl.col("date") >= cut_dt))
    if r.height == 0:
        return dict(set_code=idx.set_code, card=cname, card_id=cid, total=0, shown=0, offset=int(offset),
                    days=days, date_cutoff=cut_str, min_changes=min_ch, analytics=_empty_analytics(), results=[])

    # Compute exact non-basic cards_changed for every candidate run BEFORE filtering & analytics so minor (< min_changes) tweaks
    # are excluded consistently from totals, analytics, and results.
    basics = {c_ for c_, m in idx.card_meta.items() if m["is_basic"]}
    mb_cand = mb.join(r.select("event", "draft_id"), on=["event", "draft_id"], how="inner").sort(["event", "draft_id", "build_index"])
    cand_dks = mb_cand["deck_key"].unique().to_list()
    dc_nb = (pl.scan_parquet(idx.dc_path)
             .filter(pl.col("deck_key").is_in(cand_dks) & (~pl.col("card_id").is_in(list(basics))))
             .select("deck_key", "card_id", "n").collect())
    nb_decks: dict[int, dict[int, int]] = {}
    for k, c_, n_ in dc_nb.iter_rows():
        if n_ and n_ > 0:
            nb_decks.setdefault(int(k), {})[int(c_)] = int(n_)
    run_dks: dict[tuple[str, str], list[int]] = {}
    for ev_, did_, dk_ in mb_cand.select("event", "draft_id", "deck_key").iter_rows():
        run_dks.setdefault((ev_, did_), []).append(int(dk_))
    pair_diff_cache: dict[tuple[int, int], int] = {}

    def _pair_diff(a: int, b_: int) -> int:
        if a == b_:
            return 0
        key = (a, b_) if a < b_ else (b_, a)
        hit = pair_diff_cache.get(key)
        if hit is None:
            hit = _nonbasic_diff_count(nb_decks.get(a, {}), nb_decks.get(b_, {}))
            pair_diff_cache[key] = hit
        return hit

    ev_col, did_col, ch_col = [], [], []
    for (ev_, did_), dks in run_dks.items():
        if len(dks) < 2:
            mc_n = 0
        else:
            step_max = max(_pair_diff(dks[i], dks[i + 1]) for i in range(len(dks) - 1))
            span = _pair_diff(dks[0], dks[-1])
            mc_n = max(step_max, span)
        ev_col.append(ev_)
        did_col.append(did_)
        ch_col.append(mc_n)
    ch_df = pl.DataFrame({"event": ev_col, "draft_id": did_col, "cards_changed": ch_col},
                         schema={"event": pl.String, "draft_id": pl.String, "cards_changed": pl.Int16})
    r = r.join(ch_df, on=["event", "draft_id"], how="inner")
    if min_ch > 1:
        r = r.filter(pl.col("cards_changed") >= min_ch)
    if r.height == 0:
        return dict(set_code=idx.set_code, card=cname, card_id=cid, total=0, shown=0, offset=int(offset),
                    days=days, date_cutoff=cut_str, min_changes=min_ch, analytics=_empty_analytics(), results=[])

    # Join cached model scores for initial (b0) and final (b1) builds so Before -> After is measured without regression-to-the-mean bias
    if ms is not None:
        ms0 = ms.rename({"deck_key": "b0_dk", "model_q": "b0_model_q", "model_score": "b0_model_score"})
        ms1 = ms.rename({"deck_key": "b1_dk", "model_q": "b1_model_q", "model_score": "b1_model_score"})
        r = r.join(ms0, on=["b0_dk", "fmt"], how="left").join(ms1, on=["b1_dk", "fmt"], how="left")
        r = r.with_columns((pl.col("b1_model_score") - pl.col("b0_model_score")).alias("model_score_delta"))
    else:
        r = r.with_columns(
            pl.lit(None, dtype=pl.Float32).alias("b0_model_q"), pl.lit(None, dtype=pl.Float32).alias("b0_model_score"),
            pl.lit(None, dtype=pl.Float32).alias("b1_model_q"), pl.lit(None, dtype=pl.Float32).alias("b1_model_score"),
            pl.lit(None, dtype=pl.Float32).alias("model_score_delta"),
        )

    r = r.with_columns(
        ((pl.col("b0_losses") >= 2) & (pl.col("b0_wins") <= 1) & ((pl.col("after_wins") >= 3) | pl.col("trophy"))).alias("is_rescue"),
        (pl.col("b0_wins") / pl.col("b0_games").clip(lower_bound=1)).alias("b0_wr"),
        (pl.col("after_wins") / pl.col("after_games").clip(lower_bound=1)).alias("after_wr"),
    ).with_columns((pl.col("after_wr") - pl.col("b0_wr")).alias("wr_gain"))

    # Analytics over all qualifying rebuilt runs (before rescue/pivot toggle)
    n_multi = int(r.height)
    scored_r = r.filter(pl.col("model_score_delta").is_not_null())
    n_scored = int(scored_r.height)
    analytics = dict(
        n_multi=n_multi,
        min_changes=min_ch,
        pct_color_pivot=float(r["color_pivot"].mean()) if n_multi else 0.0,
        n_rescues=int(r["is_rescue"].sum()) if n_multi else 0,
        has_model_scores=bool(ms is not None),
        n_scored=n_scored,
        median_before_score=float(scored_r["b0_model_score"].median()) if n_scored else None,
        median_after_score=float(scored_r["b1_model_score"].median()) if n_scored else None,
        median_score_delta=float(scored_r["model_score_delta"].median()) if n_scored else None,
        pct_improved=float((scored_r["model_score_delta"] > 0).mean()) if n_scored else None,
    )

    if rescue_only:
        r = r.filter(pl.col("is_rescue"))
    if color_pivot_only:
        r = r.filter(pl.col("color_pivot"))

    s_cols, s_desc = _rebuild_sort_exprs(sort_by, sort_desc)
    r = r.sort(s_cols, descending=s_desc, nulls_last=True)
    page = r.slice(int(offset), int(limit)) if limit else r.slice(int(offset))
    if page.height == 0:
        return dict(set_code=idx.set_code, card=cname, card_id=cid, total=r.height, shown=0, offset=int(offset),
                    days=days, date_cutoff=cut_str, min_changes=min_ch, analytics=analytics, results=[])

    # Load all builds & decklists for the paged runs
    page_keys = page.select("event", "draft_id")
    pb = mb_cand.join(page_keys, on=["event", "draft_id"], how="inner").sort(["event", "draft_id", "build_index"])
    all_dks = pb["deck_key"].unique().to_list()
    dc = (pl.scan_parquet(idx.dc_path)
          .filter(pl.col("deck_key").is_in(all_dks)).select("deck_key", "card_id", "n").collect())
    decks: dict[int, dict[int, int]] = {}
    for k, c_, n_ in dc.iter_rows():
        decks.setdefault(int(k), {})[int(c_)] = int(n_)

    ms_map: dict[tuple[int, str], tuple[float, float]] = {}
    if ms is not None:
        ms_sub = ms.filter(pl.col("deck_key").is_in(all_dks))
        for k, f_, mq_, ms_ in ms_sub.select("deck_key", "fmt", "model_q", "model_score").iter_rows():
            ms_map[(int(k), str(f_))] = (float(mq_), float(ms_))

    pools: dict[str, dict[int, int]] = {}
    sealed_ids = page.filter(pl.col("fmt").is_in(["bo1", "bo3"]))["draft_id"].to_list()
    if sealed_ids and idx.has_pools:
        pdf = pl.scan_parquet(idx.pools_path).filter(pl.col("draft_id").is_in(sealed_ids)).collect()
        for d, c_, n_ in pdf.select("draft_id", "card_id", "n").iter_rows():
            pools.setdefault(d, {})[int(c_)] = int(n_)

    builds_by_run: dict[tuple[str, str], list[dict]] = {}
    for row in pb.iter_rows(named=True):
        builds_by_run.setdefault((row["event"], row["draft_id"]), []).append(row)

    results = []
    for row in page.iter_rows(named=True):
        blist = builds_by_run.get((row["event"], row["draft_id"]), [])
        hist = []
        prev_counts: dict[int, int] | None = None
        for bi in blist:
            dk = int(bi["deck_key"])
            cnt = decks.get(dk, {})
            added_c: dict[int, int] = {}
            cut_c: dict[int, int] = {}
            if prev_counts is not None:
                all_c = (set(cnt) | set(prev_counts)) - basics
                for c_ in all_c:
                    diff = cnt.get(c_, 0) - prev_counts.get(c_, 0)
                    if diff > 0:
                        added_c[c_] = diff
                    elif diff < 0:
                        cut_c[c_] = -diff
            prev_counts = cnt
            bw, bg = int(bi["wins"]), int(bi["games"])
            mc = (bi.get("main_colors") or "") + (f"+{bi['splash_colors']}" if bi.get("splash_colors") else "")
            mq_ms = ms_map.get((dk, str(row["fmt"])))
            hist.append(dict(
                build_index=int(bi["build_index"]), deck_key=dk,
                games=bg, wins=bw, losses=bg - bw, colors=mc,
                main=_card_lines(idx, cnt),
                added=_card_lines(idx, added_c),
                cut=_card_lines(idx, cut_c),
                cards_changed=max(sum(added_c.values()), sum(cut_c.values())),
                model_q=mq_ms[0] if mq_ms else None,
                model_score=mq_ms[1] if mq_ms else None,
            ))
        last_cnt = decks.get(int(row["b1_dk"]), {})
        last_main = hist[-1]["main"] if hist else []
        sideboard, note = None, ""
        if row["fmt"] in ("bo1", "bo3"):
            pool = pools.get(row["draft_id"])
            if pool:
                sb = {c_: n_ - last_cnt.get(c_, 0) for c_, n_ in pool.items() if c_ not in basics and n_ - last_cnt.get(c_, 0) > 0}
                sideboard = _card_lines(idx, sb)
            else:
                note = "No pool stored for this run."
        else:
            note = "Draft sideboards are not stored in the corpus (only the main deck)."
        date = row["date"]
        b0c = row["b0_colors"] or row["run_main_colors"] or "?"
        b1c = row["b1_colors"] or row["run_main_colors"] or "?"
        results.append(dict(
            draft_id=row["draft_id"], event=row["event"], event_label=event_label(row["event"]), fmt=row["fmt"], bo3=bool(row["bo3"]),
            wins=int(row["wins"]), losses=int(row["losses"]), games=int(row["games"]),
            match_wins=int(row["match_wins"] or 0), match_losses=int(row["match_losses"] or 0),
            incomplete_matches=int(row.get("incomplete_matches") or 0),
            record=_record(row), win_rate=float(row["win_rate"]), max_wins=int(row["max_wins"]), trophy=bool(row["trophy"]),
            score=float(row["score"]),
            date=date.isoformat(sep=" ", timespec="minutes") if date is not None else "",
            rank=row["rank"] or "", skill=None if row.get("skill") is None else float(row["skill"]),
            colors=f"{b0c} → {b1c}" if b0c != b1c else b1c,
            b0_record=f"{int(row['b0_wins'])}–{int(row['b0_losses'])}",
            after_record=f"{int(row['after_wins'])}–{int(row['after_losses'])}",
            b0_model_score=None if row.get("b0_model_score") is None else float(row["b0_model_score"]),
            b1_model_score=None if row.get("b1_model_score") is None else float(row["b1_model_score"]),
            model_score_delta=None if row.get("model_score_delta") is None else float(row["model_score_delta"]),
            color_pivot=bool(row["color_pivot"]), is_rescue=bool(row["is_rescue"]),
            wr_gain=float(row["wr_gain"]), cards_changed=int(row["cards_changed"]),
            n_builds=len(hist), build_index=hist[-1]["build_index"] if hist else 0,
            deck_key=int(row["b1_dk"]),
            n_cards=sum(last_cnt.values()), n_lands=sum(d["n"] for d in last_main if d["is_land"]),
            builds=hist, main=last_main, sideboard=sideboard, sideboard_note=note,
        ))

    return dict(
        set_code=idx.set_code, card=cname, card_id=cid,
        total=r.height, shown=len(results), offset=int(offset),
        days=days, date_cutoff=cut_str, min_changes=min_ch,
        analytics=analytics, results=results,
    )


# ---------------------------------------------------------------------------------------------
# Phase 3C-2: Real Bo3 Sideboarding Explorer (Game 1 -> Game 2/3 diffs paired with G1 opp_colors)
# ---------------------------------------------------------------------------------------------

def _get_bo3_sideboard_index(idx: _SetIndex) -> dict:
    """Lazily build and cache in-memory Bo3 Game 1 -> Game 2/3 sideboard swap tables."""
    if idx._bo3_cache is not None:
        return idx._bo3_cache
    bo3_evs = [e["event"] for e in idx.events if e["bo3"]]
    empty = dict(matches=pl.DataFrame(), swaps=pl.DataFrame(), all_postboard=pl.DataFrame(), decks={})
    if not bo3_evs:
        idx._bo3_cache = empty
        return empty
    g = (pl.scan_parquet(idx.cdir / "games.parquet")
         .filter(pl.col("event").is_in(bo3_evs))
         .select("event", "fmt", "draft_id", "match_number", "game_number", "deck_key",
                 "on_play", "won", "main_colors", "splash_colors", "opp_colors", "skill", "rank", "game_time")
         .collect())
    if g.height == 0:
        idx._bo3_cache = empty
        return empty
    g = g.with_columns(pl.col("won").fill_null(False))
    g1 = (g.filter(pl.col("game_number") == 1)
          .unique(subset=["event", "draft_id", "match_number"], keep="first")
          .select("event", "fmt", "draft_id", "match_number",
                  pl.col("deck_key").alias("dk1"), pl.col("won").alias("g1_won"),
                  pl.col("main_colors").fill_null("").alias("my_colors"),
                  pl.col("splash_colors").fill_null("").alias("my_splash"),
                  pl.col("opp_colors").fill_null("").alias("opp_colors"),
                  "skill", "rank", pl.col("game_time").alias("date")))
    # Post-board games (Game 2, Game 3): prefer the first post-board game whose deck differs from Game 1 (catches Game-3-only swaps too)
    g2 = (g.filter(pl.col("game_number") >= 2)
          .join(g1.select("event", "draft_id", "match_number", "dk1"), on=["event", "draft_id", "match_number"], how="inner")
          .with_columns((pl.col("deck_key") == pl.col("dk1")).alias("_same"))
          .sort(["event", "draft_id", "match_number", "_same", "game_number"])
          .group_by(["event", "draft_id", "match_number"], maintain_order=True)
          .agg(pl.col("deck_key").first().alias("dk2"),
               pl.col("on_play").first().alias("g2_on_play"),
               pl.col("won").sum().cast(pl.Int8).alias("post_wins"),
               pl.len().cast(pl.Int8).alias("post_games")))
    post = g1.join(g2, on=["event", "draft_id", "match_number"], how="inner").with_columns(
        (pl.col("dk1") != pl.col("dk2")).alias("boarded"),
        ((pl.col("g1_won").cast(pl.Int8) + pl.col("post_wins")) > ((1 - pl.col("g1_won").cast(pl.Int8)) + (pl.col("post_games") - pl.col("post_wins")))).alias("match_won"),
    ).with_row_index("mid")

    changed = post.filter(pl.col("boarded"))
    if changed.height == 0:
        idx._bo3_cache = dict(matches=changed, swaps=pl.DataFrame(), all_postboard=post, decks={})
        return idx._bo3_cache

    # Load non-basic deck_cards for all dk1 & dk2 involved in sideboarded matches
    dks = list(set(changed["dk1"].to_list()) | set(changed["dk2"].to_list()))
    basics = {c for c, m in idx.card_meta.items() if m["is_basic"]}
    dc = (pl.scan_parquet(idx.dc_path)
          .filter(pl.col("deck_key").is_in(dks)).select("deck_key", "card_id", "n").collect())
    decks: dict[int, dict[int, int]] = {}
    for k, c, n in dc.iter_rows():
        decks.setdefault(int(k), {})[int(c)] = int(n)

    # Compute exact spell/non-basic deltas per unique (dk1, dk2) pair (fast!)
    pair_diffs: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for dk1, dk2 in changed.select("dk1", "dk2").unique().iter_rows():
        d1 = decks.get(int(dk1), {})
        d2 = decks.get(int(dk2), {})
        diffs = []
        for c in (set(d1) | set(d2)) - basics:
            dl = d2.get(c, 0) - d1.get(c, 0)
            if dl != 0:
                diffs.append((c, dl))
        if diffs:
            pair_diffs[(int(dk1), int(dk2))] = diffs

    m_ids: list[int] = []
    c_ids: list[int] = []
    deltas: list[int] = []
    for mid, dk1, dk2 in changed.select("mid", "dk1", "dk2").iter_rows():
        diffs = pair_diffs.get((int(dk1), int(dk2)))
        if diffs:
            for c, dl in diffs:
                m_ids.append(int(mid))
                c_ids.append(c)
                deltas.append(dl)

    swaps = pl.DataFrame(
        {"mid": m_ids, "card_id": c_ids, "delta": deltas},
        schema={"mid": pl.UInt32, "card_id": pl.Int32, "delta": pl.Int8},
    ) if m_ids else pl.DataFrame(schema={"mid": pl.UInt32, "card_id": pl.Int32, "delta": pl.Int8})

    idx._bo3_cache = dict(matches=changed, swaps=swaps, all_postboard=post, decks=decks)
    return idx._bo3_cache


def search_sideboarding(set_code: str, card: str | int | None = None, events: list[str] | None = None,
                        my_colors: str | None = None, opp_colors: str | None = None,
                        on_play: bool | None = None, min_skill: float | None = None,
                        limit: int = 100, days: int | None = None,
                        pool_cards: list[str | int] | None = None) -> dict:
    """Real Bo3 sideboarding explorer from Game 1 -> Game 2/3 deck diffs paired with Game 1 `opp_colors`.

    - With `card`: shows when pilots board this card IN vs OUT, top `opp_colors` it comes in/out against,
      most common swapped cards, top-pilot (>=60%) vs overall rates, and clickable match examples.
    - Without `card`: shows the Matchup Sideboard Plan (`top_in` / `top_out`) sorted by how often strong pilots
      (`skill >= 0.60`, falling back to all pilots when strong-pilot sample is < 5) make each swap, never by win rate.
      Post-board win rates are returned strictly alongside sample size (`post_games`) and 95% Wilson CI.
    - `pool_cards`: optional list of card names/ids from the user's current pool to restrict swap plans to cards they own.
    """
    with _LOCK:
        return _search_sideboarding(set_code, card, events, my_colors, opp_colors, on_play, min_skill, limit, days,
                                    pool_cards=pool_cards)


def _match_color_filter(df: pl.DataFrame, col: str, want: str | None) -> pl.DataFrame:
    if not want:
        return df
    c = _canon_colors(want)
    if not c:
        return df
    mc = pl.col(col).fill_null("")
    cond = (mc.str.len_chars() == len(c))
    for ch in c:
        cond = cond & mc.str.contains(ch, literal=True)
    return df.filter(cond)


def _search_sideboarding(set_code, card, events, my_colors, opp_colors, on_play, min_skill, limit, days=None,
                         pool_cards: list[str | int] | None = None) -> dict:
    idx = _load(set_code)
    _need_games(idx)
    cid = _require_card(idx, card) if (card is not None and (not isinstance(card, str) or card.strip())) else None
    cname = idx.card_meta[cid]["name"] if cid is not None else None
    pool_cids: set[int] | None = None
    if pool_cards is not None:
        pool_cids = {x for x in (_resolve_in(idx, c) for c in pool_cards) if x is not None and not idx.card_meta[x]["is_basic"]}
    cache = _get_bo3_sideboard_index(idx)
    post = cache["all_postboard"]
    changed = cache["matches"]
    swaps = cache["swaps"]
    decks = cache["decks"]
    bo3_evs = [e for e in (events or []) if is_bo3_event(e)] or [e["event"] for e in idx.events if e["bo3"]]
    cut_dt, cut_str = _date_cutoff(idx, bo3_evs, days)

    def _empty():
        return dict(set_code=idx.set_code, card=cname, card_id=cid, total_postboard=0, boarded_matches=0,
                    boarded_rate=0.0, n_strong_boarded=0, sorted_by="strong", pool_filtered=bool(pool_cids is not None),
                    days=days, date_cutoff=cut_str, card_summary=None, top_in=[], top_out=[],
                    by_opp_colors=[], common_cut_for=[], common_in_for=[], matches=[])

    if post.height == 0:
        return _empty()

    # Apply event / color / play-draw / skill / date filters to both `post` and `changed`
    def _filt(df: pl.DataFrame) -> pl.DataFrame:
        if events:
            bo3_sel = [e for e in events if is_bo3_event(e)]
            if bo3_sel:
                df = df.filter(pl.col("event").is_in(bo3_sel))
        if my_colors:
            df = _match_color_filter(df, "my_colors", my_colors)
        if opp_colors:
            df = _match_color_filter(df, "opp_colors", opp_colors)
        if on_play is not None:
            df = df.filter(pl.col("g2_on_play") == bool(on_play))
        if min_skill is not None and min_skill > 0:
            df = df.filter(pl.col("skill").is_not_null() & (pl.col("skill") >= float(min_skill)))
        if cut_dt is not None:
            df = df.filter(pl.col("date").is_not_null() & (pl.col("date") >= cut_dt))
        return df

    post_f = _filt(post)
    ch_f = _filt(changed)
    n_post = int(post_f.height)
    n_boarded = int(ch_f.height)
    n_strong_boarded = int(ch_f.filter(pl.col("skill").is_not_null() & (pl.col("skill") >= 0.60)).height) if n_boarded else 0
    sorted_by = "strong" if n_strong_boarded >= 5 else "all"
    if n_post == 0 or n_boarded == 0 or swaps.height == 0:
        out = _empty()
        out["total_postboard"] = n_post
        out["boarded_matches"] = n_boarded
        out["boarded_rate"] = float(n_boarded / n_post) if n_post else 0.0
        out["n_strong_boarded"] = n_strong_boarded
        out["sorted_by"] = sorted_by
        return out

    sw_f = swaps.join(ch_f.select("mid", "my_colors", "opp_colors", "g2_on_play", "post_wins", "post_games", "match_won", "skill"),
                      on="mid", how="inner")
    sw_f = sw_f.with_columns((pl.col("skill").is_not_null() & (pl.col("skill") >= 0.60)).alias("_strong"))

    card_summary = None
    top_in: list[dict] = []
    top_out: list[dict] = []
    by_opp_colors: list[dict] = []
    common_cut_for: list[dict] = []
    common_in_for: list[dict] = []

    if cid is not None:
        # Card-specific sideboarding profile
        c_sw = sw_f.filter(pl.col("card_id") == cid)
        ins = c_sw.filter(pl.col("delta") > 0)
        outs = c_sw.filter(pl.col("delta") < 0)
        n_in, n_out = int(ins.height), int(outs.height)
        in_w = int(ins["post_wins"].sum()) if n_in else 0
        in_g = int(ins["post_games"].sum()) if n_in else 0
        out_w = int(outs["post_wins"].sum()) if n_out else 0
        out_g = int(outs["post_games"].sum()) if n_out else 0
        in_ci = _wilson_ci(in_w, in_g)
        out_ci = _wilson_ci(out_w, out_g)
        in_play = int(ins.filter(pl.col("g2_on_play") == True).height) if n_in else 0  # noqa: E712
        in_draw = int(ins.filter(pl.col("g2_on_play") == False).height) if n_in else 0  # noqa: E712
        top_in_n = int(ins.filter(pl.col("_strong")).height) if n_in else 0
        top_out_n = int(outs.filter(pl.col("_strong")).height) if n_out else 0
        card_summary = dict(
            n_in=n_in, n_out=n_out,
            in_rate=float(n_in / n_boarded), out_rate=float(n_out / n_boarded),
            top_in_count=top_in_n, top_out_count=top_out_n,
            top_in_rate=float(top_in_n / n_strong_boarded) if n_strong_boarded >= 10 else None,
            top_out_rate=float(top_out_n / n_strong_boarded) if n_strong_boarded >= 10 else None,
            in_post_games=in_g, in_post_wr=float(in_w / in_g) if in_g else None,
            in_post_lo=in_ci[0] if in_ci else None, in_post_hi=in_ci[1] if in_ci else None,
            out_post_games=out_g, out_post_wr=float(out_w / out_g) if out_g else None,
            out_post_lo=out_ci[0] if out_ci else None, out_post_hi=out_ci[1] if out_ci else None,
            in_on_play=in_play, in_on_draw=in_draw,
        )
        # Breakdown by G1 opp_colors
        if c_sw.height:
            opp_agg = (c_sw.filter(pl.col("opp_colors") != "")
                       .group_by("opp_colors")
                       .agg((pl.col("delta") > 0).sum().alias("n_in"),
                            (pl.col("delta") < 0).sum().alias("n_out"))
                       .with_columns((pl.col("n_in") - pl.col("n_out")).alias("net"),
                                     (pl.col("n_in") + pl.col("n_out")).alias("tot"))
                       .sort(["tot", "opp_colors"], descending=[True, False]).head(10))
            by_opp_colors = [dict(opp_colors=r["opp_colors"], n_in=int(r["n_in"]), n_out=int(r["n_out"]), net=int(r["net"]))
                             for r in opp_agg.iter_rows(named=True)]
        # Most common cards cut when `cid` comes IN, and brought in when `cid` goes OUT
        sw_pool = sw_f.filter(pl.col("card_id").is_in(list(pool_cids))) if pool_cids is not None else sw_f
        if n_in:
            in_mids = ins.select("mid").unique()
            cuts = (sw_pool.join(in_mids, on="mid", how="inner")
                    .filter((pl.col("card_id") != cid) & (pl.col("delta") < 0))
                    .group_by("card_id").len().sort("len", descending=True).head(6))
            common_cut_for = [dict(card_id=int(c), name=idx.card_meta[int(c)]["name"], count=int(n), share=float(n / n_in))
                              for c, n in cuts.iter_rows()]
        if n_out:
            out_mids = outs.select("mid").unique()
            adds = (sw_pool.join(out_mids, on="mid", how="inner")
                    .filter((pl.col("card_id") != cid) & (pl.col("delta") > 0))
                    .group_by("card_id").len().sort("len", descending=True).head(6))
            common_in_for = [dict(card_id=int(c), name=idx.card_meta[int(c)]["name"], count=int(n), share=float(n / n_out))
                             for c, n in adds.iter_rows()]
        ex_mids = c_sw.select("mid").unique()
        ex_df = ch_f.join(ex_mids, on="mid", how="inner")
    else:
        # Matchup Sideboard Plan: top cards boarded IN and OUT sorted by strong-pilot swap frequency (never win rate)
        sw_plan = sw_f.filter(pl.col("card_id").is_in(list(pool_cids))) if pool_cids is not None else sw_f
        sort_cols = ["strong_n", "n", "card_id"] if sorted_by == "strong" else ["n", "card_id"]
        sort_desc = [True, True, False] if sorted_by == "strong" else [True, False]

        def _pack_plan(sub: pl.DataFrame) -> list[dict]:
            agg = (sub.group_by("card_id")
                   .agg(pl.len().alias("n"),
                        pl.col("_strong").sum().cast(pl.Int32).alias("strong_n"),
                        pl.col("post_wins").sum().alias("w"),
                        pl.col("post_games").sum().alias("g"))
                   .sort(sort_cols, descending=sort_desc).head(15))
            items = []
            for r in agg.iter_rows(named=True):
                g_, w_ = int(r["g"]), int(r["w"])
                ci = _wilson_ci(w_, g_)
                sn = int(r["strong_n"])
                items.append(dict(
                    card_id=int(r["card_id"]), name=idx.card_meta[int(r["card_id"])]["name"],
                    count=int(r["n"]), rate=float(r["n"] / n_boarded),
                    strong_count=sn, strong_rate=float(sn / n_strong_boarded) if n_strong_boarded else None,
                    post_games=g_, post_wr=float(w_ / g_) if g_ else None,
                    post_wr_lo=ci[0] if ci else None, post_wr_hi=ci[1] if ci else None,
                ))
            return items

        top_in = _pack_plan(sw_plan.filter(pl.col("delta") > 0))
        top_out = _pack_plan(sw_plan.filter(pl.col("delta") < 0))
        if pool_cids is not None:
            ex_mids = sw_plan.select("mid").unique()
            ex_df = ch_f.join(ex_mids, on="mid", how="inner")
        else:
            ex_df = ch_f

    # Concrete match examples (strongest pilots & newest first; not sorted by win outcome alone)
    ex_df = ex_df.sort(["skill", "match_won", "date", "draft_id"], descending=[True, True, True, False], nulls_last=True)
    ex_page = ex_df.head(int(limit)) if limit else ex_df

    # Hydrate match examples with in_cards / out_cards / main (G2) / sideboard (for Sealed)
    ex_mids_list = ex_page["mid"].to_list()
    ex_sw = swaps.filter(pl.col("mid").is_in(ex_mids_list))
    sw_by_mid: dict[int, tuple[dict[int, int], dict[int, int]]] = {}
    for mid, c_, dl in ex_sw.iter_rows():
        inc, outc = sw_by_mid.setdefault(int(mid), ({}, {}))
        if dl > 0:
            inc[int(c_)] = int(dl)
        else:
            outc[int(c_)] = int(-dl)

    pools: dict[str, dict[int, int]] = {}
    sealed_ids = ex_page.filter(pl.col("fmt") == "bo3")["draft_id"].unique().to_list()
    if sealed_ids and idx.has_pools:
        pdf = pl.scan_parquet(idx.pools_path).filter(pl.col("draft_id").is_in(sealed_ids)).collect()
        for d, c_, n_ in pdf.select("draft_id", "card_id", "n").iter_rows():
            pools.setdefault(d, {})[int(c_)] = int(n_)
    basics = {c for c, m in idx.card_meta.items() if m["is_basic"]}

    matches = []
    for row in ex_page.iter_rows(named=True):
        mid = int(row["mid"])
        inc, outc = sw_by_mid.get(mid, ({}, {}))
        d2 = decks.get(int(row["dk2"]), {})
        main_lines = _card_lines(idx, d2)
        sideboard, note = None, ""
        if row["fmt"] == "bo3":
            pool = pools.get(row["draft_id"])
            if pool:
                sb = {c_: n_ - d2.get(c_, 0) for c_, n_ in pool.items() if c_ not in basics and n_ - d2.get(c_, 0) > 0}
                sideboard = _card_lines(idx, sb)
            else:
                note = "No pool stored for this run."
        else:
            note = "Draft sideboards are not stored in the corpus (only the main deck)."
        date = row["date"]
        mc = (row["my_colors"] or "") + (f"+{row['my_splash']}" if row.get("my_splash") else "")
        pw, pg = int(row["post_wins"]), int(row["post_games"])
        g1w = bool(row["g1_won"])
        mw = int(g1w) + pw
        ml = (0 if g1w else 1) + (pg - pw)
        matches.append(dict(
            draft_id=row["draft_id"], event=row["event"], event_label=event_label(row["event"]), fmt=row["fmt"], bo3=True,
            match_number=int(row["match_number"]),
            record=f"{'W' if row['match_won'] else 'L'} ({mw}–{ml})",
            match_won=bool(row["match_won"]), g1_won=g1w, g2_on_play=bool(row["g2_on_play"]),
            colors=mc, opp_colors=row["opp_colors"] or "?",
            date=date.isoformat(sep=" ", timespec="minutes") if date is not None else "",
            rank=row["rank"] or "", skill=None if row.get("skill") is None else float(row["skill"]),
            in_cards=_card_lines(idx, inc), out_cards=_card_lines(idx, outc),
            n_swapped=sum(inc.values()),
            n_cards=sum(d2.values()), n_lands=sum(d["n"] for d in main_lines if d["is_land"]),
            main=main_lines, sideboard=sideboard, sideboard_note=note,
        ))

    return dict(
        set_code=idx.set_code, card=cname, card_id=cid,
        total_postboard=n_post, boarded_matches=n_boarded,
        boarded_rate=float(n_boarded / n_post) if n_post else 0.0,
        n_strong_boarded=n_strong_boarded, sorted_by=sorted_by,
        pool_filtered=bool(pool_cids is not None),
        days=days, date_cutoff=cut_str,
        card_summary=card_summary, top_in=top_in, top_out=top_out,
        by_opp_colors=by_opp_colors, common_cut_for=common_cut_for, common_in_for=common_in_for,
        matches=matches,
    )


# ---------------------------------------------------------------------------------------------
# hydration & Arena export
# ---------------------------------------------------------------------------------------------

def _card_lines(idx: _SetIndex, counts: dict[int, int]) -> list[dict]:
    if not counts:
        return []
    out = []
    for cid, n in counts.items():
        m = idx.card_meta.get(int(cid))
        if m is None:
            continue
        out.append(dict(card_id=m["card_id"], name=m["name"], n=int(n),
                        is_land=m["is_land"], is_basic=m["is_basic"], is_creature=m["is_creature"],
                        mana_value=m["mana_value"], colors=m["colors"], types=m["types"], rarity=m["rarity"]))
    # spells by mana value then name, lands last (basics after the other lands)
    out.sort(key=lambda d: (d["is_land"], d["is_basic"], d["mana_value"] if not d["is_land"] else 0, d["name"].lower()))
    return out


def _result_rows(idx: _SetIndex, page: pl.DataFrame) -> list[dict]:
    if page.height == 0:
        return []
    keys = page["deck_key"].unique().to_list()
    dc = (pl.scan_parquet(idx.dc_path).filter(pl.col("deck_key").is_in(keys))
          .select("deck_key", "card_id", "n").collect())
    decks: dict[int, dict[int, int]] = {}
    for k, c, n in dc.iter_rows():
        decks.setdefault(int(k), {})[int(c)] = int(n)
    pools: dict[str, dict[int, int]] = {}
    exact: dict[str, bool] = {}
    pooled = ["bo1", "bo3", "draft"] if idx.provisional else ["bo1", "bo3"]          # a pick-file set: the drafted pool is the "pool"
    sealed_ids = page.filter(pl.col("fmt").is_in(pooled))["draft_id"].to_list()
    if sealed_ids and idx.has_pools:
        try:
            pdf = (pl.scan_parquet(idx.pools_path).filter(pl.col("draft_id").is_in(sealed_ids)).collect())
            for d, c, n in pdf.select("draft_id", "card_id", "n").iter_rows():
                pools.setdefault(d, {})[int(c)] = int(n)
            ap = idx.cdir / "pool_audit.parquet"
            if ap.exists():
                adf = pl.scan_parquet(ap).filter(pl.col("draft_id").is_in(sealed_ids)).collect()
                exact = {d: bool(v) for d, v in adf.select("draft_id", "invariant").iter_rows()}
        except Exception:  # noqa: BLE001 - the decklist is still useful without the sideboard
            log.warning("could not read the Sealed pools of %s", idx.set_code, exc_info=True)
    basics = {c for c, m in idx.card_meta.items() if m["is_basic"]}
    out = []
    for row in page.iter_rows(named=True):
        main = decks.get(int(row["deck_key"]), {})
        sideboard, note = None, ""
        if row["fmt"] in pooled:
            pool = pools.get(row["draft_id"])
            if pool:
                sb = {c: n - main.get(c, 0) for c, n in pool.items() if c not in basics and n - main.get(c, 0) > 0}
                sideboard = _card_lines(idx, sb)
                if not exact.get(row["draft_id"], True):
                    note = "Sideboard approximate: the pool 17lands logged changed between games."
            else:
                note = "No pool stored for this run."
        else:
            note = "Draft sideboards are not stored in the corpus (only the main deck)."
        main_lines = _card_lines(idx, main)
        date = row["date"]
        out.append(dict(
            draft_id=row["draft_id"], event=row["event"], event_label=event_label(row["event"]), fmt=row["fmt"], bo3=bool(row["bo3"]),
            drafted_pool=bool(idx.provisional),
            wins=int(row["wins"]), losses=int(row["losses"]), games=int(row["games"]),
            match_wins=int(row["match_wins"] or 0), match_losses=int(row["match_losses"] or 0),
            incomplete_matches=int(row.get("incomplete_matches") or 0),
            record=_record(row), win_rate=float(row["win_rate"]), max_wins=int(row["max_wins"]), trophy=bool(row["trophy"]),
            score=float(row["score"]),
            date=date.isoformat(sep=" ", timespec="minutes") if date is not None else "",
            rank=row["rank"] or "",
            skill=None if row.get("skill") is None else float(row["skill"]),
            colors=(row.get("main_colors") or row.get("run_main_colors") or "")
                   + (f"+{row['splash_colors']}" if row.get("splash_colors") else ""),
            n_builds=int(row["n_builds"] or 1), build_index=int(row["build_index"]), deck_key=int(row["deck_key"]),
            copies=int(row["copies"]), games_with_card=int(row["games_with_card"]), wins_with_card=int(row["wins_with_card"]),
            build_games=int(row["build_games"]), build_wins=int(row["build_wins"]),
            n_cards=sum(main.values()), n_lands=sum(d["n"] for d in main_lines if d["is_land"]),
            main=main_lines, sideboard=sideboard, sideboard_note=note,
            model_q=None if row.get("model_q") is None else float(row["model_q"]),
            model_score=None if row.get("model_score") is None else float(row["model_score"]),
        ))
    return out


def _record(row: dict) -> str:
    if row["bo3"]:
        inc = int(row.get("incomplete_matches") or 0)
        suffix = f", {inc} inc." if inc else ""
        return f"{int(row['match_wins'] or 0)}–{int(row['match_losses'] or 0)} ({int(row['wins'])}–{int(row['losses'])} games{suffix})"
    return f"{int(row['wins'])}–{int(row['losses'])}"


def deck_text(result: dict, set_code: str | None = None) -> str:
    """Arena-importable text of one search result (main deck, plus the sideboard when it is stored)."""
    from sealed_lab.deck import arena

    main = {d["name"]: d["n"] for d in result.get("main") or []}
    side = {d["name"]: d["n"] for d in (result.get("sideboard") or [])}
    text = arena.format_arena(main, side or None, set_code)
    if set_code and "?" in text:
        text = _fix_names(text, set_code)
    return text


def _fix_names(text: str, set_code: str) -> str:
    """Arena spelling for names 17lands wrote with '?' (from the corpus card table's `front_name`)."""
    try:
        cards = _load(set_code).cards
    except Exception:  # noqa: BLE001
        return text
    fix = {n.split(" // ")[0].strip(): f for n, f in zip(cards["name"].to_list(), cards["front_name"].to_list())
           if n and "?" in n and f and "?" not in f}
    out = []
    for line in text.split("\n"):
        head, _, nm = line.partition(" ")
        out.append(f"{head} {fix[nm]}" if head.isdigit() and nm in fix else line)
    return "\n".join(out)
