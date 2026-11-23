"""Card deck search: the real decks of a set's corpus that played a given card, with their results.

Read-only over `sets/<SET>/corpus/*.parquet` (games, builds, deck_cards, cards, pools, pool_audit).

One result is one *run* (a 17lands `draft_id`: one draft or Sealed event of one player), not one game: the record of a deck is the
record of its run.  A run "contains" the card when any of its builds (17lands `build_index`: the deck changes between games after a
rebuild or Bo3 sideboarding) has it in the main deck; the decklist shown is the build with the card that played the most games, and
the result also says in how many of the run's games the card was in the deck.

What the corpus stores:
* main decks of every event (Sealed and draft), from 17lands' `deck_*` columns;
* the Sealed pool (deck + sideboard, non-basic cards) per run, so a Sealed deck's sideboard is pool - main deck;
* NOT the sideboard of draft runs (ingest keeps 17lands' `sideboard_*` columns only for Sealed), so draft results have no sideboard.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from sealed_lab.config import corpus_dir, event_format

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
SORTS = ("result", "win_rate", "newest", "oldest")
DEFAULT_MIN_GAMES = 3
DEFAULT_LIMIT = 200
DEFAULT_CARD_SHARE = 0.5
CAP_SHARE = 0.005          # an event's "max wins" is the highest win count that at least 0.5% of its runs reach

_TABLES = ("games", "builds", "deck_cards", "cards", "pools", "pool_audit")
_LOCK = threading.RLock()
_CACHE: dict[str, tuple[tuple, "_SetIndex"]] = {}
_CACHE_KEEP = 3


def is_bo3_event(event: str) -> bool:
    """Bo3 events count matches (17lands Traditional events)."""
    return "trad" in (event or "").lower()


def event_label(event: str) -> str:
    if event in EVENT_LABELS:
        return EVENT_LABELS[event]
    return f"{event} ({'Bo3' if is_bo3_event(event) else 'Bo1'})"


def _event_rank(event: str) -> tuple[int, str]:
    return (EVENT_ORDER.index(event) if event in EVENT_ORDER else len(EVENT_ORDER), event)


@dataclass
class _SetIndex:
    set_code: str
    cdir: Path
    cards: pl.DataFrame          # card_id, name, front_name, is_basic, is_land, is_creature, mana_value, colors, types, rarity
    runs: pl.DataFrame           # one row per (event, draft_id): see _build_runs
    builds: pl.DataFrame         # event, draft_id, build_index, deck_key, games, wins
    events: list[dict]
    by_name: dict[str, int]      # lower-cased name / front name -> card_id
    names: list[str]             # non-basic card names, sorted (autocomplete)
    has_pools: bool


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


def _build_runs(games: pl.DataFrame, builds: pl.DataFrame) -> tuple[pl.DataFrame, list[dict]]:
    """Per run: games, wins, losses, matches (Bo3), date, rank, colours; and the per-event summary (label, max wins, counts)."""
    g = games.with_columns(pl.col("won").fill_null(False))
    run_g = (g.group_by(["event", "draft_id"])
             .agg(pl.len().cast(pl.Int32).alias("games"), pl.col("won").sum().cast(pl.Int32).alias("wins"),
                  pl.col("game_time").min().alias("first_game"), pl.col("game_time").max().alias("last_game"),
                  pl.col("draft_time").min().alias("draft_time"),
                  pl.col("rank").drop_nulls().first().alias("rank"),
                  pl.col("main_colors").mode().first().alias("main_colors"),
                  pl.col("splash_colors").mode().first().alias("splash_colors"),
                  pl.col("fmt").first().alias("fmt")))
    run_g = run_g.with_columns((pl.col("games") - pl.col("wins")).alias("losses"))
    # matches: a Bo3 match is won with more game wins than losses (a 1-1 match cut short counts as lost: 17lands logged no win)
    m = (g.group_by(["event", "draft_id", "match_number"])
         .agg(pl.col("won").sum().alias("gw"), (~pl.col("won")).sum().alias("gl"))
         .with_columns((pl.col("gw") > pl.col("gl")).alias("mw")))
    run_m = m.group_by(["event", "draft_id"]).agg(pl.col("mw").sum().cast(pl.Int32).alias("match_wins"),
                                                    (~pl.col("mw")).sum().cast(pl.Int32).alias("match_losses"))
    nb = builds.group_by(["event", "draft_id"]).agg(pl.len().cast(pl.Int16).alias("n_builds"))
    runs = run_g.join(run_m, on=["event", "draft_id"], how="left").join(nb, on=["event", "draft_id"], how="left")
    runs = runs.with_columns(pl.col("event").fill_null("").str.to_lowercase().str.contains("trad", literal=True).alias("bo3"))  # = is_bo3_event
    runs = runs.with_columns(
        pl.when(pl.col("bo3")).then(pl.col("match_wins")).otherwise(pl.col("wins")).alias("res_wins"),
        pl.when(pl.col("bo3")).then(pl.col("match_losses")).otherwise(pl.col("losses")).alias("res_losses"),
        pl.coalesce(pl.col("first_game"), pl.col("draft_time")).alias("date"),
        (pl.col("wins") / pl.col("games")).alias("win_rate"),
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
    return runs, events


def _load(set_code: str) -> _SetIndex:
    sc = set_code.upper()
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
        games = pl.read_parquet(cdir / "games.parquet", columns=["event", "fmt", "draft_id", "match_number", "game_time", "draft_time",
                                                                 "rank", "main_colors", "splash_colors", "won"])
        builds = pl.read_parquet(cdir / "builds.parquet", columns=["event", "draft_id", "build_index", "deck_key", "games", "wins"])
        runs, events = _build_runs(games, builds)
        by_name: dict[str, int] = {}
        names = []
        for cid, nm, fr, basic in cards.select("card_id", "name", "front_name", "is_basic").iter_rows():
            if not nm:
                continue
            for k in (nm, fr, nm.split(" // ")[0].strip()):
                if k:
                    by_name.setdefault(k.strip().lower(), int(cid))
            if not basic:
                names.append(nm)
        has_pools = (cdir / "pools.parquet").exists()
        idx = _SetIndex(sc, cdir, cards, runs, builds, events, by_name, sorted(set(names), key=str.lower), has_pools)
        _CACHE[sc] = (stamp, idx)
        while len(_CACHE) > _CACHE_KEEP:
            _CACHE.pop(next(iter(_CACHE)))
        return idx


def clear_cache() -> None:
    with _LOCK:
        _CACHE.clear()


# ---------------------------------------------------------------------------------------------
# public
# ---------------------------------------------------------------------------------------------

def card_names(set_code: str) -> list[str]:
    """The set's non-basic card names (corpus card table), sorted: the autocomplete list."""
    return list(_load(set_code).names)


def events(set_code: str) -> list[dict]:
    """[{event, label, fmt, bo3, max_wins, n_runs, n_games}] for the events in the set's corpus (Premier, Trad, ..., Sealed, TradSealed)."""
    return [dict(e) for e in _load(set_code).events]


def resolve_card(set_code: str, name: str) -> int | None:
    idx = _load(set_code)
    key = (name or "").strip().lower()
    if not key:
        return None
    if key in idx.by_name:
        return idx.by_name[key]
    hits = [n for n in idx.names if n.lower().startswith(key)]
    if len(hits) == 1:
        return idx.by_name[hits[0].lower()]
    return None


def _sort_exprs(sort: str):
    """(columns, descending) for a sort; `draft_id` last so that ties keep one order."""
    if sort == "newest":
        return ["date", "score", "res_losses", "draft_id"], [True, True, False, False]
    if sort == "oldest":
        return ["date", "score", "res_losses", "draft_id"], [False, True, False, False]
    if sort == "win_rate":
        return ["win_rate", "games", "date", "draft_id"], [True, True, True, False]
    # best result: wins relative to the event's max (7-x Premier = 3-x Traditional Draft), fewer losses, game win rate, newest
    return ["score", "res_losses", "win_rate", "games", "date", "draft_id"], [True, False, True, True, True, False]


def search(set_code: str, card: str | int, events: list[str] | None = None, sort: str = "result",
           min_games: int = DEFAULT_MIN_GAMES, limit: int = DEFAULT_LIMIT, offset: int = 0,
           min_card_share: float = DEFAULT_CARD_SHARE) -> dict:
    """The runs whose deck played `card` (a name or a corpus card_id), filtered and sorted.

    events: 17lands event types to keep (None/[] = all).  sort: "result" (default: wins as a share of the event's max wins, then fewer
    losses, game win rate, date), "win_rate" (game win rate), "newest", "oldest".  min_games: runs with fewer games are left out
    (so 1-0 runs do not flood the list).  min_card_share: keep a run only if the card was in its deck for at least this share of
    the run's games (default 0.5: a card sided in for one game of a Bo3 run does not count that run's record); 0 keeps all.  Returns {set_code, card, card_id, total, total_all, shown, offset, by_event {event: n},
    results [...]} where each result is a dict (see `_result_rows`)."""
    if sort not in SORTS:
        raise ValueError(f"sort must be one of {SORTS}")
    with _LOCK:              # one search at a time (the GUI's worker threads share the cache; polars frames are built here)
        return _search(set_code, card, events, sort, min_games, limit, offset, min_card_share)


def _search(set_code, card, events, sort, min_games, limit, offset, min_card_share) -> dict:
    idx = _load(set_code)
    cid = int(card) if isinstance(card, int) else resolve_card(set_code, str(card))
    if cid is None:
        raise ValueError(f"no card named {card!r} in {idx.set_code}")
    crow = idx.cards.filter(pl.col("card_id") == cid)
    cname = crow["name"][0] if crow.height else str(card)
    dk = (pl.scan_parquet(idx.cdir / "deck_cards.parquet").filter(pl.col("card_id") == cid)
          .select("deck_key", pl.col("n").alias("copies")).collect())
    b = idx.builds.join(dk, on="deck_key", how="inner")
    # per run: the build with the card that played most games (then the earliest build); games with the card
    b = b.sort(["event", "draft_id", "games", "build_index"], descending=[False, False, True, False])
    per_run = b.group_by(["event", "draft_id"], maintain_order=True).agg(
        pl.col("deck_key").first(), pl.col("build_index").first(), pl.col("copies").first(),
        pl.col("games").first().alias("build_games"), pl.col("wins").first().alias("build_wins"),
        pl.col("games").sum().cast(pl.Int32).alias("games_with_card"), pl.col("wins").sum().cast(pl.Int32).alias("wins_with_card"),
        pl.col("copies").max().alias("max_copies"))
    r = idx.runs.join(per_run, on=["event", "draft_id"], how="inner")
    by_event_all = {e: int(n) for e, n in r.group_by("event").len().iter_rows()}
    total_all = r.height
    if events:
        r = r.filter(pl.col("event").is_in(list(events)))
    if min_games and min_games > 1:
        r = r.filter(pl.col("games") >= int(min_games))
    if min_card_share and min_card_share > 0:
        r = r.filter(pl.col("games_with_card") >= float(min_card_share) * pl.col("games"))
    by_event = {e: int(n) for e, n in r.group_by("event").len().iter_rows()}
    cols, desc = _sort_exprs(sort)
    r = r.sort(cols, descending=desc, nulls_last=True)
    page = r.slice(int(offset), int(limit)) if limit else r.slice(int(offset))
    return dict(set_code=idx.set_code, card=cname, card_id=cid, total=r.height, total_all=total_all, shown=page.height,
                offset=int(offset), by_event=by_event, by_event_all=by_event_all, sort=sort, min_games=int(min_games),
                min_card_share=float(min_card_share or 0.0),
                results=_result_rows(idx, page))


def _card_lines(idx: _SetIndex, counts: dict[int, int]) -> list[dict]:
    if not counts:
        return []
    c = idx.cards.filter(pl.col("card_id").is_in(list(counts)))
    out = []
    for row in c.iter_rows(named=True):
        out.append(dict(card_id=int(row["card_id"]), name=row["name"], n=int(counts[int(row["card_id"])]),
                        is_land=bool(row["is_land"]), is_basic=bool(row["is_basic"]), is_creature=bool(row["is_creature"]),
                        mana_value=float(row["mana_value"] or 0.0), colors=row["colors"] or "", types=row["types"] or "",
                        rarity=row["rarity"] or ""))
    # spells by mana value then name, lands last (basics after the other lands)
    out.sort(key=lambda d: (d["is_land"], d["is_basic"], d["mana_value"] if not d["is_land"] else 0, d["name"].lower()))
    return out


def _result_rows(idx: _SetIndex, page: pl.DataFrame) -> list[dict]:
    if page.height == 0:
        return []
    keys = page["deck_key"].unique().to_list()
    dc = (pl.scan_parquet(idx.cdir / "deck_cards.parquet").filter(pl.col("deck_key").is_in(keys))
          .select("deck_key", "card_id", "n").collect())
    decks: dict[int, dict[int, int]] = {}
    for k, c, n in dc.iter_rows():
        decks.setdefault(int(k), {})[int(c)] = int(n)
    pools: dict[str, dict[int, int]] = {}
    exact: dict[str, bool] = {}
    sealed_ids = page.filter(pl.col("fmt").is_in(["bo1", "bo3"]))["draft_id"].to_list()
    if sealed_ids and idx.has_pools:
        try:
            pdf = (pl.scan_parquet(idx.cdir / "pools.parquet").filter(pl.col("draft_id").is_in(sealed_ids)).collect())
            for d, c, n in pdf.select("draft_id", "card_id", "n").iter_rows():
                pools.setdefault(d, {})[int(c)] = int(n)
            ap = idx.cdir / "pool_audit.parquet"
            if ap.exists():
                adf = pl.scan_parquet(ap).filter(pl.col("draft_id").is_in(sealed_ids)).collect()
                exact = {d: bool(v) for d, v in adf.select("draft_id", "invariant").iter_rows()}
        except Exception:  # noqa: BLE001 - the decklist is still useful without the sideboard
            log.warning("could not read the Sealed pools of %s", idx.set_code, exc_info=True)
    basics = set(idx.cards.filter(pl.col("is_basic"))["card_id"].to_list())
    out = []
    for row in page.iter_rows(named=True):
        main = decks.get(int(row["deck_key"]), {})
        sideboard, note = None, ""
        if row["fmt"] in ("bo1", "bo3"):
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
            wins=int(row["wins"]), losses=int(row["losses"]), games=int(row["games"]),
            match_wins=int(row["match_wins"] or 0), match_losses=int(row["match_losses"] or 0),
            record=_record(row), win_rate=float(row["win_rate"]), max_wins=int(row["max_wins"]), trophy=bool(row["trophy"]),
            score=float(row["score"]),
            date=date.isoformat(sep=" ", timespec="minutes") if date is not None else "",
            rank=row["rank"] or "", colors=(row["main_colors"] or "") + (f"+{row['splash_colors']}" if row["splash_colors"] else ""),
            n_builds=int(row["n_builds"] or 1), build_index=int(row["build_index"]), deck_key=int(row["deck_key"]),
            copies=int(row["copies"]), games_with_card=int(row["games_with_card"]), wins_with_card=int(row["wins_with_card"]),
            build_games=int(row["build_games"]), build_wins=int(row["build_wins"]),
            n_cards=sum(main.values()), n_lands=sum(d["n"] for d in main_lines if d["is_land"]),
            main=main_lines, sideboard=sideboard, sideboard_note=note,
        ))
    return out


def _record(row: dict) -> str:
    if row["bo3"]:
        return f"{int(row['match_wins'] or 0)}–{int(row['match_losses'] or 0)} ({int(row['wins'])}–{int(row['losses'])} games)"
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
