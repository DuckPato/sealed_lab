"""A provisional model from 17Lands pick files, for a set that has no game data yet (docs/pick_model/RULE.md, RESULT.md).

A new set often has only 17Lands pick files (`draft_data_public.<SET>.PremierDraft.csv.gz`) for some days. Each drafted run's deck is
rebuilt from them (`pick_maindeck_rate >= 0.5`, basics added), its record (`event_match_wins`, `event_match_losses`) becomes its
games, and those rows form the "draft" stage of the usual model chain. Its card values, times the mean lambda of the 5 newest sets with
game data, are the card values of both Sealed formats; every other term (deck-structure coefficients, intercept, skill, mulligans) is
the mean over those sets (the data file `data/provisional_transfer.json`, made by `tools/make_provisional_transfer.py`).

The functions of the tested method (`read_pick_file`, `build_runs`, `rebuild_deck`, `runs_to_games`, `phi_for_decks`, `draft_chain` ...)
live here and `scripts/pick_model_check.py` imports them, so the test and the app cannot drift apart. The differences, by design: the
app uses ALL runs of the set's PremierDraft pick file(s) (the test used the first 12 days) and ignores TradDraft / PickTwo pick files
(their records count Bo3 matches / another scale).

`calculate(set_code)` writes an ordinary model directory (`SetModel.load` reads it; the Deck builder works unchanged). The model's
meta.json carries a `"provisional"` entry: the labels of the Sets tab, Deck builder and Report read it, and a normal calculation
(which replaces the whole directory) drops it.
"""
from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl

from sealed_lab import config
from sealed_lab.types import Cancelled

log = logging.getLogger(__name__)
Progress = Callable[[float, str], None]

EVENT = "PremierDraft"             # the only pick-file event used (its records count Bo1 games)
MAIN_RATE = 0.5                    # a pick is in the main deck when pick_maindeck_rate >= 0.5
DECK_SIZE = 40
MIN_NONBASIC, MAX_NONBASIC = 20, 26
PICK_COLS = ["draft_id", "draft_time", "event_match_wins", "event_match_losses", "pick", "pick_maindeck_rate",
             "user_game_win_rate_bucket", "rank"]
BASIC_NAMES = {"W": "Plains", "U": "Island", "B": "Swamp", "R": "Mountain", "G": "Forest"}
DROP_REASONS = ("no_games", "too_few_nonbasic", "too_many_nonbasic", "no_main_color")
FORMATS = ["bo3", "bo1"]           # the model's format order (pipeline.FMT_ORDER)
RESULT_DOC = "docs/pick_model/RESULT.md"
TRANSFER_PATH = Path(__file__).resolve().parent / "data" / "provisional_transfer.json"
TRANSFER_VERSION = 1
TRANSFER_EXACT = ("intercept", "on_play", "mull1", "mull2p", "omull1", "omull2p", "skill_lin", "skill_na", "game2plus")
TRANSFER_PREFIX = ("skill_bin:", "opp:")
N_TRANSFER_SETS = 5

BANNER = ("Provisional model built from 17Lands pick data only (no game results yet): card values are rougher and win rates read high "
          "(by up to about 4-5 points in testing). It is replaced automatically when game data arrives.")
UNAVAILABLE = ("Not available yet: this set has only pick data. The Deck builder works with a provisional model.")
CALC_TIP = "Provisional model from pick data (no game data yet)"
DECKS_BANNER = ("Draft decks rebuilt from 17Lands pick data (Premier Draft, Bo1 records). No game-by-game data yet: Rebuilds, "
                "sideboarding and similar Sealed pools are unavailable. Model scores come from the provisional model.")
DECKS_UNSUPPORTED = {
    "similar": "Similar pools is not available for this set yet: it compares Sealed pools, and a pick-data set has only drafted decks.",
    "rebuilds": "Rebuilds are not available for this set yet: pick data holds one deck per run, with no game-by-game builds.",
    "sideboard": "Bo3 sideboarding is not available for this set yet: pick data holds Bo1 records only, with no game-by-game decks.",
}
DECKS_NO_TABLES = ("The draft decks of this set cannot be rebuilt here: the pick file is not available and no ready-made deck table "
                   "came with the set. Calculate the set again from its pick file.")
PICK_DECKS = "pick_decks.parquet"             # next to the model: one row per rebuilt run (see build_pick_decks)
PICK_DECK_CARDS = "pick_deck_cards.parquet"   # deck_key, card_id, n: the rebuilt main decks (basics included)
PICK_POOLS = "pick_pools.parquet"             # draft_id, card_id, n: every non-basic pick of the rebuilt runs (the drafted pool)
PICK_DECKS_META = "pick_decks.json"
PICK_DECKS_VERSION = 1


class ProvisionalError(ValueError):
    """The provisional model cannot be calculated (message in plain language)."""


class TransferMismatch(ProvisionalError):
    """The shipped transfer terms do not fit the running app's model features."""


# ------------------------------------------------------------------------------------------------------ which sets
def pick_files(set_code: str) -> list:
    """Registered PremierDraft pick files (`kind == 'draft'`) of a set, FileInfo objects, file order by path."""
    from sealed_lab.ingest import registry

    return sorted((f for f in registry.files(set_code.upper(), include_shared=False) if f.kind == "draft" and f.event == EVENT),
                  key=lambda f: f.path)


def applies(set_code: str) -> bool:
    """True when the set has at least one PremierDraft pick file registered and NO game file: the one case the provisional model is for."""
    from sealed_lab.ingest import registry

    fs = registry.files(set_code.upper(), include_shared=False)
    return any(f.kind == "draft" and f.event == EVENT for f in fs) and not any(f.kind == "game" for f in fs)


def is_provisional_meta(meta: dict | None) -> bool:
    return bool(meta) and bool(meta.get("provisional"))


def is_provisional(set_code: str) -> bool:
    """The set's current model is a provisional one (its meta.json says so)."""
    try:
        meta = json.loads((config.model_dir(set_code.upper()) / "meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return is_provisional_meta(meta)


def input_files(files) -> list:
    """The registered files a provisional model is computed from: the PremierDraft pick files and the card metadata files."""
    return [f for f in files if (f.kind == "draft" and f.event == EVENT) or f.kind in ("cards", "scryfall")]


def input_signature(files) -> list[str]:
    """['kind:fingerprint', ...] of `input_files` (stored in the model's meta as `source_files`)."""
    return sorted(f"{f.kind}:{f.fingerprint}" for f in input_files(files))


def clear_leftovers(set_code: str) -> None:
    """Remove what only a provisional model leaves in the set's folders (called once a normal calculation has replaced it)."""
    for p in (config.cache_dir(set_code.upper()) / "phi_provisional.npz",):
        try:
            p.unlink()
        except OSError:
            pass


# ------------------------------------------------------------------------------------------------------ pick file
def read_pick_file(path: Path) -> pl.DataFrame:
    """The needed columns of one raw PremierDraft pick file: draft_id, draft_time (datetime us), wins, losses, skill, rank, pick, rate.

    Numbers are read as float64 (some 17Lands files write "0.0" for counts); empty cells are null."""
    import pyarrow as pa
    import pyarrow.csv as pacsv

    from sealed_lab.ingest import detect

    hdr = detect.read_head_rows(Path(path), 1)[0]
    cols = [c for c in PICK_COLS if c in hdr]
    missing = [c for c in PICK_COLS if c not in hdr and c != "rank"]
    if missing:
        raise ValueError(f"{path}: columns missing: {missing}")
    strs = {"draft_id", "draft_time", "pick", "rank"}
    types = {c: (pa.string() if c in strs else pa.float64()) for c in cols}
    with detect.open_binary(Path(path)) as stream:
        tb = pacsv.read_csv(stream, read_options=pacsv.ReadOptions(block_size=24 << 20),
                            convert_options=pacsv.ConvertOptions(include_columns=cols, column_types=types))
    df = pl.from_arrow(tb)
    if "rank" not in df.columns:
        df = df.with_columns(pl.lit(None, dtype=pl.String).alias("rank"))
    return (df.rename({"event_match_wins": "wins", "event_match_losses": "losses", "user_game_win_rate_bucket": "skill",
                       "pick_maindeck_rate": "rate"})
            .with_columns(pl.col("draft_time").str.to_datetime(strict=False, time_unit="us")))


def window_filter(df: pl.DataFrame, days: int) -> tuple[pl.DataFrame, object]:
    """Rows whose draft_time is in the first `days` days after the earliest draft_time of `df` (t0 <= draft_time < t0 + days). (rows, t0)."""
    t0 = df["draft_time"].min()
    t1 = t0 + timedelta(days=days)
    return df.filter((pl.col("draft_time") >= t0) & (pl.col("draft_time") < t1)), t0


def pick_card_names(paths: list[Path], picks: pl.DataFrame | None = None) -> list[str]:
    """Every card name of the pick file(s): the `pack_card_<name>` / `pool_<name>` columns of the header and the `pick` values, plus the
    five basic lands. A double-faced card that appears both as "A" and as "A // B" is kept once (the full name)."""
    from sealed_lab.ingest import detect
    from sealed_lab.ingest import scryfall as sf

    names: set[str] = set(BASIC_NAMES.values())
    for p in paths:
        for c in detect.read_head_rows(Path(p), 1)[0]:
            for pre in ("pack_card_", "pool_"):
                if c.startswith(pre) and len(c) > len(pre):
                    names.add(c[len(pre):])
                    break
    if picks is not None and "pick" in picks.columns:
        names |= {n for n in picks["pick"].drop_nulls().unique().to_list() if n}
    best: dict[str, str] = {}
    for n in sorted(names):
        k = sf.fold(sf.front(n))
        cur = best.get(k)
        if cur is None or (" // " in n and " // " not in cur):
            best[k] = n
    return sorted(best.values())


def build_cards(set_code: str, names: list[str], fetch_scryfall: bool = True, cancel: threading.Event | None = None,
                progress: Progress | None = None) -> tuple[pl.DataFrame, list[str]]:
    """The card table of a pick-only set: the names of the pick files with the metadata the normal ingest uses (the set's Scryfall cache
    and files, then a registered 17Lands cards.csv, then inferred). Returns (cards, warnings)."""
    from pathlib import Path as _P

    from sealed_lab.config import BASICS
    from sealed_lab.ingest import cards as cards_mod
    from sealed_lab.ingest import registry, scryfall

    files = registry.files(set_code)
    sf_files = [f for f in files if f.kind == "scryfall"]
    csv_files = [f for f in files if f.kind == "cards"]
    sf_cards, warns = scryfall.get_scryfall_cards(set_code, names, [(f.path, f.fingerprint) for f in sf_files if _P(f.path).exists()],
                                                  fetch=fetch_scryfall, progress=progress, cancel=cancel)
    csv_sorted = sorted([f for f in csv_files if _P(f.path).exists()], key=lambda f: _P(f.path).stat().st_mtime)
    csv_rows = cards_mod.read_cards_csv([f.path for f in csv_sorted], set_code) if csv_sorted else {}
    cards = cards_mod.build_cards_table(names, sf_cards, csv_rows, None)
    src = {r[0]: r[1] for r in cards.group_by("meta_source").len().iter_rows()}
    n_nb = int((~cards["is_basic"]).sum())
    n_inf = src.get("inferred", 0) - int(((cards["meta_source"] == "inferred") & cards["is_basic"]).sum())
    if n_nb and n_inf > 0.5 * n_nb:
        raise ProvisionalError(
            f"{set_code}: card data (colors, mana costs) is missing for {n_inf} of {n_nb} cards, and a deck cannot be rebuilt without it. "
            "Tick 'Fetch card data from Scryfall' (needs a network connection) or add the set's Scryfall / 17Lands cards.csv file.")
    if n_inf > 0:
        warns.append(f"{n_inf} card(s) have inferred metadata only (no Scryfall / cards.csv match); colors/mana values are approximate")
    elif src.get("17lands", 0) and not src.get("scryfall", 0):
        warns.append("Scryfall data unavailable: using the 17lands cards.csv fallback (approximate pips, no oracle text)")
    _ = BASICS
    return cards, warns


# ------------------------------------------------------------------------------------------------------ deck rebuild
@dataclass
class Ctx:
    cards: pl.DataFrame
    ca: object                       # manasim.CardArrays
    alias: dict[str, int]            # pick name -> card id of a nonbasic card (full name, front name, front half of "A // B")
    basic_ids: dict[str, int]        # color letter -> basic land card id
    basic_names: set[str]


def make_ctx(cards: pl.DataFrame) -> Ctx:
    from sealed_lab.model import manasim

    ca = manasim.card_arrays(cards)
    c = cards.sort("card_id")
    ids, names = c["card_id"].to_list(), c["name"].to_list()
    fronts = c["front_name"].to_list() if "front_name" in c.columns else [None] * c.height
    basic = c["is_basic"].fill_null(False).to_list()
    alias: dict[str, int] = {}
    for i, n, b in zip(ids, names, basic):
        if not b:
            alias.setdefault(n, int(i))
    for i, n, f, b in zip(ids, names, fronts, basic):
        if b:
            continue
        for a in (f, n.split(" // ")[0].strip() if " // " in n else None):
            if a:
                alias.setdefault(a, int(i))
    by_name = {n: int(i) for i, n in zip(ids, names)}
    basic_ids = {col: by_name[nm] for col, nm in BASIC_NAMES.items() if nm in by_name}
    bn = {n for n, b in zip(names, basic) if b} | set(config.BASICS)
    return Ctx(cards=cards, ca=ca, alias=alias, basic_ids=basic_ids, basic_names=bn)


def color_symbols(nonbasic: dict[int, int], ca) -> np.ndarray:
    """Colored mana symbols (W U B R G) of the spells of a deck: pips x copies, a hybrid card's symbols weighed 1/2 each
    (the weighing `structure._color_analysis` uses to pick the main colors)."""
    ids = np.fromiter(nonbasic.keys(), dtype=np.int64, count=len(nonbasic))
    cnt = np.fromiter(nonbasic.values(), dtype=float, count=len(nonbasic))
    spell = ~ca.is_land[ids]
    pips = ca.pips[ids].astype(float)
    hyb = ca.hybrid[ids] & ((pips > 0).sum(1) > 1)
    wt = np.where(hyb, 0.5, 1.0)[:, None]
    return (pips * wt * (cnt * spell)[:, None]).sum(0)


def split_basics(n_basic: int, symbols: dict[str, float]) -> dict[str, int]:
    """`n_basic` basic lands over the main colors in proportion to `symbols` ({color: colored symbols}). Every color but the one with the
    most symbols gets its share rounded to the nearest integer; the color with the most symbols (ties: W U B R G order) takes the rest,
    so the rounding always goes to it."""
    colors = sorted(symbols, key=lambda c: (-symbols[c], "WUBRG".index(c)))
    tot = float(sum(symbols.values()))
    out = {c: int(math.floor(n_basic * symbols[c] / tot + 0.5)) for c in colors[1:]}
    out[colors[0]] = n_basic - sum(out.values())
    return {c: out[c] for c in "WUBRG" if c in out}


def rebuild_deck(nonbasic: dict[int, int], ctx: Ctx) -> tuple[dict[int, int] | None, str | None, str]:
    """(full 40-card deck {card_id: copies} or None, drop reason or None, main colors) from a run's nonbasic main-deck cards."""
    from sealed_lab.model import structure

    n = int(sum(nonbasic.values()))
    if n < MIN_NONBASIC:
        return None, "too_few_nonbasic", ""
    if n > MAX_NONBASIC:
        return None, "too_many_nonbasic", ""
    main, _splash, _needs = structure.deck_colors(nonbasic, ctx.ca)
    if not main:
        return None, "no_main_color", ""
    sym = color_symbols(nonbasic, ctx.ca)
    alloc = split_basics(DECK_SIZE - n, {c: float(sym["WUBRG".index(c)]) for c in main})
    deck = dict(nonbasic)
    for c, k in alloc.items():
        if k > 0:
            deck[ctx.basic_ids[c]] = deck.get(ctx.basic_ids[c], 0) + k
    return deck, None, main


def expand_records(wins: np.ndarray, losses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(run index, won) of the games of runs with `wins` won and `losses` lost games: wins won rows then losses lost rows per run."""
    wins = np.asarray(wins, dtype=np.int64)
    losses = np.asarray(losses, dtype=np.int64)
    n = wins + losses
    idx = np.repeat(np.arange(len(n)), n)
    start = np.repeat(np.cumsum(n) - n, n)
    j = np.arange(int(n.sum())) - start
    return idx, j < wins[idx]


def _count(x) -> int:
    """A record count from a float64 cell ("3.0" in the file): rounded, null / negative -> 0."""
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return 0
    return max(0, int(round(float(x))))


@dataclass
class Rebuilt:
    runs: pl.DataFrame               # draft_id, draft_time, skill, wins, losses, deck_idx, main_colors (rebuilt runs only)
    decks: list[dict[int, int]]      # unique full decks, index = deck_idx
    nonbasic: dict[str, dict[int, int]]   # draft_id -> nonbasic main-deck copies of every rebuilt run
    info: dict


def build_runs(picks: pl.DataFrame, ctx: Ctx, days: int | None = None) -> Rebuilt:
    """The rebuilt runs of a pick frame (read_pick_file columns). `days`: only the first `days` days after the earliest draft_time
    (the test's window); None (the app) uses every run."""
    if days is None:
        win, t0 = picks, picks["draft_time"].min()
    else:
        win, t0 = window_filter(picks, days)
    n_all_runs = int(picks["draft_id"].n_unique())
    runs = (win.sort("draft_id").group_by("draft_id", maintain_order=True)
            .agg(pl.col("draft_time").min(), pl.col("wins").first(), pl.col("losses").first(), pl.col("skill").first()))
    main = win.filter(pl.col("rate") >= MAIN_RATE).group_by("draft_id").agg(pl.col("pick"))
    picks_of = dict(zip(main["draft_id"].to_list(), main["pick"].to_list()))
    drops = {r: 0 for r in DROP_REASONS}
    unmatched: dict[str, int] = {}
    n_basic_picks = 0
    keep_rows, decks, deck_idx_of, nonbasic_of = [], [], {}, {}
    for did, dt, w, l, sk in zip(runs["draft_id"].to_list(), runs["draft_time"].to_list(), runs["wins"].to_list(),
                                 runs["losses"].to_list(), runs["skill"].to_list()):
        w, l = _count(w), _count(l)
        if w + l == 0:
            drops["no_games"] += 1
            continue
        nb: dict[int, int] = {}
        for nm in picks_of.get(did, []):
            if nm is None:
                continue
            cid = ctx.alias.get(nm)
            if cid is None:
                if nm in ctx.basic_names:
                    n_basic_picks += 1
                else:
                    unmatched[nm] = unmatched.get(nm, 0) + 1
                continue
            nb[cid] = nb.get(cid, 0) + 1
        deck, reason, mc = rebuild_deck(nb, ctx)
        if deck is None:
            drops[reason] += 1
            continue
        key = tuple(sorted(deck.items()))
        di = deck_idx_of.get(key)
        if di is None:
            di = deck_idx_of[key] = len(decks)
            decks.append(deck)
        nonbasic_of[did] = nb
        keep_rows.append((did, dt, None if sk is None or not math.isfinite(sk) else float(sk), w, l, di, mc))
    sch = dict(draft_id=pl.String, draft_time=pl.Datetime("us"), skill=pl.Float64, wins=pl.Int64, losses=pl.Int64, deck_idx=pl.Int64,
               main_colors=pl.String)
    rb = pl.DataFrame(keep_rows, schema=sch, orient="row")
    info = dict(window_start=str(t0), window_days=days, runs_in_file=n_all_runs, runs_in_window=int(runs.height), runs_rebuilt=int(rb.height),
                dropped=drops, unique_decks=len(decks), games=int(rb["wins"].sum() + rb["losses"].sum()) if rb.height else 0,
                n_basic_picks_skipped=n_basic_picks, n_unmatched_main_picks=int(sum(unmatched.values())),
                unmatched_names=dict(sorted(unmatched.items(), key=lambda kv: -kv[1])[:10]))
    return Rebuilt(runs=rb, decks=decks, nonbasic=nonbasic_of, info=info)


def runs_to_games(rb: Rebuilt, key_of_deck: np.ndarray) -> pl.DataFrame:
    """The rows of the "draft" stage: wins won rows and losses lost rows per rebuilt run, with the run's deck and skill.

    on_play is the Float64 column 0.5 (play or draw unknown; `CovEncoder.transform` casts it to float, so the encoded value is 0.5, the value
    the app uses when it predicts), mulligans 0, no opponent colors, game_number 1, game_time = the run's draft_time."""
    r = rb.runs
    idx, won = expand_records(r["wins"].to_numpy(), r["losses"].to_numpy())
    n = len(idx)
    take = lambda c: r[c].to_numpy()[idx]  # noqa: E731
    dt = r["draft_time"].to_numpy().astype("datetime64[us]")[idx]
    return pl.DataFrame(dict(
        game_key=np.arange(n, dtype=np.int64), event=np.full(n, EVENT), fmt=np.full(n, "draft"), draft_id=take("draft_id"),
        build_index=np.zeros(n, dtype=np.int16), match_number=np.ones(n, dtype=np.int16),
        game_number=np.ones(n, dtype=np.int16), deck_key=key_of_deck[take("deck_idx").astype(np.int64)].astype(np.int64),
        draft_time=dt, game_time=dt, on_play=np.full(n, 0.5), num_mulligans=np.zeros(n, dtype=np.int8),
        opp_num_mulligans=np.zeros(n, dtype=np.int8), opp_colors=pl.Series("opp_colors", [None] * n, dtype=pl.String),
        won=won, skill=pl.Series("skill", take("skill"), dtype=pl.Float64).fill_nan(None).cast(pl.Float32), main_colors=take("main_colors"),
    ), schema_overrides=dict(draft_time=pl.Datetime("us"), game_time=pl.Datetime("us"), on_play=pl.Float64, skill=pl.Float32))


def phi_for_decks(decks: list[dict[int, int]], cards: pl.DataFrame, n_jobs: int, cache: Path | None,
                  cancel: threading.Event | None = None, progress: Progress | None = None) -> np.ndarray:
    """Structure + mana-simulator features (v1, `structure.N_FEATURES` columns) of rebuilt decks, cached in `cache` by deck content."""
    import hashlib

    from sealed_lab.model import structure

    hashes = np.array([hashlib.sha1(repr(sorted(d.items())).encode()).hexdigest() for d in decks], dtype="U40")
    sig = structure.cards_signature(cards, 1, None)
    phi = np.full((len(decks), structure.N_TABLE), np.nan)
    if cache is not None and cache.exists():
        try:
            with np.load(cache, allow_pickle=False) as z:
                if str(z["sig"]) == sig and len(z["hashes"]):
                    ch, cp = z["hashes"], z["phi"]
                    order = np.argsort(ch)
                    pos = np.minimum(np.searchsorted(ch[order], hashes), len(ch) - 1)
                    hit = ch[order][pos] == hashes
                    phi[hit] = cp[order][pos[hit]]
        except Exception:  # noqa: BLE001 - recomputed
            log.warning("provisional phi cache unreadable; recomputing")
    todo = np.nonzero(np.isnan(phi[:, 0]))[0]
    if len(todo):
        offs = [0]
        cid, cnt = [], []
        for i in todo:
            items = sorted(decks[i].items())
            cid += [k for k, _ in items]
            cnt += [v for _, v in items]
            offs.append(len(cid))
        phi[todo] = structure.compute_phi(np.array(offs, dtype=np.int64), np.array(cid, dtype=np.int64), np.array(cnt, dtype=np.int64),
                                          cards, n_jobs=n_jobs, version=1, regime=None, cancel=cancel, progress=progress)
        if cache is not None:
            try:
                cache.parent.mkdir(parents=True, exist_ok=True)
                tmp = cache.with_name(cache.stem + ".tmp.npz")
                np.savez(tmp, sig=np.array(sig), hashes=hashes, phi=phi)
                os.replace(tmp, cache)
            except OSError:
                log.warning("could not write the provisional phi cache")
    return phi[:, :structure.N_FEATURES]


def draft_chain(stage, space, tmp: Path, n_jobs: int, seed: int):
    """The "draft" stage alone through the usual chain fit (Draft sigma chosen by the usual CV); returns its StageFit."""
    from sealed_lab.model import cardvalue

    tmp.mkdir(parents=True, exist_ok=True)
    fits = cardvalue.run_chain({"draft": stage}, space.n_card, space.n_feat, str(tmp), n_jobs, seed=seed, prior_ok=set(),
                               opts=cardvalue.ChainOpts())
    cardvalue.cleanup(str(tmp))
    return fits["draft"]


# ------------------------------------------------------------------------------------------------------ the transfer terms
def is_transferred(name: str) -> bool:
    return name in TRANSFER_EXACT or name.startswith(TRANSFER_PREFIX)


def load_transfer(path: Path | None = None) -> dict:
    """The shipped transfer terms (`data/provisional_transfer.json`). Raises `TransferMismatch` (a plain-language message) when the file
    is missing or was made for other deck features than the running app has: the terms are in the standardized units of those features."""
    from sealed_lab.model import structure

    p = Path(path) if path is not None else TRANSFER_PATH
    try:
        t = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise TransferMismatch(f"The provisional model's transfer terms ({p.name}) could not be read ({e}). Reinstall the app or run "
                               "tools/make_provisional_transfer.py.") from e
    if (t.get("version") != TRANSFER_VERSION or list(t.get("feature_names") or []) != list(structure.FEATURE_NAMES)
            or int(t.get("n_feat", -1)) != int(structure.N_FEATURES)):
        raise TransferMismatch(
            "The provisional model cannot be calculated: its transfer terms were made for other deck features than this version of the "
            f"app uses (file: {t.get('n_feat')} features, app: {structure.N_FEATURES}). Update the app, or the maintainer re-runs "
            "tools/make_provisional_transfer.py.")
    for f in FORMATS:
        fm = (t.get("formats") or {}).get(f)
        if not fm or len(fm.get("theta") or []) != structure.N_FEATURES or "ref" not in fm or "mull1" not in fm.get("cov", {}):
            raise TransferMismatch(f"The provisional model's transfer terms have no usable {f} entry. Re-run tools/make_provisional_transfer.py.")
    if not t.get("lam_bar"):
        raise TransferMismatch("The provisional model's transfer terms have no lambda. Re-run tools/make_provisional_transfer.py.")
    return t


def prepare_set_readonly(set_code: str, n_jobs: int):
    """(inputs, deck space, formats) of a set with game data, for the plain chain: the corpus is only read (phi of decks missing from the
    phi cache is computed but NOT written), the extra draft events (PickTwo / QuickDraft) are dropped unless the set's stored model applied
    them, as the pick-model test's `prepare_set` does."""
    from sealed_lab.model import castability, pipeline, structure

    inp = pipeline.load_inputs(set_code)
    inp.cards = castability.with_cost_groups(inp.cards, set_code)
    ed_applied = False
    rep_path = config.model_dir(set_code) / "report.json"
    if rep_path.exists():
        try:
            rep = json.loads(rep_path.read_text(encoding="utf-8"))
            ed_applied = bool((rep.get("extra_draft") or {}).get("applied", False))
        except Exception:  # noqa: BLE001
            ed_applied = False
    if pipeline.extra_draft_events(inp.games) and not ed_applied:
        inp = pipeline.without_extra_draft(inp)
    decks = inp.decks.sort("deck_key")
    keys, hashes = decks["deck_key"].to_numpy(), decks["deck_hash"].to_numpy()
    sig = structure.cards_signature(inp.cards, 1, None)
    table = np.full((len(keys), structure.N_TABLE), np.nan)
    cpath = pipeline.phi_cache_path(config.cache_dir(set_code), 1, None)
    if cpath.exists():
        with np.load(cpath, allow_pickle=False) as z:
            if str(z["sig"]) == sig and len(z["hashes"]):
                ch, cp = z["hashes"], z["phi"]
                order = np.argsort(ch)
                pos = np.minimum(np.searchsorted(ch[order], hashes), len(ch) - 1)
                hit = ch[order][pos] == hashes
                table[hit] = cp[order][pos[hit]]
    todo = np.nonzero(np.isnan(table[:, 0]))[0]
    if len(todo):
        offsets, cid, cnt = pipeline.deck_arrays(keys[todo], inp.deck_cards)
        table[todo] = structure.compute_phi(offsets, cid, cnt, inp.cards, n_jobs=n_jobs, version=1, regime=None)
    space, _ = pipeline.plain_space(inp, keys, table)
    games = inp.games
    n_games = {f: int(games.filter(pl.col("fmt") == f).height) for f in ("draft", "bo1", "bo3")}
    formats = [f for f in pipeline.FMT_ORDER if n_games.get(f, 0) >= 1]
    return inp, space, formats


def fit_plain_chain(set_code: str, n_jobs: int, tmp_root: str | None = None, seed0: int = 100) -> dict:
    """The plain chain (cards + deck features, no optional part) of a set on ALL its events, per Sealed format, as the pick-model test's `fit_m`
    (which fits the training events only). Returns {fmt: dict(names, coef, k, n_card, n_feat, ref_row, lam_bo1, ...)}; read-only on the home."""
    from sealed_lab.model import cardvalue, design, validate

    inp, space, formats = prepare_set_readonly(set_code, n_jobs)
    games = inp.games
    out: dict = {}
    for i, fmt in enumerate(formats):
        gT = games.filter(pl.col("fmt") == fmt)
        if validate._holdout_split(gT) is None:                    # (too few events / games for this format)
            out[fmt] = dict(status="skipped", reason=f"too few {fmt} events/games")
            continue
        ev = design.event_starts(gT)
        t_cut = ev["start"].max() + timedelta(days=1)             # every event counts: the prior stages keep all their events too
        st = validate.chain_stages(fmt, games, gT, t_cut, space)
        tmp = cardvalue.make_tmpdir(tmp_root)
        try:
            fits = cardvalue.run_chain(st, space.n_card, space.n_feat, tmp, n_jobs, seed=i + seed0, prior_ok=set(st) - {fmt},
                                       opts=cardvalue.ChainOpts())
        finally:
            cardvalue.cleanup(tmp)
        f = fits[fmt]
        lam = fits["bo1"].lam if fmt == "bo1" and "bo1" in fits else None
        out[fmt] = dict(status="ok", names=list(st[fmt].enc.names), coef=np.asarray(f.coef, dtype=float), k=int(f.k), n_card=int(space.n_card),
                        n_feat=int(space.n_feat), ref_row=np.asarray(st[fmt].ref_row, dtype=float), lam_bo1=None if lam is None else float(lam),
                        sigma_final=float(f.sigma), stages=list(fits), n_games=int(gT.height), n_events=int(ev.height))
    return out


def combine_transfers(per_set: dict[str, dict]) -> dict:
    """The transfer file's content from the plain-chain fits of several sets (`fit_plain_chain` output per set): the means, in the model's
    standardized units, of the covariates the pick-model test transferred (intercept, play / draw, mulligans, skill, game 2+, opponent
    pairs: by NAME, 0 where a set has no such covariate), the deck-structure coefficients, the reference-row part of the logit, and
    lambda toward the Draft stage. Event dummies and the time trend are not transferred (0)."""
    from sealed_lab.model import manasim, structure

    sets = list(per_set)
    out: dict = dict(version=TRANSFER_VERSION, kind="provisional_transfer", date=date.today().isoformat(), sets=sets,
                     feature_names=list(structure.FEATURE_NAMES), n_feat=int(structure.N_FEATURES), sim_version=int(manasim.SIM_VERSION),
                     method="plain chain (cards + deck features) on all events of each set; docs/pick_model/RULE.md", formats={})
    lams = {s: float(v["bo1"]["lam_bo1"]) for s, v in per_set.items() if (v.get("bo1") or {}).get("status") == "ok"
            and v["bo1"].get("lam_bo1") is not None}
    if not lams:
        raise ProvisionalError("no set has a Bo1 fit with a lambda: nothing to transfer")
    out["lam_bar"] = float(np.mean(list(lams.values())))
    out["lam_by_set"] = lams
    for fmt in FORMATS:
        use = [s for s in sets if (per_set[s].get(fmt) or {}).get("status") == "ok"]
        if not use:
            raise ProvisionalError(f"no set has a {fmt} fit: nothing to transfer")
        names: list[str] = []
        for s in use:
            for nm in per_set[s][fmt]["names"]:
                if is_transferred(nm) and nm not in names:
                    names.append(nm)
        cov = {nm: 0.0 for nm in names}
        thetas, refs = [], {}
        for s in use:
            d = per_set[s][fmt]
            ix = {nm: j for j, nm in enumerate(d["names"])}
            k, nc, nf = d["k"], d["n_card"], d["n_feat"]
            ref = 0.0
            for nm in names:
                if nm in ix:
                    cov[nm] += float(d["coef"][ix[nm]]) / len(use)
                    ref += float(d["coef"][ix[nm]]) * float(d["ref_row"][ix[nm]])
            refs[s] = ref
            thetas.append(np.asarray(d["coef"][k + nc:k + nc + nf], dtype=float))
        out["formats"][fmt] = dict(sets=use, cov=cov, theta=np.mean(thetas, axis=0).tolist(), ref=float(np.mean(list(refs.values()))),
                                   ref_by_set=refs, mull1=cov.get("mull1", 0.0), mull2p=cov.get("mull2p", 0.0))
    return out


def build_transfer(sets: list[str], n_jobs: int | None = None, progress: Progress | None = None) -> dict:
    """`fit_plain_chain` on each of `sets` (read-only on the app home), combined by `combine_transfers`."""
    n_jobs = int(n_jobs or os.cpu_count() or 1)
    per_set = {}
    for i, sc in enumerate(sets):
        if progress:
            progress(i / max(len(sets), 1), f"{sc}: plain chain")
        per_set[sc] = fit_plain_chain(sc, n_jobs)
    if progress:
        progress(1.0, "done")
    return combine_transfers(per_set)


# ------------------------------------------------------------------------------------------------------ the calculation
def bootstrap_draft(stage, fit, n_boot: int, tmpdir: str, n_jobs: int, seed: int, cancel=None, progress=None) -> np.ndarray:
    """Cluster bootstrap of the Draft stage: (n_boot + 1, ncols) coefficients, row 0 = the full fit, rows 1.. = refits on runs resampled
    by event (draft_id) with the full fit's sigma, as `cardvalue.bootstrap` does for the Sealed stages (same worker, same optimiser)."""
    from sealed_lab.model import cardvalue, glm

    out = np.zeros((n_boot + 1, fit.coef.shape[0]))
    out[0] = fit.coef
    if n_boot <= 0:
        return out
    path = cardvalue.stage_files(stage, tmpdir, seed + 11)
    pm = cardvalue.prior_vector(fit.k, fit.prior_deck)
    tasks = [dict(rep=b, stages=[dict(path=path, spec=fit.spec, s=fit.sigma, x0=fit.coef_m if fit.coef_m is not None else fit.coef,
                                      prior_mean=pm, stack_w=fit.w if fit.w < 1.0 else None, deck_from_prev=False,
                                      seed=seed * 100003 + b * 17, maxiter=300)]) for b in range(1, n_boot + 1)]
    for r in glm.run_parallel(glm.boot_chain_task, tasks, n_jobs, cancel=cancel, progress=progress, label="Bootstrap"):
        out[r["rep"]] = r["coefs"][0]
    return out


# ------------------------------------------------------------------------------------------------------ the decks of the runs
_PD_LOCK = threading.RLock()


def build_pick_decks(picks: pl.DataFrame, ctx: Ctx, rb: Rebuilt) -> dict[str, pl.DataFrame]:
    """The deck-search tables of a pick-only set from the runs `build_runs` rebuilt (so the same rules as the model; dropped runs are
    absent): {"runs", "deck_cards", "pools"}.

    runs: draft_id, draft_time, rank, skill, wins, losses, deck_key, main_colors, splash_colors, n_lands, n_cards (one row per run; a
    run's record is its Bo1 game record).  deck_cards: deck_key, card_id, n (the rebuilt 40-card deck, basics included).  pools: draft_id,
    card_id, n (every non-basic pick of the run: the drafted pool)."""
    from sealed_lab.model import structure

    r = rb.runs
    ids = r["draft_id"].to_list()
    rank = (picks.filter(pl.col("rank").is_not_null() & (pl.col("rank") != "") & pl.col("draft_id").is_in(ids))
            .group_by("draft_id").agg(pl.col("rank").first()))
    splash_of: dict[int, str] = {}
    for did, di in zip(ids, r["deck_idx"].to_list()):
        if di not in splash_of:
            splash_of[di] = structure.deck_colors(rb.nonbasic[did], ctx.ca)[1]
    n_lands = {i: int(sum(c for cid, c in d.items() if bool(ctx.ca.is_land[cid]))) for i, d in enumerate(rb.decks)}
    n_cards = {i: int(sum(d.values())) for i, d in enumerate(rb.decks)}
    runs = (r.join(rank, on="draft_id", how="left")
            .with_columns(pl.col("deck_idx").alias("deck_key"),
                          pl.col("deck_idx").replace_strict(splash_of, return_dtype=pl.String).alias("splash_colors"),
                          pl.col("deck_idx").replace_strict(n_lands, return_dtype=pl.Int16).alias("n_lands"),
                          pl.col("deck_idx").replace_strict(n_cards, return_dtype=pl.Int16).alias("n_cards"),
                          pl.col("skill").cast(pl.Float32), pl.col("wins").cast(pl.Int32), pl.col("losses").cast(pl.Int32))
            .select("draft_id", pl.col("draft_time"), "rank", "skill", "wins", "losses", "deck_key", "main_colors", "splash_colors",
                    "n_lands", "n_cards"))
    deck_cards = pl.DataFrame([(i, c, n) for i, d in enumerate(rb.decks) for c, n in d.items()],
                              schema=dict(deck_key=pl.Int64, card_id=pl.Int32, n=pl.Int8), orient="row")
    alias = pl.DataFrame(dict(pick=list(ctx.alias), card_id=pl.Series(list(ctx.alias.values()), dtype=pl.Int32)))
    pools = (picks.filter(pl.col("draft_id").is_in(ids) & pl.col("pick").is_not_null()).select("draft_id", "pick")
             .join(alias, on="pick", how="inner").group_by("draft_id", "card_id").agg(pl.len().cast(pl.Int8).alias("n")))
    return dict(runs=runs, deck_cards=deck_cards, pools=pools)


def _model_built_at(set_code: str) -> str | None:
    try:
        return str(json.loads((config.model_dir(set_code.upper()) / "meta.json").read_text(encoding="utf-8")).get("built_at") or "") or None
    except (OSError, ValueError):
        return None


def write_pick_decks(set_code: str, tables: dict[str, pl.DataFrame], built_at: str, dest: Path | None = None) -> None:
    """Write the deck-search tables next to the model (or into `dest`); the stamp file goes last, with the model's `built_at`."""
    d = Path(dest) if dest is not None else config.model_dir(set_code.upper())
    d.mkdir(parents=True, exist_ok=True)
    for key, fn in (("runs", PICK_DECKS), ("deck_cards", PICK_DECK_CARDS), ("pools", PICK_POOLS)):
        tmp = d / (fn + ".tmp")
        tables[key].write_parquet(tmp, compression="zstd")
        os.replace(tmp, d / fn)
    tmp = d / (PICK_DECKS_META + ".tmp")
    tmp.write_text(json.dumps(dict(version=PICK_DECKS_VERSION, built_at=built_at, n_runs=int(tables["runs"].height),
                                   n_decks=int(tables["deck_cards"]["deck_key"].n_unique()), n_pool_rows=int(tables["pools"].height))),
                   encoding="utf-8")
    os.replace(tmp, d / PICK_DECKS_META)


def pick_decks_stamp(set_code: str) -> str | None:
    """'version:built_at:n_runs' of the set's pick-deck tables when they all exist and belong to the current model, else None."""
    d = config.model_dir(set_code.upper())
    try:
        m = json.loads((d / PICK_DECKS_META).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if m.get("version") != PICK_DECKS_VERSION or m.get("built_at") != _model_built_at(set_code):
        return None
    if not all((d / fn).is_file() for fn in (PICK_DECKS, PICK_DECK_CARDS, PICK_POOLS)):
        return None
    return f"{m['version']}:{m['built_at']}:{m.get('n_runs')}"


def ensure_pick_decks(set_code: str, progress: Progress | None = None, cancel: threading.Event | None = None) -> bool:
    """True when the set's pick-deck tables are there for the current provisional model; makes them from the registered pick file(s)
    when they are missing (a model made before Deck search used them).  False when they are missing and cannot be made."""
    sc = set_code.upper()
    with _PD_LOCK:
        if pick_decks_stamp(sc) is not None:
            return True
        built = _model_built_at(sc)
        if built is None or not is_provisional(sc):
            return False
        pfiles = pick_files(sc)
        if not pfiles or any(not Path(f.path).exists() for f in pfiles):
            return False
        if progress:
            progress(0.0, "Rebuilding the draft decks from the pick file")
        paths = [Path(f.path) for f in pfiles]
        picks = pl.concat([read_pick_file(p) for p in paths]) if len(paths) > 1 else read_pick_file(paths[0])
        _check(cancel)
        ctx = make_ctx(pl.read_parquet(config.model_dir(sc) / "cards.parquet"))
        rb = build_runs(picks, ctx, None)
        write_pick_decks(sc, build_pick_decks(picks, ctx, rb), built)
        if progress:
            progress(1.0, "Draft decks ready")
        return True


def _check(cancel: threading.Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise Cancelled()


def calculate(set_code: str, progress: Progress | None = None, cancel: threading.Event | None = None, n_jobs: int | None = None,
              n_boot: int = 16, fetch_scryfall: bool = True, seed: int = 0, transfer: dict | None = None) -> dict:
    """Calculate the provisional model of a set from its PremierDraft pick file(s) and write it as the set's model (see the module doc).
    Returns the meta dict. `transfer`: the transfer terms (default: the shipped file; tests pass their own)."""
    from sealed_lab import CODE_VERSION
    from sealed_lab.ingest import registry
    from sealed_lab.model import artifact, castability, cardvalue, design, manasim, reference, structure
    from sealed_lab.model import glm  # noqa: F401  (imported in the parent: workers re-import it themselves)
    from scipy.special import expit
    import scipy.sparse as sp

    t_start = time.time()
    set_code = set_code.upper()
    n_jobs = int(n_jobs or os.cpu_count() or 1)

    def prog(f: float, m: str) -> None:
        if progress:
            progress(min(max(f, 0.0), 1.0), m)

    tr = transfer if transfer is not None else load_transfer()
    files = registry.files(set_code, include_shared=False)
    if any(f.kind == "game" for f in files):
        raise ProvisionalError(f"{set_code} has game data: it calculates a normal model, not a provisional one")
    pfiles = pick_files(set_code)
    if not pfiles:
        raise ProvisionalError(f"{set_code} has no PremierDraft pick file (draft_data_public.{set_code}.PremierDraft.csv.gz) registered")
    gone = [f.path for f in pfiles if not Path(f.path).exists()]
    if gone:
        raise ProvisionalError(f"{set_code}: pick file not found: {gone[0]}")
    paths = [Path(f.path) for f in pfiles]
    source_files = input_signature(registry.files(set_code))
    _check(cancel)

    prog(0.0, "Reading the pick file(s)")
    picks = pl.concat([read_pick_file(p) for p in paths]) if len(paths) > 1 else read_pick_file(paths[0])
    _check(cancel)
    prog(0.08, "Building the card table")
    names = pick_card_names(paths, picks)
    cards, warns = build_cards(set_code, names, fetch_scryfall=fetch_scryfall, cancel=cancel,
                               progress=lambda f, m: prog(0.08 + 0.04 * f, m))
    cards = castability.with_cost_groups(cards, set_code)
    ctx = make_ctx(cards)
    if len(ctx.basic_ids) < 5:
        raise ProvisionalError(f"{set_code}: the card table lacks a basic land ({sorted(set(BASIC_NAMES) - set(ctx.basic_ids))})")
    prog(0.12, "Rebuilding the decks")
    rb = build_runs(picks, ctx, None)
    if rb.runs.height == 0 or rb.info["games"] == 0:
        raise ProvisionalError(f"{set_code}: no usable run in the pick file(s) ({rb.info['dropped']})")
    log.info("%s provisional: %d runs in file, %d rebuilt, dropped %s, %d games, %d decks", set_code, rb.info["runs_in_file"],
             rb.info["runs_rebuilt"], rb.info["dropped"], rb.info["games"], rb.info["unique_decks"])
    _check(cancel)

    prog(0.16, "Mana simulation of the rebuilt decks")
    config.cache_dir(set_code).mkdir(parents=True, exist_ok=True)
    phi = phi_for_decks(rb.decks, cards, n_jobs, config.cache_dir(set_code) / "phi_provisional.npz", cancel=cancel,
                        progress=lambda f, m: prog(0.16 + 0.14 * f, m))
    _check(cancel)

    # the deck space of the rebuilt decks, standardised by their games (as a normal set's space is by its played decks)
    n_dk = len(rb.decks)
    keys = np.arange(n_dk, dtype=np.int64)
    dk_rows = [(i, c, n) for i, d in enumerate(rb.decks) for c, n in d.items()]
    deck_cards = pl.DataFrame(dk_rows, schema=dict(deck_key=pl.Int64, card_id=pl.Int32, n=pl.Int8), orient="row")
    r = rb.runs
    games_per_deck = np.bincount(r["deck_idx"].to_numpy(), weights=(r["wins"] + r["losses"]).to_numpy().astype(float), minlength=n_dk)
    space = design.build_deck_space(keys, phi, deck_cards, cards, weights=np.maximum(games_per_deck, 1e-9))
    g_pk = runs_to_games(rb, keys)
    stage = design.build_stage("draft", g_pk, space)
    prog(0.32, "Fitting the Draft stage")
    tmp = cardvalue.make_tmpdir(config.cache_dir(set_code))
    try:
        fit = None
        tmp_fit = Path(tmp) / "fit"
        tmp_fit.mkdir()
        fits = cardvalue.run_chain({"draft": stage}, space.n_card, space.n_feat, str(tmp_fit), n_jobs, seed=seed + 7, cancel=cancel,
                                   progress=lambda f, m: prog(0.32 + 0.20 * f, m), prior_ok=set(), opts=cardvalue.ChainOpts())
        fit = fits["draft"]
        _check(cancel)
        prog(0.52, "Bootstrap")
        coefs = bootstrap_draft(stage, fit, n_boot, tmp, n_jobs, seed + 900, cancel=cancel, progress=lambda f, m: prog(0.52 + 0.30 * f, m))
    finally:
        cardvalue.cleanup(tmp)
    _check(cancel)

    # ---- the artifact: lambda-bar x the Draft card values, transferred terms
    prog(0.84, "Writing the model")
    k, nc, nf = fit.k, space.n_card, space.n_feat
    lam = float(tr["lam_bar"])
    B = coefs.shape[0]
    u = coefs[:, k:k + nc]
    n_total = space.n_cards_total
    csort = cards.sort("card_id")
    run_deck = r["deck_idx"].to_numpy()
    run_keys = keys[run_deck]
    decks_df = pl.DataFrame(dict(deck_key=keys, n_lands=[int(sum(c for cid, c in d.items() if bool(ctx.ca.is_land[cid]))) for d in rb.decks],
                                 n_cards=[int(sum(d.values())) for d in rb.decks]),
                            schema=dict(deck_key=pl.Int64, n_lands=pl.Int16, n_cards=pl.Int16))
    draws, reference_q, ref_meta, extra_cols = {}, {}, {}, {}
    zeros_n = np.zeros(n_total)
    for f in FORMATS:
        tf = tr["formats"][f]
        beta = np.zeros((B, n_total))
        beta[:, space.nb_ids] = lam * u
        theta = np.tile(np.asarray(tf["theta"], dtype=float), (B, 1))
        ref = np.full(B, float(tf["ref"]))
        mull = np.tile([float(tf["mull1"]), float(tf["mull2p"])], (B, 1))
        draws[f] = dict(beta=beta, theta=theta, ref=ref, mull=mull)
        regime = config.format_regime(f)
        mull_p = reference.mull_inputs(decks_df, run_keys, lambda nl_, nc_, r_=regime: manasim.mull_split(nl_, nc_, r_))
        deck_coefs = np.hstack([beta[:, space.nb_ids], theta])
        reference_q[f] = reference.reference_q(space, space.rows(run_keys), deck_coefs, ref, mull=(mull_p, mull))
        se = np.zeros(n_total)
        se[space.nb_ids] = lam * np.asarray(fit.card_se, dtype=float)
        full = lambda v, fill=0.0: _scatter(n_total, space.nb_ids, v, fill)  # noqa: E731
        extra_cols[f"beta_{f}"] = full(lam * fit.beta_total)
        extra_cols[f"beta_attr_{f}"] = zeros_n
        extra_cols[f"se_{f}"] = se
        extra_cols[f"exposure_{f}"] = full(stage.exposure)
        extra_cols[f"low_data_{f}"] = full((stage.exposure < 2000).astype(float)) > 0.5
        extra_cols[f"psi_flag_{f}"] = np.zeros(n_total, dtype=bool)
        ref_meta[f] = dict(ref=float(ref[0]), draws_sd=0.0,
                           reference_wr=float(expit(np.median(np.log(np.clip(reference_q[f][0], 1e-6, 1 - 1e-6) /
                                                                    (1 - np.clip(reference_q[f][0], 1e-6, 1 - 1e-6)))))))
    extra_cols["psi"] = zeros_n
    extra_cols["psi_se"] = np.full(n_total, np.nan)
    cards_art = csort.with_columns([pl.Series(k_, v) for k_, v in extra_cols.items()])

    # what real decks look like (land counts, spell colors / sources): the builder's range and the "outside the data" flags
    land_sup: dict[str, dict[str, float]] = {}
    support: dict = {}
    try:
        lands = decks_df["n_lands"].to_numpy()
        w_dk = np.maximum(games_per_deck, 0.0)
        tot = max(float(w_dk.sum()), 1.0)
        share = {str(int(n)): round(float(w_dk[lands == n].sum() / tot), 5) for n in np.unique(lands)}
        rows_, cols_, vals_ = zip(*[(i, c, float(n)) for i, d in enumerate(rb.decks) for c, n in d.items()])
        M = sp.csr_matrix((vals_, (rows_, cols_)), shape=(n_dk, n_total))
        cc = castability.cast_cards(cards)
        ok40 = decks_df["n_cards"].to_numpy() == 40
        for f in FORMATS:
            land_sup[f] = share
            if (w_dk * ok40).sum() > 0:
                support[f] = castability.support_reference(M, w_dk * ok40, cc)
    except Exception:  # noqa: BLE001 - optional references
        log.exception("land / support reference of the provisional model unavailable")

    built_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    info = rb.info
    n_draft_games = int(stage.n_games)
    stage_info = {"draft": dict(n_games=n_draft_games, n_events=int(stage.n_groups), sigma=float(fit.sigma), lam=None, lam_se=None, lam_from=None)}
    prov = dict(source="17Lands pick files", events=[EVENT], files=[p.name for p in paths], n_runs=int(info["runs_rebuilt"]),
                n_games=n_draft_games, runs_in_file=int(info["runs_in_file"]), dropped=dict(info["dropped"]),
                unique_decks=int(info["unique_decks"]), n_unmatched_main_picks=int(info["n_unmatched_main_picks"]),
                window_start=info["window_start"], sigma_draft=float(fit.sigma), lam_bar=lam,
                transfer=dict(sets=list(tr.get("sets") or []), date=tr.get("date")), result=RESULT_DOC,
                note=BANNER)
    meta = dict(set_code=set_code, formats=list(FORMATS), built_at=built_iso, built_corpus_at=None, code_version=CODE_VERSION,
                n_games={"draft": n_draft_games}, n_boot=int(n_boot), n_draws=int(B), seed=seed, quick=False,
                feature_names=list(structure.FEATURE_NAMES), phi_mean=space.mu.tolist(), phi_sd=space.sd.tolist(), n_cards=int(n_total),
                stages=stage_info, ref=ref_meta, warnings=list(warns) + [BANNER], validation={}, deployed={}, degraded=[],
                source_files=source_files, provisional=prov, land_support=land_sup,
                recency=dict(mode="off", applied=False, half_life=None, reason="provisional model", scope="format"),
                manasim=dict(applied=False, version=1, regimes={f: config.format_regime(f) for f in FORMATS}, keep={}, keep_info={}),
                castability=dict(applied=False, mode="off", reason="provisional model", support=support),
                matchups=dict(applied=False, applied_formats=[]), playdraw=dict(applied=False), archetypes=dict(n=0),
                seconds=round(time.time() - t_start, 1))
    d = config.model_dir(set_code)
    artifact.save_extras(d, None, None, None, None, list(FORMATS), playdraw=None)       # no layers of an earlier model survive
    html = (f"<html><body><h1>{set_code}: provisional model</h1><p>{BANNER}</p><p>No report: this set has only pick data "
            f"(see {RESULT_DOC}).</p></body></html>")
    artifact.save_artifact(set_code, meta, cards_art, draws, reference_q, html, dict(set_code=set_code, provisional=prov, formats=list(FORMATS)))
    prog(0.97, "Writing the draft decks")
    try:                                       # Deck search reads these; if writing fails they are made again on first use
        write_pick_decks(set_code, build_pick_decks(picks, ctx, rb), built_iso)
    except Exception:  # noqa: BLE001
        log.exception("%s: could not write the pick-deck tables", set_code)
    prog(1.0, "Provisional model ready")
    return meta


def _scatter(n_total: int, idx: np.ndarray, v: np.ndarray, fill: float = 0.0) -> np.ndarray:
    out = np.full(n_total, fill)
    out[idx] = v
    return out
