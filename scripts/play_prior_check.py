"""Pre-registered "what strong players choose to play" card-value prior test on Sealed hold-outs (docs/play_prior/RULE.md).

Arms S_all, S_skill (Sealed pools and decks from the corpus), D_skill, D_rank (first game of Draft runs from the raw 17Lands game
files). Each play rate z_c = logit((k+1)/(n+2)) is mapped onto the zero-mean first-stage card effects exactly as the cold-start
pick prior does (pickprior.map_signal_to_betas) and the chain is refitted with it as the first stage's prior mean.

Reads the app home and the raw files strictly read-only (SEALED_LAB_HOME must point at a scratch copy); writes artefacts and the
JSON summary to OUT.

    python scripts/play_prior_check.py OUT [--sets ...] [--jobs N] [--selfcheck]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import polars as pl
from scipy.special import expit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from sealed_lab import config  # noqa: E402
from sealed_lab.ingest import detect, registry  # noqa: E402
from sealed_lab.model import cardvalue, castability, design, glm, pickprior, pipeline, pooling, validate  # noqa: E402
import cold_start_check as csc  # noqa: E402  (read_phi_readonly, paired_gain_metrics, pool_with_floor, check_out_not_home)

ALL_SETS = ["ECL", "EOE", "FIN", "FRA", "HOB", "MSH", "OM1", "SOS", "TDM", "TLA", "TMT"]
ARMS = ("S_all", "S_skill", "D_skill", "D_rank")
SEALED_ARMS = ("S_all", "S_skill")
DRAFT_ARMS = ("D_skill", "D_rank")
SKILL_HI = 0.60                     # S_skill, D_skill
RANK_SKILL = 0.55                   # D_rank
RANK_SET = ("diamond", "mythic")    # D_rank
MIN_RUNS = 30                       # eligible runs a card needs for a signal (passed to map_signal_to_betas as min_taken)
TRAP_MIN_N = 50
TRAP_RATE = 0.15
MIN_GAIN = 1e-4
SE_FLOOR = 1e-5
N_REP = 500
DRAFT_EVENTS = ("PremierDraft", "PickTwoDraft")
TRACKED = ["The Ten Rings", "Meek Attack", "Glimpse of Nature", "Key to the Side-Door", "Dragonstorm Forecaster",
           "Broadcast Takeover", "Crescent Island Temple", "Northern Air Temple"]
SKILL_COLS = ("user_game_win_rate_bucket", "skill")


# ------------------------------------------------------------------------------------------------------ safety
def check_home_not_real() -> None:
    """Refuse to run against the owner's real app home (%LOCALAPPDATA%/sealed_lab)."""
    home = config.app_home().resolve()
    la = os.environ.get("LOCALAPPDATA")
    if la and home == (Path(la) / "sealed_lab").resolve():
        raise RuntimeError(f"refusing to run on the real app home {home}: set SEALED_LAB_HOME to a scratch copy")


def check_out_not_home(out: Path) -> None:
    home = config.app_home().resolve()
    out_r = out.resolve()
    if out_r == home or home in out_r.parents:
        raise RuntimeError(f"refusing to write inside app home {home}")


# ------------------------------------------------------------------------------------------------------ the signal
def card_lookup(cards: pl.DataFrame) -> pl.DataFrame:
    """alias -> (canon, colors) of nonland, nonbasic cards; aliases = full name, front_name, front half of 'A // B'."""
    c = cards.filter(~pl.col("is_land").fill_null(False) & ~pl.col("is_basic").fill_null(False))
    alias: dict[str, tuple[str, str]] = {}
    rows = list(zip(c["name"].to_list(), c["front_name"].to_list() if "front_name" in c.columns else [None] * c.height,
                    c["colors"].to_list()))
    for nm, _fn, col in rows:
        alias.setdefault(nm, (nm, col or ""))
    for nm, fn, col in rows:
        for a in (fn, nm.split(" // ")[0].strip() if " // " in nm else None):
            if a:
                alias.setdefault(a, (nm, col or ""))
    return pl.DataFrame(dict(alias=list(alias), canon=[v[0] for v in alias.values()], colors=[v[1] for v in alias.values()]),
                        schema=dict(alias=pl.String, canon=pl.String, colors=pl.String))


def play_counts(long: pl.DataFrame, cards: pl.DataFrame) -> pl.DataFrame:
    """k (played) and n (eligible) per nonland nonbasic card from a long run frame.

    `long`: draft_id (run), main_colors, card_name, pool_n (copies in the pool), deck_n (copies in the deck); one row per run and card.
    A run is eligible for a card when the card is in its pool and the card's colors are within the run's main colors (colorless:
    always). A card played only as an off-color splash is in neither k nor n. Runs without main colors are dropped."""
    lk = card_lookup(cards)
    d = (long.filter((pl.col("pool_n") > 0) & pl.col("main_colors").is_not_null())
         .join(lk, left_on="card_name", right_on="alias", how="inner")
         .group_by(["draft_id", "canon"])
         .agg(pl.col("pool_n").sum(), pl.col("deck_n").sum(), pl.col("main_colors").first(), pl.col("colors").first()))
    if d.height == 0:
        return pl.DataFrame(schema=dict(name=pl.String, k=pl.Int64, n=pl.Int64))
    pairs = d.select("colors", "main_colors").unique()
    ok = [set(c or "") <= set(m or "") for c, m in zip(pairs["colors"].to_list(), pairs["main_colors"].to_list())]
    pairs = pairs.with_columns(pl.Series("_elig", ok, dtype=pl.Boolean))
    d = d.join(pairs, on=["colors", "main_colors"], how="left").filter(pl.col("_elig"))
    return (d.group_by("canon").agg(pl.len().cast(pl.Int64).alias("n"), (pl.col("deck_n") > 0).sum().cast(pl.Int64).alias("k"))
            .rename({"canon": "name"}).sort("name"))


def play_signal(long: pl.DataFrame, cards: pl.DataFrame) -> tuple[list[str], np.ndarray, np.ndarray]:
    """(names, k, n) of the play-rate signal of a long run frame."""
    c = play_counts(long, cards)
    return c["name"].to_list(), c["k"].to_numpy().astype(float), c["n"].to_numpy().astype(float)


sealed_play_signal = play_signal


def signal_z(k: np.ndarray, n: np.ndarray) -> np.ndarray:
    p = (k + 1.0) / (n + 2.0)
    return np.log(p / (1.0 - p))


# ------------------------------------------------------------------------------------------------------ Sealed runs
def sealed_runs(games: pl.DataFrame) -> pl.DataFrame:
    """One row per Sealed run (bo1 and bo3): draft_id, fmt, start (earliest game_time, fallback draft_time), skill (first game),
    deck_key (the run's deck) and main_colors (most common value over that deck's games).

    bo1: the deck_key with the most games in the run (ties: lowest build_index).
    bo3: the deck_key of the most matches, a match's deck being its lowest game_number game's, i.e. game 1 (ties: earliest match)."""
    g = games.filter(pl.col("fmt").is_in(["bo1", "bo3"]))
    if g.height == 0:
        return pl.DataFrame(schema=dict(draft_id=pl.String, fmt=pl.String, start=pl.Datetime("us"), skill=pl.Float64,
                                        deck_key=pl.Int64, main_colors=pl.String))
    ev = design.event_starts(g).select("draft_id", "start")
    first = (g.sort(["game_time", "match_number", "game_number"], nulls_last=True).group_by("draft_id", maintain_order=True)
             .agg(pl.col("skill").first().cast(pl.Float64).alias("skill"), pl.col("fmt").first()))
    parts = []
    g1 = g.filter(pl.col("fmt") == "bo1")
    if g1.height:
        b = g1.group_by(["draft_id", "deck_key"]).agg(pl.len().alias("n"), pl.col("build_index").min().alias("o"))
        parts.append(b.sort(["draft_id", "n", "o", "deck_key"], descending=[False, True, False, False])
                     .group_by("draft_id", maintain_order=True).first().select("draft_id", "deck_key"))
    g3 = g.filter(pl.col("fmt") == "bo3")
    if g3.height:
        m = (g3.sort(["draft_id", "match_number", "game_number"], nulls_last=True)
             .group_by(["draft_id", "match_number"], maintain_order=True).agg(pl.col("deck_key").first()))
        b = m.group_by(["draft_id", "deck_key"]).agg(pl.len().alias("n"), pl.col("match_number").min().alias("o"))
        parts.append(b.sort(["draft_id", "n", "o", "deck_key"], descending=[False, True, False, False])
                     .group_by("draft_id", maintain_order=True).first().select("draft_id", "deck_key"))
    pick = pl.concat(parts)
    mc = (g.join(pick, on=["draft_id", "deck_key"], how="inner").filter(pl.col("main_colors").is_not_null())
          .group_by(["draft_id", "main_colors"]).len().sort(["draft_id", "len", "main_colors"], descending=[False, True, False])
          .group_by("draft_id", maintain_order=True).first().select("draft_id", "main_colors"))
    return (ev.join(first, on="draft_id", how="inner").join(pick, on="draft_id", how="inner")
            .join(mc, on="draft_id", how="left").select("draft_id", "fmt", "start", "skill", "deck_key", "main_colors"))


def sealed_long(runs: pl.DataFrame, pools: pl.LazyFrame, deck_cards: pl.DataFrame, cards: pl.DataFrame) -> pl.DataFrame:
    """Long frame (draft_id, skill, main_colors, card_name, pool_n, deck_n) of `runs`: pool from pools.parquet, deck from deck_cards."""
    names = cards.select("card_id", pl.col("name").alias("card_name"))
    p = (pools.join(runs.select("draft_id", "skill", "main_colors", "deck_key").lazy(), on="draft_id", how="inner")
         .rename({"n": "pool_n"}).collect())
    dc = deck_cards.rename({"n": "deck_n"})
    p = p.join(dc, on=["deck_key", "card_id"], how="left").with_columns(pl.col("deck_n").fill_null(0)).join(names, on="card_id", how="inner")
    return p.select("draft_id", "skill", "main_colors", "card_name", pl.col("pool_n").cast(pl.Int64), pl.col("deck_n").cast(pl.Int64))


def sealed_cell_long(runs_all: pl.DataFrame, t_cut, audit: pl.DataFrame | None, pools: pl.LazyFrame, deck_cards: pl.DataFrame,
                     cards: pl.DataFrame) -> pl.DataFrame:
    r = runs_all.filter(pl.col("start") < t_cut)
    if audit is not None:
        r = r.join(audit.filter(pl.col("invariant")).select("draft_id"), on="draft_id", how="semi")
    return sealed_long(r, pools, deck_cards, cards)


# ------------------------------------------------------------------------------------------------------ Draft runs (raw files)
def draft_game_files(sc: str) -> list[Path]:
    out = []
    for f in registry.files(sc, include_shared=False):
        if f.kind == "game" and f.event in DRAFT_EVENTS and Path(f.path).exists():
            out.append(Path(f.path))
    return sorted(out)


def _parse_float(s):
    try:
        return float(s) if s not in (None, "") else None
    except ValueError:
        return None


def read_draft_runs(path: Path) -> pl.DataFrame:
    """Stream one raw 17Lands game file; keep each draft_id's FIRST game (lowest build_index, match_number, game_number) and return the
    long frame (draft_id, draft_time, rank, skill, main_colors, card_name, pool_n, deck_n) of its pool (deck + sideboard), only cards with
    pool_n > 0. Main colors, rank and skill are that game's."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.csv as pacsv

    hdr = detect.read_head_rows(path, 1)[0]
    deck = {c[len("deck_"):]: c for c in hdr if c.startswith("deck_")}
    side = {c[len("sideboard_"):]: c for c in hdr if c.startswith("sideboard_")}
    names = list(deck)
    skill_col = next((c for c in SKILL_COLS if c in hdr), None)
    key_cols = ["build_index", "match_number", "game_number"]
    for c in ["draft_id"] + key_cols:
        if c not in hdr:
            raise ValueError(f"{path}: column {c} missing")
    str_cols = [c for c in ("draft_id", "draft_time", "rank", "main_colors") if c in hdr] + ([skill_col] if skill_col else [])
    meta = str_cols + key_cols
    col_types = {**{c: pa.string() for c in str_cols}, **{c: pa.int32() for c in key_cols},
                 **{c: pa.float32() for c in list(deck.values()) + list(side.values())}}   # some files write counts as "0.0"
    include = meta + list(deck.values()) + [side[n] for n in names if n in side]
    big = 1 << 30
    best: dict[str, tuple] = {}     # draft_id -> (key, meta dict, deck vec, side vec)
    with detect.open_binary(path) as stream:
        reader = pacsv.open_csv(stream, read_options=pacsv.ReadOptions(block_size=24 << 20),
                                convert_options=pacsv.ConvertOptions(include_columns=include, column_types=col_types))
        while True:
            try:
                batch = reader.read_next_batch()
            except StopIteration:
                break
            if batch.num_rows == 0:
                continue
            t = pl.from_arrow(pa.Table.from_batches([batch]).select(meta)).with_row_index("_r")
            cand = (t.sort(key_cols, nulls_last=True).unique("draft_id", keep="first", maintain_order=True))
            take, rows_meta = [], []
            for row in cand.iter_rows(named=True):
                did = row["draft_id"]
                if did is None:
                    continue
                key = tuple(big if row[c] is None else int(row[c]) for c in key_cols)
                cur = best.get(did)
                if cur is None or key < cur[0]:
                    take.append(row["_r"])
                    rows_meta.append((did, key, row))
            if not take:
                continue
            sub = batch.take(pa.array(take, type=pa.int32()))

            def mat(cols: list[str | None]) -> np.ndarray:
                arrs = [np.zeros(len(take), dtype=np.int8) if c is None else
                        np.rint(pc.fill_null(sub.column(c), 0).to_numpy(zero_copy_only=False)).astype(np.int8) for c in cols]
                return np.column_stack(arrs)

            dm = mat([deck[n] for n in names])
            sm = mat([side.get(n) for n in names])
            for i, (did, key, row) in enumerate(rows_meta):
                best[did] = (key, row, dm[i].copy(), sm[i].copy())

    schema = dict(draft_id=pl.String, draft_time=pl.Datetime("us"), rank=pl.String, skill=pl.Float64, main_colors=pl.String,
                  card_name=pl.String, pool_n=pl.Int16, deck_n=pl.Int16)
    if not best:
        return pl.DataFrame(schema=schema)
    ids = list(best)
    name_arr = np.array(names, dtype=object)
    frames = []
    step = 20_000
    for a in range(0, len(ids), step):
        chunk = ids[a:a + step]
        dm = np.stack([best[d][2] for d in chunk]).astype(np.int16)
        sm = np.stack([best[d][3] for d in chunk]).astype(np.int16)
        pool = dm + sm
        r, c = np.nonzero(pool > 0)
        run_ids = np.array(chunk, dtype=object)
        metas = [best[d][1] for d in chunk]
        mk = lambda col: np.array([m.get(col) for m in metas], dtype=object)  # noqa: E731
        sk = np.array([_parse_float(m.get(skill_col)) if skill_col else None for m in metas], dtype=object)
        frames.append(pl.DataFrame(dict(
            draft_id=run_ids[r].tolist(),
            draft_time=mk("draft_time")[r].tolist(),
            rank=mk("rank")[r].tolist(),
            skill=sk[r].tolist(),
            main_colors=mk("main_colors")[r].tolist(),
            card_name=name_arr[c].tolist(),
            pool_n=pool[r, c].tolist(),
            deck_n=dm[r, c].tolist(),
        ), schema={**schema, "draft_time": pl.String}))
    out = pl.concat(frames)
    return out.with_columns(pl.col("draft_time").str.to_datetime(strict=False, time_unit="us"))


def draft_long_cached(sc: str, out: Path) -> pl.DataFrame | None:
    """Long frame of the Draft runs of a set (all registered PremierDraft + PickTwoDraft game files), cached in OUT."""
    files = draft_game_files(sc)
    if not files:
        return None
    cpath = out / f"draft_long_{sc}.parquet"
    meta_path = out / f"draft_long_{sc}.json"
    sig = [[str(p), p.stat().st_size] for p in files]
    if cpath.exists() and meta_path.exists():
        try:
            if json.loads(meta_path.read_text(encoding="utf-8")).get("files") == sig:
                return pl.read_parquet(cpath)
        except Exception:  # noqa: BLE001
            pass
    t0 = time.time()
    parts = [read_draft_runs(p) for p in files]
    df = pl.concat(parts) if len(parts) > 1 else parts[0]
    df.write_parquet(cpath)
    meta_path.write_text(json.dumps(dict(files=sig, n_runs=int(df["draft_id"].n_unique()), seconds=round(time.time() - t0, 1))), encoding="utf-8")
    return df


def arm_long(arm: str, sealed: pl.DataFrame | None, draft: pl.DataFrame | None, t_cut) -> pl.DataFrame | None:
    """The long frame of the runs an arm counts (training runs only: started strictly before t_cut)."""
    if arm in SEALED_ARMS:
        if sealed is None:
            return None
        if arm == "S_skill":
            return sealed.filter(pl.col("skill").round(4) >= SKILL_HI)
        return sealed
    if draft is None:
        return None
    d = draft.filter(pl.col("draft_time") < t_cut)
    if arm == "D_skill":
        return d.filter(pl.col("skill").round(4) >= SKILL_HI)
    return d.filter(pl.col("rank").str.to_lowercase().is_in(list(RANK_SET)) & (pl.col("skill").round(4) >= RANK_SKILL))


# ------------------------------------------------------------------------------------------------------ chain with a first-stage prior
def run_chain_with_first_prior(stages: dict[str, design.StageData], prior_cards: np.ndarray, n_card: int, n_feat: int, tmpdir: str,
                               n_jobs: int, seed: int, prior_ok: set[str], opts: cardvalue.ChainOpts) -> dict[str, cardvalue.StageFit]:
    """cold_start_check.run_chain_with_draft_prior, with the prior in the FIRST stage whatever its name (draft, or bo1 of OM1)."""
    order = [n for n in ("draft", "bo1", "bo3") if n in stages]
    fits: dict[str, cardvalue.StageFit] = {}
    prev: cardvalue.StageFit | None = None
    p_deck = n_card + n_feat
    for i, name in enumerate(order):
        st = stages[name]
        lam = lam_se = lam_from = None
        if prev is None:
            prior = np.zeros(p_deck)
            if i == 0:
                prior[:n_card] = prior_cards
        elif prev.name == "draft":
            lam_raw, lam_se = cardvalue.fit_lambda(st, prev.deck)
            lam = float(np.clip(lam_raw, *cardvalue.LAMBDA_CLIP))
            lam_from = prev.name
            prior = lam * prev.deck
        else:
            prior = prev.deck.copy()
            lam_from = prev.name
        s_dir = os.path.join(tmpdir, f"st_{name}")
        os.makedirs(s_dir, exist_ok=True)
        fits[name] = cardvalue._stage_fit(st, prior, n_card, n_feat, s_dir, n_jobs, seed + i, None, None,
                                          lam, lam_se, lam_from, opts=opts)
        if name in prior_ok:
            prev = fits[name]
    return fits


# ------------------------------------------------------------------------------------------------------ small statistics
def mean_resid_ci(y: np.ndarray, p: np.ndarray, groups: np.ndarray, n_rep: int = N_REP, seed: int = 0) -> dict:
    """Mean of (won - p) with a cluster bootstrap 95% CI over the groups (draft_ids) present."""
    u, inv = np.unique(groups, return_inverse=True)
    G = len(u)
    r = (y - p).astype(float)
    s_g = np.bincount(inv, weights=r, minlength=G)
    n_g = np.bincount(inv, minlength=G).astype(float)
    mult = np.random.default_rng(seed).multinomial(G, np.full(G, 1.0 / G), size=n_rep).astype(float)
    reps = (mult @ s_g) / (mult @ n_g)
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return dict(mean=float(s_g.sum() / n_g.sum()), lo=float(lo), hi=float(hi), n_games=int(len(y)), n_events=int(G))


def pool_means(items: dict[str, dict]) -> dict | None:
    """DL pool of {set: {mean, lo, hi}}; zero-width cells excluded (and listed)."""
    ys, ses, used, excl = [], [], [], []
    for sc, m in items.items():
        if not all(math.isfinite(m[k]) for k in ("mean", "lo", "hi")) or m["hi"] <= m["lo"]:
            excl.append(sc)
            continue
        ys.append(m["mean"])
        ses.append(pooling.se_from_ci(m["lo"], m["hi"]))
        used.append(sc)
    if not ys:
        return None
    return dict(pooled=pooling.dersimonian_laird(ys, ses), sets=used, excluded=excl)


def top_movers(names_by_col: list[str], b_m: np.ndarray, b_a: np.ndarray, k: int = 10) -> list[dict]:
    d = np.abs(b_a - b_m)
    idx = np.argsort(-d)[:k]
    return [dict(name=names_by_col[i], beta_M=float(b_m[i]), beta_arm=float(b_a[i]), delta=float(b_a[i] - b_m[i])) for i in idx]


def tracked_values(col_of_name: dict[str, int], b_m: np.ndarray, b_a: np.ndarray) -> dict:
    out = {}
    for nm in TRACKED:
        j = col_of_name.get(nm)
        if j is not None:
            out[nm] = dict(beta_M=float(b_m[j]), beta_arm=float(b_a[j]))
    return out


# ------------------------------------------------------------------------------------------------------ one set
def prepare_set(sc: str, n_jobs: int):
    inp = pipeline.load_inputs(sc)
    inp.cards = castability.with_cost_groups(inp.cards, sc)
    rep_path = config.model_dir(sc) / "report.json"
    ed_applied = False
    if rep_path.exists():
        try:
            rep = json.loads(rep_path.read_text(encoding="utf-8"))
            ed_applied = bool((rep.get("extra_draft") or {}).get("applied", False))
        except Exception:  # noqa: BLE001
            ed_applied = False
    if pipeline.extra_draft_events(inp.games) and not ed_applied:
        inp = pipeline.without_extra_draft(inp)
    keys, table = csc.read_phi_readonly(inp, config.cache_dir(sc), n_jobs)
    space, _ = pipeline.plain_space(inp, keys, table)
    games = inp.games
    n_games = {f: int(games.filter(pl.col("fmt") == f).height) for f in ("draft", "bo1", "bo3")}
    formats = [f for f in pipeline.FMT_ORDER if n_games.get(f, 0) >= 1]
    return inp, space, formats


def first_stage_name(st_m: dict) -> str:
    return [n for n in ("draft", "bo1", "bo3") if n in st_m][0]


def run_set(sc: str, out: Path, n_jobs: int, selfcheck: bool = False) -> dict:
    t0 = time.time()
    inp, space, formats = prepare_set(sc, n_jobs)
    games = inp.games
    opts = cardvalue.ChainOpts()
    cards = inp.cards
    csort = cards.sort("card_id")
    name_of_id = dict(zip(csort["card_id"].to_list(), csort["name"].to_list()))
    id_of_name = {n: i for i, n in name_of_id.items()}
    names_by_col = [name_of_id[int(i)] for i in space.nb_ids]
    col_of_name: dict[str, int] = {}
    for j, nm in enumerate(names_by_col):
        col_of_name.setdefault(nm, j)
        if " // " in nm:
            col_of_name.setdefault(nm.split(" // ")[0].strip(), j)

    sealed_all = draft_all = audit = pools = None
    runs_all = None
    if not selfcheck:
        cdir = config.corpus_dir(sc)
        pools = pl.scan_parquet(cdir / "pools.parquet")
        ap = cdir / "pool_audit.parquet"
        audit = pl.read_parquet(ap) if ap.exists() else None
        runs_all = sealed_runs(games)
        draft_all = draft_long_cached(sc, out)

    fmt_res: dict[str, dict] = {}
    for i, fmt in enumerate(formats):
        seed_f = i
        gT = games.filter(pl.col("fmt") == fmt)
        split = validate._holdout_split(gT)
        if split is None:
            fmt_res[fmt] = dict(status="skipped", reason=f"too few {fmt} events/games")
            continue
        hold_ids, t_cut, _nev, _nho = split
        g_tr = gT.filter(~pl.col("draft_id").is_in(hold_ids))
        g_ho = gT.filter(pl.col("draft_id").is_in(hold_ids))

        tmp_f = out / "tmp" / f"{sc}_{fmt}"
        (tmp_f / "m").mkdir(parents=True, exist_ok=True)

        # 1. M: the shipped chain refitted on the training events
        st_m = validate.chain_stages(fmt, games, g_tr, t_cut, space)
        prior_ok = set(st_m) - {fmt}
        fits_m = cardvalue.run_chain(st_m, space.n_card, space.n_feat, str(tmp_f / "m"), n_jobs, seed=seed_f + 100,
                                     prior_ok=prior_ok, opts=opts)
        first = first_stage_name(st_m)

        if selfcheck:
            (tmp_f / "zero").mkdir(parents=True, exist_ok=True)
            fits_z = run_chain_with_first_prior(st_m, np.zeros(space.n_card), space.n_card, space.n_feat, str(tmp_f / "zero"), n_jobs,
                                                seed_f + 100, prior_ok, opts)
            diffs = {n: float(np.max(np.abs(fits_z[n].beta_total - fits_m[n].beta_total))) for n in fits_m}
            mx = max(diffs.values())
            fmt_res[fmt] = dict(status="selfcheck", first_stage=first, max_abs_diff=mx, per_stage=diffs)
            print(f"[{sc} {fmt}] selfcheck first={first} stages={list(fits_m)} max|d beta_total|={mx:.3e} "
                  f"per stage={ {k: f'{v:.2e}' for k, v in diffs.items()} }", flush=True)
            if not mx < 1e-9:
                raise AssertionError(f"{sc} {fmt}: zero-prior arm path differs from M by {mx:.3e} (>= 1e-9)")
            continue

        prob_ho, _, _ = design.make_problem(g_ho, st_m[fmt].enc, space, clamp_time=True)
        p_m = expit(prob_ho.eta(fits_m[fmt].coef))
        y = prob_ho.y
        groups = np.unique(g_ho["draft_id"].to_numpy(), return_inverse=True)[1]
        is_first = validate.half_masks(g_ho)
        beta_zero = fits_m[first].u
        exposure = st_m[first].exposure
        b_m = fits_m[fmt].beta_total

        # 2. signals
        sealed_long_c = sealed_cell_long(runs_all, t_cut, audit, pools, inp.deck_cards, cards)
        counts_all = play_counts(sealed_long_c, cards)
        probs: dict[str, np.ndarray] = {}
        arms_res: dict[str, dict] = {}
        for arm in ARMS:
            lg = arm_long(arm, sealed_long_c, draft_all, t_cut)
            if lg is None or lg.height == 0:
                arms_res[arm] = dict(no_signal=True, reason="no runs" if lg is not None else "no registered Draft game files")
                continue
            cnt = counts_all if arm == "S_all" else play_counts(lg, cards)
            names = cnt["name"].to_list()
            n_el = cnt["n"].to_numpy().astype(float)
            z = signal_z(cnt["k"].to_numpy().astype(float), n_el)
            n_runs = int(lg["draft_id"].n_unique())
            prior_cards, info = pickprior.map_signal_to_betas(cards, space, exposure, beta_zero, z, n_el, names, min_taken=MIN_RUNS)
            if info["covered"] == 0:
                arms_res[arm] = dict(no_signal=True, reason="no card with a signal", n_runs=n_runs, map_info=info)
                continue
            td = tmp_f / arm
            td.mkdir(parents=True, exist_ok=True)
            fits_a = run_chain_with_first_prior(st_m, prior_cards, space.n_card, space.n_feat, str(td), n_jobs, seed_f + 100, prior_ok, opts)
            p_a = expit(prob_ho.eta(fits_a[fmt].coef))
            probs[arm] = p_a
            m = csc.paired_gain_metrics(y, p_m, p_a, groups, is_first, n_rep=N_REP, seed=seed_f)
            b_a = fits_a[fmt].beta_total
            m.update(
                map_info=info, n_runs=n_runs, n_cards_n30=int((n_el >= MIN_RUNS).sum()), signal_used=bool(info["covered"] >= pickprior.MIN_SIGNAL_CARDS),
                beta_corr=float(np.corrcoef(b_a, b_m)[0, 1]),
                movers=top_movers(names_by_col, b_m, b_a), tracked=tracked_values(col_of_name, b_m, b_a),
                sigma_first=float(fits_a[first].sigma), sigma_first_M=float(fits_m[first].sigma),
            )
            arms_res[arm] = m
            del fits_a

        # 3. trap check (reported)
        trap = None
        if counts_all.height:
            rare = counts_all.filter((pl.col("n") >= TRAP_MIN_N) & (pl.col("k") / pl.col("n") < TRAP_RATE))
            rare_ids = [id_of_name[n] for n in rare["name"].to_list() if n in id_of_name]
            if rare_ids:
                dks = inp.deck_cards.filter(pl.col("card_id").is_in(rare_ids))["deck_key"].unique()
                mask = g_ho["deck_key"].is_in(dks.implode()).to_numpy()
                if mask.sum() >= 2:
                    trap = dict(n_rare_cards=len(rare_ids), n_games=int(mask.sum()),
                                M=mean_resid_ci(y[mask], p_m[mask], groups[mask], N_REP, seed_f + 11))
                    for arm, p_a in probs.items():
                        trap[arm] = mean_resid_ci(y[mask], p_a[mask], groups[mask], N_REP, seed_f + 11)
                else:
                    trap = dict(n_rare_cards=len(rare_ids), n_games=int(mask.sum()), note="too few hold-out games")
            else:
                trap = dict(n_rare_cards=0, n_games=0, note="no rare-play card")

        # 4. skill vs rank (positive = D_rank better) and Draft-over-Sealed pairs
        svr = None
        if "D_skill" in probs and "D_rank" in probs:
            svr = csc.paired_gain_metrics(y, probs["D_skill"], probs["D_rank"], groups, is_first, n_rep=N_REP, seed=seed_f)
        d_vs_s = {}
        for d_arm in DRAFT_ARMS:
            for s_arm in SEALED_ARMS:
                if d_arm in probs and s_arm in probs:
                    d_vs_s[f"{d_arm}|{s_arm}"] = csc.paired_gain_metrics(y, probs[s_arm], probs[d_arm], groups, is_first,
                                                                          n_rep=N_REP, seed=seed_f)

        fmt_res[fmt] = dict(status="ok", first_stage=first, n_holdout_games=int(len(y)), n_holdout_events=int(groups.max()) + 1,
                            t_cut=str(t_cut), arms=arms_res, trap=trap, skill_vs_rank=svr, d_vs_s=d_vs_s)
        parts = []
        for arm in ARMS:
            a = arms_res[arm]
            if a.get("no_signal"):
                parts.append(f"{arm}=n/a")
            else:
                parts.append(f"{arm}={a['gain']*1e4:+6.2f} [{a['lo']*1e4:+6.2f},{a['hi']*1e4:+6.2f}] "
                             f"(cov={a['map_info']['covered']} r2={a['map_info']['r2']:.2f} sl={a['map_info']['slope']:+.3f})")
        print(f"[{sc} {fmt}] ho={len(y):5d} first={first} | " + " | ".join(parts), flush=True)
        cardvalue.cleanup(str(tmp_f))

    return dict(set=sc, seconds=round(time.time() - t0, 1), formats=fmt_res)


# ------------------------------------------------------------------------------------------------------ pooling and choice
def summarize(per_set: dict[str, dict], getter) -> dict:
    """cold_start summarize_arm over any per-cell metrics getter (fr -> metrics dict with halves, or None = no signal)."""
    out_fmt = {}
    for fmt in ("bo1", "bo3"):
        cells, excluded = {}, []
        for sc, r in per_set.items():
            fr = r["formats"].get(fmt)
            if not fr or fr.get("status") != "ok":
                excluded.append(dict(set=sc, reason=(fr or {}).get("reason", "absent")))
                continue
            m = getter(fr)
            if m is None or m.get("no_signal"):
                excluded.append(dict(set=sc, reason=(m or {}).get("reason", "no signal")))
                continue
            cell = pooling.make_cell(m, m["halves"])
            if cell is None:
                excluded.append(dict(set=sc, reason="zero-width or non-finite CI"))
            else:
                cells[sc] = cell
        if not cells:
            out_fmt[fmt] = dict(n_sets=0, excluded=excluded)
            continue
        raw_pool = pooling.pool_parts(list(cells.values()))
        out_fmt[fmt] = dict(n_sets=len(cells), sets=sorted(cells), excluded=excluded,
                            raw=dict(pooled=raw_pool, verdict=pooling.verdict(raw_pool)),
                            se_floored=csc.pool_with_floor(cells, SE_FLOOR))
    qual_raw = pooling.qualifies({f: v["raw"]["verdict"] for f, v in out_fmt.items() if v.get("n_sets")})
    qual_fl = pooling.qualifies({f: v["se_floored"]["verdict"] for f, v in out_fmt.items() if v.get("n_sets")})
    return dict(formats=out_fmt, passes_rule=bool(qual_raw and qual_fl))


def _flip(pooled: dict) -> dict:
    return {p: dict(pooled[p], mean=-pooled[p]["mean"], lo=-pooled[p]["hi"], hi=-pooled[p]["lo"]) for p in pooling.PARTS}


def skill_vs_rank_answer(sm: dict) -> dict:
    """RULE.md: D_rank better only if the pooled (D_rank over D_skill) gain passes the 'better' verdict, raw and floored; the reverse
    with signs flipped means skill is better; anything else: the data cannot tell them apart."""
    ans = {}
    for fmt, fv in sm["formats"].items():
        if not fv.get("n_sets"):
            continue
        rank_b = fv["raw"]["verdict"]["better"] and fv["se_floored"]["verdict"]["better"]
        sk_b = (pooling.verdict(_flip(fv["raw"]["pooled"]))["better"]
                and pooling.verdict(_flip(fv["se_floored"]["pooled"]))["better"])
        ans[fmt] = "D_rank better" if rank_b else ("D_skill better" if sk_b else "cannot tell apart")
    return ans


def pooled_trap(per_set: dict[str, dict]) -> dict:
    out = {}
    for fmt in ("bo1", "bo3"):
        res = {}
        for who in ("M",) + ARMS:
            items = {sc: r["formats"][fmt]["trap"][who] for sc, r in per_set.items()
                     if r["formats"].get(fmt, {}).get("status") == "ok" and (r["formats"][fmt].get("trap") or {}).get(who)}
            pm = pool_means(items) if items else None
            if pm:
                res[who] = pm
        out[fmt] = res
    return out


def choose_arm(pooled: dict, per_set: dict[str, dict]) -> dict:
    passing = [a for a in ARMS if pooled[a]["passes_rule"]]
    score = {}
    for a in passing:
        gains = [fv["raw"]["pooled"]["whole"]["mean"] for fv in pooled[a]["formats"].values() if fv.get("n_sets")]
        score[a] = float(np.mean(gains)) if gains else float("-inf")
    out = dict(passing=passing, score_mean_whole_gain=score)
    if not passing:
        out.update(chosen=None, why="no arm passes the rule")
        return out
    d_vs_s: dict[str, dict] = {}
    for d_arm in [a for a in passing if a in DRAFT_ARMS]:
        for s_arm in [a for a in passing if a in SEALED_ARMS]:
            sm = summarize(per_set, lambda fr, k=f"{d_arm}|{s_arm}": fr["d_vs_s"].get(k))
            per_fmt = {}
            for fmt, fv in sm["formats"].items():
                if not fv.get("n_sets"):
                    continue
                ok = True
                rec = {}
                for tag, pooled_whole in (("raw", fv["raw"]["pooled"]["whole"]), ("floored", fv["se_floored"]["pooled"]["whole"])):
                    rec[tag] = dict(mean=pooled_whole["mean"], lo=pooled_whole["lo"], hi=pooled_whole["hi"])
                    ok = ok and pooled_whole["mean"] >= MIN_GAIN and pooled_whole["lo"] > 0
                rec["beats"] = bool(ok)
                per_fmt[fmt] = rec
            d_vs_s[f"{d_arm}|{s_arm}"] = dict(formats=per_fmt, beats=bool(per_fmt) and all(v["beats"] for v in per_fmt.values()))
    out["d_vs_s"] = d_vs_s
    eligible = [a for a in passing if a in SEALED_ARMS]
    for d_arm in [a for a in passing if a in DRAFT_ARMS]:
        need = [d_vs_s[f"{d_arm}|{s}"]["beats"] for s in passing if s in SEALED_ARMS]
        if all(need):          # vacuously true when no Sealed arm passes
            eligible.append(d_arm)
    chosen = max(eligible, key=lambda a: score[a])
    out.update(chosen=chosen, eligible=eligible,
               why=("largest mean pooled whole-hold-out gain over bo1 and bo3 among passing arms; a Draft arm counts only when it beats "
                    "every passing Sealed arm by >= 1e-4 with pooled 95% CI > 0 in every format where both exist (raw and SE-floored)"))
    return out


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def fmt_pool_line(arm: str, sm: dict) -> list[str]:
    lines = [f"=== {arm} (PASS={sm['passes_rule']}) ==="]
    for fmt, fv in sm["formats"].items():
        if fv.get("n_sets"):
            rw, ra, rb = (fv["raw"]["pooled"][p] for p in ("whole", "A", "B"))
            rv, flv = fv["raw"]["verdict"], fv["se_floored"]["verdict"]
            lines.append(f"  {fmt} ({fv['n_sets']} sets): whole={rw['mean']*1e4:+6.2f} [{rw['lo']*1e4:+6.2f},{rw['hi']*1e4:+6.2f}] | "
                         f"A={ra['mean']*1e4:+6.2f} [{ra['lo']*1e4:+6.2f},{ra['hi']*1e4:+6.2f}] | "
                         f"B={rb['mean']*1e4:+6.2f} (lo90={rv['B_lo90']*1e4:+6.2f}) | raw={rv['label']} | floored={flv['label']}")
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", type=Path, nargs="?", default=Path(tempfile.gettempdir()) / "sealed_lab_play_prior")
    ap.add_argument("--sets", nargs="+", default=ALL_SETS)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--selfcheck", action="store_true", help="zero-prior arm path must equal M's card values (< 1e-9), then exit")
    a = ap.parse_args(argv)
    check_home_not_real()
    out = a.out.resolve()
    check_out_not_home(out)
    out.mkdir(parents=True, exist_ok=True)

    per_set: dict[str, dict] = {}
    t_all = time.time()
    with glm.shared_pool(a.jobs):
        for sc in a.sets:
            sc = sc.upper()
            done = out / f"per_set_{sc}.json"     # a finished set is kept, so a crash in a later set loses nothing
            if not a.selfcheck and done.exists():
                per_set[sc] = json.loads(done.read_text(encoding="utf-8"))
                print(f"[{sc}] reused {done.name}", flush=True)
                continue
            per_set[sc] = run_set(sc, out, a.jobs, selfcheck=a.selfcheck)
            if not a.selfcheck:
                done.write_text(json.dumps(per_set[sc], indent=1, default=_json_default), encoding="utf-8")
            cardvalue.cleanup(str(out / "tmp"))
    if a.selfcheck:
        mx = max((fr["max_abs_diff"] for r in per_set.values() for fr in r["formats"].values() if fr.get("status") == "selfcheck"), default=float("nan"))
        print(f"\nselfcheck OK: max |d beta_total| over all sets/formats/stages = {mx:.3e} ({time.time() - t_all:.1f}s)")
        return 0

    pooled = {arm: summarize(per_set, lambda fr, arm=arm: (fr.get("arms") or {}).get(arm)) for arm in ARMS}
    svr = summarize(per_set, lambda fr: fr.get("skill_vs_rank"))
    svr["answer"] = skill_vs_rank_answer(svr)
    summary = dict(
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        total_seconds=round(time.time() - t_all, 1),
        se_floor=SE_FLOOR, min_gain=MIN_GAIN, min_runs=MIN_RUNS, n_rep=N_REP,
        per_set=per_set,
        pooled=pooled,
        pooled_skill_vs_rank=svr,
        pooled_trap=pooled_trap(per_set),
        choice=choose_arm(pooled, per_set),
    )
    out_json = out / "play_prior_results.json"
    out_json.write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    print(f"\nWrote {out_json} in {summary['total_seconds']:.1f}s")
    for arm in ARMS:
        for line in fmt_pool_line(arm, pooled[arm]):
            print(line)
    for line in fmt_pool_line("D_rank minus D_skill (positive = rank better)", svr):
        print(line)
    print("skill vs rank:", svr["answer"])
    print("choice:", summary["choice"].get("chosen"), "| passing:", summary["choice"]["passing"])
    return 0


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    sys.exit(main())
