"""What the replay (turn-by-turn) data say about deck strength, and a check of the mana simulator against real play.

Entry points (all JSON-friendly):
  * `replay_summary(set_code)`        -> dict with every section below (cached in cache/replay/summary.json)
  * `replay_card_stats(set_code, fmt)` -> {card_id: {...}} per-card tempo / cast statistics
  * `render_replay_section(summary)`   -> HTML fragment (same CSS classes as the model report)

Sections of the summary:
  simulator   mana simulator versus real games: P(<3 lands in play after own turn 3), mean mana spent T1-T4 and T1-T7,
              P(>=9 lands in the first 16 cards), lands in the first 7 cards, by event and land count, with cluster
              (draft_id) bootstrap intervals; per-deck calibration slopes of the simulated screw probability.
  tempo       win rate by mana spent in own turns 1-4 and by the number of spells cast on own turns 2, 3, 4 ("curve
              realised"), adjusted for pilot skill bucket, play/draw and mulligans (logistic, cluster-robust SE);
              the same with deck fixed effects; which deck features predict curving out.
  cards       per-card average turn cast, cast rate when seen, win rate when cast vs when in the deck but not cast.

Caveats that apply throughout (also repeated in the report): the replay files only record *completed* own turns, so every
turn-k statistic is conditional on the game lasting past turn k; card-level win rates condition on what happened in the game
(winners get to cast more cards), they are descriptive, not causal.
"""
from __future__ import annotations

import concurrent.futures as cf
import html
import json
import logging
import math
import multiprocessing
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import numpy as np
import polars as pl
from scipy import stats as sps
from scipy.special import expit

from sealed_lab import config
from sealed_lab.ingest import replay
from sealed_lab.types import Cancelled

log = logging.getLogger(__name__)
Progress = Callable[[float, str], None]

STATS_VERSION = 4
FMT_OF_EVENT = {"TradSealed": "bo3", "Sealed": "bo1", "TradDraft": "draft", "PremierDraft": "draft"}
EVENT_ORDER = ["TradSealed", "Sealed", "TradDraft", "PremierDraft"]
EVENT_LABEL = {"TradSealed": "Bo3 TradSealed", "Sealed": "Bo1 Sealed", "TradDraft": "Trad Draft (Bo3)",
               "PremierDraft": "Premier Draft (Bo1)"}
LAND_BINS = (15, 16, 17, 18)
SEED = 20261024


def _check(cancel: threading.Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise Cancelled()


def _num(x):
    """JSON-safe scalar: numpy -> python, NaN/inf -> None."""
    if x is None:
        return None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return x if math.isfinite(x) else None
    return x


def _clean(o):
    if isinstance(o, dict):
        return {(int(k) if isinstance(k, (np.integer,)) else k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return [_clean(v) for v in o.tolist()]
    return _num(o)


# ---------------------------------------------------------------------------------------------
# small statistics toolkit (no statsmodels)
# ---------------------------------------------------------------------------------------------

def wilson(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _codes(x) -> np.ndarray:
    return np.unique(np.asarray(x), return_inverse=True)[1].astype(np.int64)


def cluster_ratio(num, den, groups, n_boot: int = 200, seed: int = SEED) -> tuple[float, float, float]:
    """Ratio of sums sum(num)/sum(den) with a 95% cluster bootstrap interval (clusters = `groups`)."""
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    if len(num) == 0 or den.sum() <= 0:
        return (float("nan"),) * 3
    g = _codes(groups)
    G = int(g.max()) + 1
    sn = np.bincount(g, weights=num, minlength=G)
    sd = np.bincount(g, weights=den, minlength=G)
    est = float(sn.sum() / sd.sum())
    if G < 2:
        return est, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, G, G)
        d = sd[idx].sum()
        out[b] = sn[idx].sum() / d if d > 0 else np.nan
    lo, hi = np.nanpercentile(out, [2.5, 97.5])
    return est, float(lo), float(hi)


def _cluster_meat(S: np.ndarray, g: np.ndarray) -> np.ndarray:
    """sum over clusters of (cluster score)(cluster score)^T for per-observation scores S (n, p)."""
    order = np.argsort(g, kind="stable")
    gs = g[order]
    starts = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1]])
    Sg = np.add.reduceat(S[order], starts, axis=0)
    return Sg.T @ Sg


def logit_fit(X: np.ndarray, y: np.ndarray, groups, ridge: float = 1e-6, max_iter: int = 60) -> dict:
    """Logistic regression by Newton steps with cluster-robust (sandwich) covariance."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape
    beta = np.zeros(p)
    mu0 = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
    beta[0] = math.log(mu0 / (1 - mu0))
    for _ in range(max_iter):
        mu = expit(X @ beta)
        W = mu * (1 - mu)
        grad = X.T @ (y - mu) - ridge * beta
        H = (X * W[:, None]).T @ X + ridge * np.eye(p)
        step = np.linalg.solve(H, grad)
        beta = beta + step
        if np.abs(step).max() < 1e-9:
            break
    mu = expit(X @ beta)
    W = mu * (1 - mu)
    H = (X * W[:, None]).T @ X + ridge * np.eye(p)
    bread = np.linalg.inv(H)
    g = _codes(groups)
    G = int(g.max()) + 1
    meat = _cluster_meat(X * (y - mu)[:, None], g)
    cov = bread @ meat @ bread * (G / max(G - 1, 1))
    return dict(beta=beta, cov=cov, se=np.sqrt(np.clip(np.diag(cov), 0, None)), n=n, n_clusters=G)


def ols_fit(X: np.ndarray, y: np.ndarray, groups, ridge: float = 1e-9) -> dict:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape
    A = X.T @ X + ridge * np.eye(p)
    bread = np.linalg.inv(A)
    beta = bread @ (X.T @ y)
    r = y - X @ beta
    g = _codes(groups)
    G = int(g.max()) + 1
    meat = _cluster_meat(X * r[:, None], g)
    cov = bread @ meat @ bread * (G / max(G - 1, 1)) * ((n - 1) / max(n - p, 1))
    return dict(beta=beta, cov=cov, se=np.sqrt(np.clip(np.diag(cov), 0, None)), n=n, n_clusters=G)


def ame(fit: dict, X1: np.ndarray, X0: np.ndarray) -> tuple[float, float, float, float]:
    """Average marginal effect mean(p(X1) - p(X0)) of a logistic fit, with delta-method 95% interval -> (est, se, lo, hi)."""
    b = fit["beta"]
    p1 = expit(X1 @ b)
    p0 = expit(X0 @ b)
    est = float(np.mean(p1 - p0))
    grad = ((p1 * (1 - p1))[:, None] * X1 - (p0 * (1 - p0))[:, None] * X0).mean(axis=0)
    se = float(math.sqrt(max(grad @ fit["cov"] @ grad, 0.0)))
    return est, se, est - 1.96 * se, est + 1.96 * se


def marginal_mean(fit: dict, X: np.ndarray) -> tuple[float, float]:
    """Mean predicted probability over the rows of X and its delta-method SE."""
    b = fit["beta"]
    p = expit(X @ b)
    grad = ((p * (1 - p))[:, None] * X).mean(axis=0)
    return float(p.mean()), float(math.sqrt(max(grad @ fit["cov"] @ grad, 0.0)))


def coef_row(fit: dict, i: int) -> dict:
    b, se = float(fit["beta"][i]), float(fit["se"][i])
    return dict(est=b, se=se, lo=b - 1.96 * se, hi=b + 1.96 * se)


# ---------------------------------------------------------------------------------------------
# data assembly
# ---------------------------------------------------------------------------------------------

STRUCT_COLS = ["n_lands", "n_creatures", "n_noncreature", "curve_le1", "curve_2", "curve_3", "curve_4", "curve_5", "curve_ge6",
               "mean_spell_mv"]
SIM_COLS = {"p_screw": "sim_screw", "p_flood": "sim_flood", "p_color_screw": "sim_color", "mana_spent": "sim_mana7",
            "mull_rate": "sim_mull", "lands7_kept": "sim_l7k", "lands7_all": "sim_l7all"}


def load_phi(set_code: str, cards: pl.DataFrame, decks: pl.DataFrame, deck_cards: pl.DataFrame, n_jobs: int | None = None,
             cancel: threading.Event | None = None, progress: Progress | None = None) -> pl.DataFrame:
    """Per-deck features (structure + simulator) from the model's phi cache (computed if missing). Columns: deck_key,
    deck n_lands/n_cards, STRUCT_COLS and SIM_COLS."""
    from sealed_lab.model import pipeline, structure

    inp = SimpleNamespace(decks=decks, cards=cards, deck_cards=deck_cards)
    keys, table = pipeline.phi_for_corpus(inp, n_jobs, cancel, progress, config.cache_dir(set_code))
    names = list(structure.TABLE_NAMES)
    cols: dict[str, np.ndarray] = {"deck_key": keys}
    for n in STRUCT_COLS:
        cols[n] = table[:, names.index(n)]
    for n, out in SIM_COLS.items():
        cols[out] = table[:, names.index(n)]
    df = pl.DataFrame(cols).rename({"n_lands": "phi_lands"})
    return df.join(decks.select("deck_key", "n_lands", "n_cards"), on="deck_key", how="left")


def idinfo_table(set_code: str, cards: pl.DataFrame, idmap: dict) -> pl.DataFrame:
    """Arena id -> card attributes."""
    c = cards.sort("card_id")
    ids = np.array(list(idmap.keys()), dtype=np.int32)
    cid = np.array([idmap[int(i)] for i in ids], dtype=np.int32)
    return pl.DataFrame({"id": ids, "card_id": cid}).join(
        c.select("card_id", "name", "rarity", "mana_value", "is_land", "is_basic", "is_creature"), on="card_id", how="left")


def _sum_cols(stem: str, ks: range) -> pl.Expr:
    return pl.sum_horizontal([pl.col(f"{stem}_{k}").fill_null(0).cast(pl.Int16) for k in ks])


def add_derived(df: pl.DataFrame) -> pl.DataFrame:
    """Real-play quantities per game (null when the needed turns were not completed)."""
    td = pl.col("turns_done")
    spells = [(pl.col(f"cre_{k}").fill_null(0) + pl.col(f"ncre_{k}").fill_null(0) + pl.col(f"inst_{k}").fill_null(0)) for k in (2, 3, 4)]
    hits = sum(((s > 0).cast(pl.Int8) for s in spells), pl.lit(0, dtype=pl.Int8))
    return df.with_columns(
        pl.when(td >= 3).then(pl.col("lands_3") < 3).otherwise(None).alias("screw3"),
        pl.when(td >= 4).then(_sum_cols("mana", range(1, 5))).otherwise(None).cast(pl.Float64).alias("mana4"),
        pl.when(td >= 7).then(_sum_cols("mana", range(1, 8))).otherwise(None).cast(pl.Float64).alias("mana7"),
        pl.when(td >= 4).then(hits).otherwise(None).alias("curve_hits"),
        pl.when(td >= 4).then(pl.col("lands_4")).otherwise(None).alias("lands4"),
        pl.col("won").cast(pl.Float64).alias("win"),
    )


def build_frame(set_code: str, events: list[str] | None = None, n_jobs: int | None = None,
                cancel: threading.Event | None = None, progress: Progress | None = None) -> tuple[pl.DataFrame, dict, dict]:
    """Replay games joined to the corpus (deck, skill) and to the per-deck simulator / structure features.

    Returns (frame, corpus tables dict, info dict)."""
    from sealed_lab.ingest.corpus import load_corpus

    set_code = set_code.upper()
    games = replay.load_games(set_code, events)
    c = load_corpus(set_code)
    cg = c["games"].select("game_key", "deck_key", "draft_id", "match_number", "game_number", "game_time", "fmt", "skill").collect()
    linked = replay.link_corpus(games, cg)
    n_all = linked.height
    linked = linked.filter(pl.col("deck_key").is_not_null())
    cards = c["cards"]
    decks = c["decks"]
    deck_cards = c["deck_cards"].collect()
    _check(cancel)
    phi = load_phi(set_code, cards, decks, deck_cards, n_jobs, cancel, progress)
    df = linked.join(phi, on="deck_key", how="left")
    df = add_derived(df)
    info = dict(n_replay_games=int(n_all), n_linked=int(linked.height), n_with_phi=int(df["sim_screw"].is_not_null().sum()))
    return df, dict(cards=cards, decks=decks, deck_cards=deck_cards, corpus=c), info


# ---------------------------------------------------------------------------------------------
# first cards seen (lands among the first 7 / first 16 cards)
# ---------------------------------------------------------------------------------------------

def first_cards_lands(draws: pl.DataFrame, idinfo: pl.DataFrame) -> pl.DataFrame:
    """Per game: n7, l7 (lands among candidate_hand_1), lands9 (lands among the first 9 drawn cards, null if the
    cumulative draw count never equals 9 exactly) and isl1 (1 if the first card drawn is a land). draws: gid, turn, kind, id, n."""
    il = idinfo.select("id", pl.col("is_land").cast(pl.Int8).alias("isl"))
    d = draws.join(il, on="id", how="left").with_columns(pl.col("isl").fill_null(0), pl.col("n").cast(pl.Int32))
    h7 = (d.filter(pl.col("kind") == 3).group_by("gid")
          .agg(pl.col("n").sum().alias("n7"), (pl.col("n") * pl.col("isl")).sum().alias("l7")))
    dr = (d.filter(pl.col("kind") == 1).group_by(["gid", "turn"])
          .agg(pl.col("n").sum().alias("n"), (pl.col("n") * pl.col("isl")).sum().alias("l"))
          .sort(["gid", "turn"])
          .with_columns(pl.col("n").cum_sum().over("gid").alias("cn"), pl.col("l").cum_sum().over("gid").alias("cl")))
    nine = dr.filter(pl.col("cn") == 9).select("gid", pl.col("cl").alias("lands9"))
    # the first card drawn (only when the first draw event is a single card): is it a land?
    d1 = d.filter(pl.col("kind") == 1)
    t1 = d1.group_by("gid").agg(pl.col("turn").min().alias("t1"))
    fr = (d1.join(t1, on="gid").filter(pl.col("turn") == pl.col("t1")).group_by("gid")
          .agg(pl.col("n").sum().alias("n1"), (pl.col("n") * pl.col("isl")).sum().alias("l1")))
    first_draw = fr.filter(pl.col("n1") == 1).select("gid", pl.col("l1").alias("isl1"))
    return h7.join(nine, on="gid", how="left").join(first_draw, on="gid", how="left")


# ---------------------------------------------------------------------------------------------
# (a) simulator versus reality
# ---------------------------------------------------------------------------------------------

def _play_draw_weights(on_play: np.ndarray) -> np.ndarray:
    """Weights that re-balance a sample to 50% on the play / 50% on the draw (the simulator averages both)."""
    p = float(on_play.mean())
    if p <= 0 or p >= 1:
        return np.ones(len(on_play))
    return np.where(on_play, 0.5 / p, 0.5 / (1 - p))


def compare(y: np.ndarray, sim: np.ndarray, groups, on_play: np.ndarray, kind: str = "rate", n_boot: int = 200,
            seed: int = SEED, norm: float = 1.0) -> dict:
    """Real mean of y (re-weighted to 50/50 play/draw) vs the mean simulated value for the same games.

    kind 'rate' adds a Wilson interval of the unweighted rate; every interval is a cluster bootstrap by `groups`."""
    n = len(y)
    if n == 0:
        return dict(n=0)
    w = _play_draw_weights(on_play)
    est, lo, hi = cluster_ratio(w * y, w, groups, n_boot, seed)
    s = float(np.average(sim, weights=w))
    out = dict(n=int(n), real=est / norm, lo=lo / norm, hi=hi / norm, sim=s / norm, diff=(est - s) / norm,
               diff_lo=(lo - s) / norm, diff_hi=(hi - s) / norm, raw=float(y.mean()) / norm,
               play_share=float(on_play.mean()))
    return out


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def calibration_slope(y: np.ndarray, sim_p: np.ndarray, groups, on_play: np.ndarray, extra: np.ndarray | None = None) -> dict:
    """Logistic regression of the real outcome on the logit of the simulated probability (+ play/draw [+ extras])."""
    cols = [np.ones(len(y)), _logit(sim_p), on_play.astype(float)]
    if extra is not None:
        cols += [extra]
    X = np.column_stack(cols)
    fit = logit_fit(X, y.astype(float), groups)
    return dict(n=int(len(y)), slope=coef_row(fit, 1), intercept=coef_row(fit, 0), n_clusters=int(fit["n_clusters"]),
                sd_sim=float(np.std(sim_p)))


def decile_table(y: np.ndarray, sim_p: np.ndarray, groups, on_play: np.ndarray, n_bins: int = 5) -> list[dict]:
    """Games grouped by quantiles of the simulated probability: real rate (cluster CI, 50/50 play-draw) vs simulated."""
    q = np.unique(np.quantile(sim_p, np.linspace(0, 1, n_bins + 1)))
    if len(q) < 3:
        return []
    b = np.clip(np.searchsorted(q, sim_p, side="right") - 1, 0, len(q) - 2)
    rows = []
    for i in range(len(q) - 1):
        m = b == i
        if m.sum() < 30:
            continue
        r = compare(y[m].astype(float), sim_p[m], np.asarray(groups)[m], on_play[m], n_boot=120)
        r["sim_lo"] = float(q[i])
        r["sim_hi"] = float(q[i + 1])
        rows.append(r)
    return rows


def _sim_cum_worker(task: dict) -> np.ndarray:
    """Spawn worker: simulated mean mana spent over own turns 1..T for a list of decks (module-level N_TURNS patched)."""
    from sealed_lab.model import manasim

    ca = task["ca"]
    off, cid, cn = task["offsets"], task["card_ids"], task["counts"]
    old = manasim.N_TURNS
    manasim.N_TURNS = int(task["turns"])
    out = np.empty(len(off) - 1)
    try:
        for i in range(len(off) - 1):
            a, b = off[i], off[i + 1]
            s = manasim.simulate(dict(zip(cid[a:b].tolist(), cn[a:b].tolist())), ca, int(task["n_trials"]))
            out[i] = s["mana_spent"] * 28.0
    finally:
        manasim.N_TURNS = old
    return out


def sim_cum_mana(decks: list[dict[int, int]], cards: pl.DataFrame, turns: int, n_jobs: int | None = None,
                 cancel: threading.Event | None = None, chunk: int = 150) -> np.ndarray:
    """Simulated mean total mana spent in own turns 1..`turns` for each deck (needs turns >= 3), in isolated processes
    because the simulator's turn horizon is a module constant."""
    from sealed_lab.model.manasim import DEFAULT_TRIALS, card_arrays

    if not decks:
        return np.zeros(0)
    ca = card_arrays(cards)
    tasks = []
    for s in range(0, len(decks), chunk):
        part = decks[s:s + chunk]
        off = np.cumsum([0] + [len(d) for d in part])
        cid = np.array([k for d in part for k in d.keys()], dtype=np.int64)
        cn = np.array([v for d in part for v in d.values()], dtype=np.int64)
        tasks.append(dict(ca=ca, offsets=off, card_ids=cid, counts=cn, turns=turns, n_trials=DEFAULT_TRIALS))
    n_jobs = int(n_jobs or os.cpu_count() or 1)
    if n_jobs <= 1 or len(tasks) == 1:
        return np.concatenate([_sim_cum_worker(t) for t in tasks])
    from sealed_lab.model.glm import map_parallel

    res = map_parallel(_sim_cum_worker, tasks, n_jobs, cancel=cancel)
    return np.concatenate(res)


def _hyper_cache(vals_fn, *arrays) -> np.ndarray:
    """Evaluate vals_fn(*key) once per unique key row of `arrays` and broadcast back."""
    key = np.stack([np.asarray(a, dtype=np.int64) for a in arrays], axis=1)
    uk, inv = np.unique(key, axis=0, return_inverse=True)
    vals = np.array([vals_fn(*[int(x) for x in row]) for row in uk], dtype=float)
    return vals[inv.ravel()]


def _hyper_per_game(n_cards: np.ndarray, n_lands: np.ndarray, draw: int, event: str) -> np.ndarray:
    """Exact P(>= 9 lands among `draw`=16 random cards) ('ge9') or P(first 7 outside 2-5 lands) ('tail7') per game."""
    def f(nc, nl):
        h = sps.hypergeom(nc, nl, draw)
        return h.sf(8) if event == "ge9" else (h.cdf(1) + h.sf(5))
    return _hyper_cache(f, n_cards, n_lands)


def _screw_given_hand(n_cards: np.ndarray, n_lands: np.ndarray, l7: np.ndarray, n_draws: np.ndarray) -> np.ndarray:
    """P(fewer than 3 lands among the 7 + `n_draws` cards seen by turn 3) given the actual kept 7-card hand."""
    def f(nc, nl, k, nd):
        if nl - k < 0 or nc - 7 < nd or nd < 0:
            return float("nan")
        h = sps.hypergeom(nc - 7, nl - k, nd)
        return float(h.cdf(2 - k)) if 2 - k >= 0 else 0.0
    return _hyper_cache(f, n_cards, n_lands, l7, n_draws)


def _flood_given_hand(n_cards: np.ndarray, n_lands: np.ndarray, l7: np.ndarray) -> np.ndarray:
    """P(>= 9 lands among the first 16 cards) given the actual first 7 cards (the next 9 are random draws)."""
    def f(nc, nl, k):
        if nl - k < 0 or nc - 7 < 9:
            return float("nan")
        return float(sps.hypergeom(nc - 7, nl - k, 9).sf(8 - k)) if 8 - k >= 0 else 1.0
    return _hyper_cache(f, n_cards, n_lands, l7)


def mull_rule_table(e: pl.DataFrame) -> dict[tuple[bool, int], tuple[int, float]]:
    """Real mulligan probability by (on_play, lands in the first 7 cards): {(play, l7): (n, p)}."""
    d = e.filter(pl.col("n7").fill_null(0) == 7)
    out: dict[tuple[bool, int], tuple[int, float]] = {}
    if d.is_empty():
        return out
    g = d.group_by(["on_play", "l7"]).agg(pl.len().alias("n"), (pl.col("num_mulligans") > 0).mean().alias("p"))
    for r in g.iter_rows(named=True):
        out[(bool(r["on_play"]), int(r["l7"]))] = (int(r["n"]), float(r["p"]))
    return out


def screw_under_rule(n_lands: int, n_cards: int, rule: dict[tuple[bool, int], tuple[int, float]], n: int = 200_000,
                     seed: int = SEED) -> float:
    """P(fewer than 3 lands in play after turn 3) when the player mulligans with the empirical probabilities `rule`
    (London mulligan, up to 3, bottoming spells first unless the hand has more than 3 lands): an independent mini-simulation
    (50/50 play-draw, one draw per turn, a land every turn when available)."""
    rng = np.random.default_rng([seed, n_lands, n_cards])
    total = 0.0
    for play in (True, False):
        pm = np.zeros(8)
        for k in range(8):
            cnt, p = rule.get((play, k), (0, 0.0))
            pm[k] = p if cnt >= 20 else (1.0 if (k <= 1 or k >= 6) else 0.0)
        l7 = rng.hypergeometric(n_lands, n_cards - n_lands, 7, size=n)
        m = np.zeros(n, dtype=np.int64)
        for _ in range(3):
            more = rng.random(n) < pm[l7]
            if not more.any():
                break
            new = rng.hypergeometric(n_lands, n_cards - n_lands, 7, size=n)
            l7 = np.where(more, new, l7)
            m = m + more
        bottom_lands = np.minimum(m, np.maximum(l7 - 3, 0))
        kept_lands = l7 - bottom_lands
        nd = 2 if play else 3
        rest_good = np.maximum(n_lands - l7, 0)
        rest = n_cards - 7
        draws = rng.hypergeometric(rest_good, np.maximum(rest - rest_good, 0), nd)
        total += 0.5 * float(np.mean(kept_lands + draws <= 2))
    return total


def simulator_section(df: pl.DataFrame, first: pl.DataFrame | None, cards: pl.DataFrame, deck_cards: pl.DataFrame,
                      n_jobs: int | None, cancel: threading.Event | None, progress: Progress | None,
                      mana_sim_games: int = 1200) -> dict:
    """Simulator versus real play, by event and land count, plus per-deck calibration."""
    from sealed_lab.model.manasim import DEFAULT_TRIALS

    if first is not None:
        df = df.join(first, on="gid", how="left")
    else:
        df = df.with_columns(pl.lit(None, dtype=pl.Int32).alias("n7"), pl.lit(None, dtype=pl.Int32).alias("l7"),
                             pl.lit(None, dtype=pl.Int32).alias("lands9"))
    df = df.filter(pl.col("sim_screw").is_not_null() & (pl.col("n_cards") == 40))
    rows: list[dict] = []
    calib: dict[str, dict] = {}
    keep_rules: dict[str, list[dict]] = {}
    draw_bias: dict[str, list[dict]] = {}
    events = [e for e in EVENT_ORDER if e in set(df["event"].unique().to_list())]
    rng = np.random.default_rng(SEED)
    dc_sorted = deck_cards.sort("deck_key")
    dk = dc_sorted["deck_key"].to_numpy()
    dcid = dc_sorted["card_id"].to_numpy()
    dn = dc_sorted["n"].to_numpy()

    def deck_of(key: int) -> dict[int, int]:
        a = np.searchsorted(dk, key, side="left")
        b = np.searchsorted(dk, key, side="right")
        return dict(zip(dcid[a:b].tolist(), dn[a:b].astype(int).tolist()))

    for ei, ev in enumerate(events):
        _check(cancel)
        if progress:
            progress(ei / max(len(events), 1), f"Simulator vs real: {ev}")
        e_ev = df.filter(pl.col("event") == ev)
        e = e_ev.filter(pl.col("n_lands").is_in(list(LAND_BINS)))
        if e.is_empty():
            continue
        grp = e["draft_id"].to_numpy()
        op = e["on_play"].to_numpy()
        nl = e["n_lands"].to_numpy()
        nm = e["num_mulligans"].to_numpy()
        n7 = e["n7"].fill_null(0).to_numpy()
        l7a = e["l7"].fill_null(0).to_numpy()
        nd3_all = e.select(pl.sum_horizontal([pl.col(f"drawn_{k}").fill_null(0).cast(pl.Int32) for k in (1, 2, 3)]).alias("x"))["x"].to_numpy()
        isl1 = e["isl1"].to_numpy() if "isl1" in e.columns else np.full(e.height, np.nan)
        # are the draws random given the opening hand? P(first draw is a land) after a kept 7-card hand
        db = []
        for k in range(1, 7):
            mk = (n7 == 7) & (nm == 0) & (l7a == k) & np.isfinite(isl1.astype(float))
            if mk.sum() >= 100:
                exp = (nl[mk] - k) / 33.0
                r_ = compare(isl1[mk].astype(float), exp, grp[mk], op[mk], n_boot=100)
                r_["lands_in_hand"] = k
                db.append(r_)
        if db:
            draw_bias[ev] = db
        rule = mull_rule_table(e)
        keep_rules[ev] =[dict(on_play=k[0], lands=k[1], n=v[0], p_mull=v[1], ci=list(wilson(v[1] * v[0], v[0])))
                          for k, v in sorted(rule.items())]
        # games whose decks are simulated for mana in turns 1-4 (turns 1-7 come from the cached features)
        t4_val = np.full(e.height, np.nan)
        ok4 = e["mana4"].is_not_null().to_numpy()
        pick = np.zeros(e.height, dtype=bool)
        for L in LAND_BINS:
            idx = np.flatnonzero(ok4 & (nl == L))
            if len(idx) > mana_sim_games:
                idx = rng.choice(idx, mana_sim_games, replace=False)
            pick[idx] = True
        sel = np.flatnonzero(pick)
        if len(sel):
            keys = e["deck_key"].to_numpy()[sel]
            uk, inv = np.unique(keys, return_inverse=True)
            sims = sim_cum_mana([deck_of(int(k)) for k in uk], cards, 4, n_jobs, cancel)
            t4_val[sel] = sims[inv]
        for L in list(LAND_BINS) + [None]:
            m = np.ones(e.height, dtype=bool) if L is None else (nl == L)
            if m.sum() < 80:
                continue
            row: dict = dict(event=ev, fmt=FMT_OF_EVENT.get(ev, "?"), n_lands=L, n_games=int(m.sum()))
            # mulligan rate (all games)
            row["mull"] = compare((nm[m] > 0).astype(float), e["sim_mull"].to_numpy()[m], grp[m], op[m])
            # P(< 3 lands in play after own turn 3): real vs simulator vs a simulation with the real keep rule
            k3 = m & e["screw3"].is_not_null().to_numpy()
            if k3.sum() >= 50:
                y = e["screw3"].to_numpy()[k3].astype(float)
                row["screw"] = compare(y, e["sim_screw"].to_numpy()[k3], grp[k3], op[k3])
                row["screw"]["wilson"] = list(wilson(float(y.sum()), len(y)))
                if L is None:
                    wts = {k: float(((nl == k) & k3).sum()) for k in LAND_BINS}
                    tot_w = sum(wts.values())
                    row["screw"]["rule"] = sum(wts[k] / tot_w * screw_under_rule(k, 40, rule) for k in LAND_BINS if wts[k])
                else:
                    row["screw"]["rule"] = screw_under_rule(L, 40, rule)
                # no-mulligan games: real vs random draws given the actual hand (isolates extra card flow)
                k0 = k3 & (n7 == 7) & (nm == 0)
                if k0.sum() >= 50:
                    y0 = e["screw3"].to_numpy()[k0].astype(float)
                    ph = _screw_given_hand(np.full(k0.sum(), 40), nl[k0], l7a[k0], np.where(op[k0], 2, 3))
                    ok = np.isfinite(ph)
                    row["screw_hand"] = compare(y0[ok], ph[ok], grp[k0][ok], op[k0][ok])
                    nd3 = nd3_all[k0]
                    ph2 = _screw_given_hand(np.full(k0.sum(), 40), nl[k0], l7a[k0], nd3)
                    ok2 = np.isfinite(ph2)
                    row["screw_hand_nd"] = compare(y0[ok2], ph2[ok2], grp[k0][ok2], op[k0][ok2])
            # mana spent in own turns
            k7 = m & e["mana7"].is_not_null().to_numpy()
            if k7.sum() >= 50:
                row["mana7"] = compare(e["mana7"].to_numpy()[k7], e["sim_mana7"].to_numpy()[k7] * 28.0, grp[k7], op[k7], norm=28.0)
            k4 = m & ~np.isnan(t4_val)
            if k4.sum() >= 50:
                row["mana4"] = compare(e["mana4"].to_numpy()[k4], t4_val[k4], grp[k4], op[k4])
                kl = k4 & e["mana7"].is_not_null().to_numpy()
                if kl.sum() >= 50:
                    row["mana4_long"] = compare(e["mana4"].to_numpy()[kl], t4_val[kl], grp[kl], op[kl])
            # lands in the first 7 cards and share of first hands outside 2-5 lands
            kk = m & (n7 == 7)
            if kk.sum() >= 50:
                l7 = l7a[kk].astype(float)
                nc_ = np.full(kk.sum(), 40)
                nl_ = nl[kk]
                row["first7"] = compare(l7, 7.0 * nl_ / nc_, grp[kk], op[kk])
                tail = ((l7 < 2) | (l7 > 5)).astype(float)
                row["first7_tail"] = compare(tail, e["sim_mull"].to_numpy()[kk], grp[kk], op[kk])
                row["first7_tail"]["exact"] = float(np.average(_hyper_per_game(nc_, nl_, 7, "tail7"), weights=_play_draw_weights(op[kk])))
            # lands among the first 16 cards (no mulligan, exactly 9 recorded draws): vs random draws given the hand
            k16 = m & e["lands9"].is_not_null().to_numpy() & (n7 == 7) & (nm == 0)
            if k16.sum() >= 50:
                l16 = (l7a[k16] + e["lands9"].to_numpy()[k16]).astype(float)
                ph = _flood_given_hand(np.full(k16.sum(), 40), nl[k16], l7a[k16])
                ok = np.isfinite(ph)
                row["flood16"] = compare((l16 >= 9).astype(float)[ok], ph[ok], grp[k16][ok], op[k16][ok])
                row["flood16"]["unconditional"] = float(np.average(_hyper_per_game(np.full(k16.sum(), 40), nl[k16], 16, "ge9"),
                                                                    weights=_play_draw_weights(op[k16])))
                row["flood16"]["sim_model"] = float(np.average(e["sim_flood"].to_numpy()[k16], weights=_play_draw_weights(op[k16])))
                row["flood16"]["mean_lands"] = float(l16.mean())
                row["flood16"]["expected_lands"] = float(np.mean(l7a[k16] + 9.0 * (nl[k16] - l7a[k16]) / 33.0))
            rows.append(row)
        # per-deck calibration (14-19 lands)
        ec = e_ev.filter(pl.col("n_lands").is_between(14, 19) & pl.col("screw3").is_not_null())
        if ec.height >= 500:
            gc, oc, nc = ec["draft_id"].to_numpy(), ec["on_play"].to_numpy(), ec["n_lands"].to_numpy()
            y = ec["screw3"].to_numpy().astype(float)
            sp = ec["sim_screw"].to_numpy()
            c = dict(n_games=int(ec.height), n_decks=int(ec["deck_key"].n_unique()))
            c["screw"] = calibration_slope(y, sp, gc, oc)
            sd_within = float(np.sqrt(np.mean([np.var(sp[nc == L]) for L in range(14, 20) if (nc == L).sum() > 20])))
            c["sim_sd_within_lands"] = sd_within
            if sd_within > 1e-3:
                c["screw_given_lands"] = calibration_slope(y, sp, gc, oc, extra=nc.astype(float))
            c["screw_deciles"] = decile_table(y, sp, gc, oc)
            c["by_lands"] = [dict(n_lands=L, **compare(y[nc == L], sp[nc == L], gc[nc == L], oc[nc == L], n_boot=100))
                             for L in range(14, 20) if (nc == L).sum() >= 100]
            em = e_ev.filter(pl.col("mana7").is_not_null() & pl.col("n_lands").is_between(14, 19))
            if em.height >= 500:
                X = np.column_stack([np.ones(em.height), em["sim_mana7"].to_numpy() * 28.0, em["on_play"].to_numpy().astype(float)])
                f = ols_fit(X, em["mana7"].to_numpy(), em["draft_id"].to_numpy())
                c["mana7_slope"] = dict(n=int(em.height), slope=coef_row(f, 1), intercept=coef_row(f, 0),
                                        sd_sim=float(np.std(em["sim_mana7"].to_numpy() * 28.0)))
            calib[ev] = c
    return dict(rows=rows, calibration=calib, keep_rules=keep_rules, draw_bias=draw_bias, n_trials=DEFAULT_TRIALS,
                land_bins=list(LAND_BINS))


# ---------------------------------------------------------------------------------------------
# (b) tempo / curve
# ---------------------------------------------------------------------------------------------

SKILL_CUTS = [0.45, 0.50, 0.55, 0.60, 0.65]   # buckets: <0.45, .45-.50, .50-.55 (reference), .55-.60, .60-.65, >=.65, missing


def base_design(df: pl.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Covariates: intercept, on_play, mulligans (1, 2+), opponent mulligans (1, 2+), pilot-skill bucket dummies."""
    n = df.height
    sk = df["skill"].to_numpy().astype(float)
    cols = [np.ones(n), df["on_play"].to_numpy().astype(float)]
    names = ["const", "on_play"]
    mu = df["num_mulligans"].to_numpy()
    om = df["opp_num_mulligans"].to_numpy()
    for nm, v in (("mull1", mu == 1), ("mull2+", mu >= 2), ("opp_mull1", om == 1), ("opp_mull2+", om >= 2)):
        cols.append(v.astype(float))
        names.append(nm)
    miss = np.isnan(sk)
    b = np.digitize(np.where(miss, 0.52, sk), SKILL_CUTS)   # 0..5
    labels = ["skill<.45", "skill.45-.50", "skill.50-.55", "skill.55-.60", "skill.60-.65", "skill>=.65"]
    for i, lab in enumerate(labels):
        if i == 2:
            continue
        cols.append(((b == i) & ~miss).astype(float))
        names.append(lab)
    cols.append(miss.astype(float))
    names.append("skill_missing")
    return np.column_stack(cols), names


def _bin_labels_mana(v: np.ndarray) -> tuple[np.ndarray, list[str]]:
    edges = [0, 3, 4, 5, 6, 7, 8, 9]
    lab = ["<=3", "4", "5", "6", "7", "8", "9+"]
    idx = np.clip(np.digitize(v, edges[1:]), 0, len(lab) - 1)
    # digitize with edges 3,4,..: v<=2 ->0 ... map explicitly
    idx = np.where(v <= 3, 0, np.where(v >= 9, 6, v.astype(int) - 3))
    return idx.astype(int), lab


def _within(x: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Subtract group means."""
    s = np.bincount(g, weights=x)
    c = np.bincount(g)
    return x - (s / c)[g]


def tempo_for_event(e: pl.DataFrame) -> dict:
    """Curving out / mana spent in turns 1-4 versus winning, for the games of one event (>= 4 completed own turns)."""
    e = e.filter(pl.col("turns_done") >= 4)
    n = e.height
    out: dict = dict(n_games=int(n))
    if n < 500:
        return out
    X0, names0 = base_design(e)
    y = e["win"].to_numpy()
    grp = e["draft_id"].to_numpy()
    m4 = e["mana4"].to_numpy()
    hits = e["curve_hits"].to_numpy().astype(int)
    l4 = e["lands4"].to_numpy().astype(int)
    out["win_rate"] = float(y.mean())
    out["mean_mana4"] = float(m4.mean())
    out["share_full_curve"] = float((hits == 3).mean())
    # -- raw + adjusted table by mana spent in turns 1-4
    mi, mlab = _bin_labels_mana(m4)
    ref = 3   # "6" mana
    D = np.column_stack([(mi == i).astype(float) for i in range(len(mlab)) if i != ref])
    fit = logit_fit(np.column_stack([X0, D]), y, grp)
    rows = []
    for i, lab in enumerate(mlab):
        msk = mi == i
        if msk.sum() < 30:
            continue
        Xi = np.column_stack([X0, np.column_stack([np.full(n, float(i == j)) for j in range(len(mlab)) if j != ref])])
        mm, se = marginal_mean(fit, Xi)
        rows.append(dict(bin=lab, n=int(msk.sum()), raw=float(y[msk].mean()), adj=mm, adj_lo=mm - 1.96 * se, adj_hi=mm + 1.96 * se))
    out["by_mana4"] = rows
    # -- by curve hits (number of own turns among 2, 3, 4 with a spell cast)
    D2 = np.column_stack([(hits == i).astype(float) for i in (1, 2, 3)])
    fit2 = logit_fit(np.column_stack([X0, D2]), y, grp)
    rows = []
    for i in range(4):
        msk = hits == i
        if msk.sum() < 30:
            continue
        Xi = np.column_stack([X0, np.column_stack([np.full(n, float(i == j)) for j in (1, 2, 3)])])
        mm, se = marginal_mean(fit2, Xi)
        rows.append(dict(hits=i, n=int(msk.sum()), raw=float(y[msk].mean()), adj=mm, adj_lo=mm - 1.96 * se, adj_hi=mm + 1.96 * se))
    out["by_hits"] = rows
    # -- effect sizes
    eff: dict = {}
    # A: linear mana
    xm = m4 - m4.mean()
    fa = logit_fit(np.column_stack([X0, xm]), y, grp)
    a1 = ame(fa, np.column_stack([X0, xm + 1]), np.column_stack([X0, xm]))
    eff["mana4_per_point"] = dict(coef=coef_row(fa, X0.shape[1]), ame_pp=[100 * v for v in (a1[0], a1[2], a1[3])])
    # B: full curve vs not
    full = (hits == 3).astype(float)
    fb = logit_fit(np.column_stack([X0, full]), y, grp)
    b1 = ame(fb, np.column_stack([X0, np.ones(n)]), np.column_stack([X0, np.zeros(n)]))
    eff["full_curve"] = dict(coef=coef_row(fb, X0.shape[1]), ame_pp=[100 * v for v in (b1[0], b1[2], b1[3])])
    # B2: one more curve turn (linear in hits)
    fb2 = logit_fit(np.column_stack([X0, hits.astype(float)]), y, grp)
    b2 = ame(fb2, np.column_stack([X0, hits + 1.0]), np.column_stack([X0, hits.astype(float)]))
    eff["per_curve_turn"] = dict(coef=coef_row(fb2, X0.shape[1]), ame_pp=[100 * v for v in (b2[0], b2[2], b2[3])])
    # C: mana controlling for land drops (lands in play at end of T4)
    L = np.column_stack([(l4 <= 2).astype(float), (l4 == 3).astype(float)])
    fc = logit_fit(np.column_stack([X0, L, xm]), y, grp)
    k = X0.shape[1] + 2
    c1 = ame(fc, np.column_stack([X0, L, xm + 1]), np.column_stack([X0, L, xm]))
    eff["mana4_given_lands"] = dict(coef=coef_row(fc, k), ame_pp=[100 * v for v in (c1[0], c1[2], c1[3])])
    # lands at T4 alone (screw cost)
    fl = logit_fit(np.column_stack([X0, L]), y, grp)
    l0 = ame(fl, np.column_stack([X0, np.zeros((n, 2))]), np.column_stack([X0, np.column_stack([np.ones(n), np.zeros(n)])]))
    eff["lands4_ge4_vs_le2"] = dict(ame_pp=[100 * v for v in (l0[0], l0[2], l0[3])], share_le2=float((l4 <= 2).mean()))
    out["effects"] = eff
    # D: deck fixed effects (linear probability model, within build)
    bid = _codes(np.char.add(np.char.add(grp.astype(str), "|"), e["build_index"].fill_null(0).to_numpy().astype(str)))
    cnt = np.bincount(bid)
    keep = cnt[bid] >= 2
    if keep.sum() >= 500 and len(np.unique(bid[keep])) >= 50:
        bk = _codes(bid[keep])
        yk = _within(y[keep], bk)
        covs = np.column_stack([X0[keep][:, 1:6]])   # on_play + mulligan dummies (no skill: constant within build)
        cols = [covs[:, j] for j in range(covs.shape[1])]
        res = {}
        for name, v in (("mana4_per_point", m4[keep]), ("full_curve", full[keep]), ("per_curve_turn", hits[keep].astype(float))):
            Z = np.column_stack([_within(c, bk) for c in cols] + [_within(v, bk)])
            f = ols_fit(Z, yk, grp[keep])
            r = coef_row(f, Z.shape[1] - 1)
            res[name] = dict(pp=100 * r["est"], lo=100 * (r["lo"]), hi=100 * r["hi"])
        res["n_games"] = int(keep.sum())
        res["n_builds"] = int(len(np.unique(bid[keep])))
        eff["within_build"] = res
    return out


def curve_predictors(e: pl.DataFrame) -> dict:
    """Deck features that predict (a) the full curve (spell on own turns 2, 3, 4) and (b) mana spent in turns 1-4.
    Coefficients are 'one more card of this kind instead of a 4+ mana spell' (40-card decks)."""
    e = e.filter((pl.col("turns_done") >= 4) & (pl.col("n_cards") == 40) & pl.col("curve_le1").is_not_null())
    n = e.height
    if n < 500:
        return dict(n_games=int(n))
    feats = ["n_lands", "curve_le1", "curve_2", "curve_3", "n_creatures"]
    F = np.column_stack([e[c].to_numpy().astype(float) for c in feats])
    mu = F.mean(axis=0)
    op = e["on_play"].to_numpy().astype(float)
    sk, _ = base_design(e)
    X = np.column_stack([np.ones(n), op, F - mu])
    grp = e["draft_id"].to_numpy()
    full = (e["curve_hits"].to_numpy() == 3).astype(float)
    fl = logit_fit(X, full, grp)
    fo = ols_fit(X, e["mana4"].to_numpy(), grp)
    rows = []
    for j, nm in enumerate(feats):
        i = j + 2
        X1, X0 = X.copy(), X.copy()
        X1[:, i] += 1
        a = ame(fl, X1, X0)
        c = coef_row(fo, i)
        rows.append(dict(feature=nm, mean=float(mu[j]), full_curve_pp=[100 * a[0], 100 * a[2], 100 * a[3]],
                         mana4=[c["est"], c["lo"], c["hi"]]))
    # land-count view: P(full curve) and mean mana4 by land count
    by = []
    for L in LAND_BINS:
        m = e["n_lands"].to_numpy() == L
        if m.sum() >= 100:
            r, lo, hi = cluster_ratio(full[m], np.ones(m.sum()), grp[m], 100)
            by.append(dict(n_lands=L, n=int(m.sum()), full_curve=r, lo=lo, hi=hi, mana4=float(e["mana4"].to_numpy()[m].mean())))
    return dict(n_games=int(n), rows=rows, by_lands=by, p_full=float(full.mean()), mean_mana4=float(e["mana4"].mean()))


# ---------------------------------------------------------------------------------------------
# (c) per-card statistics
# ---------------------------------------------------------------------------------------------

def events_for(set_code: str, fmt: str) -> list[str]:
    """Events behind a format name ('bo3', 'bo1', 'draft') or an event name."""
    have = replay.available_events(set_code)
    f = fmt.lower() if fmt else ""
    if f in ("bo3", "bo1", "draft"):
        return [e for e in have if FMT_OF_EVENT.get(e, "draft") == f]
    return [e for e in have if e.lower() == f]


def card_stats_frame(set_code: str, events: list[str], df: pl.DataFrame, cards: pl.DataFrame, deck_cards: pl.DataFrame,
                     idmap: dict, shrink_k: float = 100.0) -> dict[int, dict]:
    """Per-card statistics over the games of `events` (df: the replay frame of those events)."""
    ii = idinfo_table(set_code, cards, idmap).select("id", "card_id")
    nonland = cards.filter(~pl.col("is_land")).select("card_id", "name", "rarity", "mana_value")
    G = df.select("gid", "deck_key", "won", "turns_done", "num_mulligans")
    dc = deck_cards.join(nonland.select("card_id"), on="card_id", how="inner").select("deck_key", "card_id", pl.col("n").alias("copies"))
    gd = G.join(dc, on="deck_key", how="inner")
    draws = replay.load_draws(set_code, events, lazy=True).filter(pl.col("kind") <= 2).collect()
    seen = (draws.join(ii, on="id", how="inner").group_by(["gid", "card_id"])
            .agg(pl.col("turn").min().alias("first_seen"), pl.col("n").sum().alias("n_seen")))
    casts = replay.load_casts(set_code, events, lazy=True).collect()
    cast = (casts.join(ii, on="id", how="inner")
            .group_by(["gid", "card_id"]).agg(pl.col("turn").clip(1, 30).min().alias("first_cast"), pl.col("n").sum().alias("n_cast")))
    x = gd.join(seen, on=["gid", "card_id"], how="left").join(cast, on=["gid", "card_id"], how="left")
    x = x.with_columns(
        pl.col("first_seen").is_not_null().alias("s"), pl.col("first_cast").is_not_null().alias("c"),
        pl.col("won").cast(pl.Float64).alias("w"))
    x = x.with_columns(
        (pl.col("first_seen") == 0).alias("open"),
        ((pl.col("first_seen") <= 3) & (pl.col("turns_done") >= 7)).fill_null(False).alias("early_long"))
    agg = x.group_by("card_id").agg(
        pl.len().alias("n_deck"), pl.col("w").sum().alias("w_deck"),
        pl.col("s").sum().alias("n_seen"), (pl.col("w") * pl.col("s")).sum().alias("w_seen"),
        pl.col("c").sum().alias("n_cast"), (pl.col("w") * pl.col("c")).sum().alias("w_cast"),
        (~pl.col("c")).sum().alias("n_nocast"), (pl.col("w") * (~pl.col("c")).cast(pl.Float64)).sum().alias("w_nocast"),
        (~pl.col("s")).sum().alias("n_unseen"), (pl.col("w") * (~pl.col("s")).cast(pl.Float64)).sum().alias("w_unseen"),
        (pl.col("s") & ~pl.col("c")).sum().alias("n_seen_nocast"),
        (pl.col("w") * (pl.col("s") & ~pl.col("c")).cast(pl.Float64)).sum().alias("w_seen_nocast"),
        pl.col("first_cast").filter(pl.col("c")).mean().alias("mean_cast_turn"),
        pl.col("first_seen").filter(pl.col("s")).mean().alias("mean_seen_turn"),
        pl.col("open").fill_null(False).sum().alias("n_open"),
        (pl.col("open").fill_null(False) & pl.col("c")).sum().alias("n_open_cast"),
        pl.col("first_cast").filter(pl.col("open").fill_null(False) & pl.col("c")).mean().alias("open_cast_turn"),
        pl.col("early_long").sum().alias("n_early_long"),
        (pl.col("early_long") & pl.col("c")).sum().alias("n_early_long_cast"),
    )
    base_wr = float(G["won"].cast(pl.Float64).mean())
    meta = {r["card_id"]: r for r in nonland.iter_rows(named=True)}

    def shr(w, n):
        return (w + shrink_k * base_wr) / (n + shrink_k)

    out: dict[int, dict] = {}
    for r in agg.iter_rows(named=True):
        cid = int(r["card_id"])
        m = meta.get(cid)
        if m is None or r["n_deck"] == 0:
            continue

        def rate(w, n):
            return float(w / n) if n else None
        out[cid] = dict(
            name=m["name"], rarity=m["rarity"], mana_value=_num(m["mana_value"]),
            n_deck=int(r["n_deck"]), wr_deck=rate(r["w_deck"], r["n_deck"]),
            n_seen=int(r["n_seen"]), wr_seen=rate(r["w_seen"], r["n_seen"]),
            wr_seen_shrunk=float(shr(r["w_seen"], r["n_seen"])),
            n_unseen=int(r["n_unseen"]), wr_unseen=rate(r["w_unseen"], r["n_unseen"]),
            n_cast=int(r["n_cast"]), cast_rate=rate(r["n_cast"], r["n_seen"]),
            wr_cast=rate(r["w_cast"], r["n_cast"]), wr_cast_shrunk=float(shr(r["w_cast"], r["n_cast"])),
            wr_deck_nocast=rate(r["w_nocast"], r["n_nocast"]), n_deck_nocast=int(r["n_nocast"]),
            wr_deck_nocast_shrunk=float(shr(r["w_nocast"], r["n_nocast"])),
            wr_seen_nocast=rate(r["w_seen_nocast"], r["n_seen_nocast"]), n_seen_nocast=int(r["n_seen_nocast"]),
            cast_gap=(None if not (r["n_cast"] and r["n_nocast"]) else float(shr(r["w_cast"], r["n_cast"]) - shr(r["w_nocast"], r["n_nocast"]))),
            mean_cast_turn=_num(r["mean_cast_turn"]), mean_seen_turn=_num(r["mean_seen_turn"]),
            n_open=int(r["n_open"]), open_cast_rate=rate(r["n_open_cast"], r["n_open"]), open_cast_turn=_num(r["open_cast_turn"]),
            n_early_long=int(r["n_early_long"]), early_cast_rate=rate(r["n_early_long_cast"], r["n_early_long"]),
            base_wr=base_wr)
    return out


def _top(cs: dict[int, dict], key: str, n: int, reverse: bool, min_field: str, min_val: int, extra: tuple[str, ...] = ()) -> list[dict]:
    rows = [(k, v) for k, v in cs.items() if v.get(key) is not None and v.get(min_field, 0) >= min_val]
    rows.sort(key=lambda kv: kv[1][key], reverse=reverse)
    return [dict(card_id=k, name=v["name"], mana_value=v["mana_value"], value=v[key], n=v[min_field],
                 **{x: v.get(x) for x in extra}) for k, v in rows[:n]]


def card_highlights(cs: dict[int, dict]) -> dict:
    """Short lists for the report. Thresholds scale with the data size."""
    if not cs:
        return {}
    med_seen = float(np.median([v["n_seen"] for v in cs.values()]))
    thr = int(max(40, min(300, 0.15 * med_seen)))
    return dict(
        min_seen=thr, n_cards=len(cs),
        slow_open=_top(cs, "open_cast_turn", 12, True, "n_open", thr, ("open_cast_rate",)),
        fast_open=_top(cs, "open_cast_turn", 12, False, "n_open", thr, ("open_cast_rate",)),
        low_cast=_top(cs, "early_cast_rate", 12, False, "n_early_long", thr),
        cast_gap_high=_top(cs, "cast_gap", 12, True, "n_seen_nocast", max(20, thr // 3), ("wr_cast_shrunk", "wr_deck_nocast_shrunk", "cast_rate")),
        cast_gap_low=_top(cs, "cast_gap", 12, False, "n_seen_nocast", max(20, thr // 3), ("wr_cast_shrunk", "wr_deck_nocast_shrunk", "cast_rate")),
        gih_top=_top(cs, "wr_seen_shrunk", 12, True, "n_seen", thr),
        gih_bottom=_top(cs, "wr_seen_shrunk", 12, False, "n_seen", thr),
    )


# ---------------------------------------------------------------------------------------------
# summary / API
# ---------------------------------------------------------------------------------------------

def _signature(set_code: str) -> str:
    ms = replay._shards(set_code, None)
    try:
        man = json.loads((config.corpus_dir(set_code) / "manifest.json").read_text(encoding="utf-8"))
        csig = man.get("built_at", "")
    except (OSError, ValueError):
        csig = ""
    return json.dumps([STATS_VERSION, replay.REPLAY_VERSION, sorted(m["fingerprint"] for m in ms), csig])


def summary_path(set_code: str) -> Path:
    return replay.replay_dir(set_code) / "summary.json"


def cached_summary(set_code: str) -> dict | None:
    """The cached summary if it is current (no recomputation)."""
    p = summary_path(set_code.upper())
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return d if d.get("signature") == _signature(set_code.upper()) else None


def replay_summary(set_code: str, progress: Progress | None = None, cancel: threading.Event | None = None,
                   n_jobs: int | None = None, refresh: bool = False) -> dict:
    """Ingest (incremental) the registered replay files of the set and compute every section. JSON-able.

    Returns {"available": False, "reason": ...} when there are no replay files or no corpus. Cached by file fingerprints
    and corpus build time (`refresh=True` recomputes)."""
    set_code = set_code.upper()
    t0 = time.time()

    def prog(f: float, m: str) -> None:
        if progress:
            progress(float(min(max(f, 0.0), 1.0)), m)

    if not replay.replay_files(set_code):
        return dict(available=False, reason="no replay_data_public files are registered for this set", set_code=set_code)
    if not (config.corpus_dir(set_code) / "games.parquet").exists():
        return dict(available=False, reason="the corpus is not built yet (press Calculate first)", set_code=set_code)
    if not refresh:
        d = None
        replay.ingest_replays(set_code, lambda f, m: prog(0.25 * f, m), cancel, n_jobs)
        d = cached_summary(set_code)
        if d is not None:
            prog(1.0, "Replay analysis (cached)")
            return d
    else:
        replay.ingest_replays(set_code, lambda f, m: prog(0.25 * f, m), cancel, n_jobs)
    t_ingest = time.time() - t0
    _check(cancel)
    prog(0.26, "Linking replay games to the corpus")
    idm = replay.resolve_idmap(set_code)
    idmap = idm["map"]
    t1 = time.time()
    df, tabs, info = build_frame(set_code, None, n_jobs, cancel, lambda f, m: prog(0.26 + 0.05 * f, m))
    cards, deck_cards = tabs["cards"], tabs["deck_cards"]
    idinfo = idinfo_table(set_code, cards, idmap)
    _check(cancel)
    prog(0.32, "Lands in the first cards")
    draws_all = replay.load_draws(set_code, None, lazy=True).filter(pl.col("kind").is_in([1, 3])).collect()
    first = first_cards_lands(draws_all, idinfo)
    del draws_all
    _check(cancel)
    prog(0.38, "Simulator versus real play")
    sim = simulator_section(df, first, cards, deck_cards, n_jobs, cancel, lambda f, m: prog(0.38 + 0.22 * f, m))
    events = [e for e in EVENT_ORDER if e in set(df["event"].unique().to_list())]
    tempo, preds = {}, {}
    for i, ev in enumerate(events):
        _check(cancel)
        prog(0.62 + 0.15 * i / max(len(events), 1), f"Tempo analysis {ev}")
        e = df.filter(pl.col("event") == ev)
        tempo[ev] = tempo_for_event(e)
        preds[ev] = curve_predictors(e)
    cards_out: dict[str, dict] = {}
    for j, fmt in enumerate(("bo3", "bo1", "draft")):
        evs = [e for e in events if FMT_OF_EVENT.get(e) == fmt]
        if not evs:
            continue
        _check(cancel)
        prog(0.78 + 0.18 * j / 3, f"Per-card statistics ({fmt})")
        cs = card_stats_frame(set_code, evs, df.filter(pl.col("event").is_in(evs)), cards, deck_cards, idmap)
        cards_out[fmt] = dict(events=evs, highlights=card_highlights(cs), n_cards=len(cs))
        _save_card_stats(set_code, fmt, cs)
    ev_rows = []
    for ev in events:
        e = df.filter(pl.col("event") == ev)
        ev_rows.append(dict(event=ev, fmt=FMT_OF_EVENT.get(ev), label=EVENT_LABEL.get(ev, ev), n_games=int(e.height),
                            n_turn3=int((e["turns_done"] >= 3).sum()), n_turn4=int((e["turns_done"] >= 4).sum()),
                            n_turn7=int((e["turns_done"] >= 7).sum()), win_rate=float(e["win"].mean()),
                            on_play_share=float(e["on_play"].mean())))
    out = dict(
        available=True, set_code=set_code, version=STATS_VERSION, signature=_signature(set_code),
        built_at=time.strftime("%Y-%m-%dT%H:%M:%S"), events=ev_rows, link=info,
        idmap=dict(n_ids=idm["n_ids"], n_mapped=idm["n_mapped"], mass_mapped=idm["mass_mapped"]),
        simulator=sim, tempo=tempo, curve_predictors=preds, cards=cards_out,
        notes=_NOTES, timings=dict(ingest=round(t_ingest, 1), stats=round(time.time() - t1, 1), total=round(time.time() - t0, 1)))
    out = _clean(out)
    try:
        summary_path(set_code).write_text(json.dumps(out), encoding="utf-8")
    except OSError:
        log.warning("could not cache the replay summary")
    prog(1.0, "Replay analysis ready")
    return out


_NOTES = [
    "The replay files only record completed own turns: every statistic for turn k is conditional on the game lasting past turn k "
    "(games that end earlier have no data). Real rates are re-weighted to 50% on the play / 50% on the draw to match the simulator.",
    "Real players differ from the simulator's rules: they keep 1- and 6-land hands, they hold cards, and 'mana spent' includes abilities.",
    "Card statistics are descriptive: winners cast more cards (the game lasts and they are ahead), so 'win rate when cast' is biased upwards "
    "for every card; compare cards with each other, not with the base rate. Win rates are shrunk towards the format mean (100 pseudo-games).",
    "A very low cast rate (under about 15%) usually means the card is put into play in a way the log does not record as a cast "
    "(alternate cost, ability, transform), not that players cannot cast it.",
]


def _save_card_stats(set_code: str, fmt: str, cs: dict[int, dict]) -> None:
    try:
        (replay.replay_dir(set_code) / f"cards_{fmt}.json").write_text(json.dumps(_clean({str(k): v for k, v in cs.items()})),
                                                                       encoding="utf-8")
    except OSError:
        log.warning("could not cache card stats")


def replay_card_stats(set_code: str, fmt: str = "bo3", n_jobs: int | None = None, refresh: bool = False,
                      progress: Progress | None = None, cancel: threading.Event | None = None) -> dict[int, dict]:
    """{card_id: stats} for the games of a format ('bo3' TradSealed, 'bo1' Sealed, 'draft' = both draft events) or of one
    event name. Keys are corpus card ids; see `card_stats_frame` for the fields. Cached next to the summary."""
    set_code = set_code.upper()
    fmt = fmt.lower() if fmt.lower() in ("bo3", "bo1", "draft") else fmt
    p = replay.replay_dir(set_code) / f"cards_{fmt}.json"
    if not refresh and p.exists() and cached_summary(set_code) is not None:
        try:
            return {int(k): v for k, v in json.loads(p.read_text(encoding="utf-8")).items()}
        except (OSError, ValueError):
            pass
    evs = events_for(set_code, fmt)
    if not evs:
        replay.ingest_replays(set_code, progress, cancel, n_jobs)
        evs = events_for(set_code, fmt)
    if not evs:
        return {}
    replay.ingest_replays(set_code, progress, cancel, n_jobs, events=evs)
    idmap = replay.resolve_idmap(set_code)["map"]
    df, tabs, _ = build_frame(set_code, evs, n_jobs, cancel)
    cs = card_stats_frame(set_code, evs, df, tabs["cards"], tabs["deck_cards"], idmap)
    _save_card_stats(set_code, fmt, cs)
    return cs


# ---------------------------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------------------------

def _e(x) -> str:
    return html.escape(str(x))


def _f(x, d=3, plus=False) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{x:+.{d}f}" if plus else f"{x:.{d}f}"


def _p(x, d=1, plus=False) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{100 * x:+.{d}f}%" if plus else f"{100 * x:.{d}f}%"


def _pp(x, d=1) -> str:
    return "n/a" if x is None else f"{100 * x:+.{d}f} pp"


def _ci(lo, hi, fn) -> str:
    return f"[{fn(lo)}, {fn(hi)}]"


def _table(headers: list[str], rows: list[list[str]], left: set[int] | None = None) -> str:
    left = left if left is not None else {0}
    h = "".join(f'<th class="{"l" if i in left else ""}">{_e(c)}</th>' for i, c in enumerate(headers))
    b = "".join("<tr>" + "".join(f'<td class="{"l" if i in left else ""}">{c}</td>' for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f"<div style='overflow-x:auto'><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>"


def _cell(r: dict | None, key: str = "real", fmt=lambda v: _p(v, 1)) -> str:
    if not r or r.get(key) is None:
        return "n/a"
    return f"{fmt(r[key])}<br><span class='mut'>{_ci(r.get('lo'), r.get('hi'), fmt)}</span>" if r.get("lo") is not None else fmt(r[key])


def _diff_cell(r: dict | None, fmt=lambda v: _pp(v, 1)) -> str:
    if not r or r.get("diff") is None:
        return "n/a"
    sig = r.get("diff_lo") is not None and (r["diff_lo"] > 0 or r["diff_hi"] < 0)
    s = f"<b>{fmt(r['diff'])}</b>" if sig else fmt(r["diff"])
    return f"{s}<br><span class='mut'>{_ci(r.get('diff_lo'), r.get('diff_hi'), fmt)}</span>"


def _label(row: dict) -> str:
    return f"{_e(EVENT_LABEL.get(row['event'], row['event']))}, {row['n_lands'] if row['n_lands'] else 'all'} lands"


def render_replay_section(summary: dict) -> str:
    """HTML fragment (h2 sections) for report.html built from `replay_summary`. Safe for unavailable summaries."""
    P: list[str] = []
    if not summary or not summary.get("available"):
        reason = (summary or {}).get("reason", "no replay data")
        return f"<h2>Replay data: simulator check and tempo</h2><p class='mut'>Not available: {_e(reason)}.</p>"
    P.append("<h2>Replay data: does the mana simulator match real play?</h2>")
    ev = summary["events"]
    P.append("<p class='small mut'>Turn-by-turn replay files, "
             + "; ".join(f"{_e(r['label'])}: {r['n_games']:,} games" for r in ev)
             + ". Real rates are re-weighted to 50% play / 50% draw (the simulator's assumption) and carry 95% cluster-bootstrap "
               "intervals (resampling whole events). Differences are real minus simulated; bold = interval excludes zero. "
               "Only completed own turns are recorded, so turn-k statistics condition on the game lasting past turn k.</p>")
    sim = summary["simulator"]
    # ---- screw
    rows = []
    for r in sim["rows"]:
        s = r.get("screw")
        if not s:
            continue
        sh = r.get("screw_hand")
        sn = r.get("screw_hand_nd")

        def dc(x):
            return "n/a" if not x else f"{_pp(x['diff'], 1)}<br><span class='mut'>{_ci(x['diff_lo'], x['diff_hi'], lambda v: _pp(v, 1))}</span>"
        rows.append([_label(r), f"{s['n']:,}", _cell(s), _p(s["sim"], 1), _p(s.get("rule"), 1), _diff_cell(s), dc(sh), dc(sn)])
    if rows:
        P.append("<h3>P(fewer than 3 lands in play after own turn 3)</h3>")
        P.append("<p class='small mut'>'Simulator' = the model's p_screw. 'Real keep rule' = an independent mini-simulation that mulligans with the real "
                 "probabilities observed for each number of lands in the first 7 cards (table below). Last two columns: no-mulligan games only, real "
                 "minus the probability under random draws given the hand actually kept, first with one draw per turn, then with the number "
                 "of cards really drawn by turn 3 - what is left is lands that did not come from draws (land search, ramp) and non-random draws.</p>")
        P.append(_table(["Event, lands", "Games", "Real", "Simulator", "Sim. with real keep rule", "Real - simulator",
                         "Real - random (1 draw/turn)", "Real - random (actual draws)"], rows))
    # ---- mulligans
    rows = []
    for r in sim["rows"]:
        if r.get("n_lands") is None and r.get("mull"):
            m = r["mull"]
            rows.append([_e(EVENT_LABEL.get(r["event"], r["event"])), f"{m['n']:,}", _cell(m), _p(m["sim"], 1), _diff_cell(m)])
    if rows:
        P.append("<h3>Mulligans</h3>")
        P.append(_table(["Event", "Games", "Real mulligan rate", "Simulator", "Real - simulator"], rows))
    db = sim.get("draw_bias", {})
    if db:
        rows = []
        for evn in EVENT_ORDER:
            for x in db.get(evn, []):
                rows.append([_e(EVENT_LABEL.get(evn, evn)), str(x["lands_in_hand"]), f"{x['n']:,}", _cell(x), _p(x["sim"], 1), _diff_cell(x)])
        P.append("<h3>Are the draws random given the opening hand?</h3>")
        P.append("<p class='small mut'>P(the first card drawn is a land) after a kept 7-card hand without mulligans, by lands in that hand, against the value for a "
                 "random shuffle ((lands in deck - lands in hand) / 33). Differences mean the library is not a uniformly random shuffle "
                 "(hand smoothing that carries into the draws).</p>")
        P.append(_table(["Event", "Lands in hand", "Games", "Real", "Random shuffle", "Real - random"], rows, left={0}))
    kr = sim.get("keep_rules", {})
    if kr:
        rows = []
        for evn in EVENT_ORDER:
            lst = kr.get(evn)
            if not lst:
                continue
            for play in (True, False):
                cells = []
                for k in range(0, 8):
                    x = next((z for z in lst if z["on_play"] == play and z["lands"] == k), None)
                    cells.append("-" if not x or x["n"] < 30 else _p(x["p_mull"], 0))
                rows.append([_e(EVENT_LABEL.get(evn, evn)), "play" if play else "draw"] + cells)
        P.append("<h3>Real mulligan probability by lands in the first 7 cards</h3>")
        P.append(_table(["Event", "", "0", "1", "2", "3", "4", "5", "6", "7"], rows, left={0, 1}))
    # ---- mana
    rows = []
    for r in sim["rows"]:
        if r.get("n_lands") is None:
            continue
        for key, lab in (("mana4", "T1-T4 (games with 4+ turns)"), ("mana4_long", "T1-T4 (games with 7+ turns)"), ("mana7", "T1-T7 / 28")):
            s = r.get(key)
            if not s:
                continue
            fm = (lambda v: _f(v, 3)) if key == "mana7" else (lambda v: _f(v, 2))
            dm = (lambda v: _f(v, 3, True)) if key == "mana7" else (lambda v: _f(v, 2, True))
            rows.append([_e(EVENT_LABEL.get(r["event"], r["event"])), lab, f"{s['n']:,}", _cell(s, "real", fm), fm(s["sim"]), _diff_cell(s, dm)])
    if rows:
        P.append("<h3>Mana spent in own turns (all land counts pooled)</h3>")
        P.append("<p class='small mut'>T1-T4 is mana (a perfect curve would be 9-10). The same games are used for both T1-T4 rows, the second restricted to "
                 "games that lasted 7+ turns, so the T1-T4 and T1-T7 comparisons can be read on one population.</p>")
        P.append(_table(["Event", "Window", "Games", "Real", "Simulated", "Real - simulated"], rows, left={0, 1}))
    # ---- flood
    rows = []
    for r in sim["rows"]:
        s = r.get("flood16")
        if s and r.get("n_lands") is None:
            rows.append([_e(EVENT_LABEL.get(r["event"], r["event"])), f"{s['n']:,}", _cell(s), _p(s["sim"], 1), _diff_cell(s),
                         _p(s.get("unconditional"), 1), _p(s["sim_model"], 1), f"{_f(s['mean_lands'], 2)} vs {_f(s['expected_lands'], 2)}"])
    if rows:
        P.append("<h3>P(9 or more lands among the first 16 cards seen)</h3>")
        P.append("<p class='small mut'>Games with no mulligan and nine recorded draws (long games only). 'Random draws' = probability for the nine draws given the "
                 "7-card hand actually kept (so keep decisions do not matter); the unconditional shuffle probability and the model's p_flood "
                 "feature (which includes mulligan paths) are shown for reference.</p>")
        P.append(_table(["Event", "Games", "Real", "Random draws given hand", "Real - random", "Unconditional shuffle", "Simulator p_flood",
                         "Mean lands in 16: real vs expected"], rows))
    # ---- first 7
    rows = []
    for r in sim["rows"]:
        if r.get("n_lands") is not None:
            continue
        a, b = r.get("first7"), r.get("first7_tail")
        if a and b:
            rows.append([_e(EVENT_LABEL.get(r["event"], r["event"])), f"{a['n']:,}", _cell(a, "real", lambda v: _f(v, 3)), _f(a["sim"], 3),
                         _cell(b), _p(b.get("exact"), 1), _p(b["sim"], 1)])
    if rows:
        P.append("<h3>The first seven cards of the shuffle</h3>")
        P.append("<p class='small mut'>Lands in the first 7 cards before any mulligan (expected = 7 x lands / 40) and the share of first hands outside 2-5 "
                 "lands (the simulator mulligans exactly these). A real share far below the random-shuffle value means Arena's Bo1 hand smoothing.</p>")
        P.append(_table(["Event", "Games", "Lands in first 7", "Random shuffle", "Share outside 2-5 lands", "Random shuffle", "Simulator mull rate"], rows))
    # ---- calibration
    cal = sim.get("calibration", {})
    if cal:
        rows = []
        for evn in EVENT_ORDER:
            c = cal.get(evn)
            if not c:
                continue
            s, g = c["screw"], c.get("screw_given_lands")
            m = c.get("mana7_slope")
            rows.append([_e(EVENT_LABEL.get(evn, evn)), f"{c['n_decks']:,}",
                         f"{_f(s['slope']['est'], 2)} {_ci(s['slope']['lo'], s['slope']['hi'], lambda v: _f(v, 2))}",
                         _f(c.get("sim_sd_within_lands"), 4),
                         "n/a" if not m else f"{_f(m['slope']['est'], 2)} {_ci(m['slope']['lo'], m['slope']['hi'], lambda v: _f(v, 2))}"])
        P.append("<h3>Per-deck calibration</h3>")
        P.append("<p class='small mut'>Slope of the logit of the real screw outcome (fewer than 3 lands after turn 3, 14-19 lands) on the logit of the "
                 "deck's simulated p_screw (1 = calibrated, 0 = no signal; cluster-robust CI). The simulated p_screw depends only on the land count "
                 "(its SD within a land count is shown), so this is a calibration of the land-count effect, not a deck-specific signal. "
                 "Last column: slope of real mana spent T1-T7 on the deck's simulated value (a genuine deck-level signal).</p>")
        P.append(_table(["Event", "Decks", "Screw slope (95% CI)", "Sim SD within land count", "Mana T1-T7 slope (95% CI)"], rows))
    # ---------------- tempo
    P.append("<h2>Replay data: tempo and curving out</h2>")
    P.append("<p class='small mut'>Games with at least four completed own turns. Win probabilities are logistic-regression predictions averaged "
             "over the games (g-computation), adjusted for play/draw, mulligans and the pilot-skill bucket; intervals are cluster-robust delta-method "
             "95% CIs. Conditioning on the game reaching turn 4 removes quick wins and quick losses, so effects are, if anything, understated.</p>")
    rows = []
    for evn in EVENT_ORDER:
        t = summary["tempo"].get(evn)
        if not t or "effects" not in t:
            continue
        ef = t["effects"]
        w = ef.get("within_build")
        rows.append([_e(EVENT_LABEL.get(evn, evn)), f"{t['n_games']:,}", _p(t["share_full_curve"], 0),
                     _pp_ci(ef["mana4_per_point"]["ame_pp"]), _pp_ci(ef["mana4_given_lands"]["ame_pp"]),
                     _pp_ci(ef["full_curve"]["ame_pp"]), _pp_ci(ef["per_curve_turn"]["ame_pp"]),
                     "n/a" if not w else _pp_ci([w["full_curve"]["pp"], w["full_curve"]["lo"], w["full_curve"]["hi"]]),
                     _pp_ci([-ef["lands4_ge4_vs_le2"]["ame_pp"][0], -ef["lands4_ge4_vs_le2"]["ame_pp"][2], -ef["lands4_ge4_vs_le2"]["ame_pp"][1]])
                     + f"<br><span class='mut'>{_p(ef['lands4_ge4_vs_le2']['share_le2'], 1)} of games</span>"])
    if rows:
        P.append(_table(["Event", "Games", "Full curve", "+1 mana T1-T4", "+1 mana, same lands", "Spell on T2, T3, T4 vs not", "Per extra curve turn",
                         "Full curve, same deck", "2 or fewer lands after T4 vs 4+"], rows))
        P.append("<p class='small mut'>'Full curve' = a spell cast on each of own turns 2, 3 and 4. 'Same lands' also controls for lands in play after turn 4 "
                 "(so it isolates casting, not missed land drops). 'Same deck' compares games of the same build (fixed effects, linear "
                 "probability model), which removes deck-quality confounding.</p>")
    for evn in EVENT_ORDER:
        t = summary["tempo"].get(evn)
        if not t or "by_mana4" not in t:
            continue
        P.append(f"<h3>{_e(EVENT_LABEL.get(evn, evn))}: win rate by mana spent in turns 1-4 and by curve turns hit</h3>")
        rows = [[b["bin"], f"{b['n']:,}", _p(b["raw"], 1), f"{_p(b['adj'], 1)} {_ci(b['adj_lo'], b['adj_hi'], lambda v: _p(v, 1))}"] for b in t["by_mana4"]]
        P.append("<div class='cols'><div>" + _table(["Mana T1-T4", "Games", "Raw win rate", "Adjusted (95% CI)"], rows) + "</div>")
        rows = [[str(b["hits"]), f"{b['n']:,}", _p(b["raw"], 1), f"{_p(b['adj'], 1)} {_ci(b['adj_lo'], b['adj_hi'], lambda v: _p(v, 1))}"] for b in t["by_hits"]]
        P.append("<div>" + _table(["Turns with a spell (of 2, 3, 4)", "Games", "Raw win rate", "Adjusted (95% CI)"], rows) + "</div></div>")
    rows = []
    for evn in EVENT_ORDER:
        cp = summary["curve_predictors"].get(evn)
        if not cp or "rows" not in cp:
            continue
        for r in cp["rows"]:
            rows.append([_e(EVENT_LABEL.get(evn, evn)), _e(r["feature"]), _f(r["mean"], 1),
                         f"{_f(r['full_curve_pp'][0], 2, True)} {_ci(r['full_curve_pp'][1], r['full_curve_pp'][2], lambda v: _f(v, 2, True))}",
                         f"{_f(r['mana4'][0], 3, True)} {_ci(r['mana4'][1], r['mana4'][2], lambda v: _f(v, 3, True))}"])
    if rows:
        P.append("<h3>Which deck features predict curving out?</h3>")
        P.append("<p class='small mut'>One more card of this kind in a 40-card deck instead of a 4+ mana spell (for creatures: instead of a non-creature spell): change in P(full curve) (percentage "
                 "points) and in mana spent in turns 1-4 (cluster-robust; play/draw controlled).</p>")
        P.append(_table(["Event", "Feature", "Deck mean", "P(full curve), pp", "Mana T1-T4"], rows, left={0, 1}))
    # ---------------- cards
    P.append("<h2>Replay data: card patterns</h2>")
    P.append("<p class='small mut'>Descriptive only: winners cast more cards, so win rates of cast cards are biased upward and must be compared between cards, "
             "not with the format average. Win rates are shrunk to the format mean with 100 pseudo-games.</p>")
    titles = dict(slow_open="Slowest cast when in the opening hand (mean own turn of first cast)",
                  fast_open="Fastest cast when in the opening hand",
                  low_cast="Lowest cast rate (seen by turn 3 in a game of 7+ turns)",
                  cast_gap_high="Largest win-rate gap, cast vs in deck but not cast",
                  gih_top="Best win rate when seen (opening hand or drawn)")
    for fmt in ("bo3", "bo1", "draft"):
        c = summary["cards"].get(fmt)
        if not c or not c.get("highlights"):
            continue
        h = c["highlights"]
        P.append(f"<h3>{_e({'bo3': 'Bo3 TradSealed', 'bo1': 'Bo1 Sealed', 'draft': 'Draft (Premier + Trad)'}[fmt])} "
                 f"<span class='mut small'>(min {h['min_seen']} games)</span></h3><div class='cols'>")
        for key in ("slow_open", "fast_open", "low_cast", "cast_gap_high", "gih_top"):
            lst = h.get(key) or []
            if not lst:
                continue
            if key in ("slow_open", "fast_open"):
                rr = [[_e(x["name"]), _f(x["mana_value"], 0), f"{x['n']:,}", _f(x["value"], 2), _p(x.get("open_cast_rate"), 0)] for x in lst[:8]]
                hdr = ["Card", "MV", "Games", "Mean cast turn", "Cast"]
            elif key == "low_cast":
                rr = [[_e(x["name"]), _f(x["mana_value"], 0), f"{x['n']:,}", _p(x["value"], 0)] for x in lst[:8]]
                hdr = ["Card", "MV", "Games", "Cast rate"]
            elif key == "cast_gap_high":
                rr = [[_e(x["name"]), _f(x["mana_value"], 0), f"{x['n']:,}", _pp(x["value"], 1), _p(x.get("cast_rate"), 0)] for x in lst[:8]]
                hdr = ["Card", "MV", "Seen, uncast", "Cast - uncast", "Cast rate"]
            else:
                rr = [[_e(x["name"]), _f(x["mana_value"], 0), f"{x['n']:,}", _p(x["value"], 1)] for x in lst[:8]]
                hdr = ["Card", "MV", "Games seen", "Win rate"]
            P.append(f"<div><p class='small'><b>{_e(titles[key])}</b></p>{_table(hdr, rr)}</div>")
        P.append("</div>")
    P.append("<ul class='small mut'>" + "".join(f"<li>{_e(n)}</li>" for n in summary.get("notes", [])) + "</ul>")
    t = summary.get("timings", {})
    idm = summary.get("idmap", {})
    P.append(f"<p class='small mut'>Replay analysis: {idm.get('n_mapped', 0)} of {idm.get('n_ids', 0)} Arena card ids resolved to cards "
             f"({_p(idm.get('mass_mapped'), 2)} of card appearances); computed in {t.get('total', 0):.0f}s.</p>")
    return "".join(P)


def _pp_ci(v) -> str:
    if not v:
        return "n/a"
    return f"{v[0]:+.2f} pp<br><span class='mut'>[{v[1]:+.2f}, {v[2]:+.2f}]</span>"


__all__ = ["replay_summary", "replay_card_stats", "render_replay_section", "cached_summary", "cluster_ratio", "logit_fit", "ols_fit",
           "ame", "first_cards_lands", "build_frame", "tempo_for_event", "curve_predictors", "card_stats_frame"]
