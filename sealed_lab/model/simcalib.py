"""Calibration of the mana simulator (v2) against the replay data: what real Arena shuffling and keeping look like.

Everything here is measured from the replay shards (`ingest/replay.py`, REPLAY_VERSION >= 3) and the corpus; nothing
is assumed. `calibrate(sets)` returns the parameters shipped as constants in `manasim.py` (the shipped values come
from MSH + SOS) together with diagnostics, so a new set can be checked against them:

  keep rule   P(mulligan | lands in the 7 cards shown, on the play / draw, mulligan number) per regime (Bo3 "Trad"
              queues, Bo1 draft queues, Bo1 Sealed queue). Raw frequencies are shrunk towards the pooled play/draw
              frequency and a generic U-shaped prior in thin cells. Players differ between sets (Premier-draft mulligan
              rate 9.6% in MSH, 13.0% in SOS), so `keep_regimes` re-measures the rule for the set being modelled and
              shrinks it towards the shipped MSH + SOS average; the smoother, the draw tilt and the card flow are properties
              of Arena and the card pool and are the same in both sets.
  hand smoother  Arena shows a smoothed first hand in Bo1 queues. The land count l of the first 7 cards is modelled as
              P(l) proportional to hypergeometric(l) * exp(-(|l - 7 L / n| / s) ** g)   (L lands in n cards): an
              "accept the hand with a probability that falls with its distance from the expected land count" rule,
              fitted by maximum likelihood on the first hands of all games with 40-card decks and 14-20 lands.
              ("Draw two hands and keep the closer one" does NOT reproduce the data: it leaves 4% of the hands in the
              tails, the data have 0.9%, and it prefers 2 over 4 lands far too strongly at 17 lands.)
  draw smoothing  logistic tilt of the probability that the next card is a land by the deviation of the lands seen so far
              from their expected number: logit P = logit(hypergeometric) - beta * (lands seen - cards seen * L / n).
              beta ~ 0.04-0.05 in the draft and Trad queues, ~0.36 in the Sealed queue. In every queue a player who holds
              no land draws one a few points more often than chance (turn 3 / 4 / 5+: about +7 / +4 / +3 pp, `rescue_lift`).
  card flow   extra cards drawn by the deck's draw spells and lands fetched by its land-search spells (oracle text),
              realised with probabilities kappa_draw / kappa_fetch fitted on the real number of cards drawn by turn.
"""
from __future__ import annotations

import json
import logging
import threading
from typing import Callable

import numpy as np
import polars as pl
from scipy.optimize import minimize
from scipy.stats import hypergeom

from sealed_lab import config
from sealed_lab.types import Cancelled

log = logging.getLogger(__name__)
Progress = Callable[[float, str], None]

DEFAULT_KEEP_PRIOR = np.array([1.0, 0.85, 0.12, 0.05, 0.05, 0.20, 0.70, 0.95])
SHRINK_PLAY = 30.0         # pseudo-observations pulling a (play/draw) cell towards the pooled frequency
MIN_POOLED = 300           # hands seen less often than this (all play/draw) use the prior only
SHRINK_PRIOR = 30.0        # pseudo-observations pulling the pooled frequency towards the generic U-shaped prior
MAX_MULL_IDX = 2           # keep tables for the decision after the 1st, 2nd and 3rd hand (the sim uses the first two)


def _check(cancel: threading.Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise Cancelled()


# ---------------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------------

def hands_frame(set_code: str, events: list[str] | None = None, cancel: threading.Event | None = None) -> pl.DataFrame:
    """One row per replay game with a linked 40-card deck: event, regime, on_play, num_mulligans, deck lands, and for the
    first three hands shown their size and number of lands (`n_h1`, `l_h1`, ...), the final opening hand (`l_open`) and, per own
    turn 1..8, cards drawn / lands drawn (`nd_k`, `ld_k`, tutored cards included) and lands in play (`lands_k`)."""
    from sealed_lab.ingest import replay
    from sealed_lab.ingest.corpus import load_corpus

    set_code = set_code.upper()
    idm = replay.resolve_idmap(set_code)["map"]
    c = load_corpus(set_code)
    cards = c["cards"].sort("card_id")
    isl = dict(zip(cards["card_id"].to_list(), cards["is_land"].to_list()))
    il = pl.DataFrame({"id": [int(k) for k in idm], "isl": [int(isl[v]) for v in idm.values()]}, schema={"id": pl.Int32, "isl": pl.Int8})
    g = replay.load_games(set_code, events)
    _check(cancel)
    d = replay.load_draws(set_code, events, lazy=True).filter(pl.col("kind") <= 5).collect()
    d = d.join(il, on="id", how="left").with_columns(pl.col("isl").fill_null(0).cast(pl.Int32), pl.col("n").cast(pl.Int32))
    cg = c["games"].select("game_key", "deck_key", "fmt", "draft_id", "match_number", "game_number", "game_time", "skill").collect()
    g = replay.link_corpus(g, cg)
    g = g.join(c["decks"].select("deck_key", "n_lands", "n_cards"), on="deck_key", how="left")
    g = g.filter((pl.col("n_cards") == 40) & pl.col("n_lands").is_not_null())
    for kind, nm in ((0, "open"), (3, "h1"), (4, "h2"), (5, "h3")):
        h = d.filter(pl.col("kind") == kind).group_by("gid").agg(
            pl.col("n").sum().alias(f"n_{nm}"), (pl.col("n") * pl.col("isl")).sum().alias(f"l_{nm}"))
        g = g.join(h, on="gid", how="left")
    dr = d.filter(pl.col("kind").is_in([1, 2])).group_by("gid", "turn").agg(
        pl.col("n").sum().alias("nd"), (pl.col("n") * pl.col("isl")).sum().alias("ld"))
    for k in range(1, 9):
        a = dr.filter(pl.col("turn") == k).select("gid", pl.col("nd").alias(f"nd_{k}"), pl.col("ld").alias(f"ld_{k}"))
        g = g.join(a, on="gid", how="left")
    g = g.with_columns([pl.col(f"{p}_{k}").fill_null(0) for k in range(1, 9) for p in ("nd", "ld")])
    reg = {e: config.event_regime(e) for e in g["event"].unique().to_list()}
    return g.with_columns(pl.col("event").replace_strict(reg, default="hand").alias("regime"))


# ---------------------------------------------------------------------------------------------
# keep rule
# ---------------------------------------------------------------------------------------------

def keep_counts(h: pl.DataFrame) -> np.ndarray:
    """(mull_idx 0..2, on_play 0/1, lands 0..7, [n, mulliganed]) counts: the decision after seeing hand j+1 (it was
    mulliganed when the game's `num_mulligans` > j)."""
    out = np.zeros((MAX_MULL_IDX + 1, 2, 8, 2))
    for j in range(MAX_MULL_IDX + 1):
        nm, lm = f"n_h{j + 1}", f"l_h{j + 1}"
        d = h.filter((pl.col("num_mulligans") >= j) & (pl.col(nm) == 7))
        t = d.group_by("on_play", pl.col(lm).alias("l")).agg(pl.len().alias("n"), (pl.col("num_mulligans") > j).sum().alias("k"))
        for r in t.iter_rows(named=True):
            if r["l"] is not None and 0 <= r["l"] <= 7:
                out[j, int(bool(r["on_play"])), int(r["l"])] = [r["n"], r["k"]]
    return out


def smooth_keep(counts: np.ndarray) -> np.ndarray:
    """(mull_idx, on_play, lands) probability of mulligan from raw counts: each play/draw cell is shrunk towards the pooled
    frequency of its lands value, which is shrunk towards the generic U-shaped prior."""
    out = np.zeros(counts.shape[:3])
    for j in range(counts.shape[0]):
        pooled = counts[j].sum(0)                                    # (8, 2)
        thin = pooled[:, 0] < MIN_POOLED            # (almost) never-seen hands in a smoothed queue: generic prior only
        w0 = np.where(thin, 1e9, SHRINK_PRIOR)
        p_pool = (pooled[:, 1] + w0 * DEFAULT_KEEP_PRIOR) / (pooled[:, 0] + w0)
        for pl_ in range(2):
            n, k = np.where(thin, 0.0, counts[j, pl_, :, 0]), np.where(thin, 0.0, counts[j, pl_, :, 1])
            out[j, pl_] = (k + SHRINK_PLAY * p_pool) / (n + SHRINK_PLAY)
    return np.clip(out, 0.0, 1.0)


SET_KEEP_SHRINK = 60.0     # pseudo-observations pulling a set's (mulligan number, play/draw, lands) cell towards the shipped average
MIN_SET_GAMES = 1500       # replay games a regime needs before its own keep rule is used instead of the shipped average
KEEP_RULE_VERSION = 1


def set_keep_table(counts: np.ndarray, default: np.ndarray, m: float = SET_KEEP_SHRINK) -> np.ndarray:
    """(2 mulligan numbers, 2, 8) probability of a mulligan from a set's own counts, each cell shrunk towards `default`."""
    n, k = counts[:2, :, :, 0], counts[:2, :, :, 1]
    return np.clip((k + m * np.asarray(default, dtype=float)) / (n + m), 0.0, 1.0)


def keep_regimes(set_code: str, cancel: threading.Event | None = None, progress: Progress | None = None,
                 refresh: bool = False) -> tuple[dict, dict]:
    """The set's own keep rule per regime as `manasim.SimRegime` objects, measured from its replay data (`keep_counts`) and shrunk
    towards the shipped average; regimes with too few replay games (or sets without replay data) keep the shipped average.
    Returns ({regime: SimRegime}, info) with info[regime] = dict(n_games, mull_rate, measured). Cached in the set's cache folder
    per replay-file fingerprints."""
    from sealed_lab.ingest import replay
    from sealed_lab.model import manasim

    set_code = set_code.upper()
    default = {r: manasim.SimRegime(r) for r in config.REGIMES}
    info: dict = {r: dict(n_games=0, mull_rate=None, measured=False) for r in config.REGIMES}
    if not replay.replay_files(set_code):
        return default, info
    replay.ingest_replays(set_code, progress=progress, cancel=cancel)
    sig = json.dumps([KEEP_RULE_VERSION, replay.REPLAY_VERSION, sorted(m["fingerprint"] for m in replay._shards(set_code, None)),
                      manasim.SIM2_REVISION])
    path = config.cache_dir(set_code) / "sim_keep.json"
    if path.exists() and not refresh:
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
            if d.get("sig") == sig:
                return ({r: manasim.SimRegime.make(r, d["keep"].get(r)) for r in config.REGIMES}, d["info"])
        except (OSError, ValueError, KeyError):
            pass
    try:
        h = hands_frame(set_code, cancel=cancel)
    except FileNotFoundError:
        return default, info
    out, keep_json = {}, {}
    for r in config.REGIMES:
        d = h.filter(pl.col("regime") == r)
        info[r] = dict(n_games=int(d.height), mull_rate=float((d["num_mulligans"] > 0).mean()) if d.height else None, measured=False)
        if d.height >= MIN_SET_GAMES:
            tab = set_keep_table(keep_counts(d), np.asarray(manasim._KEEP[r]))
            out[r] = manasim.SimRegime.make(r, tab)
            keep_json[r] = np.round(tab, 4).tolist()
            info[r]["measured"] = True
        else:
            out[r] = manasim.SimRegime(r)
            keep_json[r] = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(sig=sig, keep=keep_json, info=info)), encoding="utf-8")
    except OSError:
        log.warning("could not write the keep-rule cache")
    return out, info


# ---------------------------------------------------------------------------------------------
# hand smoother
# ---------------------------------------------------------------------------------------------

def hand_pmf(L: int, n: int = 40, h: int = 7, s: float | None = None, g: float = 2.5) -> np.ndarray:
    """P(l lands in the first `h` cards), l = 0..h: hypergeometric, times the smoother's acceptance weight if s is given."""
    ls = np.arange(h + 1)
    p = hypergeom(n, L, h).pmf(ls)
    if s is not None:
        p = p * np.exp(-(np.abs(ls - h * L / n) / s) ** g)
    return p / p.sum()


def hand_tables(h: pl.DataFrame, lands: range = range(14, 21), min_n: int = 200) -> dict[int, np.ndarray]:
    """{deck lands: counts of lands in the first 7-card hand shown} for the games of `h`."""
    d = h.filter((pl.col("n_h1") == 7) & pl.col("n_lands").is_in(list(lands)))
    out: dict[int, np.ndarray] = {}
    for L in lands:
        x = d.filter(pl.col("n_lands") == L)["l_h1"].to_numpy()
        if len(x) >= min_n:
            out[L] = np.bincount(x, minlength=8)[:8].astype(float)
    return out


def fit_hand_smoother(tabs: dict[int, np.ndarray]) -> dict:
    """Maximum-likelihood (s, g) of the acceptance rule on the hand tables, plus the fit of the plain hypergeometric
    and the deviance of both against the saturated model."""
    if not tabs:
        return dict(status="no data")
    sat = -sum(float((c * np.log(np.maximum(c / c.sum(), 1e-300))).sum()) for c in tabs.values())

    def nll(x):
        s, g = abs(x[0]), abs(x[1])
        return -sum(float((c * np.log(np.maximum(hand_pmf(L, s=s, g=g), 1e-12))).sum()) for L, c in tabs.items())

    plain = -sum(float((c * np.log(np.maximum(hand_pmf(L), 1e-12))).sum()) for L, c in tabs.items())
    best = None
    for x0 in ([1.0, 2.5], [1.5, 2.0], [3.0, 3.0]):
        r = minimize(nll, x0, method="Nelder-Mead", options=dict(xatol=1e-4, fatol=1e-4, maxiter=2000))
        if best is None or r.fun < best.fun:
            best = r
    n = int(sum(c.sum() for c in tabs.values()))
    s, g = abs(float(best.x[0])), abs(float(best.x[1]))
    smoothed = (hand_pmf(17, s=s, g=g) if 17 in tabs else None)
    return dict(status="ok", s=s, g=g, n=n, deviance_fit=float(2 * (best.fun - sat)), deviance_plain=float(2 * (plain - sat)),
                # a clear smoother: the plain hypergeometric is far from the data and the acceptance rule is close
                smoothed=bool(2 * (plain - sat) > 20 * max(2 * (best.fun - sat), 1.0) and s < 5.0),
                p3_at_17=None if smoothed is None else float(smoothed[3]))


# ---------------------------------------------------------------------------------------------
# draw smoothing
# ---------------------------------------------------------------------------------------------

def draw_tilt(h: pl.DataFrame, lands: range = range(15, 19)) -> dict:
    """Logit slope per land of deviation for the next single-card draw being a land (kept 7-card hands only):
    logit P(land) = logit(hypergeometric) + b * (lands seen - cards seen * L / 40).  b = 0 for a random shuffle; the draws of
    a long game are conditioned on its survival, so even random queues give a small negative b (about -0.03)."""
    d = h.filter((pl.col("num_mulligans") == 0) & (pl.col("n_h1") == 7) & pl.col("n_lands").is_in(list(lands)) & pl.col("l_h1").is_not_null())
    if d.height < 500:
        return dict(status="too little data", n=int(d.height))
    L = d["n_lands"].to_numpy().astype(float)
    lseen = d["l_h1"].to_numpy().astype(float)
    seen = np.full(d.height, 7.0)
    alive = np.ones(d.height, dtype=bool)
    off, dev, y = [], [], []
    for k in range(1, 9):
        nd, ld = d[f"nd_{k}"].to_numpy(), d[f"ld_{k}"].to_numpy()
        single = alive & (nd == 1)
        idx = np.nonzero(single)[0]
        p = (L[idx] - lseen[idx]) / (40.0 - seen[idx])
        ok = (p > 0) & (p < 1)
        idx, p = idx[ok], p[ok]
        off.append(np.log(p / (1 - p)))
        dev.append(lseen[idx] - seen[idx] * L[idx] / 40.0)
        y.append(ld[idx].astype(float))
        alive = alive & (nd <= 1)                        # drop games with multi-card draws (the order inside is unknown)
        lseen = lseen + ld
        seen = seen + nd
    off, dev, y = np.concatenate(off), np.concatenate(dev), np.concatenate(y)
    if len(y) < 500:
        return dict(status="too little data", n=int(len(y)))

    def nll(b):
        z = off + b[0] * dev
        return float(np.sum(np.logaddexp(0, z) - y * z))

    r = minimize(nll, [0.0], method="BFGS")
    se = float(np.sqrt(max(r.hess_inv[0, 0], 0.0)))
    return dict(status="ok", b=float(r.x[0]), se=se, n=int(len(y)))


def rescue_lift(h: pl.DataFrame, lands: range = range(15, 19), min_turns: int = 7) -> dict:
    """Land rescue: how much more likely a single-card draw is a land when the player holds no land at that moment (every land
    played), by turn 3, 4, 5+, against the hypergeometric probability. Only games that lasted `min_turns` turns, so that the draw
    itself cannot decide whether the game is in the sample. In every queue this lift is a few points (about +7 / +4 / +3 pp)."""
    d = h.filter((pl.col("num_mulligans") == 0) & (pl.col("n_h1") == 7) & pl.col("n_lands").is_in(list(lands)) & pl.col("l_h1").is_not_null()
                 & (pl.col("turns_done") >= min_turns))
    if d.height < 500:
        return dict(status="too little data", n=int(d.height))
    L = d["n_lands"].to_numpy().astype(float)
    lseen = d["l_h1"].to_numpy().astype(float)
    seen = np.full(d.height, 7.0)
    alive = np.ones(d.height, dtype=bool)
    acc = {3: [0.0, 0.0, 0], 4: [0.0, 0.0, 0], 5: [0.0, 0.0, 0]}      # sum of land, sum of hypergeometric p, n
    for k in range(1, 9):
        nd, ld = d[f"nd_{k}"].to_numpy(), d[f"ld_{k}"].to_numpy()
        prev = d[f"lands_{k - 1}"].cast(pl.Float64).to_numpy() if k > 1 else np.zeros(d.height)
        held = lseen - prev
        ok = alive & (nd == 1) & np.isfinite(prev) & (held == 0) & (k >= 3)
        idx = np.nonzero(ok)[0]
        if len(idx):
            p = (L[idx] - lseen[idx]) / (40.0 - seen[idx])
            a = acc[min(k, 5)]
            a[0] += float(ld[idx].sum())
            a[1] += float(p.sum())
            a[2] += int(len(idx))
        alive = alive & (nd <= 1)
        lseen = lseen + ld
        seen = seen + nd
    out = {}
    for t, (y, p, n) in acc.items():
        if n >= 200:
            out[t] = dict(n=n, lift_pp=100.0 * (y - p) / n, p_land=y / n)
    return dict(status="ok" if out else "too little data", by_turn=out)


# ---------------------------------------------------------------------------------------------
# card flow: extra cards drawn
# ---------------------------------------------------------------------------------------------

def extra_draws(h: pl.DataFrame, upto: int = 5) -> dict:
    """Mean cards drawn (draw step, effects and tutors) beyond one per own turn by the end of own turn k (k = 2..upto), per
    event, for kept 7-card hands in games that lasted k turns (the replay files only describe completed turns)."""
    out = {}
    d = h.filter((pl.col("num_mulligans") == 0) & (pl.col("n_h1") == 7))
    op = d["on_play"].to_numpy()
    for k in range(2, upto + 1):
        m = (d["turns_done"] >= k).to_numpy()
        tot = sum(d[f"nd_{j}"].to_numpy() for j in range(1, k + 1))
        base = np.where(op, k - 1, k)
        out[k] = float((tot - base)[m].mean()) if m.any() else None
    return out


def calibrate(sets: list[str], progress: Progress | None = None, cancel: threading.Event | None = None) -> dict:
    """Measure the simulator's parameters on the replay data of `sets`, per regime. JSON-able."""
    frames = []
    for i, s in enumerate(sets):
        _check(cancel)
        if progress:
            progress(i / max(len(sets), 1), f"Calibration data {s}")
        try:
            frames.append(hands_frame(s, cancel=cancel))
        except FileNotFoundError:
            log.warning("no replay shards for %s", s)
    if not frames:
        return dict(status="no replay data")
    h = pl.concat(frames, how="diagonal_relaxed")
    out: dict = dict(status="ok", sets=list(sets), regimes={})
    for reg in config.REGIMES:
        d = h.filter(pl.col("regime") == reg)
        if d.height < 1000:
            continue
        cnt = keep_counts(d)
        out["regimes"][reg] = dict(
            n_games=int(d.height), events=sorted(d["event"].unique().to_list()),
            mull_rate=float((d["num_mulligans"] > 0).mean()),
            keep_counts=cnt.tolist(), keep=np.round(smooth_keep(cnt), 3).tolist(),
            hand=fit_hand_smoother(hand_tables(d)), draw_tilt=draw_tilt(d), rescue=rescue_lift(d), extra_draws=extra_draws(d))
    return out


def dumps(o: dict) -> str:
    return json.dumps(o, indent=1)
