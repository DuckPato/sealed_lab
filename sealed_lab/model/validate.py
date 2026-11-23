"""Out-of-sample validation: time-based hold-out, baselines B0/B1/B2, bootstrap CIs, calibration, variance budget,
simulator validation - and the self-contained HTML report (`render_report`)."""
from __future__ import annotations

import html
import logging
import math
import os
import threading
import time
from typing import Callable

import numpy as np
import polars as pl
import scipy.sparse as sp
from scipy.special import expit
from scipy.stats import norm

from sealed_lab import config
from sealed_lab.model import cardvalue, design, glm, vcache
from sealed_lab.model.design import DeckSpace, StageData
from sealed_lab.types import Cancelled

log = logging.getLogger(__name__)
Progress = Callable[[float, str], None]

HOLDOUT_FRACTION = 0.25
MIN_EVENTS = 30
MIN_GAMES = 300
MIN_PRIOR_GAMES = 200
GIH_SHRINK = 50.0
N_REP = 500
RECENCY_GRID = (60.0, 30.0, 14.0, 7.0)       # half-lives in days tried against "no weighting"
MODELS = ["M", "MR", "MA", "MX", "MS", "MC", "MO", "MP", "MM", "P", "B0", "B1", "B1d", "B2"]
MODEL_LABEL = {
    "M": "Full model (chain)",
    "MR": "Full model, recency-weighted",
    "MA": "Full model + card-attribute prior",
    "MX": "Full model + cross-set structure prior",
    "MS": "Full model + card-pair synergies",
    "MC": "Full model + archetype-conditional card values",
    "MO": "Full model + card x opponent-colour interactions",
    "MP": "Full model + card x play/draw interactions",
    "MM": "Full model with the calibrated mana simulator (v2)",
    "P": "Prior only (earlier stage x lambda)",
    "B0": "B0: covariates only",
    "B1": "B1: B0 + mean GIH WR of deck spells",
    "B1d": "B1d: B1 with the GIH WR taken from the draft games before the cut",
    "B2": "B2: B1 + colour pair",
}


def _check(cancel: threading.Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise Cancelled()


# ---------------------------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------------------------

def wilson(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(((y - p) ** 2).mean())


def auc_weighted(inv: np.ndarray, n_unique: int, y: np.ndarray, w: np.ndarray) -> float:
    pos = np.bincount(inv, weights=w * y, minlength=n_unique)
    neg = np.bincount(inv, weights=w * (1 - y), minlength=n_unique)
    below = np.cumsum(neg) - neg
    den = pos.sum() * neg.sum()
    return float((pos * (below + 0.5 * neg)).sum() / den) if den > 0 else float("nan")


def auc(y: np.ndarray, p: np.ndarray) -> float:
    u, inv = np.unique(p, return_inverse=True)
    return auc_weighted(inv, len(u), y, np.ones(len(y)))


def cluster_metrics(y: np.ndarray, preds: dict[str, np.ndarray], groups: np.ndarray, n_rep: int, seed: int) -> dict:
    """Point metrics and cluster-bootstrap (resampling draft_ids) CIs, plus baseline-minus-model differences."""
    G = int(groups.max()) + 1
    rng = np.random.default_rng(seed)
    mult = rng.multinomial(G, np.full(G, 1.0 / G), size=n_rep).astype(float)      # (n_rep, G)
    n_g = np.bincount(groups, minlength=G).astype(float)
    res: dict = {}
    reps: dict[str, dict[str, np.ndarray]] = {}
    for name, p in preds.items():
        pc = np.clip(p, 1e-9, 1 - 1e-9)
        ll = -(y * np.log(pc) + (1 - y) * np.log(1 - pc))
        br = (y - p) ** 2
        ll_g = np.bincount(groups, weights=ll, minlength=G)
        br_g = np.bincount(groups, weights=br, minlength=G)
        den = mult @ n_g
        u, inv = np.unique(p, return_inverse=True)
        aucs = np.empty(n_rep)
        for r in range(n_rep):
            aucs[r] = auc_weighted(inv, len(u), y, mult[r][groups])
        reps[name] = dict(logloss=(mult @ ll_g) / den, brier=(mult @ br_g) / den, auc=aucs)
        res[name] = dict(logloss=log_loss(y, p), brier=brier(y, p), auc=auc(y, p))
        for m in ("logloss", "brier", "auc"):
            lo, hi = np.percentile(reps[name][m], [2.5, 97.5])
            res[name][m + "_ci"] = [float(lo), float(hi)]
    if "M" in preds:
        for name in preds:
            if name == "M":
                continue
            d = {}
            for m, sign in (("logloss", 1.0), ("brier", 1.0), ("auc", -1.0)):
                # positive = the full model is better than `name`
                diff = sign * (reps[name][m] - reps["M"][m])
                point = sign * (res[name][m] - res["M"][m])
                lo, hi = np.percentile(diff, [2.5, 97.5])
                d[m] = dict(diff=float(point), lo=float(lo), hi=float(hi))
            res[name]["vs_M"] = d
    return res


def half_masks(g_ho: pl.DataFrame) -> np.ndarray:
    """Per event (indexed like the `groups` of the hold-out, i.e. np.unique order of draft_id): True for the FIRST half of the
    hold-out events by start time (half A, the decision half), False for the second half (half B, the confirmation half)."""
    order = design.event_starts(g_ho)["draft_id"].to_list()
    first = set(order[:(len(order) + 1) // 2])
    return np.array([u in first for u in np.unique(g_ho["draft_id"].to_numpy())])


def half_gains(y: np.ndarray, p_base: np.ndarray, p_new: np.ndarray, groups: np.ndarray, is_first: np.ndarray, n_rep: int,
               seed: int) -> dict:
    """Log-loss gain of `p_new` over `p_base` (positive = p_new better) on each half of the hold-out events, with
    cluster-bootstrap intervals over the events OF THAT HALF: 95% two-sided (lo, hi) and one-sided 90% (lo90, hi90).
    Half A = the earlier events, half B = the later ones (see `half_masks`)."""
    d = (-(y * np.log(np.clip(p_base, 1e-9, 1 - 1e-9)) + (1 - y) * np.log(np.clip(1 - p_base, 1e-9, 1 - 1e-9)))
         + (y * np.log(np.clip(p_new, 1e-9, 1 - 1e-9)) + (1 - y) * np.log(np.clip(1 - p_new, 1e-9, 1 - 1e-9))))
    G = int(groups.max()) + 1
    d_g = np.bincount(groups, weights=d, minlength=G)
    n_g = np.bincount(groups, minlength=G).astype(float)
    out = {}
    for name, sel in (("A", np.nonzero(is_first)[0]), ("B", np.nonzero(~is_first)[0])):
        if len(sel) == 0:
            out[name] = dict(gain=0.0, lo=0.0, hi=0.0, lo90=0.0, hi90=0.0, n_events=0, n_games=0)
            continue
        mult = np.random.default_rng([seed, 17 if name == "A" else 29]).multinomial(
            len(sel), np.full(len(sel), 1.0 / len(sel)), size=n_rep).astype(float)
        reps = (mult @ d_g[sel]) / (mult @ n_g[sel])
        lo, hi = np.percentile(reps, [2.5, 97.5])
        lo90, hi90 = np.percentile(reps, [10, 90])
        out[name] = dict(gain=float(d_g[sel].sum() / n_g[sel].sum()), lo=float(lo), hi=float(hi), lo90=float(lo90), hi90=float(hi90),
                         n_events=int(len(sel)), n_games=int(n_g[sel].sum()))
    return out


def calibration(y: np.ndarray, p: np.ndarray, groups: np.ndarray, n_bins: int = 10, n_rep: int = 300, seed: int = 0) -> dict:
    order = np.argsort(p, kind="stable")
    bins = np.array_split(order, n_bins)
    G = int(groups.max()) + 1
    rng = np.random.default_rng(seed + 1)
    mult = rng.multinomial(G, np.full(G, 1.0 / G), size=n_rep).astype(float)
    rows = []
    for b in bins:
        if len(b) == 0:
            continue
        k, n = float(y[b].sum()), float(len(b))
        lo, hi = wilson(k, n)
        gb = groups[b]
        ky = np.bincount(gb, weights=y[b], minlength=G)
        nb = np.bincount(gb, minlength=G).astype(float)
        den = mult @ nb
        with np.errstate(invalid="ignore", divide="ignore"):
            rr = (mult @ ky) / den
        rr = rr[np.isfinite(rr)]
        clo, chi = (np.percentile(rr, [2.5, 97.5]) if len(rr) else (lo, hi))
        rows.append(dict(n=int(n), pred=float(p[b].mean()), obs=k / n, wilson=[lo, hi], cluster=[float(clo), float(chi)]))
    # calibration slope / intercept: y ~ a + b * logit(p)
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))

    def slope(w):
        prob = glm.Problem(y=y, C=sp.csr_matrix(np.column_stack([np.ones(len(y)), lp])))
        r = glm.fit(prob, np.zeros(2), np.full(2, 100.0), w=w, maxiter=100)
        return r.coef

    full = slope(np.ones(len(y)))
    sl = []
    for r in range(min(n_rep, 150)):
        sl.append(slope(mult[r][groups]))
    sl = np.array(sl)
    # level of the predictions: slope fixed at 1 (an offset only) and the plain gap between observed and predicted win rate, with
    # cluster-bootstrap CIs. Absolute win rates (game / match win rate, expected wins) inherit this error; rankings and scores do not.
    pc_ = np.clip(p, 1e-6, 1 - 1e-6)

    def offset(w):
        a = 0.0
        for _ in range(25):
            q = 1.0 / (1.0 + np.exp(-(lp + a)))
            g = float((w * (y - q)).sum())
            h = float((w * q * (1 - q)).sum())
            if h <= 0:
                break
            step = g / h
            a += step
            if abs(step) < 1e-9:
                break
        return a

    w1 = np.ones(len(y))
    a_full = offset(w1)
    gap_full = 100.0 * float((y - pc_).mean())
    offs, gaps = [], []
    for r in range(min(n_rep, 150)):
        w = mult[r][groups]
        offs.append(offset(w))
        gaps.append(100.0 * float((w * (y - pc_)).sum() / max(w.sum(), 1e-9)))
    return dict(bins=rows, intercept=float(full[0]), slope=float(full[1]),
                slope_ci=[float(np.percentile(sl[:, 1], 2.5)), float(np.percentile(sl[:, 1], 97.5))],
                intercept_ci=[float(np.percentile(sl[:, 0], 2.5)), float(np.percentile(sl[:, 0], 97.5))],
                level_logit=float(a_full), level_logit_ci=[float(np.percentile(offs, 2.5)), float(np.percentile(offs, 97.5))],
                level_pp=gap_full, level_pp_ci=[float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5))],
                n=int(len(y)))


def variance_budget(y, p0, pM, groups_build, n_rep, seed) -> dict:
    """Method-of-moments deck-level variance of win rate (net of covariates) and the share the model predicts."""
    B = int(groups_build.max()) + 1
    n_i = np.bincount(groups_build, minlength=B).astype(float)
    ok = n_i > 0
    r_i = np.bincount(groups_build, weights=y - p0, minlength=B) / np.maximum(n_i, 1)
    m_i = np.bincount(groups_build, weights=pM - p0, minlength=B) / np.maximum(n_i, 1)
    v_i = np.bincount(groups_build, weights=pM * (1 - pM), minlength=B) / np.maximum(n_i, 1) ** 2
    return dict(n_i=n_i, r=r_i, m=m_i, v=v_i, ok=ok)


def _vb_stats(n_i, r, m, v, w_extra=None):
    w = n_i if w_extra is None else n_i * w_extra
    W = w.sum()
    if W <= 0:
        return None
    rbar, mbar = (w * r).sum() / W, (w * m).sum() / W
    var_obs = (w * (r - rbar) ** 2).sum() / W
    noise = (w * v).sum() / W
    s2 = var_obs - noise
    pv = (w * (m - mbar) ** 2).sum() / W
    cov = (w * (r - rbar) * (m - mbar)).sum() / W
    return dict(var_obs=float(var_obs), noise=float(noise), sigma2_deck=float(s2), pred_var=float(pv), cov=float(cov),
                share_raw=float(pv / s2) if s2 > 0 else float("nan"),
                slope=float(cov / pv) if pv > 0 else float("nan"),
                share_recal=float(cov * cov / (pv * s2)) if (pv > 0 and s2 > 0) else float("nan"))


def variance_budget_ci(y, p0, pM, game_build, build_group, n_group, n_rep, seed) -> dict:
    vb = variance_budget(y, p0, pM, game_build, n_rep, seed)
    point = _vb_stats(vb["n_i"], vb["r"], vb["m"], vb["v"])
    rng = np.random.default_rng(seed + 7)
    mult = rng.multinomial(n_group, np.full(n_group, 1.0 / n_group), size=n_rep).astype(float)
    outs = {k: [] for k in ("sigma2_deck", "share_raw", "share_recal", "slope", "pred_var")}
    for r in range(n_rep):
        s = _vb_stats(vb["n_i"], vb["r"], vb["m"], vb["v"], mult[r][build_group])
        if s is None:
            continue
        for k in outs:
            outs[k].append(s[k])
    ci = {}
    for k, v in outs.items():
        v = np.array(v, dtype=float)
        v = v[np.isfinite(v)]
        ci[k] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) > 20 else [float("nan")] * 2
    point["ci"] = ci
    point["n_builds"] = int(vb["ok"].sum())
    return point


# ---------------------------------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------------------------------

def gih_table(hands: pl.LazyFrame, games_train: pl.DataFrame, n_cards: int, p0: float) -> np.ndarray:
    """Shrunk games-in-hand win rate per card from the training games of the target format."""
    gdf = games_train.select("game_key", pl.col("won").cast(pl.Float64).alias("won")).lazy()
    h = (hands.filter((pl.col("opening") + pl.col("drawn")) > 0).select("game_key", "card_id")
         .join(gdf, on="game_key", how="inner").group_by("card_id").agg(pl.len().alias("n"), pl.col("won").sum().alias("w"))
         .collect())
    n = np.zeros(n_cards)
    w = np.zeros(n_cards)
    cid = h["card_id"].to_numpy().astype(np.int64)
    n[cid] = h["n"].to_numpy()
    w[cid] = h["w"].to_numpy()
    return (w + GIH_SHRINK * p0) / (n + GIH_SHRINK)


def deck_gih_feature(space: DeckSpace, gih: np.ndarray, p0: float, nb_is_land: np.ndarray) -> np.ndarray:
    """Per deck-space row: copy-weighted mean of (GIH WR - p0)*10 over the deck's non-land cards (0 if none)."""
    Dc = space.D[:, :space.n_card]
    spell = (~nb_is_land).astype(float)
    v = 10.0 * (gih[space.nb_ids] - p0) * spell
    num = np.asarray(Dc @ v).ravel()
    den = np.asarray(Dc @ spell).ravel()
    return np.where(den > 0, num / np.maximum(den, 1e-9), 0.0)


def _pair_onehot(main_colors: np.ndarray) -> sp.csr_matrix:
    uniq = {v: design.norm_pair(v) for v in np.unique(main_colors)}
    cat = np.array([design.OPP_LEVELS.index(uniq[v]) for v in main_colors], dtype=np.int64)
    n = len(cat)
    return sp.csr_matrix((np.ones(n), (np.arange(n), cat)), shape=(n, len(design.OPP_LEVELS)))


def _fit_cov_model(Ctr: sp.csr_matrix, y: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    prob = glm.Problem(y=y, C=Ctr)
    return glm.fit(prob, np.zeros(len(sigma)), sigma, maxiter=400).coef


# ---------------------------------------------------------------------------------------------
# the validation run
# ---------------------------------------------------------------------------------------------

def prior_stage_names(fmt: str, n_games: dict[str, int]) -> list[str]:
    """Stages (in chain order) used before the target stage `fmt`."""
    out = []
    if n_games.get("draft", 0) >= MIN_PRIOR_GAMES:
        out.append("draft")
    if fmt == "bo3" and n_games.get("bo1", 0) >= MIN_PRIOR_GAMES:
        out.append("bo1")
    return out


def time_split(g: pl.DataFrame, fraction: float = HOLDOUT_FRACTION) -> tuple[list[str], object] | None:
    """(ids of the last `fraction` of events by start time, start time of the first of them); None if too small."""
    ev = design.event_starts(g)
    if ev.height < MIN_EVENTS or g.height < MIN_GAMES:
        return None
    n_hold = max(5, int(round(fraction * ev.height)))
    return ev["draft_id"][-n_hold:].to_list(), ev["start"][-n_hold]


def chain_stages(fmt: str, games: pl.DataFrame, g_target: pl.DataFrame, t_cut, space: DeckSpace) -> dict[str, StageData]:
    """Prior stages restricted to events that started before `t_cut`, plus the target stage on `g_target`."""
    n_games = {f: int(games.filter(pl.col("fmt") == f).height) for f in ("draft", "bo1", "bo3")}
    stages: dict[str, StageData] = {}
    for name in prior_stage_names(fmt, n_games):
        gp = games.filter(pl.col("fmt") == name)
        evp = design.event_starts(gp)
        ids = evp.filter(pl.col("start") < t_cut)["draft_id"].to_list()
        gp = gp.filter(pl.col("draft_id").is_in(ids))
        if gp.height >= MIN_PRIOR_GAMES:
            stages[name] = design.build_stage(name, gp, space)
    stages[fmt] = design.build_stage(fmt, g_target, space)
    return stages


def weight_stages(stages: dict[str, StageData], t_ref, half_life: "float | dict[str, float | None] | None") -> dict[str, StageData]:
    """Recency weights per stage. `half_life` is days, or {stage: days} (a stage not listed keeps equal weights: the half-life of a
    format weights that format's own games only). A plain number weights every stage except the draft stage, which is always
    equal-weighted."""
    def h_of(n: str):
        if isinstance(half_life, dict):
            return half_life.get(n)
        return None if n == "draft" else half_life
    return {n: design.with_weights(st, design.recency_weights(st, t_ref, h_of(n))) for n, st in stages.items()}


def recency_inner(fmt: str, games: pl.DataFrame, space: DeckSpace, tmpdir: str, n_jobs: int | None,
                  grid=RECENCY_GRID, seed: int = 0, cancel: threading.Event | None = None,
                  progress: Progress | None = None, cache: "vcache.ValCache | None" = None, scope: str = "format",
                  opts: "cardvalue.ChainOpts | None" = None) -> dict:
    """Choose-a-half-life data for one format, using ONLY the training part of the outer hold-out split.

    Inner split: the last 25% (by start) of the outer-training events are scored; the chain is fitted on the events
    before them. Sigmas come from one unweighted CV fit; every half-life is then refitted at those sigmas, so the
    comparison isolates the weighting. Returns per-half-life mean log loss and the paired difference vs no weighting
    (positive = weighting better) with a 95% cluster-bootstrap CI over the inner-validation events.

    `scope` "format" weights only the target format's own games (the draft stage and the prior Bo1 stage keep equal weights), so the
    half-life of each format is chosen and tested on its own; "pooled" weights every stage but the draft stage."""
    opts = opts or cardvalue.ChainOpts()
    gT = games.filter(pl.col("fmt") == fmt)
    outer = time_split(gT)
    if outer is None:
        return dict(status="skipped", reason="too little data")
    g_tr = gT.filter(~pl.col("draft_id").is_in(outer[0]))
    inner = time_split(g_tr)
    if inner is None:
        return dict(status="skipped", reason="too little training data for an inner split")
    iv_ids, t_in = inner
    key = hit = None
    if cache is not None and cache.enabled:
        key = cache.key("rec", f"{fmt}_inner", fmt=fmt, seed=int(seed), grid=[float(g) for g in grid], space=vcache.space_digest(space),
                        scope=scope, opts=opts.key())
        hit = cache.load("rec", f"{fmt}_inner", key)
        if hit is not None:
            res = hit[0]
            res["cache"] = dict(hit=True, created=res.pop("_created", None))
            if progress:
                progress(1.0, f"Recency check ({fmt}) (cached)")
            return res
    g_it, g_iv = g_tr.filter(~pl.col("draft_id").is_in(iv_ids)), g_tr.filter(pl.col("draft_id").is_in(iv_ids))
    stages = chain_stages(fmt, games, g_it, t_in, space)
    prior_ok = set(stages) - {fmt}
    vt = os.path.join(tmpdir, f"rec_inner_{fmt}")
    os.makedirs(vt, exist_ok=True)
    ref = cardvalue.run_chain(stages, space.n_card, space.n_feat, vt, n_jobs, seed=seed + 300, cancel=cancel,
                              progress=(lambda f, m: progress(0.6 * f, m)) if progress else None, prior_ok=prior_ok, opts=opts)
    prob_iv, _, _ = design.make_problem(g_iv, stages[fmt].enc, space, clamp_time=True)
    y = prob_iv.y
    groups = np.unique(g_iv["draft_id"].to_numpy(), return_inverse=True)[1]
    G = int(groups.max()) + 1
    n_g = np.bincount(groups, minlength=G).astype(float)
    mult = np.random.default_rng(seed + 31).multinomial(G, np.full(G, 1.0 / G), size=400).astype(float)

    def per_game_loss(coef):
        p = np.clip(expit(prob_iv.eta(coef)), 1e-9, 1 - 1e-9)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))

    base = per_game_loss(ref[fmt].coef)
    base_g = np.bincount(groups, weights=base, minlength=G)
    rows = [dict(half_life=None, logloss=float(base.mean()), diff=0.0, lo=0.0, hi=0.0)]
    for i, h in enumerate(grid):
        _check(cancel)
        if progress:
            progress(0.6 + 0.4 * i / len(grid), f"Recency check ({fmt}): half-life {h:g} days")
        coefs = cardvalue.refit_chain(weight_stages(stages, t_in, {fmt: h} if scope == "format" else h), ref, prior_ok=prior_ok,
                                      cancel=cancel)
        loss = per_game_loss(coefs[fmt])
        d_g = base_g - np.bincount(groups, weights=loss, minlength=G)
        reps = (mult @ d_g) / (mult @ n_g)
        rows.append(dict(half_life=float(h), logloss=float(loss.mean()), diff=float(base.mean() - loss.mean()),
                         lo=float(np.percentile(reps, 2.5)), hi=float(np.percentile(reps, 97.5))))
    res = dict(status="ok", rows=rows, n_inner_val_events=G, n_inner_val_games=int(len(y)), inner_start=str(t_in),
               n_inner_train_games=int(g_it.height))
    if cache is not None and key is not None:
        res = vcache.normalise(res)
        cache.save("rec", f"{fmt}_inner", key, res)
        res["cache"] = dict(hit=False, created=None)
    return res


def choose_half_life(inner: dict[str, dict]) -> float | None:
    """Half-life with the best summed inner gain over the formats; None if no half-life beats no weighting."""
    ok = [v for v in inner.values() if v.get("status") == "ok"]
    if not ok:
        return None
    hs = [r["half_life"] for r in ok[0]["rows"] if r["half_life"] is not None]
    gain = {h: sum(next(r["diff"] for r in v["rows"] if r["half_life"] == h) for v in ok) for h in hs}
    best = max(gain, key=gain.get) if gain else None
    return best if best is not None and gain[best] > 0 else None


def best_finite_half_life(inner: dict[str, dict]) -> float:
    """Like choose_half_life but always returns a half-life (the least bad one); default 30 days without data."""
    ok = [v for v in inner.values() if v.get("status") == "ok"]
    if not ok:
        return 30.0
    hs = [r["half_life"] for r in ok[0]["rows"] if r["half_life"] is not None]
    gain = {h: sum(next(r["diff"] for r in v["rows"] if r["half_life"] == h) for v in ok) for h in hs}
    return max(gain, key=gain.get) if gain else 30.0


class RefFit:
    """The part of a plain-chain `StageFit` that the extra-block fits and the deployed model read (the card sigma and its CV
    curve) - small enough to be cached next to the hold-out predictions of the plain chain."""

    def __init__(self, sigma: float, cv_grid, cv_loss, cv_se, cv_done: bool, w: float = 1.0, w_grid=(), w_loss=(),
                 cv_scheme: str = "random", theta_cv_sd: float | None = None, theta_grid=(), theta_loss=(), theta_se=()):
        self.sigma = float(sigma)
        self.cv_grid, self.cv_loss, self.cv_se, self.cv_done = list(cv_grid), list(cv_loss), list(cv_se), bool(cv_done)
        self.w, self.w_grid, self.w_loss, self.cv_scheme = float(w), list(w_grid), list(w_loss), str(cv_scheme)
        # theta_sigma="cv": the structure prior SD the plain chain chose (None: the rule), reused by the extra-block fits
        self.theta_cv_sd = None if theta_cv_sd is None else float(theta_cv_sd)
        self.theta_grid, self.theta_loss, self.theta_se = list(theta_grid), list(theta_loss), list(theta_se)

    @classmethod
    def from_fit(cls, f) -> "RefFit":
        return cls(f.sigma, [float(v) for v in f.cv_grid], [float(v) for v in f.cv_loss], [float(v) for v in f.cv_se], f.cv_done,
                   f.w, [float(v) for v in f.w_grid], [float(v) for v in f.w_loss], f.cv_scheme, getattr(f, "theta_cv_sd", None),
                   [float(v) for v in getattr(f, "theta_grid", ())], [float(v) for v in getattr(f, "theta_loss", ())],
                   [float(v) for v in getattr(f, "theta_se", ())])

    @classmethod
    def from_dict(cls, d: dict) -> "RefFit":
        return cls(d["sigma"], d["cv_grid"], d["cv_loss"], d["cv_se"], d["cv_done"], d.get("w", 1.0), d.get("w_grid", ()),
                   d.get("w_loss", ()), d.get("cv_scheme", "random"), d.get("theta_cv_sd"), d.get("theta_grid", ()),
                   d.get("theta_loss", ()), d.get("theta_se", ()))

    def to_dict(self) -> dict:
        return dict(sigma=self.sigma, cv_grid=self.cv_grid, cv_loss=self.cv_loss, cv_se=self.cv_se, cv_done=self.cv_done,
                    w=self.w, w_grid=self.w_grid, w_loss=self.w_loss, cv_scheme=self.cv_scheme, theta_cv_sd=self.theta_cv_sd,
                    theta_grid=self.theta_grid, theta_loss=self.theta_loss, theta_se=self.theta_se)


# component code -> key of its result in the validation dict, in the order the models are scored
COMPONENTS = (("MR", "recency"), ("MA", "attributes"), ("MX", "crossset"), ("MS", "synergy"), ("MC", "archcond"),
              ("MM", "manasim"), ("MK", "castability"), ("MO", "opponent"), ("MP", "playdraw"), ("MF", "oppflags"))


def opponent_grid() -> np.ndarray:
    """The prior-SD grid of the pooled interaction layers (matchups, play/draw): part of the play/draw cache key."""
    from sealed_lab.model import opponent as om

    return om.SIGMA_GRID


def _holdout_split(gT: pl.DataFrame):
    """(hold-out draft_ids, cut time, n_events, event table) of the time split of one format, or None when too small."""
    ev = design.event_starts(gT)
    n_ev = ev.height
    if n_ev < MIN_EVENTS or gT.height < MIN_GAMES:
        return None
    n_hold = max(5, int(round(HOLDOUT_FRACTION * n_ev)))
    return ev["draft_id"][-n_hold:].to_list(), ev["start"][-n_hold], n_ev, n_hold


def run_validation(fmt: str, games: pl.DataFrame, space: DeckSpace, cards: pl.DataFrame, hands: pl.LazyFrame,
                   tmpdir: str, n_jobs: int | None, seed: int = 0, n_rep: int = N_REP,
                   cancel: threading.Event | None = None, progress: Progress | None = None,
                   recency_h: "float | None | Callable[[], float | None]" = None, space_a: DeckSpace | None = None,
                   xprior: dict | None = None, space_s: DeckSpace | None = None, space_c: DeckSpace | None = None,
                   opp_modes: tuple[str, ...] | None = None, syn_info: dict | None = None,
                   space_m: DeckSpace | None = None, space_k: DeckSpace | None = None,
                   cache: "vcache.ValCache | None" = None, runner=None, keep: dict | None = None,
                   recency_scope: str = "format", opts: "cardvalue.ChainOpts | None" = None,
                   flags_test: bool = False, only: tuple[str, ...] | None = None, playdraw: bool = False) -> dict:
    """Time-based hold-out validation for one target format.

    `flags_test`: also fit the chain with the opponent colour flags in its game covariates ("MF", design.OPP_FLAG_MARK) and score it
    against M. If `games` itself carries the marker column, M IS that flagged chain (the base the matchup layer is then tested on);
    `only` (codes) restricts the component variants fitted to those (the plain chain M is always fitted). Returns a JSON-able dict.

    `opts` (cardvalue.ChainOpts): prior-strength rule of every chain fitted here (time-split sigma and stacking by default; the
    plain chain "M" and all variants use the same rule). `recency_scope` "format": the recency half-life weights this format's own
    games only (the draft stage and the prior Bo1 stage keep equal weights); "pooled": every stage but the draft stage. Every
    component entry also carries `halves`: its log-loss gain over M on the earlier (A) and later (B) half of the hold-out events,
    for the confirmation rule (pipeline.decide_component).

    With `recency_h` the same chain is also fitted with recency weights (half-life in days, measured back from the
    hold-out cut) and scored on the same hold-out as model "MR" (a callable is resolved late, when the inner split that picks
    the half-life has finished).  With `space_a` (a DeckSpace with card-attribute
    columns, see design.with_attributes) the chain is refitted with the attribute prior beta_c = A_c.gamma + u_c
    ("MA"); with `xprior` (crossset.build_prior) the chain's first stage gets the cross-set structure prior ("MX").
    `space_s` (design.with_pairs) adds card-pair synergy columns ("MS"), `space_c` (design.with_archetype_columns) card x
    archetype columns ("MC"), both with their own prior SD chosen by CV at the plain chain's card sigmas; `opp_modes` runs
    the card x opponent-colour interaction layer on top of the chain ("MO", see model/opponent.py); `playdraw` the card x
    play/draw layer on top of the chain ("MP", see model/playdraw.py). `space_m` (a DeckSpace
    with regime-specific mana features, see pipeline.v2_space) refits the whole chain on the calibrated mana simulator ("MM").
    `space_k` (pipeline.cast_space) refits it with castability-weighted card copies and the dead-cards feature ("MK").
    Every variant is scored on the same hold-out, paired with M.

    Every variant (and the plain chain with its baselines) is an independent fit.  With a `runner` (parallel.Runner) they run
    concurrently, their cross-validation folds sharing one process pool.  With a `cache` (vcache.ValCache) each result is stored
    under a key covering all its inputs and reused when none changed ("cached from <date>" in the report).  `keep` (a dict)
    receives, under `fmt`, what the deployed-combination test needs of the plain chain (hold-out predictions, card sigmas)."""
    from sealed_lab.model import memory, parallel, vcache

    parallel_null = parallel._NULL
    t_start = time.time()
    runner = runner or parallel.InlineRunner(cancel)
    cache = cache or vcache.ValCache(None, {})
    gT = games.filter(pl.col("fmt") == fmt)
    out: dict = dict(fmt=fmt, status="ok", n_events=design.event_starts(gT).height, n_games=gT.height)
    split = _holdout_split(gT)
    if split is None:
        out.update(status="skipped", reason=f"too little data for a hold-out ({out['n_events']} events, {gT.height} games; "
                                            f"need >= {MIN_EVENTS} events and >= {MIN_GAMES} games)")
        return out
    hold_ids, t_cut, n_ev, n_hold = split
    out.update(n_holdout_events=n_hold, holdout_start=str(t_cut))
    g_tr = gT.filter(~pl.col("draft_id").is_in(hold_ids))
    g_ho = gT.filter(pl.col("draft_id").is_in(hold_ids))
    out.update(n_train_games=g_tr.height, n_holdout_games=g_ho.height)
    n_games = {f: int(games.filter(pl.col("fmt") == f).height) for f in ("draft", "bo1", "bo3")}
    prior_names = prior_stage_names(fmt, n_games)
    use_a = space_a is not None and space_a.n_attr > 0
    use_s = space_s is not None and space_s.n_pair > 0
    use_c = space_c is not None and space_c.n_archc > 0
    use_m = space_m is not None
    use_k = space_k is not None
    _check(cancel)

    # ---- what was asked, and the cache keys (the plain chain "core" and every component paired with it)
    opts = opts or cardvalue.ChainOpts()
    base_parts = dict(fmt=fmt, seed=int(seed), n_rep=int(n_rep), space=vcache.space_digest(space), opts=opts.key())
    if design.OPP_FLAG_MARK in games.columns:
        base_parts["oppflags_base"] = True              # (only then: the keys of every earlier cache entry stay valid)
    core_key = cache.key("val", f"{fmt}_core", **base_parts)
    comp_parts: dict[str, dict | None] = {
        "MR": None,                                                       # needs the half-life (resolved late)
        "MA": dict(space=vcache.space_digest(space_a)) if use_a else None,
        "MX": dict(prior=vcache.digest(xprior)) if xprior is not None else None,
        "MS": dict(space=vcache.space_digest(space_s), info=vcache.digest(syn_info)) if use_s else None,
        "MC": dict(space=vcache.space_digest(space_c)) if use_c else None,
        "MM": dict(space=vcache.space_digest(space_m)) if use_m else None,
        "MK": dict(space=vcache.space_digest(space_k)) if use_k else None,
        "MO": dict(modes=list(opp_modes)) if opp_modes else None,
        "MP": dict(grid=[float(x) for x in opponent_grid()]) if playdraw else None,
        "MF": dict(flags=list(design.OPP_FLAG_NAMES)) if flags_test and design.OPP_FLAG_MARK not in games.columns else None,
    }
    if only is not None:
        comp_parts = {c_: (v_ if c_ in only else None) for c_, v_ in comp_parts.items()}

    def comp_key(code: str) -> str:
        return cache.key("val", f"{fmt}_{code}", core=core_key, **comp_parts[code])

    core_hit = cache.load("val", f"{fmt}_core", core_key)
    hits: dict[str, tuple[dict, dict] | None] = {}
    for code, parts in comp_parts.items():
        if parts is not None:
            hits[code] = cache.load("val", f"{fmt}_{code}", comp_key(code))

    # ---- what has to be computed
    need_core = core_hit is None
    todo = {code for code, h in hits.items() if h is None}               # MR joins below once its half-life is known
    slot_of = {"M": 0, **{code: 1 + j for j, (code, _k) in enumerate(COMPONENTS)}}
    mux = parallel.ProgressMux(progress, 0.0, 0.7, len(slot_of))
    for code in slot_of:                                                 # nothing to do for a slot: it is complete
        if code == "M":
            if not (need_core or todo):
                mux.done(0)
        elif code != "MR" and code not in todo:
            mux.done(slot_of[code])
    vt = f"{tmpdir}/val_{fmt}"
    mr_h = {"v": None}                                                   # the half-life of the MR test, once resolved

    # estimated footprint (main process) of a fit over the games of all the stages of this format's chain - see memory.py
    chain_games = sum(n_games[n] for n in prior_names) + g_tr.height

    def prep_fn() -> dict:
        """Stages of the plain chain and the hold-out design (shared by M, MR, MX and every variant's scoring)."""
        with runner.slot(memory.estimate_chain_bytes(chain_games)):
            stages = chain_stages(fmt, games, g_tr, t_cut, space)
            stT = stages[fmt]
            prob_ho, urows, inv = design.make_problem(g_ho, stT.enc, space, clamp_time=True)
            groups = np.unique(g_ho["draft_id"].to_numpy(), return_inverse=True)[1]
            return dict(stages=stages, stT=stT, prob_ho=prob_ho, urows=urows, inv=inv, y=prob_ho.y, groups=groups)

    def chain(code, stages, n_feat, label, d, own_slot=True, **kw):
        os.makedirs(d, exist_ok=True)
        p = mux.slot(slot_of[code])
        with (runner.slot(memory.estimate_chain_bytes(chain_games, pairs=code in ("MS", "MC"))) if own_slot else parallel_null):
            return cardvalue.run_chain(stages, space.n_card, n_feat, d, n_jobs, seed=seed + 100, cancel=cancel,
                                       progress=lambda f, m: p(f, m + label), prior_ok=set(stages) - {fmt}, opts=opts, **kw)

    fut = {"prep": None, "M": None}                                      # futures shared by the component tasks

    def ensure_plain() -> None:
        if fut["prep"] is None:
            fut["prep"] = runner.spawn(prep_fn)
            fut["M"] = runner.spawn(lambda: chain("M", runner.result(fut["prep"])["stages"], space.n_feat, "", vt))

    comp_futs: dict[str, object] = {}

    def start_component(code: str) -> None:
        ensure_plain()
        h = mr_h["v"]

        def run():
            P = runner.result(fut["prep"])
            if code == "MR":
                st = weight_stages(P["stages"], t_cut, {fmt: h} if recency_scope == "format" else h)
                fits = chain(code, st, space.n_feat, " (recency-weighted)", f"{tmpdir}/val_{fmt}_recency")
                return dict(fits=fits, pred=expit(P["prob_ho"].eta(fits[fmt].coef)))
            if code == "MX":
                fits = chain(code, P["stages"], space.n_feat, " (cross-set prior)", f"{tmpdir}/val_{fmt}_xset", theta_prior=xprior)
                return dict(fits=fits, pred=expit(P["prob_ho"].eta(fits[fmt].coef)))
            if code == "MO":
                fits_M = runner.result(fut["M"])
                mux.slot(slot_of[code])(0.0, "Opponent interactions")
                d_opp = f"{tmpdir}/val_{fmt}_opp"
                os.makedirs(d_opp, exist_ok=True)
                with runner.slot(memory.estimate_opp_bytes(chain_games)):
                    o = _opponent_holdout(fmt, P["stages"], fits_M, space, P["prob_ho"], g_ho, P["inv"], d_opp,
                                          n_jobs, seed, cancel, opp_modes)
                mux.done(slot_of[code])
                return dict(opp_out=o, pred=o.pop("pred"))
            if code == "MP":
                from sealed_lab.model import playdraw as pdm
                fits_M = runner.result(fut["M"])
                mux.slot(slot_of[code])(0.0, "Play/draw interactions")
                d_pd = f"{tmpdir}/val_{fmt}_pd"
                os.makedirs(d_pd, exist_ok=True)
                with runner.slot(memory.estimate_opp_bytes(chain_games)):
                    o = pdm.holdout(fmt, P["stages"], fits_M, space.n_card, P["prob_ho"], g_ho, d_pd, n_jobs, seed, cancel)
                mux.done(slot_of[code])
                return dict(pd_out=o, pred=o.pop("pred"))
            if code == "MF":
                gf_tr, gf_ho, gf = design.with_opp_flags(g_tr), design.with_opp_flags(g_ho), design.with_opp_flags(games)
                with runner.slot(memory.estimate_chain_bytes(chain_games)):
                    st = chain_stages(fmt, gf, gf_tr, t_cut, space)
                    fits = chain(code, st, space.n_feat, " (opponent colour flags)", f"{tmpdir}/val_{fmt}_oppflags", own_slot=False)
                    pho, _, _ = design.make_problem(gf_ho, st[fmt].enc, space, clamp_time=True)
                return dict(fits=fits, stages={fmt: st[fmt]}, pred=expit(pho.eta(fits[fmt].coef)))
            # variants fitted on their own deck space (the reservation covers building their stages too)
            sp_ = dict(MA=space_a, MS=space_s, MC=space_c, MM=space_m, MK=space_k)[code]
            kw: dict = {}
            if code == "MA":
                kw = dict(attr_load=space_a.attr_load)
            elif code in ("MS", "MC"):
                kw = dict(extras=[dict(name="pairs", n=space_s.n_pair)] if code == "MS" else [dict(name="arch", n=space_c.n_archc)],
                          ref_fits=runner.result(fut["M"]))
            lab = dict(MA=" (card-attribute prior)", MS=" (pair synergies)", MC=" (archetype-conditional values)",
                       MM=" (calibrated mana simulator)", MK=" (castability)")[code]
            sub = dict(MA="attr", MS="pairs", MC="archc", MM="manasim", MK="cast")[code]
            with runner.slot(memory.estimate_chain_bytes(chain_games, pairs=code in ("MS", "MC"))):
                st = chain_stages(fmt, games, g_tr, t_cut, sp_)
                fits = chain(code, st, sp_.n_feat if code == "MK" else space.n_feat, lab, f"{tmpdir}/val_{fmt}_{sub}", own_slot=False,
                             **kw)
                pho, _, _ = design.make_problem(g_ho, st[fmt].enc, sp_, clamp_time=True)
            # (only the target stage is kept for the summaries: the draft / Bo1 stages are the big ones and are not needed again)
            return dict(fits=fits, stages={fmt: st[fmt]}, pred=expit(pho.eta(fits[fmt].coef)))

        comp_futs[code] = runner.spawn(run)

    if need_core or todo:
        ensure_plain()
        for code, _k in COMPONENTS:
            if code in todo:
                start_component(code)
    # the half-life of the recency test is known only after the inner split: resolve it now (the others are already running)
    h_val = recency_h() if callable(recency_h) else recency_h
    if h_val:
        comp_parts["MR"] = dict(h=float(h_val), scope=recency_scope)
        hits["MR"] = cache.load("val", f"{fmt}_MR", comp_key("MR"))
        if hits["MR"] is None:
            todo.add("MR")
            mr_h["v"] = float(h_val)
            start_component("MR")
    if "MR" not in todo:
        mux.done(slot_of["MR"])

    # ---- collect
    results: dict[str, dict] = {}
    fits = None
    P = None
    if fut["M"] is not None:
        P = runner.result(fut["prep"])
        fits = runner.result(fut["M"])
        mux.done(0)
    for code, f_ in comp_futs.items():
        results[code] = runner.result(f_)
        mux.done(slot_of[code])
    _check(cancel)

    pred_new: dict[str, np.ndarray] = {}
    if fits is not None:
        fT, stT = fits[fmt], P["stages"][fmt]
        prob_ho, y, groups = P["prob_ho"], P["y"], P["groups"]
        pM = expit(prob_ho.eta(fT.coef))
        for code, r in results.items():
            pred_new[code] = r["pred"]
        is_first = half_masks(g_ho)

    comp_payload: dict[str, dict] = {}           # code -> dict(met=..., out=...) as stored in the cache
    core_payload = None
    core_arrays: dict = {}
    base_preds: dict[str, np.ndarray] = {}
    if need_core:
        prog_ = progress or (lambda f, m: None)
        prog_(0.7, "Baselines")
        ytr = stT.prob.y
        sigC = stT.enc.sigma
        Ctr, Cho = stT.prob.C, prob_ho.C
        b0 = _fit_cov_model(Ctr, ytr, sigC)
        base_preds["B0"] = expit(Cho @ b0)
        # ---- B1: + shrunk mean GIH WR of the deck's spells (from training games only)
        p0 = float(ytr.mean())
        gih = gih_table(hands, g_tr.select("game_key", "won"), cards.height, p0)
        nb_is_land = cards.sort("card_id")["is_land"].to_numpy()[space.nb_ids]
        f_deck = deck_gih_feature(space, gih, p0, nb_is_land)
        f_tr = f_deck[stT.deck_rows][stT.prob.idx]
        f_ho = f_deck[P["urows"]][P["inv"]]
        C1tr = sp.hstack([Ctr, sp.csr_matrix(f_tr[:, None])], format="csr")
        C1ho = sp.hstack([Cho, sp.csr_matrix(f_ho[:, None])], format="csr")
        sig1 = np.concatenate([sigC, [10.0]])
        b1 = _fit_cov_model(C1tr, ytr, sig1)
        base_preds["B1"] = expit(C1ho @ b1)
        core_extra: dict = {}
        # ---- B1d: the same feature from the DRAFT games before the cut (a third-party-style prior that never sees the target
        #      format's games), so that "the model beats B1" cannot be explained by B1 simply having less data than the model
        try:
            dg = games.filter(pl.col("fmt") == "draft")
            if dg.height >= MIN_PRIOR_GAMES:
                dev = design.event_starts(dg)
                pre = dev.filter(pl.col("start") < t_cut)["draft_id"].to_list()
                dtr = dg.filter(pl.col("draft_id").is_in(pre))
                if dtr.height >= MIN_PRIOR_GAMES:
                    p0d = float(dtr["won"].cast(pl.Float64).mean())
                    gih_d = gih_table(hands, dtr.select("game_key", "won"), cards.height, p0d)
                    fd_deck = deck_gih_feature(space, gih_d, p0d, nb_is_land)
                    C1dtr = sp.hstack([Ctr, sp.csr_matrix(fd_deck[stT.deck_rows][stT.prob.idx][:, None])], format="csr")
                    C1dho = sp.hstack([Cho, sp.csr_matrix(fd_deck[P["urows"]][P["inv"]][:, None])], format="csr")
                    b1d = _fit_cov_model(C1dtr, ytr, sig1)
                    base_preds["B1d"] = expit(C1dho @ b1d)
                    core_extra["b1d_draft_games"] = int(dtr.height)
        except Cancelled:
            raise
        except Exception:  # noqa: BLE001 - an extra baseline must never sink the validation
            log.exception("baseline B1d skipped")
        # ---- B2: + colour pair of the deck
        P_tr = _pair_onehot(g_tr["main_colors"].fill_null("").to_numpy())
        P_ho = _pair_onehot(g_ho["main_colors"].fill_null("").to_numpy())
        C2tr = sp.hstack([C1tr, P_tr], format="csr")
        C2ho = sp.hstack([C1ho, P_ho], format="csr")
        sig2 = np.concatenate([sig1, np.full(P_tr.shape[1], 0.5)])
        b2 = _fit_cov_model(C2tr, ytr, sig2)
        base_preds["B2"] = expit(C2ho @ b2)
        # ---- P: covariates + scalar * (previous stage deck score)
        prev_name = prior_names[-1] if prior_names else None
        if prev_name is not None and prev_name in fits:
            b_prev = fits[prev_name].deck
            s_tr = (stT.prob.D @ b_prev)[stT.prob.idx]
            s_ho = (prob_ho.D @ b_prev)[prob_ho.idx]
            CPtr = sp.hstack([Ctr, sp.csr_matrix(s_tr[:, None])], format="csr")
            CPho = sp.hstack([Cho, sp.csr_matrix(s_ho[:, None])], format="csr")
            bP = _fit_cov_model(CPtr, ytr, sig1)
            base_preds["P"] = expit(CPho @ bP)
            core_extra["prior_scale"] = float(bP[-1])
            core_extra["prior_stage"] = prev_name
        _check(cancel)

    # ---- held-out metrics: only the models computed in this call (each model's entry depends on M and itself alone)
    met_new: dict = {}
    if fits is not None:
        (progress or (lambda f, m: None))(0.75, "Held-out metrics")
        preds_all: dict[str, np.ndarray] = {"M": pM}
        for code, _k in COMPONENTS:
            if code in pred_new:
                preds_all[code] = pred_new[code]
        preds_all.update(base_preds)
        met_new = cluster_metrics(y, preds_all, groups, n_rep, seed + 5)

    # ---- component entries
    for code, key in COMPONENTS:
        if code not in results:
            continue
        r = results[code]
        entry: dict = {}
        if code == "MR":
            d = met_new["MR"]["vs_M"]["logloss"]     # MR minus M: positive = unweighted better; flip to "recency gain"
            a = met_new["MR"]["vs_M"]["auc"]
            entry = dict(half_life=float(mr_h["v"]), gain=-d["diff"], lo=-d["hi"], hi=-d["lo"], auc_gain=-a["diff"],
                         auc_lo=-a["hi"], auc_hi=-a["lo"], sigma=float(r["fits"][fmt].sigma))
        elif code == "MP":
            from sealed_lab.model import playdraw as pdm
            o = pdm.finalize_summary(r["pd_out"], cards.sort("card_id")["name"].to_numpy()[space.nb_ids].tolist())
            d = met_new["MP"]["vs_M"]["logloss"]
            a = met_new["MP"]["vs_M"]["auc"]
            entry = dict(gain=-d["diff"], lo=-d["hi"], hi=-d["lo"], auc_gain=-a["diff"], auc_lo=-a["hi"], auc_hi=-a["lo"],
                         sigma_base=0.0, **o)
        elif code == "MO":
            o = finalize_opponent_summary(r["opp_out"], cards.sort("card_id")["name"].to_numpy()[space.nb_ids].tolist())
            d = met_new["MO"]["vs_M"]["logloss"]
            a = met_new["MO"]["vs_M"]["auc"]
            entry = dict(gain=-d["diff"], lo=-d["hi"], hi=-d["lo"], auc_gain=-a["diff"], auc_lo=-a["hi"], auc_hi=-a["lo"],
                         sigma=o["sigma"], sigma_base=0.0, **{k: v for k, v in o.items() if k != "sigma"})
        else:
            fts = r["fits"]
            d = met_new[code]["vs_M"]["logloss"]      # positive = M (without) better; flip to the component's gain
            a = met_new[code]["vs_M"]["auc"]
            entry = dict(gain=-d["diff"], lo=-d["hi"], hi=-d["lo"], auc_gain=-a["diff"], auc_lo=-a["hi"], auc_hi=-a["lo"],
                         sigma=float(fts[fmt].sigma), sigma_base=float(fT.sigma))
            if fts[fmt].extra_cv:
                ecv = fts[fmt].extra_cv[0]
                entry.update(extra_sigma=float(ecv["sigma"]), extra_grid=ecv["grid"], extra_loss=ecv["loss"],
                             extra_edge=bool(ecv.get("best", -1) == 0), n_columns=int(ecv["n"]),
                             extra_sigma_by_stage={n: float(f_.extra_cv[0]["sigma"]) for n, f_ in fts.items() if f_.extra_cv})
            if code == "MF":
                nm_ = r["stages"][fmt].enc.names
                kk_ = len(nm_)
                entry["coef"] = {n_: float(v_) for n_, v_ in zip(nm_, fts[fmt].coef[:kk_]) if n_ in design.OPP_FLAG_NAMES or n_.startswith("opp:")}
            if code == "MM":
                names_ = structure_names()
                entry["theta_v1"] = {nm: float(v) for nm, v in zip(names_, fT.theta)}
                entry["theta_v2"] = {nm: float(v) for nm, v in zip(names_, fts[fmt].theta)}
                entry["sd_v1"] = {nm: float(v) for nm, v in zip(names_, space.sd)}
                entry["sd_v2"] = {nm: float(v) for nm, v in zip(names_, space_m.sd)}
                entry["logloss_v1"] = float(met_new["M"]["logloss"])
                entry["logloss_v2"] = float(met_new["MM"]["logloss"])
            elif code == "MK":
                entry.update(castability_summary(fT, fts[fmt], stT, r["stages"][fmt], space, space_k, cards))
            elif code == "MS":
                entry.update(pair_effects_validation(fts[fmt], space_s, r["stages"][fmt], syn_info))
            elif code == "MC":
                entry["n_arch_columns"] = int(space_c.n_archc)
            elif code == "MA":
                entry.update(attribute_summary(fts[fmt], fT, stT, space_a, cards))
            elif code == "MX":
                entry["theta_shift"] = {nm: float(v) for nm, v in zip(structure_names(), fts[fmt].theta - fT.theta)}
        entry["halves"] = half_gains(y, pM, pred_new[code], groups, is_first, n_rep, seed + 7)
        payload = vcache.normalise(dict(met=met_new[code], out=entry))
        comp_payload[code] = payload
        cache.save("val", f"{fmt}_{code}", comp_key(code), payload)

    # ---- core entry
    if need_core:
        out_core = dict(core_extra)
        out_core["n_rep"] = n_rep
        out_core["calibration"] = calibration(y, pM, groups, seed=seed)
        out_core["calibration_B1"] = calibration(y, base_preds["B1"], groups, n_rep=60, seed=seed)
        (progress or (lambda f, m: None))(0.9, "Variance budget")
        # variance budget on held-out builds
        bk = g_ho["build_index"].fill_null(0).to_numpy()
        bkey = np.unique(np.char.add(g_ho["draft_id"].to_numpy().astype(str), np.char.add("#", bk.astype(str))), return_inverse=True)[1]
        build_group = np.zeros(bkey.max() + 1, dtype=np.int64)
        build_group[bkey] = groups
        out_core["variance_budget"] = variance_budget_ci(y, base_preds["B0"], pM, bkey, build_group, int(groups.max()) + 1,
                                                         min(n_rep, 300), seed)
        out_core["base_rate_train"] = p0
        out_core["base_rate_holdout"] = float(y.mean())
        out_core["stages"] = {n: stage_summary(f, P["stages"][n]) for n, f in fits.items()}
        out_core["seconds"] = round(time.time() - t_start, 1)
        ref_info = {n: RefFit.from_fit(f).to_dict() for n, f in fits.items()}
        mcore = {m: met_new[m] for m in ("M", "B0", "B1", "B1d", "B2", "P") if m in met_new}
        core_payload = vcache.normalise(dict(out=out_core, met=mcore, ref=ref_info))
        core_arrays = dict(y=np.asarray(y), groups=np.asarray(groups), pM=np.asarray(pM), pB0=np.asarray(base_preds["B0"]))
        cache.save("val", f"{fmt}_core", core_key, core_payload, core_arrays)
    else:
        core_payload, core_arrays = core_hit

    # ---- assemble (the same dict whether the numbers were computed now or read from the cache)
    out.update({k: v for k, v in core_payload["out"].items()})
    metrics: dict = {}
    comp_final: dict[str, dict] = {}
    parts_info: dict[str, dict] = {"core": dict(hit=core_hit is not None, created=core_payload.get("_created") if core_hit else None)}
    for code, key in COMPONENTS:
        pl_ = comp_payload.get(code)
        if pl_ is not None:
            parts_info[code] = dict(hit=False, created=None)
        elif hits.get(code) is not None:
            pl_ = hits[code][0]
            parts_info[code] = dict(hit=True, created=pl_.get("_created"))
        if pl_ is not None:
            comp_final[code] = pl_
    met_core = core_payload["met"]
    for m in ("M",):
        metrics[m] = met_core[m]
    for code, _k in COMPONENTS:
        if code in comp_final:
            metrics[code] = comp_final[code]["met"]
    for m in ("B0", "B1", "B1d", "B2", "P"):
        if m in met_core:
            metrics[m] = met_core[m]
    out["metrics"] = metrics
    for code, key in COMPONENTS:
        if code in comp_final:
            out[key] = comp_final[code]["out"]
    out["cache"] = dict(parts=parts_info, any_hit=any(p["hit"] for p in parts_info.values()),
                        dates=sorted(p["created"] for p in parts_info.values() if p["hit"] and p["created"]))
    if keep is not None:
        keep[fmt] = dict(key=core_key, y=np.asarray(core_arrays["y"]), groups=np.asarray(core_arrays["groups"]),
                         pM=np.asarray(core_arrays["pM"]), pB0=np.asarray(core_arrays["pB0"]),
                         ref_fits={n: RefFit.from_dict(d) for n, d in core_payload["ref"].items()} if fits is None
                         else dict(fits))
    (progress or (lambda f, m: None))(1.0, f"Validation done ({fmt})")
    return out


def castability_summary(f0: cardvalue.StageFit, fk: cardvalue.StageFit, st0: StageData, stk: StageData, space: DeckSpace,
                        space_k: DeckSpace, cards: pl.DataFrame, n_top: int = 14) -> dict:
    """What the castability model learned (validation fits): the dead-cards coefficient and how card values moved for the cards
    that are most often uncastable in the decks that play them."""
    nc = space.n_card
    th = fk.theta
    sd_dead = float(space_k.sd[-1])
    # coefficient of one dead card (logit) and its SE
    se_th = fk.se_total[fk.k + nc:fk.k + nc + space_k.n_feat]
    per_card = float(th[-1]) / sd_dead
    per_card_se = float(se_th[-1]) / sd_dead
    # mean castability of each card in the games that play it (copies x games weighted)
    gk = np.bincount(stk.prob.idx, minlength=stk.prob.D.shape[0]).astype(float)       # games per (deck, regime) row of MK's problem
    g0 = np.bincount(st0.prob.idx, minlength=st0.prob.D.shape[0]).astype(float)       # games per deck row of M's problem
    num = np.asarray(stk.prob.D[:, :nc].T @ gk).ravel()
    den = np.asarray(st0.prob.D[:, :nc].T @ g0).ravel()
    with np.errstate(all="ignore"):
        mean_p = np.where(den > 0, num / den, np.nan)
    expo = den
    b0, bk = f0.beta_total, fk.beta_total
    names = cards.sort("card_id")["name"].to_numpy()[space.nb_ids]
    lg = np.log(0.55 / 0.45)
    pp = lambda b: 100 * (1 / (1 + np.exp(-(lg + b))) - 0.55)  # noqa: E731
    ok = np.nonzero((expo >= 2000) & np.isfinite(mean_p))[0]
    order = ok[np.argsort(mean_p[ok])][:n_top]
    rows = [dict(name=str(names[i]), mean_p=float(mean_p[i]), beta_m=float(b0[i]), beta_mk=float(bk[i]), pp_m=float(pp(b0[i])),
                 pp_mk=float(pp(bk[i])), exposure=float(expo[i])) for i in order]
    sel = ok
    corr = float(np.corrcoef(b0[sel], bk[sel])[0, 1]) if len(sel) > 3 else None
    return dict(dead_coef=per_card, dead_coef_se=per_card_se, dead_coef_pp=float(100 * 0.55 * 0.45 * per_card), sd_dead=sd_dead,
                mean_dead=float(space_k.mu[-1]), splash_cards=rows, beta_corr=corr,
                mean_p_all=float(np.nansum(num) / max(np.nansum(den), 1e-9)))


def pair_effects_validation(f: cardvalue.StageFit, space_s: DeckSpace, st: StageData, info: dict | None) -> dict:
    """Summary of the pair coefficients of a (validation) fit: counts of reliable pairs, top rows, spread."""
    from sealed_lab.model import synergy

    coef, se = f.extra[:space_s.n_pair], f.extra_se[:space_s.n_pair]
    info = info or {}
    names = info.get("names")
    if names is None:
        return dict(pairs=dict(n_pairs=int(len(coef))))
    return dict(pairs=synergy.pair_summary(space_s.pair_ab, coef, se, info["games"], names, info.get("lift"), space_s.col_of_card),
                spearman_lift=synergy.lift_model_agreement(space_s.pair_ab, coef, info["lift"], space_s.col_of_card)
                if info.get("lift") is not None else None)


def validate_deployed(fmt: str, games: pl.DataFrame, space: DeckSpace, fspace: DeckSpace, cards: pl.DataFrame, tmpdir: str,
                      n_jobs: int | None, components: list[str], seed: int = 0, n_rep: int = N_REP,
                      cancel: threading.Event | None = None, progress: Progress | None = None,
                      recency_h: "float | dict | None" = None, attr_load: np.ndarray | None = None, theta_prior: dict | None = None,
                      extras: list | None = None, opp_mode: str | None = None, m_ref: dict | None = None,
                      cache: "vcache.ValCache | None" = None, opts: "cardvalue.ChainOpts | None" = None) -> dict:
    """The DEPLOYED combination of components (each tested alone in run_validation) refitted on the same training part
    and scored on the same hold-out as the plain chain "M" (same split and seeds as run_validation, so "M" here is the
    same model). Returns the paired log-loss gain of the combination ("MD") over M with a 95% cluster-bootstrap CI, its
    gain over B0 and its calibration.

    `m_ref` (what `run_validation` leaves in its `keep` dict: hold-out predictions of the plain chain and its card sigmas)
    spares refitting the plain chain, which is the very model run_validation already fitted (same data, same seeds); without
    it the plain chain is fitted again here.  With a `cache` the result is stored under a key that also covers the plain
    chain's key, the deployed deck space and every prior array."""
    from sealed_lab.model import vcache

    opts = opts or cardvalue.ChainOpts()

    def prog(f, m):
        if progress:
            progress(f, m)

    gT = games.filter(pl.col("fmt") == fmt)
    split = _holdout_split(gT)
    if split is None:
        return dict(status="skipped", reason="too little data for a hold-out", components=components)
    hold_ids, t_cut, _n_ev, _n_hold = split
    g_tr = gT.filter(~pl.col("draft_id").is_in(hold_ids))
    g_ho = gT.filter(pl.col("draft_id").is_in(hold_ids))
    key = None
    if cache is not None and cache.enabled and m_ref is not None and m_ref.get("key"):
        key = cache.key("val", f"{fmt}_deployed", core=m_ref["key"], components=list(components), fspace=vcache.space_digest(fspace),
                        attr_load=attr_load, theta_prior=theta_prior, extras=extras, recency_h=recency_h, opp_mode=opp_mode,
                        opts=opts.key(), **({"oppflags_base": True} if design.OPP_FLAG_MARK in games.columns else {}))
        hit = cache.load("val", f"{fmt}_deployed", key)
        if hit is not None:
            res = hit[0]
            res["cache"] = dict(hit=True, created=res.pop("_created", None))
            prog(1.0, f"Deployed model checked ({fmt}) (cached)")
            return res
    vd = os.path.join(tmpdir, f"val_{fmt}_deployed")
    for sub in ("plain", "deployed"):
        os.makedirs(os.path.join(vd, sub), exist_ok=True)
    if m_ref is None:
        stages0 = chain_stages(fmt, games.drop(design.OPP_FLAG_MARK, strict=False), g_tr.drop(design.OPP_FLAG_MARK, strict=False), t_cut, space)
        fits0 = cardvalue.run_chain(stages0, space.n_card, space.n_feat, os.path.join(vd, "plain"), n_jobs, seed=seed + 100,
                                    cancel=cancel, progress=lambda f, m: prog(0.45 * f, m + " (plain chain)"),
                                    prior_ok=set(stages0) - {fmt}, opts=opts)
        _check(cancel)
        prob0, _, _ = design.make_problem(g_ho.drop(design.OPP_FLAG_MARK, strict=False), stages0[fmt].enc, space, clamp_time=True)
        st0 = stages0[fmt]
        y = prob0.y
        groups = np.unique(g_ho["draft_id"].to_numpy(), return_inverse=True)[1]
        pM = expit(prob0.eta(fits0[fmt].coef))
        pB0 = expit(prob0.C @ _fit_cov_model(st0.prob.C, st0.prob.y, st0.enc.sigma))
        ref_fits = fits0
        lo_d = 0.45
    else:
        y, groups, pM, pB0, ref_fits = m_ref["y"], m_ref["groups"], m_ref["pM"], m_ref["pB0"], m_ref["ref_fits"]
        lo_d = 0.0
    stagesD = chain_stages(fmt, games, g_tr, t_cut, fspace)
    if recency_h:
        stagesD = weight_stages(stagesD, t_cut, recency_h)
    fitsD = cardvalue.run_chain(stagesD, space.n_card, fspace.n_feat, os.path.join(vd, "deployed"), n_jobs, seed=seed + 100,
                                cancel=cancel, progress=lambda f, m: prog(lo_d + (0.9 - lo_d) * f, m + " (deployed model)"),
                                prior_ok=set(stagesD) - {fmt}, attr_load=attr_load, theta_prior=theta_prior,
                                extras=extras or None, ref_fits=ref_fits if extras else None, opts=opts)
    _check(cancel)
    probD, _, invD = design.make_problem(g_ho, stagesD[fmt].enc, fspace, clamp_time=True)
    preds = {"M": pM, "MD": expit(probD.eta(fitsD[fmt].coef))}
    if opp_mode:
        prog(0.9, "Deployed model: opponent interactions")
        opp = _opponent_holdout(fmt, stagesD, fitsD, fspace, probD, g_ho, invD, vd, n_jobs, seed, cancel, (opp_mode,))
        preds["MD"] = opp["pred"]
    preds["B0"] = pB0
    met = cluster_metrics(y, preds, groups, n_rep, seed + 5)
    d = met["MD"]["vs_M"]["logloss"]                     # MD minus M: positive = plain chain better
    b0 = met["B0"]["vs_M"]["logloss"]                    # B0 minus M
    # gain of MD over B0 = (B0 - M) - (MD - M); its CI from the paired replicates is not stored, so report both pieces
    cal = calibration(y, preds["MD"], groups, n_rep=150, seed=seed)
    prog(1.0, f"Deployed model checked ({fmt})")
    res = dict(status="ok", components=components, n_holdout_games=int(len(y)), n_holdout_events=int(groups.max() + 1),
               gain=-d["diff"], lo=-d["hi"], hi=-d["lo"], logloss_M=met["M"]["logloss"], logloss_MD=met["MD"]["logloss"],
               logloss_B0=met["B0"]["logloss"], gain_M_vs_B0=b0["diff"], auc_M=met["M"]["auc"], auc_MD=met["MD"]["auc"],
               calibration_slope=cal["slope"], calibration_slope_ci=cal["slope_ci"], calibration_intercept=cal["intercept"])
    if key is not None:
        res = vcache.normalise(res)
        cache.save("val", f"{fmt}_deployed", key, res)
        res["cache"] = dict(hit=False, created=None)
    return res


def _opponent_holdout(fmt: str, stages: dict, fits: dict, space: DeckSpace, prob_ho, g_ho: pl.DataFrame, inv: np.ndarray, tmp: str,
                      n_jobs, seed: int, cancel, modes: tuple[str, ...]) -> dict:
    """Card x opponent-colour interactions fitted on the pooled training games (offset = the chain's own fitted predictor),
    scored on the hold-out games of `fmt`. The opponent encoding is chosen by CV loss on the training part."""
    from sealed_lab.model import opponent as om

    nc = space.n_card
    coefs = {n: f.coef for n, f in fits.items()}
    best_mode, best_pool, best_sigma, cvs = None, None, None, {}
    for mode in modes:
        _check(cancel)
        pool = om.pool_stages(stages, coefs, nc, mode)
        sg, info = om.cv_sigma(pool, tmp, n_jobs, seed=seed + 41, cancel=cancel)
        cvs[mode] = info
        score = info["loss"][info["best"]]
        if best_mode is None or score < cvs[best_mode]["loss"][cvs[best_mode]["best"]]:
            best_mode, best_pool, best_sigma = mode, pool, sg
    pool = best_pool
    delta = om.fit_deltas(pool, best_sigma)
    Dc = prob_ho.D[:, :nc].tocsr()[prob_ho.idx]
    eta = prob_ho.eta(fits[fmt].coef) + om.predict_centred(Dc, om.opp_matrix(g_ho["opp_colors"].to_numpy(), best_mode), delta, pool.mass)
    # reliability of the centred effects vs the 10 colour pairs (cluster-robust SE of the training fit)
    opps = om.default_opponents()
    ncols = pool.prob.ncols
    A = om.opp_contrasts(pool.mass, best_mode, opps, ncols)
    _, _, c_lap, c_rob = glm.cluster_se(pool.prob, delta, pool.sigma_vector(best_sigma), pool.group, contrasts=A)
    cen = om.centred(delta[:pool.n_int], pool.mass, best_mode, opps)         # (n_card, n_opp)
    se_c = c_rob.reshape(nc, len(opps))
    expo = np.asarray(pool.prob.C[:, :pool.n_int].sum(0)).ravel().reshape(nc, -1).sum(1)    # copies x games per card
    return dict(pred=expit(eta), sigma=float(best_sigma), mode=best_mode, cv=cvs, n_pool_games=int(pool.n_games),
                n_pool_events=int(pool.n_groups), opps=opps, _cen=cen, _se=se_c, _expo=expo, _mass=pool.mass)


def finalize_opponent_summary(o: dict, names: list[str]) -> dict:
    """Replace the raw arrays of `_opponent_holdout` by the JSON-able effect table."""
    from sealed_lab.model import opponent as om

    cen, se_c, expo = o.pop("_cen"), o.pop("_se"), o.pop("_expo")
    o.pop("_mass", None)
    o["table"] = om.effect_table(cen, se_c, names, o["opps"], expo)
    return o


def structure_names() -> list[str]:
    from sealed_lab.model import structure

    return list(structure.FEATURE_NAMES)


PP_PER_LOGIT = 100.0 * 0.55 * 0.45      # logit -> percentage points near a 55% deck


def gamma_table(f: cardvalue.StageFit, space: DeckSpace) -> list[dict]:
    """Fitted attribute coefficients: per-unit effect of each attribute (pp near 55%) with SE and |t|, strongest first."""
    from sealed_lab.model import attributes as at

    if f.attr_load is None or not f.n_attr:
        return []
    g = f.gamma
    se = f.se_total[f.k + f.n_card + f.n_feat:]
    sd = space.attr_sd
    rows = []
    for j, nm in enumerate(space.attr_names):
        eff, eff_se = PP_PER_LOGIT * g[j] / sd[j], PP_PER_LOGIT * se[j] / sd[j]
        rows.append(dict(name=nm, label=at.LABELS.get(nm, nm), gamma=float(g[j]), per_unit_pp=float(eff), se_pp=float(eff_se),
                         t=float(g[j] / se[j]) if se[j] > 0 else 0.0, sd_deck=float(sd[j])))
    rows.sort(key=lambda r: -abs(r["t"]))
    return rows


def attribute_summary(fa: cardvalue.StageFit, f0: cardvalue.StageFit, st: StageData, space_a: DeckSpace, cards: pl.DataFrame,
                      n_top: int = 15) -> dict:
    """What the attribute prior did in the target stage (fitted on the training part): gamma table and the cards whose
    total effect moved most compared with the plain chain, with their exposure."""
    c = cards.sort("card_id")
    names = c["name"].to_numpy()[space_a.nb_ids]
    rarity = c["rarity"].to_numpy()[space_a.nb_ids]
    b0, b1 = f0.beta_total, fa.beta_total
    moved = b1 - b0
    expo = st.exposure
    seen = expo > 0
    low = seen & (expo < 2000)
    high = expo >= 2000
    order = [i for i in np.argsort(-np.abs(moved)) if seen[i]][:n_top]
    pp = PP_PER_LOGIT

    def rec(i):
        return dict(name=str(names[i]), rarity=str(rarity[i]), exposure=float(expo[i]), beta_plain_pp=float(pp * b0[i]),
                    beta_attr_model_pp=float(pp * b1[i]), moved_pp=float(pp * moved[i]), attr_part_pp=float(pp * fa.beta_attr[i]),
                    se_plain_pp=float(pp * f0.card_se[i]), se_attr_model_pp=float(pp * fa.card_se[i]))
    ba = fa.beta_attr
    ref = high if high.sum() > 5 else seen
    share = float(np.var(ba[ref]) / np.var(b1[ref])) if np.var(b1[ref]) > 0 else float("nan")
    return dict(gamma=gamma_table(fa, space_a), movers=[rec(i) for i in order], n_attr=int(space_a.n_attr),
                mean_abs_move_pp_low=float(pp * np.abs(moved[low]).mean()) if low.any() else None,
                mean_abs_move_pp_high=float(pp * np.abs(moved[high]).mean()) if high.any() else None,
                n_low=int(low.sum()), n_high=int(high.sum()), attr_share_of_var=share,
                se_ratio_low=float(np.mean(fa.card_se[low] / np.maximum(f0.card_se[low], 1e-12))) if low.any() else None)


def stage_summary(f: cardvalue.StageFit, st: StageData) -> dict:
    return dict(sigma=float(f.sigma), lam=f.lam, lam_se=f.lam_se, lam_from=f.lam_from, n_games=int(st.n_games),
                n_events=int(st.n_groups), cv_grid=f.cv_grid, cv_loss=f.cv_loss, cv_se=f.cv_se, cv_done=f.cv_done,
                cv_scheme=f.cv_scheme, stack_w=float(f.w), stack_grid=list(f.w_grid), stack_loss=list(f.w_loss),
                theta_sd=cardvalue.theta_sd_of(f), theta_cv=f.theta_cv_sd is not None, theta_grid=list(f.theta_grid),
                theta_loss=list(f.theta_loss),
                converged=bool(f.converged), n_iter=int(f.n_iter), warnings=list(f.warnings))


# ---------------------------------------------------------------------------------------------
# simulator validation
# ---------------------------------------------------------------------------------------------

def simulator_validation(games: pl.DataFrame, decks: pl.DataFrame, phi_table: np.ndarray, phi_keys: np.ndarray,
                         cards: pl.DataFrame, hands: pl.LazyFrame, table_names: list[str],
                         lands_bins=(15, 16, 17, 18), v2_tables: dict[str, np.ndarray] | None = None) -> list[dict]:
    """Per (format, land count): observed vs simulated mulligan rate and mean lands in kept 7-card hands.

    `phi_table` is the v1 simulator's table; `v2_tables` ({regime: table}) adds the v2 simulator, each game with the table of
    the regime of its own event."""
    ix = {n: i for i, n in enumerate(table_names)}
    d_lands = decks.select("deck_key", "n_lands")
    g = games.join(d_lands, on="deck_key", how="left").select("game_key", "fmt", "event", "deck_key", "num_mulligans", "n_lands")
    land_ids = cards.filter(pl.col("is_land"))["card_id"].to_list()
    k0 = g.filter(pl.col("num_mulligans") == 0).select("game_key").lazy()
    opening_lands = (hands.filter(pl.col("card_id").is_in(land_ids) & (pl.col("opening") > 0))
                     .join(k0, on="game_key", how="inner").group_by("game_key").agg(pl.col("opening").sum().alias("ol")).collect())
    g = g.join(opening_lands, on="game_key", how="left").with_columns(pl.col("ol").fill_null(0))
    row_of = np.searchsorted(phi_keys, g["deck_key"].to_numpy())
    g = g.with_columns(pl.Series("sim_mull", phi_table[row_of, ix["mull_rate"]]),
                       pl.Series("sim_l7k", phi_table[row_of, ix["lands7_kept"]]))
    if v2_tables:
        reg = np.array([config.event_regime(str(e)) for e in g["event"].to_numpy()], dtype=object)
        m2 = np.full(g.height, np.nan)
        l2 = np.full(g.height, np.nan)
        for r, tab in v2_tables.items():
            sel = reg == r
            m2[sel] = tab[row_of[sel], ix["mull_rate"]]
            l2[sel] = tab[row_of[sel], ix["lands7_kept"]]
        g = g.with_columns(pl.Series("sim_mull2", m2), pl.Series("sim_l7k2", l2))
    rows = []
    for fmt in ("bo3", "bo1", "draft"):
        gf = g.filter(pl.col("fmt") == fmt)
        if gf.is_empty():
            continue
        for nl in lands_bins:
            c = gf.filter(pl.col("n_lands") == nl)
            n = c.height
            if n < 50:
                continue
            mulled = (c["num_mulligans"] > 0).to_numpy()
            lo, hi = wilson(float(mulled.sum()), n)
            kept = c.filter(pl.col("num_mulligans") == 0)
            ol = kept["ol"].to_numpy().astype(float)
            m = float(ol.mean()) if len(ol) else float("nan")
            se = float(ol.std(ddof=1) / math.sqrt(len(ol))) if len(ol) > 1 else float("nan")
            rows.append(dict(fmt=fmt, n_lands=nl, n_games=n, obs_mull=float(mulled.mean()), obs_mull_ci=[lo, hi],
                             sim_mull=float(c["sim_mull"].mean()), obs_lands=m, obs_lands_ci=[m - 1.96 * se, m + 1.96 * se],
                             sim_lands=float(c["sim_l7k"].mean()), n_kept=int(len(ol))))
            if v2_tables:
                rows[-1].update(sim_mull_v2=float(c["sim_mull2"].mean()), sim_lands_v2=float(c["sim_l7k2"].mean()))
    return rows


# ---------------------------------------------------------------------------------------------
# report rendering
# ---------------------------------------------------------------------------------------------
FMT_TITLE = {"bo3": "Bo3 - Traditional Sealed (TradSealed)", "bo1": "Bo1 - Sealed"}

CSS = """
:root{--bg:#ffffff;--fg:#1d2330;--mut:#667085;--line:#e3e7ee;--card:#f6f8fb;--ok:#137a3f;--okbg:#e7f5ec;--warn:#9a5b00;--warnbg:#fff4e0;
--bad:#b42318;--badbg:#fdeceb;--acc:#2f5fd0;--acc2:#e08a00;--pt:#2f5fd0;--bar:#c9d3ea}
@media (prefers-color-scheme: dark){:root{--bg:#14171d;--fg:#e7eaf0;--mut:#9aa4b5;--line:#2b313c;--card:#1b2029;--ok:#59c98a;--okbg:#17301f;
--warn:#f0b255;--warnbg:#33270f;--bad:#ff8a80;--badbg:#3a1a18;--acc:#7fa4ff;--acc2:#f0a93a;--pt:#7fa4ff;--bar:#3a4766}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:1000px;margin:0 auto;padding:28px 20px 80px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:20px;margin:38px 0 10px;padding-top:10px;border-top:1px solid var(--line)}
h3{font-size:16px;margin:22px 0 6px}p{margin:8px 0}.mut{color:var(--mut)}.small{font-size:13px}
table{border-collapse:collapse;width:100%;margin:8px 0 14px;font-size:13.5px}
th,td{padding:5px 9px;white-space:nowrap;border-bottom:1px solid var(--line);text-align:right;vertical-align:top}
th{font-weight:600;color:var(--mut);background:var(--card)}th:first-child,td:first-child{text-align:left}
td.l,th.l{text-align:left}td.w{white-space:normal}tr.hl td{background:var(--card);font-weight:600}
.verdict{border-radius:10px;padding:14px 18px;margin:14px 0;border:1px solid var(--line);background:var(--card)}
.verdict.pass{background:var(--okbg);border-color:var(--ok)}.verdict.part{background:var(--warnbg);border-color:var(--warn)}
.verdict.fail{background:var(--badbg);border-color:var(--bad)}
.verdict h3{margin:0 0 6px;font-size:17px}.badge{display:inline-block;padding:1px 9px;border-radius:99px;font-weight:700;font-size:12px;
letter-spacing:.04em}.pass .badge{background:var(--ok);color:#fff}.part .badge{background:var(--warn);color:#fff}.fail .badge{background:var(--bad);color:#fff}
.skip .badge{background:var(--mut);color:#fff}
ul{margin:6px 0 6px 20px;padding:0}li{margin:3px 0}
.kpi{display:flex;flex-wrap:wrap;gap:10px;margin:10px 0}.kpi div{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px 14px;min-width:150px}
.kpi b{display:block;font-size:20px}.kpi span{color:var(--mut);font-size:12.5px}
.warnbox{background:var(--warnbg);border-left:4px solid var(--warn);padding:8px 14px;margin:10px 0;border-radius:4px;font-size:13.5px}
svg{max-width:100%;height:auto}svg text{fill:var(--fg);font-size:11px}svg .ax{stroke:var(--mut);stroke-width:1}
svg .grid{stroke:var(--line);stroke-width:1}svg .diag{stroke:var(--mut);stroke-dasharray:4 3}
.flag{color:var(--bad);font-weight:700}.low{color:var(--acc2)}
details{margin:8px 0}summary{cursor:pointer;font-weight:600}
code{background:var(--card);padding:1px 5px;border-radius:4px}
.cols{display:flex;flex-wrap:wrap;gap:24px}.cols>div{flex:1 1 440px;min-width:0}
"""


def _e(x) -> str:
    return html.escape(str(x))


def _f(x, d=4, plus=False) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{x:+.{d}f}" if plus else f"{x:.{d}f}"


def _pct(x, d=1) -> str:
    return "n/a" if x is None or not math.isfinite(x) else f"{100 * x:.{d}f}%"


def _ci(lo, hi, d=4, plus=False) -> str:
    return f"[{_f(lo, d, plus)}, {_f(hi, d, plus)}]"


def _table(headers: list[str], rows: list[list[str]], cls: dict[int, str] | None = None, left: set[int] | None = None) -> str:
    left = left or {0}
    h = "".join(f'<th class="{"l" if i in left else ""}">{_e(c)}</th>' for i, c in enumerate(headers))
    b = []
    for r in rows:
        hl = ""
        if r and isinstance(r[0], tuple):
            hl = ' class="hl"'
            r = [r[0][0]] + list(r[1:])
        b.append(f"<tr{hl}>" + "".join(f'<td class="{"l" if i in left else ""}">{c}</td>' for i, c in enumerate(r)) + "</tr>")
    return f"<div style='overflow-x:auto'><table><thead><tr>{h}</tr></thead><tbody>{''.join(b)}</tbody></table></div>"


def svg_calibration(bins: list[dict], w: int = 380, h: int = 380) -> str:
    if not bins:
        return ""
    xs = [b["pred"] for b in bins]
    lo_all = [b["cluster"][0] for b in bins] + xs
    hi_all = [b["cluster"][1] for b in bins] + xs
    lo, hi = max(0.0, min(lo_all) - 0.02), min(1.0, max(hi_all) + 0.02)
    ml, mb, mt, mr = 46, 40, 14, 14
    pw, ph = w - ml - mr, h - mb - mt

    def X(v):
        return ml + (v - lo) / (hi - lo) * pw

    def Y(v):
        return mt + ph - (v - lo) / (hi - lo) * ph

    s = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Calibration plot">']
    ticks = np.linspace(lo, hi, 5)
    for t in ticks:
        s.append(f'<line class="grid" x1="{ml}" x2="{ml + pw}" y1="{Y(t):.1f}" y2="{Y(t):.1f}"/>')
        s.append(f'<text x="{ml - 6}" y="{Y(t) + 4:.1f}" text-anchor="end">{100 * t:.0f}%</text>')
        s.append(f'<text x="{X(t):.1f}" y="{mt + ph + 16}" text-anchor="middle">{100 * t:.0f}%</text>')
    s.append(f'<line class="ax" x1="{ml}" x2="{ml}" y1="{mt}" y2="{mt + ph}"/><line class="ax" x1="{ml}" x2="{ml + pw}" y1="{mt + ph}" y2="{mt + ph}"/>')
    s.append(f'<line class="diag" x1="{X(lo):.1f}" y1="{Y(lo):.1f}" x2="{X(hi):.1f}" y2="{Y(hi):.1f}"/>')
    for b in bins:
        x = X(b["pred"])
        s.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{Y(b["cluster"][0]):.1f}" y2="{Y(b["cluster"][1]):.1f}" stroke="var(--acc)" stroke-width="2" opacity=".55"/>')
        s.append(f'<circle cx="{x:.1f}" cy="{Y(b["obs"]):.1f}" r="4" fill="var(--pt)"/>')
    s.append(f'<text x="{ml + pw / 2}" y="{h - 6}" text-anchor="middle">predicted win probability (decile mean)</text>')
    s.append(f'<text transform="translate(12 {mt + ph / 2}) rotate(-90)" text-anchor="middle">observed win rate</text>')
    s.append("</svg>")
    return "".join(s)


def svg_scatter(x, y, flag, xlabel: str, ylabel: str, slope: float | None, w: int = 420, h: int = 360) -> str:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return ""
    ml, mb, mt, mr = 48, 40, 12, 12
    pw, ph = w - ml - mr, h - mb - mt
    xl, xh = float(np.percentile(x, 1)), float(np.percentile(x, 99))
    yl, yh = float(np.percentile(y, 1)), float(np.percentile(y, 99))
    xl, xh = xl - 0.05 * (xh - xl + 1e-9), xh + 0.05 * (xh - xl + 1e-9)
    yl, yh = yl - 0.05 * (yh - yl + 1e-9), yh + 0.05 * (yh - yl + 1e-9)

    def X(v):
        return ml + (np.clip(v, xl, xh) - xl) / (xh - xl) * pw

    def Y(v):
        return mt + ph - (np.clip(v, yl, yh) - yl) / (yh - yl) * ph

    s = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="beta versus psi">']
    for t in np.linspace(yl, yh, 5):
        s.append(f'<line class="grid" x1="{ml}" x2="{ml + pw}" y1="{Y(t):.1f}" y2="{Y(t):.1f}"/><text x="{ml - 5}" y="{Y(t) + 4:.1f}" text-anchor="end">{t:.2f}</text>')
    for t in np.linspace(xl, xh, 5):
        s.append(f'<text x="{X(t):.1f}" y="{mt + ph + 15}" text-anchor="middle">{t:.2f}</text>')
    s.append(f'<line class="ax" x1="{ml}" x2="{ml}" y1="{mt}" y2="{mt + ph}"/><line class="ax" x1="{ml}" x2="{ml + pw}" y1="{mt + ph}" y2="{mt + ph}"/>')
    if slope is not None:
        s.append(f'<line class="diag" x1="{X(xl):.1f}" y1="{Y(slope * xl):.1f}" x2="{X(xh):.1f}" y2="{Y(slope * xh):.1f}"/>')
    for xi, yi, fl in zip(x, y, flag):
        col = "var(--bad)" if fl else "var(--pt)"
        s.append(f'<circle cx="{X(xi):.1f}" cy="{Y(yi):.1f}" r="{3.6 if fl else 2.6}" fill="{col}" opacity="{.95 if fl else .55}"/>')
    s.append(f'<text x="{ml + pw / 2}" y="{h - 6}" text-anchor="middle">{_e(xlabel)}</text>')
    s.append(f'<text transform="translate(12 {mt + ph / 2}) rotate(-90)" text-anchor="middle">{_e(ylabel)}</text></svg>')
    return "".join(s)


def svg_cv(grid, loss, best_sigma, w: int = 300, h: int = 150) -> str:
    if not loss:
        return ""
    g = np.log(np.asarray(grid))
    l = np.asarray(loss) - min(loss)
    ml, mb, mt, mr = 40, 28, 10, 10
    pw, ph = w - ml - mr, h - mb - mt
    top = max(float(l.max()), 1e-6)

    def X(v):
        return ml + (v - g.min()) / (g.max() - g.min() + 1e-12) * pw

    def Y(v):
        return mt + ph - v / top * ph

    pts = " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in zip(g, l))
    s = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="CV curve">',
         f'<line class="ax" x1="{ml}" x2="{ml}" y1="{mt}" y2="{mt + ph}"/><line class="ax" x1="{ml}" x2="{ml + pw}" y1="{mt + ph}" y2="{mt + ph}"/>',
         f'<polyline fill="none" stroke="var(--acc)" stroke-width="2" points="{pts}"/>']
    for a, b, gs in zip(g, l, grid):
        s.append(f'<circle cx="{X(a):.1f}" cy="{Y(b):.1f}" r="{5 if abs(gs - best_sigma) < 1e-12 else 2.5}" fill="{"var(--acc2)" if abs(gs - best_sigma) < 1e-12 else "var(--pt)"}"/>')
    s.append(f'<text x="{ml}" y="{h - 8}" text-anchor="start">{grid[0]:.3g}</text><text x="{ml + pw}" y="{h - 8}" text-anchor="end">{grid[-1]:.3g}</text>')
    s.append(f'<text x="{ml + pw / 2}" y="{h - 8}" text-anchor="middle">sigma (log)</text>')
    s.append(f'<text x="{ml - 4}" y="{mt + 8}" text-anchor="end">+{top:.4f}</text></svg>')
    return "".join(s)


def verdict_for(v: dict) -> tuple[str, str, list[str]]:
    """(css class, label, bullet lines) for a validated format.

    PASS = the full model's held-out log-loss gain over every baseline has a 95% CI above zero; PARTIAL = significant
    gain over B0 only; INCONCLUSIVE = all point gains >= 0 but the hold-out cannot show it; FAIL = a baseline is at
    least as good (point estimate negative against B1 or B2, or significantly better than the model)."""
    if v.get("status") != "ok":
        return "skip", "NOT VALIDATED", [_e(v.get("reason", "no hold-out possible"))]
    met = v["metrics"]
    lines = []
    sig, point, worse = [], [], False
    for b in ("B0", "B1", "B1d", "B2"):
        if b not in met:
            continue
        d = met[b]["vs_M"]["logloss"]
        if b != "B1d":                      # B1d (draft-based GIH) is shown next to B1, it does not change the verdict
            sig.append(d["lo"] > 0)
            point.append(d["diff"] >= 0)
            worse |= d["hi"] < 0
        verb = "beats" if d["lo"] > 0 else ("is not distinguishable from" if d["hi"] >= 0 else "is WORSE than")
        lines.append(f"Full model {verb} <b>{b}</b> ({MODEL_LABEL[b].split(': ', 1)[1]}): held-out log-loss gain "
                     f"<b>{_f(d['diff'], 4, True)}</b> {_ci(d['lo'], d['hi'], 4, True)}; "
                     f"AUC {_f(met['M']['auc'] - met[b]['auc'], 4, True)}.")
    if v.get("prior_stage") and "P" in met:
        d = met["P"]["vs_M"]["logloss"]
        lines.append(f"Value of the {_e(v['fmt'])} games themselves (full model vs the earlier-stage prior alone): "
                     f"{_f(d['diff'], 4, True)} {_ci(d['lo'], d['hi'], 4, True)} log loss.")
    cal = v["calibration"]
    sl, slc = cal["slope"], cal["slope_ci"]
    if slc[0] > 1.0:
        txt = "under-confident (real effects are larger than predicted)"
    elif slc[1] < 1.0:
        txt = "OVER-confident (predictions are too extreme)"
    else:
        txt = "no evidence of mis-calibration (CI includes 1)"
    lines.append(f"Calibration slope {_f(sl, 2)} {_ci(slc[0], slc[1], 2)}: {txt}. Calibration-in-the-large "
                 f"{_f(cal['intercept'], 3, True)} {_ci(*cal['intercept_ci'], 3, True)} logit (time drift shows up here).")
    if "level_pp" in cal:
        lo_, hi_ = cal["level_pp_ci"]
        sig_ = lo_ > 0 or hi_ < 0
        lines.append(f"Absolute level on the newest events: observed minus predicted game win rate "
                     f"<b>{_f(cal['level_pp'], 1, True)} pp</b> {_ci(lo_, hi_, 1, True)} (slope fixed at 1: {_f(cal['level_logit'], 3, True)} logit). "
                     + (("The absolute win rates the app shows (game / match win rate, expected wins) were "
                         + ("UNDER" if cal["level_pp"] > 0 else "OVER") + "-predicted by about this much; the 80% intervals do not include "
                         "this level error. Rankings, scores and differences between decks are not affected.")
                        if sig_ else "No evidence that the absolute level is off (CI includes 0)."))
    vb = v["variance_budget"]
    if vb["sigma2_deck"] > 0:
        lines.append(f"Between-deck SD of win rate (net of pilot and covariates): <b>{100 * math.sqrt(vb['sigma2_deck']):.1f} pp</b>. "
                     f"Share of that variance the model explains out of sample: <b>{_pct(vb['share_recal'], 0)}</b> "
                     f"{_ci(*vb['ci']['share_recal'], 2)} (raw Var(pred)/sigma^2: {_pct(vb['share_raw'], 0)}).")
    else:
        lines.append("Between-deck variance cannot be estimated here: the spread of build win rates is SMALLER than coin-flip noise "
                     f"({_f(vb['var_obs'], 4)} vs {_f(vb['noise'], 4)}). Games within a run are not independent - Arena pairs players by "
                     "record, so a winning deck meets stronger opponents and its results drift back towards 50%. The method-of-moments "
                     "share is therefore not quoted; held-out log loss and calibration remain valid.")
    n_ho = v["n_holdout_games"]
    if all(sig):
        return "pass", "PASS", lines
    if worse or not point[1]:
        return "fail", "FAIL", lines
    if sig[0]:
        return "part", "PARTIAL", lines
    lines.append(f"The hold-out ({n_ho:,} games, {v['n_holdout_events']} events) is too small to separate the models: "
                 "all point estimates favour the model but none is significant.")
    return "part", "INCONCLUSIVE", lines


def _halves_cells(o: dict) -> tuple[str, str]:
    """(earlier-half cell, later-half cell) of a component entry's `halves` for the report tables."""
    h = o.get("halves")
    if not h:
        return "-", "-"
    return (f"{_f(h['A']['gain'], 4, True)} {_ci(h['A']['lo'], h['A']['hi'], 4, True)}",
            f"{_f(h['B']['gain'], 4, True)}; {_f(h['B']['lo90'], 4, True)}")


def render_confirmation(R: dict) -> str:
    """Report section: the confirmation split of every component decision (earlier half decides, later half confirms)."""
    comps = (("recency", "Recency weighting", "recency"), ("attributes", "Card-attribute prior", "attributes"),
             ("crossset", "Cross-set structure prior", "crossset"), ("synergy", "Card-pair synergies", "synergy"),
             ("archcond", "Archetype-conditional values", "archcond"), ("matchups", "Opponent matchups", "opponent"),
             ("playdraw", "Play/draw card effects", "playdraw"),
             ("manasim", "Calibrated mana simulator", "manasim"), ("castability", "Castability", "castability"))
    rows = []
    off = []
    for key, label, vkey in comps:
        d = R.get(key) or {}
        for f in R["formats"]:
            v = R["validation"].get(f) or {}
            o = v.get(vkey)
            if v.get("status") != "ok" or not o:
                continue
            a, b = _halves_cells(o)
            rows.append([_e(label), _e(f), f"{_f(o['gain'], 4, True)} {_ci(o['lo'], o['hi'], 4, True)}", a, b])
        sub = list((d.get("formats") or {}).values()) if key == "recency" and d.get("formats") else [d]
        for sd in sub:
            c = sd.get("confirmation") or {}
            if c.get("confirm") and c.get("old_rule_applied") and not sd.get("applied") and label not in off:
                off.append(label)
    if not rows:
        return ""
    P = ["<h2>Confirmation split of the component decisions</h2>",
         "<p class='small mut'>Several optional components are tested on the same hold-out, so one of them can pass by chance. The hold-out "
         "events are therefore split in time: a component is applied only if it passes the usual rule (95% CI above zero, gain at least "
         f"{_f(1e-4, 4)}) on the <b>earlier half</b> of the events AND on the <b>later half</b> its gain has the same sign with a one-sided "
         "90% bound above zero. Castability, a safety rule, must not be materially worse on either half. Gains are log loss per game "
         "(positive = the component is better than the plain chain).</p>"]
    P.append(_table(["Component", "Format", "Whole hold-out (95% CI)", "Earlier half (95% CI)", "Later half (gain; 90% lower bound)"], rows, left={0, 1}))
    if off:
        P.append("<p><b>Turned off by the confirmation split</b> (they would have been applied by the whole-hold-out rule): "
                 + _e(", ".join(off)) + ".</p>")
    return "".join(P)


def render_deployed(dp: dict) -> str:
    """Report section: the components that passed their own tests, refitted TOGETHER and scored on the same hold-out."""
    P = ["<h2>The deployed model as a whole</h2>"]
    comps = dp.get("components") or []
    if not comps:
        P.append("<p>No optional component was applied: the deployed model is the plain chain validated above.</p>")
        return "".join(P)
    P.append("<p class='small mut'>Every optional component is tested on its own against the plain chain (sections below). "
             "The ones that passed are deployed together, so the combination is refitted on the same training events and "
             "scored on the same hold-out games. If it were materially worse than the plain chain in any format "
             f"(log-loss loss of at least {_f(1e-4, 4)} per game with a 95% CI below zero), the plain chain would be deployed "
             "instead.</p>")
    rows = []
    for f, o in (dp.get("formats") or {}).items():
        if o.get("status") != "ok":
            rows.append([_e(FMT_TITLE.get(f, f)), _e(o.get("reason", "not run")), "", "", ""])
            continue
        rows.append([_e(FMT_TITLE.get(f, f)), f"{_f(o['gain'], 4, True)} {_ci(o['lo'], o['hi'], 4, True)}",
                     f"{o['logloss_MD']:.4f} vs {o['logloss_M']:.4f}", f"{o['auc_MD']:.4f} vs {o['auc_M']:.4f}",
                     f"{_f(o['calibration_slope'], 2)} {_ci(*o['calibration_slope_ci'], 2)}"])
    P.append(_table(["Format", "Gain over plain chain (95% CI)", "Log loss deployed vs plain", "AUC deployed vs plain",
                     "Calibration slope (deployed)"], rows, left={0}))
    for f, o in (dp.get("formats") or {}).items():
        if (o.get("cache") or {}).get("hit"):
            P.append(_cache_note(o["cache"], f"deployed-combination hold-out result ({FMT_TITLE.get(f, f)})"))
    cls = "fail" if dp.get("fallback") else "pass"
    label = "FELL BACK TO PLAIN CHAIN" if dp.get("fallback") else "COMBINATION OK"
    P.append(f"<div class='verdict {cls}'><h3><span class='badge'>{label}</span> &nbsp;{_e(', '.join(comps))}</h3>"
             f"<p class='small'>{_e(dp.get('reason', ''))}</p></div>")
    return "".join(P)


def render_recency(rc: dict, val: dict) -> str:
    """Report section: is weighting recent events more heavily worth it?"""
    P = ["<h2>Recency weighting: do recent events deserve more weight?</h2>"]
    mode = rc.get("mode", "auto")
    if mode == "off":
        P.append("<p>Recency weighting was switched off for this calculation; every game counts the same.</p>")
        return "".join(P)
    per_fmt = rc.get("scope") == "format"
    P.append("<p class='small mut'>Each game can be weighted by 0.5^(age / half-life), age = days from the start of its event to the end of "
             "the training data, so newer events count more (the metagame and players' skill change over a format's life). "
             + ("The half-life is chosen and tested <b>separately for each format</b>: it weights only that format's own games (the draft "
                "stage, and the Bo1 games that serve as the prior of Bo3, always keep equal weights). " if per_fmt else "")
             + "Step 1 picks the half-life using only the training part of the hold-out split (its own last 25% of events is scored; "
             "sigmas fixed so only the weighting differs). Step 2 refits the whole chain with that half-life (sigma re-chosen by CV) on the "
             "same 75% as the main model and scores the untouched hold-out. In <i>auto</i> mode the format uses the weighting only if "
             "step 2 shows a log-loss gain whose 95% CI excludes zero on the earlier half of the hold-out events, confirmed on the later half "
             "(same sign, one-sided 90% bound above zero), and the format is not significantly worse.</p>")
    inner = rc.get("inner", {})
    chd = rc.get("chosen") if isinstance(rc.get("chosen"), dict) else {f: rc.get("chosen") for f in inner}
    hdr = ["Half-life"] + [f"{_e(f)}: inner gain (95% CI)" for f in inner if inner[f].get("status") == "ok"]
    fmts = [f for f in inner if inner[f].get("status") == "ok"]
    if fmts:
        rows = []
        for i, r0 in enumerate(inner[fmts[0]]["rows"]):
            h = r0["half_life"]
            label = "none (equal weights)" if h is None else f"{h:g} days"
            cells = []
            for f in fmts:
                r = inner[f]["rows"][i]
                cells.append("reference" if h is None else f"{_f(r['diff'], 4, True)} {_ci(r['lo'], r['hi'], 4, True)}"
                             + (" (chosen)" if h is not None and h == chd.get(f) else ""))
            rows.append([label] + cells)
        P.append("<h3>Step 1 - choosing the half-life (inner split; log-loss gain vs equal weights, positive = weighting better)</h3>")
        P.append(_table(hdr, rows))
        P.append("<p class='small mut'>" + "; ".join(
            f"{_e(f)}: scored {inner[f]['n_inner_val_games']:,} games / {inner[f]['n_inner_val_events']} events from {_e(inner[f]['inner_start'][:10])}"
            for f in fmts) + ".</p>")
        for f in fmts:
            if (inner[f].get("cache") or {}).get("hit"):
                P.append(_cache_note(inner[f]["cache"], f"inner-split results ({FMT_TITLE.get(f, f)})"))
    else:
        P.append("<p>Not enough data for an inner split; no half-life could be chosen.</p>")
    if mode == "auto" and fmts and all(chd.get(f) is None for f in fmts):
        P.append("<p><b>No half-life beat equal weights on the inner split</b>, so the hold-out test was not needed and the model treats all "
                 "games equally.</p>")
    outer = [(f, val[f]["recency"]) for f in val if val[f].get("status") == "ok" and val[f].get("recency")]
    if outer:
        P.append("<h3>Step 2 - hold-out test (log-loss gain over equal weights, positive = weighting better)</h3>")
        P.append(_table(["Format", "Half-life", "Log-loss gain (95% CI)", "Earlier half of the events", "Later half (gain; 90% lower bound)",
                         "AUC gain (95% CI)"], [
            [_e(FMT_TITLE.get(f, f)), f"{o['half_life']:g} days", f"{_f(o['gain'], 4, True)} {_ci(o['lo'], o['hi'], 4, True)}",
             _halves_cells(o)[0], _halves_cells(o)[1],
             f"{_f(o['auc_gain'],4, True)} {_ci(o['auc_lo'], o['auc_hi'], 4, True)}"] for f, o in outer], left={0}))
    applied = rc.get("applied")
    cls = "pass" if applied else "skip"
    hl = rc.get("half_life")
    if isinstance(hl, dict):
        msg = ("Deployed model: " + "; ".join(f"{FMT_TITLE.get(f, f).split(' - ')[0]}: " + (f"recency-weighted, half-life {h:g} days" if h else "equal weights")
                                              for f, h in hl.items())) if applied else "Deployed model: equal weights"
    else:
        msg = (f"Deployed model: recency-weighted, half-life {hl:g} days" if applied else "Deployed model: equal weights")
    P.append(f"<div class='verdict {cls}'><h3><span class='badge'>{'APPLIED' if applied else 'NOT APPLIED'}</span> &nbsp;{_e(msg)}</h3>"
             f"<p class='small'>{_e(rc.get('reason', ''))}</p></div>")
    return "".join(P)


def _cache_note(c: dict | None, what: str = "hold-out results") -> str:
    """'Cached from <date>' line for a validation entry that was read from the cache (empty for a fresh computation)."""
    if not c:
        return ""
    if "parts" in c:                                         # a format's validation: several independently cached parts
        hit = {k: p for k, p in c["parts"].items() if p.get("hit")}
        if not hit:
            return ""
        dates = sorted({vcache.nice_date(p.get("created")) for p in hit.values()})
        names = ", ".join(("plain chain and baselines" if k == "core" else MODEL_LABEL.get(k, k)) for k in hit)
        return (f"<p class='small mut'><b>Cached from {_e(' / '.join(dates))}.</b> The {what} of {_e(names)} were computed on that date "
                "with identical data, code and settings (the key covers the corpus content, the model source code and every setting) "
                "and are reused without refitting; the other models of this table were fitted in this run.</p>")
    if c.get("hit"):
        return (f"<p class='small mut'><b>Cached from {_e(vcache.nice_date(c.get('created')))}.</b> The {what} were computed on that date "
                "with identical data, code and settings and are reused without refitting.</p>")
    return ""


def _decision_box(applied: bool, yes: str, no: str, reason: str) -> str:
    cls = "pass" if applied else "skip"
    return (f"<div class='verdict {cls}'><h3><span class='badge'>{'APPLIED' if applied else 'NOT APPLIED'}</span> &nbsp;"
            f"{_e(yes if applied else no)}</h3><p class='small'>{_e(reason)}</p></div>")


def _gain_table(val: dict, key: str, d: int = 4) -> str:
    rows = []
    for f, v in val.items():
        o = v.get(key) if v.get("status") == "ok" else None
        if not o:
            continue
        rows.append([_e(FMT_TITLE.get(f, f)), f"{_f(o['gain'], d, True)} {_ci(o['lo'], o['hi'], d, True)}",
                     f"{_f(o['auc_gain'], d, True)} {_ci(o['auc_lo'], o['auc_hi'], d, True)}",
                     f"{o['sigma']:.4f} vs {o['sigma_base']:.4f}"])
    if not rows:
        return "<p>No format had enough data for the hold-out test.</p>"
    return _table(["Format", "Log-loss gain (95% CI)", "AUC gain (95% CI)", "sigma(cards) with vs without"], rows, left={0})


def render_extra_draft(xd: dict) -> str:
    """Report section: the extra draft events (PickTwo / QuickDraft ...) kept or dropped by a hold-out test; empty when the set has none."""
    if not xd.get("available"):
        return ""
    P = ["<h2>Extra draft events (PickTwo / QuickDraft)</h2>"]
    ev = xd.get("events") or {}
    P.append("<p class='small mut'>The draft stage is the prior of the Sealed stages. Its standard data are PremierDraft and TradDraft games; the games of "
             "other draft events are used only if they pass a hold-out test, decided before any other component is tested. Test: the plain chain "
             "(M) is fitted with and without those games in the draft stage and both are scored on the same time hold-out of each Sealed format "
             "(draft and Bo1 events that started at or after the hold-out are left out of both); in <i>auto</i> mode they are kept only if the "
             f"paired log-loss gain has a 95% CI above zero and is at least {_f(1e-4, 4)} in one format, no format is materially worse, and the gain "
             "is confirmed on the later half of the hold-out events (see below). Without them a set with no other draft games has no draft stage at all.</p>")
    P.append(f"<p>Extra events in the data: <b>{_e(', '.join(f'{e} ({n:,} games)' for e, n in ev.items()))}</b>; "
             f"standard draft games: <b>{xd.get('core_draft_games', 0):,}</b>.</p>")
    rows = []
    for f, o in (xd.get("gain") or {}).items():
        a, b = _halves_cells(o)
        rows.append([_e(FMT_TITLE.get(f, f)), f"{_f(o['logloss_without'], 4)}", f"{_f(o['logloss_with'], 4)}",
                     f"{_f(o['gain'], 4, True)} {_ci(o['lo'], o['hi'], 4, True)}", a, b,
                     f"{o['n_holdout_events']:,} / {o['n_holdout_games']:,}"])
    if rows:
        P.append("<h3>Hold-out test: gain of keeping the extra events over dropping them (positive = they help)</h3>")
        P.append(_table(["Format", "Log loss without", "Log loss with", "Log-loss gain (95% CI)", "Earlier half (95% CI)",
                         "Later half (gain; 90% lower bound)", "Hold-out events / games"], rows, left={0}))
    elif xd.get("mode") == "auto":
        P.append("<p>No format had enough data for the hold-out test.</p>")
    P.append(_decision_box(bool(xd.get("applied")), "Deployed model: the extra draft events are in the draft stage",
                           "Deployed model: the draft stage uses the standard draft events only", xd.get("reason", "")))
    return "".join(P)


def render_attributes(at: dict, val: dict) -> str:
    """Report section: the card-attribute prior beta_c = A_c . gamma + u_c."""
    P = ["<h2>Card-attribute prior</h2>"]
    mode = at.get("mode", "auto")
    if mode == "off":
        P.append("<p>The card-attribute prior was switched off for this calculation; every card is shrunk towards its previous "
                 "stage's estimate (draft stage: towards zero).</p>")
        return "".join(P)
    P.append("<p class='small mut'>Each card's effect is split as beta_c = A_c . gamma + u_c: A_c are attributes of the card (mana value, "
             "rarity, colours, types, power / toughness, keywords, oracle-text tags such as removal or card draw), gamma is fitted jointly "
             f"({at.get('n_attr', 0)} deck-level columns: the sum of an attribute over the deck, standardised; prior SD 0.30 per SD in the draft "
             "stage, 0.10 around the previous stage afterwards) and u_c is the card-specific residual, penalised exactly as beta was before. "
             "The deployed card values below (and everywhere else) are the TOTAL beta = A.gamma + u. The prior should matter most for cards "
             "with little data. Test: the same chain is refitted with the attribute columns on the training part of the hold-out split and "
             "scored on the same hold-out (model MA); in <i>auto</i> mode the prior is used only if the paired log-loss gain has a 95% CI "
             "above zero in at least one format and no format is significantly worse.</p>")
    P.append("<h3>Hold-out test: gain of the attribute prior over the plain chain (positive = better)</h3>")
    P.append(_gain_table(val, "attributes"))
    P.append(_decision_box(bool(at.get("applied")), "Deployed model: card values include the attribute prior",
                           "Deployed model: plain chain (no attribute prior)", at.get("reason", "")))
    src = at.get("gamma_source", "validation")
    for f in at.get("formats", []):
        o = val.get(f, {}).get("attributes")
        if not o:
            continue
        rows = at.get("gamma", {}).get(f) or o.get("gamma") or []
        P.append(f"<h3>{_e(FMT_TITLE.get(f, f))}: fitted attribute effects</h3>")
        P.append(f"<p class='small mut'>Effect on game win probability of one more unit of the attribute in the deck (one more removal spell, "
                 f"one more point of mana value ...), percentage points near a 55% deck; SE is the cluster-robust one. Source: "
                 f"{'the deployed fit on all data' if src == 'final' else 'the training part of the validation split'}. "
                 f"The attributes explain {_pct(o.get('attr_share_of_var'), 0)} of the variance of the card effects (cards with exposure >= 2,000).</p>")
        P.append(_table(["Attribute", "Effect per unit (pp)", "SE (pp)", "t"], [
            [_e(r["label"]), _f(r["per_unit_pp"], 2, True), _f(r["se_pp"], 2), _f(r["t"], 1, True)] for r in rows[:12]], left={0}))
        mv = o.get("movers", [])
        lo_, hi_ = o.get("mean_abs_move_pp_low"), o.get("mean_abs_move_pp_high")
        if mv:
            P.append(f"<h3>{_e(FMT_TITLE.get(f, f))}: cards whose value moved most</h3>")
            note = ""
            if lo_ is not None and hi_ is not None:
                note = (f" Mean absolute move: {lo_:.2f} pp for the {o['n_low']} low-data cards (exposure < 2,000) versus {hi_:.2f} pp for the "
                        f"{o['n_high']} others.")
                if o.get("se_ratio_low") is not None:
                    note += f" The SE of a low-data card changes by a factor {o['se_ratio_low']:.2f} on average."
            P.append("<p class='small mut'>Difference between the total card effect with and without the attribute prior (both fitted on the "
                     "training part of the validation split), pp near 55%. Exposure = copies x games in the training part." + _e(note) + "</p>")
            P.append(_table(["Card", "Rarity", "Exposure", "Without (pp)", "With (pp)", "Moved (pp)", "Attribute part A.gamma (pp)"], [
                [_e(m["name"]) + (" <span class='low'>low data</span>" if m["exposure"] < 2000 else ""), _e(m["rarity"]), f"{m['exposure']:,.0f}",
                 _f(m["beta_plain_pp"], 2, True), _f(m["beta_attr_model_pp"], 2, True), _f(m["moved_pp"], 2, True),
                 _f(m["attr_part_pp"], 2, True)] for m in mv], left={0, 1}))
    return "".join(P)


def render_crossset(xs: dict, val: dict) -> str:
    """Report section: the cross-set prior of the structure terms."""
    P = ["<h2>Cross-set structure prior</h2>"]
    info = xs.get("prior") or {}
    if xs.get("mode") == "off":
        P.append("<p>The cross-set structure prior was switched off for this calculation.</p>")
        return "".join(P)
    P.append("<p class='small mut'>Land count, curve and mana terms are weakly identified inside one set (most decks play 17 lands). Other sets "
             "with a model share the game rules, so their fitted structure effects (converted through raw feature units) serve as the prior "
             "mean of this set's first stage; the prior SD is the between-set spread, never below the chain's own structure sigma. "
             "Tested like every other component: paired hold-out log-loss gain (model MX) with a 95% CI above zero, no format significantly worse.</p>")
    if not info.get("available"):
        P.append(f"<p><b>No-op.</b> {_e(info.get('note', 'no other set with a model'))}.</p>")
        return "".join(P)
    P.append(f"<p>Prior from: <b>{_e(', '.join(info['sets']))}</b>. {_e(info.get('note', ''))}</p>")
    P.append("<h3>Hold-out test: gain over the plain chain (positive = better)</h3>")
    P.append(_gain_table(val, "crossset", 6))
    P.append(_decision_box(bool(xs.get("applied")), "Deployed model: structure terms use the cross-set prior",
                           "Deployed model: structure terms shrink towards zero as before", xs.get("reason", "")))
    names = list(info["mean"].keys())
    P.append(_table(["Structure feature", "Prior mean (per SD)", "Between-set spread"], [
        [_e(n), _f(info["mean"][n], 3, True), _f(info["spread"][n], 3)] for n in names], left={0}))
    return "".join(P)


def _rz_fmt(key: str):
    """(value formatter, difference formatter, unit label) of a realism metric."""
    if key in ("mull", "hand_tail", "screw3"):
        return (lambda v: _pct(v, 1)), (lambda v: f"{100 * v:+.1f} pp"), "%"
    return (lambda v: f"{v:.2f}"), (lambda v: f"{v:+.2f}"), ""


def render_realism(rz: dict) -> str:
    """Simulated (v1 and v2) versus real statistics per replay event (see model/simcheck.py)."""
    from sealed_lab.model.simcheck import METRICS

    if not rz or rz.get("source") != "replay" or not rz.get("rows"):
        return ""
    P = ["<h3>Realism check against the replay data: v1 and v2 simulator, same decks</h3>"]
    P.append("<p class='small mut'>Every row compares real games of the event with the simulation of the same decks (each real game contributes its "
             "deck's simulated statistic); real values are re-weighted to 50/50 play/draw like the simulator. Differences are simulated minus real with "
             "the 95% cluster-bootstrap interval of the real value (events resampled); a good simulator has differences near zero. Turn-k statistics "
             "are conditional on the game lasting k turns (the replay files only describe completed turns), which flatters long games a little. "
             "'First 7 cards' are the hands Arena showed before any mulligan (already smoothed in Bo1 queues).</p>")
    labels = dict(METRICS)
    events = []
    for r in rz["rows"]:
        if r["event"] not in events:
            events.append(r["event"])
    score = {"v1": [0, 0], "v2": [0, 0]}
    for ev in events:
        rws = [r for r in rz["rows"] if r["event"] == ev]
        allr = next((r for r in rws if r["n_lands"] is None), None)
        if not allr:
            continue
        P.append(f"<h3 style='margin-top:14px'>{_e(ev)} <span class='mut'>({_e(allr['regime'])} regime; {allr['n_games']:,} games sampled, "
                 "15-18 land decks)</span></h3>")
        trs = []
        for key, lab in METRICS:
            m = allr["metrics"].get(key)
            if not m:
                continue
            f, fd, _ = _rz_fmt(key)
            row = [_e(lab), f"{f(m['real'])} <span class='mut'>[{f(m['lo'])}, {f(m['hi'])}]</span>"]
            for tag in ("v1", "v2"):
                t = m[tag]
                d, lo, hi = -t["diff"], -t["diff_hi"], -t["diff_lo"]
                sig = lo > 0 or hi < 0
                row += [f(t["sim"]), (f"<b>{fd(d)}</b>" if sig else fd(d)) + f" <span class='mut'>[{fd(lo)}, {fd(hi)}]</span>"]
                score[tag][0] += abs(d)
                score[tag][1] += 1
            row.append(f"{m['n']:,}")
            trs.append(row)
        tv = allr.get("hand_tvd")
        if tv:
            row = ["Distance (TV) of the first-hand land distribution", f"-"]
            for tag in ("v1", "v2"):
                t = tv[tag]
                row += ["-", f"{t['est']:.3f} <span class='mut'>[{t['lo']:.3f}, {t['hi']:.3f}]</span>"]
            row.append("")
            trs.append(row)
        P.append(_table(["Statistic", "Real (95% CI)", "v1", "v1 - real", "v2", "v2 - real", "Games"], trs, left={0}))
        byl = [r for r in rws if r["n_lands"] is not None]
        if len(byl) > 1:
            trs = []
            for r in byl:
                row = [str(r["n_lands"]), f"{r['n_games']:,}"]
                for key in ("mull", "screw3", "lands4", "mana4"):
                    m = r["metrics"].get(key)
                    if not m:
                        row += ["-", "-", "-"]
                        continue
                    f, fd, _ = _rz_fmt(key)
                    row += [f(m["real"]), fd(-m["v1"]["diff"]), fd(-m["v2"]["diff"])]
                trs.append(row)
            hdr = ["Lands", "Games"]
            for lab in ("Mulligan rate", "P(<3 lands, T3)", "Lands after T4", "Mana T1-T4"):
                hdr += [f"{lab}: real", "v1 - real", "v2 - real"]
            P.append("<details><summary>By land count</summary>" + _table(hdr, trs) + "</details>")
    if score["v1"][1]:
        P.append(f"<p class='small'>Mean absolute error over the statistics above (in each statistic's own unit, so a rough guide only): "
                 f"v1 {score['v1'][0] / score['v1'][1]:.3f}, v2 {score['v2'][0] / score['v2'][1]:.3f}.</p>")
    pr = rz.get("params") or {}
    if pr:
        P.append(f"<p class='small mut'>v2 revision {pr.get('revision')}: draw effects realised with probability {pr.get('kappa_draw')}, land-ramp effects "
                 f"{pr.get('kappa_fetch')}; a castable creature is cast with probability {pr.get('cast_p_creature')}, other spells {pr.get('cast_p_other')} "
                 "(instants, auras and tricks are mostly held; fitted on the real number of cards drawn and spells cast by turn 4). "
                 f"{rz.get('n_trials', '')} simulated games per deck.</p>")
    return "".join(P)


def render_manasim(mm: dict, val: dict) -> str:
    """Report section: the calibrated mana simulator (v2) as a model component."""
    P = ["<h2>Calibrated mana simulator (v2)</h2>"]
    if mm.get("mode") == "v1":
        P.append("<p>The calibrated mana simulator was switched off for this calculation (the original simulator, v1, was used).</p>")
        return "".join(P)
    P.append("<p class='small mut'>v2 replaces the simulator's assumptions by what the replay data show: the 7-card hand has the land "
             "distribution of its queue (plain shuffle in Trad / Bo3 queues, Arena's hand smoother in Bo1 queues, which is much stronger in the "
             "Sealed queue, where later draws are land-balanced too), mulligans follow the real keep rule by lands, play/draw and mulligan "
             "number, and spells that draw cards or fetch lands (oracle text) add cards to the flow. The structure features of a game are computed "
             "for the queue the game was played in (Bo3: random shuffle, Bo1 Sealed: smoothed hand and draws, Premier draft: smoothed hand); "
             "the chain's prior means pass between stages on the same feature names. Tested like every other component: the whole chain refitted "
             "on the v2 features (model MM) against the v1 chain (M) on the same hold-out.</p>")
    P.append("<h3>Hold-out test: gain over the v1 chain (positive = better)</h3>")
    P.append(_gain_table(val, "manasim", 6))
    P.append(_decision_box(bool(mm.get("applied")), "Deployed model: structure features come from the calibrated simulator (v2)",
                           "Deployed model: structure features come from the original simulator (v1)", mm.get("reason", "")))
    rows = []
    for f, v in val.items():
        o = v.get("manasim") if v.get("status") == "ok" else None
        if not o or "theta_v1" not in o:
            continue
        for nm in ("n_lands", "n_lands_dev2", "p_screw", "p_flood", "p_color_screw", "mana_spent", "mull_rate"):
            a1, a2 = o["theta_v1"][nm] / o["sd_v1"][nm], o["theta_v2"][nm] / o["sd_v2"][nm]
            rows.append([_e(FMT_TITLE.get(f, f)), _e(nm), f"{_f(o['theta_v1'][nm], 3, True)} / {_f(o['sd_v1'][nm], 3)}", f"{_f(a1, 3, True)}",
                         f"{_f(o['theta_v2'][nm], 3, True)} / {_f(o['sd_v2'][nm], 3)}", f"{_f(a2, 3, True)}"])
    if rows:
        P.append("<h3>Land and mana coefficients, v1 chain versus v2 chain (fitted on the training part of the hold-out split)</h3>")
        P.append(_table(["Format", "Feature", "v1: per SD / SD", "v1: per raw unit", "v2: per SD / SD", "v2: per raw unit"], rows, left={0, 1}))
        P.append("<p class='small mut'>Per SD = logit effect of one standard deviation of the feature across real builds; the SD differs between "
                 "simulators, so compare the per-raw-unit columns (logit per unit of the feature, e.g. per +100 percentage points of P(screw)).</p>")
    return "".join(P)



def render_castability(ck: dict, val: dict) -> str:
    """Report section: castability-weighted card copies and the dead-cards feature (model MK), with its shipping rule."""
    P = ["<h2>Castability (can the deck's mana cast the cards?)</h2>"]
    if ck.get("mode") == "off":
        P.append("<p>Castability was switched off for this calculation: card copies count fully whatever the deck's mana.</p>")
        return "".join(P)
    P.append("<p class='small mut'>Real decks almost never contain cards their mana cannot cast, so a model fitted on them would value an "
             "off-colour bomb as if it were playable. Following Frank Karsten, the probability that a card's coloured requirement is met "
             "is simulated for every deck and card: p = &frac12; P(met on curve) + &frac12; P(met two turns late), both given that the land "
             "drops were hit, with the real hands (Arena's Bo1 hand smoother, land-balanced Sealed draws), the real keep rule and "
             "mulligans. Lands, tapped / fast / slow lands, conditional duals (verge lands), pathways, filters, fetch lands, rocks and "
             "dorks, treasure makers, land cyclers and land-searching spells are modelled from their oracle text. The card part of the model "
             "uses copies &times; p, and a deck feature &ldquo;expected dead cards&rdquo; = &Sigma; copies &times; (1 &minus; p) gets its own "
             "coefficient, so the data price what an uncastable card costs.</p>")
    P.append("<h3>Hold-out test: gain over the plain chain (positive = better)</h3>")
    P.append(_gain_table(val, "castability", 6))
    P.append(_decision_box(bool(ck.get("applied")), "Deployed model: card values depend on castability",
                           "Deployed model: card values do not depend on castability", ck.get("reason", "")))
    P.append("<p class='small mut'>Shipping rule for this component: " + _e(ck.get("rule", "")) + ". It is a safety constraint against "
             "extrapolation (real data can only partly test it: real decks do contain thin splashes), so it does not have to prove a gain.</p>")
    st = ck.get("stats") or {}
    rows = []
    for f, o in st.items():
        rows.append([_e(FMT_TITLE.get(f, f)), f"{o['n_decks']:,}", _pct(o["decks_below_080"]), _pct(o["games_below_080"]),
                     _pct(o["games_below_050"]), f"{o['mean_dead']:.2f} (SD {o['sd_dead']:.2f})"])
    if rows:
        P.append("<h3>How castable are real decks?</h3>")
        P.append(_table(["Format", "Decks", "Decks with a card at p &lt; 0.8", "Games, same", "Games with a card at p &lt; 0.5",
                         "Expected dead cards per deck"], rows, left={0}))
    rows = []
    for f, v in val.items():
        o = v.get("castability") if v.get("status") == "ok" else None
        if not o or "dead_coef" not in o:
            continue
        rows.append([_e(FMT_TITLE.get(f, f)), f"{_f(o['dead_coef'], 4, True)} &plusmn; {o['dead_coef_se']:.4f}", f"{_f(o['dead_coef_pp'], 2, True)} pp",
                     f"{o['mean_p_all']:.3f}", _f(o.get("beta_corr"), 3) if o.get("beta_corr") is not None else "n/a"])
    if rows:
        P.append("<h3>Fitted price of a dead card (training part of the hold-out split)</h3>")
        P.append(_table(["Format", "Logit per dead card", "Win rate per dead card (near 55%)", "Mean castability of played copies",
                         "Correlation of card values M vs MK"], rows, left={0}))
        P.append("<p class='small mut'>Negative = a card the mana cannot cast costs the deck more than the slot it wastes (its value is "
                 "already removed by the copies &times; p weighting).</p>")
    for f, v in val.items():
        o = v.get("castability") if v.get("status") == "ok" else None
        if not o or not o.get("splash_cards"):
            continue
        rows = [[_e(r["name"]), f"{r['mean_p']:.2f}", f"{_f(r['pp_m'], 2, True)} pp", f"{_f(r['pp_mk'], 2, True)} pp",
                 f"{_f(r['pp_mk'] - r['pp_m'], 2, True)} pp"] for r in o["splash_cards"]]
        P.append(f"<h3>Card values of the least castable cards, {_e(FMT_TITLE.get(f, f))}</h3>")
        P.append(_table(["Card", "Mean castability where played", "Value M (win rate vs 55%)", "Value MK", "Change"], rows, left={0}))
    cats = ck.get("categories") or {}
    if cats:
        P.append("<h3>Mana sources recognised in this set</h3>")
        P.append(_table(["Category", "Cards", "Examples"], [[_e(k), str(v["n"]), _e(", ".join(v["names"][:6]))] for k, v in cats.items()], left={0, 2}))
    sup = ck.get("support") or {}
    rows = []
    for f, o in sup.items():
        rows.append([_e(FMT_TITLE.get(f, f)), str(o["max_spell_colours"]), ", ".join(f"{k}: {100 * v:.1f}%" for k, v in o["colours_dist"].items() if v > 0),
                     f"&le; {o['thin_sources_max']} sources"])
    if rows:
        P.append("<h3>Reference for the &ldquo;outside the data&rdquo; warnings</h3>")
        P.append(_table(["Format", "Max. spell colours (95% of games)", "Spell colours per game", "Flag when a spell colour has"], rows, left={0, 2}))
    return "".join(P)


def _extra_gain_table(val: dict, key: str, sigma_label: str, d: int = 4) -> str:
    rows = []
    for f, v in val.items():
        o = v.get(key) if v.get("status") == "ok" else None
        if not o:
            continue
        sg = o.get("extra_sigma_by_stage")
        sig_txt = ", ".join(f"{k}: {x:.4f}" for k, x in sg.items()) if sg else (f"{o['sigma']:.4f}" if o.get("sigma") else "-")
        edge = " (strongest prior: no signal)" if o.get("extra_edge") else ""
        rows.append([_e(FMT_TITLE.get(f, f)), f"{_f(o['gain'], d, True)} {_ci(o['lo'], o['hi'], d, True)}",
                     f"{_f(o['auc_gain'], d, True)} {_ci(o['auc_lo'], o['auc_hi'], d, True)}", _e(sig_txt + edge)])
    if not rows:
        return "<p>No format had enough data for the hold-out test.</p>"
    return _table(["Format", "Log-loss gain (95% CI)", "AUC gain (95% CI)", sigma_label], rows, left={0, 3})


def render_synergy(sy: dict, val: dict) -> str:
    """Report section: card-pair synergies (model MS) and the descriptive co-occurrence lift."""
    P = ["<h2>Card-pair synergies</h2>"]
    if sy.get("mode") == "off":
        P.append("<p>The synergy layer was switched off for this calculation.</p>")
        return "".join(P)
    P.append("<p class='small mut'>For every pair of spells that shared at least "
             f"{sy.get('min_games', 300)} games (at most {sy.get('max_pairs', 3000)} pairs, the most frequent first) the deck gets a column "
             "min(copies of A, copies of B): the extra logit effect of having both in the deck, on top of their single-card effects. "
             "The coefficient is shrunk towards 0 (towards the earlier stage's value in the chain) with its own prior SD per stage, chosen by "
             "grouped CV at the plain chain's card sigma. Test: the same chain with the pair columns (model MS) vs the plain chain on the "
             "same hold-out. A pair is called a synergy only if the data support its sign (P &ge; 90%) <i>and</i> it is worth &ge; 0.25 pp.</p>")
    if not sy.get("n_pairs"):
        P.append(f"<p><b>No-op.</b> {_e(sy.get('reason', 'no pair reached the co-occurrence threshold'))}</p>")
        return "".join(P)
    P.append(f"<p>{sy['n_pairs']:,} candidate pairs.</p>")
    P.append("<h3>Hold-out test: gain of the pair layer over the plain chain (positive = better)</h3>")
    P.append(_extra_gain_table(val, "synergy", "Prior SD of a pair coefficient (logit per pair)"))
    P.append(_decision_box(bool(sy.get("applied")), "Deployed model: pair synergies are part of every deck's win rate",
                           "Deployed model: no pair synergies", sy.get("reason", "")))
    fin = sy.get("final") or {}
    for f in [f for f in val if val[f].get("status") == "ok" and val[f].get("synergy")]:
        o = val[f]["synergy"]
        pr = fin.get(f) or o.get("pairs")
        if not pr or "top_pos" not in pr:
            continue
        src = "deployed fit on all data" if f in fin else "validation fit (earlier 75% of events; not deployed)"
        P.append(f"<h3>{_e(FMT_TITLE.get(f, f))}: fitted pair effects ({_e(src)})</h3>")
        P.append(f"<p class='small'>SD of the pair effects <b>{pr['sd_pp']:.3f} pp</b>, largest <b>{pr['max_abs_pp']:.2f} pp</b>. "
                 f"Pairs clearing the threshold (P &ge; 90% and &ge; 0.25 pp): <b>{pr['n_reliable']}</b> "
                 f"({pr['n_reliable_pos']} positive, {pr['n_reliable_neg']} negative). With {pr['n_pairs']:,} pairs, "
                 f"|z| &gt; 3 is expected by chance for about {pr.get('expected_abs_z3', 0):.0f}; observed {pr.get('n_abs_z3', '?')}. "
                 + (f"Spearman correlation between pair effects and log co-occurrence lift: {_f(o.get('spearman_lift'), 2, True)}."
                    if o.get("spearman_lift") is not None else "") + "</p>")
        hdr = ["Pair", "Effect (pp, 95% CI)", "P(&gt;0)", "Games with both", "Lift", "Clears threshold"]

        def row(r):
            return [_e(r["a"]) + " + " + _e(r["b"]), f"{_f(r['pp'], 3, True)} [{_f(r['lo_pp'], 3, True)}, {_f(r['hi_pp'], 3, True)}]",
                    f"{100 * r['p_pos']:.1f}%", f"{r['games']:,.0f}", _f(r.get("lift"), 2) if r.get("lift") is not None else "-",
                    "yes" if r.get("reliable") else "no"]
        P.append("<details><summary>Strongest positive pairs</summary>" + _table(hdr, [row(r) for r in pr["top_pos"]], left={0}) + "</details>")
        P.append("<details><summary>Strongest negative pairs</summary>" + _table(hdr, [row(r) for r in pr["top_neg"]], left={0}) + "</details>")
    lt = sy.get("lift_top") or []
    if lt:
        P.append("<h3>Discovery: highest colour-adjusted co-occurrence lift (descriptive, no model)</h3>")
        P.append("<p class='small mut'>Decks with both cards divided by the number expected if each card were drawn independently within decks "
                 "of the same colour pair (EDHREC-style lift). It says which cards are <i>played</i> together (archetypes), not which win more "
                 "together; the model-based effects above are the evidence for the latter.</p>")
        P.append(_table(["Pair", "Lift", "Decks with both", "Expected"], [
            [_e(r["a"]) + " + " + _e(r["b"]), f"{r['lift']:.2f}", f"{r['obs']:,.0f}", f"{r['exp']:,.0f}"] for r in lt], left={0}))
    return "".join(P)


def render_archcond(ac: dict, val: dict) -> str:
    """Report section: archetype-conditional card values (model MC)."""
    P = ["<h2>Archetype-conditional card values</h2>"]
    if ac.get("mode") == "off":
        P.append("<p>Switched off for this calculation.</p>")
        return "".join(P)
    P.append("<p class='small mut'>Each deck is assigned to its nearest archetype (see the archetype section); a card gets an extra, heavily "
             f"shrunk effect inside each archetype where it was played in enough games ({ac.get('n_columns', 0):,} card x archetype columns, "
             "prior SD chosen by grouped CV). Test: model MC vs the plain chain on the same hold-out; deployed only if the gain has a 95% CI "
             "above zero and no format is significantly worse.</p>")
    if not ac.get("n_columns"):
        P.append(f"<p><b>No-op.</b> {_e(ac.get('reason', ''))}</p>")
        return "".join(P)
    P.append(_extra_gain_table(val, "archcond", "Prior SD of a card x archetype effect (logit)"))
    P.append(_decision_box(bool(ac.get("applied")), "Deployed model: card values depend on the deck's archetype",
                           "Deployed model: one value per card", ac.get("reason", "")))
    return "".join(P)


def render_archetypes(ar: dict | None) -> str:
    """Report section: the archetype clusters with observed win rates."""
    P = ["<h2>Archetypes</h2>"]
    if not ar or not ar.get("archetypes"):
        P.append("<p>Not enough decks to cluster.</p>")
        return "".join(P)
    pr = ar.get("params", {})
    P.append("<p class='small mut'>Real decks (draft and Sealed) clustered by composition: TF-IDF of the spells, truncated SVD "
             f"({pr.get('rank')} axes), then k-means within each colour pair (k = 2..{pr.get('k_max')} by mean silhouette; k &gt; 1 only if the best "
             f"silhouette is &ge; {pr.get('sil_min')} and every cluster holds &ge; {100 * pr.get('min_share', 0):.0f}% of the group). A deck "
             "belongs to the archetype whose centroid has the highest cosine similarity. Win rates are the <b>observed</b> game win rates of "
             f"decks in the archetype (raw, then shrunk towards the format mean with {ar.get('shrink_games', 0):.0f} pseudo-games; 95% interval by "
             "bootstrapping events). They include pilot skill and the metagame of the time: descriptive, not causal.</p>")
    rows = []
    for a in ar["archetypes"]:
        cells = []
        for f in ("bo3", "bo1", "draft"):
            r = a["rates"].get(f)
            cells.append("-" if not r else f"{100 * r['shrunk']:.1f}% [{100 * r['lo']:.1f}, {100 * r['hi']:.1f}]<br><span class='mut'>{r['games']:,} games</span>")
        sig = ", ".join(f"{_e(x['name'])} ({100 * x['rate']:.0f}%)" for x in a["signature"][:4])
        rows.append([_e(a["label"]), f"{a['n_decks']:,}"] + cells + [sig])
    P.append(_table(["Archetype", "Decks", "Bo3 win rate (shrunk, 95% CI)", "Bo1", "Draft", "Signature cards (share of decks)"], rows, left={0, 5}))
    g = ar.get("groups", {})
    if g:
        P.append("<details><summary>Cluster selection (mean silhouette by k)</summary>" + _table(
            ["Colour group", "Decks", "k chosen"] + [f"k={k}" for k in range(2, pr.get("k_max", 5) + 1)],
            [[_e(k), f"{v['n']:,}", str(v["k"])] + [_f(v["silhouette"].get(str(j)), 3) for j in range(2, pr.get("k_max", 5) + 1)]
             for k, v in sorted(g.items())]) + "</details>")
    return "".join(P)


def render_playdraw(pd: dict, val: dict) -> str:
    """Report section: card x play/draw interactions (model MP) and the games-2-3 advice they feed."""
    P = ["<h2>Play/draw card effects (Bo3 games 2-3)</h2>"]
    if pd.get("mode") == "off":
        P.append("<p>The play/draw layer was switched off for this calculation.</p>")
        return "".join(P)
    P.append("<p class='small mut'>On top of the fitted chain each card gets an interaction gamma<sub>c</sub> with being on the play "
             "(s = +1/2 on the play, -1/2 on the draw), fitted on the pooled games of all formats in the training part with a zero-mean "
             "prior whose SD is chosen by grouped CV over events. A card's main effect stays its value at 50/50, so the deck score and the "
             "builder (game 1) are unchanged by construction. Test (model MP): the held-out prediction adds the card shifts for the game's "
             "own play/draw status, on the same hold-out as the plain chain.</p>")
    rows = []
    for f, v in val.items():
        o = v.get("playdraw") if v.get("status") == "ok" else None
        if not o:
            continue
        t = o.get("table", {})
        rows.append([_e(FMT_TITLE.get(f, f)), f"{_f(o['gain'], 4, True)} {_ci(o['lo'], o['hi'], 4, True)}",
                     f"{_f(o['auc_gain'], 4, True)} {_ci(o['auc_lo'], o['auc_hi'], 4, True)}", f"{o['sigma']:.4f}",
                     f"{o.get('n_pool_games', 0):,}", f"{_f(t.get('sd_pp', 0.0), 2)}"])
    if rows:
        P.append(_table(["Format", "Log-loss gain (95% CI)", "AUC gain (95% CI)", "Chosen sigma", "Pooled training games",
                         "SD of the card effects (pp, play minus draw)"], rows, left={0}))
    else:
        P.append("<p>No format had enough data for the hold-out test.</p>")
    P.append(_decision_box(bool(pd.get("applied")), "Deployed model: games-2-3 advice uses play/draw card effects",
                           "Deployed model: no play/draw advice", pd.get("reason", "")))
    return "".join(P)


def render_matchups(mo: dict, val: dict) -> str:
    """Report section: card x opponent-colour interactions (model MO) and sideboard evidence."""
    P = ["<h2>Opponent matchups and sideboard plans</h2>"]
    if mo.get("mode") == "off":
        P.append("<p>The matchup layer was switched off for this calculation.</p>")
        return "".join(P)
    P.append("<p class='small mut'>Every game records the opponent's colours. On top of the fitted chain, each card gets an interaction with the "
             "opponent's colours, delta<sub>c,o</sub>, fitted on the pooled games of all formats in the training part (draft is the bulk) with the "
             "chain's own predictor as offset, a free per-card main-effect adjustment (so the opponent-average effect stays in the main effects), "
             "and a zero-mean prior whose SD is chosen by grouped CV. Two encodings are compared by CV on the training part: the 10 colour pairs "
             "+ 'other' (3-colour and other opponents), and 'colour' (one column per opponent colour, multi-hot). Test (model MO): the held-out "
             "prediction adds only the <i>centred</i> effect (the card's value against this opponent minus its average value) on the same "
             "hold-out as the plain chain; the gain therefore also shows whether what is learned mostly from draft transfers to Sealed. "
             "Premium sideboard card: see the thresholds below.</p>")
    P.append("<h3>Hold-out test: gain of the matchup layer over the plain chain (positive = better)</h3>")
    rows = []
    for f, v in val.items():
        o = v.get("opponent") if v.get("status") == "ok" else None
        if not o:
            continue
        cv = o.get("cv", {})
        cvt = "; ".join(f"{m}: CV gain {c['gain_vs_strongest']:.5f}/game at sigma {c['grid'][c['best']]:.4f}" for m, c in cv.items())
        rows.append([_e(FMT_TITLE.get(f, f)), f"{_f(o['gain'], 4, True)} {_ci(o['lo'], o['hi'], 4, True)}",
                     f"{_f(o['auc_gain'], 4, True)} {_ci(o['auc_lo'], o['auc_hi'], 4, True)}",
                     f"{_e(o.get('mode', '?'))}, sigma {o['sigma']:.4f}", f"{o.get('n_pool_games', 0):,}", _e(cvt)])
    if rows:
        P.append(_table(["Format", "Log-loss gain (95% CI)", "AUC gain (95% CI)", "Chosen encoding", "Pooled training games",
                         "CV on the training part"], rows, left={0, 3, 5}))
    else:
        P.append("<p>No format had enough data for the hold-out test.</p>")
    P.append(_decision_box(bool(mo.get("applied")), "Deployed model: sideboard plans use opponent-specific card values",
                           "Deployed model: no opponent-specific plans (nothing is marked in the app)", mo.get("reason", "")))
    if mo.get("applied") and mo.get("applied_formats"):
        src = mo.get("source") or {}
        P.append("<p class='small'>The advice is shown in: " + _e(", ".join(f"{FMT_TITLE.get(f, f)} ({src.get(f, 'per-set test')})"
                                                                         for f in mo["applied_formats"])) + ". The layer only feeds the matchup and "
                 "sideboard advice; it never enters the deck score or the builder.</p>")
    if mo.get("pooled"):
        rows = []
        for f, s in mo["pooled"].items():
            if not s.get("decided"):
                rows.append([_e(FMT_TITLE.get(f, f)), str(s.get("n_sets", 0)), "-", "-", "-", "-", "not used", _e(s.get("why", ""))])
                continue
            r = s["re"]
            rows.append([_e(FMT_TITLE.get(f, f)), str(s["n_sets"]), f"{_f(r['whole']['gain'], 4, True)} {_ci(r['whole']['lo'], r['whole']['hi'], 4, True)}",
                         f"{_f(r['A']['gain'], 4, True)} {_ci(r['A']['lo'], r['A']['hi'], 4, True)}",
                         f"{_f(r['B']['gain'], 4, True)}; {_f(s['B_lo90'], 4, True)}", f"{r['whole']['tau2']:.1e}", _e(s["verdict"]),
                         ("applied: " if s["apply"] else "not applied: ") + _e(s["why"])])
        P.append("<h3>Evidence pooled across the other sets (this set is not part of it)</h3>")
        P.append("<p class='small mut'>The per-set test above did not apply the layer, so each format is also judged by the latest stored "
                 "hold-out numbers of the other sets, pooled with DerSimonian-Laird random effects (rule: docs/pooled_retest/POOLING_RULE.md): "
                 "pooled-better = the earlier-half gain has a 95% CI above zero and is at least the materiality floor, and the later half is "
                 "positive with a one-sided 90% bound above zero; pooled-worse (whole hold-out or either half) blocks. At least 5 other sets are "
                 "needed, and this set's own numbers must not be materially worse. Gains are log loss per game (positive = better).</p>")
        P.append(_table(["Format", "Other sets", "Pooled whole (95% CI)", "Pooled earlier half (95% CI)", "Pooled later half (gain; 90% lower bound)",
                         "tau&sup2; (whole)", "Pooled verdict", "Decision"], rows, left={0, 6, 7}))
    fin = mo.get("final")
    for f in [f for f in val if val[f].get("status") == "ok" and val[f].get("opponent")][:1]:
        o = val[f]["opponent"]
        t = fin["table"] if fin else o.get("table")
        if not t:
            continue
        src = "deployed fit on all games" if fin else f"validation fit of the {f} run (earlier 75% of events; NOT deployed)"
        P.append(f"<h3>Card x opponent effects ({_e(src)})</h3>")
        P.append(f"<p class='small'>{t['n_cells']:,} card x opponent-pair cells (centred effects: value against that opponent minus the card's average "
                 f"value). SD {t['sd_pp']:.2f} pp, largest {t['max_pp']:.2f} pp. <b>Cells clearing the thresholds</b> (P &ge; {t['p_min']:.0%} and "
                 f"&ge; {t['pp_min']:g} pp): <b>{t['n_pos']} positive</b>, {t['n_neg']} negative. By chance alone |z| &gt; 3 would occur in about "
                 f"{t.get('expected_abs_z3', 0):.0f} cells; observed {t.get('n_abs_z3', '?')}.</p>")
        P.append(_table(["Card", "Opponent", "Effect vs average (pp, 95% CI)", "P(&gt;0)", "Exposure"], [
            [_e(r["card"]), _e(r["opp"]), f"{_f(r['pp'], 2, True)} [{_f(r['lo_pp'], 2, True)}, {_f(r['hi_pp'], 2, True)}]",
             f"{100 * r['p_pos']:.1f}%", f"{r['exposure']:,.0f}"] for r in t["top"][:15]], left={0, 1}))
    P.append("<p class='small mut'>A card is a <b>premium sideboard card against strategy o</b> for a given build when its effect against o is reliably "
             "above its average (P &ge; 90%, &ge; 0.5 pp) <i>and</i> swapping it in for the weakest main-deck card it can replace raises the win "
             "rate against o by &ge; 0.5 pp in &ge; 90% of the bootstrap draws (computed with the full model, mana simulation included, by the app "
             "for the build on screen).</p>")
    return "".join(P)


def _kpi(label: str, value: str) -> str:
    return f'<div><b>{value}</b><span>{_e(label)}</span></div>'


def render_report(R: dict) -> str:
    """Self-contained HTML report from the report dictionary `R` (see pipeline.calculate)."""
    P = []
    P.append(f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
             f"<title>sealed_lab model report - {_e(R['set_code'])}</title><style>{CSS}</style></head><body><main>")
    P.append(f"<h1>Model report - {_e(R['set_code'])}</h1>")
    P.append(f"<p class='mut'>Built {_e(R['built_at'])} - code {_e(R['code_version'])} - "
             + " - ".join(f"{_e(f)}: {R['n_games'].get(f, 0):,} games" for f in ("bo3", "bo1", "draft") if R["n_games"].get(f))
             + f" - {R['n_boot']} bootstrap replicates - {R['seconds']:.0f}s</p>")
    # ---------------- headline
    P.append("<h2 style='border-top:0;margin-top:14px'>Does the model work out of sample?</h2>")
    P.append("<p class='small mut'>Train on the first 75% of events (by start time) of the target format, plus earlier draft / Bo1 data as prior; "
             "score the last 25% of events. Differences are baseline minus full model (positive = the model is better); intervals are 95% "
             "cluster bootstrap over held-out draft_ids.</p>")
    for fmt in R["formats"]:
        v = R["validation"].get(fmt, {"status": "skipped", "reason": "not run"})
        cls, label, lines = verdict_for({**v, "fmt": fmt})
        P.append(f"<div class='verdict {cls}'><h3><span class='badge'>{label}</span> &nbsp;{_e(FMT_TITLE.get(fmt, fmt))}</h3>")
        if v.get("status") == "ok":
            P.append(f"<p class='small mut'>Hold-out: {v['n_holdout_events']} events / {v['n_holdout_games']:,} games starting {_e(v['holdout_start'][:10])}; "
                     f"trained on {v['n_events'] - v['n_holdout_events']} events / {v['n_train_games']:,} games.</p>")
        P.append("<ul>" + "".join(f"<li>{l}</li>" for l in lines) + "</ul></div>")
        P.append(_cache_note(v.get("cache")))
    for w in R["warnings"]:
        P.append(f"<div class='warnbox'>{_e(w)}</div>")
    # ---------------- per-format validation details
    for fmt in R["formats"]:
        v = R["validation"].get(fmt)
        if not v or v.get("status") != "ok":
            continue
        met = v["metrics"]
        P.append(f"<h2>Held-out performance - {_e(FMT_TITLE.get(fmt, fmt))}</h2>")
        rows = []
        for m in MODELS:
            if m not in met:
                continue
            r = met[m]
            ll = f"{r['logloss']:.4f}<br><span class='mut'>{_ci(*r['logloss_ci'])}</span>"
            if m == "M":
                rows.append([("M",), _e(MODEL_LABEL[m]), ll, f"{r['brier']:.4f}", f"{r['auc']:.4f}", "-", "-"])
            elif m == "MR":
                d = r["vs_M"]
                hl = v.get("recency", {}).get("half_life")
                rows.append([m, _e(MODEL_LABEL[m]) + (f" (half-life {hl:g} d)" if hl else ""), ll, f"{r['brier']:.4f}", f"{r['auc']:.4f}",
                             f"{_f(d['logloss']['diff'], 4, True)}<br><span class='mut'>{_ci(d['logloss']['lo'], d['logloss']['hi'], 4, True)}</span>",
                             f"{_f(d['auc']['diff'], 4, True)}<br><span class='mut'>{_ci(d['auc']['lo'], d['auc']['hi'], 4, True)}</span>"])
            else:
                d = r["vs_M"]
                rows.append([m, _e(MODEL_LABEL[m]), ll, f"{r['brier']:.4f}", f"{r['auc']:.4f}",
                             f"{_f(d['logloss']['diff'], 4, True)}<br><span class='mut'>{_ci(d['logloss']['lo'], d['logloss']['hi'], 4, True)}</span>",
                             f"{_f(d['auc']['diff'], 4, True)}<br><span class='mut'>{_ci(d['auc']['lo'], d['auc']['hi'], 4, True)}</span>"])
        P.append(_table(["", "Model", "Log loss (95% CI)", "Brier", "AUC", "Model gain, log loss", "Model gain, AUC"], rows, left={0, 1}))
        P.append(f"<p class='small mut'>Base win rate: train {_pct(v['base_rate_train'], 2)}, hold-out {_pct(v['base_rate_holdout'], 2)} "
                 "(a shift between the two is a time drift that no model fitted on the past can anticipate; it lowers every row equally). "
                 "B1 uses the shrunk games-in-hand win rate (50 pseudo-games) of each spell, computed on the training games only.</p>")
        cal = v["calibration"]
        crow = [[f"{i + 1}", f"{b['n']:,}", _pct(b["pred"], 2), _pct(b["obs"], 2), f"{_pct(b['wilson'][0], 1)} - {_pct(b['wilson'][1], 1)}",
                 f"{_pct(b['cluster'][0], 1)} - {_pct(b['cluster'][1], 1)}"] for i, b in enumerate(cal["bins"])]
        P.append("<h3>Calibration (deciles of predicted win probability, hold-out games)</h3><div class='cols'><div>")
        P.append(svg_calibration(cal["bins"]))
        P.append("</div><div>")
        P.append(_table(["Decile", "Games", "Predicted", "Observed", "Wilson 95%", "Cluster 95%"], crow))
        P.append(f"<p class='small'>Slope {_f(cal['slope'], 3)} {_ci(*cal['slope_ci'], 3)}, intercept {_f(cal['intercept'], 3, True)} {_ci(*cal['intercept_ci'], 3, True)}. "
                 "Wilson intervals treat games as independent; the cluster intervals resample whole events and are the honest ones.</p></div></div>")
        vb = v["variance_budget"]
        P.append("<h3>Variance budget (hold-out builds)</h3>")
        P.append("<div class='kpi'>" + _kpi("between-deck SD of win rate", f"{100 * math.sqrt(max(vb['sigma2_deck'], 0)):.2f} pp")
                 + _kpi("SD of predicted deck effect", f"{100 * math.sqrt(max(vb['pred_var'], 0)):.2f} pp")
                 + _kpi("share predicted (raw)", _pct(vb["share_raw"], 0))
                 + _kpi("share explained (re-scaled)", _pct(vb["share_recal"], 0))
                 + _kpi("calibration of deck effect (slope)", _f(vb["slope"], 2)) + "</div>")
        P.append(f"<p class='small mut'>sigma^2_deck = Var(build win rate net of covariates) - mean binomial variance = {vb['sigma2_deck']:.5f} "
                 f"(95% CI {_ci(*vb['ci']['sigma2_deck'], 5)}) over {vb['n_builds']} builds. 'Raw' share = Var(predicted)/sigma^2_deck; "
                 "'re-scaled' = squared covariance / (Var(predicted) sigma^2_deck), the best share a linear re-calibration of the predictions could explain. "
                 f"Share CI: {_ci(*vb['ci']['share_recal'], 2)}.</p>")
    # ---------------- the deployed combination of components
    if R.get("deployed"):
        P.append(render_deployed(R["deployed"]))
    # ---------------- the draft stage's extra events (decided before every other component)
    if R.get("extra_draft"):
        P.append(render_extra_draft(R["extra_draft"]))
    # ---------------- recency
    if R.get("recency"):
        P.append(render_recency(R["recency"], R["validation"]))
    if R.get("attributes"):
        P.append(render_attributes(R["attributes"], R["validation"]))
    if R.get("crossset"):
        P.append(render_crossset(R["crossset"], R["validation"]))
    if R.get("synergy"):
        P.append(render_synergy(R["synergy"], R["validation"]))
    if R.get("archetypes") is not None or R.get("archcond"):
        P.append(render_archetypes(R.get("archetypes")))
    if R.get("archcond"):
        P.append(render_archcond(R["archcond"], R["validation"]))
    if R.get("matchups"):
        P.append(render_matchups(R["matchups"], R["validation"]))
    if R.get("playdraw"):
        P.append(render_playdraw(R["playdraw"], R["validation"]))
    if R.get("manasim"):
        P.append(render_manasim(R["manasim"], R["validation"]))
    if R.get("castability"):
        P.append(render_castability(R["castability"], R["validation"]))
    P.append(render_confirmation(R))
    # ---------------- chain
    P.append("<h2>Model chain: priors, sigma by cross-validation</h2>")
    P.append("<p class='small mut'>Draft data (stage D) is the prior of Sealed Bo1 (S1), which is the prior of TradSealed Bo3 (S3). Card effects are shrunk towards the "
             "previous stage's estimate; sigma (prior SD of a card's logit effect around its prior mean) is chosen by cross-validation. "
             "For the Sealed stages that is a <b>forward-chaining time split</b>: events in start-time order are cut into six blocks, each block "
             "from the second on is scored by a model trained on the earlier blocks only (so a pattern that holds within the period but not "
             "forward in time is not rewarded); the draft stage uses 5 random event folds. On the same folds the <b>stacking weight w</b> of "
             "q = w M + (1 - w) P is chosen at that sigma (P = the prior mean, i.e. the previous stage's values rescaled; w = 0 means the "
             "Sealed games add nothing and the draft-based values are used). The card values, structure effects and draws deployed are those of q.</p>")
    srows = []
    for n in ("draft", "bo1", "bo3"):
        s = R["stages"].get(n)
        if not s:
            continue
        lam = "-" if s.get("lam") is None else f"{s['lam']:.2f} (SE {s['lam_se']:.2f}) x {s['lam_from']}"
        wl = s.get("stack_loss") or []
        wcell = "-" if not wl else (f"{s['stack_w']:.1f}" + f" (held-out loss: w=0 {wl[0]:.5f}, w=1 {wl[-1]:.5f})")
        srows.append([{"draft": "D: draft", "bo1": "S1: Sealed Bo1", "bo3": "S3: TradSealed Bo3"}[n], f"{s['n_games']:,}", f"{s['n_events']:,}",
                      f"{s['sigma']:.4f}", lam, "yes (" + ("time split" if s.get("cv_scheme") == "forward" else "random folds") + ")"
                      if s["cv_done"] else "skipped", wcell, svg_cv(s["cv_grid"], s["cv_loss"], s["sigma"], 240, 110)])
    P.append(_table(["Stage", "Games", "Events", "sigma (cards)", "lambda (scale of prior)", "CV", "Stacking weight w", "CV log-loss curve"], srows, left={0}))
    # ---------------- cards
    for fmt in R["formats"]:
        cd = R["cards"].get(fmt)
        if not cd:
            continue
        P.append(f"<h2>Card effects - {_e(FMT_TITLE.get(fmt, fmt))}</h2>")
        P.append("<p class='small mut'>beta = change in the logit of game win probability from one extra copy in the main deck (everything else equal, "
                 "after shrinkage). pp = effect at a 55% deck. SE = approximate posterior SD from the penalised Hessian. Exposure = copies x games in this "
                 "format; <span class='low'>low data</span> = exposure under 2,000 (estimate is mostly its prior).</p>")
        hdr = ["Card", "Rarity", "beta", "pp @55%", "SE (pp)", "Exposure", "psi", "Check"]

        def crow(c):
            chk = "<span class='flag'>flag</span>" if c["flag"] else ""
            lo = " <span class='low'>low data</span>" if c["low_data"] else ""
            return [_e(c["name"]) + lo, _e(c["rarity"]), _f(c["beta"], 3, True), _f(c["pp"], 2, True), _f(c["se_pp"], 2), f"{c['exposure']:,.0f}",
                    _f(c["psi"], 3, True), chk]
        P.append("<h3>Top 20 cards</h3>" + _table(hdr, [crow(c) for c in cd["top"]], left={0, 1}))
        P.append("<h3>Bottom 20 cards</h3>" + _table(hdr, [crow(c) for c in cd["bottom"]], left={0, 1}))
        hc = cd["handcheck"]
        P.append("<h3>Deck effect (beta) versus opening-hand effect (psi)</h3>")
        sp_ = "n/a" if hc.get("spearman") is None else f"{hc['spearman']:.2f}"
        kap = hc.get("kappa")
        if hc.get("informative"):
            verdict = (f"{hc['n_flagged']} card(s) flagged (|beta - kappa psi| > 2 sqrt(SE^2 + tau^2), kappa = {_f(kap, 2)} +/- "
                       f"{_f(hc.get('kappa_se'), 2)}, between-card spread tau = {_f(hc.get('tau'), 3)}): "
                       "their deck effect is not echoed by an effect when they are actually in the opening hand, so it may be riding on "
                       "archetype or pilot selection rather than on the card's own power.")
        else:
            verdict = ("The opening-hand effects show no systematic relation to the deck effects here "
                       f"(kappa = {_f(kap, 3)} +/- {_f(hc.get('kappa_se'), 3)}), so the check has no power and no card is flagged. "
                       "(Expected on synthetic data, where the outcome depends on the deck only; on real data a clear positive relation is expected.)")
        P.append(f"<p>Spearman(beta, psi) = <b>{sp_}</b> over {hc['n']} cards with exposure >= 200. {verdict} "
                 "psi is estimated once from the no-mulligan games of all formats pooled (offset = each game's fitted logit), "
                 "so it is the same number in every format's table.</p>")
        P.append(svg_scatter(cd["scatter"]["psi"], cd["scatter"]["beta"], cd["scatter"]["flag"], "psi (opening-hand effect, logit)",
                             "beta (deck effect, logit)", hc.get("kappa")))
        if hc["n_flagged"]:
            P.append("<details><summary>Flagged cards</summary>" + _table(["Card", "beta", "psi", "kappa x psi"], [
                [_e(c["name"]), _f(c["beta"], 3, True), _f(c["psi"], 3, True), _f(c["expected"], 3, True)] for c in cd["flagged"]]) + "</details>")
    # ---------------- structure
    P.append("<h2>Structure and mana terms</h2>")
    P.append("<p class='small mut'>Effects per 1 standard deviation (across real builds) of each deck feature, logit scale, final fit of each format "
             "with approximate SE. The mana terms come from the Monte-Carlo simulator.</p>")
    hdrs = ["Feature"] + [f for f in R["formats"]]
    rws = []
    for name in R["feature_names"]:
        row = [_e(name)]
        for f in R["formats"]:
            th = R["theta"][f][name]
            row.append(f"{th['est']:+.3f} (SE {th['se']:.3f})")
        rws.append(row)
    P.append(_table(hdrs, rws))
    # ---------------- simulator
    P.append("<h2>Mana simulator versus observed games</h2>")
    P.append("<p class='small mut'>By number of lands in the 40-card deck: mulligan rate and mean lands in kept 7-card opening hands. Simulated: the simulator's own "
             "keep rule (2-5 lands). Observed: games with no mulligan; players keep more hands than the rule, and Arena's Bo1 hand smoother (draft/Bo1) "
             "pulls opening hands towards the average, so small systematic differences are expected - large ones would mean the features are not "
             "measuring what they claim.</p>")
    srows = []
    has2 = any("sim_mull_v2" in r for r in R["simulator"])
    for r in R["simulator"]:
        row = [r["fmt"], r["n_lands"], f"{r['n_games']:,}", f"{_pct(r['obs_mull'], 1)} [{_pct(r['obs_mull_ci'][0], 1)}, {_pct(r['obs_mull_ci'][1], 1)}]",
               _pct(r["sim_mull"], 1)]
        if has2:
            row.append(_pct(r.get("sim_mull_v2", float("nan")), 1))
        row += [f"{_f(r['obs_lands'], 2)} {_ci(*r['obs_lands_ci'], 2)}", _f(r["sim_lands"], 2)]
        if has2:
            row.append(_f(r.get("sim_lands_v2", float("nan")), 2))
        srows.append(row)
    if srows:
        hdr = ["Format", "Lands", "Games", "Observed mulligan rate (95% CI)", "v1" if has2 else "Simulated"] + (["v2"] if has2 else []) \
            + ["Observed lands in kept 7 (95% CI)", "v1" if has2 else "Simulated"] + (["v2"] if has2 else [])
        P.append(_table(hdr, srows, left={0}))
        if has2:
            P.append("<p class='small mut'>v2 is the calibrated simulator, evaluated for each game with the hand/draw regime of its own queue "
                     "(Bo3 Trad: plain shuffle and the real keep rule; Bo1 Sealed and Premier draft: Arena's hand smoother and the real keep rule). "
                     "The draft format mixes Trad (random) and Premier (smoothed) queues.</p>")
    else:
        P.append("<p>Not enough games per land count to compare.</p>")
    if R.get("realism"):
        P.append(render_realism(R["realism"]))
    # ---------------- audits
    P.append("<h2>Data audits</h2>")
    au = R["audit"]
    P.append("<ul>")
    for k, v in au.items():
        P.append(f"<li><b>{_e(k)}</b>: {_e(v)}</li>")
    P.append("</ul>")
    P.append("<h2>Method notes</h2><ul class='small'>"
             "<li>q(D) = P(win a game | deck D, average pilot, 50/50 play-draw, no mulligans on either side, set-average opponents, time effect at the last observed day).</li>"
             "<li>Penalised logistic regression (L-BFGS-B, analytic gradient); covariates: on-play, mulligans, pilot skill bucket, event, game number, time spline, opponent colours.</li>"
             "<li>Uncertainty: event bootstrap (draws) for deck intervals and scores; approximate Laplace SE for card tables. Draw 0 is the full-data fit.</li>"
             "<li>The deployed model is refitted on all data after this validation; the hold-out numbers describe the same procedure fitted on the earlier 75%.</li></ul>")
    P.append("</main></body></html>")
    return "".join(P)
