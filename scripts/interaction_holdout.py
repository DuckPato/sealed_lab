"""Miniature forward hold-out test of a card x Z layer, Z = play/draw or pilot skill (read-only; single process).

BY = play : Z = on_play - 1/2, first games only (the coin flip decides play/draw, independent of the deck).
BY = skill: Z = the pilot's 17lands win-rate bucket, standardised (no bucket = 0), all games.
Split per format by event start:
first 75% train, last 25% hold-out.  Two penalised logistic models on the training games:

    main : covariates + sum_c beta_c n_c
    pd   : main + sum_c gamma_c n_c Z                          gamma ~ N(0, s^2), s chosen on an inner split of the train part

Reported: hold-out log-loss gain of `pd` over `main` (positive = better) with a cluster bootstrap over events, for all
games and for the Sealed games alone; the spread of gamma.  This is NOT the app's chain (no draft -> Sealed prior, simple
covariates), only a check of whether the signal survives out of sample.

    python scripts/interaction_holdout.py OUT.json play|skill [SET,SET,...]

Session 12 audit (2026-11-12), results in docs/audit_1112/.  Reads the app home (SEALED_LAB_HOME, default the real one) and
never writes to it; OUT goes wherever you point it.  Several of these may run side by side as plain processes (JOBS = 1);
only one spawn pool at a time on this machine.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp
from scipy.optimize import minimize
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sealed_lab import config  # noqa: E402

H = str(config.sets_dir())
OUT, BY = sys.argv[1], sys.argv[2]
assert BY in ("play", "skill"), BY
SETS = sys.argv[3].split(",") if len(sys.argv) > 3 else sorted(os.listdir(H))
TOP, BOTTOM = ("best_on_play", "best_on_draw") if BY == "play" else ("best_for_strong", "best_for_weak")
S_MAIN = 0.15
S_GRID = [0.01, 0.02, 0.04, 0.08]
res = {}


def fit(X, y, prec, x0=None):
    def f(w):
        z = X @ w
        ll = np.logaddexp(0, z) - y * z
        g = X.T @ (expit(z) - y) + prec * w
        return ll.sum() + 0.5 * (prec * w * w).sum(), g
    r = minimize(f, np.zeros(X.shape[1]) if x0 is None else x0, jac=True, method="L-BFGS-B", options=dict(maxiter=300, maxfun=400))
    return r.x


def loss(X, y, w):
    z = X @ w
    return np.logaddexp(0, z) - y * z


for s in SETS:
    g = pl.read_parquet(os.path.join(H, s, "corpus", "games.parquet"),
                        columns=["fmt", "event", "draft_id", "game_number", "deck_key", "draft_time", "on_play", "won", "skill"])
    g = g.filter(pl.col("on_play").is_not_null())
    if BY == "play":
        g = g.filter((pl.col("game_number").is_null()) | (pl.col("game_number") <= 1))
    cards = pl.read_parquet(os.path.join(H, s, "corpus", "cards.parquet"), columns=["card_id", "name", "is_basic", "mana_value"])
    nb = cards.filter(~pl.col("is_basic"))["card_id"].to_numpy()
    n_card_all = int(cards["card_id"].max()) + 1
    col_of = -np.ones(n_card_all, dtype=np.int64)
    col_of[nb] = np.arange(len(nb))
    # events: time split per format
    ev = g.group_by("fmt", "draft_id").agg(pl.col("draft_time").min()).sort("fmt", "draft_time")
    ev = ev.with_columns((pl.int_range(pl.len()).over("fmt") / pl.len().over("fmt")).alias("q"))
    g = g.join(ev.select("draft_id", "q"), on="draft_id")
    dc = pl.read_parquet(os.path.join(H, s, "corpus", "deck_cards.parquet"))
    keys = g["deck_key"].unique()
    dc = dc.filter(pl.col("deck_key").is_in(keys.to_list()))
    kidx = {k: i for i, k in enumerate(keys.to_list())}
    rows = np.array([kidx[k] for k in dc["deck_key"].to_list()])
    cc = col_of[dc["card_id"].to_numpy()]
    ok = cc >= 0
    Dk = sp.csr_matrix((dc["n"].to_numpy()[ok].astype(float), (rows[ok], cc[ok])), shape=(len(kidx), len(nb)))
    gi = np.array([kidx[k] for k in g["deck_key"].to_list()])
    D = Dk[gi]
    p = g["on_play"].to_numpy().astype(float)
    sk = g["skill"].to_numpy().astype(float)
    miss = np.isnan(sk)
    sk = np.where(miss, np.nanmean(sk), sk) - 0.55
    fmt = g["fmt"].to_numpy()
    evn = g["event"].to_list()
    evs = sorted(set(evn))
    C = np.column_stack([np.ones(len(p)), p - 0.5, sk, sk ** 2, miss.astype(float)] + [(np.array(evn) == e).astype(float) for e in evs[1:]])
    # play/draw advantage may differ by format: on_play x Sealed
    C = np.column_stack([C, (p - 0.5) * (fmt != "draft")])
    Xm = sp.hstack([sp.csr_matrix(C), D], format="csr")
    z = p - 0.5 if BY == "play" else np.where(miss, 0.0, sk - sk[~miss].mean()) / sk[~miss].std()
    Xp = sp.hstack([Xm, sp.diags(z) @ D], format="csr")
    y = g["won"].to_numpy().astype(float)
    q = g["q"].to_numpy()
    tr, ho = q < 0.75, q >= 0.75
    inner_tr, inner_va = q < 0.55, (q >= 0.55) & (q < 0.75)
    nc, k = C.shape[1], len(nb)

    def prec(s_int):
        return np.r_[np.full(nc, 1e-4), np.full(k, 1 / S_MAIN ** 2), np.full(k, 1 / s_int ** 2)]

    pm = np.r_[np.full(nc, 1e-4), np.full(k, 1 / S_MAIN ** 2)]
    w_m_in = fit(Xm[inner_tr], y[inner_tr], pm)
    base = loss(Xm[inner_va], y[inner_va], w_m_in).mean()
    inner = {}
    for si in S_GRID:
        w = fit(Xp[inner_tr], y[inner_tr], prec(si), x0=np.r_[w_m_in, np.zeros(k)])
        inner[si] = float(base - loss(Xp[inner_va], y[inner_va], w).mean())
    s_best = max(inner, key=inner.get)
    w_m = fit(Xm[tr], y[tr], pm, x0=w_m_in)
    w_p = fit(Xp[tr], y[tr], prec(s_best), x0=np.r_[w_m, np.zeros(k)])
    d = loss(Xm[ho], y[ho], w_m) - loss(Xp[ho], y[ho], w_p)        # per game, positive = pd better
    evh = g["draft_id"].cast(pl.Categorical).to_physical().to_numpy()[ho]
    fh = fmt[ho]
    rng = np.random.default_rng(1)

    def ci(mask):
        dd, ee = d[mask], evh[mask]
        if len(dd) < 200:
            return None
        o = np.argsort(ee, kind="stable")
        e = ee[o]
        cuts = np.flatnonzero(np.r_[True, e[1:] != e[:-1]])
        sums = np.add.reduceat(dd[o], cuts)
        cnt = np.diff(np.r_[cuts, len(e)])
        idx = rng.integers(0, len(sums), size=(1000, len(sums)))
        bs = sums[idx].sum(1) / cnt[idx].sum(1)
        return dict(n=int(len(dd)), events=int(len(sums)), gain=float(dd.mean()), lo=float(np.quantile(bs, 0.025)), hi=float(np.quantile(bs, 0.975)),
                    se=float(bs.std()))

    gam = w_p[nc + k:]
    expo = np.asarray(D[tr].sum(0)).ravel()
    big = expo > np.quantile(expo, 0.3)
    names = cards.filter(~pl.col("is_basic")).sort("card_id")["name"].to_list()
    o = np.argsort(gam)
    res[s] = dict(n_train=int(tr.sum()), n_holdout=int(ho.sum()), s_best=s_best, inner=inner, all=ci(np.ones(ho.sum(), bool)),
                  sealed=ci(fh != "draft"), bo1=ci(fh == "bo1"), bo3=ci(fh == "bo3"), draft=ci(fh == "draft"),
                  gamma_sd_pp=float(24.75 * gam[big].std()), gamma_max_pp=float(24.75 * np.abs(gam[big]).max()),
                  **{TOP: [(names[i], round(24.75 * float(gam[i]), 2)) for i in o[::-1][:6]],
                     BOTTOM: [(names[i], round(24.75 * float(gam[i]), 2)) for i in o[:6]]})
    print(s, {k2: (v if not isinstance(v, dict) else {a: round(b, 6) for a, b in v.items()}) for k2, v in res[s].items()
              if k2 in ("s_best", "all", "sealed", "gamma_sd_pp")}, flush=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
