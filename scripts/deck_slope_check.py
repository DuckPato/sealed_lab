"""Across-deck slope of the deployed deck score on real Sealed games, by a proper logistic fit with skill-bucket dummies
(cross-check of the linear-probability estimate in field_check.py).

    python scripts/deck_slope_check.py SET:fmt [SET:fmt ...]

Session 12 audit (2026-11-12), results in docs/audit_1112/.  Reads the app home (SEALED_LAB_HOME, default the real one) and
never writes to it; OUT goes wherever you point it.  Several of these may run side by side as plain processes (JOBS = 1);
only one spawn pool at a time on this machine.
"""
import os, sys
import numpy as np, polars as pl
from scipy.optimize import minimize
from scipy.special import expit
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sealed_lab import config
from sealed_lab.deck import builder
from sealed_lab.model import design
for set_code, fmt in [a.split(":") for a in sys.argv[1:]]:
    model = builder.load_model(set_code); ctx = builder.Ctx(model, fmt)
    cd = config.corpus_dir(set_code)
    g = pl.read_parquet(cd / "games.parquet", columns=["fmt", "draft_id", "deck_key", "on_play", "won", "skill", "opp_colors", "game_number", "draft_time"]).filter(pl.col("fmt") == fmt)
    g = g.join(pl.read_parquet(cd / "decks.parquet", columns=["deck_key", "n_cards"]), on="deck_key").filter(pl.col("n_cards") == 40)
    keys = g["deck_key"].unique().to_list()
    rng = np.random.default_rng(5)
    if len(keys) > 3500: keys = list(rng.choice(np.array(keys), 3500, replace=False))
    g = g.filter(pl.col("deck_key").is_in(keys))
    dcs = pl.read_parquet(cd / "deck_cards.parquet").filter(pl.col("deck_key").is_in(keys))
    x1 = {k[0]: ctx.logit(dict(zip([int(i) for i in v["card_id"].to_list()], [int(n) for n in v["n"].to_list()]))) - ctx.ref for k, v in dcs.group_by("deck_key")}
    x = np.array([x1[k] for k in g["deck_key"].to_list()])
    y = g["won"].to_numpy().astype(float)
    sk = g["skill"].to_numpy().astype(float); miss = np.isnan(sk)
    b = np.where(miss, -1, np.round(np.where(miss, 0, sk) / 0.02)).astype(int)
    S = np.column_stack([(b == v).astype(float) for v in np.unique(b)[1:]])
    pairs = np.array([design.norm_pair(o) for o in g["opp_colors"].to_list()])
    P = np.column_stack([(pairs == p).astype(float) for p in design.PAIRS])
    base = [np.ones(len(y)), g["on_play"].fill_null(False).to_numpy().astype(float), (g["game_number"].fill_null(1).to_numpy() >= 2).astype(float)]
    ev = g["draft_id"].cast(pl.Categorical).to_physical().to_numpy()
    def logit_fit(X):
        f = lambda w: ((np.logaddexp(0, X @ w) - y * (X @ w)).sum() + 1e-6 * w @ w, X.T @ (expit(X @ w) - y) + 2e-6 * w)
        w = minimize(f, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B", options=dict(maxiter=500)).x
        p = expit(X @ w); Hm = (X * (p * (1 - p))[:, None]).T @ X + 2e-6 * np.eye(X.shape[1])
        o = np.argsort(ev, kind="stable"); e = ev[o]; cuts = np.flatnonzero(np.r_[True, e[1:] != e[:-1]])
        Sg = np.add.reduceat((X * (y - p)[:, None])[o], cuts, axis=0)
        V = np.linalg.solve(Hm, np.linalg.solve(Hm, Sg.T @ Sg).T)
        return w, np.sqrt(np.diag(V))
    out = []
    for name, cols in (("no skill control", base), ("skill buckets", base + [S]), ("skill buckets + opponent pairs", base + [S, P])):
        X = np.column_stack(cols + [x]); w, se = logit_fit(X)
        out.append(f"{name}: {w[-1]:.2f} ({se[-1]:.2f})")
    print(set_code, fmt, f"games {len(y)} decks {len(keys)} sd(x) {x.std():.3f} corr(x, skill) {np.corrcoef(x[~miss], sk[~miss])[0,1]:.2f} |", " | ".join(out), flush=True)
