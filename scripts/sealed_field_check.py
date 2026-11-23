"""Does the matchup layer's value against the SEALED field (not only against one opponent) show up in Sealed results?

Indicative only: the deployed model has seen these games (the matchup layer is fitted mostly on draft games, the card values
almost entirely on draft games, so the in-sample advantage is small but not zero).

    won = covariates + b1 * x1 + b2 * x2 + b3 * x3            (linear probability model, clusters = events)
    x1 = deck logit of the deployed model (what the builder maximises)
    x2 = sum_c n_c * mean over the format's opponents of the centred matchup effect   (the deck against the Sealed field)
    x3 = sum_c n_c * centred effect against THIS game's opponent, minus x2            (the opponent-specific part)
Coefficients are divided by mean p(1-p): 1 = realised one for one.

    python scripts/sealed_field_check.py OUT.json JOBS MAX_DECKS [SET,SET,...]

Session 12 audit (2026-11-12), results in docs/audit_1112/.  Reads the app home (SEALED_LAB_HOME, default the real one) and
never writes to it; OUT goes wherever you point it.  Several of these may run side by side as plain processes (JOBS = 1);
only one spawn pool at a time on this machine.
"""
import json
import multiprocessing
import os
import sys
from pathlib import Path
import time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _all_sets() -> list[str]:
    from sealed_lab import config

    return sorted(p.name for p in config.sets_dir().iterdir() if p.is_dir())


def task(args):
    set_code, fmt, max_decks = args
    os.environ["POLARS_MAX_THREADS"] = "2"
    import numpy as np
    import polars as pl

    from sealed_lab import config
    from sealed_lab.deck import builder

    model = builder.load_model(set_code)
    if fmt not in model.formats or model.opp is None or fmt not in model.opp_formats:
        return dict(set=set_code, fmt=fmt, error="no matchup layer in this format")
    ctx = builder.Ctx(model, fmt)
    opm = model.opp
    cd = config.corpus_dir(set_code)
    g = pl.read_parquet(cd / "games.parquet", columns=["fmt", "draft_id", "deck_key", "on_play", "won", "skill", "opp_colors",
                                                       "draft_time"]).filter(pl.col("fmt") == fmt)
    decks = pl.read_parquet(cd / "decks.parquet", columns=["deck_key", "n_cards"])
    g = g.join(decks, on="deck_key").filter(pl.col("n_cards") == 40)
    # field = the format's opponents (all games)
    c = g.group_by("opp_colors").len()
    opps = c["opp_colors"].to_list()
    share = (c["len"] / c["len"].sum()).to_numpy()
    E = np.stack([opm.effect(o, draws=False) for o in opps])               # (n_opp, n_cards)
    m_field = share @ E
    keys = g.group_by("deck_key").len().sort("len", descending=True)["deck_key"].to_list()
    if len(keys) > max_decks:
        rng = np.random.default_rng(5)
        keys = list(rng.choice(np.array(keys), size=max_decks, replace=False))
    g = g.filter(pl.col("deck_key").is_in(keys))
    dcs = pl.read_parquet(cd / "deck_cards.parquet").filter(pl.col("deck_key").is_in(keys))
    x1, x2, per_opp = {}, {}, {}
    for k, v in dcs.group_by("deck_key"):
        ids = np.array(v["card_id"].to_list(), dtype=np.int64)
        cnt = np.array(v["n"].to_list(), dtype=float)
        deck = dict(zip(ids.tolist(), [int(n) for n in cnt]))
        x1[k[0]] = ctx.logit(deck) - ctx.ref
        nb = ~ctx.is_basic[ids]
        x2[k[0]] = float(m_field[ids[nb]] @ cnt[nb])
        per_opp[k[0]] = E[:, ids[nb]] @ cnt[nb]
    oi = {o: i for i, o in enumerate(opps)}
    dk = g["deck_key"].to_list()
    oc = g["opp_colors"].to_list()
    a1 = np.array([x1[k] for k in dk])
    a2 = np.array([x2[k] for k in dk])
    a3 = np.array([per_opp[k][oi[o]] for k, o in zip(dk, oc)]) - a2
    sk = g["skill"].to_numpy().astype(float)
    miss = np.isnan(sk)
    sk = np.where(miss, np.nanmean(sk), sk)
    y = g["won"].to_numpy().astype(float)
    X = np.column_stack([np.ones(len(y)), sk, sk ** 2, miss.astype(float), g["on_play"].fill_null(False).to_numpy().astype(float), a1, a2, a3])
    ev = g["draft_id"].cast(pl.Categorical).to_physical().to_numpy()

    def fit(X):
        XtX = X.T @ X
        b = np.linalg.solve(XtX, X.T @ y)
        r = y - X @ b
        o = np.argsort(ev, kind="stable")
        e = ev[o]
        cuts = np.flatnonzero(np.r_[True, e[1:] != e[:-1]])
        S = np.add.reduceat(X[o] * r[o, None], cuts, axis=0)
        V = np.linalg.solve(XtX, np.linalg.solve(XtX, S.T @ S).T)
        return b, np.sqrt(np.diag(V))

    b, se = fit(X)
    pq = float(y.mean() * (1 - y.mean()))
    # the later 25% of events only (closer to "current" field), same regression
    return dict(set=set_code, fmt=fmt, n_games=int(len(y)), n_decks=len(keys), pq=pq,
                b1=float(b[5] / pq), se1=float(se[5] / pq), b2=float(b[6] / pq), se2=float(se[6] / pq), b3=float(b[7] / pq), se3=float(se[7] / pq),
                sd_x1=float(a1.std()), sd_x2=float(a2.std()), sd_x3=float(a3.std()), corr_x1_x2=float(np.corrcoef(a1, a2)[0, 1]))


if __name__ == "__main__":
    out, jobs, max_decks = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    sets = sys.argv[4].split(",") if len(sys.argv) > 4 else _all_sets()
    tasks = [(s, f, max_decks) for s in sets for f in ("bo3", "bo1")]
    res = []
    t0 = time.time()
    if jobs <= 1:
        for t in tasks:
            res.append(task(t))
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res[-1].items()}, round(time.time() - t0), flush=True)
            with open(out, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=1)
    else:
        with ProcessPoolExecutor(jobs, mp_context=multiprocessing.get_context("spawn")) as ex:
            for r in ex.map(task, tasks, chunksize=1):
                res.append(r)
                print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, round(time.time() - t0), flush=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
