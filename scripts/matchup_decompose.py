"""How much of the deployed matchup layer's deck-vs-opponent effect is an opponent MAIN effect (the same for every deck) and how
much is a true deck x opponent interaction (the only part that can change which deck or sideboard card is better)?

M[D, o] = sum_c n_c * centred effect of card c against opponent colours o;  M' = M minus its mean over the format's opponents
(per deck);  a_o = mean of M' over the decks that met o;  I = M' - a_o.
Regression on the format's games (linear probability, clusters = events, in-sample for the layer):
    won = skill, play/draw, opponent colour-pair dummies + b_a * a_o + b_i * I      (coefficients divided by p(1-p))

    python scripts/matchup_decompose.py OUT.json

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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sealed_lab import config  # noqa: E402
from sealed_lab.model import design  # noqa: E402
from sealed_lab.model.artifact import SetModel  # noqa: E402

H = str(config.sets_dir())
OUT = sys.argv[1]
res = []
for s in sorted(os.listdir(H)):
    model = SetModel.load(s)
    opm = model.opp
    cards = model.cards.sort("card_id")
    is_basic = cards["is_basic"].to_numpy()
    n_cards = len(is_basic)
    for fmt in ("bo3", "bo1"):
        if opm is None or fmt not in model.opp_formats:
            continue
        g = pl.read_parquet(os.path.join(H, s, "corpus", "games.parquet"),
                            columns=["fmt", "draft_id", "deck_key", "on_play", "won", "skill", "opp_colors"]).filter(pl.col("fmt") == fmt)
        decks = pl.read_parquet(os.path.join(H, s, "corpus", "decks.parquet"), columns=["deck_key", "n_cards"])
        g = g.join(decks, on="deck_key").filter(pl.col("n_cards") == 40)
        keys = g["deck_key"].unique().to_list()
        kidx = {k: i for i, k in enumerate(keys)}
        dc = pl.read_parquet(os.path.join(H, s, "corpus", "deck_cards.parquet")).filter(pl.col("deck_key").is_in(keys))
        cid = dc["card_id"].to_numpy()
        ok = ~is_basic[cid]
        Dk = sp.csr_matrix((dc["n"].to_numpy()[ok].astype(float), (np.array([kidx[k] for k in dc["deck_key"].to_list()])[ok], cid[ok])),
                           shape=(len(keys), n_cards))
        c = g.group_by("opp_colors").len()
        opps = c["opp_colors"].to_list()
        share = (c["len"] / c["len"].sum()).to_numpy()
        E = np.stack([opm.effect(o, draws=False) for o in opps])           # (n_opp, n_cards)
        M = np.asarray(Dk @ E.T)                                           # (n_decks, n_opp)
        Mc = M - (M * share[None, :]).sum(1, keepdims=True)
        oi = {o: i for i, o in enumerate(opps)}
        di = np.array([kidx[k] for k in g["deck_key"].to_list()])
        gi = np.array([oi[o] for o in g["opp_colors"].to_list()])
        m = Mc[di, gi]
        a_o = np.zeros(len(opps))
        np.add.at(a_o, gi, m)
        a_o = a_o / np.maximum(np.bincount(gi, minlength=len(opps)), 1)
        a = a_o[gi]
        inter = m - a
        pairs = np.array([design.norm_pair(o) for o in g["opp_colors"].to_list()])
        sk = g["skill"].to_numpy().astype(float)
        miss = np.isnan(sk)
        sk = np.where(miss, np.nanmean(sk), sk)
        cols = [np.ones(len(m)), sk, sk ** 2, miss.astype(float), g["on_play"].fill_null(False).to_numpy().astype(float)]
        cols += [(pairs == p).astype(float) for p in design.PAIRS]
        X = np.column_stack(cols + [a, inter])
        y = g["won"].to_numpy().astype(float)
        ev = g["draft_id"].cast(pl.Categorical).to_physical().to_numpy()
        XtX = X.T @ X + 1e-9 * np.eye(X.shape[1])
        b = np.linalg.solve(XtX, X.T @ y)
        r = y - X @ b
        o = np.argsort(ev, kind="stable")
        e = ev[o]
        cuts = np.flatnonzero(np.r_[True, e[1:] != e[:-1]])
        S = np.add.reduceat(X[o] * r[o, None], cuts, axis=0)
        V = np.linalg.solve(XtX, np.linalg.solve(XtX, S.T @ S).T)
        se = np.sqrt(np.diag(V))
        pq = float(y.mean() * (1 - y.mean()))
        other = pairs == "other"
        row = dict(set=s, fmt=fmt, mode=opm.mode, n_games=int(len(y)), share_other_opp=float(other.mean()),
                   sd_total_pp=24.75 * float(m.std()), sd_oppmain_pp=24.75 * float(a.std()), sd_inter_pp=24.75 * float(inter.std()),
                   share_var_oppmain=float(a.var() / m.var()),
                   b_a=float(b[-2] / pq), se_a=float(se[-2] / pq), b_i=float(b[-1] / pq), se_i=float(se[-1] / pq))
        res.append(row)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)
for f in ("bo1", "bo3"):
    x = np.array([[r["b_i"], r["se_i"], r["b_a"], r["se_a"]] for r in res if r["fmt"] == f])
    w = 1 / x[:, 1] ** 2
    w2 = 1 / x[:, 3] ** 2
    print(f, "pooled interaction slope %.2f (se %.2f); opponent-main slope %.2f (se %.2f)" % ((w * x[:, 0]).sum() / w.sum(), 1 / np.sqrt(w.sum()),
                                                                                         (w2 * x[:, 2]).sum() / w2.sum(), 1 / np.sqrt(w2.sum())))
