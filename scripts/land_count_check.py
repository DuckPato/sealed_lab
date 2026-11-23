"""16 against 17 lands in real 40-card Sealed decks: share, raw win rate, mulligan rate and the skill-adjusted difference
(linear probability model with skill, skill^2 and play/draw; clusters = events), per set and pooled by inverse variance.

    python scripts/land_count_check.py

Session 12 audit (2026-11-12), results in docs/audit_1112/.  Reads the app home (SEALED_LAB_HOME, default the real one) and
never writes to it; OUT goes wherever you point it.  Several of these may run side by side as plain processes (JOBS = 1);
only one spawn pool at a time on this machine.
"""
import os, sys, zlib, numpy as np, polars as pl
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sealed_lab import config  # noqa: E402
H = str(config.sets_dir())
def lpm(X, y, g):
    XtX = X.T @ X; b = np.linalg.solve(XtX, X.T @ y); r = y - X @ b
    o = np.argsort(g, kind="stable"); gs = g[o]; cuts = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1]])
    S = np.add.reduceat(X[o] * r[o, None], cuts, axis=0); V = np.linalg.solve(XtX, np.linalg.solve(XtX, S.T @ S).T)
    return b, np.sqrt(np.diag(V))
pool = {"bo1": [], "bo3": []}
for s in sorted(os.listdir(H)):
    g = pl.read_parquet(os.path.join(H, s, "corpus", "games.parquet"), columns=["fmt", "draft_id", "deck_key", "on_play", "won", "skill", "num_mulligans"])
    d = pl.read_parquet(os.path.join(H, s, "corpus", "decks.parquet"))
    g = g.join(d, on="deck_key").filter((pl.col("n_cards") == 40) & pl.col("skill").is_not_null())
    line = [s]
    for fmt in ("bo1", "bo3"):
        x = g.filter((pl.col("fmt") == fmt) & pl.col("n_lands").is_in([16, 17]))
        sh = g.filter(pl.col("fmt") == fmt).group_by("n_lands").len()
        tot = sh["len"].sum(); shd = {k: v / tot for k, v in sh.iter_rows()}
        sk = x["skill"].to_numpy().astype(float); y = x["won"].to_numpy().astype(float)
        X = np.column_stack([np.ones(len(y)), sk, sk**2, x["on_play"].to_numpy().astype(float), (x["n_lands"] == 16).to_numpy().astype(float)])
        gid = np.array([zlib.crc32(t.encode()) for t in x["draft_id"].to_list()])
        b, se = lpm(X, y, gid)
        m = x.group_by("n_lands").agg(pl.col("won").mean(), (pl.col("num_mulligans") > 0).mean().alias("mull")).sort("n_lands")
        line.append(f"{fmt}: share16 {100*shd.get(16,0):.0f}% share17 {100*shd.get(17,0):.0f}% | WR16 {100*m['won'][0]:.1f} WR17 {100*m['won'][1]:.1f} | mull16 {100*m['mull'][0]:.1f} mull17 {100*m['mull'][1]:.1f} | 16-17 skill-adj {100*b[4]:+.2f} ({100*se[4]:.2f})")
        pool[fmt].append((100*b[4], 100*se[4]))
    print(" || ".join(line))
for fmt, v in pool.items():
    b = np.array([a for a, _ in v]); se = np.array([c for _, c in v]); w = 1/se**2
    print(fmt, "pooled 16 minus 17 lands, skill-adjusted: %+.2f pp (se %.2f)" % ((w*b).sum()/w.sum(), 1/np.sqrt(w.sum())))
