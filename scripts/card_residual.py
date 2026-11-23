"""Sealed residual of one card beyond the model: logit P(win) = pilot covariates + s * model logit of the deck + gamma * copies.

    python scripts/card_residual.py SET CARD_ID   (in-sample; the real model in the normal home, read-only)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, polars as pl, scipy.sparse as sp
from sealed_lab import config
from sealed_lab.model import design, glm
from sealed_lab.model.artifact import SetModel

sc, cid = sys.argv[1], int(sys.argv[2])
m = SetModel.load(sc)
cd = config.corpus_dir(sc)
games = pl.read_parquet(cd / "games.parquet")
dc = pl.read_parquet(cd / "deck_cards.parquet")
decks = pl.read_parquet(cd / "decks.parquet").filter(pl.col("n_cards") == 40)
rng = np.random.default_rng(0)
for fmt in ("bo1", "bo3"):
    g = games.filter(pl.col("fmt") == fmt).join(decks.select("deck_key"), on="deck_key")
    keys = g["deck_key"].unique()
    with_c = set(dc.filter((pl.col("card_id") == cid) & pl.col("deck_key").is_in(keys))["deck_key"].to_list())
    others = [k for k in keys.to_list() if k not in with_c]
    samp = set(rng.choice(others, min(len(others), 6000), replace=False).tolist()) | with_c
    g = g.filter(pl.col("deck_key").is_in(list(samp)))
    sub = dc.filter(pl.col("deck_key").is_in(list(samp)))
    L, ncop = {}, {}
    for (k,), part in sub.partition_by("deck_key", as_dict=True).items():
        deck = {int(c): int(n) for c, n in part.select("card_id", "n").iter_rows()}
        L[k] = m.logit(deck, fmt)
        ncop[k] = deck.get(cid, 0)
    g = g.filter(pl.col("deck_key").is_in(list(L)))
    dk = g["deck_key"].to_list()
    off = np.array([L[k] for k in dk]); cop = np.array([ncop[k] for k in dk], float)
    enc = design.CovEncoder.fit(g)
    C = sp.csr_matrix(enc.transform(g))
    y = g["won"].cast(pl.Float64).to_numpy()
    X = sp.hstack([C, sp.csr_matrix(np.column_stack([off - np.median(off), cop]))], format="csr")
    sig = np.concatenate([enc.sigma, [10.0, 10.0]])
    prob = glm.Problem(y=y, C=X)
    fit = lambda w=None: glm.fit(prob, np.zeros(len(sig)), sig, w=w, maxiter=400).coef[-2:]
    s, gam = fit()
    grp = np.unique(np.array(g["draft_id"].to_list()), return_inverse=True)[1]; G = grp.max() + 1
    mult = rng.multinomial(G, np.full(G, 1 / G), size=200).astype(float)
    bs = np.array([fit(mult[i][grp])[1] for i in range(200)])
    print(f"{fmt}: {len(with_c)} decks with the card, {int((cop>0).sum())} games; slope on model logit {s:.2f}; "
          f"residual per copy {25*gam:+.2f} pp [{25*np.percentile(bs,2.5):+.2f}, {25*np.percentile(bs,97.5):+.2f}] "
          f"(model's own value per copy {25*m.cards.filter(pl.col('card_id')==cid)['beta_'+fmt][0]:+.2f} pp)")
