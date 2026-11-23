"""Do two cards together win more than the model predicts? (owner-named pairs, checked one by one)

    python scripts/pair_check.py SET "Card A" "Card B" ["Card C" "Card D" ...]

Per format (draft, bo1, bo3 games; in-sample, the real model in the normal home, read-only):
    logit P(win) = pilot covariates + s * model logit of the deck + g_A * copies(A) + g_B * copies(B) + d * [A and B both in the deck]
g_A / g_B absorb any misvaluation of each card on its own, so d is the combination's own effect beyond the model ("synergy").
Draft decks are scored with the bo1 model (s absorbs the scale). Decks with A or B are all used, plus a random sample of others;
95% intervals: cluster bootstrap over events. Effects in pp of game win rate at 50% (coef / 4).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import scipy.sparse as sp  # noqa: E402

from sealed_lab import config  # noqa: E402
from sealed_lab.model import design, glm  # noqa: E402
from sealed_lab.model.artifact import SetModel  # noqa: E402

N_OTHER = 6000
N_ONE = 6000
BOOT = 200


def main(argv: list[str]) -> int:
    sc, names = argv[0].upper(), argv[1:]
    pairs = list(zip(names[0::2], names[1::2]))
    m = SetModel.load(sc)
    cards = m.cards

    def cid(name: str) -> int:
        hit = cards.filter(pl.col("name").str.to_lowercase().str.contains(name.lower(), literal=True))
        if hit.height != 1:
            raise SystemExit(f"{name!r}: {hit.height} matches {hit['name'].to_list()[:5]}")
        return int(hit["card_id"][0])

    cd = config.corpus_dir(sc)
    games = pl.read_parquet(cd / "games.parquet")
    dc = pl.read_parquet(cd / "deck_cards.parquet")
    d40 = pl.read_parquet(cd / "decks.parquet").filter(pl.col("n_cards") == 40).select("deck_key")
    rng = np.random.default_rng(0)
    cache: dict[tuple, float] = {}
    for a_name, b_name in pairs:
        a, b = cid(a_name), cid(b_name)
        print(f"\n{cards.filter(pl.col('card_id') == a)['name'][0]} + {cards.filter(pl.col('card_id') == b)['name'][0]}")
        for fmt in ("draft", "bo1", "bo3"):
            g = games.filter(pl.col("fmt") == fmt).join(d40, on="deck_key")
            keys = set(g["deck_key"].unique().to_list())
            has = dc.filter(pl.col("card_id").is_in([a, b]) & pl.col("deck_key").is_in(list(keys)))
            ca = dict(zip(*has.filter(pl.col("card_id") == a).select("deck_key", "n").to_dict(as_series=False).values()))
            cb = dict(zip(*has.filter(pl.col("card_id") == b).select("deck_key", "n").to_dict(as_series=False).values()))
            both = set(ca) & set(cb)
            one = [k for k in set(ca) | set(cb) if k not in both]
            other = [k for k in keys if k not in ca and k not in cb]
            samp = set(both)
            samp |= set(rng.choice(one, min(len(one), N_ONE), replace=False).tolist()) if one else set()
            samp |= set(rng.choice(other, min(len(other), N_OTHER), replace=False).tolist())
            g = g.filter(pl.col("deck_key").is_in(list(samp)))
            mfmt = "bo1" if fmt == "draft" else fmt
            need = [k for k in samp if (mfmt, k) not in cache]
            for (k,), part in dc.filter(pl.col("deck_key").is_in(need)).partition_by("deck_key", as_dict=True).items():
                cache[(mfmt, k)] = m.logit({int(c): int(n) for c, n in part.select("card_id", "n").iter_rows()}, mfmt)
            dk = g["deck_key"].to_list()
            L = np.array([cache[(mfmt, k)] for k in dk])
            xa = np.array([ca.get(k, 0) for k in dk], float)
            xb = np.array([cb.get(k, 0) for k in dk], float)
            xab = ((xa > 0) & (xb > 0)).astype(float)
            n_both_games = int(xab.sum())
            if n_both_games < 30:
                print(f"  {fmt:5s}: only {n_both_games} games with both cards - too few to say anything")
                continue
            enc = design.CovEncoder.fit(g)
            C = sp.csr_matrix(enc.transform(g))
            y = g["won"].cast(pl.Float64).to_numpy()
            X = sp.hstack([C, sp.csr_matrix(np.column_stack([L - np.median(L), xa, xb, xab]))], format="csr")
            sig = np.concatenate([enc.sigma, [10.0] * 4])
            prob = glm.Problem(y=y, C=X)

            def fit(w=None):
                return glm.fit(prob, np.zeros(len(sig)), sig, w=w, maxiter=400).coef[-4:]

            coef = fit()
            grp = np.unique(np.array(g["draft_id"].to_list()), return_inverse=True)[1]
            G = int(grp.max()) + 1
            mult = rng.multinomial(G, np.full(G, 1.0 / G), size=BOOT).astype(float)
            bs = np.array([fit(mult[i][grp]) for i in range(BOOT)])

            def fmt_ci(j: int) -> str:
                return f"{25 * coef[j]:+.2f} pp [{25 * np.percentile(bs[:, j], 2.5):+.2f}, {25 * np.percentile(bs[:, j], 97.5):+.2f}]"

            print(f"  {fmt:5s}: {len(both)} decks / {n_both_games} games with both | together, beyond the model: {fmt_ci(3)} | "
                  f"own residual per copy: A {fmt_ci(1)}, B {fmt_ci(2)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
