"""Composition diagnostic: does the model already capture "golden rules" (creature count, 2-drops, lands)?

    python scripts/composition_check.py SET TRAIN_HOME OUTDIR [--fmt bo3 bo1] [--boot 200]

TRAIN_HOME is a home holding a model fitted only on events before the hold-out cut (the one scripts/deviation_cost.py
writes under OUTDIR/home). For every held-out game (last 25% of events, the report's split) we take the train-only model's
logit of the registered deck, L_h, and fit on the held-out games

    logit P(win) = pilot covariates + s * (L_h - ref) + sum_k c_k * [deck in bin k of the composition variable]

with the modal bin as reference. If the model already prices the composition, every c_k is ~0 (the deck's predicted
strength explains the result); a hump-shaped pattern (bins far from the mode negative) is a sweet spot the model misses.
Variables: creature count, 2-drop count, land count (control: lands already have a squared term), and the creature count
within speed groups (tertiles of mean spell mana value: fast / mid / slow). 95% intervals: cluster bootstrap over events.
Effects in pp of game win rate at 50% (c / 4). Writes OUTDIR/composition_SET.json and prints tables. Read-only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _bins(x: np.ndarray, edges: list[int]) -> tuple[np.ndarray, list[str]]:
    """Integer bin index for edges [e0, e1, ...]: "<=e0", e0+1 .. e_last-1 singly, ">=e_last"."""
    lo, hi = edges[0], edges[-1]
    labels = [f"<={lo}"] + [str(v) for v in range(lo + 1, hi)] + [f">={hi}"]
    idx = np.clip(x, lo, hi) - lo
    return idx.astype(int), labels


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("set")
    ap.add_argument("train_home")
    ap.add_argument("outdir")
    ap.add_argument("--fmt", nargs="+", default=["bo3", "bo1"])
    ap.add_argument("--boot", type=int, default=200)
    args = ap.parse_args(argv)

    import polars as pl
    import scipy.sparse as sp

    from sealed_lab import config
    from sealed_lab.model import design, glm, validate

    sc = args.set.upper()
    real = config.corpus_dir(sc)
    games_all = pl.read_parquet(real / "games.parquet")
    dcards = pl.read_parquet(real / "deck_cards.parquet")
    decks = pl.read_parquet(real / "decks.parquet")
    target = next(f for f in args.fmt if games_all.filter(pl.col("fmt") == f).height)
    _, t_cut = validate.time_split(games_all.filter(pl.col("fmt") == target))

    os.environ["SEALED_LAB_HOME"] = str(Path(args.train_home).resolve())
    from sealed_lab.model.artifact import SetModel

    model = SetModel.load(sc)
    cards = model.cards
    is_land = dict(zip(cards["card_id"].to_list(), cards["is_land"].to_list()))
    is_cre = dict(zip(cards["card_id"].to_list(), cards["is_creature"].to_list()))
    mv = dict(zip(cards["card_id"].to_list(), cards["mana_value"].fill_null(0).to_list()))

    out = {"set": sc, "cut": str(t_cut), "formats": {}}
    for fmt in args.fmt:
        g = games_all.filter(pl.col("fmt") == fmt)
        if not g.height:
            continue
        starts = design.event_starts(g)
        ho_ids = set(starts.filter(pl.col("start") >= t_cut)["draft_id"].to_list())
        tr_ids = set(starts.filter(pl.col("start") < t_cut)["draft_id"].to_list())
        g_ho = (g.filter(pl.col("draft_id").is_in(ho_ids)).join(decks.select("deck_key", "n_cards"), on="deck_key", how="left")
                .filter(pl.col("n_cards") == 40))
        keys = g_ho["deck_key"].unique().to_list()
        dc = dcards.filter(pl.col("deck_key").is_in(keys))
        t0 = time.time()
        comp, Ld = {}, {}
        for (key,), part in dc.partition_by("deck_key", as_dict=True).items():
            deck = {int(c): int(n) for c, n in part.select("card_id", "n").iter_rows()}
            spells = [(c, n) for c, n in deck.items() if not is_land.get(c, False)]
            n_sp = sum(n for _, n in spells)
            comp[key] = (sum(n for c, n in spells if is_cre.get(c, False)),
                         sum(n for c, n in spells if mv.get(c, 0) == 2),
                         sum(n for c, n in deck.items() if is_land.get(c, False)),
                         sum(mv.get(c, 0) * n for c, n in spells) / max(n_sp, 1))
            Ld[key] = model.logit(deck, fmt)
        print(f"{fmt}: {len(Ld)} held-out decks scored in {time.time() - t0:.0f}s, {g_ho.height} games")
        g_ho = g_ho.filter(pl.col("deck_key").is_in(list(Ld)))
        dk = g_ho["deck_key"].to_list()
        L = np.array([Ld[k] for k in dk])
        cre = np.array([comp[k][0] for k in dk])
        two = np.array([comp[k][1] for k in dk])
        lands = np.array([comp[k][2] for k in dk])
        mmv = np.array([comp[k][3] for k in dk])
        y = g_ho["won"].cast(pl.Float64).to_numpy()
        enc = design.CovEncoder.fit(g.filter(pl.col("draft_id").is_in(tr_ids)))
        C = sp.csr_matrix(enc.transform(g_ho, clamp_time=True))
        ref = float(np.median(L))
        groups = np.unique(np.array(g_ho["draft_id"].to_list()), return_inverse=True)[1]
        G = int(groups.max()) + 1
        rng = np.random.default_rng(0)
        mult = rng.multinomial(G, np.full(G, 1.0 / G), size=args.boot).astype(float)

        def run(D: np.ndarray, names: list[str]) -> dict:
            X = sp.hstack([C, sp.csr_matrix(np.column_stack([L - ref, D]))], format="csr")
            sig = np.concatenate([enc.sigma, [10.0], np.full(D.shape[1], 10.0)])
            prob = glm.Problem(y=y, C=X)
            k = D.shape[1] + 1

            def fit(w=None):
                return glm.fit(prob, np.zeros(len(sig)), sig, w=w, maxiter=400).coef

            coef = fit()
            p = 1 / (1 + np.exp(-(X @ coef)))
            ll1 = float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
            X0 = sp.hstack([C, sp.csr_matrix((L - ref)[:, None])], format="csr")
            c0 = glm.fit(glm.Problem(y=y, C=X0), np.zeros(len(sig) - D.shape[1]), sig[:-D.shape[1]], maxiter=400).coef
            p0 = 1 / (1 + np.exp(-(X0 @ c0)))
            ll0 = float(np.sum(y * np.log(p0) + (1 - y) * np.log(1 - p0)))
            boots = np.array([fit(mult[i][groups])[-k:] for i in range(args.boot)])
            rows = []
            for j, nm in enumerate(names):
                b = boots[:, j + 1]
                rows.append(dict(bin=nm, coef=float(coef[-k + 1 + j]), pp=25 * float(coef[-k + 1 + j]),
                                 lo_pp=25 * float(np.percentile(b, 2.5)), hi_pp=25 * float(np.percentile(b, 97.5))))
            from scipy.stats import chi2 as _chi2

            lr = max(2 * (ll1 - ll0), 0.0)
            return dict(slope=float(coef[-k]), lr_chi2=lr, df=D.shape[1], p_value=float(_chi2.sf(lr, D.shape[1])), rows=rows)

        def dummies(idx: np.ndarray, labels: list[str], extra_mask: np.ndarray | None = None):
            counts = np.bincount(idx, minlength=len(labels))
            refb = int(np.argmax(counts))
            cols, names = [], []
            for b in range(len(labels)):
                if b == refb:
                    continue
                col = (idx == b).astype(float)
                cols.append(col if extra_mask is None else col * extra_mask)
                names.append(labels[b])
            return np.column_stack(cols), names, labels[refb], counts

        res = {"n_games": int(len(y)), "n_events": G, "n_decks": len(Ld)}
        for var, x in (("creatures", cre), ("two_drops", two), ("lands", lands)):
            # open-ended end bins hold the rarest 3% on each side, so no bin is a handful of decks
            lo, hi = int(np.percentile(x, 3)), int(np.percentile(x, 97))
            edges = [lo, max(hi, lo + 2)]
            idx, labels = _bins(x, edges)
            D, names, refl, counts = dummies(idx, labels)
            r = run(D, names)
            r.update(reference=refl, share={lab: round(100 * c / len(idx), 1) for lab, c in zip(labels, counts)})
            res[var] = r
        # one curvature term per variable (plus a linear residual): the direct test of a missed sweet spot.
        # Both columns are clipped to the same 3%-97% range as the bins, so a few extreme decks cannot drive the curve.
        for var, x, c0 in (("creatures", cre, 15), ("two_drops", two, 6), ("lands", lands, 17)):
            lo, hi = np.percentile(x, 3), np.percentile(x, 97)
            z = np.clip(x, lo, hi) - c0
            r = run(np.column_stack([z, z ** 2]), ["linear", "squared"])
            sq = r["rows"][1]
            lin = r["rows"][0]
            # implied peak of the residual curve (only meaningful when the squared term is clearly negative)
            peak = c0 - lin["coef"] / (2 * sq["coef"]) if sq["coef"] < 0 else None
            r.update(center=c0, implied_peak=peak)
            res[var + "_quadratic"] = r
        # creature count within speed groups (coarse bins, reference 15-16 in every group)
        tert = np.quantile(mmv, [1 / 3, 2 / 3])
        speed = np.searchsorted(tert, mmv)
        cb = np.digitize(cre, [13, 15, 17, 19])          # <=12, 13-14, 15-16, 17-18, >=19
        clab = ["<=12", "13-14", "15-16", "17-18", ">=19"]
        cols, names = [], []
        for s_i, s_name in enumerate(("fast", "mid", "slow")):
            for b in (0, 1, 3, 4):
                cols.append(((speed == s_i) & (cb == b)).astype(float))
                names.append(f"{s_name}:{clab[b]}")
            if s_i:
                cols.append((speed == s_i).astype(float))
                names.append(f"{s_name}:group")
        r = run(np.column_stack(cols), names)
        r["mean_mv_cuts"] = [float(t) for t in tert]
        r["share"] = {f"{s}:{clab[b]}": round(100 * float(np.mean((speed == i) & (cb == b))), 1)
                      for i, s in enumerate(("fast", "mid", "slow")) for b in range(5)}
        res["creatures_by_speed"] = r
        # land count within speed groups: is 16 vs 17 lands priced right for fast and slow decks alike? (reference 17)
        cols, names = [], []
        for s_i, s_name in enumerate(("fast", "mid", "slow")):
            for lab, m in (("<=16", lands <= 16), (">=18", lands >= 18)):
                cols.append(((speed == s_i) & m).astype(float))
                names.append(f"{s_name}:{lab}")
            if s_i:
                cols.append((speed == s_i).astype(float))
                names.append(f"{s_name}:group")
        r = run(np.column_stack(cols), names)
        r["share"] = {f"{s}:{lab}": round(100 * float(np.mean((speed == i) & m)), 1) for i, s in enumerate(("fast", "mid", "slow"))
                      for lab, m in (("<=16", lands <= 16), ("17", lands == 17), (">=18", lands >= 18))}
        res["lands_by_speed"] = r
        out["formats"][fmt] = res

        print(f"  {fmt}: {G} events, {len(y)} games; model slope on held-out = {res['creatures']['slope']:.2f}")
        for var in ("creatures", "two_drops", "lands", "creatures_quadratic", "two_drops_quadratic", "lands_quadratic",
                    "creatures_by_speed", "lands_by_speed"):
            rr = res[var]
            print(f"   {var}: LR chi2 = {rr['lr_chi2']:.1f} on {rr['df']} df, p = {rr['p_value']:.3f}" +
                  (f"  (reference bin {rr['reference']})" if "reference" in rr else ""))
            for row in rr["rows"]:
                sh = rr.get("share", {}).get(row["bin"], "")
                print(f"     {row['bin']:>14s}: {row['pp']:+5.1f} pp [{row['lo_pp']:+5.1f}, {row['hi_pp']:+5.1f}]  share {sh}%")
    od = Path(args.outdir)
    od.mkdir(parents=True, exist_ok=True)
    (od / f"composition_{sc}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
