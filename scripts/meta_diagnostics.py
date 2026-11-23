"""Read-only diagnostics on the real corpora (nothing is written to the app home).

A. Opponent field: Sealed vs pooled field, early vs late; implied card-value shift from the deployed matchup layer.
B. Colour configurations of real Sealed decks (17lands main/splash columns): share, win rate, skill.
C. Play/draw: deck-structure x on_play (lands, curve) and card x on_play heterogeneity with split-half reliability.

    python scripts/meta_diagnostics.py OUT.json [SET,SET,...]

Session 12 audit (2026-11-12), results in docs/audit_1112/.  Reads the app home (SEALED_LAB_HOME, default the real one) and
never writes to it; OUT goes wherever you point it.  Several of these may run side by side as plain processes (JOBS = 1);
only one spawn pool at a time on this machine.
"""
import json
import os
import sys
from pathlib import Path
import zlib

import numpy as np
import polars as pl

os.environ["SEALED_LAB_CREATURE_SUPPORT"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sealed_lab import config  # noqa: E402
from sealed_lab.model import design  # noqa: E402
from sealed_lab.model.artifact import SetModel  # noqa: E402

H = str(config.sets_dir())
OUT = sys.argv[1]
SETS = sys.argv[2].split(",") if len(sys.argv) > 2 else sorted(os.listdir(H))
PP = 100 * 0.55 * 0.45
res = {}


def cp(set_code, t):
    return os.path.join(H, set_code, "corpus", t + ".parquet")


def lpm(X, y, g):
    """OLS with cluster-robust SEs (clusters g)."""
    XtX = X.T @ X
    b = np.linalg.solve(XtX, X.T @ y)
    r = y - X @ b
    order = np.argsort(g, kind="stable")
    gs = g[order]
    cuts = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1]])
    S = np.add.reduceat(X[order] * r[order, None], cuts, axis=0)
    V = np.linalg.solve(XtX, np.linalg.solve(XtX, S.T @ S).T)
    return b, np.sqrt(np.diag(V))


for s in SETS:
    R = res[s] = {}
    games = pl.read_parquet(cp(s, "games"), columns=["game_key", "event", "fmt", "draft_id", "game_number", "deck_key", "draft_time",
                                                     "main_colors", "splash_colors", "on_play", "opp_colors", "won", "skill"])
    cards = pl.read_parquet(cp(s, "cards"), columns=["card_id", "name", "mana_value", "is_land", "is_basic", "is_creature", "rarity"])
    decks = pl.read_parquet(cp(s, "decks"), columns=["deck_key", "n_cards", "n_lands"])
    dc = pl.read_parquet(cp(s, "deck_cards"))
    spell = cards.filter(~pl.col("is_land")).select("card_id", "mana_value", "is_creature")
    feat = (dc.join(spell, on="card_id").group_by("deck_key")
            .agg((pl.col("n") * pl.col("mana_value")).sum().alias("mv_sum"), pl.col("n").sum().alias("n_sp"),
                 (pl.col("n") * (pl.col("mana_value") <= 2)).sum().alias("n_cheap"),
                 (pl.col("n") * pl.col("is_creature")).sum().alias("n_cre"))
            .with_columns((pl.col("mv_sum") / pl.col("n_sp")).alias("mean_mv")))
    decks = decks.join(feat, on="deck_key", how="left")

    # ------------------------------------------------------------------ A. opponent field
    try:
        model = SetModel.load(s)
    except Exception as e:  # noqa: BLE001
        model = None
        R["model_error"] = repr(e)
    A = R["A"] = {}
    pooled_pair = None
    gp = games.with_columns(pl.col("opp_colors").map_elements(design.norm_pair, return_dtype=pl.String).alias("opp_pair"))
    pooled = gp.group_by("opp_pair").len()
    pooled_pair = {k: v / pooled["len"].sum() for k, v in pooled.iter_rows()}
    for fmt in ("bo1", "bo3"):
        g = gp.filter(pl.col("fmt") == fmt)
        if g.height == 0:
            continue
        ev = g.group_by("draft_id").agg(pl.col("draft_time").min()).sort("draft_time")
        n_ev = ev.height
        early = set(ev["draft_id"][: n_ev // 2].to_list())
        late = set(ev["draft_id"][int(n_ev * 0.75):].to_list())

        def dist(df, col):
            c = df.group_by(col).len()
            tot = c["len"].sum()
            return {k: v / tot for k, v in c.iter_rows()}

        d_all, d_early, d_late = dist(g, "opp_pair"), dist(g.filter(pl.col("draft_id").is_in(early)), "opp_pair"), dist(
            g.filter(pl.col("draft_id").is_in(late)), "opp_pair")
        keys = sorted(set(d_all) | set(pooled_pair))
        tv = lambda a, b: 0.5 * sum(abs(a.get(k, 0) - b.get(k, 0)) for k in keys)  # noqa: E731
        a = A[fmt] = dict(n_games=g.height, tv_sealed_vs_pooled=tv(d_all, pooled_pair), tv_early_vs_late=tv(d_early, d_late),
                          n_late_games=g.filter(pl.col("draft_id").is_in(late)).height,
                          pair_all={k: round(d_all.get(k, 0), 4) for k in keys},
                          pair_pooled={k: round(pooled_pair.get(k, 0), 4) for k in keys},
                          pair_late_minus_early={k: round(d_late.get(k, 0) - d_early.get(k, 0), 4) for k in keys})
        if model is None or model.opp is None or fmt not in model.opp_formats:
            a["layer"] = "not applied"
            continue
        opm = model.opp
        a["layer"] = opm.mode

        def field_effect(df):
            c = df.group_by("opp_colors").len()
            tot = c["len"].sum()
            out = np.zeros(opm.delta.shape[1])
            for o, k in c.iter_rows():
                out += (k / tot) * opm.effect(o, draws=False)
            return out

        m_all = field_effect(g)
        m_early = field_effect(g.filter(pl.col("draft_id").is_in(early)))
        m_late = field_effect(g.filter(pl.col("draft_id").is_in(late)))
        # exposure of each card in this format, and the deck-level sums over the format's builds
        gd = g.group_by("deck_key").len().rename({"len": "w"})
        x = dc.join(gd, on="deck_key").join(cards.select("card_id", "is_basic"), on="card_id").filter(~pl.col("is_basic"))
        ex = x.group_by("card_id").agg((pl.col("n") * pl.col("w")).sum().alias("e"))
        e = np.zeros(len(m_all))
        e[ex["card_id"].to_numpy()] = ex["e"].to_numpy()
        w = e / e.sum()

        def wsd(v):
            mu = float(w @ v)
            return float(np.sqrt(w @ (v - mu) ** 2)), mu

        sd_all, mu_all = wsd(m_all)
        sd_shift, mu_shift = wsd(m_late - m_early)
        xs = x.with_columns(pl.col("card_id").replace_strict(dict(enumerate(m_all.tolist())), default=0.0, return_dtype=pl.Float64).alias("m"),
                            pl.col("card_id").replace_strict(dict(enumerate((m_late - m_early).tolist())), default=0.0,
                                                             return_dtype=pl.Float64).alias("sh"))
        per_deck = xs.group_by("deck_key").agg((pl.col("n") * pl.col("m")).sum().alias("m"), (pl.col("n") * pl.col("sh")).sum().alias("sh"),
                                               pl.col("w").first())
        dw = per_deck["w"].to_numpy().astype(float)
        dw /= dw.sum()

        def dsd(v):
            mu = float(dw @ v)
            return float(np.sqrt(dw @ (v - mu) ** 2))

        names = cards.sort("card_id")["name"].to_list()
        top = np.argsort(-np.abs(m_all) * (e > np.quantile(e[e > 0], 0.3)))[:8]
        a.update(card_sd_pp=PP * sd_all, card_mean_pp=PP * mu_all, card_max_pp=PP * float(np.abs(m_all[e > 0]).max()),
                 deck_sd_pp=PP * dsd(per_deck["m"].to_numpy()), shift_card_sd_pp=PP * sd_shift,
                 shift_deck_sd_pp=PP * dsd(per_deck["sh"].to_numpy()),
                 top_cards=[(names[i], round(PP * float(m_all[i]), 3)) for i in top])

    # ------------------------------------------------------------------ B. colour configurations of real Sealed decks
    B = R["B"] = {}
    for fmt in ("bo1", "bo3"):
        g = games.filter(pl.col("fmt") == fmt)
        if g.height == 0:
            continue
        g = g.with_columns(pl.col("main_colors").fill_null("").str.len_chars().alias("nm"),
                           pl.col("splash_colors").fill_null("").str.len_chars().alias("ns"))
        g = g.with_columns(pl.when(pl.col("nm") >= 3).then(pl.lit("3+ main"))
                           .when(pl.col("ns") >= 2).then(pl.lit("2 main + 2+ splash colours"))
                           .when(pl.col("ns") == 1).then(pl.lit("2 main + 1 splash colour"))
                           .when(pl.col("nm") <= 1).then(pl.lit("mono (+splash)"))
                           .otherwise(pl.lit("2 main")).alias("cfg"))
        t = g.group_by("cfg").agg(pl.len().alias("games"), pl.col("won").mean().alias("wr"), pl.col("skill").mean().alias("skill"),
                                  pl.col("draft_id").n_unique().alias("events")).sort("games", descending=True)
        # skill-adjusted difference vs '2 main': linear probability model with skill bucket dummies
        gg = g.filter(pl.col("skill").is_not_null())
        cfgs = [c for c in t["cfg"].to_list() if c != "2 main"]
        sk = gg["skill"].to_numpy().astype(float)
        cols = [np.ones(gg.height), sk, sk ** 2, gg["on_play"].to_numpy().astype(float)]
        for c in cfgs:
            cols.append((gg["cfg"] == c).to_numpy().astype(float))
        X = np.column_stack(cols)
        y = gg["won"].to_numpy().astype(float)
        gid = np.array([zlib.crc32(d.encode()) for d in gg["draft_id"].to_list()])
        b, se = lpm(X, y, gid)
        adj = {c: (round(100 * b[4 + i], 2), round(100 * se[4 + i], 2)) for i, c in enumerate(cfgs)}
        B[fmt] = dict(rows=[dict(cfg=r[0], share=round(r[1] / g.height, 4), wr=round(r[2], 4), skill=round(r[3], 4), events=r[4])
                            for r in t.iter_rows()], adj_vs_2main_pp=adj)

    # ------------------------------------------------------------------ C. play / draw
    C = R["C"] = {}
    g1 = games.filter((pl.col("game_number").is_null()) | (pl.col("game_number") <= 1)).filter(pl.col("on_play").is_not_null())
    g1 = g1.join(decks, on="deck_key").filter(pl.col("n_cards") == 40)
    for grp, flt in (("draft", pl.col("fmt") == "draft"), ("bo1", pl.col("fmt") == "bo1"), ("bo3_g1", pl.col("fmt") == "bo3"),
                     ("sealed", pl.col("fmt").is_in(["bo1", "bo3"])), ("all", pl.lit(True))):
        g = g1.filter(flt).filter(pl.col("n_lands").is_between(15, 18) & pl.col("mean_mv").is_not_null())
        if g.height < 3000:
            continue
        p = g["on_play"].to_numpy().astype(float)
        nl = g["n_lands"].to_numpy().astype(float) - 16.5
        mv = g["mean_mv"].to_numpy().astype(float)
        mv = mv - mv.mean()
        ch = g["n_cheap"].to_numpy().astype(float)
        ch = ch - ch.mean()
        X = np.column_stack([np.ones(g.height), p, nl, mv, ch, p * nl, p * mv, p * ch])
        y = g["won"].to_numpy().astype(float)
        gid = np.array([zlib.crc32(d.encode()) for d in g["draft_id"].to_list()])
        b, se = lpm(X, y, gid)
        tab = g.filter(pl.col("n_lands").is_in([16, 17])).group_by("n_lands", "on_play").agg(pl.len().alias("n"), pl.col("won").mean().alias("wr")).sort(
            "n_lands", "on_play")
        C[grp] = dict(n=g.height, play_pp=(100 * b[1], 100 * se[1]), play_x_land_pp=(100 * b[5], 100 * se[5]),
                      play_x_meanmv_pp=(100 * b[6], 100 * se[6]), play_x_cheap_pp=(100 * b[7], 100 * se[7]),
                      mv_sd=float(g["mean_mv"].std()), cheap_sd=float(g["n_cheap"].std()),
                      table=[(int(r[0]), bool(r[1]), int(r[2]), round(float(r[3]), 4)) for r in tab.iter_rows()])
    # card x on_play (all formats, first games): d_c = WR(play) - WR(draw) among games whose deck holds the card
    gg = g1.select("deck_key", "on_play", "won", "draft_id").with_columns(
        (pl.col("draft_id").hash(seed=7) % 2).alias("half"))
    nb = cards.filter(~pl.col("is_land")).select("card_id")
    j = dc.join(nb, on="card_id").select("deck_key", "card_id").join(gg, on="deck_key")
    agg = j.group_by("card_id", "on_play", "half").agg(pl.len().alias("n"), pl.col("won").sum().alias("w"))
    names = dict(cards.select("card_id", "name").iter_rows())
    mvs = dict(cards.select("card_id", "mana_value").iter_rows())

    def dtab(a):
        pv = a.group_by("card_id", "on_play").agg(pl.col("n").sum(), pl.col("w").sum())
        pl_ = pv.filter(pl.col("on_play")).rename({"n": "n1", "w": "w1"}).drop("on_play")
        dr = pv.filter(~pl.col("on_play")).rename({"n": "n0", "w": "w0"}).drop("on_play")
        t = pl_.join(dr, on="card_id").filter((pl.col("n1") >= 1500) & (pl.col("n0") >= 1500))
        p1, p0 = t["w1"] / t["n1"], t["w0"] / t["n0"]
        d = (p1 - p0).to_numpy()
        v = (p1 * (1 - p1) / t["n1"] + p0 * (1 - p0) / t["n0"]).to_numpy()
        return t["card_id"].to_numpy(), d, v

    ids, d, v = dtab(agg)
    wt = 1 / v
    dbar = float((wt * d).sum() / wt.sum())
    Q = float((wt * (d - dbar) ** 2).sum())
    k = len(d)
    tau2 = max(0.0, (Q - (k - 1)) / (wt.sum() - (wt ** 2).sum() / wt.sum()))
    z = (d - dbar) / np.sqrt(v)
    # split-half reliability of the deviations
    ia, da, va = dtab(agg.filter(pl.col("half") == 0))
    ib, db, vb = dtab(agg.filter(pl.col("half") == 1))
    common = np.intersect1d(ia, ib)
    sa, sb = dict(zip(ia.tolist(), da.tolist())), dict(zip(ib.tolist(), db.tolist()))
    xa = np.array([sa[c] for c in common])
    xb = np.array([sb[c] for c in common])
    r_half = float(np.corrcoef(xa, xb)[0, 1]) if len(common) > 10 else float("nan")
    mv_arr = np.array([mvs[int(c)] for c in ids], dtype=float)
    r_mv = float(np.corrcoef(mv_arr, d)[0, 1])
    order = np.argsort(z)
    C["card_x_play"] = dict(n_cards=int(k), play_minus_draw_pp=100 * dbar, tau_pp=100 * float(np.sqrt(tau2)), Q=Q, df=k - 1,
                            z_gt3=int((np.abs(z) > 3).sum()), z_gt3_expected=round(0.0027 * k, 1), split_half_r=r_half, corr_with_mv=r_mv,
                            median_se_pp=100 * float(np.median(np.sqrt(v))),
                            best_on_play=[(names[int(ids[i])], round(100 * float(d[i] - dbar), 2), round(float(z[i]), 1), mvs[int(ids[i])])
                                          for i in order[::-1][:8]],
                            best_on_draw=[(names[int(ids[i])], round(100 * float(d[i] - dbar), 2), round(float(z[i]), 1), mvs[int(ids[i])])
                                          for i in order[:8]])
    print(s, "done", flush=True)

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(res, f, indent=1, default=float)
