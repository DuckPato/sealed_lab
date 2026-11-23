"""Opening-hand natural experiment: effect of a card being in the OPENING HAND, within the same deck, against the model's deck-level card values.
(diagnostic only: reads the corpus and the deployed model, changes nothing)

    python scripts/opening_hand_check.py run SET [SET ...] --out DIR [--jobs 1]     estimate psi per set and stage, split halves
    python scripts/opening_hand_check.py report DIR [--docs DIR2]                   opening_hand.md / opening_hand.json (default: docs/opening_hand)

Given the deck, the shuffle decides which cards start in hand, so the opening-hand copies of a card are a natural experiment. Rows = games of one
stage (draft games, Bo1 Sealed games, Bo3 Sealed games); offset = the stage's fitted logit (the plain chain refitted on all events at the
artifact's fixed sigma / stacking weight with the pipeline's own design code, `split_half_decompose.HalfSetup` / `refit_chain_fixed`: deck effect,
pilot and game covariates; no matchup / pair / archetype extras); ridge logistic (card prior SD 1.0) with a free intercept.
  E1 (primary)  kept 7-card hands (num_mulligans == 0); z[g,c] = opening[g,c] - n[d,c] * 7 / N_d (n = copies in the game's deck, N_d = deck size
                incl. basics), plus one regressor for the centred basic lands in hand. Centred on the design: the regressor is orthogonal to deck effects.
  E2            same games, z = opening - mean opening over the deck's kept games (decks with >= 2 kept games): deck fixed effects.
  S1            all games, z centred with the stored hand size (7 for every game: see the storage finding in the report), num_mulligans covariates.
  E1 at prior SD 0.3 is the sensitivity run.
psi is relative (the regressors sum to a constant): only slopes and correlations are compared. Comparisons use cards with exposure >= 200 and
`handcheck.kappa_and_flags`; the split-half uses `split_half_search.split_events` (seed 0), the draft beta of each half is `fit_draft_stage` at the
artifact's sigma. The `run` step works on a scratch copy of each set (OUT/home); the app home is only read.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

SETS = ["ECL", "EOE", "FIN", "FRA", "HOB", "MSH", "OM1", "SOS", "TDM", "TLA", "TMT"]
STAGES = ("draft", "bo1", "bo3")
ESTS = ("E1", "E1s03", "E2", "S1")
MIN_EXPO = 200.0
HAND = 7.0
WIDE = 10.0
ILLUSTRATIONS = {"FRA": ["Murmuring Volume"],
                 "MSH": ["Super Strength", "Iron Fist", "Speedball", "Borough Backup", "Phil Coulson", "Political Triumph"]}


# =====================================================================================================================
# pure arithmetic (tested on synthetic data)
# =====================================================================================================================

def centre_design(X, ND, N, hand=HAND) -> sp.csr_matrix:
    """z = X - ND * hand / N: opening copies minus the copies a random `hand`-card hand of the deck holds on average (rows = games)."""
    N = np.asarray(N, float)
    h = np.broadcast_to(np.asarray(hand, float), N.shape)
    return (sp.csr_matrix(X) - sp.diags(h / N) @ sp.csr_matrix(ND)).tocsr()


def within_centre(X, grp) -> tuple[sp.csr_matrix, np.ndarray]:
    """z = X - mean of X over the rows of the same group, groups with >= 2 rows only. Returns (Z of the kept rows, keep mask)."""
    grp = np.asarray(grp)
    u, inv, cnt = np.unique(grp, return_inverse=True, return_counts=True)
    keep = cnt[inv] >= 2
    X = sp.csr_matrix(X)[np.nonzero(keep)[0]]
    u, inv, cnt = np.unique(grp[keep], return_inverse=True, return_counts=True)
    S = sp.csr_matrix((np.ones(len(inv)), (np.arange(len(inv)), inv)), shape=(len(inv), len(u)))
    M = sp.diags(1.0 / cnt) @ (S.T @ X)
    return (X - S @ M).tocsr(), keep


def fit_hand(Z, offset, y, sigma: float = 1.0, dense=None, wide: float = WIDE) -> dict:
    """Ridge logistic y ~ offset + intercept + Z psi (+ dense covariates); the LAST column of Z (basic lands) and the covariates get a wide prior.
    Returns dict(psi, se (all but the basic column), basic, basic_se, extra (covariate coefficients), converged)."""
    from sealed_lab.model import glm

    n, p = Z.shape
    cols = [sp.csr_matrix(np.ones((n, 1))), sp.csr_matrix(Z)]
    k = 0
    if dense is not None:
        dense = np.asarray(dense, float).reshape(n, -1)
        k = dense.shape[1]
        cols.append(sp.csr_matrix(dense))
    C = sp.hstack(cols, format="csr")
    prob = glm.Problem(y=np.asarray(y, float), C=C, offset=np.asarray(offset, float))
    sig = np.concatenate([[wide], np.full(p - 1, sigma), [wide], np.full(k, wide)])
    res = glm.fit(prob, np.zeros(p + 1 + k), sig, maxiter=300)
    se = glm.posterior_se(prob, res.coef, sig)
    return dict(psi=res.coef[1:p], se=se[1:p], basic=float(res.coef[p]), basic_se=float(se[p]), extra=res.coef[p + 1:], converged=bool(res.converged))


def wls(y, X, w) -> dict:
    """Weighted least squares with an intercept, residual-scaled SEs. Returns dict(coef (without the intercept), se, t, intercept, n)."""
    y = np.asarray(y, float)
    X = np.asarray(X, float).reshape(len(y), -1)
    w = np.asarray(w, float)
    A = np.column_stack([np.ones(len(y)), X])
    G = A.T @ (A * w[:, None])
    coef = np.linalg.solve(G, A.T @ (w * y))
    e = y - A @ coef
    s2 = float((w * e ** 2).sum() / max(len(y) - A.shape[1], 1))
    se = np.sqrt(np.diag(np.linalg.inv(G)) * s2)
    return dict(coef=coef[1:], se=se[1:], t=coef[1:] / se[1:], intercept=float(coef[0]), n=int(len(y)))


def corr(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def disattenuated(cross1: float, cross2: float, rel_a: float, rel_b: float) -> float:
    """Mean cross correlation / sqrt(reliability_a * reliability_b): the correlation of the two quantities without sampling noise."""
    den = rel_a * rel_b
    return float(0.5 * (cross1 + cross2) / np.sqrt(den)) if np.isfinite(den) and den > 0 else float("nan")


def compare(beta, bse, expo, psi, pse, min_expo: float = MIN_EXPO) -> dict:
    """psi vs a beta (deck effect) over the cards with exposure >= min_expo: Pearson, Spearman, kappa, tau, flags, median SE(kappa psi)/SE(beta)."""
    from sealed_lab.model import handcheck

    beta, bse, expo, psi, pse = (np.asarray(x, float) for x in (beta, bse, expo, psi, pse))
    ok = np.isfinite(beta) & np.isfinite(bse) & np.isfinite(psi) & np.isfinite(pse)
    expo = np.where(ok, expo, 0.0)
    kf = handcheck.kappa_and_flags(np.where(ok, beta, 0), np.where(ok, bse, 1), np.where(ok, psi, 0), np.where(ok, pse, 1), expo, min_expo)
    m = expo >= min_expo
    out = dict(n=int(m.sum()), pearson=corr(beta[m], psi[m]), spearman=kf["spearman"], kappa=kf["kappa"], kappa_se=kf["kappa_se"],
               tau=kf.get("tau"), informative=bool(kf["informative"]), n_flag=int(kf["flag"].sum()), flag=kf["flag"])
    out["se_ratio"] = float(np.median(abs(kf["kappa"]) * pse[m] / bse[m])) if kf["kappa"] is not None and m.sum() else None
    return out


def pool(values, ses=None) -> dict:
    """Simple mean over sets with the between-set SD (and the inverse-variance mean when SEs are given)."""
    v = np.array([x for x in values if x is not None and np.isfinite(x)], float)
    if not len(v):
        return dict(n=0, mean=None, sd=None, iv=None, iv_se=None)
    out = dict(n=int(len(v)), mean=float(v.mean()), sd=float(v.std(ddof=1)) if len(v) > 1 else None, iv=None, iv_se=None)
    if ses is not None:
        pr = [(x, s) for x, s in zip(values, ses) if x is not None and s is not None and np.isfinite(x) and np.isfinite(s) and s > 0]
        if pr:
            w = np.array([1 / s ** 2 for _, s in pr])
            out["iv"] = float(sum(wi * x for wi, (x, _) in zip(w, pr)) / w.sum())
            out["iv_se"] = float(1 / np.sqrt(w.sum()))
    return out


# =====================================================================================================================
# data
# =====================================================================================================================

def prepare_home(set_code: str, home: Path) -> Path:
    """Scratch copy of one set (corpus tables, model meta, cache files) under HOME/sets/SET; the app home is only read."""
    from sealed_lab import config

    src = Path(config.app_home()) / "sets" / set_code
    dst = Path(home) / "sets" / set_code
    for sub in ("corpus", "model", "cache"):
        (dst / sub).mkdir(parents=True, exist_ok=True)
        for f in (src / sub).iterdir():
            if f.is_file() and not (dst / sub / f.name).exists() and not (sub == "model" and f.suffix == ".html"):
                shutil.copy2(f, dst / sub / f.name)
    return dst


def deck_matrices(gk, deck_key, hands, deck_cards, col_of_card, n_card):
    """X (G x n_card+1): opening copies per card column, last column = basic lands in hand (cards outside the card space); ND: copies in the game's
    deck (same layout); N: deck size. Rows follow `gk`."""
    import polars as pl

    G = len(gk)
    gdf = pl.DataFrame({"game_key": np.asarray(gk, np.int64), "row": np.arange(G, dtype=np.int64)})
    h = (hands.filter(pl.col("opening") > 0).select("game_key", "card_id", "opening").join(gdf.lazy(), on="game_key", how="inner").collect())
    col = col_of_card[h["card_id"].to_numpy().astype(np.int64)]
    col = np.where(col < 0, n_card, col)
    X = sp.csr_matrix((h["opening"].to_numpy().astype(float), (h["row"].to_numpy(), col)), shape=(G, n_card + 1))
    uk, inv = np.unique(np.asarray(deck_key, np.int64), return_inverse=True)
    dc = deck_cards.filter(pl.col("deck_key").is_in(uk.tolist()))
    di = np.searchsorted(uk, dc["deck_key"].to_numpy())
    dcol = col_of_card[dc["card_id"].to_numpy().astype(np.int64)]
    dcol = np.where(dcol < 0, n_card, dcol)
    ND_deck = sp.csr_matrix((dc["n"].to_numpy().astype(float), (di, dcol)), shape=(len(uk), n_card + 1))
    N_deck = np.asarray(ND_deck.sum(axis=1)).ravel()
    return X, ND_deck[inv], N_deck[inv], inv


def estimate_stage(st_games, offset, y, mats, n_card: int) -> dict:
    """E1, E1 (prior SD 0.3), E2 and S1 for one stage. `mats` = (X, ND, N, deck index) of the stage's games (rows aligned to `offset`, `y`)."""
    X, ND, N, didx = mats
    nm = st_games["num_mulligans"].fill_null(0).to_numpy()
    kept = nm == 0
    out: dict = dict(n_games=int(len(y)), n_kept=int(kept.sum()))
    Xk, NDk, Nk = X[np.nonzero(kept)[0]], ND[np.nonzero(kept)[0]], N[kept]
    Z1 = centre_design(Xk, NDk, Nk)
    out["E1"] = fit_hand(Z1, offset[kept], y[kept], 1.0)
    out["E1s03"] = fit_hand(Z1, offset[kept], y[kept], 0.3)
    Z2, k2 = within_centre(Xk, didx[kept])
    idx2 = np.nonzero(kept)[0][k2]
    out["E2"] = fit_hand(Z2, offset[idx2], y[idx2], 1.0)
    out["n_E2"], out["n_E2_decks"] = int(k2.sum()), int(len(np.unique(didx[idx2])))
    Zs = centre_design(X, ND, N)
    out["S1"] = fit_hand(Zs, offset, y, 1.0, dense=np.column_stack([nm == 1, nm >= 2]).astype(float))
    return out


def hand_storage(set_code: str, corpus: Path, cards_tab) -> dict:
    """What the stored opening hand is: cards per game, and land counts by number of mulligans (a rejected hand would show extreme counts)."""
    import polars as pl

    g = pl.read_parquet(corpus / "games.parquet", columns=["game_key", "num_mulligans"])
    h = pl.scan_parquet(corpus / "hands.parquet").filter(pl.col("opening") > 0)
    land_ids = cards_tab.filter(pl.col("is_land"))["card_id"].to_list()
    s = h.group_by("game_key").agg(pl.col("opening").sum().alias("tot"), (pl.col("opening") * pl.col("card_id").is_in(land_ids)).sum().alias("lands")).collect()
    j = g.join(s, on="game_key", how="left").with_columns(pl.col("num_mulligans").fill_null(0).clip(0, 2).alias("m"))
    out: dict = dict(n_games=int(g.height), share_7_cards=float((j["tot"] == 7).mean()), min_cards=int(j["tot"].min()), max_cards=int(j["tot"].max()), by_mull={})
    for m in (0, 1, 2):
        d = j.filter(pl.col("m") == m)
        if d.height:
            out["by_mull"][str(m)] = dict(n=int(d.height), mean_cards=float(d["tot"].mean()), mean_lands=float(d["lands"].mean()),
                                          share_extreme_lands=float(((d["lands"] <= 1) | (d["lands"] >= 6)).mean()))
    return out


def _starts(st):
    return [np.datetime64(x, "us").astype("int64") for x in st.group_start]


def _fit_draft(setup, sigma=None):
    """fit_draft_stage's fit, also returning the fitted logit of the stage's games (rows of setup.stages['draft'].games)."""
    from sealed_lab.model import cardvalue

    st = setup.stages["draft"]
    if sigma is None:
        sigma = float(setup.meta["prior_strength"]["stages"]["draft"]["sigma"])
    spec = cardvalue.make_spec(st, setup.n_card, setup.n_feat, "draft")
    coef, se, res = cardvalue.fit_stage(st, np.zeros(setup.n_card + setup.n_feat), spec, sigma, with_se=True)
    k = st.prob.k
    return dict(beta=coef[k:k + setup.n_card], se=np.asarray(se[1])[k:k + setup.n_card], expo=np.asarray(st.exposure, float),
                eta=st.prob.eta(coef), converged=bool(res.converged), sigma=float(sigma))


def run_set(set_code: str, out: Path, jobs: int = 1) -> dict:
    import polars as pl

    import split_half_decompose as sd
    from split_half_search import split_events
    from sealed_lab import config
    from sealed_lab.model import pipeline

    t0 = time.time()
    app_model = Path(config.app_home()) / "sets" / set_code / "model"
    deployed = pl.read_parquet(app_model / "cards.parquet")
    home = Path(out) / "home"
    prepare_home(set_code, home)
    corpus = home / "sets" / set_code / "corpus"
    meta = json.loads((home / "sets" / set_code / "model" / "meta.json").read_text(encoding="utf-8"))
    info: dict = dict(set=set_code, formats=meta["formats"], stages={}, half={})
    arrays: dict = {}
    with sd._home(home):
        inp = pipeline.load_inputs(set_code)
        if not (meta.get("extra_draft") or {}).get("applied"):
            inp = pipeline.without_extra_draft(inp)
        ev_all = inp.games["draft_id"].unique().to_list()
        n_in = {f: int(inp.games.filter(pl.col("fmt") == f).height) for f in STAGES}
        setup = sd.HalfSetup(set_code, home, ev_all, stages=tuple(f for f in STAGES if n_in[f] > 0), n_jobs=jobs, allow_plain=True)
        fits = sd.refit_chain_fixed(setup)
        t_setup = time.time() - t0
        space, nbi = setup.space, setup.nb_ids
        n_card = space.n_card
        names = deployed.sort("card_id")["name"].to_list()
        info["names"] = [names[i] for i in nbi]
        info["card_ids"] = [int(i) for i in nbi]
        info["storage"] = hand_storage(set_code, corpus, deployed)
        dep = deployed.sort("card_id")
        hands = pl.scan_parquet(corpus / "hands.parquet")
        # reference deck effects: draft-stage fit on all events (with SEs), deployed beta_total of the formats
        refs: dict = {}
        draft_fit = None
        if "draft" in setup.stages:
            draft_fit = _fit_draft(setup)
            refs["draft"] = dict(beta=draft_fit["beta"], se=draft_fit["se"], expo=draft_fit["expo"])
        for f in ("bo1", "bo3"):
            if f"beta_{f}" in dep.columns:
                refs[f] = dict(beta=dep[f"beta_{f}"].to_numpy()[nbi], se=dep[f"se_{f}"].to_numpy()[nbi], expo=dep[f"exposure_{f}"].to_numpy()[nbi])
        for r, d in refs.items():
            for k, v in d.items():
                arrays[f"ref/{r}/{k}"] = np.asarray(v, float)
        for f in ("bo1", "bo3"):
            if f in fits:
                ft = fits[f]
                b = ft["coef"][ft["k"]:ft["k"] + n_card]
                if f in refs:
                    m = refs[f]["expo"] >= MIN_EXPO
                    info.setdefault("recon", {})[f] = dict(rms=float(np.sqrt(np.mean((b[m] - refs[f]["beta"][m]) ** 2))),
                                                           corr=corr(b[m], refs[f]["beta"][m]), art_rms=float(np.sqrt(np.mean(refs[f]["beta"][m] ** 2))))
        if draft_fit is not None:
            info["recon_draft_sigma"] = draft_fit["sigma"]
        # the deployed (psi_flag, psi) for the illustrations
        for f in ("bo1", "bo3"):
            if f"psi_flag_{f}" in dep.columns:
                arrays[f"deployed/psi_flag_{f}"] = dep[f"psi_flag_{f}"].to_numpy()[nbi].astype(float)
        arrays["deployed/psi"] = dep["psi"].to_numpy()[nbi]
        mats_all: dict = {}
        etas: dict = {}
        for name, st in setup.stages.items():
            t1 = time.time()
            if name == "draft":
                eta = draft_fit["eta"]
            else:
                eta = st.prob.eta(fits[name]["coef"])
            y = np.asarray(st.prob.y, float)
            gk = st.games["game_key"].to_numpy()
            mats = deck_matrices(gk, st.games["deck_key"].to_numpy(), hands, inp.deck_cards, space.col_of_card, n_card)
            mats_all[name], etas[name] = mats, eta
            res = estimate_stage(st.games, eta, y, mats, n_card)
            sinfo = {k: v for k, v in res.items() if k not in ESTS}
            for e in ESTS:
                arrays[f"{name}/{e}/psi"], arrays[f"{name}/{e}/se"] = res[e]["psi"], res[e]["se"]
                sinfo[e] = dict(basic=res[e]["basic"], basic_se=res[e]["basic_se"], converged=res[e]["converged"])
                if e == "S1":
                    sinfo[e]["mull_coef"] = [float(x) for x in res[e]["extra"]]
            sinfo["seconds"] = round(time.time() - t1, 1)
            info["stages"][name] = sinfo
            print(f"  {set_code} {name}: {sinfo['n_games']} games ({sinfo['n_kept']} kept), {sinfo['seconds']}s", flush=True)
        # split halves: draft (beta and E1 psi per half), bo1 (E1 psi only)
        if "draft" in setup.stages:
            t1 = time.time()
            st = setup.stages["draft"]
            ids = list(st.group_ids)
            a, b = split_events(ids, _starts(st), ["draft"] * len(ids), 0)
            gkf = st.games["game_key"].to_numpy()
            order = np.argsort(gkf)
            yf = np.asarray(st.prob.y, float)
            X, ND, N, didx = mats_all["draft"]
            for hname, evh in (("A", a), ("B", b)):
                sh = sd.HalfSetup(set_code, home, evh, stages=("draft",), n_jobs=jobs, allow_plain=True)
                assert np.array_equal(sh.nb_ids, nbi)
                fh = _fit_draft(sh)
                sth = sh.stages["draft"]
                rows = order[np.searchsorted(gkf[order], sth.games["game_key"].to_numpy())]
                kept = np.nonzero(sth.games["num_mulligans"].fill_null(0).to_numpy() == 0)[0]
                r2 = rows[kept]
                Z = centre_design(X[r2], ND[r2], N[r2])
                fp = fit_hand(Z, fh["eta"][kept], yf[r2], 1.0)
                for k, v in (("beta", fh["beta"]), ("se", fh["se"]), ("expo", fh["expo"]), ("psi", fp["psi"]), ("pse", fp["se"])):
                    arrays[f"half/draft/{hname}/{k}"] = np.asarray(v, float)
                info["half"][f"draft_{hname}"] = dict(n_events=len(evh), n_games=int(sth.games.height), n_kept=int(len(kept)), converged=fh["converged"])
            info["half"]["draft_seconds"] = round(time.time() - t1, 1)
            print(f"  {set_code} draft halves: {info['half']['draft_seconds']}s", flush=True)
        if "bo1" in setup.stages:
            t1 = time.time()
            st = setup.stages["bo1"]
            ids = list(st.group_ids)
            a, b = split_events(ids, _starts(st), ["bo1"] * len(ids), 0)
            X, ND, N, didx = mats_all["bo1"]
            did = st.games["draft_id"].to_numpy()
            nm = st.games["num_mulligans"].fill_null(0).to_numpy()
            yf = np.asarray(st.prob.y, float)
            for hname, evh in (("A", a), ("B", b)):
                r = np.nonzero(np.isin(did, np.array(list(evh), dtype=did.dtype)) & (nm == 0))[0]
                fp = fit_hand(centre_design(X[r], ND[r], N[r]), etas["bo1"][r], yf[r], 1.0)
                arrays[f"half/bo1/{hname}/psi"], arrays[f"half/bo1/{hname}/pse"] = fp["psi"], fp["se"]
                info["half"][f"bo1_{hname}"] = dict(n_events=len(evh), n_kept=int(len(r)), converged=fp["converged"])
            info["half"]["bo1_seconds"] = round(time.time() - t1, 1)
    info["seconds"] = round(time.time() - t0, 1)
    info["seconds_setup"] = round(t_setup, 1)
    Path(out).mkdir(parents=True, exist_ok=True)
    np.savez(Path(out) / f"{set_code}.npz", **arrays)
    (Path(out) / f"{set_code}.json").write_text(json.dumps(info, indent=1, default=float), encoding="utf-8")
    print(f"opening_hand {set_code}: {info['seconds']:.0f}s (setup {t_setup:.0f}s)", flush=True)
    return info


# =====================================================================================================================
# report
# =====================================================================================================================

def _load(out: Path, sets):
    res = {}
    for sc in sets:
        p, q = Path(out) / f"{sc}.json", Path(out) / f"{sc}.npz"
        if p.exists() and q.exists():
            with np.load(q) as z:
                res[sc] = (json.loads(p.read_text(encoding="utf-8")), {k: z[k] for k in z.files})
    return res


def _matching_ref(stage: str) -> str:
    return stage


def set_stage_stats(info: dict, z: dict) -> dict:
    """Comparisons of one set: per stage and estimator vs every available reference; E1-E2 agreement; the headline = matching reference."""
    out: dict = {}
    for stage in info["stages"]:
        o: dict = dict(n_games=info["stages"][stage]["n_games"], n_kept=info["stages"][stage]["n_kept"], n_E2=info["stages"][stage]["n_E2"], vs={})
        for ref in ("draft", "bo1", "bo3"):
            if f"ref/{ref}/beta" not in z:
                continue
            r = {}
            for e in ESTS:
                c = compare(z[f"ref/{ref}/beta"], z[f"ref/{ref}/se"], z[f"ref/{ref}/expo"], z[f"{stage}/{e}/psi"], z[f"{stage}/{e}/se"])
                r[e] = c
            o["vs"][ref] = r
        ref = stage if f"ref/{stage}/beta" in z else None
        if ref:
            m = z[f"ref/{ref}/expo"] >= MIN_EXPO
            p1, p2 = z[f"{stage}/E1/psi"][m], z[f"{stage}/E2/psi"][m]
            o["e1_e2_corr"] = corr(p1, p2)
            o["e1_e2_slope"] = float(np.polyfit(p1, p2, 1)[0]) if m.sum() > 2 else None
            p3 = z[f"{stage}/S1/psi"][m]
            o["e1_s1_corr"] = corr(p1, p3)
        out[stage] = o
    return out


def split_half_stats(info: dict, z: dict) -> dict:
    out: dict = {}
    if "half/draft/A/psi" in z:
        m = z["ref/draft/expo"] >= MIN_EXPO
        g = lambda h, k: z[f"half/draft/{h}/{k}"][m]
        rp, rb = corr(g("A", "psi"), g("B", "psi")), corr(g("A", "beta"), g("B", "beta"))
        c1, c2 = corr(g("A", "psi"), g("B", "beta")), corr(g("A", "beta"), g("B", "psi"))
        d = dict(n=int(m.sum()), rel_psi=rp, rel_beta=rb, cross_psiA_betaB=c1, cross_betaA_psiB=c2, disattenuated=disattenuated(c1, c2, rp, rb), inc={})
        for s, t in (("A", "B"), ("B", "A")):
            X = np.column_stack([g(s, "beta"), g(s, "psi")])
            i = wls(g(t, "beta"), X, 1.0 / g(t, "se") ** 2)
            ii = wls(g(t, "psi"), X, 1.0 / g(t, "pse") ** 2)
            d["inc"][f"{s}->{t}"] = dict(psi_in_beta=dict(coef=float(i["coef"][1]), se=float(i["se"][1]), t=float(i["t"][1]),
                                                           beta_coef=float(i["coef"][0]), beta_t=float(i["t"][0])),
                                         beta_in_psi=dict(coef=float(ii["coef"][0]), se=float(ii["se"][0]), t=float(ii["t"][0]),
                                                          psi_coef=float(ii["coef"][1]), psi_t=float(ii["t"][1])))
        out["draft"] = d
    if "half/bo1/A/psi" in z and "ref/bo1/expo" in z:
        m = z["ref/bo1/expo"] >= MIN_EXPO
        out["bo1"] = dict(n=int(m.sum()), rel_psi=corr(z["half/bo1/A/psi"][m], z["half/bo1/B/psi"][m]))
    return out


def illustrate(sets_data: dict, stats: dict) -> list[dict]:
    rows = []
    for sc, cards in ILLUSTRATIONS.items():
        if sc not in sets_data:
            continue
        info, z = sets_data[sc]
        names = [n.lower() for n in info["names"]]
        for cname in cards:
            key = cname.lower()
            hit = ([j for j, n in enumerate(names) if n == key] or [j for j, n in enumerate(names) if n.startswith(key + ",")]
                   or [j for j, n in enumerate(names) if key in n])
            if len(hit) != 1:
                rows.append(dict(set=sc, card=cname, missing=True))
                continue
            i = hit[0]
            row = dict(set=sc, card=info["names"][i], beta={f: float(z[f"ref/{f}/beta"][i]) for f in ("draft", "bo1", "bo3") if f"ref/{f}/beta" in z},
                       beta_se={f: float(z[f"ref/{f}/se"][i]) for f in ("draft", "bo1", "bo3") if f"ref/{f}/beta" in z},
                       deployed_flag={f: bool(z[f"deployed/psi_flag_{f}"][i]) for f in ("bo1", "bo3") if f"deployed/psi_flag_{f}" in z}, stage={})
            for stage in info["stages"]:
                c = stats[sc][stage]["vs"].get(stage, {}).get("E1")
                if c is None:
                    continue
                psi, se = float(z[f"{stage}/E1/psi"][i]), float(z[f"{stage}/E1/se"][i])
                row["stage"][stage] = dict(psi=psi, se=se, kappa_psi=None if c["kappa"] is None else c["kappa"] * psi, flag=bool(c["flag"][i]))
            rows.append(row)
    return rows


def f3(x, nd=3):
    return "-" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def build_report(out: Path, sets=SETS) -> tuple[dict, str]:
    data = _load(out, sets)
    sets = [s for s in sets if s in data]
    stats = {s: set_stage_stats(*data[s]) for s in sets}
    halves = {s: split_half_stats(*data[s]) for s in sets}
    res: dict = dict(sets=sets, min_exposure=MIN_EXPO, per_set={}, halves=halves, pooled={}, illustrations=illustrate(data, stats))
    for s in sets:
        res["per_set"][s] = {st: {k: v for k, v in d.items() if k != "vs"} | {
            "vs": {ref: {e: {k: v for k, v in c.items() if k != "flag"} for e, c in r.items()} for ref, r in d["vs"].items()}}
            for st, d in stats[s].items()}
        res["per_set"][s]["_info"] = {k: data[s][0][k] for k in ("seconds", "storage", "recon", "stages") if k in data[s][0]}
    # pooled: per stage, headline = matching reference (draft stage also vs bo1)
    for stage, ref in (("draft", "draft"), ("draft", "bo1"), ("bo1", "bo1"), ("bo3", "bo3")):
        key = f"{stage}_vs_{ref}"
        pk = {}
        for e in ESTS:
            cs = [stats[s][stage]["vs"][ref][e] for s in sets if stage in stats[s] and ref in stats[s][stage]["vs"]]
            pk[e] = {m: pool([c[m] for c in cs]) for m in ("pearson", "spearman", "kappa", "tau", "se_ratio", "n_flag", "n")}
            pk[e]["kappa_iv"] = pool([c["kappa"] for c in cs], [c["kappa_se"] for c in cs])
            pk[e]["n_sets"] = len(cs)
        e12 = [stats[s][stage].get("e1_e2_corr") for s in sets if stage in stats[s] and ref == stage]
        pk["e1_e2_corr"] = pool(e12)
        pk["e1_e2_slope"] = pool([stats[s][stage].get("e1_e2_slope") for s in sets if stage in stats[s] and ref == stage])
        res["pooled"][key] = pk
    ph: dict = {}
    hd = [halves[s]["draft"] for s in sets if "draft" in halves[s]]
    if hd:
        ph["draft"] = {k: pool([d[k] for d in hd]) for k in ("rel_psi", "rel_beta", "cross_psiA_betaB", "cross_betaA_psiB", "disattenuated")}
        # coefficients: per set the mean of the two directions, then across sets
        for nm, sub in (("psi_in_beta", "psi_in_beta"), ("beta_in_psi", "beta_in_psi")):
            cf = [np.mean([v[sub]["coef"] for v in d["inc"].values()]) for d in hd]
            tt = [np.mean([v[sub]["t"] for v in d["inc"].values()]) for d in hd]
            ph["draft"][nm + "_coef"], ph["draft"][nm + "_t"] = pool(cf), pool(tt)
        cfall = [v["psi_in_beta"] for d in hd for v in d["inc"].values()]
        ph["draft"]["psi_in_beta_iv"] = pool([c["coef"] for c in cfall], [c["se"] for c in cfall])
        cfall = [v["beta_in_psi"] for d in hd for v in d["inc"].values()]
        ph["draft"]["beta_in_psi_iv"] = pool([c["coef"] for c in cfall], [c["se"] for c in cfall])
    hb = [halves[s]["bo1"] for s in sets if "bo1" in halves[s]]
    if hb:
        ph["bo1"] = {"rel_psi": pool([d["rel_psi"] for d in hb])}
    res["pooled"]["half"] = ph
    return res, markdown(res, data)


def markdown(res: dict, data: dict) -> str:
    L: list[str] = []
    A = L.append
    A("# Opening-hand natural experiment\n")
    A("Effect psi of holding a card in the opening hand, within the same deck, against the model's deck-level card values (beta). Cards with exposure >= "
      f"{MIN_EXPO:.0f}. Generated by `scripts/opening_hand_check.py` (method in its docstring). psi is the log-odds effect of one extra copy in the opening hand "
      "(relative: regressors sum to a constant); kappa is the slope of beta on psi through the origin (precision weighted), tau the between-card SD, flags the cards "
      "with |beta - kappa psi| > 2 sqrt(SE^2 + tau^2).\n")
    A("## Stored opening hand\n")
    A("| set | games | share with exactly 7 cards | cards per game (min-max) | mean lands in hand (0 / 1 / 2+ mulligans) | share of hands with <=1 or >=6 lands (0 / 1 / 2+) |")
    A("|---|---|---|---|---|---|")
    for s in res["sets"]:
        st = data[s][0]["storage"]
        bm = st["by_mull"]
        g = lambda k: " / ".join(f3(bm[m][k], 2) if m in bm else "-" for m in ("0", "1", "2"))
        A(f"| {s} | {st['n_games']:,} | {st['share_7_cards']:.4f} | {st['min_cards']}-{st['max_cards']} | {g('mean_lands')} | {g('share_extreme_lands')} |")
    A("\nEvery game stores exactly 7 opening cards (basics included), whatever num_mulligans is, and the land counts of mulligan games look like those of kept "
      "hands (not like the rejected hands, which would be mostly <=1 or >=6 lands). So the stored hand is the 7 cards of the final deal (before bottoming), "
      "not the kept 7 - num_mulligans cards: all designs centre with hand size 7, and S1 only adds num_mulligans covariates.\n")
    for stage, ref, title in (("draft", "draft", "Draft games vs the draft-stage beta (all events)"), ("draft", "bo1", "Draft games vs the deployed Bo1 beta_total"),
                              ("bo1", "bo1", "Bo1 Sealed games vs the deployed Bo1 beta_total"), ("bo3", "bo3", "Bo3 Sealed games vs the deployed Bo3 beta_total")):
        key = f"{stage}_vs_{ref}"
        A(f"## {title}\n")
        A("E1 (design-centred, primary). E1-E2 = correlation (slope of E2 on E1) of the two estimators on the same cards; SE ratio = median SE(kappa psi) / SE(beta).\n")
        A("| set | games (kept) | cards | Pearson | Spearman | kappa (SE) | tau | flagged | SE ratio | E1-E2 corr (slope) |")
        A("|---|---|---|---|---|---|---|---|---|---|")
        for s in res["sets"]:
            d = res["per_set"][s].get(stage)
            if not d or ref not in d["vs"]:
                continue
            c = d["vs"][ref]["E1"]
            e12 = f"{f3(d.get('e1_e2_corr'), 2)} ({f3(d.get('e1_e2_slope'), 2)})" if ref == stage else "-"
            A(f"| {s} | {d['n_games']:,} ({d['n_kept']:,}) | {c['n']} | {f3(c['pearson'], 2)} | {f3(c['spearman'], 2)} | {f3(c['kappa'])} ({f3(c['kappa_se'])}) | "
              f"{f3(c['tau'])} | {c['n_flag']} | {f3(c['se_ratio'], 2)} | {e12} |")
        pk = res["pooled"][key]["E1"]
        sd_ = lambda p: f3(p["sd"], 2) if p["sd"] is not None else "-"
        e12 = res["pooled"][key]
        A(f"| **pooled mean (SD over {pk['n_sets']} sets)** | | | {f3(pk['pearson']['mean'], 2)} ({sd_(pk['pearson'])}) | {f3(pk['spearman']['mean'], 2)} ({sd_(pk['spearman'])}) | "
          f"{f3(pk['kappa']['mean'])} ({f3(pk['kappa']['sd'])}); IV {f3(pk['kappa_iv']['iv'])} ({f3(pk['kappa_iv']['iv_se'])}) | {f3(pk['tau']['mean'])} | "
          f"{f3(pk['n_flag']['mean'], 1)} | {f3(pk['se_ratio']['mean'], 2)} | {f3(e12['e1_e2_corr']['mean'], 2) if ref == stage else '-'} "
          f"({f3(e12['e1_e2_slope']['mean'], 2) if ref == stage else '-'}) |\n")
        A("Estimator sensitivity (pooled mean over sets):\n")
        A("| estimator | Pearson | Spearman | kappa | SE ratio |")
        A("|---|---|---|---|---|")
        for e, lab in (("E1", "E1 design-centred, prior SD 1.0"), ("E1s03", "E1, prior SD 0.3"), ("E2", "E2 deck fixed effects"), ("S1", "S1 all games, hand size 7, mulligan covariates")):
            q = res["pooled"][key][e]
            A(f"| {lab} | {f3(q['pearson']['mean'], 2)} | {f3(q['spearman']['mean'], 2)} | {f3(q['kappa']['mean'])} | {f3(q['se_ratio']['mean'], 2)} |")
        A("")
    A("## Split-half (draft games, events split A/B with seed 0)\n")
    A("beta = draft-stage fit of each half at the artifact's sigma; psi = E1 on the half's draft games (offset = the half's own fitted logit). "
      "rel = correlation between the halves; cross = psi of one half with beta of the other; disattenuated = mean cross / sqrt(rel_psi rel_beta). "
      "Incremental regressions on cards (WLS, weights 1/SE_target^2, residual-scaled SEs), per direction: (i) beta_T ~ beta_S + psi_S; (ii) psi_T ~ beta_S + psi_S.\n")
    A("| set | cards | rel psi | rel beta | cross psiA-betaB | cross betaA-psiB | disattenuated | (i) psi_S coef (t) A->B | B->A | (ii) beta_S coef (t) A->B | B->A |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    for s in res["sets"]:
        d = res["halves"][s].get("draft")
        if not d:
            continue
        i1, i2 = d["inc"]["A->B"], d["inc"]["B->A"]
        cell = lambda v, k: f"{f3(v[k]['coef'], 3)} ({f3(v[k]['t'], 1)})"
        A(f"| {s} | {d['n']} | {f3(d['rel_psi'], 2)} | {f3(d['rel_beta'], 2)} | {f3(d['cross_psiA_betaB'], 2)} | {f3(d['cross_betaA_psiB'], 2)} | {f3(d['disattenuated'], 2)} | "
          f"{cell(i1, 'psi_in_beta')} | {cell(i2, 'psi_in_beta')} | {cell(i1, 'beta_in_psi')} | {cell(i2, 'beta_in_psi')} |")
    ph = res["pooled"]["half"].get("draft")
    if ph:
        A(f"| **mean (SD)** | | {f3(ph['rel_psi']['mean'], 2)} ({f3(ph['rel_psi']['sd'], 2)}) | {f3(ph['rel_beta']['mean'], 2)} ({f3(ph['rel_beta']['sd'], 2)}) | "
          f"{f3(ph['cross_psiA_betaB']['mean'], 2)} | {f3(ph['cross_betaA_psiB']['mean'], 2)} | {f3(ph['disattenuated']['mean'], 2)} ({f3(ph['disattenuated']['sd'], 2)}) | "
          f"{f3(ph['psi_in_beta_coef']['mean'], 3)} (t {f3(ph['psi_in_beta_t']['mean'], 1)}) | | {f3(ph['beta_in_psi_coef']['mean'], 3)} (t {f3(ph['beta_in_psi_t']['mean'], 1)}) | |\n")
        A(f"Pooled incremental coefficients (inverse-variance over all set x direction fits): psi_S in (i) {f3(ph['psi_in_beta_iv']['iv'], 4)} "
          f"(SE {f3(ph['psi_in_beta_iv']['iv_se'], 4)}); beta_S in (ii) {f3(ph['beta_in_psi_iv']['iv'], 3)} (SE {f3(ph['beta_in_psi_iv']['iv_se'], 3)}).\n")
    hb = res["pooled"]["half"].get("bo1")
    A("Bo1 Sealed games, psi reliability between halves (psi only): " + ", ".join(
        f"{s} {f3(res['halves'][s]['bo1']['rel_psi'], 2)}" for s in res["sets"] if "bo1" in res["halves"][s])
      + (f"; mean {f3(hb['rel_psi']['mean'], 2)} (SD {f3(hb['rel_psi']['sd'], 2)})." if hb else "."))
    A("")
    A("## Illustrations\n")
    A("| set | card | beta draft | beta Bo1 | beta Bo3 | deployed flag Bo1 / Bo3 | stage | psi E1 (SE) | kappa psi | flag |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for r in res["illustrations"]:
        if r.get("missing"):
            A(f"| {r['set']} | {r['card']} | card not found | | | | | | | |")
            continue
        b = lambda f: f"{f3(r['beta'].get(f), 3)} ({f3(r['beta_se'].get(f), 3)})" if f in r["beta"] else "-"
        dfl = " / ".join("yes" if r["deployed_flag"].get(f) else "no" for f in ("bo1", "bo3"))
        for k, (stage, v) in enumerate(r["stage"].items()):
            A(f"| {r['set'] if k == 0 else ''} | {r['card'] if k == 0 else ''} | {b('draft') if k == 0 else ''} | {b('bo1') if k == 0 else ''} | {b('bo3') if k == 0 else ''} | "
              f"{dfl if k == 0 else ''} | {stage} | {f3(v['psi'])} ({f3(v['se'])}) | {f3(v['kappa_psi'])} | {'yes' if v['flag'] else 'no'} |")
    A("")
    A("## Run notes\n")
    for s in res["sets"]:
        i = res["per_set"][s]["_info"]
        rc = i.get("recon", {})
        A(f"- {s}: {i['seconds']:.0f}s; offset chain vs deployed beta (exposure >= {MIN_EXPO:.0f}): "
          + ", ".join(f"{f} corr {f3(v['corr'], 3)}, rms diff {f3(v['rms'], 4)} (deployed rms {f3(v['art_rms'], 4)})" for f, v in rc.items()))
    return "\n".join(L) + "\n"


# =====================================================================================================================

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("sets", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--jobs", type=int, default=1)
    p = sub.add_parser("report")
    p.add_argument("out")
    p.add_argument("--docs", default=str(HERE.parent / "docs" / "opening_hand"))
    a = ap.parse_args(argv)
    if a.cmd == "run":
        out = Path(a.out).resolve()
        for sc in a.sets:
            run_set(sc.upper(), out, a.jobs)
    else:
        res, md = build_report(Path(a.out).resolve())
        d = Path(a.docs)
        d.mkdir(parents=True, exist_ok=True)
        (d / "opening_hand.md").write_text(md, encoding="utf-8")
        (d / "opening_hand.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
        print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
