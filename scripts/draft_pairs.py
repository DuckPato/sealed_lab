"""Do two cards raise each other's win rate beyond their own values?  (descriptive analysis on Draft-stage games; no change to the model)

Method: the Draft events of a set are split in two halves A / B by a fixed hash of draft_id. On each half the app's plain Draft stage
(cards + deck features + pilot skill + covariates) is fitted; its per-game logit is the OFFSET (it already holds each card's own value, the
deck's shape and the pilot's skill, which is what removes the "bomb drags its partners" effect). For every candidate pair (nonland,
nonbasic cards sharing >= MIN_BOTH deck-games in each half) ONE parameter gamma is fitted on top of the offset:

    logit P(win) = offset + gamma * 1[deck has both]          (and a second version with min(copies_A, copies_B))

by Newton steps; the SE is the cluster-robust (by draft_id) sandwich of the one-parameter model. Pairs are selected on half A by
Benjamini-Hochberg (q < 0.10) and a pair counts as replicated only if gamma on half B (offset from the fit on B) has the same sign and
|z| > 2. Reads the app home and the raw files strictly read-only (SEALED_LAB_HOME must point at a scratch copy); writes only to OUT.

    python scripts/draft_pairs.py OUT [--sets ...] [--jobs N] [--report]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp
from scipy.special import expit
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from sealed_lab import config  # noqa: E402
from sealed_lab.model import cardvalue, design, glm, validate  # noqa: E402
import plat_draft_check as pdc  # noqa: E402  (prepare_set, fit_chain, snapshot, compare_snapshots)
import play_prior_check as ppc  # noqa: E402

DEFAULT_SETS = ["FRA", "HOB", "MSH", "SOS", "TMT"]
MIN_BOTH = 1500
Q = 0.10
Z_REP = 2.0
PP = 0.55 * 0.45 * 100.0
_json_default = ppc._json_default


# ------------------------------------------------------------------------------------------------------ helpers
def half_of(draft_ids) -> np.ndarray:
    """0 = A, 1 = B: crc32 of the draft_id (fixed, platform independent)."""
    return np.array([zlib.crc32(str(d).encode()) & 1 for d in draft_ids], dtype=np.int8)


def bh(p: np.ndarray, q: float) -> np.ndarray:
    """Benjamini-Hochberg: boolean mask of the rejected hypotheses at level q."""
    n = len(p)
    o = np.argsort(p)
    ok = p[o] <= q * (np.arange(1, n + 1) / n)
    k = np.nonzero(ok)[0]
    m = np.zeros(n, dtype=bool)
    if len(k):
        m[o[:k.max() + 1]] = True
    return m


def one_param(eta: np.ndarray, y: np.ndarray, x: np.ndarray, grp: np.ndarray, n_grp: int) -> tuple[float, float, float]:
    """(gamma, cluster SE, model SE) of logit P = eta + gamma x on the given rows (x != 0 only matters)."""
    g = 0.0
    for _ in range(8):
        p = expit(eta + g * x)
        s = float(np.dot(x, y - p))
        i = float(np.dot(x * x, p * (1 - p)))
        if i <= 0:
            return 0.0, float("inf"), float("inf")
        step = s / i
        g += step
        if abs(step) < 1e-7:
            break
    p = expit(eta + g * x)
    i = float(np.dot(x * x, p * (1 - p)))
    sc = np.bincount(grp, weights=x * (y - p), minlength=n_grp)
    var = float(np.dot(sc, sc)) / (i * i)
    return g, math.sqrt(var), 1.0 / math.sqrt(i)


class Half:
    """One half's Draft stage: fit, offset, deck-game incidence."""

    def __init__(self, tag: str, games: pl.DataFrame, space, tmp: Path, n_jobs: int, seed: int):
        self.tag = tag
        t0 = time.time()
        st = design.build_stage("draft", games, space)
        fits = pdc.fit_chain({"draft": st}, "draft", space, tmp, n_jobs, seed, cardvalue.ChainOpts())
        self.fit = fits["draft"]
        self.sigma = float(self.fit.sigma)
        self.converged = bool(self.fit.converged)
        self.eta = st.prob.eta(self.fit.coef)
        self.y = st.prob.y
        self.n_games = st.n_games
        self.n_events = st.n_groups
        self.grp = st.group.astype(np.int64)
        self.beta = self.fit.beta_total
        urow = st.deck_rows[st.prob.idx]
        V = space.D[:, :space.n_card].tocsr()[urow].tocsc()          # games x cards (copies)
        self.V = V
        self.B = (V > 0).astype(np.float64).tocsc()
        self.games = st.games
        self.seconds_fit = time.time() - t0
        self.logloss = float(np.mean(np.logaddexp(0, self.eta) - self.y * self.eta))
        self.logloss_null = float(-np.mean(self.y * np.log(self.y.mean()) + (1 - self.y) * np.log(1 - self.y.mean())))

    def cooc(self) -> np.ndarray:
        return np.asarray((self.B.T @ self.B).todense())

    def rows_of(self, c: int) -> np.ndarray:
        return self.B.indices[self.B.indptr[c]:self.B.indptr[c + 1]]

    def pair(self, a: int, b: int) -> dict:
        ra, rb = self.rows_of(a), self.rows_of(b)
        r, ia, ib = np.intersect1d(ra, rb, assume_unique=True, return_indices=True)
        if len(r) < 5:
            return dict(n=len(r), g=float("nan"), se=float("nan"), z=float("nan"), gm=float("nan"), sem=float("nan"), wr=float("nan"))
        eta, y, grp = self.eta[r], self.y[r], self.grp[r]
        g, se, _ = one_param(eta, y, np.ones(len(r)), grp, self.n_events)
        P = self.V.indptr
        xm = np.minimum(self.V.data[P[a]:P[a + 1]][ia], self.V.data[P[b]:P[b + 1]][ib])
        gm, sem, _ = one_param(eta, y, xm, grp, self.n_events)
        return dict(n=len(r), g=g, se=se, z=g / se if se > 0 else float("nan"), gm=gm, sem=sem, wr=float(y.mean()))

    def wr_with(self, c: int, other: int) -> tuple[float, float, int, int]:
        """win rate of games with card c: (with other present, other absent), and the game counts."""
        rc = self.rows_of(c)
        ro = self.rows_of(other)
        m = np.isin(rc, ro, assume_unique=True)
        w, wo = self.y[rc[m]], self.y[rc[~m]]
        return float(w.mean()), float(wo.mean()), int(len(w)), int(len(wo))


def color_label(c) -> str:
    return "".join(sorted(c)) if c else "C"


def card_info(cards: pl.DataFrame) -> dict[int, dict]:
    out = {}
    for r in cards.iter_rows(named=True):
        t = r.get("types") or ""
        sub = set()
        if " - " in t:
            sub = set(t.split(" - ", 1)[1].split())
        elif " — " in t:
            sub = set(t.split(" — ", 1)[1].split())
        kw = {k for k in (r.get("keywords") or "").split(",") if k}
        out[int(r["card_id"])] = dict(name=r["name"], rarity=r["rarity"], colors=color_label(r.get("colors")), types=t, sub=sub, kw=kw,
                                      is_land=bool(r.get("is_land")), is_creature=bool(r.get("is_creature")),
                                      oracle=(r.get("oracle_text") or ""))
    return out


def pair_kind(ia: dict, ib: dict) -> list[str]:
    kinds = []
    ca, cb = ia["colors"], ib["colors"]
    if ca == cb and ca != "C":
        kinds.append("same colour")
    elif "C" in (ca, cb):
        kinds.append("colourless involved")
    elif len(ca) == 1 and len(cb) == 1:
        kinds.append("two-colour pair")
    else:
        kinds.append("multicolour involved")
    sub = (ia["sub"] & ib["sub"]) - {"Human", "Creature", "Warrior", "Wizard"}
    if sub:
        kinds.append("shared type: " + "/".join(sorted(sub)))
    if ia["kw"] & ib["kw"]:
        kinds.append("shared keyword: " + "/".join(sorted(ia["kw"] & ib["kw"])))
    return kinds


# ------------------------------------------------------------------------------------------------------ one set
def shipped_pairs(sc: str, info: dict, A: Half, B: Half, col_of: np.ndarray) -> dict:
    d = config.model_dir(sc)
    pf = d / "pairs.npz"
    if not pf.exists():
        return dict(present=False)
    z = np.load(pf, allow_pickle=False)
    dr = np.load(d / "draws.npz", allow_pickle=False)
    ab = z["ab"].astype(int)
    res = dict(present=True, n_pairs=int(len(ab)), formats={})
    mean_by = {}
    for f in ("bo1", "bo3"):
        k = f"{f}__pair"
        if k not in dr.files:
            continue
        c = dr[k].astype(float)                    # (draws, pairs), logit per unit of min copies
        m, s = c.mean(0), c.std(0, ddof=1)
        mean_by[f] = (m, s)
    if not mean_by:
        return dict(present=True, n_pairs=int(len(ab)), formats={}, note="no pair coefficients in the draws")
    fmain = "bo1" if "bo1" in mean_by else next(iter(mean_by))
    m, s = mean_by[fmain]
    order = np.argsort(-np.abs(m / np.maximum(s, 1e-9)))[:25]
    rows = []
    for j in order:
        a, b = int(ab[j, 0]), int(ab[j, 1])
        row = dict(a=info[a]["name"], b=info[b]["name"], colors=f"{info[a]['colors']}/{info[b]['colors']}",
                   games=float(z["games"][j]), lift=float(z["lift"][j]))
        for f, (mm, ss) in mean_by.items():
            row[f] = dict(coef_pp=float(mm[j] * PP), lo_pp=float((mm[j] - 1.96 * ss[j]) * PP), hi_pp=float((mm[j] + 1.96 * ss[j]) * PP))
        for tag, H in (("A", A), ("B", B)):
            if col_of[a] >= 0 and col_of[b] >= 0:
                r = H.pair(int(col_of[a]), int(col_of[b]))
                row[tag] = dict(n=r["n"], gamma_pp=r["g"] * PP, lo_pp=(r["g"] - 1.96 * r["se"]) * PP, hi_pp=(r["g"] + 1.96 * r["se"]) * PP,
                                gamma_min_pp=r["gm"] * PP, z=r["z"])
        rows.append(row)
    # correlation of shipped coefficients with my gamma (min copies) on the pairs where both halves have >= 200 deck-games
    ms, ga, gb = [], [], []
    for j in range(len(ab)):
        a, b = int(ab[j, 0]), int(ab[j, 1])
        if col_of[a] < 0 or col_of[b] < 0:
            continue
        ms.append(m[j])
        ga.append((a, b))
    sel = np.random.default_rng(0).choice(len(ms), size=min(600, len(ms)), replace=False)
    xs, ys_a, ys_b = [], [], []
    for i in sel:
        a, b = ga[i]
        ra, rb = A.pair(int(col_of[a]), int(col_of[b])), B.pair(int(col_of[a]), int(col_of[b]))
        if ra["n"] >= 300 and rb["n"] >= 300:
            xs.append(ms[i])
            ys_a.append(ra["gm"])
            ys_b.append(rb["gm"])
    corr = {}
    if len(xs) > 10:
        corr = dict(n=len(xs), corr_shipped_vs_gamma_A=float(np.corrcoef(xs, ys_a)[0, 1]), corr_shipped_vs_gamma_B=float(np.corrcoef(xs, ys_b)[0, 1]),
                    corr_gammaA_vs_gammaB=float(np.corrcoef(ys_a, ys_b)[0, 1]))
    return dict(present=True, n_pairs=int(len(ab)), format_ranked=fmain, top=rows, corr=corr)


def run_set(sc: str, out: Path, n_jobs: int) -> dict:
    t0 = time.time()
    inp, space, _formats, _ed = pdc.prepare_set(sc, n_jobs)
    gd = inp.games.filter(pl.col("fmt") == "draft")
    ids = sorted(gd["draft_id"].unique().to_list())
    hmap = dict(zip(ids, half_of(ids)))
    hh = np.array([hmap[d] for d in gd["draft_id"].to_list()])
    gA, gB = gd.filter(pl.Series(hh == 0)), gd.filter(pl.Series(hh == 1))
    print(f"[{sc}] draft games {gd.height} in {len(ids)} events: A {gA.height}, B {gB.height}", flush=True)
    info = card_info(inp.cards)
    nb = np.asarray(space.nb_ids)
    col_of = space.col_of_card
    tmp = out / "tmp" / sc
    A = Half("A", gA, space, tmp / "a", n_jobs, 1)
    print(f"[{sc}] fit A: sigma {A.sigma:.3f} conv {A.converged} ll {A.logloss:.4f} (null {A.logloss_null:.4f}) {A.seconds_fit:.0f}s", flush=True)
    B = Half("B", gB, space, tmp / "b", n_jobs, 2)
    print(f"[{sc}] fit B: sigma {B.sigma:.3f} conv {B.converged} ll {B.logloss:.4f} (null {B.logloss_null:.4f}) {B.seconds_fit:.0f}s", flush=True)
    cardvalue.cleanup(str(out / "tmp"))

    # candidates
    nonland = np.array([not info[int(c)]["is_land"] for c in nb])
    cA, cB = A.cooc(), B.cooc()
    ii, jj = np.nonzero(np.triu((cA >= MIN_BOTH) & (cB >= MIN_BOTH), 1))
    keep = nonland[ii] & nonland[jj]
    ii, jj = ii[keep], jj[keep]
    print(f"[{sc}] candidates {len(ii)} (>= {MIN_BOTH} deck-games in each half)", flush=True)
    resA = [A.pair(int(a), int(b)) for a, b in zip(ii, jj)]
    zA = np.array([r["z"] for r in resA])
    pA = 2 * norm.sf(np.abs(zA))
    sel = bh(pA, Q)
    resB = [B.pair(int(a), int(b)) for a, b in zip(ii, jj)]
    zB = np.array([r["z"] for r in resB])
    sgn = np.sign(zA)
    rep = sel & (sgn * zB > Z_REP)
    # null check on pairs not selected on A
    ns = ~sel
    zb_null = zB[ns]
    null = dict(n=int(ns.sum()), mean=float(np.nanmean(zb_null)), sd=float(np.nanstd(zb_null, ddof=1)),
                frac_abs_gt2=float(np.nanmean(np.abs(zb_null) > 2)), frac_expected_normal=float(2 * norm.sf(2)))
    # all-pair replication independent of selection, for context: corr of zA and zB
    ok = np.isfinite(zA) & np.isfinite(zB)
    corr_z = float(np.corrcoef(zA[ok], zB[ok])[0, 1])
    gA = np.array([r["g"] for r in resA]); gB = np.array([r["g"] for r in resB])
    sA = np.array([r["se"] for r in resA]); sB = np.array([r["se"] for r in resB])
    w = 1.0 / (sA * sB)
    mean_g = float(np.sum(w * 0.5 * (gA + gB)) / np.sum(w))
    cov_raw = float(np.sum(w * gA * gB) / np.sum(w))
    cov_cen = float(np.sum(w * (gA - mean_g) * (gB - mean_g)) / np.sum(w))
    het = dict(mean_gamma_pp=mean_g * PP, tau_pp_raw=math.sqrt(max(cov_raw, 0)) * PP, tau_pp_centred=math.sqrt(max(cov_cen, 0)) * PP,
               median_se_pp=float(np.median(sA) * PP), p10_se_pp=float(np.quantile(sA, 0.1) * PP), p90_se_pp=float(np.quantile(sA, 0.9) * PP))
    nom = (np.abs(zA) > 3)
    nom_rep = nom & (sgn * zB > Z_REP)
    nominal = dict(n_absZA_gt3=int(nom.sum()), replicated=int(nom_rep.sum()), replicated_pos=int((nom_rep & (zA > 0)).sum()),
                   replicated_neg=int((nom_rep & (zA < 0)).sum()),
                   expected_by_chance=float(len(zA) * 2 * norm.sf(3) * norm.sf(Z_REP)),
                   n_absZA_gt2=int((np.abs(zA) > 2).sum()), replicated_gt2=int(((np.abs(zA) > 2) & (sgn * zB > Z_REP)).sum()),
                   expected_gt2=float(len(zA) * 2 * norm.sf(2) * norm.sf(Z_REP)))
    nom_idx = sorted(np.nonzero(nom_rep)[0].tolist(), key=lambda k: -(abs(zA[k]) + abs(zB[k])))
    nominal["kinds_pos"] = None
    exp_chance = float(sel.sum() * norm.sf(Z_REP))      # one-sided (same sign) tail under the null
    n_pos_sel, n_neg_sel = int((sel & (zA > 0)).sum()), int((sel & (zA < 0)).sum())
    n_pos_rep, n_neg_rep = int((rep & (zA > 0)).sum()), int((rep & (zA < 0)).sum())

    def entry(k: int) -> dict:
        a, b = int(nb[ii[k]]), int(nb[jj[k]])
        ra, rb = resA[k], resB[k]
        # raw lift: win rate of the decks with both minus the mean of the two single-card win rates (one without the other)
        def raw(H):
            rows_a, rows_b = H.rows_of(int(ii[k])), H.rows_of(int(jj[k]))
            both = np.intersect1d(rows_a, rows_b, assume_unique=True)
            oa = np.setdiff1d(rows_a, both, assume_unique=True)
            ob = np.setdiff1d(rows_b, both, assume_unique=True)
            return float(H.y[both].mean() - 0.5 * (H.y[oa].mean() + H.y[ob].mean())) * 100.0
        d = dict(a=info[a]["name"], b=info[b]["name"], a_id=a, b_id=b, rarity=f"{info[a]['rarity']}/{info[b]['rarity']}",
                 colors=f"{info[a]['colors']}/{info[b]['colors']}", kinds=pair_kind(info[a], info[b]),
                 types=f"{info[a]['types']} | {info[b]['types']}",
                 n_both_A=ra["n"], n_both_B=rb["n"],
                 gamma_pp_A=ra["g"] * PP, lo_A=(ra["g"] - 1.96 * ra["se"]) * PP, hi_A=(ra["g"] + 1.96 * ra["se"]) * PP, z_A=ra["z"],
                 gamma_pp_B=rb["g"] * PP, lo_B=(rb["g"] - 1.96 * rb["se"]) * PP, hi_B=(rb["g"] + 1.96 * rb["se"]) * PP, z_B=rb["z"],
                 gamma_min_pp_A=ra["gm"] * PP, gamma_min_pp_B=rb["gm"] * PP,
                 raw_lift_pp_A=raw(A), raw_lift_pp_B=raw(B),
                 beta_own_pp=[float(A.beta[ii[k]] * PP), float(A.beta[jj[k]] * PP)])
        return d

    rep_idx = np.nonzero(rep)[0]
    replicated = sorted((entry(int(k)) for k in rep_idx), key=lambda d: -(d["gamma_pp_A"] + d["gamma_pp_B"]))
    sel_not_rep = int((sel & ~rep).sum())
    # kind breakdown of replicated pairs vs all candidates
    def kind_frac(idx):
        cnt: dict[str, int] = {}
        for k in idx:
            a, b = int(nb[ii[k]]), int(nb[jj[k]])
            for kd in pair_kind(info[a], info[b]):
                key = kd.split(":")[0]
                cnt[key] = cnt.get(key, 0) + 1
        n = max(len(idx), 1)
        return {k: v / n for k, v in sorted(cnt.items())}, len(idx)
    kinds_rep = kind_frac(rep_idx)
    kinds_all = kind_frac(range(len(ii)))

    # strongest A-selected pairs that did not replicate (top 10 by zA), for context
    nr_idx = [int(k) for k in np.nonzero(sel & ~rep)[0]]
    nr_idx.sort(key=lambda k: -abs(zA[k]))
    not_replicated = [entry(k) for k in nr_idx[:10]]

    # sanity (a): top rated rares/mythics with the common/uncommon filler they most often share decks with
    sanity = []
    rar = np.array([info[int(c)]["rarity"] in ("rare", "mythic") and not info[int(c)]["is_land"] for c in nb])
    expo = np.asarray(A.B.sum(0)).ravel()
    ok_b = np.nonzero(rar & (expo >= 5000))[0]
    top_b = ok_b[np.argsort(-A.beta[ok_b])][:3]
    com = np.array([info[int(c)]["rarity"] == "common" and not info[int(c)]["is_land"] for c in nb])
    for bcol in top_b:
        cand = [c for c in np.nonzero(com)[0] if cA[bcol, c] >= MIN_BOTH]
        cand.sort(key=lambda c: -cA[bcol, c] / max(cA[c, c], 1))      # the partner that most often shares a deck with the bomb
        for c in cand[:2]:
            ra, rb = A.pair(int(bcol), int(c)), B.pair(int(bcol), int(c))
            w1a, w0a, n1a, n0a = A.wr_with(int(c), int(bcol))
            w1b, w0b, n1b, n0b = B.wr_with(int(c), int(bcol))
            sanity.append(dict(bomb=info[int(nb[bcol])]["name"], bomb_own_pp=float(A.beta[bcol] * PP), partner=info[int(nb[c])]["name"],
                               partner_own_pp=float(A.beta[c] * PP), share_of_partner_decks_with_bomb=float(cA[bcol, c] / cA[c, c]),
                               n_both_A=ra["n"],
                               partner_wr_with_bomb_minus_without_pp_A=(w1a - w0a) * 100, partner_wr_with_bomb_minus_without_pp_B=(w1b - w0b) * 100,
                               gamma_pp_A=ra["g"] * PP, ci_A=[(ra["g"] - 1.96 * ra["se"]) * PP, (ra["g"] + 1.96 * ra["se"]) * PP],
                               gamma_pp_B=rb["g"] * PP, ci_B=[(rb["g"] - 1.96 * rb["se"]) * PP, (rb["g"] + 1.96 * rb["se"]) * PP]))
    nominal_pairs = [entry(int(k)) for k in nom_idx[:30]]
    ship = shipped_pairs(sc, info, A, B, col_of)
    res = dict(set=sc, seconds=round(time.time() - t0, 1), n_events=len(ids), n_games_A=int(A.n_games), n_games_B=int(B.n_games),
               events_A=int(A.n_events), events_B=int(B.n_events), sigma=dict(A=A.sigma, B=B.sigma),
               converged=dict(A=A.converged, B=B.converged),
               logloss=dict(A=A.logloss, B=B.logloss, null_A=A.logloss_null, null_B=B.logloss_null),
               n_candidates=int(len(ii)), selected_A=int(sel.sum()), selected_pos=n_pos_sel, selected_neg=n_neg_sel,
               replicated=int(rep.sum()), replicated_pos=n_pos_rep, replicated_neg=n_neg_rep, expected_by_chance=exp_chance,
               selected_not_replicated=sel_not_rep, corr_zA_zB_all_candidates=corr_z,
               null_zB_unselected=null, heterogeneity=het, nominal_tier=nominal,
               z_A_quantiles=[float(np.nanquantile(zA, q)) for q in (0.01, 0.05, 0.5, 0.95, 0.99)],
               kinds_replicated=dict(frac=kinds_rep[0], n=kinds_rep[1]), kinds_all_candidates=dict(frac=kinds_all[0], n=kinds_all[1]),
               replicated_pairs=replicated, nominal_pairs=nominal_pairs, strongest_not_replicated=not_replicated, sanity_bombs=sanity, shipped=ship)
    print(f"[{sc}] cand {len(ii)} sel {int(sel.sum())} (+{n_pos_sel}/-{n_neg_sel}) rep {int(rep.sum())} (+{n_pos_rep}/-{n_neg_rep}) "
          f"chance {exp_chance:.1f} | null zB mean {null['mean']:+.3f} sd {null['sd']:.3f} | {res['seconds']:.0f}s", flush=True)
    return res


# ------------------------------------------------------------------------------------------------------ report
def write_md(summary: dict, path: Path) -> None:
    L = ["# Card pairs on top of each card's own value (Draft-stage games)", "",
         f"Pairs: nonland, nonbasic cards sharing >= {MIN_BOTH} deck-games in each half. gamma in pp of game win rate at a 55% deck (logit x 24.75). "
         f"Selection: Benjamini-Hochberg q < {Q} on half A; replicated = same sign and |z| > {Z_REP} on half B (offset from the Draft fit on B).", ""]
    for sc, r in summary["per_set"].items():
        L += [f"## {sc}", "",
              f"- Draft games A {r['n_games_A']:,} / B {r['n_games_B']:,} ({r['events_A']:,} / {r['events_B']:,} events); Draft sigma A {r['sigma']['A']:.3f}, B {r['sigma']['B']:.3f}",
              f"- candidates {r['n_candidates']}, selected on A {r['selected_A']} (+{r['selected_pos']} / -{r['selected_neg']}), "
              f"replicated on B **{r['replicated']}** (+{r['replicated_pos']} / -{r['replicated_neg']}); expected by chance among the selected: {r['expected_by_chance']:.1f}",
              f"- unselected pairs, z on B: mean {r['null_zB_unselected']['mean']:+.3f}, sd {r['null_zB_unselected']['sd']:.3f}, "
              f"share |z|>2 {r['null_zB_unselected']['frac_abs_gt2']:.3f} (normal: {r['null_zB_unselected']['frac_expected_normal']:.3f}); "
              f"corr(z_A, z_B) over all candidates {r['corr_zA_zB_all_candidates']:+.3f}",
              f"- kinds among replicated: {r['kinds_replicated']['frac']} (n={r['kinds_replicated']['n']}); all candidates: "
              f"{ {k: round(v, 2) for k, v in r['kinds_all_candidates']['frac'].items()} }", ""]
        if r["replicated_pairs"]:
            L += ["| A | B | colours | gamma A pp [CI] | gamma B pp [CI] | both (A/B) | raw lift pp (A/B) | own value pp | kind |", "|---|---|---|---|---|---|---|---|---|"]
            for d in r["replicated_pairs"][:40]:
                L.append(f"| {d['a']} | {d['b']} | {d['colors']} | {d['gamma_pp_A']:+.2f} [{d['lo_A']:+.2f},{d['hi_A']:+.2f}] | "
                         f"{d['gamma_pp_B']:+.2f} [{d['lo_B']:+.2f},{d['hi_B']:+.2f}] | {d['n_both_A']:,}/{d['n_both_B']:,} | "
                         f"{d['raw_lift_pp_A']:+.2f}/{d['raw_lift_pp_B']:+.2f} | {d['beta_own_pp'][0]:+.1f}/{d['beta_own_pp'][1]:+.1f} | {'; '.join(d['kinds'])} |")
            L.append("")
        else:
            L += ["No pair replicated.", ""]
        h, nm = r["heterogeneity"], r["nominal_tier"]
        L += [f"- looser tier (no BH): |z_A| > 3: {nm['n_absZA_gt3']} pairs, replicated {nm['replicated']} (+{nm['replicated_pos']}/-{nm['replicated_neg']}) vs {nm['expected_by_chance']:.1f} by chance; "
              f"|z_A| > 2: {nm['n_absZA_gt2']} pairs, replicated {nm['replicated_gt2']} vs {nm['expected_gt2']:.1f} by chance",
              f"- mean gamma over candidates {h['mean_gamma_pp']:+.2f} pp; SD of true pair effects (from cov of the two halves) {h['tau_pp_raw']:.2f} pp raw / {h['tau_pp_centred']:.2f} pp centred; SE per pair median {h['median_se_pp']:.2f} pp (10-90%: {h['p10_se_pp']:.2f}-{h['p90_se_pp']:.2f})", ""]
        if r["nominal_pairs"]:
            L += ["Looser tier (|z_A| > 3 and same sign z_B > 2, no multiple-testing control; exploratory), top by |z_A|+|z_B|:", "",
                  "| A | B | colours | gamma A pp [CI] | gamma B pp [CI] | both (A/B) | raw lift pp (A/B) | kind |", "|---|---|---|---|---|---|---|---|"]
            for d in r["nominal_pairs"][:15]:
                L.append(f"| {d['a']} | {d['b']} | {d['colors']} | {d['gamma_pp_A']:+.2f} [{d['lo_A']:+.2f},{d['hi_A']:+.2f}] | "
                         f"{d['gamma_pp_B']:+.2f} [{d['lo_B']:+.2f},{d['hi_B']:+.2f}] | {d['n_both_A']:,}/{d['n_both_B']:,} | "
                         f"{d['raw_lift_pp_A']:+.2f}/{d['raw_lift_pp_B']:+.2f} | {'; '.join(d['kinds'])} |")
            L.append("")
        L += ["Sanity (bomb + common it often shares decks with):", "",
              "| bomb | partner | partner decks with bomb | partner WR with-minus-without bomb pp (A/B) | gamma pp A [CI] | gamma pp B [CI] |", "|---|---|---|---|---|---|"]
        for s in r["sanity_bombs"]:
            L.append(f"| {s['bomb']} ({s['bomb_own_pp']:+.1f}) | {s['partner']} ({s['partner_own_pp']:+.1f}) | {s['share_of_partner_decks_with_bomb']:.0%} | "
                     f"{s['partner_wr_with_bomb_minus_without_pp_A']:+.2f} / {s['partner_wr_with_bomb_minus_without_pp_B']:+.2f} | "
                     f"{s['gamma_pp_A']:+.2f} [{s['ci_A'][0]:+.2f},{s['ci_A'][1]:+.2f}] | {s['gamma_pp_B']:+.2f} [{s['ci_B'][0]:+.2f},{s['ci_B'][1]:+.2f}] |")
        L.append("")
        sh = r["shipped"]
        if sh.get("present") and sh.get("top"):
            L += [f"Shipped synergy layer ({sh['n_pairs']} pairs; coefficients in pp, +-1.96 sd over the draws; ranked by |mean/sd| in {sh['format_ranked']}); my gamma where both cards are in the Draft space:", "",
                  "| A | B | shipped bo1 pp [CI] | shipped bo3 pp [CI] | mine A pp [CI] | mine B pp [CI] |", "|---|---|---|---|---|---|"]
            for t in sh["top"][:15]:
                def f(x):
                    return f"{x['coef_pp']:+.2f} [{x['lo_pp']:+.2f},{x['hi_pp']:+.2f}]" if x else "-"
                def g(x):
                    return f"{x['gamma_pp']:+.2f} [{x['lo_pp']:+.2f},{x['hi_pp']:+.2f}] (n={x['n']:,})" if x else "-"
                L.append(f"| {t['a']} | {t['b']} | {f(t.get('bo1'))} | {f(t.get('bo3'))} | {g(t.get('A'))} | {g(t.get('B'))} |")
            if sh.get("corr"):
                L += ["", f"Correlation over {sh['corr']['n']} random shipped pairs with >= 300 deck-games per half: shipped coefficient vs gamma(min copies) A {sh['corr']['corr_shipped_vs_gamma_A']:+.3f}, "
                      f"B {sh['corr']['corr_shipped_vs_gamma_B']:+.3f}; gamma A vs gamma B {sh['corr']['corr_gammaA_vs_gammaB']:+.3f}."]
            L.append("")
        elif not sh.get("present"):
            L += ["No shipped synergy layer for this set (no pairs.npz).", ""]
    path.write_text("\n".join(L), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--sets", nargs="+", default=DEFAULT_SETS)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--report", action="store_true", help="only rebuild pairs.json / pairs.md from the per-set files")
    a = ap.parse_args(argv)
    pdc.check_home_not_real()
    out = a.out.resolve()
    pdc.check_out_not_home(out)
    out.mkdir(parents=True, exist_ok=True)
    sets = [s.upper() for s in a.sets]
    before = pdc.snapshot(sets)
    per_set: dict[str, dict] = {}
    t_all = time.time()
    with glm.shared_pool(a.jobs):
        for sc in sets:
            done = out / f"per_set_{sc}.json"
            if done.exists():
                per_set[sc] = json.loads(done.read_text(encoding="utf-8"))
                print(f"[{sc}] reused {done.name}", flush=True)
                continue
            if a.report:
                continue
            per_set[sc] = run_set(sc, out, a.jobs)
            done.write_text(json.dumps(per_set[sc], indent=1, default=_json_default), encoding="utf-8")
    untouched = pdc.compare_snapshots(before, pdc.snapshot(sets))
    summary = dict(created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), total_seconds=round(time.time() - t_all, 1),
                   min_both=MIN_BOTH, q=Q, z_rep=Z_REP, sets=sets, home_untouched=untouched, per_set=per_set)
    (out / "pairs.json").write_text(json.dumps(summary, indent=1, default=_json_default), encoding="utf-8")
    write_md(summary, out / "pairs.md")
    print(f"wrote pairs.json / pairs.md in {summary['total_seconds']:.0f}s; changed: {{k: v['n_changed'] for k, v in untouched.items()}}"
          .replace("{k: v['n_changed'] for k, v in untouched.items()}", str({k: v['n_changed'] for k, v in untouched.items()})), flush=True)
    return 0


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    sys.exit(main())
