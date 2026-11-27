"""Descriptive diagnostic: does leaving out the pilot's Arena RANK bend the card values of the Draft stage?

Arena pairs Premier Draft players largely by rank, so at the same 17Lands skill a Mythic player meets harder opponents than a Silver
one. The card-value model controls for skill but not rank (Sealed has no rank). On Draft games (PremierDraft, PickTwoDraft; TradDraft has
no rank) this script fits the Draft stage twice on the SAME games:

  BASE  the shipped first-stage fit (zero prior, card sigma chosen by CV)
  RANK  the same stage plus 5 rank indicators (silver, gold, platinum, diamond, mythic; bronze = reference; prior SD 1.0) appended to
        the covariate block, the card sigma FIXED at BASE's choice (so only the covariate changes)

and reports the rank coefficients, the card-value shift (RANK - BASE) and its slope against the card's high-rank share. Nothing is
shipped or decided here; it is a diagnostic.

Reads the app home strictly read-only (SEALED_LAB_HOME must point at a scratch copy); writes only to OUT.

    python scripts/rank_bias_check.py OUT [--sets ...] [--jobs N]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from sealed_lab import config  # noqa: E402
from sealed_lab.model import cardvalue, castability, design, glm, pipeline  # noqa: E402
import cold_start_check as csc  # noqa: E402  (read_phi_readonly)

ALL_SETS = ["ECL", "EOE", "FIN", "FRA", "HOB", "MSH", "OM1", "SOS", "TDM", "TLA", "TMT"]
RANKS = ["bronze", "silver", "gold", "platinum", "diamond", "mythic"]
RANK_DUMMIES = RANKS[1:]
HIGH = ("platinum", "diamond", "mythic")
RANK_SIGMA = 1.0
PP = 0.55 * 0.45 * 100.0            # logit -> percentage points at a 55% deck (0.2475 * 100)
TOP_K = 15
TRACKED = ["The Ten Rings", "Meek Attack", "Glimpse of Nature", "Key to the Side-Door", "Dragonstorm Forecaster",
           "Broadcast Takeover", "Crescent Island Temple", "Northern Air Temple"]
GAME_COLS_EXT = pipeline.GAME_COLS + ["rank"]


# ------------------------------------------------------------------------------------------------------ safety
def check_home_not_real() -> None:
    """Refuse to run against the owner's real app home (%LOCALAPPDATA%/sealed_lab)."""
    home = config.app_home().resolve()
    la = os.environ.get("LOCALAPPDATA")
    if la and home == (Path(la) / "sealed_lab").resolve():
        raise RuntimeError(f"refusing to run on the real app home {home}: set SEALED_LAB_HOME to a scratch copy")


def check_out_not_home(out: Path) -> None:
    home = config.app_home().resolve()
    out_r = out.resolve()
    if out_r == home or home in out_r.parents:
        raise RuntimeError(f"refusing to write inside app home {home}")


# ------------------------------------------------------------------------------------------------------ pure helpers
def rank_indicators(rank) -> np.ndarray:
    """(n, 5) indicators silver, gold, platinum, diamond, mythic of a rank column (bronze = all zeros)."""
    r = np.array(["" if x is None else str(x).lower() for x in rank], dtype=object)
    return np.column_stack([(r == t).astype(float) for t in RANK_DUMMIES]) if len(r) else np.zeros((0, len(RANK_DUMMIES)))


def wls_slope(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> dict:
    """Weighted least squares y = a + b x. Weights are rescaled to mean 1; SE of b = sqrt(s2 / Sxx), s2 = sum w r^2 / (n - 2).
    Also the weighted correlation. Rows with a non-finite value or non-positive weight are dropped."""
    x, y, w = (np.asarray(v, dtype=float) for v in (x, y, w))
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    n = int(m.sum())
    if n < 3:
        return dict(n=n, slope=float("nan"), se=float("nan"), intercept=float("nan"), corr=float("nan"))
    x, y, w = x[m], y[m], w[m]
    w = w / w.mean()
    xb, yb = np.average(x, weights=w), np.average(y, weights=w)
    sxx = float(np.sum(w * (x - xb) ** 2))
    syy = float(np.sum(w * (y - yb) ** 2))
    sxy = float(np.sum(w * (x - xb) * (y - yb)))
    if sxx <= 1e-300:
        return dict(n=n, slope=float("nan"), se=float("nan"), intercept=float("nan"), corr=float("nan"))
    b = sxy / sxx
    a = yb - b * xb
    s2 = float(np.sum(w * (y - a - b * x) ** 2)) / (n - 2)
    corr = sxy / math.sqrt(sxx * syy) if syy > 1e-300 else float("nan")
    return dict(n=n, slope=float(b), se=float(math.sqrt(s2 / sxx)), intercept=float(a), corr=float(corr))


def pool_inverse_variance(vals: list[float], ses: list[float]) -> dict:
    """Fixed-effect inverse-variance mean and its SE over the finite cells with a positive SE."""
    v = np.array(vals, dtype=float)
    s = np.array(ses, dtype=float)
    ok = np.isfinite(v) & np.isfinite(s) & (s > 0)
    if not ok.any():
        return dict(n=0, mean=float("nan"), se=float("nan"))
    wt = 1.0 / s[ok] ** 2
    return dict(n=int(ok.sum()), mean=float(np.sum(wt * v[ok]) / wt.sum()), se=float(1.0 / math.sqrt(wt.sum())))


def high_rank_share(Dcards: sp.csr_matrix, idx: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(share, n_games) per card: Dcards = (unique decks x cards) copies, idx = game -> unique deck row, hi = 0/1 per game.
    A game counts for a card when its deck contains the card (copies > 0); share = high-rank games / such games (NaN when none)."""
    B = (Dcards > 0).astype(float).tocsr()
    m = B.shape[0]
    n_deck = np.bincount(idx, minlength=m).astype(float)
    hi_deck = np.bincount(idx, weights=hi, minlength=m)
    n_c = np.asarray(B.T @ n_deck).ravel()
    h_c = np.asarray(B.T @ hi_deck).ravel()
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(n_c > 0, h_c / n_c, np.nan)
    return share, n_c


# ------------------------------------------------------------------------------------------------------ inputs
def load_inputs_ext(sc: str) -> pipeline.Inputs:
    """pipeline.load_inputs with the pilot's `rank` column kept on the games."""
    from sealed_lab.ingest.corpus import load_corpus, read_manifest

    c = load_corpus(sc)
    games = c["games"].select(GAME_COLS_EXT).collect()
    return pipeline.Inputs(set_code=sc, cards=c["cards"], games=games, decks=c["decks"], deck_cards=c["deck_cards"].collect(),
                           builds=c["builds"], manifest=read_manifest(sc) or {}, lazy=c)


def prepare_set(sc: str, n_jobs: int):
    """play_prior_check.prepare_set, except that extra draft events (PickTwoDraft has rank) are NOT dropped."""
    inp = load_inputs_ext(sc)
    inp.cards = castability.with_cost_groups(inp.cards, sc)
    keys, table = csc.read_phi_readonly(inp, config.cache_dir(sc), n_jobs)
    space, _ = pipeline.plain_space(inp, keys, table)
    return inp, space


def make_rank_stage(st: design.StageData) -> tuple[design.StageData, np.ndarray]:
    """The stage with the 5 rank indicators appended to the covariate block C (as skill_shrink_check appends `_skill_rel`);
    returns (stage, covariate prior SDs)."""
    ind = rank_indicators(st.games["rank"].to_list())
    C2 = sp.hstack([st.prob.C, sp.csr_matrix(ind)], format="csr")
    prob2 = glm.Problem(y=st.prob.y, C=C2, D=st.prob.D, idx=st.prob.idx)
    sig2 = np.concatenate([st.enc.sigma, np.full(ind.shape[1], RANK_SIGMA)])
    return replace(st, prob=prob2), sig2


# ------------------------------------------------------------------------------------------------------ one set
def run_set(sc: str, out: Path, n_jobs: int) -> dict:
    t0 = time.time()
    inp, space = prepare_set(sc, n_jobs)
    cards = inp.cards
    csort = cards.sort("card_id")
    name_of_id = dict(zip(csort["card_id"].to_list(), csort["name"].to_list()))
    names_by_col = [name_of_id[int(i)] for i in space.nb_ids]
    col_of_name: dict[str, int] = {}
    for j, nm in enumerate(names_by_col):
        col_of_name.setdefault(nm, j)
        if " // " in nm:
            col_of_name.setdefault(nm.split(" // ")[0].strip(), j)

    gd = inp.games.filter(pl.col("fmt") == "draft")
    g = (gd.filter(pl.col("rank").is_not_null())
         .with_columns(pl.col("rank").str.to_lowercase())
         .filter(pl.col("rank").is_in(RANKS)))
    if g.height < 1000 or g["draft_id"].n_unique() < 20:
        return dict(set=sc, status="skipped", reason=f"too few ranked Draft games ({g.height})", seconds=round(time.time() - t0, 1))
    counts = dict(
        draft_games=int(gd.height), ranked_games=int(g.height), n_events=int(g["draft_id"].n_unique()),
        per_rank={r: int(n) for r, n in g.group_by("rank").len().iter_rows() if r in RANKS},
        per_event={str(e): int(n) for e, n in g.group_by("event").len().sort("event").iter_rows()},
        per_event_rank={str(e): {str(r): int(n) for r, n in g.filter(pl.col("event") == e).group_by("rank").len().iter_rows()}
                        for e in g["event"].unique().to_list()},
        dropped_unranked={str(e): int(n) for e, n in gd.filter(pl.col("rank").is_null() | ~pl.col("rank").str.to_lowercase().is_in(RANKS))
                          .group_by("event").len().iter_rows()},
    )

    # BASE: the shipped first-stage fit (zero prior, CV card sigma)
    st = design.build_stage("draft", g, space)
    n_card, n_feat = space.n_card, space.n_feat
    k = st.prob.k
    p_deck = n_card + n_feat
    tmp = out / "tmp" / sc
    (tmp / "base").mkdir(parents=True, exist_ok=True)
    opts = cardvalue.ChainOpts()
    base = cardvalue._stage_fit(st, np.zeros(p_deck), n_card, n_feat, str(tmp / "base"), n_jobs, 100, None, None, opts=opts)
    t_base = time.time() - t0

    # RANK: same stage + rank indicators, card sigma fixed at BASE's
    st2, sig_cov2 = make_rank_stage(st)
    spec2 = glm.SigmaSpec(sig_cov=sig_cov2, n_card=n_card, n_theta=n_feat, kappa=base.spec.kappa, th_lo=base.spec.th_lo,
                          th_hi=base.spec.th_hi, th_abs=base.spec.th_abs)
    coef2, se2, res2 = cardvalue.fit_stage(st2, np.zeros(p_deck), spec2, base.sigma, with_se=True)
    k2 = st2.prob.k
    assert k2 == k + len(RANK_DUMMIES)
    se_rob2 = se2[1]
    i_skill = st.enc.names.index("skill_lin")

    # BASE refitted the same plain way (same card sigma, no stacking), so BASE and RANK differ only by the rank columns
    spec1 = glm.SigmaSpec(sig_cov=st.enc.sigma, n_card=n_card, n_theta=n_feat, kappa=base.spec.kappa, th_lo=base.spec.th_lo,
                          th_hi=base.spec.th_hi, th_abs=base.spec.th_abs)
    coef1, se1, _ = cardvalue.fit_stage(st, np.zeros(p_deck), spec1, base.sigma, with_se=True)
    b_base = coef1[k:k + n_card]
    b_rank = coef2[k2:k2 + n_card]
    delta = b_rank - b_base

    # rank coefficients
    rank_coef = {}
    for j, r in enumerate(RANK_DUMMIES):
        c, s = float(coef2[k + j]), float(se_rob2[k + j])
        rank_coef[r] = dict(logit=c, se_logit=s, pp=c * PP, se_pp=s * PP)
    skill = dict(base=float(coef1[i_skill]), rank=float(coef2[i_skill]),
                 se_base=float(se1[1][i_skill]), se_rank=float(se_rob2[i_skill]))

    # high-rank share and exposure
    hi = np.isin(g["rank"].to_numpy(), list(HIGH)).astype(float)
    Dcards = st.prob.D[:, :n_card]
    h_c, n_c = high_rank_share(Dcards, st.prob.idx, hi)
    expo = st.exposure
    sel = (expo > 0) & np.isfinite(h_c)
    fit_all = wls_slope(h_c[sel], delta[sel], expo[sel])
    fit_all_pp = dict(slope_pp=fit_all["slope"] * PP, se_pp=fit_all["se"] * PP)
    corr_br = float(np.corrcoef(b_base, b_rank)[0, 1])
    base_hi_share = float(hi.mean())

    def row(i: int) -> dict:
        return dict(name=names_by_col[i], base=float(b_base[i]), rank=float(b_rank[i]), delta=float(delta[i]),
                    delta_pp=float(delta[i] * PP), h_c=None if not np.isfinite(h_c[i]) else float(h_c[i]),
                    exposure=float(expo[i]), n_games=float(n_c[i]))

    cand = np.nonzero(sel)[0]
    order = cand[np.argsort(-delta[cand])]
    top_pos = [row(int(i)) for i in order[:TOP_K]]
    top_neg = [row(int(i)) for i in order[::-1][:TOP_K]]
    tracked = {nm: row(col_of_name[nm]) for nm in TRACKED if nm in col_of_name}

    res = dict(
        set=sc, status="ok", counts=counts, high_share_games=base_hi_share, n_cards=int(n_card), n_cards_with_exposure=int(sel.sum()),
        sigma_card=float(base.sigma), sigma_th_abs=base.spec.th_abs, cv_scheme=base.cv_scheme, base_converged=bool(base.converged),
        rank_converged=bool(res2.converged), rank_coef=rank_coef, skill_lin=skill,
        cards=dict(corr_base_rank=corr_br, mean_abs_delta=float(np.mean(np.abs(delta[sel]))),
                   mean_abs_delta_pp=float(np.mean(np.abs(delta[sel])) * PP),
                   mean_abs_delta_exposure_weighted_pp=float(np.average(np.abs(delta[sel]), weights=expo[sel]) * PP),
                   sd_delta_pp=float(np.std(delta[sel]) * PP), sd_base_pp=float(np.std(b_base[sel]) * PP),
                   mean_delta_pp=float(np.mean(delta[sel]) * PP)),
        slope=dict(**fit_all, **fit_all_pp), top_positive=top_pos, top_negative=top_neg, tracked=tracked,
        seconds=round(time.time() - t0, 1), seconds_base=round(t_base, 1),
    )
    rc = rank_coef
    print(f"[{sc}] {g.height:7d} ranked draft games ({counts['n_events']} events, {base_hi_share*100:.0f}% plat+) | sigma={base.sigma:.3f} | "
          f"rank pp: " + " ".join(f"{r[:3]}={rc[r]['pp']:+.2f}" for r in RANK_DUMMIES)
          + f" | skill_lin {skill['base']:.3f}->{skill['rank']:.3f} | corr={corr_br:.4f} mean|d|={res['cards']['mean_abs_delta_pp']:.3f}pp"
          f" | slope(d~h)={fit_all_pp['slope_pp']:+.3f}pp/unit se {fit_all_pp['se_pp']:.3f} | {res['seconds']:.0f}s", flush=True)
    cardvalue.cleanup(str(tmp))
    return res


# ------------------------------------------------------------------------------------------------------ report
def _json_default(o):
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def summarize(per_set: dict[str, dict]) -> dict:
    ok = {sc: r for sc, r in per_set.items() if r.get("status") == "ok"}
    coefs = {}
    for r in RANK_DUMMIES:
        coefs[r] = dict(
            logit=pool_inverse_variance([v["rank_coef"][r]["logit"] for v in ok.values()], [v["rank_coef"][r]["se_logit"] for v in ok.values()]),
            mean_unweighted_logit=float(np.mean([v["rank_coef"][r]["logit"] for v in ok.values()])) if ok else float("nan"))
        coefs[r]["pp"] = dict(mean=coefs[r]["logit"]["mean"] * PP, se=coefs[r]["logit"]["se"] * PP)
    sl = pool_inverse_variance([v["slope"]["slope"] for v in ok.values()], [v["slope"]["se"] for v in ok.values()])
    sl_pp = dict(mean=sl["mean"] * PP, se=sl["se"] * PP, n=sl["n"])
    n_pos = int(sum(1 for v in ok.values() if v["slope"]["slope"] > 0))
    return dict(n_sets=len(ok), sets=sorted(ok), pooled_rank_coef=coefs, pooled_slope=dict(**sl, **{"pp": sl_pp}),
                n_sets_positive_slope=n_pos,
                mean_corr=float(np.mean([v["cards"]["corr_base_rank"] for v in ok.values()])) if ok else float("nan"),
                mean_abs_delta_pp=float(np.mean([v["cards"]["mean_abs_delta_pp"] for v in ok.values()])) if ok else float("nan"))


def f(x, nd=3, sign=True):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def card_table(rows: list[dict]) -> list[str]:
    out = ["| card | BASE pp | RANK pp | delta pp | h_c | exposure |", "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        out.append(f"| {r['name']} | {f(r['base']*PP, 2)} | {f(r['rank']*PP, 2)} | {f(r['delta_pp'], 3)} | "
                   f"{f(r['h_c'], 2, False)} | {r['exposure']:.0f} |")
    return out


def write_md(summary: dict, path: Path) -> None:
    per_set = summary["per_set"]
    S = summary["summary"]
    L = ["# Rank bias check (Draft stage, BASE vs RANK)", "",
         "BASE = the shipped first-stage fit; RANK = same stage plus rank indicators (bronze = reference, prior SD 1.0), card sigma fixed "
         "at BASE's. pp = logit x 0.2475 x 100 (a 55% deck). h_c = share of a card's deck-games played by platinum/diamond/mythic players. "
         "Slope = exposure-weighted least squares of delta (RANK - BASE, logit) on h_c; SE is the plain WLS SE. Expected if rank matters: "
         "rank coefficients negative and growing with tier; positive slope.", "",
         "## Rank coefficients per set (pp at a 55% deck; cluster-robust SE over draft_id in brackets)", "",
         "| set | ranked games | % plat+ | " + " | ".join(RANK_DUMMIES) + " | skill_lin BASE | skill_lin RANK |",
         "|---|---:|---:|" + "---:|" * len(RANK_DUMMIES) + "---:|---:|"]
    for sc, r in per_set.items():
        if r.get("status") != "ok":
            L.append(f"| {sc} | skipped: {r.get('reason')} |" + " |" * (len(RANK_DUMMIES) + 4))
            continue
        cells = " | ".join(f"{f(r['rank_coef'][t]['pp'], 2)} ({r['rank_coef'][t]['se_pp']:.2f})" for t in RANK_DUMMIES)
        L.append(f"| {sc} | {r['counts']['ranked_games']} | {r['high_share_games']*100:.0f} | {cells} | "
                 f"{r['skill_lin']['base']:.3f} | {r['skill_lin']['rank']:.3f} |")
    pc = S["pooled_rank_coef"]
    L.append("| inverse-variance mean | | | " + " | ".join(f"{f(pc[t]['pp']['mean'], 2)} ({pc[t]['pp']['se']:.2f})" for t in RANK_DUMMIES) + " | | |")
    L += ["", "## Card-value shift and its slope on h_c", "",
          "| set | corr(BASE,RANK) | mean abs delta pp | SD BASE pp | slope pp per unit h_c | SE | corr(delta,h_c) | cards used |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for sc, r in per_set.items():
        if r.get("status") != "ok":
            continue
        c, s = r["cards"], r["slope"]
        L.append(f"| {sc} | {c['corr_base_rank']:.4f} | {c['mean_abs_delta_pp']:.3f} | {c['sd_base_pp']:.2f} | {f(s['slope_pp'], 3)} | "
                 f"{s['se_pp']:.3f} | {f(s['corr'], 3)} | {s['n']} |")
    ps = S["pooled_slope"]["pp"]
    L.append(f"| inverse-variance mean | {S['mean_corr']:.4f} (mean) | {S['mean_abs_delta_pp']:.3f} (mean) | | {f(ps['mean'], 3)} | {ps['se']:.3f} | | |")
    L += ["", f"Sets with a positive slope: {S['n_sets_positive_slope']} of {S['n_sets']}.", ""]
    for sc, r in per_set.items():
        if r.get("status") != "ok":
            continue
        cn = r["counts"]
        L += [f"## {sc}", "",
              f"Games per rank: " + ", ".join(f"{t} {cn['per_rank'].get(t, 0)}" for t in RANKS) + ". "
              f"Games per event: " + ", ".join(f"{e} {n}" for e, n in cn["per_event"].items()) + ". "
              + ("Unranked games dropped: " + ", ".join(f"{e} {n}" for e, n in cn["dropped_unranked"].items()) + "." if cn["dropped_unranked"] else ""),
              f"Card sigma {r['sigma_card']:.4f} (fixed in RANK). Converged: BASE {r['base_converged']}, RANK {r['rank_converged']}.", "",
              "Largest positive delta (RANK higher than BASE):", ""] + card_table(r["top_positive"]) \
            + ["", "Largest negative delta:", ""] + card_table(r["top_negative"])
        if r["tracked"]:
            L += ["", "Tracked cards:", ""] + card_table(list(r["tracked"].values()))
        L.append("")
    path.write_text("\n".join(L), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", type=Path, nargs="?", default=Path(tempfile.gettempdir()) / "sealed_lab_rank_bias")
    ap.add_argument("--sets", nargs="+", default=ALL_SETS)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    a = ap.parse_args(argv)
    check_home_not_real()
    out = a.out.resolve()
    check_out_not_home(out)
    out.mkdir(parents=True, exist_ok=True)

    per_set: dict[str, dict] = {}
    t_all = time.time()
    with glm.shared_pool(a.jobs):
        for sc in a.sets:
            sc = sc.upper()
            done = out / f"per_set_{sc}.json"          # a finished set is kept, so a crash in a later set loses nothing
            if done.exists():
                per_set[sc] = json.loads(done.read_text(encoding="utf-8"))
                print(f"[{sc}] reused {done.name}", flush=True)
                continue
            per_set[sc] = run_set(sc, out, a.jobs)
            if per_set[sc].get("status") != "ok":
                print(f"[{sc}] {per_set[sc].get('status')}: {per_set[sc].get('reason')}", flush=True)
            done.write_text(json.dumps(per_set[sc], indent=1, default=_json_default), encoding="utf-8")
    summary = dict(created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), total_seconds=round(time.time() - t_all, 1),
                   rank_sigma=RANK_SIGMA, pp_factor=PP, high_ranks=list(HIGH), per_set=per_set, summary=summarize(per_set))
    (out / "rank_bias.json").write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    write_md(summary, out / "rank_bias.md")
    S = summary["summary"]
    print(f"\nWrote {out / 'rank_bias.json'} and rank_bias.md in {summary['total_seconds']:.1f}s")
    if S["n_sets"]:
        print("pooled rank coef pp: " + " ".join(f"{t}={S['pooled_rank_coef'][t]['pp']['mean']:+.2f}({S['pooled_rank_coef'][t]['pp']['se']:.2f})"
                                                for t in RANK_DUMMIES))
        sp_ = S["pooled_slope"]["pp"]
        print(f"pooled slope of delta on h_c: {sp_['mean']:+.3f} pp per unit (SE {sp_['se']:.3f}); positive in {S['n_sets_positive_slope']}/{S['n_sets']} sets")
    return 0


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    sys.exit(main())
