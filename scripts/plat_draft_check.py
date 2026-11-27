"""Pre-registered "Draft stage learned from Platinum+ players, or corrected for rank" test on Sealed hold-outs (docs/plat_draft/RULE.md).

Arms (each changes the Draft stage only, against M = the plain chain refitted on the training events):
  P  the Draft stage trained only on Draft-stage games whose rank is platinum, diamond or mythic
  R  every Draft-stage game, plus 6 rank indicators in the Draft stage's covariate block (silver, gold, platinum, diamond, mythic,
     unranked; bronze = reference; prior SD 1.0); the Draft sigma is re-selected by CV inside the chain
  N  (reported only) a random sample of whole Draft events from M's Draft stage with about as many games as P (fixed seed)

Reads the app home and the raw files strictly read-only (SEALED_LAB_HOME must point at a scratch copy); writes only to OUT.

    python scripts/plat_draft_check.py OUT [--sets ...] [--jobs N] [--selfcheck]
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
from scipy.special import expit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from sealed_lab import config  # noqa: E402
from sealed_lab.ingest import registry  # noqa: E402
from sealed_lab.model import cardvalue, castability, design, glm, pipeline, pooling, validate  # noqa: E402
import cold_start_check as csc  # noqa: E402  (read_phi_readonly, paired_gain_metrics)
import play_prior_check as ppc  # noqa: E402  (summarize, pool helpers, safety checks, top_movers, JSON default)

DEFAULT_SETS = ["FRA", "HOB", "MSH", "SOS", "TMT"]
ARMS = ("P", "R", "N")
DECIDING = ("P", "R")
HIGH_RANKS = ("platinum", "diamond", "mythic")
RANKS = ["bronze", "silver", "gold", "platinum", "diamond", "mythic"]
RANK_COLS = ["silver", "gold", "platinum", "diamond", "mythic", "unranked"]
RANK_SIGMA = 1.0
PP = 0.55 * 0.45 * 100.0            # logit -> percentage points at a 55% deck (24.75)
N_TOL = 0.02                        # N's draft game count within 2% of P's
N_SEED0 = 12345
GAME_COLS_EXT = pipeline.GAME_COLS + ["rank"]
MIN_GAIN = ppc.MIN_GAIN
SE_FLOOR = ppc.SE_FLOOR
N_REP = ppc.N_REP
check_home_not_real = ppc.check_home_not_real
check_out_not_home = ppc.check_out_not_home
_json_default = ppc._json_default


# ------------------------------------------------------------------------------------------------------ pure helpers
def rank_filter(games: pl.DataFrame, keep=HIGH_RANKS, keep_null: bool = False) -> pl.DataFrame:
    """Arm P's filter: drop the `fmt == "draft"` games whose lower-cased rank is not in `keep` (null rank and TradDraft have no rank, so
    they go unless `keep_null`). Games of the other formats are untouched."""
    low = pl.col("rank").str.to_lowercase()
    ok = low.is_in(list(keep)).fill_null(False)
    if keep_null:
        ok = ok | pl.col("rank").is_null()
    return games.filter((pl.col("fmt") != "draft") | ok)


def rank_indicators6(rank) -> np.ndarray:
    """(n, 6) indicators silver, gold, platinum, diamond, mythic, unranked of a rank column. Bronze (the reference) is all zeros;
    unranked = null or any other value (TradDraft has no rank). Case-insensitive."""
    r = np.array(["" if x is None else str(x).strip().lower() for x in rank], dtype=object)
    if len(r) == 0:
        return np.zeros((0, len(RANK_COLS)))
    cols = [(r == t).astype(float) for t in RANK_COLS[:-1]]
    cols.append((~np.isin(r, RANKS)).astype(float))
    return np.column_stack(cols)


def make_rank_stage(st: design.StageData, zero: bool = False) -> tuple[design.StageData, int]:
    """The Draft stage with the 6 rank indicators appended to its covariate block C and to the covariate prior SDs (`enc.sigma`, which
    `cardvalue.make_spec` reads for the CV and the fit; `enc.names` follows). Returns (stage, k of the original stage). `zero` forces the
    indicator columns to zero (implementation check)."""
    k0 = st.prob.k
    ind = rank_indicators6(st.games["rank"].to_list())
    if zero:
        ind = np.zeros_like(ind)
    C2 = sp.hstack([st.prob.C, sp.csr_matrix(ind)], format="csr")
    enc2 = replace(st.enc, names=list(st.enc.names) + [f"rank:{c}" for c in RANK_COLS],
                   sigma=np.concatenate([st.enc.sigma, np.full(len(RANK_COLS), RANK_SIGMA)]))
    return replace(st, prob=replace(st.prob, C=C2), enc=enc2), k0


def sample_events(counts: dict[str, int], target: int, seed: int, tol: float = N_TOL) -> list[str]:
    """Arm N's sampler: whole events (draft_ids) drawn at random, in a seeded order, until the game count is within `tol` of `target`.
    An event that would overshoot the upper bound is skipped (a later, smaller one may fit). Raises if the target cannot be hit."""
    ids = sorted(counts)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ids))
    lo, hi = target * (1.0 - tol), target * (1.0 + tol)
    chosen, tot = [], 0
    for j in order:
        n = int(counts[ids[j]])
        if tot + n > hi:
            continue
        chosen.append(ids[j])
        tot += n
        if tot >= lo:
            return chosen
    raise ValueError(f"cannot reach {target} games (+-{tol:.0%}) from {sum(counts.values())} games in {len(ids)} events")


# ------------------------------------------------------------------------------------------------------ inputs
def load_inputs_ext(sc: str) -> pipeline.Inputs:
    """pipeline.load_inputs with the pilot's `rank` column kept on the games (as rank_bias_check)."""
    from sealed_lab.ingest.corpus import load_corpus, read_manifest

    c = load_corpus(sc)
    games = c["games"].select(GAME_COLS_EXT).collect()
    return pipeline.Inputs(set_code=sc, cards=c["cards"], games=games, decks=c["decks"], deck_cards=c["deck_cards"].collect(),
                           builds=c["builds"], manifest=read_manifest(sc) or {}, lazy=c)


def prepare_set(sc: str, n_jobs: int):
    """play_prior_check.prepare_set (extra Draft events dropped unless the set's stored report applied them), with `rank` kept."""
    inp = load_inputs_ext(sc)
    inp.cards = castability.with_cost_groups(inp.cards, sc)
    rep_path = config.model_dir(sc) / "report.json"
    ed_applied = False
    if rep_path.exists():
        try:
            rep = json.loads(rep_path.read_text(encoding="utf-8"))
            ed_applied = bool((rep.get("extra_draft") or {}).get("applied", False))
        except Exception:  # noqa: BLE001
            ed_applied = False
    if pipeline.extra_draft_events(inp.games) and not ed_applied:
        inp = pipeline.without_extra_draft(inp)
    keys, table = csc.read_phi_readonly(inp, config.cache_dir(sc), n_jobs)
    space, _ = pipeline.plain_space(inp, keys, table)
    games = inp.games
    n_games = {f: int(games.filter(pl.col("fmt") == f).height) for f in ("draft", "bo1", "bo3")}
    formats = [f for f in pipeline.FMT_ORDER if n_games.get(f, 0) >= 1]
    return inp, space, formats, ed_applied


# ------------------------------------------------------------------------------------------------------ chains
def fit_chain(stages: dict[str, design.StageData], fmt: str, space, tmp: Path, n_jobs: int, seed: int,
              opts: cardvalue.ChainOpts) -> dict[str, cardvalue.StageFit]:
    tmp.mkdir(parents=True, exist_ok=True)
    prior_ok = set(stages) - {fmt}
    fits = cardvalue.run_chain(stages, space.n_card, space.n_feat, str(tmp), n_jobs, seed=seed, prior_ok=prior_ok, opts=opts)
    return fits


def stages_p(fmt: str, games: pl.DataFrame, g_tr: pl.DataFrame, t_cut, space, keep=HIGH_RANKS, keep_null: bool = False):
    """Arm P: validate.chain_stages on games whose Draft-stage rows are filtered to `keep` ranks BEFORE design.build_stage."""
    return validate.chain_stages(fmt, rank_filter(games, keep, keep_null), g_tr, t_cut, space)


def stages_r(st_m: dict[str, design.StageData], zero: bool = False):
    """Arm R: M's stages, the Draft stage replaced by the same games with the 6 indicators in its covariate block. The bo1 / bo3 stages
    get no rank columns."""
    out = dict(st_m)
    k0 = None
    if "draft" in st_m:
        out["draft"], k0 = make_rank_stage(st_m["draft"], zero=zero)
    return out, k0


def stages_n(fmt: str, games: pl.DataFrame, g_tr: pl.DataFrame, t_cut, space, st_m, target: int, seed: int):
    """Arm N: a random sample of whole Draft events of M's Draft stage with about `target` games (within 2%); the rest as M."""
    gd = st_m["draft"].games
    cnt = {str(k): int(v) for k, v in gd.group_by("draft_id").len().iter_rows()}
    ids = set(sample_events(cnt, target, seed))
    g2 = games.filter((pl.col("fmt") != "draft") | pl.col("draft_id").is_in(list(ids)))
    return validate.chain_stages(fmt, g2, g_tr, t_cut, space), len(ids)


def max_abs_diff(fits_a: dict, fits_b: dict) -> dict[str, float]:
    return {n: float(np.max(np.abs(fits_a[n].beta_total - fits_b[n].beta_total))) for n in fits_b}


# ------------------------------------------------------------------------------------------------------ one set
def arm_summary(y, p_m, p_a, groups, is_first, seed_f, fits_a, fits_m, fmt, names_by_col, extra: dict | None = None) -> dict:
    m = csc.paired_gain_metrics(y, p_m, p_a, groups, is_first, n_rep=N_REP, seed=seed_f)
    b_a, b_m = fits_a[fmt].beta_total, fits_m[fmt].beta_total
    m.update(beta_corr=float(np.corrcoef(b_a, b_m)[0, 1]), movers=ppc.top_movers(names_by_col, b_m, b_a),
             sigma_draft=float(fits_a["draft"].sigma) if "draft" in fits_a else None,
             sigma_target=float(fits_a[fmt].sigma))
    if extra:
        m.update(extra)
    return m


def run_set(sc: str, out: Path, n_jobs: int, set_index: int) -> dict:
    t0 = time.time()
    inp, space, formats, ed_applied = prepare_set(sc, n_jobs)
    games = inp.games
    opts = cardvalue.ChainOpts()
    csort = inp.cards.sort("card_id")
    name_of_id = dict(zip(csort["card_id"].to_list(), csort["name"].to_list()))
    names_by_col = [name_of_id[int(i)] for i in space.nb_ids]

    fmt_res: dict[str, dict] = {}
    for i, fmt in enumerate(formats):
        seed_f = i
        gT = games.filter(pl.col("fmt") == fmt)
        split = validate._holdout_split(gT)
        if split is None:
            fmt_res[fmt] = dict(status="skipped", reason=f"too few {fmt} events/games")
            continue
        hold_ids, t_cut, _nev, _nho = split
        g_tr = gT.filter(~pl.col("draft_id").is_in(hold_ids))
        g_ho = gT.filter(pl.col("draft_id").is_in(hold_ids))
        tmp_f = out / "tmp" / f"{sc}_{fmt}"
        cs = seed_f + 100

        # M
        tm = time.time()
        st_m = validate.chain_stages(fmt, games, g_tr, t_cut, space)
        fits_m = fit_chain(st_m, fmt, space, tmp_f / "m", n_jobs, cs, opts)
        sec = dict(M=round(time.time() - tm, 1))
        if "draft" not in st_m:
            fmt_res[fmt] = dict(status="skipped", reason="no Draft stage")
            cardvalue.cleanup(str(tmp_f))
            continue
        prob_ho, _, _ = design.make_problem(g_ho, st_m[fmt].enc, space, clamp_time=True)
        p_m = expit(prob_ho.eta(fits_m[fmt].coef))
        y = prob_ho.y
        groups = np.unique(g_ho["draft_id"].to_numpy(), return_inverse=True)[1]
        is_first = validate.half_masks(g_ho)
        draft_games = dict(M=int(st_m["draft"].n_games), M_events=int(st_m["draft"].n_groups))
        b_m = fits_m[fmt].beta_total
        arms_res: dict[str, dict] = {}
        probs: dict[str, np.ndarray] = {}

        # P
        tm = time.time()
        st_p = stages_p(fmt, games, g_tr, t_cut, space)
        if "draft" not in st_p:
            arms_res["P"] = dict(no_signal=True, reason="no Draft stage left after the rank filter")
            fits_p = None
        else:
            draft_games.update(P=int(st_p["draft"].n_games), P_events=int(st_p["draft"].n_groups))
            fits_p = fit_chain(st_p, fmt, space, tmp_f / "p", n_jobs, cs, opts)
            probs["P"] = expit(prob_ho.eta(fits_p[fmt].coef))
            arms_res["P"] = arm_summary(y, p_m, probs["P"], groups, is_first, seed_f, fits_p, fits_m, fmt, names_by_col,
                                        dict(draft_games=draft_games["P"]))
        sec["P"] = round(time.time() - tm, 1)
        del st_p

        # R
        tm = time.time()
        st_rr, k0 = stages_r(st_m)
        fits_r = fit_chain(st_rr, fmt, space, tmp_f / "r", n_jobs, cs, opts)
        probs["R"] = expit(prob_ho.eta(fits_r[fmt].coef))
        fd = fits_r["draft"]
        rank_coef = {}
        for j, c in enumerate(RANK_COLS):
            v, s = float(fd.coef[k0 + j]), float(fd.se[k0 + j])
            rank_coef[c] = dict(logit=v, se_logit=s, pp=v * PP, se_pp=s * PP)
        arms_res["R"] = arm_summary(y, p_m, probs["R"], groups, is_first, seed_f, fits_r, fits_m, fmt, names_by_col,
                                    dict(draft_games=draft_games["M"], rank_coef=rank_coef, draft_converged=bool(fd.converged)))
        sec["R"] = round(time.time() - tm, 1)
        del st_rr

        # N (reported only; needs P's game count)
        tm = time.time()
        if "P" in draft_games:
            seed_n = N_SEED0 + set_index
            st_n, n_ev = stages_n(fmt, games, g_tr, t_cut, space, st_m, draft_games["P"], seed_n)
            draft_games.update(N=int(st_n["draft"].n_games), N_events=int(n_ev), N_seed=seed_n)
            fits_n = fit_chain(st_n, fmt, space, tmp_f / "n", n_jobs, cs, opts)
            probs["N"] = expit(prob_ho.eta(fits_n[fmt].coef))
            arms_res["N"] = arm_summary(y, p_m, probs["N"], groups, is_first, seed_f, fits_n, fits_m, fmt, names_by_col,
                                        dict(draft_games=draft_games["N"]))
            del st_n
        else:
            arms_res["N"] = dict(no_signal=True, reason="no P game count")
        sec["N"] = round(time.time() - tm, 1)

        # paired differences loss(R)-loss(P), loss(N)-loss(P) (positive = P better)
        diffs = {}
        if "P" in probs:
            for other in ("R", "N"):
                if other in probs:
                    diffs[f"{other}|P"] = csc.paired_gain_metrics(y, probs[other], probs["P"], groups, is_first, n_rep=N_REP, seed=seed_f)

        fmt_res[fmt] = dict(status="ok", n_holdout_games=int(len(y)), n_holdout_events=int(groups.max()) + 1, t_cut=str(t_cut),
                            draft_games=draft_games, sigma_draft=dict(M=float(fits_m["draft"].sigma)), seconds=sec,
                            sigma_target_M=float(fits_m[fmt].sigma), arms=arms_res, diffs=diffs, loss_M=float(
                                -np.mean(y * np.log(np.clip(p_m, 1e-9, 1 - 1e-9)) + (1 - y) * np.log(np.clip(1 - p_m, 1e-9, 1 - 1e-9)))))
        for a in ("P", "R", "N"):
            if arms_res[a].get("sigma_draft") is not None:
                fmt_res[fmt]["sigma_draft"][a] = arms_res[a]["sigma_draft"]
        parts = []
        for a in ARMS:
            r = arms_res[a]
            parts.append(f"{a}=n/a" if r.get("no_signal") else
                         f"{a}={r['gain']*1e4:+6.2f} [{r['lo']*1e4:+6.2f},{r['hi']*1e4:+6.2f}] sd={r['sigma_draft']:.3f} corr={r['beta_corr']:.4f}")
        print(f"[{sc} {fmt}] ho={len(y):5d} draft games M/P/N={draft_games['M']}/{draft_games.get('P')}/{draft_games.get('N')} | "
              + " | ".join(parts) + f" | {time.time() - t0:.0f}s", flush=True)
        cardvalue.cleanup(str(tmp_f))

    return dict(set=sc, seconds=round(time.time() - t0, 1), extra_draft_applied=ed_applied, formats=fmt_res)


# ------------------------------------------------------------------------------------------------------ pooling and choice
def choose_arm(pooled: dict) -> dict:
    passing = [a for a in DECIDING if pooled[a]["passes_rule"]]
    score = {}
    for a in DECIDING:
        gains = [fv["raw"]["pooled"]["whole"]["mean"] for fv in pooled[a]["formats"].values() if fv.get("n_sets")]
        score[a] = float(np.mean(gains)) if gains else float("-inf")
    out = dict(passing=passing, score_mean_whole_gain=score)
    if not passing:
        return dict(out, chosen=None, why="no arm passes the rule")
    return dict(out, chosen=max(passing, key=lambda a: score[a]),
                why="larger pooled whole-hold-out gain (mean over bo1 and bo3, raw) among the passing arms")


def pooled_diff_answers(sm: dict) -> dict:
    ans = {}
    for fmt, fv in sm["formats"].items():
        if not fv.get("n_sets"):
            continue
        ans[fmt] = {tag: {p: fv[tag]["pooled"][p] for p in pooling.PARTS} for tag in ("raw", "se_floored")}
        ans[fmt]["n_sets"] = fv["n_sets"]
        ans[fmt]["excluded"] = fv.get("excluded", [])
    return ans


# ------------------------------------------------------------------------------------------------------ report
def fx(x, nd=2, scale=1e4, sign=True):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    v = x * scale
    return f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"


def ci(m: dict) -> str:
    return f"{fx(m['mean'])} [{fx(m['lo'])}, {fx(m['hi'])}]"


def write_md(summary: dict, path: Path) -> None:
    per_set, pooled, pdiff = summary["per_set"], summary["pooled"], summary["pooled_diffs"]
    ch = summary["choice"]
    L = ["# Draft stage from Platinum+ players, or corrected for rank: results (plat_draft_check)", "",
         "Rule: `docs/plat_draft/RULE.md`. Gains are `loss(M) - loss(arm)` per Sealed game, x1e-4, positive = arm better. "
         f"Sets: {', '.join(summary['sets'])}. Pooling: DerSimonian-Laird per `docs/pooled_retest/POOLING_RULE.md`; verdicts raw / with SE >= {SE_FLOOR:g}.", "",
         f"## Verdict: {'chosen arm ' + ch['chosen'] if ch.get('chosen') else 'NO ARM PASSES'}", "",
         "| Arm | Format | Sets | Whole (x1e-4) | Half A (x1e-4) | Half B (x1e-4, lo90) | Verdict (raw / SE-floored) |",
         "|---|---|---:|---|---|---|---|"]
    for arm in ARMS:
        for fmt, fv in pooled[arm]["formats"].items():
            if not fv.get("n_sets"):
                continue
            rw, ra, rb = (fv["raw"]["pooled"][p] for p in ("whole", "A", "B"))
            rv, flv = fv["raw"]["verdict"], fv["se_floored"]["verdict"]
            tag = f"`{arm}`" + (" (reported)" if arm == "N" else "")
            L.append(f"| {tag} | {fmt} | {fv['n_sets']} | {fx(rw['mean'])} [{fx(rw['lo'])}, {fx(rw['hi'])}] | "
                     f"{fx(ra['mean'])} [{fx(ra['lo'])}, {fx(ra['hi'])}] | {fx(rb['mean'])} ({fx(rv['B_lo90'])}) | {rv['label']} / {flv['label']} |")
    L += ["", f"- Passing (P, R): {ch['passing'] or 'none'}; mean whole gain P {fx(ch['score_mean_whole_gain']['P'])}, R {fx(ch['score_mean_whole_gain']['R'])} (x1e-4).",
          "- Excluded cells: " + ("; ".join(f"{a} {f}: {fv['excluded']}" for a in ARMS for f, fv in pooled[a]["formats"].items() if fv.get("excluded"))
                                 or "none") + ".", ""]
    L += ["## Paired pooled differences (positive = P has the lower loss)", "",
          "| Difference | Format | Whole (x1e-4) | Half A | Half B | note |", "|---|---|---|---|---|---|"]
    for key, label in (("R|P", "loss(R) - loss(P)"), ("N|P", "loss(N) - loss(P)")):
        for fmt, fv in pdiff[key]["formats"].items():
            if not fv.get("n_sets"):
                continue
            rw, ra, rb = (fv["raw"]["pooled"][p] for p in ("whole", "A", "B"))
            note = "high-rank players carry information beyond the smaller sample" if key == "N|P" and rw["lo"] > 0 else ""
            L.append(f"| {label} | {fmt} | {ci(rw)} | {ci(ra)} | {ci(rb)} | {note} |")
    L += ["", "## Per set and format (gains vs M, x1e-4, [95% CI]; A / B halves)", "",
          "| set | fmt | arm | Whole | Half A | Half B | draft games | draft sigma | corr with M |", "|---|---|---|---|---|---|---:|---:|---:|"]
    for sc, r in per_set.items():
        for fmt, fr in r["formats"].items():
            if fr.get("status") != "ok":
                L.append(f"| {sc} | {fmt} | skipped: {fr.get('reason')} | | | | | | |")
                continue
            dg = fr["draft_games"]
            L.append(f"| {sc} | {fmt} | M | | | | {dg['M']} | {fr['sigma_draft']['M']:.3f} | 1 |")
            for a in ARMS:
                m = fr["arms"][a]
                if m.get("no_signal"):
                    L.append(f"| {sc} | {fmt} | {a} | n/a ({m.get('reason')}) | | | | | |")
                    continue
                h = m["halves"]
                L.append(f"| {sc} | {fmt} | {a} | {fx(m['gain'])} [{fx(m['lo'])}, {fx(m['hi'])}] | {fx(h['A']['gain'])} [{fx(h['A']['lo'])}, {fx(h['A']['hi'])}] | "
                         f"{fx(h['B']['gain'])} [{fx(h['B']['lo'])}, {fx(h['B']['hi'])}] | {dg.get(a)} | {m['sigma_draft']:.3f} | {m['beta_corr']:.4f} |")
    L += ["", "## R's rank coefficients (pp at a 55% deck; robust SE in brackets; bronze = 0)", "",
          "| set | fmt | " + " | ".join(RANK_COLS) + " |", "|---|---|" + "---:|" * len(RANK_COLS)]
    for sc, r in per_set.items():
        for fmt, fr in r["formats"].items():
            if fr.get("status") != "ok" or fr["arms"]["R"].get("no_signal"):
                continue
            rc = fr["arms"]["R"]["rank_coef"]
            L.append(f"| {sc} | {fmt} | " + " | ".join(f"{rc[c]['pp']:+.2f} ({rc[c]['se_pp']:.2f})" for c in RANK_COLS) + " |")
    L += ["", "## Ten largest card movers (final stage of the target format; logit; pp = logit x 24.75)", ""]
    for sc, r in per_set.items():
        for fmt, fr in r["formats"].items():
            if fr.get("status") != "ok":
                continue
            for a in ("P", "R"):
                m = fr["arms"][a]
                if m.get("no_signal"):
                    continue
                L += [f"### {sc} {fmt} {a}  (corr with M {m['beta_corr']:.4f})", "", "| card | M | arm | delta | delta pp |", "|---|---:|---:|---:|---:|"]
                for mv in m["movers"]:
                    L.append(f"| {mv['name']} | {mv['beta_M']:+.4f} | {mv['beta_arm']:+.4f} | {mv['delta']:+.4f} | {mv['delta'] * PP:+.2f} |")
                L.append("")
    path.write_text("\n".join(L), encoding="utf-8")


# ------------------------------------------------------------------------------------------------------ safety snapshots
def _tree(root: Path) -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    if not root.exists():
        return out
    for dp, _dn, fn in os.walk(root):
        for f in fn:
            p = os.path.join(dp, f)
            try:
                s = os.stat(p)
                out[p] = (s.st_size, s.st_mtime_ns)
            except OSError:
                pass
    return out


def snapshot(sets: list[str]) -> dict[str, dict]:
    """Size and mtime of: the real app home, the scratch home, and the registered raw 17Lands files of `sets`."""
    snap: dict[str, dict] = {}
    la = os.environ.get("LOCALAPPDATA")
    snap["real_home"] = _tree(Path(la) / "sealed_lab") if la else {}
    snap["scratch_home"] = _tree(config.app_home())
    raw: dict[str, tuple[int, int]] = {}
    for sc in sets:
        for f in registry.files(sc, include_shared=False):
            p = Path(f.path)
            if p.exists():
                s = p.stat()
                raw[str(p)] = (s.st_size, s.st_mtime_ns)
    snap["raw_files"] = raw
    return snap


def compare_snapshots(a: dict[str, dict], b: dict[str, dict]) -> dict:
    res = {}
    for k in a:
        changed = sorted(p for p in set(a[k]) | set(b[k]) if a[k].get(p) != b[k].get(p))
        res[k] = dict(n_files=len(a[k]), n_changed=len(changed), changed=changed[:20])
    return res


# ------------------------------------------------------------------------------------------------------ selfcheck
def selfcheck(sets_a: list[str], set_b: str, n_jobs: int, out: Path) -> int:
    """RULE.md implementation checks: (a) P with a keep-everything rank filter == M on `sets_a`; (b) R with zero indicators ~ M on `set_b`;
    (c) app home and raw files untouched."""
    sets_all = sorted(set(sets_a) | {set_b})
    before = snapshot(sets_all)
    print("snapshot before: " + ", ".join(f"{k}={len(v)} files" for k, v in before.items()), flush=True)
    opts = cardvalue.ChainOpts()
    worst_a, worst_b = 0.0, 0.0
    with glm.shared_pool(n_jobs):
        for sc in sets_all:
            inp, space, formats, _ed = prepare_set(sc, n_jobs)
            games = inp.games
            for i, fmt in enumerate(formats):
                gT = games.filter(pl.col("fmt") == fmt)
                split = validate._holdout_split(gT)
                if split is None:
                    continue
                hold_ids, t_cut, _n, _h = split
                g_tr = gT.filter(~pl.col("draft_id").is_in(hold_ids))
                tmp_f = out / "tmp" / f"{sc}_{fmt}"
                cs = i + 100
                st_m = validate.chain_stages(fmt, games, g_tr, t_cut, space)
                fits_m = fit_chain(st_m, fmt, space, tmp_f / "m", n_jobs, cs, opts)
                if sc in sets_a:
                    st_p = stages_p(fmt, games, g_tr, t_cut, space, keep=RANKS, keep_null=True)
                    assert st_p["draft"].n_games == st_m["draft"].n_games, "keep-all filter dropped games"
                    fits_p = fit_chain(st_p, fmt, space, tmp_f / "p", n_jobs, cs, opts)
                    d = max_abs_diff(fits_p, fits_m)
                    mx = max(d.values())
                    worst_a = max(worst_a, mx)
                    print(f"[{sc} {fmt}] (a) P keep-all vs M: max|d beta_total|={mx:.3e} per stage={ {k: f'{v:.2e}' for k, v in d.items()} }", flush=True)
                if sc == set_b:
                    st_rr, _k0 = stages_r(st_m, zero=True)
                    fits_r = fit_chain(st_rr, fmt, space, tmp_f / "r", n_jobs, cs, opts)
                    d = max_abs_diff(fits_r, fits_m)
                    mx = max(d.values())
                    worst_b = max(worst_b, mx)
                    sg = {n: (float(fits_r[n].sigma), float(fits_m[n].sigma)) for n in fits_m}
                    print(f"[{sc} {fmt}] (b) R zero indicators vs M: max|d beta_total|={mx:.3e} per stage={ {k: f'{v:.2e}' for k, v in d.items()} } "
                          f"sigma (R,M)={sg}", flush=True)
                cardvalue.cleanup(str(tmp_f))
    after = snapshot(sets_all)
    cmp = compare_snapshots(before, after)
    print("(c) untouched check: " + "; ".join(f"{k}: {v['n_files']} files, {v['n_changed']} changed" for k, v in cmp.items()), flush=True)
    for k, v in cmp.items():
        for p in v["changed"]:
            print(f"    CHANGED {k}: {p}")
    ok_a = worst_a < 1e-9
    ok_c = all(v["n_changed"] == 0 for v in cmp.values())
    print(f"\nselfcheck (a) max abs diff {worst_a:.3e} -> {'OK' if ok_a else 'FAIL'}; (b) max abs diff {worst_b:.3e} (reported); (c) {'OK' if ok_c else 'FAIL'}")
    return 0 if (ok_a and ok_c) else 1


# ------------------------------------------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", type=Path, nargs="?", default=Path(tempfile.gettempdir()) / "sealed_lab_plat_draft")
    ap.add_argument("--sets", nargs="+", default=DEFAULT_SETS)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--selfcheck", action="store_true", help="implementation checks of RULE.md (P keep-all == M on TMT and SOS; R with zero "
                                                            "indicators ~ M on TMT; home untouched), then exit")
    a = ap.parse_args(argv)
    check_home_not_real()
    out = a.out.resolve()
    check_out_not_home(out)
    out.mkdir(parents=True, exist_ok=True)
    if a.selfcheck:
        return selfcheck(["TMT", "SOS"], "TMT", a.jobs, out)

    sets = [s.upper() for s in a.sets]
    before = snapshot(sets)
    per_set: dict[str, dict] = {}
    t_all = time.time()
    with glm.shared_pool(a.jobs):
        for sc in sets:
            done = out / f"per_set_{sc}.json"      # a finished set is kept, so a crash in a later set loses nothing
            if done.exists():
                per_set[sc] = json.loads(done.read_text(encoding="utf-8"))
                print(f"[{sc}] reused {done.name}", flush=True)
                continue
            idx = DEFAULT_SETS.index(sc) if sc in DEFAULT_SETS else len(DEFAULT_SETS)
            per_set[sc] = run_set(sc, out, a.jobs, idx)
            done.write_text(json.dumps(per_set[sc], indent=1, default=_json_default), encoding="utf-8")
            cardvalue.cleanup(str(out / "tmp"))
    untouched = compare_snapshots(before, snapshot(sets))

    pooled = {arm: ppc.summarize(per_set, lambda fr, arm=arm: (fr.get("arms") or {}).get(arm)) for arm in ARMS}
    diffs = {key: ppc.summarize(per_set, lambda fr, key=key: (fr.get("diffs") or {}).get(key)) for key in ("R|P", "N|P")}
    summary = dict(
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), total_seconds=round(time.time() - t_all, 1),
        se_floor=SE_FLOOR, min_gain=MIN_GAIN, n_rep=N_REP, sets=sets, n_seed0=N_SEED0, home_untouched=untouched,
        per_set=per_set, pooled=pooled, pooled_diffs=diffs, pooled_diff_answers={k: pooled_diff_answers(v) for k, v in diffs.items()},
        choice=choose_arm(pooled),
    )
    out_json = out / "plat_draft_results.json"
    out_json.write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    write_md(summary, out / "plat_draft_results.md")
    print(f"\nWrote {out_json} and plat_draft_results.md in {summary['total_seconds']:.1f}s")
    for arm in ARMS:
        for line in ppc.fmt_pool_line(arm, pooled[arm]):
            print(line)
    for key, sm in diffs.items():
        for line in ppc.fmt_pool_line(f"loss({key.split('|')[0]}) - loss(P) (positive = P better)", sm):
            print(line)
    print("choice:", summary["choice"].get("chosen"), "| passing:", summary["choice"]["passing"])
    if any(v["n_changed"] for v in untouched.values()):
        print("WARNING: files changed during the run:", {k: v["changed"] for k, v in untouched.items() if v["n_changed"]})
    return 0


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    sys.exit(main())
