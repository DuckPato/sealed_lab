"""Confirmation test of the rank-corrected Draft stage (arm R of plat_draft) on sets not used before (docs/rank_draft/RULE.md).

A thin wrapper over scripts/plat_draft_check.py: the same M and R (same chain, hold-out split, seeds scheme), with P and N not run,
plus the reported split of the Bo1 gain by pilot skill (>= 0.60 vs the rest), and the rule's three-part decision:
  1. new sets, Bo1, pooled whole-hold-out gain: mean > 0 and one-sided 90% lower bound > 0;
  2. all 10 sets (new + FRA HOB MSH SOS TMT from docs/plat_draft/plat_draft_results.json), Bo1 pooled whole gain >= 1e-4 with 95% CI > 0;
  3. new sets: not pooled-worse in either format (whole, A or B).
Each must hold raw and with the SE >= 1e-5 floor.

Reads a scratch copy of the app home only (SEALED_LAB_HOME); writes only to OUT.

    python scripts/rank_draft_check.py OUT [--sets ECL EOE FIN TDM TLA] [--jobs N] [--selfcheck]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from scipy.special import expit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from sealed_lab.model import cardvalue, design, glm, pooling, validate  # noqa: E402
import cold_start_check as csc  # noqa: E402
import plat_draft_check as pdc  # noqa: E402
import play_prior_check as ppc  # noqa: E402

NEW_SETS = ["ECL", "EOE", "FIN", "TDM", "TLA"]
FIRST_RESULTS = HERE.parent / "docs" / "plat_draft" / "plat_draft_results.json"
SKILL_HI = 0.60
Z90 = 1.2815515655446004


def subset_metrics(y, p_m, p_r, g_ho: pl.DataFrame, mask: np.ndarray, seed: int) -> dict | None:
    if mask.sum() < 50:
        return dict(no_signal=True, reason=f"only {int(mask.sum())} games")
    gs = g_ho.filter(pl.Series(mask))
    groups = np.unique(gs["draft_id"].to_numpy(), return_inverse=True)[1]
    is_first = validate.half_masks(gs)
    m = csc.paired_gain_metrics(y[mask], p_m[mask], p_r[mask], groups, is_first, n_rep=pdc.N_REP, seed=seed)
    m["n_games"] = int(mask.sum())
    return m


def run_set(sc: str, out: Path, n_jobs: int) -> dict:
    t0 = time.time()
    inp, space, formats, ed_applied = pdc.prepare_set(sc, n_jobs)
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

        tm = time.time()
        st_m = validate.chain_stages(fmt, games, g_tr, t_cut, space)
        if "draft" not in st_m:
            fmt_res[fmt] = dict(status="skipped", reason="no Draft stage")
            continue
        fits_m = pdc.fit_chain(st_m, fmt, space, tmp_f / "m", n_jobs, cs, opts)
        sec = dict(M=round(time.time() - tm, 1))
        prob_ho, _, _ = design.make_problem(g_ho, st_m[fmt].enc, space, clamp_time=True)
        p_m = expit(prob_ho.eta(fits_m[fmt].coef))
        y = prob_ho.y
        groups = np.unique(g_ho["draft_id"].to_numpy(), return_inverse=True)[1]
        is_first = validate.half_masks(g_ho)

        tm = time.time()
        st_rr, k0 = pdc.stages_r(st_m)
        fits_r = pdc.fit_chain(st_rr, fmt, space, tmp_f / "r", n_jobs, cs, opts)
        p_r = expit(prob_ho.eta(fits_r[fmt].coef))
        fd = fits_r["draft"]
        rank_coef = {}
        for j, c in enumerate(pdc.RANK_COLS):
            v, s = float(fd.coef[k0 + j]), float(fd.se[k0 + j])
            rank_coef[c] = dict(logit=v, se_logit=s, pp=v * pdc.PP, se_pp=s * pdc.PP)
        r = pdc.arm_summary(y, p_m, p_r, groups, is_first, seed_f, fits_r, fits_m, fmt, names_by_col,
                            dict(draft_games=int(st_m["draft"].n_games), rank_coef=rank_coef, draft_converged=bool(fd.converged)))
        sec["R"] = round(time.time() - tm, 1)
        del st_rr

        sk = g_ho["skill"].to_numpy().astype(float)
        hi = np.nan_to_num(sk, nan=-1.0) >= SKILL_HI - 1e-9
        lo = np.isfinite(sk) & ~hi
        by_skill = dict(hi=subset_metrics(y, p_m, p_r, g_ho, hi, seed_f), lo=subset_metrics(y, p_m, p_r, g_ho, lo, seed_f),
                        n_skill_missing=int((~np.isfinite(sk)).sum()))

        fmt_res[fmt] = dict(status="ok", n_holdout_games=int(len(y)), n_holdout_events=int(groups.max()) + 1, t_cut=str(t_cut),
                            draft_games=dict(M=int(st_m["draft"].n_games)), seconds=sec,
                            sigma_draft=dict(M=float(fits_m["draft"].sigma), R=r["sigma_draft"]),
                            arms=dict(R=r), by_skill=by_skill)
        print(f"[{sc} {fmt}] ho={len(y):5d} draft games={st_m['draft'].n_games} | R={r['gain']*1e4:+6.2f} "
              f"[{r['lo']*1e4:+6.2f},{r['hi']*1e4:+6.2f}] corr={r['beta_corr']:.4f} | {time.time() - t0:.0f}s", flush=True)
        cardvalue.cleanup(str(tmp_f))
    return dict(set=sc, seconds=round(time.time() - t0, 1), extra_draft_applied=ed_applied, formats=fmt_res)


def whole(sm: dict, fmt: str, tag: str) -> dict | None:
    fv = sm["formats"].get(fmt) or {}
    if not fv.get("n_sets"):
        return None
    return (fv["raw"] if tag == "raw" else fv["se_floored"])["pooled"]["whole"]


def decide(pooled_new: dict, pooled_all: dict) -> dict:
    res = {}
    for tag in ("raw", "floored"):
        w_new = whole(pooled_new, "bo1", tag)
        w_all = whole(pooled_all, "bo1", tag)
        c1 = bool(w_new and w_new["mean"] > 0 and (w_new["mean"] - Z90 * w_new["se"]) > 0)
        c2 = bool(w_all and w_all["mean"] >= pdc.MIN_GAIN and w_all["lo"] > 0)
        key = "verdict" if tag == "raw" else None
        worse = []
        for f, fv in pooled_new["formats"].items():
            if fv.get("n_sets"):
                v = fv["raw"]["verdict"] if tag == "raw" else fv["se_floored"]["verdict"]
                if v["worse"]:
                    worse.append(f)
        c3 = not worse
        res[tag] = dict(c1_confirmed_bo1_new=c1, c2_large_enough_bo1_all10=c2, c3_worse_nowhere_new=c3, worse_formats=worse,
                        bo1_new_whole=w_new, bo1_new_lo90=(w_new["mean"] - Z90 * w_new["se"]) if w_new else None,
                        bo1_all10_whole=w_all, passes=c1 and c2 and c3)
        del key
    return dict(by_tag=res, passes=res["raw"]["passes"] and res["floored"]["passes"])


def selfcheck(sc: str, n_jobs: int, out: Path) -> int:
    """R with zero indicators reproduces M within fit tolerance on one new set (bo1)."""
    before = pdc.snapshot([sc])
    inp, space, formats, _ = pdc.prepare_set(sc, n_jobs)
    games = inp.games
    gT = games.filter(pl.col("fmt") == "bo1")
    hold_ids, t_cut, _a, _b = validate._holdout_split(gT)
    g_tr = gT.filter(~pl.col("draft_id").is_in(hold_ids))
    st_m = validate.chain_stages("bo1", games, g_tr, t_cut, space)
    opts = cardvalue.ChainOpts()
    with glm.shared_pool(n_jobs):
        fm = pdc.fit_chain(st_m, "bo1", space, out / "tmp" / "sc_m", n_jobs, 100, opts)
        st_z, _k0 = pdc.stages_r(st_m, zero=True)
        fz = pdc.fit_chain(st_z, "bo1", space, out / "tmp" / "sc_z", n_jobs, 100, opts)
    d = pdc.max_abs_diff(fz, fm)
    sig = {n: (float(fm[n].sigma), float(fz[n].sigma)) for n in fm}
    unt = pdc.compare_snapshots(before, pdc.snapshot([sc]))
    rep = dict(set=sc, max_abs_beta_diff=d, sigmas_M_Rzero=sig, untouched=unt)
    (out / "selfcheck.json").write_text(json.dumps(rep, indent=1, default=ppc._json_default), encoding="utf-8")
    print(json.dumps(rep, indent=1, default=ppc._json_default))
    cardvalue.cleanup(str(out / "tmp"))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--sets", nargs="+", default=NEW_SETS)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args(argv)
    pdc.check_home_not_real()
    out = a.out.resolve()
    pdc.check_out_not_home(out)
    out.mkdir(parents=True, exist_ok=True)
    if a.selfcheck:
        return selfcheck("ECL", a.jobs, out)

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
            per_set[sc] = run_set(sc, out, a.jobs)
            done.write_text(json.dumps(per_set[sc], indent=1, default=ppc._json_default), encoding="utf-8")
            cardvalue.cleanup(str(out / "tmp"))
    untouched = pdc.compare_snapshots(before, pdc.snapshot(sets))

    first = json.loads(FIRST_RESULTS.read_text(encoding="utf-8"))["per_set"]
    overlap = set(first) & set(per_set)
    if overlap:
        raise SystemExit(f"sets tested twice: {sorted(overlap)}")
    get_r = lambda fr: (fr.get("arms") or {}).get("R")  # noqa: E731
    pooled_new = ppc.summarize(per_set, get_r)
    pooled_all = ppc.summarize({**first, **per_set}, get_r)
    pooled_skill = {k: ppc.summarize(per_set, lambda fr, k=k: (fr.get("by_skill") or {}).get(k)) for k in ("hi", "lo")}
    decision = decide(pooled_new, pooled_all)
    summary = dict(created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), total_seconds=round(time.time() - t_all, 1),
                   rule="docs/rank_draft/RULE.md", sets=sets, first_sets=sorted(first), home_untouched=untouched,
                   per_set=per_set, pooled_new=pooled_new, pooled_all10=pooled_all, pooled_by_skill_new=pooled_skill,
                   decision=decision)
    (out / "rank_draft_results.json").write_text(json.dumps(summary, indent=2, default=ppc._json_default), encoding="utf-8")
    for line in ppc.fmt_pool_line("R, new sets", pooled_new) + ppc.fmt_pool_line("R, all 10 sets", pooled_all):
        print(line)
    for k, sm in pooled_skill.items():
        for line in ppc.fmt_pool_line(f"R, new sets, skill {'>=' if k == 'hi' else '<'} {SKILL_HI}", sm):
            print(line)
    for tag, v in decision["by_tag"].items():
        print(tag, {k: v[k] for k in ("c1_confirmed_bo1_new", "c2_large_enough_bo1_all10", "c3_worse_nowhere_new", "passes")},
              "bo1 new lo90:", v["bo1_new_lo90"])
    print("PASSES:", decision["passes"])
    if any(v["n_changed"] for v in untouched.values()):
        print("WARNING: files changed during the run:", {k: v["changed"] for k, v in untouched.items() if v["n_changed"]})
    print(f"total {summary['total_seconds']:.0f}s")
    return 0


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    sys.exit(main())
