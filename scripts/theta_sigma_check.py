"""Is a CV-chosen prior SD for the structure / mana coefficients (theta_sigma="cv") better than the rule clip(4 x card sigma)?

    python scripts/theta_sigma_check.py holdout SET [SET ...] --out OUTDIR [--fmt bo3 bo1] [--jobs N] [--n-rep 500]
    python scripts/theta_sigma_check.py splithalf SH BASE --out OUTDIR [--sets MSH SOS ...] [--steps refit report] [--halves A B]

`holdout` (the shipping test): per set and Sealed format, the plain chain M is fitted on the training part twice - theta_sigma="rule"
(= the deployed plain chain) and "cv" - and both are scored on the SAME time hold-out as the report (validate.run_validation, as in
scripts/prior_strength_compare.py). Per format: log loss of both with 95% CIs, the paired gain of "cv" over "rule" [95% CI] (positive = cv
better; validate.cluster_metrics, events resampled), the gain on the earlier (A) / later (B) half of the hold-out events
(validate.half_gains) and the pipeline's own verdict (pipeline._verdict with confirmation: material gain MIN_GAIN, 95% CI above zero on
A, same sign and 90% bound above zero on B), plus the chosen card sigma / stacking weight / structure SD of each stage and the CV curve
of the structure SD. Uses SEALED_LAB_HOME (point it at a scratch copy of the set); writes OUTDIR/holdout_SET.json. The deployed model with
all its components is not refitted here (not cheap): "rule" IS the deployed plain chain, so a gain here is the gain the option would add to it.

`splithalf` (the mechanism test): SH = the split_half_search OUT folder of the Session 10b split-half homes (homes/SET_A|B, splits,
search_rows.jsonl; read only), BASE = the split_half_decompose `structure` OUT folder of the rule models (pools/SET.pkl, refit/SET_H.*).
Step `refit` re-fits each half's chain (scripts/split_half_decompose.py HalfSetup / refit_chain_cov) with the card sigma of the half's
artifact held fixed and, in the Sealed stages, the structure SD and then the stacking weight chosen by cardvalue.select_theta_w (the same CV as
theta_sigma="cv"); it writes OUTDIR/refit/SET_H.{json,npz} (the `ok` flag there compares with the artifact and is False by design). Step
`report` recomputes, on the EXISTING best-deck pairs of the pools, the structure (struct + mana) part of the half-model disagreement T with
the new coefficients and compares it with the rule's: mean and RMS of that part in win-rate points of self-preference (50 s T_sm), the
sampling-noise SD of it (pool_noise: coefficient covariance of both halves, conditional on the decks) and RMS/noise (1 = pure noise; the
diagnostic found about 2 for the rule in bo3). Only the structure coefficients change in this comparison (the card betas of the halves, hence
the cards part of T, are those of the rule models: a cv refit also moves them slightly through the shared prior). Needs the half homes,
which hold a full corpus copy; builds are heavy (minutes per half); nothing is written to them (the refit runs on the homes in place but only
reads them: copy them to scratch first if in doubt).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

SEALED = ("bo1", "bo3")


# ---------------------------------------------------------------------------------------------------------------------
# pure helpers (tested without any model)
# ---------------------------------------------------------------------------------------------------------------------

def stage_record(f) -> dict:
    """JSON-able prior-strength choices of a fitted stage (cardvalue.StageFit / validate.RefFit)."""
    from sealed_lab.model import cardvalue

    cv = getattr(f, "theta_cv_sd", None)
    return dict(sigma=float(f.sigma), w=float(getattr(f, "w", 1.0)), theta_sd=float(cardvalue.theta_sd_of(f)), theta_cv=cv is not None,
                theta_grid=[float(v) for v in getattr(f, "theta_grid", [])], theta_loss=[float(v) for v in getattr(f, "theta_loss", [])],
                theta_se=[float(v) for v in getattr(f, "theta_se", [])])


def verdict_row(gain: dict, halves: dict, fmt: str, confirm: bool = True) -> dict:
    """The pipeline's own decision for one format: pipeline._verdict on the whole-hold-out gain and the half gains."""
    from sealed_lab.model import pipeline

    better, worse, tiny, info = pipeline._verdict({fmt: dict(gain=gain["gain"], lo=gain["lo"], hi=gain["hi"], halves=halves)}, confirm)
    status = "better" if fmt in better else "worse" if fmt in worse else "tiny" if fmt in tiny else "no significant difference"
    return dict(verdict=status, min_gain=pipeline.MIN_GAIN, info=info)


def retarget_records(records: list[dict], ctx: dict, refit: dict, sc: str) -> list[dict]:
    """The pools' records with the structure part of T recomputed from the refit coefficients of `refit` ({half: load_refit(...)}).

    Per pool: tot/bpart/ppart = feature_T_parts(b_A, b_B, phi of the four (model, deck) pairs) with b = theta / sd (raw units, FEATURES order),
    T_sm = their sum. phi, s, T, pool are kept (the decks and feature values are those of the pools)."""
    import split_half_decompose as shd

    b = {}
    for h in "AB":
        names, arr, info = refit[h]["info"]["names"], refit[h]["arr"], refit[h]["info"]
        b[h] = {fm: shd.align_features(names, arr[f"{fm}__theta"])[0] / ctx[h]["sd"] for fm in info["formats"]}
    out = []
    for r in records:
        fm = r["fmt"]
        if fm not in b["A"] or fm not in b["B"]:
            continue
        phi = r["phi"]
        tot, bpart, ppart = shd.feature_T_parts(b["A"][fm], b["B"][fm], phi[0, 0], phi[0, 1], phi[1, 0], phi[1, 1])
        out.append(dict(r, tot=tot, bpart=bpart, ppart=ppart, T_sm=float(tot.sum())))
    return out


def retarget_ctx(ctx: dict, refit: dict) -> dict:
    """ctx of split_half_decompose.model_ctx with the coefficients b[fmt] of the refit (the covariances come from the refit via _stage_cells)."""
    import split_half_decompose as shd

    new = copy.deepcopy(ctx)
    for h in "AB":
        names, arr, info = refit[h]["info"]["names"], refit[h]["arr"], refit[h]["info"]
        new[h]["b"] = {fm: shd.align_features(names, arr[f"{fm}__theta"])[0] / ctx[h]["sd"] for fm in info["formats"]}
    return new


def structure_stats(recs: list[dict], noise: dict) -> dict:
    """Mean and RMS of the structure part of T in win-rate points of self-preference (50 s T_sm), with the pools' noise SD of it."""
    if not recs:
        return {}
    pp = np.array([50.0 * r["s"] * r["T_sm"] for r in recs])
    out = dict(n_pools=len(recs), mean_pp=float(pp.mean()), rms_pp=float(np.sqrt(np.mean(pp ** 2))))
    if noise:
        out.update(noise_rms_pp=float(noise["rms_noise_logit"] * np.mean([50.0 * r["s"] for r in recs])),
                   rms_over_noise=float(noise["rms_T_structure_logit"] / noise["rms_noise_logit"]) if noise["rms_noise_logit"] > 0 else float("nan"))
    return out


# ---------------------------------------------------------------------------------------------------------------------
# hold-out comparison
# ---------------------------------------------------------------------------------------------------------------------

def run_holdout(sets: list[str], out: Path, fmts: list[str], jobs: int | None, n_rep: int) -> None:
    import polars as pl

    from sealed_lab import config
    from sealed_lab.model import cardvalue, design, pipeline, structure, validate

    out.mkdir(parents=True, exist_ok=True)
    n_jobs = int(jobs or os.cpu_count() or 1)
    variants = {"rule": cardvalue.ChainOpts(), "cv": cardvalue.ChainOpts(theta_sigma="cv")}
    for sc in [s.upper() for s in sets]:
        t0 = time.time()
        inp = pipeline.load_inputs(sc)
        games, cards = inp.games, inp.cards
        keys, table = pipeline.phi_for_corpus(inp, n_jobs, None, None, config.cache_dir(sc))
        per_deck = games.group_by("deck_key").len().sort("deck_key")
        wts = np.zeros(len(keys))
        wts[np.searchsorted(keys, per_deck["deck_key"].to_numpy())] = per_deck["len"].to_numpy()
        space = design.build_deck_space(keys, table[:, :structure.N_FEATURES], inp.deck_cards, cards, weights=np.maximum(wts, 1e-9))
        formats = [f for f in pipeline.FMT_ORDER if games.filter(pl.col("fmt") == f).height >= 1]
        res: dict = {"set": sc, "formats": {}}
        tested: dict = {}
        for i, fmt in enumerate(formats):
            if fmt not in fmts:
                continue
            runs: dict = {}
            keeps: dict = {}
            for name, opts in variants.items():
                tmp = cardvalue.make_tmpdir(config.cache_dir(sc))
                try:
                    keep: dict = {}
                    # (the same seed as calculate(), so "rule" is the report's M)
                    runs[name] = validate.run_validation(fmt, games, space, cards, inp.lazy["hands"], tmp, n_jobs, seed=i, n_rep=n_rep,
                                                         keep=keep, opts=opts)
                    keeps[name] = keep.get(fmt)
                finally:
                    cardvalue.cleanup(tmp)
                print(f"{sc} {fmt} {name}: {time.time() - t0:.0f}s", flush=True)
            if runs["rule"].get("status") != "ok" or keeps["rule"] is None or keeps["cv"] is None:
                res["formats"][fmt] = {"status": runs["rule"].get("status"), "reason": runs["rule"].get("reason")}
                continue
            y, groups = keeps["rule"]["y"], keeps["rule"]["groups"]
            p = {n: keeps[n]["pM"] for n in variants}
            gT = games.filter(pl.col("fmt") == fmt)
            hold_ids = validate._holdout_split(gT)[0]
            g_ho = gT.filter(pl.col("draft_id").is_in(hold_ids))
            is_first = validate.half_masks(g_ho)
            met = validate.cluster_metrics(y, {"M": p["cv"], "B": p["rule"]}, groups, n_rep, 5 + i)
            d = met["B"]["vs_M"]["logloss"]                                           # positive = cv better than rule
            gain = dict(gain=d["diff"], lo=d["lo"], hi=d["hi"])
            halves = validate.half_gains(y, p["rule"], p["cv"], groups, is_first, n_rep, i + 7)
            row = dict(status="ok", n_games=int(len(y)), n_events=int(groups.max() + 1),
                       logloss={n: dict(value=runs[n]["metrics"]["M"]["logloss"], ci=runs[n]["metrics"]["M"]["logloss_ci"]) for n in variants},
                       gain_cv_vs_rule=gain, halves=halves, **verdict_row(gain, halves, fmt),
                       stages={n: {st: stage_record(f) for st, f in keeps[n]["ref_fits"].items()} for n in variants})
            res["formats"][fmt] = row
            tested[fmt] = dict(gain=gain["gain"], lo=gain["lo"], hi=gain["hi"], halves=halves)
            g = lambda d_: f"{d_['gain']:+.5f} [{d_['lo']:+.5f}, {d_['hi']:+.5f}]"          # noqa: E731
            print(f"{sc} {fmt}: logloss rule {row['logloss']['rule']['value']:.5f} cv {row['logloss']['cv']['value']:.5f} | gain cv-vs-rule "
                  f"{g(gain)} | halves A {g(halves['A'])} B {g(halves['B'])} | {row['verdict']}", flush=True)
            for st, v in row["stages"]["cv"].items():
                r0 = row["stages"]["rule"][st]
                print(f"    {st}: card sigma rule {r0['sigma']:.4f} / cv {v['sigma']:.4f}; theta SD rule {r0['theta_sd']:.4f} -> cv {v['theta_sd']:.4f}"
                      f"{'' if v['theta_cv'] else ' (rule: no CV)'}; w rule {r0['w']:.1f} / cv {v['w']:.1f}", flush=True)
        if tested:
            better, worse, tiny, info = pipeline._verdict(tested, True)
            res["verdict"] = dict(better=better, worse=worse, tiny=tiny, confirmation=info)
        res["seconds"] = time.time() - t0
        (out / f"holdout_{sc}.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")


# ---------------------------------------------------------------------------------------------------------------------
# split-half structure disagreement
# ---------------------------------------------------------------------------------------------------------------------

def make_chooser(tmp_root: Path, chosen: dict):
    """The `choose` hook of split_half_decompose.refit_chain_cov: Sealed stages get the CV-chosen structure SD, then the stacking weight."""
    from sealed_lab.model import cardvalue

    opts = cardvalue.ChainOpts(theta_sigma="cv")

    def choose(st, name, prior, sigma, w, has_prior, spec):
        if name not in SEALED:
            return spec, w
        tmp = cardvalue.make_tmpdir(tmp_root)
        try:
            spec2, info = cardvalue.select_theta_w(st, prior, spec, sigma, tmp, 1, seed=1, opts=opts, has_prior=has_prior)
        finally:
            cardvalue.cleanup(tmp)
        chosen[name] = dict(theta=info.get("theta"), w_loss=info.get("w_loss"), w_artifact=float(w), w=float(info.get("w", w)))
        return spec2, float(info.get("w", w))

    return choose


def run_refit(sh: Path, out: Path, sets: list[str], halves: list[str]) -> None:
    import split_half_decompose as shd

    (out / "tmp").mkdir(parents=True, exist_ok=True)
    for sc in sets:
        for h in halves:
            chosen: dict = {}
            t0 = time.time()
            info = shd.structure_refit_half(sh, sc, h, out, choose=make_chooser(out / "tmp", chosen))
            p = out / "refit" / f"{sc}_{h}.json"
            info["chosen"] = chosen
            p.write_text(json.dumps(info, indent=1, default=float), encoding="utf-8")
            print(f"refit-cv {sc} {h}: {time.time() - t0:.0f}s; " + "; ".join(
                f"{n}: theta SD {v['theta']['value']:.4f} (loss at 0.05: {v['theta']['loss'][3]:.5f}, best {min(v['theta']['loss']):.5f}), w {v['w_artifact']:.1f}->{v['w']:.1f}"
                for n, v in chosen.items() if v.get("theta")), flush=True)


def run_report(base: Path, out: Path, sets: list[str]) -> str:
    import split_half_decompose as shd

    sp = shd.load_structure_pools(base, sets)
    rows, md = [], ["# theta_sigma=cv: the structure part of the split-half disagreement T", "",
                    "Mean / RMS of the structure (struct + mana) part of T in win-rate points of self-preference on the existing best-deck pairs; "
                    "noise = sampling SD of that part (coefficient covariance of both halves, conditional on the decks); RMS/noise = 1 for pure noise.", "",
                    "| set | fmt | pools | rule: mean pp | RMS pp | noise pp | RMS/noise | cv: mean pp | RMS pp | noise pp | RMS/noise | theta SD rule (A/B) | cv (A/B) | w rule -> cv (A/B) |",
                    "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|"]
    pooled: dict = {}
    for sc in sets:
        if sc not in sp:
            continue
        base_refit = {h: shd.load_refit(base, sc, h) for h in "AB"}
        cv_refit = {h: shd.load_refit(out, sc, h) for h in "AB"}
        if any(x is None for x in list(base_refit.values()) + list(cv_refit.values())):
            print(f"report {sc}: refits missing (base or cv), skipped")
            continue
        if not all(x["info"]["ok"] for x in base_refit.values()):
            print(f"report {sc}: the rule refit does not reproduce the artifact (not modelled: "
                  f"{[x['info']['not_modelled'] for x in base_refit.values()]}), skipped")
            continue
        ctx = sp[sc]["ctx"]
        recs_rule = sp[sc]["records"]
        # the rule refit must reproduce the recorded structure part (the same arithmetic as the cv numbers)
        chk = retarget_records(recs_rule, ctx, base_refit, sc)
        print(f"report {sc}: rule recomputation vs recorded T_sm max abs "
              f"{max(abs(a['T_sm'] - b['T_sm']) for a, b in zip(chk, recs_rule)):.1e}", flush=True)
        recs_cv = retarget_records(recs_rule, ctx, cv_refit, sc)
        sp_rule = {sc: dict(records=recs_rule, ctx=ctx)}
        sp_cv = {sc: dict(records=recs_cv, ctx=retarget_ctx(ctx, cv_refit))}
        cells_rule = shd._stage_cells(sp_rule, {(sc, h): base_refit[h] for h in "AB"}, [sc])
        cells_cv = shd._stage_cells(sp_cv, {(sc, h): cv_refit[h] for h in "AB"}, [sc])
        for fm in ("ALL",) + shd.FMTS:
            sel = [r for r in recs_rule if fm == "ALL" or r["fmt"] == fm]
            selc = [r for r in recs_cv if fm == "ALL" or r["fmt"] == fm]
            if not sel:
                continue
            st_rule = structure_stats(sel, shd.pool_noise(sel, cells_rule))
            st_cv = structure_stats(selc, shd.pool_noise(selc, cells_cv))
            rows.append(dict(set=sc, fmt=fm, rule=st_rule, cv=st_cv))
            for key, rs, cs in (("%s|%s" % (fm, "ALL"), sel, selc),):
                pooled.setdefault(fm, dict(rule=[], cv=[], cells_rule=[], cells_cv=[]))
                pooled[fm]["rule"] += sel
                pooled[fm]["cv"] += selc
            pooled[fm]["cells_rule"] += cells_rule
            pooled[fm]["cells_cv"] += cells_cv

            def th(r, f_):
                return "/".join(f"{(r[h]['info']['stages'].get(f_) or {}).get('sig_theta', float('nan')):.4f}" for h in "AB")

            def ww(f_):
                return "/".join(f"{(base_refit[h]['info']['stages'].get(f_) or {}).get('w', float('nan')):.1f}->{(cv_refit[h]['info']['stages'].get(f_) or {}).get('w', float('nan')):.1f}" for h in "AB")
            f_ = fm if fm != "ALL" else "bo3"
            md.append(f"| {sc} | {fm} | {st_rule['n_pools']} | {st_rule['mean_pp']:.3f} | {st_rule['rms_pp']:.3f} | {st_rule.get('noise_rms_pp', float('nan')):.3f} | "
                      f"{st_rule.get('rms_over_noise', float('nan')):.2f} | {st_cv['mean_pp']:.3f} | {st_cv['rms_pp']:.3f} | {st_cv.get('noise_rms_pp', float('nan')):.3f} | "
                      f"{st_cv.get('rms_over_noise', float('nan')):.2f} | {th(base_refit, f_)} | {th(cv_refit, f_)} | {ww(f_)} |")
    md += ["", "## Pooled over the sets", "", "| fmt | pools | rule: mean pp | RMS pp | RMS/noise | cv: mean pp | RMS pp | RMS/noise |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    summary = {}
    for fm, d in pooled.items():
        a = structure_stats(d["rule"], shd.pool_noise(d["rule"], d["cells_rule"]))
        b = structure_stats(d["cv"], shd.pool_noise(d["cv"], d["cells_cv"]))
        summary[fm] = dict(rule=a, cv=b)
        md.append(f"| {fm} | {a['n_pools']} | {a['mean_pp']:.3f} | {a['rms_pp']:.3f} | {a.get('rms_over_noise', float('nan')):.2f} | {b['mean_pp']:.3f} | "
                  f"{b['rms_pp']:.3f} | {b.get('rms_over_noise', float('nan')):.2f} |")
    text = "\n".join(md) + "\n"
    out.mkdir(parents=True, exist_ok=True)
    (out / "theta_structure_T.md").write_text(text, encoding="utf-8")
    (out / "theta_structure_T.json").write_text(json.dumps(dict(rows=rows, pooled=summary), indent=1, default=float), encoding="utf-8")
    return text


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("holdout")
    p.add_argument("sets", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--fmt", nargs="+", default=["bo3", "bo1"])
    p.add_argument("--jobs", type=int, default=None)
    p.add_argument("--n-rep", type=int, default=500)
    p = sub.add_parser("splithalf")
    p.add_argument("sh")
    p.add_argument("base")
    p.add_argument("--out", required=True)
    p.add_argument("--sets", nargs="+", default=["MSH", "SOS", "FRA", "HOB"])
    p.add_argument("--halves", nargs="+", default=["A", "B"], choices=["A", "B"])
    p.add_argument("--steps", nargs="+", default=["refit", "report"], choices=["refit", "report"])
    a = ap.parse_args(argv)
    if a.cmd == "holdout":
        run_holdout(a.sets, Path(a.out), a.fmt, a.jobs, a.n_rep)
        return 0
    out = Path(a.out)
    sets = [s.upper() for s in a.sets]
    if "refit" in a.steps:
        run_refit(Path(a.sh), out, sets, a.halves)
    if "report" in a.steps:
        print(run_report(Path(a.base), out, sets))
    return 0


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    sys.exit(main())
