"""Pre-registered hierarchical pilot-skill shrinkage test by n_games_bucket (docs/skill_shrink/RULE.md).

Reads the app home strictly read-only and writes scratch artefacts + JSON summary to OUT.

    python scripts/skill_shrink_check.py OUT [--sets ...] [--jobs N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp
from scipy.special import expit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from sealed_lab import config  # noqa: E402
from sealed_lab.model import cardvalue, castability, design, glm, pipeline, pooling, structure, validate, vcache  # noqa: E402

ALL_SETS = ["ECL", "EOE", "FIN", "FRA", "HOB", "MSH", "OM1", "SOS", "TDM", "TLA", "TMT"]
K_GRID = (25.0, 50.0, 100.0, 200.0)
SE_FLOOR = 1e-5
GAME_COLS_EXT = pipeline.GAME_COLS + ["n_games_bucket"]


def check_out_not_home(out: Path) -> None:
    home = config.app_home().resolve()
    out_r = out.resolve()
    if out_r == home or home in out_r.parents:
        raise RuntimeError(f"refusing to write inside app home {home}")


def read_phi_readonly(inp: pipeline.Inputs, cache_dir: Path, n_jobs: int) -> tuple[np.ndarray, np.ndarray]:
    decks = inp.decks.sort("deck_key")
    keys = decks["deck_key"].to_numpy()
    hashes = decks["deck_hash"].to_numpy()
    sig = structure.cards_signature(inp.cards, 1, None)
    phi = np.full((len(keys), structure.N_TABLE), np.nan)
    cpath = pipeline.phi_cache_path(cache_dir, 1, None)
    if cpath.exists():
        with np.load(cpath, allow_pickle=False) as z:
            if str(z["sig"]) == sig and len(z["hashes"]):
                ch, cp = z["hashes"], z["phi"]
                order = np.argsort(ch)
                pos = np.minimum(np.searchsorted(ch[order], hashes), len(ch) - 1)
                hit = ch[order][pos] == hashes
                phi[hit] = cp[order][pos[hit]]
    todo = np.nonzero(np.isnan(phi[:, 0]))[0]
    if len(todo):
        offsets, cid, cnt = pipeline.deck_arrays(keys[todo], inp.deck_cards)
        phi[todo] = structure.compute_phi(offsets, cid, cnt, inp.cards, n_jobs=n_jobs, version=1, regime=None)
    return keys, phi


def apply_skill_shrink(g: pl.DataFrame, k0: float) -> pl.DataFrame:
    """Empirical-Bayes shrinkage of `skill` towards SKILL_CENTER (0.55) using `n_games_bucket`."""
    sk = g["skill"].to_numpy().astype(float)
    ng = g["n_games_bucket"].to_numpy().astype(float)
    n_eff = np.where(np.isfinite(ng) & (ng > 0.0), ng, float(k0))
    shrunk = np.where(np.isnan(sk), np.nan, (n_eff * sk + float(k0) * design.SKILL_CENTER) / (n_eff + float(k0)))
    rel = np.where(np.isfinite(ng) & (ng > 0.0), np.log1p(ng / 100.0), 0.0)
    return g.with_columns(
        pl.Series("skill", shrunk, dtype=pl.Float64),
        pl.Series("_skill_rel", rel, dtype=pl.Float64),
    )


def make_problem_with_rel(g: pl.DataFrame, enc: design.CovEncoder, space: design.DeckSpace,
                          clamp_time: bool = False) -> tuple[glm.Problem, np.ndarray, np.ndarray]:
    """make_problem with an extra reliability covariate `_skill_rel` appended to C."""
    prob, urows, inv = design.make_problem(g, enc, space, clamp_time=clamp_time)
    rel = g["_skill_rel"].to_numpy().astype(float)[:, None]
    C2 = sp.hstack([prob.C, sp.csr_matrix(rel)], format="csr")
    return glm.Problem(y=prob.y, C=C2, D=prob.D, idx=prob.idx), urows, inv


def build_stage_shrunk(name: str, g: pl.DataFrame, space: design.DeckSpace, with_rel: bool) -> tuple[design.StageData, np.ndarray]:
    st = design.build_stage(name, g, space)
    if not with_rel:
        return st, st.enc.sigma
    prob2, _, _ = make_problem_with_rel(g, st.enc, space, clamp_time=False)
    sig2 = np.concatenate([st.enc.sigma, [1.0]])
    from dataclasses import replace

    return replace(st, prob=prob2), sig2


def fit_chain_custom(stages_info: dict[str, tuple[design.StageData, np.ndarray]], ref_fits: dict,
                     n_card: int, n_feat: int, prior_ok: set[str]) -> dict[str, np.ndarray]:
    order = [n for n in ("draft", "bo1", "bo3") if n in stages_info]
    out: dict[str, np.ndarray] = {}
    prev_deck = prev_name = None
    p_deck = n_card + n_feat
    for name in order:
        st, sig_cov = stages_info[name]
        rf = ref_fits[name]
        if prev_deck is None:
            prior = np.zeros(p_deck)
        elif prev_name == "draft":
            # Re-estimate lambda with custom sig_cov
            s = np.asarray(st.prob.D @ prev_deck).ravel()[st.prob.idx]
            C_lam = sp.hstack([st.prob.C, sp.csr_matrix(s[:, None])], format="csr")
            sig_lam = np.concatenate([sig_cov, [10.0]])
            c_lam = glm.fit(glm.Problem(y=st.prob.y, C=C_lam), np.zeros(len(sig_lam)), sig_lam, w=st.w, maxiter=200).coef
            prior = float(np.clip(c_lam[-1], *cardvalue.LAMBDA_CLIP)) * prev_deck
        else:
            prior = prev_deck.copy()
        lo, hi = cardvalue.THETA_CLIP[name]
        spec = glm.SigmaSpec(sig_cov=sig_cov, n_card=n_card, n_theta=n_feat, kappa=4.0, th_lo=lo, th_hi=hi, th_abs=rf.theta_cv_sd)
        coef, _, _ = cardvalue.fit_stage(st, prior, spec, rf.sigma, with_se=False)
        if rf.w < 1.0:
            coef = glm.stack_coef(st.prob, coef, prior, rf.w, sig_cov, st.w)
        out[name] = coef
        if name in prior_ok:
            prev_deck, prev_name = coef[st.prob.k:], name
    return out


def paired_gain_metrics(y: np.ndarray, p_base: np.ndarray, p_new: np.ndarray, groups: np.ndarray, is_first: np.ndarray,
                        n_rep: int = 500, seed: int = 0) -> dict:
    met = validate.cluster_metrics(y, {"M": p_base, "NEW": p_new}, groups, n_rep, seed + 5)
    d = met["NEW"]["vs_M"]["logloss"]
    halves = validate.half_gains(y, p_base, p_new, groups, is_first, n_rep, seed + 7)
    return dict(gain=float(-d["diff"]), lo=float(-d["hi"]), hi=float(-d["lo"]),
                logloss_base=float(met["M"]["logloss"]), logloss_new=float(met["NEW"]["logloss"]),
                halves=halves)


def pool_with_floor(cells: dict[str, dict], se_floor: float) -> dict:
    out = {}
    for p in pooling.PARTS:
        ys = [c[p]["gain"] for c in cells.values()]
        ses = [max(pooling.se_from_ci(c[p]["lo"], c[p]["hi"]), se_floor) for c in cells.values()]
        out[p] = pooling.dersimonian_laird(ys, ses)
    return dict(pooled=out, verdict=pooling.verdict(out))


def load_inputs_ext(set_code: str) -> pipeline.Inputs:
    from sealed_lab.ingest.corpus import load_corpus, read_manifest

    c = load_corpus(set_code)
    games = c["games"].select(GAME_COLS_EXT).collect()
    return pipeline.Inputs(set_code=set_code, cards=c["cards"], games=games, decks=c["decks"],
                           deck_cards=c["deck_cards"].collect(), builds=c["builds"],
                           manifest=read_manifest(set_code) or {}, lazy=c)


def eval_arm(fmt: str, games: pl.DataFrame, g_tr: pl.DataFrame, g_ho: pl.DataFrame, t_cut, space: design.DeckSpace,
             ref_fits: dict, k0: float, with_rel: bool) -> tuple[np.ndarray, np.ndarray]:
    gs = apply_skill_shrink(games, k0)
    gtr_s = apply_skill_shrink(g_tr, k0)
    gho_s = apply_skill_shrink(g_ho, k0)

    n_games = {f: int(gs.filter(pl.col("fmt") == f).height) for f in ("draft", "bo1", "bo3")}
    stages_info: dict[str, tuple[design.StageData, np.ndarray]] = {}
    for name in validate.prior_stage_names(fmt, n_games):
        gp = gs.filter(pl.col("fmt") == name)
        evp = design.event_starts(gp)
        ids = evp.filter(pl.col("start") < t_cut)["draft_id"].to_list()
        gp = gp.filter(pl.col("draft_id").is_in(ids))
        if gp.height >= validate.MIN_PRIOR_GAMES:
            stages_info[name] = build_stage_shrunk(name, gp, space, with_rel)
    stages_info[fmt] = build_stage_shrunk(fmt, gtr_s, space, with_rel)
    prior_ok = set(stages_info) - {fmt}

    coefs = fit_chain_custom(stages_info, ref_fits, space.n_card, space.n_feat, prior_ok)
    st_t = stages_info[fmt][0]
    if with_rel:
        prob_ho, _, _ = make_problem_with_rel(gho_s, st_t.enc, space, clamp_time=True)
    else:
        prob_ho, _, _ = design.make_problem(gho_s, st_t.enc, space, clamp_time=True)
    beta_cards = coefs[fmt][st_t.prob.k:st_t.prob.k + space.n_card]
    return expit(prob_ho.eta(coefs[fmt])), beta_cards


def choose_k_inner(fmt: str, games: pl.DataFrame, g_tr: pl.DataFrame, space: design.DeckSpace,
                   ref_fits: dict, with_rel: bool) -> float:
    inner = validate.time_split(g_tr)
    if inner is None:
        return 50.0
    iv_ids, t_in = inner
    g_it = g_tr.filter(~pl.col("draft_id").is_in(iv_ids))
    g_iv = g_tr.filter(pl.col("draft_id").is_in(iv_ids))
    y_iv = g_iv["won"].cast(pl.Float64).to_numpy()
    best_k, best_ll = 50.0, float("inf")
    for k0 in K_GRID:
        p_iv, _ = eval_arm(fmt, games, g_it, g_iv, t_in, space, ref_fits, k0, with_rel)
        ll = validate.log_loss(y_iv, p_iv)
        if ll < best_ll:
            best_ll, best_k = ll, k0
    return best_k


def run_set(sc: str, out: Path, n_jobs: int) -> dict:
    t0 = time.time()
    inp = load_inputs_ext(sc)
    inp.cards = castability.with_cost_groups(inp.cards, sc)
    rep_path = config.model_dir(sc) / "report.json"
    ed_applied = False
    if rep_path.exists():
        try:
            rep = json.loads(rep_path.read_text(encoding="utf-8"))
            ed_applied = bool((rep.get("extra_draft") or {}).get("applied", False))
        except Exception:
            ed_applied = False
    if pipeline.extra_draft_events(inp.games) and not ed_applied:
        inp = pipeline.without_extra_draft(inp)

    keys, table = read_phi_readonly(inp, config.cache_dir(sc), n_jobs)
    space, _ = pipeline.plain_space(inp, keys, table)
    games_ext = inp.games
    games_std = games_ext.select(pipeline.GAME_COLS)

    n_games = {f: int(games_std.filter(pl.col("fmt") == f).height) for f in ("draft", "bo1", "bo3")}
    formats = [f for f in pipeline.FMT_ORDER if n_games.get(f, 0) >= 1]
    opts = cardvalue.ChainOpts()
    vc_ro = pipeline.val_cache(sc, games_std, inp.cards)

    fmt_res: dict[str, dict] = {}
    for i, fmt in enumerate(formats):
        seed_f = i
        gT = games_ext.filter(pl.col("fmt") == fmt)
        split = validate._holdout_split(gT)
        if split is None:
            fmt_res[fmt] = dict(status="skipped", reason=f"too few {fmt} events/games")
            continue
        hold_ids, t_cut, _nev, _nho = split
        g_tr = gT.filter(~pl.col("draft_id").is_in(hold_ids))
        g_ho = gT.filter(pl.col("draft_id").is_in(hold_ids))

        core_key = vc_ro.key("val", f"{fmt}_core", fmt=fmt, seed=int(seed_f), n_rep=validate.N_REP,
                             space=vcache.space_digest(space), opts=opts.key())
        core_hit = vc_ro.load("val", f"{fmt}_core", core_key)
        if core_hit is not None:
            ref_fits = {n: validate.RefFit.from_dict(d) for n, d in core_hit[0]["ref"].items()}
        else:
            tmp_m = out / "tmp" / f"{sc}_{fmt}_m"
            tmp_m.mkdir(parents=True, exist_ok=True)
            st_m = validate.chain_stages(fmt, games_std, g_tr.select(pipeline.GAME_COLS), t_cut, space)
            fits_m = cardvalue.run_chain(st_m, space.n_card, space.n_feat, str(tmp_m), n_jobs, seed=seed_f + 100,
                                         prior_ok=set(st_m) - {fmt}, opts=opts)
            ref_fits = {n: validate.RefFit.from_fit(f) for n, f in fits_m.items()}

        # Plain chain M baseline (verify exact match against cached pM)
        st_plain = validate.chain_stages(fmt, games_std, g_tr.select(pipeline.GAME_COLS), t_cut, space)
        stages_plain_info = {n: (st_, st_.enc.sigma) for n, st_ in st_plain.items()}
        coefs_m = fit_chain_custom(stages_plain_info, ref_fits, space.n_card, space.n_feat, set(st_plain) - {fmt})
        prob_ho_m, _, _ = design.make_problem(g_ho.select(pipeline.GAME_COLS), st_plain[fmt].enc, space, clamp_time=True)
        pM = expit(prob_ho_m.eta(coefs_m[fmt]))
        beta_m = coefs_m[fmt][st_plain[fmt].prob.k:st_plain[fmt].prob.k + space.n_card]
        y = prob_ho_m.y
        groups = np.unique(g_ho["draft_id"].to_numpy(), return_inverse=True)[1]
        is_first = validate.half_masks(g_ho)

        # Primary arm 1: k0=50 + _skill_rel (exact RULE.md specification)
        p_k50_rel, beta_k50_rel = eval_arm(fmt, games_ext, g_tr, g_ho, t_cut, space, ref_fits, 50.0, True)
        m_k50_rel = paired_gain_metrics(y, pM, p_k50_rel, groups, is_first, seed=seed_f)
        m_k50_rel["beta_corr"] = float(np.corrcoef(beta_m, beta_k50_rel)[0, 1])

        # Ablation arm 2: k0=50 shrunk skill only (no extra _skill_rel column)
        p_k50_only, beta_k50_only = eval_arm(fmt, games_ext, g_tr, g_ho, t_cut, space, ref_fits, 50.0, False)
        m_k50_only = paired_gain_metrics(y, pM, p_k50_only, groups, is_first, seed=seed_f)
        m_k50_only["beta_corr"] = float(np.corrcoef(beta_m, beta_k50_only)[0, 1])

        # Secondary tuned arm 3: k0 chosen on inner split + _skill_rel
        k_best = choose_k_inner(fmt, games_ext, g_tr, space, ref_fits, True)
        p_tuned, beta_tuned = (p_k50_rel, beta_k50_rel) if k_best == 50.0 else eval_arm(fmt, games_ext, g_tr, g_ho, t_cut, space, ref_fits, k_best, True)
        m_tuned = paired_gain_metrics(y, pM, p_tuned, groups, is_first, seed=seed_f)
        m_tuned["k_chosen"] = float(k_best)
        m_tuned["beta_corr"] = float(np.corrcoef(beta_m, beta_tuned)[0, 1])

        fmt_res[fmt] = dict(
            status="ok", cache_hit=core_hit is not None,
            n_holdout_games=int(len(y)), n_holdout_events=int(groups.max()) + 1,
            k50_rel=m_k50_rel,
            k50_only=m_k50_only,
            tuned_rel=m_tuned,
        )
        print(f"[{sc} {fmt}] ho={len(y):5d}g | k50+rel={m_k50_rel['gain']*1e4:+6.2f} [{m_k50_rel['lo']*1e4:+6.2f},{m_k50_rel['hi']*1e4:+6.2f}] | "
              f"k50_only={m_k50_only['gain']*1e4:+6.2f} [{m_k50_only['lo']*1e4:+6.2f},{m_k50_only['hi']*1e4:+6.2f}] | "
              f"tuned(k={k_best:g})={m_tuned['gain']*1e4:+6.2f}", flush=True)

    return dict(set=sc, seconds=round(time.time() - t0, 1), formats=fmt_res)


def summarize_arm(per_set: dict[str, dict], arm_key: str) -> dict:
    out_fmt = {}
    for fmt in ("bo1", "bo3"):
        cells = {}
        excluded = []
        for sc, r in per_set.items():
            fr = r["formats"].get(fmt)
            if not fr or fr.get("status") != "ok":
                excluded.append(dict(set=sc, reason=(fr or {}).get("reason", "absent")))
                continue
            m = fr[arm_key]
            cell = pooling.make_cell(m, m["halves"])
            if cell is None:
                excluded.append(dict(set=sc, reason="zero-width or non-finite CI"))
            else:
                cells[sc] = cell
        if not cells:
            out_fmt[fmt] = dict(n_sets=0, excluded=excluded)
            continue
        raw_pool = pooling.pool_parts(list(cells.values()))
        raw_v = pooling.verdict(raw_pool)
        floored = pool_with_floor(cells, SE_FLOOR)
        out_fmt[fmt] = dict(
            n_sets=len(cells), sets=sorted(cells), excluded=excluded,
            raw=dict(pooled=raw_pool, verdict=raw_v),
            se_floored=floored,
        )
    qual_raw = pooling.qualifies({f: v["raw"]["verdict"] for f, v in out_fmt.items() if v.get("n_sets")})
    qual_fl = pooling.qualifies({f: v["se_floored"]["verdict"] for f, v in out_fmt.items() if v.get("n_sets")})
    return dict(formats=out_fmt, passes_rule=bool(qual_raw and qual_fl))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", type=Path, nargs="?", default=Path(tempfile.gettempdir()) / "sealed_lab_skill_shrink")
    ap.add_argument("--sets", nargs="+", default=ALL_SETS)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    a = ap.parse_args(argv)
    out = a.out.resolve()
    check_out_not_home(out)
    out.mkdir(parents=True, exist_ok=True)

    per_set: dict[str, dict] = {}
    t_all = time.time()
    for sc in a.sets:
        per_set[sc] = run_set(sc.upper(), out, a.jobs)
        cardvalue.cleanup(str(out / "tmp"))

    summary = dict(
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        total_seconds=round(time.time() - t_all, 1),
        se_floor=SE_FLOOR,
        per_set=per_set,
        pooled={arm: summarize_arm(per_set, arm) for arm in ("k50_rel", "k50_only", "tuned_rel")},
    )
    out_json = out / "skill_shrink_results.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {out_json} in {summary['total_seconds']:.1f}s")
    for arm, sm in summary["pooled"].items():
        print(f"=== {arm} (PASS={sm['passes_rule']}) ===")
        for fmt, fv in sm["formats"].items():
            if fv.get("n_sets"):
                rw = fv["raw"]["pooled"]["whole"]
                ra = fv["raw"]["pooled"]["A"]
                rb = fv["raw"]["pooled"]["B"]
                rv = fv["raw"]["verdict"]
                flv = fv["se_floored"]["verdict"]
                print(f"  {fmt} ({fv['n_sets']} sets): whole={rw['mean']*1e4:+6.2f} [{rw['lo']*1e4:+6.2f},{rw['hi']*1e4:+6.2f}] | "
                      f"A={ra['mean']*1e4:+6.2f} [{ra['lo']*1e4:+6.2f},{ra['hi']*1e4:+6.2f}] | "
                      f"B={rb['mean']*1e4:+6.2f} (lo90={rv['B_lo90']*1e4:+6.2f}) | raw={rv['label']} | floored={flv['label']}")
    return 0


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    sys.exit(main())
