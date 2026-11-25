"""Pre-registered Bo3 post-board matchup test using leak-free prior_opp_colors (docs/opp_pregame/RULE.md).

Reads the app home strictly read-only and writes scratch artefacts + JSON summary to OUT.

    python scripts/opp_pregame_check.py OUT [--sets ...] [--jobs N]
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
from sealed_lab.model import cardvalue, castability, design, glm, opponent as om, pipeline, pooling, structure, validate, vcache  # noqa: E402

ALL_SETS = ["ECL", "EOE", "FIN", "FRA", "HOB", "MSH", "OM1", "SOS", "TDM", "TLA", "TMT"]
BO3_EVENTS = ("TradDraft", "TradSealed")
SE_FLOOR = 5e-5


def check_out_not_home(out: Path) -> None:
    home = config.app_home().resolve()
    out_r = out.resolve()
    if out_r == home or home in out_r.parents:
        raise RuntimeError(f"refusing to write inside app home {home}")


def read_phi_readonly(inp: pipeline.Inputs, cache_dir: Path, n_jobs: int) -> tuple[np.ndarray, np.ndarray]:
    """Load phi_cache.npz strictly read-only; compute any uncached decks in memory without touching disk."""
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


def fit_chain_fast(stages: dict[str, design.StageData], ref_fits: dict | None, n_card: int, n_feat: int,
                   prior_ok: set[str], tmpdir: str, n_jobs: int, seed: int, opts: cardvalue.ChainOpts) -> dict[str, np.ndarray]:
    """Fit chain coefficients at cached RefFit sigmas/stack weights when available, else run full CV in tmpdir."""
    if ref_fits is not None and all(n in ref_fits for n in stages):
        order = [n for n in ("draft", "bo1", "bo3") if n in stages]
        out: dict[str, np.ndarray] = {}
        prev_deck = prev_name = None
        p_deck = n_card + n_feat
        for name in order:
            st, rf = stages[name], ref_fits[name]
            if prev_deck is None:
                prior = np.zeros(p_deck)
            elif prev_name == "draft":
                lam_raw, _ = cardvalue.fit_lambda(st, prev_deck)
                prior = float(np.clip(lam_raw, *cardvalue.LAMBDA_CLIP)) * prev_deck
            else:
                prior = prev_deck.copy()
            spec = cardvalue.make_spec(st, n_card, n_feat, name, th_abs=rf.theta_cv_sd)
            coef, _, _ = cardvalue.fit_stage(st, prior, spec, rf.sigma, with_se=False)
            if rf.w < 1.0:
                coef = glm.stack_coef(st.prob, coef, prior, rf.w, spec.sig_cov, st.w)
            out[name] = coef
            if name in prior_ok:
                prev_deck, prev_name = coef[st.prob.k:], name
        return out
    fits = cardvalue.run_chain(stages, n_card, n_feat, tmpdir, n_jobs, seed=seed + 100, prior_ok=prior_ok, opts=opts)
    return {n: f.coef for n, f in fits.items()}


def attach_prior_opp(g: pl.DataFrame) -> pl.DataFrame:
    """Attach leak-free `prior_opp_colors` (from strictly earlier games of the same Bo3 match) and `won_g1` (+0.5/-0.5/0.0)."""
    if g.height == 0:
        return g.with_columns(pl.lit("").alias("prior_opp_colors"), pl.lit(0.0).alias("won_g1"))
    ev = g["event"].fill_null("").to_list()
    did = g["draft_id"].fill_null("").to_list()
    mn = g["match_number"].fill_null(-1).to_list()
    gn = g["game_number"].fill_null(1).to_list()
    oc = g["opp_colors"].fill_null("").to_list()
    won = g["won"].fill_null(False).to_list()

    # Group row indices by Bo3 match
    matches: dict[tuple[str, str, int], list[int]] = {}
    for i, (e, d, m) in enumerate(zip(ev, did, mn)):
        if e in BO3_EVENTS and m >= 0 and d:
            matches.setdefault((e, d, int(m)), []).append(i)

    prior_oc = [""] * g.height
    won_g1 = np.zeros(g.height, dtype=float)
    for rows in matches.values():
        rows_sorted = sorted(rows, key=lambda idx: gn[idx])
        g1_val = 0.0
        for idx in rows_sorted:
            if gn[idx] == 1:
                g1_val = 0.5 if won[idx] else -0.5
                break
        # Union of opp_colors strictly before game_number `cur_gn`
        by_gn: dict[int, list[int]] = {}
        for idx in rows_sorted:
            by_gn.setdefault(int(gn[idx]), []).append(idx)
        seen_letters: set[str] = set()
        for gnum in sorted(by_gn):
            cur_union = "".join(c for c in "WUBRG" if c in seen_letters) if gnum >= 2 else ""
            for idx in by_gn[gnum]:
                prior_oc[idx] = cur_union
                won_g1[idx] = g1_val
            for idx in by_gn[gnum]:
                for ch in oc[idx]:
                    if ch in "WUBRG":
                        seen_letters.add(ch)

    return g.with_columns(
        pl.Series("prior_opp_colors", prior_oc, dtype=pl.String),
        pl.Series("won_g1", won_g1, dtype=pl.Float64),
    )


def pregame_head_matrix(df: pl.DataFrame) -> sp.csr_matrix:
    """[1, won_g1, pair_onehot(prior_opp_colors), opp_flag_matrix(prior_opp_colors)] -> (n, 19)."""
    n = df.height
    poc = df["prior_opp_colors"].fill_null("").to_list()
    wg1 = df["won_g1"].fill_null(0.0).to_numpy().astype(float)
    cat = np.array([design.OPP_LEVELS.index(design.norm_pair(v)) for v in poc], dtype=np.int64) if n else np.zeros(0, np.int64)
    P = sp.csr_matrix((np.ones(n), (np.arange(n), cat)), shape=(n, len(design.OPP_LEVELS)))
    F = sp.csr_matrix(design.opp_flag_matrix(poc))
    base = sp.csr_matrix(np.column_stack([np.ones(n), wg1]))
    return sp.hstack([base, P, F], format="csr")


HEAD_SIGMA = np.array([2.0, 2.0] + [0.5] * len(design.OPP_LEVELS) + [0.5] * 5 + [1.0], dtype=float)


def build_pooled_problem(parts: list[tuple[str, pl.DataFrame, sp.csr_matrix, np.ndarray, str]], n_card: int, mode: str) -> om.Pooled:
    """Build opponent.Pooled from [(tag, df, Dc_per_game, offset_eta, opp_col_name)]."""
    Xs, Ms, ys, offs, grp = [], [], [], [], []
    L = len(om.levels(mode))
    num = np.zeros((n_card, L))
    den = np.zeros(n_card)
    for tag, df, Dc, off, col in parts:
        O = om.opp_matrix(df[col].to_numpy(), mode)
        Xs.append(om.interaction_matrix(Dc, O))
        Ms.append(Dc)
        ys.append(df["won"].cast(pl.Float64).to_numpy())
        offs.append(off)
        grp.append(np.char.add(tag + ":", df["draft_id"].to_numpy().astype(str)))
        num += np.asarray(Dc.T @ O.todense())
        den += np.asarray(Dc.sum(0)).ravel()
    n_int = n_card * L
    X = sp.vstack([sp.hstack([x, m_], format="csr") for x, m_ in zip(Xs, Ms)], format="csr")
    g = np.unique(np.concatenate(grp), return_inverse=True)[1].astype(np.int32)
    glob = num.sum(0) / max(den.sum(), 1e-9)
    mass = np.where(den[:, None] > 0, num / np.maximum(den[:, None], 1e-9), glob[None, :])
    prob = glm.Problem(y=np.concatenate(ys), C=X, offset=np.concatenate(offs))
    return om.Pooled(prob=prob, group=g, mass=mass, n_games=prob.n, n_groups=int(g.max()) + 1, mode=mode, n_int=n_int)


def paired_gain_metrics(y: np.ndarray, p_base: np.ndarray, p_new: np.ndarray, groups: np.ndarray, is_first: np.ndarray,
                        n_rep: int = 500, seed: int = 0) -> dict:
    """Paired log-loss gain (base - new; positive = new better) on whole hold-out and halves A/B."""
    met = validate.cluster_metrics(y, {"M": p_base, "NEW": p_new}, groups, n_rep, seed + 5)
    d = met["NEW"]["vs_M"]["logloss"]       # NEW - M: positive = M better; flip sign so positive = NEW better
    halves = validate.half_gains(y, p_base, p_new, groups, is_first, n_rep, seed + 7)
    return dict(gain=float(-d["diff"]), lo=float(-d["hi"]), hi=float(-d["lo"]),
                logloss_base=float(met["M"]["logloss"]), logloss_new=float(met["NEW"]["logloss"]),
                halves=halves)


def pool_with_floor(cells: dict[str, dict], se_floor: float) -> dict:
    """DerSimonian-Laird pooling with a minimum per-study SE floor."""
    out = {}
    for p in pooling.PARTS:
        ys = [c[p]["gain"] for c in cells.values()]
        ses = [max(pooling.se_from_ci(c[p]["lo"], c[p]["hi"]), se_floor) for c in cells.values()]
        out[p] = pooling.dersimonian_laird(ys, ses)
    return dict(pooled=out, verdict=pooling.verdict(out))


def run_set(sc: str, out: Path, n_jobs: int, seed: int = 0) -> dict:
    t0 = time.time()
    inp = pipeline.load_inputs(sc)
    inp.cards = castability.with_cost_groups(inp.cards, sc)
    # Match pipeline's extra_draft choice stored in model/report.json (read-only)
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

    games_raw = inp.games
    gT_raw = games_raw.filter(pl.col("fmt") == "bo3")
    split = validate._holdout_split(gT_raw)
    if split is None:
        return dict(set=sc, status="skipped", reason=f"too few bo3 events/games ({gT_raw.height} games)")
    hold_ids, t_cut, n_ev, n_hold = split

    # Read cached RefFit sigmas strictly read-only from app home if available (fmt="bo3" has seed=0 in pipeline: FMT_ORDER=["bo3","bo1"])
    opts = cardvalue.ChainOpts()
    vc_ro = pipeline.val_cache(sc, games_raw, inp.cards)
    core_key = vc_ro.key("val", "bo3_core", fmt="bo3", seed=int(seed), n_rep=validate.N_REP,
                         space=vcache.space_digest(space), opts=opts.key())
    core_hit = vc_ro.load("val", "bo3_core", core_key)
    ref_fits = {n: validate.RefFit.from_dict(d) for n, d in core_hit[0]["ref"].items()} if core_hit is not None else None

    g_tr_raw = gT_raw.filter(~pl.col("draft_id").is_in(hold_ids))
    g_ho_raw = gT_raw.filter(pl.col("draft_id").is_in(hold_ids))
    stages = validate.chain_stages("bo3", games_raw, g_tr_raw, t_cut, space)
    prior_ok = set(stages) - {"bo3"}

    tmp = out / "tmp" / sc
    (tmp / "arm1").mkdir(parents=True, exist_ok=True)
    (tmp / "arm2").mkdir(parents=True, exist_ok=True)
    coefs = fit_chain_fast(stages, ref_fits, space.n_card, space.n_feat, prior_ok, str(tmp), n_jobs, seed, opts)

    # Sanity check against cached pM on whole bo3 hold-out if hit
    cache_max_diff = None
    if core_hit is not None:
        prob_ho_full, _, _ = design.make_problem(g_ho_raw, stages["bo3"].enc, space, clamp_time=True)
        pM_check = expit(prob_ho_full.eta(coefs["bo3"]))
        cache_max_diff = float(np.max(np.abs(pM_check - np.asarray(core_hit[1]["pM"]))))

    # Attach prior_opp_colors across all Bo3 events
    games = attach_prior_opp(games_raw)
    gT = games.filter(pl.col("fmt") == "bo3")
    g_tr = gT.filter(~pl.col("draft_id").is_in(hold_ids))
    g_ho = gT.filter(pl.col("draft_id").is_in(hold_ids))

    # Post-board slices (game_number >= 2 and non-empty prior_opp_colors)
    def post_slice(df: pl.DataFrame) -> pl.DataFrame:
        return df.filter((pl.col("game_number") >= 2) & (pl.col("prior_opp_colors") != ""))

    pb_ho = post_slice(g_ho)
    if pb_ho.height < 100 or pb_ho["draft_id"].n_unique() < 10:
        return dict(set=sc, status="skipped", reason=f"too few post-board hold-out games ({pb_ho.height})")

    pb_tr_s3 = post_slice(g_tr)
    pb_tr_td = pl.DataFrame()
    if "draft" in stages:
        td_ids = set(stages["draft"].games["draft_id"].unique().to_list())
        pb_tr_td = post_slice(games.filter((pl.col("event") == "TradDraft") & pl.col("draft_id").is_in(td_ids)))

    # Blind offsets (opp_colors blanked to "") so zero post-hoc outcome leakage enters B_pre
    def blind_eta_and_dc(df: pl.DataFrame, st: design.StageData, coef: np.ndarray, clamp: bool) -> tuple[np.ndarray, sp.csr_matrix]:
        df_blind = df.with_columns(pl.lit("").alias("opp_colors"))
        prob_b, _, _ = design.make_problem(df_blind, st.enc, space, clamp_time=clamp)
        eta_b = prob_b.eta(coef)
        Dc = prob_b.D[:, :space.n_card].tocsr()[prob_b.idx]
        return eta_b, Dc

    tr_parts_raw: list[tuple[str, pl.DataFrame, np.ndarray, sp.csr_matrix]] = []
    if pb_tr_td.height > 0:
        e_td, dc_td = blind_eta_and_dc(pb_tr_td, stages["draft"], coefs["draft"], False)
        tr_parts_raw.append(("td", pb_tr_td, e_td, dc_td))
    if pb_tr_s3.height > 0:
        e_s3, dc_s3 = blind_eta_and_dc(pb_tr_s3, stages["bo3"], coefs["bo3"], False)
        tr_parts_raw.append(("bo3", pb_tr_s3, e_s3, dc_s3))

    eta_ho_blind, dc_ho = blind_eta_and_dc(pb_ho, stages["bo3"], coefs["bo3"], True)

    # Fit leak-free pre-game head B_pre on training post-board games
    H_tr = sp.vstack([pregame_head_matrix(df) for _, df, _, _ in tr_parts_raw], format="csr")
    y_tr_pb = np.concatenate([df["won"].cast(pl.Float64).to_numpy() for _, df, _, _ in tr_parts_raw])
    off_tr_pb = np.concatenate([e for _, _, e, _ in tr_parts_raw])
    head_prob = glm.Problem(y=y_tr_pb, C=H_tr, offset=off_tr_pb)
    head_coef = glm.fit(head_prob, np.zeros(len(HEAD_SIGMA)), HEAD_SIGMA, maxiter=300).coef

    H_ho = pregame_head_matrix(pb_ho)
    eta_B_pre_ho = eta_ho_blind + np.asarray(H_ho @ head_coef).ravel()
    p_B_pre = expit(eta_B_pre_ho)
    p_blind = expit(eta_ho_blind)

    # Hold-out event clusters and Half A / Half B masks (inherited from the full bo3 hold-out split order)
    ev_order = design.event_starts(g_ho)["draft_id"].to_list()
    first_half_ids = set(ev_order[:(len(ev_order) + 1) // 2])
    uniq_pb_ids, groups_pb = np.unique(pb_ho["draft_id"].to_numpy(), return_inverse=True)
    is_first_pb = np.array([u in first_half_ids for u in uniq_pb_ids], dtype=bool)
    y_ho = pb_ho["won"].cast(pl.Float64).to_numpy()

    head_vs_blind = paired_gain_metrics(y_ho, p_blind, p_B_pre, groups_pb, is_first_pb, seed=seed + 11)

    # ---- Primary Arm: MO_pre_bo3 (fitted on Bo3 post-board training games with prior_opp_colors, offset = eta_B_pre)
    pb_pool_parts = []
    pos = 0
    for tag, df, e_b, dc in tr_parts_raw:
        h_part = pregame_head_matrix(df)
        off_part = e_b + np.asarray(h_part @ head_coef).ravel()
        pb_pool_parts.append((tag, df, dc, off_part, "prior_opp_colors"))
        pos += df.height

    best_m1, best_p1, best_s1, cvs1 = None, None, None, {}
    for mode in om.MODES:
        pool1 = build_pooled_problem(pb_pool_parts, space.n_card, mode)
        sg1, info1 = om.cv_sigma(pool1, str(tmp / "arm1"), n_jobs, seed=seed + 41, label=f"{sc} pre_bo3 ({mode})")
        cvs1[mode] = info1
        if best_m1 is None or info1["loss"][info1["best"]] < cvs1[best_m1]["loss"][cvs1[best_m1]["best"]]:
            best_m1, best_p1, best_s1 = mode, pool1, sg1
    delta1 = om.fit_deltas(best_p1, best_s1)
    O_ho_1 = om.opp_matrix(pb_ho["prior_opp_colors"].to_numpy(), best_m1)
    eta_arm1 = eta_B_pre_ho + om.predict_centred(dc_ho, O_ho_1, delta1, best_p1.mass)
    arm1_met = paired_gain_metrics(y_ho, p_B_pre, expit(eta_arm1), groups_pb, is_first_pb, seed=seed + 19)
    arm1_met.update(mode=best_m1, sigma=float(best_s1), edge_low=bool(cvs1[best_m1]["best"] == 0),
                    n_train_games=int(best_p1.n_games), n_train_events=int(best_p1.n_groups), cv=cvs1)

    # ---- Secondary Arm: MO_pre_alltrain (fitted on all training stages, evaluated strictly with prior_opp_colors on eta_B_pre_ho)
    all_parts = []
    for sname, st in stages.items():
        df_s = attach_prior_opp(st.games)
        use_col = pl.when((pl.col("game_number") >= 2) & (pl.col("prior_opp_colors") != "")).then(pl.col("prior_opp_colors")).otherwise(pl.col("opp_colors").fill_null(""))
        df_s = df_s.with_columns(use_col.alias("_train_opp"))
        Dc_s = st.prob.D[:, :space.n_card].tocsr()[st.prob.idx]
        off_s = st.prob.eta(coefs[sname])
        all_parts.append((sname, df_s, Dc_s, off_s, "_train_opp"))

    best_m2, best_p2, best_s2, cvs2 = None, None, None, {}
    for mode in om.MODES:
        pool2 = build_pooled_problem(all_parts, space.n_card, mode)
        sg2, info2 = om.cv_sigma(pool2, str(tmp / "arm2"), n_jobs, seed=seed + 53, label=f"{sc} pre_all ({mode})")
        cvs2[mode] = info2
        if best_m2 is None or info2["loss"][info2["best"]] < cvs2[best_m2]["loss"][cvs2[best_m2]["best"]]:
            best_m2, best_p2, best_s2 = mode, pool2, sg2
    delta2 = om.fit_deltas(best_p2, best_s2)
    O_ho_2 = om.opp_matrix(pb_ho["prior_opp_colors"].to_numpy(), best_m2)
    eta_arm2 = eta_B_pre_ho + om.predict_centred(dc_ho, O_ho_2, delta2, best_p2.mass)
    arm2_met = paired_gain_metrics(y_ho, p_B_pre, expit(eta_arm2), groups_pb, is_first_pb, seed=seed + 23)
    arm2_met.update(mode=best_m2, sigma=float(best_s2), edge_low=bool(cvs2[best_m2]["best"] == 0),
                    n_train_games=int(best_p2.n_games), n_train_events=int(best_p2.n_groups), cv=cvs2)

    elapsed = round(time.time() - t0, 1)
    print(f"[{sc}] {elapsed:5.1f}s | ho_pb={pb_ho.height:4d}g ({len(uniq_pb_ids):3d}ev) | "
          f"B_pre-blind={head_vs_blind['gain']*1e4:+6.2f} | "
          f"MO_pre_bo3={arm1_met['gain']*1e4:+6.2f} [{arm1_met['lo']*1e4:+6.2f},{arm1_met['hi']*1e4:+6.2f}] ({best_m1}, s={best_s1:.4f}) | "
          f"MO_pre_all={arm2_met['gain']*1e4:+6.2f} [{arm2_met['lo']*1e4:+6.2f},{arm2_met['hi']*1e4:+6.2f}] ({best_m2}, s={best_s2:.4f})",
          flush=True)
    return dict(
        set=sc, status="ok", seconds=elapsed, cache_hit=core_hit is not None, cache_max_diff=cache_max_diff,
        n_bo3_games=int(gT.height), n_holdout_games_all=int(g_ho.height),
        n_pb_train_td=int(pb_tr_td.height), n_pb_train_bo3=int(pb_tr_s3.height),
        n_pb_holdout_games=int(pb_ho.height), n_pb_holdout_events=int(len(uniq_pb_ids)),
        head_vs_blind=head_vs_blind,
        MO_pre_bo3=arm1_met,
        MO_pre_alltrain=arm2_met,
    )


def summarize_arm(per_set: dict[str, dict], arm_key: str) -> dict:
    cells = {}
    excluded = []
    for sc, r in per_set.items():
        if r.get("status") != "ok":
            excluded.append(dict(set=sc, reason=r.get("reason", "skipped")))
            continue
        m = r[arm_key]
        cell = pooling.make_cell(m, m["halves"])
        if cell is None:
            excluded.append(dict(set=sc, reason="zero-width or non-finite CI"))
        else:
            cells[sc] = cell
    if not cells:
        return dict(n_sets=0, excluded=excluded)
    raw_pool = pooling.pool_parts(list(cells.values()))
    raw_verdict = pooling.verdict(raw_pool)
    floored = pool_with_floor(cells, SE_FLOOR)
    passes = bool(raw_verdict["better"] and not raw_verdict["worse"]
                  and floored["verdict"]["better"] and not floored["verdict"]["worse"])
    return dict(
        n_sets=len(cells), sets=sorted(cells), excluded=excluded,
        raw=dict(pooled=raw_pool, verdict=raw_verdict),
        se_floored=floored,
        passes_rule=passes,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", type=Path, nargs="?", default=Path(tempfile.gettempdir()) / "sealed_lab_opp_pregame")
    ap.add_argument("--sets", nargs="+", default=ALL_SETS)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    a = ap.parse_args(argv)
    out = a.out.resolve()
    check_out_not_home(out)
    out.mkdir(parents=True, exist_ok=True)

    per_set: dict[str, dict] = {}
    t_all = time.time()
    with glm.shared_pool(a.jobs):
        for sc in a.sets:
            per_set[sc] = run_set(sc.upper(), out, a.jobs)
            cardvalue.cleanup(str(out / "tmp" / sc.upper()))

    summary = dict(
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        total_seconds=round(time.time() - t_all, 1),
        se_floor=SE_FLOOR,
        per_set=per_set,
        pooled=dict(
            head_vs_blind=summarize_arm(per_set, "head_vs_blind"),
            MO_pre_bo3=summarize_arm(per_set, "MO_pre_bo3"),
            MO_pre_alltrain=summarize_arm(per_set, "MO_pre_alltrain"),
        ),
    )
    out_json = out / "opp_pregame_results.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {out_json} in {summary['total_seconds']:.1f}s")
    for k, sm in summary["pooled"].items():
        if sm.get("n_sets"):
            rw = sm["raw"]["pooled"]["whole"]
            ra = sm["raw"]["pooled"]["A"]
            rb = sm["raw"]["pooled"]["B"]
            rv = sm["raw"]["verdict"]
            fv = sm["se_floored"]["verdict"]
            print(f"  {k:16s} ({sm['n_sets']} sets): whole={rw['mean']*1e4:+6.2f} [{rw['lo']*1e4:+6.2f},{rw['hi']*1e4:+6.2f}] | "
                  f"A={ra['mean']*1e4:+6.2f} [{ra['lo']*1e4:+6.2f},{ra['hi']*1e4:+6.2f}] | "
                  f"B={rb['mean']*1e4:+6.2f} (lo90={rv['B_lo90']*1e4:+6.2f}) | raw={rv['label']} | floored={fv['label']} | PASS={sm['passes_rule']}")
    return 0


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    sys.exit(main())
