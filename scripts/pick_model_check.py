"""Pre-registered "provisional model from 17Lands pick files" test on Sealed hold-outs (docs/pick_model/RULE.md).

For each of FRA HOB MSH SOS TMT: the PremierDraft pick file's first 12 days are turned into a "draft" stage (decks rebuilt from
`pick_maindeck_rate >= 0.5`, records from `event_match_wins/losses`), its card values (times the mean lambda of the other 4 sets) and the
other 4 sets' non-card terms predict the set's Sealed hold-out without any Sealed game. Arms: Pk (deciding), Z (zero card values),
G (draft stage from real Draft game files of the same 12 days), M (the plain chain on the set's own Sealed training games).

Reads the app home and the raw files strictly read-only (SEALED_LAB_HOME must point at a scratch copy); writes only to OUT.

    python scripts/pick_model_check.py OUT [--sets ...] [--jobs N] [--selfcheck] [--builder] [--builder-pools N]
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import sys
import tempfile
import time
from dataclasses import dataclass, replace
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp
from scipy.special import expit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from sealed_lab import config  # noqa: E402
from sealed_lab import provisional as prov  # noqa: E402
from sealed_lab.model import cardvalue, design, glm, manasim, pipeline, pooling, structure, validate  # noqa: E402
import cold_start_check as csc  # noqa: E402  (find_pick_files, paired_gain_metrics)
import plat_draft_check as pdc  # noqa: E402  (prepare_set, fit_chain, snapshots)
import play_prior_check as ppc  # noqa: E402  (summarize, mean_resid_ci, top_movers, home safety checks)

DEFAULT_SETS = ["FRA", "HOB", "MSH", "SOS", "TMT"]
WINDOW_DAYS = 12
MAIN_RATE = prov.MAIN_RATE          # a pick is in the main deck when pick_maindeck_rate >= 0.5
DECK_SIZE = prov.DECK_SIZE
MIN_NONBASIC, MAX_NONBASIC = prov.MIN_NONBASIC, prov.MAX_NONBASIC
EVENT = prov.EVENT
PICK_COLS = prov.PICK_COLS
BASIC_NAMES = prov.BASIC_NAMES
DROP_REASONS = prov.DROP_REASONS
MIN_GAIN = ppc.MIN_GAIN
SE_FLOOR = ppc.SE_FLOOR
N_REP = ppc.N_REP
PP = 0.55 * 0.45 * 100.0            # logit -> percentage points at a 55% deck
check_home_not_real = ppc.check_home_not_real
check_out_not_home = ppc.check_out_not_home
_json_default = ppc._json_default
# covariates of the final Sealed stage that are transferred between sets (by name); everything else (event dummies, the time trend) is 0
TRANSFER_EXACT = ("intercept", "on_play", "mull1", "mull2p", "omull1", "omull2p", "skill_lin", "skill_na", "game2plus")
TRANSFER_PREFIX = ("skill_bin:", "opp:")


# ------------------------------------------------------------------------------------------------------ pick file, deck rebuild
# The tested method lives in sealed_lab/provisional.py (the app uses the very same functions); only the 12-day window is this script's own.
read_pick_file = prov.read_pick_file


def window_filter(df: pl.DataFrame, days: int = WINDOW_DAYS):
    """Rows whose draft_time is in the first `days` days after the earliest draft_time of `df`. (rows, t0)."""
    return prov.window_filter(df, days)


Ctx = prov.Ctx
make_ctx = prov.make_ctx
color_symbols = prov.color_symbols
split_basics = prov.split_basics
rebuild_deck = prov.rebuild_deck
expand_records = prov.expand_records
_count = prov._count
Rebuilt = prov.Rebuilt
runs_to_games = prov.runs_to_games
phi_for_decks = prov.phi_for_decks


def build_runs(picks: pl.DataFrame, ctx: Ctx, days: int = WINDOW_DAYS) -> Rebuilt:
    """The rebuilt runs of a pick frame (read_pick_file columns) over the first `days` days after its earliest draft_time."""
    return prov.build_runs(picks, ctx, days)


def extend_space(space: design.DeckSpace, decks: list[dict[int, int]], phi: np.ndarray) -> tuple[design.DeckSpace, np.ndarray]:
    """`space` with the rebuilt decks appended as extra rows. Keys are above every real deck_key (no collision); the standardisation
    (mu, sd) stays that of the set's own space, so the rebuilt decks' features are standardised the way the set's stages standardise them.
    Returns (space, deck_key of every rebuilt deck)."""
    base = int(space.keys.max()) + 1
    keys_new = base + np.arange(len(decks), dtype=np.int64)
    rows, cols, vals = [], [], []
    for i, d in enumerate(decks):
        for c, k in d.items():
            j = int(space.col_of_card[c])
            if j >= 0:
                rows.append(i)
                cols.append(j)
                vals.append(float(k))
    Dc = sp.csr_matrix((vals, (rows, cols)), shape=(len(decks), space.n_card))
    Z = (phi - space.mu) / space.sd
    D_new = sp.hstack([Dc, sp.csr_matrix(Z)], format="csr")
    ext = replace(space, keys=np.concatenate([space.keys, keys_new]), phi=np.vstack([space.phi, phi]),
                  D=sp.vstack([space.D, D_new], format="csr"), w=None)
    return ext, keys_new


# ------------------------------------------------------------------------------------------------------ the plain chain M (cached)
def m_cache_paths(out: Path, sc: str) -> tuple[Path, Path]:
    return out / f"M_{sc}.json", out / f"M_{sc}.npz"


def fit_m(sc: str, out: Path, n_jobs: int) -> dict:
    """The plain chain of `sc` on its training events, per format (plat_draft_check's M); what the transfer needs is cached in OUT."""
    jp, npz = m_cache_paths(out, sc)
    if jp.exists() and npz.exists():
        return json.loads(jp.read_text(encoding="utf-8"))
    t0 = time.time()
    inp, space, formats, _ed = pdc.prepare_set(sc, n_jobs)
    games = inp.games
    opts = cardvalue.ChainOpts()
    res: dict = dict(set=sc, formats={})
    arrs: dict[str, np.ndarray] = {}
    for i, fmt in enumerate(formats):
        gT = games.filter(pl.col("fmt") == fmt)
        split = validate._holdout_split(gT)
        if split is None:
            res["formats"][fmt] = dict(status="skipped", reason=f"too few {fmt} events/games")
            continue
        hold_ids, t_cut, _n, _h = split
        g_tr = gT.filter(~pl.col("draft_id").is_in(hold_ids))
        tmp = out / "tmp" / f"M_{sc}_{fmt}"
        st = validate.chain_stages(fmt, games, g_tr, t_cut, space)
        fits = pdc.fit_chain(st, fmt, space, tmp, n_jobs, i + 100, opts)
        f = fits[fmt]
        lam = fits["bo1"].lam if fmt == "bo1" and "bo1" in fits else None
        res["formats"][fmt] = dict(status="ok", names=list(st[fmt].enc.names), k=int(f.k), n_card=int(space.n_card), n_feat=int(space.n_feat),
                                   lam_bo1=None if lam is None else float(lam), sigma_final=float(f.sigma),
                                   sigma_draft=float(fits["draft"].sigma) if "draft" in fits else None,
                                   stages=list(fits), n_draft_games=int(st["draft"].n_games) if "draft" in st else 0,
                                   n_train_games=int(g_tr.height))
        arrs[f"coef_{fmt}"] = np.asarray(f.coef, dtype=float)
        arrs[f"beta_{fmt}"] = np.asarray(f.beta_total, dtype=float)
        cardvalue.cleanup(str(tmp))
        print(f"  M[{sc} {fmt}] stages={list(fits)} lam_bo1={res['formats'][fmt]['lam_bo1']} sigma_final={f.sigma:.4f}", flush=True)
    res["seconds"] = round(time.time() - t0, 1)
    np.savez(npz, **arrs)
    jp.write_text(json.dumps(res, indent=1, default=_json_default), encoding="utf-8")
    return res


def load_m(out: Path, sc: str) -> tuple[dict, dict[str, np.ndarray]]:
    jp, npz = m_cache_paths(out, sc)
    with np.load(npz, allow_pickle=False) as z:
        return json.loads(jp.read_text(encoding="utf-8")), {k: z[k] for k in z.files}


# ------------------------------------------------------------------------------------------------------ the transfer
def is_transferred(name: str) -> bool:
    return name in TRANSFER_EXACT or name.startswith(TRANSFER_PREFIX)


def transfer_terms(sc: str, fmt: str, names: list[str], ms: dict[str, tuple[dict, dict[str, np.ndarray]]], sets: list[str]) -> dict:
    """The transferred non-card terms of `sc`'s final `fmt` stage: the mean over the OTHER sets of `sets` of their M fits' final-stage
    coefficients, in the model's standardised units (per SD of the deck features, each set standardised by its own deck space).

    covariate part (length len(names), the covariate order of `sc`'s own final-stage encoder): by NAME, the mean over the other sets of
    the coefficient of that name (0 where a set has no such covariate, e.g. a skill bin it did not have); only `is_transferred` names
    (intercept, on_play, mulligans, skill, game2plus, opponent-pair levels); event dummies and the time trend are 0.
    structure part (n_feat): the mean of the other sets' structure coefficients ("theta": the deck features of structure.FEATURE_NAMES,
    the same 19 columns in every set). lambda-bar: the mean of the other sets' bo1-stage lambda toward their Draft stage."""
    others = [o for o in sets if o != sc]
    cov = np.zeros(len(names))
    thetas, lams = [], {}
    for o in others:
        meta, arr = ms[o]
        fm = meta["formats"].get(fmt)
        b1 = meta["formats"].get("bo1")
        if not fm or fm.get("status") != "ok" or not b1 or b1.get("status") != "ok" or b1.get("lam_bo1") is None:
            raise RuntimeError(f"cannot transfer from {o}: no {fmt} fit or no bo1 lambda ({fm}, {b1})")
        coef = arr[f"coef_{fmt}"]
        k, nc, nf = fm["k"], fm["n_card"], fm["n_feat"]
        idx = {nm: j for j, nm in enumerate(fm["names"])}
        for j, nm in enumerate(names):
            if is_transferred(nm) and nm in idx:
                cov[j] += coef[idx[nm]] / len(others)
        thetas.append(coef[k + nc:k + nc + nf])
        lams[o] = float(b1["lam_bo1"])
    nfs = {len(t) for t in thetas}
    if len(nfs) != 1:
        raise RuntimeError(f"structure feature vectors differ in length between sets: {nfs}")
    return dict(cov=cov, theta=np.mean(thetas, axis=0), lam_bar=float(np.mean(list(lams.values()))), lam_by_set=lams, others=others)


def predict_arm(prob, cov: np.ndarray, cards: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """P(win) of the hold-out games: covariates `cov`, card values `cards`, structure terms `theta` (coef = [cov | cards | theta])."""
    return expit(prob.eta(np.concatenate([cov, cards, theta])))


def loss_of(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


# ------------------------------------------------------------------------------------------------------ one set
def pick_file_of(sc: str) -> Path | None:
    for p in csc.find_pick_files(sc):
        if f".{EVENT}." in p.name:
            return p
    return None


def real_draft_window(games: pl.DataFrame, t0, t1) -> pl.DataFrame:
    """Real Draft game rows (PremierDraft) of events whose draft_time is in [t0, t1)."""
    return games.filter((pl.col("fmt") == "draft") & (pl.col("event") == EVENT) & (pl.col("draft_time") >= t0) & (pl.col("draft_time") < t1))


draft_chain = prov.draft_chain


def corr(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is not None:
        a, b = a[mask], b[mask]
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def run_set(sc: str, out: Path, n_jobs: int, sets: list[str], ms: dict, build_pools: int | None = None) -> dict:
    t0 = time.time()
    inp, space, formats, ed_applied = pdc.prepare_set(sc, n_jobs)
    games = inp.games
    pf = pick_file_of(sc)
    if pf is None:
        raise FileNotFoundError(f"{sc}: no PremierDraft pick file found")
    csort = inp.cards.sort("card_id")
    name_of_id = dict(zip(csort["card_id"].to_list(), csort["name"].to_list()))
    names_by_col = [name_of_id[int(i)] for i in space.nb_ids]

    # ---- the rebuilt runs and the Pk draft stage
    picks = read_pick_file(pf)
    ctx = make_ctx(inp.cards)
    rb = build_runs(picks, ctx)
    print(f"[{sc}] pick file {pf.name}: runs in window {rb.info['runs_in_window']}, rebuilt {rb.info['runs_rebuilt']}, dropped {rb.info['dropped']}, "
          f"games {rb.info['games']}, decks {rb.info['unique_decks']}, unmatched picks {rb.info['n_unmatched_main_picks']}", flush=True)
    phi = phi_for_decks(rb.decks, inp.cards, n_jobs, out / f"phi_rebuilt_{sc}.npz")
    space_pk, keys_new = extend_space(space, rb.decks, phi)
    g_pk = runs_to_games(rb, keys_new)
    st_pk = design.build_stage("draft", g_pk, space_pk)
    tp = time.time()
    fit_pk = draft_chain(st_pk, space, out / "tmp" / f"{sc}_pk", n_jobs, 7)
    beta_pk = fit_pk.beta_total
    print(f"[{sc}] Pk draft stage: {st_pk.n_games} games, {st_pk.n_groups} events, sigma={fit_pk.sigma:.4f}, {time.time() - tp:.0f}s", flush=True)

    # ---- G: the real Draft stage over the same window
    t_start = rb.info["window_start"]
    t_lo = picks["draft_time"].min()
    t_hi = t_lo + timedelta(days=WINDOW_DAYS)
    g_real = real_draft_window(games, t_lo, t_hi)
    if g_real.height >= validate.MIN_PRIOR_GAMES:
        st_g = design.build_stage("draft", g_real, space)
        fit_g = draft_chain(st_g, space, out / "tmp" / f"{sc}_g", n_jobs, 8)
        beta_g = fit_g.beta_total
        g_info = dict(games=int(st_g.n_games), events=int(st_g.n_groups), sigma=float(fit_g.sigma))
    else:
        st_g = fit_g = beta_g = None
        g_info = dict(games=int(g_real.height), events=0, sigma=None, reason="too few real Draft games in the window")
    print(f"[{sc}] G draft stage: {g_info}", flush=True)

    set_info = dict(rebuilt=rb.info, pk_stage=dict(games=int(st_pk.n_games), events=int(st_pk.n_groups), sigma=float(fit_pk.sigma)), g_stage=g_info,
                    pick_file=str(pf), window_start=t_start)

    # ---- hold-out evaluation per format
    fmt_res: dict[str, dict] = {}
    meta_s, arr_s = ms[sc]
    for i, fmt in enumerate(formats):
        seed_f = i
        mf = meta_s["formats"].get(fmt)
        if not mf or mf.get("status") != "ok":
            fmt_res[fmt] = dict(status="skipped", reason=(mf or {}).get("reason", "no M fit"))
            continue
        gT = games.filter(pl.col("fmt") == fmt)
        split = validate._holdout_split(gT)
        hold_ids, t_cut, _n, _h = split
        g_tr = gT.filter(~pl.col("draft_id").is_in(hold_ids))
        g_ho = gT.filter(pl.col("draft_id").is_in(hold_ids))
        enc = design.CovEncoder.fit(g_tr)             # = the encoder of the final stage of M (chain_stages builds it the same way)
        if list(enc.names) != list(mf["names"]):
            raise RuntimeError(f"{sc} {fmt}: encoder names differ from the cached M fit")
        prob_ho, _, _ = design.make_problem(g_ho, enc, space, clamp_time=True)
        y = prob_ho.y
        groups = np.unique(g_ho["draft_id"].to_numpy(), return_inverse=True)[1]
        is_first = validate.half_masks(g_ho)
        tr = transfer_terms(sc, fmt, list(enc.names), ms, sets)
        lam_bar = tr["lam_bar"]
        n_card = space.n_card
        p_m = expit(prob_ho.eta(arr_s[f"coef_{fmt}"]))
        zero = np.zeros(n_card)
        p_z = predict_arm(prob_ho, tr["cov"], zero, tr["theta"])
        p_pk = predict_arm(prob_ho, tr["cov"], lam_bar * beta_pk, tr["theta"])
        p_pk0 = predict_arm(prob_ho, tr["cov"], lam_bar * beta_pk * 0.0, tr["theta"])
        p_g = None if beta_g is None else predict_arm(prob_ho, tr["cov"], lam_bar * beta_g, tr["theta"])
        arms = dict(Pk=csc.paired_gain_metrics(y, p_z, p_pk, groups, is_first, n_rep=N_REP, seed=seed_f))      # loss(Z) - loss(Pk)
        diffs = {"Pk_vs_M": csc.paired_gain_metrics(y, p_pk, p_m, groups, is_first, n_rep=N_REP, seed=seed_f)}   # loss(Pk) - loss(M)
        if p_g is not None:
            diffs["G_vs_Pk"] = csc.paired_gain_metrics(y, p_g, p_pk, groups, is_first, n_rep=N_REP, seed=seed_f)  # loss(G) - loss(Pk)
        calib = {nm: ppc.mean_resid_ci(y, p, groups, N_REP, seed_f + 11) for nm, p in
                 (("Pk", p_pk), ("Z", p_z), ("M", p_m)) + ((("G", p_g),) if p_g is not None else ())}
        b_m = arr_s[f"beta_{fmt}"]
        b_m1 = arr_s["beta_bo1"] if "beta_bo1" in arr_s else None
        seen = st_pk.exposure > 0
        vals_pk = lam_bar * beta_pk
        gaps = ppc.top_movers(names_by_col, b_m, vals_pk)
        cc = dict(pk_vs_M_bo1=None if b_m1 is None else corr(beta_pk, b_m1), pk_vs_M_bo1_seen=None if b_m1 is None else corr(beta_pk, b_m1, seen),
                  pk_vs_G=None if beta_g is None else corr(beta_pk, beta_g), pk_vs_G_seen=None if beta_g is None else corr(beta_pk, beta_g, seen),
                  n_cards=int(n_card), n_seen=int(seen.sum()))
        fmt_res[fmt] = dict(
            status="ok", n_holdout_games=int(len(y)), n_holdout_events=int(groups.max()) + 1, t_cut=str(t_cut),
            lam_bar=lam_bar, lam_by_set=tr["lam_by_set"], sigma_draft=dict(Pk=float(fit_pk.sigma), G=g_info["sigma"], M=mf.get("sigma_draft")),
            draft_games=dict(Pk=int(st_pk.n_games), G=g_info["games"], M=mf.get("n_draft_games")),
            arms=arms, diffs=diffs, calibration=calib, corr=cc, largest_gaps_from_M=gaps,
            loss=dict(Z=loss_of(y, p_z), Pk=loss_of(y, p_pk), M=loss_of(y, p_m), G=None if p_g is None else loss_of(y, p_g)),
            zeroed_equals_Z=bool(np.array_equal(p_pk0, p_z)))
        a = arms["Pk"]
        print(f"[{sc} {fmt}] ho={len(y):5d} lam_bar={lam_bar:.3f} | Z-Pk={a['gain']*1e4:+6.2f} [{a['lo']*1e4:+6.2f},{a['hi']*1e4:+6.2f}] "
              f"A={a['halves']['A']['gain']*1e4:+6.2f} B={a['halves']['B']['gain']*1e4:+6.2f} | "
              f"calib Pk={calib['Pk']['mean']:+.4f} Z={calib['Z']['mean']:+.4f} M={calib['M']['mean']:+.4f} | corr M(bo1)={cc['pk_vs_M_bo1']:.3f} "
              f"G={cc['pk_vs_G']} | {time.time() - t0:.0f}s", flush=True)

        if build_pools:
            try:
                fmt_res[fmt]["builder"] = builder_compare(sc, fmt, inp, space, enc, g_tr, arr_s, tr, vals_pk, n_jobs, build_pools)
            except Exception as e:  # noqa: BLE001 - reported-only part
                import traceback

                fmt_res[fmt]["builder"] = dict(status="failed", error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-1500:])
                print(f"[{sc} {fmt}] builder part failed: {e}", flush=True)
    return dict(set=sc, seconds=round(time.time() - t0, 1), extra_draft_applied=ed_applied, set_info=set_info, formats=fmt_res)


# ------------------------------------------------------------------------------------------------------ builder (reported only)
_WORKER: dict = {}


def _model_from_spec(spec: dict):
    from sealed_lab.model.artifact import SetModel

    meta = dict(spec["meta"])
    draws = {f: dict(beta=v["beta"][None, :], theta=v["theta"][None, :], ref=np.array([v["ref"]]), mull=np.array([v["mull"]]))
             for f, v in spec["fmts"].items()}
    return SetModel(spec["set"], meta, spec["cards"], draws, {f: np.zeros((1, 2)) for f in spec["fmts"]}, extras={})


def _worker_init(specs: dict) -> None:
    _WORKER["specs"] = specs
    _WORKER["models"] = {}


def _build_task(task: dict) -> dict:
    from sealed_lab.deck import builder

    key = task["variant"]
    if key not in _WORKER["models"]:
        _WORKER["models"][key] = _model_from_spec(_WORKER["specs"][key])
    model = _WORKER["models"][key]
    t0 = time.time()
    res = builder.build(model, task["pool"], task["fmt"], n_decks=1, families=True, n_jobs=1)
    top = res.decks[0]
    deck = {int(c.card_id): int(c.count) for c in top.cards if c.card_id is not None}
    return dict(variant=key, draft_id=task["draft_id"], deck=deck, colors=top.colors, seconds=time.time() - t0)


def _score_task(task: dict) -> dict:
    key = task["variant"]
    if key not in _WORKER["models"]:
        _WORKER["models"][key] = _model_from_spec(_WORKER["specs"][key])
    m = _WORKER["models"][key]
    return dict(variant=key, scores=[(k, float(m.logit({int(a): int(b) for a, b in d.items()}, task["fmt"]))) for k, d in task["items"]])


def make_spec(sc: str, fmt: str, cards_model: pl.DataFrame, meta_model: dict, space: design.DeckSpace, cov: np.ndarray, names: list[str],
              ref_row: np.ndarray, card_vals: np.ndarray, theta: np.ndarray) -> dict:
    """In-memory SetModel ingredients of one plain model (cov = covariate coefficients, card_vals / theta in the set's deck space)."""
    meta = dict(meta_model)
    meta.update(formats=[fmt], feature_names=list(structure.FEATURE_NAMES), phi_mean=space.mu.tolist(), phi_sd=space.sd.tolist(),
                manasim=dict(version=1), castability=dict(applied=False), matchups={}, playdraw={})
    beta = np.zeros(space.n_cards_total)
    beta[space.nb_ids] = card_vals
    ix = {nm: j for j, nm in enumerate(names)}
    return dict(set=sc, meta=meta, cards=cards_model,
                fmts={fmt: dict(beta=beta, theta=np.asarray(theta), ref=float(cov @ ref_row), mull=(float(cov[ix["mull1"]]), float(cov[ix["mull2p"]])))})


def builder_compare(sc: str, fmt: str, inp, space, enc, g_tr, arr_s, tr, vals_pk, n_jobs: int, n_pools: int) -> dict:
    """Decks built with Pk and with M (both as plain-chain SetModels made in memory) on the held-out pools of search_check.json."""
    import concurrent.futures as cf
    import multiprocessing

    mdir = config.model_dir(sc)
    meta_model = json.loads((mdir / "meta.json").read_text(encoding="utf-8"))
    cards_model = pl.read_parquet(mdir / "cards.parquet")
    prob_tr, _, _ = design.make_problem(g_tr, enc, space)
    ref_row = enc.set_reference(prob_tr.C)
    names = list(enc.names)
    coef_m = arr_s[f"coef_{fmt}"]
    k, nc, nf = len(names), space.n_card, space.n_feat
    spec_m = make_spec(sc, fmt, cards_model, meta_model, space, coef_m[:k], names, ref_row, coef_m[k:k + nc], coef_m[k + nc:k + nc + nf])
    spec_p = make_spec(sc, fmt, cards_model, meta_model, space, tr["cov"], names, ref_row, vals_pk, tr["theta"])
    specs = {"Pk": spec_p, "M": spec_m}
    entries = [e for e in json.loads((ROOT_DIR / "docs" / "builder_scoreboard" / "search_check.json").read_text(encoding="utf-8"))["pools"]
               if e["set"] == sc and e["fmt"] == fmt][:n_pools]
    want = {e["draft_id"] for e in entries}
    pools = pl.read_parquet(config.corpus_dir(sc) / "pools.parquet").filter(pl.col("draft_id").is_in(list(want)))
    pool_of: dict[str, dict[int, int]] = {}
    for did, c, n in zip(pools["draft_id"].to_list(), pools["card_id"].to_list(), pools["n"].to_list()):
        pool_of.setdefault(did, {})[int(c)] = int(n)
    ids = [e["draft_id"] for e in entries if e["draft_id"] in pool_of]
    ctx = multiprocessing.get_context("spawn")
    t0 = time.time()
    with cf.ProcessPoolExecutor(max_workers=max(1, min(n_jobs, 2 * len(ids))), mp_context=ctx, initializer=_worker_init, initargs=(specs,)) as ex:
        built = list(ex.map(_build_task, [dict(variant=v, draft_id=d, fmt=fmt, pool=pool_of[d]) for d in ids for v in ("Pk", "M")], chunksize=1))
        by = {(r["variant"], r["draft_id"]): r for r in built}
        scored = list(ex.map(_score_task, [dict(variant="M", fmt=fmt, items=[(d, by[("Pk", d)]["deck"]) for d in ids]),
                                           dict(variant="M", fmt=fmt, items=[(d, by[("M", d)]["deck"]) for d in ids])]))
    s_pk, s_m = dict(scored[0]["scores"]), dict(scored[1]["scores"])
    csa = manasim.card_arrays(inp.cards)
    rows = []
    for d in ids:
        dp, dm = by[("Pk", d)]["deck"], by[("M", d)]["deck"]
        nbp = {c: n for c, n in dp.items() if not csa.is_basic[c]}
        nbm = {c: n for c, n in dm.items() if not csa.is_basic[c]}
        changed = sum(max(0, n - nbp.get(c, 0)) for c, n in nbm.items())
        rows.append(dict(draft_id=d, identical=dp == dm, same_nonbasic=nbp == nbm, same_colors=by[("Pk", d)]["colors"] == by[("M", d)]["colors"],
                         cards_changed=int(changed), m_score_pk_deck_minus_own=s_pk[d] - s_m[d]))
    n = len(rows)
    diff = np.array([r["m_score_pk_deck_minus_own"] for r in rows]) if n else np.zeros(0)
    return dict(status="ok", n_pools=n, identical_share=float(np.mean([r["identical"] for r in rows])) if n else None,
                same_nonbasic_share=float(np.mean([r["same_nonbasic"] for r in rows])) if n else None,
                same_colors_share=float(np.mean([r["same_colors"] for r in rows])) if n else None,
                mean_cards_changed=float(np.mean([r["cards_changed"] for r in rows])) if n else None,
                mean_m_score_gap_logit=float(diff.mean()) if n else None, mean_m_score_gap_pp=float(diff.mean() * PP) if n else None,
                seconds=round(time.time() - t0, 1), rows=rows)


ROOT_DIR = HERE.parent


# ------------------------------------------------------------------------------------------------------ pooling and report
def pooled_pk(per_set: dict[str, dict]) -> dict:
    return ppc.summarize(per_set, lambda fr: (fr.get("arms") or {}).get("Pk"))


def pooled_diff(per_set: dict[str, dict], key: str) -> dict:
    return ppc.summarize(per_set, lambda fr: (fr.get("diffs") or {}).get(key))


def fx(x, nd=2, scale=1e4, sign=True):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    v = x * scale
    return f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"


def ci(m: dict) -> str:
    return f"{fx(m['gain'])} [{fx(m['lo'])}, {fx(m['hi'])}]"


def write_md(summary: dict, path: Path) -> None:
    per_set, pooled = summary["per_set"], summary["pooled"]
    L = ["# Provisional model from pick files: results (pick_model_check)", "",
         "Rule: `docs/pick_model/RULE.md`. Gains per Sealed game, x1e-4, positive = the second-named arm is better: `Pk` row = loss(Z) - loss(Pk). "
         f"Sets: {', '.join(summary['sets'])}. Pooling: DerSimonian-Laird per `docs/pooled_retest/POOLING_RULE.md`; verdicts raw / SE >= {SE_FLOOR:g}.", "",
         f"## Verdict: Pk {'PASSES' if pooled['passes_rule'] else 'FAILS'} (raw and SE-floored)", "",
         "| Format | Sets | Whole (x1e-4) | Half A | Half B (lo90) | Verdict raw / floored |", "|---|---:|---|---|---|---|"]
    for fmt, fv in pooled["formats"].items():
        if not fv.get("n_sets"):
            continue
        rw, ra, rb = (fv["raw"]["pooled"][p] for p in ("whole", "A", "B"))
        rv, flv = fv["raw"]["verdict"], fv["se_floored"]["verdict"]
        L.append(f"| {fmt} | {fv['n_sets']} | {fx(rw['mean'])} [{fx(rw['lo'])}, {fx(rw['hi'])}] | {fx(ra['mean'])} [{fx(ra['lo'])}, {fx(ra['hi'])}] | "
                 f"{fx(rb['mean'])} ({fx(rv['B_lo90'])}) | {rv['label']} / {flv['label']} |")
    L += ["", "- Excluded cells: " + ("; ".join(f"{f}: {fv['excluded']}" for f, fv in pooled["formats"].items() if fv.get("excluded")) or "none"), ""]
    for key, label in (("G_vs_Pk", "loss(G) - loss(Pk)"), ("Pk_vs_M", "loss(Pk) - loss(M)")):
        pd_ = summary["pooled_diffs"][key]
        L += [f"## Pooled {label} (reported)", "", "| Format | Sets | Whole | Half A | Half B |", "|---|---:|---|---|---|"]
        for fmt, fv in pd_["formats"].items():
            if not fv.get("n_sets"):
                continue
            rw, ra, rb = (fv["raw"]["pooled"][p] for p in ("whole", "A", "B"))
            L.append(f"| {fmt} | {fv['n_sets']} | {fx(rw['mean'])} [{fx(rw['lo'])}, {fx(rw['hi'])}] | {fx(ra['mean'])} [{fx(ra['lo'])}, {fx(ra['hi'])}] | "
                     f"{fx(rb['mean'])} [{fx(rb['lo'])}, {fx(rb['hi'])}] |")
        L.append("")
    L += ["## Per set and format", "", "| set | fmt | Z-Pk | G-Pk | Pk-M | calib Pk | calib Z | calib M | lam_bar | sigma draft Pk/G | corr M(bo1) | corr G |",
          "|---|---|---|---|---|---|---|---|---:|---|---:|---:|"]
    for sc, r in per_set.items():
        for fmt, fr in r["formats"].items():
            if fr.get("status") != "ok":
                L.append(f"| {sc} | {fmt} | skipped: {fr.get('reason')} | | | | | | | | | |")
                continue
            d = fr["diffs"]
            sg = fr["sigma_draft"]
            L.append(f"| {sc} | {fmt} | {ci(fr['arms']['Pk'])} | {ci(d['G_vs_Pk']) if 'G_vs_Pk' in d else 'n/a'} | {ci(d['Pk_vs_M'])} | "
                     f"{fr['calibration']['Pk']['mean']:+.4f} | {fr['calibration']['Z']['mean']:+.4f} | {fr['calibration']['M']['mean']:+.4f} | "
                     f"{fr['lam_bar']:.3f} | {sg['Pk']:.3f}/{sg['G'] if sg['G'] is None else round(sg['G'], 3)} | {fr['corr']['pk_vs_M_bo1']:.3f} | "
                     f"{fr['corr']['pk_vs_G'] if fr['corr']['pk_vs_G'] is None else round(fr['corr']['pk_vs_G'], 3)} |")
    L += ["", "## Runs rebuilt and dropped (by first reason), games used", "", "| set | in file | in window | rebuilt | no_games | too_few | too_many | no_main | games | unmatched main picks |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for sc, r in per_set.items():
        i = r["set_info"]["rebuilt"]
        dr = i["dropped"]
        L.append(f"| {sc} | {i['runs_in_file']} | {i['runs_in_window']} | {i['runs_rebuilt']} | {dr['no_games']} | {dr['too_few_nonbasic']} | "
                 f"{dr['too_many_nonbasic']} | {dr['no_main_color']} | {i['games']} | {i['n_unmatched_main_picks']} |")
    L += ["", "## Ten largest gaps from M (lam_bar x Pk card value vs M's final-stage value, logit)", ""]
    for sc, r in per_set.items():
        for fmt, fr in r["formats"].items():
            if fr.get("status") != "ok":
                continue
            L += [f"### {sc} {fmt}", "", "| card | M | Pk | gap pp |", "|---|---:|---:|---:|"]
            for mv in fr["largest_gaps_from_M"]:
                L.append(f"| {mv['name']} | {mv['beta_M']:+.4f} | {mv['beta_arm']:+.4f} | {mv['delta'] * PP:+.2f} |")
            L.append("")
    bl = [(sc, f, fr["builder"]) for sc, r in per_set.items() for f, fr in r["formats"].items() if fr.get("builder")]
    if bl:
        L += ["## Builder (reported): decks built with Pk and with M", "", "| set | fmt | pools | identical | same nonbasic | same colors | cards changed | M score gap (pp) |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
        for sc, f, b in bl:
            if b.get("status") != "ok":
                L.append(f"| {sc} | {f} | failed: {b.get('error')} | | | | | |")
                continue
            L.append(f"| {sc} | {f} | {b['n_pools']} | {b['identical_share']:.2f} | {b['same_nonbasic_share']:.2f} | {b['same_colors_share']:.2f} | "
                     f"{b['mean_cards_changed']:.2f} | {b['mean_m_score_gap_pp']:+.3f} |")
    path.write_text("\n".join(L), encoding="utf-8")


# ------------------------------------------------------------------------------------------------------ selfcheck
def real_decks(inp: pipeline.Inputs, draft_ids: set[str]) -> dict[str, dict[int, int]]:
    """Nonbasic card copies of each run's most-played build (PremierDraft Draft games; ties: lowest build_index) from the corpus."""
    b = (inp.builds.filter((pl.col("fmt") == "draft") & (pl.col("event") == EVENT) & pl.col("draft_id").is_in(list(draft_ids)))
         .sort(["draft_id", "games", "build_index"], descending=[False, True, False]).group_by("draft_id", maintain_order=True).first())
    ca = manasim.card_arrays(inp.cards)
    dc = inp.deck_cards.filter(pl.col("deck_key").is_in(b["deck_key"].implode()))
    by_key: dict[int, dict[int, int]] = {}
    for k, c, n in zip(dc["deck_key"].to_list(), dc["card_id"].to_list(), dc["n"].to_list()):
        if not ca.is_basic[c]:
            by_key.setdefault(k, {})[int(c)] = int(n)
    return {d: by_key.get(k, {}) for d, k in zip(b["draft_id"].to_list(), b["deck_key"].to_list())}


def deck_diff(a: dict[int, int], b: dict[int, int]) -> int:
    """Cards of multiset `a` missing from `b` (sum of positive differences)."""
    return int(sum(max(0, n - b.get(c, 0)) for c, n in a.items()))


def rebuild_vs_real(sc: str, n_jobs: int) -> dict:
    inp, _space, _f, _ed = pdc.prepare_set(sc, n_jobs)
    pf = pick_file_of(sc)
    picks = read_pick_file(pf)
    rb = build_runs(picks, make_ctx(inp.cards))
    real = real_decks(inp, set(rb.nonbasic))
    both = [d for d in rb.nonbasic if d in real and sum(real[d].values()) > 0]
    same = sum(1 for d in both if rb.nonbasic[d] == real[d])
    diffs = np.array([deck_diff(real[d], rb.nonbasic[d]) for d in both])
    diffs2 = np.array([deck_diff(rb.nonbasic[d], real[d]) for d in both])
    ppr = picks.group_by("draft_id").len()["len"].value_counts().sort("len").rows()
    return dict(set=sc, picks_per_run=dict(ppr), rebuilt_runs=len(rb.nonbasic), in_both=len(both), identical=int(same), share_identical=same / max(1, len(both)),
                mean_cards_differing=float(diffs.mean()) if len(both) else None, mean_cards_differing_other_way=float(diffs2.mean()) if len(both) else None,
                share_within_1=float((diffs <= 1).mean()) if len(both) else None, share_within_2=float((diffs <= 2).mean()) if len(both) else None,
                info=rb.info)


def snapshot_all(sets: list[str]) -> dict[str, dict]:
    snap = pdc.snapshot(sets)
    pk = {}
    for sc in sets:
        p = pick_file_of(sc)
        if p is not None and p.exists():
            s = p.stat()
            pk[str(p)] = (s.st_size, s.st_mtime_ns)
    snap["pick_files"] = pk
    return snap


def selfcheck(set_c: str, sets: list[str], n_jobs: int, out: Path) -> int:
    before = snapshot_all(sets)
    print("snapshot before: " + ", ".join(f"{k}={len(v)} files" for k, v in before.items()), flush=True)
    ok = True
    with glm.shared_pool(n_jobs):
        # (a) rebuilt vs real decks, every test set (the first one named is the set the other checks use)
        for sc_a in [set_c] + [x for x in sets if x != set_c]:
            r = rebuild_vs_real(sc_a, n_jobs)
            print(f"(a) {sc_a}: rebuilt runs {r['rebuilt_runs']}, in both files {r['in_both']}, identical nonbasic main deck {r['identical']} "
                  f"({r['share_identical']:.1%}); mean cards in the real deck missing from the rebuilt one {r['mean_cards_differing']:.3f} "
                  f"(other way {r['mean_cards_differing_other_way']:.3f}); within 1 card {r['share_within_1']:.1%}, within 2 {r['share_within_2']:.1%}; "
                  f"picks per run in the file {r['picks_per_run']}", flush=True)
            (out / f"selfcheck_rebuild_{sc_a}.json").write_text(json.dumps(r, indent=1, default=_json_default), encoding="utf-8")
        # (b) Pk with zeroed card values == Z (through the same transfer and prediction path as the run)
        for sc in DEFAULT_SETS:
            fit_m(sc, out, n_jobs)
            cardvalue.cleanup(str(out / "tmp"))
        ms = {sc: load_m(out, sc) for sc in DEFAULT_SETS}
        per = run_set(set_c, out, n_jobs, DEFAULT_SETS, ms)
        flags = {f: fr.get("zeroed_equals_Z") for f, fr in per["formats"].items() if fr.get("status") == "ok"}
        print(f"(b) Pk with card values zeroed == Z exactly: {flags}", flush=True)
        ok = ok and bool(flags) and all(flags.values())
    after = snapshot_all(sets)
    cmp = pdc.compare_snapshots(before, after)
    print("(c) untouched check: " + "; ".join(f"{k}: {v['n_files']} files, {v['n_changed']} changed" for k, v in cmp.items()), flush=True)
    for k, v in cmp.items():
        for p in v["changed"]:
            print(f"    CHANGED {k}: {p}")
    ok_c = all(v["n_changed"] == 0 for v in cmp.values())
    print(f"\nselfcheck (a) reported; (b) {'OK' if ok else 'FAIL'}; (c) {'OK' if ok_c else 'FAIL'}")
    return 0 if (ok and ok_c) else 1


# ------------------------------------------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", type=Path, nargs="?", default=Path(tempfile.gettempdir()) / "sealed_lab_pick_model")
    ap.add_argument("--sets", nargs="+", default=DEFAULT_SETS, help="sets to evaluate (the transfer always uses the 5 test sets)")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--selfcheck", action="store_true", help="implementation checks of RULE.md (rebuilt vs real decks, Pk zeroed == Z, home untouched)")
    ap.add_argument("--selfcheck-set", default="TMT")
    ap.add_argument("--builder", action="store_true", help="also build decks with Pk and M on the held-out pools of search_check.json (reported only)")
    ap.add_argument("--builder-pools", type=int, default=10, help="pools per set and format for --builder (the file has 10)")
    a = ap.parse_args(argv)
    check_home_not_real()
    out = a.out.resolve()
    check_out_not_home(out)
    out.mkdir(parents=True, exist_ok=True)
    if a.selfcheck:
        return selfcheck(a.selfcheck_set.upper(), DEFAULT_SETS, a.jobs, out)

    sets = [s.upper() for s in a.sets]
    before = snapshot_all(DEFAULT_SETS)
    per_set: dict[str, dict] = {}
    t_all = time.time()
    with glm.shared_pool(a.jobs):
        for sc in DEFAULT_SETS:                      # the transfer needs every test set's M
            if not all(p.exists() for p in m_cache_paths(out, sc)):
                print(f"[M {sc}] fitting", flush=True)
            fit_m(sc, out, a.jobs)
            cardvalue.cleanup(str(out / "tmp"))
        ms = {sc: load_m(out, sc) for sc in DEFAULT_SETS}
        for sc in sets:
            done = out / f"per_set_{sc}.json"
            if done.exists():
                per_set[sc] = json.loads(done.read_text(encoding="utf-8"))
                print(f"[{sc}] reused {done.name}", flush=True)
                continue
            per_set[sc] = run_set(sc, out, a.jobs, DEFAULT_SETS, ms, build_pools=a.builder_pools if a.builder else None)
            done.write_text(json.dumps(per_set[sc], indent=1, default=_json_default), encoding="utf-8")
            cardvalue.cleanup(str(out / "tmp"))
    untouched = pdc.compare_snapshots(before, snapshot_all(DEFAULT_SETS))
    pooled = pooled_pk(per_set)
    pdiffs = {k: pooled_diff(per_set, k) for k in ("G_vs_Pk", "Pk_vs_M")}
    summary = dict(created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), total_seconds=round(time.time() - t_all, 1), se_floor=SE_FLOOR,
                   min_gain=MIN_GAIN, n_rep=N_REP, sets=sets, window_days=WINDOW_DAYS, home_untouched=untouched, per_set=per_set, pooled=pooled,
                   pooled_diffs=pdiffs, passes=bool(pooled["passes_rule"]))
    (out / "pick_model_results.json").write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    write_md(summary, out / "pick_model_results.md")
    print(f"\nWrote pick_model_results.json / .md in {summary['total_seconds']:.1f}s")
    for line in ppc.fmt_pool_line("Pk: loss(Z) - loss(Pk)", pooled):
        print(line)
    for key, sm in pdiffs.items():
        for line in ppc.fmt_pool_line(key, sm):
            print(line)
    print("Pk passes:", summary["passes"])
    if any(v["n_changed"] for v in untouched.values()):
        print("WARNING: files changed during the run:", {k: v["changed"] for k, v in untouched.items() if v["n_changed"]})
    return 0


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    sys.exit(main())
