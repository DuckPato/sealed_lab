"""Does adding PickTwo draft data improve the forward hold-out of the deployed card-value model?  (data experiment: no change to the app)

    python scripts/extra_draft_check.py run    --out OUT [--sets OM1 MSH FRA HOB] [--jobs N]     all conditions, one after the other
    python scripts/extra_draft_check.py worker SET COND --out OUT [--jobs N]                      one condition (what `run` launches)
    python scripts/extra_draft_check.py report --out OUT                                         OUT/extra_draft.json + extra_draft.md

Conditions (each in its own scratch home OUT/homes/SET_COND, a copy of the real set folder's corpus / caches / registry; the real app
home is only READ and SEALED_LAB_HOME is always set to the scratch home):
    OM1  base | p2both (+ PickTwoDraft + PickTwoTradDraft game files: creates a draft stage) | p2d (+ PickTwoDraft only)
    MSH  base | p2 (+ PickTwoDraft game file)
    FRA, HOB  base (with PickTwo, as now) | nop2 (PickTwoDraft game + draft-pick files removed)
Only 17lands GAME files change the model's data (draft-pick files are "registered, not used"; replay files only feed the simulator's
keep-rule measurement).  The replay registrations are kept IDENTICAL in every condition (the cached keep rule / simulator inputs are
the same), so only the card-outcome data differs.

Per condition and format the script does what `pipeline.calculate` does for the model that is deployed, using the app's own code:
    run_validation(...)      the plain chain "M" on the time hold-out (default options: time-split sigma + stacking, recency off)
    validate_deployed(...)   the deployed combination "MD" (components as they resolved on the real set: read from the real meta.json)
and keeps the hold-out predictions (validate.cluster_metrics is wrapped to copy them; nothing in sealed_lab/ is edited).  `report` then
scores both conditions on IDENTICAL hold-out games with `validate.cluster_metrics` (paired cluster bootstrap over events) and
`validate.half_gains` (the confirmation halves) and applies `pipeline._verdict` / MIN_GAIN, the app's shipping rule.

LEAK CHECK.  The hold-out is the last 25% of the TARGET format's events (validate._holdout_split).  `validate.chain_stages` keeps a prior
stage's events only if they started before the cut (`start < t_cut`), for every prior stage including the draft stage; run_validation,
validate_deployed and the B1d baseline all go through it.  The worker re-checks it on the stages that were actually fitted
(`max draft event start < t_cut`, recorded as `before_cut`).  Not cut (as in the app, identical logic in both conditions): the deck space
standardisation / synergy-pair selection / simulator caches, which use deck lists and game counts of all events but no outcomes.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from sealed_lab.ingest.datasets import default_data_dir as _default_data_dir  # noqa: E402
DATA_DIR = Path(_default_data_dir())          # env SEALED_LAB_DATA_DIR, else the app's data folder default
PICKTWO = ("PickTwoDraft", "PickTwoTradDraft")
FMTS = ("bo3", "bo1")                      # the app's order (seed = index)
N_REP = 500                                # the app's n_rep for the per-condition fits
N_REP_CMP = 1000                           # bootstrap replicates of the cross-condition comparison
# set -> condition -> (events whose game files are ADDED, remove PickTwo game + draft files)
CONDS: dict[str, dict[str, tuple[tuple[str, ...], bool]]] = {
    "OM1": {"base": ((), False), "p2both": (("PickTwoDraft", "PickTwoTradDraft"), False), "p2d": (("PickTwoDraft",), False)},
    "MSH": {"base": ((), False), "p2": (("PickTwoDraft",), False)},
    "FRA": {"base": ((), False), "nop2": ((), True)},
    "HOB": {"base": ((), False), "nop2": ((), True)},
}
# (set, condition WITH the PickTwo data, condition WITHOUT it): gain = LL(without) - LL(with), positive = PickTwo data helps
PAIRS = [("OM1", "p2both", "base"), ("OM1", "p2d", "base"), ("MSH", "p2", "base"), ("FRA", "base", "nop2"), ("HOB", "base", "nop2")]
COMP_ORDER = ("recency", "attributes", "crossset", "synergy", "archcond", "matchups", "manasim", "castability")


# =====================================================================================================================
# pure parts (tested without any model)
# =====================================================================================================================

def registry_plan(entries: list[dict], set_code: str, add_events: tuple[str, ...], remove_picktwo: bool,
                  data_dir: Path = DATA_DIR) -> tuple[list[str], list[str]]:
    """(fingerprints to remove from the registry, game-file paths to register) for one condition.

    Removal: the PickTwo events' GAME and DRAFT-pick files (replay files stay, see the module doc).  Adding: the game file of each
    event in `add_events` (nothing is added when it is already registered)."""
    rm = [d["fingerprint"] for d in entries
          if remove_picktwo and d.get("kind") in ("game", "draft") and d.get("event") in PICKTWO]
    have = {d.get("event") for d in entries if d.get("kind") == "game"}
    add = [str(data_dir / set_code / f"game_data_public.{set_code}.{ev}.csv.gz") for ev in add_events if ev not in have]
    return rm, add


def weighted_sd(x, w=None) -> float:
    """SD of `x` (weights `w`, plain when None) around its (weighted) mean; nan when empty or all weights zero."""
    x = np.asarray(x, float)
    w = np.ones(len(x)) if w is None else np.asarray(w, float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not ok.any():
        return float("nan")
    x, w = x[ok], w[ok]
    m = float((w * x).sum() / w.sum())
    return float(np.sqrt((w * (x - m) ** 2).sum() / w.sum()))


def align_holdouts(a: dict, b: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Order that puts two conditions' hold-out games in the same sequence.

    a, b: dict(did=draft_id per game (str array), y=outcome). Returns (ia, ib, groups): index arrays into a and b (stable sort by
    event id, so games keep their order within an event) and the event code (0..G-1) of every aligned game.  Raises if the two
    hold-outs are not the same games (event ids or outcomes differ)."""
    ia = np.argsort(np.asarray(a["did"]).astype(str), kind="stable")
    ib = np.argsort(np.asarray(b["did"]).astype(str), kind="stable")
    da, db = np.asarray(a["did"]).astype(str)[ia], np.asarray(b["did"]).astype(str)[ib]
    if len(da) != len(db) or not np.array_equal(da, db):
        raise ValueError("the two conditions do not have the same hold-out events / games")
    if not np.array_equal(np.asarray(a["y"])[ia], np.asarray(b["y"])[ib]):
        raise ValueError("the two conditions' hold-out outcomes differ")
    groups = np.unique(da, return_inverse=True)[1]
    return ia, ib, groups


def paired_gain(y, p_without, p_with, groups, is_first, n_rep: int, seed: int) -> dict:
    """Log-loss gain of the model WITH the extra data over the one WITHOUT it (positive = the extra data helps), 95% cluster-bootstrap
    CI over events (`validate.cluster_metrics`, base-minus-new as in the app's tests), and the confirmation halves
    (`validate.half_gains`: A = earlier events, B = later ones)."""
    from sealed_lab.model import validate

    met = validate.cluster_metrics(np.asarray(y), {"M": np.asarray(p_with), "B": np.asarray(p_without)}, np.asarray(groups), n_rep, seed)
    d = met["B"]["vs_M"]["logloss"]            # B - M = without - with
    halves = validate.half_gains(np.asarray(y), np.asarray(p_without), np.asarray(p_with), np.asarray(groups), np.asarray(is_first), n_rep,
                                 seed + 7)
    return dict(logloss_without=met["B"]["logloss"], logloss_with=met["M"]["logloss"], gain=d["diff"], lo=d["lo"], hi=d["hi"],
                halves=halves, n_games=int(len(y)), n_events=int(np.max(groups)) + 1)


def shipping_verdict(tested: dict, confirm: bool = True) -> dict:
    """The app's rule (`pipeline.decide_component`'s core): ship only with a material (>= MIN_GAIN) significant gain on the earlier
    half of the hold-out events, confirmed on the later half, and no format materially worse (whole hold-out or either half).
    tested = {fmt: dict(gain, lo, hi, halves)}.  Returns dict(ship, better, worse, tiny, whole_better, text)."""
    from sealed_lab.model import pipeline

    better, worse, tiny, info = pipeline._verdict(tested, confirm)
    ship = bool(better) and not worse
    if ship:
        text = "ship: significant gain >= MIN_GAIN in " + ", ".join(better) + " (confirmed on the later half), no format materially worse"
    elif worse:
        text = "do not ship: materially WORSE in " + ", ".join(worse)
    elif confirm and info.get("failed_B"):
        text = "do not ship: passes the earlier half in " + ", ".join(info["failed_B"]) + " but not confirmed on the later half"
    elif info.get("whole_better"):
        text = "do not ship: whole-hold-out gain in " + ", ".join(info["whole_better"]) + " fails the earlier-half decision"
    elif tiny:
        text = "do not ship: detectable but below MIN_GAIN in " + ", ".join(tiny)
    else:
        text = "do not ship: no significant gain (every 95% CI includes zero)"
    return dict(ship=ship, better=better, worse=worse, tiny=tiny, whole_better=info.get("whole_better", []),
                min_gain=pipeline.MIN_GAIN, text=text, confirmation=info)


# =====================================================================================================================
# worker: one condition
# =====================================================================================================================

def real_sets_dir() -> Path:
    return Path(os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")) / "sealed_lab" / "sets"


def deployed_spec(real_set: Path) -> dict:
    """Components that resolved on the real (base) model of the set, from its meta.json."""
    m = json.loads((real_set / "model" / "meta.json").read_text(encoding="utf-8"))
    applied = [k for k in COMP_ORDER if (m.get(k) or {}).get("applied")]
    unsupported = [k for k in applied if k not in ("synergy", "matchups", "castability")]
    if unsupported:
        raise NotImplementedError(f"deployed components {unsupported} are not handled by this script")
    mo = m.get("matchups") or {}
    return dict(components=applied, opp_mode=mo.get("encoding") if mo.get("applied") else None,
                built_at=m.get("built_at"), code_version=m.get("code_version"))


def prepare_home(sc: str, cond: str, home: Path, n_jobs: int) -> dict:
    """Copy the real set folder (corpus, caches, registry; not the model / validation cache) into `home`, change the registry for
    the condition and rebuild the corpus there.  Returns timing and file info."""
    t0 = time.time()
    src = real_sets_dir() / sc
    dst = home / "sets" / sc
    if dst.exists():
        shutil.rmtree(home)
    dst.mkdir(parents=True)
    shutil.copytree(src / "corpus", dst / "corpus")
    shutil.copytree(src / "cache", dst / "cache", ignore=shutil.ignore_patterns("validation", "sealed_lab_*"))
    shutil.copy2(src / "registry.json", dst / "registry.json")
    from sealed_lab.ingest import registry
    from sealed_lab.ingest.corpus import build_corpus
    from sealed_lab.ingest.detect import detect_file

    add_events, rm = CONDS[sc][cond]
    ents = registry.entries(sc)
    rm_fp, add_paths = registry_plan(ents, sc, add_events, rm)
    for fp in rm_fp:
        registry.remove(sc, fp)
    notes = []
    for p in add_paths:
        info = detect_file(p)
        res = registry.register(info)
        notes.append(f"{Path(p).name}: kind={info.kind} event={info.event} fmt={info.fmt} registered={res.registered} {res.note}")
        if not res.registered or info.kind != "game":
            raise RuntimeError(f"could not register {p}: {notes[-1]}")
    t1 = time.time()
    man = build_corpus(sc, n_jobs=n_jobs, fetch_scryfall=False)
    return dict(removed=len(rm_fp), added=notes, registry=[(d["kind"], d.get("event")) for d in registry.entries(sc)],
                corpus_seconds=round(time.time() - t1, 1), prepare_seconds=round(time.time() - t0, 1),
                corpus_games=man["tables"]["games"], n_games_by_fmt=man["audit"]["n_games_by_fmt"], warnings=man.get("warnings", []))


def stage_stats(stages: dict, fits: dict, target: str, t_cut) -> dict:
    """Per fitted stage: sizes, sigma, lambda, stacking weight, card-beta SD (plain and exposure-weighted over non-basic cards), the
    draft stage's event intercepts and a re-check that prior stages hold only events that started before the cut."""
    import polars as pl

    out: dict = {}
    cut = np.datetime64(t_cut, "us")
    for name, f in fits.items():
        st = stages[name]
        beta = np.asarray(f.beta_total, float)
        ex = np.asarray(st.exposure, float)
        starts = np.asarray(st.group_start).astype("datetime64[us]")
        d = dict(n_games=int(st.n_games), n_events=int(st.n_groups), sigma=float(f.sigma), lam=None if f.lam is None else float(f.lam),
                 lam_se=None if f.lam_se is None else float(f.lam_se), lam_from=f.lam_from, stack_w=float(f.w), scheme=f.cv_scheme,
                 cv_best_index=int(np.argmin(f.cv_loss)) if f.cv_loss else None, cv_grid=[float(v) for v in f.cv_grid],
                 beta_sd_plain=weighted_sd(beta[ex > 0]), beta_sd_exposure_weighted=weighted_sd(beta, ex),
                 n_cards_with_exposure=int((ex > 0).sum()), first_start=str(starts.min()), last_start=str(starts.max()),
                 before_cut=bool(starts.max() < cut) if name != target else None)
        if name == "draft":
            g = st.games
            if "event" in g.columns:
                vc = g.group_by("event").agg(pl.len().alias("games"), pl.col("draft_id").n_unique().alias("events"))
                d["by_event"] = {r[0]: dict(games=int(r[1]), events=int(r[2])) for r in vc.iter_rows()}
            names = list(st.enc.names)
            d["ref_event"] = st.enc.ref_event
            d["event_intercepts"] = {nm: float(f.cov[j]) for j, nm in enumerate(names) if nm.startswith("event:")}
        out[name] = d
    return out


def worker(sc: str, cond: str, out: Path, n_jobs: int) -> int:
    t_all = time.time()
    home = out / "homes" / f"{sc}_{cond}"
    res_dir = out / "results" / f"{sc}_{cond}"
    res_dir.mkdir(parents=True, exist_ok=True)
    os.environ["SEALED_LAB_HOME"] = str(home)               # BEFORE any sealed_lab call (children inherit it)
    home.mkdir(parents=True, exist_ok=True)
    real_home = Path(os.environ.get("LOCALAPPDATA", "")) / "sealed_lab"
    from sealed_lab import config

    assert config.app_home().resolve() == home.resolve() and config.app_home().resolve() != real_home.resolve(), "wrong home"
    spec = deployed_spec(real_sets_dir() / sc)
    print(f"[{sc}/{cond}] deployed components of the real {sc} model: {spec}", flush=True)
    prep = prepare_home(sc, cond, home, n_jobs)
    print(f"[{sc}/{cond}] corpus ready in {prep['prepare_seconds']}s: {prep['n_games_by_fmt']}; registry {prep['registry']}", flush=True)
    for w in prep["warnings"]:
        print(f"[{sc}/{cond}] corpus warning: {w}", flush=True)

    import polars as pl
    from scipy.special import expit

    from sealed_lab.model import cardvalue, castability as cast_mod, design, glm, pipeline, structure, validate
    from sealed_lab.model import synergy as syn_mod

    t_m = time.time()
    inp = pipeline.load_inputs(sc)
    inp.cards = cast_mod.with_cost_groups(inp.cards, sc)
    games, cards = inp.games, inp.cards
    n_games = {f: int(games.filter(pl.col("fmt") == f).height) for f in ("draft", "bo1", "bo3")}
    formats = [f for f in pipeline.FMT_ORDER if n_games.get(f, 0) >= 1]
    keys, table = pipeline.phi_for_corpus(inp, n_jobs, None, None, config.cache_dir(sc))
    per_deck = games.group_by("deck_key").len().sort("deck_key")
    wts = np.zeros(len(keys))
    wts[np.searchsorted(keys, per_deck["deck_key"].to_numpy())] = per_deck["len"].to_numpy()
    space = design.build_deck_space(keys, table[:, :structure.N_FEATURES], inp.deck_cards, cards, weights=np.maximum(wts, 1e-9))
    # the deployed deck space, as pipeline.calculate.final_space builds it for castability and synergy
    base_ = space
    if "castability" in spec["components"]:
        from sealed_lab.model import simcalib

        try:
            sim_regimes, _ = simcalib.keep_regimes(sc)
        except Exception as e:  # noqa: BLE001 - calculate falls back to the shipped average too
            print(f"[{sc}/{cond}] keep rule unavailable ({e}); shipped average", flush=True)
            sim_regimes = {}
        masks_k = pipeline.regime_masks(games, keys)
        cast_tabs = {r_: pipeline.cast_for_corpus(inp, sim_regimes.get(r_, r_), n_jobs, None, None, config.cache_dir(sc), need=m_)
                     for r_, m_ in masks_k.items()}
        base_ = pipeline.cast_space(inp, games, keys, {r_: table for r_ in cast_tabs}, cast_tabs)
    fspace, extras = base_, []
    if "synergy" in spec["components"]:
        is_land_c = cards.sort("card_id")["is_land"].to_numpy()
        pair_ab, _pg = syn_mod.select_pairs(space, wts, is_land_c, min_games=syn_mod.MIN_PAIR_GAMES)
        if len(pair_ab):
            fspace = design.with_pairs(fspace, pair_ab)
            extras = [dict(name="pairs", n=len(pair_ab))]
    setup_s = time.time() - t_m
    print(f"[{sc}/{cond}] model inputs ready in {setup_s:.0f}s; formats {formats}; extras {extras}", flush=True)

    # ---- capture hooks (the app's functions are called unchanged; these only copy what they compute)
    CAP: dict = dict(tag=None, stages={}, fits={}, preds=None, tcut=None)
    o_cs, o_rc, o_cm = validate.chain_stages, cardvalue.run_chain, validate.cluster_metrics

    def cs_w(fmt, g, g_target, t_cut, sp_):
        r = o_cs(fmt, g, g_target, t_cut, sp_)
        if CAP["tag"]:
            CAP["stages"][CAP["tag"]] = r
            CAP["tcut"] = t_cut
        return r

    def rc_w(*a, **k):
        r = o_rc(*a, **k)
        if CAP["tag"]:
            CAP["fits"][CAP["tag"]] = r
        return r

    def cm_w(y, preds, groups, n_rep, seed):
        if CAP["tag"] == "deployed":
            CAP["preds"] = {k: np.array(v) for k, v in preds.items()}
        return o_cm(y, preds, groups, n_rep, seed)

    validate.chain_stages, cardvalue.run_chain, validate.cluster_metrics = cs_w, rc_w, cm_w
    opts = cardvalue.ChainOpts()                    # calculate's defaults: time-split sigma, stacking, not leak-free; recency off (not applied)
    result: dict = dict(set=sc, cond=cond, deployed_spec=spec, prepare=prep, setup_seconds=round(setup_s, 1), n_games=n_games,
                        opts=opts.key(), n_rep=N_REP, formats={}, extras=extras)
    with glm.shared_pool(n_jobs):
        for i, fmt in enumerate(FMTS):
            if fmt not in formats:
                continue
            gT = games.filter(pl.col("fmt") == fmt)
            split = validate._holdout_split(gT)
            if split is None:
                result["formats"][fmt] = dict(status="skipped")
                continue
            hold_ids, t_cut, _n_ev, _n_hold = split
            g_ho = gT.filter(pl.col("draft_id").is_in(hold_ids))
            tmp = cardvalue.make_tmpdir(config.cache_dir(sc))
            try:
                CAP.update(tag="plain", stages={}, fits={}, preds=None)
                keep: dict = {}
                t0 = time.time()
                r_val = validate.run_validation(fmt, games, space, cards, inp.lazy["hands"], tmp, n_jobs, seed=i, n_rep=N_REP, keep=keep,
                                                opts=opts)
                t_plain = time.time() - t0
                print(f"[{sc}/{cond}] {fmt}: plain chain + baselines {t_plain:.0f}s  LL(M) {r_val['metrics']['M']['logloss']:.5f}", flush=True)
                plain_stats = stage_stats(CAP["stages"]["plain"], keep[fmt]["ref_fits"], fmt, t_cut)
                CAP.update(tag="deployed", stages={}, fits={}, preds=None)
                t0 = time.time()
                dep = validate.validate_deployed(fmt, games, space, fspace, cards, tmp, n_jobs, list(spec["components"]), seed=i,
                                                 n_rep=N_REP, recency_h=None, attr_load=None, theta_prior=None, extras=extras or None,
                                                 opp_mode=spec["opp_mode"], m_ref=keep[fmt], cache=None, opts=opts)
                t_dep = time.time() - t0
                print(f"[{sc}/{cond}] {fmt}: deployed {t_dep:.0f}s  LL(MD) {dep['logloss_MD']:.5f} (M {dep['logloss_M']:.5f})", flush=True)
                dep_stats = stage_stats(CAP["stages"]["deployed"], CAP["fits"]["deployed"], fmt, t_cut)
                # draft events that are in the corpus but start at/after the cut (cannot enter either model)
                dg = games.filter(pl.col("fmt") == "draft")
                dev = design.event_starts(dg) if dg.height else None
                n_after = int(dev.filter(pl.col("start") >= t_cut).height) if dev is not None else 0
                n_draft_ev = int(dev.height) if dev is not None else 0
            finally:
                cardvalue.cleanup(tmp)
                CAP["tag"] = None
            P = CAP["preds"]
            assert P is not None and np.allclose(P["M"], keep[fmt]["pM"]), "captured predictions do not match the plain chain"
            did = g_ho["draft_id"].to_numpy().astype(str)
            first = design.event_starts(g_ho)["draft_id"].to_list()
            first_ids = np.array(first[:(len(first) + 1) // 2]).astype(str)
            np.savez(res_dir / f"preds_{fmt}.npz", y=keep[fmt]["y"], did=did, pM=P["M"], pMD=P["MD"], pB0=P["B0"], first_ids=first_ids)
            result["formats"][fmt] = dict(
                status="ok", t_cut=str(t_cut), n_holdout_events=int(len(hold_ids)), n_holdout_games=int(g_ho.height),
                logloss_M=r_val["metrics"]["M"]["logloss"], logloss_M_ci=r_val["metrics"]["M"]["logloss_ci"],
                logloss_B0=r_val["metrics"]["B0"]["logloss"], logloss_MD=dep["logloss_MD"], deployed_vs_M=dict(gain=dep["gain"], lo=dep["lo"], hi=dep["hi"]),
                seconds_plain=round(t_plain, 1), seconds_deployed=round(t_dep, 1),
                draft_events_in_corpus=n_draft_ev, draft_events_at_or_after_cut=n_after,
                plain=plain_stats, deployed=dep_stats, calibration_slope_MD=dep.get("calibration_slope"))
            (res_dir / "result.partial.json").write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    validate.chain_stages, cardvalue.run_chain, validate.cluster_metrics = o_cs, o_rc, o_cm
    result["seconds_total"] = round(time.time() - t_all, 1)
    (res_dir / "result.json").write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    print(f"[{sc}/{cond}] done in {result['seconds_total']}s", flush=True)
    return 0


# =====================================================================================================================
# run (orchestrate) and report
# =====================================================================================================================

def plan(sets: list[str]) -> list[tuple[str, str]]:
    return [(s, c) for s in sets for c in CONDS[s]]


def run(out: Path, sets: list[str], n_jobs: int) -> int:
    (out / "logs").mkdir(parents=True, exist_ok=True)
    for sc, cond in plan(sets):
        if (out / "results" / f"{sc}_{cond}" / "result.json").exists():
            print(f"skip {sc}/{cond} (done)", flush=True)
            continue
        t0 = time.time()
        log = out / "logs" / f"{sc}_{cond}.log"
        print(f"== {sc}/{cond} -> {log}", flush=True)
        env = dict(os.environ)
        env["SEALED_LAB_HOME"] = str(out / "homes" / f"{sc}_{cond}")
        with open(log, "w", encoding="utf-8") as fh:
            rc = subprocess.call([sys.executable, str(Path(__file__).resolve()), "worker", sc, cond, "--out", str(out), "--jobs", str(n_jobs)],
                                 stdout=fh, stderr=subprocess.STDOUT, env=env)
        print(f"   exit {rc} after {time.time() - t0:.0f}s", flush=True)
        if rc != 0:
            print(f"   FAILED {sc}/{cond}; continuing with the next condition", flush=True)
    return 0


def _load(out: Path, sc: str, cond: str):
    d = out / "results" / f"{sc}_{cond}"
    if not (d / "result.json").exists():
        return None
    return json.loads((d / "result.json").read_text(encoding="utf-8")), d


def compare(out: Path, sc: str, cw: str, cwo: str) -> dict | None:
    a, b = _load(out, sc, cw), _load(out, sc, cwo)
    if a is None or b is None:
        return None
    (ra, da), (rb, db) = a, b
    res: dict = dict(set=sc, with_cond=cw, without_cond=cwo, formats={})
    tested = {"M": {}, "MD": {}}
    for fmt in FMTS:
        fa, fb = ra["formats"].get(fmt), rb["formats"].get(fmt)
        if not fa or not fb or fa.get("status") != "ok" or fb.get("status") != "ok":
            continue
        za, zb = np.load(da / f"preds_{fmt}.npz"), np.load(db / f"preds_{fmt}.npz")
        ia, ib, groups = align_holdouts(dict(did=za["did"], y=za["y"]), dict(did=zb["did"], y=zb["y"]))
        uniq = np.unique(za["did"].astype(str))
        is_first = np.isin(uniq, za["first_ids"].astype(str))
        assert np.array_equal(np.sort(za["first_ids"].astype(str)), np.sort(zb["first_ids"].astype(str))), "halves differ"
        y = za["y"][ia]
        row: dict = dict(n_holdout_events=int(len(uniq)), n_holdout_games=int(len(y)), identical_holdout=True)
        for m, key in (("M", "pM"), ("MD", "pMD")):
            g = paired_gain(y, zb[key][ib], za[key][ia], groups, is_first, N_REP_CMP, seed=11 + (0 if fmt == "bo3" else 1))
            row[m] = g
            tested[m][fmt] = g
        res["formats"][fmt] = row
    res["verdict_deployed"] = shipping_verdict(tested["MD"]) if tested["MD"] else None
    res["verdict_plain"] = shipping_verdict(tested["M"]) if tested["M"] else None
    return res


def _f(x, d=5, plus=True) -> str:
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:{'+' if plus else ''}.{d}f}"


def _cond_lines(r: dict) -> list[str]:
    lines = []
    for fmt in FMTS:
        f = r["formats"].get(fmt)
        if not f or f.get("status") != "ok":
            continue
        for kind in ("deployed", "plain"):
            stg = f[kind]
            parts = []
            for name in ("draft", "bo1", "bo3"):
                s = stg.get(name)
                if not s:
                    continue
                t = f"{name}: sigma {s['sigma']:.4f}"
                if s["lam"] is not None:
                    t += f", lambda {s['lam']:.3f}"
                if s["scheme"] == "forward":
                    t += f", w {s['stack_w']:.1f}"
                t += f", beta SD {s['beta_sd_plain']:.4f} (exp-w {s['beta_sd_exposure_weighted']:.4f}), {s['n_games']:,} g / {s['n_events']:,} ev"
                parts.append(t)
            lines.append(f"  - {fmt} [{kind}]: " + "; ".join(parts))
        d = f["deployed"].get("draft")
        if d:
            lines.append(f"    draft by event: {d.get('by_event')}; ref event {d.get('ref_event')}; intercepts {({k: round(v, 3) for k, v in d.get('event_intercepts', {}).items()})}; "
                         f"before cut: {d['before_cut']} (last draft event start {d['last_start']} < cut {f['t_cut']}); draft events at/after cut excluded: "
                         f"{f['draft_events_at_or_after_cut']} of {f['draft_events_in_corpus']}")
        lines.append(f"    time: plain {f['seconds_plain']:.0f}s + deployed {f['seconds_deployed']:.0f}s; LL(M) {f['logloss_M']:.5f}, LL(MD) {f['logloss_MD']:.5f}")
    return lines


def report(out: Path) -> int:
    doc: dict = dict(comparisons=[], conditions={})
    md = ["# Does extra PickTwo draft data improve the forward hold-out?", "",
          "gain = LL(without the PickTwo data) - LL(with it); positive = the PickTwo data helps. 95% paired cluster bootstrap over hold-out events "
          f"({N_REP_CMP} replicates, `validate.cluster_metrics`); halves = `validate.half_gains` (A = earlier events, B = later). "
          "Deployed model MD = the components that resolved on the real set; M = the plain chain.", ""]
    for sc, cw, cwo in PAIRS:
        c = compare(out, sc, cw, cwo)
        if c is None:
            continue
        doc["comparisons"].append(c)
        md.append(f"## {sc}: with = `{cw}`, without = `{cwo}`")
        md.append("")
        md.append("| fmt | model | LL without | LL with | gain [95% CI] | half A | half B (90% lo) | events / games |")
        md.append("|---|---|---|---|---|---|---|---|")
        for fmt in FMTS:
            r = c["formats"].get(fmt)
            if not r:
                continue
            for m in ("MD", "M"):
                g = r[m]
                h = g["halves"]
                md.append(f"| {fmt} | {m} | {g['logloss_without']:.5f} | {g['logloss_with']:.5f} | {_f(g['gain'])} [{_f(g['lo'])}, {_f(g['hi'])}] | "
                          f"{_f(h['A']['gain'])} [{_f(h['A']['lo'])}, {_f(h['A']['hi'])}] | {_f(h['B']['gain'])} [{_f(h['B']['lo'])}, {_f(h['B']['hi'])}] ({_f(h['B']['lo90'])}) | "
                          f"{r['n_holdout_events']} / {r['n_holdout_games']} |")
        v = c["verdict_deployed"]
        if v:
            md.append("")
            md.append(f"Verdict (deployed model, MIN_GAIN {v['min_gain']:g}): **{v['text']}**")
        v = c["verdict_plain"]
        if v:
            md.append(f"Verdict (plain chain): {v['text']}")
        md.append("")
    md.append("## Per-condition statistics")
    md.append("")
    for sc in CONDS:
        for cond in CONDS[sc]:
            L = _load(out, sc, cond)
            if L is None:
                continue
            r, _d = L
            doc["conditions"][f"{sc}_{cond}"] = r
            md.append(f"### {sc} / {cond}  (corpus games {r['n_games']}, deployed components {r['deployed_spec']['components']}, matchup mode {r['deployed_spec']['opp_mode']}; total {r['seconds_total']:.0f}s, setup {r['prepare']['prepare_seconds']}s + {r['setup_seconds']}s)")
            md.append("")
            md += _cond_lines(r)
            md.append("")
    (out / "extra_draft.json").write_text(json.dumps(doc, indent=1, default=str), encoding="utf-8")
    (out / "extra_draft.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--out", required=True)
    p.add_argument("--sets", nargs="+", default=["OM1", "MSH", "FRA", "HOB"])
    p.add_argument("--jobs", type=int, default=None)
    p = sub.add_parser("worker")
    p.add_argument("set")
    p.add_argument("cond")
    p.add_argument("--out", required=True)
    p.add_argument("--jobs", type=int, default=None)
    p = sub.add_parser("report")
    p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    n_jobs = int(getattr(a, "jobs", None) or os.cpu_count() or 1)
    if a.cmd == "run":
        return run(out, [s.upper() for s in a.sets], n_jobs)
    if a.cmd == "worker":
        return worker(a.set.upper(), a.cond, out, n_jobs)
    return report(out)


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    sys.exit(main())
