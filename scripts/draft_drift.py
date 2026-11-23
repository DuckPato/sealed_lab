"""Is the draft-stage card value drifting over the life of a set?  (diagnostic only: no change to the model)

    python scripts/draft_drift.py prep   OUT [--sets ...]                  scratch homes (corpus + cache + meta copied READ-ONLY from the app home)
    python scripts/draft_drift.py fit    OUT [--sets ...] [-R 10]           all fits (time split + R random splits, mixed and per draft type)
    python scripts/draft_drift.py survey OUT                               which draft-type data is on disk vs what the draft stage uses
    python scripts/draft_drift.py report OUT [--sets ...]                  drift.json / drift.md
    python scripts/draft_drift.py time-one SET OUT                         time one set-up + fit (nothing is saved)

OUT is a scratch folder; SEALED_LAB_HOME is set to OUT/homes/SET (a scratch copy), never the app home (which is only READ by `prep`).

Design.  For a set the draft stage uses all draft games of its corpus (events PremierDraft / TradDraft, and PickTwoDraft where the owner dropped
that file in).  The events (draft_id) are sorted by start (first game time) and cut at the median into EARLY and LATE halves (equal numbers of
events).  The draft stage (zero prior mean, castability deck space, sigma fixed to the one the FULL-data artifact chose for the draft stage, no
cross-validation) is refitted on each half with the pipeline's own fit (`split_half_decompose.fit_draft_stage`, the cluster-robust SE of the
card betas).  Per card  d = beta_late - beta_early,  z = d / sqrt(se_e^2 + se_l^2).  Because the betas are penalised (shrunk), z is not exactly
N(0,1) under no drift, so the same statistic is computed for R random splits of the same events into two equal halves (stratified by event type
and time, `split_half_search.split_events`): their chi-square, share |z|>2 and mean square of d are the null.  Drift verdict = time split vs this
null.  Drift variance = E[d^2] (time) - mean E[d^2] (random) per card group (an unbiased estimate of the variance of the real change when the
noise is the same in both kinds of split; the estimate is attenuated by the shrinkage of the betas, so it is, if anything, a lower bound).

Approximations (stated in the report too): (1) one common castability deck space per set, built once on ALL draft events, is used for every
half (a refit of the space per half, as `fit_draft_stage` does when it builds its own set-up, changes the nuisance feature scaling only; the
check `check` compares the two on one split); (2) pp per copy = 100 * 0.25 * logit (slope of the win rate at 50%); (3) the exposure tertiles
and rarity groups use the whole scope's draft exposure; (4) the card set is the non-basic cards with exposure >= MIN_EXPO copy-games in both
time halves.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

SETS = ["ECL", "EOE", "FIN", "FRA", "HOB", "MSH", "OM1", "SOS", "TDM", "TLA", "TMT"]
APP_HOME_ENV = "LOCALAPPDATA"
from sealed_lab.ingest.datasets import default_data_dir as _default_data_dir  # noqa: E402
DATA_ROOT = Path(_default_data_dir())          # env SEALED_LAB_DATA_DIR, else the app's data folder default
MIN_EXPO = 100.0              # copy-games a card needs in each time half to enter the statistics
MIN_TYPE_GAMES = 40000        # a draft type gets its own analysis from this many games
TERTILES = ("T1", "T2", "T3")
RARITIES = ("C", "U", "R", "M")
PP_PER_LOGIT = 100 * 0.25     # win-rate points per logit at p = 0.5 (approximation)
USED_SCAN_EVENTS = ("Sealed", "TradSealed", "PremierDraft", "TradDraft")     # what the folder scan registers by itself


# =====================================================================================================================
# pure arithmetic (tested without any model)
# =====================================================================================================================

def time_split(ids, starts):
    """EARLY / LATE halves: events sorted by (start, id), the first n//2 are EARLY (the odd event out goes to LATE)."""
    ids = list(ids)
    order = sorted(range(len(ids)), key=lambda i: (starts[i], str(ids[i])))
    h = len(order) // 2
    return [ids[i] for i in order[:h]], [ids[i] for i in order[h:]]


def zstats(b1, se1, b2, se2, valid=None):
    """d = b2 - b1 and z = d / sqrt(se1^2 + se2^2) per card; entries outside `valid` (or without a positive finite SE) come back NaN."""
    b1, b2, se1, se2 = (np.asarray(x, float) for x in (b1, b2, se1, se2))
    d = b2 - b1
    den = np.sqrt(se1 ** 2 + se2 ** 2)
    ok = np.isfinite(d) & np.isfinite(den) & (den > 0)
    if valid is not None:
        ok &= np.asarray(valid, bool)
    z = np.full(d.shape, np.nan)
    z[ok] = d[ok] / den[ok]
    d = np.where(ok, d, np.nan)
    return d, z


def chi2_summary(z, thr: float = 2.0) -> dict:
    """Sum of z^2 over the finite entries against n (nominal chi-square df), its nominal p-value, share |z| > thr (nominal 4.55% at 2)."""
    from scipy.stats import chi2, norm

    z = np.asarray(z, float)
    z = z[np.isfinite(z)]
    n = int(len(z))
    if n == 0:
        return dict(n=0, chi2=float("nan"), chi2_over_n=float("nan"), p_nominal=float("nan"), share_gt=float("nan"), thr=thr)
    s = float((z ** 2).sum())
    return dict(n=n, chi2=s, chi2_over_n=s / n, p_nominal=float(chi2.sf(s, n)), share_gt=float((np.abs(z) > thr).mean()),
                share_nominal=float(2 * norm.sf(thr)), thr=thr)


def empirical_p(obs: float, null) -> float:
    """One-sided empirical p of `obs` against null draws: (1 + #{null >= obs}) / (R + 1)."""
    null = np.asarray(null, float)
    null = null[np.isfinite(null)]
    return float((1 + int((null >= obs).sum())) / (len(null) + 1))


def null_compare(obs: float, null) -> dict:
    null = np.asarray(null, float)
    null = null[np.isfinite(null)]
    mu = float(null.mean()) if len(null) else float("nan")
    sd = float(null.std(ddof=1)) if len(null) > 1 else float("nan")
    return dict(obs=float(obs), null_mean=mu, null_sd=sd, ratio=float(obs / mu) if mu else float("nan"),
                z_vs_null=float((obs - mu) / sd) if sd and np.isfinite(sd) and sd > 0 else float("nan"),
                p_emp=empirical_p(obs, null), R=int(len(null)))


def group_msq(d, labels, groups) -> dict:
    """Mean of d^2 (second moment about 0) and the number of finite entries per group; `groups` maps name -> bool mask over entries or a
    label (compared with `labels`)."""
    d = np.asarray(d, float)
    labels = np.asarray(labels, dtype=object)
    out = {}
    for name, g in groups.items():
        m = np.asarray(g, bool) if not isinstance(g, str) else (labels == g)
        m = m & np.isfinite(d)
        out[name] = dict(n=int(m.sum()), msq=float((d[m] ** 2).mean()) if m.any() else float("nan"))
    return out


def excess_variance(obs_msq: float, null_msqs) -> dict:
    """Variance of the real change = E[d^2](time split) - mean E[d^2](random splits); the SD (clipped at 0) and the significance of the
    excess against the spread of the null draws."""
    null = np.asarray(null_msqs, float)
    null = null[np.isfinite(null)]
    mu = float(null.mean()) if len(null) else float("nan")
    sd = float(null.std(ddof=1)) if len(null) > 1 else float("nan")
    ex = float(obs_msq - mu)
    return dict(msq_time=float(obs_msq), msq_null=mu, msq_null_sd=sd, excess=ex, drift_sd=float(np.sqrt(max(ex, 0.0))),
                rms_time=float(np.sqrt(obs_msq)) if obs_msq >= 0 else float("nan"), rms_null=float(np.sqrt(mu)) if mu >= 0 else float("nan"),
                z_excess=float(ex / sd) if sd and np.isfinite(sd) and sd > 0 else float("nan"))


def pp_per_copy(logit_sd: float) -> float:
    return float(PP_PER_LOGIT * logit_sd)


def fisher_combine(ps) -> dict:
    """Fisher's method on independent p-values: -2 sum log p ~ chi2(2k).  (The empirical p-values are discrete, min 1/(R+1): conservative-ish.)"""
    from scipy.stats import chi2

    ps = np.asarray([p for p in ps if np.isfinite(p)], float)
    s = float(-2 * np.log(ps).sum())
    return dict(k=int(len(ps)), stat=s, p=float(chi2.sf(s, 2 * len(ps))))


def pooled_null_sums(null_by_set: dict) -> np.ndarray:
    """Pooled null draws: for each random split r the sum over sets of the null statistic (sets with fewer draws use the common prefix)."""
    R = min(len(v) for v in null_by_set.values())
    return np.sum([np.asarray(v, float)[:R] for v in null_by_set.values()], axis=0)


def corr_slope(x, y) -> dict:
    """Pearson correlation of y with x and the OLS slope of y on x (finite pairs)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return dict(n=int(len(x)), corr=float("nan"), slope=float("nan"))
    c = np.corrcoef(x, y)[0, 1]
    return dict(n=int(len(x)), corr=float(c), slope=float(c * y.std() / x.std()))


def exposure_tertiles(exposure, eligible) -> np.ndarray:
    """T1 (low) .. T3 (high) by exposure rank among the eligible cards (equal sizes +- 1); others 'none'."""
    exposure = np.asarray(exposure, float)
    idx = np.nonzero(np.asarray(eligible, bool))[0]
    out = np.full(len(exposure), "none", dtype=object)
    if len(idx):
        order = idx[np.lexsort((idx, exposure[idx]))]
        for k, chunk in enumerate(np.array_split(order, 3)):
            out[chunk] = TERTILES[k]
    return out


def analyse_scope(time_fit: dict, null_fits: list[dict], labels: dict, min_expo: float = MIN_EXPO) -> dict:
    """All statistics of one scope.  time_fit / null_fits[r]: dict(b1, se1, b2, se2, e1, e2) arrays over card ids (1 = EARLY / half A, 2 = LATE
    / half B).  labels: dict(rarity (object array), basic (bool array))."""
    expo = np.asarray(time_fit["e1"], float) + np.asarray(time_fit["e2"], float)
    elig = (~np.asarray(labels["basic"], bool)) & (np.asarray(time_fit["e1"]) >= min_expo) & (np.asarray(time_fit["e2"]) >= min_expo)
    tert = exposure_tertiles(expo, elig)
    rar = np.asarray(labels["rarity"], dtype=object)
    groups = {"ALL": elig, **{t: tert == t for t in TERTILES}, **{r: elig & (rar == r) for r in RARITIES}, "R+M": elig & ((rar == "R") | (rar == "M"))}

    def one(f):
        d, z = zstats(f["b1"], f["se1"], f["b2"], f["se2"], elig)
        return d, z

    d_t, z_t = one(time_fit)
    res: dict = dict(n_cards=int(np.isfinite(z_t).sum()), sigma=time_fit.get("sigma"))
    cs_t = chi2_summary(z_t)
    nulls = [one(f) for f in null_fits]
    cs_n = [chi2_summary(z) for _, z in nulls]
    res["time"] = cs_t
    res["null"] = dict(chi2=[c["chi2"] for c in cs_n], chi2_over_n=[c["chi2_over_n"] for c in cs_n], share_gt=[c["share_gt"] for c in cs_n])
    res["verdict_chi2"] = null_compare(cs_t["chi2"], res["null"]["chi2"])
    res["verdict_share_gt2"] = null_compare(cs_t["share_gt"], res["null"]["share_gt"])
    # drift size per group
    msq_t = group_msq(d_t, rar, groups)
    msq_n = [group_msq(d, rar, groups) for d, _ in nulls]
    res["groups"] = {}
    for g in groups:
        ex = excess_variance(msq_t[g]["msq"], [m[g]["msq"] for m in msq_n]) if msq_t[g]["n"] else {}
        if ex:
            ex["n"] = msq_t[g]["n"]
            ex["drift_sd_pp"] = pp_per_copy(ex["drift_sd"])
            ex["rms_time_pp"] = pp_per_copy(ex["rms_time"])
            ex["rms_null_pp"] = pp_per_copy(ex["rms_null"])
        res["groups"][g] = ex
    # direction: d against beta_early (time) vs the same for the random splits (d against beta_A)
    cd_t = corr_slope(np.where(elig, time_fit["b1"], np.nan), d_t)
    cd_n = [corr_slope(np.where(elig, f["b1"], np.nan), d) for f, (d, _) in zip(null_fits, nulls)]
    res["direction"] = dict(time=cd_t, null_corr=[c["corr"] for c in cd_n], null_slope=[c["slope"] for c in cd_n],
                            null_corr_mean=float(np.nanmean([c["corr"] for c in cd_n])) if cd_n else float("nan"),
                            null_corr_sd=float(np.nanstd([c["corr"] for c in cd_n], ddof=1)) if len(cd_n) > 1 else float("nan"),
                            null_slope_mean=float(np.nanmean([c["slope"] for c in cd_n])) if cd_n else float("nan"),
                            null_slope_sd=float(np.nanstd([c["slope"] for c in cd_n], ddof=1)) if len(cd_n) > 1 else float("nan"),
                            mean_d_time=float(np.nanmean(d_t)), mean_d_null=[float(np.nanmean(d)) for d, _ in nulls])
    dcorr = res["direction"]
    dcorr["corr_excess"] = dcorr["time"]["corr"] - dcorr["null_corr_mean"]
    dcorr["corr_z_vs_null"] = (dcorr["time"]["corr"] - dcorr["null_corr_mean"]) / dcorr["null_corr_sd"] if dcorr["null_corr_sd"] and np.isfinite(dcorr["null_corr_sd"]) and dcorr["null_corr_sd"] > 0 else float("nan")
    # per-card top drifters (names added by the caller)
    order = np.argsort(-np.nan_to_num(np.abs(z_t), nan=0.0))[:10]
    res["top_cards"] = [dict(card_id=int(i), z=float(z_t[i]), d=float(d_t[i]), b_early=float(time_fit["b1"][i]), b_late=float(time_fit["b2"][i]))
                        for i in order if np.isfinite(z_t[i])]
    return res


# =====================================================================================================================
# set-up, fits
# =====================================================================================================================

def home_of(out: Path, sc: str) -> Path:
    return Path(out) / "homes" / sc


def app_home_real() -> Path:
    return Path(os.environ[APP_HOME_ENV]) / "sealed_lab"


def prep_home(out: Path, sc: str) -> None:
    """Scratch home of a set: corpus + cache + meta.json copied (read-only) from the app home."""
    src = app_home_real() / "sets" / sc
    dst = home_of(out, sc) / "sets" / sc
    for sub in ("corpus", "cache"):
        if (dst / sub).exists():
            continue
        if (src / sub).exists():
            shutil.copytree(src / sub, dst / sub)
    (dst / "model").mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / "model" / "meta.json", dst / "model" / "meta.json")
    print(f"prep {sc}: {dst}", flush=True)


def events_table(setup):
    """Per draft_id: event type, start (first game time, fallback draft time), games."""
    import polars as pl

    g = setup.inp.games
    return (g.group_by("draft_id").agg(pl.col("event").first().alias("event"), pl.col("game_time").min().alias("t_game"),
                                       pl.col("draft_time").min().alias("t_draft"), pl.len().alias("n"))
            .with_columns(pl.coalesce("t_game", "t_draft").alias("start")).sort("start", "draft_id"))


def fit_subset(setup, ids, sigma: float) -> dict:
    """Draft-stage fit on the given events in the set's common deck space (see the module docstring)."""
    import polars as pl

    from sealed_lab.model.design import build_stage
    from split_half_decompose import fit_draft_stage

    keep = [str(i) for i in ids]
    g = setup.inp.games.filter(pl.col("draft_id").cast(pl.Utf8).is_in(keep))
    sub = copy.copy(setup)
    sub.stages = {"draft": build_stage("draft", g, setup.fspace)}
    r = fit_draft_stage(setup.set_code, setup.home, keep, sigma=sigma, with_se=True, setup=sub)
    r.pop("setup", None)
    return r


def make_setup(out: Path, sc: str):
    import polars as pl

    from split_half_decompose import HalfSetup

    home = home_of(out, sc)
    os.environ["SEALED_LAB_HOME"] = str(home)
    from sealed_lab.ingest.corpus import load_corpus

    g = load_corpus(sc)["games"].select("draft_id", "fmt").filter(pl.col("fmt") == "draft").select(pl.col("draft_id").cast(pl.Utf8).unique()).collect()
    ids = g["draft_id"].to_list()
    return HalfSetup(sc, home, ids, stages=("draft",))


def scopes_of(ev) -> dict:
    """{scope name: list of draft_ids}: 'mixed' (all) and one scope per event type with >= MIN_TYPE_GAMES games (when >1 type exists)."""
    import polars as pl

    out = {"mixed": ev["draft_id"].to_list()}
    types = ev.group_by("event").agg(pl.col("n").sum().alias("games"), pl.len().alias("events"))
    if types.height > 1:
        for e, games in zip(types["event"].to_list(), types["games"].to_list()):
            if games >= MIN_TYPE_GAMES:
                out[e] = ev.filter(pl.col("event") == e)["draft_id"].to_list()
    return out


def fit_set(out: Path, sc: str, R: int, only_scopes=None) -> None:
    from split_half_search import set_seed, split_events

    t00 = time.time()
    setup = make_setup(out, sc)
    sigma = float(json.loads((home_of(out, sc) / "sets" / sc / "model" / "meta.json").read_text(encoding="utf-8"))["prior_strength"]["stages"]["draft"]["sigma"])
    ev = events_table(setup)
    print(f"{sc}: set-up {time.time() - t00:.0f}s, {ev.height} draft events, {int(ev['n'].sum())} games, sigma {sigma:.4f}", flush=True)
    scopes = scopes_of(ev)
    (Path(out) / "fits").mkdir(parents=True, exist_ok=True)
    for scope, ids in scopes.items():
        if only_scopes and scope not in only_scopes:
            continue
        path = Path(out) / "fits" / f"{sc}__{scope}.pkl"
        if path.exists():
            print(f"{sc}/{scope}: exists, skipped", flush=True)
            continue
        sub = ev.filter(ev["draft_id"].is_in(ids))
        sid, starts, types = sub["draft_id"].to_list(), sub["start"].to_list(), sub["event"].to_list()
        t0 = time.time()
        early, late = time_split(sid, starts)
        fits: dict = {}
        for name, (a, b) in dict(time=(early, late), **{f"rand{r}": split_events(sid, starts, types, seed=set_seed(r, sc)) for r in range(R)}).items():
            ta = time.time()
            fa, fb = fit_subset(setup, a, sigma), fit_subset(setup, b, sigma)
            fits[name] = dict(b1=fa["beta"], se1=fa["se"], e1=fa["exposure"], b2=fb["beta"], se2=fb["se"], e2=fb["exposure"], sigma=sigma,
                              n_events=(len(a), len(b)), n_games=(fa["n_games"], fb["n_games"]), converged=(fa["converged"], fb["converged"]))
            print(f"  {sc}/{scope}/{name}: {time.time() - ta:.0f}s ({fa['n_games']} + {fb['n_games']} games)", flush=True)
        with open(path, "wb") as fh:
            pickle.dump(dict(set=sc, scope=scope, R=R, sigma=sigma, n_events=len(sid), fits=fits,
                             t_early_end=str(starts[sid.index(early[-1])]) if early else None,
                             t_late_start=str(starts[sid.index(late[0])]) if late else None), fh)
        print(f"{sc}/{scope}: {len(sid)} events, {time.time() - t0:.0f}s", flush=True)
    print(f"{sc} done in {time.time() - t00:.0f}s", flush=True)


def check_set(out: Path, sc: str) -> None:
    """Compare the common-space fit with the fit that builds its own set-up on the same events (fit_draft_stage default path)."""
    from split_half_decompose import fit_draft_stage

    setup = make_setup(out, sc)
    sigma = float(setup.meta["prior_strength"]["stages"]["draft"]["sigma"])
    ev = events_table(setup)
    early, late = time_split(ev["draft_id"].to_list(), ev["start"].to_list())
    a = fit_subset(setup, late, sigma)
    b = fit_draft_stage(sc, home_of(out, sc), late, sigma=sigma, with_se=True)
    nb = ~np.isnan(a["se"])
    print(f"check {sc} (LATE half): max |beta common-space - own-space| {np.abs(a['beta'] - b['beta']).max():.2e}; rms {np.sqrt(np.mean((a['beta'] - b['beta'])[nb] ** 2)):.2e}; "
          f"rms beta {np.sqrt(np.mean(a['beta'][nb] ** 2)):.3f}; max |se diff| {np.nanmax(np.abs(a['se'] - b['se'])):.2e}", flush=True)


# =====================================================================================================================
# survey of the draft-type data on disk
# =====================================================================================================================

def survey(out: Path, sets=SETS) -> dict:
    import gzip
    import polars as pl

    from sealed_lab.config import FILE_NAME_RE

    real = app_home_real()
    res: dict = {}
    for sc in sets:
        folder = DATA_ROOT / sc
        reg = json.loads((real / "sets" / sc / "registry.json").read_text(encoding="utf-8")) if (real / "sets" / sc / "registry.json").exists() else {"files": []}
        regd = {(f["kind"], f["event"]): f for f in reg["files"]}
        man = json.loads((real / "sets" / sc / "corpus" / "manifest.json").read_text(encoding="utf-8")) if (real / "sets" / sc / "corpus" / "manifest.json").exists() else {"files": []}
        mand = {(f["kind"], f["event"]): f for f in man["files"]}
        corpus_games = None
        gp = home_of(out, sc) / "sets" / sc / "corpus" / "games.parquet"
        if gp.exists():
            corpus_games = pl.read_parquet(gp, columns=["fmt", "event", "draft_id"]).filter(pl.col("fmt") == "draft")
        rows = []
        seen = {}
        if folder.exists():
            for p in sorted(folder.iterdir()):
                m = FILE_NAME_RE.search(p.name)
                if not m:
                    continue
                kind, code, ev = m.group(1).lower(), m.group(2), m.group(3)
                if code.upper() != sc:
                    continue
                key = (kind, ev)
                if key in seen and not p.name.endswith(".gz"):
                    continue
                seen[key] = p
        for (kind, ev), p in sorted(seen.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            if kind != "game":
                continue
            row = dict(event=ev, file=p.name, size_mb=round(p.stat().st_size / 1e6, 1), registered=(kind, ev) in regd,
                       in_corpus=(kind, ev) in mand and mand[(kind, ev)].get("status") in ("ingested", "cached"),
                       draft_pick_file=("draft", ev) in seen, replay_file=("replay", ev) in seen)
            is_draft = "draft" in ev.lower() and "sealed" not in ev.lower()
            row["is_draft"] = is_draft
            if row["in_corpus"] and is_draft and corpus_games is not None:
                sub = corpus_games.filter(pl.col("event") == ev)
                row["games"], row["events"] = int(sub.height), int(sub["draft_id"].n_unique())
            elif row["in_corpus"]:
                row["games"], row["events"] = mand[(kind, ev)].get("rows"), None
            else:
                try:
                    d = pl.read_csv(p, columns=["draft_id"], schema_overrides={"draft_id": pl.Utf8})
                    row["games"], row["events"] = int(d.height), int(d["draft_id"].n_unique())
                except Exception as e:  # noqa: BLE001
                    row["games"], row["events"], row["error"] = None, None, str(e)[:80]
            rows.append(row)
        res[sc] = rows
    # other draft-type files anywhere in the data root (any set)
    other = {}
    for d in sorted(DATA_ROOT.iterdir()):
        if d.is_dir():
            for p in d.iterdir():
                m = FILE_NAME_RE.search(p.name)
                if m and m.group(3) in ("QuickDraft", "PickTwoDraft", "PickTwoTradDraft"):
                    other.setdefault(m.group(3), set()).add(f"{m.group(1)}:{m.group(2)}")
    return dict(per_set=res, other_event_files={k: sorted(v) for k, v in other.items()})


# =====================================================================================================================
# report
# =====================================================================================================================

def load_scope(out: Path, sc: str, scope: str):
    p = Path(out) / "fits" / f"{sc}__{scope}.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as fh:
        return pickle.load(fh)


def card_labels(out: Path, sc: str) -> dict:
    import polars as pl

    c = pl.read_parquet(home_of(out, sc) / "sets" / sc / "corpus" / "cards.parquet").sort("card_id")
    rar = []
    for r, il, ib in zip(c["rarity"].to_list(), c["is_land"].to_list(), c["is_basic"].to_list()):
        rar.append("basic" if ib else ("land" if il else {"common": "C", "uncommon": "U", "rare": "R", "mythic": "M"}.get(str(r), "other")))
    return dict(rarity=np.array(rar, dtype=object), basic=c["is_basic"].to_numpy().astype(bool), names=c["name"].to_list())


def analyse_set(out: Path, sc: str) -> dict:
    lab = card_labels(out, sc)
    res: dict = {}
    for scope in ("mixed", "PremierDraft", "TradDraft", "PickTwoDraft"):
        d = load_scope(out, sc, scope)
        if d is None:
            continue
        fits = d["fits"]
        a = analyse_scope(fits["time"], [fits[k] for k in sorted(fits) if k.startswith("rand")], lab)
        for t in a["top_cards"]:
            t["name"] = lab["names"][t["card_id"]]
        a.update(n_events=d["n_events"], t_early_end=d["t_early_end"], t_late_start=d["t_late_start"], R=d["R"],
                 n_games_early=int(fits["time"]["n_games"][0]), n_games_late=int(fits["time"]["n_games"][1]),
                 converged=all(all(f["converged"]) for f in fits.values()))
        res[scope] = a
    return res


def pooled(per_set: dict, scope: str = "mixed") -> dict:
    sets = [s for s in per_set if scope in per_set[s]]
    if not sets:
        return {}
    out: dict = dict(sets=sets)
    obs = float(sum(per_set[s][scope]["time"]["chi2"] for s in sets))
    out["pooled_chi2"] = null_compare(obs, pooled_null_sums({s: per_set[s][scope]["null"]["chi2"] for s in sets}))
    nn = {s: per_set[s][scope]["time"]["n"] for s in sets}
    tot = sum(nn.values())
    obs = float(sum(per_set[s][scope]["time"]["share_gt"] * nn[s] for s in sets) / tot)
    out["pooled_share_gt2"] = null_compare(obs, pooled_null_sums({s: [x * nn[s] / tot for x in per_set[s][scope]["null"]["share_gt"]] for s in sets}))
    out["fisher_emp_p"] = fisher_combine([per_set[s][scope]["verdict_chi2"]["p_emp"] for s in sets])
    zs = [per_set[s][scope]["verdict_chi2"]["z_vs_null"] for s in sets]
    zs = [z for z in zs if np.isfinite(z)]
    out["stouffer_z_vs_null"] = float(np.sum(zs) / np.sqrt(len(zs))) if zs else float("nan")
    dz = [per_set[s][scope]["direction"]["corr_z_vs_null"] for s in sets]
    dz = [z for z in dz if np.isfinite(z)]
    out["direction"] = dict(stouffer_z=float(np.sum(dz) / np.sqrt(len(dz))) if dz else float("nan"),
                            mean_corr_time=float(np.mean([per_set[s][scope]["direction"]["time"]["corr"] for s in sets])),
                            mean_corr_null=float(np.mean([per_set[s][scope]["direction"]["null_corr_mean"] for s in sets])),
                            n_positive_excess=int(sum(per_set[s][scope]["direction"]["corr_excess"] > 0 for s in sets)), n=len(sets))
    out["n_sets_p_emp_le_0.1"] = int(sum(per_set[s][scope]["verdict_chi2"]["p_emp"] <= 0.1 for s in sets))
    out["mean_ratio_chi2"] = float(np.mean([per_set[s][scope]["verdict_chi2"]["ratio"] for s in sets]))
    out["sum_ratio_chi2"] = float(sum(per_set[s][scope]["time"]["chi2"] for s in sets) / sum(np.mean(per_set[s][scope]["null"]["chi2"]) for s in sets))
    # drift sizes pooled over sets (card-count weighted mean of the excess variance)
    out["groups"] = {}
    for g in ("ALL", "T1", "T2", "T3", "C", "U", "R", "M", "R+M"):
        rows = [per_set[s][scope]["groups"].get(g) for s in sets]
        rows = [r for r in rows if r]
        n = np.array([r["n"] for r in rows], float)
        if n.sum() == 0:
            continue
        w = n / n.sum()
        ex = float(sum(wi * r["excess"] for wi, r in zip(w, rows)))
        # one-sided 95% upper bound of the excess: se from the spread of the null draws' mean squares (x sqrt(1 + 1/R) for the null mean)
        ex_se = float(np.sqrt(sum((wi * r["msq_null_sd"]) ** 2 for wi, r in zip(w, rows)) * (1 + 1.0 / max(per_set[sets[0]][scope]["R"], 1))))
        up = float(np.sqrt(max(ex + 1.645 * ex_se, 0)))
        out["groups"][g] = dict(n=int(n.sum()), excess=ex, excess_se=ex_se, drift_sd=float(np.sqrt(max(ex, 0))), drift_sd_pp=pp_per_copy(float(np.sqrt(max(ex, 0)))),
                                drift_sd_upper95=up, drift_sd_upper95_pp=pp_per_copy(up),
                                rms_time=float(np.sqrt(sum(wi * r["msq_time"] for wi, r in zip(w, rows)))),
                                rms_null=float(np.sqrt(sum(wi * r["msq_null"] for wi, r in zip(w, rows)))))
    return out


def run_report(out: Path, sets: list[str]) -> str:
    per_set = {}
    for sc in sets:
        r = analyse_set(out, sc)
        if r:
            per_set[sc] = r
    res = dict(sets=list(per_set), per_set=per_set, pooled={}, min_expo=MIN_EXPO, pp_per_logit=PP_PER_LOGIT)
    for scope in ("mixed", "PremierDraft", "TradDraft", "PickTwoDraft"):
        pl_ = pooled(per_set, scope)
        if pl_:
            res["pooled"][scope] = pl_
    sv = Path(out) / "survey.json"
    if sv.exists():
        res["survey"] = json.loads(sv.read_text(encoding="utf-8"))
    tm = Path(out) / "timing.json"
    if tm.exists():
        res["timing"] = json.loads(tm.read_text(encoding="utf-8"))
    (Path(out) / "drift.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    md = markdown(res)
    (Path(out) / "drift.md").write_text(md, encoding="utf-8")
    return md


def _f(x, nd=3):
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def markdown(res: dict) -> str:
    L = ["# Draft-stage card-value drift", "",
         "EARLY / LATE = draft events cut at the median start time (equal numbers of events). Draft stage refitted on each half at a fixed sigma (the full-data "
         "artifact's), cluster-robust SEs. z = (beta_late - beta_early) / sqrt(se_e^2 + se_l^2); chi2 = sum z^2 over cards; null = R random splits (stratified "
         "by type and time) of the same events, same statistic. Drift SD = sqrt(max(0, E[d^2]_time - mean E[d^2]_random)) in logit per copy; pp = 25 x logit "
         "(approximation: slope at 50%%). Cards: non-basic with >= %d copy-games of exposure in each half." % MIN_EXPO, ""]
    for scope in ("mixed", "PremierDraft", "TradDraft", "PickTwoDraft"):
        rows = [(s, res["per_set"][s][scope]) for s in res["sets"] if scope in res["per_set"][s]]
        if not rows:
            continue
        L += [f"## Scope: {scope}", "", "| set | cards | events | chi2 / n (time) | null mean +- sd (chi2/n) | ratio | z vs null | p emp | share abs z>2 time | null | drift SD all (logit) | top tertile logit / pp | R+M logit / pp |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for s, a in rows:
            v, n = a["verdict_chi2"], a["time"]["n"]
            g = a["groups"]
            t3, rm, al = g.get("T3") or {}, g.get("R+M") or {}, g.get("ALL") or {}
            sh = a["verdict_share_gt2"]
            L.append(f"| {s} | {n} | {a['n_events']} | {_f(n and v['obs'] / n, 2)} | {_f(v['null_mean'] / n, 2)} +- {_f(v['null_sd'] / n, 2)} | {_f(v['ratio'], 2)} | {_f(v['z_vs_null'], 1)} | {_f(v['p_emp'], 3)} | "
                     f"{100 * sh['obs']:.1f}% | {100 * sh['null_mean']:.1f}% | {_f(al.get('drift_sd'), 4)} | {_f(t3.get('drift_sd'), 4)} / {_f(t3.get('drift_sd_pp'), 2)} | {_f(rm.get('drift_sd'), 4)} / {_f(rm.get('drift_sd_pp'), 2)} |")
        p = res["pooled"].get(scope)
        if p:
            c, sg = p["pooled_chi2"], p["pooled_share_gt2"]
            L += ["", f"Pooled over {len(p['sets'])} sets ({scope}): sum of chi2 time = {c['obs']:.0f} vs pooled null {c['null_mean']:.0f} +- {c['null_sd']:.0f} (ratio {c['ratio']:.2f}, z {c['z_vs_null']:.1f}, "
                      f"empirical p {c['p_emp']:.3f} with R = {c['R']}); sum-of-ratios mean {p['mean_ratio_chi2']:.2f}; Fisher on per-set empirical p: p = {p['fisher_emp_p']['p']:.2g}; "
                      f"Stouffer z of per-set (chi2 - null mean)/null sd = {p['stouffer_z_vs_null']:.1f}; sets with empirical p <= 0.1: {p['n_sets_p_emp_le_0.1']}/{len(p['sets'])}. "
                      f"Card-weighted share |z|>2: time {100 * sg['obs']:.1f}% vs null {100 * sg['null_mean']:.1f}% (nominal 4.6%).", "",
                  "Pooled drift SD (card-weighted mean of the per-set excess variance):", "", "| group | cards | RMS d time (logit) | RMS d null | excess var +- se | drift SD logit | drift SD pp/copy | 95% upper bound logit / pp |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
            for g, x in p["groups"].items():
                L.append(f"| {g} | {x['n']} | {x['rms_time']:.4f} | {x['rms_null']:.4f} | {x['excess']:.2e} +- {x['excess_se']:.1e} | {x['drift_sd']:.4f} | {x['drift_sd_pp']:.2f} | {x['drift_sd_upper95']:.4f} / {x['drift_sd_upper95_pp']:.2f} |")
            d_ = p["direction"]
            L += ["", f"Direction (d vs beta_early): mean corr time {d_['mean_corr_time']:.3f} vs null {d_['mean_corr_null']:.3f}; excess positive in {d_['n_positive_excess']}/{d_['n']} sets; Stouffer z of per-set z = {d_['stouffer_z']:.1f}."]
        L += [""]
    # group table per set, mixed
    L += ["## Drift SD by card group per set (mixed scope; logit per copy, pp per copy in brackets)", "",
          "| set | ALL | T1 | T2 | T3 | C | U | R | M | R+M | RMS d null ALL / T3 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in res["sets"]:
        a = res["per_set"][s].get("mixed")
        if not a:
            continue
        cells = []
        for g in ("ALL", "T1", "T2", "T3", "C", "U", "R", "M", "R+M"):
            x = a["groups"].get(g)
            cells.append("-" if not x else f"{x['drift_sd']:.3f} ({x['drift_sd_pp']:.2f}) z{_f(x['z_excess'], 1)}")
        L.append(f"| {s} | " + " | ".join(cells) + f" | {_f(a['groups']['ALL']['rms_null'], 4)} / {_f(a['groups']['T3']['rms_null'], 4)} |")
    L += ["", "(z = excess variance over the spread of the null draws' mean squares.)", "",
          "## Direction (mixed): correlation of d with beta_early, time split vs random splits", "",
          "| set | corr time | corr null mean +- sd | z | slope time | slope null | mean d time |", "|---|---:|---:|---:|---:|---:|---:|"]
    for s in res["sets"]:
        a = res["per_set"][s].get("mixed")
        if a:
            d = a["direction"]
            L.append(f"| {s} | {_f(d['time']['corr'], 3)} | {_f(d['null_corr_mean'], 3)} +- {_f(d['null_corr_sd'], 3)} | {_f(d['corr_z_vs_null'], 1)} | {_f(d['time']['slope'], 3)} | {_f(d['null_slope_mean'], 3)} | {_f(d['mean_d_time'], 4)} |")
    L += ["", "## Largest single-card changes (mixed, |z|)", ""]
    for s in res["sets"]:
        a = res["per_set"][s].get("mixed")
        if a:
            L.append(f"- {s}: " + "; ".join(f"{t['name']} z {t['z']:+.1f} (early {t['b_early']:+.3f} -> late {t['b_late']:+.3f})" for t in a["top_cards"][:4]))
    L += [""]
    sv = res.get("survey")
    if sv:
        L += ["## Draft-type game data on disk vs the draft stage today", "", "In corpus = registered and ingested into the corpus (the draft stage uses every such draft-type file). "
              "The folder scan registers only Sealed / TradSealed / PremierDraft / TradDraft by itself, other types only when dropped in by hand.", "",
              "| set | event type | file MB | games | events | in corpus (used by draft stage) | registered |", "|---|---|---:|---:|---:|---|---|"]
        for s, rows in sv["per_set"].items():
            for r in rows:
                if r["is_draft"]:
                    L.append(f"| {s} | {r['event']} | {r['size_mb']} | {r.get('games') or '-'} | {r.get('events') or '-'} | {'yes' if r['in_corpus'] else 'NO'} | {'yes' if r['registered'] else 'no'} |")
        L += ["", "Other draft-type files anywhere in the data folder (kind:set): " + "; ".join(f"{k}: {', '.join(v)}" for k, v in sv["other_event_files"].items()), ""]
    if res.get("timing"):
        L += ["## Runtime", "", json.dumps(res["timing"]), ""]
    return "\n".join(L) + "\n"


# =====================================================================================================================
# driver
# =====================================================================================================================

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("prep", "fit", "survey", "report", "check"):
        p = sub.add_parser(name)
        p.add_argument("out")
        p.add_argument("--sets", nargs="+", default=SETS)
        if name == "fit":
            p.add_argument("-R", type=int, default=10)
            p.add_argument("--scopes", nargs="+", default=None)
    p = sub.add_parser("time-one")
    p.add_argument("set")
    p.add_argument("out")
    a = ap.parse_args(argv)
    if a.cmd == "time-one":
        out = Path(a.out).resolve()
        t0 = time.time()
        setup = make_setup(out, a.set)
        t1 = time.time()
        ev = events_table(setup)
        early, late = time_split(ev["draft_id"].to_list(), ev["start"].to_list())
        sigma = float(setup.meta["prior_strength"]["stages"]["draft"]["sigma"])
        r = fit_subset(setup, late, sigma)
        print(f"set-up {t1 - t0:.0f}s; one half fit (build stage + fit + SE) {time.time() - t1:.0f}s; {r['n_games']} games, {r['n_events']} events, converged {r['converged']}")
        return 0
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if a.cmd == "prep":
        for sc in a.sets:
            prep_home(out, sc)
    elif a.cmd == "fit":
        t0 = time.time()
        for sc in a.sets:
            if not (home_of(out, sc) / "sets" / sc / "corpus" / "games.parquet").exists():
                prep_home(out, sc)
            ts = time.time()
            try:
                fit_set(out, sc, a.R, a.scopes)
            except Exception as e:  # noqa: BLE001  (e.g. a set without draft games)
                print(f"{sc}: skipped ({type(e).__name__}: {str(e)[:120]})", flush=True)
            tp = out / "timing.json"
            t = json.loads(tp.read_text(encoding="utf-8")) if tp.exists() else {}
            t[sc] = round(time.time() - ts)
            tp.write_text(json.dumps(t), encoding="utf-8")
        print(f"fit total {time.time() - t0:.0f}s")
    elif a.cmd == "check":
        for sc in a.sets:
            check_set(out, sc)
    elif a.cmd == "survey":
        r = survey(out, a.sets)
        (out / "survey.json").write_text(json.dumps(r, indent=1, default=str), encoding="utf-8")
        for s, rows in r["per_set"].items():
            for x in rows:
                if x["is_draft"]:
                    print(s, x["event"], x["size_mb"], "MB", x.get("games"), "games", x.get("events"), "events", "in corpus" if x["in_corpus"] else "NOT IN CORPUS", "registered" if x["registered"] else "")
        print(r["other_event_files"])
    elif a.cmd == "report":
        print(run_report(out, a.sets))
    return 0


if __name__ == "__main__":
    sys.exit(main())
