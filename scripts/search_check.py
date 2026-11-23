"""Search check: does the builder's default search miss decks that its own model scores higher?

    python scripts/search_check.py DEVDIR OUT [--pools 200] [--seed 0] [--jobs N] [--verify-default 6]

DEVDIR holds one folder per set (as written by `scripts/deviation_cost.py SET DEVDIR/SET`): the train-only model in SET/home and the
default builder's results for the held-out events in SET/rows_{bo1,bo3}.json (`logit_b`: logit of the default search's best deck
under the full-data fit, draw 0). About `--pools` held-out pools are drawn with a fixed seed, evenly over the set x format cells.
Each is searched again with a much larger search, and the same model's logit of the best deck found is compared:

    default: builder.build(..., families=True, n_decks=1)          (what deviation_cost, and the app's top family, use)
    large  : the same code with the search budget enlarged (module constants and the local search patched in the worker only):
             16 -> 48 configurations screened, 5 -> 20 full land-count configurations, 4 -> 20 local-search starts (n_decks=20
             for every search), 14 -> 60 local-search iterations, 12 -> 48 swaps per iteration, 2-for-2 moves on every start,
             4 -> 8 main-colour seeds and 6 -> 14 seed strategies polished under every bootstrap draw (up to 12 pp behind
             instead of 6, 5 cards apart instead of 8), 40 -> 200 / 8 -> 40 candidates kept per search.
The check reports the share of pools where the large search finds a deck the same model (draw 0) scores higher, and the gap in
win-rate points (percent points): mean over all pools, mean over the pools with a gap, maximum. `--verify-default K` re-runs the
default search on K pools and checks that it reproduces the saved `logit_b` (the search is deterministic).
Writes OUT.json and OUT.md. Never touches the real model; SEALED_LAB_HOME is set per task to DEVDIR/SET/home.

Budget variants (which part of the large budget buys the gain, and at what time cost):

    python scripts/search_check.py DEVDIR OUTDIR --pools-from search_check.json --variants default draw0_large ls cfg ...
    python scripts/search_check.py DEVDIR OUTDIR --summarize                       # variants.json / variants.md from OUTDIR/var_*.json
    python scripts/search_check.py DEVDIR OUTDIR --pools-from search_check.json --inapp default large --inapp-sets MSH SOS ...

`--variants` re-searches every pool once per named variant (`VARIANTS`: a patch spec applied by `patched_search`, in the worker only),
n_jobs=1 per pool and pools in parallel, and stores per pool the draw-0 logit of the top deck, its logit under every bootstrap draw,
the deck, seconds and the number of candidates in OUTDIR/var_NAME.json. `--pools-from` reuses exactly the pools of an earlier run
(its draft_ids and saved logit_default). `--summarize` compares each variant with `default` (gap at draw 0, mean gap over the
replicate draws 1..B, share of draws in which the variant deck scores at least as high). `--inapp` times the full app build
(`builder.build(n_decks=5, families=True)` with all cores) on one bo3 pool per `--inapp-sets` set, the patch also applied in the
worker processes (through the pool initializer).

Since Session 10 the builder itself searches draw 0 with the `d0_fast` budget (builder.N_SCREEN_DRAW0 etc.), so `default` now IS
`d0_fast`; the variants above were measured against what is now `old_default` (results: docs/builder_scoreboard/search_budget.md).
The other variants patch the quick constants, which today only reach the searches under draws 1..B and are kept for the record.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import contextlib
import json
import multiprocessing
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LARGE = dict(
    builder=dict(N_SCREEN=48, N_CONFIG_FULL=20, N_LS_STARTS=20, SWAP_K=48, N_PAIR_STARTS=999),
    families=dict(N_SEED_MAINS=8, MAX_SEEDS=14, SEED_MAX_GAP_PP=12.0, SEED_APART=5, KEEP_DRAW0=200, KEEP_OTHER=40),
    ls_iter=60, n_decks=20,
)

# ---- budget variants ---------------------------------------------------------------------------------------------------------
# A patch spec: builder / families = module constants to set; ls_iter = minimum local-search iterations (Searcher.run's default
# `max_iter` is bound at definition time, so the module constant MAX_LS_ITER is inert and the wrapper raises it instead);
# n_decks = minimum n_decks passed to every run_search (this is how more local-search starts are reached: n_starts = min(N_LS_STARTS,
# n_decks + 2), and 2 starts when n_decks <= 1); pairs_all = 2-for-2 moves in EVERY local search (the polishing under the bootstrap
# draws included, which runs with pairs=False by default); only_draw0 = the builder-side patch (and the extra seed strategies of
# the polishing step) applies only where the search context is draw 0, the searches and polishing under draws 1..B stay default.
# Inert knob found by reading the code: N_PAIR_STARTS=999 alone does nothing for the polishing step (it passes pairs=False
# explicitly); `pairs_all` is what turns pairs on there.
_SEEDS = dict(N_SEED_MAINS=8, MAX_SEEDS=14, SEED_MAX_GAP_PP=12.0, SEED_APART=5)
VARIANTS: dict[str, dict | None] = {
    "default": None,
    # the pre-Session-10 default: draw 0 searched with the same quick budget as the bootstrap draws
    "old_default": dict(builder=dict(N_SCREEN_DRAW0=16, N_CONFIG_FULL_DRAW0=8, SWAP_K_DRAW0=12, MAX_LS_ITER_DRAW0=14)),
    "large": dict(builder=dict(LARGE["builder"]), families=dict(LARGE["families"]), ls_iter=60, n_decks=20, pairs_all=True),
    "draw0_large": dict(builder=dict(LARGE["builder"]), families=dict(_SEEDS, KEEP_DRAW0=200), ls_iter=60, n_decks=20,
                        pairs_all=True, only_draw0=True),
    "cfg": dict(builder=dict(N_SCREEN=48, N_CONFIG_FULL=20)),
    "ls": dict(builder=dict(N_LS_STARTS=20, SWAP_K=48, N_PAIR_STARTS=999), ls_iter=60, n_decks=20),
    "pairs": dict(builder=dict(N_PAIR_STARTS=999), pairs_all=True),
    "seeds": dict(families=dict(LARGE["families"])),
    # Combined variants, built from the single-knob results (the top deck is chosen by draw 0, so everything is draw-0 only; the seeds
    # and pairs knobs bought nothing and are left out; KEEP_DRAW0 only keeps more candidates, never a better top deck):
    "d0_iter": dict(builder=dict(SWAP_K=48), ls_iter=60, only_draw0=True),
    "d0_fast": dict(builder=dict(N_SCREEN=48, N_CONFIG_FULL=20, SWAP_K=48), ls_iter=60, only_draw0=True),
    "d0_mid": dict(builder=dict(N_SCREEN=48, N_CONFIG_FULL=20, N_LS_STARTS=8, SWAP_K=48), ls_iter=60, n_decks=6, only_draw0=True),
    "d0_ls": dict(builder=dict(N_LS_STARTS=20, SWAP_K=48), ls_iter=60, n_decks=20, only_draw0=True),
    # candidate 1 of the Session 12 audit (docs/builder_wide/RULE.md): builder switches set for the whole block (every draw,
    # configuration enumeration included), the search budget unchanged
    "wide": dict(global_builder=dict(WIDE_CONFIGS=True, DEFICIT_BOTH=True)),
    # a creature window per archetype group (docs/archetype_window/RULE.md): the builder switch, search budget unchanged
    "arch_window": dict(global_builder=dict(ARCH_WINDOW=True)),
}
ORDER = ["default", "draw0_large", "ls", "cfg", "seeds", "pairs", "large", "d0_iter", "d0_fast", "d0_mid", "d0_ls", "wide",
         "arch_window"]
_SPEC_DEFAULTS = dict(builder={}, families={}, ls_iter=None, n_decks=None, pairs_all=False, only_draw0=False, global_builder={})


@contextlib.contextmanager
def _set_consts(mod, consts: dict):
    saved = {k: getattr(mod, k) for k in consts}          # AttributeError on a misspelt constant
    for k, v in consts.items():
        setattr(mod, k, v)
    try:
        yield
    finally:
        for k, v in saved.items():
            setattr(mod, k, v)


@contextlib.contextmanager
def patched_search(spec: dict | None):
    """Apply a budget patch spec (see `VARIANTS`) in this process for the duration of the block; everything is restored after.
    `None` (or empty) = the default search."""
    if not spec:
        yield
        return
    from sealed_lab.deck import builder, families

    sp = {**_SPEC_DEFAULTS, **spec}
    if sp["global_builder"] and set(spec) <= {"global_builder"}:
        with _set_consts(builder, sp["global_builder"]):
            yield
        return
    for mod, key in ((builder, "builder"), (families, "families")):
        for k in sp[key]:
            if not hasattr(mod, k):
                raise AttributeError(f"{mod.__name__} has no constant {k}")
    orig_run, orig_search = builder.Searcher.run, builder.run_search
    orig_pick, orig_map = families._pick_seeds, families._Runner.map
    base_cfg_full = builder.N_CONFIG_FULL
    default_seeds = {k: getattr(families, k) for k in ("N_SEED_MAINS", "MAX_SEEDS", "SEED_MAX_GAP_PP", "SEED_APART")}
    state: dict = {"default_seeds": None}

    def active(ctx) -> bool:
        return (not sp["only_draw0"]) or ctx.draw == 0

    def run(self, deck, max_iter=None, deadline=None, pairs=True):          # None: the Searcher's own limit for its draw
        if not active(self.ctx):
            return orig_run(self, deck, max_iter=max_iter, deadline=deadline, pairs=pairs)
        if sp["ls_iter"]:
            max_iter = max(max_iter or 0, sp["ls_iter"])
        with _set_consts(builder, sp["builder"]):                                   # SWAP_K is read inside the moves
            return orig_run(self, deck, max_iter=max_iter, deadline=deadline, pairs=True if sp["pairs_all"] else pairs)

    def run_search(ctx, prep, n_decks=1, *a, **k):
        if not active(ctx):
            return orig_search(ctx, prep, n_decks, *a, **k)
        consts = dict(sp["builder"])
        if sp["n_decks"]:
            if "N_CONFIG_FULL" not in consts:       # forcing n_decks must not by itself change the number of configurations
                consts["N_CONFIG_FULL"] = base_cfg_full if n_decks > 1 else base_cfg_full - 3
            n_decks = max(n_decks, sp["n_decks"])
        with _set_consts(builder, consts):
            return orig_search(ctx, prep, n_decks, *a, **k)

    def pick_seeds(rows0, st):
        seeds = orig_pick(rows0, st)
        with _set_consts(families, default_seeds):
            state["default_seeds"] = {s[0] for s in orig_pick(rows0, st)}
        return seeds

    def runner_map(self, tasks, progress=None):
        """Seeds beyond the default ones are polished under draw 0 only (their other draws would be polishing under draws 1..B)."""
        ds = state["default_seeds"]
        if sp["only_draw0"] and ds is not None:
            new = []
            for t in tasks:
                if t[0] == "polish" and any(sd[0] not in ds for sd in t[2]):
                    dr = tuple(b for b in t[1] if b == 0)
                    if dr:
                        new.append(("polish", dr, t[2]))
                else:
                    new.append(t)
            tasks = new
        return orig_map(self, tasks, progress)

    builder.Searcher.run, builder.run_search = run, run_search
    families._pick_seeds, families._Runner.map = pick_seeds, runner_map
    try:
        with _set_consts(families, sp["families"]), _set_consts(builder, sp["global_builder"]):
            yield
    finally:
        builder.Searcher.run, builder.run_search = orig_run, orig_search
        families._pick_seeds, families._Runner.map = orig_pick, orig_map


def large_search():
    """Enlarge the search budget in this process for the duration of the block (the original `large` run)."""
    return patched_search(VARIANTS["large"])


def _best_logit(model, pool, fmt):
    from sealed_lab.deck import builder

    res = builder.build(model, pool, fmt, n_decks=1, families=True, n_jobs=1)
    best = {c.card_id: c.count for c in res.decks[0].cards if c.card_id is not None}
    return model.logit(best, fmt), res


def _task(task: dict) -> dict:
    os.environ["SEALED_LAB_HOME"] = task["home"]
    from sealed_lab.deck import builder

    model = builder.load_model(task["set"])
    out = dict(set=task["set"], fmt=task["fmt"], draft_id=task["draft_id"], logit_default=task["logit_default"])
    if task["verify"]:
        out["logit_default_rerun"] = _best_logit(model, task["pool"], task["fmt"])[0]
    t0 = time.time()
    with large_search():
        out["logit_large"], res = _best_logit(model, task["pool"], task["fmt"])
    out["seconds"] = time.time() - t0
    out["n_candidates"] = (res.family_meta or {}).get("n_candidates")
    return out


def summarise(rows: list[dict]) -> dict:
    from scipy.special import expit

    def stats_of(sel: list[dict]) -> dict:
        if not sel:
            return {}
        gap = np.array([100 * (expit(r["logit_large"]) - expit(r["logit_default"])) for r in sel])
        better = gap > 1e-4          # 0.0001 pp: numerical noise only
        worse = gap < -1e-4
        return dict(pools=len(sel), share_large_better=float(better.mean()), n_large_better=int(better.sum()),
                    n_large_worse=int(worse.sum()), mean_gap_pp=float(gap.mean()),
                    mean_gap_pp_when_better=float(gap[better].mean()) if better.any() else 0.0,
                    median_gap_pp_when_better=float(np.median(gap[better])) if better.any() else 0.0,
                    max_gap_pp=float(gap.max()), min_gap_pp=float(gap.min()),
                    mean_seconds=float(np.mean([r["seconds"] for r in sel])))
    out = dict(overall=stats_of(rows), cells={})
    for key in sorted({(r["set"], r["fmt"]) for r in rows}):
        out["cells"][f"{key[0]} {key[1]}"] = stats_of([r for r in rows if (r["set"], r["fmt"]) == key])
    for fmt in ("bo1", "bo3"):
        out[fmt] = stats_of([r for r in rows if r["fmt"] == fmt])
    return out


def markdown(summary: dict, n_verified: int, n_mismatch: int, wall: float) -> str:
    def line(name, s):
        return (f"| {name} | {s['pools']} | {s['n_large_better']} ({100 * s['share_large_better']:.0f}%) | {s['n_large_worse']} | "
                f"{s['mean_gap_pp']:.2f} | {s['mean_gap_pp_when_better']:.2f} | {s['max_gap_pp']:.2f} | {s['mean_seconds']:.0f} |")
    L = ["# Search check: default search vs a much larger search (same model, draw 0)", "",
         "| cell | pools | large better | large worse | mean gap pp | mean gap pp where better | max gap pp | s/pool (large) |",
         "|---|---:|---:|---:|---:|---:|---:|---:|"]
    L.append(line("ALL", summary["overall"]))
    for f in ("bo1", "bo3"):
        if summary.get(f):
            L.append(line(f, summary[f]))
    for k, s in summary["cells"].items():
        L.append(line(k, s))
    L += ["", f"Default search re-run on {n_verified} pools: {n_mismatch} did not reproduce the saved logit. Wall time {wall / 60:.0f} min."]
    return "\n".join(L) + "\n"


# =============================================================================================
# budget variants: tasks, comparison with the default search, in-app timing
# =============================================================================================

def load_pool_tasks(dev: Path, pools_from: Path) -> list[dict]:
    """The pools of an earlier run (`pools_from`: its JSON with `pools`: set, fmt, draft_id, logit_default, ...), as worker tasks."""
    import polars as pl

    from sealed_lab import config

    entries = json.loads(Path(pools_from).read_text(encoding="utf-8"))["pools"]
    by_set: dict[str, list[dict]] = {}
    for e in entries:
        by_set.setdefault(e["set"], []).append(e)
    tasks = []
    for sc, es in sorted(by_set.items()):
        home = dev / sc / "home"
        os.environ["SEALED_LAB_HOME"] = str(home)
        pools = pl.read_parquet(config.corpus_dir(sc) / "pools.parquet")
        want: dict[str, list[dict]] = {}
        for e in es:
            want.setdefault(e["draft_id"], []).append(e)
        sub = pools.filter(pl.col("draft_id").is_in(list(want)))
        for did, p in sub.partition_by("draft_id", as_dict=True, include_key=True).items():
            did = did[0] if isinstance(did, tuple) else did
            pool = {int(c): int(n) for c, n in zip(p["card_id"].to_list(), p["n"].to_list())}
            for e in want[did]:
                tasks.append(dict(home=str(home), set=sc, fmt=e["fmt"], draft_id=did, logit_default=e.get("logit_default"),
                                  prior_logit_large=e.get("logit_large"), prior_seconds=float(e.get("seconds") or 0.0), pool=pool))
    missing = len(entries) - len(tasks)
    if missing:
        print(f"WARNING: {missing} pools of {pools_from} were not found in the corpus", flush=True)
    tasks.sort(key=lambda t: (t["set"], t["fmt"], str(t["draft_id"])))
    return tasks


def _deck_of(res) -> dict[int, int]:
    return {c.card_id: c.count for c in res.decks[0].cards if c.card_id is not None}


def _variant_task(task: dict) -> dict:
    os.environ["SEALED_LAB_HOME"] = task["home"]
    from sealed_lab.deck import builder

    model = builder.load_model(task["set"])
    fmt = task["fmt"]
    out = dict(set=task["set"], fmt=fmt, draft_id=task["draft_id"], logit_default=task["logit_default"],
               logit_large_prior=task.get("prior_logit_large"))
    if task.get("info"):                                    # what the pool looks like to the search (not part of the timing)
        ctx = builder.Ctx(model, fmt)
        prep = builder.prepare(ctx, task["pool"], None, None, None)
        out["info"] = dict(pool_cards=int(sum(prep.pool.values())), pool_distinct=len(prep.pool), n_configs=len(prep.configs),
                           n_viable=len(prep.viable), n_main_colors=len({c.key.split("+")[0] for c in prep.configs}),
                           n_land_counts=len(prep.nls))
    t0 = time.time()
    with patched_search(VARIANTS[task["variant"]]):
        res = builder.build(model, task["pool"], fmt, n_decks=1, families=True, n_jobs=1)
    out["seconds"] = time.time() - t0
    deck = _deck_of(res)
    L = np.asarray(model.logit(deck, fmt, draws=True), dtype=float)
    out.update(logit=float(L[0]), L=L.tolist(), deck={str(k): int(v) for k, v in sorted(deck.items())},
               n_candidates=(res.family_meta or {}).get("n_candidates"), n_families=(res.family_meta or {}).get("n_families"))
    return out


def run_variant(name: str, tasks: list[dict], jobs: int, outdir: Path, force: bool = False) -> Path:
    path = outdir / f"var_{name}.json"
    if path.exists() and not force:
        print(f"variant {name}: {path.name} exists, skipped", flush=True)
        return path
    ts = [dict(t, variant=name, info=(name == "default")) for t in tasks]
    ts.sort(key=lambda t: (-t["prior_seconds"], t["set"], t["fmt"], str(t["draft_id"])))        # longest first: better packing
    print(f"variant {name}: {len(ts)} pools, spec {VARIANTS[name]}", flush=True)
    t0 = time.time()
    rows = []
    with cf.ProcessPoolExecutor(max_workers=jobs, mp_context=multiprocessing.get_context("spawn")) as ex:
        for k, r in enumerate(ex.map(_variant_task, ts, chunksize=1)):
            rows.append(r)
            if (k + 1) % 20 == 0:
                print(f"  {name}: {k + 1}/{len(ts)} done, {time.time() - t0:.0f}s", flush=True)
    wall = time.time() - t0
    rows.sort(key=lambda r: (r["set"], r["fmt"], str(r["draft_id"])))
    outdir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(variant=name, spec=VARIANTS[name], wall_seconds=wall, jobs=jobs, rows=rows), indent=1), encoding="utf-8")
    print(f"variant {name}: done in {wall / 60:.1f} min, mean {np.mean([r['seconds'] for r in rows]):.0f} s/pool", flush=True)
    return path


def _key(r: dict) -> tuple:
    return (r["set"], r["fmt"], str(r["draft_id"]))


def pool_gaps(rows: list[dict], base_rows: list[dict]) -> list[dict]:
    """Per pool (those in both): the variant's top deck minus the default's top deck, in win-rate points: at draw 0, averaged over the
    replicate draws 1..B, and the share of replicate draws in which the variant deck scores at least as high."""
    from scipy.special import expit

    base = {_key(r): r for r in base_rows}
    out = []
    for r in rows:
        b = base.get(_key(r))
        if b is None:
            continue
        lv, lb = np.asarray(r["L"], dtype=float), np.asarray(b["L"], dtype=float)
        rep = slice(1, None) if len(lv) > 1 else slice(0, None)
        out.append(dict(set=r["set"], fmt=r["fmt"], key=_key(r), gap0=float(100 * (expit(lv[0]) - expit(lb[0]))),
                        gap_rep=float((100 * (expit(lv[rep]) - expit(lb[rep]))).mean()),
                        share_ge=float((lv[rep] >= lb[rep] - 1e-9).mean()), same=bool(r["deck"] == b["deck"]),
                        seconds=float(r["seconds"]), seconds_base=float(b["seconds"]), n_candidates=r.get("n_candidates")))
    return out


def _se(x: np.ndarray) -> float:
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0


def aggregate(per: list[dict], large_mean_gap: float | None = None) -> dict:
    """Summary over pools of `pool_gaps` rows. `large_mean_gap`: mean draw-0 gap of the large variant on the same pools."""
    if not per:
        return {}
    g0 = np.array([p["gap0"] for p in per])
    gr = np.array([p["gap_rep"] for p in per])
    sge = np.array([p["share_ge"] for p in per])
    diff = np.array([not p["same"] for p in per])
    sec = np.array([p["seconds"] for p in per])
    secb = np.array([p["seconds_base"] for p in per])
    better = g0 > 1e-4
    out = dict(pools=len(per), n_better=int(better.sum()), share_better=float(better.mean()), n_worse=int((g0 < -1e-4).sum()),
               n_same_deck=int((~diff).sum()), mean_gap_pp=float(g0.mean()), se_gap_pp=_se(g0),
               mean_gap_pp_when_better=float(g0[better].mean()) if better.any() else 0.0,
               mean_gap_rep_pp=float(gr.mean()), se_gap_rep_pp=_se(gr), share_draws_ge=float(sge.mean()),
               share_draws_ge_diff=float(sge[diff].mean()) if diff.any() else None,
               rep_over_draw0=float(gr.mean() / g0.mean()) if abs(g0.mean()) > 1e-12 else None,
               mean_seconds=float(sec.mean()), mean_seconds_base=float(secb.mean()),
               time_ratio=float(sec.mean() / secb.mean()) if secb.mean() > 0 else None)
    out["captured"] = float(g0.mean() / large_mean_gap) if large_mean_gap and abs(large_mean_gap) > 1e-12 else None
    return out


def summarize_variants(data: dict[str, list[dict]], base: str = "default", breakdown: list[str] | None = None) -> dict:
    """`data`: variant name -> rows (from var_NAME.json). Everything is relative to `base`; `large` gives the share captured."""
    if base not in data:
        raise ValueError(f"the baseline variant {base!r} has no results")
    out: dict = dict(base=base, variants={}, per_set={}, checks={})
    large_gap = {q["key"]: q["gap0"] for q in pool_gaps(data["large"], data[base])} if "large" in data else {}
    for name, rows in data.items():
        per = pool_gaps(rows, data[base])
        lg = float(np.mean([large_gap[p["key"]] for p in per if p["key"] in large_gap])) if large_gap and per else None
        out["variants"][name] = aggregate(per, lg)
        if name in (breakdown or ["large"]):
            ps = {}
            for s in sorted({p["set"] for p in per}):
                sub = [p for p in per if p["set"] == s]
                ps[s] = aggregate(sub, float(np.mean([large_gap[p["key"]] for p in sub])) if large_gap else None)
            out["per_set"][name] = ps
    out["checks"]["default_vs_saved_max_abs_logit_diff"] = float(max(abs(r["logit"] - r["logit_default"]) for r in data[base]))
    if "large" in data:
        prior = [abs(r["logit"] - r["logit_large_prior"]) for r in data["large"] if r.get("logit_large_prior") is not None]
        out["checks"]["large_vs_prior_large_max_abs_logit_diff"] = float(max(prior)) if prior else None
        out["checks"]["large_vs_prior_large_n_differ"] = int(sum(d > 1e-6 for d in prior))
    out["set_structure"] = set_structure(data)
    return out


def set_structure(data: dict[str, list[dict]]) -> dict:
    """Per set (pools of both formats): what the search faces (default run) and what the large search makes of it."""
    drows = data.get("default", [])
    lrows = {_key(r): r for r in data.get("large", [])}
    out = {}
    for s in sorted({r["set"] for r in drows}):
        d = [r for r in drows if r["set"] == s]
        info = [r["info"] for r in d if r.get("info")]
        lg = [lrows[_key(r)] for r in d if _key(r) in lrows]
        row = dict(pools=len(d), n_candidates_default=float(np.mean([r["n_candidates"] for r in d])),
                   seconds_default=float(np.mean([r["seconds"] for r in d])))
        for k in ("pool_cards", "pool_distinct", "n_configs", "n_viable", "n_main_colors", "n_land_counts"):
            row[k] = float(np.mean([i[k] for i in info])) if info else None
        if lg:
            row["n_candidates_large"] = float(np.mean([r["n_candidates"] for r in lg]))
            row["seconds_large"] = float(np.mean([r["seconds"] for r in lg]))
        out[s] = row
    return out


def variants_markdown(summ: dict, order: list[str]) -> str:
    def f(x, nd=2, pct=False):
        if x is None:
            return "-"
        return f"{100 * x:.0f}%" if pct else f"{x:.{nd}f}"
    names = [n for n in order if n in summ["variants"]] + sorted(n for n in summ["variants"] if n not in order)
    L = ["# Search budget variants (same model, same pools, top deck = highest draw-0 logit)", "",
         f"Gaps are the variant's top deck minus the `{summ['base']}` top deck in win-rate percentage points. The replicate columns evaluate "
         "both decks under the bootstrap draws 1..B; a gap that mostly vanishes there was found by chasing draw-0 noise.", "",
         "| variant | pools | share better | mean gap pp (draw 0) | share of large gap | replicate-mean gap pp | share of draws >= default | "
         "core-s/pool | x default | in-app wall s |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    inapp = summ.get("inapp", {}).get("mean_by_variant", {})
    for n in names:
        v = summ["variants"][n]
        if not v:
            continue
        L.append(f"| {n} | {v['pools']} | {f(v['share_better'], pct=True)} | {f(v['mean_gap_pp'])} +- {f(v['se_gap_pp'])} | "
                 f"{f(v['captured'], pct=True)} | {f(v['mean_gap_rep_pp'])} +- {f(v['se_gap_rep_pp'])} | {f(v['share_draws_ge'], pct=True)} | "
                 f"{v['mean_seconds']:.0f} | {f(v['time_ratio'], 1)} | {f(inapp.get(n), 0) if n in inapp else '-'} |")
    ck = summ["checks"]
    L += ["", f"Checks: default search vs the saved logit_default, max abs diff {ck['default_vs_saved_max_abs_logit_diff']:.2e}"
          + (f"; large search vs the earlier large run, max abs diff {ck['large_vs_prior_large_max_abs_logit_diff']:.2e} "
             f"({ck['large_vs_prior_large_n_differ']} pools differ)." if ck.get("large_vs_prior_large_max_abs_logit_diff") is not None else ".")]
    for n, ps in summ["per_set"].items():
        L += ["", f"## Per set: {n}", "", "| set | pools | share better | mean gap pp | replicate-mean gap pp | share of draws >= default | s/pool | x default |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for s, v in ps.items():
            L.append(f"| {s} | {v['pools']} | {f(v['share_better'], pct=True)} | {f(v['mean_gap_pp'])} | {f(v['mean_gap_rep_pp'])} | "
                     f"{f(v['share_draws_ge'], pct=True)} | {v['mean_seconds']:.0f} | {f(v['time_ratio'], 1)} |")
    st = summ.get("set_structure") or {}
    if st:
        L += ["", "## What the search faces, per set (default run; means over the set's pools)", "",
              "| set | pool cards | distinct | configurations | viable | main-colour combos | land counts | candidates default | candidates large | "
              "s default | s large |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for s, v in st.items():
            L.append(f"| {s} | {f(v['pool_cards'], 0)} | {f(v['pool_distinct'], 0)} | {f(v['n_configs'], 1)} | {f(v['n_viable'], 1)} | "
                     f"{f(v['n_main_colors'], 1)} | {f(v['n_land_counts'], 1)} | {f(v['n_candidates_default'], 0)} | "
                     f"{f(v.get('n_candidates_large'), 0)} | {f(v['seconds_default'], 0)} | {f(v.get('seconds_large'), 0)} |")
    if summ.get("inapp"):
        ia = summ["inapp"]
        L += ["", "## In-app timing: builder.build(n_decks=5, families=True), all cores, cold worker pool, wall seconds", "",
              "| pool | " + " | ".join(ia["variants"]) + " |", "|---|" + "---:|" * len(ia["variants"])]
        for p in ia["pools"]:
            L.append(f"| {p} | " + " | ".join(f(ia["wall"].get(v, {}).get(p), 1) for v in ia["variants"]) + " |")
        L.append("| mean | " + " | ".join(f(ia["mean_by_variant"].get(v), 1) for v in ia["variants"]) + " |")
    return "\n".join(L) + "\n"


def load_variant_files(outdir: Path) -> dict[str, list[dict]]:
    data = {}
    for p in sorted(outdir.glob("var_*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        data[d["variant"]] = d["rows"]
    return data


def summarize_dir(outdir: Path, breakdown: list[str] | None = None) -> str:
    data = load_variant_files(outdir)
    summ = summarize_variants(data, breakdown=breakdown)
    ip = outdir / "inapp.json"
    if ip.exists():
        recs = json.loads(ip.read_text(encoding="utf-8"))["runs"]
        present = {r["variant"] for r in recs}
        variants = [v for v in ORDER + sorted(present) if v in present]
        variants = list(dict.fromkeys(variants))
        pools = sorted({f"{r['set']} {r['fmt']}" for r in recs})
        wall = {v: {f"{r['set']} {r['fmt']}": r["wall"] for r in recs if r["variant"] == v} for v in variants}
        summ["inapp"] = dict(variants=variants, pools=pools, wall=wall,
                             mean_by_variant={v: float(np.mean(list(wall[v].values()))) for v in variants})
    (outdir / "variants.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    md = variants_markdown(summ, ORDER)
    (outdir / "variants.md").write_text(md, encoding="utf-8")
    return md


# ---- in-app timing -------------------------------------------------------------------------------------------------------------

_WORKER_CM = None


def _init_worker(name: str) -> None:
    """Pool initializer for the app's persistent worker processes: what the app installs, plus the variant's patch for the worker's life."""
    global _WORKER_CM
    from sealed_lab.procguard import watch_parent

    watch_parent()
    _WORKER_CM = patched_search(VARIANTS[name])
    _WORKER_CM.__enter__()


def run_inapp(tasks: list[dict], names: list[str], sets: list[str], outdir: Path) -> None:
    """Time `builder.build(model, pool, "bo3", n_decks=5, families=True)` with the default n_jobs, one pool at a time, per variant. Each
    run starts from a stopped worker pool (workers start and load the model inside the timing, as for the app's first build)."""
    import functools

    from sealed_lab.deck import builder, families

    path = outdir / "inapp.json"
    runs = json.loads(path.read_text(encoding="utf-8"))["runs"] if path.exists() else []
    chosen = []
    for s in sets:
        c = [t for t in tasks if t["set"] == s.upper() and t["fmt"] == "bo3"]
        if c:
            chosen.append(c[0])
    orig_wp = families.watch_parent
    try:
        for name in names:
            for t in chosen:
                if any(r["variant"] == name and r["set"] == t["set"] and r["fmt"] == t["fmt"] for r in runs):
                    continue
                families.shutdown_pool(kill=True)
                time.sleep(2.0)
                os.environ["SEALED_LAB_HOME"] = t["home"]
                model = builder.load_model(t["set"])
                families.watch_parent = functools.partial(_init_worker, name)
                t0 = time.time()
                with patched_search(VARIANTS[name]):
                    res = builder.build(model, t["pool"], t["fmt"], n_decks=5, families=True)
                wall = time.time() - t0
                meta = res.family_meta or {}
                runs.append(dict(variant=name, set=t["set"], fmt=t["fmt"], draft_id=t["draft_id"], wall=wall,
                                 n_candidates=meta.get("n_candidates"), n_families=meta.get("n_families"), jobs=meta.get("jobs"),
                                 logit=float(model.logit(_deck_of(res), t["fmt"]))))
                print(f"in-app {name} {t['set']} {t['fmt']}: {wall:.1f} s (jobs {meta.get('jobs')}, {meta.get('n_candidates')} candidates)", flush=True)
                outdir.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(dict(runs=runs), indent=1), encoding="utf-8")
    finally:
        families.watch_parent = orig_wp
        families.shutdown_pool(kill=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("devdir")
    ap.add_argument("out", help="output path without extension: writes OUT.json and OUT.md")
    ap.add_argument("--pools", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--verify-default", type=int, default=6)
    ap.add_argument("--sets", nargs="*", default=None)
    ap.add_argument("--variants", nargs="+", default=None, choices=sorted(VARIANTS), help="budget variants to run (OUT is then a directory)")
    ap.add_argument("--pools-from", default=None, help="JSON of an earlier run: re-use exactly its pools")
    ap.add_argument("--force", action="store_true", help="re-run variants whose var_NAME.json exists")
    ap.add_argument("--summarize", action="store_true", help="write variants.json / variants.md from OUT/var_*.json (and inapp.json)")
    ap.add_argument("--breakdown", nargs="*", default=["large"], help="variants that get a per-set table")
    ap.add_argument("--inapp", nargs="+", default=None, choices=sorted(VARIANTS), help="time the app's full build for these variants")
    ap.add_argument("--inapp-sets", nargs="+", default=["MSH", "SOS", "TDM", "FIN", "OM1", "HOB"])
    args = ap.parse_args(argv)

    if args.variants or args.inapp or args.summarize:
        outdir = Path(args.out)
        if args.variants or args.inapp:
            if not args.pools_from:
                ap.error("--variants / --inapp need --pools-from")
            tasks = load_pool_tasks(Path(args.devdir).resolve(), Path(args.pools_from))
            if args.sets:
                tasks = [t for t in tasks if t["set"] in [x.upper() for x in args.sets]]
            t0 = time.time()
            for name in args.variants or []:
                run_variant(name, tasks, args.jobs or os.cpu_count() or 1, outdir, args.force)
            if args.inapp:
                run_inapp(tasks, args.inapp, args.inapp_sets, outdir)
            print(f"wall {(time.time() - t0) / 60:.1f} min", flush=True)
        if args.summarize or (args.variants and "default" in load_variant_files(outdir)):
            print(summarize_dir(outdir, args.breakdown), flush=True)
        return 0

    import polars as pl

    from sealed_lab import config

    dev = Path(args.devdir).resolve()
    cells = []
    for sd in sorted(p for p in dev.iterdir() if (p / "home").is_dir()):
        if args.sets and sd.name.upper() not in [s.upper() for s in args.sets]:
            continue
        for fmt in ("bo3", "bo1"):
            rp = sd / f"rows_{fmt}.json"
            if rp.exists():
                cells.append((sd.name.upper(), fmt, json.loads(rp.read_text(encoding="utf-8"))))
    if not cells:
        print("no rows_*.json found")
        return 1
    rng = np.random.default_rng(args.seed)
    per = int(np.ceil(args.pools / len(cells)))
    tasks = []
    for sc, fmt, rows in cells:
        home = dev / sc / "home"
        os.environ["SEALED_LAB_HOME"] = str(home)
        pools = pl.read_parquet(config.corpus_dir(sc) / "pools.parquet")
        pick = rng.choice(len(rows), min(per, len(rows)), replace=False)
        want = {rows[i]["draft_id"]: rows[i]["logit_b"] for i in sorted(pick.tolist())}
        sub = pools.filter(pl.col("draft_id").is_in(list(want)))
        for did, p in sub.partition_by("draft_id", as_dict=True, include_key=True).items():
            did = did[0] if isinstance(did, tuple) else did
            tasks.append(dict(home=str(home), set=sc, fmt=fmt, draft_id=did, logit_default=want[did], verify=False,
                              pool={int(c): int(n) for c, n in zip(p["card_id"].to_list(), p["n"].to_list())}))
    tasks.sort(key=lambda t: (t["set"], t["fmt"], str(t["draft_id"])))
    for t in tasks[:: max(1, len(tasks) // max(args.verify_default, 1))][: args.verify_default]:
        t["verify"] = True
    print(f"{len(tasks)} pools over {len(cells)} cells; large-search budget {LARGE}", flush=True)
    t0 = time.time()
    res = []
    ctx = multiprocessing.get_context("spawn")
    with cf.ProcessPoolExecutor(max_workers=args.jobs or os.cpu_count(), mp_context=ctx) as ex:
        for k, r in enumerate(ex.map(_task, tasks, chunksize=2)):
            res.append(r)
            if (k + 1) % 20 == 0:
                print(f"  {k + 1}/{len(tasks)} done, {time.time() - t0:.0f}s", flush=True)
    wall = time.time() - t0
    ver = [r for r in res if "logit_default_rerun" in r]
    mism = [r for r in ver if abs(r["logit_default_rerun"] - r["logit_default"]) > 1e-6]
    summary = summarise(res)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(dict(summary=summary, pools=res, large_budget=LARGE, wall_seconds=wall,
                                                       verified=len(ver), verify_mismatch=len(mism)), indent=1), encoding="utf-8")
    md = markdown(summary, len(ver), len(mism), wall)
    out.with_suffix(".md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
