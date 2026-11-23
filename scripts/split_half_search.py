"""Split-half cross-evaluation of the deck-search gain: is it real, or the search exploiting estimation noise in its own model?

    python scripts/split_half_search.py run     SRC DEVDIR POOLS_JSON OUT [--n-boot 4] [--jobs N] [--max-hours 6]
    python scripts/split_half_search.py fit     SET HALF SRC OUT [--n-boot 4] [--jobs N]        (HALF = A or B; one train-only half model)
    python scripts/split_half_search.py search  SRC DEVDIR POOLS_JSON OUT [--jobs N]            (all fits must exist)
    python scripts/split_half_search.py report  OUT                                              (split_half.json / split_half.md)

More variants on the same fits and pools (rows merged by pool):
    python scripts/split_half_search.py search  SRC DEVDIR POOLS_JSON OUT --variants draw0_large --rows search_rows_d0l.jsonl
    python scripts/split_half_search.py report  OUT --extra-rows search_rows_d0l.jsonl --compare draw0_large d0_fast --base default
    python scripts/split_half_search.py report  OUT --extra-rows search_rows_d0l.jsonl --compare draw0_large --base d0_fast --name vs_d0_fast
A builder switch against the shipped builder, with the decision lines of its RULE.md (docs/archetype_window/RULE.md):
    python scripts/split_half_search.py search  SRC DEVDIR POOLS_R_AND_T.json OUT --variants default arch_window
    python scripts/split_half_search.py report  OUT --rule archetype_window --compare arch_window --base default --pools-r docs/builder_scoreboard/search_check.json --pools-t targeted_pools.json
(`--variants` takes any key of search_check.VARIANTS; POOLS_R_AND_T.json = the R and T pools in one file's `pools` list.)
`default` in search_rows.jsonl is the search as it was BEFORE Session 10b; since then the builder's own default equals `d0_fast`, so
new searches must not re-run `default` (it would be d0_fast) - compare against the stored rows.

Design. For each set the TRAINING events (all formats; the events that started before the hold-out cut, exactly the events of
`scripts/deviation_cost.py`'s train-only model) are split at random BY EVENT into two halves A and B, stratified by format and by
time (events of one format sorted by start, taken in consecutive pairs, one of each pair to each half). Model A is fitted on half A
and model B on half B (same pipeline call as deviation_cost's train-only fit, default options, `recency="off"`), each in its own
scratch home (a copy of SRC's set: the corpus is the full one, as in the deviation-cost run, so the builder's land / creature
support is the same for both halves). On the pools of an earlier `search_check` run, the default search and the budget variants
`d0_fast` / `d0_mid` (scripts/search_check.py VARIANTS, applied by its `patched_search`) are run with model A and with model B; the
top deck of each (the builder's choice) is scored under BOTH models at draw 0. Per pool and searching model ("pool-direction"):

    gap_in  = variant deck minus default deck, scored by the model that searched (pp of win rate)  - what the search sees
    gap_out = the same two decks scored by the OTHER half's model                                  - independent data

If the larger search only chased noise of its own half-data model, gap_out would be ~0 while gap_in > 0; if it finds decks that are
better in a data-independent sense, gap_out ~ gap_in. Confidence intervals are cluster bootstraps over pools (both directions of a
pool in one cluster). Sanity scale: the default search's deck of model A against that of model B, and how much each model prefers
its own (`self_preference`).

Forcing model options (e.g. the matchups acceptance check, scripts/matchups_accept.py): `fit` / `run` take `--model-opt KEY=VALUE`
(repeatable; keys of `FORCEABLE_OPTS`, merged over `MODEL_OPTS` and recorded in the fit's json) so that half models with, say,
`matchups=on` land in a separate OUT tree on the SAME event split (the split depends only on SRC and --seed). `search` takes
`--only SET[:fmt] ...` (restrict the pools) and `--homes-from DIR` (use the half homes of another OUT tree, e.g. to re-search the
current models with today's code into this OUT's rows file).

Everything is written under OUT (scratch); SEALED_LAB_HOME is set per task to a scratch home, the real app home is never touched.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import contextlib
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import time
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

SETS = ["ECL", "EOE", "FIN", "FRA", "HOB", "MSH", "OM1", "SOS", "TDM", "TLA", "TMT"]
HALVES = ("A", "B")
VARIANT_NAMES = ("default", "d0_fast", "d0_mid")
COMPARED = ("d0_fast", "d0_mid")
OTHER = {"A": "B", "B": "A"}
EPS_PP = 1e-4          # 0.0001 pp: numerical noise only (as in search_check)


# =====================================================================================================================
# pure parts: the event split and the summary arithmetic (tested without any model fit)
# =====================================================================================================================

def split_events(ids, starts, fmts, seed: int = 0) -> tuple[list, list]:
    """Random split of events into two halves A and B, stratified by format and by time. Within each format the events are sorted
    by start time and taken in consecutive pairs; a fair coin decides which of the two goes to A. (An odd event out also goes by
    coin.) Every event goes whole to one half, the halves differ by at most one event per format and both cover the whole period."""
    ids, starts, fmts = list(ids), list(starts), list(fmts)
    if not (len(ids) == len(starts) == len(fmts)):
        raise ValueError("ids, starts and fmts must have the same length")
    if len(set(ids)) != len(ids):
        raise ValueError("event ids must be unique")
    rng = np.random.default_rng(seed)
    a: list = []
    b: list = []
    for f in sorted(set(fmts), key=str):
        ev = sorted((i for i in range(len(ids)) if fmts[i] == f), key=lambda i: (starts[i], str(ids[i])))
        flips = rng.integers(0, 2, size=(len(ev) + 1) // 2)
        for k in range(0, len(ev), 2):
            first, second = ev[k], (ev[k + 1] if k + 1 < len(ev) else None)
            if flips[k // 2]:
                first, second = second, first
            if first is not None:
                a.append(ids[first])
            if second is not None:
                b.append(ids[second])
    return a, b


def set_seed(seed: int, set_code: str) -> int:
    return int(zlib.crc32(f"{seed}:{set_code}".encode()) % (2 ** 31))


FORCEABLE_OPTS = ("attributes", "crossset", "synergy", "archcond", "matchups", "castability", "extra_draft", "manasim", "theta_sigma", "oppflags")
OPT_VALUES = {"manasim": ("auto", "v2", "v1"), "theta_sigma": ("rule", "cv")}


def parse_model_opts(items) -> dict:
    """['matchups=on', ...] -> {'matchups': 'on'}; only the component switches of `pipeline.calculate` may be forced (the others are
    the fixed `MODEL_OPTS` that define the split-half setup)."""
    out: dict = {}
    for it in items or ():
        if "=" not in it:
            raise ValueError(f"model option must be KEY=VALUE, got {it!r}")
        k, v = it.split("=", 1)
        k, v = k.strip(), v.strip()
        if k not in FORCEABLE_OPTS:
            raise ValueError(f"cannot force {k!r}; allowed: {', '.join(FORCEABLE_OPTS)}")
        if v not in OPT_VALUES.get(k, ("auto", "on", "off")):
            raise ValueError(f"{k} must be one of {OPT_VALUES.get(k, ('auto', 'on', 'off'))}, got {v!r}")
        out[k] = v
    return out


def select_tasks(tasks: list[dict], only) -> list[dict]:
    """Keep the tasks whose set (or 'SET:fmt') is listed in `only`; `only` empty / None keeps all."""
    if not only:
        return list(tasks)
    sets, cells = set(), set()
    for o in only:
        if ":" in o:
            sc, f = o.split(":", 1)
            cells.add((sc.upper(), f.lower()))
        else:
            sets.add(o.upper())
    return [t for t in tasks if t["set"] in sets or (t["set"], t["fmt"]) in cells]


def cluster_boot(num, den, clusters, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Ratio of sums sum(num)/sum(den) (= the mean when den == 1) with a 95% percentile interval from resampling whole clusters."""
    num, den = np.asarray(num, dtype=float), np.asarray(den, dtype=float)
    _, inv = np.unique(np.asarray([str(c) for c in clusters]), return_inverse=True)
    G = int(inv.max()) + 1 if len(inv) else 0
    if G == 0:
        return float("nan"), float("nan"), float("nan")
    sn, sd = np.bincount(inv, weights=num, minlength=G), np.bincount(inv, weights=den, minlength=G)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, G, size=(n_boot, G))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = sn[idx].sum(1) / sd[idx].sum(1)
        pt = sn.sum() / sd.sum() if sd.sum() != 0 else float("nan")
    r = r[np.isfinite(r)]
    if not len(r):
        return float(pt), float("nan"), float("nan")
    return float(pt), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def pool_direction_records(rows: list[dict], variants=COMPARED, base: str = "default") -> list[dict]:
    """rows: one per pool, {set, fmt, draft_id, dirs: {"A"|"B": {variant: {ls, lo, deck, seconds}}}} (ls / lo = draw-0 logit of that
    variant's top deck under the searching / the other half's model). One record per (pool, direction, variant)."""
    from scipy.special import expit

    out = []
    for r in rows:
        for d, byv in r["dirs"].items():
            b0 = byv.get(base)
            if b0 is None:
                continue
            for v in variants:
                x = byv.get(v)
                if x is None:
                    continue
                # The two half models have different absolute levels (their intercepts differ by up to ~0.8 logit on a small set), so
                # a gap in points is only comparable across models if both logit differences are converted at one level: that of the
                # default deck under the SEARCHING model.
                l0 = b0["ls"]
                dl_in, dl_out = x["ls"] - b0["ls"], x["lo"] - b0["lo"]
                out.append(dict(set=r["set"], fmt=r["fmt"], pool=f'{r["set"]}|{r["fmt"]}|{r["draft_id"]}', dir=d, variant=v,
                                dl_in=float(dl_in), dl_out=float(dl_out),
                                gap_in=float(100 * (expit(l0 + dl_in) - expit(l0))),
                                gap_out=float(100 * (expit(l0 + dl_out) - expit(l0))),
                                same_deck=bool(x["deck"] == b0["deck"]), seconds=float(x["seconds"]),
                                seconds_base=float(b0["seconds"])))
    return out


def summarise_records(recs: list[dict], n_boot: int = 2000, seed: int = 0) -> dict:
    """Summary of the records of ONE variant (any selection of sets / formats): means, cluster-bootstrap CIs, ratio, shares."""
    if not recs:
        return {}
    gi = np.array([r["gap_in"] for r in recs])
    go = np.array([r["gap_out"] for r in recs])
    cl = [r["pool"] for r in recs]
    ones = np.ones(len(recs))
    m_in = cluster_boot(gi, ones, cl, n_boot, seed)
    m_out = cluster_boot(go, ones, cl, n_boot, seed)
    ratio = cluster_boot(go, gi, cl, n_boot, seed)                       # sum(gap_out) / sum(gap_in), pools resampled together
    di = np.array([r["dl_in"] for r in recs])
    do = np.array([r["dl_out"] for r in recs])
    ratio_l = cluster_boot(do, di, cl, n_boot, seed)                     # the same in logit units (no level conversion)
    return dict(n=len(recs), n_pools=len(set(cl)), mean_gap_in=m_in[0], gap_in_ci=[m_in[1], m_in[2]],
                mean_gap_out=m_out[0], gap_out_ci=[m_out[1], m_out[2]], ratio=ratio[0], ratio_ci=[ratio[1], ratio[2]],
                mean_dl_in=float(di.mean()), mean_dl_out=float(do.mean()), ratio_logit=ratio_l[0], ratio_logit_ci=[ratio_l[1], ratio_l[2]],
                share_out_pos=float((go > EPS_PP).mean()), share_out_neg=float((go < -EPS_PP).mean()),
                share_in_pos=float((gi > EPS_PP).mean()), share_same_deck=float(np.mean([r["same_deck"] for r in recs])),
                mean_seconds=float(np.mean([r["seconds"] for r in recs])),
                mean_seconds_default=float(np.mean([r["seconds_base"] for r in recs])))


def self_preference(rows: list[dict]) -> list[dict]:
    """Default search against default search: per pool, deck A (found by model A) vs deck B (found by model B). For each model the
    mean win-rate points by which it prefers its own deck, from the pool's two directions."""
    from scipy.special import expit

    out = []
    for r in rows:
        a, b = r["dirs"].get("A", {}).get("default"), r["dirs"].get("B", {}).get("default")
        if a is None or b is None:
            continue
        # a["ls"]: deck A under A; a["lo"]: deck A under B; b["ls"]: deck B under B; b["lo"]: deck B under A
        pref_a = float(100 * (expit(a["ls"]) - expit(b["lo"])))          # model A: own deck minus the other's deck
        pref_b = float(100 * (expit(b["ls"]) - expit(a["lo"])))
        out.append(dict(set=r["set"], fmt=r["fmt"], pool=f'{r["set"]}|{r["fmt"]}|{r["draft_id"]}', pref_a=pref_a, pref_b=pref_b,
                        pref=0.5 * (pref_a + pref_b), same_deck=bool(a["deck"] == b["deck"])))
    return out


def summarise_pref(recs: list[dict], n_boot: int = 2000, seed: int = 0) -> dict:
    if not recs:
        return {}
    p = np.array([r["pref"] for r in recs])
    m = cluster_boot(p, np.ones(len(p)), [r["pool"] for r in recs], n_boot, seed)
    return dict(n_pools=len(recs), mean_self_preference_pp=m[0], ci=[m[1], m[2]],
                share_same_deck=float(np.mean([r["same_deck"] for r in recs])),
                mean_pref_a=float(np.mean([r["pref_a"] for r in recs])), mean_pref_b=float(np.mean([r["pref_b"] for r in recs])))


def full_summary(rows: list[dict], n_boot: int = 2000, seed: int = 0, compared=COMPARED, base: str = "default") -> dict:
    out: dict = dict(n_pools=len(rows), base=base, variants={}, default_vs_default={})
    recs_all = pool_direction_records(rows, compared, base)
    for v in compared:
        rv = [r for r in recs_all if r["variant"] == v]
        blk: dict = dict(overall=summarise_records(rv, n_boot, seed))
        for f in ("bo1", "bo3"):
            blk[f] = summarise_records([r for r in rv if r["fmt"] == f], n_boot, seed)
        blk["per_set"] = {s: summarise_records([r for r in rv if r["set"] == s], n_boot, seed)
                          for s in sorted({r["set"] for r in rv})}
        for d in HALVES:
            blk[f"dir_{d}"] = summarise_records([r for r in rv if r["dir"] == d], n_boot, seed)
        out["variants"][v] = blk
    pr = self_preference(rows)
    dd: dict = dict(overall=summarise_pref(pr, n_boot, seed))
    for f in ("bo1", "bo3"):
        dd[f] = summarise_pref([r for r in pr if r["fmt"] == f], n_boot, seed)
    dd["per_set"] = {s: summarise_pref([r for r in pr if r["set"] == s], n_boot, seed) for s in sorted({r["set"] for r in pr})}
    out["default_vs_default"] = dd
    return out


# =====================================================================================================================
# the decision rule of docs/archetype_window/RULE.md: subsets R (random pools), T (targeted pools), C (changed pool-directions)
# =====================================================================================================================

AW_MIN_CHANGED = 40            # C must hold at least this many pool-directions, else INCONCLUSIVE
AW_C_LOWER = -0.10             # mean gap_out over C: CI lower bound must be above this (pp)
AW_FMT_UPPER, AW_FMT_MEAN = 0.0, -0.05    # a format is harmed if its CI upper bound < 0 AND its mean <= -0.05 pp
AW_R_LOWER = -0.05             # mean gap_out over R: CI lower bound must be above this (pp)


def pool_keys(pools_json) -> set[tuple]:
    """{(set, fmt, draft_id)} of a pools file (`{"pools": [{set, fmt, draft_id, ...}]}`)."""
    d = json.loads(Path(pools_json).read_text(encoding="utf-8"))
    return {(e["set"], e["fmt"], str(e["draft_id"])) for e in d["pools"]}


def tag_records(recs: list[dict], r_keys: set, t_keys: set) -> list[dict]:
    """Add in_R / in_T / changed (the two top decks differ) to every pool-direction record."""
    out = []
    for r in recs:
        k = tuple(r["pool"].split("|", 2))
        out.append(dict(r, in_R=k in r_keys, in_T=k in t_keys and k not in r_keys, changed=not r["same_deck"]))
    return out


def subset_records(recs: list[dict]) -> dict[str, list[dict]]:
    """R, T and C = the changed pool-directions of R and T together."""
    return dict(R=[r for r in recs if r["in_R"]], T=[r for r in recs if r["in_T"]],
                C=[r for r in recs if (r["in_R"] or r["in_T"]) and r["changed"]])


def archetype_window_decision(c_all: dict, c_fmt: dict, r_all: dict) -> dict:
    """The four lines of the rule from the summaries (`summarise_records`) of C overall, C per format and R overall.
    Each line is PASS, FAIL or INCONCLUSIVE (no data); the overall verdict is INCONCLUSIVE when line 1 fails (not enough changed
    pool-directions: nothing ships), else FAIL if any other line fails, else PASS."""
    n_c = int(c_all.get("n", 0)) if c_all else 0
    lines: dict = {}
    lines["1_enough_changed"] = dict(value=n_c, need=AW_MIN_CHANGED, status="PASS" if n_c >= AW_MIN_CHANGED else "INCONCLUSIVE")
    lo = c_all["gap_out_ci"][0] if c_all else float("nan")
    lines["2_C_not_worse"] = dict(mean=c_all.get("mean_gap_out") if c_all else None, ci_lower=lo, need=f"> {AW_C_LOWER}",
                                  status="INCONCLUSIVE" if not np.isfinite(lo) else "PASS" if lo > AW_C_LOWER else "FAIL")
    fm = {}
    bad = []
    for f in ("bo1", "bo3"):
        s = c_fmt.get(f) or {}
        if not s:
            fm[f] = dict(status="INCONCLUSIVE")
            continue
        harmed = s["gap_out_ci"][1] < AW_FMT_UPPER and s["mean_gap_out"] <= AW_FMT_MEAN
        fm[f] = dict(n=s["n"], mean=s["mean_gap_out"], ci=s["gap_out_ci"], status="FAIL" if harmed else "PASS")
        if harmed:
            bad.append(f)
    lines["3_no_format_harmed"] = dict(formats=fm, status="FAIL" if bad else "PASS" if all(v["status"] == "PASS" for v in fm.values())
                                      else "INCONCLUSIVE")
    rl = r_all["gap_out_ci"][0] if r_all else float("nan")
    lines["4_R_no_regression"] = dict(mean=r_all.get("mean_gap_out") if r_all else None, ci_lower=rl, need=f"> {AW_R_LOWER}",
                                      status="INCONCLUSIVE" if not np.isfinite(rl) else "PASS" if rl > AW_R_LOWER else "FAIL")
    if lines["1_enough_changed"]["status"] != "PASS":
        verdict = "INCONCLUSIVE"
    elif any(lines[k]["status"] == "FAIL" for k in ("2_C_not_worse", "3_no_format_harmed", "4_R_no_regression")):
        verdict = "FAIL"
    elif all(lines[k]["status"] == "PASS" for k in lines):
        verdict = "PASS"
    else:
        verdict = "INCONCLUSIVE"
    gain = bool(c_all) and np.isfinite(c_all["gap_out_ci"][0]) and c_all["gap_out_ci"][0] > 0
    return dict(lines=lines, verdict=verdict, measurable_gain=bool(gain))


def creature_tables(out: Path, sets) -> dict:
    """{set: (is_creature, is_land)} from the half-A model of each set (all models of a set share the card ids); {} if unavailable."""
    tabs: dict = {}
    for s in sets:
        try:
            m = _model(str(home_of(out, s, "A")), s)
            tabs[s] = (np.asarray(m.card_arrays.is_creature, dtype=bool), np.asarray(m.card_arrays.is_land, dtype=bool))
        except Exception:  # noqa: BLE001 - descriptive only
            continue
    return tabs


def creature_counts(rows: list[dict], variant: str, base: str, tabs: dict, keep) -> dict:
    """Nonland creature counts of the base and variant top decks over the changed pool-directions selected by `keep(set, fmt, draft,
    dir)`: {n, mean_base, mean_variant, share_fewer, share_more, min_variant, max_variant}."""
    b, v = [], []
    for r in rows:
        if r["set"] not in tabs:
            continue
        cre, land = tabs[r["set"]]
        for d, byv in r["dirs"].items():
            if base not in byv or variant not in byv or byv[base]["deck"] == byv[variant]["deck"]:
                continue
            if not keep(r["set"], r["fmt"], str(r["draft_id"]), d):
                continue
            n = lambda deck: sum(c for i, c in deck.items() if cre[int(i)] and not land[int(i)])  # noqa: E731
            b.append(n(byv[base]["deck"]))
            v.append(n(byv[variant]["deck"]))
    if not b:
        return {}
    b, v = np.array(b), np.array(v)
    return dict(n=len(b), mean_base=float(b.mean()), mean_variant=float(v.mean()), share_fewer=float((v < b).mean()),
                share_more=float((v > b).mean()), min_variant=int(v.min()), max_variant=int(v.max()),
                min_base=int(b.min()), max_base=int(b.max()))


def rule_summary(rows: list[dict], variant: str, base: str, r_keys: set, t_keys: set, n_boot: int = 2000, seed: int = 0,
                 tabs: dict | None = None) -> dict:
    """Everything the archetype-window rule asks for: gap_in / gap_out (with the out/in ratio) for R, T and C, overall, per format, per
    set and per set x format; the share of pools whose deck changes; the creature counts of the changed decks; the decision lines."""
    recs = tag_records([r for r in pool_direction_records(rows, (variant,), base)], r_keys, t_keys)
    subs = subset_records(recs)
    out: dict = dict(variant=variant, base=base, n_R_pools=len(r_keys), n_T_pools=len(t_keys - r_keys), subsets={})
    for name, rs in subs.items():
        blk: dict = dict(overall=summarise_records(rs, n_boot, seed))
        for f in ("bo1", "bo3"):
            blk[f] = summarise_records([r for r in rs if r["fmt"] == f], n_boot, seed)
        blk["per_set"] = {s: summarise_records([r for r in rs if r["set"] == s], n_boot, seed) for s in sorted({r["set"] for r in rs})}
        blk["per_cell"] = {f"{s} {f}": summarise_records([r for r in rs if r["set"] == s and r["fmt"] == f], n_boot, seed)
                           for s, f in sorted({(r["set"], r["fmt"]) for r in rs})}
        if name != "C":
            pools = {r["pool"] for r in rs}
            ch = {r["pool"] for r in rs if r["changed"]}
            blk["share_pools_changed"] = (len(ch) / len(pools)) if pools else None
            blk["n_pools"] = len(pools)
        out["subsets"][name] = blk
    out["decision"] = archetype_window_decision(out["subsets"]["C"]["overall"], {f: out["subsets"]["C"][f] for f in ("bo1", "bo3")},
                                                out["subsets"]["R"]["overall"])
    if tabs:
        def in_(which):
            ks = r_keys if which == "R" else t_keys - r_keys
            return lambda s, f, d, _dir: (s, f, d) in ks
        out["creatures"] = {w: creature_counts(rows, variant, base, tabs, in_(w)) for w in ("R", "T")}
        out["creatures"]["C"] = creature_counts(rows, variant, base, tabs, lambda s, f, d, _dir: (s, f, d) in (r_keys | t_keys))
        out["creatures"]["SOS"] = creature_counts(rows, variant, base, tabs,
                                                  lambda s, f, d, _dir: s == "SOS" and (s, f, d) in (r_keys | t_keys))
    return out


def rule_markdown(rs: dict) -> str:
    def cell(name, s):
        if not s:
            return f"| {name} | 0 | - | - | - | - |"
        return (f"| {name} | {s['n']} | {_f(s['mean_gap_in'])} | {_f(s['mean_gap_out'])} [{_f(s['gap_out_ci'][0])}, "
                f"{_f(s['gap_out_ci'][1])}] | {_f(s['ratio'])} | {100 * s['share_same_deck']:.0f}% |")
    L = [f"# Creature window per archetype group: {rs['variant']} against {rs['base']} (docs/archetype_window/RULE.md)", "",
         f"R: {rs['n_R_pools']} random pools; T: {rs['n_T_pools']} targeted pools; C: the changed pool-directions of R and T.  Gaps are "
         "variant deck minus base deck, pp of game win rate, draw 0, converted at the searching model's base-deck level; gap_out = scored "
         "by the other half's model; 95% cluster bootstrap over pools.", "", "## Decision", ""]
    dec = rs["decision"]
    ln = dec["lines"]
    L.append(f"1. Enough evidence: C = {ln['1_enough_changed']['value']} pool-directions (need >= {ln['1_enough_changed']['need']}): "
             f"**{ln['1_enough_changed']['status']}**")
    x = ln["2_C_not_worse"]
    L.append(f"2. Not worse where it acts: C mean gap_out {_f(x['mean'])}, CI lower {_f(x['ci_lower'])} (need {x['need']}): **{x['status']}**")
    x = ln["3_no_format_harmed"]
    L.append("3. No format harmed: " + "; ".join(
        f"{f}: n {v.get('n', 0)}, mean {_f(v.get('mean'))}, CI [{_f(v['ci'][0]) if 'ci' in v else '-'}, {_f(v['ci'][1]) if 'ci' in v else '-'}] "
        f"{v['status']}" for f, v in x["formats"].items()) + f": **{x['status']}**")
    x = ln["4_R_no_regression"]
    L.append(f"4. No regression on random pools: R mean gap_out {_f(x['mean'])}, CI lower {_f(x['ci_lower'])} (need {x['need']}): **{x['status']}**")
    L += ["", f"**Verdict: {dec['verdict']}**" + ("  (a measurable gain: C gap_out CI lower bound > 0)" if dec["measurable_gain"] else ""), ""]
    for name in ("R", "T", "C"):
        blk = rs["subsets"][name]
        L += [f"## {name}", ""]
        if name != "C" and blk.get("n_pools") is not None:
            L.append(f"{blk['n_pools']} pools; the deck changes in {_f(100 * blk['share_pools_changed'], 0)}% of them.")
            L.append("")
        L += ["| cell | pool-directions | mean gap_in | mean gap_out [95% CI] | out / in | same deck |", "|---|---:|---:|---:|---:|---:|"]
        L.append(cell("ALL", blk["overall"]))
        for f in ("bo1", "bo3"):
            L.append(cell(f, blk[f]))
        for k, s in blk["per_cell"].items():
            L.append(cell(k, s))
        L.append("")
    if rs.get("creatures"):
        L += ["## Creature counts of the changed decks (nonland creature copies)", "",
              "| subset | changed pool-directions | base mean | variant mean | variant fewer | variant more | base range | variant range |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for k, c in rs["creatures"].items():
            if c:
                L.append(f"| {k} | {c['n']} | {_f(c['mean_base'], 1)} | {_f(c['mean_variant'], 1)} | {100 * c['share_fewer']:.0f}% | "
                         f"{100 * c['share_more']:.0f}% | {c['min_base']}-{c['max_base']} | {c['min_variant']}-{c['max_variant']} |")
        L.append("")
    s = rs["subsets"]["C"]["overall"]
    ra = rs["subsets"]["R"]["overall"]
    if s and ra:
        L.append(f"Seconds per pool (variant / base): R {ra['mean_seconds']:.0f} / {ra['mean_seconds_default']:.0f}; "
                 f"C {s['mean_seconds']:.0f} / {s['mean_seconds_default']:.0f}.")
    return "\n".join(L) + "\n"


def write_rule_report(out: Path, rule: str, variant: str, base: str, r_json: Path, t_json: Path | None, extra: tuple[str, ...] = (),
                      name: str | None = None, n_boot: int = 2000) -> str:
    if rule != "archetype_window":
        raise ValueError(f"unknown rule {rule!r}")
    rows = load_rows(out, extra)
    r_keys = pool_keys(r_json)
    t_keys = pool_keys(t_json) if t_json else set()
    tabs = creature_tables(out, sorted({r["set"] for r in rows}))
    rs = rule_summary(rows, variant, base, r_keys, t_keys, n_boot=n_boot, tabs=tabs)
    name = name or rule
    (out / f"{name}.json").write_text(json.dumps(rs, indent=1), encoding="utf-8")
    md = rule_markdown(rs)
    (out / f"{name}.md").write_text(md, encoding="utf-8")
    return md


def _f(x, nd=2):
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def markdown(summ: dict, meta: dict) -> str:
    L = ["# Split-half cross-evaluation of the deck-search gain", "",
         "Training events of each set (before the hold-out cut, all formats) are split at random by event into halves A and B "
         "(stratified by format and time). Model A is fitted on A, model B on B. On the same held-out pools the default search and a "
         "larger search are run with each model; the top decks are scored by the searching model (`in`, what the search sees) and by "
         "the other half's model (`out`, independent data). Gaps are variant deck minus default deck, win-rate percentage points, "
         "draw 0, both converted at the level of the searching model's default deck (the two half models differ in absolute level); one record per pool and searching model; intervals are 95% cluster bootstraps over pools.", ""]
    for v, blk in summ["variants"].items():
        L += [f"## {v} against {summ.get('base', 'default')}", "",
              "| cell | pool-directions | mean gap in | mean gap out [95% CI] | out / in [95% CI] | out / in (logit) | share out > 0 | "
              "share out < 0 | same deck | s/pool v / default |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        rows = [("ALL", blk["overall"]), ("bo1", blk["bo1"]), ("bo3", blk["bo3"])] + list(blk["per_set"].items())
        for name, s in rows:
            if not s:
                continue
            L.append(f"| {name} | {s['n']} | {_f(s['mean_gap_in'])} | {_f(s['mean_gap_out'])} [{_f(s['gap_out_ci'][0])}, {_f(s['gap_out_ci'][1])}] | "
                     f"{_f(s['ratio'])} [{_f(s['ratio_ci'][0])}, {_f(s['ratio_ci'][1])}] | {_f(s['ratio_logit'])} | {100 * s['share_out_pos']:.0f}% | "
                     f"{100 * s['share_out_neg']:.0f}% | {100 * s['share_same_deck']:.0f}% | "
                     f"{s['mean_seconds']:.0f} / {s['mean_seconds_default']:.0f} |")
        L.append("")
        L.append("By searching model: " + "; ".join(
            f"{d}: in {_f(blk[f'dir_{d}']['mean_gap_in'])}, out {_f(blk[f'dir_{d}']['mean_gap_out'])}" for d in HALVES if blk.get(f"dir_{d}")) + ".")
        L.append("")
    dd = summ["default_vs_default"]
    L += ["## Default search against default search (scale of the disagreement between the halves)", "",
          "Deck found by model A vs deck found by model B; the entry is how many win-rate points each model prefers its OWN deck "
          "(mean of the two models, per pool; 0 would mean the halves agree on the best deck).", "",
          "| cell | pools | mean self-preference pp [95% CI] | model A | model B | same deck |", "|---|---:|---:|---:|---:|---:|"]
    for name, s in [("ALL", dd["overall"]), ("bo1", dd["bo1"]), ("bo3", dd["bo3"])] + list(dd["per_set"].items()):
        if s:
            L.append(f"| {name} | {s['n_pools']} | {_f(s['mean_self_preference_pp'])} [{_f(s['ci'][0])}, {_f(s['ci'][1])}] | "
                     f"{_f(s['mean_pref_a'])} | {_f(s['mean_pref_b'])} | {100 * s['share_same_deck']:.0f}% |")
    L += ["", "## Fits", "", "| set | half | train events | train games | fit s | components applied |", "|---|---|---:|---:|---:|---|"]
    for k, m in sorted(meta.get("fits", {}).items()):
        L.append(f"| {m['set']} | {m['half']} | {m['n_events']} | {m['n_games']} | {m['fit_seconds']:.0f} | "
                 f"{', '.join(c for c, on in sorted(m.get('applied', {}).items()) if on) or '-'} |")
    L += ["", f"Search wall time {meta.get('search_wall_seconds', 0) / 60:.0f} min; options {meta.get('model_opts')}; n_boot {meta.get('n_boot')}."]
    return "\n".join(L) + "\n"


# =====================================================================================================================
# progress file
# =====================================================================================================================

class Progress:
    """OUT/progress.txt: first line = plan, then one line per finished unit, a rolling status line last. Rewritten on every update."""

    def __init__(self, out: Path, plan: str):
        self.path, self.plan, self.units, self.status = out / "progress.txt", plan, [], ""
        self._load(out)

    def _load(self, out: Path) -> None:
        p = out / "progress_units.json"
        if p.exists():
            self.units = json.loads(p.read_text(encoding="utf-8"))

    def done(self, msg: str) -> None:
        self.units.append(f"{msg} {time.strftime('%H:%M')}")
        (self.path.parent / "progress_units.json").write_text(json.dumps(self.units), encoding="utf-8")
        self.write()

    def set_status(self, msg: str) -> None:
        self.status = f"[now {time.strftime('%H:%M')}] {msg}"
        self.write()

    def write(self) -> None:
        self.path.write_text("\n".join([self.plan] + self.units + ([self.status] if self.status else [])) + "\n", encoding="utf-8")


# =====================================================================================================================
# stage 1: the half fits
# =====================================================================================================================

MODEL_OPTS = dict(sigma_cv="time", stack="auto", confirm=True, recency_scope="format")    # deviation_cost's defaults


def home_of(out: Path, set_code: str, half: str) -> Path:
    return out / "homes" / f"{set_code}_{half}"


def training_events(games_all):
    """(draft_id list, start list, fmt list, cut) of the training events of deviation_cost's train-only model for these games."""
    import polars as pl

    from sealed_lab.model import design, validate

    target = next(f for f in ("bo3", "bo1") if games_all.filter(pl.col("fmt") == f).height)
    split = validate.time_split(games_all.filter(pl.col("fmt") == target))
    if split is None:
        raise RuntimeError("not enough events for a hold-out")
    t_cut = split[1]
    starts = design.event_starts(games_all)
    ev = starts.filter(pl.col("start") < t_cut).join(games_all.group_by("draft_id").agg(pl.col("fmt").first()), on="draft_id")
    return ev["draft_id"].to_list(), ev["start"].to_list(), ev["fmt"].to_list(), t_cut


def half_events(src: Path, set_code: str, seed: int) -> dict:
    import polars as pl

    games_all = pl.read_parquet(src / "sets" / set_code / "corpus" / "games.parquet")
    ids, starts, fmts, t_cut = training_events(games_all)
    a, b = split_events(ids, starts, fmts, set_seed(seed, set_code))
    return dict(A=a, B=b, cut=str(t_cut), n_train=len(ids))


def fit_half(set_code: str, half: str, src: Path, out: Path, n_boot: int, seed: int, jobs: int | None, opts: dict | None = None) -> dict:
    import polars as pl

    from sealed_lab.model import pipeline

    set_code = set_code.upper()
    opts = dict(opts or {})
    meta_path = out / "fits" / f"{set_code}_{half}.json"
    home = home_of(out, set_code, half)
    dst_set = home / "sets" / set_code
    if dst_set.exists():
        shutil.rmtree(dst_set)
    dst_set.mkdir(parents=True)
    for sub in ("corpus", "cache"):
        shutil.copytree(src / "sets" / set_code / sub, dst_set / sub)
    shutil.copy(src / "sets" / set_code / "registry.json", dst_set / "registry.json")
    os.environ["SEALED_LAB_HOME"] = str(home)
    ev = half_events(src, set_code, seed)
    keep = set(ev[half])
    orig = pipeline.load_inputs

    def load_half(sc):
        inp = orig(sc)
        inp.games = inp.games.filter(pl.col("draft_id").is_in(keep))
        inp.builds = inp.builds.filter(pl.col("draft_id").is_in(keep))
        return inp

    pipeline.load_inputs = load_half
    t0 = time.time()
    try:
        meta = pipeline.calculate(set_code, progress=lambda f, m: None, n_jobs=jobs, n_boot=n_boot, recency="off", **MODEL_OPTS, **opts)
    finally:
        pipeline.load_inputs = orig
    secs = time.time() - t0
    applied = {k: bool(v.get("applied")) for k, v in (meta or {}).items() if isinstance(v, dict) and "applied" in v}
    games = pl.read_parquet(dst_set / "corpus" / "games.parquet", columns=["draft_id"])
    rec = dict(set=set_code, half=half, fit_seconds=secs, n_events=len(keep), n_games=int(games.filter(pl.col("draft_id").is_in(keep)).height),
               n_train_events=ev["n_train"], cut=ev["cut"], applied=applied, n_boot=n_boot, model_opts={**MODEL_OPTS, **opts})
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    # the per-set event lists, once (audit; both fits write the same file)
    (out / "splits").mkdir(exist_ok=True)
    (out / "splits" / f"{set_code}.json").write_text(json.dumps(dict(cut=ev["cut"], A=[str(x) for x in ev["A"]],
                                                                     B=[str(x) for x in ev["B"]])), encoding="utf-8")
    print(f"fit {set_code} {half}: {secs:.0f}s, {len(keep)} events, {rec['n_games']} games, applied {applied}", flush=True)
    return rec


# =====================================================================================================================
# stage 2: the searches
# =====================================================================================================================

_CACHE: dict = {}


@contextlib.contextmanager
def _home(path: str):
    old = os.environ.get("SEALED_LAB_HOME")
    os.environ["SEALED_LAB_HOME"] = path
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("SEALED_LAB_HOME", None)
        else:
            os.environ["SEALED_LAB_HOME"] = old


def _model(home: str, set_code: str):
    from sealed_lab.model.artifact import SetModel

    key = (home, set_code)
    if key not in _CACHE:
        for k in [k for k in _CACHE if k[1] != set_code]:        # tasks are ordered by set: keep only the current set's two models
            del _CACHE[k]
        with _home(home):
            _CACHE[key] = SetModel.load(set_code)
    return _CACHE[key]


def _pool_task(task: dict) -> dict:
    """One pool: for each half as the searching model, the three searches and the draw-0 logit of every top deck under both models."""
    from search_check import VARIANTS, _deck_of, patched_search

    from sealed_lab.deck import builder

    sc, fmt, pool = task["set"], task["fmt"], task["pool"]
    models = {h: _model(task["homes"][h], sc) for h in HALVES}
    out = dict(set=sc, fmt=fmt, draft_id=task["draft_id"], dirs={})
    for s in HALVES:
        o = OTHER[s]
        byv = {}
        for v in task.get("variants", VARIANT_NAMES):
            t0 = time.time()
            with _home(task["homes"][s]), patched_search(VARIANTS[v]):
                res = builder.build(models[s], pool, fmt, n_decks=1, families=True, n_jobs=1)
            secs = time.time() - t0
            deck = _deck_of(res)
            with _home(task["homes"][s]):
                ls = float(models[s].logit(deck, fmt))
            with _home(task["homes"][o]):
                lo = float(models[o].logit(deck, fmt))
            byv[v] = dict(ls=ls, lo=lo, seconds=secs, deck={str(k): int(c) for k, c in sorted(deck.items())},
                          n_candidates=(res.family_meta or {}).get("n_candidates"))
        out["dirs"][s] = byv
    return out


def _task_key(r: dict) -> tuple:
    return (r["set"], r["fmt"], str(r["draft_id"]))


def run_searches(src: Path, dev: Path, pools_json: Path, out: Path, jobs: int, prog: Progress | None = None,
                 deadline: float | None = None, variants=VARIANT_NAMES, rows_name: str = "search_rows.jsonl",
                 only=None, homes_from: Path | None = None) -> None:
    from search_check import load_pool_tasks

    tasks = select_tasks(load_pool_tasks(dev, pools_json), only)
    for t in tasks:
        t["homes"] = {h: str(home_of(homes_from or out, t["set"], h)) for h in HALVES}
        t.pop("home", None)
        t["variants"] = tuple(variants)
    rows_path = out / rows_name
    done_rows = [json.loads(x) for x in rows_path.read_text(encoding="utf-8").splitlines() if x.strip()] if rows_path.exists() else []
    done = {_task_key(r) for r in done_rows}
    todo = [t for t in tasks if _task_key(t) not in done]
    todo.sort(key=lambda t: (t["set"], -t["prior_seconds"], t["fmt"], str(t["draft_id"])))
    per_set = {s: sum(1 for t in tasks if t["set"] == s) for s in {t["set"] for t in tasks}}
    count = {s: sum(1 for r in done_rows if r["set"] == s) for s in per_set}
    finished_sets = {s for s in per_set if count[s] >= per_set[s]}
    print(f"search: {len(tasks)} pools, {len(done)} done, {len(todo)} to do, {jobs} workers", flush=True)
    t0 = time.time()
    with cf.ProcessPoolExecutor(max_workers=jobs, mp_context=multiprocessing.get_context("spawn")) as ex:
        it = iter(todo)
        pending: dict = {}

        def submit_more():
            while len(pending) < jobs * 2:
                t = next(it, None)
                if t is None:
                    return
                if deadline and time.time() > deadline:
                    return
                pending[ex.submit(_pool_task, t)] = t

        submit_more()
        while pending:
            fin, _ = cf.wait(list(pending), return_when=cf.FIRST_COMPLETED)
            for fu in fin:
                t = pending.pop(fu)
                r = fu.result()
                with open(rows_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(r) + "\n")
                count[t["set"]] += 1
                if prog:
                    n_done = sum(count.values())
                    prog.set_status(f"searching: {n_done}/{len(tasks)} pools done ({t['set']} {count[t['set']]}/{per_set[t['set']]}), "
                                    f"{(time.time() - t0) / 60:.0f} min in this run")
                    if count[t["set"]] >= per_set[t["set"]] and t["set"] not in finished_sets:
                        finished_sets.add(t["set"])
                        prog.done(f"search {len(finished_sets)}/{len(per_set)} sets done ({t['set']})")
            submit_more()
    print(f"search: wall {(time.time() - t0) / 60:.1f} min", flush=True)


# =====================================================================================================================
# report + driver
# =====================================================================================================================

def load_rows(out: Path, extra: tuple[str, ...] = ()) -> list[dict]:
    """The rows of search_rows.jsonl; the variants of each `extra` rows file are merged into the same pool's directions."""
    rows: dict = {}
    for name in ("search_rows.jsonl", *extra):
        for x in (out / name).read_text(encoding="utf-8").splitlines():
            if x.strip():
                r = json.loads(x)
                k = _task_key(r)
                if k in rows and name != "search_rows.jsonl":
                    for d, byv in r["dirs"].items():
                        rows[k]["dirs"].setdefault(d, {}).update(byv)
                elif name == "search_rows.jsonl":
                    rows[k] = r
    return [rows[k] for k in sorted(rows)]


def write_report(out: Path, search_wall: float | None = None, extra: tuple[str, ...] = (), compared=COMPARED,
                 base: str = "default", name: str = "split_half") -> str:
    rows = load_rows(out, extra)
    summ = full_summary(rows, compared=compared, base=base)
    fits = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted((out / "fits").glob("*.json"))}
    meta = dict(fits=fits, model_opts=MODEL_OPTS, search_wall_seconds=search_wall if search_wall is not None
                else float(sum(r["dirs"][d][v]["seconds"] for r in rows for d in r["dirs"] for v in r["dirs"][d]) / max(os.cpu_count() or 1, 1)),
                n_boot=next(iter(fits.values()))["n_boot"] if fits else None)
    (out / f"{name}.json").write_text(json.dumps(dict(summary=summ, meta=meta), indent=1), encoding="utf-8")
    md = markdown(summ, meta)
    (out / f"{name}.md").write_text(md, encoding="utf-8")
    return md


def run_all(src: Path, dev: Path, pools_json: Path, out: Path, n_boot: int, seed: int, jobs: int, max_hours: float,
            opts: dict | None = None) -> int:
    t_start = time.time()
    deadline = t_start + max_hours * 3600
    out.mkdir(parents=True, exist_ok=True)
    units = [(s, h) for s in SETS for h in HALVES]
    prog = Progress(out, f"PLAN: {len(units)} fits (11 sets x halves A,B), 11 set searches (220 pools x 2 directions x 3 variants), "
                         f"report, tests; started {time.strftime('%Y-%m-%d %H:%M')}")
    prog.write()
    for k, (s, h) in enumerate(units, 1):
        mp = out / "fits" / f"{s}_{h}.json"
        if mp.exists() and (home_of(out, s, h) / "sets" / s / "model" / "meta.json").exists():
            continue
        if time.time() > deadline:
            prog.done(f"STOPPED at the time limit before fit {k}/{len(units)}")
            return 2
        prog.set_status(f"fitting {k}/{len(units)} ({s} {h})")
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        (out / "logs").mkdir(exist_ok=True)
        with open(out / "logs" / f"fit_{s}_{h}.log", "w", encoding="utf-8") as lf:
            rc = subprocess.call([sys.executable, "-u", str(Path(__file__).resolve()), "fit", s, h, str(src), str(out),
                                  "--n-boot", str(n_boot), "--seed", str(seed), "--jobs", str(jobs),
                                  *[f"--model-opt={k}={v}" for k, v in (opts or {}).items()]], stdout=lf, stderr=subprocess.STDOUT, env=env)
        if rc != 0:
            prog.done(f"FIT FAILED {s} {h} (rc {rc}), see logs/fit_{s}_{h}.log")
            return 1
        fm = json.loads(mp.read_text(encoding="utf-8"))
        prog.done(f"fit {sum(1 for u in units if (out / 'fits' / f'{u[0]}_{u[1]}.json').exists())}/{len(units)} done ({s} {h}, {fm['fit_seconds']:.0f}s)")
    t0 = time.time()
    run_searches(src, dev, pools_json, out, jobs, prog, deadline)
    wall = time.time() - t0
    n_rows = len(load_rows(out)) if (out / "search_rows.jsonl").exists() else 0
    write_report(out, wall)
    prog.done(f"report written ({n_rows} pools searched)")
    prog.set_status("finished" if time.time() <= deadline else "time limit reached: partial")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "search"):
        p = sub.add_parser(name)
        p.add_argument("src", help="SEALED_LAB_HOME-style folder with sets/SET/{corpus,cache,registry.json} (scratch copy)")
        p.add_argument("devdir", help="folder with SET/home for the pools of POOLS_JSON (as for search_check --pools-from)")
        p.add_argument("pools_json")
        p.add_argument("out")
        p.add_argument("--n-boot", type=int, default=4)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
        p.add_argument("--max-hours", type=float, default=6.0)
        p.add_argument("--variants", nargs="+", default=list(VARIANT_NAMES), help="search: the variants to run (see the module docstring)")
        p.add_argument("--rows", default="search_rows.jsonl", help="search: the rows file to append to")
        p.add_argument("--only", nargs="*", default=None, help="search: only these sets / SET:fmt cells of the pools file")
        p.add_argument("--homes-from", default=None, help="search: take the half homes from this OUT tree instead of OUT")
        p.add_argument("--model-opt", action="append", default=[], help="run: force a model option in every fit, KEY=VALUE (repeatable)")
    p = sub.add_parser("fit")
    p.add_argument("set")
    p.add_argument("half", choices=HALVES)
    p.add_argument("src")
    p.add_argument("out")
    p.add_argument("--n-boot", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--jobs", type=int, default=None)
    p.add_argument("--model-opt", action="append", default=[], help="force a model option, KEY=VALUE (repeatable), e.g. matchups=on")
    p = sub.add_parser("report")
    p.add_argument("out")
    p.add_argument("--extra-rows", nargs="*", default=[])
    p.add_argument("--compare", nargs="+", default=list(COMPARED))
    p.add_argument("--base", default="default")
    p.add_argument("--name", default="split_half", help="output name: OUT/NAME.json and OUT/NAME.md")
    p.add_argument("--rule", default=None, choices=["archetype_window"],
                   help="write the decision report of that rule (docs/<rule>/RULE.md) instead of the usual one; needs --pools-r")
    p.add_argument("--pools-r", default=None, help="--rule: the R pools file (docs/builder_scoreboard/search_check.json)")
    p.add_argument("--pools-t", default=None, help="--rule: the targeted pools file (scripts/archetype_window_pools.py)")
    a = ap.parse_args(argv)
    if a.cmd == "fit":
        fit_half(a.set, a.half, Path(a.src).resolve(), Path(a.out).resolve(), a.n_boot, a.seed, a.jobs, parse_model_opts(a.model_opt))
        return 0
    if a.cmd == "report" and a.rule:
        if not a.pools_r:
            ap.error("--rule needs --pools-r")
        variant = a.compare[0] if a.compare != list(COMPARED) else "arch_window"
        print(write_rule_report(Path(a.out).resolve(), a.rule, variant, a.base, Path(a.pools_r), Path(a.pools_t) if a.pools_t else None,
                                tuple(a.extra_rows), name=None if a.name == "split_half" else a.name))
        return 0
    if a.cmd == "report":
        print(write_report(Path(a.out).resolve(), extra=tuple(a.extra_rows), compared=tuple(a.compare), base=a.base, name=a.name))
        return 0
    if a.cmd == "search":
        run_searches(Path(a.src).resolve(), Path(a.devdir).resolve(), Path(a.pools_json), Path(a.out).resolve(), a.jobs,
                     variants=a.variants, rows_name=a.rows, only=a.only, homes_from=Path(a.homes_from).resolve() if a.homes_from else None)
        return 0
    return run_all(Path(a.src).resolve(), Path(a.devdir).resolve(), Path(a.pools_json), Path(a.out).resolve(), a.n_boot, a.seed, a.jobs, a.max_hours,
                   parse_model_opts(a.model_opt))


if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
