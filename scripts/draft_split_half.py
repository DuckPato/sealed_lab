"""Draft mode, primary test (docs/draft_mode/RULE.md): split-half deck choice, draft model against the Sealed model of the matching queue.

    python scripts/draft_split_half.py run     SRC OUT [--n-boot 4] [--draft-boot 4] [--jobs N] [--sets ECL FIN ...] [--max-hours 40]
    python scripts/draft_split_half.py cuts    SRC OUT [--sets ...]                       (hold-out cuts + halves; OUT/splits/SET.json)
    python scripts/draft_split_half.py fit     SET HALF SRC OUT [--n-boot 4] [--draft-boot 4] [--jobs N]
    python scripts/draft_split_half.py pools   SRC OUT [--sets ...]                       (OUT/pools.json, written ONCE, before any search)
    python scripts/draft_split_half.py search  SRC OUT [--jobs N]                         (resumable: OUT/search_rows.jsonl)
    python scripts/draft_split_half.py report  OUT                                       (OUT/draft_split_half.md / .json)

Second test (docs/draft_mode2/RULE.md): add `--rule draft_mode2` to `run` / `pools` (default sets FRA HOB MSH SOS TMT; the first test's 10 pools per
cell are recomputed and excluded, 20 are drawn per set x format with seed 1; halves A / B keep seed 0; zero overlap asserted; pools.json also lists the
excluded draw and, with `--first-pools-json OLD/pools.json`, the comparison with the first test's saved file). Without `--rule` nothing changes.

SRC = a scratch home (SEALED_LAB_HOME layout) whose sets/SET/{corpus,cache,registry.json} exist; it is only read. OUT = a scratch folder.
Every fit runs in its own scratch home OUT/homes/SET_A|B (a COPY of the set's corpus / cache / registry, never a link).

Design (RULE.md, "Primary test"). Per set, the hold-out cut of the Sealed formats is exactly `split_half_search.training_events`' cut; the cut of
each draft format (dbo1 = PremierDraft, dbo3 = TradDraft) is `validate.time_split` on that format's games (the last 25% of its events by
start). The training events of dbo1 and dbo3 are split by event into halves A and B (`split_half_search.split_events`, seed 0). One
`pipeline.calculate(draft_formats=True)` per half, on these games (`load_inputs` patched, as `fit_half` does):
    * Sealed games of the events that started before the Sealed cut (all of them, the same in A and B),
    * dbo1 and dbo3 games of THAT HALF's training events only,
    * the other draft events (not PremierDraft / TradDraft) that started before the dbo1 cut (the dbo3 cut if the set has no dbo1).
So in one artifact: the draft models (their chains on the half's draft events, prior = the other draft events before the cut) and the Sealed
comparison models (today's Sealed chain on the train-only Sealed games, whose draft prior stage sees only the half's draft events + the other
draft events before the cut). Neither has seen the other half.

Search: per held-out draft pool and searching half s: the top deck of the draft model s in the draft format (deck_d) and of the Sealed model s in
the matching Sealed format (deck_s; bo1 for dbo1, bo3 for dbo3), the default builder (families, n_decks = 1, n_jobs = 1). Both are scored at
draw 0 by model s in the DRAFT format (gap_in) and by the other half's model in the draft format (gap_out). gap = deck_d minus deck_s, in pp of
game win rate, with the logit differences converted at ONE level, that of deck_s under the searching model (`split_half_search.pool_direction_
records`: gap = 100 * (expit(l0 + dl) - expit(l0)), l0 = logit of deck_s under model s). Decision: report().
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import split_half_search as sh  # noqa: E402  (numpy-only at import time)

SETS = list(sh.SETS)
HALVES = sh.HALVES
OTHER = sh.OTHER
DRAFT_FORMATS = ("dbo1", "dbo3")
SEALED_OF = {"dbo1": "bo1", "dbo3": "bo3"}
N_POOLS = 10               # held-out draft pools per set x draft format
MIN_POOL_CARDS = 40        # deck + sideboard, basics excluded
HARM_UPPER, HARM_MEAN = 0.0, -0.05       # RULE.md line 2: CI upper < 0 with a mean <= -0.05 pp
SEED = 0


# =====================================================================================================================
# pure parts: cuts, training groups, halves, pool eligibility, decision lines (tested without any model fit)
# =====================================================================================================================

def event_table(games):
    """One row per event: draft_id, start, ev_fmt in {bo1, bo3, dbo1, dbo3, draft} (draft = the other draft events). `games` needs the
    columns draft_id, fmt, event, game_time, draft_time."""
    import polars as pl

    from sealed_lab.model import design, draftfmt

    cols = ["draft_id", "fmt", "game_time", "draft_time"]
    sealed = games.filter(pl.col("fmt").is_in(["bo1", "bo3"])).select(cols)
    draft = draftfmt.relabel(games, named=True).select(cols)
    g = pl.concat([sealed, draft])
    ev = design.event_starts(g).join(g.group_by("draft_id").agg(pl.col("fmt").first().alias("ev_fmt")), on="draft_id")
    return ev.select("draft_id", "start", "ev_fmt").sort("start", "draft_id")


def cuts_of(games) -> dict:
    """{"sealed": the Sealed hold-out cut (split_half_search.training_events), "dbo1": ..., "dbo3": ...}; None where there is no hold-out
    (no games / too few events: validate.time_split returns None)."""
    import polars as pl

    from sealed_lab.model import draftfmt, validate

    out: dict = {"sealed": None, "dbo1": None, "dbo3": None}
    try:
        out["sealed"] = sh.training_events(games)[3]
    except (StopIteration, RuntimeError):
        pass
    g = draftfmt.relabel(games, named=True)
    for f in DRAFT_FORMATS:
        sub = g.filter(pl.col("fmt") == f)
        if sub.height:
            sp = validate.time_split(sub)
            out[f] = sp[1] if sp is not None else None
    return out


def named_games(games):
    """The whole game table with the draft games relabelled to the draft formats' names (fmt dbo1 / dbo3 / draft); Sealed games unchanged."""
    import polars as pl

    from sealed_lab.model import draftfmt

    return pl.concat([games.filter(pl.col("fmt") != "draft"), draftfmt.relabel(games, named=True)])


def train_only_ids(games) -> tuple[dict, set]:
    """(cuts, draft_ids of the train-only model of deviation_cost's draft mode): Sealed events before the Sealed cut, each draft format's
    events before its own cut, the other draft events before `other_cut`. `games` needs draft_id, fmt, event, game_time, draft_time."""
    cuts = cuts_of(games)
    groups = train_groups(event_table(games), cuts)
    return cuts, {d for g in groups.values() for d, _ in g}


def other_cut(cuts: dict):
    """The cut of the other draft events (QuickDraft, PickTwo...): the dbo1 cut, or the dbo3 cut if the set has no dbo1 hold-out."""
    return cuts.get("dbo1") if cuts.get("dbo1") is not None else cuts.get("dbo3")


def train_groups(ev, cuts: dict) -> dict:
    """Training events by group, as {group: [(draft_id, start), ...]}: sealed (before the Sealed cut), dbo1 / dbo3 (before their own cut), other
    (other draft events before `other_cut`). A group without a cut is empty."""
    import polars as pl

    def pick(fmt, cut):
        if cut is None:
            return []
        s = ev.filter((pl.col("ev_fmt") == fmt) & (pl.col("start") < cut))
        return list(zip(s["draft_id"].to_list(), s["start"].to_list()))

    sealed = pick("bo3", cuts.get("sealed")) + pick("bo1", cuts.get("sealed"))
    return dict(sealed=sealed, dbo1=pick("dbo1", cuts.get("dbo1")), dbo3=pick("dbo3", cuts.get("dbo3")),
                other=pick("draft", other_cut(cuts)))


def split_draft_halves(groups: dict, set_code: str, seed: int = SEED) -> tuple[dict, dict]:
    """Halves A and B of the dbo1 + dbo3 training events: `split_half_search.split_events` (stratified by format and time) with the seed of
    `split_half_search.set_seed(seed, set_code)`. Returns ({fmt: ids in A}, {fmt: ids in B})."""
    ids, starts, fmts = [], [], []
    for f in DRAFT_FORMATS:
        for d, s in groups[f]:
            ids.append(d)
            starts.append(s)
            fmts.append(f)
    a, b = sh.split_events(ids, starts, fmts, sh.set_seed(seed, set_code))
    fm = dict(zip(ids, fmts))
    return ({f: [d for d in a if fm[d] == f] for f in DRAFT_FORMATS}, {f: [d for d in b if fm[d] == f] for f in DRAFT_FORMATS})


def half_keep(sp: dict, half: str) -> set[str]:
    """The draft_ids of one half's fit: all Sealed training events, the half's own dbo1 / dbo3 events, the other draft events before the cut."""
    keep = {str(d) for d in sp["train"]["sealed"]} | {str(d) for d in sp["train"]["other"]}
    for f in DRAFT_FORMATS:
        keep |= {str(d) for d in sp[half][f]}
    return keep


def build_splits(src: Path, set_code: str, seed: int = SEED) -> dict:
    """Everything `cuts` stores for one set (see OUT/splits/SET.json)."""
    import polars as pl

    games = pl.read_parquet(src / "sets" / set_code / "corpus" / "games.parquet",
                            columns=["draft_id", "fmt", "event", "game_time", "draft_time"])
    cuts = cuts_of(games)
    out: dict = dict(set=set_code, seed=seed, cuts={k: (None if v is None else str(v)) for k, v in cuts.items()}, formats=[], skipped=None)
    if cuts["sealed"] is None:
        out["skipped"] = "no Sealed hold-out (the set has no Sealed games / too few events)"
        return out
    formats = [f for f in DRAFT_FORMATS if cuts[f] is not None]
    if not formats:
        out["skipped"] = "no draft format with a hold-out (no PremierDraft / TradDraft games, or too few events)"
        return out
    ev = event_table(games)
    groups = train_groups(ev, cuts)
    a, b = split_draft_halves(groups, set_code, seed)
    ho = {f: ev.filter((ev["ev_fmt"] == f) & (ev["start"] >= cuts[f])).height for f in formats}
    out.update(formats=formats, n_train={k: len(v) for k, v in groups.items()}, n_holdout=ho,
               train=dict(sealed=[str(d) for d, _ in groups["sealed"]], other=[str(d) for d, _ in groups["other"]]),
               A={f: [str(d) for d in a[f]] for f in DRAFT_FORMATS}, B={f: [str(d) for d in b[f]] for f in DRAFT_FORMATS})
    return out


def eligible_pools(pools, ev, cuts: dict, fmt: str, min_cards: int = MIN_POOL_CARDS) -> list[str]:
    """draft_ids of the held-out events of draft format `fmt` (start >= its cut) whose pool has >= `min_cards` cards (sum of copies)."""
    import polars as pl

    if cuts.get(fmt) is None:
        return []
    ho = set(ev.filter((pl.col("ev_fmt") == fmt) & (pl.col("start") >= cuts[fmt]))["draft_id"].to_list())
    size = pools.group_by("draft_id").agg(pl.col("n").sum().alias("cards"))
    ok = size.filter(pl.col("cards") >= min_cards)["draft_id"].to_list()
    return sorted(str(d) for d in set(ok) & ho)


def draw_pools(ids: list[str], k: int, seed: int) -> list[str]:
    """k pools at random (without replacement) from the sorted eligible ids; all of them if there are fewer."""
    ids = sorted(ids)
    if len(ids) <= k:
        return ids
    rng = np.random.default_rng(seed)
    return sorted(ids[i] for i in rng.choice(len(ids), size=k, replace=False))


def pool_seed(seed: int, set_code: str, fmt: str) -> int:
    return sh.set_seed(seed, f"{set_code}|{fmt}")


# ---- the second test (docs/draft_mode2/RULE.md): new pools / new events, the first test's excluded -------------------------------------
# The first test drew `N_POOLS` pools per set x format with `pool_seed(SEED, ...)` (cmd_pools) and its G1 used 300 held-out events chosen with
# `default_rng(0)` (deviation_cost.py --max-events 300). The second test excludes exactly those and draws again from the rest.
SECOND_SEED = 1            # seed of the new pool draw / new G1 event draw (the halves A / B keep seed SEED = 0)
SECOND_N_POOLS = 20        # pools per set x draft format
SECOND_SETS = ["FRA", "HOB", "MSH", "SOS", "TMT"]
FIRST_G1_MAX_EVENTS = 300  # the first G1's `--max-events`
FIRST_G1_SEED = 0          # ... and its `np.random.default_rng(0)`
SECOND_G1_MAX_EVENTS = 2000

RULES = {
    "draft_mode": dict(exclude_first=False, pool_seed=SEED, n_pools=N_POOLS, sets=None),
    "draft_mode2": dict(exclude_first=True, pool_seed=SECOND_SEED, n_pools=SECOND_N_POOLS, sets=SECOND_SETS),
}


def first_test_pools(eligible: list[str], set_code: str, fmt: str) -> list[str]:
    """The first test's draw for one set x format: exactly `cmd_pools`' default call (N_POOLS pools, seed SEED)."""
    return draw_pools(eligible, N_POOLS, pool_seed(SEED, set_code, fmt))


def draw_new_pools(eligible: list[str], first: list[str], set_code: str, fmt: str, k: int = SECOND_N_POOLS, seed: int = SECOND_SEED) -> list[str]:
    """`k` pools at random (seed `seed`) from the eligible pools that are NOT in `first`. Zero overlap with `first` is asserted."""
    used = {str(d) for d in first}
    rest = [d for d in eligible if str(d) not in used]
    pick = draw_pools(rest, k, pool_seed(seed, set_code, fmt))
    assert not ({str(d) for d in pick} & used), "new pools overlap the first test's pools"
    return pick


def first_g1_events(ho_ids: list, max_events: int = FIRST_G1_MAX_EVENTS, seed: int = FIRST_G1_SEED) -> list:
    """The held-out events the first G1 used: `deviation_cost.py`'s own choice (`rng = default_rng(0)`, `max_events` among the held-out events
    in `design.event_starts` order, sorted; all of them when there are not more than `max_events`)."""
    rng = np.random.default_rng(seed)
    if len(ho_ids) > max_events:
        return sorted(rng.choice(ho_ids, max_events, replace=False).tolist())
    return list(ho_ids)


def new_g1_events(ho_ids: list, first: list, max_events: int = SECOND_G1_MAX_EVENTS, seed: int = SECOND_SEED) -> list:
    """Up to `max_events` held-out events that are NOT in `first`, chosen with `default_rng(seed)` from the remaining ones (sorted first, so the
    choice does not depend on the table order). Returned sorted. Zero overlap with `first` is asserted."""
    used = {str(d) for d in first}
    rest = sorted(d for d in ho_ids if str(d) not in used)
    if len(rest) > max_events:
        rest = sorted(np.random.default_rng(seed).choice(rest, max_events, replace=False).tolist())
    assert not ({str(d) for d in rest} & used), "new G1 events overlap the first G1's events"
    return rest


def compare_first_pools(recomputed: dict, saved_json: Path) -> dict:
    """Compare the recomputed first-test draw ({"SET fmt": [draft_ids]}) with the first test's saved pools.json. Cells of the saved file for
    sets that are not in `recomputed` are ignored. Returns {"SET fmt": {"same": bool, "only_recomputed": [...], "only_saved": [...]}}."""
    saved: dict = {}
    for e in _jload(saved_json)["pools"]:
        saved.setdefault(f'{e["set"]} {e["fmt"]}', set()).add(str(e["draft_id"]))
    out = {}
    for cell, ids in recomputed.items():
        a, b = {str(d) for d in ids}, saved.get(cell, set())
        out[cell] = dict(same=a == b, only_recomputed=sorted(a - b), only_saved=sorted(b - a))
    return out


# ---- the decision lines -------------------------------------------------------------------------------------------------------------

def draft_records(rows: list[dict]) -> list[dict]:
    """One record per (pool, searching half): gap_in / gap_out of the draft-built deck minus the Sealed-built deck (pp, converted at the level
    of the Sealed-built deck under the searching model), plus the composition of both decks. Directions that were skipped have no record."""
    recs = sh.pool_direction_records(rows, ("draft",), "sealed")
    comp = {}
    for r in rows:
        for d, byv in r["dirs"].items():
            if "draft" in byv and "sealed" in byv:
                comp[(f'{r["set"]}|{r["fmt"]}|{r["draft_id"]}', d)] = (byv["draft"], byv["sealed"])
    out = []
    for x in recs:
        dd, ss = comp[(x["pool"], x["dir"])]
        out.append(dict(x, n_cre_d=dd.get("n_creatures"), n_cre_s=ss.get("n_creatures"), n_land_d=dd.get("n_lands"), n_land_s=ss.get("n_lands")))
    return out


def decision(recs: list[dict], n_boot: int = 2000, seed: int = 0) -> dict:
    """The two lines of RULE.md. Line 1: the pooled mean gap_out over all pool-directions, 95% cluster bootstrap over pools (both directions
    of a pool in one cluster), lower bound > 0. Line 2: neither draft format is 'harmed' (gap_out CI upper < 0 AND mean <= -0.05 pp).
    Verdict PASS only if both PASS; INCONCLUSIVE if there are no records."""
    overall = sh.summarise_records(recs, n_boot, seed) if recs else {}
    lo = overall["gap_out_ci"][0] if overall else float("nan")
    l1 = dict(n=overall.get("n", 0), n_pools=overall.get("n_pools", 0), mean=overall.get("mean_gap_out"),
              ci=overall.get("gap_out_ci"), need="CI lower bound > 0",
              status="INCONCLUSIVE" if not np.isfinite(lo) else "PASS" if lo > 0 else "FAIL")
    fm, bad = {}, []
    for f in DRAFT_FORMATS:
        s = sh.summarise_records([r for r in recs if r["fmt"] == f], n_boot, seed)
        if not s:
            fm[f] = dict(status="n/a (no pools)")
            continue
        harmed = bool(s["gap_out_ci"][1] < HARM_UPPER and s["mean_gap_out"] <= HARM_MEAN)
        fm[f] = dict(n=s["n"], n_pools=s["n_pools"], mean=s["mean_gap_out"], ci=s["gap_out_ci"], status="FAIL" if harmed else "PASS")
        if harmed:
            bad.append(f)
    l2 = dict(formats=fm, need=f"no format with CI upper < {HARM_UPPER} and mean <= {HARM_MEAN} pp",
              status="FAIL" if bad else "PASS" if any(v["status"] == "PASS" for v in fm.values()) else "INCONCLUSIVE")
    if l1["status"] == "INCONCLUSIVE":
        verdict = "INCONCLUSIVE"
    elif l1["status"] == "PASS" and l2["status"] == "PASS":
        verdict = "PASS"
    else:
        verdict = "FAIL"
    return dict(line1=l1, line2=l2, verdict=verdict)


def _mean(recs, key):
    v = [r[key] for r in recs if r.get(key) is not None]
    return float(np.mean(v)) if v else None


def cell_summary(recs: list[dict], n_boot: int = 2000, seed: int = 0) -> dict:
    """A reported table row for any selection of records."""
    s = sh.summarise_records(recs, n_boot, seed)
    if not s:
        return {}
    pools = {r["pool"] for r in recs}
    changed = {r["pool"] for r in recs if not r["same_deck"]}
    s.update(out_minus_in=s["mean_gap_out"] - s["mean_gap_in"], share_pools_changed=len(changed) / len(pools),
             share_dirs_changed=1 - s["share_same_deck"], cre_draft=_mean(recs, "n_cre_d"), cre_sealed=_mean(recs, "n_cre_s"),
             land_draft=_mean(recs, "n_land_d"), land_sealed=_mean(recs, "n_land_s"))
    return s


def full_report(rows: list[dict], n_boot: int = 2000, seed: int = 0) -> dict:
    recs = draft_records(rows)
    out: dict = dict(n_rows=len(rows), n_records=len(recs), decision=decision(recs, n_boot, seed), overall=cell_summary(recs, n_boot, seed))
    out["per_format"] = {f: cell_summary([r for r in recs if r["fmt"] == f], n_boot, seed) for f in DRAFT_FORMATS}
    out["per_cell"] = {f"{s} {f}": cell_summary([r for r in recs if r["set"] == s and r["fmt"] == f], n_boot, seed)
                       for s, f in sorted({(r["set"], r["fmt"]) for r in recs})}
    out["skipped_directions"] = [dict(set=r["set"], fmt=r["fmt"], draft_id=r["draft_id"], dir=d, why=byv.get("skipped") or byv.get("error"))
                                 for r in rows for d, byv in r["dirs"].items() if "draft" not in byv or "sealed" not in byv]
    return out


# =====================================================================================================================
# files / progress
# =====================================================================================================================

def _jload(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def write_status(out: Path, **kw) -> None:
    p = out / "status.json"
    cur = _jload(p) if p.exists() else {}
    cur.update(kw, updated=time.strftime("%Y-%m-%d %H:%M:%S"))
    p.write_text(json.dumps(cur, indent=1), encoding="utf-8")


def ensure_splits(src: Path, out: Path, set_code: str, seed: int = SEED, force: bool = False) -> dict:
    p = out / "splits" / f"{set_code}.json"
    if p.exists() and not force:
        return _jload(p)
    sp = build_splits(src, set_code, seed)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(sp), encoding="utf-8")
    return sp


def cmd_cuts(src: Path, out: Path, sets: list[str], seed: int = SEED) -> dict:
    res = {}
    for s in sets:
        sp = ensure_splits(src, out, s, seed)
        res[s] = sp
        if sp.get("skipped"):
            print(f"cuts {s}: SKIPPED ({sp['skipped']})", flush=True)
        else:
            print(f"cuts {s}: formats {sp['formats']} cuts {sp['cuts']} train {sp['n_train']} holdout {sp['n_holdout']}", flush=True)
    return res


# =====================================================================================================================
# stage 1: the half fits
# =====================================================================================================================

def fit_half(set_code: str, half: str, src: Path, out: Path, n_boot: int, draft_boot: int, jobs: int | None, seed: int = SEED) -> dict:
    import polars as pl

    from sealed_lab.model import pipeline

    set_code = set_code.upper()
    sp = ensure_splits(src, out, set_code, seed)
    if sp.get("skipped"):
        raise SystemExit(f"{set_code} is skipped: {sp['skipped']}")
    keep = half_keep(sp, half)
    home = sh.home_of(out, set_code, half)
    dst_set = home / "sets" / set_code
    if dst_set.exists():
        shutil.rmtree(dst_set)
    dst_set.mkdir(parents=True)
    for sub in ("corpus", "cache"):
        shutil.copytree(src / "sets" / set_code / sub, dst_set / sub)          # a COPY, never a link
    shutil.copy(src / "sets" / set_code / "registry.json", dst_set / "registry.json")
    os.environ["SEALED_LAB_HOME"] = str(home)
    orig = pipeline.load_inputs

    def load_half(sc):
        inp = orig(sc)
        inp.games = inp.games.filter(pl.col("draft_id").cast(pl.Utf8).is_in(keep))
        inp.builds = inp.builds.filter(pl.col("draft_id").cast(pl.Utf8).is_in(keep))
        return inp

    log: list = []
    t0 = time.time()

    def prog(fr, msg):
        if not log or log[-1][2] != msg:
            log.append([round(time.time() - t0, 1), round(float(fr), 4), str(msg)])

    pipeline.load_inputs = load_half
    try:
        meta = pipeline.calculate(set_code, progress=prog, n_jobs=jobs, n_boot=n_boot, recency="off", draft_formats=True,
                                  draft_boot=draft_boot, **sh.MODEL_OPTS)
    finally:
        pipeline.load_inputs = orig
    secs = time.time() - t0
    games = pl.read_parquet(dst_set / "corpus" / "games.parquet", columns=["draft_id", "fmt", "event"]).filter(
        pl.col("draft_id").cast(pl.Utf8).is_in(keep))
    n_games = {f"{k}": int(games.filter((pl.col("fmt") == "draft") & (pl.col("event") == ev)).height) for k, ev in
               (("dbo1", "PremierDraft"), ("dbo3", "TradDraft"))}
    n_games["sealed"] = int(games.filter(pl.col("fmt").is_in(["bo1", "bo3"])).height)
    n_games["other_draft"] = int(games.height - sum(n_games.values()))
    dlog = [e for e in log if "draft" in e[2].lower()]
    rec = dict(set=set_code, half=half, fit_seconds=secs, n_boot=n_boot, draft_boot=draft_boot, formats=meta.get("formats"),
               n_events=dict(sealed=len(sp["train"]["sealed"]), other_draft=len(sp["train"]["other"]),
                             **{f: len(sp[half][f]) for f in DRAFT_FORMATS}),
               n_games=n_games, cuts=sp["cuts"], meta_seconds=meta.get("seconds"),
               draft_part_seconds_approx=(dlog[-1][0] - dlog[0][0]) if len(dlog) > 1 else None,
               validation={f: v for f, v in (meta.get("validation") or {}).items()},
               draft_formats=meta.get("draft_formats"), n_games_meta=meta.get("n_games"),
               warnings=[w for w in (meta.get("warnings") or []) if "draft" in str(w).lower() or "dbo" in str(w)],
               progress_log=log[-400:], model_opts={**sh.MODEL_OPTS})
    p = out / "fits" / f"{set_code}_{half}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
    print(f"fit {set_code} {half}: {secs:.0f}s (draft part ~{rec['draft_part_seconds_approx']}), events {rec['n_events']}, formats {rec['formats']}", flush=True)
    return rec


# =====================================================================================================================
# pools
# =====================================================================================================================

def cmd_pools(src: Path, out: Path, sets: list[str], seed: int = SEED, n_pools: int = N_POOLS, exclude_first: bool = False,
              pool_seed_value: int | None = None, first_pools_json: Path | None = None) -> dict:
    """OUT/pools.json, written once before any search. Default (the first test): `n_pools` per set x format, `pool_seed(seed, ...)`.
    `exclude_first` (the second test, docs/draft_mode2/RULE.md): the first test's draw is recomputed (N_POOLS, SEED), excluded, and `n_pools`
    pools are drawn from the rest with `pool_seed_value` (default `seed`); the halves keep `seed`. Zero overlap is asserted per cell, the
    recomputed first draw is saved in pools.json (`first_test_pools`), and compared with `first_pools_json` (the first test's saved file) if given."""
    import polars as pl

    path = out / "pools.json"
    if path.exists():
        raise SystemExit(f"{path} exists: the pool list is fixed before any search and is never overwritten")
    ps = seed if pool_seed_value is None else pool_seed_value
    entries, cells, first_all = [], {}, {}
    for s in sets:
        sp = ensure_splits(src, out, s, seed)
        if sp.get("skipped"):
            continue
        games = pl.read_parquet(src / "sets" / s / "corpus" / "games.parquet", columns=["draft_id", "fmt", "event", "game_time", "draft_time"])
        ev = event_table(games)
        cuts = cuts_of(games)          # the cuts as datetime objects (the splits file stores their text); same function, same value
        pools = pl.read_parquet(src / "sets" / s / "corpus" / "pools.parquet", columns=["draft_id", "n"])
        for f in sp["formats"]:
            el = eligible_pools(pools, ev, cuts, f)
            if exclude_first:
                first = first_test_pools(el, s, f)
                pick = draw_new_pools(el, first, s, f, n_pools, ps)
                first_all[f"{s} {f}"] = first
                cells[f"{s} {f}"] = dict(eligible=len(el), first_test=len(first), eligible_after_exclusion=len(el) - len(first), drawn=len(pick))
            else:
                pick = draw_pools(el, n_pools, pool_seed(ps, s, f))
                cells[f"{s} {f}"] = dict(eligible=len(el), drawn=len(pick))
            entries += [dict(set=s, fmt=f, draft_id=d, logit_default=None) for d in pick]
    doc = dict(pools=entries, cells=cells, seed=ps, halves_seed=seed, n_per_cell=n_pools, min_cards=MIN_POOL_CARDS)
    if exclude_first:
        mine = {(e["set"], e["fmt"], str(e["draft_id"])) for e in entries}
        both = [k for c, ids in first_all.items() for k in [(*c.split(" "), str(d)) for d in ids] if k in mine]
        assert not both, f"overlap with the first test's pools: {both[:5]}"
        doc.update(exclude_first=True, first_test_pools=first_all, overlap_with_first_test=0)
        if first_pools_json is not None:
            doc["first_test_comparison"] = compare_first_pools(first_all, first_pools_json)
            diff = {c: v for c, v in doc["first_test_comparison"].items() if not v["same"]}
            print(f"first test's saved pools.json vs the recomputed draw: " + ("identical in every cell" if not diff else f"DIFFERENT in {sorted(diff)}"), flush=True)
    path.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print(f"pools: {len(entries)} pools in {len(cells)} cells -> {path}" + (" (first test's pools excluded, overlap 0)" if exclude_first else ""), flush=True)
    return dict(pools=entries, cells=cells)


# =====================================================================================================================
# stage 2: the searches
# =====================================================================================================================

def load_tasks(src: Path, pools_json: Path) -> list[dict]:
    """Worker tasks of pools.json: the pool of each draft_id from SRC's pools.parquet."""
    import polars as pl

    entries = _jload(pools_json)["pools"]
    by_set: dict = {}
    for e in entries:
        by_set.setdefault(e["set"], []).append(e)
    tasks = []
    for s, es in sorted(by_set.items()):
        pools = pl.read_parquet(src / "sets" / s / "corpus" / "pools.parquet", columns=["draft_id", "card_id", "n"])
        want = {str(e["draft_id"]): e for e in es}
        sub = pools.filter(pl.col("draft_id").cast(pl.Utf8).is_in(list(want)))
        for key, p in sub.partition_by("draft_id", as_dict=True, include_key=True).items():
            did = str(key[0] if isinstance(key, tuple) else key)
            tasks.append(dict(set=s, fmt=want[did]["fmt"], draft_id=did,
                              pool={int(c): int(n) for c, n in zip(p["card_id"].to_list(), p["n"].to_list())}))
    if len(tasks) != len(entries):
        print(f"WARNING: {len(entries) - len(tasks)} pools of {pools_json} were not found in the corpus", flush=True)
    tasks.sort(key=lambda t: (t["set"], t["fmt"], t["draft_id"]))
    return tasks


def _pool_task(task: dict) -> dict:
    """One pool: for each half as the searching model, the draft-built and the Sealed-built top deck and their draw-0 logits under both
    models in the draft format."""
    from search_check import _deck_of

    from sealed_lab.deck import builder

    sc, fmt, pool = task["set"], task["fmt"], task["pool"]
    sfmt = SEALED_OF[fmt]
    models = {h: sh._model(task["homes"][h], sc) for h in HALVES}
    out = dict(set=sc, fmt=fmt, draft_id=task["draft_id"], dirs={})
    for s in HALVES:
        o = OTHER[s]
        missing = [f"model {s} lacks {f}" for f in (fmt, sfmt) if f not in models[s].formats]
        missing += [f"model {o} lacks {fmt}"] if fmt not in models[o].formats else []
        if missing:
            out["dirs"][s] = dict(skipped="; ".join(missing))
            continue
        ms = models[s]
        ca = ms.card_arrays
        byv = {}
        try:
            for name, bf in (("draft", fmt), ("sealed", sfmt)):
                t0 = time.time()
                with sh._home(task["homes"][s]):
                    res = builder.build(ms, pool, bf, n_decks=1, families=True, n_jobs=1)
                secs = time.time() - t0
                deck = _deck_of(res)
                with sh._home(task["homes"][s]):
                    ls = float(ms.logit(deck, fmt))
                with sh._home(task["homes"][o]):
                    lo = float(models[o].logit(deck, fmt))
                byv[name] = dict(ls=ls, lo=lo, seconds=secs, build_fmt=bf, deck={str(k): int(c) for k, c in sorted(deck.items())},
                                 n_creatures=int(sum(c for i, c in deck.items() if ca.is_creature[i] and not ca.is_land[i])),
                                 n_lands=int(sum(c for i, c in deck.items() if ca.is_land[i])),
                                 n_candidates=(res.family_meta or {}).get("n_candidates"))
        except Exception as e:  # noqa: BLE001 - one unbuildable pool must not stop the run; it is listed in the report
            out["dirs"][s] = dict(error=f"{type(e).__name__}: {e}")
            continue
        out["dirs"][s] = byv
    return out


def _key(r: dict) -> tuple:
    return (r["set"], r["fmt"], str(r["draft_id"]))


def run_searches(src: Path, out: Path, jobs: int, prog: sh.Progress | None = None, deadline: float | None = None) -> None:
    tasks = load_tasks(src, out / "pools.json")
    need = sorted({(t["set"], h) for t in tasks for h in HALVES})
    miss = [f"{s}_{h}" for s, h in need if not (sh.home_of(out, s, h) / "sets" / s / "model" / "meta.json").exists()]
    if miss:
        raise SystemExit(f"fits missing for: {', '.join(miss)}")
    for t in tasks:
        t["homes"] = {h: str(sh.home_of(out, t["set"], h)) for h in HALVES}
    rows_path = out / "search_rows.jsonl"
    done_rows = [json.loads(x) for x in rows_path.read_text(encoding="utf-8").splitlines() if x.strip()] if rows_path.exists() else []
    done = {_key(r) for r in done_rows}
    todo = [t for t in tasks if _key(t) not in done]
    per_set = {s: sum(1 for t in tasks if t["set"] == s) for s in {t["set"] for t in tasks}}
    count = {s: sum(1 for r in done_rows if r["set"] == s) for s in per_set}
    finished = {s for s in per_set if count[s] >= per_set[s]}
    print(f"search: {len(tasks)} pools, {len(done)} done, {len(todo)} to do, {jobs} workers", flush=True)
    t0 = time.time()

    def record(t, r):
        with open(rows_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(r) + "\n")
        count[t["set"]] += 1
        if prog:
            prog.set_status(f"searching: {sum(count.values())}/{len(tasks)} pools done ({t['set']} {count[t['set']]}/{per_set[t['set']]}), "
                            f"{(time.time() - t0) / 60:.0f} min in this run")
            if count[t["set"]] >= per_set[t["set"]] and t["set"] not in finished:
                finished.add(t["set"])
                prog.done(f"search {len(finished)}/{len(per_set)} sets done ({t['set']})")

    if jobs <= 1:
        for t in todo:
            if deadline and time.time() > deadline:
                break
            record(t, _pool_task(t))
    else:
        with cf.ProcessPoolExecutor(max_workers=jobs, mp_context=multiprocessing.get_context("spawn")) as ex:
            it = iter(todo)
            pending: dict = {}

            def submit_more():
                while len(pending) < jobs * 2:
                    t = next(it, None)
                    if t is None or (deadline and time.time() > deadline):
                        return
                    pending[ex.submit(_pool_task, t)] = t

            submit_more()
            while pending:
                fin, _ = cf.wait(list(pending), return_when=cf.FIRST_COMPLETED)
                for fu in fin:
                    record(pending.pop(fu), fu.result())
                submit_more()
    print(f"search: wall {(time.time() - t0) / 60:.1f} min", flush=True)


# =====================================================================================================================
# report
# =====================================================================================================================

def _f(x, nd=2):
    return sh._f(x, nd)


def _ci(c, nd=2):
    return "-" if not c else f"[{_f(c[0], nd)}, {_f(c[1], nd)}]"


def load_rows(out: Path) -> list[dict]:
    p = out / "search_rows.jsonl"
    rows = {}
    for x in p.read_text(encoding="utf-8").splitlines():
        if x.strip():
            r = json.loads(x)
            rows[_key(r)] = r
    return [rows[k] for k in sorted(rows)]


def holdout_quality(fits: dict) -> list[dict]:
    """The draft models' own hold-out (the last 25% of the half's training events, inside each half fit): log loss, AUC, gain over the B1
    baseline, calibration slope and level, per set, half and draft format."""
    out = []
    for k, m in sorted(fits.items()):
        for f in DRAFT_FORMATS:
            v = (m.get("validation") or {}).get(f)
            if not v:
                continue
            cal = v.get("calibration") or {}
            out.append(dict(set=m["set"], half=m["half"], fmt=f, status=v.get("status"), logloss=v.get("model_logloss"), auc=v.get("model_auc"),
                            gain_vs_B1=v.get("gain_vs_B1"), gain_ci=v.get("ci"), slope=v.get("calibration_slope"), slope_ci=cal.get("slope_ci"),
                            level_pp=cal.get("level_pp"), level_pp_ci=cal.get("level_pp_ci"), reason=v.get("reason")))
    return out


def markdown(rep: dict, fits: dict, quality: list[dict], pools_meta: dict, splits: dict, wall: float | None) -> str:
    dec = rep["decision"]
    l1, l2 = dec["line1"], dec["line2"]
    L = ["# Draft mode: split-half deck choice, draft model against the Sealed model of the matching queue (docs/draft_mode/RULE.md)", "",
         "Per set and draft format (dbo1 = PremierDraft, dbo3 = TradDraft), held-out draft pools are searched with the draft model of one half "
         "and with the Sealed model (bo1 for dbo1, bo3 for dbo3) of the same half, both fitted on the same draft events. Both top decks are "
         "scored in the draft format by the OTHER half's draft model (gap_out) and by the searching model (gap_in). Gap = draft-built deck minus "
         "Sealed-built deck, pp of game win rate, draw 0, logit differences converted at the level of the Sealed-built deck under the searching "
         "model; one record per pool and searching model; 95% cluster bootstraps over pools (both directions of a pool in one cluster).", "",
         "## Decision", "",
         f"1. Pooled mean gap_out over {l1['n']} pool-directions of {l1['n_pools']} pools: {_f(l1['mean'])} pp {_ci(l1['ci'])} (need: lower bound > 0): **{l1['status']}**"]
    fm = "; ".join(f"{f}: " + (f"n {v['n']} ({v['n_pools']} pools), mean {_f(v['mean'])} {_ci(v['ci'])} {v['status']}" if "n" in v else v["status"])
                   for f, v in l2["formats"].items())
    L += [f"2. No draft format harmed ({l2['need']}): {fm}: **{l2['status']}**", "",
          f"**Verdict: {dec['verdict']}**" + ("" if dec["verdict"] != "PASS" else "  (the primary test passes; G1, G2 still to run)"), ""]
    L += ["## Per cell (reported, not deciding)", "",
          "| cell | pools | pool-directions | gap_in | gap_out [95% CI] | out - in | out / in | pools with a different deck | "
          "creatures draft / Sealed deck | lands draft / Sealed deck | s per build draft / Sealed |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|"]

    def row(name, s):
        if not s:
            return f"| {name} | 0 | 0 | - | - | - | - | - | - | - | - |"
        return (f"| {name} | {s['n_pools']} | {s['n']} | {_f(s['mean_gap_in'])} | {_f(s['mean_gap_out'])} {_ci(s['gap_out_ci'])} | "
                f"{_f(s['out_minus_in'])} | {_f(s['ratio'])} | {100 * s['share_pools_changed']:.0f}% | "
                f"{_f(s['cre_draft'], 1)} / {_f(s['cre_sealed'], 1)} | {_f(s['land_draft'], 1)} / {_f(s['land_sealed'], 1)} | "
                f"{s['mean_seconds']:.0f} / {s['mean_seconds_default']:.0f} |")
    L.append(row("ALL", rep["overall"]))
    for f in DRAFT_FORMATS:
        L.append(row(f, rep["per_format"].get(f)))
    for k, s in rep["per_cell"].items():
        L.append(row(k, s))
    if rep["skipped_directions"]:
        L += ["", f"{len(rep['skipped_directions'])} pool-directions were skipped (no model / build error): " +
              "; ".join(f"{x['set']} {x['fmt']} {x['draft_id']} {x['dir']}: {x['why']}" for x in rep["skipped_directions"][:10])]
    L += ["", "## Draft models' own hold-out (inside each half fit; reported, not deciding)", "",
          "| set | half | fmt | log loss | AUC | gain vs B1 [95% CI] | calibration slope [95% CI] | level pp [95% CI] |", "|---|---|---|---:|---:|---|---|---|"]
    for q in quality:
        if q["status"] != "ok":
            L.append(f"| {q['set']} | {q['half']} | {q['fmt']} | {q['status']}: {q.get('reason')} | | | | |")
            continue
        L.append(f"| {q['set']} | {q['half']} | {q['fmt']} | {_f(q['logloss'], 4)} | {_f(q['auc'], 3)} | {_f(q['gain_vs_B1'], 4)} {_ci(q['gain_ci'], 4)} | "
                 f"{_f(q['slope'])} {_ci(q['slope_ci'])} | {_f(q['level_pp'])} {_ci(q['level_pp_ci'])} |")
    L += ["", "## Fits", "", "| set | half | Sealed events | other draft events | dbo1 events | dbo3 events | fit s | draft part s (approx.) | formats |",
          "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for k, m in sorted(fits.items()):
        e = m["n_events"]
        L.append(f"| {m['set']} | {m['half']} | {e['sealed']} | {e['other_draft']} | {e['dbo1']} | {e['dbo3']} | {m['fit_seconds']:.0f} | "
                 f"{_f(m.get('draft_part_seconds_approx'), 0)} | {', '.join(m.get('formats') or [])} |")
    if pools_meta.get("cells"):
        L += ["", "## Pool cells (eligible = held-out events with >= 40 pool cards)", "", "| cell | eligible | drawn |", "|---|---:|---:|"]
        for k, v in pools_meta["cells"].items():
            L.append(f"| {k} | {v['eligible']} | {v['drawn']} |")
    skipped = {s: sp["skipped"] for s, sp in splits.items() if sp.get("skipped")}
    if skipped:
        L += ["", "## Sets left out", ""] + [f"- {s}: {why}" for s, why in sorted(skipped.items())]
    if wall is not None:
        L += ["", f"Search wall time {wall / 60:.0f} min."]
    return "\n".join(L) + "\n"


def write_report(out: Path, n_boot: int = 2000, wall: float | None = None) -> str:
    rows = load_rows(out)
    rep = full_report(rows, n_boot)
    fits = {p.stem: _jload(p) for p in sorted((out / "fits").glob("*.json"))} if (out / "fits").exists() else {}
    quality = holdout_quality(fits)
    pools_meta = _jload(out / "pools.json") if (out / "pools.json").exists() else {}
    splits = {p.stem: {k: v for k, v in _jload(p).items() if k in ("set", "skipped", "formats", "cuts", "n_train", "n_holdout")}
              for p in sorted((out / "splits").glob("*.json"))} if (out / "splits").exists() else {}
    (out / "draft_split_half.json").write_text(json.dumps(dict(report=rep, holdout_quality=quality, splits=splits,
                                                              fits={k: {a: b for a, b in v.items() if a != "progress_log"} for k, v in fits.items()},
                                                              pools=pools_meta.get("cells")), indent=1, default=str), encoding="utf-8")
    md = markdown(rep, fits, quality, pools_meta, splits, wall)
    (out / "draft_split_half.md").write_text(md, encoding="utf-8")
    return md


# =====================================================================================================================
# driver
# =====================================================================================================================

def run_all(src: Path, out: Path, sets: list[str], n_boot: int, draft_boot: int, jobs: int, max_hours: float, seed: int = SEED,
            pools_kw: dict | None = None) -> int:
    t_start = time.time()
    deadline = t_start + max_hours * 3600
    out.mkdir(parents=True, exist_ok=True)
    prog = sh.Progress(out, f"PLAN: cuts, {2 * len(sets)} fits ({len(sets)} sets x halves A,B), pools, search (up to {len(sets) * 2 * (pools_kw or {}).get('n_pools', N_POOLS)} pools x 2 directions x 2 builds), "
                            f"report; started {time.strftime('%Y-%m-%d %H:%M')}")
    prog.write()
    write_status(out, stage="cuts", sets=sets)
    splits = cmd_cuts(src, out, sets, seed)
    skipped = {s: sp["skipped"] for s, sp in splits.items() if sp.get("skipped")}
    use = [s for s in sets if s not in skipped]
    prog.done(f"cuts done; {len(use)} sets in the test" + (f"; skipped: {skipped}" if skipped else ""))
    units = [(s, h) for s in use for h in HALVES]
    for k, (s, h) in enumerate(units, 1):
        if (out / "fits" / f"{s}_{h}.json").exists() and (sh.home_of(out, s, h) / "sets" / s / "model" / "meta.json").exists():
            continue
        if time.time() > deadline:
            prog.done(f"STOPPED at the time limit before fit {k}/{len(units)}")
            write_status(out, stage="stopped (time limit)")
            return 2
        prog.set_status(f"fitting {k}/{len(units)} ({s} {h})")
        write_status(out, stage=f"fit {k}/{len(units)} ({s} {h})")
        (out / "logs").mkdir(exist_ok=True)
        with open(out / "logs" / f"fit_{s}_{h}.log", "w", encoding="utf-8") as lf:
            rc = subprocess.call([sys.executable, "-u", str(Path(__file__).resolve()), "fit", s, h, str(src), str(out), "--n-boot", str(n_boot),
                                  "--draft-boot", str(draft_boot), "--seed", str(seed), "--jobs", str(jobs)],
                                 stdout=lf, stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONUNBUFFERED="1"))
        if rc != 0:
            prog.done(f"FIT FAILED {s} {h} (rc {rc}), see logs/fit_{s}_{h}.log")
            write_status(out, stage=f"FIT FAILED {s} {h}")
            return 1
        fm = _jload(out / "fits" / f"{s}_{h}.json")
        prog.done(f"fit {sum(1 for u in units if (out / 'fits' / f'{u[0]}_{u[1]}.json').exists())}/{len(units)} done ({s} {h}, {fm['fit_seconds']:.0f}s)")
    if not (out / "pools.json").exists():
        cmd_pools(src, out, use, seed, **(pools_kw or {}))
        prog.done("pools.json written (before any search)")
    write_status(out, stage="search")
    t0 = time.time()
    run_searches(src, out, jobs, prog, deadline)
    wall = time.time() - t0
    n_rows = len(load_rows(out)) if (out / "search_rows.jsonl").exists() else 0
    write_report(out, wall=wall)
    prog.done(f"report written ({n_rows} pools searched)")
    prog.set_status("finished" if time.time() <= deadline else "time limit reached: partial")
    write_status(out, stage="finished")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "cuts", "pools", "search"):
        p = sub.add_parser(name)
        p.add_argument("src")
        p.add_argument("out")
        p.add_argument("--seed", type=int, default=SEED)
        p.add_argument("--sets", nargs="*", default=None, help="restrict to these sets (default: the 11 sets; the 5 of the rule with --rule draft_mode2)")
        p.add_argument("--rule", default="draft_mode", choices=sorted(RULES),
                       help="draft_mode (default, unchanged): 10 pools per set x format, seed 0. draft_mode2 (docs/draft_mode2/RULE.md): exclude the first "
                            "test's pools (recomputed), draw 20 per set x format with seed 1 from the rest; default sets FRA HOB MSH SOS TMT")
        if name in ("run", "pools"):
            p.add_argument("--exclude-first", action="store_true", help="exclude the first test's pools (recomputed: 10 per cell, seed 0); implied by --rule draft_mode2")
            p.add_argument("--pool-seed", type=int, default=None, help="seed of the pool draw (default: the rule's; the halves keep --seed)")
            p.add_argument("--n-pools", type=int, default=None, help="pools per set x format (default: the rule's)")
            p.add_argument("--first-pools-json", default=None, help="the first test's saved pools.json: compared with the recomputed draw, differences reported")
        if name in ("run", "search"):
            p.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
        if name == "run":
            p.add_argument("--n-boot", type=int, default=4)
            p.add_argument("--draft-boot", type=int, default=4)
            p.add_argument("--max-hours", type=float, default=40.0)
    p = sub.add_parser("fit")
    p.add_argument("set")
    p.add_argument("half", choices=HALVES)
    p.add_argument("src")
    p.add_argument("out")
    p.add_argument("--n-boot", type=int, default=4)
    p.add_argument("--draft-boot", type=int, default=4)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--jobs", type=int, default=None)
    p = sub.add_parser("report")
    p.add_argument("out")
    a = ap.parse_args(argv)
    if a.cmd == "fit":
        fit_half(a.set, a.half, Path(a.src).resolve(), Path(a.out).resolve(), a.n_boot, a.draft_boot, a.jobs, a.seed)
        return 0
    if a.cmd == "report":
        print(write_report(Path(a.out).resolve()))
        return 0
    src, out = Path(a.src).resolve(), Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rule = RULES[getattr(a, "rule", "draft_mode")]
    sets = [s.upper() for s in a.sets] if a.sets else (rule["sets"] or SETS)
    pools_kw = dict(n_pools=getattr(a, "n_pools", None) or rule["n_pools"], pool_seed_value=rule["pool_seed"] if getattr(a, "pool_seed", None) is None else a.pool_seed,
                    exclude_first=bool(rule["exclude_first"] or getattr(a, "exclude_first", False)),
                    first_pools_json=Path(a.first_pools_json).resolve() if getattr(a, "first_pools_json", None) else None)
    if a.cmd == "cuts":
        cmd_cuts(src, out, sets, a.seed)
        return 0
    if a.cmd == "pools":
        cmd_pools(src, out, [s for s in sets if not ensure_splits(src, out, s, a.seed).get("skipped")], a.seed, **pools_kw)
        return 0
    if a.cmd == "search":
        run_searches(src, out, a.jobs)
        return 0
    return run_all(src, out, sets, a.n_boot, a.draft_boot, a.jobs, a.max_hours, a.seed, pools_kw)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
