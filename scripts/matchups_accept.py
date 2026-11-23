"""Acceptance checks before the card x opponent-colour matchups component (model MO) is switched on by the pooled rule.

    python scripts/matchups_accept.py cross  CUR_SH ON_SH OUT [--cur-rows search_rows_cur.jsonl] [--on-rows search_rows.jsonl] [--variant default] [--sets ...]
    python scripts/matchups_accept.py report OUT [--b-dir BDIR]                                    (matchups_accept.json / .md)

Criteria fixed BEFORE running (task of 2026-11-02; thresholds are not changed after seeing results):

Check A - deck choice under an independent half-model.  CUR_SH = the split-half tree of Session 10b (half models fitted with the
default options: matchups resolved by `auto`); ON_SH = a tree on the SAME event split and pools whose half models were fitted with
`matchups=on` (`split_half_search.py fit ... --model-opt matchups=on`).  For the 13 set-formats where the pooled rule would newly
switch matchups on (EOE bo1 bo3, FRA bo3, MSH, SOS, TDM, TLA, TMT bo1 bo3) every pool is searched with the shipped default search by
each of the four half models A_cur, B_cur, A_on, B_on, and every top deck is scored under all four.  Per pool and searching half X
(Y = the other half), at draw 0, in win-rate points:
    gap_out = score_Y(deck of X_on) - score_Y(deck of X_cur)       primary: score_Y = Y_on model; secondary: Y_cur model
(a win-rate difference under Y's own model; unlike split_half_search's gap there is no level conversion, because both decks are scored
by the same model).  gap_in = the same difference under X_on (what the matchups search itself sees).  Mean gap_out with 95% cluster
bootstrap over pools (both directions of a pool in one cluster), per set, per format and pooled; share of pool-directions (and of
pools in either direction) whose deck changes; the half-model disagreement T in the scale of split_half_decompose (T/2 in win-rate
points = the mean of the two models' preference for their own deck), with matchups on (A_on, B_on, their decks) and current.
PASS A: the pooled primary gap_out has a 95% CI upper bound >= 0.

Check B - deviation-cost guardrail.  `deviation_cost.py SET BDIR/<arm>/SET --refit --max-events 300` for arm `cur` (--matchups auto) and
`on` (--matchups on), same seed and events (FRA: --fmt bo3).  Per format, b is pooled over the sets by inverse variance (fixed effect,
weights from each arm's own b interval, as builder_scoreboard).  The saved bootstrap replicates `b_boots` are paired (same events, same
rng), so the replicate r of the pooled b of an arm is sum(w_i b_boots_i[r]) / sum(w_i) and
    paired difference = |b_pooled_on + 1| - |b_pooled_cur + 1|  (point, and 95% percentile interval over the replicates).
PASS B: in neither format the 95% interval of this difference lies entirely above 0.  Secondary (not a criterion): the inverse-variance
pooled cell-level differences of |b + 1| and of b (builder_scoreboard.pair).  Arms whose train-only `auto` fit already switched matchups
on coincide with their `on` arm (checked on the saved model's meta.json) and contribute a zero difference.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

# cells where the leave-one-set-out pooled rule switches matchups on and the per-set test leaves it off (POOLING_RULE.md, pooled.md)
SWITCH_CELLS = {"EOE": ("bo1", "bo3"), "FRA": ("bo3",), "MSH": ("bo1", "bo3"), "SOS": ("bo1", "bo3"), "TDM": ("bo1", "bo3"),
                "TLA": ("bo1", "bo3"), "TMT": ("bo1", "bo3")}
FORMATS = ("bo1", "bo3")
Z95 = 1.959964


def _load_module(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, m)
    spec.loader.exec_module(m)
    return m


def _sh():
    return sys.modules.get("split_half_search") or _load_module("split_half_search")


def _bs():
    return sys.modules.get("builder_scoreboard") or _load_module("builder_scoreboard")


def _pa(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, float)))


# =====================================================================================================================
# Check A: pure arithmetic
# =====================================================================================================================

def in_switch_cells(set_code: str, fmt: str) -> bool:
    return fmt in SWITCH_CELLS.get(set_code, ())


def pool_records(row: dict) -> tuple[list[dict], dict]:
    """One cross row -> ([pool-direction records], the pool's T record).

    row: {set, fmt, draft_id, decks: {"A_cur"|"B_cur"|"A_on"|"B_on": {card: count}}, L: {deckname: {modelname: logit}}} - the logit of
    every one of the four top decks under every one of the four half models (draw 0)."""
    L, decks = row["L"], row["decks"]
    pool = f'{row["set"]}|{row["fmt"]}|{row["draft_id"]}'
    recs = []
    for x, y in (("A", "B"), ("B", "A")):
        xon, xcur = f"{x}_on", f"{x}_cur"
        gap = lambda deck_hi, deck_lo, model: float(100.0 * (_pa(L[deck_hi][model]) - _pa(L[deck_lo][model])))
        recs.append(dict(set=row["set"], fmt=row["fmt"], pool=pool, dir=x,
                         gap_out=gap(xon, xcur, f"{y}_on"), gap_out_cur=gap(xon, xcur, f"{y}_cur"),
                         gap_in=gap(xon, xcur, xon), gap_in_cur=gap(xon, xcur, xcur),
                         changed=bool(decks[xon] != decks[xcur])))

    def pref(tag: str) -> float:
        a, b = f"A_{tag}", f"B_{tag}"
        # model A: own deck minus the other's deck; model B likewise; mean of the two (= T/2 in win-rate points)
        return float(100.0 * 0.5 * ((_pa(L[a][a]) - _pa(L[b][a])) + (_pa(L[b][b]) - _pa(L[a][b]))))

    trec = dict(set=row["set"], fmt=row["fmt"], pool=pool, T_cur=pref("cur"), T_on=pref("on"),
                changed_any=bool(any(r["changed"] for r in recs)))
    trec["T_diff"] = trec["T_on"] - trec["T_cur"]
    return recs, trec


def model_logit_shift(rows: list[dict]) -> float:
    """Largest |logit of a deck under a matchups-on half model - logit of the same deck under the current half model| over all pools,
    decks and halves (0 = the matchups layer does not enter the builder's draw-0 logit at all)."""
    m = 0.0
    for r in rows:
        for dn in r["L"]:
            for h in ("A", "B"):
                m = max(m, abs(r["L"][dn][f"{h}_on"] - r["L"][dn][f"{h}_cur"]))
    return float(m)


def summarise_gap(recs: list[dict], n_boot: int = 2000, seed: int = 0) -> dict:
    """Mean gap_out (primary and secondary) and gap_in with 95% cluster-bootstrap CIs over pools, shares of changed decks."""
    if not recs:
        return {}
    cb = _sh().cluster_boot
    cl = [r["pool"] for r in recs]
    ones = np.ones(len(recs))
    out = dict(n=len(recs), n_pools=len(set(cl)))
    for key in ("gap_out", "gap_out_cur", "gap_in", "gap_in_cur"):
        m = cb(np.array([r[key] for r in recs]), ones, cl, n_boot, seed)
        out[key] = dict(mean=m[0], ci=[m[1], m[2]])
    chg = np.array([r["changed"] for r in recs], float)
    out["share_changed"] = float(chg.mean())
    by_pool: dict = {}
    for r in recs:
        by_pool[r["pool"]] = by_pool.get(r["pool"], False) or bool(r["changed"])
    out["share_pools_changed"] = float(np.mean(list(by_pool.values())))
    go = np.array([r["gap_out"] for r in recs])
    out["share_out_pos"] = float((go > 1e-4).mean())
    out["share_out_neg"] = float((go < -1e-4).mean())
    return out


def summarise_T(trecs: list[dict], n_boot: int = 2000, seed: int = 0) -> dict:
    if not trecs:
        return {}
    cb = _sh().cluster_boot
    cl = [r["pool"] for r in trecs]
    ones = np.ones(len(trecs))
    out = dict(n_pools=len(trecs))
    for key in ("T_cur", "T_on", "T_diff"):
        m = cb(np.array([r[key] for r in trecs]), ones, cl, n_boot, seed)
        out[key] = dict(mean=m[0], ci=[m[1], m[2]])
    return out


def pass_a(pooled: dict) -> bool:
    """PASS A: the 95% CI upper bound of the pooled primary gap_out is >= 0 (matchups decks not significantly worse)."""
    hi = pooled.get("gap_out", {}).get("ci", [float("nan"), float("nan")])[1]
    return bool(np.isfinite(hi) and hi >= 0.0)


def check_a_summary(rows: list[dict], n_boot: int = 2000, seed: int = 0, both_on_sets=()) -> dict:
    """rows: cross rows; only the cells of SWITCH_CELLS are used. `both_on_sets`: sets whose two current half models already had
    matchups on (arms coincide there): a sensitivity pool without them is added."""
    recs, trecs = [], []
    for r in rows:
        if not in_switch_cells(r["set"], r["fmt"]):
            continue
        a, t = pool_records(r)
        recs += a
        trecs.append(t)
    out: dict = dict(n_pools=len(trecs), cells=sorted({f'{r["set"]} {r["fmt"]}' for r in recs}),
                     max_logit_shift=model_logit_shift([r for r in rows if in_switch_cells(r["set"], r["fmt"])]))
    out["pooled"] = dict(gap=summarise_gap(recs, n_boot, seed), T=summarise_T(trecs, n_boot, seed))
    out["by_format"] = {f: dict(gap=summarise_gap([r for r in recs if r["fmt"] == f], n_boot, seed),
                                T=summarise_T([t for t in trecs if t["fmt"] == f], n_boot, seed)) for f in FORMATS}
    out["per_set"] = {s: dict(gap=summarise_gap([r for r in recs if r["set"] == s], n_boot, seed),
                              T=summarise_T([t for t in trecs if t["set"] == s], n_boot, seed))
                      for s in sorted({r["set"] for r in recs})}
    out["per_cell"] = {f"{s} {f}": dict(gap=summarise_gap([r for r in recs if r["set"] == s and r["fmt"] == f], n_boot, seed),
                                        T=summarise_T([t for t in trecs if t["set"] == s and t["fmt"] == f], n_boot, seed))
                       for s in sorted({r["set"] for r in recs}) for f in FORMATS if any(r["set"] == s and r["fmt"] == f for r in recs)}
    both = set(both_on_sets)
    if both:
        keep = [r for r in recs if r["set"] not in both]
        keep_t = [t for t in trecs if t["set"] not in both]
        out["pooled_excl_both_on"] = dict(excluded=sorted(both), gap=summarise_gap(keep, n_boot, seed), T=summarise_T(keep_t, n_boot, seed))
    out["pass_a"] = pass_a(out["pooled"]["gap"])
    return out


# =====================================================================================================================
# Check A: scoring (needs the half models; scratch homes only)
# =====================================================================================================================

@contextlib.contextmanager
def _home(path):
    old = os.environ.get("SEALED_LAB_HOME")
    os.environ["SEALED_LAB_HOME"] = str(path)
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("SEALED_LAB_HOME", None)
        else:
            os.environ["SEALED_LAB_HOME"] = old


def _rows_by_key(path: Path, variant: str) -> dict:
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if all(variant in r["dirs"].get(h, {}) for h in ("A", "B")):
                out[(r["set"], r["fmt"], str(r["draft_id"]))] = r
    return out


def cross_rows(cur_sh: Path, on_sh: Path, cur_rows: Path, on_rows: Path, variant: str, sets=None, log=print) -> list[dict]:
    """Score the four top decks of every common pool under the four half models."""
    from sealed_lab.model.artifact import SetModel

    rc, ro = _rows_by_key(cur_rows, variant), _rows_by_key(on_rows, variant)
    keys = sorted(set(rc) & set(ro))
    want = {s.upper() for s in sets} if sets else None
    out = []
    for sc in sorted({k[0] for k in keys}):
        if want and sc not in want:
            continue
        homes = {f"{h}_{tag}": Path(sh) / "homes" / f"{sc}_{h}" for tag, sh in (("cur", cur_sh), ("on", on_sh)) for h in ("A", "B")}
        models = {}
        for name, home in homes.items():
            with _home(home):
                models[name] = SetModel.load(sc)
        for k in [k for k in keys if k[0] == sc]:
            fmt = k[1]
            decks = {}
            for tag, rows in (("cur", rc), ("on", ro)):
                for h in ("A", "B"):
                    d = rows[k]["dirs"][h][variant]["deck"]
                    decks[f"{h}_{tag}"] = {str(c): int(n) for c, n in sorted(d.items(), key=lambda kv: int(kv[0]))}
            L = {}
            for dn, deck in decks.items():
                di = {int(c): n for c, n in deck.items()}
                L[dn] = {}
                for mn, m in models.items():
                    with _home(homes[mn]):
                        L[dn][mn] = float(m.logit(di, fmt))
            # consistency with the stored search rows (own and cross logits of the searching model)
            chk = 0.0
            for tag, rows in (("cur", rc), ("on", ro)):
                for h, o in (("A", "B"), ("B", "A")):
                    e = rows[k]["dirs"][h][variant]
                    chk = max(chk, abs(e["ls"] - L[f"{h}_{tag}"][f"{h}_{tag}"]), abs(e["lo"] - L[f"{h}_{tag}"][f"{o}_{tag}"]))
            out.append(dict(set=sc, fmt=fmt, draft_id=k[2], decks=decks, L=L, chk_stored=chk))
        log(f"cross {sc}: {sum(1 for r in out if r['set'] == sc)} pools scored")
    return out


# =====================================================================================================================
# Check B: pure arithmetic
# =====================================================================================================================

def pooled_b(cells: list[dict]) -> dict:
    """Inverse-variance fixed-effect pooled b of cells {b, b_ci, b_boots?}: weights from each cell's own interval (builder_scoreboard.se_of),
    analytic CI, and (when every cell has replicates of one length) the pooled replicates and their percentile CI."""
    bs = _bs()
    b = np.array([c["b"] for c in cells], float)
    w = 1.0 / np.array([bs.se_of(c["b_ci"]) for c in cells]) ** 2
    est = float((w * b).sum() / w.sum())
    se = float(1.0 / math.sqrt(w.sum()))
    out = dict(k=len(cells), b=est, se=se, ci=[est - Z95 * se, est + Z95 * se], weights=w.tolist())
    boots = [c.get("b_boots") for c in cells]
    if all(x is not None for x in boots) and len({len(x) for x in boots}) == 1:
        reps = (w[:, None] * np.array(boots, float)).sum(0) / w.sum()
        out["reps"] = reps.tolist()
        out["ci_boot"] = [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))]
    return out


def paired_abs_dev(cur: list[dict], on: list[dict]) -> dict:
    """cur / on: aligned lists of cells (same sets, same events, replicates paired). Pooled b of each arm and the paired difference
    |b_pooled_on + 1| - |b_pooled_cur + 1| with its percentile interval over the paired replicates."""
    if len(cur) != len(on) or not cur:
        raise ValueError("arms must hold the same, non-empty list of cells")
    pc, po = pooled_b(cur), pooled_b(on)
    if "reps" not in pc or "reps" not in po or len(pc["reps"]) != len(po["reps"]):
        raise ValueError("paired replicates (b_boots) are missing or of different length")
    rc, ro = np.array(pc["reps"]), np.array(po["reps"])
    d = np.abs(ro + 1.0) - np.abs(rc + 1.0)
    point = abs(po["b"] + 1.0) - abs(pc["b"] + 1.0)
    ci = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
    db = ro - rc
    return dict(cur={k: v for k, v in pc.items() if k not in ("reps", "weights")}, on={k: v for k, v in po.items() if k not in ("reps", "weights")},
                abs_dev_cur=abs(pc["b"] + 1.0), abs_dev_on=abs(po["b"] + 1.0), diff_abs_dev=float(point), diff_abs_dev_ci=ci,
                diff_b=float(po["b"] - pc["b"]), diff_b_ci=[float(np.percentile(db, 2.5)), float(np.percentile(db, 97.5))],
                significantly_worse=bool(ci[0] > 0.0))


def cell_pairs(cur: dict, on: dict) -> list[dict]:
    """Per cell: both arms' b with CIs, the paired difference of b and of |b + 1| with percentile intervals; `identical` when the arms
    produced the same replicates (coinciding arms)."""
    rows = []
    for sc in sorted(set(cur) & set(on)):
        for fmt in FORMATS:
            fc, fo = cur[sc].get("formats", {}).get(fmt), on[sc].get("formats", {}).get(fmt)
            if not fc or not fo:
                continue
            same = fc.get("event_ids") is not None and fc.get("event_ids") == fo.get("event_ids")
            row = dict(set=sc, fmt=fmt, n_events=fc["n_events"], same_events=bool(same), b_cur=fc["b"], b_cur_ci=fc["b_ci"],
                       b_on=fo["b"], b_on_ci=fo["b_ci"], diff_b=fo["b"] - fc["b"],
                       diff_abs_dev=abs(fo["b"] + 1) - abs(fc["b"] + 1), identical=bool(fc.get("b_boots") == fo.get("b_boots")))
            bc, bo = fc.get("b_boots"), fo.get("b_boots")
            if same and bc and bo and len(bc) == len(bo):
                bc, bo = np.array(bc), np.array(bo)
                d = bo - bc
                da = np.abs(bo + 1) - np.abs(bc + 1)
                row["diff_b_ci"] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
                row["diff_abs_dev_ci"] = [float(np.percentile(da, 2.5)), float(np.percentile(da, 97.5))]
                row["diff_abs_dev_se"] = float(da.std(ddof=1))
            rows.append(row)
    return rows


def pooled_cell_abs_dev(rows: list[dict], fmt: str) -> dict:
    """Secondary: inverse-variance pool of the cell-level paired differences of |b + 1| (cells with a zero SE, i.e. coinciding arms, carry
    no information and are left out of the weights; their difference is exactly 0)."""
    sel = [r for r in rows if r["fmt"] == fmt and r.get("diff_abs_dev_se", 0.0) > 1e-9]
    if not sel:
        return {}
    d = np.array([r["diff_abs_dev"] for r in sel])
    w = 1.0 / np.array([r["diff_abs_dev_se"] for r in sel]) ** 2
    est = float((w * d).sum() / w.sum())
    se = float(1.0 / math.sqrt(w.sum()))
    return dict(k=len(sel), diff=est, ci=[est - Z95 * se, est + Z95 * se])


def pass_b(by_fmt: dict) -> bool:
    """PASS B: in no format the paired pooled difference of |b + 1| (on minus current) is significantly above zero."""
    return not any(v.get("significantly_worse") for v in by_fmt.values())


def load_arm(d: Path) -> dict:
    out = {}
    for p in sorted(Path(d).glob("*/deviation_cost.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        meta = p.parent / "home" / "sets" / r["set"] / "model" / "meta.json"
        try:
            m = json.loads(meta.read_text(encoding="utf-8"))
            r["matchups_applied"] = bool(m.get("matchups", {}).get("applied"))
        except Exception:
            r["matchups_applied"] = None
        out[r["set"]] = r
    return out


def check_b_summary(bdirs) -> dict:
    """bdirs: one directory (or a list) holding cur/SET and on/SET deviation-cost runs. Each directory is pooled on its own (runs with
    different numbers of bootstrap replicates, or event counts, cannot share replicates); `pass_b` needs every group to pass."""
    bdirs = [bdirs] if isinstance(bdirs, (str, Path)) else list(bdirs)
    groups, all_rows, auto_on, on_applied, fit_s, build_s = {}, [], {}, {}, {"cur": {}, "on": {}}, {"cur": {}, "on": {}}
    for d in bdirs:
        cur, on = load_arm(Path(d) / "cur"), load_arm(Path(d) / "on")
        rows = cell_pairs(cur, on)
        by_fmt = {}
        for fmt in FORMATS:
            sel = [r for r in rows if r["fmt"] == fmt and r["same_events"]]
            if not sel:
                continue
            res = paired_abs_dev([cur[r["set"]]["formats"][fmt] for r in sel], [on[r["set"]]["formats"][fmt] for r in sel])
            res["sets"] = [r["set"] for r in sel]
            res["n_boot"] = len(cur[sel[0]["set"]]["formats"][fmt]["b_boots"])
            res["secondary_cell_pool"] = pooled_cell_abs_dev(rows, fmt)
            by_fmt[fmt] = res
        name = Path(d).name
        groups[name] = dict(by_format=by_fmt, sets=sorted(cur), pass_b=pass_b(by_fmt) if by_fmt else None)
        all_rows += [dict(r, group=name) for r in rows]
        auto_on.update({s: cur[s].get("matchups_applied") for s in cur})
        on_applied.update({s: on[s].get("matchups_applied") for s in on})
        for arm, dd in (("cur", cur), ("on", on)):
            for s in dd:
                fit_s[arm][s] = dd[s].get("fit_seconds")
                build_s[arm][s] = {f: dd[s]["formats"][f].get("build_seconds") for f in dd[s].get("formats", {})}
    flags = [g["pass_b"] for g in groups.values() if g["pass_b"] is not None]
    return dict(cells=all_rows, groups=groups, auto_already_on=dict(sorted(auto_on.items())), on_applied=dict(sorted(on_applied.items())),
                fit_seconds=fit_s, build_seconds=build_s, pass_b=bool(all(flags)) if flags else None)


# =====================================================================================================================
# model artifacts: does forcing matchups on change anything the builder reads?
# =====================================================================================================================

# files of a model directory that legitimately differ between a matchups-off and a matchups-on fit: the matchup layer itself (opp.npz),
# descriptive archetype win-rate intervals (bootstrap draws) and the report/meta text. Anything else differing would reach the builder.
EXPECTED_DIFF = ("opp.npz", "archetypes.json", "report.json", "report.html", "meta.json")


def model_dir_diff(a: Path, b: Path) -> dict:
    """{file: 'identical' | 'DIFFERS' | 'DIFFERS in k1,k2' | 'only in A' | 'only in B' | '(not compared)'} for two model directories
    (npz arrays and parquet tables compared exactly; json files other than meta.json by content)."""
    import polars as pl

    a, b = Path(a), Path(b)
    out = {}
    for n in sorted({p.name for p in a.iterdir()} | {p.name for p in b.iterdir()}):
        pa, pb = a / n, b / n
        if not pa.exists() or not pb.exists():
            out[n] = "only in " + ("A" if pa.exists() else "B")
        elif n.endswith(".npz"):
            za, zb = np.load(pa, allow_pickle=True), np.load(pb, allow_pickle=True)
            bad = [k for k in sorted(set(za.files) | set(zb.files))
                   if k not in za.files or k not in zb.files or not np.array_equal(za[k], zb[k], equal_nan=za[k].dtype.kind == "f")]
            out[n] = "identical" if not bad else "DIFFERS in " + ",".join(bad)
        elif n.endswith(".parquet"):
            out[n] = "identical" if pl.read_parquet(pa).equals(pl.read_parquet(pb)) else "DIFFERS"
        elif n.endswith(".json") and n != "meta.json":
            out[n] = "identical" if json.loads(pa.read_text("utf-8")) == json.loads(pb.read_text("utf-8")) else "DIFFERS"
        else:
            out[n] = "(not compared)"
    return out


def unexpected_diffs(diff: dict) -> dict:
    """The entries of a `model_dir_diff` that differ although they are not in EXPECTED_DIFF (these would reach the builder)."""
    return {k: v for k, v in diff.items() if k not in EXPECTED_DIFF and v not in ("identical", "(not compared)")}


def model_diff_summary(pairs: dict[str, tuple[Path, Path]]) -> dict:
    """pairs: label -> (current model dir, matchups-on model dir)."""
    res = {k: model_dir_diff(a, b) for k, (a, b) in pairs.items()}
    return dict(n_pairs=len(res), n_unexpected=sum(1 for d in res.values() if unexpected_diffs(d)),
                unexpected={k: unexpected_diffs(d) for k, d in res.items() if unexpected_diffs(d)}, files=res)


# =====================================================================================================================
# report
# =====================================================================================================================

def _f(x, nd=2, sign=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def _ci(ci, nd=2, sign=True):
    return f"[{_f(ci[0], nd, sign)}, {_f(ci[1], nd, sign)}]"


def _mci(d, nd=2, sign=True):
    return f"{_f(d['mean'], nd, sign)} {_ci(d['ci'], nd, sign)}" if d else "-"


def markdown(a: dict | None, b: dict | None, meta: dict) -> str:
    L = ["# Acceptance checks before switching matchups (MO) on by the pooled rule", ""]
    if a and a.get("max_logit_shift", 1.0) < 1e-9:
        L += ["**Finding that frames both checks.** In every half model (14) and train-only model (7) compared, forcing matchups on changes "
              "only the matchup layer itself (opp.npz) and descriptive archetype intervals; the card values, reference distributions, structure "
              "and mana terms, and therefore the builder's draw-0 logit of any deck, are bit-identical to the matchups-off fit "
              "(`SetModel.logit` does not read the opponent layer; it is used by the matchup/sideboard advice only). So the builder picks the "
              "same deck and the deviation-cost slope b cannot move: both checks pass by construction and test nothing about the matchup "
              "advice itself.", ""]
    if a:
        L += ["## Check A: deck choice under an independent half-model", "",
              f"{a['n_pools']} pools in the {len(a['cells'])} cells where matchups would newly switch on ({', '.join(a['cells'])}). "
              "gap_out = score_Y(deck found by X with matchups) minus score_Y(deck found by X with the current options), win-rate pp at draw 0, "
              "X = A or B searching, Y = the other half; primary scored by Y's matchups-on model, secondary by Y's current model. "
              "Mean over pool-directions with 95% cluster bootstrap over pools. T = half-model disagreement in the scale of split_half_decompose "
              "(T/2 in win-rate points: how much more each model prefers its own deck than the other's), matchups on vs current, mean over pools.", "",
              "| cell | pool-dirs | gap_out primary [95% CI] | gap_out secondary [95% CI] | gap_in (X_on) | deck changes (dirs) | pools changed | T current | T on | T on - current [95% CI] |",
              "|---|---:|---|---|---:|---:|---:|---:|---:|---|"]

        def row(name, blk):
            g, t = blk["gap"], blk["T"]
            if not g:
                return
            L.append(f"| {name} | {g['n']} | {_mci(g['gap_out'])} | {_mci(g['gap_out_cur'])} | {_f(g['gap_in']['mean'], 2, True)} | "
                     f"{100 * g['share_changed']:.0f}% | {100 * g['share_pools_changed']:.0f}% | {_f(t['T_cur']['mean'])} | {_f(t['T_on']['mean'])} | "
                     f"{_mci(t['T_diff'])} |")

        row("**POOLED**", a["pooled"])
        for f in FORMATS:
            row(f"pooled {f}", a["by_format"][f])
        for s, blk in a["per_set"].items():
            row(s, blk)
        for c, blk in a["per_cell"].items():
            row(f"- {c}", blk)
        if "pooled_excl_both_on" in a:
            e = a["pooled_excl_both_on"]
            row(f"pooled excluding {', '.join(e['excluded'])} (current half models already on in both halves)", e)
        L += ["", f"Largest change of any deck's draw-0 logit between the current and the matchups-on half model (all pools, decks, halves): "
               f"{a['max_logit_shift']:.2e}.", "", f"**PASS A (pooled primary gap_out CI upper bound >= 0): {'PASS' if a['pass_a'] else 'FAIL'}**", ""]
    if b:
        L += ["## Check B: deviation-cost guardrail", "",
              "Per cell, b with 95% bootstrap CI (cluster bootstrap over held-out events, model: -1) for the current (`--matchups auto`) and the "
              "`--matchups on` arm, same events. Pooled = inverse-variance fixed effect over the sets. The paired difference of |b + 1| is "
              "|b_pooled_on + 1| - |b_pooled_cur + 1| with a percentile interval over the paired bootstrap replicates (positive = on deviates "
              "more from -1).", "",
              "| cell | events | b current [CI] | b on [CI] | on - current b [paired CI] | on - current abs(b+1) [paired CI] | arms coincide |",
              "|---|---:|---|---|---|---|---|"]
        for r in b["cells"]:
            dci = _ci(r["diff_b_ci"]) if "diff_b_ci" in r else "n/a"
            aci = _ci(r["diff_abs_dev_ci"]) if "diff_abs_dev_ci" in r else "n/a"
            L.append(f"| {r['set']} {r['fmt']} ({r['group']}) | {r['n_events']} | {_f(r['b_cur'], 2, True)} {_ci(r['b_cur_ci'])} | {_f(r['b_on'], 2, True)} {_ci(r['b_on_ci'])} | "
                     f"{_f(r['diff_b'], 2, True)} {dci} | {_f(r['diff_abs_dev'], 2, True)} {aci} | {'yes' if r['identical'] else 'no'} |")
        L += ["", "| run | format | sets | pooled b current [95% CI] | pooled b on [95% CI] | abs(b+1) current | abs(b+1) on | on - current abs(b+1) [paired 95% CI, replicates] | "
              "on - current b [paired CI] | secondary: cell-pooled abs(b+1) diff [CI] |", "|---|---|---:|---|---|---:|---:|---|---|---|"]
        for gname, g in b["groups"].items():
            for fmt, v in g["by_format"].items():
                sec = v.get("secondary_cell_pool") or {}
                L.append(f"| {gname} | {fmt} | {v['cur']['k']} ({', '.join(v['sets'])}) | {_f(v['cur']['b'], 3, True)} {_ci(v['cur']['ci'], 3)} | "
                         f"{_f(v['on']['b'], 3, True)} {_ci(v['on']['ci'], 3)} | {v['abs_dev_cur']:.3f} | {v['abs_dev_on']:.3f} | "
                         f"{_f(v['diff_abs_dev'], 3, True)} {_ci(v['diff_abs_dev_ci'], 3)} ({v['n_boot']}) | "
                         f"{_f(v['diff_b'], 3, True)} {_ci(v['diff_b_ci'], 3)} | "
                         f"{(_f(sec['diff'], 3, True) + ' ' + _ci(sec['ci'], 3)) if sec else '-'} |")
        L += ["", "Train-only `--matchups auto` arm already had matchups ON: " +
              (", ".join(f"{s}={'yes' if v else ('no' if v is not None else '?')}" for s, v in b["auto_already_on"].items())) + ".", "",
              f"**PASS B (no format with the paired pooled abs(b+1) difference significantly above 0): "
              f"{'PASS' if b['pass_b'] else 'FAIL'}**", ""]
    for k, v in meta.items():
        L.append(f"- {k}: {v}")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("cross")
    p.add_argument("cur_sh")
    p.add_argument("on_sh")
    p.add_argument("out")
    p.add_argument("--cur-rows", default="search_rows_cur.jsonl", help="rows file under ON_SH of the current half models searched with today's code")
    p.add_argument("--on-rows", default="search_rows.jsonl", help="rows file under ON_SH of the matchups-on half models")
    p.add_argument("--variant", default="default")
    p.add_argument("--sets", nargs="*", default=None)
    p = sub.add_parser("modeldiff")
    p.add_argument("out_json")
    p.add_argument("pairs", nargs="+", help="LABEL=CURDIR,ONDIR")
    p = sub.add_parser("report")
    p.add_argument("out")
    p.add_argument("--b-dir", nargs="+", default=None)
    p.add_argument("--model-diff", default=None, help="json of {label: model_dir_diff} files to list in the report (from the modeldiff command)")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--note", action="append", default=[], help="free-text line added to the report notes (repeatable)")
    a = ap.parse_args(argv)
    if a.cmd == "modeldiff":
        pairs = {x.split("=", 1)[0]: tuple(Path(y) for y in x.split("=", 1)[1].split(",")) for x in a.pairs}
        res = model_diff_summary(pairs)
        Path(a.out_json).write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(f"{res['n_pairs']} pairs, {res['n_unexpected']} with unexpected differences: {res['unexpected']}")
        return 0
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if a.cmd == "cross":
        on_sh = Path(a.on_sh).resolve()
        rows = cross_rows(Path(a.cur_sh).resolve(), on_sh, on_sh / a.cur_rows, on_sh / a.on_rows, a.variant, a.sets)
        (out / "cross_rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        print(f"wrote {len(rows)} cross rows; max |stored - rescored| logit = {max(r['chk_stored'] for r in rows):.2e}")
        return 0
    rows_path = out / "cross_rows.jsonl"
    a_sum = None
    meta: dict = {}
    if rows_path.exists():
        rows = [json.loads(x) for x in rows_path.read_text(encoding="utf-8").splitlines() if x.strip()]
        meta["max |stored - rescored| logit (check A)"] = f"{max(r['chk_stored'] for r in rows):.2e}"
        both = []
        fits = out / "both_on_sets.json"
        if fits.exists():
            both = json.loads(fits.read_text(encoding="utf-8"))
        a_sum = check_a_summary(rows, a.n_boot, 0, both_on_sets=both)
    b_sum = check_b_summary([Path(x) for x in a.b_dir]) if a.b_dir else None
    for name in (a.model_diff or "").split(","):
        if name:
            md_ = json.loads(Path(name).read_text(encoding="utf-8"))
            meta[f"model artifacts compared ({Path(name).name})"] = (
                f"{md_['n_pairs']} pairs, {md_['n_unexpected']} with a difference outside {', '.join(EXPECTED_DIFF)}")
    (out / "matchups_accept.json").write_text(json.dumps(dict(check_a=a_sum, check_b=b_sum, meta=meta), indent=1), encoding="utf-8")
    for i, n in enumerate(a.note, 1):
        meta[f"note {i}"] = n
    md = markdown(a_sum, b_sum, meta)
    (out / "matchups_accept.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
