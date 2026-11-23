"""Evidence pooled across sets: DerSimonian-Laird random effects over the sets' hold-out gains and the pre-registered verdict.

The rule is written in docs/pooled_retest/POOLING_RULE.md (and run offline by scripts/pooled_retest.py, which imports the
arithmetic from here).  Per set and format a component has three paired log-loss gains against the plain chain, each with a 95%
cluster-bootstrap CI: on the whole hold-out, on half A (earlier events) and on half B (later events).  `pool_parts` pools each
over sets (SE = (hi - lo) / 3.92); `verdict` mirrors `pipeline._verdict` on the pooled numbers; `pooled_decision` is the app's
use of it for ONE set: the verdict pooled over the OTHER sets only, plus the safety check on the set's own numbers.

A cell is {'whole': {gain, lo, hi}, 'A': {gain, lo, hi}, 'B': {gain, lo, hi}} (extra keys such as lo90 are ignored).
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path

log = logging.getLogger(__name__)

Z95 = 1.96
Z90 = 1.2816
PARTS = ("whole", "A", "B")
MIN_SETS = 5                         # other sets with numbers a format needs before the pooled evidence is used at all


def _min_gain() -> float:
    from sealed_lab.model.pipeline import MIN_GAIN

    return MIN_GAIN


# ---------------------------------------------------------------------------------------------------- pure arithmetic
def se_from_ci(lo: float, hi: float) -> float:
    return (hi - lo) / 3.92


def fixed_effect(ys: list[float], ses: list[float]) -> dict:
    """Inverse-variance fixed-effect mean with its 95% normal CI, Cochran's Q."""
    w = [1.0 / s ** 2 for s in ses]
    sw = sum(w)
    mean = sum(wi * y for wi, y in zip(w, ys)) / sw
    se = math.sqrt(1.0 / sw)
    q = sum(wi * (y - mean) ** 2 for wi, y in zip(w, ys))
    return dict(mean=mean, se=se, lo=mean - Z95 * se, hi=mean + Z95 * se, Q=q)


def dersimonian_laird(ys: list[float], ses: list[float]) -> dict:
    """DerSimonian-Laird random-effects mean (weights 1/(se^2 + tau2)), 95% normal CI, tau2, Cochran's Q, I2, and the fixed-effect mean."""
    k = len(ys)
    if k < 1 or any(s <= 0 for s in ses):
        raise ValueError("need >= 1 study with SE > 0")
    fe = fixed_effect(ys, ses)
    w = [1.0 / s ** 2 for s in ses]
    sw, sw2 = sum(w), sum(wi * wi for wi in w)
    q = fe["Q"]
    df = k - 1
    c = sw - sw2 / sw
    tau2 = max(0.0, (q - df) / c) if (k > 1 and c > 0) else 0.0
    i2 = max(0.0, (q - df) / q) if (k > 1 and q > 0) else 0.0
    ws = [1.0 / (s ** 2 + tau2) for s in ses]
    sws = sum(ws)
    mean = sum(wi * y for wi, y in zip(ws, ys)) / sws
    se = math.sqrt(1.0 / sws)
    return dict(k=k, mean=mean, se=se, lo=mean - Z95 * se, hi=mean + Z95 * se, tau2=tau2, Q=q, I2=i2,
                fe_mean=fe["mean"], fe_se=fe["se"], fe_lo=fe["lo"], fe_hi=fe["hi"])


def pool_parts(cells: list[dict]) -> dict:
    """cells: [{'whole': {gain, lo, hi}, 'A': {...}, 'B': {...}}]  ->  DL pooled result for whole / A / B."""
    out = {}
    for p in PARTS:
        ys = [c[p]["gain"] for c in cells]
        ses = [se_from_ci(c[p]["lo"], c[p]["hi"]) for c in cells]
        out[p] = dersimonian_laird(ys, ses)
    return out


def materially_worse(o: dict) -> bool:
    """CI (lo, hi) of a gain `o` entirely below zero and the gain <= -MIN_GAIN (the app's safety rule)."""
    return o["hi"] < 0 and o["gain"] <= -_min_gain()


def verdict(pooled: dict) -> dict:
    """The pre-registered decision on pooled whole / A / B results (DL means).

    better: half A 95% CI above zero and mean >= MIN_GAIN, AND half B mean > 0 with the one-sided 90% bound (mean - 1.2816 se) > 0.
    worse:  on the whole hold-out or either half, 95% CI below zero and mean <= -MIN_GAIN.
    The label is 'worse' if worse (a safety veto), else 'better' if better, else 'neither' (both flags are kept)."""
    mg = _min_gain()
    a, b = pooled["A"], pooled["B"]
    b_lo90 = b["mean"] - Z90 * b["se"]
    better = a["lo"] > 0 and a["mean"] >= mg and b["mean"] > 0 and b_lo90 > 0
    worse = any(pooled[p]["hi"] < 0 and pooled[p]["mean"] <= -mg for p in PARTS)
    label = "pooled-worse" if worse else ("pooled-better" if better else "neither")
    return dict(better=better, worse=worse, label=label, B_lo90=b_lo90)


def qualifies(verdicts: dict) -> bool:
    """verdicts: {format: verdict dict}. Pooled-better in >= 1 format and pooled-worse in none."""
    return any(v["better"] for v in verdicts.values()) and not any(v["worse"] for v in verdicts.values())


def leave_one_out(cells: dict) -> dict:
    """cells: {set: cell}. For each set the verdict pooled over all the OTHER sets."""
    out = {}
    for s in cells:
        others = [c for t, c in cells.items() if t != s]
        if not others:
            continue
        pooled = pool_parts(others)
        out[s] = dict(pooled=pooled, verdict=verdict(pooled))
    return out


def own_materially_worse(cell: dict) -> bool:
    return any(materially_worse(cell[p]) for p in PARTS)


# ------------------------------------------------------------------------------------------------- cells from reports
def _num_ok(o: object) -> bool:
    try:
        return all(math.isfinite(float(o[k])) for k in ("gain", "lo", "hi")) and float(o["hi"]) > float(o["lo"])    # type: ignore[index]
    except (KeyError, TypeError, ValueError):
        return False


def make_cell(whole: dict | None, halves: dict | None) -> dict | None:
    """Cell of one set-format from its whole-hold-out numbers and its confirmation halves {'A': .., 'B': ..}; None if any is missing
    or unusable (non-finite, or a CI of zero width: no standard error)."""
    if not isinstance(whole, dict) or not isinstance(halves, dict):
        return None
    parts = {"whole": whole, "A": halves.get("A"), "B": halves.get("B")}
    if not all(_num_ok(parts[p]) for p in PARTS):
        return None
    return {p: {k: float(parts[p][k]) for k in ("gain", "lo", "hi")} for p in PARTS}


def cell_from_report(rep: dict, component: str, vkey: str, fmt: str) -> dict | None:
    """Cell of `fmt` from a report.json dict: the whole-hold-out numbers from report['validation'][fmt][vkey] (the matchups test is
    stored under 'opponent'), the halves from report[component]['confirmation']['halves'][fmt].  None where the test was not run."""
    val = (rep.get("validation") or {}).get(fmt) or {}
    if val.get("status") != "ok":
        return None
    halves = (((rep.get(component) or {}).get("confirmation") or {}).get("halves") or {}).get(fmt)
    return make_cell(val.get(vkey), halves)


def other_cells(set_code: str, component: str, vkey: str, formats: tuple[str, ...] = ("bo1", "bo3"),
                home: Path | None = None) -> dict[str, dict[str, dict]]:
    """{fmt: {set: cell}} from the latest stored report.json of every OTHER set in the app home (`home` / sets / SET / model).
    Sets without a readable report, or whose report lacks the numbers for a format, are skipped for that format.  Never writes."""
    from sealed_lab import config

    root = (Path(home) if home is not None else config.app_home()) / "sets"
    out: dict[str, dict[str, dict]] = {f: {} for f in formats}
    if not root.exists():
        return out
    for d in sorted(root.iterdir()):
        code = d.name.upper()
        p = d / "model" / "report.json"
        if code == set_code.upper() or not p.exists():
            continue
        try:
            rep = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001 - a broken neighbour set must never break this set's calculation
            log.warning("pooled evidence: cannot read the report of %s (%s)", code, e)
            continue
        if not isinstance(rep, dict):
            continue
        for f in formats:
            cell = cell_from_report(rep, component, vkey, f)
            if cell is not None:
                out[f][code] = cell
    return out


# ------------------------------------------------------------------------------------------------ the app's decision
def pooled_decision(own: dict[str, dict | None], others: dict[str, dict[str, dict]], min_sets: int = MIN_SETS,
                    exclude: str | None = None) -> dict[str, dict]:
    """Per format the pooled decision for ONE set.  `own`: {fmt: cell or None} of the set being calculated; `others`: {fmt: {set: cell}}
    of the other sets (a set named `exclude`, the set itself, is dropped even if present).

    Returns {fmt: dict(decided, apply, why, n_sets, sets, pooled, verdict, own_worse)}.  `decided` is False when the pooled evidence
    cannot be used (fewer than `min_sets` other sets, or no own numbers): the caller falls back to the per-set rule.  `apply` = the
    pooled verdict is better (and not worse) AND the set's own numbers are not materially worse (whole hold-out or either half)."""
    out: dict[str, dict] = {}
    for f, cell in own.items():
        oth = {s: c for s, c in (others.get(f) or {}).items() if exclude is None or s.upper() != exclude.upper()}
        d: dict = dict(decided=False, apply=False, n_sets=len(oth), sets=sorted(oth))
        if cell is None:
            d["why"] = "no own hold-out numbers (whole hold-out and both halves) in this format"
        elif len(oth) < min_sets:
            d["why"] = f"only {len(oth)} other set(s) with numbers in this format (at least {min_sets} needed)"
        else:
            pooled = pool_parts(list(oth.values()))
            v = verdict(pooled)
            worse_own = own_materially_worse(cell)
            d.update(decided=True, pooled=pooled, verdict=v, own_worse=worse_own, apply=bool(v["better"] and not v["worse"] and not worse_own))
            if d["apply"]:
                d["why"] = "pooled-better and this set's own numbers are not materially worse"
            elif v["worse"]:
                d["why"] = "the pooled evidence is materially worse (blocks)"
            elif worse_own:
                d["why"] = "this set's own numbers are materially worse (blocks)" + ("" if v["better"] else "; the pooled evidence is not better either")
            else:
                d["why"] = "the pooled evidence does not pass (half A not above zero, or not confirmed on half B)"
        out[f] = d
    return out


def summary(d: dict) -> dict:
    """JSON-ready summary of one `pooled_decision` entry for meta / report: n sets, RE gain [CI] on whole / A / B, tau2, verdict."""
    out = dict(decided=bool(d["decided"]), apply=bool(d["apply"]), n_sets=int(d["n_sets"]), sets=list(d["sets"]), why=d["why"])
    if d.get("decided"):
        p = d["pooled"]
        out.update(own_worse=bool(d["own_worse"]), verdict=d["verdict"]["label"], B_lo90=float(d["verdict"]["B_lo90"]),
                   re={x: dict(gain=float(p[x]["mean"]), lo=float(p[x]["lo"]), hi=float(p[x]["hi"]), tau2=float(p[x]["tau2"]),
                               I2=float(p[x]["I2"])) for x in PARTS})
    return out


def describe(f: str, s: dict) -> str:
    """One-line plain-language text of a `summary` entry for a format."""
    if not s.get("decided"):
        return f"{f}: pooled evidence not used ({s['why']})"
    r = s["re"]
    txt = (f"{f}: pooled over {s['n_sets']} other sets, random-effects gain whole {r['whole']['gain']:+.5f} "
           f"[{r['whole']['lo']:+.5f}, {r['whole']['hi']:+.5f}], earlier half {r['A']['gain']:+.5f} [{r['A']['lo']:+.5f}, {r['A']['hi']:+.5f}], "
           f"later half {r['B']['gain']:+.5f} (90% lower bound {s['B_lo90']:+.5f}), tau2 {r['whole']['tau2']:.1e}: {s['verdict']}")
    return txt + ("; applied" if s["apply"] else "; not applied (" + s["why"] + ")")
