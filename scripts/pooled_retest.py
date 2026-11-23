"""Pooled re-test of synergy / archetype-conditional values / matchups, following docs/pooled_retest/POOLING_RULE.md literally.

    python scripts/pooled_retest.py OUT_DIR [--app-home PATH]

Reads (never writes) each set's model/report.json in the app home, pools the per-set hold-out gains (DerSimonian-Laird random
effects, fixed effect as a secondary), applies the pre-registered verdict and, for a component that qualifies, the leave-one-set-out
honest check.  Writes OUT_DIR/pooled.json and OUT_DIR/pooled.md.  No model is fitted.

Data mapping (report.json):  halves A/B   -> report[c]["confirmation"]["halves"][f][A|B]  (gain, lo, hi, lo90)
                              whole hold-out -> report["validation"][f][VKEY[c]]           (gain, lo, hi); the whole-hold-out numbers
                                                are NOT in `halves`; the matchups test is stored under the key "opponent".
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sealed_lab.model.pipeline import MIN_GAIN  # noqa: E402
from sealed_lab.model.pooling import (PARTS, Z90, Z95, dersimonian_laird, fixed_effect, leave_one_out, materially_worse,  # noqa: E402,F401
                                      own_materially_worse, pool_parts, qualifies, se_from_ci, verdict)

SETS = ["ECL", "EOE", "FIN", "FRA", "HOB", "MSH", "OM1", "SOS", "TDM", "TLA", "TMT"]
FORMATS = ["bo1", "bo3"]
COMPONENTS = ["synergy", "archcond", "matchups", "playdraw", "oppflags"]
VKEY = {"synergy": "synergy", "archcond": "archcond", "matchups": "opponent", "playdraw": "playdraw", "oppflags": "oppflags"}   # key in report["validation"][f]


# (the pure arithmetic - DerSimonian-Laird, verdict, leave-one-out - lives in sealed_lab/model/pooling.py, which the app uses too)


# ------------------------------------------------------------------------------------------------------------- data
def app_home(arg: str | None) -> Path:
    return Path(arg) if arg else Path(os.environ["LOCALAPPDATA"]) / "sealed_lab"


def load_cells(home: Path) -> tuple[dict, list[dict], dict]:
    """-> (cells[c][f][set] = {'whole','A','B','confirmed','applied'}, left_out list, notes)."""
    cells = {c: {f: {} for f in FORMATS} for c in COMPONENTS}
    left_out = []
    for s in SETS:
        p = home / "sets" / s / "model" / "report.json"
        try:
            rep = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            for c in COMPONENTS:
                for f in FORMATS:
                    left_out.append(dict(set=s, format=f, component=c, why=f"report.json unreadable: {e}"))
            continue
        for c in COMPONENTS:
            comp = rep.get(c) or {}
            conf = comp.get("confirmation") or {}
            halves = conf.get("halves") or {}
            for f in FORMATS:
                val = (rep.get("validation") or {}).get(f) or {}
                why = None
                if val.get("status") != "ok":
                    why = f"validation status {val.get('status')!r}"
                whole = val.get(VKEY[c]) if not why else None
                h = halves.get(f)
                if not why and not (isinstance(whole, dict) and "gain" in whole):
                    why = "no whole-hold-out test"
                if not why and not (isinstance(h, dict) and "A" in h and "B" in h):
                    why = "no half-A/half-B numbers"
                if why:
                    left_out.append(dict(set=s, format=f, component=c, why=why))
                    continue
                cells[c][f][s] = dict(
                    whole={k: whole[k] for k in ("gain", "lo", "hi")},
                    A={k: h["A"][k] for k in ("gain", "lo", "hi", "lo90")},
                    B={k: h["B"][k] for k in ("gain", "lo", "hi", "lo90")},
                    confirmed=f in (conf.get("confirmed") or []),   # per-format decision of the app's confirmation rule
                    applied=bool(comp.get("applied")))              # component-level decision the app actually uses
    return cells, left_out, {}


def analyse(cells: dict, left_out: list[dict], decision_key: str = "applied") -> dict:
    """The full analysis. decision_key: the per-set decision used for 'current' (flipped cells): 'applied' or 'confirmed'."""
    res = dict(min_gain=MIN_GAIN, current_decision_key=decision_key, left_out=left_out, components={})
    for c in COMPONENTS:
        comp = dict(formats={})
        verd = {}
        for f in FORMATS:
            cf = cells[c][f]
            if not cf:
                comp["formats"][f] = dict(n_sets=0)
                continue
            pooled = pool_parts(list(cf.values()))
            v = verdict(pooled)
            verd[f] = v
            comp["formats"][f] = dict(n_sets=len(cf), sets=sorted(cf), pooled=pooled, verdict=v)
        comp["qualifies"] = qualifies(verd) if verd else False
        if comp["qualifies"]:
            comp["loo"] = loo_check(cells[c], decision_key)
        res["components"][c] = comp
    return res


def loo_check(cf_by_format: dict, decision_key: str) -> dict:
    """LOO table and the shipping test over flipped cells for one qualifying component."""
    rows, flipped = [], []
    for f in FORMATS:
        cf = cf_by_format[f]
        loo = leave_one_out(cf)
        for s, cell in cf.items():
            lv = loo[s]["verdict"]
            worse_own = own_materially_worse(cell)
            # the rule: apply if the LOO verdict is pooled-better in f AND the set's own numbers are not materially worse.
            pooled_applies = lv["better"] and not worse_own
            current = bool(cell[decision_key])
            row = dict(set=s, format=f, loo_label=lv["label"], loo_mean_A=loo[s]["pooled"]["A"]["mean"],
                       own=cell["whole"], own_worse=worse_own, current=current, pooled=pooled_applies,
                       flipped=pooled_applies and not current)
            rows.append(row)
            if row["flipped"]:
                flipped.append(row)
    ship = dict(n_flipped=len(flipped))
    if flipped:
        fe = fixed_effect([r["own"]["gain"] for r in flipped], [se_from_ci(r["own"]["lo"], r["own"]["hi"]) for r in flipped])
        ship.update(fe_gain=fe["mean"], fe_lo=fe["lo"], fe_hi=fe["hi"],
                    any_materially_worse=any(r["own_worse"] for r in flipped),
                    passes=fe["mean"] > 0 and not any(r["own_worse"] for r in flipped))
    else:
        ship.update(passes=False, note="no flipped cell")
    return dict(rows=rows, shipping=ship)


# ----------------------------------------------------------------------------------------------------------- report
def f5(x: float) -> str:
    return f"{x:+.5f}"


def ci(o: dict, lo="lo", hi="hi", m="mean") -> str:
    return f"{o[m]:+.5f} [{o[lo]:+.5f}, {o[hi]:+.5f}]"


def to_markdown(res: dict) -> str:
    L = ["# Pooled re-test (rule: POOLING_RULE.md, as written)", "",
         f"MIN_GAIN = {res['min_gain']:g}. Gains are paired log-loss gains per game vs the plain chain; SE = (hi - lo) / 3.92; "
         "DerSimonian-Laird random effects, 95% normal CI.", ""]
    L += ["## Pooled results", "",
          "| component | fmt | n sets | RE whole [95% CI] | RE half A [95% CI] | RE half B [95% CI] (90% lo) | tau2 (whole/A/B) | I2 (whole/A/B) | FE whole | verdict | qualifies |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c, comp in res["components"].items():
        for f, d in comp["formats"].items():
            if not d.get("n_sets"):
                L.append(f"| {c} | {f} | 0 | - | - | - | - | - | - | - | no |")
                continue
            p, v = d["pooled"], d["verdict"]
            L.append(f"| {c} | {f} | {d['n_sets']} | {ci(p['whole'])} | {ci(p['A'])} | {ci(p['B'])} ({v['B_lo90']:+.5f}) | "
                     f"{p['whole']['tau2']:.2e} / {p['A']['tau2']:.2e} / {p['B']['tau2']:.2e} | "
                     f"{p['whole']['I2']:.0%} / {p['A']['I2']:.0%} / {p['B']['I2']:.0%} | "
                     f"{p['whole']['fe_mean']:+.5f} [{p['whole']['fe_lo']:+.5f}, {p['whole']['fe_hi']:+.5f}] | "
                     f"{v['label']} | {'yes' if comp['qualifies'] else 'no'} |")
    L += ["", "Fixed-effect (secondary, not used to decide), half A / half B:", ""]
    for c, comp in res["components"].items():
        for f, d in comp["formats"].items():
            if d.get("n_sets"):
                p = d["pooled"]
                L.append(f"- {c} {f}: A {p['A']['fe_mean']:+.5f} [{p['A']['fe_lo']:+.5f}, {p['A']['fe_hi']:+.5f}], "
                         f"B {p['B']['fe_mean']:+.5f} [{p['B']['fe_lo']:+.5f}, {p['B']['fe_hi']:+.5f}]")
    L += ["", "## Left-out cells", ""]
    L += [f"- {r['set']} {r['format']} {r['component']}: {r['why']}" for r in res["left_out"]] or ["- none (all 11 sets x 2 formats x 3 components tested)"]
    for c, comp in res["components"].items():
        if "loo" not in comp:
            continue
        L += ["", f"## Leave-one-set-out: {c} (current decision = report.json {res['current_decision_key']})", "",
              "| set | fmt | LOO verdict | own whole gain [CI] | own materially worse | current | pooled | flipped |", "|---|---|---|---|---|---|---|---|"]
        for r in comp["loo"]["rows"]:
            L.append(f"| {r['set']} | {r['format']} | {'pooled-better' if r['loo_label'] == 'pooled-better' else r['loo_label']} | "
                     f"{r['own']['gain']:+.5f} [{r['own']['lo']:+.5f}, {r['own']['hi']:+.5f}] | {'yes' if r['own_worse'] else 'no'} | "
                     f"{'apply' if r['current'] else 'off'} | {'apply' if r['pooled'] else 'off'} | {'FLIPPED' if r['flipped'] else ''} |")
        s = comp["loo"]["shipping"]
        L += ["", f"Shipping test ({s['n_flipped']} flipped cells): " + (
            f"fixed-effect own whole gain {s['fe_gain']:+.5f} [{s['fe_lo']:+.5f}, {s['fe_hi']:+.5f}], any materially worse: "
            f"{s['any_materially_worse']}, passes: {s['passes']}" if s["n_flipped"] else "no flipped cell, nothing to ship")]
    if not any("loo" in comp for comp in res["components"].values()):
        L += ["", "## Result", "", "No component qualifies for pooled application; nothing is built."]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out")
    ap.add_argument("--app-home", default=None)
    a = ap.parse_args(argv)
    cells, left_out, _ = load_cells(app_home(a.app_home))
    res = analyse(cells, left_out, "applied")
    # the alternative reading of "current decision" (per-format `confirmed`), reported alongside
    res["alt_confirmed"] = {c: comp.get("loo") for c, comp in analyse(cells, left_out, "confirmed")["components"].items() if "loo" in comp}
    res["cells"] = cells
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "pooled.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (out / "pooled.md").write_text(to_markdown(res), encoding="utf-8")
    print(to_markdown(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
