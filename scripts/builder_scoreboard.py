"""Builder scoreboard: the deviation-cost slope b of every set x format, pooled across sets, and the paired comparison with
the pre-Session-8 model.

    python scripts/builder_scoreboard.py NEWDIR OUT [--legacy LEGACYDIR]

NEWDIR (and LEGACYDIR) hold one folder per set with the `deviation_cost.json` that `scripts/deviation_cost.py SET NEWDIR/SET`
wrote (LEGACYDIR: the same run with `--legacy`, same events). Writes OUT.json and OUT.md (OUT without extension) and prints the table.

Per set x format: events, games, b with its 95% interval (cluster bootstrap over held-out events, as in deviation_cost), a, and
the mean predicted gain. The model predicts b = -1 (b = 0: the builder's predicted gains are an artefact).

Pooling across sets (inverse-variance, because the saved results are per-set estimates with bootstrap intervals; the events of
different sets are independent, so this is the honest combination the saved data support - an event-level pooled regression would
need one pilot-covariate encoding for all sets):
    SE_i = (hi_i - lo_i) / (2 * 1.96)                 (from the bootstrap interval)
    fixed effect  b = sum(w b_i) / sum(w), w = 1 / SE_i^2, SE = 1 / sqrt(sum w)
    heterogeneity Cochran's Q = sum w (b_i - b)^2 on k - 1 df, I^2 = max(0, (Q - df) / Q), DerSimonian-Laird tau^2
    random effects b = sum(w* b_i) / sum(w*), w* = 1 / (SE_i^2 + tau^2)   (what to read when the sets disagree)
Rows pooled: bo1, bo3, and all set x format cells. z against -1 tests "b = -1".

Within event (docs/within_event/RULE.md): when NEWDIR/SET/within_event.json exists (`scripts/within_event_check.py --train-only
NEWDIR`), the slope of results on the train-only deck score within hold-out events where one player registered several decks
(same pilot, same pool) is listed per cell and pooled (model: +1).

Draft formats (docs/draft_mode/RULE.md, guardrail G1): `deviation_cost.py --fmt dbo1 dbo3` results are listed per cell and pooled per draft
format (`dbo1`, `dbo3`; the Sealed rows `bo1` / `bo3` / `all` never include them). The G1 line: the guardrail BLOCKS when, in either draft format, the upper
bound of the 95% CI of the pooled slope b (fixed effect; the random-effects interval is printed beside it) is above -0.5, i.e. less than half
of the builder's predicted advantage is realised; PASS otherwise.

Paired comparison (--legacy): b_new - b_legacy per cell with a paired bootstrap interval (both runs draw the same bootstrap
multiplicities over the same events; checked via the event ids and the number of events), plus the mean |b + 1| of each.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

FORMATS = ("bo1", "bo3")
DRAFT_FORMATS = ("dbo1", "dbo3")
G1_UPPER = -0.5        # G1 blocks if the pooled CI upper bound of b is above this, in either draft format
Z95 = 1.959964


def load_dir(d: Path) -> dict[str, dict]:
    out = {}
    for p in sorted(Path(d).glob("*/deviation_cost.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        out[r["set"]] = r
    return out


def se_of(ci: list[float]) -> float:
    return max((ci[1] - ci[0]) / (2 * Z95), 1e-9)


def pool(bs: list[float], ses: list[float], target: float = -1.0) -> dict:
    """Inverse-variance fixed-effect and DerSimonian-Laird random-effects pooled estimate with heterogeneity."""
    b = np.asarray(bs, float)
    se = np.asarray(ses, float)
    k = len(b)
    if k == 0:
        return {}
    w = 1 / se ** 2
    fe = float((w * b).sum() / w.sum())
    fe_se = float(1 / math.sqrt(w.sum()))
    q = float((w * (b - fe) ** 2).sum())
    df = k - 1
    i2 = max(0.0, (q - df) / q) if q > 0 and df > 0 else 0.0
    c = float(w.sum() - (w ** 2).sum() / w.sum())
    tau2 = max(0.0, (q - df) / c) if df > 0 and c > 0 else 0.0
    wr = 1 / (se ** 2 + tau2)
    re = float((wr * b).sum() / wr.sum())
    re_se = float(1 / math.sqrt(wr.sum()))
    return dict(k=k, fixed=dict(b=fe, se=fe_se, ci=[fe - Z95 * fe_se, fe + Z95 * fe_se], z_vs_target=(fe - target) / fe_se),
                random=dict(b=re, se=re_se, ci=[re - Z95 * re_se, re + Z95 * re_se], z_vs_target=(re - target) / re_se),
                Q=q, df=df, Q_p=float(stats.chi2.sf(q, df)) if df > 0 else float("nan"), I2=i2, tau=math.sqrt(tau2),
                mean_abs_dev_from_target=float(np.mean(np.abs(b - target))))


def within_rows(d: Path) -> list[dict]:
    rows = []
    for p in sorted(Path(d).glob("*/within_event.json")):
        for r in json.loads(p.read_text(encoding="utf-8")):
            if "b_logit" in r:
                rows.append(dict(set=r["set"], fmt=r["fmt"], n_events=r["n_events"], n_games=r["n_games"], slope=r["b_logit"],
                                 se=r["se_logit"]))
    return sorted(rows, key=lambda r: (r["set"], r["fmt"]))


def within_pooled(rows: list[dict]) -> dict:
    out = {}
    for name, sel in (("bo1", [r for r in rows if r["fmt"] == "bo1"]), ("bo3", [r for r in rows if r["fmt"] == "bo3"]), ("all", rows)):
        if sel:
            out[name] = pool([r["slope"] for r in sel], [r["se"] for r in sel], target=1.0)
            out[name]["n_events"] = int(sum(r["n_events"] for r in sel))
    return out


def cells(res: dict[str, dict]) -> list[dict]:
    rows = []
    for sc in sorted(res):
        for fmt in FORMATS + DRAFT_FORMATS:
            f = res[sc].get("formats", {}).get(fmt)
            if not f:
                continue
            rows.append(dict(set=sc, fmt=fmt, n_events=f["n_events"], n_games=f["n_games"], b=f["b"], b_ci=f["b_ci"],
                             b_se=se_of(f["b_ci"]), a=f["a"], a_ci=f["a_ci"], mean_predicted_gain_pp=f["mean_predicted_gain_pp"],
                             build_seconds=f.get("build_seconds"), fit_seconds=res[sc].get("fit_seconds")))
    return rows


def pooled_rows(rows: list[dict], key_b="b", key_ci="b_ci") -> dict:
    out = {}
    sealed = [r for r in rows if r["fmt"] in FORMATS]
    for name, sel in (("bo1", [r for r in rows if r["fmt"] == "bo1"]), ("bo3", [r for r in rows if r["fmt"] == "bo3"]), ("all", sealed),
                      ("dbo1", [r for r in rows if r["fmt"] == "dbo1"]), ("dbo3", [r for r in rows if r["fmt"] == "dbo3"])):
        if sel:
            p = pool([r[key_b] for r in sel], [se_of(r[key_ci]) for r in sel])
            p["n_events"] = int(sum(r["n_events"] for r in sel))
            out[name] = p
            pa = pool([r["a"] for r in sel], [se_of(r["a_ci"]) for r in sel], target=1.0)
            out[name]["a_fixed"] = pa["fixed"]
            out[name]["a_random"] = pa["random"]
            out[name]["a_I2"] = pa["I2"]
    return out


def pair(new: dict[str, dict], old: dict[str, dict]) -> list[dict]:
    rows = []
    for sc in sorted(set(new) & set(old)):
        for fmt in FORMATS:
            fn, fo = new[sc].get("formats", {}).get(fmt), old[sc].get("formats", {}).get(fmt)
            if not fn or not fo:
                continue
            same = fn.get("event_ids") is not None and fn.get("event_ids") == fo.get("event_ids")
            row = dict(set=sc, fmt=fmt, same_events=bool(same), n_events=fn["n_events"], b_new=fn["b"], b_new_ci=fn["b_ci"],
                       b_legacy=fo["b"], b_legacy_ci=fo["b_ci"], diff=fn["b"] - fo["b"],
                       abs_dev_new=abs(fn["b"] + 1), abs_dev_legacy=abs(fo["b"] + 1))
            bn, bo = fn.get("b_boots"), fo.get("b_boots")
            if same and fn["n_events"] == fo["n_events"] and bn and bo and len(bn) == len(bo):
                d = np.asarray(bn) - np.asarray(bo)
                row["diff_ci"] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
                row["diff_se"] = float(d.std(ddof=1))
                ad = np.abs(np.asarray(bn) + 1) - np.abs(np.asarray(bo) + 1)       # paired |b + 1|, new minus legacy
                row["absdiff"] = row["abs_dev_new"] - row["abs_dev_legacy"]
                row["absdiff_ci"] = [float(np.percentile(ad, 2.5)), float(np.percentile(ad, 97.5))]
                row["absdiff_se"] = float(ad.std(ddof=1))
            rows.append(row)
    return rows


def pair_summary(rows: list[dict]) -> dict:
    out = {}
    for name, sel in (("bo1", [r for r in rows if r["fmt"] == "bo1"]), ("bo3", [r for r in rows if r["fmt"] == "bo3"]), ("all", rows)):
        sel = [r for r in sel if "diff_se" in r]
        if sel:
            p = pool([r["diff"] for r in sel], [max(r["diff_se"], 1e-9) for r in sel], target=0.0)
            pa = pool([r["absdiff"] for r in sel], [max(r["absdiff_se"], 1e-9) for r in sel], target=0.0)
            out[name] = dict(k=p["k"], diff=p["fixed"]["b"], ci=p["fixed"]["ci"], Q_p=p["Q_p"], I2=p["I2"],
                             mean_abs_dev_new=float(np.mean([r["abs_dev_new"] for r in sel])),
                             mean_abs_dev_legacy=float(np.mean([r["abs_dev_legacy"] for r in sel])),
                             absdiff=pa["fixed"]["b"], absdiff_ci=pa["fixed"]["ci"],
                             absdiff_worse=bool(pa["fixed"]["ci"][0] > 0))
    return out


def g1_lines(pooled: dict) -> dict:
    """Guardrail G1 of docs/draft_mode/RULE.md from the pooled rows: per draft format, PASS if the upper bound of the pooled (fixed effect) 95% CI
    of b is <= -0.5, BLOCK if it is above; the overall status BLOCKs if any draft format blocks and is n/a without draft rows."""
    out: dict = {}
    for f in DRAFT_FORMATS:
        p = pooled.get(f)
        if not p:
            continue
        up = p["fixed"]["ci"][1]
        out[f] = dict(b=p["fixed"]["b"], ci=p["fixed"]["ci"], upper=up, need=f"CI upper bound <= {G1_UPPER}", n_cells=p["k"],
                      n_events=p["n_events"], random_b=p["random"]["b"], random_ci=p["random"]["ci"], status="BLOCK" if up > G1_UPPER else "PASS")
    out["status"] = "n/a" if not out else "BLOCK" if any(v["status"] == "BLOCK" for v in out.values()) else "PASS"
    return out


def fmt_ci(ci) -> str:
    return f"[{ci[0]:+.2f}, {ci[1]:+.2f}]"


def markdown(rows, pooled, prow=None, psum=None, wrows=None, wpooled=None) -> str:
    L = ["# Builder scoreboard: deviation-cost slope b (model: -1, artefact: 0)", "",
         "| set | fmt | events | games | b | 95% CI | a | a 95% CI | predicted gain pp |", "|---|---|---:|---:|---:|---|---:|---|---:|"]
    for r in rows:
        L.append(f"| {r['set']} | {r['fmt']} | {r['n_events']} | {r['n_games']} | {r['b']:+.2f} | {fmt_ci(r['b_ci'])} | "
                 f"{r['a']:+.2f} | {fmt_ci(r['a_ci'])} | {r['mean_predicted_gain_pp']:.2f} |")
    L += ["", "## Pooled across sets (inverse variance)", "",
          "| rows | cells | events | b fixed | 95% CI | z vs -1 | b random | 95% CI | Q (p) | I2 | tau | mean abs(b+1) | a fixed | a 95% CI |",
          "|---|---:|---:|---:|---|---:|---:|---|---|---:|---:|---:|---:|---|"]
    for name in ("bo1", "bo3", "all", "dbo1", "dbo3"):
        p = pooled.get(name)
        if not p:
            continue
        L.append(f"| {name} | {p['k']} | {p['n_events']} | {p['fixed']['b']:+.2f} | {fmt_ci(p['fixed']['ci'])} | {p['fixed']['z_vs_target']:+.1f} | "
                 f"{p['random']['b']:+.2f} | {fmt_ci(p['random']['ci'])} | {p['Q']:.1f} ({p['Q_p']:.2f}) | {100 * p['I2']:.0f}% | "
                 f"{p['tau']:.2f} | {p['mean_abs_dev_from_target']:.2f} | {p['a_fixed']['b']:+.2f} | {fmt_ci(p['a_fixed']['ci'])} |")
    g1 = g1_lines(pooled)
    if g1["status"] != "n/a":
        L += ["", "## G1 (draft formats): blocks if the pooled CI upper bound of b is above -0.5", ""]
        for f in DRAFT_FORMATS:
            if f in g1:
                v = g1[f]
                L.append(f"- {f}: pooled b {v['b']:+.2f} {fmt_ci(v['ci'])} (fixed effect, {v['n_cells']} sets, {v['n_events']} events; random effects "
                         f"{v['random_b']:+.2f} {fmt_ci(v['random_ci'])}): CI upper {v['upper']:+.2f} vs -0.5: **{v['status']}**")
        L.append(f"- G1: **{g1['status']}**")
    if wrows:
        L += ["", "## Within event: same pilot, same pool, several decks (model: +1; train-only model, hold-out events)", "",
              "| set | fmt | events | games | slope | 95% CI |", "|---|---|---:|---:|---:|---|"]
        for r in wrows:
            L.append(f"| {r['set']} | {r['fmt']} | {r['n_events']} | {r['n_games']} | {r['slope']:+.2f} | "
                     f"{fmt_ci([r['slope'] - Z95 * r['se'], r['slope'] + Z95 * r['se']])} |")
        L += ["", "| rows | cells | events | slope fixed | 95% CI | slope random | 95% CI | I2 |", "|---|---:|---:|---:|---|---:|---|---:|"]
        for name, p in (wpooled or {}).items():
            L.append(f"| {name} | {p['k']} | {p['n_events']} | {p['fixed']['b']:+.2f} | {fmt_ci(p['fixed']['ci'])} | {p['random']['b']:+.2f} | "
                     f"{fmt_ci(p['random']['ci'])} | {100 * p['I2']:.0f}% |")
    if prow:
        L += ["", "## Paired: NEWDIR vs --legacy DIR (same events)", "",
              "| set | fmt | events | b new | 95% CI | b legacy | 95% CI | new - legacy | paired 95% CI | same events |",
              "|---|---|---:|---:|---|---:|---|---:|---|---|"]
        for r in prow:
            dci = fmt_ci(r["diff_ci"]) if "diff_ci" in r else "n/a"
            L.append(f"| {r['set']} | {r['fmt']} | {r['n_events']} | {r['b_new']:+.2f} | {fmt_ci(r['b_new_ci'])} | {r['b_legacy']:+.2f} | "
                     f"{fmt_ci(r['b_legacy_ci'])} | {r['diff']:+.2f} | {dci} | {'yes' if r['same_events'] else 'NO'} |")
        if psum:
            L += ["", "| rows | cells | pooled new - legacy | 95% CI | het. p | mean abs(b+1) new | legacy | paired abs(b+1) new - legacy | 95% CI | worse |",
                  "|---|---:|---:|---|---:|---:|---:|---:|---|---|"]
            for name, s in psum.items():
                L.append(f"| {name} | {s['k']} | {s['diff']:+.2f} | {fmt_ci(s['ci'])} | {s['Q_p']:.2f} | {s['mean_abs_dev_new']:.2f} | {s['mean_abs_dev_legacy']:.2f} | "
                         f"{s['absdiff']:+.2f} | {fmt_ci(s['absdiff_ci'])} | {'YES' if s['absdiff_worse'] else 'no'} |")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("newdir")
    ap.add_argument("out", help="output path without extension: writes OUT.json and OUT.md")
    ap.add_argument("--legacy", default=None)
    args = ap.parse_args(argv)
    new = load_dir(Path(args.newdir))
    if not new:
        print("no deviation_cost.json found in", args.newdir)
        return 1
    rows = cells(new)
    pooled = pooled_rows(rows)
    prow = psum = None
    if args.legacy:
        old = load_dir(Path(args.legacy))
        prow = pair(new, old)
        psum = pair_summary(prow)
    wrows = within_rows(Path(args.newdir))
    wpooled = within_pooled(wrows) if wrows else None
    res = dict(cells=rows, pooled=pooled, paired=prow, paired_pooled=psum, within_event=wrows, within_event_pooled=wpooled)
    g1 = g1_lines(pooled)
    if g1["status"] != "n/a":
        res["g1"] = g1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    md = markdown(rows, pooled, prow, psum, wrows, wpooled)
    out.with_suffix(".md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
