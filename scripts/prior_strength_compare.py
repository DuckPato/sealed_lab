"""Acceptance test of the time-split prior strength + stacking (roadmap 2.1): hold-out log loss of the new plain chain M against the
current M (random event folds, original sigma grid) and against P (prior only), per format, on the SAME time hold-out as the report.

    python scripts/prior_strength_compare.py SET [SET ...] --out OUTDIR [--fmt bo3 bo1] [--jobs N]

Three plain chains are fitted on the training part (events before the hold-out cut) and scored on the hold-out:
  old   = random event folds, original sigma grid, no stacking        (the models of earlier versions)
  leaky = forward-chaining time folds + extended grid + stacking w, prior P fitted on all training data (first version; its P
          has seen the events being scored in each fold)
  new   = the same with P refitted per fold from events before the scored block (leak-free; what calculate() deploys by default)
Paired cluster-bootstrap CIs (events resampled) of every difference, as in the report: a positive "gain" means the first model is
better. P is the report's baseline (covariates refitted + a scalar on the previous stage's deck score). Uses SEALED_LAB_HOME
(point it at a scratch copy); writes OUTDIR/compare_SET.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sets", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fmt", nargs="+", default=["bo3", "bo1"])
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--n-rep", type=int, default=500)
    args = ap.parse_args(argv)

    import os

    import polars as pl

    from sealed_lab import config
    from sealed_lab.model import cardvalue, design, pipeline, structure, validate

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n_jobs = int(args.jobs or os.cpu_count() or 1)
    variants = {"old": cardvalue.LEGACY_OPTS, "leaky": cardvalue.LEAKY_OPTS, "new": cardvalue.ChainOpts("time", True, leak_free=True)}
    for sc in [s.upper() for s in args.sets]:
        t0 = time.time()
        inp = pipeline.load_inputs(sc)
        games, cards = inp.games, inp.cards
        keys, table = pipeline.phi_for_corpus(inp, n_jobs, None, None, config.cache_dir(sc))
        per_deck = games.group_by("deck_key").len().sort("deck_key")
        wts = np.zeros(len(keys))
        wts[np.searchsorted(keys, per_deck["deck_key"].to_numpy())] = per_deck["len"].to_numpy()
        space = design.build_deck_space(keys, table[:, :structure.N_FEATURES], inp.deck_cards, cards, weights=np.maximum(wts, 1e-9))
        formats = [f for f in pipeline.FMT_ORDER if games.filter(pl.col("fmt") == f).height >= 1]
        res: dict = {"set": sc, "formats": {}}
        for i, fmt in enumerate(formats):
            if fmt not in args.fmt:
                continue
            runs: dict = {}
            keeps: dict = {}
            for name, opts in variants.items():
                tmp = cardvalue.make_tmpdir(config.cache_dir(sc))
                try:
                    keep: dict = {}
                    # (the same seed as calculate(), so "new" is the report's M)
                    runs[name] = validate.run_validation(fmt, games, space, cards, inp.lazy["hands"], tmp, n_jobs, seed=i, n_rep=args.n_rep,
                                                         keep=keep, opts=opts)
                    keeps[name] = keep[fmt]
                finally:
                    cardvalue.cleanup(tmp)
                print(f"{sc} {fmt} {name}: {time.time() - t0:.0f}s", flush=True)
            if runs["new"].get("status") != "ok":
                res["formats"][fmt] = {"status": runs["new"].get("status"), "reason": runs["new"].get("reason")}
                continue
            y, groups = keeps["new"]["y"], keeps["new"]["groups"]
            p = {n: keeps[n]["pM"] for n in variants}
            row: dict = {"status": "ok", "n_games": int(len(y)), "n_events": int(groups.max() + 1)}
            for n in variants:
                m = runs[n]["metrics"]
                row[n] = {"logloss": m["M"]["logloss"], "logloss_ci": m["M"]["logloss_ci"],
                          "P_minus_M": m["P"]["vs_M"]["logloss"] if "P" in m else None}     # P - M (positive = M better), CI
                if "P" in m:
                    row[n]["P_logloss"] = m["P"]["logloss"]
                fits = keeps[n]["ref_fits"]
                row[n]["stages"] = {st: {"sigma": float(getattr(f, "sigma")), "w": float(getattr(f, "w", 1.0)),
                                         "scheme": getattr(f, "cv_scheme", "random"), "grid_lo": float(f.cv_grid[0]),
                                         "grid_hi": float(f.cv_grid[-1]), "best_index": int(np.argmin(f.cv_loss)) if f.cv_loss else None}
                                    for st, f in fits.items()}
                row[n]["lower_edge"] = {st: bool(f.cv_loss and int(np.argmin(f.cv_loss)) == 0) for st, f in fits.items()
                                        if st in ("bo1", "bo3")}
            # paired differences between the variants (positive = the first one is better): reference = the second
            for a, b in (("new", "old"), ("leaky", "old"), ("new", "leaky")):
                met = validate.cluster_metrics(y, {"M": p[a], "B": p[b]}, groups, args.n_rep, 5 + i)
                d = met["B"]["vs_M"]["logloss"]
                row[f"{a}_vs_{b}"] = dict(gain=d["diff"], lo=d["lo"], hi=d["hi"])
            # new M against P: the report's P - M_new difference, as a gain of M
            if row["new"]["P_minus_M"]:
                d = row["new"]["P_minus_M"]
                row["new_vs_P"] = dict(gain=d["diff"], lo=d["lo"], hi=d["hi"])
            if row["old"]["P_minus_M"]:
                d = row["old"]["P_minus_M"]
                row["old_vs_P"] = dict(gain=d["diff"], lo=d["lo"], hi=d["hi"])
            res["formats"][fmt] = row
            f = lambda d: f"{d['gain']:+.5f} [{d['lo']:+.5f}, {d['hi']:+.5f}]"          # noqa: E731
            print(f"{sc} {fmt}: logloss old {row['old']['logloss']:.5f} leaky {row['leaky']['logloss']:.5f} new {row['new']['logloss']:.5f}"
                  f" P {row['new'].get('P_logloss', float('nan')):.5f} | new-vs-old {f(row['new_vs_old'])} | new-vs-P "
                  f"{f(row['new_vs_P']) if 'new_vs_P' in row else '-'} | old-vs-P {f(row['old_vs_P']) if 'old_vs_P' in row else '-'}", flush=True)
            for st, v in row["new"]["stages"].items():
                o = row["old"]["stages"][st]
                print(f"    {st}: sigma old {o['sigma']:.4f} -> new {v['sigma']:.4f}, w {v['w']:.1f}, lower edge old/new "
                      f"{row['old']['lower_edge'].get(st)}/{row['new']['lower_edge'].get(st)}", flush=True)
        (out / f"compare_{sc}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    sys.exit(main())
