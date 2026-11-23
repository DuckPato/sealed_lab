"""Validation of the deck families on real pools (read-only use of the app's models).

    python scripts/family_validation.py SET FMT [--n 20] [--seed 1] [--jobs N] [--out FILE.json]

For N random real pools of the format (corpus pools.parquet): the family build (timing, families shown, P(best) of #1, core / flex
sizes), the classic builder (best deck + 2 alternatives >= 10 cards apart) for comparison:
  - is family #1 at least as good as the classic best deck under the full-data fit? (and by how much)
  - did the classic rule hide a close alternative (a shown family < 10 nonland cards away from #1 with P(best) >= 5%)?
  - did the classic rule show a clearly inferior alternative (its 80% gap interval to #1 entirely > 1 pp)?
Prints a summary; --out writes the per-pool rows as JSON.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def pick_pools(set_code: str, fmt: str, n: int, seed: int, min_cards: int = 60) -> list[tuple[str, dict[int, int]]]:
    import polars as pl

    from sealed_lab import config

    cd = config.corpus_dir(set_code)
    g = pl.read_parquet(cd / "games.parquet", columns=["fmt", "draft_id"]).filter(pl.col("fmt") == fmt)
    ids = sorted(g["draft_id"].unique().to_list())
    pools = pl.read_parquet(cd / "pools.parquet").filter(pl.col("draft_id").is_in(ids))
    big = sorted(pools.group_by("draft_id").agg(pl.col("n").sum().alias("tot")).filter(pl.col("tot") >= min_cards)["draft_id"].to_list())
    rng = np.random.default_rng(seed)
    out = []
    for d in rng.choice(big, min(n, len(big)), replace=False):
        p = pools.filter(pl.col("draft_id") == d)
        out.append((str(d), dict(zip(p["card_id"].to_list(), p["n"].to_list()))))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("set")
    ap.add_argument("fmt")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    from scipy.special import expit

    from sealed_lab.deck import builder, families

    m = builder.load_model(args.set.upper())
    rows = []
    for k, (did, pool) in enumerate(pick_pools(args.set.upper(), args.fmt, args.n, args.seed)):
        t0 = time.time()
        res = builder.build(m, pool, args.fmt, n_decks=families.MAX_FAMILIES, n_jobs=args.jobs)
        t_new = time.time() - t0
        t0 = time.time()
        old = builder.build(m, pool, args.fmt, n_decks=3, min_diff=10, families=False)
        t_old = time.time() - t0
        top = {c.card_id: c.count for c in res.decks[0].cards if c.card_id is not None}
        Lt = np.asarray(m.logit(top, args.fmt, draws=True))
        old_decks = [{c.card_id: c.count for c in e.cards if c.card_id is not None} for e in old.decks]
        ctx = builder.Ctx(m, args.fmt)
        old_rows = []
        for j, od in enumerate(old_decks):
            Lo = np.asarray(m.logit(od, args.fmt, draws=True))
            gap = 100 * (expit(Lt) - expit(Lo))                        # pp, #1 minus old deck, per draw
            lo, hi = np.percentile(gap[1:], [10, 90])
            old_rows.append(dict(j=j, wr=float(expit(Lo[0])), gap_pp=float(gap[0]), gap_lo=float(lo), gap_hi=float(hi),
                                 diff_to_family1=builder.deck_diff(ctx, top, od)))
        fam = res.families
        hidden = [f["rank"] for f in fam[1:] if f["diff_to_top"] < 10 and f["p_best"] >= families.SHOW_P_BEST]
        inferior = [r["j"] for r in old_rows[1:] if r["gap_lo"] > 1.0]
        row = dict(pool=did, cards=sum(pool.values()), seconds=t_new, seconds_old=t_old, n_shown=len(fam), n_families=res.family_meta["n_families"],
                   n_candidates=res.family_meta["n_candidates"], p_best_top=fam[0]["p_best"], wr_top=fam[0]["wr"],
                   core=sum(c["count"] for c in fam[0]["core"]), flex=sum(c["count"] for c in fam[0]["flex"]),
                   old_best_wr=old_rows[0]["wr"], top_minus_old_pp=old_rows[0]["gap_pp"], old_best_diff=old_rows[0]["diff_to_family1"],
                   hidden_close=hidden, old_inferior=inferior, old=old_rows,
                   families=[dict(rank=f["rank"], colors=f["colors"] + ("+" + f["splash"] if f["splash"] else ""), p_best=f["p_best"], gap_pp=f["gap_pp"],
                                  lo=f["gap_lo_pp"], hi=f["gap_hi_pp"], diff=f["diff_to_top"], n_members=f["n_members"],
                                  core=sum(c["count"] for c in f["core"]), flex=sum(c["count"] for c in f["flex"]), why=f["why"]) for f in fam])
        rows.append(row)
        print(f"{args.set} {args.fmt} pool {k + 1:2d} {did[:6]}: {t_new:5.1f}s (classic {t_old:4.1f}s) shown {len(fam)}/{row['n_families']} "
              f"P(best #1) {row['p_best_top']:.2f} core/flex {row['core']}/{row['flex']} top-old {row['top_minus_old_pp']:+.2f}pp "
              f"hidden-close {hidden} old-inferior {inferior}", flush=True)
    n = len(rows)
    if n:
        ns = np.array([r["n_shown"] for r in rows])
        sec = np.array([r["seconds"] for r in rows])
        d_old = np.array([r["top_minus_old_pp"] for r in rows])
        print(f"\n== {args.set} {args.fmt}: {n} pools ==")
        print("families shown: " + ", ".join(f"{k}: {(ns == k).sum()}" for k in range(1, families.MAX_FAMILIES + 1)))
        print(f"P(best) of #1: median {np.median([r['p_best_top'] for r in rows]):.2f}, min {min(r['p_best_top'] for r in rows):.2f}")
        print(f"core/flex of #1: median {np.median([r['core'] for r in rows]):.0f} / {np.median([r['flex'] for r in rows]):.0f}")
        print(f"time: median {np.median(sec):.1f}s, max {sec.max():.1f}s (first build includes starting the worker pool)")
        print(f"#1 vs classic best (full-data fit): >= 0 in {(d_old >= -1e-9).sum()}/{n}; mean {d_old.mean():+.2f}pp, min {d_old.min():+.2f}pp")
        print(f"classic rule hid a close alternative in {sum(1 for r in rows if r['hidden_close'])}/{n} pools; "
              f"showed a clearly inferior alternative in {sum(1 for r in rows if r['old_inferior'])}/{n}")
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
