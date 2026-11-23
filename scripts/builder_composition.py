"""Does the builder build decks with the composition real players use? (creatures, 2-drops, lands, splash)

    python scripts/builder_composition.py SET OUTDIR [--fmt bo1 bo3] [--pools 120] [--jobs N]

For a random sample of real pools (deck + sideboard of 17lands builds), the builder's best deck (top family, the deployed
model in the normal home, read-only) is compared with the decks the players registered from the same pool: all players and
top players (17lands win-rate bucket >= 0.60). Prints the distributions, the paired builder-minus-player differences and the
share of builder decks outside the 3%-97% range of real decks. Writes OUTDIR/builder_composition_SET.json.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import multiprocessing
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TOP_SKILL = 0.60


def _build(task: dict) -> dict:
    from sealed_lab.deck import builder

    model = builder.load_model(task["set"])
    res = builder.build(model, task["pool"], task["fmt"], n_decks=1, families=True, n_jobs=1)
    best = {c.card_id: c.count for c in res.decks[0].cards if c.card_id is not None}
    return dict(draft_id=task["draft_id"], deck=best)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("set")
    ap.add_argument("outdir")
    ap.add_argument("--fmt", nargs="+", default=["bo1", "bo3"])
    ap.add_argument("--pools", type=int, default=120)
    ap.add_argument("--jobs", type=int, default=None)
    args = ap.parse_args(argv)

    import polars as pl

    from sealed_lab import config
    from sealed_lab.model.artifact import SetModel

    sc = args.set.upper()
    cd = config.corpus_dir(sc)
    games = pl.read_parquet(cd / "games.parquet")
    pools = pl.read_parquet(cd / "pools.parquet")
    dcards = pl.read_parquet(cd / "deck_cards.parquet")
    decks = pl.read_parquet(cd / "decks.parquet")
    cards = SetModel.load(sc).cards
    is_land = dict(zip(cards["card_id"].to_list(), cards["is_land"].to_list()))
    is_cre = dict(zip(cards["card_id"].to_list(), cards["is_creature"].to_list()))
    mv = dict(zip(cards["card_id"].to_list(), cards["mana_value"].fill_null(0).to_list()))

    def comp(deck: dict[int, int]) -> dict:
        sp = {c: n for c, n in deck.items() if not is_land.get(c, False)}
        return dict(creatures=sum(n for c, n in sp.items() if is_cre.get(c, False)),
                    two_drops=sum(n for c, n in sp.items() if mv.get(c, 0) == 2),
                    lands=sum(n for c, n in deck.items() if is_land.get(c, False)),
                    mean_mv=sum(mv.get(c, 0) * n for c, n in sp.items()) / max(sum(sp.values()), 1))

    out = {"set": sc, "formats": {}}
    rng = np.random.default_rng(1)
    for fmt in args.fmt:
        g = (games.filter(pl.col("fmt") == fmt).join(decks.select("deck_key", "n_cards"), on="deck_key", how="left")
             .filter(pl.col("n_cards") == 40))
        per_deck = g.group_by("draft_id", "deck_key").agg(pl.col("skill").mean().alias("skill"))
        ids = sorted(set(per_deck["draft_id"].to_list()) & set(pools["draft_id"].unique().to_list()))
        if not ids:
            continue
        ids = sorted(rng.choice(ids, min(args.pools, len(ids)), replace=False).tolist())
        pool_by = {}
        for (did,), p in pools.filter(pl.col("draft_id").is_in(ids)).partition_by("draft_id", as_dict=True).items():
            pool_by[did] = {int(c): int(n) for c, n in p.select("card_id", "n").iter_rows()}
        tasks = [dict(set=sc, fmt=fmt, draft_id=d, pool=pool_by[d]) for d in ids if d in pool_by]
        ctx = multiprocessing.get_context("spawn")
        with cf.ProcessPoolExecutor(max_workers=args.jobs or os.cpu_count(), mp_context=ctx) as ex:
            built = {r["draft_id"]: comp(r["deck"]) for r in ex.map(_build, tasks, chunksize=2)}
        pd = per_deck.filter(pl.col("draft_id").is_in(list(built)))
        real_all, real_top, diff_all, diff_top = [], [], [], []
        for did, key, sk in pd.select("draft_id", "deck_key", "skill").iter_rows():
            deck = {int(c): int(n) for c, n in dcards.filter(pl.col("deck_key") == key).select("card_id", "n").iter_rows()}
            r = comp(deck)
            b = built[did]
            real_all.append(r)
            diff_all.append({k: b[k] - r[k] for k in r})
            if sk is not None and sk >= TOP_SKILL:
                real_top.append(r)
                diff_top.append({k: b[k] - r[k] for k in r})
        res = {"n_pools": len(built), "n_real": len(real_all), "n_top": len(real_top)}
        print(f"\n{sc} {fmt}: {len(built)} pools, {len(real_all)} player decks ({len(real_top)} by top players)")
        print(f"  {'':10s} {'builder':>16s} {'players':>16s} {'top players':>16s}   builder-player   builder-top   builder outside real 3-97%")
        for k in ("creatures", "two_drops", "lands", "mean_mv"):
            bv = np.array([v[k] for v in built.values()])
            av = np.array([v[k] for v in real_all])
            tv = np.array([v[k] for v in real_top]) if real_top else np.array([np.nan])
            da = np.array([d[k] for d in diff_all])
            dt = np.array([d[k] for d in diff_top]) if diff_top else np.array([np.nan])
            lo, hi = np.percentile(av, 3), np.percentile(av, 97)
            outside = float(np.mean((bv < lo) | (bv > hi)))
            res[k] = dict(builder_mean=float(bv.mean()), builder_sd=float(bv.std()), players_mean=float(av.mean()),
                          players_sd=float(av.std()), top_mean=float(np.nanmean(tv)), diff_players=float(da.mean()),
                          diff_players_se=float(da.std() / max(np.sqrt(len(da)), 1)), diff_top=float(np.nanmean(dt)),
                          real_range=[float(lo), float(hi)], builder_outside=outside,
                          builder_hist={int(v): int(c) for v, c in zip(*np.unique(np.round(bv), return_counts=True))})
            print(f"  {k:10s} {bv.mean():7.2f} ± {bv.std():5.2f}  {av.mean():7.2f} ± {av.std():5.2f}  {np.nanmean(tv):7.2f}          "
                  f"{da.mean():+6.2f} ± {res[k]['diff_players_se']:.2f}   {np.nanmean(dt):+6.2f}     {100 * outside:4.1f}% "
                  f"(range {lo:.1f}-{hi:.1f})")
        out["formats"][fmt] = res
    od = Path(args.outdir)
    od.mkdir(parents=True, exist_ok=True)
    (od / f"builder_composition_{sc}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
