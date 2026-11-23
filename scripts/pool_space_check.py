"""Real-pool check of the deployed builder (read-only on the app home).

Per set x format, N real Sealed pools: build with the deployed model (draw-0 search, single deck), check legality and
Ctx.logit == SetModel.logit, and compare the model's score of the builder's deck with the deck the player registered.
For player decks outside the builder's colour configurations, a local search from the player's deck (permissive configuration)
shows whether the model itself prefers something the configuration space cannot reach.

    python scripts/pool_space_check.py OUT.json N_POOLS JOBS [SET,SET,...]      (summary: scripts/pool_space_report.py OUT.json ...)

Session 12 audit (2026-11-12), results in docs/audit_1112/.  Reads the app home (SEALED_LAB_HOME, default the real one) and
never writes to it; OUT goes wherever you point it.  Several of these may run side by side as plain processes (JOBS = 1);
only one spawn pool at a time on this machine.
"""
import json
import multiprocessing
import os
import sys
from pathlib import Path
import time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _all_sets() -> list[str]:
    from sealed_lab import config

    return sorted(p.name for p in config.sets_dir().iterdir() if p.is_dir())
PP = 100 * 0.55 * 0.45


def task(args):
    set_code, fmt, n_pools, scratch = args
    os.environ["POLARS_MAX_THREADS"] = "2"
    import numpy as np
    import polars as pl
    from scipy.special import expit

    from sealed_lab import config
    from sealed_lab.deck import builder

    model = builder.load_model(set_code)
    if fmt not in model.formats:
        return dict(set=set_code, fmt=fmt, rows=[], error="format not in model")
    ctx = builder.Ctx(model, fmt)
    # never write the app home: when the creature-range cache lacks this format, compute it into a scratch folder
    cf = config.cache_dir(set_code) / "creature_support.json"
    have = cf.exists() and fmt in json.loads(cf.read_text(encoding="utf-8")).get("formats", {})
    if not have:
        real = config.cache_dir
        config.cache_dir = lambda s: Path(scratch) / "cache" / s          # type: ignore[assignment]
        try:
            builder.creature_support(model, fmt)
        finally:
            config.cache_dir = real                                           # type: ignore[assignment]
    cd = config.corpus_dir(set_code)
    builds = (pl.read_parquet(cd / "builds.parquet").filter(pl.col("fmt") == fmt)
              .join(pl.read_parquet(cd / "decks.parquet", columns=["deck_key", "n_cards", "n_lands"]), on="deck_key")
              .filter(pl.col("n_cards") == 40).sort("games", descending=True).unique("draft_id", keep="first", maintain_order=True))
    builds = builds.sort("draft_id").sample(n=min(n_pools, builds.height), seed=11, shuffle=True)
    ids = builds["draft_id"].to_list()
    pools = pl.read_parquet(cd / "pools.parquet").filter(pl.col("draft_id").is_in(ids))
    dcs = pl.read_parquet(cd / "deck_cards.parquet").filter(pl.col("deck_key").is_in(builds["deck_key"].to_list()))
    pool_of = {k[0]: dict(zip(v["card_id"].to_list(), v["n"].to_list())) for k, v in pools.group_by("draft_id")}
    deck_of = {k[0]: dict(zip(v["card_id"].to_list(), v["n"].to_list())) for k, v in dcs.group_by("deck_key")}
    rows = []
    for did, dk, games, wins in builds.select("draft_id", "deck_key", "games", "wins").iter_rows():
        pool = {int(i): int(n) for i, n in pool_of.get(did, {}).items() if not ctx.is_basic[int(i)]}
        human = {int(i): int(n) for i, n in deck_of.get(dk, {}).items()}
        if not pool or sum(human.values()) != 40:
            continue
        r = dict(draft_id=did, games=games, wins=wins)
        t0 = time.time()
        try:
            res = builder.build(model, pool, fmt, n_decks=1, families=False)
        except Exception as e:  # noqa: BLE001
            r["error"] = repr(e)
            rows.append(r)
            continue
        r["sec"] = time.time() - t0
        ev = res.decks[0]
        deck = {int(c.card_id): int(c.count) for c in ev.cards if c.card_id is not None and c.count > 0}
        # ---- legality of the builder's deck
        n = sum(deck.values())
        over = [i for i, k in deck.items() if not ctx.is_basic[i] and k > pool.get(i, 0)]
        nl = builder.nl_of(ctx, deck)
        lg_b = ctx.logit(deck)
        lg_model = float(model.logit(deck, fmt))
        r.update(n_cards=n, over_pool=len(over), n_lands=nl, colors=ev.colors, splash=ev.splash, q_b=float(expit(lg_b)),
                 ctx_vs_model=abs(lg_b - lg_model), eval_gwr=float(ev.game_wr), warnings=len(res.warnings))
        # ---- the same build with the sign of Searcher._deficit flipped (positive = the colour needs more sources, as its callers assume)
        orig = builder.Searcher._deficit
        builder.Searcher._deficit = lambda self, d, c, _o=orig: -_o(self, d, c)
        try:
            t1 = time.time()
            res2 = builder.build(model, pool, fmt, n_decks=1, families=False)
            deck2 = {int(c.card_id): int(c.count) for c in res2.decks[0].cards if c.card_id is not None and c.count > 0}
            lg2b = ctx.logit(deck2)
            r.update(fix_pp=100 * float(expit(lg2b) - expit(lg_b)), fix_same=deck2 == deck, fix_sec=time.time() - t1,
                     fix_colors=res2.decks[0].colors + "+" + res2.decks[0].splash, n_colour_basics=sum(1 for i, k in deck.items() if ctx.is_basic[i] and k > 0),
                     fix_lands=builder.nl_of(ctx, deck2))
        finally:
            builder.Searcher._deficit = orig
        # ---- the player's deck under the same model
        human_in_pool = all(ctx.is_basic[i] or k <= pool.get(i, 0) for i, k in human.items())
        lg_h = ctx.logit(human)
        prep = builder.prepare(ctx, pool, None, None)
        bad_best, key_best = 10 ** 9, ""
        for cfg in prep.configs:
            bad = sum(k for i, k in human.items() if not ctx.is_basic[i] and
                      ((ctx.is_land[i] and not cfg.land_ok[i]) or (not ctx.is_land[i] and cfg.cls[i] == 0)))
            bad += max(0, sum(k for i, k in human.items() if cfg.cls[i] == 2) - builder.SPLASH_CAP)
            if bad < bad_best:
                bad_best, key_best = bad, cfg.key
        spell_cols = int((np.array([ctx.ca.pips[i] * k for i, k in human.items() if not ctx.is_land[i]]).sum(0) > 0).sum())
        nl_h = builder.nl_of(ctx, human)
        r.update(human_in_pool=human_in_pool, q_h=float(expit(lg_h)), gap_pp=100 * float(expit(lg_b) - expit(lg_h)), human_bad=int(bad_best),
                 human_cfg=key_best, human_cols=spell_cols, human_lands=nl_h, lands_supported=list(prep.nls))
        if bad_best > 0 or lg_h > lg_b:
            # local search from the player's deck in a configuration that allows all its cards
            cfg = builder._config_of_deck(ctx, human, prep.configs, prep.pool)
            lo, hi = min(prep.nl_bounds[0], nl_h), max(prep.nl_bounds[1], nl_h)
            try:
                s = builder.Searcher(ctx, cfg, prep.pool, (lo, hi), cre_bounds=None)
                d2, lg2 = s.run(human)
                r.update(q_h_ls=float(expit(lg2)), ls_minus_builder_pp=100 * float(expit(lg2) - expit(lg_b)),
                         ls_lands=builder.nl_of(ctx, d2), ls_changed=builder.deck_diff(ctx, d2, human))
            except Exception as e:  # noqa: BLE001
                r["ls_error"] = repr(e)
        rows.append(r)
    return dict(set=set_code, fmt=fmt, rows=rows)


if __name__ == "__main__":
    out, n_pools, jobs = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    sets = sys.argv[4].split(",") if len(sys.argv) > 4 else _all_sets()
    scratch = os.path.dirname(out)
    tasks = [(s, f, n_pools, scratch) for s in sets for f in ("bo3", "bo1")]
    res = []
    t0 = time.time()
    if jobs <= 1:
        for t in tasks:
            res.append(task(t))
            print(t[0], t[1], len(res[-1]["rows"]), round(time.time() - t0), flush=True)
    else:
        with ProcessPoolExecutor(jobs, mp_context=multiprocessing.get_context("spawn")) as ex:
            for r in ex.map(task, tasks, chunksize=1):
                res.append(r)
                print(r["set"], r["fmt"], len(r["rows"]), r.get("error", ""), round(time.time() - t0), flush=True)
                with open(out, "w", encoding="utf-8") as f:
                    json.dump(res, f)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f)
