"""Bo3, games 2-3: how much would it be worth to play the best of the builder's candidate decks AGAINST THE OPPONENT'S COLOURS
instead of always the main deck?  (read-only; deployed models; in-model estimate with a selection-noise correction)

For a pool: candidates = everything the draw-0 search ranks (all colour configurations).  L_D = deck logit (no matchups),
M_D[o] = sum_c n_c * centred matchup effect of card c against opponent colours o, re-centred so that its mean over the format's
opponents is zero (only the opponent-specific part counts).

    gain(sel, ev) = sum_o share_o * [ q(L_pick + M_pick[ev, o]) - q(L_main + M_main[ev, o]) ],   pick = argmax_D L_D + M_D[sel, o]

G00 = gain(draw 0, draw 0) is what the model sees.  Gbb (select and evaluate under the same bootstrap replicate) and Gbb' (select
under one replicate, evaluate under another) bracket the optimism from choosing on noisy matchup estimates:
corrected ~ 2 Gbb' - Gbb.

    python scripts/matchup_switch_check.py OUT.json N_POOLS SET,SET,... [bo3,bo1]

Session 12 audit (2026-11-12), results in docs/audit_1112/.  Reads the app home (SEALED_LAB_HOME, default the real one) and
never writes to it; OUT goes wherever you point it.  Several of these may run side by side as plain processes (JOBS = 1);
only one spawn pool at a time on this machine.
"""
import json
import os
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def task(set_code, fmt, n_pools, scratch):
    os.environ["POLARS_MAX_THREADS"] = "2"
    import numpy as np
    import polars as pl
    from scipy.special import expit

    from sealed_lab import config
    from sealed_lab.deck import builder

    model = builder.load_model(set_code)
    if fmt not in model.formats or model.opp is None or fmt not in model.opp_formats:
        return dict(set=set_code, fmt=fmt, rows=[], error="no matchup layer in this format")
    ctx = builder.Ctx(model, fmt)
    opm = model.opp
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
    g = pl.read_parquet(cd / "games.parquet", columns=["fmt", "opp_colors"]).filter(pl.col("fmt") == fmt)
    c = g.group_by("opp_colors").len().sort("len", descending=True)
    opps = c["opp_colors"].to_list()
    share = (c["len"] / c["len"].sum()).to_numpy()
    E = np.stack([opm.effect(o, draws=True) for o in opps])                # (n_opp, B+1, n_cards)
    B = E.shape[1] - 1
    builds = (pl.read_parquet(cd / "builds.parquet").filter(pl.col("fmt") == fmt)
              .sort("games", descending=True).unique("draft_id", keep="first", maintain_order=True))
    builds = builds.sort("draft_id").sample(n=min(n_pools, builds.height), seed=23, shuffle=True)
    ids = builds["draft_id"].to_list()
    pools = pl.read_parquet(cd / "pools.parquet").filter(pl.col("draft_id").is_in(ids))
    pool_of = {k[0]: dict(zip(v["card_id"].to_list(), v["n"].to_list())) for k, v in pools.group_by("draft_id")}
    rows = []
    for did in ids:
        pool = {int(i): int(n) for i, n in pool_of.get(did, {}).items() if not ctx.is_basic[int(i)]}
        if not pool:
            continue
        try:
            prep = builder.prepare(ctx, pool, None, None)
            ranked, _, _ = builder.run_search(ctx, prep, n_decks=5)
        except Exception as e:  # noqa: BLE001
            rows.append(dict(draft_id=did, error=repr(e)))
            continue
        cands = ranked[:80]
        L = np.array([cd_.logit for cd_ in cands])
        M = np.zeros((len(cands), B + 1, len(opps)))
        for k, cd_ in enumerate(cands):
            ii = np.array([i for i in cd_.deck if not ctx.is_basic[i]], dtype=np.int64)
            nn = np.array([cd_.deck[int(i)] for i in ii], dtype=float)
            M[k] = (E[:, :, ii] @ nn).T                                      # (B+1, n_opp)
        M = M - (M * share[None, None, :]).sum(2, keepdims=True)             # opponent-specific part only
        diff = np.array([builder.deck_diff(ctx, cands[0].deck, cd_.deck) for cd_ in cands])

        def gain(sel, ev, mask=None):
            S = L[:, None] + M[:, sel, :]
            if mask is not None:
                S = np.where(mask[:, None], S, -np.inf)
            pick = S.argmax(0)                                               # per opponent
            oo = np.arange(len(opps))
            val = expit(L[pick] + M[pick, ev, oo]) - expit(L[0] + M[0, ev, oo])
            return float(share @ val), pick

        g00, pick0 = gain(0, 0)
        gbb = float(np.mean([gain(b, b)[0] for b in range(1, B + 1)])) if B > 0 else float("nan")
        gbx = float(np.mean([gain(b, b % B + 1)[0] for b in range(1, B + 1)])) if B > 1 else float("nan")
        small = diff <= 3
        g00s, _ = gain(0, 0, small)
        gbbs = float(np.mean([gain(b, b, small)[0] for b in range(1, B + 1)])) if B > 0 else float("nan")
        gbxs = float(np.mean([gain(b, b % B + 1, small)[0] for b in range(1, B + 1)])) if B > 1 else float("nan")
        keys = [cd_.cfg.main for cd_ in cands]
        rows.append(dict(draft_id=did, n_cands=len(cands), q_main=float(expit(L[0])),
                         main_matchup_sd_pp=100 * float(np.sqrt(share @ (expit(L[0] + M[0, 0, :]) - share @ expit(L[0] + M[0, 0, :])) ** 2)),
                         g00_pp=100 * g00, gbb_pp=100 * gbb, gbx_pp=100 * gbx, g00_small_pp=100 * g00s, gbb_small_pp=100 * gbbs,
                         gbx_small_pp=100 * gbxs, share_switch=float(share @ (pick0 != 0)),
                         share_other_colours=float(share @ np.array([keys[p] != keys[0] for p in pick0])),
                         mean_cards_changed=float(share @ diff[pick0]),
                         gap_to_second_cfg_pp=100 * float(expit(L[0]) - max([expit(L[k]) for k in range(len(cands)) if keys[k] != keys[0]],
                                                                             default=np.nan))))
    return dict(set=set_code, fmt=fmt, rows=rows, n_opps=len(opps), B=B)


if __name__ == "__main__":
    out, n_pools = sys.argv[1], int(sys.argv[2])
    sets = sys.argv[3].split(",")
    fmts = sys.argv[4].split(",") if len(sys.argv) > 4 else ["bo3"]
    res = []
    t0 = time.time()
    for s in sets:
        for f in fmts:
            res.append(task(s, f, n_pools, os.path.dirname(out)))
            print(s, f, len(res[-1]["rows"]), res[-1].get("error", ""), round(time.time() - t0), flush=True)
            with open(out, "w", encoding="utf-8") as fh:
                json.dump(res, fh)
