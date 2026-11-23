"""The targeted pool set T of docs/archetype_window/RULE.md (written and saved BEFORE any search is run).

    python scripts/archetype_window_pools.py SRC R_POOLS_JSON OUT_JSON [--per-cell 40] [--seed 0] [--sets ECL EOE ...]

SRC       a scratch home (SEALED_LAB_HOME-style folder) with sets/SET/{corpus,cache,model,registry.json}: the FULL-data models
          (a byte copy of the app home's, never the app home itself: group windows are cached into SRC/sets/SET/cache).
R_POOLS_JSON  docs/builder_scoreboard/search_check.json (the 220 random pools R; T never repeats a draft_id of R).
OUT_JSON  `{"pools": [{set, fmt, draft_id, logit_default: null, group, group_label, window, global_window, ...}], "summary": {...}}`,
          the format `search_check.load_pool_tasks` reads.

Held-out events = the events that start at or after the cut of `split_half_search.training_events` (the same cut as every other
split-half / deviation-cost test).  Affected group (rule): a group with >= 400 games in the format whose own window (`builder.
group_creature_support`, the full-data model of SRC) is at least 2 creatures wider than the global one (`builder.creature_support`) at
either end: lo_g <= lo - 2 or hi_g >= hi + 2.  A held-out pool is eligible when the real deck its player submitted (the 40-card deck
with most games of that event in that format) is assigned (`Archetypes.assign_matrix`) to an affected group.  Per affected set x format
up to --per-cell eligible pools not in R are drawn at random (numpy default_rng seeded from --seed, the set and the format).
No model is calculated and no search is run; the app home is never touched (SRC only).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

SETS = ["ECL", "EOE", "FIN", "FRA", "HOB", "MSH", "OM1", "SOS", "TDM", "TLA", "TMT"]
FORMATS = ("bo1", "bo3")
WIDER_BY = 2            # a group is affected when its window is at least this much wider than the global one at either end


def is_affected(win: tuple[int, int], glob: tuple[int, int]) -> bool:
    return win[0] <= glob[0] - WIDER_BY or win[1] >= glob[1] + WIDER_BY


def cell_seed(seed: int, set_code: str, fmt: str) -> int:
    return int(zlib.crc32(f"{seed}:{set_code}:{fmt}:archetype_window".encode()) % (2 ** 31))


def draw_pools(eligible: list[str], k: int, seed: int) -> list[str]:
    """Up to k of the eligible draft ids, at random (deterministic for a given list and seed)."""
    ids = sorted(eligible)
    if len(ids) <= k:
        return ids
    rng = np.random.default_rng(seed)
    return sorted(ids[i] for i in rng.choice(len(ids), size=k, replace=False))


def held_out_real_decks(games, decks, cut) -> dict[tuple[str, str], int]:
    """{(draft_id, fmt): deck_key} of the held-out events: the 40-card deck with the most games of that event in that format
    (ties: the smaller deck_key)."""
    import polars as pl

    from sealed_lab.model import design

    ev = design.event_starts(games).filter(pl.col("start") >= cut)
    d40 = decks.filter(pl.col("n_cards") == 40).select("deck_key")
    g = (games.filter(pl.col("draft_id").is_in(ev["draft_id"].to_list())).join(d40, on="deck_key")
         .group_by("draft_id", "fmt", "deck_key").len().sort(["draft_id", "fmt", "len", "deck_key"], descending=[False, False, True, False])
         .unique(subset=["draft_id", "fmt"], keep="first", maintain_order=True))
    return {(str(a), str(f)): int(k) for a, f, k in zip(g["draft_id"].to_list(), g["fmt"].to_list(), g["deck_key"].to_list())}


def one_set(src: Path, set_code: str, r_ids: set[str], per_cell: int, seed: int) -> dict:
    import polars as pl

    from split_half_search import training_events

    from sealed_lab import config
    from sealed_lab.deck import builder
    from sealed_lab.model import archetypes as arch_mod
    from sealed_lab.model.artifact import SetModel

    os.environ["SEALED_LAB_HOME"] = str(src)
    cd = config.corpus_dir(set_code)
    games = pl.read_parquet(cd / "games.parquet")
    decks = pl.read_parquet(cd / "decks.parquet", columns=["deck_key", "n_cards"])
    pools = pl.read_parquet(cd / "pools.parquet", columns=["draft_id"]).unique()
    _ids, _starts, _fmts, cut = training_events(games)
    model = SetModel.load(set_code)
    arch = model.archetypes
    out: dict = dict(set=set_code, cells={}, pools=[], cut=str(cut))
    if arch is None:
        out["note"] = "no archetypes in the model"
        return out
    real = held_out_real_decks(games, decks, cut)
    have = set(pools["draft_id"].to_list())
    real = {k: v for k, v in real.items() if k[0] in have}
    keys = np.array(sorted(set(real.values())), dtype=np.int64)
    if not len(keys):
        return out
    dc = pl.read_parquet(cd / "deck_cards.parquet", columns=["deck_key", "card_id", "n"]).filter(pl.col("deck_key").is_in(keys.tolist()))
    dc = dc.filter((dc["card_id"] >= 0) & (dc["card_id"] < arch.n_cards))
    X = arch_mod.deck_matrix(keys, dc, arch.n_cards, np.asarray(model._spell, dtype=bool))
    grp, _ = arch.assign_matrix(X)
    group_of = {int(k): int(g) for k, g in zip(keys, grp)}
    cre = np.asarray(model.card_arrays.is_creature, dtype=bool) & ~np.asarray(model.card_arrays.is_land, dtype=bool)
    ncre = np.asarray(X @ cre.astype(float)).ravel()
    cre_of = {int(k): int(round(v)) for k, v in zip(keys, ncre)}
    for fmt in FORMATS:
        if fmt not in model.formats:
            continue
        glob = builder.creature_support(model, fmt)
        win = builder.group_creature_support(model, fmt)
        if glob is None or not win:
            continue
        aff = {a: w for a, w in win.items() if is_affected(w, glob)}
        cell: dict = dict(global_window=list(glob), affected={}, eligible=0, in_R=0, chosen=0)
        for a, w in sorted(aff.items()):
            cell["affected"][str(a)] = dict(label=arch.info[a]["label"], window=list(w))
        elig = [(d, dk) for (d, f), dk in real.items() if f == fmt and group_of[dk] in aff]
        cell["eligible_total"] = len(elig)
        cell["in_R"] = sum(1 for d, _ in elig if d in r_ids)
        elig = [(d, dk) for d, dk in elig if d not in r_ids]
        cell["eligible"] = len(elig)
        chosen = draw_pools([d for d, _ in elig], per_cell, cell_seed(seed, set_code, fmt))
        cell["chosen"] = len(chosen)
        dk_of = dict(elig)
        for d in chosen:
            a = group_of[dk_of[d]]
            out["pools"].append(dict(set=set_code, fmt=fmt, draft_id=d, logit_default=None, group=a, group_label=arch.info[a]["label"],
                                     window=list(aff[a]), global_window=list(glob), real_deck_creatures=cre_of[dk_of[d]]))
        out["cells"][fmt] = cell
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("r_pools")
    ap.add_argument("out")
    ap.add_argument("--per-cell", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sets", nargs="*", default=SETS)
    a = ap.parse_args(argv)
    src = Path(a.src).resolve()
    r_ids = {str(e["draft_id"]) for e in json.loads(Path(a.r_pools).read_text(encoding="utf-8"))["pools"]}
    pools: list[dict] = []
    summary: dict = {}
    print(f"{'set':5} {'fmt':4} {'affected groups (window; global)':64} {'eligible':>8} {'in R':>5} {'chosen':>6}")
    for s in a.sets:
        if not (src / "sets" / s / "model").exists():
            print(f"{s:5} (no model in SRC)")
            continue
        r = one_set(src, s, r_ids, a.per_cell, a.seed)
        pools += r["pools"]
        summary[s] = dict(cut=r.get("cut"), cells=r["cells"])
        for fmt, c in r["cells"].items():
            desc = "; ".join(f"{v['label'][:34]} {v['window'][0]}-{v['window'][1]}" for v in c["affected"].values()) or "-"
            print(f"{s:5} {fmt:4} {desc[:50]:50} (g {c['global_window'][0]}-{c['global_window'][1]}) {c['eligible']:>8} {c['in_R']:>5} {c['chosen']:>6}")
    Path(a.out).write_text(json.dumps(dict(pools=pools, summary=summary, per_cell=a.per_cell, seed=a.seed), indent=1), encoding="utf-8")
    print(f"T: {len(pools)} pools -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
