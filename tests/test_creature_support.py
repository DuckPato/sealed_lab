from __future__ import annotations

import numpy as np

from sealed_lab.deck import builder
from sealed_lab.deck.milp import MilpSpec, solve_milp


def test_relaxed_bounds_stay_feasible():
    rb = builder.relaxed_creature_bounds
    assert rb(None, 23, 10, 20) is None
    assert rb((12, 17), 23, 30, 30) == (12, 17)
    assert rb((12, 17), 23, 9, 30) == (9, 17)            # only 9 creatures in the pool
    assert rb((12, 17), 23, 30, 4) == (12, 19)           # only 4 other playables: 19 creatures forced
    assert rb((12, 17), 23, 30, 30, locked_cre=18) == (12, 18)
    assert rb((12, 17), 23, 30, 30, locked_non=13) == (10, 17)


def test_milp_honours_the_creature_range():
    # 10 creatures worth 1.0, 20 non-creatures worth 2.0: unconstrained the MILP takes the non-creatures
    n = 30
    coef = np.array([1.0] * 10 + [2.0] * 20)
    creature = np.array([True] * 10 + [False] * 20)
    base = dict(coef=coef, copies=np.ones(n, dtype=int), is_land=np.zeros(n, dtype=bool), produces=np.zeros((n, 5), dtype=bool),
                splash=np.zeros(n, dtype=bool), basic_colors=[0], floors={}, n_lands=17, deck_size=40)
    free = solve_milp(MilpSpec(**base))[0]
    assert int(free.counts[creature].sum()) == 3
    con = solve_milp(MilpSpec(**base, creature=creature, creature_range=(8, 12)))[0]
    assert 8 <= int(con.counts[creature].sum()) <= 12
    assert int(con.counts[creature].sum()) == 8           # the cheapest way to satisfy the floor


def test_build_respects_the_support(model, monkeypatch):
    import json

    from sealed_lab import config

    fmt = model.formats[0]
    lo_hi = builder.creature_support(model, fmt)
    assert lo_hi is not None and lo_hi[0] <= lo_hi[1]
    cache = json.loads((config.cache_dir(model.set_code) / "creature_support.json").read_text(encoding="utf-8"))
    assert fmt in cache["formats"]
    # force a narrow band and check the builder's best deck lands inside it (when the pool allows)
    monkeypatch.setattr(model, "_cre_support", {fmt: (14, 15)}, raising=False)
    pool = {int(i): 1 for i in model.cards.filter(~model.cards["is_land"])["card_id"].to_list()[:60]}
    res = builder.build(model, pool, fmt, n_decks=1, families=False)
    deck = {c.card_id: c.count for c in res.decks[0].cards if c.card_id is not None}
    ca = model.card_arrays
    n_cre = sum(n for i, n in deck.items() if not ca.is_land[i] and ca.is_creature[i])
    assert 14 <= n_cre <= 15
