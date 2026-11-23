"""builder.ARCH_WINDOW (docs/archetype_window/RULE.md): group windows, validity, search; scripts / variant registration."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from sealed_lab.deck import builder

ROOT = Path(__file__).resolve().parents[1]
BASICS = ("Plains", "Island", "Swamp", "Mountain", "Forest")


def _load_script(name: str):
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def name2id(model):
    return {n: i for i, n in enumerate(model.cards.sort("card_id")["name"].to_list())}


@pytest.fixture(scope="module")
def held(calc):
    return json.loads((calc["synth"] / "heldout_pools.json").read_text(encoding="utf-8"))


@pytest.fixture()
def arch_model(model):
    if model.archetypes is None:
        pytest.skip("the synthetic model has no archetypes")
    return model


def _pool(h, name2id):
    return {name2id[n]: c for n, c in h["pool"].items() if n not in BASICS}


def _cre(model, deck):
    ca = model.card_arrays
    return sum(n for i, n in deck.items() if n > 0 and not ca.is_land[i] and ca.is_creature[i])


def _deck(ev):
    return {c.card_id: c.count for c in ev.cards if c.card_id is not None}


# ---- (a) the group window is the global window when every deck is one group --------------------------------------------------

def test_group_window_equals_global_with_one_group(arch_model, tmp_path, monkeypatch):
    model = arch_model
    fmt = model.formats[0]
    glob = builder.creature_support(model, fmt)
    assert glob is not None
    monkeypatch.setattr(model.archetypes, "assign_matrix", lambda X: (np.zeros(X.shape[0], dtype=np.int32), np.ones(X.shape[0])))
    monkeypatch.setattr(model, "_group_cre_support", {}, raising=False)
    monkeypatch.setattr(builder.config, "cache_dir", lambda s: tmp_path)
    assert builder.group_creature_support(model, fmt, min_games=0) == {0: glob}


def test_group_window_cache_and_min_games(arch_model, tmp_path, monkeypatch):
    model = arch_model
    fmt = model.formats[0]
    monkeypatch.setattr(builder.config, "cache_dir", lambda s: tmp_path)
    monkeypatch.setattr(model, "_group_cre_support", {}, raising=False)
    full = builder.group_creature_support(model, fmt, min_games=0)
    assert full and all(lo <= hi for lo, hi in full.values())
    cache = json.loads((tmp_path / "creature_support_groups.json").read_text(encoding="utf-8"))
    assert fmt in cache["formats"]
    monkeypatch.setattr(model, "_group_cre_support", {}, raising=False)
    assert builder.group_creature_support(model, fmt, min_games=0) == full          # read back from the cache file
    monkeypatch.setattr(model, "_group_cre_support", {}, raising=False)
    assert builder.group_creature_support(model, fmt, min_games=10 ** 12) == {}      # below the games floor: no window of its own
    # other centroids (a recalculated model) invalidate the cached windows
    monkeypatch.setattr(model, "_group_cre_support", {}, raising=False)
    monkeypatch.setattr(model.archetypes, "centroids", model.archetypes.centroids[::-1].copy())
    builder.group_creature_support(model, fmt, min_games=0)
    assert json.loads((tmp_path / "creature_support_groups.json").read_text(encoding="utf-8"))["stamp"] != cache["stamp"]


# ---- (b) switch off = the shipped builder -----------------------------------------------------------------------------------

def test_switch_off_is_the_shipped_search(model, name2id, held, monkeypatch):
    assert builder.ARCH_WINDOW is False
    for h in held[:3]:
        pool = _pool(h, name2id)
        a = builder.build(model, pool, "bo3", n_decks=1, families=False, n_jobs=1)
        ctx = builder.Ctx(model, "bo3")
        prep = builder.prepare(ctx, pool, None, None)
        assert prep.arch_win == {} and prep.cfg_wins == {}
        r1 = builder.run_search(ctx, prep, 1)[0]
        ctx2 = builder.Ctx(model, "bo3")
        r2 = builder._run_search(ctx2, builder.prepare(ctx2, pool, None, None), 1)[0]
        assert [(c.key, c.logit) for c in r1] == [(c.key, c.logit) for c in r2]
        assert a.decks[0].creature_window is None
        # the switch on without any group window changes nothing either
        with monkeypatch.context() as mp:
            mp.setattr(builder, "ARCH_WINDOW", True)
            mp.setattr(builder, "group_creature_support", lambda m, f, min_games=None: {})
            b = builder.build(model, pool, "bo3", n_decks=1, families=False, n_jobs=1)
        assert _deck(a.decks[0]) == _deck(b.decks[0])


# ---- (c) a lower group window is honoured ------------------------------------------------------------------------------------

@pytest.mark.parametrize("families", [False, True])
def test_group_window_below_the_global_one(arch_model, name2id, held, monkeypatch, families):
    model = arch_model
    fmt = "bo3"
    glob = builder.creature_support(model, fmt)
    assert glob is not None and glob[0] >= 3
    win = (max(glob[0] - 6, 0), glob[0] - 2)                  # every archetype group: a window entirely below the global one
    ca = model.card_arrays
    with monkeypatch.context() as mp:
        mp.setattr(builder, "ARCH_WINDOW", True)
        mp.setattr(builder, "group_creature_support", lambda m, f, min_games=None: {a: win for a in range(m.archetypes.n)})
        n_below = 0
        for h in held[:3]:
            pool = _pool(h, name2id)
            ctx = builder.Ctx(model, fmt)
            prep = builder.prepare(ctx, pool, None, None)
            res = builder.build(model, pool, fmt, n_decks=2 if families else 1, families=families, n_jobs=1)
            for ev in res.decks:
                deck = _deck(ev)
                n = _cre(model, deck)
                assert (ev.creature_window["lo"], ev.creature_window["hi"]) == win
                # inside the group's window as relaxed for this pool (a pool short of non-creatures forces more creatures)
                cfg = builder._config_of_deck(ctx, deck, prep.configs, prep.pool)
                lo, hi = builder.relaxed_creature_bounds(win, 40 - builder.nl_of(ctx, deck), *builder._cre_avail(ctx, prep, cfg))
                assert lo <= n <= hi, (n, win, lo, hi)
                assert builder.arch_ok(ctx, prep, cfg, deck)
                n_below += n < glob[0]
        assert n_below > 0                                     # decks below the global lo are reachable (and chosen)


def test_a_deck_invalid_for_its_group_is_dropped(arch_model, name2id, held, monkeypatch):
    model = arch_model
    pool = _pool(held[0], name2id)
    ctx = builder.Ctx(model, "bo3")
    prep = builder.prepare(ctx, pool, None, None)
    deck = _deck(builder.build(model, pool, "bo3", n_decks=1, families=False, n_jobs=1).decks[0])
    n = _cre(model, deck)
    cfg = builder._config_of_deck(ctx, deck, prep.configs, prep.pool)
    monkeypatch.setattr(builder, "ARCH_WINDOW", True)
    prep.arch_win = {a: (n + 2, n + 4) for a in range(model.archetypes.n)}       # every group wants more creatures than this deck has
    assert not builder.arch_ok(ctx, prep, cfg, deck)
    prep.arch_win = {a: (n, n) for a in range(model.archetypes.n)}
    assert builder.arch_ok(ctx, prep, cfg, deck)


# ---- (d) the variant is registered; the report arithmetic ------------------------------------------------------------------

def test_variant_registered_and_patches_the_switch():
    sc = _load_script("search_check")
    assert sc.VARIANTS["arch_window"] == dict(global_builder=dict(ARCH_WINDOW=True))
    assert "arch_window" in sc.ORDER
    assert builder.ARCH_WINDOW is False
    with sc.patched_search(sc.VARIANTS["arch_window"]):
        assert builder.ARCH_WINDOW is True
    assert builder.ARCH_WINDOW is False


def test_decision_lines_and_targeted_pool_helpers():
    sh = _load_script("split_half_search")

    def s(n, mean, lo, hi):
        return dict(n=n, mean_gap_out=mean, gap_out_ci=[lo, hi])

    ok = sh.archetype_window_decision(s(50, 0.0, -0.05, 0.05), {"bo1": s(25, 0.0, -0.1, 0.1), "bo3": s(25, 0.0, -0.1, 0.1)},
                                      s(440, 0.0, -0.01, 0.01))
    assert ok["verdict"] == "PASS" and not ok["measurable_gain"]
    few = sh.archetype_window_decision(s(39, 0.0, -0.05, 0.05), {"bo1": s(20, 0, -.1, .1), "bo3": s(19, 0, -.1, .1)}, s(440, 0, -.01, .01))
    assert few["verdict"] == "INCONCLUSIVE"
    bad = sh.archetype_window_decision(s(50, -0.2, -0.3, -0.01), {"bo1": s(25, -0.2, -0.3, -0.1), "bo3": s(25, 0, -.1, .1)}, s(440, 0, -.01, .01))
    assert bad["verdict"] == "FAIL" and bad["lines"]["2_C_not_worse"]["status"] == "FAIL"
    assert bad["lines"]["3_no_format_harmed"]["formats"]["bo1"]["status"] == "FAIL"
    # a format whose CI upper bound is < 0 but whose mean is above -0.05 is not "harmed"
    mild = sh.archetype_window_decision(s(50, -0.03, -0.08, -0.01), {"bo1": s(25, -0.03, -0.04, -0.01), "bo3": s(25, 0, -.1, .1)},
                                        s(440, 0, -.01, .01))
    assert mild["lines"]["3_no_format_harmed"]["status"] == "PASS" and mild["verdict"] == "PASS"
    rbad = sh.archetype_window_decision(s(50, 0.1, 0.01, 0.2), {"bo1": s(25, 0.1, 0.0, .2), "bo3": s(25, 0.1, 0, .2)}, s(440, -0.1, -0.12, -0.06))
    assert rbad["verdict"] == "FAIL" and rbad["measurable_gain"]

    pools = _load_script("archetype_window_pools")
    assert pools.is_affected((5, 15), (7, 17)) and pools.is_affected((7, 19), (7, 17)) and not pools.is_affected((6, 18), (7, 17))
    ids = [f"d{i}" for i in range(100)]
    a, b = pools.draw_pools(ids, 40, 7), pools.draw_pools(ids, 40, 7)
    assert a == b and len(set(a)) == 40 and pools.draw_pools(ids[:5], 40, 7) == sorted(ids[:5])


def test_load_pool_tasks_tolerates_a_missing_logit(calc, tmp_path, monkeypatch):
    import polars as pl

    from sealed_lab import config

    sc = _load_script("search_check")
    monkeypatch.setenv("SEALED_LAB_HOME", str(calc["home"]))          # (load_pool_tasks sets it per set; restored after)
    did = pl.read_parquet(config.corpus_dir("TST") / "pools.parquet")["draft_id"][0]
    dev = tmp_path / "dev"
    (dev / "TST").mkdir(parents=True)
    p = tmp_path / "pools.json"
    p.write_text(json.dumps(dict(pools=[dict(set="TST", fmt="bo3", draft_id=did, logit_default=None)])), encoding="utf-8")
    import shutil

    shutil.copytree(calc["home"], dev / "TST" / "home")                  # a copy, never a link
    tasks = sc.load_pool_tasks(dev, p)
    assert len(tasks) == 1 and tasks[0]["logit_default"] is None and tasks[0]["pool"]
