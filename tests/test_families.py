"""Deck families: per-draw contexts, clustering, family statistics, display rule, end-to-end on the synthetic set."""
from __future__ import annotations

import json
import threading

import numpy as np
import pytest
from scipy.special import expit

from sealed_lab import api
from sealed_lab.deck import builder, families
from sealed_lab.types import Cancelled

BASICS = ("Plains", "Island", "Swamp", "Mountain", "Forest")


@pytest.fixture(scope="module")
def name2id(model):
    return {n: i for i, n in enumerate(model.cards.sort("card_id")["name"].to_list())}


@pytest.fixture(scope="module")
def held(calc):
    return json.loads((calc["synth"] / "heldout_pools.json").read_text(encoding="utf-8"))


def pool_ids(h, name2id):
    return {name2id[n]: c for n, c in h["pool"].items() if n not in BASICS}


@pytest.fixture(autouse=True, scope="module")
def _stop_pool():
    yield
    families.shutdown_pool(kill=True)


# ------------------------------------------------------------------------------------------ draws

def test_ctx_draws_equal_model_draws(model, name2id, held):
    for fmt in model.formats:
        base = builder.Ctx(model, fmt)
        B = model.n_draws(fmt)
        assert B > 1
        for h in held[:3]:
            d = {name2id[n]: c for n, c in h["player_deck"].items()}
            ref = np.asarray(model.logit(d, fmt, draws=True))
            assert base.logit_draws(d) == pytest.approx(ref, abs=1e-9)
            for b in (0, 1, B - 1):
                assert base.for_draw(b).logit(d) == pytest.approx(ref[b], abs=1e-9)
    with pytest.raises(ValueError):
        builder.Ctx(model, "bo3", draw=model.n_draws("bo3"))


# ------------------------------------------------------------------------------------------ clustering

def _spells(*groups, n=60):
    S = np.zeros((sum(g[0] for g in groups), n))
    r = 0
    for cnt, cards in groups:
        for _ in range(cnt):
            S[r, cards] = 1
            r += 1
    return S


def test_pairwise_diff_and_clusters():
    a = list(range(0, 23))
    a2 = list(range(1, 24))              # 22 shared of 23 -> 1 card apart
    b = list(range(10, 33))              # 13 shared -> 10 apart
    S = _spells((3, a), (2, a2), (3, b))
    D = families.pairwise_diff(S)
    assert D[0, 0] == 0 and D[0, 3] == 1 and D[0, 5] == 10 and (D == D.T).all()
    lab = families.cluster_decks(S)
    assert len(set(lab)) == 2 and len(set(lab[:5])) == 1 and len(set(lab[5:])) == 1 and lab[0] != lab[5]
    assert families.cluster_decks(S[:1]).tolist() == [0]
    # colors: two decks 6 cards apart merge, but not when their main colors differ by a color swap (+4)
    c = list(range(6, 29))
    S2 = _spells((1, a), (1, c))
    assert len(set(families.cluster_decks(S2))) == 1
    cols = np.array([[1, 1, 0, 0, 0], [1, 0, 1, 0, 0]], dtype=bool)
    assert len(set(families.cluster_decks(S2, colors=cols))) == 2


# ------------------------------------------------------------------------------------------ family statistics

def _logits(rows):
    return np.log(np.array(rows) / (1 - np.array(rows)))


def test_family_stats_pbest_gap_and_rule():
    # 4 candidates, 2 families: {0,1} and {2}, 3 draws (0 = full fit, 1..2 = replicates)
    L = _logits([[0.60, 0.58, 0.60],      # fam A, best at draw 0
                 [0.59, 0.60, 0.55],      # fam A, best at replicate 1 ... family best at rep 1 is this one
                 [0.595, 0.57, 0.62],     # fam B: wins replicate 2
                 [0.50, 0.50, 0.50]])
    lab = np.array([0, 0, 1, 2])
    fams = families.family_stats(L, lab)
    assert [f.rep for f in fams] == [0, 2, 3]
    assert [f.p_best for f in fams] == [0.5, 0.5, 0.0]            # replicates only: rep 1 -> A (0.60), rep 2 -> B (0.62)
    assert fams[0].gap_pp == 0.0 and fams[1].gap_pp == pytest.approx(0.5, abs=1e-6)
    assert fams[1].gap_lo_pp == fams[1].gap_hi_pp == fams[1].gap_pp          # 2 replicates: no interval
    assert fams[1].p_rep_beats_top == pytest.approx(0.5)           # rep of B vs rep of A: rep1 0.57<0.58, rep2 0.62>0.60
    assert fams[0].best_decks == [1, 0]
    shown = families.select_shown(fams)
    assert shown == [0, 1]                                         # B: P(best) 50%; the weak family is hidden
    assert fams[0].why.startswith("best") and "50%" in fams[1].why


def test_select_shown_rules_and_cap():
    def fam(p=0.0, gap=5.0, lo=2.0, hi=8.0):
        return families.Family(members=[0], rep=0, p_best=p, gap_pp=gap, gap_lo_pp=lo, gap_hi_pp=hi)
    fs = [fam(1.0, 0, 0, 0), fam(p=0.0625), fam(gap=0.8), fam(gap=3.0, lo=-0.5, hi=5.0), fam(), fam(p=0.5, gap=0.0, lo=-1, hi=1)]
    assert families.select_shown(fs) == [0, 1, 2, 3, 5]            # the clearly worse family is dropped; cap of 5 applies
    assert families.select_shown(fs, max_families=2) == [0, 1]
    assert families.select_shown([fam(1.0, 0, 0, 0), fam()]) == [0]   # one clear deck -> one family
    assert "interval" in fs[3].why and "within" in fs[2].why


def test_core_flex_split():
    names = [f"c{i}" for i in range(12)]
    basic = np.zeros(12, dtype=bool)
    rep = {0: 1, 1: 2, 2: 1, 3: 1}
    decks = [dict(rep) for _ in range(6)] + [{0: 1, 1: 2, 2: 1, 4: 1}] * 2 + [{0: 1, 1: 1, 3: 1, 4: 1}] * 2
    cf = families.core_flex(rep, decks, names, basic)
    assert cf["n_decks"] == 10
    assert {c["card_id"]: c["count"] for c in cf["core"]} == {0: 1, 1: 1}      # card 1: first copy in every deck, second in 8/10
    flex = {x["card_id"]: x for x in cf["flex"]}
    assert set(flex) == {1, 2, 3}
    assert flex[1]["count"] == 1 and flex[1]["core_count"] == 1 and flex[1]["freq"] == pytest.approx(0.8)
    assert flex[2]["freq"] == pytest.approx(0.8) and flex[3]["freq"] == pytest.approx(0.8)
    assert [a["card_id"] for a in flex[2]["alts"]] == [4] and flex[2]["alts"][0]["freq"] == pytest.approx(0.4)
    assert cf["extra"] == [dict(card_id=4, name="c4", freq=pytest.approx(0.4))]
    # basics are ignored
    basic[0] = True
    assert 0 not in {c["card_id"] for c in families.core_flex(rep, decks, names, basic)["core"]}


# ------------------------------------------------------------------------------------------ end to end

def test_build_families_serial(model, held, name2id):
    pool = pool_ids(held[2], name2id)
    log = []
    res = builder.build(model, pool, "bo3", n_decks=5, n_jobs=1, progress=lambda f, m: log.append(f))
    fam, meta = res.families, res.family_meta
    assert 1 <= len(res.decks) == len(fam) == len(res.p_best) == len(res.diffs) <= 5
    assert meta["n_draws"] == model.n_draws("bo3") - 1 and meta["n_candidates"] >= len(fam)
    assert log == sorted(log) and log[-1] == 1.0
    wr = [e.game_wr for e in res.decks]
    assert wr == sorted(wr, reverse=True) and res.diffs[0] == 0 and fam[0]["gap_pp"] == 0
    assert all(0 <= p <= 1 for p in res.p_best) and sum(res.p_best) <= 1 + 1e-9
    for k, f in enumerate(fam):
        assert f["rank"] == k and f["n_members"] >= 1 and f["why"]
        assert f["gap_lo_pp"] <= f["gap_hi_pp"] and f["gap_pp"] >= -1e-9
        core_n = sum(c["count"] for c in f["core"])
        flex_n = sum(c["count"] for c in f["flex"])
        deck = {c.card_id: c.count for c in res.decks[k].cards if c.card_id is not None}
        non_basic = sum(n for i, n in deck.items() if not model.card_arrays.is_basic[i])
        assert core_n + flex_n == non_basic
        for x in f["flex"]:
            assert 0 <= x["freq"] < families.CORE_FREQ and deck[x["card_id"]] >= x["count"]
    # family #1 is at least as good as the classic best deck found by the same draw-0 search
    classic = builder.build(model, pool, "bo3", n_decks=2, families=False)
    assert res.decks[0].game_wr >= classic.decks[0].game_wr - 1e-12
    # classic selection is unchanged and has no families
    assert classic.families == [] and classic.family_meta == {}


def test_build_families_locks_excludes_and_start(model, held, name2id):
    pool = pool_ids(held[1], name2id)
    base = builder.build(model, pool, "bo3", n_decks=1, families=False)
    deck = {c.card_id: c.count for c in base.decks[0].cards if c.card_id is not None}
    spells = sorted((i for i in deck if not model.card_arrays.is_land[i]), key=lambda i: -model.card_beta("bo3")[i])
    lock, ex = spells[-1], spells[0]
    res = builder.build(model, pool, "bo3", n_decks=5, n_jobs=1, locked={lock: 1}, excluded={ex})
    assert res.decks
    for e in res.decks:
        d = {c.card_id: c.count for c in e.cards}
        assert d.get(lock, 0) >= 1 and d.get(ex, 0) == 0
    for f in res.families:
        assert ex not in {c["card_id"] for c in f["core"]} | {x["card_id"] for x in f["flex"]}
    assert any("locked" in w for w in res.warnings)
    # a start deck that respects nothing about the constraints is dropped with a warning, the build still works
    res2 = builder.build(model, pool, "bo3", n_decks=2, n_jobs=1, excluded={ex}, start_decks=[deck])
    assert any("start decks" in w for w in res2.warnings)


def test_parallel_equals_serial_and_cancel(calc, model, held, name2id):
    m = builder.load_model("TST")                     # loaded from disk: the worker processes can load the same model
    pool = pool_ids(held[3], name2id)
    ser = builder.build(m, pool, "bo3", n_decks=5, n_jobs=1)
    par = builder.build(m, pool, "bo3", n_decks=5, n_jobs=2)
    assert par.family_meta["jobs"] == 2 and ser.family_meta["jobs"] == 1
    assert [e.game_wr for e in par.decks] == pytest.approx([e.game_wr for e in ser.decks], abs=1e-12)
    assert par.p_best == pytest.approx(ser.p_best) and par.diffs == ser.diffs
    again = builder.build(m, pool, "bo3", n_decks=5, n_jobs=2)           # second build reuses the persistent pool
    assert again.p_best == par.p_best
    ev = threading.Event()
    ev.set()
    with pytest.raises(Cancelled):
        builder.build(m, pool, "bo3", n_decks=5, n_jobs=1, cancel=ev)
    ev2 = threading.Event()
    seen = []

    def prog(f, msg):
        seen.append(f)
        if f > 0.2:
            ev2.set()
    with pytest.raises(Cancelled):
        builder.build(m, pool, "bo3", n_decks=5, n_jobs=2, cancel=ev2, progress=prog)
    ok = builder.build(m, pool, "bo3", n_decks=5, n_jobs=2)              # the pool is still usable afterwards
    assert ok.p_best == par.p_best


def test_fallback_without_draws_and_api(calc, model, held, name2id, monkeypatch):
    monkeypatch.setenv("SEALED_LAB_FAMILY_JOBS", "2")
    pool = pool_ids(held[0], name2id)
    monkeypatch.setattr(type(model), "n_draws", lambda self, fmt: 1)
    res = builder.build(model, pool, "bo3", n_decks=3, min_diff=10)
    assert res.families == [] and all(d >= 10 for d in res.diffs[1:])
    monkeypatch.undo()
    monkeypatch.setenv("SEALED_LAB_FAMILY_JOBS", "2")
    text = "Deck\n" + "".join(f"{c} {n}\n" for n, c in held[0]["pool"].items())
    one = api.build_pool("TST", text, "bo3", n_decks=1)
    assert len(one.decks) == 1 and one.families == []
    fam = api.build_pool("TST", text, "bo3", n_decks=5, min_diff=10)
    assert fam.families and fam.families[0]["rank"] == 0 and fam.pool
    fb = api.build_from_pool("TST", fam.pool, "bo3", n_decks=1)
    assert fb.families and len(fb.decks) >= 1          # the rebuild always uses the family search (even for n_decks=1)
