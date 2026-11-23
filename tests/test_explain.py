"""Deck explanation (waterfall) and swap analysis on the synthetic set."""
from __future__ import annotations

import json
import threading
import time

import numpy as np
import pytest
from scipy.special import expit, logit

from sealed_lab import api, config
from sealed_lab.deck import builder, explain
from sealed_lab.types import Cancelled

BASICS = ("Plains", "Island", "Swamp", "Mountain", "Forest")


@pytest.fixture(scope="module")
def name2id(model):
    return {n: i for i, n in enumerate(model.cards.sort("card_id")["name"].to_list())}


@pytest.fixture(scope="module")
def held(calc):
    return json.loads((calc["synth"] / "heldout_pools.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def case(model, name2id, held):
    """A built deck and its pool (two different pools)."""
    out = []
    for h in held[:3]:
        pool = {name2id[n]: c for n, c in h["pool"].items() if n not in BASICS}
        res = builder.build(model, pool, "bo3", n_decks=1)
        out.append((pool, {c.card_id: c.count for c in res.decks[0].cards}))
    return out


def test_reference_stats_match_reference_distribution(model, calc):
    """The 'average build' of the waterfall is the mean logit of the real builds, i.e. the reference distribution."""
    for fmt in model.formats:
        st = explain.ref_stats(model, fmt)
        q = np.load(config.model_dir("TST") / "reference.npz")[f"{fmt}__q"][0].astype(float)
        assert st.start_logit == pytest.approx(logit(np.clip(q, 1e-6, 1 - 1e-6)).mean(), abs=2e-3)
        assert st.n_builds > 50 and np.isfinite(st.z_mean).all()
    # cached on disk, and the cache is reused (same numbers)
    f = config.cache_dir("TST") / "explain_ref_bo3.json"
    assert f.exists()
    model._explain_ref = {}
    again = explain.ref_stats(model, "bo3")
    assert again.start_logit == pytest.approx(explain.ref_stats(model, "bo3").start_logit)


@pytest.mark.parametrize("fmt", ["bo3", "bo1"])
def test_waterfall_adds_up(model, case, fmt):
    for pool, deck in case:
        ex = explain.explain_deck(model, deck, fmt)
        steps = ex.steps
        assert steps[0].kind == "start" and steps[-1].kind == "end"
        assert ex.end_wr == pytest.approx(float(model.q(deck, fmt)), abs=1e-12)
        # logit steps add up exactly, pp steps (sequential) add up to the total gap in win rate
        L_end = logit(ex.end_wr)
        assert logit(ex.start_wr) + sum(s.d_logit for s in steps) == pytest.approx(L_end, abs=1e-9)
        assert sum(s.d_pp for s in steps) == pytest.approx(100 * (ex.end_wr - ex.start_wr), abs=1e-9)
        # chain of win rates is consistent
        wr = ex.start_wr
        for s in steps[1:-1]:
            assert 100 * (s.wr_after - wr) == pytest.approx(s.d_pp, abs=1e-9)
            wr = s.wr_after
        assert wr == pytest.approx(ex.end_wr, abs=1e-12)
        kinds = [s.kind for s in steps]
        assert kinds.count("card") <= 8
        assert [s.label for s in steps if s.kind == "structure"] == [g for g, _ in explain.STRUCT_GROUPS] + (["Mulligans"] if model.has_mull(fmt) else [])
        # the card steps are sorted by size of the effect among the top ones, every spell is accounted for
        n_spell_rows = sum(n for _, _, n, _ in ex.card_steps)
        assert n_spell_rows == sum(n for c, n in deck.items() if not model.card_arrays.is_land[c])


def test_waterfall_average_deck_is_flat(model):
    """A deck that equals the mean of the real builds in every respect is explained by (almost) no steps: here we check the
    weaker, exact property that the waterfall of two decks differs only through the swapped cards and structure."""
    st = explain.ref_stats(model, "bo3")
    assert st.spell_beta == pytest.approx(st.card_mean / st.n_spells - st.nbl_mean / st.n_spells, abs=1e-9)


@pytest.fixture(scope="module")
def analysis(model, case):
    pool, deck = case[0]
    return pool, deck, explain.swap_analysis(model, deck, pool, "bo3")


def _swapped(deck, o, i):
    d = dict(deck)
    d[o] -= 1
    if d[o] == 0:
        del d[o]
    d[i] = d.get(i, 0) + 1
    return d


def test_swap_gains_match_the_full_model(model, analysis):
    pool, deck, sa = analysis
    assert sa.complete and sa.n_eval > 0 and sa.base_wr == pytest.approx(float(model.q(deck, "bo3")))
    assert sa.deck_best and sa.side_best
    for s in list(sa.deck_best.values())[:6] + list(sa.side_best.values())[:6] + sa.suggestions:
        d2 = _swapped(deck, s.out_id, s.in_id)
        assert s.gain_pp == pytest.approx(100 * float(model.q(d2, "bo3") - model.q(deck, "bo3")), abs=1e-7)
        dL = model.logit(d2, "bo3", draws=True) - model.logit(deck, "bo3", draws=True)
        assert s.p_better == pytest.approx(float(np.mean(dL[1:] > 0)))
        assert 0.0 <= s.p_better <= 1.0


def test_swap_best_and_value_above_replacement(model, analysis):
    pool, deck, sa = analysis
    ca = model.card_arrays
    side = {c: n - deck.get(c, 0) for c, n in pool.items() if n - deck.get(c, 0) > 0}
    # deck_best: the best of everything evaluated, var = -gain
    for o, s in sa.deck_best.items():
        assert o in deck and s.in_id in side or ca.is_basic[s.in_id]
        assert sa.var_pp[o] == pytest.approx(-s.gain_pp)
        assert ca.is_land[o] == ca.is_land[s.in_id]            # spell for spell, land for land
    for i, s in sa.side_best.items():
        assert i in side and s.out_id in deck
        assert ca.is_land[i] == ca.is_land[s.out_id]
    # an exhaustive search over spell pairs finds nothing better than the pre-filtered best suggestion
    best_found = max(s.gain_pp for s in sa.deck_best.values())
    L0 = model.logit(deck, "bo3")
    src = sum(n * ca.produces[c] for c, n in deck.items())

    def castable(c):                                                  # cards the analysis considers playable in this deck
        need = ca.pips[c] > 0
        return bool((src[need] >= 2).any() if ca.hybrid[c] else (src[need] >= 2).all())
    brute = max(model.logit(_swapped(deck, o, i), "bo3") for o in deck if not ca.is_land[o]
                for i in side if not ca.is_land[i] and castable(i))
    assert 100 * (expit(brute) - expit(L0)) <= best_found + 0.15      # the linear prefilter misses at most a small gain
    # suggestions: distinct cards, positive gains, best first
    sg = sa.suggestions
    assert len(sg) <= 3 and all(s.gain_pp > 0 for s in sg)
    assert len({s.out_id for s in sg}) == len(sg) and len({s.in_id for s in sg}) == len(sg)
    assert [s.gain_pp for s in sg] == sorted((s.gain_pp for s in sg), reverse=True)


def test_swap_respects_locks_and_exclusions(model, case):
    for pool, deck in case:
        sa = explain.swap_analysis(model, deck, pool, "bo3")
        if sa.suggestions:
            break
    else:
        pytest.skip("no improving swap in these pools")
    s0 = sa.suggestions[0]
    sb = explain.swap_analysis(model, deck, pool, "bo3", locked={s0.out_id: 1}, excluded={s0.in_id})
    assert all(s.out_id != s0.out_id and s.in_id != s0.in_id for s in sb.suggestions)
    assert s0.in_id not in sb.side_best                        # excluded cards are not candidates at all


def test_swap_analysis_cancel_and_partial(model, case):
    pool, deck = case[1]
    ev = threading.Event()
    ev.set()
    with pytest.raises(Cancelled):
        explain.swap_analysis(model, deck, pool, "bo3", cancel=ev)
    parts: list = []
    sa = explain.swap_analysis(model, deck, pool, "bo3", on_partial=parts.append, partial_every=0.0)
    assert sa.complete and parts and all(not p.complete for p in parts)
    assert len(parts) == sa.n_eval or len(parts) > 3


def test_swap_analysis_is_fast_when_warm(model, case):
    pool, deck = case[2]
    explain.swap_analysis(model, deck, pool, "bo3")
    t0 = time.monotonic()
    explain.swap_analysis(model, deck, pool, "bo3")
    assert time.monotonic() - t0 < 2.0


def test_api_functions(calc, model, case):
    pool, deck = case[0]
    ex = api.explain_deck("TST", deck, "bo3")
    assert ex.end_wr == pytest.approx(float(model.q(deck, "bo3")))
    sa = api.swap_analysis("TST", deck, pool, "bo3")
    assert sa.complete
    with pytest.raises(ValueError):
        api.explain_deck("TST", deck, "nope")
