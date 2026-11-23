"""Sample hands (sealed_lab.deck.samplehand): deterministic with a seeded Random; Bo1 smoother shared with the mana simulator."""
from __future__ import annotations

import random

import numpy as np

from sealed_lab.deck import samplehand as S
from sealed_lab.model import manasim

INFO = {
    1: dict(name="Plains", is_land=True, produces="W", mana_value=0),
    2: dict(name="Island", is_land=True, produces="U", mana_value=0),
    3: dict(name="Dual", is_land=True, produces="WU", mana_value=0),
    10: dict(name="One W", mana_value=1, cost="W"),
    11: dict(name="Two WW", mana_value=2, cost="WW"),
    12: dict(name="Three 1UU", mana_value=3, cost="1UU"),
    13: dict(name="Five", mana_value=5, cost="3WU"),
}
DECK = {1: 6, 2: 5, 3: 2, 10: 8, 11: 7, 12: 7, 13: 5}          # 13 lands, 27 spells = 40


def cards():
    return S.library_from_deck(DECK, INFO)


def test_library_has_every_card_and_lands_flagged():
    c = cards()
    assert len(c) == 40 and sum(x.is_land for x in c) == 13
    assert {x.name for x in c if x.is_land} == {"Plains", "Island", "Dual"}


def test_deterministic_with_a_fixed_seed():
    a = S.Deal(cards(), random.Random(5), "bo1")
    b = S.Deal(cards(), random.Random(5), "bo1")
    assert [c.cid for c in a.hand] == [c.cid for c in b.hand] and [c.cid for c in a.library] == [c.cid for c in b.library]
    assert len(a.hand) == 7 and len(a.library) == 33


def _land_freq(fmt, n=20000, seed=1):
    rng = random.Random(seed)
    cs = cards()
    f = np.zeros(8)
    for _ in range(n):
        hand, lib = S.shuffle_deal(cs, rng, fmt)
        assert len(hand) == 7 and len(lib) == 33
        f[sum(c.is_land for c in hand)] += 1
    return f / n


def test_bo1_land_counts_follow_the_simulators_smoother():
    pmf = manasim.hand_land_pmf(13, 40, "sealed")
    f = _land_freq("bo1")
    assert np.abs(f - pmf).max() < 0.012
    plain = manasim.hand_land_pmf(13, 40, "random")
    assert np.abs(pmf - plain).max() > 0.02                      # the smoother really differs from a plain shuffle here


def test_bo3_is_a_plain_shuffle():
    plain = manasim.hand_land_pmf(13, 40, "random")
    assert np.abs(_land_freq("bo3") - plain).max() < 0.012


def test_bo1_uses_the_simulators_own_function(monkeypatch):
    seen = []
    real = S.hand_land_pmf

    def spy(L, n, regime, *a):
        seen.append((L, n, regime))
        return real(L, n, regime, *a)
    monkeypatch.setattr(S, "hand_land_pmf", spy)
    S.shuffle_deal(cards(), random.Random(1), "bo1")
    assert seen == [(13, 40, "sealed")]


def test_london_mulligan_bottoms_exactly_n_cards():
    d = S.Deal(cards(), random.Random(3), "bo3")
    assert d.mulligan() and d.mulligans == 1 and len(d.hand) == 7 and d.to_bottom == 1 and not d.can_draw()
    assert not d.bottom([0, 1]) and not d.bottom([9]) and len(d.hand) == 7           # wrong counts change nothing
    gone = d.hand[2]
    assert d.bottom([2]) and len(d.hand) == 6 and len(d.library) == 34 and d.library[-1] is gone
    assert d.can_mulligan() and d.mulligan() and d.to_bottom == 2
    a, b = d.hand[0], d.hand[5]
    assert d.bottom([5, 0]) and len(d.hand) == 5 and len(d.library) == 35
    assert {id(d.library[-1]), id(d.library[-2])} == {id(a), id(b)}
    assert sorted(c.cid for c in d.hand + d.library) == sorted(c.cid for c in cards())


def test_draw_and_turns():
    play = S.Deal(cards(), random.Random(2), "bo3", on_play=True)
    assert play.draw() is None and play.turn == 1 and len(play.hand) == 7           # no draw on turn 1 on the play
    top = play.library[0]
    assert play.draw() is top and play.turn == 2 and len(play.hand) == 8
    on_draw = S.Deal(cards(), random.Random(2), "bo3", on_play=False)
    assert on_draw.draw() is not None and on_draw.turn == 1 and len(on_draw.hand) == 8
    while on_draw.can_draw():
        on_draw.draw()
    assert on_draw.turn == S.MAX_TURN and on_draw.draw() is None


def test_castable_by_lands_and_colours():
    lib = {c.name: c for c in cards()}
    hand = [lib["Plains"], lib["Plains"], lib["Island"], lib["One W"], lib["Two WW"], lib["Three 1UU"], lib["Five"]]
    n, src, ok = S.castable(hand)
    assert n == 3 and src["W"] == 2 and src["U"] == 1
    assert {c.name for c in ok} == {"One W", "Two WW"}            # 1UU needs two blue sources, Five needs 5 lands
    hand2 = [lib["Plains"], lib["Dual"], lib["Island"], lib["Three 1UU"]]            # U sources: Dual + Island (the dual counts)
    assert [c.name for c in S.castable(hand2)[2]] == ["Three 1UU"]


def test_castable_each_land_pays_one_pip():
    five = S.SCard(90, "Five-colour land", True, produces="WUBRG")
    dual = S.SCard(91, "WB dual", True, produces="WB")
    wb = S.SCard(92, "Two WB", mv=2, cost="WB")
    wwb = S.SCard(93, "Three WWB", mv=3, cost="WWB")
    assert S.castable([five, wb])[2] == []                       # one land cannot pay W and B
    assert S.castable([five, dual, wb])[2] == [wb]
    assert S.castable([five, dual, S.SCard(1, "Forest", True, produces="G"), wwb])[2] == []   # 3 lands, only 2 can pay W/B
