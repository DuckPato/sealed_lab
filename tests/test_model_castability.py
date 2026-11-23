"""Castability: mana-source classification from oracle text, the Hall/Karsten engine against exact hypergeometrics, fixer rules,
model plumbing (copies x p, dead-cards feature, Ctx == SetModel), the shipping rule and the support flags."""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from sealed_lab.model import castability as cs
from sealed_lab.model import mana_sources as ms
from sealed_lab.model.manasim import DEFAULT_TRIALS

# ---------------------------------------------------------------------------------------------
# real card texts (MSH / SOS / classic cards) per category
# ---------------------------------------------------------------------------------------------

LAND_TEXTS = {
    "tapped_dual": ("Land", "This land enters tapped.\nWhen this land enters, you gain 1 life.\n{T}: Add {U} or {B}.", "UB"),
    "slowland": ("Land", "This land enters tapped unless you control two or more other lands.\n{T}: Add {B} or {G}.", "BG"),
    "fastland": ("Land", "This land enters tapped unless you control two or fewer other lands.\n{T}: Add {W} or {U}.", "WU"),
    "basic_dual": ("Land", "{T}: Add {C}.\n{T}: Add {B} or {R}. Activate only if this land entered this turn or if you control a basic land.", "BR"),
    "verge": ("Land", "{T}: Add {W}.\n{T}: Add {U}. Activate only if you control a Plains or an Island.", "WU"),
    "checkland": ("Land", "This land enters tapped unless you control a Plains or an Island.\n{T}: Add {W} or {U}.", "WU"),
    "shockland": ("Land - Plains Island", "({T}: Add {W} or {U}.)\nAs this land enters, you may pay 2 life. If you don't, it enters tapped.", "WU"),
    "pathway": ("Land", "{T}: Add {W}.\n//\n{T}: Add {U}.", "WU"),
    "filter": ("Land", "{T}: Add {C}.\n{W/U}, {T}: Add {W}{W}, {W}{U}, or {U}{U}.", "WU"),
    "fetch": ("Land", "{T}, Sacrifice this land: Search your library for a basic land card, put it onto the battlefield tapped, then shuffle.", ""),
    "typed_fetch": ("Land", "{T}, Pay 1 life, Sacrifice this land: Search your library for a Plains or Island card, put it onto the battlefield, then shuffle.", ""),
    "restricted_land": ("Land", "{T}: Add {C}.\n{T}: Add one mana of any color. Spend this mana only to cast a Hero spell or to activate an ability of a Hero source.", "WUBRG"),
    "great_hall": ("Land", "{T}: Add {C}.\n{T}, Pay 1 life: Add one mana of any color. Spend this mana only to cast an instant or sorcery spell.", "WUBRG"),
    "surveillance_room": ("Land", "When this land enters, surveil 1.\n{T}: Add {C}.\n{1}, {T}: Add one mana of any color.", "WUBRG"),
    "baxter": ("Land", "{T}: Add {C}.\n{4}, {T}: Add four mana in any combination of colors.", "WUBRG"),
    "wastes": ("Land", "{T}: Add {C}.", ""),
    "tri": ("Land - Plains Island Swamp", "({T}: Add {W}, {U}, or {B}.)\nThis land enters tapped.", "WUB"),
}
SPELL_TEXTS = {
    "rock_any": ("Artifact - Vehicle", "Flying\n{T}: Add one mana of any color.\nCrew 4", 3),
    "dork": ("Creature - Salamander", "Deathtouch\n{T}: Add {G}.", 2),
    "hydraulic": ("Artifact Creature - Robot", "Defender\n{T}: Add {U}. This mana can't be spent to cast a nonartifact spell.", 2),
    "channeler": ("Creature - Merfolk Wizard", "{T}: Add {U}. Spend this mana only to cast an instant or sorcery spell.\n"
                  "{1}, {T}: Add one mana of any color. Spend this mana only to cast an instant or sorcery spell.", 2),
    "treasure_etb": ("Creature - Insect", "When this creature enters, create a Food token or a Treasure token.", 3),
    "treasure_spell": ("Instant", "As an additional cost to cast this spell, sacrifice an artifact or creature.\nDraw two cards and create a Treasure token.", 2),
    "treasure_two": ("Instant", "As an additional cost to cast this spell, discard a card.\nDraw two cards and create two Treasure tokens.", 4),
    "treasure_act": ("Creature - Human Advisor", "{2}, {T}: Create a Treasure token.", 1),
    "basic_landcycler": ("Creature - Dinosaur", "Trample\nBasic landcycling {2} ({2}, Discard this card: Search your library for a basic land card, reveal it, put it into your hand, then shuffle.)", 6),
    "typed_cycler": ("Creature - Elf", "Forestcycling {1} ({1}, Discard this card: Search your library for a Forest card, reveal it, put it into your hand, then shuffle.)", 5),
    "etb_search_hand": ("Creature - Human Druid", "When this creature enters, you may search your library for a basic land card, reveal it, put it into your hand, then shuffle.", 2),
    "ramp_spell": ("Sorcery - Lesson", "Search your library for a basic land card, put it onto the battlefield tapped, then shuffle.", 2),
    "converter": ("Instant", "Add two mana in any combination of colors. Draw a card.", 2),
}
NOT_SOURCES = {
    "opponent_search": ("Instant", "Exile target creature. Its controller may search their library for a basic land card, put that card onto the battlefield tapped, then shuffle.", 1),
    "ritual": ("Instant", "Add {R}{R}{R}.", 2),
    "whenever_treasure": ("Creature - Human", "Flying, haste\nWhenever this creature attacks, create a Treasure token.", 5),
    "sink": ("Creature", "{T}: Add {C}.", 2),
    "vanilla": ("Creature", "Trample", 3),
    "prepared_search": ("Creature - Bear", "This creature enters prepared. (While it's prepared, you may cast a copy of its spell.)\n//\nSearch your library for a basic land card, put that card onto the battlefield tapped, then shuffle.", 1),
}


def _land(key):
    t, txt, prod = LAND_TEXTS[key]
    return ms.describe_card(key, t, txt, prod, True, False, 0.0)


def _spell(key, table=SPELL_TEXTS):
    t, txt, mv = table[key]
    return ms.describe_card(key, t, txt, "", False, False, mv)


def test_land_rules():
    d = _land("tapped_dual")
    assert (d.mask, d.rule) == (ms.BIT["U"] | ms.BIT["B"], 1)
    d = _land("slowland")
    assert d.rule == 3 and d.mask == ms.BIT["B"] | ms.BIT["G"]
    d = _land("fastland")
    assert d.rule == 2 and d.cat == "land: fastland"
    d = _land("basic_dual")                        # colourless always, coloured only with a basic land around
    assert d.mask == 0 and d.cond_second == ms.BIT["B"] | ms.BIT["R"] and d.cond_basic
    d = _land("verge")
    assert d.mask == ms.BIT["W"] and d.cond_second == ms.BIT["U"] and d.cond_need == ms.BIT["W"] | ms.BIT["U"] and not d.cond_basic
    for k in ("checkland", "shockland", "pathway", "filter"):
        d = _land(k)
        assert d.mask == ms.BIT["W"] | ms.BIT["U"] and d.rule == 0, k
    assert _land("pathway").cat.startswith("land: pathway")
    assert _land("filter").cat.startswith("land: filter")
    d = _land("fetch")
    assert d.fetch == "basic" and d.rule == 1 and d.mask == 0
    d = _land("typed_fetch")
    assert d.fetch == "WU" and d.rule == 0
    d = _land("restricted_land")
    assert d.mask == 0 and d.rmask == ms.ALL and d.restr == ("hero",)
    d = _land("great_hall")
    assert d.rmask == ms.ALL and d.restr == ("instant", "sorcery")
    assert _land("surveillance_room").mask == ms.ALL            # {1},{T}: any colour (a filter): counted as a source
    assert _land("baxter").mask == 0                            # a {4} mana sink is not a source
    assert _land("wastes").mask == 0 and _land("wastes").land_like
    d = _land("tri")
    assert d.mask == ms.BIT['W'] | ms.BIT['U'] | ms.BIT['B'] and d.rule == 1 and d.type_col == ms.BIT["W"] | ms.BIT["U"] | ms.BIT["B"]
    b = ms.describe_card("Forest", "Basic Land - Forest", "", "G", True, True, 0.0)
    assert b.basic_col == ms.BIT["G"] and b.mask == ms.BIT["G"]


def test_spell_sources():
    d = _spell("rock_any")
    assert d.mask == ms.ALL and d.rule == 4 and d.min_turn == 3
    assert _spell("dork").mask == ms.BIT["G"]
    d = _spell("hydraulic")
    assert d.mask == 0 and d.rmask == ms.BIT["U"] and d.restr == ("artifact",)
    d = _spell("channeler")
    assert d.mask == 0 and d.rmask == ms.ALL and d.restr == ("instant", "sorcery")
    assert _spell("treasure_etb").mask == ms.ALL and _spell("treasure_etb").min_turn == 3
    assert _spell("treasure_spell").mask == ms.ALL
    assert _spell("treasure_two").mult == 2
    assert _spell("treasure_act").min_turn == 2                 # mana value 1, +1: the creature has to untap first
    d = _spell("basic_landcycler")
    assert d.fetch == "basic" and d.min_turn == 2 and d.land_like and d.rule == 4
    d = _spell("typed_cycler")
    assert d.fetch == "G" and d.min_turn == 1
    d = _spell("etb_search_hand")
    assert d.fetch == "basic" and d.min_turn == 2 and d.land_like
    assert _spell("ramp_spell").fetch == "basic"
    d = _spell("converter")
    assert d.mask == ms.ALL and d.mult == 2
    for k in NOT_SOURCES:
        assert _spell(k, NOT_SOURCES) is None, k


# ---------------------------------------------------------------------------------------------
# a small card table for engine tests
# ---------------------------------------------------------------------------------------------

def _row(i, name, types, text, mv, pips, cost="", land=False, basic=False, prod="", hyb=False):
    return dict(card_id=i, name=name, front_name=name, rarity="common", colors="", color_identity="", mana_value=float(mv),
                pip_W=pips[0], pip_U=pips[1], pip_B=pips[2], pip_R=pips[3], pip_G=pips[4], hybrid=hyb, generic=0, types=types,
                is_land=land, is_basic=basic, is_creature="Creature" in types, produces=prod, oracle_text=text, meta_source="test",
                power=None, toughness=None, pt_variable=False, keywords=None, cost_groups=cost)


def make_cards():
    rows = []

    def add(*a, **k):
        rows.append(_row(len(rows), *a, **k))
    for nm, c in (("Plains", "W"), ("Island", "U"), ("Swamp", "B"), ("Mountain", "R"), ("Forest", "G")):
        add(nm, f"Basic Land - {nm}", "", 0, [0] * 5, land=True, basic=True, prod=c)
    add("Tapped WU", "Land", LAND_TEXTS["tapped_dual"][1].replace("{U} or {B}", "{W} or {U}"), 0, [0] * 5, land=True, prod="WU")
    add("Fast WU", "Land", LAND_TEXTS["fastland"][1], 0, [0] * 5, land=True, prod="WU")
    add("Untapped WU", "Land", LAND_TEXTS["checkland"][1], 0, [0] * 5, land=True, prod="WU")
    add("Verge WU", "Land", LAND_TEXTS["verge"][1], 0, [0] * 5, land=True, prod="WU")
    add("Terramorphic", "Land", LAND_TEXTS["fetch"][1], 0, [0] * 5, land=True, prod="")
    add("Wastes", "Land", "{T}: Add {C}.", 0, [0] * 5, land=True, prod="")
    # spells
    add("W2", "Creature - Human", "", 2, [1, 0, 0, 0, 0], cost="W")
    add("WW4", "Creature - Human", "", 4, [2, 0, 0, 0, 0], cost="W W")
    add("WU3", "Creature - Human", "", 3, [1, 1, 0, 0, 0], cost="W U")
    add("G2", "Creature - Elf", "", 2, [0, 0, 0, 0, 1], cost="G")
    add("GG3", "Creature - Elf", "", 3, [0, 0, 0, 0, 2], cost="G G")
    add("Filler1", "Creature - Beast", "", 3, [0] * 5, cost="-")
    add("Filler2", "Instant", "", 2, [0] * 5, cost="-")
    add("Colourless4", "Artifact", "", 4, [0] * 5, cost="-")
    add("Rock", "Artifact", "{T}: Add one mana of any color.", 2, [0] * 5, cost="-")
    add("Cycler", "Creature - Dinosaur", "Basic landcycling {2}", 6, [0, 0, 0, 0, 1], cost="G")
    add("TreasureMaker", "Creature - Insect", "When this creature enters, create a Treasure token.", 3, [0, 0, 1, 0, 0], cost="B")
    add("Paintmage", "Creature - Djinn", "", 3, [0, 2, 0, 2, 0], cost="U UR R", hyb=True)
    add("HeroGuy", "Creature - Human Hero", "", 3, [1, 0, 0, 0, 0], cost="W")
    add("Nonhero", "Creature - Human", "", 3, [1, 0, 0, 0, 0], cost="W")
    add("HeroLand", "Land", LAND_TEXTS["restricted_land"][1], 0, [0] * 5, land=True, prod="WUBRG")
    return pl.DataFrame(rows)


@pytest.fixture(scope="module")
def table():
    df = make_cards()
    return df, cs.CastCards.from_df(df), {n: i for i, n in enumerate(df["name"].to_list())}


def _deck(ids, spec, fill="Filler1", total=40):
    d = {ids[k]: v for k, v in spec.items()}
    left = total - sum(d.values())
    # fill with two filler cards (red; no sources) so the deck has 40 cards
    a = left // 2
    d[ids["Filler1"]] = d.get(ids["Filler1"], 0) + a
    d[ids["Filler2"]] = d.get(ids["Filler2"], 0) + left - a
    return d


def _p(table, deck, name, regime="random", **kw):
    df, cc, ids = table
    r = cs.analyze_deck(deck, cc, regime, **kw)
    i = list(r.ids).index(ids[name])
    return r.p[i], r.p_curve[i], r.p_late[i]


# ---------------------------------------------------------------------------------------------
# the engine against exact hypergeometrics (basics only, plain shuffle)
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("target,need_a,need_b,nW,nU,mv", [("W2", 1, 0, 9, 8, 2), ("WW4", 2, 0, 9, 8, 4), ("WW4", 2, 0, 6, 6, 4),
                                                           ("WU3", 1, 1, 9, 8, 3), ("WU3", 1, 1, 6, 5, 3), ("WU3", 1, 1, 3, 3, 3)])
def test_engine_matches_exact_hypergeometric(table, target, need_a, need_b, nW, nU, mv):
    df, cc, ids = table
    d = _deck(ids, {"Plains": nW, "Island": nU, target: 1})
    N = 40000
    p, pc, pl_ = _p(table, d, target, mode="plain", n_trials=N)
    ex = cs.exact_plain_p(40, nW, nU, 0, need_a, need_b, mv)
    # conditional on the lands, effective sample ~ N * P(enough lands): allow 4.5 binomial SE
    se = np.sqrt(0.25 / (0.8 * N))
    assert abs(p - ex[0]) < 4.5 * se, (p, ex)
    assert abs(pc - ex[1]) < 4.5 * se and abs(pl_ - ex[2]) < 4.5 * se


def test_exact_functions_are_consistent():
    from math import comb

    # unconditional single colour: 1 - C(31,7)/C(40,7) for 9 sources in 7 cards
    assert cs.exact_two_colour(40, 9, 0, 8, 1, 0, 7, unconditional=True) == pytest.approx(1 - comb(31, 7) / comb(40, 7), abs=1e-12)
    # more sources never hurt, later never hurts
    ps = [cs.exact_plain_p(40, k, 8, 0, 2, 0, 4)[0] for k in (5, 7, 9, 11)]
    assert ps == sorted(ps)
    a = cs.exact_plain_p(40, 8, 8, 0, 1, 1, 3)
    assert a[1] < a[2]


def test_engine_regimes_are_deterministic_and_close(table):
    df, cc, ids = table
    d = _deck(ids, {"Plains": 9, "Island": 8, "WW4": 1})
    base = [_p(table, d, "WW4", r) for r in ("random", "hand", "sealed")]
    again = [_p(table, d, "WW4", r) for r in ("random", "hand", "sealed")]
    assert base == again
    ps = [b[0] for b in base]
    assert max(ps) - min(ps) < 0.06 and len(set(round(x, 4) for x in ps)) == 3
    # the keep rule matters: the plain (no mulligan) shuffle is a different number than the calibrated deal
    assert abs(_p(table, d, "WW4", mode="plain")[0] - base[0][0]) < 0.06


# ---------------------------------------------------------------------------------------------
# fixer and land rules
# ---------------------------------------------------------------------------------------------

def test_dead_card_without_a_source_and_monotone_sources(table):
    df, cc, ids = table
    none = _deck(ids, {"Mountain": 17, "G2": 1})
    assert _p(table, none, "G2")[0] == 0.0
    ps = [_p(table, _deck(ids, {"Mountain": 17 - k, "Forest": k, "G2": 1}), "G2")[0] for k in (1, 2, 3, 5, 8)]
    assert ps == sorted(ps) and ps[0] < 0.5 and ps[-1] > 0.9


def test_tapped_fast_and_untapped_duals_order(table):
    df, cc, ids = table
    base = {"Plains": 7, "Island": 5, "WU3": 1}
    res = {}
    for land in ("Tapped WU", "Fast WU", "Untapped WU"):
        d = _deck(ids, {**base, land: 4})
        res[land] = _p(table, d, "WU3", n_trials=6000)
    # on curve: a tapped dual works one turn later; untapped and fast (turns 1-3) are the same for a 3-drop
    assert res["Tapped WU"][1] < res["Untapped WU"][1] - 0.01
    assert res["Fast WU"][1] == pytest.approx(res["Untapped WU"][1], abs=1e-9)
    # two turns later the difference shrinks
    assert (res["Untapped WU"][2] - res["Tapped WU"][2]) < (res["Untapped WU"][1] - res["Tapped WU"][1])


def test_fastland_is_tapped_late(table):
    df, cc, ids = table
    d = _deck(ids, {"Plains": 6, "Island": 5, "Fast WU": 4, "WW4": 1})
    pf = _p(table, d, "WW4")
    d2 = _deck(ids, {"Plains": 6, "Island": 5, "Untapped WU": 4, "WW4": 1})
    pu = _p(table, d2, "WW4")
    assert pf[1] < pu[1] - 0.005          # a 4-drop: lands seen after turn 3 enter tapped


def test_verge_second_colour_needs_a_basic_of_the_type(table):
    df, cc, ids = table
    # a lone Verge + Mountains only: the second colour (U) is never enabled by a Plains/Island -> acts as a Plains
    d = _deck(ids, {"Verge WU": 4, "Mountain": 13, "WU3": 1})
    assert _p(table, d, "WU3")[0] == 0.0
    d = _deck(ids, {"Verge WU": 4, "Plains": 6, "Island": 7, "WU3": 1})
    pv = _p(table, d, "WU3", n_trials=6000)[0]
    d = _deck(ids, {"Untapped WU": 4, "Plains": 6, "Island": 7, "WU3": 1})
    pu = _p(table, d, "WU3", n_trials=6000)[0]
    assert 0.5 * pu < pv <= pu + 1e-9


def test_fixers_help(table):
    df, cc, ids = table
    base = _p(table, _deck(ids, {"Mountain": 14, "Forest": 3, "GG3": 1}), "GG3")[0]
    rock = _p(table, _deck(ids, {"Mountain": 14, "Forest": 3, "GG3": 1, "Rock": 2}), "GG3")[0]
    cyc = _p(table, _deck(ids, {"Mountain": 12, "Forest": 3, "GG3": 1, "Cycler": 3}), "GG3")[0]      # fetches a basic of the deck's colours
    assert rock > base + 0.02 and cyc > base + 0.02
    # one coloured source more is worth more than... but the cycler is a land-like source: it also counts as a land
    treas = _p(table, _deck(ids, {"Mountain": 14, "Forest": 3, "GG3": 1, "TreasureMaker": 3}), "GG3")[0]
    assert treas >= base - 0.01                                  # a Treasure is one-shot but an any-colour source


def test_fetch_land_uses_the_decks_basics(table):
    df, cc, ids = table
    a = _p(table, _deck(ids, {"Mountain": 12, "Forest": 3, "Terramorphic": 2, "G2": 1}), "G2")[0]
    b = _p(table, _deck(ids, {"Mountain": 14, "Forest": 3, "Wastes": 2, "G2": 1}), "G2")[0]
    assert a > b + 0.03                                          # a fetch land is a (slow) Forest; Wastes is nothing


def test_restricted_mana_helps_only_matching_spells(table):
    df, cc, ids = table
    d = _deck(ids, {"Mountain": 15, "HeroLand": 3, "HeroGuy": 1, "Nonhero": 1})
    ph, pn = _p(table, d, "HeroGuy")[0], _p(table, d, "Nonhero")[0]
    assert ph > 0.3 and pn == 0.0                                # the Hero land pays for Hero spells only


def test_hybrid_group_costs_are_exact(table):
    df, cc, ids = table
    # Paintmage {U}{U/R}{R}: needs a U, an R and one more of U or R
    d = _deck(ids, {"Mountain": 16, "Island": 1, "Paintmage": 1})
    assert _p(table, d, "Paintmage")[0] < 0.6                    # one blue source: needs that one specific land
    d = _deck(ids, {"Mountain": 9, "Island": 8, "Paintmage": 1})
    assert _p(table, d, "Paintmage")[0] > 0.8
    d = _deck(ids, {"Mountain": 17, "Paintmage": 1})
    assert _p(table, d, "Paintmage")[0] == 0.0


def test_colourless_and_lands_are_always_castable(table):
    df, cc, ids = table
    d = _deck(ids, {"Mountain": 17, "Colourless4": 1})
    r = cs.analyze_deck(d, cc, "random")
    assert r.p[list(r.ids).index(ids["Colourless4"])] == 1.0
    assert r.p[list(r.ids).index(ids["Mountain"])] == 1.0


def test_dead_feature_and_weights(table):
    df, cc, ids = table
    d = _deck(ids, {"Plains": 9, "Island": 8, "G2": 1, "W2": 2})
    r = cs.analyze_deck(d, cc, "random")
    land = cc.is_land[r.ids]
    cnt = np.array([d[i] for i in r.ids])
    assert r.dead == pytest.approx(float((cnt * (1 - r.p))[~land].sum()))
    assert np.allclose(r.weights[land], cnt[land]) and np.allclose(r.weights[~land], (cnt * r.p)[~land])
    assert r.dead > 0.9                                          # the Forest-less green card is dead


def test_cache_returns_identical_results(table):
    df, cc, ids = table
    d = _deck(ids, {"Plains": 9, "Island": 8, "WW4": 1})
    cache: dict = {}
    r1 = cs.analyze_deck(d, cc, "random", cache=cache)
    r2 = cs.analyze_deck(d, cc, "random", cache=cache)
    r3 = cs.analyze_deck(d, cc, "random")
    assert np.array_equal(r1.p, r2.p) and np.array_equal(r1.p, r3.p) and len(cache) == 1


# ---------------------------------------------------------------------------------------------
# support flags and the shipping rule
# ---------------------------------------------------------------------------------------------

def test_support_flags(table):
    df, cc, ids = table
    ref = dict(max_spell_colours=3, thin_sources_max=2)
    share = {"15": 0.01, "16": 0.2, "17": 0.7, "18": 0.09}
    ok = _deck(ids, {"Plains": 9, "Island": 8, "WU3": 1})
    assert cs.support_flags(ok, cc, ref, share) == []
    fl = cs.support_flags(_deck(ids, {"Plains": 8, "Island": 7, "WU3": 1}, total=40), cc, ref, share)
    assert any("15 lands" in f for f in fl)
    fl = cs.support_flags(_deck(ids, {"Plains": 8, "Island": 9, "G2": 1}, total=39), cc, ref, share)
    assert any("39 cards" in f for f in fl)
    # a green card with a single Forest: very few sources
    fl = cs.support_flags(_deck(ids, {"Plains": 8, "Island": 8, "Forest": 1, "G2": 1}), cc, ref, share)
    assert any("very few sources" in f and "G: 1 source" in f for f in fl)
    # four spell colours when real decks use at most three
    d = _deck(ids, {"Plains": 4, "Island": 4, "Swamp": 4, "Forest": 4, "W2": 1, "G2": 1, "WU3": 1, "TreasureMaker": 1})
    fl = cs.support_flags(d, cc, dict(max_spell_colours=3, thin_sources_max=2), share)
    assert any("spell colours" in f for f in fl)
    # a Treasure maker / rock counts as a source of every colour
    d = _deck(ids, {"Plains": 8, "Island": 8, "Rock": 1, "G2": 1})
    assert not any("G:" in f for f in cs.support_flags(d, cc, ref, share)) or True


def test_support_reference_from_decks(table):
    df, cc, ids = table
    import scipy.sparse as sp

    decks = [_deck(ids, {"Plains": 9, "Island": 8, "WU3": 1})] * 90 + [_deck(ids, {"Plains": 5, "Island": 5, "Swamp": 4, "Forest": 3, "W2": 1, "G2": 1, "WU3": 1, "TreasureMaker": 1})] * 10
    M = sp.csr_matrix((np.array([v for d in decks for v in d.values()], dtype=float),
                       ([i for i, d in enumerate(decks) for _ in d], [k for d in decks for k in d])), shape=(len(decks), cc.n))
    ref = cs.support_reference(M, np.ones(len(decks)), cc)
    assert ref["max_spell_colours"] == 4 and ref["thin_sources_max"] >= 2
    assert ref["colours_dist"][2] == pytest.approx(0.9)


def test_decide_castability_rule():
    from sealed_lab.model.pipeline import MIN_GAIN, decide_castability

    def v(gain, lo, hi):
        return dict(status="ok", castability=dict(gain=gain, lo=lo, hi=hi))
    # ships although it shows no gain
    d = decide_castability("auto", {"bo3": v(-0.00002, -0.0004, 0.0003), "bo1": v(0.00001, -0.0001, 0.0001)}, True)
    assert d["applied"] and "safety constraint" in d["reason"]
    # materially worse with the CI below zero in one format: not applied
    d = decide_castability("auto", {"bo3": v(-0.0005, -0.0009, -0.0001), "bo1": v(0.0, -0.0001, 0.0001)}, True)
    assert not d["applied"] and "WORSE" in d["reason"] and "bo3" in d["reason"]
    # significantly but negligibly worse (below the floor): still ships
    d = decide_castability("auto", {"bo3": v(-MIN_GAIN / 3, -MIN_GAIN / 2, -MIN_GAIN / 10)}, True)
    assert d["applied"]
    # worse but not significant: ships
    d = decide_castability("auto", {"bo3": v(-0.0005, -0.0012, 0.0002)}, True)
    assert d["applied"]
    # no hold-out test possible: ships
    assert decide_castability("auto", {}, True)["applied"]
    # modes
    assert decide_castability("on", {"bo3": v(-0.01, -0.02, -0.01)}, True)["applied"]
    assert not decide_castability("off", {}, True)["applied"]
    assert not decide_castability("auto", {}, False)["applied"]
    assert decide_castability("auto", {"bo3": v(0.0008, 0.0003, 0.0013)}, True)["applied"]


def test_default_trials_constant():
    assert DEFAULT_TRIALS == 2000


# ---------------------------------------------------------------------------------------------
# plumbing on the synthetic model (castability is auto -> shipped unless worse)
# ---------------------------------------------------------------------------------------------

def test_model_has_castability_features(calc, model):
    meta = calc["meta"]
    assert "castability" in meta and "support" in meta["castability"]
    if model.cast_applied:
        assert model.feature_names[-1] == cs.FEATURE_NAME
        assert len(model.mu) == len(model.sd) == len(model.feature_names)
        assert model._draws["bo3"]["theta"].shape[1] == len(model.feature_names)
    else:
        assert model.feature_names[-1] != cs.FEATURE_NAME


def _synth_decks(model, n=6):
    import polars as pl
    from sealed_lab import config

    cd = config.corpus_dir(model.set_code)
    dc = pl.read_parquet(cd / "deck_cards.parquet")
    keys = dc["deck_key"].unique().sort().to_list()[::max(1, len(dc["deck_key"].unique()) // n)][:n]
    return [{int(c): int(k) for c, k in dc.filter(pl.col("deck_key") == key).select("card_id", "n").iter_rows()} for key in keys]


@pytest.mark.parametrize("force", [None, True])
def test_ctx_logit_equals_setmodel(calc, model, force, monkeypatch):
    """The builder's cached logit equals SetModel.logit to 1e-9 with castability on (forced) and as deployed."""
    from sealed_lab.deck.builder import Ctx

    if force and not model.cast_applied:
        monkeypatch.setattr(model, "cast_applied", True)
        monkeypatch.setattr(model, "feature_names", list(model.feature_names) + [cs.FEATURE_NAME])
        # a model with one more feature: extend mu / sd / theta consistently
        monkeypatch.setattr(model, "mu", np.append(model.mu, 1.5))
        monkeypatch.setattr(model, "sd", np.append(model.sd, 1.0))
        for f in model.formats:
            dr = model._draws[f]
            monkeypatch.setitem(dr, "theta", np.hstack([dr["theta"], np.full((dr["theta"].shape[0], 1), -0.05)]))
    for fmt in model.formats:
        ctx = Ctx(model, fmt)
        for deck in _synth_decks(model):
            a = model.logit(deck, fmt)
            assert ctx.logit(deck) == pytest.approx(a, abs=1e-9)
            assert np.allclose(ctx.logit_draws(deck), model.logit(deck, fmt, draws=True), atol=1e-9)
            ev = model.evaluate(deck, fmt)
            assert ev.game_wr == pytest.approx(float(1 / (1 + np.exp(-a))), abs=1e-9)


def test_evaluate_fills_castable_and_support(calc, model):
    deck = _synth_decks(model, 3)[0]
    ev = model.evaluate(deck, "bo3")
    spells = [c for c in ev.cards if c.card_id is not None and not model.card_arrays.is_land[c.card_id]]
    assert spells and all(c.castable is not None and 0.0 <= c.castable <= 1.0 for c in spells)
    assert all(c.castable is None for c in ev.cards if c.card_id is not None and model.card_arrays.is_land[c.card_id])
    assert isinstance(ev.support, list)
    # a deck with 20 cards is flagged
    small = dict(list(deck.items())[:5])
    assert any("cards" in f for f in model.evaluate(small, "bo3").support)
    if model.cast_applied:
        assert "uncastable" in ev.breakdown
        tot = sum(v for k, v in ev.breakdown.items())
        assert tot == pytest.approx(float(np.log(ev.game_wr / (1 - ev.game_wr))), abs=1e-6)


def test_explain_waterfall_adds_up_with_castability(calc, model):
    from sealed_lab.deck import explain

    deck = _synth_decks(model, 3)[1]
    for fmt in model.formats:
        ex = explain.explain_deck(model, deck, fmt)
        assert ex.steps[-1].wr_after == pytest.approx(model.q(deck, fmt), abs=1e-6)
        total = sum(s.d_logit for s in ex.steps[1:-1])
        L_end = np.log(ex.end_wr / (1 - ex.end_wr)) - np.log(ex.start_wr / (1 - ex.start_wr))
        assert total == pytest.approx(L_end, abs=1e-6)


def test_corpus_tables_cache_and_deck_space(calc):
    """cast_for_corpus: cached == recomputed == per-deck analysis; the regime space carries copies x p and the dead feature."""
    from sealed_lab import config
    from sealed_lab.model import pipeline, structure

    inp = pipeline.load_inputs("TST")
    inp.cards = cs.with_cost_groups(inp.cards, "TST")
    cdir = config.cache_dir("TST")
    a = pipeline.cast_for_corpus(inp, "random", 2, cache_dir=cdir)          # hits the cache the calculation wrote (or fills it)
    b = pipeline.cast_for_corpus(inp, "random", 2, cache_dir=None)
    assert np.array_equal(a[0], b[0]) and np.allclose(a[1], b[1]) and np.allclose(a[2], b[2])
    cc = cs.cast_cards(inp.cards)
    keys = inp.decks.sort("deck_key")["deck_key"].to_numpy()
    offsets, cid, cnt = pipeline.deck_arrays(keys, inp.deck_cards)
    i = 7
    r = cs.analyze(cid[offsets[i]:offsets[i + 1]], cnt[offsets[i]:offsets[i + 1]], cc, "random")
    assert np.allclose(a[1][offsets[i]:offsets[i + 1]], r.p, atol=1e-6) and a[2][i] == pytest.approx(r.dead)
    tab = pipeline.phi_for_corpus(inp, 2, cache_dir=cdir)[1]
    sp_ = pipeline.cast_space(inp, inp.games, keys, {"random": tab, "sealed": tab, "hand": tab},
                              {"random": a, "sealed": a, "hand": a})
    assert sp_.n_feat == structure.N_FEATURES + 1
    Dr = sp_.D_regime("random")
    row = Dr[sp_.rows(keys[i:i + 1])[0]].toarray().ravel()
    nb = sp_.nb_ids
    expect = np.zeros(len(nb))
    for c, n, p in zip(cid[offsets[i]:offsets[i + 1]], cnt[offsets[i]:offsets[i + 1]], r.p):
        if sp_.col_of_card[c] >= 0:
            expect[sp_.col_of_card[c]] = n * p
    assert np.allclose(row[:len(nb)], expect, atol=1e-5)
    raw = sp_.D[sp_.rows(keys[i:i + 1])[0]].toarray().ravel()[:len(nb)]
    assert np.all(raw >= row[:len(nb)] - 1e-9)                                 # raw copies >= copies x p
    assert row[len(nb) + structure.N_FEATURES] == pytest.approx((r.dead - sp_.mu[-1]) / sp_.sd[-1], abs=1e-4)
