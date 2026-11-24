"""Audit 1124 fixes in the mana model: adventure / omen cost groups, multi-type land searches, 0-land decks, London-mulligan
bottoming, source parsing (Vivi, Plaza of Heroes, chosen-colour lands), the splash string."""
from __future__ import annotations

import numpy as np
import pytest

from sealed_lab.deck.samplehand import SCard
from sealed_lab.model import castability as cs
from sealed_lab.model import manasim, structure
from sealed_lab.model import mana_sources as ms
from tests.test_model_castability import make_cards
from tests.test_model_manasim import DUAL, ISLAND, PLAINS, W2, make_cards as make_ca


def _groups(cost, layout, faces, name="X"):
    import polars as pl

    sc = [dict(name=name, mana_cost=cost, layout=layout, card_faces=[dict(mana_cost=f) for f in faces], cmc=0)]
    return cs.cost_groups_column(pl.DataFrame({"name": [name]}), sc)[0]


def test_adventure_uses_main_face():
    assert _groups("{4}{G} // {1}{B}", "adventure", ["{4}{G}", "{1}{B}"]) == "G"
    assert _groups("{3}{W} // {G/W}", "prepare", ["{3}{W}", "{G/W}"]) == "W"


def test_split_uses_cheaper_face_and_plain_cards_unchanged():
    assert _groups("{1}{R} // {2}{U}{U}", "split", ["{1}{R}", "{2}{U}{U}"]) == "R"
    assert _groups("{2}{W}{W}", "normal", []) == "W W"
    assert _groups("{3}{W}{W}", "adventure", ["", "{3}{W}{W}"]) == "W W"          # top-level without "//"
    assert _groups("", "transform", ["{2}{W}", ""]) == "W"


def _fetch(text):
    d = ms.describe_card("x", "Sorcery", text, "", False, False, 2)
    return d.fetch


def test_multi_type_land_searches():
    assert _fetch("Search your library for a basic Plains, Swamp, or Forest card, reveal it, put it into your hand.") == "WBG"
    assert _fetch("Search your library for a Plains, Island, Swamp, or Mountain card, put it onto the battlefield tapped, then shuffle.") == "WUBR"
    assert _fetch("Search your library for a basic land card, put it onto the battlefield tapped, then shuffle.") == "basic"


def test_source_parsing_vivi_plaza_chosen():
    d = ms.describe_card("Vivi", "Legendary Creature - Wizard",
                         "{0}: Add X mana in any combination of {U} and/or {R}, where X is Vivi's power. Activate only during your turn.", "UR", False, False, 2)
    assert d.mask == ms.BIT["U"] | ms.BIT["R"]
    plaza = ("{T}: Add {C}.\n{T}: Add one mana of any color. Spend this mana only to cast a legendary spell.\n"
             "{T}: Add one mana of any color among legendary permanents you control.")
    d = ms.describe_card("Plaza of Heroes", "Land", plaza, "WUBRG", True, False, 0)
    assert d.mask == 0 and d.restr == ("legendary",)                       # colourless + legendary-only mana, not any colour
    d = ms.describe_card("Room", "Land", "This land enters tapped. As it enters, choose a color.\n{T}: Add one mana of the chosen color.", "WUBRG", True, False, 0)
    assert d.mask == ms.ALL and d.rule == 1                                # the player picks the colour the deck needs


def test_zero_land_deck():
    df = make_cards()
    cc = cs.CastCards.from_df(df)
    ids = {n: i for i, n in enumerate(df["name"].to_list())}
    deck = {ids["W2"]: 20, ids["G2"]: 10, ids["Filler1"]: 10}
    r = cs.analyze_deck(deck, cc)
    i = list(r.ids).index(ids["W2"])
    assert r.p[i] == 0.0 and r.p_curve[i] == 0.0 and r.dead == pytest.approx(40.0)
    ca = make_ca([W2] * 40)
    for fn in (manasim.simulate_v1, manasim.simulate_v2):
        out = fn({0: 40}, ca, 200, extra=True, detail=True)
        assert out["p_screw"] == 1.0 and out["mana_spent"] == 0.0 and out["mull_rate"] == 1.0


def _plain(dc):
    """The class table without the bottoming hints (`mv`, `prod`): the old class-order rule."""
    from types import SimpleNamespace

    return SimpleNamespace(counts=dc.counts, pos_class=dc.pos_class, is_land=dc.is_land, Kl=dc.Kl)


ALWAYS_MULL = np.ones((2, 2, 8))


def test_mulligan_bottoms_dearest_spell():
    ca = make_ca([PLAINS, ISLAND, dict(mv=6, pips="W")] + [dict(mv=1, pips="W")])
    dc = manasim.deck_classes({0: 8, 1: 8, 2: 6, 3: 18}, ca, v2=True)
    n, L = 40, 16
    deal = manasim.deal_v2(dc, n, L, n - L, 1500, "random", ALWAYS_MULL)
    Kc = len(dc.counts)
    dear = [k for k in range(Kc) if not dc.is_land[k] and dc.mv[k] == 6][0]
    assert (deal["m"] == 2).all() and ((deal["hand"] == Kc).sum(1) == 2).all()
    # a random 5 of 7 would hold 0.75 six-drops on average; bottoming the dearest spells first leaves far fewer
    assert (deal["hand"] == dear).sum(1).mean() < 0.6
    again = manasim.deal_v2(dc, n, L, n - L, 1500, "random", ALWAYS_MULL)
    assert np.array_equal(deal["hand"], again["hand"])


def test_mulligan_keeps_duals_when_bottoming_a_land():
    ca = make_ca([PLAINS, ISLAND, DUAL] + [dict(mv=2, pips="W")])
    dc = manasim.deck_classes({0: 7, 1: 7, 2: 4, 3: 22}, ca, v2=True)
    n, L = 40, 18
    new = manasim.deal_v2(dc, n, L, n - L, 3000, "random", ALWAYS_MULL)["hand"]
    old = manasim.deal_v2(_plain(dc), n, L, n - L, 3000, "random", ALWAYS_MULL)["hand"]
    dual = [k for k in range(len(dc.counts)) if dc.is_land[k] and dc.prod[k].sum() == 2][0]
    assert (new == dual).sum() > (old == dual).sum()


def test_splash_string_ignores_hybrid_only_colours():
    # W/R main; one {W/U} hybrid card must not make U a splash
    ca = make_ca([PLAINS, dict(land=True, basic=True, produces="R"), dict(mv=2, pips="W"), dict(mv=2, pips="R"),
                  dict(mv=2, pips="WU", hybrid=True), dict(mv=3, pips="B")])
    main, splash, ncards, npips, needs = structure._color_analysis(np.array([2, 3, 4]), np.array([8.0, 8.0, 4.0]), ca)
    assert "".join(c for c, f in zip("WUBRG", splash) if f) == "" and ncards == 0.0
    main, splash, ncards, npips, needs = structure._color_analysis(np.array([2, 3, 4, 5]), np.array([8.0, 8.0, 4.0, 1.0]), ca)
    assert "".join(c for c, f in zip("WUBRG", splash) if f) == "B" and ncards == 1.0


def test_samplehand_adventure_cost_main_face_only():
    assert SCard(1, "x", cost="{4}{G} // {1}{B}").pips() == {"G": 1}
