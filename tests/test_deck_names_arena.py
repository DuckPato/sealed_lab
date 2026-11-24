"""Name matching and Arena text parsing / export (pure unit tests, no model needed)."""
from __future__ import annotations

import polars as pl
import pytest

from sealed_lab.deck import arena
from sealed_lab.deck.names import NameIndex, normalize, resolve_names
from sealed_lab.types import CardLine, DeckEval

NAMES = ["Plains", "Island", "Swamp", "Mountain", "Forest", "Fire // Ice", "Bonecrusher Giant // Stomp",
         "Lim-Dûl's Vault", "Gideon's Reproach", "Jötun Grunt", "Okaun, Eye of Chaos", "Rat Colony",
         "Dryad Arbor", "Sol Ring", "Who // What // When // Where // Why"]


@pytest.fixture(scope="module")
def index() -> NameIndex:
    df = pl.DataFrame({
        "card_id": list(range(len(NAMES))), "name": NAMES,
        "front_name": [n.split(" // ")[0] for n in NAMES],
        "is_basic": [n in ("Plains", "Island", "Swamp", "Mountain", "Forest") for n in NAMES]})
    return NameIndex(df)


# ----------------------------------------------------------------------------------------- names

def test_normalize():
    assert normalize("Lim-Dûl’s Vault") == "lim duls vault"
    assert normalize("  Fire //  Ice ") == "fire ice"
    assert normalize("Jötun Grunt") == normalize("Jotun Grunt")


@pytest.mark.parametrize("text,expected,how", [
    ("Fire // Ice", "Fire // Ice", "exact"),
    ("Fire", "Fire // Ice", "front"),
    ("Bonecrusher Giant", "Bonecrusher Giant // Stomp", "front"),
    ("fire // ice", "Fire // Ice", "normalized"),
    ("FIRE", "Fire // Ice", "normalized"),
    ("Ice", "Fire // Ice", "face"),
    ("Lim-Dul's Vault", "Lim-Dûl's Vault", "normalized"),
    ("Lim-Dûl’s Vault", "Lim-Dûl's Vault", "normalized"),
    ("Jotun Grunt", "Jötun Grunt", "normalized"),
    ("Gideons Reproach", "Gideon's Reproach", "normalized"),
    ("A-Rat Colony", "Rat Colony", "normalized"),
    ("Okaun Eye of Chaos", "Okaun, Eye of Chaos", "normalized"),
    ("Who", "Who // What // When // Where // Why", "front"),
])
def test_matching(index, text, expected, how):
    m = index.match(text)
    assert m.ok and m.name == expected and m.how == how, m


def test_fuzzy_match_warns(index):
    m = index.match("Gideons Reproch")
    assert m.how == "fuzzy" and m.name == "Gideon's Reproach" and "assuming" in m.warning
    assert not index.match("Totally Unknown Card").ok
    assert not index.match("Fir").ok          # too short to be a safe fuzzy hit
    assert not index.match("").ok


def test_basics_always_resolve(index):
    for n in ("Plains", "plains", "Snow-Covered Island", "Swamp"):
        m = index.match(n)
        assert m.ok and m.is_basic, n
    assert index.match("Snow-Covered Island").name == "Island"


def test_resolve_reports_unmatched(index):
    r = resolve_names({"Fire": 1, "Nope Nope": 2, "Plains": 9, "Gideons Reproch": 1}, index)
    assert r.counts == {5: 1, 0: 9, 8: 1}
    assert r.unmatched == ["Nope Nope"] and r.n_unmatched_cards == 2 and len(r.warnings) == 1
    r2 = resolve_names({"Fire": 1, "Plains": 9}, index, drop_basics=True)
    assert r2.counts == {5: 1}


# ----------------------------------------------------------------------------------------- parsing

ARENA = """Deck
1 Fire // Ice (MH2) 290
2 Lim-Dûl's Vault (ELD) 12
4 Plains (ANA) 1
1 Okaun, Eye of Chaos

Sideboard
1 Sol Ring (C21) 263
"""


def test_parse_arena_basic():
    main, side = arena.parse_arena(ARENA)
    assert main == {"Fire // Ice": 1, "Lim-Dûl's Vault": 2, "Plains": 4, "Okaun, Eye of Chaos": 1}
    assert side == {"Sol Ring": 1}


def test_parse_headerless_blank_line_splits_sideboard():
    main, side = arena.parse_arena("2 Rat Colony\n1 Sol Ring\n\n1 Dryad Arbor\n")
    assert main == {"Rat Colony": 2, "Sol Ring": 1} and side == {"Dryad Arbor": 1}


def test_parse_deck_header_then_blank_stays_in_the_deck_until_a_sideboard_header():
    main, side = arena.parse_arena("Deck\n1 Rat Colony\n\n1 Sol Ring\n")
    assert main == {"Rat Colony": 1, "Sol Ring": 1} and not side
    main, side = arena.parse_arena("Deck\n1 Rat Colony\n\nSideboard\n1 Sol Ring\n")
    assert main == {"Rat Colony": 1} and side == {"Sol Ring": 1}


def test_parse_foil_markers_after_the_collector_number():
    main, _ = arena.parse_arena("Deck\n1 Rat Colony (RNA) 5 *F*\n2 Sol Ring (C21) 263 *E*\n1 Plains (RNA) *F*\n")
    assert main == {"Rat Colony": 1, "Sol Ring": 2, "Plains": 1}


@pytest.mark.parametrize("prefix,eol", [("﻿", "\r\n"), ("", "\r\n"), ("", "\r"), ("﻿", "\n")])
def test_parse_bom_crlf_trailing_spaces(prefix, eol):
    text = prefix + eol.join(["Deck ", "1 Fire // Ice (MH2) 290   ", "  2 Rat Colony\t", "", "Sideboard", "1 Sol Ring  ", ""])
    main, side = arena.parse_arena(text)
    assert main == {"Fire // Ice": 1, "Rat Colony": 2} and side == {"Sol Ring": 1}


def test_parse_about_companion_commander_and_junk():
    text = "About\nName My Sealed Deck\n\nDeck\n3 Rat Colony\n1x Sol Ring\n\nCompanion\n1 Dryad Arbor\n\nSideboard\n1 Plains\nthis is junk\n0 Nothing\n"
    p = arena.parse_arena_full(text)
    assert p.deck_name == "My Sealed Deck"
    assert p.main == {"Rat Colony": 3, "Sol Ring": 1}
    assert p.side == {"Dryad Arbor": 1, "Plains": 1}
    assert p.ignored == ["this is junk", "0 Nothing"]
    assert p.had_headers


def test_parse_name_starting_with_x_and_duplicates():
    main, _ = arena.parse_arena("1 X Marks the Spot\n1 Rat Colony\n1 Rat Colony (RNA) 5\n")
    assert main == {"X Marks the Spot": 1, "Rat Colony": 2}


def test_parse_empty_and_garbage_never_raise():
    assert arena.parse_arena("") == ({}, {})
    assert arena.parse_arena("\n\n \n") == ({}, {})
    p = arena.parse_arena_full("hello\nworld")
    assert p.main == {} and p.ignored == ["hello", "world"]


def test_pool_is_main_plus_side():
    main, side = arena.parse_arena(ARENA)
    pool = arena.pool_of(main, side)
    assert pool["Sol Ring"] == 1 and pool["Fire // Ice"] == 1 and sum(pool.values()) == 9


def test_clean_card_name():
    assert arena.clean_card_name("Fire // Ice (MH2) 290") == "Fire // Ice"
    assert arena.clean_card_name("Island (ANA) 3a") == "Island"
    assert arena.clean_card_name("Sol Ring (C21)") == "Sol Ring"
    assert arena.clean_card_name("Okaun, Eye of Chaos") == "Okaun, Eye of Chaos"


# ----------------------------------------------------------------------------------------- export

def _deck(cards: list[tuple[str, int]]) -> DeckEval:
    lines = [CardLine(card_id=i, name=n, count=c, value_pp=None, low_data=False) for i, (n, c) in enumerate(cards)]
    return DeckEval(cards=lines, colors="W", splash="", game_wr=0.5, game_wr_lo=0.45, game_wr_hi=0.55, match_wr=None,
                    score=50.0, score_lo=40.0, score_hi=60.0, exp_wins=3.0, p_max_wins=0.1, breakdown={}, mana={}, warnings=[])


def test_export_front_face_and_roundtrip():
    d = _deck([("Rat Colony", 2), ("Fire // Ice", 1), ("Bonecrusher Giant // Stomp", 1), ("Plains", 16)])
    txt = arena.export_arena(d, {"Sol Ring": 1, "Who // What // When // Where // Why": 1}, "TST")
    assert txt == ("Deck\n2 Rat Colony\n1 Fire\n1 Bonecrusher Giant\n16 Plains\n\nSideboard\n1 Sol Ring\n1 Who\n")
    main, side = arena.parse_arena(txt)
    assert main == {"Rat Colony": 2, "Fire": 1, "Bonecrusher Giant": 1, "Plains": 16} and side == {"Sol Ring": 1, "Who": 1}
    assert arena.export_arena(d).count("Sideboard") == 0


def test_export_names_reimport_through_index(index):
    d = _deck([("Fire // Ice", 1), ("Lim-Dûl's Vault", 2), ("Plains", 3)])
    main, _ = arena.parse_arena(arena.export_arena(d))
    r = resolve_names(main, index)
    assert r.counts == {5: 1, 7: 2, 0: 3} and not r.unmatched
