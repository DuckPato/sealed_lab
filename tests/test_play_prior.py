"""scripts/play_prior_check.py: play-rate counting, the 30-run minimum, Sealed deck choice and the raw Draft first-game pool read."""
from __future__ import annotations

import csv
import gzip
import importlib.util
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))
_SPEC = importlib.util.spec_from_file_location("play_prior_check", _SCRIPTS / "play_prior_check.py")
pp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pp)

from sealed_lab.model import pickprior  # noqa: E402


def _cards(rows):
    """rows: (card_id, name, colors, is_land, is_basic, rarity)."""
    return pl.DataFrame(dict(card_id=[r[0] for r in rows], name=[r[1] for r in rows], front_name=[r[1].split(" // ")[0] for r in rows],
                             colors=[r[2] for r in rows], is_land=[r[3] for r in rows], is_basic=[r[4] for r in rows],
                             rarity=[r[5] for r in rows]))


CARDS = _cards([
    (0, "Red Bolt", "R", False, False, "common"),
    (1, "Blue Drake", "U", False, False, "common"),
    (2, "Gold Dragon", "RU", False, False, "rare"),
    (3, "Grey Golem", "", False, False, "uncommon"),
    (4, "Mountain", "", True, True, "common"),
    (5, "Fire // Ice", "RU", False, False, "uncommon"),
])


def _long(rows):
    """rows: (run, main_colors, card_name, pool_n, deck_n)."""
    return pl.DataFrame(dict(draft_id=[r[0] for r in rows], main_colors=[r[1] for r in rows], card_name=[r[2] for r in rows],
                             pool_n=[r[3] for r in rows], deck_n=[r[4] for r in rows]))


def test_play_counts_eligibility_colorless_splash_and_z():
    long = _long([
        ("r1", "RU", "Red Bolt", 1, 1), ("r1", "RU", "Gold Dragon", 1, 1), ("r1", "RU", "Grey Golem", 1, 0),
        ("r2", "RU", "Red Bolt", 2, 1), ("r2", "RU", "Gold Dragon", 1, 0), ("r2", "RU", "Grey Golem", 1, 1),
        ("r3", "UG", "Red Bolt", 1, 1),        # off-color (a splashed copy in the deck): in neither k nor n
        ("r3", "UG", "Gold Dragon", 1, 0),     # RU card in a UG deck: not eligible
        ("r3", "UG", "Grey Golem", 1, 0),      # colorless: always eligible
        ("r3", "UG", "Blue Drake", 1, 1),
        ("r4", "W", "Mountain", 5, 5),         # basic: no signal
        ("r4", "W", "Red Bolt", 0, 0),         # not in the pool: not eligible
    ])
    c = pp.play_counts(long, CARDS).sort("name")
    got = {r["name"]: (r["k"], r["n"]) for r in c.iter_rows(named=True)}
    assert got == {"Red Bolt": (2, 2), "Gold Dragon": (1, 2), "Grey Golem": (1, 3), "Blue Drake": (1, 1)}
    names, k, n = pp.play_signal(long, CARDS)
    z = pp.signal_z(k, n)
    i = names.index("Gold Dragon")
    assert z[i] == pytest.approx(np.log((1 + 1) / (2 + 2 - 1 - 1)))       # (k+1)/(n+2) = 2/4 -> logit 0
    assert z[i] == pytest.approx(0.0)
    j = names.index("Red Bolt")
    assert z[j] == pytest.approx(np.log((3 / 4) / (1 / 4)))


def test_play_counts_aliases_and_missing_main_colors():
    long = _long([("r1", "RU", "Fire", 1, 1), ("r2", "RU", "Fire // Ice", 1, 0), ("r3", None, "Grey Golem", 1, 1)])
    c = pp.play_counts(long, CARDS)
    assert c.filter(pl.col("name") == "Fire // Ice").row(0, named=True)["n"] == 2          # front half and full name both map
    assert c.filter(pl.col("name") == "Fire // Ice").row(0, named=True)["k"] == 1
    assert c.filter(pl.col("name") == "Grey Golem").height == 0                             # run without main colors dropped


def test_minimum_30_eligible_runs_enforced_by_map_signal():
    n_card = 40
    cards = _cards([(i, f"C{i}", "", False, False, "common" if i % 2 else "uncommon") for i in range(n_card)])
    space = SimpleNamespace(nb_ids=np.arange(n_card), n_card=n_card)
    rng = np.random.default_rng(0)
    names = [f"C{i}" for i in range(n_card)]
    n = np.array([40.0] * 20 + [10.0] * 10 + [29.0] * 10)               # 20 cards with >= 30 eligible runs
    k = np.minimum(n, np.round(rng.uniform(0.1, 0.9, n_card) * n))
    z = pp.signal_z(k, n)
    beta0 = rng.normal(0, 0.1, n_card)
    prior, info = pickprior.map_signal_to_betas(cards, space, np.ones(n_card), beta0, z, n, names, min_taken=pp.MIN_RUNS)
    assert info["covered"] == 20 and len(prior) == n_card
    n2 = n.copy()
    n2[19] = 30.0
    n2[20] = 30.0                                                        # exactly 30 counts
    _p, info2 = pickprior.map_signal_to_betas(cards, space, np.ones(n_card), beta0, z, n2, names, min_taken=pp.MIN_RUNS)
    assert info2["covered"] == 21


def test_sealed_runs_deck_choice_bo1_and_bo3():
    t = datetime(2025, 1, 1)
    rows = []
    # bo1 run A: deck 10 in 2 games, deck 11 in 3 games -> 11 ; run B: tie 1-1, lower build_index (deck 21) wins
    for i, (dk, bi) in enumerate([(10, 0), (10, 0), (11, 1), (11, 1), (11, 1)]):
        rows.append(("A", "bo1", bi, i + 1, 1, dk, t.replace(minute=i), "RU", 0.7))
    rows += [("B", "bo1", 1, 1, 1, 20, t, "UG", 0.5), ("B", "bo1", 0, 2, 1, 21, t.replace(minute=5), "RG", 0.5)]
    # bo3 run C: match 1 g1 deck 30, g2 deck 31 (sideboarded); match 2 g1 deck 31; match 3 g1 deck 30 -> 30 and 31 tie with 2 and 1.. 30 wins
    rows += [("C", "bo3", 0, 1, 1, 30, t, "WB", 0.62), ("C", "bo3", 0, 1, 2, 31, t.replace(minute=1), "WB", 0.62),
             ("C", "bo3", 0, 2, 1, 31, t.replace(minute=2), "BG", 0.62), ("C", "bo3", 0, 3, 1, 30, t.replace(minute=3), "WB", 0.62)]
    g = pl.DataFrame(rows, schema=["draft_id", "fmt", "build_index", "match_number", "game_number", "deck_key", "game_time",
                                   "main_colors", "skill"], orient="row").with_columns(
        pl.col("game_time").cast(pl.Datetime("us")), pl.col("game_time").cast(pl.Datetime("us")).alias("draft_time"),
        pl.col("skill").cast(pl.Float32), pl.col("deck_key").cast(pl.Int64))
    r = pp.sealed_runs(g).sort("draft_id")
    got = {x["draft_id"]: (x["deck_key"], x["main_colors"], x["start"]) for x in r.iter_rows(named=True)}
    assert got["A"][0] == 11 and got["A"][1] == "RU"
    assert got["B"][0] == 21 and got["B"][1] == "RG"
    assert got["C"][0] == 30 and got["C"][1] == "WB"
    assert got["C"][2] == t
    assert r.filter(pl.col("draft_id") == "C")["skill"][0] == pytest.approx(0.62, abs=1e-6)


def _write_draft_csv(path: Path, gz: bool = False):
    weird = "Card, A\nB"                     # a card name with a comma and a newline inside the quoted header
    names = ["Red Bolt", weird, "Grey Golem"]
    meta = ["expansion", "event_type", "draft_id", "draft_time", "game_time", "build_index", "match_number", "game_number", "rank",
            "main_colors", "won", "user_n_games_bucket", "user_game_win_rate_bucket"]
    hdr = meta + [f"opening_hand_{n}" for n in names] + [f"deck_{n}" for n in names] + [f"sideboard_{n}" for n in names]

    def row(did, dt, bi, mn, gn, rank, mc, skill, deck, side):
        return (["X", "PremierDraft", did, dt, dt, bi, mn, gn, rank, mc, "True", "100", skill] + [0] * 3 + deck + side)

    rows = [
        row("d1", "2025-03-01 10:00:00", 1, 2, 1, "bronze", "UG", "0.50", [9, 9, 9], [9, 9, 9]),       # later game of d1: ignored
        row("d2", "2025-03-02 12:00:00", 0, 1, 1, "mythic", "RU", "0.58", [1, 0, 2], [0, 3, 0]),
        row("d1", "2025-03-01 10:00:00", 0, 1, 2, "gold", "RG", "0.65", [0, 0, 0], [0, 0, 0]),         # same build/match, game 2: ignored
        row("d1", "2025-03-01 10:00:00", 0, 1, 1, "diamond", "RU", "0.61", [1, 2, 0], [2, 0, 4]),      # d1's first game
        row("d2", "2025-03-02 12:00:00", 0, 1, 2, "mythic", "UG", "0.58", [7, 7, 7], [7, 7, 7]),
    ]
    op = gzip.open if gz else open
    with op(path, "wt", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        w.writerows(rows)
    return weird


@pytest.mark.parametrize("gz", [False, True])
def test_read_draft_runs_first_game_pool_is_deck_plus_sideboard(tmp_path, gz):
    p = tmp_path / ("g.csv.gz" if gz else "g.csv")
    weird = _write_draft_csv(p, gz)
    df = pp.read_draft_runs(p)
    assert set(df["draft_id"].unique()) == {"d1", "d2"}
    d1 = {r["card_name"]: (r["pool_n"], r["deck_n"]) for r in df.filter(pl.col("draft_id") == "d1").iter_rows(named=True)}
    assert d1 == {"Red Bolt": (3, 1), weird: (2, 2), "Grey Golem": (4, 0)}
    d2 = {r["card_name"]: (r["pool_n"], r["deck_n"]) for r in df.filter(pl.col("draft_id") == "d2").iter_rows(named=True)}
    assert d2 == {"Red Bolt": (1, 1), weird: (3, 0), "Grey Golem": (2, 2)}
    r1 = df.filter(pl.col("draft_id") == "d1").row(0, named=True)
    assert (r1["rank"], r1["main_colors"], r1["skill"]) == ("diamond", "RU", pytest.approx(0.61))
    assert r1["draft_time"] == datetime(2025, 3, 1, 10, 0, 0)
    r2 = df.filter(pl.col("draft_id") == "d2").row(0, named=True)
    assert (r2["rank"], r2["main_colors"], r2["skill"]) == ("mythic", "RU", pytest.approx(0.58))


def test_arm_selection_thresholds():
    t = datetime(2025, 3, 5)
    rows = []
    for did, rank, skill, when in [("a", "Mythic", 0.56, datetime(2025, 3, 1)), ("b", "gold", 0.70, datetime(2025, 3, 1)),
                                   ("c", "diamond", 0.54, datetime(2025, 3, 1)), ("d", "diamond", 0.60, datetime(2025, 3, 9)),
                                   ("e", "platinum", 0.60, datetime(2025, 3, 1))]:
        rows.append((did, when, rank, skill, "RU", "Red Bolt", 1, 1))
    d = pl.DataFrame(rows, schema=["draft_id", "draft_time", "rank", "skill", "main_colors", "card_name", "pool_n", "deck_n"], orient="row")
    assert sorted(pp.arm_long("D_skill", None, d, t)["draft_id"]) == ["b", "e"]          # skill >= 0.60, before the cut
    assert sorted(pp.arm_long("D_rank", None, d, t)["draft_id"]) == ["a"]                # diamond/mythic and skill >= 0.55
