"""scripts/pick_model_check.py: the deck rebuild (0.5 threshold, basics split, 20-26 bounds), the record expansion, the 12-day window."""
from __future__ import annotations

import gzip
import importlib.util
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))
_SPEC = importlib.util.spec_from_file_location("pick_model_check", _SCRIPTS / "pick_model_check.py")
pm = importlib.util.module_from_spec(_SPEC)
sys.modules["pick_model_check"] = pm          # dataclasses look the module up by name
_SPEC.loader.exec_module(pm)


def _cards() -> pl.DataFrame:
    """ids 0-5 basics (Plains Island Swamp Mountain Forest + Wastes), 6.. white spells, 8.. blue spells, a hybrid, a colorless card, a land."""
    rows = []

    def add(name, pips="", land=False, basic=False, hybrid=False, mv=2.0, produces=""):
        w, u, b, r, g = (pips.count(c) for c in "WUBRG")
        rows.append(dict(card_id=len(rows), name=name, front_name=name.split(" // ")[0], rarity="common", colors="".join(c for c in "WUBRG" if c in pips),
                         mana_value=mv, pip_W=w, pip_U=u, pip_B=b, pip_R=r, pip_G=g, hybrid=hybrid, is_land=land, is_basic=basic,
                         is_creature=not land, produces=produces, oracle_text=""))

    for nm, col in (("Plains", "W"), ("Island", "U"), ("Swamp", "B"), ("Mountain", "R"), ("Forest", "G")):
        add(nm, land=True, basic=True, mv=0.0, produces=col)
    add("White A", "WW")            # 5
    add("White B", "W")             # 6
    add("Blue A", "UU")             # 7
    add("Blue B", "U")              # 8
    add("Grey Golem")               # 9
    add("Dual Land", land=True, mv=0.0, produces="WU")  # 10
    add("Fire // Ice", "R")         # 11
    for i in range(30):
        add(f"Filler {i}")          # 12..41 colorless
    return pl.DataFrame(rows).with_columns(pl.col("pip_W", "pip_U", "pip_B", "pip_R", "pip_G").cast(pl.Int8))


CARDS = _cards()
CTX = pm.make_ctx(CARDS)
NAME = {r["card_id"]: r["name"] for r in CARDS.iter_rows(named=True)}


def _nonbasic(n_white_sym: int, n_blue_sym: int, total: int) -> dict[int, int]:
    """A deck of `total` nonbasic cards with about the given colored symbols (W: White A x2 each, U: Blue A x2 each) padded with fillers."""
    d: dict[int, int] = {}
    if n_white_sym:
        d[5] = n_white_sym // 2
    if n_blue_sym:
        d[7] = n_blue_sym // 2
    fill = total - sum(d.values())
    for i in range(fill):
        d[12 + i] = 1
    return d


# ---------------------------------------------------------------------------------------------- basics split
def test_split_basics_proportional_and_sums():
    out = pm.split_basics(17, {"W": 10.0, "U": 5.0})
    assert sum(out.values()) == 17
    assert out == {"W": 11, "U": 6}


def test_split_basics_rounding_goes_to_the_color_with_most_symbols():
    # 3 colors, 15 basics: raw 6.43 / 4.29 / 4.29 -> the two smaller round to 4 each, the biggest takes the rest (7)
    out = pm.split_basics(15, {"W": 6.0, "U": 4.0, "B": 4.0})
    assert out == {"W": 7, "U": 4, "B": 4}
    assert sum(out.values()) == 15


def test_split_basics_tie_top_color_is_first_in_wubrg_order_and_absorbs_the_rounding():
    out = pm.split_basics(17, {"U": 5.0, "W": 5.0})        # W is "the color with the most symbols" (tie); U = round(8.5) = 9; W takes the rest
    assert out == {"W": 8, "U": 9}


def test_split_basics_one_color_gets_all():
    assert pm.split_basics(17, {"R": 12.0}) == {"R": 17}


# ---------------------------------------------------------------------------------------------- deck rebuild
def test_rebuild_deck_adds_basics_up_to_40_split_by_symbols():
    nb = _nonbasic(8, 4, 23)               # 4 White A (8 W symbols), 2 Blue A (4 U symbols), 17 fillers
    deck, reason, main = pm.rebuild_deck(nb, CTX)
    assert reason is None and main == "WU"
    assert sum(deck.values()) == 40
    plains, islands = deck.get(CTX.basic_ids["W"], 0), deck.get(CTX.basic_ids["U"], 0)
    assert plains + islands == 17
    assert (plains, islands) == (11, 6)    # 17 * 8/12 = 11.33 -> U rounds to 6, W takes the rest
    assert {c: n for c, n in deck.items() if c not in CTX.basic_ids.values()} == nb


def test_rebuild_deck_bounds_20_to_26_nonbasic():
    for total, ok in ((19, False), (20, True), (26, True), (27, False)):
        deck, reason, _ = pm.rebuild_deck(_nonbasic(8, 0, total), CTX)
        assert (deck is not None) is ok, total
        if not ok:
            assert reason == ("too_few_nonbasic" if total < 20 else "too_many_nonbasic")
        else:
            assert sum(deck.values()) == 40


def test_rebuild_deck_nonbasic_land_counts_as_nonbasic():
    nb = _nonbasic(8, 0, 19)
    nb[10] = 1                              # a nonbasic land: 20 nonbasic cards
    deck, reason, _ = pm.rebuild_deck(nb, CTX)
    assert reason is None
    assert sum(n for c, n in deck.items() if c in CTX.basic_ids.values()) == 20


def test_rebuild_deck_no_main_color_dropped():
    deck, reason, _ = pm.rebuild_deck({12 + i: 1 for i in range(23)}, CTX)      # colorless only
    assert deck is None and reason == "no_main_color"


def _picks(rows):
    """rows: (draft_id, draft_time, wins, losses, skill, pick, rate)."""
    return pl.DataFrame(rows, schema=dict(draft_id=pl.String, draft_time=pl.Datetime("us"), wins=pl.Float64, losses=pl.Float64, skill=pl.Float64,
                                          pick=pl.String, rate=pl.Float64), orient="row").with_columns(pl.lit(None, dtype=pl.String).alias("rank"))


def _run_rows(did, t, w, l, rates_names, skill=0.55):
    return [(did, t, w, l, skill, nm, r) for nm, r in rates_names]


def test_build_runs_threshold_is_half_inclusive():
    t = datetime(2026, 1, 1)
    names = [f"Filler {i}" for i in range(18)] + ["White A", "White A", "White B", "Blue A", "Blue B"]
    rows = _run_rows("a", t, 3.0, 2.0, [(n, 0.5) for n in names])                      # exactly 0.5 -> main deck (23 nonbasic)
    rows += _run_rows("b", t, 3.0, 2.0, [(n, 0.4999) for n in names])                  # below -> empty deck -> too few
    rb = pm.build_runs(_picks(rows), CTX)
    assert rb.runs["draft_id"].to_list() == ["a"]
    assert rb.info["dropped"]["too_few_nonbasic"] == 1
    assert sum(rb.nonbasic["a"].values()) == 23
    assert rb.nonbasic["a"][5] == 2                                                    # one copy per pick: White A picked twice


def test_build_runs_drops_runs_without_games_and_reports_reasons():
    t = datetime(2026, 1, 1)
    names = [(f"Filler {i}", 1.0) for i in range(15)] + [("White A", 1.0)] * 4 + [("White B", 1.0)] * 4
    rows = _run_rows("ok", t, 2.0, 1.0, names)
    rows += _run_rows("nog", t, 0.0, 0.0, names)
    rows += _run_rows("nan", t, None, None, names)
    rows += _run_rows("few", t, 2.0, 1.0, names[:10])
    rb = pm.build_runs(_picks(rows), CTX)
    assert rb.runs["draft_id"].to_list() == ["ok"]
    assert rb.info["dropped"] == dict(no_games=2, too_few_nonbasic=1, too_many_nonbasic=0, no_main_color=0)
    assert rb.info["runs_in_window"] == 4 and rb.info["runs_rebuilt"] == 1


def test_build_runs_matches_split_names_and_skips_basic_picks():
    t = datetime(2026, 1, 1)
    names = [(f"Filler {i}", 1.0) for i in range(20)] + [("Fire", 1.0), ("Plains", 1.0), ("Not A Card", 1.0), ("Fire // Ice", 1.0), ("White A", 1.0)]
    rb = pm.build_runs(_picks(_run_rows("a", t, 1.0, 1.0, names)), CTX)
    assert rb.info["n_basic_picks_skipped"] == 1
    assert rb.info["n_unmatched_main_picks"] == 1 and "Not A Card" in rb.info["unmatched_names"]
    assert rb.nonbasic["a"][11] == 2                                                   # "Fire" (front half) and "Fire // Ice" are the same card
    assert sum(rb.nonbasic["a"].values()) == 20 + 2 + 1


# ---------------------------------------------------------------------------------------------- record expansion
def test_expand_records_wins_then_losses():
    idx, won = pm.expand_records(np.array([2, 0, 1]), np.array([1, 3, 0]))
    assert idx.tolist() == [0, 0, 0, 1, 1, 1, 2]
    assert won.tolist() == [True, True, False, False, False, False, True]


def test_expand_records_empty_run():
    idx, won = pm.expand_records(np.array([0, 0]), np.array([0, 0]))
    assert len(idx) == 0 and len(won) == 0


def test_runs_to_games_rows_and_columns():
    t = datetime(2026, 1, 1, 12)
    names = [(f"Filler {i}", 1.0) for i in range(16)] + [("White A", 1.0)] * 4 + [("Blue A", 1.0)] * 2 + [("White B", 1.0)]
    rb = pm.build_runs(_picks(_run_rows("a", t, 7.0, 2.0, names, skill=0.6) + _run_rows("b", t, 0.0, 1.0, names, skill=None)), CTX)
    g = pm.runs_to_games(rb, np.arange(len(rb.decks)) + 1000)
    assert g.height == 10
    assert g.filter(pl.col("draft_id") == "a")["won"].sum() == 7 and g.filter(pl.col("draft_id") == "a").height == 9
    assert g.filter(pl.col("draft_id") == "b")["won"].sum() == 0
    assert g["on_play"].dtype == pl.Float64 and set(g["on_play"].to_list()) == {0.5}
    assert set(g["num_mulligans"].to_list()) == {0} and set(g["opp_num_mulligans"].to_list()) == {0}
    assert g.filter(pl.col("draft_id") == "a")["skill"].to_list()[0] == pytest.approx(0.6, abs=1e-6)
    assert g.filter(pl.col("draft_id") == "b")["skill"].null_count() == 1
    assert (g["game_time"] == g["draft_time"]).all() and set(g["event"].to_list()) == {"PremierDraft"} and set(g["fmt"].to_list()) == {"draft"}
    for c in pm.pipeline.GAME_COLS:
        assert c in g.columns, c


def test_on_play_encodes_as_half():
    from sealed_lab.model import design

    t = datetime(2026, 1, 1, 12)
    names = [(f"Filler {i}", 1.0) for i in range(16)] + [("White A", 1.0)] * 4 + [("Blue A", 1.0)] * 2 + [("White B", 1.0)]
    rb = pm.build_runs(_picks(_run_rows("a", t, 3.0, 2.0, names)), CTX)
    g = pm.runs_to_games(rb, np.arange(len(rb.decks)) + 1000)
    enc = design.CovEncoder.fit(g)
    C = enc.transform(g).toarray()
    assert np.allclose(C[:, enc.names.index("on_play")], 0.5)
    assert np.allclose(C[:, enc.names.index("mull1")], 0.0)


# ---------------------------------------------------------------------------------------------- 12-day window
def test_window_first_12_days_after_earliest_draft_time():
    t0 = datetime(2026, 3, 3, 23, 59, 23)
    df = pl.DataFrame(dict(draft_id=list("abcde"), draft_time=[t0, t0 + timedelta(days=11, hours=23), t0 + timedelta(days=12) - timedelta(seconds=1),
                                                                 t0 + timedelta(days=12), t0 + timedelta(days=30)]), schema_overrides=dict(draft_time=pl.Datetime("us")))
    win, start = pm.window_filter(df)
    assert start == t0
    assert win["draft_id"].to_list() == ["a", "b", "c"]            # t0 <= draft_time < t0 + 12 days


def test_window_anchor_is_the_file_earliest_not_the_first_row():
    t0 = datetime(2026, 1, 1)
    df = pl.DataFrame(dict(draft_id=["late", "early"], draft_time=[t0 + timedelta(days=11), t0]), schema_overrides=dict(draft_time=pl.Datetime("us")))
    win, start = pm.window_filter(df)
    assert start == t0 and sorted(win["draft_id"].to_list()) == ["early", "late"]


# ---------------------------------------------------------------------------------------------- pick file reading
def test_read_pick_file_parses_float_counts(tmp_path):
    p = tmp_path / "draft_data_public.XXX.PremierDraft.csv.gz"
    hdr = "draft_id,draft_time,rank,event_match_wins,event_match_losses,pick,pick_maindeck_rate,user_game_win_rate_bucket,pack_card_X"
    rows = ["d1,2026-03-03 10:00:00,gold,3.0,2.0,White A,1.0,0.55,1", "d1,2026-03-03 10:00:00,gold,3.0,2.0,Blue A,0.0,0.55,0",
            "d2,2026-03-04 10:00:00,,,,White B,,,0"]
    with gzip.open(p, "wt", encoding="utf-8") as f:
        f.write(hdr + "\n" + "\n".join(rows) + "\n")
    df = pm.read_pick_file(p)
    assert df.columns[:8] == ["draft_id", "draft_time", "rank", "wins", "losses", "pick", "rate", "skill"] or set(df.columns) >= {"wins", "losses", "rate", "skill"}
    assert df["wins"].to_list() == [3.0, 3.0, None]
    assert df["draft_time"].dtype == pl.Datetime("us")
    assert pm._count(3.0) == 3 and pm._count(None) == 0 and pm._count(float("nan")) == 0 and pm._count(2.6) == 3


# ---------------------------------------------------------------------------------------------- transfer / safety helpers
def test_is_transferred_names():
    for n in ("intercept", "on_play", "mull1", "mull2p", "omull1", "omull2p", "skill_lin", "skill_na", "skill_bin:0.54", "opp:WU", "opp:other"):
        assert pm.is_transferred(n), n
    for n in ("event:Sealed", "t", "t_h1", "t_h2", "oppflag:W"):
        assert not pm.is_transferred(n), n


def test_extend_space_keys_above_real_and_same_standardisation():
    from sealed_lab.model import design, structure

    n_nb = 4
    keys = np.array([10, 20, 30], dtype=np.int64)
    phi = np.random.default_rng(0).normal(size=(3, structure.N_FEATURES))
    cards = pl.DataFrame(dict(card_id=list(range(6)), is_basic=[False] * n_nb + [True] * 2))
    dc = pl.DataFrame(dict(deck_key=[10, 20, 30], card_id=[0, 1, 2], n=[1, 1, 2]), schema_overrides=dict(deck_key=pl.Int64, card_id=pl.Int32, n=pl.Int8))
    space = design.build_deck_space(keys, phi, dc, cards)
    new_phi = np.random.default_rng(1).normal(size=(2, structure.N_FEATURES))
    ext, nk = pm.extend_space(space, [{0: 2, 4: 5}, {3: 1}], new_phi)
    assert nk.min() > keys.max() and ext.keys.tolist() == sorted(ext.keys.tolist())
    assert np.array_equal(ext.mu, space.mu) and np.array_equal(ext.sd, space.sd)
    D = ext.D.toarray()
    r0, r1 = ext.rows(nk)
    assert D[r0, 0] == 2 and D[r1, 3] == 1
    assert np.allclose(D[r0, n_nb:], (new_phi[0] - space.mu) / space.sd)
    assert np.allclose(ext.D.toarray()[:3], space.D.toarray())          # real rows untouched


def test_predict_arm_zero_cards_is_z():
    from sealed_lab.model import glm
    import scipy.sparse as sp

    rng = np.random.default_rng(0)
    C = sp.csr_matrix(rng.normal(size=(6, 2)))
    D = sp.csr_matrix(rng.normal(size=(3, 5)))
    prob = glm.Problem(y=np.array([1, 0, 1, 0, 1, 0.0]), C=C, D=D, idx=np.array([0, 1, 2, 0, 1, 2], dtype=np.int32))
    cov, theta = np.array([0.1, -0.2]), np.array([0.3, 0.1])
    cards = rng.normal(size=3)
    z = pm.predict_arm(prob, cov, np.zeros(3), theta)
    assert np.array_equal(pm.predict_arm(prob, cov, cards * 0.0, theta), z)
    assert not np.array_equal(pm.predict_arm(prob, cov, cards, theta), z)
