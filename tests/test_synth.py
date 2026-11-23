from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest
from scipy.stats import spearmanr

from sealed_lab.config import BASICS
from sealed_lab.synth import generate as G

META = ["expansion", "event_type", "draft_id", "draft_time", "game_time", "build_index", "match_number",
        "game_number", "rank", "opp_rank", "main_colors", "splash_colors", "on_play", "num_mulligans",
        "opp_num_mulligans", "opp_colors", "num_turns", "won", "user_n_games_bucket", "user_game_win_rate_bucket"]


def _read(synth_dir, ev):
    return pl.read_csv(synth_dir / f"game_data_public.TST.{ev}.csv.gz", infer_schema_length=2000,
                       schema_overrides={"won": pl.String, "on_play": pl.String})


@pytest.fixture(scope="module")
def truth(synth_dir):
    return json.loads((synth_dir / "truth.json").read_text(encoding="utf-8"))


def test_files_and_format(synth_dir, truth):
    names = sorted(truth["cards"])
    assert 140 <= len(names) <= 170
    assert {"Plains", "Island", "Swamp", "Mountain", "Forest"} <= set(names)
    for ev in G.EVENTS:
        df = _read(synth_dir, ev)
        assert df.columns[:20] == META
        rest = df.columns[20:]
        assert len(rest) == 5 * len(names)
        for i, pre in enumerate(["opening_hand_", "drawn_", "tutored_", "deck_", "sideboard_"]):
            assert rest[i * len(names):(i + 1) * len(names)] == [pre + n for n in names]
        assert set(df["won"].unique()) <= {"True", "False"}
        assert set(df["on_play"].unique()) <= {"True", "False"}
        assert (df["expansion"] == "TST").all() and (df["event_type"] == ev).all()
    assert (synth_dir / "draft_data_public.TST.PremierDraft.csv.gz").exists()
    d = pl.read_csv(synth_dir / "draft_data_public.TST.PremierDraft.csv.gz")
    assert d.height == 100 and "pick" in d.columns and any(c.startswith("pack_card_") for c in d.columns)


def test_special_names(truth):
    names = list(truth["cards"])
    assert any("," in n for n in names)
    assert any("'" in n for n in names)
    assert any("//" in n for n in names)
    assert any(any(ord(ch) > 127 for ch in n) for n in names)


def test_decks_and_hands_consistent(synth_dir, truth):
    names = sorted(truth["cards"])
    nb = [n for n in names if n not in BASICS]
    land = np.array([truth["cards"][n]["is_land"] for n in names])
    for ev in G.EVENTS:
        df = _read(synth_dir, ev)
        deck = df.select([f"deck_{n}" for n in names]).to_numpy()
        side = df.select([f"sideboard_{n}" for n in names]).to_numpy()
        op = df.select([f"opening_hand_{n}" for n in names]).to_numpy()
        dr = df.select([f"drawn_{n}" for n in names]).to_numpy()
        assert (deck.sum(1) == 40).all()
        nl = deck[:, land].sum(1)
        assert nl.min() >= 16 and nl.max() <= 18
        assert (op + dr <= deck).all()
        mull = df["num_mulligans"].to_numpy()
        assert (op.sum(1) == 7 - mull).all()
        bidx = [names.index(b) for b in BASICS if b in names]
        assert side[:, bidx].sum() == 0
        # two colours (+ occasional splash)
        mc = df["main_colors"]
        assert (mc.str.len_chars() == 2).all()
        assert (df["splash_colors"].str.len_chars() <= 1).mean() > 0.95
        # deck + sideboard pool is invariant per event (basics excluded)
        pool = deck + side
        ids = df["draft_id"].to_numpy()
        for did in np.unique(ids)[:60]:
            rows = pool[ids == did][:, [names.index(n) for n in nb]]
            assert (rows == rows[0]).all()
        if "Sealed" in ev:
            assert (pool[:, [names.index(n) for n in nb]].sum(1) == 84).all()  # 6 boosters x 14


def test_run_structure(synth_dir):
    s = _read(synth_dir, "Sealed")
    for did, g in s.group_by("draft_id"):
        w = (g["won"] == "True").sum()
        l = g.height - w
        assert (w == 7 or l == 3) and w <= 7 and l <= 3
        assert g["match_number"].to_list() == list(range(1, g.height + 1))
    t = _read(synth_dir, "TradSealed")
    n_full = 0
    for did, g in t.group_by("draft_id", maintain_order=True):
        mw = ml = 0
        for m, mg in g.group_by("match_number", maintain_order=True):
            gw = (mg["won"] == "True").sum()
            gl = mg.height - gw
            assert mg["game_number"].to_list() == list(range(1, mg.height + 1))
            assert gw == 2 or gl == 2
            if gw == 2:
                mw += 1
            else:
                ml += 1
        assert mw == 4 or ml == 2
        n_full += 1
    assert n_full > 50


def test_truth_function(synth_dir, truth):
    held = json.loads((synth_dir / "heldout_pools.json").read_text(encoding="utf-8"))
    assert len(held) >= 30
    vals = []
    for h in held:
        assert sum(h["player_deck"].values()) == 40
        assert sum(h["pool"].values()) == 84
        a = G.truth_strength(truth, h["player_deck"])
        b = G.true_strength(h["player_deck"])  # default world == same seed
        assert abs(a - b) < 1e-9
        vals.append(a)
    assert 0.1 < np.std(vals) < 1.5


def test_land_count_effect_peaks_at_17(truth):
    t = json.loads(json.dumps(truth))
    t["beta"] = {k: 0.0 for k in t["beta"]}
    t["colour_screw"]["gamma"] = 0.0
    arr = G._arrays_from_truth(t)
    names = arr["names"]
    vals = {}
    for nl in (15, 16, 17, 18, 19):
        D = np.zeros((1, len(names)))
        D[0, names.index("Plains")] = nl
        D[0, names.index("Aldric, Dawn's Herald")] = 40 - nl
        vals[nl] = G.strength_matrix(t, arr, D)[0]
    assert max(vals, key=vals.get) == 17
    assert vals[16] == pytest.approx(vals[18])


def test_colour_screw_penalises_bad_mana(truth):
    t = json.loads(json.dumps(truth))
    t["beta"] = {k: 0.0 for k in t["beta"]}
    t["land"]["coef"] = 0.0
    arr = G._arrays_from_truth(t)
    names = arr["names"]
    ix = names.index

    def deck(w_basics, r_basics):
        D = np.zeros((1, len(names)))
        D[0, ix("Plains")], D[0, ix("Mountain")] = w_basics, r_basics
        D[0, ix("Argent Lancer")] = 12   # white single pip
        D[0, ix("Blazing Giant")] = 11   # red
        return G.strength_matrix(t, arr, D)[0]

    assert deck(9, 8) > deck(14, 3)
    assert deck(8, 9) > deck(3, 14)


def test_signal_visible_in_outcomes(synth_dir, truth):
    """Decks with higher planted strength win more often (a coarse recoverability check)."""
    names = sorted(truth["cards"])
    df = pl.concat([_read(synth_dir, "Sealed"), _read(synth_dir, "TradSealed"), _read(synth_dir, "PremierDraft")])
    deck = df.select([f"deck_{n}" for n in names]).to_numpy()
    arr = G._arrays_from_truth(truth)
    ts = G.strength_matrix(truth, arr, deck)
    won = (df["won"] == "True").to_numpy().astype(float)
    skill = df["user_game_win_rate_bucket"].fill_null(0.55).to_numpy()
    q = np.digitize(ts, np.quantile(ts, [0.2, 0.4, 0.6, 0.8]))
    wr = [won[q == i].mean() for i in range(5)]
    assert wr[4] - wr[0] > 0.10
    assert all(np.diff(wr) > -0.02)
    assert spearmanr(skill, won)[0] > 0.02
