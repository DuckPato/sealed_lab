"""Mana simulator v2: oracle-text card flow, regime hands, keep rule, calibration helpers, regime-aware deck space, pipeline."""
from __future__ import annotations

import shutil
import time

import numpy as np
import polars as pl
import pytest
from scipy.stats import hypergeom

from sealed_lab import config
from sealed_lab.model import design, manasim, simcalib, structure
from sealed_lab.model.manasim import CardArrays, parse_card_flow

from .test_model_manasim import DUAL, ISLAND, PLAINS, W2, W3, make_cards


# ------------------------------------------------------------------------------------------ oracle text

def test_parse_card_flow():
    assert parse_card_flow("Draw a card.") == (1, 0)
    assert parse_card_flow("When this creature enters, draw two cards.") == (2, 0)
    assert parse_card_flow("Target opponent draws a card.") == (0, 0)
    assert parse_card_flow("Draw a card, then discard a card.") == (0, 0)
    assert parse_card_flow("Whenever you cast a spell, you may draw a card.\nDraw three cards.") == (3, 0)
    assert parse_card_flow("Search your library for a basic land card, reveal it, put it into your hand.") == (0, 1)
    assert parse_card_flow("Search your library for a basic land card, put it onto the battlefield tapped, then shuffle.") == (0, 2)
    assert parse_card_flow("When this enters, draw a card, then you may put a land card from your hand onto the battlefield.") == (1, 3)
    assert parse_card_flow("Search your library for a creature card") == (0, 0)


def test_cardarrays_reads_oracle_text():
    df = pl.DataFrame({
        "card_id": [0, 1, 2], "is_land": [True, False, False], "is_basic": [True, False, False], "is_creature": [False, False, False],
        "mana_value": [0.0, 2.0, 3.0], "pip_W": [0, 1, 1], "pip_U": [0, 0, 0], "pip_B": [0, 0, 0], "pip_R": [0, 0, 0], "pip_G": [0, 0, 0],
        "hybrid": [False] * 3, "produces": ["W", "", ""], "oracle_text": ["{T}: Add {W}.", "Draw a card.", "Search your library for a land card."]})
    ca = CardArrays.from_df(df)
    assert list(ca.draw_arr) == [0, 1, 0] and list(ca.fetch_arr) == [0, 0, 1]


# ------------------------------------------------------------------------------------------ hands, regimes

def _plain_deck(L=17):
    ca = make_cards([PLAINS, ISLAND, dict(mv=1, pips="W", creature=True), W2, W3, dict(mv=4, pips="U", creature=True),
                     dict(mv=5, pips="WU", creature=True)])
    deck = {0: L // 2 + L % 2, 1: L // 2, 2: 3, 3: 5, 4: 5, 5: 5, 6: 40 - L - 18}
    return ca, deck


def test_hand_land_pmf_regimes():
    L, n = 17, 40
    ref = hypergeom.pmf(np.arange(8), n, L, 7)
    assert np.allclose(manasim.hand_land_pmf(L, n, "random"), ref)
    pd_, ps = manasim.hand_land_pmf(L, n, "hand"), manasim.hand_land_pmf(L, n, "sealed")
    assert pd_.sum() == pytest.approx(1) and ps.sum() == pytest.approx(1)
    assert 0.54 < pd_[3] < 0.58 and 0.63 < ps[3] < 0.67          # the measured P(3 lands in the first 7 cards) at 17 lands
    assert pd_[0] + pd_[1] + pd_[5] + pd_[6] + pd_[7] < 0.012 and ps[:2].sum() + ps[5:].sum() < 0.004
    assert abs(float(pd_ @ np.arange(8)) - 7 * L / n) < 0.05      # the smoother keeps the mean


def test_v2_deterministic_and_order_invariant():
    ca, d = _plain_deck()
    a = manasim.simulate(d, ca, version=2, regime="hand", extra=True)
    assert a == manasim.simulate(dict(reversed(list(d.items()))), ca, version=2, regime="hand", extra=True)
    assert all(np.isfinite(v) for v in a.values())
    with pytest.raises(ValueError):
        manasim.simulate(d, ca, version=2, regime="nope")


def test_v2_regimes_screw_and_mulligan_rates():
    ca, d = _plain_deck(17)
    r = {g: manasim.simulate(d, ca, 4000, version=2, regime=g) for g in config.REGIMES}
    assert r["sealed"]["p_screw"] < r["hand"]["p_screw"] < r["random"]["p_screw"]
    assert r["sealed"]["p_flood"] < r["random"]["p_flood"]
    assert 0.17 < r["random"]["mull_rate"] < 0.25            # real Trad queues mulligan ~20%
    assert 0.08 < r["hand"]["mull_rate"] < 0.16 and 0.07 < r["sealed"]["mull_rate"] < 0.14
    assert manasim.simulate(d, ca, 4000)["mull_rate"] < r["random"]["mull_rate"]    # v1 keeps every 2-5 land hand


def test_v2_first_hand_follows_the_regime_pmf():
    ca, d = _plain_deck(17)
    for reg in config.REGIMES:
        det = manasim.simulate(d, ca, 4000, version=2, regime=reg, detail=True)
        pmf = manasim.hand_land_pmf(17, 40, reg)
        assert np.max(np.abs(np.array(det["hand7"]) - pmf)) < 0.03, reg
        assert len(det["lands_by_turn"]) == 7 and det["lands_by_turn"][0] == pytest.approx(1.0, abs=0.02)
        assert all(a <= b + 1e-9 for a, b in zip(det["lands_by_turn"], det["lands_by_turn"][1:]))


def test_v2_card_flow_adds_cards_and_lands():
    ca = make_cards([PLAINS, DUAL, dict(mv=2, pips="W", creature=True), dict(mv=2, pips="W", creature=True)])
    object.__setattr__(ca, "draw_n", np.array([0, 0, 0, 1]))
    object.__setattr__(ca, "fetch", np.array([0, 0, 0, 0]))
    deck = {0: 14, 1: 3, 2: 23}
    deck_draw = {0: 14, 1: 3, 2: 13, 3: 10}
    base = manasim.simulate(deck, ca, 3000, version=2, regime="random", detail=True)
    flow = manasim.simulate(deck_draw, ca, 3000, version=2, regime="random", detail=True)
    assert base["xdraw_by_turn"][-1] == 0.0 and flow["xdraw_by_turn"][-1] > 0.5
    ca2 = make_cards([PLAINS, DUAL, dict(mv=2, pips="W", creature=True), dict(mv=2, pips="W", creature=True)])
    object.__setattr__(ca2, "fetch", np.array([0, 0, 0, 2]))
    fetch = manasim.simulate(deck_draw, ca2, 3000, version=2, regime="random", detail=True)
    assert fetch["lands_by_turn"][4] > base["lands_by_turn"][4] + 0.05


def test_v2_speed():
    ca, d = _plain_deck()
    manasim.simulate(d, ca, version=2, regime="sealed")
    t = time.perf_counter()
    for k in range(20):
        manasim.simulate({**d, 3: 5 + (k % 2), 4: 5 - (k % 2)}, ca, version=2, regime="sealed")
    assert (time.perf_counter() - t) / 20 < 0.04          # ~10 ms on a normal machine; generous for a loaded CI box


def test_v1_unchanged_by_the_default_dispatcher():
    ca, d = _plain_deck()
    assert manasim.simulate(d, ca) == manasim.simulate_v1(d, ca)
    assert set(manasim.simulate(d, ca, detail=True)) >= {"lands_by_turn", "mana_by_turn", "hand7"}


# ------------------------------------------------------------------------------------------ structure / caches

def test_featurize_version_and_regime():
    ca, d = _plain_deck()
    f1 = structure.featurize(d, ca)
    fr = structure.featurize(d, ca, version=2, regime="random")
    fs = structure.featurize(d, ca, version=2, regime="sealed")
    n = structure.N_STRUCT
    assert np.array_equal(f1[:n], fr[:n]) and np.array_equal(fr[:n], fs[:n])      # structure counts do not depend on the simulator
    assert not np.array_equal(fr[n:], fs[n:]) and not np.array_equal(f1[n:], fr[n:])
    assert np.array_equal(fs, structure.featurize(d, ca, version=2, regime="sealed"))
    sigs = {structure.cards_signature(ca, v, r) for v, r in ((1, None), (2, "random"), (2, "hand"), (2, "sealed"))}
    assert len(sigs) == 4
    assert structure.cards_signature(ca) == structure.cards_signature(ca, 1, None)


# ------------------------------------------------------------------------------------------ calibration helpers

def test_fit_hand_smoother_recovers_parameters():
    rng = np.random.default_rng(0)
    tabs = {L: rng.multinomial(60000, simcalib.hand_pmf(L, s=1.1, g=2.4)).astype(float) for L in (15, 16, 17, 18)}
    fit = simcalib.fit_hand_smoother(tabs)
    assert fit["smoothed"] and abs(fit["s"] - 1.1) < 0.06 and abs(fit["g"] - 2.4) < 0.25
    plain = {L: rng.multinomial(60000, simcalib.hand_pmf(L)).astype(float) for L in (16, 17)}
    assert not simcalib.fit_hand_smoother(plain)["smoothed"]


def test_smooth_keep_shrinks_thin_cells_to_the_prior():
    cnt = np.zeros((3, 2, 8, 2))
    cnt[0, :, 3] = [[4000, 260], [4000, 240]]
    cnt[0, :, 2] = [[900, 150], [900, 170]]
    cnt[0, :, 1] = [[10, 0], [10, 0]]             # a hand never seen: the generic prior (mulligan), not the 0 of 20 observations
    p = simcalib.smooth_keep(cnt)
    assert p.shape == (3, 2, 8) and 0.05 < p[0, 0, 3] < 0.07 and 0.15 < p[0, 1, 2] < 0.2
    assert p[0, 0, 1] > 0.8 and (p >= 0).all() and (p <= 1).all()


# ------------------------------------------------------------------------------------------ regime-aware deck space

def _tiny_space():
    cards = pl.DataFrame({"card_id": [0, 1, 2], "is_basic": [True, False, False]})
    dc = pl.DataFrame({"deck_key": [10, 10, 11, 11], "card_id": [1, 2, 1, 2], "n": [1, 1, 2, 1]})
    keys = np.array([10, 11])
    k = structure.N_FEATURES
    p_rand = np.tile(np.arange(k, dtype=float), (2, 1)) + np.array([[0.0], [1.0]])
    p_sealed = p_rand + 5.0
    return design.build_deck_space(keys, None, dc, cards, regime_phi={"random": p_rand, "sealed": p_sealed},
                                   regime_weights={"random": np.array([3.0, 1.0]), "sealed": np.array([1.0, 3.0])}), p_rand, p_sealed


def test_deck_space_regimes_share_the_standardisation():
    sp, p_rand, p_sealed = _tiny_space()
    assert list(sp.regime_phi)[0] == "random" or list(sp.regime_phi)[0] == "sealed"
    Dr, Ds = sp.D_regime("random"), sp.D_regime("sealed")
    nc = sp.n_card
    assert np.allclose(Dr[:, :nc].toarray(), Ds[:, :nc].toarray())
    zr, zs = Dr[:, nc:nc + sp.n_feat].toarray(), Ds[:, nc:nc + sp.n_feat].toarray()
    assert np.allclose(zs - zr, 5.0 / sp.sd)                       # same mu / sd: only the raw features differ
    assert np.allclose(zr, (p_rand - sp.mu) / sp.sd)


def test_make_problem_uses_the_games_regime():
    sp, p_rand, p_sealed = _tiny_space()
    g = pl.DataFrame({"deck_key": [10, 10, 11, 10], "event": ["TradSealed", "Sealed", "Sealed", "TradSealed"],
                      "won": [True, False, True, False], "on_play": [True] * 4, "num_mulligans": [0] * 4, "opp_num_mulligans": [0] * 4,
                      "skill": [0.5] * 4, "game_number": [1] * 4, "game_time": [None] * 4, "opp_colors": ["WU"] * 4}).with_columns(
        pl.col("game_time").cast(pl.Datetime("us")))
    enc = design.CovEncoder.fit(g)
    prob, urows, inv = design.make_problem(g, enc, sp)
    assert len(urows) == 3                                          # (10, random), (10, sealed), (11, sealed)
    assert list(urows[inv]) == [0, 0, 1, 0]
    nc, nf = sp.n_card, sp.n_feat
    zr = ((p_rand - sp.mu) / sp.sd)[0]
    zs = ((p_sealed - sp.mu) / sp.sd)[0]
    rows = [prob.D[int(i)].toarray().ravel()[nc:nc + nf] for i in inv]
    assert np.allclose(rows[0], zr) and np.allclose(rows[3], zr) and np.allclose(rows[1], zs)
    assert np.allclose(rows[2], ((p_sealed - sp.mu) / sp.sd)[1])


def test_event_regime_mapping():
    assert config.event_regime("TradSealed") == "random" and config.event_regime("TradDraft") == "random"
    assert config.event_regime("Sealed") == "sealed"
    assert config.event_regime("PremierDraft") == config.event_regime("QuickDraft") == config.event_regime("PickTwoDraft") == "hand"
    assert config.format_regime("bo3") == "random" and config.format_regime("bo1") == "sealed" and config.format_regime("draft") == "hand"


# ------------------------------------------------------------------------------------------ pipeline (synthetic set)

@pytest.fixture(scope="module")
def calc_v2(calc, tmp_path_factory):
    """The synthetic set recalculated with the calibrated simulator forced on (copy of the shared home)."""
    from sealed_lab.model import pipeline

    home = tmp_path_factory.mktemp("v2_home")
    shutil.copytree(calc["home"], home, dirs_exist_ok=True)
    mp = pytest.MonkeyPatch()
    mp.setenv("SEALED_LAB_HOME", str(home))
    meta = pipeline.calculate("TST", n_jobs=2, n_boot=3, manasim="v2")
    yield dict(home=home, meta=meta)
    mp.undo()


def test_pipeline_manasim_modes_are_validated():
    from sealed_lab.model import pipeline

    with pytest.raises(ValueError):
        pipeline.calculate("TST", manasim="maybe")


def test_default_calculation_reports_the_manasim_test(calc):
    mm = calc["meta"]["manasim"]
    assert mm["mode"] == "auto" and "applied" in mm and mm["reason"]
    assert set(mm["gain"]) <= {"bo3", "bo1"} and mm["gain"]            # the hold-out test ran in at least one format
    for g in mm["gain"].values():
        assert g["lo"] <= g["gain"] <= g["hi"]


def test_forced_v2_model_uses_regime_features(calc_v2, monkeypatch):
    from sealed_lab.deck import builder
    from sealed_lab.model.artifact import SetModel

    monkeypatch.setenv("SEALED_LAB_HOME", str(calc_v2["home"]))
    m = calc_v2["meta"]["manasim"]
    assert m["applied"] and m["version"] == 2 and m["regimes"]["bo3"] == "random" and m["regimes"]["bo1"] == "sealed"
    model = SetModel.load("TST")
    assert model.sim_version == 2 and [manasim.regime_name(model.sim_spec(f)[1]) for f in ("bo3", "bo1")] == ["random", "sealed"]
    pool = {i: 2 for i in range(40) if not model.card_arrays.is_basic[i]}
    deck = {}
    for i in sorted(pool):
        if len(deck) < 22 and not model.card_arrays.is_land[i]:
            deck[i] = 1
    basics = [i for i in range(model.card_arrays.n) if model.card_arrays.is_basic[i]][:2]
    deck[basics[0]] = 9
    deck[basics[1]] = 40 - sum(deck.values())
    f3 = model.featurize(deck, "bo3")
    f1 = model.featurize(deck, "bo1")
    n = structure.N_STRUCT
    assert np.array_equal(f3[:n], f1[:n]) and not np.array_equal(f3[n:], f1[n:])
    assert np.array_equal(f3, structure.featurize(deck, model.card_arrays, version=2, regime="random"))
    for fmt in model.formats:                                       # the builder's own cached logit agrees with the model
        ctx = builder.Ctx(model, fmt)
        assert ctx.sim_version == 2 and manasim.regime_name(ctx.regime) == config.format_regime(fmt)
        assert ctx.logit(deck) == pytest.approx(model.logit(deck, fmt), abs=1e-9)
    assert (calc_v2["home"] / "sets" / "TST" / "cache" / "phi_cache_v2_random.npz").exists()
    assert "Calibrated mana simulator (v2)" in (calc_v2["home"] / "sets" / "TST" / "model" / "report.html").read_text(encoding="utf-8")


def test_realism_without_replay_data_and_v2_columns_in_the_report(calc):
    from sealed_lab.model import simcheck

    assert simcheck.realism("TST")["source"] == "none"          # the synthetic set has no replay files
    html = (calc["home"] / "sets" / "TST" / "model" / "report.html").read_text(encoding="utf-8")
    assert "Mana simulator versus observed games" in html and ">v2<" in html
