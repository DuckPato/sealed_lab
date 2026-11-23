"""model/playdraw.py + deck/playdraw.py: the card x play/draw layer (docs/playdraw/RULE.md).

Unit checks of the s coding and the deployed container, then an end-to-end run on the synthetic set (the generator plants only a
main on-play effect, so the layer is forced on): it is tested on the hold-out and reported, its gamma draws are stored, the games-2-3
plan runs, and q(D) is identical to a calculation with the layer off."""
from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest
import scipy.sparse as sp

from sealed_lab.model import pipeline
from sealed_lab.model import playdraw as pdm

BASICS = {"Plains", "Island", "Swamp", "Mountain", "Forest"}


def test_s_vector_codes_play_draw_and_missing():
    s, known = pdm.s_vector(pl.Series([True, False, None, True], dtype=pl.Boolean))
    assert s.tolist() == [0.5, -0.5, 0.0, 0.5] and known.tolist() == [True, True, False, True]


def test_predict_extra_is_zero_without_on_play_and_antisymmetric():
    Dc = sp.csr_matrix(np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]))
    e = pdm.predict_extra(Dc, np.array([0.5, -0.5, 0.0]), np.array([0.3, 0.0]))
    assert e[0] == pytest.approx(0.15) and e[1] == pytest.approx(-0.15) and e[2] == 0.0


def test_playdraw_model_effect_is_plus_minus_half_gamma():
    g = np.random.default_rng(0).normal(size=(5, 7))
    m = pdm.PlayDrawModel(gamma=g, sigma=0.02)
    assert np.allclose(m.effect(True), g / 2) and np.allclose(m.effect(False), -g / 2)
    assert np.allclose(m.effect(True, draws=False), g[0] / 2) and m.effect(True).shape == (5, 7)


# ---------------------------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def run_on(synth_dir, tmp_path_factory):
    """Corpus + model with the play/draw layer forced on; yields what the 'off' run later compares with."""
    mp = pytest.MonkeyPatch()
    home = tmp_path_factory.mktemp("playdraw_home")
    mp.setenv("SEALED_LAB_HOME", str(home))
    from sealed_lab import api
    from sealed_lab.deck import builder
    from sealed_lab.ingest.corpus import build_corpus

    api.add_files([str(synth_dir)])
    build_corpus("TST", n_jobs=2, fetch_scryfall=False)
    meta = pipeline.calculate("TST", n_jobs=2, n_boot=3, n_rep=60, playdraw="on", matchups="off", synergy="off", archcond="off")
    model = builder.load_model("TST")
    held = json.loads((synth_dir / "heldout_pools.json").read_text(encoding="utf-8"))
    n2i = {n: i for i, n in enumerate(model.cards.sort("card_id")["name"].to_list())}
    decks = [({n2i[n]: c for n, c in h["player_deck"].items()}, {n2i[n]: c for n, c in h["pool"].items() if n not in BASICS})
             for h in held[:3]]
    q_on = {f: [model.logit(d, f) for d, _ in decks] for f in model.formats}
    yield dict(home=home, meta=meta, model=model, decks=decks, q_on=q_on, synth=synth_dir)
    mp.undo()


def test_component_runs_decides_and_stores_draws(run_on):
    m, model = run_on["meta"], run_on["model"]
    assert m["playdraw"]["applied"] and "forced on" in m["playdraw"]["reason"] and m["playdraw"]["sigma"] > 0
    assert model.playdraw is not None and model.has_playdraw("bo3")
    n_cards = len(model.cards)
    assert model.playdraw.gamma.shape == (m["n_draws"], n_cards)
    assert np.isfinite(model.playdraw.gamma).all() and np.abs(model.playdraw.gamma).max() > 0
    for fmt in model.formats:
        e_p, e_d = model.playdraw_effect(True), model.playdraw_effect(False)
        assert e_p.shape == (m["n_draws"], n_cards) and np.allclose(e_p, -e_d)
    assert model.playdraw_effect(True, draws=False).shape == (n_cards,)
    assert (run_on["home"] / "sets" / "TST" / "model" / "playdraw.npz").exists()


def test_holdout_test_reported(run_on):
    R = json.loads((run_on["home"] / "sets" / "TST" / "model" / "report.json").read_text(encoding="utf-8"))
    html = (run_on["home"] / "sets" / "TST" / "model" / "report.html").read_text(encoding="utf-8")
    assert "Play/draw card effects" in html
    tested = 0
    for f in R["formats"]:
        v = R["validation"][f]
        if v.get("status") != "ok":
            continue
        o = v["playdraw"]
        tested += 1
        assert o["lo"] <= o["gain"] <= o["hi"] and "MP" in v["metrics"] and o["sigma"] > 0
        assert o["halves"]["A"] and o["halves"]["B"] and o["table"]["n_cards"] > 0
    assert tested and R["playdraw"]["applied"] and "final" in R["playdraw"]
    # (the confirmation table lists it)
    assert "Play/draw card effects" in html.split("Confirmation split")[1]


def test_q_is_identical_with_the_layer_off(run_on):
    """The layer is centred (s = +-1/2): q(D) and the stored chain draws must not move when it is switched off."""
    from sealed_lab.deck import builder

    model_on = run_on["model"]
    beta_on = {f: model_on._draws[f]["beta"].copy() for f in model_on.formats}
    meta_off = pipeline.calculate("TST", n_jobs=2, n_boot=3, n_rep=60, playdraw="off", matchups="off", synergy="off", archcond="off")
    assert not meta_off["playdraw"]["applied"] and meta_off["playdraw"]["reason"] == "switched off by the user"
    model_off = builder.load_model("TST")
    assert model_off.playdraw is None
    assert not (run_on["home"] / "sets" / "TST" / "model" / "playdraw.npz").exists()
    for f in model_off.formats:
        assert np.allclose(model_off._draws[f]["beta"], beta_on[f], atol=1e-9)
        got = [model_off.logit(d, f) for d, _ in run_on["decks"]]
        assert got == pytest.approx(run_on["q_on"][f], abs=1e-9)
        # and the 'on' model's own q does not use the layer: dropping it changes nothing
        saved, model_on.playdraw = model_on.playdraw, None
        try:
            assert [model_on.logit(d, f) for d, _ in run_on["decks"]] == run_on["q_on"][f]
        finally:
            model_on.playdraw = saved


def test_plan_gates_and_status(run_on):
    from sealed_lab.deck import playdraw as plan_mod

    model = run_on["model"]
    deck, pool = run_on["decks"][0]
    plan = plan_mod.playdraw_plan(model, deck, pool, "bo1")
    assert plan.status == "off"
    plan = plan_mod.playdraw_plan(model, deck, pool, "bo3")
    assert plan.status == "ok" and set(plan.base_wr) == {"play", "draw"}
    for key, swaps in plan.swaps.items():
        for s in swaps:
            assert s.gain_pp >= plan_mod.om.OPP_MIN_PP and (s.p_better is None or s.p_better >= plan_mod.om.OPP_P_MIN)
            assert s.on_play == (key == "play")
    # the plan is derived from play/draw shifts: with the layer removed the model says so
    saved, model.playdraw = model.playdraw, None
    try:
        assert plan_mod.playdraw_plan(model, deck, pool, "bo3").status == "not_supported"
    finally:
        model.playdraw = saved
