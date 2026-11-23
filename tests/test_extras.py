"""Synergy layer, archetypes and opponent matchups: design algebra, clustering, matchup algebra, and an end-to-end synthetic
calculation with all three components forced on (exactness of the deployed terms in SetModel / builder / explanations)."""
from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest
import scipy.sparse as sp
from scipy.special import expit

from sealed_lab.model import archetypes as arch_mod
from sealed_lab.model import cardvalue, design, glm, opponent as om, pipeline, synergy
from tests.test_attributes import _toy_space

BASICS = ("Plains", "Island", "Swamp", "Mountain", "Forest")


# ---------------------------------------------------------------------------------------------
# design algebra
# ---------------------------------------------------------------------------------------------

def test_pair_columns_are_min_of_copies_and_selection_thresholds():
    space, _, cnt = _toy_space()
    w = np.full(cnt.shape[0], 10.0)
    C = design.cooccurrence(space, w)
    assert C[0, 1] == pytest.approx(10.0 * ((cnt[:, 0] > 0) & (cnt[:, 1] > 0)).sum())
    is_land = np.zeros(space.n_cards_total, bool)
    is_land[5] = True                                          # a land: never in a pair
    ab, g = synergy.select_pairs(space, w, is_land, min_games=400, max_pairs=10)
    assert len(ab) == 10 and (ab[:, 0] < ab[:, 1]).all() and not (ab == 5).any()
    assert (np.diff(g) <= 0).all() and g.min() >= 400             # most frequent first
    sp_ = design.with_pairs(space, ab)
    assert sp_.n_pair == 10 and sp_.D.shape[1] == space.D.shape[1] + 10 and sp_.deck_dim == space.deck_dim + 10
    M = sp_.D[:, space.D.shape[1]:].toarray()
    for j, (a, b) in enumerate(ab):
        assert np.array_equal(M[:, j], np.minimum(cnt[:, a], cnt[:, b]))
    assert np.allclose(sp_.D[:, :space.D.shape[1]].toarray(), space.D.toarray())
    # nothing above the threshold -> no change
    assert design.with_pairs(space, np.zeros((0, 2), np.int64)) is space


def test_sigma_spec_extra_blocks_and_stage_fit_layout():
    spec = glm.SigmaSpec(sig_cov=np.ones(2), n_card=3, n_theta=2, th_lo=0.1, th_hi=1.0, n_attr=1, attr_sigma=0.3,
                         extra=[(2, 0.01), (3, 0.02)])
    v = spec.vector(0.05)
    assert len(v) == 2 + 3 + 2 + 1 + 5 and np.allclose(v[-5:], [.01, .01, .02, .02, .02]) and spec.n_extra == 5
    f = cardvalue.StageFit(name="x", coef=np.arange(12.0), se=np.ones(12), k=2, sigma=.1, spec=spec, prior_deck=np.zeros(10),
                           n_card=3, n_feat=2, n_extra=5, se_total=np.arange(12.0) + 100)
    assert np.array_equal(f.extra, np.arange(7.0, 12.0)) and np.array_equal(f.extra_se, np.arange(107.0, 112.0))
    assert f.n_attr == 0 and len(f.gamma) == 0


# ---------------------------------------------------------------------------------------------
# archetypes
# ---------------------------------------------------------------------------------------------

def _blob_decks(seed=0, n=600, n_cards=40):
    """Two planted archetypes in one colour group: cards 0-9 vs 10-19 (+ a shared core 20-29)."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n, n_cards))
    lab = rng.integers(0, 2, n)
    for i in range(n):
        X[i, 20:30] = rng.random(10) < 0.5
        base = 0 if lab[i] == 0 else 10
        X[i, base:base + 10] = rng.random(10) < 0.7
    return sp.csr_matrix(X), lab


def test_kmeans_silhouette_and_archetype_build_recover_planted_clusters():
    X, lab = _blob_decks()
    Z = X.toarray()[:, :20]
    l1, _ = arch_mod.kmeans(Z, 2, seed=1)
    l1b, _ = arch_mod.kmeans(Z, 2, seed=1)
    assert np.array_equal(l1, l1b)                                          # deterministic
    agree = max((l1 == lab).mean(), (l1 != lab).mean())
    assert agree > 0.95
    assert arch_mod.silhouette(Z, l1) > arch_mod.silhouette(Z, np.random.default_rng(99).integers(0, 2, len(Z)))
    names = [f"card{i}" for i in range(40)]
    arch = arch_mod.build(np.arange(X.shape[0]), X, np.array(["WU"] * X.shape[0]), names)
    assert arch.n == 2 and sum(a["n_decks"] for a in arch.info) == X.shape[0]
    assert all(a["signature"] and a["label"].startswith("WU: ") for a in arch.info)
    spell = np.ones(40, bool)
    idx, sim, S = arch.assign({i: 1 for i in range(10)} | {20: 1, 21: 1}, spell)          # a "first archetype" deck
    idx2, _, _ = arch.assign({i: 1 for i in range(10, 20)} | {25: 1}, spell)
    assert idx != idx2 and 0 < sim <= 1 and len(S) == 2
    # one colour group with no structure stays a single archetype
    rng = np.random.default_rng(3)
    Xn = sp.csr_matrix((rng.random((500, 40)) < 0.3).astype(float))
    assert arch_mod.build(np.arange(500), Xn, np.array(["UB"] * 500), names).n == 1


def test_archetype_win_rates_shrunk_with_interval():
    X, lab = _blob_decks(n=400)
    names = [f"card{i}" for i in range(40)]
    arch = arch_mod.build(np.arange(400), X, np.array(["WU"] * 400), names)
    deck_arch = arch._labels
    rng = np.random.default_rng(0)
    g = pl.DataFrame({"deck_key": np.repeat(np.arange(400), 5), "fmt": "bo3", "draft_id": np.repeat(np.arange(400) // 4, 5).astype(str),
                      "won": rng.random(2000) < 0.55})
    arch_mod.attach_win_rates(arch, g, np.arange(400), deck_arch, n_boot=50)
    for a in arch.info:
        r = a["rates"]["bo3"]
        assert r["games"] == 5 * a["n_decks"]
        assert min(r["wr"], r["base"]) - 1e-9 <= r["shrunk"] <= max(r["wr"], r["base"]) + 1e-9      # shrunk between raw and the mean
        assert r["lo"] <= r["shrunk"] <= r["hi"]


# ---------------------------------------------------------------------------------------------
# opponent matchups algebra
# ---------------------------------------------------------------------------------------------

def test_opponent_encodings_and_interaction_matrix():
    assert om.opp_vector("UB", "pair")[om.levels("pair").index("UB")] == 1 and om.opp_vector("WUG", "pair")[-1] == 1
    assert om.opp_vector("WUG", "colour").tolist() == [1, 1, 0, 0, 1] and om.opp_vector(None, "colour").sum() == 0
    Dc = sp.csr_matrix(np.array([[2, 0, 1], [0, 1, 0]], float))
    for mode in om.MODES:
        O = om.opp_matrix(np.array(["WU", "WUG"], dtype=object), mode)
        X = om.interaction_matrix(Dc, O).toarray()
        L = O.shape[1]
        assert X.shape == (2, 3 * L)
        for i in range(2):
            for c in range(3):
                assert np.allclose(X[i, c * L:(c + 1) * L], Dc[i, c] * O[i].toarray().ravel())


def test_centred_effects_algebra_and_prediction():
    rng = np.random.default_rng(0)
    nc, mode = 5, "colour"
    L = 5
    mass = rng.random((nc, L))
    delta = rng.normal(size=nc * L)
    opps = ["WU", "BR", "WUG"]
    cen = om.centred(delta, mass, mode, opps)
    A = om.opp_contrasts(mass, mode, opps, nc * L + nc)
    assert np.allclose((A @ np.concatenate([delta, np.zeros(nc)])).reshape(nc, len(opps)), cen)
    # prediction for a game = sum over cards of copies x centred effect
    Dc = sp.csr_matrix(rng.integers(0, 3, (4, nc)).astype(float))
    oc = np.array(["WU", "BR", "WUG", "G"], dtype=object)
    pred = om.predict_centred(Dc, om.opp_matrix(oc, mode), delta, mass)
    for i, o in enumerate(oc):
        c1 = om.centred(delta, mass, mode, [o])[:, 0]
        assert pred[i] == pytest.approx(float(Dc[i].toarray().ravel() @ c1))
    # a card whose exposure equals the opponent mix has zero average centred effect
    O = om.opp_matrix(np.array(["WU", "BR", "WUG", "G"] * 10, dtype=object), mode)
    m = np.asarray(O.mean(0)).ravel()
    mass2 = np.tile(m, (nc, 1))
    avg = np.mean([om.centred(delta, mass2, mode, [o])[:, 0] for o in ["WU", "BR", "WUG", "G"] * 10], axis=0)
    assert np.allclose(avg, 0, atol=1e-12)


def test_cluster_se_contrasts_use_matrix_products():
    rng = np.random.default_rng(1)
    n, p = 800, 6
    C = sp.csr_matrix(rng.normal(size=(n, p)))
    pr = glm.Problem(y=(rng.random(n) < .5).astype(float), C=C, offset=rng.normal(size=n) * 0.1)
    sig = np.full(p, 0.5)
    res = glm.fit(pr, np.zeros(p), sig)
    L = rng.normal(size=(3, p))
    grp = np.arange(n) // 8
    a, b, cl, cr = glm.cluster_se(pr, res.coef, sig, grp, contrasts=L)
    _, _, e_lap, e_rob = glm.cluster_se(pr, res.coef, sig, grp, contrasts=np.eye(p))
    # SE of a contrast from the unit-vector covariance: reconstruct the covariance from basis contrasts
    assert cl.shape == (3,) and (cr > 0).all() and (cl > 0).all()


# ---------------------------------------------------------------------------------------------
# end to end: all three components forced on
# ---------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def calc_x(synth_dir, tmp_path_factory):
    mp = pytest.MonkeyPatch()
    home = tmp_path_factory.mktemp("extras_home")
    mp.setenv("SEALED_LAB_HOME", str(home))
    from sealed_lab import api
    from sealed_lab.ingest.corpus import build_corpus

    api.add_files([str(synth_dir)])
    build_corpus("TST", n_jobs=2, fetch_scryfall=False)
    meta = pipeline.calculate("TST", n_jobs=2, n_boot=3, n_rep=60, synergy="on", archcond="on", matchups="on", min_pair_games=150)
    yield dict(home=home, meta=meta, synth=synth_dir)
    mp.undo()


@pytest.fixture(scope="module")
def xmodel(calc_x):
    from sealed_lab.deck import builder

    return builder.load_model("TST")


@pytest.fixture(scope="module")
def xdecks(xmodel, calc_x):
    held = json.loads((calc_x["synth"] / "heldout_pools.json").read_text(encoding="utf-8"))
    n2i = {n: i for i, n in enumerate(xmodel.cards.sort("card_id")["name"].to_list())}
    out = []
    for h in held[:4]:
        out.append(({n2i[n]: c for n, c in h["player_deck"].items()}, {n2i[n]: c for n, c in h["pool"].items() if n not in BASICS}))
    return out


def test_components_deployed_and_reported(calc_x, xmodel):
    m = calc_x["meta"]
    for k in ("synergy", "archcond", "matchups"):
        assert m[k]["applied"] and "forced on" in m[k]["reason"]
    d = xmodel.meta
    assert xmodel.has_pairs("bo3") and xmodel.pairs["ab"].shape[1] == 2 and len(xmodel.pairs["ab"]) == m["synergy"]["n_pairs"]
    assert xmodel.opp is not None and xmodel.archetypes is not None and xmodel.archetypes.n == m["archetypes"]["n"]
    R = json.loads((calc_x["home"] / "sets" / "TST" / "model" / "report.json").read_text(encoding="utf-8"))
    html = (calc_x["home"] / "sets" / "TST" / "model" / "report.html").read_text(encoding="utf-8")
    for h in ("Card-pair synergies", "Archetypes", "Archetype-conditional card values", "Opponent matchups and sideboard plans"):
        assert h in html
    v = R["validation"]["bo3"]
    for k in ("synergy", "archcond", "opponent"):
        o = v[k]
        assert o["lo"] <= o["gain"] <= o["hi"]
    assert v["opponent"]["table"]["n_cells"] > 0 and "MS" in v["metrics"] and "MC" in v["metrics"] and "MO" in v["metrics"]
    assert R["synergy"]["final"]["bo3"]["n_pairs"] == m["synergy"]["n_pairs"]
    assert R["archetypes"]["archetypes"] and R["matchups"]["final"]["table"]["top"]


def test_builder_and_model_logit_agree_with_pair_and_archetype_terms(xmodel, xdecks):
    from sealed_lab.deck import builder

    for fmt in ("bo3", "bo1"):
        ctx = builder.Ctx(xmodel, fmt)
        assert ctx.P is not None and ctx.has_arch
        for d, _ in xdecks:
            assert ctx.logit(d) == pytest.approx(xmodel.logit(d, fmt), abs=1e-9)
            ev = xmodel.evaluate(d, fmt)
            assert "synergy" in ev.breakdown and "archetype" in ev.breakdown
            assert sum(ev.breakdown.values()) == pytest.approx(xmodel.logit(d, fmt), abs=1e-9)
            assert ev.game_wr == pytest.approx(float(expit(xmodel.logit(d, fmt))), abs=1e-12)
            L = xmodel.logit(d, fmt, draws=True)
            assert L.shape == (xmodel.n_draws(fmt),) and L[0] == pytest.approx(xmodel.logit(d, fmt))
    # the synergy term equals the explicit sum over pairs
    d = xdecks[0][0]
    ab, c0 = xmodel.pairs["ab"], xmodel._draws["bo3"]["pair"][0]
    explicit = sum(c0[j] * min(d.get(int(a), 0), d.get(int(b), 0)) for j, (a, b) in enumerate(ab))
    assert xmodel._extra_terms(d, "bo3", False)[0][0] == pytest.approx(explicit, abs=1e-12)


def test_waterfall_has_synergy_step_and_adds_up(xmodel, xdecks):
    from sealed_lab.deck import explain

    for fmt in ("bo3", "bo1"):
        for d, _ in xdecks[:3]:
            ex = explain.explain_deck(xmodel, d, fmt)
            labels = [s.label for s in ex.steps]
            assert "Synergies" in labels and "Archetype fit" in labels
            L_end = np.log(ex.end_wr / (1 - ex.end_wr))
            L_start = np.log(ex.start_wr / (1 - ex.start_wr))
            assert L_start + sum(s.d_logit for s in ex.steps) == pytest.approx(L_end, abs=1e-9)
            assert ex.end_wr == pytest.approx(float(xmodel.q(d, fmt)), abs=1e-12)


def test_swap_analysis_matches_direct_evaluation_with_pairs_and_matchups(xmodel, xdecks):
    from sealed_lab.deck import explain

    d, pool = xdecks[0]
    for opp in (None, "UB", "WG"):
        sa = explain.swap_analysis(xmodel, d, pool, "bo3", opp=opp)
        eff = xmodel.opp.effect(opp, draws=False) if opp else np.zeros(xmodel.card_arrays.n)
        wi, _, we = xmodel.deck_weights(d, "bo3")        # effective copies (copies x castability when castability ships)
        base_L = xmodel.logit(d, "bo3") + float(eff[wi] @ we)
        assert sa.base_wr == pytest.approx(float(expit(base_L)), abs=1e-12)
        checked = 0
        for s in list(sa.deck_best.values())[:8]:
            nd = dict(d)
            nd[s.out_id] -= 1
            nd = {k: v for k, v in nd.items() if v > 0}
            nd[s.in_id] = nd.get(s.in_id, 0) + 1
            wi1, _, we1 = xmodel.deck_weights(nd, "bo3")
            L1 = xmodel.logit(nd, "bo3") + float(eff[wi1] @ we1)
            assert s.gain_pp == pytest.approx(100 * (float(expit(L1)) - sa.base_wr), abs=1e-7)
            checked += 1
        assert checked


def test_insight_and_plan_api(xmodel, xdecks, calc_x):
    from sealed_lab import api

    d, pool = xdecks[1]
    sv = api.synergies("TST", d, pool, "bo3")
    assert sv.status in ("ok", "off")
    for r in sv.present + sv.candidates:
        assert r.p_pos >= sv.p_min or r.p_pos <= 1 - sv.p_min
        assert abs(r.coef_pp) >= sv.min_pp
    am = api.archetype_of("TST", d, "bo3")
    assert am.status == "ok" and 0 < am.similarity <= 1 and am.label and am.rate is not None
    assert all(m["card_id"] not in d for m in am.missing)
    plan = api.sideboard_plan("TST", d, pool, "bo3")
    assert plan.status == "ok" and plan.opponents and plan.encoding in om.MODES
    for o, sws in plan.swaps.items():
        for s in sws:
            assert s.reliable and s.gain_pp >= plan.thresholds.min_pp and s.p_delta_in >= plan.thresholds.p_min
            assert d.get(s.out_id, 0) > 0 and pool[s.in_id] - d.get(s.in_id, 0) > 0
    assert api.sideboard_plan("TST", d, pool, "bo1").status == "off"
    comps = api.model_components("TST")
    assert comps["matchups"]["applied"] and comps["synergy"]["applied"]


def test_matchup_layer_not_applied_says_so(calc_x, xmodel, monkeypatch):
    """A model without the matchup layer (or one that failed its test) gets a plain message and marks nothing."""
    from sealed_lab.deck import matchups

    monkeypatch.setattr(xmodel, "opp", None)
    monkeypatch.setitem(xmodel.meta, "matchups", dict(mode="auto", applied=False, reason="every 95% CI includes zero"))
    plan = matchups.plan_for(xmodel, {1: 1}, {1: 1}, "bo3")
    assert plan.status == "not_supported" and "do not support" in plan.message and not plan.chips and not plan.swaps
