"""The Set report, phase 2 (sealed_lab.report_arch): sub-archetypes with the replication rule, names, deck-shape ranges, the sections."""
from __future__ import annotations

import json

import numpy as np
import pytest
import scipy.sparse as sp

from sealed_lab import report as R
from sealed_lab import report_arch as A


def synth_decks(kind: str, n: int = 1200, seed: int = 0):
    """(binary card matrix, style matrix, event code of each deck, true group) for synthetic decks of ONE color pair."""
    rng = np.random.default_rng(seed)
    X = rng.random((n, 80)) < 0.15                                     # shared / random cards
    S = rng.normal(size=(n, len(A.STYLE_NAMES)))
    grp = np.zeros(n, dtype=int)
    if kind == "two":                                                  # two different cores (cards 40-54 / 55-69) and different styles
        grp = rng.integers(0, 2, n)
        for j, (lo, hi) in enumerate(((40, 55), (55, 70))):
            m = grp == j
            X[np.ix_(m, np.arange(lo, hi))] |= rng.random((m.sum(), hi - lo)) < 0.8
        S[:, 0] += 2.5 * grp
        S[:, 5] -= 2.0 * grp
    elif kind == "continuum":                                          # ONE smooth axis drives many cards and the style
        z = rng.normal(size=n)
        a = rng.choice([-1.0, 1.0], size=40) * rng.uniform(0.5, 1.2, size=40)
        X[:, 40:] = rng.random((n, 40)) < 1 / (1 + np.exp(-(a[None, :] * z[:, None] - 1.0)))
        S[:, 0] += 1.0 * z
        S[:, 5] -= 1.0 * z
    return sp.csr_matrix(X.astype(float)), S, np.arange(n) // 3, grp


# --------------------------------------------------------------------------- clustering with replication

def test_adjusted_rand_index():
    a = np.array([0, 0, 1, 1, 2, 2] * 20)
    assert A.adjusted_rand(a, (a + 1) % 3) == pytest.approx(1.0)                      # same grouping, other names
    rng = np.random.default_rng(0)
    assert abs(A.adjusted_rand(rng.integers(0, 3, 3000), rng.integers(0, 3, 3000))) < 0.05
    assert A.adjusted_rand(a, np.zeros_like(a)) == pytest.approx(0.0, abs=1e-9)


def test_two_clear_cores_are_split_and_replicate():
    Xb, S, ev, truth = synth_decks("two")
    res = A.cluster_pair(Xb, S, ev, seed=1)
    assert res["k"] == 2 and res["ari"] >= A.ARI_MIN
    assert A.adjusted_rand(res["labels"], truth) > 0.8                                # the split found is the planted one
    sizes = np.bincount(res["labels"])
    assert sizes.min() >= A.MIN_CLUSTER_DECKS and sizes[0] >= sizes[1]               # biggest cluster first
    t = res["tried"][0]
    assert t["k"] == 2 and t["accepted"] and t["replicates"] and t["distinct"]


def test_one_homogeneous_group_is_not_split():
    Xb, S, ev, _ = synth_decks("homogeneous")
    res = A.cluster_pair(Xb, S, ev, seed=1)
    assert res["k"] == 1 and (res["labels"] == 0).all()
    # even without the "sharper than a smooth cloud" guard, the halves of pure noise do not agree
    res2 = A.cluster_pair(Xb, S, ev, seed=1, require_distinct=False)
    assert res2["k"] == 1 and all(not t["accepted"] for t in res2["tried"])
    assert all(t["ari"] < A.ARI_MIN for t in res2["tried"])


def test_smooth_cloud_and_too_small_clusters_are_rejected():
    Xb, S, ev, _ = synth_decks("continuum")
    lenient = A.cluster_pair(Xb, S, ev, seed=1, require_distinct=False)
    strict = A.cluster_pair(Xb, S, ev, seed=1)
    assert strict["k"] == 1                                                            # a smooth axis is not an archetype
    assert lenient["k"] >= 2 and lenient["ari"] >= A.ARI_MIN                          # (it does replicate: that is why the extra guard exists)
    assert A.cluster_pair(Xb[:200], S[:200], ev[:200])["k"] == 1                      # too small a pair is never split
    Xs, Ss, evs, _ = synth_decks("two", n=1200, seed=3)
    assert A.cluster_pair(Xs, Ss, evs, seed=1, min_decks=700)["k"] == 1               # clusters below the size rules are refused
    assert A.cluster_pair(Xs, Ss, evs, seed=1, min_share=0.6)["k"] == 1


def test_cluster_pair_is_deterministic():
    Xb, S, ev, _ = synth_decks("two", n=800)
    a, b = A.cluster_pair(Xb, S, ev, seed=5), A.cluster_pair(Xb, S, ev, seed=5)
    assert a["k"] == b["k"] and (a["labels"] == b["labels"]).all() and a["ari"] == b["ari"]


# --------------------------------------------------------------------------- names, shape statistics, ranges

def test_style_words_and_names():
    base = dict(dmv=0.0, dcre=0.0, drem=0.0, ddraw=0.0, dfly=0.0, dcheap=0.0)
    w, name = A.style_words(dict(base, dmv=-0.2, dcre=0.04))
    assert name == "aggro" and "lower curve" in w and "more creatures" in w
    w, name = A.style_words(dict(base, dmv=0.2, dcre=-0.05, drem=0.6, ddraw=0.4))
    assert name == "control" and "higher curve" in w and "more removal" in w and "more card draw" in w
    assert A.style_words(base) == ([], "")
    assert A.style_words(dict(base, dmv=0.2, dcre=0.05))[1] == ""                      # higher curve AND more creatures, no removal: not "control"


def card_vectors() -> A.CardVectors:
    # ids: 0 Plains, 1 two-drop creature (W), 2 four-drop creature (W), 3 removal spell (W), 4 off-color splash spell (R), 5 five-drop (WU)
    z = np.zeros(6)
    return A.CardVectors(mv=np.array([0, 2, 4, 2, 3, 5.0]), land=np.array([1, 0, 0, 0, 0, 0], dtype=bool),
                         creature=np.array([0, 1, 1, 0, 0, 0], dtype=bool), cmask=np.array([0, 1, 1, 1, 8, 3]), removal=np.array([0, 0, 0, 1, 0, 0.0]),
                         draw=z.copy(), flying=z.copy(), names=[f"c{i}" for i in range(6)], basic=np.array([1, 0, 0, 0, 0, 0], dtype=bool))


def test_deck_shape_counts():
    cv = card_vectors()
    X = sp.csr_matrix(np.array([[17, 4, 4, 5, 0, 10], [16, 8, 0, 4, 2, 10.0]]))          # copies per deck (rows add to 40)
    sh = A.deck_shape(X, cv, np.array([1 | 2, 1 | 2]))                                    # both decks are white-blue
    assert sh["lands"].tolist() == [17, 16] and sh["spells"].tolist() == [23, 24]
    assert sh["creatures"].tolist() == [8, 8] and sh["two_drops"].tolist() == [4, 8]
    assert sh["splash_cards"].tolist() == [0, 2]                                          # the red card is a splash card
    assert sh["avg_mv"][0] == pytest.approx((4 * 2 + 4 * 4 + 5 * 2 + 0 + 10 * 5) / 23)
    S = A.style_matrix(sh)
    assert S.shape == (2, len(A.STYLE_NAMES)) and S[0, 5] == 5                            # removal count


def test_shape_ranges_and_strong_results():
    rng = np.random.default_rng(0)
    n = 400
    sh = {k: rng.integers(14, 19, n).astype(float) for k, _ in A.SHAPE_METRICS}
    sh["lands"] = rng.choice([16.0, 17.0], n)
    strong = np.arange(n) < 120
    sh["lands"][strong] = 17.0
    row = A.shape_rows(sh, np.ones(n, dtype=bool), strong)
    r = row["metrics"]["lands"]
    assert row["n_decks"] == n and row["n_strong"] == 120
    assert r["all"]["lo"] == 16.0 and r["all"]["hi"] == 17.0 and r["strong"] == dict(lo=17.0, med=17.0, hi=17.0)
    c = row["metrics"]["creatures"]["all"]
    assert c["lo"] <= c["med"] <= c["hi"]
    few = A.shape_rows(sh, np.ones(n, dtype=bool), np.arange(n) < 10)                    # fewer than STRONG_MIN_DECKS strong decks: no range
    assert few["metrics"]["lands"]["strong"] is None and few["metrics"]["lands"]["all"] is not None
    assert A.range_of(np.array([])) is None
    assert A._fmt_range(dict(lo=16.0, med=17.0, hi=17.0), 0) == "16-17" and A._fmt_range(dict(lo=17.2, med=17, hi=17.3), 0) == "17"


# --------------------------------------------------------------------------- on the synthetic set

@pytest.mark.parametrize("fmt", ["bo1", "bo3"])
def test_archetype_and_shape_sections_on_the_synthetic_set(calc, model, fmt):
    rep = R.compute_report("TST", fmt, model=model)
    json.dumps(rep)
    ar, sh = rep["archetypes"], rep["shape"]
    assert "same in every archetype" in ar["note"] and "handles curve and land count through the model" in sh["note"]
    assert sh["rows"][0]["key"] == "all" and sh["rows"][0]["n_decks"] > 100
    m = sh["rows"][0]["metrics"]
    assert set(m) == {k for k, _ in A.SHAPE_METRICS} and 14 <= m["lands"]["all"]["med"] <= 19 and m["lands"]["all"]["lo"] <= m["lands"]["all"]["hi"]
    assert rep["notes"]["archetypes"] and rep["notes"]["shape"]
    pr = {r["key"]: r for r in rep["pairs"]["rows"]}
    for a in ar["pairs"]:
        assert a["code"] == A.code_of(a["key"], a["splash_cols"]) and a["splashy"] == bool(a["splash_cols"])
        assert a["spectrum"]["shown"] in (True, False) and (not a["spectrum"]["shown"] or a["spectrum"]["text"])
        assert a["share"] == pytest.approx(pr[a["key"]]["share"]) and a["badge"] in ("strong", "likely", "weak") and a["text"]
        assert a["shape"]["lands"] > 10 and a["split"]["k"] >= 1
        for c in a["signature"]:
            assert c["rate"] >= A.SIG_MIN_RATE and c["rate"] > c["ref_rate"]
        for c in a["best_cards"]:
            assert c["value_pp"] > 0 and c["badge"] in ("strong", "likely", "weak")
        for key in ("best", "worst"):
            if key in a["matchups"]:
                assert a["matchups"][key]["badge"] != "weak" and a["matchups"][key]["vs"] != a["key"]
    assert ar["n_with_subs"] == sum(1 for a in ar["pairs"] if a["subs"])


def test_decks_and_archetype_keys_on_the_synthetic_set(calc, model):
    g = R.load_games("TST", "bo1")
    cards = model.cards.sort("card_id")
    decks, X = A.deck_table("TST", g, cards)
    cv = A.CardVectors.from_cards(cards)
    assert decks.height == X.shape[0] and decks["grp"].is_not_null().all()
    sh = A.deck_shape(X, cv, decks["mmask"].to_numpy())
    assert np.allclose(sh["lands"] + sh["spells"], 40) and (sh["avg_mv"] > 0).all()
    bld = A.build_archetypes(decks, X, cv)
    assert len(bld["akey"]) == decks.height and set(bld["pairs"]) <= set(A.design.PAIRS)
    strong = A.strong_mask(g, decks, "bo1")
    assert strong.shape == (decks.height,) and (strong == (decks["wins"].to_numpy() >= 5)).all()
    assert A.strong_mask(R.load_games("TST", "bo3"), A.deck_table("TST", R.load_games("TST", "bo3"), cards)[0], "bo3").dtype == bool


def test_effects_for_planted_archetypes(calc, model):
    """`fit_effects` on the synthetic games with an arbitrary two-way split of one pair: runs, and the sub keys carry an in-pair effect."""
    g = R.load_games("TST", "bo1")
    cards = model.cards.sort("card_id")
    decks, _X = A.deck_table("TST", g, cards)
    big = decks["grp"].value_counts().sort("count", descending=True)["grp"][0]
    rng = np.random.default_rng(0)
    half = rng.random(decks.height) < 0.5
    akey = np.where(decks["grp"].to_numpy() == big, np.where(half, f"{big}#1", f"{big}#2"), decks["grp"].to_numpy()).astype(str)
    gk = g.join(decks.select("draft_id", "deck_key", __import__("polars").Series("akey", akey)), on=["draft_id", "deck_key"], how="inner")
    pair_of = {k: (k.split("#")[0]) for k in set(akey.tolist())}
    eff = A.fit_effects(gk, gk["akey"].to_numpy(), pair_of)
    assert f"{big}#1" in eff and "in_pair_eff" in eff[f"{big}#1"] and eff[f"{big}#1"]["se"] > 0
    assert abs(eff[f"{big}#1"]["in_pair_eff"]) < 10 and all("in_pair_eff" not in e for k, e in eff.items() if "#" not in k)


# --------------------------------------------------------------------------- splash label, style spectrum

def test_splash_label_names_the_specific_colours():
    assert A.splash_colors_of("UB", {"R": 30, "G": 26, "W": 10}, 100) == ["R", "G"]
    assert A.code_of("UB", ["R", "G"]) == "UB+r/g" and A.code_of("UB", ["R"]) == "UB+r" and A.code_of("UB", []) == "UB"
    assert A.splash_colors_of("UB", {"R": 24, "G": 20}, 100) == []                      # below a quarter of the pair's decks: just the pair
    assert A.splash_colors_of("UB", {"R": 25}, 100) == ["R"]                             # exactly a quarter counts
    assert A.splash_colors_of("UB", {"U": 90, "B": 80}, 100) == []                       # the pair's own colors are never splashes
    assert A.splash_colors_of("WG", {"U": 40, "R": 30, "B": 26}, 100) == ["U", "B", "R"]  # WUBRG order
    assert A.splash_colors_of("WG", {"U": 40}, 0) == []


def test_style_axis_replicates_on_a_continuum_and_not_on_noise():
    Xb, S, ev, _ = synth_decks("continuum")
    ax = A.spectrum_axis(Xb, S, ev, seed=1)
    assert ax["ok"] and ax["corr"] >= A.AXIS_MIN_CORR and len(ax["corrs"]) == A.N_SPLITS and ax["half_a"].shape == (Xb.shape[0],)
    Xn, Sn, evn, _ = synth_decks("homogeneous")
    assert not A.spectrum_axis(Xn, Sn, evn, seed=1)["ok"]                                # pure noise: the two halves find unrelated axes
    assert A.spectrum_axis(Xb[:200], S[:200], ev[:200]) is None                         # too few decks
    a2 = A.spectrum_axis(Xb, S, ev, seed=1)
    assert a2["corr"] == ax["corr"] and np.allclose(a2["score"], ax["score"])           # deterministic


def test_end_descriptions_name_only_clear_differences():
    rng = np.random.default_rng(0)
    n = 900
    z = rng.normal(size=n)
    sh = dict(spells=np.full(n, 23.0), avg_mv=3.0 + 0.4 * z + 0.05 * rng.normal(size=n), creatures=np.round(14 + rng.normal(size=n)),
              removal=np.clip(np.round(3 - 1.2 * z + 0.3 * rng.normal(size=n)), 0, None), draw=np.clip(np.round(1 + 0.05 * rng.normal(size=n)), 0, None),
              flying=np.clip(np.round(2 + 1.5 * z), 0, None), two_drops=np.round(4 + 0.1 * rng.normal(size=n)))
    lo, hi = z <= np.quantile(z, 1 / 3), z >= np.quantile(z, 2 / 3)
    a, b, d = A.describe_ends(sh, lo, hi, np.ones(n, dtype=bool))
    assert "lower curve" in a["words"] and "more removal" in a["words"] and "higher curve" in b["words"] and "more fliers" in b["words"]
    assert not any("draw" in w or "2-drops" in w or "creatures" in w for w in a["words"] + b["words"])      # no clear difference: not named
    assert len(a["label"].split(" / ")) <= 2 and "aggro" not in (a["description"] + b["description"]).lower()
    assert a["description"] and d["avg_mv"] > 0.3


def test_verdict_needs_a_clear_badge_and_the_same_sign_in_both_halves():
    full = dict(eff=3.0, se=1.2, z=2.5)
    assert A.spectrum_verdict(full, [dict(eff=2.0), dict(eff=4.0)]) == dict(finding=True, replicated=True)
    assert A.spectrum_verdict(full, [dict(eff=2.0), dict(eff=-0.5)]) == dict(finding=False, replicated=False)    # a half disagrees
    assert A.spectrum_verdict(full, [dict(eff=2.0), None])["finding"] is False                                  # a half cannot be fitted
    assert A.spectrum_verdict(dict(eff=1.0, se=1.2, z=0.8), [dict(eff=1.0), dict(eff=1.0)]) == dict(finding=False, replicated=True)   # weak badge
    assert A.spectrum_verdict(None, [])["finding"] is False
    a = dict(label="low curve / removal", description="lower curve and more removal")
    b = dict(label="fliers / card draw", description="more fliers and more card draw")
    win = dict(finding=True, effect_pp=-2.6, badge="likely")
    t = A.spectrum_texts("UB", a, b, win)
    assert t[0] == "UB decks range from lower curve and more removal to more fliers and more card draw (a spectrum, not separate archetypes)."
    assert "low curve / removal end win 2.6 points more often than decks at the fliers / card draw end" in t[1]
    assert A.spectrum_texts("UB", a, b, dict(win, finding=False))[1] == "No clear difference in win rate between the two styles."


def _planted(model, boost: float, seed: int = 0):
    """Synthetic games with a random 3-level axis over the decks; `boost` = share of the top level's lost games turned into wins."""
    import polars as pl

    g = R.load_games("TST", "bo1")
    decks, _X = A.deck_table("TST", g, model.cards.sort("card_id"))
    rng = np.random.default_rng(seed)
    lvl = rng.integers(0, 3, decks.height)
    top = decks.select("draft_id", "deck_key").with_columns(pl.Series("lvl", lvl))
    gk = g.join(top, on=["draft_id", "deck_key"], how="inner")
    won = gk["won"].to_numpy().copy()
    flip = (gk["lvl"].to_numpy() == 2) & ~won & (rng.random(gk.height) < boost)
    gk = gk.with_columns(pl.Series("won", won | flip), pl.Series("akey", [("a", "b", "c")[v] for v in gk["lvl"].to_list()]))
    ev = gk["draft_id"].unique().to_list()
    return gk, set(ev[: len(ev) // 2])


def test_win_difference_between_the_ends_with_a_planted_effect(calc, model):
    gk, half = _planted(model, boost=0.35)
    res = A.pair_win_difference(gk, half)
    assert all(h is not None and h["eff"] > 3.0 for h in res["halves"])                  # a strong planted effect shows in BOTH halves
    eff = A.fit_effects(gk, gk["akey"].to_numpy(), {k: "p" for k in "abc"}, diffs=[("c", "a")])
    d = eff["__diffs__"]["c|a"]
    assert d["eff"] > 3.0 and d["z"] > 2.0
    assert A.spectrum_verdict(d, res["halves"])["finding"] is True


def test_no_win_difference_without_a_planted_effect(calc, model):
    gk, half = _planted(model, boost=0.0, seed=2)
    res = A.pair_win_difference(gk, half)
    d = A.fit_effects(gk, gk["akey"].to_numpy(), {k: "p" for k in "abc"}, diffs=[("c", "a")])["__diffs__"]["c|a"]
    assert abs(d["eff"]) < 3.0
    assert A.spectrum_verdict(d, res["halves"])["finding"] is False
    assert A.pair_win_difference(gk.head(300), half)["halves"] == [None, None]           # too few games to fit a half


def test_spectrum_block_assembles_finding_no_difference_and_hidden_cases():
    rng = np.random.default_rng(1)
    n = 900
    z = rng.normal(size=n)
    lo = z <= np.quantile(z, 1 / 3)
    sh = dict(spells=np.full(n, 23.0), avg_mv=3.0 + 0.4 * z, creatures=np.round(14 + 0 * z), removal=np.clip(np.round(3 - 1.2 * z), 0, None),
              draw=np.zeros(n), flying=np.clip(np.round(2 + 1.5 * z), 0, None), two_drops=np.full(n, 4.0))
    nc = 12
    pr = np.where(np.arange(nc) < 6, 1 / (1 + np.exp(z[:, None] * 2)), 1 / (1 + np.exp(-z[:, None] * 2)))      # cards 0-5 at the low end, 6-11 at the high end
    B = sp.csr_matrix((rng.random((n, nc)) < pr).astype(float))
    cv = A.CardVectors(mv=np.full(nc, 3.0), land=np.zeros(nc, dtype=bool), creature=np.zeros(nc, dtype=bool), cmask=np.ones(nc, dtype=np.int64),
                       removal=np.zeros(nc), draw=np.zeros(nc), flying=np.zeros(nc), names=[f"card{i}" for i in range(nc)], basic=np.zeros(nc, dtype=bool))
    res = dict(idx=np.arange(n), axis=dict(score=z, corr=0.95, corrs=[0.9, 0.95, 0.97], ok=True, half_a=np.arange(n) % 2 == 0))
    mp = np.ones(n, dtype=bool)
    win_yes = dict(lo_eff=dict(eff=1.5, se=1.0, z=1.5), hi_eff=dict(eff=-1.5, se=1.0, z=-1.5), diff=dict(eff=-3.0, se=1.0, z=-3.0),
                   halves=[dict(eff=-2.0, se=1.5, z=-1.3), dict(eff=-4.0, se=1.5, z=-2.6)], verdict=dict(finding=True, replicated=True))
    blk = A.spectrum_block("UB", res, win_yes, sh, mp, B, cv)
    assert blk["shown"] and blk["win"]["finding"] and blk["win"]["badge"] in ("likely", "strong") and blk["win"]["halves"] == [-2.0, -4.0]
    assert "removal" in blk["a"]["label"] and "fliers" in blk["b"]["label"] and blk["a"]["n_decks"] == int(lo.sum())
    assert 1 <= len(blk["a"]["signature"]) <= A.SIG_N and blk["a"]["signature"][0]["card_id"] < 6 <= blk["b"]["signature"][0]["card_id"]
    assert "No clear difference" not in " ".join(blk["text"]) and blk["text"][0].startswith("UB decks range from")
    json.dumps(blk, default=float)
    win_no = dict(win_yes, diff=dict(eff=-0.4, se=1.0, z=-0.4), halves=[dict(eff=-2.0, se=1.5, z=-1.3), dict(eff=1.0, se=1.5, z=0.6)],
                  verdict=dict(finding=False, replicated=False))
    blk2 = A.spectrum_block("UB", res, win_no, sh, mp, B, cv)
    assert blk2["shown"] and not blk2["win"]["finding"] and blk2["text"][1] == "No clear difference in win rate between the two styles."
    bad = dict(res, axis=dict(res["axis"], corr=0.3, ok=False))
    assert A.spectrum_block("UB", bad, win_yes, sh, mp, B, cv)["shown"] is False
    assert A.spectrum_block("UB", dict(res, axis=None), None, sh, mp, B, cv)["shown"] is False
