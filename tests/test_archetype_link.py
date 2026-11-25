"""Pool tab -> archetype link (sealed_lab.archetype_link): the stored style projector reproduces the report's thirds, the line's wording, the API."""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest
import scipy.sparse as sp

from sealed_lab import archetype_link as L
from sealed_lab import report as R
from sealed_lab import report_arch as A
from tests.test_report_arch import synth_decks


# --------------------------------------------------------------------------- the projector

@pytest.mark.parametrize("kind", ["continuum", "two"])
def test_projector_reproduces_the_fitted_scores_and_thirds(kind):
    Xb, S, _ev, _truth = synth_decks(kind, n=900, seed=4)
    n_cols, n_cards = Xb.shape[1], Xb.shape[1] + 30
    ids = np.arange(n_cols) + 7                                   # the pair's cards sit at scattered card ids of a bigger card list
    avg_mv = 3.0 - S[:, 0] + 0.1 * np.arange(len(S)) / len(S)    # any deck feature the orientation looks at
    pp = L.fit_pair_projector(Xb, ids, S, avg_mv, n_cards)
    # the fit as report_arch does it (Embedder rank 1 on all the pair's decks, flipped to follow the average mana value)
    sc = A.Embedder(Xb, S, rank=1).transform(Xb, S)[:, 0]
    if np.corrcoef(sc, avg_mv)[0, 1] < 0:
        sc = -sc
    q1, q2 = np.quantile(sc, [1 / 3, 2 / 3])
    want = np.where(sc <= q1, 0, np.where(sc >= q2, 2, 1))
    # project from the copies matrix over ALL cards (copies, other columns, ids shifted), not from the fitted binary matrix
    X = sp.lil_matrix((Xb.shape[0], n_cards))
    X[:, ids] = Xb * 2.0
    X[:, 0] = 17
    got_sc = L.project_score(pp, X.tocsr(), S)
    assert np.allclose(got_sc, sc, atol=1e-9)
    assert (L.thirds_of(pp, got_sc) == want).all()
    assert np.bincount(want, minlength=3).tolist() == [pp["lo_n"], int((want == 1).sum()), pp["hi_n"]]
    assert pp["q1"] == pytest.approx(q1) and pp["q2"] == pytest.approx(q2)
    json.dumps(pp)


def test_projector_follows_the_orientation_rule():
    Xb, S, _ev, _t = synth_decks("continuum", n=600, seed=1)
    ids = np.arange(Xb.shape[1])
    up = L.fit_pair_projector(Xb, ids, S, S[:, 0], Xb.shape[1])
    down = L.fit_pair_projector(Xb, ids, S, -S[:, 0], Xb.shape[1])
    X = sp.csr_matrix(Xb)
    a, b = L.project_score(up, X, S), L.project_score(down, X, S)
    assert np.allclose(a, -b, atol=1e-9) and np.corrcoef(a, S[:, 0])[0, 1] > 0       # the high end follows the higher average mana value


def _fake_report_from_build(model, fmt):
    """A report whose archetype entries say what `build_archetypes` itself found (axis scores, thirds), for every pair with an axis."""
    g = R.load_games("TST", fmt)
    cards = model.cards.sort("card_id")
    decks, X = A.deck_table("TST", g, cards)
    cv = A.CardVectors.from_cards(cards)
    sh = A.deck_shape(X, cv, decks["mmask"].to_numpy())
    S = A.style_matrix(sh)
    grp = decks["grp"].to_numpy()
    pairs, thirds = [], {}
    for p in A.design.PAIRS:                              # the loop of build_archetypes, with a small-set size limit for the axis
        idx = np.nonzero(grp == p)[0]
        if len(idx) < 20:
            continue
        Xb, _ = A._pair_binary(X[idx], cv, A.colour_mask(p))
        ax = dict(score=A.Embedder(Xb, S[idx], rank=1).transform(Xb, S[idx])[:, 0])        # (the replication check needs more decks than the synthetic set has)
        if np.corrcoef(ax["score"], sh["avg_mv"][idx])[0, 1] < 0:
            ax["score"] = -ax["score"]
        sc = ax["score"]
        q1, q2 = np.quantile(sc, [1 / 3, 2 / 3])
        lv = np.where(sc <= q1, 0, np.where(sc >= q2, 2, 1))
        thirds[p] = (idx, lv)
        end = lambda lab, n: dict(label=lab, n_decks=n, effect_pp=1.0, badge="likely", range="likely between +0.1 and +1.9 points")
        pairs.append(dict(key=p, code=p, label=R.group_label(p), n_decks=int(len(sc)), effect_pp=0.5, badge="weak", range="r",
                          spectrum=dict(shown=True, a=end("low", int((lv == 0).sum())), b=end("high", int((lv == 2).sum())))))
    return dict(archetypes=dict(pairs=pairs)), thirds, (decks, X, cv)


@pytest.mark.parametrize("fmt", ["bo1", "bo3"])
def test_fitted_decks_get_the_thirds_the_report_assigned(calc, model, fmt):
    rep, thirds, (decks, X, cv) = _fake_report_from_build(model, fmt)
    assert thirds, "the synthetic set has at least one pair with an axis"
    proj = L.fit_projectors("TST", fmt, model, rep)
    assert set(proj["pairs"]) == set(thirds)                                     # none left out by the size check
    sh = A.deck_shape(X, cv, decks["mmask"].to_numpy())
    S = A.style_matrix(sh)
    for p, (idx, lv) in thirds.items():
        got = L.thirds_of(proj["pairs"][p], L.project_score(proj["pairs"][p], X[idx], S[idx]))
        assert (got == lv).all(), p
    # one deck at a time, from {card: copies} as the builder holds it
    p, (idx, lv) = next(iter(thirds.items()))
    for i in range(0, len(idx), max(1, len(idx) // 12)):
        row = X[idx[i]]
        deck = {int(c): int(n) for c, n in zip(row.indices, row.data)}
        main = decks["main_colors"][int(idx[i])]
        assert L.deck_style_score(proj["pairs"][p], deck, cv, main)[1] == lv[i]
    # a report whose stored end sizes differ is not the same axis: left out
    bad = copy.deepcopy(rep)
    bad["archetypes"]["pairs"][0]["spectrum"]["a"]["n_decks"] += 1
    assert bad["archetypes"]["pairs"][0]["key"] not in L.fit_projectors("TST", fmt, model, bad)["pairs"]


# --------------------------------------------------------------------------- the line

def _pair(**kw):
    end = lambda label, eff, badge: dict(label=label, effect_pp=eff, badge=badge, range="likely between a and b points")
    a = dict(key="UB", code="UB+r/g", label="Blue-Black", n_decks=900, effect_pp=1.8, badge="likely", range="likely between +0.4 and +3.2 points",
             spectrum=dict(shown=True, a=end("low curve / 2-drops", 2.2, "likely"), b=end("high curve / fliers", -0.3, "weak")))
    a.update(kw)
    return a


def test_line_wording_follows_the_stored_figures():
    t = L.compose(_pair(), "UB", 0)["text"]
    assert t == ("Color pair: Blue-Black (UB+r/g). Decks in this pair: +1.8 points (Likely). Style: closer to the 'low curve / 2-drops' end "
                 "(that end: +2.2 points, Likely). Observed in 17lands data; the builder does not use archetypes.")
    weak_end = L.compose(_pair(), "UB", 2)
    assert "'high curve / fliers' end (that end shows no clear difference from an average deck)" in weak_end["text"]
    assert weak_end["spectrum"]["end"] == "b" and weak_end["spectrum"]["states_effect"] is False and "-0.3" not in weak_end["text"]
    mid = L.compose(_pair(), "UB", 1)
    assert "in the middle" in mid["text"] and mid["spectrum"]["end"] == "mid" and "that end" not in mid["text"]
    weak_pair = L.compose(_pair(effect_pp=0.2, badge="weak"), "UB", None)
    assert "Decks in this pair: no clear difference from an average deck." in weak_pair["text"] and "0.2" not in weak_pair["text"]
    assert "Style" not in weak_pair["text"] and weak_pair["spectrum"] == dict(shown=False)
    hidden = _pair()
    hidden["spectrum"] = dict(shown=False, why="x")
    assert "Style" not in L.compose(hidden, "UB", 0)["text"]                       # no spectrum -> no style claim
    neg = _pair(effect_pp=-2.1, badge="strong")
    neg["spectrum"]["a"]["effect_pp"] = -3.3
    assert "-2.1 points (Strong)" in L.compose(neg, "UB", 0)["text"] and "-3.3 points, Likely" in L.compose(neg, "UB", 0)["text"]


def test_deck_without_a_pair_has_no_line():
    rep = dict(archetypes=dict(pairs=[_pair()]))
    for main, word in (("W", "one main color"), ("WUB", "three or more")):
        out = L.describe(rep, {"pairs": {}}, {1: 1}, None, main)                 # returns before it needs card data
        assert out["available"] is False and word in out["reason"]
    assert L.describe(rep, {"pairs": {}}, {1: 1}, None, "WR")["available"] is False        # pair not in the report (too few decks)


# --------------------------------------------------------------------------- the API

def test_deck_archetype_api_on_the_synthetic_set(calc, model):
    import polars as pl

    from sealed_lab import api, config

    assert api.deck_archetype("NOPE", "bo1", {1: 40}).get("available") is False             # unknown set: no exception
    assert api.deck_archetype("TST", "bo7", {1: 40})["available"] is False
    cd = config.corpus_dir("TST")
    dk = pl.read_parquet(cd / "deck_cards.parquet", columns=["deck_key", "card_id", "n"])
    key = dk["deck_key"].unique().sort()[0]
    deck = dict(dk.filter(pl.col("deck_key") == key).select("card_id", "n").iter_rows())
    rp, pj = R.cache_path("TST", "bo1"), L.proj_path("TST", "bo1")
    rp.unlink(missing_ok=True)
    pj.unlink(missing_ok=True)
    assert "not been computed" in api.deck_archetype("TST", "bo1", deck)["reason"]            # reads the stored report only, never computes it
    api.set_report("TST", "bo1")
    before = rp.read_bytes()
    out = api.deck_archetype("TST", "bo1", deck)
    assert isinstance(out, dict) and "available" in out
    if out["available"]:
        assert out["text"].startswith("Color pair: ") and out["key"] in A.design.PAIRS and "builder does not use archetypes" in out["text"]
    assert rp.read_bytes() == before                                                              # the report file is untouched
    rep = json.loads(before)
    if any(a["spectrum"].get("shown") for a in rep["archetypes"]["pairs"]):
        d = json.loads(pj.read_text(encoding="utf-8"))
        assert d["version"] == L.PROJ_VERSION and d["built_at"] == model.meta.get("built_at") and d["fmt"] == "bo1"
    assert rep["version"] == R.REPORT_VERSION
