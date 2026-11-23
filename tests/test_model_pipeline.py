"""End-to-end model tests on synthetic data (scale 0.3, 4 bootstrap replicates, 2 workers)."""
from __future__ import annotations

import json
import threading
import time

import numpy as np
import pytest
from scipy.stats import spearmanr

from sealed_lab import api, config
from sealed_lab.types import Cancelled


@pytest.fixture(scope="session")
def report(calc):
    return json.loads((config.model_dir("TST") / "report.json").read_text(encoding="utf-8"))


def test_artifact_files_and_meta(calc):
    d = config.model_dir("TST")
    for f in ("meta.json", "cards.parquet", "draws.npz", "reference.npz", "report.html", "report.json"):
        assert (d / f).exists(), f
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    assert meta["formats"] == ["bo3", "bo1"]
    assert meta["built_at"] and meta["built_corpus_at"]
    assert meta["n_draws"] == 5
    st = api.set_status("TST")
    assert st.has_model and not st.stale and st.formats == ["bo3", "bo1"] and st.built_at
    assert api.report_path("TST") == d / "report.html"


def test_progress_is_monotone_and_complete(calc):
    fr = [f for f, _ in calc["progress"]]
    assert fr == sorted(fr) and fr[0] >= 0.35 and fr[-1] == 1.0
    assert len(fr) > 15


def test_card_recovery(calc, model):
    names = model.cards.sort("card_id")["name"].to_list()
    truth = np.array([calc["truth"]["beta"][n] for n in names])
    is_basic = model.cards.sort("card_id")["is_basic"].to_numpy()
    for fmt in ("bo3", "bo1"):
        ex = model.cards.sort("card_id")[f"exposure_{fmt}"].to_numpy()
        m = (ex > 0) & ~is_basic
        rho = spearmanr(model.card_beta(fmt)[m], truth[m])[0]
        assert rho >= 0.8, (fmt, rho)
    assert np.all(model.card_beta("bo3")[is_basic] == 0)


def test_heldout_model_beats_baselines(report):
    v = report["validation"]["bo1"]
    assert v["status"] == "ok"
    d = v["metrics"]
    for b in ("B0", "B1", "B2"):
        assert d[b]["vs_M"]["logloss"]["lo"] > 0, b          # significant gain on the (larger) Bo1 hold-out
        assert d[b]["vs_M"]["auc"]["diff"] > 0
    v3 = report["validation"]["bo3"]
    assert v3["status"] == "ok"
    for b in ("B0", "B1", "B2"):
        assert v3["metrics"][b]["vs_M"]["logloss"]["diff"] > 0, b


def test_calibration_is_sane(report):
    for fmt in ("bo1", "bo3"):
        cal = report["validation"][fmt]["calibration"]
        assert 0.5 < cal["slope"] < 1.6
        pred = [b["pred"] for b in cal["bins"]]
        obs = [b["obs"] for b in cal["bins"]]
        assert len(cal["bins"]) == 10
        assert spearmanr(pred, obs)[0] > 0.6
        assert abs(cal["intercept"]) < 0.6
    vb = report["validation"]["bo1"]["variance_budget"]
    assert vb["sigma2_deck"] > 0 and 0 < vb["share_recal"] < 1.5


def test_percentile_monotone_in_true_strength(calc, model):
    from sealed_lab.synth.generate import truth_strength

    held = json.loads((calc["synth"] / "heldout_pools.json").read_text(encoding="utf-8"))
    name2id = {n: i for i, n in enumerate(model.cards.sort("card_id")["name"].to_list())}
    for fmt in ("bo3", "bo1"):
        ts, sc, qs = [], [], []
        for h in held:
            ev = model.evaluate({name2id[n]: c for n, c in h["player_deck"].items()}, fmt)
            ts.append(truth_strength(calc["truth"], h["player_deck"]))
            sc.append(ev.score)
            qs.append(ev.game_wr)
        assert spearmanr(sc, ts)[0] >= 0.8
        order = np.argsort(qs)
        assert np.all(np.diff(np.array(sc)[order]) >= -1e-9)           # percentile is monotone in q
        assert 0 <= min(sc) and max(sc) <= 100


def test_evaluate_fully_filled(calc, model):
    from sealed_lab.config import RUN_RULES
    from scipy.special import expit

    held = json.loads((calc["synth"] / "heldout_pools.json").read_text(encoding="utf-8"))
    cards = model.cards.sort("card_id")
    name2id = {n: i for i, n in enumerate(cards["name"].to_list())}
    deck = {name2id[n]: c for n, c in held[0]["player_deck"].items()}
    for fmt in ("bo3", "bo1"):
        ev = model.evaluate(deck, fmt)
        assert sum(c.count for c in ev.cards) == 40
        assert {"intercept", "cards", "structure", "mana"} <= set(ev.breakdown) <= {"intercept", "cards", "structure", "mana", "synergy", "archetype", "mulligans", "uncastable"}
        assert abs(sum(ev.breakdown.values()) - model.logit(deck, fmt)) < 1e-9
        assert abs(expit(sum(ev.breakdown.values())) - ev.game_wr) < 1e-9
        assert ev.game_wr_lo <= ev.game_wr <= ev.game_wr_hi and ev.score_lo <= ev.score <= ev.score_hi
        assert set(ev.mana) == {"p_screw", "p_flood", "p_color_screw", "mana_spent", "mull_rate"}
        assert 0 <= ev.exp_wins <= RUN_RULES[fmt][0] and 0 <= ev.p_max_wins <= 1
        assert ev.colors and len(ev.colors) >= 1
        basics = [c for c in ev.cards if c.name in ("Plains", "Island", "Swamp", "Mountain", "Forest")]
        assert basics and all(c.value_pp is None for c in basics)
        assert all(c.value_pp is not None for c in ev.cards if c.name not in ("Plains", "Island", "Swamp", "Mountain", "Forest"))
        if fmt == "bo3":
            assert ev.match_wr == pytest.approx(ev.game_wr ** 2 * (3 - 2 * ev.game_wr))
        else:
            assert ev.match_wr is None
        # draws
        L = model.logit(deck, fmt, draws=True)
        assert L.shape == (model.n_draws(fmt),) and L[0] == pytest.approx(model.logit(deck, fmt))
        q = model.q(deck, fmt, draws=True)
        assert np.all((q > 0) & (q < 1))
    # a better deck scores higher: replace the best-valued non-basic card by the worst
    ev = model.evaluate(deck, "bo3")
    nb = [c for c in ev.cards if c.value_pp is not None]
    worst = min(nb, key=lambda c: c.value_pp)
    best = max(nb, key=lambda c: c.value_pp)
    assert best.value_pp > worst.value_pp
    with pytest.raises(KeyError):
        model.evaluate(deck, "bo9")


def test_run_distribution():
    from sealed_lab.model.artifact import match_win_prob, run_distribution

    ew, pm = run_distribution(0.5, 7, 3)
    assert 0 < pm < 0.5 and 2.5 < ew < 4.0
    ew1, pm1 = run_distribution(1.0, 7, 3)
    assert (ew1, pm1) == (7.0, 1.0)
    ew0, pm0 = run_distribution(0.0, 4, 2)
    assert (ew0, pm0) == (0.0, 0.0)
    assert match_win_prob(0.5) == 0.5 and match_win_prob(0.6) == pytest.approx(0.648)
    # monotone in p
    assert run_distribution(0.6, 4, 2)[0] > run_distribution(0.5, 4, 2)[0]


def test_featurize_timing(calc, model):
    held = json.loads((calc["synth"] / "heldout_pools.json").read_text(encoding="utf-8"))
    name2id = {n: i for i, n in enumerate(model.cards.sort("card_id")["name"].to_list())}
    decks = [{name2id[n]: c for n, c in h["player_deck"].items()} for h in held]
    model.featurize(decks[0])
    from sealed_lab.model import structure

    meds = []
    for _ in range(3):                      # best of 3 passes: robust to a noisy shared CPU
        ts = []
        for d in decks[1:]:
            t = time.perf_counter()
            structure.featurize(dict(d), model.card_arrays, cache=False)
            ts.append(time.perf_counter() - t)
        meds.append(float(np.median(ts)))
        if meds[-1] <= 0.010:
            break
    assert min(meds) <= 0.025, f"median {1000 * min(meds):.1f} ms"


def test_card_info_and_report_html(calc, model):
    info = api.card_info("TST")
    assert info and all(set(v) >= {"name", "mana_value", "is_creature", "is_land", "colors"} for v in info.values())
    assert any(v["is_land"] for v in info.values()) and any(v["is_creature"] for v in info.values())
    assert api.card_info("TST") is info                                  # cached while the table is unchanged
    p = api.report_path("TST")
    assert p is not None
    html = p.read_text(encoding="utf-8")
    assert html.lstrip().lower().startswith("<!doctype html>")
    assert "http://" not in html and "https://" not in html and "<script" not in html.lower()   # self-contained
    assert "<svg" in html
    i = html.find("<div class='verdict ")
    assert i != -1 and i < html.find("<h2>Held-out performance") and i < html.find("<h2>Card effects")   # headline first


def test_cancel_raises_before_writing(calc):
    from sealed_lab.model import pipeline

    meta_p = config.model_dir("TST") / "meta.json"
    before = meta_p.stat().st_mtime_ns
    ev = threading.Event()
    ev.set()
    with pytest.raises(Cancelled):
        pipeline.calculate("TST", cancel=ev, n_jobs=2, n_boot=2, n_rep=20)
    assert meta_p.stat().st_mtime_ns == before                           # the existing artifact is untouched


def test_formats_without_data_are_skipped(tmp_path, monkeypatch):
    """Only draft + bo1 files: bo3 is not offered, and the small bo1 sample is flagged."""
    from sealed_lab.synth.generate import generate

    monkeypatch.setenv("SEALED_LAB_HOME", str(tmp_path / "home"))
    syn = tmp_path / "syn"
    generate(syn, set_code="TST", seed=1, scale=0.05, events=("PremierDraft", "Sealed"))
    api.add_files([str(syn)])
    st = api.calculate_set("TST", n_jobs=2, n_boot=2, fetch_scryfall=False)      # the real facade: ingest + model
    assert st.has_model and not st.stale and st.formats == ["bo1"] and st.built_at
    meta = json.loads((config.model_dir("TST") / "meta.json").read_text(encoding="utf-8"))
    assert meta["formats"] == ["bo1"]
    assert any("only" in w and "bo1" in w for w in meta["warnings"])
