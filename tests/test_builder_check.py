"""Builder check (sealed_lab.builder_check): arithmetic, badges, planted effects on synthetic games, and the corpus path."""
from __future__ import annotations

import json
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from sealed_lab import builder_check as B


def test_badge_thresholds():
    assert [B.badge(z) for z in (3.0, -3.0, 2.99, 2.0, -2.0, 1.99, 0.0, None)] == \
        ["Strong", "Strong", "Likely", "Likely", "Likely", "Weak", "Weak", "Weak"]


def test_combine_hand_example():
    # two metrics, beta = (0.10, -0.20) logit per unit, cov = [[0.0004, 0.0001], [0.0001, 0.0009]], d = (2, 1), se_d = (0.5, 0.25), p = 0.5
    beta, cov, d, se_d = np.array([0.10, -0.20]), np.array([[0.0004, 0.0001], [0.0001, 0.0009]]), np.array([2.0, 1.0]), np.array([0.5, 0.25])
    c = B.combine(beta, cov, d, se_d, 0.5)
    assert B.pp_scale(0.5) == pytest.approx(25.0)
    delta = 0.10 * 2 - 0.20 * 1                                           # 0 logit
    assert c["pp"] == pytest.approx(25.0 * delta)
    var_fixed = 4 * 0.0004 + 2 * 2 * 1 * 0.0001 + 1 * 0.0009              # d' S d = 0.0016 + 0.0004 + 0.0009 = 0.0029
    assert c["se_fixed"] == pytest.approx(25.0 * np.sqrt(0.0029))
    var_total = var_fixed + 0.10 ** 2 * 0.25 + 0.20 ** 2 * 0.0625         # + 0.0025 + 0.0025
    assert c["se_total"] == pytest.approx(25.0 * np.sqrt(0.0079))
    assert c["se_total"] > c["se_fixed"] and c["badge"] == "Weak" and c["z"] == pytest.approx(0.0)
    c2 = B.combine([0.3], [[0.0001]], [1.0], [0.0], 0.55)
    assert c2["pp"] == pytest.approx(100 * 0.55 * 0.45 * 0.3) and c2["se_fixed"] == pytest.approx(c2["se_total"])


def games(n: int, effects: dict, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    skill = rng.uniform(0.40, 0.70, n)
    X = dict(lands=rng.normal(17, 1.0, n).round(), creatures=rng.normal(15, 2.0, n).round(), two_drops=rng.normal(5, 1.5, n).round(),
             mean_mv=rng.normal(2.9, 0.3, n))
    eta = 0.1 + 6.0 * (skill - 0.55)
    for k, b in effects.items():
        eta = eta + b * (X[k] - X[k].mean())
    won = rng.random(n) < 1 / (1 + np.exp(-eta))
    t0 = datetime(2026, 1, 1)
    return pl.DataFrame(dict(
        fmt=["bo1"] * n, event=rng.choice(["e1", "e2"], n), draft_id=[f"d{i // 3}" for i in range(n)], deck_key=[f"k{i // 3}" for i in range(n)],
        match_number=[1] * n, game_number=[1] * n, game_time=[t0 + timedelta(hours=int(h)) for h in rng.integers(0, 24 * 20, n)],
        main_colors=rng.choice(["WU", "WB", "UB", "RG", "BR"], n), splash_colors=[""] * n, on_play=rng.random(n) < 0.5,
        num_mulligans=rng.integers(0, 2, n), opp_num_mulligans=rng.integers(0, 2, n), opp_colors=rng.choice(["WU", "RG"], n),
        num_turns=[8] * n, won=won, skill=skill, **X))


DIFFS = {"lands": (1.0, 0.1), "creatures": (2.0, 0.2), "two_drops": (-1.0, 0.1), "mean_mv": (0.2, 0.02)}


def test_planted_effect_recovered():
    # creatures: +0.10 logit per card (sd 2 -> 0.2 per sd); two_drops: -0.10; builder has +2 creatures and -1 two-drop -> +0.3 logit
    g = games(30000, dict(creatures=0.10, two_drops=-0.10), seed=1)
    r = B.analyze(g, DIFFS)
    assert r["n_games"] == 30000 and "skipped" not in r
    bt = r["betas"]
    assert bt["creatures"]["pp_per_unit"] > 0 > bt["two_drops"]["pp_per_unit"]
    assert bt["creatures"]["pp_per_unit"] == pytest.approx(25 * 0.10, abs=0.8) and bt["creatures"]["badge"] == "Strong"
    assert bt["lands"]["badge"] == "Weak" and bt["mean_mv"]["badge"] == "Weak"
    c = r["combined"]
    assert c["pp"] == pytest.approx(25 * 0.30, abs=1.5) and c["pp"] > 0 and c["badge"] in ("Likely", "Strong")
    assert c["se_total"] >= c["se_fixed"] > 0
    assert bt["creatures"]["builder_direction"] == "more" and bt["two_drops"]["builder_direction"] == "fewer"
    json.dumps(r)


def test_no_planted_effect_is_weak():
    for seed in (2, 3):
        r = B.analyze(games(20000, {}, seed=seed), DIFFS)
        assert r["combined"]["badge"] == "Weak", r["combined"]
        assert all(v["badge"] == "Weak" for v in r["betas"].values())


def test_too_few_games_and_missing_metric():
    r = B.analyze(games(2000, {}), DIFFS)
    assert "minimum" in r["skipped"] and "betas" not in r
    r = B.analyze(games(6000, {"creatures": 0.1}), {"creatures": (1.0, 0.1)})
    assert r["combined"]["metrics_used"] == ["creatures"] and set(r["skipped_metrics"]) == {"lands", "two_drops", "mean_mv"}
    r = B.analyze(games(6000, {}), {})
    assert r["combined"] is None and "skipped" in r


def test_run_on_the_synthetic_corpus(built, tmp_path):
    comp = tmp_path / "builder_composition_TST.json"
    comp.write_text(json.dumps({"formats": {"bo1": {k: dict(diff_players=v[0], diff_players_se=v[1]) for k, v in DIFFS.items() if k != "lands"}}}))
    res = B.run("TST", composition_path=comp)
    json.dumps(res)
    assert res["set"] == "TST" and set(res["formats"]) == {"bo1", "bo3"}
    assert "no builder composition data" in res["formats"]["bo3"]["skipped"]
    f = res["formats"]["bo1"]
    assert "skipped" in f or set(f["builder_diffs"]) == {"creatures", "two_drops", "mean_mv"}
    p = B.write(res, tmp_path / "out")
    assert p.name == "builder_check_TST.json" and json.loads(p.read_text())["set"] == "TST"
    miss = B.run("TST", composition_path=tmp_path / "nope.json")
    assert "missing" in miss["skipped"]
