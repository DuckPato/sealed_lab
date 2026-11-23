"""scripts/builder_scoreboard.py: pooling, heterogeneity and the paired table (pure arithmetic, no model)."""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np

_SPEC = importlib.util.spec_from_file_location("builder_scoreboard", Path(__file__).resolve().parents[1] / "scripts" / "builder_scoreboard.py")
bs = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bs)


def test_pool_identical_estimates_have_no_heterogeneity():
    p = bs.pool([-1.0, -1.0, -1.0, -1.0], [0.4, 0.4, 0.4, 0.4])
    assert math.isclose(p["fixed"]["b"], -1.0)
    assert math.isclose(p["fixed"]["se"], 0.2)                    # 0.4 / sqrt(4)
    assert p["Q"] == 0 and p["I2"] == 0 and p["tau"] == 0
    assert math.isclose(p["random"]["se"], p["fixed"]["se"])


def test_pool_weights_by_inverse_variance_and_detects_heterogeneity():
    p = bs.pool([-0.5, -1.5], [0.1, 1.0])
    assert -0.51 < p["fixed"]["b"] < -0.5                         # the precise cell dominates
    far = bs.pool([-0.2, -1.8, -0.3, -1.7], [0.1, 0.1, 0.1, 0.1])
    assert far["Q_p"] < 0.001 and far["I2"] > 0.9 and far["tau"] > 0.5
    assert far["random"]["se"] > far["fixed"]["se"]               # sets disagree: the random-effects interval is wider
    assert math.isclose(far["fixed"]["z_vs_target"], (far["fixed"]["b"] + 1) / far["fixed"]["se"])


def _res(sc, b_bo1, b_bo3, ids=("1", "2")):
    def fmt(b, seed):
        boots = (b + np.random.default_rng(seed).normal(0, 0.3, 50)).tolist()
        return dict(n_events=2, n_games=10, a=1.0, a_ci=[0.0, 2.0], b=b, b_ci=[b - 0.6, b + 0.6], mean_predicted_gain_pp=5.0,
                    event_ids=list(ids), b_boots=boots)
    return dict(set=sc, formats=dict(bo1=fmt(b_bo1, 1), bo3=fmt(b_bo3, 2)))


def test_cells_pooled_rows_and_pairing(tmp_path):
    new = {"AAA": _res("AAA", -1.0, -0.8), "BBB": _res("BBB", -1.2, -1.0)}
    old = {"AAA": _res("AAA", -0.9, -0.7), "BBB": _res("BBB", -1.2, -1.1, ids=("1", "3"))}
    rows = bs.cells(new)
    assert [(r["set"], r["fmt"]) for r in rows] == [("AAA", "bo1"), ("AAA", "bo3"), ("BBB", "bo1"), ("BBB", "bo3")]
    pooled = bs.pooled_rows(rows)
    assert set(pooled) == {"bo1", "bo3", "all"} and pooled["all"]["k"] == 4 and pooled["all"]["n_events"] == 8
    prow = bs.pair(new, old)
    flags = {(r["set"], r["fmt"]): r["same_events"] for r in prow}
    assert flags[("AAA", "bo1")] and not flags[("BBB", "bo3")]
    assert "diff_ci" in next(r for r in prow if r["set"] == "AAA" and r["fmt"] == "bo1")
    assert "diff_ci" not in next(r for r in prow if r["set"] == "BBB" and r["fmt"] == "bo3")      # different events: no paired interval
    assert bs.pair_summary(prow)["all"]["k"] == 2
    # the CLI writes both files
    for sc, r in new.items():
        (tmp_path / "new" / sc).mkdir(parents=True)
        (tmp_path / "new" / sc / "deviation_cost.json").write_text(json.dumps(r), encoding="utf-8")
    assert bs.main([str(tmp_path / "new"), str(tmp_path / "out" / "board")]) == 0
    assert (tmp_path / "out" / "board.json").exists() and (tmp_path / "out" / "board.md").exists()
