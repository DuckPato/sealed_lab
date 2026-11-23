"""scripts/pooled_retest.py: the pooling arithmetic and the pre-registered verdict (synthetic numbers, no report.json)."""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))
_SPEC = importlib.util.spec_from_file_location("pooled_retest", _SCRIPTS / "pooled_retest.py")
pr = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pr)


def _o(gain, se):
    return dict(gain=gain, lo=gain - 1.96 * se, hi=gain + 1.96 * se, lo90=gain - 1.2816 * se)


def _cell(w, a, b, se=0.0005, **kw):
    return dict(whole=_o(w, se), A=_o(a, se), B=_o(b, se), confirmed=False, applied=False, **kw)


def test_min_gain_imported():
    assert pr.MIN_GAIN == 1e-4


def test_se_from_ci():
    assert pr.se_from_ci(-1.96, 1.96) == pytest.approx(1.0)


def test_dl_hand_computed_heterogeneous():
    # y = 0, 2, 4, se = 1: fixed mean 2, Q = 4 + 0 + 4 = 8, df 2, C = 3 - 3/3 = 2, tau2 = 3, I2 = 0.75,
    # RE weights 1/4 each -> mean 2, se = sqrt(1 / 0.75)
    r = pr.dersimonian_laird([0, 2, 4], [1, 1, 1])
    assert r["Q"] == pytest.approx(8.0)
    assert r["tau2"] == pytest.approx(3.0)
    assert r["I2"] == pytest.approx(0.75)
    assert r["mean"] == pytest.approx(2.0)
    assert r["se"] == pytest.approx(math.sqrt(1 / 0.75))
    assert r["lo"] == pytest.approx(2 - 1.96 * math.sqrt(1 / 0.75))
    assert r["fe_mean"] == pytest.approx(2.0) and r["fe_se"] == pytest.approx(math.sqrt(1 / 3))


def test_dl_unequal_weights_homogeneous():
    # y = 1, 3 ; se = 1, 2: w = 1, .25 ; fixed mean 1.4 ; Q = .8 < df 1 -> tau2 = 0, I2 = 0, RE == FE
    r = pr.dersimonian_laird([1, 3], [1, 2])
    assert r["fe_mean"] == pytest.approx(1.4)
    assert r["Q"] == pytest.approx(0.8)
    assert r["tau2"] == 0.0 and r["I2"] == 0.0
    assert r["mean"] == pytest.approx(1.4) and r["se"] == pytest.approx(math.sqrt(1 / 1.25))


def test_dl_unequal_weights_with_tau():
    # y = 0, 4 ; se = 1, 1: Q = 8, df 1, C = 2 - 2/2 = 1, tau2 = 7 ; RE weights 1/8 each -> mean 2, se = 2
    r = pr.dersimonian_laird([0, 4], [1, 1])
    assert r["tau2"] == pytest.approx(7.0) and r["mean"] == pytest.approx(2.0) and r["se"] == pytest.approx(2.0)


def test_single_study():
    r = pr.dersimonian_laird([0.5], [0.1])
    assert r["mean"] == 0.5 and r["tau2"] == 0 and r["I2"] == 0 and r["se"] == pytest.approx(0.1)


def test_verdict_better_needs_both_halves():
    cells = [_cell(0.0010, 0.0010, 0.0010) for _ in range(5)]
    v = pr.verdict(pr.pool_parts(cells))
    assert v["better"] and not v["worse"] and v["label"] == "pooled-better"
    # half B pointing the wrong way blocks it
    v = pr.verdict(pr.pool_parts([_cell(0.0005, 0.0010, -0.0001) for _ in range(5)]))
    assert not v["better"] and v["label"] == "neither"
    # half B positive but its 90% lower bound below zero blocks it (se 0.001 per set, 4 sets -> se 0.0005, mean 0.0002)
    v = pr.verdict(pr.pool_parts([_cell(0.0005, 0.0010, 0.0002, se=0.001) for _ in range(4)]))
    assert not v["better"]


def test_verdict_materiality_floor():
    # tiny but precise: significant, below MIN_GAIN -> neither
    v = pr.verdict(pr.pool_parts([_cell(5e-5, 5e-5, 5e-5, se=1e-5) for _ in range(4)]))
    assert v["label"] == "neither"


def test_verdict_worse_on_a_half_vetoes():
    cells = [_cell(0.0, 0.0010, -0.0010) for _ in range(6)]
    v = pr.verdict(pr.pool_parts(cells))
    assert v["worse"] and v["label"] == "pooled-worse"
    assert not pr.qualifies({"bo1": v})


def test_qualifies():
    good = dict(better=True, worse=False)
    bad = dict(better=False, worse=True)
    none = dict(better=False, worse=False)
    assert pr.qualifies({"bo1": good, "bo3": none})
    assert not pr.qualifies({"bo1": good, "bo3": bad})
    assert not pr.qualifies({"bo1": none, "bo3": none})


def test_materially_worse():
    assert pr.materially_worse(_o(-0.001, 0.0002))
    assert not pr.materially_worse(_o(-0.001, 0.001))      # CI includes zero
    assert not pr.materially_worse(_o(-5e-5, 1e-6))        # significant but below the floor


def test_leave_one_out_excludes_own_set():
    # set X is a huge outlier; its own LOO pool must not contain it
    cells = {s: _cell(0.001, 0.001, 0.001) for s in "ABCDE"}
    cells["X"] = _cell(-0.05, -0.05, -0.05)
    loo = pr.leave_one_out(cells)
    assert loo["X"]["verdict"]["better"]
    assert loo["X"]["pooled"]["whole"]["k"] == 5
    assert loo["A"]["pooled"]["whole"]["k"] == 5 and not loo["A"]["verdict"]["better"]


def test_loo_check_flips_and_ships():
    good = {s: _cell(0.001, 0.001, 0.001) for s in "ABCDEFGH"}
    good["A"]["applied"] = True
    bad_own = _cell(-0.0012, -0.0012, -0.0012, se=0.0004)         # materially worse on its own: never pooled-applied
    good["Z"] = bad_own
    out = pr.loo_check({"bo1": good, "bo3": {}}, "applied")
    rows = {r["set"]: r for r in out["rows"]}
    assert rows["A"]["pooled"] and not rows["A"]["flipped"]
    assert rows["B"]["flipped"]
    assert not rows["Z"]["pooled"] and rows["Z"]["own_worse"]
    assert out["shipping"]["n_flipped"] == 7
    assert out["shipping"]["fe_gain"] == pytest.approx(0.001) and out["shipping"]["passes"]
