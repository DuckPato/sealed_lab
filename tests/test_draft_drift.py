"""scripts/draft_drift.py: the pure arithmetic of the drift test (synthetic numbers, no model fit)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))
_SPEC = importlib.util.spec_from_file_location("draft_drift", _SCRIPTS / "draft_drift.py")
dd = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(dd)


def _fit(rng, n, noise, drift=0.0, expo=1000.0):
    """One synthetic split: true betas, +drift in the second half, estimates with noise SD `noise` (reported SE = noise)."""
    truth = rng.normal(0, 0.05, n)
    b1 = truth + rng.normal(0, noise, n)
    b2 = truth + drift + rng.normal(0, noise, n)
    return dict(b1=b1, se1=np.full(n, noise), e1=np.full(n, expo), b2=b2, se2=np.full(n, noise), e2=np.full(n, expo))


def test_time_split_orders_and_halves_equal():
    ids = list("abcdef")
    starts = [5, 1, 3, 2, 6, 4]
    e, l = dd.time_split(ids, starts)
    assert e == ["b", "d", "c"] and set(l) == {"a", "e", "f"}
    assert max(starts[ids.index(i)] for i in e) < min(starts[ids.index(i)] for i in l)
    e2, l2 = dd.time_split([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    assert (len(e2), len(l2)) == (2, 3)


def test_zstats_and_nan_handling():
    d, z = dd.zstats([0, 0, 0], [3, 0, 1], [3, 0, 1], [4, 0, 1], valid=[True, True, False])
    assert d[0] == pytest.approx(3.0) and z[0] == pytest.approx(3.0 / 5.0)
    assert np.isnan(z[1]) and np.isnan(z[2])                        # zero SE / invalid card


def test_chi2_calibrated_under_no_drift_and_detects_drift():
    rng = np.random.default_rng(0)
    n = 4000
    f0 = _fit(rng, n, 0.02)
    _, z0 = dd.zstats(f0["b1"], f0["se1"], f0["b2"], f0["se2"])
    c0 = dd.chi2_summary(z0)
    assert c0["n"] == n and c0["chi2_over_n"] == pytest.approx(1.0, abs=0.08)       # SE = true noise SD for each half -> N(0, 1)
    assert c0["share_gt"] == pytest.approx(0.0455, abs=0.012)
    assert c0["p_nominal"] > 0.001
    f1 = _fit(rng, n, 0.02, drift=0.0)
    f1["b2"] = f1["b2"] + rng.normal(0, 0.03, n)                                      # real change of SD 0.03
    _, z1 = dd.zstats(f1["b1"], f1["se1"], f1["b2"], f1["se2"])
    c1 = dd.chi2_summary(z1)
    assert c1["chi2_over_n"] > 2.0 and c1["p_nominal"] < 1e-6 and c1["share_gt"] > 0.12


def test_empirical_p_and_null_compare():
    assert dd.empirical_p(5.0, [1, 2, 3, 4]) == pytest.approx(1 / 5)
    assert dd.empirical_p(0.0, [1, 2, 3, 4]) == pytest.approx(1.0)
    c = dd.null_compare(10.0, [1.0, 2.0, 3.0])
    assert c["null_mean"] == pytest.approx(2.0) and c["ratio"] == pytest.approx(5.0) and c["p_emp"] == pytest.approx(0.25)
    assert c["z_vs_null"] == pytest.approx(8.0)


def test_excess_variance_recovers_injected_drift_sd():
    rng = np.random.default_rng(1)
    n, noise, sd_drift = 20000, 0.02, 0.03
    f = _fit(rng, n, noise)
    shift = rng.normal(0, sd_drift, n)
    f["b2"] = f["b2"] + shift
    labels = np.array(["C", "U", "R", "M"] * (n // 4), dtype=object)
    d_t, _ = dd.zstats(f["b1"], f["se1"], f["b2"], f["se2"])
    nulls = []
    for _ in range(5):
        g = _fit(rng, n, noise)
        nulls.append(dd.zstats(g["b1"], g["se1"], g["b2"], g["se2"])[0])
    groups = {"ALL": np.ones(n, bool), "R+M": (labels == "R") | (labels == "M")}
    mt = dd.group_msq(d_t, labels, groups)
    mn = [dd.group_msq(d, labels, groups) for d in nulls]
    for g in groups:
        ex = dd.excess_variance(mt[g]["msq"], [m[g]["msq"] for m in mn])
        assert ex["drift_sd"] == pytest.approx(sd_drift, rel=0.05)
        assert ex["rms_null"] == pytest.approx(noise * np.sqrt(2), rel=0.03)
        assert ex["z_excess"] > 5
    assert dd.excess_variance(1e-4, [2e-4, 2e-4, 2e-4])["drift_sd"] == 0.0           # negative excess clips to 0
    assert dd.pp_per_copy(0.02) == pytest.approx(0.5)


def test_analyse_scope_end_to_end_with_and_without_drift():
    rng = np.random.default_rng(2)
    n = 1200
    lab = dict(rarity=np.array((["C"] * 4 + ["U"] * 3 + ["R"] * 2 + ["M"]) * (n // 10), dtype=object), basic=np.zeros(n, bool))
    lab["basic"][:5] = True
    noise = 0.02
    nulls = [_fit(rng, n, noise) for _ in range(8)]
    for t in (_fit(rng, n, noise), ):
        a = dd.analyse_scope(t, nulls, lab)
        assert a["n_cards"] == n - 5
        assert a["verdict_chi2"]["ratio"] == pytest.approx(1.0, abs=0.15) and a["verdict_chi2"]["p_emp"] > 0.1
        assert a["groups"]["ALL"]["drift_sd"] < 0.01
    t = _fit(rng, n, noise)
    t["b2"] = t["b2"] + rng.normal(0, 0.04, n)
    a = dd.analyse_scope(t, nulls, lab)
    assert a["verdict_chi2"]["ratio"] > 3 and a["verdict_chi2"]["p_emp"] == pytest.approx(1 / 9)
    assert a["groups"]["ALL"]["drift_sd"] == pytest.approx(0.04, rel=0.12)
    assert a["groups"]["T3"]["drift_sd_pp"] == pytest.approx(25 * a["groups"]["T3"]["drift_sd"])


def test_direction_null_is_negative_and_reflects_regression_to_the_mean():
    rng = np.random.default_rng(3)
    n = 3000
    f = _fit(rng, n, 0.03)
    c = dd.corr_slope(f["b1"], f["b2"] - f["b1"])
    assert c["corr"] < -0.2                                           # pure noise in the first half already gives a negative corr
    x = rng.normal(size=500)
    assert dd.corr_slope(x, 2 * x + 1)["slope"] == pytest.approx(2.0)


def test_pooling_helpers():
    p = dd.fisher_combine([0.5, 0.5, 0.5])
    assert p["k"] == 3 and 0.3 < p["p"] < 1
    assert dd.fisher_combine([0.001] * 5)["p"] < 1e-8
    s = dd.pooled_null_sums({"A": [1, 2, 3], "B": [10, 20, 30, 40]})
    assert list(s) == [11, 22, 33]


def test_exposure_tertiles_equal_sizes():
    expo = np.arange(30, dtype=float)
    elig = np.ones(30, bool)
    elig[:3] = False
    t = dd.exposure_tertiles(expo, elig)
    assert [int((t == k).sum()) for k in ("T1", "T2", "T3")] == [9, 9, 9] and int((t == "none").sum()) == 3
    assert set(np.nonzero(t == "T3")[0]) == set(range(21, 30))
