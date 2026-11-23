"""scripts/pick_prior_check.py: the pure parts (synthetic picks and numbers, no model fit, no data files)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))
_SPEC = importlib.util.spec_from_file_location("pick_prior_check", _SCRIPTS / "pick_prior_check.py")
pp = importlib.util.module_from_spec(_SPEC)
sys.modules["pick_prior_check"] = pp
_SPEC.loader.exec_module(pp)


def _synthetic_picks(rng, theta, n_picks=6000, pack=6):
    """Picks drawn from a Luce model with strengths exp(theta): CSR packs (offsets, cards, counts) and the taken items."""
    n = len(theta)
    cards, offsets, taken = [], [0], []
    for _ in range(n_picks):
        c = rng.choice(n, size=pack, replace=False)
        p = np.exp(theta[c])
        p /= p.sum()
        taken.append(c[rng.choice(pack, p=p)])
        cards.extend(c.tolist())
        offsets.append(len(cards))
    return np.array(taken), np.array(offsets), np.array(cards, np.int16), np.ones(len(cards), np.int8)


def test_pack_csr_roundtrip():
    mat = np.array([[0, 1, 2], [1, 0, 0], [0, 0, 0], [1, 1, 1]], dtype=np.int8)
    off, cards, cnt = pp.pack_csr(mat)
    assert off.tolist() == [0, 2, 3, 3, 6]
    assert cards.tolist() == [1, 2, 0, 0, 1, 2] and cnt.tolist() == [1, 2, 1, 1, 1, 1]


def test_subset_picks_selects_rows():
    mat = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=np.int8)
    off, cards, cnt = pp.pack_csr(mat)
    o2, c2, n2 = pp.subset_picks(off, cards, cnt, [2, 0])
    assert o2.tolist() == [0, 2, 4] and c2.tolist() == [0, 2, 0, 1]


def test_take_stats_and_ata_signal():
    taken = np.array([0, 0, 1, 2, 2, 2, -1])
    pos = np.array([1, 3, 5, 14, 14, 14, 7])
    n, mean = pp.take_stats(taken, pos, 4)
    assert n.tolist() == [2, 1, 3, 0] and mean[0] == 2.0 and np.isnan(mean[3])
    sig, _ = pp.ata_signal(taken, pos, 4, min_taken=2)
    assert sig[0] == -2.0 and sig[2] == -14.0 and np.isnan(sig[1]) and np.isnan(sig[3])      # early pick = higher signal


def test_luce_fit_recovers_strengths_and_order():
    rng = np.random.default_rng(1)
    theta = np.array([2.0, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5, -2.0])
    taken, off, cards, cnt = _synthetic_picks(rng, theta, n_picks=20000)
    est, takes, appear = pp.luce_fit(taken, off, cards, cnt, len(theta), ridge=1e-3)
    assert np.corrcoef(est, theta)[0, 1] > 0.99
    assert np.all(np.diff(est) < 0)                                       # strict order recovered
    assert np.allclose(est - est.mean(), theta - theta.mean(), atol=0.12)
    assert abs(float(np.nanmean(est))) < 1e-9                              # centred


def test_luce_fit_items_in_no_pack_are_nan_and_counts_matter():
    # item 2 never appears; item 0 appears twice in every pack (count 2) and is taken half the time -> its strength is below the single copy's
    off = np.array([0, 2, 4, 6, 8])
    cards = np.array([0, 1, 0, 1, 0, 1, 0, 1], np.int16)
    cnt = np.array([2, 1, 2, 1, 2, 1, 2, 1], np.int8)
    taken = np.array([0, 1, 0, 1])
    est, _, appear = pp.luce_fit(taken, off, cards, cnt, 3, ridge=1e-6)
    assert np.isnan(est[2]) and appear[2] == 0
    # P(0) = 2 e^t0 / (2 e^t0 + e^t1) = 0.5  ->  e^(t0 - t1) = 0.5
    assert est[0] - est[1] == pytest.approx(np.log(0.5), abs=1e-2)


def test_standardise_and_merge_groups():
    z, ok = pp.standardise([1.0, 2.0, 3.0, np.nan])
    assert ok.tolist() == [True, True, True, False] and z[3] == 0.0 and abs(z[:3].mean()) < 1e-12
    z0, ok0 = pp.standardise([5.0, 5.0, 5.0])
    assert np.all(z0 == 0.0)
    lab = pp.merge_small_groups(["C"] * 7 + ["M"] * 2 + ["R"] * 6, np.ones(15, bool), min_group=6)
    assert set(lab.tolist()) == {"C", "rest", "R"} and (lab == "rest").sum() == 2


def test_fit_map_recovers_group_intercepts_and_slope_and_control():
    rng = np.random.default_rng(2)
    n = 400
    groups = np.where(np.arange(n) % 2 == 0, "C", "R").astype(object)
    z = rng.normal(size=n)
    beta = np.where(groups == "C", -0.02, 0.05) + 0.03 * z
    w = rng.uniform(1, 5, n)
    fm = pp.fit_map(z, beta, w, groups, np.ones(n, bool))
    assert fm["params"]["a[C]"] == pytest.approx(-0.02, abs=1e-9) and fm["params"]["b"] == pytest.approx(0.03, abs=1e-9)
    assert fm["r2"] == pytest.approx(1.0)
    ctrl = pp.fit_map(z, beta, w, groups, np.ones(n, bool), use_signal=False)
    assert "b" not in ctrl["params"] and ctrl["r2"] < fm["r2"]
    fm2 = pp.fit_map(z, beta, w, groups, np.ones(n, bool), per_group_slope=True)
    assert fm2["params"]["b[C]"] == pytest.approx(0.03, abs=1e-9)
    # entries outside the fit mask do not influence the fit but are predicted
    mask = np.arange(n) < 300
    beta_bad = beta.copy()
    beta_bad[300:] += 10.0
    fm3 = pp.fit_map(z, beta_bad, w, groups, mask)
    assert fm3["params"]["b"] == pytest.approx(0.03, abs=1e-9) and len(fm3["pred"]) == n


def test_assign_halves_uses_game_split_then_splits_the_rest():
    ids = [f"d{i}" for i in range(10)]
    starts = list(range(10))
    types = ["P"] * 10
    half, rate = pp.assign_halves(ids, starts, types, game_a=["d0", "d2"], game_b=["d1", "d3"], seed=0)
    assert half["d0"] == "A" and half["d1"] == "B" and half["d2"] == "A" and half["d3"] == "B"
    assert set(half) == set(ids) and rate == pytest.approx(0.4)
    rest = [half[f"d{i}"] for i in range(4, 10)]
    assert rest.count("A") == rest.count("B") == 3                           # time-stratified pairs: halves differ by at most one per format


def test_exposure_tertiles_and_weighted_mse():
    t = pp.exposure_tertiles([5, 1, 3, 9, 7, 2], [True, True, True, True, True, False])
    assert t[5] == "none" and t[1] == "T1" and t[3] == "T3"
    assert pp.weighted_mse([1.0, 2.0, np.nan], [0.0, 0.0, 0.0], [1.0, 3.0, 5.0]) == pytest.approx((1 + 12) / 4)
    assert np.isnan(pp.weighted_mse([1.0], [0.0], [0.0]))


def test_card_boot_diff_sign_and_paired_boot():
    rng = np.random.default_rng(3)
    sq_base = rng.uniform(1, 2, 200)
    sq_prior = sq_base - 0.5
    d, lo, hi = pp.card_boot_diff(sq_base, sq_prior, np.ones(200), n_boot=300)
    assert d == pytest.approx(0.5) and lo > 0 and hi < 0.51 + 0.5
    diff = rng.normal(0.01, 0.1, 5000)
    ev = np.repeat(np.arange(500), 10)
    m, lo2, hi2 = pp.paired_boot(diff, ev, n_boot=300)
    assert lo2 < m < hi2 and m == pytest.approx(diff.mean())


def test_spearman_corr_and_nan_handling():
    assert pp.weighted_corr_spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert pp.weighted_corr_spearman([1, 2, np.nan, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert np.isnan(pp.weighted_corr_spearman([1, 1, 1], [1, 2, 3]))


def test_boot_blocks_point_and_draws_consistent():
    num = np.array([1.0, 2.0, 3.0, 4.0])
    den = np.array([10.0, 10.0, 10.0, 10.0])
    sn, sd, pn, pd_ = pp.boot_blocks(num, den, 500, seed=0)
    assert pn == 10.0 and pd_ == 40.0 and sn.shape == (500,) and np.all(sd == 40.0)
    s = pp.summarize(sn / sd, pn / pd_)
    assert s["lo"] <= s["mean"] <= s["hi"]


def test_thin_ids_share_order_and_determinism():
    ids = [f"e{i}" for i in range(100)]
    assert pp.thin_ids(ids, 1.0, 0) == ids
    a = pp.thin_ids(ids, 0.2, 7)
    assert len(a) == 20 and a == sorted(a, key=ids.index) and a == pp.thin_ids(ids, 0.2, 7) and set(a) <= set(ids)
    assert a != pp.thin_ids(ids, 0.2, 8)
    assert len(pp.thin_ids(ids[:3], 0.01, 0)) == 2
