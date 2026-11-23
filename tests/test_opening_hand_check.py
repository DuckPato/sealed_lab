"""scripts/opening_hand_check.py: centring and regression arithmetic on synthetic data (no corpus, no model)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))
_SPEC = importlib.util.spec_from_file_location("opening_hand_check", _SCRIPTS / "opening_hand_check.py")
oh = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(oh)


def _synthetic(seed=0, n_decks=300, games_per=6, n_card=12, deck_size=40):
    """Decks of n_card+1 columns (the last = basics); 7-card hands dealt without replacement; true psi per card."""
    rng = np.random.default_rng(seed)
    psi = rng.normal(0, 0.3, n_card)
    rows_x, rows_n, deck_id, y = [], [], [], []
    for d in range(n_decks):
        copies = np.zeros(n_card + 1, int)
        pick = rng.choice(n_card, size=8, replace=False)
        copies[pick] = rng.integers(1, 3, size=8)
        copies[n_card] = deck_size - copies.sum()
        cards = np.repeat(np.arange(n_card + 1), copies)
        base = rng.normal(0, 0.5)
        for _ in range(games_per):
            hand = np.bincount(rng.permutation(cards)[:7], minlength=n_card + 1)
            eta = base + hand[:n_card] @ psi
            y.append(rng.random() < 1 / (1 + np.exp(-eta)))
            rows_x.append(hand)
            rows_n.append(copies)
            deck_id.append(d)
    return (sp.csr_matrix(np.array(rows_x, float)), sp.csr_matrix(np.array(rows_n, float)), np.full(len(y), float(deck_size)),
            np.array(deck_id), np.array(y, float), psi)


def test_design_centre_rows_sum_to_zero_and_mean_is_zero():
    X, ND, N, did, y, psi = _synthetic()
    Z = oh.centre_design(X, ND, N)
    assert np.allclose(np.asarray(Z.sum(axis=1)).ravel(), 0.0, atol=1e-12)            # hand size 7 incl. basics, deck size N
    assert np.abs(np.asarray(Z.mean(axis=0)).ravel()).max() < 0.1                       # random dealing: E[z] = 0
    ND2 = sp.csr_matrix(ND.toarray() * 2.0)
    Z2 = oh.centre_design(X, ND2, N * 2)                                               # same ratio n / N -> same centring
    assert np.allclose(Z.toarray(), Z2.toarray())


def test_design_centre_uses_the_given_hand_size():
    X = sp.csr_matrix(np.array([[1.0, 0.0, 4.0], [0.0, 1.0, 3.0]]))
    ND = sp.csr_matrix(np.array([[2.0, 0.0, 38.0], [0.0, 1.0, 39.0]]))
    Z = oh.centre_design(X, ND, np.array([40.0, 40.0]), hand=np.array([7.0, 6.0]))
    assert Z.toarray()[0] == pytest.approx([1 - 2 * 7 / 40, 0.0, 4 - 38 * 7 / 40])
    assert Z.toarray()[1] == pytest.approx([0.0, 1 - 6 / 40, 3 - 39 * 6 / 40])


def test_within_centre_removes_deck_means_and_drops_singletons():
    X = sp.csr_matrix(np.array([[1.0, 0], [3.0, 2.0], [5.0, 4.0], [7.0, 7.0], [2.0, 2.0]]))
    grp = np.array([0, 0, 1, 2, 2])
    Z, keep = oh.within_centre(X, grp)
    assert keep.tolist() == [True, True, False, True, True]
    assert Z.toarray() == pytest.approx(np.array([[-1.0, -1.0], [1.0, 1.0], [2.5, 2.5], [-2.5, -2.5]]))


def test_wls_matches_numpy_and_reports_t():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 2))
    w = rng.uniform(0.5, 2.0, 200)
    y = 1.0 + X @ np.array([0.7, -0.2]) + rng.normal(size=200) / np.sqrt(w)
    r = oh.wls(y, X, w)
    A = np.column_stack([np.ones(200), X])
    ref = np.linalg.solve(A.T @ (A * w[:, None]), A.T @ (w * y))
    assert r["coef"] == pytest.approx(ref[1:]) and r["intercept"] == pytest.approx(ref[0])
    assert r["t"][0] == pytest.approx(r["coef"][0] / r["se"][0])
    assert abs(r["coef"][0] - 0.7) < 0.15


def test_fit_hand_recovers_psi_with_centred_design_and_within():
    X, ND, N, did, y, psi = _synthetic(n_decks=900)
    base = np.zeros(len(y))
    Z = oh.centre_design(X, ND, N)
    f = oh.fit_hand(Z, base, y, 1.0)
    # (the generator's deck intercept is unmodelled here, which only adds noise)
    assert oh.corr(f["psi"], psi) > 0.8
    assert f["psi"].shape == (12,) and np.isfinite(f["basic"])
    Zw, keep = oh.within_centre(X, did)
    fw = oh.fit_hand(Zw, base[keep], y[keep], 1.0)
    assert oh.corr(fw["psi"], psi) > 0.8
    assert oh.corr(fw["psi"], f["psi"]) > 0.8
    # nothing in hand matters -> psi shrinks to ~0
    f0 = oh.fit_hand(Z, base, (np.random.default_rng(3).random(len(y)) < 0.5).astype(float), 1.0)
    assert np.abs(f0["psi"]).max() < 0.2


def test_disattenuated_and_compare_on_known_relation():
    assert oh.disattenuated(0.3, 0.3, 0.5, 0.72) == pytest.approx(0.3 / np.sqrt(0.36))
    assert np.isnan(oh.disattenuated(0.3, 0.3, -0.1, 0.5))
    rng = np.random.default_rng(2)
    n = 120
    psi = rng.normal(0, 0.3, n)
    beta = 0.2 * psi + rng.normal(0, 0.01, n)
    c = oh.compare(beta, np.full(n, 0.01), np.full(n, 1000.0), psi, np.full(n, 0.05))
    assert c["n"] == n and c["pearson"] > 0.9 and c["kappa"] == pytest.approx(0.2, abs=0.02) and c["informative"]
    assert c["se_ratio"] == pytest.approx(abs(c["kappa"]) * 0.05 / 0.01)
    c2 = oh.compare(beta, np.full(n, 0.01), np.full(n, 50.0), psi, np.full(n, 0.05))        # below the exposure floor
    assert c2["n"] == 0


def test_pool_mean_sd_and_inverse_variance():
    p = oh.pool([1.0, 3.0, None], [0.1, 0.1, 0.1])
    assert p["n"] == 2 and p["mean"] == pytest.approx(2.0) and p["sd"] == pytest.approx(np.sqrt(2.0))
    assert p["iv"] == pytest.approx(2.0) and p["iv_se"] == pytest.approx(0.1 / np.sqrt(2))
    assert oh.pool([None])["mean"] is None
