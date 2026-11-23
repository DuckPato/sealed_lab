from __future__ import annotations

import types

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.stats import spearmanr

from sealed_lab.model import cardvalue, glm


def _problem(seed=0, n=3000, k=4, m=300, p=25, with_deck=True):
    rng = np.random.default_rng(seed)
    C = sp.csr_matrix(rng.normal(size=(n, k)))
    D = sp.random(m, p, density=0.3, random_state=seed + 1, format="csr") if with_deck else None
    idx = rng.integers(0, m, n).astype(np.int32) if with_deck else None
    true = rng.normal(size=k + (p if with_deck else 0)) * 0.4
    pr = glm.Problem(y=np.zeros(n), C=C, D=D, idx=idx)
    y = (rng.random(n) < 1 / (1 + np.exp(-pr.eta(true)))).astype(float)
    pr.y = y
    return pr, true, rng


def test_gradient_matches_finite_differences():
    pr, _, rng = _problem()
    nc = pr.ncols
    pm, sg = rng.normal(size=nc) * 0.1, rng.uniform(0.2, 3.0, size=nc)
    w = rng.random(pr.n) + 0.5
    c = rng.normal(size=nc) * 0.2
    g = glm.gradient(pr, c, pm, sg, w)
    eye = np.eye(nc)
    num = np.array([(glm.objective(pr, c + 1e-6 * eye[j], pm, sg, w) - glm.objective(pr, c - 1e-6 * eye[j], pm, sg, w)) / 2e-6
                    for j in range(nc)])
    np.testing.assert_allclose(g, num, rtol=1e-5, atol=1e-5)
    # plain sparse-X problem (no deck level)
    pr2, _, _ = _problem(with_deck=False)
    c2 = rng.normal(size=pr2.ncols) * 0.2
    g2 = glm.gradient(pr2, c2, np.zeros(pr2.ncols), np.ones(pr2.ncols))
    num2 = np.array([(glm.objective(pr2, c2 + 1e-6 * np.eye(pr2.ncols)[j], np.zeros(pr2.ncols), np.ones(pr2.ncols))
                      - glm.objective(pr2, c2 - 1e-6 * np.eye(pr2.ncols)[j], np.zeros(pr2.ncols), np.ones(pr2.ncols))) / 2e-6
                     for j in range(pr2.ncols)])
    np.testing.assert_allclose(g2, num2, rtol=1e-5, atol=1e-5)


def test_hessian_matches_gradient_differences():
    pr, _, rng = _problem(n=1500, m=120, p=10)
    nc = pr.ncols
    pm, sg = np.zeros(nc), np.full(nc, 1.5)
    res = glm.fit(pr, pm, sg)
    assert res.converged
    assert np.abs(glm.gradient(pr, res.coef, pm, sg)).max() < 1e-2
    H = glm.hessian(pr, res.coef, sg)
    eye = np.eye(nc)
    Hn = np.array([(glm.gradient(pr, res.coef + 1e-5 * eye[j], pm, sg) - glm.gradient(pr, res.coef - 1e-5 * eye[j], pm, sg)) / 2e-5
                   for j in range(nc)])
    np.testing.assert_allclose(H, Hn, atol=1e-5)


def test_recovery_and_se_coverage():
    """Large-n recovery of known coefficients; SE are about right (|z| < 3 for nearly every coefficient)."""
    pr, true, _ = _problem(seed=3, n=40000, k=4, m=800, p=30)
    nc = pr.ncols
    sg = np.full(nc, 5.0)
    res = glm.fit(pr, np.zeros(nc), sg)
    assert np.corrcoef(res.coef, true)[0, 1] > 0.98
    se_lap, se_rob = glm.cluster_se(pr, res.coef, sg, group=np.arange(pr.n) // 6)
    z = (res.coef - true) / se_lap
    assert np.mean(np.abs(z) < 2.6) >= 0.9
    assert np.all(se_lap > 0) and np.all(np.isfinite(se_rob))


def test_weights_and_prior_mean_shift():
    pr, _, rng = _problem(seed=5, n=800, m=60, p=6)
    nc = pr.ncols
    w = np.zeros(pr.n)
    w[:400] = 1.0
    a = glm.fit(pr, np.zeros(nc), np.full(nc, 2.0), w=w)
    b = glm.fit(pr.subset(w > 0), np.zeros(nc), np.full(nc, 2.0))
    np.testing.assert_allclose(a.coef, b.coef, atol=1e-4)
    # a very tight prior pins the coefficients to its mean
    pm = rng.normal(size=nc)
    c = glm.fit(pr, pm, np.full(nc, 1e-4))
    np.testing.assert_allclose(c.coef, pm, atol=1e-3)


def test_cv_picks_sensible_sigma(tmp_path):
    """True card effects ~ N(0, 0.1^2): grouped 5-fold CV over the sigma grid should pick a moderate sigma."""
    rng = np.random.default_rng(11)
    n_groups, per, p = 700, 6, 80
    n = n_groups * per
    group = np.repeat(np.arange(n_groups), per).astype(np.int32)
    X = sp.random(n, p, density=0.25, random_state=4, format="csr")
    X.data[:] = 1.0
    beta = rng.normal(size=p) * 0.1
    y = (rng.random(n) < 1 / (1 + np.exp(-(0.1 + X @ beta)))).astype(float)
    C = sp.csr_matrix(np.ones((n, 1)))
    # card columns carried by D = X rows (one deck per game)
    prob = glm.Problem(y=y, C=C, D=X.tocsr(), idx=np.arange(n, dtype=np.int32))
    stage = types.SimpleNamespace(name="draft", n_groups=n_groups, n_games=n, prob=prob, group=group,
                                  enc=types.SimpleNamespace(sigma=np.array([10.0])))
    s, info = cardvalue.select_sigma(stage, np.zeros(p), p, 0, str(tmp_path), n_jobs=2, seed=1)
    assert info["done"] and len(info["grid"]) == 8 and len(info["loss"]) == 8
    assert 0.03 <= s <= 0.35
    best = int(np.argmin(info["loss"]))
    assert 0 < best < 7
    # both extremes are clearly worse than the choice
    assert info["loss"][0] > min(info["loss"]) and info["loss"][-1] > min(info["loss"])


def test_parallel_runner_cancel():
    import threading

    from sealed_lab.types import Cancelled

    ev = threading.Event()
    ev.set()
    with pytest.raises(Cancelled):
        list(glm.run_parallel(glm.cv_fold_task, [dict(path="x", fold=0, spec=None, prior_mean=None, grid=[1.0])], 2, cancel=ev))
