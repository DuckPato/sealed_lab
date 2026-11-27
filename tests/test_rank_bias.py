"""Pure helpers of scripts/rank_bias_check.py."""
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import rank_bias_check as rb  # noqa: E402


def test_rank_indicators():
    m = rb.rank_indicators(["Bronze", "silver", None, "mythic", "gold"])
    assert m.shape == (5, 5)
    assert m[0].sum() == 0 and m[2].sum() == 0
    assert m[1, 0] == 1 and m[4, 1] == 1 and m[3, 4] == 1


def test_wls_slope_exact_line():
    x = np.linspace(0, 1, 20)
    r = rb.wls_slope(x, 2.0 * x + 1.0, np.ones(20))
    assert abs(r["slope"] - 2.0) < 1e-9 and abs(r["intercept"] - 1.0) < 1e-9 and abs(r["corr"] - 1.0) < 1e-9


def test_pool_inverse_variance():
    r = rb.pool_inverse_variance([1.0, 3.0], [1.0, 1.0])
    assert abs(r["mean"] - 2.0) < 1e-12 and abs(r["se"] - 1 / np.sqrt(2)) < 1e-12


def test_high_rank_share():
    D = sp.csr_matrix(np.array([[1, 0], [1, 1]], dtype=float))     # deck 0 has card 0; deck 1 has both
    idx = np.array([0, 0, 1, 1, 1])
    hi = np.array([1, 0, 1, 1, 0], dtype=float)
    share, n = rb.high_rank_share(D, idx, hi)
    assert np.allclose(n, [5, 3]) and np.allclose(share, [3 / 5, 2 / 3])
