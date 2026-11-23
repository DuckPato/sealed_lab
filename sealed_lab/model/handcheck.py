"""Opening-hand check: psi_c = effect of holding card c in the opening hand, given the fitted deck model.

Logistic regression with offset = fitted logit of the full model, regressors = opening-hand copies of each card,
no-mulligan games of all stages pooled.  A card whose deck effect beta is out of line with its in-hand effect psi
(scaled by the cross-card relation kappa) is flagged: its beta may be confounded by archetype / pilot rather than
caused by the card."""
from __future__ import annotations

import logging

import numpy as np
import polars as pl
import scipy.sparse as sp
from scipy.stats import spearmanr

from sealed_lab.model import glm
from sealed_lab.model.design import DeckSpace

log = logging.getLogger(__name__)

PSI_SIGMA = 1.0


def compute_psi(parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]], hands: pl.LazyFrame, space: DeckSpace,
                sigma: float = PSI_SIGMA) -> dict:
    """`parts` = per stage (game_key, fitted logit, won) for the no-mulligan games. Returns dict(psi, se, n_games)."""
    n_card = space.n_card
    if not parts or sum(len(p[0]) for p in parts) == 0:
        return dict(psi=np.zeros(n_card), se=np.full(n_card, sigma), n_games=0)
    keys = np.concatenate([p[0] for p in parts]).astype(np.int64)
    off = np.concatenate([p[1] for p in parts]).astype(float)
    y = np.concatenate([p[2] for p in parts]).astype(float)
    gdf = pl.DataFrame({"game_key": keys, "row": np.arange(len(keys), dtype=np.int64)})
    h = (hands.filter(pl.col("opening") > 0).select("game_key", "card_id", "opening")
         .join(gdf.lazy(), on="game_key", how="inner").collect())
    row = h["row"].to_numpy()
    col = space.col_of_card[h["card_id"].to_numpy().astype(np.int64)]
    ok = col >= 0
    X = sp.csr_matrix((h["opening"].to_numpy().astype(float)[ok], (row[ok], col[ok])), shape=(len(keys), n_card))
    psi, se = fit_psi(X, off, y, sigma)
    return dict(psi=psi, se=se, n_games=int(len(keys)))


def fit_psi(X: sp.csr_matrix, offset: np.ndarray, y: np.ndarray, sigma: float = PSI_SIGMA) -> tuple[np.ndarray, np.ndarray]:
    n, p = X.shape
    C = sp.hstack([sp.csr_matrix(np.ones((n, 1))), X], format="csr")
    prob = glm.Problem(y=y, C=C, offset=offset)
    sig = np.concatenate([[10.0], np.full(p, sigma)])
    res = glm.fit(prob, np.zeros(p + 1), sig, maxiter=300)
    se = glm.posterior_se(prob, res.coef, sig)
    return res.coef[1:], se[1:]


def kappa_and_flags(beta: np.ndarray, beta_se: np.ndarray, psi: np.ndarray, psi_se: np.ndarray, exposure: np.ndarray,
                    min_exposure: float = 200.0) -> dict:
    """Relation beta ~ kappa * psi (precision-weighted, through the origin) over cards with exposure, and flags.

    Cards are flagged when |beta - kappa psi| > 2 sqrt(SE^2 + tau^2), tau = between-card spread. The flags are only meaningful if the opening-hand effect
    actually tracks the deck effect (kappa > 2 SE); otherwise the check has no power and nothing is flagged
    (`informative` False) - the report then says so instead of flagging every card."""
    m = exposure >= min_exposure
    out = dict(kappa=None, kappa_se=None, spearman=None, n=int(m.sum()), informative=False)
    flag = np.zeros(len(beta), dtype=bool)
    if m.sum() >= 8:
        w = 1.0 / (beta_se[m] ** 2 + 1e-12)
        den = float((w * psi[m] ** 2).sum())
        kappa = float((w * beta[m] * psi[m]).sum() / den) if den > 0 else 0.0
        kse = float(1.0 / np.sqrt(den)) if den > 0 else float("inf")
        out["kappa"], out["kappa_se"] = kappa, kse
        try:
            out["spearman"] = float(spearmanr(beta[m], psi[m])[0])
        except Exception:  # noqa: BLE001
            out["spearman"] = None
        out["informative"] = bool(kappa > 2.0 * kse)
        if out["informative"]:
            # Honest cards scatter around beta = kappa psi for real reasons (a 6-drop and a 1-drop gain differently
            # from being in the opening hand), so the test allows a between-card SD tau, estimated by the
            # DerSimonian-Laird moment estimator; only cards outside that spread are flagged.
            var = beta_se ** 2 + (kappa * psi_se) ** 2
            e = beta - kappa * psi
            wv = 1.0 / np.maximum(var[m], 1e-12)
            q = float((wv * e[m] ** 2).sum())
            c = float(wv.sum() - (wv ** 2).sum() / wv.sum())
            tau2 = max(0.0, (q - (int(m.sum()) - 1)) / c) if c > 0 else 0.0
            out["tau"] = float(np.sqrt(tau2))
            flag = m & (np.abs(e) > 2.0 * np.sqrt(var + tau2))
    out["flag"] = flag
    return out
