"""Play/draw card interactions: gamma_c on top of the fitted chain (model "MP", component key "playdraw").

Every game records whether the deck's player was on the play.  With s = +1/2 on the play and -1/2 on the draw the win logit becomes

    logit P(win) = eta_M(game) + sum_c n_c * gamma_c * s(game)

where eta_M is the chain's fitted linear predictor of the game's own stage and gamma has a zero-mean normal prior with ONE SD sigma
chosen by grouped CV over events on the training part (the matchup layer's grid and CV machinery, `opponent.SIGMA_GRID` /
`opponent.cv_sigma`).  The fit pools the games of ALL formats of the training part and all game numbers; games without `on_play` are
left out of the fit and get s = 0 (no shift) at prediction.  gamma_c is the card's logit advantage on the play over the draw.
Because s is symmetric around 0 the chain's own card effects stay their values at 50/50 and q(D) (game 1, play/draw unknown) is
unchanged by construction: the layer only feeds games-2-3 advice ("on the play / on the draw: swap X for Y").  Like the matchup
layer a free per-card main-effect adjustment is fitted next to the gammas (prior SD `opponent.SIGMA_MAIN`), so a play share
different from 50% in the training games cannot leak a card's average effect into gamma; it is not used at prediction.

The interaction columns are `opponent`'s pooled-problem machinery with one "opponent column" scaled by s, so the `Pooled` container,
the CV, the MAP fit and the event bootstrap are shared with it.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.special import expit

from sealed_lab.model import glm
from sealed_lab.model import opponent as om
from sealed_lab.model.design import StageData

log = logging.getLogger(__name__)

MODE = "playdraw"                 # `Pooled.mode`: names the CV scratch file
PP_PER_LOGIT = om.PP_PER_LOGIT
S_PLAY = 0.5


def s_vector(on_play) -> tuple[np.ndarray, np.ndarray]:
    """(s, known): s = +1/2 on the play, -1/2 on the draw, 0 where on_play is missing; `known` = on_play is not null."""
    a = list(on_play.to_list() if hasattr(on_play, "to_list") else on_play)
    known = np.array([x is not None for x in a], dtype=bool)
    s = np.zeros(len(a))
    for i in np.nonzero(known)[0]:
        s[i] = S_PLAY if a[i] else -S_PLAY
    return s, known


def pool_stages(stages: dict[str, StageData], coefs: dict[str, np.ndarray], n_card: int) -> om.Pooled:
    """Pool the games (with a known on_play) of all stages: columns [n_c * s | n_c (main-effect adjustment)],
    offset = each stage's fitted linear predictor."""
    Xs, ys, offs, grp = [], [], [], []
    for name, st in stages.items():
        s, known = s_vector(st.games["on_play"])
        Dc = st.prob.D[:, :n_card].tocsr()[st.prob.idx][known]
        Xs.append(sp.hstack([sp.diags(s[known]) @ Dc, Dc], format="csr"))
        ys.append(st.prob.y[known])
        offs.append(st.prob.eta(coefs[name])[known])
        grp.append(np.char.add(name + ":", st.games["draft_id"].to_numpy().astype(str)[known]))
    X = sp.vstack(Xs, format="csr")
    g = np.unique(np.concatenate(grp), return_inverse=True)[1].astype(np.int32)
    prob = glm.Problem(y=np.concatenate(ys), C=X, offset=np.concatenate(offs))
    return om.Pooled(prob=prob, group=g, mass=np.zeros((n_card, 1)), n_games=prob.n, n_groups=int(g.max()) + 1, mode=MODE, n_int=n_card)


def cv_sigma(pool: om.Pooled, tmpdir: str, n_jobs: int | None, seed: int = 0, cancel: threading.Event | None = None,
             progress=None) -> tuple[float, dict]:
    """Grouped 5-fold CV over events of the prior SD of the gammas (same grid and fold rule as the matchup layer)."""
    return om.cv_sigma(pool, tmpdir, n_jobs, seed=seed, cancel=cancel, progress=progress, label="CV play/draw")


def fit_gamma(pool: om.Pooled, sigma: float, x0: np.ndarray | None = None) -> np.ndarray:
    """MAP coefficients [gammas | main-effect adjustments]; `[:pool.n_int]` are the gammas."""
    return om.fit_deltas(pool, sigma, x0=x0)


def predict_extra(Dc: sp.csr_matrix, s: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """Per game sum_c n_c gamma_c s (logit shift); s = 0 where on_play is unknown."""
    return np.asarray(Dc @ np.asarray(gamma)).ravel() * s


@dataclass
class PlayDrawModel:
    """Deployed play/draw layer: gamma draws (draw 0 = full fit) over the full card index."""
    gamma: np.ndarray            # (B, n_cards_total)
    sigma: float

    def effect(self, on_play: bool, draws: bool = True) -> np.ndarray:
        """Per-card logit shift of a game on the play (+gamma/2) or on the draw (-gamma/2): (B, n_cards) or (n_cards,) for draw 0."""
        sgn = S_PLAY if on_play else -S_PLAY
        return sgn * (self.gamma if draws else self.gamma[0])


def fit_final(stages: dict[str, StageData], coefs: dict[str, np.ndarray], n_card: int, nb_ids: np.ndarray, n_cards_total: int,
              sigma: float, n_boot: int, tmpdir: str, n_jobs: int | None, seed: int = 0, cancel: threading.Event | None = None,
              progress=None) -> tuple[PlayDrawModel, dict]:
    """Deployed layer: gammas on all games of all stages (offset = the final chain fits) with the event bootstrap for the draws."""
    pool = pool_stages(stages, coefs, n_card)
    coef0 = fit_gamma(pool, sigma)
    draws = om.boot_draws(pool, sigma, coef0, n_boot, "pd_final.npz", tmpdir, n_jobs, seed, cancel, progress, "Bootstrap (play/draw)")
    full = np.zeros((n_boot + 1, n_cards_total))
    full[:, nb_ids] = draws[:, :n_card]
    info = dict(n_games=int(pool.n_games), n_events=int(pool.n_groups), sigma=float(sigma),
                sd_pp=float(PP_PER_LOGIT * np.sqrt(np.mean(coef0[:n_card] ** 2))))
    return PlayDrawModel(gamma=full, sigma=float(sigma)), info


def holdout(fmt: str, stages: dict, fits: dict, n_card: int, prob_ho, g_ho, tmp: str, n_jobs, seed: int, cancel) -> dict:
    """Gammas fitted on the pooled training games (offset = the chain's own fitted predictor), scored on the hold-out games of `fmt`
    with each game's own on_play.  Returns the predicted win probabilities, sigma, the CV curve and the raw `_gamma` / `_se` /
    `_expo` arrays that `finalize_summary` turns into a table."""
    coefs = {n: f.coef for n, f in fits.items()}
    pool = pool_stages(stages, coefs, n_card)
    sg, info = cv_sigma(pool, tmp, n_jobs, seed=seed + 43, cancel=cancel)
    coef = fit_gamma(pool, sg)
    s_ho, _known = s_vector(g_ho["on_play"])
    Dc = prob_ho.D[:, :n_card].tocsr()[prob_ho.idx]
    eta = prob_ho.eta(fits[fmt].coef) + predict_extra(Dc, s_ho, coef[:n_card])
    A = np.zeros((n_card, pool.prob.ncols))
    A[np.arange(n_card), np.arange(n_card)] = 1.0
    _, _, _, c_rob = glm.cluster_se(pool.prob, coef, pool.sigma_vector(sg), pool.group, contrasts=A)
    expo = np.asarray(abs(pool.prob.C[:, :n_card]).sum(0)).ravel()           # sum of |s| n_c = (copies x games) / 2
    return dict(pred=expit(eta), sigma=float(sg), cv=info, n_pool_games=int(pool.n_games), n_pool_events=int(pool.n_groups),
                _gamma=coef[:n_card], _se=np.asarray(c_rob).ravel(), _expo=2.0 * expo)


def finalize_summary(o: dict, names: list[str], n_top: int = 15) -> dict:
    """Replace the raw arrays of `holdout` by a JSON-able table: the strongest play/draw cards (play minus draw, pp)."""
    g, se, expo = o.pop("_gamma"), o.pop("_se"), o.pop("_expo")
    z = g / np.maximum(se, 1e-12)
    seen = expo > 0
    order = [int(j) for j in np.argsort(-np.abs(z)) if seen[j]][:n_top]
    o["table"] = dict(n_cards=int(seen.sum()), n_abs_z3=int((np.abs(z[seen]) > 3).sum()), expected_abs_z3=float(seen.sum() * 0.0027),
                      sd_pp=float(PP_PER_LOGIT * np.sqrt(np.mean(g[seen] ** 2))) if seen.any() else 0.0,
                      top=[dict(card=names[j], pp=float(PP_PER_LOGIT * g[j]), lo_pp=float(PP_PER_LOGIT * (g[j] - 1.96 * se[j])),
                                hi_pp=float(PP_PER_LOGIT * (g[j] + 1.96 * se[j])), exposure=float(expo[j])) for j in order])
    return o
