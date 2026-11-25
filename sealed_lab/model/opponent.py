"""Opponent matchups: card x opponent-colors interactions delta_{c,o} on top of the fitted chain (model "MO").

Every game records the opponent's colors.  For card c (copies n_c in the deck) and opponent strategy o the win logit becomes

    logit P(win) = eta_M(game) + sum_c n_c * delta_{c,o(game)}

where eta_M is the chain's fitted linear predictor of the game's own stage (covariates, card effects, structure terms) and the
deltas have a zero-mean normal prior with one SD sigma chosen by grouped CV.  The fit pools the games of ALL formats of the
training part (draft is the bulk, Sealed adds a little); the main effects stay as fitted, so only what depends on the opponent
is estimated.  Opponent encodings: "pair" (one-hot: the 10 two-color pairs + "other" for everything else, e.g. 3-color
opponents) and "color" (multi-hot over the 5 colors: an opponent playing W/U/G has 3 active columns); the encoding is chosen by
the CV loss on the training part, never by the hold-out.

Because the main effects were fitted against the field's average opponent, the value of a card against strategy o is its
CENTRED effect  delta°_{c,o} = sum_l (O_o[l] - m_{c,l}) delta_{c,l}  with m_{c,l} the average exposure of card c to opponent
column l; delta°_{c,o} > 0 means the card is better against o than against the average opponent.
"""
from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass

import numpy as np
import polars as pl
import scipy.sparse as sp
from scipy.special import expit
from scipy.stats import norm

from sealed_lab.model import design, glm
from sealed_lab.model.design import OPP_LEVELS, PAIRS, StageData

log = logging.getLogger(__name__)

MODES = ("pair", "colour")
COLOURS = "WUBRG"
SIGMA_GRID = np.geomspace(0.002, 0.08, 6)
SIGMA_MAIN = 0.05              # prior SD of the free per-card main-effect adjustments fitted next to the deltas
N_FOLDS = 5
OPP_P_MIN = 0.90               # P(delta° > 0) needed to call a card better against a strategy
OPP_MIN_PP = 0.50              # pp of game win rate (near 55%) that effect must reach
PP_PER_LOGIT = 100.0 * 0.55 * 0.45


def levels(mode: str) -> list[str]:
    return list(OPP_LEVELS) if mode == "pair" else list(COLOURS)


def opp_vector(opp: str | None, mode: str) -> np.ndarray:
    """Exposure vector O over the columns of `mode` for an opponent's color string (e.g. 'UB', 'WUG')."""
    lv = levels(mode)
    v = np.zeros(len(lv))
    if mode == "pair":
        v[lv.index(design.norm_pair(opp))] = 1.0
    else:
        for c in opp or "":
            if c in COLOURS:
                v[lv.index(c)] = 1.0
    return v


def opp_matrix(opp_colors: np.ndarray, mode: str) -> sp.csr_matrix:
    """(n_games, L) exposure matrix of the opponents' color strings."""
    oc = np.array(["" if x is None else str(x) for x in opp_colors], dtype=str)
    uniq, inv = np.unique(oc, return_inverse=True)
    table = np.array([opp_vector(u, mode) for u in uniq]) if len(uniq) else np.zeros((0, len(levels(mode))))
    return sp.csr_matrix(table[inv]) if len(inv) else sp.csr_matrix((0, len(levels(mode))))


def interaction_matrix(Dc: sp.csr_matrix, O: sp.csr_matrix) -> sp.csr_matrix:
    """(n_games, n_card * L) matrix with entry [i, c * L + l] = copies_c(game i) * O[i, l].  `Dc` = per-game card copies."""
    n, nc = Dc.shape
    L = O.shape[1]
    Dc = Dc.tocsr()
    O = O.tocsr()
    rows_c = np.repeat(np.arange(n), np.diff(Dc.indptr))
    rows, cols, vals = [], [], []
    ol = np.diff(O.indptr)
    max_l = int(ol.max()) if len(ol) else 0
    # O rows are tiny (<= 5 entries): handle the s-th active level of every game at once
    for s in range(max_l):
        has = ol > s
        lev = np.full(n, -1, dtype=np.int64)
        wt = np.zeros(n)
        pos = O.indptr[:-1] + s
        lev[has] = O.indices[pos[has]]
        wt[has] = O.data[pos[has]]
        m = lev[rows_c] >= 0
        rows.append(rows_c[m])
        cols.append(Dc.indices[m].astype(np.int64) * L + lev[rows_c[m]])
        vals.append(Dc.data[m] * wt[rows_c[m]])
    if not rows:
        return sp.csr_matrix((n, nc * L))
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, nc * L))


@dataclass
class Pooled:
    prob: glm.Problem            # y, C = [interaction columns | per-card main-effect adjustments], offset = fitted eta
    group: np.ndarray            # int event code per game
    mass: np.ndarray             # (n_card, L) average exposure m_{c,l} of each card to each opponent column
    n_games: int
    n_groups: int
    mode: str
    n_int: int = 0               # number of delta columns (n_card * L); the rest are the main-effect adjustments

    def sigma_vector(self, sigma: float) -> np.ndarray:
        return np.concatenate([np.full(self.n_int, float(sigma)), np.full(self.prob.ncols - self.n_int, SIGMA_MAIN)])


def stage_interactions(st: StageData, n_card: int, mode: str, games: pl.DataFrame | None = None, D: sp.csr_matrix | None = None,
                       idx: np.ndarray | None = None) -> sp.csr_matrix:
    g = st.games if games is None else games
    D = st.prob.D if D is None else D
    idx = st.prob.idx if idx is None else idx
    Dc = D[:, :n_card].tocsr()[idx]
    return interaction_matrix(Dc, opp_matrix(g["opp_colors"].to_numpy(), mode))


def pool_stages(stages: dict[str, StageData], coefs: dict[str, np.ndarray], n_card: int, mode: str) -> Pooled:
    """Pool the games of all stages: one interaction problem with offset = each stage's fitted linear predictor."""
    Xs, Ms, ys, offs, grp = [], [], [], [], []
    num = np.zeros((n_card, len(levels(mode))))
    den = np.zeros(n_card)
    for name, st in stages.items():
        Dc = st.prob.D[:, :n_card].tocsr()[st.prob.idx]
        O = opp_matrix(st.games["opp_colors"].to_numpy(), mode)
        Xs.append(interaction_matrix(Dc, O))
        Ms.append(Dc)
        ys.append(st.prob.y)
        offs.append(st.prob.eta(coefs[name]))
        grp.append(np.char.add(name + ":", st.games["draft_id"].to_numpy().astype(str)))
        num += np.asarray(Dc.T @ O.todense())
        den += np.asarray(Dc.sum(0)).ravel()
    n_int = n_card * len(levels(mode))
    X = sp.vstack([sp.hstack([x, m_], format="csr") for x, m_ in zip(Xs, Ms)], format="csr")
    g = np.unique(np.concatenate(grp), return_inverse=True)[1].astype(np.int32)
    glob = num.sum(0) / max(den.sum(), 1e-9)
    mass = np.where(den[:, None] > 0, num / np.maximum(den[:, None], 1e-9), glob[None, :])
    prob = glm.Problem(y=np.concatenate(ys), C=X, offset=np.concatenate(offs))
    return Pooled(prob=prob, group=g, mass=mass, n_games=prob.n, n_groups=int(g.max()) + 1, mode=mode, n_int=n_int)


def cv_sigma(pool: Pooled, tmpdir: str, n_jobs: int | None, seed: int = 0, cancel: threading.Event | None = None,
             progress=None, grid: np.ndarray = SIGMA_GRID, label: str | None = None) -> tuple[float, dict]:
    """Grouped 5-fold CV of the prior SD of the deltas (offset fixed). Returns (sigma, info with the loss curve)."""
    ncols = pool.prob.ncols
    path = os.path.join(tmpdir, f"opp_{pool.mode}.npz")
    rng = np.random.default_rng(seed)
    fold_of_group = (rng.permutation(pool.n_groups) % N_FOLDS).astype(np.int8)
    glm.save_problem(path, pool.prob, fold=fold_of_group[pool.group], w=np.ones(pool.n_games), group=pool.group)
    spec = glm.SigmaSpec(sig_cov=np.ones(ncols), n_card=0, n_theta=0)
    vecs = [pool.sigma_vector(s) for s in grid]
    tasks = [dict(path=path, fold=f, spec=spec, prior_mean=np.zeros(ncols), grid=None, vectors=vecs, x0=None, maxiter=200)
             for f in range(N_FOLDS)]
    res = sorted(glm.run_parallel(glm.cv_fold_task, tasks, n_jobs, cancel=cancel, progress=progress, label=label or f"CV opponent ({pool.mode})"),
                 key=lambda r: r["fold"])
    L = np.array([r["loss"] for r in res])
    W = np.array([r["weight"] for r in res])
    tot = L.sum(0) / W.sum()
    best = int(np.argmin(tot))
    # loss improvement of the best sigma over the strongest prior (~ no interactions), per game
    info = dict(mode=pool.mode, grid=[float(g) for g in grid], loss=[float(v) for v in tot], best=best,
                gain_vs_strongest=float(tot[0] - tot[best]))
    return float(grid[best]), info


def fit_deltas(pool: Pooled, sigma: float, w: np.ndarray | None = None, x0: np.ndarray | None = None) -> np.ndarray:
    """MAP coefficients [deltas | main-effect adjustments] of the pooled problem (use `[:pool.n_int]` for the deltas)."""
    ncols = pool.prob.ncols
    return glm.fit(pool.prob, np.zeros(ncols), pool.sigma_vector(sigma), w=w, x0=x0, maxiter=400).coef


def predict_centred(Dc: sp.csr_matrix, O: sp.csr_matrix, delta: np.ndarray, mass: np.ndarray) -> np.ndarray:
    """Per game sum_c n_c delta°_{c,o}: the opponent-specific part only (the opponent-average part belongs to the main effects)."""
    nc, L = mass.shape
    d = np.asarray(delta[:nc * L])
    avg = (d.reshape(nc, L) * mass).sum(1)
    return np.asarray(interaction_matrix(Dc, O) @ d).ravel() - np.asarray(Dc @ avg).ravel()


def opp_contrasts(mass: np.ndarray, mode: str, opps: list[str], n_cols: int | None = None) -> np.ndarray:
    """Matrix mapping the coefficient vector to centred effects delta°_{c,o}: rows ordered (card, opponent); `n_cols` pads
    with zero columns for the main-effect adjustments."""
    nc, L = mass.shape
    n_o = len(opps)
    A = np.zeros((nc * n_o, n_cols or nc * L))
    Ov = np.array([opp_vector(o, mode) for o in opps])            # (n_o, L)
    for c in range(nc):
        blk = Ov - mass[c][None, :]                               # (n_o, L)
        A[c * n_o:(c + 1) * n_o, c * L:(c + 1) * L] = blk
    return A


def centred(delta: np.ndarray, mass: np.ndarray, mode: str, opps: list[str]) -> np.ndarray:
    """Centred effects delta°_{c,o} (logit) for coefficient vector(s) `delta` ((..., n_card*L)) -> (..., n_card, n_opp)."""
    nc, L = mass.shape
    d = np.asarray(delta).reshape(delta.shape[:-1] + (nc, L))
    Ov = np.array([opp_vector(o, mode) for o in opps])            # (n_o, L)
    # sum_l (O_o[l] - m_cl) d_cl = d_c . O_o - d_c . m_c
    return np.einsum("...cl,ol->...co", d, Ov) - np.einsum("...cl,cl->...c", d, mass)[..., None]


def default_opponents() -> list[str]:
    return list(PAIRS)


def effect_table(delta: np.ndarray, se_c: np.ndarray, names: list[str], opps: list[str], exposure: np.ndarray, n_top: int = 20,
                 pp_min: float = OPP_MIN_PP, p_min: float = OPP_P_MIN) -> dict:
    """Centred effects (n_card, n_opp) with SEs -> counts clearing the thresholds and the top rows.

    `delta`: (n_card, n_opp) centred effects in logit, `se_c`: same shape."""
    p = norm.cdf(delta / np.maximum(se_c, 1e-12))
    pp = PP_PER_LOGIT * delta
    ok_pos = (p >= p_min) & (pp >= pp_min)
    ok_neg = (1 - p >= p_min) & (-pp >= pp_min)
    seen = exposure > 0

    def rec(c, o):
        return dict(card=names[c], opp=opps[o], pp=float(pp[c, o]), lo_pp=float(PP_PER_LOGIT * (delta[c, o] - 1.96 * se_c[c, o])),
                    hi_pp=float(PP_PER_LOGIT * (delta[c, o] + 1.96 * se_c[c, o])), p_pos=float(p[c, o]),
                    exposure=float(exposure[c]))
    flat = [(c, o) for c in range(delta.shape[0]) for o in range(delta.shape[1]) if seen[c]]
    z = np.array([delta[c, o] / max(se_c[c, o], 1e-12) for c, o in flat]) if flat else np.zeros(0)
    order = np.argsort(-np.abs(z))
    top = [rec(*flat[j]) for j in order[:n_top]]
    return dict(n_cells=int(len(flat)), n_pos=int((ok_pos & seen[:, None]).sum()), n_neg=int((ok_neg & seen[:, None]).sum()),
                n_abs_z3=int((np.abs(z) > 3).sum()), expected_abs_z3=float(len(z) * 0.0027),
                top=top, sd_pp=float(np.sqrt(np.mean(pp[seen] ** 2))) if seen.any() else 0.0,
                max_pp=float(pp[seen].max()) if seen.any() else 0.0,
                z_sd=float(np.std(z)) if len(z) else 0.0, p_min=p_min, pp_min=pp_min)


def predict_logit_extra(X: sp.csr_matrix, delta: np.ndarray) -> np.ndarray:
    return np.asarray(X @ delta).ravel()


@dataclass
class OppModel:
    """Deployed matchup layer: delta draws (draw 0 = full fit) and what is needed to centre them."""
    mode: str
    delta: np.ndarray            # (B, n_cards_total, L) float
    mass: np.ndarray             # (n_cards_total, L)
    se_c: np.ndarray             # (n_cards_total, n_opp) cluster-robust SE of the centred effects vs the default opponents
    opps: list[str]
    sigma: float

    def effect(self, opp: str | None, draws: bool = True) -> np.ndarray:
        """Centred effect (logit) of every card against opponent colors `opp`: (B, n_cards) or (n_cards,) for draw 0."""
        O = opp_vector(opp, self.mode)
        d = self.delta if draws else self.delta[:1]
        out = np.einsum("bcl,l->bc", d, O) - np.einsum("bcl,cl->bc", d, self.mass)
        return out if draws else out[0]

    def p_positive(self, opp: str) -> np.ndarray:
        """P(centred effect > 0) per card against one of `self.opps` (normal approximation, cluster-robust SE)."""
        j = self.opps.index(opp)
        return norm.cdf(self.effect(opp, draws=False) / np.maximum(self.se_c[:, j], 1e-12))


def boot_draws(pool: Pooled, sigma: float, coef0: np.ndarray, n_boot: int, fname: str, tmpdir: str, n_jobs: int | None, seed: int,
               cancel: threading.Event | None, progress, label: str) -> np.ndarray:
    """(n_boot + 1, ncols) coefficient draws of a pooled layer: row 0 = the full fit `coef0`, rows 1.. = refits on event-resampled
    games (the same event bootstrap as the chain's), each warm-started at `coef0`.  Shared by the matchup and play/draw layers."""
    ncols = pool.prob.ncols
    draws = np.zeros((n_boot + 1, ncols))
    draws[0] = coef0
    if n_boot > 0:
        sigv = pool.sigma_vector(sigma)
        path = os.path.join(tmpdir, fname)
        glm.save_problem(path, pool.prob, group=pool.group, w=np.ones(pool.n_games))
        spec = glm.SigmaSpec(sig_cov=sigv, n_card=0, n_theta=0)
        tasks = [dict(rep=b, stages=[dict(path=path, spec=spec, s=sigma, x0=coef0, prior_mean=np.zeros(ncols),
                                          deck_from_prev=False, seed=seed * 7919 + b * 31 + 5, maxiter=250)])
                 for b in range(1, n_boot + 1)]
        for r in glm.run_parallel(glm.boot_chain_task, tasks, n_jobs, cancel=cancel, progress=progress, label=label):
            draws[r["rep"]] = r["coefs"][0]
    return draws


def fit_final(stages: dict[str, StageData], coefs: dict[str, np.ndarray], space: "design.DeckSpace", n_cards_total: int, mode: str,
              sigma: float, n_boot: int, tmpdir: str, n_jobs: int | None, seed: int = 0, cancel: threading.Event | None = None,
              progress=None) -> tuple[OppModel, dict]:
    """Deployed layer: deltas on all games of all stages (offset = the final chain fits), event bootstrap for the draws,
    cluster-robust SEs of the centred effects. Returns (OppModel over the full card index, summary dict for the report)."""
    nc = space.n_card
    pool = pool_stages(stages, coefs, nc, mode)
    ncols = pool.prob.ncols
    delta0 = fit_deltas(pool, sigma)
    sigv = pool.sigma_vector(sigma)
    draws = boot_draws(pool, sigma, delta0, n_boot, f"opp_final_{mode}.npz", tmpdir, n_jobs, seed, cancel, progress, "Bootstrap (matchups)")
    opps = default_opponents()
    A = opp_contrasts(pool.mass, mode, opps, ncols)
    _, _, _, c_rob = glm.cluster_se(pool.prob, delta0, sigv, pool.group, contrasts=A)
    se_c = c_rob.reshape(nc, len(opps))
    L = pool.mass.shape[1]
    full_delta = np.zeros((n_boot + 1, n_cards_total, L))
    full_delta[:, space.nb_ids, :] = draws[:, :nc * L].reshape(n_boot + 1, nc, L)
    full_mass = np.zeros((n_cards_total, L))
    full_mass[space.nb_ids] = pool.mass
    full_se = np.full((n_cards_total, len(opps)), 1e3)
    full_se[space.nb_ids] = se_c
    model = OppModel(mode=mode, delta=full_delta, mass=full_mass, se_c=full_se, opps=opps, sigma=float(sigma))
    expo = np.asarray(pool.prob.C[:, :nc * L].sum(0)).ravel().reshape(nc, -1).sum(1)
    cen = centred(delta0[:nc * L], pool.mass, mode, opps)
    return model, dict(n_games=int(pool.n_games), n_events=int(pool.n_groups), opps=opps, mode=mode, sigma=float(sigma),
                       _cen=cen, _se=se_c, _expo=expo)
