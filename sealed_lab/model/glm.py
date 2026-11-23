"""Penalised logistic regression (L-BFGS-B, analytic gradient) on a two-level sparse design, plus the shared
spawn-process parallel runner used by the model code.

Design.  Every game i has a *covariate row* C[i] (sparse, ~10 non-zeros) and belongs to a *deck* idx[i]; the deck
part D[idx[i]] (sparse card copies + dense structure features) is shared by all games of that deck.  The linear
predictor is eta = C @ g + (D @ b)[idx] + offset, i.e. X = [C, G D] with G the game->deck incidence matrix - the
classical sparse CSR design, just stored without repeating the deck rows (6x less memory/time on real data).
With D=None the problem is an ordinary sparse-X logistic regression.

Objective (sum over games, not mean):
    sum_i w_i [log(1+exp(eta_i)) - y_i eta_i]  +  sum_j (c_j - m_j)^2 / (2 sigma_j^2)
solved in whitened coordinates z = (c - m)/sigma so that all penalties are z'z/2.
"""
from __future__ import annotations

import concurrent.futures as cf
import contextlib
import logging
import multiprocessing
import os
import threading
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator

import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize
from scipy.special import expit

from sealed_lab.procguard import watch_parent
from sealed_lab.types import Cancelled

log = logging.getLogger(__name__)

THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "NUMEXPR_NUM_THREADS")


# ---------------------------------------------------------------------------------------------
# problem container
# ---------------------------------------------------------------------------------------------

@dataclass
class Problem:
    y: np.ndarray
    C: sp.csr_matrix
    D: sp.csr_matrix | None = None
    idx: np.ndarray | None = None
    offset: np.ndarray | None = None

    @property
    def n(self) -> int:
        return int(self.y.shape[0])

    @property
    def k(self) -> int:
        return int(self.C.shape[1])

    @property
    def p(self) -> int:
        return 0 if self.D is None else int(self.D.shape[1])

    @property
    def ncols(self) -> int:
        return self.k + self.p

    def eta(self, coef: np.ndarray) -> np.ndarray:
        e = self.C @ coef[:self.k]
        if self.D is not None:
            e = e + (self.D @ coef[self.k:])[self.idx]
        if self.offset is not None:
            e = e + self.offset
        return e

    def rmatvec(self, r: np.ndarray) -> np.ndarray:
        g = np.empty(self.ncols)
        g[:self.k] = self.C.T @ r
        if self.D is not None:
            rd = np.bincount(self.idx, weights=r, minlength=self.D.shape[0])
            g[self.k:] = self.D.T @ rd
        return g

    def subset(self, mask: np.ndarray) -> "Problem":
        rows = np.nonzero(mask)[0]
        return Problem(y=self.y[rows], C=self.C[rows], D=self.D,
                       idx=None if self.idx is None else self.idx[rows],
                       offset=None if self.offset is None else self.offset[rows])


def nll(prob: Problem, coef: np.ndarray, w: np.ndarray | None = None) -> float:
    """Weighted negative log-likelihood (sum)."""
    e = prob.eta(coef)
    l = np.logaddexp(0.0, e) - prob.y * e
    return float(l.sum() if w is None else np.dot(w, l))


@dataclass
class FitResult:
    coef: np.ndarray
    loss: float
    n_iter: int
    converged: bool
    message: str = ""


def fit(prob: Problem, prior_mean: np.ndarray, prior_sigma: np.ndarray, w: np.ndarray | None = None,
        x0: np.ndarray | None = None, maxiter: int = 600, gtol: float = 1e-7, ftol: float = 1e-13) -> FitResult:
    """MAP estimate of the penalised logistic regression. `w` are non-negative sample weights."""
    if w is None:
        w = np.ones(prob.n)
    if (w <= 0).mean() > 0.2:
        keep = w > 0
        prob = prob.subset(keep)
        w = w[keep]
    y = prob.y
    m0 = np.asarray(prior_mean, dtype=float)
    sig = np.asarray(prior_sigma, dtype=float)
    scale = max(float(w.sum()), 1.0)
    # Jacobi preconditioner: z-coordinates scaled by the (approximate, p=1/2) posterior curvature of each column
    csq = np.asarray(prob.C.multiply(prob.C).T @ w).ravel()
    if prob.D is not None:
        wd = np.bincount(prob.idx, weights=w, minlength=prob.D.shape[0])
        csq = np.concatenate([csq, np.asarray(prob.D.multiply(prob.D).T @ wd).ravel()])
    a = 1.0 / np.sqrt(0.25 * csq + 1.0 / sig ** 2)
    pen = (a / sig) ** 2

    def fg(z: np.ndarray):
        c = m0 + a * z
        e = prob.eta(c)
        l = np.logaddexp(0.0, e) - y * e
        r = w * (expit(e) - y)
        g = prob.rmatvec(r)
        f = (np.dot(w, l) + 0.5 * np.dot(pen * z, z)) / scale
        return f, (a * g + pen * z) / scale

    z0 = np.zeros_like(m0) if x0 is None else (np.asarray(x0, dtype=float) - m0) / a
    res = minimize(fg, z0, jac=True, method="L-BFGS-B",
                   options=dict(maxiter=maxiter, maxcor=25, ftol=ftol, gtol=gtol, maxfun=maxiter * 2))
    coef = m0 + a * res.x
    return FitResult(coef=coef, loss=float(res.fun * scale), n_iter=int(res.nit), converged=bool(res.success),
                     message=str(res.message))


def gradient(prob: Problem, coef, prior_mean, prior_sigma, w=None) -> np.ndarray:
    """Analytic gradient of the (un-scaled) objective in coefficient space (for tests)."""
    if w is None:
        w = np.ones(prob.n)
    e = prob.eta(coef)
    return prob.rmatvec(w * (expit(e) - prob.y)) + (coef - prior_mean) / prior_sigma ** 2


def objective(prob: Problem, coef, prior_mean, prior_sigma, w=None) -> float:
    if w is None:
        w = np.ones(prob.n)
    return nll(prob, coef, w) + float(np.sum((coef - prior_mean) ** 2 / (2 * prior_sigma ** 2)))


def hessian(prob: Problem, coef: np.ndarray, prior_sigma: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Dense Hessian X'WX + diag(1/sigma^2) of the penalised objective (W = w p (1-p))."""
    if w is None:
        w = np.ones(prob.n)
    e = prob.eta(coef)
    pr = expit(e)
    Wg = w * pr * (1 - pr)
    k, p = prob.k, prob.p
    H = np.zeros((k + p, k + p))
    WC = sp.diags(Wg) @ prob.C
    H[:k, :k] = (prob.C.T @ WC).toarray()
    if prob.D is not None:
        m = prob.D.shape[0]
        G = sp.csr_matrix((np.ones(prob.n), (prob.idx, np.arange(prob.n))), shape=(m, prob.n))
        A = G @ WC                                                    # (m, k) sparse
        Hcd = (A.T @ prob.D).toarray()
        H[:k, k:] = Hcd
        H[k:, :k] = Hcd.T
        wd = np.bincount(prob.idx, weights=Wg, minlength=m)
        H[k:, k:] = (prob.D.T @ (sp.diags(wd) @ prob.D)).toarray()
    H[np.diag_indices_from(H)] += 1.0 / np.asarray(prior_sigma) ** 2
    return H


# The dense linear algebra of the standard errors (Cholesky, inverse, p x p products) runs one call at a time: each call already
# uses all cores through OpenBLAS, and this OpenBLAS build crashed the process when called from several Python threads at once.
_DENSE_LOCK = threading.RLock()


def posterior_se(prob: Problem, coef: np.ndarray, prior_sigma: np.ndarray, w: np.ndarray | None = None,
                 max_dense: int = 6000) -> np.ndarray:
    with _DENSE_LOCK:
        return _posterior_se(prob, coef, prior_sigma, w, max_dense)


def _posterior_se(prob: Problem, coef: np.ndarray, prior_sigma: np.ndarray, w: np.ndarray | None = None,
                  max_dense: int = 6000) -> np.ndarray:
    """Approximate (Laplace) standard errors: sqrt(diag(H^-1)). Exact dense inverse (n_cols <= max_dense)."""
    from scipy.linalg import cho_factor, cho_solve

    if prob.ncols > max_dense:
        H = hessian(prob, coef, prior_sigma, w)
        return 1.0 / np.sqrt(np.diag(H))  # lower bound on the SE if too big for the exact route
    H = hessian(prob, coef, prior_sigma, w)
    try:
        cf_ = cho_factor(H, lower=True)
        inv = cho_solve(cf_, np.eye(H.shape[0]))
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(H)
    return np.sqrt(np.maximum(np.diag(inv), 0.0))


def cluster_se(prob: Problem, coef: np.ndarray, prior_sigma: np.ndarray, group: np.ndarray,
               w: np.ndarray | None = None, max_dense: int = 6000, contrasts: np.ndarray | None = None):
    with _DENSE_LOCK:
        return _cluster_se(prob, coef, prior_sigma, group, w, max_dense, contrasts)


def _cluster_se(prob: Problem, coef: np.ndarray, prior_sigma: np.ndarray, group: np.ndarray,
                w: np.ndarray | None = None, max_dense: int = 6000, contrasts: np.ndarray | None = None):
    """(Laplace SE, cluster-robust SE) of every coefficient.

    With `contrasts` (m x ncols matrix of linear combinations of the coefficients) the SEs of those combinations are
    returned as well: (se_lap, se_rob, contrast_se_lap, contrast_se_rob).

    The robust SE is the sandwich H^-1 (sum_g s_g s_g') H^-1 with one score vector s_g per cluster (`group` = draft_id
    code, i.e. all games of an event), so unmodelled deck/pilot effects shared by the games of an event widen it.
    Penalty gradients are deterministic and excluded from the scores."""
    from scipy.linalg import cho_factor, cho_solve

    if w is None:
        w = np.ones(prob.n)
    H = hessian(prob, coef, prior_sigma, w)
    p = H.shape[0]
    try:
        Hinv = cho_solve(cho_factor(H, lower=True), np.eye(p))
    except np.linalg.LinAlgError:
        Hinv = np.linalg.pinv(H)
    se_lap = np.sqrt(np.maximum(np.diag(Hinv), 0.0))
    G = int(group.max()) + 1
    r = w * (expit(prob.eta(coef)) - prob.y)
    n = prob.n
    SC = sp.csr_matrix((r, (group, np.arange(n))), shape=(G, n)) @ prob.C
    blocks = [SC]
    if prob.D is not None:
        m = prob.D.shape[0]
        key = group.astype(np.int64) * m + prob.idx.astype(np.int64)
        uk, inv = np.unique(key, return_inverse=True)
        rsum = np.bincount(inv, weights=r, minlength=len(uk))
        pg, pd = (uk // m).astype(np.int64), (uk % m).astype(np.int64)
        SD = sp.csr_matrix((rsum, (pg, np.arange(len(uk)))), shape=(G, len(uk))) @ prob.D[pd]
        blocks.append(SD)
    S = sp.hstack(blocks, format="csr")
    M = (S.T @ S).toarray()
    cov = Hinv @ M @ Hinv * (G / max(G - 1, 1))
    se_rob = np.sqrt(np.maximum(np.diag(cov), 0.0))
    if contrasts is None:
        return se_lap, se_rob
    L = np.asarray(contrasts, dtype=float)
    c_lap = np.sqrt(np.maximum(((L @ Hinv) * L).sum(1), 0.0))
    c_rob = np.sqrt(np.maximum(((L @ cov) * L).sum(1), 0.0))
    return se_lap, se_rob, c_lap, c_rob


@dataclass
class SigmaSpec:
    """Prior standard deviations: fixed for covariates, `s` for card columns, clip(kappa*s) for structure columns."""
    sig_cov: np.ndarray
    n_card: int
    n_theta: int
    kappa: float = 4.0
    th_lo: float = 0.1
    th_hi: float = 1.0
    n_attr: int = 0                         # card-attribute columns (gamma), after the structure columns
    attr_sigma: float = 0.3                 # their (fixed) prior SD, logit per SD of the deck-level attribute sum
    th_floor: np.ndarray | None = None      # per-feature lower bound of the structure sigma (cross-set spread)
    extra: list = field(default_factory=list)   # trailing blocks [(n_columns, sigma)] after the attributes (pairs, archetypes)
    th_abs: float | None = None             # fixed structure sigma (absolute, standardised columns) instead of clip(kappa*s)

    def theta_sd(self, s: float) -> float:
        """The structure prior SD before the cross-set floor: `th_abs` if set, else clip(kappa*s, th_lo, th_hi)."""
        return float(self.th_abs) if self.th_abs is not None else float(np.clip(self.kappa * s, self.th_lo, self.th_hi))

    def vector(self, s: float) -> np.ndarray:
        th = np.full(self.n_theta, self.theta_sd(s))
        if self.th_floor is not None:
            th = np.maximum(th, self.th_floor)
        parts = [self.sig_cov, np.full(self.n_card, float(s)), th, np.full(self.n_attr, float(self.attr_sigma))]
        parts += [np.full(int(n), float(sg)) for n, sg in self.extra]
        return np.concatenate(parts)

    @property
    def n_extra(self) -> int:
        return int(sum(n for n, _ in self.extra))


# ---------------------------------------------------------------------------------------------
# problem files shared with worker processes
# ---------------------------------------------------------------------------------------------

def save_problem(path, prob: Problem, **extras) -> None:
    d = dict(y=prob.y, C_data=prob.C.data, C_indices=prob.C.indices, C_indptr=prob.C.indptr, C_shape=np.array(prob.C.shape))
    if prob.D is not None:
        d.update(D_data=prob.D.data, D_indices=prob.D.indices, D_indptr=prob.D.indptr, D_shape=np.array(prob.D.shape),
                 idx=prob.idx)
    if prob.offset is not None:
        d["offset"] = prob.offset
    for k, v in extras.items():
        d["x_" + k] = np.asarray(v)
    tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"      # a failed write (e.g. out of memory) must never leave a half file
    try:
        with open(tmp, "wb") as fh:
            np.savez(fh, **d)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def load_problem(path) -> tuple[Problem, dict]:
    z = np.load(path, allow_pickle=False)
    C = sp.csr_matrix((z["C_data"], z["C_indices"], z["C_indptr"]), shape=tuple(z["C_shape"]))
    D = idx = None
    if "D_data" in z.files:
        D = sp.csr_matrix((z["D_data"], z["D_indices"], z["D_indptr"]), shape=tuple(z["D_shape"]))
        idx = z["idx"]
    off = z["offset"] if "offset" in z.files else None
    extras = {k[2:]: z[k] for k in z.files if k.startswith("x_")}
    return Problem(y=z["y"], C=C, D=D, idx=idx, offset=off), extras


_PROB_CACHE: dict[str, tuple[Problem, dict]] = {}
_PROB_LOCK = threading.Lock()


def _get_problem(path: str) -> tuple[Problem, dict]:
    with _PROB_LOCK:
        hit = _PROB_CACHE.get(path)
        if hit is None:
            _PROB_CACHE.clear()                  # one problem at most: stale ones were the largest idle allocations of a worker
            hit = load_problem(path)
            _PROB_CACHE[path] = hit
        return hit


# ---------------------------------------------------------------------------------------------
# worker functions (top level => picklable under spawn)
# ---------------------------------------------------------------------------------------------

def cv_fold_task(task: dict) -> dict:
    """Held-out log loss of one fold for every sigma of the grid (warm-started, strongest prior first)."""
    prob, ex = _get_problem(task["path"])
    f = int(task["fold"])
    w0 = ex["w"]
    # "forward" scheme: folds are consecutive blocks of events in time order (`fold_game`, per game) and block f is scored by a
    # model trained on the blocks BEFORE it only; the default scheme trains on every other fold
    fold = task["fold_game"] if task.get("fold_game") is not None else ex["fold"]
    fwd = task.get("scheme") == "forward"
    tr = ((fold < f) if fwd else (fold != f)) & (w0 > 0)
    va = (fold == f) & (w0 > 0)
    ptr, pva = prob.subset(tr), prob.subset(va)
    wtr, wva = w0[tr], w0[va]
    spec: SigmaSpec = task["spec"]
    pm = task["prior_mean"]
    x = np.asarray(task["x0"] if task.get("x0") is not None else pm, dtype=float)
    loss = []
    vectors = task.get("vectors")           # explicit prior-SD vectors (weakest last) instead of the card-sigma grid
    for s in (task["grid"] if vectors is None else range(len(vectors))):
        res = fit(ptr, pm, spec.vector(s) if vectors is None else vectors[s], w=wtr, x0=x, maxiter=task.get("maxiter", 300))
        x = res.coef
        loss.append(nll(pva, x, wva))
    return dict(fold=f, loss=np.array(loss), weight=float(wva.sum()), n_va=int(va.sum()))


def stack_coef(prob: Problem, coef: np.ndarray, prior_deck: np.ndarray, w: float, sig_cov: np.ndarray,
               weights: np.ndarray | None = None) -> np.ndarray:
    """The stacked estimate q = w*M + (1-w)*P of a fit `coef` of `prob`: the deck part (cards, structure, extras) moves from the
    prior mean `prior_deck` (P) towards the fit (M) by the factor w, and the covariates (skill, play/draw, mulligans, events, time)
    are REFITTED given that deck part (offset), as the prior-only baseline P does. Without the refit the covariates keep the level
    they had next to M's deck effects and the mean prediction is off by a shift (seen in FRA Bo1: w = 0.5 scored worse than both
    w = 0 and w = 1). w = 1 returns `coef` itself."""
    out = np.array(coef, dtype=float, copy=True)
    if w >= 1.0:
        return out
    k = prob.k
    out[k:] = prior_deck + float(w) * (out[k:] - prior_deck)
    off = (prob.D @ out[k:])[prob.idx]
    out[:k] = fit(Problem(y=prob.y, C=prob.C, offset=off), np.zeros(k), np.asarray(sig_cov, dtype=float), w=weights,
                  x0=coef[:k], maxiter=400).coef
    return out


def fit_lambda(prob: Problem, b_prev: np.ndarray, sig_cov: np.ndarray, w: np.ndarray | None = None) -> tuple[float, float]:
    """logit = covs + lambda * (D b_prev): scalar transfer of the previous stage's deck effect. Returns (lam, se)."""
    s_game = (prob.D @ b_prev)[prob.idx]
    C2 = sp.hstack([prob.C, sp.csr_matrix(s_game[:, None])], format="csr")
    p2 = Problem(y=prob.y, C=C2)
    sig = np.concatenate([np.asarray(sig_cov, dtype=float), [10.0]])
    res = fit(p2, np.zeros(len(sig)), sig, w=w, maxiter=300)
    se = posterior_se(p2, res.coef, sig, w=w)
    return float(res.coef[-1]), float(se[-1])


def prior_chain_task(task: dict) -> dict:
    """The prior mean (deck part) of a stage as it would have been known BEFORE the scored block of a time fold.

    `chain`: the prior stages in order, each dict(path, mask (games before the block), spec, sigma, w (stacking), prior0 (first
    stage only), x0, lam_from_draft). Every one is refitted on its masked games at its own chosen sigma / w (the prior of the
    next one being the previous result, times a lambda refitted on the masked games when the previous stage is the draft stage).
    `target`: dict(path, fold_game, sig_cov, lam_from_draft): the target stage's own lambda is fitted on its blocks BEFORE
    `fold` only. Returns dict(fold, prior)."""
    prev = None
    lo, hi = task["lam_clip"]
    for st in task["chain"]:
        prob, ex = _get_problem(st["path"])
        m = st["mask"] & (ex["w"] > 0)
        p, wt = prob.subset(m), ex["w"][m]
        if prev is None:
            prior = np.asarray(st["prior0"], dtype=float)
        elif st["lam_from_draft"]:
            prior = float(np.clip(fit_lambda(p, prev, st["spec"].sig_cov, wt)[0], lo, hi)) * prev
        else:
            prior = prev.copy()
        pm = np.concatenate([np.zeros(p.k), prior])
        coef = fit(p, pm, st["spec"].vector(st["sigma"]), w=wt, x0=st["x0"], maxiter=300).coef
        if st["w"] < 1.0:
            coef = stack_coef(p, coef, prior, st["w"], st["spec"].sig_cov, wt)
        prev = coef[p.k:]
    t = task["target"]
    if t["lam_from_draft"]:
        prob, ex = _get_problem(t["path"])
        m = (t["fold_game"] < int(task["fold"])) & (ex["w"] > 0)
        prior = float(np.clip(fit_lambda(prob.subset(m), prev, t["sig_cov"], ex["w"][m])[0], lo, hi)) * prev
    else:
        prior = prev.copy()
    return dict(fold=int(task["fold"]), prior=prior)


def stack_fold_task(task: dict) -> dict:
    """Held-out loss of one fold at ONE sigma for every stacking weight w of `w_grid` (q = w*M + (1-w)*prior, covariates refitted
    given q's deck part, see `stack_coef`). Same folds / scheme as `cv_fold_task`."""
    prob, ex = _get_problem(task["path"])
    f = int(task["fold"])
    w0 = ex["w"]
    fold = task["fold_game"] if task.get("fold_game") is not None else ex["fold"]
    fwd = task.get("scheme") == "forward"
    tr = ((fold < f) if fwd else (fold != f)) & (w0 > 0)
    va = (fold == f) & (w0 > 0)
    ptr, pva = prob.subset(tr), prob.subset(va)
    wtr, wva = w0[tr], w0[va]
    spec: SigmaSpec = task["spec"]
    pm = task["prior_mean"]
    x0 = np.asarray(task["x0"] if task.get("x0") is not None else pm, dtype=float)
    m = fit(ptr, pm, spec.vector(task["sigma"]), w=wtr, x0=x0, maxiter=task.get("maxiter", 400)).coef
    loss = [nll(pva, stack_coef(ptr, m, pm[ptr.k:], float(w), spec.sig_cov, wtr), wva) for w in task["w_grid"]]
    return dict(fold=f, loss_w=np.array(loss), weight=float(wva.sum()), n_va=int(va.sum()))


def boot_chain_task(task: dict) -> dict:
    """One bootstrap replicate: refit one or two chained stages on event-resampled data (multiplicity weights).

    task["stages"] = list of dict(path, spec, s, x0, prior_mean, deck_from_prev: bool, seed)."""
    prev = None
    prev_k = 0
    coefs = []
    for st in task["stages"]:
        prob, ex = _get_problem(st["path"])
        g = ex["group"]
        ng = int(g.max()) + 1
        rng = np.random.default_rng(st["seed"])
        mult = np.bincount(rng.integers(0, ng, size=ng), minlength=ng).astype(float)
        w = mult[g] * ex["w"]
        pm = np.array(st["prior_mean"], dtype=float)
        if st.get("deck_from_prev") and prev is not None:
            k = prob.k
            pm[k:] = prev[prev_k:]
        res = fit(prob, pm, st["spec"].vector(st["s"]), w=w, x0=st.get("x0"), maxiter=st.get("maxiter", 400))
        coef = res.coef
        if st.get("stack_w") is not None and st["stack_w"] < 1.0:      # q = w M + (1-w) prior, the prior being this replicate's own
            coef = stack_coef(prob, coef, pm[prob.k:], st["stack_w"], st["spec"].sig_cov, w)
        prev, prev_k = coef, prob.k
        coefs.append(coef)
    return dict(rep=task["rep"], coefs=coefs)


# ---------------------------------------------------------------------------------------------
# parallel runner
# ---------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------
# shared worker pool
# ---------------------------------------------------------------------------------------------

def _noop(i):
    return i


class SharedPool:
    """One spawn process pool used by many concurrent callers (threads of the calculation): independent tasks of different model
    fits then keep every core busy instead of each fit starting (and idling) its own small pool.  While it lives the BLAS / OMP
    thread variables of the workers are pinned to 1, exactly as in `run_parallel`."""

    def __init__(self, n_workers: int):
        self.n_workers = max(1, int(n_workers))
        self._saved = {k: os.environ.get(k) for k in THREAD_VARS}
        for k in THREAD_VARS:
            os.environ[k] = "1"
        self.executor = cf.ProcessPoolExecutor(max_workers=self.n_workers, mp_context=multiprocessing.get_context("spawn"),
                                               initializer=watch_parent)
        self._submit_lock = threading.Lock()
        # Start every worker now, one at a time.  The executor starts workers lazily inside submit(), and on Windows starting two
        # processes from two threads at once corrupts the handles handed to the children (multiprocessing keeps the "process
        # being spawned" in a module global) - the workers then die with "WinError 6, invalid handle" and the pool breaks.
        for f in [self.submit(_noop, i) for i in range(self.n_workers)]:
            f.result()

    def submit(self, fn: Callable, task) -> "cf.Future":
        with self._submit_lock:               # submit() may spawn a worker: never from two threads at once
            return self.executor.submit(fn, task)

    def close(self, kill: bool = False) -> None:
        if kill:
            for p in list(getattr(self.executor, "_processes", {}).values()):
                try:
                    p.terminate()
                except Exception:  # noqa: BLE001
                    pass
            self.executor.shutdown(wait=False, cancel_futures=True)
        else:
            self.executor.shutdown(wait=True)
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


_SHARED: SharedPool | None = None


@contextlib.contextmanager
def shared_pool(n_workers: int):
    """Route every `run_parallel` call (from any thread) through one common process pool while the context is open."""
    global _SHARED
    pool = SharedPool(n_workers)
    prev, _SHARED = _SHARED, pool
    try:
        yield pool
    except BaseException:
        _SHARED = prev
        pool.close(kill=True)
        raise
    else:
        _SHARED = prev
        pool.close()


def task_cost(task) -> int:
    """Estimated peak memory (bytes) of one pool task, from the size of the problem file(s) it loads (see `memory`)."""
    from sealed_lab.model import memory

    try:
        if isinstance(task, dict):
            if "path" in task and "fold" in task:                                   # cross-validation fold
                return int(memory.WORKER_BASE + memory.CV_FACTOR * os.path.getsize(task["path"]))
            if "stages" in task:                                                    # bootstrap replicate
                sizes = [os.path.getsize(st["path"]) for st in task["stages"] if isinstance(st, dict) and "path" in st]
                return int(memory.WORKER_BASE + memory.BOOT_FACTOR * max(sizes, default=0))
    except OSError:
        pass
    return int(memory.WORKER_BASE)


def _run_on_shared(pool: SharedPool, fn: Callable, tasks: list, cancel: threading.Event | None,
                   progress: Callable[[float, str], None] | None, label: str) -> Iterator:
    """Run `tasks` on the shared pool, admitting them only while their estimated footprint fits the pool's memory gate
    (`memory.Budget`); a task that dies with a MemoryError is retried once, alone, after the others have finished."""
    from sealed_lab.model import memory

    b = memory.current()
    gate = None if b is None else b.pool
    n = len(tasks)
    costs = [task_cost(t) for t in tasks]
    pending: dict = {}                                   # future -> (index, bytes reserved)
    retry: list[int] = []
    nxt = done = 0

    def finish(f):
        nonlocal done
        i, held = pending.pop(f)
        if gate is not None:
            gate.release(held)
        try:
            res = f.result()
        except BaseException as e:  # noqa: BLE001
            if memory.is_memory_error(e):
                log.warning("%s: a task ran out of memory (estimated %.0f MB); it is retried alone", label or "pool task",
                            costs[i] / memory.MB)
                retry.append(i)
                return None
            raise
        done += 1
        if progress:
            progress(done / n, f"{label} {done}/{n}".strip())
        return (res,)

    try:
        while nxt < n or pending:
            while nxt < n:                               # admit as many tasks as the budget allows
                held = 0 if gate is None else gate.try_acquire(costs[nxt])
                if held is None:
                    if pending:
                        break
                    held = gate.acquire(costs[nxt], cancel)       # nothing of ours in flight: wait for the others to free memory
                pending[pool.submit(fn, tasks[nxt])] = (nxt, held)
                nxt += 1
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            finished, _ = cf.wait(list(pending), timeout=0.25, return_when=cf.FIRST_COMPLETED)
            for f in finished:
                out = finish(f)
                if out is not None:
                    yield out[0]
        if retry:                                        # the memory-failed tasks, one at a time, with the pool to themselves
            ctx = gate.exclusive(cancel) if gate is not None else contextlib.nullcontext()
            with ctx:
                for i in retry:
                    if cancel is not None and cancel.is_set():
                        raise Cancelled()
                    res = pool.submit(fn, tasks[i]).result()
                    done += 1
                    if progress:
                        progress(done / n, f"{label} {done}/{n}".strip())
                    yield res
    finally:
        for f, (_i, held) in list(pending.items()):      # only OUR tasks: the pool keeps serving the other callers
            f.cancel()
            if gate is not None:
                gate.release(held)


def run_parallel(fn: Callable, tasks: Iterable, n_jobs: int | None, cancel: threading.Event | None = None,
                 progress: Callable[[float, str], None] | None = None, label: str = "",
                 ) -> Iterator:
    """Yield results of `fn(task)` as they complete, using a spawn ProcessPoolExecutor.

    Single threads per worker (BLAS/OMP/Polars) are enforced through the environment while the pool lives.
    Raises `Cancelled` (and kills the workers) when `cancel` is set.  Inside `shared_pool()` the tasks go to the shared pool
    instead (`n_jobs` is then ignored and a cancel only drops this call's own pending tasks)."""
    tasks = list(tasks)
    n = len(tasks)
    if n == 0:
        return
    pool = _SHARED
    if pool is not None:
        yield from _run_on_shared(pool, fn, tasks, cancel, progress, label)
        return
    n_jobs = int(n_jobs or os.cpu_count() or 1)
    if n_jobs == 1:                       # one core asked for: no pool, the tasks run here one after the other
        for i, t in enumerate(tasks):
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            res = fn(t)
            if progress:
                progress((i + 1) / n, f"{label} {i + 1}/{n}".strip())
            yield res
        return
    n_workers = max(1, min(n, n_jobs))
    ctx = multiprocessing.get_context("spawn")
    saved = {k: os.environ.get(k) for k in THREAD_VARS}
    for k in THREAD_VARS:
        os.environ[k] = "1"
    ex = cf.ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx, initializer=watch_parent)
    done = 0
    try:
        futs = {ex.submit(fn, t): i for i, t in enumerate(tasks)}
        pending = set(futs)
        while pending:
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            finished, pending = cf.wait(pending, timeout=0.25, return_when=cf.FIRST_COMPLETED)
            for f in finished:
                res = f.result()
                done += 1
                if progress:
                    progress(done / n, f"{label} {done}/{n}".strip())
                yield res
    except BaseException:
        for p in list(getattr(ex, "_processes", {}).values()):
            try:
                p.terminate()
            except Exception:  # noqa: BLE001
                pass
        ex.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        ex.shutdown(wait=True)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _call_indexed(arg: tuple) -> tuple:
    fn, i, task = arg
    return i, fn(task)


def map_parallel(fn: Callable, tasks: Iterable, n_jobs: int | None, cancel: threading.Event | None = None) -> list:
    """`[fn(t) for t in tasks]` computed through `run_parallel` (so inside `shared_pool()` it shares the common pool), results in
    task order."""
    tasks = list(tasks)
    out: list = [None] * len(tasks)
    for i, r in run_parallel(_call_indexed, [(fn, i, t) for i, t in enumerate(tasks)], n_jobs, cancel=cancel):
        out[i] = r
    return out
