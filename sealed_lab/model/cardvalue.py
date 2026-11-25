"""The empirical-Bayes chain of penalised GLMs: draft (D) -> Sealed bo1 (S1) -> TradSealed bo3 (S3).

Each stage's prior mean for the deck part (card copies + structure features) is the previous stage's estimate
(scaled by a fitted scalar lambda when the stage that precedes it is the draft stage), sigma comes from 5-fold
cross-validation grouped by draft_id, and bootstrap replicates resample events (draft_ids) with the sigma fixed.
"""
from __future__ import annotations

import logging
import os
import shutil
import tempfile
import threading
from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np
import scipy.sparse as sp

from sealed_lab.model import glm
from sealed_lab.model.design import StageData
from sealed_lab.types import Cancelled

log = logging.getLogger(__name__)

N_FOLDS = 5
GRIDS_RANDOM = {"draft": np.geomspace(0.01, 0.40, 8), "bo1": np.geomspace(0.002, 0.25, 8), "bo3": np.geomspace(0.002, 0.25, 8)}
GRIDS = GRIDS_RANDOM
# time-split CV of the Sealed stages: the grid reaches two octaves lower (the old lower edge, 0.002, was the CV's choice in several sets)
GRIDS_TIME = {**GRIDS_RANDOM, "bo1": np.geomspace(0.0005, 0.25, 10), "bo3": np.geomspace(0.0005, 0.25, 10)}
# forward-chaining CV: events in start-time order are cut into N_TIME_FOLDS blocks of about equal numbers of games; block f (f >= 1)
# is scored by a model trained on blocks 0..f-1 only. The draft stage keeps random event folds (its prior is zero, not an earlier stage).
N_TIME_FOLDS = 6
TIME_STAGES = ("bo1", "bo3")
STACK_GRID = tuple(float(v) for v in np.linspace(0.0, 1.0, 11))      # weights w of q = w*M + (1-w)*P
THETA_CLIP = {"draft": (0.2, 1.0), "bo1": (0.05, 0.5), "bo3": (0.05, 0.5)}
# theta_sigma="cv" (Sealed stages): absolute prior SDs of the structure / mana coefficients (standardised columns) searched by the
# same CV as the card sigma, with the card sigma fixed; 0.05 is the rule's usual value (the floor of THETA_CLIP)
THETA_GRID = (0.0063, 0.0125, 0.025, 0.05, 0.1, 0.2)
THETA_SIGMA_MODES = ("rule", "cv")
# prior SD of the card-attribute coefficients gamma (logit per SD of a deck-level attribute sum): loose in the draft stage,
# where the attributes are learned from scratch, tighter around the previous stage's estimate afterwards
ATTR_SIGMA = {"draft": 0.30, "bo1": 0.10, "bo3": 0.10}
LAMBDA_CLIP = (0.0, 1.5)
MIN_GROUPS_FOR_CV = 10
# extra deck-level blocks (card pairs, card x archetype): prior SD grid per unit of the column, chosen by grouped CV
EXTRA_GRID = np.geomspace(0.004, 0.2, 6)
EXTRA_DEFAULT = 0.02

Progress = Callable[[float, str], None]


@dataclass(frozen=True)
class ChainOpts:
    """How the Sealed stages choose their prior strength.

    sigma_cv "time": forward-chaining folds over event start times (see N_TIME_FOLDS) and the extended sigma grid; "random": the
    original random event folds and grid (reproduces the models of earlier versions). stack: also choose, per stage with a prior
    stage, the weight w of q = w*M + (1-w)*P (P = the stage's prior mean: the previous stage's values, rescaled) on the same
    folds; the deployed card values, structure effects and bootstrap draws are those of q. Needs sigma_cv="time".
    leak_free (time folds only): the prior P of every fold is refitted on the prior stages' events that start BEFORE the scored
    block (see `fold_priors`), so neither P nor M of a fold has seen later events; False reuses the prior fitted on all of the
    training data (the default: faster, and better on the hold-out, although its fit has seen the events being scored).
    theta_sigma "rule" (default): the structure terms' prior SD is clip(4 x card sigma, THETA_CLIP) as always; "cv": in the Sealed
    stages it is chosen from THETA_GRID by the same CV folds and loss as the card sigma (card sigma held fixed, chosen first;
    stacking afterwards at both)."""
    sigma_cv: str = "time"
    stack: bool = True
    leak_free: bool = False     # leak-free lost on the hold-out (SOS bo1 -.00021 [-.00034, -.00007] vs leaky); opt-in
    theta_sigma: str = "rule"

    def __post_init__(self):
        if self.sigma_cv not in ("time", "random"):
            raise ValueError(f"sigma_cv must be 'time' or 'random', got {self.sigma_cv!r}")
        if self.theta_sigma not in THETA_SIGMA_MODES:
            raise ValueError(f"theta_sigma must be one of {THETA_SIGMA_MODES}, got {self.theta_sigma!r}")

    def scheme(self, name: str) -> str:
        return "forward" if self.sigma_cv == "time" and name in TIME_STAGES else "random"

    def grid(self, name: str) -> np.ndarray:
        return (GRIDS_TIME if self.scheme(name) == "forward" else GRIDS_RANDOM)[name]

    def stacking(self, name: str) -> bool:
        return bool(self.stack) and self.scheme(name) == "forward"

    def theta_cv(self, name: str) -> bool:
        return self.theta_sigma == "cv" and name in TIME_STAGES

    def key(self) -> dict:
        k = dict(sigma_cv=self.sigma_cv, stack=bool(self.stack), leak_free=bool(self.leak_free))
        if self.theta_sigma != "rule":          # (absent for the rule: the cache keys of earlier results stay valid)
            k["theta_sigma"] = self.theta_sigma
        return k


LEGACY_OPTS = ChainOpts("random", False, False)
LEAKY_OPTS = ChainOpts("time", True, leak_free=False)        # the time-split rule of the first version (prior fitted on all training data)


def time_folds(stage: StageData, k: int = N_TIME_FOLDS) -> np.ndarray:
    """Fold (0..k-1) of every event (group): consecutive blocks in start-time order holding about equal numbers of games."""
    start = np.asarray(stage.group_start).astype("datetime64[us]")
    key = np.where(np.isnat(start), np.iinfo(np.int64).max, start.astype("int64"))
    ng = len(key)
    order = np.lexsort((np.arange(ng), key))
    n_g = np.bincount(np.asarray(stage.group), minlength=ng)[order].astype(float)
    mid = np.cumsum(n_g) - n_g / 2.0
    fold = np.empty(ng, dtype=np.int8)
    fold[order] = np.minimum((k * mid / max(n_g.sum(), 1.0)).astype(np.int64), k - 1)
    return fold


@dataclass
class StageFit:
    name: str
    coef: np.ndarray                 # [covariates | cards | structure]
    se: np.ndarray
    k: int
    sigma: float                     # chosen card sigma
    spec: glm.SigmaSpec
    prior_deck: np.ndarray           # prior mean for the deck part (cards | structure)
    lam: float | None = None         # scalar used to scale the previous stage (None: copied as is / none)
    lam_se: float | None = None
    lam_from: str | None = None
    cv_grid: list[float] = field(default_factory=list)
    cv_loss: list[float] = field(default_factory=list)
    cv_se: list[float] = field(default_factory=list)
    cv_done: bool = False
    se_lap: np.ndarray | None = None     # Laplace (model-based) SE given the prior mean
    se_total: np.ndarray | None = None   # cluster-robust SE incl. the propagated uncertainty of the prior mean
    n_card: int = 0
    n_feat: int = 0
    attr_load: np.ndarray | None = None  # (n_card, n_attr) attribute / sd, or None without attribute columns
    n_extra: int = 0                     # trailing extra columns (card pairs, card x archetype) after the attributes
    extra_cv: list = field(default_factory=list)   # per extra block: dict(name, n, sigma, grid, loss, done)
    beta_se_cond: np.ndarray | None = None   # cluster-robust SE of the TOTAL card effect u + A gamma (given the prior mean)
    beta_se: np.ndarray | None = None        # same, incl. the propagated uncertainty of the prior mean
    converged: bool = True
    n_iter: int = 0
    warnings: list[str] = field(default_factory=list)
    w: float = 1.0                       # stacking weight: coef = w * M + (1-w) * prior (1 = the plain fit M)
    coef_m: np.ndarray | None = None     # the plain fit M when w < 1 (coef then holds the stacked estimate)
    w_grid: list[float] = field(default_factory=list)
    w_loss: list[float] = field(default_factory=list)    # CV loss of each w at the chosen sigma
    cv_scheme: str = "random"
    theta_cv_sd: float | None = None     # theta_sigma="cv": the chosen structure prior SD (None: the rule clip(4 sigma) was used)
    theta_grid: list[float] = field(default_factory=list)
    theta_loss: list[float] = field(default_factory=list)    # CV loss of each grid value at the chosen card sigma
    theta_se: list[float] = field(default_factory=list)

    @property
    def deck(self) -> np.ndarray:
        return self.coef[self.k:]

    @property
    def cov(self) -> np.ndarray:
        return self.coef[:self.k]

    @property
    def n_attr(self) -> int:
        return 0 if self.attr_load is None else self.attr_load.shape[1]

    @property
    def u(self) -> np.ndarray:
        """Card-specific part of the card effects (the whole effect without attribute columns)."""
        return self.coef[self.k:self.k + self.n_card]

    @property
    def theta(self) -> np.ndarray:
        return self.coef[self.k + self.n_card:self.k + self.n_card + self.n_feat]

    @property
    def gamma(self) -> np.ndarray:
        a = self.k + self.n_card + self.n_feat
        return self.coef[a:a + self.n_attr]

    @property
    def extra(self) -> np.ndarray:
        """Coefficients of the trailing extra blocks (pairs first, then card x archetype)."""
        return self.coef[self.coef.shape[0] - self.n_extra:] if self.n_extra else np.zeros(0)

    @property
    def extra_se(self) -> np.ndarray:
        return self.se_total[self.se_total.shape[0] - self.n_extra:] if self.n_extra else np.zeros(0)

    @property
    def beta_attr(self) -> np.ndarray:
        """Attribute-predicted part A.gamma of each card's effect (zeros without attribute columns)."""
        return np.zeros(self.n_card) if self.attr_load is None else self.attr_load @ self.gamma

    @property
    def beta_total(self) -> np.ndarray:
        return self.u + self.beta_attr

    @property
    def card_se(self) -> np.ndarray:
        """SE of the total card effects."""
        if self.beta_se is not None:
            return self.beta_se
        return self.se_total[self.k:self.k + self.n_card]


def theta_sd_of(f: StageFit) -> float:
    """The structure prior SD a fitted stage used (before any cross-set floor): the CV choice or the rule's clip(4 sigma)."""
    return float(f.spec.theta_sd(f.sigma))


def _check(cancel: threading.Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise Cancelled()


def make_spec(stage: StageData, n_card: int, n_feat: int, name: str, n_attr: int = 0,
              theta_floor: np.ndarray | None = None, extra: list | None = None, th_abs: float | None = None) -> glm.SigmaSpec:
    lo, hi = THETA_CLIP[name]
    return glm.SigmaSpec(sig_cov=stage.enc.sigma, n_card=n_card, n_theta=n_feat, kappa=4.0, th_lo=lo, th_hi=hi,
                         n_attr=n_attr, attr_sigma=ATTR_SIGMA[name], th_floor=theta_floor, extra=list(extra or []), th_abs=th_abs)


def prior_vector(k: int, prior_deck: np.ndarray) -> np.ndarray:
    return np.concatenate([np.zeros(k), prior_deck])


def stage_files(stage: StageData, tmpdir: str, seed: int) -> str:
    """Write the stage problem (+ folds, groups, weights) for the worker processes."""
    path = os.path.join(tmpdir, f"problem_{stage.name}.npz")
    if os.path.exists(path):
        return path
    rng = np.random.default_rng(seed)
    ng = stage.n_groups
    fold_of_group = (rng.permutation(ng) % N_FOLDS).astype(np.int8)
    w = getattr(stage, "w", None)
    w = np.ones(stage.n_games) if w is None else w
    glm.save_problem(path, stage.prob, fold=fold_of_group[stage.group], w=w, group=stage.group)
    return path


def _cv_setup(stage: StageData, prior_deck: np.ndarray, scheme: str, tmpdir: str, seed: int, has_prior: bool, fold_prior_fn=None):
    """(problem file, folds, extra task keys, per-fold prior-mean function, per-fold priors or None) of a stage's CV."""
    path = stage_files(stage, tmpdir, seed)
    pm = prior_vector(stage.prob.k, prior_deck)
    extra: dict = {}
    folds = range(N_FOLDS)
    if scheme == "forward":
        extra = dict(scheme="forward", fold_game=time_folds(stage)[np.asarray(stage.group)])
        folds = range(1, N_TIME_FOLDS)
    fp = fold_prior_fn() if (fold_prior_fn is not None and scheme == "forward" and has_prior) else None
    pm_of = (lambda f: prior_vector(stage.prob.k, fp[f])) if fp is not None else (lambda f: pm)
    return path, folds, extra, pm_of, fp


def _theta_and_w(info: dict, name: str, spec: glm.SigmaSpec, s: float, path: str, folds, extra: dict, pm_of, x0, n_jobs, cancel,
                 progress, theta: bool, stack: bool) -> glm.SigmaSpec:
    """The two CV steps after the card sigma `s`: (theta) the structure terms' prior SD from THETA_GRID on the same folds and loss with
    the card sigma fixed (info["theta"]); then (stack) the stacking weight w at those prior SDs (info["w"], ...). Returns the spec with
    the chosen structure SD (th_abs)."""
    if theta:
        vecs = [replace(spec, th_abs=float(g)).vector(s) for g in THETA_GRID]                    # weakest prior last (warm starts)
        t3 = [dict(path=path, fold=f, spec=spec, prior_mean=pm_of(f), grid=None, vectors=vecs, x0=x0, maxiter=250, **extra)
              for f in folds]
        res3 = sorted(glm.run_parallel(glm.cv_fold_task, t3, n_jobs, cancel=cancel, progress=progress, label=f"CV {name} structure"),
                      key=lambda r: r["fold"])
        L3 = np.array([r["loss"] for r in res3])
        W3 = np.array([r["weight"] for r in res3])
        tot3 = L3.sum(0) / W3.sum()
        pf3 = L3 / W3[:, None]
        se3 = pf3.std(0, ddof=1) / np.sqrt(len(res3)) if len(res3) > 1 else np.zeros(len(THETA_GRID))
        b3 = int(np.argmin(tot3))
        info["theta"] = dict(grid=[float(g) for g in THETA_GRID], loss=[float(v) for v in tot3], se=[float(v) for v in se3], best=b3,
                             value=float(THETA_GRID[b3]))
        spec = replace(spec, th_abs=float(THETA_GRID[b3]))
    if stack:
        t2 = [dict(path=path, fold=f, spec=spec, prior_mean=pm_of(f), sigma=s, w_grid=list(STACK_GRID), x0=x0, maxiter=400,
                   **extra) for f in folds]
        res2 = sorted(glm.run_parallel(glm.stack_fold_task, t2, n_jobs, cancel=cancel, progress=progress, label=f"CV {name} stacking"),
                      key=lambda r: r["fold"])
        lw = np.array([r["loss_w"] for r in res2]).sum(0) / np.array([r["weight"] for r in res2]).sum()     # (w,) at the chosen sigma
        wi = len(STACK_GRID) - 1 - int(np.argmin(lw[::-1]))                     # ties go to the larger w (the plain fit M)
        info.update(w=float(STACK_GRID[wi]), w_grid=list(STACK_GRID), w_loss=[float(v) for v in lw])
    return spec


def select_theta_w(stage: StageData, prior_deck: np.ndarray, spec: glm.SigmaSpec, s: float, tmpdir: str, n_jobs: int | None,
                   seed: int = 0, cancel: threading.Event | None = None, progress: Progress | None = None,
                   x0: np.ndarray | None = None, opts: "ChainOpts | None" = None, has_prior: bool = True,
                   fold_prior_fn=None, theta: bool = True, stack: bool = True) -> tuple[glm.SigmaSpec, dict]:
    """The structure-SD step (and the stacking weight after it) of `select_sigma` for a stage whose card sigma `s` and prior are
    already fixed (diagnostics: refit a stage at a given sigma with theta_sigma="cv"). Returns (spec with th_abs, info) with
    info["theta"] / info["w"]; the folds are those of `opts.scheme(stage.name)`."""
    opts = opts or ChainOpts()
    name = stage.name
    scheme = opts.scheme(name)
    info: dict = dict(scheme=scheme)
    path, folds, extra, pm_of, _fp = _cv_setup(stage, prior_deck, scheme, tmpdir, seed, has_prior, fold_prior_fn)
    spec = _theta_and_w(info, name, spec, float(s), path, folds, extra, pm_of, x0, n_jobs, cancel, progress,
                        theta=theta and spec.n_theta > 0, stack=stack and opts.stacking(name) and has_prior)
    return spec, info


def select_sigma(stage: StageData, prior_deck: np.ndarray, n_card: int, n_feat: int, tmpdir: str, n_jobs: int | None,
                 seed: int = 0, cancel: threading.Event | None = None, progress: Progress | None = None,
                 x0: np.ndarray | None = None, n_attr: int = 0, theta_floor: np.ndarray | None = None,
                 extras: list | None = None, opts: "ChainOpts | None" = None, has_prior: bool = True,
                 fold_prior_fn: "Callable[[], dict[int, np.ndarray]] | None" = None) -> tuple[float, dict]:
    """Cross-validation of the stage's prior strength over its log-grid of sigma values. Parallel over folds.

    `opts.sigma_cv` "random": 5 folds of groups of draft_id (the original rule); "time" (Sealed stages): forward chaining over
    event start times, `N_TIME_FOLDS` blocks, block f scored by a model fitted on the earlier blocks only. With stacking (and a
    prior stage: `has_prior`) the same fits also give the held-out loss of q = w*M + (1-w)*P for each w of STACK_GRID at every
    sigma; sigma is chosen first (at w = 1), then w at that sigma (info["w"], info["w_loss"]).
    `fold_prior_fn` (time folds with a prior stage): returns {fold: prior mean of the deck part} fitted without the events of that
    block and later ones (`fold_priors`); without it every fold uses `prior_deck`.
    With `opts.theta_sigma` "cv" (Sealed stages, n_feat > 0) the structure terms' prior SD is chosen right after the card sigma, from
    THETA_GRID on the same folds and loss with the card sigma held fixed (info["theta"]: grid, loss, se, best, value); the
    stacking weight is then chosen at both. Extra blocks (`extras`, [dict(name, n)]) take their default prior SD here; their own SDs
    are chosen afterwards."""
    opts = opts or ChainOpts()
    name = stage.name
    scheme = opts.scheme(name)
    grid = opts.grid(name)
    stack = opts.stacking(name) and has_prior
    spec = make_spec(stage, n_card, n_feat, name, n_attr, theta_floor, [(e["n"], EXTRA_DEFAULT) for e in (extras or [])])
    info = dict(grid=[float(g) for g in grid], loss=[], se=[], done=False, scheme=scheme)
    if stage.n_groups < MIN_GROUPS_FOR_CV or stage.n_games < 50:
        s = float(grid[len(grid) // 2])
        info["note"] = f"only {stage.n_groups} events: cross-validation skipped, default sigma {s:.3f}"
        return s, info
    path, folds, extra, pm_of, fp = _cv_setup(stage, prior_deck, scheme, tmpdir, seed, has_prior, fold_prior_fn)
    info["leak_free"] = fp is not None
    tasks = [dict(path=path, fold=f, spec=spec, prior_mean=pm_of(f), grid=[float(g) for g in grid], x0=x0, maxiter=250, **extra)
             for f in folds]
    res = list(glm.run_parallel(glm.cv_fold_task, tasks, n_jobs, cancel=cancel, progress=progress, label=f"CV {name}"))
    res.sort(key=lambda r: r["fold"])
    L = np.array([r["loss"] for r in res])             # (folds, grid)
    W = np.array([r["weight"] for r in res])
    tot = L.sum(0) / W.sum()
    per_fold = L / W[:, None]
    se = per_fold.std(0, ddof=1) / np.sqrt(len(res)) if len(res) > 1 else np.zeros(len(grid))
    best = int(np.argmin(tot))
    info.update(loss=[float(v) for v in tot], se=[float(v) for v in se], done=True, best=best)
    spec = _theta_and_w(info, name, spec, float(grid[best]), path, folds, extra, pm_of, x0, n_jobs, cancel, progress,
                        theta=opts.theta_cv(name) and n_feat > 0, stack=stack)
    return float(grid[best]), info


def card_contrasts(k: int, n_card: int, n_feat: int, attr_load: np.ndarray, n_extra: int = 0) -> np.ndarray:
    """Matrix L with L @ coef = total card effects u + A gamma (rows = non-basic cards)."""
    n_attr = attr_load.shape[1]
    L = np.zeros((n_card, k + n_card + n_feat + n_attr + n_extra))
    L[np.arange(n_card), k + np.arange(n_card)] = 1.0
    L[:, k + n_card + n_feat:k + n_card + n_feat + n_attr] = attr_load
    return L


def fit_stage(stage: StageData, prior_deck: np.ndarray, spec: glm.SigmaSpec, s: float, x0: np.ndarray | None = None,
              with_se: bool = True, attr_load: np.ndarray | None = None):
    """(coef, se, FitResult); se = (se_lap, se_rob) or, with `attr_load`, also the SEs of the total card effects."""
    k = stage.prob.k
    pm = prior_vector(k, prior_deck)
    sig = spec.vector(s)
    res = glm.fit(stage.prob, pm, sig, w=stage.w, x0=x0 if x0 is not None else pm, maxiter=800)
    se = None
    if with_se:
        L = None if attr_load is None else card_contrasts(k, spec.n_card, spec.n_theta, attr_load, spec.n_extra)
        se = glm.cluster_se(stage.prob, res.coef, sig, stage.group, w=stage.w, contrasts=L)
    return res.coef, se, res


def fit_lambda(stage: StageData, b_prev: np.ndarray) -> tuple[float, float]:
    """logit = covs + lambda * (D b_prev): scalar transfer of the previous stage's deck effect. Returns (lam, se)."""
    return glm.fit_lambda(stage.prob, b_prev, stage.enc.sigma, stage.w)


def fold_priors(stages: dict[str, StageData], fits: dict[str, "StageFit"], chain: list[str], name: str, tmpdir: str,
                n_jobs: int | None, seed: int = 0, cancel: threading.Event | None = None,
                progress: Progress | None = None) -> dict[int, np.ndarray]:
    """{fold: prior mean (deck part) of stage `name`} for the scored blocks (1..N_TIME_FOLDS-1) of its time folds, leak-free.

    The prior of block f is what the earlier stages (`chain`, in order, already fitted in `fits`) would have estimated from their
    events starting BEFORE the first event of block f (and, for lambda, from the target stage's blocks before f): each is refitted
    on those events at the sigma / stacking weight it chose on the full training part (two scalars per stage; the only thing a fold
    keeps from later events). So the P of a fold, and M fitted on top of it, never saw an event they are scored on."""
    tgt = stages[name]
    fold_g = time_folds(tgt)
    fog = fold_g[np.asarray(tgt.group)]
    t_start = np.asarray(tgt.group_start).astype("datetime64[us]")
    tasks = []
    for f in range(1, N_TIME_FOLDS):
        t_f = t_start[fold_g == f].min()
        links = []
        for j, n in enumerate(chain):
            s, ft = stages[n], fits[n]
            gs = np.asarray(s.group_start).astype("datetime64[us]")
            links.append(dict(path=stage_files(s, tmpdir, seed), mask=(gs < t_f)[np.asarray(s.group)], spec=ft.spec, sigma=ft.sigma,
                              w=ft.w, prior0=ft.prior_deck if j == 0 else None,
                              x0=ft.coef_m if ft.coef_m is not None else ft.coef, lam_from_draft=j > 0 and chain[j - 1] == "draft"))
        tasks.append(dict(fold=f, chain=links, lam_clip=LAMBDA_CLIP,
                          target=dict(path=stage_files(tgt, tmpdir, seed), fold_game=fog, sig_cov=tgt.enc.sigma,
                                      lam_from_draft=chain[-1] == "draft")))
    res = glm.run_parallel(glm.prior_chain_task, tasks, n_jobs, cancel=cancel, progress=progress, label=f"Prior per fold ({name})")
    return {r["fold"]: r["prior"] for r in res}


def propagate_se(f: StageFit, prev: "StageFit | None") -> np.ndarray:
    """Standard errors of the deck part that include the uncertainty of the prior mean (analytic, per coefficient).

    Normal-normal shrinkage: variance = SE_cond^2 + w_prior^2 * Var(prior mean), where SE_cond is the cluster-robust
    SE given the prior mean and w_prior = (SE_lap/sigma)^2 is the weight the prior receives."""
    se = f.se.copy()
    if f.beta_se_cond is not None:
        f.beta_se = f.beta_se_cond.copy()
    if prev is None or prev.se_total is None:
        return se
    k = f.k
    sig = f.spec.vector(f.sigma)[k:]
    scale = f.lam if f.lam is not None else 1.0
    w_prior = (f.se_lap[k:] / sig) ** 2
    # With a stacking weight w < 1 the deployed estimate is q = P + w (M - P), M = (1 - w_prior) data + w_prior P: the data carry
    # w (1 - w_prior) of it and the prior the rest, 1 - w (1 - w_prior) (= w_prior at w = 1). SE_cond is M's SE given the prior.
    if f.w >= 1.0:
        se[k:] = np.sqrt(se[k:] ** 2 + w_prior ** 2 * (scale * prev.se_total[prev.k:]) ** 2)
    else:
        se[k:] = np.sqrt((f.w * se[k:]) ** 2 + ((1.0 - f.w * (1.0 - w_prior)) * scale * prev.se_total[prev.k:]) ** 2)
    if f.beta_se_cond is not None:
        # total card effects: the weight the prior has on the card-specific part carries the previous total's uncertainty
        wu = w_prior[:f.n_card]
        if f.w >= 1.0:
            f.beta_se = np.sqrt(f.beta_se_cond ** 2 + wu ** 2 * (scale * prev.card_se) ** 2)
        else:
            f.beta_se = np.sqrt((f.w * f.beta_se_cond) ** 2 + ((1.0 - f.w * (1.0 - wu)) * scale * prev.card_se) ** 2)
    return se


def select_extra_sigma(stage: StageData, prior_deck: np.ndarray, spec: glm.SigmaSpec, s: float, j: int, tmpdir: str,
                       n_jobs: int | None, seed: int = 0, cancel: threading.Event | None = None,
                       progress: Progress | None = None, grid: np.ndarray | None = None) -> tuple[float, dict]:
    """Grouped 5-fold CV of the prior SD of extra block `j` (card sigma `s` and the other blocks fixed)."""
    grid = EXTRA_GRID if grid is None else np.asarray(grid, dtype=float)
    info = dict(grid=[float(g) for g in grid], loss=[], done=False)
    if stage.n_groups < MIN_GROUPS_FOR_CV or stage.n_games < 50:
        info["note"] = f"only {stage.n_groups} events: cross-validation skipped"
        return EXTRA_DEFAULT, info
    path = stage_files(stage, tmpdir, seed)
    pm = prior_vector(stage.prob.k, prior_deck)
    vecs = []
    for g in grid:
        ex = [(n, float(g) if i == j else sg) for i, (n, sg) in enumerate(spec.extra)]
        vecs.append(replace(spec, extra=ex).vector(s))
    tasks = [dict(path=path, fold=f, spec=spec, prior_mean=pm, grid=None, vectors=vecs, x0=None, maxiter=250)
             for f in range(N_FOLDS)]
    res = list(glm.run_parallel(glm.cv_fold_task, tasks, n_jobs, cancel=cancel, progress=progress, label=f"CV {stage.name} extra{j}"))
    res.sort(key=lambda r: r["fold"])
    L = np.array([r["loss"] for r in res])
    W = np.array([r["weight"] for r in res])
    tot = L.sum(0) / W.sum()
    best = int(np.argmin(tot))
    info.update(loss=[float(v) for v in tot], done=True, best=best)
    return float(grid[best]), info


def _stage_fit(stage: StageData, prior_deck: np.ndarray, n_card: int, n_feat: int, tmpdir: str, n_jobs, seed, cancel,
               progress, lam=None, lam_se=None, lam_from=None, attr_load: np.ndarray | None = None,
               theta_floor: np.ndarray | None = None, extras: list | None = None, ref_fit: "StageFit | None" = None,
               opts: "ChainOpts | None" = None, fold_prior_fn=None) -> StageFit:
    """`extras`: [dict(name, n)] trailing blocks; `ref_fit`: the plain chain's fit of this stage (supplies the card sigma, its CV
    curve and the stacking weight, so only the extra blocks' sigmas are cross-validated)."""
    name = stage.name
    n_attr = 0 if attr_load is None else attr_load.shape[1]
    extras = list(extras or [])
    if ref_fit is not None:
        s = ref_fit.sigma
        info = dict(grid=ref_fit.cv_grid, loss=ref_fit.cv_loss, se=ref_fit.cv_se, done=ref_fit.cv_done, scheme=ref_fit.cv_scheme)
        if ref_fit.cv_loss:
            info["best"] = int(np.argmin(ref_fit.cv_loss))
        if ref_fit.w_loss:
            info.update(w=ref_fit.w, w_grid=ref_fit.w_grid, w_loss=ref_fit.w_loss)
        if ref_fit.theta_cv_sd is not None:
            info["theta"] = dict(grid=ref_fit.theta_grid, loss=ref_fit.theta_loss, se=ref_fit.theta_se, value=ref_fit.theta_cv_sd)
    else:
        s, info = select_sigma(stage, prior_deck, n_card, n_feat, tmpdir, n_jobs, seed, cancel, progress, n_attr=n_attr,
                               theta_floor=theta_floor, extras=extras, opts=opts, has_prior=lam_from is not None,
                               fold_prior_fn=fold_prior_fn)
    _check(cancel)
    extra_cv = []
    th_cv = info.get("theta")
    spec = make_spec(stage, n_card, n_feat, name, n_attr, theta_floor, [(e["n"], EXTRA_DEFAULT) for e in extras],
                     th_abs=None if th_cv is None else float(th_cv["value"]))
    for j, e in enumerate(extras):
        _check(cancel)
        sg, einfo = select_extra_sigma(stage, prior_deck, spec, s, j, tmpdir, n_jobs, seed + 7 * (j + 1), cancel, progress,
                                       grid=e.get("grid"))
        spec.extra[j] = (e["n"], sg)
        extra_cv.append(dict(name=e["name"], n=int(e["n"]), sigma=sg, **einfo))
    coef, se_all, res = fit_stage(stage, prior_deck, spec, s, attr_load=attr_load)
    se_lap, se_rob = se_all[0], se_all[1]
    sf = StageFit(name=name, coef=coef, se=se_rob, se_lap=se_lap, k=stage.prob.k, sigma=s, spec=spec, prior_deck=prior_deck, lam=lam,
                  lam_se=lam_se, lam_from=lam_from, cv_grid=info["grid"], cv_loss=info["loss"], cv_se=info["se"],
                  cv_done=info["done"], converged=res.converged, n_iter=res.n_iter, n_card=n_card, n_feat=n_feat,
                  attr_load=attr_load, beta_se_cond=se_all[3] if attr_load is not None else None,
                  n_extra=spec.n_extra, extra_cv=extra_cv, cv_scheme=info.get("scheme", "random"))
    if th_cv is not None:
        sf.theta_cv_sd, sf.theta_grid, sf.theta_loss, sf.theta_se = (float(th_cv["value"]), list(th_cv["grid"]), list(th_cv["loss"]),
                                                                       list(th_cv["se"]))
    if info.get("w_loss"):                  # stacking: the deployed estimate is q = w M + (1-w) P (P = the stage's prior mean)
        sf.w_grid, sf.w_loss, sf.w = list(info["w_grid"]), list(info["w_loss"]), float(info["w"])
        if sf.w < 1.0:
            sf.coef_m = coef
            sf.coef = glm.stack_coef(stage.prob, coef, prior_deck, sf.w, spec.sig_cov, stage.w)
    if "note" in info:
        sf.warnings.append(f"{name}: {info['note']}")
    if not res.converged:
        sf.warnings.append(f"{name}: optimiser did not fully converge ({res.message})")
    if info["done"] and info["loss"]:
        b = info["best"]
        if b == 0:
            sf.warnings.append(f"{name}: cross-validation prefers the strongest prior (sigma {s:.4f}, lower edge of the grid): "
                               "this format's own games add nothing detectable beyond the earlier stage"
                               + (f" (stacking weight of the refit: w = {sf.w:g})" if sf.w_loss else ""))
        elif b == len(info["grid"]) - 1:
            sf.warnings.append(f"{name}: cross-validation prefers the weakest prior (sigma {s:.4f}, upper edge of the grid)")
    return sf


def run_chain(stages: dict[str, StageData], n_card: int, n_feat: int, tmpdir: str, n_jobs: int | None, seed: int = 0,
              cancel: threading.Event | None = None, progress: Progress | None = None,
              frac: tuple[float, float] = (0.0, 1.0), prior_ok: set[str] | None = None,
              attr_load: np.ndarray | None = None, theta_prior: dict | None = None, extras: list | None = None,
              ref_fits: dict[str, StageFit] | None = None, opts: "ChainOpts | None" = None,
              draft_prior: np.ndarray | None = None) -> dict[str, StageFit]:
    """Fit the available stages in order draft -> bo1 -> bo3. Returns {stage: StageFit}.

    `opts` (ChainOpts): how the Sealed stages choose their prior strength - time-split CV and stacking by default. A stacked
    stage's `coef` is the deployed estimate q (`coef_m` the plain fit M) and q is the next stage's prior.

    Only stages in `prior_ok` (default: all) serve as the prior of a later stage.  `attr_load` ((n_card, n_attr), from
    `DeckSpace.attr_load`) declares that the deck part has n_attr trailing card-attribute columns; `theta_prior`
    (dict(mean, spread), structure units of this set) replaces the zero prior of the FIRST stage's structure terms.
    `extras` ([dict(name, n)]) declares trailing extra deck columns (card pairs, card x archetype) with their own prior SD
    per stage, chosen by CV; `ref_fits` (the plain chain) then supplies the card sigmas so only those SDs are searched."""
    order = [n for n in ("draft", "bo1", "bo3") if n in stages]
    fits: dict[str, StageFit] = {}
    prev: StageFit | None = None
    n_attr = 0 if attr_load is None else attr_load.shape[1]
    n_extra = int(sum(e["n"] for e in (extras or [])))
    p_deck = n_card + n_feat + n_attr + n_extra
    a, b = frac
    for i, name in enumerate(order):
        st = stages[name]
        lo, hi = a + (b - a) * i / len(order), a + (b - a) * (i + 1) / len(order)

        def prog(f, m, lo=lo, hi=hi):
            if progress:
                progress(lo + (hi - lo) * f, m)

        _check(cancel)
        lam = lam_se = lam_from = None
        floor = None
        if prev is None:
            prior = np.zeros(p_deck)
            if draft_prior is not None and name == "draft":
                prior[:n_card] = np.asarray(draft_prior, dtype=float)
            if theta_prior is not None and i == 0:
                prior[n_card:n_card + n_feat] = theta_prior["mean"]
                floor = np.asarray(theta_prior["spread"], dtype=float)
        elif prev.name == "draft":
            lam_raw, lam_se = fit_lambda(st, prev.deck)
            lam = float(np.clip(lam_raw, *LAMBDA_CLIP))
            lam_from = prev.name
            prior = lam * prev.deck
            log.info("lambda(%s <- %s) = %.3f (se %.3f)", name, prev.name, lam_raw, lam_se)
        else:
            prior = prev.deck.copy()
            lam_from = prev.name
        fpf = None
        if (opts or ChainOpts()).leak_free:
            chain_ = [n for n in order[:i] if prior_ok is None or n in prior_ok]
            if chain_ and prev is not None:
                fpf = (lambda st=st, name=name, chain_=chain_, i=i, prog=prog:
                       fold_priors(stages, fits, chain_, name, tmpdir, n_jobs, seed + i, cancel, prog))
        fits[name] = _stage_fit(st, prior, n_card, n_feat, tmpdir, n_jobs, seed + i, cancel, prog, lam, lam_se, lam_from,
                                attr_load=attr_load, theta_floor=floor, extras=extras,
                                ref_fit=(ref_fits or {}).get(name), opts=opts, fold_prior_fn=fpf)
        fits[name].se_total = propagate_se(fits[name], prev if lam_from else None)
        if prior_ok is None or name in prior_ok:
            prev = fits[name]
        prog(1.0, f"{name} fitted")
    return fits


def refit_chain(stages: dict[str, StageData], ref: dict[str, StageFit], prior_ok: set[str] | None = None,
                cancel: threading.Event | None = None) -> dict[str, np.ndarray]:
    """Refit the chain with each stage's sigma fixed at `ref`'s choice (no CV, no SEs), honouring `stage.w`.

    Lambda is re-estimated (it depends on the weights); the stacking weights of `ref` are kept. Returns {stage: coef}. Used to
    compare weightings cheaply."""
    order = [n for n in ("draft", "bo1", "bo3") if n in stages]
    out: dict[str, np.ndarray] = {}
    prev_deck = prev_name = None
    for name in order:
        _check(cancel)
        st, rf = stages[name], ref[name]
        if prev_deck is None:
            prior = rf.prior_deck.copy()
        elif prev_name == "draft":
            lam_raw, _ = fit_lambda(st, prev_deck)
            prior = float(np.clip(lam_raw, *LAMBDA_CLIP)) * prev_deck
        else:
            prior = prev_deck.copy()
        coef, _, _ = fit_stage(st, prior, rf.spec, rf.sigma, x0=rf.coef_m if rf.coef_m is not None else rf.coef, with_se=False)
        if rf.w < 1.0:
            coef = glm.stack_coef(st.prob, coef, prior, rf.w, rf.spec.sig_cov, st.w)
        out[name] = coef
        if prior_ok is None or name in prior_ok:
            prev_deck, prev_name = coef[st.prob.k:], name
    return out


# ---------------------------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------------------------

def prior_perturbation(fits: dict[str, StageFit], name: str, rng: np.random.Generator) -> np.ndarray | None:
    """One draw of the error of stage `name`'s prior mean (deck part), or None when the stage has no estimated prior.

    The prior mean is the previous stage's estimate (times the fitted scalar lambda when that stage is the draft): it is itself an
    estimate. A bootstrap that keeps it fixed omits that uncertainty although `StageFit.se_total` includes it, so intervals,
    P(better) and P(best) came out too narrow. Each replicate therefore perturbs the prior mean with
    N(0, (lambda se_prev)^2) per coefficient and one shared draw of lambda's own error times the previous estimate
    (variance (beta_prev lambda_se)^2); marginally Var = (lambda se_prev)^2 + (beta_prev lambda_se)^2. The perturbation is
    independent between coefficients (the previous stage's covariances are not carried)."""
    f = fits[name]
    prev = fits.get(f.lam_from) if f.lam_from else None
    if prev is None or prev.se_total is None:
        return None
    scale = f.lam if f.lam is not None else 1.0
    sd = np.asarray(prev.se_total[prev.k:], dtype=float)
    out = scale * sd * rng.standard_normal(len(sd))
    if f.lam is not None and f.lam_se:
        out = out + float(f.lam_se) * rng.standard_normal() * np.asarray(prev.deck, dtype=float)
    return out


def bootstrap(stages: dict[str, StageData], fits: dict[str, StageFit], n_boot: int, tmpdir: str, n_jobs: int | None,
              targets: list[str], seed: int = 0, cancel: threading.Event | None = None,
              progress: Progress | None = None, prior_ok: set[str] | None = None) -> dict[str, np.ndarray]:
    """Event-resampling bootstrap of the target stages. Returns {stage: coef array (n_boot+1, ncols)}, row 0 = full fit.

    For bo3 the bo1 stage is re-sampled and re-fitted first and its replicate estimate is the prior of the bo3
    replicate (so the uncertainty of the prior propagates).  The prior mean of the FIRST stage of that chain (the draft estimate
    times lambda) is perturbed by its own estimation error in every replicate (`prior_perturbation`), so the replicates
    spread like the model's total standard errors (`StageFit.se_total`) instead of only the target stage's sampling error."""
    out = {t: np.zeros((n_boot + 1, fits[t].coef.shape[0])) for t in targets}
    for t in targets:
        out[t][0] = fits[t].coef
    if n_boot <= 0:
        return out
    ok = set(stages) if prior_ok is None else prior_ok
    chain = [n for n in ("bo1", "bo3") if n in stages and (n in targets or (n == "bo1" and "bo3" in targets and "bo1" in ok))]
    paths = {n: stage_files(stages[n], tmpdir, seed + 11) for n in chain}
    tasks = []
    for b in range(1, n_boot + 1):
        sts = []
        for j, n in enumerate(chain):
            f = fits[n]
            pm = prior_vector(f.k, f.prior_deck)
            if j == 0:                                          # later stages take the (perturbed) replicate estimate of the earlier one
                pert = prior_perturbation(fits, n, np.random.default_rng([seed, b, 7919]))
                if pert is not None:
                    pm = pm.copy()
                    pm[f.k:] += pert
            sts.append(dict(path=paths[n], spec=f.spec, s=f.sigma, x0=f.coef_m if f.coef_m is not None else f.coef, prior_mean=pm,
                            stack_w=f.w if f.w < 1.0 else None,
                            deck_from_prev=(n == "bo3" and "bo1" in chain and "bo1" in ok), seed=seed * 100003 + b * 17 + (1 if n == "bo3" else 0),
                            maxiter=300))
        tasks.append(dict(rep=b, stages=sts))
    for r in glm.run_parallel(glm.boot_chain_task, tasks, n_jobs, cancel=cancel, progress=progress, label="Bootstrap"):
        for n, c in zip(chain, r["coefs"]):
            if n in out:
                out[n][r["rep"]] = c
    return out


def make_tmpdir(base: str | os.PathLike | None = None) -> str:
    return tempfile.mkdtemp(prefix="sealed_lab_", dir=str(base) if base else None)


def cleanup(tmpdir: str) -> None:
    shutil.rmtree(tmpdir, ignore_errors=True)
