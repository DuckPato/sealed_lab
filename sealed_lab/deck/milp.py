"""The integer program behind the deck builder (scipy.optimize.milp / HiGHS).

One *item* is a distinct card (spell or non-basic land) with `copies` available in the pool.  Every copy is a binary
variable (copies of one card are ordered x_k >= x_{k+1} to break symmetry, so a deck has a unique representation and
no-good cuts are valid); basic lands are integer variables per colour.

    maximise   sum_i coef_i * count_i
    s.t.       sum of all cards                         = deck_size
               non-basic lands + basics                 = n_lands
               splash-class cards                       <= splash_cap
               sources(colour) >= floor(colour)         (lands producing the colour + basics of the colour)
               optional: overlap with already selected decks <= n_spells - min_diff   (diversity)
               optional: no-good cuts for the solutions already returned              (top-k)

`coef` is whatever the caller linearised (card beta + the structure terms that are linear in card counts).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

log = logging.getLogger(__name__)


@dataclass
class MilpSpec:
    coef: np.ndarray                 # (n_items,) objective gain per copy
    copies: np.ndarray               # (n_items,) copies available
    is_land: np.ndarray              # (n_items,) bool: non-basic land item
    produces: np.ndarray             # (n_items, 5) bool: colours a land item produces (ignored for spells)
    splash: np.ndarray               # (n_items,) bool: spell needing the splash colour
    basic_colors: list[int]          # colour indices (0..4) for which basic lands are variables
    floors: dict[int, int]           # colour index -> minimum number of sources
    n_lands: int
    deck_size: int = 40
    splash_cap: int = 3
    locked: np.ndarray | None = None  # (n_items,) copies that must be in the deck (lower bounds; <= copies)
    creature: np.ndarray | None = None              # (n_items,) bool: creature spell
    creature_range: tuple[int, int] | None = None   # (lo, hi) creature copies allowed in the deck


@dataclass
class MilpSolution:
    obj: float
    counts: np.ndarray               # (n_items,) copies chosen
    basics: dict[int, int] = field(default_factory=dict)   # colour index -> basics chosen


def solve_milp(spec: MilpSpec, k: int = 1, apart: list[tuple[np.ndarray, int]] | None = None,
               time_limit: float = 5.0) -> list[MilpSolution]:
    """Best `k` distinct (in the chosen cards; basics split is free) solutions, best first. [] if infeasible.

    `apart`: list of (counts_of_other_deck_over_items, min_diff): the spells of the result must have at least
    `min_diff` cards that are not in the other deck.
    """
    n_items = len(spec.coef)
    cop = np.minimum(np.asarray(spec.copies, dtype=int), spec.deck_size)
    off = np.concatenate([[0], np.cumsum(cop)]).astype(int)
    nb = int(off[-1])
    colors = list(spec.basic_colors)
    nB = len(colors)
    nv = nb + nB
    if nv == 0:
        return []
    item_of = np.repeat(np.arange(n_items), cop)
    is_land_v = np.concatenate([spec.is_land[item_of], np.zeros(nB, dtype=bool)])
    c = np.concatenate([-np.asarray(spec.coef, dtype=float)[item_of], np.zeros(nB)])
    A: list[np.ndarray] = []
    lo: list[float] = []
    hi: list[float] = []

    def row(vec: np.ndarray, lb: float, ub: float) -> None:
        A.append(vec)
        lo.append(lb)
        hi.append(ub)

    row(np.ones(nv), spec.deck_size, spec.deck_size)
    v = np.zeros(nv)
    v[:nb][is_land_v[:nb]] = 1.0
    v[nb:] = 1.0
    row(v, spec.n_lands, spec.n_lands)
    sp_v = np.concatenate([spec.splash[item_of] & ~spec.is_land[item_of], np.zeros(nB, dtype=bool)])
    if sp_v.any():
        row(sp_v.astype(float), 0, spec.splash_cap)
    if spec.creature is not None and spec.creature_range is not None:
        cre_v = np.concatenate([spec.creature[item_of], np.zeros(nB, dtype=bool)])
        row(cre_v.astype(float), spec.creature_range[0], spec.creature_range[1])
    for col, f in spec.floors.items():
        v = np.zeros(nv)
        v[:nb] = (is_land_v[:nb] & spec.produces[item_of, col]).astype(float)
        if col in colors:
            v[nb + colors.index(col)] = 1.0
        row(v, f, np.inf)
    for i in range(n_items):          # symmetry breaking between identical copies
        for j in range(cop[i] - 1):
            v = np.zeros(nv)
            v[off[i] + j] = 1.0
            v[off[i] + j + 1] = -1.0
            row(v, 0.0, np.inf)
    n_spells = spec.deck_size - spec.n_lands
    for other, md in (apart or []):
        v = np.zeros(nv)
        for i in range(n_items):
            if spec.is_land[i]:
                continue
            d = min(int(other[i]), int(cop[i]))
            v[off[i]:off[i] + d] = 1.0
        row(v, -np.inf, n_spells - md)

    ub = np.concatenate([np.ones(nb), np.full(nB, float(spec.n_lands))])
    lb = np.zeros(nv)
    if spec.locked is not None:
        for i in range(n_items):          # the first L (ordered) copies of a locked card are fixed to 1
            n_fix = min(int(spec.locked[i]), int(cop[i]))
            if n_fix > 0:
                lb[off[i]:off[i] + n_fix] = 1.0
    bounds = Bounds(lb, ub)
    integrality = np.ones(nv)
    out: list[MilpSolution] = []
    for _ in range(k):
        cons = LinearConstraint(np.array(A), np.array(lo), np.array(hi))
        res = milp(c, constraints=cons, integrality=integrality, bounds=bounds,
                   options={"time_limit": time_limit, "presolve": True})
        if res.x is None or res.status not in (0, 1):
            break
        x = np.rint(res.x)
        counts = np.zeros(n_items, dtype=int)
        np.add.at(counts, item_of, x[:nb].astype(int))
        basics = {colors[j]: int(x[nb + j]) for j in range(nB)}
        out.append(MilpSolution(float(-res.fun), counts, basics))
        if res.status == 1:  # time limit: do not trust further cuts
            break
        sel = x[:nb] > 0.5
        v = np.zeros(nv)
        v[:nb] = np.where(sel, 1.0, -1.0)
        row(v, -np.inf, float(sel.sum()) - 1.0)
    return out
