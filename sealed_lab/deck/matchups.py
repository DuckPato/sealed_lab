"""Sideboard plans against opponent colours (Bo3), built on the card x opponent interactions of the model (layer "MO").

A sideboard card is *premium against strategy o* for a given build when ALL of these hold (thresholds in `Thresholds`):
  1. the card's value against o is reliably above its average value:  P(delta°_{c,o} > 0) >= p_min  (normal approximation
     with the cluster-robust SE) and delta°_{c,o} >= min_pp (pp of game win rate near a 55% deck);
  2. the opponent-specific advantage of the incoming card over the outgoing one is itself supported (P >= p_min, normal
     approximation with the two cluster-robust SEs);
  3. against o, taking it in for the weakest main-deck card it can replace (the best swap by the full model, mana simulation
     included) raises the win rate by >= min_pp pp, and the swap helps in >= p_min of the bootstrap draws;
  4. the premium is the opponent's doing: the same swap against the average opponent gains < min_pp (otherwise it is an
     ordinary suggested swap) and the opponent-specific increment is >= min_pp.
If the matchup layer did not pass its hold-out test (or is absent) the plan says so and marks nothing.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm

from sealed_lab.deck import explain
from sealed_lab.model import opponent as om

log = logging.getLogger(__name__)


@dataclass
class Thresholds:
    p_min: float = om.OPP_P_MIN
    min_pp: float = om.OPP_MIN_PP


@dataclass
class PlanSwap:
    opp: str
    out_id: int
    in_id: int
    gain_pp: float                 # win rate against `opp` after minus before, pp
    p_better: float | None         # share of bootstrap draws in which the swap helps
    gain_avg_pp: float             # the same swap against the average opponent
    delta_in_pp: float             # centred effect of the incoming card against `opp` (pp)
    delta_out_pp: float            # ... of the outgoing card
    p_delta_in: float              # P(delta°_in > 0)
    reliable: bool = False


@dataclass
class SideboardPlan:
    status: str                    # "ok" | "not_supported" | "off"
    message: str = ""
    fmt: str = "bo3"
    opponents: list[str] = field(default_factory=list)
    swaps: dict[str, list[PlanSwap]] = field(default_factory=dict)          # opp -> reliable swaps, best first, distinct cards
    chips: dict[int, list[PlanSwap]] = field(default_factory=dict)          # sideboard card -> reliable swaps (any opp)
    base_wr: dict[str, float] = field(default_factory=dict)                 # the deck's win rate against each opponent
    thresholds: Thresholds = field(default_factory=Thresholds)
    encoding: str = ""
    n_checked: int = 0


def support_message(model) -> tuple[bool, str]:
    """(matchup layer usable, plain-language explanation)."""
    opp = getattr(model, "opp", None)
    meta = (model.meta.get("matchups") or {})
    if opp is not None:
        return True, ""
    if not meta:
        return False, "This model was calculated without the matchup layer: recalculate the set to test it."
    if meta.get("mode") == "off":
        return False, "The matchup layer was switched off for this calculation."
    return False, ("The data do not support opponent-specific plans for this set: card x opponent-colour interactions did not "
                   "improve held-out predictions (" + str(meta.get("reason", "")) + "). Nothing is marked.")


def plan_for(model, deck: dict[int, int], pool: dict[int, int], fmt: str, locked: dict[int, int] | None = None,
             excluded: set[int] | None = None, cancel: threading.Event | None = None, opps: list[str] | None = None,
             th: Thresholds | None = None) -> SideboardPlan:
    """Suggested in/out swaps against each opponent colour pair for the current build (see the module docstring)."""
    th = th or Thresholds()
    ok, msg = support_message(model)
    if not ok:
        return SideboardPlan(status="not_supported", message=msg, fmt=fmt, thresholds=th)
    if fmt != "bo3":
        return SideboardPlan(status="off", message="Sideboarding only matters in Bo3.", fmt=fmt, thresholds=th)
    if fmt not in getattr(model, "opp_formats", [fmt]):
        return SideboardPlan(status="not_supported", fmt=fmt, thresholds=th, message=(
            f"The matchup layer is not applied in this format for this set (it is applied in "
            f"{', '.join(model.opp_formats) or 'no format'}; see the model report). Nothing is marked."))
    opm = model.opp
    opps = list(opps or opm.opps)
    plan = SideboardPlan(status="ok", fmt=fmt, opponents=opps, thresholds=th, encoding=opm.mode)
    locked = locked or {}
    for o in opps:
        sa = explain.swap_analysis(model, deck, pool, fmt, locked=locked, excluded=excluded, cancel=cancel, opp=o)
        plan.base_wr[o] = sa.base_wr
        j = opm.opps.index(o) if o in opm.opps else None
        eff = opm.effect(o, draws=False)
        cands: list[PlanSwap] = []
        for i, s in sa.side_best.items():
            if s.out_id in locked and locked[s.out_id] >= deck.get(s.out_id, 0):
                continue
            plan.n_checked += 1
            p_in = float(norm.cdf(eff[i] / max(opm.se_c[i, j], 1e-12))) if j is not None else float("nan")
            diff = float(eff[i] - eff[s.out_id])
            se_diff = float(np.hypot(opm.se_c[i, j], opm.se_c[s.out_id, j])) if j is not None else float("inf")
            p_diff = float(norm.cdf(diff / max(se_diff, 1e-12)))      # P(in is better than out against o, opponent part only)
            ps = PlanSwap(opp=o, out_id=s.out_id, in_id=s.in_id, gain_pp=s.gain_pp, p_better=s.p_better,
                          gain_avg_pp=float(s.gain_avg_pp if s.gain_avg_pp is not None else float("nan")),
                          delta_in_pp=om.PP_PER_LOGIT * float(eff[i]), delta_out_pp=om.PP_PER_LOGIT * float(eff[s.out_id]),
                          p_delta_in=p_in)
            # the premium must come from the opponent: the same swap against the average opponent is not already a general upgrade
            # (that case is the ordinary "suggested swap"), and the opponent-specific increment is itself >= min_pp
            ps.reliable = bool(ps.gain_avg_pp < th.min_pp and s.gain_pp - ps.gain_avg_pp >= th.min_pp and s.gain_pp >= th.min_pp and (s.p_better is None or s.p_better >= th.p_min)
                               and ps.delta_in_pp >= th.min_pp and p_in >= th.p_min and p_diff >= th.p_min)
            if ps.reliable:
                cands.append(ps)
        cands.sort(key=lambda x: -x.gain_pp)
        used_o: set[int] = set()
        used_i: set[int] = set()
        picked: list[PlanSwap] = []
        for ps in cands:
            plan.chips.setdefault(ps.in_id, []).append(ps)
            if ps.out_id in used_o or ps.in_id in used_i:
                continue
            used_o.add(ps.out_id)
            used_i.add(ps.in_id)
            picked.append(ps)
        if picked:
            plan.swaps[o] = picked
    for v in plan.chips.values():
        v.sort(key=lambda x: -x.gain_pp)
    if not plan.swaps:
        plan.message = ("No sideboard card is reliably better than a main-deck card against any opponent colour pair for this "
                        f"build (needs >= {th.min_pp:g} pp and P >= {th.p_min:.0%}).")
    return plan
