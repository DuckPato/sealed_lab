"""Bo3 games 2-3 advice from the card x play/draw layer of the model (layer "MP", pre-registered in docs/playdraw/RULE.md).

"On the play / on the draw: swap X for Y" is shown when the swap raises the game win rate IN A GAME ON THE PLAY (resp. ON THE DRAW) by
at least `OPP_MIN_PP` pp AND helps in at least `OPP_P_MIN` of the bootstrap draws (the sideboard plan's gates, `opponent`).  The
gain is the full model's (card effects, mana simulation, castability, plus the +-gamma/2 shifts), computed by
`explain.swap_analysis(on_play=...)`.  No deck check or deviation-cost guardrail is needed because the main deck stays as it is: the
layer never enters q(D) (game 1), only the advice for games 2-3.  `PlaySwap.gain_avg_pp` is the same swap without the play/draw
shifts: when it already reaches the gate the swap is an ordinary suggested swap and `premium` is False (shown for information; the
rule's gates are on `gain_pp` and `p_better`).
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

from sealed_lab.deck import explain
from sealed_lab.model import opponent as om


@dataclass
class PlaySwap:
    on_play: bool
    out_id: int
    in_id: int
    gain_pp: float                 # win rate on the play (draw) after minus before, pp
    p_better: float | None         # share of bootstrap draws in which the swap helps
    gain_avg_pp: float             # the same swap without the play/draw shifts
    premium: bool                  # the play/draw shifts make the difference (gain_avg_pp below the gate)


@dataclass
class PlayDrawPlan:
    status: str                    # "ok" | "not_supported" | "off"
    message: str = ""
    fmt: str = "bo3"
    swaps: dict[str, list[PlaySwap]] = field(default_factory=dict)         # "play" / "draw" -> swaps, best first, distinct cards
    base_wr: dict[str, float] = field(default_factory=dict)                # the deck's win rate on the play / on the draw
    min_pp: float = om.OPP_MIN_PP
    p_min: float = om.OPP_P_MIN


def playdraw_plan(model, deck: dict[int, int], pool: dict[int, int], fmt: str, on_play: bool | None = None,
                  locked: dict[int, int] | None = None, excluded: set[int] | None = None, cancel: threading.Event | None = None,
                  min_pp: float = om.OPP_MIN_PP, p_min: float = om.OPP_P_MIN) -> PlayDrawPlan:
    """Reliable play/draw swaps X -> Y for the current build (`on_play` None = both sides; True / False = one).  Bo3 only."""
    plan = PlayDrawPlan(status="ok", fmt=fmt, min_pp=min_pp, p_min=p_min)
    if fmt != "bo3":
        plan.status, plan.message = "off", "Play/draw advice only matters in Bo3 games 2-3."
        return plan
    if getattr(model, "playdraw", None) is None:
        meta = model.meta.get("playdraw") or {}
        plan.status = "not_supported"
        plan.message = ("This model was calculated without the play/draw layer: recalculate the set to test it." if not meta else
                        "The play/draw layer did not pass its hold-out test for this set (" + str(meta.get("reason", "")) + "). Nothing is shown.")
        return plan
    locked = locked or {}
    for side in ([True, False] if on_play is None else [bool(on_play)]):
        key = "play" if side else "draw"
        sa = explain.swap_analysis(model, deck, pool, fmt, locked=locked, excluded=excluded, cancel=cancel, on_play=side)
        plan.base_wr[key] = sa.base_wr
        cands = []
        for s in sa.side_best.values():
            if s.out_id in locked and locked[s.out_id] >= deck.get(s.out_id, 0):
                continue
            if s.gain_pp >= min_pp and (s.p_better is None or s.p_better >= p_min):
                gavg = float(s.gain_avg_pp if s.gain_avg_pp is not None else float("nan"))
                cands.append(PlaySwap(on_play=side, out_id=s.out_id, in_id=s.in_id, gain_pp=s.gain_pp, p_better=s.p_better,
                                      gain_avg_pp=gavg, premium=bool(gavg < min_pp)))
        cands.sort(key=lambda x: -x.gain_pp)
        used_o: set[int] = set()
        used_i: set[int] = set()
        picked = []
        for c in cands:
            if c.out_id in used_o or c.in_id in used_i:
                continue
            used_o.add(c.out_id)
            used_i.add(c.in_id)
            picked.append(c)
        if picked:
            plan.swaps[key] = picked
    if not plan.swaps:
        plan.message = (f"No sideboard card is reliably better than a main-deck card on the play or on the draw for this build "
                        f"(needs >= {min_pp:g} pp and P >= {p_min:.0%}).")
    return plan
