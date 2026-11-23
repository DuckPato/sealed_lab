"""Deck scoring: percentile of q(D) among real 17lands builds, plus efficiency against the best build from the pool."""
from __future__ import annotations

import logging
import threading
from typing import Callable

from sealed_lab.deck import arena as arena_mod
from sealed_lab.deck import builder
from sealed_lab.deck.names import NameIndex, apply_overrides, resolve_names
from sealed_lab.model.artifact import SetModel
from sealed_lab.types import ScoreResult

log = logging.getLogger(__name__)

Progress = Callable[[float, str], None]


def score(model: SetModel, main: dict[int, int], side: dict[int, int] | None, fmt: str,
          progress: Progress | None = None, cancel: threading.Event | None = None) -> ScoreResult:
    """Score `main` (card_id -> copies, basics included).

    Decks that are not 40 cards are still scored (the evaluation carries a warning).  With a sideboard, the builder
    is run on main + side (the user's deck is always one of its starting points, so the result is never worse than
    the deck) and `efficiency_pp = (game WR of the deck - game WR of the best build) in pp`, which is <= 0.
    """
    if fmt not in model.formats:
        raise ValueError(f"format {fmt!r} is not available for {model.set_code}; available: {', '.join(model.formats)}")
    main = {int(i): int(n) for i, n in main.items() if n > 0}
    if not main:
        raise ValueError("the deck has no cards the model knows")
    deck = model.evaluate(main, fmt)
    warnings: list[str] = []
    eff: float | None = None
    alt = None
    side = {int(i): int(n) for i, n in (side or {}).items() if n > 0}
    if side:
        pool = dict(side)
        for i, n in main.items():
            pool[i] = pool.get(i, 0) + n
        start = [main] if sum(main.values()) == builder.DECK_SIZE else None
        # the best build = representative of the top deck family (best under the full-data fit over the candidates of all draws)
        res = builder.build(model, pool, fmt, n_decks=1, progress=progress, cancel=cancel, start_decks=start, families=True)
        best = res.decks[0]
        eff = 100.0 * (deck.game_wr - best.game_wr)
        if eff > 0:      # cannot happen when the deck was a start point; guards against rounding
            eff = 0.0
        if eff < -1e-9:
            alt = best
            tip = _tip(model, main, builder_deck(res, model), deck.game_wr, best.game_wr)
            if tip:
                warnings.append(tip)
        else:
            eff = 0.0
        warnings += res.warnings
    return ScoreResult(set_code=model.set_code, fmt=fmt, deck=deck, efficiency_pp=eff, best_alternative=alt,
                       unmatched=[], warnings=warnings)


def builder_deck(res, model: SetModel) -> dict[int, int]:
    return {c.card_id: c.count for c in res.decks[0].cards if c.card_id is not None}


def _tip(model: SetModel, main: dict[int, int], best: dict[int, int], q_user: float, q_best: float) -> str:
    names = model.cards.sort("card_id")["name"].to_list()
    is_basic = model.cards.sort("card_id")["is_basic"].to_list()
    add: dict[str, int] = {}
    cut: dict[str, int] = {}
    for i in set(main) | set(best):
        d = best.get(i, 0) - main.get(i, 0)
        if is_basic[i] or d == 0:
            continue
        (add if d > 0 else cut)[names[i].split(" // ")[0]] = abs(d)
    if not add and not cut:
        return "The best build differs from your deck only in the basic land split."

    def fmt(d: dict[str, int]) -> str:
        items = [f"{n}x {k}" if n > 1 else k for k, n in sorted(d.items(), key=lambda kv: -kv[1])]
        return ", ".join(items[:6]) + (f" and {len(items) - 6} more" if len(items) > 6 else "") if items else "nothing"

    return (f"Tip: the best build from your pool (+{100 * (q_best - q_user):.1f}pp game win rate) plays {fmt(add)} "
            f"instead of {fmt(cut)}.")


def score_deck(set_code: str, text: str, fmt: str = "bo3", progress: Progress | None = None,
               cancel: threading.Event | None = None, name_overrides: dict[str, str] | None = None,
               search: bool = True) -> ScoreResult:
    """Parse Arena deck text (Deck + optional Sideboard) and score it (entry point used by `api.score_deck`).

    `name_overrides` ({pasted name: confirmed card name, "" = drop}) replaces names the user corrected.  `search=False` scores the
    Deck section only (fast): no builder run for the efficiency against the best build from deck + sideboard; the pool of deck +
    sideboard is still returned in `ScoreResult.pool` (the visual builder edits the deck with it)."""
    model = builder.load_model(set_code)
    parsed = arena_mod.parse_arena_full(text)
    if not parsed.main and not parsed.side:
        raise ValueError("no cards found in the deck text (paste an Arena export: lines like '1 Card Name')")
    index = NameIndex(model.cards)
    rm = resolve_names(apply_overrides(parsed.main, name_overrides), index)
    rs = resolve_names(apply_overrides(parsed.side, name_overrides), index, drop_basics=True)
    if not rm.counts:
        raise ValueError("none of the card names in the deck were recognised for set " + model.set_code)
    out = score(model, rm.counts, rs.counts if search else None, fmt, progress=progress, cancel=cancel)
    out.unmatched = rm.unmatched + rs.unmatched
    pool = {int(k): int(v) for k, v in rs.counts.items()}
    for k, v in rm.counts.items():
        if not index.is_basic.get(int(k), False):
            pool[int(k)] = pool.get(int(k), 0) + int(v)
    out.pool = pool
    w = list(rm.warnings) + list(rs.warnings)
    if parsed.ignored:
        w.append(f"{len(parsed.ignored)} line(s) could not be read: " + "; ".join(parsed.ignored[:3]))
    out.warnings = w + out.warnings
    return out
