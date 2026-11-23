"""Which set does a pasted pool / deck belong to?

1. Arena export lines carry a ` (SET) 123` suffix: the majority of the suffixed non-basic copies decides (Arena / Scryfall codes are
   mapped to the app's set codes, case-insensitively).  Special printings (Special Guests, Secret Lair ...) do not vote.
2. Otherwise (or when the suffix set matches too few names) every modelled set's card table is tried: the set that matches the most
   copies wins if it matches at least 70% of the pool and is clearly ahead of the runner-up; else the result is "ambiguous" and the
   caller keeps its current choice.

Pure functions over `NameIndex`es, so the real API and the fake backend share the logic.
"""
from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from typing import Mapping

from sealed_lab.config import BASICS
from sealed_lab.deck.arena import parse_arena_full
from sealed_lab.deck.names import NameIndex

MIN_MATCH_FRACTION = 0.70          # name-based detection: share of the pool's copies the winning set must match
MIN_SUFFIX_SHARE = 0.60            # suffix detection: share of suffixed copies carrying the winning code
MIN_SUFFIX_COVERAGE = 0.50         # ... and share of the pool's copies that carry a suffix at all
MIN_SUFFIX_MATCH = 0.50            # the suffix set must also match this share of the names
MARGIN_FRACTION = 0.15             # lead over the runner-up, as a share of the pool (at least MARGIN_MIN copies)
MARGIN_MIN = 3

# Arena set code -> app (Scryfall / 17lands) set code, where they differ
ALIASES = {"DAR": "DOM"}
NON_VOTING = {"SPG", "SLD", "PLST", "PRM", "PLIST"}   # special printings: not evidence of the pool's set

_LINE_RE = re.compile(r"^\s*(\d+)(?:[xX]\s+|\s+)(.+?)\s+\(([A-Za-z0-9]{2,6})\)(?:\s+[A-Za-z0-9★*†/-]+)?\s*$")
_BASIC_NAMES = {b.casefold() for b in BASICS}


@dataclass
class SetGuess:
    set_code: str | None = None        # set to switch to (None: keep the current choice)
    matched: int = 0                   # copies of the pool matched in that set's card table (basic lands excluded)
    total: int = 0                     # copies in the pool (basic lands excluded)
    runner_up: str | None = None
    runner_up_matched: int = 0
    source: str = "none"               # suffix | names | none
    ambiguous: bool = False
    detected_code: str | None = None   # code read from the suffixes even when the app has no such set
    message: str = ""
    scores: dict[str, int] = field(default_factory=dict)   # matched copies per modelled set

    def to_dict(self) -> dict:
        return asdict(self)


def suffix_votes(text: str) -> tuple[dict[str, int], int]:
    """({SET code: copies}, copies of non-basic cards with a suffix) of an Arena export; basic lands never vote."""
    votes: dict[str, int] = {}
    n = 0
    for line in (text or "").replace("\r", "\n").split("\n"):
        m = _LINE_RE.match(line.replace("﻿", "").replace(" ", " "))
        if m is None:
            continue
        name = m.group(2).strip()
        if name.casefold() in _BASIC_NAMES:
            continue
        code = m.group(3).upper()
        code = ALIASES.get(code, code)
        n += int(m.group(1))
        if code in NON_VOTING:
            continue
        votes[code] = votes.get(code, 0) + int(m.group(1))
    return votes, n


def pool_copies(text: str) -> dict[str, int]:
    """{name: copies} of everything in the paste (deck + sideboard), basic lands excluded."""
    p = parse_arena_full(text)
    out: dict[str, int] = {}
    for d in (p.main, p.side):
        for k, v in d.items():
            if k.casefold() not in _BASIC_NAMES:
                out[k] = out.get(k, 0) + v
    return out


def matched_copies(names: Mapping[str, int], index: NameIndex) -> int:
    n = 0
    for name, c in names.items():
        m = index.match(name, fuzzy=False)
        if m.ok and not m.is_basic:
            n += c
    return n


def detect(text: str, indexes: Mapping[str, NameIndex]) -> SetGuess:
    """Guess the set of `text` among the modelled sets in `indexes` ({set code: NameIndex}). Never raises."""
    names = pool_copies(text)
    total = sum(names.values())
    g = SetGuess(total=total)
    if total == 0:
        g.message = ""
        return g
    scores = {s: matched_copies(names, idx) for s, idx in indexes.items()}
    g.scores = scores
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))

    # 1. suffix evidence ---------------------------------------------------------------------
    votes, n_suffixed = suffix_votes(text)
    if votes and n_suffixed >= MIN_SUFFIX_COVERAGE * total:
        top, top_n = max(votes.items(), key=lambda kv: (kv[1], kv[0]))
        if top_n >= MIN_SUFFIX_SHARE * sum(votes.values()):
            g.detected_code = top
            if top in indexes:
                if scores[top] >= MIN_SUFFIX_MATCH * total:
                    g.set_code, g.matched, g.source = top, scores[top], "suffix"
                    other = [kv for kv in ranked if kv[0] != top]
                    if other:
                        g.runner_up, g.runner_up_matched = other[0]
                    g.message = f"Detected {top} ({g.matched}/{total} cards matched)"
                    return g
            else:
                g.source = "suffix"
                g.message = f"The pool looks like {top} (from the set codes in the list), but the app has no data for that set"
                return g

    # 2. name matching ----------------------------------------------------------------------
    if not ranked:
        g.message = "No set with a card table yet: calculate a set first"
        return g
    best, best_n = ranked[0]
    second, second_n = ranked[1] if len(ranked) > 1 else (None, 0)
    g.runner_up, g.runner_up_matched = second, second_n
    margin = max(MARGIN_MIN, math.ceil(MARGIN_FRACTION * total))
    if best_n >= MIN_MATCH_FRACTION * total and best_n - second_n >= margin:
        g.set_code, g.matched, g.source = best, best_n, "names"
        g.message = f"Detected {best} ({best_n}/{total} cards matched)"
        return g
    g.ambiguous = True
    g.source = "names"
    g.matched = best_n
    tail = f", then {second} {second_n}/{total}" if second is not None else ""
    g.message = (f"Could not tell the set from the card names (best: {best} {best_n}/{total}{tail}); keeping your choice"
                 if best_n else "No card of the pool was recognised in any calculated set; keeping your choice")
    return g
