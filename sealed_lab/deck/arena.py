"""MTG Arena deck / pool text: parsing and export.

Input (what Arena's "Export Deck" / pool lists, and what people paste) is tolerant:
  * optional BOM, CRLF / LF / CR line endings, trailing spaces, non-breaking spaces, tabs;
  * optional headers `Deck`, `Sideboard`, `Companion`, `Commander`, `About` (+ `Name <deck name>` lines under About);
  * card lines `N Name`, `N Name (SET) 123`, `Nx Name`; names may contain commas, apostrophes, accents, " // ";
  * without a `Deck` header a blank line separates deck and sideboard (after one, only a `Sideboard` header does);
  * `*F*` / `*E*` foil markers after the collector number are ignored.
Companion and Commander cards are filed with the sideboard (Sealed has neither; they still belong to the pool).

Export: `Deck` / `Sideboard` sections with `N Name` lines.  We have no collector numbers, so no `(SET) 123` suffix.
Cards whose 17lands name contains " // " (split cards, double-faced cards, adventures) are exported by FRONT FACE
only: Arena resolves a bare front-face name for every layout, whereas the full "A // B" spelling is only
reliably accepted for split cards, and the card layout is not known here.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from sealed_lab.types import DeckEval

log = logging.getLogger(__name__)

_HEADERS = {"deck": "main", "sideboard": "side", "companion": "side", "commander": "side", "about": "about"}
_CARD_RE = re.compile(r"^(\d+)(?:[xX]\s+|\s+)(\S.*?)\s*$")
_FOIL = r"(?:\s+\*[FE]\*)?"                                   # Arena's foil / etched markers after the collector number
_SET_SUFFIX_RE = re.compile(r"\s+\([A-Za-z0-9]{2,6}\)(?:\s+(?!\*[FE]\*)[A-Za-z0-9★*†/-]+)?" + _FOIL + r"\s*$")
_SET_ONLY_RE = re.compile(r"\s+\([A-Za-z0-9]{2,6}\)" + _FOIL + r"\s*$")


@dataclass
class ParsedDeck:
    main: dict[str, int] = field(default_factory=dict)
    side: dict[str, int] = field(default_factory=dict)
    ignored: list[str] = field(default_factory=list)      # non-empty lines that were neither header nor card line
    deck_name: str = ""
    had_headers: bool = False

    @property
    def n_main(self) -> int:
        return sum(self.main.values())

    @property
    def n_side(self) -> int:
        return sum(self.side.values())


def clean_card_name(raw: str) -> str:
    """Strip a trailing ` (SET) 123` / ` (SET)` suffix and whitespace from a card-line name."""
    s = raw.strip()
    m = _SET_SUFFIX_RE.search(s)
    if m is None:
        m = _SET_ONLY_RE.search(s)
    if m is not None and m.start() > 0:
        s = s[:m.start()]
    return " ".join(s.split())


def card_name_of_line(line: str) -> str | None:
    """The card name of one pasted line ("2 Lightning Strike (M21) 159" -> "Lightning Strike"), None for headers and other lines."""
    line = line.replace("\ufeff", "").strip()
    if not line or line.rstrip(":").strip().casefold() in _HEADERS:
        return None
    m = _CARD_RE.match(line)
    if m is None:
        return None
    name = clean_card_name(m.group(2))
    return name or None


def parse_arena_full(text: str) -> ParsedDeck:
    """Parse Arena export text. Never raises; unparseable lines are collected in `ignored`."""
    out = ParsedDeck()
    if not text:
        return out
    text = text.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n").replace(" ", " ")
    section = "main"
    pending_side = False       # blank line seen after cards of the main section
    deck_header = False        # an explicit `Deck` header: only a `Sideboard` header starts the sideboard then
    seen_card_in_section = False
    for raw in text.split("\n"):
        line = raw.replace("﻿", "").strip()
        if not line:
            if section == "main" and seen_card_in_section and not deck_header:
                pending_side = True
            continue
        key = line.rstrip(":").strip().casefold()
        if key in _HEADERS:
            section = _HEADERS[key]
            if key == "deck":
                deck_header = True
            pending_side = False
            seen_card_in_section = False
            out.had_headers = True
            continue
        if section == "about":
            if line.casefold().startswith("name"):
                out.deck_name = line[4:].strip()
            continue
        m = _CARD_RE.match(line)
        if m is None:
            out.ignored.append(line)
            continue
        n = int(m.group(1))
        name = clean_card_name(m.group(2))
        if n <= 0 or not name:
            out.ignored.append(line)
            continue
        if section == "main" and pending_side:
            section = "side"
            pending_side = False
        target = out.main if section == "main" else out.side
        target[name] = target.get(name, 0) + n
        seen_card_in_section = True
    return out


def parse_arena(text: str) -> tuple[dict[str, int], dict[str, int]]:
    """`(main, side)` as {card name: copies}. For pool input use main + side (all cards)."""
    p = parse_arena_full(text)
    return p.main, p.side


def parse_mtgo(text: str) -> tuple[dict[str, int], dict[str, int]]:  # pragma: no cover
    """TODO: MTGO .dek / .txt import is not part of v1."""
    raise NotImplementedError("MTGO import is not supported yet")


def pool_of(main: dict[str, int], side: dict[str, int]) -> dict[str, int]:
    """Union of deck and sideboard names (summed copies)."""
    out = dict(main)
    for k, v in side.items():
        out[k] = out.get(k, 0) + v
    return out


# ---------------------------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------------------------

def arena_name(name: str) -> str:
    """Name as written in an Arena import: front face for anything containing " // "."""
    return name.split(" // ")[0].strip()


def format_arena(main: dict[str, int], side: dict[str, int] | None = None, set_code: str | None = None) -> str:
    """Arena-importable text. `set_code` is accepted for API symmetry; collector numbers are unknown so none are written."""
    def merged(d: dict[str, int]) -> dict[str, int]:
        o: dict[str, int] = {}
        for k, v in d.items():
            if v > 0:
                o[arena_name(k)] = o.get(arena_name(k), 0) + int(v)
        return o

    lines = ["Deck"] + [f"{v} {k}" for k, v in merged(main).items()]
    sb = merged(side or {})
    if sb:
        lines += ["", "Sideboard"] + [f"{v} {k}" for k, v in sb.items()]
    return "\n".join(lines) + "\n"


def export_arena(deck: DeckEval, sideboard: dict[str, int] | None = None, set_code: str | None = None) -> str:
    """Arena import text for a `DeckEval` (+ optional sideboard names)."""
    main: dict[str, int] = {}
    for c in deck.cards:
        main[c.name] = main.get(c.name, 0) + c.count
    return format_arena(main, sideboard, set_code)
