"""Sample hands: deal opening hands and draws from a deck, the way the app's mana simulator deals them (Qt-free).

* Bo1 ("bo1"): Arena's Bo1 opening-hand smoother, exactly as the mana simulator v2 models it: the number of lands among the
  7 cards is drawn from `manasim.hand_land_pmf(L, n, "sealed")` (hypergeometric x the fitted smoother of the Bo1 Sealed queue),
  and the cards themselves are random given that count (which lands, which spells).  Every London-mulligan redeal uses the
  same smoother, as in the simulator.
* Bo3 ("bo3"): a plain random shuffle (the simulator's "random" regime).
* Draws are plain random from the shuffled library (the simulator's mild land-balancing tilt of the draws is NOT applied here).
* London mulligan: draw 7 new cards, then put N cards (N = mulligans so far) on the bottom of the library.

`Deal` holds the state of one sample game; the randomness comes from an injectable `random.Random`.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from sealed_lab.config import format_regime
from sealed_lab.model.manasim import hand_land_pmf

HAND = 7
MAX_TURN = 10
MAX_MULL = 6                       # a mulligan to 1 card is the last one offered
COLORS = "WUBRG"


@dataclass(frozen=True)
class SCard:
    """One physical card of the deck."""
    cid: int
    name: str
    is_land: bool = False
    mv: int = 0
    cost: str = ""                 # e.g. "2WW" / "{2}{W}{W}": only the colored letters are read
    produces: str = ""             # colors a land (or any source) makes

    def pips(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for ch in self.cost.split("//")[0]:       # "A // B" (adventure etc.): the main face only, never both summed
            if ch in COLORS:
                out[ch] = out.get(ch, 0) + 1
        return out


def library_from_deck(deck: dict[int, int], info: dict[int, dict]) -> list[SCard]:
    """The deck's 40 physical cards ({card_id: copies} + the builder's card info), in a fixed order (lands first)."""
    cards: list[SCard] = []
    for cid in sorted(deck):
        d = info.get(cid) or {}
        land = bool(d.get("is_land"))
        c = SCard(cid, str(d.get("name") or f"#{cid}"), land, int(round(float(d.get("mana_value") or 0.0))),
                  str(d.get("cost") or ""), "".join(ch for ch in str(d.get("produces") or "") if ch in COLORS) if land else "")
        cards.extend([c] * int(deck[cid]))
    cards.sort(key=lambda c: not c.is_land)
    return cards


def smoothed_hand_lands(L: int, n: int, rng: random.Random, fmt: str = "bo1") -> int:
    """Lands among the 7 cards shown: the simulator's own pmf (`manasim.hand_land_pmf`) for Bo1, plain hypergeometric for Bo3."""
    pmf = hand_land_pmf(L, n, format_regime(fmt))
    u = rng.random()
    acc = 0.0
    for k, p in enumerate(pmf):
        acc += float(p)
        if u < acc:
            return k
    return int(max((k for k, p in enumerate(pmf) if p > 0), default=0))


def shuffle_deal(cards: list[SCard], rng: random.Random, fmt: str = "bo1") -> tuple[list[SCard], list[SCard]]:
    """(hand of 7, library top-first) for a fresh shuffle.  Bo1: the land count of the hand follows the smoother."""
    lands = [c for c in cards if c.is_land]
    spells = [c for c in cards if not c.is_land]
    rng.shuffle(lands)
    rng.shuffle(spells)
    h = min(HAND, len(cards))
    if fmt == "bo1" and lands and spells:
        k = smoothed_hand_lands(len(lands), len(cards), rng, fmt)
        k = min(k, len(lands))
        k = max(k, h - len(spells))
        hand = lands[:k] + spells[:h - k]
        rest = lands[k:] + spells[h - k:]
        rng.shuffle(rest)
        return hand, rest
    allc = lands + spells
    rng.shuffle(allc)
    return allc[:h], allc[h:]


def pips_payable(pips: dict[str, int], lands: list[SCard]) -> bool:
    """Can each colored pip get its own land that makes that color?  (Each land pays one pip: a dual or a five-color land
    covers one symbol, not one of every color.)  Small bipartite matching by augmenting paths."""
    need = [col for col, n in pips.items() for _ in range(n)]
    if len(need) > len(lands):
        return False
    owner: dict[int, int] = {}                     # land index -> pip index

    def assign(p: int, seen: set[int]) -> bool:
        for j, land in enumerate(lands):
            if need[p] in land.produces and j not in seen:
                seen.add(j)
                if j not in owner or assign(owner[j], seen):
                    owner[j] = p
                    return True
        return False
    return all(assign(p, set()) for p in range(len(need)))


def castable(hand: list[SCard]) -> tuple[int, dict[str, int], list[SCard]]:
    """(lands, land sources per color, spells castable now) from the cards in hand.

    Rough on purpose: a spell counts if the lands in hand >= its mana value and its colored pips can each be paid by a different
    land of that color.  Mana rocks, hybrid costs and X are ignored."""
    lands = [c for c in hand if c.is_land]
    src = {col: sum(1 for c in lands if col in c.produces) for col in COLORS}
    ok = [c for c in hand if not c.is_land and len(lands) >= c.mv and pips_payable(c.pips(), lands)]
    return len(lands), src, ok


@dataclass
class Deal:
    """State of one sample game: shuffled library (top first), hand, mulligans, turn."""
    cards: list[SCard]
    rng: random.Random = field(default_factory=random.Random)
    fmt: str = "bo1"
    on_play: bool = True
    hand: list[SCard] = field(default_factory=list)
    library: list[SCard] = field(default_factory=list)
    mulligans: int = 0
    to_bottom: int = 0             # cards still to put on the bottom after a mulligan
    turn: int = 0                  # 0 = opening hand, then turns taken

    def __post_init__(self) -> None:
        self.deal()

    def deal(self) -> None:
        self.hand, self.library = shuffle_deal(self.cards, self.rng, self.fmt)

    def new_hand(self, cards: list[SCard] | None = None) -> None:
        if cards is not None:
            self.cards = list(cards)
        self.mulligans, self.to_bottom, self.turn = 0, 0, 0
        self.deal()

    def can_mulligan(self) -> bool:
        return self.turn == 0 and self.to_bottom == 0 and self.mulligans < MAX_MULL

    def mulligan(self) -> bool:
        """London mulligan: a new 7; N = mulligans so far cards then have to go to the bottom (`bottom`)."""
        if not self.can_mulligan():
            return False
        self.mulligans += 1
        self.deal()
        self.to_bottom = self.mulligans
        return True

    def bottom(self, idx: list[int]) -> bool:
        """Put exactly `to_bottom` cards of the hand (by position) on the bottom of the library."""
        if len(set(idx)) != self.to_bottom or any(not 0 <= i < len(self.hand) for i in idx):
            return False
        gone = [self.hand[i] for i in sorted(set(idx))]
        self.hand = [c for i, c in enumerate(self.hand) if i not in set(idx)]
        self.library.extend(gone)
        self.to_bottom = 0
        return True

    def can_draw(self) -> bool:
        return self.to_bottom == 0 and self.turn < MAX_TURN

    def draw(self) -> SCard | None:
        """Next turn: turn counter +1 and the turn's draw (none on turn 1 on the play).  Returns the card drawn, if any."""
        if not self.can_draw():
            return None
        self.turn += 1
        if self.turn == 1 and self.on_play or not self.library:
            return None
        c = self.library.pop(0)
        self.hand.append(c)
        return c
