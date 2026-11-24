"""Mana sources of a card, read off its oracle text (used by the castability model, `model/castability.py`).

A *source* is anything that gives a deck coloured mana, or a way to get a coloured land:

  lands        basics; duals (untapped / tapped / fast / slow); conditional duals (verge lands: the second colour needs a
               land of a given basic type; "basic land" duals); tri/any-colour lands; pathways and filter lands (counted as
               either colour, never both at once); fetch lands (a basic of the deck's colours, one turn later);
               lands whose coloured mana is restricted to some spells (Hero / Villain / artifact / instant-sorcery ...).
  other cards  mana rocks and dorks (`{T}: Add ...`), restricted producers, treasure makers, land cyclers ("basic landcycling",
               "<type>cycling"), cards that search for a land when they enter / are cast, any-colour converters.

Rules that make the classes behave in the simulation (see `castability.py` for how they are used):

  rule 0  untapped land: works from the turn it is played.
  rule 1  tapped land (also a fetched land that enters tapped): works from the turn after it is played.
  rule 2  fastland ("enters tapped unless you control two or fewer other lands"): untapped when played on turns 1-3.
  rule 3  slowland ("... two or more other lands"): untapped when played on turn 3 or later, tapped before.
  rule 4  non-land source: cast / cycled on turn `min_turn` (its mana value, the cycling cost, +1 for abilities with {T}),
          it works from the turn after (summoning sickness, tapped token, land arriving).

Anything the parser does not recognise falls back to Scryfall's `produced_mana` (the card's `produces`): lands are untapped
duals, other cards are not counted. Restricted mana ("Spend this mana only to cast X") helps only spells of that type.
All text matching is on lower-cased text with reminder text removed; ability lines after the `//` of a "prepared" creature
(a spell the creature lets you cast) are ignored.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

COLORS = "WUBRG"
BIT = {c: 1 << i for i, c in enumerate(COLORS)}
ALL = 31
NONE_WORD = "x-never"       # restriction no spell satisfies ("only to activate abilities")
BASIC_TYPES = {"plains": "W", "island": "U", "swamp": "B", "mountain": "R", "forest": "G"}


@dataclass(frozen=True)
class SourceDesc:
    cat: str                    # category label (counts, report)
    land: bool
    mask: int = 0               # colours produced without any condition
    rmask: int = 0              # extra colours produced only for spells of `restr`
    restr: tuple[str, ...] = ()
    fetch: str = ""             # "" | "basic" (colours of the deck's basic lands) | colour letters of a typed search
    rule: int = 0
    min_turn: int = 0
    mult: int = 1               # sources delivered (a card making two Treasures)
    land_like: bool = False     # counts as a land for the "enough lands" condition
    cond_second: int = 0        # colours added when `cond_need` is met (verge lands)
    cond_need: int = 0          # basic land TYPES (bit mask) one of which another land must have ...
    cond_basic: bool = False    # ... or (True) another land must be a BASIC land of any type
    basic_col: int = 0          # this card is a basic land of these colours
    type_col: int = 0           # basic land types this land has (basics and typed duals)


LAND_NONE = SourceDesc("colourless land", True, land_like=True)


def _strip(text: str) -> str:
    return re.sub(r"\([^)]*\)", "", (text or "").lower())


def _main_lines(text: str, cut: bool = True) -> list[str]:
    """Ability lines of the card, without reminder text and without anything after a bare `//` line (prepared spell)."""
    out = []
    for ln in (text or "").lower().replace("•", "").split("\n"):
        ln = re.sub(r"\([^)]*\)", "", ln).strip()
        if ln == "//":
            if cut:
                break
            continue
        if ln:
            out.append(ln)
    return out


def _colors_in(s: str) -> int:
    m = 0
    for g in re.findall(r"\{([wubrgc])\}", s):
        if g in "wubrg":
            m |= BIT[g.upper()]
    return m


# Any colour. Not: "any combination of {U} and/or {R}" (Vivi: just those colours) and "any color among legendary permanents you
# control" (Plaza of Heroes: needs a permanent, counted as nothing). "Of the chosen color" (a land that picks its colour as it
# enters) counts as any colour: the player picks the colour the deck needs, so the land is a fixer for castability.
_ANYCOL = re.compile(r"mana of any (one )?colou?r(?! among)|any combination of colou?rs|mana in any combination(?! of \{)|one mana of any type"
                     r"|mana of the chosen colou?r", re.I)
_RESTR = re.compile(r"(?:spend this mana only (?:to|on) |this mana can't be spent to )(.*)", re.I)
_QTY = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "x": 1}


def _restriction(effect: str) -> tuple[bool, tuple[str, ...]]:
    """(restricted, words) of the mana an ability makes. Words = spell types the mana can pay for; NONE_WORD = no spell."""
    m = _RESTR.search(effect)
    if not m:
        return False, ()
    r = m.group(1)
    if "can't be spent to cast a nonartifact" in effect:
        return True, ("artifact",)
    segs = re.findall(r"cast (?:an? |the next )?([a-z ,]+?) spells?", r)
    words: set[str] = set()
    for seg in segs:
        words |= {w for w in re.split(r"[^a-z]+", seg) if w and w not in ("a", "an", "or", "and", "of", "the", "with", "mana", "value")}
    return True, tuple(sorted(words)) if words else (NONE_WORD,)


def _mana_lines(text: str, cut: bool = True):
    """Yield (cost, effect) for ability lines `<cost>: add <mana>` (and 'at the beginning of ..., add')."""
    for ln in _main_lines(text, cut):
        m = re.match(r"^(.*?):\s*(add .*)$", ln)
        if m:
            paid = sum(int(t) if t.isdigit() else 1 for t in re.findall(r"\{([^}t]+)\}", m.group(1)))
            if paid <= 2:                      # a {4} mana sink is not a source
                yield m.group(1), m.group(2)
            continue
        m = re.match(r"^(at the beginning of your (?:first )?main phase),\s*(add .*)$", ln)
        if m:
            yield "trigger", m.group(2)


def _fetch_types(s: str) -> str:
    """Colour letters of the land types named in a search clause; 'basic' for 'basic land' / 'land card'."""
    found = {BASIC_TYPES[w] for w in re.findall(r"\b(plains|island|swamp|mountain|forest)\b", s)}
    cols = "".join(c for c in COLORS if c in found)
    if cols and "basic land" not in s:
        return cols
    return "basic"


# ---------------------------------------------------------------------------------------------
# lands
# ---------------------------------------------------------------------------------------------

def _land_entry(txt: str) -> int:
    """Availability rule of a land from its entering-the-battlefield text."""
    if re.search(r"enters? (the battlefield )?tapped unless you control two or more other lands", txt):
        return 3
    if re.search(r"enters? (the battlefield )?tapped unless you control (two or fewer|fewer than three) other lands", txt):
        return 2
    if re.search(r"enters? (the battlefield )?tapped unless", txt):
        return 0                                    # checkland ("unless you control a Plains or an Island"): untapped in practice
    if re.search(r"you may pay \d+ life\. if you don't, it enters (the battlefield )?tapped", txt):
        return 0                                    # shockland: pay 2 life
    if re.search(r"enters? (the battlefield )?tapped", txt):
        return 1
    return 0


def parse_land(name: str, types: str, text: str, produces: str, is_basic: bool) -> SourceDesc:
    tl = (types or "").lower()
    low = (text or "").lower()
    txt = " ".join(_main_lines(text, cut=False))               # both faces of a pathway / modal land
    prod_mask = sum(BIT[c] for c in (produces or "") if c in BIT)
    tcol = sum(BIT[BASIC_TYPES[w]] for w in re.findall(r"\b(plains|island|swamp|mountain|forest)\b", tl.split("—")[-1].split(" - ")[-1])
               if w in BASIC_TYPES)
    if is_basic or tl.startswith("basic land"):
        m = prod_mask or tcol
        return SourceDesc("basic land", True, mask=m, land_like=True, basic_col=m, type_col=tcol or m)
    # fetch lands: "{T}, sacrifice this land: search your library for a ... card, put it onto the battlefield"
    fm = re.search(r"sacrifice (?:this land|\w+)[^:]*:\s*search your library for (?:a|an) ([^.]*?) card[^.]*onto the battlefield( tapped)?", txt)
    if fm:
        ft = _fetch_types(fm.group(1))
        return SourceDesc("fetch land", True, fetch=ft, rule=1 if fm.group(2) else 0, land_like=True, type_col=0)
    mask = 0
    rmask = 0
    restr: tuple[str, ...] = ()
    cond_second = 0
    cond_need = 0
    cond_basic = False
    any_line = False
    for cost, eff in _mana_lines(text, cut=False):
        colored = _colors_in(eff)
        anyc = bool(_ANYCOL.search(eff))
        m = ALL if anyc else colored
        if not m:
            continue
        restricted, words = _restriction(eff)
        any_line = True
        if restricted:
            if words == (NONE_WORD,):
                continue
            rmask |= m
            restr = tuple(sorted(set(restr) | set(words)))
            continue
        cm = re.search(r"activate (?:this ability )?only if ([^.]*)", eff)
        if cm:
            cond = cm.group(1)
            need = 0
            for w in re.findall(r"\b(plains|island|swamp|mountain|forest)\b", cond):
                need |= BIT[BASIC_TYPES[w]]
            basic_any = bool(re.search(r"control a basic land|control a basic", cond))
            if need or basic_any:
                cond_second |= m
                cond_need |= need
                cond_basic = cond_basic or basic_any
                continue
        mask |= m
    if not any_line and not re.search(r":\s*add ", txt):
        mask = prod_mask                     # no recognisable mana ability: Scryfall's produced_mana
    rule = _land_entry(txt)
    # classification for the category counts
    if rmask and not mask and not cond_second:
        cat = "land: restricted any-colour mana"
    elif cond_second:
        cat = "land: conditional colours (verge / basic-land dual)"
    elif "//" in low or ("\n//" in (text or "").lower()):
        cat = "land: pathway (either colour, once)"
    elif re.search(r"\{[wubrg](?:/[wubrg])?\}[^:]*,\s*\{t\}:\s*add", txt):
        cat = "land: filter (counted as a dual)"
    elif bin(mask).count("1") >= 3 and not rmask:
        cat = "land: three- or any-colour"
    elif bin(mask).count("1") == 2:
        cat = {0: "land: dual (untapped)", 1: "land: dual (enters tapped)", 2: "land: fastland", 3: "land: slowland"}[rule]
    elif bin(mask).count("1") == 1:
        cat = "land: one colour"
    elif rmask:
        cat = "land: restricted any-colour mana"
    else:
        cat = "land: colourless"
    if mask == 0 and not rmask and not cond_second:
        return SourceDesc(cat, True, rule=rule, land_like=True, type_col=tcol)
    return SourceDesc(cat, True, mask=mask, rmask=rmask, restr=restr, rule=rule, land_like=True, cond_second=cond_second,
                      cond_need=cond_need, cond_basic=cond_basic, type_col=tcol)


# ---------------------------------------------------------------------------------------------
# non-land cards
# ---------------------------------------------------------------------------------------------

_CYCLE_RE = re.compile(r"\b((?:basic )?land|plains|island|swamp|mountain|forest)cycling (\{[^\s]*(?:\}\{[^\s]*)*\}|\d+)")
_SEARCH_HEAD = r"search(?:es)? (?:your|their) library for (?:a|an|up to (?:one|two|three|four)|one|two|three|four) "
# the whole clause up to " card" ("a basic Plains, Swamp, or Forest card" must give all three types); without "card" the old cut
_SEARCH_RE = re.compile(_SEARCH_HEAD + r"([^.;]*?) cards?\b|" + _SEARCH_HEAD + r"([^.;]*?)(?:, | and |\.|;|$)")


def _cost_value(sym: str) -> int:
    n = 0
    for tok in re.findall(r"\{([^}]+)\}", sym):
        n += int(tok) if tok.isdigit() else 1
    return n if n else (int(sym) if sym.isdigit() else 1)


def _is_ours(txt: str, pos: int) -> bool:
    """A library search at `pos` is the player's own unless the sentence before it is about a controller / opponent."""
    before = txt[max(0, pos - 60):pos]
    return not re.search(r"controller|opponent|owner|each player|its |their ", before.split(".")[-1])


def parse_spell(name: str, types: str, text: str, produces: str, mv: float) -> SourceDesc | None:
    """Source description of a nonland card, or None when it is not a colour source."""
    mvi = max(int(round(mv)), 0)
    lines = _main_lines(text)
    txt = " ".join(lines)
    tl = (types or "").lower()
    # ---- land cyclers: "basic landcycling {2}", "forestcycling {1}"
    cm = _CYCLE_RE.search(txt)
    if cm:
        kind = cm.group(1)
        fetch = "basic" if kind in ("land", "basic land") or txt[max(0, cm.start() - 6):cm.start()].strip().endswith("basic") else BASIC_TYPES.get(kind, "basic")
        cost = _cost_value(cm.group(2))
        return SourceDesc("landcycler", False, fetch=fetch, rule=4, min_turn=max(cost, 1), land_like=True)
    # ---- mana abilities
    mask = rmask = 0
    restr: tuple[str, ...] = ()
    tapped = False
    for cost, eff in _mana_lines(text):
        if "trigger" == cost or "{t}" in cost or "sacrifice" in cost or re.match(r"^(pay|\{)", cost):
            anyc = bool(_ANYCOL.search(eff))
            m = ALL if anyc else _colors_in(eff)
            if not m:
                continue
            restricted, words = _restriction(eff)
            if restricted:
                if words != (NONE_WORD,):
                    rmask |= m
                    restr = tuple(sorted(set(restr) | set(words)))
                continue
            mask |= m
            tapped = tapped or ("{t}" in cost)
    if mask or rmask:
        cat = "mana producer: " + ("restricted to some spells" if not mask else ("any colour" if mask == ALL else "fixed colours"))
        return SourceDesc(cat, False, mask=mask, rmask=rmask, restr=restr, rule=4, min_turn=max(mvi, 1))
    # ---- treasure makers
    for ln in lines:
        if "treasure" not in ln or not re.search(r"\bcreates? ", ln):
            continue
        ln = re.sub(r"^[\w' ]+ — ", "", ln)                 # ability word ("Genius Industrialist — ...")
        if re.match(r"^(whenever|at the beginning|each|if |[ivx]+\b)", ln):
            continue
        qm = re.search(r"creates? (a|an|one|two|three|four|x)\b[^.]*?treasure", ln)
        mult = _QTY.get(qm.group(1), 1) if qm else 1
        act = re.match(r"^[^:]*\{t\}[^:]*:", ln) is not None
        return SourceDesc("treasure maker", False, mask=ALL, rule=4, min_turn=max(mvi, 1) + (1 if act else 0), mult=mult)
    # ---- library searches for lands
    for ln in lines:
        sm = _SEARCH_RE.search(ln)
        clause = (sm.group(1) or sm.group(2) or "") if sm else ""
        if not sm or "land" not in clause and not re.search(r"plains|island|swamp|mountain|forest", clause):
            continue
        if not _is_ours(ln, sm.start()) or re.search(r"sacrifice (a|two|three) lands?", txt):
            continue
        if re.match(r"^(whenever|at the beginning)", ln):
            continue
        ft = _fetch_types(clause)
        onto = "onto the battlefield" in ln
        if not (onto or "into your hand" in ln or "put it into your hand" in ln or "reveal" in ln):
            continue
        return SourceDesc("land search (" + ("onto the battlefield" if onto else "to hand") + ")", False, fetch=ft, rule=4,
                          min_turn=max(mvi, 1), land_like=True)
    # ---- any-colour converters ("Add two mana in any combination of colors")
    if ("instant" in tl or "sorcery" in tl):
        for ln in lines:
            m = re.match(r"^(?:.*[.,] )?add (a|an|one|two|three|four|x)? ?(?:mana )?.*$", ln)
            if m and _ANYCOL.search(ln) and ":" not in ln:
                n = _QTY.get(m.group(1) or "a", 1)
                return SourceDesc("converter (any colour)", False, mask=ALL, rule=4, min_turn=max(mvi, 1), mult=max(n, 1))
    return None


def describe_card(name: str, types: str, text: str, produces: str, is_land: bool, is_basic: bool, mv: float) -> SourceDesc | None:
    """Source description of one card (None = not a mana source)."""
    if is_land:
        return parse_land(name, types, text, produces, is_basic)
    return parse_spell(name, types, text, produces, mv)


def static_colors(d: SourceDesc, basic_mask: int = 0) -> int:
    """Colours a source can ever give a deck with these basic lands (for 'sources per colour' counts); restricted mana excluded."""
    m = d.mask | d.cond_second
    if d.fetch:
        m |= basic_mask if d.fetch == "basic" else sum(BIT[c] for c in d.fetch)
    return m


def category_counts(cards) -> dict[str, list[str]]:
    """{category: [card names]} over a card table (polars DataFrame with the corpus card columns)."""
    out: dict[str, list[str]] = {}
    cols = cards.columns
    for r in cards.iter_rows(named=True):
        d = describe_card(r["name"], r.get("types") or "", r.get("oracle_text") or "", r.get("produces") or "", bool(r["is_land"]),
                          bool(r["is_basic"]), float(r.get("mana_value") or 0.0))
        if d is None:
            continue
        out.setdefault(d.cat, []).append(r["name"])
    return out
