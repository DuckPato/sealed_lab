"""Card-name matching: Arena / MTGO / user-typed names -> corpus `card_id`.

Matching ladder (first hit wins):
  1. exact 17lands name                      ("Fire // Ice")
  2. exact front-face name                   (Arena writes "Fire" for the split card "Fire // Ice")
  3. normalised full name                    (casefold, accents stripped, punctuation dropped, "A-" Alchemy prefix)
  4. normalised front face / back face       (only if unambiguous)
  5. difflib ratio >= 0.92 on normalised keys (returned with a warning)
For the GUI, `NameIndex.candidates` ranks the closest names at any similarity and `analyse_pool` lists every guessed name of a
pasted pool (fuzzy matches and unmatched names with a close candidate) so the user can confirm or change each guess;
`apply_overrides` then maps the pasted names to the confirmed ones before the pool is resolved.
Basic lands always resolve (snow-covered basics map to the plain basic when the snow one is not in the set).
"""
from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass

from sealed_lab.config import BASICS

FUZZY_CUTOFF = 0.92
GUESS_CUTOFF = 0.60       # an unmatched name is replaced by its closest card only when at least this similar (else "not recognised")
CANDIDATE_CUTOFF = 0.30   # the alternatives offered to the user go down to this similarity
_SEP = " // "
_NON_ALNUM = re.compile(r"[^0-9a-z]+")
_APOS = re.compile(r"[\u2019\u2018\u02bc`'\u00b4]")


def normalize(name: str) -> str:
    """Casefold, strip accents / punctuation, collapse whitespace: "Lim-Dûl's Vault" -> "lim duls vault"."""
    s = unicodedata.normalize("NFKD", name.replace("\u00a0", " "))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = _APOS.sub("", s.casefold())          # "Gideon's" -> "gideons"
    s = s.replace("&", " and ")
    return _NON_ALNUM.sub(" ", s).strip()


_BASIC_NORM = {normalize(b): b for b in BASICS}


def _strip_alchemy(name: str) -> str:
    return name[2:] if name.startswith("A-") else name


@dataclass(frozen=True)
class Match:
    card_id: int | None
    name: str | None          # canonical 17lands name (None if unmatched)
    how: str                  # exact | front | normalized | face | fuzzy | basic | none
    is_basic: bool = False
    warning: str = ""

    @property
    def ok(self) -> bool:
        return self.card_id is not None


class NameIndex:
    """Name lookup over the corpus `cards` table (needs `card_id`, `name`, and optionally `front_name`, `is_basic`)."""

    def __init__(self, cards):
        names = cards["name"].to_list()
        ids = [int(i) for i in cards["card_id"].to_list()]
        fronts = cards["front_name"].to_list() if "front_name" in cards.columns else [n.split(_SEP)[0] for n in names]
        basics = cards["is_basic"].to_list() if "is_basic" in cards.columns else [n in BASICS for n in names]
        self.names: dict[int, str] = dict(zip(ids, names))
        self.is_basic: dict[int, bool] = {i: bool(b) for i, b in zip(ids, basics)}
        self._exact: dict[str, int] = {}
        self._front: dict[str, int] = {}
        self._norm: dict[str, set[int]] = {}
        self._face: dict[str, set[int]] = {}
        for i, n, f in zip(ids, names, fronts):
            f = f or n.split(_SEP)[0]
            self._exact.setdefault(n, i)
            self._front.setdefault(f, i)
            self._norm.setdefault(normalize(n), set()).add(i)
            for k, face in enumerate(n.split(_SEP)):
                (self._norm if k == 0 else self._face).setdefault(normalize(face), set()).add(i)
        # basics: plain basic name -> id (for snow basics and the "always resolves" rule)
        self._basic_by_name: dict[str, int] = {n: i for i, n in self.names.items() if n in BASICS}
        self._keys = sorted(self._norm)

    # ------------------------------------------------------------------ matching
    def _basic(self, name: str) -> Match | None:
        canon = _BASIC_NORM.get(normalize(name))
        if canon is None:
            return None
        cid = self._basic_by_name.get(canon)
        if cid is None and canon.startswith("Snow-Covered "):
            cid = self._basic_by_name.get(canon[len("Snow-Covered "):])
        if cid is None:
            return Match(None, None, "basic", True, f"basic land {name!r} is not in this set's card table")
        return Match(cid, self.names[cid], "basic", True)

    def match(self, raw: str, fuzzy: bool = True) -> Match:
        name = " ".join(raw.replace("\u00a0", " ").split())
        if not name:
            return Match(None, None, "none")
        b = self._basic(name)
        if b is not None:
            return b
        if name in self._exact:
            return self._mk(self._exact[name], "exact")
        if name in self._front:
            return self._mk(self._front[name], "front")
        for cand in (name, _strip_alchemy(name)):
            key = normalize(cand)
            hit = self._norm.get(key)
            if hit:
                if len(hit) == 1:
                    return self._mk(next(iter(hit)), "normalized")
                # ambiguous: prefer the one whose full name normalises to the key exactly
                exact = [i for i in hit if normalize(self.names[i]) == key]
                pick = min(exact or hit)
                return self._mk(pick, "normalized", f"{name!r} is ambiguous; using {self.names[pick]!r}")
        for cand in (name, _strip_alchemy(name)):
            for face in cand.split(_SEP):  # user typed "A // B" for a card stored under another spelling
                hit = self._norm.get(normalize(face)) or self._face.get(normalize(face))
                if hit and len(hit) == 1:
                    return self._mk(next(iter(hit)), "face")
        if not fuzzy:
            return Match(None, None, "none")
        key = normalize(_strip_alchemy(name))
        close = difflib.get_close_matches(key, self._keys, n=1, cutoff=FUZZY_CUTOFF)
        if close:
            i = min(self._norm[close[0]])
            return Match(i, self.names[i], "fuzzy", self.is_basic.get(i, False),
                         f"{name!r} not found; assuming {self.names[i]!r}")
        return Match(None, None, "none")

    def candidates(self, raw: str, n: int = 5, cutoff: float = CANDIDATE_CUTOFF) -> list[tuple[int, str, float]]:
        """The `n` closest non-basic card names to `raw` as [(card_id, canonical name, similarity 0..1)], best first.

        Similarity = difflib ratio of the normalised names (accents, case and punctuation ignored); a name that contains the typed
        text (or the other way round) scores at least 0.85 x the length ratio, so a half-typed "Lightning Str" finds its card."""
        key = normalize(_strip_alchemy(" ".join(raw.replace(" ", " ").split())))
        if not key:
            return []
        sm = difflib.SequenceMatcher()
        sm.set_seq2(key)
        scored: list[tuple[float, int]] = []
        for k in self._keys:
            ids = self._norm[k]
            cid = min(ids)
            if self.is_basic.get(cid, False):
                continue
            sm.set_seq1(k)
            if sm.real_quick_ratio() < cutoff or sm.quick_ratio() < cutoff:
                r = 0.0
            else:
                r = sm.ratio()
            if (key in k or k in key) and min(len(key), len(k)) >= 4:
                r = max(r, 0.85 * min(len(key), len(k)) / max(len(key), len(k)))
            if r >= cutoff:
                scored.append((r, cid))
        scored.sort(key=lambda t: (-t[0], self.names[t[1]]))
        out: list[tuple[int, str, float]] = []
        seen: set[int] = set()
        for r, cid in scored:
            if cid in seen:
                continue
            seen.add(cid)
            out.append((cid, self.names[cid], float(r)))
            if len(out) >= n:
                break
        return out

    def _mk(self, cid: int, how: str, warning: str = "") -> Match:
        return Match(cid, self.names[cid], how, self.is_basic.get(cid, False), warning)

    def front_name(self, name: str) -> str:
        """Front face of a 17lands name ("A // B" -> "A")."""
        return name.split(_SEP)[0]


@dataclass
class Resolved:
    counts: dict[int, int]          # card_id -> copies (basics included)
    unmatched: list[str]            # names (with count in the text) that could not be matched
    warnings: list[str]             # fuzzy / ambiguous matches
    n_unmatched_cards: int = 0


def resolve_names(named: dict[str, int], index: NameIndex, drop_basics: bool = False) -> Resolved:
    """Resolve {name: copies} to {card_id: copies}. Unmatched names are reported, never raised."""
    counts: dict[int, int] = {}
    unmatched: list[str] = []
    warnings: list[str] = []
    n_un = 0
    for name, n in named.items():
        m = index.match(name)
        if not m.ok:
            unmatched.append(name)
            n_un += n
            continue
        if m.warning:
            warnings.append(m.warning)
        if m.is_basic and drop_basics:
            continue
        counts[m.card_id] = counts.get(m.card_id, 0) + n
    return Resolved(counts, unmatched, warnings, n_un)


# ---------------------------------------------------------------------------------------------
# guessed names of a pasted pool (GUI-independent; shared by the real API and the fake backend)
# ---------------------------------------------------------------------------------------------

def apply_overrides(named: dict[str, int], overrides: dict[str, str] | None) -> dict[str, int]:
    """Replace pasted names by the names the user confirmed: `overrides[pasted name]` = canonical name, or "" to drop the card."""
    if not overrides:
        return dict(named)
    out: dict[str, int] = {}
    for name, n in named.items():
        if name in overrides:
            target = overrides[name]
            if not target:
                continue
            out[target] = out.get(target, 0) + n
        else:
            out[name] = out.get(name, 0) + n
    return out


def _cand_dicts(cands: list[tuple[int, str, float]]) -> list[dict]:
    return [dict(card_id=int(i), name=nm, score=round(float(sc), 3)) for i, nm, sc in cands]


def analyse_pool(text: str, index: NameIndex, n_candidates: int = 5) -> dict:
    """What the pasted pool / deck text looks like to a set's card table:

    {n_copies, matched (copies recognised), guesses [{raw, count, section 'deck'|'side', card_id, name, score, how, candidates}],
     unmatched [{raw, count, section, candidates}] (nothing close enough), ignored [unreadable lines], has_deck (an Arena deck export
     with a Deck AND a Sideboard section: the Deck section is the user's own build), n_main, n_side}.

    A *guess* is a name that only matched through difflib (>= 0.92) or that matched nothing but has a card at least 0.60 similar;
    `card_id` / `name` is the closest card.  Basic lands never count."""
    from sealed_lab.deck.arena import parse_arena_full

    parsed = parse_arena_full(text)
    out: dict = dict(n_copies=0, matched=0, guesses=[], unmatched=[], ignored=list(parsed.ignored),
                     has_deck=bool(parsed.had_headers and parsed.main and parsed.side), n_main=parsed.n_main, n_side=parsed.n_side)
    seen: dict[str, dict] = {}
    for section, d in (("deck", parsed.main), ("side", parsed.side)):
        for raw, n in d.items():
            m = index.match(raw)
            if m.ok and m.is_basic:
                continue
            out["n_copies"] += n
            if m.ok and m.how != "fuzzy":
                out["matched"] += n
                continue
            if raw in seen:                                   # the same pasted name in both sections: one entry, summed copies
                seen[raw]["count"] += n
                continue
            cands = index.candidates(raw, n_candidates)
            if m.ok:                                          # fuzzy match: guessed
                if not any(c[0] == m.card_id for c in cands):
                    cands = [(m.card_id, m.name, FUZZY_CUTOFF)] + cands[:n_candidates - 1]
                e = dict(raw=raw, count=n, section=section, card_id=int(m.card_id), name=m.name, how="fuzzy",
                         score=next((round(c[2], 3) for c in cands if c[0] == m.card_id), FUZZY_CUTOFF), candidates=_cand_dicts(cands))
                out["guesses"].append(e)
            elif cands and cands[0][2] >= GUESS_CUTOFF:
                e = dict(raw=raw, count=n, section=section, card_id=int(cands[0][0]), name=cands[0][1], how="closest",
                         score=round(cands[0][2], 3), candidates=_cand_dicts(cands))
                out["guesses"].append(e)
            else:
                e = dict(raw=raw, count=n, section=section, candidates=_cand_dicts(cands))
                out["unmatched"].append(e)
            seen[raw] = e
    return out
