"""Cross-set lessons: what the per-set facts have in common, as counts (never pooled effects).

    build_cross_facts(sets=None)   -> dict   (facts of the pseudo-set "ALL", same structure as a per-set facts file)
    cross_from_facts(per_set)      -> dict   (facts= the facts dict, lessons= the structure the Analysis tab shows; pure and deterministic)
    render_text(facts)             -> str    (the digest the writer reads; `report_facts.render_text` calls it for set "ALL")

PRE-REGISTERED RULES (written before any result was seen; not to be changed afterwards)
R1. Only per-set results with a Likely or Strong badge count. Weak/NOCLEAR results count as "no clear result". Never average or pool effects across sets; no cross-set significance tests.
R2. Every lesson is a count over set-format results: "clear in N of M" with the direction, plus the list of sets and the range of the clear effects. M = number of set-formats where the quantity exists.
R3. A lesson is called a PATTERN only if >= 3 set-formats show a clear result in the same direction AND none shows a clear result in the opposite direction. If both directions occur it is MIXED; if fewer than 3 clear results in total it is RARE. Patterns, mixed and rare all go into the facts (rare ones as NOCLEAR-style items so the writer cannot claim them).
R4. Bo1 and Bo3 are counted separately (M per format), and also listed together.
Lessons to compute (each per format):
 L1 Playing first: PLAY facts (direction, size range).
 L2 Style spectrum: among SPEC claims (a pair whose spectrum has a clear winning end), does the LOWER-curve end win (end with lower avg_mv) or the higher-curve end? Count per pair-set-format; report which sets/pairs.
 L3 Splashing: SPLASH facts direction.
 L4 Land count: SHAPEDET lands bands: clear results for the low-land band (16 or fewer) positive/negative, and for the high-land band.
 L5 Creature count: SHAPEDET creatures bands similarly (low band / high band).
 L6 Trap rares: number of sets with at least one clear trap rare (CARD facts flagged as trap) and the total count.
 L7 Colour pairs: for each of the 10 pairs, clear above/below average counts across sets (a pair that is clearly above average in >= 3 sets and never clearly below is a pattern).
 L8 Builder: BUILDER facts - in how many set-formats the builder plays clearly fewer lands / fewer creatures / lower mana value (and more), and BUILDERCHECK clear results count.
 L9 Descriptive ranges (no claims): mulligan rate and typical game length per format across sets (GLANCE), as min-max with the sets.

Implementation notes (how the rules above were made exact; they add no rule)
- Order of evaluation of R3: PATTERN first; else MIXED when both directions occur (even one against one); else RARE (zero, one or two clear
  results in one direction).  A MIXED result is a fact (badge "likely", the text says mixed); a RARE one is a NOCLEAR item.
- "Together" (R4) is the same count over the Bo1 and Bo3 set-formats of the item (id XSET.both.*), judged by the same R3; it adds up counts only.
- A result "exists" when the per-set facts hold it as a claim or as a NOCLEAR item (a pair hidden for lack of decks, or a builder metric without
  usable data, does not exist and is not in M).  A fact whose badge is not Likely/Strong is never counted as clear, whatever list it is in.
- L2: a pair-set-format with a clear winning end counts as "lower-curve end wins" or "higher-curve end wins" by the two ends' average mana value
  (stored with the SPEC fact); equal averages count as no clear result.  Unit: a pair in a set and format.
- L4/L5: the bands are L16 (16 or fewer lands) / L18 (18 or more) and C1 / C3 (the lowest and highest third of the creature counts).  The
  quantity exists in a set-format when the per-set facts list a band of that dimension or its NOCLEAR item.
- L6: per set-format; the per-set facts list at most three trap rares per format, so the card total is of those listed.
- L8: direction is the builder minus the players on the same pools (more / fewer; higher / lower mana value).  BUILDERCHECK: real decks shaped
  like the builder's did better (+) or worse (-).
- Ids: XSET.<bo1|bo3|both>.<PLAY|SPEC_LOWCURVE|SPLASH|LANDS_LOW|LANDS_HIGH|CREATURES_LOW|CREATURES_HIGH|TRAPS|PAIR.<XY>|BUILDER.<metric>|BUILDERCHECK|GLANCE>.
"""
from __future__ import annotations

import re

from sealed_lab import report as R
from sealed_lab.model import design
from sealed_lab.report_facts import FACTS_VERSION, FMT_WORD, FORMATS, LIKELY_UP, _fact

CROSS_VERSION = 1
MIN_PATTERN = 3                      # R3
LEVELS = ("bo1", "bo3", "both")
LEVEL_WORD = {"bo1": "Best of 1", "bo3": "Best of 3", "both": "Best of 1 and Best of 3 together"}
FMT_SHORT = {"bo1": "Bo1", "bo3": "Bo3"}
VERDICT_WORD = {"pattern": "a pattern", "mixed": "mixed", "rare": "rare (too few clear results)"}
PAIR_KEY = re.compile(r"^[WUBRG]{2}$")
BUILDER_LABEL = {"lands": ("lands", "more lands", "fewer lands", "lands", 1), "creatures": ("creatures", "more creatures", "fewer creatures", "creatures", 1),
                 "two_drops": ("two-drops", "more two-drops", "fewer two-drops", "two-drops", 1),
                 "mean_mv": ("average mana value", "a higher average mana value", "a lower average mana value", "mana value", 2)}


class OldFacts(ValueError):
    """A per-set facts file without the fields this module needs (it was made by an older version)."""


# ---------------------------------------------------------------------------------------------
# reading the per-set facts
# ---------------------------------------------------------------------------------------------

def _get(F: dict, fid: str) -> tuple[dict | None, bool]:
    """(fact, exists): the claim, or None with exists True for a NOCLEAR item."""
    f = (F.get("facts") or {}).get(fid)
    if f is not None:
        return f, True
    return None, fid in (F.get("noclear") or {})


def _is_clear(f: dict | None) -> bool:
    return f is not None and str(f.get("badge", "")).lower() in LIKELY_UP


def _val(f: dict, key: str = "value") -> float:
    if key not in f:
        raise OldFacts(f"fact {f.get('id')} has no '{key}': the per-set facts were made by an older version; rebuild them (cli report-facts SET)")
    return float(f[key])


def _present(F: dict, fmt: str) -> bool:
    if fmt not in (F.get("formats") or []):
        return False
    return any(x.get("fmt") == fmt for x in list((F.get("facts") or {}).values()) + list((F.get("noclear") or {}).values()))


def _obs(code: str, fmt: str, tag: str, f: dict | None, direction=None, dec: int = 1, cell: str = "", values=None) -> dict:
    """One set-format result.  `clear` only for a Likely/Strong claim; `dir` +1 / -1 / 0 (0: clear but with no direction to count)."""
    clear = _is_clear(f)
    d = 0
    vals: list[float] = []
    if clear:
        raw = list(values) if values is not None else [_val(f)]
        vals = [round(v, dec) for v in raw]
        d = direction(f) if direction else (1 if raw[0] > 0 else -1 if raw[0] < 0 else 0)
        if d == 0:
            clear = False
    return dict(set=code, fmt=fmt, tag=tag, clear=clear, dir=d if clear else 0, values=vals if clear else [], badge=str(f.get("badge", "")) if clear else "",
                cell=cell if clear else "")


def _signed(v: float, dec: int) -> str:
    return f"{v:+.{dec}f}"


def _cell(tag: str, vals: list[float], dec: int, badge: str) -> str:
    return (tag + " " if tag else "") + ", ".join(_signed(v, dec) for v in vals) + f" {badge.capitalize()}"


# ---------------------------------------------------------------------------------------------
# the items (one per quantity) and their observations
# ---------------------------------------------------------------------------------------------

def _item(lesson: str, key: str, label: str, short: str, up: str, down: str, unit: str, dec: int = 1, per: str = "sets", extra_nums=()) -> dict:
    return dict(lesson=lesson, key=key, label=label, short=short, up=up, down=down, unit=unit, dec=dec, per=per, extra_nums=list(extra_nums), obs=[])


def _collect(per_set: dict[str, dict]) -> list[dict]:
    items: list[dict] = []

    def add(it: dict) -> dict:
        items.append(it)
        return it

    def simple(it: dict, fid: str, short: str | None = None, direction=None) -> None:
        for code, F in per_set.items():
            for fmt in FORMATS:
                f, ex = _get(F, fid.format(fmt=fmt))
                if ex:
                    it["obs"].append(_obs(code, fmt, "", f, direction, it["dec"], values=None))

    # L1 -----------------------------------------------------------------------------------
    simple(add(_item("L1", "PLAY", "Playing first against playing second", "play first", "playing first did better", "playing second did better", "points")),
           "PLAY.{fmt}")
    # L2 -----------------------------------------------------------------------------------
    it = add(_item("L2", "SPEC_LOWCURVE", "Style spectrum inside a colour pair: the end with the lower average mana value against the end with the higher one",
                   "spectrum", "the lower-curve end won", "the higher-curve end won", "points", per="pairs in sets"))
    for code, F in per_set.items():
        for fmt in FORMATS:
            pre = f"SPEC.{fmt}."
            keys = sorted({i[len(pre):] for i in list(F.get("facts") or {}) + list(F.get("noclear") or {}) if i.startswith(pre)})
            for k in keys:
                f, _ = _get(F, pre + k)
                if _is_clear(f):
                    ends = f.get("ends") or {}
                    if "winner_avg_mv" not in ends or "other_avg_mv" not in ends:
                        raise OldFacts(f"fact {f['id']} has no end mana values: the per-set facts were made by an older version; rebuild them")
                    d = 1 if ends["winner_avg_mv"] < ends["other_avg_mv"] else -1 if ends["winner_avg_mv"] > ends["other_avg_mv"] else 0
                    ob = _obs(code, fmt, k, f, lambda _f, d=d: d, 1)
                    if ob["clear"]:
                        ob["cell"] = f"{k} {'low' if d > 0 else 'high'}-curve end {_signed(ob['values'][0], 1)} {ob['badge'].capitalize()}"
                else:
                    ob = _obs(code, fmt, k, None)
                it["obs"].append(ob)
    # L3 -----------------------------------------------------------------------------------
    simple(add(_item("L3", "SPLASH", "Splashing a third colour", "splash", "splashing paid off", "splashing cost", "points")), "SPLASH.{fmt}")
    # L4, L5 -------------------------------------------------------------------------------
    for lesson, dim, lo_b, hi_b, lo_l, hi_l, lo_s, hi_s, nums in (
            ("L4", "lands", "L16", "L18", "Decks with 16 or fewer lands against an average deck", "Decks with 18 or more lands against an average deck",
             "16 or fewer lands", "18 or more lands", {"LANDS_LOW": [16], "LANDS_HIGH": [18]}),
            ("L5", "creatures", "C1", "C3", "Decks in the lowest third by creature count against an average deck",
             "Decks in the highest third by creature count against an average deck", "few creatures", "many creatures", {})):
        for key, band, label, short in ((f"{dim.upper()}_LOW", lo_b, lo_l, lo_s), (f"{dim.upper()}_HIGH", hi_b, hi_l, hi_s)):
            it = add(_item(lesson, key, label, short, "the band did better than an average deck", "the band did worse than an average deck", "points",
                           extra_nums=nums.get(key, [])))
            for code, F in per_set.items():
                for fmt in FORMATS:
                    pre = f"SHAPEDET.{fmt}.{dim}."
                    if not (any(i.startswith(pre) for i in (F.get("facts") or {})) or f"SHAPEDET.{fmt}.{dim}" in (F.get("noclear") or {})):
                        continue
                    f, _ = _get(F, pre + band)
                    it["obs"].append(_obs(code, fmt, "", f))
    # L6 -----------------------------------------------------------------------------------
    it = add(_item("L6", "TRAPS", "Trap rares (rares and mythics most players play and that lose)", "trap rares", "", "a trap rare was listed",
                   "points per copy", per="sets"))
    for code, F in per_set.items():
        for fmt in FORMATS:
            if not _present(F, fmt):
                continue
            traps = [f for f in (F.get("facts") or {}).values() if f.get("fmt") == fmt and f.get("kind") == "card_trap" and _is_clear(f)]
            if traps:
                vals = [round(_val(f), 1) for f in sorted(traps, key=lambda f: f["id"])]
                ob = _obs(code, fmt, "", traps[0], lambda _f: -1, 1, values=vals)
                ob["names"] = [f.get("name", f["id"]) for f in sorted(traps, key=lambda f: f["id"])]
                ob["badge"] = "strong" if all(f["badge"] == "strong" for f in traps) else "likely"
                ob["cell"] = ", ".join(f"{n} {_signed(v, 1)} {f['badge'].capitalize()}" for n, v, f in zip(ob["names"], vals, sorted(traps, key=lambda f: f["id"])))
            else:
                ob = _obs(code, fmt, "", None)
                ob["names"] = []
            it["obs"].append(ob)
    # L7 -----------------------------------------------------------------------------------
    for pair in design.PAIRS:
        it = add(_item("L7", f"PAIR.{pair}", f"{R.group_label(pair)} decks against an average deck", pair, "clearly above average", "clearly below average", "points"))
        simple(it, "PAIR.{fmt}." + pair)
    # L8 -----------------------------------------------------------------------------------
    for m, (word, more, fewer, unit, dec) in BUILDER_LABEL.items():
        it = add(_item("L8", f"BUILDER.{m}", f"Deck builder against players on the same pools: {word}", f"builder {word}", f"the builder plays {more}",
                       f"the builder plays {fewer}", unit, dec=dec))
        simple(it, "BUILDER.{fmt}." + m)
    it = add(_item("L8", "BUILDERCHECK", "Real player decks shaped like the builder's, same colour pair and skill", "builder-shaped decks",
                   "such decks did better", "such decks did worse", "points"))
    simple(it, "BUILDERCHECK.{fmt}")
    for it in items:
        for ob in it["obs"]:
            if ob["clear"] and not ob["cell"]:
                ob["cell"] = _cell(it["short"] if it["lesson"] in ("L4", "L5", "L7", "L8") else "", ob["values"], it["dec"], ob["badge"])
    return items


# ---------------------------------------------------------------------------------------------
# counting (R2, R3, R4)
# ---------------------------------------------------------------------------------------------

def verdict_of(n_up: int, n_down: int) -> str:
    """R3: 'pattern' (>= 3 in one direction and none in the other), else 'mixed' (both directions occur), else 'rare'."""
    if (n_up >= MIN_PATTERN and n_down == 0) or (n_down >= MIN_PATTERN and n_up == 0):
        return "pattern"
    if n_up and n_down:
        return "mixed"
    return "rare"


def _label_of(ob: dict, level: str) -> str:
    s = f"{ob['set']} {ob['tag']}".strip()
    return f"{s} ({FMT_SHORT[ob['fmt']]})" if level == "both" else s


def _tally(it: dict, level: str) -> dict:
    obs = [o for o in it["obs"] if level == "both" or o["fmt"] == level]
    up = [o for o in obs if o["clear"] and o["dir"] > 0]
    down = [o for o in obs if o["clear"] and o["dir"] < 0]
    t = dict(m=len(obs), n=len(up) + len(down), n_up=len(up), n_down=len(down), verdict=verdict_of(len(up), len(down)))
    for name, grp in (("up", up), ("down", down)):
        vals = [v for o in grp for v in o["values"]]
        t[name] = dict(sets=sorted({_label_of(o, level) for o in grp}), lo=min(vals) if vals else None, hi=max(vals) if vals else None)
    t["cards"] = sum(len(o.get("names", [])) for o in down) if it["key"] == "TRAPS" else None
    t["n_sets"] = len({o["set"] for o in up + down}) if it["key"] == "TRAPS" else None
    return t


def _range_text(g: dict, dec: int, unit: str) -> tuple[str, list[float]]:
    lo, hi = g["lo"], g["hi"]
    if lo == hi:
        return f"effect {_signed(lo, dec)} {unit}", [lo]
    return f"effects {_signed(lo, dec)} to {_signed(hi, dec)} {unit}", [lo, hi]


def _unit_noun(it: dict, level: str) -> str:
    if level == "both":
        return "set results" if it["per"] == "sets" else "pair results in sets"
    return "sets" if it["per"] == "sets" else "pairs in sets"


def _text(it: dict, level: str, t: dict) -> tuple[str, list[float]]:
    nums: list[float] = [t["m"], t["n"], t["n_up"], t["n_down"], *it["extra_nums"]]
    where = LEVEL_WORD[level]
    head = f"{it['label']}, {where}: clear in {t['n']} of {t['m']} {_unit_noun(it, level)}"
    parts = []
    for name, phrase, k in (("up", it["up"], t["n_up"]), ("down", it["down"], t["n_down"])):
        if k:
            g = t[name]
            rt, rn = _range_text(g, it["dec"], it["unit"])
            nums += rn
            parts.append(f"{k} where {phrase} ({', '.join(g['sets'])}; {rt})")
    text = head + (": " + "; ".join(parts) if parts else "")
    if it["key"] == "TRAPS" and t["n"]:
        text += f"; {t['cards']} trap rares listed in {t['n_sets']} sets"
        nums += [t["cards"], t["n_sets"]]
    if t["verdict"] == "pattern":
        text += ". A pattern: every clear result goes the same way"
    elif t["verdict"] == "mixed":
        text += ". Mixed: clear results go both ways"
    else:
        text += ". Too few clear results to call a pattern"
    return text + ".", nums


def _glance(per_set: dict[str, dict]) -> list[dict]:
    """L9: per set-format (mulligan rate in % of games, typical game length in turns)."""
    rows = []
    for code, F in per_set.items():
        for fmt in FORMATS:
            f = (F.get("facts") or {}).get(f"GLANCE.{fmt}")
            if f is not None:
                rows.append(dict(set=code, fmt=fmt, mulligan=round(_val(f, "mulligan_pct"), 1), turns=round(_val(f, "median_turns"), 1)))
    return rows


def _glance_fact(rows: list[dict], level: str) -> tuple[str, list[float]] | None:
    rr = [r for r in rows if level == "both" or r["fmt"] == level]
    if not rr:
        return None
    nums: list[float] = [len(rr)]
    bits = []
    for key, word, suffix in (("mulligan", "games with a mulligan", "%"), ("turns", "typical game length", " turns")):
        lo, hi = min(r[key] for r in rr), max(r[key] for r in rr)
        who = lambda v: ", ".join(sorted({(f"{r['set']} ({FMT_SHORT[r['fmt']]})" if level == "both" else r["set"]) for r in rr if r[key] == v}))
        if lo == hi:
            bits.append(f"{word} {lo:.1f}{suffix} in every one")
            nums.append(lo)
            continue
        bits.append(f"{word} from {lo:.1f}{suffix} ({who(lo)}) to {hi:.1f}{suffix} ({who(hi)})")
        nums += [lo, hi]
    return f"Across {len(rr)} {'set results' if level == 'both' else 'sets'}, {LEVEL_WORD[level]}: " + "; ".join(bits) + ". Descriptive only.", nums


# ---------------------------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------------------------

LESSON_TITLES = {
    "L1": ("Playing first", "Does playing first win more than playing second?"),
    "L2": ("Style spectrum", "Inside a colour pair, does the lower-curve or the higher-curve end win?"),
    "L3": ("Splashing", "Does splashing a third colour pay off or cost?"),
    "L4": ("Land count", "Do decks with few or many lands do better than an average deck?"),
    "L5": ("Creature count", "Do decks with few or many creatures do better than an average deck?"),
    "L6": ("Trap rares", "In how many sets is there a clear trap rare?"),
    "L7": ("Colour pairs", "For each colour pair: clearly above or below an average deck, set by set."),
    "L8": ("The deck builder", "Where the builder's decks differ from players' decks on the same pools."),
    "L9": ("The set at a glance", "Mulligan rate and game length across sets (descriptive: no claim)."),
}


def cross_from_facts(per_set: dict[str, dict]) -> dict:
    """The cross-set facts and lessons from the per-set facts ({set code: facts}).  Pure and deterministic (sets are taken in code order)."""
    per_set = {c.upper(): per_set[c] for c in sorted(per_set, key=str.upper)}
    items = [it for it in _collect(per_set) if it["obs"]]
    facts: list[dict] = []
    noclear: list[dict] = []
    for it in items:
        it["levels"] = {}
        for level in LEVELS:
            t = _tally(it, level)
            if t["m"] == 0:
                continue
            text, nums = _text(it, level, t)
            fid = f"XSET.{level}.{it['key']}"
            t["id"], t["text"] = fid, text
            it["levels"][level] = t
            if t["verdict"] == "rare":
                noclear.append(dict(id=fid, fmt=level, text=f"{it['label']}, {LEVEL_WORD[level]}: no pattern across sets"))
            else:
                facts.append(_fact(fid, "xset", text, nums, "likely", "observed", fmt=level, lesson=it["lesson"], verdict=t["verdict"],
                                   n_clear=t["n"], m=t["m"], n_up=t["n_up"], n_down=t["n_down"]))
    rows = _glance(per_set)
    glance = []
    for level in LEVELS:
        g = _glance_fact(rows, level)
        if g:
            text, nums = g
            fid = f"XSET.{level}.GLANCE"
            facts.append(_fact(fid, "xset_glance", text, nums, "descriptive", "observed", fmt=level, lesson="L9"))
            glance.append(dict(id=fid, level=level, text=text))
    ids = [f["id"] for f in facts] + [n["id"] for n in noclear]
    assert len(ids) == len(set(ids)), "duplicate fact ids"
    fmts = [f for f in FORMATS if any(f in (F.get("formats") or []) for F in per_set.values())]
    out = dict(version=FACTS_VERSION, cross_version=CROSS_VERSION, report_version=R.REPORT_VERSION, set="ALL", sets=list(per_set), formats=fmts,
               stamps={f"{c}.{f}": s for c, F in per_set.items() for f, s in (F.get("stamps") or {}).items() if s is not None},
               builder_available=any(F.get("builder_available") for F in per_set.values()),
               facts={f["id"]: f for f in facts}, noclear={n["id"]: n for n in noclear})
    return dict(facts=out, lessons=_lessons(items, rows, glance))


def _lessons(items: list[dict], rows: list[dict], glance: list[dict]) -> list[dict]:
    out = []
    for lid, (title, question) in LESSON_TITLES.items():
        if lid == "L9":
            out.append(dict(id=lid, title=title, question=question, items=[], rows=rows, glance=glance))
            continue
        its = [dict(key=i["key"], label=i["label"], short=i["short"], up=i["up"], down=i["down"], unit=i["unit"], per=i["per"], obs=i["obs"], levels=i["levels"]) for i in items if i["lesson"] == lid]
        out.append(dict(id=lid, title=title, question=question, items=its))
    return out


def build_cross_facts(sets: list[str] | None = None, refresh: bool = False, progress=None) -> dict:
    """The facts of "ALL": from the per-set facts of `sets` (default: every set with a model and at least one report)."""
    return build_cross(sets, refresh, progress)["facts"]


def build_cross(sets: list[str] | None = None, refresh: bool = False, progress=None) -> dict:
    from sealed_lab import api, report_facts

    if sets is None:
        sets = sorted(st.set_code for st in api.list_sets() if st.has_model)
    per_set = {}
    for i, s in enumerate(sorted({x.upper() for x in sets})):
        if progress:
            progress(i / max(len(sets), 1), f"Facts of {s}")
        F = report_facts.build_facts(s, refresh=refresh)
        if F.get("facts") or F.get("noclear"):
            per_set[s] = F
    return cross_from_facts(per_set)


def render_text(facts: dict) -> str:
    """Plain text for the writer: the cross-set lessons by format, then the NOCLEAR ids."""
    sets = facts.get("sets") or []
    L = [f"FACTS ALL (cross-set lessons from {len(sets)} sets: {' '.join(sets)}; statistics v{facts['report_version']}). Line = [id] fact. Every lesson is a "
         "count of per-set results; only results with a Likely or Strong badge in their own set count as clear, effects are never averaged. 'A pattern' = "
         "at least 3 clear results in the same direction and none the other way; 'mixed' = both directions occur. Only these ids may back a claim; "
         "say 'probably' for these counts at most, and name the format."]
    by = list(facts["facts"].values())
    for level in LEVELS:
        mine = [f for f in by if f["fmt"] == level]
        if mine:
            L += ["", f"== {LEVEL_WORD[level]} =="] + [f"[{f['id']}] {f['text']}" for f in mine]
    L += ["", "NOCLEAR (no pattern; the only allowed statement: there is no clear pattern across sets):"]
    for level in LEVELS:
        ids = [n["id"] for n in facts["noclear"].values() if n["fmt"] == level]
        if ids:
            L.append("  " + " ".join(ids))
    return "\n".join(L) + "\n"
