"""Deterministic checker of a written report against its facts file (core of scripts/check_report.py; used by `cli report-install`).

What it checks, per sentence (the sentences are exactly the lines `export_html.markdown_to_html` renders, headings included):
  * a number or a direction word (more, better, best, wins, costs, ...; digits and spelled-out numbers alike) needs a tag with a known id;
  * every number must be the tagged fact's EFFECT (its `value`, or the middle of its "range A to B"), the ends of its range (only in a sentence
    that speaks of a range), or one of its other figures (shares, win rates, counts) - the others never as "points" and never as the size of
    a "more / better / less" comparison;
  * the direction words must agree with the sign of the tagged effect (wins more / better / pays off for a positive one, less / worse /
    costs / loses for a negative one);
  * a no-clear-difference fact allows no direction word at all.
"""
from __future__ import annotations

import re

_V_POS = r"wins?(?!\s+rates?)|beats?|gains?|pays off|outperforms?|improves?|adds?|helps?|leads?|boosts?"
_V_NEG = r"loses?|lost|costs|costing|hurts?|underperforms?|trails?"
_Q_POS = r"more|higher|greater|bigger|larger|above"
_Q_NEG = r"less|fewer|lower|smaller|below"
_A_POS = r"better|best|stronger|strongest|highest|ahead|edge|advantage|underrated|under-rated"
_A_NEG = r"worse|worst|weaker|weakest|lowest|behind|disadvantage|overrated|over-rated"
DIRECTION = re.compile(rf"\b(?:{_V_POS}|{_V_NEG}|{_Q_POS}|{_Q_NEG}|{_A_POS}|{_A_NEG})\b", re.I)
# words that only give a sign (section vocabulary such as "trap rare" and "pick up" is not a claim on its own, so they are not in DIRECTION)
_A_POS_X = r"good|top|pick up|picking up"
_A_NEG_X = r"traps?|avoid|stop auto-including|do not auto-include|leave out|bad|poor"
_TOKEN = re.compile(rf"\b(?:(?P<vp>{_V_POS})|(?P<vn>{_V_NEG})|(?P<qp>{_Q_POS})|(?P<qn>{_Q_NEG})|(?P<ap>{_A_POS}|{_A_POS_X})|(?P<an>{_A_NEG}|{_A_NEG_X}))\b", re.I)
# "three or more colours", "a more aggressive style", "the chance to win a game" are not comparisons
_NOT_COMPARISON = re.compile(r"\b(?:or|and)\s+(?:more|fewer|less)\b|\b(?:more|less)\s+\w+ive\b|\bchance (?:to win|of winning)(?:\s+a game)?|\bas above\b", re.I)
_NEGATION = re.compile(r"(?:\bnot|\bnever|\bno longer|\bhardly|\bbarely|n't)\s+(?:\w+\s+){0,2}$", re.I)
NO_CLEAR = re.compile(r"\bno (?:clear|real|obvious|reliable) (?:difference|winner|edge|advantage|gap)|\bno difference\b(?: in (?:who|which|what)\b[^,;.]*)?|\bnot clearly\b|\bno clear\b", re.I)
TAG = re.compile(r"<!--\s*F:\s*([^>]*?)\s*-->")
SENTENCE = re.compile(r"(?:.+?[.!?](?:\s*<!--[^>]*-->)*(?=\s|$))|.+$")
NUMBER = re.compile(r"(?<![\w.])([+\-\u2212\u2013]?)(\d+(?:\.\d+)?)(%?)")
SKIP_NUM = re.compile(r"\bbest[- ]of[- ]?[13]\b|\bbo[13]\b", re.I)
LEN_MIN, LEN_MAX = 400, 800

# the section headings of docs/report_writer/RULES.md (and the cross-set report): their words ("Best archetypes") are labels, not claims
FIXED_HEADINGS = {h.lower() for h in (
    "Headlines", "Best archetypes and how to build them", "Cards to pick up, cards to stop auto-including", "Deck shape", "Bo1 vs Bo3",
    "Bo1 and Bo3", "What the builder does differently", "Not enough evidence", "The builder across sets", "What depends on the set",
    "What holds across sets")}
_HEADING = re.compile(r"(#{1,6})\s+(.*)")

_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_SPELLED = {w: float(i) for i, w in enumerate(_ONES)}
_SPELLED.update({w: float(v) for w, v in _TENS.items()})
_SPELLED["dozen"] = 12.0
_UNIT = r"(?:percentage[- ]points?|percent|per ?cent|points?|games?|decks?|lands?|creatures?|copies|turns?|two-drops?|cards?|mulligans?|pools?|sets?)"
# a spelled-out number counts when a unit follows ("five points", "two-point", "twelve decks"); "one of the best" is not a number
SPELLED = re.compile(rf"\b(?P<w>(?:{'|'.join(_TENS)})(?:[- ](?:one|two|three|four|five|six|seven|eight|nine))?|{'|'.join(sorted(_ONES + ['dozen'], key=len, reverse=True))})(?=[- ]{_UNIT}\b)", re.I)
_DEFINITION = re.compile(r"\bone percentage points?\b|\b(?:means|is|equals?)\s+one points?\b", re.I)
_RANGE_WORDS = re.compile(r"\b(?:between|range|ranging|ranges|from)\b", re.I)
_RANGE_IN_TEXT = re.compile(r"range\s+([+\-\u2212\u2013]?\d+(?:\.\d+)?)\s+to\s+([+\-\u2212\u2013]?\d+(?:\.\d+)?)")
_COUNT_NOUN = re.compile(r"^\s*(?:lands?|creatures?|decks?|games?|turns?|copies|cards?|pools?|two-drops?|mulligans?|sets?)\b", re.I)
_COMPARES = re.compile(r"^\s*(?:points?\b|percentage\b|(?:more|less|higher|lower|better|worse|fewer|greater|bigger|larger|smaller)\b)", re.I)
_POINTS_AFTER = re.compile(r"\s*(?:percentage\s+)?points?\b", re.I)


def split_sentences(text: str) -> list[tuple[int, str]]:
    """[(line number, sentence including its tags)] of every line `export_html.markdown_to_html` renders: only a line that is nothing but
    comments is skipped (a comment next to text leaves the text to be checked), headings are included (the fixed section headings of the
    writer rules are labels and are skipped), list markers are dropped."""
    out = []
    for n, line in enumerate(text.splitlines(), start=1):
        s = line.strip()
        if not s or (s.startswith("<!--") and s.endswith("-->") and not re.sub(r"<!--.*?-->", "", s).strip()):
            continue
        h = _HEADING.match(s)
        if h:
            if re.sub(r"<!--.*?-->", "", h.group(2)).strip().lower() in FIXED_HEADINGS:
                continue
            s = h.group(2)
        else:
            s = re.sub(r"^(?:[-*+]\s+|\d+[.)]\s+)", "", s)
        for m in SENTENCE.finditer(s):
            if m.group(0).strip():
                out.append((n, m.group(0).strip()))
    return out


def tags_of(sentence: str) -> list[str]:
    return [i for m in TAG.finditer(sentence) for i in re.split(r"[,\s]+", m.group(1)) if i]


def plain(sentence: str) -> str:
    t = TAG.sub("", sentence)
    t = re.sub(r"<!--.*?-->", "", t)
    t = re.sub(r"[*_`]", "", t)
    return t


def numbers_of(text: str) -> list[tuple[float, int, bool, str]]:
    """[(value, decimals written, signed?, text)] of the digit numbers in a sentence."""
    return [(n["v"], n["dec"], n["signed"], n["raw"]) for n in _numbers(text) if not n["spelled"]]


def _numbers(text: str) -> list[dict]:
    """Digit and spelled-out numbers of a sentence, each with the text that follows it (its unit)."""
    t = SKIP_NUM.sub(" ", re.sub(r"(?<=\d),(?=\d{3}\b)", "", text))
    out = []
    for m in NUMBER.finditer(t):
        sign, digits = m.group(1), m.group(2)
        v = float(digits)
        if sign in ("-", "\u2212", "\u2013"):
            v = -v
        out.append(dict(v=v, dec=len(digits.split(".")[1]) if "." in digits else 0, signed=bool(sign), raw=m.group(0), pct=bool(m.group(3)),
                        after=t[m.end():m.end() + 30], spelled=False))
    for m in SPELLED.finditer(t):
        if _DEFINITION.search(t[max(0, m.start() - 12):m.end() + 30]):
            continue
        parts = re.split(r"[- ]", m.group("w").lower())
        v = _SPELLED[parts[0]] + (_SPELLED[parts[1]] if len(parts) > 1 else 0)
        out.append(dict(v=v, dec=0, signed=False, raw=m.group(0), pct=False, after=t[m.end():m.end() + 30], spelled=True))
    return out


def number_ok(v: float, dec: int, signed: bool, allowed: list[float]) -> bool:
    tol = 0.5 * 10 ** (-dec) + 1e-9
    for n in allowed:
        if signed:
            if abs(v - n) <= tol:
                return True
        elif abs(abs(v) - abs(n)) <= tol:
            return True
    return False


def fact_roles(f: dict) -> dict:
    """What the numbers of one fact are: {"effect": float | None, "ci": [lo, hi] | None, "others": [...], "strict": bool}.
    The effect is the fact's `value` (else the middle of the "range A to B" of its text, else - for a bare list of numbers - the first number
    when the next two bracket it); the range ends come from the text (or are that bracketing pair).  A fact with no effect (glance, shape,
    ...) is not strict: every number of it is allowed anywhere and its direction words cannot be checked."""
    nums = [float(x) for x in f.get("nums", [])]
    # the unsigned numbers of the fact's own label ("decks with 17 lands") are quotable too
    nums += [float(x) for x in re.findall(r"(?<![\w.+\-−–])\d+(?:\.\d+)?", f.get("text") or "")]
    ci = None
    m = _RANGE_IN_TEXT.search(f.get("text") or "")
    if m:
        ci = [float(x.replace("\u2212", "-").replace("\u2013", "-")) for x in m.groups()]
    eff = None
    if f.get("value") is not None:
        eff = float(f["value"])
    elif ci:
        eff = (ci[0] + ci[1]) / 2
    elif len(nums) >= 3 and nums[1] < nums[0] < nums[2]:
        eff, ci = nums[0], [nums[1], nums[2]]
    if eff is None:
        return dict(effect=None, ci=None, others=nums, strict=False, loose=True)

    def snap(x: float, tol: float) -> float:
        near = min(nums, key=lambda n: abs(n - x), default=None)
        return near if near is not None and abs(near - x) <= tol else x

    e = snap(eff, 0.011)
    cis = [snap(c, 0.06) for c in ci] if ci else []
    return dict(effect=e, ci=cis or None, others=[n for n in nums if n != e and n not in cis], strict=True,
                loose=f.get("value") is None)     # no `value`: the other numbers are not only shares (early / late effects ...), so they are allowed as points


def _sign_tokens(body: str) -> dict[str, list[int]]:
    toks: dict[str, list[int]] = {"v": [], "q": [], "a": []}
    for m in _TOKEN.finditer(body):
        kind = m.lastgroup
        sign = 1 if kind in ("vp", "qp", "ap") else -1
        if _NEGATION.search(body[max(0, m.start() - 30):m.start()]):
            sign = -sign
        toks[kind[0]].append(sign)
    return toks


def _sentence_signs(toks: dict[str, list[int]]) -> list[int]:
    """The signs the direction words of a sentence claim.  Verbs and comparatives multiply ("wins ... less" is negative, "loses ... more" is
    negative); adjectives (better, worst, overrated) count on their own; a group of words that disagree among themselves gives no sign."""
    signs = []
    if (toks["v"] or toks["q"]) and len(set(toks["v"])) <= 1 and len(set(toks["q"])) <= 1:
        signs.append((toks["v"] or [1])[0] * (toks["q"] or [1])[0])
    if toks["a"] and len(set(toks["a"])) == 1:
        signs.append(toks["a"][0])
    return signs


def check(text: str, facts: dict, check_length: bool = False) -> list[str]:
    claims, noclear = facts.get("facts", {}), facts.get("noclear", {})
    roles = {i: fact_roles(f) for i, f in claims.items()}
    problems: list[str] = []
    for n, s in split_sentences(text):
        ids = tags_of(s)
        body = plain(s).strip()
        shown = (body[:90] + "...") if len(body) > 90 else body
        unknown = [i for i in ids if i not in claims and i not in noclear]
        for i in unknown:
            problems.append(f"line {n}: unknown fact id {i}: {shown}")
        good = [i for i in ids if i in claims or i in noclear]
        nums = _numbers(body)
        dtext = _NOT_COMPARISON.sub(" ", NO_CLEAR.sub(" ", SKIP_NUM.sub(" ", body)))
        dirs = DIRECTION.findall(dtext)
        if not good and not nums and re.search(r"\btrap (?:rares?|cards?)\b", dtext, re.I):
            dirs = [d for d in dirs if not re.fullmatch(r"loses?|lost", d, re.I)]      # the definition "trap rares: played a lot, yet they lose"
        if (nums or dirs) and not good:
            if not unknown:
                what = "number" if nums else f"direction word '{dirs[0]}'"
                problems.append(f"line {n}: untagged claim ({what}): {shown}")
            continue
        rl = [roles[i] for i in good if i in claims]
        ranged = bool(_RANGE_WORDS.search(body))
        for nm in nums:
            why = _number_verdict(nm, rl, ranged, bool(dirs))
            if why:
                problems.append(f"line {n}: number {nm['raw']} {why} (tagged facts {','.join(good)}): {shown}")
        nc = [i for i in good if i in noclear]
        if dirs and nc and not (len(nc) < len(good) and NO_CLEAR.search(body)):
            # only a sentence that also states a clear result (a claim fact) and says "no clear ..." for the other may use direction words
            problems.append(f"line {n}: direction word '{dirs[0]}' on a no-clear-difference fact ({','.join(nc)}): {shown}")
            continue
        toks = _sign_tokens(dtext)
        signs = _sentence_signs(toks)
        signed_facts = [(i, roles[i]["effect"]) for i in good if i in claims and roles[i]["strict"] and roles[i]["effect"]]
        fact_signs = {1 if eff > 0 else -1 for _, eff in signed_facts}
        if len(fact_signs) == 1 and signs and any(sg not in fact_signs for sg in signs):
            i = signed_facts[0][0]
            problems.append(f"line {n}: the wording says {'better' if signs[0] > 0 else 'worse'} but {i} is {'positive' if 1 in fact_signs else 'negative'}: {shown}")
        elif len(fact_signs) == 2 and (toks["v"] or toks["q"] or toks["a"]) and len({x for v in toks.values() for x in v}) == 1:
            problems.append(f"line {n}: the tagged facts point both ways but the wording says only {'better' if signs and signs[0] > 0 else 'one way'}: {shown}")
    if check_length:
        words = len(re.findall(r"\b[\w'-]+\b", "\n".join(plain(l) for l in text.splitlines() if not l.lstrip().startswith("#"))))
        if not LEN_MIN <= words <= LEN_MAX:
            problems.append(f"length: {words} words, expected {LEN_MIN}-{LEN_MAX}")
    return problems


def _number_verdict(nm: dict, rl: list[dict], ranged: bool, direction: bool) -> str:
    """"" when the number is one the tagged facts allow here, else why not."""
    v, dec, signed = nm["v"], nm["dec"], nm["signed"]
    after = nm["after"]
    why = "does not match the tagged facts"
    for r in rl:
        if not r["strict"]:
            if number_ok(v, dec, signed, r["others"]):
                return ""
            continue
        is_other = number_ok(v, dec, signed, r["others"])
        if is_other and nm["pct"] and not _COMPARES.match(after):
            return ""                                        # "32% of decks", "59.6% against 55.4%"
        if number_ok(v, dec, signed, [r["effect"]]):
            return ""
        if r["ci"] and number_ok(v, dec, signed, r["ci"]):
            if ranged:
                return ""
            why = "is an end of the range, but the sentence does not speak of a range"
        elif is_other:
            if r["loose"] or _COUNT_NOUN.match(after) or (not direction and not _POINTS_AFTER.match(after) and not _COMPARES.match(after)):
                return ""
            if _POINTS_AFTER.match(after):
                why = "is a share or win rate of the fact, not its effect, but is written as points"
            else:
                why = "is a share or other figure of the fact, not its effect, used in a comparison"
    return why
