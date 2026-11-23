"""Deterministic checker of a written report against its facts file (core of scripts/check_report.py; used by `cli report-install`)."""
from __future__ import annotations

import re

DIRECTION = re.compile(r"\b(more|less|fewer|better|worse|wins|loses|beats|higher|lower|best|worst|overrated|underrated|over-rated|under-rated)\b", re.I)
NO_CLEAR = re.compile(r"\bno (?:clear|real|obvious|reliable) (?:difference|winner|edge|advantage|gap)|\bno difference\b|\bnot clearly\b|\bno clear\b", re.I)
TAG = re.compile(r"<!--\s*F:\s*([^>]*?)\s*-->")
SENTENCE = re.compile(r"(?:.+?[.!?](?:\s*<!--[^>]*-->)*(?=\s|$))|.+$")
NUMBER = re.compile(r"(?<![\w.])([+\-\u2212\u2013]?)(\d+(?:\.\d+)?)(%?)")
SKIP_NUM = re.compile(r"\bbest[- ]of[- ]?[13]\b|\bbo[13]\b", re.I)
LEN_MIN, LEN_MAX = 400, 800


def split_sentences(text: str) -> list[tuple[int, str]]:
    """[(line number, sentence including its tags)] of the non-heading lines."""
    out = []
    for n, line in enumerate(text.splitlines(), start=1):
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("<!--") and s.endswith("-->") and not TAG.search(s):
            continue
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
    """[(value, decimals written, signed?, text)] of the numbers in a sentence."""
    t = SKIP_NUM.sub(" ", re.sub(r"(?<=\d),(?=\d{3}\b)", "", text))
    out = []
    for m in NUMBER.finditer(t):
        sign, digits = m.group(1), m.group(2)
        v = float(digits)
        if sign in ("-", "\u2212", "\u2013"):
            v = -v
        out.append((v, len(digits.split(".")[1]) if "." in digits else 0, bool(sign), m.group(0)))
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


def check(text: str, facts: dict, check_length: bool = False) -> list[str]:
    claims, noclear = facts.get("facts", {}), facts.get("noclear", {})
    problems: list[str] = []
    for n, s in split_sentences(text):
        ids = tags_of(s)
        body = plain(s).strip()
        shown = (body[:90] + "...") if len(body) > 90 else body
        unknown = [i for i in ids if i not in claims and i not in noclear]
        for i in unknown:
            problems.append(f"line {n}: unknown fact id {i}: {shown}")
        good = [i for i in ids if i in claims or i in noclear]
        nums = numbers_of(body)
        dirs = DIRECTION.findall(body)
        if (nums or dirs) and not good:
            if not unknown:
                what = "number" if nums else f"direction word '{dirs[0]}'"
                problems.append(f"line {n}: untagged claim ({what}): {shown}")
            continue
        allowed = [x for i in good if i in claims for x in claims[i].get("nums", [])]
        for v, dec, signed, raw in nums:
            if not number_ok(v, dec, signed, allowed):
                problems.append(f"line {n}: number {raw} does not match the tagged facts {','.join(good)}: {shown}")
        if dirs and any(i in noclear for i in good) and not NO_CLEAR.search(body):
            problems.append(f"line {n}: direction word '{dirs[0]}' on a no-clear-difference fact ({','.join(i for i in good if i in noclear)}): {shown}")
    if check_length:
        words = len(re.findall(r"\b[\w'-]+\b", "\n".join(plain(l) for l in text.splitlines() if not l.lstrip().startswith("#"))))
        if not LEN_MIN <= words <= LEN_MAX:
            problems.append(f"length: {words} words, expected {LEN_MIN}-{LEN_MAX}")
    return problems
