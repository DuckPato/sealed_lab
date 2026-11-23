"""Card attributes A_c for the card-attribute prior: beta_c = A_c . gamma + u_c.

`attribute_matrix(cards)` turns the card table into a numeric matrix (one row per card_id): mana value, rarity, number of
colours, type flags, power / toughness, keyword flags and oracle-text tags. Everything is derived from what the card
table knows: Scryfall fields when present (power, toughness, keywords) and the oracle text otherwise; cards with only
inferred metadata get the entries that can still be computed (mana value, colour count) and zeros elsewhere.
"""
from __future__ import annotations

import re

import numpy as np
import polars as pl

KEYWORDS = ["flying", "trample", "deathtouch", "lifelink", "haste", "vigilance", "first strike", "double strike", "flash",
            "reach", "menace", "ward", "defender", "hexproof", "indestructible"]
# keywords merged into one flag
KW_FLAGS = [("kw_flying", ["flying"]), ("kw_trample", ["trample"]), ("kw_deathtouch", ["deathtouch"]),
            ("kw_lifelink", ["lifelink"]), ("kw_haste", ["haste"]), ("kw_vigilance", ["vigilance"]),
            ("kw_first_double_strike", ["first strike", "double strike"]), ("kw_flash", ["flash"]),
            ("kw_reach", ["reach"]), ("kw_menace", ["menace"]), ("kw_ward", ["ward"]), ("kw_defender", ["defender"]),
            ("kw_hexproof_indestructible", ["hexproof", "indestructible"])]
TYPE_FLAGS = [("t_creature", "creature"), ("t_instant", "instant"), ("t_sorcery", "sorcery"), ("t_artifact", "artifact"),
              ("t_enchantment", "enchantment"), ("t_planeswalker", "planeswalker"), ("t_land", "land"),
              ("t_legendary", "legendary")]
TAGS = ["tag_removal", "tag_draw", "tag_counterspell", "tag_pump_aura_equipment", "tag_token", "tag_etb", "tag_fixing"]
NAMES = (["mv", "rarity_uncommon", "rarity_rare", "rarity_mythic", "n_colors"] + [n for n, _ in TYPE_FLAGS]
         + ["power", "toughness", "pt_variable"] + [n for n, _ in KW_FLAGS] + TAGS)

LABELS = {
    "mv": "mana value (per point)", "rarity_uncommon": "uncommon (vs common)", "rarity_rare": "rare (vs common)",
    "rarity_mythic": "mythic (vs common)", "n_colors": "number of colours", "t_creature": "creature",
    "t_instant": "instant", "t_sorcery": "sorcery", "t_artifact": "artifact", "t_enchantment": "enchantment",
    "t_planeswalker": "planeswalker", "t_land": "non-basic land", "t_legendary": "legendary",
    "power": "power (per point)", "toughness": "toughness (per point)", "pt_variable": "variable power/toughness (*)",
    "kw_flying": "flying", "kw_trample": "trample", "kw_deathtouch": "deathtouch", "kw_lifelink": "lifelink",
    "kw_haste": "haste", "kw_vigilance": "vigilance", "kw_first_double_strike": "first / double strike",
    "kw_flash": "flash", "kw_reach": "reach", "kw_menace": "menace", "kw_ward": "ward", "kw_defender": "defender",
    "kw_hexproof_indestructible": "hexproof / indestructible", "tag_removal": "removal (text)",
    "tag_draw": "card draw", "tag_counterspell": "counterspell", "tag_pump_aura_equipment": "pump / aura / equipment",
    "tag_token": "token maker", "tag_etb": "enters-the-battlefield trigger", "tag_fixing": "mana fixing / ramp",
}

# a deck-level column is kept only if this many non-basic cards carry the attribute (otherwise gamma is just that
# card's own effect, which u already models)
MIN_CARDS = 4

_REMOVAL = [
    re.compile(r"\b(destroy|exile)s?\s+(up to \w+ )?target\s+([\w\-/ ,]+? )?(creature|permanent|planeswalker)(?! (you|that player) control)(?! you own)"),
    re.compile(r"\bdeals?\s+(\d+|x|that much)\s+damage\s+(divided as you choose\s+)?to\s+(any target|target\s+([\w\-/ ]+ )?(creature|planeswalker)|up to (one|two|three) target)"),
    re.compile(r"\btarget\s+[\w\-/ ]*creature[\w\-/ ]*\s+gets?\s+-(\d+|x)/-(\d+|x)"),
    re.compile(r"\bfights?\b"),
]
_DRAW = re.compile(r"\bdraws?\s+(a|an additional|two|three|four|x|\d+|that many)\s+cards?\b")
_COUNTER = re.compile(r"\bcounter target\b")
_PUMP = re.compile(r"(\bgets?\s+\+\d+/\+\d+|\+1/\+1 counter)")
_TOKEN = re.compile(r"\bcreates?\b[^.\n]*\btokens?\b")
_FIX = re.compile(r"(\badd\b[^.\n]{0,40}(\{[wubrgc]\}|\bmana\b)|search your library for [^.\n]*\bland\b)")
_REMINDER = re.compile(r"\([^)]*\)")


def keywords_from_text(oracle: str) -> set[str]:
    """Intrinsic keyword abilities read off the oracle text: lines made only of keywords (reminder text removed)."""
    found: set[str] = set()
    for line in (oracle or "").lower().split("\n"):
        line = _REMINDER.sub("", line).strip()
        if not line:
            continue
        parts = [p.strip() for p in re.split(r",|;", line) if p.strip()]
        hit = []
        for p in parts:
            k = next((k for k in KEYWORDS if p == k or p.startswith(k + " ")), None)
            if k is None:
                hit = []
                break
            hit.append(k)
        found.update(hit)
    return found


def _kw_set(keywords: str | None, oracle: str) -> set[str]:
    if keywords is None:                      # cache from before the keywords field existed: parse the text
        return keywords_from_text(oracle)
    return {k for k in keywords.split(",") if k}


def _tags(text: str, name: str, types: str) -> list[float]:
    t = (text or "").lower()
    front = name.split(" // ")[0].lower()
    short = front.split(",")[0].strip()
    etb = re.compile(r"\bwhen(ever)?\s+(this [a-z]+|" + re.escape(front) + "|" + re.escape(short) + r")\s+enters\b")
    removal = any(r.search(t) for r in _REMOVAL)
    sub = types.lower().split(" - ")[-1].split() if " - " in types else []
    pump = "aura" in sub or "equipment" in sub or ("creature" not in types.lower().split(" - ")[0].split()
                                                   and bool(_PUMP.search(t)))
    return [float(removal), float(bool(_DRAW.search(t))), float(bool(_COUNTER.search(t))), float(pump),
            float(bool(_TOKEN.search(t))), float(bool(etb.search(t))), float(bool(_FIX.search(t)))]


def attribute_matrix(cards: pl.DataFrame) -> tuple[np.ndarray, list[str]]:
    """(n_cards, len(NAMES)) float matrix for the cards sorted by card_id, and the attribute names."""
    c = cards.sort("card_id")
    n = c.height
    A = np.zeros((n, len(NAMES)))
    ix = {nm: i for i, nm in enumerate(NAMES)}
    cols = set(c.columns)
    names = c["name"].to_list()
    types = [t or "" for t in c["types"].to_list()]
    oracle = [t or "" for t in c["oracle_text"].to_list()]
    rarity = c["rarity"].to_list()
    is_land = c["is_land"].to_list()
    is_basic = c["is_basic"].to_list()
    colors = [s or "" for s in c["colors"].to_list()]
    mv = c["mana_value"].to_list()
    power = c["power"].to_list() if "power" in cols else [None] * n
    tough = c["toughness"].to_list() if "toughness" in cols else [None] * n
    pvar = c["pt_variable"].to_list() if "pt_variable" in cols else [False] * n
    kws = c["keywords"].to_list() if "keywords" in cols else [None] * n
    for i in range(n):
        if is_basic[i]:
            continue
        head = types[i].split(" - ")[0].lower().split()
        A[i, ix["mv"]] = 0.0 if is_land[i] else float(mv[i] or 0.0)
        A[i, ix["n_colors"]] = 0.0 if is_land[i] else float(len(colors[i]))
        r = rarity[i]
        for k in ("uncommon", "rare", "mythic"):
            A[i, ix["rarity_" + k]] = float(r == k)
        for nm, word in TYPE_FLAGS:
            A[i, ix[nm]] = float(word in head)
        if "creature" in head:
            A[i, ix["power"]] = float(power[i] or 0.0)
            A[i, ix["toughness"]] = float(tough[i] or 0.0)
            A[i, ix["pt_variable"]] = float(bool(pvar[i]))
        ks = _kw_set(kws[i], oracle[i])
        for nm, members in KW_FLAGS:
            A[i, ix[nm]] = float(any(m in ks for m in members))
        t0 = len(NAMES) - len(TAGS)
        A[i, t0:t0 + len(TAGS)] = _tags(oracle[i], names[i], types[i])
    return A, list(NAMES)


def select_columns(A: np.ndarray, nb_ids: np.ndarray, min_cards: int = MIN_CARDS) -> np.ndarray:
    """Indices of the attributes worth a deck-level column: carried by >= `min_cards` non-basic cards and not by all."""
    sub = A[nb_ids]
    cnt = (sub != 0).sum(0)
    var = sub.std(0)
    return np.nonzero((cnt >= min_cards) & (var > 1e-9))[0]
