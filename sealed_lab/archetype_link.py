"""Where does a deck sit in the Analysis tab's archetypes?  (link from the Pool tab to the deck's color pair and style position)

The Analysis report (report_arch) places every REAL deck of a color pair on the pair's style axis (first principal component of
[idf-weighted card presence | standardised style features]) and cuts the pair's decks in thirds.  The fitted axis is not stored in the report, so
this module re-fits it the same way (`Embedder(Xb, S, rank=1)` on all the pair's decks, same orientation rule, same quantiles) and keeps the
minimal projector in a SEPARATE cache file next to the report:

    config.cache_dir(set)/archetype_proj_<fmt>.json    keyed by the model's built_at and PROJ_VERSION

The score of any deck is then  score = presence(cards) . w + style_features . c + offset  (a linear function; sign already applied so that the
low end is end `a` and the high end is end `b` of the report's spectrum), and its third is  0 if score <= q1, 2 if score >= q2, else 1  -
exactly the rule that split the real decks.  Nothing here changes the report JSON, REPORT_VERSION or the written reports' stamps.

Observed description only: the builder does not use archetypes and never chooses a style.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from sealed_lab import config
from sealed_lab import report as R
from sealed_lab import report_arch as A

log = logging.getLogger(__name__)

PROJ_VERSION = 1
_LOCK = threading.RLock()
_MEM: dict[tuple[str, str], dict] = {}                 # (set, fmt) -> projector file content
_CV: dict[str, tuple[str, A.CardVectors]] = {}         # set -> (built_at, card vectors)


def proj_path(set_code: str, fmt: str) -> Path:
    return config.cache_dir(set_code) / f"archetype_proj_{fmt}.json"


# ---------------------------------------------------------------------------------------------
# the projector of one color pair
# ---------------------------------------------------------------------------------------------

def fit_pair_projector(Xb: sp.csr_matrix, card_ids: np.ndarray, S: np.ndarray, avg_mv: np.ndarray, n_cards: int) -> dict:
    """Projector of one pair from its decks (binary pair-card matrix `Xb` over the cards `card_ids`, style matrix `S`, average mana value of each
    deck for the orientation).  Reproduces `report_arch.build_archetypes` (score = rank-1 embedding, flipped so that a higher score = a higher
    average mana value) and its thirds."""
    emb = A.Embedder(Xb, S, rank=1)
    nb = len(emb.cols)
    v = emb.V[:, 0]
    w = np.zeros(n_cards)
    if nb:
        w[card_ids[emb.cols]] = emb.idf * emb.sc_b * v[:nb]
    c = v[nb:] * emb.sc_s / emb.s_sd
    off = -float(emb.mu @ v) - float(emb.s_mu @ c)
    sc = emb.transform(Xb, S)[:, 0]
    corr = np.corrcoef(sc, avg_mv)[0, 1] if sc.std() > 0 and avg_mv.std() > 0 else 1.0
    sign = -1.0 if corr < 0 else 1.0
    w, c, off, sc = sign * w, sign * c, sign * off, sign * sc
    q1, q2 = np.quantile(sc, [1 / 3, 2 / 3])
    return dict(n_cards=int(n_cards), w=w.tolist(), c=c.tolist(), offset=float(off), q1=float(q1), q2=float(q2), n_decks=int(Xb.shape[0]),
                lo_n=int((sc <= q1).sum()), hi_n=int((sc >= q2).sum()))


def project_score(pp: dict, X: sp.csr_matrix, S: np.ndarray) -> np.ndarray:
    """Style-axis score of decks (rows of the copies matrix `X` over all cards, style matrix `S`)."""
    w = np.asarray(pp["w"])
    n = min(len(w), X.shape[1])
    pres = (X[:, :n] > 0).astype(float)
    return np.asarray(pres @ w[:n]).ravel() + S @ np.asarray(pp["c"]) + pp["offset"]


def thirds_of(pp: dict, score: np.ndarray) -> np.ndarray:
    """0 = low third (end a), 1 = middle, 2 = high third (end b): the rule that cut the real decks."""
    e1, e2 = 1e-9 * (1 + abs(pp["q1"])), 1e-9 * (1 + abs(pp["q2"]))      # the deck sitting exactly on a quantile must not flip on rounding noise (1e-16)
    return np.where(score <= pp["q1"] + e1, 0, np.where(score >= pp["q2"] - e2, 2, 1))


# ---------------------------------------------------------------------------------------------
# fitting for a set + format, cache
# ---------------------------------------------------------------------------------------------

def _card_vectors(model) -> A.CardVectors:
    built = str(model.meta.get("built_at") or "")
    hit = _CV.get(model.set_code)
    if hit is not None and hit[0] == built:
        return hit[1]
    cv = A.CardVectors.from_cards(model.cards.sort("card_id"))
    _CV[model.set_code] = (built, cv)
    return cv


def fit_projectors(set_code: str, fmt: str, model, rep: dict) -> dict:
    """Projectors of every pair of the report `rep` whose spectrum is shown.  A pair whose re-fitted thirds do not have the sizes the report
    stored (the ends' n_decks) is left out: it would not be the same axis."""
    wanted = [a for a in rep.get("archetypes", {}).get("pairs", []) if a.get("spectrum", {}).get("shown")]
    out: dict = dict(version=PROJ_VERSION, set_code=set_code.upper(), fmt=fmt, built_at=model.meta.get("built_at"), pairs={})
    if not wanted:
        return out
    g = R.load_games(set_code, fmt)
    cards = model.cards.sort("card_id")
    cv = _card_vectors(model)
    decks, X = A.deck_table(set_code, g, cards)
    sh = A.deck_shape(X, cv, decks["mmask"].to_numpy())
    S = A.style_matrix(sh)
    grp = decks["grp"].to_numpy()
    for a in wanted:
        p = a["key"]
        idx = np.nonzero(grp == p)[0]
        if len(idx) == 0:
            continue
        Xb, ids = A._pair_binary(X[idx], cv, A.colour_mask(p))
        pp = fit_pair_projector(Xb, ids, S[idx], sh["avg_mv"][idx], X.shape[1])
        sp_ = a["spectrum"]
        if pp["lo_n"] != sp_["a"]["n_decks"] or pp["hi_n"] != sp_["b"]["n_decks"]:
            log.info("archetype projector of %s %s %s does not match the report (%s/%s vs %s/%s): left out", set_code, fmt, p,
                     pp["lo_n"], pp["hi_n"], sp_["a"]["n_decks"], sp_["b"]["n_decks"])
            continue
        out["pairs"][p] = pp
    return out


def _read(path: Path, built_at) -> dict | None:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    ok = isinstance(d, dict) and d.get("version") == PROJ_VERSION and d.get("built_at") == built_at and isinstance(d.get("pairs"), dict)
    return d if ok else None


def get_projectors(set_code: str, fmt: str, model, rep: dict) -> dict:
    """The stored projectors (memory, then the cache file), fitted and stored on first request."""
    set_code = set_code.upper()
    built = model.meta.get("built_at")
    with _LOCK:
        hit = _MEM.get((set_code, fmt))
        if hit is not None and hit.get("built_at") == built:
            return hit
        d = _read(proj_path(set_code, fmt), built)
        if d is None:
            d = fit_projectors(set_code, fmt, model, rep)
            try:
                p = proj_path(set_code, fmt)
                p.parent.mkdir(parents=True, exist_ok=True)
                tmp = p.with_name(p.name + ".tmp")
                tmp.write_text(json.dumps(d), encoding="utf-8")
                os.replace(tmp, p)
            except OSError:
                log.info("archetype projector of %s %s could not be written", set_code, fmt, exc_info=True)
        _MEM[(set_code, fmt)] = d
        return d


def forget(set_code: str) -> None:
    with _LOCK:
        for k in [k for k in _MEM if k[0] == set_code.upper()]:
            _MEM.pop(k, None)


# ---------------------------------------------------------------------------------------------
# a deck
# ---------------------------------------------------------------------------------------------

def deck_row(deck: dict[int, int], n_cards: int) -> sp.csr_matrix:
    ids = [int(k) for k, n in deck.items() if n > 0 and 0 <= int(k) < n_cards]
    return sp.csr_matrix((np.array([float(deck[i]) for i in ids]), (np.zeros(len(ids), dtype=np.int64), np.array(ids, dtype=np.int64))),
                         shape=(1, n_cards))


def deck_style_score(pp: dict, deck: dict[int, int], cv: A.CardVectors, main: str) -> tuple[float, int]:
    """(score, third) of one deck (card_id -> copies, basic lands included) on the pair's axis; `main` = the deck's main colors."""
    n = len(cv.mv)
    X = deck_row(deck, n)
    sh = A.deck_shape(X, cv, np.array([A.colour_mask(main)]))
    sc = project_score(pp, X, A.style_matrix(sh))
    return float(sc[0]), int(thirds_of(pp, sc)[0])


BADGE_WORD = {"strong": "Strong", "likely": "Likely", "weak": "Weak"}


def _effect_words(eff: float | None, badge: str | None) -> str:
    if eff is None or badge in (None, "weak"):
        return "no clear difference from an average deck"
    return f"{R.signed(eff)} points ({BADGE_WORD.get(badge, badge)})"


def describe(rep: dict, projectors: dict, deck: dict[int, int], cv: A.CardVectors, main: str) -> dict:
    """The archetype line of a deck (see `api.deck_archetype`).  `rep` = the stored report, `main` = the deck's main colors."""
    key = R.pair_group(main)
    if key is None or key not in R_PAIRS:
        what = {"mono": "one main color", "tri": "three or more main colors"}.get(key or "", "no clear main colors")
        return dict(available=False, reason=f"The deck has {what}: the Analysis tab describes two-color archetypes.")
    a = next((x for x in rep.get("archetypes", {}).get("pairs", []) if x["key"] == key), None)
    if a is None:
        return dict(available=False, reason=f"{R.group_label(key)} has too few decks in this format for an archetype entry.")
    pp = projectors.get("pairs", {}).get(key)
    spec = a.get("spectrum") or {}
    third, score = None, None
    if spec.get("shown") and pp is not None:
        score, third = deck_style_score(pp, deck, cv, main)
    return compose(a, main, third, score)


def compose(a: dict, main: str, third: int | None, score: float | None = None) -> dict:
    """The line for the archetype-pair entry `a` of the report and the deck's third on its axis (None = no spectrum position)."""
    out: dict = dict(available=True, key=a["key"], code=a["code"], label=a["label"], n_decks=a["n_decks"], effect_pp=a["effect_pp"], badge=a["badge"],
                     range=a["range"], main=main, spectrum=dict(shown=False))
    parts = [f"Color pair: {a['label']} ({a['code']}). Decks in this pair: {_effect_words(a['effect_pp'], a['badge'])}."]
    spec = a.get("spectrum") or {}
    if spec.get("shown") and third is not None:
        ea, eb = spec["a"], spec["b"]
        if third == 1:
            out["spectrum"] = dict(shown=True, end="mid", third=1, score=score, a_label=ea["label"], b_label=eb["label"])
            parts.append(f"Style: in the middle between the '{ea['label']}' and the '{eb['label']}' ends.")
        else:
            e, nm = (ea, "a") if third == 0 else (eb, "b")
            claim = e.get("badge") in ("likely", "strong")
            out["spectrum"] = dict(shown=True, end=nm, third=third, score=score, a_label=ea["label"], b_label=eb["label"], label=e["label"],
                                   effect_pp=e["effect_pp"], badge=e["badge"], range=e["range"], states_effect=claim)
            tail = (f"that end: {R.signed(e['effect_pp'])} points, {BADGE_WORD[e['badge']]}" if claim
                    else "that end shows no clear difference from an average deck")
            parts.append(f"Style: closer to the '{e['label']}' end ({tail}).")
    parts.append("Observed in 17lands data; the builder does not use archetypes.")
    out["text"] = " ".join(parts)
    return out


R_PAIRS = set(A.design.PAIRS)
