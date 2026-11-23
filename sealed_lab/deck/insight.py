"""What the model knows about a deck beyond its win rate: card-pair synergies and the deck's archetype.

Both are read from the optional layers of the model artifact (`SetModel.pairs`, `SetModel.archetypes`); without them the
functions say so instead of inventing numbers.  Only effects whose sign is supported by the data (P >= `SYN_P_MIN`) and that
are at least `SYN_MIN_PP` large are reported as synergies.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import expit

from sealed_lab.model import synergy as syn

MIN_RATE_GAMES = 200        # an archetype's win rate in a format needs this many games, else the draft rate is shown instead


@dataclass
class SynRow:
    a: int                     # card id (the partner already in the deck)
    b: int                     # card id (the other card: in the deck, or the sideboard card that would complete the pair)
    pp: float                  # effect on the deck's game win rate in pp (present: of having the pair; candidate: of adding b)
    coef_pp: float             # per-unit coefficient in pp near a 55% deck
    lo_pp: float               # 95% interval of coef_pp
    hi_pp: float
    p_pos: float               # P(effect > 0)
    games: float               # games in which both cards were in the same deck (exposure)


@dataclass
class SynergyView:
    status: str                # "ok" | "off"
    message: str = ""
    present: list[SynRow] = field(default_factory=list)       # reliable pairs inside the deck (positive and negative)
    candidates: list[SynRow] = field(default_factory=list)    # reliable positive pairs completed by a sideboard card
    by_card: dict[int, list[SynRow]] = field(default_factory=dict)
    p_min: float = syn.SYN_P_MIN
    min_pp: float = syn.SYN_MIN_PP
    n_reliable: int = 0


def _reliable(model, fmt: str):
    key = ("_syn_rel", fmt)
    hit = model.__dict__.get("_syn_cache", {}).get(key)
    if hit is not None:
        return hit
    ab = model.pairs["ab"]
    coef = np.asarray(model._draws[fmt]["pair"][0], dtype=float)
    se = np.asarray(model.pairs["se"][fmt], dtype=float)
    p = syn.p_positive(coef, se)
    rel = syn.reliable_mask(coef, se)
    out = (ab, coef, se, p, rel, np.asarray(model.pairs["games"], dtype=float))
    model.__dict__.setdefault("_syn_cache", {})[key] = out
    return out


def synergy_view(model, deck: dict[int, int], pool: dict[int, int] | None, fmt: str) -> SynergyView:
    """Reliable synergies inside `deck` and the sideboard cards (rest of `pool`) that would add one."""
    if fmt not in model.formats or not model.has_pairs(fmt):
        why = (model.meta.get("synergy") or {}).get("reason") or "the synergy layer is not in this model"
        return SynergyView(status="off", message="Pair synergies are not used: " + str(why))
    ab, coef, se, p, rel, games = _reliable(model, fmt)
    n = model.card_arrays.n
    cf = np.zeros(n)
    for c, v in deck.items():
        if 0 <= c < n and v > 0:
            cf[c] = v
    side = np.zeros(n)
    for c, v in (pool or {}).items():
        if 0 <= c < n:
            side[c] = max(0, v - deck.get(c, 0))
    L = float(model.logit(deck, fmt))
    out = SynergyView(status="ok", n_reliable=int(rel.sum()))
    z = syn.PP_PER_LOGIT
    ids = np.nonzero(rel)[0]
    for j in ids:
        a, b = int(ab[j, 0]), int(ab[j, 1])
        row = dict(coef_pp=float(z * coef[j]), lo_pp=float(z * (coef[j] - 1.96 * se[j])), hi_pp=float(z * (coef[j] + 1.96 * se[j])),
                   p_pos=float(p[j]), games=float(games[j]))
        m_now = min(cf[a], cf[b])
        if m_now > 0:
            pp = 100.0 * float(expit(L) - expit(L - coef[j] * m_now))
            r = SynRow(a=a, b=b, pp=pp, **row)
            out.present.append(r)
            out.by_card.setdefault(a, []).append(r)
            out.by_card.setdefault(b, []).append(r)
            continue
        if coef[j] <= 0:
            continue
        for s_, p_ in ((a, b), (b, a)):               # s_ = sideboard card, p_ = partner in the deck
            if side[s_] > 0 and cf[p_] > 0:
                dm = min(cf[s_] + 1, cf[p_]) - min(cf[s_], cf[p_])
                if dm > 0:
                    pp = 100.0 * float(expit(L + coef[j] * dm) - expit(L))
                    r = SynRow(a=p_, b=s_, pp=pp, **row)
                    out.candidates.append(r)
                    out.by_card.setdefault(s_, []).append(r)
    out.present.sort(key=lambda r: -abs(r.pp))
    out.candidates.sort(key=lambda r: -r.pp)
    for v in out.by_card.values():
        v.sort(key=lambda r: -abs(r.pp))
    return out


# ---------------------------------------------------------------------------------------------
# archetype
# ---------------------------------------------------------------------------------------------

@dataclass
class ArchetypeMatch:
    status: str                              # "ok" | "off"
    message: str = ""
    index: int = -1
    label: str = ""
    group: str = ""
    similarity: float = 0.0                  # cosine similarity of the deck to the archetype centroid
    n_decks: int = 0
    rates: dict[str, dict] = field(default_factory=dict)      # per format: games, wr, shrunk, lo, hi
    rate: dict | None = None                 # the rate shown for the requested format (None: no data)
    rate_fmt: str = ""                       # format that `rate` comes from
    rate_note: str = ""
    missing: list[dict] = field(default_factory=list)         # signature cards not in the deck: name, card_id, rate
    runner_up: tuple[str, float] | None = None


def archetype_of(model, deck: dict[int, int], fmt: str) -> ArchetypeMatch:
    """Nearest archetype of `deck`, similarity, its observed (shrunk) win rate and the signature cards the deck lacks."""
    arch = getattr(model, "archetypes", None)
    if arch is None or arch.n == 0:
        return ArchetypeMatch(status="off", message="No archetypes in this model (recalculate the set).")
    i, sim, S = arch.assign({int(k): int(v) for k, v in deck.items() if v > 0}, model._spell)
    info = arch.info[i]
    rates = info.get("rates", {})
    m = ArchetypeMatch(status="ok", index=i, label=info["label"], group=info["group"], similarity=sim, n_decks=int(info["n_decks"]),
                       rates=rates)
    r = rates.get(fmt)
    if r is not None and r["games"] >= MIN_RATE_GAMES:
        m.rate, m.rate_fmt = r, fmt
    elif rates.get("draft") is not None and fmt != "draft":
        m.rate, m.rate_fmt = rates["draft"], "draft"
        got = 0 if r is None else r["games"]
        m.rate_note = (f"only {got} {fmt} games for this archetype (need {MIN_RATE_GAMES}): draft win rate shown, "
                       "which is not comparable with Sealed win rates")
    elif r is not None:
        m.rate, m.rate_fmt = r, fmt
        m.rate_note = f"only {r['games']} games"
    for s in info.get("signature", []):
        if deck.get(int(s["card_id"]), 0) <= 0 and s["rate"] >= 0.25:
            m.missing.append(dict(card_id=int(s["card_id"]), name=s["name"], rate=float(s["rate"]), ref=float(s["ref"])))
    order = np.argsort(-S)
    if len(order) > 1:
        j = int(order[1])
        m.runner_up = (arch.info[j]["label"], float(S[j]))
    return m
