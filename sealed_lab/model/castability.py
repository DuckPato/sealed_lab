"""Castability: can a deck's mana cast a given card, on time?

Frank Karsten's method, by simulation.  For a card with coloured pips the question is whether the deck has found the
coloured sources it needs by the turn the card is meant to be cast, *given that the land drops were hit*.  His 90%
"consistency" target is a threshold on exactly this probability; we use the probability itself:

    p(card) = 1/2 P(coloured requirement met by turn T)  +  1/2 P(met by turn T + 2),     T = max(mana value, 1)

both conditional on having seen at least `mana value` lands by that turn (enough lands, so p measures the colours and not the
land count - screw is the mana simulator's business).  "Met" is Hall's condition on the sources seen: for every set S of
colours, the number of sources able to produce a colour of S is at least the number of pips that only S can pay (a dual land is
one source, never two at once; hybrid pips can be paid by any of their colours; Phyrexian and twobrid pips are free; X = 0).
On curve counts half and "two turns late" the other half: a card whose colours arrive two turns late is still a late-game
play, a card whose colours never arrive is dead; the data price the exact trade-off through the model's dead-card coefficient.

The hands and draws come from the calibrated deal of the v2 mana simulator (`manasim.deal_v2`): the Arena hand smoother of the
Bo1 queues, the land-balanced Sealed draws, the real keep rule and London mulligans - so the probability is regime-aware.
`mode="plain"` uses a plain shuffle without mulligans, for which the answer is an exact hypergeometric (`exact_two_colour`),
the cross-check in the tests.

Which cards count as sources, and when they work, is in `mana_sources.py`.
"""
from __future__ import annotations

import hashlib
import logging
import os
import threading
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Callable

import numpy as np

from sealed_lab.model import manasim
from sealed_lab.model.mana_sources import ALL, BIT, COLORS, SourceDesc, describe_card, static_colors
from sealed_lab.types import Cancelled

log = logging.getLogger(__name__)

CAST_REVISION = 3          # bump with any change of the model below (part of the cache keys)
SLACK = 2                  # the "late" evaluation turn is T + SLACK
TM = 9                     # last turn tracked (T <= 7, + SLACK)
MV_CAP = 7
N_LIB = TM + 1
FEATURE_NAME = "dead_cards"
_POW2 = 1 << np.arange(5)
SPELL_BASE = 1_000_000


# ---------------------------------------------------------------------------------------------
# card arrays
# ---------------------------------------------------------------------------------------------

def cost_groups_from_mana_cost(cost: str | None) -> list[int]:
    """Colour-set bit masks of the coloured pips of a Scryfall mana cost ({U/R} -> {U, R}; generic, X, Phyrexian, twobrid: none)."""
    import re

    out = []
    for tok in re.findall(r"\{([^}]+)\}", cost or ""):
        t = tok.upper()
        parts = t.split("/")
        if any(p == "P" for p in parts) or (parts[0].isdigit() and len(parts) > 1):
            continue
        m = 0
        for p in parts:
            if p in COLORS:
                m |= BIT[p]
        if m:
            out.append(m)
    return out


def _groups_str(g: list[int]) -> str:
    return " ".join("".join(c for c in COLORS if m & BIT[c]) for m in g)


def _groups_of(s: str) -> list[int]:
    return [sum(BIT[c] for c in tok) for tok in s.split() if tok != "-"] if s else []


# layouts whose faces are alternatives of equal standing (the cheaper half counts); other multi-face layouts cast the first face
_SPLIT_LAYOUTS = frozenset({"split", "flip", "aftermath", "modal_dfc", "transform"})


def cost_groups_column(cards, scryfall_cards: list[dict] | None) -> list[str]:
    """`cost_groups` for every row of the card table: the pips of the mana cost as colour groups ('U UR R' = {U}{U/R}{R}).
    From the Scryfall cards when available (split cards: the cheaper half), else from the pip columns."""
    idx = {}
    if scryfall_cards:
        from sealed_lab.ingest import scryfall as sf

        idx = sf.build_index(scryfall_cards)
    out = []
    for r in cards.iter_rows(named=True):
        o = None
        if idx:
            from sealed_lab.ingest import scryfall as sf

            o = sf.lookup(idx, r["name"])
        g = None
        if o is not None:
            faces = o.get("card_faces") or []
            top = o.get("mana_cost") or ""
            if faces and (not top or "//" in top):
                # top-level "A // B" (adventure, omen, prepare, split ...) must not be summed: use the faces
                fcosts = [f.get("mana_cost") for f in faces if f.get("mana_cost")]
                costs = fcosts if o.get("layout") in _SPLIT_LAYOUTS else fcosts[:1]     # split: cheaper half; else the main face
            else:
                costs = [top] if top else []
            if costs:
                gs = [cost_groups_from_mana_cost(c) for c in costs]
                g = min(gs, key=lambda x: (len(x), x))
        out.append((_groups_str(g) or "-") if g is not None else "")
    return out


def with_cost_groups(cards, set_code: str | None = None):
    """The card table with a `cost_groups` column (from the set's Scryfall cache; falls back to the pip columns at use time)."""
    if "cost_groups" in cards.columns:
        return cards
    import json

    import polars as pl

    sc = None
    if set_code:
        try:
            from sealed_lab import config

            p = config.cache_dir(set_code) / "scryfall.json"
            if p.exists():
                sc = json.loads(p.read_text(encoding="utf-8")).get("cards")
        except Exception:  # noqa: BLE001 - optional refinement
            log.info("no Scryfall cache for cost groups", exc_info=True)
    return cards.with_columns(pl.Series("cost_groups", cost_groups_column(cards, sc), dtype=pl.String))


@dataclass
class CastCards:
    """Per-card arrays of the castability model (indexed by dense `card_id`)."""
    n: int
    is_land: np.ndarray
    is_basic: np.ndarray
    mv_eff: np.ndarray            # int: max(rounded mana value, number of coloured pips)
    need: np.ndarray              # (n, 32) int8: need[c, S] = pips of c payable only with colours inside S
    cmask: np.ndarray             # int: colours the card's cost needs
    src_idx: np.ndarray           # int: index into `descs`, -1 = not a source (lands always have one)
    descs: list[SourceDesc]
    tw: list[frozenset]           # lower-case type words (restricted-mana eligibility)
    names: list[str] = field(default_factory=list)
    signature: str = ""

    @classmethod
    def from_df(cls, df) -> "CastCards":
        d = df.sort("card_id")
        n = d.height
        cols = d.columns
        is_land = d["is_land"].to_numpy().astype(bool)
        is_basic = d["is_basic"].to_numpy().astype(bool)
        mv = d["mana_value"].fill_null(0.0).to_numpy().astype(float)
        pips = np.stack([d[f"pip_{c}"].fill_null(0).to_numpy().astype(np.int64) for c in COLORS], axis=1) if n else np.zeros((0, 5), np.int64)
        hyb = d["hybrid"].fill_null(False).to_numpy().astype(bool)
        cg = d["cost_groups"].fill_null("").to_list() if "cost_groups" in cols else [None] * n
        types = d["types"].fill_null("").to_list()
        text = d["oracle_text"].fill_null("").to_list() if "oracle_text" in cols else [""] * n
        prod = d["produces"].fill_null("").to_list()
        names = d["name"].to_list()
        need = np.zeros((n, 32), dtype=np.int8)
        cmask = np.zeros(n, dtype=np.int64)
        mv_eff = np.zeros(n, dtype=np.int64)
        subsets = np.arange(32)
        src_idx = np.full(n, -1, dtype=np.int64)
        descs: list[SourceDesc] = []
        dmap: dict[SourceDesc, int] = {}
        tw = []
        for i in range(n):
            tw.append(frozenset(w for w in types[i].lower().replace("-", " ").replace("—", " ").split() if w))
            if is_land[i]:
                groups = []
            else:
                if cg[i] is not None and (cg[i] != "" or pips[i].sum() == 0):      # "-" = a cost without coloured pips
                    groups = _groups_of(cg[i])
                elif hyb[i] and (pips[i] > 0).sum() > 1:
                    a = sum(BIT[c] for k, c in enumerate(COLORS) if pips[i, k] > 0)
                    groups = [a] * int(pips[i].max())
                else:
                    groups = [BIT[c] for k, c in enumerate(COLORS) for _ in range(int(pips[i, k]))]
            for g in groups:
                need[i, (subsets & g) == g] += 1
                cmask[i] |= g
            mv_eff[i] = max(int(round(mv[i])), len(groups))
            sd = describe_card(names[i], types[i], text[i], prod[i], bool(is_land[i]), bool(is_basic[i]), float(mv[i]))
            if sd is not None:
                k = dmap.get(sd)
                if k is None:
                    k = dmap[sd] = len(descs)
                    descs.append(sd)
                src_idx[i] = k
        h = hashlib.sha1()
        h.update(str((CAST_REVISION, n)).encode())
        for a in (is_land, mv_eff, need, src_idx):
            h.update(np.ascontiguousarray(a).tobytes())
        h.update(repr(descs).encode())
        return cls(n=n, is_land=is_land, is_basic=is_basic, mv_eff=mv_eff, need=need, cmask=cmask, src_idx=src_idx, descs=descs, tw=tw,
                   names=names, signature=h.hexdigest())


_CC_CACHE: dict[int, tuple[object, CastCards]] = {}


def cast_cards(cards) -> CastCards:
    """Accept a CastCards or the card DataFrame (cached by object identity)."""
    if isinstance(cards, CastCards):
        return cards
    key = id(cards)
    hit = _CC_CACHE.get(key)
    if hit is not None and hit[0] is cards:
        return hit[1]
    cc = CastCards.from_df(cards)
    if len(_CC_CACHE) > 8:
        _CC_CACHE.clear()
    _CC_CACHE[key] = (cards, cc)
    return cc


# ---------------------------------------------------------------------------------------------
# the engine
# ---------------------------------------------------------------------------------------------

@dataclass
class CastResult:
    ids: np.ndarray               # card ids of the deck (as passed)
    p: np.ndarray                 # castability per id (1.0 for lands)
    p_curve: np.ndarray           # P(on curve | enough lands)
    p_late: np.ndarray            # P(by T + 2 | enough lands)
    dead: float                   # sum over nonland cards of copies * (1 - p)
    weights: np.ndarray           # copies * p (lands: copies)


def _regime_args(regime):
    keep = getattr(regime, "keep", None)
    return manasim.regime_name(regime or "random"), keep


def regime_key(regime) -> str:
    return getattr(regime, "key", None) or manasim.regime_name(regime or "random")


def analyze(ids: np.ndarray, cnt: np.ndarray, cc: CastCards, regime="random", n_trials: int = manasim.DEFAULT_TRIALS,
            mode: str = "v2", cache: dict | None = None, unconditional: bool = False) -> CastResult:
    """Castability of every card of a deck given as parallel arrays (card ids, copies).

    `regime`: a hand/draw regime name or `manasim.SimRegime` (set-specific keep rule); `mode`: "v2" (calibrated deal) or
    "plain" (plain shuffle, no mulligans). `cache` (a dict) memoises by the deck's source/target class table."""
    ids = np.asarray(ids, dtype=np.int64)
    cnt = np.asarray(cnt, dtype=np.int64)
    keep_ = cnt > 0
    ids, cnt = ids[keep_], cnt[keep_]
    m = len(ids)
    land = cc.is_land[ids]
    p_all = np.ones(m)
    p_cur = np.ones(m)
    p_late = np.ones(m)
    n = int(cnt.sum())
    spell_rows = np.nonzero(~land)[0]
    if n < 8 or len(spell_rows) == 0:
        return CastResult(ids, p_all, p_cur, p_late, 0.0, cnt.astype(float))
    if int(cnt[land].sum()) == 0:
        # no land at all: nothing with a cost can be paid (a card without cost / pips stays castable)
        costly = (cc.cmask[ids] > 0) | (cc.mv_eff[ids] > 0)
        for a_ in (p_all, p_cur, p_late):
            a_[~land & costly] = 0.0
        return _finish(ids, cnt, land, p_all, p_cur, p_late)
    # ---- class table: lands by source description, spells that are sources by description, everything else one dud class
    sidx = cc.src_idx[ids]
    key = np.where(land, sidx, SPELL_BASE + sidx)
    uk, inv = np.unique(key, return_inverse=True)
    counts = np.bincount(inv, weights=cnt, minlength=len(uk)).astype(np.int64)
    Kc = len(uk)
    is_land_c = uk < SPELL_BASE // 2
    Kl = int(is_land_c.sum())
    descs = [cc.descs[k if k < SPELL_BASE // 2 else k - SPELL_BASE] if (k if k < SPELL_BASE // 2 else k - SPELL_BASE) >= 0 else None
             for k in uk.tolist()]
    L = int(counts[:Kl].sum())
    S = n - L
    basic_mask = 0
    for d in descs:
        if d is not None and d.land:
            basic_mask |= d.basic_col
    # ---- targets: distinct (effective mana value, requirement row, restricted-mana eligibility)
    srows = spell_rows
    sids = ids[srows]
    colset = int(np.bitwise_or.reduce(cc.cmask[sids])) if len(sids) else 0
    subs = np.array([s for s in range(1, 32) if (s & ~colset) == 0 and s != 0], dtype=np.int64)
    rsets = sorted({d.restr for d in descs if d is not None and d.rmask and d.restr})
    elig = np.zeros(len(srows), dtype=np.int64)
    for j, rs in enumerate(rsets):
        rw = set(rs)
        for a, c in enumerate(sids.tolist()):
            if rw & cc.tw[c]:
                elig[a] |= 1 << j
    mvs = np.minimum(cc.mv_eff[sids], MV_CAP)
    needm = cc.need[sids][:, subs].astype(np.int64) if len(subs) else np.zeros((len(sids), 0), np.int64)
    tkey = np.column_stack([mvs, elig, needm]) if len(sids) else np.zeros((0, 2), np.int64)
    utk, tinv = np.unique(tkey, axis=0, return_inverse=True)
    tinv = np.asarray(tinv).reshape(-1)
    ck = None
    if cache is not None:
        ck = (uk.tobytes(), counts.tobytes(), utk.tobytes(), basic_mask, regime_key(regime), n_trials, mode, bool(unconditional), cc.signature)
        hit = cache.get(ck)
        if hit is not None:
            pa, pc, pl_ = hit
            p_all[srows], p_cur[srows], p_late[srows] = pa[tinv], pc[tinv], pl_[tinv]
            return _finish(ids, cnt, land, p_all, p_cur, p_late)
    # ---- deal
    N = n_trials
    pos_class = np.repeat(np.arange(Kc), counts)
    # `prod`: colours of each land class (the London-mulligan bottoming keeps duals and bottoms the most redundant basic)
    prod_c = np.zeros((Kc, 5), dtype=bool)
    for k in range(Kl):
        d = descs[k]
        if d is not None:
            prod_c[k] = [bool(static_colors(d, basic_mask) & BIT[c]) for c in COLORS]
    dc = SimpleNamespace(counts=counts, pos_class=pos_class, is_land=is_land_c, Kl=Kl, prod=prod_c)
    if mode == "plain":
        p1, _p2, ondraw = manasim._tables(N, n)
        seq = pos_class.astype(np.int16)[p1[:, :min(7 + N_LIB, n)]]
        if seq.shape[1] < 7 + N_LIB:
            seq = np.concatenate([seq, np.full((N, 7 + N_LIB - seq.shape[1]), Kc, np.int16)], axis=1)
        hand, lib = seq[:, :7], seq[:, 7:7 + N_LIB]
        onplay = (~ondraw).astype(np.int64)
    else:
        rn, keep = _regime_args(regime)
        deal = manasim.deal_v2(dc, n, L, S, N, rn, keep, w_lib=N_LIB)
        hand, lib = deal["hand"], deal["lib"]
        onplay = deal["play"]
    # first-seen turn of every card seen: opening hand 0, library card i on turn i + 1 + onplay
    cls_all = np.concatenate([hand, lib], axis=1).astype(np.int64)
    turn_all = np.concatenate([np.zeros((N, 7), np.int64), np.arange(N_LIB)[None, :] + 1 + onplay[:, None]], axis=1)
    okc = (cls_all < Kc) & (turn_all <= TM)
    trial = np.broadcast_to(np.arange(N)[:, None], cls_all.shape)
    flat = ((cls_all * (TM + 1) + turn_all) * N + trial)[okc]
    CUM = np.cumsum(np.bincount(flat, minlength=Kc * (TM + 1) * N).reshape(Kc, TM + 1, N), axis=1).astype(np.float32)

    def avail(k: int, T: int) -> np.ndarray:
        d = descs[k]
        if d is None:
            return np.zeros(N, np.float32)
        if d.land:
            t = (T, T - 1, max(T - 1, min(T, 3)), T if T >= 3 else T - 1)[d.rule]
            return CUM[k, t] if t >= 0 else np.zeros(N, np.float32)
        if T - 1 >= d.min_turn:
            return CUM[k, T - 1] * d.mult
        return np.zeros(N, np.float32)

    def landlike(T: int) -> np.ndarray:
        tot = np.zeros(N, np.float32)
        for k, d in enumerate(descs):
            if d is not None and d.land_like:
                tot += CUM[k, T] if d.land else (CUM[k, T - 1] if T - 1 >= d.min_turn else 0.0)
        return tot

    enab_basic = [k for k, d in enumerate(descs) if d is not None and d.land and d.basic_col]
    enab_type = [k for k, d in enumerate(descs) if d is not None and d.land and d.type_col]
    rset_of = {rs: j for j, rs in enumerate(rsets)}
    cov_cache: dict[tuple[int, int], np.ndarray] = {}

    def ok_rows(T: int, e: int) -> np.ndarray:
        """(nS, N) number of sources able to produce a colour of each subset, for spells with eligibility bits `e` at turn T."""
        ck2 = (T, e)
        hit_ = cov_cache.get(ck2)
        if hit_ is not None:
            return hit_
        masks: list[int] = []
        vecs: list[np.ndarray] = []
        for k, d in enumerate(descs):
            if d is None:
                continue
            a = avail(k, T)
            if not a.any():
                continue
            base = d.mask
            if d.fetch:
                base |= basic_mask if d.fetch == "basic" else sum(BIT[c] for c in d.fetch)
            if d.rmask and d.restr and (e >> rset_of[d.restr]) & 1:
                base |= d.rmask
            if d.cond_second:
                en = np.zeros(N, np.float32)
                if d.cond_basic:
                    for kk in enab_basic:
                        en += CUM[kk, T]
                if d.cond_need:
                    for kk in enab_type:
                        if descs[kk].type_col & d.cond_need:
                            en += CUM[kk, T]
                enb = (en >= 1).astype(np.float32)
                masks += [base | d.cond_second, base]
                vecs += [a * enb, a * (1.0 - enb)]
            else:
                masks.append(base)
                vecs.append(a)
        if not masks:
            cov = np.zeros((len(subs), N), np.float32)
        else:
            SPm = np.stack(vecs)                                            # (Kv, N)
            CM = ((np.array(masks)[None, :] & subs[:, None]) > 0).astype(np.float32)      # (nS, Kv)
            cov = CM @ SPm
        cov_cache[ck2] = cov
        return cov

    nU = len(utk)
    pa = np.ones(nU)
    pc = np.ones(nU)
    pl_ = np.ones(nU)
    ll_cache: dict[int, np.ndarray] = {}
    for u in range(nU):
        mv_u, e_u = int(utk[u, 0]), int(utk[u, 1])
        need_u = utk[u, 2:]
        T0 = max(mv_u, 1)
        res = []
        for T in (T0, min(T0 + SLACK, TM)):
            cov = ok_rows(T, e_u)
            ok = (cov >= need_u[:, None]).all(axis=0) if len(need_u) else np.ones(N, bool)
            if T not in ll_cache:
                ll_cache[T] = landlike(T)
            cond = ll_cache[T] >= mv_u
            if unconditional or not cond.any():
                res.append(float(ok.mean()))
            else:
                res.append(float((ok & cond).sum() / cond.sum()))
        pc[u], pl_[u] = res
        pa[u] = 0.5 * (res[0] + res[1])
    if cache is not None:
        if len(cache) > 30000:
            cache.clear()
        cache[ck] = (pa, pc, pl_)
    p_all[srows], p_cur[srows], p_late[srows] = pa[tinv], pc[tinv], pl_[tinv]
    return _finish(ids, cnt, land, p_all, p_cur, p_late)


def _finish(ids, cnt, land, p, pcur, plate) -> CastResult:
    w = np.where(land, cnt, cnt * p).astype(float)
    dead = float((cnt * (1.0 - p))[~land].sum())
    return CastResult(ids, p, pcur, plate, dead, w)


def analyze_deck(deck: dict[int, int], cc: CastCards, regime="random", **kw) -> CastResult:
    ids = np.fromiter((k for k, v in deck.items() if v > 0), dtype=np.int64)
    cnt = np.fromiter((v for v in deck.values() if v > 0), dtype=np.int64)
    return analyze(ids, cnt, cc, regime, **kw)


# ---------------------------------------------------------------------------------------------
# exact cross-check for basics-only decks
# ---------------------------------------------------------------------------------------------

def exact_two_colour(n_cards: int, n_a: int, n_b: int, n_other_lands: int, need_a: int, need_b: int, seen: int, min_lands: int = 0,
                     unconditional: bool = False) -> float:
    """P(at least `need_a` sources of colour A and `need_b` of colour B among `seen` cards | at least `min_lands` lands seen)
    for a plain shuffle of a deck with n_a / n_b single-colour lands, `n_other_lands` other lands and the rest spells.
    A pure two-colour basics deck (the other-land and the 'enough lands' conditions optional)."""
    from math import comb

    n_sp = n_cards - n_a - n_b - n_other_lands
    tot = comb(n_cards, seen)
    num = den = 0.0
    for a in range(0, min(n_a, seen) + 1):
        for b in range(0, min(n_b, seen - a) + 1):
            for o in range(0, min(n_other_lands, seen - a - b) + 1):
                s = seen - a - b - o
                if s < 0 or s > n_sp:
                    continue
                pr = comb(n_a, a) * comb(n_b, b) * comb(n_other_lands, o) * comb(n_sp, s) / tot
                if not unconditional and a + b + o < min_lands:
                    continue
                den += pr
                if a >= need_a and b >= need_b:
                    num += pr
    return num / den if den > 0 else float("nan")


def exact_plain_p(n_cards: int, n_a: int, n_b: int, n_other_lands: int, need_a: int, need_b: int, mv: int, slack: int = SLACK,
                  unconditional: bool = False) -> tuple[float, float, float]:
    """Exact (p, p_curve, p_late) of a spell with `need_a` A-pips and `need_b` B-pips and mana value `mv` under the plain
    shuffle, averaged over play / draw (cards seen by turn T: 6 + T on the play, 7 + T on the draw)."""
    T0 = max(mv, 1)
    res = []
    for T in (T0, min(T0 + slack, TM)):
        v = 0.5 * (exact_two_colour(n_cards, n_a, n_b, n_other_lands, need_a, need_b, 6 + T, mv, unconditional)
                   + exact_two_colour(n_cards, n_a, n_b, n_other_lands, need_a, need_b, 7 + T, mv, unconditional))
        res.append(v)
    return 0.5 * (res[0] + res[1]), res[0], res[1]


# ---------------------------------------------------------------------------------------------
# whole corpus
# ---------------------------------------------------------------------------------------------

def cast_chunk_worker(task: dict):
    """Top-level worker: castability of a chunk of decks given as CSR-like arrays."""
    cc: CastCards = task["cc"]
    off, cid, cn = task["offsets"], task["card_ids"], task["counts"]
    m = len(off) - 1
    dead = np.zeros(m)
    p = np.ones(len(cid), dtype=np.float32)
    cache: dict = {}
    for i in range(m):
        a, b = off[i], off[i + 1]
        r = analyze(cid[a:b], cn[a:b], cc, task["regime"], task["n_trials"], cache=cache)
        dead[i] = r.dead
        if len(r.p) == b - a:
            p[a:b] = r.p
    return task["start"], dead, p


def compute_cast(offsets: np.ndarray, card_ids: np.ndarray, counts: np.ndarray, cards, regime="random",
                 n_trials: int = manasim.DEFAULT_TRIALS, n_jobs: int | None = None, progress: Callable[[float, str], None] | None = None,
                 cancel: threading.Event | None = None, chunk: int = 600, label: str = "Castability") -> tuple[np.ndarray, np.ndarray]:
    """(dead per deck, p per (deck, card) aligned with `card_ids`) for decks in CSR arrays. Parallel (spawn)."""
    from sealed_lab.model.glm import run_parallel

    cc = cast_cards(cards)
    n = len(offsets) - 1
    dead = np.zeros(n)
    p = np.ones(len(card_ids), dtype=np.float32)
    if n == 0:
        return dead, p
    tasks = []
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        a, b = offsets[s], offsets[e]
        tasks.append(dict(cc=cc, offsets=offsets[s:e + 1] - a, card_ids=card_ids[a:b], counts=counts[a:b], n_trials=n_trials,
                          start=s, regime=regime, a0=int(a)))
    n_jobs = int(n_jobs or os.cpu_count() or 1)
    done = 0

    def put(res, task):
        s, d, pp = res
        dead[s:s + len(d)] = d
        a0 = task["a0"]
        p[a0:a0 + len(pp)] = pp

    if n_jobs <= 1 or len(tasks) == 1:
        for t in tasks:
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            put(cast_chunk_worker(t), t)
            done += len(t["offsets"]) - 1
            if progress:
                progress(done / n, f"{label} {done}/{n} decks")
        return dead, p
    by_start = {t["start"]: t for t in tasks}
    for res in run_parallel(cast_chunk_worker, tasks, n_jobs, cancel=cancel):
        put(res, by_start[res[0]])
        done += len(res[1])
        if progress:
            progress(done / n, f"{label} {done}/{n} decks")
    return dead, p


def cache_path(cache_dir, regime) -> "os.PathLike":
    from pathlib import Path

    return Path(cache_dir) / f"cast_cache_{regime_key(regime).replace('+', '_')}.npz"


def cast_signature(cc: CastCards, regime, n_trials: int = manasim.DEFAULT_TRIALS) -> str:
    return hashlib.sha1(f"{cc.signature}|{regime_key(regime)}|{n_trials}|{manasim.SIM2_REVISION}|{SLACK}".encode()).hexdigest()


# ---------------------------------------------------------------------------------------------
# colour sources per deck and "is this deck like the real ones" reference statistics
# ---------------------------------------------------------------------------------------------

@dataclass
class SourceArrays:
    """Per-card contributions to 'sources of colour c' (unrestricted mana only) and to the deck's spell colours."""
    src: np.ndarray            # (n, 5) float: static sources of each colour per copy
    fetch_basic: np.ndarray    # (n,) float: copies that fetch a basic of the deck's colours (count once per colour the deck has basics of)
    basic_col: np.ndarray      # (n, 5) bool: the card is a basic land of colour c
    spell_col: np.ndarray      # (n, 5) bool: a nonland card with a single-colour pip requirement of colour c


_SA_CACHE: dict[int, tuple[object, SourceArrays]] = {}


def source_arrays(cc: CastCards) -> SourceArrays:
    hit = _SA_CACHE.get(id(cc))
    if hit is not None and hit[0] is cc:
        return hit[1]
    n = cc.n
    src = np.zeros((n, 5))
    fb = np.zeros(n)
    bc = np.zeros((n, 5), dtype=bool)
    for i in range(n):
        k = cc.src_idx[i]
        if k < 0:
            continue
        d = cc.descs[k]
        m = d.mask | d.cond_second
        if d.fetch and d.fetch != "basic":
            m |= sum(BIT[c] for c in d.fetch)
        for c in range(5):
            if m >> c & 1:
                src[i, c] = d.mult
        if d.fetch == "basic":
            fb[i] = d.mult
        if d.land and d.basic_col:
            for c in range(5):
                bc[i, c] = bool(d.basic_col >> c & 1)
    sc = np.zeros((n, 5), dtype=bool)
    for c in range(5):
        sc[:, c] = (cc.need[:, 1 << c] > 0) & ~cc.is_land
    out = SourceArrays(src, fb, bc, sc)
    if len(_SA_CACHE) > 8:
        _SA_CACHE.clear()
    _SA_CACHE[id(cc)] = (cc, out)
    return out


def sources_and_colours(copies, cc: CastCards) -> tuple[np.ndarray, np.ndarray]:
    """For decks given as a dense or sparse (n_decks, n_cards) copies matrix: sources per colour (n_decks, 5) and a bool matrix of
    the colours the deck's spells need. Basic-land fetchers count for each colour the deck has a basic land of."""
    import scipy.sparse as sp

    sa = source_arrays(cc)
    M = copies if sp.issparse(copies) else sp.csr_matrix(copies)
    src = np.asarray(M @ sa.src)
    nb = np.asarray(M @ sa.basic_col.astype(float)) > 0
    fb = np.asarray(M @ sa.fetch_basic).reshape(-1, 1)
    src = src + fb * nb
    need = (np.asarray(M @ sa.spell_col.astype(float)) > 0)
    return src, need


def deck_copies(deck: dict[int, int], n_cards: int):
    import scipy.sparse as sp

    ids = [int(k) for k, v in deck.items() if v > 0 and 0 <= int(k) < n_cards]
    return sp.csr_matrix((np.array([deck[i] for i in ids], dtype=float), ([0] * len(ids), ids)), shape=(1, n_cards))


def support_reference(copies, weights: np.ndarray, cc: CastCards, quantile: float = 0.95, tail: float = 0.02) -> dict:
    """What real decks look like: distribution (weighted by games) of the number of spell colours and of the thinnest spell
    colour's source count; the thresholds beyond which a deck is outside the data (more colours than `quantile` of the games use /
    a colour with so few sources that fewer than `tail` of the games have one)."""
    src, need = sources_and_colours(copies, cc)
    ncol = need.sum(1)
    thin = np.where(need, src, np.inf).min(1)
    w = np.asarray(weights, dtype=float)
    w = w / max(w.sum(), 1e-12)
    dist_c = {int(k): float(w[ncol == k].sum()) for k in range(0, 6)}
    cum = 0.0
    kmax = 5
    for k in range(0, 6):
        cum += dist_c[k]
        if cum >= quantile:
            kmax = k
            break
    finite = thin[np.isfinite(thin)]
    wf = w[np.isfinite(thin)]
    dist_s = {int(s): float(wf[finite == s].sum()) for s in range(0, 7)}
    t, cum = 1, 0.0
    for s in range(0, 7):
        cum += dist_s[s]
        if cum <= tail:
            t = s
        else:
            break
    return dict(max_spell_colours=int(kmax), colours_dist=dist_c, thin_sources_max=int(max(t, 2)), thin_dist=dist_s,
                quantile=quantile, tail=tail)


def support_flags(deck: dict[int, int], cc: CastCards, ref: dict | None, n_lands_share: dict | None, fmt_label: str = "") -> list[str]:
    """Plain-language reasons the deck is outside what real decks of the format look like (empty = within the data)."""
    out: list[str] = []
    n = int(sum(v for v in deck.values() if v > 0))
    if n != 40:
        out.append(f"{n} cards: real decks have 40")
    nl = int(sum(v for k, v in deck.items() if v > 0 and 0 <= k < cc.n and cc.is_land[k]))
    if n_lands_share:
        ok = {int(k) for k, v in n_lands_share.items() if float(v) >= 0.02}
        if ok and nl not in ok:
            share = float(n_lands_share.get(str(nl), n_lands_share.get(nl, 0.0)))
            out.append(f"{nl} lands: only {100 * share:.1f}% of real games use that many (the model is fitted on "
                       f"{min(ok)}-{max(ok)})")
    M = deck_copies(deck, cc.n)
    src, need = sources_and_colours(M, cc)
    src, need = src[0], need[0]
    if ref is not None:
        kmax = int(ref.get("max_spell_colours", 5))
        if int(need.sum()) > kmax:
            out.append(f"{int(need.sum())} spell colours: {100 * float(ref.get('quantile', 0.95)):.0f}% of real games use at most {kmax}")
        t = int(ref.get("thin_sources_max", 2))
    else:
        t = 2
    thin = [(COLORS[c], int(src[c])) for c in range(5) if need[c] and src[c] <= t]
    if thin:
        parts = ", ".join(f"{c}: {k} source{'s' if k != 1 else ''}" for c, k in thin)
        out.append(f"spells need a colour with very few sources ({parts}); real decks almost never go that thin")
    return out


def category_summary(cards) -> dict[str, dict]:
    """{source category: dict(n, names)} over the card table (report / documentation)."""
    from sealed_lab.model.mana_sources import category_counts

    return {k: dict(n=len(v), names=v[:12]) for k, v in sorted(category_counts(cards).items())}
