"""Vectorised Monte-Carlo mana simulator (the "mana" structure features).

All trials of one deck are simulated at once with numpy; the random shuffles are *common random numbers*
(a fixed table of permutations), so the result is a deterministic function of the deck multiset and two decks
that differ by a card differ only by the effect of that card plus small, coherent Monte-Carlo noise.

Simplifications (documented, validated against 17lands in the report):
  * cards are collapsed into *classes* (identical cost / colours / mana production), spells are cast greedily
    highest-mana-value first; colour feasibility uses Hall's condition on the sources in play (a dual land is one
    source, hybrid pips can be paid by any of their colours); mana spent in one turn is only limited by the total
    mana available, not by colour overlaps between several spells cast in the same turn;
  * mana rocks / dorks produce from the turn after they are cast;
  * mulligans: keep 2-5 lands in the opening 7, otherwise London-mulligan to 6 (bottom the card that brings the
    land count closest to 3) and keep any 6; play/draw is 50/50 per trial;
  * turns 1..7 are simulated (every statistic only needs up to T7).

Two versions live in this module:
  * v1 (`SIM_VERSION`, the default of `simulate`): a plain random shuffle, keep any 2-5 land hand, one draw per turn.
  * v2 (`simulate(..., version=2, regime=...)`, `SIM2_REVISION`): calibrated to the 17lands replay data (see
    `model/simcalib.py` for how every number was measured):
      - the hand: land count of the 7 cards shown drawn from the regime's distribution (random shuffle, or the Arena
        Bo1 hand smoother fitted as hypergeometric x exp(-(|l - 7L/n|/s)^g)); the cards themselves are random given the count;
      - the keep rule: the real probability of a mulligan by lands in the hand, on the play / draw and mulligan number
        (London mulligans, up to two, bottoming a land when it holds >= 4 lands, else the dearest spell);
      - the draws: a random library, or (Sealed queue) a land-balanced one - the next card is a land with probability
        expit(logit(hypergeometric) - beta * (lands seen - cards seen * L / n));
      - card flow: spells whose oracle text draws cards draw them when cast (each with probability KAPPA_DRAW), spells that
        search the library for a land put one into the hand (KAPPA_FETCH).
    Regimes: "random" (Trad / Bo3 queues), "hand" (Bo1 draft queues), "sealed" (the Bo1 Sealed queue).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

SIM_VERSION = 5          # v1; unchanged so that cached v1 tables stay valid
SIM2_REVISION = 2        # v2: bump with any change to the v2 simulator or its parameters (cache key)
SIM_SEED = 20240229
DEFAULT_TRIALS = 2000
N_TURNS = 7
STAT_NAMES = ["p_screw", "p_flood", "p_color_screw", "mana_spent", "mull_rate"]
EXTRA_STATS = ["lands7_kept", "lands7_all"]
_POW2 = 1 << np.arange(5)


@dataclass(frozen=True)
class CardArrays:
    """Per-card numpy view of the corpus card table, indexed by dense `card_id`."""
    n: int
    is_land: np.ndarray
    is_basic: np.ndarray
    is_creature: np.ndarray
    mv: np.ndarray          # float
    pips: np.ndarray        # (n, 5) int
    hybrid: np.ndarray      # bool
    produces: np.ndarray    # (n, 5) bool
    pbits: np.ndarray       # int, bitmask of produced colours
    draw_n: np.ndarray | None = None   # int: cards the spell draws when cast (oracle text); None = unknown (zeros)
    fetch: np.ndarray | None = None    # int: land ramp code (0 none, 1 search -> hand, 2 search -> battlefield, 3 hand -> battlefield)

    @property
    def draw_arr(self) -> np.ndarray:
        return self.draw_n if self.draw_n is not None else np.zeros(self.n, dtype=np.int64)

    @property
    def fetch_arr(self) -> np.ndarray:
        return self.fetch if self.fetch is not None else np.zeros(self.n, dtype=np.int64)

    @classmethod
    def from_df(cls, df) -> "CardArrays":
        """Build from the corpus `cards` DataFrame (must have dense card_id 0..N-1)."""
        d = df.sort("card_id")
        n = d.height
        ids = d["card_id"].to_numpy()
        if n and (ids[0] != 0 or ids[-1] != n - 1):
            raise ValueError("card_id must be dense 0..N-1")
        pips = np.stack([d[f"pip_{c}"].to_numpy().astype(np.int64) for c in "WUBRG"], axis=1) if n else np.zeros((0, 5), np.int64)
        prod = np.zeros((n, 5), dtype=bool)
        for i, s in enumerate(d["produces"].fill_null("").to_list()):
            for ch in s:
                k = "WUBRG".find(ch)
                if k >= 0:
                    prod[i, k] = True
        is_land = d["is_land"].to_numpy().astype(bool)
        dn = np.zeros(n, dtype=np.int64)
        fe = np.zeros(n, dtype=np.int64)
        if "oracle_text" in d.columns:
            for i, t in enumerate(d["oracle_text"].fill_null("").to_list()):
                if t and not is_land[i]:
                    dn[i], fe[i] = parse_card_flow(t)
        return cls(n=n, is_land=is_land, is_basic=d["is_basic"].to_numpy().astype(bool),
                   is_creature=d["is_creature"].to_numpy().astype(bool),
                   mv=d["mana_value"].fill_null(0.0).to_numpy().astype(float), pips=pips,
                   hybrid=d["hybrid"].fill_null(False).to_numpy().astype(bool), produces=prod,
                   pbits=(prod * _POW2).sum(1).astype(np.int64), draw_n=dn, fetch=fe)


_NUM = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "x": 1}
_DRAW_RE = re.compile(r"\bdraws? (a|an|one|two|three|four|x|\d+) cards?\b(?! from)", re.I)
_OPP_RE = re.compile(r"(opponent|target player|that player|defending player|target creature's controller)[^.]{0,25}$", re.I)
_SEARCH_RE = re.compile(r"searche?s? (?:your|their) library for (?:a|an|up to (?:one|two|three)|one|two)[^.;]{0,40}\bland", re.I)
_DROP_RE = re.compile(r"put (?:a|up to (?:one|two)) land cards? from your hand onto the battlefield|play an additional land", re.I)


def parse_card_flow(text: str) -> tuple[int, int]:
    """(cards drawn, land ramp code) of a non-land card's oracle text.

    cards drawn: 'draw a card / two cards' (not by an opponent, not a loot that discards right after).
    ramp code: 1 = searches the library for a land card into the hand, 2 = ... and puts it onto the battlefield, 3 = puts a
    land from the hand onto the battlefield (or allows an additional land drop).
    Conditional and repeatable triggers count like unconditional ones; KAPPA_DRAW / KAPPA_FETCH scale them to real games."""
    t = text.replace("\n", " ")
    drawn = 0
    for m in _DRAW_RE.finditer(t):
        if _OPP_RE.search(t[max(0, m.start() - 45):m.start()]):
            continue
        if re.search(r"then discard|, discard a card", t[m.end():m.end() + 30], re.I):
            continue
        k = m.group(1).lower()
        drawn += _NUM.get(k) or (int(k) if k.isdigit() else 1)
    code = 0
    m = _SEARCH_RE.search(t)
    if m:
        sentence = t[m.start():t.find(".", m.end()) if t.find(".", m.end()) > 0 else len(t)]
        code = 2 if "onto the battlefield" in sentence else 1
    if _DROP_RE.search(t):
        code = max(code, 3) if code == 0 else code
    return min(drawn, 3), code


_CA_CACHE: dict[int, tuple[object, CardArrays]] = {}


def card_arrays(cards) -> CardArrays:
    """Accept a CardArrays or the corpus cards DataFrame (cached by object identity)."""
    if isinstance(cards, CardArrays):
        return cards
    key = id(cards)
    hit = _CA_CACHE.get(key)
    if hit is not None and hit[0] is cards:
        return hit[1]
    ca = CardArrays.from_df(cards)
    if len(_CA_CACHE) > 8:
        _CA_CACHE.clear()
    _CA_CACHE[key] = (cards, ca)
    return ca


# ---------------------------------------------------------------------------------------------
# class table of one deck
# ---------------------------------------------------------------------------------------------

@dataclass
class DeckClasses:
    n_cards: int
    counts: np.ndarray      # (Kc,) copies per class
    Kl: int                 # number of land classes (first Kl classes)
    mv: np.ndarray          # (Kc,) total cost (coloured + generic)
    is_land: np.ndarray
    prod: np.ndarray        # (Kc,5) bool
    pips: np.ndarray        # (Kc,5) int (coloured pips; hybrid: n pips payable by any colour listed in `hyb_mask`)
    hyb: np.ndarray         # (Kc,) bool
    pos_class: np.ndarray   # (n_cards,) class index per canonical position
    draw_n: np.ndarray | None = None   # (Kc,) cards drawn on cast (v2 class tables only)
    creature: np.ndarray | None = None  # (Kc,) bool, v2 class tables only
    fetch: np.ndarray | None = None    # (Kc,) land ramp code on cast (v2 class tables only)


def deck_classes(deck: dict[int, int], ca: CardArrays, v2: bool = False) -> DeckClasses:
    ids = np.fromiter(deck.keys(), dtype=np.int64, count=len(deck))
    cnt = np.fromiter(deck.values(), dtype=np.int64, count=len(deck))
    keep = cnt > 0
    ids, cnt = ids[keep], cnt[keep]
    land = ca.is_land[ids]
    pb = ca.pbits[ids]
    pips = ca.pips[ids]
    hyb = ca.hybrid[ids] & ((pips > 0).sum(1) > 1)
    colored = np.where(hyb, pips.max(1), pips.sum(1))
    mvf = np.maximum(np.rint(ca.mv[ids]), colored)
    mvi = np.clip(mvf, 0, 15).astype(np.int64)
    pl_ = np.clip(pips, 0, 15)
    spell = ~land
    key = ((spell.astype(np.int64) << 40) | (np.where(spell, mvi, 0) << 32) | (np.where(spell | land, pb, 0) << 24)
           | (np.where(spell, hyb, False).astype(np.int64) << 20)
           | np.where(spell, (pl_[:, 0] << 16) | (pl_[:, 1] << 12) | (pl_[:, 2] << 8) | (pl_[:, 3] << 4) | pl_[:, 4], 0))
    if v2:      # spells that draw cards / fetch lands are their own classes
        dn_ = np.where(spell, np.clip(ca.draw_arr[ids], 0, 3), 0).astype(np.int64)
        fe_ = np.where(spell, np.clip(ca.fetch_arr[ids], 0, 3), 0).astype(np.int64)
        key = key | (dn_ << 42) | (fe_ << 45) | (np.where(spell, ca.is_creature[ids], False).astype(np.int64) << 48)
    uk, first, inv = np.unique(key, return_index=True, return_inverse=True)
    Kc = len(uk)
    counts = np.bincount(inv, weights=cnt, minlength=Kc).astype(np.int64)
    rep = ids[first]
    r_land = land[first]
    r_spell = ~r_land
    r_pips = np.where(r_spell[:, None], pl_[first], 0)
    r_hyb = np.where(r_spell, hyb[first], False)
    r_mv = np.where(r_spell, mvi[first], 1).astype(np.int64)
    r_prod = ca.produces[rep]
    order_pos = np.repeat(np.arange(Kc), counts)
    r_dn = r_fe = r_cr = None
    if v2:
        r_cr = np.where(r_spell, ca.is_creature[rep], False)
        r_dn = np.where(r_spell, np.clip(ca.draw_arr[rep], 0, 3), 0).astype(np.int64)
        r_fe = np.where(r_spell, np.clip(ca.fetch_arr[rep], 0, 3), 0).astype(np.int64)
    return DeckClasses(n_cards=int(counts.sum()), counts=counts, Kl=int(r_land.sum()), mv=r_mv, is_land=r_land,
                       prod=r_prod, pips=r_pips, hyb=r_hyb, pos_class=order_pos, draw_n=r_dn, fetch=r_fe, creature=r_cr)


# ---------------------------------------------------------------------------------------------
# fixed random tables (common random numbers)
# ---------------------------------------------------------------------------------------------
_TABLES: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def _tables(n_trials: int, n: int):
    key = (n_trials, n)
    t = _TABLES.get(key)
    if t is None:
        rng = np.random.default_rng([SIM_SEED, n, n_trials])
        dt = np.uint8 if n < 256 else np.uint16
        p1 = np.argsort(rng.random((n_trials, n), dtype=np.float32), axis=1).astype(dt)
        p2 = np.argsort(rng.random((n_trials, n), dtype=np.float32), axis=1).astype(dt)
        ondraw = rng.random(n_trials) < 0.5
        if len(_TABLES) > 12:
            _TABLES.clear()
        t = (p1, p2, ondraw)
        _TABLES[key] = t
    return t


def _subsets(colset: int) -> np.ndarray:
    """All non-empty sub-masks of `colset`."""
    out = []
    s = colset
    while s:
        out.append(s)
        s = (s - 1) & colset
    return np.array(sorted(out), dtype=np.int64) if out else np.zeros(0, dtype=np.int64)


# ---------------------------------------------------------------------------------------------
# the simulator
# ---------------------------------------------------------------------------------------------

def _no_lands(out: dict, extra: bool, detail: bool) -> dict:
    """A deck without a single land never casts anything: screwed every game, every hand mulliganed (0 lands is never kept)."""
    out.update(p_screw=1.0, p_flood=0.0, p_color_screw=0.0, mana_spent=0.0, mull_rate=1.0)
    if extra:
        out.update(lands7_kept=0.0, lands7_all=0.0)
    if detail:
        out.update(lands_by_turn=[0.0] * N_TURNS, mana_by_turn=[0.0] * N_TURNS, xdraw_by_turn=[0.0] * N_TURNS, cre_by_turn=[0.0] * N_TURNS, oth_by_turn=[0.0] * N_TURNS,
                   hand7=[1.0] + [0.0] * 7, mull_n=[1.0, 1.0])
    return out


def simulate_v1(deck: dict[int, int], cards, n_trials: int = DEFAULT_TRIALS, extra: bool = False,
                detail: bool = False) -> dict:
    """v1: return {p_screw, p_flood, p_color_screw, mana_spent, mull_rate} (+ lands7_kept, lands7_all if `extra`)."""
    ca = card_arrays(cards)
    dc = deck_classes(deck, ca)
    n = dc.n_cards
    N = n_trials
    out = {k: 0.0 for k in STAT_NAMES}
    if n < 8:
        if extra:
            out.update({k: 0.0 for k in EXTRA_STATS})
        return out
    if dc.Kl == 0:
        return _no_lands(out, extra, detail)
    Kc, Kl = len(dc.counts), dc.Kl
    Ksp = Kc - Kl
    p1, p2, ondraw = _tables(N, n)
    ar = np.arange(N)
    pos_class = dc.pos_class.astype(np.int16)
    seq = pos_class[p1[:, :16]]                          # (N, 16) class of the first 16 cards of shuffle 1
    land_flag = dc.is_land
    l7 = land_flag[seq[:, :7]].sum(1)
    mulled = (l7 < 2) | (l7 > 5)
    mi = np.nonzero(mulled)[0]
    hand = seq[:, :7].copy()
    lib = seq[:, 7:7 + N_TURNS + 1].copy()                # (N, 8) draws for turns
    lands16 = land_flag[seq[:, :16]].sum(1)
    if len(mi):
        seq2 = pos_class[p2[mi, :17]]
        h2 = seq2[:, :7].copy()
        lm = land_flag[h2]
        L = lm.sum(1)
        # bottom: a land if L >= 4 (the highest class index land), else the highest-mv spell (highest class index spell)
        cls_idx = np.arange(7)
        # position of the max-class land / spell in the hand
        key_land = np.where(lm, h2, -1)
        key_spell = np.where(~lm, h2, -1)
        pl = key_land.argmax(1)
        ps = key_spell.argmax(1)
        has_spell = (~lm).any(1)
        bottom_pos = np.where((L >= 4) | ~has_spell, pl, ps)
        keep_mask = np.ones((len(mi), 7), dtype=bool)
        keep_mask[np.arange(len(mi)), bottom_pos] = False
        hand[mi] = h2
        # replace the bottomed card by a sentinel (class Kc -> dropped)
        hand[mi[:, None], bottom_pos[:, None]] = Kc
        lib[mi] = seq2[:, 7:7 + N_TURNS + 1]
        # cards actually seen: the 7 drawn minus the bottomed card, then the next draws (16 cards in total)
        lands16[mi] = land_flag[seq2[:, :17]].sum(1) - land_flag[h2[np.arange(len(mi)), bottom_pos]]
    H = np.zeros((Kc + 1, N), dtype=np.int8)
    for j in range(7):
        H[hand[:, j], ar] += 1
    H = H[:Kc]
    # sources: lands + spell classes that produce mana
    rock_cls = np.nonzero(~dc.is_land & dc.prod.any(1))[0]
    src_cls = np.concatenate([np.arange(Kl), rock_cls]).astype(np.int64)
    Ks = len(src_cls)
    SP = np.zeros((Ks, N), dtype=np.float32)
    src_prod = dc.prod[src_cls]                         # (Ks,5)
    # colour subsets and Hall requirements
    sp_idx = np.arange(Kl, Kc)
    colset = 0
    if Ksp:
        colset = int((((dc.pips[sp_idx] > 0) * _POW2).sum(1)).max(initial=0)) if Ksp else 0
        colset = int(np.bitwise_or.reduce(((dc.pips[sp_idx] > 0) * _POW2).sum(1))) if Ksp else 0
    subs = _subsets(colset)
    nS = len(subs)
    if nS:
        in_s = ((subs[:, None] & _POW2[None, :]) > 0)           # (nS,5)
        CM = ((src_prod[None, :, :] & in_s[:, None, :]).any(2)).astype(np.float32)   # (nS, Ks)
    else:
        CM = np.zeros((0, Ks), dtype=np.float32)
    mv_sp = dc.mv[sp_idx]
    if Ksp:
        pips_sp = dc.pips[sp_idx]
        hyb_sp = dc.hyb[sp_idx]
        nonhyb_need = pips_sp @ in_s.T.astype(np.int64) if nS else np.zeros((Ksp, 0), np.int64)    # (Ksp,nS)
        A_bits = ((pips_sp > 0) * _POW2).sum(1)                                                     # (Ksp,)
        nh = pips_sp.max(1)
        hyb_need = np.where(((A_bits[:, None] & ~subs[None, :]) == 0), nh[:, None], 0) if nS else np.zeros((Ksp, 0), np.int64)
        need = np.where(hyb_sp[:, None], hyb_need, nonhyb_need).astype(np.int16)                    # (Ksp,nS)
        # land-choice weights: Wm[(c,m), j] = w_j * [pips_jc >= m+1]
        wj = 1.0 / (1.0 + mv_sp.astype(np.float32)) ** 2
        Wm = np.zeros((5, 3, Ksp), dtype=np.float32)
        for m in range(3):
            Wm[:, m, :] = (pips_sp.T >= (m + 1)) * wj[None, :]
        Wm = Wm.reshape(15, Ksp)
        order = np.argsort(-mv_sp, kind="stable")
        rock_of = {int(c): k for k, c in enumerate(src_cls)}
        reps = np.minimum(dc.counts[sp_idx], 4)
        cast_plan = [(int(j), int(mv_sp[j]), int(reps[j]), rock_of.get(int(Kl + j), -1)) for j in order]
    prod_land = dc.prod[:Kl].astype(np.float32)                        # (Kl,5)
    P_src = src_prod.T.astype(np.float32)                              # (5,Ks)
    tie = (0.01 * dc.prod[:Kl].sum(1) - 1e-4 * np.arange(Kl)).astype(np.float32)[:, None]

    spent_tot = np.zeros(N, dtype=np.int16)
    share_sum = 0.0
    lands_t3 = None
    lands_turn, mana_turn = [], []
    ondraw_i = ondraw.astype(np.int64)
    for t in range(1, N_TURNS + 1):
        # draw
        j = t - 2 + ondraw_i
        m = j >= 0
        if m.any():
            cj = lib[ar, np.maximum(j, 0)]
            if m.all():
                H[cj, ar] += 1
            else:
                H[cj[m], ar[m]] += 1
        # land drop
        LH = H[:Kl]
        have = LH > 0
        any_land = have.any(0)
        if Ksp:
            present = (H[Kl:] > 0).astype(np.float32)
            A = (Wm @ present).reshape(5, 3, N)
            srcc = P_src @ SP                                           # (5,N)
            want = A[:, 0] * (srcc < 0.5) + A[:, 1] * ((srcc > 0.5) & (srcc < 1.5)) + A[:, 2] * ((srcc > 1.5) & (srcc < 2.5))
            score = prod_land @ want + tie + 1e-3 * LH
        else:
            score = tie + 1e-3 * LH
        score = np.where(have, score, -1e9)
        k = score.argmax(0)
        sel = np.nonzero(any_land)[0]
        if len(sel):
            ks = k[sel]
            H[ks, sel] -= 1
            SP[ks, sel] += 1
        M = SP.sum(0).astype(np.int16)
        spent_t = np.zeros(N, dtype=np.int16)
        if Ksp:
            # colour feasibility (Hall's condition on the sources in play)
            if nS:
                cov = CM @ SP                                               # (nS,N)
                ok = cov[0][None, :] >= need[:, 0, None]
                for si in range(1, nS):
                    ok &= cov[si][None, :] >= need[:, si, None]
            else:
                ok = np.ones((Ksp, N), dtype=bool)
            okr = ok & (H[Kl:] > 0)
            rem = M.copy()
            add = None
            maxM = int(M.max())
            avail = okr.any(1)
            for (jj, mvj, rp, rs) in cast_plan:
                if mvj > maxM or not avail[jj]:
                    continue
                row = H[Kl + jj]
                can = okr[jj] & (rem >= mvj)
                for _ in range(rp):
                    if not can.any():
                        break
                    ci = can.view(np.int8)
                    row -= ci
                    rem -= ci * np.int16(mvj)
                    if rs >= 0:
                        if add is None:
                            add = np.zeros((Ks, N), dtype=np.float32)
                        add[rs] += can
                    can = (row > 0) & ok[jj] & (rem >= mvj)
            spent_t = M - rem
            if 2 <= t <= 6:
                Hs = H[Kl:]
                tot = Hs.sum(0, dtype=np.int16)
                stuck = (Hs * ((~ok) & (mv_sp[:, None] <= M[None, :]))).sum(0, dtype=np.int16)
                share_sum += float((stuck / np.maximum(tot, 1)).mean())
            if add is not None:
                SP += add
        if t <= 7:
            spent_tot += spent_t
        if t == 3:
            lands_t3 = SP[:Kl].sum(0)
        if detail:
            lands_turn.append(float(SP[:Kl].sum(0).mean()))
            mana_turn.append(float(spent_t.mean()))
    out["p_screw"] = float((lands_t3 < 3).mean())
    # >= 9 lands in the first 16 cards: ~7% at 16 lands, ~10% at 17, ~14% at 18 (hypergeometric).
    # (>= 7 was ~50% for a normal deck, which read as "flooding" for every build.)
    out["p_flood"] = float((lands16 >= 9).mean())
    out["p_color_screw"] = share_sum / 5.0
    out["mana_spent"] = float(spent_tot.mean() / 28.0)
    out["mull_rate"] = float(mulled.mean())
    if extra:
        out["lands7_kept"] = float(l7[~mulled].mean()) if (~mulled).any() else 0.0
        out["lands7_all"] = float(l7.mean())
    if detail:
        out.update(lands_by_turn=lands_turn, mana_by_turn=mana_turn, hand7=(np.bincount(l7, minlength=8)[:8] / N).tolist(),
                   mull_n=[float(mulled.mean()), 0.0])
    return out


# =============================================================================================
# v2: calibrated to the replay data (regime-specific hands, real keep rule, card flow)
# =============================================================================================
# Parameters measured on MSH + SOS with model/simcalib.py (`python scripts/fit_manasim.py MSH SOS`).
KAPPA_DRAW = 1.0         # share of the draw effects in a deck's oracle text that deliver a card when the spell is cast (fitted: 1)
KAPPA_FETCH = 0.9        # the same for land-ramp effects (search for a land, put a land from the hand onto the battlefield)
CAST_P_CREATURE = 1.0    # chance that a castable creature is actually cast in a turn (players sequence badly, tap out)
CAST_P_OTHER = 0.25      # the same for other spells (instants, sorceries, auras, artifacts: mostly held for a better moment)
# hand smoother: P(l) ~ hypergeometric(l) * exp(-(|l - 7 L / n| / s) ** g); None = plain shuffle
_HAND = {"random": None, "hand": (1.197, 2.432), "sealed": (0.999, 2.467)}
# land-balanced draws: logit P(land) = logit(hypergeometric) - beta * (lands seen - cards seen * L / n)
_BETA = {"random": 0.04, "hand": 0.05, "sealed": 0.36}
# land rescue: logit added to P(land) of a draw on turn 3, 4, 5+ when the player holds no land at that moment
_RESCUE = {"random": (0.28, 0.16, 0.12), "hand": (0.28, 0.16, 0.12), "sealed": (0.0, 0.0, 0.0)}
# P(mulligan | lands in the 7 cards shown), [mulligan number 0/1][on the draw / on the play][lands 0..7]
_KEEP = {
    "random": [[[0.989, 0.882, 0.163, 0.065, 0.063, 0.212, 0.729, 0.95], [0.999, 0.944, 0.187, 0.065, 0.061, 0.194, 0.733, 0.95]],
               [[0.984, 0.681, 0.06, 0.019, 0.015, 0.061, 0.369, 0.95], [0.999, 0.761, 0.067, 0.019, 0.025, 0.062, 0.418, 0.95]]],
    "hand": [[[1.0, 0.825, 0.186, 0.084, 0.091, 0.271, 0.7, 0.95], [1.0, 0.882, 0.208, 0.078, 0.078, 0.27, 0.7, 0.95]],
             [[1.0, 0.664, 0.075, 0.03, 0.026, 0.109, 0.7, 0.95], [1.0, 0.742, 0.088, 0.03, 0.026, 0.101, 0.7, 0.95]]],
    "sealed": [[[1.0, 0.85, 0.185, 0.077, 0.089, 0.2, 0.7, 0.95], [1.0, 0.85, 0.207, 0.076, 0.073, 0.2, 0.7, 0.95]],
               [[1.0, 0.85, 0.081, 0.027, 0.026, 0.2, 0.7, 0.95], [1.0, 0.85, 0.088, 0.025, 0.024, 0.2, 0.7, 0.95]]],
}
REGIME_NAMES = ("random", "hand", "sealed")
W_LIB = 20               # library cards pre-arranged per trial (7 turns of draws + card-flow draws)
PAD = W_LIB + 8


@dataclass(frozen=True)
class SimRegime:
    """A hand/draw regime together with a set-specific keep rule. The smoother and the draw tilt are properties of Arena's
    queues (the same in every set), but how often players mulligan a hand with k lands depends on the players: `keep`
    ((2 mulligan numbers, 2 on-the-draw/on-the-play, 8 lands) probabilities as nested tuples; None = the shipped
    MSH + SOS average) is measured per set from its replay data (`simcalib.keep_regimes`)."""
    name: str
    keep: tuple | None = None

    def __post_init__(self):
        if self.name not in REGIME_NAMES:
            raise ValueError(f"regime must be one of {REGIME_NAMES}, got {self.name!r}")

    @property
    def key(self) -> str:
        if self.keep is None:
            return self.name
        import hashlib

        return self.name + "+" + hashlib.sha1(np.asarray(self.keep, dtype=np.float64).round(4).tobytes()).hexdigest()[:10]

    @classmethod
    def make(cls, name: str, keep=None) -> "SimRegime":
        return cls(name, None if keep is None else tuple(tuple(tuple(float(round(x, 4)) for x in row) for row in m) for m in keep))


def regime_name(regime) -> str:
    return getattr(regime, "name", regime)


def sim_label(version: int, regime) -> str:
    """Cache label of a simulator configuration: 'v1' or 'v2.<revision>:<regime>[+<keep-rule hash>]'."""
    if int(version) == 1:
        return "v1"
    return f"v2.{SIM2_REVISION}:{getattr(regime, 'key', regime) or 'random'}"


def hand_land_pmf(L: int, n: int, regime, h: int = 7) -> np.ndarray:
    """P(l lands among the 7 cards shown) for a deck with L lands in n cards under the regime's smoother."""
    from math import comb

    p = np.array([comb(L, k) * comb(n - L, h - k) for k in range(h + 1)], dtype=float) / comb(n, h)
    par = _HAND[regime_name(regime)]
    if par is not None:
        s_, g_ = par
        p = p * np.exp(-(np.abs(np.arange(h + 1) - h * L / n) / s_) ** g_)
    return p / p.sum()


def mull_probs(L: int, n: int, regime="random", keep=None) -> tuple[float, float]:
    """(P(at least one mulligan), P(two mulligans)) of a deck with `L` lands in `n` cards, exactly (no sampling noise), under the
    v2 dealing rules: the regime's hand smoother, the real keep rule (`keep`, or the one carried by a `SimRegime`, or the shipped
    average) and a 50/50 play/draw. This is the same law `deal_v2` samples from, so it equals the simulator's mulligan
    statistics up to Monte-Carlo noise; it depends on the land count and the deck size only."""
    if n < 8 or L < 0 or L > n:
        return 0.0, 0.0
    kp = getattr(regime, "keep", None) if keep is None else keep
    name = regime_name(regime)
    k = np.asarray(_KEEP[name] if kp is None else kp, dtype=float)       # (2, 2, 8)
    pmf = hand_land_pmf(int(L), int(n), name)
    p1 = p2 = 0.0
    for play in (0, 1):
        m0 = pmf * k[0, play]
        p1 += 0.5 * float(m0.sum())
        p2 += 0.5 * float(m0.sum()) * float((pmf * k[1, play]).sum())
    return p1, p2


def mull_split(L: int, n: int, regime="random", keep=None) -> tuple[float, float]:
    """(P(exactly one mulligan), P(two or more)) of `mull_probs`: the two levels of the model's mulligan covariates."""
    ge1, ge2 = mull_probs(L, n, regime, keep)
    return max(ge1 - ge2, 0.0), ge2


_V2T: dict[tuple, object] = {}


def _v2_uniforms(N: int):
    """Common random numbers of the v2 simulator: on the draw, per shuffle (hand, keep decision, library types), card-flow draws."""
    t = _V2T.get(("u", N))
    if t is None:
        rng = np.random.default_rng([SIM_SEED, 3, N])
        t = dict(ondraw=rng.random(N) < 0.5,
                 hand=[rng.random(N) for _ in range(3)], keep=[rng.random(N) for _ in range(3)],
                 typ=[rng.random((N, W_LIB)) for _ in range(3)],
                 xd=rng.random((N, 48)), xf=rng.random((N, 16)), cs=rng.random((N, 64)))
        if len(_V2T) > 30:
            _V2T.clear()
        _V2T[("u", N)] = t
    return t


def _v2_queue(N: int, size: int, sh: int, kind: int) -> np.ndarray:
    """(N, size + PAD) int16: a random order of 0..size-1 per trial followed by `size` (the sentinel) PAD times."""
    key = ("q", N, size, sh, kind)
    t = _V2T.get(key)
    if t is None:
        rng = np.random.default_rng([SIM_SEED, 4, N, size, sh, kind])
        perm = np.argsort(rng.random((N, size), dtype=np.float32), axis=1).astype(np.int16) if size else np.zeros((N, 0), np.int16)
        t = np.concatenate([perm, np.full((N, PAD), size, dtype=np.int16)], axis=1)
        if len(_V2T) > 60:
            _V2T.clear()
        _V2T[key] = t
    return t


def _deal(sh: int, idx: np.ndarray, u, N: int, dc: DeckClasses, n: int, L: int, S: int, cdf: np.ndarray, beta: float,
          rescue: tuple[float, float, float] = (0.0, 0.0, 0.0), w_lib: int = W_LIB):
    """Hand (k,7) and pre-arranged library (k,W_LIB) of shuffle `sh` for the trials `idx`: classes, and a land flag per library card."""
    Kc = len(dc.counts)
    landcls = np.concatenate([dc.pos_class[:L], np.full(PAD, Kc)]).astype(np.int16)
    spellcls = np.concatenate([dc.pos_class[L:], np.full(PAD, Kc)]).astype(np.int16)
    lqc = landcls[_v2_queue(N, L, sh, 0)[idx]]
    sqc = spellcls[_v2_queue(N, S, sh, 1)[idx]]
    k = len(idx)
    ak = np.arange(k)
    l = (u["hand"][sh][idx][:, None] > cdf[None, :-1]).sum(1).astype(np.int64)
    c = np.arange(7)[None, :]
    hand = np.where(c < l[:, None], lqc[:, :7], np.take_along_axis(sqc, np.maximum(c - l[:, None], 0), 1))
    nl, ns = l.copy(), 7 - l
    lands_left = (L - l).astype(float)
    lseen = l.astype(float)
    lib = np.empty((k, w_lib), dtype=np.int16)
    isl = np.zeros((k, w_lib), dtype=bool)
    ut = u["typ"][sh][idx]
    onplay = (~u["ondraw"][idx]).astype(np.int64)
    lh = np.where(onplay == 1, np.maximum(l - 1, 0), l)             # lands held when the first regular draw happens
    for i in range(w_lib):
        left = n - 7 - i
        if left > 0:
            p = np.clip(lands_left / left, 0.0, 1.0)
            if beta or any(rescue):
                pc = np.clip(p, 1e-9, 1 - 1e-9)
                z = np.log(pc / (1 - pc)) - beta * (lseen - (7 + i) * L / n)
                turn = i + 1 + onplay                              # the turn of this regular draw
                if any(rescue):
                    z = z + np.where((lh == 0) & (turn >= 3), np.asarray(rescue)[np.minimum(turn - 3, 2)], 0.0)
                p = np.where((p > 0) & (p < 1), 1.0 / (1.0 + np.exp(-z)), p)
        else:
            p = np.zeros(k)
        t = ut[:, i] < p
        lh = lh + t
        lh = lh - (lh > 0)                                         # the land drop of that turn
        lib[:, i] = np.where(t, lqc[ak, np.minimum(nl, lqc.shape[1] - 1)], sqc[ak, np.minimum(ns, sqc.shape[1] - 1)])
        isl[:, i] = t
        nl += t
        ns += ~t
        lands_left -= t
        lseen += t
    return hand.astype(np.int16), lib, isl, l


def deal_v2(dc, n: int, L: int, S: int, N: int, regime, keep_override=None, w_lib: int = W_LIB) -> dict:
    """The v2 deal of N trials: opening hand (7 class ids, the London-mulligan bottomed cards set to the sentinel `Kc`), the
    pre-arranged library (N, w_lib) with a land flag per card, the number of mulligans `m`, the first-hand land count.
    `dc` needs `counts`, `pos_class` (canonical positions, lands first) and `is_land` (per class); `regime` is a regime name.
    Optional `dc.mv` / `dc.prod` (per class) set the London-mulligan bottoming priority (dearest spell; most redundant basic land);
    without them the class order decides.
    Shared by the play simulation and the castability model, so both see the same hands, mulligans and draws."""
    u = _v2_uniforms(N)
    ar = np.arange(N)
    ondraw = u["ondraw"]
    play = (~ondraw).astype(np.int64)
    Kc = len(dc.counts)
    beta = _BETA[regime]
    rescue = _RESCUE[regime]
    cdf = np.cumsum(hand_land_pmf(L, n, regime))
    keep = np.asarray(_KEEP[regime] if keep_override is None else keep_override, dtype=float)       # (2, 2, 8)
    land_ext = np.concatenate([dc.is_land, [False]])

    # ---- hands, mulligans
    hand, lib, lib_isl, l0 = _deal(0, ar, u, N, dc, n, L, S, cdf, beta, rescue, w_lib)
    mull0 = u["keep"][0] < keep[0, play, l0]
    m = mull0.astype(np.int64)
    mi = np.nonzero(mull0)[0]
    if len(mi):
        h1, lb1, li1, l1 = _deal(1, mi, u, N, dc, n, L, S, cdf, beta, rescue, w_lib)
        mull1 = u["keep"][1][mi] < keep[1, play[mi], l1]
        hand[mi], lib[mi], lib_isl[mi] = h1, lb1, li1
        m[mi[mull1]] = 2
        mj = mi[mull1]
        if len(mj):
            h2, lb2, li2, _ = _deal(2, mj, u, N, dc, n, L, S, cdf, beta, rescue, w_lib)
            hand[mj], lib[mj], lib_isl[mj] = h2, lb2, li2
    # London mulligan: bottom a land from a hand with >= 4 lands (or without a spell), else the dearest spell
    kc_ar = np.arange(Kc + 1)
    mv_c = getattr(dc, "mv", None)
    spell_pri = (np.concatenate([np.asarray(mv_c, float), [0.0]]) if mv_c is not None else 0.0 * kc_ar) + 1e-3 * kc_ar     # dearest first
    prod_c = getattr(dc, "prod", None)
    prod_ext = np.concatenate([np.asarray(prod_c, bool), np.zeros((1, 5), bool)]) if prod_c is not None else None
    ncol_c = prod_ext.sum(1) if prod_ext is not None else None
    for b in range(2):
        sel = np.nonzero(m > b)[0]
        if not len(sel):
            break
        hh = hand[sel]
        lm = land_ext[hh]
        real_spell = ~lm & (hh < Kc)
        if prod_ext is not None:
            # land to bottom: colourless first, then the fewest colours (a basic before a dual), then the colour the hand's
            # lands supply most (the most redundant); ties: the highest class
            supply = (prod_ext[hh] & lm[:, :, None]).sum(1)                                  # (k, 5) lands per colour
            red = (prod_ext[hh] * supply[:, None, :]).max(2)                                  # (k, 7) most-supplied colour of each card
            nc = ncol_c[hh]
            pri_l = np.where(nc == 0, 100.0, (3 - np.minimum(nc, 3)) * 10.0) + red + 1e-3 * hh
        else:
            pri_l = hh.astype(float)
        pos_l = np.where(lm, pri_l, -1e9).argmax(1)
        pos_s = np.where(real_spell, spell_pri[hh], -1e9).argmax(1)
        pos = np.where((lm.sum(1) >= 4) | ~real_spell.any(1), pos_l, pos_s)
        hand[sel, pos] = Kc
    kept_lands = land_ext[hand].sum(1)
    return dict(hand=hand, lib=lib, lib_isl=lib_isl, m=m, mull0=mull0, l_first=l0, kept_lands=kept_lands, land_ext=land_ext, play=play)


def simulate_v2(deck: dict[int, int], cards, n_trials: int = DEFAULT_TRIALS, extra: bool = False, regime="random",
                detail: bool = False) -> dict:
    """v2 mana simulation (see the module docstring). Same outputs as v1; `detail` adds per-turn arrays. `regime` is a name
    (random / hand / sealed) or a `SimRegime` carrying a set-specific keep rule."""
    keep_override = getattr(regime, "keep", None)
    regime = regime_name(regime)
    if regime not in REGIME_NAMES:
        raise ValueError(f"regime must be one of {REGIME_NAMES}, got {regime!r}")
    ca = card_arrays(cards)
    dc = deck_classes(deck, ca, v2=True)
    n = dc.n_cards
    N = n_trials
    out: dict = {k: 0.0 for k in STAT_NAMES}
    if n < 8:
        if extra:
            out.update({k: 0.0 for k in EXTRA_STATS})
        return out
    if dc.Kl == 0:
        return _no_lands(out, extra, detail)
    Kc, Kl = len(dc.counts), dc.Kl
    Ksp = Kc - Kl
    L = int(dc.counts[:Kl].sum())
    S = n - L
    u = _v2_uniforms(N)
    ar = np.arange(N)
    ondraw = u["ondraw"]
    deal = deal_v2(dc, n, L, S, N, regime, keep_override)
    hand, lib, lib_isl, l0, m, mull0 = deal["hand"], deal["lib"], deal["lib_isl"], deal["l_first"], deal["m"], deal["mull0"]
    l_first = l0
    land_ext = deal["land_ext"]
    kept_lands = deal["kept_lands"]
    cs = np.cumsum(lib_isl, axis=1)
    lands16 = kept_lands + cs[ar, 8 + m]              # the first 16 cards seen: the kept hand plus the next 9 (+ bottomed) draws
    Hf = np.zeros((Kc + 1, N), dtype=np.int8)
    for j in range(7):
        Hf[hand[:, j], ar] += 1
    H = Hf[:Kc]

    # ---- casting structures (as v1)
    rock_cls = np.nonzero(~dc.is_land & dc.prod.any(1))[0]
    src_cls = np.concatenate([np.arange(Kl), rock_cls]).astype(np.int64)
    Ks = len(src_cls)
    SP = np.zeros((Ks, N), dtype=np.float32)
    src_prod = dc.prod[src_cls]
    sp_idx = np.arange(Kl, Kc)
    colset = int(np.bitwise_or.reduce(((dc.pips[sp_idx] > 0) * _POW2).sum(1))) if Ksp else 0
    subs = _subsets(colset)
    nS = len(subs)
    if nS:
        in_s = ((subs[:, None] & _POW2[None, :]) > 0)
        CM = ((src_prod[None, :, :] & in_s[:, None, :]).any(2)).astype(np.float32)
    else:
        CM = np.zeros((0, Ks), dtype=np.float32)
    mv_sp = dc.mv[sp_idx]
    fetch_cls = -1
    if Ksp:
        pips_sp = dc.pips[sp_idx]
        hyb_sp = dc.hyb[sp_idx]
        nonhyb_need = pips_sp @ in_s.T.astype(np.int64) if nS else np.zeros((Ksp, 0), np.int64)
        A_bits = ((pips_sp > 0) * _POW2).sum(1)
        nh = pips_sp.max(1)
        hyb_need = np.where(((A_bits[:, None] & ~subs[None, :]) == 0), nh[:, None], 0) if nS else np.zeros((Ksp, 0), np.int64)
        need = np.where(hyb_sp[:, None], hyb_need, nonhyb_need).astype(np.int16)
        wj = 1.0 / (1.0 + mv_sp.astype(np.float32)) ** 2
        Wm = np.zeros((5, 3, Ksp), dtype=np.float32)
        for mm in range(3):
            Wm[:, mm, :] = (pips_sp.T >= (mm + 1)) * wj[None, :]
        Wm = Wm.reshape(15, Ksp)
        order = np.argsort(-mv_sp, kind="stable")
        rock_of = {int(c): k for k, c in enumerate(src_cls)}
        reps = np.minimum(dc.counts[sp_idx], 4)
        dn_sp, fe_sp = dc.draw_n[sp_idx], dc.fetch[sp_idx]
        cr_sp = dc.creature[sp_idx]
        cast_plan = [(int(j), int(mv_sp[j]), int(reps[j]), rock_of.get(int(Kl + j), -1), int(dn_sp[j]), int(fe_sp[j]),
                      CAST_P_CREATURE if cr_sp[j] else CAST_P_OTHER, bool(cr_sp[j])) for j in order]
        if Kl and ((fe_sp == 1) | (fe_sp == 2)).any():
            # the land a land-search spell fetches: the one producing the colours the deck's spells need most
            need_c = (dc.pips[sp_idx] * dc.counts[sp_idx][:, None]).sum(0).astype(float)
            fetch_cls = int(np.argmax(dc.prod[:Kl] @ need_c + 1e-3 * dc.counts[:Kl]))
    prod_land = dc.prod[:Kl].astype(np.float32)
    P_src = src_prod.T.astype(np.float32)
    tie = (0.01 * dc.prod[:Kl].sum(1) - 1e-4 * np.arange(Kl)).astype(np.float32)[:, None]

    spent_tot = np.zeros(N, dtype=np.int16)
    share_sum = 0.0
    lands_t3 = None
    ptr = np.zeros(N, dtype=np.int64)
    xd_tot = np.zeros(N, dtype=np.int64)
    lands_turn, mana_turn, xdraw_turn, cre_turn, oth_turn = [], [], [], [], []
    for t in range(1, N_TURNS + 1):
        # draw step (on the play there is none on turn 1)
        di = np.nonzero(ondraw)[0] if t == 1 else ar
        if len(di):
            Hf[np.where(ptr[di] < W_LIB, lib[di, np.minimum(ptr[di], W_LIB - 1)], Kc), di] += 1       # past the last pre-arranged card: no draw
            ptr[di] += 1
        # land drop
        LH = H[:Kl]
        have = LH > 0
        any_land = have.any(0)
        if Ksp:
            present = (H[Kl:] > 0).astype(np.float32)
            A = (Wm @ present).reshape(5, 3, N)
            srcc = P_src @ SP
            want = A[:, 0] * (srcc < 0.5) + A[:, 1] * ((srcc > 0.5) & (srcc < 1.5)) + A[:, 2] * ((srcc > 1.5) & (srcc < 2.5))
            score = prod_land @ want + tie + 1e-3 * LH
        else:
            score = tie + 1e-3 * LH
        score = np.where(have, score, -1e9)
        k = score.argmax(0)
        sel = np.nonzero(any_land)[0]
        if len(sel):
            ks = k[sel]
            H[ks, sel] -= 1
            SP[ks, sel] += 1
        M = SP.sum(0).astype(np.int16)
        spent_t = np.zeros(N, dtype=np.int16)
        if Ksp:
            if nS:
                cov = CM @ SP
                ok = cov[0][None, :] >= need[:, 0, None]
                for si in range(1, nS):
                    ok &= cov[si][None, :] >= need[:, si, None]
            else:
                ok = np.ones((Ksp, N), dtype=bool)
            okr = ok & (H[Kl:] > 0)
            rem = M.copy()
            add = None
            maxM = int(M.max())
            avail = okr.any(1)
            xd = None
            xf = None
            n_cre = n_oth = 0.0
            for (jj, mvj, rp, rs, dnj, fej, cp, iscr) in cast_plan:
                if mvj > maxM or not avail[jj]:
                    continue
                row = H[Kl + jj]
                can = okr[jj] & (rem >= mvj)
                if cp < 1.0:
                    can &= u["cs"][:, (5 * jj + 3 * t) % 64] < cp
                for r_ in range(rp):
                    if not can.any():
                        break
                    ci = can.view(np.int8)
                    row -= ci
                    rem -= ci * np.int16(mvj)
                    if detail:
                        if iscr:
                            n_cre += float(ci.sum())
                        else:
                            n_oth += float(ci.sum())
                    if rs >= 0:
                        if add is None:
                            add = np.zeros((Ks, N), dtype=np.float32)
                        add[rs] += can
                    for q in range(dnj):         # card draw effects: each delivers with probability KAPPA_DRAW
                        if xd is None:
                            xd = np.zeros(N, dtype=np.int64)
                        xd += can & (u["xd"][:, (3 * jj + q + 7 * t + 11 * r_) % 48] < KAPPA_DRAW)
                    if fej:                      # land ramp: search -> hand (1), search -> battlefield (2), hand -> battlefield (3)
                        if xf is None:
                            xf = np.zeros((4, N), dtype=np.int8)
                        xf[fej] += (can & (u["xf"][:, (3 * jj + 5 * t + 11 * r_) % 16] < KAPPA_FETCH)).view(np.int8)
                    can = (row > 0) & ok[jj] & (rem >= mvj)
                    if cp < 1.0:
                        can &= u["cs"][:, (5 * jj + 3 * t + 7 * (r_ + 1)) % 64] < cp
            spent_t = M - rem
            if 2 <= t <= 6:
                Hs = H[Kl:]
                tot = Hs.sum(0, dtype=np.int16)
                stuck = (Hs * ((~ok) & (mv_sp[:, None] <= M[None, :]))).sum(0, dtype=np.int16)
                share_sum += float((stuck / np.maximum(tot, 1)).mean())
            if add is not None:
                SP += add
            if xd is not None:
                xd = np.minimum(xd, 3)
                for e_ in range(int(xd.max())):
                    mm_ = np.nonzero(xd > e_)[0]
                    Hf[np.where(ptr[mm_] < W_LIB, lib[mm_, np.minimum(ptr[mm_], W_LIB - 1)], Kc), mm_] += 1
                    ptr[mm_] += 1
                xd_tot += xd
            if xf is not None:
                if fetch_cls >= 0:
                    H[fetch_cls] += xf[1]
                    SP[fetch_cls] += xf[2]                  # enters tapped: counts from the next turn on
                    # the fetched lands leave the library: close the gap at the next copy of that class not yet drawn
                    for e_ in range(int((xf[1] + xf[2]).max())):
                        r_ = np.nonzero((xf[1] + xf[2]) > e_)[0]
                        sub = lib[r_]
                        hit = (sub == fetch_cls) & (np.arange(W_LIB)[None, :] >= ptr[r_][:, None])
                        has = hit.any(1)
                        pos = np.where(has, hit.argmax(1), W_LIB)
                        sh = np.concatenate([sub[:, 1:], np.full((len(r_), 1), Kc, sub.dtype)], axis=1)
                        lib[r_] = np.where(np.arange(W_LIB)[None, :] >= pos[:, None], sh, sub)
                if xf[3].any():
                    i3 = np.nonzero(xf[3] > 0)[0]
                    LH3 = H[:Kl][:, i3]
                    s3 = np.where(LH3 > 0, tie + 1e-3 * LH3, -1e9)
                    k3 = s3.argmax(0)
                    o3 = (LH3 > 0).any(0)
                    H[k3[o3], i3[o3]] -= 1
                    SP[k3[o3], i3[o3]] += 1
        spent_tot += spent_t
        if t == 3:
            lands_t3 = SP[:Kl].sum(0)
        if detail:
            lands_turn.append(float(SP[:Kl].sum(0).mean()))
            mana_turn.append(float(spent_t.mean()))
            xdraw_turn.append(float(xd_tot.mean()))
            cre_turn.append(n_cre / N if Ksp else 0.0)
            oth_turn.append(n_oth / N if Ksp else 0.0)
    out["p_screw"] = float((lands_t3 < 3).mean())
    out["p_flood"] = float((lands16 >= 9).mean())
    out["p_color_screw"] = share_sum / 5.0
    out["mana_spent"] = float(spent_tot.mean() / 28.0)
    out["mull_rate"] = float(mull0.mean())
    if extra:
        out["lands7_kept"] = float(l_first[~mull0].mean()) if (~mull0).any() else 0.0
        out["lands7_all"] = float(l_first.mean())
    if detail:
        out.update(lands_by_turn=lands_turn, mana_by_turn=mana_turn, xdraw_by_turn=xdraw_turn, cre_by_turn=cre_turn, oth_by_turn=oth_turn,
                   hand7=(np.bincount(l_first, minlength=8)[:8] / N).tolist(), mull_n=[float((m >= 1).mean()), float((m >= 2).mean())])
    return out


def simulate(deck: dict[int, int], cards, n_trials: int = DEFAULT_TRIALS, extra: bool = False, version: int = 1,
             regime=None, detail: bool = False) -> dict:
    """Mana simulation. `version` 1 = the original simulator, 2 = the calibrated one (needs `regime`: random / hand / sealed)."""
    if int(version) == 1:
        return simulate_v1(deck, cards, n_trials, extra, detail)
    return simulate_v2(deck, cards, n_trials, extra, regime or "random", detail)


def class_signature(dc: DeckClasses) -> bytes:
    """Hashable identity of a deck's class table (decks with the same signature simulate identically)."""
    parts = [dc.counts, dc.mv, dc.is_land, dc.prod, dc.pips, dc.hyb]
    if dc.draw_n is not None:
        parts += [dc.draw_n, dc.fetch, dc.creature]
    return b"|".join(a.tobytes() for a in parts)
