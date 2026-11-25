"""Explain a deck: why is its win rate what it is, and which 1-for-1 swaps would change it.

Waterfall
---------
The model is additive in the logit:  L(D) = ref + sum_c beta_c n_c + theta . z(D)   with z the standardised structure
features (land count, curve, ..., mana simulation).  The *average real build* of the format (all 17lands builds of
that format, one per event x build) has  L_avg = ref + E[sum_c beta_c n_c] + theta . E[z].  The waterfall walks from
L_avg to L(D):

    cards      sum_c n_c (beta_c - b)           b = average beta per *spell* slot of the real builds
    land slots (N_spells(D) - E[N_spells]) * b   (a land swapped for a spell adds an average spell)  -> "Land count"
    nonbasic lands  sum beta n - E[sum beta n]
    structure  theta_g . (z_g(D) - E[z_g])      grouped: land count / creatures-spells / curve / colors & splash / mana sim

so that the steps add up exactly to L(D) - L_avg.  Percentage-point steps are sequential along the displayed path
(each step is measured from the win rate reached so far), so they add up to the total gap in win rate.

Swap analysis
-------------
For every nonbasic deck card the best replacement from the sideboard (spell for spell, land for land; basics are always
available as land replacements), for every sideboard card the best slot, both with the FULL model (mana simulation
included) for draw 0 and the whole set of bootstrap draws (P(gain > 0) = share of bootstrap replicates in which the swap
helps).  Candidate pairs are pre-ranked by the part of the model that is linear in the card counts and only the
promising ones are evaluated with the full model (their mana simulations are cached by card class).
"Value above replacement" of a deck card = q(deck) - max_s q(deck - card + s), in pp: positive = better than anything
the sideboard offers.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.special import expit

from sealed_lab import config

log = logging.getLogger(__name__)

# structure features -> readable groups (label, feature names); the mana simulation group lists its parts
STRUCT_GROUPS: list[tuple[str, list[str]]] = [
    ("Land count", ["n_lands", "n_lands_dev2"]),
    ("Creatures / spells", ["n_creatures", "n_noncreature"]),
    ("Curve", ["curve_le1", "curve_2", "curve_3", "curve_4", "curve_5", "curve_ge6", "mean_spell_mv"]),
    ("Colors & splash", ["n_main_colors", "n_splash_cards", "n_splash_pips"]),
    ("Mana simulation", ["p_screw", "p_flood", "p_color_screw", "mana_spent", "mull_rate"]),
    ("Castability", ["dead_cards"]),
]
FEATURE_LABELS = {
    "n_lands": "lands", "n_lands_dev2": "lands away from 17", "n_creatures": "creatures", "n_noncreature": "other spells",
    "curve_le1": "1-drops", "curve_2": "2-drops", "curve_3": "3-drops", "curve_4": "4-drops", "curve_5": "5-drops",
    "curve_ge6": "6+ drops", "mean_spell_mv": "mean mana value", "n_main_colors": "number of main colors",
    "n_splash_cards": "splash cards", "n_splash_pips": "splash pips", "p_screw": "mana screw", "p_flood": "flood",
    "p_color_screw": "color screw", "mana_spent": "mana spent", "mull_rate": "mulligans", "dead_cards": "cards the mana cannot cast",
}


# =============================================================================================
# data classes
# =============================================================================================

@dataclass
class WaterfallStep:
    label: str
    kind: str                     # "start" | "card" | "cards" | "lands" | "structure" | "end"
    d_logit: float                # change of the logit (0 for start / end)
    d_pp: float                   # change of the game win rate in pp, measured along the displayed path
    wr_after: float               # game win rate after the step
    card_id: int | None = None
    detail: list[tuple[str, float]] = field(default_factory=list)   # (part, d_logit) breakdown of grouped steps


@dataclass
class DeckExplanation:
    fmt: str
    start_wr: float               # average real build of the format
    end_wr: float                 # this deck
    steps: list[WaterfallStep]    # start, ..., end
    card_steps: list[tuple[int, str, int, float]]   # every nonbasic spell: (card_id, name, copies, d_logit vs average slot)
    n_builds: int = 0

    @property
    def total_pp(self) -> float:
        return 100.0 * (self.end_wr - self.start_wr)


@dataclass
class Swap:
    out_id: int
    in_id: int
    gain_pp: float                # game win rate after minus before, in pp
    p_better: float | None        # share of bootstrap replicates in which the swap helps (None without a bootstrap)
    gain_avg_pp: float | None = None   # opponent-specific analysis only: the same swap against the average opponent


@dataclass
class SwapAnalysis:
    base_wr: float
    deck_best: dict[int, Swap] = field(default_factory=dict)   # deck card -> its best replacement from the sideboard
    side_best: dict[int, Swap] = field(default_factory=dict)   # sideboard card -> its best slot in the deck
    var_pp: dict[int, float] = field(default_factory=dict)     # deck card -> value above replacement (pp)
    suggestions: list[Swap] = field(default_factory=list)      # best swaps with a positive gain, distinct cards
    complete: bool = True
    n_eval: int = 0
    seconds: float = 0.0


# =============================================================================================
# reference statistics (average real build)
# =============================================================================================

@dataclass
class RefStats:
    fmt: str
    n_builds: int
    z_mean: np.ndarray            # E[z] over the real builds (standardised features)
    card_mean: float              # E[sum_c beta_c n_c] over the nonbasic cards of the real builds
    spell_beta: float             # b: average beta per spell slot
    n_spells: float               # E[number of non-land cards]
    nbl_mean: float               # E[sum of beta n over nonbasic lands]
    start_logit: float
    source: str = ""              # "phi cache" | "sample"
    pair_mean: float = 0.0        # E[synergy term] over the real builds (0 without a synergy layer)
    arch_mean: float = 0.0        # E[archetype-conditional term]
    mull_mean: float = 0.0        # E[expected mulligan cost of the real builds] (0 for a model without the mulligan term)


def _stamp(model) -> list:
    out = []
    for p in (config.model_dir(model.set_code) / "meta.json", config.corpus_dir(model.set_code) / "builds.parquet"):
        try:
            out.append(p.stat().st_mtime_ns)
        except OSError:
            out.append(0)
    return out


def _cache_file(model, fmt: str):
    return config.cache_dir(model.set_code) / f"explain_ref_{fmt}.json"


def _from_json(d: dict, fmt: str) -> RefStats:
    return RefStats(fmt=fmt, n_builds=int(d["n_builds"]), z_mean=np.asarray(d["z_mean"], dtype=float), card_mean=float(d["card_mean"]),
                    spell_beta=float(d["spell_beta"]), n_spells=float(d["n_spells"]), nbl_mean=float(d["nbl_mean"]),
                    start_logit=float(d["start_logit"]), source=str(d.get("source", "")), pair_mean=float(d.get("pair_mean", 0.0)),
                    arch_mean=float(d.get("arch_mean", 0.0)), mull_mean=float(d.get("mull_mean", 0.0)))


def compute_ref_stats(model, fmt: str, max_sample: int = 400) -> RefStats:
    """Averages over the real builds of `fmt` (one per event x build_index) from the corpus tables."""
    import polars as pl

    from sealed_lab.model import structure

    cd = config.corpus_dir(model.set_code)
    if not (cd / "builds.parquet").exists() or not (cd / "deck_cards.parquet").exists():
        raise FileNotFoundError(f"the corpus of {model.set_code} is needed to explain decks (run Calculate)")
    b = (pl.read_parquet(cd / "builds.parquet", columns=["fmt", "draft_id", "build_index", "deck_key"])
         .filter(pl.col("fmt") == fmt).unique(subset=["draft_id", "build_index"], maintain_order=True))
    if b.height == 0:
        raise ValueError(f"no {fmt} builds in the corpus of {model.set_code}")
    keys, mult = np.unique(b["deck_key"].to_numpy().astype(np.int64), return_counts=True)
    w = mult / mult.sum()
    dc = pl.read_parquet(cd / "deck_cards.parquet", columns=["deck_key", "card_id", "n"]).filter(pl.col("deck_key").is_in(keys.tolist()))
    idx = np.searchsorted(keys, dc["deck_key"].to_numpy().astype(np.int64))
    cid = dc["card_id"].to_numpy().astype(np.int64)
    n = dc["n"].to_numpy().astype(float)
    remap = _corpus_to_model_ids(model, cd)
    if remap is not None:            # the corpus was rebuilt after the model (e.g. a cancelled Calculate): translate its card ids
        cid = np.where(cid < len(remap), remap[np.minimum(cid, len(remap) - 1)], -1)
        okm = cid >= 0
        idx, cid, n = idx[okm], cid[okm], n[okm]
    ca = model.card_arrays
    beta = model.card_beta(fmt)
    ok = cid < len(beta)
    idx, cid, n = idx[ok], cid[ok], n[ok]
    land = ca.is_land[cid]
    basic = ca.is_basic[cid]
    K = len(keys)
    n_eff, dead_deck = n, None
    if model.cast_applied:                 # castability deployed: spell copies count n x p
        got = _cast_from_cache(model, fmt, cd, keys, idx, cid)
        if got is None:
            return _compute_ref_stats_sampled(model, fmt, keys, idx, cid, n, w, max_sample)
        p_entry, dead_deck = got
        n_eff = np.where(land, n, n * p_entry)
    sp_sum = np.bincount(idx, weights=n_eff * beta[cid] * ~land, minlength=K)
    n_sp = np.bincount(idx, weights=n * ~land, minlength=K)
    nbl_sum = np.bincount(idx, weights=n * beta[cid] * (land & ~basic), minlength=K)
    S_sp, N_sp, S_nbl = float(w @ sp_sum), float(w @ n_sp), float(w @ nbl_sum)
    b_spell = S_sp / N_sp if N_sp > 0 else 0.0

    # standardised structure features of the real builds: phi cache of the model run, else a weighted sample
    z_mean = None
    source = ""
    try:
        decks = pl.read_parquet(cd / "decks.parquet", columns=["deck_key", "deck_hash"])
        hashes = decks.sort("deck_key")["deck_hash"].to_numpy()
        dkeys = decks.sort("deck_key")["deck_key"].to_numpy().astype(np.int64)
        from sealed_lab.model import pipeline

        sv, sr = model.sim_spec(fmt)
        p = pipeline.phi_cache_path(config.cache_dir(model.set_code), sv, sr)
        if p.exists():
            with np.load(p, allow_pickle=False) as z:
                same = str(z["sig"]) == structure.cards_signature(model.card_arrays, sv, sr)
                ch, cp = (z["hashes"], z["phi"]) if same else (None, None)
            if ch is not None:
                pos = np.searchsorted(dkeys, keys)
                pos = np.minimum(pos, len(dkeys) - 1)
                h = hashes[pos]
                order = np.argsort(ch)
                q = np.minimum(np.searchsorted(ch[order], h), len(ch) - 1)
                hit = (ch[order][q] == h) & (dkeys[pos] == keys)
                if hit.mean() >= 0.98:
                    phi = cp[order][q[hit]][:, :structure.N_FEATURES]
                    if dead_deck is not None:
                        phi = np.column_stack([phi, dead_deck[hit]])
                    ww = w[hit] / w[hit].sum()
                    z_mean = ww @ ((phi - model.mu) / model.sd)
                    source = "phi cache"
    except Exception:  # noqa: BLE001 - fall back to the sample
        log.info("phi cache not usable for the explanation reference", exc_info=True)
    if z_mean is None:
        rng = np.random.default_rng(0)
        pick = np.unique(rng.choice(K, size=min(max_sample, K), p=w))
        order = np.argsort(idx, kind="stable")
        sidx, scid, sn = idx[order], cid[order], n[order]
        zs = []
        for k in pick:
            lo, hi = np.searchsorted(sidx, k), np.searchsorted(sidx, k, side="right")
            zs.append(model.standardise(model.featurize({int(c): int(v) for c, v in zip(scid[lo:hi], sn[lo:hi])}, fmt)))
        z_mean = np.mean(zs, axis=0)
        source = f"sample of {len(pick)} builds"
    theta = model._draws[fmt]["theta"][0]
    ref0 = float(model._draws[fmt]["ref"][0])
    pair_mean, arch_mean = _extra_means(model, fmt, idx, cid, n, w, K)
    mull_mean = _mull_mean(model, fmt, idx, cid, n, w, K)
    start = ref0 + S_sp + S_nbl + float(theta @ z_mean) + pair_mean + arch_mean + mull_mean
    return RefStats(fmt=fmt, n_builds=int(mult.sum()), z_mean=np.asarray(z_mean, dtype=float), card_mean=S_sp + S_nbl,
                    spell_beta=b_spell, n_spells=N_sp, nbl_mean=S_nbl, start_logit=start, source=source, pair_mean=pair_mean,
                    arch_mean=arch_mean, mull_mean=mull_mean)



def _corpus_to_model_ids(model, cd) -> "np.ndarray | None":
    """corpus card_id -> the model's card_id (-1 = not in the model), or None when both tables number the cards identically.

    The model's own card table is the truth for ids; a corpus rebuilt after the model (cancelled / failed Calculate) can number
    the cards differently, and indexing the model's arrays with its ids would silently value the wrong cards."""
    import polars as pl

    try:
        cc = pl.read_parquet(cd / "cards.parquet", columns=["card_id", "name"]).sort("card_id")
    except Exception:  # noqa: BLE001
        return None
    mnames = model.cards.sort("card_id")["name"].to_list()
    cnames = cc["name"].to_list()
    if cnames == mnames:
        return None
    log.warning("the corpus card table of %s differs from the model's; translating card ids by name", model.set_code)
    pos = {nm: i for i, nm in enumerate(mnames)}
    arr = np.full(int(cc["card_id"].max()) + 1, -1, dtype=np.int64)
    for i, nm in zip(cc["card_id"].to_list(), cnames):
        arr[int(i)] = pos.get(nm, -1)
    return arr


def _cast_from_cache(model, fmt: str, cd, keys: np.ndarray, idx: np.ndarray, cid: np.ndarray):
    """(castability per (build deck, card) entry, dead cards per build deck) from the model run's castability cache, or None.

    `idx` indexes `keys` (the build decks); entries are in any order."""
    import polars as pl

    from sealed_lab.model import castability

    try:
        regime = model.cast_regime(fmt)
        path = castability.cache_path(config.cache_dir(model.set_code), regime)
        if not path.exists():
            return None
        decks = pl.read_parquet(cd / "decks.parquet", columns=["deck_key", "deck_hash"]).sort("deck_key")
        hashes = decks["deck_hash"].to_numpy()
        dkeys = decks["deck_key"].to_numpy().astype(np.int64)
        with np.load(path, allow_pickle=False) as z:
            if str(z["sig"]) != castability.cast_signature(model.cast_cards, regime, model.cast_trials):
                return None
            ch, cl, cd_, cp = z["hashes"], z["lens"], z["dead"], z["p"]
        pos = np.minimum(np.searchsorted(dkeys, keys), len(dkeys) - 1)
        h = hashes[pos]
        order = np.argsort(ch)
        q = np.minimum(np.searchsorted(ch[order], h), len(ch) - 1)
        src = order[q]
        if not np.all((ch[src] == h) & (dkeys[pos] == keys)):
            return None
        cstart = np.concatenate([[0], np.cumsum(cl)])
        o = np.lexsort((cid, idx))                                  # entries sorted by (deck, card id) = the cache's order
        si = idx[o]
        rank = np.arange(len(si)) - np.searchsorted(si, si)
        if np.any(rank >= cl[src[si]]):
            return None
        p = np.empty(len(cid))
        p[o] = cp[cstart[src[si]] + rank]
        return p, cd_[src]
    except Exception:  # noqa: BLE001
        log.info("castability cache not usable for the explanation reference", exc_info=True)
        return None


def _compute_ref_stats_sampled(model, fmt: str, keys, idx, cid, n, w, max_sample: int) -> RefStats:
    """Reference statistics from a weighted sample of the real builds, for models whose castability cache is missing."""
    ca = model.card_arrays
    beta = model.card_beta(fmt)
    rng = np.random.default_rng(0)
    K = len(keys)
    pick = np.unique(rng.choice(K, size=min(max_sample, K), p=w))
    order = np.argsort(idx, kind="stable")
    sidx, scid, sn = idx[order], cid[order], n[order]
    zs, sp_s, nsp_s, nbl_s, mu_s = [], [], [], [], []
    for k in pick:
        lo, hi = np.searchsorted(sidx, k), np.searchsorted(sidx, k, side="right")
        d = {int(c): int(v) for c, v in zip(scid[lo:hi], sn[lo:hi])}
        ids, cnt, we = model.deck_weights(d, fmt)
        land, basic = ca.is_land[ids], ca.is_basic[ids]
        sp_s.append(float((we * beta[ids] * ~land).sum()))
        nsp_s.append(float((cnt * ~land).sum()))
        nbl_s.append(float((cnt * beta[ids] * (land & ~basic)).sum()))
        zs.append(model.standardise(model.featurize(d, fmt)))
        mu_s.append(float(model.mull_cost(d, fmt)[0]))
    S_sp, N_sp, S_nbl = float(np.mean(sp_s)), float(np.mean(nsp_s)), float(np.mean(nbl_s))
    z_mean = np.mean(zs, axis=0)
    theta = model._draws[fmt]["theta"][0]
    ref0 = float(model._draws[fmt]["ref"][0])
    mull_mean = float(np.mean(mu_s))
    start = ref0 + S_sp + S_nbl + float(theta @ z_mean) + mull_mean
    return RefStats(fmt=fmt, n_builds=int(len(pick)), z_mean=np.asarray(z_mean, dtype=float), card_mean=S_sp + S_nbl,
                    spell_beta=S_sp / N_sp if N_sp > 0 else 0.0, n_spells=N_sp, nbl_mean=S_nbl, start_logit=start,
                    source=f"sample of {len(pick)} builds (no castability cache)", mull_mean=mull_mean)


def _mull_mean(model, fmt: str, idx: np.ndarray, cid: np.ndarray, n: np.ndarray, w: np.ndarray, K: int) -> float:
    """Mean over the real builds (weights `w`) of the expected mulligan cost (logit) of the deck itself (draw 0)."""
    if not model.has_mull(fmt):
        return 0.0
    land = model.card_arrays.is_land[cid]
    nl = np.bincount(idx, weights=n * land, minlength=K)
    nc = np.bincount(idx, weights=n, minlength=K)
    m = model._draws[fmt]["mull"][0]
    cost = np.empty(K)
    for k in range(K):
        p1, p2 = model.mull_probs(int(round(nl[k])), int(round(nc[k])), fmt)
        cost[k] = m[0] * p1 + m[1] * p2
    return float(w @ cost)


def _extra_means(model, fmt: str, idx: np.ndarray, cid: np.ndarray, n: np.ndarray, w: np.ndarray, K: int) -> tuple[float, float]:
    """Mean over the real builds (weights `w`) of the pair-synergy and the archetype-conditional terms."""
    import scipy.sparse as sp

    from sealed_lab.model import design

    dr = model._draws[fmt]
    if "pair" not in dr and "arch" not in dr:
        return 0.0, 0.0
    V = sp.csr_matrix((n, (idx, cid)), shape=(K, model.card_arrays.n))
    pm = am = 0.0
    if "pair" in dr and model.pairs is not None:
        M = design.pair_columns(V, model.pairs["ab"])
        pm = float(w @ (M @ np.asarray(dr["pair"][0], dtype=float)))
    if "arch" in dr and model.archcols is not None and model.archetypes is not None:
        a, _ = model.archetypes.assign_matrix(V.multiply(model._spell[None, :]).tocsr())
        ca = model.archcols["ca"]
        coef = np.asarray(dr["arch"][0], dtype=float)
        term = np.zeros(K)
        for j in range(len(ca)):
            m = a == ca[j, 1]
            term[m] += coef[j] * V[:, int(ca[j, 0])].toarray().ravel()[m]
        am = float(w @ term)
    return pm, am


_REF_LOCK = threading.Lock()


def ref_stats(model, fmt: str) -> RefStats:
    """`compute_ref_stats`, cached in memory and in `cache/explain_ref_<fmt>.json` (invalidated when the model changes)."""
    memo = getattr(model, "_explain_ref", None)
    if memo is None:
        memo = model._explain_ref = {}                   # type: ignore[attr-defined]
    with _REF_LOCK:
        if fmt in memo:
            return memo[fmt]
        stamp = _stamp(model)
        f = _cache_file(model, fmt)
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            if d.get("stamp") == stamp and d.get("version") == 3:
                memo[fmt] = _from_json(d, fmt)
                return memo[fmt]
        except (OSError, ValueError, KeyError):
            pass
        st = compute_ref_stats(model, fmt)
        memo[fmt] = st
        try:
            f.parent.mkdir(parents=True, exist_ok=True)
            d = dict(version=3, stamp=stamp, n_builds=st.n_builds, z_mean=st.z_mean.tolist(), card_mean=st.card_mean,
                     spell_beta=st.spell_beta, n_spells=st.n_spells, nbl_mean=st.nbl_mean, start_logit=st.start_logit,
                     source=st.source, pair_mean=st.pair_mean, arch_mean=st.arch_mean, mull_mean=st.mull_mean)
            tmp = f.with_suffix(".tmp")
            tmp.write_text(json.dumps(d), encoding="utf-8")
            os.replace(tmp, f)
        except OSError:
            log.info("could not write the explanation reference cache")
        return st


# =============================================================================================
# waterfall
# =============================================================================================

def _pair_parts(model, fmt: str, deck: dict[int, int], n: int = 8) -> list[tuple[str, float]]:
    """Largest contributions (logit) of single card pairs to the deck's synergy term."""
    coef = model._draws[fmt]["pair"][0]
    ab = model.pairs["ab"]
    cf = np.zeros(model.card_arrays.n)
    for c, v in deck.items():
        if 0 <= c < len(cf):
            cf[c] = v
    vals = np.minimum(cf[ab[:, 0]], cf[ab[:, 1]]) * coef
    order = np.argsort(-np.abs(vals))[:n]
    return [(f"{model.card_name(int(ab[j, 0]))} + {model.card_name(int(ab[j, 1]))}", float(vals[j])) for j in order if vals[j] != 0.0]


def explain_deck(model, deck: dict[int, int], fmt: str, top_cards: int = 8) -> DeckExplanation:
    """Waterfall from the average real build to `deck` (draw 0; steps add up exactly to the logit gap)."""
    deck = {int(k): int(v) for k, v in deck.items() if v > 0}
    st = ref_stats(model, fmt)
    ref, cards, struct, mana, phi, ids, cnt = model._parts(deck, fmt, False)
    beta = model._draws[fmt]["beta"][0]
    theta = model._draws[fmt]["theta"][0]
    z = model.standardise(phi)
    ca = model.card_arrays
    pt = at = 0.0
    if model.has_extra(fmt):
        p_, a_ = model._extra_terms(deck, fmt, False)
        pt, at = float(p_[0]), float(a_[0])
    mull = float(model.mull_cost(deck, fmt)[0])
    L_deck = float(ref[0] + cards[0] + struct[0] + mana[0]) + pt + at + mull

    card_rows: list[tuple[int, str, int, float]] = []
    d_nbl = 0.0
    n_sp = 0
    _, _, w_eff = model.deck_weights(deck, fmt)                 # copies x castability (the copies without castability)
    for c, n, we in zip(ids.tolist(), cnt.tolist(), w_eff.tolist()):
        n = int(n)
        if ca.is_land[c]:
            if not ca.is_basic[c]:
                d_nbl += n * float(beta[c])
            continue
        n_sp += n
        card_rows.append((c, model.card_name(c), n, we * float(beta[c]) - n * st.spell_beta))
    d_slot = (n_sp - st.n_spells) * st.spell_beta
    contrib = theta * (z - st.z_mean)                          # per feature
    names = list(model.feature_names)
    fi = {nm: i for i, nm in enumerate(names)}

    # ---- steps in display order
    card_rows.sort(key=lambda r: -abs(r[3]))
    top = sorted(card_rows[:top_cards], key=lambda r: -r[3])
    rest = card_rows[top_cards:]
    raw: list[tuple[str, str, float, int | None, list[tuple[str, float]]]] = []
    for c, nm, n, d in top:
        raw.append((nm + (f" ×{n}" if n > 1 else ""), "card", d, c, []))
    if rest:
        raw.append((f"{len(rest)} other card{'s' if len(rest) != 1 else ''}", "cards", sum(r[3] for r in rest), None,
                    [(r[1], r[3]) for r in sorted(rest, key=lambda r: -abs(r[3]))[:12]]))
    d_nbl_rel = d_nbl - st.nbl_mean
    if abs(d_nbl_rel) > 1e-12 or d_nbl != 0.0:
        raw.append(("Nonbasic lands", "lands", d_nbl_rel, None, []))
    if "pair" in model._draws[fmt]:
        raw.append(("Synergies", "synergy", pt - st.pair_mean, None, _pair_parts(model, fmt, deck)))
    if "arch" in model._draws[fmt]:
        raw.append(("Archetype fit", "synergy", at - st.arch_mean, None, []))
    for label, feats in STRUCT_GROUPS:
        parts = [(FEATURE_LABELS.get(f, f), float(contrib[fi[f]])) for f in feats if f in fi]
        d = sum(v for _, v in parts)
        if label == "Land count":
            parts.append(("spell slots (land swapped for an average spell)", float(d_slot)))
            d += d_slot
        raw.append((label, "structure", d, None, parts))
    if model.has_mull(fmt):
        raw.append(("Mulligans", "structure", mull - st.mull_mean,
                    None, [("expected mulligan cost of this deck's land count, vs the average real build", mull - st.mull_mean)]))

    steps = [WaterfallStep("Average build", "start", 0.0, 0.0, float(expit(st.start_logit)))]
    L = st.start_logit
    for label, kind, d, c, detail in raw:
        nxt = L + d
        steps.append(WaterfallStep(label, kind, float(d), float(100.0 * (expit(nxt) - expit(L))), float(expit(nxt)), c, detail))
        L = nxt
    if abs(L - L_deck) > 1e-6:                                  # cannot happen unless the reference is inconsistent
        log.warning("explanation does not add up: %.6f vs %.6f", L, L_deck)
    steps.append(WaterfallStep("This deck", "end", 0.0, 0.0, float(expit(L_deck))))
    return DeckExplanation(fmt=fmt, start_wr=float(expit(st.start_logit)), end_wr=float(expit(L_deck)), steps=steps,
                           card_steps=card_rows, n_builds=st.n_builds)


# =============================================================================================
# swaps
# =============================================================================================

_CTX_LOCK = threading.RLock()
_CTXS: dict[tuple[int, str], tuple[object, object]] = {}


def _ctx_for(model, fmt: str):
    from sealed_lab.deck.builder import Ctx

    key = (id(model), fmt)
    hit = _CTXS.get(key)
    if hit is not None and hit[0] is model:
        return hit[1]
    if len(_CTXS) > 4:
        _CTXS.clear()
    ctx = Ctx(model, fmt)
    _CTXS[key] = (model, ctx)
    return ctx


def _reduce(base_wr: float, pairs: dict[tuple[int, int], tuple], locked: dict[int, int],
            excluded: set[int], n_suggest: int, complete: bool, n_eval: int, t0: float,
            is_basic: np.ndarray | None = None) -> SwapAnalysis:
    out = SwapAnalysis(base_wr=base_wr, complete=complete, n_eval=n_eval, seconds=time.monotonic() - t0)
    for (o, i), (g, p, *rest) in pairs.items():
        s = Swap(o, i, g, p, rest[0] if rest else None)
        b = out.deck_best.get(o)
        if b is None or g > b.gain_pp:
            out.deck_best[o] = s
        b = out.side_best.get(i)
        if (is_basic is None or not is_basic[i]) and (b is None or g > b.gain_pp):
            out.side_best[i] = s
    out.var_pp = {o: -s.gain_pp for o, s in out.deck_best.items()}
    used_o: set[int] = set()
    used_i: set[int] = set()
    for (o, i), (g, p, *rest) in sorted(pairs.items(), key=lambda kv: -kv[1][0]):
        if g <= 1e-4 or len(out.suggestions) >= n_suggest:
            break
        if o in used_o or i in used_i or i in excluded or (o in locked and locked[o] >= 1):
            continue
        used_o.add(o)
        used_i.add(i)
        out.suggestions.append(Swap(o, i, g, p, rest[0] if rest else None))
    return out


def swap_analysis(model, deck: dict[int, int], pool: dict[int, int], fmt: str, locked: dict[int, int] | None = None,
                  excluded: set[int] | None = None, cancel: threading.Event | None = None,
                  on_partial: Callable[[SwapAnalysis], None] | None = None, n_suggest: int = 3, k_out: int = 4,
                  k_in: int = 3, full_below: int = 160, partial_every: float = 0.5, opp: str | None = None,
                  on_play: bool | None = None) -> SwapAnalysis:
    """Best 1-for-1 swaps between `deck` and the rest of `pool` (see the module docstring).

    With `opp` (the opponent's colors, e.g. "UB") and a deployed matchup layer every card is valued against that opponent
    (its centred card x opponent effect added to the card effect, draw by draw), so base_wr and the gains are win rates
    against that strategy.  With `on_play` (True / False) and a deployed play/draw layer every card also gets its play/draw shift
    (+gamma/2 on the play, -gamma/2 on the draw), so the gains are win rates in a game on the play / on the draw; `Swap.gain_avg_pp`
    is then the same swap without the opponent / play-draw shifts.  Raises `Cancelled` when `cancel` is set. `on_partial` receives intermediate results."""
    from sealed_lab.types import Cancelled

    t0 = time.monotonic()
    locked = {int(k): int(v) for k, v in (locked or {}).items() if v > 0}
    excluded = {int(c) for c in (excluded or ())}
    with _CTX_LOCK:
        ctx = _ctx_for(model, fmt)
        ca = ctx.ca
        deck = {int(k): int(v) for k, v in deck.items() if v > 0}
        side = {int(c): int(n) - deck.get(int(c), 0) for c, n in pool.items()
                if not ca.is_basic[int(c)] and int(n) - deck.get(int(c), 0) > 0}
        dr = model._draws[fmt]
        Bb, Bt = dr["beta"].astype(float), dr["theta"].astype(float)
        mu, sd = ctx.mu, ctx.sd
        phi0 = ctx.phi(deck)
        z0 = (phi0 - mu) / sd
        L0 = ctx.logit(deck)
        cast_on = ctx.cast_applied
        w0i, w0 = ctx.weights(deck)                                # effective copies (copies x castability with castability)
        E0 = None
        E = None
        if opp is not None and getattr(model, "opp", None) is not None and fmt in getattr(model, "opp_formats", [fmt]):
            E = model.opp.effect(opp, draws=True)                  # (B, n_cards) centred card x opponent effects
        if on_play is not None and getattr(model, "playdraw", None) is not None:
            Ep = model.playdraw_effect(bool(on_play), draws=True)  # (B, n_cards) card x play/draw shifts (+-gamma/2)
            E = Ep if E is None else E + Ep
        if E is not None:
            Bb = Bb + E
            E0 = E[0]
            L0 = L0 + float(E0[w0i] @ w0)
        base = float(expit(L0))
        nl = sum(n for c, n in deck.items() if ca.is_land[c])
        cf0 = ctx.copies_vector(deck)
        Pd = dr.get("pair")
        pairs_of: dict[int, np.ndarray] = {}
        if Pd is not None and ctx.pair_a is not None:
            for cc in np.unique(np.concatenate([ctx.pair_a, ctx.pair_b])):
                pairs_of[int(cc)] = np.nonzero((ctx.pair_a == cc) | (ctx.pair_b == cc))[0]
        at0 = model._extra_terms(deck, fmt, True)[1] if ctx.has_arch else None
        m0 = model.mull_cost(deck, fmt, True)                      # the deck's own expected mulligan cost, per draw

        # sources per color in the deck (every card that produces mana counts)
        src = np.zeros(5)
        for c, n in deck.items():
            src += n * ca.produces[c]
        pips_in_deck = np.zeros(5)
        for c, n in deck.items():
            if not ca.is_land[c]:
                pips_in_deck += n * (ca.pips[c] > 0)

        def castable(c: int) -> bool:
            need = ca.pips[c] > 0
            if not need.any():
                return True
            ok = src >= 2
            return bool(ok[need].any()) if ca.hybrid[c] else bool(ok[need].all())

        out_sp = [c for c in deck if not ca.is_land[c]]
        in_sp = [c for c, n in side.items() if not ca.is_land[c] and c not in excluded and castable(c)]
        out_land = [c for c in deck if ca.is_land[c]]                 # nonbasic and basic
        basic_cols = [col for col, bid in ctx.basic_id.items() if (src[col] > 0 or pips_in_deck[col] > 0)]
        basics_in = [ctx.basic_id[col] for col in basic_cols]
        in_land = [c for c, n in side.items() if ca.is_land[c] and c not in excluded] + basics_in

        pairs_todo: list[tuple[int, int]] = []
        # lands: all combinations except basic <-> basic
        for o in out_land:
            for i in in_land:
                if o == i or (ca.is_basic[o] and ca.is_basic[i]):
                    continue
                pairs_todo.append((o, i))
        # spells: rank by the linear part of the model, keep the most promising pairs
        if out_sp and in_sp:
            lo_ids = np.array(out_sp, dtype=np.int64)
            li_ids = np.array(in_sp, dtype=np.int64)
            lin_o = ctx.lin_spell(lo_ids, nl)
            lin_i = ctx.lin_spell(li_ids, nl)
            if ctx.P is not None:
                lin_o = lin_o - ctx.pair_delta(lo_ids, cf0, -1)
                lin_i = lin_i + ctx.pair_delta(li_ids, cf0, +1)
            if E0 is not None:
                lin_o = lin_o + E0[lo_ids]
                lin_i = lin_i + E0[li_ids]
            D = lin_i[None, :] - lin_o[:, None]                    # (out, in)
            same = lo_ids[:, None] == li_ids[None, :]
            D = np.where(same, -np.inf, D)
            keep = np.zeros(D.shape, dtype=bool)
            if D.size <= full_below:
                keep[:] = True
            else:
                for a in range(D.shape[0]):
                    keep[a, np.argsort(-D[a])[:k_out]] = True
                for b in range(D.shape[1]):
                    keep[np.argsort(-D[:, b])[:k_in], b] = True
            keep &= ~same
            sel = np.argwhere(keep)
            sel = sel[np.argsort(-D[sel[:, 0], sel[:, 1]], kind="stable")]
            pairs_todo += [(int(lo_ids[a]), int(li_ids[b])) for a, b in sel]

        results: dict[tuple[int, int], tuple[float, float | None]] = {}
        n_eval = 0
        last_emit = time.monotonic()
        has_boot = Bb.shape[0] > 1
        for o, i in pairs_todo:
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            nd = dict(deck)
            nd[o] -= 1
            if nd[o] <= 0:
                del nd[o]
            nd[i] = nd.get(i, 0) + 1
            z1 = (ctx.phi(nd) - mu) / sd
            if cast_on:                                            # every card's castability changes with the swap
                w1i, w1 = ctx.weights(nd)
                dL = Bb[:, w1i] @ w1 - Bb[:, w0i] @ w0 + Bt @ (z1 - z0)
            else:
                dL = (Bb[:, i] - Bb[:, o]) + Bt @ (z1 - z0)        # per draw (row 0 = full-data fit)
            if pairs_of:
                ix = np.union1d(pairs_of.get(o, np.zeros(0, int)), pairs_of.get(i, np.zeros(0, int)))
                if len(ix):
                    cf1 = cf0.copy()
                    cf1[o] -= 1
                    cf1[i] += 1
                    a_, b_ = ctx.pair_a[ix], ctx.pair_b[ix]
                    dv = np.minimum(cf1[a_], cf1[b_]) - np.minimum(cf0[a_], cf0[b_])
                    dL = dL + Pd[:, ix] @ dv
            if at0 is not None:
                dL = dL + model._extra_terms(nd, fmt, True)[1] - at0
            if ctx.mull_c is not None:
                dL = dL + model.mull_cost(nd, fmt, True) - m0          # a land <-> spell swap changes the mulligan rate
            gain = 100.0 * float(expit(L0 + dL[0]) - base)
            p = float(np.mean(dL[1:] > 0)) if has_boot else None
            if E0 is not None:
                L_avg = float(ctx.logit(deck))
                dE = float(E0[w1i] @ w1 - E0[w0i] @ w0) if cast_on else float(E0[i] - E0[o])
                gavg = 100.0 * float(expit(L_avg + dL[0] - dE) - expit(L_avg))
                results[(o, i)] = (gain, p, gavg)
            else:
                results[(o, i)] = (gain, p)
            n_eval += 1
            now = time.monotonic()
            if on_partial is not None and now - last_emit >= partial_every:
                last_emit = now
                on_partial(_reduce(base, results, locked, excluded, n_suggest, False, n_eval, t0, ca.is_basic))
        return _reduce(base, results, locked, excluded, n_suggest, True, n_eval, t0, ca.is_basic)


def swap_label(model, s: Swap) -> str:
    return f"{model.card_name(s.out_id)} → {model.card_name(s.in_id)}"
