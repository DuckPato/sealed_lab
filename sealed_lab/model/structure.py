"""Deck structure features phi(D): land count, curve, colours, plus the mana-simulator statistics.

`featurize(deck, cards)` is fast (a few ms, dominated by the simulator), deterministic for a deck multiset and cached.
Also provides the per-card linear feature matrix used by the deck builder and a parallel precompute for corpus decks.
"""
from __future__ import annotations

import logging
import os
import threading
from collections import OrderedDict
from typing import Callable, Sequence

import numpy as np

from sealed_lab.model.manasim import (DEFAULT_TRIALS, EXTRA_STATS, SIM_VERSION, STAT_NAMES, CardArrays, card_arrays,
                                      sim_label, simulate)
from sealed_lab.types import Cancelled

log = logging.getLogger(__name__)

STRUCT_NAMES = ["n_lands", "n_lands_dev2", "n_creatures", "n_noncreature", "curve_le1", "curve_2", "curve_3",
                "curve_4", "curve_5", "curve_ge6", "mean_spell_mv", "n_main_colors", "n_splash_cards", "n_splash_pips"]
MANA_NAMES = list(STAT_NAMES)  # p_screw, p_flood, p_color_screw, mana_spent, mull_rate
FEATURE_NAMES = STRUCT_NAMES + MANA_NAMES
N_FEATURES = len(FEATURE_NAMES)
N_STRUCT = len(STRUCT_NAMES)
TABLE_NAMES = FEATURE_NAMES + list(EXTRA_STATS)  # corpus phi table: features + simulator opening-hand statistics
N_TABLE = len(TABLE_NAMES)
LAND_PEAK = 17
MAIN_PIP_SHARE = 0.25

# linear (per-card additive) features usable by an optimiser: contribution of ONE copy of a card
LINEAR_NAMES = ["n_lands", "n_creatures", "n_noncreature", "curve_le1", "curve_2", "curve_3", "curve_4", "curve_5",
                "curve_ge6"]


def card_feature_matrix(cards) -> tuple[np.ndarray, list[str]]:
    """(n_cards, 9) matrix: what one copy of each card adds to the linear structure features (LINEAR_NAMES)."""
    ca = card_arrays(cards)
    M = np.zeros((ca.n, len(LINEAR_NAMES)))
    spell = ~ca.is_land
    M[:, 0] = ca.is_land
    M[:, 1] = spell & ca.is_creature
    M[:, 2] = spell & ~ca.is_creature
    mv = np.rint(ca.mv)
    bucket = np.clip(mv, 1, 6).astype(int)  # <=1 -> 1
    for b in range(1, 7):
        M[:, 2 + b] = spell & (bucket == b)
    return M, list(LINEAR_NAMES)


def _color_analysis(ids: np.ndarray, cnt: np.ndarray, ca: CardArrays):
    """Main / splash colours of a deck. Returns (main_mask[5], splash_mask[5], n_splash_cards, n_splash_pips,
    per-distinct-card boolean `needs_splash`)."""
    spell = ~ca.is_land[ids]
    pips = ca.pips[ids].astype(float)
    hyb = ca.hybrid[ids] & ((pips > 0).sum(1) > 1)
    wt = np.where(hyb, 0.5, 1.0)[:, None]
    pt = (pips * wt * (cnt * spell)[:, None]).sum(0)
    tot = pt.sum()
    main = np.zeros(5, dtype=bool)
    if tot > 0:
        order = np.argsort(-pt, kind="stable")
        for c in order[:2]:
            if pt[c] > 0:
                main[c] = True
        main |= pt >= MAIN_PIP_SHARE * tot
    present = pt > 0
    splash = present & ~main
    if splash.any():
        has_main_pip = (pips > 0)[:, main].any(1) if main.any() else np.zeros(len(ids), dtype=bool)
        needs = spell & (pips > 0)[:, splash].any(1) & ~(hyb & has_main_pip)
        # a colour is a splash only when a card needs it: not a colour whose only pips are hybrid / {2/X} / omen-face ones
        splash = splash & (pips[needs] > 0).any(0)
        n_cards = float((cnt * needs).sum())
        n_pips = float((pips[:, splash].sum(1) * cnt * needs).sum())
    else:
        needs = np.zeros(len(ids), dtype=bool)
        n_cards = n_pips = 0.0
    return main, splash, n_cards, n_pips, needs


def deck_colors(deck: dict[int, int], cards) -> tuple[str, str, set[int]]:
    """(main colours, splash colours, card ids needing a splash colour) of a deck."""
    ca = card_arrays(cards)
    ids = np.fromiter(deck.keys(), dtype=np.int64, count=len(deck))
    cnt = np.fromiter(deck.values(), dtype=float, count=len(deck))
    main, splash, _, _, needs = _color_analysis(ids, cnt, ca)
    s = lambda m: "".join(c for c, f in zip("WUBRG", m) if f)  # noqa: E731
    return s(main), s(splash), {int(i) for i, nd in zip(ids, needs) if nd}


def structure_counts(deck: dict[int, int], ca: CardArrays) -> np.ndarray:
    ids = np.fromiter(deck.keys(), dtype=np.int64, count=len(deck))
    cnt = np.fromiter(deck.values(), dtype=float, count=len(deck))
    land = ca.is_land[ids]
    spell = ~land
    cre = ca.is_creature[ids] & spell
    nl = float((cnt * land).sum())
    ns = float((cnt * spell).sum())
    mv = np.rint(ca.mv[ids])
    bucket = np.clip(mv, 1, 6).astype(int)
    curve = [float((cnt * (spell & (bucket == b))).sum()) for b in range(1, 7)]
    mean_mv = float((cnt * spell * ca.mv[ids]).sum() / ns) if ns > 0 else 0.0
    main, splash, n_sc, n_sp, _ = _color_analysis(ids, cnt, ca)
    out = np.empty(N_STRUCT)
    out[:] = [nl, (nl - LAND_PEAK) ** 2, float((cnt * cre).sum()), ns - float((cnt * cre).sum()), *curve, mean_mv,
              float(main.sum()), n_sc, n_sp]
    return out


_SIG_MEMO: "OrderedDict[int, tuple[object, str]]" = OrderedDict()


def content_signature(ca: CardArrays) -> str:
    """Hash of everything in a CardArrays that phi depends on (memoised per object; the memo keeps the object alive, so the id
    cannot be reused while it is cached). The phi cache is keyed by it: keying by `id(ca)` alone could serve a stale phi when a
    freed CardArrays' address is reused by a different card table."""
    import hashlib

    with _lock:
        hit = _SIG_MEMO.get(id(ca))
        if hit is not None and hit[0] is ca:
            return hit[1]
    h = hashlib.sha1(str(ca.n).encode())
    for a in (ca.is_land, ca.is_creature, ca.mv, ca.pips, ca.hybrid, ca.pbits, ca.draw_arr, ca.fetch_arr):
        h.update(np.ascontiguousarray(a).tobytes())
    sig = h.hexdigest()
    with _lock:
        _SIG_MEMO[id(ca)] = (ca, sig)
        while len(_SIG_MEMO) > 16:
            _SIG_MEMO.popitem(last=False)
    return sig


_CACHE: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
_CACHE_MAX = 50_000
_lock = threading.Lock()


def featurize_full(deck: dict[int, int], ca: CardArrays, n_trials: int = DEFAULT_TRIALS, version: int = 1,
                   regime: str | None = None) -> np.ndarray:
    """phi plus the extra simulator statistics (`TABLE_NAMES`), uncached."""
    out = np.empty(N_TABLE)
    out[:N_STRUCT] = structure_counts(deck, ca)
    sim = simulate(deck, ca, n_trials, extra=True, version=version, regime=regime)
    out[N_STRUCT:] = [sim[k] for k in MANA_NAMES + list(EXTRA_STATS)]
    return out


def featurize(deck: dict[int, int], cards, n_trials: int = DEFAULT_TRIALS, cache: bool = True, version: int = 1,
              regime: str | None = None) -> np.ndarray:
    """phi(D) (raw, un-standardised) with `FEATURE_NAMES`. Deterministic for a given deck multiset. `version` / `regime`
    select the mana simulator (v1, or v2 for the hand/draw regime of the format being evaluated, see manasim)."""
    ca = card_arrays(cards)
    key = None
    if cache:
        key = (content_signature(ca), n_trials, sim_label(version, regime), tuple(sorted((int(k), int(v)) for k, v in deck.items() if v > 0)))
        with _lock:
            hit = _CACHE.get(key)
            if hit is not None:
                _CACHE.move_to_end(key)
                return hit.copy()
    d = {int(k): int(v) for k, v in deck.items() if v > 0}
    out = np.empty(N_FEATURES)
    out[:N_STRUCT] = structure_counts(d, ca)
    sim = simulate(d, ca, n_trials, version=version, regime=regime)
    out[N_STRUCT:] = [sim[k] for k in MANA_NAMES]
    if cache:
        with _lock:
            _CACHE[key] = out.copy()
            if len(_CACHE) > _CACHE_MAX:
                for _ in range(_CACHE_MAX // 10):
                    _CACHE.popitem(last=False)
    return out


# ---------------------------------------------------------------------------------------------
# parallel precompute for many decks
# ---------------------------------------------------------------------------------------------

def phi_chunk_worker(task: dict) -> tuple[int, np.ndarray]:
    """Top-level worker: phi for a chunk of decks given as CSR-like arrays (offsets, card ids, counts)."""
    ca: CardArrays = task["ca"]
    off, cid, cn = task["offsets"], task["card_ids"], task["counts"]
    n_trials = task["n_trials"]
    m = len(off) - 1
    out = np.empty((m, N_TABLE))
    for i in range(m):
        a, b = off[i], off[i + 1]
        out[i] = featurize_full(dict(zip(cid[a:b].tolist(), cn[a:b].tolist())), ca, n_trials, task.get("version", 1), task.get("regime"))
    return task["start"], out


def compute_phi(offsets: np.ndarray, card_ids: np.ndarray, counts: np.ndarray, cards, n_trials: int = DEFAULT_TRIALS,
                n_jobs: int | None = None, progress: Callable[[float, str], None] | None = None,
                cancel: threading.Event | None = None, chunk: int = 400, version: int = 1,
                regime: str | None = None) -> np.ndarray:
    """phi table (`TABLE_NAMES`) for every deck described by CSR arrays. Parallel (spawn). Returns (n_decks, N_TABLE)."""
    from sealed_lab.model.glm import run_parallel

    ca = card_arrays(cards)
    n = len(offsets) - 1
    out = np.empty((n, N_TABLE))
    if n == 0:
        return out
    tasks = []
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        a, b = offsets[s], offsets[e]
        tasks.append(dict(ca=ca, offsets=offsets[s:e + 1] - a, card_ids=card_ids[a:b], counts=counts[a:b],
                          n_trials=n_trials, start=s, version=version, regime=regime))
    n_jobs = int(n_jobs or os.cpu_count() or 1)
    done = 0
    if n_jobs <= 1 or len(tasks) == 1:
        for t in tasks:
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            s, arr = phi_chunk_worker(t)
            out[s:s + len(arr)] = arr
            done += len(arr)
            if progress:
                progress(done / n, f"Mana simulation {done}/{n} decks")
        return out
    for s, arr in run_parallel(phi_chunk_worker, tasks, n_jobs, cancel=cancel):
        out[s:s + len(arr)] = arr
        done += len(arr)
        if progress:
            progress(done / n, f"Mana simulation {done}/{n} decks")
    return out


def cards_signature(cards, version: int = 1, regime: str | None = None) -> str:
    """Hash of everything about the card table that phi depends on (cache key). v1: unchanged from before the v2 simulator
    existed (so cached v1 tables stay valid); v2: also the simulator revision, the regime and the oracle-text effects."""
    import hashlib

    ca = card_arrays(cards)
    h = hashlib.sha1()
    h.update(str((SIM_VERSION, DEFAULT_TRIALS, LAND_PEAK, ca.n)).encode())
    for a in (ca.is_land, ca.is_creature, ca.mv, ca.pips, ca.hybrid, ca.pbits):
        h.update(np.ascontiguousarray(a).tobytes())
    if int(version) != 1:
        h.update(sim_label(version, regime).encode())
        for a in (ca.draw_arr, ca.fetch_arr):
            h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()
