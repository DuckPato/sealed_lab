"""Does the mana simulator look like real play? v1 and v2 side by side against the replay data, per event and land count.

For every registered replay event of a set (and every land count with enough games) the real games are compared with the
simulations of the very same decks: mulligan rate, the land distribution of the first 7-card hand, P(< 3 lands in play after
own turn 3), lands in play after own turns 4 and 5, mana spent in own turns 1-4 and 1-7. Real values are re-weighted to
50/50 play/draw (the simulator averages both); intervals are cluster bootstraps over events (draft_id). Conditioning: the
replay files only describe completed own turns, so every turn-k number is conditional on the game lasting k turns.

Without replay data the report falls back to `validate.simulator_validation` (mulligan rate and lands in kept hands from the
corpus), which now carries the v2 columns too.
"""
from __future__ import annotations

import concurrent.futures as cf
import logging
import multiprocessing
import os
import threading
from typing import Callable

import numpy as np
import polars as pl

from sealed_lab import config
from sealed_lab.model import manasim
from sealed_lab.model.replaystats import _play_draw_weights, compare
from sealed_lab.types import Cancelled

log = logging.getLogger(__name__)
Progress = Callable[[float, str], None]

LAND_BINS = (15, 16, 17, 18)
MIN_BIN_GAMES = 80
EVENT_ORDER = ["TradSealed", "Sealed", "TradDraft", "PremierDraft", "QuickDraft", "PickTwoDraft", "PickTwoTradDraft"]
SEED = 20261025
N_TRIALS = 3000
METRICS = [("mull", "Mulligan rate"), ("hand_mean", "Lands in the first 7 cards"), ("hand_tail", "First 7 cards outside 2-4 lands"),
           ("screw3", "P(< 3 lands in play after own turn 3)"), ("lands4", "Lands in play after own turn 4"),
           ("lands5", "Lands in play after own turn 5"), ("mana4", "Mana spent, own turns 1-4"), ("mana7", "Mana spent, own turns 1-7")]


def _check(cancel: threading.Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise Cancelled()


def _worker(task: dict) -> list[dict]:
    ca = task["ca"]
    out = []
    for d in task["decks"]:
        row = {}
        for tag, (version, regime) in task["configs"].items():
            r = manasim.simulate(d, ca, N_TRIALS, version=version, regime=regime, detail=True)
            row[tag] = dict(mull=r["mull_n"][0], hand7=r["hand7"], screw3=r["p_screw"], lands=r["lands_by_turn"], mana=r["mana_by_turn"])
        out.append(row)
    return out


def simulate_decks(decks: list[dict[int, int]], cards, configs: dict[str, tuple[int, str | None]], n_jobs: int | None,
                   cancel: threading.Event | None = None, chunk: int = 40) -> list[dict]:
    """Per deck, per configuration tag: the detailed simulator output (parallel, spawn)."""
    ca = manasim.card_arrays(cards)
    tasks = [dict(ca=ca, decks=decks[i:i + chunk], configs=configs) for i in range(0, len(decks), chunk)]
    n_jobs = int(n_jobs or os.cpu_count() or 1)
    if n_jobs <= 1 or len(tasks) <= 1:
        res = []
        for t in tasks:
            _check(cancel)
            res.append(_worker(t))
        return [r for part in res for r in part]
    from sealed_lab.model.glm import map_parallel

    res = map_parallel(_worker, tasks, n_jobs, cancel=cancel)
    return [r for part in res for r in part]


def _tvd_ci(real: np.ndarray, w: np.ndarray, sim_pmf: np.ndarray, groups: np.ndarray, n_boot: int = 120, seed: int = SEED) -> dict:
    """Total variation distance between the (weighted) real distribution of lands in the first hand and a simulated pmf."""
    def tv(idx):
        p = np.bincount(real[idx], weights=w[idx], minlength=8)[:8]
        p = p / max(p.sum(), 1e-12)
        return float(0.5 * np.abs(p - sim_pmf).sum())
    est = tv(np.arange(len(real)))
    u, inv = np.unique(groups, return_inverse=True)
    G = len(u)
    order = np.argsort(inv, kind="stable")
    starts = np.searchsorted(inv[order], np.arange(G + 1))
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(n_boot):
        pick = rng.integers(0, G, G)
        idx = np.concatenate([order[starts[g]:starts[g + 1]] for g in pick])
        reps.append(tv(idx))
    return dict(est=est, lo=float(np.percentile(reps, 2.5)), hi=float(np.percentile(reps, 97.5)))


def realism(set_code: str, n_jobs: int | None = None, max_games_per_bin: int = 900, cancel: threading.Event | None = None,
            progress: Progress | None = None, regimes: dict | None = None) -> dict:
    """Real versus simulated (v1 and v2) statistics per replay event and land count. See the module docstring."""
    from sealed_lab.ingest.corpus import load_corpus
    from sealed_lab.model import simcalib

    from sealed_lab.ingest import replay

    set_code = set_code.upper()
    if not replay.replay_files(set_code):
        return dict(source="none", rows=[], note="no replay data for this set")
    replay.ingest_replays(set_code, progress=(lambda f, m: progress(0.3 * f, m)) if progress else None, cancel=cancel, n_jobs=n_jobs)
    try:
        h = simcalib.hands_frame(set_code, cancel=cancel)
    except FileNotFoundError:
        return dict(source="none", rows=[], note="no replay data for this set")
    if h.is_empty():
        return dict(source="none", rows=[], note="no replay games could be linked to the corpus")
    c = load_corpus(set_code)
    cards = c["cards"]
    dcs = c["deck_cards"].collect()
    rng = np.random.default_rng(SEED)
    events = [e for e in EVENT_ORDER if e in set(h["event"].unique().to_list())] + sorted(
        e for e in h["event"].unique().to_list() if e not in EVENT_ORDER)
    rows: list[dict] = []
    for ei, ev in enumerate(events):
        _check(cancel)
        if progress:
            progress(ei / max(len(events), 1), f"Simulator realism: {ev}")
        e = h.filter((pl.col("event") == ev) & pl.col("n_lands").is_in(list(LAND_BINS)))
        if e.height < MIN_BIN_GAMES:
            continue
        regime = config.event_regime(ev)
        sim_regime = (regimes or {}).get(regime, regime)
        parts = []
        for L in LAND_BINS:
            eb = e.filter(pl.col("n_lands") == L)
            if eb.height >= MIN_BIN_GAMES:
                parts.append(eb.sample(min(eb.height, max_games_per_bin), seed=int(rng.integers(1 << 30))))
        if not parts:
            continue
        e = pl.concat(parts)
        keys = e["deck_key"].unique().to_numpy()
        sub = dcs.filter(pl.col("deck_key").is_in(keys.tolist()))
        deck_of: dict[int, dict[int, int]] = {}
        for k, g in sub.partition_by("deck_key", as_dict=True, include_key=True).items():
            kk = int(k[0] if isinstance(k, tuple) else k)
            deck_of[kk] = dict(zip(g["card_id"].to_list(), g["n"].to_list()))
        klist = [int(k) for k in keys if int(k) in deck_of]
        configs = {"v1": (1, None), "v2": (2, sim_regime)}
        sims = simulate_decks([deck_of[k] for k in klist], cards, configs, n_jobs, cancel)
        kidx = {k: i for i, k in enumerate(klist)}
        e = e.filter(pl.col("deck_key").is_in(klist))
        gi = np.array([kidx[int(k)] for k in e["deck_key"].to_list()])
        grp = e["draft_id"].to_numpy()
        op = e["on_play"].to_numpy()
        nl = e["n_lands"].to_numpy()
        td = e["turns_done"].to_numpy()
        nm = e["num_mulligans"].to_numpy()
        n_h1 = e["n_h1"].fill_null(0).to_numpy()
        l_h1 = e["l_h1"].fill_null(0).to_numpy()

        def col(name):
            return e[name].cast(pl.Float64).to_numpy()
        lands = {t: col(f"lands_{t}") for t in (3, 4, 5)}
        mana = {t: sum(np.nan_to_num(col(f"mana_{k}")) for k in range(1, t + 1)) for t in (4, 7)}
        sim_by = {tag: {
            "mull": np.array([sims[i][tag]["mull"] for i in gi]),
            "hand": np.array([sims[i][tag]["hand7"] for i in gi]),
            "screw3": np.array([sims[i][tag]["screw3"] for i in gi]),
            "lands": np.array([sims[i][tag]["lands"] for i in gi]),
            "mana": np.array([sims[i][tag]["mana"] for i in gi])} for tag in configs}
        ls = np.arange(8)
        for L in list(LAND_BINS) + [None]:
            m0 = np.ones(len(gi), dtype=bool) if L is None else (nl == L)
            if m0.sum() < MIN_BIN_GAMES:
                continue
            row: dict = dict(event=ev, regime=regime, n_lands=L, n_games=int(m0.sum()), metrics={})

            def add(key, y, mask, sim_fn):
                mk = m0 & mask & np.isfinite(y)
                if mk.sum() < 50:
                    return
                entry = {}
                for tag in configs:
                    r = compare(y[mk], sim_fn(tag)[mk], grp[mk], op[mk], n_boot=120, seed=SEED)
                    entry[tag] = r
                base = entry["v2"]
                row["metrics"][key] = dict(n=base["n"], real=base["real"], lo=base["lo"], hi=base["hi"],
                                           **{tag: dict(sim=entry[tag]["sim"], diff=entry[tag]["diff"], diff_lo=entry[tag]["diff_lo"],
                                                        diff_hi=entry[tag]["diff_hi"]) for tag in configs})

            allm = np.ones(len(gi), dtype=bool)
            add("mull", (nm > 0).astype(float), allm, lambda t: sim_by[t]["mull"])
            h1 = n_h1 == 7
            add("hand_mean", l_h1.astype(float), h1, lambda t: sim_by[t]["hand"] @ ls)
            add("hand_tail", ((l_h1 < 2) | (l_h1 > 4)).astype(float), h1, lambda t: sim_by[t]["hand"][:, [0, 1, 5, 6, 7]].sum(1))
            add("screw3", (lands[3] < 3).astype(float), td >= 3, lambda t: sim_by[t]["screw3"])
            add("lands4", lands[4], td >= 4, lambda t: sim_by[t]["lands"][:, 3])
            add("lands5", lands[5], td >= 5, lambda t: sim_by[t]["lands"][:, 4])
            add("mana4", mana[4], td >= 4, lambda t: sim_by[t]["mana"][:, :4].sum(1))
            add("mana7", mana[7], td >= 7, lambda t: sim_by[t]["mana"][:, :7].sum(1))
            # distance between the real and the simulated land distribution of the first hand
            mk = m0 & h1
            if mk.sum() >= 100:
                w = _play_draw_weights(op[mk])
                row["hand_tvd"] = {tag: _tvd_ci(l_h1[mk], w, np.average(sim_by[tag]["hand"][mk], axis=0, weights=w), grp[mk])
                                   for tag in configs}
            rows.append(row)
    return dict(source="replay", rows=rows, n_trials=N_TRIALS, regimes={e: config.event_regime(e) for e in events},
                params=dict(kappa_draw=manasim.KAPPA_DRAW, kappa_fetch=manasim.KAPPA_FETCH, cast_p_creature=manasim.CAST_P_CREATURE,
                            cast_p_other=manasim.CAST_P_OTHER, revision=manasim.SIM2_REVISION))
