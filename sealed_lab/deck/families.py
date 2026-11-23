"""Deck families: which different decks does the model's uncertainty leave open, and which cards are firmly in or still flexible?

The model gives B bootstrap refits of every card value and structure term (draw 0 = full-data fit).  A single "best deck" hides how
much of it is estimation noise: another refit would pick another deck.  This module

1. searches for the best deck under EVERY draw (`builder.run_search` with a `Ctx` bound to that draw), then polishes a set of
   seed decks - the draw-0 best deck of each of the top colour combinations plus the winners of the other draws - under every
   draw by local search, so alternative strategies get a fair chance under each parameter set;
2. evaluates every distinct candidate under every draw (`Ctx.logit_draws`);
3. clusters the candidates by card overlap: distance = nonland cards that differ (the larger deck's spell count minus the
   shared copies) + 2 per colour that is a main colour of only one of the decks; average linkage, cut at `CLUSTER_CUT` = 7.  Why 7: on 12
   real pools (MSH, SOS, bo1, bo3) the dendrogram has a valley between merge heights ~7 and ~9 (the distinct strategies merge at
   9-15, the continuum of near-identical decks below), and 7 is where "members share >= 16 of ~23 spells" still describes one deck
   with some flex cards; the old 10-card rule is the upper end of that valley.  The colour term makes WU and WB decks that share
   their white cards two families (without it 2-colour decks that share a colour merged in 8 of 12 pools);
4. summarises each family: representative = its best member under draw 0; P(best) = share of the bootstrap draws in which the
   family's best member is the best candidate overall; win-rate gap of the representative to #1 with an 80% interval over the
   paired draws; core cards (in >= `CORE_FREQ` of the family's per-draw best decks) and flex cards (the rest, with the cards that
   replace them);
5. decides how many families are worth showing (`select_shown`).

Display rule (also in the GUI tooltip): the top family (best representative under the full-data fit) is always shown.  Another
family is shown if ANY of
    - P(best) >= `SHOW_P_BEST` (5%; with B = 16 replicates P(best) moves in steps of 1/B = 6.25%, so this means "wins at least one
      replicate"),
    - its representative is within `SHOW_GAP_PP` (1 pp) of the top representative under the full-data fit,
    - the 80% interval of the paired gap to #1 reaches zero (the representative beats #1 in >= ~10% of the draws),
and at most `MAX_FAMILIES` (5) families are shown, in order of their full-data win rate.  A pool with one clear best deck shows one.

Every probability is over the bootstrap replicates (draws 1..B), never including draw 0, which is the full-data fit, not a
replicate.  Gaps compare the two representatives (the decks a player would actually choose between), evaluated under each draw.
"""
from __future__ import annotations

import atexit
import concurrent.futures as cf
import concurrent.futures.process  # noqa: F401  (cf.process.BrokenProcessPool)
import logging
import multiprocessing
import os
import threading
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.special import expit
from scipy.spatial.distance import squareform

from sealed_lab.deck import builder
from sealed_lab.deck.builder import Ctx, Searcher
from sealed_lab.procguard import watch_parent
from sealed_lab.model import structure
from sealed_lab.model.artifact import SetModel
from sealed_lab.types import BuildResult, Cancelled

log = logging.getLogger(__name__)

MAX_FAMILIES = 5
CLUSTER_CUT = 7.0          # average-linkage cut: families are groups whose members differ in <= ~7 of ~23 nonland cards on average
COLOR_PENALTY = 2.0        # added to the card distance for every colour that is a main colour of one deck but not of the other
DECK_SPELLS = 23
SHOW_P_BEST = 0.05
SHOW_GAP_PP = 1.0
CORE_FREQ = 0.9            # a card is core if it is in >= 90% of the family's per-draw best decks
ALT_FREQ = 0.1             # an alternative (not in the representative) is listed if it is in >= 10% of them
MAX_ALTS = 3
INTERVAL = (10.0, 90.0)    # 80% interval, as for the game win rates
N_SEED_MAINS = 4           # draw-0 best decks of this many different main colour combinations are polished under every draw
MAX_SEEDS = 6
SEED_MAX_GAP_PP = 6.0     # a seed more than this far (pp, draw 0, before polishing) behind the best deck is dropped
SEED_APART = 8            # extra seeds must differ from the earlier ones by at least this many nonland cards
KEEP_DRAW0 = 40            # candidates kept from the draw-0 search (other draws: KEEP_OTHER)
KEEP_OTHER = 8


# =============================================================================================
# search tasks (run in worker processes, or in this process when n_jobs = 1)
# =============================================================================================

Row = tuple[tuple[tuple[int, int], ...], str, np.ndarray]       # (deck items, configuration key, logit under every draw)


class _State:
    """One pool's search problem: base context, prepared configurations, one `Ctx` per draw (shared mana-simulation cache)."""

    def __init__(self, model: SetModel, fmt: str, pool: dict[int, int], locked: dict[int, int] | None,
                 excluded: set[int] | None, start_decks: list[dict[int, int]] | None, sim_cache: dict | None = None):
        self.model, self.fmt = model, fmt
        self.ctx0 = Ctx(model, fmt)
        if sim_cache is not None:
            self.ctx0._sim_cache = sim_cache
        self.prep = builder.prepare(self.ctx0, pool, locked, excluded, start_decks)
        self._ctx: dict[int, Ctx] = {0: self.ctx0}

    def ctx(self, draw: int) -> Ctx:
        c = self._ctx.get(draw)
        if c is None:
            c = self._ctx[draw] = self.ctx0.for_draw(draw)
        return c

    def row(self, deck: dict[int, int], cfg_key: str) -> Row:
        return tuple(sorted((int(i), int(n)) for i, n in deck.items() if n > 0)), cfg_key, self.ctx0.logit_draws(deck)


def run_task(st: _State, task: tuple, cancel: threading.Event | None = None) -> list[Row]:
    """("search", b): fresh search under draw b.  ("polish", draws, seeds): local search from each seed deck under each of `draws`."""
    kind = task[0]
    prep = st.prep
    if kind == "search":
        b = int(task[1])
        # draw 0 gets the wider search (it also has to reproduce the classic best deck); the others the quick one
        ranked, _spare, _top = builder.run_search(st.ctx(b), prep, n_decks=2 if b == 0 else 1, cancel=cancel)
        return [st.row(c.deck, c.cfg.key) for c in ranked[:KEEP_DRAW0 if b == 0 else KEEP_OTHER]]
    out: list[Row] = []
    by_key = {c.key: c for c in prep.configs}
    for items, cfg_key in task[2]:                      # one seed after the other: its neighbourhood stays in the simulation cache
        deck0 = {int(i): int(n) for i, n in items}
        for b in task[1]:
            ctx = st.ctx(int(b))
            cfg = by_key.get(cfg_key)
            win = builder.deck_window(ctx, prep, deck0)          # (the global window unless builder.ARCH_WINDOW)
            s = Searcher(ctx, cfg, prep.pool, prep.nl_bounds, cancel=cancel, cre_bounds=win) if cfg is not None else None
            if s is None or not s.valid(deck0):
                cfg = builder._config_of_deck(ctx, deck0, prep.configs, prep.pool)
                s = Searcher(ctx, cfg, prep.pool, prep.nl_bounds, cancel=cancel, cre_bounds=win)
            new, _lg = s.run(deck0, pairs=False)
            out.append(st.row(new, cfg.key))
    return out


# ---- worker processes: one persistent spawn pool per (set, model file), each worker loads the model once ----------------------

_W: dict = {"key": None, "model": None, "states": {}, "sim": {}}


def _worker_ping() -> int:
    return os.getpid()


def _worker_state(job: dict) -> _State:
    key = (job["set_code"], job["stamp"])
    if _W["key"] != key:
        _W.update(key=key, model=builder.load_model(job["set_code"]), states={}, sim={})
    st = _W["states"].get(job["id"])
    if st is None:
        if len(_W["states"]) >= 2:
            _W["states"].pop(next(iter(_W["states"])))
        st = _W["states"][job["id"]] = _State(_W["model"], job["fmt"], job["pool"], job["locked"], set(job["excluded"]),
                                              job["start_decks"], _W["sim"].setdefault(job["fmt"], {}))
    return st


def _worker_task(job: dict, task: tuple) -> list[Row]:
    return run_task(_worker_state(job), task)


IDLE_SECONDS = 600.0           # the worker pool is shut down after this long without a build
_POOL_LOCK = threading.Lock()
_POOL: dict = {"ex": None, "key": None, "timer": None, "busy": 0}


def shutdown_pool(kill: bool = False) -> None:
    """Stop the persistent worker pool (it restarts on demand), whatever it is doing.  Also runs at interpreter exit; the idle
    timeout uses `_idle_expire`, which never stops a pool that a build is using."""
    with _POOL_LOCK:
        ex, _POOL["ex"], _POOL["key"] = _POOL["ex"], None, None
        t, _POOL["timer"] = _POOL["timer"], None
    if t is not None:
        t.cancel()
    if ex is None:
        return
    procs = list(getattr(ex, "_processes", {}).values()) if kill else []
    try:
        ex.shutdown(wait=False, cancel_futures=True)
    except Exception:  # noqa: BLE001
        log.debug("pool shutdown failed", exc_info=True)
    for p in procs:
        try:
            p.terminate()
        except Exception:  # noqa: BLE001
            pass


def _idle_expire() -> None:
    """Timer callback: stop the pool only if no build is using it (a build that started just before the timer fired keeps it)."""
    with _POOL_LOCK:
        if _POOL["busy"] > 0 or _POOL["ex"] is None:
            _POOL["timer"] = None
            return
    shutdown_pool()


def _arm_idle_timer_locked() -> None:
    if _POOL["timer"] is not None:
        _POOL["timer"].cancel()
        _POOL["timer"] = None
    if _POOL["ex"] is None or _POOL["busy"] > 0:
        return
    t = threading.Timer(IDLE_SECONDS, _idle_expire)
    t.daemon = True
    _POOL["timer"] = t
    t.start()


def _acquire_pool(set_code: str, stamp: int, n: int) -> cf.ProcessPoolExecutor | None:
    """The persistent pool for (set, model file, n workers), marked busy until `_release_pool` (the idle timer is cancelled while it
    is in use).  A pool that serves another build or another set is never shut down from here: returns None (the caller then
    searches in its own process).  An idle pool of a different key is replaced."""
    key = (set_code, stamp, n)
    with _POOL_LOCK:
        ex = _POOL["ex"]
        if ex is not None and (getattr(ex, "_broken", False) or _POOL["key"] != key):
            if _POOL["busy"] > 0:
                log.info("worker pool is busy with another build; searching in this process")
                return None
            ex = None
        if ex is None:
            old, _POOL["ex"], _POOL["key"] = _POOL["ex"], None, None
            if _POOL["timer"] is not None:
                _POOL["timer"].cancel()
                _POOL["timer"] = None
            if old is not None:
                try:
                    old.shutdown(wait=False, cancel_futures=True)
                except Exception:  # noqa: BLE001
                    log.debug("old pool shutdown failed", exc_info=True)
            ex = cf.ProcessPoolExecutor(max_workers=n, mp_context=multiprocessing.get_context("spawn"), initializer=watch_parent)
            _POOL["ex"], _POOL["key"] = ex, key
        _POOL["busy"] += 1
        if _POOL["timer"] is not None:
            _POOL["timer"].cancel()
            _POOL["timer"] = None
        return ex


def _release_pool() -> None:
    """The build is done with the pool: re-arm the idle timer once nobody uses it."""
    with _POOL_LOCK:
        _POOL["busy"] = max(0, _POOL["busy"] - 1)
        if _POOL["busy"] == 0:
            _arm_idle_timer_locked()


atexit.register(shutdown_pool, True)


class _Runner:
    """Runs search tasks serially in this process or on the persistent spawned process pool."""

    def __init__(self, st: _State, n_jobs: int, cancel: threading.Event | None):
        self.st, self.cancel = st, cancel
        self.ex: cf.ProcessPoolExecutor | None = None
        self.n_workers = 1
        self._held = False
        model = st.model
        stamp = getattr(model, "_source_stamp", None)
        n = max(1, int(n_jobs))
        if n > 1 and stamp is not None and stamp == builder.model_stamp(model.set_code):
            try:
                self.ex = _acquire_pool(model.set_code, stamp, n)
                if self.ex is None:
                    return
                self._held = True
                self.n_workers = n
                self.job = dict(id=uuid.uuid4().hex, set_code=model.set_code, stamp=stamp, fmt=st.fmt, pool=st.prep.pool,
                                locked=dict(st.ctx0.locked), excluded=sorted(st.ctx0.excluded),
                                start_decks=[dict(d) for d in st.prep.start_decks])
                for _ in range(n):                       # starts all workers now, while the caller does its own first search
                    self.ex.submit(_worker_ping)
            except Exception:  # noqa: BLE001 - cannot spawn: fall back to this process
                log.warning("process pool unavailable; searching in this process", exc_info=True)
                if self._held:
                    self._held = False
                    _release_pool()
                shutdown_pool(kill=True)
                self.ex, self.n_workers = None, 1
        elif n > 1:
            log.info("model was not loaded from disk by builder.load_model: searching in this process")

    def close(self) -> None:
        """Give the pool back (re-arms the idle timer when nobody else uses it)."""
        if self._held:
            self._held = False
            _release_pool()

    def map(self, tasks: list[tuple], progress=None) -> list[list[Row]]:
        results: list[list[Row] | None] = [None] * len(tasks)
        if self.ex is None:
            for k, t in enumerate(tasks):
                if self.cancel is not None and self.cancel.is_set():
                    raise Cancelled()
                results[k] = run_task(self.st, t, self.cancel)
                if progress:
                    progress(k + 1, len(tasks))
            return results  # type: ignore[return-value]
        futs = {self.ex.submit(_worker_task, self.job, t): k for k, t in enumerate(tasks)}
        pending = set(futs)
        done_n = 0
        redo: list[int] = []
        try:
            while pending:
                done, pending = cf.wait(pending, timeout=0.1, return_when=cf.FIRST_COMPLETED)
                if self.cancel is not None and self.cancel.is_set():
                    raise Cancelled()
                for f in done:
                    try:
                        results[futs[f]] = f.result()
                    except (cf.CancelledError, cf.process.BrokenProcessPool):
                        redo.append(futs[f])             # the pool was stopped under us (shutdown / crash): redone here below
                    done_n += 1
                    if progress:
                        progress(done_n, len(tasks))
        except BaseException:
            for f in pending:
                f.cancel()                               # queued tasks are dropped; the running ones finish within seconds
            raise
        if redo:
            log.warning("the worker pool stopped during a build; %d search task(s) are redone in this process", len(redo))
            for k in sorted(redo):
                if self.cancel is not None and self.cancel.is_set():
                    raise Cancelled()
                results[k] = run_task(self.st, tasks[k], self.cancel)
        return results  # type: ignore[return-value]


# =============================================================================================
# clustering
# =============================================================================================

def spell_matrix(decks: list[dict[int, int]], is_land: np.ndarray, n_cards: int) -> np.ndarray:
    """(n_decks, n_cards) copies of every NONLAND card."""
    S = np.zeros((len(decks), n_cards))
    for k, d in enumerate(decks):
        for i, n in d.items():
            if not is_land[i]:
                S[k, i] = n
    return S


def pairwise_diff(S: np.ndarray) -> np.ndarray:
    """Number of nonland copies that differ: max(cards of A not in B, cards of B not in A) = max(|A|, |B|) - shared copies."""
    tot = S.sum(1)
    shared = np.minimum(S[:, None, :], S[None, :, :]).sum(2)
    return np.maximum(tot[:, None], tot[None, :]) - shared


def cluster_decks(S: np.ndarray, cut: float = CLUSTER_CUT, colors: np.ndarray | None = None) -> np.ndarray:
    """Family label (0..k-1) of every deck: average-linkage hierarchical clustering on `pairwise_diff` (+ `COLOR_PENALTY` per colour
    in the symmetric difference of the decks' main colours, `colors` = (n, 5) bool), cut at `cut`."""
    n = len(S)
    if n <= 1:
        return np.zeros(n, dtype=int)
    D = pairwise_diff(S)
    if colors is not None:
        D = D + COLOR_PENALTY * (colors[:, None, :] != colors[None, :, :]).sum(2)
    np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(D, checks=False), method="average")
    return fcluster(Z, t=cut, criterion="distance") - 1


# =============================================================================================
# family statistics
# =============================================================================================

@dataclass
class Family:
    members: list[int]                  # candidate indices
    rep: int                            # representative: best member under draw 0
    p_best: float = 0.0
    gap_pp: float = 0.0                 # win rate of the top representative minus this one, full-data fit (>= 0 for non-top)
    gap_lo_pp: float = 0.0              # 80% interval of that gap over the draws
    gap_hi_pp: float = 0.0
    p_rep_beats_top: float = 0.0        # share of draws in which this representative is at least as good as the top one
    wr: float = 0.0
    best_decks: list[int] = field(default_factory=list)      # per replicate: the family's best member (candidate index)
    why: str = ""


def replicate_slice(n_draws: int) -> slice:
    """Draws used for probabilities: bootstrap replicates 1..B (draw 0 is the full-data fit); draw 0 alone without a bootstrap."""
    return slice(1, None) if n_draws > 1 else slice(0, None)


def family_stats(L: np.ndarray, labels: np.ndarray) -> list[Family]:
    """Families (best first by full-data win rate) from the (n_candidates, B + 1) logit matrix and the cluster labels."""
    R = replicate_slice(L.shape[1])
    Lr = L[:, R]
    best_cand = Lr.argmax(0)                                         # overall best candidate per replicate
    fams: list[Family] = []
    for lab in np.unique(labels):
        mem = [int(i) for i in np.nonzero(labels == lab)[0]]
        rep = max(mem, key=lambda i: L[i, 0])
        sub = Lr[mem]                                                # (m, B)
        best_in = [mem[int(j)] for j in sub.argmax(0)]
        wins = sum(1 for b in range(Lr.shape[1]) if labels[best_cand[b]] == lab)
        fams.append(Family(members=mem, rep=rep, p_best=wins / Lr.shape[1], wr=float(expit(L[rep, 0])), best_decks=best_in))
    fams.sort(key=lambda f: -L[f.rep, 0])
    top = fams[0].rep
    W = expit(L) * 100.0
    for f in fams:
        g = W[top] - W[f.rep]                                        # pp, per draw
        f.gap_pp = float(g[0])
        gr = g[R]
        if len(gr) >= 3:
            f.gap_lo_pp, f.gap_hi_pp = (float(x) for x in np.percentile(gr, INTERVAL))
        else:
            f.gap_lo_pp = f.gap_hi_pp = f.gap_pp
        f.p_rep_beats_top = float((gr <= 1e-12).mean())
    return fams


def select_shown(fams: list[Family], max_families: int = MAX_FAMILIES) -> list[int]:
    """Indices (into `fams`, best first) of the families to show; sets `Family.why`.  Rule: see the module docstring."""
    shown: list[int] = []
    for k, f in enumerate(fams):
        if k == 0:
            f.why = "best representative under the full-data fit"
        elif f.p_best >= SHOW_P_BEST:
            f.why = f"best deck in {f.p_best * 100:.0f}% of the model refits"
        elif f.gap_pp <= SHOW_GAP_PP:
            f.why = f"within {SHOW_GAP_PP:g} pp of #1 ({f.gap_pp:.1f} pp)"
        elif f.gap_lo_pp <= 0.0:
            f.why = "its gap to #1 is not distinguishable from zero (80% interval reaches 0)"
        else:
            continue
        if len(shown) < max(1, max_families):
            shown.append(k)
    return shown


def unit_freq(decks: list[dict[int, int]], is_basic: np.ndarray) -> Counter:
    """Share-count of 'copy units': (card, k) = at least k copies of a non-basic card."""
    c: Counter = Counter()
    for d in decks:
        for i, n in d.items():
            if n > 0 and not is_basic[i]:
                for k in range(1, int(n) + 1):
                    c[(int(i), k)] += 1
    return c


def core_flex(rep: dict[int, int], decks: list[dict[int, int]], names: list[str], is_basic: np.ndarray) -> dict:
    """Core / flex split of a family's representative from the family's best decks under each model refit.

    core: units (card copies) of the representative in >= CORE_FREQ of `decks`.  flex: the other units of the representative, with
    their frequency and the cards that replace them.  extra: cards outside the representative that are in >= ALT_FREQ of the decks."""
    n = max(len(decks), 1)
    freq = unit_freq(decks, is_basic)
    rep_units = [(int(i), k) for i, c in sorted(rep.items()) if not is_basic[i] for k in range(1, int(c) + 1)]
    rep_set = set(rep_units)
    core_units = [u for u in rep_units if freq[u] / n >= CORE_FREQ]
    flex_units = [u for u in rep_units if freq[u] / n < CORE_FREQ]
    extras = [(u, v / n) for u, v in freq.items() if u not in rep_set and v / n >= ALT_FREQ]
    extras.sort(key=lambda t: -t[1])

    def card_rows(units: list[tuple[int, int]]) -> dict[int, int]:
        out: dict[int, int] = {}
        for i, _k in units:
            out[i] = out.get(i, 0) + 1
        return out
    core = [dict(card_id=i, name=names[i], count=c) for i, c in sorted(card_rows(core_units).items(), key=lambda t: names[t[0]])]
    flex = []
    for i, c in sorted(card_rows(flex_units).items(), key=lambda t: names[t[0]]):
        first = min((u for u in flex_units if u[0] == i), key=lambda u: u[1])
        lacking = [d for d in decks if d.get(i, 0) < first[1]]
        tally: Counter = Counter()
        for d in lacking:
            for j, m in d.items():
                if is_basic[j]:
                    continue
                for k in range(1, int(m) + 1):
                    if (int(j), k) not in rep_set and freq[(int(j), k)] / n >= ALT_FREQ:
                        tally[(int(j), k)] += 1
        alts = [dict(card_id=j, name=names[j], freq=freq[(j, k)] / n) for (j, k), _t in tally.most_common(MAX_ALTS)]
        flex.append(dict(card_id=i, name=names[i], count=c, core_count=int(first[1]) - 1, freq=freq[first] / n, alts=alts))
    extra = [dict(card_id=u[0], name=names[u[0]], freq=f) for u, f in extras[:6]]
    return dict(core=core, flex=flex, extra=extra, n_decks=len(decks))


# =============================================================================================
# orchestration
# =============================================================================================

def build_families(model: SetModel, pool: dict[int, int], fmt: str, max_families: int = MAX_FAMILIES,
                   progress=None, cancel: threading.Event | None = None, start_decks: list[dict[int, int]] | None = None,
                   locked: dict[int, int] | None = None, excluded: set[int] | None = None, n_jobs: int | None = None) -> BuildResult:
    """Deck families for `pool` (see the module docstring).  `BuildResult.decks` = the shown families' representatives (best first),
    `p_best` their P(best), `diffs` their nonland-copy difference to #1, `families` the details."""
    t0 = time.monotonic()

    def prog(f: float, msg: str) -> None:
        if progress:
            progress(min(1.0, f), msg)

    prog(0.0, "Enumerating colour configurations")
    st = _State(model, fmt, pool, locked, excluded, start_decks)
    ctx = st.ctx0
    n_draws = model.n_draws(fmt)
    B = n_draws - 1
    jobs = int(n_jobs) if n_jobs else (int(os.environ.get("SEALED_LAB_FAMILY_JOBS", 0) or 0) or os.cpu_count() or 1)
    if not n_jobs and multiprocessing.parent_process() is not None:
        jobs = 1                                         # already inside a worker process: no pool of pools
    try:
        rows_all, n_workers = _collect_candidates(st, jobs, cancel, prog)
    except cf.process.BrokenProcessPool:
        log.warning("worker processes failed; repeating the search in this process", exc_info=True)
        shutdown_pool(kill=True)
        rows_all, n_workers = _collect_candidates(st, 1, cancel, prog)
    prog(0.88, "Grouping candidates into families")
    rows: dict[tuple, Row] = {}
    for rs in rows_all:
        for r in rs:
            rows.setdefault(r[0], r)
    if builder.ARCH_WINDOW and st.prep.arch_win:
        # every candidate (the polished ones too) is valid only under the window of the archetype group it ends up in
        by_cfg = {c.key: c for c in st.prep.configs}
        starts = {tuple(sorted((int(i), int(n)) for i, n in sd.items() if n > 0)) for sd in st.prep.start_decks}
        kept = {k: r for k, r in rows.items()
                if k in starts or builder.arch_ok(ctx, st.prep, by_cfg.get(r[1]), dict(k))}
        if kept:
            rows = kept
    keys = list(rows)
    decks = [dict(k) for k in keys]
    L = np.stack([rows[k][2] for k in keys])
    S = spell_matrix(decks, ctx.is_land, ctx.n_cards)
    mains = np.array([[c in structure.deck_colors(d, model.card_arrays)[0] for c in "WUBRG"] for d in decks])
    labels = cluster_decks(S, colors=mains)
    fams = family_stats(L, labels)
    shown = select_shown(fams, max_families)
    prog(0.93, "Describing families")
    names = ctx.names
    top_deck = decks[fams[0].rep]
    evals, p_best, diffs, fam_out = [], [], [], []
    for rank, k in enumerate(shown):
        f = fams[k]
        rep = decks[f.rep]
        ev = model.evaluate(rep, fmt)
        if builder.ARCH_WINDOW:
            ev.creature_window = builder.deck_window_meta(model, fmt, rep)
        evals.append(ev)
        p_best.append(f.p_best)
        diffs.append(0 if rank == 0 else builder.deck_diff(ctx, top_deck, rep))
        cf_ = core_flex(rep, [decks[i] for i in f.best_decks], names, ctx.is_basic)
        arch = _archetype(model, rep, fmt)
        fam_out.append(dict(rank=rank, colors=ev.colors, splash=ev.splash, archetype=arch, p_best=f.p_best, wr=f.wr,
                            gap_pp=f.gap_pp, gap_lo_pp=f.gap_lo_pp, gap_hi_pp=f.gap_hi_pp, p_beats_top=f.p_rep_beats_top,
                            n_members=len(f.members), why=f.why, diff_to_top=diffs[-1], **cf_))
    warnings = list(st.prep.warnings)
    if ctx.locked:
        locked_n = sum(ctx.locked.values())
        warnings.append(f"{locked_n} locked card cop{'y' if locked_n == 1 else 'ies'} in every deck")
    meta = dict(n_draws=B, n_candidates=len(keys), n_families=len(fams), n_shown=len(shown), n_hidden=len(fams) - len(shown),
                n_capped=sum(1 for f in fams if f.why) - len(shown),
                cut=CLUSTER_CUT, show_p_best=SHOW_P_BEST, show_gap_pp=SHOW_GAP_PP, max_families=max_families, jobs=n_workers,
                seconds=time.monotonic() - t0, step=(1.0 / B) if B else 1.0)
    prog(1.0, "Done")
    return BuildResult(set_code=model.set_code, fmt=fmt, decks=evals, p_best=p_best, diffs=diffs, pool_unmatched=[],
                       warnings=warnings, families=fam_out, family_meta=meta)


def _collect_candidates(st: _State, jobs: int, cancel: threading.Event | None, prog) -> tuple[list[list[Row]], int]:
    """All candidate decks (with their logit under every draw): the draw-0 search runs in this process while the workers start;
    then, in parallel, the searches under draws 1..B and the polishing of the draw-0 seed decks under every draw."""
    n_draws = st.model.n_draws(st.fmt)
    runner = _Runner(st, jobs, cancel)
    try:
        prog(0.02, "Searching under the full-data model" + (f" ({runner.n_workers} processes starting)" if runner.n_workers > 1 else ""))
        rows0 = run_task(st, ("search", 0), cancel)
        seeds = _pick_seeds(rows0, st)
        n_chunks = max(1, min(n_draws, -(-runner.n_workers // max(len(seeds), 1))))
        chunks = [tuple(range(n_draws))[k::n_chunks] for k in range(n_chunks)]
        tasks = [("polish", ch, [sd]) for sd in seeds for ch in chunks] + [("search", b) for b in range(1, n_draws)]
        prog(0.12, f"Searching under {n_draws - 1} bootstrap draws and polishing {len(seeds)} alternative strategies")
        res = runner.map(tasks, lambda d, n: prog(0.12 + 0.74 * d / n, f"Search step {d}/{n}"))
    finally:
        runner.close()
    return [rows0] + res, runner.n_workers


def _pick_seeds(rows0: list[Row], st: _State) -> list[tuple[tuple, str]]:
    """Seed decks polished under every draw: the draw-0 best deck of each of the top `N_SEED_MAINS` main colour combinations, then
    up to two more strategies (>= `SEED_APART` nonland cards away from the seeds so far), then the user's start decks."""
    ctx = st.ctx0
    seeds: list[tuple[tuple, str]] = []
    mains: set[str] = set()
    ordered = sorted(rows0, key=lambda r: -r[2][0])
    best_wr = float(expit(ordered[0][2][0]))
    ordered = [r for r in ordered if 100 * (best_wr - float(expit(r[2][0]))) <= SEED_MAX_GAP_PP]      # hopeless starts are not polished

    def far(row: Row) -> bool:
        return all(builder.deck_diff(ctx, dict(row[0]), dict(s[0])) >= SEED_APART for s in seeds)
    for row in ordered:
        m = row[1].split("+")[0]
        if m not in mains and len(mains) < N_SEED_MAINS:
            mains.add(m)
            seeds.append((row[0], row[1]))
    for row in ordered:
        if len(seeds) >= MAX_SEEDS:
            break
        if far(row):
            seeds.append((row[0], row[1]))
    seen = {s[0] for s in seeds}
    for sd in st.prep.start_decks:
        sdd = tuple(sorted((int(i), int(n)) for i, n in sd.items() if n > 0))
        if sdd not in seen:
            seen.add(sdd)
            seeds.append((sdd, builder._config_of_deck(ctx, dict(sdd), st.prep.configs, st.prep.pool).key))
    return seeds


def _archetype(model: SetModel, deck: dict[int, int], fmt: str) -> str:
    try:
        from sealed_lab.deck import insight

        m = insight.archetype_of(model, deck, fmt)
        return m.label if m.status == "ok" else ""
    except Exception:  # noqa: BLE001 - descriptive only
        return ""
