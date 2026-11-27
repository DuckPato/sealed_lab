"""Pool -> best decks.

Pipeline (see SPEC "Deck tools"):
  1. enumerate color configurations (mono, pairs, pair+splash, triples with fixing);
  2. per configuration and land count 15..18 a MILP over the part of q that is linear in card counts
     (card betas + curve / creature / splash structure terms); top-k per configuration through no-good cuts;
  3. the candidates are ranked with the FULL model (mana simulator included) and the best ones of different
     configurations are polished by local search with the full model (1-for-1 swaps, +-1 land, basic split,
     non-basic land swaps, 2-for-2 as a last resort);
  4. diverse selection (>= min_diff nonland cards apart); if the candidates are not diverse enough, extra MILPs with
     a distance constraint fill the remaining slots;
  5. p_best from the bootstrap draws, DeckEval per deck.

`Ctx.logit` is an exact re-implementation of `SetModel.logit(draws=False)` that caches the mana simulation by card
*class* (cards with identical cost / colors / mana production share one simulation), which makes swaps between
equivalent cards free.  A test checks the two agree.
"""
from __future__ import annotations

import itertools
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from sealed_lab import config
from sealed_lab.deck import arena as arena_mod
from sealed_lab.deck import names as names_mod
from sealed_lab.deck.milp import MilpSolution, MilpSpec, solve_milp
from sealed_lab.deck.names import NameIndex, resolve_names
from sealed_lab.model import castability, manasim, structure
from sealed_lab.model.artifact import SetModel
from sealed_lab.types import BuildResult, Cancelled, DeckEval

log = logging.getLogger(__name__)

Progress = Callable[[float, str], None]

DECK_SIZE = 40
NL_RANGE = (15, 16, 17, 18)
MIN_LAND_SHARE = 0.02    # a land count needs >= 2% of the format's games: the model is not trusted outside its data
CREATURE_TAIL = 0.02     # creature counts outside the central 96% of the format's games are outside the model's data too
SPLASH_CAP = 3
MIN_CASTABLE = 21
N_CONFIG_FULL = 8          # configurations that get the full land-count enumeration (n_decks > 1)
N_SCREEN = 16              # configurations (best linear upper bounds) solved and screened with the full model
N_LS_STARTS = 5            # candidates polished by local search
SWAP_K = 12                # 1-for-1 swaps evaluated per local-search iteration
MAX_LS_ITER = 14
N_PAIR_STARTS = 1          # local-search starts that also try 2-for-2 moves
# Draw 0 (the full-data fit, which picks the deck) gets a wider search; the searches under the bootstrap draws keep the quick one.
# Session 10 (docs/builder_scoreboard/search_budget.md, split_half.md): on 220 held-out pools this finds a deck the model scores
# +0.41 pp higher in 80% of pools at ~1.3x the time; about half of that gain holds under a model fitted on independent events.
N_SCREEN_DRAW0 = 48
N_CONFIG_FULL_DRAW0 = 20
SWAP_K_DRAW0 = 48
MAX_LS_ITER_DRAW0 = 60
COLORS = "WUBRG"
# Candidate 1 of the Session 12 audit (docs/builder_wide/RULE.md); both off = the shipped builder until that test passes.
WIDE_CONFIGS = False       # also three main colors + a splash, and two main colors + two splash colors
DEFICIT_BOTH = False       # `_deficit` with its documented sign, and deficit-driven land moves proposed in both directions
# A creature window per archetype group (docs/archetype_window/RULE.md): a deck is valid iff its creature count lies in the window
# of the archetype group it is assigned to (`group_creature_support`; the global `creature_support` window for a group without
# one).  False = the shipped builder, byte for byte, until that test passes.
ARCH_WINDOW = False
ARCH_MIN_GAMES = 400       # a group with fewer games in the format has no window of its own


# =============================================================================================
# model context: exact fast logit
# =============================================================================================

class Ctx:
    """Everything the search needs from a `SetModel` for one format.

    `draw` selects the parameter set: 0 = full-data fit, 1..B = bootstrap replicates (the deck families search under each)."""

    def __init__(self, model: SetModel, fmt: str, draw: int = 0):
        if fmt not in model.formats:
            raise ValueError(f"format {fmt!r} is not available for {model.set_code} (have {model.formats})")
        self.model = model
        self.fmt = fmt
        dr = model._draws[fmt]
        if not 0 <= draw < dr["beta"].shape[0]:
            raise ValueError(f"draw {draw} out of range (the model has {dr['beta'].shape[0]})")
        self.draw = int(draw)
        self.ca = model.card_arrays
        self.n_cards = self.ca.n
        self.beta = np.asarray(dr["beta"][draw], dtype=float)
        self.theta = np.asarray(dr["theta"][draw], dtype=float)
        self.ref = float(dr["ref"][draw])
        self.mu = np.asarray(model.mu, dtype=float)
        self.sd = np.asarray(model.sd, dtype=float)
        self.names = model.cards.sort("card_id")["name"].to_list()
        self.n_struct = structure.N_STRUCT
        fi = {n: i for i, n in enumerate(model.feature_names)}
        w = self.theta[:self.n_struct] / self.sd[:self.n_struct]
        self.w = w
        M, lnames = structure.card_feature_matrix(model.cards)
        lin_w = np.array([w[fi[n]] for n in lnames])
        lin_w[0] = 0.0                                   # n_lands: constant once the land count is fixed
        self.lin_base = self.beta + M @ lin_w            # per copy: beta + curve / creature terms
        self.w_mv = float(w[fi["mean_spell_mv"]])
        self.w_nsc = float(w[fi["n_splash_cards"]])
        self.w_nsp = float(w[fi["n_splash_pips"]])
        self.mv = self.ca.mv
        # card-pair synergy terms (min of the copies) and card x archetype values, when the model carries them
        self.pair_a = self.pair_b = self.pair_c = None
        self.P: np.ndarray | None = None
        if "pair" in dr and model.pairs is not None:
            ab = model.pairs["ab"]
            self.pair_a, self.pair_b = ab[:, 0], ab[:, 1]
            self.pair_c = np.asarray(dr["pair"][draw], dtype=float)
            self.P = np.zeros((self.n_cards, self.n_cards))
            self.P[self.pair_a, self.pair_b] = self.pair_c
            self.P[self.pair_b, self.pair_a] = self.pair_c
        self.mull_c = np.asarray(dr["mull"][draw], dtype=float) if "mull" in dr else None     # (coef of 1 mulligan, of 2+)
        self.has_arch = "arch" in dr and model.archcols is not None and model.archetypes is not None
        self._cf = np.zeros(self.n_cards)
        # basic land ids per color
        self.basic_id: dict[int, int] = {}
        for i, nm in enumerate(self.names):
            if nm in ("Plains", "Island", "Swamp", "Mountain", "Forest"):
                self.basic_id[COLORS.index(config.BASICS[nm])] = i
        self.is_basic = self.ca.is_basic
        self.is_land = self.ca.is_land
        self.sim_version, self.regime = model.sim_spec(fmt)       # the mana simulator the model was fitted with, for this format
        self.cast_applied = bool(model.cast_applied)             # castability-weighted copies + dead-cards feature deployed
        if self.cast_applied:
            self.cc = model.cast_cards
            self.cast_regime = model.cast_regime(fmt)
            self.cast_trials = model.cast_trials
        self.locked: dict[int, int] = {}        # card_id -> copies every deck must contain (set by `build`)
        self.excluded: set[int] = set()         # cards no deck may contain (already removed from the pool)
        self._logit_cache: dict[tuple, float] = {}
        self._sim_cache: dict[bytes, np.ndarray] = {}
        self.n_eval = 0
        self.n_sim = 0

    # --------------------------------------------------------------------- exact logit
    def sim(self, deck: dict[int, int]) -> np.ndarray:
        dc = manasim.deck_classes(deck, self.ca, v2=self.sim_version == 2)
        key = manasim.class_signature(dc)
        hit = self._sim_cache.get(key)
        if hit is None:
            s = manasim.simulate(deck, self.ca, manasim.DEFAULT_TRIALS, version=self.sim_version, regime=self.regime)
            hit = np.array([s[k] for k in structure.MANA_NAMES])
            if len(self._sim_cache) > 20000:
                self._sim_cache.clear()
            self._sim_cache[key] = hit
            self.n_sim += 1
        return hit

    def mull_term(self, deck: dict[int, int]) -> float:
        """Expected mulligan cost (logit) of the deck itself under this draw (depends on the land count only)."""
        if self.mull_c is None:
            return 0.0
        nl, n = self.model.deck_lands(deck)
        p1, p2 = self.model.mull_probs(nl, n, self.fmt)
        return float(self.mull_c[0] * p1 + self.mull_c[1] * p2)

    def cast(self, deck: dict[int, int]):
        """Castability of the deck's cards (`castability.CastResult`; its cache lives next to the mana-simulation cache)."""
        ids = np.fromiter((k for k, v in deck.items() if v > 0), dtype=np.int64)
        cnt = np.fromiter((v for v in deck.values() if v > 0), dtype=np.int64)
        return castability.analyze(ids, cnt, self.cc, self.cast_regime, self.cast_trials, cache=self._sim_cache)

    def phi(self, deck: dict[int, int]) -> np.ndarray:
        base = np.concatenate([structure.structure_counts(deck, self.ca), self.sim(deck)])
        if self.cast_applied:
            return np.append(base, self.cast(deck).dead)
        return base

    def weights(self, deck: dict[int, int]) -> tuple[np.ndarray, np.ndarray]:
        """(card ids, effective copies): copies x castability with castability deployed, else the copies."""
        if self.cast_applied:
            r = self.cast(deck)
            return r.ids, r.weights
        ids = np.fromiter((k for k, v in deck.items() if v > 0), dtype=np.int64)
        return ids, np.fromiter((v for v in deck.values() if v > 0), dtype=float)

    def logit(self, deck: dict[int, int]) -> float:
        key = tuple(sorted(deck.items()))
        hit = self._logit_cache.get(key)
        if hit is not None:
            return hit
        ids = np.fromiter(deck.keys(), dtype=np.int64, count=len(deck))
        cnt = np.fromiter(deck.values(), dtype=float, count=len(deck))
        z = (self.phi(deck) - self.mu) / self.sd
        if self.cast_applied:
            wi, ww = self.weights(deck)
            out = self.ref + float(self.beta[wi] @ ww) + float(self.theta @ z)
        else:
            out = self.ref + float(self.beta[ids] @ cnt) + float(self.theta @ z)
        if self.pair_c is not None:
            cf = self._cf
            cf[ids] = cnt
            out += float(self.pair_c @ np.minimum(cf[self.pair_a], cf[self.pair_b]))
            cf[ids] = 0.0
        if self.has_arch:
            out += float(self.model._extra_terms(deck, self.fmt, self.draw > 0)[1][self.draw])
        out += self.mull_term(deck)
        if len(self._logit_cache) > 200000:
            self._logit_cache.clear()
        self._logit_cache[key] = out
        self.n_eval += 1
        return out

    def for_draw(self, draw: int) -> "Ctx":
        """The same context under another draw: shares the (draw independent) mana-simulation cache, locks and exclusions."""
        out = Ctx(self.model, self.fmt, draw)
        out._sim_cache = self._sim_cache
        out.locked, out.excluded = self.locked, self.excluded
        return out

    def logit_draws(self, deck: dict[int, int]) -> np.ndarray:
        """Logit of `deck` under every draw (shape (B + 1,)); equals `SetModel.logit(deck, fmt, draws=True)` (tested) but
        reuses the cached mana simulation."""
        dr = self.model._draws[self.fmt]
        wi, ww = self.weights(deck)
        z = (self.phi(deck) - self.mu) / self.sd
        out = np.asarray(dr["ref"], dtype=float) + np.asarray(dr["beta"][:, wi], dtype=float) @ ww + np.asarray(dr["theta"], dtype=float) @ z
        if self.pair_c is not None:
            ids = np.fromiter((k for k, v in deck.items() if v > 0), dtype=np.int64)
            cf = np.zeros(self.n_cards)
            cf[ids] = np.fromiter((v for v in deck.values() if v > 0), dtype=float)
            out = out + np.asarray(dr["pair"], dtype=float) @ np.minimum(cf[self.pair_a], cf[self.pair_b])
        if self.has_arch:
            out = out + self.model._extra_terms(deck, self.fmt, True)[1]
        return out + self.model.mull_cost(deck, self.fmt, True)

    def copies_vector(self, deck: dict[int, int]) -> np.ndarray:
        cf = np.zeros(self.n_cards)
        for i, n in deck.items():
            cf[i] = n
        return cf

    def pair_delta(self, cand: np.ndarray, cf: np.ndarray, sign: int) -> np.ndarray:
        """Change of the pair-synergy term (logit) from adding (sign=+1) / removing (-1) ONE copy of each candidate card
        to / from a deck with copies `cf` (zeros without a synergy layer). Used to rank moves; the exact value comes from `logit`."""
        if self.P is None or len(cand) == 0:
            return np.zeros(len(cand))
        ci = cf[cand][:, None]
        diff = np.minimum(ci + sign, cf[None, :]) - np.minimum(ci, cf[None, :])
        return (self.P[cand] * diff).sum(1)

    # --------------------------------------------------------------------- linear part
    def lin_spell(self, ids: np.ndarray, n_lands: int) -> np.ndarray:
        """Gain per copy of spells `ids` when the deck has `n_lands` lands (mean spell mv is linear then)."""
        return self.lin_base[ids] + self.w_mv * self.mv[ids] / (DECK_SIZE - n_lands)

    def splash_term(self, ids: np.ndarray, sc) -> np.ndarray:
        """Splash cost per copy of the splash cards `ids`; `sc` = one splash color index or a list of them."""
        scs = [sc] if isinstance(sc, (int, np.integer)) else list(sc)
        return self.w_nsc + self.w_nsp * self.ca.pips[np.ix_(ids, scs)].sum(1)


# =============================================================================================
# configurations
# =============================================================================================

@dataclass
class Config:
    main: str
    splash: str
    cls: np.ndarray                  # (n_cards,) int8: 0 unplayable, 1 main colors, 2 splash (spells only)
    land_ok: np.ndarray              # (n_cards,) bool: non-basic lands usable in this configuration
    floors: dict[int, int]
    castable: int                    # castable spell copies (splash capped)

    @property
    def key(self) -> str:
        return self.main + ("+" + self.splash if self.splash else "")

    @property
    def colors(self) -> list[int]:
        return [COLORS.index(c) for c in self.main + self.splash]

    @property
    def sc(self) -> list[int]:
        """Splash color indices (empty without a splash; two with WIDE_CONFIGS' two-splash configurations)."""
        return [COLORS.index(c) for c in self.splash]


def _classify(ca, pool_ids: np.ndarray, main: str, splash: str) -> np.ndarray:
    """Class (0/1/2) of every pool card for one color configuration."""
    mm = np.array([c in main for c in COLORS])
    pips = ca.pips[pool_ids]
    need = pips > 0
    hyb = ca.hybrid[pool_ids] & (need.sum(1) > 1)
    off_main = need & ~mm[None, :]
    in_main = need & mm[None, :]
    spell = ~ca.is_land[pool_ids]
    main_ok = np.where(hyb, in_main.any(1), ~off_main.any(1)) & spell
    cls = np.where(main_ok, 1, 0).astype(np.int8)
    if splash:
        sm = np.array([c in splash for c in COLORS])
        only_s = off_main.sum(1) == 1
        in_s = (need & sm[None, :]).any(1)
        one_pip = (pips * off_main).sum(1) == 1
        s_ok = np.where(hyb, in_s & ~in_main.any(1), only_s & (off_main & sm[None, :]).any(1) & one_pip) & spell & ~main_ok
        cls[s_ok] = 2
    return cls


def enumerate_configs(ctx: Ctx, pool: dict[int, int]) -> list[Config]:
    ca = ctx.ca
    ids = np.array(sorted(pool), dtype=np.int64)
    cnt = np.array([pool[i] for i in ids], dtype=int)
    land = ca.is_land[ids]
    prod = ca.produces[ids]
    full_cls = np.zeros(ctx.n_cards, dtype=np.int8)
    out: list[Config] = []

    def make(main: str, splash: str, floors: dict[int, int]) -> Config | None:
        cls_p = _classify(ca, ids, main, splash)
        cls = full_cls.copy()
        cls[ids] = cls_p
        colset = np.array([c in main + splash for c in COLORS])
        lok = np.zeros(ctx.n_cards, dtype=bool)
        lp = land & (prod[:, colset].any(1) | ~prod.any(1))
        lok[ids[lp]] = True
        castable = int(cnt[cls_p == 1].sum()) + min(SPLASH_CAP, int(cnt[cls_p == 2].sum()))
        return Config(main, splash, cls, lok, floors, castable)

    for c in range(5):
        cfg = make(COLORS[c], "", {})
        if cfg:
            out.append(cfg)
    for a, b in itertools.combinations(range(5), 2):
        main = COLORS[a] + COLORS[b]
        out.append(make(main, "", {a: 6, b: 6}))
        for s in range(5):
            if s not in (a, b):
                out.append(make(main, COLORS[s], {a: 5, b: 5, s: 3}))
    for tri in itertools.combinations(range(5), 3):
        cols = np.zeros(5, dtype=bool)
        cols[list(tri)] = True
        n_fix = int(cnt[land & ((prod & cols[None, :]).sum(1) >= 2)].sum())
        if n_fix >= 2:
            main = "".join(COLORS[i] for i in tri)
            out.append(make(main, "", {i: 5 for i in tri}))
            if WIDE_CONFIGS:
                for s in range(5):
                    if s not in tri:
                        out.append(make(main, COLORS[s], {**{i: 4 for i in tri}, s: 3}))
    if WIDE_CONFIGS:
        nb_land = land & ~ca.is_basic[ids]
        for a, b in itertools.combinations(range(5), 2):
            rest = [c for c in range(5) if c not in (a, b)]
            for s1, s2 in itertools.combinations(rest, 2):
                n_src = int(cnt[nb_land & (prod[:, s1] | prod[:, s2]).astype(bool)].sum())
                if n_src >= 2:
                    out.append(make(COLORS[a] + COLORS[b], COLORS[s1] + COLORS[s2], {a: 5, b: 5, s1: 2, s2: 2}))
    return out


# =============================================================================================
# deck assembly
# =============================================================================================

def assign_basics(ctx: Ctx, cfg: Config, deck: dict[int, int], n_basics: int) -> dict[int, int]:
    """Split `n_basics` basic lands over the configuration colors: floors first, then proportional to pip demand."""
    cols = cfg.colors
    if not cols or n_basics <= 0:
        return {}
    ca = ctx.ca
    demand = np.zeros(5)
    src = np.zeros(5)
    for cid, n in deck.items():
        if ca.is_land[cid]:
            src += n * ca.produces[cid]
        else:
            w = 0.5 if ca.hybrid[cid] else 1.0
            w *= 1.5 if ca.mv[cid] <= 2 else 1.0
            demand += n * w * ca.pips[cid]
    mask = np.zeros(5, dtype=bool)
    mask[cols] = True
    demand = np.where(mask, demand, 0.0)
    if demand.sum() <= 0:
        demand[cols] = 1.0
    share = demand / demand.sum()
    basics = {c: 0 for c in cols}
    for _ in range(n_basics):
        short = {c: cfg.floors.get(c, 0) - (src[c] + basics[c]) for c in cols}
        c0 = max(cols, key=lambda c: (short[c], share[c]))
        if short[c0] > 0:
            pick = c0
        else:
            tot = src.sum() + sum(basics.values()) + 1
            pick = max(cols, key=lambda c: (share[c] * tot - (src[c] + basics[c]), share[c]))
        basics[pick] += 1
    return {c: n for c, n in basics.items() if n > 0}


def deck_from_solution(ctx: Ctx, cfg: Config, item_ids: np.ndarray, sol: MilpSolution, n_lands: int) -> dict[int, int]:
    deck: dict[int, int] = {int(i): int(n) for i, n in zip(item_ids, sol.counts) if n > 0}
    n_nb_land = sum(n for i, n in deck.items() if ctx.is_land[i])
    for c, n in assign_basics(ctx, cfg, deck, n_lands - n_nb_land).items():
        deck[ctx.basic_id[c]] = deck.get(ctx.basic_id[c], 0) + n
    return deck


def make_spec(ctx: Ctx, cfg: Config, pool: dict[int, int], n_lands: int, linear: bool = True,
              cre_bounds: tuple[int, int] | None = None) -> tuple[MilpSpec, np.ndarray]:
    """MILP for (configuration, land count). Returns the spec and the card id of every item.  `cre_bounds`: creature-count
    support (relaxed to what this pool can do)."""
    ids = np.array([i for i in sorted(pool) if cfg.cls[i] > 0 or cfg.land_ok[i]], dtype=np.int64)
    is_land = ctx.is_land[ids]
    cls = cfg.cls[ids]
    coef = np.where(is_land, ctx.beta[ids], 0.0)
    sp = ~is_land
    if linear:
        coef[sp] = ctx.lin_spell(ids[sp], n_lands)
        if cfg.splash:
            m = sp & (cls == 2)
            coef[m] += ctx.splash_term(ids[m], cfg.sc)
    else:
        coef[sp] = ctx.beta[ids[sp]]
    copies = np.array([pool[int(i)] for i in ids], dtype=int)
    locked = np.array([min(ctx.locked.get(int(i), 0), pool[int(i)]) for i in ids], dtype=int) if ctx.locked else None
    creature = sp & (cls > 0) & ctx.ca.is_creature[ids]
    cre_range = None
    if cre_bounds is not None:
        cop_cap = np.minimum(copies, DECK_SIZE)
        lk = locked if locked is not None else np.zeros(len(ids), dtype=int)
        noncre = sp & (cls > 0) & ~creature
        cre_range = relaxed_creature_bounds(cre_bounds, DECK_SIZE - n_lands, int(cop_cap[creature].sum()),
                                            int(cop_cap[noncre].sum()), int(lk[creature].sum()), int(lk[noncre].sum()))
    spec = MilpSpec(coef=coef, copies=copies, is_land=is_land, produces=ctx.ca.produces[ids], splash=(cls == 2),
                    basic_colors=[c for c in cfg.colors if c in ctx.basic_id], floors=dict(cfg.floors),
                    n_lands=n_lands, deck_size=DECK_SIZE, splash_cap=SPLASH_CAP, locked=locked,
                    creature=creature, creature_range=cre_range)
    return spec, ids


# =============================================================================================
# diversity
# =============================================================================================

def spell_counts(ctx: Ctx, deck: dict[int, int]) -> dict[int, int]:
    return {i: n for i, n in deck.items() if not ctx.is_land[i]}


def deck_diff(ctx: Ctx, a: dict[int, int], b: dict[int, int]) -> int:
    """Nonland copies that differ: max(cards of A not in B, cards of B not in A)."""
    sa, sb = spell_counts(ctx, a), spell_counts(ctx, b)
    plus = sum(max(0, n - sb.get(i, 0)) for i, n in sa.items())
    minus = sum(max(0, n - sa.get(i, 0)) for i, n in sb.items())
    return max(plus, minus)


# =============================================================================================
# local search
# =============================================================================================

class Searcher:
    """Local search in one configuration's neighbourhood with the full model."""

    def __init__(self, ctx: Ctx, cfg: Config, pool: dict[int, int], nl_bounds: tuple[int, int],
                 legal: Callable[[dict[int, int]], bool] | None = None, cancel: threading.Event | None = None,
                 cre_bounds: tuple[int, int] | None = None):
        self.ctx, self.cfg, self.pool = ctx, cfg, pool
        self.nl_lo, self.nl_hi = nl_bounds
        self.legal = legal
        self.cancel = cancel
        self.ids = np.array(sorted(pool), dtype=np.int64)
        self.spell_ids = np.array([i for i in self.ids if cfg.cls[i] > 0], dtype=np.int64)
        # creature-count support, relaxed per land count to what this pool / configuration / locks allow
        self.cre_bounds = cre_bounds
        is_cre = ctx.ca.is_creature
        self._avail_cre = sum(min(pool[int(i)], DECK_SIZE) for i in self.spell_ids if is_cre[i])
        self._avail_non = sum(min(pool[int(i)], DECK_SIZE) for i in self.spell_ids if not is_cre[i])
        self._locked_cre = sum(n for i, n in ctx.locked.items() if not ctx.is_land[i] and is_cre[i])
        self._locked_non = sum(n for i, n in ctx.locked.items() if not ctx.is_land[i] and not is_cre[i])
        self.land_ids = [int(i) for i in self.ids if cfg.land_ok[i]]
        self.cols = [c for c in cfg.colors if c in ctx.basic_id]
        self._ca = ctx.ca
        self.locked = ctx.locked
        self.excluded_ids = ctx.excluded

    # ------------------------------------------------------------------ legality
    def valid(self, deck: dict[int, int]) -> bool:
        ctx, cfg = self.ctx, self.cfg
        n = 0
        nl = 0
        sp = 0
        cre = 0
        src = np.zeros(5)
        for i, c in deck.items():
            if c < 0:
                return False
            if c == 0:
                continue
            if i in self.excluded_ids:
                return False
            n += c
            if ctx.is_basic[i]:
                nl += c
                for col, b in ctx.basic_id.items():
                    if b == i:
                        src[col] += c
                continue
            if c > self.pool.get(i, 0):
                return False
            if ctx.is_land[i]:
                if not cfg.land_ok[i]:
                    return False
                nl += c
                src += c * self._ca.produces[i]
            else:
                k = cfg.cls[i]
                if k == 0:
                    return False
                if k == 2:
                    sp += c
                if self._ca.is_creature[i]:
                    cre += c
        if n != DECK_SIZE or sp > SPLASH_CAP or not (self.nl_lo <= nl <= self.nl_hi):
            return False
        if self.cre_bounds is not None:
            lo, hi = relaxed_creature_bounds(self.cre_bounds, DECK_SIZE - nl, self._avail_cre, self._avail_non,
                                             self._locked_cre, self._locked_non)
            if not lo <= cre <= hi:
                return False
        for i, k in self.locked.items():
            if deck.get(i, 0) < k:
                return False
        for col, f in cfg.floors.items():
            if src[col] < f:
                return False
        return self.legal(deck) if self.legal is not None else True

    @staticmethod
    def apply(deck: dict[int, int], delta: dict[int, int]) -> dict[int, int]:
        out = dict(deck)
        for k, v in delta.items():
            nv = out.get(k, 0) + v
            if nv < 0:
                return {}
            if nv == 0:
                out.pop(k, None)
            else:
                out[k] = nv
        return out

    # ------------------------------------------------------------------ moves
    def moves(self, deck: dict[int, int]) -> list[dict[int, int]]:
        ctx, cfg = self.ctx, self.cfg
        nl = sum(n for i, n in deck.items() if ctx.is_land[i])
        ns = DECK_SIZE - nl
        out: list[dict[int, int]] = []
        lk = self.locked
        in_deck = np.array([i for i, n in deck.items() if n > lk.get(i, 0) and not ctx.is_land[i]], dtype=np.int64)
        remaining = np.array([self.pool.get(int(i), 0) - deck.get(int(i), 0) for i in self.spell_ids], dtype=int)
        cand_in = self.spell_ids[remaining > 0]
        cf = ctx.copies_vector(deck) if ctx.P is not None else None
        if len(in_deck) and len(cand_in):
            lin_out = ctx.lin_spell(in_deck, nl)
            lin_in = ctx.lin_spell(cand_in, nl)
            if cf is not None:           # synergies with the rest of the deck (ranking only; the full logit decides)
                lin_out = lin_out - ctx.pair_delta(in_deck, cf, -1)
                lin_in = lin_in + ctx.pair_delta(cand_in, cf, +1)
            if cfg.splash:
                lin_out = lin_out + np.where(cfg.cls[in_deck] == 2, ctx.splash_term(in_deck, cfg.sc), 0.0)
                lin_in = lin_in + np.where(cfg.cls[cand_in] == 2, ctx.splash_term(cand_in, cfg.sc), 0.0)
            D = lin_in[:, None] - lin_out[None, :]
            D[cand_in[:, None] == in_deck[None, :]] = -np.inf
            flat = np.argsort(-D, axis=None)[:SWAP_K_DRAW0 if ctx.draw == 0 else SWAP_K]
            for f in flat:
                a, b = divmod(int(f), len(in_deck))
                if np.isfinite(D[a, b]):
                    out.append({int(cand_in[a]): 1, int(in_deck[b]): -1})
        basics = {c: deck.get(ctx.basic_id[c], 0) for c in self.cols}
        # +-1 land
        if len(in_deck):
            lin_out = ctx.lin_spell(in_deck, nl)
            worst = in_deck[np.argsort(lin_out)[:2]]
            order = self._by_deficit(deck, self.cols, largest=True)
            for o in worst:
                for c in order:
                    out.append({int(o): -1, ctx.basic_id[c]: 1})
        if len(cand_in):
            lin_in = ctx.lin_spell(cand_in, nl - 1)
            best_in = cand_in[np.argsort(-lin_in)[:2]]
            order = self._by_deficit(deck, [c for c in self.cols if basics[c] > 0], largest=False)
            for i in best_in:
                for c in order:
                    out.append({int(i): 1, ctx.basic_id[c]: -1})
        # basic split: from the colors with the most surplus to the colors with the largest deficit
        defs = {c: self._deficit(deck, c) for c in self.cols}
        src_cols = self._by_deficit(deck, [c for c in self.cols if basics[c] > 0], largest=False, defs=defs)
        dst_cols = self._by_deficit(deck, self.cols, largest=True, defs=defs)
        for a in src_cols:
            for b in dst_cols:
                if a != b:
                    out.append({ctx.basic_id[a]: -1, ctx.basic_id[b]: 1})
        # non-basic lands <-> basics (the basic replaced / added is the one of the most surplus / deficit color)
        prod = self._ca.produces
        for L in self.land_ids:
            have = deck.get(L, 0)
            pcols = [c for c in self.cols if prod[L, c]] or list(self.cols)
            if have < self.pool[L]:
                give = [c for c in pcols if basics[c] > 0]
                if give:
                    picks = {min(give, key=lambda c: defs[c])}
                    if DEFICIT_BOTH:
                        picks.add(max(give, key=lambda c: defs[c]))
                    for c in sorted(picks):
                        out.append({L: 1, ctx.basic_id[c]: -1})
            if have > lk.get(L, 0):
                picks = {max(pcols, key=lambda c: defs[c])}
                if DEFICIT_BOTH:
                    picks.add(min(pcols, key=lambda c: defs[c]))
                for c in sorted(picks):
                    out.append({L: -1, ctx.basic_id[c]: 1})
        return out

    def _deficit(self, deck: dict[int, int], col: int) -> float:
        """demand share * sources - current sources: positive = color needs more sources."""
        ca = self._ca
        demand = np.zeros(5)
        src = np.zeros(5)
        for i, n in deck.items():
            if ca.is_land[i]:
                src += n * ca.produces[i]
            else:
                demand += n * ca.pips[i] * (0.5 if ca.hybrid[i] else 1.0)
        mask = np.zeros(5, dtype=bool)
        mask[self.cfg.colors] = True
        demand = np.where(mask, demand, 0.0)
        if demand.sum() <= 0:
            return 0.0
        d = float(demand[col] / demand.sum() * src.sum() - src[col])
        return d if DEFICIT_BOTH else -d          # shipped: sign inverted (Session 12 audit, section 4)

    def _by_deficit(self, deck: dict[int, int], cols: list[int], largest: bool, k: int = 2,
                    defs: dict[int, float] | None = None) -> list[int]:
        """The `k` colors of `cols` with the largest (or smallest) `_deficit`; with DEFICIT_BOTH also the `k` at the other end."""
        d = defs if defs is not None else {c: self._deficit(deck, c) for c in cols}
        order = sorted(cols, key=lambda c: -d[c]) if largest else sorted(cols, key=lambda c: d[c])
        out = order[:k]
        if DEFICIT_BOTH:
            out += [c for c in order[::-1][:k] if c not in out]
        return out

    # ------------------------------------------------------------------ search
    def run(self, deck: dict[int, int], max_iter: int | None = None, deadline: float | None = None,
            pairs: bool = True) -> tuple[dict[int, int], float]:
        ctx = self.ctx
        if max_iter is None:
            max_iter = MAX_LS_ITER_DRAW0 if ctx.draw == 0 else MAX_LS_ITER
        cur = dict(deck)
        cur_l = ctx.logit(cur)
        for _ in range(max_iter):
            if self.cancel is not None and self.cancel.is_set():
                raise Cancelled()
            if deadline is not None and time.monotonic() > deadline:
                break
            scored: list[tuple[float, dict[int, int], dict[int, int]]] = []
            seen: set[tuple] = set()
            for mv in self.moves(cur):
                nd = self.apply(cur, mv)
                if not nd:
                    continue
                key = tuple(sorted(nd.items()))
                if key in seen or not self.valid(nd):
                    continue
                seen.add(key)
                scored.append((ctx.logit(nd), nd, mv))
            if not scored:
                break
            scored.sort(key=lambda t: -t[0])
            if scored[0][0] > cur_l + 1e-7:
                cur_l, cur = scored[0][0], scored[0][1]
                continue
            # 2-for-2: compose the most promising single moves
            improved = False
            top = scored[:6] if pairs else []
            for (l1, d1, m1), (l2, d2, m2) in itertools.combinations(top, 2):
                delta = dict(m1)
                for k, v in m2.items():
                    delta[k] = delta.get(k, 0) + v
                nd = self.apply(cur, {k: v for k, v in delta.items() if v != 0})
                if not nd:
                    continue
                key = tuple(sorted(nd.items()))
                if key in seen or not self.valid(nd):
                    continue
                seen.add(key)
                lv = ctx.logit(nd)
                if lv > cur_l + 1e-7:
                    cur_l, cur, improved = lv, nd, True
                    break
            if not improved:
                break
        return cur, cur_l


# =============================================================================================
# orchestration
# =============================================================================================

@dataclass
class Cand:
    deck: dict[int, int]
    logit: float
    cfg: Config
    src: str = "milp"

    @property
    def key(self) -> tuple:
        return tuple(sorted(self.deck.items()))


def _check(cancel: threading.Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise Cancelled()


def nl_of(ctx: Ctx, deck: dict[int, int]) -> int:
    return sum(n for i, n in deck.items() if ctx.is_land[i])


def land_share(model: SetModel, fmt: str) -> dict[int, float]:
    """Share of the format's games played with each land count (from the model meta, else from the corpus)."""
    sup = (model.meta.get("land_support") or {}).get(fmt)
    if sup:
        return {int(k): float(v) for k, v in sup.items()}
    cached = getattr(model, "_land_share", {})
    if fmt in cached:
        return cached[fmt]
    out: dict[int, float] = {}
    try:
        import polars as pl

        cd = config.corpus_dir(model.set_code)
        g = pl.read_parquet(cd / "games.parquet", columns=["fmt", "deck_key"]).filter(pl.col("fmt") == fmt)
        d = pl.read_parquet(cd / "decks.parquet", columns=["deck_key", "n_lands"])
        c = g.join(d, on="deck_key").group_by("n_lands").len()
        tot = max(int(c["len"].sum()), 1)
        out = {int(n): int(k) / tot for n, k in c.iter_rows()}
    except Exception:  # noqa: BLE001 - no corpus: fall back to the fixed range
        log.info("land-count support unavailable for %s %s", model.set_code, fmt)
    cached[fmt] = out
    model._land_share = cached  # type: ignore[attr-defined]
    return out


def supported_lands(model: SetModel, fmt: str) -> tuple[int, ...]:
    """Contiguous range of land counts with enough real games behind them (always within NL_RANGE)."""
    share = land_share(model, fmt)
    ok = [n for n in NL_RANGE if share.get(n, 0.0) >= MIN_LAND_SHARE]
    if not ok:
        return NL_RANGE
    return tuple(range(min(ok), max(ok) + 1))


def _corpus_remap_tag(model: SetModel) -> str:
    """"" when the corpus numbers the cards like the model; else a short hash of the corpus card names (the stamp of cached
    corpus-derived tables then changes with the id translation)."""
    import hashlib

    import polars as pl

    from sealed_lab.deck.explain import _corpus_to_model_ids

    cd = config.corpus_dir(model.set_code)
    if _corpus_to_model_ids(model, cd) is None:
        return ""
    names = pl.read_parquet(cd / "cards.parquet", columns=["card_id", "name"]).sort("card_id")["name"].to_list()
    return hashlib.md5("\n".join(names).encode("utf-8")).hexdigest()[:12]


def _corpus_creature_table(model: SetModel, fmt: str):
    """(per, dcards): `per` = one row per 40-card corpus deck played in `fmt` (deck_key, cre = nonland creature copies, w = games
    in `fmt`); `dcards` = those decks' (deck_key, card_id, n) rows.  The one definition of the corpus decks, their weights and
    their creature count that `creature_support` and `group_creature_support` share."""
    import polars as pl

    from sealed_lab.deck.explain import _corpus_to_model_ids

    cd = config.corpus_dir(model.set_code)
    gp = cd / "games.parquet"
    cards = model.cards.select("card_id", "is_creature", "is_land")
    g = (pl.read_parquet(gp, columns=["fmt", "deck_key"]).filter(pl.col("fmt") == fmt)
         .group_by("deck_key").len().rename({"len": "w"}))
    d = pl.read_parquet(cd / "decks.parquet", columns=["deck_key", "n_cards"]).filter(pl.col("n_cards") == 40)
    dcards = (pl.read_parquet(cd / "deck_cards.parquet", columns=["deck_key", "card_id", "n"])
              .join(g.select("deck_key"), on="deck_key").join(d.select("deck_key"), on="deck_key"))
    remap = _corpus_to_model_ids(model, cd)
    if remap is not None:            # the corpus was rebuilt after the model: translate its card ids to the model's (drop unknown cards)
        cid = dcards["card_id"].to_numpy().astype(np.int64)
        new = np.where(cid < len(remap), remap[np.minimum(cid, len(remap) - 1)], -1)
        dcards = dcards.with_columns(card_id=pl.Series("card_id", new, dtype=dcards.schema["card_id"])).filter(pl.col("card_id") >= 0)
    dc = dcards.join(cards, on="card_id", how="left")
    per = (dc.group_by("deck_key", maintain_order=True)
           .agg((pl.col("n") * (pl.col("is_creature").fill_null(False) & ~pl.col("is_land").fill_null(False))).sum()
                .alias("cre"))
           .join(g, on="deck_key"))
    return per, dcards


def _weighted_window(x: np.ndarray, w: np.ndarray) -> tuple[int, int]:
    """The CREATURE_TAIL / 1 - CREATURE_TAIL quantiles of the creature counts `x` weighted by games `w`."""
    order = np.argsort(x, kind="stable")
    cw = np.cumsum(w[order]) / w.sum()
    lo = int(x[order][np.searchsorted(cw, CREATURE_TAIL)])
    hi = int(x[order][min(np.searchsorted(cw, 1.0 - CREATURE_TAIL), len(cw) - 1)])
    return lo, hi


def creature_support(model: SetModel, fmt: str) -> tuple[int, int] | None:
    """(lo, hi): the creature counts (nonland creature copies) of the central 1 - 2 * CREATURE_TAIL of the format's games
    with 40-card decks.  The composition diagnostics (scripts/composition_check.py) found the model prices creature counts
    correctly inside the range real decks use, while its creature / non-creature terms are straight lines: outside that range
    the builder would follow an extrapolation (e.g. 9-creature decks where real decks run 13-14).  None = no corpus, or
    switched off with SEALED_LAB_CREATURE_SUPPORT=0 (diagnostics).  Cached on the model and in the set's cache folder."""
    if os.environ.get("SEALED_LAB_CREATURE_SUPPORT", "1") == "0":
        return None
    cached = getattr(model, "_cre_support", {})
    if fmt in cached:
        return cached[fmt]
    out: tuple[int, int] | None = None
    try:
        import json

        import polars as pl

        cd = config.corpus_dir(model.set_code)
        gp = cd / "games.parquet"
        st = gp.stat()
        stamp = f"{st.st_size}:{st.st_mtime_ns}:{CREATURE_TAIL}:{_corpus_remap_tag(model)}"
        cache_file = config.cache_dir(model.set_code) / "creature_support.json"
        disk = {}
        if cache_file.exists():
            try:
                disk = json.loads(cache_file.read_text(encoding="utf-8"))
            except ValueError:
                disk = {}
        if disk.get("stamp") == stamp and fmt in disk.get("formats", {}):
            lo, hi = disk["formats"][fmt]
            out = (int(lo), int(hi))
        else:
            per, _dcards = _corpus_creature_table(model, fmt)
            if per.height:
                out = _weighted_window(per["cre"].to_numpy().astype(float), per["w"].to_numpy().astype(float))
                lo, hi = out
                fm = dict(disk.get("formats", {})) if disk.get("stamp") == stamp else {}
                fm[fmt] = [lo, hi]
                try:
                    cache_file.parent.mkdir(parents=True, exist_ok=True)
                    cache_file.write_text(json.dumps({"stamp": stamp, "formats": fm}), encoding="utf-8")
                except OSError:
                    pass
    except Exception:  # noqa: BLE001 - no corpus: no creature-count constraint
        log.info("creature-count support unavailable for %s %s", model.set_code, fmt)
    cached[fmt] = out
    model._cre_support = cached  # type: ignore[attr-defined]
    return out


def group_creature_support(model: SetModel, fmt: str, min_games: int | None = None) -> dict[int, tuple[int, int]]:
    """{archetype index: (lo, hi)}: `creature_support`'s window (same corpus decks, weighting, creature definition and quantile
    code) restricted to the 40-card corpus decks that the model's archetypes assign to each group.  Groups with fewer than
    `min_games` (default `ARCH_MIN_GAMES`) games in `fmt` are absent: they use the global window.  {} without archetypes or
    corpus (or with SEALED_LAB_CREATURE_SUPPORT=0).  Cached on the model and in the set's cache folder (the stamp holds the
    corpus file, the tail and a hash of the centroids, so a recalculated model recomputes it)."""
    min_games = ARCH_MIN_GAMES if min_games is None else min_games
    if os.environ.get("SEALED_LAB_CREATURE_SUPPORT", "1") == "0" or getattr(model, "archetypes", None) is None:
        return {}
    ckey = (fmt, min_games)
    cached = getattr(model, "_group_cre_support", {})
    if ckey in cached:
        return cached[ckey]
    out: dict[int, tuple[int, int]] = {}
    try:
        import hashlib
        import json

        from sealed_lab.model import archetypes as arch_mod

        arch = model.archetypes
        cd = config.corpus_dir(model.set_code)
        st = (cd / "games.parquet").stat()
        cen = np.ascontiguousarray(arch.centroids, dtype=np.float32)
        stamp = f"{st.st_size}:{st.st_mtime_ns}:{CREATURE_TAIL}:{hashlib.md5(cen.tobytes()).hexdigest()}:{cen.shape}:{_corpus_remap_tag(model)}"
        cache_file = config.cache_dir(model.set_code) / "creature_support_groups.json"
        disk: dict = {}
        if cache_file.exists():
            try:
                disk = json.loads(cache_file.read_text(encoding="utf-8"))
            except ValueError:
                disk = {}
        if disk.get("stamp") != stamp:
            disk = {"stamp": stamp, "formats": {}}
        if fmt in disk["formats"]:
            table = disk["formats"][fmt]          # {group: [lo, hi, games]}
        else:
            per, dcards = _corpus_creature_table(model, fmt)
            table = {}
            if per.height:
                dk = per["deck_key"].to_numpy().astype(np.int64)
                ordr = np.argsort(dk, kind="stable")
                keys = dk[ordr]
                x = per["cre"].to_numpy().astype(float)[ordr]
                w = per["w"].to_numpy().astype(float)[ordr]
                n_cards = arch.n_cards
                dc = dcards.filter((dcards["card_id"] >= 0) & (dcards["card_id"] < n_cards))
                X = arch_mod.deck_matrix(keys, dc, n_cards, np.asarray(model._spell, dtype=bool))
                grp, _sim = arch.assign_matrix(X)
                for a in np.unique(grp):
                    m = grp == a
                    lo, hi = _weighted_window(x[m], w[m])
                    table[str(int(a))] = [lo, hi, float(w[m].sum())]
            disk["formats"][fmt] = table
            try:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(json.dumps(disk), encoding="utf-8")
            except OSError:
                pass
        out = {int(a): (int(v[0]), int(v[1])) for a, v in table.items() if v[2] >= min_games}
    except Exception:  # noqa: BLE001 - no corpus: the global window only
        log.info("group creature-count support unavailable for %s %s", model.set_code, fmt, exc_info=True)
        out = {}
    cached[ckey] = out
    model._group_cre_support = cached  # type: ignore[attr-defined]
    return out


def relaxed_creature_bounds(bounds: tuple[int, int] | None, n_spells: int, avail_cre: int, avail_non: int,
                            locked_cre: int = 0, locked_non: int = 0) -> tuple[int, int] | None:
    """The support range widened just enough to stay feasible for this pool / configuration / land count: a pool with few
    creatures (or few other playables) and locked cards may force a count outside the range."""
    if bounds is None:
        return None
    lo, hi = bounds
    lo = min(lo, avail_cre, max(n_spells - locked_non, 0))
    hi = max(hi, n_spells - avail_non, locked_cre)
    return (max(lo, 0), min(max(hi, lo), n_spells))


def _land_range(configs: list[Config], base: tuple[int, ...] = NL_RANGE) -> tuple[tuple[int, ...], list[str]]:
    best = max((c.castable for c in configs), default=0)
    need = DECK_SIZE - base[-1]
    if best >= need:
        return base, []
    hi = DECK_SIZE - best
    rng = tuple(range(base[0], max(base[-1], hi) + 1))
    return rng, [f"the pool has only {best} playable nonland cards for any color combination; "
                 f"decks use up to {rng[-1]} lands"]


@dataclass
class Prepared:
    """A pool turned into a search problem: constraints applied, color configurations enumerated, land counts fixed."""
    pool: dict[int, int]
    configs: list[Config]
    viable: list[Config]
    nls: tuple[int, ...]
    nl_bounds: tuple[int, int]
    extended: bool
    base_lands: tuple[int, ...]
    warnings: list[str]
    start_decks: list[dict[int, int]]
    cre_bounds: tuple[int, int] | None = None     # creature-count support of the format (`creature_support`)
    # ARCH_WINDOW only (empty otherwise): {archetype index: window} (`group_creature_support`), and per configuration key the
    # distinct group windows (other than the global one) of the groups relevant to it
    arch_win: dict[int, tuple[int, int]] = field(default_factory=dict)
    cfg_wins: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    _avail: dict = field(default_factory=dict, repr=False)


def prepare(ctx: Ctx, pool: dict[int, int], locked: dict[int, int] | None, excluded: set[int] | None,
            start_decks: list[dict[int, int]] | None = None) -> Prepared:
    """Validate the pool / locks / exclusions (they are stored on `ctx`) and enumerate configurations and land counts."""
    model, fmt = ctx.model, ctx.fmt
    pool = {int(i): int(n) for i, n in pool.items() if n > 0 and 0 <= i < ctx.n_cards and not ctx.is_basic[i]}
    if not pool:
        raise ValueError("the pool contains no cards the model knows")
    warnings: list[str] = []
    pool, ctx.locked, ctx.excluded, lw = apply_constraints(ctx, pool, locked, excluded)
    warnings += lw
    if start_decks:
        keep = [sd for sd in start_decks if all(sd.get(i, 0) >= k for i, k in ctx.locked.items())
                and not any(sd.get(i, 0) > 0 for i in ctx.excluded)]
        if len(keep) < len(start_decks):
            warnings.append("start decks that break the locks / exclusions were ignored")
        start_decks = keep
    configs = [c for c in enumerate_configs(ctx, pool) if c is not None]
    if ctx.locked:
        configs = [c for c in configs if config_allows_locked(ctx, c)]
        if not configs:
            raise ValueError("no color combination can play all the locked cards together "
                             "(too many colors, or more than 3 splash cards): unlock some cards")
    base = supported_lands(model, fmt)
    nls, w = _land_range(configs, base)
    warnings += w
    viable = [c for c in configs if c.castable >= MIN_CASTABLE]
    if not viable:
        viable = sorted(configs, key=lambda c: -c.castable)[:8]
        if not w:
            warnings.append("fewer than 21 castable nonland cards in every color combination")
    prep = Prepared(pool=pool, configs=configs, viable=viable, nls=nls, nl_bounds=(nls[0], nls[-1]), extended=nls[-1] > base[-1],
                    base_lands=base, warnings=warnings, start_decks=list(start_decks or []),
                    cre_bounds=creature_support(model, fmt))
    if ARCH_WINDOW and prep.cre_bounds is not None:
        _prepare_arch_windows(ctx, prep)
    return prep


def _prepare_arch_windows(ctx: Ctx, prep: Prepared) -> None:
    """ARCH_WINDOW: fill `prep.arch_win` and, per configuration, the distinct windows (other than the global one) of the archetype
    groups relevant to it: the groups whose color group is the configuration's main color pair ("other" for mono / 3 colors)."""
    model = ctx.model
    win = group_creature_support(model, ctx.fmt)
    arch = getattr(model, "archetypes", None)
    if not win or arch is None:
        return
    prep.arch_win = {int(a): (int(w[0]), int(w[1])) for a, w in win.items()}
    by_group: dict[str, set[tuple[int, int]]] = {}
    for a, w in prep.arch_win.items():
        if w != prep.cre_bounds and a < len(arch.info):
            by_group.setdefault(str(arch.info[a].get("group", "other")), set()).add(w)
    for cfg in prep.configs:
        g = cfg.main if len(cfg.main) == 2 else "other"
        ws = sorted(by_group.get(g, ()))
        if ws:
            prep.cfg_wins[cfg.key] = ws


def deck_window(ctx: Ctx, prep: Prepared, deck: dict[int, int]) -> tuple[int, int] | None:
    """The creature window a deck must satisfy: the window of the archetype group it is assigned to (ARCH_WINDOW and the group
    has one), else the global one."""
    if not (ARCH_WINDOW and prep.arch_win):
        return prep.cre_bounds
    a, _sim = ctx.model.archetype_index(deck)
    return prep.arch_win.get(int(a), prep.cre_bounds)


def deck_window_meta(model: SetModel, fmt: str, deck: dict[int, int]) -> dict | None:
    """What the GUI can say about the creature range of a built deck: {group, label, lo, hi, own} (own = the archetype group has
    its own window, not the global one).  None unless ARCH_WINDOW and the model has archetypes."""
    if not ARCH_WINDOW or getattr(model, "archetypes", None) is None:
        return None
    glob = creature_support(model, fmt)
    a, _sim = model.archetype_index(deck)
    w = group_creature_support(model, fmt).get(int(a))
    use = w or glob
    if use is None:
        return None
    info = model.archetypes.info[a] if 0 <= a < len(model.archetypes.info) else {}
    return dict(group=int(a), label=info.get("label", ""), lo=int(use[0]), hi=int(use[1]), own=w is not None)


def _cre_avail(ctx: Ctx, prep: Prepared, cfg: Config) -> tuple[int, int, int, int]:
    """(available creature copies, available other copies, locked creatures, locked others) of a configuration: the numbers
    `Searcher` feeds to `relaxed_creature_bounds`."""
    hit = prep._avail.get(cfg.key)
    if hit is None:
        is_cre = ctx.ca.is_creature
        sp_ids = [int(i) for i in sorted(prep.pool) if cfg.cls[i] > 0]
        hit = (sum(min(prep.pool[i], DECK_SIZE) for i in sp_ids if is_cre[i]),
               sum(min(prep.pool[i], DECK_SIZE) for i in sp_ids if not is_cre[i]),
               sum(n for i, n in ctx.locked.items() if not ctx.is_land[i] and is_cre[i]),
               sum(n for i, n in ctx.locked.items() if not ctx.is_land[i] and not is_cre[i]))
        prep._avail[cfg.key] = hit
    return hit


def arch_ok(ctx: Ctx, prep: Prepared, cfg: Config | None, deck: dict[int, int]) -> bool:
    """ARCH_WINDOW validity: the deck's creature count lies in the (relaxed) window of the group it is assigned to.
    Always True without the switch or without group windows."""
    if not (ARCH_WINDOW and prep.arch_win) or cfg is None:
        return True
    win = deck_window(ctx, prep, deck)
    if win is None:
        return True
    nl = nl_of(ctx, deck)
    cre = sum(n for i, n in deck.items() if n > 0 and not ctx.is_land[i] and ctx.ca.is_creature[i])
    lo, hi = relaxed_creature_bounds(win, DECK_SIZE - nl, *_cre_avail(ctx, prep, cfg))
    return lo <= cre <= hi


def run_search(ctx: Ctx, prep: Prepared, n_decks: int = 1, time_budget: float | None = None,
               cancel: threading.Event | None = None, prog: Progress | None = None
               ) -> tuple[list[Cand], list[Cand], list[Config]]:
    """The search under `ctx` (one draw): `_run_search` once under the global creature window (the shipped search).  With
    ARCH_WINDOW (docs/archetype_window/RULE.md) also once per distinct group window (see `_prepare_arch_windows`), restricted to
    the configurations that window is relevant for; the candidates are merged and every one is kept only if it is valid under
    its own group's window (`arch_ok`; the user's start decks are exempt, as they are from every other check)."""
    if not (ARCH_WINDOW and prep.cfg_wins):
        return _run_search(ctx, prep, n_decks, time_budget, cancel, prog)
    import dataclasses

    windows = sorted({w for ws in prep.cfg_wins.values() for w in ws})
    n_runs = 1 + len(windows)
    prog0 = prog or (lambda f, m: None)

    def scaled(k: int) -> Progress:
        return lambda f, m: prog0((k + f) / n_runs, m)

    ranked, spare, top = _run_search(ctx, prep, n_decks, time_budget, cancel, scaled(0))
    seen = {c.key for c in ranked}
    for k, w in enumerate(windows, start=1):
        sub_viable = [c for c in prep.viable if w in prep.cfg_wins.get(c.key, ())]
        if not sub_viable:
            continue
        sub = dataclasses.replace(prep, viable=sub_viable, cre_bounds=w, start_decks=[], warnings=[], cfg_wins={},
                                  _avail=prep._avail)
        try:
            r2, s2, t2 = _run_search(ctx, sub, n_decks, time_budget, cancel, scaled(k))
        except ValueError:                       # no legal deck for these configurations under this window
            continue
        ranked += [c for c in r2 if c.key not in seen]
        seen |= {c.key for c in r2}
        spare += s2
        top += [c for c in t2 if all(c.key != o.key for o in top)]
    ranked.sort(key=lambda c: -c.logit)
    keep = [c for c in ranked if c.src == "start" or arch_ok(ctx, prep, c.cfg, c.deck)]
    if keep:                                      # (never leave a pool without a deck: the global run's best deck stays)
        ranked = keep
    spare = [c for c in spare if arch_ok(ctx, prep, c.cfg, c.deck)]
    return ranked, spare, top


def _run_search(ctx: Ctx, prep: Prepared, n_decks: int = 1, time_budget: float | None = None,
                cancel: threading.Event | None = None, prog: Progress | None = None
                ) -> tuple[list[Cand], list[Cand], list[Config]]:
    """The search under `ctx` (one draw): MILP screening of the configurations, all land counts for the best ones, local search
    with the full model.  Returns (all candidates best first, spare lower-ranked MILP solutions, the configurations that got the
    full land-count enumeration).  `n_decks` > 1 widens it (more configurations, more local-search starts, top-3 MILP solutions)."""
    t0 = time.monotonic()
    pool, nls, nl_bounds = prep.pool, prep.nls, prep.nl_bounds
    configs, viable, extended = prep.configs, prep.viable, prep.extended
    prog = prog or (lambda f, m: None)

    # ---- stage 1: one MILP per configuration at 17 lands; the best ones (by the linear objective) get the full model
    n_screen, cfg_full = (N_SCREEN_DRAW0, N_CONFIG_FULL_DRAW0) if ctx.draw == 0 else (N_SCREEN, N_CONFIG_FULL)
    n_cfg_full = cfg_full if n_decks > 1 else cfg_full - 3
    n_starts = 2 if n_decks <= 1 else min(N_LS_STARTS, n_decks + 2)
    cands: list[Cand] = []
    spare: list[Cand] = []               # lower-ranked MILP solutions, evaluated only if diversity needs them
    stage1: list[tuple[float, Config, MilpSolution, np.ndarray, int]] = []
    bounds = [(_upper_bound(ctx, cfg, pool, max(17, DECK_SIZE - cfg.castable)), cfg) for cfg in viable]
    bounds.sort(key=lambda t: -t[0])
    screen = [cfg for _, cfg in bounds[:n_screen]]
    for k, cfg in enumerate(screen):
        _check(cancel)
        prog(0.02 + 0.13 * k / len(screen), f"Screening configuration {cfg.key}")
        mid_nl = max(17, DECK_SIZE - cfg.castable)
        if mid_nl > nls[-1]:
            continue
        spec, ids = make_spec(ctx, cfg, pool, mid_nl, cre_bounds=prep.cre_bounds)
        sols = solve_milp(spec, k=1)
        if sols:
            stage1.append((sols[0].obj, cfg, sols[0], ids, mid_nl))
    if not stage1:
        raise ValueError("could not build a legal 40-card deck from this pool"
                         + (" with these locks / exclusions" if ctx.locked or ctx.excluded else ""))
    stage1.sort(key=lambda t: -t[0])
    ranked_cfgs: list[tuple[float, Config]] = []
    for k, (_, cfg, sol, ids, mid_nl) in enumerate(stage1):
        _check(cancel)
        prog(0.15 + 0.05 * k / len(stage1), f"Screening configuration {cfg.key}")
        deck = deck_from_solution(ctx, cfg, ids, sol, mid_nl)
        c = Cand(deck, ctx.logit(deck), cfg)
        ranked_cfgs.append((c.logit, cfg))
        cands.append(c)
    ranked_cfgs.sort(key=lambda t: -t[0])
    top_cfgs = [cfg for _, cfg in ranked_cfgs[:(3 if extended else n_cfg_full)]]

    # ---- stage 2: all land counts, top-3 each (top-1 evaluated with the full model right away)
    for k, cfg in enumerate(top_cfgs):
        for nl in nls:
            _check(cancel)
            prog(0.2 + 0.2 * (k * len(nls) + nls.index(nl)) / (len(top_cfgs) * len(nls)), f"Optimising {cfg.key}, {nl} lands")
            spec, ids = make_spec(ctx, cfg, pool, nl, cre_bounds=prep.cre_bounds)
            for j, sol in enumerate(solve_milp(spec, k=3 if n_decks > 1 else 1)):
                deck = deck_from_solution(ctx, cfg, ids, sol, nl)
                if j == 0:
                    cands.append(Cand(deck, ctx.logit(deck), cfg))
                else:
                    spare.append(Cand(deck, float("nan"), cfg))
    for sd in prep.start_decks:
        sdd = {int(i): int(n) for i, n in sd.items() if n > 0}
        cfg = _config_of_deck(ctx, sdd, configs, pool)
        cands.append(Cand(sdd, ctx.logit(sdd), cfg, "start"))
    # dedupe
    uniq: dict[tuple, Cand] = {}
    for c in cands:
        uniq.setdefault(c.key, c)
    cands = sorted(uniq.values(), key=lambda c: -c.logit)

    # ---- stage 3: local search on the best candidate of each of the top configurations
    starts: list[Cand] = []
    per_cfg: dict[str, int] = {}
    for c in cands:
        n = per_cfg.get(c.cfg.key, 0)
        limit = 2 if n_decks > 1 and (not starts or c.cfg.key == starts[0].cfg.key) else 1
        if n < limit:
            starts.append(c)
            per_cfg[c.cfg.key] = n + 1
        if len(starts) >= n_starts:
            break
    for c in cands:
        if c.src == "start" and c not in starts:
            starts.append(c)
    finals: list[Cand] = []
    deadline = None if time_budget is None else t0 + time_budget
    for k, c in enumerate(starts):
        _check(cancel)
        prog(0.4 + 0.45 * k / len(starts), f"Refining {c.cfg.key} deck ({k + 1}/{len(starts)})")
        s = Searcher(ctx, c.cfg, pool, nl_bounds, cancel=cancel, cre_bounds=prep.cre_bounds)
        deck, lg = s.run(c.deck, deadline=deadline, pairs=k < N_PAIR_STARTS)
        finals.append(Cand(deck, lg, c.cfg, "ls"))
    allc: dict[tuple, Cand] = {}
    for c in finals + cands:
        allc.setdefault(c.key, c)
    ranked = sorted(allc.values(), key=lambda c: -c.logit)
    return ranked, spare, top_cfgs


def build(model: SetModel, pool: dict[int, int], fmt: str, n_decks: int = 3, min_diff: int = 10, seed: int = 0,
          progress: Progress | None = None, cancel: threading.Event | None = None,
          start_decks: list[dict[int, int]] | None = None, time_budget: float | None = None,
          locked: dict[int, int] | None = None, excluded: set[int] | None = None,
          families: bool | None = None, n_jobs: int | None = None,
          draw_subset: tuple[int, ...] | None = None) -> BuildResult:
    """Best decks from `pool` ({card_id: copies}, basics ignored).

    `families` (default: True when `n_decks` > 1 and the model has bootstrap draws): deck FAMILIES derived from the model's
    uncertainty (see `deck/families.py`): the search runs under every bootstrap draw, the candidates are clustered by card overlap and
    the result holds one representative per family worth showing (at most `n_decks`), best first, with P(best) and the gap
    to #1 (`BuildResult.families`).  `n_jobs` workers (default: all cores; 1 = in this process) are used for that.
    Otherwise (`n_decks` = 1, or a model without draws): the single best deck, plus - for n_decks > 1 - up to `n_decks - 1`
    more that are >= `min_diff` nonland cards apart.

    `start_decks`: decks (e.g. the user's own) that are always considered; the result is never worse than them.
    `locked`: {card_id: copies} every returned deck must contain (clipped to the copies in the pool);
    `excluded`: card ids no returned deck may contain.  Both are honoured by the MILP (lower bounds / removal from the
    pool), the local search (locked copies are never removed) and the diversity selection.
    `seed` is accepted for API symmetry: the search is deterministic.
    """
    if families is None:
        families = n_decks > 1
    if families and model.n_draws(fmt) <= 1:
        families = False                                  # nothing to derive families from: the classic selection below
    if families:
        from sealed_lab.deck import families as fam_mod

        return fam_mod.build_families(model, pool, fmt, max_families=max(1, n_decks), progress=progress, cancel=cancel,
                                      start_decks=start_decks, locked=locked, excluded=excluded, n_jobs=n_jobs,
                                      draw_subset=draw_subset)

    def prog(f: float, msg: str) -> None:
        if progress:
            progress(min(1.0, f), msg)

    ctx = Ctx(model, fmt)
    prog(0.0, "Enumerating color configurations")
    prep = prepare(ctx, pool, locked, excluded, start_decks)
    pool, nls, nl_bounds, warnings = prep.pool, prep.nls, prep.nl_bounds, list(prep.warnings)
    ranked, spare, top_cfgs = run_search(ctx, prep, n_decks, time_budget, cancel, prog)

    # ---- selection with diversity
    chosen: list[Cand] = []
    for c in ranked:
        if all(deck_diff(ctx, c.deck, o.deck) >= min_diff for o in chosen):
            chosen.append(c)
            if len(chosen) >= n_decks:
                break
    while len(chosen) < n_decks and spare:
        ok = [c for c in spare if all(deck_diff(ctx, c.deck, o.deck) >= min_diff for o in chosen)
              and arch_ok(ctx, prep, c.cfg, c.deck)][:8]
        if not ok:
            break
        for c in ok:
            c.logit = ctx.logit(c.deck)
        best = max(ok, key=lambda c: c.logit)
        chosen.append(best)
        spare.remove(best)
    tries = 0
    while len(chosen) < n_decks and tries < n_decks * 2:
        tries += 1
        prog(0.85 + 0.1 * len(chosen) / n_decks, "Searching for a different deck")
        extra = _diverse_candidate(ctx, pool, top_cfgs, nls, nl_bounds, chosen, min_diff, cancel, prep.cre_bounds)
        if extra is None or not arch_ok(ctx, prep, extra.cfg, extra.deck):
            break
        chosen.append(extra)
    if len(chosen) < n_decks:
        warnings.append(f"only {len(chosen)} deck(s) at least {min_diff} cards apart exist in this pool")

    chosen.sort(key=lambda c: -c.logit)
    if ctx.locked:
        locked_n = sum(ctx.locked.values())
        warnings.append(f"{locked_n} locked card cop{'y' if locked_n == 1 else 'ies'} in every deck")
    prog(0.96, "Evaluating decks")
    evals = [model.evaluate(c.deck, fmt) for c in chosen]
    if ARCH_WINDOW:
        for ev, c in zip(evals, chosen):
            ev.creature_window = deck_window_meta(model, fmt, c.deck)
    p_best = p_best_from_draws(model, [c.deck for c in chosen], fmt)
    diffs = [0] + [deck_diff(ctx, chosen[0].deck, c.deck) for c in chosen[1:]]
    prog(1.0, "Done")
    return BuildResult(set_code=model.set_code, fmt=fmt, decks=evals, p_best=p_best, diffs=diffs, pool_unmatched=[],
                       warnings=warnings)


def apply_constraints(ctx: Ctx, pool: dict[int, int], locked: dict[int, int] | None, excluded: set[int] | None
                      ) -> tuple[dict[int, int], dict[int, int], set[int], list[str]]:
    """Validate locks / exclusions against the pool. Returns (pool without excluded cards, locked, excluded, warnings)."""
    warnings: list[str] = []
    exc = {int(i) for i in (excluded or ()) if int(i) in pool}
    lk: dict[int, int] = {}
    for i, n in (locked or {}).items():
        i, n = int(i), int(n)
        if n <= 0 or i < 0 or i >= ctx.n_cards or ctx.is_basic[i]:
            continue
        if i not in pool:
            warnings.append(f"locked card {ctx.names[i]!r} is not in the pool and was ignored")
            continue
        if i in exc:
            raise ValueError(f"{ctx.names[i]!r} is both locked and excluded")
        if n > pool[i]:
            warnings.append(f"only {pool[i]} cop{'y' if pool[i] == 1 else 'ies'} of {ctx.names[i]!r} in the pool "
                            f"(lock reduced from {n})")
            n = pool[i]
        lk[i] = n
    if sum(n for i, n in lk.items() if not ctx.is_land[i]) > DECK_SIZE - min(NL_RANGE):
        raise ValueError("too many locked cards: a deck needs room for lands")
    return {i: n for i, n in pool.items() if i not in exc}, lk, exc, warnings


def config_allows_locked(ctx: Ctx, cfg: Config) -> bool:
    """Can every locked card be played in this color configuration (splash cap included)?"""
    sp = 0
    for i in ctx.locked:
        if ctx.is_land[i]:
            if not cfg.land_ok[i]:
                return False
        elif cfg.cls[i] == 0:
            return False
        elif cfg.cls[i] == 2:
            sp += ctx.locked[i]
    return sp <= SPLASH_CAP


def _upper_bound(ctx: Ctx, cfg: Config, pool: dict[int, int], n_lands: int) -> float:
    """Cheap upper bound of the MILP objective (best spells by linear value, floors / land choices ignored)."""
    ids = np.array([i for i in pool if cfg.cls[i] > 0], dtype=np.int64)
    if len(ids) == 0:
        return -1e18
    lin = ctx.lin_spell(ids, n_lands)
    if cfg.splash:
        lin = lin + np.where(cfg.cls[ids] == 2, ctx.splash_term(ids, cfg.sc), 0.0)
    cnt = np.array([pool[int(i)] for i in ids])
    order = np.argsort(-lin)
    need, tot, splash_left = DECK_SIZE - n_lands, 0.0, SPLASH_CAP
    for j in order:
        take = min(int(cnt[j]), need)
        if cfg.cls[ids[j]] == 2:
            take = min(take, splash_left)
            splash_left -= take
        tot += take * lin[j]
        need -= take
        if need <= 0:
            break
    return tot - 1e6 * max(need, 0)


def _config_of_deck(ctx: Ctx, deck: dict[int, int], configs: list[Config], pool: dict[int, int]) -> Config:
    """The configuration (from `configs`) that explains a given deck best (fewest unplayable cards)."""
    best, best_bad = None, 1e9
    for cfg in configs:
        bad = sum(n for i, n in deck.items() if not ctx.is_basic[i] and
                  ((ctx.is_land[i] and not cfg.land_ok[i]) or (not ctx.is_land[i] and cfg.cls[i] == 0)))
        bad += max(0, sum(n for i, n in deck.items() if cfg.cls[i] == 2) - SPLASH_CAP)
        if bad < best_bad:
            best, best_bad = cfg, bad
    assert best is not None
    if best_bad > 0:
        # no configuration allows the whole deck: make a permissive one for local search around it
        cls = best.cls.copy()
        for i in deck:
            if not ctx.is_land[i] and not ctx.is_basic[i]:
                cls[i] = max(cls[i], 1)
        land_ok = best.land_ok.copy()
        for i in deck:
            if ctx.is_land[i] and not ctx.is_basic[i]:
                land_ok[i] = True
        best = Config(best.main, best.splash, cls, land_ok, {}, best.castable)
    return best


def _diverse_candidate(ctx: Ctx, pool: dict[int, int], cfgs: list[Config], nls: tuple[int, ...],
                       nl_bounds: tuple[int, int], chosen: list[Cand], min_diff: int,
                       cancel: threading.Event | None, cre_bounds: tuple[int, int] | None = None) -> Cand | None:
    """Best deck that is >= min_diff nonland cards away from every chosen deck (MILP distance constraint + local search)."""
    pool_cands: list[Cand] = []
    nls = tuple(n for n in nls if n >= 16) or nls
    for cfg in cfgs[:5]:
        for nl in nls:
            _check(cancel)
            spec, ids = make_spec(ctx, cfg, pool, nl, cre_bounds=cre_bounds)
            apart = []
            for o in chosen:
                other = np.array([o.deck.get(int(i), 0) if not ctx.is_land[i] else 0 for i in ids], dtype=int)
                apart.append((other, min_diff))
            sols = solve_milp(spec, k=1, apart=apart)
            if sols:
                deck = deck_from_solution(ctx, cfg, ids, sols[0], nl)
                pool_cands.append(Cand(deck, ctx.logit(deck), cfg))
    if not pool_cands:
        return None
    pool_cands.sort(key=lambda c: -c.logit)

    def legal(d: dict[int, int]) -> bool:
        return all(deck_diff(ctx, d, o.deck) >= min_diff for o in chosen)

    best: Cand | None = None
    done_cfg: dict[str, int] = {}
    for c in pool_cands:
        if done_cfg.get(c.cfg.key, 0) >= 1 or len(done_cfg) >= 2:
            continue
        done_cfg[c.cfg.key] = done_cfg.get(c.cfg.key, 0) + 1
        s = Searcher(ctx, c.cfg, pool, nl_bounds, legal=legal, cancel=cancel, cre_bounds=cre_bounds)
        d, lg = s.run(c.deck)
        if best is None or lg > best.logit:
            best = Cand(d, lg, c.cfg, "diverse")
    return best


def p_best_from_draws(model: SetModel, decks: list[dict[int, int]], fmt: str) -> list[float]:
    """P(deck i has the highest win probability among `decks`) over the bootstrap draws (draw 0 only if no bootstrap)."""
    if len(decks) == 1:
        return [1.0]
    L = np.stack([np.asarray(model.logit(d, fmt, draws=True), dtype=float) for d in decks])   # (n_decks, B)
    if L.shape[1] > 1:
        L = L[:, 1:]
    win = L.argmax(0)
    return [float((win == i).mean()) for i in range(len(decks))]


# =============================================================================================
# model cache + text entry point
# =============================================================================================

_MODELS: dict[str, tuple[int, SetModel]] = {}
_MODELS_LOCK = threading.Lock()


def model_stamp(set_code: str) -> int | None:
    """Modification time (ns) of the set's model file, None without a model."""
    try:
        return (config.model_dir(set_code.upper()) / "meta.json").stat().st_mtime_ns
    except OSError:
        return None


def load_model(set_code: str) -> SetModel:
    """`SetModel.load` cached until the model files change."""
    set_code = set_code.upper()
    stamp = model_stamp(set_code)
    if stamp is None:
        raise FileNotFoundError(f"no model for set {set_code}: run Calculate on the Sets tab first")
    with _MODELS_LOCK:
        hit = _MODELS.get(set_code)
        if hit is not None and hit[0] == stamp:
            return hit[1]
        m = SetModel.load(set_code)
        m._source_stamp = stamp                     # type: ignore[attr-defined]  # lets worker processes load the same model
        _MODELS.clear()
        _MODELS[set_code] = (stamp, m)
        return m


def _resolve_cards(index: NameIndex, spec, what: str) -> tuple[dict[int, int], list[str]]:
    """Locked / excluded cards given as names or ids (dict {card: copies}, list / set of cards, or a single card)."""
    if spec is None:
        return {}, []
    if isinstance(spec, (str, int)):
        spec = [spec]
    items = spec.items() if isinstance(spec, dict) else ((c, 1) for c in spec)
    out: dict[int, int] = {}
    bad: list[str] = []
    for card, n in items:
        if isinstance(card, (int, np.integer)):
            cid: int | None = int(card)
        else:
            m = index.match(str(card))
            cid = m.card_id if m.ok else None
        if cid is None or cid not in index.names:
            bad.append(str(card))
            continue
        out[cid] = out.get(cid, 0) + int(n)
    return out, ([f"{what} card(s) not recognised and ignored: " + ", ".join(bad)] if bad else [])


def build_pool(set_code: str, text: str, fmt: str = "bo3", n_decks: int = 3, min_diff: int = 10,
               progress: Progress | None = None, cancel: threading.Event | None = None,
               locked: dict | list | set | None = None, excluded: dict | list | set | None = None,
               families: bool | None = None, n_jobs: int | None = None,
               name_overrides: dict[str, str] | None = None) -> BuildResult:
    """Parse an Arena pool export and build the best decks (entry point used by `api.build_pool`).

    `locked` ({name or id: copies}, or a list of names / ids = 1 copy each) must be in every deck; `excluded`
    (names or ids) in none.  `name_overrides` ({pasted name: confirmed card name, "" = drop the card}) replaces the names of the
    text the user confirmed or corrected (see `names.analyse_pool`).  When the text is an Arena deck export with a Deck AND a
    Sideboard section, its Deck section is the user's own build: it is evaluated too (`BuildResult.user_deck`)."""
    model = load_model(set_code)
    if fmt not in model.formats:
        raise ValueError(f"format {fmt!r} is not available for {model.set_code}; available: {', '.join(model.formats)}")
    parsed = arena_mod.parse_arena_full(text)
    named = names_mod.apply_overrides(arena_mod.pool_of(parsed.main, parsed.side), name_overrides)
    if not named:
        raise ValueError("no cards found in the pool text (paste an Arena export: lines like '1 Card Name')")
    index = NameIndex(model.cards)
    res = resolve_names(named, index, drop_basics=True)
    if not res.counts:
        raise ValueError("none of the card names in the pool were recognised for set " + model.set_code)
    lk, w1 = _resolve_cards(index, locked, "locked")
    ex, w2 = _resolve_cards(index, excluded, "excluded")
    out = build(model, res.counts, fmt, n_decks=n_decks, min_diff=min_diff, progress=progress, cancel=cancel,
                locked=lk or None, excluded=set(ex) or None, families=families, n_jobs=n_jobs)
    out.pool_unmatched = list(res.unmatched)
    out.pool = {int(k): int(v) for k, v in res.counts.items()}
    out.warnings = list(res.warnings) + w1 + w2 + out.warnings
    if parsed.ignored:
        out.warnings.append(f"{len(parsed.ignored)} line(s) could not be read: " + "; ".join(parsed.ignored[:3]))
    if parsed.had_headers and parsed.main and parsed.side:       # the user's own build: Deck section of an Arena deck export
        rm = resolve_names(names_mod.apply_overrides(parsed.main, name_overrides), index)
        out.user_unmatched = list(rm.unmatched)
        if rm.counts:
            try:
                out.user_deck = model.evaluate({int(k): int(v) for k, v in rm.counts.items() if v > 0}, fmt)
            except Exception as e:  # noqa: BLE001 - the pool builds are still returned
                log.warning("could not evaluate the pasted deck: %s", e)
                out.warnings.append(f"your own deck could not be evaluated: {e}")
    return out


def build_ids(set_code: str, pool: dict[int, int], fmt: str = "bo3", n_decks: int = 1, min_diff: int = 10,
              progress: Progress | None = None, cancel: threading.Event | None = None,
              locked: dict | list | set | None = None, excluded: dict | list | set | None = None,
              families: bool | None = None, n_jobs: int | None = None,
              draw_subset: tuple[int, ...] | None = None) -> BuildResult:
    """Like `build_pool` for a pool already resolved to {card_id: copies} (used by the visual builder's rebuild)."""
    model = load_model(set_code)
    if fmt not in model.formats:
        raise ValueError(f"format {fmt!r} is not available for {model.set_code}; available: {', '.join(model.formats)}")
    index = NameIndex(model.cards)
    lk, w1 = _resolve_cards(index, locked, "locked")
    ex, w2 = _resolve_cards(index, excluded, "excluded")
    out = build(model, pool, fmt, n_decks=n_decks, min_diff=min_diff, progress=progress, cancel=cancel,
                locked=lk or None, excluded=set(ex) or None, families=families, n_jobs=n_jobs,
                draw_subset=draw_subset)
    out.pool = {int(k): int(v) for k, v in pool.items() if v > 0 and not index.is_basic.get(int(k), False)}
    out.warnings = w1 + w2 + out.warnings
    return out
