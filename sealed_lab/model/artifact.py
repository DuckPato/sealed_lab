"""The deployable model artifact (`APP_HOME/sets/<SET>/model/`) and the `SetModel` that serves it."""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path

import numpy as np
import polars as pl
from scipy.special import expit

from sealed_lab import CODE_VERSION, config
from sealed_lab.model import castability, structure
from sealed_lab.model import manasim
from sealed_lab.model.manasim import CardArrays, SimRegime
from sealed_lab.types import CardLine, DeckEval

log = logging.getLogger(__name__)

ARTIFACT_VERSION = 1
LOW_DATA_EXPOSURE = 2000.0
INTERVAL = (10.0, 90.0)


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------

def run_distribution(p: float, wins: int, losses: int) -> tuple[float, float]:
    """Exact Markov chain of a run that ends at `wins` wins or `losses` losses with per-round win prob `p`.
    Returns (expected wins at the end of the run, P(reach `wins` wins))."""
    prob = np.zeros((wins + 1, losses + 1))
    prob[0, 0] = 1.0
    exp_w = 0.0
    p_max = 0.0
    for w in range(wins + 1):
        for l in range(losses + 1):
            pr = prob[w, l]
            if pr == 0.0:
                continue
            if w == wins:
                exp_w += pr * w
                p_max += pr
                continue
            if l == losses:
                exp_w += pr * w
                continue
            prob[w + 1, l] += pr * p
            prob[w, l + 1] += pr * (1 - p)
    return float(exp_w), float(p_max)


def match_win_prob(p: float) -> float:
    """Best-of-three match win probability from the per-game probability."""
    return p * p * (3 - 2 * p)


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------------------------
# SetModel
# ---------------------------------------------------------------------------------------------

class SetModel:
    set_code: str
    formats: list[str]
    cards: pl.DataFrame
    feature_names: list[str]

    def __init__(self, set_code: str, meta: dict, cards: pl.DataFrame, draws: dict[str, dict[str, np.ndarray]],
                 reference: dict[str, np.ndarray], extras: dict | None = None):
        self.set_code = set_code
        self.meta = meta
        self.formats = list(meta["formats"])
        self.cards = cards
        self.feature_names = list(meta["feature_names"])
        self.mu = np.asarray(meta["phi_mean"], dtype=float)
        self.sd = np.asarray(meta["phi_sd"], dtype=float)
        self._draws = draws
        self._ref = reference
        self._ca = CardArrays.from_df(cards)
        self._names = cards.sort("card_id")["name"].to_list()
        self._is_basic = cards.sort("card_id")["is_basic"].to_numpy()
        self._rarity = cards.sort("card_id")["rarity"].to_list()
        self._n_struct = structure.N_STRUCT
        # mana simulator of the deployed model: v2 (hand / draw regime of each format) or v1
        self.sim_version = int((meta.get("manasim") or {}).get("version", 1))
        self.sim_regimes = {r: SimRegime.make(r, k) for r, k in ((meta.get("manasim") or {}).get("keep") or {}).items()}
        cm = meta.get("castability") or {}
        self.cast_applied = bool(cm.get("applied"))          # castability-weighted card copies + the dead-cards feature (model "MK")
        self.cast_meta = cm
        self.cast_trials = int(cm.get("n_trials") or castability.manasim.DEFAULT_TRIALS)
        self._cc: castability.CastCards | None = None
        self._cast_cache: dict = {}
        self._mull_cache: dict = {}
        self._col = {}
        for f in self.formats:
            for c in ("exposure", "low_data"):
                self._col[(f, c)] = cards.sort("card_id")[f"{c}_{f}"].to_numpy()
        self._setup_extras(extras or {})

    # ----------------------------------------------------------------- optional layers
    def _setup_extras(self, ex: dict) -> None:
        """Pair synergies, card x archetype values, archetype clusters, opponent matchups (all optional, from the artifact)."""
        n = len(self._names)
        self.load_warnings: list[str] = []         # optional layers that were damaged and are ignored (shown with the deck warnings)
        if ex.get("_errors"):
            self.load_warnings.append("some optional model files could not be read and their layers are ignored "
                                      f"({ex['_errors']}); recalculate the set")
        self.pairs = ex.get("pairs")                  # dict(ab (n,2), games (n,), se {fmt: (n,)}, lift (n,)) or None
        self.archcols = ex.get("archcols")            # dict(ca (m,2), se {fmt: (m,)})
        self.archetypes = ex.get("archetypes")        # Archetypes or None
        self.opp = ex.get("opp")                      # opponent.OppModel or None
        self.playdraw = ex.get("playdraw")            # playdraw.PlayDrawModel or None
        if self.playdraw is None and (self.meta.get("playdraw") or {}).get("applied"):
            self.load_warnings.append("the play/draw layer is deployed but playdraw.npz is missing or damaged: play/draw advice is "
                                      "unavailable; recalculate the set")
        if self.opp is None and (self.meta.get("matchups") or {}).get("applied"):
            self.load_warnings.append("the matchup layer is deployed but opp.npz is missing or damaged: matchup information is "
                                      "unavailable; recalculate the set")
        mm = self.meta.get("matchups") or {}
        self.opp_formats: list[str] = list(mm["applied_formats"]) if "applied_formats" in mm else list(self.formats)  # (older models: all)
        self._spell = ~np.asarray(self._ca.is_land, dtype=bool) & ~np.asarray(self._is_basic, dtype=bool)
        self._cf = np.zeros(n)
        self._arch_cols: dict[int, np.ndarray] = {}
        if self.archcols is not None:
            ca = self.archcols["ca"]
            for a in np.unique(ca[:, 1]):
                self._arch_cols[int(a)] = np.nonzero(ca[:, 1] == a)[0]
        self._extra_fmts = {f for f in self.formats if f"pair" in self._draws[f] or "arch" in self._draws[f]}

    def has_playdraw(self, fmt: str | None = None) -> bool:
        """The play/draw layer (games-2-3 advice only) is deployed (and, with `fmt`, `fmt` is a format of this model)."""
        return self.playdraw is not None and (fmt is None or fmt in self.formats)

    def playdraw_effect(self, on_play: bool, draws: bool = True) -> np.ndarray:
        """Per-card logit shift of a game on the play (+gamma/2) or on the draw (-gamma/2): (B, n_cards) or (n_cards,) for draw 0.
        Zeros without the layer.  It never enters q(D) (game 1, unknown play/draw); only games-2-3 advice adds it to the card effects."""
        n = len(self._names)
        if self.playdraw is None:
            return np.zeros((self._draws[self.formats[0]]["beta"].shape[0], n)) if draws else np.zeros(n)
        return self.playdraw.effect(on_play, draws)

    def has_opp(self, fmt: str) -> bool:
        """The matchup layer is deployed AND applied in `fmt` (a pooled decision can apply it in Bo3 but not in Bo1)."""
        return self.opp is not None and fmt in self.opp_formats

    def has_pairs(self, fmt: str | None = None) -> bool:
        return self.pairs is not None and (fmt is None or "pair" in self._draws.get(fmt, {}))

    def has_extra(self, fmt: str) -> bool:
        return fmt in self._extra_fmts

    def archetype_index(self, deck: dict[int, int]) -> tuple[int, float]:
        if self.archetypes is None:
            return -1, 0.0
        i, sim, _ = self.archetypes.assign(deck, self._spell)
        return i, sim

    def _extra_terms(self, deck: dict[int, int], fmt: str, draws: bool) -> tuple[np.ndarray, np.ndarray]:
        """(pair term, archetype-conditional term), each (B,) logit, for a deck {card_id: copies} (zeros if not deployed)."""
        dr = self._draws[fmt]
        B = dr["beta"].shape[0] if draws else 1
        pt = np.zeros(B)
        at = np.zeros(B)
        if "pair" in dr and self.pairs is not None:
            cf = np.zeros(len(self._names))
            ids = [c for c, v in deck.items() if v > 0 and 0 <= c < len(cf)]
            for c in ids:
                cf[c] = deck[c]
            ab = self.pairs["ab"]
            vals = np.minimum(cf[ab[:, 0]], cf[ab[:, 1]])
            coef = dr["pair"] if draws else dr["pair"][:1]
            pt = coef @ vals
        if "arch" in dr and self.archcols is not None and self.archetypes is not None:
            a, _ = self.archetype_index(deck)
            cols = self._arch_cols.get(a)
            if cols is not None and len(cols):
                ca = self.archcols["ca"][cols]
                v = np.array([deck.get(int(c), 0) for c in ca[:, 0]], dtype=float)
                coef = dr["arch"][:, cols] if draws else dr["arch"][:1, cols]
                at = coef @ v
        return pt, at

    # ----------------------------------------------------------------- mulligans
    def has_mull(self, fmt: str) -> bool:
        """The model carries the mulligan coefficients (artifacts written before this term existed do not)."""
        return "mull" in self._draws.get(fmt, {})

    def mull_probs(self, n_lands: int, n_cards: int, fmt: str) -> tuple[float, float]:
        """(P(exactly one mulligan), P(two or more)) of a deck with `n_lands` lands in `n_cards` cards: the dealing law of the
        format's queue with the set's measured keep rule (`manasim.mull_probs`). A function of the land count only."""
        key = (fmt, int(n_lands), int(n_cards))
        hit = self._mull_cache.get(key)
        if hit is None:
            hit = self._mull_cache[key] = manasim.mull_split(int(n_lands), int(n_cards), self.cast_regime(fmt))
        return hit

    def deck_lands(self, deck: dict[int, int]) -> tuple[int, int]:
        """(land cards incl. basics, cards) of a deck {card_id: copies}."""
        n = nl = 0
        for c, v in deck.items():
            if v <= 0:
                continue
            n += int(v)
            if 0 <= c < self._ca.n and bool(self._ca.is_land[c]):
                nl += int(v)
        return nl, n

    def mull_cost(self, deck: dict[int, int], fmt: str, draws: bool = False) -> np.ndarray:
        """Expected logit cost of the mulligans THIS deck makes (shape (B,) or (1,)): P(1) coef[mull1] + P(2+) coef[mull2p].

        The model is fitted with the games' actual mulligans as covariates; the prediction for a deck must add the mulligans the
        deck itself makes (a land count away from 16-17 makes more of them), otherwise q(D) is the win rate of a deck that never
        mulligans. Zeros for an artifact without the coefficients."""
        dr = self._draws[fmt]
        B = dr["beta"].shape[0] if draws else 1
        if "mull" not in dr:
            return np.zeros(B)
        nl, n = self.deck_lands(deck)
        p1, p2 = self.mull_probs(nl, n, fmt)
        m = np.asarray(dr["mull"] if draws else dr["mull"][:1], dtype=float)
        return m[:, 0] * p1 + m[:, 1] * p2

    # ----------------------------------------------------------------- loading
    @classmethod
    def load(cls, set_code: str) -> "SetModel":
        set_code = set_code.upper()
        d = config.model_dir(set_code)
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        cards = pl.read_parquet(d / "cards.parquet")
        z = np.load(d / "draws.npz", allow_pickle=False)
        draws = {f: dict(beta=z[f"{f}__beta"], theta=z[f"{f}__theta"], ref=z[f"{f}__ref"]) for f in meta["formats"]}
        for f in meta["formats"]:
            for k in ("pair", "arch", "mull"):
                if f"{f}__{k}" in z.files:
                    draws[f][k] = z[f"{f}__{k}"].astype(float)
        r = np.load(d / "reference.npz", allow_pickle=False)
        reference = {f: r[f"{f}__q"] for f in meta["formats"]}
        extras = load_extras(d, meta)
        # draws that carry pair / archetype coefficients need their column definitions: without them the logit and the
        # reference distribution would silently disagree, so a damaged file is an error, not a warning
        has_pair = any("pair" in draws[f] for f in draws)
        has_arch = any("arch" in draws[f] for f in draws)
        if has_pair and "pairs" not in extras:
            raise ValueError(f"{set_code}: the model's draws carry card-pair coefficients but pairs.npz is missing or damaged "
                             f"({extras.get('_errors') or 'file not found'}); recalculate the set")
        if has_arch and not ("archcols" in extras and "archetypes" in extras):
            raise ValueError(f"{set_code}: the model's draws carry archetype-conditional coefficients but archcols.npz / the "
                             f"archetype files are missing or damaged ({extras.get('_errors') or 'file not found'}); recalculate the set")
        try:                       # the corpus rebuilt after the model (interrupted / later calculation): its card ids may differ
            man = json.loads((config.corpus_dir(set_code) / "manifest.json").read_text(encoding="utf-8"))
            built = meta.get("built_corpus_at") or meta.get("corpus_built_at")
            if built and man.get("built_at") and built != man["built_at"]:
                log.warning("%s: the corpus was rebuilt after the model (%s vs %s); recalculate the set", set_code, man["built_at"], built)
        except (OSError, ValueError, AttributeError):
            pass
        return cls(set_code, meta, cards, draws, reference, extras=extras)

    # ----------------------------------------------------------------- features
    @property
    def card_arrays(self) -> CardArrays:
        return self._ca

    def sim_spec(self, fmt: str | None = None) -> tuple[int, "SimRegime | str | None"]:
        """(mana simulator version, hand/draw regime incl. the set's keep rule) the deployed model uses for format `fmt`."""
        if self.sim_version == 1:
            return 1, None
        name = config.format_regime(fmt or self.formats[0])
        return 2, self.sim_regimes.get(name, name)

    def featurize(self, deck: dict[int, int], fmt: str | None = None) -> np.ndarray:
        """phi(D) with `feature_names` (the dead-cards feature last when castability is deployed)."""
        v, r = self.sim_spec(fmt)
        phi = structure.featurize(deck, self._ca, version=v, regime=r)
        if self.cast_applied:
            phi = np.append(phi, self.cast(deck, fmt or self.formats[0]).dead)
        return phi

    # ----------------------------------------------------------------- castability
    @property
    def cast_cards(self) -> "castability.CastCards":
        if self._cc is None:
            self._cc = castability.CastCards.from_df(self.cards)
        return self._cc

    def cast_regime(self, fmt: str):
        """Hand / draw regime (with the set's measured keep rule) castability is computed under for a format."""
        name = config.format_regime(fmt)
        return self.sim_regimes.get(name, name)

    def cast(self, deck: dict[int, int], fmt: str) -> "castability.CastResult":
        """Castability of every card of `deck` under the format's hand / draw regime (cached by the deck's source / target classes)."""
        if len(self._cast_cache) > 60000:
            self._cast_cache.clear()
        return castability.analyze_deck(deck, self.cast_cards, self.cast_regime(fmt), n_trials=self.cast_trials, cache=self._cast_cache)

    def support_reference(self, fmt: str) -> dict | None:
        return ((self.cast_meta.get("support") or {}).get(fmt))

    def support_flags(self, deck: dict[int, int], fmt: str) -> list[str]:
        return castability.support_flags(deck, self.cast_cards, self.support_reference(fmt), (self.meta.get("land_support") or {}).get(fmt))

    def deck_weights(self, deck: dict[int, int], fmt: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(card ids, copies, effective copies) of the deck's cards the model knows: effective = copies x castability when
        castability is deployed, else the copies."""
        self._check_fmt(fmt)
        n_cards = self._draws[fmt]["beta"].shape[1]
        d = {int(k): int(v) for k, v in deck.items() if v > 0 and 0 <= int(k) < n_cards}
        ids = np.fromiter(d.keys(), dtype=np.int64, count=len(d))
        cnt = np.fromiter(d.values(), dtype=float, count=len(d))
        if not self.cast_applied or not len(ids):
            return ids, cnt, cnt.copy()
        return ids, cnt, self.cast(d, fmt).weights

    def copy_contribution(self, cid: int, fmt: str, p: float = 1.0) -> float:
        """Logit contribution (draw 0) of ONE copy of card `cid` that is castable on time with probability `p`.

        Without castability: the card's beta.  With it the model is sum_c beta_c n_c p_c + theta_dead (dead - mu) / sd, dead =
        sum n_c (1 - p_c), so a copy contributes beta p + theta_dead (1 - p) / sd_dead (p = 1 for lands and for decks without the
        castability layer). The other cards' castabilities moving when a card is added or removed are NOT part of it."""
        b = float(self._draws[fmt]["beta"][0][cid])
        if not self.cast_applied or bool(self._ca.is_land[cid]):
            return b
        w = float(self._draws[fmt]["theta"][0][-1]) / float(self.sd[-1])
        return b * p + w * (1.0 - p)

    def add_values(self, deck: dict[int, int], fmt: str, cids: list[int] | None = None) -> dict[int, dict]:
        """Value of ADDING one copy of each card in `cids` (default: every non-basic card) to this deck, with its castability in
        THIS deck (the engine is run on the deck plus the card): {card_id: dict(value_pp, castable, contrib, low_data)}.

        value_pp = pp change of the game win rate from the card's own contribution (`copy_contribution`); everything else equal."""
        deck = {int(k): int(v) for k, v in deck.items() if v > 0}
        self._check_fmt(fmt)
        q0 = float(self.logit(deck, fmt))
        n_known = self._draws[fmt]["beta"].shape[1]
        expo = self._col[(fmt, "exposure")]
        out: dict[int, dict] = {}
        for c in (range(n_known) if cids is None else cids):
            c = int(c)
            if not 0 <= c < min(n_known, self._ca.n) or bool(self._is_basic[c]):
                continue
            p = 1.0
            if self.cast_applied and not bool(self._ca.is_land[c]):
                try:
                    nd = dict(deck)
                    nd[c] = nd.get(c, 0) + 1
                    r = self.cast(nd, fmt)
                    p = float(r.p[int(np.nonzero(r.ids == c)[0][0])])
                except Exception:  # noqa: BLE001 - a missing castability only costs precision of a badge
                    log.debug("castability of a hypothetical addition failed", exc_info=True)
                    p = 1.0
            ct = self.copy_contribution(c, fmt, p)
            out[c] = dict(value_pp=100.0 * float(expit(q0 + ct) - expit(q0)), castable=None if bool(self._ca.is_land[c]) or not self.cast_applied else p,
                          contrib=ct, low_data=bool(expo[c] < LOW_DATA_EXPOSURE))
        return out

    def standardise(self, phi: np.ndarray) -> np.ndarray:
        return (phi - self.mu) / self.sd

    def n_draws(self, fmt: str) -> int:
        return self._draws[fmt]["beta"].shape[0]

    def _check_fmt(self, fmt: str) -> None:
        if fmt not in self._draws:
            raise KeyError(f"format {fmt!r} not available for {self.set_code}; have {self.formats}")

    # ----------------------------------------------------------------- predictions
    def _parts(self, deck: dict[int, int], fmt: str, draws: bool):
        self._check_fmt(fmt)
        dr = self._draws[fmt]
        ids = np.fromiter((k for k, v in deck.items() if v > 0), dtype=np.int64)
        cnt = np.fromiter((v for k, v in deck.items() if v > 0), dtype=float)
        keep = (ids >= 0) & (ids < dr["beta"].shape[1])
        ids, cnt = ids[keep], cnt[keep]
        v, rg = self.sim_spec(fmt)
        phi = structure.featurize(deck, self._ca, version=v, regime=rg)
        w = cnt
        if self.cast_applied:
            cr = self.cast({int(i): int(c) for i, c in zip(ids, cnt)}, fmt)
            phi = np.append(phi, cr.dead)
            w = cr.weights if len(ids) else cnt
        z = self.standardise(phi)
        beta, theta, ref = dr["beta"], dr["theta"], dr["ref"]
        if not draws:
            beta, theta, ref = beta[:1], theta[:1], ref[:1]
        cards = beta[:, ids] @ w if len(ids) else np.zeros(beta.shape[0])
        ns = self._n_struct
        struct = theta[:, :ns] @ z[:ns]
        mana = theta[:, ns:] @ z[ns:]
        return ref, cards, struct, mana, phi, ids, cnt

    def logit(self, deck: dict[int, int], fmt: str, draws: bool = False):
        ref, cards, struct, mana, *_ = self._parts(deck, fmt, draws)
        L = ref + cards + struct + mana
        if fmt in self._extra_fmts:
            pt, at = self._extra_terms(deck, fmt, draws)
            L = L + pt + at
        L = L + self.mull_cost(deck, fmt, draws)
        return L if draws else float(L[0])

    def q(self, deck: dict[int, int], fmt: str, draws: bool = False):
        L = self.logit(deck, fmt, draws)
        return expit(L)

    def percentile(self, q: float, fmt: str, draw: int = 0) -> float:
        """Percentile (0..100, mid-rank) of `q` among the real builds of the target format (reference draw `draw`)."""
        self._check_fmt(fmt)
        r = self._ref[fmt][draw]
        lo = np.searchsorted(r, q, side="left")
        hi = np.searchsorted(r, q, side="right")
        return float(100.0 * (lo + hi) / (2.0 * len(r)))

    def percentiles(self, q: np.ndarray, fmt: str) -> np.ndarray:
        """Per-draw percentiles for per-draw q (same draw index for the reference)."""
        ref = self._ref[fmt]
        out = np.empty(len(q))
        for b in range(len(q)):
            r = ref[min(b, ref.shape[0] - 1)]
            out[b] = 100.0 * (np.searchsorted(r, q[b], side="left") + np.searchsorted(r, q[b], side="right")) / (2.0 * len(r))
        return out

    def card_beta(self, fmt: str) -> np.ndarray:
        self._check_fmt(fmt)
        return self._draws[fmt]["beta"][0].copy()

    def card_se(self, fmt: str) -> np.ndarray:
        return self.cards.sort("card_id")[f"se_{fmt}"].to_numpy()

    def card_name(self, card_id: int) -> str:
        return self._names[card_id]

    # ----------------------------------------------------------------- evaluation
    def evaluate(self, deck: dict[int, int], fmt: str) -> DeckEval:
        deck = {int(k): int(v) for k, v in deck.items() if v > 0}
        ref, cards, struct, mana, phi, ids, cnt = self._parts(deck, fmt, True)
        L = ref + cards + struct + mana
        has_x = fmt in self._extra_fmts
        if has_x:
            pt, at = self._extra_terms(deck, fmt, True)
            L = L + pt + at
        mull = self.mull_cost(deck, fmt, True)
        L = L + mull
        q = expit(L)
        q0 = float(q[0])
        lo_p, hi_p = INTERVAL
        if len(q) >= 3:
            rep = q[1:]
            lo, hi = float(np.percentile(rep, lo_p)), float(np.percentile(rep, hi_p))
            pct = self.percentiles(q, fmt)
            s_lo, s_hi = float(np.percentile(pct[1:], lo_p)), float(np.percentile(pct[1:], hi_p))
        else:
            lo = hi = q0
            pct = self.percentiles(q, fmt)
            s_lo = s_hi = float(pct[0])
        lo, hi = min(lo, q0), max(hi, q0)
        score = float(pct[0])
        s_lo, s_hi = min(s_lo, score), max(s_hi, score)
        wins, losses = config.RUN_RULES[fmt]
        p_round = match_win_prob(q0) if fmt == "bo3" else q0
        exp_w, p_max = run_distribution(p_round, wins, losses)
        colors, splash, needs = structure.deck_colors(deck, self._ca)
        cast_err = ""
        try:
            cres = self.cast(deck, fmt)
            cast_p = {int(i): float(p) for i, p in zip(cres.ids, cres.p)}
        except Exception as e:  # noqa: BLE001 - castability is informational here; never fail an evaluation over it
            log.exception("castability of the deck could not be computed")
            cast_p = {}
            cast_err = f"castability of this deck's cards could not be computed ({type(e).__name__}: {e}); the castability column is empty"
        exposure = self._col[(fmt, "exposure")]
        lines: list[CardLine] = []
        warnings: list[str] = []
        L0 = float(L[0])
        for cid in sorted(deck, key=lambda c: (bool(self._ca.is_land[c]) if c < self._ca.n else True,
                                                float(self._ca.mv[c]) if c < self._ca.n else 0.0, self._names[c] if c < len(self._names) else "")):
            n = deck[cid]
            if cid >= self._ca.n:
                continue
            basic = bool(self._is_basic[cid])
            if basic:
                lines.append(CardLine(card_id=cid, name=self._names[cid], count=n, value_pp=None, low_data=False))
                continue
            # the copy's own contribution under the deployed model (castability-weighted beta + the dead-cards term)
            ct = self.copy_contribution(cid, fmt, cast_p.get(cid, 1.0) if not bool(self._ca.is_land[cid]) else 1.0)
            pp = 100.0 * float(expit(L0) - expit(L0 - ct))
            low = bool(exposure[cid] < LOW_DATA_EXPOSURE)
            lines.append(CardLine(card_id=cid, name=self._names[cid], count=n, value_pp=pp, low_data=low,
                                  note="splash" if cid in needs else "",
                                  castable=None if bool(self._ca.is_land[cid]) or cid not in cast_p else cast_p[cid]))
        if cast_err:
            warnings.append(cast_err)
        n_cards = int(sum(deck.values()))
        if n_cards != 40:
            warnings.append(f"deck has {n_cards} cards (the model is fitted on 40-card decks)")
        low_names = [l.name for l in lines if l.low_data]
        if low_names:
            warnings.append(f"{len(low_names)} card(s) with little data (estimates lean on priors): " + ", ".join(low_names[:8])
                            + ("..." if len(low_names) > 8 else ""))
        i = {n: k for k, n in enumerate(self.feature_names)}
        mana_stats = {k: float(phi[i[k]]) for k in structure.MANA_NAMES}
        breakdown = {"intercept": float(ref[0]), "cards": float(cards[0]), "structure": float(struct[0]), "mana": float(mana[0])}
        if self.cast_applied:           # the dead-cards feature is the last column of the mana block
            unc = float(self._draws[fmt]["theta"][0][-1] * self.standardise(phi)[-1])
            breakdown["mana"] = float(mana[0]) - unc
            breakdown["uncastable"] = unc
        if self.has_mull(fmt):
            breakdown["mulligans"] = float(mull[0])
        if has_x:
            if "pair" in self._draws[fmt]:
                breakdown["synergy"] = float(pt[0])
            if "arch" in self._draws[fmt]:
                breakdown["archetype"] = float(at[0])
        support, sup_err = self._support(deck, fmt)
        if sup_err:
            warnings.append(sup_err)
        warnings += self.load_warnings
        return DeckEval(cards=lines, colors=colors, splash=splash, game_wr=q0, game_wr_lo=lo, game_wr_hi=hi,
                        match_wr=match_win_prob(q0) if fmt == "bo3" else None, score=score, score_lo=s_lo, score_hi=s_hi,
                        exp_wins=exp_w, p_max_wins=p_max, breakdown=breakdown, mana=mana_stats, warnings=warnings,
                        support=support)

    def _support(self, deck: dict[int, int], fmt: str) -> tuple[list[str], str]:
        """(outside-the-data flags, error message): a failure is reported to the user, never silently an empty list."""
        try:
            return self.support_flags(deck, fmt), ""
        except Exception as e:  # noqa: BLE001
            log.exception("support flags could not be computed")
            return [], f"the 'outside the data' check of this deck could not be computed ({type(e).__name__}: {e}); do not read the absence of flags as a clean bill"


# ---------------------------------------------------------------------------------------------
# optional layers on disk
# ---------------------------------------------------------------------------------------------

def load_extras(d: Path, meta: dict) -> dict:
    """pairs.npz / archcols.npz / archetypes.* / opp.npz of a model directory (each only if present)."""
    out: dict = {}
    errors: list[str] = []
    formats = meta.get("formats", [])
    try:
        if (d / "pairs.npz").exists():
            z = np.load(d / "pairs.npz", allow_pickle=False)
            out["pairs"] = dict(ab=z["ab"].astype(np.int64), games=z["games"], lift=z["lift"],
                                se={f: z[f"{f}__se"] for f in formats if f"{f}__se" in z.files})
        if (d / "archcols.npz").exists():
            z = np.load(d / "archcols.npz", allow_pickle=False)
            out["archcols"] = dict(ca=z["ca"].astype(np.int64), se={f: z[f"{f}__se"] for f in formats if f"{f}__se" in z.files})
        from sealed_lab.model.archetypes import Archetypes

        if (d / "playdraw.npz").exists():
            from sealed_lab.model.playdraw import PlayDrawModel

            z = np.load(d / "playdraw.npz", allow_pickle=False)
            out["playdraw"] = PlayDrawModel(gamma=z["gamma"].astype(float), sigma=float(z["sigma"]))

        arch = Archetypes.load(d)
        if arch is not None:
            out["archetypes"] = arch
        if (d / "opp.npz").exists():
            from sealed_lab.model.opponent import OppModel

            z = np.load(d / "opp.npz", allow_pickle=False)
            out["opp"] = OppModel(mode=str(z["mode"]), delta=z["delta"].astype(float), mass=z["mass"].astype(float),
                                  se_c=z["se_c"].astype(float), opps=[str(x) for x in z["opps"]], sigma=float(z["sigma"]))
    except Exception as e:  # noqa: BLE001 - a damaged optional file must not take the model down (SetModel.load decides)
        log.warning("optional model layers unreadable; ignored", exc_info=True)
        errors.append(f"{type(e).__name__}: {e}")
    if errors:
        out["_errors"] = "; ".join(errors)
    return out


def _invalidate_meta(d: Path) -> None:
    """Remove meta.json before any model file is replaced: a model without meta is "no model", so a crash or kill part way through a
    save never leaves new arrays next to the old (valid-looking) meta.  `save_artifact` writes the new meta last."""
    try:
        (d / "meta.json").unlink()
    except FileNotFoundError:
        pass


def save_extras(d: Path, pairs: dict | None, archcols: dict | None, archetypes, opp, formats: list[str], playdraw=None) -> None:
    """Write (or remove, for None) the optional layers. Call before `save_artifact` (whose meta.json is written last).
    The old meta.json is removed first (see `_invalidate_meta`)."""
    d.mkdir(parents=True, exist_ok=True)
    _invalidate_meta(d)

    def put(name: str, **arrs) -> None:
        tmp = d / (name + ".tmp.npz")
        np.savez_compressed(tmp, **arrs)
        os.replace(tmp, d / (name + ".npz"))

    def drop(name: str) -> None:
        try:
            (d / name).unlink()
        except FileNotFoundError:
            pass
    if pairs is not None:
        put("pairs", ab=pairs["ab"], games=np.asarray(pairs["games"], dtype=np.float64), lift=np.asarray(pairs["lift"], dtype=np.float32),
            **{f"{f}__se": np.asarray(v, dtype=np.float32) for f, v in pairs["se"].items()})
    else:
        drop("pairs.npz")
    if archcols is not None:
        put("archcols", ca=archcols["ca"], **{f"{f}__se": np.asarray(v, dtype=np.float32) for f, v in archcols["se"].items()})
    else:
        drop("archcols.npz")
    if archetypes is not None:
        archetypes.save(d)
    else:
        drop("archetypes.npz")
        drop("archetypes.json")
    if opp is not None:
        put("opp", mode=np.array(opp.mode), delta=opp.delta.astype(np.float32), mass=opp.mass.astype(np.float32),
            se_c=opp.se_c.astype(np.float32), opps=np.array(opp.opps), sigma=np.array(opp.sigma))
    else:
        drop("opp.npz")
    if playdraw is not None:
        put("playdraw", gamma=playdraw.gamma.astype(np.float32), sigma=np.array(playdraw.sigma))
    else:
        drop("playdraw.npz")


# ---------------------------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------------------------

def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def save_artifact(set_code: str, meta: dict, cards: pl.DataFrame, draws: dict[str, dict[str, np.ndarray]],
                  reference: dict[str, np.ndarray], report_html: str, report_json: dict) -> Path:
    """Write the artifact atomically (meta.json last, so a half-written model never looks complete)."""
    d = config.model_dir(set_code)
    d.mkdir(parents=True, exist_ok=True)
    _invalidate_meta(d)
    cards.write_parquet(d / "cards.parquet.tmp", compression="zstd")
    os.replace(d / "cards.parquet.tmp", d / "cards.parquet")
    flat = {}
    for f, v in draws.items():
        flat[f"{f}__beta"] = v["beta"].astype(np.float32)
        flat[f"{f}__theta"] = v["theta"].astype(np.float32)
        flat[f"{f}__ref"] = v["ref"].astype(np.float64)
        for k in ("pair", "arch", "mull"):
            if k in v:
                flat[f"{f}__{k}"] = np.asarray(v[k], dtype=np.float32)
    tmp = d / "draws.tmp.npz"
    np.savez_compressed(tmp, **flat)
    os.replace(tmp, d / "draws.npz")
    tmp = d / "reference.tmp.npz"
    np.savez_compressed(tmp, **{f"{f}__q": r.astype(np.float32) for f, r in reference.items()})
    os.replace(tmp, d / "reference.npz")
    _atomic_write(d / "report.html", report_html.encode("utf-8"))
    _atomic_write(d / "report.json", json.dumps(report_json, indent=1, default=_json_default, ensure_ascii=False).encode("utf-8"))
    meta = dict(meta)
    meta.setdefault("artifact_version", ARTIFACT_VERSION)
    meta.setdefault("code_version", CODE_VERSION)
    _atomic_write(d / "meta.json", json.dumps(meta, indent=1, default=_json_default, ensure_ascii=False).encode("utf-8"))
    return d
