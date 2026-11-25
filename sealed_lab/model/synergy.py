"""Synergy layer: pairwise card interactions.

For card pairs (a, b) that share enough games, the deck gets one extra column  min(copies_a, copies_b)  (1 when each card is
played once; the minimum, not the product, because a second copy of one card adds nothing to a pair without a second copy of
the other).  Its coefficient is the extra logit effect of having the pair in the same deck, on top of the two single-card
effects; it is shrunk towards 0 (or towards the previous stage's value in the chain) with its own prior SD chosen by grouped
cross-validation.  Model "MS" in the validation: the chain with these columns, scored on the same hold-out as "M".

Descriptive discovery (no model): the color-adjusted co-occurrence lift of a pair, observed decks with both cards divided by
the number expected if each card were drawn independently within decks of the same color group (EDHREC-style).
"""
from __future__ import annotations

import logging

import numpy as np
import polars as pl
import scipy.sparse as sp
from scipy.stats import norm

from sealed_lab.model import design
from sealed_lab.model.design import DeckSpace

log = logging.getLogger(__name__)

MIN_PAIR_GAMES = 300           # games (all formats) in which both cards were in the deck
MAX_PAIRS = 3000               # cap on the number of columns (the exact cluster-robust SE needs a dense Hessian)
PP_PER_LOGIT = 100.0 * 0.55 * 0.45
# a synergy is shown to the user only if the data support its sign and it is not negligible
SYN_P_MIN = 0.90               # P(effect has the stated sign) from the normal approximation (cluster-robust SE)
SYN_MIN_PP = 0.25              # pp of game win rate near a 55% deck


def select_pairs(space: DeckSpace, weights: np.ndarray, cards_is_land: np.ndarray, min_games: int = MIN_PAIR_GAMES,
                 max_pairs: int = MAX_PAIRS) -> tuple[np.ndarray, np.ndarray]:
    """(pair_ab (n, 2) card ids a < b, games with both (n,)) of the spell pairs sharing >= `min_games` games.

    `weights`: games per row of the deck space; `cards_is_land`: bool per card_id. The strongest `max_pairs` by shared games
    are kept (deterministic ties by card id)."""
    spell = ~np.asarray(cards_is_land)[space.nb_ids]
    C = design.cooccurrence(space, weights, spell)
    iu, ju = np.triu_indices(C.shape[0], k=1)
    g = C[iu, ju]
    ok = np.nonzero(g >= min_games)[0]
    if len(ok) == 0:
        return np.zeros((0, 2), dtype=np.int64), np.zeros(0)
    order = ok[np.lexsort((ju[ok], iu[ok], -g[ok]))][:max_pairs]
    ab = np.column_stack([space.nb_ids[iu[order]], space.nb_ids[ju[order]]]).astype(np.int64)
    return ab, g[order]


# ---------------------------------------------------------------------------------------------
# descriptive lift
# ---------------------------------------------------------------------------------------------

def deck_groups(games: pl.DataFrame, keys: np.ndarray) -> np.ndarray:
    """Color group (modal main_colors; mono and 3+ color decks -> 'other') per deck key, aligned to `keys`."""
    g = (games.select("deck_key", pl.col("main_colors").fill_null("")).group_by("deck_key", "main_colors").len()
         # fully ordered (ties between color pairs broken by name) so the modal group does not depend on the order polars
         # happens to return the groups in - it changed from process to process and with it the archetypes and the lift table
         .sort(["deck_key", "len", "main_colors"], descending=[False, True, False]).unique(subset="deck_key", keep="first",
                                                                                         maintain_order=True))
    m = dict(zip(g["deck_key"].to_list(), g["main_colors"].to_list()))
    return np.array([design.norm_pair(m.get(int(k), "")) for k in keys])


def lift_table(space: DeckSpace, groups: np.ndarray, rows_mask: np.ndarray | None = None, min_obs: int = 30
               ) -> dict[str, np.ndarray]:
    """Color-adjusted co-occurrence lift for every pair of non-basic cards (over the unique decks in `rows_mask`).

    Returns dict(obs, exp, lift (obs/exp, NaN below `min_obs`), z (Poisson z of obs vs exp)) as (n_card, n_card) arrays
    indexed by the card COLUMNS of the deck space."""
    V = (design.card_copies(space) > 0).astype(float)
    if rows_mask is not None:
        V = V[np.nonzero(rows_mask)[0]]
        groups = groups[rows_mask]
    levels = sorted(set(groups.tolist()))
    G = sp.csr_matrix((np.ones(len(groups)), (np.arange(len(groups)), [levels.index(x) for x in groups])), shape=(len(groups), len(levels)))
    N = np.asarray(G.sum(0)).ravel()
    Cg = np.asarray((G.T @ V).todense())                    # (n_group, n_card) decks with card per group
    obs = np.asarray((V.T @ V).todense())
    exp = (Cg / np.maximum(N, 1)[:, None]).T @ Cg           # sum_g c_ga c_gb / N_g
    with np.errstate(divide="ignore", invalid="ignore"):
        lift = np.where(obs >= min_obs, obs / np.maximum(exp, 1e-9), np.nan)
        z = (obs - exp) / np.sqrt(np.maximum(exp, 1e-9))
    return dict(obs=obs, exp=exp, lift=lift, z=z)


# ---------------------------------------------------------------------------------------------
# model-based effects of the fitted pair columns
# ---------------------------------------------------------------------------------------------

def p_positive(coef: np.ndarray, se: np.ndarray) -> np.ndarray:
    """P(effect > 0) under the normal approximation N(coef, se^2) of the shrunk estimate."""
    return norm.cdf(np.asarray(coef) / np.maximum(np.asarray(se), 1e-12))


def reliable_mask(coef: np.ndarray, se: np.ndarray, p_min: float = SYN_P_MIN, min_pp: float = SYN_MIN_PP) -> np.ndarray:
    """Pairs whose sign is supported (P >= p_min) and whose size is at least `min_pp` (pp of win rate near 55%)."""
    p = p_positive(coef, se)
    return (np.maximum(p, 1 - p) >= p_min) & (np.abs(coef) * PP_PER_LOGIT >= min_pp)


def pair_summary(pair_ab: np.ndarray, coef: np.ndarray, se: np.ndarray, games: np.ndarray, names: list[str],
                 lift: dict[str, np.ndarray] | None = None, col_of_card: np.ndarray | None = None, n_top: int = 15) -> dict:
    """Report rows: strongest synergies and anti-synergies (by |z|) with CI, exposure and descriptive lift."""
    z = coef / np.maximum(se, 1e-12)
    p = p_positive(coef, se)
    rel = reliable_mask(coef, se)

    def rec(j):
        a, b = int(pair_ab[j, 0]), int(pair_ab[j, 1])
        r = dict(a=names[a], b=names[b], pp=float(PP_PER_LOGIT * coef[j]), lo_pp=float(PP_PER_LOGIT * (coef[j] - 1.96 * se[j])),
                 hi_pp=float(PP_PER_LOGIT * (coef[j] + 1.96 * se[j])), p_pos=float(p[j]), z=float(z[j]), games=float(games[j]),
                 reliable=bool(rel[j]))
        if lift is not None and col_of_card is not None:
            ca, cb = int(col_of_card[a]), int(col_of_card[b])
            r.update(lift=float(lift["lift"][ca, cb]), obs_decks=float(lift["obs"][ca, cb]), exp_decks=float(lift["exp"][ca, cb]))
        return r
    order = np.argsort(-np.abs(z))
    pos = [rec(j) for j in order if coef[j] > 0][:n_top]
    neg = [rec(j) for j in order if coef[j] < 0][:n_top]
    return dict(n_pairs=int(len(coef)), n_reliable=int(rel.sum()), n_reliable_pos=int((rel & (coef > 0)).sum()),
                n_abs_z3=int((np.abs(z) > 3).sum()), expected_abs_z3=float(len(coef) * 0.0027),
                n_reliable_neg=int((rel & (coef < 0)).sum()), top_pos=pos, top_neg=neg,
                sd_pp=float(PP_PER_LOGIT * np.sqrt(np.mean(coef ** 2))) if len(coef) else 0.0,
                max_abs_pp=float(PP_PER_LOGIT * np.abs(coef).max()) if len(coef) else 0.0)


def lift_summary(lift: dict[str, np.ndarray], space: DeckSpace, names: list[str], n_top: int = 12, min_obs: int = 100) -> list[dict]:
    """Descriptive: pairs with the highest color-adjusted co-occurrence lift among pairs seen in >= min_obs decks."""
    n = lift["obs"].shape[0]
    iu, ju = np.triu_indices(n, k=1)
    ok = (lift["obs"][iu, ju] >= min_obs) & np.isfinite(lift["lift"][iu, ju])
    iu, ju = iu[ok], ju[ok]
    order = np.argsort(-lift["lift"][iu, ju])[:n_top]
    out = []
    for o in order:
        i, j = iu[o], ju[o]
        out.append(dict(a=names[int(space.nb_ids[i])], b=names[int(space.nb_ids[j])], lift=float(lift["lift"][i, j]),
                        obs=float(lift["obs"][i, j]), exp=float(lift["exp"][i, j])))
    return out


def lift_model_agreement(pair_ab: np.ndarray, coef: np.ndarray, lift: dict[str, np.ndarray], col_of_card: np.ndarray) -> float | None:
    """Spearman correlation between the fitted pair effects and log lift (are co-played pairs also co-winning?)."""
    from scipy.stats import spearmanr

    if len(coef) < 10:
        return None
    ca, cb = col_of_card[pair_ab[:, 0]], col_of_card[pair_ab[:, 1]]
    ll = np.log(np.maximum(lift["lift"][ca, cb], 1e-3))
    ok = np.isfinite(ll)
    if ok.sum() < 10:
        return None
    return float(spearmanr(coef[ok], ll[ok]).statistic)
