"""Reference distribution of q(D) over the real builds of a target format (for the percentile score)."""
from __future__ import annotations

import numpy as np
import polars as pl
from scipy.special import expit

from sealed_lab.model.design import DeckSpace


def build_deck_keys(builds: pl.DataFrame, fmt: str) -> np.ndarray:
    """deck_key of the real builds of `fmt`, one per (draft_id, build_index), weight 1, in a stable (first-seen) order."""
    b = builds.filter(pl.col("fmt") == fmt).unique(subset=["draft_id", "build_index"], maintain_order=True)
    return b["deck_key"].to_numpy().astype(np.int64)


def build_rows(builds: pl.DataFrame, fmt: str, space: DeckSpace) -> np.ndarray:
    """Deck-space rows of the real builds of `fmt`, one per (draft_id, build_index), weight 1."""
    return space.rows(build_deck_keys(builds, fmt))


def mull_inputs(decks: pl.DataFrame, keys: np.ndarray, probs) -> np.ndarray:
    """(n, 2) [P(exactly one mulligan), P(two or more)] of the decks `keys` (land count and size from the `decks` table);
    `probs(n_lands, n_cards) -> (p1, p2p)` is `SetModel.mull_probs`-like (one value per distinct (lands, cards))."""
    d = decks.select("deck_key", "n_lands", "n_cards").sort("deck_key")
    dk = d["deck_key"].to_numpy().astype(np.int64)
    pos = np.searchsorted(dk, keys)
    if len(keys) and (np.any(pos >= len(dk)) or np.any(dk[np.minimum(pos, len(dk) - 1)] != keys)):
        raise KeyError("deck_key not found in the decks table")
    nl = d["n_lands"].fill_null(0).to_numpy()[pos].astype(np.int64)
    nc = d["n_cards"].fill_null(0).to_numpy()[pos].astype(np.int64)
    uniq, inv = np.unique(np.column_stack([nl, nc]), axis=0, return_inverse=True)
    tab = np.array([probs(int(a), int(b)) for a, b in uniq], dtype=float).reshape(len(uniq), 2)
    return tab[np.asarray(inv).reshape(-1)]


def reference_q(space: DeckSpace, rows: np.ndarray, deck_coefs: np.ndarray, ref: np.ndarray,
                extra: "tuple | None" = None, regime: str | None = None, mull: "tuple | None" = None) -> np.ndarray:
    """q of every reference build under every draw, each draw sorted. deck_coefs: (B, p) [cards | theta]; ref: (B,).

    `extra` = (sparse columns of the deck space for ALL rows, coefficient draws (B, m)) adds trailing terms (card pairs,
    card x archetype) that the plain `space.D` does not carry.  `mull` = (probabilities (n_builds, 2) of exactly one / two or more
    mulligans of each build, coefficient draws (B, 2)) adds the expected mulligan cost of every build (a deck's own mulligans)."""
    D = (space.D_regime(regime) if regime else space.D)[rows]       # the builds' features under their queue's regime
    L = np.asarray(D @ deck_coefs.T) + ref[None, :]          # (n_builds, B)
    if extra is not None and extra[1].shape[1]:
        L = L + np.asarray(extra[0][rows] @ extra[1].T)
    if mull is not None:
        L = L + np.asarray(mull[0], dtype=float) @ np.asarray(mull[1], dtype=float).T
    q = expit(L).T.astype(np.float32)
    q.sort(axis=1)
    return q
