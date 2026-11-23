"""Design matrices: covariate encoder, deck space (card copies + standardised structure features), stage problems."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import polars as pl
import scipy.sparse as sp

from sealed_lab import config
from sealed_lab.model.glm import Problem

log = logging.getLogger(__name__)

SKILL_STEP = 0.02
SKILL_CENTER, SKILL_SCALE = 0.55, 0.05
MIN_SKILL_BIN = 30
PAIRS = ["WU", "WB", "WR", "WG", "UB", "UR", "UG", "BR", "BG", "RG"]
OPP_LEVELS = PAIRS + ["other"]
OPP_FLAG_MARK = "_oppflags"          # a constant column on the games frame switches the opponent colour flags on (see CovEncoder)
OPP_FLAG_NAMES = [f"oppflag:{c}" for c in "WUBRG"] + ["opp_ncolors"]


def with_opp_flags(g: pl.DataFrame) -> pl.DataFrame:
    """`g` marked so that every CovEncoder fitted on / applied to it carries the opponent colour flags."""
    return g if OPP_FLAG_MARK in g.columns else g.with_columns(pl.lit(True).alias(OPP_FLAG_MARK))


def opp_flag_matrix(opp_colors) -> np.ndarray:
    """(n, 6): multi-hot W U B R G of the opponent's deck colours and their number (null / empty / unknown letters -> 0)."""
    oc = ["" if x is None else str(x) for x in opp_colors]
    F = np.zeros((len(oc), 6))
    for j, c in enumerate("WUBRG"):
        F[:, j] = [1.0 if c in s else 0.0 for s in oc]
    F[:, 5] = F[:, :5].sum(axis=1)
    return F
DAY = np.timedelta64(1, "D")
TIME_UNIT_DAYS = 30.0


def norm_pair(s: str | None) -> str:
    if not s:
        return "other"
    t = "".join(c for c in "WUBRG" if c in s)
    return t if t in PAIRS else "other"


# ---------------------------------------------------------------------------------------------
# covariates
# ---------------------------------------------------------------------------------------------

@dataclass
class CovEncoder:
    """Encodes the per-game nuisance covariates; fitted on a stage's training games, applied to any games."""
    events: list[str]
    ref_event: str
    skill_bins: list[int]
    use_g2: bool
    t0: np.datetime64 | None
    umax: float
    use_time: bool
    names: list[str] = field(default_factory=list)
    sigma: np.ndarray = field(default_factory=lambda: np.zeros(0))
    ref_means: np.ndarray | None = None
    oppflags: bool = False

    @classmethod
    def fit(cls, g: pl.DataFrame) -> "CovEncoder":
        ev = g["event"].value_counts().sort("count", descending=True)
        events = ev["event"].to_list()
        gt = g["game_time"].drop_nulls()
        t0 = gt.min() if gt.len() else None
        use_time = False
        umax = 0.0
        if t0 is not None:
            span = (gt.max() - t0).total_seconds() / 86400.0
            umax = span / TIME_UNIT_DAYS
            use_time = span >= 2.0
        sk = g["skill"].drop_nulls().to_numpy().astype(float)
        bins = np.rint(sk / SKILL_STEP).astype(int) if len(sk) else np.zeros(0, int)
        vals, cnts = np.unique(bins, return_counts=True)
        sbins = [int(v) for v, c in zip(vals, cnts) if c >= MIN_SKILL_BIN][:60]
        gn = g["game_number"].fill_null(1).to_numpy()
        enc = cls(events=events, ref_event=events[0] if events else "", skill_bins=sbins, use_g2=int((gn >= 2).sum()) >= 50,
                  t0=np.datetime64(t0, "us") if t0 is not None else None, umax=float(umax), use_time=use_time,
                  oppflags=OPP_FLAG_MARK in g.columns)
        enc._build_names()
        return enc

    def _build_names(self) -> None:
        n, s = ["intercept", "on_play", "mull1", "mull2p", "omull1", "omull2p", "skill_lin", "skill_na"], [10.0] * 8
        for b in self.skill_bins:
            n.append(f"skill_bin:{b * SKILL_STEP:.2f}")
            s.append(0.3)
        for e in self.events:
            if e != self.ref_event:
                n.append(f"event:{e}")
                s.append(10.0)
        if self.use_g2:
            n.append("game2plus")
            s.append(10.0)
        if self.use_time:
            n += ["t", "t_h1", "t_h2"]
            s += [2.0, 1.0, 1.0]
        for o in OPP_LEVELS:
            n.append(f"opp:{o}")
            s.append(0.5)
        if self.oppflags:
            n += OPP_FLAG_NAMES
            s += [0.5] * 5 + [1.0]
        self.names = n
        self.sigma = np.array(s)

    @property
    def k(self) -> int:
        return len(self.names)

    def _u(self, g: pl.DataFrame, clamp: bool) -> np.ndarray:
        gt = g["game_time"].to_numpy().astype("datetime64[us]")
        t0 = self.t0 if self.t0 is not None else np.datetime64("2000-01-01", "us")
        u = (gt - t0) / DAY / TIME_UNIT_DAYS
        u = np.where(np.isnat(gt), 0.0, u).astype(float)
        u = np.maximum(u, 0.0)
        if clamp:
            u = np.minimum(u, self.umax)
        return u

    def transform(self, g: pl.DataFrame, clamp_time: bool = False) -> sp.csr_matrix:
        n = g.height
        cols: list[sp.spmatrix] = []
        op = g["on_play"].fill_null(False).cast(pl.Float64).to_numpy()
        m = g["num_mulligans"].fill_null(0).to_numpy()
        om = g["opp_num_mulligans"].fill_null(0).to_numpy()
        sk = g["skill"].to_numpy().astype(float)
        na = np.isnan(sk)
        lin = np.where(na, 0.0, np.clip((np.nan_to_num(sk) - SKILL_CENTER) / SKILL_SCALE, -4, 4))
        dense = [np.ones(n), op, (m == 1).astype(float), (m >= 2).astype(float), (om == 1).astype(float),
                 (om >= 2).astype(float), lin, na.astype(float)]
        blocks = [sp.csr_matrix(np.column_stack(dense))]
        if self.skill_bins:
            sb = np.where(na, 10 ** 6, np.rint(np.nan_to_num(sk) / SKILL_STEP)).astype(np.int64)
            arr = np.array(self.skill_bins, dtype=np.int64)       # sorted ascending (np.unique)
            r_idx = np.nonzero(np.isin(sb, arr))[0]
            c_idx = np.searchsorted(arr, sb[r_idx])
            blocks.append(sp.csr_matrix((np.ones(len(r_idx)), (r_idx, c_idx)), shape=(n, len(arr))))
        evs = [e for e in self.events if e != self.ref_event]
        if evs:
            ev = g["event"].to_numpy()
            blocks.append(sp.csr_matrix(np.column_stack([(ev == e).astype(float) for e in evs])))
        if self.use_g2:
            gn = g["game_number"].fill_null(1).to_numpy()
            blocks.append(sp.csr_matrix((gn >= 2).astype(float)[:, None]))
        if self.use_time:
            u = self._u(g, clamp_time)
            k1, k2 = self.umax / 3.0, 2.0 * self.umax / 3.0
            blocks.append(sp.csr_matrix(np.column_stack([u, np.maximum(u - k1, 0), np.maximum(u - k2, 0)])))
        oc = g["opp_colors"].fill_null("").to_numpy()
        uniq = {v: norm_pair(v) for v in np.unique(oc)}
        cat = np.array([OPP_LEVELS.index(uniq[v]) for v in oc], dtype=np.int64) if n else np.zeros(0, np.int64)
        blocks.append(sp.csr_matrix((np.ones(n), (np.arange(n), cat)), shape=(n, len(OPP_LEVELS))))
        if self.oppflags:
            blocks.append(sp.csr_matrix(opp_flag_matrix(g["opp_colors"].to_list())))
        X = sp.hstack(blocks, format="csr")
        assert X.shape[1] == self.k, (X.shape, self.k)
        return X

    def set_reference(self, C: sp.csr_matrix) -> np.ndarray:
        """Pilot-neutral reference covariate row from training covariates (see SPEC)."""
        r = np.asarray(C.mean(axis=0)).ravel()
        ix = {nm: i for i, nm in enumerate(self.names)}
        r[ix["intercept"]] = 1.0
        r[ix["on_play"]] = 0.5
        for nm in ("mull1", "mull2p", "omull1", "omull2p"):
            r[ix[nm]] = 0.0
        if self.use_time:
            u = self.umax
            r[ix["t"]], r[ix["t_h1"]], r[ix["t_h2"]] = u, max(u - self.umax / 3, 0.0), max(u - 2 * self.umax / 3, 0.0)
        self.ref_means = r
        return r


# ---------------------------------------------------------------------------------------------
# decks
# ---------------------------------------------------------------------------------------------

@dataclass
class DeckSpace:
    """Rows = all corpus decks. Columns = [non-basic card copies | standardised structure features]."""
    keys: np.ndarray             # sorted deck_key
    phi: np.ndarray              # raw phi, aligned to keys
    mu: np.ndarray
    sd: np.ndarray
    nb_ids: np.ndarray           # card_ids of the non-basic cards (card columns)
    col_of_card: np.ndarray      # card_id -> column or -1
    D: sp.csr_matrix
    n_cards_total: int
    w: np.ndarray | None = None  # normalised per-deck weights of the standardisation distribution
    # card-attribute columns (appended after the structure columns; empty in a plain space)
    attr_names: list[str] = field(default_factory=list)
    attr_mu: np.ndarray = field(default_factory=lambda: np.zeros(0))
    attr_sd: np.ndarray = field(default_factory=lambda: np.ones(0))
    attr_load: np.ndarray | None = None  # (n_card, n_attr): attribute / sd, so beta_attr = attr_load @ gamma
    # extra deck-level blocks after the attribute columns: card pairs (min of the copies) and card x archetype columns
    pair_ab: np.ndarray | None = None    # (n_pair, 2) card ids a < b
    arch_ca: np.ndarray | None = None    # (n_arch_col, 2) (card id, archetype id)
    deck_arch: np.ndarray | None = None  # archetype id of every row of D (aligned to `keys`), -1 = none
    # regime-specific structure features (mana simulator v2): raw phi of every deck under each hand/draw regime, aligned to
    # `keys`. `D` then carries the first regime's table; `D_regime(r)` swaps the structure block, so a game is
    # modelled with the mana statistics of the queue it was played in. All regimes share `mu` / `sd`, which keeps the
    # coefficients (and the chain's prior means) on one scale.
    regime_phi: dict[str, np.ndarray] | None = None
    # castability (model "MK"): per regime, the card block holding copies x castability (n_c * p_c(D)) instead of the copies; the
    # structure block then has the expected-dead-cards feature as its last column (part of `regime_phi`). `D` itself keeps the raw copies.
    cast_cards: dict[str, sp.csr_matrix] | None = None
    _dcache: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    def D_regime(self, regime: str) -> sp.csr_matrix:
        """Deck matrix whose structure columns hold the features computed under `regime` (D itself without regime tables)."""
        if not self.regime_phi:
            return self.D
        hit = self._dcache.get(regime)
        if hit is None:
            phi = self.regime_phi.get(regime)
            if phi is None:
                phi = self.regime_phi[next(iter(self.regime_phi))]
            Z = (phi - self.mu) / self.sd
            cb = self.D[:, :self.n_card]
            if self.cast_cards:
                cb = self.cast_cards.get(regime)
                if cb is None:
                    cb = self.cast_cards[next(iter(self.cast_cards))]
            tail = self.D[:, self.n_card + self.n_feat:]
            if self.cast_cards and self.n_attr and self.attr_load is not None:
                # card-attribute sums must be taken over the SAME (castability-weighted) copies the card block holds: the
                # attribute effect is folded into the per-card betas, which are applied to the weighted copies when serving
                # (standardisation constants stay those of the raw copies: they only shift the folded reference constant)
                Zw = (cb @ (self.attr_load * self.attr_sd[None, :]) - self.attr_mu[None, :]) / self.attr_sd[None, :]
                tail = sp.hstack([sp.csr_matrix(np.asarray(Zw)), tail[:, self.n_attr:]], format="csr")
            hit = sp.hstack([cb, sp.csr_matrix(Z), tail], format="csr")
            self._dcache[regime] = hit
        return hit

    @property
    def n_card(self) -> int:
        return len(self.nb_ids)

    @property
    def n_pair(self) -> int:
        return 0 if self.pair_ab is None else len(self.pair_ab)

    @property
    def n_archc(self) -> int:
        return 0 if self.arch_ca is None else len(self.arch_ca)

    @property
    def n_extra(self) -> int:
        return self.n_pair + self.n_archc

    @property
    def extra_start(self) -> int:
        return self.n_card + self.n_feat + self.n_attr

    @property
    def n_feat(self) -> int:
        return self.phi.shape[1]

    @property
    def n_attr(self) -> int:
        return len(self.attr_names)

    @property
    def deck_dim(self) -> int:
        return self.n_card + self.n_feat + self.n_attr + self.n_extra

    def rows(self, deck_keys: np.ndarray) -> np.ndarray:
        r = np.searchsorted(self.keys, deck_keys)
        if len(r) and (np.any(r >= len(self.keys)) or np.any(self.keys[np.minimum(r, len(self.keys) - 1)] != deck_keys)):
            raise KeyError("deck_key not found in deck space")
        return r

    def standardise(self, phi: np.ndarray) -> np.ndarray:
        return (phi - self.mu) / self.sd


def build_deck_space(deck_keys: np.ndarray, phi: np.ndarray | None, deck_cards: pl.DataFrame, cards: pl.DataFrame,
                     weights: np.ndarray | None = None, regime_phi: dict[str, np.ndarray] | None = None,
                     regime_weights: dict[str, np.ndarray] | None = None,
                     cast_cards: dict[str, sp.csr_matrix] | None = None) -> DeckSpace:
    """`deck_keys`/`phi` aligned (any order). `weights` (per deck) define the standardisation distribution.

    With `regime_phi` ({regime: phi aligned to deck_keys}) the structure block depends on the regime of the game (see
    `DeckSpace.regime_phi`); `regime_weights` ({regime: games per deck}) then define the standardisation distribution (every
    deck counts in the regime(s) of the games it was played in); `phi` is ignored."""
    order = np.argsort(deck_keys)
    keys = deck_keys[order]
    rw = None
    if regime_phi:
        regime_phi = {r: np.asarray(v)[order] for r, v in regime_phi.items()}
        rw = regime_weights or {r: np.ones(len(keys)) for r in regime_phi}
        rw = {r: np.asarray(rw[r], dtype=float)[order] for r in regime_phi}
        main = max(regime_phi, key=lambda r: float(rw[r].sum()))
        regime_phi = {main: regime_phi[main], **{r: v for r, v in regime_phi.items() if r != main}}
        phi = regime_phi[main]
    else:
        phi = phi[order]
    n_cards_total = cards.height
    nb = cards.filter(~pl.col("is_basic"))["card_id"].to_numpy().astype(np.int64)
    col_of = np.full(n_cards_total, -1, dtype=np.int64)
    col_of[nb] = np.arange(len(nb))
    dc = deck_cards.select("deck_key", "card_id", "n").to_numpy() if deck_cards.height else np.zeros((0, 3), np.int64)
    dk, ci, cn = dc[:, 0].astype(np.int64), dc[:, 1].astype(np.int64), dc[:, 2].astype(float)
    cols = col_of[ci]
    keep = cols >= 0
    rows = np.searchsorted(keys, dk[keep])
    Dc = sp.csr_matrix((cn[keep], (rows, cols[keep])), shape=(len(keys), len(nb)))
    if regime_phi:
        tot = sum(float(rw[r].sum()) for r in regime_phi)
        wr = {r: rw[r] / tot for r in regime_phi}
        w = sum(wr.values())
        mu = sum((regime_phi[r] * wr[r][:, None]).sum(0) for r in regime_phi)
        var = sum((((regime_phi[r] - mu) ** 2) * wr[r][:, None]).sum(0) for r in regime_phi)
    else:
        w = np.ones(len(keys)) if weights is None else weights[order]
        w = w / w.sum()
        mu = (phi * w[:, None]).sum(0)
        var = (((phi - mu) ** 2) * w[:, None]).sum(0)
    sd = np.sqrt(var)
    sd = np.where(sd < 1e-9, 1.0, sd)
    Z = (phi - mu) / sd
    D = sp.hstack([Dc, sp.csr_matrix(Z)], format="csr")
    if cast_cards:
        cast_cards = {r: m[order] for r, m in cast_cards.items()}
    return DeckSpace(keys=keys, phi=phi, mu=mu, sd=sd, nb_ids=nb, col_of_card=col_of, D=D, n_cards_total=n_cards_total, w=w,
                     regime_phi=regime_phi or None, cast_cards=cast_cards or None)


def with_attributes(space: DeckSpace, A: np.ndarray, names: list[str], columns: np.ndarray | None = None) -> DeckSpace:
    """Copy of `space` with standardised deck-level attribute sums appended: z_a = (sum_c copies_c A_ca - mu_a) / sd_a.

    `A` is (n_cards_total, n_attr) indexed by card_id (basics are ignored: they have no card column). The gamma
    coefficient of column a is the logit effect of one SD of the deck's attribute sum; spread over the cards,
    beta_c = u_c + sum_a gamma_a A_ca / sd_a (the constant -sum_a gamma_a mu_a / sd_a belongs to the intercept)."""
    if columns is not None:
        A = A[:, columns]
        names = [names[i] for i in columns]
    if A.shape[1] == 0:
        return space
    Anb = A[space.nb_ids]
    Dc = space.D[:, :space.n_card]
    S = np.asarray(Dc @ Anb)
    w = np.full(S.shape[0], 1.0 / S.shape[0]) if space.w is None else space.w
    mu = (S * w[:, None]).sum(0)
    sd = np.sqrt((((S - mu) ** 2) * w[:, None]).sum(0))
    sd = np.where(sd < 1e-9, 1.0, sd)
    Z = (S - mu) / sd
    D = sp.hstack([space.D[:, :space.n_card + space.n_feat], sp.csr_matrix(Z)], format="csr")
    return DeckSpace(keys=space.keys, phi=space.phi, mu=space.mu, sd=space.sd, nb_ids=space.nb_ids, col_of_card=space.col_of_card,
                     D=D, n_cards_total=space.n_cards_total, w=space.w, attr_names=list(names), attr_mu=mu, attr_sd=sd,
                     attr_load=Anb / sd[None, :], regime_phi=space.regime_phi, cast_cards=space.cast_cards)


def card_copies(space: DeckSpace) -> sp.csr_matrix:
    """(n_decks, n_card) copies of every non-basic card in every deck-space row."""
    return space.D[:, :space.n_card].tocsr()


def cooccurrence(space: DeckSpace, weights: np.ndarray, spells_only: np.ndarray | None = None) -> np.ndarray:
    """Dense (n_card, n_card) weighted co-occurrence: sum over decks of weight * [a in deck] * [b in deck].

    `weights` aligned to the space's rows (e.g. games played with each deck); `spells_only` (bool per card column) zeroes
    the others. The diagonal is the weighted inclusion count of the card."""
    B = (card_copies(space) > 0).astype(float)
    if spells_only is not None:
        B = B @ sp.diags(spells_only.astype(float))
    W = sp.diags(np.asarray(weights, dtype=float))
    return np.asarray((B.T @ W @ B).todense())


def pair_columns(V: sp.csr_matrix, cols_ab: np.ndarray) -> sp.csr_matrix:
    """(n_decks, n_pair) sparse matrix: min(copies of a, copies of b) per deck, for pairs of card COLUMNS."""
    Vc = V.tocsc()
    ptr, ind, dat = Vc.indptr, Vc.indices, Vc.data
    rows, cols, vals = [], [], []
    for j, (a, b) in enumerate(cols_ab):
        ra, rb = ind[ptr[a]:ptr[a + 1]], ind[ptr[b]:ptr[b + 1]]
        _, ia, ib = np.intersect1d(ra, rb, assume_unique=True, return_indices=True)
        if len(ia):
            rows.append(ra[ia])
            cols.append(np.full(len(ia), j, dtype=np.int64))
            vals.append(np.minimum(dat[ptr[a]:ptr[a + 1]][ia], dat[ptr[b]:ptr[b + 1]][ib]))
    if not rows:
        return sp.csr_matrix((V.shape[0], len(cols_ab)))
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(V.shape[0], len(cols_ab)))


def with_pairs(space: DeckSpace, pair_ab: np.ndarray) -> DeckSpace:
    """Copy of `space` with one column per card pair (card ids a < b): min(copies a, copies b), after any earlier blocks.

    The coefficient of a column is the logit effect of having the pair in the deck (per unit of min copies)."""
    from dataclasses import replace

    if len(pair_ab) == 0:
        return space
    if space.arch_ca is not None:
        raise ValueError("add the pair block before the archetype block")
    cols = space.col_of_card[np.asarray(pair_ab)]
    if (cols < 0).any():
        raise ValueError("pairs must be made of non-basic cards")
    M = pair_columns(card_copies(space), cols)
    return replace(space, D=sp.hstack([space.D, M], format="csr"), pair_ab=np.asarray(pair_ab, dtype=np.int64))


def with_archetype_columns(space: DeckSpace, deck_arch: np.ndarray, ca: np.ndarray) -> DeckSpace:
    """Copy of `space` with card x archetype columns: copies of card c in decks assigned to archetype a (ca = (c, a) rows)."""
    from dataclasses import replace

    if len(ca) == 0:
        return space
    V = card_copies(space).tocsc()
    cols = space.col_of_card[ca[:, 0]]
    rows, cc, vals = [], [], []
    for j, (c, a) in enumerate(zip(cols, ca[:, 1])):
        r = V.indices[V.indptr[c]:V.indptr[c + 1]]
        v = V.data[V.indptr[c]:V.indptr[c + 1]]
        m = deck_arch[r] == a
        rows.append(r[m])
        cc.append(np.full(int(m.sum()), j, dtype=np.int64))
        vals.append(v[m])
    M = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cc))), shape=(V.shape[0], len(ca)))
    return replace(space, D=sp.hstack([space.D, M], format="csr"), arch_ca=np.asarray(ca, dtype=np.int64),
                   deck_arch=np.asarray(deck_arch, dtype=np.int32))


# ---------------------------------------------------------------------------------------------
# stage problems
# ---------------------------------------------------------------------------------------------

@dataclass
class StageData:
    name: str
    games: pl.DataFrame                  # rows used (needed columns)
    enc: CovEncoder
    prob: Problem
    group: np.ndarray                    # int32 draft_id code per game
    group_ids: list[str]
    group_start: np.ndarray              # datetime64 per group
    deck_rows: np.ndarray                # rows in DeckSpace of the stage's unique decks
    exposure: np.ndarray                 # per card column: total copies x games
    ref_row: np.ndarray
    n_games: int
    n_groups: int
    w: np.ndarray | None = None          # per-game likelihood weights (recency); None = all 1


def recency_weights(stage: StageData, t_ref, half_life_days: float | None) -> np.ndarray | None:
    """Exponential recency weight per game: 0.5 ** (age / half-life), age = days from the event's first game to `t_ref`.

    All games of an event share its weight (events are the resampling/cluster unit). None when no half-life."""
    if half_life_days is None or not np.isfinite(half_life_days):
        return None
    t = np.datetime64(t_ref, "us")
    start = stage.group_start.astype("datetime64[us]")
    age = np.where(np.isnat(start), 0.0, (t - start) / DAY)
    age = np.maximum(age.astype(float), 0.0)
    return (0.5 ** (age / float(half_life_days)))[stage.group]


def with_weights(stage: StageData, w: np.ndarray | None) -> StageData:
    from dataclasses import replace

    return replace(stage, w=w)


def game_regimes(g: pl.DataFrame) -> np.ndarray:
    """Hand/draw regime (config.REGIMES) of every game, from its event type."""
    ev = g["event"].to_numpy()
    m = {e: config.event_regime(str(e)) for e in np.unique(ev)}
    return np.array([m[e] for e in ev], dtype=object)


def make_problem(g: pl.DataFrame, enc: CovEncoder, space: DeckSpace, clamp_time: bool = False,
                 ) -> tuple[Problem, np.ndarray, np.ndarray]:
    """(Problem, deck rows, game->row index) for games `g` under encoder `enc`. Without regime tables the rows are the unique
    decks; with them they are the unique (deck, regime) pairs (`urows[i]` is the deck row of unique row i, so `urows[inv]`
    is the deck row of every game), each carrying the structure features of its regime."""
    C = enc.transform(g, clamp_time=clamp_time)
    rows = space.rows(g["deck_key"].to_numpy().astype(np.int64))
    if space.regime_phi:
        names = list(space.regime_phi)
        rg = game_regimes(g)
        ridx = np.array([names.index(r) if r in names else 0 for r in rg], dtype=np.int64)
        comb = rows * len(names) + ridx
        ucomb, inv = np.unique(comb, return_inverse=True)
        urows, ur = ucomb // len(names), ucomb % len(names)
        parts, pos = [], []
        for k, nm in enumerate(names):
            m = np.nonzero(ur == k)[0]
            if len(m):
                parts.append(space.D_regime(nm)[urows[m]])
                pos.append(m)
        D = sp.vstack(parts, format="csr")[np.argsort(np.concatenate(pos))]
        y = g["won"].cast(pl.Float64).to_numpy()
        return Problem(y=y, C=C, D=D, idx=inv.astype(np.int32)), urows, inv
    urows, inv = np.unique(rows, return_inverse=True)
    D = space.D[urows]
    y = g["won"].cast(pl.Float64).to_numpy()
    return Problem(y=y, C=C, D=D, idx=inv.astype(np.int32)), urows, inv


def build_stage(name: str, g: pl.DataFrame, space: DeckSpace, enc: CovEncoder | None = None) -> StageData:
    if enc is None:
        enc = CovEncoder.fit(g)
    prob, urows, inv = make_problem(g, enc, space)
    ref = enc.set_reference(prob.C)
    did = g["draft_id"].to_numpy()
    uniq, grp = np.unique(did, return_inverse=True)
    gt = g["game_time"].to_numpy().astype("datetime64[us]")
    order = np.lexsort((gt.astype("int64"), grp))
    first = np.searchsorted(grp[order], np.arange(len(uniq)))
    start = gt[order[first]]
    games_per_deck = np.bincount(inv, minlength=len(urows)).astype(float)
    Dcards = space.D[urows][:, :space.n_card]
    exposure = np.asarray(Dcards.T @ games_per_deck).ravel()
    return StageData(name=name, games=g, enc=enc, prob=prob, group=grp.astype(np.int32), group_ids=list(uniq),
                     group_start=start, deck_rows=urows, exposure=exposure, ref_row=ref, n_games=g.height,
                     n_groups=len(uniq))


def event_starts(g: pl.DataFrame) -> pl.DataFrame:
    """Per draft_id: first game time (fallback: draft_time) and number of games."""
    return (g.group_by("draft_id").agg(pl.col("game_time").min().alias("t_game"), pl.col("draft_time").min().alias("t_draft"),
                                       pl.len().alias("n"))
            .with_columns(pl.coalesce("t_game", "t_draft").alias("start")).sort("start", "draft_id"))
